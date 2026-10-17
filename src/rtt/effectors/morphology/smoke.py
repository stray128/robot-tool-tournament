"""The designer's smoke run and physics gate: Nemotron versus the random control on the training parts.

One run, product mode (docs/EFFECTORS.md, track B3):

1. **Nemotron, 2 rounds of 4.** Round 1 asks :func:`~rtt.effectors.morphology.proposer.propose_recipes`
   for 4 designs for all six training parts at once (one effector for every part, buildable on
   at least two thirds of them). Every design is screened on every part (the part's 4
   ``train_v2`` placements x 2 jitter seeds, 8 episodes per part). Round 2 sees round 1's
   measured outcomes (:class:`~rtt.effectors.morphology.proposer.DesignOutcome`, per part) and
   proposes 4 more, screened the same way.
2. **Random control, the same budget.** 8 recipes from the same spaces with the same
   validators and the same multi-part rule (:func:`~rtt.effectors.morphology.random_baseline.random_recipes`),
   screened the same way.
3. **Nominal test.** Every design of both arms runs the certificate's nominal episode (no
   placement error, jitter off) on every part.
4. **Gate.** Each arm's best design is chosen on the screen alone (held rate over all screen
   episodes, then registered rate, fewer servos, lower mass, recipe id), then judged on the
   nominal test: the gate asks the best Nemotron design to be held at nominal on at least 4 of
   the 6 parts. The random control's best is reported beside it; no superiority is claimed
   (one run, one seed, product mode).

``--live`` calls Token Factory under a hard spend cap (a shared :class:`~rtt.llm.ledger.Ledger`
budget); without it the run replays the recorded cache with zero API calls. The manifest
(``manifest.json``), the ledger and the scrubbed response cache are written to ``--out``.

Usage::

    python -m rtt.effectors.morphology.smoke --out DIR --live --env-file PATH/.env --budget 0.05
    python -m rtt.effectors.morphology.smoke --out DIR            # replay DIR/cache, no API calls
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rtt.effectors.morphology.layout import layout_of
from rtt.effectors.morphology.proposer import (
    DESIGNER_MODEL,
    DESIGNER_SAMPLING,
    DesignOutcome,
    ProposalBatch,
    make_designer_client,
    outcome_json,
    propose_recipes,
)
from rtt.effectors.morphology.random_baseline import random_recipes_with_stats
from rtt.effectors.runner import EffectorJob, run_effector_jobs
from rtt.effectors.types import PRODUCT_MODE_LABEL, EffectorEpisodeResult, MorphologyRecipe

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

__all__ = ["ArmResult", "Evaluated", "evaluate", "main", "pick_best", "run_smoke", "screen_conditions"]

ROUNDS = 2
PER_ROUND = 4
GATE_PARTS = 4


@dataclass
class Evaluated:
    """One design's screen and nominal results on every part."""

    recipe: MorphologyRecipe
    screen: dict[str, list[EffectorEpisodeResult]]  # part -> its 8 screen episodes
    nominal: dict[str, EffectorEpisodeResult]  # part -> the nominal certificate episode

    def outcome(self) -> DesignOutcome:
        eps = [e for part_eps in self.screen.values() for e in part_eps]
        modes = collections.Counter(e.failure_mode for e in eps if e.failure_mode)
        n = len(eps)
        per_part = {
            part: (sum(e.held for e in es) / len(es), sum(e.success for e in es) / len(es))
            for part, es in self.screen.items()
        }
        return DesignOutcome(
            recipe=self.recipe,
            held_rate=sum(e.held for e in eps) / n if n else 0.0,
            registered_rate=sum(e.success for e in eps) / n if n else 0.0,
            failure_modes=dict(modes),
            n_episodes=n,
            per_part=per_part,
        )

    @property
    def nominal_held(self) -> list[str]:
        return [p for p, r in self.nominal.items() if r.held]

    def to_dict(self) -> dict[str, Any]:
        out = outcome_json(self.outcome())
        out["per_part"] = {p: list(v) for p, v in self.outcome().per_part.items()}
        out["nominal"] = {
            p: {
                "held": r.held,
                "registered": r.success,
                "failure_mode": r.failure_mode,
                "traj_sha": r.traj_sha,
            }
            for p, r in self.nominal.items()
        }
        out["nominal_held_parts"] = self.nominal_held
        lay = layout_of(self.recipe)
        out["n_actuators"] = lay.n_actuators
        out["mass_kg"] = round(lay.mass(), 4)
        return out


def screen_conditions(part: PartSpec) -> list[tuple[str, Any, int]]:
    """(condition id, placement, seed) of the part's ``train_v2`` screen: 4 placements x 2 seeds."""
    from rtt.parts.suite import train_suite

    out = []
    for cond in train_suite():
        if cond.part.name != part.name:
            continue
        for seed in cond.seeds:
            out.append((cond.condition_id, cond.placement, int(seed)))
    if not out:
        raise ValueError(f"{part.name} has no train_v2 conditions")
    return out


def evaluate(
    recipes: Sequence[MorphologyRecipe], parts: Sequence[PartSpec], task: TaskV2, workers: int
) -> list[Evaluated]:
    """Screen and nominal episodes of every recipe on every part (one parallel batch)."""
    from rtt.effectors.morphology import spec_for
    from rtt.sim import objects as objs
    from rtt.sim.certify import certificate_task

    ctask = certificate_task(task)
    jobs: list[EffectorJob] = []
    keys: list[tuple[int, str, str]] = []
    for i, r in enumerate(recipes):
        spec = spec_for(r)
        for part in parts:
            for cid, placement, seed in screen_conditions(part):
                jobs.append(EffectorJob(spec, part, placement, seed, task=task, condition_id=cid))
                keys.append((i, part.name, "screen"))
            jobs.append(
                EffectorJob(spec, part, objs.Placement(), 0, task=ctask, condition_id="certificate/nominal")
            )
            keys.append((i, part.name, "nominal"))
    results = run_effector_jobs(jobs, workers=workers)
    out = [Evaluated(r, {p.name: [] for p in parts}, {}) for r in recipes]
    for (i, part, what), res in zip(keys, results, strict=True):
        if what == "screen":
            out[i].screen[part].append(res)
        else:
            out[i].nominal[part] = res
    return out


def pick_best(items: Sequence[Evaluated]) -> Evaluated | None:
    """The best design on the screen alone (module docstring)."""
    if not items:
        return None

    def key(e: Evaluated) -> tuple[float, float, int, float, str]:
        o = e.outcome()
        lay = layout_of(e.recipe)
        return (-o.held_rate, -o.registered_rate, lay.n_actuators, lay.mass(), e.recipe.recipe_id)

    return min(items, key=key)


@dataclass
class ArmResult:
    name: str
    evaluated: list[Evaluated]
    batches: list[ProposalBatch]
    extra: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        best = pick_best(self.evaluated)
        nominal_counts = {e.recipe.recipe_id: len(e.nominal_held) for e in self.evaluated}
        return {
            "arm": self.name,
            "designs": len(self.evaluated),
            "best_by_screen": best.recipe.recipe_id if best else None,
            "best_nominal_held_parts": best.nominal_held if best else [],
            "best_nominal_held": len(best.nominal_held) if best else 0,
            "most_parts_held_by_any_design": max(nominal_counts.values(), default=0),
            "parts_held_by_some_design": sorted({p for e in self.evaluated for p in e.nominal_held}),
        }


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_smoke(
    out: Path,
    *,
    live: bool,
    env_file: Path | None = None,
    budget_usd: float = 0.05,
    workers: int = 8,
    seed: int = 0,
    log: Any = print,
    client: Any = None,
) -> dict[str, Any]:
    """Run both arms and write ``manifest.json``, ``ledger.jsonl`` and ``cache/`` under ``out``."""
    from rtt.parts.library import training_parts
    from rtt.sim.task import default_task_v2, task_digest

    out.mkdir(parents=True, exist_ok=True)
    task = default_task_v2()
    parts = list(training_parts())
    ledger_path = out / "ledger.jsonl"
    if live and ledger_path.exists():
        raise FileExistsError(f"{ledger_path} exists: a live run writes a fresh ledger (move the old one)")
    client = client or make_designer_client(
        out / "cache",
        cache_mode="readwrite" if live else "replay",
        ledger_path=ledger_path if live else None,
        budget_usd=budget_usd if live else None,
        env_file=env_file if live else None,
    )
    t0 = time.perf_counter()

    history: list[DesignOutcome] = []
    batches: list[ProposalBatch] = []
    nemotron: list[Evaluated] = []
    for rnd in range(1, ROUNDS + 1):
        batch = propose_recipes(parts, history, PER_ROUND, client=client, task=task, tag=f"smoke/round{rnd}")
        batches.append(batch)
        log(
            f"round {rnd}: {len(batch.recipes)} designs, {len(batch.rejections)} rejections, "
            f"{len(batch.calls)} calls, ${batch.cost_usd:.4f}"
        )
        evaluated = evaluate(batch.recipes, parts, task, workers)
        nemotron.extend(evaluated)
        history.extend(e.outcome() for e in evaluated)

    randoms, stats = random_recipes_with_stats(parts, ROUNDS * PER_ROUND, seed, task=task)
    random_eval = evaluate(randoms, parts, task, workers)

    arms = [
        ArmResult("nemotron", nemotron, batches, {}),
        ArmResult("random", random_eval, [], {"draws": stats.to_dict(), "seed": seed}),
    ]
    ledger = client.ledger.session_entries()
    live_entries = [e for e in ledger if not e.cached]
    nem = arms[0].summary()
    manifest: dict[str, Any] = {
        "label": PRODUCT_MODE_LABEL,
        "kind": "rtt-effectors/designer-smoke",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": _git_rev(),
        "task": task.name,
        "task_digest": task_digest(task),
        "parts": [p.name for p in parts],
        "model": DESIGNER_MODEL,
        "sampling": dict(DESIGNER_SAMPLING),
        "mode": "live" if live else "replay",
        "budget_usd": budget_usd if live else None,
        "llm": {
            "calls": len(ledger),
            "live_calls": len(live_entries),
            "prompt_tokens": sum(e.prompt_tokens for e in live_entries),
            "completion_tokens": sum(e.completion_tokens for e in live_entries),
            "cost_usd": round(sum(e.cost_usd for e in live_entries), 6),
            "errors": [e.error for e in ledger if e.error],
        },
        "protocol": {
            "rounds": ROUNDS,
            "per_round": PER_ROUND,
            "screen": "train_v2: 4 placements x 2 jitter seeds per part",
            "nominal": "certificate nominal test (no placement error, jitter off, seed 0)",
            "best": "on the screen: held rate, registered rate, fewer servos, lower mass, recipe id",
            "gate": f"best Nemotron design held at nominal on at least {GATE_PARTS} of {len(parts)} parts",
        },
        "batches": [b.to_dict() for b in batches],
        "arms": {
            a.name: {"summary": a.summary(), "designs": [e.to_dict() for e in a.evaluated], **a.extra}
            for a in arms
        },
        "gate": {
            "nemotron_best_nominal_held": nem["best_nominal_held"],
            "required": GATE_PARTS,
            "passed": nem["best_nominal_held"] >= GATE_PARTS,
            "random_best_nominal_held": arms[1].summary()["best_nominal_held"],
            "claim": "none: one run, one seed; the random control is reported beside Nemotron, not ranked against it",
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="call Token Factory (spend-capped)")
    parser.add_argument("--env-file", type=Path, default=None, help=".env holding NEBIUS_API_KEY (live only)")
    parser.add_argument("--budget", type=float, default=0.05, help="hard spend cap in USD (live only)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="random control seed")
    args = parser.parse_args(argv)
    if args.live and args.env_file is None:
        parser.error("--live needs --env-file")
    manifest = run_smoke(
        args.out,
        live=args.live,
        env_file=args.env_file,
        budget_usd=args.budget,
        workers=args.workers,
        seed=args.seed,
    )
    gate = manifest["gate"]
    print(json.dumps({"llm": manifest["llm"], "gate": gate, "wall_s": manifest["wall_s"]}, indent=2))
    return 0 if gate["passed"] and math.isfinite(manifest["llm"]["cost_usd"]) else 1


if __name__ == "__main__":
    sys.exit(main())
