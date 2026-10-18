#!/usr/bin/env python
"""Use-case benchmark v1 of the Studio (track S6): freeze, commit, evaluate, analyse, estimate.

**Product mode: not part of the pre-registered experiment.** The protocol is
``docs/USECASE_V1.md``; its parameters are ``configs/usecase_v1.yaml``. Commands, in order:

* ``freeze-arms``: the development search of every hero, on the development conditions only:
  tournament (b) on the full candidate pool and tournament (c) on the library entrants only; arm
  (a) is (b)'s best single off-the-shelf effector. Writes the frozen block of the config (setup ids,
  the setups themselves, the development runs and their trajectory hashes, which heroes are tested).
  Either tournament can be reused from an existing report (``--reuse-b``, ``--reuse-c``), checked
  against the protocol and this code's candidate pool; every frozen arm is then re-run on its first
  development conditions and must reproduce its trajectory hashes. Runs inside the vault guard,
  before any secret exists; ``--refreeze`` regenerates the block until a secret is committed.
* ``commit-secret``: draws the held-out secret (``secrets/``, git-ignored) and writes its SHA-256
  into the frozen block. Refuses before the arms are frozen.
* ``finalize``: checks the frozen block and the secret, then sets ``status: frozen``. The founder
  approves; the commit is tagged ``usecase-v1``.
* ``evaluate``: after the tag. Checks the tag, the config at the tag and the secret, derives the 32
  held-out conditions per hero, runs every arm on them (plus the parity runs on the development
  conditions) through the fan-out, writes the rows, the conditions and the revealed secret, then
  the analysis.
* ``analyse``: the primary test (exact one-sided paired test, Holm) and the descriptive tables.
* ``dry-run``: all of the above on a reduced set (some heroes, a few held-out conditions) with a
  throwaway secret and the local fake cloud. Never reads the real secret.
* ``estimate``: the cost table from measured seconds per run (any fan-out rows).
* ``fetch``: download a fan-out run's bucket prefix (recovery).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

REPO: Final = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Final = REPO / "configs" / "usecase_v1.yaml"
DEFAULT_DEV: Final = REPO / "runs" / "usecase_v1" / "dev"
DEFAULT_RESULTS: Final = REPO / "runs" / "usecase_v1" / "results"
FROZEN_MARKER: Final = (
    "# ---- frozen block: written by scripts/studio_usecase_v1.py, never edited by hand ----"
)
CONFIG_FORMAT: Final = "rtt-usecase-benchmark/1"
RESULTS_FORMAT: Final = "rtt-usecase-v1-results/1"
ANALYSIS_FORMAT: Final = "rtt-usecase-v1-analysis/1"
ARMS: Final = ("a", "b", "c")
PLAN_KEYS: Final = (
    "title",
    "description",
    "objects",
    "steps",
    "assumptions",
    "clarifications",
    "unsupported",
    "refusal",
)
DRY_RUN_NOTE: Final = "DRY RUN on a throwaway seed: infrastructure check, not a result"


def _label() -> str:
    from rtt.effectors.types import PRODUCT_MODE_LABEL

    return PRODUCT_MODE_LABEL


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=False, default=str) + "\n", encoding="utf-8")
    return path


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False)


def git_state() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD").stdout.strip() or None
    dirty = bool(_git("status", "--porcelain").stdout.strip())
    return {"git_sha": sha, "git_dirty": dirty}


# --------------------------------------------------------------------------- config


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("format") != CONFIG_FORMAT:
        raise ValueError(f"{p} is not a {CONFIG_FORMAT} config")
    if cfg.get("label") != _label():
        raise ValueError(f"{p} must carry the product-mode label")
    if FROZEN_MARKER not in p.read_text(encoding="utf-8"):
        raise ValueError(f"{p} has no frozen-block marker")
    return cfg


def write_frozen(path: str | Path, frozen: Mapping[str, Any] | None) -> None:
    """Replace everything after the marker with ``frozen: ...`` (the hand-written part is kept)."""
    p = Path(path)
    head = p.read_text(encoding="utf-8").split(FROZEN_MARKER)[0].rstrip() + "\n\n"
    body = yaml.safe_dump({"frozen": json.loads(json.dumps(frozen))}, sort_keys=False, width=100)
    p.write_text(head + FROZEN_MARKER + "\n" + body, encoding="utf-8")
    load_config(p)


def set_status(path: str | Path, status: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith("status: ")]
    if len(hits) != 1:
        raise ValueError(f"{p}: expected one top-level 'status:' line")
    lines[hits[0]] = f"status: {status}"
    p.write_text("\n".join(lines), encoding="utf-8")
    load_config(p)


def resolve(path: str | Path, base: Path = REPO) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def real_secret_path() -> Path:
    """Where the repository's own config keeps the real secret (a dry run never reads it)."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return resolve(cfg["heldout"]["secret_file"]).resolve()


# --------------------------------------------------------------------------- heroes and conditions


def hero_specs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen hero plans as TaskSpecs, each checked against its pinned task id."""
    from rtt.studio.engineer.webdata import recorded_plans
    from rtt.studio.types import task_spec_from_tool_args

    plans = recorded_plans(resolve(cfg["heroes"]["plans"]))
    examples = {e["case_id"]: e for e in plans["examples"]}
    out = {}
    for case, task_id in cfg["heroes"]["cases"].items():
        plan = examples[case]["plan"]
        spec = task_spec_from_tool_args({k: plan[k] for k in PLAN_KEYS}, source="recorded")
        if spec.task_id != task_id:
            raise ValueError(f"{case}: the recorded plan hashes to {spec.task_id[:12]}, not {task_id[:12]}")
        out[case] = spec
    return out


def dev_conditions(spec: Any, cfg: Mapping[str, Any]) -> tuple[Any, ...]:
    from rtt.studio.runner import condition_grid

    dev = cfg["development"]
    return condition_grid(spec, int(dev["conditions"]), radius_mm=float(dev["placement_mm"]))


def heldout_conditions(secret: str, spec: Any, heldout: Mapping[str, Any]) -> tuple[Any, ...]:
    """The hero's held-out conditions, drawn from the secret, domain-separated by the hero's task id.

    Condition ``heldoutNN``: a seed of at least ``seed_floor`` (the development seeds are 0 to 7;
    the seed also draws the friction jitter, +-10 %, and the +-0.5 mm aim jitter of task_v2) and, for
    every object in task order, a hidden aim offset uniform in +-``placement_mm`` on x and y.
    """
    from rtt.studio.episode import StudioCondition

    n = int(heldout["conditions"])
    radius = float(heldout["placement_mm"])
    floor = int(heldout["seed_floor"])
    digest = hashlib.sha256(f"{heldout['domain']}|{spec.task_id}|{secret.strip()}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
    seen: set[int] = set()
    out = []
    for i in range(n):
        seed = int(rng.integers(floor, 2**31 - 1))
        while seed in seen:
            seed = int(rng.integers(floor, 2**31 - 1))
        seen.add(seed)
        offsets = {}
        for obj in spec.objects:
            u = rng.uniform(-1.0, 1.0, size=2)
            offsets[obj.name] = (round(float(u[0]) * radius, 3), round(float(u[1]) * radius, 3))
        out.append(StudioCondition(f"heldout{i:02d}", seed, offsets))
    return tuple(out)


# --------------------------------------------------------------------------- executors


def make_pool(
    executor: str,
    run_id: str,
    session: Path,
    *,
    prefix: str = "",
    parallel: int = 4,
    cfg: Mapping[str, Any] | None = None,
    nebius: Any = None,
) -> Any:
    """``local`` (S5's own worker pool, at most 4 processes), ``fake`` (the fan-out in local job
    subprocesses) or ``nebius`` (Serverless Jobs; ``nebius`` is a ready :class:`NebiusExecutor`)."""
    from rtt.studio.cloud import CloudPool, SubprocessExecutor
    from rtt.studio.runner import WorkerPool

    if executor == "local":
        return WorkerPool(min(4, parallel))
    if executor == "fake":
        ex = SubprocessExecutor(session / "bucket", parallel=min(4, parallel), workers=1)
        return CloudPool(ex, run_id, session_dir=session, prefix=prefix)
    if executor == "nebius":
        if nebius is None:
            raise ValueError("a nebius pool needs a NebiusExecutor")
        return CloudPool(nebius, run_id, session_dir=session, prefix=prefix)
    raise ValueError(f"unknown executor {executor!r}")


def nebius_executor(cfg: Mapping[str, Any], staging: Path) -> Any:
    """The benchmark's :class:`~rtt.studio.cloud.NebiusExecutor`: its own capped ledger (the
    benchmark's dollar cap over every benchmark job), each line also appended to the project's."""
    from rtt.cloud.config import load_config as load_nebius
    from rtt.cloud.cost import SpendLedger
    from rtt.cloud.runs import DEFAULT_LEDGER
    from rtt.studio.cloud import NebiusExecutor

    ex = cfg["executor"]
    if float(ex.get("token_factory_usd", 0.0)) != 0.0:
        raise ValueError("the benchmark has a $0 Token Factory cap: no job may call a model")
    ledger = SpendLedger(resolve(ex["ledger"]), cap_usd=float(ex["cap_usd"]))
    return NebiusExecutor(
        load_nebius(),
        staging=staging,
        ledger=ledger,
        project_ledger=SpendLedger(DEFAULT_LEDGER),
        target=ex["target"],
        vcpus=int(ex["vcpus"]),
        fallback=bool(ex.get("fallback")),
        max_concurrent=int(ex.get("max_concurrent_jobs", 3)),
        expected_run_s=float(ex.get("expected_run_s", 900)),
        on_update=lambda u: print(f"  [{u.elapsed_s:6.0f} s] {u.job.name}: {u.job.state}", flush=True),
    )


# --------------------------------------------------------------------------- freeze


def _candidates(spec: Any, pool: str) -> list[Any]:
    from rtt.studio.search import default_candidates

    cands = list(default_candidates(library=True, task_spec=spec))
    if pool == "full":
        return cands
    if pool == "library":
        return [c for c in cands if c.source == "library"]
    raise ValueError(f"unknown pool {pool!r}")


def run_dev_tournament(spec: Any, pool_kind: str, cfg: Mapping[str, Any], pool: Any) -> dict[str, Any] | None:
    """S5's tournament on the development conditions, with the full pool or the library pool."""
    from rtt.studio.search import run_task_tournament

    dev = cfg["development"]
    cands = _candidates(spec, pool_kind)
    if not cands:
        return None
    rep = run_task_tournament(
        spec,
        candidates=cands,
        n_conditions=int(dev["conditions"]),
        n_screen=int(dev["screen"]),
        n_finalists=int(dev["finalists"]),
        top_k=int(dev["top_k"]),
        multi_head_pairs=int(dev["multi_head_pairs"]),
        payload_kg=float(dev["payload_kg"]),
        controllers=dev["controllers"],
        pool=pool,
    )
    return rep.to_dict()


def _finalist(report: Mapping[str, Any], setup_id: str) -> dict[str, Any]:
    return next(f for f in report["finalists"] if f["setup"]["setup_id"] == setup_id)


def arm_from_report(report: Mapping[str, Any] | None, which: str) -> dict[str, Any]:
    """An arm from a tournament report: ``best_single`` (arm a) or ``recommendation`` (arms b, c).

    No recommendation: the closest setup (S5's ``no_winner.closest[0]``), status ``descriptive``.
    Nothing at all: status ``missing``.
    """
    if report is None:
        return {"setup_id": None, "status": "missing", "basis": "no candidate in this pool"}
    rec = report["recommendation"]
    if which == "best_single":
        sid = report.get("best_single")
        if sid:
            status, basis = "tested", "S5's best single off-the-shelf effector of tournament b"
        else:
            status, basis = "missing", "no single off-the-shelf effector survived pruning"
    else:
        pick = rec.get("recommendation")
        if pick:
            sid, status, basis = pick["setup_id"], "tested", "S5's recommendation"
        else:
            closest = (rec.get("no_winner") or {}).get("closest") or []
            sid = closest[0]["setup_id"] if closest else None
            status = "descriptive" if sid else "missing"
            basis = (
                "no recommendation: S5's closest setup, reported descriptively"
                if sid
                else "no recommendation and no closest setup"
            )
    if sid is None:
        return {"setup_id": None, "status": "missing", "basis": basis}
    f = _finalist(report, sid)
    return {
        "setup_id": sid,
        "status": status,
        "basis": basis,
        "kind": f["setup"]["kind"],
        "label": f["setup"]["label"],
        "tools": [t["display_name"] for t in f["setup"]["tools"]],
        "complexity": f["result"]["complexity"],
        "fidelity": f["result"]["fidelity"],
        "dev": {
            "completion_rate": f["result"]["completion_rate"],
            "runs": [
                {
                    "condition_id": r["condition_id"],
                    "seed": r["seed"],
                    "completed": r["completed"],
                    "traj_sha": r["traj_sha"],
                }
                for r in f["runs"]
            ],
        },
        "setup": f["setup"],
    }


def primary_rule(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Whether a hero is in the primary family (declared at freeze, before any held-out run)."""
    a, b = arms["a"], arms["b"]
    if a["status"] != "tested":
        return {"tested": False, "reason": "arm a is missing: no single off-the-shelf effector"}
    if b["status"] != "tested":
        return {"tested": False, "reason": "arm b has no recommendation: reported descriptively, excluded"}
    if a["setup_id"] == b["setup_id"]:
        return {"tested": False, "reason": "arm b is arm a's setup: identical by construction, excluded"}
    return {"tested": True, "reason": None}


POOLS: Final = {"b": "full", "c": "library"}


def check_reused_report(report: Mapping[str, Any], spec: Any, pool: str, cfg: Mapping[str, Any]) -> list[str]:
    """Why an existing tournament report cannot stand for tournament b or c (empty: it can): the
    task, the controllers, the development conditions and the candidate pool must be this protocol's."""
    from rtt.studio.runner import condition_dict

    dev = cfg["development"]
    grid = [condition_dict(c) for c in dev_conditions(spec, cfg)]
    problems = []
    if report.get("task_id") != spec.task_id:
        problems.append("another task")
    if report.get("controllers") != dev["controllers"] or not report.get("evidence"):
        problems.append("not the default controllers")
    if report.get("conditions") != grid:
        problems.append("not the development conditions")
    if report["screen"]["conditions"] != grid[: int(dev["screen"])]:
        problems.append("not the screen conditions")
    if len(report.get("finalists") or ()) > int(dev["finalists"]):
        problems.append("more finalists than the protocol's")
    want = sorted(c.effector_id for c in _candidates(spec, pool))
    if sorted(c["effector_id"] for c in report["screen"]["candidates"]) != want:
        problems.append(f"its candidate pool is not the {pool} pool of this code")
    return problems


def verify_arms(
    heroes: Mapping[str, Mapping[str, Any]],
    specs: Mapping[str, Any],
    cfg: Mapping[str, Any],
    n: int,
    pool: Any,
) -> dict[str, Any]:
    """Re-run every frozen arm on its first ``n`` development conditions with this code and compare
    each ``traj_sha`` with the frozen one (a reused report must be this code's)."""
    from rtt.sim.task import default_task_v2
    from rtt.studio.episode import StudioJob, run_task

    task = default_task_v2()
    items: list[Any] = []
    want: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for case, h in heroes.items():
        spec = specs[case]
        conds = {c.condition_id: c for c in dev_conditions(spec, cfg)}
        for arm in ARMS:
            a = h["arms"][arm]
            if not a.get("setup_id") or (case, a["setup_id"]) in seen:
                continue  # arms that share a setup are checked once
            seen.add((case, a["setup_id"]))
            setup = _setup(a)
            for run in a["dev"]["runs"][:n]:
                items.append(StudioJob(setup, spec, conds[run["condition_id"]], task=task))
                want.append((case, arm, run["condition_id"], run["traj_sha"]))
    got = [r.traj_sha for r in pool.map(run_task, items)] if items else []
    bad = [f"{c}/{a}/{k}" for (c, a, k, sha), g in zip(want, got, strict=True) if sha != g]
    return {"conditions": n, "runs": len(items), "equal": len(items) - len(bad), "unequal": bad}


def freeze_arms(
    config: Path,
    *,
    executor: str,
    out: Path,
    parallel: int = 4,
    allow_dirty: bool = False,
    heroes: Sequence[str] | None = None,
    reuse: Mapping[str, Path] | None = None,
    refreeze: bool = False,
    verify: int = 2,
) -> dict[str, Any]:
    """The development search (module docstring), written into the frozen block.

    ``reuse``: ``{"b": DIR}`` and/or ``{"c": DIR}`` takes that tournament from
    ``DIR/<case>/<arm>/report.json`` (an earlier freeze) or ``DIR/<case>/report.json`` (an S5 hero
    gate) instead of running it, after :func:`check_reused_report`; the report
    is copied next to the others. ``verify``: re-run each arm's first ``verify`` development
    conditions and refuse unless every ``traj_sha`` matches (:func:`verify_arms`). ``refreeze``:
    replace a frozen block, allowed only while no secret has been committed.
    """
    from rtt.store.vault import VaultGuard

    cfg = load_config(config)
    old = cfg.get("frozen")
    if old and old.get("commitment"):
        raise ValueError(f"{config}: a secret is committed; the arms can no longer change")
    if old and not refreeze:
        raise ValueError(f"{config} already holds a frozen block; pass refreeze to regenerate it")
    git = git_state()
    if git["git_dirty"] and not allow_dirty:
        raise ValueError("freeze on a clean commit (the arms must be reproducible from it)")
    reuse = {k: Path(v) for k, v in (reuse or {}).items()}
    if set(reuse) - set(POOLS):
        raise ValueError(f"reuse names tournaments {sorted(POOLS)} only")
    specs = hero_specs(cfg)
    secret_dir = resolve(cfg["heldout"]["secret_file"]).parent
    neb = nebius_executor(cfg, out / "staging") if executor == "nebius" else None
    frozen: dict[str, Any] = {
        "label": _label(),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {**git, "executor": executor, "image": _image_ref() if executor == "nebius" else None},
        "heroes": {},
        "commitment": None,
    }
    timings: dict[str, float] = {}
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())  # a rerun never reuses a bucket prefix
    with VaultGuard([secret_dir]):
        for case, spec in specs.items():
            if heroes and case not in heroes:
                continue
            reports: dict[str, Any] = {}
            sources: dict[str, str] = {}
            for arm, pool_kind in POOLS.items():
                session = out / case / arm
                if arm in reuse:
                    src = reuse[arm] / case / arm / "report.json"  # a freeze's own layout
                    if not src.is_file():
                        src = reuse[arm] / case / "report.json"  # an S5 hero-gate layout
                    rep = json.loads(src.read_text(encoding="utf-8"))
                    problems = check_reused_report(rep, spec, pool_kind, cfg)
                    if problems:
                        raise ValueError(
                            f"{case}: the reused tournament {arm} report: " + "; ".join(problems)
                        )
                    session.mkdir(parents=True, exist_ok=True)
                    (session / "report.json").write_bytes(src.read_bytes())
                    reports[arm], sources[arm] = rep, "reused"
                    print(f"{case} {arm}: reused", flush=True)
                    continue
                t0 = time.perf_counter()
                run_id = f"usecase-dev-{case.replace('_', '-')}-{arm}-{stamp}"
                pool = make_pool(executor, run_id, session, parallel=parallel, nebius=neb)
                with pool:
                    rep = run_dev_tournament(spec, pool_kind, cfg, pool)
                timings[f"{case}/{arm}"] = round(time.perf_counter() - t0, 1)
                if rep is not None:
                    _write_json(session / "report.json", rep)
                reports[arm], sources[arm] = rep, f"run here ({executor})"
                print(f"{case} {arm}: {timings[f'{case}/{arm}']} s", flush=True)
            arms = {
                "a": arm_from_report(reports["b"], "best_single"),
                "b": arm_from_report(reports["b"], "recommendation"),
                "c": arm_from_report(reports["c"], "recommendation"),
            }
            frozen["heroes"][case] = {
                "task_id": spec.task_id,
                "dev_reports": {
                    arm: (
                        {
                            "sha256": _sha256_file(out / case / arm / "report.json"),
                            "path": f"{case}/{arm}/report.json",
                            "source": sources[arm],
                        }
                        if reports[arm] is not None
                        else None
                    )
                    for arm in POOLS
                },
                "arms": arms,
                "primary": primary_rule(arms),
            }
        if verify > 0:
            t0 = time.perf_counter()
            run_id = f"usecase-dev-verify-{stamp}"
            pool = make_pool(executor, run_id, out / "verify", parallel=parallel, nebius=neb)
            with pool:
                check = verify_arms(frozen["heroes"], specs, cfg, verify, pool)
            timings["verify"] = round(time.perf_counter() - t0, 1)
            if check["unequal"]:
                raise ValueError(f"frozen arms do not reproduce on this code: {check['unequal'][:6]}")
            frozen["code"]["verified"] = check
    frozen["timings_s"] = timings
    write_frozen(config, frozen)
    return frozen


def frozen_problems(cfg: Mapping[str, Any]) -> list[str]:
    """What is missing or inconsistent in the frozen block (empty: complete). The secret is not checked."""
    frozen = cfg.get("frozen") or {}
    problems = []
    for case, task_id in cfg["heroes"]["cases"].items():
        h = (frozen.get("heroes") or {}).get(case)
        if h is None or h.get("task_id") != task_id:
            problems.append(f"{case}: not frozen")
            continue
        n_dev = int(cfg["development"]["conditions"])
        for arm in ARMS:
            a = h["arms"].get(arm) or {}
            if a.get("status") not in ("tested", "descriptive", "missing"):
                problems.append(f"{case}/{arm}: no status")
            if not a.get("setup_id"):
                continue
            if _setup(a).setup_id != a["setup_id"]:
                problems.append(f"{case}/{arm}: the setup does not hash to its id")
            if len(a["dev"]["runs"]) != n_dev:
                problems.append(f"{case}/{arm}: {len(a['dev']['runs'])} development runs, not {n_dev}")
        if h.get("primary") != primary_rule(h["arms"]):
            problems.append(f"{case}: the primary rule does not follow from the arms")
    if not (frozen.get("code") or {}).get("verified"):
        problems.append("the arms were not verified on the code")
    return problems


def _image_ref() -> dict[str, Any] | None:
    from rtt.cloud.config import load_config as load_nebius
    from rtt.cloud.runs import split_image_ref

    image = load_nebius().image
    if not image:
        return None
    name, digest = split_image_ref(image)
    return {"name": name, "digest": digest}


def commit_secret(config: Path, *, secret_file: Path | None = None) -> str:
    from rtt.store.vault import HeldoutVault, seed_sha256

    cfg = load_config(config)
    frozen = cfg.get("frozen")
    if not frozen or not frozen.get("heroes"):
        raise ValueError("freeze the arms first: the secret is drawn after the arms are fixed")
    if frozen.get("commitment"):
        raise ValueError("a commitment is already recorded; a committed secret is never replaced")
    path = secret_file or resolve(cfg["heldout"]["secret_file"])
    vault = HeldoutVault.create(path)
    commitment = seed_sha256(vault.read_seed())
    frozen = dict(
        frozen, commitment=commitment, secret_drawn_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    write_frozen(config, frozen)
    return commitment


def finalize(config: Path, *, secret_file: Path | None = None) -> None:
    from rtt.store.vault import HeldoutVault

    cfg = load_config(config)
    frozen = cfg.get("frozen") or {}
    problems = frozen_problems(cfg)
    if not frozen.get("commitment"):
        problems.append("no commitment")
    else:
        HeldoutVault(secret_file or resolve(cfg["heldout"]["secret_file"])).verify(frozen["commitment"])
    if problems:
        raise ValueError("not ready to freeze: " + "; ".join(problems))
    set_status(config, "frozen")


# --------------------------------------------------------------------------- evaluate


def _setup(arm: Mapping[str, Any]) -> Any:
    import rtt.studio.search  # noqa: F401 - part jaws and library families register on import
    from rtt.studio.buildpack import setup_from_dict

    return setup_from_dict(arm["setup"])


#: What a job runs: a commit's code for the job is these paths (the protocol's config aside).
JOB_CODE_PATHS: Final = ("src/", "configs/", "assets/", "docker/", "uv.lock", "pyproject.toml")
PROTOCOL_CONFIG: Final = "configs/usecase_v1.yaml"


def same_code(a: str, b: str) -> bool:
    """Whether commits ``a`` and ``b`` hold the same job code (they may differ in the docs, tests,
    scripts, the site and this protocol's config, which no job reads)."""
    diff = _git("diff", "--name-only", a, b)
    if diff.returncode != 0:
        return False
    return not [f for f in diff.stdout.split() if f.startswith(JOB_CODE_PATHS) and f != PROTOCOL_CONFIG]


def check_tagged(config: Path, tag: str) -> str:
    """The tag's commit; refuses when the tag is missing or the config differs from the tagged one."""
    sha = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}").stdout.strip()
    if not sha:
        raise ValueError(f"no tag {tag}: the protocol is tagged before any held-out run")
    rel = config.resolve().relative_to(REPO).as_posix()
    tagged = _git("show", f"{tag}:{rel}")
    if tagged.returncode != 0 or tagged.stdout != config.read_text(encoding="utf-8"):
        raise ValueError(f"{rel} differs from the copy at {tag}")
    return sha


def evaluate(
    config: Path,
    *,
    executor: str,
    out: Path,
    dry_run: bool = False,
    secret_file: Path | None = None,
    parallel: int = 4,
) -> dict[str, Any]:
    from rtt.sim.task import default_task_v2
    from rtt.store.vault import HeldoutVault
    from rtt.studio.episode import StudioJob, run_task
    from rtt.studio.runner import condition_dict

    cfg = load_config(config)
    frozen = cfg.get("frozen") or {}
    if not frozen.get("heroes") or not frozen.get("commitment"):
        raise ValueError("evaluate needs frozen arms and a committed secret")
    secret_path = (secret_file or resolve(cfg["heldout"]["secret_file"])).resolve()
    tag_sha = None
    if dry_run:
        if secret_path == real_secret_path() or secret_path.is_relative_to(REPO):
            raise ValueError("a dry run never reads the real secret (or any secret inside the repository)")
    else:
        if cfg.get("status") != "frozen":
            raise ValueError("the config is not frozen")
        tag_sha = check_tagged(config, cfg["tag"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: results are never overwritten")
    secret = HeldoutVault(secret_path).verify(frozen["commitment"])
    specs = hero_specs(cfg)
    task = default_task_v2()
    items: list[Any] = []
    index: dict[tuple[str, str], int] = {}
    slots: list[dict[str, Any]] = []
    conditions: dict[str, list[dict[str, Any]]] = {}
    for case, h in frozen["heroes"].items():
        spec = specs[case]
        held = heldout_conditions(secret, spec, cfg["heldout"])
        conditions[case] = [condition_dict(c) for c in held]
        dev = dev_conditions(spec, cfg)
        for arm in ARMS:
            a = h["arms"][arm]
            if not a.get("setup_id"):
                continue
            setup = _setup(a)
            for which, conds in (("heldout", held), ("parity", dev)):
                for cond in conds:
                    key = (setup.setup_id, f"{cond.condition_id}:{cond.seed}:{spec.task_id}")
                    if key not in index:
                        index[key] = len(items)
                        items.append(StudioJob(setup, spec, cond, task=task, controllers="default"))
                    slots.append(
                        {
                            "hero": case,
                            "arm": arm,
                            "set": which,
                            "condition_id": cond.condition_id,
                            "item": index[key],
                        }
                    )
    image = None
    if executor == "nebius":
        built = os.environ.get(
            "RTT_IMAGE_GIT_SHA"
        )  # written by push_image.sh, exported by jobs/nebius/env.sh
        frozen_sha = (frozen.get("code") or {}).get("git_sha")
        if built and frozen_sha and not same_code(frozen_sha, built):
            raise ValueError("the image was built from code that differs from the frozen arms' code")
        image = _image_ref()
        frozen_image = (frozen.get("code") or {}).get("image")
        if frozen_image and image and frozen_image["digest"] != image["digest"]:
            raise ValueError(
                "the image is not the one the arms were frozen with: push that build or refreeze"
            )
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    neb = nebius_executor(cfg, out / "staging") if executor == "nebius" else None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_id = f"usecase-dry-eval-{stamp}" if dry_run else f"usecase-v1-eval-{stamp}"
    pool = make_pool(executor, run_id, out / "fanout", prefix="eval-", parallel=parallel, nebius=neb)
    with pool:
        results = pool.map(run_task, items)
    wall = time.perf_counter() - t0
    fan_rows = getattr(pool, "rows", None)
    rows = []
    for slot in slots:
        res = results[slot["item"]]
        row = {"hero": slot["hero"], "arm": slot["arm"], "set": slot["set"], **res.to_record()}
        row["cloud"] = fan_rows[slot["item"]]["cloud"] if fan_rows else {"job_id": "local", "est_usd": 0.0}
        rows.append(row)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    _write_json(out / "conditions.json", {"label": _label(), "conditions": conditions})
    revealed = out / "usecase_v1_seed.revealed.txt"
    revealed.write_text(secret + "\n", encoding="utf-8")
    summary = pool.summary() if hasattr(pool, "summary") else None
    builds = []
    for phase in getattr(pool, "phases", []):
        for job in phase.jobs:
            m = json.loads((job.out_dir / "manifest.json").read_text(encoding="utf-8"))
            build = m.get("build") or {}
            builds.append(
                {
                    "job_id": job.job_id,
                    "git_sha": build.get("git_sha"),
                    "git_dirty": build.get("git_dirty"),
                    "image_digest": m.get("image_digest"),
                }
            )
    manifest = {
        "format": RESULTS_FORMAT,
        "label": _label(),
        "mode": "dry_run" if dry_run else "confirmatory",
        "note": DRY_RUN_NOTE if dry_run else None,
        "config_sha256": _sha256_file(config),
        "tag": None if dry_run else cfg["tag"],
        "tag_commit": tag_sha,
        "code": git_state(),
        "executor": executor,
        "run_id": run_id,
        "image": image,
        "job_builds": builds,
        "commitment": frozen["commitment"],
        "runs": len(items),
        "rows": len(rows),
        "wall_s": round(wall, 1),
        "fanout": summary,
    }
    _write_json(out / "evaluation.json", manifest)
    analysis = analyse_results(config, out, secret_file=revealed)
    return {"manifest": manifest, "analysis": analysis}


# --------------------------------------------------------------------------- statistics


def exact_paired_one_sided(b: Sequence[bool], a: Sequence[bool]) -> dict[str, Any]:
    """Exact one-sided paired test of b > a on paired binary outcomes.

    The sign-flip permutation test of the per-condition differences: under the null hypothesis the
    two outcomes of a condition are exchangeable, so every discordant pair is b-only or a-only with
    probability 1/2 (an exact McNemar test). p = P(Binomial(n10 + n01, 1/2) >= n10).
    """
    if len(a) != len(b):
        raise ValueError("paired outcomes need equal lengths")
    n10 = sum(1 for x, y in zip(b, a, strict=True) if x and not y)
    n01 = sum(1 for x, y in zip(b, a, strict=True) if y and not x)
    n = n10 + n01
    p = Fraction(sum(math.comb(n, k) for k in range(n10, n + 1)), 2**n) if n else Fraction(1)
    return {"n": len(a), "n10": n10, "n01": n01, "ties": len(a) - n, "p": float(p)}


def holm(pvalues: Mapping[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    """Holm's step-down adjustment: adjusted p-values and rejections at ``alpha``."""
    order = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(order)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    stop = False
    for k, key in enumerate(order):
        adj = min(1.0, (m - k) * pvalues[key])
        running = max(running, adj)
        reject = not stop and pvalues[key] <= alpha / (m - k)
        stop = stop or not reject
        out[key] = {"p": pvalues[key], "p_holm": running, "reject": reject, "rank": k + 1, "m": m}
    return out


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float] | None:
    if n == 0:
        return None
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf((1 - level) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - level) / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


# --------------------------------------------------------------------------- analysis


def describe_arm(rows: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> dict[str, Any]:
    """Completion with its 95 % interval, per-step success with fidelity tiers and failure modes,
    cycle times, tools and mass, physics flags."""
    n = len(rows)
    done = [r for r in rows if r["completed"]]
    steps: dict[str, dict[str, Any]] = {}
    for r in rows:
        for s in r["steps"]:
            d = steps.setdefault(
                s["step_id"],
                {
                    "primitive": s["primitive"],
                    "n": 0,
                    "successes": 0,
                    "fidelity": Counter(),
                    "failure_modes": Counter(),
                },
            )
            d["n"] += 1
            d["successes"] += bool(s["success"])
            d["fidelity"][s["fidelity"]] += 1
            if s["failure_mode"]:
                d["failure_modes"][s["failure_mode"]] += 1
    step_rows = {
        k: {
            "primitive": v["primitive"],
            "n": v["n"],
            "successes": v["successes"],
            "rate": round(v["successes"] / v["n"], 4) if v["n"] else None,
            "fidelity": dict(sorted(v["fidelity"].items())),
            "failure_modes": dict(v["failure_modes"].most_common()),
        }
        for k, v in steps.items()
    }
    modes = Counter(s["failure_mode"] for r in rows for s in r["steps"] if s["failure_mode"])
    complexity = arm.get("complexity") or {}
    return {
        "setup_id": arm.get("setup_id"),
        "status": arm.get("status"),
        "kind": arm.get("kind"),
        "label": arm.get("label"),
        "tools": arm.get("tools"),
        "n": n,
        "completed": len(done),
        "rate": round(len(done) / n, 4) if n else None,
        "ci95": clopper_pearson(len(done), n),
        "steps": step_rows,
        "cycle_time_s": {
            "median_completed": round(statistics.median(r["cycle_time_s"] for r in done), 3)
            if done
            else None,
            "median_all": round(statistics.median(r["cycle_time_s"] for r in rows), 3) if rows else None,
        },
        "tools_and_mass": {
            "n_tools": complexity.get("n_tools"),
            "n_actuators": complexity.get("n_actuators"),
            "carried_kg": max(
                (m for r in rows for m in r["tool_mass_kg"]), default=complexity.get("tool_mass_kg")
            ),
            "total_kg": complexity.get("total_mass_kg"),
            "changes_per_run": complexity.get("n_changes"),
        },
        "physics": {
            "physics_invalid": modes.get("physics_invalid", 0),
            "physics_unreliable": modes.get("physics_unreliable", 0),
        },
        "fidelity_worst": _worst(r["fidelity"] for r in rows),
        "dev_completion_rate": (arm.get("dev") or {}).get("completion_rate"),
    }


def _worst(tiers: Iterable[str]) -> str | None:
    order = {"rigid": 0, "approximation": 1, "not_simulated": 2}
    tiers = list(tiers)
    return max(tiers, key=lambda t: order.get(t, 3)) if tiers else None


def pairing_problems(
    by_arm: Mapping[str, Sequence[Mapping[str, Any]]], expected: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Every present arm ran exactly the expected conditions (ids and seeds, in order), with one
    fairness hash and the default controllers (common random numbers, STUDIO.md section 5)."""
    want = [(c["condition_id"], int(c["seed"])) for c in expected]
    problems = []
    shas = set()
    for arm, rows in by_arm.items():
        got = [(r["condition_id"], int(r["seed"])) for r in rows]
        if got != want:
            problems.append(f"arm {arm}: its conditions are not the held-out conditions in order")
        if any(r["controllers"] != "default" for r in rows):
            problems.append(f"arm {arm}: a run used reference controllers")
        shas |= {r["fairness_sha"] for r in rows if r["fairness_sha"] != "unbuilt"}
    if len(shas) > 1:
        problems.append("the arms do not share one fairness hash")
    return problems


def analyse(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    conditions: Mapping[str, Sequence[Mapping[str, Any]]],
    mode: str,
    secret: str | None = None,
) -> dict[str, Any]:
    """The analysis of record (docs/USECASE_V1.md section 7): primary test per tested hero, Holm
    across them, the descriptive tables, the parity check and the spend."""
    frozen = cfg["frozen"]
    alpha = float(cfg["analysis"]["alpha"])
    heroes: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    problems: list[str] = []
    for case, h in frozen["heroes"].items():
        held = [r for r in rows if r["hero"] == case and r["set"] == "heldout"]
        by_arm = {arm: [r for r in held if r["arm"] == arm] for arm in ARMS if h["arms"][arm].get("setup_id")}
        exp = conditions[case]
        order = {c["condition_id"]: i for i, c in enumerate(exp)}
        by_arm = {
            k: sorted(v, key=lambda r: order.get(r["condition_id"], 1 << 30)) for k, v in by_arm.items()
        }
        hp = pairing_problems(by_arm, exp)
        problems += [f"{case}: {p}" for p in hp]
        arms = {arm: describe_arm(by_arm.get(arm, []), h["arms"][arm]) for arm in ARMS}
        prim = dict(h["primary"])
        if prim["tested"] and not hp:
            test = exact_paired_one_sided(
                [r["completed"] for r in by_arm["b"]], [r["completed"] for r in by_arm["a"]]
            )
            prim.update(test)
            pvalues[case] = test["p"]
        elif prim["tested"]:
            prim.update(tested=False, reason="pairing problems: " + "; ".join(hp))
        secondary = None
        if "b" in by_arm and "c" in by_arm and not hp:
            t = exact_paired_one_sided(
                [r["completed"] for r in by_arm["b"]], [r["completed"] for r in by_arm["c"]]
            )
            secondary = {
                "descriptive": True,
                "rate_b": arms["b"]["rate"],
                "rate_c": arms["c"]["rate"],
                "difference": round((arms["b"]["rate"] or 0) - (arms["c"]["rate"] or 0), 4),
                "b_only": t["n10"],
                "c_only": t["n01"],
                "same_setup": h["arms"]["b"]["setup_id"] == h["arms"]["c"]["setup_id"],
            }
        parity = []
        for arm in ARMS:
            a = h["arms"][arm]
            if not a.get("setup_id"):
                continue
            want = {d["condition_id"]: d["traj_sha"] for d in a["dev"]["runs"]}
            got = {
                r["condition_id"]: r["traj_sha"]
                for r in rows
                if r["hero"] == case and r["arm"] == arm and r["set"] == "parity"
            }
            parity.append(
                {
                    "arm": arm,
                    "runs": len(got),
                    "equal": sum(1 for k, v in got.items() if want.get(k) == v),
                    "expected": len(want),
                }
            )
        heroes[case] = {
            "task_id": h["task_id"],
            "n_conditions": len(exp),
            "arms": arms,
            "primary": prim,
            "secondary_b_vs_c": secondary,
            "parity": parity,
        }
    family = holm(pvalues, alpha) if pvalues else {}
    for case, res in family.items():
        heroes[case]["primary"].update(p_holm=res["p_holm"], reject=res["reject"])
    spend_jobs = {}
    for r in rows:
        c = r.get("cloud") or {}
        if c.get("job_id") and c.get("job_id") != "local":
            spend_jobs[c["job_id"]] = c.get("job_est_usd", 0.0)
    return {
        "format": ANALYSIS_FORMAT,
        "label": _label(),
        "mode": mode,
        "note": DRY_RUN_NOTE if mode == "dry_run" else None,
        "commitment": frozen["commitment"],
        "secret_revealed": secret,
        "alpha": alpha,
        "test": cfg["analysis"]["test"],
        "multiplicity": cfg["analysis"]["multiplicity"],
        "family": sorted(pvalues),
        "excluded": {k: v["primary"]["reason"] for k, v in heroes.items() if not v["primary"]["tested"]},
        "holm": family,
        "heroes": heroes,
        "problems": problems,
        "spend": {"jobs": len(spend_jobs), "est_usd": round(sum(spend_jobs.values()), 6)},
    }


def analyse_results(config: Path, results: Path, *, secret_file: Path | None = None) -> dict[str, Any]:
    """Read a results folder, check the revealed secret against the commitment and the conditions
    against the secret, analyse, and write ``analysis.json`` and ``summary.txt``."""
    from rtt.store.vault import seed_sha256

    cfg = load_config(config)
    manifest = json.loads((results / "evaluation.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line) for line in (results / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    conditions = json.loads((results / "conditions.json").read_text(encoding="utf-8"))["conditions"]
    secret = None
    secret_path = secret_file or results / "usecase_v1_seed.revealed.txt"
    if secret_path.is_file():
        secret = secret_path.read_text(encoding="utf-8").strip()
        if seed_sha256(secret) != cfg["frozen"]["commitment"]:
            raise ValueError("the revealed secret does not match the commitment")
        from rtt.studio.runner import condition_dict

        specs = hero_specs(cfg)
        for case in cfg["frozen"]["heroes"]:
            again = [condition_dict(c) for c in heldout_conditions(secret, specs[case], cfg["heldout"])]
            if again != conditions[case]:
                raise ValueError(f"{case}: the recorded conditions are not the ones the secret gives")
    out = analyse(rows, cfg, conditions=conditions, mode=manifest["mode"], secret=secret)
    if manifest["mode"] == "confirmatory" and out["problems"]:
        out["note"] = "pairing problems: see problems; tested heroes with problems left the family"
    _write_json(results / "analysis.json", out)
    (results / "summary.txt").write_text(summary_text(out), encoding="utf-8")
    return out


def summary_text(a: Mapping[str, Any]) -> str:
    lines = [a["label"]]
    if a.get("note"):
        lines.append(a["note"])
    lines.append(
        f"test: {a['test']}; {a['multiplicity']} at alpha {a['alpha']}; family: {', '.join(a['family']) or 'none'}"
    )
    for case, h in a["heroes"].items():
        lines.append("")
        lines.append(f"{case} ({h['task_id'][:12]}), {h['n_conditions']} held-out conditions")
        for arm in ARMS:
            d = h["arms"][arm]
            if d["setup_id"] is None:
                lines.append(f"  {arm}: missing")
                continue
            ci = d["ci95"] or ["-", "-"]
            lines.append(
                f"  {arm} [{d['status']}] {d['label']}: {d['completed']}/{d['n']} complete"
                f" (95 % CI {ci[0]}-{ci[1]}), dev {d['dev_completion_rate']}, tools {d['tools_and_mass']['n_tools']},"
                f" carried {d['tools_and_mass']['carried_kg']} kg, fidelity {d['fidelity_worst']}"
            )
            for sid, s in d["steps"].items():
                lines.append(
                    f"      {sid} ({s['primitive']}): {s['successes']}/{s['n']}  tiers {s['fidelity']}"
                    f"  modes {s['failure_modes']}"
                )
        p = h["primary"]
        if p["tested"]:
            lines.append(
                f"  primary b > a: b only {p['n10']}, a only {p['n01']}, ties {p['ties']};"
                f" p {p['p']:.3g}, Holm {p.get('p_holm', float('nan')):.3g}, reject {p.get('reject')}"
            )
        else:
            lines.append(f"  primary: excluded ({p['reason']})")
        if h["secondary_b_vs_c"]:
            s = h["secondary_b_vs_c"]
            lines.append(
                f"  b vs c (descriptive): {s['rate_b']} vs {s['rate_c']}, b only {s['b_only']}, c only {s['c_only']}"
            )
        lines.append(
            "  parity on development conditions: "
            + ", ".join(f"{x['arm']} {x['equal']}/{x['expected']}" for x in h["parity"])
        )
    lines.append("")
    lines.append(f"spend: {a['spend']['jobs']} jobs, ${a['spend']['est_usd']:.4f} (wall-based estimate)")
    if a["problems"]:
        lines.append("problems: " + "; ".join(a["problems"]))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- dry run and estimate


def compare_reports(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Whether two tournament reports of one task agree: finalists' run hashes, screen cells,
    combinations and the recommendation (timings and wall times left out)."""

    def runs(d: Mapping[str, Any]) -> list[tuple[str, str, str]]:
        return sorted(
            (f["setup"]["setup_id"], r["condition_id"], r["traj_sha"])
            for f in d["finalists"]
            for r in f["runs"]
        )

    def rec(d: Mapping[str, Any]) -> str | None:
        return (d["recommendation"].get("recommendation") or {}).get("setup_id")

    return {
        "finalist_runs": len(runs(a)),
        "finalist_traj_sha_equal": runs(a) == runs(b),
        "screen_cells_equal": a["screen"]["cells"] == b["screen"]["cells"],
        "screen_runs": [a["screen"]["runs"], b["screen"]["runs"]],
        "combinations_equal": a["combinations"] == b["combinations"],
        "recommendation_equal": rec(a) == rec(b),
        "best_single_equal": a["best_single"] == b["best_single"],
    }


def _jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(paths):
        out += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return out


def dry_run(
    *,
    heroes: Sequence[str],
    n_conditions: int,
    out: Path,
    parallel: int = 4,
    reference: Path | None = None,
    reuse: Mapping[str, Path] | None = None,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """The whole pipeline on ``heroes`` and ``n_conditions`` held-out conditions, with a throwaway
    secret and the local fake cloud, in ``out`` (outside the repository). ``reuse`` as in
    :func:`freeze_arms`; ``reference``: ``DIR/<case>/report.json`` to compare a tournament b run here with."""
    if out.resolve().is_relative_to(REPO):
        raise ValueError("a dry run writes outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.yaml"
    text = config.read_text(encoding="utf-8").split(FROZEN_MARKER)[0]
    cfg = yaml.safe_load(text)
    unknown = set(heroes) - set(cfg["heroes"]["cases"])
    if unknown:
        raise ValueError(f"unknown heroes {sorted(unknown)}")
    cfg["heroes"]["cases"] = {h: cfg["heroes"]["cases"][h] for h in heroes}
    cfg["heroes"]["plans"] = str(resolve(cfg["heroes"]["plans"]))
    cfg["heldout"]["conditions"] = int(n_conditions)
    cfg["heldout"]["secret_file"] = str(out / "secrets" / "usecase_v1_seed.txt")
    cfg["status"] = "draft"
    cfg_path.write_text(
        f"# {DRY_RUN_NOTE}\n"
        + yaml.safe_dump(cfg, sort_keys=False, width=100)
        + "\n"
        + FROZEN_MARKER
        + "\nfrozen: null\n",
        encoding="utf-8",
    )
    timings = {}
    t = time.perf_counter()
    freeze_arms(cfg_path, executor="fake", out=out / "dev", parallel=parallel, allow_dirty=True, reuse=reuse)
    timings["freeze_s"] = round(time.perf_counter() - t, 1)
    commit_secret(cfg_path)
    t = time.perf_counter()
    res = evaluate(cfg_path, executor="fake", out=out / "results", dry_run=True, parallel=parallel)
    timings["evaluate_s"] = round(time.perf_counter() - t, 1)
    from rtt.studio.cloud import cost_per_run

    jobs = _jsonl(out.rglob("jobs.jsonl"))
    summary = {
        "label": _label(),
        "note": DRY_RUN_NOTE,
        "heroes": list(heroes),
        "heldout_conditions": n_conditions,
        "reused": sorted(reuse or {}),
        "timings_s": timings,
        "per_run_s": cost_per_run(_jsonl(out.rglob("rows.jsonl"))),
        "jobs": len(jobs),
        "job_cpu_s": round(sum(j["cpu_s"] or 0 for j in jobs), 1),
        "job_wall_s": round(sum(j["spend"]["wall_s"] for j in jobs), 1),
        "verified": yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["frozen"]["code"].get("verified"),
        "analysis": {k: res["analysis"][k] for k in ("family", "excluded", "holm", "problems", "spend")},
    }
    if reference is not None and "b" not in (reuse or {}):
        summary["versus_reference"] = {
            h: compare_reports(
                json.loads((reference / h / "report.json").read_text(encoding="utf-8")),
                json.loads((out / "dev" / h / "b" / "report.json").read_text(encoding="utf-8")),
            )
            for h in heroes
        }
    _write_json(out / "dry_run.json", summary)
    return summary


#: Nebius timings used by the estimate (docs/NEBIUS.md, measured on uk-south2 cpu-d3).
PROVISION_S: Final = 120.0  # created -> container start: 111-119 s measured [derived]
CONTAINER_OVERHEAD_S: Final = 15.0  # imports, environment probe, publish [assumed]
PARALLEL_EFFICIENCY: Final = 0.8  # of the vCPUs while a phase runs [assumed]
BUILD_PUSH_S: Final = 15.0 + 156.0  # image build with cached layers, push to both registries [measured]


def job_cost(runs: int, s_per_run: float, longest_s: float, vcpus: int) -> dict[str, float]:
    from rtt.cloud.cost import hourly_usd
    from rtt.cloud.nebius_jobs import cpu_preset

    work = max(longest_s, runs * s_per_run / (vcpus * PARALLEL_EFFICIENCY)) if runs else 0.0
    wall = PROVISION_S + CONTAINER_OVERHEAD_S + work
    rate = hourly_usd("cpu-d3", cpu_preset(vcpus))
    return {
        "wall_s": wall,
        "work_s": work,
        "usd": wall * rate / 3600.0,
        "usd_run_only": (wall - PROVISION_S) * rate / 3600.0,
    }


def estimate(
    *, measured: Sequence[Path], counts: Mapping[str, Any], vcpus: int = 32, heldout: int = 32, n_dev: int = 8
) -> dict[str, Any]:
    """The cost table of the real run from measured seconds per run (the fan-out rows under
    ``measured``) and the
    run counts per hero and pool (``counts``: ``{hero: {pool: {"screen": [runs per chain], "finalists": k}}}``)."""
    from rtt.cloud.cost import hourly_usd
    from rtt.cloud.nebius_jobs import cpu_preset
    from rtt.studio.cloud import cost_per_run

    meas = cost_per_run(_jsonl(p for d in measured for p in Path(d).rglob("rows.jsonl")))
    s_screen = float(meas["screen"]["mean_s"])
    s_run = float(meas["run"]["mean_s"])
    longest = 15.0  # the slowest single full-task run seen on the heroes is about 10 s [measured]
    rows = []
    dev_usd = dev_run = dev_jobs = dev_runs = 0.0
    dev_cpu = dev_wall = 0.0
    for pools in counts.values():
        for c in pools.values():
            for n in c["screen"]:
                if not n:
                    continue
                cost = job_cost(n, s_screen, longest, vcpus)
                dev_wall += cost["wall_s"]
                dev_usd += cost["usd"]
                dev_run += cost["usd_run_only"]
                dev_jobs += 1
                dev_runs += n
                dev_cpu += n * s_screen
            nf = c["finalists"] * n_dev
            cost = job_cost(nf, s_run, longest, vcpus)
            dev_wall += cost["wall_s"]
            dev_usd += cost["usd"]
            dev_run += cost["usd_run_only"]
            dev_jobs += 1
            dev_runs += nf
            dev_cpu += nf * s_run
    rows.append(
        {
            "stage": "development search (freeze-arms on Nebius; optional: local is free)",
            "jobs": int(dev_jobs),
            "runs": int(dev_runs),
            "cpu_s": round(dev_cpu),
            "usd": round(dev_usd, 3),
            "usd_run_only": round(dev_run, 3),
            "wall_s": round(dev_wall),
        }
    )
    n_arms = sum(1 for case in counts for _ in range(3))
    eval_runs = n_arms * (heldout + n_dev)
    cost = job_cost(eval_runs, s_run, longest, vcpus)
    rows.append(
        {
            "stage": f"held-out evaluation: {heldout} conditions + {n_dev} parity runs per arm, one job",
            "jobs": 1,
            "runs": eval_runs,
            "cpu_s": round(eval_runs * s_run),
            "usd": round(cost["usd"], 3),
            "usd_run_only": round(cost["usd_run_only"], 3),
            "wall_s": round(cost["wall_s"]),
        }
    )
    rate = hourly_usd("cpu-d3", cpu_preset(vcpus))
    retry = (900.0 + 60.0) * rate / 3600.0  # one attempt cancelled at the provisioning timeout
    rows.append(
        {
            "stage": "contingency: two attempts cancelled at the 15 min provisioning timeout",
            "jobs": 2,
            "runs": 0,
            "cpu_s": 0,
            "usd": round(2 * retry, 3),
            "usd_run_only": 0.0,
            "wall_s": 2 * 960,
        }
    )
    rows.append(
        {
            "stage": "image build (local) and push to both registries",
            "jobs": 0,
            "runs": 0,
            "cpu_s": 0,
            "usd": 0.0,
            "usd_run_only": 0.0,
            "wall_s": BUILD_PUSH_S,
        }
    )
    total = sum(r["usd"] for r in rows)
    return {
        "label": _label(),
        "preset": f"cpu-d3 {cpu_preset(vcpus)} at ${rate:.3f}/h [sourced]",
        "measured_s_per_run": {"screen": s_screen, "full_run": s_run},
        "measured_runs": {k: v["runs"] for k, v in meas.items()},
        "assumptions": {
            "provisioning_s": PROVISION_S,
            "container_overhead_s": CONTAINER_OVERHEAD_S,
            "parallel_efficiency": PARALLEL_EFFICIENCY,
            "build_push_s": BUILD_PUSH_S,
        },
        "rows": rows,
        "total_usd": round(total, 3),
        "cap_usd": 10.0,
        "token_factory_usd": 0.0,
    }


def count_runs(cfg: Mapping[str, Any], heroes: Sequence[str] | None = None) -> dict[str, Any]:
    """Screen runs per hold chain and finalists per hero and pool, from S5's pruning alone (no
    physics episodes): what the development search will run."""
    from rtt.sim.task import default_task_v2
    from rtt.studio.search import hold_chains, prune

    dev = cfg["development"]
    task = default_task_v2()
    out: dict[str, Any] = {}
    for case, spec in hero_specs(cfg).items():
        if heroes and case not in heroes:
            continue
        out[case] = {}
        for pool in ("full", "library"):
            cands = _candidates(spec, pool)
            eligible, _, _ = prune(spec, cands, task=task, payload_kg=float(dev["payload_kg"]))
            chains = hold_chains(spec)
            out[case][pool] = {
                "candidates": len(cands),
                "screen": [len(eligible[j]) * int(dev["screen"]) for j in range(len(chains))],
                "finalists": int(dev["finalists"]),
            }
    return out


def cost_table_text(est: Mapping[str, Any]) -> str:
    lines = [
        "| Stage | Jobs | Runs | CPU-s | Wall, one after another | Est. $ (wall-based) | Est. $ (run-only) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in est["rows"]:
        wall = f"{r['wall_s'] / 60:.0f} min" if r.get("wall_s") else "-"
        lines.append(
            f"| {r['stage']} | {r['jobs']} | {r['runs']} | {r['cpu_s']} | {wall} | {r['usd']:.3f}"
            f" | {r['usd_run_only']:.3f} |"
        )
    lines.append(f"| **Total** | | | | | **{est['total_usd']:.3f}** | |")
    return "\n".join(lines) + "\n"


def fetch(run_id: str, out: Path, *, region: str = "uk-south2") -> list[Path]:
    """Download ``runs/<run_id>/`` of a fan-out run from the region's bucket (recovery after the
    laptop lost a run it submitted; the job outputs are verified by ``verify_run`` per job)."""
    from rtt.cloud.config import load_config as load_nebius
    from rtt.cloud.runs import run_prefix, verify_run
    from rtt.cloud.storage import BucketStore

    store = BucketStore.from_config(load_nebius().for_region(region))
    files = store.download_prefix(run_prefix(run_id), out)
    for manifest in sorted(out.glob("*/jobs/*/manifest.json")):
        print(manifest.parent.relative_to(out), verify_run(manifest.parent).describe().splitlines()[0])
    return files


# --------------------------------------------------------------------------- command line


def _reuse(a: argparse.Namespace) -> dict[str, Path]:
    return {k: v for k, v in (("b", a.reuse_b), ("c", a.reuse_c)) if v is not None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/studio_usecase_v1.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze-arms", help="development search; writes the frozen arms")
    fz.add_argument("--executor", choices=("local", "fake", "nebius"), default="nebius")
    fz.add_argument("--out", type=Path, default=DEFAULT_DEV)
    fz.add_argument("--parallel", type=int, default=4)
    fz.add_argument("--allow-dirty", action="store_true")
    fz.add_argument("--refreeze", action="store_true", help="replace the frozen block (before any secret)")
    fz.add_argument("--verify", type=int, default=2, help="development conditions re-run per arm (0: none)")
    for p in (fz,):
        p.add_argument("--reuse-b", type=Path, default=None, help="tournament b from DIR/<case>/report.json")
        p.add_argument("--reuse-c", type=Path, default=None, help="tournament c from DIR/<case>/report.json")
    cs = sub.add_parser("commit-secret", help="draw the held-out secret and commit its SHA-256")
    cs.add_argument("--secret-file", type=Path, default=None)
    fn = sub.add_parser("finalize", help="check everything, then set status: frozen")
    fn.add_argument("--secret-file", type=Path, default=None)
    ev = sub.add_parser("evaluate", help="after the tag: the held-out evaluation and the analysis")
    ev.add_argument("--executor", choices=("local", "fake", "nebius"), default="nebius")
    ev.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    ev.add_argument("--secret-file", type=Path, default=None)
    ev.add_argument("--parallel", type=int, default=4)
    an = sub.add_parser("analyse", help="analyse a results folder")
    an.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    dr = sub.add_parser("dry-run", help="the whole pipeline on a reduced set with a throwaway secret")
    dr.add_argument("--heroes", default="h1_tote", help="comma-separated hero ids, or all")
    dr.add_argument("--conditions", type=int, default=2)
    dr.add_argument("--out", type=Path, required=True)
    dr.add_argument("--parallel", type=int, default=4)
    dr.add_argument(
        "--reference", type=Path, default=None, help="S5 reports DIR/<case>/report.json to compare with"
    )
    dr.add_argument("--reuse-b", type=Path, default=None)
    dr.add_argument("--reuse-c", type=Path, default=None)
    fe = sub.add_parser("fetch", help="download a fan-out run's bucket prefix (recovery)")
    fe.add_argument("--run-id", required=True)
    fe.add_argument("--out", type=Path, required=True)
    fe.add_argument("--region", default="uk-south2")
    es = sub.add_parser("estimate", help="the cost table from measured fan-out runs")
    es.add_argument("--measured", type=Path, nargs="+", required=True, help="folders holding rows.jsonl")
    es.add_argument("--vcpus", type=int, default=32)
    es.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "freeze-arms":
        frozen = freeze_arms(
            a.config,
            executor=a.executor,
            out=a.out,
            parallel=a.parallel,
            allow_dirty=a.allow_dirty,
            reuse=_reuse(a),
            refreeze=a.refreeze,
            verify=a.verify,
        )
        arms = {
            k: {arm: (v["arms"][arm]["status"], (v["arms"][arm]["setup_id"] or "")[:12]) for arm in ARMS}
            for k, v in frozen["heroes"].items()
        }
        print(json.dumps({"arms": arms, "verified": frozen["code"].get("verified")}))
    elif a.cmd == "commit-secret":
        print("commitment", commit_secret(a.config, secret_file=a.secret_file))
    elif a.cmd == "finalize":
        finalize(a.config, secret_file=a.secret_file)
        print("status: frozen")
    elif a.cmd == "evaluate":
        evaluate(a.config, executor=a.executor, out=a.out, secret_file=a.secret_file, parallel=a.parallel)
        print((a.out / "summary.txt").read_text(encoding="utf-8"))
    elif a.cmd == "analyse":
        analyse_results(a.config, a.results)
        print((a.results / "summary.txt").read_text(encoding="utf-8"))
    elif a.cmd == "dry-run":
        cases = list(load_config(a.config)["heroes"]["cases"]) if a.heroes == "all" else a.heroes.split(",")
        s = dry_run(
            heroes=cases,
            n_conditions=a.conditions,
            out=a.out,
            parallel=a.parallel,
            reference=a.reference,
            reuse=_reuse(a),
            config=a.config,
        )
        print(json.dumps(s, indent=1))
        print((a.out / "results" / "summary.txt").read_text(encoding="utf-8"))
    elif a.cmd == "fetch":
        print(f"{len(fetch(a.run_id, a.out, region=a.region))} files")
    elif a.cmd == "estimate":
        cfg = load_config(a.config)
        counts = count_runs(cfg)
        est = estimate(
            measured=a.measured, counts=counts, vcpus=a.vcpus, heldout=int(cfg["heldout"]["conditions"])
        )
        est["counts"] = counts
        if a.out:
            _write_json(a.out, est)
        print(cost_table_text(est))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
