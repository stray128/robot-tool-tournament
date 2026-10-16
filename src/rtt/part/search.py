"""Per-part design search and certification, on the existing runner and certificate.

Nothing here is a new search or a new test: :func:`run_part_search` hands the part's
training suite (:func:`rtt.part.conditions.conditions_for_part`) to the benchmark's
:class:`~rtt.search.sim_adapter.SimEvaluator` and :func:`rtt.search.runner.run_search`
(same arms, budget ledger, validator, scoring and top-1 freeze), and
:func:`certify_part` runs :func:`rtt.sim.certify.certify` on each tested resting pose.
:func:`run_part` chains intake, search, certificate and print pack, timing each stage.

Run-directory layout (what :func:`rtt.part.pack.make_print_pack` reads)::

    <run_dir>/part/part.json, part_mm.stl, objects/*.npz   the validated part
    <run_dir>/conditions.json                               the part's training suite
    <run_dir>/search/                                       run_search output (manifest, frozen, ...)
    <run_dir>/certificate.json                              PartCertificate
    <run_dir>/part_run.json                                 stage timings and the chosen design
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from rtt.design.schema import FingerDesign, load_entrants
from rtt.design.validate import validate
from rtt.llm.cache import CacheMode
from rtt.llm.client import LLMClient
from rtt.part.conditions import conditions_for_part
from rtt.part.intake import PartPose, PartSpec, intake, save_part
from rtt.search.base import Evaluator
from rtt.search.runner import ARMS, DEFAULT_ENTRANTS, SearchConfig, SearchOutcome, run_search
from rtt.search.select import FrozenDesign, read_frozen
from rtt.sim.certify import Certificate, certify
from rtt.sim.conditions import Suite
from rtt.sim.objects import Placement
from rtt.sim.task import TaskConfig, default_task
from rtt.store.manifest import write_json_atomic

__all__ = [
    "DEFAULT_PART_BUDGET",
    "DEFAULT_PART_SPEND_CAP_USD",
    "MAX_PART_SPEND_CAP_USD",
    "PartCertificate",
    "PartRun",
    "PartSearch",
    "PoseCertificate",
    "best_frozen",
    "certify_part",
    "check_spend_cap",
    "load_certificate",
    "run_part",
    "run_part_search",
]

DEFAULT_PART_BUDGET: Final = 24
"""Designs simulated per run (round 0 of 8, then two rounds of 8)."""
DEFAULT_PART_SPEND_CAP_USD: Final = 1.0
MAX_PART_SPEND_CAP_USD: Final = 5.0
"""Ceiling on a part run's live model spend cap; a larger, infinite or NaN cap is refused."""


def check_spend_cap(spend_cap_usd: float) -> float:
    """``spend_cap_usd`` as a float, if it is a finite number in ``[0, MAX_PART_SPEND_CAP_USD]``.

    The ledger treats ``None`` as "uncapped" and a NaN or infinite cap never refuses a
    call, so neither may reach it from here.
    """
    if isinstance(spend_cap_usd, bool) or not isinstance(spend_cap_usd, int | float):
        raise TypeError(f"spend_cap_usd must be a number of US dollars, got {spend_cap_usd!r}")
    cap = float(spend_cap_usd)
    if not (math.isfinite(cap) and 0.0 <= cap <= MAX_PART_SPEND_CAP_USD):
        raise ValueError(f"spend_cap_usd must be between 0 and {MAX_PART_SPEND_CAP_USD:g} USD, got {cap!r}")
    return cap


# --------------------------------------------------------------------------- search


@dataclass
class PartSearch:
    """What :func:`run_part_search` produced: the runner's outcome and the frozen designs, best first."""

    part: PartSpec
    suite: Suite
    outcome: SearchOutcome
    best_designs: list[FrozenDesign]
    wall_s: float

    @property
    def best(self) -> FrozenDesign:
        return self.best_designs[0]


def best_frozen(frozen: Sequence[FrozenDesign]) -> list[FrozenDesign]:
    """Frozen designs ranked like the runner ranks history (success, hold, lower mass, earlier run)."""
    return sorted(
        frozen, key=lambda f: (-f.training_success, -f.training_hold_fraction, f.mass_g, f.run_index)
    )


def _part_seed(part: PartSpec) -> int:
    return int(hashlib.sha256(f"rtt-part-search:{part.source_sha256}".encode()).hexdigest()[:12], 16)


def _entrants_for(
    task_entrants: Sequence[FingerDesign], evaluator: Evaluator, limit: int
) -> list[FingerDesign]:
    """Manual entrants that the validator accepts for this part's opening (round 0 needs them valid)."""
    gtask = evaluator.describe().gripper_task
    return [d for d in task_entrants if validate(d, gtask).ok][:limit]


def run_part_search(
    part: PartSpec,
    arm: str = "nemotron",
    budget: int = DEFAULT_PART_BUDGET,
    evaluator: Evaluator | None = None,
    llm: LLMClient | None = None,
    *,
    out_dir: str | Path,
    runs: int = 1,
    seeds: Sequence[int] | None = None,
    suite: Suite | None = None,
    config: SearchConfig | None = None,
    task: TaskConfig | None = None,
    workers: int | None = None,
    spend_cap_usd: float = DEFAULT_PART_SPEND_CAP_USD,
    cache_mode: CacheMode = "readwrite",
) -> PartSearch:
    """Search finger designs for one part with the existing runner.

    Args:
        part: The validated part (:func:`rtt.part.intake.intake`).
        arm: ``nemotron`` (default), ``random`` or ``zeroshot``.
        budget: Designs simulated per run (8 + a multiple of 8).
        evaluator: Default: a CPU :class:`~rtt.search.sim_adapter.SimEvaluator` on the
            part's suite.
        llm: Proposer client for the LLM arms. Default: the pre-registered Nemotron
            client with its cache and spend ledger in ``out_dir`` (capped at
            ``spend_cap_usd``).
        out_dir: Run directory (see the module docstring); ``search/`` must be new.
        runs: Independent runs (R); the best frozen design across them wins.
        seeds: One per run (default: derived from the part's SHA-256).
        suite: Training suite (default :func:`conditions_for_part`).
        config: Search protocol knobs (default :class:`SearchConfig`).
        task: Simulator task (default ``configs/task_v1.yaml``).
        workers: CPU worker processes for the default evaluator.
        spend_cap_usd: Live model spend cap for the default client: a finite number from
            0 to :data:`MAX_PART_SPEND_CAP_USD` (a client passed as ``llm`` keeps its own).
        cache_mode: Default client's cache mode (``replay`` re-runs with zero API calls).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    cap = check_spend_cap(spend_cap_usd)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task = task or default_task()
    suite = suite or conditions_for_part(part)
    if not (out / "part" / "part.json").exists():
        save_part(part, out / "part")
    write_json_atomic(out / "conditions.json", {**suite.describe(), "digest": suite.digest()})

    if llm is None and ARMS[arm].uses_llm:
        from rtt.search.nemotron import make_proposer_client

        llm = make_proposer_client(
            out / "llm_cache",
            cache_mode=cache_mode,
            ledger_path=out / "llm_ledger.jsonl",
            budget_usd=cap,
        )
    pool = None
    if evaluator is None:
        from rtt.search.sim_adapter import SimEvaluator
        from rtt.sim.exec_cpu import CpuExecutor

        pool = CpuExecutor(workers)
    started = time.perf_counter()
    try:
        if pool is not None:
            evaluator = SimEvaluator(suite, task=task, executor=pool, workers=workers)
        assert evaluator is not None
        config = config or SearchConfig()
        run_seeds = list(seeds) if seeds is not None else [_part_seed(part) + i for i in range(runs)]
        entrants = _entrants_for(load_entrants(DEFAULT_ENTRANTS), evaluator, config.round0_size)
        outcome = run_search(
            arm, runs, budget, run_seeds, evaluator, out / "search", llm, config=config, entrants=entrants
        )
    finally:
        if pool is not None:
            pool.close()
    ranked = best_frozen([r.frozen for r in outcome.runs])
    return PartSearch(part, suite, outcome, ranked, time.perf_counter() - started)


# --------------------------------------------------------------------------- certificate


@dataclass(frozen=True)
class PoseCertificate:
    """The robustness battery for one resting pose (and grasp candidate) of the part."""

    pose_rank: int
    probability: float
    candidate: str
    offset_mm: float
    certificate: Certificate

    def to_json(self) -> dict[str, Any]:
        return {
            "pose_rank": self.pose_rank,
            "probability": round(self.probability, 6),
            "candidate": self.candidate,
            "offset_mm": self.offset_mm,
            **self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class PartCertificate:
    """Certificates over the part's tested poses, plus the combined numbers a reader needs.

    ``score`` weights each pose's certificate score by how often the part lands in that
    pose (normalised over the certified poses); ``placement_curve`` gives, per gripper
    axis, the share of certified poses still held at each placement error (mm).
    """

    design_id: str
    controller_setting: int
    poses: tuple[PoseCertificate, ...]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def score(self) -> float:
        total = sum(p.probability for p in self.poses)
        if total <= 0:
            return math.nan
        return sum(p.probability * p.certificate.score for p in self.poses) / total

    @property
    def passed(self) -> int:
        return sum(p.certificate.passed for p in self.poses)

    @property
    def n_tests(self) -> int:
        return sum(len(p.certificate.tests) for p in self.poses)

    def placement_curve(self) -> dict[str, list[tuple[float, float]]]:
        """``{"x": [(error_mm, share held), ...], "y": [...]}``, errors ascending (0 included)."""
        out: dict[str, list[tuple[float, float]]] = {}
        for axis in ("x", "y"):
            by_err: dict[float, list[bool]] = {}
            for pc in self.poses:
                for t in pc.certificate.tests:
                    if t.category != "placement":
                        continue
                    if t.params["axis"] in (axis, None):
                        by_err.setdefault(float(t.params["error_mm"]), []).append(t.success)
            out[axis] = [(e, sum(v) / len(v)) for e, v in sorted(by_err.items())]
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "rtt-part-certificate",
            "version": 1,
            "design_id": self.design_id,
            "controller_setting": self.controller_setting,
            "score": round(self.score, 4),
            "passed": self.passed,
            "n_tests": self.n_tests,
            "created_at": self.created_at,
            "placement_curve": {
                k: [[e, round(s, 4)] for e, s in v] for k, v in self.placement_curve().items()
            },
            "poses": [p.to_json() for p in self.poses],
            "disclaimer": "Simulated in MuJoCo; untested on hardware.",
        }


def _pick_candidate(pose: PartPose, per_condition: Mapping[str, float] | None) -> tuple[str, float]:
    """The pose's narrow-axis grasp candidate the design did best on in training (first on a tie)."""
    cands = [c for c in pose.candidates if c.axis == "narrow"] or list(pose.candidates)
    if not per_condition:
        return cands[0].label, cands[0].offset_mm

    def trained(label: str) -> float:
        hits = [v for cid, v in per_condition.items() if f"/pose{pose.rank}-{label}/" in cid]
        return sum(hits) / len(hits) if hits else -1.0

    best = max(cands, key=lambda c: (trained(c.label), -cands.index(c)))
    return best.label, best.offset_mm


def certify_part(
    design: FingerDesign,
    part: PartSpec,
    *,
    controller_setting: int | None = None,
    per_condition: Mapping[str, float] | None = None,
    poses: Literal["tested", "primary"] = "tested",
    task: TaskConfig | None = None,
    workers: int | None = None,
) -> PartCertificate:
    """Run :func:`rtt.sim.certify.certify` for ``design`` on the part's resting poses.

    Each pose is certified at one grasp candidate: the one whose training conditions
    the design scored best on (``per_condition``, from its training score), else the
    first. ``controller_setting`` defaults to the certificate's own pick per pose.
    """
    from rtt.sim.exec_cpu import CpuExecutor

    task = task or default_task()
    chosen = part.poses if poses == "tested" else part.poses[:1]
    results = []
    with CpuExecutor(workers) as executor:
        for pose in chosen:
            label, offset = _pick_candidate(pose, per_condition)
            cert = certify(
                design,
                pose.obj("narrow"),
                controller_setting=controller_setting,
                placement=Placement(err_x=-offset / 1000.0),
                task=task,
                executor=executor,
            )
            results.append(PoseCertificate(pose.rank, pose.probability, label, offset, cert))
    k = controller_setting if controller_setting is not None else results[0].certificate.controller_setting
    return PartCertificate(design.design_id(), k, tuple(results))


def load_certificate(path: str | Path) -> dict[str, Any]:
    """The JSON written for a :class:`PartCertificate` (``certificate.json``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != "rtt-part-certificate":
        raise ValueError(f"{path} is not a part certificate")
    return data


# --------------------------------------------------------------------------- end to end


@dataclass
class PartRun:
    """Everything :func:`run_part` produced, with stage wall times (seconds)."""

    run_dir: Path
    part: PartSpec
    search: PartSearch
    certificate: PartCertificate
    pack: Path | None
    timings: dict[str, float]


def _per_condition(search_dir: Path, frozen: FrozenDesign) -> dict[str, float]:
    """Training per-condition success of the frozen design (from its run's history)."""
    history = search_dir / "runs" / f"{frozen.arm}-r{frozen.run_index:02d}" / "history.jsonl"
    if not history.exists():
        return {}
    for line in history.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["label"] == frozen.label:
            return {k: float(v) for k, v in item["score"]["per_condition"].items()}
    return {}


def run_part(
    stl_path: str | Path,
    run_dir: str | Path,
    *,
    arm: str = "nemotron",
    budget: int = DEFAULT_PART_BUDGET,
    runs: int = 1,
    units: str = "auto",
    material: str | None = None,
    density: float | None = None,
    name: str | None = None,
    top_k: int = 3,
    llm: LLMClient | None = None,
    evaluator: Evaluator | None = None,
    workers: int | None = None,
    spend_cap_usd: float = DEFAULT_PART_SPEND_CAP_USD,
    cache_mode: CacheMode = "readwrite",
    pack: bool = True,
    intake_kwargs: Mapping[str, Any] | None = None,
) -> PartRun:
    """Intake -> per-part search -> certificate -> print pack, in one run directory."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    check_spend_cap(spend_cap_usd)  # before intake spends minutes of CPU on a doomed run
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    part = intake(
        stl_path,
        units=units,  # type: ignore[arg-type]
        material=material,
        density=density,
        name=name,
        top_k=top_k,
        cache_dir=out / "part" / "cache",
        **dict(intake_kwargs or {}),
    )
    save_part(part, out / "part")
    timings["intake_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    search = run_part_search(
        part,
        arm,
        budget,
        evaluator,
        llm,
        out_dir=out,
        runs=runs,
        workers=workers,
        spend_cap_usd=spend_cap_usd,
        cache_mode=cache_mode,
    )
    timings["search_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    best = search.best
    cert = certify_part(
        best.design,
        part,
        controller_setting=best.controller_setting,
        per_condition=_per_condition(out / "search", best),
        workers=workers,
    )
    write_json_atomic(out / "certificate.json", cert.to_json())
    timings["certify_s"] = time.perf_counter() - t2

    summary: dict[str, Any] = {
        "format": "rtt-part-run",
        "version": 1,
        "part": part.name,
        "arm": arm,
        "budget": budget,
        "runs": runs,
        "best": {
            "frozen": f"search/frozen/{best.arm}-r{best.run_index:02d}.json",
            "design_id": best.design_id,
            "training_success": best.training_success,
            "controller_setting": best.controller_setting,
        },
        "certificate": {"score": round(cert.score, 4), "passed": cert.passed, "n_tests": cert.n_tests},
        "timings_s": {},
    }
    pack_path = None
    if pack:
        from rtt.part.pack import make_print_pack

        t3 = time.perf_counter()
        write_json_atomic(out / "part_run.json", {**summary, "timings_s": _round(timings)})
        pack_path = make_print_pack(out)
        timings["pack_s"] = time.perf_counter() - t3
    timings["total_s"] = time.perf_counter() - t0
    write_json_atomic(out / "part_run.json", {**summary, "timings_s": _round(timings)})
    return PartRun(out, part, search, cert, pack_path, timings)


def _round(timings: Mapping[str, float]) -> dict[str, float]:
    return {k: round(v, 3) for k, v in timings.items()}


def frozen_from_run_dir(run_dir: str | Path) -> FrozenDesign:
    """The best frozen design of a part run directory (verified against its hash)."""
    frozen = [read_frozen(p) for p in sorted((Path(run_dir) / "search" / "frozen").glob("*.json"))]
    if not frozen:
        raise FileNotFoundError(f"{run_dir}: no frozen design under search/frozen/")
    return best_frozen(frozen)[0]
