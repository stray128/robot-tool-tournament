"""Shared pieces of the v2 held-out runners: frozen picks, entrants, builds, scoring, provenance.

* :func:`collect_frozen_v2` loads every run's frozen top-1 (each record's SHA-256 and design
  id are verified on load by :func:`rtt.search.select.read_frozen`) and refuses anything but
  ``freeform-v2`` picks, or two picks for one run.
* :func:`held_out_entries` lists what is scored: the frozen picks, then the four round-0
  entrants and the two reference entrants of ``configs/entrants_v2.yaml``.
* :func:`build_one` compiles one design against one part exactly as the search did
  (:meth:`rtt.search.space_v2.FreeformSpaceV2.build`: the generator's ``build_pair`` on the
  space's ``buildable`` form). :func:`prebuild` does it for every design and part in a
  ``spawn`` process pool, because a held-out suite has 64-100 parts and a build takes about
  0.1 s (0.5 s for the stock blade).
* :func:`simulate` scores every design on a v2 part suite through
  :func:`rtt.sim.evaluate.evaluate_designs_v2`, the training path: a design that cannot be
  built for a part gets ``invalid_build`` records there, and every episode runs under the
  given ``task_v2``. :func:`design_summary` reduces one design's records with
  :func:`rtt.search.sim_adapter.episode_from_record_v2` and
  :meth:`rtt.search.base.DesignScore.from_episodes_v2`, the functions that give a design
  its training score, so ``success`` is the mean ``success_v2`` and ``lift_success`` the
  lift-only mean, with ``physics_invalid`` and ``physics_unreliable`` failures on both.
* :func:`code_listing` hashes this package (and the command-line and job files that call
  it) for every report, so a result names the code that produced it.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.search.select import FrozenDesign, read_frozen
from rtt.store.manifest import file_sha256

__all__ = [
    "COMMITMENT_FILE",
    "DESIGN_SPACE",
    "PACKAGE_DIR",
    "SEED_MOUNT",
    "SUPPORT_FILES",
    "Entry",
    "Simulation",
    "arm_summary",
    "build_one",
    "code_listing",
    "collect_frozen_v2",
    "design_summary",
    "frozen_digest",
    "held_out_entries",
    "load_entrants",
    "mean_or_none",
    "physics_shares",
    "prebuild",
    "read_commitment",
    "simulate",
    "utc_now",
]

DESIGN_SPACE: Final = "freeform-v2"
PACKAGE_DIR: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
#: The committed SHA-256 of the held-out secret (PREREG A5, A6).
COMMITMENT_FILE: Final = REPO_ROOT / "configs" / "suites" / "heldout_v1.sha256"
#: Where the held-out job receives the secret (``--inject-file``); the vault guard protects it.
SEED_MOUNT: Final = "/secret/heldout_seed"
#: Files outside this package that the held-out commands go through (recorded in reports).
SUPPORT_FILES: Final = (
    "src/rtt/cli.py",
    "src/rtt/cloud/__main__.py",
    "src/rtt/cloud/entry.py",
    "src/rtt/cloud/nebius_jobs.py",
    "src/rtt/cloud/runs.py",
)
ROLES: Final = ("frozen", "round0_entrant", "reference_entrant")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def code_listing() -> dict[str, Any]:
    """SHA-256 of this package's ``sha256sum *.py`` listing, and of each support file.

    ``listing_sha256`` is what ``(export LC_ALL=C; cd src/rtt/heldout_v2 && sha256sum *.py |
    sha256sum)`` prints in bash, or the same command in ``dash``: files in bytewise name
    order. The locale must be set before the glob expands (``LC_ALL=C sha256sum *.py`` sets
    it for the command only, after bash has sorted the glob). In bash under ``en_US.UTF-8``
    the glob sorts ``__init__.py`` elsewhere and the hash differs, as for the PREREG A5
    listings; ``docs/DEVIATIONS.md`` records both.
    """
    names = sorted((p.name for p in PACKAGE_DIR.glob("*.py")), key=lambda n: n.encode())
    files = {name: file_sha256(PACKAGE_DIR / name) for name in names}
    listing = "".join(f"{files[name]}  {name}\n" for name in names)
    return {
        "package": "src/rtt/heldout_v2",
        "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "files": files,
        "support_files": {rel: file_sha256(REPO_ROOT / rel) for rel in SUPPORT_FILES},
    }


def read_commitment(path: str | Path = COMMITMENT_FILE) -> str | None:
    """The first token of a commitment file (``<sha256>  <name>``), or ``None`` if it is missing."""
    p = Path(path)
    if not p.is_file():
        return None
    words = p.read_text(encoding="utf-8").split()
    return words[0].lower() if words else None


def mean_or_none(values: Iterable[float]) -> float | None:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else None


# --------------------------------------------------------------------------- frozen picks


def _frozen_files(raw: str | Path) -> list[Path]:
    p = Path(raw)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"{p}: no frozen record or directory")
    for sub in (p / "search" / "frozen", p / "frozen"):
        if sub.is_dir():
            return sorted(sub.glob("*.json"))
    if (p / "manifest.json").is_file():
        raise FileNotFoundError(f"{p}: a run directory without frozen/ (did the search finish?)")
    return sorted(p.glob("*.json"))


def collect_frozen_v2(paths: Iterable[str | Path]) -> list[FrozenDesign]:
    """Frozen ``freeform-v2`` picks from files, ``frozen/`` directories, search outputs or
    fetched job directories (``runs/cloud/<run_id>``, whose search is under ``search/``).

    Each record is verified on load; records are returned sorted by arm and run index.

    Raises:
        ValueError: a pick from another design space, two picks for one run (arm and run
            index) or one seed used twice in an arm.
        FileNotFoundError: a path holds no frozen records.
    """
    files: list[Path] = []
    for raw in paths:
        found = _frozen_files(raw)
        if not found:
            raise FileNotFoundError(f"{raw}: no frozen records")
        files += found
    frozen = [read_frozen(f) for f in files]
    for f, path in zip(frozen, files, strict=True):
        if f.design_space != DESIGN_SPACE:
            raise ValueError(
                f"{path}: a {f.design_space} pick; the v2 held-out runners score {DESIGN_SPACE} only"
            )
    runs = [(f.arm, f.run_index) for f in frozen]
    if len(set(runs)) != len(runs):
        raise ValueError("two frozen records share an arm and run index")
    seeds = [(f.arm, f.run_seed) for f in frozen]
    if len(set(seeds)) != len(seeds):
        raise ValueError("two frozen records of one arm share a run seed")
    return sorted(frozen, key=lambda f: (f.arm, f.run_index))


def frozen_digest(frozen: Sequence[FrozenDesign]) -> str:
    """SHA-256 over every pick's arm, run index, seed, design id and record hash (order-free)."""
    rows = sorted([f.arm, f.run_index, f.run_seed, f.design_id, f.content_sha256] for f in frozen)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- what is scored


@dataclass(frozen=True)
class Entry:
    """One design to score: a run's frozen pick or an entrant."""

    key: str
    role: str  # "frozen", "round0_entrant" or "reference_entrant"
    arm: str  # the run's arm, or "entrant"
    run_index: int | None
    run_seed: int | None
    label: str
    origin: str
    design: Any
    design_id: str
    frozen_sha256: str | None
    training_success: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "arm": self.arm,
            "run_index": self.run_index,
            "run_seed": self.run_seed,
            "label": self.label,
            "origin": self.origin,
            "design_id": self.design_id,
            "frozen_sha256": self.frozen_sha256,
            "training_success": self.training_success,
        }


def _space() -> Any:
    from rtt.search.space import V2, get_space

    return get_space(V2)


def held_out_entries(frozen: Sequence[FrozenDesign], entrants: Sequence[Any]) -> list[Entry]:
    """The frozen picks (by arm and run index), then the entrants in file order."""
    space = _space()
    out = [
        Entry(
            key=f"{f.arm}-r{f.run_index:02d}",
            role="frozen",
            arm=f.arm,
            run_index=f.run_index,
            run_seed=f.run_seed,
            label=f.label,
            origin=f.origin,
            design=f.design,
            design_id=f.design_id,
            frozen_sha256=f.content_sha256,
            training_success=f.training_success,
        )
        for f in sorted(frozen, key=lambda f: (f.arm, f.run_index))
    ]
    for b in entrants:
        role = "reference_entrant" if getattr(b, "reference", False) else "round0_entrant"
        out.append(
            Entry(
                key=f"entrant-{b.name}",
                role=role,
                arm="entrant",
                run_index=None,
                run_seed=None,
                label=str(b.name),
                origin="entrant",
                design=b,
                design_id=space.design_id(b),
                frozen_sha256=None,
                training_success=None,
            )
        )
    keys = [e.key for e in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate design keys: {keys}")
    return out


def load_entrants(path: str | Path | None = None) -> list[Any]:
    """The v2 entrants (``configs/entrants_v2.yaml`` unless ``path``): round 0 and references."""
    from rtt.design.freeform.baselines import load_entrants_v2

    return list(load_entrants_v2(path))


# --------------------------------------------------------------------------- builds


def build_one(design: Any, part: Any, task: Any) -> Any:
    """One jaw pair, built as the search builds it: ``build_pair(space.buildable(design), ...)``.

    A ``ValueError`` (the generator's ``InvalidDesign``) means the design has no jaws for
    this part; it scores ``invalid_build`` there.
    """
    from rtt.design.freeform import build_pair

    return build_pair(_space().buildable(design), part, task)


def _build_design(args: tuple[int, Any, Sequence[Any], Any]) -> tuple[int, list[tuple[str, Any]]]:
    index, design, parts, task = args
    out: list[tuple[str, Any]] = []
    for part in parts:
        try:
            out.append((str(part.part_id), build_one(design, part, task)))
        except ValueError as exc:
            out.append((str(part.part_id), str(exc) or type(exc).__name__))
    return index, out


def prebuild(
    designs: Sequence[Any], parts: Sequence[Any], task: Any, workers: int | None = None
) -> list[dict[str, Any]]:
    """Per design, ``{part_id: FingerPair, or the build error}``, built in a ``spawn`` pool.

    The result does not depend on ``workers``: every build is a pure function of the
    design, the part and the task (gate GV4 found every geometry hash equal across hosts).
    """
    jobs = [(i, d, tuple(parts), task) for i, d in enumerate(designs)]
    if workers is None or workers <= 1 or len(jobs) <= 1:
        results = [_build_design(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as ex:
            results = list(ex.map(_build_design, jobs))
    out: list[dict[str, Any]] = [{} for _ in designs]
    for index, built in results:
        out[index] = dict(built)
    return out


class _Prebuilt:
    """A ``(design, part, task) -> FingerPair`` builder that serves :func:`prebuild`'s pairs."""

    def __init__(self, designs: Sequence[Any], table: Sequence[Mapping[str, Any]]) -> None:
        self._index = {id(d): i for i, d in enumerate(designs)}
        self._table = table

    def __call__(self, design: Any, part: Any, task: Any) -> Any:
        value = self._table[self._index[id(design)]][str(part.part_id)]
        if isinstance(value, str):
            raise ValueError(value)  # noqa: TRY004 - the invalid_build signal evaluate_designs_v2 expects
        return value


# --------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Simulation:
    """Every record of one suite, by design id, with timings."""

    records: list[Any]
    by_design: dict[str, list[Any]]
    build_s: float
    sim_s: float
    builds: int
    invalid_builds: int


def simulate(
    entries: Sequence[Entry],
    suite: Any,
    task: Any,
    *,
    executor: Any = "cpu",
    workers: int | None = None,
) -> Simulation:
    """Build and run every distinct design of ``entries`` on every condition and seed of ``suite``.

    Records come back from :func:`rtt.sim.evaluate.evaluate_designs_v2` in design, condition,
    seed order and are matched to designs by position, as in the training evaluator.
    """
    from rtt.sim.evaluate import evaluate_designs_v2
    from rtt.sim.exec_cpu import CpuExecutor

    unique: dict[str, Any] = {}
    for e in entries:
        unique.setdefault(e.design_id, e.design)
    designs = list(unique.values())
    parts = suite.parts()
    t0 = time.perf_counter()
    table = prebuild(designs, parts, task, workers)
    build_s = time.perf_counter() - t0
    own = executor == "cpu"
    pool = CpuExecutor(workers) if own else executor
    t1 = time.perf_counter()
    try:
        records = evaluate_designs_v2(
            designs, suite, builder=_Prebuilt(designs, table), executor=pool, task=task
        )
    finally:
        if own:
            pool.close()
    sim_s = time.perf_counter() - t1
    n = suite.n_episodes
    if len(records) != n * len(designs):
        raise RuntimeError(f"{len(records)} records for {len(designs)} designs x {n} episodes")
    space = _space()
    by_design: dict[str, list[Any]] = {}
    for i, (did, design) in enumerate(unique.items()):
        chunk = records[i * n : (i + 1) * n]
        expected = {did, space.design_id(space.buildable(design))}
        named = {r.design_id for r in chunk if r.failure_mode != "invalid_build"}
        if named - expected:
            raise RuntimeError(f"records for design {did[:12]} name other designs: not in design order")
        by_design[did] = chunk
    invalid = sum(isinstance(v, str) for built in table for v in built.values())
    return Simulation(
        records=list(records),
        by_design=by_design,
        build_s=build_s,
        sim_s=sim_s,
        builds=sum(len(b) for b in table),
        invalid_builds=invalid,
    )


def _episodes(records: Sequence[Any]) -> list[Any]:
    from rtt.search.sim_adapter import episode_from_record_v2

    part_of = {r.condition_id: r.part_id for r in records}
    placement_of = {r.condition_id: r.placement_label for r in records}
    return [episode_from_record_v2(r, part_of, placement_of) for r in records]


def _group_rates(
    episodes: Sequence[Any], records: Sequence[Any], key: Callable[[Any], str]
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    succ: dict[str, list[bool]] = {}
    lift: dict[str, list[bool]] = {}
    for ep, rec in zip(episodes, records, strict=True):
        k = key(rec)
        succ.setdefault(k, []).append(bool(ep.success))
        lift.setdefault(k, []).append(bool(ep.lift_success))
    return (
        {k: sum(v) / len(v) for k, v in sorted(succ.items())},
        {k: sum(v) / len(v) for k, v in sorted(lift.items())},
        {k: len(v) for k, v in sorted(succ.items())},
    )


def design_summary(
    entry: Entry, records: Sequence[Any], groups: Mapping[str, Callable[[Any], str]] | None = None
) -> dict[str, Any]:
    """One design's held-out result: registered and lift-only means, failures, physics flags."""
    from rtt.search.base import DesignScore

    if not records:
        raise ValueError(f"{entry.key}: no records")
    episodes = _episodes(records)
    masses = [r.finger_mass_g for r in records if math.isfinite(r.finger_mass_g)]
    score = DesignScore.from_episodes_v2(entry.design_id, mean_or_none(masses) or 0.0, episodes)
    simulated = [r for r in records if r.failure_mode != "invalid_build"]
    modes = Counter(ep.failure_mode or "unknown" for ep in episodes if not ep.success)
    out = {
        **entry.to_json(),
        "success": score.success,
        "lift_success": score.lift_success,
        "episodes": len(records),
        "invalid_build_episodes": len(records) - len(simulated),
        "invalid_build_parts": sorted({r.part_id for r in records if r.failure_mode == "invalid_build"}),
        "physics_invalid": sum(ep.failure_mode == "physics_invalid" for ep in episodes),
        "physics_unreliable": sum(ep.failure_mode == "physics_unreliable" for ep in episodes),
        "jam_on_close": score.jam_on_close,
        "off_centre": score.off_centre,
        "failure_modes": dict(sorted(modes.items())),
        "mean_jaw_pair_mass_g": mean_or_none(masses),
    }
    for name, key in (groups or {}).items():
        succ, lift, count = _group_rates(episodes, records, key)
        out[name] = {"success": succ, "lift_success": lift, "episodes": count}
    return out


def arm_summary(designs: Sequence[Mapping[str, Any]], *, extra: Sequence[str] = ()) -> dict[str, Any]:
    """Run-level means per arm (the frozen picks only), in run order.

    ``extra`` names grouped fields (e.g. ``"by_band"``) whose per-group success and lift-only
    rates are carried into each run row, so a planned split can be tested run by run.
    """
    arms: dict[str, list[Mapping[str, Any]]] = {}
    for d in designs:
        if d["role"] == "frozen":
            arms.setdefault(str(d["arm"]), []).append(d)
    out: dict[str, Any] = {}
    for arm, rows in sorted(arms.items()):
        rows = sorted(rows, key=lambda d: int(d["run_index"]))
        runs = []
        for d in rows:
            row = {k: d[k] for k in ("key", "run_index", "run_seed", "design_id", "frozen_sha256")}
            row["success"] = d["success"]
            row["lift_success"] = d["lift_success"]
            for name in extra:
                row[name] = {"success": d[name]["success"], "lift_success": d[name]["lift_success"]}
            runs.append(row)
        values = [r["success"] for r in runs]
        lifts = [r["lift_success"] for r in runs]
        out[arm] = {
            "n": len(runs),
            "runs": runs,
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "lift_mean": statistics.fmean(lifts),
            "lift_sd": statistics.stdev(lifts) if len(lifts) > 1 else None,
        }
    return out


def physics_shares(records: Sequence[Any]) -> dict[str, Any]:
    """Counts and shares of the physics flags over the simulated (built) episodes."""
    simulated = [r for r in records if r.failure_mode != "invalid_build"]
    n = len(simulated)
    out: dict[str, Any] = {"simulated_episodes": n, "invalid_build_episodes": len(records) - n}
    for mode in ("physics_invalid", "physics_unreliable"):
        k = sum(r.failure_mode == mode or (mode == "physics_invalid" and r.bad_qacc > 0) for r in simulated)
        out[mode] = k
        out[f"{mode}_share"] = k / n if n else None
    return out
