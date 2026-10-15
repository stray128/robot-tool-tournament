"""Milestone 1 (gate G1): the manual entrants plus three deliberate failures.

Every case runs at controller setting k = 0.5 on one box, 90 x 40 x 120 mm
(object-frame x, y, z; the jaw closes along y), 0.30 kg at 700 kg/m^3. Its 40 mm
grip width is inside the design validator's widest training object (45 mm), so
every valid finger opens around it.

Fingers come from the real design pipeline: each manual entrant in
``configs/entrants.yaml`` is validated against the simulator's own gripper
(:func:`~rtt.sim.task.gripper_task_for`) and built with
:func:`rtt.design.generate.generate_finger`.

Cases:

* One per entrant (``straight``, ``concave``, ``hooked``): lift and hold, so success.
* ``too_high`` (straight entrant): the tips close 30 mm above the box -> ``missed_grasp``.
* ``low_friction`` (straight entrant): mu = 0.05, so the friction capacity
  2 * mu * F/2 = 2.0 N is below the box's 2.97 N weight -> ``slip_during_lift``.
* ``too_wide`` (straight entrant): the same box turned so its 90 mm side spans the
  80 mm opening -> ``missed_grasp``.

The gate passes when every entrant is valid, every case ends as expected (outcome
and failure mode), and the box rests on the table, not on a finger, before the
fingers descend. ``rtt m1`` runs the gate and writes a replay and an MP4 per case.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtt.design.generate import generate_finger
from rtt.design.schema import FingerDesign, load_entrants
from rtt.design.types import FingerGeometry
from rtt.design.validate import ValidationResult, validate
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.objects import ObjectSpec, Placement, height
from rtt.sim.render import render_mp4
from rtt.sim.replay import write_replay
from rtt.sim.score import FailureMode
from rtt.sim.task import TaskConfig, default_task, gripper_task_for

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTRANTS = REPO_ROOT / "configs" / "entrants.yaml"
DEFAULT_OUT = REPO_ROOT / "data" / "demo" / "m1"

G1_SETTING = 1  # controller setting k = 0.5
G1_BOX = ObjectSpec("box", (0.09, 0.04, 0.12))
G1_WIDE_BOX = ObjectSpec("box", (0.04, 0.09, 0.12))  # G1_BOX turned: 90 mm across the jaw
BASELINE_ENTRANT = "straight"
TOO_HIGH_CLEARANCE = 0.03  # m above the object's top
LOW_FRICTION = 0.05
GRAVITY = 9.81


@dataclass(frozen=True)
class Entrant:
    """A manual entrant: its design, its validation against the sim gripper, and its finger."""

    design: FingerDesign
    validation: ValidationResult
    finger: FingerGeometry


@dataclass(frozen=True)
class GateCase:
    """One G1 episode and the outcome it must produce."""

    name: str
    entrant: str
    finger: FingerGeometry = field(repr=False)
    obj: ObjectSpec
    task: TaskConfig = field(repr=False)
    expect_success: bool
    expect_mode: FailureMode | None
    description: str


@dataclass(frozen=True)
class CaseOutcome:
    """A gate case, its episode and the files written for it."""

    case: GateCase
    result: EpisodeResult
    files: tuple[Path, ...] = ()

    @property
    def rests_on_table(self) -> bool:
        """Before the fingers descend the table carries the object's weight and no finger touches it."""
        m = self.result.metrics
        return abs(m["rest_table_support"] - 1.0) < 0.02 and m["rest_finger_force"] < 1e-6

    @property
    def as_expected(self) -> bool:
        """Success when success is expected; otherwise a failure with the expected mode."""
        res, case = self.result, self.case
        if case.expect_success:
            return res.success
        return not res.success and res.failure_mode == case.expect_mode

    @property
    def passed(self) -> bool:
        return self.as_expected and self.rests_on_table


@dataclass(frozen=True)
class GateReport:
    """Outcome of :func:`run_gate`."""

    seed: int
    task_name: str
    grasp_height_frac: float
    entrants: Mapping[str, Entrant]
    outcomes: tuple[CaseOutcome, ...]
    wall_time_s: float

    @property
    def invalid_entrants(self) -> dict[str, list[str]]:
        return {n: e.validation.errors for n, e in self.entrants.items() if not e.validation.ok}

    @property
    def passed(self) -> bool:
        return not self.invalid_entrants and all(o.passed for o in self.outcomes)

    @property
    def sim_steps(self) -> int:
        return sum(o.result.n_steps for o in self.outcomes)

    @property
    def sim_wall_s(self) -> float:
        """Wall time spent inside the episodes (excludes file export and video rendering)."""
        return sum(o.result.wall_time_s for o in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (no recordings)."""
        return {
            "passed": self.passed,
            "seed": self.seed,
            "task": self.task_name,
            "controller_setting": G1_SETTING,
            "grasp_height_frac": self.grasp_height_frac,
            "invalid_entrants": self.invalid_entrants,
            "cases": {
                o.case.name: {
                    "passed": o.passed,
                    "description": o.case.description,
                    "entrant": o.case.entrant,
                    "object": {"kind": o.case.obj.kind, "size_m": list(o.case.obj.size)},
                    "expect": "success" if o.case.expect_success else o.case.expect_mode,
                    "success": o.result.success,
                    "failure_mode": o.result.failure_mode,
                    "rests_on_table": o.rests_on_table,
                    "max_lift_m": round(o.result.max_lift, 4),
                    "hold_time_s": round(o.result.hold_time, 3),
                    "flags": list(o.result.flags),
                    "sim_steps": o.result.n_steps,
                    "wall_time_s": round(o.result.wall_time_s, 3),
                    "files": [str(p) for p in o.files],
                }
                for o in self.outcomes
            },
            "sim_steps": self.sim_steps,
            "sim_wall_s": round(self.sim_wall_s, 3),
            "wall_time_s": round(self.wall_time_s, 2),
        }

    def table(self) -> str:
        """Plain-text results table followed by a one-line verdict."""
        header = (
            "case",
            "finger",
            "object mm",
            "expect",
            "result",
            "lift mm",
            "hold s",
            "steps",
            "wall s",
            "check",
        )
        rows = [header]
        for o in self.outcomes:
            res = o.result
            if o.passed:
                verdict = "ok"
            else:
                verdict = "FAIL" if not o.as_expected else "FAIL: object not resting on the table"
            rows.append(
                (
                    o.case.name,
                    o.case.entrant,
                    _object_label(o.case.obj),
                    "success" if o.case.expect_success else str(o.case.expect_mode),
                    "success" if res.success else str(res.failure_mode),
                    f"{1000.0 * res.max_lift:.1f}",
                    f"{res.hold_time:.2f}",
                    str(res.n_steps),
                    f"{res.wall_time_s:.2f}",
                    verdict,
                )
            )
        right = {5, 6, 7, 8}
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        lines = [
            "  ".join(
                c.rjust(w) if i in right else c.ljust(w)
                for i, (c, w) in enumerate(zip(r, widths, strict=True))
            ).rstrip()
            for r in rows
        ]
        lines.insert(1, "  ".join("-" * w for w in widths).rstrip())
        for name, errors in self.invalid_entrants.items():
            lines.append(f"invalid entrant {name}: " + "; ".join(errors))
        n_ok = sum(o.passed for o in self.outcomes)
        rate = self.sim_steps / self.sim_wall_s if self.sim_wall_s > 0 else 0.0
        lines += [
            "",
            (
                f"G1 {'PASSED' if self.passed else 'FAILED'}: {n_ok}/{len(self.outcomes)} cases as expected"
                f" (task {self.task_name}, seed {self.seed}, k = {self.grasp_height_frac:g})."
                f" Physics {self.sim_wall_s:.2f} s for {self.sim_steps} steps ({rate:,.0f} steps/s);"
                f" total wall {self.wall_time_s:.1f} s."
            ),
        ]
        return "\n".join(lines)


def _object_label(obj: ObjectSpec) -> str:
    return f"{obj.kind} " + "x".join(f"{1000.0 * s:g}" for s in obj.size)


def load_gate_entrants(
    task: TaskConfig | None = None, path: str | Path = DEFAULT_ENTRANTS
) -> dict[str, Entrant]:
    """Load, validate and generate every manual entrant, keyed by name in file order."""
    gripper = gripper_task_for(task or default_task())
    entrants: dict[str, Entrant] = {}
    for design in load_entrants(path):
        name = design.name or design.design_id()
        if name in entrants:
            raise ValueError(f"{path}: duplicate entrant name {name!r}")
        entrants[name] = Entrant(design, validate(design, gripper), generate_finger(design))
    return entrants


def g1_cases(task: TaskConfig | None = None, entrants: Mapping[str, Entrant] | None = None) -> list[GateCase]:
    """The G1 cases under ``task`` (default: the repository task) for ``entrants``."""
    task = task or default_task()
    entrants = entrants if entrants is not None else load_gate_entrants(task)
    if BASELINE_ENTRANT not in entrants:
        raise ValueError(f"the G1 failure cases need an entrant named {BASELINE_ENTRANT!r}")
    base = entrants[BASELINE_ENTRANT].finger
    k = task.controller.grasp_height_fracs[G1_SETTING]
    h = height(G1_BOX)
    weight = math.prod(G1_BOX.size) * task.physics.object_density * GRAVITY
    capacity = 2.0 * LOW_FRICTION * task.gripper.grip_force / 2.0
    cases = [
        GateCase(
            name, name, e.finger, G1_BOX, task, True, None, f"the {name} entrant lifts and holds the box"
        )
        for name, e in entrants.items()
    ]
    cases += [
        GateCase(
            name="too_high",
            entrant=BASELINE_ENTRANT,
            finger=base,
            obj=G1_BOX,
            task=task.with_overrides(controller={"grasp_z_offset": h + TOO_HIGH_CLEARANCE - k * h}),
            expect_success=False,
            expect_mode="missed_grasp",
            description=f"the tips close {1000 * TOO_HIGH_CLEARANCE:g} mm above the box top",
        ),
        GateCase(
            name="low_friction",
            entrant=BASELINE_ENTRANT,
            finger=base,
            obj=G1_BOX,
            task=task.with_overrides(physics={"friction": LOW_FRICTION}),
            expect_success=False,
            expect_mode="slip_during_lift",
            description=f"mu = {LOW_FRICTION:g}: friction capacity {capacity:.1f} N < weight {weight:.2f} N",
        ),
        GateCase(
            name="too_wide",
            entrant=BASELINE_ENTRANT,
            finger=base,
            obj=G1_WIDE_BOX,
            task=task,
            expect_success=False,
            expect_mode="missed_grasp",
            description="the box's 90 mm side spans the jaw, wider than its opening",
        ),
    ]
    return cases


def run_case(
    case: GateCase, seed: int, record: bool = False, finger: FingerGeometry | None = None
) -> EpisodeResult:
    """Run one gate case (``finger`` replaces the case's finger, for tests)."""
    return run_episode(
        finger or case.finger, case.obj, Placement(), G1_SETTING, seed, record=record, task=case.task
    )


def export_demo(
    result: EpisodeResult,
    case: GateCase,
    stem: Path,
    video: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Write ``<stem>.bin/.json`` (and ``<stem>.mp4``) for a recorded episode.

    ``meta`` is merged into the replay header's ``meta`` block.
    """
    if result.replay is None or result.qpos is None:
        raise ValueError("episode was not recorded")
    result.replay.header["meta"].update(meta or {})
    paths = list(write_replay(result.replay, stem))
    if video:
        paths.append(
            render_mp4(
                case.finger,
                case.obj,
                case.task,
                result.qpos,
                stem.with_suffix(".mp4"),
                fps=case.task.replay.fps,
            )
        )
    return paths


def run_gate(
    out_dir: str | Path | None = DEFAULT_OUT,
    seed: int = 0,
    video: bool = True,
    task: TaskConfig | None = None,
    entrants_path: str | Path = DEFAULT_ENTRANTS,
) -> GateReport:
    """Run every G1 case; with ``out_dir``, write ``<case>.bin/.json`` (and ``.mp4``) for each."""
    t0 = time.perf_counter()
    task = task or default_task()
    entrants = load_gate_entrants(task, entrants_path)
    outcomes = []
    for case in g1_cases(task, entrants):
        res = run_case(case, seed, record=out_dir is not None)
        files: tuple[Path, ...] = ()
        if out_dir is not None:
            design = entrants[case.entrant].design
            meta = {
                "case": case.name,
                "description": case.description,
                "entrant": case.entrant,
                "design_id": design.design_id(),
                "design": design.model_dump(mode="json", exclude={"rationale"}),
                "expect_success": case.expect_success,
                "expect_failure_mode": case.expect_mode,
            }
            files = tuple(export_demo(res, case, Path(out_dir) / case.name, video=video, meta=meta))
        outcomes.append(CaseOutcome(case, res, files))
    return GateReport(
        seed=seed,
        task_name=task.name,
        grasp_height_frac=task.controller.grasp_height_fracs[G1_SETTING],
        entrants=entrants,
        outcomes=tuple(outcomes),
        wall_time_s=time.perf_counter() - t0,
    )
