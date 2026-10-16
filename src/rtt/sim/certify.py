"""Robustness certificate: how much placement error and disturbance a grasp tolerates.

:func:`certify` runs one design on one object (primitive or mesh) through a fixed,
deterministic battery and returns a :class:`Certificate`:

* **placement**: the object sits ``e`` mm from where the controller aims, along the
  gripper x axis (finger width) and y axis (closing axis) separately, for
  ``e`` in ``0, +-1, +-2, +-3`` (``robustness.placement_errors_mm``): 13 episodes.
* **shake**: after the lift, the palm shakes sinusoidally along x, y and z
  (``robustness.shake_*``; default 10 mm at 5 Hz for 5 cycles, peak 9.9 m/s^2, ~1 g):
  3 episodes.
* **push**: after the lift, a 0.5 N s horizontal impulse (25 N for 20 ms) hits the
  object at the height of its top, along x and along y (``robustness.push_*``), so it
  both shoves and twists the object in the grasp: 2 episodes.

Seed jitter is switched off (pose and friction), so every episode is exactly the
stated condition and the certificate is deterministic. A disturbed episode succeeds
only if the object is still held ``post_perturb_hold`` s after the disturbance ends.

Scores: each category's pass rate, and ``score``, their unweighted mean. The
placement tolerance along an axis is the largest ``e`` such that every error of
magnitude <= ``e`` on that axis succeeded (``None`` if even the nominal grasp
fails). ``controller_setting=None`` picks the setting whose nominal grasp works
best (success, then hold, then lift; ties prefer the middle setting).

Task v2 (:func:`certify_v2`) runs the same battery on a freeform jaw pair and its part (there
is no controller setting: the pair carries its grip height) and reports, per test and per
category, both whether the part was **held** (the v1 lift-and-hold criterion) and whether the
grasp was **registered** (seated and centred before the lift, upright after the hold), with a
placement tolerance for each.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from rtt.sim import objects as objs
from rtt.sim.episode import EpisodeResult
from rtt.sim.evaluate import DesignLike, resolve_design
from rtt.sim.exec_cpu import CpuExecutor, EpisodeJob
from rtt.sim.perturb import Perturbation, Push, Shake
from rtt.sim.task import TaskConfig, default_task, task_digest

Category = Literal["placement", "shake", "push"]
CATEGORIES: tuple[Category, ...] = ("placement", "shake", "push")


@dataclass(frozen=True)
class CertificateTest:
    """One certificate episode: its condition and outcome."""

    name: str
    category: Category
    params: dict[str, Any]
    success: bool
    failure_mode: str | None
    max_lift: float
    hold_time: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class Certificate:
    """Outcome of :func:`certify` (see the module docstring)."""

    design_id: str
    object_label: str
    controller_setting: int
    task: str
    task_digest: str
    tests: tuple[CertificateTest, ...]
    category_scores: dict[str, float]
    score: float
    tolerance_mm: dict[str, float | None]
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(t.success for t in self.tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "object": self.object_label,
            "controller_setting": self.controller_setting,
            "task": self.task,
            "task_digest": self.task_digest,
            "score": round(self.score, 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "tolerance_mm": self.tolerance_mm,
            "passed": f"{self.passed}/{len(self.tests)}",
            "settings": self.settings,
            "tests": [
                {
                    "name": t.name,
                    "category": t.category,
                    "params": t.params,
                    "success": t.success,
                    "failure_mode": t.failure_mode,
                    "max_lift_m": round(t.max_lift, 4),
                    "hold_time_s": round(t.hold_time, 3),
                    "flags": list(t.flags),
                }
                for t in self.tests
            ],
        }


@dataclass(frozen=True)
class _Planned:
    name: str
    category: Category
    params: dict[str, Any]
    placement: objs.Placement
    perturbation: Perturbation | None


def certificate_task(task: TaskConfig) -> TaskConfig:
    """``task`` with seed jitter switched off (the certificate states its errors exactly)."""
    return task.with_overrides(jitter={"pos": 0.0, "yaw_deg": 0.0, "mu_frac": 0.0})


def push_height(obj: objs.AnyObject) -> float:
    """Height of the object's top above its centre of mass (m), where a top push acts."""
    com_z = obj.com[2] if isinstance(obj, objs.MeshObject) else 0.0
    return objs.height(obj) / 2.0 - com_z


def plan_tests(obj: objs.AnyObject, placement: objs.Placement, task: TaskConfig) -> list[_Planned]:
    """The certificate battery for ``obj`` at a nominal ``placement`` under ``task.robustness``."""
    rb = task.robustness
    lever = push_height(obj) if rb.push_at_top else 0.0
    tests = [_Planned("nominal", "placement", {"axis": None, "error_mm": 0.0}, placement, None)]
    for axis in ("x", "y"):
        for e in rb.placement_errors_mm:
            for sign in (1.0, -1.0):
                err_mm = sign * e
                field_name = "err_x" if axis == "x" else "err_y"
                shifted = replace(placement, **{field_name: getattr(placement, field_name) + err_mm / 1000.0})
                tests.append(
                    _Planned(
                        f"offset_{axis}_{err_mm:+g}mm",
                        "placement",
                        {"axis": axis, "error_mm": err_mm},
                        shifted,
                        None,
                    )
                )
    for axis in rb.shake_axes:
        shake = Shake(
            axis=axis,  # type: ignore[arg-type]
            amplitude=rb.shake_amplitude,
            frequency=rb.shake_frequency,
            cycles=rb.shake_cycles,
            ramp_cycles=rb.shake_ramp_cycles,
            delay=rb.perturb_delay,
        )
        params = {
            "axis": axis,
            "amplitude_m": rb.shake_amplitude,
            "frequency_hz": rb.shake_frequency,
            "cycles": rb.shake_cycles,
            "peak_accel_m_s2": round(shake.peak_acceleration, 3),
        }
        tests.append(_Planned(f"shake_{axis}", "shake", params, placement, shake))
    for axis in rb.push_axes:
        push = Push(
            axis=axis,  # type: ignore[arg-type]
            impulse=rb.push_impulse,
            duration=rb.push_duration,
            delay=rb.perturb_delay,
            height=lever,
        )
        params = {
            "axis": axis,
            "impulse_ns": rb.push_impulse,
            "duration_s": rb.push_duration,
            "force_n": push.force,
            "height_above_com_m": round(lever, 6),
        }
        tests.append(_Planned(f"push_{axis}", "push", params, placement, push))
    return tests


def _tolerance(tests: Sequence[CertificateTest], axis: str) -> float | None:
    nominal = next(t for t in tests if t.name == "nominal")
    if not nominal.success:
        return None
    by_mag: dict[float, list[bool]] = {}
    for t in tests:
        if t.category == "placement" and t.params["axis"] == axis:
            by_mag.setdefault(abs(t.params["error_mm"]), []).append(t.success)
    tol = 0.0
    for mag in sorted(by_mag):
        if not all(by_mag[mag]):
            break
        tol = mag
    return tol


def _choose_setting(
    finger_job: EpisodeJob, settings: Sequence[int], executor: CpuExecutor
) -> tuple[int, list[EpisodeResult]]:
    jobs = [replace(finger_job, controller_setting=k) for k in settings]
    results = executor.run(jobs).results
    middle = len(settings) // 2
    order = sorted(
        range(len(settings)),
        key=lambda i: (-results[i].success, -results[i].hold_time, -results[i].max_lift, abs(i - middle), i),
    )
    return settings[order[0]], results


def certify(
    design: DesignLike,
    object_spec: objs.AnyObject,
    *,
    controller_setting: int | None = None,
    placement: objs.Placement | None = None,
    seed: int = 0,
    task: TaskConfig | None = None,
    workers: int | None = 1,
    executor: CpuExecutor | None = None,
) -> Certificate:
    """Run the robustness battery for ``design`` grasping ``object_spec`` (module docstring).

    ``workers`` sizes a temporary process pool (1: in-process); pass an open
    ``executor`` to reuse one. ``seed`` is recorded but, with jitter off, cannot
    change the outcome.
    """
    base_task = task or default_task()
    task = certificate_task(base_task)
    placement = placement or objs.Placement()
    design_id, finger = resolve_design(design)
    own = executor is None
    ex = executor or CpuExecutor(workers)
    try:
        template = EpisodeJob(finger, object_spec, placement, 0, seed, task=task)
        if controller_setting is None:
            settings = list(range(len(task.controller.grasp_height_fracs)))
            controller_setting, _ = _choose_setting(template, settings, ex)
        planned = plan_tests(object_spec, placement, task)
        jobs = [
            EpisodeJob(
                finger,
                object_spec,
                p.placement,
                controller_setting,
                seed,
                task=task,
                perturbation=p.perturbation,
            )
            for p in planned
        ]
        results = ex.run(jobs).results
    finally:
        if own:
            ex.close()

    tests = tuple(
        CertificateTest(
            name=p.name,
            category=p.category,
            params=p.params,
            success=r.success,
            failure_mode=r.failure_mode,
            max_lift=r.max_lift,
            hold_time=r.hold_time,
            flags=tuple(f.split("@", 1)[0] for f in r.flags),
        )
        for p, r in zip(planned, results, strict=True)
    )
    category_scores = {}
    for cat in CATEGORIES:
        members = [t.success for t in tests if t.category == cat]
        if members:
            category_scores[cat] = sum(members) / len(members)
    rb = task.robustness
    return Certificate(
        design_id=design_id,
        object_label=getattr(object_spec, "label", object_spec.kind),
        controller_setting=controller_setting,
        task=task.name,
        task_digest=task_digest(base_task),
        tests=tests,
        category_scores=category_scores,
        score=sum(category_scores.values()) / len(category_scores) if category_scores else math.nan,
        tolerance_mm={"x": _tolerance(tests, "x"), "y": _tolerance(tests, "y")},
        settings={
            "grasp_fraction": task.controller.grasp_height_fracs[controller_setting],
            "placement_errors_mm": list(rb.placement_errors_mm),
            "shake": {
                "amplitude_m": rb.shake_amplitude,
                "frequency_hz": rb.shake_frequency,
                "cycles": rb.shake_cycles,
            },
            "push": {"impulse_ns": rb.push_impulse, "duration_s": rb.push_duration, "at_top": rb.push_at_top},
            "post_perturb_hold_s": rb.post_perturb_hold,
            "jitter": "off",
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------- task v2


@dataclass(frozen=True)
class CertificateTestV2:
    """One v2 certificate episode: its condition, whether it held and whether it registered."""

    name: str
    category: Category
    params: dict[str, Any]
    held: bool
    registered: bool
    failure_mode: str | None
    registration_flags: tuple[str, ...]
    seat_short_mm: float
    offset_mm: float  # |(dx, dy)| at the registration moment
    max_lift: float
    hold_time: float
    flags: tuple[str, ...]


@dataclass(frozen=True)
class CertificateV2:
    """Outcome of :func:`certify_v2`: held and registered, per test and category."""

    design_id: str
    part_id: str
    geometry_hash: str
    task: str
    task_digest: str
    tests: tuple[CertificateTestV2, ...]
    held_scores: dict[str, float]
    registered_scores: dict[str, float]
    held_score: float
    registered_score: float
    held_tolerance_mm: dict[str, float | None]
    registered_tolerance_mm: dict[str, float | None]
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def num(v: float) -> float | None:
            return None if not math.isfinite(v) else round(v, 4)

        return {
            "design_id": self.design_id,
            "part_id": self.part_id,
            "geometry_hash": self.geometry_hash,
            "task": self.task,
            "task_digest": self.task_digest,
            "held_score": round(self.held_score, 4),
            "registered_score": round(self.registered_score, 4),
            "held_scores": {k: round(v, 4) for k, v in self.held_scores.items()},
            "registered_scores": {k: round(v, 4) for k, v in self.registered_scores.items()},
            "held_tolerance_mm": self.held_tolerance_mm,
            "registered_tolerance_mm": self.registered_tolerance_mm,
            "held": f"{sum(t.held for t in self.tests)}/{len(self.tests)}",
            "registered": f"{sum(t.registered for t in self.tests)}/{len(self.tests)}",
            "settings": self.settings,
            "tests": [
                {
                    "name": t.name,
                    "category": t.category,
                    "params": t.params,
                    "held": t.held,
                    "registered": t.registered,
                    "failure_mode": t.failure_mode,
                    "registration_flags": list(t.registration_flags),
                    "seat_short_mm": num(t.seat_short_mm),
                    "offset_mm": num(t.offset_mm),
                    "max_lift_m": round(t.max_lift, 4),
                    "hold_time_s": round(t.hold_time, 3),
                    "flags": list(t.flags),
                }
                for t in self.tests
            ],
        }


def _tolerance_by(tests: Sequence[CertificateTestV2], axis: str, attr: str) -> float | None:
    nominal = next(t for t in tests if t.name == "nominal")
    if not getattr(nominal, attr):
        return None
    by_mag: dict[float, list[bool]] = {}
    for t in tests:
        if t.category == "placement" and t.params["axis"] == axis:
            by_mag.setdefault(abs(t.params["error_mm"]), []).append(getattr(t, attr))
    tol = 0.0
    for mag in sorted(by_mag):
        if not all(by_mag[mag]):
            break
        tol = mag
    return tol


def certify_v2(
    pair: Any,
    part: Any,
    *,
    placement: objs.Placement | None = None,
    seed: int = 0,
    task: TaskConfig | None = None,
    workers: int | None = 1,
    executor: CpuExecutor | None = None,
) -> CertificateV2:
    """Run the robustness battery for a v2 jaw pair on its part (module docstring)."""
    from rtt.sim.episode import _as_object
    from rtt.sim.task import default_task_v2

    base_task = task or default_task_v2()
    task = certificate_task(base_task)
    placement = placement or objs.Placement()
    obj = _as_object(part)
    planned = plan_tests(obj, placement, task)
    jobs = [
        EpisodeJob(pair, part, p.placement, -1, seed, task=task, perturbation=p.perturbation) for p in planned
    ]
    own = executor is None
    ex = executor or CpuExecutor(workers)
    try:
        results = ex.run(jobs).results
    finally:
        if own:
            ex.close()
    tests = []
    for p, r in zip(planned, results, strict=True):
        reg = r.registration
        tests.append(
            CertificateTestV2(
                name=p.name,
                category=p.category,
                params=p.params,
                held=r.lift_success,
                registered=r.success,
                failure_mode=r.failure_mode,
                registration_flags=tuple(reg.sub_flags),
                seat_short_mm=reg.seat_short_mm,
                offset_mm=math.hypot(reg.dx_mm, reg.dy_mm),
                max_lift=r.max_lift,
                hold_time=r.hold_time,
                flags=tuple(f.split("@", 1)[0] for f in r.flags),
            )
        )
    tests_t = tuple(tests)
    held_scores, reg_scores = {}, {}
    for cat in CATEGORIES:
        members = [t for t in tests_t if t.category == cat]
        if members:
            held_scores[cat] = sum(t.held for t in members) / len(members)
            reg_scores[cat] = sum(t.registered for t in members) / len(members)
    rb = task.robustness
    return CertificateV2(
        design_id=pair.design_id,
        part_id=str(getattr(part, "part_id", obj.label)),
        geometry_hash=pair.geometry_hash,
        task=task.name,
        task_digest=task_digest(base_task),
        tests=tests_t,
        held_scores=held_scores,
        registered_scores=reg_scores,
        held_score=sum(held_scores.values()) / len(held_scores),
        registered_score=sum(reg_scores.values()) / len(reg_scores),
        held_tolerance_mm={a: _tolerance_by(tests_t, a, "held") for a in ("x", "y")},
        registered_tolerance_mm={a: _tolerance_by(tests_t, a, "registered") for a in ("x", "y")},
        settings={
            "tip_height_m": pair.tip_height,
            "placement_errors_mm": list(rb.placement_errors_mm),
            "shake": {
                "amplitude_m": rb.shake_amplitude,
                "frequency_hz": rb.shake_frequency,
                "cycles": rb.shake_cycles,
            },
            "push": {"impulse_ns": rb.push_impulse, "duration_s": rb.push_duration, "at_top": rb.push_at_top},
            "post_perturb_hold_s": rb.post_perturb_hold,
            "registration": dataclasses.asdict(base_task.registration),
            "jitter": "off",
            "seed": seed,
        },
    )
