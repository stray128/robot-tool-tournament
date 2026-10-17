"""Robustness certificate for any effector: the pre-registered battery, unchanged.

The battery is :func:`rtt.sim.certify.plan_tests` with seed jitter off
(:func:`rtt.sim.certify.certificate_task`): the nominal grasp, placement errors of
+-1, +-2 and +-3 mm along the part's x and y axes, a 1 g shake along x, y and z after the lift,
and a 0.5 N s push at the part's top along x and y: 18 episodes, the same for every family.

Each test reports **held** (the shared lift-and-hold criterion) and **registered** (held,
engaged and in the reference pose). For a ``reference_mode = "nominal_episode"`` family the
nominal episode runs first and its pose becomes the reference of every test (the nominal test
is then re-run against it: same trajectory, now judged against its own pose). If the nominal
episode does not hold, nothing can be registered and every test reports ``registered = False``.

Scores and tolerances follow :class:`rtt.sim.certify.CertificateV2`: per-category pass rates,
their unweighted mean, and per axis the largest placement error up to which every test of that
magnitude passed (``None`` when the nominal grasp fails).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rtt.effectors.registry import get_family
from rtt.effectors.runner import EffectorJob, run_effector_jobs
from rtt.effectors.types import (
    CONTRACT_VERSION,
    PRODUCT_MODE_LABEL,
    EffectorEpisodeResult,
    EffectorSpec,
    RegistrationReference,
)
from rtt.sim import objects as objs
from rtt.sim.certify import CATEGORIES, _tolerance_by, certificate_task, plan_tests
from rtt.sim.task import TaskV2, default_task_v2, task_digest

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["EffectorCertificate", "EffectorCertificateTest", "certify_effector"]


@dataclass(frozen=True)
class EffectorCertificateTest:
    """One certificate episode: its condition and both verdicts."""

    name: str
    category: str
    params: dict[str, Any]
    held: bool
    registered: bool
    engaged: bool
    failure_mode: str | None
    registration_flags: tuple[str, ...]
    offset_mm: float  # |(dx, dy)| from the reference at the registration moment (NaN if not measured)
    max_lift: float
    hold_time: float
    penetration_mm: float
    bad_qacc: int
    flags: tuple[str, ...]
    traj_sha: str


@dataclass(frozen=True)
class EffectorCertificate:
    """Outcome of :func:`certify_effector` (module docstring)."""

    spec: EffectorSpec
    part_id: str
    task: str
    task_digest: str
    reference: str  # "analytic" | "nominal_episode" | "none" (the nominal grasp did not hold)
    tests: tuple[EffectorCertificateTest, ...]
    held_scores: dict[str, float]
    registered_scores: dict[str, float]
    held_score: float
    registered_score: float
    held_tolerance_mm: dict[str, float | None]
    registered_tolerance_mm: dict[str, float | None]
    effector_mass_kg: float
    settings: dict[str, Any] = field(default_factory=dict)
    label: str = PRODUCT_MODE_LABEL

    @property
    def nominal_held(self) -> bool:
        return next(t for t in self.tests if t.name == "nominal").held

    def to_dict(self) -> dict[str, Any]:
        def num(v: float) -> float | None:
            return None if not math.isfinite(v) else round(v, 4)

        return {
            "contract": CONTRACT_VERSION,
            "label": self.label,
            "effector": self.spec.to_dict(),
            "part_id": self.part_id,
            "task": self.task,
            "task_digest": self.task_digest,
            "fidelity": self.spec.fidelity.value,
            "reference": self.reference,
            "effector_mass_kg": num(self.effector_mass_kg),
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
                    "engaged": t.engaged,
                    "failure_mode": t.failure_mode,
                    "registration_flags": list(t.registration_flags),
                    "offset_mm": num(t.offset_mm),
                    "max_lift_m": round(t.max_lift, 4),
                    "hold_time_s": round(t.hold_time, 3),
                    "penetration_mm": num(t.penetration_mm),
                    "bad_qacc": t.bad_qacc,
                    "flags": list(t.flags),
                    "traj_sha": t.traj_sha,
                }
                for t in self.tests
            ],
        }


def _offset_mm(r: EffectorEpisodeResult) -> float:
    dx, dy = r.metrics.get("reg_dx_mm", math.nan), r.metrics.get("reg_dy_mm", math.nan)
    return math.hypot(dx, dy) if math.isfinite(dx) and math.isfinite(dy) else math.nan


def certify_effector(
    spec: EffectorSpec,
    part: PartSpec,
    *,
    placement: objs.Placement | None = None,
    seed: int = 0,
    task: TaskV2 | None = None,
    workers: int | None = 1,
) -> EffectorCertificate:
    """Run the 18-test battery for ``spec`` on ``part`` (module docstring)."""
    base_task = task or default_task_v2()
    ctask = certificate_task(base_task)
    placement = placement or objs.Placement()
    impl = get_family(spec.family)
    planned = plan_tests(part.to_mesh_object(), placement, ctask)

    def jobs(reference: RegistrationReference | None) -> list[EffectorJob]:
        return [
            EffectorJob(
                spec,
                part,
                p.placement,
                seed,
                task=ctask,  # type: ignore[arg-type]
                perturbation=p.perturbation,
                reference=reference,
                condition_id=f"certificate/{p.name}",
            )
            for p in planned
        ]

    reference_source = "analytic"
    reference: RegistrationReference | None = None
    if impl.reference_mode == "nominal_episode":
        nominal = run_effector_jobs(jobs(None)[:1], workers=1)[0]
        reference = nominal.reference_pose() if nominal.held else None
        reference_source = "nominal_episode" if reference is not None else "none"
    results = run_effector_jobs(jobs(reference), workers=workers)

    tests = []
    for p, r in zip(planned, results, strict=True):
        registered = bool(r.success) and reference_source != "none"
        tests.append(
            EffectorCertificateTest(
                name=p.name,
                category=p.category,
                params=p.params,
                held=bool(r.held),
                registered=registered,
                engaged=bool(r.engaged),
                failure_mode=r.failure_mode,
                registration_flags=tuple(r.registration_flags),
                offset_mm=_offset_mm(r),
                max_lift=r.max_lift,
                hold_time=r.hold_time,
                penetration_mm=r.penetration_mm,
                bad_qacc=r.bad_qacc,
                flags=tuple(r.flags),
                traj_sha=r.traj_sha,
            )
        )
    tests_t = tuple(tests)
    held_scores: dict[str, float] = {}
    reg_scores: dict[str, float] = {}
    for cat in CATEGORIES:
        members = [t for t in tests_t if t.category == cat]
        if members:
            held_scores[cat] = sum(t.held for t in members) / len(members)
            reg_scores[cat] = sum(t.registered for t in members) / len(members)
    rb = ctask.robustness
    mass = next((r.effector_mass_kg for r in results if math.isfinite(r.effector_mass_kg)), math.nan)
    return EffectorCertificate(
        spec=spec,
        part_id=part.name,
        task=ctask.name,
        task_digest=task_digest(base_task),
        reference=reference_source,
        tests=tests_t,
        held_scores=held_scores,
        registered_scores=reg_scores,
        held_score=sum(held_scores.values()) / len(held_scores),
        registered_score=sum(reg_scores.values()) / len(reg_scores),
        held_tolerance_mm={a: _tolerance_by(tests_t, a, "held") for a in ("x", "y")},  # type: ignore[arg-type]
        registered_tolerance_mm={a: _tolerance_by(tests_t, a, "registered") for a in ("x", "y")},  # type: ignore[arg-type]
        effector_mass_kg=mass,
        settings={
            "placement_errors_mm": list(rb.placement_errors_mm),
            "shake": {
                "amplitude_m": rb.shake_amplitude,
                "frequency_hz": rb.shake_frequency,
                "cycles": rb.shake_cycles,
            },
            "push": {"impulse_ns": rb.push_impulse, "duration_s": rb.push_duration, "at_top": rb.push_at_top},
            "post_perturb_hold_s": rb.post_perturb_hold,
            "registration": {
                "dx_max_mm": base_task.registration.dx_max_mm,
                "dy_max_mm": base_task.registration.dy_max_mm,
                "yaw_max_deg": base_task.registration.yaw_max_deg,
                "tilt_max_deg": base_task.registration.tilt_max_deg,
                "hold_tilt_max_deg": base_task.registration.hold_tilt_max_deg,
            },
            "jitter": "off",
            "seed": seed,
        },
    )
