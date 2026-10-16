"""Task configuration: physics, gripper, controller and scoring constants.

Every entrant is evaluated under one :class:`TaskConfig`. The defaults below
mirror ``configs/task_v1.yaml`` exactly (a test keeps them in sync); load the
YAML with :func:`load_task` so that tuned values live in one reviewed file.

The v2 task (freeform jaw pairs, ``configs/task_v2.yaml``) is a :class:`TaskV2`: every v1
section plus ``registration`` (the registered-grasp rule) and ``checks`` (the extra v2
physics checks). It is a subclass, so every v1 helper accepts it, but its controller has no
grasp-height grid (``grasp_height_fracs`` is empty): a v2 jaw pair carries its own designed
grip height. Load it with :func:`load_task_v2`. :class:`TaskConfig` itself is unchanged, so
``task_v1`` and its digest stay exactly as frozen.

Units are SI throughout (metres, kilograms, seconds, newtons, radians) unless
a field name says otherwise (``*_deg``).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rtt.design.schema import GripperTask

DEFAULT_TASK_PATH = Path(__file__).resolve().parents[3] / "configs" / "task_v1.yaml"
DEFAULT_TASK_V2_PATH = DEFAULT_TASK_PATH.with_name("task_v2.yaml")


@dataclass(frozen=True)
class PhysicsConfig:
    """MuJoCo solver and contact settings shared by every episode."""

    timestep: float = 0.002  # 2 ms: matches 1 ms on 99.5 % of a 200-episode parity set (docs/BENCH.md)
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 10.0
    condim: int = 4
    friction: float = 0.3
    torsional_friction: float = 0.001
    rolling_friction: float = 0.0001
    contact_solref: tuple[float, float] = (0.004, 1.0)
    contact_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    limit_solref: tuple[float, float] = (0.004, 1.0)  # joint-limit stiffness (finger stops)
    iterations: int = 100
    ls_iterations: int = 50
    native_ccd: bool = False  # False: libccd MPR for convex pairs (cylinder-box); see task_v1.yaml
    multi_ccd: bool = True  # multiple contact points per convex pair
    object_density: float = 1200.0
    table_top_z: float = 0.0
    table_half_size: tuple[float, float, float] = (0.4, 0.4, 0.375)


@dataclass(frozen=True)
class GripperConfig:
    """Floating palm and parallel-jaw finger drive (identical for every entrant).

    Dimensions follow an open-source rack-and-pinion parallel gripper (two moving
    jaws, 84 mm maximum opening); ``configs/task_v1.yaml`` records the sources and
    assumptions behind each value.
    """

    # Palm = the gripper frame: 47.6 (x) x 128 (y, closing axis) x 24 (z) mm. Its
    # underside is the finger mounting plane (the frame's front face).
    palm_half_size: tuple[float, float, float] = (0.024, 0.064, 0.012)
    palm_mass: float = 0.4
    # Lateral offset of each finger root from the palm centre when fully open.
    # A finger whose inner face sits at local y = 0 leaves a 2 * 0.042 = 84 mm gap.
    open_half_gap: float = 0.042
    stroke: float = 0.042  # per finger: closure q in [0, stroke], 0 = fully open; closed gap 0
    grip_force: float = 50.0  # grip actuator force limit F (N), acts on the mean-closure tendon
    grip_kp: float = 4000.0
    grip_kv: float = 100.0  # overdamped: unobstructed fingers close at <= ~0.4 m/s
    grip_overdrive: float = 0.01  # closing target beyond full stroke so the force saturates at F
    finger_armature: float = 0.02
    finger_damping: float = 2.0
    xy_kp: float = 2000.0
    xy_kv: float = 80.0
    xy_force: float = 100.0
    z_kp: float = 4000.0
    z_kv: float = 120.0
    z_force: float = 150.0
    yaw_kp: float = 20.0
    yaw_kv: float = 1.0
    yaw_torque: float = 15.0
    slide_armature: float = 0.05
    yaw_armature: float = 0.001
    equality_solref: tuple[float, float] = (0.004, 1.0)


@dataclass(frozen=True)
class ControllerConfig:
    """Open-loop, time-based grasp state machine (identical for every entrant)."""

    # Controller settings k: finger-tip height as a fraction of the reachable band, from the
    # lowest palm-clear tip height (0 for long fingers) up to the object's top.
    grasp_height_fracs: tuple[float, ...] = (0.3, 0.5, 0.7)
    grasp_z_offset: float = 0.0  # extra tip height (m); used by failure tests, 0 in evaluation
    min_tip_clearance: float = 0.003
    # The palm underside (the finger roots) stays at least this far above the object's
    # top at the grasp pose, so short fingers grasp tall objects higher up.
    palm_clearance: float = 0.01
    approach_offset: float = 0.08
    approach_clearance: float = 0.02  # tips start at least this far above the object top
    settle_time: float = 0.3
    descend_time: float = 1.0
    close_time: float = 1.0
    squeeze_time: float = 0.3
    lift_height: float = 0.15
    lift_time: float = 1.5
    open_target: float = -0.002  # grip target while open (presses fingers onto the open stop)
    control_period: float = 0.002  # 500 Hz; must be a multiple of the physics timestep


@dataclass(frozen=True)
class ScoreConfig:
    """Success criterion and physics-check thresholds."""

    success_lift: float = 0.10
    success_hold: float = 2.0
    max_time: float = 12.0
    check_period: float = 0.01  # s between checks and success samples; multiple of control_period
    finger_table_force_max: float = 5.0
    # Palm-object contact while the object is carried (LIFT and HOLD) above this normal
    # force raises ``palm_contact``: only the fingers may hold the object.
    palm_contact_force_max: float = 0.2
    penetration_max: float = 0.002
    speed_cap: float = 5.0
    eject_dist: float = 0.025
    eject_speed: float = 1.0
    off_table_drop: float = 0.05
    grasp_min_squeeze_frac: float = 0.25  # of F/2, per finger, to count as grasped at lift start
    # ... and no finger may push the object down harder than this times its squeeze (1.0:
    # more than 45 deg below the closing axis), i.e. fingers resting on its top are no grasp.
    grasp_max_press_down_ratio: float = 1.0
    fail_settle_time: float = 1.0  # after lift ends, stop once the object stays below success_lift
    record_tail: float = 0.5  # extra simulated seconds kept in recordings after the outcome


@dataclass(frozen=True)
class JitterConfig:
    """Per-episode perturbations drawn from the episode seed."""

    pos: float = 0.002
    yaw_deg: float = 3.0
    mu_frac: float = 0.10


@dataclass(frozen=True)
class RobustnessConfig:
    """Robustness-certificate tests (see :mod:`rtt.sim.certify`); jitter is off in every test.

    * Placement error: the object sits ``e`` mm from where the controller aims, along the
      gripper's x (finger width) or y (closing) axis, for every ``e`` in
      ``0, +-placement_errors_mm``.
    * Shake: after the palm stops at the lift height and ``perturb_delay`` has passed,
      it oscillates sinusoidally along one gripper axis with ``shake_amplitude`` (m) at
      ``shake_frequency`` (Hz) for ``shake_cycles`` cycles; the first and last
      ``shake_ramp_cycles`` cycles ramp the amplitude with a raised-cosine window.
    * Push: at the same moment a horizontal force ``push_impulse / push_duration`` acts on
      the object for ``push_duration`` s along a gripper axis; with ``push_at_top`` it
      acts at the height of the object's top (so it also twists the object in the grasp),
      otherwise through its centre of mass.

    A perturbed test succeeds only if the object is still held (lifted >= ``success_lift``)
    ``post_perturb_hold`` s after the perturbation ends, on top of the usual criterion.
    """

    placement_errors_mm: tuple[float, ...] = (1.0, 2.0, 3.0)
    shake_axes: tuple[str, ...] = ("x", "y", "z")
    shake_amplitude: float = 0.01
    shake_frequency: float = 5.0
    shake_cycles: int = 5
    shake_ramp_cycles: float = 0.5
    push_axes: tuple[str, ...] = ("x", "y")
    push_impulse: float = 0.5
    push_duration: float = 0.02
    push_at_top: bool = True
    perturb_delay: float = 0.2
    post_perturb_hold: float = 1.0


@dataclass(frozen=True)
class ReplayConfig:
    fps: int = 30


@dataclass(frozen=True)
class TaskConfig:
    name: str = "task_v1"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def with_overrides(self, **sections: dict[str, Any]) -> TaskConfig:
        """Return a copy with per-section field overrides, e.g. ``physics={"friction": 0.05}``."""
        changes: dict[str, Any] = {}
        for section, values in sections.items():
            current = getattr(self, section)
            changes[section] = dataclasses.replace(current, **values)
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class RegistrationConfig:
    """The registered-grasp rule of task v2 (``rtt.sim.register``; spec D1).

    Registration is measured ``settle_time`` seconds after the grip ramp ends (the end of
    closing, ``t_close + close_time``), which must be before the lift starts, with the part's
    pose taken in the palm frame:

    * seated: ``seat_short_mm = q_seat - q <= seat_short_max_mm`` (``q``: mean jaw closure);
    * centred: ``|dx| <= dx_max_mm`` (across the jaw), ``|dy| <= dy_max_mm`` (closing axis),
      yaw residual (modulo the part's yaw symmetry) ``<= yaw_max_deg`` and tilt
      ``<= tilt_max_deg``; a part checks only the axes it declares in ``register_axes``;
    * upright after the hold: tilt ``<= hold_tilt_max_deg`` when the hold is confirmed.

    ``creep_delay``: seconds after HOLD starts at which the hold-creep reference pose is
    taken (``hold_creep_mm`` is logged per episode and never scored).
    """

    settle_time: float = 0.25
    seat_short_max_mm: float = 1.0
    dx_max_mm: float = 0.75
    dy_max_mm: float = 0.75
    yaw_max_deg: float = 2.0
    tilt_max_deg: float = 3.0
    hold_tilt_max_deg: float = 5.0
    creep_delay: float = 0.2

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not value >= 0:
                raise ValueError(f"registration.{f.name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class ChecksConfig:
    """Extra physics checks of task v2 (``rtt.sim.checks``).

    * ``bad_qacc``: MuJoCo's bad-acceleration (and bad position / velocity) warning counter
      went up during the scored part of the episode (up to the moment the outcome is decided;
      the replay-only tail is never checked), i.e. MuJoCo silently reset the simulation; the
      episode becomes ``physics_invalid``.
    * ``held_penetration_max_mm``: a held episode (the v1 lift-and-hold criterion met) whose
      deepest jaw-part penetration over the scored part of the episode exceeds this becomes
      ``physics_unreliable`` and fails both the registered and the lift-only criterion
      (founder ruling 4, 2026-10-16: the same rule for every arm and design). ``inf`` turns
      the rule off.
    * ``trajectory_hash``: keep a SHA-256 of the state at every check tick (determinism).
    """

    bad_qacc: bool = True
    held_penetration_max_mm: float = 1.0
    trajectory_hash: bool = True

    def __post_init__(self) -> None:
        v = self.held_penetration_max_mm
        if not isinstance(v, int | float) or isinstance(v, bool) or not v > 0:
            raise ValueError(f"checks.held_penetration_max_mm must be a number > 0, got {v!r}")


def _build[T](cls: type[T], raw: dict[str, Any] | None, where: str) -> T:
    raw = raw or {}
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def task_from_dict(raw: dict[str, Any]) -> TaskConfig:
    """Build a :class:`TaskConfig` from a parsed YAML mapping, rejecting unknown keys."""
    sections = {
        "physics": PhysicsConfig,
        "gripper": GripperConfig,
        "controller": ControllerConfig,
        "score": ScoreConfig,
        "jitter": JitterConfig,
        "robustness": RobustnessConfig,
        "replay": ReplayConfig,
    }
    unknown = set(raw) - set(sections) - {"name"}
    if unknown:
        raise ValueError(f"unknown task sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {k: _build(cls, raw.get(k), k) for k, cls in sections.items()}
    if "name" in raw:
        kwargs["name"] = str(raw["name"])
    return TaskConfig(**kwargs)


@dataclass(frozen=True)
class TaskV2(TaskConfig):
    """Task v2: every :class:`TaskConfig` section plus ``registration`` and ``checks``.

    The controller's grasp-height grid is removed (spec D5): ``grasp_height_fracs`` must be
    empty, so any v1 code path that would pick a controller setting ``k`` fails loudly instead
    of silently using a grid. The registration moment (``settle_time`` after the grip ramp)
    must fall on a control tick inside the squeeze dwell, before the lift.
    """

    name: str = "task_v2"
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)

    def __post_init__(self) -> None:
        c = self.controller
        if c.grasp_height_fracs:
            raise ValueError(
                "task v2 has no controller grasp-height grid (the jaw pair sets the grip height); "
                f"set controller.grasp_height_fracs to [], got {list(c.grasp_height_fracs)}"
            )
        settle = self.registration.settle_time
        if not 0.0 < settle < c.squeeze_time:
            raise ValueError(
                f"registration.settle_time ({settle} s) must be inside the squeeze dwell "
                f"(0, {c.squeeze_time}) so registration is measured before the lift"
            )
        ticks = settle / c.control_period
        if abs(ticks - round(ticks)) > 1e-9:
            raise ValueError(
                f"registration.settle_time ({settle} s) must be a multiple of the control period "
                f"({c.control_period} s)"
            )


_V2_SECTIONS: dict[str, type] = {"registration": RegistrationConfig, "checks": ChecksConfig}


def task_v2_from_dict(raw: dict[str, Any]) -> TaskV2:
    """Build a :class:`TaskV2` from a parsed YAML mapping, rejecting unknown keys."""
    raw = dict(raw)
    extra = {k: _build(cls, raw.pop(k, None), k) for k, cls in _V2_SECTIONS.items()}
    base = task_from_dict(raw)
    values = {f.name: getattr(base, f.name) for f in dataclasses.fields(TaskConfig)}
    if "name" not in raw:
        values["name"] = "task_v2"
    return TaskV2(**values, **extra)


def load_task(path: str | Path | None = None) -> TaskConfig:
    """Load a task YAML (default: ``configs/task_v1.yaml``; built-in defaults if absent)."""
    p = Path(path) if path is not None else DEFAULT_TASK_PATH
    if path is None and not p.exists():
        return TaskConfig()
    with open(p, encoding="utf-8") as fh:
        return task_from_dict(yaml.safe_load(fh) or {})


def load_task_v2(path: str | Path | None = None) -> TaskV2:
    """Load a v2 task YAML (default: ``configs/task_v2.yaml``)."""
    p = Path(path) if path is not None else DEFAULT_TASK_V2_PATH
    with open(p, encoding="utf-8") as fh:
        return task_v2_from_dict(yaml.safe_load(fh) or {})


@functools.lru_cache(maxsize=1)
def default_task_v2() -> TaskV2:
    """The repository's v2 task (``configs/task_v2.yaml``), loaded once per process."""
    return load_task_v2()


@functools.lru_cache(maxsize=1)
def default_task() -> TaskConfig:
    """The repository's default task (``configs/task_v1.yaml``), loaded once per process."""
    return load_task()


def task_digest(task: TaskConfig) -> str:
    """Stable 12-hex-digit fingerprint of every value in ``task`` (for result provenance)."""
    payload = json.dumps(dataclasses.asdict(task), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def gripper_task_for(task: TaskConfig, **overrides: Any) -> GripperTask:
    """The design validator's :class:`~rtt.design.schema.GripperTask` for this simulator task.

    The finger stroke and the root gap at full close are taken from ``task.gripper``
    (metres, converted to millimetres), so the validator mounts fingers exactly where
    the simulator does. ``overrides`` set the validator-only fields such as
    ``widest_training_object_mm``, ``mass_budget_g`` or ``envelope``.
    """
    g = task.gripper
    derived = {
        "stroke_mm": round(g.stroke * 1000.0, 9),
        "closed_gap_mm": round(2.0 * (g.open_half_gap - g.stroke) * 1000.0, 9),
    }
    clash = sorted(derived.keys() & overrides.keys())
    if clash:
        raise ValueError(f"{clash} come from the simulator task; change task.gripper instead")
    return GripperTask(**derived, **overrides)
