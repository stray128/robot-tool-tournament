"""Task configuration: physics, gripper, controller and scoring constants.

Every entrant is evaluated under one :class:`TaskConfig`. The defaults below
mirror ``configs/task_v1.yaml`` exactly (a test keeps them in sync); load the
YAML with :func:`load_task` so that tuned values live in one reviewed file.

Units are SI throughout (metres, kilograms, seconds, newtons, radians) unless
a field name says otherwise (``*_deg``).
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rtt.design.schema import GripperTask

DEFAULT_TASK_PATH = Path(__file__).resolve().parents[3] / "configs" / "task_v1.yaml"


@dataclass(frozen=True)
class PhysicsConfig:
    """MuJoCo solver and contact settings shared by every episode."""

    timestep: float = 0.001
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 10.0
    condim: int = 4
    friction: float = 0.8
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001
    contact_solref: tuple[float, float] = (0.004, 1.0)
    contact_solimp: tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    limit_solref: tuple[float, float] = (0.004, 1.0)  # joint-limit stiffness (finger stops)
    iterations: int = 100
    ls_iterations: int = 50
    native_ccd: bool = False  # False: libccd MPR for convex pairs (cylinder-box); see task_v1.yaml
    multi_ccd: bool = True  # multiple contact points per convex pair
    object_density: float = 700.0
    table_top_z: float = 0.0
    table_half_size: tuple[float, float, float] = (0.4, 0.4, 0.375)


@dataclass(frozen=True)
class GripperConfig:
    """Floating palm and parallel-jaw finger drive (identical for every entrant)."""

    palm_half_size: tuple[float, float, float] = (0.03, 0.065, 0.015)
    palm_mass: float = 0.4
    # Lateral offset of each finger root from the palm centre when fully open.
    # A finger whose inner face sits at local y = 0 leaves a 2 * 0.04 = 80 mm gap.
    open_half_gap: float = 0.04
    stroke: float = 0.04  # per finger: closure q in [0, stroke], 0 = fully open
    grip_force: float = 40.0  # grip actuator force limit F (N), acts on the mean-closure tendon
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

    # Controller settings k: finger-tip height above the table as a fraction of object height.
    grasp_height_fracs: tuple[float, ...] = (0.3, 0.5, 0.7)
    grasp_z_offset: float = 0.0  # extra tip height (m); used by failure tests, 0 in evaluation
    min_tip_clearance: float = 0.003
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
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    def with_overrides(self, **sections: dict[str, Any]) -> TaskConfig:
        """Return a copy with per-section field overrides, e.g. ``physics={"friction": 0.05}``."""
        changes: dict[str, Any] = {}
        for section, values in sections.items():
            current = getattr(self, section)
            changes[section] = dataclasses.replace(current, **values)
        return dataclasses.replace(self, **changes)


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
        "replay": ReplayConfig,
    }
    unknown = set(raw) - set(sections) - {"name"}
    if unknown:
        raise ValueError(f"unknown task sections: {sorted(unknown)}")
    kwargs: dict[str, Any] = {k: _build(cls, raw.get(k), k) for k, cls in sections.items()}
    if "name" in raw:
        kwargs["name"] = str(raw["name"])
    return TaskConfig(**kwargs)


def load_task(path: str | Path | None = None) -> TaskConfig:
    """Load a task YAML (default: ``configs/task_v1.yaml``; built-in defaults if absent)."""
    p = Path(path) if path is not None else DEFAULT_TASK_PATH
    if path is None and not p.exists():
        return TaskConfig()
    with open(p, encoding="utf-8") as fh:
        return task_from_dict(yaml.safe_load(fh) or {})


@functools.lru_cache(maxsize=1)
def default_task() -> TaskConfig:
    """The repository's default task (``configs/task_v1.yaml``), loaded once per process."""
    return load_task()


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
