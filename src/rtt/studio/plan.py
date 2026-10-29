"""The primitive plan protocol: per-primitive controllers and success predicates (``rtt-effectors/2``).

A step runs as: :meth:`PrimitiveController.plan` once at the step's start (it locates objects
through :meth:`StepContext.observe_site`, so the condition's placement error stays hidden), then
:meth:`PrimitiveController.command` every control tick, which returns targets for the six wrist
servos (x, y, z, yaw, roll, pitch) and controls for the active head's actuators
(:class:`PrimitiveCommand`). A controller may return the first five targets only: the runner then
holds the pitch where it was when the step began (:attr:`StepContext.wrist_pitch_start`). The
:class:`SuccessPredicate` observes the true state at every check
tick and at the step's end (:meth:`SuccessPredicate.observe`, accumulating peaks and last values)
and gives the :class:`StepVerdict` when the step ends. A predicate that also defines
``observe_tick(t, ctx, model, data, acc)`` is called after every control tick, which is every
physics step at the task's 2 ms timestep: for what moves too fast for the check ticks, such as a
lid meeting its stop. A verdict may carry plain caveats for the step
(:attr:`StepVerdict.caveats`), which the step's outcome shows with its own. The runner
(:mod:`rtt.studio.episode`) owns
the physics, the clock, the trajectory hash and the contract checks:

* a controller acts only through its returned targets and controls (never ``qfrc_applied`` or
  ``xfrc_applied``), every value finite, one effector control per active actuator;
* the roll target may not move faster than ``STUDIO_WRIST.roll_speed_max_deg_s``;
* the pitch target may not move faster than ``STUDIO_WRIST.pitch_speed_dps`` nor leave
  +-``STUDIO_WRIST.pitch_range_deg``;
* the idle head's actuators are held at their open controls, and the objects' passive actuators
  are driven by their template builders.

Registry: track S2's modules ``rtt.studio.primitives.<primitive>`` call :func:`register_primitive`
on import, and :func:`get_primitive` imports them on first use (a spawned worker finds them). A
primitive without a module is ``not_simulated`` (:class:`PrimitiveNotSimulated`), never a pass.
The trivial S1 reference controllers (:mod:`rtt.studio.reference`) form a separate set,
``controllers="reference"``, that only tests and smoke runs select; their results say so.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

import numpy as np

from rtt.effectors.types import FrozenDict
from rtt.studio.scene import wrist_rotation
from rtt.studio.types import MAX_STEP_S, STEP_FAILURE_MODES, PrimitiveKind, Step

if TYPE_CHECKING:
    import mujoco

    from rtt.sim.task import TaskV2
    from rtt.studio.scene import HeadBuild, StudioScene
    from rtt.studio.templates import ResolvedHandles, TemplateBuilder
    from rtt.studio.types import TaskSpec

__all__ = [
    "HELD_PENETRATION_MAX_MM",
    "PENETRATION_TRANSIENT_MS",
    "ControllerSet",
    "PrimitiveCommand",
    "PrimitiveController",
    "PrimitiveNotSimulated",
    "StepContext",
    "StepPlan",
    "StepVerdict",
    "SuccessPredicate",
    "V1SceneView",
    "contact_summary",
    "criteria_verdict",
    "get_primitive",
    "override_primitive",
    "penetration_unreliable",
    "register_primitive",
]

ControllerSet = Literal["default", "reference"]
#: Penetration above this (mm) in any contact pair but a designed one (the runner measures them:
#: :class:`rtt.studio.episode.PenetrationMonitor`) is a physics-validity concern (the v1 limit).
HELD_PENETRATION_MAX_MM: Final = 1.0
#: ms: penetration above :data:`HELD_PENETRATION_MAX_MM` for no longer than this is an impact
#: transient (a part landing on a floor): flagged in the step's quantities, not failed. Longer, the
#: result is ``physics_unreliable``. The time is the longest unbroken run of control ticks one
#: contact pair stayed above the limit, times the control period.
PENETRATION_TRANSIENT_MS: Final = 10.0


class PrimitiveNotSimulated(LookupError):
    """No controller is registered for this primitive: the step is ``not_simulated``."""


@dataclass(frozen=True, eq=False)
class StepContext:
    """Everything a controller and a predicate may read about the step they run.

    * ``obj``/``target``: the step's object (and destination) resolved in the compiled scene.
    * ``head``: the active head (its family build, prefix ``""``, TCP and axis in the wrist frame).
    * ``wrist_start``: the first five wrist servo targets when the step starts (x, y, z in m, yaw
      and roll in rad); ``wrist_pitch_start``: the sixth, the pitch (rad, 0 = tool straight down);
      ``t0``: the task clock then (s).
    * ``aim_offsets``: per object, the condition's placement error (world m). Controllers locate
      objects only through :meth:`observe_site`, which includes it, so the error stays unknown to
      them (the v1 rule); predicates read the true state.
    """

    step: Step
    task_spec: TaskSpec
    scene: StudioScene
    obj: ResolvedHandles
    target: ResolvedHandles | None
    head: HeadBuild
    wrist_start: np.ndarray
    task: TaskV2
    seed: int
    t0: float = 0.0
    aim_offsets: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    wrist_pitch_start: float = 0.0

    def observe_site(self, data: mujoco.MjData, handle: str, obj: str | None = None) -> np.ndarray:
        """Where the controller believes a handle is: the true site plus the object's aim offset."""
        name = obj or self.step.object
        handles = self.scene.objects[name]
        return handles.site_pos(data, handle) + np.asarray(self.aim_offsets.get(name, (0.0, 0.0, 0.0)))

    @property
    def builder(self) -> TemplateBuilder:
        return self.scene.builders[self.step.object]

    def object_state(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return self.builder.state(self.obj, model, data)

    def tcp_offset(self, yaw: float, pitch: float = 0.0) -> np.ndarray:
        """The active TCP relative to the wrist origin, world frame, for wrist ``yaw`` and
        ``pitch`` (roll 0)."""
        return wrist_rotation(yaw, 0.0, pitch) @ np.asarray(self.head.tcp_wrist, dtype=np.float64)


@dataclass(frozen=True)
class StepPlan:
    """A controller's plan for one step: its duration and its own private values."""

    duration_s: float
    values: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.duration_s) and 0.0 < self.duration_s <= MAX_STEP_S):
            raise ValueError(f"a step lasts (0, {MAX_STEP_S}] s, got {self.duration_s!r}")
        object.__setattr__(self, "values", FrozenDict(dict(self.values)))


@dataclass(frozen=True)
class PrimitiveCommand:
    """Targets of the six wrist servos and controls of the active head's actuators.

    ``wrist`` may hold the first five targets only; the runner then holds the pitch where it was
    when the step began.
    """

    wrist: np.ndarray  # (6,): x, y, z (m, wrist origin), yaw, roll (rad, continuous), pitch (rad); or (5,)
    effector: np.ndarray  # (n_active,)


@runtime_checkable
class PrimitiveController(Protocol):
    """A scripted controller for one primitive (module docstring)."""

    primitive: PrimitiveKind
    name: str

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan: ...

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        """Targets at ``t`` s after the step started."""
        ...


@dataclass(frozen=True)
class StepVerdict:
    """A predicate's verdict: success, the failure mode (``None`` on success), what it measured,
    and plain caveats on what happened (shown on the step's outcome; never a failure by itself)."""

    success: bool
    failure_mode: str | None
    quantities: Mapping[str, float]
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.failure_mode is not None and self.failure_mode not in STEP_FAILURE_MODES:
            raise ValueError(f"unknown step failure mode {self.failure_mode!r}")
        if self.success == (self.failure_mode is not None):
            raise ValueError("a verdict succeeds exactly when it has no failure mode")
        object.__setattr__(self, "quantities", FrozenDict({k: float(v) for k, v in self.quantities.items()}))
        object.__setattr__(self, "caveats", tuple(str(c) for c in self.caveats))


@runtime_checkable
class SuccessPredicate(Protocol):
    """A primitive's success test (module docstring). ``acc`` starts empty for every step."""

    primitive: PrimitiveKind
    name: str

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None: ...

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict: ...


def criteria_verdict(step: Step, measured: Mapping[str, float]) -> StepVerdict:
    """The shared rule every predicate ends with.

    In order: a penetration above :data:`HELD_PENETRATION_MAX_MM` that lasted more than
    :data:`PENETRATION_TRANSIENT_MS` is ``physics_unreliable`` (:func:`penetration_unreliable`); a
    peak ``force_n`` or ``torque_nm`` above the step's limit is ``constraint_exceeded``; a success
    criterion that fails, or whose quantity was not measured, is ``criterion_not_met``. A bound is
    met within its quantity's tolerance (:data:`~rtt.studio.types.CRITERION_TOLERANCE`).
    """
    q = {k: float(v) for k, v in measured.items()}
    if penetration_unreliable(q):
        return StepVerdict(False, "physics_unreliable", q)
    c = step.constraints
    if c.max_force_n is not None and q.get("force_n", 0.0) > c.max_force_n:
        return StepVerdict(False, "constraint_exceeded", q)
    if c.max_torque_nm is not None and q.get("torque_nm", 0.0) > c.max_torque_nm:
        return StepVerdict(False, "constraint_exceeded", q)
    if not all(crit.passes(q.get(crit.quantity)) for crit in step.success):
        return StepVerdict(False, "criterion_not_met", q)
    return StepVerdict(True, None, q)


def penetration_unreliable(q: Mapping[str, float]) -> bool:
    """Whether the measured penetration makes a step ``physics_unreliable``.

    The runner measures ``penetration_mm`` (the peak over every contact pair but the designed
    ones) and ``penetration_ms`` (how long one pair stayed above the limit). A peak at or under
    :data:`HELD_PENETRATION_MAX_MM` is fine. Above it, an excursion of
    :data:`PENETRATION_TRANSIENT_MS` or less is an impact transient (the runner flags it as
    ``penetration_transient`` and the step is not failed); a longer one is unreliable. Without a
    ``penetration_ms`` (no runner measured it) the peak alone decides, the v1 rule. A NaN peak is
    unreliable.
    """
    peak = float(q.get("penetration_mm", 0.0))
    if math.isnan(peak):
        return True
    if peak <= HELD_PENETRATION_MAX_MM:
        return False
    ms = q.get("penetration_ms")
    return ms is None or not float(ms) <= PENETRATION_TRANSIENT_MS


def contact_summary(
    model: mujoco.MjModel, data: mujoco.MjData, tool_geoms: np.ndarray, object_geoms: np.ndarray
) -> tuple[float, float, int]:
    """Total normal force (N), deepest penetration (m) and count of the active contacts between
    ``tool_geoms`` and ``object_geoms`` (call at check ticks; it uses ``mj_contactForce``)."""
    import mujoco

    if not data.ncon or not len(tool_geoms) or not len(object_geoms):
        return 0.0, 0.0, 0
    is_tool = np.zeros(model.ngeom, dtype=bool)
    is_tool[tool_geoms] = True
    is_obj = np.zeros(model.ngeom, dtype=bool)
    is_obj[object_geoms] = True
    force, depth, n = 0.0, 0.0, 0
    wrench = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        g0, g1 = int(con.geom[0]), int(con.geom[1])
        if con.efc_address < 0 or not ((is_tool[g0] and is_obj[g1]) or (is_obj[g0] and is_tool[g1])):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        force += float(wrench[0])
        depth = max(depth, float(-con.dist))
        n += 1
    return force, depth, n


# --------------------------------------------------------------------------- v1 primitives in the Studio


@dataclass(frozen=True)
class _V1Ids:
    palm_body: int
    object_body: int
    effector_bodies: np.ndarray
    effector_geoms: np.ndarray
    object_geoms: np.ndarray
    effector_actuators: np.ndarray


@dataclass(frozen=True, eq=False)
class V1SceneView:
    """The ``scene`` argument a v1 :class:`~rtt.effectors.types.GraspPrimitive` or
    :class:`~rtt.effectors.types.GraspCriterion` reads, for the active head and the step's object.

    The active head keeps its family's element names, so the family's primitive and criterion
    run unchanged (track S2 adapts grasp, suction and wrap this way).
    """

    model: Any
    ids: _V1Ids
    build: Any

    @classmethod
    def of(cls, ctx: StepContext) -> V1SceneView:
        s = ctx.scene
        return cls(
            model=s.model,
            ids=_V1Ids(
                palm_body=s.ids.palm_body,
                object_body=ctx.obj.body_ids[0] if ctx.obj.body_ids else 0,
                effector_bodies=s.ids.active_bodies,
                effector_geoms=s.ids.active_geoms,
                object_geoms=ctx.obj.geom_ids,
                effector_actuators=s.ids.effector_actuators,
            ),
            build=ctx.head.build,
        )


# --------------------------------------------------------------------------- registry

_SETS: dict[str, dict[PrimitiveKind, tuple[PrimitiveController, SuccessPredicate]]] = {
    "default": {},
    "reference": {},
}
_LOCK = threading.RLock()
_MODULES: Final[Mapping[str, str]] = {
    "default": "rtt.studio.primitives.{}",
    "reference": "rtt.studio.reference",
}


def register_primitive(
    controller: PrimitiveController,
    predicate: SuccessPredicate,
    *,
    controllers: ControllerSet = "default",
    replace: bool = False,
) -> None:
    """Register the controller and predicate of ``controller.primitive`` in a controller set."""
    if not isinstance(controller, PrimitiveController) or not isinstance(predicate, SuccessPredicate):
        raise TypeError("register_primitive takes a PrimitiveController and a SuccessPredicate")
    kind = PrimitiveKind(controller.primitive)
    if PrimitiveKind(predicate.primitive) is not kind:
        raise ValueError("the controller and the predicate are for different primitives")
    with _LOCK:
        table = _SETS[controllers]
        if kind in table and not replace and table[kind] != (controller, predicate):
            raise ValueError(f"{kind.value!r} is already registered in the {controllers} set")
        table[kind] = (controller, predicate)


def get_primitive(
    primitive: PrimitiveKind | str, controllers: ControllerSet = "default"
) -> tuple[PrimitiveController, SuccessPredicate]:
    """The controller and predicate of ``primitive`` (importing its module on first use)."""
    kind = PrimitiveKind(primitive)
    if controllers not in _SETS:
        raise ValueError(f"controllers must be one of {sorted(_SETS)}")
    with _LOCK:
        if kind in _SETS[controllers]:
            return _SETS[controllers][kind]
    module = _MODULES[controllers].format(kind.value)
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in (module, module.rsplit(".", 1)[0]):
            raise
        raise PrimitiveNotSimulated(f"{kind.value} is not simulated yet (no module {module})") from None
    with _LOCK:
        if kind not in _SETS[controllers]:
            raise PrimitiveNotSimulated(f"importing {module} did not register {kind.value}")
        return _SETS[controllers][kind]


@contextlib.contextmanager
def override_primitive(
    controller: PrimitiveController, predicate: SuccessPredicate, *, controllers: ControllerSet = "default"
) -> Iterator[None]:
    """Temporarily register a controller and predicate (tests), restoring the previous ones."""
    kind = PrimitiveKind(controller.primitive)
    with _LOCK:
        previous = _SETS[controllers].get(kind)
        _SETS[controllers][kind] = (controller, predicate)
    try:
        yield
    finally:
        with _LOCK:
            if previous is None:
                _SETS[controllers].pop(kind, None)
            else:
                _SETS[controllers][kind] = previous
