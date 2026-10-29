"""Full-task episodes: segments, carried world state and the tool-change hook (``rtt-effectors/2``).

**How a task runs.** :meth:`Setup.segments <rtt.studio.types.Setup.segments>` splits the steps
into segments: consecutive steps with the same active tool. Each segment is one continuous
episode in its own compiled scene (:func:`rtt.studio.scene.build_studio_scene` for that tool);
inside a segment the steps run back to back on one ``MjData``, so a part grasped in one step is
still in the tool for the next (``Step.hold_after``).

**The tool-change hook (the simplest correct model).** Between segments nothing is held
(:meth:`Setup.problems` refuses a change while holding), so a change is modelled as the end of
one episode and the start of the next with the **world state carried over**
(:class:`WorldState`: every object joint's position, by name, at rest) and the **task clock
advanced** by the change time: ``ToolChanger.changeover_s`` for a tool change, ``present_s`` for a
multi-tool head presenting its other tool. The new scene contains the new tool (and, for a
tool change, only it: the mass swap) and the wrist starts at its home pose (:func:`home_pose`): over
the table centre, roll and pitch 0, :data:`HOME_HEIGHT` up or high enough to clear every object. Swapping a MuJoCo subtree inside one running episode was
rejected: MjSpec recompilation mid-episode would reset the contact warm-start and actuator state
anyway, and carrying object joints by name is exact for resting objects.

**Conditions.** A :class:`StudioCondition` is a condition id, a seed and optional per-object aim
offsets. The seed draws the v1 jitter (:func:`rtt.sim.episode.sample_jitter`): every object
geom's nominal friction (the task's 0.3 unless a template sets its own) is scaled by its
``mu_scale`` (+-10 %) in every segment, and its ``dx``/``dy``
(+-0.5 mm) is added to every object's aim offset. Controllers see objects through
:meth:`StepContext.observe_site`, which adds the object's aim offset to the true handle position:
the placement error stays unknown to the controller, as in v1, also for objects an earlier step
moved. Identical conditions and seeds for every candidate are asserted by
:mod:`rtt.studio.fairness`.

**Scoring.** Each step ends with its predicate's :class:`~rtt.studio.plan.StepVerdict`; a MuJoCo
state reset makes it ``physics_invalid``. The task stops at the first failed step (the rest are
``not_reached``). A step whose primitive has no controller, or whose object has no physics,
is ``not_simulated`` and never passes.

**Penetration.** The runner, not the predicates, measures physics validity: at every control tick
a :class:`PenetrationMonitor` reads every active contact except the designed ones
(:func:`designed_pairs`: a tool's own links, a template's spring-loaded pads), so a part driven
into a floor, a wall, a fixture or another part counts as much as the tool pressing on its object.
It writes ``penetration_mm`` (the peak), ``penetration_ms`` (the longest unbroken time one body
pair stayed above 1 mm) and ``penetration_transient`` (1 when it went above 1 mm for
:data:`~rtt.studio.plan.PENETRATION_TRANSIENT_MS` or less: flagged, not failed) into the step's
quantities before the predicate's verdict, whose shared rule
(:func:`~rtt.studio.plan.criteria_verdict`) makes a longer excursion ``physics_unreliable``.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import math
import multiprocessing as mp
import re
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import mujoco
import numpy as np

from rtt.effectors.registry import get_family
from rtt.effectors.types import ContractError, FrozenDict, InvalidEffector
from rtt.sim.checks import bad_state_count
from rtt.sim.episode import sample_jitter
from rtt.sim.task import TaskV2, default_task_v2, task_digest
from rtt.studio.plan import (
    HELD_PENETRATION_MAX_MM,
    PENETRATION_TRANSIENT_MS,
    ControllerSet,
    PrimitiveNotSimulated,
    StepContext,
    get_primitive,
)
from rtt.studio.scene import StudioScene, build_studio_scene
from rtt.studio.templates import TemplateUnavailable
from rtt.studio.types import (
    ChangeKind,
    FidelityTier,
    Setup,
    StepOutcome,
    TaskResult,
    TaskSpec,
    TemplateKind,
    tier_of_effector,
    worst_tier,
)

if TYPE_CHECKING:
    from rtt.studio.types import Step

__all__ = [
    "HOME_CLEARANCE",
    "HOME_HEIGHT",
    "PenetrationMonitor",
    "StudioCondition",
    "StudioJob",
    "WorldState",
    "apply_world_state",
    "capture_world_state",
    "designed_pairs",
    "home_pose",
    "run_task",
    "run_task_jobs",
]

#: m above the table top: the lowest wrist position (at the table centre, yaw, roll and pitch 0) at
#: the start of every segment. :func:`home_pose` raises it when the tool stack would reach an object.
HOME_HEIGHT: Final = 0.30
#: m [assumed]: the gap between the tool stack's lowest point and the tallest object at home.
HOME_CLEARANCE: Final = 0.03
_CORNERS: Final = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _z_range(model: mujoco.MjModel, data: mujoco.MjData, geoms: np.ndarray) -> tuple[float, float]:
    """The lowest and highest world z of ``geoms``' bounding boxes."""
    lo, hi = math.inf, -math.inf
    for g in geoms:
        box = model.geom_aabb[g]
        pts = data.geom_xpos[g] + (data.geom_xmat[g].reshape(3, 3) @ (box[:3] + _CORNERS * box[3:]).T).T
        lo, hi = min(lo, float(pts[:, 2].min())), max(hi, float(pts[:, 2].max()))
    return lo, hi


def home_pose(scene: StudioScene, task: TaskV2, state: WorldState | None = None) -> np.ndarray:
    """The wrist's start pose for a segment of ``scene`` (x, y, z, yaw; roll and pitch are 0).

    Over the table centre, :data:`HOME_HEIGHT` above the table, or higher when needed so the whole
    tool stack (every head, the bracket, the changer) clears the tallest object, in the world
    ``state`` the segment starts from, by :data:`HOME_CLEARANCE`. A robot's home is clear of the
    work; at a fixed height a tall object (a cabinet door, an open lid) at the table centre would
    start inside the tool (CR-S2b-2).
    """
    model = scene.model
    data = mujoco.MjData(model)
    base = float(task.physics.table_top_z) + HOME_HEIGHT
    data.qpos[scene.ids.palm_qpos] = [0.0, 0.0, base, 0.0]
    data.qpos[scene.ids.roll_qpos] = 0.0
    data.qpos[scene.ids.pitch_qpos] = 0.0
    if state is not None:
        apply_world_state(scene, data, state)
    mujoco.mj_forward(model, data)
    root = model.body_rootid[model.geom_bodyid]
    wrist = model.body_rootid[scene.ids.palm_body]
    tool = np.flatnonzero(root == wrist)
    objects = np.flatnonzero((root != wrist) & (model.geom_bodyid != 0))
    z = base
    if len(tool) and len(objects):
        tool_lo, _ = _z_range(model, data, tool)
        _, top = _z_range(model, data, objects)
        z = max(base, top + (base - tool_lo) + HOME_CLEARANCE)
    return np.array([0.0, 0.0, z, 0.0])


# --------------------------------------------------------------------------- penetration

_PAD_GEOM: Final = re.compile(r"pad\d+$")
_DESIGNED: weakref.WeakKeyDictionary[StudioScene, np.ndarray] = weakref.WeakKeyDictionary()


def designed_pairs(scene: StudioScene) -> np.ndarray:
    """``(ngeom, ngeom)`` booleans: the geom pairs that touch by design, so their penetration is not
    counted (:class:`PenetrationMonitor`). What the mask covers, and why, is ``docs/STUDIO.md``
    section 3.3 (Penetration):

    * a tool's own links: both geoms on the wrist's kinematic tree (every head, the bracket, the
      adapters, the changer), such as a closing linkage's links against each other;
    * a template's own spring-loaded pads: a ``port_plug`` pad against any other geom of its own
      instance (the plug it grips). The pad's contact is the insertion friction, and its stop sits
      inside the socket wall by construction (that pair is excluded from collision).

    Every other pair counts: tool and object, object and object, object and the table or a wall
    or a fixture, and the lid and the box of one template.
    """
    cached = _DESIGNED.get(scene)
    if cached is not None:
        return cached
    m = scene.model
    on_tool = m.body_rootid[m.geom_bodyid] == m.body_rootid[scene.ids.palm_body]
    mask = np.outer(on_tool, on_tool)
    for obj in scene.objects.values():
        if obj.kind is not TemplateKind.PORT_PLUG:
            continue
        pads = obj.geom_ids[[i for i, g in enumerate(obj.handles.geoms) if _PAD_GEOM.search(g)]]
        mask[np.ix_(pads, obj.geom_ids)] = True
        mask[np.ix_(obj.geom_ids, pads)] = True
    mask.flags.writeable = False  # one shared mask per scene: nobody edits it
    _DESIGNED[scene] = mask
    return mask


class PenetrationMonitor:
    """The deepest penetration over every contact pair that is not by design, and how long it lasted.

    Call :meth:`observe` once per control tick, after the physics step. It reads every active
    contact (``efc_address >= 0``) that overlaps, skips the pairs ``designed`` marks
    (:func:`designed_pairs`; ``None`` marks none), and keeps the peak depth. For the duration it
    tracks each body pair separately: the number of control ticks in a row the pair stayed deeper
    than ``limit_mm``; :attr:`longest` is the longest such run, so two different parts landing one
    after the other are two transients, not one long one. A tick is one control period, and a run
    of ``k`` ticks lasts ``k`` control periods (10 ms is 5 ticks at 2 ms).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        designed: np.ndarray | None = None,
        limit_mm: float = HELD_PENETRATION_MAX_MM,
    ) -> None:
        self._body = np.asarray(model.geom_bodyid)
        self._designed = designed
        self._limit = limit_mm / 1000.0
        self._run: dict[tuple[int, int], int] = {}
        self.peak_m = 0.0
        self.longest = 0

    def observe(self, data: mujoco.MjData) -> None:
        n = int(data.ncon)
        if not n:
            self._run.clear()
            return
        con = data.contact
        depth = -con.dist[:n]
        idx = np.flatnonzero((depth > 0.0) & (con.efc_address[:n] >= 0))
        if len(idx) and self._designed is not None:
            geom = con.geom[:n][idx]
            idx = idx[~self._designed[geom[:, 0], geom[:, 1]]]
        if not len(idx):
            self._run.clear()
            return
        depth = depth[idx]
        self.peak_m = max(self.peak_m, float(depth.max()))
        deep = idx[depth > self._limit]
        if not len(deep):
            self._run.clear()
            return
        pairs = np.sort(self._body[con.geom[:n][deep]], axis=1)
        self._run = {(int(a), int(b)): self._run.get((int(a), int(b)), 0) + 1 for a, b in pairs}
        self.longest = max(self.longest, max(self._run.values()))

    def quantities(self, control_period_s: float) -> dict[str, float]:
        """``penetration_mm`` (peak), ``penetration_ms`` (the longest run above the limit) and
        ``penetration_transient`` (1 when it was above the limit for at most
        :data:`~rtt.studio.plan.PENETRATION_TRANSIENT_MS`)."""
        ms = round(1000.0 * control_period_s * self.longest, 6)
        return {
            "penetration_mm": 1000.0 * self.peak_m,
            "penetration_ms": ms,
            "penetration_transient": float(0.0 < ms <= PENETRATION_TRANSIENT_MS),
        }

    def merge(self, acc: dict[str, float], control_period_s: float) -> None:
        """Write :meth:`quantities` into a step's ``acc``; ``penetration_mm`` keeps the larger of
        its own and what a predicate observed (a predicate sees a subset of these pairs)."""
        q = self.quantities(control_period_s)
        q["penetration_mm"] = max(q["penetration_mm"], acc.get("penetration_mm", 0.0))
        acc.update(q)


_APPROX_EFFECTOR = "This effector family is a labelled approximation (docs/EFFECTORS.md section 10)."


@dataclass(frozen=True)
class StudioCondition:
    """One condition of a comparison: an id, a seed and per-object aim offsets (mm, world x, y)."""

    condition_id: str
    seed: int
    aim_offsets_mm: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.condition_id or len(self.condition_id) > 80:
            raise ValueError("condition_id must be a short non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        offsets = {}
        for k, v in dict(self.aim_offsets_mm).items():
            dx, dy = (float(c) for c in v)
            if not (math.isfinite(dx) and math.isfinite(dy) and abs(dx) <= 50 and abs(dy) <= 50):
                raise ValueError(f"aim offset of {k!r} must be finite and within 50 mm")
            offsets[k] = (dx, dy)
        object.__setattr__(self, "aim_offsets_mm", FrozenDict(offsets))


@dataclass(frozen=True)
class WorldState:
    """What carries from one segment to the next: object joint positions by name, and the clock."""

    qpos: Mapping[str, tuple[float, ...]]
    clock_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "qpos", FrozenDict({k: tuple(float(v) for v in q) for k, q in self.qpos.items()})
        )


def _joint_width(model: mujoco.MjModel, j: int) -> int:
    return {0: 7, 1: 4}.get(int(model.jnt_type[j]), 1)


def _object_joints(scene: StudioScene) -> list[tuple[str, int]]:
    out = []
    for obj in scene.objects.values():
        for name in obj.handles.joints.values():
            out.append((name, scene.model.joint(name).id))
    return out


def capture_world_state(scene: StudioScene, data: mujoco.MjData, clock_s: float) -> WorldState:
    """Every object joint's position now (by name) and the task clock."""
    m = scene.model
    qpos = {}
    for name, j in _object_joints(scene):
        adr = int(m.jnt_qposadr[j])
        qpos[name] = tuple(float(v) for v in data.qpos[adr : adr + _joint_width(m, j)])
    return WorldState(qpos, clock_s)


def apply_world_state(scene: StudioScene, data: mujoco.MjData, state: WorldState) -> None:
    """Put the objects where ``state`` says, at rest (missing joints keep their model default)."""
    m = scene.model
    for name, j in _object_joints(scene):
        if name not in state.qpos:
            continue
        adr, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        width = _joint_width(m, j)
        if len(state.qpos[name]) != width:
            raise ContractError(
                f"carried state of {name!r} has {len(state.qpos[name])} values, needs {width}"
            )
        data.qpos[adr : adr + width] = state.qpos[name]
        data.qvel[dof : dof + {7: 6, 4: 3}.get(width, 1)] = 0.0


@dataclass(frozen=True)
class StudioJob:
    """Everything needed to reproduce one full-task run (picklable for worker processes)."""

    setup: Setup
    task_spec: TaskSpec
    condition: StudioCondition
    task: TaskV2 | None = None
    controllers: ControllerSet = "default"


_SCENES: collections.OrderedDict[tuple[str, int, str, str], StudioScene] = collections.OrderedDict()
_SCENE_CACHE_SIZE = 8


def _scene(setup: Setup, spec: TaskSpec, tool: int, task: TaskV2) -> StudioScene:
    key = (setup.setup_id, tool, spec.task_id, task_digest(task))
    scene = _SCENES.get(key)
    if scene is None:
        scene = build_studio_scene(setup, spec, tool, task=task)
        _SCENES[key] = scene
        while len(_SCENES) > _SCENE_CACHE_SIZE:
            _SCENES.popitem(last=False)
    else:
        _SCENES.move_to_end(key)
    return scene


def _outcome(
    step: Step,
    setup: Setup,
    spec: TaskSpec,
    *,
    success: bool = False,
    failure_mode: str | None,
    quantities: Mapping[str, float] | None = None,
    fidelity: FidelityTier | None = None,
    start_s: float = 0.0,
    duration_s: float = 0.0,
    traj_sha: str = "",
    controller: str = "",
    observed: Sequence[str] = (),
) -> StepOutcome:
    tool = setup.assignment[step.id]
    effector = setup.tools[tool]
    tier = fidelity or worst_tier(spec.fidelity(step.id), tier_of_effector(effector.fidelity))
    caveats = list(spec.caveats(step.id))
    if tier_of_effector(effector.fidelity) is FidelityTier.APPROXIMATION:
        caveats.append(str(getattr(get_family(effector.family), "caveat", "") or _APPROX_EFFECTOR))
    if tier is FidelityTier.NOT_SIMULATED:
        caveats.append("Not simulated yet: this step has no physics model, so it is never scored as a pass.")
    caveats.extend(observed)  # what the predicate saw happen (StepVerdict.caveats)
    return StepOutcome(
        step_id=step.id,
        primitive=step.primitive,
        object=step.object,
        tool=tool,
        effector_id=effector.effector_id,
        success=success,
        failure_mode=failure_mode,
        quantities=quantities or {},
        fidelity=tier,
        caveats=tuple(dict.fromkeys(caveats)),
        start_s=start_s,
        duration_s=duration_s,
        traj_sha=traj_sha,
        controller=controller,
    )


def _aim_offsets(scene: StudioScene, cond: StudioCondition, task: TaskV2) -> dict[str, np.ndarray]:
    jit = sample_jitter(cond.seed, task.jitter)
    out = {}
    for name in scene.objects:
        dx, dy = cond.aim_offsets_mm.get(name, (0.0, 0.0))
        out[name] = np.array([dx / 1000.0 + jit.dx, dy / 1000.0 + jit.dy, 0.0])
    return out


def run_task(job: StudioJob) -> TaskResult:
    """Run every segment of ``job`` (module docstring) and score it."""
    from rtt.studio.fairness import fairness_sha

    wall0 = time.perf_counter()
    task = job.task or default_task_v2()
    setup, spec, cond = job.setup, job.task_spec, job.condition
    segments = setup.segments(spec)
    clock = 0.0
    outcomes: list[StepOutcome] = []
    seg_shas: list[str] = []
    masses: list[float] = []
    change_time = 0.0
    n_changes = {ChangeKind.TOOL_CHANGE: 0, ChangeKind.PRESENT: 0}
    state: WorldState | None = None
    fsha = ""
    stopped = False
    jit = sample_jitter(cond.seed, task.jitter)
    for seg in segments:
        steps = [spec.step(s) for s in seg.step_ids]
        if stopped:
            outcomes += [_outcome(s, setup, spec, failure_mode="not_reached", start_s=clock) for s in steps]
            continue
        if seg.change is not ChangeKind.NONE:
            clock += seg.change_s
            change_time += seg.change_s
            n_changes[seg.change] += 1
        try:
            scene = _scene(setup, spec, seg.tool, task)
        except InvalidEffector:
            outcomes += [_outcome(s, setup, spec, failure_mode="invalid_build", start_s=clock) for s in steps]
            stopped = True
            continue
        except TemplateUnavailable:
            outcomes += [
                _outcome(
                    s,
                    setup,
                    spec,
                    failure_mode="not_simulated",
                    fidelity=FidelityTier.NOT_SIMULATED,
                    start_s=clock,
                )
                for s in steps
            ]
            stopped = True
            continue
        fsha = fsha or fairness_sha(scene)
        masses.append(scene.tool_mass_kg)
        model = scene.model
        data = mujoco.MjData(model)
        model.geom_friction[scene.ids.object_geoms, 0] = scene.object_friction * jit.mu_scale
        home = home_pose(scene, task, state)
        data.qpos[scene.ids.palm_qpos] = home
        data.qpos[scene.ids.roll_qpos] = 0.0
        data.qpos[scene.ids.pitch_qpos] = 0.0
        if state is not None:
            apply_world_state(scene, data, state)
        mujoco.mj_forward(model, data)
        ctrl = np.zeros(model.nu)
        ctrl[:6] = [*home, 0.0, 0.0]  # x, y, z, yaw, roll, pitch
        for h in scene.heads:
            if not h.active:
                ctrl[[model.actuator(a).id for a in h.actuators]] = h.build.ctrl_open
        ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
        aims = _aim_offsets(scene, cond, task)
        seg_sha = hashlib.sha256()
        for step in steps:
            if stopped:
                outcomes.append(_outcome(step, setup, spec, failure_mode="not_reached", start_s=clock))
                continue
            out, ctrl = _run_step(scene, data, step, ctrl, aims, job, task, clock, seg_sha)
            outcomes.append(out)
            clock += out.duration_s
            stopped = not out.success
        seg_shas.append(seg_sha.hexdigest())
        state = capture_world_state(scene, data, clock)
    fsha = fsha or "unbuilt"  # no scene was built (invalid build or no physics)
    traj = hashlib.sha256("|".join(seg_shas).encode()).hexdigest()
    return TaskResult(
        task_id=spec.task_id,
        setup_id=setup.setup_id,
        setup_kind=setup.kind,
        condition_id=cond.condition_id,
        seed=cond.seed,
        steps=tuple(outcomes),
        cycle_time_s=clock,
        change_time_s=change_time,
        n_tool_changes=n_changes[ChangeKind.TOOL_CHANGE],
        n_presentations=n_changes[ChangeKind.PRESENT],
        tool_mass_kg=tuple(masses),
        traj_sha=traj,
        fairness_sha=fsha,
        controllers=job.controllers,
        wall_time_s=time.perf_counter() - wall0,
    )


def _run_step(
    scene: StudioScene,
    data: mujoco.MjData,
    step: Step,
    ctrl: np.ndarray,
    aims: Mapping[str, np.ndarray],
    job: StudioJob,
    task: TaskV2,
    clock: float,
    seg_sha: Any,
) -> tuple[StepOutcome, np.ndarray]:
    setup, spec = job.setup, job.task_spec
    model = scene.model
    try:
        controller, predicate = get_primitive(step.primitive, job.controllers)
    except PrimitiveNotSimulated:
        return _outcome(
            step,
            setup,
            spec,
            failure_mode="not_simulated",
            fidelity=FidelityTier.NOT_SIMULATED,
            start_s=clock,
        ), ctrl
    ctx = StepContext(
        step=step,
        task_spec=spec,
        scene=scene,
        obj=scene.objects[step.object],
        target=scene.objects[step.target] if step.target else None,
        head=scene.active,
        wrist_start=ctrl[:5].copy(),
        task=task,
        seed=job.condition.seed,
        t0=clock,
        aim_offsets=FrozenDict({k: tuple(map(float, v)) for k, v in aims.items()}),
        wrist_pitch_start=float(ctrl[5]),
    )
    plan = controller.plan(ctx, model, data)
    cp = task.controller.control_period
    steps_per_tick = max(1, round(cp / model.opt.timestep))
    ticks_per_check = max(1, round(task.score.check_period / cp))
    n_ticks = max(1, round(plan.duration_s / cp))
    n_act = len(scene.ids.effector_actuators)
    roll_cap = math.radians(scene.wrist.roll_speed_max_deg_s) * cp * (1.0 + 1e-6) + 1e-9
    pitch_cap = math.radians(scene.wrist.pitch_speed_dps) * cp * (1.0 + 1e-6) + 1e-9
    pitch_lim = math.radians(scene.wrist.pitch_range_deg) + 1e-9
    pitch_hold = float(ctrl[5])  # a five-target controller holds the pitch here
    resets0 = bad_state_count(data)
    acc: dict[str, float] = {}
    monitor = PenetrationMonitor(model, designed_pairs(scene))
    sha = hashlib.sha256()
    reset_at: float | None = None
    observe_tick = getattr(predicate, "observe_tick", None)  # optional (rtt.studio.plan)
    for tick in range(n_ticks):
        t = tick * cp
        cmd = controller.command(t, ctx, plan, model, data)
        wrist = np.asarray(cmd.wrist, dtype=np.float64)
        u = np.asarray(cmd.effector, dtype=np.float64)
        if wrist.shape == (5,):
            wrist = np.append(wrist, pitch_hold)
        if (
            wrist.shape != (6,)
            or u.shape != (n_act,)
            or not (np.all(np.isfinite(wrist)) and np.all(np.isfinite(u)))
        ):
            raise ContractError(
                f"controller {controller.name!r} returned wrist {wrist.shape} and effector {u.shape}; "
                f"needs 6 (or 5) and {n_act} finite values"
            )
        if abs(wrist[4] - ctrl[4]) > roll_cap:
            raise ContractError(
                f"controller {controller.name!r} moved the roll target faster than "
                f"{scene.wrist.roll_speed_max_deg_s:g} deg/s"
            )
        if abs(wrist[5] - ctrl[5]) > pitch_cap:
            raise ContractError(
                f"controller {controller.name!r} moved the pitch target faster than "
                f"{scene.wrist.pitch_speed_dps:g} deg/s"
            )
        if abs(wrist[5]) > pitch_lim:
            raise ContractError(
                f"controller {controller.name!r} moved the pitch target past "
                f"+-{scene.wrist.pitch_range_deg:g} deg"
            )
        ctrl[:6] = wrist
        ctrl[scene.ids.effector_actuators] = u
        for obj in scene.objects.values():
            if len(obj.actuator_ids):
                ctrl[obj.actuator_ids] = scene.builders[obj.name].passive(obj, model, data)
        data.ctrl[:] = ctrl
        if data.qfrc_applied.any() or data.xfrc_applied.any():
            raise ContractError(
                f"controller {controller.name!r} applied forces directly; act through actuators"
            )
        mujoco.mj_step(model, data, nstep=steps_per_tick)
        if bad_state_count(data) > resets0:
            reset_at = (tick + 1) * cp
            break
        monitor.observe(data)
        if observe_tick is not None:
            observe_tick((tick + 1) * cp, ctx, model, data, acc)
        if (tick + 1) % ticks_per_check == 0 or tick == n_ticks - 1:  # check ticks and the step's end
            sha.update(data.qpos.tobytes())
            sha.update(data.qvel.tobytes())
            predicate.observe((tick + 1) * cp, ctx, model, data, acc)
    digest = sha.hexdigest()
    seg_sha.update(digest.encode())
    monitor.merge(acc, cp)
    if reset_at is not None:
        out = _outcome(
            step,
            setup,
            spec,
            failure_mode="physics_invalid",
            quantities=acc,
            start_s=clock,
            duration_s=reset_at,
            traj_sha=digest,
            controller=controller.name,
        )
        return out, ctrl
    v = predicate.verdict(ctx, acc)
    out = _outcome(
        step,
        setup,
        spec,
        success=v.success,
        failure_mode=v.failure_mode,
        quantities=v.quantities,
        start_s=clock,
        duration_s=n_ticks * cp,
        traj_sha=digest,
        controller=controller.name,
        observed=v.caveats,
    )
    return out, ctrl


def run_task_jobs(jobs: Sequence[StudioJob], workers: int | None = 1) -> list[TaskResult]:
    """Run jobs in order, in-process (``workers == 1``) or on a spawned pool; identical results."""
    jobs = list(jobs)
    if workers is None:
        workers = max(1, (mp.cpu_count() or 2) - 1)
    if workers <= 1 or len(jobs) <= 1:
        return [run_task(j) for j in jobs]
    with mp.get_context("spawn").Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(run_task, jobs, chunksize=1)
