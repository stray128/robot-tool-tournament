"""Shared helpers of the Studio primitive controllers and predicates (track S2).

* **Tools.** :func:`tool_of` describes the step's active head: its mechanism (``fingers``,
  ``suction``, ``soft_arm``), a fresh v1 primitive and criterion with a
  :class:`~rtt.studio.plan.V1SceneView` on the step's object, the TCP and the tip in the wrist
  frame, and its footprint near the tip measured in a calibration run
  (:func:`footprint`): the head alone, high above the table, its controls held at a closure for
  0.6 s, the active collision geoms read in the wrist frame. Calibration never touches the
  step's ``MjData`` and is cached per compiled scene.
* **Closures.** Every effector control comes from the head's own v1 primitive,
  ``command(t, phase, closure, ...)`` with the controller's closure profile (0 open, 1 closed),
  so a gripper closes, a cup switches its vacuum only while sealed, and a soft arm pulls its
  cable, exactly as in the v1 layer.
* **Wrist poses.** The palm's yaw and the roll turn about the same vertical axis, so a tool
  point ``p`` (wrist frame, roll 0) is at ``palm + Rz(yaw + roll) p``. Controllers change the
  heading with the yaw and turn objects with the roll (capped at the roll speed limit).
* **Perception.** Objects are located only through :meth:`StepContext.observe_site` (the true
  site plus the object's hidden aim offset); a controller reads its own wrist (proprioception)
  and the tool's contact forces (a wrist force sensor): force feedback within a step is allowed
  through wrist and effector commands only, with the budgets in :data:`BUDGET_S`, equal for
  every candidate.
* **Schedules.** A controller records the moments its predicate needs (engaged, released...)
  in a :class:`Schedule` keyed by the step's context (:func:`schedule`).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.registry import get_family
from rtt.effectors.scene import MOUNT_Z
from rtt.sim.controller import Phase, min_jerk
from rtt.studio.plan import V1SceneView, _V1Ids
from rtt.studio.types import SetupKind, TemplateKind

if TYPE_CHECKING:
    from rtt.studio.plan import StepContext
    from rtt.studio.scene import StudioScene

__all__ = [
    "BUDGET_S",
    "PRESHAPE_CLOSURES",
    "Contact",
    "Guarded",
    "Path",
    "Schedule",
    "Tool",
    "angle_about",
    "contacts",
    "criterion_bounds",
    "footprint",
    "heading_for_axis",
    "observed_state",
    "obstacle_top",
    "panel_angle",
    "panel_normal",
    "rot_z",
    "rotate",
    "schedule",
    "tool_geoms",
    "tool_of",
    "tool_point_world",
    "wrist_target",
]

#: Equal per-step time budgets (s) of the closed-loop phases, the same for every candidate.
BUDGET_S: Final[Mapping[str, float]] = {
    "insert_search": 2.0,  # spiral search at the socket mouth
    "insert_slack": 1.0,  # on top of the nominal insertion time
    "insert_reseat": 2.6,  # jam recovery: let go, press the plug home from its top
    "push_slack": 1.5,  # force-limited pushing may pause
    "twist_slack": 1.0,  # on top of the nominal turn time (x 1.3)
    "hook_slack": 0.5,
    "engage_slack": 0.4,  # compliant descent onto a feature
}
TRANSIT_SPEED: Final = 0.40  # m/s peak of a min-jerk free-space move [assumed]
CALIBRATION_Z: Final = 0.75  # m above the table: the head alone, clear of every object
Mechanism = Literal["fingers", "suction", "soft_arm"]


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- tools


@dataclass(frozen=True)
class Footprint:
    """The active head near its tip (wrist frame, roll 0) at one closure, from calibration."""

    tip_z: float  # lowest point of the active collision geoms
    boxes: np.ndarray  # (n, 4): x_lo, x_hi, y_lo, y_hi of each geom whose bottom is near the tip
    centres: np.ndarray  # (n, 2)
    solids: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 6))
    )  # (m, 6): lo xyz, hi xyz of every geom

    def hook(self, band: float = 0.015) -> tuple[np.ndarray, float, float, float]:
        """The tool's best sideways hook: the horizontal wrist-frame direction ``d`` in which the
        geoms lying wholly within ``band`` of the tip reach furthest beyond the rest of the tool,
        how far they reach beyond it (m; 0 for a straight tool), how far they reach from the axis,
        and the top of that protrusion (wrist z). A soft arm's horizontal arm is such a hook; so
        would be any L-shaped finger."""
        best = (np.array([1.0, 0.0, 0.0]), 0.0, 0.0, self.tip_z)
        if not len(self.solids):
            return best
        low = self.solids[self.solids[:, 5] <= self.tip_z + band]
        body = self.solids[self.solids[:, 5] > self.tip_z + band]
        if not len(low):
            return best

        def reach(boxes: np.ndarray, d: np.ndarray) -> float:
            if not len(boxes):
                return 0.0
            xs = np.where(d[0] >= 0, boxes[:, 3], boxes[:, 0])
            ys = np.where(d[1] >= 0, boxes[:, 4], boxes[:, 1])
            return float(np.max(xs * d[0] + ys * d[1]))

        for k in range(72):
            a = 2.0 * math.pi * k / 72
            d = np.array([math.cos(a), math.sin(a), 0.0])
            far, near = reach(low, d), reach(body, d)
            if far - near > best[1] + 1e-6:
                out = low[
                    (
                        np.where(d[0] >= 0, low[:, 3], low[:, 0]) * d[0]
                        + np.where(d[1] >= 0, low[:, 4], low[:, 1]) * d[1]
                    )
                    > near
                ]
                best = (d, far - near, far, float(out[:, 5].max()) if len(out) else self.tip_z)
        return best

    def aperture(self) -> float | None:
        """The gap between the tip geoms on either side of the wrist axis along the tool's y (the
        closing axis), or ``None`` when the head has no such pair (one central tip, a cup)."""
        if not len(self.boxes):
            return None
        pos = [b[2] for b in self.boxes if 0.5 * (b[2] + b[3]) > 0.002]
        neg = [b[3] for b in self.boxes if 0.5 * (b[2] + b[3]) < -0.002]
        if not pos or not neg:
            return None
        return float(min(pos) - max(neg))

    def front(self, direction: np.ndarray) -> float:
        """How far the whole tool (every active geom) reaches along a horizontal wrist-frame
        ``direction`` from its axis (m)."""
        if not len(self.solids):
            return self.half_extent(direction)
        d = np.asarray(direction[:2], dtype=np.float64)
        d = d / max(1e-12, float(np.linalg.norm(d)))
        xs = np.where(d[0] >= 0, self.solids[:, 3], self.solids[:, 0])
        ys = np.where(d[1] >= 0, self.solids[:, 4], self.solids[:, 1])
        return float(np.max(xs * d[0] + ys * d[1]))

    def grip_centre(self, radius: float) -> float:
        """Where along the tool's x (across the closing axis) a round part ``radius`` m across
        should sit between the fingers: on the axis (0) when the tip geoms on both sides of the
        closing axis cover it there, else at the middle of the x span they share (a head whose
        pads are mounted off its TCP). 0 for a head without such a pair."""
        pos = [b for b in self.boxes if 0.5 * (b[2] + b[3]) > 0.002]
        neg = [b for b in self.boxes if 0.5 * (b[2] + b[3]) < -0.002]
        if not pos or not neg:
            return 0.0
        lo = max(min(b[0] for b in pos), min(b[0] for b in neg))
        hi = min(max(b[1] for b in pos), max(b[1] for b in neg))
        if hi <= lo or lo + radius <= 0.0 <= hi - radius:
            return 0.0
        return float(0.5 * (lo + hi))

    def half_extent(self, direction: np.ndarray) -> float:
        """How far the tip region reaches along a horizontal wrist-frame ``direction``."""
        d = np.asarray(direction[:2], dtype=np.float64)
        d = d / max(1e-12, float(np.linalg.norm(d)))
        corners = [(b[i], b[j]) for b in self.boxes for i in (0, 1) for j in (2, 3)]
        return max((float(c[0] * d[0] + c[1] * d[1]) for c in corners), default=0.0)


_FOOTPRINTS: dict[tuple[Any, ...], Footprint] = {}
PRESHAPE_CLOSURES: Final = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
PRESHAPE_CLEARANCE: Final = 0.006  # m each side of a feature when the fingers are pre-shaped [assumed]
GUARD_FORCE_N: Final = 5.0  # a guarded move stops advancing above this tool-object contact force
NEAR_TIP: Final = 0.030  # m: geoms whose lowest point is this close to the tip form the footprint


def _mechanism(family: str) -> Mechanism:
    if family == "suction":
        return "suction"
    if family == "soft_arm":
        return "soft_arm"
    return "fingers"


def _mount_z(scene: StudioScene) -> float:
    s = scene.setup
    if s.kind is SetupKind.SINGLE:
        return MOUNT_Z
    if s.kind is SetupKind.TOOL_CHANGE:
        assert s.changer is not None
        return -s.changer.stack_mm / 1000.0
    assert s.bracket is not None
    return -s.bracket.adapter_thickness_mm / 1000.0


def _view(ctx: StepContext, body: int | None) -> V1SceneView:
    s = ctx.scene
    obj = ctx.obj
    return V1SceneView(
        model=s.model,
        ids=_V1Ids(
            palm_body=s.ids.palm_body,
            object_body=int(body if body is not None else (obj.body_ids[-1] if obj.body_ids else 0)),
            effector_bodies=s.ids.active_bodies,
            effector_geoms=s.ids.active_geoms,
            object_geoms=obj.geom_ids,
            effector_actuators=s.ids.effector_actuators,
        ),
        build=ctx.head.build,
    )


@dataclass(eq=False)
class Tool:
    """The step's active head, ready to drive (module docstring)."""

    mechanism: Mechanism
    primitive: Any  # a fresh v1 GraspPrimitive of the head's family
    criterion: Any  # a fresh v1 GraspCriterion
    view: V1SceneView
    tcp: np.ndarray  # wrist frame, roll 0
    tip_drop: float  # the lowest point is this far below the TCP along the tool axis (m)
    n_act: int
    scene: StudioScene

    @property
    def tip(self) -> np.ndarray:
        """The tool's lowest point on its axis (wrist frame, roll 0)."""
        return self.tcp - np.array([0.0, 0.0, self.tip_drop])

    def controls(
        self, t: float, closure: float, model: mujoco.MjModel, data: mujoco.MjData, phase: Phase | None = None
    ) -> np.ndarray:
        """The head's controls at ``closure`` through its own v1 primitive."""
        if phase is None:
            phase = Phase.HOLD if closure > 0.0 else Phase.SETTLE
        u = np.asarray(
            self.primitive.command(t, phase, float(closure), model, data, self.view), dtype=np.float64
        )
        return u.reshape(self.n_act)

    def footprint(self, closure: float) -> Footprint:
        return footprint(self.scene, closure)

    def closing_tip(self, pre: float) -> np.ndarray:
        """The tool's tip as it closes from the pre-shape ``pre`` (wrist frame, roll 0): on the
        axis, at the lowest point its geoms reach over the calibrated closures from ``pre`` to 1
        (a linkage finger's tip swings down as it closes), so a tip height planned for it holds
        through the whole close. Fingers only (else :attr:`tip`)."""
        if self.mechanism != "fingers":
            return self.tip
        closures = [c for c in (*PRESHAPE_CLOSURES, 0.8, 1.0) if c >= pre - 1e-9]
        low = min(self.footprint(c).tip_z for c in closures)
        return np.array([self.tip[0], self.tip[1], min(float(self.tip[2]), low)])

    def preshape(self, width: float, clearance: float = PRESHAPE_CLEARANCE) -> float:
        """The largest calibrated closure whose aperture along the closing axis still clears a
        feature ``width`` m across by ``clearance`` m on each side (0: fully open). Fingers only."""
        if self.mechanism != "fingers":
            return 0.0
        best = 0.0
        for c in PRESHAPE_CLOSURES:
            ap = self.footprint(c).aperture()
            if ap is not None and ap >= width + 2.0 * clearance:
                best = c
        return best


def tool_of(ctx: StepContext, body: int | None = None) -> Tool:
    """Describe the step's active head (module docstring); ``body``: the object body the v1
    primitive and criterion should treat as the part (default: the object's last body)."""
    head = ctx.head
    impl = get_family(head.spec.family)
    build = head.build
    view = _view(ctx, body)
    tip_drop = float(build.tcp[2]) - (_mount_z(ctx.scene) - float(build.reach))
    return Tool(
        mechanism=_mechanism(head.spec.family.value),
        primitive=impl.primitive(head.spec, build),
        criterion=impl.criterion(head.spec, build),
        view=view,
        tcp=np.asarray(head.tcp_wrist, dtype=np.float64),
        tip_drop=max(0.0, tip_drop),
        n_act=len(ctx.scene.ids.effector_actuators),
        scene=ctx.scene,
    )


def footprint(scene: StudioScene, closure: float) -> Footprint:
    """Calibrate the active head at ``closure`` (module docstring); cached per scene."""
    key = (*scene.key, id(scene.model), round(float(closure), 3))
    hit = _FOOTPRINTS.get(key)
    if hit is not None:
        return hit
    model = scene.model
    data = mujoco.MjData(model)
    ids = scene.ids
    home = [0.0, 0.0, CALIBRATION_Z, 0.0]
    data.qpos[ids.palm_qpos] = home
    mujoco.mj_forward(model, data)
    impl = get_family(scene.active.spec.family)
    prim = impl.primitive(scene.active.spec, scene.active.build)
    view = V1SceneView(
        model=model,
        ids=_V1Ids(
            palm_body=ids.palm_body,
            object_body=0,
            effector_bodies=ids.active_bodies,
            effector_geoms=ids.active_geoms,
            object_geoms=np.zeros(0, dtype=np.int64),
            effector_actuators=ids.effector_actuators,
        ),
        build=scene.active.build,
    )
    ctrl = np.zeros(model.nu)
    ctrl[:4] = home
    for h in scene.heads:
        if not h.active:
            ctrl[[model.actuator(a).id for a in h.actuators]] = h.build.ctrl_open
    phase = Phase.HOLD if closure > 0 else Phase.SETTLE
    for k in range(300):
        u = np.asarray(prim.command(k * 0.002, phase, float(closure), model, data, view), dtype=np.float64)
        ctrl[ids.effector_actuators] = u
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data, nstep=2)
    palm = data.xpos[ids.palm_body].copy()
    r_palm = data.xmat[ids.palm_body].reshape(3, 3)
    lows, boxes, centres, solids = [], [], [], []
    for g in ids.active_geoms:
        c = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3]
        ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]
        lo = r_palm.T @ (c - ext - palm)
        hi = r_palm.T @ (c + ext - palm)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        lows.append(lo[2])
        solids.append((*lo, *hi))
        boxes.append((lo[0], hi[0], lo[1], hi[1]))
        centres.append((0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])))
    tip = float(min(lows)) if lows else 0.0
    near = [i for i, z in enumerate(lows) if z <= tip + NEAR_TIP]
    fp = Footprint(
        tip_z=tip,
        boxes=np.array([boxes[i] for i in near], dtype=np.float64).reshape(-1, 4),
        centres=np.array([centres[i] for i in near], dtype=np.float64).reshape(-1, 2),
        solids=np.array(solids, dtype=np.float64).reshape(-1, 6),
    )
    _FOOTPRINTS[key] = fp
    while len(_FOOTPRINTS) > 64:
        _FOOTPRINTS.pop(next(iter(_FOOTPRINTS)))
    return fp


# --------------------------------------------------------------------------- wrist poses and paths


def wrist_target(point: np.ndarray, tool_point: np.ndarray, yaw: float, roll: float) -> np.ndarray:
    """The five wrist targets that put ``tool_point`` (wrist frame, roll 0) at world ``point``."""
    palm = np.asarray(point, dtype=np.float64) - rot_z(yaw + roll) @ np.asarray(tool_point, dtype=np.float64)
    return np.array([palm[0], palm[1], palm[2], yaw, roll])


def tool_point_world(scene: StudioScene, data: mujoco.MjData, tool_point: np.ndarray) -> np.ndarray:
    """Where ``tool_point`` (wrist frame, roll 0) is now (proprioception: the wrist's own joints)."""
    q = data.qpos
    x, y, z, yaw = (float(q[a]) for a in scene.ids.palm_qpos)
    roll = float(q[scene.ids.roll_qpos])
    return np.array([x, y, z]) + rot_z(yaw + roll) @ np.asarray(tool_point, dtype=np.float64)


def heading_for_axis(direction: np.ndarray, yaw_offset: float, current: float) -> float:
    """The wrist heading (yaw + roll) whose closing axis (the tool's y) lies along the horizontal
    ``direction``, the equivalent (mod pi) nearest ``current``."""
    base = math.atan2(float(direction[1]), float(direction[0])) - 0.5 * math.pi + yaw_offset
    k = round((current - base) / math.pi)
    return base + k * math.pi


def move_time(a: np.ndarray, b: np.ndarray, speed: float = TRANSIT_SPEED, floor: float = 0.4) -> float:
    """Duration of a min-jerk move between two wrist targets (positions only; peak ``speed``)."""
    return max(floor, 1.875 * float(np.linalg.norm(np.asarray(b[:3]) - np.asarray(a[:3]))) / speed)


@dataclass
class Path:
    """Min-jerk waypoints of the five wrist targets; :meth:`at` interpolates."""

    times: list[float] = field(default_factory=lambda: [0.0])
    poses: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def start(cls, pose: np.ndarray) -> Path:
        return cls([0.0], [np.asarray(pose, dtype=np.float64).copy()])

    @property
    def end(self) -> float:
        return self.times[-1]

    @property
    def last(self) -> np.ndarray:
        return self.poses[-1].copy()

    def to(self, pose: np.ndarray, duration: float | None = None) -> float:
        """Append a waypoint reached ``duration`` s later (default: :func:`move_time`); returns its time."""
        pose = np.asarray(pose, dtype=np.float64).copy()
        d = move_time(self.poses[-1], pose) if duration is None else float(duration)
        self.times.append(self.times[-1] + max(1e-3, d))
        self.poses.append(pose)
        return self.times[-1]

    def hold(self, duration: float) -> float:
        return self.to(self.poses[-1], duration)

    def at(self, t: float) -> np.ndarray:
        times = self.times
        if t <= times[0]:
            return self.poses[0].copy()
        if t >= times[-1]:
            return self.poses[-1].copy()
        i = int(np.searchsorted(times, t, side="right")) - 1
        s = min_jerk((t - times[i]) / (times[i + 1] - times[i]))
        return self.poses[i] + s * (self.poses[i + 1] - self.poses[i])


@dataclass
class Guarded:
    """A straight move from ``a`` to ``b`` over ``duration`` s that stops advancing while the
    tool presses on an object (any object in the scene) harder than :data:`GUARD_FORCE_N` (force
    feedback); it keeps its progress, so a blocked move ends where it was stopped."""

    a: np.ndarray
    b: np.ndarray
    t0: float
    duration: float
    u: float = 0.0
    last_t: float = math.nan

    def at(self, t: float, pressing: float) -> np.ndarray:
        if not math.isnan(self.last_t) and pressing <= GUARD_FORCE_N:
            self.u = min(1.0, self.u + max(0.0, t - self.last_t) / self.duration)
        self.last_t = t
        s = min_jerk(self.u)
        return self.a + s * (self.b - self.a)

    @property
    def end(self) -> float:
        return self.t0 + self.duration


def transit(path: Path, goal: np.ndarray, safe_z: float) -> float:
    """Rise to ``safe_z`` (palm), travel and turn there, then descend to ``goal``."""
    here = path.last
    if here[2] < safe_z - 1e-4:
        up = here.copy()
        up[2] = safe_z
        path.to(up)
    over = np.asarray(goal, dtype=np.float64).copy()
    over[2] = max(safe_z, over[2])
    path.to(over, max(move_time(path.last, over), 0.4 + 0.3 * abs(float(over[3] - path.last[3]))))
    return path.to(goal)


def obstacle_top(model: mujoco.MjModel, data: mujoco.MjData, geoms: Sequence[int] | np.ndarray) -> float:
    """The highest point of ``geoms`` now (world AABB of each, m)."""
    top = -math.inf
    for g in np.asarray(geoms, dtype=np.int64):
        c = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3]
        ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]
        top = max(top, float(c[2] + ext[2]))
    return top


def safe_palm_z(
    ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, tool: Tool, carried: float = 0.0
) -> float:
    """A palm height whose tool tip (and ``carried`` m below it) clears every object by 30 mm."""
    geoms = np.concatenate([o.geom_ids for o in ctx.scene.objects.values()]) if ctx.scene.objects else []
    top = obstacle_top(model, data, geoms) if len(geoms) else ctx.task.physics.table_top_z
    return top + 0.030 + carried - float(tool.tip[2])


# --------------------------------------------------------------------------- contacts


def tool_geoms(scene: StudioScene) -> np.ndarray:
    """Every colliding tool-side geom: all heads, the bracket or changer, and the wrist flange
    (a wrist that presses on an object is the tool pressing on it)."""
    hit = _TOOL_GEOMS.get(id(scene))
    if hit is not None and hit[0] is scene:
        return hit[1]
    model = scene.model
    wrist = np.flatnonzero(
        np.isin(model.geom_bodyid, [scene.ids.palm_body, scene.ids.roll_body, scene.ids.flange_body])
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    out = np.union1d(scene.ids.effector_geoms, wrist).astype(np.int64)
    _TOOL_GEOMS[id(scene)] = (scene, out)
    while len(_TOOL_GEOMS) > 32:
        _TOOL_GEOMS.pop(next(iter(_TOOL_GEOMS)))
    return out


_TOOL_GEOMS: dict[int, tuple[Any, np.ndarray]] = {}


@dataclass(frozen=True)
class Contact:
    """Tool-object contacts now: the net force the tool applies to the object (world, N), the sum
    of normal forces (N), the deepest penetration (m), the contact count and the tool bodies."""

    net: np.ndarray
    normal: float
    depth: float
    n: int
    bodies: frozenset[int]

    @property
    def lateral(self) -> float:
        return float(math.hypot(self.net[0], self.net[1]))


_EMPTY = Contact(np.zeros(3), 0.0, 0.0, 0, frozenset())


def contacts(
    model: mujoco.MjModel, data: mujoco.MjData, tool_geoms: np.ndarray, obj_geoms: np.ndarray
) -> Contact:
    """Every active contact between ``tool_geoms`` and ``obj_geoms`` (module docstring)."""
    if not data.ncon or not len(tool_geoms) or not len(obj_geoms):
        return _EMPTY
    con = data.contact
    g = con.geom
    is_tool = np.zeros(model.ngeom, dtype=bool)
    is_tool[tool_geoms] = True
    is_obj = np.zeros(model.ngeom, dtype=bool)
    is_obj[obj_geoms] = True
    a = is_tool[g[:, 0]] & is_obj[g[:, 1]]
    b = is_obj[g[:, 0]] & is_tool[g[:, 1]]
    idx = np.flatnonzero((a | b) & (con.efc_address >= 0))
    if not idx.size:
        return _EMPTY
    net = np.zeros(3)
    normal, depth = 0.0, 0.0
    bodies = set()
    wrench = np.zeros(6)
    for i in idx:
        mujoco.mj_contactForce(model, data, int(i), wrench)
        frame = con.frame[i].reshape(3, 3)
        f = frame.T @ wrench[:3]  # force on geom[1] along +normal
        sign = 1.0 if a[i] else -1.0  # tool is geom[0]: the object (geom[1]) receives +f
        net += sign * f
        normal += float(wrench[0])
        depth = max(depth, float(-con.dist[i]))
        bodies.add(int(model.geom_bodyid[g[i, 0] if a[i] else g[i, 1]]))
    return Contact(net, normal, depth, int(idx.size), frozenset(bodies))


# --------------------------------------------------------------------------- objects and criteria


def rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """``v`` turned by ``angle`` about the unit ``axis`` (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64) / float(np.linalg.norm(axis))
    v = np.asarray(v, dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c)


def _yaw_of(ctx: StepContext) -> float:
    return math.radians(ctx.task_spec.object(ctx.step.object).pose.yaw_deg)


def panel_angle(ctx: StepContext, data: mujoco.MjData) -> float:
    """A lid's or door's opening angle (rad) as the controller sees it: the observed ``lid`` site
    about the observed ``hinge`` site, from the closed pose in the template's ``info``."""
    info = ctx.obj.handles.info
    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    v0 = np.array([info["lid0_x"], info["lid0_y"], info["lid0_z"]])
    v = ctx.observe_site(data, "lid") - ctx.observe_site(data, "hinge")
    v0 = v0 - axis * float(axis @ v0)
    v = v - axis * float(axis @ v)
    return math.atan2(float(axis @ np.cross(v0, v)), float(v0 @ v))


def panel_normal(ctx: StepContext, angle: float) -> np.ndarray:
    """The panel's outer-face normal at ``angle`` (world)."""
    info = ctx.obj.handles.info
    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    return rotate(np.array([info["out0_x"], info["out0_y"], info["out0_z"]]), axis, angle)


def observed_state(ctx: StepContext, data: mujoco.MjData) -> dict[str, float]:
    """The step object's template quantities as a controller sees them: computed from observed
    sites only (controllers never read the true state). Approximations: a valve's angle is
    known modulo a turn, a thread's turns only past the thread start."""
    kind, info = ctx.obj.kind, ctx.obj.handles.info
    if kind is TemplateKind.HINGED_LID:
        return {"angle_deg": math.degrees(panel_angle(ctx, data))}
    if kind is TemplateKind.DRAWER:
        pull = np.array([info["pull_x"], info["pull_y"], 0.0])
        rel = ctx.observe_site(data, "drawer") - ctx.observe_site(data, "front")
        return {"travel_mm": 1000.0 * (float(pull @ rel) - info["front_gap_m"])}
    if kind is TemplateKind.QUARTER_TURN_LATCH:
        a = angle_about(ctx.observe_site(data, "head"), ctx.observe_site(data, "mark")) - _yaw_of(ctx)
        return {"angle_deg": math.degrees(math.remainder(a, 2.0 * math.pi))}
    if kind is TemplateKind.HANDWHEEL_VALVE:
        a = angle_about(ctx.observe_site(data, "wheel"), ctx.observe_site(data, "spoke0")) - _yaw_of(ctx)
        return {"angle_deg": math.degrees(math.remainder(a - math.pi + 0.05, 2.0 * math.pi) + math.pi - 0.05)}
    if kind is TemplateKind.THREADED_JOINT:
        depth = float(
            ctx.observe_site(data, "seat")[2] + info["length_m"] - ctx.observe_site(data, "member")[2]
        )
        turns = max(0.0, depth - info["lead_m"]) / info["pitch_m"]
        return {"depth_mm": 1000.0 * depth, "turns": turns, "angle_deg": 360.0 * turns}
    if kind is TemplateKind.PORT_PLUG:
        tip, mouth = ctx.observe_site(data, "plug_tip"), ctx.observe_site(data, "socket")
        inside = (
            math.hypot(float(tip[0] - mouth[0]), float(tip[1] - mouth[1]))
            <= info["bore_m"] + info["chamfer_m"] + 0.001
        )
        return {"depth_mm": 1000.0 * max(0.0, float(mouth[2] - tip[2])) if inside else 0.0}
    return {}


def angle_about(centre: np.ndarray, point: np.ndarray) -> float:
    """The horizontal angle of ``point`` about ``centre`` (rad, counter-clockwise from +x)."""
    return math.atan2(float(point[1] - centre[1]), float(point[0] - centre[0]))


def unwrap(previous: float, angle: float) -> float:
    """``angle`` shifted by whole turns to lie within half a turn of ``previous``."""
    return previous + math.remainder(angle - previous, 2.0 * math.pi)


def criterion_bounds(ctx: StepContext, quantity: str) -> tuple[float | None, float | None]:
    """The step's ``(at_least, at_most)`` on ``quantity`` (``None`` where unset)."""
    lo = hi = None
    for c in ctx.step.success:
        if c.quantity == quantity:
            lo = c.at_least if c.at_least is not None else lo
            hi = c.at_most if c.at_most is not None else hi
    return lo, hi


def force_limit(ctx: StepContext, default: float) -> float:
    """The push or pull force a controller allows itself: 80 % of the step's limit, else ``default``."""
    lim = ctx.step.constraints.max_force_n
    return 0.8 * lim if lim is not None else default


def torque_limit(ctx: StepContext) -> float:
    """The roll torque a controller allows itself: 90 % of the step's limit, else 95 % of the wrist's."""
    wrist = 0.95 * ctx.scene.wrist.roll_torque_nm
    lim = ctx.step.constraints.max_torque_nm
    return min(wrist, 0.9 * lim) if lim is not None else wrist


# --------------------------------------------------------------------------- schedules


@dataclass
class Schedule:
    """Moments of one step a predicate needs (s from the step's start; ``nan`` until they happen)
    and the controller's own notes (``values``)."""

    engaged: float = math.nan
    released: float = math.nan
    reg: float = math.nan  # pick: the registration moment of the v1 criterion
    hold: tuple[float, float] = (math.nan, math.nan)
    values: dict[str, float] = field(default_factory=dict)
    tool: Tool | None = None

    def active(self, t: float) -> bool:
        """Engaged and not yet released at ``t``."""
        return (
            not math.isnan(self.engaged)
            and t >= self.engaged
            and (math.isnan(self.released) or t < self.released)
        )


_SCHEDULES: weakref.WeakKeyDictionary[Any, Schedule] = weakref.WeakKeyDictionary()


def schedule(ctx: StepContext) -> Schedule:
    """The step's :class:`Schedule` (created on first use; lives as long as the context)."""
    s = _SCHEDULES.get(ctx)
    if s is None:
        s = Schedule()
        _SCHEDULES[ctx] = s
    return s
