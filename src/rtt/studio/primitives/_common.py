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
* **Wrist poses.** Every controller commands the six wrist servos (x, y, z, yaw, roll, pitch;
  :func:`wrist_target`, :func:`start_pose`). A tool point ``p`` (wrist frame, pitch and roll 0)
  is at ``palm + Rz(yaw) Rx(pitch) Rz(roll) p`` (:func:`rtt.studio.scene.wrist_rotation`), which
  is ``palm + Rz(yaw + roll) p`` with the tool straight down (pitch 0). Controllers change the
  heading with the yaw and turn objects with the roll (capped at the roll speed limit). The tool
  stays straight down except where the tilt matters (a lid's lip pinched from the front, a push
  from the front, :mod:`.hook_pull` and :mod:`.push`); a step that starts tilted turns back to
  straight down in its first move, within the pitch speed cap (:meth:`Path.to`).
* **Perception.** Objects are located only through :meth:`StepContext.observe_site` (the true
  site plus the object's hidden aim offset); a controller reads its own wrist (proprioception)
  and the tool's contact forces (a wrist force sensor): force feedback within a step is allowed
  through wrist and effector commands only, with the budgets in :data:`BUDGET_S`, equal for
  every candidate.
* **Schedules.** A controller records the moments its predicate needs (engaged, released...)
  in a :class:`Schedule` keyed by the step's context (:func:`schedule`).
* **Reach.** Every controller checks its planned palm poses against the palm's travel
  (+-350 mm, :data:`rtt.sim.scene.PALM_XY_LIMIT`, less :data:`REACH_MARGIN`; a lid's pinch or hook
  less :data:`~.hook_pull.PINCH_REACH_MARGIN`) before it commits (:func:`reach_problem`): a step
  whose target lies outside is not attempted and fails ``out_of_reach`` with where the palm would
  have to go and a plain reason (:func:`hold_out_of_reach`, :func:`out_of_reach_verdict`).
* **Letting go.** A controller sets a part down before it lets go of it: the part's lowest point
  at most :data:`RELEASE_HEIGHT_MAX_MM` above what it comes to rest on (the picks lower it until
  the wrist feels the support take it, :func:`wrist_load`). A lid is carried to its stop, not
  dropped onto it (:mod:`.push`, :mod:`.hook_pull`).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.registry import get_family
from rtt.effectors.scene import MOUNT_Z
from rtt.sim.controller import Phase, min_jerk
from rtt.sim.scene import PALM_XY_LIMIT
from rtt.studio.plan import PrimitiveCommand, StepPlan, StepVerdict, V1SceneView, _V1Ids
from rtt.studio.scene import rot_z, world_tcp, wrist_rotation
from rtt.studio.templates import thread_info
from rtt.studio.types import SLIDE_KINDS, THREAD_KINDS, TemplateKind

if TYPE_CHECKING:
    from rtt.studio.plan import StepContext
    from rtt.studio.scene import StudioScene

__all__ = [
    "BUDGET_S",
    "PRESHAPE_CLOSURES",
    "REACH_MARGIN",
    "RELEASE_HEIGHT_MAX_MM",
    "TURN_MARGIN",
    "Contact",
    "Guarded",
    "Path",
    "Schedule",
    "Tool",
    "angle_about",
    "box_corners",
    "contacts",
    "criterion_bounds",
    "footprint",
    "force_allowed",
    "force_goal",
    "geom_boxes",
    "heading_for_axis",
    "held_closure",
    "hold_out_of_reach",
    "idle_command",
    "idle_run",
    "in_reach",
    "observed_state",
    "obstacle_top",
    "offset",
    "out_of_reach_verdict",
    "panel_angle",
    "panel_normal",
    "preshape_closure",
    "reach_problem",
    "rot_z",
    "rotate",
    "schedule",
    "scheduled",
    "start_pose",
    "tool_geoms",
    "tool_of",
    "tool_point_world",
    "top_within",
    "touching",
    "turn_time",
    "wrist_frame",
    "wrist_load",
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
    "force_press": 1.0,  # a push or an insert pressing on to a force_n at_least goal
}
#: Over a force or torque goal (as a rotation goal's 3 deg), held three check ticks [assumed].
FORCE_MARGIN: Final = 0.03
FORCE_HOLD_S: Final = 0.03
TRANSIT_SPEED: Final = 0.40  # m/s peak of a min-jerk free-space move [assumed]
CALIBRATION_Z: Final = 0.75  # m above the table: the head alone, clear of every object
Mechanism = Literal["fingers", "suction", "soft_arm"]
#: Pitch and roll moves a controller plans take this much longer than the speed caps allow [assumed].
TURN_MARGIN: Final = 1.1
#: A part is let go with its lowest point at most this far (mm) above what it comes to rest on [assumed].
RELEASE_HEIGHT_MAX_MM: Final = 20.0
#: m a plan keeps every palm pose inside the palm's travel: the 2 mm hidden aim error with room to
#: spare [assumed]. A lid's pinch or hook keeps :data:`~.hook_pull.PINCH_REACH_MARGIN` (12 mm: the
#: pinch's 10 mm yield along the tilted tool on top of the aim error). The panel rail's reach
#: envelope (+-182.5 mm at any yaw) is sized on this one.
REACH_MARGIN: Final = 0.005


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

    def hook(
        self, band: float = 0.015, flange: tuple[float, float, float] | None = None
    ) -> tuple[np.ndarray, float, float, float]:
        """The tool's best sideways hook: the horizontal wrist-frame direction ``d`` in which the
        geoms lying wholly within ``band`` of the tip reach furthest beyond the rest of the tool,
        how far they reach beyond it (m; 0 for a straight tool), how far they reach from the axis,
        and the top of that protrusion (wrist z). A soft arm's horizontal arm is such a hook; so
        would be any L-shaped finger.

        ``flange``: ``(face z, radius, throat)`` (m) counts the flange as part of the tool when its
        face is less than ``throat`` above the protrusion (nothing fits between them), so a plate
        flush under the flange is no hook (the search's capability check,
        :func:`rtt.studio.search.measured_hook_mm`); ``None`` (the hook pull) leaves it out."""
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

        def beyond(near: float, d: np.ndarray) -> np.ndarray:
            return low[
                (
                    np.where(d[0] >= 0, low[:, 3], low[:, 0]) * d[0]
                    + np.where(d[1] >= 0, low[:, 4], low[:, 1]) * d[1]
                )
                > near
            ]

        for k in range(72):
            a = 2.0 * math.pi * k / 72
            d = np.array([math.cos(a), math.sin(a), 0.0])
            far, near = reach(low, d), reach(body, d)
            if flange is not None:
                out = beyond(near, d)
                if len(out) and flange[0] - float(out[:, 5].max()) < flange[2]:
                    near = max(near, flange[1])
            if far - near > best[1] + 1e-6:
                out = beyond(near, d)
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
        return preshape_closure(self.footprint, width, clearance)


def preshape_closure(
    footprint_at: Callable[[float], Footprint], width: float, clearance: float = PRESHAPE_CLEARANCE
) -> float:
    """:meth:`Tool.preshape` from a head's calibrated footprints (``footprint_at(closure)``)."""
    best = 0.0
    for c in PRESHAPE_CLOSURES:
        ap = footprint_at(c).aperture()
        if ap is not None and ap >= width + 2.0 * clearance:
            best = c
    return best


_CORNER_SIGNS: Final = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=bool)


def box_corners(boxes: np.ndarray) -> np.ndarray:
    """``(m, 8, 3)`` corners of ``(m, 6)`` boxes (lo xyz, hi xyz)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
    return np.where(_CORNER_SIGNS[None, :, :], b[:, None, 3:], b[:, None, :3])


def geom_boxes(model: mujoco.MjModel, data: mujoco.MjData, geoms: Sequence[int] | np.ndarray) -> np.ndarray:
    """``(k, 6)`` world boxes (lo xyz, hi xyz) of ``geoms`` now, seen as :func:`obstacle_top` sees
    heights."""
    out = np.zeros((len(geoms), 6))
    for i, g in enumerate(np.asarray(geoms, dtype=np.int64)):
        c = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3]
        ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]
        out[i, :3], out[i, 3:] = c - ext, c + ext
    return out


def tool_of(ctx: StepContext, body: int | None = None) -> Tool:
    """Describe the step's active head (module docstring); ``body``: the object body the v1
    primitive and criterion should treat as the part (default: the object's last body)."""
    head = ctx.head
    impl = get_family(head.spec.family)
    build = head.build
    view = _view(ctx, body)
    # build.tcp and build.reach are in the head's mount frame, whose mount plane is MOUNT_Z on the
    # flange, a changer and a bracket alike (the scene builds every head so)
    tip_drop = float(build.tcp[2]) - (MOUNT_Z - float(build.reach))
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


def wrist_target(
    point: np.ndarray, tool_point: np.ndarray, yaw: float, roll: float, pitch: float = 0.0
) -> np.ndarray:
    """The six wrist targets (x, y, z, yaw, roll, pitch) that put ``tool_point`` (wrist frame,
    pitch and roll 0) at world ``point``; ``pitch`` 0 is the tool straight down."""
    r = wrist_rotation(yaw, roll, pitch)
    palm = np.asarray(point, dtype=np.float64) - r @ np.asarray(tool_point, dtype=np.float64)
    return np.array([palm[0], palm[1], palm[2], yaw, roll, pitch])


def start_pose(ctx: StepContext) -> np.ndarray:
    """The six wrist targets when the step starts: :attr:`StepContext.wrist_start` (five) and
    :attr:`StepContext.wrist_pitch_start`."""
    return np.append(np.asarray(ctx.wrist_start, dtype=np.float64)[:5], float(ctx.wrist_pitch_start))


def in_reach(*poses: np.ndarray, margin: float = REACH_MARGIN) -> bool:
    """Whether the wrist reaches every pose: its x and y within the palm's travel
    (:data:`rtt.sim.scene.PALM_XY_LIMIT`) less ``margin``. A tilted tool puts the palm its own
    length out from where it works, so a tilted approach checks it before it commits."""
    return all(max(abs(float(p[0])), abs(float(p[1]))) <= PALM_XY_LIMIT - margin for p in poses)


# --------------------------------------------------------------------------- out of reach


def reach_problem(*poses: np.ndarray, margin: float = REACH_MARGIN) -> dict[str, float] | None:
    """Where the palm would have to go (mm) for the first of ``poses`` outside its travel less
    ``margin`` (:func:`in_reach`), the travel and the margin; ``None`` when it reaches every one.
    Every primitive checks its planned poses with it (Studio v2): a step whose target the palm
    cannot reach is not attempted (:func:`hold_out_of_reach`)."""
    for pose in poses:
        if not in_reach(pose, margin=margin):
            return {
                "reach_x_mm": 1000.0 * float(pose[0]),
                "reach_y_mm": 1000.0 * float(pose[1]),
                "palm_limit_mm": 1000.0 * PALM_XY_LIMIT,
                "reach_margin_mm": 1000.0 * margin,
            }
    return None


def hold_out_of_reach(
    ctx: StepContext, start: np.ndarray, far: Mapping[str, float], what: str = "its target"
) -> StepPlan:
    """The plan of a step whose target is out of reach (``far``: :func:`reach_problem`): hold still
    0.2 s, the tool as it was (:func:`idle_command`; the plan's ``run`` is ``{"idle": start,
    "closure": ...}``); its predicate fails it ``out_of_reach`` (:func:`out_of_reach_verdict`).
    ``what``: the move that would take the palm out, in words, for the step's plain reason."""
    sch = schedule(ctx)
    sch.engaged = sch.released = sch.reg = math.nan  # nothing is attempted
    sch.hold = (math.nan, math.nan)
    sch.values.update(far)
    sch.values["out_of_reach"] = 1.0
    sch.values["reach_what"] = what  # type: ignore[assignment]
    note = (
        f"out of reach: the palm would go to x {far['reach_x_mm']:.0f}, y {far['reach_y_mm']:.0f} mm; "
        f"it reaches +-{far['palm_limit_mm']:.0f} mm"
    )
    return StepPlan(duration_s=0.2, values={"run": idle_run(ctx, start), "schedule": sch}, note=note)


def held_closure(ctx: StepContext) -> float:
    """The tool's closure when the step starts: 1 when the previous step ends holding this step's
    object (``hold_after``: the same tool, closed on it), else 0 (open)."""
    steps = ctx.task_spec.steps
    i = next(k for k, s in enumerate(steps) if s.id == ctx.step.id)
    return 1.0 if i > 0 and steps[i - 1].hold_after else 0.0


def idle_run(ctx: StepContext, start: np.ndarray) -> dict[str, Any]:
    """The ``run`` of a step that is not tried: hold the wrist at ``start``, the tool as it was."""
    return {"idle": start.copy(), "closure": held_closure(ctx)}


def idle_command(
    t: float, ctx: StepContext, run: Mapping[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> PrimitiveCommand:
    """One tick of a step that is not tried (:func:`idle_run`): the wrist holds still and the tool
    keeps its closure, so a part the previous step holds stays held (open otherwise)."""
    closure = float(run["closure"])
    phase = Phase.HOLD if closure > 0.0 else Phase.DESCEND
    return PrimitiveCommand(run["idle"], schedule(ctx).tool.controls(t, closure, model, data, phase))


def out_of_reach_verdict(ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict | None:
    """``out_of_reach`` for a step planned by :func:`hold_out_of_reach`, with where the palm would
    have to go (``reach_x_mm``, ``reach_y_mm``), its travel (``palm_limit_mm``) and a plain
    reason; else ``None``."""
    sch = scheduled(ctx)
    if sch is None or not sch.values.get("out_of_reach"):
        return None
    v = sch.values
    keys = ("reach_x_mm", "reach_y_mm", "palm_limit_mm")
    margin = float(v.get("reach_margin_mm", 1000.0 * REACH_MARGIN))
    reason = (
        f"Out of reach: {v.get('reach_what', 'its target')} would take the palm to x "
        f"{v['reach_x_mm']:.0f} mm, y {v['reach_y_mm']:.0f} mm, past its +-{v['palm_limit_mm']:.0f} mm "
        f"travel less {margin:.0f} mm; the step was not attempted"
    )
    return StepVerdict(False, "out_of_reach", {**acc, **{k: float(v[k]) for k in keys}}, (reason,))


def offset(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """A change of the six wrist targets that moves the wrist by ``(x, y, z)`` m and turns nothing."""
    return np.array([x, y, z, 0.0, 0.0, 0.0])


def turn_time(angle: float, speed_dps: float) -> float:
    """The duration of a min-jerk turn by ``angle`` rad whose peak speed stays under ``speed_dps``
    by :data:`TURN_MARGIN` (0 for no turn)."""
    return TURN_MARGIN * 1.875 * abs(float(angle)) / math.radians(speed_dps)


def tool_point_world(scene: StudioScene, data: mujoco.MjData, tool_point: np.ndarray) -> np.ndarray:
    """Where ``tool_point`` (wrist frame, pitch and roll 0) is now (proprioception: the wrist's own
    joints, :func:`rtt.studio.scene.world_tcp`)."""
    return world_tcp(scene, data, tool_point)


def wrist_frame(scene: StudioScene, data: mujoco.MjData) -> np.ndarray:
    """The tool's orientation now (world from wrist frame), from the wrist's own joints."""
    q = data.qpos
    return wrist_rotation(
        float(q[scene.ids.palm_qpos[3]]), float(q[scene.ids.roll_qpos]), float(q[scene.ids.pitch_qpos])
    )


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
    """Min-jerk waypoints of the six wrist targets; :meth:`at` interpolates. ``pitch_dps``: the
    scene's pitch speed cap (``scene.wrist.pitch_speed_dps``), which bounds every pitch move."""

    times: list[float] = field(default_factory=lambda: [0.0])
    poses: list[np.ndarray] = field(default_factory=list)
    pitch_dps: float = math.nan  # set by :meth:`start`; a pitch move without it is refused

    @classmethod
    def start(cls, pose: np.ndarray, pitch_dps: float) -> Path:
        """A path from ``pose`` on a wrist whose pitch moves at most ``pitch_dps``."""
        return cls([0.0], [np.asarray(pose, dtype=np.float64).copy()], float(pitch_dps))

    @property
    def end(self) -> float:
        return self.times[-1]

    @property
    def last(self) -> np.ndarray:
        return self.poses[-1].copy()

    def to(self, pose: np.ndarray, duration: float | None = None) -> float:
        """Append a waypoint reached ``duration`` s later (default: :func:`move_time`); returns its
        time. A move that changes the pitch lasts at least its :func:`turn_time` at the scene's
        pitch speed cap (a step that starts tilted turns back to straight down within it)."""
        pose = np.asarray(pose, dtype=np.float64).copy()
        d = move_time(self.poses[-1], pose) if duration is None else float(duration)
        dp = float(pose[5] - self.poses[-1][5])
        if dp:
            if not math.isfinite(self.pitch_dps):
                raise ValueError("a path that turns the pitch needs the scene's pitch cap (Path.start)")
            d = max(d, turn_time(dp, self.pitch_dps))
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
    tool presses on an object (any object in the scene) harder than ``force`` (default
    :data:`GUARD_FORCE_N`; force feedback); it keeps its progress, so a blocked move ends where it
    was stopped (:attr:`blocked`)."""

    a: np.ndarray
    b: np.ndarray
    t0: float
    duration: float
    u: float = 0.0
    last_t: float = math.nan
    force: float = GUARD_FORCE_N

    @property
    def blocked(self) -> bool:
        """Stopped short of ``b``."""
        return self.u < 1.0

    def at(self, t: float, pressing: float) -> np.ndarray:
        if not math.isnan(self.last_t) and pressing <= self.force:
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


def top_within(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geoms: Sequence[int] | np.ndarray,
    centre: np.ndarray,
    radius: float,
    below: float = math.inf,
) -> float:
    """The highest point now of the ``geoms`` whose world AABB overlaps the horizontal disc of
    ``radius`` m about ``centre`` (x, y) and whose top is below ``below`` (m); ``-inf`` for none.
    Seen as :func:`obstacle_top` sees heights (what a part set down there would land on)."""
    top = -math.inf
    cx, cy = float(centre[0]), float(centre[1])
    for g in np.asarray(geoms, dtype=np.int64):
        c = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g][:3]
        ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ model.geom_aabb[g][3:]
        dx = max(0.0, abs(cx - float(c[0])) - float(ext[0]))
        dy = max(0.0, abs(cy - float(c[1])) - float(ext[1]))
        z = float(c[2] + ext[2])
        if math.hypot(dx, dy) < radius and z < below:
            top = max(top, z)
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
        np.isin(
            model.geom_bodyid,
            [scene.ids.palm_body, scene.ids.pitch_body, scene.ids.roll_body, scene.ids.flange_body],
        )
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


def touching(model: mujoco.MjModel, data: mujoco.MjData, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether any geom of ``a`` is in active contact with any geom of ``b`` now (no forces read)."""
    n = int(data.ncon)
    if not n or not len(a) or not len(b):
        return False
    g = data.contact.geom[:n]
    live = data.contact.efc_address[:n] >= 0
    in_a0, in_a1 = np.isin(g[:, 0], a), np.isin(g[:, 1], a)
    in_b0, in_b1 = np.isin(g[:, 0], b), np.isin(g[:, 1], b)
    return bool(np.any(live & ((in_a0 & in_b1) | (in_b0 & in_a1))))


#: every colliding geom that is not the tool's, per scene (weakly keyed: a scene's mask dies with it)
_OTHERS: weakref.WeakKeyDictionary[StudioScene, np.ndarray] = weakref.WeakKeyDictionary()


def wrist_load(scene: StudioScene, model: mujoco.MjModel, data: mujoco.MjData) -> Contact:
    """The tool's contacts with everything that is not the tool (objects, the table): what a force
    sensor at the wrist reads of the world (:attr:`Contact.net` is the force the tool applies to
    it). A carried part's weight, and its support taking that weight, show up here."""
    others = _OTHERS.get(scene)
    if others is None:
        live = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        others = np.setdiff1d(live, tool_geoms(scene)).astype(np.int64)
        others.flags.writeable = False
        _OTHERS[scene] = others
    return contacts(model, data, tool_geoms(scene), others)


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
    if kind in SLIDE_KINDS:  # a drawer, or a door sliding sideways (its moving site is ``door``)
        pull = np.array([info["pull_x"], info["pull_y"], 0.0])
        moving = "drawer" if kind is TemplateKind.DRAWER else "door"
        rel = ctx.observe_site(data, moving) - ctx.observe_site(data, "front")
        return {"travel_mm": 1000.0 * (float(pull @ rel) - info["front_gap_m"])}
    if kind is TemplateKind.QUARTER_TURN_LATCH:
        a = angle_about(ctx.observe_site(data, "head"), ctx.observe_site(data, "mark")) - _yaw_of(ctx)
        return {"angle_deg": math.degrees(math.remainder(a, 2.0 * math.pi))}
    if kind is TemplateKind.HANDWHEEL_VALVE:
        a = angle_about(ctx.observe_site(data, "wheel"), ctx.observe_site(data, "spoke0")) - _yaw_of(ctx)
        return {"angle_deg": math.degrees(math.remainder(a - math.pi + 0.05, 2.0 * math.pi) + math.pi - 0.05)}
    if kind in THREAD_KINDS:  # a threaded joint, or a panel's clamp (:func:`thread_info`)
        info = thread_info(ctx.obj)
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
    """The step's ``(at_least, at_most)`` on ``quantity`` (``None`` where unset); with several
    criteria on it, every one must hold: the largest at_least and the smallest at_most."""
    lo: float | None = None
    hi: float | None = None
    for c in ctx.step.success:
        if c.quantity == quantity:
            if c.at_least is not None:
                lo = c.at_least if lo is None else max(lo, c.at_least)
            if c.at_most is not None:
                hi = c.at_most if hi is None else min(hi, c.at_most)
    return lo, hi


def force_limit(ctx: StepContext, default: float) -> float:
    """The push or pull force a controller allows itself: 80 % of the step's limit, else ``default``."""
    lim = ctx.step.constraints.max_force_n
    return 0.8 * lim if lim is not None else default


def force_allowed(ctx: StepContext, default: float) -> float:
    """:func:`force_limit`, and 80 % of a ``force_n`` at_most criterion when the step has one (an
    upper bound on the force limits a push or a pull as the step's ``max_force_n`` does)."""
    _, hi = criterion_bounds(ctx, "force_n")
    out = force_limit(ctx, default)
    return min(out, 0.8 * hi) if hi is not None else out


def press_reach(ctx: StepContext, limit: float) -> float:
    """How far past where it meets resistance a push or a pull may command the wrist (m) to press
    with ``limit``: the force over the stiffer-axis servo's stiffness (the softer axis), plus 2 mm."""
    g = ctx.task.gripper
    return limit / min(g.xy_kp, g.z_kp) + 0.002


def force_goal(ctx: StepContext) -> float | None:
    """The force a push or an insert presses to when the step asks for a ``force_n`` at_least:
    that force plus :data:`FORCE_MARGIN` (``None`` without one). The press ends once the measured
    force has been over it for :data:`FORCE_HOLD_S`, so the predicate's check ticks see it."""
    lo, _ = criterion_bounds(ctx, "force_n")
    return None if lo is None else lo * (1.0 + FORCE_MARGIN)


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


def scheduled(ctx: Any) -> Schedule | None:
    """The step's :class:`Schedule` if its controller made one, without creating it (``None`` for a
    context that has none, such as a bare namespace a predicate is unit-tested with)."""
    try:
        return _SCHEDULES.get(ctx)
    except TypeError:  # it cannot be weakly referenced, so it was never given a schedule
        return None


def schedule(ctx: StepContext) -> Schedule:
    """The step's :class:`Schedule` (created on first use; lives as long as the context)."""
    s = _SCHEDULES.get(ctx)
    if s is None:
        s = Schedule()
        _SCHEDULES[ctx] = s
    return s
