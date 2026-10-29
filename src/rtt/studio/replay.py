"""Studio replays for the site: full-task runs as replay v2 lanes, and the blocker probe (track S7).

Product mode: every lane is labelled so (``provenance.product_mode``) and is a re-simulation of a
scored run, never a new result.

**Recording.** :func:`record_run` re-runs one ``(setup, task, condition)`` of a tournament report
through the frozen full-task runner (:func:`rtt.studio.episode.run_task`, in process) with a
sampler wrapped around the step controllers and predicates: it only reads ``data.xpos`` and
``data.xquat`` at the controller ticks (before each physics step) and at the predicate's check
ticks (after one): the first tick at or after every frame time, the tick at which a contact
starts or stops (an impact lands between two frame times, and a pose interpolated across it
would be off by millimetres) and every tick for :data:`SETTLE_S` after a contact change, after
the start of a segment and after a body moves faster than :data:`FAST_M_S` (a finger settling, a
jaw snapping open: motion the frame rate does not follow; a fast move also takes back the ticks
since the last frame). Nothing is written to the simulation, so the run's ``traj_sha`` is
the scored one; :func:`record_run` refuses a run whose ``traj_sha`` differs from the report's
(:class:`ReplayMismatchError`). Bodies are sampled at a fixed rate (:data:`RECORD_FPS`) on the
task clock; the contacts the simulation reports are read only and logged (:func:`contact_times`).

**Segments.** A setup's segments (:meth:`rtt.studio.types.Setup.segments`) run in their own
scenes. A tool change or a head switch is a fixed clock advance in the contract, not simulated
motion, so the replay holds the last pose of a segment through the change time and then cuts to
the next segment's first frame (its home pose). Bodies are matched across scenes by a stable key:
``t<k>/<family name>`` for the bodies of tool ``k`` (an idle head's ``eff_h<k>_`` prefix is
removed), the element name otherwise (wrist, bracket, changer, objects). A tool-change tool that
is not on the wrist rests on a display-only **tool stand** behind the table, in the pose it hangs
in on the wrist at its first frame (the stand is drawn as a prop and labelled not simulated).
A segment boundary is a **cut** of the lane: the lane holds the last frame of a segment, then
shows the first frame of the next; no pose is ever interpolated across it.

**Drawing.** Our own geometry (template objects, procedural tools, the bracket, the changer and
the flange) is drawn from content-addressed GLB assets made from each body's collision geoms
(:func:`rtt.render.glb.mesh_from_geoms`, Apache-2.0). Bodies of third-party tool models (any build
with licence records: vendored grippers and hands, the stock blade) get no GLB: they are drawn from
their collision geoms in the header, as the effector lanes are, and the lane lists their brand-free
credit with the credits page anchor. Bodies that never move relative to each other are merged into
one replay body (the members become visuals or geoms placed in its frame), and bodies that never
move at all become static, which keeps the pose payload small.

**Keyframes.** A lane keeps only the frames its bodies need (:func:`select_keys`): the recording's
samples are thinned until linear position and spherical rotation between the kept frames stay
within :data:`KEY_TOL_M` (1 mm) of the recording at every vertex of every body (so the farthest
vertex of a body is within 1 mm at every recorded sample; between two samples a body that
shakes faster than the recording can see is not followed). One frame list is shared by all
bodies (a frame is kept when any body needs it). A frame is always kept at the start and the end of a run, at the first
and the last sample of every segment (so one lies exactly on each cut) and at the onset and the
release of every contact that lasts at least :data:`CONTACT_HOLD_S`. The header's ``t`` gives the
time of each kept frame and ``cuts`` the index of the first frame of each segment after the first.
Each lane must fit :data:`LANE_BUDGET_BYTES` (header plus poses as served): when float32 poses
do not, the poses are quantised to int16 (``quantised``; position in 0.1 mm, rotation in three
components) before any further thinning, and only then is the tolerance relaxed to the next of
:data:`KEY_TOLERANCES_M` (1, 1.5 and 2 mm: nothing looser passes the replay gate,
:mod:`rtt.studio.replay_qa`, so a lane that does not fit at 2 mm is refused). A lane whose bodies go
beyond what int16 holds (+-3.27 m) skips the int16 candidate and goes down the ladder as float32.
The lane's meta records the tolerance and the worst error against the recording
(``max_error_vs_recording_mm``: the recording, not the simulation, which the gate measures).

**Blocker probe.** :func:`blocker_probe` explains a task nothing completes: the step most finalist
runs stop at is re-run for every candidate the screen ran on it, from the same carried context
(the screen's best earlier tools, on the screen's conditions), and the peak of the step's most
missed goal quantity (a twist's torque, for example) is reported per tool family against its
target. The probe re-checks each candidate's successes against the report's success matrix.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.sim import replay as rp

__all__ = [
    "CONTACT_HOLD_S",
    "FAST_M_S",
    "KEY_TOLERANCES_M",
    "KEY_TOL_M",
    "LANE_BUDGET_BYTES",
    "QUANT_POSE_FIELDS",
    "RECORD_FPS",
    "SETTLE_S",
    "LaneBuild",
    "LaneReplay",
    "QuantiseRangeError",
    "Recording",
    "ReplayMismatchError",
    "blocker_probe",
    "build_lane",
    "check_lane_header",
    "contact_times",
    "dequantise_poses",
    "export_lanes",
    "lane_bytes",
    "lane_choices",
    "publish_assets",
    "quantise_poses",
    "read_lane",
    "record_run",
    "sample_lane",
    "select_keys",
    "serve_assets",
    "staging_store",
    "write_lane",
]

#: Frames per second of the recording: the rate the simulation is sampled at (the keyframes of a
#: lane are chosen from these samples, and from the extra ones taken at every contact change).
RECORD_FPS: Final = 100
#: Seconds after a contact change and after the start of a segment (and of the run) for which every
#: control tick is a candidate frame of the recording: the keyframer keeps the ones it needs.
SETTLE_S: Final = 0.030
#: Metres a second [assumed]: a body that moves faster than this between two control ticks (a jaw
#: snapping open, a part dropped) opens the same settle window: at 100 Hz such a move is a handful of
#: samples that a straight line between them does not follow within 2 mm. It is a heuristic for
#: choosing what to record, not a guarantee: the replay gate (:mod:`rtt.studio.replay_qa`) measures the
#: lane against a 500 Hz re-run and refuses one this rule did not keep close enough.
FAST_M_S: Final = 0.5
#: Metres a lane's interpolation may be from the recording at any vertex of any body.
KEY_TOL_M: Final = 1e-3
#: From the tightest to the loosest: a lane that does not fit its budget even with int16 poses is
#: thinned to the first of these that fits (the first is :data:`KEY_TOL_M`). The loosest is the
#: replay gate's own limit against the simulation (``rtt.studio.replay_qa.INTERP_TOL_M``): a lane
#: that needs more is refused, not published loose.
KEY_TOLERANCES_M: Final = (KEY_TOL_M, 1.5e-3, 2e-3)
#: A contact (a pair of bodies touching) counts once it has lasted this long, and a gap shorter
#: than this inside one is not a release: the solver's chatter is not an event.
CONTACT_HOLD_S: Final = 0.05
#: Bytes a lane may serve: its header and poses (its GLBs are shared and counted apart).
LANE_BUDGET_BYTES: Final = 480_000
#: Circle segments of the GLB meshes made from cylinders and capsules (small tool parts).
GLB_SEGMENTS: Final = 24
#: Two bodies are one rigid group when their relative pose never moves more than this.
RIGID_TOL_M: Final = 5e-5
RIGID_TOL_RAD: Final = math.radians(0.05)
#: The display-only tool stand: flange frames of parked tools, behind the table (world m).
STAND_Y_M: Final = 0.45
STAND_Z_M: Final = 0.42
STAND_PITCH_M: Final = 0.17
#: Credits page anchors of third-party tool models, by where their files live.
_CREDIT_ANCHORS: Final = (
    ("assets/effectors/library/", "library"),
    ("assets/effectors/catalog/", "effectors"),
    ("assets/gripper/", "gripper"),
)
_LOOKS_BY_KIND: Final[Mapping[str, str]] = {
    "part": "plastic_blue",
    "bin": "pvc_white",
    "hinged_lid": "pvc_white",
    "drawer": "pvc_white",
    "quarter_turn_latch": "steel_ground",
    "threaded_joint": "steel_zinc",
    "handwheel_valve": "steel_stamped",
    "port_plug": "aluminium_anodised",
    # Studio v2 job templates: rails and clamp aluminium, a mortar bed, a plastic button housing, a
    # steel machine door (the parts that differ are in _LOOKS_BY_PART)
    "panel_rail": "aluminium_anodised",
    "tile_bed": "mortar",
    "spring_button": "plastic_grey",
    "sliding_door": "steel_stamped",
}
#: Bodies of a template drawn in another look than the rest of it, by (template, body name after the
#: object's prefix): the solar module's glass on its rails, the tile on its bed, the button's cap.
_LOOKS_BY_PART: Final[Mapping[tuple[str, str], str]] = {
    ("panel_rail", "panel"): "glass_panel",
    ("tile_bed", "tile"): "ceramic_tile",
    ("spring_button", "cap"): "plastic_blue",
}


class ReplayMismatchError(RuntimeError):
    """A re-run did not reproduce the scored run (its ``traj_sha`` differs)."""


# --------------------------------------------------------------------------- poses


def _quat_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([q_wxyz[..., 1:4], q_wxyz[..., 0:1]], axis=-1)


def _rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrices of xyzw quaternions ``(..., 4) -> (..., 3, 3)``."""
    x, y, z, w = (q_xyzw[..., i] for i in range(4))
    out = np.empty((*q_xyzw.shape[:-1], 3, 3))
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _mat_quat(r: np.ndarray) -> np.ndarray:
    """An xyzw quaternion of a rotation matrix (w >= 0)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).ravel())
    if q[0] < 0:
        q = -q
    return np.array([q[1], q[2], q[3], q[0]])


def _relative(parent: np.ndarray, child: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``child`` in ``parent``'s frame for pose arrays ``(..., 7)``: positions and rotations."""
    rp, rc = _rot(parent[..., 3:]), _rot(child[..., 3:])
    rpt = np.swapaxes(rp, -1, -2)
    pos = np.einsum("...ij,...j->...i", rpt, child[..., :3] - parent[..., :3])
    return pos, rpt @ rc


def _compose(parent: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """The world pose (7,) of a frame at ``pos``/``rot`` in ``parent`` (7,)."""
    rp = _rot(parent[3:])
    out = np.empty(7)
    out[:3] = parent[:3] + rp @ pos
    out[3:] = _mat_quat(rp @ rot)
    return out


def _rigid(parent: np.ndarray, child: np.ndarray) -> bool:
    """Whether ``child`` keeps one pose in ``parent``'s frame over every frame (``(F, 7)`` each)."""
    pos, rot = _relative(parent, child)
    if np.max(np.linalg.norm(pos - pos[0], axis=-1)) > RIGID_TOL_M:
        return False
    d = np.einsum("ij,fjk->fik", rot[0].T, rot)
    cos = np.clip((np.trace(d, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return bool(np.max(np.arccos(cos)) <= RIGID_TOL_RAD)


def _static(track: np.ndarray) -> bool:
    if np.max(np.linalg.norm(track[:, :3] - track[0, :3], axis=-1)) > RIGID_TOL_M:
        return False
    dots = np.abs(track[:, 3:] @ track[0, 3:])
    return bool(np.min(dots) >= math.cos(RIGID_TOL_RAD / 2.0))


# --------------------------------------------------------------------------- recording


@dataclass
class _BodyInfo:
    """What a replay body (before grouping) is: where it belongs and how it is drawn."""

    key: str
    category: str  # wrist | stack | tool | object
    tool: int | None
    geoms: list[dict[str, Any]]
    third_party: bool
    licences: tuple[str, ...] = ()
    anchor: str | None = None
    template: str | None = None
    label: str = ""
    asset: Any = ...  # the GLB asset id once made (None: does not mesh)
    part: str = ""  # an object body's name after its object's prefix ("panel" of obj_panel_panel)


@dataclass
class _SceneMap:
    bodies: np.ndarray  # body ids
    keys: list[str]
    geom_slot: np.ndarray  # per geom: index into ``keys``, ``len(keys)`` for the world, -1 not drawn


@dataclass
class Recording:
    """One re-simulated run: frames on the task clock and the result it reproduced."""

    setup: Any
    task_spec: Any
    condition: Any
    result: Any
    fps: int
    times: list[float] = field(default_factory=list)
    frames: list[dict[str, np.ndarray]] = field(default_factory=list)
    scene_of_frame: list[int] = field(default_factory=list)
    bodies: dict[str, _BodyInfo] = field(default_factory=dict)
    scene_tools: list[int] = field(default_factory=list)
    world_static: list[dict[str, Any]] = field(default_factory=list)
    #: ``(t, touching, key_a, key_b)``: a pair of bodies (the world is ``"world"``) starts or stops
    #: touching in the simulation's contact list; raw, so chatter included (:func:`contact_times`).
    contact_log: list[tuple[float, bool, str, str]] = field(default_factory=list)
    #: How the run was made (the gate re-runs it at 500 Hz the same way).
    controllers: str = "default"
    task: Any = None


def _geom_entries(model: mujoco.MjModel, body: int) -> list[dict[str, Any]]:
    from rtt.effectors.runner import _effector_geom_entry

    out = []
    for g in np.flatnonzero(model.geom_bodyid == body):
        if model.geom_rgba[g][3] <= 0.0 or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        out.append(_effector_geom_entry(model, int(g)))
    return out


def _anchor(sources: Sequence[Any]) -> str | None:
    for s in sources:
        for prefix, anchor in _CREDIT_ANCHORS:
            if str(getattr(s, "path", "")).startswith(prefix):
                return anchor
    return "effectors" if sources else None


class _Sampler:
    """Samples body poses on the task clock (module docstring); reads the simulation only."""

    def __init__(self, recording: Recording) -> None:
        self.rec = recording
        self.next_t = 0.0
        self._maps: dict[int, _SceneMap] = {}
        self._scenes: list[int] = []
        self._touching: frozenset[int] = frozenset()
        self._seen: tuple[int, float] = (-1, -math.inf)  # the scene and time of the last contact look
        self._scene_seen = -1  # the scene of the last tick (a change starts a segment)
        self._settle_until = -math.inf  # every tick up to this time is a frame (SETTLE_S)
        self._tick_t = -math.inf  # the time of the last control tick looked at for speed
        self._prev_pos: dict[int, np.ndarray] = {}  # the poses of the last such tick, by scene
        self._pending: list[tuple[float, np.ndarray]] = []  # ticks since the last frame, not kept (yet)

    def _map(self, scene: Any) -> _SceneMap:
        sid = id(scene)
        found = self._maps.get(sid)
        if found is not None:
            return found
        from rtt.studio.scene import WRIST_BODIES

        model = scene.model
        heads = list(scene.heads)
        active = next(h for h in heads if h.active)
        own = {active.name(b) for b in active.build.bodies}
        multi = scene.setup.kind.value == "multi_head"
        ids, keys = [], []
        for b in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            geoms = _geom_entries(model, b)
            tool: int | None = None
            fam = name
            for h in heads:
                if h.prefix and name.startswith(h.prefix):
                    tool, fam = h.tool, name[len(h.prefix) :]
                    break
            if tool is None and (name in own or (multi and name in ("eff_studio_mount", "eff_studio_face"))):
                tool = active.tool
            if tool is not None:
                key, category = f"t{tool}/{fam}", "tool"
            elif name.startswith("obj_"):
                key, category = name, "object"
            elif name in WRIST_BODIES:
                key, category = name, "wrist"
            else:
                key, category = name, "stack"
            if not geoms:
                continue  # nothing to draw (the palm, the roll stage, mount frames)
            if key not in self.rec.bodies:
                head = next((h for h in heads if h.tool == tool), None) if tool is not None else None
                # an adapter plate of a multi-tool head is the Studio's own part, not the tool's
                sources = tuple(head.build.sources) if head is not None and "studio_" not in fam else ()
                template, part = None, ""
                if category == "object":
                    inst = next(
                        (o for o in scene.task_spec.objects if name.startswith(f"obj_{o.name}_")), None
                    )
                    template = inst.kind.value if inst is not None else None
                    part = name[len(f"obj_{inst.name}_") :] if inst is not None else ""
                self.rec.bodies[key] = _BodyInfo(
                    key=key,
                    category=category,
                    tool=tool,
                    geoms=geoms,
                    third_party=bool(sources),
                    licences=tuple(sorted({s.license for s in sources})),
                    anchor=_anchor(sources),
                    template=template,
                    label=fam,
                    part=part,
                )
            ids.append(b)
            keys.append(key)
        if not self.rec.world_static:
            self.rec.world_static = _geom_entries(model, 0)
        slot_of_body = {int(b): i for i, b in enumerate(ids)}
        slots = [
            len(keys) if int(model.geom_bodyid[g]) == 0 else slot_of_body.get(int(model.geom_bodyid[g]), -1)
            for g in range(model.ngeom)
        ]
        found = _SceneMap(np.array(ids, dtype=int), keys, np.array(slots, dtype=int))
        self._maps[sid] = found
        self._scenes.append(sid)
        self.rec.scene_tools.append(int(scene.tool))
        return found

    def _watch_contacts(
        self, sid: int, smap: _SceneMap, data: mujoco.MjData, t: float, step_dt: float
    ) -> bool:
        """Log the pairs of drawn bodies (and the world) that start or stop touching (read only);
        true when this tick changed the set. A scene's first look starts from no contacts and a
        scene's last is not a release: a segment boundary is a cut and gets its frames anyway."""
        if sid == self._seen[0] and t < self._seen[1] + 0.5 * step_dt:
            return False  # the controller and the predicate look at the same tick
        n = int(data.ncon)
        modulus = len(smap.keys) + 1
        now: frozenset[int] = frozenset()
        if n:
            slots = smap.geom_slot[np.asarray(data.contact.geom[:n])]
            ok = np.all(slots >= 0, axis=1) & (slots[:, 0] != slots[:, 1])
            if ok.any():
                pairs = np.sort(slots[ok], axis=1)
                now = frozenset((pairs[:, 0] * modulus + pairs[:, 1]).tolist())
        before = self._touching if sid == self._seen[0] else frozenset()
        names = [*smap.keys, "world"]
        for code in sorted(now - before):
            a, b = divmod(code, modulus)
            self.rec.contact_log.append((t, True, names[a], names[b]))
        for code in sorted(before - now):
            a, b = divmod(code, modulus)
            self.rec.contact_log.append((t, False, names[a], names[b]))
        self._touching, self._seen = now, (sid, t)
        return now != before

    def _fast(self, sid: int, smap: _SceneMap, data: mujoco.MjData, t: float, step_dt: float) -> bool:
        """Whether a drawn body moved faster than :data:`FAST_M_S` since the last control tick (the
        controller and the predicate look at the same tick: it counts once)."""
        if t < self._tick_t + 0.5 * step_dt:
            return False
        pos = data.xpos[smap.bodies]
        prev = self._prev_pos.get(sid)
        fast = False
        if prev is not None and self._tick_t > -math.inf:
            fast = float(np.max(np.linalg.norm(pos - prev, axis=1))) > FAST_M_S * (t - self._tick_t)
        self._prev_pos[sid] = pos
        self._tick_t = t
        return fast

    def sample(self, ctx: Any, data: mujoco.MjData, t_local: float, *, step_dt: float) -> None:
        """Take a frame at this tick when it is the first at or after the next frame time, also
        whenever a contact starts or stops (an impact lands between two frame times, and a pose
        interpolated across it would be wrong by millimetres) and at every tick for
        :data:`SETTLE_S` after a contact change, after the start of a segment and after a body moves
        faster than :data:`FAST_M_S` (a finger settling, a jaw snapping open: motion a frame rate of
        100 Hz does not follow). A fast move also takes back the ticks since the last frame, so that
        the interval it starts in is not a straight line either."""
        t = float(ctx.t0) + float(t_local)
        smap = self._map(ctx.scene)
        changed = self._watch_contacts(id(ctx.scene), smap, data, t, step_dt)
        scene = self._scenes.index(id(ctx.scene))
        if scene != self._scene_seen:
            self._scene_seen = scene
            self._pending = []
            self._settle_until = max(self._settle_until, t + SETTLE_S)
        fast = self._fast(id(ctx.scene), smap, data, t, step_dt)
        if changed or fast:
            self._settle_until = max(self._settle_until, t + SETTLE_S)
        due = t + 0.5 * step_dt >= self.next_t
        if self.rec.times and self.rec.scene_of_frame[-1] == scene and t < self.rec.times[-1] + 0.5 * step_dt:
            return  # this tick has its frame already
        poses = np.empty((len(smap.bodies), 7))
        poses[:, :3] = data.xpos[smap.bodies]
        poses[:, 3:] = _quat_xyzw(data.xquat[smap.bodies])
        if not (due or changed or t <= self._settle_until + 1e-9):
            if not (self._pending and t < self._pending[-1][0] + 0.5 * step_dt):
                self._pending.append((t, poses))
            return
        if fast:
            for t_before, before in self._pending:
                self._frame(t_before, scene, smap, before)
        self._pending = []
        self._frame(t, scene, smap, poses)
        if due:
            self.next_t = math.floor(t * self.rec.fps + 1.0 + 1e-6) / self.rec.fps

    def _frame(self, t: float, scene: int, smap: _SceneMap, poses: np.ndarray) -> None:
        self.rec.times.append(t)
        self.rec.frames.append(dict(zip(smap.keys, poses, strict=True)))
        self.rec.scene_of_frame.append(scene)


class _Controller:
    def __init__(self, inner: Any, sampler: _Sampler) -> None:
        self._inner, self._sampler = inner, sampler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def plan(self, ctx: Any, model: Any, data: Any) -> Any:
        return self._inner.plan(ctx, model, data)

    def command(self, t: float, ctx: Any, plan: Any, model: Any, data: Any) -> Any:
        self._sampler.sample(ctx, data, t, step_dt=model.opt.timestep)
        return self._inner.command(t, ctx, plan, model, data)


class _Predicate:
    def __init__(self, inner: Any, sampler: _Sampler) -> None:
        self._inner, self._sampler = inner, sampler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def observe(self, t: float, ctx: Any, model: Any, data: Any, acc: Any) -> Any:
        self._sampler.sample(ctx, data, t, step_dt=model.opt.timestep)
        return self._inner.observe(t, ctx, model, data, acc)

    def verdict(self, ctx: Any, acc: Any) -> Any:
        return self._inner.verdict(ctx, acc)


@contextlib.contextmanager
def _sampling(sampler: _Sampler) -> Iterator[None]:
    """Wrap the step controllers of :mod:`rtt.studio.episode` with ``sampler`` (in this process)."""
    from rtt.studio import episode

    original = episode.get_primitive

    def wrapped(kind: Any, controllers: Any = "default") -> tuple[Any, Any]:
        controller, predicate = original(kind, controllers)
        return _Controller(controller, sampler), _Predicate(predicate, sampler)

    episode.get_primitive = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        episode.get_primitive = original  # type: ignore[assignment]


def record_run(
    setup: Any,
    task_spec: Any,
    condition: Any,
    *,
    expected_traj_sha: str | None = None,
    fps: int = RECORD_FPS,
    task: Any = None,
    controllers: str = "default",
) -> Recording:
    """Re-run one scored run with the pose sampler on (module docstring).

    Raises:
        ReplayMismatchError: the run's ``traj_sha`` is not ``expected_traj_sha``.
    """
    import rtt.effectors.part_jaw  # noqa: F401 - registers the Studio build of part-shaped jaws
    from rtt.studio.episode import StudioJob, run_task

    rec = Recording(
        setup=setup,
        task_spec=task_spec,
        condition=condition,
        result=None,
        fps=int(fps),
        controllers=controllers,
        task=task,
    )
    sampler = _Sampler(rec)
    with _sampling(sampler):
        rec.result = run_task(StudioJob(setup, task_spec, condition, task=task, controllers=controllers))
    if expected_traj_sha is not None and rec.result.traj_sha != expected_traj_sha:
        raise ReplayMismatchError(
            f"setup {setup.setup_id[:12]} on {condition.condition_id}: the re-run gives traj_sha "
            f"{rec.result.traj_sha[:12]}, the report {expected_traj_sha[:12]}"
        )
    if not rec.frames:
        raise ReplayMismatchError("the run recorded no frames (no step ran)")
    return rec


# --------------------------------------------------------------------------- frames


def contact_times(log: Sequence[tuple[float, bool, str, str]], hold_s: float = CONTACT_HOLD_S) -> list[float]:
    """The times (sorted) at which a pair of bodies settles into or out of contact: the onset and
    the release of every contact that lasts at least ``hold_s``. Gaps shorter than ``hold_s``
    inside a contact do not release it, and a touch shorter than ``hold_s`` is no contact (the
    solver's chatter at a press or a grasp is hundreds of flickers). ``log`` is
    :attr:`Recording.contact_log`."""
    by_pair: dict[tuple[str, str], list[tuple[float, bool]]] = {}
    for t, touching, a, b in log:
        by_pair.setdefault((a, b) if a <= b else (b, a), []).append((float(t), bool(touching)))
    out: set[float] = set()
    for events in by_pair.values():
        spans: list[list[float]] = []
        began: float | None = None
        for t, touching in sorted(events):
            if touching:
                began = t if began is None else began
            elif began is not None:
                spans.append([began, t])
                began = None
        if began is not None:
            spans.append([began, math.inf])
        merged: list[list[float]] = []
        for span in spans:
            if merged and span[0] - merged[-1][1] < hold_s:
                merged[-1][1] = span[1]
            else:
                merged.append(span)
        for on, off in merged:
            if off - on < hold_s:
                continue
            out.add(on)
            if math.isfinite(off):
                out.add(off)
    return sorted(out)


def _blend(p0: np.ndarray, p1: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Poses ``(..., 7)`` a fraction ``a`` (``(..., 1)``-broadcastable) of the way from ``p0`` to
    ``p1``: linear position and the shortest-path spherical rotation (nlerp when nearly equal),
    the rule of the player's ``sample()`` (web/lib/replay.ts)."""
    a = np.asarray(a, dtype=np.float64)
    pos = p0[..., :3] + (p1[..., :3] - p0[..., :3]) * a
    q0, q1 = p0[..., 3:], p1[..., 3:]
    cos = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(cos < 0.0, -q1, q1)
    cos = np.abs(cos)
    near = cos > 0.9995
    theta = np.arccos(np.minimum(cos, 1.0))
    sin = np.where(near, 1.0, np.sin(theta))
    s0 = np.where(near, 1.0 - a, np.sin((1.0 - a) * theta) / sin)
    s1 = np.where(near, a, np.sin(a * theta) / sin)
    q = s0 * q0 + s1 * q1
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.concatenate([pos, q / np.where(norm > 0.0, norm, 1.0)], axis=-1)


def sample_lane(
    frames: np.ndarray, t: Sequence[float], cuts: Sequence[int], at: Sequence[float]
) -> np.ndarray:
    """The poses ``(len(at), n_bodies, 7)`` a player shows at times ``at`` for a keyframed lane
    (``frames`` ``(n, bodies, 7)`` at times ``t``): linear position and spherical rotation between
    the frames around each time, never across a cut (the frame before a cut holds until the cut
    frame), the first and last frames outside the lane's time span."""
    frames = np.asarray(frames, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    at = np.asarray(at, dtype=np.float64)
    last = len(t) - 1
    i0 = np.clip(np.searchsorted(t, at, side="right") - 1, 0, last)
    i1 = np.minimum(i0 + 1, last)
    span = t[i1] - t[i0]
    frac = np.clip(np.where(span > 0.0, (at - t[i0]) / np.where(span > 0.0, span, 1.0), 0.0), 0.0, 1.0)
    is_cut = np.zeros(len(t), dtype=bool)
    for c in cuts:
        if 0 < c <= last:
            is_cut[c] = True
    frac = np.where(is_cut[i1] & (i1 != i0), 0.0, frac)
    return _blend(frames[i0], frames[i1], frac[:, None, None])


_AXES_POINTS: Final = 0.1 * np.vstack([np.eye(3), -np.eye(3)])
_CHUNK: Final = 512
#: Directions whose support points stand in for a large hull while keys are being chosen (the
#: final check uses every vertex): a golden-spiral cover of the sphere.
_COARSE_POINTS: Final = 96


def _directions(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0**0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def _coarse(points: np.ndarray, n: int = _COARSE_POINTS) -> np.ndarray:
    """At most ``n`` of ``points``: the extreme point in each of ``n`` directions (all of them for
    a small set)."""
    if len(points) <= n:
        return points
    return points[np.unique(np.argmax(points @ _directions(n).T, axis=0))]


def _body_points(geoms: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """The vertices of a body that can be farthest from where it should be: the corners of its
    convex hull (a displacement is a norm of an affine map of the point, so convex)."""
    try:
        v = np.asarray(mesh_of_geoms(geoms).vertices, dtype=np.float64)
    except ValueError:
        return np.zeros((1, 3))
    try:
        from scipy.spatial import ConvexHull, QhullError

        return v[ConvexHull(v).vertices]
    except (QhullError, ValueError):
        return np.unique(v, axis=0)


class _Fit:
    """What an interpolated lane is measured against: ``N`` samples of ``G`` drawn bodies
    (``reps``) and the simulated poses of ``M`` keys (``truth``), each key drawn as its group's
    body composed with a fixed pose (``rel_pos``/``rel_rot``; a static key is drawn there in the
    world), and the vertices of every key that must stay within tolerance."""

    def __init__(
        self,
        times: np.ndarray,
        seg: np.ndarray,
        reps: np.ndarray,
        truth: np.ndarray,
        owner: np.ndarray,
        rel_pos: np.ndarray,
        rel_rot: np.ndarray,
        points: Sequence[np.ndarray],
    ) -> None:
        self.times, self.seg, self.reps, self.owner = times, seg, reps, owner
        self.rel_pos, self.rel_rot, self.points = rel_pos, rel_rot, list(points)
        self.truth_pos = truth[..., :3]
        self.truth_rot = _rot(truth[..., 3:])
        self.coarse = [_coarse(p) for p in self.points]

    @classmethod
    def of_tracks(
        cls,
        times: np.ndarray,
        seg: np.ndarray,
        tracks: np.ndarray,
        points: Sequence[np.ndarray] | None = None,
    ) -> _Fit:
        """Every track is its own drawn body and is checked at ``points`` (body frame)."""
        n_bodies = tracks.shape[1]
        pts = [np.asarray(p, dtype=np.float64) for p in points] if points is not None else None
        return cls(
            np.asarray(times, dtype=np.float64),
            np.asarray(seg),
            np.asarray(tracks, dtype=np.float64),
            np.asarray(tracks, dtype=np.float64),
            np.arange(n_bodies),
            np.zeros((n_bodies, 3)),
            np.broadcast_to(np.eye(3), (n_bodies, 3, 3)).copy(),
            pts if pts is not None else [_AXES_POINTS] * n_bodies,
        )

    def with_reps(self, reps: np.ndarray) -> _Fit:
        """The same measure of a lane whose drawn bodies follow ``reps`` instead."""
        other = copy.copy(self)
        other.reps = reps
        return other

    def errors(
        self,
        idx: np.ndarray,
        rep_pose: np.ndarray,
        *,
        statics: bool = False,
        coarse: bool = False,
        per_key: bool = False,
    ) -> np.ndarray:
        """Metres: for each sample in ``idx`` (``rep_pose`` ``(len(idx), G, 7)`` is what is drawn
        there) the largest distance between a vertex as drawn and as simulated, over the moving
        keys (the static ones too with ``statics``); per key with ``per_key``. ``coarse`` looks
        at a handful of extreme vertices per key instead of all of them (a quick under-estimate)."""
        keys = np.arange(len(self.owner)) if statics else np.flatnonzero(self.owner >= 0)
        points = self.coarse if coarse else self.points
        out = np.zeros((len(idx), len(keys)))
        for c0 in range(0, len(idx), _CHUNK):
            ii = idx[c0 : c0 + _CHUNK]
            rp = rep_pose[c0 : c0 + _CHUNK]
            rg = _rot(rp[..., 3:])
            for k, m in enumerate(keys):
                g = int(self.owner[m])
                if g >= 0:
                    pos = rp[:, g, :3] + rg[:, g] @ self.rel_pos[m]
                    rot = rg[:, g] @ self.rel_rot[m]
                else:
                    pos, rot = self.rel_pos[m], self.rel_rot[m]
                dpos = pos - self.truth_pos[ii, m]
                drot = rot - self.truth_rot[ii, m]
                disp = dpos[:, :, None] + drot @ points[m].T
                out[c0 : c0 + len(ii), k] = np.sqrt(np.max(np.sum(disp * disp, axis=1), axis=1))
        return out if per_key else out.max(axis=1, initial=0.0)

    def lane_errors(
        self, frames: np.ndarray, t: np.ndarray, cuts: Sequence[int], *, per_key: bool = False
    ) -> np.ndarray:
        """:meth:`errors` of a lane (``frames`` at times ``t``, cut at ``cuts``) at every sample,
        every vertex of every key, statics included."""
        idx = np.arange(len(self.times))
        return self.errors(idx, sample_lane(frames, t, cuts, self.times), statics=True, per_key=per_key)


def _forced(seg: np.ndarray, extra: Sequence[int] = ()) -> np.ndarray:
    """Samples a lane must keep: the first and the last of the run and of every segment, and
    ``extra``."""
    n = len(seg)
    keep = np.zeros(n, dtype=bool)
    keep[[0, n - 1]] = True
    edge = np.flatnonzero(seg[1:] != seg[:-1])
    keep[edge] = True
    keep[edge + 1] = True
    for i in extra:
        if 0 <= i < n:
            keep[i] = True
    return keep


def _split(fit: _Fit, keep: np.ndarray, tol_m: float, *, coarse: bool) -> np.ndarray:
    """Douglas-Peucker over every body at once: from the samples already kept (a mask), keep
    splitting a run of samples at the one the interpolation of its two ends is farthest from the
    simulation (over all bodies) until none is farther than ``tol_m``; never across a segment
    boundary. Returns the new mask."""
    keep = keep.copy()
    marks = np.flatnonzero(keep)
    stack = [(int(a), int(b)) for a, b in itertools.pairwise(marks) if fit.seg[a] == fit.seg[b]]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        idx = np.arange(a + 1, b)
        frac = (fit.times[idx] - fit.times[a]) / (fit.times[b] - fit.times[a])
        drawn = _blend(fit.reps[a][None], fit.reps[b][None], frac[:, None, None])
        err = fit.errors(idx, drawn, coarse=coarse)
        j = int(np.argmax(err))
        if err[j] > tol_m:
            keep[idx[j]] = True
            stack.append((a, int(idx[j])))
            stack.append((int(idx[j]), b))
    return keep


def _select(fit: _Fit, forced: np.ndarray, tol_m: float) -> np.ndarray:
    """The kept samples (indices, sorted): split on a few extreme vertices per body first (fast),
    then on every vertex from there (so the result is exact and little is re-done)."""
    return np.flatnonzero(_split(fit, _split(fit, forced, tol_m, coarse=True), tol_m, coarse=False))


def select_keys(
    times: Sequence[float],
    segment: Sequence[int],
    tracks: np.ndarray,
    *,
    points: Sequence[np.ndarray] | None = None,
    forced: Sequence[int] = (),
    tol_m: float = KEY_TOL_M,
) -> np.ndarray:
    """The samples (indices, sorted) a keyframed lane of ``tracks`` ``(N, bodies, 7)`` keeps, so
    that linear position and spherical rotation between them stay within ``tol_m`` of every track
    at every vertex in ``points`` (per body, in its frame; a cross of 10 cm by default). One list
    serves all bodies: a sample is kept when any body needs it. The first and last samples of the
    run and of every ``segment`` (ids per sample; a change of id is a cut), and ``forced``, are
    always kept."""
    seg = np.asarray(segment)
    fit = _Fit.of_tracks(np.asarray(times, dtype=np.float64), seg, tracks, points)
    return _select(fit, _forced(seg, forced), tol_m)


def _key_order(bodies: Mapping[str, _BodyInfo]) -> list[str]:
    rank = {"wrist": 0, "stack": 1, "tool": 2, "object": 3}
    return sorted(bodies, key=lambda k: (rank[bodies[k].category], bodies[k].tool or 0, k))


def _parked_tracks(rec: Recording, keys: Sequence[str], sample_idx: Sequence[int]) -> np.ndarray:
    """``(F, B, 7)`` poses: a body's recorded pose, or its tool-stand pose when its tool is off
    the wrist (tool changes)."""
    n_tools = len(rec.setup.tools)
    first: dict[int, int] = {}
    for i, _ in enumerate(rec.frames):
        tool = rec.scene_tools[rec.scene_of_frame[i]]
        first.setdefault(tool, i)
    stands = {
        k: np.array([(k - (n_tools - 1) / 2.0) * STAND_PITCH_M, STAND_Y_M, STAND_Z_M, 0.0, 0.0, 0.0, 1.0])
        for k in range(n_tools)
    }
    sel = [rec.frames[i] for i in sample_idx]
    out = np.empty((len(sel), len(keys), 7))
    for b, key in enumerate(keys):
        here = [i for i, frame in enumerate(sel) if key in frame]
        if len(here) == len(sel):
            out[:, b] = np.stack([frame[key] for frame in sel])
            continue
        info = rec.bodies[key]
        if info.tool is None or info.tool not in first:
            raise ReplayMismatchError(f"body {key} is missing from a frame and is not a parked tool")
        f0 = rec.frames[first[info.tool]]
        pos, rot = _relative(f0["wrist_flange"], f0[key])
        out[:, b] = _compose(stands[info.tool], pos, rot)
        if here:
            out[here, b] = np.stack([sel[i][key] for i in here])
    return out


@dataclass
class _Group:
    rep: int
    members: list[int]


def _groups(
    tracks: np.ndarray, keys: Sequence[str], bodies: Mapping[str, _BodyInfo]
) -> tuple[list[_Group], list[int]]:
    """Rigid groups of bodies (same category and drawing), and the bodies that never move."""
    static = [b for b in range(len(keys)) if _static(tracks[:, b])]
    groups: list[_Group] = []
    for b in range(len(keys)):
        if b in static:
            continue
        info = bodies[keys[b]]
        for g in groups:
            gi = bodies[keys[g.rep]]
            if (gi.category, gi.tool) != (info.category, info.tool):
                continue
            if _rigid(tracks[:, g.rep], tracks[:, b]):
                g.members.append(b)
                break
        else:
            groups.append(_Group(b, [b]))
    return groups, static


# --------------------------------------------------------------------------- assets


def staging_store(raw_dir: Path) -> Any:
    """A scratch :class:`~rtt.render.glb.AssetStore` for the raw GLBs of an export (served by
    :func:`serve_assets`, written to the site by :func:`publish_assets`)."""
    from rtt.render.glb import AssetStore

    raw_dir.mkdir(parents=True, exist_ok=True)
    return AssetStore(raw_dir)


def object_look(template: str | None, part: str = "") -> str:
    """The :data:`~rtt.render.materials.PART_LOOKS` name an object body is drawn in: its own
    (:data:`_LOOKS_BY_PART`), else its template's (:data:`_LOOKS_BY_KIND`), else clay."""
    key = template or ""
    return _LOOKS_BY_PART.get((key, part)) or _LOOKS_BY_KIND.get(key, "clay")


def _look(info: _BodyInfo) -> Any:
    from rtt.render import materials as mat

    if info.category == "object":
        return mat.PART_LOOKS[object_look(info.template, info.part)]
    if info.category == "tool":
        return mat.FINGER_DESIGNED
    if info.category == "wrist":
        return mat.CARRIAGE
    return mat.FRAME


#: From the most to the least demanding (the build pack's rule): a body made from several of a
#: tool's sources carries the first of its licences in this order.
_LICENCE_ORDER: Final = ("Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MIT", "CC-BY-4.0", "CC0-1.0")
THIRD_PARTY_CHANGES: Final = (
    "the simulation model's collision geometry only, as the simulator used it",
    "re-posed into the replay body frame",
    "re-encoded as GLB",
)


def mesh_of_geoms(geoms: Sequence[Mapping[str, Any]], segments: int = GLB_SEGMENTS) -> Any:
    """One triangle mesh of a body's replay geoms in its frame: each primitive tessellated and
    each convex mesh piece taken as compiled, then concatenated (overlapping solids draw the same
    as their union). Unlike a boolean union (:func:`rtt.render.glb.mesh_from_geoms`, whose output
    order varies from call to call), the same geoms always give the same triangles, so a body's
    GLB, and so its content-addressed id, is the same on every export."""
    import trimesh

    parts = []
    for g in geoms:
        kind = g["type"]
        if kind == "box":
            half = np.asarray(g["half_size"], dtype=np.float64)
            if np.any(half <= 0):
                continue
            m = trimesh.creation.box(extents=2.0 * half)
        elif kind == "cylinder":
            m = trimesh.creation.cylinder(
                radius=float(g["radius"]), height=2.0 * float(g["half_height"]), sections=segments
            )
        elif kind == "capsule":
            r, hl = float(g["radius"]), float(g["half_length"])
            if hl <= 0.0:
                m = trimesh.creation.uv_sphere(radius=r, count=[segments, segments // 2])
            else:
                m = trimesh.creation.capsule(height=2.0 * hl, radius=r, count=[segments, segments // 2])
        elif kind == "mesh":
            v = np.asarray(g["vertices"], dtype=np.float64).reshape(-1, 3)
            f = np.asarray(g["faces"], dtype=np.int64).reshape(-1, 3)
            if len(v) < 4 or len(f) < 4:
                continue
            m = trimesh.Trimesh(v, f, process=False)
        else:
            continue
        tf = np.eye(4)
        tf[:3, :3] = _rot(np.asarray(g["quat_xyzw"], dtype=np.float64))
        tf[:3, 3] = np.asarray(g["pos"], dtype=np.float64)
        m.apply_transform(tf)
        parts.append(m)
    if not parts:
        raise ValueError("no solid geoms to mesh")
    joined = trimesh.util.concatenate(parts)
    return trimesh.Trimesh(np.asarray(joined.vertices), np.asarray(joined.faces), process=False)


def _licence(licences: Sequence[str]) -> str:
    ranked = [lic for lic in _LICENCE_ORDER if lic in licences]
    return ranked[0] if ranked else min(licences)


def credit_line(info: _BodyInfo, family_title: str) -> str:
    """The brand-free credit a third-party body's asset carries (full credit on the credits page)."""
    from rtt.render.glb import GRIPPER_ATTRIBUTION

    if info.anchor == "gripper":
        return GRIPPER_ATTRIBUTION
    return (
        f"Simulation model of an open-source {family_title.lower()}, {_licence(info.licences)}, modified "
        f"(full credit: /credits#{info.anchor or 'effectors'})"
    )


def _body_asset(store: Any, info: _BodyInfo, family_title: str = "") -> str | None:
    """The GLB asset of one body, made from its collision geoms (``None`` when they do not mesh).
    Our own geometry is Apache-2.0; a third-party tool body carries its model's licence, a
    brand-free credit line and the list of changes."""
    from rtt.pipeline.replays import _own_asset

    if info.asset is not ...:
        return info.asset
    try:
        mesh = mesh_of_geoms(info.geoms)
    except Exception:  # noqa: BLE001 - a body that does not mesh is drawn from its geoms
        info.asset = None
        return None
    role = "object" if info.category == "object" else "gripper_frame"
    what = {"object": "task object", "tool": "tool body", "wrist": "wrist flange", "stack": "tool stack"}[
        info.category
    ]
    extra: dict[str, Any] = {}
    if info.third_party:
        extra = {
            "license": _licence(info.licences),
            "attribution": credit_line(info, family_title),
            "changes": list(THIRD_PARTY_CHANGES),
        }
    info.asset = _own_asset(store, mesh, role, _look(info), f"Studio {what}: {info.label}", **extra)
    return info.asset


def _stand_asset(store: Any, n_tools: int) -> str:
    from rtt.pipeline.replays import _own_asset
    from rtt.render import materials as mat

    width = STAND_PITCH_M * max(1, n_tools - 1) + 0.12
    geoms = [
        {
            "type": "box",
            "half_size": [width / 2, 0.012, 0.012],
            "pos": [0.0, 0.0, 0.0],
            "quat_xyzw": [0, 0, 0, 1],
        },
        *[
            {
                "type": "box",
                "half_size": [0.012, 0.012, (STAND_Z_M + 0.03) / 2],
                "pos": [sx * width / 2, 0.0, -(STAND_Z_M + 0.03) / 2],
                "quat_xyzw": [0, 0, 0, 1],
            }
            for sx in (-1.0, 1.0)
        ],
    ]
    return _own_asset(
        store, mesh_of_geoms(geoms), "prop", mat.FRAME, "Tool stand (display only, not simulated)"
    )


def _move_geom(geom: Mapping[str, Any], pos: np.ndarray, rot: np.ndarray) -> dict[str, Any]:
    """``geom`` (in a member's frame) placed in its group's frame (member at ``pos``/``rot``)."""
    g = dict(geom)
    gp = np.asarray(geom["pos"], dtype=np.float64)
    gr = _rot(np.asarray(geom["quat_xyzw"], dtype=np.float64))
    g["pos"] = [round(float(v), 6) for v in pos + rot @ gp]
    g["quat_xyzw"] = [round(float(v), 6) for v in _mat_quat(rot @ gr)]
    return g


# --------------------------------------------------------------------------- the lane


@dataclass
class LaneBuild:
    """One lane before its assets are served: the replay, the page block, the raw assets it draws."""

    replay: Any
    doc: dict[str, Any]
    assets: set[str]
    stem: str
    #: The run the lane was made from (the replay gate re-runs and measures it).
    recording: Recording | None = None


def _segments(rec: Recording) -> list[dict[str, Any]]:
    setup, spec, result = rec.setup, rec.task_spec, rec.result
    outcomes = {o.step_id: o for o in result.steps}
    out = []
    prev_end = 0.0
    for i, seg in enumerate(setup.segments(spec)):
        steps = [outcomes[s] for s in seg.step_ids]
        ran = [o for o in steps if o.failure_mode != "not_reached"]
        if not ran:
            break
        start = float(ran[0].start_s)
        end = max(float(o.start_s + o.duration_s) for o in ran)
        out.append(
            {
                "index": i,
                "tool": int(seg.tool),
                "tool_label": setup.tools[seg.tool].display_name,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "change": seg.change.value if hasattr(seg.change, "value") else str(seg.change),
                "change_s": round(float(seg.change_s), 3),
                "change_start_s": round(prev_end, 3),
                "steps": [o.step_id for o in ran],
            }
        )
        prev_end = end
    return out


def _steps(rec: Recording) -> list[dict[str, Any]]:
    out = []
    for o in rec.result.steps:
        if o.failure_mode == "not_reached":
            continue
        out.append(
            {
                "step_id": o.step_id,
                "primitive": o.primitive.value if hasattr(o.primitive, "value") else str(o.primitive),
                "tool": int(o.tool),
                "start_s": round(float(o.start_s), 3),
                "duration_s": round(float(o.duration_s), 3),
                "success": bool(o.success),
                "failure_mode": o.failure_mode,
                "fidelity": o.fidelity.value if hasattr(o.fidelity, "value") else str(o.fidelity),
                "quantities": {
                    k: round(float(v), 3) for k, v in sorted(o.quantities.items()) if math.isfinite(float(v))
                },
            }
        )
    return out


def _credits(rec: Recording) -> list[dict[str, Any]]:
    """The lane's third-party credits (brand-free text and the credits page anchor), once each."""
    import re

    from rtt.effectors.lanes import FAMILY_TITLES

    seen: dict[str, dict[str, Any]] = {}
    for info in rec.bodies.values():
        if not info.third_party or info.tool is None:
            continue
        spec = rec.setup.tools[info.tool]
        line = credit_line(info, FAMILY_TITLES.get(spec.family, spec.family))
        m = re.search(r"\s*\(full credit: /credits#([A-Za-z0-9_-]+)\)\s*$", line)
        text, anchor = (line[: m.start()], m.group(1)) if m else (line, info.anchor or "effectors")
        seen.setdefault(text, {"text": text, "anchor": anchor})
    return list(seen.values())


# --------------------------------------------------------------------------- lane files


#: Pose fields of a quantised lane (``quantised`` true): the position, the three quaternion
#: components left when the largest is dropped, and the index of the dropped one (0 to 3 for
#: x, y, z, w). Seven int16 words a pose, 14 bytes.
QUANT_POSE_FIELDS: Final = ("px", "py", "pz", "qa", "qb", "qc", "qi")
#: Metres per position word of a quantised lane (0.1 mm); a word holds +-3.2767 m.
QUANT_POS_M: Final = 1e-4
#: A kept quaternion component is at most 1/sqrt(2); words hold round(c / QUANT_QUAT_SCALE * 32767).
QUANT_QUAT_SCALE: Final = 0.70710678
_INT16_MAX: Final = 32767
#: The header keys a lane may carry beyond the replay v2 header's own (``src/rtt/sim/replay.py`` is
#: frozen, so these are checked and written here).
_LANE_KEYS: Final = ("t", "cuts", "quantised")


class QuantiseRangeError(ValueError):
    """A position is beyond what an int16 lane holds (+-3.2767 m): that candidate is skipped."""


@dataclass
class LaneReplay(rp.Replay):
    """A Studio lane: a replay v2 whose header may carry ``t``, ``cuts`` and ``quantised``.
    ``frames`` is always float32 ``(n_frames, n_bodies, 7)``; for a quantised lane ``words`` holds
    the int16 poses ``(n_frames, n_bodies, 7)`` that :func:`write_lane` writes as they are
    (``frames`` must be what they decode to; without ``words`` the writer quantises ``frames``)."""

    words: np.ndarray | None = None


def quantise_poses(poses: np.ndarray) -> np.ndarray:
    """Int16 poses ``(..., 7)`` (px, py, pz, qa, qb, qc, qi) of float poses ``(..., 7)`` (px, py,
    pz, qx, qy, qz, qw). The position is round(mm * 10). The quaternion is normalised and its
    component of largest magnitude, at index ``qi`` (0 to 3 for x, y, z, w), is dropped after
    flipping the sign so that it is positive; the other three, in x, y, z, w order with ``qi``
    skipped, are round(c / 0.70710678 * 32767).

    Raises:
        ValueError: a pose is NaN or infinite (a diverged simulation: nothing to quantise).
        QuantiseRangeError: a position is outside +-3.2767 m (it would wrap)."""
    f = np.asarray(poses, dtype=np.float64)
    if f.shape[-1] != len(rp.POSE_FIELDS):
        raise ValueError(f"poses must end in {len(rp.POSE_FIELDS)} values, got shape {f.shape}")
    if not np.all(np.isfinite(f)):
        raise ValueError("poses must be finite: a pose is NaN or infinite")
    pos = np.rint(f[..., :3] / QUANT_POS_M)
    if np.any(np.abs(pos) > _INT16_MAX):
        raise QuantiseRangeError(
            f"a position is outside +-{_INT16_MAX * QUANT_POS_M:.4f} m (3.27 m), which an int16 lane cannot hold"
        )
    q = f[..., 3:]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(norm > 0, norm, 1.0)
    qi = np.argmax(np.abs(q), axis=-1)
    sign = np.where(np.take_along_axis(q, qi[..., None], axis=-1) < 0, -1.0, 1.0)
    keep = np.arange(3) + (np.arange(3) >= qi[..., None])  # x, y, z, w without the dropped one
    rest = np.take_along_axis(q, keep, axis=-1) * sign
    out = np.empty((*f.shape[:-1], len(QUANT_POSE_FIELDS)), dtype=np.int16)
    out[..., :3] = pos
    out[..., 3:6] = np.clip(np.rint(rest / QUANT_QUAT_SCALE * _INT16_MAX), -_INT16_MAX, _INT16_MAX)
    out[..., 6] = qi
    return out


def dequantise_poses(words: np.ndarray) -> np.ndarray:
    """Float32 poses ``(..., 7)`` of the int16 poses :func:`quantise_poses` makes: the three
    stored components times 0.70710678 / 32767, the dropped one sqrt(max(0, 1 - sum of squares))."""
    w = np.asarray(words)
    if w.shape[-1] != len(QUANT_POSE_FIELDS):
        raise ValueError(f"words must end in {len(QUANT_POSE_FIELDS)} values, got shape {w.shape}")
    qi = w[..., 6].astype(np.int64)
    if np.any((qi < 0) | (qi > 3)):
        raise ValueError("the index of a dropped quaternion component (the last word) is 0 to 3")
    out = np.empty((*w.shape[:-1], 7), dtype=np.float64)
    out[..., :3] = w[..., :3].astype(np.float64) * QUANT_POS_M
    comps = w[..., 3:6].astype(np.float64) * (QUANT_QUAT_SCALE / _INT16_MAX)
    rest = np.sqrt(np.maximum(0.0, 1.0 - np.sum(comps * comps, axis=-1)))
    slot = np.arange(4)
    src = slot - (slot > qi[..., None])  # which stored component fills slot j
    q = np.take_along_axis(comps, np.minimum(src, 2), axis=-1)
    out[..., 3:] = np.where(slot == qi[..., None], rest[..., None], q)
    return out.astype(np.float32)


def _lane_errors(header: Mapping[str, Any]) -> list[str]:
    """The rules of a lane's ``t``, ``cuts`` and ``quantised`` (all optional)."""
    errs: list[str] = []
    quantised = header.get("quantised", False)
    if not isinstance(quantised, bool):
        errs.append("quantised must be true or false")
        quantised = False
    if quantised:
        if header.get("dtype") != "int16":
            errs.append("dtype must be 'int16' (a quantised lane)")
        if header.get("pose_fields") != list(QUANT_POSE_FIELDS):
            errs.append(f"pose_fields must be {list(QUANT_POSE_FIELDS)!r} (a quantised lane)")
    n = header["n_frames"] if isinstance(header["n_frames"], int) else 0
    if "t" in header:
        t = header["t"]
        numbers = isinstance(t, list) and all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in t
        )
        if not numbers:
            errs.append("t must be a list of finite numbers")
        elif len(t) != n:
            errs.append(f"t needs one time per frame ({n}), got {len(t)}")
        elif t and (t[0] < 0 or any(b <= a for a, b in itertools.pairwise(t))):
            errs.append("t must start at 0 or later and strictly increase")
    if "cuts" in header:
        cuts = header["cuts"]
        if (
            not isinstance(cuts, list)
            or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c < max(n, 1) for c in cuts)
            or any(b <= a for a, b in itertools.pairwise(cuts))
        ):
            errs.append(f"cuts must be strictly increasing frame indices in 0..{max(n - 1, 0)}")
    return errs


def check_lane_header(header: Mapping[str, Any], frames: np.ndarray | None = None) -> list[str]:
    """The problems of a lane header: the replay v2 rules of
    :func:`rtt.sim.replay.check_header` (applied to the header as a float32 lane without the
    lane keys) and those of ``t``, ``cuts`` and ``quantised``. Headers of today's uniform float32
    lanes pass as before."""
    base = {k: v for k, v in header.items() if k not in _LANE_KEYS}
    if header.get("quantised") is True:  # int16 poses: the rest of the header is a float32 lane's
        base["dtype"], base["pose_fields"] = "float32", list(rp.POSE_FIELDS)
    errs = rp.check_header(base, frames)
    if not any(e.startswith("missing keys") for e in errs):
        errs += _lane_errors(header)
    return errs


def write_lane(replay: rp.Replay, stem: str | Path) -> tuple[Path, Path]:
    """Write ``<stem>.bin`` and ``<stem>.json`` of a lane (:class:`LaneReplay`, or any v2 replay
    without overlays); returns the pose and header paths. The header is checked first
    (:func:`check_lane_header`) and nothing is written if it breaks a rule; it is written as
    compact JSON. A header with ``quantised`` true writes int16 poses (:func:`quantise_poses`)."""
    stem = Path(stem)
    bin_path, json_path = stem.with_suffix(".bin"), stem.with_suffix(".json")
    frames = np.ascontiguousarray(replay.frames, dtype="<f4")
    header = dict(replay.header, bin=bin_path.name, n_frames=int(frames.shape[0]))
    quantised = header.get("quantised") is True
    words = getattr(replay, "words", None)
    errs = check_lane_header(header, frames)
    if replay.overlays is not None or "overlays" in header:
        errs.append("a Studio lane has no overlays")
    if words is not None:
        if not quantised:
            errs.append("int16 words need a header with quantised true")
        elif tuple(words.shape) != (*frames.shape[:2], len(QUANT_POSE_FIELDS)) or not np.array_equal(
            dequantise_poses(words), frames
        ):
            errs.append("the int16 words are not what the frames decode to")
    if errs:
        raise ValueError(f"{stem.name}: replay v2 header is invalid: " + "; ".join(errs))
    if quantised and words is None:
        words = quantise_poses(frames)
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = np.ascontiguousarray(words, dtype="<i2") if quantised else frames
    bin_path.write_bytes(payload.tobytes())
    json_path.write_text(
        json.dumps(header, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bin_path, json_path


def read_lane(stem: str | Path) -> LaneReplay:
    """Load a lane written by :func:`write_lane` (or any replay v2 or v1 files): ``frames`` is
    float32 ``(n_frames, n_bodies, 7)`` whether the poses were stored as float32 or int16."""
    stem = Path(stem)
    header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if header.get("version") == rp.VERSION and header.get("quantised") is True:
        words = np.frombuffer(stem.with_suffix(".bin").read_bytes(), dtype="<i2")
        words = words.reshape(header["n_frames"], header["n_bodies"], len(QUANT_POSE_FIELDS))
        return LaneReplay(frames=dequantise_poses(words), header=header, words=words)
    base = rp.read_replay(stem)
    return LaneReplay(frames=base.frames, header=base.header, overlays=base.overlays)


def lane_bytes(replay: Any) -> int:
    """Bytes a lane serves: its header as written (compact JSON and a newline) and its poses."""
    h = replay.header
    words = 7 * 2 if h.get("quantised") else 7 * 4  # int16 or float32 words of a pose
    return (
        len(json.dumps(h, separators=(",", ":"), ensure_ascii=False))
        + 1
        + int(h["n_frames"]) * int(h["n_bodies"]) * words
    )


def _sample_segments(rec: Recording) -> np.ndarray:
    """The segment (and so the cut) every recorded sample belongs to: it changes at the first
    sample of each of the run's segments (:func:`_segments`) and wherever the scene changes."""
    times = np.asarray(rec.times)
    seg = np.zeros(len(times), dtype=int)
    for s in _segments(rec)[1:]:
        seg += times >= s["start_s"] - 5e-4  # start_s is rounded to the millisecond
    scene = np.asarray(rec.scene_of_frame)
    return np.cumsum(np.r_[False, (seg[1:] != seg[:-1]) | (scene[1:] != scene[:-1])])


def _fit_of(
    rec: Recording,
    keys: Sequence[str],
    tracks: np.ndarray,
    groups: Sequence[_Group],
    static: Sequence[int],
) -> _Fit:
    """The lane as it is drawn (every key as its group's body plus a fixed offset, statics where
    they rest) measured against the recording, at the vertices of every key."""
    owner = np.full(len(keys), -1, dtype=int)
    rel_pos = np.zeros((len(keys), 3))
    rel_rot = np.broadcast_to(np.eye(3), (len(keys), 3, 3)).copy()
    for gi, g in enumerate(groups):
        for m in g.members:
            owner[m] = gi
            rel_pos[m], rel_rot[m] = _relative(tracks[0, g.rep], tracks[0, m])
    for m in static:
        rel_pos[m], rel_rot[m] = tracks[0, m, :3], _rot(tracks[0, m, 3:])
    reps = np.stack([tracks[:, g.rep] for g in groups], axis=1) if groups else np.zeros((len(tracks), 0, 7))
    return _Fit(
        np.asarray(rec.times, dtype=np.float64),
        _sample_segments(rec),
        reps,
        tracks,
        owner,
        rel_pos,
        rel_rot,
        [_body_points(rec.bodies[k].geoms) for k in keys],
    )


def _forced_samples(rec: Recording, fit: _Fit) -> np.ndarray:
    """The samples a lane must keep: run and segment ends, and the two samples around every
    contact onset and release (:func:`contact_times`)."""
    extra: list[int] = []
    for when in contact_times(rec.contact_log):
        j = int(np.searchsorted(fit.times, when - 1e-9, side="left"))
        extra += [j - 1, j]
    return _forced(fit.seg, extra)


@dataclass
class _Keyed:
    keys: np.ndarray  # recorded sample of each frame
    frames: np.ndarray  # float32 (n, bodies, 7), as a player reads them back
    t: np.ndarray
    cuts: list[int]
    words: np.ndarray | None  # the int16 poses of a quantised lane: what is written, bit for bit
    worst_m: float


def _key_lane(
    fit: _Fit,
    chosen: dict[tuple[float, bool], np.ndarray],
    forced: np.ndarray,
    tol_m: float,
    quantised: bool,
) -> _Keyed:
    """The frames of a lane that stays within ``tol_m`` of the simulation, measured as a player
    reads them: with the poses rounded to int16 when ``quantised`` (the keys are chosen against
    the rounded poses, so what rounding costs is spent inside the tolerance).

    Raises:
        ReplayMismatchError: the lane cannot be held within ``tol_m`` (the bodies that are drawn
            as one, or static ones, are already that far from the simulation).
    """
    use, words = fit, None
    if quantised:
        words = quantise_poses(fit.reps)
        use = fit.with_reps(dequantise_poses(words).astype(np.float64))
    if (tol_m, quantised) not in chosen:
        chosen[(tol_m, quantised)] = _select(use, forced, tol_m)
    keys = chosen[(tol_m, quantised)]
    frames = use.reps[keys].astype(np.float32)
    t = fit.times[keys]
    cuts = [i for i in range(1, len(keys)) if fit.seg[keys[i]] != fit.seg[keys[i - 1]]]
    worst = float(use.lane_errors(frames, t, cuts).max())
    if worst > tol_m:
        raise ReplayMismatchError(
            f"no frames hold the lane within {1000 * tol_m:.2f} mm: its bodies are {1000 * worst:.3f} mm off "
            f"at the keys themselves"
        )
    return _Keyed(
        keys,
        frames,
        t,
        cuts,
        None if words is None else words[keys],
        worst,
    )


def build_lane(
    rec: Recording,
    store: Any,
    *,
    hero: str,
    lane: str,
    code_version: str,
    budget: int = LANE_BUDGET_BYTES,
) -> tuple[LaneReplay, set[str], int]:
    """The replay v2 of a recording (module docstring): keyframes within :data:`KEY_TOL_M` of the
    recording at every vertex, float32 poses when they fit ``budget``, else int16, and only when
    those do not fit either the tolerance relaxed along :data:`KEY_TOLERANCES_M` (2 mm at most).
    Returns the replay, the raw asset ids it draws and the rate the recording was taken at
    (``rec.fps``, which the replay's ``fps`` also states).

    Raises:
        ReplayMismatchError: (naming ``hero``/``lane``) the recording holds a non-finite pose, or
            no lane within 2 mm fits ``budget`` (a body beyond +-3.27 m rules int16 out).
    """
    keys = _key_order(rec.bodies)
    tracks = _parked_tracks(rec, keys, range(len(rec.frames)))
    if not np.all(np.isfinite(tracks)):
        raise ReplayMismatchError(
            f"lane {hero}/{lane}: the recording holds a pose that is NaN or infinite (the simulation diverged)"
        )
    groups, static = _groups(tracks, keys, rec.bodies)
    fit = _fit_of(rec, keys, tracks, groups, static)
    forced = _forced_samples(rec, fit)
    base, used = _lane_header(
        rec, store, keys, tracks, groups, static, hero=hero, lane=lane, code_version=code_version
    )
    chosen: dict[tuple[float, bool], np.ndarray] = {}
    smallest: tuple[int, float, bool] | None = None  # bytes, tolerance and encoding of the smallest lane
    reasons: list[str] = []
    int16_range = False
    for tol in KEY_TOLERANCES_M:
        for quantised in (False, True):
            try:
                keyed = _key_lane(fit, chosen, forced, tol, quantised)
            except QuantiseRangeError:
                int16_range = True  # this candidate cannot be made: on down the ladder
                continue
            except ReplayMismatchError as exc:
                reasons.append(str(exc))
                continue
            replay = _lane_replay(base, keyed, tol, len(rec.frames), rec.fps)
            size = lane_bytes(replay)
            if size <= budget:
                return replay, used, int(rec.fps)
            if smallest is None or size < smallest[0]:
                smallest = (size, tol, quantised)
    if smallest is None:
        raise ReplayMismatchError(
            f"lane {hero}/{lane}: {reasons[-1] if reasons else 'no lane could be made'}"
        )
    size, tol, quantised = smallest
    why = (
        "; int16 poses are out of range (a body is beyond +-3.27 m), so only float32 was tried"
        if int16_range
        else ""
    )
    raise ReplayMismatchError(
        f"lane {hero}/{lane} does not fit {budget} bytes: the smallest is {size} bytes "
        f"({'int16' if quantised else 'float32'} poses) at {1000 * tol:.1f} mm, which with "
        f"{1000 * KEY_TOLERANCES_M[-1]:.1f} mm is the loosest tolerance a lane may have{why}"
    )


def _lane_replay(
    base: Mapping[str, Any], keyed: _Keyed, tol_m: float, n_samples: int, recorded_fps: int
) -> Any:
    """The replay of one set of keyed frames: the lane's header with the timeline, the cuts and
    the pose encoding."""
    header = dict(base)
    meta = dict(base["meta"])
    meta["keyframes"] = {
        "recorded_fps": int(recorded_fps),
        "samples": int(n_samples),
        "tolerance_mm": round(1000 * tol_m, 3),
        # against the recording (the samples the keys were chosen from), not the simulation: the
        # replay gate measures that one (rtt.studio.replay_qa)
        "max_error_vs_recording_mm": math.ceil(1000 * keyed.worst_m * 1000) / 1000,
    }
    header.update(
        fps=int(recorded_fps),
        n_frames=len(keyed.keys),
        t=[round(float(x), 6) for x in keyed.t],
        cuts=list(keyed.cuts),
        quantised=keyed.words is not None,
        meta=meta,
    )
    if keyed.words is not None:
        header.update(dtype="int16", pose_fields=list(QUANT_POSE_FIELDS))
    return LaneReplay(frames=np.ascontiguousarray(keyed.frames), header=header, words=keyed.words)


def _lane_header(
    rec: Recording,
    store: Any,
    keys: Sequence[str],
    tracks: np.ndarray,
    groups: Sequence[_Group],
    static: Sequence[int],
    *,
    hero: str,
    lane: str,
    code_version: str,
) -> tuple[dict[str, Any], set[str]]:
    """Everything of a lane's header that does not depend on which frames are kept: the bodies
    and their assets, the static scene and the meta and provenance."""
    used: set[str] = set()
    bodies = []
    names: set[str] = set()

    from rtt.effectors.lanes import FAMILY_TITLES

    def visual_of(b: int, pos: np.ndarray, rot: np.ndarray) -> dict[str, Any] | None:
        info = rec.bodies[keys[b]]
        family = rec.setup.tools[info.tool].family if info.tool is not None else ""
        aid = _body_asset(store, info, FAMILY_TITLES.get(family, family))
        if aid is None:
            return None
        used.add(aid)
        return rp.visual(aid, pos, _mat_quat(rot))

    for g in groups:
        rep_info = rec.bodies[keys[g.rep]]
        base = {"tool": "finger", "object": "object", "wrist": "palm", "stack": "palm"}[rep_info.category]
        slug = keys[g.rep].replace("/", "_")
        name = f"{base}_{slug}"
        while name in names:
            name += "_"
        names.add(name)
        visuals, geoms = [], []
        for m in g.members:
            pos, rot = _relative(tracks[0, g.rep], tracks[0, m])
            v = visual_of(m, pos, rot)
            if v is not None:
                visuals.append(v)
            geoms += [_move_geom(geom, pos, rot) for geom in rec.bodies[keys[m]].geoms]
        body: dict[str, Any] = {"name": name, "visuals": visuals}
        if len(visuals) != len(g.members):
            # a member that does not mesh: draw the whole group from its geoms (a body with
            # visuals does not draw its geoms)
            body["visuals"] = []
            body["geoms"] = geoms
        bodies.append(body)
    static_visuals = []
    static_geoms = [dict(g) for g in rec.world_static]
    for b in static:
        info = rec.bodies[keys[b]]
        pose = tracks[0, b]
        v = visual_of(b, pose[:3], _rot(pose[3:]))
        if v is not None:
            static_visuals.append(v)
        else:
            static_geoms += [_move_geom(g, pose[:3], _rot(pose[3:])) for g in info.geoms]
    n_tools = len(rec.setup.tools)
    if rec.setup.kind.value == "tool_change" and n_tools > 1:
        static_visuals.append(rp.visual(_stand_asset(store, n_tools), (0.0, STAND_Y_M, STAND_Z_M + 0.024)))
    used = {v["asset"] for b in bodies for v in b["visuals"]} | {v["asset"] for v in static_visuals}
    result = rec.result
    setup = rec.setup
    meta = {
        "design_space": f"studio-{setup.kind.value}",
        "design_id": setup.setup_id,
        "geometry_id": None,
        "part_id": hero,
        "label": PRODUCT_MODE_LABEL,
        "lane": lane,
        "setup_label": setup.label,
        "task_id": rec.task_spec.task_id,
        "traj_sha": result.traj_sha,
        "condition_id": rec.condition.condition_id,
        "seed": int(rec.condition.seed),
        "completed": bool(result.completed),
        "cycle_time_s": round(float(result.cycle_time_s), 3),
        "segments": _segments(rec),
        "steps": _steps(rec),
        "tool_stand": "display only: a tool that is not on the wrist rests here; nothing about it is simulated",
    }
    prov = {
        "source": "simulation",
        "engine": "mujoco",
        "engine_version": mujoco.__version__,
        "run_id": f"studio-{hero}-{setup.setup_id[:12]}",
        "code_version": code_version,
        "product_mode": True,
        "episode": "studio full-task run",
        "suite": "studio-heroes",
        "condition": rec.condition.condition_id,
        "seed": int(rec.condition.seed),
        "task_sha256": rec.task_spec.task_id,
        "record_version": 1,
    }
    header: dict[str, Any] = {
        "format": rp.FORMAT,
        "version": rp.VERSION,
        "fps": int(rec.fps),
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": list(rp.POSE_FIELDS),
        "up_axis": "z",
        "units": "m",
        "n_frames": 0,
        "n_bodies": len(bodies),
        "bin": f"{lane}.bin",
        "assets": store.subset(used),
        "bodies": bodies,
        "static": static_geoms,
        "meta": meta,
        "provenance": prov,
    }
    if static_visuals:
        header["static_visuals"] = static_visuals
    return header, used


# --------------------------------------------------------------------------- lanes of a report


def lane_choices(report: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(lane id, title, setup_id)``: the recommendation (or the closest setup) and the best
    single off-the-shelf baseline (once when they are the same setup)."""
    rec = report["recommendation"]
    out: list[tuple[str, str, str]] = []
    r = rec.get("recommendation")
    if r:
        out.append(("recommended", "Recommended", r["setup_id"]))
    else:
        closest = [
            c
            for c in (rec.get("no_winner") or {}).get("closest") or []
            if c["setup_id"] != report.get("best_single")
        ]
        if closest:
            out.append(("closest", "Closest setup", closest[0]["setup_id"]))
    best = report.get("best_single")
    if best and all(sid != best for _, _, sid in out):
        out.append(("best_single", "Best single off-the-shelf", best))
    return out


def _condition_text(cond: Any) -> str:
    offsets = dict(cond.aim_offsets_mm)
    if not offsets:
        return f"nominal placement, seed {cond.seed}"
    worst = max(math.hypot(*map(float, v)) for v in offsets.values())
    return f"placement errors up to {worst:.1f} mm per object, seed {cond.seed}"


def _lane_condition(report: Mapping[str, Any], setup_ids: Sequence[str]) -> Mapping[str, Any]:
    """The condition every lane shows: the first of the grid on which the first lane's setup
    completed (the grid's first when it never did)."""
    finalist = next(f for f in report["finalists"] if f["setup"]["setup_id"] == setup_ids[0])
    done = {r["condition_id"] for r in finalist["runs"] if r["completed"]}
    for cond in report["conditions"]:
        if cond["condition_id"] in done:
            return cond
    return report["conditions"][0]


def export_lanes(
    hero: str,
    report: Mapping[str, Any],
    store: Any,
    *,
    code_version: str,
    log: Callable[[str], Any] = print,
    failures: list[str] | None = None,
    budget: int = LANE_BUDGET_BYTES,
) -> list[LaneBuild]:
    """Record and build every lane of one hero (:func:`lane_choices`), all on one condition.

    A lane that does not reproduce the report's ``traj_sha``, or that cannot be built (it does not
    fit ``budget``, :data:`LANE_BUDGET_BYTES` by default, within 2 mm, a pose is not finite), raises
    :class:`ReplayMismatchError` (naming ``hero``/``lane``); with ``failures`` (a list) it is skipped
    and described there instead, as ``"<hero>/<lane>: traj_sha: ..."`` or ``"<hero>/<lane>: build: ..."``,
    so an export can name every lane that fails."""
    from rtt.studio.buildpack import setup_from_dict, task_from_dict
    from rtt.studio.episode import StudioCondition

    spec = task_from_dict(report["task"])
    choices = lane_choices(report)
    if not choices:
        return []
    cond_doc = _lane_condition(report, [sid for _, _, sid in choices])
    cond = StudioCondition(
        cond_doc["condition_id"],
        int(cond_doc["seed"]),
        {k: tuple(map(float, v)) for k, v in (cond_doc.get("aim_offsets_mm") or {}).items()},
    )
    out = []
    for lane, title, sid in choices:
        finalist = next(f for f in report["finalists"] if f["setup"]["setup_id"] == sid)
        setup = setup_from_dict(finalist["setup"])
        run = next(r for r in finalist["runs"] if r["condition_id"] == cond.condition_id)
        try:
            rec = record_run(
                setup,
                spec,
                cond,
                expected_traj_sha=run["traj_sha"],
                controllers=report.get("controllers") or "default",
            )
        except ReplayMismatchError as exc:
            message = f"{hero}/{lane}: traj_sha: {exc}"
            if failures is None:
                raise ReplayMismatchError(message) from exc
            failures.append(message)
            continue
        try:
            replay, used, fps = build_lane(
                rec, store, hero=hero, lane=lane, code_version=code_version, budget=budget
            )
        except ReplayMismatchError as exc:
            if failures is None:
                raise
            failures.append(f"{hero}/{lane}: build: {exc}")
            continue
        kf = replay.header["meta"]["keyframes"]
        log(
            f"  lane {lane}: {setup.label[:50]} on {cond.condition_id}: traj_sha {rec.result.traj_sha[:12]} "
            f"reproduced; {replay.header['n_bodies']} bodies x {replay.header['n_frames']} keyframes of "
            f"{kf['samples']} samples, {'int16' if replay.header['quantised'] else 'float32'}, "
            f"{lane_bytes(replay) / 1e3:.0f} kB, worst vertex {kf['max_error_vs_recording_mm']:.3f} mm "
            f"of {kf['tolerance_mm']:.1f} against the recording"
        )
        doc = {
            "id": lane,
            "title": title,
            "setup_id": sid,
            "kind": setup.kind.value,
            "label": setup.label,
            "replay": "",  # set by the caller
            "condition_id": cond.condition_id,
            "seed": int(cond.seed),
            "condition_text": _condition_text(cond),
            "traj_sha": rec.result.traj_sha,
            "completed": bool(rec.result.completed),
            "cycle_time_s": round(float(rec.result.cycle_time_s), 3),
            "fps": int(fps),  # the rate the keyframes were taken from; their times are in the replay's `t`
            "n_frames": int(replay.header["n_frames"]),
            "duration_s": round(float(replay.header["t"][-1]), 3),
            "quantised": bool(replay.header["quantised"]),
            # the largest distance of any vertex from the recording the keys were chosen from (the
            # gate's measure against the simulation, rtt.studio.replay_qa, is separate)
            "max_error_vs_recording_mm": kf["max_error_vs_recording_mm"],
            "bytes": 0,
            "asset_bytes": 0,
            "segments": replay.header["meta"]["segments"],
            "steps": replay.header["meta"]["steps"],
            "credits": _credits(rec),
        }
        out.append(LaneBuild(replay=replay, doc=doc, assets=used, stem=lane, recording=rec))
    return out


# --------------------------------------------------------------------------- serving assets


def serve_assets(
    store: Any, lanes: Sequence[LaneBuild], *, compress: bool = True
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    """Served bytes and entries of every raw asset the lanes draw (meshopt-compressed by the web
    build's compressor unless ``compress`` is false), and every lane's header pointed at them."""
    raw_ids = sorted({a for lane in lanes for a in lane.assets})
    raw = {aid: (store.raw_bytes(aid), store.entry(aid)) for aid in raw_ids}
    if compress:
        from rtt.render.compress import compress_assets

        served = compress_assets(raw)
    else:
        served = {aid: (data, dict(entry)) for aid, (data, entry) in raw.items()}
    new_id = {aid: hashlib.sha256(data).hexdigest()[:16] for aid, (data, _) in served.items()}
    for lane in lanes:
        h = lane.replay.header
        for body in h["bodies"]:
            for v in body["visuals"]:
                v["asset"] = new_id[v["asset"]]
        for v in h.get("static_visuals", []):
            v["asset"] = new_id[v["asset"]]
        h["assets"] = {new_id[a]: served[a][1] for a in sorted(lane.assets)}
        h["assets"] = dict(sorted(h["assets"].items()))
        lane.assets = {new_id[a] for a in lane.assets}
    return {new_id[a]: served[a] for a in raw_ids}


#: Labels of the assets this module makes; an entry with one of them that no lane draws any more is
#: removed from the site's store when the Studio publishes (no other producer uses these labels).
STUDIO_LABEL_PREFIXES: Final = ("Studio ", "Tool stand (display only")


def publish_assets(
    served: Mapping[str, tuple[bytes, Mapping[str, Any]]],
    assets_dir: Path,
    *,
    prune: bool = True,
    keep: Collection[str] = (),
) -> list[str]:
    """Write the served GLBs into the site's asset store and merge their entries into its manifest.
    With ``prune``, Studio assets (:data:`STUDIO_LABEL_PREFIXES`) that neither ``served`` nor
    ``keep`` holds are removed: pass every lane of every hero exported, and in ``keep`` the assets
    of the published heroes a partial export leaves as they are (the export script does both)."""
    from rtt.render.glb import MANIFEST_FORMAT, MANIFEST_VERSION, _write_atomic

    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / "manifest.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"assets": {}}
    if doc.get("format", MANIFEST_FORMAT) != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not an asset manifest")
    entries = dict(doc.get("assets") or {})
    written = []
    for aid, (data, entry) in sorted(served.items()):
        if hashlib.sha256(data).hexdigest()[:16] != aid:
            raise ValueError(f"asset {aid}: bytes do not hash to its id")
        f = assets_dir / f"{aid}.glb"
        if not (f.is_file() and f.read_bytes() == data):
            _write_atomic(f, data)
        entries[aid] = dict(entry)
        written.append(aid)
    if prune:
        for aid in sorted(entries):
            label = str(entries[aid].get("label", ""))
            if aid not in served and aid not in keep and label.startswith(STUDIO_LABEL_PREFIXES):
                del entries[aid]
                (assets_dir / f"{aid}.glb").unlink(missing_ok=True)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "assets": dict(sorted(entries.items())),
    }
    _write_atomic(path, (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return written


# --------------------------------------------------------------------------- blocker probe


def _probe_job(job: Any) -> tuple[Any, Any]:
    from rtt.studio.runner import run_segment_job

    return run_segment_job(job)


def _blocking_step(report: Mapping[str, Any]) -> str | None:
    counts: dict[str, int] = {}
    for row in report["recommendation"]["ranking"]:
        for step, v in (row.get("stops_at") or {}).items():
            counts[step] = counts.get(step, 0) + int(v.get("runs", 0))
    if not counts:
        return None
    order = [s["id"] for s in report["task"]["steps"]]
    return max(counts, key=lambda s: (counts[s], -order.index(s)))


def blocker_probe(
    report: Mapping[str, Any],
    *,
    workers: int = 4,
    candidates: Sequence[Any] = (),
    log: Callable[[str], Any] = print,
) -> dict[str, Any] | None:
    """The physical blocker of a task nothing completes (module docstring), or ``None``.

    ``candidates`` (:class:`rtt.studio.search.Candidate`) default to this build's candidate pool for
    the task (:func:`rtt.studio.search.default_candidates`), which is what the tournament ran."""
    from rtt.effectors.lanes import FAMILY_TITLES
    from rtt.studio.buildpack import task_from_dict
    from rtt.studio.episode import StudioCondition
    from rtt.studio.runner import SegmentJob, WorkerPool
    from rtt.studio.search import default_candidates, hold_chains, sub_task
    from rtt.studio.types import QUANTITIES, Setup

    if report["recommendation"].get("recommendation"):
        return None
    step_id = _blocking_step(report)
    if step_id is None:
        return None
    spec = task_from_dict(report["task"])
    step = spec.step(step_id)
    chains = hold_chains(spec)
    j = next(i for i, c in enumerate(chains) if step_id in c)
    sc = report["screen"]
    conds = [
        StudioCondition(
            c["condition_id"],
            int(c["seed"]),
            {k: tuple(map(float, v)) for k, v in (c.get("aim_offsets_mm") or {}).items()},
        )
        for c in sc["conditions"]
    ]
    controllers = report.get("controllers") or "default"
    cands = {c.effector_id: c for c in candidates} if candidates else {}
    if not cands:
        cands = {c.effector_id: c for c in default_candidates(task_spec=spec)}
    cells = {c["effector_id"]: c for c in sc["cells"] if c["step_id"] == step_id}
    missing = [e for e in cells if e not in cands]
    if missing:
        raise ReplayMismatchError(
            f"{len(missing)} screened candidates are not in this build's candidate pool"
        )
    pool = WorkerPool(max(1, min(int(workers), 4)))
    try:
        context: dict[str, Any] = {c.condition_id: None for c in conds}
        for i in range(j):
            tool = sc["context_tools"][i]
            if tool is None:
                break
            sub = sub_task(spec, chains[i])
            jobs = [
                SegmentJob(
                    Setup.single(cands[tool].spec, chains[i]),
                    sub,
                    c,
                    initial=context[c.condition_id],
                    controllers=controllers,
                )
                for c in conds
            ]
            for c, (_, state) in zip(conds, pool.map(_probe_job, jobs), strict=True):
                context[c.condition_id] = state
        sub = sub_task(spec, chains[j])
        order = sorted(cells)
        jobs = [
            SegmentJob(
                Setup.single(cands[e].spec, chains[j]),
                sub,
                c,
                initial=context[c.condition_id],
                controllers=controllers,
                tag=(e, c.condition_id),
            )
            for e in order
            for c in conds
        ]
        results = pool.map(_probe_job, jobs, chunksize=len(conds))
    finally:
        pool.close()
    by_cand: dict[str, list[Any]] = {}
    for job, (res, _) in zip(jobs, results, strict=True):
        o = next(x for x in res.steps if x.step_id == step_id)
        by_cand.setdefault(job.tag[0], []).append(o)
    for e, outs in by_cand.items():
        got = sum(1 for o in outs if o.success)
        if got != int(cells[e]["successes"]):
            raise ReplayMismatchError(
                f"probe of {cands[e].name[:40]} on {step_id}: {got} successes, the report's matrix says {cells[e]['successes']}"
            )
    goals = [c for c in step.success if c.at_least is not None]
    if not goals:
        return None

    def shortfall(c: Any) -> float:
        vals = [float(o.quantities.get(c.quantity, 0.0)) for outs in by_cand.values() for o in outs]
        return float(np.median(vals)) / float(c.at_least) if c.at_least else 1.0

    crit = min(goals, key=shortfall)
    q = QUANTITIES[crit.quantity]
    families: dict[str, dict[str, Any]] = {}
    for e, outs in by_cand.items():
        cand = cands[e]
        fam = cand.spec.family
        peak = max(float(o.quantities.get(crit.quantity, 0.0)) for o in outs)
        row = families.setdefault(
            fam,
            {
                "family": fam,
                "family_title": FAMILY_TITLES.get(fam, fam),
                "n_candidates": 0,
                "n_runs": 0,
                "peak": -math.inf,
                "best": None,
                "failure_modes": {},
            },
        )
        row["n_candidates"] += 1
        row["n_runs"] += len(outs)
        if peak > row["peak"]:
            row["peak"], row["best"] = peak, cand.spec.display_name
        for o in outs:
            if o.failure_mode:
                row["failure_modes"][o.failure_mode] = row["failure_modes"].get(o.failure_mode, 0) + 1
    rows = sorted(families.values(), key=lambda r: (-r["peak"], r["family"]))
    for r in rows:
        r["peak"] = round(r["peak"], 3)
    max_peak = rows[0]["peak"] if rows else 0.0
    unit = q.unit
    name = {"torque_nm": "torque", "angle_deg": "angle", "depth_mm": "depth", "force_n": "force"}.get(
        crit.quantity, crit.quantity
    )
    passed = sum(int(cells[e]["successes"]) for e in cells)
    text = (
        f"Every tool screened on this step reaches a peak {name} of at most {max_peak:g} {unit}, "
        f"against the {crit.at_least:g} {unit} the step needs"
    )
    text += "." if passed == 0 else f"; {passed} screen runs still passed."
    log(
        f"  blocker: {step_id} {crit.quantity} peak {max_peak:g} {unit} vs {crit.at_least:g} over {len(jobs)} runs"
    )
    doc = {
        "step_id": step_id,
        "primitive": step.primitive.value,
        "quantity": crit.quantity,
        "unit": unit,
        "at_least": crit.at_least,
        "at_most": crit.at_most,
        "text": text,
        "conditions": [c.condition_id for c in conds],
        "n_candidates": len(by_cand),
        "n_runs": len(jobs),
        "families": rows,
        "max_peak": max_peak,
    }
    hits = find_brand_tokens(json.dumps(doc))
    if hits:
        raise ReplayMismatchError(f"the blocker table names a brand ({', '.join(sorted(set(hits)))})")
    return doc
