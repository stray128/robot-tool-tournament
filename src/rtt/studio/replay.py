"""Studio replays for the site: full-task runs as replay v2 lanes, and the blocker probe (track S7).

Product mode: every lane is labelled so (``provenance.product_mode``) and is a re-simulation of a
scored run, never a new result.

**Recording.** :func:`record_run` re-runs one ``(setup, task, condition)`` of a tournament report
through the frozen full-task runner (:func:`rtt.studio.episode.run_task`, in process) with a
sampler wrapped around the step controllers and predicates: it only reads ``data.xpos`` and
``data.xquat`` at the controller ticks (before each physics step) and at the predicate's check
ticks (after one), the first tick at or after every frame time. Nothing is written to the
simulation, so the run's ``traj_sha`` is the scored one; :func:`record_run` refuses a run whose
``traj_sha`` differs from the report's (:class:`ReplayMismatchError`). Bodies are sampled at a
fixed rate (:data:`RECORD_FPS`) on the task clock.

**Segments.** A setup's segments (:meth:`rtt.studio.types.Setup.segments`) run in their own
scenes. A tool change or a head switch is a fixed clock advance in the contract, not simulated
motion, so the replay holds the last pose of a segment through the change time and then cuts to
the next segment's first frame (its home pose). Bodies are matched across scenes by a stable key:
``t<k>/<family name>`` for the bodies of tool ``k`` (an idle head's ``eff_h<k>_`` prefix is
removed), the element name otherwise (wrist, bracket, changer, objects). A tool-change tool that
is not on the wrist rests on a display-only **tool stand** behind the table, in the pose it hangs
in on the wrist at its first frame (the stand is drawn as a prop and labelled not simulated).

**Drawing.** Our own geometry (template objects, procedural tools, the bracket, the changer and
the flange) is drawn from content-addressed GLB assets made from each body's collision geoms
(:func:`rtt.render.glb.mesh_from_geoms`, Apache-2.0). Bodies of third-party tool models (any build
with licence records: vendored grippers and hands, the stock blade) get no GLB: they are drawn from
their collision geoms in the header, as the effector lanes are, and the lane lists their brand-free
credit with the credits page anchor. Bodies that never move relative to each other are merged into
one replay body (the members become visuals or geoms placed in its frame), and bodies that never
move at all become static, which keeps the pose payload small. Each lane must fit
:data:`LANE_BUDGET_BYTES` (header plus poses as served); it is decimated to the first of
:data:`FPS_LADDER` that fits.

**Blocker probe.** :func:`blocker_probe` explains a task nothing completes: the step most finalist
runs stop at is re-run for every candidate the screen ran on it, from the same carried context
(the screen's best earlier tools, on the screen's conditions), and the peak of the step's most
missed goal quantity (a twist's torque, for example) is reported per tool family against its
target. The probe re-checks each candidate's successes against the report's success matrix.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens

__all__ = [
    "FPS_LADDER",
    "LANE_BUDGET_BYTES",
    "RECORD_FPS",
    "LaneBuild",
    "Recording",
    "ReplayMismatchError",
    "blocker_probe",
    "build_lane",
    "export_lanes",
    "lane_choices",
    "publish_assets",
    "record_run",
    "serve_assets",
    "staging_store",
]

#: Frames per second of the recording (the highest a lane is published at).
RECORD_FPS: Final = 30
#: Published frame rates, best first: a lane is decimated to the first that fits its budget.
FPS_LADDER: Final = (30, 15, 10, 6)
#: Bytes a lane may serve: its header and poses (its GLBs are shared and counted apart).
LANE_BUDGET_BYTES: Final = 480_000
#: Two bodies are one rigid group when their relative pose never moves more than this.
#: Circle segments of the GLB meshes made from cylinders and capsules (small tool parts).
GLB_SEGMENTS: Final = 24
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


@dataclass
class _SceneMap:
    bodies: np.ndarray  # body ids
    keys: list[str]


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
                template = None
                if category == "object":
                    inst = next(
                        (o for o in scene.task_spec.objects if name.startswith(f"obj_{o.name}_")), None
                    )
                    template = inst.kind.value if inst is not None else None
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
                )
            ids.append(b)
            keys.append(key)
        if not self.rec.world_static:
            self.rec.world_static = _geom_entries(model, 0)
        found = _SceneMap(np.array(ids, dtype=int), keys)
        self._maps[sid] = found
        self._scenes.append(sid)
        self.rec.scene_tools.append(int(scene.tool))
        return found

    def sample(self, ctx: Any, data: mujoco.MjData, t_local: float, *, step_dt: float) -> None:
        t = float(ctx.t0) + float(t_local)
        if t + 0.5 * step_dt < self.next_t:
            return
        smap = self._map(ctx.scene)
        poses = np.empty((len(smap.bodies), 7))
        poses[:, :3] = data.xpos[smap.bodies]
        poses[:, 3:] = _quat_xyzw(data.xquat[smap.bodies])
        self.rec.times.append(t)
        self.rec.frames.append(dict(zip(smap.keys, poses, strict=True)))
        self.rec.scene_of_frame.append(self._scenes.index(id(ctx.scene)))
        self.next_t = math.floor(t * self.rec.fps + 1.0 + 1e-6) / self.rec.fps


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

    rec = Recording(setup=setup, task_spec=task_spec, condition=condition, result=None, fps=int(fps))
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


def _grid(rec: Recording, fps: int) -> tuple[np.ndarray, list[int]]:
    """Frame times ``k / fps`` up to the cycle time and, per frame, the raw sample it shows (the
    latest at or before it: a change gap holds the previous segment's last pose)."""
    end = max(float(rec.result.cycle_time_s), rec.times[-1])
    n = math.floor(end * fps + 1e-6) + 1
    times = np.arange(n) / fps
    raw = np.asarray(rec.times)
    idx = np.searchsorted(raw, times + 0.5 / RECORD_FPS, side="right") - 1
    return times, [int(max(0, i)) for i in idx]


def _key_order(bodies: Mapping[str, _BodyInfo]) -> list[str]:
    rank = {"wrist": 0, "stack": 1, "tool": 2, "object": 3}
    return sorted(bodies, key=lambda k: (rank[bodies[k].category], bodies[k].tool or 0, k))


def _parked_tracks(rec: Recording, keys: Sequence[str], sample_idx: Sequence[int]) -> np.ndarray:
    """``(F, B, 7)`` poses: a body's recorded pose, or its tool-stand pose when its tool is off
    the wrist (tool changes)."""
    n_tools = len(rec.setup.tools)
    first: dict[int, int] = {}
    for i, frame in enumerate(rec.frames):
        tool = rec.scene_tools[rec.scene_of_frame[i]]
        first.setdefault(tool, i)
    stands = {
        k: np.array([(k - (n_tools - 1) / 2.0) * STAND_PITCH_M, STAND_Y_M, STAND_Z_M, 0.0, 0.0, 0.0, 1.0])
        for k in range(n_tools)
    }
    rel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out = np.empty((len(sample_idx), len(keys), 7))
    for f, i in enumerate(sample_idx):
        frame = rec.frames[i]
        for b, key in enumerate(keys):
            pose = frame.get(key)
            if pose is not None:
                out[f, b] = pose
                continue
            info = rec.bodies[key]
            if info.tool is None or info.tool not in first:
                raise ReplayMismatchError(f"body {key} is missing from a frame and is not a parked tool")
            if key not in rel:
                f0 = rec.frames[first[info.tool]]
                rel[key] = _relative(f0["wrist_flange"], f0[key])
            pos, rot = rel[key]
            out[f, b] = _compose(stands[info.tool], pos, rot)
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


def _look(info: _BodyInfo) -> Any:
    from rtt.render import materials as mat

    if info.category == "object":
        return mat.PART_LOOKS[_LOOKS_BY_KIND.get(info.template or "", "clay")]
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


def build_lane(
    rec: Recording,
    store: Any,
    *,
    hero: str,
    lane: str,
    code_version: str,
    budget: int = LANE_BUDGET_BYTES,
) -> tuple[Any, set[str], int]:
    """The replay v2 of a recording (module docstring) at the first frame rate of
    :data:`FPS_LADDER` that fits ``budget``; returns the replay, the raw asset ids it draws and
    its fps."""

    keys = _key_order(rec.bodies)
    for fps in FPS_LADDER:
        if fps > rec.fps or rec.fps % fps:
            continue
        _, idx = _grid(rec, fps)
        tracks = _parked_tracks(rec, keys, idx)
        replay, used = _assemble(
            rec, store, keys, tracks, fps, hero=hero, lane=lane, code_version=code_version
        )
        size = (
            len(json.dumps(replay.header, separators=(",", ":"), ensure_ascii=False))
            + 1
            + replay.frames.nbytes
        )
        if size <= budget:
            return replay, used, fps
        last = (fps, size, replay.frames.nbytes)
    raise ReplayMismatchError(
        f"lane {hero}/{lane} does not fit {budget} bytes: {last[1]} at {last[0]} fps ({last[2]} of poses)"
    )


def _assemble(
    rec: Recording,
    store: Any,
    keys: Sequence[str],
    tracks: np.ndarray,
    fps: int,
    *,
    hero: str,
    lane: str,
    code_version: str,
) -> tuple[Any, set[str]]:
    from rtt.sim import replay as rp

    groups, static = _groups(tracks, keys, rec.bodies)
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
    frames = np.ascontiguousarray(
        np.stack([tracks[:, g.rep] for g in groups], axis=1).astype(np.float32)
        if groups
        else np.zeros((tracks.shape[0], 0, 7), np.float32)
    )
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
        "fps": int(fps),
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": list(rp.POSE_FIELDS),
        "up_axis": "z",
        "units": "m",
        "n_frames": int(frames.shape[0]),
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
    return rp.Replay(frames=frames, header=header, overlays=None), used


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
) -> list[LaneBuild]:
    """Record and build every lane of one hero (:func:`lane_choices`), all on one condition."""
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
        rec = record_run(
            setup,
            spec,
            cond,
            expected_traj_sha=run["traj_sha"],
            controllers=report.get("controllers") or "default",
        )
        replay, used, fps = build_lane(rec, store, hero=hero, lane=lane, code_version=code_version)
        log(
            f"  lane {lane}: {setup.label[:50]} on {cond.condition_id}: traj_sha {rec.result.traj_sha[:12]} "
            f"reproduced; {replay.header['n_bodies']} bodies x {replay.header['n_frames']} frames at {fps} fps"
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
            "fps": int(fps),
            "n_frames": int(replay.header["n_frames"]),
            "duration_s": round((replay.header["n_frames"] - 1) / fps, 3),
            "bytes": 0,
            "asset_bytes": 0,
            "segments": replay.header["meta"]["segments"],
            "steps": replay.header["meta"]["steps"],
            "credits": _credits(rec),
        }
        out.append(LaneBuild(replay=replay, doc=doc, assets=used, stem=lane))
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
    served: Mapping[str, tuple[bytes, Mapping[str, Any]]], assets_dir: Path, *, prune: bool = True
) -> list[str]:
    """Write the served GLBs into the site's asset store and merge their entries into its manifest.
    With ``prune``, Studio assets (:data:`STUDIO_LABEL_PREFIXES`) that ``served`` no longer holds
    are removed (pass every lane of every hero, as the export script does)."""
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
            if aid not in served and label.startswith(STUDIO_LABEL_PREFIXES):
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
