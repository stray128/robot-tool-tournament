"""The replay quality gate: a Studio lane that does not faithfully show the simulation is never exported.

A lane (:class:`rtt.studio.replay.LaneBuild`) is judged as the player will show it: the poses it reads
back (float32, or int16 decoded), interpolated linearly in position and spherically in rotation
between keyframes, held across a cut, drawn from the geometry the header points at. Six checks
(:data:`CHECKS`); every one must pass or the export writes nothing (:func:`check_lane` returns the
problems, the export script names hero, lane and check):

``traj_sha``
    A re-run reproduces the scored run: the 500 Hz re-run, the recording the lane was made from and
    the lane's meta all give the report's ``traj_sha`` for that setup and condition.
``cuts``
    The lane's ``cuts`` are exactly the starts of the run's segments (a tool change or a head
    switch), with a frame at the end of the segment before and at the start of the one after. A lane
    that lists no cut where the run has one would slide a tool across the change.
``interpolation``
    Every vertex of every body, as the lane draws it at the 60 Hz instants of the player, is within
    :data:`INTERP_TOL_M` (2 mm) of the simulation, outside the gaps of cuts (nothing is simulated
    there). The simulation is a **500 Hz re-run** of the same run (every control tick), interpolated
    to those instants. The measure is ``max_error_vs_sim_mm`` (with its p99, the worst body and time);
    the lane's own ``max_error_vs_recording_mm`` is against the 100 Hz recording, which is not the same
    thing and not what is checked here.
``jumps``
    No body jumps between two consecutive 60 Hz display frames: a frame pair is flagged only where
    the drawn displacement e is over :data:`JUMP_TOL_M` (50 mm) **and** over the simulation's own
    displacement s over the same two instants (the 500 Hz re-run) by more than :data:`JUMP_SLACK_M`
    (5 mm), and never across a cut or a gap nothing is simulated in. It is a rule on the frames the
    player shows, not on consecutive keys (keys of a healthy lane can be 116 mm apart), and relative to
    the simulation, because honest fast motion is fast: a carton in free fall from 1 m moves 71 mm
    between display frames and a faithful lane shows exactly that.
``penetration``
    The drawn world (every body's collision geoms, which its GLB is meshed from, at the lane's poses)
    may overlap no more than ``max(1 mm, simulated overlap + 0.5 mm)`` for the same pair of bodies at
    the same time, beyond one display frame: a pair of bodies fails when two of its display instants
    over the allowance fall within any three consecutive display frames. Bodies of one rigid tool
    stack are not compared with each other.
``glb``
    The served GLBs coincide with the collision geoms of the bodies they draw within
    :data:`GLB_TOL_M` (0.05 mm), vertex to vertex in both directions, in the placement the header gives
    (meshopt-compressed assets are decoded with Node and the web app's packages).

The checks are ported from the diagnosis tools that found objects passing through walls and each
other in the published replays. A lane that has no frame where the player shows one (or whose
header the replay v2 rules refuse) fails ``header`` before any of them.
"""

from __future__ import annotations

import io
import itertools
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np

from rtt.render.compress import WEB_DIR
from rtt.render.glb import read_glb, read_glb_json
from rtt.studio import replay as sr

__all__ = [
    "CHECKS",
    "DISPLAY_FPS",
    "GLB_TOL_M",
    "INTERP_TOL_M",
    "JUMP_SLACK_M",
    "JUMP_TOL_M",
    "PEN_FLOOR_M",
    "PEN_SLACK_M",
    "QA_RECORD_FPS",
    "QAResult",
    "check_lane",
    "render_frames",
]

#: The player's frame rate: the instants a lane is judged at.
DISPLAY_FPS: Final = 60
#: The rate of the re-run a lane is measured against: every control tick of the simulation.
QA_RECORD_FPS: Final = 500
#: Metres any vertex of the drawn lane may be from the simulation (outside cuts).
INTERP_TOL_M: Final = 2e-3
#: Metres a body may move between two consecutive display frames before it is looked at ...
JUMP_TOL_M: Final = 50e-3
#: ... and the metres by which that must also exceed what the simulation itself moves it over the
#: same two instants for the frame to be a jump. [assumed]: about five times the p99 of the drawn minus
#: simulated displacement on the real lanes (under 1 mm), far under a teleport (50 mm and more).
JUMP_SLACK_M: Final = 5e-3
#: Drawn overlap allowed whatever the simulation did, and the slack over the simulation's own.
PEN_FLOOR_M: Final = 1e-3
PEN_SLACK_M: Final = 0.5e-3
#: Metres a GLB vertex may be from the collision geoms (and the reverse).
GLB_TOL_M: Final = 5e-5
CHECKS: Final = ("traj_sha", "cuts", "interpolation", "jumps", "penetration", "glb")
_EPS_T: Final = 1e-6  # seconds: the header writes its times to the microsecond


@dataclass(frozen=True)
class QAResult:
    """The verdict of :func:`check_lane` on one lane: every problem found (each starts with the check's
    name and a colon), and what each check measured."""

    passed: bool
    problems: list[str]
    metrics: dict[str, Any]


# --------------------------------------------------------------------------- the lane as drawn


class _Drawn:
    """A lane as a player draws it, in the frame of the recording it was made from."""

    def __init__(self, lane: Any) -> None:
        rec: sr.Recording = lane.recording
        self.rec = rec
        self.header: dict[str, Any] = lane.replay.header
        self.frames = np.asarray(lane.replay.frames, dtype=np.float64)
        self.t = np.asarray(self.header["t"], dtype=np.float64)
        self.cuts: list[int] = list(self.header.get("cuts", []))
        self.keys = sr._key_order(rec.bodies)
        self.tracks = sr._parked_tracks(rec, self.keys, range(len(rec.frames)))
        self.groups, self.static = sr._groups(self.tracks, self.keys, rec.bodies)
        fit = sr._fit_of(rec, self.keys, self.tracks, self.groups, self.static)
        self.owner, self.rel_pos, self.rel_rot = fit.owner, fit.rel_pos, fit.rel_rot
        self.points = fit.points
        self.rel_quat = np.array([sr._mat_quat(r) for r in self.rel_rot])

    def instants(self) -> np.ndarray:
        """The display instants: 60 Hz from the start to the last frame of the lane."""
        end = float(self.t[-1])
        return np.unique(np.append(np.arange(0.0, end, 1.0 / DISPLAY_FPS), end))

    def reps(self, at: np.ndarray) -> np.ndarray:
        """Poses ``(len(at), groups, 7)`` of the drawn bodies at ``at`` (the player's ``sample()``)."""
        return sr.sample_lane(self.frames, self.t, self.cuts, at)

    def keys_at(self, reps: np.ndarray) -> np.ndarray:
        """Poses ``(n, keys, 7)`` of every key (a body drawn as its group's body composed with a fixed
        offset, or where it rests) for drawn bodies ``reps`` ``(n, groups, 7)``."""
        n = reps.shape[0]
        out = np.empty((n, len(self.keys), 7))
        rot = sr._rot(reps[..., 3:]) if reps.shape[1] else None
        for m in range(len(self.keys)):
            g = int(self.owner[m])
            if g >= 0:
                out[:, m, :3] = reps[:, g, :3] + rot[:, g] @ self.rel_pos[m]
                out[:, m, 3:] = _qmul(reps[:, g, 3:], self.rel_quat[m])
            else:
                out[:, m, :3] = self.rel_pos[m]
                out[:, m, 3:] = self.rel_quat[m]
        return out


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Products of xyzw quaternions ``a`` (``(n, 4)``) and ``b`` (``(4,)``), unit length."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (float(v) for v in b)
    q = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _displacement(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Metres: for poses ``a`` and ``b`` ``(n, 7)`` of one body, the farthest any of its ``points``
    (body frame, ``(P, 3)``) is between them."""
    out = np.empty(len(a))
    ra, rb = sr._rot(a[:, 3:]), sr._rot(b[:, 3:])
    for c0 in range(0, len(a), 1024):
        sl = slice(c0, c0 + 1024)
        d = (b[sl, :3] - a[sl, :3])[:, :, None] + (rb[sl] - ra[sl]) @ points.T
        out[sl] = np.sqrt(np.max(np.sum(d * d, axis=1), axis=1))
    return out


def _sim_gaps(rec: sr.Recording) -> list[tuple[float, float]]:
    """``(last time of one segment, first of the next)``: nothing is simulated in between."""
    seg = sr._sample_segments(rec)
    times = np.asarray(rec.times)
    return [(float(times[i - 1]), float(times[i])) for i in np.flatnonzero(seg[1:] != seg[:-1]) + 1]


def _outside(at: np.ndarray, gaps: Sequence[tuple[float, float]]) -> np.ndarray:
    keep = np.ones(len(at), dtype=bool)
    for a, b in gaps:
        keep &= ~((at > a + _EPS_T) & (at < b - _EPS_T))
    return keep


# --------------------------------------------------------------------------- the checks


def _expected_sha(report: Mapping[str, Any], rec: sr.Recording) -> str | None:
    for f in report.get("finalists", []):
        if f["setup"]["setup_id"] == rec.setup.setup_id:
            for run in f["runs"]:
                if run["condition_id"] == rec.condition.condition_id:
                    return str(run["traj_sha"])
    return None


def _check_sha(report: Mapping[str, Any], lane: Any, dense: sr.Recording) -> tuple[list[str], dict[str, Any]]:
    rec: sr.Recording = lane.recording
    want = _expected_sha(report, rec)
    if want is None:
        missing = (
            f"traj_sha: the report holds no run of setup {rec.setup.setup_id[:12]} on "
            f"{rec.condition.condition_id}, so nothing says what the replay should show"
        )
        return [missing], {"reproduced": False}
    got = {
        "the recording the lane was made from": rec.result.traj_sha,
        f"the {QA_RECORD_FPS} Hz re-run": dense.result.traj_sha,
        "the lane's meta": str(lane.replay.header.get("meta", {}).get("traj_sha")),
    }
    problems = [
        f"traj_sha: {who} gives {sha[:12]}, the report {want[:12]}" for who, sha in got.items() if sha != want
    ]
    return problems, {"reproduced": not problems, "traj_sha": want[:12]}


def _check_cuts(d: _Drawn) -> tuple[list[str], dict[str, Any]]:
    """The lane's cuts are the run's segment starts, framed by a key on each side."""
    rec = d.rec
    seg = sr._sample_segments(rec)
    edges = [int(i) + 1 for i in np.flatnonzero(seg[1:] != seg[:-1])]
    problems: list[str] = []
    metrics: dict[str, Any] = {"cuts": len(d.cuts), "segments_after_the_first": len(edges)}
    if len(d.cuts) != len(edges):
        problems.append(
            f"cuts: the run changes segment {len(edges)} time(s) but the lane lists {len(d.cuts)} cut(s): "
            f"the player would interpolate across the change"
        )
    for k, i in enumerate(edges):
        before, after = float(rec.times[i - 1]), float(rec.times[i])
        if k >= len(d.cuts):
            break
        c = d.cuts[k]
        if not (0 < c < len(d.t)):
            problems.append(f"cuts: cut {k + 1} is at frame {c}, outside the lane")
            continue
        if abs(d.t[c] - after) > _EPS_T or abs(d.t[c - 1] - before) > _EPS_T:
            problems.append(
                f"cuts: cut {k + 1} is between frames at {d.t[c - 1]:.3f} s and {d.t[c]:.3f} s but the "
                f"segment ends at {before:.3f} s and the next starts at {after:.3f} s: the player would "
                f"interpolate across the cut"
            )
    # a lane with no key at the end of a segment or at the start of the next cannot hold or cut there
    present = np.asarray(d.t)
    for before, after in _sim_gaps(rec):
        if np.min(np.abs(present - before)) > _EPS_T or np.min(np.abs(present - after)) > _EPS_T:
            problems.append(
                f"cuts: the lane has no frame at {before:.3f} s or {after:.3f} s, the ends of a segment: "
                f"it is interpolated across the cut"
            )
    return problems, metrics


class _Truth:
    """The 500 Hz re-run as poses of every key, interpolated to any time."""

    def __init__(self, d: _Drawn, dense: sr.Recording) -> None:
        self.dense = dense
        self.times = np.asarray(dense.times)
        self.tracks = sr._parked_tracks(dense, d.keys, range(len(dense.frames)))
        self.gaps = _sim_gaps(dense)

    def at(self, instants: np.ndarray) -> np.ndarray:
        return sr.sample_lane(self.tracks, self.times, [], instants)


def _check_interpolation(
    d: _Drawn, truth: _Truth, instants: np.ndarray, drawn: np.ndarray, sim: np.ndarray
) -> tuple[list[str], dict[str, Any]]:
    keep = _outside(instants, truth.gaps)
    at = instants[keep]
    mine = drawn[keep]
    sim = sim[keep]
    worst, worst_key, worst_t = 0.0, "", 0.0
    errs = []
    for m, key in enumerate(d.keys):
        e = _displacement(d.points[m], mine[:, m], sim[:, m])
        errs.append(e)
        j = int(np.argmax(e))
        if e[j] > worst:
            worst, worst_key, worst_t = float(e[j]), key, float(at[j])
    flat = np.concatenate(errs) if errs else np.zeros(1)
    quantised = bool(d.header.get("quantised"))
    metrics = {
        # against the simulation (the 500 Hz re-run), not the recording the keys were chosen from
        "max_error_vs_sim_mm": round(1000 * worst, 3),
        "p99_error_vs_sim_mm": round(1000 * float(np.percentile(flat, 99)), 3),
        "worst_body": worst_key,
        "worst_time_s": round(worst_t, 3),
        "instants": len(at),
        "against_hz": QA_RECORD_FPS if truth.dense.fps >= QA_RECORD_FPS else int(truth.dense.fps),
        "truth_tick_ms": round(1000 * float(np.median(np.diff(truth.times))), 3),
        "quantised": quantised,
        "tolerance_mm": 1000 * INTERP_TOL_M,
    }
    problems = []
    if worst > INTERP_TOL_M:
        problems.append(
            f"interpolation: {worst_key} is {1000 * worst:.2f} mm from the simulation at {worst_t:.3f} s "
            f"(at most {1000 * INTERP_TOL_M:.1f} mm; {len(at)} display instants checked)"
        )
    return problems, metrics


def _check_jumps(
    d: _Drawn, truth: _Truth, instants: np.ndarray, drawn: np.ndarray, sim: np.ndarray
) -> tuple[list[str], dict[str, Any]]:
    exempt = set()
    for c in d.cuts:
        if 0 < c < len(d.t):
            j = int(np.searchsorted(instants, d.t[c] - _EPS_T, side="left"))
            exempt.add(max(j, 1) - 1)  # the pair of display frames the cut falls between
    keep = _outside(instants, truth.gaps)
    valid = keep[:-1] & keep[1:]  # both instants simulated
    for i in exempt:
        if i < len(valid):
            valid[i] = False
    moving = [m for m in range(len(d.keys)) if d.owner[m] >= 0]
    worst, worst_key, worst_t = 0.0, "", 0.0  # the largest drawn displacement
    worst_sim = worst_excess = 0.0  # the simulation's largest; the largest drawn minus simulated
    flagged: list[tuple[float, str, float, float, float]] = []  # excess, body, time, drawn, simulated
    for m in moving:
        e = _displacement(d.points[m], drawn[:-1, m], drawn[1:, m])
        s = _displacement(d.points[m], sim[:-1, m], sim[1:, m])
        e, s = np.where(valid, e, 0.0), np.where(valid, s, 0.0)
        if not len(e):
            continue
        j = int(np.argmax(e))
        if e[j] > worst:
            worst, worst_key, worst_t = float(e[j]), d.keys[m], float(instants[j + 1])
        worst_sim = max(worst_sim, float(s.max()))
        worst_excess = max(worst_excess, float((e - s).max()))
        for i in np.flatnonzero((e > JUMP_TOL_M) & (e - s > JUMP_SLACK_M)):
            flagged.append((float(e[i] - s[i]), d.keys[m], float(instants[i + 1]), float(e[i]), float(s[i])))
    metrics = {
        "max_jump_mm": round(1000 * worst, 3),
        "jump_body": worst_key,
        "jump_time_s": round(worst_t, 3),
        "max_sim_jump_mm": round(1000 * worst_sim, 3),
        "max_drawn_minus_sim_mm": round(1000 * worst_excess, 3),
        "flagged_frames": len(flagged),
        "display_frames": len(instants),
        "cut_frames_exempt": len(exempt),
        "tolerance_mm": 1000 * JUMP_TOL_M,
        "slack_mm": 1000 * JUMP_SLACK_M,
    }
    problems = []
    if flagged:
        _, key, when, e, s = max(flagged)
        problems.append(
            f"jumps: {key} moves {1000 * e:.1f} mm in one display frame at {when:.3f} s where the simulation "
            f"moves it {1000 * s:.1f} mm, away from any cut ({len(flagged)} such frames; a frame must be over "
            f"{1000 * JUMP_TOL_M:.0f} mm and over the simulation's by more than {1000 * JUMP_SLACK_M:.0f} mm)"
        )
    return problems, metrics


def _body_cat(rec: sr.Recording, key: str) -> tuple[str, int | None]:
    if key.startswith("world:"):
        return "world", None
    info = rec.bodies[key]
    return info.category, info.tool


class _DrawnWorld:
    """The drawn geometry as a MuJoCo model: one mocap body per key carrying exactly the collision geoms
    its GLB is meshed from, and the static world's geoms. Nothing collides in it; it is only asked
    how far apart two geoms are."""

    def __init__(self, rec: sr.Recording, keys: Sequence[str], *, sky: bool = False) -> None:
        self.keys = list(keys)
        spec = mujoco.MjSpec()
        if sky:  # a light background for pictures (it changes no geometry)
            spec.add_texture(
                name="sky",
                type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=[0.92, 0.94, 0.97],
                rgb2=[0.62, 0.66, 0.72],
                width=64,
                height=64,
            )
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
        gkey: list[str] = []
        self._meshes = 0

        def add(body: Any, g: Mapping[str, Any], owner: str) -> None:
            kw = {"pos": list(g["pos"]), "quat": [g["quat_xyzw"][3], *g["quat_xyzw"][:3]]}
            kind = g["type"]
            if kind == "box":
                if min(g["half_size"]) <= 0:
                    return
                geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(g["half_size"]), **kw)
            elif kind == "cylinder":
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[g["radius"], g["half_height"], 0], **kw
                )
            elif kind == "capsule":
                if g["half_length"] <= 0:
                    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[g["radius"], 0, 0], **kw)
                else:
                    geom = body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[g["radius"], g["half_length"], 0], **kw
                    )
            elif kind == "mesh":
                name = f"m{self._meshes}"
                self._meshes += 1
                mesh = spec.add_mesh(name=name)
                mesh.uservert = list(map(float, g["vertices"]))
                mesh.userface = list(map(int, g["faces"]))
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, **kw)
            else:
                return
            geom.contype = 0
            geom.conaffinity = 0
            gkey.append(owner)

        for g in rec.world_static:
            add(spec.worldbody, g, "world:" + str(g.get("name", "?")))
        for k in self.keys:
            body = spec.worldbody.add_body(
                name=k.replace("/", "__"), mocap=True, mass=1.0, inertia=[1e-3] * 3
            )
            for g in rec.bodies[k].geoms:
                add(body, g, k)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.gkey = np.array(gkey)
        self.mocap = np.array(
            [int(self.model.body_mocapid[self.model.body(k.replace("/", "__")).id]) for k in self.keys]
        )
        kind = rec.setup.kind.value
        pairs = []
        for a in range(self.model.ngeom):
            for b in range(a + 1, self.model.ngeom):
                if self._compare(rec, self.gkey[a], self.gkey[b], kind):
                    pairs.append((a, b))
        self.pairs = np.array(pairs, dtype=int).reshape(-1, 2)

    @staticmethod
    def _compare(rec: sr.Recording, ka: str, kb: str, kind: str) -> bool:
        """Which bodies are compared: not a body with itself, not the world with itself, and not
        the tool-side bodies (wrist, stack, tools) with each other: they are one rigid assembly by
        design, except a parked tool against the working one in a tool change."""
        if ka == kb:
            return False
        (ca, ta), (cb, tb) = _body_cat(rec, ka), _body_cat(rec, kb)
        if ca == "world" and cb == "world":
            return False
        side = {"wrist", "stack", "tool"}
        if ca in side and cb in side:
            return kind == "tool_change" and ca == cb == "tool" and ta != tb
        return True

    def body_pair(self, i: int) -> tuple[str, str]:
        """The two bodies (sorted keys) of compared geom pair ``i``: a pair of bodies with several
        geoms is one pair."""
        a, b = self.pairs[i]
        return tuple(sorted((str(self.gkey[a]), str(self.gkey[b]))))  # type: ignore[return-value]

    def pair_name(self, i: int) -> str:
        a, b = self.body_pair(i)
        return f"{a} | {b}"

    def pose(self, poses: np.ndarray) -> None:
        """Put every key at ``poses`` ``(keys, 7)`` (x, y, z, qx, qy, qz, qw)."""
        self.data.mocap_pos[self.mocap] = poses[:, :3]
        self.data.mocap_quat[self.mocap] = poses[:, [6, 3, 4, 5]]
        mujoco.mj_kinematics(self.model, self.data)

    def depths(self, only: np.ndarray | None = None) -> np.ndarray:
        """Metres each compared pair overlaps (positive: penetration; 0: apart), for pairs ``only``."""
        if not len(self.pairs):
            return np.zeros(0)
        idx = np.arange(len(self.pairs)) if only is None else np.asarray(only)
        m, d = self.model, self.data
        pa, pb = self.pairs[idx, 0], self.pairs[idx, 1]
        near = (
            np.linalg.norm(d.geom_xpos[pa] - d.geom_xpos[pb], axis=1) < m.geom_rbound[pa] + m.geom_rbound[pb]
        )
        out = np.zeros(len(idx))
        ft = np.zeros(6)
        for j in np.flatnonzero(near):
            dist = mujoco.mj_geomDistance(m, d, int(pa[j]), int(pb[j]), 0.01, ft)
            out[j] = max(0.0, -dist)
        return out


def _persists(frames: Sequence[int]) -> bool:
    """Whether two of the display instants ``frames`` fall within three consecutive display frames
    (a violation that lasts beyond one display frame, even with a frame between)."""
    ordered = sorted(frames)
    return any(b - a <= 2 for a, b in itertools.pairwise(ordered))


def _check_penetration(
    d: _Drawn, truth: _Truth, instants: np.ndarray, drawn: np.ndarray, sim: np.ndarray
) -> tuple[list[str], dict[str, Any]]:
    world = _DrawnWorld(d.rec, d.keys)
    keep = _outside(instants, truth.gaps)
    at = instants[keep]
    mine = drawn[keep]
    sim = sim[keep]
    violating: dict[tuple[str, str], set[int]] = {}  # by pair of bodies, not of geoms
    worst_drawn = 0.0
    worst_excess, worst_where = 0.0, ("", 0.0, 0.0, 0.0)
    for i in range(len(at)):
        world.pose(mine[i])
        depth = world.depths()
        if not len(depth):
            break
        worst_drawn = max(worst_drawn, float(depth.max()))
        deep = np.flatnonzero(depth > PEN_SLACK_M)  # only these can exceed an allowance of at least the floor
        if not deep.size:
            continue
        world.pose(sim[i])
        sim_depth = world.depths(deep)
        for p, sd in zip(deep, sim_depth, strict=True):
            allowed = max(PEN_FLOOR_M, float(sd) + PEN_SLACK_M)
            if depth[p] > allowed:
                violating.setdefault(world.body_pair(int(p)), set()).add(i)
                if depth[p] - allowed > worst_excess:
                    worst_excess = float(depth[p] - allowed)
                    worst_where = (world.pair_name(int(p)), float(at[i]), float(depth[p]), float(sd))
    failing = {p: sorted(fr) for p, fr in violating.items() if _persists(fr)}
    metrics = {
        "max_drawn_mm": round(1000 * worst_drawn, 3),
        "max_excess_mm": round(1000 * worst_excess, 3),
        "violations": sum(len(fr) for fr in violating.values()),
        "failing_pairs": len(failing),
        "pairs_compared": len(world.pairs),
        "allowed_floor_mm": 1000 * PEN_FLOOR_M,
        "slack_mm": 1000 * PEN_SLACK_M,
        "display_frames": len(at),
    }
    problems = []
    for p in sorted(failing, key=lambda q: -len(failing[q]))[:5]:
        frames = failing[p]
        problems.append(
            f"penetration: {p[0]} | {p[1]} overlap in the drawn world at {len(frames)} display "
            f"frames from {at[frames[0]]:.3f} s (worst anywhere: {worst_where[0]} {1000 * worst_where[2]:.2f} mm "
            f"at {worst_where[1]:.3f} s against the simulation's {1000 * worst_where[3]:.2f} mm; allowed: the "
            f"larger of {1000 * PEN_FLOOR_M:.0f} mm and the simulation's overlap + {1000 * PEN_SLACK_M:.1f} mm, "
            f"for no two instants within three display frames)"
        )
    return problems, metrics


# --- GLBs

_DECODE_JS: Final = """
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
const [webDir, out, ...files] = process.argv.slice(1);
const require = createRequire(webDir + '/package.json');
const { NodeIO } = require('@gltf-transform/core');
const { ALL_EXTENSIONS } = require('@gltf-transform/extensions');
const { MeshoptDecoder } = require('meshoptimizer');
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const res = {};
for (const spec of files) {
  const eq = spec.indexOf('=');
  const doc = await io.readBinary(new Uint8Array(readFileSync(spec.slice(eq + 1))));
  const verts = [];
  for (const node of doc.getRoot().listNodes()) {
    const mesh = node.getMesh();
    if (!mesh) continue;
    const m = node.getWorldMatrix();
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION');
      const el = [0, 0, 0];
      for (let i = 0; i < pos.getCount(); i++) {
        pos.getElement(i, el);
        verts.push([
          m[0] * el[0] + m[4] * el[1] + m[8] * el[2] + m[12],
          m[1] * el[0] + m[5] * el[1] + m[9] * el[2] + m[13],
          m[2] * el[0] + m[6] * el[1] + m[10] * el[2] + m[14],
        ]);
      }
    }
  }
  res[spec.slice(0, eq)] = verts;
}
writeFileSync(out, JSON.stringify(res));
"""


def _compressed(data: bytes) -> bool:
    doc = read_glb_json(data)
    return "EXT_meshopt_compression" in {*doc.get("extensionsRequired", []), *doc.get("extensionsUsed", [])}


def glb_vertices(assets: Mapping[str, bytes], web_dir: str | Path = WEB_DIR) -> dict[str, np.ndarray]:
    """Decoded vertices ``(n, 3)`` (node transforms applied, body frame) of GLB files by asset id.
    Uncompressed files are read in Python; meshopt-compressed ones (what the site serves) are
    decoded by Node with the web app's packages.

    Raises:
        RuntimeError: a compressed file cannot be decoded here (no Node, or the web packages are not
            installed: ``pnpm install --frozen-lockfile`` in ``web/``).
    """
    out: dict[str, np.ndarray] = {}
    packed: dict[str, bytes] = {}
    for aid, data in assets.items():
        if _compressed(data):
            packed[aid] = data
        else:
            doc = read_glb(data)
            out[aid] = np.concatenate([m.vertices for m in doc.meshes]) if doc.meshes else np.zeros((0, 3))
    if packed:
        web = Path(web_dir)
        node = shutil.which("node")
        needs = ("@gltf-transform/core", "@gltf-transform/extensions", "meshoptimizer")
        if node is None or not all((web / "node_modules" / m / "package.json").is_file() for m in needs):
            raise RuntimeError(
                "the served GLBs are meshopt-compressed: checking them needs Node and the web app's "
                "packages (pnpm install --frozen-lockfile in web/)"
            )
        with tempfile.TemporaryDirectory(prefix="rtt-glbcheck-") as tmp:
            args = []
            for aid, data in sorted(packed.items()):
                f = Path(tmp) / f"{aid}.glb"
                f.write_bytes(data)
                args.append(f"{aid}={f}")
            dest = Path(tmp) / "verts.json"
            proc = subprocess.run(
                [node, "--input-type=module", "-e", _DECODE_JS, str(web), str(dest), *args],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                raise RuntimeError("decoding the served GLBs failed: " + " | ".join(tail))
            decoded = json.loads(dest.read_text(encoding="utf-8"))
        for aid, verts in decoded.items():
            out[aid] = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    return out


def _check_glb(
    d: _Drawn, assets: Mapping[str, bytes] | None, web_dir: str | Path
) -> tuple[list[str], dict[str, Any]]:
    from scipy.spatial import cKDTree

    header, rec, keys = d.header, d.rec, d.keys
    items: list[tuple[str, str, Mapping[str, Any], np.ndarray, np.ndarray]] = []
    problems: list[str] = []
    for gi, g in enumerate(d.groups):
        vis = header["bodies"][gi].get("visuals") or []
        if not vis:
            continue  # drawn from its geoms, as they are
        if len(vis) != len(g.members):
            problems.append(
                f"glb: body {header['bodies'][gi]['name']} has {len(vis)} visuals for {len(g.members)} parts"
            )
            continue
        for v, m in zip(vis, g.members, strict=True):
            pos, rot = sr._relative(d.tracks[0, g.rep], d.tracks[0, m])
            items.append((header["bodies"][gi]["name"], keys[m], v, pos, rot))
    meshed = [b for b in d.static if rec.bodies[keys[b]].asset not in (None, ...)]
    for v, b in zip(header.get("static_visuals", []), meshed, strict=False):
        items.append(("static", keys[b], v, d.tracks[0, b, :3], sr._rot(d.tracks[0, b, 3:])))
    if not items:
        return problems, {"max_deviation_mm": 0.0, "visuals": 0, "tolerance_mm": 1000 * GLB_TOL_M}
    need = sorted({v["asset"] for _, _, v, _, _ in items})
    if assets is None:
        return problems + ["glb: no GLB files were given to check the drawn meshes against their geoms"], {}
    absent = [a for a in need if a not in assets]
    if absent:
        return problems + [f"glb: no bytes for asset {absent[0]} (and {len(absent) - 1} more)"], {}
    try:
        verts = glb_vertices({a: assets[a] for a in need}, web_dir)
    except RuntimeError as exc:
        return problems + [f"glb: {exc}"], {}
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:  # a hung or missing Node, a bad file
        return problems + [f"glb: the served GLBs could not be decoded ({type(exc).__name__}: {exc})"], {}
    refs: dict[str, np.ndarray] = {}
    worst, worst_item = 0.0, ("", "")
    for where, key, v, pos, rot in items:
        if key not in refs:
            refs[key] = np.asarray(sr.mesh_of_geoms(rec.bodies[key].geoms).vertices)
        placed = (verts[v["asset"]] * np.asarray(v["scale"])) @ sr._rot(
            np.asarray(v["quat_xyzw"])
        ).T + np.asarray(v["pos"])
        ref = refs[key] @ rot.T + pos
        dev = max(float(cKDTree(ref).query(placed)[0].max()), float(cKDTree(placed).query(ref)[0].max()))
        if dev > worst:
            worst, worst_item = dev, (where, key)
    metrics = {
        "max_deviation_mm": round(1000 * worst, 4),
        "visuals": len(items),
        "worst_body": worst_item[1],
        "tolerance_mm": 1000 * GLB_TOL_M,
    }
    if worst > GLB_TOL_M:
        problems.append(
            f"glb: {worst_item[1]} is drawn {1000 * worst:.3f} mm from its collision geoms "
            f"(at most {1000 * GLB_TOL_M:.2f} mm)"
        )
    return problems, metrics


# --------------------------------------------------------------------------- the gate


def check_lane(
    report: Mapping[str, Any],
    lane: Any,
    *,
    assets: Mapping[str, bytes] | None = None,
    checks: Sequence[str] | None = None,
    dense: sr.Recording | None = None,
    web_dir: str | Path = WEB_DIR,
    log: Callable[[str], Any] | None = None,
) -> QAResult:
    """Judge one lane (module docstring).

    ``report`` is the tournament report the lane's run belongs to (it holds the scored
    ``traj_sha``), ``lane`` a :class:`rtt.studio.replay.LaneBuild` with its ``recording`` and its
    ``replay`` as it will be written (served asset ids in its header), ``assets`` the GLB bytes by the
    ids that header uses (the ``glb`` check fails without them). ``checks`` runs a subset of
    :data:`CHECKS`; ``dense`` is a re-run to measure against instead of making the 500 Hz one.

    Raises:
        ValueError: the lane has no recording (there is nothing to measure it against).
    """
    if getattr(lane, "recording", None) is None:
        raise ValueError("the lane has no recording: build it with rtt.studio.replay.export_lanes")
    unknown = sorted(set(checks or ()) - set(CHECKS))
    if unknown:
        raise ValueError(f"unknown checks {unknown}; known: {list(CHECKS)}")
    want = [c for c in CHECKS if checks is None or c in checks]
    say = log or (lambda _msg: None)
    rec: sr.Recording = lane.recording
    problems: list[str] = []
    metrics: dict[str, Any] = {}

    def run_dense() -> sr.Recording:
        nonlocal dense
        if dense is None:
            say("    re-running at 500 Hz")
            dense = sr.record_run(
                rec.setup,
                rec.task_spec,
                rec.condition,
                fps=QA_RECORD_FPS,
                task=rec.task,
                controllers=rec.controllers,
            )
        return dense

    if "traj_sha" in want:
        p, m = _check_sha(report, lane, run_dense())
        problems += p
        metrics["traj_sha"] = m
        if p:  # a run that is not the scored one is not worth measuring
            metrics["skipped"] = [c for c in want if c != "traj_sha"]
            return QAResult(False, problems, metrics)
    header_errors = sr.check_lane_header(lane.replay.header, lane.replay.frames)
    if header_errors:
        return QAResult(False, [f"header: {e}" for e in header_errors], metrics)
    try:
        d = _Drawn(lane)
    except sr.ReplayMismatchError as exc:
        return QAResult(False, [f"header: {exc}", *problems], metrics)
    if len(d.groups) != int(d.header["n_bodies"]):
        problems.append(
            f"header: the lane draws {d.header['n_bodies']} bodies, the recording makes {len(d.groups)}"
        )
        return QAResult(False, problems, metrics)
    if "cuts" in want:
        p, m = _check_cuts(d)
        problems += p
        metrics["cuts"] = m
    measured = [c for c in ("interpolation", "jumps", "penetration") if c in want]
    if measured:
        instants = d.instants()
        drawn = d.keys_at(d.reps(instants))
        truth = _Truth(d, run_dense())  # the simulation: every measured check is against it
        sim = truth.at(instants)
        if "interpolation" in want:
            p, m = _check_interpolation(d, truth, instants, drawn, sim)
            problems += p
            metrics["interpolation"] = m
        if "jumps" in want:
            p, m = _check_jumps(d, truth, instants, drawn, sim)
            problems += p
            metrics["jumps"] = m
        if "penetration" in want:
            say("    drawn-world penetration")
            p, m = _check_penetration(d, truth, instants, drawn, sim)
            problems += p
            metrics["penetration"] = m
    if "glb" in want:
        p, m = _check_glb(d, assets, web_dir)
        problems += p
        metrics["glb"] = m
    return QAResult(not problems, problems, metrics)


# --------------------------------------------------------------------------- frames for the vision review

_LOOKS: Final = {
    "world": (0.45, 0.42, 0.38, 1.0),
    "object": (0.25, 0.5, 0.9, 1.0),
    "tool": (1.0, 0.55, 0.15, 1.0),
    "wrist": (0.15, 0.15, 0.15, 1.0),
    "stack": (0.15, 0.15, 0.15, 1.0),
}


def render_frames(
    lane: Any,
    times: Sequence[float],
    *,
    size: tuple[int, int] = (640, 480),
    azimuth: float = 135.0,
    elevation: float = -25.0,
    distance: float = 0.75,
) -> list[bytes]:
    """PNG bytes of the drawn world (the collision geoms its GLBs are meshed from, flat colours: tool
    orange, objects blue, the wrist and bracket dark, the table brown) as the player draws the lane at
    each time in ``times``, looking at the wrist. An offscreen MuJoCo render (``MUJOCO_GL=egl`` unless
    set); raises ``RuntimeError`` where no GL context can be made."""
    import os

    from PIL import Image

    os.environ.setdefault("MUJOCO_GL", "egl")
    d = _Drawn(lane)
    world = _DrawnWorld(d.rec, d.keys, sky=True)
    model, data = world.model, world.data
    width, height = size
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    rgba = np.zeros((model.ngeom, 4))
    for g in range(model.ngeom):
        k = str(world.gkey[g])
        cat = "world" if k.startswith("world:") else d.rec.bodies[k].category
        rgba[g] = _LOOKS[cat]
    model.geom_rgba[:] = rgba
    try:
        renderer = mujoco.Renderer(model, height, width)
    except Exception as exc:
        raise RuntimeError(f"cannot make an offscreen render context ({type(exc).__name__}: {exc})") from exc
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    wrist = d.keys.index("wrist_flange") if "wrist_flange" in d.keys else None
    out: list[bytes] = []
    try:
        for when in times:
            poses = d.keys_at(d.reps(np.array([float(when)])))[0]
            world.pose(poses)
            cam.lookat[:] = poses[wrist, :3] if wrist is not None else poses[:, :3].mean(axis=0)
            renderer.update_scene(data, camera=cam)
            buf = io.BytesIO()
            Image.fromarray(renderer.render()).save(buf, format="PNG")
            out.append(buf.getvalue())
    finally:
        renderer.close()
    return out
