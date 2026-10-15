"""Compact episode replays for the web viewer.

A replay is two files sharing a stem:

* ``<stem>.bin``: little-endian float32 array ``[frames x bodies x 7]`` holding
  each body frame's world pose ``(px, py, pz, qx, qy, qz, qw)``, sampled at
  ``fps`` (30). MuJoCo stores quaternions as wxyz; they are converted to the
  xyzw order three.js uses.
* ``<stem>.json``: a header with the layout, the body names, and for every
  body the primitive geoms needed to rebuild it (body-frame poses, xyzw
  quaternions, RGBA), plus the static table and floor. Coordinates are
  MuJoCo's Z-up world; a Y-up viewer rotates its root by -90 degrees about X.

Box geoms are described by ``half_size``; cylinders by ``radius`` and
``half_height`` (axis = geom local Z); capsules by ``radius`` and
``half_length`` of the cylindrical part (axis = geom local Z); convex mesh
pieces (mesh objects) by flat ``vertices`` (x, y, z triples, geom frame) and
``faces`` (vertex-index triples, counter-clockwise seen from outside).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rtt.sim.scene import REPLAY_BODIES, SceneIds

FORMAT = "rtt-replay"
VERSION = 1
POSE_FIELDS = ("px", "py", "pz", "qx", "qy", "qz", "qw")


@dataclass
class Replay:
    """Pose frames plus the JSON header describing how to draw them."""

    frames: np.ndarray  # float32, (n_frames, n_bodies, 7)
    header: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder quaternion components along the last axis from wxyz to xyzw."""
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _round(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _geom_entry(model: mujoco.MjModel, g: int) -> dict[str, Any]:
    gtype = mujoco.mjtGeom(model.geom_type[g])
    size = model.geom_size[g]
    entry: dict[str, Any] = {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"}
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        entry.update(type="box", half_size=_round(size))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        entry.update(type="cylinder", radius=round(float(size[0]), 6), half_height=round(float(size[1]), 6))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        entry.update(type="capsule", radius=round(float(size[0]), 6), half_length=round(float(size[1]), 6))
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        entry.update(
            type="mesh",
            vertices=_round(model.mesh_vert[va : va + vn].ravel()),
            faces=[int(i) for i in model.mesh_face[fa : fa + fn].ravel()],
        )
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        entry.update(type="plane", half_size=_round(size[:2]))
    else:
        raise ValueError(f"replay export does not support geom type {gtype.name}")
    entry.update(
        pos=_round(model.geom_pos[g]),
        quat_xyzw=_round(wxyz_to_xyzw(model.geom_quat[g])),
        rgba=_round(model.geom_rgba[g], 3),
    )
    return entry


def scene_header(model: mujoco.MjModel, ids: SceneIds, fps: int) -> dict[str, Any]:
    """Static description of the scene for a replay (no frame count yet)."""
    bodies = []
    for name, body_id in zip(REPLAY_BODIES, ids.replay_bodies, strict=True):
        geoms = [_geom_entry(model, g) for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        bodies.append({"name": name, "geoms": geoms})
    static = [_geom_entry(model, g) for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    return {
        "format": FORMAT,
        "version": VERSION,
        "fps": fps,
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": list(POSE_FIELDS),
        "up_axis": "z",
        "units": "m",
        "bodies": bodies,
        "static": static,
    }


class ReplayRecorder:
    """Samples body poses (and full qpos, for video rendering) at a fixed frame rate."""

    def __init__(self, model: mujoco.MjModel, ids: SceneIds, fps: int) -> None:
        self._model = model
        self._ids = ids
        self.fps = fps
        self._next_t = 0.0
        self._poses: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._half_step = 0.5 * model.opt.timestep

    def maybe_record(self, data: mujoco.MjData) -> None:
        """Record a frame if the simulation clock has reached the next frame time."""
        if data.time + self._half_step >= self._next_t:
            body = self._ids.replay_bodies
            pose = np.empty((len(body), 7), dtype=np.float32)
            pose[:, :3] = data.xpos[body]
            pose[:, 3:] = wxyz_to_xyzw(data.xquat[body])
            self._poses.append(pose)
            self._qpos.append(data.qpos.copy())
            self._next_t = len(self._poses) / self.fps

    def finish(self, meta: dict[str, Any] | None = None) -> tuple[Replay, np.ndarray]:
        """Return the replay and the matching ``(n_frames, nq)`` qpos trajectory."""
        frames = np.stack(self._poses).astype(np.float32, copy=False)
        header = scene_header(self._model, self._ids, self.fps)
        header["n_frames"] = int(frames.shape[0])
        header["n_bodies"] = int(frames.shape[1])
        header["meta"] = meta or {}
        return Replay(frames=frames, header=header), np.stack(self._qpos)


def write_replay(replay: Replay, stem: str | Path) -> tuple[Path, Path]:
    """Write ``<stem>.bin`` and ``<stem>.json``; returns both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    bin_path, json_path = stem.with_suffix(".bin"), stem.with_suffix(".json")
    frames = np.ascontiguousarray(replay.frames, dtype="<f4")
    bin_path.write_bytes(frames.tobytes())
    header = dict(replay.header, bin=bin_path.name, n_frames=int(frames.shape[0]))
    json_path.write_text(json.dumps(header, indent=1) + "\n", encoding="utf-8")
    return bin_path, json_path


def read_replay(stem: str | Path) -> Replay:
    """Load a replay written by :func:`write_replay`."""
    stem = Path(stem)
    header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    raw = np.frombuffer(stem.with_suffix(".bin").read_bytes(), dtype="<f4")
    frames = raw.reshape(header["n_frames"], header["n_bodies"], len(POSE_FIELDS)).astype(np.float32)
    return Replay(frames=frames, header=header)
