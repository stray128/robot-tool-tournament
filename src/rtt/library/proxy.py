"""Rigid simulation proxies for items that ship meshes but no simulation model.

A **rigid proxy** turns one printed part (a hook, a pusher, a tool body) into a MuJoCo body:

1. load the mesh (STL units are millimetres: ``scale = 0.001``), merge vertices, drop
   degenerate faces; refuse anything larger than 0.5 m (a unit error);
2. turn it into the mount frame: ``down`` (the tool's working direction in the mesh frame)
   to -z, ``forward`` to +y, then raise it so its highest point sits on the mount plane
   (z = 0) and centre it on the mount axis (x = y = 0 at the top face's centre of bounds);
3. split it into convex pieces with CoACD (threshold 0.02, seed 0, at most ``max_parts``),
   the project's pinned decomposition rule; MuJoCo collides each piece as its convex hull;
4. mass properties from the mesh volume x material density x infill (PLA 1.24, PETG 1.27,
   TPU 1.21, aluminium 2.70 g/cm3: typical datasheet values, assumed), centre of mass and
   inertia from the mesh (trimesh), or from the convex pieces when the mesh does not enclose
   a volume;
5. the result: OBJ files of the pieces and an MJCF with one body, explicit inertial, a
   ``tcp`` site at the lowest point and an optional spindle (a powered tool's rotating bit:
   a child body on a hinge along the tool axis driven by a velocity servo).

Fidelity is ``rigid``: the shape is the part's own, the material is assumed homogeneous, and
nothing bends.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import trimesh

__all__ = ["DENSITY_G_CM3", "ProxyPart", "RigidProxy", "build_rigid_proxy", "load_mesh", "mount_transform"]

#: g/cm3, typical datasheet values (assumed).
DENSITY_G_CM3: Final[Mapping[str, float]] = {"PLA": 1.24, "PETG": 1.27, "TPU": 1.21, "aluminium": 2.70}
MAX_SIZE_M: Final = 0.5
COACD_THRESHOLD: Final = 0.02
COACD_SEED: Final = 0


def load_mesh(data: bytes, file_type: str, scale: float) -> trimesh.Trimesh:
    """A clean triangle mesh from file bytes (``scale``: file units to metres)."""
    import io

    loaded = trimesh.load(io.BytesIO(data), file_type=file_type, force="mesh", process=False)
    mesh = trimesh.Trimesh(
        np.asarray(loaded.vertices, np.float64) * scale, np.asarray(loaded.faces), process=False
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    size = float((mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max())
    if not 0.001 < size <= MAX_SIZE_M:
        raise ValueError(f"mesh is {size:.4f} m across: a unit error (expected 1 mm to 0.5 m)")
    return mesh


def mount_transform(down: Sequence[float], forward: Sequence[float]) -> np.ndarray:
    """4x4 rotation taking ``down`` to -z and ``forward`` (made orthogonal) to +y."""
    u = np.asarray(down, np.float64)
    u = u / np.linalg.norm(u)
    v = np.asarray(forward, np.float64)
    v = v - (v @ u) * u
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("forward must not be parallel to down")
    v = v / np.linalg.norm(v)
    w = np.cross(v, u)
    src = np.column_stack([u, v, w])
    dst = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    tf = np.eye(4)
    tf[:3, :3] = dst @ src.T
    return tf


def _decompose(mesh: trimesh.Trimesh, max_parts: int) -> list[trimesh.Trimesh]:
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)),
        threshold=COACD_THRESHOLD,
        max_convex_hull=max_parts,
        preprocess_mode="auto",
        max_ch_vertex=64,
        seed=COACD_SEED,
    )
    out = []
    for verts, _faces in parts:
        hull = trimesh.Trimesh(np.asarray(verts, np.float64), process=False).convex_hull
        if hull.volume > 1e-10:
            out.append(hull)
    return out


@dataclass(frozen=True)
class ProxyPart:
    """One rigid body of a proxy: its convex pieces and mass properties (mount frame, metres)."""

    name: str
    mesh: trimesh.Trimesh
    pieces: tuple[trimesh.Trimesh, ...]
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]  # about com: ixx iyy izz ixy ixz iyz
    from_volume: bool  # mass from the closed mesh (True) or from the convex pieces (False)


@dataclass(frozen=True)
class RigidProxy:
    """A generated rigid proxy (module docstring)."""

    parts: tuple[ProxyPart, ...]
    material: str
    infill: float
    reach: float  # m, mount plane to the lowest point
    notes: tuple[str, ...] = field(default=())

    @property
    def mass(self) -> float:
        return float(sum(p.mass for p in self.parts))


def _mass_properties(
    mesh: trimesh.Trimesh, pieces: Sequence[trimesh.Trimesh], density: float
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 1e-9:
        m = mesh.copy()
        m.density = density
        return float(m.mass), np.asarray(m.center_mass), np.asarray(m.moment_inertia), True
    total = trimesh.util.concatenate(list(pieces))
    total.density = density
    return float(total.mass), np.asarray(total.center_mass), np.asarray(total.moment_inertia), False


def build_rigid_proxy(
    meshes: Mapping[str, trimesh.Trimesh],
    *,
    down: Sequence[float],
    forward: Sequence[float],
    material: str = "PLA",
    infill: float = 1.0,
    max_parts: int = 16,
    offsets: Mapping[str, Sequence[float]] | None = None,
    stack: bool = False,
) -> RigidProxy:
    """Proxy bodies for ``meshes`` (name -> mesh in metres, one rigid part each, in one frame).

    All parts share the mount transform; the first part is the root. ``offsets`` shifts a part
    (metres, mount frame) after the transform, for parts drawn in separate files.
    """
    if material not in DENSITY_G_CM3:
        raise ValueError(f"material must be one of {sorted(DENSITY_G_CM3)}")
    if not 0.0 < infill <= 1.0:
        raise ValueError("infill must be in (0, 1]")
    density = DENSITY_G_CM3[material] * 1000.0 * infill  # kg/m3
    tf = mount_transform(down, forward)
    moved = {}
    for name, mesh in meshes.items():
        m = mesh.copy()
        m.apply_transform(tf)
        if offsets and name in offsets:
            m.apply_translation(np.asarray(offsets[name], np.float64))
        moved[name] = m
    if stack:  # parts drawn in separate files: each one centred under the one before it
        names = list(moved)
        for prev, cur in itertools.pairwise(names):
            lo_p, hi_p = moved[prev].bounds
            lo_c, hi_c = moved[cur].bounds
            centre_p = 0.5 * (lo_p[:2] + hi_p[:2])
            centre_c = 0.5 * (lo_c[:2] + hi_c[:2])
            moved[cur].apply_translation(
                [centre_p[0] - centre_c[0], centre_p[1] - centre_c[1], lo_p[2] - hi_c[2]]
            )
    allv = np.vstack([m.vertices for m in moved.values()])
    top = float(allv[:, 2].max())
    root = next(iter(moved.values()))
    root_top = float(root.vertices[:, 2].max())
    top_band = root.vertices[root.vertices[:, 2] > root_top - 0.002]
    centre = 0.5 * (top_band[:, :2].min(axis=0) + top_band[:, :2].max(axis=0))
    shift = np.array([-centre[0], -centre[1], -top])
    parts = []
    for name, m in moved.items():
        m.apply_translation(shift)
        pieces = _decompose(m, max_parts)
        if not pieces:
            pieces = [m.convex_hull]
        mass, com, inertia, from_volume = _mass_properties(m, pieces, density)
        parts.append(
            ProxyPart(
                name=name,
                mesh=m,
                pieces=tuple(pieces),
                mass=mass,
                com=tuple(float(v) for v in com),  # type: ignore[arg-type]
                inertia=(
                    float(inertia[0, 0]),
                    float(inertia[1, 1]),
                    float(inertia[2, 2]),
                    float(inertia[0, 1]),
                    float(inertia[0, 2]),
                    float(inertia[1, 2]),
                ),
                from_volume=from_volume,
            )
        )
    bottom = float(min(p.mesh.vertices[:, 2].min() for p in parts))
    notes = [
        (
            f"rigid proxy: {sum(len(p.pieces) for p in parts)} convex pieces (CoACD threshold {COACD_THRESHOLD}, "
            f"seed {COACD_SEED}); {material} at {DENSITY_G_CM3[material]} g/cm3 x {infill:g} infill (assumed)"
        ),
    ]
    if not all(p.from_volume for p in parts):
        notes.append("mass from the convex pieces: the mesh does not enclose a volume")
    return RigidProxy(parts=tuple(parts), material=material, infill=infill, reach=-bottom, notes=tuple(notes))


def obj_text(mesh: trimesh.Trimesh, label: str) -> bytes:
    """A convex piece as OBJ text (metres), deterministic."""
    lines = [f"# {label}", "# convex piece of a rigid proxy (CoACD); metres, mount frame"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(mesh.vertices)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(mesh.faces)]
    return ("\n".join(lines) + "\n").encode()


def proxy_mjcf(
    proxy: RigidProxy,
    model: str,
    piece_files: Mapping[str, Sequence[str]],
    *,
    spindle: Mapping[str, Any] | None = None,
) -> str:
    """The proxy as MJCF text: the root part's body, and a spindle child body if given.

    ``piece_files``: part name -> its pieces' OBJ paths relative to the MJCF. ``spindle``:
    ``{"part": name, "axis": [0, 0, 1], "max_speed": rad/s, "max_torque": N m}`` puts that part
    on a hinge about ``axis`` through its own centre of bounds, driven by a velocity servo.
    """

    def f(v: float) -> str:
        return f"{v:.9g}"

    lines = [f'<mujoco model="{model}">', '  <compiler angle="radian"/>', "  <asset>"]
    for part in proxy.parts:
        for i, path in enumerate(piece_files[part.name]):
            lines.append(f'    <mesh name="{part.name}_{i}" file="{path}"/>')
    lines += ["  </asset>", "  <worldbody>"]
    root = proxy.parts[0]

    def inertial(p: ProxyPart, origin: np.ndarray) -> str:
        c = np.asarray(p.com) - origin
        ixx, iyy, izz, ixy, ixz, iyz = p.inertia
        return (
            f'<inertial pos="{f(c[0])} {f(c[1])} {f(c[2])}" mass="{f(p.mass)}" '
            f'fullinertia="{f(ixx)} {f(iyy)} {f(izz)} {f(ixy)} {f(ixz)} {f(iyz)}"/>'
        )

    lines.append(f'    <body name="{root.name}">')
    lines.append("      " + inertial(root, np.zeros(3)))
    for i in range(len(piece_files[root.name])):
        lines.append(f'      <geom type="mesh" mesh="{root.name}_{i}"/>')
    lowest = min(float(p.mesh.vertices[:, 2].min()) for p in proxy.parts)
    lines.append(f'      <site name="tcp" pos="0 0 {f(lowest)}" size="0.003"/>')
    for part in proxy.parts[1:]:
        if spindle is not None and spindle.get("part") == part.name:
            lo, hi = part.mesh.bounds
            origin = 0.5 * (lo + hi)
            axis = np.asarray(spindle.get("axis", (0.0, 0.0, 1.0)), np.float64)
            lines.append(
                f'      <body name="{part.name}" pos="{f(origin[0])} {f(origin[1])} {f(origin[2])}">'
            )
            lines.append("        " + inertial(part, origin))
            lines.append(
                f'        <joint name="spindle" type="hinge" axis="{f(axis[0])} {f(axis[1])} {f(axis[2])}" '
                'damping="0.001" armature="0.0001"/>'
            )
            for i, path in enumerate(piece_files[part.name]):
                # pieces are in the mount frame: shift them into the spindle body's frame
                lines.append(
                    f'        <geom type="mesh" mesh="{part.name}_{i}" pos="{f(-origin[0])} {f(-origin[1])} {f(-origin[2])}"/>'
                )
            lines.append("      </body>")
        else:
            lines.append(f'      <body name="{part.name}">')
            lines.append("        " + inertial(part, np.zeros(3)))
            for i in range(len(piece_files[part.name])):
                lines.append(f'        <geom type="mesh" mesh="{part.name}_{i}"/>')
            lines.append("      </body>")
    lines += ["    </body>", "  </worldbody>"]
    if spindle is not None:
        speed = float(spindle.get("max_speed", 10.0))
        torque = float(spindle.get("max_torque", 0.5))
        kv = float(spindle.get("kv", 0.05))
        lines += [
            "  <actuator>",
            (
                f'    <velocity name="spindle_speed" joint="spindle" kv="{f(kv)}" ctrlrange="{f(-speed)} {f(speed)}" '
                f'forcerange="{f(-torque)} {f(torque)}"/>'
            ),
            "  </actuator>",
        ]
    lines.append("</mujoco>")
    return "\n".join(lines) + "\n"
