"""Printable parts of a build pack: the multi-tool bracket and the tool-changer plates.

Both are built as solids with manifold3d, in millimetres, then refused unless each is exactly one
closed, consistently oriented solid of positive volume with no enclosed void (the print pack's own
check, :func:`rtt.part.pack._check_one_solid`). STL only: no CAD kernel is installed, so STEP is
not provided (``docs/STUDIO_SEARCH.md``).

**Bracket** (a ``multi_head`` setup), in the bracket frame the simulation uses
(:mod:`rtt.studio.scene`): origin at the centre of the face that bolts to the wrist flange, z up
toward the wrist, x the tilt axis of the heads. It is the union of

* the hub: a disc of the flange's 63 mm diameter, ``hub_thickness_mm`` thick (``z`` in
  ``[-hub, 0]``);
* one arm per head: the simulation's capsule, ``arm_width_mm`` across, from the hub's underside to
  the head's mount centre, trimmed flush with the hub's flange face and the adapter's tool face;
* one adapter per head: a 63 mm disc ``adapter_thickness_mm`` thick whose wrist-side face is the
  :class:`~rtt.studio.types.HeadMount` plane (tilted ``angle_deg`` about x, ``standoff_mm`` from
  the wrist centre, turned ``spin_deg``); the tool bolts to its other face. A tilted adapter may
  rise beside the flange, never into the arm's last link (the flange radius plus 10 mm).

**Changer plate** (a ``tool_change`` setup, one per tool): a 63 mm disc half the changer stack thick
that carries the tool on the changer's tool side, with a central pass-through bore.

Bolt patterns are ISO 9409-1-50-4-M6 [assumed: the flange pattern is not sourced from a target
arm]: 4 clearance holes of 6.6 mm on a 50 mm circle, on the hub, every adapter and both faces of a
changer plate (the changer side turned 45 deg). Check them against your arm and tools.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import trimesh
from manifold3d import Manifold

from rtt.effectors.scene import FLANGE_RADIUS, MOUNT_Z
from rtt.part.pack import _check_one_solid
from rtt.studio.types import BracketSpec, HeadMount, Setup, SetupKind, ToolChanger

__all__ = [
    "BOLT_CIRCLE_MM",
    "BOLT_HOLE_MM",
    "FLANGE_DIAMETER_MM",
    "PrintablePart",
    "bracket_part",
    "changer_plate_part",
    "check_printable",
    "printable_parts",
    "tool_reference_mesh",
    "write_binary_stl",
]

FLANGE_DIAMETER_MM: Final = 2000.0 * FLANGE_RADIUS
#: ISO 9409-1-50-4-M6 [assumed]: 4 x 6.6 mm clearance holes on a 50 mm circle.
BOLT_CIRCLE_MM: Final = 50.0
BOLT_HOLE_MM: Final = 6.6
BORE_MM: Final = 20.0  # changer plate pass-through bore [assumed]
#: The arm's last link above the flange: the flange's radius plus 10 mm (as the search's keep-out).
LAST_LINK_RADIUS_MM: Final = 1000.0 * FLANGE_RADIUS + 10.0
SEGMENTS: Final = 64


@dataclass(frozen=True)
class PrintablePart:
    """One generated printable part: its watertight mesh (mm) and what it is."""

    name: str
    kind: str  # "bracket" | "changer_plate"
    mesh: trimesh.Trimesh
    description: str
    density_kg_m3: float

    @property
    def volume_mm3(self) -> float:
        return float(self.mesh.volume)

    @property
    def mass_kg(self) -> float:
        return self.volume_mm3 * 1e-9 * self.density_kg_m3

    def facts(self) -> dict[str, Any]:
        lo, hi = self.mesh.bounds
        return {
            "name": self.name,
            "kind": self.kind,
            "volume_mm3": round(self.volume_mm3, 1),
            "mass_kg": round(self.mass_kg, 4),
            "bounds_mm": [[round(float(v), 2) for v in lo], [round(float(v), 2) for v in hi]],
            "triangles": len(self.mesh.faces),
            "watertight": bool(self.mesh.is_watertight),
            "genus": round((2 - int(self.mesh.euler_number)) / 2),
        }


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _affine(rot: np.ndarray, pos: Sequence[float]) -> list[list[float]]:
    return [[float(rot[i, 0]), float(rot[i, 1]), float(rot[i, 2]), float(pos[i])] for i in range(3)]


def _to_trimesh(solid: Manifold) -> trimesh.Trimesh:
    mesh = solid.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        faces=np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )


def _disc_with_holes(thickness: float, *, hole_angles: Sequence[float], bore: float = 0.0) -> Manifold:
    """A flange-sized disc, ``z`` in ``[-thickness, 0]``, with bolt holes (and a bore)."""
    disc = Manifold.cylinder(thickness, 0.5 * FLANGE_DIAMETER_MM, circular_segments=SEGMENTS).translate(
        [0.0, 0.0, -thickness]
    )
    cutters = []
    r = 0.5 * BOLT_CIRCLE_MM
    for a in hole_angles:
        x, y = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        cutters.append(
            Manifold.cylinder(thickness + 2.0, 0.5 * BOLT_HOLE_MM, circular_segments=32).translate(
                [x, y, -thickness - 1.0]
            )
        )
    if bore > 0:
        cutters.append(
            Manifold.cylinder(thickness + 2.0, 0.5 * bore, circular_segments=48).translate(
                [0.0, 0.0, -thickness - 1.0]
            )
        )
    for c in cutters:
        disc = disc - c
    return disc


_PATTERN: Final = (45.0, 135.0, 225.0, 315.0)
_PATTERN_TURNED: Final = (0.0, 90.0, 180.0, 270.0)


def check_printable(mesh: trimesh.Trimesh, what: str) -> None:
    """Refuse anything but one closed, outward-facing solid of positive volume with no void."""
    _check_one_solid(mesh, what)


def mount_centre_mm(mount: HeadMount) -> np.ndarray:
    """A head's mount centre in the bracket frame (mm): the scene's formula."""
    centre = _rot_x(mount.angle_deg) @ np.array([mount.lateral_mm, 0.0, -mount.standoff_mm])
    return centre - np.array([0.0, 0.0, 1000.0 * MOUNT_Z])


def bracket_part(mounts: Sequence[HeadMount], bracket: BracketSpec) -> PrintablePart:
    """The two-tool bracket as one watertight solid (module docstring)."""
    hub_t, arm_r, ad_t = bracket.hub_thickness_mm, 0.5 * bracket.arm_width_mm, bracket.adapter_thickness_mm
    solid = _disc_with_holes(hub_t, hole_angles=_PATTERN)
    start = np.array([0.0, 0.0, -hub_t])
    for m in mounts:
        centre = mount_centre_mm(m)
        rot = _rot_x(m.angle_deg) @ _rot_z(m.spin_deg)
        up = rot @ np.array([0.0, 0.0, 1.0])  # toward the wrist side of the adapter
        arm = Manifold.batch_hull(
            [
                Manifold.sphere(arm_r, circular_segments=48).translate(start.tolist()),
                Manifold.sphere(arm_r, circular_segments=48).translate(centre.tolist()),
            ]
        )
        arm = arm.trim_by_plane([0.0, 0.0, -1.0], 0.0)  # flush with the flange face
        tool_face = centre - ad_t * up
        arm = arm.trim_by_plane(up.tolist(), float(up @ tool_face))  # flush with the tool face
        adapter = _disc_with_holes(ad_t, hole_angles=_PATTERN).transform(_affine(rot, centre))
        solid = solid + arm + adapter
    mesh = _to_trimesh(solid)
    v = np.asarray(mesh.vertices)
    above = v[:, 2] > 0.5
    if np.any(above & (np.hypot(v[:, 0], v[:, 1]) < LAST_LINK_RADIUS_MM)):
        raise ValueError("the bracket reaches above the flange's mounting face within the arm's last link")
    check_printable(mesh, "the bracket")
    angles = ", ".join(f"{m.angle_deg:g}" for m in mounts)
    return PrintablePart(
        "bracket",
        "bracket",
        mesh,
        f"two-tool bracket: hub, two arms and two adapters; heads at {angles} deg, "
        f"{mounts[0].standoff_mm:g} mm standoff",
        bracket.density_kg_m3,
    )


def changer_plate_part(changer: ToolChanger, tool: int, density_kg_m3: float = 1300.0) -> PrintablePart:
    """The plate that carries one tool on the changer's tool side (module docstring)."""
    t = 0.5 * changer.stack_mm
    disc = _disc_with_holes(t, hole_angles=_PATTERN, bore=BORE_MM)
    r = 0.5 * BOLT_CIRCLE_MM
    for a in _PATTERN_TURNED:  # the changer side's pattern, turned 45 deg
        x, y = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        disc = disc - Manifold.cylinder(t + 2.0, 0.5 * BOLT_HOLE_MM, circular_segments=32).translate(
            [x, y, -t - 1.0]
        )
    mesh = _to_trimesh(disc)
    check_printable(mesh, f"changer plate {tool}")
    return PrintablePart(
        f"changer_plate_tool{tool}",
        "changer_plate",
        mesh,
        f"changer tool-side plate for tool {tool}: {FLANGE_DIAMETER_MM:.0f} mm x {t:g} mm, "
        f"{BORE_MM:g} mm pass-through bore",
        density_kg_m3,
    )


def printable_parts(setup: Setup) -> list[PrintablePart]:
    """Every generated printable part of ``setup`` (none for a single tool on the flange)."""
    if setup.kind is SetupKind.MULTI_HEAD:
        assert setup.bracket is not None
        return [bracket_part(setup.mounts, setup.bracket)]
    if setup.kind is SetupKind.TOOL_CHANGE:
        assert setup.changer is not None
        return [changer_plate_part(setup.changer, k) for k in range(len(setup.tools))]
    return []


def write_binary_stl(mesh: trimesh.Trimesh, path: Path, header: str = "rtt studio build pack, mm") -> Path:
    """A deterministic binary STL (fixed header, float32, facet normals from the winding)."""
    tris = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tris
    head = header.encode()[:80].ljust(80, b"\0")
    path.write_bytes(head + struct.pack("<I", len(tris)) + rec.tobytes())
    return path


def _primitive(model: Any, g: int) -> trimesh.Trimesh | None:
    """A geom's shape in its own frame (m), or ``None`` for a plane."""
    t, size = int(model.geom_type[g]), np.asarray(model.geom_size[g], dtype=np.float64)
    if t == 6:  # box
        return trimesh.creation.box(extents=2.0 * size)
    if t == 2:  # sphere
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if t == 3:  # capsule
        return trimesh.creation.capsule(height=2.0 * float(size[1]), radius=float(size[0]), count=[16, 16])
    if t == 5:  # cylinder
        return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]), sections=32)
    if t == 4:  # ellipsoid
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.apply_scale(size)
        return m
    if t == 7:  # mesh
        mid = int(model.geom_dataid[g])
        v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        return trimesh.Trimesh(
            vertices=np.asarray(model.mesh_vert[v0 : v0 + nv], dtype=np.float64),
            faces=np.asarray(model.mesh_face[f0 : f0 + nf], dtype=np.int64),
            process=False,
        )
    return None


def tool_reference_mesh(scene: Any) -> trimesh.Trimesh:
    """The active tool's simulated collision geometry in the wrist (palm) frame, in mm, roll 0:
    reference geometry for sizing and clearance checks (not a manufacturing model)."""
    import mujoco

    model = scene.model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    rp, pp = data.xmat[scene.ids.palm_body].reshape(3, 3), data.xpos[scene.ids.palm_body]
    pieces = []
    for g in scene.ids.active_geoms:
        shape = _primitive(model, int(g))
        if shape is None:
            continue
        r = rp.T @ data.geom_xmat[g].reshape(3, 3)
        t = rp.T @ (data.geom_xpos[g] - pp)
        tf = np.eye(4)
        tf[:3, :3], tf[:3, 3] = r, t
        shape.apply_transform(tf)
        pieces.append(shape)
    if not pieces:
        raise ValueError("the tool has no collision geometry")
    mesh = trimesh.util.concatenate(pieces)
    mesh.apply_scale(1000.0)
    return mesh
