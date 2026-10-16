"""Printable jaw: a carriage block for the gripper's guide rods, fused with a generated finger.

The simulated gripper is an open-source rack-and-pinion parallel gripper whose two
jaws slide on a pair of guide rods; each stock jaw is one piece, a carriage block
inside the frame plus the finger below the frame's front face. :func:`build_jaw`
reproduces that clamp interface from documented dimensions and fuses it with any
generated finger into one watertight solid (manifold3d), so a design can be printed
as a drop-in jaw.

Every dimension of :class:`ClampInterface` is labelled with its source:

* ``documented``: measured from the gripper's published hardware sources
  (CERN-OHL-P-2.0; attribution in the project README): the two rod bores
  (6.1 mm at 32 mm centres), the jaw's 42 x 16 mm root footprint, the 42 mm stroke
  and the module 0.8, 18-tooth pinion (7.2 mm pitch radius).
* ``assumed``: not documented anywhere we could check: the block height, the rod
  axis height and, when enabled, the rack's position, length and face width.

No CAD file from that project is used or included: the geometry is built here from
those numbers. The drive (rack) is **off by default** because its position is
assumed; enable it with ``ClampInterface(rack=True)`` or transfer the drive from a
stock jaw. Check every ``assumed`` value against your gripper before printing.

Frame: the left finger's frame (:mod:`rtt.design.types`), in millimetres. The finger
root sits on the frame's front face ``z = 0`` and hangs toward -Z; its inner (contact)
face is the plane ``y = 0`` and the body lies at ``y >= 0``; the block occupies
``z >= 0`` above it. The right jaw is the same part turned 180 degrees about Z (the
finger is symmetric across its width), so one STL serves both jaws.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Final

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold, Mesh, OpType

from rtt.design.export import to_trimesh
from rtt.design.types import FingerGeometry

__all__ = [
    "DIMENSION_SOURCES",
    "ClampInterface",
    "build_jaw",
    "jaw_print_orientation",
    "manifold_from_trimesh",
    "rack_profile",
]

WELD_MM: Final = 0.01
"""The block reaches this far below the frame face so it fuses with the finger root."""
SEGMENTS: Final = 64


@dataclass(frozen=True)
class ClampInterface:
    """Dimensions of the jaw's clamp interface, in millimetres (see the module docstring)."""

    rod_bore_d_mm: float = 6.1
    rod_spacing_mm: float = 32.0
    block_width_mm: float = 42.0
    block_depth_mm: float = 16.0
    block_height_mm: float = 18.0
    rod_axis_height_mm: float = 9.0
    stroke_mm: float = 42.0
    rack: bool = False
    rack_module_mm: float = 0.8
    pinion_teeth: int = 18
    pressure_angle_deg: float = 20.0
    rack_length_mm: float = 58.0
    rack_face_width_mm: float = 6.0
    rack_bar_mm: float = 3.0
    rack_z_mm: float = 9.0
    clearance_mm: float = 0.5

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool):
                continue
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ValueError(f"{f.name} must be a positive number, got {value!r}")
        if self.rod_spacing_mm / 2.0 + self.rod_bore_d_mm / 2.0 >= self.block_width_mm / 2.0:
            raise ValueError("the rod bores must lie inside the block width")
        r = self.rod_bore_d_mm / 2.0
        if not r < self.rod_axis_height_mm < self.block_height_mm - r:
            raise ValueError("the rod axis must leave material above and below the bores")
        if self.rack:
            outer = self.pitch_radius_mm + 1.25 * self.rack_module_mm + self.rack_bar_mm + self.clearance_mm
            if outer >= self.rod_spacing_mm / 2.0 - r:
                raise ValueError("the rack and its clearance slot would cut into the rod bores")
            if self.rack_length_mm < self.stroke_mm + 2.0 * self.pitch_radius_mm:
                raise ValueError("the rack must cover the stroke plus the pinion's engagement")
            half = self.rack_face_width_mm / 2.0 + self.clearance_mm
            if not half < self.rack_z_mm < self.block_height_mm - half:
                raise ValueError("the rack must sit inside the block height")

    @property
    def pitch_radius_mm(self) -> float:
        return self.rack_module_mm * self.pinion_teeth / 2.0

    def describe(self) -> dict[str, Any]:
        """Every dimension with its value and source (``documented`` or ``assumed``)."""
        return {
            k: {"value": v, "source": DIMENSION_SOURCES.get(k, "setting")} for k, v in asdict(self).items()
        }


DIMENSION_SOURCES: Final[dict[str, str]] = {
    "rod_bore_d_mm": "documented",
    "rod_spacing_mm": "documented",
    "block_width_mm": "documented",
    "block_depth_mm": "documented",
    "block_height_mm": "assumed",
    "rod_axis_height_mm": "assumed",
    "stroke_mm": "documented",
    "rack": "setting",
    "rack_module_mm": "documented",
    "pinion_teeth": "documented",
    "pressure_angle_deg": "assumed",
    "rack_length_mm": "assumed",
    "rack_face_width_mm": "assumed",
    "rack_bar_mm": "assumed",
    "rack_z_mm": "assumed",
    "clearance_mm": "assumed",
}


def manifold_from_trimesh(mesh: trimesh.Trimesh) -> Manifold:
    """A manifold3d solid from a watertight trimesh (raises if it is not a valid manifold)."""
    solid = Manifold(
        Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
    )
    status = solid.status()
    if solid.is_empty() or "NoError" not in str(status):
        raise ValueError(f"mesh is not a valid closed solid ({status})")
    return solid


def _to_trimesh(solid: Manifold) -> trimesh.Trimesh:
    mesh = solid.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        faces=np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )


def rack_profile(iface: ClampInterface) -> CrossSection:
    """Left jaw's rack in the x-y plane: teeth toward -x on the pitch line ``x = r_p``, along -y."""
    m = iface.rack_module_mm
    pitch = math.pi * m
    tan_a = math.tan(math.radians(iface.pressure_angle_deg))
    x_pitch = iface.pitch_radius_mm
    x_tip, x_root = x_pitch - m, x_pitch + 1.25 * m
    half_tip = pitch / 4.0 - m * tan_a
    half_root = pitch / 4.0 + 1.25 * m * tan_a
    y_end, y_start = -iface.rack_length_mm, iface.block_depth_mm / 2.0
    bar = CrossSection.square((iface.rack_bar_mm + 0.01, y_start - y_end)).translate((x_root - 0.01, y_end))
    teeth = []
    n = int((0.0 - y_end) // pitch)
    for i in range(n):
        yc = -pitch / 2.0 - i * pitch
        teeth.append(
            CrossSection(
                [
                    [
                        (x_root, yc - half_root),
                        (x_root, yc + half_root),
                        (x_tip, yc + half_tip),
                        (x_tip, yc - half_tip),
                    ]
                ]
            )
        )
    return CrossSection.batch_boolean([bar, *teeth], OpType.Add)


def _block(iface: ClampInterface) -> Manifold:
    w, d, h = iface.block_width_mm, iface.block_depth_mm, iface.block_height_mm
    block = Manifold.cube((w, d, h + WELD_MM)).translate((-w / 2.0, 0.0, -WELD_MM))
    bore = (
        Manifold.cylinder(d + 2.0, iface.rod_bore_d_mm / 2.0, circular_segments=SEGMENTS)
        .rotate((-90.0, 0.0, 0.0))
        .translate((0.0, -1.0, iface.rod_axis_height_mm))
    )
    s = iface.rod_spacing_mm / 2.0
    cuts = [bore.translate((-s, 0.0, 0.0)), bore.translate((s, 0.0, 0.0))]
    if iface.rack:
        c = iface.clearance_mm
        m = iface.rack_module_mm
        fw = iface.rack_face_width_mm + 2.0 * c
        z0 = iface.rack_z_mm - fw / 2.0
        # Slot for the opposite jaw's rack (it runs on the -x side of the pinion) ...
        x_in = iface.pitch_radius_mm - m - c
        x_out = iface.pitch_radius_mm + 1.25 * m + iface.rack_bar_mm + c
        cuts.append(Manifold.cube((x_out - x_in, d + 2.0, fw)).translate((-x_out, -1.0, z0)))
        # ... and a pocket for the pinion, which enters the block near full close.
        r_tip = iface.pitch_radius_mm + m + c
        cuts.append(Manifold.cube((2.0 * r_tip, r_tip + 1.0, fw)).translate((-r_tip, -1.0, z0)))
    return block - Manifold.batch_boolean(cuts, OpType.Add)


def build_jaw(geom: FingerGeometry, iface: ClampInterface | None = None) -> trimesh.Trimesh:
    """One jaw (carriage block, optional rack, finger) as a watertight mesh in mm, assembly frame."""
    iface = iface or ClampInterface()
    finger_mesh = to_trimesh(geom).copy()
    finger_mesh.apply_scale(1000.0)
    lo, hi = finger_mesh.bounds
    if lo[0] < -iface.block_width_mm / 2.0 or hi[0] > iface.block_width_mm / 2.0:
        raise ValueError("the finger is wider than the carriage block")
    parts = [_block(iface), manifold_from_trimesh(finger_mesh)]
    if iface.rack:
        fw = iface.rack_face_width_mm
        parts.append(rack_profile(iface).extrude(fw).translate((0.0, 0.0, iface.rack_z_mm - fw / 2.0)))
    solid = Manifold.batch_boolean(parts, OpType.Add)
    if solid.is_empty():
        raise RuntimeError("the jaw union is empty")
    return _to_trimesh(solid)


def jaw_print_orientation(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, bool]:
    """The jaw turned for printing: inner face (``y = 0``) down on the bed, bores vertical.

    Returns the moved mesh (resting on ``z = 0``, centred in x-y) and whether anything
    hangs below the inner-face plane (an inward bend or hook), i.e. needs support.
    """
    out = mesh.copy()
    rot = np.eye(4)
    rot[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]  # +90 deg about x: z' = y
    out.apply_transform(rot)
    lo, hi = out.bounds
    needs_support = bool(lo[2] < -0.05)
    shift = np.eye(4)
    shift[:3, 3] = [-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]]
    out.apply_transform(shift)
    return out, needs_support
