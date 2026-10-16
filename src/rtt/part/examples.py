"""Generic example parts for "Bring Your Part", modelled procedurally (CC0-1.0).

Four everyday shop-floor shapes, each a single watertight solid built with manifold3d
booleans and written as a binary STL in **millimetres** (the unit slicers and CAD
exports use), so the intake path sees exactly what a user upload looks like:

* ``l_bracket``: a 60 x 30 x 4 mm angle bracket, 50 mm tall, with a 3 mm inside
  fillet and two 5.5 mm mounting holes in the base.
* ``spur_gear``: a 20-tooth, module 2.5 gear blank (55 mm across the tips), 10 mm
  thick, with an 8 mm bore.
* ``rounded_tub``: an open-top 70 x 50 x 40 mm container with 8 mm corner radii,
  a 2.5 mm wall and a 3 mm floor.
* ``dumbbell_handle``: a 16 mm bar 60 mm long between two 40 mm discs 12 mm thick.

Every part fits the simulated gripper's 84 mm opening in at least its most likely
resting pose. Regenerate them with ``python -m rtt.part.examples [out_dir]`` (default
``assets/parts``); the output is byte-identical on every run.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold

__all__ = [
    "EXAMPLES_DIR",
    "EXAMPLE_PARTS",
    "dumbbell_handle",
    "example_path",
    "l_bracket",
    "rounded_tub",
    "spur_gear",
    "write_examples",
]

EXAMPLES_DIR: Final = Path(__file__).resolve().parents[3] / "assets" / "parts"
SEGMENTS: Final = 48
"""Facets around every round feature (enough for grasping, small enough for fast decomposition)."""


def _to_trimesh(solid: Manifold) -> trimesh.Trimesh:
    mesh = solid.to_mesh()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not out.is_watertight:  # manifold3d guarantees this; keep the check for regressions
        raise RuntimeError("example part is not watertight")
    return out


def l_bracket() -> trimesh.Trimesh:
    """Angle bracket: 60 (x) x 30 (y) x 4 mm base, 4 mm upright 50 mm tall, two holes."""
    base = Manifold.cube((60.0, 30.0, 4.0))
    upright = Manifold.cube((4.0, 30.0, 50.0))
    # Inside fillet: a 3 mm square minus a quarter cylinder, along y in the corner.
    block = Manifold.cube((3.0, 30.0, 3.0)).translate((4.0, 0.0, 4.0))
    round_ = (
        Manifold.cylinder(30.0, 3.0, circular_segments=SEGMENTS)
        .rotate((-90.0, 0.0, 0.0))
        .translate((7.0, 0.0, 7.0))
    )
    fillet = block - round_
    hole = Manifold.cylinder(10.0, 2.75, circular_segments=SEGMENTS).translate((0.0, 0.0, -3.0))
    holes = hole.translate((25.0, 15.0, 0.0)) + hole.translate((47.0, 15.0, 0.0))
    return _to_trimesh(base + upright + fillet - holes)


def spur_gear(
    teeth: int = 20, module: float = 2.5, face: float = 10.0, bore_d: float = 8.0
) -> trimesh.Trimesh:
    """Spur gear blank with trapezoidal teeth (addendum 1 m, dedendum 1.25 m) and a round bore."""
    r_pitch = module * teeth / 2.0
    r_tip, r_root = r_pitch + module, r_pitch - 1.25 * module
    pitch = 2.0 * math.pi / teeth
    tip_half, root_half = 0.18 * pitch, 0.30 * pitch
    outline = []
    for i in range(teeth):
        c = i * pitch
        for r, a in (
            (r_root, c - root_half),
            (r_tip, c - tip_half),
            (r_tip, c + tip_half),
            (r_root, c + root_half),
        ):
            outline.append((r * math.cos(a), r * math.sin(a)))
    body = CrossSection([outline]).extrude(face)
    bore = Manifold.cylinder(face + 2.0, bore_d / 2.0, circular_segments=SEGMENTS).translate((0.0, 0.0, -1.0))
    return _to_trimesh(body - bore)


def rounded_tub(
    length: float = 70.0, width: float = 50.0, height: float = 40.0, radius: float = 8.0, wall: float = 2.5
) -> trimesh.Trimesh:
    """Open-top container with rounded vertical corners and a 3 mm floor."""
    floor = 3.0

    def rounded_rect(lx: float, ly: float, r: float) -> CrossSection:
        core = CrossSection.square((lx - 2.0 * r, ly - 2.0 * r), center=True)
        return core.offset(r, JoinType.Round, circular_segments=SEGMENTS)

    outer = rounded_rect(length, width, radius).extrude(height)
    inner = rounded_rect(length - 2.0 * wall, width - 2.0 * wall, radius - wall).extrude(height)
    return _to_trimesh(outer - inner.translate((0.0, 0.0, floor)))


def dumbbell_handle(
    bar_d: float = 16.0, bar_len: float = 60.0, disc_d: float = 40.0, disc_t: float = 12.0
) -> trimesh.Trimesh:
    """Two discs joined by a round bar, axis along x."""

    def along_x(length: float, radius: float) -> Manifold:
        return Manifold.cylinder(length, radius, circular_segments=SEGMENTS).rotate((0.0, 90.0, 0.0))

    bar = along_x(bar_len + 2.0, bar_d / 2.0).translate((-(bar_len + 2.0) / 2.0, 0.0, 0.0))
    disc = along_x(disc_t, disc_d / 2.0)
    left = disc.translate((-bar_len / 2.0 - disc_t, 0.0, 0.0))
    right = disc.translate((bar_len / 2.0, 0.0, 0.0))
    return _to_trimesh(bar + left + right)


EXAMPLE_PARTS: Final[dict[str, Callable[[], trimesh.Trimesh]]] = {
    "l_bracket": l_bracket,
    "spur_gear": spur_gear,
    "rounded_tub": rounded_tub,
    "dumbbell_handle": dumbbell_handle,
}


def example_path(name: str, directory: str | Path | None = None) -> Path:
    """Path of an example part's STL (``assets/parts/<name>.stl`` by default)."""
    if name not in EXAMPLE_PARTS:
        raise KeyError(f"unknown example part {name!r}; choose from {sorted(EXAMPLE_PARTS)}")
    return Path(directory or EXAMPLES_DIR) / f"{name}.stl"


def write_examples(out_dir: str | Path | None = None) -> list[Path]:
    """Write every example part as a binary STL in millimetres; returns the paths."""
    out = Path(out_dir or EXAMPLES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in EXAMPLE_PARTS.items():
        path = out / f"{name}.stl"
        path.write_bytes(build().export(file_type="stl"))
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_examples(sys.argv[1] if len(sys.argv) > 1 else None):
        print(written)
