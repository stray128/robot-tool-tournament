"""Static preview images for the certificate: the part at rest and the printed jaw pair.

Rendered with matplotlib's Agg canvas (no display, no GL, no global pyplot state):
flat-shaded triangles sorted back to front, on a transparent background so the
images sit on the certificate's light or dark surface. Returns PNG bytes.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import trimesh

from rtt.sim.mesh import stable_poses

__all__ = ["JAW_COLOR", "PART_COLOR", "ROD_COLOR", "jaw_pair_scene", "posed_part", "render_png"]

PART_COLOR: Final = (0.36, 0.55, 0.78)
JAW_COLOR: Final = (0.90, 0.49, 0.13)
ROD_COLOR: Final = (0.62, 0.64, 0.68)
_LIGHT: Final = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])


def posed_part(mesh_mm: trimesh.Trimesh, pose_index: int = 0) -> trimesh.Trimesh:
    """The part resting on ``z = 0`` in stable pose ``pose_index``, its narrowest side along y."""
    poses = stable_poses(mesh_mm)
    transform = poses[min(pose_index, len(poses) - 1)][0]
    out = mesh_mm.copy()
    out.apply_transform(transform)
    pts = np.asarray(out.convex_hull.vertices)[:, :2]
    angles = np.radians(np.arange(0.0, 180.0, 0.5))
    widths = [np.ptp(pts @ np.array([math.sin(a), math.cos(a)])) for a in angles]
    a = float(angles[int(np.argmin(widths))])
    turn = np.eye(4)
    turn[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    out.apply_transform(turn)
    lo, hi = out.bounds
    shift = np.eye(4)
    shift[:3, 3] = [-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]]
    out.apply_transform(shift)
    return out


def jaw_pair_scene(
    jaw_mm: trimesh.Trimesh,
    stroke_mm: float,
    opening: float = 1.0,
    rod_spacing_mm: float = 32.0,
    rod_axis_height_mm: float = 9.0,
    rod_d_mm: float = 6.0,
) -> list[tuple[trimesh.Trimesh, tuple[float, float, float]]]:
    """Both jaws on their rods (assembly frame, mm): the right jaw is the left turned 180 deg about Z."""
    root = opening * stroke_mm
    left = jaw_mm.copy()
    left.apply_translation((0.0, root, 0.0))
    right = jaw_mm.copy()
    right.apply_transform(trimesh.transformations.rotation_matrix(math.pi, (0.0, 0.0, 1.0)))
    right.apply_translation((0.0, -root, 0.0))
    span = 2.0 * (root + float(jaw_mm.bounds[1][1])) + 10.0
    items: list[tuple[trimesh.Trimesh, tuple[float, float, float]]] = []
    for sx in (-rod_spacing_mm / 2.0, rod_spacing_mm / 2.0):
        rod = trimesh.creation.cylinder(radius=rod_d_mm / 2.0, height=span, sections=24)
        rod.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, (1.0, 0.0, 0.0)))
        rod.apply_translation((sx, 0.0, rod_axis_height_mm))
        items.append((rod, ROD_COLOR))
    items.extend([(left, JAW_COLOR), (right, JAW_COLOR)])
    return items


def render_png(
    items: Sequence[tuple[trimesh.Trimesh, tuple[float, float, float]]],
    *,
    layers: Sequence[int] | None = None,
    elev: float = 24.0,
    azim: float = -58.0,
    size_px: tuple[int, int] = (720, 540),
    dpi: int = 120,
    pad_px: int = 12,
) -> bytes:
    """Flat-shaded render of coloured meshes as PNG bytes, cropped to the drawing.

    Items with the same ``layers`` value share one depth-sorted collection; layers are
    drawn in increasing order (a later layer always covers an earlier one), which
    avoids matplotlib's per-collection sorting artefacts for parts inside parts.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    layer_of = list(layers) if layers is not None else [0] * len(items)
    if len(layer_of) != len(items):
        raise ValueError("layers must give one value per item")
    groups: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for (mesh, rgb), layer in zip(items, layer_of, strict=True):
        v = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        shade = 0.38 + 0.62 * np.clip(np.asarray(mesh.face_normals) @ _LIGHT, 0.0, 1.0)
        tris, colors = groups.setdefault(layer, ([], []))
        tris.append(v)
        colors.append(np.column_stack([np.outer(shade, rgb), np.ones(len(shade))]))

    fig = Figure(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d", computed_zorder=False)
    ax.set_facecolor((0, 0, 0, 0))
    all_pts = []
    for z, layer in enumerate(sorted(groups)):
        tris, colors = groups[layer]
        stacked = np.concatenate(tris)
        all_pts.append(stacked.reshape(-1, 3))
        coll = Poly3DCollection(
            stacked, facecolors=np.clip(np.concatenate(colors), 0.0, 1.0), edgecolors="none", linewidths=0.0
        )
        coll.set_zorder(z)
        ax.add_collection3d(coll)
    pts = np.concatenate(all_pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre, half = (lo + hi) / 2.0, max(float((hi - lo).max()) / 2.0, 1e-6)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=dpi)
    image = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
    box = image.getchannel("A").getbbox()
    if box is not None:
        x0, y0, x1, y1 = box
        image = image.crop(
            (
                max(0, x0 - pad_px),
                max(0, y0 - pad_px),
                min(image.width, x1 + pad_px),
                min(image.height, y1 + pad_px),
            )
        )
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
