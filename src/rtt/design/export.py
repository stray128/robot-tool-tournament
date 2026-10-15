"""Mesh export of generated fingers: STL for printing, GLB for previews.

The visual mesh is the boolean union (manifold3d) of exactly the boxes the
simulator collides with, so what is shown is what is simulated. If manifold3d is
unavailable or fails, the boxes are concatenated instead (same shape, overlapping
shells, not a single watertight solid). For the union every box is grown by
10 micrometres so that segments whose faces exactly touch fuse into one solid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

from rtt.design.generate import box_arrays, is_parked
from rtt.design.schema import GripperTask
from rtt.design.types import FingerGeometry

logger = logging.getLogger(__name__)

FINGER_COLOR = (230, 126, 34, 255)
PALM_COLOR = (90, 98, 110, 255)
PALM_HEIGHT_M = 0.03
PALM_MARGIN_M = 0.01
WELD_M = 1e-5
"""Boxes are grown by this much (10 um) for the union so that touching faces fuse into one solid."""
SIMPLIFY_M = 1e-5
"""Edges shorter than this left over from the union are collapsed."""

_MIRROR_Y = np.diag([1.0, -1.0, 1.0, 1.0])
_Z_UP_TO_Y_UP = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])


def _active_boxes(geom: FingerGeometry) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(center, half size, rotation) of every non-parked box (parked boxes lie inside others)."""
    centers, halves, rots = box_arrays(geom)
    return [(centers[i], halves[i], rots[i]) for i, box in enumerate(geom.boxes) if not is_parked(box)]


def _union_manifold(geom: FingerGeometry) -> trimesh.Trimesh:
    from manifold3d import Manifold, OpType

    parts = []
    for center, half, rot in _active_boxes(geom):
        affine = np.hstack([rot, center[:, None]])
        size = tuple(float(v) for v in 2.0 * (half + WELD_M))
        parts.append(Manifold.cube(size, center=True).transform(affine))
    solid = Manifold.batch_boolean(parts, OpType.Add).simplify(SIMPLIFY_M)
    mesh = solid.to_mesh64() if hasattr(solid, "to_mesh64") else solid.to_mesh()
    vertices = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    # process=False: the union is already a clean manifold; merging near-duplicate
    # vertices would only risk creating non-manifold edges.
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _concatenated(geom: FingerGeometry) -> trimesh.Trimesh:
    meshes = []
    for center, half, rot in _active_boxes(geom):
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = center
        meshes.append(trimesh.creation.box(extents=2.0 * half, transform=transform))
    return trimesh.util.concatenate(meshes)


def to_trimesh(geom: FingerGeometry, mirror: bool = False) -> trimesh.Trimesh:
    """Mesh of one finger in its local frame (metres).

    ``mirror=True`` returns the right finger (reflected across the XZ plane), with
    face winding fixed so normals still point outward.
    """
    try:
        mesh = _union_manifold(geom)
    except Exception:  # manifold3d missing or failed: fall back to overlapping boxes
        logger.warning("manifold3d union failed; exporting overlapping boxes instead", exc_info=True)
        mesh = _concatenated(geom)
    if mirror:
        mesh.apply_transform(_MIRROR_Y)  # trimesh flips the winding for reflections
    return mesh


def export_stl(
    geom: FingerGeometry,
    path: str | Path,
    *,
    mirror: bool = False,
    units: Literal["mm", "m"] = "mm",
) -> Path:
    """Write one finger as a binary STL (millimetres by default, as slicers expect)."""
    if units not in ("mm", "m"):
        raise ValueError(f"units must be 'mm' or 'm', got {units!r}")
    mesh = to_trimesh(geom, mirror=mirror)
    if units == "mm":
        mesh.apply_scale(1000.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path, file_type="stl")
    return path


def gripper_scene(
    geom: FingerGeometry,
    task: GripperTask | None = None,
    *,
    opening: float = 1.0,
    with_palm: bool = True,
    y_up: bool = True,
) -> trimesh.Scene:
    """Both fingers mounted on a palm block, in metres.

    ``opening`` is the slide position as a fraction of the stroke (0 = full close,
    1 = full open). Positions follow the ``GripperTask`` mounting convention. With
    ``y_up=True`` the Z-up palm frame is rotated to glTF's Y-up convention.
    """
    task = task or GripperTask()
    if not 0.0 <= opening <= 1.0:
        raise ValueError(f"opening must be in [0, 1], got {opening}")
    root_y = (task.closed_gap_mm / 2.0 + opening * task.stroke_mm) / 1000.0
    finger = to_trimesh(geom)
    finger.visual.face_colors = FINGER_COLOR
    mirrored = to_trimesh(geom, mirror=True)
    mirrored.visual.face_colors = FINGER_COLOR

    base = _Z_UP_TO_Y_UP if y_up else np.eye(4)
    scene = trimesh.Scene()
    left = np.eye(4)
    left[1, 3] = root_y
    right = np.eye(4)
    right[1, 3] = -root_y
    scene.add_geometry(finger, node_name="finger_left", geom_name="finger_left", transform=base @ left)
    scene.add_geometry(mirrored, node_name="finger_right", geom_name="finger_right", transform=base @ right)
    if with_palm:
        lo, hi = finger.bounds
        span_y = 2.0 * (task.closed_gap_mm / 2.0 + task.stroke_mm) / 1000.0 + 2.0 * max(hi[1], 0.0)
        extents = (
            float(hi[0] - lo[0]) + 2.0 * PALM_MARGIN_M,
            span_y + 2.0 * PALM_MARGIN_M,
            PALM_HEIGHT_M,
        )
        palm_tf = np.eye(4)
        palm_tf[2, 3] = PALM_HEIGHT_M / 2.0
        palm = trimesh.creation.box(extents=extents)
        palm.visual.face_colors = PALM_COLOR
        scene.add_geometry(palm, node_name="palm", geom_name="palm", transform=base @ palm_tf)
    return scene


def export_glb(
    geom: FingerGeometry,
    path: str | Path,
    task: GripperTask | None = None,
    *,
    opening: float = 1.0,
    y_up: bool = True,
) -> Path:
    """Write a GLB preview of the full gripper (palm + finger pair), in metres."""
    scene = gripper_scene(geom, task, opening=opening, y_up=y_up)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))
    return path
