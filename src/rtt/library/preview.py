"""3D previews for the catalog page: small, deterministic GLB files.

A preview is drawn from an item's own geometry (its upstream visual meshes, or its print
files) and is a **derived file**: it carries the licence of what it was made from, and its
``SOURCE.json`` record says so. It is never a print file.

* :func:`cluster_decimate` reduces a mesh by vertex clustering on a uniform grid (no extra
  dependency, deterministic): vertices in one cell merge into their mean, and faces that
  collapse are dropped. The cell size grows until the mesh fits the face budget.
* :func:`scene_glb` writes meshes as one GLB with generic node names (``part_0``, ...), Y up
  (glTF's convention), metres, with no materials beyond a flat grey, so a preview never
  carries an upstream name or logo texture.
* :func:`model_meshes` collects the geometry of a compiled MuJoCo model's bodies (a subtree)
  in the model's reference pose, visual geoms first.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterable, Sequence

import mujoco
import numpy as np
import trimesh

__all__ = ["MAX_PREVIEW_BYTES", "cluster_decimate", "model_meshes", "scene_glb"]

#: Download budget of one preview, as served.
MAX_PREVIEW_BYTES = 400_000
_GREY = (0.62, 0.64, 0.66, 1.0)


def cluster_decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """``mesh`` with at most ``max_faces`` faces, by vertex clustering (module docstring)."""
    mesh = trimesh.Trimesh(
        np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64), process=False
    )
    if len(mesh.faces) <= max_faces:
        return mesh
    lo = mesh.vertices.min(axis=0)
    extent = float((mesh.vertices.max(axis=0) - lo).max()) or 1e-6
    cell = extent / 256.0
    for _ in range(40):
        idx = np.floor((mesh.vertices - lo) / cell).astype(np.int64)
        _, inverse = np.unique(idx, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        n = int(inverse.max()) + 1
        sums = np.zeros((n, 3))
        np.add.at(sums, inverse, mesh.vertices)
        counts = np.bincount(inverse, minlength=n).astype(np.float64)
        verts = sums / counts[:, None]
        faces = inverse[mesh.faces]
        keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
        faces = faces[keep]
        # drop duplicate faces (same vertex set) deterministically
        key = np.sort(faces, axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        faces = faces[np.sort(first)]
        if len(faces) <= max_faces:
            out = trimesh.Trimesh(verts, faces, process=False)
            out.remove_unreferenced_vertices()
            return out
        cell *= 1.25
    raise ValueError("could not decimate the mesh to the face budget")


def _round(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Vertices rounded to 0.01 mm, so float noise never changes the bytes of a preview."""
    return trimesh.Trimesh(np.round(np.asarray(mesh.vertices, np.float64), 5), mesh.faces, process=False)


def scene_glb(meshes: Sequence[trimesh.Trimesh], *, max_faces: int = 24_000) -> bytes:
    """One GLB of ``meshes`` (metres, Z up in, Y up out), decimated to ``max_faces`` in total."""
    meshes = [m for m in meshes if len(m.faces) > 0]
    if not meshes:
        raise ValueError("nothing to preview")
    total = sum(len(m.faces) for m in meshes)
    budget = [max(12, math.floor(max_faces * len(m.faces) / total)) for m in meshes]
    z_up_to_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    scene = trimesh.Scene()
    for i, (m, b) in enumerate(zip(meshes, budget, strict=True)):
        d = _round(cluster_decimate(m, b))
        d.apply_transform(z_up_to_y_up)
        d.visual = trimesh.visual.ColorVisuals(
            d, face_colors=np.tile(np.array(_GREY) * 255, (len(d.faces), 1))
        )
        scene.add_geometry(d, node_name=f"part_{i}", geom_name=f"part_{i}")
    buf = io.BytesIO()
    scene.export(buf, file_type="glb")
    return buf.getvalue()


def _primitive(gtype: int, size: np.ndarray) -> trimesh.Trimesh | None:
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size[:3])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * float(size[1]), radius=float(size[0]), count=[16, 8])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]), sections=24)
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        s = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        s.apply_scale(size[:3])
        return s
    return None


def model_meshes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bodies: Iterable[int] | None = None,
    *,
    prefer_visual: bool = True,
) -> list[trimesh.Trimesh]:
    """World-frame meshes of the geoms of ``bodies`` (all non-world bodies when ``None``).

    With ``prefer_visual``, a body that has visual geoms (no collision) is drawn from them and
    its collision geoms are skipped; otherwise from its collision geoms. Planes and height
    fields are skipped.
    """
    wanted = set(range(1, model.nbody)) if bodies is None else set(bodies)
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    by_body: dict[int, list[int]] = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in wanted and int(model.geom_type[g]) not in (
            mujoco.mjtGeom.mjGEOM_PLANE,
            mujoco.mjtGeom.mjGEOM_HFIELD,
        ):
            by_body.setdefault(b, []).append(g)
    out: list[trimesh.Trimesh] = []
    for b in sorted(by_body):
        geoms = by_body[b]
        visual = [g for g in geoms if not collides[g] and model.geom_rgba[g, 3] > 0.0]
        use = visual if (prefer_visual and visual) else [g for g in geoms if collides[g]] or geoms
        for g in use:
            gtype = int(model.geom_type[g])
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[g])
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                verts = model.mesh_vert[va : va + vn].astype(np.float64)
                faces = model.mesh_face[fa : fa + fn].astype(np.int64)
                mesh = trimesh.Trimesh(verts, faces, process=False)
            else:
                mesh = _primitive(gtype, model.geom_size[g])
                if mesh is None:
                    continue
            tf = np.eye(4)
            tf[:3, :3] = data.geom_xmat[g].reshape(3, 3)
            tf[:3, 3] = data.geom_xpos[g]
            mesh.apply_transform(tf)
            out.append(mesh)
    return out
