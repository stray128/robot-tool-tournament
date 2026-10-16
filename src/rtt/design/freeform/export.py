"""Meshes out of the v2 generator: mass properties, print STLs (mm) and GLB visuals (m).

* :func:`mass_properties` integrates a closed triangle mesh exactly (Eberly's polyhedral
  mass properties): mass, centre of mass and the inertia tensor about it.
* :func:`stl_bytes` writes a *canonical* binary STL in millimetres: fixed header, triangles
  rotated to start at their smallest vertex and sorted, so the bytes (and
  ``JawGeometry.print_mesh_sha``) depend only on the triangle set, not on the order a
  boolean library emitted it in.
* :func:`glb_bytes` writes an uncompressed glTF binary of one jaw in its body frame (Z up,
  metres) with a PBR material. The web pipeline compresses it (meshopt) afterwards; the
  render and replay code assigns asset ids from the bytes.
* :func:`export_pair` writes ``left.stl``, ``right.stl``, ``left.glb``, ``right.glb`` and
  ``recipe.json`` for one compiled (design, part) pair.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STL_HEADER = b"rtt freeform v2 jaw, binary STL, millimetres".ljust(80, b" ")
JAW_COLOR = (0.902, 0.494, 0.133, 1.0)  # sRGB, the site's designed-finger orange


@dataclass(frozen=True, eq=False)
class JawMesh:
    """A closed triangle mesh of one printed jaw in its body frame (metres)."""

    vertices: np.ndarray  # (n, 3) float64
    faces: np.ndarray  # (m, 3) int64, counter-clockwise seen from outside

    def mirrored_y(self) -> JawMesh:
        v = self.vertices * np.array([1.0, -1.0, 1.0])
        return JawMesh(v, self.faces[:, ::-1].copy())

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])


def _subexpr(w0: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> tuple[np.ndarray, ...]:
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0 * w0
    temp2 = temp1 + w1 * temp0
    f2 = temp2 + w2 * f1
    f3 = w0 * temp1 + w1 * temp2 + w2 * f2
    g0 = f2 + w0 * (f1 + w0)
    g1 = f2 + w1 * (f1 + w1)
    g2 = f2 + w2 * (f1 + w2)
    return f1, f2, f3, g0, g1, g2


def mass_properties(
    vertices: np.ndarray, faces: np.ndarray, density: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, centre of mass, 3x3 inertia tensor about the centre of mass) of a closed mesh."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    d = np.cross(p1 - p0, p2 - p0)
    x0, y0, z0 = p0.T
    x1, y1, z1 = p1.T
    x2, y2, z2 = p2.T
    f1x, f2x, f3x, g0x, g1x, g2x = _subexpr(x0, x1, x2)
    _, f2y, f3y, g0y, g1y, g2y = _subexpr(y0, y1, y2)
    _, f2z, f3z, g0z, g1z, g2z = _subexpr(z0, z1, z2)
    integ = np.array(
        [
            np.sum(d[:, 0] * f1x),
            np.sum(d[:, 0] * f2x),
            np.sum(d[:, 1] * f2y),
            np.sum(d[:, 2] * f2z),
            np.sum(d[:, 0] * f3x),
            np.sum(d[:, 1] * f3y),
            np.sum(d[:, 2] * f3z),
            np.sum(d[:, 0] * (y0 * g0x + y1 * g1x + y2 * g2x)),
            np.sum(d[:, 1] * (z0 * g0y + z1 * g1y + z2 * g2y)),
            np.sum(d[:, 2] * (x0 * g0z + x1 * g1z + x2 * g2z)),
        ]
    )
    integ *= np.array([1 / 6, 1 / 24, 1 / 24, 1 / 24, 1 / 60, 1 / 60, 1 / 60, 1 / 120, 1 / 120, 1 / 120])
    volume = integ[0]
    if volume <= 0:
        raise ValueError(f"mesh volume is not positive ({volume:g}); faces may be inverted")
    com = integ[1:4] / volume
    cx, cy, cz = com
    ixx = integ[5] + integ[6] - volume * (cy * cy + cz * cz)
    iyy = integ[4] + integ[6] - volume * (cz * cz + cx * cx)
    izz = integ[4] + integ[5] - volume * (cx * cx + cy * cy)
    ixy = -(integ[7] - volume * cx * cy)
    iyz = -(integ[8] - volume * cy * cz)
    ixz = -(integ[9] - volume * cz * cx)
    inertia = density * np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return float(volume * density), com, inertia


def inertia6(tensor: np.ndarray) -> np.ndarray:
    """MuJoCo ``fullinertia`` order: Ixx, Iyy, Izz, Ixy, Ixz, Iyz."""
    t = np.asarray(tensor)
    return np.array([t[0, 0], t[1, 1], t[2, 2], t[0, 1], t[0, 2], t[1, 2]])


def weld_print_mesh(vertices_m: np.ndarray, faces: np.ndarray) -> JawMesh:
    """The mesh exactly as the print STL stores it: float32 millimetres, welded.

    Boolean libraries leave sliver vertices nanometres apart; written as float32 some of them
    coincide, which breaks the STL's watertightness and makes the triangle order ambiguous.
    Vertices are rounded to float32 millimetres, coincident ones merged, and triangles that
    collapse are dropped. The returned vertices (metres, float64) round-trip to the same
    float32 millimetres, so :func:`stl_bytes` of the result is canonical and every check
    (watertightness, mass) runs on what is printed.
    """
    v32 = (np.asarray(vertices_m, dtype=np.float64) * 1000.0).astype(np.float32)
    uniq, inv = np.unique(v32, axis=0, return_inverse=True)
    f = inv.reshape(-1)[np.asarray(faces, dtype=np.int64)]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    used = np.unique(f)
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return JawMesh(uniq[used].astype(np.float64) / 1000.0, remap[f])


def edge_report(faces: np.ndarray) -> tuple[bool, int]:
    """(watertight, number of connected components) from the triangle connectivity alone.

    Watertight means every directed edge appears exactly once and its reverse exactly once.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces, dtype=np.int64)
    n = int(f.max()) + 1
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    key = e[:, 0] * n + e[:, 1]
    rkey = e[:, 1] * n + e[:, 0]
    uniq, counts = np.unique(key, return_counts=True)
    watertight = bool(np.all(counts == 1)) and bool(np.isin(rkey, uniq).all())
    used = np.unique(f)
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return watertight, len(np.unique(labels[used]))


def canonical_triangles(vertices_mm: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(m, 3, 3)`` float32 triangles, each rotated to start at its smallest vertex, then sorted."""
    tri = np.asarray(vertices_mm, dtype=np.float32)[np.asarray(faces)]
    first = np.zeros(len(tri), dtype=np.int64)
    for k in (1, 2):
        a, b = tri[np.arange(len(tri)), first], tri[:, k]
        less = (b[:, 0] < a[:, 0]) | (
            (b[:, 0] == a[:, 0]) & ((b[:, 1] < a[:, 1]) | ((b[:, 1] == a[:, 1]) & (b[:, 2] < a[:, 2])))
        )
        first = np.where(less, k, first)
    idx = (first[:, None] + np.arange(3)[None, :]) % 3
    rot = tri[np.arange(len(tri))[:, None], idx]
    flat = rot.reshape(len(rot), 9)
    order = np.lexsort(flat.T[::-1])
    return rot[order]


def stl_bytes(vertices_m: np.ndarray, faces: np.ndarray) -> bytes:
    """Canonical binary STL of the mesh, in millimetres."""
    tri = canonical_triangles(np.asarray(vertices_m, dtype=np.float64) * 1000.0, faces)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float64)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(norm > 0, n / np.where(norm > 0, norm, 1.0), 0.0).astype(np.float32)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    return STL_HEADER + struct.pack("<I", len(tri)) + rec.tobytes()


def stl_sha256(vertices_m: np.ndarray, faces: np.ndarray) -> str:
    return hashlib.sha256(stl_bytes(vertices_m, faces)).hexdigest()


def glb_bytes(
    mesh: JawMesh, color: tuple[float, float, float, float] = JAW_COLOR, name: str = "jaw"
) -> bytes:
    """Uncompressed GLB of one jaw (body frame, Z up, metres), flat-shaded PBR material."""
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    vb = v.tobytes()
    ib = f.tobytes()
    pad = lambda b: b + b"\x00" * ((4 - len(b) % 4) % 4)
    vb_p, ib_p = pad(vb), pad(ib)
    gltf = {
        "asset": {"version": "2.0", "generator": "rtt.design.freeform.export"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {"name": name, "primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}
        ],
        "materials": [
            {
                "name": "pla",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(color),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.55,
                },
            }
        ],
        "buffers": [{"byteLength": len(vb_p) + len(ib_p)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vb), "target": 34962},
            {"buffer": 0, "byteOffset": len(vb_p), "byteLength": len(ib), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(v),
                "type": "VEC3",
                "min": [float(x) for x in v.min(axis=0)],
                "max": [float(x) for x in v.max(axis=0)],
            },
            {"bufferView": 1, "componentType": 5125, "count": int(f.size), "type": "SCALAR"},
        ],
        "extras": {"up_axis": "Z", "units": "m"},
    }
    js = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    binary = vb_p + ib_p
    total = 12 + 8 + len(js) + 8 + len(binary)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(js), b"JSON") + js
    out += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    return out


def export_pair(build: Any, out_dir: str | Path) -> dict[str, Path]:
    """Write the print STLs (mm), the GLB visuals (m) and the recipe of one compiled pair."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for side, mesh in (("left", build.left_mesh), ("right", build.right_mesh)):
        stl = out / f"{side}.stl"
        stl.write_bytes(stl_bytes(mesh.vertices, mesh.faces))
        glb = out / f"{side}.glb"
        glb.write_bytes(glb_bytes(mesh, name=f"{build.pair.part_id}_{side}"))
        paths[f"{side}_stl"] = stl
        paths[f"{side}_glb"] = glb
    recipe = out / "recipe.json"
    recipe.write_text(json.dumps(build.recipe_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["recipe"] = recipe
    return paths
