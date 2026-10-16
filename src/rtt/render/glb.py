"""Deterministic GLB files and the content-addressed asset store behind replay v2.

**Writer.** :func:`glb_bytes` writes glTF 2.0 binaries with no third-party dependency. The
same meshes always give the same bytes (sorted compact JSON, quantised attributes, no
timestamps), so an asset's id, the first 16 hex digits of its sha256, is stable across runs
and machines. Each mesh gets one node and one primitive with a metallic-roughness material
(:class:`rtt.render.materials.Look`, linear base colour) and, optionally, an embedded PNG or
JPEG base-colour texture. With ``quantize=True`` (default) positions are 16-bit and normals
8-bit (``KHR_mesh_quantization``, which three.js, Blender and gltf-transform read); the node
carries the dequantisation as a uniform scale plus a translation, so the mesh still sits in
its body frame: Z up, metres. The credit line goes into ``asset.copyright`` and the licence,
role and changes into ``asset.extras``, so a GLB that travels alone still says what it is.

**Reader.** :func:`read_glb` reads these files back (node transforms applied) for the
offline renderers. It is not a general glTF reader.

**Store.** :class:`AssetStore` writes ``<root>/<id>.glb`` and keeps the manifest entries
(``docs/ASSET_MANIFEST.md``; one schema, ``#/$defs/asset`` in
``src/rtt/sim/replay_v2.schema.json``). A compressor (for example the web build's meshopt
step) can be plugged in; the served file is then the compressed one and the uncompressed GLB
is kept under ``raw_root`` with the same id for offline rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from rtt.render.materials import Look, linear_to_srgb

__all__ = [
    "GENERATOR",
    "GRIPPER_ATTRIBUTION",
    "GRIPPER_CHANGES",
    "GSO_ATTRIBUTION",
    "MANIFEST_FORMAT",
    "OURS_ATTRIBUTION",
    "OURS_LICENSE",
    "ROLES",
    "AssetStore",
    "GlbDocument",
    "GlbMesh",
    "asset_id",
    "check_entry",
    "glb_bytes",
    "mesh_from_geoms",
    "mesh_from_trimesh",
    "read_glb",
]

GENERATOR: Final = "rtt.render.glb"
MANIFEST_FORMAT: Final = "rtt-asset-manifest"
MANIFEST_VERSION: Final = 1
ROLES: Final = ("finger_left", "finger_right", "gripper_frame", "carriage", "object", "prop")
OURS_LICENSE: Final = "Apache-2.0"
OURS_ATTRIBUTION: Final = "Robot Tool Tournament (Apache-2.0)"
# Third-party credit lines as carried by assets and replays: the short on-screen form, which
# points at the full credit (docs/ATTRIBUTION.md and the site's credits page).
GRIPPER_ATTRIBUTION: Final = (
    "An open-source parallel gripper, CC BY 4.0, modified (full credit: /credits#gripper)"
)
GRIPPER_CHANGES: Final = (
    "cut at the jaw root plane",
    "re-posed into the palm and jaw frames",
    "re-encoded as GLB",
)
GSO_ATTRIBUTION: Final = "Google Scanned Objects, CC BY 4.0, modified (full credit: /credits#gso)"

_MAGIC, _JSON, _BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
_FLOAT, _SHORT, _BYTE, _USHORT, _UINT = 5126, 5122, 5120, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_QUANT = "KHR_mesh_quantization"
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_PRIVATE = re.compile(r"(^[/~\\])|(^[A-Za-z]:[\\/])|(/home/)|(/Users/)|(/tmp/)|(\\Users\\)|(\.\.)")


def asset_id(data: bytes) -> str:
    """An asset's id: the first 16 hex characters of the file's sha256."""
    return hashlib.sha256(data).hexdigest()[:16]


# --------------------------------------------------------------------------- meshes


@dataclass(frozen=True, eq=False)
class GlbMesh:
    """One triangle mesh with its look, in the frame of the body it is attached to."""

    vertices: np.ndarray  # (n, 3) metres
    faces: np.ndarray  # (m, 3) vertex indices, counter-clockwise seen from outside
    look: Look
    normals: np.ndarray | None = None  # (n, 3) unit vertex normals
    uv: np.ndarray | None = None  # (n, 2) glTF texture coordinates (origin top-left)
    texture: bytes | None = None  # PNG or JPEG bytes
    name: str = "mesh"

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.all(np.isfinite(v)):
            raise ValueError(f"{self.name}: vertices must be a finite (n >= 3, 3) array")
        if f.ndim != 2 or f.shape[1] != 3 or len(f) < 1 or f.min() < 0 or f.max() >= len(v):
            raise ValueError(f"{self.name}: faces must be (m, 3) indices into the vertices")
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)
        for name, width in (("normals", 3), ("uv", 2)):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != (len(v), width) or not np.all(np.isfinite(arr)):
                    raise ValueError(f"{self.name}: {name} must be a finite ({len(v)}, {width}) array")
                object.__setattr__(self, name, arr)
        if self.texture is not None and self.uv is None:
            raise ValueError(f"{self.name}: a texture needs uv coordinates")
        if self.texture is not None and _mime(self.texture) is None:
            raise ValueError(f"{self.name}: texture must be PNG or JPEG bytes")


def mesh_from_trimesh(mesh: Any, look: Look, *, crease_deg: float = 28.0, name: str = "mesh") -> GlbMesh:
    """A :class:`GlbMesh` from a trimesh mesh: vertices split at creases sharper than ``crease_deg``,
    smooth normals elsewhere (what CAD needs to read as machined rather than faceted)."""
    import trimesh

    m = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    m.merge_vertices()
    shaded = trimesh.graph.smooth_shade(m, angle=np.radians(crease_deg), facet_minarea=None)
    return GlbMesh(
        vertices=np.asarray(shaded.vertices),
        faces=np.asarray(shaded.faces),
        normals=np.asarray(shaded.vertex_normals),
        look=look,
        name=name,
    )


WELD_M: Final = 1e-5  # solids are grown by 10 um before the union so touching faces fuse


def mesh_from_geoms(geoms: Sequence[Mapping[str, Any]], segments: int = 48) -> Any:
    """One watertight trimesh: the union of v1 replay geoms (box, cylinder, capsule, convex
    mesh) in their body frame. Planes are skipped. Used to give v1 bodies (box fingers,
    primitive objects) a GLB asset, so a v1 replay can be drawn in the v2 look."""
    import trimesh
    from manifold3d import Manifold, OpType

    parts = []
    for g in geoms:
        kind = g["type"]
        if kind == "box":
            half = np.asarray(g["half_size"], dtype=np.float64)
            if np.any(half <= 0):
                continue
            solid = Manifold.cube(tuple(float(v) for v in 2.0 * (half + WELD_M)), center=True)
        elif kind == "cylinder":
            r, hh = float(g["radius"]) + WELD_M, float(g["half_height"]) + WELD_M
            solid = Manifold.cylinder(2.0 * hh, r, r, circular_segments=segments, center=True)
        elif kind == "capsule":
            r, hl = float(g["radius"]) + WELD_M, float(g["half_length"])
            ball = Manifold.sphere(r, circular_segments=segments)
            solid = Manifold.batch_boolean(
                [
                    Manifold.cylinder(2.0 * hl, r, r, circular_segments=segments, center=True),
                    ball.translate((0.0, 0.0, hl)),
                    ball.translate((0.0, 0.0, -hl)),
                ],
                OpType.Add,
            )
        elif kind == "mesh":
            solid = Manifold.hull_points(np.asarray(g["vertices"], dtype=np.float64).reshape(-1, 3))
        elif kind == "plane":
            continue
        else:
            raise ValueError(f"unknown geom type {kind!r}")
        x, y, z, w = (float(c) for c in g["quat_xyzw"])
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        parts.append(solid.transform(np.hstack([rot, np.asarray(g["pos"], dtype=np.float64)[:, None]])))
    if not parts:
        raise ValueError("no solid geoms to mesh")
    union = Manifold.batch_boolean(parts, OpType.Add).simplify(WELD_M)
    mesh = union.to_mesh64() if hasattr(union, "to_mesh64") else union.to_mesh()
    return trimesh.Trimesh(
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )


def _mime(data: bytes) -> str | None:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return None


def _r(x: float, nd: int = 9) -> float:
    v = round(float(x), nd)
    return 0.0 if v == 0 else v  # no "-0.0" in the JSON


class _Bin:
    """The binary chunk under construction: 4-byte aligned buffer views."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []

    def add(self, payload: bytes, target: int | None, stride: int | None = None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        if stride is not None:
            view["byteStride"] = stride
        self.data += payload
        self.views.append(view)
        return len(self.views) - 1


def glb_bytes(
    meshes: Sequence[GlbMesh],
    *,
    copyright: str,
    extras: Mapping[str, Any] | None = None,
    quantize: bool = True,
) -> bytes:
    """Encode ``meshes`` as one GLB (see the module docstring); deterministic."""
    if not meshes:
        raise ValueError("a GLB needs at least one mesh")
    if not copyright:
        raise ValueError("every GLB carries its credit line in asset.copyright")
    buf = _Bin()
    accessors: list[dict[str, Any]] = []
    gl_meshes, nodes, materials = [], [], []
    textures: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []

    def accessor(view: int, ctype: int, count: int, kind: str, **kw: Any) -> int:
        accessors.append({"bufferView": view, "componentType": ctype, "count": count, "type": kind, **kw})
        return len(accessors) - 1

    for mesh in meshes:
        v, n = mesh.vertices, len(mesh.vertices)
        node: dict[str, Any] = {"mesh": len(gl_meshes), "name": mesh.name}
        attrs: dict[str, int] = {}
        if quantize:
            lo, hi = v.min(axis=0), v.max(axis=0)
            centre = np.round((lo + hi) / 2.0, 9)
            half = max(float(np.max(hi - lo)) / 2.0, 1e-6) * (1.0 + 1e-6)  # uniform: normals unaffected
            half = float(np.float32(half))
            q = np.clip(np.round((v - centre) / half * 32767.0), -32767, 32767).astype("<i2")
            padded = np.zeros((n, 4), dtype="<i2")
            padded[:, :3] = q
            view = buf.add(padded.tobytes(), _ARRAY_BUFFER, stride=8)
            attrs["POSITION"] = accessor(
                view,
                _SHORT,
                n,
                "VEC3",
                normalized=True,
                min=q.min(axis=0).tolist(),
                max=q.max(axis=0).tolist(),
            )
            node["translation"] = [_r(c) for c in centre]
            node["scale"] = [_r(half, 12)] * 3
        else:
            p = v.astype("<f4")
            view = buf.add(p.tobytes(), _ARRAY_BUFFER)
            attrs["POSITION"] = accessor(
                view,
                _FLOAT,
                n,
                "VEC3",
                min=[float(x) for x in p.min(axis=0)],
                max=[float(x) for x in p.max(axis=0)],
            )
        if mesh.normals is not None:
            nrm = mesh.normals / np.maximum(np.linalg.norm(mesh.normals, axis=1, keepdims=True), 1e-12)
            if quantize:
                qn = np.zeros((n, 4), dtype="<i1")
                qn[:, :3] = np.clip(np.round(nrm * 127.0), -127, 127)
                attrs["NORMAL"] = accessor(
                    buf.add(qn.tobytes(), _ARRAY_BUFFER, stride=4), _BYTE, n, "VEC3", normalized=True
                )
            else:
                attrs["NORMAL"] = accessor(
                    buf.add(nrm.astype("<f4").tobytes(), _ARRAY_BUFFER), _FLOAT, n, "VEC3"
                )
        if mesh.uv is not None:
            attrs["TEXCOORD_0"] = accessor(
                buf.add(mesh.uv.astype("<f4").tobytes(), _ARRAY_BUFFER), _FLOAT, n, "VEC2"
            )
        itype, idt = (_USHORT, "<u2") if n < 65536 else (_UINT, "<u4")
        idx = mesh.faces.astype(idt).ravel()
        ind = accessor(buf.add(idx.tobytes(), _ELEMENT_ARRAY_BUFFER), itype, len(idx), "SCALAR")

        pbr = mesh.look.gltf()
        mat: dict[str, Any] = {"name": mesh.look.name, "pbrMetallicRoughness": pbr}
        if mesh.look.alpha < 1.0:
            mat["alphaMode"] = "BLEND"
        if mesh.texture is not None:
            images.append({"bufferView": buf.add(mesh.texture, None), "mimeType": _mime(mesh.texture)})
            textures.append({"sampler": 0, "source": len(images) - 1})
            pbr["baseColorTexture"] = {"index": len(textures) - 1}
        materials.append(mat)
        gl_meshes.append(
            {
                "name": mesh.name,
                "primitives": [
                    {"attributes": attrs, "indices": ind, "material": len(materials) - 1, "mode": 4}
                ],
            }
        )
        nodes.append(node)

    while len(buf.data) % 4:
        buf.data.append(0)
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": GENERATOR, "copyright": copyright},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
    }
    if extras:
        doc["asset"]["extras"] = json.loads(json.dumps(dict(extras), sort_keys=True))
    if textures:
        doc["textures"], doc["images"] = textures, images
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    if quantize:
        doc["extensionsUsed"] = doc["extensionsRequired"] = [_QUANT]
    js = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(buf.data)
    out = bytearray(struct.pack("<III", _MAGIC, 2, total))
    out += struct.pack("<II", len(js), _JSON) + js
    out += struct.pack("<II", len(buf.data), _BIN) + bytes(buf.data)
    return bytes(out)


# --------------------------------------------------------------------------- reader


@dataclass(frozen=True, eq=False)
class GlbDocument:
    meshes: list[GlbMesh]
    asset: dict[str, Any]

    @property
    def copyright(self) -> str:
        return str(self.asset.get("copyright", ""))

    @property
    def extras(self) -> dict[str, Any]:
        return dict(self.asset.get("extras") or {})


_DTYPES: Final = {_FLOAT: "<f4", _SHORT: "<i2", _BYTE: "<i1", _USHORT: "<u2", _UINT: "<u4", 5121: "<u1"}
_WIDTH: Final = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_NORM: Final = {_SHORT: 32767.0, _BYTE: 127.0, _USHORT: 65535.0, 5121: 255.0}


def _quat_mat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_glb(data: bytes) -> GlbDocument:
    """Read a GLB written by :func:`glb_bytes` (node TRS applied; one level of nodes)."""
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC or version != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    if jtype != _JSON:
        raise ValueError("GLB: first chunk is not JSON")
    doc = json.loads(data[20 : 20 + jlen])
    off = 20 + jlen
    blen, btype = struct.unpack_from("<II", data, off)
    if btype != _BIN:
        raise ValueError("GLB: second chunk is not BIN")
    bin_ = data[off + 8 : off + 8 + blen]
    unknown = set(doc.get("extensionsRequired", [])) - {_QUANT}
    if unknown:
        raise ValueError(
            f"GLB needs unsupported extensions {sorted(unknown)} (read the raw, uncompressed file)"
        )

    def view_bytes(i: int) -> bytes:
        v = doc["bufferViews"][i]
        return bin_[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]]

    def read(i: int) -> np.ndarray:
        acc = doc["accessors"][i]
        view = doc["bufferViews"][acc["bufferView"]]
        dt = np.dtype(_DTYPES[acc["componentType"]])
        width = _WIDTH[acc["type"]]
        stride = view.get("byteStride", dt.itemsize * width)
        raw = np.frombuffer(view_bytes(acc["bufferView"]), dtype=np.uint8)
        start = acc.get("byteOffset", 0)
        rows = np.lib.stride_tricks.as_strided(
            raw[start:], shape=(acc["count"], width * dt.itemsize), strides=(stride, 1)
        )
        arr = np.ascontiguousarray(rows).view(dt).reshape(acc["count"], width).astype(np.float64)
        if acc.get("normalized"):
            arr = np.maximum(arr / _NORM[acc["componentType"]], -1.0)
        return arr

    meshes = []
    for node in doc.get("nodes", []):
        if "mesh" not in node:
            continue
        t = np.asarray(node.get("translation", [0.0, 0.0, 0.0]))
        s = np.asarray(node.get("scale", [1.0, 1.0, 1.0]))
        r = _quat_mat(node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            a = prim["attributes"]
            verts = (r @ (read(a["POSITION"]) * s).T).T + t
            normals = None
            if "NORMAL" in a:
                normals = (r @ (read(a["NORMAL"]) / s).T).T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            uv = read(a["TEXCOORD_0"]) if "TEXCOORD_0" in a else None
            faces = read(prim["indices"]).astype(np.int64).reshape(-1, 3)
            mat = doc["materials"][prim["material"]] if "material" in prim else {}
            pbr = mat.get("pbrMetallicRoughness", {})
            rgba = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            texture = None
            if "baseColorTexture" in pbr:
                img = doc["images"][doc["textures"][pbr["baseColorTexture"]["index"]]["source"]]
                texture = view_bytes(img["bufferView"])
            look = Look(
                mat.get("name", "material"),
                tuple(min(1.0, max(0.0, linear_to_srgb(c))) for c in rgba[:3]),  # type: ignore[arg-type]
                alpha=float(rgba[3]),
                metallic=float(pbr.get("metallicFactor", 1.0)),
                roughness=float(pbr.get("roughnessFactor", 1.0)),
            )
            meshes.append(
                GlbMesh(
                    verts, faces, look, normals=normals, uv=uv, texture=texture, name=node.get("name", "mesh")
                )
            )
    return GlbDocument(meshes=meshes, asset=dict(doc.get("asset", {})))


# --------------------------------------------------------------------------- asset store


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def check_entry(aid: str, entry: Mapping[str, Any]) -> list[str]:
    """Manifest rules (ASSET_MANIFEST.md) for one entry; returns the problems found."""
    errs = []
    if not re.fullmatch(r"[0-9a-f]{16}", aid):
        errs.append(f"{aid}: an asset id is 16 lowercase hex digits")
    if entry.get("uri") != f"assets/{aid}.glb":
        errs.append(f"{aid}: uri must be assets/{aid}.glb")
    if not str(entry.get("sha256", "")).startswith(aid):
        errs.append(f"{aid}: the id must be the first 16 hex digits of sha256")
    if entry.get("role") not in ROLES:
        errs.append(f"{aid}: role must be one of {ROLES}")
    if not _SPDX.match(str(entry.get("license", ""))):
        errs.append(f"{aid}: license must be an SPDX identifier")
    if not entry.get("attribution"):
        errs.append(f"{aid}: attribution is required")
    changes = entry.get("changes")
    if not isinstance(changes, list) or not all(isinstance(c, str) and c for c in changes):
        errs.append(f"{aid}: changes must be a list of non-empty strings")
    elif entry.get("license") != OURS_LICENSE and not changes:
        errs.append(f"{aid}: a third-party asset must list its changes")
    for key in ("label",):
        if key in entry and _PRIVATE.search(str(entry[key])):
            errs.append(f"{aid}: {key} looks like a local path")
    src = entry.get("source") or {}
    if "path" in src and _PRIVATE.search(str(src["path"])):
        errs.append(f"{aid}: source.path must be the upstream repository path, not a local one")
    return errs


@dataclass
class AssetStore:
    """Content-addressed GLB files plus their manifest entries.

    ``root`` is the served directory (``web/public/assets`` for the site). Entries already in
    ``root/manifest.json`` are loaded, so a store can be reopened and extended.
    """

    root: Path
    compressor: Callable[[bytes], bytes] | None = None
    raw_root: Path | None = None
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.raw_root is not None:
            self.raw_root = Path(self.raw_root)
        if self.compressor is not None and self.raw_root is None:
            raise ValueError("a compressing store needs raw_root for the uncompressed files")
        manifest = self.root / "manifest.json"
        if manifest.is_file() and not self.entries:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            if doc.get("format") != MANIFEST_FORMAT:
                raise ValueError(f"{manifest}: not an asset manifest")
            self.entries = dict(doc.get("assets", {}))

    def add(
        self,
        glb: bytes,
        *,
        role: str,
        license: str = OURS_LICENSE,
        attribution: str = OURS_ATTRIBUTION,
        changes: Iterable[str] = (),
        label: str | None = None,
        source: Mapping[str, Any] | None = None,
        brand_visible: bool | None = None,
    ) -> str:
        """Store one GLB and return its id; the same bytes and metadata are stored once."""
        served = self.compressor(glb) if self.compressor is not None else glb
        sha = hashlib.sha256(served).hexdigest()
        aid = sha[:16]
        entry: dict[str, Any] = {
            "uri": f"assets/{aid}.glb",
            "sha256": sha,
            "bytes": len(served),
            "role": role,
            "license": license,
            "attribution": attribution,
            "changes": list(changes),
            "media_type": "model/gltf-binary",
        }
        if label is not None:
            entry["label"] = label
        if brand_visible is not None:
            entry["brand_visible"] = bool(brand_visible)
        entry["source"] = (
            dict(source) if source is not None else {"kind": "generated", "generator": GENERATOR}
        )
        errs = check_entry(aid, entry)
        if errs:
            raise ValueError("; ".join(errs))
        old = self.entries.get(aid)
        if old is not None:
            keys = ("role", "license", "attribution", "changes", "source")
            if any(old.get(k) != entry.get(k) for k in keys):
                raise ValueError(f"asset {aid} is already stored with different metadata ({old.get('role')})")
            return aid
        path = self.root / f"{aid}.glb"
        if not (path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha):
            _write_atomic(path, served)
        if self.raw_root is not None:
            _write_atomic(self.raw_root / f"{aid}.glb", glb)
        self.entries[aid] = entry
        return aid

    def entry(self, aid: str) -> dict[str, Any]:
        return json.loads(json.dumps(self.entries[aid]))

    def subset(self, ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Copies of the entries for ``ids`` (what a replay header's ``assets`` map holds)."""
        return {aid: self.entry(aid) for aid in sorted(set(ids))}

    def path(self, aid: str) -> Path:
        return self.root / f"{aid}.glb"

    def raw_bytes(self, aid: str) -> bytes:
        """The uncompressed GLB for offline rendering (verified against the manifest when served raw)."""
        if self.raw_root is not None and (self.raw_root / f"{aid}.glb").is_file():
            return (self.raw_root / f"{aid}.glb").read_bytes()
        data = self.path(aid).read_bytes()
        want = self.entries.get(aid, {}).get("sha256")
        if want is not None and hashlib.sha256(data).hexdigest() != want:
            raise ValueError(f"asset {aid}: file does not match its sha256")
        return data

    def manifest(self) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "assets": {aid: self.entries[aid] for aid in sorted(self.entries)},
        }

    def write_manifest(self) -> Path:
        """Write ``root/manifest.json``, merged with entries other stores wrote there meanwhile."""
        path = self.root / "manifest.json"
        merged = dict(self.entries)
        if path.is_file():
            other = json.loads(path.read_text(encoding="utf-8")).get("assets", {})
            merged = {**other, **merged}
        self.entries = merged
        _write_atomic(path, (json.dumps(self.manifest(), indent=1, sort_keys=True) + "\n").encode("utf-8"))
        return path
