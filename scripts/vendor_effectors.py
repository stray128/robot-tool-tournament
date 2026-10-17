#!/usr/bin/env python
"""Vendor and verify the catalog effector models (product mode, not pre-registered).

Each model directory ``assets/effectors/catalog/<key>/`` holds a hand-written ``SOURCE.json``
head (the upstream repository, pinned commit, directory and model file, licence, copyright,
credit line and the licence chain) and, under ``vendor``, what to derive. This script fills in
the rest from the upstream files at the pinned commit:

* the model's MJCF and ``LICENSE``, byte for byte;
* the meshes the physics needs after the loading rules of
  :func:`rtt.effectors.catalog.vendored.prepare_child` (collision meshes, and visual meshes
  that carry a body's mass), byte for byte;
* for the meshes listed under ``vendor.hulls``: their convex hull instead (the shape MuJoCo's
  collision detector uses), optionally reduced to at most N vertices, written as OBJ in the
  upstream file's frame and units, so a replay lane can carry it inline;
* for the bodies listed under ``vendor.freeze_inertia``: their mass, centre of mass and
  inertia measured from the upstream model compiled on its own, so swapping meshes for hulls
  never changes a mass;
* one :class:`~rtt.effectors.types.AssetSource` record per shipped file (``files``) and the
  load-time replacements (``derived``).

Usage (repository root, with the repository package importable):
    python scripts/vendor_effectors.py --from MIRROR   vendor from a local checkout of the
                                                       upstream repository at the pinned commit
    python scripts/vendor_effectors.py                 same, downloading each file over https
    python scripts/vendor_effectors.py --check         re-hash every shipped file against
                                                       SOURCE.json (no network); exit 1 on a problem
    python scripts/vendor_effectors.py --only KEY ...  limit to some models

Downloads are https only, capped at 50 MB per file, and every byte that is shipped
unchanged is checked against the upstream sha256 it is recorded with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco
import numpy as np

from rtt.effectors.catalog.models import CATALOG_MODELS
from rtt.effectors.catalog.vendored import (
    CATALOG_DIR,
    SOURCE_FORMAT,
    body_subtree,
    is_visual,
    prepare_child,
    read_model,
    vendored_keys,
    verify_model,
)
from rtt.effectors.types import AssetSource

MAX_BYTES = 50 * 1024 * 1024
TIMEOUT_S = 60
HULL_DIR = "hull"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "rtt-vendor-effectors"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # https only, checked above
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url}: larger than {MAX_BYTES} bytes")
    return data


class Upstream:
    """Upstream files of one model directory, from a local mirror or the pinned commit online."""

    def __init__(self, head: dict[str, Any], mirror: Path | None, staging: Path) -> None:
        up = head["upstream"]
        self.repository = up["repository"].rstrip("/")
        self.commit = up["commit"]
        self.directory = up["directory"]
        self.mirror = mirror
        self.staging = staging / self.directory
        self.staging.mkdir(parents=True, exist_ok=True)

    def upstream_path(self, rel: str) -> str:
        return f"{self.directory}/{rel}"

    def fetch(self, rel: str) -> Path:
        """Stage ``rel`` (relative to the model directory) and return its local path."""
        dest = self.staging / rel
        if dest.is_file():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.mirror is not None:
            src = self.mirror / self.directory / rel
            if not src.is_file():
                raise FileNotFoundError(f"{src}: not in the mirror")
            shutil.copyfile(src, dest)
        else:
            raw = self.repository.replace("https://github.com/", "https://raw.githubusercontent.com/")
            dest.write_bytes(_download(f"{raw}/{self.commit}/{self.upstream_path(rel)}"))
        return dest


def _mesh_rel(spec: mujoco.MjSpec, mesh: mujoco.MjsMesh) -> str:
    meshdir = spec.meshdir or ""
    return str(Path(meshdir) / mesh.file) if meshdir else mesh.file


def _texture_rel(spec: mujoco.MjSpec, tex: mujoco.MjsTexture) -> str:
    texdir = spec.texturedir or ""
    return str(Path(texdir) / tex.file) if texdir else tex.file


def _stage_everything(up: Upstream, model_file: str) -> Path:
    """Stage the model file and every mesh and texture it references; returns the model path."""
    path = up.fetch(model_file)
    spec = mujoco.MjSpec.from_file(str(path))
    for mesh in spec.meshes:
        if mesh.file:
            up.fetch(_mesh_rel(spec, mesh))
    for tex in spec.textures:
        if tex.file:
            up.fetch(_texture_rel(spec, tex))
    return path


def _frozen_inertials(model_path: Path, bodies: list[str], root: str | None) -> dict[str, dict[str, Any]]:
    """Mass properties of ``bodies`` in the upstream model compiled on its own."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mujoco.MjModel.from_xml_path(str(model_path))
    out = {}
    for name in bodies:
        if name == "base" and root is None:
            spec = mujoco.MjSpec.from_file(str(model_path))
            first = spec.worldbody.first_body()
            bid = 1 if not first.name else model.body(first.name).id
        else:
            bid = model.body(name).id
        out[name] = {
            "mass": round(float(model.body_mass[bid]), 9),
            "ipos": [round(float(v), 9) for v in model.body_ipos[bid]],
            "iquat": [round(float(v), 9) for v in model.body_iquat[bid]],
            "inertia": [float(f"{v:.9g}") for v in model.body_inertia[bid]],
        }
    return out


def _hull_obj(mesh_path: Path, scale: np.ndarray, maxhullvert: int | None, label: str) -> bytes:
    """The convex hull MuJoCo builds for a mesh, as OBJ text in the file's own frame and units."""
    attrs = f' maxhullvert="{maxhullvert}"' if maxhullvert else ""
    xml = (
        f'<mujoco><asset><mesh name="m" file="{mesh_path}" scale="{scale[0]:.9g} {scale[1]:.9g} '
        f'{scale[2]:.9g}"{attrs}/></asset><worldbody><body><geom type="mesh" mesh="m"/></body>'
        "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    nv = int(model.mesh_vertnum[0])
    verts = model.mesh_vert[int(model.mesh_vertadr[0]) : int(model.mesh_vertadr[0]) + nv].astype(np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[0])
    verts = verts @ rot.reshape(3, 3).T + model.mesh_pos[0]  # back to the file frame (scaled)
    verts = verts / scale  # and the file's units
    g = int(model.mesh_graphadr[0])
    graph = model.mesh_graph
    hv, hf = int(graph[g]), int(graph[g + 1])
    vert_globalid = graph[g + 2 + hv : g + 2 + 2 * hv]
    face_globalid = graph[g + 2 + 2 * hv + hv + 3 * hf : g + 2 + 2 * hv + hv + 3 * hf + 3 * hf].reshape(hf, 3)
    local = {int(gid): i for i, gid in enumerate(vert_globalid)}
    lines = [
        f"# convex hull of {label}",
        f"# computed by MuJoCo {mujoco.__version__} (qhull)"
        + (f", at most {maxhullvert} vertices" if maxhullvert else "")
        + "; file frame and units",
    ]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts[vert_globalid]]
    lines += [f"f {local[int(a)] + 1} {local[int(b)] + 1} {local[int(c)] + 1}" for a, b, c in face_globalid]
    return ("\n".join(lines) + "\n").encode()


def vendor_model(key: str, mirror: Path | None, staging: Path) -> list[str]:
    """Vendor one model; returns a short report."""
    cm = CATALOG_MODELS[key]
    head_path = CATALOG_DIR / key / "SOURCE.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    if head.get("format") != SOURCE_FORMAT:
        raise ValueError(f"{key}: not a {SOURCE_FORMAT} file")
    up = Upstream(head, mirror, staging)
    model_file = head["upstream"]["model_file"]
    vend = head.get("vendor", {})
    hulls: dict[str, int | None] = dict(vend.get("hulls", {}))
    freeze: list[str] = list(vend.get("freeze_inertia", []))
    model_path = _stage_everything(up, model_file)
    license_path = up.fetch("LICENSE")
    inertials = _frozen_inertials(model_path, freeze, cm.root) if freeze else {}

    spec = mujoco.MjSpec.from_file(str(model_path))
    meshdir = spec.meshdir or ""
    body = prepare_child(spec, root=cm.root, drop_joints=cm.drop_joints, inertials=inertials)
    needed = {g.meshname for b in body_subtree(body) for g in b.geoms if g.meshname}
    unknown = set(hulls) - needed
    if unknown:
        raise ValueError(f"{key}: hulls listed for meshes the model does not use: {sorted(unknown)}")

    out_dir = CATALOG_DIR / key
    common = {
        "upstream_url": head["upstream"]["repository"],
        "upstream_commit": head["upstream"]["commit"],
        "license": head["license"],
        "copyright": head["copyright"],
        "attribution": head["attribution"],
        "status": head["status"],
        "note": head.get("note", ""),
    }
    records: list[AssetSource] = []
    shipped: set[Path] = set()

    def ship(
        rel_out: str, data: bytes, upstream_rel: str, upstream_sha: str, changes: tuple[str, ...]
    ) -> None:
        dest = out_dir / rel_out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        shipped.add(dest.resolve())
        records.append(
            AssetSource(
                path=str(dest.relative_to(REPO)),
                sha256=_sha256_bytes(data),
                upstream_path=up.upstream_path(upstream_rel),
                upstream_sha256=upstream_sha,
                changes=changes,
                **common,
            )
        )

    for rel, path in ((model_file, model_path), ("LICENSE", license_path)):
        data = path.read_bytes()
        ship(rel, data, rel, _sha256_bytes(data), ())

    mesh_files: dict[str, str] = {}
    for mesh in sorted(spec.meshes, key=lambda m: m.name):
        rel = _mesh_rel(spec, mesh)
        src = up.fetch(rel)
        data = src.read_bytes()
        if mesh.name in hulls:
            cap = hulls[mesh.name]
            hull_rel_in_meshdir = f"{HULL_DIR}/{mesh.name}.obj"
            hull = _hull_obj(
                src.resolve(), np.asarray(mesh.scale, dtype=np.float64), cap, up.upstream_path(rel)
            )
            changes = (
                "replaced by its convex hull, the shape MuJoCo's collision detector uses (MuJoCo "
                f"{mujoco.__version__}, qhull)" + (f", reduced to at most {cap} vertices" if cap else ""),
                "written as OBJ in the upstream file's frame and units",
            )
            out_rel = str(Path(meshdir) / hull_rel_in_meshdir) if meshdir else hull_rel_in_meshdir
            ship(out_rel, hull, rel, _sha256_bytes(data), changes)
            mesh_files[mesh.name] = hull_rel_in_meshdir
        else:
            ship(rel, data, rel, _sha256_bytes(data), ())
    for tex in spec.textures:
        if tex.file:
            rel = _texture_rel(spec, tex)
            data = up.fetch(rel).read_bytes()
            ship(rel, data, rel, _sha256_bytes(data), ())

    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "SOURCE.json" and path.resolve() not in shipped:
            path.unlink()
    for path in sorted(out_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    head["files"] = [r.to_dict() for r in sorted(records, key=lambda r: r.path)]
    head["derived"] = {"mesh_files": dict(sorted(mesh_files.items())), "inertials": inertials}
    head_path.write_text(json.dumps(head, indent=2) + "\n", encoding="utf-8")
    read_model.cache_clear()
    n_visual = sum(1 for b in body_subtree(body) for g in b.geoms if is_visual(g))
    total = sum((REPO / r.path).stat().st_size for r in records)
    return [
        (
            f"{key}: {len(records)} files, {total / 1024:.0f} KiB, {len(mesh_files)} hull(s), "
            f"{len(inertials)} pinned inertial(s), {n_visual} mass-bearing visual geom(s)"
        )
    ]


def check(keys: list[str]) -> int:
    bad = 0
    for key in keys:
        problems = verify_model(key)
        for p in problems:
            print(f"  {p}")
        bad += bool(problems)
        print(f"{key}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="mirror", type=Path, help="local checkout of the upstream repository")
    ap.add_argument("--check", action="store_true", help="verify shipped files only (no network)")
    ap.add_argument("--only", nargs="*", default=None, help="model keys to process")
    args = ap.parse_args(argv)
    keys = args.only or list(vendored_keys())
    if args.check:
        return check(keys)
    with tempfile.TemporaryDirectory(prefix="rtt-vendor-") as tmp:
        for key in keys:
            for line in vendor_model(key, args.mirror, Path(tmp)):
                print(line)
    return check(keys)


if __name__ == "__main__":
    sys.exit(main())
