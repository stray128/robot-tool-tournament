"""Vendored catalog models: licence records, verification and loading into the shared scene.

Every vendored model lives in ``assets/effectors/catalog/<key>/`` under a generic key
(``hand-4f-16dof``), with the upstream licence text beside it and a ``SOURCE.json`` that
pins each file (:class:`~rtt.effectors.types.AssetSource`: path, sha256 as shipped, upstream
URL, commit, upstream path and sha256, SPDX licence, copyright, credit line, changes).
``scripts/vendor_effectors.py`` writes and checks those records; :func:`verify_model`
re-hashes the shipped files against them.

``SOURCE.json`` (``format = "rtt-effector-source"``, version 1):

* ``model``: the generic key; ``display_name``: the brand-free screen name;
* ``upstream``: ``repository``, ``commit``, ``directory`` and ``model_file`` (the MJCF);
* ``license``, ``copyright``, ``attribution``, ``status`` (``green``/``amber``), ``note``,
  ``chain`` (the upstream licence chain, as audited in ``docs/EFFECTORS.md`` section 9);
* ``vendor``: what the vendoring script derived (``hulls``: collision meshes replaced by
  their convex hull; ``freeze_inertia``: bodies whose inertia is pinned at the upstream value);
* ``files``: one :class:`AssetSource` record per shipped file;
* ``derived``: ``mesh_files`` (mesh name -> shipped derived file, relative to the model's mesh
  directory) and ``inertials`` (body -> ``mass``, ``ipos``, ``iquat``, ``inertia``, measured from
  the upstream model compiled on its own).

:func:`load_child` turns the vendored MJCF into an attachable :class:`mujoco.MjSpec`:

* keyframes, lights, cameras and sensors go (the scene owns them);
* the upstream wrist or positioning joints named by the model go, with their actuators (the
  shared 4-DoF wrist replaces them), and the root body is rigid under the mount;
* visual geoms (no collision) that carry no mass go, then every mesh, material and texture
  nothing uses: the scene loads only what the physics needs. Visual geoms that carry mass (a
  body whose inertia comes from its visual mesh) stay, so masses are the upstream ones;
* derived files replace upstream collision meshes and pinned inertials replace geometry-derived
  ones, exactly as ``SOURCE.json`` records;
* default classes are renamed to neutral names, so no upstream brand word reaches an element
  name (and from there a lane header).

Actuator gains, force ranges, joint ranges, damping and contact parameters stay as upstream.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rtt.effectors.scene import REPO_ROOT
from rtt.effectors.types import AssetSource, find_brand_tokens

__all__ = [
    "CATALOG_DIR",
    "SOURCE_FORMAT",
    "VendoredModel",
    "body_subtree",
    "is_visual",
    "load_child",
    "mesh_vertex_count",
    "prepare_child",
    "read_model",
    "vendored_keys",
    "verify_model",
]

CATALOG_DIR = REPO_ROOT / "assets" / "effectors" / "catalog"
SOURCE_FORMAT = "rtt-effector-source"
SOURCE_VERSION = 1


@dataclass(frozen=True)
class VendoredModel:
    """One vendored model's ``SOURCE.json`` (module docstring)."""

    key: str
    directory: Path
    display_name: str
    upstream: Mapping[str, str]
    license: str
    copyright: str
    attribution: str
    status: str
    note: str
    sources: tuple[AssetSource, ...]
    mesh_files: Mapping[str, str] = field(default_factory=dict)
    inertials: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def model_path(self) -> Path:
        return self.directory / self.upstream["model_file"]


def vendored_keys() -> tuple[str, ...]:
    """Every model directory with a ``SOURCE.json`` (sorted)."""
    if not CATALOG_DIR.is_dir():
        return ()
    return tuple(sorted(p.parent.name for p in CATALOG_DIR.glob("*/SOURCE.json")))


@cache
def read_model(key: str) -> VendoredModel:
    """Parse ``assets/effectors/catalog/<key>/SOURCE.json`` (licence rules checked on load)."""
    directory = CATALOG_DIR / key
    raw = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
    if raw.get("format") != SOURCE_FORMAT or raw.get("version") != SOURCE_VERSION:
        raise ValueError(f"{key}: SOURCE.json is not {SOURCE_FORMAT} version {SOURCE_VERSION}")
    if raw.get("model") != key:
        raise ValueError(f"{key}: SOURCE.json describes {raw.get('model')!r}")
    if raw.get("status") not in ("green", "amber"):
        raise ValueError(f"{key}: status must be green or amber (red models are never vendored)")
    sources = tuple(
        AssetSource(**{**f, "changes": tuple(f.get("changes", ()))}) for f in raw.get("files", ())
    )
    derived = raw.get("derived", {})
    return VendoredModel(
        key=key,
        directory=directory,
        display_name=raw["display_name"],
        upstream=dict(raw["upstream"]),
        license=raw["license"],
        copyright=raw["copyright"],
        attribution=raw["attribution"],
        status=raw["status"],
        note=raw.get("note", ""),
        sources=sources,
        mesh_files=dict(derived.get("mesh_files", {})),
        inertials={k: dict(v) for k, v in derived.get("inertials", {}).items()},
        raw=raw,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model(key: str) -> list[str]:
    """Problems with a vendored model's files (empty: every file re-hashes to its record)."""
    vm = read_model(key)
    problems = []
    listed = set()
    for src in vm.sources:
        path = REPO_ROOT / src.path
        listed.add(path.resolve())
        if not path.is_file():
            problems.append(f"{src.path}: missing")
        elif _sha256(path) != src.sha256:
            problems.append(f"{src.path}: sha256 differs from SOURCE.json")
        if not src.path.startswith(f"assets/effectors/catalog/{key}/"):
            problems.append(f"{src.path}: outside the model's directory")
    if vm.license not in {s.license for s in vm.sources}:
        problems.append(f"{key}: no file carries the model licence {vm.license}")
    for path in sorted(vm.directory.rglob("*")):
        if path.is_file() and path.name != "SOURCE.json" and path.resolve() not in listed:
            problems.append(f"{path.relative_to(REPO_ROOT)}: shipped but not in SOURCE.json")
    if not vm.model_path.is_file():
        problems.append(f"{key}: model file {vm.upstream['model_file']} missing")
    return problems


# --------------------------------------------------------------------------- loading


def body_subtree(body: mujoco.MjsBody) -> Iterator[mujoco.MjsBody]:
    """``body`` and every body below it (depth first)."""
    yield body
    for child in body.bodies:
        yield from body_subtree(child)


def is_visual(geom: mujoco.MjsGeom) -> bool:
    """A geom that never collides (``contype`` and ``conaffinity`` both 0)."""
    return geom.contype == 0 and geom.conaffinity == 0


def _carries_mass(geom: mujoco.MjsGeom, body: mujoco.MjsBody) -> bool:
    if body.explicitinertial:
        return False
    if not math.isnan(geom.mass):
        return geom.mass > 0
    return geom.density > 0


def _set_inertial(body: mujoco.MjsBody, inertial: Mapping[str, Any]) -> None:
    body.explicitinertial = True
    body.mass = float(inertial["mass"])
    body.ipos = [float(v) for v in inertial["ipos"]]
    body.iquat = [float(v) for v in inertial["iquat"]]
    body.inertia = [float(v) for v in inertial["inertia"]]
    body.fullinertia = [math.nan] * 6


def load_child(
    key: str,
    *,
    root: str | None,
    drop_joints: tuple[str, ...] = (),
) -> tuple[mujoco.MjSpec, mujoco.MjsBody]:
    """The vendored model as an attachable spec, and the body to attach (module docstring).

    ``root`` names the upstream body whose subtree is the effector (``None``: the worldbody's
    first body). ``drop_joints`` are upstream joints removed together with their actuators.
    """
    vm = read_model(key)
    child = mujoco.MjSpec.from_file(str(vm.model_path))
    body = prepare_child(
        child, root=root, drop_joints=drop_joints, inertials=vm.inertials, mesh_files=vm.mesh_files
    )
    return child, body


def prepare_child(
    child: mujoco.MjSpec,
    *,
    root: str | None,
    drop_joints: tuple[str, ...] = (),
    inertials: Mapping[str, Mapping[str, Any]] | None = None,
    mesh_files: Mapping[str, str] | None = None,
) -> mujoco.MjsBody:
    """Apply the loading rules of the module docstring to a parsed upstream spec, in place.

    Returns the body to attach (its pose reset to the identity, so the mount frame places it).
    ``scripts/vendor_effectors.py`` runs the same rules on the upstream file to decide which
    files to ship.
    """
    inertials = inertials or {}
    mesh_files = mesh_files or {}
    for mesh in child.meshes:  # an unnamed mesh is named after its file at compile time: pin that now
        if not mesh.name:
            mesh.name = Path(mesh.file).stem
    for kind in ("keys", "lights", "cameras", "sensors"):
        for element in list(getattr(child, kind)):
            child.delete(element)
    body = child.body(root) if root is not None else child.worldbody.first_body()
    if body is None:
        raise ValueError(f"no body {root!r} in the upstream model")
    if not body.name:
        body.name = "base"
    dropped = set(drop_joints)
    unknown = dropped - {j.name for j in child.joints}
    if unknown:
        raise ValueError(f"no joints {sorted(unknown)} to drop")
    for act in list(child.actuators):
        if act.target in dropped:
            child.delete(act)
    for j in list(child.joints):
        if j.name in dropped:
            child.delete(j)
    subtree = list(body_subtree(body))
    for b in subtree:
        if b.name in inertials:
            _set_inertial(b, inertials[b.name])
    for b in subtree:
        for g in list(b.geoms):
            if is_visual(g) and not _carries_mass(g, b):
                child.delete(g)
    for mesh in child.meshes:
        if mesh.name in mesh_files:
            mesh.file = mesh_files[mesh.name]
    _drop_unused_assets(child, subtree)
    _neutral_classes(child)
    body.pos = [0.0, 0.0, 0.0]
    body.quat = [1.0, 0.0, 0.0, 0.0]
    return body


def _drop_unused_assets(child: mujoco.MjSpec, subtree: list[mujoco.MjsBody]) -> None:
    """Delete meshes, materials and textures nothing in the attached subtree uses."""
    geoms = [g for b in subtree for g in b.geoms]
    used_meshes = {g.meshname for g in geoms if g.meshname}
    for mesh in list(child.meshes):
        if mesh.name not in used_meshes:
            child.delete(mesh)
    used_materials = {g.material for g in geoms if g.material}
    used_materials |= {s.material for b in subtree for s in b.sites if s.material}
    for mat in list(child.materials):
        if mat.name not in used_materials:
            child.delete(mat)
    used_textures = {t for m in child.materials for t in m.textures if t}
    for tex in list(child.textures):
        if tex.name not in used_textures:
            child.delete(tex)


def _neutral_classes(child: mujoco.MjSpec) -> None:
    """Rename every default class whose name carries a brand word to a neutral ``c<i>``."""
    for i, d in enumerate(_classes(child)):
        if d.name != "main" and (find_brand_tokens(d.name) or find_brand_tokens(d.name.replace("_", " "))):
            d.name = f"c{i}"


def _classes(child: mujoco.MjSpec) -> list[mujoco.MjsDefault]:
    """Every default class referenced by an element or a body's childclass (stable order)."""
    out: list[mujoco.MjsDefault] = []
    names: set[str] = set()

    def add(d: Any) -> None:
        name = getattr(d, "name", None)
        if name and name not in names:
            names.add(name)
            out.append(d)

    for kind in ("bodies", "geoms", "joints", "sites", "actuators", "tendons", "meshes", "materials"):
        for e in getattr(child, kind):
            add(getattr(e, "classname", None))
            if kind == "bodies" and e.childclass:
                add(child.find_default(e.childclass))
    return out


def mesh_vertex_count(model: mujoco.MjModel, geoms: np.ndarray) -> int:
    """Vertices the replay header carries for these geoms' meshes."""
    total = 0
    for g in geoms:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            total += int(model.mesh_vertnum[model.geom_dataid[g]])
    return total
