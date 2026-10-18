"""Vendoring library items: fetch at the pinned commit, derive the simulation model, record.

``scripts/vendor_library.py`` drives this module. Each item's ``SOURCE.json`` head names what
to ingest (``ingest``):

* ``licence_files``: upstream licence and notice texts, shipped byte for byte under
  ``LICENSES/``. Their sha256 at the pinned commit is recorded in ``lock.json``; when a
  re-vendor finds a different licence file at the same pin (or a bump changes it), the run
  stops with :class:`LicenceDrift` until someone reviews the new text.
* ``sim``, one of:

  * ``{"type": "mjcf", "entry": path, "root": body, ...}``: an upstream MJCF (includes
    resolved). The effector subtree is attached into a fresh spec and written as one MJCF
    (``sim/<key>.xml``): upstream positioning joints and marker bodies removed, every body's
    mass properties pinned at the upstream values, contact margins set to 0 when asked (the
    shared scene refuses effector margins), visual-only geoms dropped (their bodies carry
    pinned masses), collision meshes replaced by their convex hull (the shape MuJoCo collides
    anyway) and element names that carry a maker's or product's name renamed;
  * ``{"type": "urdf", "entry": path, "root": link, ...}``: a URDF (or a plain-XML xacro)
    subtree converted to MJCF the same way, with the actuators the plan declares;
  * ``{"type": "sdf_boxes", ...}``: an SDF model built from box primitives (converted);
  * ``{"type": "rigid_proxy", "parts": [...], ...}``: printed parts turned into a rigid
    proxy (:mod:`rtt.library.proxy`);
  * ``{"type": "family", "family": "suction", "params": {...}}``: a mapping onto an existing
    effector family (no files);
  * absent: no simulation (``visual_only`` or ``none``).

* ``files``: print, CAD and documentation files shipped byte for byte (each with its own SPDX
  id), unless larger than :data:`MAX_FILE_BYTES`; skipped files are recorded with a link.
* ``preview``: the catalog page's GLB (:mod:`rtt.library.preview`), a derived file under
  ``web/public/data/library/glb/``.

Everything written is deterministic: the same pins and the same code give the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mujoco
import numpy as np
import trimesh

from rtt.library.brands import find_library_brands
from rtt.library.fetch import Fetcher, blob_url
from rtt.library.licence import ChainLink, Rating
from rtt.library.preview import MAX_PREVIEW_BYTES, model_meshes, scene_glb
from rtt.library.proxy import build_rigid_proxy, load_mesh, obj_text, proxy_mjcf
from rtt.library.records import (
    GENERATED,
    LIBRARY_DIR,
    REPO_ROOT,
    SOURCE_FORMAT,
    LibraryFile,
    licence_key,
    read_item,
)

__all__ = [
    "MAX_FILE_BYTES",
    "PREVIEW_DIR",
    "IngestReport",
    "LicenceDrift",
    "vendor_item",
]

#: Per-file cap for shipped print, CAD and documentation files; larger ones are linked.
MAX_FILE_BYTES: Final = 3 * 1024 * 1024
PREVIEW_DIR: Final = REPO_ROOT / "web" / "public" / "data" / "library" / "glb"
OWN_REPOSITORY: Final = "https://github.com/stray128/robot-tool-tournament"
_INCLUDE = re.compile(r'<include\s+file="([^"]+)"')


class LicenceDrift(RuntimeError):
    """An upstream licence file differs from the one ``lock.json`` pinned: review it first."""


@dataclass
class IngestReport:
    key: str
    files: int
    bytes: int
    skipped: int
    preview_bytes: int
    notes: list[str]

    def line(self) -> str:
        return (
            f"{self.key}: {self.files} files, {self.bytes / 1024:.0f} KiB, {self.skipped} skipped, "
            f"preview {self.preview_bytes / 1024:.0f} KiB"
            + (f" ({'; '.join(self.notes)})" if self.notes else "")
        )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm(path: str) -> str:
    return os.path.normpath(path).replace(os.sep, "/")


def _neutral(name: str, kind: str, i: int) -> str:
    """``name`` unless it carries a brand word (any separator), else ``<kind><i>``."""
    if not name:
        return name
    spaced = re.sub(r"[_/.-]+", " ", name)
    split = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)  # "abc7" -> "abc 7"
    if find_library_brands(name) or find_library_brands(spaced) or find_library_brands(split):
        return f"{kind}{i}"
    return name


class _Writer:
    """Collects the shipped files of one item and their records."""

    def __init__(self, key: str, head: Mapping[str, Any]) -> None:
        self.key = key
        self.head = head
        self.dir = LIBRARY_DIR / key
        self.repo = head["upstream"]["repository"]
        self.commit = head["upstream"]["commit"]
        self.files: list[LibraryFile] = []
        self.skipped: list[dict[str, Any]] = []
        self.written: set[Path] = set()

    def ship(
        self,
        rel: str,
        data: bytes,
        *,
        role: str,
        upstream_path: str,
        upstream_sha256: str,
        license: str,
        copyright: str | None = None,
        changes: Sequence[str] = (),
        chain: Sequence[ChainLink] = (),
        generated: bool = False,
        repository: str | None = None,
        commit: str | None = None,
    ) -> str:
        dest = self.dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.written.add(dest.resolve())
        path = str(dest.relative_to(REPO_ROOT))
        self.files.append(
            LibraryFile(
                path=path,
                sha256=_sha(data),
                size=len(data),
                role=role,
                upstream_url=OWN_REPOSITORY if generated else (repository or self.repo),
                upstream_commit=GENERATED if generated else (commit or self.commit),
                upstream_path=upstream_path,
                upstream_sha256=upstream_sha256,
                license=license,
                copyright=copyright or self.head["copyright"],
                changes=tuple(changes),
                chain=tuple(chain),
            )
        )
        return path

    def prune(self) -> None:
        for path in sorted(self.dir.rglob("*")):
            if path.is_file() and path.name != "SOURCE.json" and path.resolve() not in self.written:
                path.unlink()
        for path in sorted(self.dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()


# --------------------------------------------------------------------------- MJCF


def _stage_mjcf(fetcher: Fetcher, repo: str, commit: str, entry: str) -> tuple[mujoco.MjSpec, dict[str, str]]:
    """Parse ``entry`` (includes fetched first) with every mesh resolved to a staged local file.

    Returns the spec, with absolute mesh paths and no textures (nothing here needs them), and
    ``{mesh name: upstream path}``. Mesh paths in included files are tried against the entry's
    directory, then against the including file's directory, so models written for either
    MuJoCo convention load.
    """
    path = fetcher.path(repo, commit, entry)
    base = str(Path(entry).parent)
    todo = [entry]
    seen = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        text = fetcher.path(repo, commit, cur).read_text(encoding="utf-8", errors="replace")
        for inc in _INCLUDE.findall(text):
            todo.append(_norm(str(Path(base) / inc)))  # includes resolve against the entry's directory
    spec = mujoco.MjSpec.from_file(str(path))
    root = fetcher.path(repo, commit, entry).parents[len(Path(entry).parts) - 1]
    sources: dict[str, str] = {}
    for i, mesh in enumerate(spec.meshes):
        if not mesh.name:
            mesh.name = Path(mesh.file).stem if mesh.file else f"mesh{i}"
        if not mesh.file:
            continue
        f = mesh.file
        candidates = []
        if os.path.isabs(f):
            try:
                rel = str(Path(f).resolve().relative_to(root.resolve()))
            except ValueError:
                rel = Path(f).name
            candidates.append(_norm(rel))
            parts = Path(rel).parts
            for k in range(1, len(parts)):  # strip the including file's directory, step by step
                candidates.append(
                    _norm(str(Path(base) / Path(*parts[len(Path(base).parts) + k :])))
                    if len(parts) > len(Path(base).parts) + k
                    else ""
                )
        else:
            candidates.append(_norm(str(Path(base) / (spec.meshdir or "") / f)))
        local = None
        for cand in [c for c in candidates if c]:
            try:
                local = fetcher.path(repo, commit, cand)
            except Exception:  # noqa: BLE001, S112 - not there: try the next convention
                continue
            sources[mesh.name] = cand
            break
        if local is None:
            raise FileNotFoundError(f"{entry}: mesh {mesh.name} ({f}) not found upstream")
        mesh.file = str(local.resolve())
    spec.meshdir = ""
    for tex in list(spec.textures):
        spec.delete(tex)
    for mat in spec.materials:
        mat.textures = [""] * len(mat.textures)
    return spec, sources


def _compile(spec_or_path: Any) -> mujoco.MjModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(spec_or_path, Path | str):
            return mujoco.MjModel.from_xml_path(str(spec_or_path))
        return spec_or_path.compile()


def _hull_obj(verts: np.ndarray, faces: np.ndarray, label: str) -> bytes:
    lines = [
        f"# convex hull of {label}",
        f"# computed by MuJoCo {mujoco.__version__} (qhull); file frame, scaled to metres",
    ]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def _mesh_hull(
    mesh_path: Path, scale: Sequence[float], label: str, refpos: Any = None, refquat: Any = None
) -> bytes:
    """The convex hull MuJoCo builds for a mesh file, in the file's frame scaled to metres."""
    attrs = f'scale="{scale[0]:.9g} {scale[1]:.9g} {scale[2]:.9g}"'
    xml = (
        f'<mujoco><asset><mesh name="m" file="{mesh_path}" {attrs}/></asset><worldbody><body>'
        '<geom type="mesh" mesh="m"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    nv = int(model.mesh_vertnum[0])
    verts = model.mesh_vert[int(model.mesh_vertadr[0]) : int(model.mesh_vertadr[0]) + nv].astype(np.float64)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.mesh_quat[0])
    verts = verts @ rot.reshape(3, 3).T + model.mesh_pos[0]  # back to the file frame (scaled)
    g = int(model.mesh_graphadr[0])
    graph = model.mesh_graph
    hv, hf = int(graph[g]), int(graph[g + 1])
    vert_globalid = graph[g + 2 + hv : g + 2 + 2 * hv]
    face_globalid = graph[g + 2 + 2 * hv + hv + 3 * hf : g + 2 + 2 * hv + hv + 6 * hf].reshape(hf, 3)
    local = {int(gid): i for i, gid in enumerate(vert_globalid)}
    faces = np.array([[local[int(a)], local[int(b)], local[int(c)]] for a, b, c in face_globalid])
    return _hull_obj(verts[vert_globalid], faces, label)


def _pin_inertials(spec: mujoco.MjSpec, model: mujoco.MjModel, names: Sequence[str]) -> None:
    for name in names:
        b = spec.body(name)
        bid = model.body(name).id
        b.explicitinertial = True
        b.mass = float(model.body_mass[bid])
        b.ipos = [float(v) for v in model.body_ipos[bid]]
        b.iquat = [float(v) for v in model.body_iquat[bid]]
        b.inertia = [float(v) for v in model.body_inertia[bid]]
        b.fullinertia = [math.nan] * 6


def _effector_spec(
    upstream: mujoco.MjSpec,
    *,
    root: str | None,
    drop_joints: Sequence[str],
    delete_bodies: Sequence[str],
) -> mujoco.MjSpec:
    """A fresh spec holding only the effector subtree of ``upstream`` (attached at the world)."""
    for kind in ("keys", "lights", "cameras", "sensors"):
        for element in list(getattr(upstream, kind)):
            upstream.delete(element)
    for name in delete_bodies:
        upstream.delete(upstream.body(name))
    dropped = set(drop_joints)
    for act in list(upstream.actuators):
        if act.target in dropped:
            upstream.delete(act)
    for j in list(upstream.joints):
        if j.name in dropped:
            upstream.delete(j)
    body = upstream.body(root) if root else upstream.worldbody.first_body()
    if body is None:
        raise ValueError(f"no body {root!r} upstream")
    out = mujoco.MjSpec()
    out.compiler.degree = False
    frame = out.worldbody.add_frame()
    # MjSpec.attach copies the subtree and everything that refers only to it; a temporary
    # prefix keeps the child's default classes distinct, then the names are restored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame.attach_body(body, _TMP, "")
    # an XML round trip restores the names (every name and reference carries the prefix) and
    # drops what attach keeps hidden (the child's asset directories)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = out.to_xml().replace(f'"{_TMP}main"', '"upstream"')
    xml = re.sub(rf'(["\s]){_TMP}', r"\1", xml)
    clean = mujoco.MjSpec.from_string(xml)
    clean.compiler.degree = False
    return clean


_TMP: Final = "upstream0_"


def _wrap_bodies(spec: mujoco.MjSpec, name: str, bodies: Sequence[str]) -> mujoco.MjSpec:
    """Move top-level ``bodies`` into one new massless body ``name`` at the world origin.

    For a model whose effector is several world bodies (a hand base and separately placed
    fingers): the poses, masses and joints are unchanged, they only share one root.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = ET.fromstring(spec.to_xml())
    world = root.find("worldbody")
    if world is None:
        raise ValueError("no worldbody")
    wrapper = ET.Element("body", {"name": name})
    for el in list(world):
        if el.tag == "body" and el.get("name") in bodies:
            world.remove(el)
            wrapper.append(el)
    missing = set(bodies) - {el.get("name") for el in wrapper}
    if missing:
        raise ValueError(f"no top-level bodies {sorted(missing)} to wrap")
    world.append(wrapper)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def _vendor_mjcf(w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher) -> dict[str, Any]:
    entry = plan["entry"]
    upstream, mesh_sources = _stage_mjcf(fetcher, w.repo, w.commit, entry)
    if plan.get("wrap"):
        upstream = _wrap_bodies(upstream, plan["wrap"]["name"], plan["wrap"]["bodies"])
    for i, b in enumerate(upstream.bodies):
        if b.name == "" and i > 0:
            b.name = f"body{i}"
    up_model = _compile(upstream)
    root = plan.get("root")
    spec = _effector_spec(
        upstream,
        root=root,
        drop_joints=plan.get("drop_joints", ()),
        delete_bodies=plan.get("delete_bodies", ()),
    )
    return _finish_spec(
        w, {**plan, "mesh_sources": mesh_sources}, spec, up_model, entry, fetcher, upstream_root=root
    )


def _finish_spec(
    w: _Writer,
    plan: Mapping[str, Any],
    spec: mujoco.MjSpec,
    up_model: mujoco.MjModel | None,
    entry: str,
    fetcher: Fetcher,
    *,
    upstream_root: str | None,
    extra_changes: Sequence[str] = (),
) -> dict[str, Any]:
    """Pin inertials, hull collision meshes, rename brand words, write MJCF and meshes."""
    lic = plan["license"]
    changes = [*extra_changes]
    bodies = [b for b in spec.bodies if b.name != "world"]
    names = [b.name for b in bodies]
    if up_model is not None:
        _pin_inertials(spec, up_model, [n for n in names if n])
        changes.append(
            "every body's mass, centre of mass and inertia pinned at the upstream model's compiled values"
        )
    # visual-only geoms carry no mass once inertials are pinned: drop them (and their meshes)
    n_visual = 0
    for b in bodies:
        for g in list(b.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
                n_visual += 1
    if n_visual:
        changes.append(
            f"{n_visual} visual-only geoms removed (the catalog preview draws the upstream visuals)"
        )
    # the shared scene refuses effector contact margins and geom priorities (docs/EFFECTORS.md 2.3)
    n_margin = n_priority = 0
    for b in bodies:
        for g in b.geoms:
            if g.margin > 0:
                g.margin = 0.0
                n_margin += 1
            if g.priority > 0:
                g.priority = 0
                n_priority += 1
    if n_margin:
        changes.append(
            f"contact margin set to 0 on {n_margin} geoms (the shared scene refuses effector contact margins)"
        )
    if n_priority:
        changes.append(
            f"contact priority set to 0 on {n_priority} geoms (the part's contact parameters govern every contact)"
        )
    for tex in list(spec.textures):
        spec.delete(tex)
    for mat in spec.materials:
        mat.textures = [""] * len(mat.textures)
    used = {g.meshname for b in bodies for g in b.geoms if g.meshname}
    for m in list(spec.meshes):
        if m.name not in used:
            spec.delete(m)
    # collision meshes -> convex hulls, written in the file frame, scaled to metres
    mesh_lic = plan.get("mesh_licenses", {})
    for m in sorted(spec.meshes, key=lambda x: x.name):
        src_rel = plan["mesh_sources"][m.name]
        src_path = fetcher.path(w.repo, w.commit, src_rel)
        hull = _mesh_hull(src_path.resolve(), list(m.scale), src_rel)
        out_rel = f"sim/meshes/{_neutral(m.name, 'mesh', len(w.files))}.obj"
        w.ship(
            out_rel,
            hull,
            role="derived",
            upstream_path=src_rel,
            upstream_sha256=_sha(src_path.read_bytes()),
            license=mesh_lic.get(Path(src_rel).name, lic),
            changes=(
                f"replaced by its convex hull, the shape MuJoCo's collision detector uses (MuJoCo {mujoco.__version__}, qhull)",
                "written as OBJ in the upstream file's frame, scaled to metres",
            ),
        )
        m.file = out_rel.removeprefix("sim/")
        m.scale = [1.0, 1.0, 1.0]
    spec.meshdir = str(w.dir / "sim")  # to_xml compiles: resolve the shipped hulls
    spec.compiler.degree = False
    xml = _relative_xml(spec, w)
    xml, renames = _neutral_xml(xml)
    if renames:
        changes.append(f"{len(renames)} element or class names carrying a maker's or product's name renamed")
    model_rel = f"sim/{w.key}.xml"
    up_bytes = fetcher.read(w.repo, w.commit, entry)
    w.ship(
        model_rel,
        xml.encode(),
        role="derived",
        upstream_path=entry,
        upstream_sha256=_sha(up_bytes),
        license=lic,
        changes=(
            f"the effector subtree{' from body ' + repr(upstream_root) if upstream_root else ''} written as one MJCF by MuJoCo {mujoco.__version__}",
            *changes,
        ),
    )
    model = _compile(w.dir / model_rel)  # the written file must compile on its own
    if up_model is not None:
        _check_equivalent(up_model, model, renames)
    return {
        "model_file": model_rel,
        "n_bodies": int(model.nbody - 1),
        "n_joints": int(model.njnt),
        "n_actuators": int(model.nu),
        "mass_kg": round(float(model.body_mass[1:].sum()), 6),
        "changes": changes,
    }


_NAME_ATTR = re.compile(r'(\s(?:name|class|childclass)=)"([^"]*)"')


def _neutral_xml(xml: str) -> tuple[str, dict[str, str]]:
    """Rename every name and class that carries a brand word, with every reference to it.

    References in MJCF are attribute values equal to a name, so exact attribute values are
    replaced. Returns the new text and ``{new: old}``.
    """
    names = [m.group(2) for m in _NAME_ATTR.finditer(xml)]
    mapping: dict[str, str] = {}
    for i, old in enumerate(dict.fromkeys(names)):
        new = _neutral(old, "n", i)
        if new != old:
            mapping[old] = new
    if not mapping:
        return xml, {}
    out = re.sub(r'="([^"]*)"', lambda m: '="' + mapping.get(m.group(1), m.group(1)) + '"', xml)
    return out, {v: k for k, v in mapping.items()}


def _relative_xml(spec: mujoco.MjSpec, w: _Writer) -> str:
    """``spec`` as MJCF text with mesh paths relative to the written file (no absolute paths)."""
    xml = spec.to_xml()
    absdir = str(w.dir / "sim")
    xml = re.sub(r'\s+meshdir="[^"]*"', "", xml)
    if absdir in xml:
        raise ValueError(f"{w.key}: the written MJCF still names a local path")
    return xml


def _check_equivalent(up: mujoco.MjModel, model: mujoco.MjModel, renames: Mapping[str, str]) -> None:
    """The written effector must match the upstream subtree: masses, relative body poses,
    joint types, axes and ranges, actuator gains, biases and ranges (renamed elements mapped)."""
    problems = []

    def up_name(name: str) -> str:
        return renames.get(name, name)

    def up_id(name: str) -> int | None:
        try:
            return int(up.body(up_name(name)).id)
        except KeyError:
            return None

    for b in range(2, model.nbody):  # body 1 is the effector root (its pose is the mount's)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        ub = up_id(name)
        if ub is None:
            problems.append(f"body {name}: not in the upstream model")
            continue
        if not np.allclose(model.body_pos[b], up.body_pos[ub], atol=1e-5):
            problems.append(f"body {name}: position differs")
        q, uq = model.body_quat[b], up.body_quat[ub]
        if min(np.abs(q - uq).max(), np.abs(q + uq).max()) > 1e-5:
            problems.append(f"body {name}: orientation differs")
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        ub = up_id(name)
        if ub is None:
            continue
        if not math.isclose(float(model.body_mass[b]), float(up.body_mass[ub]), rel_tol=1e-5, abs_tol=1e-9):
            problems.append(f"body {name}: mass {model.body_mass[b]:.6g} != upstream {up.body_mass[ub]:.6g}")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        uj = up.joint(up_name(name)).id
        same = (
            model.jnt_type[j] == up.jnt_type[uj]
            and np.allclose(model.jnt_axis[j], up.jnt_axis[uj], atol=1e-6)
            and np.allclose(model.jnt_range[j], up.jnt_range[uj], atol=1e-6)
            and np.allclose(model.jnt_pos[j], up.jnt_pos[uj], atol=1e-6)
        )
        if not same:
            problems.append(f"joint {name}: type, axis, anchor or range differs")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if not name:
            continue
        ua = up.actuator(up_name(name)).id
        same = (
            np.allclose(model.actuator_gainprm[a, :3], up.actuator_gainprm[ua, :3])
            and np.allclose(model.actuator_biasprm[a, :3], up.actuator_biasprm[ua, :3])
            and np.allclose(model.actuator_ctrlrange[a], up.actuator_ctrlrange[ua])
            and np.allclose(model.actuator_forcerange[a], up.actuator_forcerange[ua])
        )
        if not same:
            problems.append(f"actuator {name}: gain, bias or range differs")
    if problems:
        raise ValueError("the written model differs from the upstream subtree: " + "; ".join(problems))


# --------------------------------------------------------------------------- URDF


def _floats(text: str | None, n: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * n
    vals = [float(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return vals


def _rpy_quat(rpy: Sequence[float]) -> list[float]:
    """URDF roll-pitch-yaw (fixed axes x, y, z) as a wxyz quaternion."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _urdf_spec(w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher) -> mujoco.MjSpec:
    """A URDF subtree (plain XML; xacro macros are not expanded) as an MjSpec.

    ``plan["geometry"]`` picks the ``collision`` (default) or ``visual`` elements. Mesh files
    are staged and named by their upstream path (``mesh.file``); package URIs resolve through
    ``mesh_paths`` (exact) or ``package_roots`` (package name -> upstream directory).
    """
    entry = plan["entry"]
    text = fetcher.read(w.repo, w.commit, entry).decode("utf-8")
    tree = ET.fromstring(text)
    links = {link.get("name"): link for link in tree.iter("link")}
    joints = [j for j in tree.iter("joint")]
    children: dict[str, list[ET.Element]] = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)  # type: ignore[union-attr]
    root = plan["root"]
    keep = set(plan.get("links") or ())
    mesh_map: Mapping[str, str] = plan.get("mesh_paths", {})  # package:// URI -> upstream path
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    mesh_names: dict[str, str] = {}

    def add_link(
        parent: mujoco.MjsBody, name: str, pos: Sequence[float], quat: Sequence[float]
    ) -> mujoco.MjsBody:
        link = links[name]
        body = parent.add_body(name=name, pos=list(pos), quat=list(quat))
        inert = link.find("inertial")
        if inert is not None:
            org = inert.find("origin")
            ipos = _floats(org.get("xyz") if org is not None else None, 3)
            irpy = _floats(org.get("rpy") if org is not None else None, 3)
            i = inert.find("inertia")
            full = [float(i.get(k, 0.0)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]  # type: ignore[union-attr]
            body.explicitinertial = True
            body.mass = float(inert.find("mass").get("value"))  # type: ignore[union-attr]
            body.ipos = ipos
            body.iquat = _rpy_quat(irpy)
            body.fullinertia = full
        for col in link.findall(plan.get("geometry", "collision")):
            org = col.find("origin")
            gpos = _floats(org.get("xyz") if org is not None else None, 3)
            gquat = _rpy_quat(_floats(org.get("rpy") if org is not None else None, 3))
            geo = col.find("geometry")
            mesh = geo.find("mesh") if geo is not None else None
            if mesh is not None:
                uri = mesh.get("filename", "")
                up_path = mesh_map.get(uri) or _package_path(uri, plan.get("package_roots", {}))
                scale = _floats(mesh.get("scale"), 3, 1.0)
                mname = mesh_names.setdefault(up_path, f"m{len(mesh_names)}")
                if not any(m.name == mname for m in spec.meshes):
                    spec.add_mesh(name=mname, file=up_path, scale=scale)
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, pos=gpos, quat=gquat)
            elif geo is not None and geo.find("box") is not None:
                size = [v / 2 for v in _floats(geo.find("box").get("size"), 3)]  # type: ignore[union-attr]
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=gpos, quat=gquat)
            elif geo is not None and geo.find("cylinder") is not None:
                c = geo.find("cylinder")
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[float(c.get("radius")), float(c.get("length")) / 2, 0],  # type: ignore[union-attr,arg-type]
                    pos=gpos,
                    quat=gquat,
                )
        for j in children.get(name, []):
            child = j.find("child").get("link")  # type: ignore[union-attr]
            if keep and child not in keep:
                continue
            org = j.find("origin")
            jpos = _floats(org.get("xyz") if org is not None else None, 3)
            jquat = _rpy_quat(_floats(org.get("rpy") if org is not None else None, 3))
            cb = add_link(body, child, jpos, jquat)  # type: ignore[arg-type]
            jtype = j.get("type")
            if jtype in ("revolute", "prismatic", "continuous"):
                axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0", 3)  # type: ignore[union-attr]
                kind = mujoco.mjtJoint.mjJNT_SLIDE if jtype == "prismatic" else mujoco.mjtJoint.mjJNT_HINGE
                jj = cb.add_joint(name=j.get("name"), type=kind, axis=axis)
                lim = j.find("limit")
                if lim is not None and jtype != "continuous":
                    jj.range = [float(lim.get("lower", 0)), float(lim.get("upper", 0))]
                    jj.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                dyn = j.find("dynamics")
                if dyn is not None:
                    damping = np.zeros(len(np.atleast_1d(jj.damping)))
                    damping[0] = float(dyn.get("damping", 0.0))
                    jj.damping = damping if damping.size > 1 else float(damping[0])
                    jj.frictionloss = float(dyn.get("friction", 0.0))
                mim = j.find("mimic")
                if mim is not None:
                    eq = spec.add_equality(
                        type=mujoco.mjtEq.mjEQ_JOINT, name1=j.get("name"), name2=mim.get("joint")
                    )
                    eq.data[:5] = [float(mim.get("offset", 0.0)), float(mim.get("multiplier", 1.0)), 0, 0, 0]
        return body

    add_link(spec.worldbody, root, [0, 0, 0], [1, 0, 0, 0])
    for act in plan.get("actuators", ()):
        a = spec.add_actuator(name=act["name"], target=act["joint"], trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(act["kp"]), kv=float(act.get("kv", 0.0)))
        a.ctrlrange = list(act["ctrlrange"])
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = list(act["forcerange"])
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return spec


def _package_path(uri: str, roots: Mapping[str, str]) -> str:
    if uri.startswith("package://"):
        pkg, _, rest = uri.removeprefix("package://").partition("/")
        if pkg in roots:
            return f"{roots[pkg]}/{rest}"
    return uri


def _vendor_urdf(w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher) -> dict[str, Any]:
    """Convert a URDF subtree to MJCF (:func:`_urdf_spec`, collision geometry)."""
    entry = plan["entry"]
    spec = _urdf_spec(w, plan, fetcher)
    # stage meshes relative to a scratch dir that mirrors upstream paths
    for m in spec.meshes:
        fetcher.path(w.repo, w.commit, m.file)
    staged_root = fetcher.path(w.repo, w.commit, entry).parents[len(Path(entry).parts) - 1]
    spec.meshdir = str(staged_root)
    changes = [
        f"converted from the upstream URDF ({Path(entry).name}) to MJCF: links {', '.join(sorted({b.name for b in spec.bodies if b.name != 'world'}))}",
        *plan.get("conversion_notes", ()),
    ]
    plan2 = {**plan, "mesh_sources": {m.name: m.file for m in spec.meshes}}
    return _finish_converted(w, plan2, spec, entry, fetcher, changes)


def _finish_converted(
    w: _Writer,
    plan: Mapping[str, Any],
    spec: mujoco.MjSpec,
    entry: str,
    fetcher: Fetcher,
    changes: Sequence[str],
) -> dict[str, Any]:
    lic = plan["license"]
    mesh_lic = plan.get("mesh_licenses", {})
    for m in sorted(spec.meshes, key=lambda x: x.name):
        src_rel = plan["mesh_sources"][m.name]
        src_path = fetcher.path(w.repo, w.commit, src_rel)
        hull = _mesh_hull(src_path.resolve(), list(m.scale), src_rel)
        out_rel = f"sim/meshes/{m.name}.obj"
        w.ship(
            out_rel,
            hull,
            role="derived",
            upstream_path=src_rel,
            upstream_sha256=_sha(src_path.read_bytes()),
            license=mesh_lic.get(Path(src_rel).name, lic),
            changes=(
                f"replaced by its convex hull, the shape MuJoCo's collision detector uses (MuJoCo {mujoco.__version__}, qhull)",
                "written as OBJ in the upstream file's frame, scaled to metres",
            ),
        )
        m.file = out_rel.removeprefix("sim/")
        m.scale = [1.0, 1.0, 1.0]
    spec.modelname = w.key
    spec.meshdir = str(w.dir / "sim")
    xml, renames = _neutral_xml(_relative_xml(spec, w))
    if renames:
        changes = [
            *changes,
            f"{len(renames)} element or class names carrying a maker's or product's name renamed",
        ]
    model_rel = f"sim/{w.key}.xml"
    up = fetcher.read(w.repo, w.commit, entry)
    w.ship(
        model_rel,
        xml.encode(),
        role="derived",
        upstream_path=entry,
        upstream_sha256=_sha(up),
        license=lic,
        changes=tuple(changes),
    )
    model = _compile(w.dir / model_rel)
    return {
        "model_file": model_rel,
        "n_bodies": int(model.nbody - 1),
        "n_joints": int(model.njnt),
        "n_actuators": int(model.nu),
        "mass_kg": round(float(model.body_mass[1:].sum()), 6),
        "changes": list(changes),
    }


# --------------------------------------------------------------------------- SDF (box primitives)


def _vendor_sdf_boxes(w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher) -> dict[str, Any]:
    """Convert an SDF model built from boxes and prismatic or revolute joints to MJCF."""
    entry = plan["entry"]
    root_el = ET.fromstring(fetcher.read(w.repo, w.commit, entry).decode("utf-8"))
    model_el = root_el.find("model")
    links = {lk.get("name"): lk for lk in model_el.findall("link")}  # type: ignore[union-attr]
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    root = plan["root"]

    def pose(el: ET.Element) -> tuple[list[float], list[float]]:
        p = el.find("pose")
        vals = _floats(p.text if p is not None else None, 6)
        return vals[:3], _rpy_quat(vals[3:])

    root_pos, _root_quat = pose(links[root])
    inv_root = np.asarray(root_pos)  # links are posed in the model frame; the root defines the effector frame

    def add(parent: mujoco.MjsBody, name: str) -> mujoco.MjsBody:
        lk = links[name]
        pos, quat = pose(lk)
        rel = (np.asarray(pos) - inv_root).tolist() if name != root else [0.0, 0.0, 0.0]
        body = parent.add_body(name=name, pos=rel if parent is not spec.worldbody else [0, 0, 0], quat=quat)
        mass = float(lk.find("inertial/mass").text)  # type: ignore[union-attr,arg-type]
        boxes = []
        for col in lk.findall("collision"):
            size = _floats(col.find("geometry/box/size").text, 3)  # type: ignore[union-attr]
            boxes.append(size)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in size])
        # rotational inertia from the box at the declared mass (upstream values are placeholders)
        sx, sy, sz = boxes[0]
        body.explicitinertial = True
        body.mass = mass
        body.ipos = [0.0, 0.0, 0.0]
        body.inertia = [mass * (sy**2 + sz**2) / 12, mass * (sx**2 + sz**2) / 12, mass * (sx**2 + sy**2) / 12]
        return body

    base = add(spec.worldbody, root)
    for j in model_el.findall("joint"):  # type: ignore[union-attr]
        if j.find("parent").text != root:  # type: ignore[union-attr]
            continue
        child = j.find("child").text  # type: ignore[union-attr]
        cb = add(base, child)  # type: ignore[arg-type]
        axis = _floats(j.find("axis/xyz").text, 3)  # type: ignore[union-attr]
        kind = mujoco.mjtJoint.mjJNT_SLIDE if j.get("type") == "prismatic" else mujoco.mjtJoint.mjJNT_HINGE
        jj = cb.add_joint(name=j.get("name"), type=kind, axis=axis)
        lo = float(j.find("axis/limit/lower").text)  # type: ignore[union-attr,arg-type]
        hi = float(j.find("axis/limit/upper").text)  # type: ignore[union-attr,arg-type]
        jj.range = [lo, hi]
        jj.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    for eq in plan.get("couple", ()):
        e = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=eq["joint1"], name2=eq["joint2"])
        e.data[:5] = [0.0, float(eq["multiplier"]), 0, 0, 0]
    for act in plan.get("actuators", ()):
        a = spec.add_actuator(name=act["name"], target=act["joint"], trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=float(act["kp"]), kv=float(act.get("kv", 0.0)))
        a.ctrlrange = list(act["ctrlrange"])
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = list(act["forcerange"])
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    changes = [
        f"converted from the upstream SDF ({Path(entry).name}) to MJCF: the same boxes, poses, masses and joint limits",
        "rotational inertia computed from each collision box at the declared mass (the upstream inertia values are placeholders)",
        *plan.get("conversion_notes", ()),
    ]
    return _finish_converted(w, {**plan, "mesh_sources": {}}, spec, entry, fetcher, changes)


# --------------------------------------------------------------------------- rigid proxy


def _vendor_proxy(w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher) -> tuple[dict[str, Any], Any]:
    meshes = {}
    sources = {}
    for part in plan["parts"]:
        data = fetcher.read(w.repo, w.commit, part["path"])
        meshes[part["name"]] = load_mesh(
            data, Path(part["path"]).suffix.lstrip(".").lower(), float(part.get("scale", 0.001))
        )
        sources[part["name"]] = (part["path"], _sha(data), part.get("license", plan["license"]))
    proxy = build_rigid_proxy(
        meshes,
        down=plan["down"],
        forward=plan["forward"],
        material=plan.get("material", "PLA"),
        infill=float(plan.get("infill", 1.0)),
        max_parts=int(plan.get("max_parts", 16)),
        offsets=plan.get("offsets"),
        stack=bool(plan.get("stack", False)),
    )
    piece_files: dict[str, list[str]] = {}
    for part in proxy.parts:
        up_path, up_sha, lic = sources[part.name]
        files = []
        for i, piece in enumerate(part.pieces):
            rel = f"sim/pieces/{part.name}_{i}.obj"
            w.ship(
                rel,
                obj_text(piece, f"{part.name} piece {i}"),
                role="derived",
                upstream_path=up_path,
                upstream_sha256=up_sha,
                license=lic,
                changes=(
                    "convex piece of a rigid proxy (CoACD threshold 0.02, seed 0), in the mount frame, metres",
                ),
            )
            files.append(rel.removeprefix("sim/"))
        piece_files[part.name] = files
    xml = proxy_mjcf(proxy, w.key, piece_files, spindle=plan.get("spindle"))
    model_rel = f"sim/{w.key}.xml"
    first = sources[proxy.parts[0].name]
    w.ship(
        model_rel,
        xml.encode(),
        role="derived",
        upstream_path=first[0],
        upstream_sha256=first[1],
        license=plan["license"],
        changes=(
            "a rigid simulation proxy generated by Effector Lab from the part's own mesh: "
            + "; ".join(proxy.notes),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mujoco.MjModel.from_xml_path(str(w.dir / model_rel))
    info = {
        "model_file": model_rel,
        "n_bodies": int(model.nbody - 1),
        "n_joints": int(model.njnt),
        "n_actuators": int(model.nu),
        "mass_kg": round(float(model.body_mass[1:].sum()), 6),
        "pieces": int(sum(len(p.pieces) for p in proxy.parts)),
        "reach_m": round(proxy.reach, 6),
        "changes": list(proxy.notes),
    }
    return info, proxy


# --------------------------------------------------------------------------- generated adapters


def _vendor_adapter(w: _Writer, plan: Mapping[str, Any]) -> dict[str, Any]:
    """One of our generated ISO 9409-1 plates (:mod:`rtt.library.adapters`), our own work."""
    import tempfile

    from rtt.library.adapters import CONFIRMED_ADAPTERS, write_adapter

    spec = next(a for a in CONFIRMED_ADAPTERS if a.key == plan["adapter"])
    licence = (REPO_ROOT / "LICENSE").read_bytes()
    w.ship(
        "LICENSES/LICENSE",
        licence,
        role="licence",
        upstream_path="LICENSE",
        upstream_sha256=_sha(licence),
        license="Apache-2.0",
        copyright="Copyright 2026 Effector Lab contributors",
        generated=True,
    )
    with tempfile.TemporaryDirectory(prefix="rtt-adapter-") as tmp:
        paths = write_adapter(spec, Path(tmp))
        roles = {
            "stl": ("print", f"print/{spec.key}.stl"),
            "xml": ("derived", f"sim/{spec.key}.xml"),
            "json": ("doc", f"doc/{spec.key}.json"),
        }
        for kind, path in sorted(paths.items()):
            role, rel = roles[kind]
            data = Path(path).read_bytes()
            if kind == "xml":  # the MJCF names the STL next to it: point it at the shipped print file
                data = data.replace(
                    f'file="{spec.key}.stl"'.encode(), f'file="../print/{spec.key}.stl"'.encode()
                )
            w.ship(
                rel,
                data,
                role=role,
                upstream_path=f"generated by rtt.library.adapters ({spec.key})",
                upstream_sha256=_sha(data),
                license="Apache-2.0",
                copyright="Copyright 2026 Effector Lab contributors",
                generated=True,
            )
    model = _compile(w.dir / f"sim/{spec.key}.xml")
    return {
        "model_file": f"sim/{spec.key}.xml",
        "n_bodies": int(model.nbody - 1),
        "n_joints": int(model.njnt),
        "n_actuators": int(model.nu),
        "mass_kg": round(float(model.body_mass[1:].sum()), 6),
    }


# --------------------------------------------------------------------------- files and previews


def _ship_files(w: _Writer, specs: Sequence[Mapping[str, Any]], fetcher: Fetcher) -> list[str]:
    out = []
    for f in specs:
        up = f["path"]
        repo, commit = f.get("repository", w.repo), f.get("commit", w.commit)
        size_hint = f.get("size")
        if size_hint is not None and int(size_hint) > MAX_FILE_BYTES:
            w.skipped.append(
                _skip(w, up, int(size_hint), f.get("license", ""), "larger than the per-file cap")
            )
            continue
        data = fetcher.read(repo, commit, up)
        if len(data) > MAX_FILE_BYTES:
            w.skipped.append(_skip(w, up, len(data), f.get("license", ""), "larger than the per-file cap"))
            continue
        out.append(
            w.ship(
                f["out"],
                data,
                role=f.get("role", "print"),
                upstream_path=up,
                upstream_sha256=_sha(data),
                license=f["license"],
                copyright=f.get("copyright"),
                repository=repo,
                commit=commit,
            )
        )
    return out


def _skip(w: _Writer, up: str, size: int, lic: str, why: str) -> dict[str, Any]:
    return {
        "upstream_path": up,
        "size": size,
        "license": lic,
        "reason": f"{why} ({size / 1048576:.1f} MB > {MAX_FILE_BYTES / 1048576:.0f} MB): linked, not mirrored",
        "url": blob_url(w.repo, w.commit, up),
    }


def _preview_meshes(
    w: _Writer, plan: Mapping[str, Any], fetcher: Fetcher, proxy: Any
) -> tuple[list[trimesh.Trimesh], list[str]]:
    kind = plan.get("from")
    if kind == "proxy" and proxy is not None:
        return [p.mesh for p in proxy.parts], [p for p in plan.get("sources", ())]
    if kind == "mjcf":
        entry = plan["entry"]
        up_spec, _ = _stage_mjcf(fetcher, w.repo, w.commit, entry)
        model = _compile(up_spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        root = plan.get("root")
        rid = model.body(root).id if root else 0  # no root: every body of the model
        dropped = [model.body(n).id for n in plan.get("delete_bodies", ())]
        sub = [
            b
            for b in range(1, model.nbody)
            if _is_under(model, b, rid) and not any(_is_under(model, b, d) for d in dropped)
        ]
        for g in range(model.ngeom):  # helper shapes in high groups (bounding cylinders, markers) stay out
            if model.geom_group[g] > 2:
                model.geom_rgba[g, 3] = 0.0
                model.geom_contype[g] = model.geom_conaffinity[g] = 0
        skip = tuple(plan.get("skip_meshes", ()))
        if skip:
            for g in range(model.ngeom):
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mname = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g])) or ""
                    )
                    if any(k in mname.lower() for k in skip):
                        model.geom_rgba[g, 3] = 0.0
                        model.geom_contype[g] = model.geom_conaffinity[g] = 0
        meshes = model_meshes(model, data, sub)
        # into the root body's frame
        inv = np.eye(4)
        inv[:3, :3] = data.xmat[rid].reshape(3, 3).T
        inv[:3, 3] = -inv[:3, :3] @ data.xpos[rid]
        for m in meshes:
            m.apply_transform(inv)
        return meshes, [entry]
    if kind == "shipped":  # files this item ships (a generated plate, for example)
        meshes = []
        for rel in plan["files"]:
            mesh = load_mesh(
                (w.dir / rel).read_bytes(),
                Path(rel).suffix.lstrip(".").lower(),
                float(plan.get("scale", 0.001)),
            )
            meshes.append(mesh)
        return meshes, list(plan["files"])
    if kind == "files":
        meshes = []
        used = []
        for f in plan["files"]:
            data = fetcher.read(f.get("repository", w.repo), f.get("commit", w.commit), f["path"])
            m = load_mesh(data, Path(f["path"]).suffix.lstrip(".").lower(), float(f.get("scale", 0.001)))
            if f.get("rpy"):
                tf = np.eye(4)
                q = _rpy_quat(f["rpy"])
                r = np.zeros(9)
                mujoco.mju_quat2Mat(r, np.asarray(q))
                tf[:3, :3] = r.reshape(3, 3)
                m.apply_transform(tf)
            if f.get("offset"):
                m.apply_translation(np.asarray(f["offset"], np.float64))
            meshes.append(m)
            used.append(f["path"])
        return meshes, used
    if kind == "urdf_visual":  # a converted URDF, drawn from its visual meshes
        spec = _urdf_spec(w, {**plan, "geometry": "visual"}, fetcher)
        for m in spec.meshes:
            m.file = str(fetcher.path(w.repo, w.commit, m.file).resolve())
        spec.meshdir = ""
        model = _compile(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model_meshes(model, data, None, prefer_visual=False), [plan["entry"]]
    if kind == "model":  # a model this script just wrote (converted or generated)
        model = _compile(w.dir / plan["model_file"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model_meshes(model, data, None, prefer_visual=False), [plan.get("source", "")]
    raise ValueError(f"{w.key}: unknown preview source {kind!r}")


def _is_under(model: mujoco.MjModel, b: int, root: int) -> bool:
    if root == 0:
        return True
    while b != 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return root == 0


def _write_preview(
    w: _Writer, plan: Mapping[str, Any] | None, fetcher: Fetcher, proxy: Any
) -> dict[str, Any] | None:
    if not plan:
        return None
    meshes, used = _preview_meshes(w, plan, fetcher, proxy)
    glb = scene_glb(meshes, max_faces=int(plan.get("max_faces", 20_000)))
    if len(glb) > MAX_PREVIEW_BYTES:
        glb = scene_glb(meshes, max_faces=int(plan.get("max_faces", 20_000)) // 3)
    if len(glb) > MAX_PREVIEW_BYTES:
        raise ValueError(f"{w.key}: preview is {len(glb)} bytes, over {MAX_PREVIEW_BYTES}")
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dest = PREVIEW_DIR / f"{w.key}.glb"
    dest.write_bytes(glb)
    return {
        "path": str(dest.relative_to(REPO_ROOT)),
        "sha256": _sha(glb),
        "size": len(glb),
        "license": plan.get("license", w.head["licence"][0]["spdx"]),
        "derived_from": [u for u in used if u],
        "changes": [
            "drawn from the item's own geometry in its reference pose, simplified by vertex clustering",
            "re-encoded as GLB (Y up, metres) in one flat grey, with generic node names",
        ],
    }


# --------------------------------------------------------------------------- one item


def vendor_item(key: str, fetcher: Fetcher, *, lock: Mapping[str, Any] | None = None) -> IngestReport:
    """Vendor one item from its ``SOURCE.json`` head (module docstring)."""
    head_path = LIBRARY_DIR / key / "SOURCE.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    if head.get("format") != SOURCE_FORMAT or head.get("key") != key:
        raise ValueError(f"{key}: not a {SOURCE_FORMAT} head for this key")
    ingest = head.get("ingest", {})
    w = _Writer(key, head)
    notes: list[str] = []
    pinned = (lock or {}).get("items", {}).get(key, {})
    same_pin = pinned.get("upstream", {}).get("commit") == w.commit
    for lf in ingest.get("licence_files", ()):
        repo, commit = lf.get("repository", w.repo), lf.get("commit", w.commit)
        data = fetcher.read(repo, commit, lf["path"])
        sha = _sha(data)
        old = pinned.get("licence_files", {}).get(licence_key(head, repo, lf["path"]))
        if old is not None and old != sha:
            where = "at the same pinned commit" if same_pin else "after the upstream bump"
            raise LicenceDrift(
                f"{key}: {lf['path']} changed {where} ({old[:12]} -> {sha[:12]}); review it before vendoring"
            )
        w.ship(
            f"LICENSES/{lf.get('out', Path(lf['path']).name)}",
            data,
            role="licence",
            upstream_path=lf["path"],
            upstream_sha256=sha,
            license=lf["license"],
            copyright=lf.get("copyright"),
            repository=repo,
            commit=commit,
        )
    sim_plan = ingest.get("sim")
    proxy = None
    sim_info: dict[str, Any] = {}
    if sim_plan:
        t = sim_plan["type"]
        if t == "mjcf":
            sim_info = _vendor_mjcf(w, sim_plan, fetcher)
        elif t == "urdf":
            sim_info = _vendor_urdf(w, sim_plan, fetcher)
        elif t == "sdf_boxes":
            sim_info = _vendor_sdf_boxes(w, sim_plan, fetcher)
        elif t == "rigid_proxy":
            sim_info, proxy = _vendor_proxy(w, sim_plan, fetcher)
        elif t == "adapter":
            sim_info = _vendor_adapter(w, sim_plan)
        elif t == "family":
            sim_info = {"family": sim_plan["family"], "params": dict(sim_plan.get("params", {}))}
        else:
            raise ValueError(f"{key}: unknown sim type {t!r}")
    _ship_files(w, ingest.get("files", ()), fetcher)
    for link in ingest.get("links", ()):
        w.skipped.append(
            {
                "upstream_path": link["path"],
                "size": int(link.get("size", 0)),
                "license": link.get("license", ""),
                "reason": link.get("reason", "linked, not mirrored"),
                "url": blob_url(link.get("repository", w.repo), link.get("commit", w.commit), link["path"]),
            }
        )
    preview_plan = ingest.get("preview")
    if preview_plan and preview_plan.get("from") == "model":
        preview_plan = {**preview_plan, "model_file": sim_info["model_file"]}
    preview = _write_preview(w, preview_plan, fetcher, proxy)
    w.prune()
    # the head keeps its hand-written fields; the script owns the rest
    out = {k: v for k, v in head.items() if k not in ("files", "skipped", "preview", "built")}
    sim = dict(out.get("sim", {}))
    if sim_info:
        sim.update({k: v for k, v in sim_info.items() if k not in ("changes",)})
    out["sim"] = sim
    out["files"] = [f.to_dict() for f in sorted(w.files, key=lambda f: f.path)]
    out["skipped"] = sorted(w.skipped, key=lambda s: s["upstream_path"])
    out["preview"] = preview
    head_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    read_item.cache_clear()
    total = sum(f.size for f in w.files)
    for f in w.files:
        if f.rating is not Rating.GREEN:
            notes.append(f"{f.path} is {f.rating.value}")
    return IngestReport(key, len(w.files), total, len(w.skipped), preview["size"] if preview else 0, notes)
