"""Full-setup MJCF export of a build pack: the exact Studio scene a setup was tested in.

For each tool configuration of the setup (the one tool of a ``single``; every tool of a
``tool_change``, each on the changer; every presented head of a ``multi_head``) the Studio scene is
built with :func:`rtt.studio.scene.build_studio_scene` (wrist with its continuous roll, the tool
stack, the task's objects), its authored ``MjSpec`` is captured, and written with
``MjSpec.to_xml``. Mesh files are copied into ``meshes/`` next to the XML and referenced
relatively, so the folder loads anywhere: binary STL is copied byte for byte; OBJ (which the
build pack format does not carry) is converted to binary STL, geometry only. Texture files, if a
model had any, are dropped (visual only). The export is then reloaded with MuJoCo from the pack
and compared with the scene it came from (bodies, geoms, joints, actuators, total mass).

Every copied mesh is matched to the licence record (``AssetSource``) of the tool it belongs to;
a mesh without a record, or one whose record is amber, stops the export (amber items are never
included). Procedural tools (no vendored files) are all primitives.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from rtt.effectors.types import AssetSource
from rtt.sim.task import TaskV2, default_task_v2
from rtt.studio.buildpack.geometry import write_binary_stl
from rtt.studio.scene import build_studio_scene, idle_prefix
from rtt.studio.types import Setup, SetupKind, TaskSpec

__all__ = ["MeshFile", "MjcfExport", "export_mjcf", "tool_configurations"]


@dataclass(frozen=True)
class MeshFile:
    """One mesh copied into the pack: where it went, where it came from, its licence record."""

    path: Path  # in the pack
    rel: str  # relative to the pack root
    tool: int
    source: AssetSource
    converted: bool
    watertight: bool


@dataclass
class MjcfExport:
    """One exported scene and the meshes it references."""

    tool: int
    path: Path
    rel: str
    meshes: list[MeshFile] = field(default_factory=list)
    verified: dict[str, Any] = field(default_factory=dict)
    dropped_textures: list[str] = field(default_factory=list)
    sources: list[AssetSource] = field(default_factory=list)


def tool_configurations(setup: Setup) -> list[int]:
    """The active tool of each exported scene: every tool (one per segment configuration)."""
    return list(range(len(setup.tools)))


def _owner(mesh_name: str, setup: Setup, tool: int) -> int:
    if setup.kind is SetupKind.MULTI_HEAD:
        for k in range(len(setup.tools)):
            if k != tool and mesh_name.startswith(idle_prefix(k)):
                return k
    return tool


def _match(file: str, sources: Sequence[AssetSource]) -> AssetSource:
    base = Path(file).name
    hits = [s for s in sources if s.path.endswith("/" + file) or s.path == file]
    if not hits:
        hits = [s for s in sources if Path(s.path).name == base]
    if len(hits) != 1:
        raise ValueError(
            f"mesh file {base!r} matches {len(hits)} licence records; every shipped mesh needs exactly one"
        )
    return hits[0]


#: Compiled arrays the reloaded export must match (OBJ meshes become float32 STL: a tolerance).
_FIELDS = (
    "body_mass",
    "body_pos",
    "body_quat",
    "geom_size",
    "geom_pos",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_contype",
    "geom_conaffinity",
    "geom_priority",
    "jnt_range",
    "dof_damping",
    "dof_armature",
    "dof_frictionloss",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrlrange",
)


def _same_fields(a: mujoco.MjModel, b: mujoco.MjModel) -> dict[str, bool]:
    out = {}
    for f in _FIELDS:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        out[f] = bool(x.shape == y.shape and np.allclose(x, y, rtol=1e-5, atol=1e-6))
    return out


def _flatten_classes(spec: mujoco.MjSpec) -> None:
    """Point every element at the main default class (and drop body child classes).

    Element attributes already hold their resolved values, so the compiled model is unchanged
    (the reload check proves it); writing attached models' default-class references makes
    ``MjSpec.to_xml`` crash on some vendored hands (MuJoCo 3.x), so the export does not keep them.
    """
    main = spec.default
    for kind in ("bodies", "geoms", "joints", "sites", "actuators", "tendons", "meshes", "materials"):
        for e in getattr(spec, kind, []):
            if hasattr(e, "classname"):
                e.classname = main
            if kind == "bodies":
                e.childclass = ""


_UNNAMED = re.compile(r'^<default(?:\s+class="")?\s*(/?)>$')


def _name_unnamed_classes(xml: str) -> str:
    """Name the nested ``<default>`` classes a head attached without a prefix brings along (its
    scratch spec's main class, written without a name, which MuJoCo then refuses to load). Every
    element points at the main class after :func:`_flatten_classes`, so the names are unused."""
    out, n, seen_top = [], 0, False
    for line in xml.splitlines(keepends=True):
        m = _UNNAMED.match(line.strip())
        if m:
            if seen_top:
                n += 1
                indent = line[: len(line) - len(line.lstrip())]
                line = f'{indent}<default class="studio_attached_{n}"{m.group(1)}>\n'
            seen_top = True
        out.append(line)
    return "".join(out)


def _summary(model: mujoco.MjModel) -> dict[str, Any]:
    return {
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "nmesh": int(model.nmesh),
        "mass_kg": round(float(np.sum(model.body_mass)), 6),
    }


def export_mjcf(
    setup: Setup,
    task_spec: TaskSpec,
    tool: int,
    out_dir: Path,
    *,
    task: TaskV2 | None = None,
    repo_root: Path | None = None,
    pack_root: Path | None = None,
) -> MjcfExport:
    """Write ``setup_tool<k>.xml`` (and its meshes) for ``tool`` active into ``out_dir``."""
    from rtt.effectors.scene import REPO_ROOT

    task = task or default_task_v2()
    repo_root = repo_root or REPO_ROOT
    pack_root = pack_root or out_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    captured: list[mujoco.MjSpec] = []
    scene = build_studio_scene(setup, task_spec, tool, task=task, edit=captured.append)
    spec = captured[0]
    sources = {h.tool: tuple(h.build.sources) for h in scene.heads}
    if any(s.status == "amber" for srcs in sources.values() for s in srcs):
        raise ValueError("a tool of this setup has an amber licence record: its files are never shipped")
    xml_path = out_dir / f"setup_tool{tool}.xml"
    export = MjcfExport(tool=tool, path=xml_path, rel=os.path.relpath(xml_path, pack_root))
    used: set[str] = set()
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        owner = _owner(mesh.name, setup, tool)
        src = _match(mesh.file, sources.get(owner, ()))
        origin = repo_root / src.path
        if not origin.is_file():
            raise FileNotFoundError(f"vendored mesh {src.path} is missing")
        name = f"tool{owner}_{src.sha256[:12]}.stl"  # content-addressed: shared by every scene
        mesh_dir.mkdir(exist_ok=True)
        dest = mesh_dir / name
        converted = origin.suffix.lower() != ".stl"
        if converted:
            tm = trimesh.load(str(origin), force="mesh", process=False)
            write_binary_stl(tm, dest, header="rtt studio build pack, converted from OBJ")
        else:
            shutil.copyfile(origin, dest)
            tm = trimesh.load(str(dest), force="mesh", process=False)
        watertight = bool(trimesh.Trimesh(tm.vertices, tm.faces, process=True).is_watertight)
        mesh.file = str(dest.resolve())
        used.add(src.path)
        export.meshes.append(
            MeshFile(dest, os.path.relpath(dest, pack_root), owner, src, converted, watertight)
        )
    for tex in list(spec.textures):
        if tex.file:
            export.dropped_textures.append(tex.name)
            for mat in spec.materials:
                mat.textures = ["" if t == tex.name else t for t in mat.textures]
            spec.delete(tex)
    spec.meshdir = ""
    spec.texturedir = ""
    _flatten_classes(spec)
    xml = spec.to_xml()
    xml = _name_unnamed_classes(xml.replace(str(mesh_dir.resolve()) + os.sep, "meshes/"))
    xml_path.write_text(xml, encoding="utf-8")
    reloaded = mujoco.MjModel.from_xml_path(str(xml_path))
    want, got = _summary(scene.model), _summary(reloaded)
    mass_ok = math.isclose(want["mass_kg"], got["mass_kg"], rel_tol=1e-5, abs_tol=1e-9)
    fields = _same_fields(scene.model, reloaded)
    same = all(want[k] == got[k] for k in want if k != "mass_kg") and mass_ok and all(fields.values())
    export.verified = {"same_as_tested": same, "tested": want, "exported": got, "fields": fields}
    if not same:
        raise ValueError(f"the exported MJCF does not reload as the tested scene: {want} != {got}")
    export.sources = [s for srcs in sources.values() for s in srcs]
    return export


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
