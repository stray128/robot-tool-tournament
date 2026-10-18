"""Build packs (track S5, product mode): the files that build a tested Studio setup.

:func:`export_build_pack` (the plain function behind the agent tool of the same name) writes, for
one finalist of a tournament (the recommendation by default):

* generated printable parts: ``bracket.stl`` (a multi-tool head) or ``changer_plate_tool<k>.stl``
  (a tool change), each one watertight closed solid (:mod:`.geometry`); STL only (STEP is not
  provided: no CAD kernel is installed);
* ``mjcf/setup_tool<k>.xml`` and ``mjcf/meshes/``: the exact Studio scene of every tool
  configuration, reloaded and checked against the tested model (:mod:`.mjcf`);
* ``cad/tool<k>_reference.stl``: the simulated collision geometry of a procedural tool (reference
  only); a vendored tool gets an upstream link instead, plus its licence text under ``licences/``;
* ``bom.csv``, ``ASSEMBLY.md``, ``README.txt``, ``certificate.html`` (self-contained, per-step
  success, conditions, fidelity tiers and caveats, assumed constants) and ``CREDITS.txt``;
* ``manifest.json``: the :class:`~rtt.studio.types.BuildPackManifest` with the sha256, kind,
  format, SPDX licence, source and attribution of every file; amber or red items are never
  included (a setup with one is refused);
* ``<folder>.zip`` beside the folder: the same files, deterministic (sorted, fixed timestamps).

Every file with text carries "Product mode: not part of the pre-registered experiment".
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from rtt.effectors.types import ALLOWED_LICENSES, AssetSource, find_brand_tokens
from rtt.sim.task import TaskV2, default_task_v2
from rtt.studio.buildpack.documents import (
    assembly_notes,
    bom_csv,
    bom_lines,
    certificate_html,
    credits_text,
    readme_text,
)
from rtt.studio.buildpack.geometry import printable_parts, tool_reference_mesh, write_binary_stl
from rtt.studio.buildpack.mjcf import export_mjcf, tool_configurations
from rtt.studio.types import (
    PRESENT_S,
    BracketSpec,
    BuildPackFile,
    BuildPackLink,
    BuildPackManifest,
    HeadMount,
    Setup,
    SetupKind,
    TaskSpec,
    ToolChanger,
    task_spec_from_tool_args,
)

__all__ = ["GENERATED_LICENCE", "BuildPack", "export_build_pack", "setup_from_dict", "task_from_dict"]

#: Licence of every file the Studio generates (as the print pack).
GENERATED_LICENCE: Final = "CC0-1.0"
#: From the most to the least demanding: a file combining several sources carries the first.
_LICENCE_ORDER: Final = ("Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MIT", "CC-BY-4.0", "CC0-1.0")
_ZIP_TIME: Final = (2026, 1, 1, 0, 0, 0)


@dataclass
class BuildPack:
    """What :func:`export_build_pack` wrote."""

    out_dir: Path
    manifest: BuildPackManifest
    zip_path: Path | None
    setup_id: str
    recommended: bool
    notes: list[str] = field(default_factory=list)
    mjcf: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "zip": str(self.zip_path) if self.zip_path else None,
            "setup_id": self.setup_id,
            "recommended": self.recommended,
            "manifest": self.manifest.to_dict(),
            "notes": list(self.notes),
            "mjcf": list(self.mjcf),
        }


# --------------------------------------------------------------------------- rebuilding from JSON


def setup_from_dict(d: Mapping[str, Any]) -> Setup:
    """A :class:`Setup` back from :meth:`Setup.to_dict` (its id must match)."""
    from rtt.effectors.tournament import spec_from_dict

    setup = Setup(
        SetupKind(d["kind"]),
        tuple(spec_from_dict(t) for t in d["tools"]),
        dict(d["assignment"]),
        mounts=tuple(HeadMount(**m) for m in d.get("mounts") or ()),
        bracket=BracketSpec(**d["bracket"]) if d.get("bracket") else None,
        changer=ToolChanger(**d["changer"]) if d.get("changer") else None,
        present_s=float(d.get("present_s") or PRESENT_S),
        label=d.get("label") or "",
    )
    if "setup_id" in d and setup.setup_id != d["setup_id"]:
        raise ValueError("the setup does not hash to its recorded setup_id")
    return setup


def task_from_dict(d: Mapping[str, Any]) -> TaskSpec:
    """A :class:`TaskSpec` back from :meth:`TaskSpec.to_dict` (its id must match)."""
    keys = (
        "title",
        "description",
        "objects",
        "steps",
        "assumptions",
        "clarifications",
        "unsupported",
        "refusal",
    )
    spec = task_spec_from_tool_args({k: d[k] for k in keys if k in d}, source=d.get("source") or "manual")
    if "task_id" in d and spec.task_id != d["task_id"]:
        raise ValueError("the task does not hash to its recorded task_id")
    return spec


def _report_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    if isinstance(report, str | Path):
        return json.loads(Path(report).read_text(encoding="utf-8"))
    if hasattr(report, "to_dict"):
        return report.to_dict()
    raise TypeError("report must be a TournamentReport, its to_dict(), or a path to its JSON")


# --------------------------------------------------------------------------- helpers


def _repo_root() -> Path:
    from rtt.effectors.scene import REPO_ROOT

    return REPO_ROOT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strongest(licences: set[str]) -> str:
    for lic in _LICENCE_ORDER:
        if lic in licences:
            return lic
    return GENERATED_LICENCE


def _attribution(src: AssetSource) -> str:
    """A brand-free attribution for the manifest (the full credit line is in CREDITS.txt)."""
    text = f"{src.license}; {src.copyright}"
    if find_brand_tokens(text) or len(text) > 250:
        return f"{src.license}; credit line and copyright in CREDITS.txt"
    return f"{text}; credit line in CREDITS.txt"


def _write_zip(root: Path, dest: Path) -> Path:
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
            info = zipfile.ZipInfo(f"{root.name}/{path.relative_to(root).as_posix()}", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    tmp.replace(dest)
    return dest


def _pick_finalist(report: Mapping[str, Any], setup_id: str | None) -> tuple[Mapping[str, Any], bool]:
    rec = report.get("recommendation") or {}
    rec_id = (rec.get("recommendation") or {}).get("setup_id")
    want = setup_id or rec_id
    if want is None:
        reason = rec.get("reason") or (rec.get("no_winner") or {}).get("reason") or "no recommendation"
        raise ValueError(f"there is no recommendation to pack ({reason}); pass setup_id to pack a finalist")
    hits = [f for f in report["finalists"] if f["setup"]["setup_id"].startswith(want)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} finalists match setup id {want!r}")
    return hits[0], hits[0]["setup"]["setup_id"] == rec_id


# --------------------------------------------------------------------------- the pack


def export_build_pack(
    report: Any,
    out_dir: str | Path,
    *,
    setup_id: str | None = None,
    zip_pack: bool = True,
    overwrite: bool = False,
    task: TaskV2 | None = None,
) -> BuildPack:
    """Write the build pack of one finalist of ``report`` (module docstring) into ``out_dir``.

    ``report``: a :class:`~rtt.studio.search.TournamentReport`, its ``to_dict()`` or the path of
    that JSON. ``setup_id`` (or a unique prefix): which finalist; default the recommendation (a
    report without one needs it). ``out_dir`` must be new or empty unless ``overwrite`` (then a
    previous pack there, recognised by its ``manifest.json``, is replaced).
    """
    from rtt.studio.search import probe_scene

    data = _report_dict(report)
    finalist, recommended = _pick_finalist(data, setup_id)
    setup = setup_from_dict(finalist["setup"])
    spec = task_from_dict(data["task"])
    task = task or default_task_v2()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not (overwrite and (out / "manifest.json").is_file()):
            raise FileExistsError(f"{out} is not empty (pass overwrite=True to replace a previous pack)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    files: list[BuildPackFile] = []
    links: list[BuildPackLink] = []
    notes: list[str] = []
    credits: list[dict[str, Any]] = []

    def add(rel: str, kind: str, fmt: str, lic: str, source: str, ref: str, **kw: Any) -> None:
        files.append(BuildPackFile(rel, _sha(out / rel), kind, fmt, lic, source, ref, **kw))  # type: ignore[arg-type]

    # 1. generated printable parts
    printed = []
    for part in printable_parts(setup):
        rel = f"{part.name}.stl"
        write_binary_stl(part.mesh, out / rel)
        add(
            rel,
            part.kind,
            "stl",
            GENERATED_LICENCE,
            "generated",
            "rtt.studio.buildpack.geometry",
            watertight=True,
        )
        printed.append(
            {
                **part.facts(),
                "file": rel,
                "description": part.description,
                "density_kg_m3": part.density_kg_m3,
            }
        )
    # 2. the tested scenes, one per tool configuration
    mjcf_info = []
    for k in tool_configurations(setup):
        exp = export_mjcf(setup, spec, k, out / "mjcf", task=task, pack_root=out)
        mjcf_info.append(
            {"tool": k, "file": exp.rel, "verified": exp.verified, "dropped_textures": exp.dropped_textures}
        )
        lics = {m.source.license for m in exp.meshes}
        add(exp.rel, "mjcf", "xml", _strongest(lics | {GENERATED_LICENCE}), "studio", "rtt.studio.scene")
        for m in exp.meshes:
            if any(f.path == m.rel for f in files):
                continue
            add(
                m.rel,
                "mjcf",
                "stl",
                m.source.license,
                "library",
                m.source.path,
                attribution=_attribution(m.source),
                watertight=m.watertight,
            )
            if m.converted:
                notes.append(f"{m.rel}: converted from OBJ to binary STL (geometry only)")
    # 3. every tool: an upstream link and its licence text (vendored) or reference geometry (procedural)
    upstream: dict[str, dict[str, str]] = {}
    for k, tool in enumerate(setup.tools):
        scene, why = probe_scene(tool, task)
        if scene is None:
            raise ValueError(f"tool {k} cannot be built: {why}")
        srcs = list(scene.active.build.sources)
        if any(s.status == "amber" for s in srcs):
            raise ValueError(f"tool {k} has an amber licence record: it is never shipped")
        credits += [s.to_dict() for s in srcs]
        if srcs:
            model_src = next((s for s in srcs if s.path.endswith(".xml")), srcs[0])
            upstream[tool.effector_id] = {"url": model_src.upstream_url, "license": model_src.license}
            links.append(
                BuildPackLink(
                    name=f"Tool {k}: {tool.display_name}"[:60],
                    url=model_src.upstream_url,
                    license=model_src.license,
                    reason="the tool's own CAD and documentation are upstream; the pack carries its simulation meshes",
                )
            )
            lic_src = next((s for s in srcs if s.path.rsplit("/", 1)[-1].upper().startswith("LICENSE")), None)
            if lic_src is not None:
                rel = f"licences/tool{k}_LICENSE.txt"
                (out / "licences").mkdir(exist_ok=True)
                shutil.copyfile(_repo_root() / lic_src.path, out / rel)
                add(
                    rel,
                    "licence",
                    "txt",
                    lic_src.license,
                    "library",
                    lic_src.path,
                    attribution=_attribution(lic_src),
                )
        else:
            rel = f"cad/tool{k}_reference.stl"
            (out / "cad").mkdir(exist_ok=True)
            mesh = tool_reference_mesh(scene)
            write_binary_stl(
                mesh, out / rel, header="rtt studio: simulated tool geometry, reference only, mm"
            )
            add(
                rel,
                "effector_cad",
                "stl",
                GENERATED_LICENCE,
                "generated",
                f"rtt.effectors.{tool.family.value}",
                watertight=bool(mesh.is_watertight),
            )
            notes.append(f"{rel}: the simulated collision geometry of tool {k} (reference, not a drawing)")
    # 4. documents
    bom = bom_lines(finalist["setup"], data["screen"]["tools"], printed, upstream)
    (out / "bom.csv").write_text(bom_csv(bom), encoding="utf-8")
    add("bom.csv", "bom", "csv", GENERATED_LICENCE, "studio", "rtt.studio.buildpack.documents")
    used = {
        c["step_id"]: c["primitive_used"]
        for c in data["screen"]["cells"]
        if c["step_id"] in setup.assignment
        and c["effector_id"] == setup.effector_for(c["step_id"]).effector_id
    }
    (out / "certificate.html").write_text(
        certificate_html(finalist, data, used=used, printed=printed), encoding="utf-8"
    )
    add(
        "certificate.html",
        "certificate",
        "html",
        GENERATED_LICENCE,
        "studio",
        "rtt.studio.buildpack.documents",
    )
    (out / "CREDITS.txt").write_text(credits_text(credits), encoding="utf-8")
    add("CREDITS.txt", "licence", "txt", GENERATED_LICENCE, "studio", "rtt.studio.buildpack.documents")
    listed = sorted([f.path for f in files] + ["ASSEMBLY.md", "README.txt", "manifest.json"])
    (out / "ASSEMBLY.md").write_text(assembly_notes(finalist, data, listed), encoding="utf-8")
    add("ASSEMBLY.md", "notes", "md", GENERATED_LICENCE, "studio", "rtt.studio.buildpack.documents")
    (out / "README.txt").write_text(readme_text(data, finalist, listed), encoding="utf-8")
    add("README.txt", "notes", "txt", GENERATED_LICENCE, "studio", "rtt.studio.buildpack.documents")
    # 5. the manifest and the zip
    caveats = list(dict.fromkeys([*finalist["result"].get("caveats", []), *data.get("fixed_caveats", [])]))
    manifest = BuildPackManifest(
        task_id=spec.task_id,
        setup_id=setup.setup_id,
        title=spec.title,
        files=tuple(sorted(files, key=lambda f: f.path)),
        bom=tuple(bom),
        links=tuple(links),
        caveats=tuple(c[:300] for c in caveats[:20]),
    )
    bad = [f.path for f in manifest.files if f.license not in ALLOWED_LICENSES]
    if bad:
        raise ValueError(f"files with a licence that may not be redistributed: {bad}")
    (out / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = _write_zip(out, out.with_name(out.name + ".zip")) if zip_pack else None
    return BuildPack(out, manifest, zip_path, setup.setup_id, recommended, notes, mjcf_info)
