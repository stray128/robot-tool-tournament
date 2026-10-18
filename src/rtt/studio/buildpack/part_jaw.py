"""Printed jaws in a build pack (product mode): the part-shaped fingers of a setup's parallel jaws.

**Product mode, not part of the pre-registered experiment.** A setup tool that is a Studio
part-shaped jaw (:mod:`rtt.effectors.part_jaw`) is our own generated geometry, so its jaws can be
printed. :func:`add_jaw_prints` adds them to a pack that :func:`rtt.studio.buildpack.export_build_pack`
wrote, through the v2 print pack (:func:`rtt.part.pack.make_print_pack_v2`, the same solids and
checks): for tool ``k``

* ``jaws/tool<k>_left.stl``, ``jaws/tool<k>_right.stl``: each jaw upright for printing (root plane on
  the bed, tip up), one watertight solid; the right jaw is its own part;
* ``jaws/tool<k>_mount_left.stl``, ``jaws/tool<k>_mount_right.stl``: each jaw fused with its carriage
  block for the gripper's guide rods, one watertight solid;
* ``jaws/JAWS.txt``: what each file is, the object each jaw pair is shaped for, the jaw force,
  and the label.

Every file is CC0-1.0 (generated). The stock blade is the open-source gripper's own part (CC BY
4.0): it is never printed here (its tool carries the upstream link and licence text the pack
already writes for a vendored tool). Its hull geometry is written inline into the exported MJCF of
every scene that holds it, which the pack would otherwise label CC0-1.0 (it reads licences from
mesh files only), so those MJCF entries carry the blade's licence and a brand-free attribution.
The manifest is rewritten and the zip rebuilt. :func:`export_part_jaw_pack` does both steps.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

from rtt.effectors.part_jaw import is_part_jaw, jaw_print_files, jaw_target
from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.studio.buildpack import (
    GENERATED_LICENCE,
    BuildPack,
    _attribution,
    _pick_finalist,
    _report_dict,
    _sha,
    _strongest,
    _write_zip,
    export_build_pack,
    setup_from_dict,
)
from rtt.studio.types import BuildPackFile, Setup, SetupKind

__all__ = ["JAW_DIR", "add_jaw_prints", "export_part_jaw_pack"]

#: Where a pack keeps the printed jaws.
JAW_DIR: Final = "jaws"
_SOURCE_REF: Final = "rtt.part.pack.make_print_pack_v2"


def add_jaw_prints(pack: BuildPack, setup: Setup, *, zip_pack: bool = True, task: Any = None) -> BuildPack:
    """Add the printed jaws of every part-shaped jaw tool of ``setup`` to ``pack`` (module
    docstring); returns the updated pack (the same pack when no tool has printable jaws)."""
    if setup.setup_id != pack.setup_id:
        raise ValueError("the setup is not the pack's")
    out = pack.out_dir
    files = list(pack.manifest.files)
    notes = list(pack.notes)
    lines = [
        PRODUCT_MODE_LABEL,
        "",
        "Printed jaws of the setup's part-shaped parallel jaws (Studio builds of the v2 fingers on the",
        "open 84 mm parallel gripper; not the pre-registered result). Generated geometry, CC0-1.0.",
        "Simulated only: check the fit on the real object before relying on it.",
        "",
    ]
    wrote = changed = False
    for k, tool in enumerate(setup.tools):
        if not is_part_jaw(tool):
            continue
        t = jaw_target(tool)
        if t["entrant"] == "stock":
            files = _stock_mjcf_licence(files, setup, k, tool, task)
            changed = True
            notes.append(f"tool {k} uses the gripper's own stock blades: nothing to print (upstream link)")
            continue
        facts = jaw_print_files(tool, out / JAW_DIR, prefix=f"tool{k}_", task=task)
        lines.append(f"Tool {k}: {tool.display_name}, shaped for {t['object']} ({t['kind']} {t['handle']})")
        for name in sorted(facts):
            f = facts[name]
            if not f["watertight"]:
                raise ValueError(f"{name} is not one watertight solid")
            rel = f"{JAW_DIR}/{name}"
            files.append(
                BuildPackFile(
                    rel,
                    f["sha256"],
                    "effector_cad",
                    "stl",
                    GENERATED_LICENCE,
                    "generated",
                    _SOURCE_REF,
                    watertight=True,
                )
            )
            lines.append(f"  {rel}: {f['what']} ({f['volume_mm3']:.0f} mm^3)")
        lines.append(
            "  print two-off as listed (left and right are different parts); PETG-CF as simulated (mu 0.3)"
        )
        lines.append(
            "  jaw force 25 N per jaw (the gripper's protected servo, sustained), stroke 42 mm per jaw"
        )
        lines.append("")
        wrote = changed = True
    if not changed:
        return replace(pack, notes=notes)
    if wrote:
        (out / JAW_DIR / "JAWS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rel = f"{JAW_DIR}/JAWS.txt"
        files.append(
            BuildPackFile(
                rel,
                _sha(out / rel),
                "notes",
                "txt",
                GENERATED_LICENCE,
                "studio",
                "rtt.studio.buildpack.part_jaw",
            )
        )
        notes.append(
            f"{JAW_DIR}/: the printed jaws of the setup's part-shaped parallel jaws (v2 print pack solids)"
        )
    manifest = replace(pack.manifest, files=tuple(sorted(files, key=lambda f: f.path)))
    (out / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = _write_zip(out, out.with_name(out.name + ".zip")) if zip_pack else pack.zip_path
    return replace(pack, manifest=manifest, zip_path=zip_path, notes=notes)


def _stock_mjcf_licence(
    files: list[BuildPackFile], setup: Setup, k: int, tool: Any, task: Any
) -> list[BuildPackFile]:
    """The MJCF entries of the scenes that hold the stock-blade tool ``k`` (every scene but the
    other tools' scenes of a tool change), relabelled with the blade's licence (module docstring)."""
    from rtt.studio.search import probe_scene

    scene, why = probe_scene(tool, task)
    if scene is None:
        raise ValueError(f"tool {k} cannot be built: {why}")
    src = next(s for s in scene.active.build.sources if s.path.endswith(".npz"))
    holds = {f"mjcf/setup_tool{j}.xml" for j in range(len(setup.tools))}
    if setup.kind is SetupKind.TOOL_CHANGE:
        holds = {f"mjcf/setup_tool{k}.xml"}
    out = []
    for f in files:
        if f.path in holds and f.kind == "mjcf":
            lic = _strongest({f.license, src.license})
            f = replace(f, license=lic, attribution=_attribution(src))
        out.append(f)
    return out


def export_part_jaw_pack(report: Any, out_dir: str | Path, **kwargs: Any) -> BuildPack:
    """:func:`~rtt.studio.buildpack.export_build_pack`, then :func:`add_jaw_prints` for the packed
    finalist (same arguments)."""
    zip_pack = kwargs.get("zip_pack", True)
    pack = export_build_pack(report, out_dir, **kwargs)
    data = _report_dict(report)
    finalist, _ = _pick_finalist(data, pack.setup_id)
    return add_jaw_prints(
        pack, setup_from_dict(finalist["setup"]), zip_pack=zip_pack, task=kwargs.get("task")
    )
