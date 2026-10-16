"""Print pack: everything needed to print and judge a part run's fingers, in one zip.

:func:`make_print_pack` reads a part run directory (see :mod:`rtt.part.search`) and
writes ``<run_dir>/print_pack/`` plus ``<run_dir>/print_pack.zip`` containing:

* ``finger_left.stl``, ``finger_right.stl``: the bare fingers, millimetres, in the
  gripper's assembly frame (root on ``z = 0``, hanging toward -Z, contact face on
  ``y = 0``); the right one is the mirror image.
* ``jaw.stl``: carriage block for the guide rods fused with the finger
  (:func:`rtt.part.mount.build_jaw`), millimetres, turned for printing (contact face
  on the bed, rod bores vertical). Print two; the second jaw is the same part turned
  180 degrees.
* ``design.json``: the frozen design record (with its SHA-256 and freeze time).
* ``certificate.json`` and ``certificate.html`` (self-contained).
* ``manifest.json``: every file with its size and SHA-256, units, the part and
  certificate summaries, the clamp interface with each dimension's source, provenance
  and the disclaimer.
* ``README.txt``: print and assembly notes.

The zip is deterministic (sorted names, fixed timestamps), so the same run always
gives the same bytes apart from ``created_at`` fields.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.design.export import export_stl
from rtt.design.generate import generate_finger
from rtt.design.schema import FingerDesign
from rtt.part.certificate import DISCLAIMER, render_certificate_html
from rtt.part.intake import PartSpec, load_part
from rtt.part.mount import ClampInterface, build_jaw, jaw_print_orientation
from rtt.part.preview import PART_COLOR, ROD_COLOR, jaw_pair_scene, posed_part, render_png
from rtt.search.select import FrozenDesign
from rtt.sim.task import default_task
from rtt.store.manifest import write_json_atomic

__all__ = ["PACK_FILES", "collect_run", "make_print_pack"]

PACK_FILES: Final = (
    "README.txt",
    "certificate.html",
    "certificate.json",
    "design.json",
    "finger_left.stl",
    "finger_right.stl",
    "jaw.stl",
    "manifest.json",
)
_ZIP_TIME: Final = (1980, 1, 1, 0, 0, 0)
_SAFE_ARM: Final = re.compile(r"[a-z0-9_]{1,32}")

_README: Final = """Print pack: fingers for {part}
==============================================

{disclaimer}

Files
-----
finger_left.stl, finger_right.stl
    The bare fingers in millimetres, in the gripper's assembly frame: the root is on
    the frame's front face (z = 0) and the finger hangs toward -Z; the contact face
    is the plane y = 0. Use these if you attach fingers to your own jaws.
jaw.stl  (print 2)
    Carriage block for the two guide rods (bores {bore} mm at {spacing} mm centres),
    fused with the finger, in millimetres and already turned for printing: contact
    face on the bed, rod bores vertical.{support}
    The second jaw is the same part turned 180 degrees about the gripper's vertical axis.
    Drive: {drive}
certificate.html
    The simulated stress-test results; open it in any browser (no internet needed).
design.json, certificate.json, manifest.json
    The frozen design record, the raw test results and checksums of every file.

Printing
--------
PLA or PETG, 0.2 mm layers, 4 perimeters, 40 % infill or more. Ream the rod bores
to size if they print tight. The simulation assumed solid PLA fingers ({mass} g each)
and a friction coefficient of {mu} between finger and part.

Check before printing
---------------------
Dimensions marked "assumed" in manifest.json (block height, rod axis height and,
if enabled, the rack) are not documented; compare them with a stock jaw first.
The interface dimensions were measured from an open-source parallel gripper's
published hardware sources (CERN-OHL-P-2.0, attribution in the project README).
This model is new; it contains no files from that project.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_records(run_dir: Path) -> list[dict[str, Any]]:
    files = sorted((run_dir / "search" / "frozen").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{run_dir}: no frozen design under search/frozen/; run the search first")
    from rtt.search.select import read_frozen

    records = []
    for f in files:
        frozen: FrozenDesign = read_frozen(f)  # verifies the content hash
        records.append(frozen.to_json())
    records.sort(
        key=lambda r: (-r["training_success"], -r["training_hold_fraction"], r["mass_g"], r["run_index"])
    )
    return records


def _proposal_facts(run_dir: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    """The frozen design's proposal (hypothesis, origin) and the model's analysis from that call."""
    arm = str(frozen.get("arm", ""))
    if not _SAFE_ARM.fullmatch(arm):  # the arm names a directory below; keep it inside the run
        return {}
    run = run_dir / "search" / "runs" / f"{arm}-r{int(frozen['run_index']):02d}"
    facts: dict[str, Any] = {}
    history = run / "history.jsonl"
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("label") == frozen["label"]:
                prop = item["proposal"]
                facts = {
                    "hypothesis": prop.get("hypothesis"),
                    "origin": prop.get("origin"),
                    "operator": prop.get("operator"),
                    "parents": prop.get("parents"),
                    "request_hash": prop.get("request_hash"),
                    "round": item.get("round"),
                    "per_condition": item["score"].get("per_condition"),
                }
                break
    calls = run / "llm_calls.jsonl"
    if facts.get("request_hash") and calls.exists():
        for line in calls.read_text(encoding="utf-8").splitlines():
            call = json.loads(line)
            if call.get("request_hash") == facts["request_hash"] and call.get("analysis"):
                facts["analysis"] = call["analysis"]
                break
    return facts


def collect_run(run_dir: str | Path) -> dict[str, Any]:
    """Every fact the certificate and the manifest need, read from a part run directory."""
    d = Path(run_dir)
    part_json = _read_json(d / "part" / "part.json")
    frozen = _frozen_records(d)[0]
    cert_path = d / "certificate.json"
    if not cert_path.exists():
        raise FileNotFoundError(f"{d}: no certificate.json; run certify_part first (or run_part)")
    manifest_path = d / "search" / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    run = _read_json(d / "part_run.json") if (d / "part_run.json").exists() else {}
    evaluator = manifest.get("evaluator") or {}
    return {
        "part": part_json,
        "frozen": frozen,
        "certificate": _read_json(cert_path),
        "manifest": manifest,
        "proposal": _proposal_facts(d, frozen),
        "episodes_per_design": evaluator.get("episodes_per_design"),
        "run": run,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _images(part: PartSpec | None, jaw_mm: Any, iface: ClampInterface) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {"part": None, "jaws": None}
    if part is not None and part.mesh_mm is not None:
        pose = part.poses[0].pose_index if part.poses else 0
        out["part"] = render_png([(posed_part(part.mesh_mm, pose), PART_COLOR)])
    scene = jaw_pair_scene(jaw_mm, iface.stroke_mm, 1.0, iface.rod_spacing_mm, iface.rod_axis_height_mm)
    out["jaws"] = render_png(
        scene, layers=[0 if color == ROD_COLOR else 1 for _, color in scene], elev=18.0, azim=-35.0
    )
    return out


def _write_zip(src: Path, dest: Path) -> Path:
    """Zip the regular files directly in ``src`` (never symlinks or subdirectories, so
    nothing outside the pack can be pulled in), deterministically."""
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in src.iterdir() if p.is_file() and not p.is_symlink()):
            info = zipfile.ZipInfo(f"{src.name}/{path.name}", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    tmp.replace(dest)
    return dest


def make_print_pack(
    run_dir: str | Path,
    *,
    clamp: ClampInterface | None = None,
    out_dir: str | Path | None = None,
) -> Path:
    """Build the print pack for a part run (module docstring); returns the zip's path.

    Args:
        run_dir: A part run directory with ``part/``, ``search/frozen/`` and
            ``certificate.json``.
        clamp: Clamp interface for the jaw (default :class:`ClampInterface`).
        out_dir: Where ``print_pack/`` and ``print_pack.zip`` go (default ``run_dir``).
    """
    d = Path(run_dir)
    iface = clamp or ClampInterface()
    ctx = collect_run(d)
    frozen = ctx["frozen"]
    design = FingerDesign.model_validate(frozen["design"])
    if design.design_id() != frozen["design_id"]:
        raise ValueError("the frozen design does not match its design id")
    geom = generate_finger(design)

    base = Path(out_dir) if out_dir is not None else d
    pack = base / "print_pack"
    for target in (pack, base / "print_pack.zip", base / "print_pack.zip.tmp"):
        if target.is_symlink():  # stale files are deleted below: never follow a link out of the run
            raise ValueError(f"{target} is a symbolic link; refusing to write the print pack through it")
    pack.mkdir(parents=True, exist_ok=True)
    for stale in pack.iterdir():
        if stale.is_file() or stale.is_symlink():
            stale.unlink()

    export_stl(geom, pack / "finger_left.stl")
    export_stl(geom, pack / "finger_right.stl", mirror=True)
    jaw_mm = build_jaw(geom, iface)
    printable, needs_support = jaw_print_orientation(jaw_mm)
    (pack / "jaw.stl").write_bytes(printable.export(file_type="stl"))

    try:
        part = load_part(d / "part")
    except (FileNotFoundError, ValueError, KeyError):
        part = None
    ctx["images"] = _images(part, jaw_mm, iface)

    write_json_atomic(pack / "design.json", frozen)
    write_json_atomic(pack / "certificate.json", ctx["certificate"])
    (pack / "certificate.html").write_text(render_certificate_html(ctx), encoding="utf-8")
    (pack / "README.txt").write_text(
        _README.format(
            part=ctx["part"]["name"],
            disclaimer=DISCLAIMER,
            bore=f"{iface.rod_bore_d_mm:g}",
            spacing=f"{iface.rod_spacing_mm:g}",
            support="\n    Needs support under the inward bend, hook or rack." if needs_support else "",
            drive=(
                f"module {iface.rack_module_mm:g} rack for an {iface.pinion_teeth}-tooth pinion "
                "at an assumed position (check it)."
                if iface.rack
                else "not included (its position is not documented); transfer it from a stock jaw."
            ),
            mass=f"{frozen['mass_g']:.1f}",
            mu=f"{default_task().physics.friction:g}",
        ),
        encoding="utf-8",
    )

    manifest = ctx["manifest"]
    cert = ctx["certificate"]
    files = {
        name: {"bytes": (pack / name).stat().st_size, "sha256": _sha256(pack / name)}
        for name in sorted(p.name for p in pack.iterdir() if p.is_file())
    }
    for name in ("finger_left.stl", "finger_right.stl", "jaw.stl"):
        files[name]["units"] = "mm"
    files["jaw.stl"].update(quantity=2, orientation="print (contact face down)", needs_support=needs_support)
    files["finger_left.stl"]["orientation"] = files["finger_right.stl"]["orientation"] = "assembly frame"
    pack_manifest = {
        "format": "rtt-print-pack",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "license": "Generated files: CC0-1.0. Generator: Apache-2.0.",
        "notice": (
            "Clamp interface dimensions measured from an open-source parallel gripper's published hardware "
            "sources (CERN-OHL-P-2.0; attribution in the project README). New model; no files from that project."
        ),
        "files": files,
        "part": {
            k: ctx["part"][k]
            for k in ("name", "source_name", "source_sha256", "extents_mm", "mass_g", "material", "units")
        }
        | {"poses": [{"rank": p["rank"], "probability": p["probability"]} for p in ctx["part"]["poses"]]},
        "design": {
            "design_id": frozen["design_id"],
            "params": {k: v for k, v in frozen["design"].items() if k not in ("name", "rationale")},
            "origin": frozen["origin"],
            "training_success": frozen["training_success"],
            "controller_setting": frozen["controller_setting"],
            "mass_g": frozen["mass_g"],
            "frozen_sha256": frozen["content_sha256"],
        },
        "certificate": {
            "score": cert["score"],
            "passed": cert["passed"],
            "n_tests": cert["n_tests"],
            "placement_curve": cert["placement_curve"],
        },
        "clamp_interface": iface.describe(),
        "provenance": {
            "arm": manifest.get("arm"),
            "budget": manifest.get("budget"),
            "runs": manifest.get("runs"),
            "model": (manifest.get("model") or {}).get("model_id"),
            "provider": manifest.get("provider"),
            "llm_cost_usd": (manifest.get("llm") or {}).get("cost_usd"),
            "job": manifest.get("job"),
            "code": manifest.get("code"),
            "task_hash": (manifest.get("evaluator") or {}).get("task_hash"),
            "suite_hash": (manifest.get("evaluator") or {}).get("suite_hash"),
            "search_wall_s": manifest.get("wall_s"),
            "timings_s": (ctx.get("run") or {}).get("timings_s"),
        },
    }
    write_json_atomic(pack / "manifest.json", pack_manifest)
    return _write_zip(pack, base / "print_pack.zip")
