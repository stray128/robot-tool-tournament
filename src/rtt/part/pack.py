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

**Print pack v2** (:func:`make_print_pack_v2`) packs one freeform jaw pair compiled for one
part (:class:`rtt.design.freeform.types.FingerPair`):

* ``left.stl``, ``right.stl``: each jaw's printed solid, binary, millimetres, watertight,
  turned for printing: root plane on the bed, tip up, so the extruded pad faces print as
  vertical walls (no supports) and the pad profile lies in the layer plane. The right jaw
  is its own part (built against the mirrored part), never the left one mirrored.
* ``left.glb``, ``right.glb``: the same jaws for viewing, in their jaw (body) frames.
* ``mount_left.stl``, ``mount_right.stl``: the jaw fused with the carriage block for the
  gripper's guide rods (:mod:`rtt.part.mount`), upright on the block. The block is as wide as
  the jaw's root block (:func:`v2_clamp`: 46 mm, the clamp cut face measured on the gripper),
  and the jaw's two M3 bolt holes, which open on the root plane the block covers, are filled,
  so each mount is one closed solid with no enclosed void.
* ``recipe.json``, ``certificate.html`` (when given), ``ATTRIBUTION.txt``,
  ``manifest.json`` (every file's bytes and sha256, the jaw-frame STL hashes checked
  against ``JawGeometry.print_mesh_sha``, masses, ids) and ``README.txt``.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import trimesh

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

__all__ = [
    "PACK_FILES",
    "PACK_V2_FILES",
    "collect_run",
    "make_print_pack",
    "make_print_pack_v2",
    "v2_clamp",
]

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
_SAFE_FOLDER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}")

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


# --------------------------------------------------------------------------- print pack v2

PACK_V2_FILES: Final = (
    "ATTRIBUTION.txt",
    "README.txt",
    "left.glb",
    "left.stl",
    "manifest.json",
    "mount_left.stl",
    "mount_right.stl",
    "recipe.json",
    "right.glb",
    "right.stl",
)

_README_V2: Final = """Print pack: {design} for {part}
==============================================

{disclaimer}

Files
-----
left.stl, right.stl
    The two jaws in millimetres, already turned for printing: root plane on the bed,
    tip up. The pad faces print as vertical walls, so no supports are needed. The right
    jaw is its own part; do not mirror the left one.
    Each has two M3 holes (3.2 mm, 6 mm deep) in its root block, to bolt it onto a carriage.
mount_left.stl, mount_right.stl
    The same jaws fused with a {width} mm carriage block for the gripper's two guide rods
    (bores {bore} mm at {spacing} mm centres), upright on the block. Each is one piece, so
    its M3 holes are filled. Print these to replace the stock jaws; print left/right.stl
    to bolt onto your own carriages instead.
left.glb, right.glb
    The jaws for any 3D viewer, in the gripper's frame (metres, Z up).
recipe.json, manifest.json
    The design recipe and the checksums of every file.

Printing
--------
PLA or PETG, 0.2 mm layers, 4 perimeters, 40 % infill or more. The simulation assumed
solid PLA jaws ({mass_left} g left, {mass_right} g right) and a friction coefficient of
{mu} between jaw and part.

Credits: see ATTRIBUTION.txt.
"""

_ATTRIBUTION_V2: Final = """Attribution
===========

Generated files in this pack (STL, GLB, JSON): CC0-1.0. The generator is Apache-2.0
(Robot Tool Tournament).

The carriage block's interface dimensions were measured from the published hardware
sources of an open-source parallel gripper: {credit} (simulation files CC BY 4.0;
hardware sources CERN-OHL-P-2.0). The block is a new model drawn from those dimensions;
no file from that project is included in this pack.
"""


# the pack's GLBs are generated files like its STLs, so they carry the pack's licence (CC0-1.0)
_PACK_GLB_CREDIT: Final = "Robot Tool Tournament print pack, generated jaw (CC0-1.0)"


def _print_upright(mesh_mm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Root plane on the bed, tip up: a 180 degree turn about x (a rotation, never a mirror)."""
    out = mesh_mm.copy()
    rot = np.eye(4)
    rot[1, 1] = rot[2, 2] = -1.0
    out.apply_transform(rot)
    lo, hi = out.bounds
    out.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    return out


V2_BLOCK_SOURCE: Final = (
    "measured: the clamp cut face of the gripper's simulation mesh (the jaw's root block)"
)
WIDTH_TOL_MM: Final = 1e-3  # float32 STL coordinates on the 46 mm root block land within this


def v2_clamp() -> ClampInterface:
    """The carriage block for a v2 jaw: as wide as the jaw's root block.

    v2 jaws carry the footprint of the clamp cut face measured on the gripper's own mesh
    (:data:`rtt.design.freeform.body.ROOT_BLOCK_X`, 46 mm across), wider than the 42 mm root
    footprint of the v1 clamp interface, so the v2 block is 46 mm wide. The rod bores, their
    spacing and the block depth are the v1 interface's.
    """
    from rtt.design.freeform.body import ROOT_BLOCK_X

    return ClampInterface(block_width_mm=round(1000.0 * (ROOT_BLOCK_X[1] - ROOT_BLOCK_X[0]), 4))


def _check_one_solid(mesh: trimesh.Trimesh, what: str) -> None:
    """Refuse anything but one closed, outward-facing solid of positive volume, with no void."""
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume):
        raise ValueError(f"{what} is not a closed, consistently oriented solid")
    if not mesh.volume > 0.0:
        raise ValueError(f"{what} has no positive volume ({mesh.volume:g} mm^3)")
    if mesh.body_count != 1:
        shells = mesh.split(only_watertight=False)
        voids = sum(1 for m in shells if m.volume < 0.0)
        if voids:
            raise ValueError(f"{what} encloses {voids} void(s); a printed solid must not")
        raise ValueError(f"{what} is {mesh.body_count} separate solids, not one")


def _fill_voids(mesh: trimesh.Trimesh, what: str) -> tuple[trimesh.Trimesh, int]:
    """The solid with every enclosed void filled, and how many there were.

    A void is a closed shell facing inward (negative volume) inside the solid; dropping its
    shell fills it exactly. More than one outward shell means separate solids: an error.
    """
    if mesh.body_count == 1:
        return mesh, 0
    shells = mesh.split(only_watertight=False)
    outer = [m for m in shells if m.volume > 0.0]
    if len(outer) != 1:
        raise ValueError(
            f"{what} is {len(outer)} separate solids, not one (the jaw does not reach its carriage block)"
        )
    return outer[0], len(shells) - 1


def _mount(jaw_mm: trimesh.Trimesh, side: str, iface: ClampInterface) -> tuple[trimesh.Trimesh, int]:
    """The jaw (jaw frame, mm) fused with the carriage block, turned upright on the block.

    The jaw's two M3 bolt holes open on its root plane, which the block covers: fused, they
    would be two sealed voids inside the print. They are filled (:func:`_fill_voids`), so the
    mount is exactly one closed solid; returns it and the number of voids filled.
    """
    from manifold3d import Manifold, OpType

    from rtt.design.freeform.types import CARRIAGE_INNER_Y
    from rtt.part.mount import _block, manifold_from_trimesh  # the v1 carriage block model

    block = _block(iface).translate((0.0, 1000.0 * CARRIAGE_INNER_Y, 0.0))
    if side == "right":  # the same block turned 180 degrees about z, as the stock right jaw
        block = block.rotate((0.0, 0.0, 180.0))
    lo, hi = jaw_mm.bounds
    half = iface.block_width_mm / 2.0 + WIDTH_TOL_MM
    if lo[0] < -half or hi[0] > half:
        raise ValueError(
            f"the {side} jaw spans x {lo[0]:.3f} to {hi[0]:.3f} mm, wider than the "
            f"{iface.block_width_mm:g} mm carriage block"
        )
    solid = Manifold.batch_boolean([block, manifold_from_trimesh(jaw_mm)], OpType.Add)
    mesh = solid.to_mesh()
    fused = trimesh.Trimesh(
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3], np.asarray(mesh.tri_verts), process=True
    )
    out, filled = _fill_voids(fused, f"the {side} mount")
    out = out.copy()
    lo, hi = out.bounds
    out.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])  # block down
    rot = np.eye(4)
    rot[1, 1] = rot[2, 2] = -1.0
    out.apply_transform(rot)  # tip up, block on the bed
    lo, hi = out.bounds
    out.apply_translation([0.0, 0.0, -lo[2]])
    return out, filled


def make_print_pack_v2(
    pair: Any,
    left_stl: bytes,
    right_stl: bytes,
    out_dir: str | Path,
    *,
    recipe: dict[str, Any],
    part: dict[str, Any] | None = None,
    certificate_html: str | None = None,
    clamp: ClampInterface | None = None,
    name: str | None = None,
) -> Path:
    """Build the v2 print pack for one jaw pair (module docstring); returns the zip's path.

    ``left_stl`` and ``right_stl`` are the generator's binary STLs (millimetres, jaw frame);
    their sha256 must equal the pair's ``print_mesh_sha``, so the pack ships exactly the
    solids whose mass the simulator used. Each jaw, and each jaw fused with its carriage
    block, must be one closed solid. The stock entrant is refused: its blade is the
    open-source gripper's own part (CC BY 4.0), not a generated CC0-1.0 jaw.
    """
    from rtt.design.freeform.types import FingerPair
    from rtt.render import glb
    from rtt.render import materials as mat
    from rtt.render.gripper import load_lock, stock_blade_shas

    if not isinstance(pair, FingerPair):
        raise TypeError("pair must be a rtt.design.freeform.FingerPair")
    stock = set(stock_blade_shas().values())
    if pair.left.print_mesh_sha in stock or pair.right.print_mesh_sha in stock:
        raise ValueError(
            "the stock blade is the open-source gripper's own part (CC BY 4.0), not a generated "
            "jaw; this pack's files are CC0-1.0, so it is not packed here"
        )
    iface = clamp or v2_clamp()
    meshes: dict[str, trimesh.Trimesh] = {}
    mounts: dict[str, trimesh.Trimesh] = {}
    voids: dict[str, int] = {}
    for side, data, jaw in (("left", left_stl, pair.left), ("right", right_stl, pair.right)):
        if hashlib.sha256(data).hexdigest() != jaw.print_mesh_sha:
            raise ValueError(f"{side}.stl does not match the pair's print_mesh_sha")
        mesh = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh", process=True)
        mesh.merge_vertices()
        _check_one_solid(mesh, f"{side}.stl")
        mount, filled = _mount(mesh, side, iface)
        _check_one_solid(mount, f"the {side} mount (jaw fused with its carriage block)")
        meshes[side], mounts[side], voids[side] = mesh, mount, filled
    base = Path(out_dir)
    folder = name or f"print_pack_{pair.design_id[:12]}_{pair.part_id}"
    if not _SAFE_FOLDER.fullmatch(folder):
        raise ValueError(f"pack name {folder!r} must be letters, digits, '_', '.' or '-'")
    pack = base / folder
    if pack.is_symlink():
        raise ValueError(f"{pack} is a symbolic link; refusing to write the print pack through it")
    pack.mkdir(parents=True, exist_ok=True)
    for stale in pack.iterdir():
        if stale.is_file() or stale.is_symlink():
            stale.unlink()

    files_meta: dict[str, dict[str, Any]] = {}
    for side, mesh in meshes.items():
        upright = _print_upright(mesh)
        (pack / f"{side}.stl").write_bytes(upright.export(file_type="stl"))
        (pack / f"mount_{side}.stl").write_bytes(mounts[side].export(file_type="stl"))
        jaw_m = mesh.copy()
        jaw_m.apply_scale(1e-3)
        (pack / f"{side}.glb").write_bytes(
            glb.glb_bytes(
                [glb.mesh_from_trimesh(jaw_m, mat.FINGER_DESIGNED, name=f"{side}_jaw")],
                copyright=_PACK_GLB_CREDIT,
                extras={"license": "CC0-1.0", "role": f"finger_{side}", "design_id": pair.design_id},
            )
        )
        files_meta[f"{side}.stl"] = {
            "units": "mm",
            "orientation": "print (root plane on the bed, tip up)",
            "watertight": True,
            "volume_mm3": round(float(mesh.volume), 3),
            "jaw_frame_sha256": pair.left.print_mesh_sha if side == "left" else pair.right.print_mesh_sha,
        }
        files_meta[f"mount_{side}.stl"] = {
            "units": "mm",
            "orientation": "print (carriage block on the bed)",
            "watertight": True,
            "volume_mm3": round(float(mounts[side].volume), 3),
            "voids_filled": voids[side],
        }
        files_meta[f"{side}.glb"] = {"units": "m", "frame": "jaw body frame, Z up"}
    write_json_atomic(pack / "recipe.json", recipe)
    if certificate_html is not None:
        (pack / "certificate.html").write_text(certificate_html, encoding="utf-8")
    lock = load_lock()
    (pack / "ATTRIBUTION.txt").write_text(
        _ATTRIBUTION_V2.format(credit=f"{lock['attribution']} (commit {lock['commit'][:8]})"),
        encoding="utf-8",
    )
    (pack / "README.txt").write_text(
        _README_V2.format(
            design=pair.design_id[:12],
            part=pair.part_id,
            disclaimer=DISCLAIMER,
            width=f"{iface.block_width_mm:g}",
            bore=f"{iface.rod_bore_d_mm:g}",
            spacing=f"{iface.rod_spacing_mm:g}",
            mass_left=f"{1000.0 * pair.left.mass:.1f}",
            mass_right=f"{1000.0 * pair.right.mass:.1f}",
            mu=f"{default_task().physics.friction:g}",
        ),
        encoding="utf-8",
    )
    clamp_interface = iface.describe()
    if iface.block_width_mm == v2_clamp().block_width_mm:
        clamp_interface["block_width_mm"]["source"] = V2_BLOCK_SOURCE
    files = {}
    for path in sorted(p for p in pack.iterdir() if p.is_file()):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            **files_meta.get(path.name, {}),
        }
    write_json_atomic(
        pack / "manifest.json",
        {
            "format": "rtt-print-pack",
            "version": 2,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "disclaimer": DISCLAIMER,
            "license": "Generated files: CC0-1.0. Generator: Apache-2.0.",
            "design": {
                "design_id": pair.design_id,
                "geometry_hash": pair.geometry_hash,
                "q_seat_mm": round(1000.0 * pair.q_seat, 4),
                "tip_height_mm": round(1000.0 * pair.tip_height, 4),
                "finger_length_mm": round(1000.0 * pair.finger_length, 4),
                "mass_g": {
                    "left": round(1000.0 * pair.left.mass, 3),
                    "right": round(1000.0 * pair.right.mass, 3),
                },
                "pieces": {"left": pair.left.n_pieces, "right": pair.right.n_pieces},
            },
            "part": {"part_id": pair.part_id, **(part or {})},
            "clamp_interface": clamp_interface,
            "files": files,
        },
    )
    return _write_zip(pack, base / f"{folder}.zip")
