"""Track S5: build packs (printable parts, MJCF export, documents, manifest, licences).

The packs are built from a small tournament report made with the S1 reference controllers (fast,
independent of S2's tuning); such a report has no recommendation (reference runs are not
evidence), so every pack here names its finalist explicitly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from rtt.effectors.registry import get_family, override_family
from rtt.effectors.types import ALLOWED_LICENSES, PRODUCT_MODE_LABEL, AssetSource, find_brand_tokens
from rtt.studio import search
from rtt.studio.buildpack import export_build_pack, setup_from_dict, task_from_dict
from rtt.studio.buildpack.geometry import (
    bracket_part,
    changer_plate_part,
    check_printable,
    write_binary_stl,
)
from rtt.studio.buildpack.mjcf import export_mjcf
from rtt.studio.recommend import recommend_setups
from rtt.studio.runner import WorkerPool, condition_grid, run_full_tasks
from rtt.studio.search import (
    Combinations,
    TournamentReport,
    _predict,
    bracket_search,
    candidate_of,
    prune,
    screen,
    setup_result,
)
from rtt.studio.types import (
    BracketSpec,
    BuildPackManifest,
    HeadMount,
    Setup,
    SetupKind,
    ToolChanger,
    buildpack_manifest_json_schema,
)
from schema_lite import Validator
from studio_fixtures import cup, gripper, preset, two_press_task


def _hand():
    return preset("catalog_hand", "hand-5f-18act", "power")


def _finray():
    return preset("catalog_gripper", "gripper-2f-finray", "pinch")


@pytest.fixture(scope="module")
def report() -> dict:
    """Three finalists of every kind on the two-press task, reference controllers, 2 conditions."""
    task = two_press_task()
    cands = [candidate_of(s, "preset") for s in (_hand(), gripper(), _finray(), cup())]
    grid = condition_grid(task, 2)
    with WorkerPool(1) as pool:
        sc = screen(task, cands, grid[:1], pool=pool, controllers="reference")
        assignment = {"press_carton": 0, "press_flat": 1}
        geo, reasons = bracket_search((_hand(), gripper()), assignment, task)
        assert geo is not None, reasons
        multi = Setup(
            SetupKind.MULTI_HEAD, (_hand(), gripper()), assignment, mounts=geo.mounts, bracket=geo.bracket
        )
        plans = [
            _predict(sc, Setup.single(_finray(), task.step_ids), [_finray().effector_id] * 2),
            _predict(
                sc,
                Setup.tool_change({"press_carton": cup(), "press_flat": gripper()}),
                [cup().effector_id, gripper().effector_id],
            ),
            _predict(sc, multi, [_hand().effector_id, gripper().effector_id], geometry=geo),
        ]
        runs = run_full_tasks([p.setup for p in plans], task, grid, pool=pool, controllers="reference")
    results = [setup_result(p, task, runs[p.setup.setup_id]) for p in plans]
    rec = recommend_setups(results, best_single_id=plans[0].setup.setup_id)
    rep = TournamentReport(
        task_spec=task,
        conditions=grid,
        screen=sc,
        combinations=Combinations(plans, [], {}),
        finalists=plans,
        results=results,
        best_single=plans[0],
        recommendation=rec,
        sensitivity={},
        controllers="reference",
        timings={},
        workers=1,
    )
    return json.loads(json.dumps(rep.to_dict()))


# --------------------------------------------------------------------------- printable parts


@pytest.mark.parametrize(
    "mounts",
    [
        (HeadMount(-45.0, 70.0), HeadMount(45.0, 70.0)),
        (HeadMount(-60.0, 100.0), HeadMount(60.0, 100.0, spin_deg=30.0)),
        (HeadMount(0.0, 50.0), HeadMount(60.0, 150.0)),
        (HeadMount(0.0, 50.0), HeadMount(60.0, 50.0)),  # the tilted adapter rises beside the flange
    ],
)
def test_the_bracket_is_one_watertight_solid_below_the_flange(mounts, tmp_path: Path) -> None:
    part = bracket_part(mounts, BracketSpec())
    path = write_binary_stl(part.mesh, tmp_path / "bracket.stl")
    mesh = trimesh.load(str(path), force="mesh")
    check_printable(mesh, "reloaded bracket")
    assert mesh.is_watertight and mesh.body_count == 1 and mesh.volume > 0
    v = mesh.vertices  # nothing above the flange's mounting face within the arm's last link
    assert not ((v[:, 2] > 0.5) & (np.hypot(v[:, 0], v[:, 1]) < 41.5)).any()
    assert part.facts()["genus"] >= 12  # 4 bolt holes on the hub and on each adapter
    assert 0.03 < part.mass_kg < 0.5


def test_a_bracket_reaching_above_the_flange_is_refused() -> None:
    with pytest.raises(ValueError):
        bracket_part((HeadMount(0.0, 60.0), HeadMount(120.0, 60.0)), BracketSpec())


def test_changer_plates_are_watertight_and_stl_writing_is_deterministic(tmp_path: Path) -> None:
    ch = ToolChanger()
    part = changer_plate_part(ch, 0)
    check_printable(part.mesh, "plate")
    assert part.mesh.bounds[1][2] - part.mesh.bounds[0][2] == pytest.approx(0.5 * ch.stack_mm)
    a = write_binary_stl(part.mesh, tmp_path / "a.stl").read_bytes()
    b = write_binary_stl(changer_plate_part(ch, 0).mesh, tmp_path / "b.stl").read_bytes()
    assert a == b and len(a) == 84 + 50 * len(part.mesh.faces)


# --------------------------------------------------------------------------- packs


def _text_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix in (".html", ".md", ".txt", ".csv", ".json")]


@pytest.mark.parametrize("kind", ["single", "tool_change", "multi_head"])
def test_a_build_pack_is_complete_hashed_licensed_and_loads(report: dict, kind: str, tmp_path: Path) -> None:
    fin = next(f for f in report["finalists"] if f["setup"]["kind"] == kind)
    sid = fin["setup"]["setup_id"]
    pack = export_build_pack(report, tmp_path / "pack", setup_id=sid[:12])
    root = pack.out_dir
    manifest = json.loads((root / "manifest.json").read_text())
    assert Validator(buildpack_manifest_json_schema()).errors(manifest) == []
    assert manifest["label"] == PRODUCT_MODE_LABEL and manifest["setup_id"] == sid
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"manifest.json"}
    listed = {f["path"] for f in manifest["files"]}
    assert on_disk == listed
    kinds = {f["kind"] for f in manifest["files"]}
    assert {"mjcf", "certificate", "bom", "notes", "licence"} <= kinds
    for f in manifest["files"]:
        assert hashlib.sha256((root / f["path"]).read_bytes()).hexdigest() == f["sha256"]
        assert f["license"] in ALLOWED_LICENSES
        assert not find_brand_tokens(f["path"]) and not find_brand_tokens(f["attribution"])
        if f["source"] == "library":
            assert f["attribution"]
        if f["kind"] in ("bracket", "changer_plate"):
            assert f["watertight"] is True
            check_printable(trimesh.load(str(root / f["path"]), force="mesh"), f["path"])
    if kind == "multi_head":
        assert "bracket" in kinds
    if kind == "tool_change":
        assert sum(f["kind"] == "changer_plate" for f in manifest["files"]) == 2
    # every tool has CAD or an upstream link
    n_tools = len(fin["setup"]["tools"])
    cad = sum(f["kind"] == "effector_cad" for f in manifest["files"])
    assert cad + len(manifest["links"]) >= n_tools
    # the exported scenes reload as tested
    for info in pack.mjcf:
        assert info["verified"]["same_as_tested"], info
        model = mujoco.MjModel.from_xml_path(str(root / info["file"]))
        assert model.nu == info["verified"]["tested"]["nu"]
    # the certificate: self-contained, labelled, brand-free, per-step and conditions
    html = (root / "certificate.html").read_text()
    assert "<script" not in html.lower() and not re.search(r'(src|href)="https?:', html)
    assert html.count(PRODUCT_MODE_LABEL) >= 2 and "Not evidence" in html
    for s in report["task"]["steps"]:
        assert s["id"] in html
    for c in report["conditions"]:
        assert c["condition_id"] in html
    assert "roll_torque_nm" in html and "tool_change_s" in html
    for p in _text_files(root):
        if p.name == "CREDITS.txt" or p.parent.name == "licences":
            continue  # the credits file is the one place upstream names may appear
        text = p.read_text()
        if p.name == "manifest.json":  # upstream link URLs are credits, as on the library page
            data = json.loads(text)
            for link in data["links"]:
                link["url"] = ""
            text = json.dumps(data)
        assert not find_brand_tokens(text), p
    assert BuildPackManifest is not None and pack.zip_path is not None and pack.zip_path.is_file()


def test_build_pack_zips_are_deterministic(report: dict, tmp_path: Path) -> None:
    sid = next(f for f in report["finalists"] if f["setup"]["kind"] == "tool_change")["setup"]["setup_id"]
    a = export_build_pack(report, tmp_path / "a" / "pack", setup_id=sid)
    b = export_build_pack(report, tmp_path / "b" / "pack", setup_id=sid)
    assert a.zip_path.read_bytes() == b.zip_path.read_bytes()
    with pytest.raises(FileExistsError):
        export_build_pack(report, tmp_path / "a" / "pack", setup_id=sid)
    c = export_build_pack(report, tmp_path / "a" / "pack", setup_id=sid, overwrite=True)
    assert c.manifest.to_dict() == a.manifest.to_dict()


def test_a_report_without_a_recommendation_needs_a_setup_id(report: dict, tmp_path: Path) -> None:
    assert report["recommendation"]["recommendation"] is None
    with pytest.raises(ValueError, match="setup_id"):
        export_build_pack(report, tmp_path / "pack")
    for f in report["finalists"]:
        assert setup_from_dict(f["setup"]).setup_id == f["setup"]["setup_id"]
    assert task_from_dict(report["task"]).task_id == report["task_id"]


def test_an_amber_tool_never_competes_and_is_never_shipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = type(get_family("suction"))
    amber = AssetSource(
        path="assets/effectors/example/LICENSE",
        sha256="0" * 64,
        upstream_url="https://example.org/cup",
        upstream_commit="abcdef0",
        upstream_path="LICENSE",
        upstream_sha256="0" * 64,
        license="MIT",
        copyright="Copyright example",
        attribution="Example cup",
        status="amber",
        note="the licence chain is not confirmed",
    )

    class AmberCups(base):  # type: ignore[misc, valid-type]
        def build(self, ctx):  # type: ignore[no-untyped-def]
            return replace(super().build(ctx), sources=(amber,))

    task = two_press_task()
    monkeypatch.setattr(search, "_TOOLS", {})
    with override_family(AmberCups()):
        _, pruned, infos = prune(task, [candidate_of(cup(), "preset")])
        assert infos[cup().effector_id].amber
        assert any("amber" in p.reason for p in pruned)
        with pytest.raises(ValueError, match="amber"):
            export_mjcf(Setup.single(cup(), task.step_ids), task, 0, tmp_path / "mjcf")
