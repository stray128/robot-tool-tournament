"""The ``rtt`` command line: m1, probe-llm (offline paths only) and export-geom."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from rtt.cli import main
from rtt.llm.probe import DEFAULT_DOC

CASES = ("straight", "concave", "hooked", "stock", "too_high", "low_friction", "too_wide")


def test_m1_gate_passes_and_writes_one_replay_per_case(tmp_path: Path, capsys) -> None:
    status = main(["m1", "--no-video", "--json", "--out", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert status == 0 and report["passed"] is True
    assert tuple(report["cases"]) == CASES
    assert report["invalid_entrants"] == {}
    for name, case in report["cases"].items():
        assert case["passed"], (name, case)
        assert case["rests_on_table"]
        assert case["success"] == (case["expect"] == "success")
        assert sorted(Path(f).name for f in case["files"]) == [f"{name}.bin", f"{name}.json"]
    header = json.loads((tmp_path / "hooked.json").read_text())
    assert header["meta"]["entrant"] == "hooked"
    assert header["meta"]["design"]["tip_profile"] == "hook"
    assert header["meta"]["success"] is True


def test_m1_table_without_files(tmp_path: Path, capsys) -> None:
    assert main(["m1", "--no-files", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "G1 PASSED: 7/7 cases as expected" in out
    for name in CASES:
        assert f"\n{name} " in out
    assert not any(tmp_path.iterdir())


def test_export_geom_stl_and_glb(tmp_path: Path, capsys) -> None:
    stl = tmp_path / "hooked.stl"
    assert main(["export-geom", "hooked", "--out", str(stl)]) == 0
    mesh = trimesh.load(stl)
    assert mesh.is_watertight
    # millimetres: 20 mm wide, about 80 mm long, hook protruding toward -Y
    assert mesh.extents[0] == pytest.approx(20.0, abs=0.1)
    assert 70.0 < mesh.extents[2] < 81.0
    assert mesh.bounds[0][1] < -5.0

    right = tmp_path / "hooked_right.stl"
    assert main(["export-geom", "hooked", "--mirror", "--out", str(right)]) == 0
    assert trimesh.load(right).bounds[1][1] > 5.0  # the mirrored hook points toward +Y

    glb = tmp_path / "concave.glb"
    assert main(["export-geom", "concave", "--out", str(glb)]) == 0
    scene = trimesh.load(glb)
    assert set(scene.geometry) == {"finger_left", "finger_right", "palm"}
    out = capsys.readouterr().out
    assert "hooked (design " in out and "valid" in out and "watertight" in out


def test_export_geom_rejects_bad_requests(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["export-geom", "nonexistent", "--out", str(tmp_path / "x.stl")])
    assert exc.value.code == 2
    assert "straight, concave, hooked" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["export-geom", "straight", "--out", str(tmp_path / "x.obj")])

    bad = tmp_path / "entrants.yaml"
    bad.write_text(
        "entrants:\n"
        "  - {name: flimsy, finger_length_mm: 80, finger_thickness_mm: 4, finger_width_mm: 20,\n"
        "     bend_angle_deg: 0, bend_start_frac: 0.5, inner_face: flat, inner_face_depth_mm: 0,\n"
        "     tip_profile: blunt, tip_size_mm: 0, taper_ratio: 0.5}\n"
    )
    out_stl = tmp_path / "flimsy.stl"
    assert main(["export-geom", "flimsy", "--entrants", str(bad), "--out", str(out_stl)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "thinnest section" in out and not out_stl.exists()
    assert (
        main(["export-geom", "flimsy", "--entrants", str(bad), "--out", str(out_stl), "--allow-invalid"]) == 0
    )
    assert out_stl.exists()


def test_probe_llm_default_prints_the_report_summary(capsys) -> None:
    if not DEFAULT_DOC.is_file():
        pytest.skip("docs/PROBE_LLM.md has not been generated")
    assert main(["probe-llm"]) == 0
    out = capsys.readouterr().out
    assert "nemotron-3-super-120b-a12b" in out and "Recommendation:" in out
    assert "rtt probe-llm --live" in out


def test_probe_llm_missing_report(tmp_path: Path, capsys) -> None:
    assert main(["probe-llm", "--doc", str(tmp_path / "none.md")]) == 1
    assert "no probe report" in capsys.readouterr().out


def test_probe_llm_replay_never_needs_a_key(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    doc = tmp_path / "probe.md"
    args = ["probe-llm", "--replay", "--calls", "1", "--out", str(tmp_path / "run"), "--doc", str(doc)]
    assert main(args) == 1  # empty cache: stops at the first miss instead of calling the API
    assert "replay cache miss" in capsys.readouterr().err
    assert doc.is_file()
    with pytest.raises(SystemExit):
        main(["probe-llm", "--live", "--replay"])
