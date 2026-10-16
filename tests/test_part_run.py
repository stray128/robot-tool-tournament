"""Per-part search, certificate and print pack, end to end on a real (small) simulated run."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest
import trimesh

from rtt.llm.client import TokenBucket
from rtt.part.__main__ import main as cli_main
from rtt.part.certificate import DISCLAIMER, render_certificate_html
from rtt.part.examples import example_path
from rtt.part.intake import intake
from rtt.part.pack import PACK_FILES, collect_run, make_print_pack
from rtt.part.search import (
    MAX_PART_SPEND_CAP_USD,
    PartCertificate,
    PartRun,
    PoseCertificate,
    _pick_candidate,
    best_frozen,
    check_spend_cap,
    frozen_from_run_dir,
    run_part,
    run_part_search,
)
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.sim.certify import Certificate, CertificateTest


def _attributed_names() -> tuple[str, ...]:
    """The upstream hardware project's owner and repository names, read from the README's
    attribution (they may appear there and nowhere on screen)."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    names: list[str] = []
    for owner, repo in re.findall(r"https://github\.com/([\w.-]+)/([\w.-]+)", readme):
        names += [owner.removesuffix("com"), repo, repo.rsplit("-", 1)[0]]
    return tuple(n for n in dict.fromkeys(names) if len(n) >= 5)


BRAND_WORDS = _attributed_names()


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> PartRun:
    """Random arm, budget 16, two resting poses of the dumbbell: about 30 s on 16 cores."""
    out = tmp_path_factory.mktemp("part_run")
    return run_part(example_path("dumbbell_handle"), out, arm="random", budget=16, top_k=2)


# --------------------------------------------------------------------------- end to end


def test_run_part_end_to_end(run: PartRun) -> None:
    d = run.run_dir
    manifest = json.loads((d / "search" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["arm"] == "random" and manifest["budget"] == 16
    assert manifest["evaluator"]["n_conditions"] == len(run.search.suite)
    assert manifest["evaluator"]["suite_hash"] == run.search.suite.digest()
    assert [f.design_id for f in run.search.best_designs] == [run.search.best.design_id]
    assert frozen_from_run_dir(d).design_id == run.search.best.design_id
    cert = json.loads((d / "certificate.json").read_text())
    assert cert["format"] == "rtt-part-certificate"
    assert cert["n_tests"] == 18 * len(run.part.poses)
    assert 0.0 <= cert["score"] <= 1.0 and cert["passed"] <= cert["n_tests"]
    assert cert["design_id"] == run.search.best.design_id
    timings = json.loads((d / "part_run.json").read_text())["timings_s"]
    assert set(timings) == {"intake_s", "search_s", "certify_s", "pack_s", "total_s"}
    assert run.pack is not None and run.pack.exists()
    assert (d / "conditions.json").exists() and (d / "part" / "part.json").exists()


def test_entrants_that_fit_are_round_zero(run: PartRun) -> None:
    history = (run.run_dir / "search" / "runs" / "random-r00" / "history.jsonl").read_text().splitlines()
    origins = [json.loads(line)["proposal"]["origin"] for line in history]
    assert origins[:4] == ["entrant"] * 4 and len(origins) == 16


def test_print_pack_contents(run: PartRun) -> None:
    assert run.pack is not None
    with zipfile.ZipFile(run.pack) as zf:
        names = zf.namelist()
        assert names == sorted(names)
        assert sorted(n.removeprefix("print_pack/") for n in names) == sorted(PACK_FILES)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in zf.infolist())
        manifest = json.loads(zf.read("print_pack/manifest.json"))
        import hashlib

        for name, facts in manifest["files"].items():
            data = zf.read(f"print_pack/{name}")
            assert hashlib.sha256(data).hexdigest() == facts["sha256"] and len(data) == facts["bytes"]
        texts = {n: zf.read(n).decode("utf-8") for n in names if not n.endswith(".stl")}
    assert manifest["files"]["jaw.stl"]["quantity"] == 2
    assert manifest["clamp_interface"]["rod_bore_d_mm"]["value"] == 6.1
    assert manifest["disclaimer"] == DISCLAIMER
    assert manifest["design"]["design_id"] == run.search.best.design_id
    for name, text in texts.items():
        for word in BRAND_WORDS:
            assert word.lower() not in text.lower(), f"an attributed name is in {name}"
    page = texts["print_pack/certificate.html"]
    assert DISCLAIMER in page and page.count("untested on hardware") >= 2
    assert "<script" not in page and "https://" not in page and "http://" not in page  # self-contained
    assert "data:image/png;base64," in page and "<svg" in page
    assert run.search.best.design_id in page


def test_print_pack_stls_are_millimetres(run: PartRun) -> None:
    pack = run.run_dir / "print_pack"
    design = run.search.best.design
    left = trimesh.load(pack / "finger_left.stl", force="mesh")
    right = trimesh.load(pack / "finger_right.stl", force="mesh")
    assert left.is_watertight and right.is_watertight
    assert left.extents[2] == pytest.approx(design.finger_length_mm, rel=0.15)  # mm, not m
    assert right.bounds[0][1] == pytest.approx(-left.bounds[1][1], abs=1e-4)  # mirror image across y
    jaw = trimesh.load(pack / "jaw.stl", force="mesh")
    assert jaw.is_watertight and jaw.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert jaw.extents[0] == pytest.approx(42.0, abs=1e-3)


def test_print_pack_rebuild_is_stable(run: PartRun, tmp_path: Path) -> None:
    again = make_print_pack(run.run_dir, out_dir=tmp_path)
    with zipfile.ZipFile(run.pack) as a, zipfile.ZipFile(again) as b:  # type: ignore[arg-type]
        assert a.namelist() == b.namelist()
        for name in ("finger_left.stl", "finger_right.stl", "jaw.stl", "design.json", "certificate.json"):
            assert a.read(f"print_pack/{name}") == b.read(f"print_pack/{name}"), name


def test_certificate_escapes_model_text(run: PartRun) -> None:
    ctx = collect_run(run.run_dir)
    ctx["proposal"] = {"hypothesis": "<script>alert('x')</script> & a hook", "analysis": "<b>bold</b>"}
    ctx["frozen"] = {**ctx["frozen"], "origin": "nemotron"}
    page = render_certificate_html(ctx)
    assert "<script>" not in page and "&lt;script&gt;" in page and "&lt;b&gt;bold" in page
    assert "Nemotron&#x27;s hypothesis" in page or "Nemotron's hypothesis" in page


def test_cli_intake_and_refusal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        cli_main(["intake", str(example_path("spur_gear")), "--top-k", "1", "--decomposition", "hull"]) == 0
    )
    assert "spur_gear" in capsys.readouterr().out
    cube = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(100.0, 100.0, 100.0)).export(cube)
    assert cli_main(["intake", str(cube), "--decomposition", "hull"]) == 2
    assert "part refused" in capsys.readouterr().err


# --------------------------------------------------------------------------- search plumbing (no simulator)


def test_nemotron_arm_runs_through_the_existing_runner(tmp_path: Path) -> None:
    part = intake(example_path("spur_gear"), decomposition="hull", top_k=1)
    client = make_proposer_client(None, transport=ScriptedProposer(5), limiter=TokenBucket(1e12))
    result = run_part_search(part, "nemotron", 16, FakeEvaluator(0), client, out_dir=tmp_path, runs=2)
    assert len(result.best_designs) == 2
    keys = [(-f.training_success, -f.training_hold_fraction, f.mass_g) for f in result.best_designs]
    assert keys == sorted(keys)
    manifest = json.loads((tmp_path / "search" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["model"]["model_id"]
    assert result.outcome.runs[0].ctx.run_seed != result.outcome.runs[1].ctx.run_seed
    with pytest.raises(ValueError, match="unknown arm"):
        run_part_search(part, "tpe", 16, FakeEvaluator(0), out_dir=tmp_path / "x")
    assert best_frozen([]) == []


# --------------------------------------------------------------------------- certificate maths


def _cert(results: list[tuple[str, str, dict, bool]], score: float) -> Certificate:
    tests = tuple(
        CertificateTest(n, c, p, ok, None if ok else "slip_during_lift", 0.15, 2.0, ())
        for n, c, p, ok in results
    )
    return Certificate("d", "obj", 1, "task_v1", "abc", tests, {}, score, {"x": 1.0, "y": 0.0})


def test_part_certificate_weights_poses_by_probability() -> None:
    a = _cert(
        [
            ("nominal", "placement", {"axis": None, "error_mm": 0.0}, True),
            ("offset_x_+1mm", "placement", {"axis": "x", "error_mm": 1.0}, True),
            ("offset_y_+1mm", "placement", {"axis": "y", "error_mm": 1.0}, False),
        ],
        1.0,
    )
    b = _cert(
        [
            ("nominal", "placement", {"axis": None, "error_mm": 0.0}, True),
            ("offset_x_+1mm", "placement", {"axis": "x", "error_mm": 1.0}, False),
            ("offset_y_+1mm", "placement", {"axis": "y", "error_mm": 1.0}, False),
        ],
        0.0,
    )
    pc = PartCertificate(
        "d",
        1,
        (
            PoseCertificate(0, 0.75, "narrow-centre", 0.0, a),
            PoseCertificate(1, 0.25, "narrow-centre", 0.0, b),
        ),
    )
    assert pc.score == pytest.approx(0.75)
    assert pc.passed == 3 and pc.n_tests == 6
    curve = pc.placement_curve()
    assert curve["x"] == [(0.0, 1.0), (1.0, 0.5)]
    assert curve["y"] == [(0.0, 1.0), (1.0, 0.0)]
    data = pc.to_json()
    assert data["score"] == 0.75 and "untested on hardware" in data["disclaimer"]


def test_pick_candidate_prefers_what_the_design_trained_well_on() -> None:
    part = intake(example_path("dumbbell_handle"), top_k=1)
    pose = part.poses[0]
    labels = [c.label for c in pose.candidates if c.axis == "narrow"]
    assert len(labels) == 2
    per = {
        f"part_x/pose0-{labels[0]}/centred": 0.0,
        f"part_x/pose0-{labels[0]}/sloppy": 0.0,
        f"part_x/pose0-{labels[1]}/centred": 1.0,
        f"part_x/pose0-{labels[1]}/sloppy": 0.5,
    }
    assert _pick_candidate(pose, per)[0] == labels[1]
    assert _pick_candidate(pose, None)[0] == labels[0]


# --------------------------------------------------------------------------- caps and pack hardening


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -0.01, 5.01, None, True, "1"])
def test_spend_cap_must_be_finite_and_bounded(cap: object, tmp_path: Path) -> None:
    """The ledger treats None as uncapped and never refuses against NaN or inf."""
    part = intake(example_path("rounded_tub"), decomposition="hull", top_k=1)
    with pytest.raises((ValueError, TypeError), match="spend_cap_usd"):
        run_part_search(part, "nemotron", 16, FakeEvaluator(0), out_dir=tmp_path, spend_cap_usd=cap)  # type: ignore[arg-type]
    assert not (tmp_path / "search").exists() and not (tmp_path / "llm_ledger.jsonl").exists()
    with pytest.raises((ValueError, TypeError), match="spend_cap_usd"):
        run_part(example_path("rounded_tub"), tmp_path / "run", arm="random", spend_cap_usd=cap)  # type: ignore[arg-type]
    assert not (tmp_path / "run").exists()  # refused before intake
    assert check_spend_cap(0) == 0.0 and check_spend_cap(MAX_PART_SPEND_CAP_USD) == MAX_PART_SPEND_CAP_USD


@pytest.mark.parametrize("text", ["nan", "inf", "-1", "6", "lots"])
def test_cli_refuses_an_unbounded_spend_cap(text: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["run", str(example_path("rounded_tub")), "--out", "unused", "--spend-cap", text])
    assert exc.value.code == 2 and "spend" in capsys.readouterr().err


def test_print_pack_zips_only_its_own_files(run: PartRun, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the pack")
    planted = tmp_path / "print_pack"
    (planted / "nested").mkdir(parents=True)
    (planted / "nested" / "copy.txt").write_text("nested")
    (planted / "link.txt").symlink_to(secret)
    (planted / "stale.txt").write_text("old")
    zpath = make_print_pack(run.run_dir, out_dir=tmp_path)
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(n.removeprefix("print_pack/") for n in zf.namelist())
    assert names == sorted(PACK_FILES)
    assert secret.read_text() == "not for the pack"  # the link was removed, not followed


def test_certificate_text_follows_the_recorded_battery(run: PartRun) -> None:
    ctx = collect_run(run.run_dir)
    settings = ctx["certificate"]["poses"][0]["settings"]
    page = render_certificate_html(ctx)
    assert f"{settings['shake']['frequency_hz']:g} Hz" in page
    assert f"&plusmn;{max(settings['placement_errors_mm']):g} mm" in page
    assert f"held {settings['post_perturb_hold_s']:g} s after" in page


def test_print_pack_never_writes_through_a_symlinked_directory(run: PartRun, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    keep = elsewhere / "keep.txt"
    keep.write_text("someone else's file")
    out = tmp_path / "out"
    out.mkdir()
    (out / "print_pack").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        make_print_pack(run.run_dir, out_dir=out)
    assert keep.read_text() == "someone else's file"  # the stale-file sweep never reached it
