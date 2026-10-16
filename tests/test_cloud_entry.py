"""rtt.cloud.entry (in-container runner), rtt.cloud.runs and the laptop CLI's offline paths."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rtt.cloud import entry
from rtt.cloud.__main__ import main as cloud_main
from rtt.cloud.config import ConfigError, NebiusConfig
from rtt.cloud.cost import JobSpend, SpendLedger
from rtt.cloud.runs import (
    SearchJobArgs,
    entry_args,
    job_spec_for,
    mp4_complete,
    new_run_id,
    run_prefix,
    split_image_ref,
    verify_run,
)

DIGEST = "sha256:" + "1" * 64
IMAGE = "cr.eu-north1.nebius.cloud/abc123/rtt-sim@" + DIGEST  # "abc123": the registry path
CONFIG = NebiusConfig(
    project_id="project-test",
    subnet_id="vpcsubnet-test",
    bucket_id="storagebucket-test",
    bucket_name="bucket",
    image=IMAGE,
)


def test_run_ids_and_entry_args() -> None:
    assert new_run_id("m1", datetime(2026, 10, 15, 10, 2, 3, tzinfo=UTC)) == "m1-20261015-100203"
    assert run_prefix("m1-x") == "runs/m1-x"
    assert entry_args("m1", "m1-x", seed=3, video=False) == [
        "-m", "rtt.cloud.entry", "m1", "--out", "/out/runs/m1-x", "--seed", "3", "--no-video", "--probe-warp",
    ]  # fmt: skip
    assert entry_args("probe", "p", probe_warp=False)[-1] == "--no-warp"
    with pytest.raises(ValueError, match="unknown run kind"):
        entry_args("train", "x")


@pytest.mark.parametrize("bad", ["", "../etc", "runs/x", "/abs", "M1-X", "-lead", "a_b", "x" * 60, "a\nb"])
def test_unsafe_run_ids_are_refused(bad: str, tmp_path: Path, capsys) -> None:
    with pytest.raises(ValueError, match="run ID"):
        run_prefix(bad)
    # the CLI refuses before touching the bucket or the local runs folder
    assert cloud_main(["fetch", "--fetch-dir", str(tmp_path / "runs"), "--", bad]) == 2
    assert "run ID" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
    with pytest.raises(ValueError, match="needs SearchJobArgs"):
        entry_args("search", "x")
    search = SearchJobArgs(arm="nemotron", runs=2, budget=40, seeds="7,8", spend_cap_usd=0.25)
    assert entry_args("search", "s-x", search=search) == [
        "-m", "rtt.cloud.entry", "search", "--out", "/out/runs/s-x",
        "--arm", "nemotron", "--runs", "2", "--budget", "40", "--executor", "cpu", "--spend-cap", "0.25",
        "--seeds", "7,8",
    ]  # fmt: skip


def test_job_spec_for_wires_image_secret_and_bucket() -> None:
    spec = job_spec_for(CONFIG, "m1", "m1-x")
    spec.validate()
    assert spec.name == "rtt-m1-x" and spec.image == IMAGE
    assert spec.command == "/app/.venv/bin/python"
    assert spec.env_secrets == {}  # the gate never calls the model, so it never gets the key
    assert job_spec_for(CONFIG, "probe", "p-x").env_secrets == {"NEBIUS_API_KEY": "tf-api-key"}
    llm = SearchJobArgs(arm="nemotron", runs=1, budget=4)
    assert job_spec_for(CONFIG, "search", "s-x", search=llm).env_secrets == {"NEBIUS_API_KEY": "tf-api-key"}
    rnd = SearchJobArgs(arm="random", runs=1, budget=4)
    assert job_spec_for(CONFIG, "search", "s-y", search=rnd).env_secrets == {}
    assert job_spec_for(CONFIG, "m1", "m1-y", attach_key=True).env_secrets == {"NEBIUS_API_KEY": "tf-api-key"}
    assert "NEBIUS_API_KEY" not in spec.env
    # manifests learn the image name and digest, never the registry path
    assert spec.env["RTT_IMAGE_NAME"] == "rtt-sim" and spec.env["RTT_IMAGE_DIGEST"] == DIGEST
    assert spec.env["RTT_LAUNCHER"] == "nebius-serverless-jobs"
    assert not any("abc123" in v for v in spec.env.values())
    assert spec.volumes == ["storagebucket-test:/out:rw"]
    assert spec.restart_policy == "never" and spec.timeout == "1h"
    with pytest.raises(ConfigError, match="RTT_IMAGE"):
        job_spec_for(NebiusConfig(), "m1", "m1-x")


def test_split_image_ref() -> None:
    assert split_image_ref(IMAGE) == ("rtt-sim", DIGEST)
    with pytest.raises(ValueError, match="digest"):
        split_image_ref("cr.eu-north1.nebius.cloud/abc123/rtt-sim:dev")


def test_m1_entry_writes_report_and_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("RTT_RUN_ID", "m1-test")
    monkeypatch.setenv("RTT_PLATFORM", "gpu-h100-sxm")
    monkeypatch.setenv("RTT_IMAGE_NAME", "rtt-sim")
    monkeypatch.setenv("RTT_IMAGE_DIGEST", DIGEST)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    out = tmp_path / "runs" / "m1-test"
    status = entry.main(["m1", "--out", str(out), "--no-video"])
    assert status == 0
    summary = capsys.readouterr().out.strip().splitlines()[-1]
    assert summary.startswith("RTT_SUMMARY ") and json.loads(summary.split(" ", 1)[1])["exit_status"] == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema"] == entry.MANIFEST_SCHEMA and manifest["kind"] == "m1"
    assert manifest["run_id"] == "m1-test" and manifest["image"] == f"rtt-sim@{DIGEST}"
    assert manifest["image_digest"] == DIGEST
    assert manifest["nebius"]["platform"] == "gpu-h100-sxm"
    assert manifest["provider"] == "nebius-serverless-jobs"
    assert manifest["result"]["passed"] is True
    assert len(manifest["inputs"]["task_sha256"]) == 64 and manifest["build"]["uv_lock_sha256"]
    assert manifest["environment"]["token_factory_key_present"] is False
    assert "RTT_IMAGE_DIGEST" in manifest["environment"]["env_names"]
    assert DIGEST not in json.dumps(manifest["environment"])  # names only, never values

    # staged locally, then published whole: no temp files reach the destination
    assert not list(out.glob("*.tmp"))
    assert manifest["publish"]["files"] == len(manifest["files"]) - 1
    assert manifest["publish"]["size_mismatches"] == []
    assert manifest["files"][-1] == "manifest.json"

    report = json.loads((out / "report.json").read_text())
    assert report["passed"] is True
    for case in report["cases"].values():
        assert all(not Path(f).is_absolute() for f in case["files"])
    check = verify_run(out)
    assert check.ok, check.describe()


def test_probe_entry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    manifest, status = entry.run_probe(tmp_path / "p", probe_warp=False)
    assert status == 0 and manifest["kind"] == "probe"
    probe = json.loads((tmp_path / "p" / "probe.json").read_text())
    assert probe["gl"]["error"].startswith("set MUJOCO_GL")
    assert "packages" in probe and probe["packages"]["mujoco"]
    assert verify_run(tmp_path / "p").ok


def test_publish_copies_whole_files_and_skips_manifest(tmp_path: Path) -> None:
    stage, out = tmp_path / "stage", tmp_path / "out"
    (stage / "sub").mkdir(parents=True)
    (stage / "a.mp4").write_bytes(b"x" * 3_000_000)
    (stage / "sub" / "b.json").write_text("{}")
    (stage / "manifest.json").write_text("{}")
    stats = entry.publish(stage, out)
    assert stats["files"] == 2 and stats["bytes"] == 3_000_002 and stats["size_mismatches"] == []
    assert stats["names"] == ["a.mp4", "sub/b.json"]
    assert (out / "a.mp4").read_bytes() == (stage / "a.mp4").read_bytes()
    assert not (out / "manifest.json").exists()


def test_explicit_stage_is_kept(tmp_path: Path) -> None:
    manifest, status = entry.run_probe(tmp_path / "out", probe_warp=False, stage_dir=tmp_path / "stage")
    assert status == 0 and (tmp_path / "stage" / "probe.json").is_file()
    assert (tmp_path / "out" / "manifest.json").is_file() and manifest["publish"]["files"] == 1


def test_gpu_info_without_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    assert "error" in entry.gpu_info()


def test_cpu_info_reads_model_name(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 9654 96-Core Processor\n"
    )
    assert entry.cpu_info(cpuinfo)["model"] == "AMD EPYC 9654 96-Core Processor"
    assert entry.cpu_info(tmp_path / "absent")["model"] is None
    assert "cpu" in entry.environment(probe_gl=False)


def test_verify_run_reports_problems(tmp_path: Path) -> None:
    assert verify_run(tmp_path).problems == ("missing manifest.json",)
    (tmp_path / "manifest.json").write_text("")  # what a rename on the bucket mount left behind
    assert verify_run(tmp_path).problems == ("manifest.json is empty or not JSON",)
    manifest = {
        "kind": "m1",
        "exit_status": 1,
        "files": ["report.json", "gone.bin"],
        "inputs": {"video": True},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = {"passed": False, "cases": {"straight": {"passed": False}}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "straight.bin").write_bytes(b"")
    check = verify_run(tmp_path)
    assert not check.ok
    assert set(check.problems) == {
        "exit_status 1",
        "listed file missing: gone.bin",
        "gate report says FAILED",
        "case straight not as expected",
        "case straight: missing straight.json",
        "case straight: missing straight.mp4",
    }
    assert "NOT verified" in check.describe()


def _mp4(boxes: list[tuple[bytes, bytes]]) -> bytes:
    return b"".join((8 + len(body)).to_bytes(4, "big") + kind + body for kind, body in boxes)


def test_verify_run_catches_empty_truncated_and_short_files(tmp_path: Path) -> None:
    good = _mp4([(b"ftyp", b"isom"), (b"mdat", b"x" * 100), (b"moov", b"index")])
    no_index = _mp4([(b"ftyp", b"isom"), (b"mdat", b"x" * 100)])
    unpatched = good[:12] + (10_000).to_bytes(4, "big") + good[16:]  # mdat size past EOF
    assert mp4_complete_bytes(tmp_path, good)
    assert not mp4_complete_bytes(tmp_path, no_index)
    assert not mp4_complete_bytes(tmp_path, unpatched)

    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"passed": True, "cases": {"s": {"passed": True}}}))
    (run / "s.bin").write_bytes(b"")  # what a failed write on the mount leaves
    (run / "s.json").write_text("{}")
    (run / "s.mp4").write_bytes(no_index)
    files = ["report.json", "s.bin", "s.json", "s.mp4"]
    manifest = {
        "kind": "m1",
        "exit_status": 0,
        "files": [*files, "manifest.json"],
        "inputs": {"video": True},
        "publish": {"bytes": 1},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    total = sum((run / f).stat().st_size for f in files)
    # with a published byte total, the empty file shows up as a short total
    assert set(verify_run(run).problems) == {
        f"files total {total} bytes, the job published 1",
        "case s: s.mp4 is truncated (no ftyp/moov box)",
    }
    # without one, any empty file is suspect
    (run / "manifest.json").write_text(json.dumps({**manifest, "publish": {}}))
    assert "listed file empty: s.bin" in verify_run(run).problems
    # a legitimately empty file (e.g. no rejections) that the total accounts for is fine
    (run / "s.bin").write_bytes(b"x")
    (run / "s.mp4").write_bytes(good)
    (run / "empty.jsonl").write_bytes(b"")
    listed = [*files, "empty.jsonl"]
    exact = {
        **manifest,
        "files": [*listed, "manifest.json"],
        "publish": {"bytes": sum((run / f).stat().st_size for f in listed)},
    }
    (run / "manifest.json").write_text(json.dumps(exact))
    assert verify_run(run).ok, verify_run(run).problems
    (run / "report.json").write_text(json.dumps({"passed": True, "cases": {}}))
    assert "gate report has no cases" in verify_run(run).problems


def mp4_complete_bytes(tmp_path: Path, data: bytes) -> bool:
    path = tmp_path / "probe.mp4"
    path.write_bytes(data)
    return mp4_complete(path)


def test_cli_spend_and_parser(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RTT_NEBIUS_ENV", str(tmp_path / "absent.env"))
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ledger.append(
        JobSpend("aijob-1", "rtt-m1-x", "gpu-h100-sxm", "1gpu-16vcpu-200gb", "eu-north1", "COMPLETED",
                 False, 120.0, 60.0, 300.0, 4.5, 0.375, 0.075)
    )  # fmt: skip
    assert cloud_main(["spend", "--ledger", str(ledger.path)]) == 0
    out = capsys.readouterr().out
    assert "rtt-m1-x" in out and "1 jobs" in out and "est $0.38" in out
    # a run without settings fails cleanly, naming the missing variables
    assert cloud_main(["run", "m1", "--dry-run"]) == 2
    assert "RTT_IMAGE" in capsys.readouterr().err
