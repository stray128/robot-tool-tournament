"""The integrated ``rtt`` commands: search, certify, heldout, export-web, job, calibrate (parsing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtt.cli import build_parser, main


def test_every_command_is_registered() -> None:
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "cmd")
    assert {
        "m1",
        "calibrate",
        "search",
        "replay-llm",
        "heldout",
        "certify",
        "slice",
        "export-web",
        "job",
        "cloud",
    } <= set(sub.choices)
    args = parser.parse_args(["search", "--arm", "nemotron", "--runs", "2", "--out", "x"])
    assert (args.budget, args.executor, args.spend_cap, args.cache_mode) == (40, "cpu", 2.0, "readwrite")
    args = parser.parse_args(["slice"])
    assert args.spend_cap == pytest.approx(0.20) and args.baseline == "stock"
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "--arm", "tpe", "--runs", "1", "--out", "x"])


def test_search_random_arm_end_to_end(tmp_path: Path, capsys) -> None:
    out = tmp_path / "search"
    assert (
        main(
            ["search", "--arm", "random", "--runs", "1", "--budget", "8", "--out", str(out), "--workers", "4"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "random-r00 (seed 1): best" in text
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["sim"]["episodes_simulated"] == 8 * 48
    assert (out / "frozen" / "random-r00.json").is_file()


def test_certify_writes_a_certificate_into_a_run_dir(tmp_path: Path, capsys) -> None:
    status = main(
        [
            "certify",
            "--entrant",
            "stock",
            "--object",
            "box:80x50x110",
            "--k",
            "0",
            "--workers",
            "4",
            "--out",
            str(tmp_path),
        ]
    )
    assert status == 0
    files = list((tmp_path / "certificates").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["schema"] == "rtt.certificates/1" and len(doc["certificates"][0]["tests"]) == 18
    assert "/18 passed" in capsys.readouterr().out
    assert doc["certificates"][0]["passed"].endswith("/18")


def test_certify_rejects_a_bad_object(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["certify", "--entrant", "stock", "--object", "sphere:10"])


def test_heldout_refuses_to_run_without_a_commitment(tmp_path: Path, capsys) -> None:
    seed = tmp_path / "seed.txt"
    seed.write_text("ab" * 32)
    status = main(
        [
            "heldout",
            "--seed-file",
            str(seed),
            "--out",
            str(tmp_path / "o"),
            "--commitment-file",
            str(tmp_path / "none"),
        ]
    )
    assert status == 2 and "no commitment" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_export_web_skips_unknown_directories(tmp_path: Path, capsys) -> None:
    (tmp_path / "empty").mkdir()
    assert main(["export-web", str(tmp_path / "empty"), "--data-dir", str(tmp_path / "data")]) == 1
    assert "skipped" in capsys.readouterr().out


def test_job_search_needs_its_arguments(capsys) -> None:
    assert main(["job", "search"]) == 2
    assert "needs --arm, --runs and --budget" in capsys.readouterr().out


# --------------------------------------------------------------------------- integration wiring


def test_passthrough_commands_are_listed_and_dispatched(monkeypatch, capsys) -> None:
    """`rtt part|agent|serve|heldout-real|analyze-s1` hand every argument to the module's parser."""
    from rtt import cli

    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "cmd")
    assert {"part", "agent", "serve", "heldout-real", "analyze-s1"} <= set(sub.choices)

    seen: dict[str, list[str]] = {}

    def fake(name: str):
        def handler(ns) -> int:
            seen[name] = list(ns.rest)
            return 7

        return handler

    monkeypatch.setattr(cli, "PASSTHROUGH", {k: (fake(k), v[1]) for k, v in cli.PASSTHROUGH.items()})
    assert main(["part", "run", "--help", "x.stl"]) == 7
    assert main(["serve", "--port", "9000"]) == 7
    assert main(["heldout-real", "select", "--mode", "dev", "--out", "o"]) == 7
    assert seen == {
        "part": ["run", "--help", "x.stl"],
        "serve": ["--port", "9000"],
        "heldout-real": ["select", "--mode", "dev", "--out", "o"],
    }


def test_part_help_uses_the_rtt_program_name(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["part", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: rtt part") and "certify" in out and "printpack" in out


def test_job_search_defaults_to_the_uk_south2_cpu_target() -> None:
    """No --target: the cloud CLI's default (uks2-cpu) applies; the flags pass through when given."""
    from rtt.cli import job_argv

    parser = build_parser()
    args = parser.parse_args(["job", "search", "--arm", "random", "--runs", "3", "--budget", "40"])
    argv = job_argv(args)
    assert argv is not None and argv[:2] == ["run", "search"]
    assert "--target" not in argv and "--platform" not in argv and "--no-fallback" not in argv
    from rtt.cloud.nebius_jobs import DEFAULT_TARGET

    assert DEFAULT_TARGET == "uks2-cpu"
    args = parser.parse_args(
        ["job", "search", "--arm", "random", "--runs", "3", "--budget", "40", "--target", "uks2-cpu"]
        + ["--vcpus", "32", "--no-fallback", "--workers", "30", "--seeds", "1,2,3"]
    )
    argv = job_argv(args)
    assert argv is not None
    joined = " ".join(argv)
    assert "--target uks2-cpu" in joined and "--vcpus 32" in joined and "--no-fallback" in joined
    assert "--workers 30" in joined and "--seeds 1,2,3" in joined
