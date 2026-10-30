"""The job entry point (rtt.runner.job) and the summary it publishes (rtt.runner.summary).

One small real search (two candidates, two conditions; about 20 s) runs through the entry point
against a local folder that stands for the bucket mount; the rest check the refusals and the
summary's shape. Also the Lambda handler's event routing and the local server's wiring."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from rtt.runner import lambda_handler
from rtt.runner.job import JobPool, run_job
from rtt.runner.summary import SUMMARY_FORMAT, summarize_report
from studio_fixtures import cup, gripper, two_press_task

RUN = "abcdefghijklmnopqrstuvwxyz234567"
SMALL = {
    "candidates": [gripper(), cup()],
    "library": False,
    "n_conditions": 2,
    "n_screen": 1,
    "n_finalists": 2,
    "kinds": ["single"],
}


def _write_plan(run_dir: Path, plan: dict[str, Any]) -> str:
    raw = json.dumps(plan, sort_keys=True).encode()
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "input" / "plan.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], int]:
    run_dir = tmp_path_factory.mktemp("bucket") / "runs" / RUN
    sha = _write_plan(run_dir, two_press_task().to_dict())
    manifest, status = run_job(run_dir, sha, workers=1, search_kwargs=SMALL)
    return run_dir, manifest, status


def test_the_job_publishes_report_summary_and_manifest_last(
    published: tuple[Path, dict[str, Any], int],
) -> None:
    run_dir, manifest, status = published
    assert status == 0 and manifest["exit_status"] == 0
    assert {"report.json", "summary.json", "manifest.json"} <= set(manifest["files"])
    assert manifest["inputs"]["task_id"] == two_press_task().task_id
    on_disk = json.loads((run_dir / "manifest.json").read_text())
    assert on_disk["exit_status"] == 0 and on_disk["run_id"] == RUN
    report = json.loads((run_dir / "report.json").read_text())
    assert report["format"] == "rtt-studio-tournament" and report["task_id"] == two_press_task().task_id


def test_the_summary_has_every_field_the_site_reads(published: tuple[Path, dict[str, Any], int]) -> None:
    run_dir, _, _ = published
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["format"] == SUMMARY_FORMAT and s["run_id"] == RUN and s["status"] == "done"
    assert s["outcome"] == "recommended"
    assert set(s["pick"]) == {"setup_id", "label", "kind", "tools"} and s["pick"]["kind"] == "single"
    assert s["completion"]["n"] == 2 and 0 <= s["completion"]["k"] <= 2
    assert isinstance(s["why"], str) and s["why"]
    assert s["baseline"]["label"] and set(s["baseline"]["completion"]) == {"k", "n"}
    assert s["ruled_out"]["candidates"] == 2 and s["ruled_out"]["finalists"] == 2
    assert [r["step"] for r in s["steps"]] == [st.id for st in two_press_task().steps]
    assert all(r["tool"] == s["pick"]["tools"][0] and 0 <= r["success_rate"] <= 1 for r in s["steps"])
    assert s["code_version"]
    assert len(json.dumps(s)) < 8000  # compact


def test_summary_without_a_winner(published: tuple[Path, dict[str, Any], int]) -> None:
    run_dir, _, _ = published
    report = json.loads((run_dir / "report.json").read_text())
    rec = report["recommendation"]
    closest = copy.deepcopy(rec["ranking"][0])
    closest["completion_rate"] = 0.0
    rec.update(
        recommendation=None,
        no_winner={"reason": "no setup completed the task", "closest": [closest]},
        baseline={"setup": None, "basis": "no single tool is eligible"},
        best_single=None,
    )
    s = summarize_report(report, run_id=RUN, code_version="v")
    assert s["outcome"] == "no_recommendation" and s["pick"] is None
    assert s["closest"]["setup_id"] == closest["setup_id"] and s["completion"]["k"] == 0
    assert s["why"] == "no setup completed the task" and s["baseline"] is None
    assert s["baseline_note"] == "no single tool is eligible" and len(s["steps"]) == 2


def test_a_plan_that_does_not_match_its_hash_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / RUN
    _write_plan(run_dir, two_press_task().to_dict())
    manifest, status = run_job(run_dir, "0" * 64, workers=1, search_kwargs=SMALL)
    assert status != 0 and "SHA-256" in manifest["error"]
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "failed" and "SHA-256" in s["error"] and s["run_id"] == RUN
    assert not (run_dir / "report.json").exists()


def test_an_invalid_plan_is_refused_in_the_job_too(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / RUN
    plan = two_press_task().to_dict()
    plan["steps"][0]["object"] = "ghost"
    sha = _write_plan(run_dir, plan)
    manifest, status = run_job(run_dir, sha, workers=1, search_kwargs=SMALL)
    assert status != 0 and "checks" in manifest["error"]


def test_a_missing_plan_is_refused(tmp_path: Path) -> None:
    manifest, status = run_job(tmp_path / "runs" / RUN, "0" * 64, workers=1)
    assert status != 0 and "missing" in manifest["error"]


def test_the_job_pool_is_not_capped_at_six() -> None:
    assert JobPool(32).workers == 32 and JobPool(0).workers == 1


# --------------------------------------------------------------------------- lambda handler


class _StubRunner:
    def __init__(self) -> None:
        self.events: list[str] = []

    def sweep(self) -> dict[str, int]:
        self.events.append("sweep")
        return {"active": 0, "refreshed": 0}

    def handle_event(self, event: Any, context: Any = None) -> dict[str, Any]:
        self.events.append("http")
        return {"statusCode": 200, "headers": {}, "body": "{}"}


def test_lambda_handler_routes_sweeps_and_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubRunner()
    monkeypatch.setattr(lambda_handler, "_RUNNER", stub)
    assert lambda_handler.handler({"runner_task": "sweep"}) == {"active": 0, "refreshed": 0}
    assert (
        lambda_handler.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/health"})[
            "statusCode"
        ]
        == 200
    )
    assert lambda_handler.handler({"something": "else"})["statusCode"] == 400
    # a Function URL request cannot pose as the schedule: its fields arrive inside the body
    lambda_handler.handler({"requestContext": {"http": {}}, "body": json.dumps({"runner_task": "sweep"})})
    assert stub.events == ["sweep", "http", "http"]


def test_build_runner_needs_a_table() -> None:
    with pytest.raises(RuntimeError, match="RUNNER_TABLE_NAME"):
        lambda_handler.build_runner({})
    runner = lambda_handler.build_runner({"RUNNER_TABLE_NAME": "t", "RUNNER_KILL_SWITCH": "on"})
    assert runner.cfg.table_name == "t" and runner.cfg.emit_cors is False


# --------------------------------------------------------------------------- rtt runner serve


def test_local_fake_server_round_trip() -> None:
    from http.server import ThreadingHTTPServer

    from rtt.runner.local import _handler_for, build_local_runner

    runner = build_local_runner(fake=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(runner))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/health", headers={"Origin": "http://localhost:4341"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert (
                resp.status == 200 and resp.headers["Access-Control-Allow-Origin"] == "http://localhost:4341"
            )
            assert json.loads(resp.read())["ok"] is True
        body = json.dumps({"description": "Open a drawer.", "provider": "free", "key": None}).encode()
        req = urllib.request.Request(
            base + "/plan", data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            plan = json.loads(resp.read())
        assert plan["status"] == "valid"
        req = urllib.request.Request(
            base + "/runs", data=json.dumps({"plan": plan["plan"]}).encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 202 and json.loads(resp.read())["status"] == "queued"
    finally:
        server.shutdown()
        server.server_close()


def test_rtt_runner_is_a_cli_command() -> None:
    from rtt.cli import PASSTHROUGH, build_parser

    assert "runner" in PASSTHROUGH
    assert "runner" in build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]


def test_the_job_checks_the_plan_with_the_runners_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_MAX_STEPS", "1")
    run_dir = tmp_path / "runs" / RUN
    sha = _write_plan(run_dir, two_press_task().to_dict())  # two steps
    manifest, status = run_job(run_dir, sha, workers=1, search_kwargs=SMALL)
    assert status != 0 and "at most 1" in manifest["error"]


def test_lambda_handler_turns_any_failure_into_json_500(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(environ: Any = None) -> Any:
        raise RuntimeError("detail that must not be logged")

    monkeypatch.setattr(lambda_handler, "_RUNNER", None)
    monkeypatch.setattr(lambda_handler, "build_runner", broken)
    monkeypatch.setattr(lambda_handler, "configure_logging", lambda: None)
    with caplog.at_level("INFO", logger="rtt.runner"):
        out = lambda_handler.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/health"})
        sweep = lambda_handler.handler({"runner_task": "sweep"})
    assert out["statusCode"] == 500 and json.loads(out["body"]) == {"error": "internal error"}
    assert sweep == {"error": "RuntimeError"}
    assert "must not be logged" not in caplog.text and "RuntimeError" in caplog.text
