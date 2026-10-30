"""The hosted runner's API (rtt.runner.app): every route and status code, the caps and their
resets, the kill switch and auto-off, CORS, server-side plan checks and the run lifecycle through
fake job states. No network: the model, Nebius and DynamoDB are fakes (tests/runner_support.py)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date, timedelta

import pytest

from rtt.runner.app import RUN_ID, new_run_id
from rtt.runner.caps import job_cost_micros, reservation_micros
from rtt.runner.config import Caps, RunnerConfig, to_usd
from runner_support import FREE_KEY, ORIGIN, VISITOR_KEY, Harness, good_args, good_plan, summary_for

# --------------------------------------------------------------------------- health and routing


def test_health_reports_the_contract_fields() -> None:
    h = Harness()
    status, body, headers = h.call("GET", "/health")
    assert status == 200
    assert set(body) == {
        "ok",
        "kill_switch",
        "auto_off",
        "plans_left_today",
        "runs_left_today",
        "spend_today_usd",
        "spend_total_usd",
        "open",
    }
    assert body["ok"] is True and body["kill_switch"] is False and body["auto_off"] == "2026-12-15"
    assert body["open"] is True
    assert body["plans_left_today"] == 20
    # $2 a day over a reservation of (20 + 5) min at $1.056/h = $0.44: four runs fit
    assert body["runs_left_today"] == 4
    assert body["spend_today_usd"] == 0.0 and body["spend_total_usd"] == 0.0
    assert headers["Content-Type"] == "application/json" and headers["Cache-Control"] == "no-store"


def test_unknown_routes_and_methods() -> None:
    h = Harness()
    assert h.call("GET", "/nope")[0] == 404
    assert h.call("GET", "/")[0] == 404
    status, _, headers = h.call("POST", "/health")
    assert status == 405 and headers["Allow"] == "GET"
    assert h.call("GET", "/plan")[0] == 405
    assert h.call("GET", "/runs")[0] == 405
    assert h.call("POST", "/runs/" + "a" * 32)[0] == 405


# --------------------------------------------------------------------------- POST /plan


def test_free_plan_returns_the_engineers_plan() -> None:
    h = Harness()
    status, body = h.plan("free")
    assert status == 200
    assert set(body) == {"status", "plan", "problems", "calls", "cost_usd", "model"}
    assert body["status"] == "valid" and body["problems"] == []
    assert body["calls"] == 1 and body["cost_usd"] > 0
    assert body["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert body["plan"]["task_id"] == good_plan()["task_id"]
    # the free tier used our key, on Token Factory, with the current prompt
    (model,) = h.models
    assert model.key == FREE_KEY
    assert "submit_task_plan" in json.dumps(model.payloads[0]["tools"])
    llm = h.store.numbers("spend#llm#day#2026-10-28")
    assert llm["micros"] > 0


def test_plan_with_the_repair_loop_and_a_failure() -> None:
    bad = good_args()
    bad["steps"][0]["object"] = "no_such_object"
    h = Harness(answers=[bad, good_args()])
    status, body = h.plan("tokenfactory", VISITOR_KEY)
    assert status == 200 and body["status"] == "valid" and body["calls"] == 2
    h2 = Harness(answers=[bad, bad])
    status, body = h2.plan("nvidia", VISITOR_KEY)
    assert status == 200 and body["status"] == "failed" and body["plan"] is None and body["problems"]


def test_refused_plan() -> None:
    refusal = {
        "title": "Fold a shirt",
        "description": "Fold a shirt.",
        "objects": [],
        "steps": [],
        "assumptions": [],
        "clarifications": [],
        "unsupported": [
            {"what": "Folding cloth", "reason": "Cloth is deformable.", "nearest_primitive": "grasp"}
        ],
        "refusal": "Cloth cannot be simulated here.",
    }
    h = Harness(answers=[refusal])
    status, body = h.plan("free")
    assert status == 200 and body["status"] == "refused" and body["plan"]["refusal"]


def test_providers_use_their_endpoints() -> None:
    h = Harness()
    h.plan("nvidia", VISITOR_KEY)
    h.plan("tokenfactory", VISITOR_KEY)
    assert [m.key for m in h.models] == [VISITOR_KEY, VISITOR_KEY]
    assert h.cfg.providers["nvidia"].base_url == "https://integrate.api.nvidia.com/v1"
    assert h.cfg.providers["tokenfactory"].base_url.startswith("https://api.tokenfactory.nebius.com")


@pytest.mark.parametrize(
    "body",
    [
        {"description": "", "provider": "free", "key": None},
        {"description": "   ", "provider": "free", "key": None},
        {"description": "x" * 4001, "provider": "free", "key": None},
        {"description": "ok", "provider": "openai", "key": None},
        {"description": "ok", "provider": "free", "key": "abcdefgh123"},
        {"description": "ok", "provider": "nvidia", "key": None},
        {"description": "ok", "provider": "nvidia", "key": "short"},
        {"description": "ok", "provider": "nvidia", "key": "has a space in it"},
        {"description": "ok", "provider": "nvidia", "key": "k" * 600},
        {"description": "ok", "provider": "free", "key": None, "extra": 1},
        {"description": 5, "provider": "free", "key": None},
        [1, 2],
    ],
)
def test_plan_bad_input_is_400(body: object) -> None:
    h = Harness()
    status, payload, _ = h.call("POST", "/plan", body)
    assert status == 400 and payload["error"]
    assert h.models == []  # nothing was called


def test_plan_body_that_is_not_json_or_too_large() -> None:
    h = Harness()
    assert h.call("POST", "/plan", raw=b"{not json")[0] == 400
    assert h.call("POST", "/plan", raw=b"\xff\xfe")[0] == 400
    big = json.dumps({"description": "x", "provider": "free", "key": None, "pad": "y" * 70000}).encode()
    status, body, _ = h.call("POST", "/plan", raw=big)
    assert status == 400 and "larger" in body["error"]


def test_a_rejected_key_is_401_and_a_rejected_free_key_is_503() -> None:
    h = Harness(answers=[401])
    status, body = h.plan("nvidia", VISITOR_KEY)
    assert status == 401 and body == {"error": "the provider rejected this key"}
    h = Harness(answers=[402])
    status, body = h.plan("tokenfactory", VISITOR_KEY)
    assert status == 401 and "billing" in body["error"]
    h = Harness(answers=[401])
    status, body = h.plan("free")
    assert status == 503


def test_provider_outage_is_a_failed_plan_not_an_error() -> None:
    h = Harness(answers=[500, 500, 500, 500])
    status, body = h.plan("tokenfactory", VISITOR_KEY)
    assert status == 200 and body["status"] == "failed" and body["plan"] is None


# --------------------------------------------------------------------------- caps and resets


def test_free_plan_caps_per_ip_then_site_wide_and_their_reset() -> None:
    h = Harness()
    for _ in range(3):
        assert h.plan("free", ip="198.51.100.1")[0] == 200
    status, body = h.plan("free", ip="198.51.100.1")
    assert status == 429
    assert body["cap"] == "free_plans_per_ip_per_day" and body["limit"] == 3
    assert body["resets_at"] == "2026-10-29T00:00:00Z"
    # 17 more from other addresses reach the site-wide 20
    for i in range(17):
        assert h.plan("free", ip=f"198.51.100.{10 + i}")[0] == 200
    status, body = h.plan("free", ip="198.51.100.200")
    assert status == 429 and body["cap"] == "free_plans_per_day" and body["limit"] == 20
    assert h.call("GET", "/health")[1]["plans_left_today"] == 0
    # a refused plan never reaches the model
    assert len(h.models) == 20
    # the keyed path is a separate cap
    assert h.plan("nvidia", VISITOR_KEY, ip="198.51.100.200")[0] == 200
    # next UTC day: everything resets
    h.advance(15 * 3600)
    assert h.call("GET", "/health")[1]["plans_left_today"] == 20
    assert h.plan("free", ip="198.51.100.1")[0] == 200


def test_keyed_plans_are_capped_per_ip() -> None:
    h = Harness(replace(RunnerConfig(), caps=replace(Caps(), keyed_plans_per_ip_per_day=2)))
    assert h.plan("nvidia", VISITOR_KEY)[0] == 200
    assert h.plan("tokenfactory", VISITOR_KEY)[0] == 200
    status, body = h.plan("nvidia", VISITOR_KEY)
    assert status == 429 and body["cap"] == "keyed_plans_per_ip_per_day"


def test_runs_per_ip_and_spend_caps() -> None:
    h = Harness()
    reserve = reservation_micros(h.cfg)
    assert to_usd(reserve) == pytest.approx(0.44)
    assert h.start_run(ip="192.0.2.1")[0] == 202
    assert h.start_run(ip="192.0.2.1")[0] == 202
    status, body = h.start_run(ip="192.0.2.1")
    assert status == 429 and body["cap"] == "runs_per_ip_per_day" and body["limit"] == 2
    # two more addresses: four reservations of $0.44 fit in $2, the fifth does not
    assert h.start_run(ip="192.0.2.2")[0] == 202
    assert h.start_run(ip="192.0.2.2")[0] == 202
    status, body = h.start_run(ip="192.0.2.3")
    assert (
        status == 429 and body["cap"] == "spend_per_day_usd" and body["resets_at"] == "2026-10-29T00:00:00Z"
    )
    health = h.call("GET", "/health")[1]
    assert health["runs_left_today"] == 0 and health["spend_today_usd"] == pytest.approx(1.76)
    assert len(h.nebius.submissions) == 4


def test_the_total_spend_cap_never_resets() -> None:
    caps = replace(Caps(), spend_total_usd=1.0, runs_per_ip_per_day=10)
    h = Harness(replace(RunnerConfig(), caps=caps))
    assert h.start_run()[0] == 202
    assert h.start_run()[0] == 202
    status, body = h.start_run()
    assert status == 429 and body["cap"] == "spend_total_usd" and body["resets_at"] is None
    h.advance(86400 * 3)
    status, body = h.start_run()
    assert status == 429 and body["cap"] == "spend_total_usd"


def test_settling_returns_the_unused_reservation() -> None:
    h = Harness()
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.advance(600)
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[run["run_id"]] = ({"exit_status": 0}, summary_for(run["run_id"], good_plan()))
    status, body, _ = h.call("GET", f"/runs/{run['run_id']}")
    assert status == 200 and body["status"] == "done"
    cost = job_cost_micros(h.cfg, 600)
    assert body["cost_estimate_usd"] == to_usd(cost) == pytest.approx(0.176)
    health = h.call("GET", "/health")[1]
    assert health["spend_today_usd"] == to_usd(cost) and health["spend_total_usd"] == to_usd(cost)
    day = h.store.numbers("spend#nebius#day#2026-10-28")
    assert day["settled"] == cost and day["committed"] == cost


# --------------------------------------------------------------------------- kill switch, auto-off


def test_kill_switch_from_ssm_and_env() -> None:
    h = Harness()
    h.params["/test/kill"] = "on"
    assert h.plan("free")[0] == 503
    assert h.start_run()[0] == 503
    status, body, _ = h.call("GET", "/health")
    assert status == 200 and body["kill_switch"] is True and body["open"] is False
    assert body["plans_left_today"] == 0 and body["runs_left_today"] == 0
    h.params["/test/kill"] = "off"
    assert h.plan("free")[0] == 200
    env = Harness(replace(RunnerConfig(), kill_switch=True))
    assert env.plan("free")[0] == 503 and env.call("GET", "/health")[1]["kill_switch"] is True


def test_an_unreadable_kill_switch_fails_closed() -> None:
    h = Harness()
    del h.params["/test/kill"]
    assert h.plan("free")[0] == 503
    assert h.call("GET", "/health")[1]["kill_switch"] is True


def test_kill_switch_cancels_runs_in_flight() -> None:
    h = Harness()
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.params["/test/kill"] = "on"
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]
    _, body, _ = h.call("GET", f"/runs/{run['run_id']}")
    assert body["status"] == "failed" and "switched off" in body["error"]


def test_auto_off_after_the_date() -> None:
    h = Harness(replace(RunnerConfig(), auto_off=date(2026, 10, 28)))
    assert h.plan("free")[0] == 200  # the last day still serves
    h.advance(15 * 3600)  # 2026-10-29
    status, body = h.plan("free")
    assert status == 503 and "2026-10-28" in body["error"]
    assert h.start_run()[0] == 503
    health = h.call("GET", "/health")[1]
    assert health["kill_switch"] is False and health["open"] is False and health["plans_left_today"] == 0


# --------------------------------------------------------------------------- CORS


def test_cors_only_for_the_allowed_origins() -> None:
    local = Harness(replace(RunnerConfig(), emit_cors=True))
    status, _, headers = local.call("GET", "/health", origin=ORIGIN)
    assert status == 200 and headers["Access-Control-Allow-Origin"] == ORIGIN and headers["Vary"] == "Origin"
    status, _, headers = local.call("OPTIONS", "/plan", origin="http://localhost:4341")
    assert status == 204 and headers["Access-Control-Allow-Origin"] == "http://localhost:4341"
    assert "POST" in headers["Access-Control-Allow-Methods"]
    status, _, headers = local.call("POST", "/plan", {"description": "x"}, origin="https://evil.example")
    assert status == 403 and "Access-Control-Allow-Origin" not in headers
    assert local.call("GET", "/health", origin="null")[0] == 403
    # no Origin (curl, the smoke test): allowed, no CORS headers
    status, _, headers = local.call("GET", "/health", origin=None)
    assert status == 200 and "Access-Control-Allow-Origin" not in headers


def test_behind_a_function_url_the_function_sends_no_cors_headers() -> None:
    h = Harness()  # emit_cors defaults to False: the Function URL's CORS config sends them
    status, _, headers = h.call("GET", "/health", origin=ORIGIN)
    assert status == 200 and not any(k.startswith("Access-Control") for k in headers)
    assert h.call("GET", "/health", origin="https://evil.example")[0] == 403


def test_default_origins_are_the_site_and_its_dev_server() -> None:
    assert RunnerConfig().allowed_origins == ("https://stray128.github.io", "http://localhost:4341")


# --------------------------------------------------------------------------- POST /runs validation


def test_run_ids_are_160_bit_url_safe() -> None:
    ids = {new_run_id() for _ in range(200)}
    assert len(ids) == 200 and all(RUN_ID.match(i) for i in ids)


def test_a_valid_plan_starts_a_job_with_the_server_parsed_plan() -> None:
    h = Harness()
    plan = good_plan()
    plan["source"] = "nemotron"
    status, body = h.start_run(plan)
    assert status == 202 and set(body) == {"run_id", "status", "eta_s"}
    assert body["status"] == "queued" and body["eta_s"] == 480 and RUN_ID.match(body["run_id"])
    ((run_id, sent),) = h.nebius.submissions
    assert run_id == body["run_id"]
    parsed = json.loads(sent)
    assert parsed["task_id"] == plan["task_id"] and parsed["source"] == "manual"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p["steps"][0].update(object="ghost"), "ghost"),
        (lambda p: p.update(steps=[]), "nothing to run"),
        (lambda p: p.update(refusal="no"), "nothing to run"),
        (lambda p: p.update(extra="x"), "unknown fields"),
        (lambda p: p.update(contract="rtt-effectors/9"), "contract"),
        (lambda p: p["objects"][0].update(ref="upload:abc123"), "uploaded parts"),
        (lambda p: p.update(objects=p["objects"] * 3), "at most 6"),
        (lambda p: p.update(steps=p["steps"] * 3), "at most 8"),
        (lambda p: p.update(objects="nope"), "must be lists"),
        (lambda p: p.update(description="d" * 60000), "bytes"),
    ],
)
def test_bad_plans_are_refused_server_side(mutate: object, needle: str) -> None:
    h = Harness(replace(RunnerConfig(), max_body_bytes=256 * 1024))
    plan = copy.deepcopy(good_plan())
    mutate(plan)  # type: ignore[operator]
    status, body = h.start_run(plan)
    assert status == 400, body
    text = json.dumps(body)
    assert needle in text
    assert h.nebius.submissions == []
    assert h.store.numbers("spend#nebius#day#2026-10-28") == {}


def test_runs_body_must_hold_only_a_plan() -> None:
    h = Harness()
    assert h.call("POST", "/runs", {"plan": good_plan(), "priority": 1})[0] == 400
    assert h.call("POST", "/runs", {})[0] == 400
    assert h.call("POST", "/runs", {"plan": "text"})[0] == 400


def test_runs_without_nebius_settings_are_503_and_charge_nothing() -> None:
    cfg = RunnerConfig()
    h = Harness(cfg)
    h.runner.cfg = replace(h.runner.cfg, nebius=replace(h.runner.cfg.nebius, image=None))
    status, body = h.start_run()
    assert status == 503 and "not configured" in body["error"]
    assert h.store.numbers("spend#nebius#total") == {}


def test_a_failed_submission_gives_everything_back() -> None:
    h = Harness()
    h.nebius.fail_submit = True
    status, body = h.start_run(ip="192.0.2.9")
    assert status == 503 and "could not be started" in body["error"]
    assert h.store.numbers("spend#nebius#day#2026-10-28")["committed"] == 0
    assert h.store.numbers("spend#nebius#total")["committed"] == 0
    h.nebius.fail_submit = False
    assert h.start_run(ip="192.0.2.9")[0] == 202
    assert h.start_run(ip="192.0.2.9")[0] == 202  # the failed one did not use up a run


def test_a_job_created_despite_a_failed_submission_stays_tracked_and_counted() -> None:
    h = Harness()
    h.nebius.fail_after_create = True
    status, body = h.start_run(ip="192.0.2.10")
    assert status == 202 and body["status"] == "queued"
    (job_id,) = h.nebius.jobs
    record, _ = h.store.get_run(body["run_id"])  # type: ignore[misc]
    assert record["job_id"] == job_id and record["settled"] is False
    assert h.store.numbers("spend#nebius#total")["committed"] == reservation_micros(h.cfg)
    assert [r["run_id"] for r, _ in h.store.active_runs()] == [body["run_id"]]


# --------------------------------------------------------------------------- the run lifecycle


def test_run_lifecycle_queued_running_done() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    (job_id,) = h.nebius.jobs
    status, body, _ = h.call("GET", f"/runs/{rid}")
    assert status == 200
    assert set(body) == {"status", "created_at", "cost_estimate_usd", "error", "result"}
    assert (
        body["status"] == "queued" and body["result"] is None and body["created_at"] == "2026-10-28T09:00:00Z"
    )
    h.advance(120)
    h.nebius.set_state(job_id, "RUNNING")
    status, body, _ = h.call("GET", f"/runs/{rid}")
    assert body["status"] == "running" and body["cost_estimate_usd"] == to_usd(job_cost_micros(h.cfg, 120))
    h.advance(300)
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[rid] = ({"exit_status": 0}, summary_for(rid, good_plan()))
    status, body, _ = h.call("GET", f"/runs/{rid}")
    assert body["status"] == "done" and body["error"] is None
    result = body["result"]
    assert result["pick"]["label"] == "Gripper" and result["completion"] == {"k": 8, "n": 8}
    assert result["report_url"].endswith("/report.json?expires=900")
    assert [s["step"] for s in result["steps"]] == [s["id"] for s in good_plan()["steps"]]
    # settled once: polling again changes nothing
    before = h.store.numbers("spend#nebius#total")
    h.advance(60)
    h.call("GET", f"/runs/{rid}")
    h.runner.sweep()
    assert h.store.numbers("spend#nebius#total") == before
    assert h.store.active_runs() == []
    # the report redirect
    status, body, headers = h.call("GET", f"/runs/{rid}/report")
    assert status == 302 and headers["Location"] == body["report_url"]


def test_polling_is_rate_limited() -> None:
    h = Harness()
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.call("GET", f"/runs/{run['run_id']}")
    h.nebius.set_state(job_id, "RUNNING")
    h.advance(3)
    assert h.call("GET", f"/runs/{run['run_id']}")[1]["status"] == "queued"  # within 10 s: stored state
    h.advance(10)
    assert h.call("GET", f"/runs/{run['run_id']}")[1]["status"] == "running"


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "ERROR"])
def test_a_job_that_ends_badly_is_a_failed_run_with_its_cost(state: str) -> None:
    h = Harness()
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.advance(200)
    h.nebius.set_state(job_id, state)
    _, body, _ = h.call("GET", f"/runs/{run['run_id']}")
    assert body["status"] == "failed" and state.lower() in body["error"] and body["result"] is None
    assert body["cost_estimate_usd"] == to_usd(job_cost_micros(h.cfg, 200))
    assert h.call("GET", f"/runs/{run['run_id']}/report")[0] == 404


def test_a_failed_search_reports_its_own_error() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "FAILED")
    h.nebius.results[rid] = (
        {"exit_status": 1},
        {
            "format": "effector-lab-run-summary",
            "run_id": rid,
            "status": "failed",
            "error": "the search passed its time limit",
        },
    )
    body = h.call("GET", f"/runs/{rid}")[1]
    assert body["status"] == "failed" and "time limit" in body["error"]


def test_a_completed_job_without_a_valid_summary_is_failed() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[rid] = (
        {"exit_status": 0},
        {"format": "effector-lab-run-summary", "run_id": "other", "status": "done"},
    )
    body = h.call("GET", f"/runs/{rid}")[1]
    assert body["status"] == "failed" and "summary" in body["error"]


def test_the_wall_time_limit_cancels_the_job_then_settles_when_it_stops() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    h.advance(1199)
    h.runner.sweep()
    assert h.nebius.cancels == []
    h.advance(2)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]
    body = h.call("GET", f"/runs/{rid}")[1]
    assert body["status"] == "failed" and "20-minute" in body["error"]
    assert len(h.store.active_runs()) == 1  # still tracked until the job stops
    h.advance(60)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]  # cancelled once
    h.nebius.set_state(job_id, "CANCELLED")
    h.runner.sweep()
    assert h.store.active_runs() == []
    total = h.store.numbers("spend#nebius#total")
    assert total["settled"] == job_cost_micros(h.cfg, 1261) == total["committed"]


def test_a_job_that_never_reports_is_given_up_after_the_platform_timeout() -> None:
    h = Harness()
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    h.advance(1300)
    h.runner.sweep()
    h.advance(3600)
    h.runner.sweep()
    assert h.store.active_runs() == []
    body = h.call("GET", f"/runs/{run['run_id']}")[1]
    assert body["status"] == "failed"


def test_a_run_whose_job_id_was_lost_is_found_by_name_or_released() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    record, _ = h.store.get_run(rid)  # type: ignore[misc]
    record["job_id"] = None  # as if the function died between submit and save
    h.store.items[f"run#{rid}"]["doc"] = json.dumps(record)
    h.advance(h.cfg.unsubmitted_grace_s - 10)  # the submitting request may still be alive: no change
    h.runner.sweep()
    assert h.store.get_run(rid)[0]["job_id"] is None  # type: ignore[index]
    h.advance(20)
    h.runner.sweep()
    record, _ = h.store.get_run(rid)  # type: ignore[misc]
    assert record["job_id"] == next(iter(h.nebius.jobs))
    # a record whose job was never created is released
    h2 = Harness()
    h2.nebius.fail_submit = False
    _, run2 = h2.start_run()
    h2.nebius.jobs.clear()
    rec2, _ = h2.store.get_run(run2["run_id"])  # type: ignore[misc]
    rec2["job_id"] = None
    h2.store.items[f"run#{run2['run_id']}"]["doc"] = json.dumps(rec2)
    h2.advance(h2.cfg.unsubmitted_grace_s + 10)
    h2.runner.sweep()
    body = h2.call("GET", f"/runs/{run2['run_id']}")[1]
    assert body["status"] == "failed" and body["cost_estimate_usd"] == 0.0
    assert h2.store.numbers("spend#nebius#total")["committed"] == 0


def test_unknown_or_malformed_run_ids_are_404() -> None:
    h = Harness()
    assert h.call("GET", "/runs/" + "a" * 32)[0] == 404
    assert h.call("GET", "/runs/../../etc")[0] == 404
    assert h.call("GET", "/runs/ABC")[0] == 404
    assert h.call("GET", "/runs/" + "a" * 32 + "/report")[0] == 404


def test_a_nebius_outage_still_answers_from_the_stored_record() -> None:
    h = Harness()
    _, run = h.start_run()
    h.nebius.fail_status = True
    h.advance(30)
    status, body, _ = h.call("GET", f"/runs/{run['run_id']}")
    assert status == 200 and body["status"] == "queued"
    assert h.runner.sweep() == {"active": 1, "refreshed": 0}


def test_two_pollers_settle_a_run_once() -> None:
    h = Harness()
    _, run = h.start_run()
    rid = run["run_id"]
    (job_id,) = h.nebius.jobs
    h.advance(300)
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[rid] = ({"exit_status": 0}, summary_for(rid, good_plan()))
    stale = h.store.get_run(rid)
    assert stale is not None
    h.runner.refresh(*stale, force=True)
    h.runner.refresh(*stale, force=True)  # the second writer holds an old version: no double settle
    total = h.store.numbers("spend#nebius#total")
    assert total["settled"] == job_cost_micros(h.cfg, 300)


def test_internal_errors_are_500_without_detail() -> None:
    h = Harness()

    def boom() -> None:
        raise RuntimeError("secret detail /home/someone")

    h.runner.store.numbers = lambda key: boom()  # type: ignore[assignment,method-assign]
    status, body, _ = h.call("GET", "/health")
    assert status == 500 and body == {"error": "internal error"}


def test_config_from_env() -> None:
    cfg = RunnerConfig.from_env(
        {
            "RUNNER_FREE_PLANS_PER_DAY": "5",
            "RUNNER_SPEND_TOTAL_USD": "3.5",
            "RUNNER_AUTO_OFF": "2026-12-01",
            "RUNNER_KILL_SWITCH": "on",
            "RUNNER_ALLOWED_ORIGINS": "https://a.example, https://b.example",
            "RUNNER_NEBIUS_PROJECT_ID": "project-1",
            "RUNNER_NEBIUS_IMAGE": "",
            "RUNNER_NVIDIA_MODEL": "nvidia/other",
            "RUNNER_TOKENFACTORY_BASE_URL": "https://tf.example/v1/",
        }
    )
    assert cfg.caps.free_plans_per_day == 5 and cfg.caps.spend_total_usd == 3.5
    assert cfg.auto_off == date(2026, 12, 1) and cfg.kill_switch is True
    assert cfg.allowed_origins == ("https://a.example", "https://b.example")
    assert cfg.nebius.project_id == "project-1" and cfg.nebius.image is None
    assert cfg.providers["nvidia"].model == "nvidia/other"
    assert cfg.providers["free"].base_url == "https://tf.example/v1/"
    assert "project-1" not in repr(cfg.nebius)
    assert cfg.nebius.deadline_s == 960
    short = RunnerConfig.from_env({"RUNNER_JOB_WALL_LIMIT_S": "600"})
    assert short.caps.job_wall_limit_s == 600 and short.nebius.deadline_s == 360


def test_eta_and_cap_defaults_match_the_contract() -> None:
    caps = Caps()
    assert (caps.free_plans_per_day, caps.free_plans_per_ip_per_day, caps.runs_per_ip_per_day) == (20, 3, 2)
    assert (caps.spend_per_day_usd, caps.spend_total_usd, caps.job_wall_limit_s) == (2.0, 10.0, 1200)
    assert RunnerConfig().auto_off == date(2026, 12, 15)
    assert RunnerConfig().nebius.platform == "cpu-d3" and RunnerConfig().nebius.preset == "32vcpu-128gb"
    assert RunnerConfig().nebius.region == "uk-south2"


def test_health_after_a_day_rolls_over_counts_fresh_spend() -> None:
    h = Harness()
    h.start_run()
    h.advance(timedelta(days=1).total_seconds())
    health = h.call("GET", "/health")[1]
    assert health["spend_today_usd"] == 0.0 and health["spend_total_usd"] == pytest.approx(0.44)


# --------------------------------------------------------------------------- on a DynamoDB-shaped store


@pytest.mark.parametrize("dynamo", [False, True], ids=["memory", "dynamo"])
def test_full_lifecycle_settles_on_both_stores(dynamo: bool) -> None:
    """Create -> running -> done -> settled, with the store that refuses two actions on one item
    the way DynamoDB does: settlement must commit, release the unused reservation and drop the run
    from the active index."""
    h = Harness(dynamo=dynamo)
    h.nebius.timeline = (60.0, 400.0)
    h.nebius.summary_for = summary_for
    status, run = h.start_run()
    assert status == 202
    for _ in range(5):
        h.advance(120)
        h.runner.sweep()
    status, body, _ = h.call("GET", f"/runs/{run['run_id']}")
    assert status == 200 and body["status"] == "done"
    cost = job_cost_micros(h.cfg, 460)
    assert body["cost_estimate_usd"] == to_usd(cost)
    day = h.store.numbers("spend#nebius#day#2026-10-28")
    assert day["committed"] == cost and day["settled"] == cost
    assert h.store.numbers("spend#nebius#total")["settled"] == cost
    assert h.store.active_runs() == []
    assert h.call("GET", "/health")[1]["runs_left_today"] == (2_000_000 - cost) // reservation_micros(h.cfg)
    if dynamo:
        assert h.dynamo.rejected_duplicates == 0  # type: ignore[union-attr]


@pytest.mark.parametrize("dynamo", [False, True], ids=["memory", "dynamo"])
def test_cancel_then_settle_and_caps_on_both_stores(dynamo: bool) -> None:
    h = Harness(dynamo=dynamo)
    assert h.start_run(ip="192.0.2.1")[0] == 202
    assert h.start_run(ip="192.0.2.1")[0] == 202
    status, body = h.start_run(ip="192.0.2.1")
    assert status == 429 and body["cap"] == "runs_per_ip_per_day"
    job_ids = list(h.nebius.jobs)
    for j in job_ids:
        h.nebius.set_state(j, "RUNNING")
    h.advance(1250)
    h.runner.sweep()
    assert sorted(h.nebius.cancels) == sorted(job_ids)
    for j in job_ids:
        h.nebius.set_state(j, "CANCELLED")
    h.runner.sweep()
    assert h.store.active_runs() == []
    total = h.store.numbers("spend#nebius#total")
    assert total["committed"] == total["settled"] == 2 * job_cost_micros(h.cfg, 1250)


def test_a_store_outage_while_settling_leaves_the_run_active_for_the_next_sweep() -> None:
    h = Harness(dynamo=True)
    _, run = h.start_run()
    (job_id,) = h.nebius.jobs
    h.advance(300)
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[run["run_id"]] = ({"exit_status": 0}, summary_for(run["run_id"], good_plan()))
    h.dynamo.fail_next = 1  # type: ignore[union-attr]
    assert h.runner.sweep() == {"active": 1, "refreshed": 1}
    assert len(h.store.active_runs()) == 1  # nothing was written; still tracked
    assert h.store.get_run(run["run_id"])[0]["settled"] is False
    h.runner.sweep()  # the store is back
    assert h.store.active_runs() == []
    assert h.call("GET", f"/runs/{run['run_id']}")[1]["status"] == "done"


# --------------------------------------------------------------------------- jobs the ledger might miss


def test_an_unknown_submission_keeps_the_reservation_until_the_job_is_found() -> None:
    """The create call fails in a way that leaves the job's existence unknown and the lookup by
    name fails too: the run stays reserved and active, and the sweeper tracks the job once the
    platform answers."""
    h = Harness()
    h.nebius.fail_after_create = True
    h.nebius.fail_find = True
    status, body = h.start_run(ip="192.0.2.20")
    assert status == 202 and body["status"] == "queued"
    rec, _ = h.store.get_run(body["run_id"])  # type: ignore[misc]
    assert rec["job_id"] is None and rec["settled"] is False
    assert h.store.numbers("spend#nebius#total")["committed"] == reservation_micros(h.cfg)
    h.advance(h.cfg.unsubmitted_grace_s + 30)
    h.runner.sweep()  # the lookup still fails: unchanged
    assert h.store.get_run(body["run_id"])[0]["job_id"] is None  # type: ignore[index]
    h.nebius.fail_find = False
    h.runner.sweep()
    rec, _ = h.store.get_run(body["run_id"])  # type: ignore[misc]
    assert rec["job_id"] == next(iter(h.nebius.jobs))
    # it is now an ordinary run: past the wall limit it is cancelled
    h.advance(1300)
    h.runner.sweep()
    assert h.nebius.cancels == [rec["job_id"]]


def test_an_unknown_submission_with_no_job_is_released_after_the_grace() -> None:
    h = Harness()
    h.nebius.fail_create = True  # no job was created, but the runner cannot know
    status, body = h.start_run()
    assert status == 202
    h.advance(h.cfg.unsubmitted_grace_s + 30)
    h.runner.sweep()
    view = h.call("GET", f"/runs/{body['run_id']}")[1]
    assert view["status"] == "failed" and "never started" in view["error"] and view["cost_estimate_usd"] == 0
    assert h.store.numbers("spend#nebius#total")["committed"] == 0


def test_a_job_that_cannot_be_confirmed_for_long_is_charged_its_reservation() -> None:
    h = Harness()
    h.nebius.fail_create = True
    h.nebius.fail_find = True
    _, body = h.start_run()
    h.advance(4600)
    h.runner.sweep()
    view = h.call("GET", f"/runs/{body['run_id']}")[1]
    assert view["status"] == "failed" and "could not be confirmed" in view["error"]
    assert view["cost_estimate_usd"] == to_usd(reservation_micros(h.cfg))
    total = h.store.numbers("spend#nebius#total")
    assert total["committed"] == total["settled"] == reservation_micros(h.cfg)
    assert h.store.active_runs() == []


def test_a_create_that_times_out_after_the_job_exists_with_a_failing_lookup_stays_tracked() -> None:
    """The real backend over a CLI whose create times out after the platform created the job, and
    whose lookup by name is unavailable: the run must stay tracked (reserved, active), not be
    released as never started."""
    from rtt.cloud.nebius_jobs import CommandResult, NebiusCLI
    from rtt.runner.nebius import NebiusBackend

    created: list[list[str]] = []
    lookups = {"ok": False}

    def cli(argv: list[str], timeout: float | None) -> CommandResult:
        if argv[1:4] == ["ai", "job", "create"]:
            created.append(list(argv))
            return CommandResult(124, "", "timed out after 60.0 s")
        if "get-by-name" in argv:
            if not lookups["ok"]:
                return CommandResult(1, "", "rpc error: code = Unavailable")
            job = {
                "metadata": {"id": "aijob-e00late", "name": "x", "created_at": "2026-10-28T09:00:00Z"},
                "status": {"state": "RUNNING"},
            }
            return CommandResult(0, json.dumps(job), "")
        if argv[1:4] == ["ai", "job", "get"]:
            job = {
                "metadata": {"id": "aijob-e00late", "created_at": "2026-10-28T09:00:00Z"},
                "status": {"state": "RUNNING"},
            }
            return CommandResult(0, json.dumps(job), "")
        return CommandResult(1, "", "unexpected")

    class S3:
        def put_object(self, **kw: object) -> dict[str, object]:
            return {}

    h = Harness()
    h.runner._jobs = NebiusBackend(h.cfg.nebius, NebiusCLI("nebius", "runner", cli, timeout_s=60.0), S3())
    status, body = h.start_run()
    assert status == 202 and len(created) == 1
    rec, _ = h.store.get_run(body["run_id"])  # type: ignore[misc]
    assert rec["settled"] is False and rec["job_id"] is None
    assert h.store.numbers("spend#nebius#total")["committed"] == reservation_micros(h.cfg)
    assert [r["run_id"] for r, _ in h.store.active_runs()] == [body["run_id"]]
    lookups["ok"] = True
    h.advance(h.cfg.unsubmitted_grace_s + 30)
    h.runner.sweep()
    assert h.store.get_run(body["run_id"])[0]["job_id"] == "aijob-e00late"  # type: ignore[index]


def test_a_slow_submission_outlives_a_sweep_and_stays_tracked() -> None:
    """Create plus read-back take 125 s and the sweeper runs meanwhile (the job is not listed yet):
    the run is left alone, the submission records its job, and past the wall limit it is
    cancelled."""
    h = Harness()
    real = h.nebius

    class SlowSubmit:
        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

        def submit(self, run_id: str, plan_json: bytes) -> object:
            h.advance(125)
            real.fail_find = True  # not listed yet
            h.runner.sweep()
            real.fail_find = False
            return real.submit(run_id, plan_json)

    h.runner._jobs = SlowSubmit()  # type: ignore[assignment]
    status, body = h.start_run()
    assert status == 202
    rec, _ = h.store.get_run(body["run_id"])  # type: ignore[misc]
    (job_id,) = real.jobs
    assert rec["job_id"] == job_id and rec["settled"] is False
    h.advance(1500)
    h.runner.sweep()
    assert real.cancels == [job_id]


def test_a_record_lost_to_a_race_cancels_its_job_at_once() -> None:
    h = Harness()
    real = h.nebius

    class RacySubmit:
        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

        def submit(self, run_id: str, plan_json: bytes) -> object:
            sub = real.submit(run_id, plan_json)
            rec, version = h.store.get_run(run_id)  # type: ignore[misc]
            from rtt.runner.store import RunWrite

            # someone else settled the run as never started while this request was submitting
            h.store.commit(run=RunWrite({**rec, "status": "failed", "settled": True}, version, False, 1))
            return sub

    h.runner._jobs = RacySubmit()  # type: ignore[assignment]
    status, body = h.start_run()
    assert status == 503 and "cancelled" in body["error"]
    assert real.cancels == list(real.jobs)


def test_a_job_still_running_after_a_cancel_is_cancelled_again() -> None:
    h = Harness()
    h.start_run()
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    h.advance(1201)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]
    h.advance(h.cfg.caps.cancel_grace_s - 10)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]
    h.advance(20)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id, job_id]


# --------------------------------------------------------------------------- global bounds


def test_keyed_plans_are_capped_site_wide() -> None:
    h = Harness(replace(RunnerConfig(), caps=replace(Caps(), keyed_plans_per_day=3)))
    for i in range(3):
        assert h.plan("nvidia", VISITOR_KEY, ip=f"198.51.100.{i}")[0] == 200
    status, body = h.plan("tokenfactory", VISITOR_KEY, ip="198.51.100.99")
    assert status == 429 and body["cap"] == "keyed_plans_per_day" and body["limit"] == 3
    assert RunnerConfig().caps.keyed_plans_per_day == 300


def test_ipv6_visitors_are_counted_by_their_64_prefix() -> None:
    from rtt.runner.caps import client_key

    assert client_key("2001:db8:1:2:aaaa::1") == client_key("2001:db8:1:2:ffff::9") == "2001:db8:1:2::/64"
    assert client_key("2001:db8:1:3::1") != client_key("2001:db8:1:2::1")
    assert client_key("::ffff:203.0.113.7") == client_key("203.0.113.7") == "203.0.113.7"
    assert client_key("not an ip") == "not an ip"
    h = Harness()
    for i in range(3):
        assert h.plan("free", ip=f"2001:db8:1:2::{i + 1}")[0] == 200
    status, body = h.plan("free", ip="2001:db8:1:2:dead:beef:0:1")
    assert status == 429 and body["cap"] == "free_plans_per_ip_per_day"
    assert h.plan("free", ip="2001:db8:1:3::1")[0] == 200


def test_the_job_gets_the_runners_plan_limits() -> None:
    cfg = replace(RunnerConfig(), max_objects=4, max_steps=5, max_plan_bytes=20000)
    assert cfg.job_env() == {
        "RUNNER_MAX_OBJECTS": "4",
        "RUNNER_MAX_STEPS": "5",
        "RUNNER_MAX_PLAN_BYTES": "20000",
    }
    assert RunnerConfig.from_env(cfg.job_env()).max_steps == 5
    assert RunnerConfig().unsubmitted_grace_s == 360 + 120
    assert RunnerConfig.from_env({"RUNNER_LAMBDA_TIMEOUT_S": "600"}).unsubmitted_grace_s == 720


# --------------------------------------------------------------------------- the job backend cannot be built


def _broken_backend(h: Harness) -> None:
    from rtt.runner.secrets import SecretError

    def broken() -> object:
        raise SecretError("secret effector-lab/runner/nebius-sa: missing s3_secret_access_key")

    h.runner._jobs = None
    h.runner._jobs_factory = broken  # type: ignore[assignment]


def test_an_unbuildable_backend_is_503_before_anything_is_reserved() -> None:
    h = Harness()
    _broken_backend(h)
    status, body = h.start_run(ip="198.51.100.30")
    assert status == 503 and "nothing was reserved" in body["error"]
    assert h.store.numbers("spend#nebius#day#2026-10-28") == {}
    assert h.store.numbers("spend#nebius#total") == {}
    assert not [k for k in h.store.items if k.startswith(("run#", "count#runs#"))]
    # the visitor's runs are untouched: once the backend works, both are still there
    h.runner._jobs_factory = lambda: h.nebius  # type: ignore[assignment]
    assert h.start_run(ip="198.51.100.30")[0] == 202
    assert h.start_run(ip="198.51.100.30")[0] == 202


def test_a_run_without_a_job_id_gives_up_when_the_backend_stays_unbuildable() -> None:
    h = Harness()
    h.nebius.fail_after_create = True
    h.nebius.fail_find = True
    _, body = h.start_run()  # unknown: reserved, active, no job id
    _broken_backend(h)  # a new container whose secret no longer parses
    h.advance(h.cfg.unsubmitted_grace_s + 30)
    assert h.runner.sweep() == {"active": 1, "refreshed": 1}  # left alone, quietly
    assert h.store.get_run(body["run_id"])[0]["settled"] is False  # type: ignore[index]
    h.advance(4500)  # past 75 minutes after creation
    h.runner.sweep()
    view = h.call("GET", f"/runs/{body['run_id']}")[1]
    assert view["status"] == "failed" and "could not be confirmed" in view["error"]
    assert view["cost_estimate_usd"] == to_usd(reservation_micros(h.cfg))
    assert h.store.active_runs() == []


def test_a_run_with_a_job_gives_up_at_its_cost_to_now_when_the_backend_stays_unbuildable() -> None:
    h = Harness()
    _, body = h.start_run()
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    _broken_backend(h)
    h.advance(1800)
    assert h.runner.sweep() == {"active": 1, "refreshed": 0}  # the failure is let through and logged
    status, view, _ = h.call("GET", f"/runs/{body['run_id']}")
    assert status == 200 and view["status"] == "queued"  # answered from the stored record
    h.advance(2800)  # 4600 s after creation
    h.runner.sweep()
    view = h.call("GET", f"/runs/{body['run_id']}")[1]
    assert view["status"] == "failed" and "could not be read" in view["error"]
    assert view["cost_estimate_usd"] == to_usd(job_cost_micros(h.cfg, 4600))
    assert h.store.active_runs() == []
    assert h.call("GET", f"/runs/{body['run_id']}/report")[0] == 404


def test_a_run_whose_status_cannot_be_read_for_long_gives_up_at_its_cost_to_now() -> None:
    h = Harness()
    assert h.start_run()[0] == 202
    h.nebius.fail_status = True
    h.advance(1800)
    assert h.runner.sweep() == {"active": 1, "refreshed": 0}
    h.advance(2800)
    h.runner.sweep()
    total = h.store.numbers("spend#nebius#total")
    assert total["committed"] == total["settled"] == job_cost_micros(h.cfg, 4600)
    assert h.store.active_runs() == []


def test_the_report_link_is_503_when_the_backend_cannot_be_built() -> None:
    h = Harness()
    _, body = h.start_run()
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "COMPLETED")
    h.nebius.results[body["run_id"]] = ({"exit_status": 0}, summary_for(body["run_id"], good_plan()))
    assert h.call("GET", f"/runs/{body['run_id']}")[1]["status"] == "done"
    _broken_backend(h)
    status, view, _ = h.call("GET", f"/runs/{body['run_id']}")
    assert status == 200 and view["status"] == "done" and view["result"]["report_url"] is None
    assert h.call("GET", f"/runs/{body['run_id']}/report")[0] == 503


# --------------------------------------------------------------------------- a refused create


def test_a_refused_create_that_the_platform_reports_absent_is_given_back_at_once() -> None:
    h = Harness()
    h.nebius.refuse_create = True
    status, body = h.start_run(ip="198.51.100.31")
    assert status == 503 and "refused" in body["error"]
    assert h.store.numbers("spend#nebius#total")["committed"] == 0
    (key,) = [k for k in h.store.items if k.startswith("run#")]
    rec = json.loads(h.store.items[key]["doc"])
    assert rec["status"] == "failed" and rec["settled"] is True and "active" not in h.store.items[key]
    h.nebius.refuse_create = False  # the visitor's two runs are both still there
    assert h.start_run(ip="198.51.100.31")[0] == 202
    assert h.start_run(ip="198.51.100.31")[0] == 202


def test_a_refused_create_through_the_cli() -> None:
    """The real backend over a CLI that refuses create (ResourceExhausted) and then reports the job
    absent (NotFound): 503 and everything given back. If the lookup instead finds a job, or fails,
    the run is kept."""
    from rtt.cloud.nebius_jobs import CommandResult, NebiusCLI
    from rtt.runner.nebius import NebiusBackend

    class S3:
        def put_object(self, **kw: object) -> dict[str, object]:
            return {}

    def run_with(create: CommandResult, lookup: CommandResult) -> tuple[Harness, int, dict[str, object]]:
        def cli(argv: list[str], timeout: float | None) -> CommandResult:
            if argv[1:4] == ["ai", "job", "create"]:
                return create
            if "get-by-name" in argv:
                return lookup
            if argv[1:4] == ["ai", "job", "get"]:
                return CommandResult(0, json.dumps(_JOB), "")
            return CommandResult(1, "", "unexpected")

        h = Harness()
        h.runner._jobs = NebiusBackend(h.cfg.nebius, NebiusCLI("nebius", "runner", cli), S3())
        status, body = h.start_run(ip="198.51.100.32")
        return h, status, body

    refused = CommandResult(
        1, "", "Error: rpc error: code = ResourceExhausted desc = quota exceeded for cpu-d3"
    )
    absent = CommandResult(1, "", "rpc error: code = NotFound desc = job not found")
    found = CommandResult(0, json.dumps(_JOB), "")
    flaky = CommandResult(1, "", "rpc error: code = Unavailable desc = connection reset")
    timed_out = CommandResult(124, "", "timed out after 60.0 s")
    exists = CommandResult(1, "", "rpc error: code = AlreadyExists desc = job elab-x exists")

    h, status, _ = run_with(refused, absent)
    assert status == 503 and h.store.numbers("spend#nebius#total")["committed"] == 0
    h, status, body = run_with(refused, found)  # refused, yet the job exists: tracked
    assert status == 202 and h.store.get_run(body["run_id"])[0]["job_id"] == "aijob-e00cli"  # type: ignore[index]
    h, status, body = run_with(refused, flaky)  # cannot tell: kept, reserved
    assert status == 202 and h.store.numbers("spend#nebius#total")["committed"] == reservation_micros(h.cfg)
    h, status, body = run_with(timed_out, absent)  # a timeout is never a refusal: kept until the grace
    assert status == 202 and h.store.get_run(body["run_id"])[0]["settled"] is False  # type: ignore[index]
    h, status, body = run_with(exists, found)  # a retried create that went through: tracked
    assert status == 202 and h.store.get_run(body["run_id"])[0]["job_id"] == "aijob-e00cli"  # type: ignore[index]


_JOB = {
    "metadata": {"id": "aijob-e00cli", "name": "elab-x", "created_at": "2026-10-28T09:00:00Z"},
    "status": {"state": "PROVISIONING"},
}


# --------------------------------------------------------------------------- outages of the status call


def _manifest(finished_at: str) -> dict[str, object]:
    return {"exit_status": 0, "finished_at": finished_at}


def test_a_job_that_finished_during_a_status_outage_is_done_at_its_real_cost() -> None:
    """Status calls fail from minute 2 to past 75 minutes; the job had finished at minute 8 and
    published its result. The give-up reads the result first: done, at job creation to the
    manifest's finished_at, not at 75 minutes."""
    h = Harness()
    _, body = h.start_run()
    rid = body["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    h.advance(120)
    h.runner.sweep()
    h.nebius.fail_status = True
    h.nebius.results[rid] = (_manifest("2026-10-28T09:08:00Z"), summary_for(rid, good_plan()))
    for _ in range(37):  # to minute 76
        h.advance(120)
        h.runner.sweep()
    view = h.call("GET", f"/runs/{rid}")[1]
    assert view["status"] == "done" and view["error"] is None and view["result"]["pick"]["label"] == "Gripper"
    assert view["cost_estimate_usd"] == to_usd(job_cost_micros(h.cfg, 480))
    total = h.store.numbers("spend#nebius#total")
    assert total["committed"] == total["settled"] == job_cost_micros(h.cfg, 480)
    assert h.store.active_runs() == []


@pytest.mark.parametrize(
    "result",
    [
        ({"exit_status": 1, "finished_at": "2026-10-28T09:08:00Z"}, None),  # the search failed
        (None, None),  # nothing published
        "raises",
    ],
)
def test_without_a_complete_result_the_give_up_cancels_and_counts_cost_to_now(result: object) -> None:
    h = Harness()
    _, body = h.start_run()
    rid = body["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.fail_status = True
    if result == "raises":

        def boom(run_id: str) -> object:
            raise RuntimeError("storage unreachable")

        h.nebius.read_result = boom  # type: ignore[method-assign]
    else:
        h.nebius.results[rid] = result  # type: ignore[assignment]
    h.advance(4600)
    h.runner.sweep()
    view = h.call("GET", f"/runs/{rid}")[1]
    assert view["status"] == "failed" and view["cost_estimate_usd"] == to_usd(job_cost_micros(h.cfg, 4600))
    assert h.nebius.cancels[-1] == job_id  # stopped before the runner lets go of it
    assert h.store.active_runs() == []


def test_a_run_past_the_wall_limit_is_cancelled_even_while_its_status_cannot_be_read() -> None:
    h = Harness()
    _, body = h.start_run()
    rid = body["run_id"]
    (job_id,) = h.nebius.jobs
    h.nebius.set_state(job_id, "RUNNING")
    h.nebius.fail_status = True
    h.advance(1100)
    assert h.runner.sweep() == {"active": 1, "refreshed": 0}
    assert h.nebius.cancels == []  # not due yet
    h.advance(120)  # 1220 s: past the 20-minute limit, status still failing
    assert h.runner.sweep() == {"active": 1, "refreshed": 0}  # the failure is still let through
    assert h.nebius.cancels == [job_id]
    view = h.call("GET", f"/runs/{rid}")[1]
    assert view["status"] == "failed" and "20-minute" in view["error"]
    h.advance(120)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]  # within the grace: not again
    h.advance(h.cfg.caps.cancel_grace_s)
    h.runner.sweep()
    assert h.nebius.cancels == [job_id, job_id]  # still running after the grace: again


def test_the_kill_switch_cancels_even_while_status_cannot_be_read() -> None:
    h = Harness()
    h.start_run()
    (job_id,) = h.nebius.jobs
    h.nebius.fail_status = True
    h.advance(120)
    h.params["/test/kill"] = "on"
    h.runner.sweep()
    assert h.nebius.cancels == [job_id]


def test_failed_cancels_never_block_settling() -> None:
    h = Harness()
    _, body = h.start_run()
    rid = body["run_id"]
    h.nebius.fail_status = True
    h.nebius.fail_cancel = True
    h.advance(1300)
    h.runner.sweep()  # the cancel fails: logged, retried next time
    view = h.call("GET", f"/runs/{rid}")[1]
    assert view["status"] == "failed" and "20-minute" in view["error"]
    h.advance(3300)
    h.runner.sweep()  # the give-up's cancel fails too: it settles anyway
    assert h.store.active_runs() == []
    assert h.store.numbers("spend#nebius#total")["settled"] == job_cost_micros(h.cfg, 4600)


@pytest.mark.parametrize(
    "manifest",
    [{"exit_status": 0}, {"exit_status": 0, "finished_at": "2026-10-28T09:08:00"}],  # none; no zone
    ids=["no-finish-time", "naive-finish-time"],
)
def test_a_complete_result_without_a_usable_finish_time_is_done_at_cost_to_now(
    manifest: dict[str, object],
) -> None:
    h = Harness()
    _, body = h.start_run()
    rid = body["run_id"]
    h.nebius.fail_status = True
    h.nebius.results[rid] = (manifest, summary_for(rid, good_plan()))
    h.advance(4600)
    h.runner.sweep()
    view = h.call("GET", f"/runs/{rid}")[1]
    assert view["status"] == "done"  # the result is complete
    assert view["cost_estimate_usd"] == to_usd(
        job_cost_micros(h.cfg, 4600)
    )  # its cost is not known: safe side
    assert h.store.active_runs() == []
