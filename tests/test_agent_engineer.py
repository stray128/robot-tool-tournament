"""The Grasp Engineer loop, its proposer, summarizer, recorder and backends (scripted model, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import trimesh

from rtt.agent.backends import SimPartBackend, certificate_summary
from rtt.agent.caps import AgentCaps, DailySpendBook, GuardedLedger, SpendGuard
from rtt.agent.engineer import AgentLLM, GraspEngineer, _parse_turn, system_prompt
from rtt.agent.fakes import FakePartBackend, ScriptedAgentTransport
from rtt.agent.jobs import LocalJobRunner
from rtt.agent.proposer import NemotronDesignProposer, check_proposal
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SCRIPTED_MODEL, SessionRecorder, load_session
from rtt.agent.store import AgentStore
from rtt.agent.summarizer import LightningSummarizer
from rtt.agent.tools import Toolbox
from rtt.design.schema import GripperTask, load_entrants
from rtt.llm.providers import NEMOTRON_LIGHTNING, NEMOTRON_SUPER, TransportResponse

REPO = Path(__file__).resolve().parents[1]
ENTRANTS = load_entrants(REPO / "configs" / "entrants.yaml")
PART = "part-0123456789ab"
SESSION = "sess-0123456789ab"


def box_stl(path: Path, extents: tuple[float, float, float] = (40.0, 30.0, 60.0)) -> Path:
    trimesh.creation.box(extents=extents).export(path)
    return path


def build(
    tmp_path: Path,
    transport: Any,
    *,
    caps: AgentCaps | None = None,
    offline: bool = True,
    summarize: bool = True,
) -> tuple[GraspEngineer, SessionRecorder, Toolbox]:
    backend = FakePartBackend(tmp_path / "parts", delay_s=0.2)  # jobs are still running at launch
    backend.intake(box_stl(tmp_path / "box.stl"), part_id=PART, name="box")
    caps = caps or AgentCaps(poll_wait_s=5.0)
    guard = SpendGuard(caps.session_usd, DailySpendBook(None, caps.day_usd), SESSION)
    ledger = GuardedLedger(guard)
    store = AgentStore(tmp_path / "agent")
    recorder = SessionRecorder(SESSION, store.session_path(SESSION))
    toolbox = Toolbox(
        session_id=SESSION,
        part_id=PART,
        backend=backend,
        proposer=NemotronDesignProposer.create(ledger, transport=transport),
        runners={"local": LocalJobRunner(backend.evaluate)},
        store=store,
        caps=caps,
        guard=guard,
        unlocked=True,
        ledger=ledger,
        recorder=recorder,
        baselines=ENTRANTS,
        offline=offline,
    )
    engineer = GraspEngineer(
        AgentLLM.create(ledger, transport=transport),
        toolbox,
        recorder,
        caps=caps,
        summarizer=LightningSummarizer.create(ledger, transport=transport) if summarize else None,
        offline=offline,
    )
    return engineer, recorder, toolbox


# --------------------------------------------------------------------------- loop


def test_scripted_session_runs_every_tool_and_records_it(tmp_path: Path) -> None:
    transport = ScriptedAgentTransport()
    engineer, recorder, _toolbox = build(tmp_path, transport)
    result = engineer.run()
    assert result.status == "completed", result.reason
    assert result.tool_calls <= 12
    doc = load_session(recorder.path)
    assert doc["label"] == PRODUCT_MODE_LABEL and doc["mode"] == "product" and doc["status"] == "completed"
    types = [e["type"] for e in doc["events"]]
    assert types[0] == "session.started" and types[-1] == "session.finished"
    assert [e["seq"] for e in doc["events"]] == list(range(1, len(types) + 1))
    called = [e["name"] for e in doc["events"] if e["type"] == "tool.call"]
    assert called == [
        "inspect_part",
        "propose_designs",
        "launch_sim_job",
        "poll_job",
        "fetch_results",
        "stress_test",
        "export_print_pack",
        "cost_ledger",
    ]
    roles = {e["role"] for e in doc["events"] if e["type"] == "llm.call"}
    assert roles == {"agent", "proposer", "summarizer"}
    assert {e["model"] for e in doc["events"] if e["type"] == "llm.call"} == {SCRIPTED_MODEL}
    assert doc["events"][0]["models"]["agent"] == SCRIPTED_MODEL
    assert any(e["type"] == "summary" for e in doc["events"])
    assert any(e["type"] == "job.update" and e["state"] == "COMPLETED" for e in doc["events"])
    assert doc["verdicts"] and all(
        v["verdict"] in ("pass", "fail") for v in doc["verdicts"] if "verdict" in v
    )
    assert doc["spend"]["spent_usd"] > 0
    # the transcript replays as an OpenAI conversation: each tool call is answered once
    calls = [c["id"] for m in doc["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    answers = [m["tool_call_id"] for m in doc["messages"] if m["role"] == "tool"]
    assert calls == answers
    assert doc["answer"].startswith("Recommended design")
    # the agent's own requests: tools offered with tool_choice auto, model is Super
    agent_payloads = [p for p in transport.payloads if p.get("tool_choice") == "auto"]
    assert agent_payloads and all(
        p["model"] == NEMOTRON_SUPER and len(p["tools"]) == 8 for p in agent_payloads
    )
    lightning = [p for p in transport.payloads if p["model"] == NEMOTRON_LIGHTNING]
    assert lightning and all(p["chat_template_kwargs"] == {"enable_thinking": False} for p in lightning)


def test_invalid_tool_arguments_are_reported_and_counted(tmp_path: Path) -> None:
    engineer, recorder, _toolbox = build(
        tmp_path, ScriptedAgentTransport(bad_first_call=True), summarize=False
    )
    result = engineer.run()
    assert result.status == "completed"
    results = [e for e in recorder.to_json()["events"] if e["type"] == "tool.result"]
    assert results[0]["ok"] is False and results[0]["code"] == "invalid_args"
    assert result.tool_calls == 9  # the bad call counts


def test_tool_budget_forces_a_final_answer(tmp_path: Path) -> None:
    transport = ScriptedAgentTransport()
    engineer, recorder, _toolbox = build(
        tmp_path, transport, caps=AgentCaps(max_tool_calls=3), summarize=False
    )
    result = engineer.run()
    assert result.status == "completed" and result.tool_calls == 3
    assert transport.payloads[-1]["tool_choice"] == "none"
    assert not any(e["type"] == "tool.refused" for e in recorder.to_json()["events"])


def test_spend_cap_stops_the_session_before_any_call(tmp_path: Path) -> None:
    transport = ScriptedAgentTransport()
    engineer, _recorder, _ = build(tmp_path, transport, caps=AgentCaps(session_usd=0.001), summarize=False)
    result = engineer.run()
    assert result.status == "stopped" and "cap" in (result.reason or "")
    assert transport.payloads == []


def test_stop_flag_ends_the_session(tmp_path: Path) -> None:
    engineer, recorder, _ = build(tmp_path, ScriptedAgentTransport(), summarize=False)
    engineer.stop.set()
    result = engineer.run()
    assert result.status == "stopped" and result.turns == 0
    assert recorder.to_json()["events"][-1]["type"] == "session.finished"


class BrokenTransport:
    def chat_completions(self, payload: Any, timeout_s: float) -> TransportResponse:
        return TransportResponse(401, {}, {"error": {"message": "invalid key"}})

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(401, {}, None)


def test_auth_failure_fails_the_session_cleanly(tmp_path: Path) -> None:
    engineer, recorder, _ = build(tmp_path, BrokenTransport(), offline=False)
    result = engineer.run()
    assert result.status == "failed" and "LLMAuthError" in (result.reason or "")
    assert recorder.done


def test_parse_turn_handles_bad_arguments_and_reasoning() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>plan</think>Calling tools.",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "cost_ledger", "arguments": "{}"},
                        },
                        {
                            "id": "b",
                            "type": "function",
                            "function": {"name": "poll_job", "arguments": "{oops"},
                        },
                        {"id": "c", "type": "function", "function": {"name": "poll_job", "arguments": "[1]"}},
                    ],
                }
            }
        ]
    }
    turn, notes = _parse_turn(response)
    assert turn.content == "Calling tools." and turn.reasoning == "plan"
    assert [c.args for c in turn.tool_calls] == [{}, None, None]
    assert "not valid JSON" in (turn.tool_calls[1].parse_error or "")
    assert turn.message()["tool_calls"][1]["function"]["arguments"] == "{oops"
    assert "tool_args_not_json" in notes


def test_system_prompt_states_the_caps_and_no_brand_names() -> None:
    text = system_prompt(AgentCaps(), ["local"])
    assert "12 tool calls" in text and "2 jobs" in text and "$0.50" in text
    assert "nebius" not in text.lower()
    assert "SO-" not in text  # no gripper brand on screen
    assert "'nebius'" in system_prompt(AgentCaps(), ["local", "nebius"])


# --------------------------------------------------------------------------- proposer + summarizer


def test_proposer_validates_and_rejects_near_copies(tmp_path: Path) -> None:
    backend = FakePartBackend(tmp_path)
    info = backend.intake(box_stl(tmp_path / "box.stl"), part_id=PART, name="held out box")
    transport = ScriptedAgentTransport()
    proposer = NemotronDesignProposer.create(
        GuardedLedger(SpendGuard(1.0, DailySpendBook(None, 2.0), SESSION)), transport=transport
    )
    task = backend.gripper_task(PART)
    batch = proposer.propose(info, 2, gripper_task=task, history=[], known=ENTRANTS)
    assert len(batch.designs) == 2 and not batch.rejections
    prompt = transport.payloads[0]["messages"][1]["content"]
    assert "40 mm" in prompt or "30 mm" in prompt
    assert "held out" not in prompt.lower()  # user text is scrubbed before it reaches a prompt
    assert transport.payloads[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_designs"},
    }
    again = proposer.propose(info, 2, gripper_task=task, history=[], known=[d.design for d in batch.designs])
    assert len(again.designs) == 2  # the next two scripted designs are new
    same = check_proposal(batch.designs[0].design.geometry_params(), "", task, [batch.designs[0].design])
    assert same[0] is None and "too similar" in same[1][0]


def test_check_proposal_reports_validator_errors() -> None:
    bad = dict(ENTRANTS[0].geometry_params(), inner_face="flat", inner_face_depth_mm=4.0)
    design, reasons = check_proposal(bad, "x", GripperTask(), [])
    assert design is None and reasons


def test_summarizer_returns_none_on_failure() -> None:
    ledger = GuardedLedger(SpendGuard(1.0, DailySpendBook(None, 2.0), SESSION))
    assert LightningSummarizer.create(ledger, transport=BrokenTransport()).summarize("x", {"a": 1}) is None
    ok = LightningSummarizer.create(ledger, transport=ScriptedAgentTransport())
    assert ok.summarize("fetch_results", {"best": "abc"}).startswith("Scripted summary")
    assert ok.last_meta is not None and ok.last_meta.model == NEMOTRON_LIGHTNING


# --------------------------------------------------------------------------- recorder


def test_recorder_streams_events_and_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    rec = SessionRecorder(SESSION, path, meta={"goal": "g"})
    rec.emit("a", value=float("nan"))
    assert rec.wait_events(0, 0.0)[0]["value"] is None
    assert rec.wait_events(1, 0.01) == []
    rec.finish("completed", answer="done")
    assert rec.done
    doc = load_session(path)
    assert doc["goal"] == "g" and doc["events"][-1]["answer"] == "done"
    with pytest.raises(ValueError):
        rec.finish("weird")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        load_session(bad)


# --------------------------------------------------------------------------- backends


def test_certificate_summary_pools_poses() -> None:
    raw = {
        "design_id": "abc",
        "score": 0.8,
        "passed": 30,
        "n_tests": 36,
        "poses": [
            {
                "pose_rank": 0,
                "probability": 0.75,
                "category_scores": {"placement": 1.0, "shake": 1.0, "push": 0.5},
                "tolerance_mm": {"x": 3.0, "y": 2.0},
                "tests": [{"name": "nominal", "success": True}],
            },
            {
                "pose_rank": 1,
                "probability": 0.25,
                "category_scores": {"placement": 0.6, "shake": 0.0, "push": 0.5},
                "tolerance_mm": {"x": 1.0, "y": None},
                "tests": [{"name": "nominal", "success": False}],
            },
        ],
    }
    out = certificate_summary(raw)
    assert out["category_scores"]["placement"] == pytest.approx(0.9)
    assert out["category_scores"]["shake"] == pytest.approx(0.75)
    assert out["tolerance_mm"] == {"x": 1.0, "y": None}
    assert [t["name"] for t in out["tests"]] == ["pose0/nominal", "pose1/nominal"]
    assert out["passed"] == "30/36"


def test_sim_backend_simulates_an_uploaded_part(tmp_path: Path) -> None:
    """Real MuJoCo: intake a box, score one entrant on it, write a print pack (no certificate)."""
    backend = SimPartBackend(tmp_path / "parts", workers=2, seeds=1)
    info = backend.intake(box_stl(tmp_path / "box.stl"), part_id=PART, name="box")
    assert info.fits_gripper and info.grip_width_mm == pytest.approx(
        40.0, abs=0.5
    )  # rests on its 40 x 60 face
    assert info.episodes_per_design == 3 and info.backend == "sim"
    (score,) = backend.evaluate(PART, ENTRANTS[:1])
    assert score.design_id == ENTRANTS[0].design_id() and score.n_episodes == 3
    files = backend.print_pack(
        PART, ENTRANTS[0], tmp_path / "pack", results=score.to_json(include_episodes=False)
    )
    names = {f.name for f in files}
    assert {"finger_left.stl", "finger_right.stl", "design.json", "results.json"} <= names
    assert (tmp_path / "pack" / "finger_left.stl").stat().st_size > 1000
    fresh = SimPartBackend(tmp_path / "parts", workers=2, seeds=1)  # restart: reloaded from the stored upload
    assert fresh.info(PART).source_sha256 == info.source_sha256


def test_summaries_cannot_announce_a_verdict_the_step_did_not_produce() -> None:
    from rtt.agent.summarizer import drop_verdict_claims

    text = "The bracket is 78 mm long and weighs 15 g. The physics simulator verdict is pass."
    assert drop_verdict_claims(text) == "The bracket is 78 mm long and weighs 15 g."

    class Claims(ScriptedAgentTransport):
        def chat_completions(self, payload: Any, timeout_s: float) -> TransportResponse:
            response = super().chat_completions(payload, timeout_s)
            response.body["choices"][0]["message"]["content"] = "It is 40 mm wide. It passed."
            return response

    ledger = GuardedLedger(SpendGuard(1.0, DailySpendBook(None, 2.0), SESSION))
    summarizer = LightningSummarizer.create(ledger, transport=Claims())
    assert summarizer.summarize("inspect_part", {"part": {"grip_width_mm": 40}}) == "It is 40 mm wide."
    assert summarizer.summarize("fetch_results", {"verdict": "pass"}) == "It is 40 mm wide. It passed."


def test_a_failing_summary_never_leaves_the_session_open(tmp_path: Path) -> None:
    # e.g. an unreadable day book makes the summary's reservation raise RuntimeError
    engineer, recorder, _ = build(tmp_path, ScriptedAgentTransport())
    assert engineer.summarizer is not None

    def broken(kind: str, payload: Any) -> str | None:
        raise RuntimeError(f"daily spend book {tmp_path}/x/y/daily_spend.json is unreadable")

    engineer.summarizer.summarize = broken  # type: ignore[method-assign]
    result = engineer.run()
    doc = load_session(recorder.path)
    assert result.status == "completed" and doc["status"] == "completed"
    assert doc["events"][-1]["type"] == "session.finished"
    errors = [e["message"] for e in doc["events"] if e["type"] == "error"]
    assert errors and all(str(tmp_path) not in m for m in errors)


def test_summaries_cannot_upgrade_a_fail() -> None:
    from rtt.agent.summarizer import drop_unsupported_passes, verdicts_in

    payload = {"designs": [{"verdict": "fail"}], "stress": {"certificate_verdict": "fail"}}
    assert verdicts_in(payload) == {"fail"}
    assert verdicts_in({"verdicts": [{"verdict": "pass"}, {"certificate_verdict": "fail"}]}) == {
        "pass",
        "fail",
    }
    assert drop_unsupported_passes("Best was 75%. It passed the stress test. Push failed.") == (
        "Best was 75%. Push failed."
    )

    class Upgrades(ScriptedAgentTransport):
        def chat_completions(self, payload: Any, timeout_s: float) -> TransportResponse:
            response = super().chat_completions(payload, timeout_s)
            response.body["choices"][0]["message"]["content"] = "Design abc scored 75%. It passes."
            return response

    ledger = GuardedLedger(SpendGuard(1.0, DailySpendBook(None, 2.0), SESSION))
    summarizer = LightningSummarizer.create(ledger, transport=Upgrades())
    assert (
        summarizer.summarize("fetch_results", {"designs": [{"verdict": "fail"}]}) == "Design abc scored 75%."
    )
    mixed = {"designs": [{"verdict": "fail"}, {"verdict": "pass"}]}
    assert summarizer.summarize("fetch_results", mixed) == "Design abc scored 75%. It passes."
