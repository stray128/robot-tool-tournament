"""The Grasp Engineer's Studio tools: plan_task, run_task_tournament, export_build_pack (scripted models only).

No network and no spend: recorded plans replay the S4e gate cache, a live plan answers from a scripted
transport, and the tournament is a fake that returns one small real report (two dummy push steps, two
off-the-shelf tools, two conditions, S2's default controllers), made once per module.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import trimesh

from rtt.agent.caps import AgentCaps, DailySpendBook, GuardedLedger, SpendGuard
from rtt.agent.fakes import FakePartBackend
from rtt.agent.jobs import LocalJobRunner
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SCRIPTED_MODEL, SessionRecorder, load_session
from rtt.agent.schemas import ArgsError, openai_tools
from rtt.agent.store import AgentStore
from rtt.agent.studio_tool import (
    HARD_MAX_WORKERS,
    HERO_CASE_IDS,
    STUDIO_SESSION_TOOLS,
    STUDIO_TOOL_NAMES,
    STUDIO_TOOL_SPECS,
    DeadlinePool,
    ScriptedStudioTransport,
    StudioCaps,
    StudioTools,
    WallTimeCapError,
    hero_text,
    main,
    record_studio_session,
    studio_system_prompt,
    studio_toolbox,
    validate_studio_args,
    whole_sentences,
)
from rtt.agent.tools import Toolbox
from rtt.studio.runner import WorkerPool
from rtt.studio.search import run_task_tournament
from studio_fixtures import cup, gripper, two_press_task
from test_studio_engineer import FROZEN_HERO_TASK_IDS

SESSION = "sess-0123456789ab"


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """A small real tournament report with a recommendation (the suction cup, an approximation)
    and the adaptive gripper as the best single off-the-shelf effector."""
    rep = run_task_tournament(
        two_press_task(),
        candidates=[gripper(), cup()],
        library=False,
        n_conditions=2,
        n_screen=1,
        n_finalists=2,
        workers=1,
        kinds=["single"],
    )
    data = json.loads(json.dumps(rep.to_dict()))
    assert data["recommendation"]["recommendation"] is not None
    return data


def without_winner(report: dict[str, Any]) -> dict[str, Any]:
    """The same report as if no setup had completed the task (stopping at the first step)."""
    data = copy.deepcopy(report)
    rec = data["recommendation"]
    for f in data["finalists"]:
        for run in f["runs"]:
            run["completed"] = False
    for row in rec["ranking"]:
        row.update(
            eligible=False,
            completion_rate=0.0,
            reason="never completed the task (stops at press_carton in 2/2 runs: slip 2x)",
            stops_at={"press_carton": {"runs": 2, "failure_modes": {"slip": 2}}},
        )
    rec.update(recommendation=None, runner_up_rigid=None, versus_best_single=None)
    rec["best_single"]["completion_rate"] = 0.0
    rec["no_winner"] = {
        "reason": "no setup completed the task on the condition grid (nothing is forced)",
        "closest": rec["ranking"][:3],
    }
    return data


class _Report:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


class FakeTournament:
    """Returns ``report``; records every call; ``hook(pool)`` runs before it returns."""

    def __init__(self, report: dict[str, Any], hook: Any = None) -> None:
        self.report = report
        self.hook = hook
        self.calls: list[dict[str, Any]] = []

    def __call__(self, spec: Any, *, workers: int, n_conditions: int, pool: WorkerPool) -> _Report:
        self.calls.append(
            {"task_id": spec.task_id, "workers": workers, "n_conditions": n_conditions, "pool": pool}
        )
        if self.hook is not None:
            self.hook(pool)
        return _Report(self.report)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_box(
    tmp_path: Path,
    *,
    unlocked: bool = False,
    caps: StudioCaps | None = None,
    session_usd: float = 0.5,
    session_id: str = SESSION,
    **studio_kwargs: Any,
) -> Toolbox:
    agent_caps = AgentCaps(session_usd=session_usd)
    guard = SpendGuard(session_usd, DailySpendBook(None, agent_caps.day_usd), session_id)
    studio = StudioTools(caps=caps or StudioCaps(workers=1), **studio_kwargs)
    return studio_toolbox(
        session_id=session_id,
        store=AgentStore(tmp_path / "agent"),
        caps=agent_caps,
        guard=guard,
        studio=studio,
        unlocked=unlocked,
        ledger=GuardedLedger(guard),
        recorder=SessionRecorder(session_id, None),
    )


def ok(outcome: Any) -> dict[str, Any]:
    assert outcome.ok, (outcome.code, outcome.error)
    return outcome.result


def assert_clean(result: Any, tmp_path: Path) -> None:
    text = json.dumps(result)
    assert str(tmp_path) not in text and "/home/" not in text and "/tmp/" not in text
    assert result["label"] == PRODUCT_MODE_LABEL


def plan(box: Toolbox, case_id: str = "h1_tote") -> dict[str, Any]:
    return ok(box.call("plan_task", {"text": hero_text(case_id)}))


# --------------------------------------------------------------------------- schemas


@pytest.mark.parametrize("spec", STUDIO_TOOL_SPECS, ids=lambda s: s.name)
def test_json_schemas_match_the_argument_models(spec: Any) -> None:
    props = set(spec.parameters["properties"])
    fields = set(spec.args_model.model_fields)
    assert props == fields and spec.parameters["additionalProperties"] is False
    required_by_model = {n for n, f in spec.args_model.model_fields.items() if f.is_required()}
    assert required_by_model <= set(spec.parameters["required"]) <= fields
    assert spec.openai()["function"]["name"] == spec.name
    assert tuple(s.name for s in STUDIO_TOOL_SPECS) == STUDIO_TOOL_NAMES


@pytest.mark.parametrize(
    ("name", "args", "problem"),
    [
        ("plan_task", {}, "text"),
        ("plan_task", {"text": ""}, "text"),
        ("plan_task", {"text": "   "}, "describe"),
        ("plan_task", {"text": "x" * 9000}, "text"),
        ("plan_task", {"text": "open a drawer", "budget": 1}, "budget"),
        ("run_task_tournament", {"plan_id": "plan-xyz"}, "plan_id"),
        ("run_task_tournament", {"plan_id": "../../etc/passwd"}, "plan_id"),
        ("run_task_tournament", {"plan_id": "plan-0123456789ab", "n_conditions": 2}, "n_conditions"),
        ("run_task_tournament", {"plan_id": "plan-0123456789ab", "n_conditions": 17}, "n_conditions"),
        ("run_task_tournament", {"plan_id": "plan-0123456789ab", "workers": 8}, "workers"),
        ("export_build_pack", {"tournament_id": "tour-123"}, "tournament_id"),
        ("export_build_pack", {"tournament_id": "tour-0123456789ab", "setup_id": "abc"}, "setup_id"),
        ("export_build_pack", {"tournament_id": "tour-0123456789ab", "setup_id": "g" * 12}, "setup_id"),
        ("export_build_pack", {"tournament_id": "tour-0123456789ab", "setup_id": "a" * 20}, "setup_id"),
        ("no_such_tool", {}, "unknown Studio tool"),
    ],
)
def test_studio_arguments_are_rejected(name: str, args: dict[str, Any], problem: str) -> None:
    with pytest.raises(ArgsError) as info:
        validate_studio_args(name, args)
    assert problem in str(info.value)


def test_studio_arguments_are_accepted() -> None:
    args = validate_studio_args("run_task_tournament", {"plan_id": "plan-0123456789ab", "n_conditions": 8})
    assert args.model_dump() == {"plan_id": "plan-0123456789ab", "n_conditions": 8}
    for sid in ("0123456789ab", "a" * 64):
        got = validate_studio_args(
            "export_build_pack", {"tournament_id": "tour-0123456789ab", "setup_id": sid}
        )
        assert got.setup_id == sid
    with pytest.raises(ArgsError, match="JSON object"):
        validate_studio_args("plan_task", "not a dict")


def test_studio_caps_have_hard_ceilings() -> None:
    assert StudioCaps().to_json() == {
        "plan_usd": 0.05,
        "wall_s": 900.0,
        "workers": 4,
        "n_conditions": 8,
        "max_zip_bytes": 20 * 1024 * 1024,
    }
    for bad in (
        {"workers": HARD_MAX_WORKERS + 1},
        {"workers": 0},
        {"wall_s": 1801.0},
        {"wall_s": 0.0},
        {"wall_s": float("nan")},
        {"plan_usd": float("nan")},
        {"plan_usd": -0.01},
        {"n_conditions": 2},
        {"max_zip_bytes": 0},
    ):
        with pytest.raises(ValueError):
            StudioCaps(**bad)
    assert StudioCaps(wall_s=1800.0).wall_s == 1800.0


# --------------------------------------------------------------------------- plan_task


def test_recorded_hero_plans_are_free_and_need_no_passcode(tmp_path: Path) -> None:
    box = make_box(tmp_path, unlocked=False)
    for case_id in HERO_CASE_IDS:
        result = plan(box, case_id)
        assert result["source"] == "recorded" and result["status"] == "valid"
        assert result["task_id"] == FROZEN_HERO_TASK_IDS[case_id]
        assert result["cost_usd"] == 0 and result["steps"] and result["objects"]
        assert all(s["fidelity"] in ("rigid", "approximation") for s in result["steps"])
        assert result["plan_id"].startswith("plan-")
        saved = json.loads((box.store.root / "studio" / "plans" / f"{result['plan_id']}.json").read_text())
        assert saved["status"] == "valid" and saved["label"] == PRODUCT_MODE_LABEL
        assert saved["plan"]["task_id"] == FROZEN_HERO_TASK_IDS[case_id]
        assert_clean(result, tmp_path)
    assert box.guard.snapshot()["spent_usd"] == 0 and box.guard.reserved_usd == 0
    h3 = plan(box, "h3_cabinet")
    assert h3["result"] == "valid_after_layout_fix" and h3["layout_moves"]
    calls = [e for e in box.recorder.to_json()["events"] if e["type"] == "llm.call"]
    assert calls and all(e["role"] == "task_engineer" and e["cached"] and e["cost_usd"] == 0 for e in calls)


def test_a_recorded_description_copied_with_other_spacing_is_still_recorded(tmp_path: Path) -> None:
    box = make_box(tmp_path, unlocked=False)
    copied = '"' + "  ".join(hero_text("h2_coupling").upper().split()) + '"\n'
    result = ok(box.call("plan_task", {"text": copied}))
    assert result["source"] == "recorded" and result["task_id"] == FROZEN_HERO_TASK_IDS["h2_coupling"]
    assert "matched a recorded description" in result["note"] and result["cost_usd"] == 0
    reworded = box.call("plan_task", {"text": hero_text("h2_coupling").replace("coupling", "fitting")})
    assert reworded.code == "passcode_required"


def test_a_description_that_is_not_recorded_needs_the_passcode(tmp_path: Path) -> None:
    transport = ScriptedStudioTransport()
    box = make_box(tmp_path, unlocked=False, transport=transport)
    out = box.call("plan_task", {"text": "Pick a ball off the bench and drop it in a cup."})
    assert out.refused and out.code == "passcode_required" and "recorded" in out.error
    assert transport.payloads == []
    offline = make_box(tmp_path / "offline", unlocked=True, transport=transport, live_plans=False)
    out = offline.call("plan_task", {"text": "Pick a ball off the bench and drop it in a cup."})
    assert out.refused and out.code == "not_recorded" and transport.payloads == []


def test_a_live_plan_is_capped_per_plan_and_charged_to_the_session(tmp_path: Path) -> None:
    recorded = StudioTools().recorded_plan(hero_text("h1_tote"))
    assert recorded is not None and recorded.spec is not None
    text = hero_text("h1_tote") + " The tote stays on the bench."
    transport = ScriptedStudioTransport(plan_args=recorded.spec.to_tool_args())
    box = make_box(tmp_path, unlocked=True, transport=transport)
    result = ok(box.call("plan_task", {"text": text}))
    assert (
        result["source"] == "live" and result["status"] == "valid" and result["result"] == "valid_first_call"
    )
    assert len(transport.payloads) == 1 and isinstance(transport.payloads[0]["tool_choice"], dict)
    spent = box.guard.snapshot()["by_kind_usd"]["llm"]
    assert 0 < spent <= 0.05 and result["cost_usd"] == pytest.approx(spent, abs=1e-6)
    assert box.guard.reserved_usd == 0
    assert_clean(result, tmp_path)
    # over the per-plan cap: refused before anything is sent
    capped = make_box(
        tmp_path / "capped", unlocked=True, transport=transport, caps=StudioCaps(plan_usd=0.001)
    )
    out = capped.call("plan_task", {"text": text + " Twice."})
    assert out.refused and out.code == "spend_cap"
    assert len(transport.payloads) == 1 and capped.guard.snapshot()["spent_usd"] == 0
    # inside the per-plan cap but over what the session has left: refused by the session guard
    poor = make_box(tmp_path / "poor", unlocked=True, transport=transport, session_usd=0.002)
    out = poor.call("plan_task", {"text": text + " Three times."})
    assert out.refused and out.code == "spend_cap" and len(transport.payloads) == 1


# --------------------------------------------------------------------------- run_task_tournament


def test_the_tournament_result_verdict_rows_and_saved_report(tmp_path: Path, report: dict[str, Any]) -> None:
    fake = FakeTournament(report)
    box = make_box(tmp_path, tournament=fake)
    planned = plan(box)
    result = ok(box.call("run_task_tournament", {"plan_id": planned["plan_id"]}))
    call = fake.calls[0]
    assert call["task_id"] == FROZEN_HERO_TASK_IDS["h1_tote"]
    assert call["workers"] == 1 and call["n_conditions"] == 8 and isinstance(call["pool"], DeadlinePool)
    assert result["verdict"] == "recommended" and result["tournament_id"].startswith("tour-")
    rec = result["recommendation"]
    card = report["recommendation"]["recommendation"]
    assert rec["setup_id"] == card["setup_id"][:12] and rec["label"] == card["label"]
    assert rec["fidelity"] == "approximation" and rec["verify_on_hardware"] is True
    assert rec["caveat"] == "approximation: verify on hardware"
    assert (rec["completed_runs"], rec["total_runs"]) == (2, 2) and rec["completion_pct"] == 100.0
    assert set(rec["step_fidelity"]) == {"press_carton", "press_flat"}
    best = result["best_single"]
    assert best["label"] == report["recommendation"]["best_single"]["label"] and best["total_runs"] == 2
    assert result["versus_best_single"]["same_setup"] is False
    assert len(result["finalists"]) == 2 and result["runs"] == 4 and result["conditions"] == 2
    assert "no_recommendation" not in result
    assert_clean(result, tmp_path)
    saved = json.loads(
        (box.store.root / "studio" / "tournaments" / f"{result['tournament_id']}.json").read_text()
    )
    assert saved["task_id"] == report["task_id"] and saved["plan_id"] == planned["plan_id"]
    rows = [v for v in box.verdicts() if v["kind"] == "studio_recommendation"]
    assert rows == [
        {
            "kind": "studio_recommendation",
            "tournament_id": result["tournament_id"],
            "plan_id": planned["plan_id"],
            "task_id": report["task_id"],
            "recommendation": "recommended",
            "setup_id": rec["setup_id"],
            "setup_label": rec["label"],
            "completion_pct": 100.0,
            "completed_runs": 2,
            "total_runs": 2,
            "fidelity": "approximation",
            "verify_on_hardware": True,
            "best_single_label": best["label"],
            "best_single_completion_pct": best["completion_pct"],
            "label": PRODUCT_MODE_LABEL,
        }
    ]
    ok(box.call("run_task_tournament", {"plan_id": planned["plan_id"], "n_conditions": 3}))
    assert fake.calls[1]["n_conditions"] == 3 and len(box.verdicts()) == 2


def test_a_tournament_without_a_winner_names_the_closest_setups(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    box = make_box(tmp_path, tournament=FakeTournament(without_winner(report)))
    result = ok(box.call("run_task_tournament", {"plan_id": plan(box)["plan_id"]}))
    assert result["verdict"] == "no_recommendation" and result["recommendation"] is None
    nr = result["no_recommendation"]
    assert "no setup completed" in nr["reason"] and nr["closest"]
    first = nr["closest"][0]
    assert first["stops_at"] == {"step_id": "press_carton", "runs": 2, "failure_modes": {"slip": 2}}
    assert (first["completed_runs"], first["total_runs"]) == (0, 2)
    assert result["best_single"]["stops_at"]["step_id"] == "press_carton"
    assert box.verdicts()[0]["recommendation"] == "no_recommendation"


def test_only_a_valid_plan_of_this_session_runs(tmp_path: Path, report: dict[str, Any]) -> None:
    fake = FakeTournament(report)
    box = make_box(tmp_path, tournament=fake)
    assert box.call("run_task_tournament", {"plan_id": "plan-000000000000"}).code == "not_found"
    refused = plan(box, "u1_fold_shirt")
    assert refused["status"] == "refused" and refused["task_id"] is None and refused["unsupported"]
    out = box.call("run_task_tournament", {"plan_id": refused["plan_id"]})
    assert out.code == "invalid_args" and "not valid" in out.error
    other = make_box(tmp_path / "other", tournament=fake, session_id="sess-00000000000f")
    theirs = plan(other)
    assert box.call("run_task_tournament", {"plan_id": theirs["plan_id"]}).code == "not_found"
    assert fake.calls == []


def test_the_wall_time_cap_refuses_a_slow_tournament(tmp_path: Path, report: dict[str, Any]) -> None:
    clock = FakeClock()

    def slow(pool: WorkerPool) -> None:
        assert pool.map(str, [1, 2]) == ["1", "2"]
        clock.t += 901.0  # the screen took longer than the cap
        pool.map(str, [3])  # the finalists' map is refused before it starts

    fake = FakeTournament(report, hook=slow)
    box = make_box(tmp_path, tournament=fake, clock=clock)
    planned = plan(box)
    out = box.call("run_task_tournament", {"plan_id": planned["plan_id"]})
    assert out.refused and out.code == "wall_time_cap" and "900 s" in out.error
    assert fake.calls[0]["pool"].timed_out and box.verdicts() == []
    assert not (box.store.root / "studio" / "tournaments").exists()
    fake.hook = None  # the lock was released: the next tournament runs
    ok(box.call("run_task_tournament", {"plan_id": planned["plan_id"]}))


def test_the_deadline_pool_checks_between_items_in_process() -> None:
    clock = FakeClock()
    seen: list[int] = []

    def step(x: int) -> int:
        seen.append(x)
        clock.t += 6.0
        return x

    pool = DeadlinePool(1, wall_s=10.0, clock=clock)
    assert isinstance(pool, WorkerPool) and DeadlinePool(9, wall_s=1.0).workers == HARD_MAX_WORKERS
    with pytest.raises(WallTimeCapError):
        pool.map(step, [1, 2, 3])
    assert seen == [1, 2]
    with pytest.raises(WallTimeCapError):
        pool.map(step, [])  # past the deadline even an empty map is refused


def test_the_deadline_pool_terminates_a_real_process_pool() -> None:
    pool = DeadlinePool(2, wall_s=1.0)
    t0 = time.monotonic()
    with pytest.raises(WallTimeCapError):
        pool.map(time.sleep, [30.0, 30.0])
    assert time.monotonic() - t0 < 20.0
    assert pool._pool is None and pool.timed_out
    deadline = time.monotonic() + 10.0
    while mp.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert mp.active_children() == []


def test_one_tournament_at_a_time(tmp_path: Path, report: dict[str, Any]) -> None:
    entered, release = threading.Event(), threading.Event()

    def hold(pool: WorkerPool) -> None:
        entered.set()
        assert release.wait(10.0)

    first = make_box(tmp_path / "a", tournament=FakeTournament(report, hook=hold))
    second = make_box(tmp_path / "b", tournament=FakeTournament(report), session_id="sess-00000000000b")
    plan_a, plan_b = plan(first)["plan_id"], plan(second)["plan_id"]
    outcomes: list[Any] = []
    worker = threading.Thread(
        target=lambda: outcomes.append(first.call("run_task_tournament", {"plan_id": plan_a}))
    )
    worker.start()
    try:
        assert entered.wait(10.0)
        busy = second.call("run_task_tournament", {"plan_id": plan_b})
        assert busy.refused and busy.code == "tournament_busy"
    finally:
        release.set()
        worker.join(10.0)
    assert outcomes and outcomes[0].ok
    ok(second.call("run_task_tournament", {"plan_id": plan_b}))


# --------------------------------------------------------------------------- export_build_pack


def tournament(box: Toolbox) -> dict[str, Any]:
    return ok(box.call("run_task_tournament", {"plan_id": plan(box)["plan_id"]}))


def test_export_packs_the_recommendation(tmp_path: Path, report: dict[str, Any]) -> None:
    box = make_box(tmp_path, tournament=FakeTournament(report))
    tour = tournament(box)
    pack = ok(box.call("export_build_pack", {"tournament_id": tour["tournament_id"]}))
    assert pack["for"] == "recommended" and pack["setup_id"] == tour["recommendation"]["setup_id"]
    assert pack["setup_label"] == tour["recommendation"]["label"] and pack["pack_id"].startswith("spack-")
    kinds = {f["kind"] for f in pack["files"]}
    assert {"mjcf", "certificate", "bom"} <= kinds
    assert all(f["bytes"] and f["bytes"] > 0 and f["license"] for f in pack["files"])
    assert pack["n_files"] == len(pack["files"]) and pack["zip_bytes"] > 0
    assert isinstance(pack["links"], int) and pack["caveats"] > 0
    zip_file = box.store.root / "studio" / "packs" / f"{pack['pack_id']}.zip"
    assert zip_file.stat().st_size == pack["zip_bytes"]
    assert_clean(pack, tmp_path)
    best = tour["best_single"]["setup_id"]
    chosen = ok(box.call("export_build_pack", {"tournament_id": tour["tournament_id"], "setup_id": best}))
    assert chosen["for"] == "chosen" and chosen["setup_id"] == best
    missing = box.call("export_build_pack", {"tournament_id": tour["tournament_id"], "setup_id": "0" * 12})
    assert missing.code == "not_found"
    assert box.call("export_build_pack", {"tournament_id": "tour-000000000000"}).code == "not_found"
    packs = sorted(p.name for p in (box.store.root / "studio" / "packs").iterdir())
    assert len(packs) == 4  # two folders and their zips; nothing left of the failed export


def test_export_without_a_recommendation_packs_the_closest_setup(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    data = without_winner(report)
    box = make_box(tmp_path, tournament=FakeTournament(data))
    tour = tournament(box)
    pack = ok(box.call("export_build_pack", {"tournament_id": tour["tournament_id"]}))
    assert pack["for"] == "closest" and "no recommendation" in pack["note"]
    assert pack["setup_id"] == data["recommendation"]["ranking"][0]["setup_id"][:12]
    assert_clean(pack, tmp_path)


def test_the_zip_size_cap_refuses_and_deletes_the_pack(tmp_path: Path, report: dict[str, Any]) -> None:
    box = make_box(
        tmp_path, tournament=FakeTournament(report), caps=StudioCaps(workers=1, max_zip_bytes=1000)
    )
    tour = tournament(box)
    out = box.call("export_build_pack", {"tournament_id": tour["tournament_id"]})
    assert out.refused and out.code == "pack_size_cap" and "deleted" in out.error
    assert list((box.store.root / "studio" / "packs").iterdir()) == []


# --------------------------------------------------------------------------- sessions


def test_a_studio_session_offers_only_the_studio_tools(tmp_path: Path) -> None:
    box = make_box(tmp_path)
    out = box.call("inspect_part", {"part_id": "part-0123456789ab"})
    assert out.refused and out.code == "not_offered"
    ledger = ok(box.call("cost_ledger", {}))
    assert ledger["studio_caps"]["workers"] == 1 and ledger["tool_calls_used"] == 2
    names = [t["function"]["name"] for t in box.studio.openai_tools()]
    assert tuple(names) == STUDIO_SESSION_TOOLS
    prompt = studio_system_prompt(AgentCaps(), StudioCaps())
    for needle in ("120 words", "verify on hardware", "product mode", "900 s", "4 worker", "$0.05"):
        assert needle in prompt


def test_a_scripted_studio_session_is_recorded_with_zero_spend(
    tmp_path: Path, report: dict[str, Any]
) -> None:
    transport = ScriptedStudioTransport()
    result, path = record_studio_session(
        "h1_tote", tmp_path / "rec", workers=1, transport=transport, tournament=FakeTournament(report)
    )
    assert result.status == "completed", result.reason
    doc = load_session(path)
    assert doc["label"] == PRODUCT_MODE_LABEL and doc["status"] == "completed" and doc["case_id"] == "h1_tote"
    types = [e["type"] for e in doc["events"]]
    assert types[0] == "session.started" and types[-1] == "session.finished"
    started = doc["events"][0]
    assert started["kind"] == "studio" and started["use_case"] == hero_text("h1_tote")
    assert started["models"]["agent"] == SCRIPTED_MODEL and started["unlocked"] is False
    assert started["tools"] == list(STUDIO_SESSION_TOOLS)
    called = [e["name"] for e in doc["events"] if e["type"] == "tool.call"]
    assert called == list(STUDIO_TOOL_NAMES)
    results = [e for e in doc["events"] if e["type"] == "tool.result"]
    assert [e["name"] for e in results] == list(STUDIO_TOOL_NAMES) and all(e["ok"] for e in results)
    assert {e["step"] for e in doc["events"] if e["type"] == "summary"} >= set(STUDIO_TOOL_NAMES)
    assert doc["spend"]["spent_usd"] == 0 and doc["spend"]["reserved_usd"] == 0
    assert all(e["cost_usd"] == 0 for e in doc["events"] if e["type"] == "llm.call")
    assert (
        doc["answer"].startswith("Recommended setup: Suction cup") and "verify on hardware" in doc["answer"]
    )
    assert [v["kind"] for v in doc["verdicts"]] == ["studio_recommendation"]
    agent = [p for p in transport.payloads if p.get("tool_choice") == "auto"]
    assert agent and all(
        [t["function"]["name"] for t in p["tools"]] == list(STUDIO_SESSION_TOOLS) for p in agent
    )
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
    assert (tmp_path / "rec" / "store" / "studio" / "packs").is_dir()


def test_the_record_command_checks_its_limits(tmp_path: Path) -> None:
    for argv in (
        ["record", "--case", "h1_tote", "--out", str(tmp_path), "--workers", "5"],
        ["record", "--case", "h1_tote", "--out", str(tmp_path), "--wall-s", "1801"],
        ["record", "--case", "h1_tote", "--out", str(tmp_path), "--live"],
        ["record", "--case", "u1_fold_shirt", "--out", str(tmp_path)],
    ):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2
    with pytest.raises(ValueError, match="env_file"):
        record_studio_session("h1_tote", tmp_path, live=True)


def test_sessions_without_studio_are_unchanged(tmp_path: Path) -> None:
    backend = FakePartBackend(tmp_path / "parts")
    trimesh.creation.box(extents=(40.0, 30.0, 60.0)).export(tmp_path / "box.stl")
    backend.intake(tmp_path / "box.stl", part_id="part-0123456789ab", name="box")
    caps = AgentCaps()
    guard = SpendGuard(caps.session_usd, DailySpendBook(None, caps.day_usd), SESSION)
    box = Toolbox(
        session_id=SESSION,
        part_id="part-0123456789ab",
        backend=backend,
        proposer=None,  # type: ignore[arg-type]
        runners={"local": LocalJobRunner(backend.evaluate)},
        store=AgentStore(tmp_path / "agent"),
        caps=caps,
        guard=guard,
        unlocked=False,
    )
    assert box.studio is None and "plan_task" not in box._handlers
    out = box.call("plan_task", {"text": hero_text("h1_tote")})
    assert out.code == "invalid_args" and "unknown tool" in out.error
    ledger = ok(box.call("cost_ledger", {}))
    assert "studio_caps" not in ledger and box.verdicts() == []
    inspected = ok(box.call("inspect_part", {"part_id": "part-0123456789ab"}))
    assert inspected["part"]["part_id"] == "part-0123456789ab"
    assert len(openai_tools()) == 8


def test_a_long_summary_is_cut_at_a_sentence_never_mid_word() -> None:
    text = "First sentence here. Second one is long " + "word " * 100
    out = whole_sentences(text, 400)
    assert out == "First sentence here." and len(out) <= 400
    assert whole_sentences("no stop " * 80, 40).endswith(" ...")
    assert whole_sentences("Short.", 400) == "Short."
