"""Arms C and Z: tool schema, prompt content, repair, diversity reflection, fallback and call caps."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rtt.design.schema import GEOMETRY_PARAMS, FingerDesign, load_entrants
from rtt.llm.client import LLMAuthError, LLMClient, TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.search.base import BudgetLedger, Evaluated, Proposal, RunContext, derive_seed
from rtt.search.diversity import MIN_DISTANCE, nearest
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import (
    PROPOSER_SLOTS,
    NemotronProposer,
    PromptLeakError,
    assert_prompt_clean,
    make_proposer_client,
    render_task,
    submit_designs_tool,
)
from rtt.search.random_search import DesignSampler, round0_designs
from rtt.search.runner import DEFAULT_ENTRANTS, SearchConfig, run_search
from rtt.search.zeroshot import ZEROSHOT_SLOTS

Reply = Mapping[str, Any] | str | TransportResponse | Callable[[Mapping[str, Any]], Any]


class ReplyTransport:
    """Answers each call with the next scripted reply: tool arguments, ``"garbage"`` or a raw response."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.payloads: list[dict[str, Any]] = []

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        if not self.replies:
            raise AssertionError("unexpected model call")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(payload)
        if isinstance(reply, TransportResponse):
            return reply
        usage = {"prompt_tokens": 1000, "completion_tokens": 200}
        if reply == "garbage":
            message: dict[str, Any] = {"role": "assistant", "content": "Here are some thoughts, no call."}
        else:
            call = {
                "id": "c",
                "type": "function",
                "function": {"name": "submit_designs", "arguments": json.dumps(reply)},
            }
            message = {"role": "assistant", "content": None, "tool_calls": [call]}
        body = {
            "id": "r",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage,
        }
        return TransportResponse(200, {}, body)

    def user_text(self, i: int) -> str:
        return self.payloads[i]["messages"][-1]["content"]


def client_for(transport: Any, cache_dir: Path | None = None) -> LLMClient:
    return make_proposer_client(cache_dir, transport=transport, limiter=TokenBucket(1e12))


def round0_context(seed: int = 5, cap: int = 16, arm: str = "nemotron") -> tuple[RunContext, list[Evaluated]]:
    ev = FakeEvaluator(0)
    info = ev.describe()
    ctx = RunContext(arm, 0, seed, info.gripper_task, info, BudgetLedger(40, 8, 8, cap))
    r0 = round0_designs(load_entrants(DEFAULT_ENTRANTS), seed, 8, info.gripper_task)
    ctx.budget.charge(0, 8)
    items = [ctx.history.add(0, p, s) for p, s in zip(r0, ev.evaluate([p.design for p in r0]), strict=True)]
    return ctx, items


def diverse_designs(ctx: RunContext, n: int, seed: int = 99) -> list[FingerDesign]:
    """``n`` valid designs that pass the diversity check against the run and each other."""
    chosen: list[FingerDesign] = []
    sampler = DesignSampler(seed, ctx.gripper_task)
    refs = ctx.history.designs()

    def far(design: FingerDesign) -> bool:
        return nearest(design, refs + chosen).distance >= MIN_DISTANCE + 0.02

    for _ in range(n):
        chosen += sampler.sample(1, accept=far)
    return chosen


def slot_args(
    design: FingerDesign, parents: tuple[str, ...] = (), hypothesis: str = "because"
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if parents:
        out["parents"] = list(parents)
    out["hypothesis"] = hypothesis
    out.update(design.geometry_params())
    return out


def submission(designs: list[FingerDesign], analysis: str = "analysis") -> dict[str, Any]:
    parents = (("d01",), ("d02",), ("d01", "d03"), ())
    args: dict[str, Any] = {"analysis": analysis}
    for spec, design, par in zip(PROPOSER_SLOTS, designs, parents, strict=True):
        args[spec.name] = slot_args(design, par, f"{spec.name} idea")
    return args


# --------------------------------------------------------------------------- schema and prompt


def test_tool_schema_orders_analysis_first_and_fixes_operator_quota() -> None:
    tool = submit_designs_tool(PROPOSER_SLOTS)
    params = tool["function"]["parameters"]
    assert list(params["properties"]) == ["analysis", "mutate_1", "mutate_2", "crossover", "explore"]
    assert params["required"] == list(params["properties"])
    mutate = params["properties"]["mutate_1"]
    assert list(mutate["properties"])[:2] == ["parents", "hypothesis"]
    assert mutate["properties"]["parents"]["minItems"] == mutate["properties"]["parents"]["maxItems"] == 1
    assert params["properties"]["crossover"]["properties"]["parents"]["minItems"] == 2
    assert "parents" not in params["properties"]["explore"]["properties"]
    assert set(GEOMETRY_PARAMS) <= set(mutate["properties"])
    assert mutate["properties"]["inner_face"]["enum"] == ["flat", "concave_arc", "v_groove"]
    assert mutate["properties"]["finger_length_mm"] == {"type": "number", "minimum": 40.0, "maximum": 120.0}
    zs = submit_designs_tool(ZEROSHOT_SLOTS)["function"]["parameters"]["properties"]
    assert list(zs) == ["analysis", "design_1", "design_2", "design_3", "design_4"]


def test_prompt_has_every_section_with_elites_ascending() -> None:
    ctx, items = round0_context()
    strategy = NemotronProposer(ctx, client_for(ReplyTransport()))
    strategy.tell(items)
    system, user = (m["content"] for m in strategy.build_messages(1, 0))
    assert "submit_designs" in system and "2 independent design teams" in system
    headings = [line for line in user.splitlines() if line.startswith("# ")]
    assert headings == [
        "# Task",
        "# Design parameters",
        "# Validator rules",
        "# Tested designs (top 8 of 8, worst first, best last)",
        "# Informative failures",
        "# Validator rejections in your last round",
        "# Target",
        "# Round 1 of 4: what to submit",
    ]
    elites = user.split("# Tested designs")[1].split("# Informative failures")[0]
    lines = [line for line in elites.splitlines() if re.match(r"^d\d\d \[", line)]
    assert len(lines) == 8
    shown = [int(re.search(r"score (\d+)% at k=", line).group(1)) for line in lines]
    assert shown == sorted(shown)
    best = ctx.history.best()
    assert lines[-1].startswith(best.label)
    for line in lines:
        assert re.search(r"\| box \d+%, cylinder \d+%, rod \d+% \|", line)
        assert "failures: " in line and re.search(r"mass \d+\.\d g$", line)
    assert '{"finger_length_mm": ' in elites
    failures = user.split("# Informative failures")[1].split("# Validator rejections")[0]
    assert 1 <= len([line for line in failures.splitlines() if line.startswith("- ")]) <= 4
    target = round(100 * min(1.0, 1.2 * best.score.success))
    assert f"{target}%" in user.split("# Target")[1].split("# Round")[0]
    assert "Validator rejections in your last round\n\nNone." in user
    for c in ctx.info.conditions:
        assert c.condition_id in user
    assert_prompt_clean([{"content": user}])


def test_prompt_guard_refuses_held_out_mentions() -> None:
    for text in ("the held-out suite", "HELDOUT conditions", "read secrets/heldout_seed.txt", "held out set"):
        with pytest.raises(PromptLeakError):
            assert_prompt_clean([{"role": "user", "content": text}])


# --------------------------------------------------------------------------- one round


def test_clean_round_uses_every_llm_design() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 8)
    transport = ReplyTransport(submission(designs[:4]), submission(designs[4:]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert [p.design.geometry_params() for p in proposals] == [d.geometry_params() for d in designs]
    assert [p.operator for p in proposals] == ["mutate", "mutate", "crossover", "explore"] * 2
    assert [p.island for p in proposals] == [0] * 4 + [1] * 4
    assert proposals[2].parents == ("d01", "d03") and proposals[0].parents == ("d01",)
    assert proposals[3].hypothesis == "explore idea" and proposals[3].design.rationale == "explore idea"
    assert {p.via for p in proposals} == {"main"} and {p.origin for p in proposals} == {"nemotron"}
    assert len(transport.payloads) == 2 and ctx.budget.llm_calls == 2
    assert ctx.rejections == [] and ctx.budget.fallbacks == 0
    assert [c.kind for c in ctx.llm_calls] == ["main", "main"]
    assert ctx.llm_calls[0].analysis == "analysis" and ctx.llm_calls[0].request_hash


def test_invalid_slot_is_repaired_with_validator_messages() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 9)
    broken = submission(designs[:4])
    broken["mutate_2"]["inner_face"], broken["mutate_2"]["inner_face_depth_mm"] = "flat", 3.0
    fixed = submission([designs[0], designs[8], designs[2], designs[3]])
    transport = ReplyTransport(broken, fixed, submission(designs[4:8]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    repair_prompt = transport.user_text(1)
    assert "# Correction needed" in repair_prompt
    assert "mutate_2: inner_face='flat' requires inner_face_depth_mm=0" in repair_prompt
    assert "Copy these slots unchanged: mutate_1, crossover, explore." in repair_prompt
    assert proposals[1].via == "repair" and proposals[1].design.design_id() == designs[8].design_id()
    assert [p.via for p in proposals].count("main") == 7
    assert [(r.reason, r.stage, r.slot) for r in ctx.rejections] == [("invalid", "main", "mutate_2")]
    assert [c.kind for c in ctx.llm_calls] == ["main", "repair", "main"]


def test_unparseable_reply_is_retried_once_then_falls_back() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 4)
    transport = ReplyTransport("garbage", submission(designs), "garbage", "garbage")
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert "did not contain a usable submit_designs call" in transport.user_text(1)
    assert [p.via for p in proposals[:4]] == ["repair"] * 4
    assert [p.origin for p in proposals[4:]] == ["fallback_random"] * 4
    assert {p.fallback_reason for p in proposals[4:]} == {"parse_error"}
    assert ctx.budget.fallbacks == 4
    assert sum(r.reason == "parse_error" for r in ctx.rejections) == 4 + 4 + 4
    refs = ctx.history.designs()
    for i, p in enumerate(proposals):
        assert nearest(p.design, refs + [q.design for q in proposals[:i]]).distance >= MIN_DISTANCE


def test_near_duplicate_triggers_one_reflection_call() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 9)
    copy = items[0].design  # d01, an entrant
    first = submission([designs[0], designs[1], designs[2], copy])
    transport = ReplyTransport(first, submission(designs[4:8]), submission([*designs[:3], designs[8]]))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    reflection = transport.user_text(2)
    assert "# Too similar to designs already tested" in reflection
    assert "- explore:" in reflection and "from d01" in reflection
    assert "Copy these slots unchanged: mutate_1, mutate_2, crossover." in reflection
    assert proposals[3].via == "reflection" and proposals[3].design.design_id() == designs[8].design_id()
    assert [(r.reason, r.slot) for r in ctx.rejections] == [("too_similar", "explore")]
    assert [c.kind for c in ctx.llm_calls] == ["main", "main", "reflection"]


def test_auth_errors_abort_the_run() -> None:
    ctx, items = round0_context()
    transport = ReplyTransport(TransportResponse(401, {}, {"error": {"message": "bad key"}}))
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    with pytest.raises(LLMAuthError):
        strategy.ask(1, 8)


def test_http_errors_fall_back_without_repair() -> None:
    ctx, items = round0_context()
    bad = TransportResponse(400, {}, {"error": {"message": "bad request"}})
    transport = ReplyTransport(bad, bad)
    strategy = NemotronProposer(ctx, client_for(transport))
    strategy.tell(items)
    proposals = strategy.ask(1, 8)
    assert {p.origin for p in proposals} == {"fallback_random"}
    assert {p.fallback_reason for p in proposals} == {"call_failed"}
    assert [c.ok for c in ctx.llm_calls] == [False, False]


# --------------------------------------------------------------------------- whole runs


def test_call_cap_never_starves_main_calls(tmp_path: Path) -> None:
    always_garbage = ReplyTransport(*(["garbage"] * 40))
    out = run_search("nemotron", 1, 40, [3], FakeEvaluator(0), tmp_path, client_for(always_garbage))
    calls = out.runs[0].ctx.llm_calls
    assert len(calls) == 16
    for r in range(1, 5):
        assert [c.kind for c in calls if c.round == r].count("main") == 2
    assert out.runs[0].ctx.budget.fallbacks == 32


def test_low_call_cap_falls_back_once_spent(tmp_path: Path) -> None:
    config = SearchConfig(llm_call_cap=4)
    out = run_search(
        "nemotron", 1, 40, [3], FakeEvaluator(0), tmp_path, client_for(ScriptedProposer(2)), config=config
    )
    ctx = out.runs[0].ctx
    assert len(ctx.llm_calls) == 4 and {c.kind for c in ctx.llm_calls} == {"main"}
    late = [e.proposal for e in ctx.history if e.round >= 3]
    assert len(late) == 16 and {p.origin for p in late} == {"fallback_random"}
    assert {p.fallback_reason for p in late} == {"llm_call_cap"}


def test_islands_keep_separate_lineages_until_migration() -> None:
    ctx, items = round0_context()
    ev = FakeEvaluator(0)
    strategy = NemotronProposer(ctx, client_for(ReplyTransport()))
    strategy.tell(items)
    for round_index in (1, 2):
        designs = diverse_designs(ctx, 8, seed=round_index)
        ctx.budget.charge(round_index, 8)
        new = [
            ctx.history.add(round_index, Proposal(d, "nemotron", "explore", i // 4, "explore"), s)
            for i, (d, s) in enumerate(zip(designs, ev.evaluate(designs), strict=True))
        ]
        strategy.tell(new)
        pools = [{e.index for e in strategy.pool(i)} for i in (0, 1)]
        own = [{e.index for e in new if e.proposal.island == i} for i in (0, 1)]
        shared = set(range(1, 9))
        if round_index == 1:
            assert pools[0] == shared | own[0] and pools[1] == shared | own[1]
    # After round 2 each island also holds the other island's best design (one migration).
    own_pools = [[e for e in ctx.history if e.round == 0 or e.proposal.island == i] for i in (0, 1)]
    own_best = [ctx.history.ranked(pool)[0].index for pool in own_pools]
    for i in (0, 1):
        assert {e.index for e in strategy.pool(i)} == {e.index for e in own_pools[i]} | {own_best[1 - i]}


def test_zeroshot_prompts_never_depend_on_results(tmp_path: Path) -> None:
    hashes = []
    for landscape in (0, 1):
        transport = ScriptedProposer(4)
        out = run_search(
            "zeroshot",
            1,
            40,
            [9],
            FakeEvaluator(landscape),
            tmp_path / f"z{landscape}",
            client_for(transport),
        )
        hashes.append([c.request_hash for c in out.runs[0].ctx.llm_calls])
        for payload in transport.payloads:
            text = payload["messages"][-1]["content"]
            assert not re.search(r"score \d+%|failures:|Informative failures|Target", text)
    assert hashes[0] == hashes[1]
    c_hashes = []
    for landscape in (0, 1):
        out = run_search(
            "nemotron",
            1,
            40,
            [9],
            FakeEvaluator(landscape),
            tmp_path / f"c{landscape}",
            client_for(ScriptedProposer(4)),
        )
        c_hashes.append([c.request_hash for c in out.runs[0].ctx.llm_calls])
    assert c_hashes[0][0] != c_hashes[1][0]  # arm C does see results


# --------------------------------------------------------------------------- robustness


class HeldOutWords(ScriptedProposer):
    """A model whose own words trip the prompt guard: "held outward" hypotheses, odd enum values."""

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        response = super().chat_completions(payload, timeout_s)
        message = response.body["choices"][0]["message"]
        if message.get("tool_calls"):
            args = json.loads(message["tool_calls"][0]["function"]["arguments"])
            for name, value in args.items():
                if isinstance(value, dict):
                    value["hypothesis"] = (
                        "The lip keeps the rod held outward, away from secrets/ of the palm."
                    )
                    if name == "explore":
                        value["inner_face"] = "held-out groove"  # invalid: echoed by the validator
            message["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
        return response


def test_model_text_that_trips_the_guard_is_scrubbed_not_fatal(tmp_path: Path) -> None:
    model = HeldOutWords(6)
    out = run_search("nemotron", 1, 40, [12], FakeEvaluator(0), tmp_path, client_for(model))
    assert out.manifest["status"] == "complete"
    later = [p["messages"][-1]["content"] for p in model.payloads[2:]]
    assert any("[removed]" in text for text in later)  # quoted back, with the words removed
    for payload in model.payloads:
        assert_prompt_clean(payload["messages"])
    assert any(r.reason == "invalid" for r in out.runs[0].ctx.rejections)


def test_request_key_matches_the_client_cache_key() -> None:
    ctx, items = round0_context()
    designs = diverse_designs(ctx, 8)
    strategy = NemotronProposer(
        ctx, client_for(ReplyTransport(submission(designs[:4]), submission(designs[4:])))
    )
    strategy.tell(items)
    strategy.ask(1, 8)
    messages = strategy.build_messages(1, 0)
    sample = derive_seed(ctx.run_seed, ctx.arm, 1, 0, "main") % (2**31)
    assert strategy.request_key(messages, sample) == ctx.llm_calls[0].request_hash


def test_task_text_describes_the_reachable_grasp_band() -> None:
    info = FakeEvaluator(0).describe()
    text = render_task(info, info.gripper_task)
    assert "palm always stays at least 10 mm above the object's top" in text
    assert "0.3, 0.5, 0.7 of the way up from the lowest height they can reach" in text
    assert "shorter fingers grasp tall objects higher up" in text
