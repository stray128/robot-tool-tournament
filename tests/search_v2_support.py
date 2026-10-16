"""Shared helpers for the freeform-v2 search tests (scripted replies, a round-0 run context, recipes)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rtt.llm.client import LLMClient, TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.search.base import BudgetLedger, Evaluated, RunContext
from rtt.search.diversity import EXPLORE_MIN_DISTANCE_V2, MIN_DISTANCE_V2, explore_check, screen
from rtt.search.fakes_v2 import FakeEvaluatorV2, stub_entrants
from rtt.search.nemotron import PROPOSER_SLOTS, make_proposer_client
from rtt.search.random_search import round0_designs
from rtt.search.space import V2, get_space
from rtt.search.space_v2 import INACTIVE, NUMERIC_PARAMS

SPACE = get_space(V2)
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
        if reply == "garbage":
            message: dict[str, Any] = {"role": "assistant", "content": "Some thoughts, no call."}
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
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        }
        return TransportResponse(200, {}, body)

    def user_text(self, i: int) -> str:
        return self.payloads[i]["messages"][-1]["content"]


def client_for(transport: Any, cache_dir: Path | None = None) -> LLMClient:
    return make_proposer_client(cache_dir, transport=transport, limiter=TokenBucket(1e12))


def v2_round0(seed: int = 5, cap: int = 16, arm: str = "nemotron") -> tuple[RunContext, list[Evaluated]]:
    """A v2 run context whose round 0 (stub entrants + Sobol fill) is simulated and in the history."""
    ev = FakeEvaluatorV2(0)
    info = ev.describe()
    ctx = RunContext(
        arm,
        0,
        seed,
        info.gripper_task,
        info,
        BudgetLedger(40, 8, 8, cap),
        space=SPACE,
        shared_min_distance=MIN_DISTANCE_V2,
    )
    r0 = round0_designs(
        stub_entrants(), seed, 8, info.gripper_task, space=SPACE, min_distance=MIN_DISTANCE_V2
    )
    ctx.budget.charge(0, 8)
    items = [ctx.history.add(0, p, s) for p, s in zip(r0, ev.evaluate([p.design for p in r0]), strict=True)]
    return ctx, items


def diverse_recipes(ctx: RunContext, n: int, seed: int = 99) -> list[Any]:
    """``n`` valid recipes that pass the v2 screen and the explore rule against the run and each other."""
    chosen: list[Any] = []
    sampler = SPACE.sample_sobol(seed, ctx.gripper_task)
    refs = ctx.history.designs()

    def far(recipe: Any) -> bool:
        if screen(recipe, refs + chosen, None, MIN_DISTANCE_V2 + 0.02, space=SPACE, context=ctx.gripper_task):
            return False
        return explore_check(recipe, refs + chosen, None, EXPLORE_MIN_DISTANCE_V2 + 0.02, space=SPACE)[0]

    for _ in range(n):
        chosen += sampler.sample(1, accept=far)
    return chosen


def active_params(recipe: Any) -> dict[str, float]:
    params = SPACE.params(recipe)
    out = {}
    for name in NUMERIC_PARAMS:
        rule = INACTIVE.get(name)
        if rule is not None and params[rule[0]] in rule[1]:
            continue
        out[name] = params[name]
    return out


def recipe_slot(
    recipe: Any, parents: tuple[str, ...] = (), hypothesis: str = "because", **extra: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if parents:
        out["parents"] = list(parents)
    out["hypothesis"] = hypothesis
    params = SPACE.params(recipe)
    out["pad"], out["tip"] = params["pad"], params["tip"]
    out["params"] = {**active_params(recipe), **extra}
    return out


def submission_v2(recipes: list[Any], analysis: str = "analysis") -> dict[str, Any]:
    parents = (("d05",), ("d06",), ("d05", "d07"), ())
    args: dict[str, Any] = {"analysis": analysis}
    for spec, recipe, par in zip(PROPOSER_SLOTS, recipes, parents, strict=True):
        args[spec.name] = recipe_slot(recipe, par, f"{spec.name} idea")
    return args
