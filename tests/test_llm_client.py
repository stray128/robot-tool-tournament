"""Offline tests for the Token Factory client, cache and ledger (no network)."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx2
import pytest

from rtt.llm import (
    NEMOTRON_NANO,
    NEMOTRON_SUPER,
    BudgetExceededError,
    CacheMiss,
    Ledger,
    LLMAuthError,
    LLMBillingError,
    LLMClient,
    LLMHTTPError,
    LLMParseError,
    LLMTransportError,
    ResponseCache,
    RetryPolicy,
    TokenBucket,
    request_hash,
)
from rtt.llm.cache import wire_json
from rtt.llm.client import estimate_prompt_tokens, extract_json_object, parse_duration_s
from rtt.llm.ledger import read_ledger
from rtt.llm.providers import OpenAITransport, TransportError, TransportResponse, redact

MESSAGES = [
    {"role": "system", "content": "You design gripper fingers."},
    {"role": "user", "content": "Propose one design."},
]
TOOL = {
    "type": "function",
    "function": {
        "name": "submit_designs",
        "description": "Submit designs.",
        "parameters": {
            "type": "object",
            "properties": {"length_mm": {"type": "number"}},
            "required": ["length_mm"],
        },
    },
}
SCHEMA = TOOL["function"]["parameters"]


def completion(
    *,
    content: str | None = None,
    tool_args: str | dict[str, Any] | None = None,
    tool_name: str = "submit_designs",
    reasoning: str | None = None,
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_args is not None:
        message["tool_calls"] = [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }
        ]
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": NEMOTRON_SUPER,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class FakeTransport:
    """Replays a scripted sequence of responses and records every payload."""

    def __init__(self, *responses: TransportResponse | Exception) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        if not self.responses:
            raise AssertionError("unexpected live call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})


def ok(body: dict[str, Any], headers: dict[str, str] | None = None) -> TransportResponse:
    return TransportResponse(200, headers or {}, body)


def err(status: int, message: str = "boom", headers: dict[str, str] | None = None) -> TransportResponse:
    return TransportResponse(status, headers or {}, {"error": {"message": message}})


class Sleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_client(transport: FakeTransport, **kwargs: Any) -> tuple[LLMClient, Sleeper]:
    sleeper = Sleeper()
    kwargs.setdefault("retry", RetryPolicy(max_retries=3, base_delay_s=1.0, jitter_frac=0.0))
    client = LLMClient(
        kwargs.pop("model", NEMOTRON_SUPER),
        temperature=0.6,
        top_p=0.95,
        max_tokens=kwargs.pop("max_tokens", 2000),
        transport=transport,
        sleep=sleeper,
        limiter=kwargs.pop("limiter", TokenBucket(10**12)),
        **kwargs,
    )
    return client, sleeper


# --------------------------------------------------------------------------- request shape


def test_call_tool_forces_named_function_and_parses_arguments() -> None:
    transport = FakeTransport(ok(completion(tool_args='{"length_mm": 80}')))
    client, _ = make_client(transport)

    args, meta = client.call_tool(MESSAGES, TOOL)

    assert args == {"length_mm": 80}
    sent = transport.payloads[0]
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "submit_designs"}}
    assert sent["tools"] == [TOOL]
    assert sent["model"] == NEMOTRON_SUPER
    assert (sent["temperature"], sent["top_p"], sent["max_tokens"]) == (0.6, 0.95, 2000)
    assert "seed" not in sent
    assert meta.provider == "nebius-tokenfactory"
    assert meta.model == NEMOTRON_SUPER and meta.mode == "tool"
    assert (meta.prompt_tokens, meta.completion_tokens) == (1000, 500)
    assert meta.cost_usd == pytest.approx((1000 * 0.30 + 500 * 0.90) / 1e6)
    assert meta.cached is False and meta.attempts == 1
    assert meta.request_hash == client.request_hash_for(sent)


def test_call_tool_accepts_bare_function_and_auto_choice() -> None:
    transport = FakeTransport(ok(completion(tool_args={"length_mm": 60})))
    client, _ = make_client(transport)

    args, _ = client.call_tool(MESSAGES, TOOL["function"], force=False)

    assert args == {"length_mm": 60}
    assert transport.payloads[0]["tool_choice"] == "auto"
    assert transport.payloads[0]["tools"][0]["type"] == "function"


def test_call_tool_falls_back_to_json_in_content() -> None:
    transport = FakeTransport(ok(completion(content='```json\n{"length_mm": 70}\n```')))
    client, _ = make_client(transport)

    args, meta = client.call_tool(MESSAGES, TOOL)

    assert args == {"length_mm": 70}
    assert "tool_args_from_content" in meta.notes
    assert "code_fence_stripped" in meta.notes


def test_call_json_schema_sets_strict_response_format() -> None:
    transport = FakeTransport(ok(completion(content='{"length_mm": 90}')))
    client, _ = make_client(transport)

    value, meta = client.call_json_schema(MESSAGES, SCHEMA, name="design")

    assert value == {"length_mm": 90}
    assert transport.payloads[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "design", "schema": SCHEMA, "strict": True},
    }
    assert "tools" not in transport.payloads[0]
    assert meta.mode == "json_schema" and meta.notes == ()


def test_chat_returns_text_and_extra_body_is_sent() -> None:
    transport = FakeTransport(ok(completion(content="hello")))
    client, _ = make_client(transport, extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    text, meta = client.chat(MESSAGES)

    assert text == "hello"
    assert transport.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert meta.mode == "chat"


# --------------------------------------------------------------------------- reasoning


def test_separate_reasoning_is_kept_but_not_parsed() -> None:
    transport = FakeTransport(
        ok(completion(tool_args='{"length_mm": 50}', reasoning="Let me think {not json"))
    )
    client, _ = make_client(transport)

    args, meta = client.call_tool(MESSAGES, TOOL)

    assert args == {"length_mm": 50}
    assert meta.reasoning_separate is True and meta.reasoning_field == "reasoning_content"
    assert meta.reasoning_text == "Let me think {not json"
    assert "reasoning_text" not in meta.to_json()
    assert meta.to_json(include_reasoning=True)["reasoning_text"] == "Let me think {not json"


def test_inline_think_block_is_stripped_from_json_content() -> None:
    content = '<think>maybe {"length_mm": 1}?</think>\n{"length_mm": 42}'
    transport = FakeTransport(ok(completion(content=content)))
    client, _ = make_client(transport)

    value, meta = client.call_json_schema(MESSAGES, SCHEMA)

    assert value == {"length_mm": 42}
    assert meta.reasoning_separate is False and meta.reasoning_field is None
    assert meta.reasoning_text == 'maybe {"length_mm": 1}?'
    assert "think_tags_stripped" in meta.notes


def test_extract_json_object_variants() -> None:
    assert extract_json_object('{"a": 1}') == ({"a": 1}, ())
    value, notes = extract_json_object('Here you go: {"a": {"b": 2}} done')
    assert value == {"a": {"b": 2}} and notes == ("json_substring",)
    value, notes = extract_json_object('reasoning...</think>{"a": 3}')
    assert value == {"a": 3} and "think_tags_stripped" in notes
    with pytest.raises(LLMParseError):
        extract_json_object("[1, 2, 3]")
    with pytest.raises(LLMParseError):
        extract_json_object("no json here")


def test_unclosed_think_is_reasoning_not_answer() -> None:
    # max_tokens cut the model off mid-thought; the draft inside must not be taken as the answer.
    with pytest.raises(LLMParseError):
        extract_json_object('<think>I could answer {"length_mm": 999} but first')
    value, notes = extract_json_object('{"length_mm": 3}\n<think>trailing thoughts {"length_mm": 9}')
    assert value == {"length_mm": 3} and "think_tags_stripped" in notes


def test_extract_json_tries_every_fence_last_first() -> None:
    text = '```python\nx = 1\n```\nDraft:\n```json\n{"a": 1}\n```\nFinal:\n```json\n{"a": 2}\n```'
    value, notes = extract_json_object(text)
    assert value == {"a": 2} and notes == ("code_fence_stripped",)
    value, _ = extract_json_object('```json\n{"a": 5}\n```\nThen an example: ```python\nprint(1)\n```')
    assert value == {"a": 5}
    with pytest.raises(LLMParseError) as info:
        extract_json_object("```text\nnothing here\n```")
    assert "code_fence_stripped" in info.value.notes


def test_parse_error_carries_meta_and_is_ledgered() -> None:
    transport = FakeTransport(ok(completion(content="I cannot comply", finish_reason="length")))
    ledger = Ledger()
    client, _ = make_client(transport, ledger=ledger)

    with pytest.raises(LLMParseError) as info:
        client.call_tool(MESSAGES, TOOL)

    assert info.value.meta is not None
    assert info.value.meta.cost_usd > 0
    assert "truncated" in info.value.meta.notes
    assert info.value.raw_text == "I cannot comply"
    (entry,) = ledger.session_entries()
    assert entry.ok is False and entry.error is not None and entry.error.startswith("LLMParseError")
    assert ledger.spent_usd == pytest.approx(info.value.meta.cost_usd)


# --------------------------------------------------------------------------- cache


def test_request_hash_is_deterministic_and_sensitive_to_schema_order() -> None:
    a = {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5}
    assert request_hash(a) == request_hash(json.loads(json.dumps(a)))
    assert request_hash(a) != request_hash({**a, "temperature": 0.6})
    # Property order is rendered into the prompt and fixes the decoding order, so
    # "analysis before designs" and the reverse must not share a cache entry.
    first = {"type": "object", "properties": {"analysis": {}, "designs": {}}}
    reverse = {"type": "object", "properties": {"designs": {}, "analysis": {}}}
    client, _ = make_client(FakeTransport())
    hash_first = client.request_hash_for(client.build_payload(MESSAGES, {"schema": first}))
    hash_reverse = client.request_hash_for(client.build_payload(MESSAGES, {"schema": reverse}))
    assert hash_first != hash_reverse


def test_cache_stores_request_as_sent_and_rejects_tampered_files(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    schema = {"type": "object", "properties": {"z_last": {"type": "string"}, "a_first": {"type": "number"}}}
    client, _ = make_client(FakeTransport(ok(completion(content='{"z_last": "x", "a_first": 1}'))))
    client.cache = ResponseCache(root)
    _, meta = client.call_json_schema(MESSAGES, schema)

    path = client.cache.path_for(meta.request_hash)
    stored = json.loads(path.read_text())
    props = stored["request"]["payload"]["response_format"]["json_schema"]["schema"]["properties"]
    assert list(props) == ["z_last", "a_first"]  # not alphabetised
    assert request_hash(stored["request"]) == meta.request_hash

    stored["request"]["payload"]["temperature"] = 0.0  # edited after the fact
    path.write_text(json.dumps(stored))
    replayer, _ = make_client(FakeTransport(), cache=ResponseCache(root, mode="replay"))
    with pytest.raises(ValueError, match="does not hold"):
        replayer.call_json_schema(MESSAGES, schema)


def test_cache_hit_skips_transport_and_sample_index_separates_entries(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    transport = FakeTransport(
        ok(completion(tool_args='{"length_mm": 1}')),
        ok(completion(tool_args='{"length_mm": 2}')),
    )
    client, _ = make_client(transport, cache=cache, ledger=ledger)

    first, meta1 = client.call_tool(MESSAGES, TOOL)
    again, meta2 = client.call_tool(MESSAGES, TOOL)
    second, meta3 = client.call_tool(MESSAGES, TOOL, sample=1)

    assert first == again == {"length_mm": 1}
    assert second == {"length_mm": 2}
    assert len(transport.payloads) == 2
    assert meta2.cached is True and meta2.request_hash == meta1.request_hash
    assert meta2.cost_usd == meta1.cost_usd and meta2.latency_s == meta1.latency_s
    assert meta3.request_hash != meta1.request_hash
    assert len(cache) == 2
    stored = json.loads(cache.path_for(meta1.request_hash).read_text())
    assert stored["request"]["payload"]["messages"] == MESSAGES
    assert stored["request_hash"] == meta1.request_hash

    totals = ledger.totals_by_model()[NEMOTRON_SUPER]
    assert (totals.calls, totals.live_calls, totals.cached_calls) == (3, 2, 1)
    assert totals.cost_usd == pytest.approx(meta1.cost_usd + meta3.cost_usd)


def test_replay_mode_needs_no_transport_and_errors_on_miss(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    recorder, _ = make_client(
        FakeTransport(ok(completion(content='{"length_mm": 5}'))), cache=ResponseCache(root)
    )
    recorder.call_json_schema(MESSAGES, SCHEMA)

    exploding = FakeTransport()  # any live call fails the test
    replayer, _ = make_client(exploding, cache=ResponseCache(root, mode="replay"))
    value, meta = replayer.call_json_schema(MESSAGES, SCHEMA)
    assert value == {"length_mm": 5} and meta.cached is True

    with pytest.raises(CacheMiss) as info:
        replayer.call_json_schema(MESSAGES, SCHEMA, sample=7)
    assert len(info.value.request_hash) == 64
    assert exploding.payloads == []


def test_replay_reproduces_parse_failures(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    recorder, _ = make_client(FakeTransport(ok(completion(content="not json"))), cache=ResponseCache(root))
    with pytest.raises(LLMParseError):
        recorder.call_json_schema(MESSAGES, SCHEMA)
    replayer, _ = make_client(FakeTransport(), cache=ResponseCache(root, mode="replay"))
    with pytest.raises(LLMParseError) as info:
        replayer.call_json_schema(MESSAGES, SCHEMA)
    assert info.value.meta is not None and info.value.meta.cached is True


def test_cache_keeps_rate_limit_headers_of_the_original_call(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    headers = {"x-ratelimit-limit-tokens": "200000", "content-type": "application/json"}
    recorder, _ = make_client(FakeTransport(ok(completion(content="hi"), headers)), cache=ResponseCache(root))
    _, live = recorder.chat(MESSAGES)
    replayer, _ = make_client(FakeTransport(), cache=ResponseCache(root, mode="replay"))
    _, replayed = replayer.chat(MESSAGES)
    assert live.rate_limit_headers == replayed.rate_limit_headers == {"x-ratelimit-limit-tokens": "200000"}


def test_refresh_mode_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    first, _ = make_client(
        FakeTransport(ok(completion(content='{"length_mm": 1}'))), cache=ResponseCache(root)
    )
    first.call_json_schema(MESSAGES, SCHEMA)
    second, _ = make_client(
        FakeTransport(ok(completion(content='{"length_mm": 2}'))), cache=ResponseCache(root, mode="refresh")
    )
    value, meta = second.call_json_schema(MESSAGES, SCHEMA)
    assert value == {"length_mm": 2} and meta.cached is False
    third, _ = make_client(FakeTransport(), cache=ResponseCache(root, mode="replay"))
    assert third.call_json_schema(MESSAGES, SCHEMA)[0] == {"length_mm": 2}


# --------------------------------------------------------------------------- retries and errors


def test_retries_429_using_retry_after_header() -> None:
    transport = FakeTransport(
        err(429, "rate limited", {"retry-after": "3", "x-ratelimit-remaining-tokens": "0"}),
        ok(completion(tool_args='{"length_mm": 1}'), {"x-ratelimit-remaining-tokens": "9000"}),
    )
    client, sleeper = make_client(transport)

    _, meta = client.call_tool(MESSAGES, TOOL)

    assert sleeper.calls == [3.0]
    assert meta.attempts == 2
    assert meta.rate_limit_headers == {"x-ratelimit-remaining-tokens": "9000"}


def test_retries_429_using_ratelimit_reset_duration() -> None:
    transport = FakeTransport(
        err(429, "slow down", {"x-ratelimit-reset-tokens": "1m30s"}),
        ok(completion(tool_args='{"length_mm": 1}')),
    )
    client, sleeper = make_client(transport)
    client.call_tool(MESSAGES, TOOL)
    assert sleeper.calls == [60.0]  # capped at RetryPolicy.max_delay_s


def test_5xx_backs_off_exponentially_then_gives_up() -> None:
    ledger = Ledger()
    transport = FakeTransport(*[err(503, "overloaded")] * 4)
    client, sleeper = make_client(transport, ledger=ledger)

    with pytest.raises(LLMHTTPError) as info:
        client.chat(MESSAGES)

    assert info.value.status_code == 503
    assert sleeper.calls == [1.0, 2.0, 4.0]
    assert len(transport.payloads) == 4
    (entry,) = ledger.session_entries()
    assert entry.ok is False and entry.attempts == 4 and entry.cost_usd == 0


def test_ok_status_without_choices_is_retried_and_never_cached(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    transport = FakeTransport(
        ok({"error": {"message": "upstream overloaded"}}),
        ok(completion(content='{"length_mm": 4}')),
    )
    client, sleeper = make_client(transport, cache=ResponseCache(root))
    value, meta = client.call_json_schema(MESSAGES, SCHEMA)
    assert value == {"length_mm": 4} and meta.attempts == 2 and sleeper.calls == [1.0]

    always_bad = FakeTransport(*[ok({"object": "error"})] * 4)
    other, _ = make_client(always_bad, cache=ResponseCache(tmp_path / "other"))
    with pytest.raises(LLMHTTPError) as info:
        other.chat(MESSAGES)
    assert info.value.status_code == 200
    assert len(ResponseCache(tmp_path / "other")) == 0


def test_server_retry_hint_is_never_undercut_by_jitter() -> None:
    policy = RetryPolicy(jitter_frac=0.25)
    rng = random.Random(0)
    hinted = [policy.delay_s(1, rng, hint_s=4.0) for _ in range(500)]
    assert min(hinted) >= 4.0 and max(hinted) <= 5.0
    plain = [policy.delay_s(1, rng) for _ in range(500)]
    assert min(plain) < 1.0 < max(plain)  # without a hint, jitter is symmetric


def test_billing_detection_matches_whole_words_only() -> None:
    transport = FakeTransport(
        err(429, "Too many requests at the load balancer"),
        err(429, "Tokens-per-minute quota exceeded"),
        ok(completion(content="hi")),
    )
    client, sleeper = make_client(transport)
    text, meta = client.chat(MESSAGES)
    assert text == "hi" and meta.attempts == 3 and len(sleeper.calls) == 2


def test_prompt_estimate_is_an_upper_bound_for_rendered_tool_prompts() -> None:
    # Probe measurement: a tool-mode payload of 3,423 compact-JSON characters was
    # billed 1,383 prompt tokens (about 2.5 characters per token).
    client, _ = make_client(FakeTransport())
    payload = client.build_payload(
        [{"role": "user", "content": "x" * 2000}], {"tools": [TOOL], "tool_choice": "auto"}
    )
    parts = {k: payload[k] for k in ("messages", "tools")}
    assert estimate_prompt_tokens(payload) >= len(wire_json(parts)) / 2.5


def test_transport_errors_are_retried() -> None:
    transport = FakeTransport(TransportError("APITimeoutError"), ok(completion(content="hi")))
    client, sleeper = make_client(transport)
    text, meta = client.chat(MESSAGES)
    assert text == "hi" and meta.attempts == 2 and sleeper.calls == [1.0]

    always = FakeTransport(*[TransportError("APIConnectionError")] * 4)
    client, _ = make_client(always)
    with pytest.raises(LLMTransportError):
        client.chat(MESSAGES)


@pytest.mark.parametrize(
    ("response", "exc_type"),
    [
        (err(401, "invalid api key"), LLMAuthError),
        (err(403, "model not allowed"), LLMAuthError),
        (err(402, "payment required"), LLMBillingError),
        (err(429, "Insufficient balance on account"), LLMBillingError),
        (err(400, "tool_choice is not supported"), LLMHTTPError),
    ],
)
def test_non_retryable_errors_fail_fast(response: TransportResponse, exc_type: type[Exception]) -> None:
    transport = FakeTransport(response)
    client, sleeper = make_client(transport)
    with pytest.raises(exc_type) as info:
        client.call_tool(MESSAGES, TOOL)
    assert sleeper.calls == []
    assert len(transport.payloads) == 1
    assert type(info.value) is exc_type


# --------------------------------------------------------------------------- budget and ledger


def test_budget_refuses_call_before_any_network_traffic(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", budget_usd=0.001)
    transport = FakeTransport()
    # Super worst case: 2000 completion tokens * $0.90/M = $0.0018 > $0.001.
    client, _ = make_client(transport, ledger=ledger)
    with pytest.raises(BudgetExceededError):
        client.chat(MESSAGES)
    assert transport.payloads == []
    assert ledger.remaining_usd == pytest.approx(0.001)


def test_budget_is_shared_across_clients_and_tracks_actual_spend() -> None:
    ledger = Ledger(budget_usd=0.002)
    nano, _ = make_client(
        FakeTransport(ok(completion(content="a", prompt_tokens=1000, completion_tokens=1000))),
        model=NEMOTRON_NANO,
        ledger=ledger,
    )
    nano.chat(MESSAGES)
    assert ledger.spent_usd == pytest.approx((1000 * 0.06 + 1000 * 0.24) / 1e6)
    super_client, _ = make_client(FakeTransport(), ledger=ledger, max_tokens=2000)
    with pytest.raises(BudgetExceededError):
        super_client.chat(MESSAGES)


def test_unknown_model_price_is_refused_under_a_budget() -> None:
    client, _ = make_client(FakeTransport(), model="someone/unknown", ledger=Ledger(budget_usd=1.0))
    with pytest.raises(BudgetExceededError):
        client.chat(MESSAGES)


def test_ledger_is_append_only_jsonl_without_bodies(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    client, _ = make_client(
        FakeTransport(ok(completion(content="hi")), ok(completion(content="there"))),
        ledger=Ledger(path),
    )
    client.chat(MESSAGES, tag="run0/round1")
    # A new Ledger on the same file appends rather than truncating.
    client2, _ = make_client(FakeTransport(ok(completion(content="x"))), ledger=Ledger(path))
    client2.chat(MESSAGES, sample=1)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    entries = read_ledger(path)
    assert entries[0].tag == "run0/round1"
    assert all(e.provider == "nebius-tokenfactory" for e in entries)
    raw = path.read_text()
    assert "design gripper fingers" not in raw and "Propose one design" not in raw
    assert Ledger(path).total_cost_usd() == pytest.approx(2 * (1000 * 0.30 + 500 * 0.90) / 1e6)


# --------------------------------------------------------------------------- rate limiting


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_token_bucket_blocks_until_refilled() -> None:
    clock = FakeClock()
    bucket = TokenBucket(6000, clock=clock, sleep=clock.sleep)  # 100 tokens/s
    assert bucket.acquire(6000) == 0.0
    waited = bucket.acquire(500)
    assert waited == pytest.approx(5.0)
    bucket.adjust(+300)
    assert bucket.level == pytest.approx(300)
    bucket.clamp(100)
    assert bucket.level == pytest.approx(100)
    assert bucket.acquire(10**9) == pytest.approx((6000 - 100) / 100)  # clipped to capacity


def test_client_limiter_reserves_then_refunds_unused_tokens() -> None:
    clock = FakeClock()
    bucket = TokenBucket(100_000, clock=clock, sleep=clock.sleep)
    transport = FakeTransport(ok(completion(content="x", prompt_tokens=100, completion_tokens=50)))
    client, _ = make_client(transport, limiter=bucket, clock=clock)
    client.chat(MESSAGES)
    assert bucket.level == pytest.approx(100_000 - 150)


def test_client_limiter_honours_remaining_tokens_header() -> None:
    clock = FakeClock()
    bucket = TokenBucket(100_000, clock=clock, sleep=clock.sleep)
    transport = FakeTransport(
        ok(
            completion(content="x", prompt_tokens=10, completion_tokens=10),
            {"x-ratelimit-remaining-tokens": "500"},
        )
    )
    client, _ = make_client(transport, limiter=bucket, clock=clock)
    client.chat(MESSAGES)
    assert bucket.level <= 500


def test_default_limiter_is_shared_per_model_at_catalogue_tpm() -> None:
    a = LLMClient(NEMOTRON_SUPER, transport=FakeTransport())
    b = LLMClient(NEMOTRON_SUPER, temperature=0.2, transport=FakeTransport())
    assert a.limiter is not None and a.limiter is b.limiter
    assert a.limiter.capacity == pytest.approx(150_000)
    assert LLMClient(NEMOTRON_NANO, transport=FakeTransport()).limiter is not a.limiter
    assert LLMClient("someone/unknown", transport=FakeTransport()).limiter is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", 2.0),
        ("0.5", 0.5),
        ("250ms", 0.25),
        ("1s", 1.0),
        ("6m0s", 360.0),
        ("1h2m3.5s", 3723.5),
        ("Wed, 21 Oct 2015 07:28:00 GMT", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_duration(value: str | None, expected: float | None) -> None:
    assert parse_duration_s(value) == expected


# --------------------------------------------------------------------------- real transport, mocked HTTP

SECRET = "sk-test-not-a-real-key-123456"


def _sdk_transport(handler: Any) -> OpenAITransport:
    return OpenAITransport(
        SECRET,
        "https://tokenfactory.invalid/v1/",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def test_openai_transport_forwards_payload_verbatim() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        return httpx2.Response(
            200,
            headers={"x-ratelimit-remaining-tokens": "199000"},
            json=completion(tool_args='{"length_mm": 77}'),
        )

    client, _ = make_client(FakeTransport())
    client._transport = _sdk_transport(handler)
    args, meta = client.call_tool(MESSAGES, TOOL)

    assert args == {"length_mm": 77}
    assert seen[0]["tool_choice"] == {"type": "function", "function": {"name": "submit_designs"}}
    assert seen[0]["max_tokens"] == 2000 and seen[0]["messages"] == MESSAGES
    assert meta.rate_limit_headers == {"x-ratelimit-remaining-tokens": "199000"}
    assert SECRET not in repr(client._transport)


def test_openai_transport_errors_never_leak_the_key() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        echoed = request.headers["authorization"]
        return httpx2.Response(401, json={"error": {"message": f"invalid key: {echoed}"}})

    client, _ = make_client(FakeTransport())
    client._transport = _sdk_transport(handler)
    with pytest.raises(LLMAuthError) as info:
        client.chat(MESSAGES)
    assert SECRET not in str(info.value)
    assert "[REDACTED]" in str(info.value)


def test_openai_transport_does_not_forward_ambient_openai_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-elsewhere")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-elsewhere")
    seen: list[httpx2.Headers] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers)
        return httpx2.Response(200, json=completion(content="ok"))

    _sdk_transport(handler).chat_completions({"model": "m", "messages": MESSAGES}, timeout_s=1.0)
    assert "openai-organization" not in seen[0] and "openai-project" not in seen[0]


def test_openai_transport_maps_connection_failures() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    transport = _sdk_transport(handler)
    with pytest.raises(TransportError) as info:
        transport.chat_completions({"model": "m", "messages": MESSAGES}, timeout_s=1.0)
    assert SECRET not in str(info.value)


def test_redact_removes_secret_and_bearer_tokens() -> None:
    text = f"key={SECRET} header='Authorization: Bearer abc.def-123' tail"
    cleaned = redact(text, SECRET)
    assert SECRET not in cleaned and "abc.def-123" not in cleaned
    assert cleaned.endswith("tail")
