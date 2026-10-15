"""LLM cache replay: a recorded LLM arm re-runs from its response cache with zero API calls.

This is what ``make replay-llm`` relies on: with the cache in ``replay`` mode and no
usable transport, the search must reproduce the recorded run exactly (same designs,
same scores, same rejections, same model-call identities, same result files), and any
request that differs from the recording must fail loudly instead of calling the API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rtt.llm.cache import CacheMiss, ResponseCache
from rtt.llm.client import TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import ProposerConfig, make_proposer_client
from rtt.search.runner import SearchConfig, run_search
from rtt.store.manifest import read_manifest

SEEDS = [404, 505]


class NoNetwork:
    """Any model call fails the test."""

    calls = 0

    def list_models(self, timeout_s: float) -> TransportResponse:
        raise AssertionError("no API call is allowed during replay")

    def chat_completions(self, payload, timeout_s):  # type: ignore[no-untyped-def]
        NoNetwork.calls += 1
        raise AssertionError("no API call is allowed during replay")


def client(cache_dir: Path, mode: str, transport) -> object:  # type: ignore[no-untyped-def]
    return make_proposer_client(cache_dir, cache_mode=mode, transport=transport, limiter=TokenBucket(1e12))


@pytest.mark.parametrize("arm", ["nemotron", "zeroshot"])
def test_replay_reproduces_the_run_with_zero_api_calls(tmp_path: Path, arm: str) -> None:
    cache = tmp_path / "llm_cache"
    recorder = ScriptedProposer(11, invalid_rate=0.3, similar_rate=0.2, garbage_rate=0.15)
    live = run_search(
        arm, 2, 40, SEEDS, FakeEvaluator(0), tmp_path / "live", client(cache, "readwrite", recorder)
    )
    assert recorder.calls > 0
    assert len(ResponseCache(cache)) == recorder.calls

    NoNetwork.calls = 0
    replay_client = client(cache, "replay", NoNetwork())
    replayed = run_search(arm, 2, 40, SEEDS, FakeEvaluator(0), tmp_path / "replay", replay_client)
    assert NoNetwork.calls == 0

    for a, b in zip(live.runs, replayed.runs, strict=True):
        assert a.fingerprint == b.fingerprint
        assert a.frozen.content_sha256 == b.frozen.content_sha256
        assert [c.request_hash for c in a.ctx.llm_calls] == [c.request_hash for c in b.ctx.llm_calls]
        assert all(c.cached for c in b.ctx.llm_calls)
        assert not any(c.cached for c in a.ctx.llm_calls)
        for name in ("history.jsonl", "rejections.jsonl"):
            run_dir = Path("runs") / a.ctx.run_id / name
            assert (tmp_path / "live" / run_dir).read_bytes() == (tmp_path / "replay" / run_dir).read_bytes()
    for name in ("designs.jsonl", "designs.parquet"):
        assert (tmp_path / "live/results" / name).read_bytes() == (
            tmp_path / "replay/results" / name
        ).read_bytes()

    live_m, replay_m = (
        read_manifest(tmp_path / "live/manifest.json"),
        read_manifest(tmp_path / "replay/manifest.json"),
    )
    assert replay_m["llm"]["live_calls"] == 0 and replay_m["llm"]["cached_calls"] == live_m["llm"]["calls"]
    # Token counts and dollars are those of the recorded calls, so the replay reports the same totals.
    for key in ("prompt_tokens", "completion_tokens", "cost_usd"):
        assert replay_m["llm"][key] == live_m["llm"][key]
    assert replay_m["model"]["cache_mode"] == "replay"


def test_a_changed_prompt_misses_the_cache_instead_of_calling_the_api(tmp_path: Path) -> None:
    cache = tmp_path / "llm_cache"
    run_search(
        "nemotron",
        1,
        24,
        SEEDS[:1],
        FakeEvaluator(0),
        tmp_path / "live",
        client(cache, "readwrite", ScriptedProposer(3)),
    )
    changed = SearchConfig(proposer=ProposerConfig(elites_shown=7))
    NoNetwork.calls = 0
    with pytest.raises(CacheMiss):
        run_search(
            "nemotron", 1, 24, SEEDS[:1], FakeEvaluator(0), tmp_path / "replay",
            client(cache, "replay", NoNetwork()), config=changed,
        )  # fmt: skip
    assert NoNetwork.calls == 0
    assert read_manifest(tmp_path / "replay/manifest.json")["status"] == "failed"


class Flaky(ScriptedProposer):
    """Fails one call with a non-retryable HTTP error; answers every other call."""

    def chat_completions(self, payload, timeout_s):  # type: ignore[no-untyped-def]
        if self.calls == 2:
            self.calls += 1
            return TransportResponse(400, {}, {"error": {"message": "bad request"}})
        return super().chat_completions(payload, timeout_s)


@pytest.mark.parametrize("failure", ["http", "spend_cap"])
def test_runs_with_failed_calls_replay_exactly(tmp_path: Path, failure: str) -> None:
    cache = tmp_path / "llm_cache"
    if failure == "http":
        live_client = client(cache, "readwrite", Flaky(13))
    else:  # the cap admits a few calls, then refuses the rest of the run
        live_client = make_proposer_client(
            cache, transport=ScriptedProposer(13), limiter=TokenBucket(1e12), budget_usd=0.015
        )
    live = run_search("nemotron", 1, 40, SEEDS[:1], FakeEvaluator(0), tmp_path / "live", live_client)
    failed = [
        c for c in live.runs[0].ctx.llm_calls if not c.ok and c.error and c.error.startswith("call_failed")
    ]
    assert failed, "the live run should contain a failed call"
    assert (live.manifest["per_run"][0]["llm_disabled"] == "spend_cap") == (failure == "spend_cap")

    NoNetwork.calls = 0
    replayed = run_search(
        "nemotron",
        1,
        40,
        SEEDS[:1],
        FakeEvaluator(0),
        tmp_path / "replay",
        client(cache, "replay", NoNetwork()),
    )
    assert NoNetwork.calls == 0
    assert replayed.runs[0].fingerprint == live.runs[0].fingerprint
    assert replayed.manifest["per_run"][0]["llm_disabled"] == live.manifest["per_run"][0]["llm_disabled"]
    assert all(c.cached for c in replayed.runs[0].ctx.llm_calls)
