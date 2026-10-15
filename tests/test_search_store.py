"""Store: manifests, LLM usage aggregation, JSONL logs and Parquet tables."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtt.store.manifest import (
    JobContext,
    LLMUsage,
    file_sha256,
    read_manifest,
    write_json_atomic,
    write_manifest,
)
from rtt.store.runs import append_jsonl, read_jsonl, read_table, write_jsonl, write_parquet


def test_job_context_reads_only_rtt_variables() -> None:
    env = {"RTT_JOB_ID": "job-1", "RTT_PLATFORM": "gpu-x", "RTT_REGION": "", "OTHER": "y"}
    job = JobContext.from_env(env)
    assert job.job_id == "job-1" and job.platform == "gpu-x" and job.region is None
    assert job.is_cloud_job and not JobContext.from_env({}).is_cloud_job


def test_llm_usage_counts_live_and_cached_calls() -> None:
    records = [
        {"request_hash": "a", "cached": False, "ok": True, "prompt_tokens": 100, "completion_tokens": 50,
         "reasoning_tokens": 10, "cost_usd": 0.01, "latency_s": 2.0},
        {"request_hash": "b", "cached": True, "ok": True, "prompt_tokens": 80, "completion_tokens": 20,
         "cost_usd": 0.005, "latency_s": 4.0},
        {"request_hash": None, "cached": False, "ok": False},
    ]  # fmt: skip
    usage = LLMUsage.from_records(records)
    assert (usage.calls, usage.live_calls, usage.cached_calls, usage.errors) == (3, 1, 1, 1)
    assert usage.prompt_tokens == 180 and usage.completion_tokens == 70 and usage.reasoning_tokens == 10
    assert usage.cost_usd == pytest.approx(0.015)
    assert usage.latency_s_p50 == pytest.approx(3.0) and usage.latency_s_max == 4.0
    total = LLMUsage.combine([usage, usage])
    assert total.calls == 6 and total.cost_usd == pytest.approx(0.03) and total.latency_s_p50 is None


def test_manifest_round_trip_and_version_check(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "m" / "manifest.json", {"arm": "random"})
    assert read_manifest(path) == {"manifest_version": 1, "arm": "random"}
    write_json_atomic(path, {"manifest_version": 99})
    with pytest.raises(ValueError):
        read_manifest(path)
    assert not list((tmp_path / "m").glob(".tmp-*"))


def test_file_sha256(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"abc")
    assert file_sha256(tmp_path / "f") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(tmp_path / "missing") is None


def test_jsonl_and_parquet(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    append_jsonl(path, [{"b": 1, "a": [1, 2]}])
    append_jsonl(path, [{"c": None}])
    assert read_jsonl(path) == [{"a": [1, 2], "b": 1}, {"c": None}]
    assert path.read_text().splitlines()[0] == json.dumps({"a": [1, 2], "b": 1}, sort_keys=True)
    write_jsonl(path, [{"z": 0}])
    assert read_jsonl(path) == [{"z": 0}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
    rows = [{"a": 1, "b": "x", "c": None}, {"a": 2, "b": None, "c": 0.5}]
    assert read_table(write_parquet(tmp_path / "t.parquet", rows)) == rows
