"""Lockstep runner: outputs, manifest, frozen top-1, run independence and argument checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtt.llm.client import TokenBucket
from rtt.llm.providers import NEMOTRON_SUPER
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.search.runner import load_seeds, run_search
from rtt.search.select import freeze_top1, read_frozen, verify_frozen, write_frozen
from rtt.store.manifest import read_manifest
from rtt.store.runs import read_jsonl, read_table


def scripted_client(cache: Path | None = None, seed: int = 1):
    return make_proposer_client(cache, transport=ScriptedProposer(seed), limiter=TokenBucket(1e12))


def test_nemotron_run_writes_every_output(tmp_path: Path) -> None:
    out = run_search("nemotron", 2, 40, [21, 22, 23], FakeEvaluator(0), tmp_path, scripted_client())
    manifest = read_manifest(tmp_path / "manifest.json")
    assert manifest == json.loads(json.dumps(out.manifest | {"manifest_version": 1}))
    assert manifest["status"] == "complete" and manifest["arm"] == "nemotron"
    assert manifest["seeds"] == [21, 22] and manifest["round_sizes"] == [8, 8, 8, 8, 8]
    assert manifest["model"]["model_id"] == NEMOTRON_SUPER
    assert manifest["model"]["temperature"] == 1.0 and manifest["model"]["top_p"] == 0.95
    assert manifest["provider"] == "nebius-tokenfactory" and manifest["executor"] == "fake"
    assert set(manifest["prompt_hashes"]) >= {"template:proposer_user.md", "tool:proposer", "parameter_table"}
    for key in ("git_sha", "uv_lock_sha256", "package_version"):
        assert key in manifest["code"]
    assert len(manifest["evaluator"]["task_hash"]) == 64 and len(manifest["evaluator"]["suite_hash"]) == 64
    assert set(manifest["job"]) == {"job_id", "platform", "preset", "region", "image_digest", "launcher"}
    assert manifest["sim"]["designs_charged"] == 80
    assert manifest["llm"]["calls"] == sum(r["llm"]["calls"] for r in manifest["per_run"])
    for run in manifest["per_run"]:
        assert run["budget"]["charged"] == [8, 8, 8, 8, 8]
        assert run["llm"]["calls"] <= 16 and run["llm"]["prompt_tokens"] > 0 and run["llm"]["cost_usd"] > 0
        assert run["rejections"] == sum(run["rejections_by_reason"].values())
        assert len(run["best_so_far"]) == 40 and run["best_so_far"] == sorted(run["best_so_far"])
        run_dir = tmp_path / "runs" / run["run_id"]
        history = read_jsonl(run_dir / "history.jsonl")
        assert [h["index"] for h in history] == list(range(1, 41))
        assert len(read_jsonl(run_dir / "rejections.jsonl")) == run["rejections"]
        assert len(read_jsonl(run_dir / "llm_calls.jsonl")) == run["llm"]["calls"]
        frozen = read_frozen(tmp_path / "frozen" / f"{run['run_id']}.json")
        assert frozen.content_sha256 == run["frozen_sha256"]
        assert frozen.training_success == run["best"]["success"]
    designs = read_table(tmp_path / "results" / "designs.parquet")
    assert len(designs) == 80
    assert {d["origin"] for d in designs} >= {"entrant", "sobol", "nemotron"}
    assert all(d["n_episodes"] == 36 for d in designs)
    assert len(read_table(tmp_path / "results" / "episodes.parquet")) == 80 * 36


def test_runs_share_nothing(tmp_path: Path) -> None:
    together = run_search("nemotron", 2, 40, [31, 32], FakeEvaluator(0), tmp_path / "a", scripted_client())
    alone = run_search("nemotron", 1, 40, [32], FakeEvaluator(0), tmp_path / "b", scripted_client())
    # Run 32 behaves identically whether or not another run shares its job.
    assert together.runs[1].ctx.history.fingerprint() == alone.runs[0].ctx.history.fingerprint()
    assert [c.request_hash for c in together.runs[1].ctx.llm_calls] == [
        c.request_hash for c in alone.runs[0].ctx.llm_calls
    ]


def test_identical_designs_are_simulated_once(tmp_path: Path) -> None:
    evaluator = FakeEvaluator(0)
    out = run_search("random", 3, 40, [1, 2, 3], evaluator, tmp_path)
    n_entrants = len(out.manifest["entrants"])
    assert out.manifest["sim"]["designs_charged"] == 120
    assert evaluator.designs_evaluated == out.manifest["sim"]["designs_simulated"] == 120 - 2 * n_entrants


def test_argument_checks(tmp_path: Path) -> None:
    ev = FakeEvaluator(0)
    with pytest.raises(ValueError, match="unknown arm"):
        run_search("tpe", 1, 40, [1], ev, tmp_path / "x")
    with pytest.raises(ValueError, match="seeds"):
        run_search("random", 2, 40, [1], ev, tmp_path / "x")
    with pytest.raises(ValueError, match="distinct"):
        run_search("random", 2, 40, [1, 1], ev, tmp_path / "x")
    with pytest.raises(ValueError, match="llm_client"):
        run_search("nemotron", 1, 40, [1], ev, tmp_path / "x")
    with pytest.raises(ValueError, match="budget"):
        run_search("random", 1, 41, [1], ev, tmp_path / "y")
    run_search("random", 1, 16, [1], ev, tmp_path / "z")
    with pytest.raises(FileExistsError):
        run_search("random", 1, 16, [1], ev, tmp_path / "z")


def test_failed_search_leaves_a_failed_manifest(tmp_path: Path) -> None:
    class Broken(FakeEvaluator):
        def evaluate(self, designs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulator crashed")

    with pytest.raises(RuntimeError, match="simulator crashed"):
        run_search("random", 1, 16, [1], Broken(0), tmp_path)
    manifest = read_manifest(tmp_path / "manifest.json")
    assert manifest["status"] == "failed" and "simulator crashed" in manifest["error"]


def test_freeze_picks_the_best_and_detects_tampering(tmp_path: Path) -> None:
    out = run_search("random", 1, 16, [5], FakeEvaluator(0), tmp_path / "run")
    ctx = out.runs[0].ctx
    frozen = freeze_top1("random", 0, 5, ctx.history, ctx.info)
    best = max(ctx.history, key=lambda e: (e.score.rank_key(), -e.index))
    assert frozen.label == best.label and frozen.controller_setting == best.score.best_k
    assert frozen.grasp_height_frac == ctx.info.controller_settings[best.score.best_k]
    assert verify_frozen(frozen)
    path = write_frozen(frozen, tmp_path / "frozen")
    with pytest.raises(FileExistsError):
        write_frozen(frozen, tmp_path / "frozen")
    data = json.loads(path.read_text())
    data["controller_setting"] = (data["controller_setting"] + 1) % 3
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="sha256"):
        read_frozen(path)


def test_load_seeds(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("seeds: [3, 1, 2]\n")
    (tmp_path / "b.yaml").write_text("- 9\n- 8\n")
    (tmp_path / "c.yaml").write_text("seeds: [1, x]\n")
    assert load_seeds(tmp_path / "a.yaml") == [3, 1, 2]
    assert load_seeds(tmp_path / "b.yaml") == [9, 8]
    with pytest.raises(TypeError):
        load_seeds(tmp_path / "c.yaml")
