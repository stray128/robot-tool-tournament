"""End-to-end workflows: slice -> replays -> export-web, held-out scoring, replay checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtt.design.schema import load_entrants
from rtt.llm.client import TokenBucket
from rtt.pipeline.export_web import export_web
from rtt.pipeline.heldout import collect_frozen, run_heldout
from rtt.pipeline.replays import ReplayMismatchError, ReplayRequest, record_replays
from rtt.pipeline.slice import SliceConfig, read_slice, run_slice
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.search.runner import run_search
from rtt.search.sim_adapter import SimEvaluator
from rtt.sim.conditions import Suite, seed_commitment, training_suite
from rtt.sim.exec_cpu import CpuExecutor

ENTRANTS = load_entrants("configs/entrants.yaml")


def tiny_suite(n: int = 2) -> Suite:
    suite = training_suite()
    return Suite(suite.name, suite.conditions[:n])


@pytest.fixture(scope="module")
def slice_run(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("slice") / "slice-test"
    client = make_proposer_client(out / "llm_cache", transport=ScriptedProposer(3), limiter=TokenBucket(1e12))
    with CpuExecutor(4) as pool:
        sim = SimEvaluator(tiny_suite(), executor=pool)
        outcome = run_slice(out, SliceConfig(seed=5), llm_client=client, evaluator=sim, executor=pool)
    return outcome


def test_slice_writes_designs_lanes_and_replays(slice_run) -> None:
    s = read_slice(slice_run.out_dir)
    assert len(s["designs"]) == 12  # 8 in round 0, 4 proposed
    assert [d["round"] for d in s["designs"]].count(1) == 4
    assert {d["origin"] for d in s["designs"] if d["round"] == 1} <= {"nemotron", "fallback_random"}
    assert len(s["lanes"]) == 5 and s["lanes"][0]["stem"] == "baseline-stock"
    by_id = {d["design_id"]: d for d in s["designs"]}
    for lane in s["lanes"]:
        assert (slice_run.out_dir / f"{lane['replay']}.bin").is_file()
        assert lane["controller_setting"] == by_id[lane["design_id"]]["score"]["best_k"]
    assert s["llm"]["calls"] >= 1 and s["config"]["spend_cap_usd"] == 0.20
    assert (slice_run.out_dir / "search" / "manifest.json").is_file()


def test_export_web_publishes_the_slice_once(slice_run, tmp_path: Path) -> None:
    data = tmp_path / "data"
    first = export_web([slice_run.out_dir], data)
    assert len(first.bouts) == 1 and first.replays == 5
    again = export_web([slice_run.out_dir], data)  # re-export replaces its own entries
    assert again.bouts == first.bouts
    bouts = json.loads((data / "bouts.json").read_text())["bouts"]
    designs = json.loads((data / "designs.json").read_text())["designs"]
    jobs = json.loads((data / "jobs.json").read_text())["jobs"]
    assert len(bouts) == 1 and len(jobs) == 1 and jobs[0]["kind"] == "local"
    bout = bouts[0]
    assert bout["kind"] == "training" and bout["compute"]["kind"] == "local" and len(bout["lanes"]) == 5
    lane_ids = {lane["design_id"] for lane in bout["lanes"]}
    assert {d["id"] for d in designs} == lane_ids
    for lane in bout["lanes"]:
        header = json.loads((data / f"{lane['replay']}.json").read_text())
        assert header["n_frames"] == lane["n_frames"]
        assert lane["object"]["kind"] in ("box", "cylinder", "capsule")
    for d in designs:
        for rel in d["files"].values():
            assert (data / rel).is_file()
        if d["provenance"]["kind"] == "llm":
            assert d["provenance"]["provider"] == "nebius-tokenfactory"


def test_export_web_keeps_other_sources_and_skips_unknown_dirs(slice_run, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    other = {"id": "x", "source": "demo-m1", "kind": "gate"}
    (data / "jobs.json").write_text(json.dumps({"schema": "rtt-web-jobs", "version": 1, "jobs": [other]}))
    result = export_web([slice_run.out_dir, tmp_path / "nothing"], data)
    assert result.skipped == [str(tmp_path / "nothing")]
    jobs = json.loads((data / "jobs.json").read_text())["jobs"]
    assert jobs[0] == other and len(jobs) == 2


def test_record_replays_refuses_an_outcome_that_disagrees(tmp_path: Path) -> None:
    cond = tiny_suite(1).conditions[0]
    stock = next(d for d in ENTRANTS if d.name == "stock")
    req = ReplayRequest("a", stock, cond, 0, expect_success=True)
    (got,) = record_replays([req], tmp_path, workers=1)
    wrong = ReplayRequest(
        "b", stock, cond, 0, expect_success=not got.success, expect_failure_mode="missed_grasp"
    )
    with pytest.raises(ReplayMismatchError):
        record_replays([wrong], tmp_path / "b", workers=1)
    assert not (tmp_path / "b").exists()  # nothing is written when an outcome disagrees


def test_heldout_scores_frozen_designs_and_never_writes_the_seed(tmp_path: Path) -> None:
    seed = "ab" * 32
    seed_file = tmp_path / "heldout_seed.txt"
    seed_file.write_text(seed + "\n")
    search = run_search("random", 2, 8, [1, 2], FakeEvaluator(0), tmp_path / "search")
    frozen = collect_frozen([search.out_dir])
    assert len(frozen) == 2
    commitment = seed_commitment(seed)
    report = run_heldout(
        seed_file,
        tmp_path / "heldout",
        frozen,
        commitment=commitment,
        n_conditions=2,
        seeds_per_condition=1,
        workers=1,
    )
    assert report["commitment_verified"] and report["episodes"] == 4
    assert set(report["arms"]) == {"random"} and len(report["arms"]["random"]["runs"]) == 2
    text = (tmp_path / "heldout" / "heldout.json").read_text()
    assert seed not in text and (tmp_path / "heldout" / "episodes.parquet").is_file()
    with pytest.raises(ValueError, match="commitment"):
        run_heldout(seed_file, tmp_path / "bad", frozen, commitment="0" * 64, n_conditions=1)
