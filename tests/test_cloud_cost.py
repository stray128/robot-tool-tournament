"""rtt.cloud.cost: price table, estimates and the spend ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.cost import (
    PRICES,
    JobSpend,
    SpendCapError,
    SpendLedger,
    estimate_usd,
    gpu_count,
    hourly_usd,
    read_spend,
    summarize,
)


def _spend(name: str, est: float, wall: float = 300.0, run: float | None = 60.0) -> JobSpend:
    return JobSpend(
        job_id=f"aijob-{name}",
        name=name,
        platform="gpu-h100-sxm",
        preset="1gpu-16vcpu-200gb",
        region="eu-north1",
        state="COMPLETED",
        preemptible=False,
        provisioning_s=120.0,
        run_s=run,
        wall_s=wall,
        usd_per_hour=4.5,
        est_usd=est,
        est_usd_run=est / 5,
    )


def test_fallback_platforms_are_priced() -> None:
    for platform in ("gpu-h100-sxm", "gpu-h200-sxm", "gpu-l40s-a"):
        assert PRICES[platform].gpu_hour_usd > 0
    assert PRICES["gpu-h100-sxm"].tag == "sourced"
    assert PRICES["gpu-l40s-a"].tag == "assumed"  # a "from" price


def test_gpu_count_and_hourly_price_scale_with_preset() -> None:
    assert gpu_count("1gpu-16vcpu-200gb") == 1
    assert gpu_count("8gpu-128vcpu-1600gb") == 8
    assert gpu_count(None) == 1
    assert hourly_usd("gpu-h100-sxm", "8gpu-128vcpu-1600gb") == pytest.approx(8 * 4.50)
    with pytest.raises(KeyError, match="gpu-unknown"):
        hourly_usd("gpu-unknown")
    with pytest.raises(ValueError, match="vcpu"):
        hourly_usd("cpu-e2")  # a CPU platform is priced by its preset's vCPUs and RAM


def test_estimate_usd() -> None:
    assert estimate_usd("gpu-h100-sxm", "1gpu-16vcpu-200gb", 3600) == pytest.approx(4.50)
    assert estimate_usd("gpu-h100-sxm", "1gpu-16vcpu-200gb", 300) == pytest.approx(0.375)
    assert estimate_usd("gpu-h100-sxm", None, -5) == 0.0


def test_ledger_appends_reads_and_totals(tmp_path: Path) -> None:
    path = tmp_path / "cloud" / "spend.jsonl"
    ledger = SpendLedger(path, cap_usd=3.0)
    assert ledger.total_usd() == 0.0 and ledger.remaining_usd() == 3.0
    ledger.append(_spend("a", 0.40))
    ledger.append(_spend("b", 0.60, run=None))
    assert [e.name for e in read_spend(path)] == ["a", "b"]
    assert ledger.total_usd() == pytest.approx(1.0)
    assert ledger.remaining_usd() == pytest.approx(2.0)
    totals = summarize(ledger.entries())
    assert totals["jobs"] == 2 and totals["run_s"] == 60.0 and totals["wall_s"] == 600.0


def test_ledger_cap_refuses_overrun(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=1.0)
    ledger.append(_spend("a", 0.75))
    ledger.check(0.25)
    with pytest.raises(SpendCapError, match=r"\$1.00 cap"):
        ledger.check(0.30)
    SpendLedger(tmp_path / "other.jsonl").check(1e9)  # uncapped
    with pytest.raises(ValueError):
        SpendLedger(tmp_path / "x.jsonl", cap_usd=-1)


def test_entry_round_trip_ignores_unknown_fields() -> None:
    entry = _spend("a", 0.1)
    data = entry.to_json() | {"future_field": 1}
    assert JobSpend.from_json(data) == entry


# --------------------------------------------------------------------------- in-flight reservations


def test_reservations_of_another_process_count_against_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "cloud" / "spend.jsonl"
    mine, theirs = SpendLedger(path, cap_usd=3.0), SpendLedger(path, cap_usd=3.0)  # two `rtt job`s
    mine.append(_spend("done", 1.0))
    held = theirs.reserve(1.5, "rtt-search-a")
    assert mine.reserved_usd() == pytest.approx(1.5) and mine.remaining_usd() == pytest.approx(0.5)
    assert mine.remaining_usd(exclude=held) == pytest.approx(2.0)
    with pytest.raises(SpendCapError, match=r"\$1.50 held by 1 job\(s\) in flight"):
        mine.reserve(0.6, "rtt-m1-b")
    with pytest.raises(SpendCapError, match="held by jobs in flight"):
        mine.check(0.6)
    ok = mine.reserve(0.5, "rtt-m1-b")
    assert sorted(r.label for r in theirs.inflight()) == ["rtt-m1-b", "rtt-search-a"]
    # a running job raises its hold to its live spend; everyone else sees it at once
    theirs.hold(held, 1.9)
    assert mine.remaining_usd(exclude=ok) == pytest.approx(0.1)
    # recording the spend and dropping the hold are one step
    theirs.append(_spend("search-a", 1.9), release=held)
    assert mine.total_usd() == pytest.approx(2.9) and mine.reserved_usd() == pytest.approx(0.5)
    mine.release(ok)
    mine.release(ok)  # idempotent
    assert mine.inflight() == [] and not mine.inflight_path.exists()
    uncapped = SpendLedger(tmp_path / "u.jsonl")
    tok = uncapped.reserve(1e6, "big")  # no cap: recorded, never refused
    assert uncapped.remaining_usd() is None and uncapped.reserved_usd() == 1e6
    uncapped.release(tok)


def test_reservations_of_dead_processes_are_dropped(tmp_path: Path) -> None:
    import json
    import socket

    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=1.0)
    row = {"usd": 0.9, "label": "crashed", "ts": "2026-10-15T00:00:00Z"}
    ledger.inflight_path.write_text(
        json.dumps(
            {
                "dead": {**row, "token": "dead", "pid": 999_999_999, "host": socket.gethostname()},
                "remote": {**row, "token": "remote", "pid": 1, "host": "another-laptop", "usd": 0.25},
            }
        )
    )
    assert [r.token for r in ledger.inflight()] == ["remote"]  # a pid elsewhere cannot be checked
    assert ledger.remaining_usd() == pytest.approx(0.75)


def test_unreadable_inflight_file_fails_closed(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=1.0)
    ledger.inflight_path.write_text("{not json")
    with pytest.raises(SpendCapError, match="unreadable in-flight file"):
        ledger.reserve(0.1, "x")
    with pytest.raises(ValueError):
        ledger.reserve(-1.0, "x")


def _reserve_one(path: str, barrier: Any) -> bool:
    barrier.wait()
    try:
        SpendLedger(path, cap_usd=3.0).reserve(1.0, "parallel")
    except SpendCapError:
        return False
    return True


def test_reserve_is_atomic_across_processes(tmp_path: Path) -> None:
    import multiprocessing as mp
    import os

    ctx = mp.get_context("fork")
    path = str(tmp_path / "spend.jsonl")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(6)
        with ctx.Pool(6) as pool:
            granted = pool.starmap(_reserve_one, [(path, barrier)] * 6)
    assert sum(granted) == 3  # exactly the cap's worth, however the six raced
    # the pool's workers have exited, so their holds are stale and dropped
    assert os.getpid() not in {r.pid for r in SpendLedger(path).inflight()}
    assert SpendLedger(path).inflight() == []
