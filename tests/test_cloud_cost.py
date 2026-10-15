"""rtt.cloud.cost: price table, estimates and the spend ledger."""

from __future__ import annotations

from pathlib import Path

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
    with pytest.raises(KeyError, match="cpu-e2"):
        hourly_usd("cpu-e2")


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
