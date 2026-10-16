"""Gate GV4's comparison (rtt.sim.gv4): what passes and what fails, on synthetic reports."""

from __future__ import annotations

import copy
import json

import pytest

from rtt.sim.gv4 import GV4_MIN_EPISODES, SCHEMA, compare, main, record_digest


def _record(i: int, success: bool = True) -> dict:
    r = {
        "design_id": f"d{i % 4}",
        "condition_id": f"train_v2/part/c{i % 3}",
        "seed": i,
        "success": success,
        "lift_success": True,
        "failure_mode": None if success else "not_registered",
        "reg_dx_mm": 0.1 * i,
        "seat_short_mm": None,
        "geometry_hash": f"g{i % 4}",
    }
    r["digest"] = record_digest(r)
    return r


def _report(n: int = GV4_MIN_EPISODES + 20) -> dict:
    return {
        "schema": SCHEMA,
        "task_digest": "t",
        "suite_digest": "s",
        "parts": {"block": "p1", "pipe_tee": "p2"},
        "stock_hulls_sha256": "h",
        "designs": [
            {"label": f"d{i}", "design_id": f"d{i}", "geometry_id": f"G{i}", "score": 0.5} for i in range(4)
        ],
        "episodes": n,
        "records": [_record(i) for i in range(n)],
        "records_sha256": "r",
    }


def test_identical_reports_pass_bit_for_bit() -> None:
    a = _report()
    result = compare(a, copy.deepcopy(a))
    assert result["passed"] and result["outcome_agreement"] == 1.0
    assert result["bit_identical_records"] == a["episodes"] and result["max_abs_diff"] == {}


def test_one_outcome_in_a_hundred_and_twenty_still_passes_two_do_not() -> None:
    a = _report()
    b = copy.deepcopy(a)
    b["records"][3] = _record(3, success=False)
    result = compare(a, b)
    assert result["passed"] and result["outcome_agree"] == a["episodes"] - 1
    assert result["outcome_differences"][0]["seed"] == 3
    b["records"][5] = _record(5, success=False)
    assert not compare(a, b)["checks"]["outcome_agreement"]


def test_float_drift_without_an_outcome_change_is_reported_not_failed() -> None:
    a = _report()
    b = copy.deepcopy(a)
    b["records"][7]["reg_dx_mm"] += 1e-9
    b["records"][7]["digest"] = record_digest(b["records"][7])
    result = compare(a, b)
    assert result["passed"] and result["bit_identical_records"] == a["episodes"] - 1
    assert result["max_abs_diff"]["reg_dx_mm"] == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("field", "check"),
    [
        ("parts", "part_hashes_equal"),
        ("stock_hulls_sha256", "stock_hulls_equal"),
        ("designs", "design_geometry_ids_equal"),
        ("geometry_hash", "jaw_pair_hashes_equal"),
        ("task_digest", "same_task_and_suite"),
    ],
)
def test_any_geometry_or_hull_hash_difference_fails(field: str, check: str) -> None:
    a = _report()
    b = copy.deepcopy(a)
    if field == "parts":
        b["parts"]["block"] = "other"
    elif field == "designs":
        b["designs"][0]["geometry_id"] = "other"
    elif field == "geometry_hash":
        b["records"][0]["geometry_hash"] = "other"
    else:
        b[field] = "other"
    result = compare(a, b)
    assert not result["checks"][check] and not result["passed"]


def test_too_few_or_missing_episodes_fail() -> None:
    small = _report(GV4_MIN_EPISODES - 1)
    assert not compare(small, copy.deepcopy(small))["checks"]["enough_episodes"]
    a = _report()
    b = copy.deepcopy(a)
    b["records"].pop()
    result = compare(a, b)
    assert not result["checks"]["same_episodes"] and result["only_in_a"] == 1


def test_digest_ignores_wall_time_only() -> None:
    r = _record(1)
    assert record_digest({**r, "wall_time_s": 9.0}) == record_digest(r)
    assert record_digest({**r, "seed": 2}) != record_digest(r)


def test_cli_compare_exit_status(tmp_path) -> None:
    a = _report()
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(a))
    assert (
        main(
            ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(tmp_path / "c.json")]
        )
        == 0
    )
    assert json.loads((tmp_path / "c.json").read_text())["passed"]
    a["schema"] = "other"
    (tmp_path / "b.json").write_text(json.dumps(a))
    with pytest.raises(ValueError, match="not a GV4 report"):
        main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
