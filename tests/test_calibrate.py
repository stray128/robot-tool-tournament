"""Gate G3 calibration report: statistics and checks (synthetic rows; one tiny real run)."""

from __future__ import annotations

import pytest

from rtt.analysis.calibrate import (
    CalibrationReport,
    DesignRow,
    G3Targets,
    calibration_designs,
    run_calibration,
)
from rtt.design.schema import load_entrants
from rtt.sim.conditions import training_suite
from rtt.sim.task import default_task

FAMILIES = ("box", "cylinder", "rod")


def row(
    name: str, score: float, origin: str = "sobol", fam: tuple[float, float, float] | None = None
) -> DesignRow:
    per_family = dict(zip(FAMILIES, fam or (score, score, score), strict=True))
    return DesignRow(
        name=name,
        design_id=name,
        origin=origin,  # type: ignore[arg-type]
        score=score,
        best_k=0,
        hold_frac=score,
        mass_g=10.0,
        success_by_k={0: score},
        success_by_family=per_family,
        failure_modes={},
        flags={},
        params={},
    )


def report(entrants: list[DesignRow], sobol: list[DesignRow]) -> CalibrationReport:
    return CalibrationReport(
        task_name="t",
        task_digest="d",
        suite_name="s",
        suite_digest="sd",
        families=FAMILIES,
        n_conditions=12,
        episodes_per_design=36,
        sobol_seed=1,
        executor="cpu",
        workers=1,
        wall_s=1.0,
        episodes=0,
        entrants=entrants,
        sobol=sobol,
    )


def test_a_saturated_task_fails_every_check() -> None:
    rep = report([row("a", 1.0, "entrant"), row("b", 1.0, "entrant")], [row(f"s{i}", 1.0) for i in range(8)])
    checks = {c.name: c.passed for c in rep.checks()}
    assert checks == {
        "entrants_in_range": False,
        "entrant_family_gap": False,
        "sobol_sd": False,
        "headroom": False,
        "kill_trigger_4": False,
    }
    assert not rep.passed
    assert "G3 FAILED" in rep.table()


def test_a_calibrated_task_passes() -> None:
    entrants = [row("a", 0.5, "entrant", (0.75, 0.5, 0.25)), row("b", 0.6, "entrant", (0.5, 0.5, 0.75))]
    sobol = [row(f"s{i}", v) for i, v in enumerate([0.1, 0.2, 0.3, 0.9, 0.9, 0.95, 1.0, 0.7, 0.8, 0.4])]
    rep = report(entrants, sobol)
    stats = rep.sobol_stats()
    assert stats["n_beat_best_entrant"] == 6
    assert stats["sd"] == pytest.approx(0.3410, abs=1e-3)
    assert rep.family_gaps()["box"] == pytest.approx(0.25)
    assert rep.passed, rep.table()
    doc = rep.to_dict()
    assert doc["passed"] is True and len(doc["sobol"]) == 10


def test_boundaries_are_inclusive_and_headroom_is_strict() -> None:
    entrants = [row("a", 0.3, "entrant", (0.3, 0.3, 0.3)), row("b", 0.7, "entrant", (0.55, 0.7, 0.7))]
    sobol = [row(f"s{i}", v) for i, v in enumerate([0.7, 0.7, 0.71, 0.0, 1.0, 1.0, 1.0, 0.2])]
    rep = report(entrants, sobol)
    checks = {c.name: c.passed for c in rep.checks()}
    assert checks["entrants_in_range"] and checks["entrant_family_gap"]
    assert rep.sobol_stats()["n_beat_best_entrant"] == 4  # ties with the best entrant do not count
    assert not checks["headroom"]
    assert G3Targets().headroom_min == 5


def test_calibration_sample_is_reproducible_and_valid() -> None:
    task = default_task()
    suite = training_suite()
    entrants = load_entrants("configs/entrants.yaml")
    a = calibration_designs(entrants, suite, task, 6, 123)
    b = calibration_designs(entrants, suite, task, 6, 123)
    assert [d.design_id() for _, _, d in a] == [d.design_id() for _, _, d in b]
    assert [o for _, o, _ in a] == ["entrant"] * len(entrants) + ["sobol"] * 6
    assert a[0][0] == entrants[0].name


def test_run_calibration_end_to_end_on_one_condition() -> None:
    suite = training_suite()
    tiny = type(suite)(suite.name, suite.conditions[:1])
    entrants = load_entrants("configs/entrants.yaml")[:1]
    rep = run_calibration(suite=tiny, entrants=entrants, n_sobol=2, workers=1)
    assert rep.episodes == 3 * 3  # (1 entrant + 2 sobol) x 1 condition x 3 controller settings
    assert len(rep.entrants) == 1 and len(rep.sobol) == 2
    assert 0.0 <= rep.entrants[0].score <= 1.0
