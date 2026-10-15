"""Robustness certificate: perturbation primitives, the test battery and its scoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.sim.certify import (
    Certificate,
    CertificateTest,
    _tolerance,
    certificate_task,
    certify,
    plan_tests,
    push_height,
)
from rtt.sim.controller import plan_grasp
from rtt.sim.episode import run_episode
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.perturb import Push, Shake, axis_vector
from rtt.sim.task import load_task

TASK = load_task()
STOCK_DESIGN = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}["stock"]
STOCK = generate_finger(STOCK_DESIGN)
CAN = ObjectSpec("cylinder", (0.02, 0.08))


@pytest.fixture(scope="module")
def cert() -> Certificate:
    return certify(STOCK_DESIGN, CAN, workers=4, task=TASK)


# --------------------------------------------------------------------------- primitives


def test_axis_vectors_follow_the_palm_yaw() -> None:
    assert np.allclose(axis_vector("x", 0.0), [1, 0, 0])
    assert np.allclose(axis_vector("y", math.pi / 2), [-1, 0, 0])
    assert np.allclose(axis_vector("z", 1.0), [0, 0, 1])
    with pytest.raises(ValueError):
        axis_vector("w", 0.0)  # type: ignore[arg-type]


def test_shake_is_a_windowed_sinusoid() -> None:
    shake = Shake("x", amplitude=0.01, frequency=5.0, cycles=5, ramp_cycles=0.5)
    assert shake.duration == pytest.approx(1.0)
    assert shake.peak_acceleration == pytest.approx((2 * math.pi * 5) ** 2 * 0.01)
    s = np.linspace(-0.1, 1.1, 2401)
    d = np.array([shake.displacement(v) for v in s])
    assert d[s <= 0].max() == 0.0 and d[s >= 1.0].max() == 0.0
    assert np.abs(d).max() <= 0.01 + 1e-12
    assert np.abs(d).max() > 0.0099  # reaches full amplitude between the ramps
    assert abs(shake.displacement(0.01)) < 0.1 * abs(shake.displacement(0.45))  # ramped in
    with pytest.raises(ValueError):
        Shake("x", 0.01, 5.0, 1, ramp_cycles=0.6)


def test_push_wrench_twists_about_the_centre_of_mass() -> None:
    push = Push("x", impulse=0.5, duration=0.02, height=0.05)
    assert push.force == pytest.approx(25.0)
    w = push.wrench(0.0)
    assert np.allclose(w[:3], [25.0, 0, 0])
    assert np.allclose(w[3:], np.cross([0, 0, 0.05], [25.0, 0, 0]))  # +y torque
    with pytest.raises(ValueError):
        Push("z", 0.1)  # type: ignore[arg-type]


def test_oversized_push_knocks_the_object_out() -> None:
    """With jitter off the nominal grasp holds; a 5 N s push during the hold dislodges it."""
    task = certificate_task(TASK)
    nominal = run_episode(STOCK, CAN, Placement(), 1, 0, record=False, task=task)
    assert nominal.success
    knock = run_episode(
        STOCK, CAN, Placement(), 1, 0, record=False, task=task, perturbation=Push("x", 5.0, 0.02)
    )
    assert not knock.success and knock.failure_mode == "unstable_hold"


def test_shake_moves_the_palm_by_its_amplitude() -> None:
    task = certificate_task(TASK)
    shake = Shake("y", 0.01, 5.0, 5, 0.5, delay=0.2)
    res = run_episode(STOCK, CAN, Placement(), 1, 0, record=True, task=task, perturbation=shake)
    assert res.success, res.summary()
    assert res.qpos is not None
    plan = plan_grasp(STOCK, CAN, Placement(), 1, task)
    t = np.arange(len(res.qpos)) / task.replay.fps
    window = (t > plan.t_hold + 0.2) & (t < plan.t_hold + 1.2)
    palm_y = res.qpos[window, 1]
    assert 0.006 < np.abs(palm_y).max() < 0.012  # servo tracking of the 10 mm command
    assert res.metrics["perturb_end"] == pytest.approx(plan.t_hold + 1.2)
    # success is only confirmed after the shake plus the settling hold
    assert res.metrics["success_time"] >= plan.t_hold + 1.2 + task.robustness.post_perturb_hold - 1e-9


# --------------------------------------------------------------------------- battery


def test_plan_has_13_placement_3_shake_2_push_tests() -> None:
    tests = plan_tests(CAN, Placement(0.01, 0.02, 0.3), TASK)
    assert [t.category for t in tests].count("placement") == 13
    assert [t.category for t in tests].count("shake") == 3
    assert [t.category for t in tests].count("push") == 2
    names = [t.name for t in tests]
    assert names[:3] == ["nominal", "offset_x_+1mm", "offset_x_-1mm"]
    assert "offset_y_-3mm" in names and "shake_z" in names and "push_y" in names
    by_name = {t.name: t for t in tests}
    assert by_name["offset_y_-3mm"].placement.err_y == pytest.approx(-0.003)
    assert by_name["offset_x_+2mm"].placement.err_x == pytest.approx(0.002)
    assert by_name["push_x"].perturbation.height == pytest.approx(push_height(CAN)) == pytest.approx(0.04)


def test_tolerance_is_the_largest_error_with_no_failure() -> None:
    def t(name: str, axis: str | None, err: float, ok: bool) -> CertificateTest:
        return CertificateTest(name, "placement", {"axis": axis, "error_mm": err}, ok, None, 0.0, 0.0, ())

    tests = [t("nominal", None, 0.0, True)]
    tests += [t("", "x", e, True) for e in (1, -1, 2, -2)] + [t("", "x", 3, False), t("", "x", -3, True)]
    tests += [t("", "y", 1, False), t("", "y", -1, True)]
    assert _tolerance(tests, "x") == 2.0
    assert _tolerance(tests, "y") == 0.0
    tests[0] = t("nominal", None, 0.0, False)
    assert _tolerance(tests, "x") is None


def test_certificate_structure_and_scores(cert: Certificate) -> None:
    assert len(cert.tests) == 18
    assert cert.design_id == STOCK_DESIGN.design_id()
    assert cert.controller_setting in (0, 1, 2)
    assert set(cert.category_scores) == {"placement", "shake", "push"}
    assert cert.score == pytest.approx(sum(cert.category_scores.values()) / 3)
    for cat, value in cert.category_scores.items():
        members = [x.success for x in cert.tests if x.category == cat]
        assert value == pytest.approx(sum(members) / len(members))
    nominal = cert.tests[0]
    assert nominal.name == "nominal" and nominal.success
    assert cert.tolerance_mm["x"] is not None and cert.tolerance_mm["y"] is not None
    d = cert.to_dict()
    assert d["passed"] == f"{cert.passed}/18"
    assert d["settings"]["jitter"] == "off"
    assert d["settings"]["shake"] == {"amplitude_m": 0.01, "frequency_hz": 5.0, "cycles": 5}


def test_certificate_is_deterministic(cert: Certificate) -> None:
    again = certify(
        STOCK_DESIGN, CAN, controller_setting=cert.controller_setting, workers=4, task=TASK, seed=99
    )
    assert [t.success for t in again.tests] == [t.success for t in cert.tests]
    assert [t.max_lift for t in again.tests] == [t.max_lift for t in cert.tests]  # jitter off: seed is inert
