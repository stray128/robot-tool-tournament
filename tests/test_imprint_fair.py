"""Gate GV1 pieces: the simulated imprint matches its design, and it registers at zero error.

The anti-strawman rule (spec section 9): the configurator imprint and the swept imprint must
be fair comparators, i.e. the clearance MuJoCo sees at the seat is the designed one (within
0.05 mm) and both register when the part arrives with no placement error. These tests run on
the fixture builder (``freeform_fixtures``); the integrator re-runs ``rtt.sim.gates_v2 gv1``
with the generator.
"""

from __future__ import annotations

import pytest

from freeform_fixtures import (
    FINE,
    block,
    build_fixture_pair,
    flat,
    gv1_designs,
    imprint_configurator,
    imprint_swept,
    training_parts,
)
from rtt.sim.controller import plan_grasp_v2
from rtt.sim.episode import run_episode_v2
from rtt.sim.gates_v2 import FIDELITY_TOL_MM, run_gv1
from rtt.sim.objects import Placement
from rtt.sim.register import lateral_clearance, probe_first_contact, seat_clearance
from rtt.sim.scene import build_model_v2
from rtt.sim.task import load_task_v2

TASK = load_task_v2()
ZERO = TASK.with_overrides(jitter={"pos": 0.0, "yaw_deg": 0.0})
PARTS = training_parts()


@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.part_id)
def test_the_simulated_imprint_clearance_is_the_designed_one(part) -> None:
    pair = build_fixture_pair(imprint_configurator(**FINE), part, TASK)
    obj = part.to_mesh_object()
    model, ids = build_model_v2(pair, obj, TASK)
    plan = plan_grasp_v2(pair, obj, Placement(), TASK)
    probe = seat_clearance(model, ids, plan, obj, pair.q_seat, [0.0004, 0.001])
    for side in ("left", "right"):
        assert probe[side][0] * 1000 == pytest.approx(0.4, abs=FIDELITY_TOL_MM)
        # backed off further, the pocket's side walls (0.4 mm away) may still be the nearest
        assert 0.4 - FIDELITY_TOL_MM <= probe[side][1] * 1000 <= 1.0 + FIDELITY_TOL_MM
    for q in probe["first_contact"]:
        assert (q - pair.q_seat) * 1000 == pytest.approx(0.0, abs=FIDELITY_TOL_MM)


@pytest.mark.parametrize(("design", "side_mm"), [(imprint_configurator, 0.4), (imprint_swept, 2.4)])
def test_the_lateral_slide_at_the_seat_is_the_side_clearance_on_straight_sides(design, side_mm) -> None:
    # the closing-axis check above cannot see a side wall that is too far out; this can
    blk = block()
    pair = build_fixture_pair(design(**FINE), blk, TASK)
    obj = blk.to_mesh_object()
    model, ids = build_model_v2(pair, obj, TASK)
    plan = plan_grasp_v2(pair, obj, Placement(), TASK)
    plus, minus = lateral_clearance(model, ids, plan, obj, pair.q_seat)
    assert plus * 1000 == pytest.approx(side_mm, abs=FIDELITY_TOL_MM)
    assert minus * 1000 == pytest.approx(side_mm, abs=FIDELITY_TOL_MM)
    # with the jaws past the seat the part is held, so it cannot slide at all
    assert lateral_clearance(model, ids, plan, obj, pair.q_seat + 0.0002) == (0.0, 0.0)


def test_a_flat_pad_touches_at_its_seat_with_no_clearance() -> None:
    blk = block()
    pair = build_fixture_pair(flat(finger_width_mm=40.0), blk, TASK)
    obj = blk.to_mesh_object()
    model, ids = build_model_v2(pair, obj, TASK)
    plan = plan_grasp_v2(pair, obj, Placement(), TASK)
    q_left, q_right = probe_first_contact(model, ids, plan, obj)
    assert q_left * 1000 == pytest.approx(pair.q_seat * 1000, abs=0.01)
    assert q_right * 1000 == pytest.approx(pair.q_seat * 1000, abs=0.01)


def test_the_probe_sees_a_placement_error_as_an_earlier_contact_on_one_side() -> None:
    blk = block()
    pair = build_fixture_pair(flat(finger_width_mm=40.0), blk, TASK)
    obj = blk.to_mesh_object()
    model, ids = build_model_v2(pair, obj, TASK)
    plan = plan_grasp_v2(pair, obj, Placement(), TASK)
    shifted = Placement(err_y=0.002)  # toward the left jaw
    q_left, q_right = probe_first_contact(model, ids, plan, obj, placement=shifted)
    assert (pair.q_seat - q_left) * 1000 == pytest.approx(2.0, abs=0.01)
    assert (q_right - pair.q_seat) * 1000 == pytest.approx(2.0, abs=0.01)


@pytest.mark.parametrize("part", [PARTS[1], PARTS[2], PARTS[5]], ids=lambda p: p.part_id)
def test_the_configurator_imprint_registers_at_zero_error(part) -> None:
    pair = build_fixture_pair(imprint_configurator(**FINE), part, TASK)
    for seed in (0, 1):
        res = run_episode_v2(pair, part, Placement(), seed, record=False, task=ZERO)
        assert res.success, res.summary()
        reg = res.registration
        assert abs(reg.seat_short_mm) <= 0.2 and abs(reg.dx_mm) <= 0.2 and abs(reg.dy_mm) <= 0.2


def test_the_block_imprint_jams_on_the_rim_under_a_placement_error() -> None:
    """The measured jam case (no lead-in): at 2 mm the pocket rides the rim and stops short."""
    blk = block()
    pair = build_fixture_pair(imprint_configurator(**FINE), blk, TASK)
    res = run_episode_v2(pair, blk, Placement(err_x=0.002), 0, record=False, task=ZERO)
    assert not res.success
    assert res.registration.seat_short_mm > 1.0
    assert "jam_on_close" in res.registration.sub_flags
    if res.lift_success:  # held where it landed: lift-only passes, the registered grasp does not
        assert res.failure_mode == "not_registered"


def test_gv1_report_shape_on_one_part() -> None:
    imprints = {k: v for k, v in gv1_designs().items() if k == "imprint_configurator"}
    report = run_gv1(build_fixture_pair, [block()], imprints, TASK, seeds=2, workers=1)
    cell = report["clearance"]["imprint_configurator/block_30x40x60"]
    assert cell["passed"] and cell["designed_mm"] == pytest.approx(0.4)
    reg = report["registration"]["imprint_configurator/block_30x40x60"]
    assert reg["episodes"] == 2 and reg["zero_error_registered"] == 1.0
    assert report["passed"] == {"clearance": True, "zero_error_registration": True, "all": True}
    assert report["summary"]["worst_clearance_error_mm"] <= FIDELITY_TOL_MM


def test_gv1_fails_when_a_build_fails() -> None:
    def broken(design, part, task):
        raise ValueError("the part outline in the pad band is empty")

    imprints = {"imprint_configurator": gv1_designs()["imprint_configurator"]}
    report = run_gv1(broken, [block()], imprints, TASK, seeds=1, workers=1)
    assert report["invalid_builds"] and not report["passed"]["all"]
