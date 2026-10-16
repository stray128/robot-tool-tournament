"""Registration (task v2, D1): the rule, symmetry, the seat probe, and v2 episodes on v1 boxes."""

from __future__ import annotations

import dataclasses
import math

import pytest

from rtt.design.freeform.types import FingerPair
from rtt.sim.controller import plan_grasp_v2
from rtt.sim.episode import run_episode_v2
from rtt.sim.fingers_v2 import v1_pair
from rtt.sim.m1 import G1_BOX
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.register import (
    SUB_JAM,
    SUB_NOT_MEASURED,
    SUB_OFF_CENTRE,
    SUB_TILT_HOLD,
    PalmPose,
    RegistrationSpec,
    assess,
    not_measured,
    probe_first_contact,
    registration_spec,
)
from rtt.sim.scene import build_model_v2
from rtt.sim.task import (
    DEFAULT_TASK_V2_PATH,
    RegistrationConfig,
    TaskV2,
    load_task,
    load_task_v2,
    task_digest,
    task_v2_from_dict,
)
from sim_fingers import straight_finger

TASK = load_task_v2()
CFG = TASK.registration


def _pose(dx=0.0, dy=0.0, yaw_deg=0.0, tilt_deg=0.0) -> PalmPose:
    return PalmPose(dx / 1000, dy / 1000, 0.0, math.radians(yaw_deg), math.radians(tilt_deg))


# --------------------------------------------------------------------------- task v2


def test_task_v2_yaml_matches_the_dataclass_plus_its_overrides() -> None:
    v1 = load_task()
    expected = TaskV2(
        controller=dataclasses.replace(v1.controller, grasp_height_fracs=()),
        jitter=dataclasses.replace(v1.jitter, pos=0.0005, yaw_deg=1.0, mu_frac=0.10),
        physics=dataclasses.replace(v1.physics, timestep=0.001, native_ccd=True),
    )
    assert TASK == expected
    assert TASK.name == "task_v2" and DEFAULT_TASK_V2_PATH.name == "task_v2.yaml"
    assert TASK.registration == RegistrationConfig()
    # v2 differs from v1 only where the spec and the founder rulings of 2026-10-16 say so:
    # 1 ms (ruling 3), native collision (ruling 2), the penetration rule (ruling 4)
    for section in ("gripper", "score", "robustness", "replay"):
        assert getattr(TASK, section) == getattr(v1, section)
    assert dataclasses.replace(TASK.physics, timestep=0.002, native_ccd=False) == v1.physics
    assert TASK.physics.timestep == 0.001 and TASK.controller.control_period == 0.002
    assert TASK.checks.held_penetration_max_mm == 1.0


def test_task_v1_is_untouched_by_v2() -> None:
    v1 = load_task()
    assert type(v1).__name__ == "TaskConfig" and not hasattr(v1, "registration")
    assert task_digest(v1) != task_digest(TASK)


def test_task_v2_rejects_a_grasp_height_grid_and_bad_settle_times() -> None:
    raw = {"controller": {"grasp_height_fracs": [0.5]}}
    with pytest.raises(ValueError, match="grasp-height grid"):
        task_v2_from_dict(raw)
    with pytest.raises(ValueError, match="squeeze dwell"):
        task_v2_from_dict({"controller": {"grasp_height_fracs": []}, "registration": {"settle_time": 0.4}})
    with pytest.raises(ValueError, match="multiple of the control period"):
        task_v2_from_dict({"controller": {"grasp_height_fracs": []}, "registration": {"settle_time": 0.2501}})
    with pytest.raises(ValueError, match="unknown keys in registration"):
        task_v2_from_dict({"controller": {"grasp_height_fracs": []}, "registration": {"dz_max_mm": 1.0}})


def test_with_overrides_keeps_the_v2_type() -> None:
    t = TASK.with_overrides(registration={"dx_max_mm": 1.0}, physics={"friction": 0.05})
    assert isinstance(t, TaskV2) and t.registration.dx_max_mm == 1.0 and t.physics.friction == 0.05


# --------------------------------------------------------------------------- the rule


def test_a_seated_centred_part_is_registered() -> None:
    r = assess(_pose(0.2, -0.3, 1.0, 0.5), q=0.0201, q_seat=0.0200, spec=RegistrationSpec(), cfg=CFG)
    assert r.registered and r.seated and r.centred and r.sub_flags == ()
    assert r.seat_short_mm == pytest.approx(-0.1)
    assert r.dx_mm == pytest.approx(0.2) and r.dy_mm == pytest.approx(-0.3)


@pytest.mark.parametrize(
    ("pose", "q", "flag"),
    [
        (_pose(), 0.0185, SUB_JAM),  # stalled 1.5 mm short of the seat
        (_pose(dx=0.8), 0.020, SUB_OFF_CENTRE),
        (_pose(dy=-0.8), 0.020, SUB_OFF_CENTRE),
        (_pose(yaw_deg=2.5), 0.020, SUB_OFF_CENTRE),
        (_pose(tilt_deg=3.5), 0.020, SUB_OFF_CENTRE),
    ],
)
def test_each_tolerance_fails_on_its_own(pose: PalmPose, q: float, flag: str) -> None:
    r = assess(pose, q=q, q_seat=0.020, spec=RegistrationSpec(), cfg=CFG)
    assert not r.registered and r.sub_flags == (flag,)


def test_jam_at_the_measured_tee_numbers_is_caught_by_the_seat_not_the_pose() -> None:
    """A part jammed 5 mm short can sit only 0.42 mm off centre: the pose rule alone misses it."""
    r = assess(_pose(dx=0.42), q=0.0292, q_seat=0.03425, spec=RegistrationSpec(), cfg=CFG)
    assert r.centred and not r.seated and r.sub_flags == (SUB_JAM,)


def test_tilt_after_the_hold() -> None:
    upright = assess(_pose(), 0.02, 0.02, RegistrationSpec(), CFG, hold_tilt=math.radians(4.9))
    tipped = assess(_pose(), 0.02, 0.02, RegistrationSpec(), CFG, hold_tilt=math.radians(5.1))
    assert upright.registered and not tipped.registered and tipped.sub_flags == (SUB_TILT_HOLD,)
    unheld = assess(_pose(), 0.02, 0.02, RegistrationSpec(), CFG)  # hold never confirmed: NaN
    assert unheld.registered and math.isnan(unheld.hold_tilt_deg)


def test_not_measured_is_not_registered() -> None:
    r = not_measured(0.02)
    assert not r.registered and r.sub_flags == (SUB_NOT_MEASURED,) and math.isnan(r.seat_short_mm)


def test_yaw_is_taken_modulo_the_symmetry() -> None:
    hexagon = RegistrationSpec(6)
    assert hexagon.yaw_residual_deg(59.0) == pytest.approx(1.0)
    assert hexagon.yaw_residual_deg(-61.5) == pytest.approx(1.5)
    assert RegistrationSpec(1).yaw_residual_deg(359.0) == pytest.approx(1.0)
    assert RegistrationSpec(0).yaw_residual_deg(37.0) == 0.0
    r = assess(_pose(yaw_deg=59.0), 0.02, 0.02, hexagon, CFG)
    assert r.registered and r.dyaw_deg == pytest.approx(1.0)


def test_undeclared_axes_are_not_checked() -> None:
    spec = RegistrationSpec(1, ("y", "tilt"))
    r = assess(_pose(dx=5.0, yaw_deg=10.0), 0.02, 0.02, spec, CFG)
    assert r.registered
    with pytest.raises(ValueError):
        RegistrationSpec(1, ("x", "roll"))


def test_registration_spec_of_v1_objects_and_parts() -> None:
    assert registration_spec(ObjectSpec("box", (0.03, 0.04, 0.05))) == RegistrationSpec(2)
    assert registration_spec(ObjectSpec("box", (0.04, 0.04, 0.05))) == RegistrationSpec(4)
    assert registration_spec(ObjectSpec("cylinder", (0.02, 0.05))).yaw_symmetry == 0
    assert not registration_spec(ObjectSpec("cylinder", (0.02, 0.05))).checks("yaw")
    assert registration_spec(ObjectSpec("rod", (0.01, 0.1))) == RegistrationSpec(2)
    from freeform_fixtures import hex_standoff, steel_slug

    assert registration_spec(hex_standoff()) == RegistrationSpec(6, ("x", "y", "yaw", "tilt"))
    assert registration_spec(steel_slug()) == RegistrationSpec(0, ("x", "y", "tilt"))


# --------------------------------------------------------------------------- probes and v1 boxes


def _box_pair(obj, tip=0.06) -> FingerPair:
    """The straight 80 mm test finger as a pair (tips 60 mm up: the palm clears a 120 mm box),
    with its seat closure probed in the scored scene."""
    finger = straight_finger()
    rough = v1_pair(finger, part_id="g1_box", tip_height=tip, q_seat=0.01)
    model, ids = build_model_v2(rough, obj, TASK)
    q_left, q_right = probe_first_contact(model, ids, plan_grasp_v2(rough, obj, Placement(), TASK), obj)
    return v1_pair(finger, part_id="g1_box", tip_height=tip, q_seat=min(q_left, q_right))


def test_probe_finds_the_analytic_seat_of_flat_boxes() -> None:
    # inner faces at local y = 0 touch a 40 mm wide box at q = 42 - 20 = 22 mm
    pair = _box_pair(G1_BOX)
    assert pair.q_seat == pytest.approx(0.022, abs=1e-6)
    cyl = ObjectSpec("cylinder", (0.025, 0.09))
    assert _box_pair(cyl).q_seat == pytest.approx(0.017, abs=1e-5)


def test_v1_boxes_register_when_centred_and_not_when_offset() -> None:
    pair = _box_pair(G1_BOX)
    centred = run_episode_v2(pair, G1_BOX, Placement(), seed=3, record=False, task=TASK, check_part=False)
    assert centred.success and centred.lift_success and centred.registered, centred.summary()
    reg = centred.registration
    assert abs(reg.dx_mm) <= 0.5 + 0.1 and abs(reg.seat_short_mm) < 0.5  # only the +-0.5 mm jitter
    offset = run_episode_v2(
        pair, G1_BOX, Placement(err_x=0.002), seed=3, record=False, task=TASK, check_part=False
    )
    assert offset.lift_success and not offset.success
    assert offset.failure_mode == "not_registered"
    assert offset.registration.sub_flags == (SUB_OFF_CENTRE,)
    assert offset.registration.dx_mm == pytest.approx(2.0, abs=0.6)
    assert offset.metrics["registered"] == 0.0 and offset.metrics["lift_success"] == 1.0


def test_v2_episode_needs_a_v2_task_and_the_right_part() -> None:
    pair = _box_pair(G1_BOX)
    with pytest.raises(TypeError, match="TaskV2"):
        run_episode_v2(pair, G1_BOX, Placement(), seed=0, record=False, task=load_task())
    with pytest.raises(ValueError, match="built for part"):
        run_episode_v2(pair, G1_BOX, Placement(), seed=0, record=False, task=TASK)
