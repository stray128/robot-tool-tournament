"""Track S2: the six articulated Studio templates in physics (``docs/STUDIO.md`` 3.2, ``docs/STUDIO_PRIMITIVES.md``).

* every template kind has a builder; every variant (lid or door, lip or handle; bar, knob or lip
  pulls; wing or slot heads; cap, coupling or nut; spoked or solid wheels; plain or latching
  ports) compiles in the Studio scene with its handle sites and rests still;
* every param is honoured: joint ranges, frictions and dampings, the passive torque curves
  through the templates' own actuators (catch, detent, thread ramp, valve curve), the thread's
  helical flank, the socket's spring pads and latch groove;
* the state each builder reports is the template's quantities, and the controllers' view of
  it (:func:`rtt.studio.primitives._common.observed_state`, observed sites only) agrees with it
  and does not move with the hidden aim offset;
* Studio v2's four job templates (``docs/STUDIO_PRIMITIVES.md`` section 13: a panel on rails with
  a clamp, a tile on a mortar bed that keeps its set, a push button, a sliding machine door): their
  physics, the S2 rule on each (a fitting tool succeeds where a wrong one fails), the push's optional
  ``approach`` param (the frozen hero ids unchanged) and that the Task Engineer is not offered them yet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.sim.checks import bad_state_count
from rtt.sim.task import default_task_v2
from rtt.studio.plan import StepContext
from rtt.studio.primitives._common import observed_state
from rtt.studio.scene import build_studio_scene
from rtt.studio.templates import builder_kinds, get_template_builder
from rtt.studio.templates.handwheel_valve import torque_curve
from rtt.studio.templates.hinged_lid import CATCH_DEG
from rtt.studio.templates.quarter_turn_latch import DETENT_DEG
from rtt.studio.templates.threaded_joint import lead_of
from rtt.studio.types import (
    PRIMITIVES,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
    TemplateKind,
    get_template,
)
from studio_fixtures import gripper

HANDLE_OF = {
    "hinged_lid": ("hook_pull", "handle"),
    "drawer": ("hook_pull", "handle"),
    "quarter_turn_latch": ("twist", "head"),
    "threaded_joint": ("twist", "member"),
    "handwheel_valve": ("twist", "wheel"),
    "port_plug": ("insert", "plug"),
    "panel_rail": ("twist", "member"),
    "tile_bed": ("push", "part"),
    "spring_button": ("push", "button"),
    "sliding_door": ("hook_pull", "handle"),
}
VARIANTS: dict[str, dict[str, Any]] = {
    "lid_lip": {"kind": "hinged_lid"},
    "lid_handle": {"kind": "hinged_lid", "feature": "handle"},
    "door_lip": {"kind": "hinged_lid", "orientation": "door"},
    "door_handle": {"kind": "hinged_lid", "orientation": "door", "feature": "handle", "max_angle_deg": 180},
    "drawer_bar": {"kind": "drawer"},
    "drawer_knob": {"kind": "drawer", "handle": "knob"},
    "drawer_lip": {"kind": "drawer", "handle": "lip"},
    "latch_wing": {"kind": "quarter_turn_latch"},
    "latch_slot": {"kind": "quarter_turn_latch", "head": "slot", "detent": False},
    "thread_coupling": {"kind": "threaded_joint"},
    "thread_cap": {"kind": "threaded_joint", "member": "cap"},
    "thread_nut": {"kind": "threaded_joint", "member": "nut", "chamfer_mm": 0.0},
    "valve_spoked": {"kind": "handwheel_valve"},
    "valve_disc": {"kind": "handwheel_valve", "spokes": 0},
    "port": {"kind": "port_plug"},
    "port_latch": {"kind": "port_plug", "latch": True},
    "panel_rail": {"kind": "panel_rail"},
    "tile_bed": {"kind": "tile_bed"},
    "button_wall": {"kind": "spring_button"},
    "button_desk": {"kind": "spring_button", "mount": "desk"},
    "sliding_door": {"kind": "sliding_door"},
}


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


def one_object_task(name: str, pose: ObjectPose | None = None, **params: Any) -> TaskSpec:
    params = dict(params)
    kind = params.pop("kind")
    inst = ObjectInstance.of(name, kind, name.replace("_", " "), pose or ObjectPose(0.0, 80.0, 0.0), **params)
    primitive, handle = HANDLE_OF[kind]
    quantity = get_template(kind).quantities[0]
    step = Step.of("s", primitive, name, handle, "probe", [SuccessCriterion(quantity, at_least=0.0)])
    return TaskSpec(f"{kind} probe", "", (inst,), (step,), source="test")


def scene_of(spec: TaskSpec, task: Any) -> Any:
    return build_studio_scene(Setup.single(gripper(), spec.step_ids), spec, 0, task=task)


def settle(scene: Any, seconds: float = 0.5, applied: dict[int, float] | None = None) -> mujoco.MjData:
    """Run the scene with the wrist parked high and the objects' passive actuators driven."""
    m = scene.model
    d = mujoco.MjData(m)
    d.qpos[scene.ids.palm_qpos] = [0.0, 0.0, 0.45, 0.0]
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = [0.0, 0.0, 0.45, 0.0]
    ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
    for _ in range(int(seconds / 0.002)):
        for obj in scene.objects.values():
            if len(obj.actuator_ids):
                ctrl[obj.actuator_ids] = scene.builders[obj.name].passive(obj, m, d)
        d.ctrl[:] = ctrl
        for dof, f in (applied or {}).items():
            d.qfrc_applied[dof] = f
        mujoco.mj_step(m, d, nstep=2)
    return d


def ctx_for(
    scene: Any, spec: TaskSpec, task: Any, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> StepContext:
    step = spec.steps[0]
    return StepContext(
        step=step,
        task_spec=spec,
        scene=scene,
        obj=scene.objects[step.object],
        target=None,
        head=scene.active,
        wrist_start=np.zeros(5),
        task=task,
        seed=0,
        aim_offsets={step.object: offset},
    )


# --------------------------------------------------------------------------- every variant


def test_every_template_kind_has_a_builder() -> None:
    assert set(builder_kinds()) == set(TemplateKind)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_builds_rests_and_reports_its_quantities(variant: str, task: Any) -> None:
    params = VARIANTS[variant]
    spec = one_object_task("thing", **params)
    scene = scene_of(spec, task)
    obj = scene.objects["thing"]
    template = get_template(params["kind"])
    assert set(template.handles) <= set(obj.handles.sites)
    assert all(n.startswith("obj_thing_") for n in obj.handles.geoms)
    assert np.all(scene.model.geom_priority[obj.geom_ids] == 1)
    builder = scene.builders["thing"]
    d0 = mujoco.MjData(scene.model)
    mujoco.mj_forward(scene.model, d0)
    s0 = builder.state(obj, scene.model, d0)
    assert set(s0) == set(template.quantities)
    d = settle(scene, 0.6)
    s1 = builder.state(obj, scene.model, d)
    for q, v in s1.items():
        assert v == pytest.approx(s0[q], abs=0.25), (variant, q)  # at rest: no drift, no creep
    assert bad_state_count(d) == 0


# --------------------------------------------------------------------------- params honoured


def test_hinged_lid_params_hinge_catch_and_features(task: Any) -> None:
    spec = one_object_task("lid", kind="hinged_lid", max_angle_deg=130, damping_nms=0.4, open_torque_nm=0.5)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["lid"]
    j = m.joint("obj_lid_hinge")
    assert j.range[1] == pytest.approx(math.radians(130)) and m.dof_damping[j.dofadr[0]] == pytest.approx(0.4)
    builder = scene.builders["lid"]
    d = mujoco.MjData(m)
    for deg, want in ((1.0, -0.5), (CATCH_DEG, -0.5), (1.5 * CATCH_DEG, -0.25), (3 * CATCH_DEG, 0.0)):
        d.qpos[obj.joint_qpos["hinge"]] = math.radians(deg)
        assert builder.passive(obj, m, d)[0] == pytest.approx(want), deg
    axis = [obj.handles.info[k] for k in ("axis_x", "axis_y", "axis_z")]
    assert axis == pytest.approx([-1.0, 0.0, 0.0])  # a lid: horizontal hinge, positive opens
    door = scene_of(one_object_task("door", kind="hinged_lid", orientation="door", feature="handle"), task)
    dinfo = door.objects["door"].handles.info
    assert [dinfo[k] for k in ("axis_x", "axis_y", "axis_z")] == pytest.approx([0.0, 0.0, -1.0])
    assert any(door.model.geom(g).name == "obj_door_bar" for g in door.objects["door"].geom_ids)
    # the lid's lip handle is the middle of the free edge; a lid stays open past vertical
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    edge = obj.site_pos(d, "handle")
    assert edge[1] == pytest.approx(0.080 - 0.100, abs=1e-6) and edge[2] == pytest.approx(
        0.120 + 0.003, abs=1e-6
    )
    d.qpos[obj.joint_qpos["hinge"]] = math.radians(100)
    mujoco.mj_forward(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d, nstep=2)
    assert math.degrees(d.qpos[obj.joint_qpos["hinge"]]) > 105  # past vertical it falls open, not shut


def test_drawer_slide_friction_travel_and_handles(task: Any) -> None:
    spec = one_object_task("dr", kind="drawer", travel_mm=180, pull_force_n=12)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["dr"]
    j = m.joint("obj_dr_slide")
    assert j.range[1] == pytest.approx(0.180) and m.dof_frictionloss[j.dofadr[0]] == pytest.approx(12.0)
    # a push below the friction does not move it (near-hard dry friction); above it does
    for force, moves in ((8.0, False), (20.0, True)):
        d = settle(scene, 0.5, applied={int(j.dofadr[0]): force})
        travel = scene.builders["dr"].state(obj, m, d)["travel_mm"]
        assert (travel > 5.0) is moves, (force, travel)
    for handle, geom in (("knob", "obj_k_knob"), ("lip", "obj_k_lip")):
        s = scene_of(one_object_task("k", kind="drawer", handle=handle), task)
        assert geom in {s.model.geom(g).name for g in s.objects["k"].geom_ids}


def test_latch_breakaway_detent_and_stop(task: Any) -> None:
    spec = one_object_task("la", kind="quarter_turn_latch", torque_nm=0.6, turn_deg=90, detent=True)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["la"]
    dof = int(obj.joint_dof["hinge"])
    assert m.dof_frictionloss[dof] == pytest.approx(0.15)
    d = mujoco.MjData(m)
    assert scene.builders["la"].passive(obj, m, d)[0] == pytest.approx(-0.45)
    d.qpos[obj.joint_qpos["hinge"]] = math.radians(DETENT_DEG + 1)
    assert scene.builders["la"].passive(obj, m, d)[0] == 0.0
    # below the breakaway torque it stays locked; above it, it turns counter-clockwise to its stop
    held = settle(scene, 0.4, applied={dof: 0.5})
    assert scene.builders["la"].state(obj, m, held)["angle_deg"] < 2.0
    opened = settle(scene, 0.8, applied={dof: 0.7})
    assert scene.builders["la"].state(obj, m, opened)["angle_deg"] == pytest.approx(90.0, abs=1.5)
    slot = scene_of(one_object_task("sl", kind="quarter_turn_latch", head="slot"), task)
    names = {slot.model.geom(g).name for g in slot.objects["sl"].geom_ids}
    assert {"obj_sl_cheek0", "obj_sl_cheek1"} <= names and "obj_sl_fin" not in names


def test_threaded_joint_lead_in_flank_seat_and_torque_ramp(task: Any) -> None:
    spec = one_object_task(
        "th", kind="threaded_joint", pitch_mm=2.0, engage_turns=2.0, start_torque_nm=0.1, final_torque_nm=1.0
    )
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["th"]
    info = obj.handles.info
    lead = lead_of(1.0)
    assert info["lead_m"] == pytest.approx(lead) and info["seat_m"] == pytest.approx(0.004)
    slide, hinge = int(obj.joint_dof["slide"]), int(obj.joint_dof["hinge"])
    builder = scene.builders["th"]
    # it rests on the lead-in; a push slides it to the thread start and no further (the flank)
    rest = builder.state(obj, m, settle(scene, 0.4))
    assert rest["depth_mm"] == pytest.approx(0.0, abs=0.05)
    pushed = builder.state(obj, m, settle(scene, 0.6, applied={slide: 20.0}))
    assert pushed["depth_mm"] == pytest.approx(1000 * lead, abs=0.3) and pushed["turns"] == pytest.approx(
        0, abs=0.01
    )
    # tightening (above the final torque) while pushed advances one pitch per turn until seated
    d = settle(scene, 4.0, applied={slide: 5.0, hinge: 1.2})
    st = builder.state(obj, m, d)
    assert st["depth_mm"] == pytest.approx(1000 * (lead + 0.004), abs=0.3)  # seated: the slide's stop
    assert st["turns"] >= 2.0
    # the friction ramp (start torque at the thread start, final when seated) is a Coulomb
    # friction: under it the member holds (no creep), over it the member turns (S2c)
    for depth, ramp in ((lead, 0.1), (lead + 0.002, 0.55), (lead + 0.004, 1.0)):
        for factor, turns in ((0.85, False), (1.2, True)):
            first, second = _turn_under(scene, depth, factor * ramp, info["pitch_m"])
            if turns:
                assert first + second > 10.0, (depth, factor)
            else:
                assert first < 1.0 and abs(second) < 0.5, (depth, factor, first, second)  # deg, <= 1 deg/s
    for member, geom in (("cap", "obj_c_top"), ("nut", "obj_c_piece05")):
        s = scene_of(one_object_task("c", kind="threaded_joint", member=member), task)
        assert geom in {s.model.geom(g).name for g in s.objects["c"].geom_ids}


def _turn_under(scene: Any, depth: float, torque: float, pitch: float) -> tuple[float, float]:
    """A threaded member set ``depth`` down its thread (on the flank) under a steady tightening
    ``torque`` (ramped in over 0.2 s): how far it turns in the first and the second half second
    (deg). The template's passive control runs every 2 ms tick, as in the Studio runner."""
    m = scene.model
    obj = next(iter(scene.objects.values()))
    info = obj.handles.info
    builder = next(iter(scene.builders.values()))
    d = mujoco.MjData(m)
    d.qpos[scene.ids.palm_qpos] = [0.0, 0.0, 0.45, 0.0]
    d.qpos[obj.joint_qpos["slide"]] = depth
    d.qpos[obj.joint_qpos["hinge"]] = max(0.0, depth - info["lead_m"]) / pitch * 2.0 * math.pi
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = [0.0, 0.0, 0.45, 0.0]
    ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
    hinge = int(obj.joint_dof["hinge"])
    angles = [float(d.qpos[obj.joint_qpos["hinge"]])]
    for tick in range(500):
        ctrl[obj.actuator_ids] = builder.passive(obj, m, d)
        d.ctrl[:] = ctrl
        d.qfrc_applied[hinge] = torque * min(1.0, tick * 0.002 / 0.2)
        mujoco.mj_step(m, d, nstep=2)
        if tick in (249, 499):
            angles.append(float(d.qpos[obj.joint_qpos["hinge"]]))
    return math.degrees(angles[1] - angles[0]), math.degrees(angles[2] - angles[1])


@pytest.mark.parametrize("depth_frac", [0.0, 0.5, 1.0])
def test_a_driven_thread_resists_with_its_ramp_torque(depth_frac: float, task: Any) -> None:
    """A stiff speed loop turns the member at 60 deg/s at a fixed depth down its thread: the torque
    it needs is the ramp's (plus the dry hold), within 10 % (S2c)."""
    spec = one_object_task(
        "th", kind="threaded_joint", pitch_mm=1.5, engage_turns=2.0, start_torque_nm=0.4, final_torque_nm=4.0
    )
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["th"]
    info = obj.handles.info
    builder = scene.builders["th"]
    d = mujoco.MjData(m)
    d.qpos[scene.ids.palm_qpos] = [0.0, 0.0, 0.45, 0.0]
    depth = info["lead_m"] + depth_frac * info["seat_m"]
    d.qpos[obj.joint_qpos["slide"]] = depth
    d.qpos[obj.joint_qpos["hinge"]] = depth_frac * info["seat_m"] / info["pitch_m"] * 2.0 * math.pi
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = [0.0, 0.0, 0.45, 0.0]
    ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
    hinge, hq = int(obj.joint_dof["hinge"]), obj.joint_qpos["hinge"]
    target = math.radians(60.0)
    start = float(d.qpos[hq])
    applied = []
    for tick in range(750):  # 1.5 s; a PD loop on a ramped angle (stiff enough, explicit-stable)
        ctrl[obj.actuator_ids] = builder.passive(obj, m, d)
        d.ctrl[:] = ctrl
        for _ in range(2):
            want = start + target * float(d.time)
            tau = 20.0 * (want - float(d.qpos[hq])) + 0.2 * (target - float(d.qvel[hinge]))
            d.qfrc_applied[hinge] = tau
            applied.append(tau)
            mujoco.mj_step(m, d)
    s = min(1.0, depth_frac)
    ramp = info["start_nm"] + (info["final_nm"] - info["start_nm"]) * s
    turning = np.array(applied[750:])  # the second half: steady turning
    assert float(np.mean(turning)) == pytest.approx(ramp, rel=0.10), (
        depth_frac,
        float(np.mean(turning)),
        ramp,
    )
    assert float(d.qvel[hinge]) == pytest.approx(target, rel=0.1)


def test_handwheel_torque_curve_travel_and_geometry(task: Any) -> None:
    assert torque_curve(0.0, 1.0) == pytest.approx(1.0)
    assert torque_curve(0.5, 1.0) == pytest.approx(0.6)
    assert torque_curve(1.0, 1.0) == pytest.approx(0.8)
    spec = one_object_task(
        "va", kind="handwheel_valve", torque_nm=2.0, travel_deg=270, spokes=4, rim_diameter_mm=160
    )
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["va"]
    dof = int(obj.joint_dof["hinge"])
    assert m.joint("obj_va_hinge").range[1] == pytest.approx(math.radians(270))
    assert m.dof_frictionloss[dof] == pytest.approx(1.0)
    names = {m.geom(g).name for g in obj.geom_ids}
    assert {f"obj_va_spoke{k}" for k in range(4)} <= names
    # below the curve the wheel holds (S2c: also between its dry half and the curve, where the
    # old velocity-saturated motor let it creep open); above the curve it turns open
    for torque in (0.8, 1.8):  # a suddenly applied torque: the friction spring's twist, then no creep
        early = scene.builders["va"].state(obj, m, settle(scene, 0.3, applied={dof: torque}))["angle_deg"]
        late = scene.builders["va"].state(obj, m, settle(scene, 0.6, applied={dof: torque}))["angle_deg"]
        assert late < 2.0 and (late - early) / 0.3 < 1.0, (torque, early, late)  # deg, deg/s
    turned = settle(scene, 1.0, applied={dof: 2.6})
    assert scene.builders["va"].state(obj, m, turned)["angle_deg"] > 20.0


def test_port_socket_pads_and_latch_groove(task: Any) -> None:
    spec = one_object_task("po", kind="port_plug", insert_force_n=6.0, clearance_mm=0.3)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["po"]
    info = obj.handles.info
    assert info["pads"] == 3.0
    mu = task.physics.friction
    # three pads, preloaded 2 mm, squeezed 0.4 mm by a centred plug, give the insertion friction
    assert 3 * mu * info["pad_stiffness"] * 0.0024 == pytest.approx(6.0)
    assert info["bore_m"] == pytest.approx(0.5 * (0.012 + 0.0003))
    # the plug sits beside the socket, upright on the table; in the bore it reports its depth
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert scene.builders["po"].state(obj, m, d)["depth_mm"] == 0.0
    adr = obj.joint_qpos["free"]
    d.qpos[adr : adr + 3] = [
        info["socket_x"],
        info["socket_y"],
        info["socket_top_z"] + 0.5 * info["plug_len_m"] - 0.010,
    ]
    mujoco.mj_forward(m, d)
    assert scene.builders["po"].state(obj, m, d)["depth_mm"] == pytest.approx(10.0, abs=1e-6)
    latch = scene_of(one_object_task("pl", kind="port_plug", latch=True), task)
    radii = sorted(
        {
            round(float(latch.model.geom_size[g][0]), 5)
            for g in latch.objects["pl"].geom_ids
            if latch.model.geom(g).name.startswith("obj_pl_plug")
            and latch.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        }
    )
    assert radii == [pytest.approx(0.006 - 0.0008), pytest.approx(0.006)]  # the groove


def test_states_read_within_the_joints_own_limits(task: Any) -> None:
    """The solver's soft limits let a joint resting on its stop lean a hair past it, and a plug
    pressed onto the socket floor spring back a few hundredths of a millimetre: the states read
    within the limits, so a step that drives an object onto its stop meets a criterion at the stop."""
    lid_spec = one_object_task("lid", kind="hinged_lid", max_angle_deg=110)
    scene = scene_of(lid_spec, task)
    m, obj = scene.model, scene.objects["lid"]
    d = mujoco.MjData(m)
    d.qpos[obj.joint_qpos["hinge"]] = math.radians(110.0017)
    mujoco.mj_forward(m, d)
    assert scene.builders["lid"].state(obj, m, d)["angle_deg"] == pytest.approx(110.0, abs=1e-9)
    dr_spec = one_object_task("dr", kind="drawer", travel_mm=180)
    dr_scene = scene_of(dr_spec, task)
    dr, dd = dr_scene.objects["dr"], mujoco.MjData(dr_scene.model)
    dd.qpos[dr.joint_qpos["slide"]] = 0.1802
    assert dr_scene.builders["dr"].state(dr, dr_scene.model, dd)["travel_mm"] == pytest.approx(180.0)
    po_spec = one_object_task("po", kind="port_plug", depth_mm=15)
    po_scene = scene_of(po_spec, task)
    po, pm = po_scene.objects["po"], po_scene.model
    info = po.handles.info
    pd = mujoco.MjData(pm)
    adr = po.joint_qpos["free"]
    for below_floor, want in ((-0.00005, 15.0), (0.00005, 15.0), (0.0005, 14.5)):
        pd.qpos[adr : adr + 3] = [
            info["socket_x"],
            info["socket_y"],
            info["socket_top_z"] + 0.5 * info["plug_len_m"] - (info["depth_m"] - below_floor),
        ]
        mujoco.mj_forward(pm, pd)
        assert po_scene.builders["po"].state(po, pm, pd)["depth_mm"] == pytest.approx(want, abs=1e-6), (
            below_floor
        )


# --------------------------------------------------------------------------- the controllers' view


@pytest.mark.parametrize(
    "variant", ["lid_handle", "door_handle", "drawer_bar", "latch_wing", "valve_spoked", "thread_nut", "port"]
)
def test_observed_state_matches_the_truth_and_ignores_the_aim_offset(variant: str, task: Any) -> None:
    params = VARIANTS[variant]
    spec = one_object_task("thing", **params)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["thing"]
    d = mujoco.MjData(m)
    kind = params["kind"]
    if kind == "hinged_lid":
        d.qpos[obj.joint_qpos["hinge"]] = math.radians(37.0)
    elif kind == "drawer":
        d.qpos[obj.joint_qpos["slide"]] = 0.064
    elif kind in ("quarter_turn_latch", "handwheel_valve"):
        d.qpos[obj.joint_qpos["hinge"]] = math.radians(52.0)
    elif kind == "threaded_joint":
        d.qpos[obj.joint_qpos["slide"]] = obj.handles.info["lead_m"] + 0.0036
        d.qpos[obj.joint_qpos["hinge"]] = 2.0 * math.pi * 0.0036 / obj.handles.info["pitch_m"]
    else:
        adr = obj.joint_qpos["free"]
        info = obj.handles.info
        d.qpos[adr : adr + 3] = [
            info["socket_x"],
            info["socket_y"],
            info["socket_top_z"] + 0.5 * info["plug_len_m"] - 0.007,
        ]
    mujoco.mj_forward(m, d)
    truth = scene.builders["thing"].state(obj, m, d)
    for offset in ((0.0, 0.0, 0.0), (0.004, -0.003, 0.0)):
        seen = observed_state(ctx_for(scene, spec, task, offset), d)
        for q in seen:
            assert seen[q] == pytest.approx(truth[q], abs=0.05), (variant, q, offset)


def test_template_builders_are_registered_once() -> None:
    for kind in (
        "hinged_lid",
        "drawer",
        "quarter_turn_latch",
        "threaded_joint",
        "handwheel_valve",
        "port_plug",
        "panel_rail",
        "tile_bed",
        "spring_button",
        "sliding_door",
    ):
        assert get_template_builder(kind) is get_template_builder(kind)


# --------------------------------------------------------------------------- Studio v2: four job templates
#
# panel_rail (solar install), tile_bed (tile setting), spring_button (a call button) and
# sliding_door (machine tending). Each builds, rests and reports its quantities (the parametrized
# test above); here: their params and physics, then the S2 rule on each (a fitting tool succeeds
# where a wrong one fails), run in spawned workers (at most four).

V2_KINDS = ("panel_rail", "tile_bed", "spring_button", "sliding_door")


def _press_body(
    scene: Any, body: int, force: np.ndarray, seconds: float, d: mujoco.MjData | None = None
) -> mujoco.MjData:
    """Run the scene (wrist parked high, passive actuators driven) with ``force`` (world, N) on
    ``body``'s centre of mass, ramped in over 0.1 s; continue from ``d`` when given."""
    m = scene.model
    if d is None:
        d = mujoco.MjData(m)
        d.qpos[scene.ids.palm_qpos] = [0.0, 0.0, 0.45, 0.0]
        mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = [0.0, 0.0, 0.45, 0.0]
    ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
    t0 = float(d.time)
    for _ in range(int(seconds / 0.002)):
        for obj in scene.objects.values():
            if len(obj.actuator_ids):
                ctrl[obj.actuator_ids] = scene.builders[obj.name].passive(obj, m, d)
        d.ctrl[:] = ctrl
        d.xfrc_applied[body, :3] = force * min(1.0, (float(d.time) - t0) / 0.1)
        mujoco.mj_step(m, d, nstep=2)
    d.xfrc_applied[body, :3] = 0.0
    return d


def test_the_job_templates_are_registered_with_their_quantities() -> None:
    from rtt.studio.types import QUANTITIES

    want = {
        "panel_rail": {"gap_mm", "clamp_torque_nm"},
        "tile_bed": {"height_err_mm", "tilt_deg"},
        "spring_button": {"travel_mm", "pressed"},
        "sliding_door": {"travel_mm"},
    }
    for kind, quantities in want.items():
        t = get_template(kind)
        assert quantities <= set(t.quantities), kind
        assert all(q in QUANTITIES for q in t.quantities)
        assert t.fidelity.value in ("rigid", "approximation") and (t.fidelity.value == "rigid" or t.caveat)
    assert get_template("spring_button").defaults()["mount"] == "wall"
    assert set(get_template("spring_button").params[0].choices or ()) | {"wall", "desk"} == {"wall", "desk"}


def test_panel_rail_panel_rails_stop_and_clamp(task: Any) -> None:
    from rtt.effectors.process_tool import MIN_MEMBER_MM

    spec = one_object_task("pr", kind="panel_rail")
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["pr"]
    info = obj.handles.info
    panel = obj.body_ids[-1]
    assert m.body(panel).name == "obj_pr_panel"  # the picks lift the last body
    assert float(m.body_mass[panel]) == pytest.approx(8.0)
    assert 2.0 * m.geom_size[m.geom("obj_pr_panel").id] == pytest.approx([1.0, 0.6, 0.035])
    # the clamp: a 13 mm across-flats hex, 15 mm tall [assumed], on an M8 coarse thread
    assert 2.0 * info["clamp_flat_m"] == pytest.approx(0.013, abs=1e-6)
    assert info["clamp_length_m"] == pytest.approx(0.015) and 1000.0 * info["clamp_length_m"] >= MIN_MEMBER_MM
    assert info["clamp_pitch_m"] == pytest.approx(0.00125) and info["clamp_final_nm"] == pytest.approx(6.0)
    builder = scene.builders["pr"]
    rest = builder.state(obj, m, settle(scene, 0.4))
    assert rest["gap_mm"] == pytest.approx(2.0, abs=0.3) and rest["clamp_torque_nm"] == 0.0
    # pushed toward its stops (local +y) the panel slides onto them and the gap closes
    mu = float(task.physics.friction)
    force = np.array([0.0, 2.0 * mu * 8.0 * 9.81, 0.0])
    pushed = builder.state(obj, m, _press_body(scene, panel, force, 0.6))
    assert pushed["gap_mm"] < 0.3
    # the clamp's torque: the thread's ramp at the deepest point it reached (the latch's): 0 on the
    # lead-in, the start torque at the thread start, its rated torque on the seat stop
    seat = info["clamp_lead_m"] + info["clamp_seat_m"]
    d = mujoco.MjData(m)
    for deepest, want in (
        (0.5 * info["clamp_lead_m"], 0.0),
        (info["clamp_lead_m"], info["clamp_start_nm"]),
        (info["clamp_lead_m"] + 0.5 * info["clamp_seat_m"], 0.5 * (info["clamp_start_nm"] + 6.0)),
        (seat, 6.0),
    ):
        d.qpos[obj.joint_qpos["slide"]] = d.qpos[obj.joint_qpos["latch"]] = deepest
        mujoco.mj_forward(m, d)
        assert builder.state(obj, m, d)["clamp_torque_nm"] == pytest.approx(want, abs=1e-6), deepest
    d.qpos[obj.joint_qpos["slide"]] = seat - 0.00005  # sprung back off its stop when the socket let go
    mujoco.mj_forward(m, d)
    assert builder.state(obj, m, d)["clamp_torque_nm"] == 6.0  # it was tightened to its torque
    # the latch follows the clamp down (a push down the lead-in to the thread start) and stays
    slide = int(obj.joint_dof["slide"])
    pushed = settle(scene, 0.6, applied={slide: 0.5})
    lag = float(pushed.qpos[obj.joint_qpos["latch"]] - pushed.qpos[obj.joint_qpos["slide"]])
    assert -2e-5 <= lag <= 1e-4, lag  # never behind the clamp, at most 0.1 mm past where it stopped
    assert pushed.qpos[obj.joint_qpos["slide"]] == pytest.approx(info["clamp_lead_m"], abs=1e-4)
    assert builder.state(obj, m, pushed)["clamp_torque_nm"] == pytest.approx(info["clamp_start_nm"], abs=0.2)
    # a custom clamp torque is the thread's final torque
    other = scene_of(one_object_task("p2", kind="panel_rail", clamp_torque_nm=4.0), task)
    assert other.objects["p2"].handles.info["clamp_final_nm"] == pytest.approx(4.0)


def _signed_height(scene: Any, d: mujoco.MjData, name: str) -> float:
    obj = scene.objects[name]
    info = obj.handles.info
    tile = obj.body_ids[-1]
    top = d.xpos[tile] + d.xmat[tile].reshape(3, 3) @ np.array([0.0, 0.0, 0.5 * info["tile_t_m"]])
    return 1000.0 * (float(top[2]) - info["level_top_z"])


def test_tile_bed_keeps_its_set_and_a_harder_press_seats_it_deeper(task: Any) -> None:
    """The bed is plastic with hardening: under a press the tile sinks until the bed's yield
    (rising with the sink) carries the press and the tile's weight; let go, it keeps its set
    (no elastic rebound to hide the press). A light press leaves the tile proud, the tamper's
    rated press seats it within 1 mm of its level."""
    spec = one_object_task("tb", kind="tile_bed")
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["tb"]
    info = obj.handles.info
    tile = obj.body_ids[-1]
    assert m.body(tile).name == "obj_tb_tile"
    assert float(m.body_mass[tile]) == pytest.approx(2.16)
    assert 2.0 * m.geom_size[m.geom("obj_tb_tile").id] == pytest.approx([0.3, 0.3, 0.010])
    builder = scene.builders["tb"]
    loaded, after = {}, {}
    for press in (0.0, 15.0, 40.0, 60.0):
        d = _press_body(scene, tile, np.array([0.0, 0.0, -press]), 1.0)
        loaded[press] = _signed_height(scene, d, "tb")
        d = _press_body(scene, tile, np.zeros(3), 0.5, d)
        after[press] = _signed_height(scene, d, "tb")
        st = builder.state(obj, m, d)
        assert st["height_err_mm"] == pytest.approx(abs(after[press]), abs=1e-6)
        assert st["tilt_deg"] < 0.2, press  # a centred press sinks it level
        assert abs(after[press] - loaded[press]) <= 0.35, press  # it keeps its set (elastic give only)
    assert after[0.0] == pytest.approx(info["proud_m"] * 1000.0, abs=0.25)  # it rests proud
    assert after[0.0] > after[15.0] > after[40.0] > after[60.0]  # a harder press seats it deeper
    assert abs(after[15.0]) > 1.0  # an under-pressed tile stays proud of its level
    assert abs(after[40.0]) <= 1.0  # the tamper's rated press seats it


@pytest.mark.parametrize("mount", ["wall", "desk"])
def test_spring_button_threshold_travel_and_latched_press(mount: str, task: Any) -> None:
    spec = one_object_task("bt", kind="spring_button", mount=mount, actuation_force_n=5.0, travel_mm=3.0)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["bt"]
    info = obj.handles.info
    normal = np.array([info["normal_x"], info["normal_y"], info["normal_z"]])
    assert normal == pytest.approx([0.0, -1.0, 0.0] if mount == "wall" else [0.0, 0.0, 1.0])
    dof = int(obj.joint_dof["cap"])
    assert m.joint("obj_bt_cap").range[1] == pytest.approx(0.003)
    builder = scene.builders["bt"]
    light = builder.state(obj, m, settle(scene, 0.5, applied={dof: 0.6 * 5.0}))
    assert light["pressed"] == 0.0 and light["travel_mm"] < 1000.0 * info["actuate_m"]
    d = _press_body(scene, obj.body_ids[1], -1.5 * 5.0 * normal, 0.5)
    hard = builder.state(obj, m, d)
    assert hard["pressed"] == 1.0 and hard["travel_mm"] == pytest.approx(3.0, abs=0.1)
    released = builder.state(obj, m, _press_body(scene, obj.body_ids[1], np.zeros(3), 0.5, d))
    assert (
        released["travel_mm"] < 0.2 and released["pressed"] == 1.0
    )  # it springs back; the call stays latched


def test_sliding_door_slides_sideways_with_friction_and_covers_the_opening(task: Any) -> None:
    spec = one_object_task("sd", kind="sliding_door", travel_mm=240, pull_force_n=10)
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["sd"]
    info = obj.handles.info
    j = m.joint("obj_sd_slide")
    assert j.range[1] == pytest.approx(0.240) and m.dof_frictionloss[j.dofadr[0]] == pytest.approx(10.0)
    assert [info["pull_x"], info["pull_y"]] == pytest.approx([0.0, -1.0])  # sideways along local -y
    for force, moves in ((6.0, False), (20.0, True)):
        d = settle(scene, 0.5, applied={int(j.dofadr[0]): force})
        assert (scene.builders["sd"].state(obj, m, d)["travel_mm"] > 5.0) is moves, force
    # closed, the door covers the roof opening; open, a straight drop through the opening is clear
    door_geoms = {int(g) for g in obj.geom_ids if m.geom_bodyid[g] == obj.body_ids[-1]}
    for travel, blocked in ((0.0, True), (0.240, False)):
        d = mujoco.MjData(m)
        d.qpos[obj.joint_qpos["slide"]] = travel
        mujoco.mj_forward(m, d)
        top = obj.site_pos(d, "opening") + np.array([0.0, 0.0, 0.5])
        hit = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, top, np.array([0.0, 0.0, -1.0]), None, True, -1, hit)
        assert (int(hit[0]) in door_geoms) is blocked, (travel, int(hit[0]), dist)
        if not blocked:
            assert int(hit[0]) not in {int(g) for g in obj.geom_ids}  # nothing of the machine in the way


@pytest.mark.parametrize("variant", ["panel_rail", "sliding_door"])
def test_v2_observed_state_matches_the_truth_and_ignores_the_aim_offset(variant: str, task: Any) -> None:
    spec = one_object_task("thing", **VARIANTS[variant])
    scene = scene_of(spec, task)
    m, obj = scene.model, scene.objects["thing"]
    d = mujoco.MjData(m)
    info = obj.handles.info
    if variant == "panel_rail":
        d.qpos[obj.joint_qpos["slide"]] = info["clamp_lead_m"] + 0.0036
        d.qpos[obj.joint_qpos["hinge"]] = 2.0 * math.pi * 0.0036 / info["clamp_pitch_m"]
    else:
        d.qpos[obj.joint_qpos["slide"]] = 0.064
    mujoco.mj_forward(m, d)
    truth = scene.builders["thing"].state(obj, m, d)
    for offset in ((0.0, 0.0, 0.0), (0.004, -0.003, 0.0)):
        seen = observed_state(ctx_for(scene, spec, task, offset), d)
        assert seen, variant
        for q in seen:
            assert seen[q] == pytest.approx(truth[q], abs=0.05), (variant, q, offset)


# --------------------------------------------------------------------------- the step param ``approach``


def test_push_approach_is_an_optional_param_omitted_at_its_default() -> None:
    """``approach`` on a push is optional: "top" (the default) is omitted from the step, so every
    step written before it hashes as it did; "front" is kept; anything else is refused. On a
    hook_pull it stays the required param it has always been (in every recorded plan)."""
    crit = [SuccessCriterion("force_n", at_most=40.0)]
    plain = Step.of("p", "push", "carton", "part", "Press it", crit)
    top = Step.of("p", "push", "carton", "part", "Press it", crit, params={"approach": "top"})
    front = Step.of("p", "push", "carton", "part", "Press it", crit, params={"approach": "front"})
    assert dict(plain.params) == {"speed_mm_s": 20.0} and plain.canonical_dict() == top.canonical_dict()
    assert front.params["approach"] == "front" and front.canonical_dict() != plain.canonical_dict()
    with pytest.raises(ValueError, match="approach"):
        Step.of("p", "push", "carton", "part", "Press it", crit, params={"approach": "side"})
    pull = Step.of("o", "hook_pull", "lid", "handle", "Open it", crit)
    assert pull.params["approach"] == "top"  # hook_pull: the required row, unchanged
    with pytest.raises(ValueError, match="missing param 'approach'"):
        Step("o", PrimitiveKind.HOOK_PULL, "lid", "handle", "Open it", {"speed_mm_s": 50.0}, tuple(crit))
    assert [p.name for p in PRIMITIVES[PrimitiveKind.PUSH].params] == ["speed_mm_s"]  # the planner's row


def test_frozen_hero_task_ids_are_unchanged() -> None:
    """The three heroes frozen by use-case benchmark v1 (``configs/usecase_v1.yaml``, a PREREG-frozen
    file) re-hash to their pinned ids from the recorded plans after the contract additions."""
    import yaml

    from rtt.studio.engineer.gate import RECORDED_GATE_V1
    from rtt.studio.engineer.webdata import recorded_plans
    from rtt.studio.types import task_spec_from_tool_args

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "usecase_v1.yaml").read_text())
    pinned: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and isinstance(value.get("task_id"), str) and key.startswith("h"):
                    pinned[key] = value["task_id"]
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)
    assert set(pinned) == {"h1_tote", "h2_coupling", "h3_cabinet"}, pinned
    examples = {e["case_id"]: e for e in recorded_plans(RECORDED_GATE_V1)["examples"]}
    for case_id, task_id in pinned.items():
        plan = examples[case_id]["plan"]
        keys = ("title", "description", "objects", "steps", "assumptions", "clarifications", "unsupported")
        spec = task_spec_from_tool_args({**{k: plan[k] for k in keys}, "refusal": plan["refusal"]})
        assert spec.task_id == task_id, case_id


def test_the_new_kinds_are_offered_to_the_task_engineer() -> None:
    """Studio v2 taught the Task Engineer the four job templates, their quantities and
    handles, and push's optional ``approach`` (its prompt, tool and web catalog are recorded and
    hashed: the v2 gate re-recorded them)."""
    from rtt.studio.engineer.catalog import (
        render_primitives,
        render_quantities,
        render_templates,
        web_catalog,
    )
    from rtt.studio.types import planner_templates, task_plan_json_schema

    offered = {t.kind.value for t in planner_templates()}
    assert set(V2_KINDS) <= offered and len(offered) == 12
    text = (
        render_templates() + render_primitives() + render_quantities() + json.dumps(task_plan_json_schema())
    )
    for word in (
        *V2_KINDS,
        "gap_mm",
        "clamp_torque_nm",
        "height_err_mm",
        "| pressed |",
        '"pressed"',
        '"panel"',
        '"button"',
        '"approach"',
    ):
        assert word in text, word
    catalog = web_catalog()
    assert {t["kind"] for t in catalog["templates"]} == offered
    assert all(set(p["objects"]) <= offered for p in catalog["primitives"])
    push = next(p for p in catalog["primitives"] if p["kind"] == "push")
    assert [(q["name"], q["optional"]) for q in push["params"]] == [("speed_mm_s", False), ("approach", True)]


# --------------------------------------------------------------------------- the S2 rule on each


def _one(name: str, objects: tuple[ObjectInstance, ...], *steps: Step) -> TaskSpec:
    return TaskSpec(name, "", objects, steps, source="test")


def _runs(cases: dict[str, tuple[Any, TaskSpec]], condition: Any = None) -> dict[str, Any]:
    """Run each (tool, task) case on the nominal condition (or ``condition``) in spawned workers."""
    from rtt.studio.episode import StudioCondition, StudioJob, run_task_jobs

    condition = condition or StudioCondition("nominal", 0)
    jobs = [StudioJob(Setup.single(tool, t.step_ids), t, condition) for tool, t in cases.values()]
    return dict(zip(cases, run_task_jobs(jobs, workers=min(4, len(jobs))), strict=True))


def _why(result: Any) -> list[Any]:
    return [(s.step_id, s.success, s.failure_mode, dict(s.quantities)) for s in result.steps]


def _clamp_steps() -> tuple[Step, Step]:
    start = Step.of(
        "start", "insert", "panel", "member", "Start the clamp", [SuccessCriterion("depth_mm", at_least=3.5)]
    )
    tighten = Step.of(
        "tighten",
        "twist",
        "panel",
        "member",
        "Torque the clamp",
        [SuccessCriterion("clamp_torque_nm", at_least=6.0)],
        params={"direction": "cw", "speed_deg_s": 180.0},
        constraints=StepConstraints(max_torque_nm=15.0),
    )
    return start, tighten


def panel_cases() -> dict[str, tuple[Any, TaskSpec]]:
    panel = ObjectInstance.of("panel", "panel_rail", "Solar panel on its rails", ObjectPose(0.0, 0.0, 0.0))
    light = ObjectInstance.of("panel", "panel_rail", "Light panel", ObjectPose(0.0, 0.0, 0.0), mass_g=2000.0)
    held = [
        SuccessCriterion("lift_mm", at_least=25.0),
        SuccessCriterion("hold_s", at_least=1.0),
        SuccessCriterion("gap_mm", at_most=10.0),
    ]
    lift = {"lift_mm": 60.0, "hold_s": 1.5}
    frame = Step.of("lift", "suction", "panel", "panel", "Lift the panel, set it back", held, params=lift)
    jaw = Step.of("lift", "grasp", "panel", "panel", "Lift the panel, set it back", held, params=lift)
    start, tighten = _clamp_steps()
    small = _preset("suction", "vacuum-frame-2x2-300")  # 61 N: rated 4.15 kg
    return {
        "frame": (_preset("suction", "vacuum-frame-2x3-600"), _one("panel lift", (panel,), frame)),
        "small_frame_2kg": (small, _one("panel lift", (light,), frame)),
        "small_frame_8kg": (small, _one("panel lift", (panel,), frame)),
        "jaw": (
            _preset("catalog_gripper", "gripper-2f-hobby-parallel-84"),
            _one("panel lift", (panel,), jaw),
        ),
        "nut_runner": (
            _preset("process_tool", "nut-runner-socket-13"),
            _one("clamp", (panel,), start, tighten),
        ),
        "gripper": (gripper(), _one("clamp", (panel,), start, tighten)),
    }


@pytest.fixture(scope="module")
def panel_runs() -> dict[str, Any]:
    return _runs(panel_cases())


def _preset(family: str, model: str) -> Any:
    """A Studio candidate tool by family and model (catalog presets and library entrants)."""
    from rtt.studio.search import default_candidates

    return next(
        c.spec for c in default_candidates() if c.spec.family.value == family and c.spec.model == model
    )


def test_s2_a_vacuum_frame_holds_the_8_kg_panel_and_a_parallel_gripper_cannot(
    panel_runs: dict[str, Any],
) -> None:
    frame = panel_runs["frame"]
    assert frame.completed, _why(frame)
    q = frame.steps[0].quantities
    assert q["lift_mm"] >= 25.0 and q["hold_s"] >= 1.0 and q["gap_mm"] <= 10.0 and q["penetration_mm"] <= 1.0
    jaw = panel_runs["jaw"].steps[0]
    assert not jaw.success and jaw.quantities["lift_mm"] < 5.0, _why(panel_runs["jaw"])
    # the wrong tool fails on the mass: the small frame (61 N) lifts a 2 kg panel, not the 8 kg one
    assert panel_runs["small_frame_2kg"].completed, _why(panel_runs["small_frame_2kg"])
    heavy = panel_runs["small_frame_8kg"].steps[0]
    assert not heavy.success and heavy.quantities["lift_mm"] < 1.0, _why(panel_runs["small_frame_8kg"])


def test_s2_the_nut_runner_torques_the_clamp_and_a_gripper_twist_cannot(panel_runs: dict[str, Any]) -> None:
    nut = panel_runs["nut_runner"]
    assert nut.completed, _why(nut)
    q = nut.steps[1].quantities
    # seated on its stop: the clamp holds its 6 N m; the wrist turned it against the ramp to there
    assert q["clamp_torque_nm"] >= 6.0 and q["depth_mm"] == pytest.approx(7.75, abs=0.01)
    assert q["torque_nm"] >= 0.95 * 6.0 and q["penetration_mm"] <= 1.0
    tighten = panel_runs["gripper"].steps[-1]
    assert not tighten.success and tighten.quantities.get("clamp_torque_nm", 0.0) < 6.0, _why(
        panel_runs["gripper"]
    )


def tile_cases() -> dict[str, tuple[Any, TaskSpec]]:
    tile = ObjectInstance.of("tile", "tile_bed", "Floor tile on its bed", ObjectPose(0.0, 0.0, 0.0))

    def seat(force: float) -> Step:
        return Step.of(
            "seat",
            "push",
            "tile",
            "part",
            "Press the tile to its level",
            [
                SuccessCriterion("force_n", at_least=force),
                SuccessCriterion("height_err_mm", at_most=1.0),
                SuccessCriterion("tilt_deg", at_most=0.5),
            ],
            constraints=StepConstraints(max_force_n=60.0),
        )

    lay = Step.of(
        "lay",
        "suction",
        "tile",
        "part",
        "Lay the tile on its bed",
        [SuccessCriterion("lift_mm", at_least=20.0), SuccessCriterion("hold_s", at_least=0.5)],
        params={"lift_mm": 40.0, "hold_s": 0.8},
    )
    tamper = _preset("process_tool", "tamper-pad-200")
    return {
        "rated": (tamper, _one("tile", (tile,), seat(40.0))),
        "light": (tamper, _one("tile", (tile,), seat(10.0))),
        "lay": (_preset("suction", "vacuum-frame-2x2-300"), _one("tile", (tile,), lay)),
    }


@pytest.fixture(scope="module")
def tile_runs() -> dict[str, Any]:
    return _runs(tile_cases())


def test_s2_the_tamper_seats_the_tile_and_a_light_press_leaves_it_proud(tile_runs: dict[str, Any]) -> None:
    rated = tile_runs["rated"]
    assert rated.completed, _why(rated)
    q = rated.steps[0].quantities
    assert q["height_err_mm"] <= 1.0 and q["tilt_deg"] <= 0.5 and q["force_n"] >= 40.0
    light = tile_runs["light"].steps[0]
    assert not light.success and light.quantities["height_err_mm"] > 1.0, _why(tile_runs["light"])
    lay = tile_runs["lay"]  # the frame lifts the 2.16 kg tile off its bed and lays it back
    assert lay.completed, _why(lay)


def button_cases() -> dict[str, tuple[Any, TaskSpec]]:
    wall = ObjectInstance.of("call", "spring_button", "Call button", ObjectPose(0.0, 220.0, 0.0))
    desk = ObjectInstance.of(
        "call", "spring_button", "Call button", ObjectPose(0.0, 100.0, 0.0), mount="desk"
    )
    crit = [SuccessCriterion("pressed", at_least=1.0)]
    front = Step.of(
        "press",
        "push",
        "call",
        "button",
        "Press the call button",
        crit,
        params={"approach": "front"},
        constraints=StepConstraints(max_force_n=20.0),
    )
    top = Step.of(
        "press",
        "push",
        "call",
        "button",
        "Press the call button",
        crit,
        constraints=StepConstraints(max_force_n=20.0),
    )
    stiff = ObjectInstance.of(
        "call", "spring_button", "Stiff button", ObjectPose(0.0, 220.0, 0.0), actuation_force_n=15.0
    )
    too_stiff = ObjectInstance.of(
        "call", "spring_button", "Stiff button", ObjectPose(0.0, 220.0, 0.0), actuation_force_n=25.0
    )
    at_30 = Step.of(
        "press",
        "push",
        "call",
        "button",
        "Press the call button",
        crit,
        params={"approach": "front"},
        constraints=StepConstraints(max_force_n=30.0),
    )

    def desk_at(force: float, limit: float) -> TaskSpec:
        button = ObjectInstance.of(
            "call",
            "spring_button",
            "Desk button",
            ObjectPose(0.0, 100.0, 0.0),
            mount="desk",
            actuation_force_n=force,
        )
        step = Step.of(
            "press",
            "push",
            "call",
            "button",
            "Press the desk button",
            crit,
            constraints=StepConstraints(max_force_n=limit),
        )
        return _one("call", (button,), step)

    return {
        "front": (gripper(), _one("call", (wall,), front)),
        "tamper": (_preset("process_tool", "tamper-pad-200"), _one("call", (wall,), front)),
        "desk": (gripper(), _one("call", (desk,), top)),
        "stiff": (gripper(), _one("call", (stiff,), front)),  # 15 N, at a 20 N limit, no force goal
        "too_stiff": (gripper(), _one("call", (too_stiff,), at_30)),  # 25 N at a 30 N limit: 24 N allowed
        "desk_at_cap": (gripper(), desk_at(8.0, 10.0)),  # an 8 N cap on an 8 N button, from the top
        "desk_below": (gripper(), desk_at(15.0, 18.0)),  # a 14.4 N cap on a 15 N button
    }


@pytest.fixture(scope="module")
def button_runs() -> dict[str, Any]:
    return _runs(button_cases())


def test_s2_a_front_push_presses_the_wall_button_and_a_broad_pad_cannot(button_runs: dict[str, Any]) -> None:
    front = button_runs["front"]
    assert front.completed, _why(front)
    assert front.steps[0].quantities["pressed"] == 1.0 and front.steps[0].quantities["penetration_mm"] <= 1.0
    pad = button_runs["tamper"].steps[0]
    assert not pad.success and pad.quantities["pressed"] == 0.0, _why(button_runs["tamper"])
    desk = button_runs["desk"]
    assert desk.completed, _why(desk)
    stiff = button_runs["stiff"]  # pressed at 1.5 x its 15 N within the step's limit (80 % of 20 N)
    assert stiff.completed and stiff.steps[0].quantities["force_n"] <= 20.0, _why(stiff)


def test_a_button_the_force_limit_cannot_actuate_fails_fast_and_says_why(button_runs: dict[str, Any]) -> None:
    """A 25 N button at a 30 N limit: the press may push with 24 N (80 % of the limit),
    under the 25 N that actuates the button. It used to press for 8 s and fail silently; it is not
    attempted and its outcome says why. The same holds for a 15 N desk button at an 18 N limit
    (a 14.4 N cap)."""
    out = button_runs["too_stiff"].steps[0]
    assert not out.success and out.failure_mode == "criterion_not_met", _why(button_runs["too_stiff"])
    assert out.duration_s <= 0.25 and out.quantities["pressed"] == 0.0
    assert out.quantities["press_cap_n"] == pytest.approx(24.0)
    assert out.quantities["actuation_force_n"] == pytest.approx(25.0)
    why = [c for c in out.caveats if "force limit allows 24 N" in c]
    assert why and "this button needs at least 25 N to actuate" in why[0], out.caveats
    desk = button_runs["desk_below"].steps[0]
    assert not desk.success and desk.duration_s <= 0.25, _why(button_runs["desk_below"])
    assert [c for c in desk.caveats if "allows 14.4 N; this button needs at least 15 N" in c], desk.caveats


def test_a_button_whose_cap_reaches_its_actuation_force_is_pressed(button_runs: dict[str, Any]) -> None:
    """The refusal is only for a cap under the actuation force. Pressed from the
    top, a desk button latches with less than the spring force at the latch point, and the press goes
    0.4 to 0.5 N over its cap: an 8 N desk button at a 10 N limit (an 8 N cap) latches at about
    8.4 N, as on base, so it is tried and pressed."""
    out = button_runs["desk_at_cap"].steps[0]
    assert out.success and out.quantities["pressed"] == 1.0, _why(button_runs["desk_at_cap"])
    assert out.duration_s > 1.0 and out.quantities["force_n"] <= 10.0


def door_cases() -> dict[str, tuple[Any, TaskSpec]]:
    door = ObjectInstance.of("door", "sliding_door", "Machine door", ObjectPose(0.0, 100.0, 0.0))
    open_ = Step.of(
        "open",
        "hook_pull",
        "door",
        "handle",
        "Slide the door open",
        [SuccessCriterion("travel_mm", at_least=200.0)],
    )
    shut = Step.of(
        "shut",
        "push",
        "door",
        "door",
        "Slide the door shut",
        [SuccessCriterion("travel_mm", at_most=5.0)],
        constraints=StepConstraints(max_force_n=40.0),
    )
    return {
        "gripper": (gripper(), _one("door", (door,), open_, shut)),
        "cup": (_preset("suction", "cup-25mm"), _one("door", (door,), open_)),
    }


@pytest.fixture(scope="module")
def door_runs() -> dict[str, Any]:
    return _runs(door_cases())


def test_s2_the_gripper_opens_and_shuts_the_sliding_door_and_a_cup_cannot(door_runs: dict[str, Any]) -> None:
    run = door_runs["gripper"]
    assert run.completed, _why(run)
    assert run.steps[0].quantities["travel_mm"] >= 200.0 and run.steps[1].quantities["travel_mm"] <= 5.0
    cup = door_runs["cup"].steps[0]
    assert not cup.success and cup.quantities["travel_mm"] < 200.0, _why(door_runs["cup"])


def machine_cases() -> dict[str, tuple[Any, TaskSpec]]:
    """Machine tending: open the door, load a part into the fixture under it, shut the door (the
    door's default 10 N of friction); and the same task without the load."""
    door = ObjectInstance.of("door", "sliding_door", "Machine door", ObjectPose(0.0, 100.0, 0.0))
    fixture = ObjectInstance.of(
        "fixture", "bin", "Fixture", ObjectPose(0.0, 100.0, 0.0), length_mm=150, width_mm=150, height_mm=60
    )
    part = ObjectInstance.of(
        "part",
        "part",
        "Part",
        ObjectPose(200.0, 100.0, 0.0),
        size_x_mm=40,
        size_y_mm=40,
        size_z_mm=40,
        mass_g=100,
    )
    open_, shut = door_cases()["gripper"][1].steps
    load = Step.of(
        "load",
        "grasp",
        "part",
        "part",
        "Load the part",
        [SuccessCriterion("in_target", at_least=1)],
        target="fixture",
    )
    objects = (door, fixture, part)
    return {
        "load": (gripper(), _one("machine", objects, open_, load, shut)),
        "no_load": (gripper(), _one("machine", objects, open_, shut)),
    }


@pytest.fixture(scope="module")
def machine_runs() -> dict[str, Any]:
    return _runs(machine_cases())


def test_a_door_shut_after_loading_the_fixture_closes_within_5_mm(machine_runs: dict[str, Any]) -> None:
    """After the pick lets go fully open, the gripper's fist settles differently and
    swings on contact with the door (a finger to its stop), using about 43 mm of the push. The push
    used to go only its 18 mm ``over`` past its path while the door was still open, then let go 40 mm
    open. A push to an observed-state criterion now goes on past that while the state is unmet and
    the force is under the step's limit (at most :data:`~rtt.studio.primitives.push.PUSH_ON_MAX`
    more, within its time budget)."""
    run = machine_runs["load"]
    assert run.completed, _why(run)
    shut = run.steps[2].quantities
    assert shut["travel_mm"] <= 5.0 and shut["force_n"] <= 40.0, _why(run)
    plain = machine_runs["no_load"]
    assert plain.completed and plain.steps[1].quantities["travel_mm"] <= 5.0, _why(plain)


# --------------------------------------------------------------------------- reach

#: Panel poses (x, y mm, yaw deg): the review's failing ones, then corners of the reach envelope
#: (+-182 mm at any yaw) turned so the clamp or the panel's centre lands at its extreme
REACH_POSES: dict[str, tuple[float, float, float]] = {
    "y_minus_60": (0.0, -60.0, 0.0),
    "x_plus_130": (130.0, 0.0, 0.0),
    "yaw_plus_30": (0.0, 0.0, 30.0),
    "yaw_minus_30": (0.0, 0.0, -30.0),
    "corner_clamp_minus_y": (180.0, -180.0, 0.0),
    "corner_clamp_plus_y": (-180.0, 180.0, 180.0),
    "corner_clamp_plus_x": (180.0, 180.0, 90.0),
    "corner_clamp_minus_x": (-180.0, -180.0, -90.0),
}


def _panel_at(x: float, y: float, yaw: float) -> ObjectInstance:
    return ObjectInstance.of("panel", "panel_rail", "Solar panel on its rails", ObjectPose(x, y, yaw))


def reach_cases() -> dict[str, tuple[Any, TaskSpec]]:
    nut = _preset("process_tool", "nut-runner-socket-13")
    start, tighten = _clamp_steps()
    out = {k: (nut, _one("clamp", (_panel_at(*pose),), start, tighten)) for k, pose in REACH_POSES.items()}
    lift = Step.of(
        "lift",
        "suction",
        "panel",
        "panel",
        "Lift the panel, set it back",
        [SuccessCriterion("lift_mm", at_least=25.0), SuccessCriterion("hold_s", at_least=1.0)],
        params={"lift_mm": 60.0, "hold_s": 1.5},
    )
    frame = _preset("suction", "vacuum-frame-2x3-600")
    for k, pose in (
        ("lift_corner_plus_y", (180.0, 180.0, 0.0)),
        ("lift_corner_minus_x", (-180.0, 180.0, 90.0)),
    ):
        out[k] = (frame, _one("panel lift", (_panel_at(*pose),), lift))
    # out of reach: the clamp 162.5 mm beyond a pose 300 mm out puts it at -462 mm (the palm: 350)
    out["far_insert"] = (nut, _one("clamp", (_panel_at(0.0, -300.0, 0.0),), start, tighten))
    out["far_twist"] = (nut, _one("clamp", (_panel_at(0.0, -300.0, 0.0),), tighten))
    return out


@pytest.fixture(scope="module")
def reach_runs() -> dict[str, Any]:
    return _runs(reach_cases())


def test_the_panels_centre_and_its_clamp_sit_either_side_of_the_pose(task: Any) -> None:
    from rtt.sim.scene import PALM_XY_LIMIT
    from rtt.studio.templates.panel_rail import reach_of

    for pose in ((0.0, 0.0, 0.0), (120.0, -90.0, 35.0)):
        spec = one_object_task("pr", pose=ObjectPose(*pose), kind="panel_rail")
        scene = scene_of(spec, task)
        obj = scene.objects["pr"]
        d = mujoco.MjData(scene.model)
        mujoco.mj_forward(scene.model, d)
        base = np.array([pose[0], pose[1]]) / 1000.0
        reach = obj.handles.info["reach_m"]
        assert reach == pytest.approx(reach_of(0.6)) == pytest.approx(0.1625)
        for handle in ("panel", "member"):
            assert float(np.linalg.norm(obj.site_pos(d, handle)[:2] - base)) == pytest.approx(reach, abs=1e-6)
        # the envelope: both work points inside the palm's travel for any pose within it at any yaw
        assert PALM_XY_LIMIT - 0.005 - reach == pytest.approx(0.1825)


def test_the_nut_runner_seats_the_clamp_across_the_reach_envelope(reach_runs: dict[str, Any]) -> None:
    """The review's failing poses (y -60, x +130, yaw +-30 deg) and the envelope's corners."""
    for key in REACH_POSES:
        run = reach_runs[key]
        assert run.completed, (key, _why(run))
        assert run.steps[1].quantities["clamp_torque_nm"] >= 6.0, key
    for key in ("lift_corner_plus_y", "lift_corner_minus_x"):  # the frame reaches the panel's centre there
        assert reach_runs[key].completed, (key, _why(reach_runs[key]))


def test_a_clamp_out_of_reach_fails_as_out_of_reach_not_slip_or_jam(reach_runs: dict[str, Any]) -> None:
    for key in ("far_insert", "far_twist"):
        first = reach_runs[key].steps[0]
        assert not first.success and first.failure_mode == "out_of_reach", (key, _why(reach_runs[key]))
        q = first.quantities
        assert q["reach_y_mm"] == pytest.approx(-462.5, abs=3.0) and q["palm_limit_mm"] == pytest.approx(
            350.0
        )
        assert first.duration_s <= 0.25  # not attempted: no slip, no jam, no time spent
        assert [c for c in first.caveats if c.startswith("Out of reach:")], first.caveats


def far_cases() -> dict[str, tuple[Any, TaskSpec]]:
    """Targets past the palm's travel for the picks and the pulls: a panel 300 mm out
    has its centre 462.5 mm out (lifted by a frame, or gripped); a door at y 300 mm turned to open
    toward +y is pulled 215 mm further out."""
    lift = Step.of(
        "lift",
        "suction",
        "panel",
        "panel",
        "Lift the panel, set it back",
        [SuccessCriterion("lift_mm", at_least=25.0), SuccessCriterion("hold_s", at_least=1.0)],
        params={"lift_mm": 60.0, "hold_s": 1.5},
    )
    grip = Step.of(
        "lift", "grasp", "panel", "panel", "Grip the panel", [SuccessCriterion("lift_mm", at_least=25.0)]
    )
    door = ObjectInstance.of("door", "sliding_door", "Machine door", ObjectPose(0.0, 300.0, 180.0))
    open_ = door_cases()["gripper"][1].steps[0]
    return {
        "far_lift": (
            _preset("suction", "vacuum-frame-2x3-600"),
            _one("lift", (_panel_at(0.0, 300.0, 0.0),), lift),
        ),
        "far_grasp": (gripper(), _one("grip", (_panel_at(0.0, 300.0, 0.0),), grip)),
        "far_pull": (gripper(), _one("door", (door,), open_)),
    }


@pytest.fixture(scope="module")
def far_runs() -> dict[str, Any]:
    return _runs(far_cases())


def test_a_pick_or_a_pull_out_of_reach_fails_as_out_of_reach_and_says_why(far_runs: dict[str, Any]) -> None:
    """Only twist and insert used to check their targets against the palm's travel; a
    pick, a hook pull or a push whose target the palm cannot reach (+-350 mm less 5 mm) failed as a
    silent ``criterion_not_met`` (or slipped) after running in vain. Each now fails fast as
    ``out_of_reach``, where the palm would have to go and a plain reason on the outcome."""
    for key, axis, at_least in (
        ("far_lift", "y", 455.0),
        ("far_grasp", "y", 455.0),
        ("far_pull", "y", 345.0),
    ):
        out = far_runs[key].steps[0]
        assert not out.success and out.failure_mode == "out_of_reach", (key, _why(far_runs[key]))
        assert out.quantities[f"reach_{axis}_mm"] >= at_least, (key, dict(out.quantities))
        assert out.quantities["palm_limit_mm"] == pytest.approx(350.0)
        assert out.duration_s <= 0.25, key
        why = [c for c in out.caveats if c.startswith("Out of reach:")]
        assert why and "+-350 mm" in why[0], (key, out.caveats)


def hotel_door_cases() -> dict[str, tuple[Any, TaskSpec]]:
    """The hotel hero's room door (300 mm deep, a 20 mm handle, hinged on its left, at most 90 deg),
    pulled 80 deg open from the front at 50 mm/s, where its recorded plan stood before the layout
    fixer's door rule (x -140 mm) and 30 mm further out: a door's bar is gripped from the front only,
    and there the front grip's palm would leave its travel."""

    def door_at(x_mm: float) -> TaskSpec:
        door = ObjectInstance.of(
            "room_door",
            "hinged_lid",
            "Room door",
            ObjectPose(x_mm, 290.0, 0.0),
            orientation="door",
            feature="handle",
            feature_mm=20,
            depth_mm=300,
            width_mm=400,
            thickness_mm=20,
            max_angle_deg=90,
            open_torque_nm=0.2,
            damping_nms=0.05,
        )
        pull = Step.of(
            "open_door",
            "hook_pull",
            "room_door",
            "handle",
            "Pull the door open by its handle",
            [SuccessCriterion("angle_deg", at_least=80.0)],
            params={"approach": "front", "speed_mm_s": 50},
        )
        return _one("door", (door,), pull)

    return {
        "plan": (gripper(), door_at(-140.0)),
        "further": (gripper(), door_at(-170.0)),
        "plan_servo": (_preset("catalog_gripper", "gripper-2f-servo-110"), door_at(-140.0)),
        "plan_angular": (_preset("catalog_gripper", "gripper-angular-jaw-hobby"), door_at(-140.0)),
    }


@pytest.fixture(scope="module")
def hotel_door_runs() -> dict[str, Any]:
    return _runs(hotel_door_cases())


def test_a_door_bar_the_front_grip_cannot_reach_or_fit_is_not_attempted(
    hotel_door_runs: dict[str, Any],
) -> None:
    """A door's bar is gripped from the front only. At x -140 mm the hotel door swings its face toward
    the table's left edge, so the front grip's palm would leave its travel (the two-finger adaptive and
    the servo-110 grippers alike, and 30 mm further out too): the step is not attempted and fails
    out_of_reach, saying it is the front grip's pull. The angular-jaw gripper has no calibrated pair of
    fingers to close across the bar: its step fails criterion_not_met with that reason. Nothing grips
    the bar from above, and the door stays shut."""
    for key in ("plan", "plan_servo", "further"):
        out = hotel_door_runs[key].steps[0]
        assert not out.success and out.failure_mode == "out_of_reach", (key, _why(hotel_door_runs[key]))
        assert out.duration_s <= 0.25 and out.quantities["angle_deg"] < 1.0, (key, dict(out.quantities))
        assert out.quantities["reach_x_mm"] < -338.0, (key, dict(out.quantities))
        why = [c for c in out.caveats if c.startswith("Out of reach:")]
        assert why and why[0].startswith("Out of reach: the front grip's pull would take the palm"), (
            out.caveats
        )
        assert "+-350 mm travel less 12 mm" in why[0], why
    out = hotel_door_runs["plan_angular"].steps[0]
    assert not out.success and out.failure_mode == "criterion_not_met", _why(hotel_door_runs["plan_angular"])
    assert out.duration_s <= 0.25 and out.quantities["angle_deg"] < 1.0, dict(out.quantities)
    why = [c for c in out.caveats if c.startswith("Cannot grip the door's bar from the front:")]
    assert why and "no pair of fingers either side of its axis" in why[0], out.caveats


def edge_drawer_cases() -> dict[str, tuple[Any, TaskSpec]]:
    """A drawer at the table's -y edge pulled 10 mm open, gripped from above: at y -100 mm its bar handle
    ends the pull 343 mm out, inside the travel less 5 mm, and the retreat 40 mm off the face would put
    the palm at 383 mm; 10 mm further out the pull itself leaves the travel."""

    def drawer_at(y_mm: float) -> TaskSpec:
        drawer = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0.0, y_mm, 0.0), depth_mm=350)
        pull = Step.of(
            "open",
            "hook_pull",
            "drawer",
            "handle",
            "Pull the drawer open by its bar",
            [SuccessCriterion("travel_mm", at_least=10.0)],
            params={"approach": "top", "speed_mm_s": 50},
        )
        return _one("drawer", (drawer,), pull)

    return {"edge": (gripper(), drawer_at(-100.0)), "further": (gripper(), drawer_at(-110.0))}


@pytest.fixture(scope="module")
def edge_drawer_runs() -> dict[str, Any]:
    return _runs(edge_drawer_cases())


def test_a_pull_whose_retreat_alone_meets_the_travel_is_made(
    edge_drawer_runs: dict[str, Any], task: Any
) -> None:
    """The retreat after a pull (40 mm off the face, then up) is not a reason to refuse a step: it
    stops at the palm's travel less the margin. The drawer at the table's edge is pulled open although
    its full retreat would put the palm past its travel (the plan's retreat is shortened to fit); a
    drawer whose pull itself leaves the travel is still refused, as the pull."""
    from rtt.studio.primitives.hook_pull import RETRACT_OUT, HookPullController

    out = edge_drawer_runs["edge"].steps[0]
    assert out.success and out.quantities["travel_mm"] >= 10.0, _why(edge_drawer_runs["edge"])
    assert out.failure_mode is None and not [c for c in out.caveats if c.startswith("Out of reach:")]
    tool, spec = edge_drawer_cases()["edge"]
    scene = build_studio_scene(Setup.single(tool, spec.step_ids), spec, 0, task=task)
    data = mujoco.MjData(scene.model)
    mujoco.mj_kinematics(scene.model, data)
    retreat = (
        HookPullController().plan(ctx_for(scene, spec, task), scene.model, data).values["run"]["retreat"]
    )
    assert 0.0 <= retreat < RETRACT_OUT - 0.01, retreat  # shortened at the travel, not refused
    far = edge_drawer_runs["further"].steps[0]
    assert not far.success and far.failure_mode == "out_of_reach", _why(edge_drawer_runs["further"])
    assert far.quantities["reach_y_mm"] < -345.0 and far.duration_s <= 0.25, dict(far.quantities)
    why = [c for c in far.caveats if c.startswith("Out of reach:")]
    assert why and why[0].startswith("Out of reach: the pull would take the palm"), far.caveats


def test_a_push_out_of_reach_fails_as_out_of_reach(task: Any) -> None:
    """A door at y -300 mm left 200 mm open (it opens toward -y) puts its handle 500 mm out:
    the push that would shut it is not attempted."""
    from rtt.studio.episode import StudioCondition, WorldState
    from rtt.studio.runner import run_segments

    door = ObjectInstance.of("door", "sliding_door", "Machine door", ObjectPose(0.0, -300.0, 0.0))
    shut = door_cases()["gripper"][1].steps[1]
    spec = _one("door", (door,), shut)
    setup = Setup.single(gripper(), spec.step_ids)
    slide = scene_of(spec, task).objects["door"].handles.joints["slide"]
    r, _ = run_segments(setup, spec, StudioCondition("nominal", 0), initial=WorldState({slide: (0.2,)}, 0.0))
    out = r.steps[0]
    assert not out.success and out.failure_mode == "out_of_reach", _why(r)
    assert out.quantities["reach_y_mm"] <= -460.0 and out.duration_s <= 0.25, dict(out.quantities)
    assert out.quantities["travel_mm"] == pytest.approx(200.0, abs=1.0)  # left where it was


def test_a_tile_heavier_than_its_bed_yield_or_of_an_impossible_mass_is_refused() -> None:
    with pytest.raises(ValueError, match="weighs 39.2 N, at or over the bed's 30 N yield"):
        ObjectInstance.of("tile", "tile_bed", "Tile", mass_g=4000.0, bed_yield_n=30.0, thickness_mm=12.0)
    with pytest.raises(ValueError, match="kg/m\\^3; a tile is 800 to 4000 kg/m\\^3"):
        ObjectInstance.of("tile", "tile_bed", "Tile", mass_g=300.0)
    ObjectInstance.of("tile", "tile_bed", "Tile", mass_g=2900.0, bed_yield_n=30.0)  # 28.4 N: carried
