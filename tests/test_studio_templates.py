"""Track S2: the six articulated Studio templates in physics (``docs/STUDIO.md`` 3.2, ``docs/STUDIO_PRIMITIVES.md``).

* every template kind has a builder; every variant (lid or door, lip or handle; bar, knob or lip
  pulls; wing or slot heads; cap, coupling or nut; spoked or solid wheels; plain or latching
  ports) compiles in the Studio scene with its handle sites and rests still;
* every param is honoured: joint ranges, frictions and dampings, the passive torque curves
  through the templates' own actuators (catch, detent, thread ramp, valve curve), the thread's
  helical flank, the socket's spring pads and latch groove;
* the state each builder reports is the template's quantities, and the controllers' view of
  it (:func:`rtt.studio.primitives._common.observed_state`, observed sites only) agrees with it
  and does not move with the hidden aim offset.
"""

from __future__ import annotations

import math
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
    ObjectInstance,
    ObjectPose,
    Setup,
    Step,
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
    # tightening while pushed advances one pitch per turn until seated
    d = settle(scene, 4.0, applied={slide: 5.0, hinge: 0.6})
    st = builder.state(obj, m, d)
    assert st["depth_mm"] == pytest.approx(1000 * (lead + 0.004), abs=0.3)  # seated: the slide's stop
    assert st["turns"] >= 2.0
    # the friction ramp: start torque at the thread start, final when seated
    probe = mujoco.MjData(m)
    probe.qvel[hinge] = 10.0
    for depth, want in ((lead, 0.1), (lead + 0.002, 0.55), (lead + 0.004, 1.0)):
        probe.qpos[obj.joint_qpos["slide"]] = depth
        resist = -builder.passive(obj, m, probe)[0] + info["hold_nm"]
        assert resist == pytest.approx(want, rel=1e-3), depth
    for member, geom in (("cap", "obj_c_top"), ("nut", "obj_c_piece05")):
        s = scene_of(one_object_task("c", kind="threaded_joint", member=member), task)
        assert geom in {s.model.geom(g).name for g in s.objects["c"].geom_ids}


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
    # below half the breakaway torque the wheel holds; above the curve it turns open
    held = settle(scene, 0.4, applied={dof: 0.8})
    assert scene.builders["va"].state(obj, m, held)["angle_deg"] < 1.0
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
    ):
        assert get_template_builder(kind) is get_template_builder(kind)
