"""Track B2: the soft arm and the suction cup (product mode, approximation; docs/EFFECTORS.md 11.3).

Gates proved here:

* suction holds ``block`` and ``steel_slug`` at nominal and refuses, with ``no_seal``, the
  bushing's bore, the tee's curved run and the standoff's tapped hole (every preset);
* suction seal loss drops the part within 0.2 s; the holding force never exceeds vacuum x area;
* the soft arm straightens to within 5 mm of straight within 1 s of releasing the cable, holds
  at least 2 of the 6 training parts at nominal and reports ``no_wrap`` where it fails to wrap;
* held penetration at most 1 mm in every held episode of both families;
* ``param_space`` rows validate and round-trip through ``MorphologyRecipe`` and
  ``spec_from_recipe``;
* the common gates C3 (determinism in and out of process), C4 (physics validity), C5 (labels,
  no brands, valid lanes) and C6 (runtime) for every preset.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import statistics
import time
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.effectors import registry
from rtt.effectors.registry import get_family, presets_of
from rtt.effectors.runner import EffectorJob, run_effector_job, run_effector_jobs
from rtt.effectors.scene import build_effector_scene
from rtt.effectors.soft_arm import (
    ARM_KIND,
    MAX_TENSION_N,
    MIN_SEGMENTS,
    ArmGeometry,
    SoftArmFamily,
    arm_spec,
    straighten_test,
)
from rtt.effectors.soft_arm import (
    FIDELITY_CAVEAT as ARM_CAVEAT,
)
from rtt.effectors.suction import (
    CUP_KIND,
    VACUUM_MAX_KPA,
    CupGeometry,
    SealTest,
    SuctionFamily,
    cup_spec,
)
from rtt.effectors.suction import (
    FIDELITY_CAVEAT as CUP_CAVEAT,
)
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    EffectorFamily,
    EffectorSpec,
    EntrantGroup,
    Fidelity,
    InvalidEffector,
    MorphologyRecipe,
    find_brand_tokens,
    params_json_schema,
)
from rtt.parts.library import get_part
from rtt.parts.suite import train_suite
from rtt.sim import objects as objs
from rtt.sim import replay as rp
from rtt.sim.certify import certificate_task
from rtt.sim.controller import GraspController, _plan, reach_floor
from rtt.sim.task import default_task_v2
from schema_lite import Validator

TRAIN_PARTS = ("block", "steel_slug", "flanged_bushing", "pipe_tee", "hex_standoff", "l_bracket")
FAMILIES = (EffectorFamily.SUCTION, EffectorFamily.SOFT_ARM)
G = 9.81


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


@pytest.fixture(scope="module")
def ctask(task: Any) -> Any:
    """The certificate's task (jitter off): the "nominal" episode of every gate."""
    return certificate_task(task)


@pytest.fixture(scope="module")
def nominal_jobs(ctask: Any) -> list[EffectorJob]:
    return [
        EffectorJob(spec, get_part(p), objs.Placement(), 0, task=ctask, condition_id=f"nominal/{p}")
        for spec in presets_of(FAMILIES)
        for p in TRAIN_PARTS
    ]


@pytest.fixture(scope="module")
def nominal(nominal_jobs: list[EffectorJob]) -> dict[tuple[str, str], Any]:
    """Every preset of both families at nominal on the six training parts, in spawned workers."""
    results = run_effector_jobs(nominal_jobs, workers=4)
    return {(j.spec.model, j.part.name): r for j, r in zip(nominal_jobs, results, strict=True)}


# --------------------------------------------------------------------------- registry and data contract


@pytest.mark.parametrize(
    ("family", "cls", "group", "caveat"),
    [
        (EffectorFamily.SUCTION, SuctionFamily, EntrantGroup.SUCTION, CUP_CAVEAT),
        (EffectorFamily.SOFT_ARM, SoftArmFamily, EntrantGroup.SOFT_ARM, ARM_CAVEAT),
    ],
)
def test_family_registers_as_a_labelled_approximation(family: Any, cls: Any, group: Any, caveat: str) -> None:
    impl = get_family(family)
    assert isinstance(impl, cls) and impl.fidelity is Fidelity.APPROXIMATION
    assert impl.reference_mode == "nominal_episode" and impl.caveat == caveat
    assert "approximation" in caveat.lower() and "not modelled" in caveat.lower()
    specs = impl.presets()
    assert len(specs) == 3 and len({s.effector_id for s in specs}) == 3
    for s in specs:
        assert s.origin == "preset" and s.group is group and s.fidelity is Fidelity.APPROXIMATION
        assert not find_brand_tokens(s.display_name) and len(s.display_name) <= 60
        d = s.to_dict()
        assert d["fidelity"] == "approximation" and d["label"] == PRODUCT_MODE_LABEL
        json.dumps(d, allow_nan=False)
        assert pickle.loads(pickle.dumps(s)).effector_id == s.effector_id
    assert "verify on hardware" in CUP_CAVEAT


def _random_params(space: Any, rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in space:
        if p.kind == "int":
            out[p.name] = rng.randint(int(p.low), int(p.high))
        elif p.kind == "float":
            out[p.name] = round(rng.uniform(p.low, p.high), 3)
        elif p.kind == "choice":
            out[p.name] = rng.choice(p.choices)
        else:
            out[p.name] = rng.random() < 0.5
    return out


@pytest.mark.parametrize(
    ("family", "kind"), [(EffectorFamily.SUCTION, CUP_KIND), (EffectorFamily.SOFT_ARM, ARM_KIND)]
)
def test_param_space_round_trips_through_recipes_and_builds(family: Any, kind: str, task: Any) -> None:
    impl = get_family(family)
    assert impl.recipe_kinds() == (kind,) and impl.param_space("nope") == ()
    space = impl.param_space(kind)
    schema = params_json_schema(space)
    assert schema["required"] == [p.name for p in space] and not schema["additionalProperties"]
    assert all(p.default is not None and p.problems(p.default) == [] for p in space)
    block = get_part("block")
    rng = random.Random(20261017)
    built = refused = 0
    for i in range(24):
        origin = ("nemotron", "random", "manual")[i % 3]
        recipe = MorphologyRecipe(family, kind, _random_params(space, rng), hypothesis="test", origin=origin)
        assert recipe.problems(space) == []
        spec = impl.spec_from_recipe(recipe)
        assert spec.family is family and spec.recipe is recipe and spec.origin == origin
        assert spec.group is {"nemotron": EntrantGroup.DESIGNED, "random": EntrantGroup.RANDOM}.get(
            origin, EntrantGroup(family.value)
        )
        assert not find_brand_tokens(spec.display_name)
        again = impl.spec_from_recipe(MorphologyRecipe(family, kind, dict(recipe.params), origin=origin))
        assert again.effector_id == spec.effector_id  # the hypothesis is not hashed
        if i < 8:  # build a few into the shared scene: no ContractError, or an honest refusal
            try:
                scene = build_effector_scene(impl, spec, block, task)
            except InvalidEffector as exc:
                assert str(exc)
                refused += 1
            else:
                assert scene.build.n_actuators == 1 and scene.model.nu == 5
                built += 1
    assert built >= 4
    bad_value = {**{p.name: p.default for p in space}, space[0].name: space[0].high + 1}
    with pytest.raises(ValueError, match="invalid"):
        impl.spec_from_recipe(MorphologyRecipe(family, kind, bad_value))
    with pytest.raises(ValueError, match="invalid"):
        impl.spec_from_recipe(
            MorphologyRecipe(family, kind, {**bad_value, space[0].name: space[0].default, "x": 1})
        )
    with pytest.raises(ValueError, match="not a"):
        impl.spec_from_recipe(MorphologyRecipe(family, "other_kind", {p.name: p.default for p in space}))


def test_presets_build_with_their_declared_actuation(task: Any) -> None:
    block = get_part("block")
    for spec in presets_of(FAMILIES):
        impl = get_family(spec.family)
        scene = build_effector_scene(impl, spec, block, task)
        m, b = scene.model, scene.build
        a = int(scene.ids.effector_actuators[0])
        assert b.n_actuators == 1 and m.nu == 5 and 0.0 < b.mass < 0.5
        assert b.notes[0] in (CUP_CAVEAT, ARM_CAVEAT)
        geoms = np.flatnonzero(np.isin(m.geom_bodyid, scene.ids.effector_bodies))
        # no contact at a distance, no out-ranking, no sticky geoms
        assert np.all(m.geom_margin[geoms] == 0) and np.all(m.geom_gap[geoms] == 0)
        assert np.all(m.geom_priority[geoms] == 0) and np.all(m.geom_adhesion[geoms] == 0)
        if spec.family is EffectorFamily.SUCTION:
            g = CupGeometry.from_params(spec.params)
            assert int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_BODY)
            assert int(m.actuator_trnid[a, 0]) == m.body("eff_cup").id
            assert g.vacuum_pa <= 1000.0 * VACUUM_MAX_KPA
            cap = g.vacuum_pa * math.pi * (0.5 * g.diameter) ** 2
            assert math.isclose(m.actuator_gainprm[a, 0], cap) and math.isclose(g.max_force, cap)
            assert list(m.actuator_ctrlrange[a]) == [0.0, 1.0]
            assert list(m.actuator_forcerange[a]) == pytest.approx([0.0, cap])
            assert math.isclose(b.actuation["holding_force_max_n"], cap) and b.actuation["vacuum_kpa"] <= 60
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(n)) or "" for n in geoms]
            assert sum(n.startswith("eff_lip_") for n in names) == g.lip_segments
        else:
            g = ArmGeometry.from_params(spec.params)
            assert g.n >= MIN_SEGMENTS and g.radius(g.n - 1) < g.radius(0)  # at least 12 segments, tapered
            assert int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON)
            assert (
                m.actuator_ctrlrange[a, 1] <= MAX_TENSION_N and m.actuator_forcerange[a, 1] <= MAX_TENSION_N
            )
            assert b.actuation["cable_tension_n"] <= MAX_TENSION_N
            hinges = [m.joint(f"eff_seg_{i:02d}") for i in range(g.n)]
            assert all(float(j.stiffness[0]) > 0 and float(j.damping[0]) > 0 for j in hinges)
            assert m.ntendon == 1 and m.tendon_num[0] == 1 + 2 * g.n  # through every segment


def test_soft_arm_refuses_a_part_wider_than_its_clearance(task: Any) -> None:
    narrow = EffectorSpec(
        "soft_arm",
        "narrow",
        "Soft arm, close post",
        "wrap",
        {**{p.name: p.default for p in get_family("soft_arm").param_space(ARM_KIND)}, "stem_offset_mm": 20.0},
    )
    r = get_family("soft_arm").run_episode(narrow, get_part("block"), objs.Placement(), 0, task=task)
    assert r.failure_mode == "invalid_build" and "wide" in r.flags[0]


# --------------------------------------------------------------------------- suction gates


def test_suction_holds_the_flat_parts(nominal: dict[tuple[str, str], Any]) -> None:
    for model in ("cup-16mm", "cup-25mm"):
        for part in ("block", "steel_slug"):
            r = nominal[(model, part)]
            assert r.held and r.engaged and r.failure_mode is None, (model, part, r.failure_mode)
            assert r.penetration_mm <= 1.0
            assert r.metrics["eng_vacuum_force_n"] <= r.metrics["eng_holding_force_cap_n"] + 1e-9
    assert nominal[("cup-10mm", "block")].held
    # the 10 mm cup's cap (60 kPa x 78.5 mm^2 = 4.7 N) is below the steel slug's weight: it seals, cannot lift
    slug = nominal[("cup-10mm", "steel_slug")]
    assert slug.engaged and not slug.held
    assert slug.metrics["eng_holding_force_cap_n"] < slug.metrics["object_mass"] * G


@pytest.mark.parametrize("part", ["flanged_bushing", "pipe_tee", "hex_standoff"])
def test_suction_refuses_bores_curves_and_holes(part: str, nominal: dict[tuple[str, str], Any]) -> None:
    for spec in get_family("suction").presets():
        r = nominal[(spec.model, part)]
        assert not r.held and not r.engaged and not r.success, (spec.model, part)
        assert "no_seal" in r.registration_flags, (spec.model, part, r.registration_flags)
        assert r.metrics["eng_vacuum_on"] == 0.0 and r.max_lift < 0.005


def _drive(impl: Any, spec: EffectorSpec, part: Any, task: Any) -> tuple[Any, Any, Any, Any, Any]:
    """A unit scene: the shared scene, the frozen wrist plan and the family's primitive."""
    scene = build_effector_scene(impl, spec, part, task)
    m, ids, b = scene.model, scene.ids, scene.build
    d = mujoco.MjData(m)
    prim = impl.primitive(spec, b)
    obj = part.to_mesh_object()
    h = objs.height(obj)
    pp = prim.plan(part, objs.Placement(), task, b)
    plan = _plan(task, objs.Placement(), h, b.reach, pp.tip_height, reach_floor(b.reach, h, task))
    pos, quat = objs.rest_pose(obj, objs.Placement(), task.physics.table_top_z)
    d.qpos[ids.palm_qpos] = plan.palm_start
    d.qpos[ids.object_qpos : ids.object_qpos + 3] = pos
    d.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = quat
    mujoco.mj_forward(m, d)
    return scene, d, prim, plan, GraspController(plan)


def test_suction_seal_loss_drops_the_part_within_0p2_s(ctask: Any) -> None:
    """Hold the block, then tilt it past the lip's compliance: the vacuum cuts, the part falls."""
    impl, spec, block = get_family("suction"), cup_spec(16.0), get_part("block")
    g = CupGeometry.from_params(spec.params)
    scene, d, prim, plan, wrist = _drive(impl, spec, block, ctask)
    m, ids = scene.model, scene.ids
    probe = SealTest(g)
    vac = m.actuator("eff_vacuum").id
    tilt = m.joint("eff_cup_tilt_x").qposadr[0]
    is_eff = np.zeros(m.ngeom, dtype=bool)
    is_eff[ids.effector_geoms] = True
    is_obj = np.zeros(m.ngeom, dtype=bool)
    is_obj[ids.object_geoms] = True
    cp = ctask.controller.control_period
    t_tilt = plan.t_hold + 0.5
    t_loss = t_free = z_loss = None
    force_max, lift_at_tilt, tilt_at_loss = 0.0, 0.0, 0.0
    z0 = float(d.xpos[ids.object_body][2])
    for tick in range(round((t_tilt + 1.5) / cp)):
        t = tick * cp
        d.ctrl[:4] = wrist.command(t)[:4]
        closure = min(max((t - plan.t_close) / plan.close_time, 0.0), 1.0)
        d.ctrl[4:] = prim.command(t, wrist.phase(t), closure, m, d, scene)
        # the test bench (not the effector) tilts the face: a torque on the part, ramped to 0.1 N m
        d.xfrc_applied[ids.object_body, 3] = (
            0.1 * min(1.0, (t - t_tilt) / 0.5) if t >= t_tilt and t_loss is None else 0.0
        )
        mujoco.mj_step(m, d, nstep=round(cp / m.opt.timestep))
        force_max = max(force_max, float(d.actuator_force[vac]))
        if t < t_tilt:
            lift_at_tilt = float(d.xpos[ids.object_body][2]) - z0
            continue
        if t_loss is None and not probe(m, d, scene).sealed:
            t_loss, z_loss = d.time, float(d.xpos[ids.object_body][2])
            tilt_at_loss = abs(math.degrees(float(d.qpos[tilt])))
        elif t_loss is not None and t_free is None:
            g_ = d.contact.geom
            touching = d.ncon and bool(
                ((is_eff[g_[:, 0]] & is_obj[g_[:, 1]]) | (is_obj[g_[:, 0]] & is_eff[g_[:, 1]])).any()
            )
            if not touching:
                t_free = d.time
                assert d.ctrl[4] == 0.0  # the switch is off
        if t_loss is not None and d.time >= t_loss + 0.2:
            break
    assert lift_at_tilt > 0.10  # it was holding the block in the air
    assert t_loss is not None and t_free is not None and z_loss is not None
    assert tilt_at_loss >= math.degrees(g.max_tilt) - 1.0  # the face went past the lip's tilt compliance
    assert t_free - t_loss <= 0.2
    assert z_loss - float(d.xpos[ids.object_body][2]) > 0.02  # and it is falling
    assert force_max <= g.max_force + 1e-9  # never more than vacuum x area
    assert g.max_force == pytest.approx(60e3 * math.pi * 0.008**2)


# --------------------------------------------------------------------------- soft arm gates


@pytest.mark.parametrize("model", ["tentacle-12", "tentacle-16", "tentacle-20-soft"])
def test_soft_arm_straightens_within_1_s_of_release(model: str) -> None:
    r = straighten_test(arm_spec(model))
    assert r.curl_deg > 180.0 and r.tip_deviation_mm > 50.0  # it really curled
    assert r.ok and r.settle_time_s <= 1.0 and r.deviation_at_horizon_mm <= 5.0, r


def test_soft_arm_holds_at_least_two_training_parts(nominal: dict[tuple[str, str], Any]) -> None:
    for spec in get_family("soft_arm").presets():
        held = [p for p in TRAIN_PARTS if nominal[(spec.model, p)].held]
        assert len(held) >= 2, (spec.model, held)
        for p in TRAIN_PARTS:
            r = nominal[(spec.model, p)]
            if r.engaged:
                assert r.metrics["eng_wrap_deg"] >= 180.0 and r.metrics["eng_segments"] >= 3
            elif "not_measured" not in r.registration_flags:
                assert "no_wrap" in r.registration_flags, (spec.model, p, r.registration_flags)
    assert nominal[("tentacle-16", "hex_standoff")].engaged  # a wrap, not a pinch


# --------------------------------------------------------------------------- common gates


def test_held_penetration_is_at_most_1_mm(nominal: dict[tuple[str, str], Any]) -> None:
    held = [r for r in nominal.values() if r.held]
    assert len(held) >= 10
    assert all(r.penetration_mm <= 1.0 for r in held), [
        (r.spec.model, r.part_id) for r in held if r.penetration_mm > 1
    ]


def test_physics_validity_c4(nominal: dict[tuple[str, str], Any]) -> None:
    results = list(nominal.values())
    assert all(r.bad_qacc == 0 for r in results)
    assert not any(r.failure_mode == "physics_invalid" for r in results)
    lifted = [r for r in results if r.held or r.failure_mode == "physics_unreliable"]
    assert sum(r.failure_mode == "physics_unreliable" for r in lifted) <= 0.05 * len(lifted)
    for r in results:  # no effector-table contact flag in any nominal episode
        assert "gripper_env_force" not in r.flags and r.failure_mode != "finger_env_collision", (
            r.spec.model,
            r.part_id,
        )


def test_determinism_c3(nominal_jobs: list[EffectorJob], nominal: dict[tuple[str, str], Any]) -> None:
    """Per preset: two in-process runs and the spawned-worker run give the same trajectory hash."""
    for job in nominal_jobs:
        if job.part.name != "block":
            continue
        there = nominal[(job.spec.model, "block")].traj_sha
        first, second = run_effector_job(job), run_effector_job(job)
        assert len(there) == 64 and first.traj_sha == second.traj_sha == there, job.spec.model


@pytest.mark.parametrize("model", ["cup-16mm", "tentacle-16"])
def test_records_and_lanes_are_labelled_and_brand_free_c5(model: str, task: Any) -> None:
    spec = next(s for s in presets_of(FAMILIES) if s.model == model)
    cond = next(c for c in train_suite() if c.part.name == "block" and c.placement_label == "c2")
    r = get_family(spec.family).run_episode(
        spec, cond.part, cond.placement, cond.seeds[0], task=task, record=True, condition_id=cond.condition_id
    )
    rec = r.to_record()
    json.dumps(rec, allow_nan=False)
    assert rec["label"] == PRODUCT_MODE_LABEL and rec["fidelity"] == "approximation"
    assert not find_brand_tokens(json.dumps(rec))
    assert r.replay is not None and r.overlays is not None
    meta = r.replay.header["meta"]
    assert meta["effector"]["fidelity"] == "approximation" and meta["effector"]["label"] == PRODUCT_MODE_LABEL
    header = rp.upgrade_header(
        r.replay.header,
        visuals={},
        assets={},
        meta={"geometry_id": None},
        provenance=rp.provenance(run_id="effectors-b2", code_version="815ac94", product_mode=True),
    )
    assert rp.check_header(header, r.replay.frames) == []
    assert Validator(json.loads(rp.SCHEMA_PATH.read_text(encoding="utf-8"))).errors(header) == []
    assert not find_brand_tokens(json.dumps(header))
    payload = len(json.dumps(header).encode()) + r.replay.frames.nbytes
    payload += sum(v.nbytes for v in r.overlays.values())
    assert payload <= 300_000  # the lane budget (docs/EFFECTORS.md section 6)


@pytest.mark.parametrize(("model", "budget_s"), [("cup-16mm", 3.0), ("tentacle-16", 5.0)])
def test_runtime_c6(model: str, budget_s: float, task: Any) -> None:
    """Median CPU time of the six parts' c0 episodes on one core (wall time is load-dependent)."""
    spec = next(s for s in presets_of(FAMILIES) if s.model == model)
    impl = get_family(spec.family)
    cpu = []
    for cond in (c for c in train_suite() if c.placement_label == "c0"):
        t0 = time.process_time()
        impl.run_episode(spec, cond.part, cond.placement, cond.seeds[0], task=task)
        cpu.append(time.process_time() - t0)
    assert len(cpu) == 6 and statistics.median(cpu) <= budget_s, cpu


def test_test_override_keeps_registry_clean() -> None:
    """The two families are registered by import, never replaced by the tests."""
    assert isinstance(registry.get_family("suction"), SuctionFamily)
    assert isinstance(registry.get_family("soft_arm"), SoftArmFamily)
