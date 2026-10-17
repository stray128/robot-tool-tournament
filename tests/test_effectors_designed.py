"""The ``designed`` family (product mode, track B3): our own morphologies in the shared scene.

Builds, contract checks, physics and determinism of designed effectors; the build gate over
256 random recipes per generator on all six training parts; and the recorded smoke run's
physics gate (``src/rtt/effectors/morphology/recorded/``). The designer itself (spaces,
validators, proposer) is tested in ``tests/test_effectors_morphology.py``.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

import rtt.effectors as eff
from rtt.effectors.certificate import certify_effector
from rtt.effectors.designed import (
    DesignedFamily,
    DesignedPrimitive,
    display_name_of,
    reference_recipes,
    spec_from_recipe,
)
from rtt.effectors.morphology import generator as gen
from rtt.effectors.morphology import spec_for, validate_recipe
from rtt.effectors.morphology.diversity import canonical_recipe
from rtt.effectors.morphology.layout import layout_of
from rtt.effectors.morphology.random_baseline import draw_params
from rtt.effectors.morphology.spaces import DESIGNED_KINDS, space_for
from rtt.effectors.morphology.validators import validate_structure
from rtt.effectors.registry import get_family
from rtt.effectors.runner import EffectorJob, run_effector_jobs
from rtt.effectors.scene import MOUNT_Z, build_effector_scene
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    ContractError,
    EffectorFamily,
    EntrantGroup,
    InvalidEffector,
    MorphologyRecipe,
    find_brand_tokens,
)
from rtt.parts.library import get_part, training_parts
from rtt.parts.suite import train_suite
from rtt.sim import objects as objs
from rtt.sim import replay as rp
from rtt.sim.certify import certificate_task
from rtt.sim.task import default_task_v2
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
RECORDED = REPO / "src/rtt/effectors/morphology/recorded/smoke-2026-10-17"
#: Reference designs and a part each one holds at nominal (tests below re-check it).
HOLDS = {0: "hex_standoff", 1: "block", 2: "steel_slug", 3: "hex_standoff"}


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


@pytest.fixture(scope="module")
def ctask(task: Any) -> Any:
    return certificate_task(task)


def _spec(i: int) -> Any:
    return spec_from_recipe(reference_recipes()[i])


# --------------------------------------------------------------------------- the family


def test_family_registers_and_exposes_its_generators() -> None:
    impl = get_family("designed")
    assert isinstance(impl, DesignedFamily) and impl.reference_mode == "nominal_episode"
    assert impl.fidelity.value == "rigid" and impl.presets() == ()
    assert impl.recipe_kinds() == DESIGNED_KINDS
    for kind in DESIGNED_KINDS:
        assert impl.param_space(kind) == space_for("designed", kind)
    assert impl.param_space("nope") == ()


def test_specs_from_recipes_are_brand_free_and_grouped_by_origin() -> None:
    names = set()
    for r in reference_recipes():
        spec = spec_for(r)
        assert spec.family is EffectorFamily.DESIGNED and spec.recipe is r
        assert len(spec.display_name) <= 60 and not find_brand_tokens(spec.display_name)
        assert spec.primitive in ("power", "pinch") and spec.group is EntrantGroup.DESIGNED
        assert spec.origin == "manual" and spec.to_dict()["label"] == PRODUCT_MODE_LABEL
        names.add(spec.display_name)
    assert len(names) == len(reference_recipes())
    base = reference_recipes()[0]
    nem = MorphologyRecipe(base.family, base.kind, base.params, "idea", origin="nemotron")
    rnd = MorphologyRecipe(base.family, base.kind, base.params, "", origin="random")
    assert spec_for(nem).group is EntrantGroup.DESIGNED and spec_for(rnd).group is EntrantGroup.RANDOM
    assert spec_for(nem).recipe.recipe_id == spec_for(rnd).recipe.recipe_id  # type: ignore[union-attr]
    assert spec_for(nem).effector_id != spec_for(rnd).effector_id  # origin is part of the effector
    assert display_name_of(base).startswith("Designed three-finger hand")
    with pytest.raises(ValueError):
        spec_from_recipe(MorphologyRecipe("soft_arm", "tentacle", {}))


# --------------------------------------------------------------------------- the build gate


def _gate_worker(kind: str, part_name: str, n: int) -> dict[str, Any]:
    """Build ``n`` random structurally valid recipes of ``kind`` on one part (spawned worker)."""
    task = default_task_v2()
    part = get_part(part_name)
    impl = get_family("designed")
    rng = np.random.default_rng(12345)
    space = space_for("designed", kind)
    built = invalid = contract = other = 0
    cpu = []
    reasons: dict[str, int] = {}
    count = 0
    while count < n:
        r = canonical_recipe(MorphologyRecipe("designed", kind, draw_params(space, rng), origin="random"))
        if validate_structure(r):
            continue
        count += 1
        t0 = time.process_time()
        try:
            build_effector_scene(impl, spec_from_recipe(r), part, task)
            built += 1
        except InvalidEffector as exc:
            invalid += 1
            text = str(exc)
            assert text.strip(), "an InvalidEffector without a reason"
            code = text.split(":", 1)[0]
            reasons[code] = reasons.get(code, 0) + 1
        except ContractError:
            contract += 1
        except Exception:  # noqa: BLE001 - any other failure is a gate failure, counted
            other += 1
        cpu.append(time.process_time() - t0)
    return {
        "kind": kind,
        "part": part_name,
        "built": built,
        "invalid": invalid,
        "contract": contract,
        "other": other,
        "cpu_max_s": max(cpu),
        "cpu_median_s": float(np.median(cpu)),
        "reasons": reasons,
    }


def test_build_gate_256_random_recipes_per_generator_on_all_six_parts() -> None:
    """Every recipe builds or raises InvalidEffector with a reason; 0 ContractError; <= 200 ms each."""
    tasks = [(kind, part.name, 256) for kind in DESIGNED_KINDS for part in training_parts()]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(_gate_worker, *zip(*tasks, strict=True)))
    assert len(rows) == 18
    for row in rows:
        assert row["contract"] == 0 and row["other"] == 0, row
        assert row["built"] + row["invalid"] == 256, row
        assert row["cpu_max_s"] <= 0.2, row
    for kind in DESIGNED_KINDS:
        assert sum(r["built"] for r in rows if r["kind"] == kind) > 0, kind


# --------------------------------------------------------------------------- the scene contract


@pytest.mark.parametrize("index", list(HOLDS))
def test_reference_designs_build_into_the_shared_scene(index: int, task: Any) -> None:
    spec = _spec(index)
    part = get_part(HOLDS[index])
    scene = build_effector_scene(get_family("designed"), spec, part, task)
    lay = layout_of(spec.recipe)  # type: ignore[arg-type]
    m, b = scene.model, scene.build
    assert b.n_actuators == lay.n_actuators and m.nu == 4 + lay.n_actuators
    assert all(n.startswith("eff_") for n in b.bodies + b.actuators)
    assert b.mass is not None and math.isclose(b.mass, lay.mass(), rel_tol=1e-6)
    assert dict(b.actuation) == lay.actuation_card() and b.max_opening == pytest.approx(lay.max_opening)
    assert all(m.body_gravcomp[i] == 1.0 for i in scene.ids.effector_bodies)
    # every servo stays within the declared budget
    forces = m.actuator_forcerange[scene.ids.effector_actuators]
    hinge = [
        a
        for a in scene.ids.effector_actuators
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
        or m.jnt_type[m.actuator_trnid[a][0]] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert np.all(forces[:, 1] > 0) and np.all(forces[:, 0] == -forces[:, 1])
    if lay.actuation == "per_joint" and hinge:
        assert float(m.actuator_forcerange[hinge, 1].max()) <= 1.5 + 1e-9
    # reach: the lowest point in the reference pose sits `reach` below the mount plane
    data = mujoco.MjData(m)
    mujoco.mj_kinematics(m, data)
    wrist_z = float(data.xpos[scene.ids.palm_body][2])
    lowest = gen.lowest_point(m, data, scene.ids.effector_bodies)
    assert math.isclose(wrist_z + MOUNT_Z - lowest, b.reach, abs_tol=1e-9)


def test_an_unbuildable_design_scores_invalid_build_with_its_reason(task: Any) -> None:
    r = MorphologyRecipe(
        "designed", "fingered_hand", {**reference_recipes()[0].params, "palm_radius_mm": 20.0}
    )
    res = eff.run_effector_episode(spec_for(r), get_part("pipe_tee"), objs.Placement(), 0, task=task)
    assert res.failure_mode == "invalid_build" and res.n_steps == 0
    assert res.flags[0].startswith("invalid_build: too_narrow")
    assert validate_recipe(r, get_part("pipe_tee"), task)


def test_pinch_keeps_the_last_phalanx_upright(task: Any) -> None:
    spec = _spec(0)  # per-joint pinch, two phalanges
    scene = build_effector_scene(get_family("designed"), spec, get_part("hex_standoff"), task)
    prim = get_family("designed").primitive(spec, scene.build)
    assert isinstance(prim, DesignedPrimitive) and len(prim.upright) == 3
    data = mujoco.MjData(scene.model)
    for d in range(3):
        data.qpos[scene.model.jnt_qposadr[scene.model.joint(gen.joint_name("eff_", d, 0)).id]] = 0.4
    u = prim.command(2.0, None, 1.0, scene.model, data, scene)  # type: ignore[arg-type]
    for a, _pairs in prim.upright:
        assert math.isclose(u[a], -0.4)
    assert np.all(np.isfinite(u)) and u.shape == (scene.build.n_actuators,)


# --------------------------------------------------------------------------- physics


@pytest.mark.parametrize("index", list(HOLDS))
def test_reference_designs_hold_their_part_at_nominal(index: int, ctask: Any) -> None:
    spec = _spec(index)
    r = eff.run_effector_episode(spec, get_part(HOLDS[index]), objs.Placement(), 0, task=ctask)
    assert r.held and r.engaged and r.failure_mode in (None, "not_registered"), (r.failure_mode, r.flags)
    assert r.bad_qacc == 0 and r.penetration_mm <= 1.0 and "gripper_env_force" not in r.flags
    assert r.metrics["eng_bodies"] >= 2 and r.label == PRODUCT_MODE_LABEL
    rec = r.to_record()
    json.dumps(rec, allow_nan=False)
    assert rec["family"] == "designed" and rec["recipe_id"] == spec.recipe.recipe_id  # type: ignore[union-attr]


def test_episodes_are_identical_in_process_and_in_workers(task: Any) -> None:
    """C3: two in-process runs and a spawned-worker run give the same trajectory hash."""
    cond = next(c for c in train_suite() if c.part.name == "steel_slug" and c.placement_label == "c2")
    jobs = []
    for i, name in HOLDS.items():
        part = get_part(name) if i != 2 else cond.part
        jobs.append(EffectorJob(_spec(i), part, cond.placement, cond.seeds[0], task=task))
    here = [eff.run_effector_episode(j.spec, j.part, j.placement, j.seed, task=task) for j in jobs]
    again = run_effector_jobs(jobs, workers=1)
    there = run_effector_jobs(jobs, workers=2)
    assert [r.traj_sha for r in here] == [r.traj_sha for r in again] == [r.traj_sha for r in there]
    assert all(len(r.traj_sha) == 64 for r in here)


def test_runtime_and_validity_on_the_six_parts_c0(task: Any) -> None:
    """C4 and C6 on the chuck: no MuJoCo reset, no table flag, median episode under 3 s."""
    spec = _spec(2)
    walls, results = [], []
    for cond in train_suite():
        if cond.placement_label != "c0":
            continue
        r = eff.run_effector_episode(spec, cond.part, cond.placement, cond.seeds[0], task=task)
        results.append(r)
        if r.failure_mode != "invalid_build":
            walls.append(r.wall_time_s)
    assert walls and float(np.median(walls)) <= 3.0
    assert all(r.bad_qacc == 0 and "gripper_env_force" not in r.flags for r in results)


def test_certificate_and_lane_of_a_designed_effector(task: Any) -> None:
    spec = _spec(2)
    part = get_part("steel_slug")
    cert = certify_effector(spec, part, task=task, workers=4)
    assert cert.reference == "nominal_episode" and len(cert.tests) == 18 and cert.nominal_held
    assert all(t.bad_qacc == 0 for t in cert.tests)
    d = cert.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["label"] == PRODUCT_MODE_LABEL and d["fidelity"] == "rigid"
    cond = next(c for c in train_suite() if c.part.name == "steel_slug" and c.placement_label == "c2")
    r = eff.run_effector_episode(spec, part, cond.placement, cond.seeds[0], task=task, record=True)
    assert r.replay is not None
    header = rp.upgrade_header(
        r.replay.header,
        visuals={},
        assets={},
        meta={"geometry_id": None},
        provenance=rp.provenance(run_id="effectors-b3", code_version="815ac94", product_mode=True),
    )
    assert rp.check_header(header, r.replay.frames) == []
    assert Validator(json.loads(rp.SCHEMA_PATH.read_text(encoding="utf-8"))).errors(header) == []
    assert not find_brand_tokens(json.dumps(header))
    assert header["meta"]["effector"]["family"] == "designed"


# --------------------------------------------------------------------------- the recorded physics gate


def _manifest() -> dict[str, Any]:
    path = RECORDED / "manifest.json"
    if not path.is_file():
        pytest.skip("no recorded smoke run")
    return json.loads(path.read_text(encoding="utf-8"))


def test_recorded_physics_gate_and_random_control() -> None:
    """The best Nemotron design (chosen on the screen) held at nominal on at least 4 of 6 parts;
    the random control on the same budget is reported beside it, not ranked against it."""
    m = _manifest()
    gate = m["gate"]
    nem, rnd = m["arms"]["nemotron"], m["arms"]["random"]
    assert gate["required"] == 4 and gate["passed"] and gate["nemotron_best_nominal_held"] >= 4
    assert nem["summary"]["best_nominal_held"] == gate["nemotron_best_nominal_held"]
    assert rnd["summary"]["designs"] == 8 and "random_best_nominal_held" in gate
    assert gate["claim"].startswith("none")
    assert len(nem["designs"]) <= 8 and all(d["recipe"]["origin"] == "nemotron" for d in nem["designs"])
    assert all(d["recipe"]["origin"] == "random" for d in rnd["designs"])
    for arm in (nem, rnd):
        for d in arm["designs"]:
            assert d["n_episodes"] == 48 and len(d["nominal"]) == 6


def test_recorded_best_design_reproduces_its_nominal_episodes(ctask: Any) -> None:
    """Re-simulating the recorded best Nemotron design gives the recorded trajectories."""
    m = _manifest()
    best = m["arms"]["nemotron"]["summary"]["best_by_screen"]
    d = next(x for x in m["arms"]["nemotron"]["designs"] if x["recipe"]["recipe_id"] == best)
    rec = d["recipe"]
    recipe = MorphologyRecipe(rec["family"], rec["kind"], rec["params"], rec["hypothesis"], rec["origin"])
    assert recipe.recipe_id == best
    spec = spec_for(recipe)
    for part_name in sorted(d["nominal"])[:3]:
        r = eff.run_effector_episode(spec, get_part(part_name), objs.Placement(), 0, task=ctask)
        assert r.traj_sha == d["nominal"][part_name]["traj_sha"], part_name
        assert r.held == d["nominal"][part_name]["held"]
