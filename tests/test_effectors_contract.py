"""The frozen multi-effector contract (``rtt.effectors``, product mode, not pre-registered).

What this pins down for the four builder tracks (``docs/EFFECTORS.md``):

* the data contract: ids, canonical params, brand-free display names, licence records,
  morphology recipes and parameter spaces, pickling for worker processes;
* the registry: one implementation per family, auto-import by module name, overrides for tests;
* the shared scene: the core world, wrist, servos and part, the contract checks on a family's
  additions, measured mass properties;
* the shared runner and certificate, end to end on a toy native family;
* the reference family: the frozen v2 jaw path, bit-identical to ``run_episode_v2`` and
  ``certify_v2``;
* the guard rail: the PREREG-frozen code listings and files are unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

import rtt.effectors as eff
from effector_toy import TOY_MASS, ToyTwoPad, toy_spec
from rtt.effectors import registry
from rtt.effectors.certificate import certify_effector
from rtt.effectors.parallel_jaw import ParallelJawReference, jaw_spec
from rtt.effectors.runner import EffectorJob, run_effector_job, run_effector_jobs, run_native_episode
from rtt.effectors.scene import FLANGE_GEOM, MOUNT_Z, build_effector_scene
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    AssetSource,
    ContractError,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    Engagement,
    EntrantGroup,
    Fidelity,
    GraspCriterion,
    GraspPrimitive,
    MorphologyRecipe,
    ParamSpec,
    RegistrationReference,
    find_brand_tokens,
    params_json_schema,
)
from rtt.parts.library import get_part
from rtt.parts.suite import train_suite
from rtt.sim import objects as objs
from rtt.sim import replay as rp
from rtt.sim.certify import certify_v2
from rtt.sim.episode import run_episode_v2
from rtt.sim.scene import build_model_v2
from rtt.sim.task import default_task_v2
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
SHA = "0" * 64


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


@pytest.fixture(scope="module")
def block() -> Any:
    return get_part("block")


@pytest.fixture()
def toy() -> Any:
    impl = ToyTwoPad()
    with registry.override_family(impl):
        yield impl


# --------------------------------------------------------------------------- guard rails


def _listing(directory: Path, pattern: str = "*.py") -> str:
    """``(cd DIR && sha256sum PATTERN | sha256sum)`` with bytewise (C locale) name order."""
    files = sorted(directory.glob(pattern), key=lambda f: f.name.encode())
    lines = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def test_prereg_frozen_code_and_files_are_unchanged() -> None:
    """The product layer never touches what PREREG A5 hashed (C-locale listings)."""
    listings = {
        "src/rtt/design/freeform": "f62376d69fa331ccb66da5d78828bd17e1f5f0d7d4fba60661f54f51a27e9422",
        "src/rtt/parts": "5f80c1776d413fc1c4150ddc61b936f8145e0904ec640c9049cd957ecc5a3f99",
        "src/rtt/analysis": "c78462aadacfc24a714085cd4b9c5476fbfa94d050289e76c572cf6cd2914043",
        "src/rtt/search": "42861eee09114842c0af0dd7404add70b762e7254c9d90ae0cf20682052d138d",
        "src/rtt/sim": "069763de5090a66cef204186a81bb221433ca7ed1e2b54b447dfb728a48d04ea",
    }
    for rel, want in listings.items():
        assert _listing(REPO / rel) == want, f"{rel} changed: it is frozen by PREREG"
    assert _listing(REPO / "src/rtt/search/prompts/v2", "*") == (
        "f9d2b5ac0c39fb1af6f7e56663e32209bfca9e573605fab1d46f2ffda7d5575e"
    )
    files = {
        "configs/task_v2.yaml": "bef7297f8e1ff2a6270cd3f1ffb6de8690928f46485828a58fd1ce2313d87487",
        "uv.lock": "e876d63062a8c3d88f748b28b1618f9c474b75fdf56c3b9011b3ca61df24e922",
    }
    for rel, want in files.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == want, f"{rel} is frozen"


def test_effector_code_never_reads_held_out_material() -> None:
    root = Path(eff.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "vault" not in node.module and "heldout" not in node.module, path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                assert "heldout" not in low and "secrets/" not in low, path.name


def test_product_mode_label_matches_the_agent() -> None:
    from rtt.agent.recorder import PRODUCT_MODE_LABEL as AGENT_LABEL

    assert PRODUCT_MODE_LABEL == AGENT_LABEL


# --------------------------------------------------------------------------- data contract


def test_spec_id_is_canonical_and_ignores_the_display_name() -> None:
    a = EffectorSpec(
        EffectorFamily.SUCTION, "cup-20mm", "Suction cup, 20 mm", "seal", {"b": 1, "a": [1.0, 2]}
    )
    b = EffectorSpec("suction", "cup-20mm", "Small suction cup", "seal", {"a": [1, 2.0], "b": 1.0})
    assert a.effector_id == b.effector_id and len(a.effector_id) == 64
    c = EffectorSpec("suction", "cup-20mm", "Suction cup, 20 mm", "seal", {"a": [1, 2], "b": 2})
    assert c.effector_id != a.effector_id
    assert a.fidelity is Fidelity.APPROXIMATION and a.group is EntrantGroup.SUCTION
    with pytest.raises(TypeError):
        a.params["b"] = 3  # type: ignore[index]
    again = pickle.loads(pickle.dumps(a))
    assert again.effector_id == a.effector_id and dict(again.params) == dict(a.params)
    d = a.to_dict()
    assert d["label"] == PRODUCT_MODE_LABEL and d["effector_id"] == a.effector_id
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"display_name": "Allegro-style 2F-85"}, "brand"),
        ({"display_name": "leap hand clone"}, "brand"),
        ({"display_name": ""}, "display_name"),
        ({"model": "Cup 20"}, "slug"),
        ({"primitive": "Seal!"}, "slug"),
        ({"params": {"x": math.nan}}, "finite"),
        ({"origin": "nemotron"}, "MorphologyRecipe"),
        ({"origin": "somewhere"}, "origin"),
    ],
)
def test_spec_rejects_bad_fields(kwargs: dict[str, Any], match: str) -> None:
    base = {
        "family": "suction",
        "model": "cup-20mm",
        "display_name": "Suction cup",
        "primitive": "seal",
        "params": {},
    }
    with pytest.raises(ValueError, match=match):
        EffectorSpec(**{**base, **kwargs})


def test_brand_tokens_are_whole_words() -> None:
    assert find_brand_tokens("A LEAP Hand and an allegro 2F-85") == ["leap hand", "allegro", "2f-85"]
    assert find_brand_tokens("Shadow Hand style, 24 joints") == ["shadow hand"]
    assert find_brand_tokens("anodised aluminium; soft shadows; inspired pads; 4-finger hand") == []


def test_param_space_and_recipe() -> None:
    space = (
        ParamSpec("n_fingers", "int", "Number of fingers around the palm.", "count", 2, 5, default=3),
        ParamSpec("link_mm", "float", "Length of each phalanx.", "mm", 10.0, 60.0),
        ParamSpec("pad", "choice", "Pad shape on the distal phalanx.", choices=("flat", "round")),
        ParamSpec("thumb", "bool", "Whether one finger opposes the others."),
    )
    schema = params_json_schema(space)
    assert (
        schema["required"] == ["n_fingers", "link_mm", "pad", "thumb"] and not schema["additionalProperties"]
    )
    assert schema["properties"]["n_fingers"] == {
        "type": "integer",
        "minimum": 2,
        "maximum": 5,
        "description": "Number of fingers around the palm. (count)",
    }
    params = {"n_fingers": 3, "link_mm": 30.0, "pad": "round", "thumb": True}
    r1 = MorphologyRecipe(EffectorFamily.DESIGNED, "fingered_hand", params, hypothesis="a", origin="nemotron")
    r2 = MorphologyRecipe(
        "designed", "fingered_hand", dict(reversed(list(params.items()))), "b", origin="random"
    )
    assert r1.recipe_id == r2.recipe_id  # hypothesis and origin are not hashed
    assert r1.problems(space) == []
    bad = MorphologyRecipe("designed", "fingered_hand", {**params, "n_fingers": 7, "extra": 1})
    assert sorted(bad.problems(space)) == ["n_fingers must be in [2, 5] count", "unknown parameter 'extra'"]
    assert (
        MorphologyRecipe("designed", "fingered_hand", {"pad": "round"})
        .problems(space)[0]
        .startswith("missing")
    )
    spec = EffectorSpec(
        "designed", "fingered-hand", "Designed hand A", "power", {}, recipe=r1, origin="nemotron"
    )
    assert spec.group is EntrantGroup.DESIGNED and spec.to_dict()["recipe"]["recipe_id"] == r1.recipe_id
    rand = EffectorSpec(
        "designed", "fingered-hand", "Designed hand B", "power", {}, recipe=r2, origin="random"
    )
    assert rand.group is EntrantGroup.RANDOM
    with pytest.raises(ValueError, match="does not match"):
        EffectorSpec("designed", "fingered-hand", "Designed hand C", "power", {}, recipe=r1, origin="random")
    assert pickle.loads(pickle.dumps(spec)).effector_id == spec.effector_id
    with pytest.raises(ValueError):
        ParamSpec("x", "float", "no bounds")


def test_asset_source_enforces_the_licence_rules() -> None:
    ok = {
        "path": "assets/effectors/catalog/hand/right_hand.xml",
        "sha256": SHA,
        "upstream_url": "https://github.com/example/models",
        "upstream_commit": "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35",
        "upstream_path": "hand/right_hand.xml",
        "upstream_sha256": SHA,
        "license": "BSD-2-Clause",
        "copyright": "Copyright (c) 2016 Example",
        "attribution": "Example hand model, BSD-2-Clause",
    }
    assert AssetSource(**ok).to_dict()["license"] == "BSD-2-Clause"
    for lic in ("CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "GPL-3.0-only"):
        with pytest.raises(ValueError, match="may not be vendored"):
            AssetSource(**{**ok, "license": lic})
    with pytest.raises(ValueError, match="changes"):
        AssetSource(**{**ok, "sha256": "1" * 64})
    assert AssetSource(**{**ok, "sha256": "1" * 64, "changes": ("prefixed names",)}).changes
    with pytest.raises(ValueError, match="caveat"):
        AssetSource(**{**ok, "status": "amber"})
    with pytest.raises(ValueError, match="repository-relative"):
        AssetSource(**{**ok, "path": "/abs/file.xml"})


def test_effector_build_validation() -> None:
    spec = toy_spec()
    good = {
        "spec": spec,
        "bodies": ("eff_base",),
        "actuators": ("eff_a",),
        "ctrl_open": (0.0,),
        "reach": 0.05,
        "tcp": (0, 0, -0.03),
        "actuation": {"grip_force_n": 25},
    }
    assert EffectorBuild(**good).n_actuators == 1
    for change, match in (
        ({"bodies": ("base",)}, "eff_"),
        ({"actuators": ("eff_a", "eff_a"), "ctrl_open": (0, 0)}, "distinct"),
        ({"ctrl_open": (0.0, 1.0)}, "ctrl_open"),
        ({"reach": 0.0}, "reach"),
        ({"actuation": {}}, "actuation"),
        ({"mass": 12.0}, "mass"),
    ):
        with pytest.raises(ValueError, match=match):
            EffectorBuild(**{**good, **change})
    assert Engagement(True).ok
    with pytest.raises(ValueError, match="unknown"):
        Engagement(False, ("slipped",))
    ref = RegistrationReference.analytic(math.pi / 2)
    assert ref.source == "analytic" and abs(np.linalg.norm(ref.quat_wxyz) - 1.0) < 1e-12


# --------------------------------------------------------------------------- registry


def test_registry_loads_the_reference_by_module_name() -> None:
    impl = registry.get_family("parallel_jaw")
    assert isinstance(impl, ParallelJawReference) and impl.fidelity is Fidelity.CORE
    assert impl.reference_mode == "analytic"
    assert [s.model for s in impl.presets()] == ["stock", "flat", "imprint-configurator"]
    assert all(not find_brand_tokens(s.display_name) for s in impl.presets())
    with pytest.raises(ValueError, match="already registered"):
        registry.register_family(ParallelJawReference())
    with pytest.raises(ValueError):
        registry.get_family("gecko_pad")


def test_registry_reports_a_family_that_is_not_built(monkeypatch: pytest.MonkeyPatch) -> None:
    fam = EffectorFamily.SUCTION
    previous = registry._REGISTRY.pop(fam, None)
    monkeypatch.setattr(registry, "FAMILY_MODULES", {fam: "rtt.effectors._no_such_family_module"})
    try:
        with pytest.raises(registry.UnknownFamilyError, match="not built yet"):
            registry.get_family(fam)
    finally:
        if previous is not None:
            registry._REGISTRY[fam] = previous


def test_override_family_restores_the_previous_registration() -> None:
    before = registry.registered_families()
    with registry.override_family(ToyTwoPad()) as impl:
        assert registry.get_family(EffectorFamily.CATALOG_GRIPPER) is impl
    assert registry.registered_families() == before


# --------------------------------------------------------------------------- the shared scene


def test_scene_keeps_the_core_world_wrist_and_part(toy: Any, task: Any, block: Any) -> None:
    from rtt.effectors.registry import get_family

    scene = build_effector_scene(toy, toy_spec(), block, task)
    m = scene.model
    stock = get_family("parallel_jaw").pair_for(jaw_spec("stock"), block, task)  # type: ignore[attr-defined]
    core, core_ids = build_model_v2(stock, block.to_mesh_object(), task)
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations", "disableflags"):
        assert getattr(m.opt, name) == getattr(core.opt, name), name
    for joint in ("palm_x", "palm_y", "palm_z", "palm_yaw"):
        a, b = m.joint(joint), core.joint(joint)
        assert np.array_equal(a.range, b.range) and np.array_equal(a.armature, b.armature), joint
        x, y = m.actuator(joint), core.actuator(joint)
        for field in ("gainprm", "biasprm", "forcerange", "ctrlrange"):
            assert np.array_equal(getattr(x, field), getattr(y, field)), (joint, field)
    assert m.body_gravcomp[scene.ids.palm_body] == core.body_gravcomp[core_ids.palm_body] == 1.0
    assert math.isclose(
        m.body_subtreemass[scene.ids.palm_body] - scene.build.mass, core.body_mass[core_ids.palm_body]
    )
    obj, cobj = m.body("object").id, core.body("object").id
    assert math.isclose(m.body_mass[obj], core.body_mass[cobj])
    assert all(m.geom_priority[g] == 1 for g in scene.ids.object_geoms)
    for g in ("table", "floor"):
        assert np.array_equal(m.geom(g).size, core.geom(g).size)
    assert m.geom(FLANGE_GEOM).size[1] == -MOUNT_Z
    assert all(m.body_gravcomp[b] == 1.0 for b in scene.ids.effector_bodies)
    assert math.isclose(scene.build.mass, TOY_MASS) and scene.build.com is not None
    assert scene.ids.replay_names == ("palm", "eff_base", "eff_left", "eff_right", "object")
    assert list(scene.ids.palm_actuators) == [0, 1, 2, 3] and list(scene.ids.effector_actuators) == [4, 5]


class _Broken(ToyTwoPad):
    def __init__(self, how: str) -> None:
        self.how = how

    def build(self, ctx: Any) -> EffectorBuild:
        b = super().build(ctx)
        if self.how == "name":
            ctx.mount.add_body(name="stray").add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0])
        elif self.how == "world":
            ctx.spec.worldbody.add_body(name="eff_loose").add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0]
            )
        elif self.how == "actuators":
            return EffectorBuild(**{**b.__dict__, "actuators": b.actuators[::-1]})
        elif self.how == "wrist_actuator":
            act = ctx.spec.add_actuator(name="eff_lift", target="palm_z", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_motor()
            extra = {"actuators": (*b.actuators, "eff_lift"), "ctrl_open": (*b.ctrl_open, 0.0)}
            return EffectorBuild(**{**b.__dict__, **extra})
        elif self.how == "weld":
            ctx.spec.add_equality(
                name="eff_glue",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1="eff_left",
                name2="world",
            )
        elif self.how == "priority":
            ctx.spec.geom("eff_left_pad").priority = 2
        elif self.how == "sticky":
            ctx.spec.geom("eff_left_pad").adhesion = 5.0
        elif self.how == "margin":
            ctx.spec.geom("eff_left_pad").margin = 0.002
        elif self.how == "option":
            ctx.spec.option.impratio = 100.0
        elif self.how == "default":
            ctx.spec.default.geom.condim = 6
        return b


@pytest.mark.parametrize(
    ("how", "match"),
    [
        ("name", "prefix"),
        ("world", "under the wrist"),
        ("actuators", "declared"),
        ("wrist_actuator", "acts outside the effector"),
        ("weld", "ties the effector to the scene"),
        ("priority", "priority"),
        ("sticky", "geom adhesion"),
        ("margin", "contact margin"),
        ("option", "option settings"),
        ("default", "default geom settings"),
    ],
)
def test_scene_refuses_a_family_that_breaks_the_contract(how: str, match: str, task: Any, block: Any) -> None:
    impl = _Broken(how)
    with registry.override_family(impl), pytest.raises(ContractError, match=match):
        build_effector_scene(impl, toy_spec(), block, task)


def test_an_unbuildable_effector_scores_invalid_build(task: Any, block: Any) -> None:
    class Narrow(ToyTwoPad):
        def build(self, ctx: Any) -> EffectorBuild:
            raise eff.InvalidEffector("no flat face under the cup")

    with registry.override_family(Narrow()):
        r = eff.run_effector_episode(toy_spec(), block, objs.Placement(), 0, task=task)
    assert (r.failure_mode, r.success, r.held, r.n_steps) == ("invalid_build", False, False, 0)
    assert r.flags[0].startswith("invalid_build: no flat face")
    json.dumps(r.to_record(), allow_nan=False)


# --------------------------------------------------------------------------- the shared runner


def test_native_episode_end_to_end(toy: Any, task: Any, block: Any) -> None:
    spec = toy_spec()
    assert isinstance(toy.primitive(spec, None), GraspPrimitive)  # type: ignore[arg-type]
    assert isinstance(toy.criterion(spec, None), GraspCriterion)  # type: ignore[arg-type]
    cond = next(c for c in train_suite() if c.part.name == "block")
    r = eff.run_effector_episode(
        spec, block, cond.placement, cond.seeds[0], task=task, record=True, condition_id=cond.condition_id
    )
    assert r.held and r.engaged and r.success and r.failure_mode is None
    assert r.bad_qacc == 0 and r.penetration_mm < 1.0 and r.flags == ()
    assert r.reference == "analytic" and r.reg_pos is not None and r.reg_quat_wxyz is not None
    assert math.isclose(r.effector_mass_kg, TOY_MASS) and r.metrics["eng_bodies"] == 2.0
    again = eff.run_effector_episode(spec, block, cond.placement, cond.seeds[0], task=task)
    assert again.traj_sha == r.traj_sha and len(r.traj_sha) == 64  # recording never changes the physics
    scene = build_effector_scene(toy, spec, block, task)
    reused = run_native_episode(toy, spec, block, cond.placement, cond.seeds[0], task=task, scene=scene)
    assert reused.traj_sha == r.traj_sha
    job = EffectorJob(spec, block, cond.placement, cond.seeds[0], task=task)
    assert run_effector_job(job).traj_sha == r.traj_sha == run_effector_job(job).traj_sha  # cached scene

    rec = r.to_record()
    json.dumps(rec, allow_nan=False)
    assert (
        rec["label"] == PRODUCT_MODE_LABEL and rec["family"] == "catalog_gripper" and rec["success"] is True
    )

    # the replay lifts to a valid version 2 lane
    replay = r.replay
    assert replay is not None and r.overlays is not None
    assert replay.frames.shape[1] == 5 and set(r.overlays) == set(rp.OVERLAY_CHANNELS)
    meta = replay.header["meta"]
    assert meta["effector"]["display_name"] == spec.display_name and meta["design_id"] == spec.effector_id
    header = rp.upgrade_header(
        replay.header,
        visuals={},
        assets={},
        meta={"geometry_id": None},
        provenance=rp.provenance(run_id="effectors-contract", code_version="89b37f3", product_mode=True),
    )
    assert rp.check_header(header, replay.frames) == []
    assert Validator(json.loads(rp.SCHEMA_PATH.read_text(encoding="utf-8"))).errors(header) == []
    assert not find_brand_tokens(json.dumps(header))


def test_primitive_must_return_one_control_per_actuator(task: Any, block: Any) -> None:
    class Bad(ToyTwoPad):
        def primitive(self, spec: Any, build: Any) -> Any:
            prim = super().primitive(spec, build)

            class Wrapped:
                name = "bad"

                def plan(self, *a: Any) -> Any:
                    return prim.plan(*a)

                def command(self, *a: Any) -> np.ndarray:
                    return np.zeros(3)

            return Wrapped()

    with registry.override_family(Bad()), pytest.raises(ContractError, match="controls"):
        eff.run_effector_episode(toy_spec(), block, objs.Placement(), 0, task=task)


def test_primitives_may_not_apply_forces_directly(task: Any, block: Any) -> None:
    class Pushy(ToyTwoPad):
        def primitive(self, spec: Any, build: Any) -> Any:
            prim = super().primitive(spec, build)

            class Wrapped:
                name = "pushy"

                def plan(self, *a: Any) -> Any:
                    return prim.plan(*a)

                def command(
                    self, t: float, phase: Any, closure: float, model: Any, data: Any, scene: Any
                ) -> Any:
                    data.xfrc_applied[scene.ids.palm_body, 2] = 5.0
                    return prim.command(t, phase, closure, model, data, scene)

            return Wrapped()

    with registry.override_family(Pushy()), pytest.raises(ContractError, match="applied forces"):
        eff.run_effector_episode(toy_spec(), block, objs.Placement(), 0, task=task)


def test_native_certificate_uses_the_nominal_episode_as_reference(toy: Any, task: Any, block: Any) -> None:
    cert = certify_effector(toy_spec(), block, task=task, workers=1)
    assert cert.reference == "nominal_episode" and len(cert.tests) == 18
    nominal = cert.tests[0]
    assert nominal.name == "nominal" and nominal.held and nominal.registered and nominal.offset_mm < 1e-6
    assert set(cert.held_scores) == {"placement", "shake", "push"}
    assert cert.held_tolerance_mm["y"] is not None and all(t.bad_qacc == 0 for t in cert.tests)
    d = cert.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["label"] == PRODUCT_MODE_LABEL and d["fidelity"] == "rigid" and d["held"].endswith("/18")


# --------------------------------------------------------------------------- the reference family


def test_reference_family_is_the_frozen_v2_episode(task: Any) -> None:
    part = get_part("l_bracket")
    cond = next(c for c in train_suite() if c.part.name == "l_bracket" and c.placement_label == "c2")
    spec = jaw_spec("stock")
    r = eff.run_effector_episode(
        spec, part, cond.placement, cond.seeds[0], task=task, condition_id=cond.condition_id
    )
    pair = registry.get_family("parallel_jaw").pair_for(spec, part, task)  # type: ignore[attr-defined]
    v2 = run_episode_v2(pair, part, cond.placement, cond.seeds[0], record=False, task=task)
    assert r.traj_sha == v2.traj_sha and len(r.traj_sha) == 64
    assert (r.success, r.held, r.failure_mode, r.engaged) == (
        v2.success,
        v2.lift_success,
        v2.failure_mode,
        v2.registration.seated,
    )
    assert r.registration_flags == v2.registration.sub_flags and r.reference == "analytic"
    assert r.spec.fidelity is Fidelity.CORE and r.to_record()["group"] == "reference"


def test_reference_certificate_equals_certify_v2(task: Any) -> None:
    part = get_part("l_bracket")
    spec = jaw_spec("imprint_configurator")
    mine = certify_effector(spec, part, task=task, workers=4)
    pair = registry.get_family("parallel_jaw").pair_for(spec, part, task)  # type: ignore[attr-defined]
    frozen = certify_v2(pair, part, task=task, workers=4)
    assert [(t.name, t.held, t.registered, t.failure_mode) for t in mine.tests] == [
        (t.name, t.held, t.registered, t.failure_mode) for t in frozen.tests
    ]
    assert mine.held_tolerance_mm == frozen.held_tolerance_mm
    assert mine.registered_tolerance_mm == frozen.registered_tolerance_mm
    assert mine.held_score == frozen.held_score and mine.registered_score == frozen.registered_score


def test_jobs_are_identical_in_process_and_in_workers(task: Any) -> None:
    part = get_part("hex_standoff")
    conds = [c for c in train_suite() if c.part.name == "hex_standoff"][:2]
    jobs = [EffectorJob(jaw_spec("flat"), part, c.placement, c.seeds[0], task=task) for c in conds]
    here = run_effector_jobs(jobs, workers=1)
    there = run_effector_jobs(jobs, workers=2)
    assert [r.traj_sha for r in here] == [r.traj_sha for r in there]
    assert [r.to_record()["success"] for r in here] == [r.to_record()["success"] for r in there]


def test_an_unbuildable_jaw_scores_invalid_build(task: Any, block: Any) -> None:
    spec = EffectorSpec(
        "parallel_jaw",
        "nope",
        "Parallel jaw, missing entrant",
        "jaw",
        {"entrant": "nope"},
        origin="reference",
    )
    r = eff.run_effector_episode(spec, block, objs.Placement(), 0, task=task)
    assert r.failure_mode == "invalid_build" and "unknown v2 entrant" in r.flags[0]


# --------------------------------------------------------------------------- frozen contract and brands

#: sha256 of each frozen contract file. Only the contract owner updates these (docs/EFFECTORS.md 11.6).
CONTRACT_FILES: dict[str, str] = {
    "__init__.py": "5136c825b4542174e4d2ef3de9e27f62e09426005142bb914c04136e931bf226",
    "types.py": "2281e0de232e376346fe29f01be65669af99a2b14cc2aaab23cce5e829245258",
    "registry.py": "9415e10df8a443a1a4ad55f360764103ec745073e3563cdad025c6940782d58f",
    "scene.py": "d6a35fb47d220ea740b23ad79066dee46a23b5f0ee518b51acee9773e36b49f9",
    "runner.py": "420b01ab7a556384fa1f68cbb049d666d8535b1532c2fdeec2f2e422f1c73639",
    "certificate.py": "62f209a8d603e7fabdc9fda7b8b4ccabd21f88f7e9eb027ec0b6e74fcf64ad9c",
    "primitives.py": "3c90c0259ab9749242665b4f78356eb3b0a4e137a583e11ca86727c6c594bc03",
    "parallel_jaw.py": "a01aca3faf209e6253dbd73263ba8f4c3c562155a7894e4311d72b1a67638e7c",
}

#: Where brand and product names may appear (docs/EFFECTORS.md section 8).
BRAND_ALLOWED_FILES = {
    "README.md",
    "NOTICE",
    "docs/ATTRIBUTION.md",
    "docs/DEVIATIONS.md",
    "docs/EFFECTORS.md",
    "web/app/credits/page.tsx",
    "web/scripts/smoke_v2.mjs",
    "src/rtt/effectors/types.py",
}
BRAND_ALLOWED_PREFIXES = ("tests/", "assets/gripper/", "assets/effectors/")
_URL = re.compile(r"https?://[^\s\"')>\]]+")


def test_contract_files_are_frozen() -> None:
    root = Path(eff.__file__).parent
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CONTRACT_FILES}
    assert current == CONTRACT_FILES, "a frozen contract file changed: use a contract change request"


def test_brand_tokens_appear_only_in_credit_files() -> None:
    import subprocess

    if not (REPO / ".git").exists():
        pytest.skip("needs the git checkout")
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hits = {}
    for rel in listed:
        if rel in BRAND_ALLOWED_FILES or rel.startswith(BRAND_ALLOWED_PREFIXES):
            continue
        path = REPO / rel
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = find_brand_tokens(_URL.sub(" ", text))  # a source URL is a pin, not an on-screen credit
        if found:
            hits[rel] = sorted(set(found))
    assert hits == {}
