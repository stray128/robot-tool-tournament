"""Track S5d: part-shaped parallel jaws in the Studio (product mode, not the pre-registered result).

The v2 fingers on the open 84 mm parallel gripper built under the Studio mount against the object a
step touches (:mod:`rtt.effectors.part_jaw`), their place in the S5 candidate pool and pruning
(:mod:`rtt.studio.search`) and their printed jaws in a build pack (:mod:`rtt.studio.buildpack.part_jaw`).
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import trimesh

from rtt.effectors import part_jaw as pj
from rtt.effectors.parallel_jaw import ParallelJawReference
from rtt.effectors.registry import get_family
from rtt.effectors.tournament import spec_from_dict
from rtt.effectors.types import ALLOWED_LICENSES, PRODUCT_MODE_LABEL, InvalidEffector, find_brand_tokens
from rtt.parts.library import get_part
from rtt.sim.task import default_task_v2
from rtt.studio.episode import StudioJob, run_task
from rtt.studio.plan import get_primitive, override_primitive
from rtt.studio.runner import WorkerPool, condition_grid
from rtt.studio.scene import build_studio_scene
from rtt.studio.search import (
    candidate_of,
    default_candidates,
    part_jaw_candidates,
    probe_tool,
    prune,
    run_task_tournament,
)
from rtt.studio.types import (
    BracketSpec,
    FidelityTier,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    SetupKind,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
)
from studio_fixtures import gripper, preset

#: The H2 coupling of the recorded S4c plan (gate-2026-10-18-s4c, h2_coupling).
H2_COUPLING: dict[str, Any] = {
    "chamfer_mm": 1,
    "diameter_mm": 40,
    "engage_turns": 3,
    "final_torque_nm": 8,
    "length_mm": 45,
    "member": "coupling",
    "pitch_mm": 1.5,
    "start_torque_nm": 0.2,
}


def _coupling(name: str = "pipe_coupling", **params: Any) -> ObjectInstance:
    return ObjectInstance.of(name, "threaded_joint", "Pipe coupling on stub", **{**H2_COUPLING, **params})


def _box(name: str, x: float, y: float, z: float, mass: float, at: tuple[float, float] = (120.0, 0.0)):
    return ObjectInstance.of(
        name,
        "part",
        name.replace("_", " ").capitalize(),
        ObjectPose(*at, 0.0),
        shape="box",
        size_x_mm=x,
        size_y_mm=y,
        size_z_mm=z,
        mass_g=mass,
    )


def _twist_task(coupling: ObjectInstance | None = None, degrees: float = 20.0) -> TaskSpec:
    obj = coupling or _coupling()
    step = Step.of(
        "tighten",
        "twist",
        obj.name,
        "member",
        "Turn the coupling on",
        [SuccessCriterion("angle_deg", at_least=degrees)],
        params={"direction": "cw", "speed_deg_s": 60},
        constraints=StepConstraints(max_torque_nm=15.0),
    )
    return TaskSpec("Coupling twist", "", (obj,), (step,), source="test")


def _pick_task() -> TaskSpec:
    box = _box("small_box", 30.0, 30.0, 40.0, 60.0)
    step = Step.of(
        "pick_box",
        "grasp",
        "small_box",
        "part",
        "Lift the box",
        [SuccessCriterion("lift_mm", at_least=20.0)],
        params={"lift_mm": 40, "hold_s": 0.5},
    )
    return TaskSpec("Box pick", "", (box,), (step,), source="test")


def _imprint(obj: ObjectInstance):
    return pj.part_jaw_spec(obj, "imprint_configurator")


# --------------------------------------------------------------------------- the family slot


def test_the_family_slot_keeps_every_frozen_behaviour() -> None:
    impl = get_family("parallel_jaw")
    assert isinstance(impl, pj.StudioPartJaw) and isinstance(impl, ParallelJawReference)
    frozen = ParallelJawReference()
    assert [s.effector_id for s in impl.presets()] == [s.effector_id for s in frozen.presets()]
    assert impl.fidelity is frozen.fidelity and impl.reference_mode == frozen.reference_mode
    task, block = default_task_v2(), get_part("block")
    stock = impl.presets()[0]
    assert (
        impl.pair_for(stock, block, task).geometry_hash == frozen.pair_for(stock, block, task).geometry_hash
    )
    # a pre-registered entrant still runs only in the frozen v2 episode
    with pytest.raises(NotImplementedError):
        build_studio_scene(Setup.single(stock, ("pick_box",)), _pick_task(), 0)
    assert "frozen v2 episode" in (probe_tool(stock).reason or "")
    # and a Studio jaw is not a v1 entrant: the delegated v2 episode refuses it
    spec = _imprint(_coupling())
    with pytest.raises(ValueError, match="params.entrant or params.recipe"):
        impl.pair_for(spec, block, task)


def test_specs_name_their_object_round_trip_and_carry_no_brand() -> None:
    task = _twist_task()
    specs = pj.part_jaw_specs(task)
    assert [pj.jaw_target(s)["entrant"] for s in specs] == list(pj.STUDIO_ENTRANTS)
    assert len({s.effector_id for s in specs}) == 3
    for s in specs:
        assert pj.is_part_jaw(s) and s.model.startswith("studio-") and s.primitive == "jaw"
        assert len(s.display_name) <= 60 and not find_brand_tokens(s.display_name)
        t = pj.jaw_target(s)
        assert (t["object"], t["kind"], t["handle"]) == ("pipe_coupling", "threaded_joint", "member")
        assert spec_from_dict(s.to_dict()).effector_id == s.effector_id
        assert s.to_dict()["label"] == PRODUCT_MODE_LABEL
    # a different coupling is a different jaw
    other = _imprint(_coupling(diameter_mm=41))
    assert other.effector_id != specs[0].effector_id
    for kind, handle_kw in (("handwheel_valve", {}), ("hinged_lid", {}), ("bin", {})):
        inst = ObjectInstance.of("thing", kind, "Thing", **handle_kw)
        with pytest.raises(ValueError, match="no part-shaped jaw"):
            pj.part_jaw_spec(inst)


def test_specs_pickle_for_worker_processes_and_others_pickle_as_before() -> None:
    from multiprocessing.reduction import ForkingPickler

    spec = _imprint(_coupling())
    back = pickle.loads(bytes(ForkingPickler.dumps(spec)))
    assert back.effector_id == spec.effector_id and back.display_name == spec.display_name
    assert b"_restore_spec" in bytes(ForkingPickler.dumps(spec))
    plain = gripper()
    assert b"_restore_spec" not in bytes(ForkingPickler.dumps(plain))
    assert pickle.loads(bytes(ForkingPickler.dumps(plain))).effector_id == plain.effector_id


# --------------------------------------------------------------------------- the gripped feature


def _turned(points: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return points @ r.T


def _same_sets(a: list[np.ndarray], b: list[np.ndarray], tol: float = 2e-7) -> bool:
    """Every piece of ``a`` has a piece of ``b`` with the same vertex set (within ``tol``)."""

    def key(p: np.ndarray) -> np.ndarray:
        q = np.unique(np.round(np.asarray(p) / tol).astype(np.int64), axis=0)
        return q

    kb = [key(p) for p in b]
    for p in a:
        ka = key(p)
        if not any(ka.shape == k.shape and np.abs(ka - k).max() <= 2 for k in kb):
            return False
    return len(a) == len(b)


def test_the_gripped_coupling_is_the_simulated_member_turned_onto_the_closing_axis() -> None:
    from rtt.studio.templates.threaded_joint import _ring

    inst = _coupling()
    feature = pj.gripped_feature(inst)
    d, length = 0.040, 0.045
    r_out = 0.5 * d
    expected = _ring(12, r_out - max(0.0015, 0.15 * d), r_out, 0.0, length)  # the builder's own pieces
    lo = np.concatenate(expected).min(axis=0)
    hi = np.concatenate(expected).max(axis=0)
    centre = 0.5 * (lo + hi)
    want = [_turned(p - centre, 90.0) for p in expected]
    assert _same_sets([np.asarray(p) for p in feature.part.pieces], want)
    assert feature.grip_tip_m == pytest.approx(pj.MEMBER_TIP_ABOVE_BOTTOM)
    assert feature.part.mass == pytest.approx(
        2000.0 * length * 12 * math.tan(math.pi / 12) * (r_out**2 - (r_out - 0.006) ** 2), rel=1e-6
    )
    # a hexagon nut is turned so that a flat faces each jaw: y spans the flats, x the corners
    nut = pj.feature_part(_coupling("nut", member="nut", diameter_mm=27.0))
    assert nut.size[0] == pytest.approx(0.027, abs=1e-6)
    assert nut.size[1] == pytest.approx(0.027 * math.cos(math.pi / 6), abs=1e-6)


def test_plug_and_latch_features_and_their_grip_heights() -> None:
    from rtt.studio.primitives.insert import grip_tip

    task = default_task_v2()
    plug = ObjectInstance.of("plug", "port_plug", "Plug", plug_diameter_mm=20.0)
    f = pj.gripped_feature(plug)
    assert f.part.size[0] == pytest.approx(0.020, rel=2e-3) and f.part.size[2] == pytest.approx(0.040)
    info = {"depth_m": 0.015, "plug_d_m": 0.020}
    assert f.grip_tip_m == pytest.approx(
        grip_tip(info, 0.040, task.physics.friction, task.controller.min_tip_clearance)
    )
    wing = pj.gripped_feature(ObjectInstance.of("latch", "quarter_turn_latch", "Latch"))
    assert wing.part.size[0] == pytest.approx(0.025, abs=1e-6)  # the fin along x, gripped across (y)
    assert wing.part.size[1] == pytest.approx(0.5 * 0.025, rel=2e-3)  # the boss, 25 % of the diameter
    assert wing.grip_tip_m == pytest.approx(0.002, abs=1e-6)


def test_a_jaw_that_cannot_be_built_says_why() -> None:
    wide = _box("flat_pack", 150.0, 100.0, 20.0, 150.0)
    with pytest.raises(InvalidEffector, match="too wide for the opening"):
        pj.studio_pair(pj.part_jaw_spec(wide, "flat"))
    carton = _box("carton", 120.0, 80.0, 60.0, 300.0)
    with pytest.raises(InvalidEffector, match="open_clearance"):
        pj.studio_pair(_imprint(carton))
    assert pj.studio_pair(pj.part_jaw_spec(carton, "stock")).q_seat > 0.0


# --------------------------------------------------------------------------- the Studio build


def test_jaws_build_under_the_mount_with_the_v2_drive() -> None:
    task = default_task_v2()
    g, c = task.gripper, task.controller
    tt = _twist_task()
    spec = _imprint(tt.objects[0])
    pair = pj.studio_pair(spec)
    scene = build_studio_scene(Setup.single(spec, tt.step_ids), tt, 0)
    m = scene.model
    b = scene.active.build
    assert b.actuators == ("eff_pj_grip",) and b.bodies == ("eff_pj_housing", "eff_pj_left", "eff_pj_right")
    assert m.body("eff_pj_left").mass[0] == pytest.approx(pair.left.mass)
    assert m.body("eff_pj_right").mass[0] == pytest.approx(pair.right.mass)
    assert m.body("eff_pj_housing").mass[0] == pytest.approx(g.palm_mass)
    assert b.mass == pytest.approx(g.palm_mass + pair.left.mass + pair.right.mass)
    act = m.actuator("eff_pj_grip")
    assert act.gainprm[0] == pytest.approx(g.grip_kp) and act.biasprm[2] == pytest.approx(-g.grip_kv)
    assert list(act.forcerange) == pytest.approx([-g.grip_force, g.grip_force])
    assert list(act.ctrlrange) == pytest.approx([c.open_target, g.stroke + g.grip_overdrive])
    for side in ("left", "right"):
        assert list(m.joint(f"eff_pj_{side}").range) == pytest.approx([0.0, g.stroke])
    assert m.equality("eff_pj_coupling").type == mujoco.mjtEq.mjEQ_JOINT
    assert m.tendon("eff_pj_grip").id >= 0
    # root plane 24 mm below the mount plane (the housing), jaws at the 84 mm opening
    data = mujoco.MjData(m)
    mujoco.mj_kinematics(m, data)
    palm = data.xpos[scene.ids.palm_body]
    assert data.xpos[m.body("eff_pj_left").id][2] - palm[2] == pytest.approx(-0.012 - 0.024)
    assert data.xpos[m.body("eff_pj_left").id][1] - palm[1] == pytest.approx(g.open_half_gap)
    assert b.max_opening == pytest.approx(0.0848, abs=1e-5)
    assert b.notes[0] == PRODUCT_MODE_LABEL and "not the pre-registered result" in b.notes[1]
    assert not b.sources  # generated geometry
    stock = build_studio_scene(Setup.single(pj.part_jaw_spec(tt.objects[0], "stock"), tt.step_ids), tt, 0)
    srcs = stock.active.build.sources
    assert srcs and all(s.license in ALLOWED_LICENSES and s.status == "green" for s in srcs)


def test_jaws_build_on_a_tool_changer_and_on_a_two_tool_head() -> None:
    coupling = _coupling()
    box = _box("box", 30.0, 30.0, 40.0, 60.0)
    steps = (
        _twist_task(coupling).steps[0],
        Step.of("press", "push", "box", "part", "Press the box", [SuccessCriterion("force_n", at_least=1.0)]),
    )
    task = TaskSpec("Two tools", "", (coupling, box), steps, source="test")
    jaw = _imprint(coupling)
    tc = Setup.tool_change({"tighten": jaw, "press": gripper()})
    k = tc.tools.index(jaw)
    sc = build_studio_scene(tc, task, k)
    assert sc.active.build.actuators == ("eff_pj_grip",)
    assert sc.model.equality("eff_pj_coupling").id >= 0
    multi = Setup(
        SetupKind.MULTI_HEAD,
        (jaw, gripper()),
        {"tighten": 0, "press": 1},
        mounts=(HeadMount(-45.0, 70.0), HeadMount(45.0, 70.0)),
        bracket=BracketSpec(),
    )
    for active in (0, 1):
        sc = build_studio_scene(multi, task, active)
        head = next(h for h in sc.heads if h.tool == 0)  # the jaw: active, or idle with eff_h0_
        assert head.prefix == ("" if active == 0 else "eff_h0_")
        names = {sc.model.equality(i).name for i in range(sc.model.neq)}
        assert head.name("eff_pj_coupling") in names
        assert sc.model.tendon(head.name("eff_pj_grip")).id >= 0
        assert sc.model.actuator(head.name("eff_pj_grip")).id >= 0


@dataclass(frozen=True)
class _Recording:
    """The real S2 twist controller, recording the jaw closure when the turn starts."""

    inner: Any
    log: list
    primitive: PrimitiveKind = PrimitiveKind.TWIST
    name: str = "studio_twist"

    def plan(self, ctx, model, data):  # type: ignore[no-untyped-def]
        plan = self.inner.plan(ctx, model, data)
        self.log.append({"t_turn": plan.values["run"]["t_turn"]})
        return plan

    def command(self, t, ctx, plan, model, data):  # type: ignore[no-untyped-def]
        if t >= self.log[-1]["t_turn"] and "q" not in self.log[-1]:
            self.log[-1]["q"] = float(data.qpos[model.jnt_qposadr[model.joint("eff_pj_left").id]])
        return self.inner.command(t, ctx, plan, model, data)


def test_the_imprint_seats_on_the_flats_where_the_twist_grips() -> None:
    task = _twist_task()
    spec = _imprint(task.objects[0])
    pair = pj.studio_pair(spec)
    controller, predicate = get_primitive("twist")
    log: list = []
    with override_primitive(_Recording(controller, log), predicate):
        r = run_task(StudioJob(Setup.single(spec, task.step_ids), task, condition_grid(task, 1)[0]))
    # the jaws close into the imprint (seat closure), not onto its rim
    assert log and abs(log[0]["q"] - pair.q_seat) < 0.001
    out = r.steps[0]
    assert out.fidelity is FidelityTier.APPROXIMATION and any("Threads" in c for c in out.caveats)
    assert out.quantities["angle_deg"] > 5.0


def test_runs_are_deterministic_in_process_and_in_a_spawned_worker() -> None:
    task = _pick_task()
    spec = _imprint(task.objects[0])
    job = StudioJob(Setup.single(spec, task.step_ids), task, condition_grid(task, 2)[1])
    here = [run_task(job), run_task(job)]
    with WorkerPool(2) as pool:
        there = pool.map(run_task, [job, job])
    shas = {r.traj_sha for r in here + there}
    assert len(shas) == 1 and here[0].steps[0].success == there[0].steps[0].success
    assert here[0].steps[0].fidelity is FidelityTier.RIGID


# --------------------------------------------------------------------------- the search


def test_the_pool_replaces_the_frozen_presets_with_jaws_built_per_object() -> None:
    coupling = _coupling()
    box = _box("box", 30.0, 30.0, 40.0, 60.0)
    task = TaskSpec(
        "Pool",
        "",
        (coupling, box),
        (
            _twist_task(coupling).steps[0],
            Step.of(
                "press", "push", "box", "part", "Press the box", [SuccessCriterion("force_n", at_least=1.0)]
            ),
        ),
        source="test",
    )
    pool = default_candidates(library=False, designed=False, task_spec=task)
    jaws = [c for c in pool if c.spec.family.value == "parallel_jaw"]
    assert all(pj.is_part_jaw(c.spec) for c in jaws) and len(jaws) == 6
    assert {pj.jaw_target(c.spec)["object"] for c in jaws} == {"pipe_coupling", "box"}
    for c in jaws:
        imprint = pj.jaw_target(c.spec)["entrant"] == "imprint_configurator"
        assert c.off_the_shelf is not imprint and c.source == ("designed" if imprint else "preset")
        d = c.to_dict()
        assert d["fidelity"] == "rigid" and d["built_for"] in ("pipe_coupling", "box")
    # without a task the pool is as before (the pre-registered entrants, pruned in the Studio)
    plain = default_candidates(library=False, designed=False)
    assert not any(pj.is_part_jaw(c.spec) for c in plain)
    eligible, pruned, infos = prune(task, jaws)
    reasons = {(p.display_name, p.step_id): p.reason for p in pruned}
    for c in jaws:
        assert infos[c.effector_id].buildable, infos[c.effector_id].reason
        assert "frozen v2" not in (infos[c.effector_id].reason or "")
        other = "press" if pj.jaw_target(c.spec)["object"] == "pipe_coupling" else "tighten"
        assert "shaped and sized for" in reasons[(c.name, other)]
    assert {c.effector_id for c in eligible[0]} == {c.effector_id for c in jaws[:3]}


def test_pruning_keeps_the_generator_reason_for_a_jaw_that_cannot_open() -> None:
    carton = _box("carton", 120.0, 80.0, 60.0, 300.0)
    step = Step.of("pick", "grasp", "carton", "part", "Lift it", [SuccessCriterion("lift_mm", at_least=10.0)])
    task = TaskSpec("Carton", "", (carton,), (step,), source="test")
    jaws = part_jaw_candidates(task)
    _eligible, pruned, _infos = prune(task, jaws)
    by_entrant = {pj.jaw_target(c.spec)["entrant"]: c for c in jaws}
    reasons = {p.effector_id: p.reason for p in pruned}
    assert "open_clearance" in reasons[by_entrant["imprint_configurator"].effector_id]
    assert "open_clearance" in reasons[by_entrant["flat"].effector_id]
    assert by_entrant["stock"].effector_id not in reasons


# --------------------------------------------------------------------------- print files and packs


def test_print_files_are_one_watertight_solid_each_and_the_stock_blade_is_not_printed(tmp_path: Path) -> None:
    coupling = _coupling()
    facts = pj.jaw_print_files(_imprint(coupling), tmp_path, prefix="t_")
    assert sorted(facts) == ["t_left.stl", "t_mount_left.stl", "t_mount_right.stl", "t_right.stl"]
    for name, f in facts.items():
        mesh = trimesh.load(tmp_path / name, file_type="stl", force="mesh")
        mesh.merge_vertices()
        assert f["watertight"] and mesh.is_watertight and mesh.body_count == 1 and mesh.volume > 0
    with pytest.raises(ValueError, match="stock blade"):
        pj.jaw_print_files(pj.part_jaw_spec(coupling, "stock"), tmp_path)


def test_a_build_pack_ships_the_printed_jaws(tmp_path: Path) -> None:
    from rtt.studio.buildpack.part_jaw import export_part_jaw_pack

    box = _box("box", 30.0, 30.0, 40.0, 60.0)
    step = Step.of(
        "press", "push", "box", "part", "Press the box", [SuccessCriterion("force_n", at_least=1.0)]
    )
    task = TaskSpec("Press", "", (box,), (step,), source="test")
    cand = next(
        c for c in part_jaw_candidates(task) if pj.jaw_target(c.spec)["entrant"] == "imprint_configurator"
    )
    report = run_task_tournament(
        task,
        candidates=[cand],
        n_conditions=1,
        n_screen=1,
        n_finalists=1,
        workers=1,
        controllers="reference",
        kinds=("single",),
    )
    data = json.loads(json.dumps(report.to_dict()))
    setup_id = data["finalists"][0]["setup"]["setup_id"]
    pack = export_part_jaw_pack(data, tmp_path / "pack", setup_id=setup_id)
    paths = {f.path: f for f in pack.manifest.files}
    for side in ("left", "right", "mount_left", "mount_right"):
        f = paths[f"jaws/tool0_{side}.stl"]
        assert f.watertight and f.license == "CC0-1.0" and f.source == "generated"
    assert "jaws/JAWS.txt" in paths and "mjcf/setup_tool0.xml" in paths
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    assert {f["path"] for f in manifest["files"]} == set(paths)
    assert PRODUCT_MODE_LABEL in (tmp_path / "pack" / "jaws" / "JAWS.txt").read_text()
    assert not find_brand_tokens((tmp_path / "pack" / "jaws" / "JAWS.txt").read_text())
    import zipfile

    with zipfile.ZipFile(pack.zip_path) as zf:
        assert "pack/jaws/tool0_left.stl" in zf.namelist()
    assert all(m["verified"]["same_as_tested"] for m in pack.mjcf)
    assert candidate_of(cand.spec).spec.effector_id == cand.effector_id
    _ = preset  # the fixtures module is shared with the other Studio tests


def test_a_stock_blade_pack_links_upstream_and_labels_its_scene_with_the_blade_licence(
    tmp_path: Path,
) -> None:
    from rtt.studio.buildpack.part_jaw import export_part_jaw_pack

    box = _box("box", 30.0, 30.0, 40.0, 60.0)
    step = Step.of(
        "press", "push", "box", "part", "Press the box", [SuccessCriterion("force_n", at_least=1.0)]
    )
    task = TaskSpec("Press", "", (box,), (step,), source="test")
    cand = next(c for c in part_jaw_candidates(task) if pj.jaw_target(c.spec)["entrant"] == "stock")
    report = run_task_tournament(
        task,
        candidates=[cand],
        n_conditions=1,
        n_screen=1,
        n_finalists=1,
        workers=1,
        controllers="reference",
        kinds=("single",),
    )
    data = json.loads(json.dumps(report.to_dict()))
    pack = export_part_jaw_pack(data, tmp_path / "pack", setup_id=data["finalists"][0]["setup"]["setup_id"])
    files = {f.path: f for f in pack.manifest.files}
    assert not any(p.startswith("jaws/") for p in files)  # the blade is the gripper's own part
    assert files["mjcf/setup_tool0.xml"].license == "CC-BY-4.0"  # its hulls are inline in the scene
    assert not find_brand_tokens(files["mjcf/setup_tool0.xml"].attribution)
    assert "licences/tool0_LICENSE.txt" in files and pack.manifest.links
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    assert {f["path"]: f["license"] for f in manifest["files"]}["mjcf/setup_tool0.xml"] == "CC-BY-4.0"
