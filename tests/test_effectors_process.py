"""Studio v2: vacuum area frames and process tools (a nut runner and a tamper pad).

Each preset builds in the Studio scene, passes the attach gate, does the job it is rated for with
the safety factor (a frame lifts its rated payload, the nut runner tightens to its rated torque, the
tamper presses its rated force) and fails at 1.6 times the rating (1.6 > the safety factor 1.5, so the
rating's margin is gone). A wrong tool fails the same job (the Studio's S2 rule).

The Studio runs are in-process (the heavy panel is a stand-in part registered through the intake
resolver, which a spawned worker would not have) and use the frozen task, controllers and fairness
rules; nothing here changes a primitive.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Iterator
from typing import Any

import mujoco
import pytest

import rtt.parts.types as parts_types
import rtt.studio.templates.part as part_template
from rtt.effectors import heights
from rtt.effectors import process_tool as pt
from rtt.effectors import suction as sc
from rtt.effectors.heights import top_height
from rtt.effectors.primitives import part_height
from rtt.effectors.registry import available_families, get_family
from rtt.effectors.tournament import (
    GROUP_ORDER,
    PART_FAMILIES,
    ScreenCondition,
    family_status,
    per_part_presets,
    run_tournament,
)
from rtt.effectors.types import (
    FAMILY_FIDELITY,
    PRODUCT_MODE_LABEL,
    EffectorFamily,
    EffectorSpec,
    EntrantGroup,
    Fidelity,
    find_brand_tokens,
)
from rtt.parts.library import Material, build_part, get_part
from rtt.sim.objects import Placement
from rtt.sim.task import default_task_v2
from rtt.studio.episode import StudioCondition, StudioJob, run_task, run_task_jobs
from rtt.studio.runner import condition_grid
from rtt.studio.scene import build_studio_scene
from rtt.studio.search import ARM_PAYLOAD_KG, default_candidates, probe_scene, probe_tool
from rtt.studio.types import (
    STUDIO_WRIST,
    BracketSpec,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    Setup,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
)
from studio_fixtures import gripper

G = 9.81
FRAME_2X2, FRAME_2X3 = "vacuum-frame-2x2-300", "vacuum-frame-2x3-600"
NUT_RUNNER, TAMPER = "nut-runner-socket-13", "tamper-pad-200"
PRESETS = [
    ("suction", FRAME_2X2),
    ("suction", FRAME_2X3),
    ("process_tool", NUT_RUNNER),
    ("process_tool", TAMPER),
]


def spec_of(family: str, model: str) -> EffectorSpec:
    return next(s for s in get_family(family).presets() if s.model == model)


def _task_of(obj: ObjectInstance, step: Step) -> TaskSpec:
    return TaskSpec(f"process: {step.id}", "", (obj,), (step,), source="test")


def _run(
    spec: EffectorSpec, task: TaskSpec, seed: int = 0, offsets: dict[str, tuple[float, float]] | None = None
):
    job = StudioJob(Setup.single(spec, task.step_ids), task, StudioCondition("c", seed, offsets or {}))
    return run_task(job)


# --------------------------------------------------------------------------- the presets


@pytest.mark.parametrize(("family", "model"), PRESETS)
def test_presets_are_labelled_approximations_with_generic_names(family: str, model: str) -> None:
    impl = get_family(family)
    spec = spec_of(family, model)
    assert spec.origin == "preset" and spec.fidelity is Fidelity.APPROXIMATION
    assert not find_brand_tokens(spec.display_name) and len(spec.display_name) <= 60
    d = spec.to_dict()
    assert d["fidelity"] == "approximation" and d["label"] == PRODUCT_MODE_LABEL
    json.dumps(d, allow_nan=False)
    assert pickle.loads(pickle.dumps(spec)).effector_id == spec.effector_id
    caveat = impl.caveat.lower()
    assert "approximation" in caveat and "verify on hardware" in caveat


def test_the_process_family_is_a_registered_approximation_in_its_own_group() -> None:
    impl = get_family("process_tool")
    assert isinstance(impl, pt.ProcessToolFamily) and impl.family is EffectorFamily.PROCESS_TOOL
    assert FAMILY_FIDELITY[EffectorFamily.PROCESS_TOOL] is Fidelity.APPROXIMATION
    specs = impl.presets()
    assert [s.model for s in specs] == [NUT_RUNNER, TAMPER]
    assert [s.primitive for s in specs] == ["drive", "press"]
    assert all(s.group is EntrantGroup.PROCESS for s in specs)
    # a tool that does a process never competes in the per-part pick tournament
    assert EntrantGroup.PROCESS not in GROUP_ORDER and len(GROUP_ORDER) == len(EntrantGroup) - 1
    models = {c.spec.model for c in default_candidates(library=False, designed=False)}
    assert {FRAME_2X2, FRAME_2X3, NUT_RUNNER, TAMPER} <= models  # but the Studio's search sees them


def test_the_process_group_is_not_in_the_per_part_tournament_and_says_so() -> None:
    fams, status = family_status(None)
    assert EffectorFamily.PROCESS_TOOL not in PART_FAMILIES + tuple(fams) and "process_tool" not in status
    assert set(status) == {f.value for f in PART_FAMILIES}
    block = get_part("block")
    with pytest.raises(ValueError, match="per-part tournament"):
        run_tournament(block, groups=["process"])
    with pytest.raises(ValueError, match="per-part tournament"):
        run_tournament(block, families=["process_tool"])


def test_the_vacuum_frames_stay_out_of_the_per_part_tournament_and_its_lanes() -> None:
    """Frames are area tools for large flat parts (every v1 part is at most 0.5 m): the published v1
    pages keep the three cups, and the Studio keeps the frames."""
    cups = ["cup-10mm", "cup-16mm", "cup-25mm"]
    assert [s.model for s in per_part_presets(EffectorFamily.SUCTION)] == cups
    assert [s.model for s in get_family("suction").presets()] == [*cups, FRAME_2X2, FRAME_2X3]
    for fam in available_families():  # every other per-part family offers all its presets
        if fam in PART_FAMILIES and fam is not EffectorFamily.SUCTION:
            every = [s.effector_id for s in get_family(fam).presets()]
            assert [s.effector_id for s in per_part_presets(fam)] == every, fam
    # end to end: the suction group of a per-part tournament holds exactly its three cups
    nominal = ScreenCondition("nominal", "Nominal", Placement(), (0,))
    res = run_tournament(
        get_part("block"), groups=["suction"], families=["suction"], conditions=[nominal], workers=2
    )
    group = res.groups[EntrantGroup.SUCTION]
    assert [c.spec.model for c in group.candidates] == cups
    assert not any(sc.is_frame(c.spec.params) for c in res.candidates())


def test_the_families_command_lists_what_the_per_part_tournament_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m rtt.effectors families`` lists the per-part presets: the cups, not the frames;
    the Studio-only frames and process tools are named on a closing line."""
    from rtt.effectors.__main__ import main

    assert main(["families"]) == 0
    out = capsys.readouterr().out
    listed = [line for line in out.splitlines() if line.startswith("  ")]
    assert any("Suction cup, 16 mm bellows" in line for line in listed)
    assert not any("Vacuum area frame" in line or "Nut runner" in line for line in listed)
    assert "Studio only" in out and "Vacuum area frame, 2 x 2 cups, 300 mm" in out and "Tamper pad" in out


def test_the_suction_family_keeps_its_cups_and_adds_the_frames() -> None:
    specs = get_family("suction").presets()
    assert [s.model for s in specs] == ["cup-10mm", "cup-16mm", "cup-25mm", FRAME_2X2, FRAME_2X3]
    assert get_family("suction").recipe_kinds() == ("suction_cup",)  # frames are not a designable kind
    assert all(s.group is EntrantGroup.SUCTION for s in specs)


def test_the_nut_runners_spindle_is_the_wrist_and_its_cap_is_one_the_simulator_delivers() -> None:
    assert pt.ROLL_TORQUE_NM == STUDIO_WRIST.roll_torque_nm
    assert pt.ROLL_SPEED_DPS == STUDIO_WRIST.roll_speed_max_deg_s
    spec = spec_of("process_tool", NUT_RUNNER)
    assert "spindle_speed_dps" not in spec.params  # the tool has no speed of its own
    g = pt.NutRunnerGeometry.from_params(spec.params)
    assert g.torque_cap <= pt.TORQUE_CAP_MAX_NM <= pt.ROLL_TORQUE_NM
    assert g.rated_torque == pytest.approx(g.torque_cap / pt.RATED_SAFETY_FACTOR)
    # a cap the simulator cannot deliver (the hex rides up at about 10.5 N m), a speed param and a
    # silly socket are refused
    for bad in (
        {"torque_cap_nm": 10.5},
        {"torque_cap_nm": 15.0},
        {"spindle_speed_dps": 180.0},
        {"across_flats_mm": 4.0},
    ):
        with pytest.raises(ValueError):
            pt.nut_runner_spec(**{"across_flats_mm": 13.0, **bad})
    # the card says what turns it, and which members it takes
    build = probe_scene(spec)[0].active.build
    card = " ".join(build.notes).lower()
    assert "wrist roll" in card and "wrist's speed" in card and "12 mm" in card


@pytest.mark.parametrize(("family", "model"), PRESETS)
def test_each_preset_builds_in_the_studio_scene_and_passes_the_attach_gate(family: str, model: str) -> None:
    spec = spec_of(family, model)
    info = probe_tool(spec)
    assert info.buildable, info.reason
    assert 0.0 < info.mass_kg <= ARM_PAYLOAD_KG and info.reach_mm > 0.0
    scene, reason = probe_scene(spec)
    assert scene is not None, reason
    build = scene.active.build
    assert build.notes and "approximation" in build.notes[0].lower() and "hardware" in build.notes[0].lower()
    assert build.actuation and all(isinstance(v, float | int) for v in build.actuation.values())
    n_cups = {FRAME_2X2: 4, FRAME_2X3: 6}.get(model, 1)
    assert build.n_actuators == n_cups and len(scene.ids.effector_actuators) == n_cups
    assert scene.tool_mass_kg == pytest.approx(info.mass_kg)


def test_a_process_tool_builds_on_a_bracket_and_a_changer_beside_a_gripper() -> None:
    """Its mesh walls and actuator attach under a generated bracket (the idle head's names are renamed)
    and under a tool changer's face."""
    nut = ObjectInstance.of(
        "clamp_nut", "threaded_joint", "Clamp nut", ObjectPose(0.0, 90.0, 0.0), member="nut"
    )
    box = ObjectInstance.of(
        "box",
        "part",
        "Box",
        ObjectPose(-90.0, -60.0, 0.0),
        size_x_mm=30,
        size_y_mm=30,
        size_z_mm=30,
        mass_g=50,
    )
    tighten = Step.of(
        "tighten",
        "twist",
        "clamp_nut",
        "member",
        "Tighten",
        [SuccessCriterion("angle_deg", at_least=20)],
        params={"direction": "cw", "speed_deg_s": 60},
    )
    grab = Step.of("grab", "grasp", "box", "part", "Lift", [SuccessCriterion("lift_mm", at_least=20)])
    task = TaskSpec("bracket", "", (nut, box), (tighten, grab), source="test")
    runner = spec_of("process_tool", NUT_RUNNER)
    setups = {
        "bracket": Setup.multi_head(
            ((runner, HeadMount(60.0)), (gripper(), HeadMount(-60.0))),
            {"tighten": 0, "grab": 1},
            BracketSpec(),
        ),
        "changer": Setup.tool_change({"tighten": runner, "grab": gripper()}),
    }
    for kind, setup in setups.items():
        for active in (0, 1):
            scene = build_studio_scene(setup, task, active, task=default_task_v2())
            assert scene.active.spec.effector_id == setup.tools[active].effector_id, (kind, active)
            assert len(scene.heads) == (2 if kind == "bracket" else 1)


# --------------------------------------------------------------------------- vacuum area frames


def test_frame_payloads_are_the_holding_force_over_the_safety_factor() -> None:
    assert dict(sc.SAFETY_FACTORS) == {"smooth": 1.5, "porous": 2.0, "swivelling": 2.5}
    for model in (FRAME_2X2, FRAME_2X3):
        f = sc.FrameGeometry.from_params(spec_of("suction", model).params)
        cap = f.n_cups * 60e3 * math.pi * (0.5 * f.cup.diameter) ** 2  # vacuum x lip area, every cup
        assert f.max_force == pytest.approx(cap)
        for surface, factor in sc.SAFETY_FACTORS.items():
            assert f.payload_n(surface) == pytest.approx(cap / factor)
            assert f.payload_kg(surface) == pytest.approx(cap / factor / sc.G_ACCEL)
        assert f.payload_kg("smooth") > f.payload_kg("porous") > f.payload_kg("swivelling")
        # what the card prints is what the geometry computes
        act = probe_scene(spec_of("suction", model))[0].active.build.actuation
        assert act["payload_smooth_kg"] == pytest.approx(f.payload_kg("smooth"))
        assert act["payload_porous_kg"] == pytest.approx(f.payload_kg("porous"))
        assert act["payload_swivelling_kg"] == pytest.approx(f.payload_kg("swivelling"))
        assert act["holding_force_max_n"] == pytest.approx(cap) and act["cups"] == f.n_cups


def test_the_frames_are_sized_for_their_hero_loads() -> None:
    tile, panel = 2.16, 8.0  # kg: 300 x 300 x 10 mm porcelain at 2400 kg/m^3; the 1000 x 600 x 35 mm panel
    small = sc.FrameGeometry.from_params(spec_of("suction", FRAME_2X2).params)
    large = sc.FrameGeometry.from_params(spec_of("suction", FRAME_2X3).params)
    assert (small.cups_y, small.cups_x, large.cups_y, large.cups_x) == (2, 2, 2, 3)
    assert round(1000 * small.length) == 300 and round(1000 * large.length) == 600
    assert small.payload_kg("swivelling") >= tile  # the tile frame carries the tile on every face class
    assert large.payload_kg("smooth") >= panel > large.payload_kg("swivelling")  # honest: not swivelling
    assert small.min_sealed == 3 and large.min_sealed == 3


def test_a_frame_engages_on_half_its_cups_at_least_three_never_more_than_it_has() -> None:
    def frame(x: int, y: int) -> sc.FrameGeometry:
        return sc.FrameGeometry.from_params(sc.FramePreset(x, y, 100.0, 100.0, 18.0, "test").params)

    assert {(x, y): frame(x, y).min_sealed for x, y in ((2, 1), (3, 1), (2, 2), (3, 2), (4, 2), (4, 4))} == {
        (2, 1): 2,  # a frame of two needs both
        (3, 1): 3,
        (2, 2): 3,
        (3, 2): 3,
        (4, 2): 4,
        (4, 4): 8,
    }


def test_the_frame_card_says_the_porous_and_swivelling_ratings_are_arithmetic() -> None:
    caveat = sc.FRAME_CAVEAT.lower()
    assert "arithmetic" in caveat and "porous" in caveat and "swivel" in caveat and "[assumed]" in caveat
    build = probe_scene(spec_of("suction", FRAME_2X3))[0].active.build
    assert build.notes[0] == sc.FRAME_CAVEAT


def test_every_tool_plans_on_a_part_wider_than_a_v1_mesh_object(stand_ins: dict[str, Any]) -> None:
    """A 1000 x 600 mm panel: the frame, the single cup, the nut runner and the tamper all plan their tip
    height from its bounding box (``part_height`` would raise: a v1 mesh object is at most 0.5 m)."""
    panel = build_part(
        "panel", "block", {"x": 1000.0, "y": 600.0, "z": 35.0}, Material("p", 900.0, (0.2, 0.5, 0.9))
    )
    with pytest.raises(ValueError, match="across"):
        part_height(panel)
    task = default_task_v2()
    floor = task.controller.min_tip_clearance
    for family, model in (
        ("suction", "cup-16mm"),
        ("suction", FRAME_2X3),
        ("process_tool", NUT_RUNNER),
        ("process_tool", TAMPER),
    ):
        spec = spec_of(family, model)
        build = probe_scene(spec)[0].active.build
        plan = get_family(family).primitive(spec, build).plan(panel, None, task, build)  # type: ignore[arg-type]
        if family == "suction":
            cup = (
                sc.FrameGeometry.from_params(spec.params).cup
                if sc.is_frame(spec.params)
                else sc.CupGeometry.from_params(spec.params)
            )
            want = 0.035 - cup.press_depth
        elif model == NUT_RUNNER:
            want = 0.035 - pt.NutRunnerGeometry.from_params(spec.params).socket_depth
        else:
            want = 0.035
        assert plan.tip_height == pytest.approx(max(floor, want)), model


def test_top_height_only_widens_the_size_check_a_degenerate_mesh_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = get_part("block")
    assert top_height(block) == pytest.approx(part_height(block))  # a v1 part: the very same number

    def degenerate(part: Any) -> float:
        raise ValueError("mesh object 'block' is degenerate")

    monkeypatch.setattr(heights, "part_height", degenerate)
    with pytest.raises(ValueError, match="degenerate"):
        top_height(block)


def test_a_frame_plans_its_press_from_the_same_height_as_a_cup() -> None:
    block = get_part("block")
    task = default_task_v2()
    for model in ("cup-16mm", FRAME_2X2):
        spec = spec_of("suction", model)
        impl = get_family("suction")
        build = probe_scene(spec)[0].active.build
        plan = impl.primitive(spec, build).plan(block, None, task, build)  # type: ignore[arg-type]
        cup = (
            sc.FrameGeometry.from_params(spec.params).cup
            if sc.is_frame(spec.params)
            else sc.CupGeometry.from_params(spec.params)
        )
        press = cup.press_depth
        assert plan.tip_height == pytest.approx(
            max(task.controller.min_tip_clearance, part_height(block) - press)
        )


@pytest.mark.parametrize(
    "params",
    [
        {"cups_x": 1, "cups_y": 1},  # one cup is the cup model itself
        {"cups_x": 5},
        {"cups_x": 2.5},
        {"spacing_x_mm": 10.0},  # the cups would touch
        {"extra": 1},
    ],
)
def test_frame_params_are_validated(params: dict[str, Any]) -> None:
    base = dict(spec_of("suction", FRAME_2X2).params)
    with pytest.raises(ValueError):
        sc.FrameGeometry.from_params({**base, **params})


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Heavy or wide parts for the Studio's part template through its intake resolver: the frozen
    parts library caps a part at 0.5 m across, so a 1 m panel is built with the cap lifted."""
    parts: dict[str, Any] = {}
    monkeypatch.setitem(part_template._RESOLVERS, "upload", lambda key: parts[key])
    monkeypatch.setattr(parts_types, "MAX_PART_SIZE", 2.0)
    yield parts


def _slab(parts: dict[str, Any], key: str, x: float, y: float, z: float, kg: float) -> ObjectInstance:
    density = kg / (x * y * z / 1e9)
    parts[key] = build_part(
        key, "block", {"x": x, "y": y, "z": z}, Material("slab", density, (0.2, 0.5, 0.9))
    )
    return ObjectInstance.of(
        "slab", "part", "Slab", ObjectPose(0.0, 0.0, 0.0), ref=f"upload:{key}", shape="mesh"
    )


def _lift(obj: ObjectInstance) -> TaskSpec:
    step = Step.of(
        "lift",
        "suction",
        obj.name,
        "part",
        "Lift it",
        [SuccessCriterion("lift_mm", at_least=25), SuccessCriterion("hold_s", at_least=0.8)],
        params={"lift_mm": 60, "hold_s": 1.0},
    )
    return _task_of(obj, step)


@pytest.mark.parametrize(
    ("model", "x", "y", "z", "control_cup_mm"),
    [(FRAME_2X2, 300.0, 300.0, 10.0, 22.0), (FRAME_2X3, 1000.0, 600.0, 35.0, 25.0)],
)
def test_a_frame_lifts_its_rated_payload_and_fails_1_6_times_it(
    model: str, x: float, y: float, z: float, control_cup_mm: float, stand_ins: dict[str, Any]
) -> None:
    spec = spec_of("suction", model)
    rated = sc.FrameGeometry.from_params(spec.params).payload_kg("smooth")
    ok = _run(spec, _lift(_slab(stand_ins, "rated", x, y, z, rated)))
    assert ok.steps[0].success, (ok.steps[0].failure_mode, dict(ok.steps[0].quantities))
    assert ok.steps[0].quantities["lift_mm"] >= 25.0 and ok.steps[0].quantities["penetration_mm"] <= 1.0
    over = _run(spec, _lift(_slab(stand_ins, "over", x, y, z, 1.6 * rated))).steps[0]
    assert not over.success and over.failure_mode == "criterion_not_met"
    assert (
        over.quantities["lift_mm"] < 1.0
    )  # the cups' cap (vacuum x area) is below its weight: it never leaves the table
    # control: the same layout on bigger cups lifts the same 1.6 x slab by 10 mm or more, so the
    # failure above is the cups' cap and not the arm. (The arm's 150 N z servo lifts and holds only
    # about 11 kg on these criteria, so the control is read from the lift, not from the success.)
    layout = sc.FrameGeometry.from_params(spec.params)
    bigger = sc.FramePreset(
        layout.cups_x,
        layout.cups_y,
        1000.0 * layout.spacing_x,
        1000.0 * layout.spacing_y,
        control_cup_mm,
        "control",
    )
    control = _run(sc.frame_spec(bigger), _lift(_slab(stand_ins, "control", x, y, z, 1.6 * rated))).steps[0]
    assert control.quantities["lift_mm"] >= 10.0, dict(control.quantities)
    assert sc.FrameGeometry.from_params(sc.frame_spec(bigger).params).max_force > 1.6 * rated * G


def test_a_frame_lifts_the_real_tile_under_the_placement_grid() -> None:
    tile = ObjectInstance.of(
        "tile",
        "part",
        "Floor tile",
        ObjectPose(0.0, 0.0, 0.0),
        shape="box",
        size_x_mm=300,
        size_y_mm=300,
        size_z_mm=10,
        mass_g=2160,
    )
    task = _lift(tile)
    spec = spec_of("suction", FRAME_2X2)
    conds = condition_grid(task, 4)
    results = run_task_jobs([StudioJob(Setup.single(spec, task.step_ids), task, c) for c in conds], workers=2)
    assert all(r.steps[0].success for r in results), [dict(r.steps[0].quantities) for r in results]


def test_an_unsealed_cup_stays_off_so_the_frame_holds_only_what_the_covered_cups_can() -> None:
    """A 400 mm part under the 2 x 3 frame covers only its two middle cups (x = 0): 2 x 21 mm cups hold
    41.6 N, a 3 kg part (29 N) lifts and a 5 kg one (49 N) does not; the four outer cups hang in the
    air and their valves stay shut."""
    spec = spec_of("suction", FRAME_2X3)
    cap = 2 * sc.FrameGeometry.from_params(spec.params).cup.max_force
    assert 3.0 * G < cap < 5.0 * G

    def box(mass_g: float) -> TaskSpec:
        return _lift(
            ObjectInstance.of(
                "slab",
                "part",
                "Slab",
                ObjectPose(0.0, 0.0, 0.0),
                shape="box",
                size_x_mm=400,
                size_y_mm=400,
                size_z_mm=30,
                mass_g=mass_g,
            )
        )

    assert _run(spec, box(3000)).steps[0].success
    heavy = _run(spec, box(5000)).steps[0]
    assert not heavy.success and heavy.quantities["lift_mm"] < 1.0


# --------------------------------------------------------------------------- the nut runner


def _clamp_nut(final_nm: float, *, length_mm: float = 16.0) -> ObjectInstance:
    """A hex clamp nut on a threaded stub: 13 mm across flats (15.011 mm across corners), M8-ish pitch, three
    turns to seat, tightening to ``final_nm``."""
    return ObjectInstance.of(
        "clamp_nut",
        "threaded_joint",
        "Clamp nut",
        ObjectPose(0.0, 0.0, 0.0),
        member="nut",
        diameter_mm=15.011,
        length_mm=length_mm,
        pitch_mm=1.25,
        engage_turns=3.0,
        start_torque_nm=0.3,
        final_torque_nm=final_nm,
        chamfer_mm=1.0,
    )


def _tighten(final_nm: float, goal_nm: float, length_mm: float = 16.0) -> TaskSpec:
    """Start the nut on its thread, then tighten it to at least ``goal_nm`` within a 15 N m wrist limit."""
    nut = _clamp_nut(final_nm, length_mm=length_mm)
    start = Step.of(
        "start",
        "insert",
        "clamp_nut",
        "member",
        "Start the nut on the thread",
        [SuccessCriterion("depth_mm", at_least=3.5)],
    )
    tighten = Step.of(
        "tighten",
        "twist",
        "clamp_nut",
        "member",
        "Tighten the nut",
        [SuccessCriterion("torque_nm", at_least=goal_nm)],
        params={"direction": "cw", "speed_deg_s": 180},
        constraints=StepConstraints(max_torque_nm=15.0),
    )
    return TaskSpec("clamp nut", "", (nut,), (start, tighten), source="test")


@pytest.fixture(scope="module")
def nut_runs() -> dict[str, Any]:
    """Every nut-runner run of this file, in spawned workers (each is a 20 s simulation)."""
    spec = spec_of("process_tool", NUT_RUNNER)
    g = pt.NutRunnerGeometry.from_params(spec.params)
    rated = g.rated_torque
    cases = {
        "rated": (spec, _tighten(rated, 0.95 * rated), StudioCondition("nominal", 0)),
        # 1.6x rated is beyond the clutch cap plus the torque tolerance: what fails is the clutch
        "over": (spec, _tighten(1.6 * rated, 1.6 * rated), StudioCondition("nominal", 0)),
        "cup": (spec_of("suction", "cup-25mm"), _tighten(rated, 0.95 * rated), StudioCondition("nominal", 0)),
    }
    # the supported range of members (the module docstring): the shortest and one too short
    cases["shortest"] = (spec, _tighten(rated, 0.95 * rated, pt.MIN_MEMBER_MM), StudioCondition("nominal", 0))
    cases["too_short"] = (spec, _tighten(rated, 0.95 * rated, 8.0), StudioCondition("nominal", 0))
    grid = condition_grid(cases["rated"][1], 4)
    for c in grid[1:]:
        cases[f"rated_{c.condition_id}"] = (spec, cases["rated"][1], c)
    jobs = [StudioJob(Setup.single(s, t.step_ids), t, c) for s, t, c in cases.values()]
    return dict(zip(cases, run_task_jobs(jobs, workers=4), strict=True))


def test_the_nut_runner_tightens_a_clamp_nut_to_its_rated_torque(nut_runs: dict[str, Any]) -> None:
    """The nominal nut, three placement conditions of the grid, and the shortest supported member."""
    rated = pt.NutRunnerGeometry.from_params(spec_of("process_tool", NUT_RUNNER).params).rated_torque
    keys = [k for k in nut_runs if k.startswith("rated")] + ["shortest"]
    assert len(keys) == 5
    for key in keys:
        start, tighten = nut_runs[key].steps
        assert start.success and tighten.success, (
            key,
            [(s.failure_mode, dict(s.quantities)) for s in (start, tighten)],
        )
        q = tighten.quantities
        assert q["torque_nm"] >= 0.95 * rated and q["torque_nm"] <= pt.ROLL_TORQUE_NM
        assert q["penetration_mm"] <= 1.0 and q["turns"] > 2.5
        assert q["depth_mm"] > 7.0  # driven home and left seated: the release did not pull it up its thread


def test_a_member_too_short_for_the_socket_is_unsupported_and_fails_as_documented(
    nut_runs: dict[str, Any],
) -> None:
    """With the 12 mm socket the Studio's insert controller aims the tip 3 mm above the member's bottom,
    so a member under :data:`MIN_MEMBER_MM` (10 mm and below, measured) jams on insert and is never
    started; the documented remedy of a shallower socket does not exist."""
    start, tighten = nut_runs["too_short"].steps
    assert not start.success and start.failure_mode == "jam" and start.quantities["depth_mm"] < 1.0
    assert not tighten.success
    assert pt.MIN_MEMBER_MM == 12.0


def test_the_nut_runner_cannot_pass_1_6_times_its_rated_torque(nut_runs: dict[str, Any]) -> None:
    """The clutch is what fails: a goal of 1.6x the rated torque lies beyond the clutch cap by more
    than 5 % (and far beyond the criterion's 0.02 N m tolerance), and the torque the tool reaches
    stays under the goal by a clear margin. (A goal of 0.95 x 1.6 x rated, 10.13 N m, sat on the
    10 N m cap, where the dynamic peak of 10.127 N m met it within the tolerance.)"""
    from rtt.studio.types import CRITERION_TOLERANCE

    g = pt.NutRunnerGeometry.from_params(spec_of("process_tool", NUT_RUNNER).params)
    goal = 1.6 * g.rated_torque
    start, tighten = nut_runs["over"].steps
    assert start.success and not tighten.success and tighten.failure_mode in ("criterion_not_met", "slip")
    assert goal >= 1.05 * g.torque_cap  # beyond what the clutch passes
    assert tighten.quantities["torque_nm"] <= g.torque_cap + 0.6
    assert tighten.quantities["torque_nm"] < goal - CRITERION_TOLERANCE["torque_nm"] - 0.3


def test_a_suction_cup_cannot_tighten_a_hex_nut(nut_runs: dict[str, Any]) -> None:
    steps = nut_runs["cup"].steps
    assert not all(s.success for s in steps)  # the S2 rule: a wrong tool fails the same job


# --------------------------------------------------------------------------- the tamper pad


def _press(force_n: float, *, limit_n: float = 100.0, mass_g: float = 2160.0) -> TaskSpec:
    tile = ObjectInstance.of(
        "tile",
        "part",
        "Floor tile",
        ObjectPose(0.0, 0.0, 0.0),
        shape="box",
        size_x_mm=300,
        size_y_mm=300,
        size_z_mm=10,
        mass_g=mass_g,
    )
    step = Step.of(
        "tamp",
        "push",
        "tile",
        "part",
        "Press the tile level",
        [SuccessCriterion("force_n", at_least=force_n), SuccessCriterion("tilt_deg", at_most=3.0)],
        constraints=StepConstraints(max_force_n=limit_n),
    )
    return _task_of(tile, step)


def test_the_tamper_presses_its_rated_force_and_never_more_than_its_cap() -> None:
    spec = spec_of("process_tool", TAMPER)
    g = pt.TamperGeometry.from_params(spec.params)
    assert g.rated_force == pytest.approx(g.max_force / pt.RATED_SAFETY_FACTOR) and g.max_force == 60.0
    ok = _run(spec, _press(g.rated_force)).steps[0]
    assert ok.success, (ok.failure_mode, dict(ok.quantities))
    assert g.rated_force <= ok.quantities["force_n"] <= g.max_force
    over = _run(spec, _press(1.6 * g.rated_force)).steps[0]
    assert not over.success and over.failure_mode == "criterion_not_met"
    assert (
        over.quantities["force_n"] <= g.max_force + 1.0
    )  # the stage slips at its cap (the wrist can push on)


def test_the_tamper_stage_is_a_spring_below_its_cap() -> None:
    """Pad force = stiffness x compression up to the cap: the actuator law, read from the scene."""
    spec = spec_of("process_tool", TAMPER)
    g = pt.TamperGeometry.from_params(spec.params)
    scene = probe_scene(spec)[0]
    prim = get_family("process_tool").primitive(spec, scene.active.build)
    data = mujoco.MjData(scene.model)
    j = scene.model.joint("eff_tamper_stage")
    for compression, want in ((0.0, 0.0), (0.010, 40.0), (0.030, g.force_cap), (0.050, g.force_cap)):
        data.qpos[j.qposadr[0]], data.qvel[j.dofadr[0]] = compression, 0.0
        assert prim.command(0.0, 0, 1.0, scene.model, data, scene)[0] == pytest.approx(-want)


def test_a_process_run_is_deterministic_in_process_and_in_a_spawned_worker() -> None:
    spec = spec_of("process_tool", TAMPER)
    task = _press(30.0)
    job = StudioJob(Setup.single(spec, task.step_ids), task, StudioCondition("c", 0))
    a, b = run_task(job), run_task(job)
    w = run_task_jobs([job, job], workers=2)
    assert a.traj_sha == b.traj_sha == w[0].traj_sha == w[1].traj_sha and a.steps[0].success
