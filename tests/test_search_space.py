"""DesignSpace layer: the v1 wrapper delegates exactly, the v2 space follows spec section 4."""

from __future__ import annotations

import json
import math

import pytest

from rtt.design.distance import design_distance
from rtt.design.schema import FingerDesign, load_entrants
from rtt.design.validate import validate
from rtt.search.base import DesignScore, EpisodeOutcome, Proposal
from rtt.search.diversity import all_cells, points_of
from rtt.search.fakes import FakeEvaluator
from rtt.search.fakes_v2 import FakeEvaluatorV2, stub_entrants
from rtt.search.nemotron import PROPOSER_SLOTS, _slot_schema, render_parameters
from rtt.search.random_search import DesignSampler
from rtt.search.runner import DEFAULT_ENTRANTS
from rtt.search.space import SPACE_NAMES, V1, V2, DesignSpace, get_space, space_of
from rtt.search.space_v2 import (
    INACTIVE,
    NUMERIC_BOUNDS,
    NUMERIC_PARAMS,
    QUANT_STEPS,
    SCORED_PARAMS,
    BaselineEntrant,
    FingerRecipeStub,
    quantise,
    track_g,
)

V1S, V2S = get_space(V1), get_space(V2)


def test_registry_and_dispatch() -> None:
    assert SPACE_NAMES == ("box-v1", "freeform-v2")
    assert get_space() is V1S and get_space(V2) is V2S
    for space in (V1S, V2S):
        assert isinstance(space, DesignSpace)
    entrant = load_entrants(DEFAULT_ENTRANTS)[0]
    assert space_of(entrant) is V1S
    recipe = V2S.recipe(pad="flat", tip="blunt", grip_height_frac=0.5, pad_height_mm=20, finger_width_mm=20)
    assert space_of(recipe) is V2S and space_of(stub_entrants()[0]) is V2S
    with pytest.raises(KeyError):
        get_space("box-v3")
    with pytest.raises(TypeError):
        space_of({"pad": "flat"})


# --------------------------------------------------------------------------- v1 wrapper


def test_v1_wrapper_delegates_to_the_v1_code() -> None:
    task = FakeEvaluator(0).describe().gripper_task
    designs = load_entrants(DEFAULT_ENTRANTS) + DesignSampler(3, task).sample(6)
    for a in designs:
        assert V1S.design_id(a) == a.design_id()
        assert V1S.load(V1S.dump(a)) == a
        assert V1S.points(a) == points_of(a)
        assert V1S.cell(a) == (a.inner_face, a.tip_profile)
        for b in designs:
            assert V1S.distance(a, b) == design_distance(a, b)
        check = V1S.validate(a.geometry_params(), task)
        assert check.ok == validate(a.geometry_params(), task).ok
    bad = {**designs[0].geometry_params(), "inner_face": "flat", "inner_face_depth_mm": 3.0}
    assert V1S.validate(bad, task).errors == tuple(validate(bad, task).errors)
    assert V1S.cells() == all_cells() and V1S.describe() == render_parameters()
    a, b = V1S.sample_sobol(9, task), DesignSampler(9, task)
    assert a.sample(5) == b.sample(5)
    v1_slot = _slot_schema(PROPOSER_SLOTS[3], False)["properties"]
    assert {k: v1_slot[k] for k in V1S.tool_schema()} == V1S.tool_schema()


def test_v1_proposal_json_is_unchanged() -> None:
    design = load_entrants(DEFAULT_ENTRANTS)[1]
    data = Proposal(design=design, origin="entrant").to_json()
    assert "design_space" not in data and "canon_notes" not in data
    assert list(data) == [
        "design",
        "design_id",
        "origin",
        "operator",
        "island",
        "slot",
        "parents",
        "hypothesis",
        "via",
        "request_hash",
        "fallback_reason",
    ]
    assert Proposal.from_json(data).design == design


# --------------------------------------------------------------------------- v2 space


def test_v2_scored_parameters_follow_the_spec_table() -> None:
    assert SCORED_PARAMS == (
        "pad",
        "tip",
        "grip_height_frac",
        "pad_height_mm",
        "finger_width_mm",
        "pad_depth_mm",
        "wall_draft_deg",
        "clearance_mm",
        "tip_size_mm",
    )
    assert NUMERIC_BOUNDS == {
        "grip_height_frac": (0.15, 0.85),
        "pad_height_mm": (8.0, 30.0),
        "finger_width_mm": (14.0, 40.0),
        "pad_depth_mm": (0.0, 15.0),
        "wall_draft_deg": (0.0, 70.0),
        "clearance_mm": (0.2, 1.5),
        "tip_size_mm": (0.0, 8.0),
    }
    assert INACTIVE == {
        "pad_depth_mm": ("pad", ("flat",), 0.0),
        "wall_draft_deg": ("pad", ("flat",), 0.0),
        "clearance_mm": ("pad", ("flat", "v_funnel"), 0.2),
        "tip_size_mm": ("tip", ("blunt",), 0.0),
    }
    assert set(QUANT_STEPS.values()) == {0.01, 0.1, 0.5}
    assert V2S.n_terms == 9 and len(V2S.cells()) == 9
    assert V2S.cells()[0] == ("flat", "blunt") and V2S.cells()[-1] == ("v_funnel", "hook")
    table = V2S.describe()
    for name in SCORED_PARAMS:
        assert f"| {name} |" in table
    assert "flat, v_funnel (0.2)" in table


def test_v2_tool_schema_requires_only_the_always_active_numbers() -> None:
    props = V2S.tool_schema()
    assert list(props) == ["pad", "tip", "params"]
    assert props["pad"]["enum"] == ["flat", "conform", "v_funnel"]
    assert props["tip"]["enum"] == ["blunt", "chamfer", "hook"]
    params = props["params"]
    assert params["additionalProperties"] is False
    assert params["required"] == ["grip_height_frac", "pad_height_mm", "finger_width_mm"]
    assert params["properties"]["wall_draft_deg"] == {"type": "number", "minimum": 0.0, "maximum": 70.0}


def test_v2_metric_points_and_cells() -> None:
    a = V2S.recipe(
        pad="conform",
        tip="chamfer",
        grip_height_frac=0.5,
        pad_height_mm=19,
        finger_width_mm=27,
        pad_depth_mm=7.5,
        wall_draft_deg=35,
        clearance_mm=0.85,
        tip_size_mm=4,
    )
    b = V2S.recipe(
        pad="conform",
        tip="hook",
        grip_height_frac=0.5,
        pad_height_mm=19,
        finger_width_mm=27,
        pad_depth_mm=7.5,
        wall_draft_deg=35,
        clearance_mm=0.85,
        tip_size_mm=4,
    )
    assert V2S.distance(a, a) == 0.0
    assert V2S.distance(a, b) == pytest.approx(1 / 9) == V2S.distance(b, a)
    assert V2S.params(a)["clearance_mm"] == 0.9  # 0.85 is not on the 0.1 mm grid: half-up, as in G
    assert V2S.points(a) == {
        "grip_height_frac": 50,
        "pad_height_mm": 50,
        "finger_width_mm": 50,
        "pad_depth_mm": 50,
        "wall_draft_deg": 50,
        "clearance_mm": 54,
        "tip_size_mm": 50,
    }
    assert V2S.cell(a) == ("conform", "chamfer")
    assert V2S.point_differences(a, b) == "tip 100"
    assert V2S.points_row(a).startswith("grip 50 pad_h 50 width 50")
    stock = stub_entrants()[0]
    assert V2S.cell(stock) is None and V2S.points(stock) is None and math.isinf(V2S.distance(stock, a))
    flat = stub_entrants()[1]
    assert V2S.design_id(flat) == V2S.design_id(flat.point)  # a named recipe entrant is that recipe
    assert V2S.params_json(flat) == V2S.params_json(flat.point)


def test_v2_dump_load_and_proposal_json_round_trip() -> None:
    for design in stub_entrants() + V2S.sample_sobol(4, None).sample(5):
        data = Proposal(design=design, origin="sobol", notes=("tip_size_mm ignored",)).to_json()
        assert data["design_space"] == V2 and data["canon_notes"] == ["tip_size_mm ignored"]
        assert json.loads(json.dumps(data)) == data
        back = Proposal.from_json(data)
        assert V2S.design_id(back.design) == data["design_id"] and back.notes == ("tip_size_mm ignored",)


def test_v2_score_ranks_on_registered_then_lift_then_mass() -> None:
    def ep(success: bool, lift: bool, seed: int) -> EpisodeOutcome:
        return EpisodeOutcome(
            "tee_c0",
            "tee",
            0,
            seed,
            success,
            None if success else "not_registered",
            0.1,
            1.0,
            lift_success=lift,
            jam_on_close=not success,
            placement="c0",
        )

    a = DesignScore.from_episodes_v2("a", 30.0, [ep(True, True, 0), ep(False, True, 1)])
    b = DesignScore.from_episodes_v2("b", 20.0, [ep(True, True, 0), ep(False, False, 1)])
    c = DesignScore.from_episodes_v2("c", 25.0, [ep(True, True, 0), ep(False, True, 1)])
    assert a.success == b.success == 0.5 and a.lift_success == 1.0 and b.lift_success == 0.5
    assert sorted([a, b, c], key=lambda s: s.rank_key(), reverse=True) == [c, a, b]
    assert a.jam_on_close == 1 and a.per_placement == {"c0": 0.5} and a.best_k == 0
    again = DesignScore.from_json(a.to_json())
    assert again == a


def test_fake_v2_evaluator_is_batch_invariant_and_has_headroom() -> None:
    ev = FakeEvaluatorV2(0)
    info = ev.describe()
    assert info.design_space == V2 and info.episodes_per_design == 48 and info.controller_settings == ()
    designs = stub_entrants() + V2S.sample_sobol(1, info.gripper_task).sample(12)
    together = ev.evaluate(designs)
    alone = [ev.evaluate([d])[0] for d in designs]
    assert [s.to_json() for s in together] == [s.to_json() for s in alone]
    entrants = [s.success for s in together[:4]]
    assert max(entrants) < 0.5 and max(s.success for s in together) > max(entrants)


@pytest.mark.skipif(track_g().recipe_cls is None, reason="track G's FingerRecipe has not landed")
def test_stub_matches_track_g_schema() -> None:
    """The stub and track G agree on bounds, steps, inactive rules and quantisation; G's model wins."""
    import importlib

    schema = importlib.import_module("rtt.design.freeform.schema")
    g = track_g().recipe_cls
    assert g is schema.FingerRecipe and set(SCORED_PARAMS) <= set(g.model_fields)
    assert dict(schema.PARAM_BOUNDS) == NUMERIC_BOUNDS
    stub = FingerRecipeStub.model_json_schema()["properties"]
    assert {n: (stub[n]["minimum"], stub[n]["maximum"]) for n in NUMERIC_BOUNDS} == NUMERIC_BOUNDS
    assert dict(schema.QUANTUM) == QUANT_STEPS
    assert tuple(schema.SCORED_PARAMS) == SCORED_PARAMS
    for pad, tip in V2S.cells():
        inactive = {
            n for n, (owner, where, _) in INACTIVE.items() if (pad if owner == "pad" else tip) in where
        }
        assert set(schema.inactive_params(pad, tip)) == inactive
    for name, (_, _, fixed) in INACTIVE.items():
        assert schema.FIXED_VALUES[name] == fixed
    for name in NUMERIC_PARAMS:
        lo, hi = NUMERIC_BOUNDS[name]
        for k in range(81):  # grid points, half steps and points in between
            value = lo + (hi - lo) * k / 80
            for v in (value, value + QUANT_STEPS[name] / 2, value + QUANT_STEPS[name] / 3):
                assert quantise(name, v) == schema.quantize(name, v), (name, v)
    for recipe in V2S.sample_sobol(7, None).sample(200):
        assert isinstance(recipe, g)
        canon, _ = importlib.import_module("rtt.design.freeform.canon").canonicalize(recipe)
        assert V2S.params(canon) == V2S.params(recipe)  # the space's canonical form is G's
        assert V2S.design_id(recipe) == track_g().design_id(recipe)


def test_baseline_entrants_validate_and_load() -> None:
    for entrant in stub_entrants():
        assert V2S.validate(entrant, None).ok
        back = V2S.load(V2S.dump(entrant))
        assert isinstance(back, BaselineEntrant) and V2S.design_id(back) == V2S.design_id(entrant)
    assert len({V2S.design_id(e) for e in stub_entrants()}) == 4
    assert not isinstance(stub_entrants()[0], FingerDesign)


# --------------------------------------------------------------------------- with track G (skipped until it lands)

needs_g = pytest.mark.skipif(
    track_g().baseline_cls is None or track_g().load_entrants is None,
    reason="track G's entrants have not landed",
)


@needs_g
def test_track_g_entrants_drive_round0_ids_dumps_and_metric() -> None:
    from rtt.search.runner import DEFAULT_ENTRANTS_V2, _load_entrants_v2, reference_entrants_v2

    if not DEFAULT_ENTRANTS_V2.exists():
        pytest.skip("configs/entrants_v2.yaml has not landed")
    round0 = _load_entrants_v2(DEFAULT_ENTRANTS_V2)
    refs = reference_entrants_v2(DEFAULT_ENTRANTS_V2)
    assert [e.name for e in round0] == ["stock", "flat", "imprint_configurator", "imprint_swept"]
    assert {e.name for e in refs} == {"ref_wrap", "ref_hook"}
    g_id = track_g().design_id
    for entrant in round0 + refs:
        assert V2S.is_design(entrant) and V2S.design_id(entrant) == g_id(entrant)
        assert V2S.validate(entrant, None).ok
        back = V2S.load(V2S.dump(entrant))
        assert type(back) is type(entrant) and V2S.design_id(back) == V2S.design_id(entrant)
    stock, flat, configurator, swept = round0
    assert V2S.buildable(configurator) is configurator and V2S.cell(configurator) == ("conform", "blunt")
    assert V2S.cell(stock) is None and V2S.distance(stock, flat) == float("inf")
    assert V2S.buildable(flat) is flat.recipe and V2S.design_id(flat) == V2S.design_id(flat.recipe)
    assert V2S.buildable(swept) is swept  # the sweep is outside the search space: G builds the baseline
    assert len({V2S.design_id(e) for e in round0}) == 4
    copy = V2S.recipe(**V2S.params(flat.recipe))
    assert V2S.design_id(copy) == V2S.design_id(flat)  # a proposal that repeats flat is a duplicate


@needs_g
def test_track_g_validators_map_to_stages_and_cache_geometry() -> None:
    """With the training parts and the v2 task, every design goes through G's section-13 battery."""
    library = pytest.importorskip("rtt.parts.library")
    task_mod = pytest.importorskip("rtt.sim.task")
    if track_g().validate is None or track_g().build_pair is None or not hasattr(task_mod, "default_task_v2"):
        pytest.skip("track G's validators or track S's task_v2 have not landed")
    from rtt.search.space_v2 import G_CODE_STAGES, V2Context

    ctx = V2Context(parts=tuple(library.training_parts()), task=task_mod.default_task_v2())
    designs = V2S.sample_sobol(3, None).sample(12)  # schema-valid: geometry may still reject some
    stages = []
    for design in designs:
        check = V2S.validate(design, ctx)
        stages.append(check.stage)
        assert check.ok == (check.stage is None)
        assert check.stage is None or check.stage in set(G_CODE_STAGES.values())
        if check.ok:
            gid = V2S.geometry_id(design, ctx)
            assert gid is not None and ctx.geometry[V2S.design_id(design)] == gid
            assert len([k for k in ctx.pairs if k[0] == V2S.design_id(design)]) == len(ctx.parts)
    assert None in stages  # at least one random recipe builds on all six parts


def test_a_generator_crash_rejects_the_design_instead_of_ending_the_run(monkeypatch) -> None:
    """A validator that raises something other than a rejection is a build failure, logged as such."""
    from rtt.search import space_v2
    from rtt.search.space_v2 import V2Context, _TrackG

    def crash(design, parts, task):  # type: ignore[no-untyped-def]
        raise RuntimeError("pad simplification did not converge")

    fake_g = _TrackG(recipe_cls=FingerRecipeStub, validate=crash)
    monkeypatch.setattr(space_v2, "track_g", lambda: fake_g)
    ctx = V2Context(parts=("part",), task="task", part_ids=("p",))
    recipe = FingerRecipeStub(**V2S.params(V2S.sample_sobol(1, None).sample(1)[0]))
    check = V2S.validate(recipe, ctx)
    assert not check.ok and check.stage == "build" and check.metrics == {"generator_error": 1.0}
    assert check.errors[0].startswith("generator_error:") and "did not converge" in check.errors[0]
