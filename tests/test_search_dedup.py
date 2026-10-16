"""v2 canonicalisation and de-duplication (GV5): inactive parameters, quantisation, ids and the shared screen."""

from __future__ import annotations

import re

import numpy as np
import pytest

from rtt.search.diversity import (
    EXPLORE_MIN_DISTANCE_V2,
    MIN_DISTANCE_V2,
    SCREEN_REASONS,
    screen,
    threshold_rule,
)
from rtt.search.fakes_v2 import FakeEvaluatorV2
from rtt.search.nemotron_v2 import NemotronProposerV2
from rtt.search.space import V2, get_space
from rtt.search.space_v2 import GENERATOR_VERSION_STUB, canonical_params, track_g
from search_v2_support import (
    ReplyTransport,
    client_for,
    diverse_recipes,
    recipe_slot,
    submission_v2,
    v2_round0,
)

SPACE = get_space(V2)
BASE = {"grip_height_frac": 0.5, "pad_height_mm": 20.0, "finger_width_mm": 24.0}


def ident(**params: object) -> str:
    return SPACE.design_id(SPACE.recipe(**params))


def test_inactive_only_differences_give_identical_ids_and_are_logged() -> None:
    flat = ident(pad="flat", tip="blunt", **BASE)
    assert flat == ident(
        pad="flat", tip="blunt", pad_depth_mm=9.0, wall_draft_deg=40, clearance_mm=1.3, **BASE
    )
    assert flat == ident(pad="flat", tip="blunt", tip_size_mm=5.0, **BASE)
    funnel = dict(pad="v_funnel", tip="hook", pad_depth_mm=6.0, wall_draft_deg=30.0, tip_size_mm=3.0, **BASE)
    assert ident(**funnel) == ident(**funnel, clearance_mm=1.4)  # clearance is inactive for a V
    assert ident(**funnel) != ident(**{**funnel, "tip": "chamfer"})
    canon, notes, errors = canonical_params({"pad": "flat", "tip": "blunt", "pad_depth_mm": 9.0, **BASE})
    assert not errors and canon is not None and canon["pad_depth_mm"] == 0.0
    assert notes == ("pad_depth_mm is inactive for a flat pad: 9.0 ignored, set to 0",)
    # A given value equal to the fixed one is not worth a note.
    assert canonical_params({"pad": "flat", "tip": "blunt", "pad_depth_mm": 0.0, **BASE})[1] == ()


def test_quantisation_to_the_grid() -> None:
    conform = {
        "pad": "conform",
        "tip": "chamfer",
        "pad_depth_mm": 6.0,
        "wall_draft_deg": 30.0,
        "clearance_mm": 0.5,
        "tip_size_mm": 2.0,
        "finger_width_mm": 24.0,
        "pad_height_mm": 20.0,
    }
    assert ident(grip_height_frac=0.501, **conform) == ident(grip_height_frac=0.5, **conform)
    assert ident(grip_height_frac=0.506, **conform) != ident(grip_height_frac=0.5, **conform)
    assert ident(**{**conform, "wall_draft_deg": 30.2}, grip_height_frac=0.5) == ident(
        grip_height_frac=0.5, **conform
    )
    assert ident(**{**conform, "wall_draft_deg": 30.3}, grip_height_frac=0.5) != ident(
        grip_height_frac=0.5, **conform
    )
    assert ident(**{**conform, "pad_depth_mm": 6.04}, grip_height_frac=0.5) == ident(
        grip_height_frac=0.5, **conform
    )
    canon, notes, _ = canonical_params({**conform, "grip_height_frac": 0.5, "pad_depth_mm": 6.04})
    assert (
        canon is not None and canon["pad_depth_mm"] == 6.0 and notes == ("quantised: pad_depth_mm 6.04 -> 6",)
    )
    # Quantisation never rescues a value outside the bounds, and bounds use the quantised value.
    assert canonical_params({**conform, "grip_height_frac": 0.5, "pad_depth_mm": 15.04})[0] is not None
    assert "outside" in canonical_params({**conform, "grip_height_frac": 0.5, "pad_depth_mm": 15.06})[2][0]


def test_design_id_is_a_pinned_sha256_of_the_generator_and_the_canonical_recipe() -> None:
    value = ident(pad="flat", tip="blunt", **BASE)
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    if track_g().generator_version is None:  # pinned until track G's generator version replaces the stub
        assert GENERATOR_VERSION_STUB == "freeform-v2/search-stub-1"
        assert value == "709ad2e70154ee5985d36595660e42604be45e5f584158da8e32088a561a98d4"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            {"pad": "v_funnel", "tip": "blunt", "pad_depth_mm": 6, "wall_draft_deg": 10, **BASE},
            "between 15 and 70",
        ),
        (
            {"pad": "v_funnel", "tip": "blunt", "pad_depth_mm": 2, "wall_draft_deg": 30, **BASE},
            "pad_depth_mm >= 3",
        ),
        (
            {
                "pad": "conform",
                "tip": "blunt",
                "pad_depth_mm": 0.5,
                "wall_draft_deg": 0,
                "clearance_mm": 0.4,
                **BASE,
            },
            "pad_depth_mm >= 1",
        ),
        ({"pad": "flat", "tip": "hook", "tip_size_mm": 1.5, **BASE}, "tip_size_mm >= 2"),
        (
            {"pad": "flat", "tip": "hook", "tip_size_mm": 8.0, **BASE, "pad_height_mm": 11.0},
            "needs pad_height_mm >= 12",
        ),
        (
            {"pad": "flat", "tip": "chamfer", "tip_size_mm": 8.0, **BASE, "pad_height_mm": 8.0},
            "tip_size_mm <= pad_height_mm - 0.5",
        ),
        (
            {"pad": "flat", "tip": "blunt", "grip_height_frac": 0.5, "pad_height_mm": 20.0},
            "needs finger_width_mm",
        ),
        ({"pad": "conform", "tip": "blunt", **BASE}, "pad 'conform' needs pad_depth_mm"),
        ({"pad": "flat", "tip": "chamfer", **BASE}, "tip 'chamfer' needs tip_size_mm"),
        ({"pad": "round", "tip": "blunt", **BASE}, "pad must be one of"),
        ({"pad": "flat", "tip": "blunt", "taper_ratio": 0.5, **BASE}, "unknown parameter(s): taper_ratio"),
        ({"pad": "flat", "tip": "blunt", **BASE, "finger_width_mm": float("nan")}, "must be a finite number"),
        ({"pad": "flat", "tip": "blunt", **BASE, "finger_width_mm": "wide"}, "must be a finite number"),
        ({"pad": "flat", "tip": "blunt", **BASE, "finger_width_mm": 44.0}, "outside 14 to 40"),
        # Magnitudes a JSON tool call can carry: rejected, never an OverflowError that ends the run.
        (
            {"pad": "flat", "tip": "blunt", **BASE, "pad_height_mm": 1e308},
            "pad_height_mm = 1e+308 is outside",
        ),
        ({"pad": "flat", "tip": "blunt", **BASE, "pad_height_mm": -1e308}, "is outside 8 to 30"),
        ({"pad": "flat", "tip": "blunt", **BASE, "finger_width_mm": 10**400}, "must be a finite number"),
    ],
)
def test_planted_invalid_recipes_are_rejected_with_specific_messages(raw: dict, message: str) -> None:
    check = SPACE.validate(raw, None)
    assert not check.ok and check.stage == "schema"
    assert any(message in e for e in check.errors), check.errors


def test_screen_order_duplicate_design_then_geometry_then_distance() -> None:
    ev = FakeEvaluatorV2(0)
    ctx = ev.describe().gripper_task
    chamfer0 = SPACE.recipe(pad="flat", tip="chamfer", tip_size_mm=0.3, **BASE)
    blunt = SPACE.recipe(pad="flat", tip="blunt", **BASE)
    near = SPACE.recipe(pad="flat", tip="blunt", **{**BASE, "finger_width_mm": 26.0})
    far = SPACE.recipe(pad="v_funnel", tip="hook", pad_depth_mm=8, wall_draft_deg=40, tip_size_mm=3, **BASE)
    same = SPACE.recipe(pad="flat", tip="blunt", pad_depth_mm=4.0, **BASE)
    assert SCREEN_REASONS == ("duplicate_design", "duplicate_geometry", "too_similar")
    hit = screen(same, [far, blunt], ["d01", "d02"], MIN_DISTANCE_V2, space=SPACE, context=ctx)
    assert hit is not None and hit.reason == "duplicate_design" and hit.near.label == "d02"
    hit = screen(chamfer0, [blunt], ["d02"], MIN_DISTANCE_V2, space=SPACE, context=ctx)
    assert hit is not None and hit.reason == "duplicate_geometry"  # a 0.3 mm chamfer builds a blunt jaw
    assert (
        screen(chamfer0, [blunt], ["d02"], MIN_DISTANCE_V2, space=SPACE, context=None) is None
    )  # no geometry
    hit = screen(near, [blunt], ["d02"], MIN_DISTANCE_V2, space=SPACE, context=ctx)
    assert hit is not None and hit.reason == "too_similar"
    assert screen(far, [blunt, near], None, MIN_DISTANCE_V2, space=SPACE, context=ctx) is None


#: The rule's bars with the schema rules only (no training parts): the stub mirrors track G's
#: schema rules, so these hold with or without G.
SCHEMA_ONLY_BARS = (0.080, 0.17)


def test_threshold_rule_reproduces_the_pinned_bars() -> None:
    rule = threshold_rule(SPACE, None)
    assert (rule.min_distance, rule.explore_min_distance) == SCHEMA_ONLY_BARS
    assert rule.n_pairs == 10_000 and rule.quantile == 0.01
    # The rule's own property: at most 1 % of the random valid pairs fall within t.
    designs = SPACE.sample_sobol(rule.seed, None).sample(rule.pool)
    rng = np.random.default_rng(rule.seed)
    i = rng.integers(0, rule.pool, size=rule.n_pairs)
    j = (i + rng.integers(1, rule.pool, size=rule.n_pairs)) % rule.pool
    d = np.array([SPACE.distance(designs[a], designs[b]) for a, b in zip(i, j, strict=True)])
    assert (d < rule.min_distance).mean() <= 0.01
    assert MIN_DISTANCE_V2 * 9 < 1.0  # below one categorical switch, so different cells always pass


def test_scored_bars_are_the_recorded_rule_on_the_training_parts() -> None:
    """The bars the arms use are the rule's output under the full validators (GV5), recorded with
    the hashes of the inputs it depends on; a change to those inputs needs a new recording."""
    import hashlib
    import json

    from rtt.search.threshold_v2 import DEFAULT_OUT

    rec = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    assert (rec["min_distance"], rec["explore_min_distance"]) == (MIN_DISTANCE_V2, EXPLORE_MIN_DISTANCE_V2)
    assert rec["n_pairs"] == 10_000 and rec["quantile"] == 0.01 and rec["pool"] == 2_000
    assert rec["pair_quantiles"]["p1"] >= MIN_DISTANCE_V2  # at most 1 % of the pairs within t
    assert rec["generator_version"] == track_g().generator_version
    root = DEFAULT_OUT.parents[1]
    for rel, digest in rec["inputs_sha256"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel


def test_parallel_valid_pool_matches_the_sequential_sampler() -> None:
    from rtt.search.diversity import valid_sobol_pool

    seq, n_seq = valid_sobol_pool(SPACE, None, 123, 40)
    par, n_par = valid_sobol_pool(SPACE, None, 123, 40, workers=2, chunk=16)
    assert n_seq == n_par
    assert [SPACE.design_id(d) for d in seq] == [SPACE.design_id(d) for d in par]


def test_llm_duplicates_are_rejected_named_and_reflected() -> None:
    ctx, items = v2_round0()
    fresh = diverse_recipes(ctx, 4)
    parent = next(i for i in items if i.proposal.origin == "sobol").design
    first = submission_v2(fresh)
    first["mutate_1"] = recipe_slot(parent, ("d05",), "copy with an inactive extra")
    params = SPACE.params(parent)
    if params["pad"] == "flat":
        first["mutate_1"]["params"]["pad_depth_mm"] = 3.0
    elif params["tip"] == "blunt":
        first["mutate_1"]["params"]["tip_size_mm"] = 2.5
    for name in ("reserve_1", "reserve_2", "reserve_3", "reserve_4"):
        first[name] = {**recipe_slot(parent, (), "dup reserve"), "operator": "explore"}
    reflected = submission_v2(diverse_recipes(ctx, 4, seed=123))
    transport = ReplyTransport(first, reflected, first, reflected)
    proposer = NemotronProposerV2(ctx, client_for(transport))
    proposer.tell(items)
    out = proposer.ask(1, 8)
    assert len(out) == 8
    reasons = [r.reason for r in ctx.rejections]
    assert reasons[0] == "duplicate_design"
    assert ctx.rejections[0].slot == "mutate_1" and "after canonicalisation" in ctx.rejections[0].errors[0]
    assert any(c.kind == "reflection" for c in ctx.llm_calls)
    reflection = next(p for p in transport.payloads if "# Too similar" in p["messages"][-1]["content"])
    assert (
        "the same design as d05 once inactive parameters are dropped" in reflection["messages"][-1]["content"]
    )
    simulated = {SPACE.design_id(i.design) for i in ctx.history}
    assert not any(SPACE.design_id(p.design) in simulated for p in out)
