"""Canonicalisation, design ids, geometry ids and the recipe distance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rtt.design.freeform.baselines import load_entrants_v2
from rtt.design.freeform.canon import canonical_dict, canonical_json, canonicalize
from rtt.design.freeform.distance import (
    pairwise_distances,
    recipe_cell,
    recipe_distance,
    recipe_vector,
    too_similar,
    vectors_distance,
)
from rtt.design.freeform.hashing import GENERATOR_VERSION, design_id, geometry_hash, geometry_id
from rtt.design.freeform.schema import FingerRecipe, recipe_from_unit

BASE = {
    "pad": "conform",
    "tip": "blunt",
    "grip_height_frac": 0.5,
    "pad_height_mm": 20.0,
    "finger_width_mm": 30.0,
}
CONFORM = dict(BASE, pad_depth_mm=4.0, wall_draft_deg=30.0, clearance_mm=0.4)


def test_canonicalize_resets_inactive_and_logs() -> None:
    raw = FingerRecipe(**dict(BASE, pad="flat", pad_depth_mm=12.0, clearance_mm=0.9, tip_size_mm=3.0))
    canon, repairs = canonicalize(raw)
    assert canon.pad_depth_mm == 0.0 and canon.wall_draft_deg == 0.0 and canon.clearance_mm == 0.2
    assert canon.tip_size_mm == 0.0
    text = " ".join(repairs)
    assert "pad_depth_mm is inactive" in text and "clearance_mm is inactive" in text and "tip_size_mm" in text
    assert "wall_draft_deg" not in text  # it was not given, so nothing changed


def test_canonicalize_quantises_and_logs() -> None:
    raw = FingerRecipe(**dict(CONFORM, pad_depth_mm=4.04, wall_draft_deg=30.3, grip_height_frac=0.503))
    canon, repairs = canonicalize(raw)
    assert (canon.pad_depth_mm, canon.wall_draft_deg, canon.grip_height_frac) == (4.0, 30.5, 0.5)
    assert any("quantised" in r for r in repairs)
    again, more = canonicalize(canon)
    assert again == canon and more == ()


def test_inactive_only_differences_give_identical_ids() -> None:
    a = FingerRecipe(**dict(BASE, pad="flat"))
    b = FingerRecipe(
        **dict(BASE, pad="flat", pad_depth_mm=9.0, wall_draft_deg=60.0, clearance_mm=1.5, tip_size_mm=4.0)
    )
    c = FingerRecipe(**dict(BASE, pad="flat", name="other", rationale="a reason"))
    assert design_id(a) == design_id(b) == design_id(c)
    assert canonical_json(a) == canonical_json(b)
    d = FingerRecipe(**dict(CONFORM, tip="blunt", tip_size_mm=5.0))
    e = FingerRecipe(**CONFORM)
    assert design_id(d) == design_id(e)
    v1 = FingerRecipe(**dict(BASE, pad="v_funnel", pad_depth_mm=5.0, wall_draft_deg=40.0, clearance_mm=1.3))
    v2 = FingerRecipe(**dict(BASE, pad="v_funnel", pad_depth_mm=5.0, wall_draft_deg=40.0))
    assert design_id(v1) == design_id(v2)


def test_quantisation_noise_gives_identical_ids() -> None:
    a = FingerRecipe(**CONFORM)
    b = FingerRecipe(**dict(CONFORM, pad_depth_mm=4.0000001, finger_width_mm=30.04))
    assert design_id(a) == design_id(b)


def test_active_differences_change_the_id() -> None:
    a = FingerRecipe(**CONFORM)
    for change in (
        {"pad_depth_mm": 4.1},
        {"wall_draft_deg": 30.5},
        {"clearance_mm": 0.5},
        {"tip": "chamfer", "tip_size_mm": 2.0},
        {"body_style": "solid"},
        {"locator": "conical_pin"},
    ):
        assert design_id(FingerRecipe(**dict(CONFORM, **change))) != design_id(a), change


def test_design_id_format_and_version() -> None:
    did = design_id(FingerRecipe(**CONFORM))
    assert len(did) == 64 and int(did, 16) >= 0
    assert GENERATOR_VERSION.startswith("rtt-freeform-gen/")
    with pytest.raises(TypeError):
        design_id(object())


def test_canonical_dict_has_scored_fields_and_options_only() -> None:
    d = canonical_dict(FingerRecipe(**dict(CONFORM, name="x")))
    assert set(d) == {
        "pad",
        "tip",
        "grip_height_frac",
        "pad_height_mm",
        "finger_width_mm",
        "pad_depth_mm",
        "wall_draft_deg",
        "clearance_mm",
        "tip_size_mm",
        "locator",
        "body_style",
    }
    assert json.loads(canonical_json(FingerRecipe(**CONFORM))) == d


def test_geometry_hash_and_id() -> None:
    p = [np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]], float)]
    h1 = geometry_hash(p, p, 0.003, 0.03)
    assert h1 == geometry_hash([p[0] + 1e-9], p, 0.003, 0.03)  # below the 1e-7 m grid
    assert h1 != geometry_hash([p[0] + 2e-7], p, 0.003, 0.03)
    assert h1 != geometry_hash(p, p, 0.004, 0.03)
    assert h1 != geometry_hash(p, p, 0.003, 0.031)
    gid = geometry_id({"b": h1, "a": "0" * 64})
    assert gid == geometry_id([("a", "0" * 64), ("b", h1)])
    assert gid != geometry_id({"a": h1, "b": "0" * 64})
    with pytest.raises(ValueError):
        geometry_id({})


def test_entrant_points_share_the_recipe_id() -> None:
    entrants = {b.name: b for b in load_entrants_v2()}
    flat = entrants["flat"]
    assert flat.is_point and design_id(flat) == design_id(flat.recipe)
    imprint = entrants["imprint_configurator"]
    swept = entrants["imprint_swept"]
    assert not imprint.is_point and design_id(imprint) != design_id(imprint.recipe)
    assert design_id(imprint) != design_id(swept)
    assert len({design_id(b) for b in entrants.values()}) == len(entrants)


def test_distance_is_normalised_mean_of_nine_terms() -> None:
    a = FingerRecipe(**CONFORM)
    assert recipe_distance(a, a) == 0.0
    b = FingerRecipe(**dict(CONFORM, pad_depth_mm=4.0 + 1.5))  # 10% of the 0-15 mm range
    assert recipe_distance(a, b) == pytest.approx(0.1 / 9)
    c = FingerRecipe(**dict(CONFORM, tip="chamfer", tip_size_mm=0.8))  # categorical + 10% of tip range
    assert recipe_distance(a, c) == pytest.approx((1.0 + 0.1) / 9)
    flat_a = FingerRecipe(**dict(BASE, pad="flat"))
    flat_b = FingerRecipe(**dict(BASE, pad="flat", pad_depth_mm=15.0))  # inactive difference
    assert recipe_distance(flat_a, flat_b) == 0.0
    assert recipe_cell(c) == "conform/chamfer"


def test_distance_vectorised_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    recipes = []
    while len(recipes) < 12:
        try:
            recipes.append(recipe_from_unit(rng.random(9)))
        except ValueError:
            pass
    m = pairwise_distances(recipes)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 0.0)
    for i in range(4):
        for j in range(4):
            assert m[i, j] == pytest.approx(recipe_distance(recipes[i], recipes[j]))
    va = recipe_vector(recipes[0])
    assert vectors_distance(va, recipe_vector(recipes[1])) == pytest.approx(
        recipe_distance(recipes[0], recipes[1])
    )
    assert too_similar(recipes[0], [recipes[0]], 0.01)
    assert not too_similar(recipes[0], [], 0.01)
