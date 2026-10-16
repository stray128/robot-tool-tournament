"""Validators: planted invalid recipes are rejected with specific codes and messages (GV0)."""

from __future__ import annotations

import numpy as np
import pytest

from freeform_parts import (
    flanged_bushing,
    hex_standoff,
    tall_block,
    tall_post,
    thin_plate,
    training_parts,
    wide_block,
)
from rtt.design.freeform import validate as V
from rtt.design.freeform.baselines import load_entrants_v2
from rtt.design.freeform.generate import build_pair_full
from rtt.design.freeform.schema import FingerRecipe
from rtt.design.freeform.tip import Piece, piece_from_box
from rtt.design.freeform.validate import CODES, InvalidDesign, validate_design
from rtt.sim.task import default_task

TASK = default_task()
OK = {
    "pad": "conform",
    "tip": "blunt",
    "grip_height_frac": 0.15,
    "pad_height_mm": 12.0,
    "finger_width_mm": 40.0,
    "pad_depth_mm": 3.0,
    "wall_draft_deg": 40.0,
    "clearance_mm": 0.4,
}


def _raw(**changes: object) -> dict:
    return {**OK, **changes}


#: (label, recipe fields, parts, expected code, words the message must contain)
PLANTED = [
    ("unknown field", _raw(finger_length_mm=80.0), None, "schema", "finger_length_mm"),
    ("too wide", _raw(finger_width_mm=44.0), None, "schema", "finger_width_mm"),
    (
        "v_funnel too steep",
        dict(_raw(pad="v_funnel", wall_draft_deg=10.0), pad_depth_mm=5.0),
        None,
        "schema",
        "v_funnel needs wall_draft_deg",
    ),
    ("conform too shallow", _raw(pad_depth_mm=0.5), None, "schema", "conform needs pad_depth_mm"),
    ("hook too small", _raw(tip="hook", tip_size_mm=1.0), None, "schema", "hook needs tip_size_mm"),
    (
        "hook does not fit",
        _raw(tip="hook", tip_size_mm=6.0, pad_height_mm=9.0),
        None,
        "schema",
        "pad_height_mm >= 10",
    ),
    (
        "deep straight imprint",
        _raw(pad_depth_mm=15.0, wall_draft_deg=0.0, grip_height_frac=0.5),
        (hex_standoff,),
        "q_seat_range",
        "hex_standoff",
    ),
    ("nothing in the band", _raw(), (thin_plate,), "outline_empty", "thin_plate"),
    ("finger too long", _raw(), (tall_post,), "length", "118.00 mm"),
    (
        "part too wide to open around",
        _raw(pad="flat", grip_height_frac=0.5),
        (wide_block,),
        "open_clearance",
        "wide_block",
    ),
    (
        "too heavy",
        _raw(pad_height_mm=30.0, pad_depth_mm=8.0, wall_draft_deg=0.0, clearance_mm=0.2, body_style="solid"),
        (tall_block,),
        "mass",
        "g of PLA",
    ),
    (
        "pad just under the flange",
        _raw(grip_height_frac=0.63, pad_height_mm=12.0, finger_width_mm=34.0, pad_depth_mm=1.7),
        (flanged_bushing,),
        "seat_distance",
        "flanged_bushing",
    ),
]


def test_every_code_is_documented() -> None:
    for _, _, _, code, _ in PLANTED:
        assert code in CODES
    with pytest.raises(ValueError):
        InvalidDesign("no_such_rule", "x")


@pytest.mark.parametrize(("label", "fields", "parts", "code", "words"), PLANTED, ids=[p[0] for p in PLANTED])
def test_planted_invalid_recipes_are_rejected(label, fields, parts, code, words) -> None:
    if parts is None:  # schema-level: rejected before any part is seen
        with pytest.raises(ValueError, match=words):
            FingerRecipe(**fields)
        report = validate_design(fields, training_parts(), TASK)
        assert not report.valid and report.code == "schema"
        return
    recipe = FingerRecipe(**fields)
    report = validate_design(recipe, [p() for p in parts], TASK)
    assert not report.valid, label
    assert report.code == code, (label, report.message)
    assert words in report.message and report.message.startswith(code)
    with pytest.raises(InvalidDesign) as info:
        build_pair_full(recipe, parts[0](), TASK, use_cache=False)
    assert info.value.code == code


def test_at_least_eight_planted_rejections() -> None:
    assert len(PLANTED) >= 8
    assert len({p[3] for p in PLANTED}) >= 7


def test_valid_design_reports_hashes_and_printability() -> None:
    recipe = FingerRecipe(**OK)
    report = validate_design(recipe, training_parts(), TASK)
    assert report.valid, report.message
    assert len(report.pairs) == 6 and len(report.geometry_hashes) == 6
    assert report.geometry_id and len(report.geometry_id) == 64
    for side in report.printability.values():
        assert side["left"]["mass_g"] < 50 and 0.0 <= side["left"]["overhang_share"] < 1.0
    dup = validate_design(recipe, training_parts(), TASK, seen_geometry_ids={report.geometry_id})
    assert not dup.valid and dup.code == "duplicate_geometry"


def test_inactive_differences_are_duplicate_geometry() -> None:
    a = FingerRecipe(pad="flat", tip="blunt", grip_height_frac=0.15, pad_height_mm=12.0, finger_width_mm=40.0)
    b = FingerRecipe(**{**a.model_dump(exclude_none=True), "pad_depth_mm": 9.0, "clearance_mm": 1.0})
    parts = training_parts()
    ra = validate_design(a, parts, TASK)
    assert ra.valid, ra.message
    rb = validate_design(b, parts, TASK, seen_geometry_ids={ra.geometry_id})
    assert rb.code == "duplicate_geometry" and rb.design_id == ra.design_id


def test_round0_entrants_valid_on_the_procedural_parts_they_can_reach() -> None:
    entrants = {b.name: b for b in load_entrants_v2()}
    parts = [p for p in training_parts() if p.name != "l_bracket"]
    for name in ("flat", "imprint_configurator", "imprint_swept"):
        report = validate_design(entrants[name], parts, TASK)
        assert report.valid, (name, report.message)


# ----------------------------------------------------------------------------- checkers


def test_convexity_checker_catches_a_reflex_piece() -> None:
    good = piece_from_box((0, 0, 0), (0.01, 0.01, 0.01), "pad")
    V.check_convex([good], "p", "left")
    # a dented hexahedron: move one top vertex inward so a face folds in
    v = good.vertices.copy()
    top = np.flatnonzero((v[:, 2] > 0.005) & (v[:, 0] > 0.005) & (v[:, 1] > 0.005))
    v[top[0]] = [0.005, 0.005, 0.005]
    with pytest.raises(InvalidDesign, match="not_convex"):
        V.check_convex([Piece(v, good.faces, "pad")], "p", "left")


def test_seat_checkers() -> None:
    with pytest.raises(InvalidDesign, match="pad_cuts_part"):
        V.check_face_behind_outline(np.array([0.94, 0.9]), np.array([0.95, -np.inf]), "p", "left")
    V.check_face_behind_outline(np.array([1.0, 0.9]), np.array([1.0, 0.9]), "p", "left")
    with pytest.raises(InvalidDesign, match="jaws_intersect"):
        V.check_jaws_apart(0.0, "p")
    with pytest.raises(InvalidDesign, match="table_clearance"):
        V.check_table(0.0015, "p")
    with pytest.raises(InvalidDesign, match="pads_overlap"):
        V.check_pads_apart(np.array([0.002, 0.0005]), "p")
    with pytest.raises(InvalidDesign, match="too_many_pieces"):
        V.check_piece_count(41, "p", "left")
    with pytest.raises(InvalidDesign, match="print_not_manifold"):
        V.check_print_manifold(2, True, "p", "left")
    with pytest.raises(InvalidDesign, match="width"):
        V.check_width(0.041, "p")


def test_printability_checkers() -> None:
    from types import SimpleNamespace

    from rtt.design.freeform.export import JawMesh
    from rtt.design.freeform.printability import check_printability

    box = piece_from_box((0, 0, -0.01), (0.01, 0.01, 0.0), "pad")
    mesh = JawMesh(box.vertices, box.faces)
    ok = {
        "min_strip_wall": 0.003,
        "hook_height": None,
        "min_body_wall": 0.006,
        "feature_excess": 0.0,
        "mass": 0.02,
        "body_slope": 0.5,
        "mesh": mesh,
    }
    out = check_printability(SimpleNamespace(**ok), "p", "left")
    assert out["mass_g"] == pytest.approx(20.0) and out["overhang_share"] == pytest.approx(0.0)
    for change, code in (
        ({"min_strip_wall": 0.001}, "min_wall"),
        ({"hook_height": 0.001}, "min_wall"),
        ({"feature_excess": 0.002}, "min_feature"),
        ({"mass": 0.051}, "mass"),
        ({"body_slope": 1.2}, "body_slope"),
    ):
        with pytest.raises(InvalidDesign, match=code):
            check_printability(SimpleNamespace(**{**ok, **change}), "p", "left")
