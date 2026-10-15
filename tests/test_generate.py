"""Finger generator: box count/well-formedness, frame convention, features, mass."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from rtt.design.generate import (
    BOX_ROLES,
    CORE_INDICES,
    TIP_INDICES,
    box_corners,
    build_finger,
    generate_finger,
    geometry_bounds,
    is_parked,
    mass_and_inertia,
    matrix_from_quat,
    point_inside,
)
from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.design.types import BOXES_PER_FINGER, FingerGeometry
from rtt.design.validate import validate

MM = 1e-3
TOL = 1e-9
TASK = GripperTask()

BASE: dict[str, object] = {
    "finger_length_mm": 80.0,
    "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0,
    "bend_angle_deg": 0.0,
    "bend_start_frac": 0.5,
    "inner_face": "flat",
    "inner_face_depth_mm": 0.0,
    "tip_profile": "blunt",
    "tip_size_mm": 0.0,
    "taper_ratio": 1.0,
}


def make(**overrides: object) -> FingerDesign:
    return FingerDesign(**{**BASE, **overrides})


def _float(key: str) -> st.SearchStrategy[float]:
    lo, hi = PARAM_BOUNDS[key]
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


any_design = st.builds(
    FingerDesign,
    **{key: _float(key) for key in PARAM_BOUNDS},
    inner_face=st.sampled_from(CATEGORICAL_OPTIONS["inner_face"]),
    tip_profile=st.sampled_from(CATEGORICAL_OPTIONS["tip_profile"]),
)


@st.composite
def plausible_design(draw: st.DrawFn) -> FingerDesign:
    """Designs drawn so that most pass validation (parameter rules satisfied by construction)."""
    thick = draw(_float("finger_thickness_mm"))
    width = draw(_float("finger_width_mm"))
    face = draw(st.sampled_from(CATEGORICAL_OPTIONS["inner_face"]))
    max_depth = 0.0 if face == "flat" else 0.5 * thick
    if face == "v_groove":
        max_depth = min(max_depth, 0.35 * width)
    tip = draw(st.sampled_from(CATEGORICAL_OPTIONS["tip_profile"]))
    max_tip = min(15.0, thick + 6.0) if tip == "hook" else 15.0
    return FingerDesign(
        finger_length_mm=draw(_float("finger_length_mm")),
        finger_thickness_mm=thick,
        finger_width_mm=width,
        bend_angle_deg=draw(st.floats(-15.0, 20.0)),
        bend_start_frac=draw(st.floats(0.5, 0.9)),
        inner_face=face,
        inner_face_depth_mm=draw(st.floats(0.0, max_depth)),
        tip_profile=tip,
        tip_size_mm=draw(st.floats(0.0, max_tip)),
        taper_ratio=draw(_float("taper_ratio")),
    )


def _assert_well_formed(geom: FingerGeometry) -> None:
    assert len(geom.boxes) == BOXES_PER_FINGER
    for box in geom.boxes:
        values = (*box.center, *box.half_size, *box.quat_wxyz)
        assert all(math.isfinite(v) for v in values)
        assert min(box.half_size) > 0.0
        assert math.isclose(sum(q * q for q in box.quat_wxyz), 1.0, rel_tol=1e-9)


def _hook_opening_is_open(design: FingerDesign) -> bool:
    """A point just above the hook lip, in front of the inner face, is empty.

    Only meaningful when the probe is clear of the bend's concave corner, which is
    guaranteed when the straight part before the lip is longer than the thickness.
    """
    build = build_finger(design)
    assert build.distal_length_mm - build.tip_region_mm > design.finger_thickness_mm + 0.5
    lip = build.geometry.boxes[TIP_INDICES[0]]
    rot = matrix_from_quat(lip.quat_wxyz)
    toward_root, inward = rot[:, 2], -rot[:, 1]
    probe = (
        np.asarray(lip.center)
        + toward_root * (lip.half_size[2] + 0.5 * MM)
        + inward * (lip.half_size[1] - 0.5 * design.tip_size_mm * MM)
    )
    return not point_inside(build.geometry, probe)


def _has_hook_opening(design: FingerDesign) -> bool:
    if design.tip_profile != "hook" or design.tip_size_mm <= 1.0:
        return False
    build = build_finger(design)
    return build.distal_length_mm - build.tip_region_mm > design.finger_thickness_mm + 0.5


# --- well-formedness over the whole schema ------------------------------------------


@settings(max_examples=300, deadline=None)
@given(any_design)
def test_every_schema_valid_design_gives_14_well_formed_boxes(design: FingerDesign) -> None:
    _assert_well_formed(generate_finger(design))


@settings(max_examples=50, deadline=None)
@given(any_design)
def test_generation_is_deterministic(design: FingerDesign) -> None:
    assert generate_finger(design) == generate_finger(design.model_copy())


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(plausible_design())
def test_valid_designs_meet_section_gap_and_mount_rules(design: FingerDesign) -> None:
    result = validate(design, TASK)
    assume(result.ok)
    geom = generate_finger(design)
    _assert_well_formed(geom)
    for i in CORE_INDICES:  # the load-bearing core of every segment
        half = geom.boxes[i].half_size
        assert 2 * half[0] >= TASK.min_section_mm * MM - TOL
        assert 2 * half[1] >= TASK.min_section_mm * MM - TOL
    lo, hi = geometry_bounds(geom)
    protrusion_mm = max(0.0, -lo[1]) / MM
    assert TASK.max_opening_mm - 2 * protrusion_mm >= TASK.required_open_gap_mm - 1e-6
    assert protrusion_mm < TASK.closed_gap_mm / 2 + TASK.stroke_mm  # clear of each other when open
    assert hi[2] <= TOL
    if _has_hook_opening(design):
        assert _hook_opening_is_open(design)


def _assert_sections_from_boxes(design: FingerDesign, geom: FingerGeometry) -> None:
    """Minimum-section rule re-derived from the box sizes, independently of the validator."""
    min_m = TASK.min_section_mm * MM - TOL
    for i in CORE_INDICES:  # core: width x (thickness minus groove depth)
        half = geom.boxes[i].half_size
        assert 2 * half[0] >= min_m and 2 * half[1] >= min_m
    if design.inner_face == "concave_arc":  # lip widths
        for i in range(TIP_INDICES[0]):
            if i not in CORE_INDICES and not is_parked(geom.boxes[i]):
                assert 2 * geom.boxes[i].half_size[0] >= min_m
    lip = geom.boxes[TIP_INDICES[0]]
    if design.tip_profile == "hook" and not is_parked(lip):  # hook lip along the finger
        assert 2 * lip.half_size[2] >= min_m


def test_sweep_of_10k_valid_random_designs() -> None:
    """Gate G0: rejection-sample 10k valid designs; each one satisfies the invariants.

    The invariants are recomputed from the generated boxes rather than read back from
    the validator's own metrics, so a bookkeeping error in the validator cannot hide.
    """
    rng = np.random.default_rng(20261015)
    target, n_valid, n_drawn = 10_000, 0, 0
    while n_valid < target:
        n_drawn += 1
        assert n_drawn <= 5 * target, f"only {n_valid} valid designs in {n_drawn} draws"
        params = {key: float(rng.uniform(lo, hi)) for key, (lo, hi) in PARAM_BOUNDS.items()}
        params["inner_face"] = str(rng.choice(CATEGORICAL_OPTIONS["inner_face"]))
        params["tip_profile"] = str(rng.choice(CATEGORICAL_OPTIONS["tip_profile"]))
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        design = FingerDesign(**params)
        result = validate(design, TASK)
        if not result.ok:
            assert result.errors
            continue
        n_valid += 1
        geom = generate_finger(design)
        _assert_well_formed(geom)
        _assert_sections_from_boxes(design, geom)
        lo, hi = geometry_bounds(geom)
        open_gap_mm = TASK.max_opening_mm + 2 * lo[1] / MM
        assert open_gap_mm >= TASK.required_open_gap_mm - 1e-6
        assert result.metrics["open_gap_mm"] == pytest.approx(open_gap_mm, abs=1e-9)
        assert -lo[1] / MM < TASK.closed_gap_mm / 2 + TASK.stroke_mm  # clear of each other when open
        assert hi[2] <= TOL  # nothing above the mount
        assert mass_and_inertia(geom).mass_kg * 1e3 <= TASK.mass_budget_g + 1e-9
        if _has_hook_opening(design):
            assert _hook_opening_is_open(design)
    # Random search rejection-samples valid designs, so a healthy share must be valid.
    assert 0.2 <= n_valid / n_drawn <= 0.95


# --- frame convention and features ------------------------------------------------


def test_box_roles_layout() -> None:
    assert len(BOX_ROLES) == BOXES_PER_FINGER
    assert [BOX_ROLES[i] for i in CORE_INDICES] == [f"segment{i}_centre" for i in range(4)]
    assert BOX_ROLES[12:] == ("tip_a", "tip_b")


def test_straight_finger_follows_frame_convention() -> None:
    geom = generate_finger(make())
    lo, hi = geometry_bounds(geom)
    assert np.allclose(lo, [-0.010, 0.0, -0.080], atol=TOL)
    assert np.allclose(hi, [0.010, 0.010, 0.0], atol=TOL)
    assert all(box.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0)) for box in geom.boxes)
    assert sum(not is_parked(b) for b in geom.boxes) == 4  # just the 4 core segments


def test_parked_boxes_sit_inside_the_root_core() -> None:
    geom = generate_finger(make())
    core = geom.boxes[CORE_INDICES[0]]
    for box in geom.boxes:
        if is_parked(box):
            local = box_corners(box) - np.asarray(core.center)
            assert np.all(np.abs(local) < np.asarray(core.half_size))


def test_positive_bend_moves_tip_inward_and_negative_outward() -> None:
    inward = geometry_bounds(generate_finger(make(bend_angle_deg=20.0)))
    outward = geometry_bounds(generate_finger(make(bend_angle_deg=-10.0)))
    distal, thick = 40.0 * MM, 10.0 * MM
    up, down = math.radians(20.0), math.radians(10.0)
    # Inward bend pivots on the back corner, so the inner face starts slightly behind y=0.
    assert inward[0][1] == pytest.approx(thick * (1 - math.cos(up)) - distal * math.sin(up), abs=1e-9)
    assert outward[1][1] == pytest.approx(distal * math.sin(down) + thick * math.cos(down), abs=1e-9)
    assert outward[0][1] == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize("angle", [-15.0, -5.0, 10.0, 30.0, 45.0])
def test_bend_leaves_no_notch_on_the_convex_side(angle: float) -> None:
    """Material just past the bend line, near the convex face, must be solid."""
    geom = generate_finger(make(bend_angle_deg=angle, bend_start_frac=0.6))
    s_bend = 0.6 * 80.0
    convex_y = 10.0 - 0.5 if angle > 0 else 0.5  # back face for inward bends, inner face otherwise
    assert point_inside(geom, (0.0, convex_y * MM, -(s_bend + 0.2) * MM))
    for depth in np.linspace(0.5, 9.5, 10):
        assert point_inside(geom, (0.0, depth * MM, -(s_bend - 0.1) * MM))


def test_concave_arc_recesses_the_centre_between_lips() -> None:
    geom = generate_finger(make(inner_face="concave_arc", inner_face_depth_mm=4.0))
    z = -20.0 * MM
    assert not point_inside(geom, (0.0, 2.0 * MM, z))  # inside the channel
    assert point_inside(geom, (0.0, 4.5 * MM, z))  # channel floor
    assert point_inside(geom, (9.0 * MM, 1.0 * MM, z))  # lip
    assert geometry_bounds(geom)[0][1] == pytest.approx(0.0, abs=TOL)


def test_v_groove_has_sloped_flanks() -> None:
    geom = generate_finger(make(inner_face="v_groove", inner_face_depth_mm=4.0))
    z = -20.0 * MM
    # The face line runs from (x=10, y=0) to (x=0, y=4): probe just in front of / behind it.
    for x in (2.0, 5.0, 8.0):
        face_y = 4.0 * (1 - x / 10.0)
        assert not point_inside(geom, (x * MM, (face_y - 0.2) * MM, z))
        assert point_inside(geom, (x * MM, (face_y + 0.2) * MM, z))
        assert point_inside(geom, (-x * MM, (face_y + 0.2) * MM, z))
    assert not all(b.quat_wxyz == pytest.approx((1, 0, 0, 0)) for b in geom.boxes)
    assert geometry_bounds(geom)[1][0] == pytest.approx(10.0 * MM, abs=TOL)  # never wider


def test_hook_protrudes_inward_and_keeps_its_opening() -> None:
    design = make(tip_profile="hook", tip_size_mm=8.0)
    geom = generate_finger(design)
    assert geometry_bounds(geom)[0][1] == pytest.approx(-8.0 * MM, abs=TOL)
    assert point_inside(geom, (0.0, -4.0 * MM, -79.0 * MM))  # inside the lip
    assert not point_inside(geom, (0.0, -4.0 * MM, -70.0 * MM))  # the hook opening
    assert _hook_opening_is_open(design)


def test_chamfer_cuts_the_inner_tip_corner() -> None:
    geom = generate_finger(make(tip_profile="chamfer", tip_size_mm=4.0))
    assert not point_inside(geom, (0.0, 0.5 * MM, -79.5 * MM))
    assert point_inside(geom, (0.0, 9.5 * MM, -79.5 * MM))
    assert point_inside(geom, (0.0, 4.5 * MM, -79.9 * MM))
    assert geometry_bounds(geom)[0][2] == pytest.approx(-0.080, abs=TOL)


def test_rounded_tip_cuts_both_corners() -> None:
    geom = generate_finger(make(tip_profile="rounded", tip_size_mm=5.0))
    assert not point_inside(geom, (0.0, 0.5 * MM, -79.5 * MM))
    assert not point_inside(geom, (0.0, 9.5 * MM, -79.5 * MM))
    assert point_inside(geom, (0.0, 5.0 * MM, -79.9 * MM))


def test_blunt_cap_closes_the_groove_at_the_tip() -> None:
    capped = generate_finger(make(inner_face="concave_arc", inner_face_depth_mm=4.0, tip_size_mm=5.0))
    assert point_inside(capped, (0.0, 2.0 * MM, -78.0 * MM))
    assert not point_inside(capped, (0.0, 2.0 * MM, -70.0 * MM))


def test_capped_tip_sizes_are_reported() -> None:
    build = build_finger(make(tip_profile="chamfer", tip_size_mm=15.0))
    assert build.tip_feature_mm == pytest.approx(5.0)  # half of the 10 mm tip thickness
    assert any("capped" in note for note in build.notes)


def test_taper_shrinks_sections_toward_the_tip() -> None:
    build = build_finger(make(taper_ratio=0.5))
    widths = [build.geometry.boxes[i].half_size[0] for i in CORE_INDICES]
    assert widths == sorted(widths, reverse=True)
    assert widths[-1] < 0.7 * widths[0]
    assert build.min_section.size_mm == pytest.approx(10.0 * (1 - 0.5 * 0.875))


# --- mass properties ---------------------------------------------------------------


def test_mass_and_inertia_of_a_straight_slab() -> None:
    geom = generate_finger(make())
    props = mass_and_inertia(geom)
    w, t, length = 0.020, 0.010, 0.080
    mass = 1240.0 * w * t * length
    assert props.mass_kg == pytest.approx(mass, rel=1e-4)  # + the tiny parked boxes
    assert props.com_m == pytest.approx((0.0, t / 2, -length / 2), abs=1e-6)
    expected = (mass / 12) * np.diag([t**2 + length**2, w**2 + length**2, w**2 + t**2])
    assert np.allclose(props.inertia_com, expected, rtol=1e-4, atol=1e-12)
    com = np.asarray(props.com_m)
    shifted = props.inertia_com + props.mass_kg * (com @ com * np.eye(3) - np.outer(com, com))
    assert np.allclose(props.inertia_root, shifted, rtol=1e-6, atol=1e-12)


def test_principal_axes_reconstruct_the_inertia() -> None:
    geom = generate_finger(make(bend_angle_deg=25.0, inner_face="v_groove", inner_face_depth_mm=3.0))
    props = mass_and_inertia(geom)
    rot = matrix_from_quat(props.principal_quat_wxyz)
    rebuilt = rot @ np.diag(props.principal_moments) @ rot.T
    assert np.allclose(rebuilt, props.inertia_com, rtol=1e-8, atol=1e-14)
    assert min(props.principal_moments) > 0.0
    assert mass_and_inertia(geom, density_kg_m3=2480.0).mass_kg == pytest.approx(2 * props.mass_kg)
