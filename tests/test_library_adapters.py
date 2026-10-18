"""Generated ISO 9409-1 flange adapter plates: geometry, provenance, bolts and files."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest
import trimesh

from rtt.library.adapters import (
    CONFIRMED_ADAPTERS,
    MAX_THREAD_DEPTH_MM,
    AdapterResult,
    AdapterSpec,
    Feature,
    bolt_check,
    default_key,
    generate,
    recommend_bolt,
    write_adapter,
)
from rtt.library.mounts import CONFIRMED_PATTERNS

ISO50, ISO31 = CONFIRMED_PATTERNS
EXPECTED_OD = {
    "plate-iso50-4-m6": 86.0,  # 50 + 2 x 3 x 6
    "plate-iso31p5-4-m5": 62.0,  # 31.5 + 2 x 3 x 5 = 61.5, rounded up
    "adapter-iso50-4-m6-to-iso31p5-4-m5": 86.0,
}
ADAPTER_KEY = "adapter-iso50-4-m6-to-iso31p5-4-m5"
# A small offset keeps the vertical ray off triangle edges at a hole centre.
RAY_OFFSET = (0.0137, 0.0291)


@pytest.fixture(scope="module")
def results() -> dict[str, AdapterResult]:
    return {spec.key: generate(spec) for spec in CONFIRMED_ADAPTERS}


def z_hits(mesh: trimesh.Trimesh, x: float, y: float) -> list[float]:
    """Heights where a vertical line through (x, y) crosses the surface, low to high."""
    px, py = x + RAY_OFFSET[0], y + RAY_OFFSET[1]
    tri = mesh.triangles
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    p = np.array([px, py])
    area = cross(b[:, :2] - a[:, :2], c[:, :2] - a[:, :2])
    ok = np.abs(area) > 1e-12
    safe = np.where(ok, area, 1.0)
    wa = cross(b[:, :2] - p, c[:, :2] - p) / safe
    wb = cross(c[:, :2] - p, a[:, :2] - p) / safe
    wc = 1.0 - wa - wb
    inside = ok & (wa >= 0) & (wb >= 0) & (wc >= 0)
    z = wa * a[:, 2] + wb * b[:, 2] + wc * c[:, 2]
    return sorted({round(float(v), 6) for v in z[inside]})


def features(result: AdapterResult, prefix: str) -> list[Feature]:
    return [f for f in result.features if f.name.startswith(prefix)]


def wall(a: Feature, b: Feature) -> float:
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) - (a.diameter_mm + b.diameter_mm) / 2.0


def test_confirmed_set_and_keys() -> None:
    keys = [s.key for s in CONFIRMED_ADAPTERS]
    assert keys == ["plate-iso50-4-m6", "plate-iso31p5-4-m5", ADAPTER_KEY]
    for spec in CONFIRMED_ADAPTERS:
        assert spec.key == default_key(spec.robot, spec.tool)
        assert AdapterSpec(robot=spec.robot, tool=spec.tool).key == spec.key


@pytest.mark.parametrize("key", list(EXPECTED_OD))
def test_mesh_is_a_closed_solid_of_the_right_size(results: dict[str, AdapterResult], key: str) -> None:
    result = results[key]
    mesh = result.mesh
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert result.volume_mm3 == pytest.approx(mesh.volume)
    assert result.outer_diameter_mm == EXPECTED_OD[key]
    lo, hi = mesh.bounds
    assert hi[0] - lo[0] == pytest.approx(EXPECTED_OD[key], abs=1e-6)
    assert hi[1] - lo[1] == pytest.approx(EXPECTED_OD[key], abs=1e-6)
    t = result.spec.thickness_mm
    assert lo[2] == pytest.approx(-2.0)  # the spigot
    assert hi[2] == pytest.approx(t)


@pytest.mark.parametrize("key", list(EXPECTED_OD))
def test_robot_holes_go_through_and_the_dowel_is_blind(results: dict[str, AdapterResult], key: str) -> None:
    result = results[key]
    t = result.spec.thickness_mm
    bolts = features(result, "robot_bolt_")
    assert len(bolts) == 4
    for f in bolts:
        assert z_hits(result.mesh, f.x_mm, f.y_mm) == [], f"material in {f.name}"
    clearance = bolts[0].diameter_mm
    for f in features(result, "robot_counterbore_"):
        # Between the clearance hole and the counterbore wall: plate under the screw head only.
        r = (f.diameter_mm + clearance) / 4.0
        hits = z_hits(result.mesh, f.x_mm + r, f.y_mm)
        assert hits == [pytest.approx(0.0), pytest.approx(result.bolt.clamp_mm)]
    (dowel,) = features(result, "robot_dowel")
    assert z_hits(result.mesh, dowel.x_mm, dowel.y_mm) == [pytest.approx(dowel.z_max_mm), pytest.approx(t)]
    assert 0.0 < dowel.z_max_mm < t


def test_adapter_tool_side_is_blind_and_clear_of_the_robot_holes(results: dict[str, AdapterResult]) -> None:
    result = results[ADAPTER_KEY]
    t = result.spec.thickness_mm
    tool = features(result, "tool_hole_") + features(result, "tool_dowel")
    assert len(tool) == 5
    robot = [f for f in result.features if f.side == "robot" and f.kind != "spigot"]
    for f in tool:
        hits = z_hits(result.mesh, f.x_mm, f.y_mm)
        assert len(hits) == 2, f"{f.name} hits {hits}"
        assert hits[0] == pytest.approx(0.0)
        assert hits[1] == pytest.approx(f.z_min_mm)
        assert hits[1] < t - 1.0  # blind: material below, open above
        assert min(wall(f, r) for r in robot) >= 2.0 - 1e-9
    holes = features(result, "tool_hole_")
    assert all(math.hypot(f.x_mm, f.y_mm) == pytest.approx(31.5 / 2.0) for f in holes)
    # Centring recess: open above, plate and spigot below.
    assert z_hits(result.mesh, 3.0, 4.0) == [pytest.approx(-2.0), pytest.approx(t - 2.5)]


@pytest.mark.parametrize("key", ["plate-iso50-4-m6", "plate-iso31p5-4-m5"])
def test_blank_plates_carry_a_blind_m3_grid(results: dict[str, AdapterResult], key: str) -> None:
    result = results[key]
    t = result.spec.thickness_mm
    grid = features(result, "grid_")
    assert len(grid) >= 8
    robot = [f for f in result.features if f.side == "robot" and f.kind != "spigot"]
    radius = result.outer_diameter_mm / 2.0
    for f in grid:
        hits = z_hits(result.mesh, f.x_mm, f.y_mm)
        assert len(hits) == 2 and hits[1] == pytest.approx(f.z_min_mm) and hits[1] < t - 1.0
        assert min(wall(f, r) for r in robot) >= 2.0 - 1e-9
        assert math.hypot(f.x_mm, f.y_mm) + f.diameter_mm / 2.0 <= radius - 2.0 + 1e-9
    assert any("M3 pilot holes" in n for n in result.notes)
    assert "tool_grid_layout" in result.unverified


@pytest.mark.parametrize("key", list(EXPECTED_OD))
def test_provenance_is_reported(results: dict[str, AdapterResult], key: str) -> None:
    result = results[key]
    statuses = {d["status"] for d in result.dimensions.values()}
    assert statuses <= {"sourced", "secondary", "assumed"}
    assert list(result.unverified) == sorted(result.unverified)
    assert set(result.unverified) == {n for n, d in result.dimensions.items() if d["status"] != "sourced"}
    assert "robot_d2_mm" in result.unverified
    assert result.dimensions["robot_d2_mm"]["status"] == "secondary"
    assert "spigot_height_mm" in result.unverified
    assert result.dimensions["spigot_height_mm"] == {
        "value": 2.0,
        "status": "assumed",
        "source": result.dimensions["spigot_height_mm"]["source"],
    }
    assert "robot_d1_mm" not in result.unverified  # d1 is sourced for both confirmed patterns
    for name in (
        "robot_clearance_hole_mm",
        "robot_counterbore_mm",
        "print_allowance_mm",
        "robot_hole_layout",
    ):
        assert name in result.unverified
    assert any("ISO 9409-1:2004" in n for n in result.notes)
    json.dumps(result.dimensions)  # JSON-ready


def test_adapter_reports_both_patterns(results: dict[str, AdapterResult]) -> None:
    result = results[ADAPTER_KEY]
    for name in ("tool_d2_mm", "tool_recess_depth_mm", "tool_tap_drill_mm", "tool_hole_layout"):
        assert name in result.unverified
    assert result.dimensions["tool_d1_mm"]["value"] == 31.5
    assert result.dimensions["tool_dowel_mm"]["status"] == "assumed"


@pytest.mark.parametrize("key", list(EXPECTED_OD))
def test_recommended_robot_bolt_keeps_the_thread_depth_rule(
    results: dict[str, AdapterResult], key: str
) -> None:
    result = results[key]
    bolt = result.bolt
    d = {"M5": 5.0, "M6": 6.0}[bolt.bolt.split("x")[0]]
    assert bolt.ok and bolt.reasons == ()
    assert bolt.clamp_mm == pytest.approx(result.spec.thickness_mm - (d + 0.5))
    assert bolt.engagement_mm == pytest.approx(bolt.length_mm - bolt.clamp_mm)
    assert bolt.engagement_mm <= MAX_THREAD_DEPTH_MM == 8.0
    assert bolt.engagement_mm >= bolt.min_engagement_mm
    assert bolt.min_engagement_mm >= 1.0 * d


def test_bolt_rule_choice_is_visible(results: dict[str, AdapterResult]) -> None:
    m5 = results["plate-iso31p5-4-m5"].bolt
    assert (m5.bolt, m5.engagement_mm, m5.min_engagement_mm) == ("M5x12", 7.5, 7.5)  # meets 1.5 x d
    # An M6 cannot meet 1.5 x d (9 mm) inside 8 mm, so the generator says it used 1.0 x d.
    with pytest.raises(ValueError, match="9.0 and 8.0 mm"):
        recommend_bolt(ISO50, 3.5)
    m6 = results["plate-iso50-4-m6"]
    assert m6.bolt.bolt == "M6x10" and m6.bolt.min_engagement_mm == 6.0
    assert m6.dimensions["min_engagement_factor"]["value"] == 1.0
    assert any("1.0 x d" in n for n in m6.notes)


def test_bolt_check_explains_failures() -> None:
    long = bolt_check(ISO50, clamp_mm=3.5, bolt_length_mm=16)
    assert not long.ok
    assert long.engagement_mm == 12.5
    assert any("reaches 12.5 mm" in r and "8.0 mm" in r for r in long.reasons)
    short = bolt_check(ISO31, clamp_mm=4.5, bolt_length_mm=8)
    assert not short.ok
    assert any("only 3.5 mm" in r and "7.5 mm" in r for r in short.reasons)
    assert bolt_check(ISO31, clamp_mm=4.5, bolt_length_mm=12).ok
    with pytest.raises(ValueError):
        bolt_check("ISO 9409-1-99-4-M6", 3.5, 10)


def test_write_adapter_is_byte_identical(tmp_path) -> None:
    spec = CONFIRMED_ADAPTERS[2]
    first = write_adapter(spec, tmp_path / "a")
    second = write_adapter(spec, tmp_path / "b")
    assert set(first) == {"stl", "xml", "json"}
    for kind in first:
        assert first[kind].name == f"{spec.key}.{kind}"
        assert first[kind].read_bytes() == second[kind].read_bytes(), kind
    stl = first["stl"].read_bytes()
    assert stl.startswith(b"rtt.library.adapters") and not stl.startswith(b"solid")
    loaded = trimesh.load(first["stl"], file_type="stl")
    assert loaded.is_watertight
    text = first["json"].read_text()
    record = json.loads(text)
    assert text == json.dumps(record, sort_keys=True, indent=2) + "\n"
    assert record["licence"] == "Apache-2.0"
    assert record["generator"] == "rtt.library.adapters"
    assert record["claims_iso_conformance"] is False
    assert record["spec"]["robot"] == ISO50 and record["spec"]["tool"] == ISO31
    assert record["bolt"]["bolt"] == "M6x10"
    assert "robot_d2_mm" in record["unverified"]


@pytest.mark.parametrize("spec", CONFIRMED_ADAPTERS, ids=lambda s: s.key)
def test_mjcf_compiles_with_the_computed_mass(tmp_path, spec: AdapterSpec) -> None:
    paths = write_adapter(spec, tmp_path)
    record = json.loads(paths["json"].read_text())
    model = mujoco.MjModel.from_xml_path(str(paths["xml"]))
    mass = model.body_mass[model.body(spec.key).id]
    expected = generate(spec).mass_kg
    assert mass == pytest.approx(expected, rel=1e-6)
    assert record["mass_kg"] == pytest.approx(expected, abs=1e-6)
    assert 0.005 < mass < 0.5
    assert np.allclose(model.site("mount_robot").pos, 0.0)
    assert np.allclose(model.site("mount_tool").pos, [0.0, 0.0, spec.thickness_mm / 1000.0])


def test_aluminium_and_infill_change_the_mass(results: dict[str, AdapterResult]) -> None:
    pla = results["plate-iso31p5-4-m5"]
    alu = generate(AdapterSpec(robot=ISO31, material="aluminium", print_allowance_mm=0.0))
    half = generate(AdapterSpec(robot=ISO31, infill=0.5))
    assert half.mass_kg == pytest.approx(pla.mass_kg / 2.0)
    assert alu.mass_kg > 2.0 * pla.mass_kg
    assert alu.dimensions["density_g_cm3"]["value"] == 2.70


def test_infeasible_layouts_raise() -> None:
    # Robot screws on a 31.5 mm circle would land inside the 50 mm tool centring recess.
    with pytest.raises(ValueError, match="recess"):
        generate(AdapterSpec(robot=ISO31, tool="ISO 9409-1-80-6-M8", thickness_mm=8.0))
    # Same pattern on both sides: no rotation keeps 2 mm between the two hole sets.
    with pytest.raises(ValueError, match="no rotation .* 2 mm of wall"):
        generate(AdapterSpec(robot=ISO50, tool=ISO50))
    # An M8 counterbore leaves too little plate under the head.
    with pytest.raises(ValueError, match="under the M8 screw heads"):
        generate(AdapterSpec(robot="ISO 9409-1-80-6-M8"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"robot": "ISO 9409-1-63-4-M6"},
        {"robot": ISO50, "tool": "ISO 9409-1-99-4-M6"},
        {"robot": ISO50, "thickness_mm": 5.9},
        {"robot": ISO50, "thickness_mm": 30.5},
        {"robot": ISO50, "infill": 0.0},
        {"robot": ISO50, "infill": 1.2},
        {"robot": ISO50, "material": "steel"},
        {"robot": ISO50, "key": "Bad Key"},
    ],
)
def test_invalid_specs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AdapterSpec(**kwargs)
