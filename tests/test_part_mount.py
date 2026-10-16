"""Printable jaw: carriage block with the documented rod bores, fused with a generated finger."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from manifold3d import Manifold

from rtt.design.export import to_trimesh
from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.part.mount import (
    DIMENSION_SOURCES,
    ClampInterface,
    build_jaw,
    jaw_print_orientation,
    manifold_from_trimesh,
    rack_profile,
)
from rtt.search.runner import DEFAULT_ENTRANTS

ENTRANTS = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}


def _solid_at(mesh: trimesh.Trimesh, points: np.ndarray, probe_mm: float = 0.2) -> list[bool]:
    """Whether each point lies inside the solid (a tiny cube there overlaps it); no rtree needed."""
    solid = manifold_from_trimesh(mesh)
    out = []
    for p in np.atleast_2d(points):
        cube = Manifold.cube((probe_mm, probe_mm, probe_mm), center=True).translate(
            tuple(float(v) for v in p)
        )
        out.append((solid ^ cube).volume() > 0.5 * probe_mm**3)
    return out


def _vertical_line_hits(mesh: trimesh.Trimesh, x: float, y: float, d_mm: float = 0.4) -> float:
    """Volume of the solid inside a thin vertical rod through (x, y)."""
    lo, hi = mesh.bounds
    rod = Manifold.cylinder(float(hi[2] - lo[2]) + 2.0, d_mm / 2.0, circular_segments=12).translate(
        (x, y, float(lo[2]) - 1.0)
    )
    return (manifold_from_trimesh(mesh) ^ rod).volume()


@pytest.fixture(scope="module", params=sorted(ENTRANTS))
def jaw(request: pytest.FixtureRequest) -> tuple[str, trimesh.Trimesh]:
    return request.param, build_jaw(generate_finger(ENTRANTS[request.param]))


def test_jaw_is_one_watertight_solid(jaw: tuple[str, trimesh.Trimesh]) -> None:
    name, mesh = jaw
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, name
    assert len(mesh.split(only_watertight=False)) == 1
    finger_mm3 = to_trimesh(generate_finger(ENTRANTS[name])).volume * 1e9
    block_mm3 = 42.0 * 16.0 * 18.0
    bores_mm3 = 2 * np.pi * 3.05**2 * 16.0
    assert mesh.volume == pytest.approx(block_mm3 - bores_mm3 + finger_mm3, rel=0.02)


def test_rod_bores_are_open_and_at_the_documented_spacing(jaw: tuple[str, trimesh.Trimesh]) -> None:
    _, mesh = jaw
    iface = ClampInterface()
    s, z = iface.rod_spacing_mm / 2.0, iface.rod_axis_height_mm
    probes = np.array(
        [
            [-s, 8.0, z],  # bore centres: empty
            [s, 8.0, z],
            [-s + 2.9, 8.0, z],  # just inside the 6.1 mm bore
            [0.0, 8.0, z],  # block between the bores: solid
            [-s + 3.3, 8.0, z],  # just outside the bore wall: solid
            [0.0, 8.0, iface.block_height_mm - 0.5],
        ]
    )
    assert _solid_at(mesh, probes) == [False, False, False, True, True, True]
    lo, hi = mesh.bounds
    assert lo[0] == pytest.approx(-21.0, abs=1e-3) and hi[0] == pytest.approx(21.0, abs=1e-3)
    assert hi[2] == pytest.approx(iface.block_height_mm, abs=1e-3)


def test_finger_is_symmetric_so_one_jaw_serves_both_sides() -> None:
    for design in ENTRANTS.values():
        finger = to_trimesh(generate_finger(design))
        mirrored = finger.copy()
        mirrored.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
        assert np.allclose(finger.bounds, mirrored.bounds, atol=1e-7)
        assert mirrored.volume == pytest.approx(finger.volume, rel=1e-9)
        a, b = manifold_from_trimesh(finger), manifold_from_trimesh(mirrored)
        assert (a - b).volume() < 1e-6 * a.volume()  # (metres^3) mirror image fills the same space


def test_print_orientation_puts_the_contact_face_on_the_bed() -> None:
    straight = build_jaw(generate_finger(ENTRANTS["straight"]))
    printed, support = jaw_print_orientation(straight)
    assert printed.bounds[0][2] == pytest.approx(0.0, abs=1e-9)
    assert not support
    assert printed.extents[2] == pytest.approx(16.0, abs=0.05)  # the block depth is now the height
    # the bores now run vertically: a thin vertical rod through a bore centre misses the solid
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    turned = straight.vertices @ rot.T
    lo, hi = turned.min(axis=0), turned.max(axis=0)
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    iface = ClampInterface()
    for sx in (-iface.rod_spacing_mm / 2.0, iface.rod_spacing_mm / 2.0):
        centre = rot @ np.array([sx, 8.0, iface.rod_axis_height_mm]) + shift
        assert _vertical_line_hits(printed, centre[0], centre[1]) == pytest.approx(0.0, abs=1e-6)
        beside = centre[0] - np.sign(sx) * iface.rod_bore_d_mm  # in the wall between bore and centre
        assert _vertical_line_hits(printed, beside, centre[1]) > 1.0
    _hooked, support = jaw_print_orientation(build_jaw(generate_finger(ENTRANTS["hooked"])))
    assert support  # the hook reaches past the contact face


def test_rack_option_adds_a_drive_and_stays_watertight() -> None:
    iface = ClampInterface(rack=True)
    mesh = build_jaw(generate_finger(ENTRANTS["stock"]), iface)
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
    assert mesh.bounds[0][1] == pytest.approx(-iface.rack_length_mm, abs=1e-3)
    profile = rack_profile(iface)
    pitch = np.pi * iface.rack_module_mm
    n_teeth = int(iface.rack_length_mm // pitch)
    assert profile.area() == pytest.approx(
        iface.rack_bar_mm * (iface.rack_length_mm + iface.block_depth_mm / 2.0)
        + n_teeth * 2.25 * iface.rack_module_mm * pitch / 2.0,
        rel=0.05,
    )


def test_interface_validation_and_sources() -> None:
    with pytest.raises(ValueError, match="inside the block width"):
        ClampInterface(rod_spacing_mm=40.0)
    with pytest.raises(ValueError, match="positive"):
        ClampInterface(rod_bore_d_mm=-1.0)
    with pytest.raises(ValueError, match="rod axis"):
        ClampInterface(rod_axis_height_mm=17.0)
    with pytest.raises(ValueError, match="stroke"):
        ClampInterface(rack=True, rack_length_mm=40.0)
    desc = ClampInterface().describe()
    assert desc["rod_bore_d_mm"] == {"value": 6.1, "source": "documented"}
    assert desc["rod_spacing_mm"]["value"] == 32.0 and desc["rod_spacing_mm"]["source"] == "documented"
    assert desc["block_height_mm"]["source"] == "assumed"
    assert set(DIMENSION_SOURCES) == set(desc)
    assert ClampInterface().pitch_radius_mm == pytest.approx(7.2)


def test_a_finger_wider_than_the_block_is_refused() -> None:
    with pytest.raises(ValueError, match="wider than the carriage"):
        build_jaw(
            generate_finger(ENTRANTS["stock"]), ClampInterface(block_width_mm=20.0, rod_spacing_mm=10.0)
        )
