"""Procedural parts library (rtt.parts.pieces, rtt.parts.library): exact convex pieces.

Covers track O's share of gate GV0: every piece is convex and closed, the union is one
watertight solid, mass properties are exact and reproducible across processes, the six
training parts have the specified dimensions, and each rests stably in its declared pose.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import trimesh
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.spatial import ConvexHull

from rtt.parts import library as lib
from rtt.parts import pieces as pc
from rtt.parts.materials import MATERIALS, material
from rtt.parts.types import PartSpec

ALL_FAMILIES = lib.TRAINING_FAMILIES + lib.HELDOUT_FAMILIES


@pytest.fixture(scope="module")
def parts() -> dict[str, PartSpec]:
    out = {p.name: p for p in lib.training_parts()}
    out["pipe_elbow"] = lib.build_part("pipe_elbow", "pipe_elbow")
    out["spur_gear"] = lib.build_part("spur_gear", "spur_gear")
    return out


# --------------------------------------------------------------------------- pieces


def test_box_mass_properties_are_exact() -> None:
    b = pc.box((0.0, 0.0, 0.0), (0.03, 0.04, 0.06))
    mp = pc.mass_properties([b])
    v = 0.03 * 0.04 * 0.06
    assert mp.volume == pytest.approx(v, rel=1e-12)
    assert mp.com == pytest.approx((0.015, 0.02, 0.03), abs=1e-12)
    assert mp.unit_inertia[:3] == pytest.approx(
        (v * (0.04**2 + 0.06**2) / 12, v * (0.03**2 + 0.06**2) / 12, v * (0.03**2 + 0.04**2) / 12), rel=1e-10
    )
    assert mp.unit_inertia[3:] == (0.0, 0.0, 0.0)


def test_overlaps_are_removed_exactly() -> None:
    a = pc.box((0, 0, 0), (2, 1, 1))
    b = pc.box((1, 0, 0), (3, 1, 1))
    touching = pc.box((2, 0, 0), (3, 1, 1))
    assert pc.overlaps([a, b]) == [(0, 1, pytest.approx(1.0))]
    assert pc.overlaps([a, touching]) == []
    mp = pc.mass_properties([a, b])
    assert mp.volume == pytest.approx(3.0)
    assert mp.com == pytest.approx((1.5, 0.5, 0.5))
    # perpendicular 16-gon prisms (the tee's cores): against the boolean union
    px = pc.prism(pc.ngon(0.0135, 16), "x", -0.0135, 0.0135)
    py = pc.prism(pc.ngon(0.0135, 16), "y", 0.0, 0.0135)
    mp = pc.mass_properties([px, py])
    union = pc.union_mesh([px, py])
    assert mp.volume == pytest.approx(union.volume, rel=1e-6)
    assert mp.com == pytest.approx(tuple(union.center_mass), abs=1e-8)
    inertia = union.moment_inertia
    assert mp.unit_inertia[:3] == pytest.approx(tuple(np.diag(inertia)), rel=1e-5)


def test_triple_overlaps_are_refused() -> None:
    boxes = [pc.box((0, 0, 0), (2, 2, 2)), pc.box((1, 0, 0), (3, 2, 2)), pc.box((0.5, 0, 0), (2.5, 2, 2))]
    with pytest.raises(ValueError, match="triple"):
        pc.mass_properties(boxes)


def test_ngon_flats_sit_at_the_apothem() -> None:
    poly = pc.ngon(0.01, 16)
    assert np.max(poly[:, 0]) == pytest.approx(0.01)  # a flat faces +u
    assert np.max(poly[:, 1]) == pytest.approx(0.01)  # and +v
    assert np.min(np.hypot(*poly.T)) == pytest.approx(0.01 / math.cos(math.pi / 16))
    hexagon = pc.ngon(0.0085, 6, flat_at_deg=90.0)
    assert np.max(hexagon[:, 1]) == pytest.approx(0.0085)  # flats face +-y
    assert np.max(hexagon[:, 0]) == pytest.approx(0.0085 / math.cos(math.pi / 6))  # corners at +-x


def test_ring_wedges_tile_the_wall() -> None:
    wedges = pc.ring_wedges(0.006, 0.010, 16, "z", 0.0, 0.03)
    assert pc.overlaps(wedges) == []
    expected = 16 * (0.010**2 - 0.006**2) * math.tan(math.pi / 16) * 0.03
    assert pc.mass_properties(wedges).volume == pytest.approx(expected, rel=1e-4)  # vertices are rounded


def test_prisms_refuse_degenerate_input() -> None:
    with pytest.raises(ValueError):
        pc.prism([(0, 0), (1, 0), (2, 0)], "z", 0.0, 1.0)
    with pytest.raises(ValueError):
        pc.prism(pc.ngon(0.01, 8), "z", 0.0, 0.0)
    with pytest.raises(ValueError):
        pc.prism(pc.ngon(0.01, 8), "w", 0.0, 1.0)  # type: ignore[arg-type]


def test_quantize_normalises_negative_zero() -> None:
    q = pc.quantize([-0.0, -1e-12, 0.12345678])
    assert q.tobytes() == np.array([0.0, 0.0, 0.1234568]).tobytes()


# --------------------------------------------------------------------------- every family


@pytest.mark.parametrize(
    "name",
    [
        "pipe_tee",
        "block",
        "hex_standoff",
        "flanged_bushing",
        "l_bracket",
        "steel_slug",
        "pipe_elbow",
        "spur_gear",
    ],
)
def test_every_piece_is_an_exact_closed_convex_prism(parts: dict[str, PartSpec], name: str) -> None:
    part = parts[name]
    for verts, faces in zip(part.pieces, part.faces, strict=True):
        mesh = trimesh.Trimesh(verts, faces, process=False)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert mesh.volume > 0  # outward faces
        hull = ConvexHull(verts)
        assert len(hull.vertices) == len(verts)  # every vertex is a hull vertex
        assert abs(hull.volume - mesh.volume) <= 1e-9 * hull.volume  # GV0: hull volume = piece volume
        assert np.array_equal(verts, np.round(verts, pc.QUANTUM_DECIMALS))


@pytest.mark.parametrize(
    "name",
    [
        "pipe_tee",
        "block",
        "hex_standoff",
        "flanged_bushing",
        "l_bracket",
        "steel_slug",
        "pipe_elbow",
        "spur_gear",
    ],
)
def test_union_is_one_watertight_solid_with_the_stated_mass(parts: dict[str, PartSpec], name: str) -> None:
    part = parts[name]
    union = pc.union_manifold(list(part.pieces), list(part.faces))
    assert len(union.decompose()) == 1  # one connected solid
    mesh = lib.visual_mesh(part)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert part.volume == pytest.approx(mesh.volume, rel=2e-6)
    assert part.com == pytest.approx(tuple(mesh.center_mass), abs=2e-7)
    inertia = mesh.moment_inertia
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    scale = max(full[:3])
    assert np.allclose(part.unit_inertia, full, rtol=0, atol=1e-5 * scale)
    lo, hi = np.concatenate(part.pieces).min(axis=0), np.concatenate(part.pieces).max(axis=0)
    assert lo[2] == pytest.approx(-hi[2], abs=2e-7)  # z: bounding box centred on the origin
    if name == "l_bracket":  # x/y: centred on the standing leg, which the jaws grip
        leg = part.pieces[1]
        assert np.allclose(leg[:, :2].min(axis=0), -leg[:, :2].max(axis=0), atol=2e-7)
    else:
        assert np.allclose(lo, -hi, atol=2e-7)  # bounding box centred on the origin
    assert len(part.pieces) <= 128
    assert part.decomposition == "exact"


def test_training_parts_have_the_specified_dimensions_and_roles(parts: dict[str, PartSpec]) -> None:
    def size_mm(name: str) -> list[float]:
        return [round(1000 * s, 3) for s in parts[name].size]

    assert size_mm("pipe_tee") == [60.0, 44.25, 28.5]  # run 60, branch end to run back, hub OD
    assert size_mm("block") == [30.0, 40.0, 60.0]  # 40 mm across the jaws, no chamfer
    assert size_mm("hex_standoff")[1:] == [17.0, 45.0]  # 17 across flats, on the closing axis
    assert size_mm("hex_standoff")[0] == pytest.approx(17.0 / math.cos(math.pi / 6), abs=1e-3)
    assert size_mm("flanged_bushing") == [36.0, 36.0, 35.0]  # 20 x 30 body, 36 x 5 flange on top
    assert size_mm("l_bracket") == [40.0, 30.0, 40.0]  # lying flat, standing leg's edges toward the jaws
    assert size_mm("steel_slug") == [40.0, 40.0, 50.0]
    mass_g = {n: 1000 * p.mass for n, p in parts.items()}
    assert mass_g["steel_slug"] == pytest.approx(490, rel=0.015)
    assert mass_g["block"] == pytest.approx(194.4, abs=0.1)
    assert all(m < 917 for m in mass_g.values())  # under the grip cap
    assert parts["hex_standoff"].yaw_symmetry == 6
    assert parts["block"].yaw_symmetry == 2
    for round_part in ("steel_slug", "flanged_bushing"):
        assert parts[round_part].yaw_symmetry == 0 and "yaw" not in parts[round_part].register_axes
    assert parts["pipe_tee"].register_axes == ("x", "y", "yaw", "tilt")
    materials = {n: p.material.name for n, p in parts.items()}
    assert materials["pipe_tee"] == "pvc_white" and materials["flanged_bushing"] == "brass"
    assert materials["block"] == "anodised_aluminium" and materials["steel_slug"] == "ground_steel"


def test_tee_branch_socket_faces_the_left_jaw(parts: dict[str, PartSpec]) -> None:
    tee = parts["pipe_tee"]
    branch = {f.name: f for f in tee.features}["branch_socket"]
    assert branch.facing() == "left"  # +y
    assert branch.origin[1] == pytest.approx(tee.size[1] / 2)  # the mouth is the +y extreme
    assert {f.facing() for f in tee.features if f.name != "branch_socket"} == {None}
    assert branch.radius == pytest.approx(0.01075) and branch.depth == pytest.approx(0.0165)


def test_l_bracket_is_gripped_on_its_standing_leg(parts: dict[str, PartSpec]) -> None:
    part = parts["l_bracket"]
    pts = np.concatenate(part.pieces)
    high = pts[pts[:, 2] > -0.016]  # above the 4 mm leg on the table only the standing leg remains
    # on the grip axis, with its edges toward the jaws
    assert np.allclose(high[:, 0].min(), -0.002) and np.allclose(high[:, 0].max(), 0.002)
    assert np.allclose(high[:, 1].min(), -0.015) and np.allclose(high[:, 1].max(), 0.015)
    assert pts[:, 0].min() == pytest.approx(-0.038)  # the leg on the table reaches out to -x
    assert part.com[0] == pytest.approx(-0.0095, abs=2e-4)  # 9.5 mm off the grip axis


def test_part_specs_match_their_mesh_objects(parts: dict[str, PartSpec]) -> None:
    for part in parts.values():
        obj = part.to_mesh_object()
        assert obj.geometry_sha256 == part.geometry_sha256
        assert obj.density == part.material.density
        back = lib.part_from_mesh_object(obj, family=part.family, mat=part.material)
        assert back.geometry_sha256 == part.geometry_sha256


@pytest.mark.parametrize(
    "name", ["pipe_tee", "block", "hex_standoff", "flanged_bushing", "l_bracket", "steel_slug"]
)
def test_training_parts_rest_stably_in_the_scene(parts: dict[str, PartSpec], name: str) -> None:
    from rtt.sim.gso import rest_test

    check = rest_test(parts[name].to_mesh_object())
    assert check.passed, check.describe()


# --------------------------------------------------------------------------- determinism and digests


_DIGEST_CODE = (
    "import json; from rtt.parts import library as L; "
    "ps = list(L.training_parts()) + [L.build_part('e', 'pipe_elbow'), L.build_part('g', 'spur_gear')]; "
    "print(json.dumps([L.library_digest()] + [p.geometry_sha256 for p in ps]))"
)


def test_hashes_are_identical_across_processes(parts: dict[str, PartSpec]) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), PYTHONHASHSEED="random")
    runs = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", _DIGEST_CODE], capture_output=True, text=True, env=env, check=True
            ).stdout
        )
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
    here = [lib.library_digest()] + [
        parts[n].geometry_sha256
        for n in (
            "pipe_tee",
            "block",
            "hex_standoff",
            "flanged_bushing",
            "l_bracket",
            "steel_slug",
            "pipe_elbow",
            "spur_gear",
        )
    ]
    assert runs[0] == here


#: The training parts as built on 2026-10-16. A change here changes the training suite: update
#: these only together with a recorded reason (before the freeze) or a deviation (after it).
#: They also pin cross-machine parity (GV4): the pieces are built analytically, so the same
#: values must come out on any IEEE-754 machine.
EXPECTED_GEOMETRY_SHA256 = {
    "pipe_tee": "98740094d82da2f86be41147051213c143fc57b37f34729654ab186a8596c7a3",
    "block": "e8ea8c7bb6df0bc934c8e77fda68923fa6fb1f7d851b69e426d854b00a314a8c",
    "hex_standoff": "c0692629573c666b8d5754016fb7d6d473aa72c44e6080819abce607054ec380",
    "flanged_bushing": "1a1523da52fbefe9b43bed3d1a7407ab8c16dfa884d4c71ede23cacbf07c0042",
    "l_bracket": "04568f49bb463d38b8d224e3cff4295ab5bc6be123ae7f11b90c43564cf0668b",
    "steel_slug": "3ce55fded0d427e50055d04951aa8c235771a4bb26154ae71be4ea598ad647b6",
}


def test_training_geometry_is_frozen(parts: dict[str, PartSpec]) -> None:
    assert {n: parts[n].geometry_sha256 for n in EXPECTED_GEOMETRY_SHA256} == EXPECTED_GEOMETRY_SHA256


def test_library_digest_covers_dimensions_and_materials(tmp_path) -> None:
    base = lib.library_digest()
    raw = (lib.DEFAULT_PARTS_FILE).read_text()
    changed = tmp_path / "parts.yaml"
    changed.write_text(raw.replace("x: 30.0, y: 40.0, z: 60.0", "x: 30.0, y: 40.0, z: 61.0"))
    assert lib.library_digest(changed) != base
    changed.write_text(raw.replace("material: brass", "material: zinc_plated_steel"))
    assert lib.library_digest(changed) != base


# --------------------------------------------------------------------------- config and validation


def test_library_file_lists_the_six_training_parts() -> None:
    library = lib.load_parts_library()
    assert library.names() == (
        "pipe_tee",
        "block",
        "hex_standoff",
        "flanged_bushing",
        "l_bracket",
        "steel_slug",
    )
    assert {e.family for e in library.entries} == set(lib.TRAINING_FAMILIES)
    table = lib.part_table()
    assert [r["name"] for r in table] == list(library.names())
    assert table[0]["features"][0] == {
        "kind": "socket",
        "name": "branch_socket",
        "faces_jaw": "left",
        "radius_mm": 10.75,
        "depth_mm": 16.5,
    }
    text = json.dumps(table)
    for unseen in lib.HELDOUT_FAMILIES:  # held-out families never reach a prompt table
        assert unseen not in text


@pytest.mark.parametrize(
    "item, message",
    [
        ({"name": "x", "family": "nope"}, "unknown family"),
        ({"name": "x", "family": "block", "params": {"w": 1}}, "unknown parameters"),
        ({"name": "x", "family": "block", "params": {"x": 1.0}}, "at least 2 mm"),
        ({"name": "x", "family": "pipe_tee", "params": {"socket_id": 26.0}}, "socket_id"),
        ({"name": "x", "family": "pipe_tee", "params": {"segments": 10}}, "multiple of 4"),
        ({"name": "x", "family": "pipe_tee", "params": {"centre_to_end": 25.0}}, "socket floors"),
        ({"name": "x", "family": "flanged_bushing", "params": {"flange_od": 22.0}}, "flange_od"),
        ({"name": "x", "family": "hex_standoff", "params": {"hole_d": 14.0}}, "hole_d"),
        ({"name": "x", "family": "l_bracket", "params": {"wall": 1.0}}, "wall"),
        ({"name": "x", "family": "slug", "params": {"segments": 2.5}}, "integer"),
        ({"name": "x", "family": "block", "params": {"x": True}}, "finite number"),
        ({"name": "x", "family": "block", "material": "unobtainium"}, "unknown material"),
        ({"name": "x", "family": "block", "colour": "red"}, "unknown keys"),
    ],
)
def test_library_file_rejects_bad_entries(item: dict, message: str) -> None:
    with pytest.raises((ValueError, KeyError, TypeError), match=message):
        lib.parts_library_from_dict({"parts": [item]})


def test_library_file_rejects_duplicate_names() -> None:
    item = {"name": "a", "family": "block"}
    with pytest.raises(ValueError, match="unique"):
        lib.parts_library_from_dict({"parts": [item, item]})


def test_materials_are_named_and_physical() -> None:
    assert material("brass").density == 8500.0
    assert all(1000.0 <= m.density <= 9000.0 for m in MATERIALS.values())
    with pytest.raises(KeyError, match="known"):
        material("wood")


def _family_params(family: str) -> st.SearchStrategy[dict]:
    fam = lib.FAMILIES[family]
    lengths = {
        k: st.floats(0.6 * v, 1.4 * v, allow_nan=False)
        for k, v in fam.defaults.items()
        if k not in fam.integers and k not in fam.angles
    }
    return st.fixed_dictionaries(lengths)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), family=st.sampled_from(ALL_FAMILIES))
def test_random_dimensions_build_exactly_or_fail_with_a_message(data: st.DataObject, family: str) -> None:
    params = data.draw(_family_params(family))
    try:
        part = lib.build_part("h", family, params)
    except ValueError as exc:
        assert "invalid dimensions" in str(exc) or "convex" in str(exc)
        return
    pieces = lib.part_pieces(part)
    for p in pieces:
        hull = ConvexHull(p.verts)
        assert abs(hull.volume - pc.piece_volume(p)) <= 1e-9 * hull.volume
    assert part.volume > 0 and math.isfinite(part.mass)
    lo, hi = np.concatenate(part.pieces).min(axis=0), np.concatenate(part.pieces).max(axis=0)
    assert lo[2] == pytest.approx(-hi[2], abs=2e-7)
    if family != "l_bracket":  # the L-bracket's x/y origin is its standing leg
        assert np.allclose(lo, -hi, atol=2e-7)


def test_mirror_y_is_exact_and_flips_the_jaw_sides(parts: dict[str, PartSpec]) -> None:
    from rtt.parts.features import check_features

    tee = parts["pipe_tee"]
    m = lib.mirror_y(tee)
    assert m.name == "pipe_tee.mirror_y"
    assert {f.name: f.facing() for f in m.features}["branch_socket"] == "right"
    assert m.com == pytest.approx((tee.com[0], -tee.com[1], tee.com[2]))
    for verts, faces in zip(m.pieces, m.faces, strict=True):
        mesh = trimesh.Trimesh(verts, faces, process=False)
        assert mesh.volume > 0  # winding reversed with the mirror: faces stay outward
    union = lib.visual_mesh(m)
    assert union.volume == pytest.approx(tee.volume, rel=2e-6)
    assert union.center_mass == pytest.approx(m.com, abs=2e-7)
    assert check_features(m) == []
    assert lib.mirror_y(m).geometry_sha256 == tee.geometry_sha256
    bracket = parts["l_bracket"]
    mirrored_com = (bracket.com[0], -bracket.com[1], bracket.com[2])  # symmetric in y
    assert np.allclose(lib.mirror_y(bracket).com, mirrored_com)
    assert lib.mirror_y(parts["block"]).size == parts["block"].size


def test_package_exports_resolve() -> None:
    """Every name in ``rtt.parts.__all__`` resolves (the lazy ones on first use), including
    the helpers other tracks are told to import from the package (``mirror_y`` for the
    right jaw)."""
    import rtt.parts as parts_pkg

    for name in parts_pkg.__all__:
        assert getattr(parts_pkg, name) is not None, name
    assert parts_pkg.mirror_y is lib.mirror_y
    assert parts_pkg.part_from_mesh_object is lib.part_from_mesh_object
    with pytest.raises(AttributeError):
        _ = parts_pkg.no_such_name
