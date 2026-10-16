"""Part features (rtt.parts.features): declared sockets and flanges match the pieces."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rtt.parts import features as F
from rtt.parts import library as lib
from rtt.parts import pieces as pc
from rtt.parts.types import Feature, PartSpec


@pytest.fixture(scope="module")
def parts() -> dict[str, PartSpec]:
    out = {p.name: p for p in lib.training_parts()}
    out["pipe_elbow"] = lib.build_part("pipe_elbow", "pipe_elbow")
    out["spur_gear"] = lib.build_part("spur_gear", "spur_gear")
    return out


@pytest.mark.parametrize("name", ["pipe_tee", "hex_standoff", "flanged_bushing", "pipe_elbow", "spur_gear"])
def test_declared_features_match_the_geometry(parts: dict[str, PartSpec], name: str) -> None:
    assert parts[name].features
    assert F.check_features(parts[name]) == []


def test_parts_without_features_check_clean(parts: dict[str, PartSpec]) -> None:
    for name in ("block", "l_bracket", "steel_slug"):
        assert parts[name].features == () and F.check_features(parts[name]) == []


def _with(part: PartSpec, name: str, **changes: object) -> PartSpec:
    feats = tuple(dataclasses.replace(f, **changes) if f.name == name else f for f in part.features)
    return dataclasses.replace(part, features=feats)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"depth": 0.020}, "material inside the bore"),  # deeper than the real 16.5 mm
        ({"depth": 0.010}, "no floor"),  # shallower than real
        ({"radius": 0.012}, "material inside the bore"),  # wider than the bore
        ({"radius": 0.008}, "no wall"),  # much narrower than the bore
        ({"axis": (0.0, -1.0, 0.0)}, "no wall"),  # pointing out of the part
        ({"origin": (0.0, 0.002625, 0.0)}, "mouth is covered"),  # 3 mm below the floor, in the core
        ({"origin": (0.005, 0.022125, 0.0)}, "material inside the bore"),  # off the bore axis
    ],
)
def test_planted_socket_errors_are_caught(parts: dict[str, PartSpec], changes: dict, message: str) -> None:
    bad = _with(parts["pipe_tee"], "branch_socket", **changes)
    problems = F.check_features(bad)
    assert any(message in p for p in problems), problems


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"radius": 0.022}, "no rim material"),
        ({"radius": 0.014}, "material beyond the declared radius"),
        ({"depth": 0.008}, "no rim material"),
        ({"depth": 0.003}, "material beyond the declared thickness"),
        ({"axis": (0.0, 0.0, 1.0)}, "no free space under the rim"),
    ],
)
def test_planted_flange_errors_are_caught(parts: dict[str, PartSpec], changes: dict, message: str) -> None:
    bad = _with(parts["flanged_bushing"], "flange", **changes)
    problems = F.check_features(bad)
    assert any(message in p for p in problems), problems


def test_a_pin_with_clearance_reaches_the_socket_floor(parts: dict[str, PartSpec]) -> None:
    tee = parts["pipe_tee"]
    branch = F.sockets_facing(tee, "left")
    assert [f.name for f in branch] == ["branch_socket"]
    assert F.sockets_facing(tee, "right") == ()
    f = branch[0]
    assert F.free_depth(tee, f, f.radius - 0.4e-3) == pytest.approx(f.depth, abs=1e-4)
    assert F.free_depth(tee, f, f.radius + 1.0e-3) == 0.0  # too fat to enter
    with pytest.raises(ValueError, match="not a socket"):
        F.free_depth(parts["flanged_bushing"], parts["flanged_bushing"].features[1])


def test_through_bore_runs_the_full_height(parts: dict[str, PartSpec]) -> None:
    bush = parts["flanged_bushing"]
    bore = bush.sockets[0]
    assert bore.depth == pytest.approx(bush.size[2])
    assert F.free_depth(bush, bore, bore.radius - 0.3e-3) == pytest.approx(bush.size[2], abs=1e-4)


def test_inside_reports_material(parts: dict[str, PartSpec]) -> None:
    block = parts["block"]
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.031], [0.0149, 0.0, 0.0]])
    assert F.inside(block, pts).tolist() == [True, False, True]


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda: [pc.prism(pc.ngon(0.0085, 6, 90.0), "z", 0, 0.04).verts], 6),
        (lambda: [pc.box((-0.02, -0.02, 0), (0.02, 0.02, 0.01)).verts], 4),
        (lambda: [pc.box((-0.03, -0.02, 0), (0.03, 0.02, 0.01)).verts], 2),
        (lambda: [pc.prism(pc.ngon(0.02, 64), "z", 0, 0.05).verts], 0),
        (lambda: [pc.prism([(0, 0), (0.04, 0), (0, 0.03)], "z", 0, 0.01).verts], 1),
    ],
)
def test_yaw_symmetry_detection(build, expected: int) -> None:
    assert F.detect_yaw_symmetry(build()) == expected


def test_scan_like_parts_get_detected_symmetry(parts: dict[str, PartSpec]) -> None:
    slug = parts["steel_slug"].to_mesh_object()
    as_scan = lib.part_from_mesh_object(slug)
    assert as_scan.family == "gso" and as_scan.yaw_symmetry == 0 and "yaw" not in as_scan.register_axes
    tee = lib.part_from_mesh_object(parts["pipe_tee"].to_mesh_object())
    assert tee.yaw_symmetry == 1 and tee.register_axes == ("x", "y", "yaw", "tilt")
    assert tee.geometry_sha256 == parts["pipe_tee"].geometry_sha256


def test_feature_facing_uses_the_closing_axis() -> None:
    f = Feature("socket", "s", (0, 0, 0), (0.0, 1.0, 0.0), 0.005, 0.01)
    assert f.facing() == "left"
    assert dataclasses.replace(f, axis=(0.0, -1.0, 0.0)).facing() == "right"
    assert dataclasses.replace(f, axis=(1.0, 0.0, 0.0)).facing() is None
