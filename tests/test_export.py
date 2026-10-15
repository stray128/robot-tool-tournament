"""Mesh export: watertight single-solid unions, mirroring, STL and GLB files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from rtt.design import export
from rtt.design.export import export_glb, export_stl, gripper_scene, to_trimesh
from rtt.design.generate import generate_finger, geometry_bounds, mass_and_inertia
from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask, load_entrants
from rtt.design.validate import validate

ENTRANTS = load_entrants(Path(__file__).resolve().parents[1] / "configs" / "entrants.yaml")
WELD = export.WELD_M


def _box_volume(design: FingerDesign) -> float:
    geom = generate_finger(design)
    return mass_and_inertia(geom).mass_kg / geom.density_kg_m3


@pytest.mark.parametrize("entrant", ENTRANTS, ids=lambda e: e.name)
def test_entrant_mesh_is_one_watertight_solid(entrant: FingerDesign) -> None:
    geom = generate_finger(entrant)
    mesh = to_trimesh(geom)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.body_count == 1
    assert 0.0 < mesh.volume <= _box_volume(entrant) * 1.01
    lo, hi = geometry_bounds(geom)
    assert np.allclose(mesh.bounds, [lo, hi], atol=2 * WELD)


def test_mirror_gives_the_right_finger() -> None:
    geom = generate_finger(ENTRANTS[2])
    left, right = to_trimesh(geom), to_trimesh(geom, mirror=True)
    assert right.is_watertight and right.volume > 0.0
    assert right.volume == pytest.approx(left.volume, rel=1e-9)
    assert np.allclose(right.bounds[:, 1], -left.bounds[::-1, 1])
    assert np.allclose(right.bounds[:, [0, 2]], left.bounds[:, [0, 2]])


def test_random_designs_export_as_single_watertight_solids() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 40:
        params = {key: float(rng.uniform(lo, hi)) for key, (lo, hi) in PARAM_BOUNDS.items()}
        params["inner_face"] = str(rng.choice(CATEGORICAL_OPTIONS["inner_face"]))
        params["tip_profile"] = str(rng.choice(CATEGORICAL_OPTIONS["tip_profile"]))
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        design = FingerDesign(**params)
        if not validate(design).ok:
            continue
        mesh = to_trimesh(generate_finger(design))
        assert mesh.is_watertight and mesh.body_count == 1, params
        assert mesh.volume <= _box_volume(design) * 1.01
        checked += 1


def test_export_stl_round_trip_in_millimetres(tmp_path: Path) -> None:
    geom = generate_finger(ENTRANTS[1])
    path = export_stl(geom, tmp_path / "out" / "concave.stl")
    loaded = trimesh.load(path, force="mesh")
    assert loaded.is_watertight
    lo, hi = geometry_bounds(geom)
    assert np.allclose(loaded.bounds, np.array([lo, hi]) * 1000.0, atol=0.05)
    metres = trimesh.load(export_stl(geom, tmp_path / "m.stl", units="m"), force="mesh")
    assert np.allclose(metres.bounds, [lo, hi], atol=5e-5)
    with pytest.raises(ValueError, match="units"):
        export_stl(geom, tmp_path / "cm.stl", units="cm")  # type: ignore[arg-type]


def test_gripper_scene_places_fingers_by_stroke() -> None:
    task = GripperTask()
    geom = generate_finger(ENTRANTS[0])  # straight: inner faces at the root planes
    for opening, gap_mm in ((1.0, task.max_opening_mm), (0.0, task.closed_gap_mm)):
        scene = gripper_scene(geom, task, opening=opening, y_up=False)
        left = scene.geometry["finger_left"].copy()
        left.apply_transform(scene.graph.get("finger_left")[0])
        right = scene.geometry["finger_right"].copy()
        right.apply_transform(scene.graph.get("finger_right")[0])
        gap = (left.bounds[0, 1] - right.bounds[1, 1]) * 1000.0
        assert gap == pytest.approx(gap_mm, abs=0.05)
    with pytest.raises(ValueError):
        gripper_scene(geom, task, opening=1.5)


def test_export_glb_preview(tmp_path: Path) -> None:
    path = export_glb(generate_finger(ENTRANTS[2]), tmp_path / "hooked.glb")
    scene = trimesh.load(path)
    assert isinstance(scene, trimesh.Scene)
    assert {"finger_left", "finger_right", "palm"} <= set(scene.geometry)
    lo, hi = scene.bounds
    assert hi[1] > 0.02 and lo[1] < -0.07  # Y-up: palm above, fingers hanging down


def test_fallback_without_manifold_still_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_geom: object) -> trimesh.Trimesh:
        raise RuntimeError("manifold3d unavailable")

    monkeypatch.setattr(export, "_union_manifold", broken)
    geom = generate_finger(ENTRANTS[1])
    mesh = to_trimesh(geom)
    lo, hi = geometry_bounds(geom)
    assert np.allclose(mesh.bounds, [lo, hi], atol=1e-9)
