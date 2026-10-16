"""Bring Your Part intake: example parts, units, repair, mass, stable poses and grip candidates."""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rtt.part.examples import EXAMPLE_PARTS, EXAMPLES_DIR, example_path, write_examples
from rtt.part.intake import (
    MATERIALS,
    PartError,
    detect_units,
    intake,
    load_part,
    rotate_quarter_turn,
    save_part,
)
from rtt.sim.task import default_task

TASK = default_task()


def _hull_intake(path: Path, **kwargs: object):
    """Intake with one convex hull per body: milliseconds instead of CoACD's seconds."""
    kwargs.setdefault("decomposition", "hull")
    return intake(path, **kwargs)  # type: ignore[arg-type]


def _box_stl(tmp_path: Path, extents_mm: tuple[float, float, float], name: str = "box") -> Path:
    path = tmp_path / f"{name}.stl"
    trimesh.creation.box(extents=extents_mm).export(path)
    return path


# --------------------------------------------------------------------------- example parts


def test_example_parts_are_watertight_and_match_the_committed_files(tmp_path: Path) -> None:
    written = write_examples(tmp_path)
    assert [p.stem for p in written] == list(EXAMPLE_PARTS)
    for path in written:
        mesh = trimesh.load(path, force="mesh")
        assert mesh.is_watertight and mesh.volume > 0
        committed = EXAMPLES_DIR / path.name
        assert committed.read_bytes() == path.read_bytes(), (
            f"{committed} is stale: run python -m rtt.part.examples"
        )
    assert example_path("spur_gear") == EXAMPLES_DIR / "spur_gear.stl"
    with pytest.raises(KeyError):
        example_path("sprocket")


@pytest.mark.parametrize("name", list(EXAMPLE_PARTS))
def test_every_example_part_passes_intake(name: str) -> None:
    part = _hull_intake(example_path(name))
    assert part.units == "mm" and part.units_detected
    assert part.watertight and not part.mass_is_approximate
    mesh = trimesh.load(example_path(name), force="mesh")
    assert part.volume_cm3 == pytest.approx(mesh.volume / 1000.0, rel=1e-6)
    assert part.mass_g == pytest.approx(mesh.volume * 1e-9 * MATERIALS["plastic"] * 1000.0, rel=1e-6)
    probs = [p.probability for p in part.poses]
    assert probs == sorted(probs, reverse=True) and 0 < sum(probs) <= 1.0 + 1e-9
    assert 1 <= len(part.poses) <= 3
    for pose in part.poses:
        assert pose.extents_mm[1] <= part.max_grip_width_mm + 1e-6  # fits the opening
        assert pose.candidates and pose.candidates[0].axis == "narrow"
        assert np.allclose(sorted(pose.extents_mm), sorted(part.extents_mm), atol=1.5) or name == "l_bracket"


def test_symmetric_poses_are_merged() -> None:
    gear = _hull_intake(example_path("spur_gear"))
    assert gear.poses[0].probability > 0.8  # both flat faces of the gear are one pose
    assert (
        len(gear.poses[0].merged) >= 2 and len(gear.poses[1].merged) >= 2
    )  # and every tooth it can roll onto
    dumbbell = _hull_intake(example_path("dumbbell_handle"))
    assert dumbbell.poses[0].probability > 0.85  # every facet of the round bar/discs, lying down
    assert dumbbell.poses[0].extents_mm[2] == pytest.approx(40.0, abs=0.5)


def test_off_centre_grasp_points_find_the_material() -> None:
    dumbbell = _hull_intake(example_path("dumbbell_handle"), decomposition="coacd", top_k=1)
    offsets = [c.offset_mm for c in dumbbell.poses[0].candidates if c.axis == "narrow"]
    assert offsets[0] == 0.0 and len(offsets) == 2 and abs(offsets[1]) > 25.0  # the bar and a disc
    bracket = _hull_intake(example_path("l_bracket"), decomposition="coacd", top_k=2)
    standing = next(p for p in bracket.poses if p.extents_mm[2] == pytest.approx(50.0, abs=0.5))
    narrow = [c for c in standing.candidates if c.axis == "narrow"]
    # the upright; the centre only has the 4 mm base plate, below the lowest fingertip height
    assert [abs(c.offset_mm) >= 15.0 for c in narrow] == [True]


def test_wide_axis_candidates_turn_the_object() -> None:
    bracket = _hull_intake(example_path("l_bracket"))
    pose = next(p for p in bracket.poses if "wide" in p.objects)
    narrow, wide = pose.obj("narrow"), pose.obj("wide")
    assert wide.size[0] == pytest.approx(narrow.size[1]) and wide.size[1] == pytest.approx(narrow.size[0])
    assert any(c.axis == "wide" for c in pose.candidates)
    no_wide = _hull_intake(example_path("l_bracket"), wide_axis=False)
    assert all(set(p.objects) == {"narrow"} for p in no_wide.poses)


def test_rotate_quarter_turn_keeps_mass_properties() -> None:
    obj = _hull_intake(example_path("l_bracket")).poses[0].obj()
    turned = rotate_quarter_turn(obj)
    assert turned.volume == obj.volume
    assert turned.size == pytest.approx((obj.size[1], obj.size[0], obj.size[2]))
    ixx, iyy, izz, ixy, _ixz, _iyz = obj.unit_inertia
    assert turned.unit_inertia[:3] == pytest.approx((iyy, ixx, izz))
    assert turned.unit_inertia[3] == pytest.approx(-ixy, abs=1e-15)
    assert turned.com == pytest.approx((-obj.com[1], obj.com[0], obj.com[2]))
    assert turned.geometry_sha256 != obj.geometry_sha256
    assert rotate_quarter_turn(rotate_quarter_turn(turned)).size == pytest.approx(turned.size)


# --------------------------------------------------------------------------- units and size


def test_detect_units() -> None:
    assert detect_units(84.0) == "mm"
    assert detect_units(0.084) == "m"
    with pytest.raises(PartError):
        detect_units(0.0)


def test_metre_files_are_detected_and_match_millimetre_files(tmp_path: Path) -> None:
    mesh = trimesh.load(example_path("rounded_tub"), force="mesh")
    mesh.apply_scale(0.001)
    path = tmp_path / "tub_m.stl"
    mesh.export(path)
    in_m = _hull_intake(path)
    in_mm = _hull_intake(example_path("rounded_tub"))
    assert in_m.units == "m" and in_m.units_detected
    assert in_m.extents_mm == pytest.approx(in_mm.extents_mm, abs=1e-3)
    assert in_m.mass_g == pytest.approx(in_mm.mass_g, rel=1e-6)


def test_explicit_units_override_detection(tmp_path: Path) -> None:
    path = _box_stl(tmp_path, (4.0, 3.0, 2.0))  # 4 cm as centimetres
    part = _hull_intake(path, units="cm")
    assert part.units == "cm" and not part.units_detected
    assert part.extents_mm == pytest.approx((40.0, 30.0, 20.0))
    with pytest.raises(PartError, match="units"):
        _hull_intake(path, units="kilometres")


def test_too_large_and_too_small_parts_are_refused(tmp_path: Path) -> None:
    big = _box_stl(tmp_path, (900.0, 30.0, 30.0), "big")
    with pytest.raises(PartError, match="even read as millimetres"):
        _hull_intake(big)
    with pytest.raises(PartError, match="in cm"):
        _hull_intake(example_path("dumbbell_handle"), units="cm")
    tiny = _box_stl(tmp_path, (2.0, 2.0, 2.0), "tiny")
    with pytest.raises(PartError, match="smallest part"):
        _hull_intake(tiny)
    small = _box_stl(tmp_path, (12.0, 8.0, 6.0), "small")
    part = _hull_intake(small)
    assert any("cm or inches" in w for w in part.warnings)


def test_flat_meshes_are_refused_before_the_pose_search(tmp_path: Path) -> None:
    """A zero-thickness sheet would put trimesh's stable-pose search into an endless loop."""
    xs, ys = np.meshgrid(np.linspace(0, 40, 5), np.linspace(0, 30, 4))
    grid = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    faces = []
    for r in range(3):
        for c in range(4):
            a, b, d = r * 5 + c, r * 5 + c + 1, (r + 1) * 5 + c
            faces += [[a, b, d + 1], [a, d + 1, d]]
    for name, z_of in (("sheet", lambda v: 0.0 * v[:, 0]), ("tilted", lambda v: 0.25 * v[:, 0])):
        verts = grid.copy()
        verts[:, 2] = z_of(verts)
        path = tmp_path / f"{name}.stl"
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)
        t0 = time.perf_counter()
        with pytest.raises(PartError, match="flat"):
            _hull_intake(path)
        assert time.perf_counter() - t0 < 10.0
    # a 1e-6 mm box is a sheet too
    with pytest.raises(PartError, match="flat"):
        _hull_intake(_box_stl(tmp_path, (40.0, 30.0, 1e-6), "film"))


def test_parts_wider_than_the_opening_are_refused(tmp_path: Path) -> None:
    cube = _box_stl(tmp_path, (100.0, 100.0, 100.0), "cube")
    with pytest.raises(PartError, match="at most 79 mm"):
        _hull_intake(cube)


def test_wide_poses_are_reported_but_not_tested(tmp_path: Path) -> None:
    plate = _box_stl(tmp_path, (120.0, 90.0, 10.0), "plate")  # flat: 90 mm across; on edge: 10 mm
    part = _hull_intake(plate)
    assert all(p.extents_mm[1] <= 79.0 for p in part.poses)
    assert any(o["reason"] == "too wide for the opening" for o in part.other_poses)
    assert any("too wide for the gripper" in w for w in part.warnings)


# --------------------------------------------------------------------------- repair, mass, errors


def test_a_small_hole_is_repaired(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(40.0, 30.0, 20.0))
    mesh.update_faces(np.arange(1, len(mesh.faces)))  # drop one triangle
    assert not mesh.is_watertight
    path = tmp_path / "holed.stl"
    mesh.export(path)
    part = _hull_intake(path)
    assert part.watertight and any("filled holes" in r for r in part.repairs)
    assert part.volume_cm3 == pytest.approx(24.0, rel=1e-6)


def test_an_open_mesh_is_simulated_from_its_convex_pieces(tmp_path: Path) -> None:
    mesh = trimesh.creation.box(extents=(40.0, 30.0, 20.0))
    normals = mesh.face_normals
    keep = ~((normals[:, 2] > 0.9) | (normals[:, 0] > 0.9))  # remove the top and one side: a 6-edge hole
    mesh.update_faces(np.flatnonzero(keep))
    path = tmp_path / "open.stl"
    mesh.export(path)
    part = _hull_intake(path)
    assert not part.watertight and part.mass_is_approximate
    assert any("not a closed solid" in w for w in part.warnings)
    assert part.volume_cm3 == pytest.approx(24.0, rel=0.02)  # the hull of the remaining faces


def test_material_and_density(tmp_path: Path) -> None:
    path = _box_stl(tmp_path, (60.0, 60.0, 60.0), "block")
    steel = _hull_intake(path, material="steel")
    assert steel.mass_g == pytest.approx(216.0 * 7.85, rel=1e-6)
    assert any("friction alone" in w for w in steel.warnings)  # 1.7 kg > ~1.5 kg
    custom = _hull_intake(path, density=500.0)
    assert custom.material == "custom" and custom.mass_g == pytest.approx(108.0)
    with pytest.raises(PartError, match="not both"):
        _hull_intake(path, material="pla", density=1000.0)
    with pytest.raises(PartError, match="unknown material"):
        _hull_intake(path, material="unobtainium")
    with pytest.raises(PartError, match="between"):
        _hull_intake(path, density=1e6)
    heavy = _box_stl(tmp_path, (70.0, 70.0, 70.0), "heavy")
    with pytest.raises(PartError, match="tested up to"):
        _hull_intake(heavy, material="brass")


def test_unreadable_uploads_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PartError, match="no file"):
        intake(tmp_path / "missing.stl")
    step = tmp_path / "part.step"
    step.write_text("ISO-10303-21;")
    with pytest.raises(PartError, match="unsupported file type"):
        intake(step)
    junk = tmp_path / "junk.stl"
    junk.write_bytes(b"\x00" * 10)
    with pytest.raises(PartError):
        intake(junk)


def test_names_are_sanitised(tmp_path: Path) -> None:
    path = _box_stl(tmp_path, (40.0, 30.0, 20.0), "Box Part")
    part = _hull_intake(path, name="My <Part> #2")
    assert part.name == "my_part_2"
    assert part.source_name == "Box Part.stl"
    assert str(tmp_path) not in json.dumps(part.to_json())  # no local paths leak into the spec


# --------------------------------------------------------------------------- decomposition, cache, persistence


def test_coacd_decomposition_is_cached_and_deterministic(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    t0 = time.perf_counter()
    first = intake(example_path("dumbbell_handle"), cache_dir=cache, top_k=1)
    cold = time.perf_counter() - t0
    t1 = time.perf_counter()
    second = intake(example_path("dumbbell_handle"), cache_dir=cache, top_k=1)
    warm = time.perf_counter() - t1
    assert first.decomposition == "coacd" and first.poses[0].n_pieces > 1
    assert first.poses[0].obj().geometry_sha256 == second.poses[0].obj().geometry_sha256
    assert list(cache.glob("*.npz")) and warm < cold


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    part = _hull_intake(example_path("l_bracket"))
    save_part(part, tmp_path / "part")
    again = load_part(tmp_path / "part")
    assert again.to_json() == part.to_json()
    for a, b in zip(part.poses, again.poses, strict=True):
        assert {k: o.geometry_sha256 for k, o in a.objects.items()} == {
            k: o.geometry_sha256 for k, o in b.objects.items()
        }
    assert again.mesh_mm is not None and again.mesh_mm.volume == pytest.approx(part.mesh_mm.volume, rel=1e-5)
    assert "l_bracket" in part.summary() and "pose 0" in part.summary()


def test_top_k_and_min_probability(tmp_path: Path) -> None:
    one = _hull_intake(example_path("rounded_tub"), top_k=1)
    assert len(one.poses) == 1
    strict = _hull_intake(example_path("rounded_tub"), min_probability=0.99)
    assert len(strict.poses) == 1  # the most likely fitting pose is always kept
    with pytest.raises(ValueError):
        _hull_intake(example_path("rounded_tub"), top_k=0)


# --------------------------------------------------------------------------- upload hardening


def test_obj_side_files_are_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An OBJ's mtllib / texture references must not make intake read other files."""
    (tmp_path / "evil.mtl").write_text("newmtl m\nKd 1 0 0\nmap_Kd secret.png\n")
    sub = tmp_path / "upload"
    sub.mkdir()
    (sub / "same.mtl").write_text("newmtl m\nKd 1 0 0\n")
    box = trimesh.creation.box(extents=(30.0, 20.0, 10.0))
    body = trimesh.exchange.obj.export_obj(box, include_normals=False)
    obj = sub / "part.obj"
    obj.write_text("mtllib same.mtl\nmtllib ../evil.mtl\nmtllib /etc/passwd\nusemtl m\n" + body)
    opened: list[str] = []
    real_open = open

    def spy(file: object, *args: object, **kwargs: object) -> object:
        opened.append(str(file))
        return real_open(file, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr("builtins.open", spy)
    part = _hull_intake(obj)
    monkeypatch.undo()
    assert part.watertight and part.extents_mm == pytest.approx((30.0, 20.0, 10.0))
    assert not [f for f in opened if f.endswith((".mtl", "passwd", ".png"))]


def test_the_hash_is_of_the_bytes_that_were_parsed(tmp_path: Path) -> None:
    import hashlib

    path = _box_stl(tmp_path, (40.0, 30.0, 20.0))
    assert _hull_intake(path).source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_oversized_files_are_refused_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    intake_mod = importlib.import_module("rtt.part.intake")  # the package re-exports a function of that name

    path = _box_stl(tmp_path, (40.0, 30.0, 20.0))
    monkeypatch.setattr(intake_mod, "MAX_FILE_BYTES", path.stat().st_size - 1)
    monkeypatch.setattr(intake_mod.trimesh, "load", lambda *a, **k: pytest.fail("parsed an oversized file"))
    with pytest.raises(PartError, match="the limit is"):
        intake(path, decomposition="hull")


def test_heavy_closed_parts_are_refused_before_decomposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_mod = importlib.import_module("rtt.part.intake")  # the package re-exports a function of that name

    heavy = _box_stl(tmp_path, (70.0, 70.0, 70.0), "heavy")
    monkeypatch.setattr(intake_mod, "_decompose", lambda *a, **k: pytest.fail("decomposed a refused part"))
    with pytest.raises(PartError, match="tested up to"):
        intake(heavy, material="steel")


def test_load_part_refuses_unknown_axes(tmp_path: Path) -> None:
    part = _hull_intake(example_path("rounded_tub"), top_k=1)
    spec = save_part(part, tmp_path / "part")
    data = json.loads(spec.read_text())
    data["poses"][0]["objects"]["../../escape"] = data["poses"][0]["objects"]["narrow"]
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown jaw axes"):
        load_part(tmp_path / "part")


# --------------------------------------------------------------------------- resource bounds


def test_too_many_bodies_are_refused_without_splitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_mod = importlib.import_module("rtt.part.intake")
    n = intake_mod.MAX_BODIES + 1
    tet = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    bodies = [tet.copy().apply_translation((3.0 * (i % 9), 3.0 * (i // 9), 0.0)) for i in range(n)]
    path = tmp_path / "assembly.stl"
    trimesh.util.concatenate(bodies).export(path)
    # Splitting tens of thousands of bodies took minutes; the count must not need it.
    monkeypatch.setattr(trimesh.Trimesh, "split", lambda *a, **k: pytest.fail("split every body"))
    with pytest.raises(PartError, match=f"{n} separate bodies"):
        intake(path, decomposition="hull")


def test_a_finely_tessellated_outer_surface_is_refused(tmp_path: Path) -> None:
    from rtt.part.intake import MAX_HULL_FACES

    ball = trimesh.creation.icosphere(subdivisions=5, radius=30.0)  # 20,480 faces, all on the hull
    assert len(ball.faces) > MAX_HULL_FACES
    path = tmp_path / "ball.stl"
    ball.export(path)
    started = time.perf_counter()
    with pytest.raises(PartError, match="coarser tessellation"):
        intake(path, decomposition="hull")
    assert time.perf_counter() - started < 10.0  # refused before the stable-pose search (~100 s)


def test_section_profile_is_identical_in_small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    intake_mod = importlib.import_module("rtt.part.intake")
    obj = _hull_intake(example_path("l_bracket"), top_k=1).primary.obj()
    xs, area = intake_mod._section_profile(obj, 0.3)
    monkeypatch.setattr(intake_mod, "PROFILE_CHUNK", 997)  # batches far smaller than one x column
    xs2, area2 = intake_mod._section_profile(obj, 0.3)
    assert np.array_equal(xs, xs2) and np.array_equal(area, area2)
    assert area.max() > 0


def test_a_half_metre_part_stays_within_memory(tmp_path: Path) -> None:
    import tracemalloc

    bar = trimesh.creation.cylinder(radius=8.0, height=470.0, sections=48)
    bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0.0, 1.0, 0.0)))
    discs = [
        trimesh.creation.cylinder(radius=38.0, height=10.0, sections=48)
        .apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0.0, 1.0, 0.0)))
        .apply_translation((x, 0.0, 0.0))
        for x in (-230.0, 230.0)
    ]
    path = tmp_path / "long_bar.stl"
    trimesh.boolean.union([bar, *discs], engine="manifold").export(path)
    tracemalloc.start()  # numpy reports its buffers to tracemalloc
    try:
        part = intake(path, decomposition="hull", top_k=1)
        peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()
    assert part.extents_mm[0] == pytest.approx(470.0, abs=0.5)
    assert peak_mb < 400, f"intake peaked at {peak_mb:.0f} MB (the unbatched grid took ~2.3 GB)"


def test_a_slow_decomposition_is_stopped(tmp_path: Path) -> None:
    import multiprocessing

    from rtt.sim.mesh import coacd_available

    if not coacd_available():
        pytest.skip("CoACD is not installed")
    with pytest.raises(PartError, match="did not finish within"):
        intake(example_path("l_bracket"), decomposition="coacd", top_k=1, decompose_timeout_s=0.01)
    time.sleep(0.2)
    assert not multiprocessing.active_children(), "a decomposition worker outlived its deadline"
    with pytest.raises(ValueError, match="decompose_timeout_s"):
        intake(example_path("l_bracket"), decompose_timeout_s=float("nan"))
