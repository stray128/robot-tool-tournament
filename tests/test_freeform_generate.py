"""The v2 generator: exact convex pieces, seat-aligned jaws, the print solid and determinism."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import manifold3d as m3d
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from freeform_parts import block, flanged_bushing, mvp_tee, socket_hub, steel_slug, training_parts
from rtt.design.freeform import CARRIAGE_INNER_Y, FingerPair, JawGeometry, build_pair
from rtt.design.freeform.baselines import REPO, build_entrant, load_entrants_v2
from rtt.design.freeform.generate import build_pair_full
from rtt.design.freeform.hashing import design_id
from rtt.design.freeform.outline import part_geometry
from rtt.design.freeform.schema import FingerRecipe
from rtt.design.freeform.tip import piece_volume
from rtt.design.freeform.validate import InvalidDesign
from rtt.sim.task import default_task

TASK = default_task()
BASE = {"grip_height_frac": 0.5, "pad_height_mm": 20.0, "finger_width_mm": 36.0}
RECIPES = {
    "conform": FingerRecipe(
        pad="conform", tip="blunt", pad_depth_mm=5.0, wall_draft_deg=30.0, clearance_mm=0.4, **BASE
    ),
    "imprint": FingerRecipe(
        pad="conform", tip="blunt", pad_depth_mm=4.0, wall_draft_deg=0.0, clearance_mm=0.5, **BASE
    ),
    "v_funnel": FingerRecipe(
        pad="v_funnel", tip="chamfer", pad_depth_mm=6.0, wall_draft_deg=40.0, tip_size_mm=2.0, **BASE
    ),
    "flat": FingerRecipe(pad="flat", tip="blunt", **BASE),
    "hook": FingerRecipe(
        pad="conform",
        tip="hook",
        pad_depth_mm=2.0,
        wall_draft_deg=35.0,
        clearance_mm=0.4,
        tip_size_mm=4.0,
        **BASE,
    ),
}
FIVE = (mvp_tee, block, flanged_bushing, steel_slug)  # parts every recipe above builds on


def _palm_pieces(jaw: JawGeometry, side: str, y0: float, zr: float) -> list[np.ndarray]:
    """Jaw pieces in the palm frame at closure where the origin sits at palm y = +-y0, table z."""
    sgn = 1.0 if side == "left" else -1.0
    return [p + np.array([0.0, sgn * y0, zr]) for p in jaw.pieces]


def _manifold(pieces: list[np.ndarray]) -> m3d.Manifold:
    return m3d.Manifold.batch_boolean([m3d.Manifold.hull_points(p * 1000.0) for p in pieces], m3d.OpType.Add)


def _part_table(part) -> m3d.Manifold:
    g, _ = part_geometry(part)
    return _manifold([h.vertices for h in g.pieces])


def _point_triangle_distance(points: np.ndarray, tri: np.ndarray, chunk: int = 96) -> np.ndarray:
    """Unsigned distance from each point to the nearest triangle (brute force)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    out = np.full(len(points), np.inf)
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk][:, None, :]
        ap = p - a
        d1, d2 = np.einsum("ijk,jk->ij", ap, ab), np.einsum("ijk,jk->ij", ap, ac)
        bp = p - b
        d3, d4 = np.einsum("ijk,jk->ij", bp, ab), np.einsum("ijk,jk->ij", bp, ac)
        cp = p - c
        d5, d6 = np.einsum("ijk,jk->ij", cp, ab), np.einsum("ijk,jk->ij", cp, ac)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(denom != 0, vb / denom, 0.0)
            w = np.where(denom != 0, vc / denom, 0.0)
        # clamp to the triangle by projecting onto the closest feature (barycentric clamp)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        over = v + w > 1
        tot = np.where(over, v + w, 1.0)
        v, w = np.where(over, v / tot, v), np.where(over, w / tot, w)
        q = a[None] + v[..., None] * ab[None] + w[..., None] * ac[None]
        d = np.linalg.norm(p - q, axis=-1)
        # edges, for points whose clamped projection is not the closest
        for e0, e1 in ((a, b), (b, c), (c, a)):
            ed = e1 - e0
            t = np.clip(np.einsum("ijk,jk->ij", p - e0, ed) / np.einsum("jk,jk->j", ed, ed), 0, 1)
            qe = e0[None] + t[..., None] * ed[None]
            d = np.minimum(d, np.linalg.norm(p - qe, axis=-1))
        out[s : s + chunk] = d.min(axis=1)
    return out


def _hull_triangles(pieces: list[np.ndarray]) -> np.ndarray:
    tris = []
    for p in pieces:
        h = ConvexHull(p)
        tris.append(p[h.simplices])
    return np.concatenate(tris)


# ----------------------------------------------------------------------------- contract


@pytest.mark.parametrize("name", list(RECIPES))
def test_builds_valid_pairs_on_the_training_parts(name: str) -> None:
    recipe = RECIPES[name]
    for part_fn in FIVE:
        part = part_fn()
        b = build_pair_full(recipe, part, TASK)
        pair = b.pair
        assert isinstance(pair, FingerPair) and pair.part_id == part.name
        assert pair.boxes is None
        for jaw in (b.left, b.right):
            assert len(jaw.pieces) <= 40
            for pc in jaw.pieces:
                hull = ConvexHull(pc.vertices).volume
                assert abs(hull - piece_volume(pc)) <= 1e-9 * hull, (name, part.name, pc.role)
        assert 2e-3 <= pair.q_seat <= 40e-3
        assert pair.tip_height >= 3e-3 - 1e-12
        assert pair.finger_length >= recipe.pad_height_mm / 1000 + 0.015 - 1e-9
        assert pair.capture.mu == pytest.approx(TASK.physics.friction)
        assert pair.left.mass < 0.050 and pair.right.mass < 0.050


def test_jaw_frames_follow_the_contract() -> None:
    b = build_pair_full(RECIPES["conform"], mvp_tee(), TASK)
    pair = b.pair
    lmin = min(float(p[:, 1].min()) for p in pair.left.pieces)
    rmax = max(float(p[:, 1].max()) for p in pair.right.pieces)
    # the innermost point of either jaw sits on its carriage inner face
    assert min(lmin, -rmax) == pytest.approx(CARRIAGE_INNER_Y, abs=1e-9)
    assert lmin >= CARRIAGE_INNER_Y - 1e-9 and rmax <= -CARRIAGE_INNER_Y + 1e-9
    for jaw in (pair.left, pair.right):
        lo, hi = jaw.bounds
        assert hi[2] == pytest.approx(0.0, abs=1e-12)
        assert lo[2] == pytest.approx(-pair.finger_length, abs=1e-9)
    # the left jaw faces -y: its pad is at small y, its body behind it
    assert pair.left.com[1] > lmin and pair.right.com[1] < rmax


def test_right_jaw_has_its_own_geometry() -> None:
    tee = build_pair_full(RECIPES["conform"], mvp_tee(), TASK).pair
    assert tee.left.n_pieces != tee.right.n_pieces or tee.left.mass != pytest.approx(tee.right.mass)
    assert tee.left.print_mesh_sha != tee.right.print_mesh_sha
    slug = build_pair_full(RECIPES["conform"], steel_slug(), TASK).pair
    mirror = np.array([1.0, -1.0, 1.0])
    assert slug.left.n_pieces == slug.right.n_pieces
    for a, c in zip(slug.left.pieces, slug.right.pieces, strict=True):
        assert np.allclose(a * mirror, c, atol=1e-12)
    assert slug.right.mass == pytest.approx(slug.left.mass, rel=1e-9)
    assert slug.right.com == pytest.approx(slug.left.com * mirror, abs=1e-12)
    i6l, i6r = slug.left.inertia6, slug.right.inertia6
    assert i6r == pytest.approx(i6l * np.array([1, 1, 1, -1, 1, -1]), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("name", ["conform", "imprint", "v_funnel", "flat", "hook"])
def test_both_jaws_touch_the_centred_part_at_q_seat(name: str) -> None:
    """Independent 3D check of the seat alignment from the collision pieces themselves."""
    for part_fn in (mvp_tee, steel_slug):
        part = part_fn()
        b = build_pair_full(RECIPES[name], part, TASK)
        pair, plan = b.pair, b.plan
        ohg = TASK.gripper.open_half_gap
        zr = plan.z_root_table
        target = _part_table(part)
        gaps = {}
        for dq in (0.0, -5e-5):
            y0 = ohg - (pair.q_seat + dq)
            for side, jaw in (("left", pair.left), ("right", pair.right)):
                gaps[(side, dq)] = _manifold(_palm_pieces(jaw, side, y0, zr)).min_gap(target, 2.0) / 1000.0
        for side in ("left", "right"):
            assert gaps[(side, 0.0)] <= 2e-6, (name, part.name, side, gaps)
            assert gaps[(side, -5e-5)] >= 4e-5, (name, part.name, side, gaps)


def test_q_seat_is_where_the_part_blocks_the_jaws() -> None:
    """Closing 0.1 mm past q_seat pushes both jaws into the part: the seat is a blocking
    contact, not a face lying on the pad band's boundary plane that the jaws slide over.

    The stand-in bushing's flange bottom lies exactly at the top of the entrants' pad band
    (grip 0.5, pad 20 mm on a 30 mm part); the seat is on its body, not on the flange rim.
    """
    ohg = TASK.gripper.open_half_gap
    checked = 0
    for part in training_parts():
        target = _part_table(part)
        for entrant in load_entrants_v2():
            if entrant.kind != "recipe":
                continue
            try:
                b = build_pair_full(entrant, part, TASK)
            except InvalidDesign:
                continue
            pair, zr = b.pair, b.plan.z_root_table
            for side, jaw in (("left", pair.left), ("right", pair.right)):
                past = _manifold(_palm_pieces(jaw, side, ohg - (pair.q_seat + 1e-4), zr))
                before = _manifold(_palm_pieces(jaw, side, ohg - (pair.q_seat - 2e-5), zr))
                assert (past ^ target).volume() > 1e-3, (entrant.name, part.name, side)  # mm^3
                assert (before ^ target).volume() < 1e-6, (entrant.name, part.name, side)  # may touch, no cut
            checked += 1
    assert checked >= 20
    flat = {b.name: b for b in load_entrants_v2()}["flat"]
    body_r = 0.010  # the bushing's body radius: the flat pads seat on it, 0.4 mm off the carriage
    assert build_pair_full(flat, flanged_bushing(), TASK).pair.q_seat == pytest.approx(
        ohg - (body_r - CARRIAGE_INNER_Y), abs=1e-7
    )


@pytest.mark.parametrize("name", ["conform", "imprint", "v_funnel", "hook"])
def test_pad_keeps_its_clearance_at_the_designed_pose(name: str) -> None:
    recipe = RECIPES[name]
    c = (recipe.clearance_mm if recipe.pad == "conform" else 0.2) / 1000.0
    for part_fn in FIVE:
        part = part_fn()
        b = build_pair_full(recipe, part, TASK)
        plan = b.plan
        g, gm = part_geometry(part)
        for side, jp, geom in (("left", plan.left, g), ("right", plan.right, gm)):
            # grid check: the face is at or behind the grown outline everywhere
            fin = np.isfinite(jp.floor)
            assert np.all(jp.face_grid[fin] >= jp.floor[fin] - 1e-9)
            jaw = b.left if side == "left" else b.right
            pieces = [
                pc.vertices * (np.array([1.0, -1.0, 1.0]) if side == "right" else 1.0) for pc in jaw.pieces
            ]
            designed = [p + np.array([0.0, plan.y0 + jp.gap, plan.z_root_table]) for p in pieces]
            gap = _manifold(designed).min_gap(_manifold([h.vertices for h in geom.pieces]), 3.0) / 1000.0
            assert gap >= c - 2e-5, (name, part.name, side, gap)


def test_pad_zone_print_equals_collision_within_50_um() -> None:
    rng = np.random.default_rng(7)
    for name in ("conform", "v_funnel", "hook"):
        for part_fn in (mvp_tee, flanged_bushing):
            b = build_pair_full(RECIPES[name], part_fn(), TASK)
            jaw = b.left
            yk = jaw.yk_local
            zone_pieces = [pc.vertices for pc in jaw.pieces if pc.role in ("pad", "hook")]
            z_top = max(float(p[:, 2].max()) for p in zone_pieces)
            z0 = -b.pair.finger_length
            mesh = jaw.mesh
            tri_print = mesh.vertices[mesh.faces]
            # print -> collision: print surface points inside the pad zone (3 mm behind the face)
            cent = tri_print.mean(axis=1)
            face_at = np.interp(cent[:, 0], jaw.xk, yk)
            hook_at = np.interp(cent[:, 0], jaw.xk, jaw.hook_local) if jaw.hook_local is not None else face_at
            front = np.minimum(face_at, hook_at)
            zone = (cent[:, 2] >= z0 + 1e-4) & (cent[:, 2] <= z_top - 1e-4) & (cent[:, 1] <= face_at + 0.003)
            zone &= np.abs(cent[:, 0]) < b.plan.spec.width / 2 - 1e-4
            zone &= cent[:, 1] >= front - 1e-4
            pts = []
            for t in tri_print[zone]:
                w = rng.dirichlet([1, 1, 1], 12)
                pts.append(w @ t)
            pts = np.concatenate([np.concatenate(pts), cent[zone]])
            # keep the samples that are themselves inside the zone (big triangles reach out of it)
            f_at = np.interp(pts[:, 0], jaw.xk, yk)
            h_at = np.interp(pts[:, 0], jaw.xk, jaw.hook_local) if jaw.hook_local is not None else f_at
            keep = (pts[:, 2] >= z0 + 1e-4) & (pts[:, 2] <= z_top - 1e-4) & (pts[:, 1] <= f_at + 0.003)
            keep &= (np.abs(pts[:, 0]) < b.plan.spec.width / 2 - 1e-4) & (
                pts[:, 1] >= np.minimum(f_at, h_at) - 1e-4
            )
            pts = pts[keep]
            assert len(pts) > 100
            d1 = _point_triangle_distance(pts, _hull_triangles(zone_pieces))
            assert float(d1.max()) <= 5e-5, (name, part_fn.__name__, float(d1.max()))
            # collision -> print: the exposed front faces of the pad strips
            xs = rng.uniform(-b.plan.spec.width / 2 + 2e-4, b.plan.spec.width / 2 - 2e-4, 300)
            zs = rng.uniform(
                z0 + 1e-3 + (b.plan.spec.tip_size if name == "v_funnel" else 0), z_top - 1e-4, 300
            )
            if name == "hook":
                zs = rng.uniform(z0 + 9e-3, z_top - 1e-4, 300)  # above the hook
            fp = np.stack([xs, np.interp(xs, jaw.xk, yk), zs], axis=1)
            d2 = _point_triangle_distance(fp, tri_print)
            assert float(d2.max()) <= 5e-5, (name, part_fn.__name__, float(d2.max()))


def test_print_is_one_closed_solid_and_mass_comes_from_it() -> None:
    import trimesh

    b = build_pair_full(RECIPES["hook"], flanged_bushing(), TASK)
    for jaw in (b.left, b.right):
        m = trimesh.Trimesh(jaw.mesh.vertices, jaw.mesh.faces, process=False)
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0
        assert len(m.split(only_watertight=False)) == 1
        assert jaw.mass == pytest.approx(m.volume * 1240.0, rel=1e-6)
        assert np.allclose(jaw.com, m.center_mass, atol=1e-9)


def test_windows_change_mass_not_collision() -> None:
    solid = FingerRecipe(**{**RECIPES["conform"].model_dump(exclude_none=True), "body_style": "solid"})
    win = RECIPES["conform"]
    truss = FingerRecipe(**{**RECIPES["conform"].model_dump(exclude_none=True), "body_style": "truss"})
    part = block()
    a, b, c = (build_pair_full(r, part, TASK) for r in (solid, win, truss))
    assert a.pair.geometry_hash == b.pair.geometry_hash == c.pair.geometry_hash
    assert a.pair.design_id != b.pair.design_id
    assert a.left.n_window_cuts == 0 and b.left.n_window_cuts > 0
    assert a.pair.left.mass > b.pair.left.mass and a.pair.left.mass > c.pair.left.mass


def test_hook_protrudes_and_chamfer_bevels() -> None:
    b = build_pair_full(RECIPES["hook"], flanged_bushing(), TASK)
    hooks = [pc for pc in b.left.pieces if pc.role == "hook"]
    assert hooks
    pad_front = min(float(pc.vertices[:, 1].min()) for pc in b.left.pieces if pc.role == "pad")
    assert min(float(pc.vertices[:, 1].min()) for pc in hooks) < pad_front - 1e-3
    v = build_pair_full(RECIPES["v_funnel"], steel_slug(), TASK)
    strip = next(pc for pc in v.left.pieces if pc.role == "pad")
    assert len(strip.vertices) == 10  # pentagon sections: the 45-degree bevel


def test_vertices_live_on_the_1e7_grid() -> None:
    b = build_pair_full(RECIPES["v_funnel"], mvp_tee(), TASK)
    for pc in b.left.pieces + b.right.pieces:
        q = pc.vertices * 1e7
        assert np.allclose(q, np.rint(q), atol=1e-6), pc.role


def test_locator_pin_enters_a_facing_socket() -> None:
    plain = FingerRecipe(
        pad="conform",
        tip="blunt",
        pad_depth_mm=3.0,
        wall_draft_deg=30.0,
        clearance_mm=0.4,
        grip_height_frac=0.5,
        pad_height_mm=26.0,
        finger_width_mm=34.0,
    )
    pinned = FingerRecipe(**{**plain.model_dump(exclude_none=True), "locator": "conical_pin"})
    part = socket_hub()
    a = build_pair_full(plain, part, TASK)
    b = build_pair_full(pinned, part, TASK)
    pins = [pc for pc in b.left.pieces if pc.role == "locator"]
    assert len(pins) == 1 and not [pc for pc in b.right.pieces if pc.role == "locator"]
    assert any("no socket faces it" in r for r in b.pair.auto_repairs)
    assert b.pair.q_seat > a.pair.q_seat  # the pin is now the innermost point
    pin = b.plan.left.pin
    assert pin is not None and pin.r_tip < pin.r_base and pin.length > 0.003
    straight = FingerRecipe(**{**plain.model_dump(exclude_none=True), "locator": "straight_pin"})
    s = build_pair_full(straight, part, TASK)
    assert s.plan.left.pin.r_tip == pytest.approx(s.plan.left.pin.r_base)


def test_entrants_build_and_carry_their_rules() -> None:
    entrants = {b.name: b for b in load_entrants_v2()}
    for part_fn in FIVE:
        part = part_fn()
        conf = build_entrant(entrants["imprint_configurator"], part, TASK)
        assert any("configurator imprint depth" in r for r in conf.auto_repairs)
        swept = build_entrant(entrants["imprint_swept"], part, TASK)
        assert swept.design_id != conf.design_id
        flat = build_entrant(entrants["flat"], part, TASK)
        assert flat.design_id == design_id(entrants["flat"].recipe) and flat.part_id == part.name
        stock = build_entrant(entrants["stock"], part, TASK)
        if (REPO / entrants["stock"].hulls).is_file():  # the vendored blade hulls
            assert stock.boxes is None and stock.left.n_pieces <= 40
        else:  # the v1 box approximation until the hulls are vendored
            assert stock.boxes is not None and len(stock.boxes) == 14
        assert stock.q_seat > 0 and stock.left.n_pieces == stock.right.n_pieces


def test_stock_entrant_meets_the_seat_closure_rule() -> None:
    """The stock path applies the same q_seat range as every recipe (validator stage 2)."""
    from freeform_parts import thin_fin

    stock = {b.name: b for b in load_entrants_v2()}["stock"]
    with pytest.raises(InvalidDesign) as info:  # 3 mm across the jaws: the blades seat at ~41 mm
        build_entrant(stock, thin_fin(), TASK)
    assert info.value.code == "q_seat_range" and "thin_fin" in str(info.value)


def test_invalid_build_raises_with_a_code() -> None:
    deep = FingerRecipe(
        pad="conform",
        tip="blunt",
        pad_depth_mm=15.0,
        wall_draft_deg=0.0,
        clearance_mm=0.2,
        grip_height_frac=0.5,
        pad_height_mm=20.0,
        finger_width_mm=40.0,
    )
    from freeform_parts import hex_standoff

    with pytest.raises(InvalidDesign) as info:
        build_pair(deep, hex_standoff(), TASK)
    assert info.value.code == "q_seat_range" and "hex_standoff" in str(info.value)


def test_build_is_cached_and_deterministic() -> None:
    r = RECIPES["hook"]
    a = build_pair_full(r, mvp_tee(), TASK)
    assert build_pair_full(r, mvp_tee(), TASK) is a
    b = build_pair_full(r, mvp_tee(), TASK, use_cache=False)
    assert b is not a
    assert a.pair.geometry_hash == b.pair.geometry_hash
    assert a.pair.left.print_mesh_sha == b.pair.left.print_mesh_sha
    assert a.pair.left.mass == b.pair.left.mass
    assert a.pair.capture == b.pair.capture


def test_cache_keeps_part_ids_and_repair_notes_apart() -> None:
    """Same geometry under another name, or a recipe that differs only in an inactive value,
    hits the geometry but never another build's part id or canonicalisation notes."""
    import dataclasses

    tee = mvp_tee()
    twin = dataclasses.replace(tee, name="tee_twin")
    flat = RECIPES["flat"]
    assert build_pair(flat, tee, TASK).part_id == tee.part_id
    assert build_pair(flat, twin, TASK).part_id == "tee_twin"
    noisy = FingerRecipe(**dict(BASE, pad="flat", tip="blunt", pad_depth_mm=5.0))
    a, b = build_pair(noisy, steel_slug(), TASK), build_pair(flat, steel_slug(), TASK)
    assert a.design_id == b.design_id and a.geometry_hash == b.geometry_hash
    assert any("pad_depth_mm is inactive" in n for n in a.auto_repairs)
    assert not any("inactive" in n for n in b.auto_repairs)


def test_configurator_stroke_cap_keeps_q_seat_in_the_stroke() -> None:
    """Where the stroke cap sets the imprint depth, the imprint seats at q_max exactly
    (it used to come out 0.8 mm past it and fail q_seat_range)."""
    from freeform_parts import ALU, box, make_part

    part = make_part(
        "step_block",
        "block",
        [
            box((-0.015, -0.030, 0.0), (0.015, 0.030, 0.005)),
            box((-0.015, -0.008, 0.005), (0.015, 0.030, 0.040)),  # 8 mm on the right jaw's side
        ],
        ALU,
    )
    entrant = {b.name: b for b in load_entrants_v2()}["imprint_configurator"]
    b = build_pair_full(entrant, part, TASK)
    assert b.pair.q_seat == pytest.approx(0.040, abs=1e-7)
    assert any("depth on step_block: 5.6 mm" in n for n in b.pair.auto_repairs)


_HASH_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from freeform_parts import mvp_tee, flanged_bushing
from rtt.design.freeform.generate import build_pair_full
from rtt.design.freeform.hashing import design_id
from rtt.design.freeform.schema import FingerRecipe
from rtt.sim.task import default_task
task = default_task()
out = []
for spec in json.loads(sys.argv[2]):
    r = FingerRecipe(**spec)
    for part in (mvp_tee(), flanged_bushing()):
        p = build_pair_full(r, part, task).pair
        out.append([p.design_id, p.geometry_hash, p.left.print_mesh_sha, p.right.print_mesh_sha, repr(p.left.mass)])
print(json.dumps(out))
"""


def test_hashes_identical_across_three_processes() -> None:
    specs = [RECIPES[k].model_dump(exclude_none=True) for k in ("conform", "v_funnel", "hook")]
    tests_dir = str(Path(__file__).resolve().parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p), PYTHONHASHSEED="0")
    outs = []
    for seed in ("1", "2", "3"):
        env["PYTHONHASHSEED"] = seed
        res = subprocess.run(
            [sys.executable, "-c", _HASH_SCRIPT, tests_dir, json.dumps(specs)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1] == outs[2]
    assert len({row[1] for row in outs[0]}) == len(outs[0])


def test_build_time_budget() -> None:
    times = []
    for r in RECIPES.values():
        for part_fn in FIVE:
            t = time.perf_counter()
            build_pair_full(r, part_fn(), TASK, use_cache=False)
            times.append(time.perf_counter() - t)
    # GV0 asks for p95 <= 0.5 s per pair on the gate machine; this guards against regressions
    assert float(np.percentile(times, 95)) <= 1.0, times


def test_all_six_training_parts_build_for_a_wide_low_pad() -> None:
    r = FingerRecipe(
        pad="conform",
        tip="chamfer",
        pad_depth_mm=3.0,
        wall_draft_deg=40.0,
        clearance_mm=0.4,
        tip_size_mm=1.5,
        grip_height_frac=0.15,
        pad_height_mm=12.0,
        finger_width_mm=40.0,
    )
    ids = {}
    for part in training_parts():
        ids[part.name] = build_pair(r, part, TASK).geometry_hash
    assert len(set(ids.values())) == 6
