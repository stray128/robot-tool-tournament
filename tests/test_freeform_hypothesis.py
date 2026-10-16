"""Gate GV0 for the freeform v2 generator: random recipes on the training parts.

Every recipe drawn from the unit cube either builds on every part or is rejected with a
specific code and message, and every build passes independent geometry checks:

* every collision piece is convex (hull volume = piece volume within 1e-9 relative);
* each print solid is one closed manifold, also after writing it as a float32 STL;
* at its seat closure each jaw touches the centred part and 10 um earlier it does not, so no
  pad cuts into the part before it seats; at its designed pose a non-flat jaw keeps its
  clearance (minus 0.02 mm) from the part;
* on a sample, the pad-zone print equals the collision pieces within 0.05 mm (both ways).

The default run is small (Hypothesis plus a 16-point scrambled Sobol sweep). The full gate
(2,048 Sobol recipes x 6 parts) runs with ``RTT_GV0_SOBOL=2048``; ``RTT_GV0_WORKERS`` sets the
process count and ``RTT_GV0_REPORT`` a path for the JSON report.

Parts: the parts library's training suite when it is importable, else the exact-piece
stand-ins in ``freeform_parts``.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rtt.design.freeform.schema import SCORED_PARAMS, recipe_from_unit
from rtt.design.freeform.validate import CODES, InvalidDesign
from rtt.sim.task import default_task

SEAT_BACKOFF = 1e-5  # m; a jaw this far before its seat must not touch the part
TOUCH_TOL = 2e-6  # m; at its seat a jaw is within this of the part
CLEARANCE_TOL = 2e-5  # m; designed-pose distance >= clearance - 0.02 mm
HAUSDORFF_MAX = 5e-5  # m
CONVEX_RTOL = 1e-9


def suite_parts() -> tuple[Any, ...]:
    """The six training parts: the parts library's when present, else the test stand-ins."""
    try:
        from rtt.parts.library import training_parts as library_parts

        return tuple(library_parts())
    except ImportError:
        from freeform_parts import training_parts

        return training_parts()


# ----------------------------------------------------------------------------- checks


def _manifold(pieces: list[np.ndarray], shift: np.ndarray) -> Any:
    import manifold3d as m3d

    hulls = [m3d.Manifold.hull_points((np.asarray(p) + shift) * 1000.0) for p in pieces]
    return m3d.Manifold.batch_boolean(hulls, m3d.OpType.Add)


def _part_manifold(part: Any) -> Any:
    from rtt.design.freeform.outline import part_geometry

    g, _ = part_geometry(part)
    return _manifold([h.vertices for h in g.pieces], np.zeros(3))


def convex_error(pieces: list[Any]) -> float:
    """Largest relative difference between a piece's hull volume and its own volume."""
    from scipy.spatial import ConvexHull

    from rtt.design.freeform.tip import piece_volume

    worst = 0.0
    for pc in pieces:
        hull = ConvexHull(pc.vertices).volume
        worst = max(worst, abs(hull - piece_volume(pc)) / hull)
    return worst


def stl_watertight(mesh: Any) -> bool:
    """The jaw written as its binary STL and read back (float32 mm, welded) is one closed solid."""
    from rtt.design.freeform.export import edge_report, stl_bytes

    data = stl_bytes(mesh.vertices, mesh.faces)
    tri = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"]
    verts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    if np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])):
        return False
    return edge_report(faces) == (True, 1) and len(verts) >= 4


def seat_checks(build: Any, task: Any, part_m: Any) -> dict[str, float]:
    """Independent seat and clearance checks from the pair's own pieces (metres)."""
    pair, plan = build.pair, build.plan
    ohg = float(task.gripper.open_half_gap)
    zr = pair.tip_height + pair.finger_length
    out: dict[str, float] = {"touch": 0.0, "early": math.inf, "clear_margin": math.inf}
    for side, jaw, jp in (("left", pair.left, plan.left), ("right", pair.right, plan.right)):
        sgn = 1.0 if side == "left" else -1.0
        pieces = list(jaw.pieces)

        def gap_at(q: float, pieces: list[np.ndarray] = pieces, sgn: float = sgn) -> float:
            shift = np.array([0.0, sgn * (ohg - q), zr])
            return _manifold(pieces, shift).min_gap(part_m, 3.0) / 1000.0

        out["touch"] = max(out["touch"], gap_at(pair.q_seat))
        out["early"] = min(out["early"], gap_at(pair.q_seat - SEAT_BACKOFF))
        if plan.spec.pad != "flat":
            out["clear_margin"] = min(out["clear_margin"], gap_at(pair.q_seat - jp.gap) - plan.spec.clearance)
    return out


def _point_triangle_distance(points: np.ndarray, tri: np.ndarray, chunk: int = 128) -> np.ndarray:
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    out = np.full(len(points), np.inf)
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    for s in range(0, len(points), chunk):
        p = points[s : s + chunk][:, None, :]
        # projection onto the plane, kept when it falls inside the triangle
        t = np.einsum("ijk,jk->ij", p - a[None], n) / np.where(nn > 0, nn, 1.0)
        q = p - t[..., None] * n[None]
        inside = np.ones(t.shape, dtype=bool)
        for e0, e1 in ((a, b), (b, c), (c, a)):
            inside &= np.einsum("ijk,jk->ij", np.cross(e1 - e0, q - e0[None]), n) >= -1e-18
        d = np.where(inside & (nn > 0), np.abs(t) * np.sqrt(nn), np.inf)
        for e0, e1 in ((a, b), (b, c), (c, a)):
            ed = e1 - e0
            u = np.clip(np.einsum("ijk,jk->ij", p - e0[None], ed) / np.einsum("jk,jk->j", ed, ed), 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(p - (e0[None] + u[..., None] * ed[None]), axis=-1))
        out[s : s + chunk] = d.min(axis=1)
    return out


def _hull_triangles(pieces: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial import ConvexHull

    return np.concatenate([p[ConvexHull(p).simplices] for p in pieces])


def pad_zone_hausdorff(build: Any, rng: np.random.Generator, side: str = "left") -> tuple[float, float]:
    """(print -> collision, collision -> print) largest distances in the pad zone, metres.

    The pad zone is the pad band across the jaw, from the jaw's front surface (the pad face,
    its chamfer bevel or the hook) to 3 mm behind the pad face: the part of the jaw that can
    touch the part. Only the outermost 0.2 mm at the jaw's two side edges is left out.
    """
    jaw = build.left if side == "left" else build.right
    flip = np.array([1.0, -1.0, 1.0]) if side == "right" else np.ones(3)
    spec = build.plan.spec
    xk, yk = jaw.xk, jaw.yk_local
    hk = jaw.hook_local
    z0 = -build.pair.finger_length
    chamfer = spec.tip_size if spec.tip == "chamfer" else 0.0
    hh = jaw.hook_height or 0.0
    lo, hi = z0 + 1e-4, z0 + spec.pad_height - 1e-4
    hw = spec.width / 2.0 - 2e-4
    zone_pieces = [pc.vertices * flip for pc in jaw.pieces if pc.role in ("pad", "hook")]
    tri = (jaw.mesh.vertices * flip)[jaw.mesh.faces]

    def faces_at(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.interp(x, xk, yk)
        front = f + np.maximum(0.0, z0 + chamfer - z) if chamfer > 0.0 else f
        if hk is not None:
            front = np.minimum(front, np.interp(x, xk, hk) + np.maximum(0.0, z - z0 - hh))
        return f, front

    def in_zone(pts: np.ndarray) -> np.ndarray:
        f, front = faces_at(pts[:, 0], pts[:, 2])
        m = (pts[:, 2] >= lo) & (pts[:, 2] <= hi) & (np.abs(pts[:, 0]) <= hw)
        return m & (pts[:, 1] >= front - 1e-6) & (pts[:, 1] <= f + 0.003)

    cent = tri.mean(axis=1)
    near = in_zone(cent)
    samples = [cent[near]]
    for t in tri[near]:
        samples.append(rng.dirichlet([1.0, 1.0, 1.0], 8) @ t)
    pts = np.concatenate(samples)
    pts = pts[in_zone(pts)]
    d1 = float(_point_triangle_distance(pts, _hull_triangles(zone_pieces)).max()) if len(pts) else 0.0
    xs = rng.uniform(-hw, hw, 300)
    zs = rng.uniform(lo, hi, 300)
    surface = np.stack([xs, faces_at(xs, zs)[1], zs], axis=1)
    d2 = float(_point_triangle_distance(surface, tri).max())
    return d1, d2


# ----------------------------------------------------------------------------- sweep


_W: dict[str, Any] = {}


def _init(parts: tuple[Any, ...], hausdorff_every: int) -> None:
    _W["parts"] = parts
    _W["task"] = default_task()
    _W["part_m"] = {p.name: _part_manifold(p) for p in parts}
    _W["every"] = hausdorff_every


def evaluate_point(job: tuple[int, list[float]], only: int | None = None) -> dict[str, Any]:
    """Build one unit-cube point on every part (or only part ``only``) and run the checks."""
    from pydantic import ValidationError

    from rtt.design.freeform.generate import build_pair_full

    index, u = job
    task, parts = _W["task"], _W["parts"]
    row: dict[str, Any] = {"index": index, "u": u, "pairs": []}
    try:
        recipe = recipe_from_unit(np.array(u))
    except ValidationError as exc:
        row.update(stage="schema", code="schema", message=str(exc.errors()[0]["msg"]))
        return row
    row["recipe"] = recipe.model_dump(exclude_none=True)
    rng = np.random.default_rng(index)
    for part in parts if only is None else (parts[only],):
        t0 = time.perf_counter()
        entry: dict[str, Any] = {"part": part.name}
        try:
            b = build_pair_full(recipe, part, task, use_cache=False)
        except InvalidDesign as exc:
            entry.update(ok=False, code=exc.code, message=str(exc), seconds=time.perf_counter() - t0)
            row["pairs"].append(entry)
            continue
        except Exception as exc:  # noqa: BLE001 - anything else is a gate failure
            entry.update(ok=False, code="crash", message=f"{type(exc).__name__}: {exc}")
            row["pairs"].append(entry)
            continue
        entry.update(
            ok=True,
            seconds=time.perf_counter() - t0,
            build_s=b.pair.build_s,
            geometry_hash=b.pair.geometry_hash,
            pieces=[b.pair.left.n_pieces, b.pair.right.n_pieces],
            mass_g=[b.pair.left.mass * 1000.0, b.pair.right.mass * 1000.0],
            convex_rel=max(convex_error(b.left.pieces), convex_error(b.right.pieces)),
            stl_watertight=stl_watertight(b.left.mesh) and stl_watertight(b.right.mesh),
            **seat_checks(b, task, _W["part_m"][part.name]),
        )
        if _W["every"] and index % _W["every"] == 0:
            d1, d2 = pad_zone_hausdorff(b, rng)
            entry.update(hausdorff_print_mm=d1 * 1000.0, hausdorff_collision_mm=d2 * 1000.0)
        row["pairs"].append(entry)
    codes = [p.get("code") for p in row["pairs"] if not p["ok"]]
    row["stage"] = "valid" if not codes else "part"
    row["code"] = codes[0] if codes else None
    return row


def sobol_points(n: int, seed: int = 20261017) -> np.ndarray:
    from scipy.stats import qmc

    pts = qmc.Sobol(d=len(SCORED_PARAMS), scramble=True, seed=seed).random(n)
    return np.minimum(pts, np.nextafter(1.0, 0.0))


def run_sweep(
    n: int, parts: tuple[Any, ...], workers: int = 1, hausdorff_every: int = 20
) -> list[dict[str, Any]]:
    jobs = [(i, [float(v) for v in u]) for i, u in enumerate(sobol_points(n))]
    if workers <= 1:
        _init(parts, hausdorff_every)
        return [evaluate_point(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(parts, hausdorff_every)) as pool:
        return pool.map(evaluate_point, jobs, chunksize=4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [p for r in rows for p in r["pairs"]]
    built = [p for p in pairs if p["ok"]]
    rejected = [p for p in pairs if not p["ok"]]
    secs = [p["seconds"] for p in pairs if "seconds" in p]
    haus = [p for p in built if "hausdorff_print_mm" in p]
    stage = Counter(r["stage"] for r in rows)
    first = Counter(r["code"] for r in rows if r["code"])
    return {
        "recipes": len(rows),
        "schema_rejected": stage["schema"],
        "valid_on_all_parts": stage["valid"],
        "rejected_on_a_part": stage["part"],
        "first_rejection_codes": dict(first.most_common()),
        "pairs_attempted": len(pairs),
        "pairs_built": len(built),
        "pair_rejection_codes": dict(Counter(p["code"] for p in rejected).most_common()),
        "crashes": [p for p in rejected if p["code"] == "crash"][:10],
        "unspecific_rejections": [p for p in rejected if p["code"] not in CODES][:10],
        "max_convex_rel": max((p["convex_rel"] for p in built), default=0.0),
        "stl_not_watertight": sum(not p["stl_watertight"] for p in built),
        "max_touch_gap_mm": max((p["touch"] for p in built), default=0.0) * 1000.0,
        "min_early_gap_mm": min((p["early"] for p in built), default=math.inf) * 1000.0,
        "min_clearance_margin_mm": min((p["clear_margin"] for p in built), default=math.inf) * 1000.0,
        "hausdorff_samples": len(haus),
        "max_hausdorff_print_mm": max((p["hausdorff_print_mm"] for p in haus), default=0.0),
        "max_hausdorff_collision_mm": max((p["hausdorff_collision_mm"] for p in haus), default=0.0),
        "build_s_p50": float(np.percentile(secs, 50)) if secs else 0.0,
        "build_s_p95": float(np.percentile(secs, 95)) if secs else 0.0,
        "build_s_max": float(np.max(secs)) if secs else 0.0,
        "max_pieces": max((max(p["pieces"]) for p in built), default=0),
        "max_mass_g": max((max(p["mass_g"]) for p in built), default=0.0),
    }


def gate_failures(s: dict[str, Any]) -> list[str]:
    """The GV0 sweep sub-checks that fail, in words (empty when the sweep passes)."""
    out = []
    if s["crashes"] or s["unspecific_rejections"]:
        out.append("a build crashed or was rejected without a specific code")
    if s["max_convex_rel"] > CONVEX_RTOL:
        out.append(f"a piece is not convex ({s['max_convex_rel']:.2e})")
    if s["stl_not_watertight"]:
        out.append(f"{s['stl_not_watertight']} print STLs are not watertight")
    if s["max_touch_gap_mm"] > TOUCH_TOL * 1000.0:
        out.append(f"a jaw does not touch the part at its seat ({s['max_touch_gap_mm']:.4f} mm)")
    if s["min_early_gap_mm"] <= 0.0:
        out.append("a jaw touches the part before its seat (the pad cuts into the part)")
    if s["min_clearance_margin_mm"] < -CLEARANCE_TOL * 1000.0:
        out.append(f"a jaw breaks its clearance at the designed pose ({s['min_clearance_margin_mm']:.4f} mm)")
    if max(s["max_hausdorff_print_mm"], s["max_hausdorff_collision_mm"]) > HAUSDORFF_MAX * 1000.0:
        out.append("the pad-zone print differs from the collision pieces by more than 0.05 mm")
    return out


# ----------------------------------------------------------------------------- tests


def test_sobol_sweep_builds_or_rejects_specifically() -> None:
    n = int(os.environ.get("RTT_GV0_SOBOL", "16"))
    workers = int(os.environ.get("RTT_GV0_WORKERS", "1"))
    parts = suite_parts()
    assert len(parts) == 6
    t0 = time.perf_counter()
    rows = run_sweep(n, parts, workers=workers, hausdorff_every=int(os.environ.get("RTT_GV0_HAUSDORFF", "4")))
    summary = summarize(rows)
    summary["wall_s"] = time.perf_counter() - t0
    summary["parts"] = [p.name for p in parts]
    report = os.environ.get("RTT_GV0_REPORT")
    if report:
        Path(report).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=1, default=float) + "\n"
        )
    assert summary["recipes"] == n
    assert gate_failures(summary) == [], summary
    assert summary["pairs_built"] > 0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.floats(0.0, 1.0, exclude_max=True), min_size=9, max_size=9), st.integers(0, 5))
def test_any_recipe_builds_or_is_rejected_with_a_code(u: list[float], k: int) -> None:
    """Hypothesis: any unit-cube point on any training part builds cleanly or is rejected."""
    if "parts" not in _W:
        _init(suite_parts(), 0)
    row = evaluate_point((k, u), only=k)
    if row["stage"] == "schema":
        assert row["message"]
        return
    (pair,) = row["pairs"]
    if not pair["ok"]:
        assert pair["code"] in CODES and pair["message"].startswith(pair["code"])
        return
    assert pair["convex_rel"] <= CONVEX_RTOL
    assert pair["stl_watertight"]
    assert pair["touch"] <= TOUCH_TOL and pair["early"] > 0.0
    assert pair["clear_margin"] >= -CLEARANCE_TOL


def test_checks_catch_a_jaw_that_cuts_into_the_part() -> None:
    """The independent seat check fails when the seat closure is moved past contact."""
    from dataclasses import replace

    from freeform_parts import block
    from rtt.design.freeform.generate import build_pair_full
    from rtt.design.freeform.schema import FingerRecipe

    task = default_task()
    recipe = FingerRecipe(
        pad="conform",
        tip="blunt",
        grip_height_frac=0.5,
        pad_height_mm=20.0,
        finger_width_mm=36.0,
        pad_depth_mm=5.0,
        wall_draft_deg=30.0,
        clearance_mm=0.4,
    )
    b = build_pair_full(recipe, block(), task)
    part_m = _part_manifold(block())
    ok = seat_checks(b, task, part_m)
    assert ok["touch"] <= TOUCH_TOL and ok["early"] > 0.0 and ok["clear_margin"] >= -CLEARANCE_TOL

    class Late:
        pair = replace(b.pair, q_seat=b.pair.q_seat + 0.0005)  # 0.5 mm past contact
        plan = b.plan

    bad = seat_checks(Late, task, part_m)
    assert bad["early"] <= 0.0
    d1, d2 = pad_zone_hausdorff(b, np.random.default_rng(0))
    assert max(d1, d2) <= HAUSDORFF_MAX
    from rtt.design.freeform.export import JawMesh

    moved = JawMesh(b.left.mesh.vertices + np.array([0.0, 1e-4, 0.0]), b.left.mesh.faces)  # 0.1 mm back
    shifted = replace(b, left=replace(b.left, mesh=moved))
    assert pad_zone_hausdorff(shifted, np.random.default_rng(0))[1] > HAUSDORFF_MAX


def test_point_triangle_distance() -> None:
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pts = np.array([[0.2, 0.2, 0.5], [2.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [0.5, 0.5, 0.0]])
    d = _point_triangle_distance(pts, tri)
    assert d == pytest.approx([0.5, 1.0, math.sqrt(2.0), 0.0], abs=1e-12)
