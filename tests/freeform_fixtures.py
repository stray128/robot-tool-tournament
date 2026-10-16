"""Test fixtures for the v2 simulator: six exact-piece training parts and a small jaw builder.

The real part library (:mod:`rtt.parts`) and the real jaw generator
(:func:`rtt.design.freeform.build_pair`) are other tracks' work. Until they land, the
simulator's v2 tests and its gates (GV1, GV2) run on these stand-ins, which follow the frozen
contracts exactly:

* :func:`training_parts`: the six ``train_v2`` parts (tee, block, hex standoff, flanged
  bushing, L-bracket, steel slug) as :class:`~rtt.parts.types.PartSpec` objects made of exact
  convex pieces, posed as the spec describes (y is the closing axis, the left jaw at +y).
* :func:`build_fixture_pair`: a port of the 2.5D prototype generator (part outline from the
  convex pieces by halfspace maths, pad face per strategy, simplification to at most 12
  segments that never move into the part, one 8-vertex prism per segment, seat-aligned jaws,
  the right jaw built on the mirrored part and mirrored once). It produces a
  :class:`~rtt.design.freeform.types.FingerPair` in the contract's jaw frames. It has no
  lofted body, windows, tips or locator: a plain slab carries the pad up to the root plane.

These are fixtures, not the product: dimensions and materials are the spec's, the details
(bores, chamfers, the exact stable pose) are not modelled.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import manifold3d as m3d
import numpy as np
import trimesh

from rtt.design.freeform.types import (
    CARRIAGE_INNER_Y,
    PLA_DENSITY,
    CaptureReport,
    FingerPair,
    JawGeometry,
)
from rtt.parts.types import Material, PartSpec
from rtt.sim.fingers_v2 import pieces_hash
from rtt.sim.task import TaskConfig

MM = 1e-3
STROKE_CAP = 0.017  # m, pad-depth cap of the configurator imprint (spec section 5)

PVC = Material("pvc_white", 1400.0, (0.92, 0.92, 0.90), 0.0, 0.6)
ALUMINIUM = Material("anodised_aluminium", 2700.0, (0.55, 0.57, 0.60), 0.8, 0.35)
ZINC_STEEL = Material("zinc_plated_steel", 7850.0, (0.75, 0.76, 0.78), 1.0, 0.3)
BRASS = Material("brass", 8500.0, (0.78, 0.62, 0.30), 1.0, 0.3)
STAMPED_STEEL = Material("stamped_steel", 7850.0, (0.45, 0.46, 0.48), 1.0, 0.45)
GROUND_STEEL = Material("ground_steel", 7850.0, (0.70, 0.71, 0.73), 1.0, 0.2)


# --------------------------------------------------------------------------- parts


def _prism_ring(n: int, r: float, z0: float, z1: float, cx: float = 0.0, cy: float = 0.0, phase=0.0):
    t = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
    return np.array([[x, y, z] for z in (z0, z1) for x, y in ring])


def _cyl(axis: str, c, r: float, h: float, n: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    pts = []
    for s in (-h / 2, h / 2):
        for a, b in ring:
            pts.append({"x": (s, a, b), "y": (a, s, b), "z": (a, b, s)}[axis])
    return np.asarray(pts) + np.asarray(c)


def _box(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _union(pieces: list[np.ndarray]) -> trimesh.Trimesh:
    mans = []
    for p in pieces:
        h = trimesh.convex.convex_hull(p)
        mans.append(
            m3d.Manifold(m3d.Mesh(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.uint32)))
        )
    u = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    mm = u.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def make_part(
    name: str,
    family: str,
    pieces: list[np.ndarray],
    material: Material,
    yaw_symmetry: int = 1,
    register_axes: tuple[str, ...] = ("x", "y", "yaw", "tilt"),
) -> PartSpec:
    """A PartSpec from convex vertex sets (any frame): centred on its bounding box."""
    allp = np.concatenate(pieces)
    shift = -(allp.min(axis=0) + allp.max(axis=0)) / 2.0
    hulls = [trimesh.convex.convex_hull(np.asarray(p) + shift) for p in pieces]
    solid = _union([np.asarray(h.vertices) for h in hulls])
    solid.density = 1.0
    inertia = np.asarray(solid.moment_inertia)
    return PartSpec(
        name=name,
        family=family,
        pieces=tuple(np.ascontiguousarray(h.vertices, dtype=np.float64) for h in hulls),
        faces=tuple(np.ascontiguousarray(h.faces, dtype=np.int64) for h in hulls),
        volume=float(solid.volume),
        com=tuple(float(v) for v in solid.center_mass),
        unit_inertia=(
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ),
        material=material,
        yaw_symmetry=yaw_symmetry,
        register_axes=register_axes,
        decomposition="exact",
    )


def pipe_tee() -> PartSpec:
    """1/2-inch slip tee lying flat: run along x, branch toward +y (the left jaw), beaded ends."""
    r, rb, run, branch, band = 0.0105, 0.0115, 0.050, 0.024, 0.004
    zc = rb
    pieces = [
        _cyl("x", (0, 0, zc), r, run),
        _cyl("y", (0, branch / 2, zc), r, branch),
        _cyl("x", (-(run / 2 - band / 2), 0, zc), rb, band),
        _cyl("x", (run / 2 - band / 2, 0, zc), rb, band),
        _cyl("y", (0, branch - band / 2, zc), rb, band),
    ]
    return make_part("pipe_tee", "pipe_tee", pieces, PVC)


def block() -> PartSpec:
    """30 (across the jaw) x 40 (closing axis) x 60 mm block, no chamfer."""
    return make_part("block_30x40x60", "block", [_box((0, 0, 0), (0.030, 0.040, 0.060))], ALUMINIUM, 2)


def hex_standoff() -> PartSpec:
    """Hex standoff standing on end, 17 mm across flats (flats facing the jaws), 40 mm tall."""
    r = 0.0085 / math.cos(math.pi / 6)
    return make_part(
        "hex_standoff_17", "hex_standoff", [_prism_ring(6, r, 0.0, 0.040)], ZINC_STEEL, yaw_symmetry=6
    )


def flanged_bushing() -> PartSpec:
    """Flanged bushing standing: a 20 x 30 mm body under a 36 x 5 mm flange."""
    pieces = [_prism_ring(32, 0.010, 0.0, 0.030), _prism_ring(32, 0.018, 0.030, 0.035)]
    return make_part("flanged_bushing_20", "flanged_bushing", pieces, BRASS, 0, ("x", "y", "tilt"))


def l_bracket() -> PartSpec:
    """L-bracket 40 x 40 x 30 mm, 4 mm wall, lying on its L-shaped side (top view is an L)."""
    pieces = [
        _box((-0.020, 0.016, 0.0), (0.020, 0.020, 0.030)),  # leg along x, the left jaw's side
        _box((-0.020, -0.020, 0.0), (-0.016, 0.016, 0.030)),  # leg along y
    ]
    return make_part("l_bracket_40", "l_bracket", pieces, STAMPED_STEEL)


def steel_slug() -> PartSpec:
    """Ground steel slug 40 x 50 mm, standing (about 0.49 kg)."""
    return make_part(
        "steel_slug_40",
        "steel_slug",
        [_prism_ring(32, 0.020, 0.0, 0.050)],
        GROUND_STEEL,
        0,
        ("x", "y", "tilt"),
    )


def training_parts() -> tuple[PartSpec, ...]:
    """The six train_v2 parts, in suite order."""
    return (pipe_tee(), block(), hex_standoff(), flanged_bushing(), l_bracket(), steel_slug())


# --------------------------------------------------------------------------- jaw builder


@dataclass(frozen=True)
class FixtureRecipe:
    """The scored v2 parameters the fixture builder understands (blunt tips only)."""

    pad: str = "conform"  # flat | conform | v_funnel
    clearance_mm: float = 0.4
    pad_depth_mm: float = 10.0  # None-like 0 for flat
    wall_draft_deg: float = 30.0
    sweep_mm: float = 0.0  # baselines only (the swept imprint)
    finger_width_mm: float = 40.0
    pad_height_mm: float = 20.0
    grip_height_frac: float = 0.5
    max_seg: int = 12
    body_thickness_mm: float = 8.0
    configurator_depth: bool = False  # depth = min(40 % of the band half-extent, stroke cap), >= 4 mm
    # Contact-quality options (experiments for the generator, see _strips): strips narrower
    # than min_strip_mm are absorbed by their deeper neighbour (a wall becomes a strip's side
    # face); strips wider than max_strip_mm are split into equal strips.
    min_strip_mm: float = 0.0
    max_strip_mm: float = 5.0
    # Outline sampling: x grid step and z rows of the pad band (the prototype used 0.1 mm, 25).
    grid_mm: float = 0.1
    z_rows: int = 25

    def design_id(self) -> str:
        payload = json.dumps({"fixture": "rtt-tests-2.5d", **asdict(self)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def flat(**kw) -> FixtureRecipe:
    return FixtureRecipe(pad="flat", clearance_mm=0.0, pad_depth_mm=0.0, wall_draft_deg=0.0, **kw)


def imprint_configurator(**kw) -> FixtureRecipe:
    base = {
        "pad": "conform", "clearance_mm": 0.4, "wall_draft_deg": 0.0, "finger_width_mm": 40.0, "pad_height_mm": 20.0,
        "grip_height_frac": 0.5, "configurator_depth": True,
    }  # fmt: skip
    base.update(kw)
    return FixtureRecipe(**base)


def imprint_swept(**kw) -> FixtureRecipe:
    return imprint_configurator(sweep_mm=2.0, **kw)


def conform(**kw) -> FixtureRecipe:
    base = {
        "pad": "conform",
        "clearance_mm": 0.4,
        "pad_depth_mm": 4.0,
        "wall_draft_deg": 30.0,
        "finger_width_mm": 30.0,
    }
    base.update(kw)
    return FixtureRecipe(**base)


def v_funnel(**kw) -> FixtureRecipe:
    base = {
        "pad": "v_funnel",
        "clearance_mm": 0.2,
        "pad_depth_mm": 6.0,
        "wall_draft_deg": 45.0,
        "finger_width_mm": 30.0,
    }
    base.update(kw)
    return FixtureRecipe(**base)


def _halfspaces(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = trimesh.convex.convex_hull(verts)
    n = np.asarray(h.face_normals)
    b = np.einsum("ij,ij->i", n, np.asarray(h.triangles)[:, 0, :])
    key = np.round(np.c_[n, b], 9)
    _, idx = np.unique(key, axis=0, return_index=True)
    return n[idx], b[idx]


def profile(
    part: PartSpec, side: float, xs: np.ndarray, z_lo: float, z_hi: float, nz: int = 25
) -> np.ndarray:
    """max over z in [z_lo, z_hi] (table frame) of ``side * y`` of the part at each x; -inf where empty."""
    zc = part.size[2] / 2
    zs = np.linspace(z_lo, z_hi, nz) - zc
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    best = np.full(gx.shape, -np.inf)
    for verts in part.pieces:
        n, b = _halfspaces(verts * np.array([1.0, side, 1.0]))
        ny = n[:, 1]
        rhs = b[None, None, :] - n[None, None, :, 0] * gx[..., None] - n[None, None, :, 2] * gz[..., None]
        up, lo = ny > 1e-9, ny < -1e-9
        zero = ~(up | lo)
        ymax = np.min(rhs[..., up] / ny[up], axis=-1) if up.any() else np.full(gx.shape, np.inf)
        ymin = np.max(rhs[..., lo] / ny[lo], axis=-1) if lo.any() else np.full(gx.shape, -np.inf)
        ok = ymax >= ymin
        if zero.any():
            ok &= np.all(rhs[..., zero] >= -1e-12, axis=-1)
        best = np.where(ok, np.maximum(best, ymax), best)
    return best.max(axis=1)


def _dilate_x(p: np.ndarray, dx: float, r: float) -> np.ndarray:
    k = round(r / dx)
    out = p.copy()
    for s in range(1, k + 1):
        out[s:] = np.maximum(out[s:], p[:-s])
        out[:-s] = np.maximum(out[:-s], p[s:])
    return out


def _pad_face(p: np.ndarray, xs: np.ndarray, r: FixtureRecipe, depth: float) -> np.ndarray:
    dx = float(xs[1] - xs[0])
    if r.pad == "flat":
        return np.full_like(xs, np.max(p[np.isfinite(p)]))
    c = r.clearance_mm * MM
    h = _dilate_x(p, dx, c + r.sweep_mm * MM) + c
    finite = np.isfinite(h)
    top = float(h[finite].max())
    slope = math.inf if r.wall_draft_deg <= 0 else 1.0 / math.tan(math.radians(r.wall_draft_deg))
    hx, hv = xs[finite], h[finite]
    if r.pad == "conform":
        env = (
            np.where(finite, h, -np.inf)
            if math.isinf(slope)
            else np.max(hv[None, :] - slope * np.abs(xs[:, None] - hx[None, :]), axis=1)
        )
        return np.maximum(env, top - depth)
    if r.pad == "v_funnel":
        y0 = min(float(np.max(hv + slope * np.abs(hx))), top + depth)
        return np.maximum(y0 - slope * np.abs(xs), y0 - depth)
    raise ValueError(f"unknown pad {r.pad!r}")


def _simplify_upper(xs: np.ndarray, f: np.ndarray, max_seg: int) -> tuple[np.ndarray, np.ndarray]:
    def rdp(i0: int, i1: int, tol: float, out: list[int]) -> None:
        seg = np.interp(xs[i0 : i1 + 1], [xs[i0], xs[i1]], [f[i0], f[i1]])
        err = np.abs(f[i0 : i1 + 1] - seg)
        k = int(np.argmax(err))
        if err[k] > tol and 0 < k < i1 - i0:
            rdp(i0, i0 + k, tol, out)
            rdp(i0 + k, i1, tol, out)
        else:
            out.append(i1)

    tol = 2e-6
    while True:
        out = [0]
        rdp(0, len(xs) - 1, tol, out)
        if len(out) - 1 <= max_seg:
            break
        tol *= 1.6
    idx = np.array(out)
    yk = f[idx].copy()
    for _ in range(3):  # lift segments that dip below f (never move into the part)
        lift = np.zeros_like(yk)
        for j in range(len(idx) - 1):
            a, b = idx[j], idx[j + 1]
            line = np.interp(xs[a : b + 1], [xs[a], xs[b]], [yk[j], yk[j + 1]])
            v = float(np.max(f[a : b + 1] - line))
            if v > 0:
                lift[j] = max(lift[j], v)
                lift[j + 1] = max(lift[j + 1], v)
        if not lift.any():
            break
        yk += lift
    return xs[idx], yk


def _strips(
    xk: np.ndarray, yk: np.ndarray, min_w: float, max_w: float
) -> list[tuple[float, float, float, float]]:
    """Pad strips ``(x0, x1, front0, front1)`` from the simplified face knots.

    A strip narrower than ``min_w`` is dropped and its deeper neighbour (larger front y, farther
    from the part) is extended flat across it, so the face only ever moves back. A strip wider
    than ``max_w`` is split into equal strips along the same line.
    """
    strips = [(float(xk[j]), float(xk[j + 1]), float(yk[j]), float(yk[j + 1])) for j in range(len(xk) - 1)]
    if min_w > 0:
        out: list[list[float]] = []
        i = 0
        while i < len(strips):
            a, b, fa, fb = strips[i]
            if b - a < min_w and len(strips) > 1:
                left = out[-1] if out else None
                right = list(strips[i + 1]) if i + 1 < len(strips) else None
                deep_left = left is not None and (
                    right is None or max(left[2], left[3]) >= max(right[2], right[3])
                )
                if deep_left:
                    left[1], left[3] = b, max(left[3], fa, fb)
                    i += 1
                    continue
                right[0], right[2] = a, max(right[2], fa, fb)
                strips[i + 1] = tuple(right)
                i += 1
                continue
            out.append([a, b, fa, fb])
            i += 1
        strips = [tuple(v) for v in out]
    if max_w > 0:
        split = []
        for a, b, fa, fb in strips:
            n = max(1, math.ceil((b - a) / max_w - 1e-9))
            xs = np.linspace(a, b, n + 1)
            fs = np.linspace(fa, fb, n + 1)
            split.extend((xs[k], xs[k + 1], fs[k], fs[k + 1]) for k in range(n))
        strips = split
    return strips


def _prism(x0, x1, y0f, y1f, yb, z0, z1) -> np.ndarray:
    quad = ((x0, y0f), (x1, y1f), (x1, yb), (x0, yb))
    return np.array([[x, y, z] for z in (z0, z1) for (x, y) in quad], dtype=np.float64)


def _mass_props(pieces: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass (kg), COM and inertia6 about the COM of non-overlapping convex PLA pieces."""
    masses, coms, tensors = [], [], []
    for p in pieces:
        h = trimesh.convex.convex_hull(p)
        h.density = PLA_DENSITY
        masses.append(float(h.mass))
        coms.append(np.asarray(h.center_mass))
        tensors.append(np.asarray(h.moment_inertia))
    total = float(sum(masses))
    com = sum(m * c for m, c in zip(masses, coms, strict=True)) / total
    inertia = np.zeros((3, 3))
    for m, c, i_c in zip(masses, coms, tensors, strict=True):
        d = c - com
        inertia += i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    i6 = np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])
    return total, com, i6


@dataclass
class _SideJaw:
    pieces_local: list[np.ndarray]  # side-local part frame (as a left jaw), before the y0 shift
    gap: float
    slab_front: float


def _side_jaw(part: PartSpec, side: float, r: FixtureRecipe, tip: float, length: float):
    w = r.finger_width_mm * MM
    dx = r.grid_mm * MM
    xs = np.arange(-w / 2, w / 2 + dx / 2, dx)
    pad_h = r.pad_height_mm * MM
    p = profile(part, side, xs, tip, tip + pad_h, nz=r.z_rows)
    if not np.isfinite(p).any():
        raise ValueError(
            f"the part outline in the pad band is empty on the {'left' if side > 0 else 'right'} jaw"
        )
    if r.configurator_depth:
        both = [profile(part, s, xs, tip, tip + pad_h, nz=r.z_rows) for s in (1.0, -1.0)]
        half = 0.5 * sum(float(b[np.isfinite(b)].max()) for b in both)
        depth = max(4 * MM, min(0.4 * half, STROKE_CAP))
    else:
        depth = r.pad_depth_mm * MM
    f = _pad_face(p, xs, r, depth)
    xk, yk = _simplify_upper(xs, f, r.max_seg)
    finite = np.isfinite(p)
    gap = float(np.min(np.interp(xs, xk, yk)[finite] - p[finite]))
    # everything above the pad band stays >= 2 mm behind the part's outline there
    h = part.size[2]
    above = profile(part, side, np.linspace(-w / 2, w / 2, 201), tip, max(tip + 1e-4, h), nz=41)
    above_max = float(above[np.isfinite(above)].max()) if np.isfinite(above).any() else -np.inf
    y_back = max(float(yk.max()) + 3 * MM, above_max + 2 * MM)
    z0, z1 = -length, -length + pad_h
    strips = _strips(xk, yk, r.min_strip_mm * MM, r.max_strip_mm * MM)
    pieces = [_prism(a, b, fa, fb, y_back, z0, z1) for a, b, fa, fb in strips]
    pieces.append(_prism(-w / 2, w / 2, y_back, y_back, y_back + r.body_thickness_mm * MM, -length, 0.0))
    return _SideJaw(pieces, gap, y_back)


def _jaw_geometry(pieces: list[np.ndarray]) -> JawGeometry:
    rounded = [np.round(p / 1e-7) * 1e-7 for p in pieces]
    mass, com, i6 = _mass_props(rounded)
    sha = pieces_hash(tuple(rounded))
    return JawGeometry(pieces=tuple(rounded), mass=mass, com=com, inertia6=i6, print_mesh_sha=sha)


def build_fixture_pair(design: FixtureRecipe, part: PartSpec, task: TaskConfig) -> FingerPair:
    """Compile ``design`` against ``part`` into a seat-aligned pair in the contract's jaw frames."""
    r = design
    h = part.size[2]
    pad_h = r.pad_height_mm * MM
    tip = max(0.003, r.grip_height_frac * h - pad_h / 2)
    palm_clear = task.controller.palm_clearance
    length = max(h + palm_clear - tip, pad_h + 15 * MM)
    ohg = task.gripper.open_half_gap
    jaws = [_side_jaw(part, s, r, tip, length) for s in (1.0, -1.0)]
    q_seat = max(ohg + j.gap - j.slab_front + CARRIAGE_INNER_Y for j in jaws)
    if not 0.002 <= q_seat <= task.gripper.stroke - 0.002:
        raise ValueError(
            f"q_seat {q_seat * 1000:.1f} mm is outside [2, {task.gripper.stroke * 1000 - 2:.0f}] mm"
        )
    out = []
    for j, side in zip(jaws, (1.0, -1.0), strict=True):
        y0 = ohg + j.gap - q_seat
        local = [pc - np.array([0.0, y0, 0.0]) for pc in j.pieces_local]
        if side < 0:  # the right jaw: built on the mirrored part, mirrored once
            local = [pc * np.array([1.0, -1.0, 1.0]) for pc in local]
        out.append(_jaw_geometry(local))
    left, right = out
    return FingerPair(
        left=left,
        right=right,
        tip_height=tip,
        finger_length=length,
        q_seat=q_seat,
        design_id=design.design_id(),
        part_id=part.part_id,
        geometry_hash=pieces_hash(left.pieces, right.pieces),
        capture=CaptureReport(no_jam_mm=0.0, centring_mm=0.0, mu=task.physics.friction),
        build_s=0.0,
        auto_repairs=("test fixture builder",),
    )
