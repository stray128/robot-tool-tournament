"""Exact-piece test parts for the freeform v2 generator tests.

These stand in for the procedural training suite (spec section 8) until the parts library
lands: the same six families at the spec's dimensions, every piece an exact convex prism.
``mvp_tee`` is the 3/8-inch style tee the 2.5D prototype simulated (its predictor table is
the reference for the predictor test). Extra parts exercise single validators.

Frames follow :class:`rtt.parts.types.PartSpec`: metres, z up, bounding box centred on the
origin, the closing axis along y (left jaw at +y).
"""

from __future__ import annotations

import functools
import math

import manifold3d as m3d
import numpy as np
from scipy.spatial import ConvexHull

from rtt.design.freeform.export import mass_properties
from rtt.parts.types import Feature, Material, PartSpec

PVC = Material("pvc_white", 1400.0, (0.93, 0.93, 0.9), 0.0, 0.6)
ALU = Material("anodised_aluminium", 2700.0, (0.62, 0.64, 0.68), 0.6, 0.35)
ZINC = Material("zinc_steel", 7850.0, (0.78, 0.79, 0.8), 1.0, 0.35)
BRASS = Material("brass", 8500.0, (0.78, 0.62, 0.3), 1.0, 0.3)
STEEL = Material("stamped_steel", 7850.0, (0.55, 0.57, 0.6), 1.0, 0.45)
GROUND = Material("ground_steel", 7850.0, (0.7, 0.71, 0.73), 1.0, 0.25)


def _hull_faces(v: np.ndarray) -> np.ndarray:
    hull = ConvexHull(v)
    faces = hull.simplices.copy()
    c = v.mean(axis=0)
    for i, f in enumerate(faces):
        a, b, d = v[f]
        if np.dot(np.cross(b - a, d - a), a - c) < 0:
            faces[i] = f[::-1]
    return faces.astype(np.int64)


def make_part(
    name: str,
    family: str,
    pieces: list[np.ndarray],
    material: Material,
    *,
    yaw_symmetry: int = 1,
    features: tuple[Feature, ...] = (),
    register_axes: tuple[str, ...] | None = None,
    shift_features: bool = True,
) -> PartSpec:
    """A PartSpec from convex pieces given in table heights; centres the bounding box."""
    pts = np.concatenate(pieces)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    shift = -(lo + hi) / 2.0
    pieces = [np.ascontiguousarray(p + shift) for p in pieces]
    hulls = [m3d.Manifold.hull_points(p * 1000.0) for p in pieces]
    union = m3d.Manifold.batch_boolean(hulls, m3d.OpType.Add).to_mesh64()
    verts = np.asarray(union.vert_properties)[:, :3] / 1000.0
    faces = np.asarray(union.tri_verts, dtype=np.int64)
    vol, com, inertia = mass_properties(verts, faces, 1.0)
    if shift_features:
        features = tuple(
            Feature(f.kind, f.name, tuple(np.asarray(f.origin) + shift), f.axis, f.radius, f.depth)
            for f in features
        )
    kwargs = {}
    if register_axes is not None:
        kwargs["register_axes"] = register_axes
    elif yaw_symmetry == 0:
        kwargs["register_axes"] = ("x", "y", "tilt")
    return PartSpec(
        name=name,
        family=family,
        pieces=tuple(pieces),
        faces=tuple(_hull_faces(p) for p in pieces),
        volume=vol,
        com=tuple(com),
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
        features=features,
        decomposition="exact",
        **kwargs,
    )


def cylinder(axis: str, centre: tuple[float, float, float], r: float, h: float, n: int = 32) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    pts = []
    for s in (-h / 2, h / 2):
        for a, b in ring:
            pts.append({"x": (s, a, b), "y": (a, s, b), "z": (a, b, s)}[axis])
    return np.asarray(pts) + np.asarray(centre)


def box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float
    )


def prism_z(poly_xy: np.ndarray, z0: float, z1: float) -> np.ndarray:
    return np.concatenate([np.column_stack([poly_xy, np.full(len(poly_xy), z)]) for z in (z0, z1)])


@functools.cache
def mvp_tee() -> PartSpec:
    """The 2.5D prototype's tee: run along x (50 mm), branch toward +y, beaded ends, 32-gons."""
    r, rb, run, branch, band = 0.0105, 0.0115, 0.050, 0.024, 0.004
    zc = rb
    pieces = [
        cylinder("x", (0, 0, zc), r, run),
        cylinder("y", (0, branch / 2, zc), r, branch),
        cylinder("x", (-(run / 2 - band / 2), 0, zc), rb, band),
        cylinder("x", (run / 2 - band / 2, 0, zc), rb, band),
        cylinder("y", (0, branch - band / 2, zc), rb, band),
    ]
    return make_part("pipe_tee", "pipe_tee", pieces, PVC)


@functools.cache
def block() -> PartSpec:
    """30 (x) x 40 (y) x 60 (z) mm machined block, no chamfer."""
    return make_part("block", "block", [box((-0.015, -0.02, 0.0), (0.015, 0.02, 0.06))], ALU, yaw_symmetry=2)


@functools.cache
def hex_standoff() -> PartSpec:
    """17 mm across flats (flats facing the jaws), 30 mm tall, standing."""
    rc = 0.0085 / math.cos(math.radians(30))
    t = np.radians(np.arange(6) * 60.0)
    poly = np.stack([rc * np.cos(t), rc * np.sin(t)], 1)
    return make_part("hex_standoff", "hex_standoff", [prism_z(poly, 0.0, 0.030)], ZINC, yaw_symmetry=6)


@functools.cache
def flanged_bushing() -> PartSpec:
    """Ø20 x 30 mm bushing standing on its body, with a Ø36 x 5 mm flange on top."""
    pieces = [cylinder("z", (0, 0, 0.0125), 0.010, 0.025), cylinder("z", (0, 0, 0.0275), 0.018, 0.005)]
    flange = Feature("flange", "flange", (0.0, 0.0, 0.025), (0.0, 0.0, -1.0), 0.018, 0.005)
    return make_part("flanged_bushing", "flanged_bushing", pieces, BRASS, yaw_symmetry=0, features=(flange,))


@functools.cache
def l_bracket() -> PartSpec:
    """40 x 40 mm angle, 4 mm wall, 30 mm wide, lying flat on one leg (the other leg stands up
    at -x); the jaws close across its 30 mm width."""
    pieces = [
        box((-0.020, -0.015, 0.0), (0.020, 0.015, 0.004)),
        box((-0.020, -0.015, 0.004), (-0.016, 0.015, 0.040)),
    ]
    return make_part("l_bracket", "l_bracket", pieces, STEEL)


@functools.cache
def steel_slug() -> PartSpec:
    """Ø40 x 50 mm ground steel slug, standing (490 g)."""
    return make_part(
        "steel_slug", "steel_slug", [cylinder("z", (0, 0, 0.025), 0.020, 0.050)], GROUND, yaw_symmetry=0
    )


def training_parts() -> tuple[PartSpec, ...]:
    return (mvp_tee(), block(), hex_standoff(), flanged_bushing(), l_bracket(), steel_slug())


# ----------------------------------------------------------------------------- test-only parts


@functools.cache
def thin_plate() -> PartSpec:
    """A 2 mm plate: nothing to grip 3 mm above the table."""
    return make_part(
        "thin_plate", "plate", [box((-0.02, -0.02, 0.0), (0.02, 0.02, 0.002))], ALU, yaw_symmetry=2
    )


@functools.cache
def thin_fin() -> PartSpec:
    """A 20 x 3 x 30 mm fin standing across the jaws: deep straight walls wrap past it."""
    return make_part(
        "thin_fin", "plate", [box((-0.01, -0.0015, 0.0), (0.01, 0.0015, 0.03))], ALU, yaw_symmetry=2
    )


@functools.cache
def tall_post() -> PartSpec:
    """Ø20 x 120 mm post: the palm-clearance rule makes fingers too long when gripped low."""
    return make_part("tall_post", "post", [cylinder("z", (0, 0, 0.06), 0.010, 0.120)], ALU, yaw_symmetry=0)


@functools.cache
def wide_block() -> PartSpec:
    """80 mm across the closing axis: barely fits the 84 mm opening."""
    return make_part(
        "wide_block", "block", [box((-0.015, -0.04, 0.0), (0.015, 0.04, 0.03))], ALU, yaw_symmetry=2
    )


@functools.cache
def socket_hub() -> PartSpec:
    """Ø24 x 30 mm hub lying along y with a Ø12 x 10 mm blind socket facing the left jaw."""
    r_out, r_in, n = 0.012, 0.006, 16
    zc = r_out
    pieces = [cylinder("y", (0.0, -0.005, zc), r_out, 0.020, n)]
    t = 2 * np.pi * np.arange(n + 1) / n
    for k in range(n):
        pts = []
        for y in (0.005, 0.015):
            for a in (t[k], t[k + 1]):
                for r in (r_in, r_out):
                    pts.append((r * np.cos(a), y, zc + r * np.sin(a)))
        pieces.append(np.array(pts))
    sock = Feature("socket", "bore", (0.0, 0.015, zc), (0.0, 1.0, 0.0), r_in * math.cos(math.pi / n), 0.010)
    return make_part("socket_hub", "hub", pieces, PVC, yaw_symmetry=1, features=(sock,))


@functools.cache
def tall_block() -> PartSpec:
    """30 x 40 x 95 mm block: gripped low, the palm-clearance rule makes a long, heavy finger."""
    return make_part(
        "tall_block", "block", [box((-0.015, -0.02, 0.0), (0.015, 0.02, 0.095))], ALU, yaw_symmetry=2
    )
