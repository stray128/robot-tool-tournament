"""Part outlines along the closing axis, computed exactly from the part's convex pieces.

Everything here works in the *left-jaw convention*: the jaw approaches from +y and the
outline is the part's largest y. The right jaw uses the same functions on the part mirrored
across the closing axis (:meth:`PartGeometry.mirrored`), so one code path serves both jaws.

Heights are table heights (``z = 0`` on the table); x and y are the part's object frame.

The pad outline is a *cell maximum*: the jaw's x axis is cut into cells of width ``dx``
(0.1 mm) centred on the grid points, and each cell holds the largest y the part reaches
anywhere in that cell over the pad's height band. Per z row that maximum is exact: it is
attained either on a cell boundary (evaluated from the pieces' halfspaces) or at a vertex of
the row's cross-section that falls inside the cell (edge-plane intersections). Rows are 25
evenly spaced heights plus every piece-vertex height inside the band, which makes the band
maximum exact for prisms along any axis. Any face built at or beyond the cell outline plus
its clearance, and interpolated linearly between grid points, therefore never cuts into the
part. No ray casting and no spatial index are used, and the same code serves procedural
parts and decomposed scans: the pads conform to exactly what the physics engine collides.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_NEG = -np.inf
_EPS = 1e-12


@dataclass(frozen=True, eq=False)
class PieceH:
    """One convex piece: halfspaces ``normals @ p <= offsets``, vertices and hull edges."""

    normals: np.ndarray
    offsets: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray


def piece_halfspaces(vertices: np.ndarray) -> PieceH:
    """Halfspace, vertex and edge representation of the convex hull of ``vertices``."""
    v = np.asarray(vertices, dtype=np.float64)
    hull = ConvexHull(v)
    eq = hull.equations
    key = np.round(eq, 12)
    _, idx = np.unique(key, axis=0, return_index=True)
    eq = eq[np.sort(idx)]
    normals = np.ascontiguousarray(eq[:, :3])
    offsets = np.ascontiguousarray(-eq[:, 3])
    simplices = hull.simplices
    pairs = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    used = np.unique(pairs)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return PieceH(normals, offsets, np.ascontiguousarray(v[used]), remap[pairs])


@dataclass(frozen=True, eq=False)
class PartGeometry:
    """A part's convex pieces in table heights, ready for outline queries."""

    pieces: tuple[PieceH, ...]
    height: float
    size: tuple[float, float, float]
    vertices: np.ndarray  # all vertices, table heights
    edge_a: np.ndarray  # all edges' first endpoints
    edge_b: np.ndarray  # all edges' second endpoints

    @staticmethod
    def from_pieces(pieces: tuple[np.ndarray, ...], z_shift: float) -> PartGeometry:
        shift = np.array([0.0, 0.0, z_shift])
        hs = tuple(piece_halfspaces(np.asarray(p, dtype=np.float64) + shift) for p in pieces)
        return PartGeometry._assemble(hs)

    @staticmethod
    def _assemble(hs: tuple[PieceH, ...]) -> PartGeometry:
        verts = np.concatenate([h.vertices for h in hs])
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        ea = np.concatenate([h.vertices[h.edges[:, 0]] for h in hs])
        eb = np.concatenate([h.vertices[h.edges[:, 1]] for h in hs])
        size = tuple(float(s) for s in hi - lo)
        return PartGeometry(hs, float(hi[2]), size, verts, ea, eb)  # type: ignore[arg-type]

    def mirrored(self) -> PartGeometry:
        """The part mirrored across the closing axis (y -> -y)."""
        m = np.array([1.0, -1.0, 1.0])
        hs = tuple(
            PieceH(h.normals * m, h.offsets.copy(), h.vertices * m, h.edges.copy()) for h in self.pieces
        )
        return PartGeometry._assemble(hs)


@functools.lru_cache(maxsize=64)
def _cached_geometry(
    key: str, pieces: tuple[bytes, ...], shapes: tuple[tuple[int, int], ...], z_shift: float
):
    arrays = tuple(np.frombuffer(b, dtype=np.float64).reshape(s) for b, s in zip(pieces, shapes, strict=True))
    g = PartGeometry.from_pieces(arrays, z_shift)
    return g, g.mirrored()


def part_geometry(part: object) -> tuple[PartGeometry, PartGeometry]:
    """(as-is, mirrored) outline geometry of a part (``PartSpec`` or ``MeshObject``), cached."""
    pieces = tuple(np.ascontiguousarray(p, dtype=np.float64) for p in part.pieces)  # type: ignore[attr-defined]
    pts = np.concatenate(pieces)
    z_shift = -float(pts[:, 2].min())  # object frame -> table heights (the part rests on z = 0)
    key = str(getattr(part, "geometry_sha256", ""))
    return _cached_geometry(
        key, tuple(p.tobytes() for p in pieces), tuple(p.shape for p in pieces), round(z_shift, 15)
    )


# ----------------------------------------------------------------------------- queries


def envelope(g: PartGeometry, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """``(len(xs), len(zs))``: the part's largest y at each (x, z), ``-inf`` where it is absent."""
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)
    best = np.full((len(xs), len(zs)), _NEG)
    for h in g.pieces:
        lo, hi = h.vertices.min(axis=0), h.vertices.max(axis=0)
        ix = np.flatnonzero((xs >= lo[0] - 1e-12) & (xs <= hi[0] + 1e-12))
        iz = np.flatnonzero((zs >= lo[2] - 1e-12) & (zs <= hi[2] + 1e-12))
        if not len(ix) or not len(iz):
            continue
        n, b = h.normals, h.offsets
        ny = n[:, 1]
        X = xs[ix][:, None, None]
        Z = zs[iz][None, :, None]
        up, lo_, flat = ny > 1e-12, ny < -1e-12, np.abs(ny) <= 1e-12

        def rhs(m: np.ndarray, b=b, n=n, X=X, Z=Z) -> np.ndarray:
            return b[m][None, None, :] - n[m, 0][None, None, :] * X - n[m, 2][None, None, :] * Z

        ymax = np.min(rhs(up) / ny[up], axis=-1) if up.any() else np.full((len(ix), len(iz)), np.inf)
        ymin = np.max(rhs(lo_) / ny[lo_], axis=-1) if lo_.any() else np.full((len(ix), len(iz)), _NEG)
        ok = ymax >= ymin - _EPS
        if flat.any():
            ok &= np.all(rhs(flat) >= -1e-11, axis=-1)
        sub = best[np.ix_(ix, iz)]
        best[np.ix_(ix, iz)] = np.where(ok, np.maximum(sub, ymax), sub)
    return best


def slice_points(g: PartGeometry, z: float) -> np.ndarray:
    """``(m, 2)`` (x, y) points where the part's edges meet the plane at height ``z``."""
    za, zb = g.edge_a[:, 2], g.edge_b[:, 2]
    da, db = za - z, zb - z
    crossing = (da * db <= 0.0) & (np.abs(za - zb) > 1e-15)
    t = np.where(crossing, da / np.where(crossing, za - zb, 1.0), 0.0)
    pa, pb = g.edge_a[crossing], g.edge_b[crossing]
    pts = pa[:, :2] + (pb[:, :2] - pa[:, :2]) * t[crossing, None]
    on = np.abs(g.vertices[:, 2] - z) <= 1e-15
    return np.concatenate([pts, g.vertices[on, :2]])


def band_rows(g: PartGeometry, z_lo: float, z_hi: float, n_rows: int = 25, max_extra: int = 96) -> np.ndarray:
    """Row heights for a band: ``n_rows`` evenly spaced plus the piece-vertex heights inside it."""
    base = np.linspace(z_lo, z_hi, n_rows)
    vz = g.vertices[:, 2]
    extra = np.unique(np.round(vz[(vz > z_lo) & (vz < z_hi)], 12))
    if len(extra) > max_extra:
        extra = np.quantile(extra, np.linspace(0.0, 1.0, max_extra))
    return np.unique(np.concatenate([base, extra]))


def cell_outline_rows(g: PartGeometry, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """``(len(xs), len(zs))``: per-cell maximum y at each row height (exact per row)."""
    xs = np.asarray(xs, dtype=np.float64)
    dx = float(xs[1] - xs[0])
    bounds = np.concatenate([xs - dx / 2.0, [xs[-1] + dx / 2.0]])
    env = envelope(g, bounds, zs)
    cell = np.maximum(env[:-1], env[1:])
    for j, z in enumerate(zs):
        pts = slice_points(g, float(z))
        if not len(pts):
            continue
        k = np.floor((pts[:, 0] - bounds[0]) / dx).astype(np.int64)
        keep = (k >= 0) & (k < len(xs))
        if keep.any():
            col = cell[:, j]
            np.maximum.at(col, k[keep], pts[keep, 1])
    return cell


def cell_outline(g: PartGeometry, xs: np.ndarray, z_lo: float, z_hi: float, n_rows: int = 25) -> np.ndarray:
    """Per-cell maximum y over the band ``[z_lo, z_hi]`` (see the module docstring)."""
    return cell_outline_rows(g, xs, band_rows(g, z_lo, z_hi, n_rows)).max(axis=1)


def polyline_gap(g: PartGeometry, xk: np.ndarray, yk: np.ndarray, zs: np.ndarray) -> float:
    """Exact smallest ``face(x) - outline(x)`` over the polyline's x range, at the given rows.

    Between consecutive breakpoints of the face (``xk``) and of each piece's row section (its
    vertices) the difference is linear, so its minimum is at one of those breakpoints. ``inf``
    if the part is absent.
    """
    best = np.inf
    env = envelope(g, xk, zs)
    fin = np.isfinite(env)
    if fin.any():
        best = min(best, float(np.min((yk[:, None] - env)[fin])))
    for z in zs:
        pts = slice_points(g, float(z))
        m = (pts[:, 0] >= xk[0]) & (pts[:, 0] <= xk[-1])
        if m.any():
            best = min(best, float(np.min(np.interp(pts[m, 0], xk, yk) - pts[m, 1])))
    return best


def column_max(g: PartGeometry, xs: np.ndarray) -> np.ndarray:
    """Per-cell maximum y over the whole part (every height), from each piece's x-y projection."""
    xs = np.asarray(xs, dtype=np.float64)
    dx = float(xs[1] - xs[0])
    bounds = np.concatenate([xs - dx / 2.0, [xs[-1] + dx / 2.0]])
    out = np.full(len(xs), _NEG)
    for h in g.pieces:
        p2 = h.vertices[:, :2]
        try:
            hull = ConvexHull(p2)
            poly = p2[hull.vertices]
        except QhullError:  # degenerate projection (a piece seen edge-on): use its points
            poly = p2
        order = np.lexsort((-poly[:, 1], poly[:, 0]))
        chain_x, chain_y = _upper_chain(poly[order])
        inside = (bounds >= chain_x[0]) & (bounds <= chain_x[-1])
        yb = np.where(inside, np.interp(bounds, chain_x, chain_y), _NEG)
        cell = np.maximum(yb[:-1], yb[1:])
        k = np.floor((chain_x - bounds[0]) / dx).astype(np.int64)
        keep = (k >= 0) & (k < len(xs))
        np.maximum.at(cell, k[keep], chain_y[keep])
        out = np.maximum(out, cell)
    return out


def _upper_chain(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper hull (monotone chain) of points sorted by x: the largest y at every x."""
    chain: list[np.ndarray] = []
    for p in pts:
        while len(chain) >= 2:
            a, b = chain[-2], chain[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0:
                chain.pop()
            else:
                break
        if chain and abs(chain[-1][0] - p[0]) < 1e-15:
            continue  # same x: keep the higher point (sorted first)
        chain.append(p)
    arr = np.array(chain)
    return arr[:, 0], arr[:, 1]


def max_y_in_box(g: PartGeometry, x_lo: float, x_hi: float, z_lo: float, z_hi: float) -> float:
    """The part's largest y inside the slab ``x in [x_lo, x_hi], z in [z_lo, z_hi]`` (exact).

    The maximum of y over a convex piece clipped by the slab is at a vertex of the clipped
    piece: a piece vertex inside the slab, a piece edge crossing a slab face, or a slab edge
    (a line along y) crossing the piece. ``-inf`` if the part does not enter the slab.
    """
    if z_hi < z_lo or x_hi < x_lo:
        return _NEG
    best = _NEG
    v = g.vertices
    inside = (v[:, 0] >= x_lo) & (v[:, 0] <= x_hi) & (v[:, 2] >= z_lo) & (v[:, 2] <= z_hi)
    if inside.any():
        best = float(v[inside, 1].max())
    a, b = g.edge_a, g.edge_b
    for axis, value in ((0, x_lo), (0, x_hi), (2, z_lo), (2, z_hi)):
        da, db = a[:, axis] - value, b[:, axis] - value
        cross = (da * db <= 0.0) & (np.abs(a[:, axis] - b[:, axis]) > 1e-15)
        if not cross.any():
            continue
        t = da[cross] / (a[cross, axis] - b[cross, axis])
        p = a[cross] + (b[cross] - a[cross]) * t[:, None]
        ok = (p[:, 0] >= x_lo - 1e-12) & (p[:, 0] <= x_hi + 1e-12) & (p[:, 2] >= z_lo - 1e-12)
        ok &= p[:, 2] <= z_hi + 1e-12
        if ok.any():
            best = max(best, float(p[ok, 1].max()))
    corners = envelope(g, np.array([x_lo, x_hi]), np.array([z_lo, z_hi]))
    if np.isfinite(corners).any():
        best = max(best, float(corners.max()))
    return best
