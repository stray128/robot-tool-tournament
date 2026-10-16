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

    @functools.cached_property
    def packed(self) -> _Packed:
        """Every piece's halfspaces in flat arrays, grouped by piece (for vectorised queries)."""
        return _pack(self.pieces)

    @functools.cached_property
    def upper_chains(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Per piece, the upper chain of its x-y projection (largest y at every x)."""
        out = []
        for h in self.pieces:
            p2 = h.vertices[:, :2]
            try:
                poly = p2[ConvexHull(p2).vertices]
            except QhullError:  # degenerate projection (a piece seen edge-on): use its points
                poly = p2
            out.append(_upper_chain(poly[np.lexsort((-poly[:, 1], poly[:, 0]))]))
        return tuple(out)

    def mirrored(self) -> PartGeometry:
        """The part mirrored across the closing axis (y -> -y)."""
        m = np.array([1.0, -1.0, 1.0])
        hs = tuple(
            PieceH(h.normals * m, h.offsets.copy(), h.vertices * m, h.edges.copy()) for h in self.pieces
        )
        return PartGeometry._assemble(hs)


@dataclass(frozen=True, eq=False)
class _Packed:
    """Every piece's halfspaces in padded ``(P, F)`` arrays, split by the sign of their y normal.

    Padding never binds: an extra upper bound at ``+inf``, an extra lower bound at ``-inf`` and
    an extra y-parallel face that every point satisfies.
    """

    up: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # nx, ny, nz, b (n_y > 0)
    down: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # n_y < 0
    flat: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # n_y = 0
    lo: np.ndarray  # (P, 3) piece bounding boxes
    hi: np.ndarray


def _pack(pieces: tuple[PieceH, ...]) -> _Packed:
    kinds = {"up": [], "down": [], "flat": []}  # type: dict[str, list[tuple[np.ndarray, np.ndarray]]]
    for h in pieces:
        ny = h.normals[:, 1]
        kinds["up"].append((h.normals[ny > 1e-12], h.offsets[ny > 1e-12]))
        kinds["down"].append((h.normals[ny < -1e-12], h.offsets[ny < -1e-12]))
        kinds["flat"].append((h.normals[np.abs(ny) <= 1e-12], h.offsets[np.abs(ny) <= 1e-12]))

    def pad(items: list[tuple[np.ndarray, np.ndarray]], ny_pad: float) -> tuple[np.ndarray, ...]:
        width = max(len(b) for _, b in items) + 1
        n = np.zeros((len(items), width, 3))
        n[:, :, 1] = ny_pad
        b = np.full((len(items), width), np.inf)
        for i, (nn, bb) in enumerate(items):
            n[i, : len(bb)] = nn
            b[i, : len(bb)] = bb
        return n[:, :, 0].copy(), n[:, :, 1].copy(), n[:, :, 2].copy(), b

    lo = np.stack([h.vertices.min(axis=0) for h in pieces])
    hi = np.stack([h.vertices.max(axis=0) for h in pieces])
    return _Packed(pad(kinds["up"], 1.0), pad(kinds["down"], -1.0), pad(kinds["flat"], 0.0), lo, hi)


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
    """``(len(xs), len(zs))``: the part's largest y at each (x, z), ``-inf`` where it is absent.

    Per piece the line ``(x, ., z)`` meets it on ``[max of the lower bounds, min of the upper
    bounds]`` when that is non-empty and no face parallel to y excludes the line. Only the
    (x, piece) pairs inside a piece's x extent are evaluated, all at once.
    """
    xs = np.asarray(xs, float)
    zs = np.asarray(zs, float)
    best = np.full((len(xs), len(zs)), _NEG)
    if not len(xs) or not len(zs):
        return best
    pk = g.packed
    order = np.argsort(xs, kind="stable")
    sx = xs[order]
    i0 = np.searchsorted(sx, pk.lo[:, 0] - 1e-12, side="left")
    i1 = np.searchsorted(sx, pk.hi[:, 0] + 1e-12, side="right")
    cnt = np.maximum(i1 - i0, 0)
    total = int(cnt.sum())
    if total == 0:
        return best
    piece = np.repeat(np.arange(len(cnt)), cnt)
    pos = np.repeat(i0, cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    xi = order[pos]
    width = max(pk.up[0].shape[1], pk.down[0].shape[1], pk.flat[0].shape[1])
    chunk = max(1, 1_000_000 // (len(zs) * width))
    Z = zs[None, :, None]
    n_pieces = len(cnt)
    dense = np.full((n_pieces, len(xs), len(zs)), _NEG) if n_pieces * best.size <= 4_000_000 else None
    for s0 in range(0, total, chunk):
        pc = piece[s0 : s0 + chunk]
        X = xs[xi[s0 : s0 + chunk]][:, None, None]

        def rhs(group: tuple[np.ndarray, ...], pc: np.ndarray = pc, X: np.ndarray = X) -> np.ndarray:
            nx, _, nz, b = group
            return b[pc][:, None, :] - nx[pc][:, None, :] * X - nz[pc][:, None, :] * Z

        ymax = np.min(rhs(pk.up) / pk.up[1][pc][:, None, :], axis=2)
        ymin = np.max(rhs(pk.down) / pk.down[1][pc][:, None, :], axis=2)
        ok = (ymax >= ymin - _EPS) & np.all(rhs(pk.flat) >= -1e-11, axis=2)
        ok &= (zs[None, :] >= pk.lo[pc, 2][:, None] - 1e-12) & (zs[None, :] <= pk.hi[pc, 2][:, None] + 1e-12)
        vals = np.where(ok, ymax, _NEG)
        if dense is not None:  # each (piece, x) pair occurs once
            dense[pc, xi[s0 : s0 + chunk]] = vals
        else:
            np.maximum.at(best, xi[s0 : s0 + chunk], vals)
    return dense.max(axis=0) if dense is not None else best


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


def _bin_max(out: np.ndarray, bounds: np.ndarray, x: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> None:
    """Raise ``out[i]`` to ``y`` for every closed cell ``[bounds[i], bounds[i + 1]]`` holding ``x``."""
    if not len(x):
        return
    n = len(out)
    lo = np.searchsorted(bounds, x - tol, side="left") - 1
    hi = np.searchsorted(bounds, x + tol, side="right") - 1
    for k in (lo, hi):
        keep = (k >= 0) & (k < n)
        if keep.any():
            np.maximum.at(out, k[keep], y[keep])


def _plane_crossings(
    g: PartGeometry, planes: np.ndarray, z_lo: float, z_hi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where the part's edges cross the planes ``x = planes[j]`` inside the band: (plane index, point)."""
    a, b = g.edge_a, g.edge_b
    d = b[:, 0] - a[:, 0]
    ok = np.abs(d) > 1e-15
    a, b, d = a[ok], b[ok], d[ok]
    lo, hi = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
    j0 = np.searchsorted(planes, lo, side="left")
    cnt = np.searchsorted(planes, hi, side="right") - j0
    total = int(cnt.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    e = np.repeat(np.arange(len(a)), cnt)
    j = np.repeat(j0, cnt) + (np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    t = (planes[j] - a[e, 0]) / d[e]
    p = a[e] + (b[e] - a[e]) * t[:, None]
    p[:, 0] = planes[j]
    inside = (p[:, 2] >= z_lo - 1e-12) & (p[:, 2] <= z_hi + 1e-12)
    return j[inside], p[inside]


def _band_points(g: PartGeometry, z_lo: float, z_hi: float) -> np.ndarray:
    """``(m, 3)``: piece vertices inside the band and the edge crossings of its two planes."""
    v = g.vertices
    m = (v[:, 2] >= z_lo - 1e-12) & (v[:, 2] <= z_hi + 1e-12)
    pts = [v[m]]
    for z in (z_lo, z_hi):
        sp = slice_points(g, z)
        pts.append(np.column_stack([sp, np.full(len(sp), z)]))
    return np.concatenate(pts)


def band_max_cells(g: PartGeometry, bounds: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """Exact largest y of the part in each cell ``[bounds[i], bounds[i+1]] x [z_lo, z_hi]``.

    The maximum of y over a convex piece clipped by a slab is attained at a vertex of the
    clipped piece: a piece vertex inside the slab, an edge crossing one of the slab's four
    planes, or one of the slab's four edges (lines along y) crossing the piece. All four kinds
    are evaluated, so the result is exact for any convex pieces (procedural or decomposed).
    ``-inf`` where the part is absent.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    out = np.full(len(bounds) - 1, _NEG)
    pts = _band_points(g, z_lo, z_hi)
    _bin_max(out, bounds, pts[:, 0], pts[:, 1])
    j, p = _plane_crossings(g, bounds, z_lo, z_hi)
    for cell in (j - 1, j):
        keep = (cell >= 0) & (cell < len(out))
        if keep.any():
            np.maximum.at(out, cell[keep], p[keep, 1])
    env = envelope(g, bounds, np.array([z_lo, z_hi])).max(axis=1)
    return np.maximum(out, np.maximum(env[:-1], env[1:]))


def cell_outline(
    g: PartGeometry, xs: np.ndarray, z_lo: float, z_hi: float, n_rows: int | None = None
) -> np.ndarray:
    """Per-cell maximum y over the band ``[z_lo, z_hi]``, exact (:func:`band_max_cells`).

    Cells have width ``dx`` and are centred on the grid points ``xs``. ``n_rows`` is accepted
    for compatibility and ignored: the maximum no longer depends on sampled rows.
    """
    xs = np.asarray(xs, dtype=np.float64)
    dx = float(xs[1] - xs[0])
    bounds = np.concatenate([xs - dx / 2.0, [xs[-1] + dx / 2.0]])
    return band_max_cells(g, bounds, z_lo, z_hi)


def polyline_gap(
    g: PartGeometry,
    xk: np.ndarray,
    yk: np.ndarray,
    z_lo: float,
    z_hi: float,
    kz: float = 0.0,
    z_ref: float = 0.0,
) -> float:
    """Exact smallest ``face(x) + kz (z - z_ref) - y`` over the part's material in the band.

    ``face`` is the polyline through ``(xk, yk)``, for ``x`` in its range; ``kz`` tilts the face
    in height (``-1`` for a 45-degree bevel that recedes below ``z_ref``, ``+1`` for one that
    recedes above it). On each segment the objective is linear, so its minimum over the part
    clipped by the segment's x slab and the band is at a vertex of the clipped part (see
    :func:`band_max_cells`). ``inf`` if the part is absent.
    """
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    if z_hi < z_lo:
        return np.inf
    best = np.inf
    pts = _band_points(g, z_lo, z_hi)
    m = (pts[:, 0] >= xk[0] - 1e-12) & (pts[:, 0] <= xk[-1] + 1e-12)
    if m.any():
        q = pts[m]
        best = min(best, float(np.min(np.interp(q[:, 0], xk, yk) + kz * (q[:, 2] - z_ref) - q[:, 1])))
    j, p = _plane_crossings(g, xk, z_lo, z_hi)
    if len(j):
        best = min(best, float(np.min(yk[j] + kz * (p[:, 2] - z_ref) - p[:, 1])))
    zs = np.array([z_lo, z_hi])
    env = envelope(g, xk, zs)
    fin = np.isfinite(env)
    if fin.any():
        obj = yk[:, None] + kz * (zs[None, :] - z_ref) - env
        best = min(best, float(np.min(obj[fin])))
    return best


def column_max(g: PartGeometry, xs: np.ndarray) -> np.ndarray:
    """Per-cell maximum y over the whole part (every height), from each piece's x-y projection."""
    xs = np.asarray(xs, dtype=np.float64)
    dx = float(xs[1] - xs[0])
    bounds = np.concatenate([xs - dx / 2.0, [xs[-1] + dx / 2.0]])
    out = np.full(len(xs), _NEG)
    for chain_x, chain_y in g.upper_chains:
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
