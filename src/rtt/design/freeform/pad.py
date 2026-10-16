"""Pad faces in plan view: grown outline, strategy face, rib removal and simplification.

All functions work in the left-jaw convention on the jaw's x grid (0.1 mm): ``p[i]`` is the
part's outline (largest y, see :mod:`rtt.design.freeform.outline`) in cell ``i`` and a face
``f[i] >= p[i]`` is the pad's front; the jaw's material lies behind it (larger y). ``-inf``
marks cells the part does not reach.

* **Grown outline** ``h = dilate_x(p, clearance + sweep) + clearance``: grown sideways by
  the clearance (plus the baseline-only sweep) and moved back by the clearance.
* **flat**: a plane at the outline's frontmost point (contact allowed).
* **conform**: ``max(env, P - d)``, where ``P`` is the frontmost point of the grown outline,
  ``d`` the pad depth and ``env`` the smallest surface above the grown outline whose slope is
  at most ``cot(draft)`` (walls at least ``draft`` from the closing axis; draft 0 = the
  grown outline itself, a straight-walled imprint).
* **v_funnel**: a V centred on the jaw with flanks ``draft`` from the closing axis that just
  touches the grown outline, cut off ``d`` below its apex.
* **Rib removal**: material ribs narrower than 1.6 mm are trimmed back (a morphological
  opening of the material); the face only ever moves back, away from the part.
* **Simplify**: at most 12 linear segments that never lie in front of the face.
"""

from __future__ import annotations

import math

import numpy as np

_NEG = -np.inf
MIN_FEATURE = 0.0016  # m; printable feature width (ribs narrower than this are trimmed)


def dilate_x(p: np.ndarray, k: int) -> np.ndarray:
    """Running maximum over ``+-k`` cells (``-inf`` outside the grid)."""
    if k <= 0:
        return p.copy()
    out = p.copy()
    for s in range(1, k + 1):
        out[s:] = np.maximum(out[s:], p[:-s])
        out[:-s] = np.maximum(out[:-s], p[s:])
    return out


def _erode_x(f: np.ndarray, k: int) -> np.ndarray:
    """Running minimum over ``+-k`` cells, treating cells outside the grid as ``+inf``."""
    out = f.copy()
    for s in range(1, k + 1):
        out[s:] = np.minimum(out[s:], f[:-s])
        out[:-s] = np.minimum(out[:-s], f[s:])
    return out


def _max_filter_open(f: np.ndarray, k: int) -> np.ndarray:
    """Running maximum over ``+-k`` cells, treating cells outside the grid as ``+inf``."""
    out = f.copy()
    n = len(f)
    for s in range(1, k + 1):
        out[s:] = np.maximum(out[s:], f[:-s])
        out[:-s] = np.maximum(out[:-s], f[s:])
        out[: min(s, n)] = np.inf
        out[max(n - s, 0) :] = np.inf
    return out


def grown_outline(p: np.ndarray, dx: float, clearance: float, sweep: float = 0.0) -> np.ndarray:
    """``dilate_x(p, clearance + sweep) + clearance`` (``-inf`` stays ``-inf``)."""
    k = math.ceil((clearance + sweep) / dx - 1e-9)
    return dilate_x(p, k) + clearance


def pad_face(
    p: np.ndarray,
    xs: np.ndarray,
    pad: str,
    clearance: float,
    depth: float,
    draft_rad: float,
    sweep: float = 0.0,
) -> np.ndarray:
    """The strategy's face on the grid (left-jaw convention, part frame y), from the outline ``p``."""
    finite = np.isfinite(p)
    if not finite.any():
        raise ValueError("the part's outline is empty in the pad band")
    if pad == "flat":
        return np.full_like(xs, float(np.max(p[finite])))
    dx = float(xs[1] - xs[0])
    return face_on_floor(grown_outline(p, dx, clearance, sweep), xs, pad, depth, draft_rad)


def face_on_floor(h: np.ndarray, xs: np.ndarray, pad: str, depth: float, draft_rad: float) -> np.ndarray:
    """The conform or v_funnel face over a grown outline ``h`` (the surface it must stay behind)."""
    fin = np.isfinite(h)
    if not fin.any():
        raise ValueError("the part's outline is empty in the pad band")
    P = float(h[fin].max())
    lip = math.inf if draft_rad <= 0.0 else 1.0 / math.tan(draft_rad)
    hx, hv = xs[fin], h[fin]
    if pad == "conform":
        if math.isinf(lip):
            env = np.where(fin, h, _NEG)
        else:
            env = np.max(hv[None, :] - lip * np.abs(xs[:, None] - hx[None, :]), axis=1)
        return np.maximum(env, P - depth)
    if pad == "v_funnel":
        if math.isinf(lip):
            raise ValueError("a v_funnel needs a positive draft")
        apex = float(np.max(hv + lip * np.abs(hx)))
        return np.maximum(apex - lip * np.abs(xs), apex - depth)
    raise ValueError(f"unknown pad strategy {pad!r}")


def remove_thin_ribs(f: np.ndarray, dx: float, width: float = MIN_FEATURE) -> np.ndarray:
    """Morphological opening of the material behind ``f`` by a ``width``-wide segment.

    Material protrusions toward the part that are narrower than ``width`` are trimmed back
    to the level of their surroundings. The result is never in front of ``f``.
    """
    k = math.ceil(width / 2.0 / dx - 1e-9)
    if k <= 0:
        return f.copy()
    return np.maximum(_erode_x(_max_filter_open(f, k), k), f)


def rib_excess(f: np.ndarray, dx: float, width: float = MIN_FEATURE) -> float:
    """How far (m) :func:`remove_thin_ribs` would trim ``f``: 0 for a face with no thin ribs."""
    return float(np.max(remove_thin_ribs(f, dx, width) - f))


def simplify_upper(xs: np.ndarray, f: np.ndarray, max_seg: int) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear approximation with at most ``max_seg`` segments, never in front of ``f``.

    Douglas-Peucker on the grid points with a growing tolerance, then vertices are moved back
    until every segment lies at or behind ``f`` at every grid point.
    """
    n = len(xs)

    def rdp(i0: int, i1: int, tol: float, out: list[int]) -> None:
        stack = [(i0, i1)]
        while stack:
            a, b = stack.pop()
            seg = np.interp(xs[a : b + 1], [xs[a], xs[b]], [f[a], f[b]])
            err = np.abs(f[a : b + 1] - seg)
            k = int(np.argmax(err))
            if err[k] > tol and 0 < k < b - a:
                stack.append((a + k, b))
                stack.append((a, a + k))
            else:
                out.append(b)

    tol = 2e-6
    while True:
        out = [0]
        rdp(0, n - 1, tol, out)
        if len(out) - 1 <= max_seg:
            break
        tol *= 1.6
    idx = np.array(out)
    yk = f[idx].copy()
    for _ in range(200):
        line = np.interp(xs, xs[idx], yk)
        dip = f - line
        if float(dip.max()) <= 0.0:
            break
        lift = np.zeros_like(yk)
        for j in range(len(idx) - 1):
            a, b = idx[j], idx[j + 1]
            v = float(dip[a : b + 1].max())
            if v > 0.0:
                lift[j] = max(lift[j], v)
                lift[j + 1] = max(lift[j + 1], v)
        yk += lift
    else:  # pragma: no cover - the lift loop always converges well before this
        raise RuntimeError("pad simplification did not converge")
    return _merge_collinear(xs[idx].copy(), yk)


def _merge_collinear(xk: np.ndarray, yk: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    keep = [0]
    for i in range(1, len(xk) - 1):
        a, b, c = keep[-1], i, i + 1
        cross = (xk[b] - xk[a]) * (yk[c] - yk[a]) - (yk[b] - yk[a]) * (xk[c] - xk[a])
        if abs(cross) > tol * max(1.0, abs(xk[c] - xk[a])):
            keep.append(i)
    keep.append(len(xk) - 1)
    return xk[keep], yk[keep]


def lift_polyline(xs: np.ndarray, xk: np.ndarray, yk: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Move the vertices ``yk`` back (larger y) until the polyline is at or behind ``floor``."""
    yk = yk.copy()
    idx = np.searchsorted(xs, xk)
    for _ in range(200):
        dip = floor - np.interp(xs, xk, yk)
        if not np.isfinite(dip).any() or float(np.nanmax(np.where(np.isfinite(dip), dip, _NEG))) <= 0.0:
            return yk
        lift = np.zeros_like(yk)
        for j in range(len(xk) - 1):
            a, b = idx[j], idx[j + 1]
            seg = dip[a : b + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size and float(seg.max()) > 0.0:
                v = float(seg.max())
                lift[j] = max(lift[j], v)
                lift[j + 1] = max(lift[j + 1], v)
        yk += lift
    raise RuntimeError("polyline lift did not converge")  # pragma: no cover
