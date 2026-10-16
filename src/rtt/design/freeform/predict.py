"""Static capture predictor: will a jaw pair centre, leave or jam a part placed off-centre?

For a lateral placement error ``e`` (across the jaw, along x) each jaw's simplified face is
compared with the part outline shifted by ``e``:

* **seat**: the jaw reaches its seat (stops at most 0.05 mm short of where it would on the
  centred part), so it pushes the part nowhere sideways;
* **centre**: its first contact is on a wall steep enough to slide on, i.e. the wall's angle
  from the closing axis is at most ``90 - atan(mu) - margin`` degrees, so the squeeze pushes
  the part toward the middle;
* **jam**: its first contact is on a surface too flat to slide on, before its seat.

The pair is ``jammed`` if either jaw jams, ``centred`` if either centres (or the error is
within the 0.75 mm registration tolerance), otherwise ``uncentred``. This is the MVP's rule;
it reproduces the simulated registered / not-registered table on the pipe tee 24 / 24.

:func:`capture_report` scans ``e`` in 0.1 mm steps up to 5 mm in both directions and returns
the no-jam and centring ranges plus the outcome at -3 ... +3 mm. It takes a few ms. It is
guidance on design cards and in proposer feedback, a test oracle and a hero pre-screen; it
never scores a design.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rtt.design.freeform.types import CaptureReport

SEAT_TOL = 5e-5  # m; a jaw within this of its seat gap has seated
TIE_TOL = 1e-5  # m; first-contact set
REGISTER_TOL = 0.00075  # m; |error| within this counts as centred without any push
MARGIN_DEG = 5.0
SCAN_MAX = 0.005  # m
SCAN_STEP = 0.0001  # m
REPORT_ERRORS_MM = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)


def slide_limit_deg(mu: float, margin_deg: float = MARGIN_DEG) -> float:
    """Largest wall angle from the closing axis that still slides at friction ``mu``."""
    return 90.0 - math.degrees(math.atan(mu)) - margin_deg


def _angles(slope: np.ndarray) -> np.ndarray:
    """Angle of a surface from the closing axis: 90 = flat face, 0 = wall along the closing axis."""
    return np.degrees(np.arctan2(1.0, np.abs(slope)))


def jaw_responses(
    xs: np.ndarray, p: np.ndarray, xk: np.ndarray, yk: np.ndarray, shifts: np.ndarray, limit_deg: float
) -> np.ndarray:
    """Per shift (in grid cells): 0 = seat, 1 = centre, 2 = jam.

    The first-contact set is every grid point within :data:`TIE_TOL` of the smallest gap. Each
    contact point slides when its contact surface is within ``limit_deg`` of the closing axis:
    inside a pad segment that is the segment; at a concave pad corner (a V's root) the flatter
    of its two segments; at a convex pad edge the part's own surface there (the edge meets the
    part's facet), falling back to the flatter segment where the part has no clear slope.
    """
    xs = np.asarray(xs, dtype=np.float64)
    xk = np.asarray(xk, dtype=np.float64)
    yk = np.asarray(yk, dtype=np.float64)
    fk = np.interp(xs, xk, yk)
    n = len(xs)
    nseg = len(xk) - 1
    seg_alpha = _angles(np.diff(yk) / np.maximum(np.diff(xk), 1e-15))
    seg = np.clip(np.searchsorted(xk, xs, side="right") - 1, 0, nseg - 1)
    alpha = seg_alpha[seg]
    convex = np.zeros(n, dtype=bool)
    for j in range(1, nseg):  # interior breakpoints that fall on the grid
        i = int(np.argmin(np.abs(xs - xk[j])))
        if abs(xs[i] - xk[j]) > 1e-9:
            continue
        left, right = (yk[j] - yk[j - 1]) / (xk[j] - xk[j - 1]), (yk[j + 1] - yk[j]) / (xk[j + 1] - xk[j])
        alpha[i] = max(seg_alpha[j - 1], seg_alpha[j])
        convex[i] = right > left + 1e-9  # the pad material bulges toward the part here
    fin = np.isfinite(p)
    g0 = float(np.min(np.where(fin, fk - p, np.inf)))
    idx = np.arange(n)[None, :] - np.asarray(shifts, dtype=np.int64)[:, None]
    valid = (idx >= 0) & (idx < n)
    ps = np.where(valid, p[np.clip(idx, 0, n - 1)], -np.inf)
    ge = np.where(np.isfinite(ps), fk[None, :] - ps, np.inf)
    gmin = ge.min(axis=1)
    seat = gmin >= g0 - SEAT_TOL
    tied = ge <= gmin[:, None] + TIE_TOL
    eff = np.broadcast_to(alpha, ps.shape).copy()
    if convex.any():
        dx = float(xs[1] - xs[0])
        prev, nxt = np.roll(ps, 1, axis=1), np.roll(ps, -1, axis=1)
        prev[:, 0], nxt[:, -1] = -np.inf, -np.inf
        ok = np.isfinite(prev) & np.isfinite(nxt) & np.isfinite(ps)
        slope = np.zeros(ps.shape)
        slope[ok] = (nxt[ok] - prev[ok]) / (2.0 * dx)
        part_alpha = _angles(slope)
        use = ok & convex[None, :]
        eff = np.where(use, part_alpha, eff)
    amax = np.max(np.where(tied, eff, -np.inf), axis=1)
    return np.where(seat, 0, np.where(amax <= limit_deg, 1, 2))


def pair_outcomes(left: Any, right: Any, errors_m: np.ndarray, mu: float) -> list[str]:
    """Predicted outcome per lateral error (metres) for two planned jaws (see the module docstring)."""
    dx = float(left.xs[1] - left.xs[0])
    shifts = np.rint(np.asarray(errors_m) / dx).astype(np.int64)
    limit = slide_limit_deg(mu)
    rl = jaw_responses(left.xs, left.p, left.xk, left.yk, shifts, limit)
    rr = jaw_responses(right.xs, right.p, right.xk, right.yk, shifts, limit)
    out = []
    for e, a, b in zip(errors_m, rl, rr, strict=True):
        if a == 2 or b == 2:
            out.append("jammed")
        elif a == 1 or b == 1:
            out.append("centred")
        else:
            out.append("centred" if abs(e) <= REGISTER_TOL + 1e-9 else "uncentred")
    return out


def capture_ranges(left: Any, right: Any, mu: float) -> tuple[float, float]:
    """(no-jam range, centring range) in mm, scanning +-e up to 5 mm."""
    steps = np.arange(0.0, SCAN_MAX + SCAN_STEP / 2, SCAN_STEP)
    both = np.concatenate([steps, -steps])
    outs = pair_outcomes(left, right, both, mu)
    m = len(steps)
    nojam = centred = None
    for i, e in enumerate(steps):
        o = {outs[i], outs[m + i]}
        if nojam is None and "jammed" in o:
            nojam = e - SCAN_STEP
        if centred is None and o != {"centred"}:
            centred = e - SCAN_STEP
    nj = SCAN_MAX if nojam is None else max(0.0, nojam)
    ce = SCAN_MAX if centred is None else max(0.0, centred)
    return round(nj * 1000.0, 6), round(ce * 1000.0, 6)


def capture_report(plan: Any, mu: float) -> CaptureReport:
    """The static capture prediction for one planned (design, part) pair."""
    nojam, centring = capture_ranges(plan.left, plan.right, mu)
    errs = np.array(REPORT_ERRORS_MM) / 1000.0
    outs = pair_outcomes(plan.left, plan.right, errs, mu)
    return CaptureReport(
        no_jam_mm=nojam,
        centring_mm=centring,
        mu=mu,
        by_error=tuple((e, o) for e, o in zip(REPORT_ERRORS_MM, outs, strict=True)),  # type: ignore[misc]
    )


def predict(plan: Any, error_mm: float, mu: float = 0.3) -> str:
    """Predicted outcome for one lateral placement error (mm)."""
    return pair_outcomes(plan.left, plan.right, np.array([error_mm / 1000.0]), mu)[0]
