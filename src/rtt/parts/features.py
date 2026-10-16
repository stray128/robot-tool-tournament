"""Part features (sockets, flanges): checks against the geometry, probes, yaw symmetry.

A :class:`~rtt.parts.types.Feature` is a promise about the collision pieces: a socket is an
empty bore of the declared radius and depth behind its mouth, and a flange is a rim of the
declared radius and thickness with free space under it. :func:`check_features` verifies
those promises point by point against the convex pieces (the geometry the simulator sees),
so the jaw generator can size a locator pin or a hook from the feature alone.

:func:`free_depth` measures how far a pin of a given radius can enter a socket, and
:func:`detect_yaw_symmetry` estimates the yaw symmetry of a scanned part from the support
function of its footprint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from rtt.parts.types import Feature, PartSpec

PROBE_STEP = 0.5e-3  # m; how far probes sit from a surface they test
_RADIAL_FRAC = (0.0, 0.5, 0.9, 1.0)
_N_AZIMUTH = 12


def _halfspaces(part: PartSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    from rtt.parts.library import part_pieces

    return [(p.normals, p.offsets) for p in part_pieces(part)]


def inside(part: PartSpec, points: np.ndarray, margin: float = 1e-6, _hs: list | None = None) -> np.ndarray:
    """Which ``points`` (``(m, 3)``, object frame) lie inside some piece, deeper than ``margin``."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    hs = _hs if _hs is not None else _halfspaces(part)
    hit = np.zeros(len(pts), dtype=bool)
    for normals, offsets in hs:
        hit |= np.all(pts @ normals.T <= offsets - margin, axis=1)
    return hit


def _basis(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def _disc(origin: np.ndarray, axis: Sequence[float], radius: float, offset: float) -> np.ndarray:
    """Probe points on a disc of ``radius`` at ``offset`` along ``axis`` from ``origin``."""
    a, u, v = _basis(axis)
    pts = [origin + offset * a]
    for frac in _RADIAL_FRAC[1:]:
        for k in range(_N_AZIMUTH):
            t = 2 * math.pi * (k + 0.5) / _N_AZIMUTH
            pts.append(origin + offset * a + frac * radius * (math.cos(t) * u + math.sin(t) * v))
    return np.array(pts)


def _ring(origin: np.ndarray, axis: Sequence[float], radius: float, offset: float) -> np.ndarray:
    a, u, v = _basis(axis)
    return np.array(
        [
            origin + offset * a + radius * (math.cos(t) * u + math.sin(t) * v)
            for t in (2 * math.pi * (k + 0.5) / _N_AZIMUTH for k in range(_N_AZIMUTH))
        ]
    )


def _extent_along(part: PartSpec, origin: np.ndarray, axis: Sequence[float]) -> float:
    """How far the part reaches behind ``origin`` against ``axis`` (m)."""
    a = np.asarray(axis, dtype=np.float64)
    pts = np.concatenate(part.pieces)
    return float(np.max((origin - pts) @ a))


def check_socket(part: PartSpec, f: Feature, _hs: list | None = None) -> list[str]:
    origin = np.asarray(f.origin)
    hs = _hs if _hs is not None else _halfspaces(part)
    out = []
    a = np.asarray(f.axis)
    for depth_frac in (0.02, 0.25, 0.5, 0.75, 0.98):
        pts = _disc(origin, a, 0.98 * f.radius, -depth_frac * f.depth)
        if inside(part, pts, _hs=hs).any():
            out.append(f"{f.name}: material inside the bore at {depth_frac:.0%} of the depth")
    if inside(part, _disc(origin, a, 0.98 * f.radius, PROBE_STEP), _hs=hs).any():
        out.append(f"{f.name}: the mouth is covered")
    wall = _ring(origin, a, f.radius + PROBE_STEP, -0.5 * f.depth)
    if not inside(part, wall, margin=0.0, _hs=hs).all():
        out.append(f"{f.name}: no wall around the bore at half depth (radius too small?)")
    through = f.depth >= _extent_along(part, origin, a) - 1e-6
    if not through and not inside(part, origin[None, :] - (f.depth + PROBE_STEP) * a, _hs=hs).all():
        out.append(f"{f.name}: no floor just beyond the declared depth (depth too small?)")
    return out


def check_flange(part: PartSpec, f: Feature, _hs: list | None = None) -> list[str]:
    origin = np.asarray(f.origin)
    a = np.asarray(f.axis)
    hs = _hs if _hs is not None else _halfspaces(part)
    out = []
    rim = 0.97 * f.radius
    if inside(part, _ring(origin, a, rim, PROBE_STEP), _hs=hs).any():
        out.append(f"{f.name}: no free space under the rim")
    for frac in (0.1, 0.5, 0.9):
        if not inside(part, _ring(origin, a, rim, -frac * f.depth), margin=0.0, _hs=hs).all():
            out.append(f"{f.name}: no rim material at {frac:.0%} of the thickness")
    if inside(part, _ring(origin, a, rim, -(f.depth + PROBE_STEP)), _hs=hs).any():
        out.append(f"{f.name}: material beyond the declared thickness")
    if inside(part, _ring(origin, a, f.radius + PROBE_STEP, -0.5 * f.depth), _hs=hs).any():
        out.append(f"{f.name}: material beyond the declared radius")
    return out


def check_features(part: PartSpec) -> list[str]:
    """Every way the part's declared features disagree with its pieces (empty when exact)."""
    hs = _halfspaces(part)
    out: list[str] = []
    for f in part.features:
        out += check_socket(part, f, hs) if f.kind == "socket" else check_flange(part, f, hs)
    return out


def free_depth(part: PartSpec, f: Feature, radius: float | None = None, step: float = 0.1e-3) -> float:
    """How deep (m) a pin of ``radius`` (default: the socket's) enters before it touches the
    part, measured from the mouth along the socket axis, at ``step`` resolution."""
    if f.kind != "socket":
        raise ValueError(f"{f.name} is a {f.kind}, not a socket")
    hs = _halfspaces(part)
    r = f.radius if radius is None else radius
    origin = np.asarray(f.origin)
    a = np.asarray(f.axis)
    limit = _extent_along(part, origin, a)
    depth = 0.0
    while depth + step <= limit + 1e-12:
        pts = np.concatenate(
            [_disc(origin, a, 0.999 * r, -(depth + step)), _ring(origin, a, 0.999 * r, -(depth + step))]
        )
        if inside(part, pts, _hs=hs).any():
            return depth
        depth += step
    return limit


def sockets_facing(part: PartSpec, side: str, tol_deg: float = 10.0) -> tuple[Feature, ...]:
    """The part's sockets whose axis faces the ``side`` jaw (``"left"`` is +y)."""
    return tuple(f for f in part.sockets if f.facing(tol_deg) == side)


def _support(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return np.max(points @ d.T, axis=0)


def detect_yaw_symmetry(
    pieces: Sequence[np.ndarray], tol: float = 0.5e-3, candidates: Sequence[int] = (6, 4, 3, 2)
) -> int:
    """Yaw symmetry of a part's footprint: 0 if round within ``tol`` (m), else the largest
    ``n`` in ``candidates`` whose rotation by ``360/n`` maps the footprint onto itself within
    ``tol``, else 1. Uses the support function of the horizontal projection about the
    footprint's centroid at 0.5 degree steps (it sees the outline, not internal structure)."""
    pts = np.concatenate([np.asarray(p, dtype=np.float64)[:, :2] for p in pieces])
    pts = pts - (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    angles = np.radians(np.arange(0.0, 360.0, 0.5))
    h = _support(pts, angles)
    if float(h.max() - h.min()) <= tol:
        return 0
    for n in sorted(candidates, reverse=True):
        shift = round(len(angles) / n)
        if len(angles) % n == 0 and float(np.max(np.abs(np.roll(h, shift) - h))) <= tol:
            return int(n)
    return 1
