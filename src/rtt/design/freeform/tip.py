"""Exact convex pad pieces: one strip per pad segment, beveled or hooked at the tip.

Jaw-local frame, left-jaw convention: x across the jaw, y away from the part (the jaw's
material lies at larger y than its face), z up with the root plane at ``z = 0``. A pad
segment ``j`` spans ``x in [xk[j], xk[j+1]]`` with a linear front ``F`` from ``yk[j]`` to
``yk[j+1]``; its strip reaches back to ``y_back`` and up the pad band ``[z0, z1]``.

Every piece is the hull of two convex sections in the y-z plane, one at each end of its
x interval, whose corresponding edges are parallel. All faces are therefore planar and the
piece is exactly convex (no CoACD on fingers). Each piece keeps its construction faces, so
its volume can be checked against its convex hull (:func:`piece_volume`).

* **blunt**: a rectangle section.
* **chamfer**: the lower front corner is cut at 45 degrees by the chamfer leg ``s``. The
  same plane cuts every strip; the generator makes the strips deep enough
  (``y_back >= max(yk) + s + MIN_LAND``) that the bevel never reaches a strip's back.
* **hook**: one extra piece per segment in front of the strip, from the hook front ``H`` back
  to ``F`` (plus :data:`HOOK_OVERLAP` into the strip), ``hh`` tall at its front with a
  45-degree top rising toward the face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_LAND = 0.0005  # m; land left at a strip's bottom by the chamfer
HOOK_OVERLAP = 0.0005  # m; hook pieces reach this far into the pad strips (fuses the print)
HOOK_MIN_PROTRUSION = 5e-5  # m; segments whose hook would protrude less are left out
_DEC = 7


@dataclass(frozen=True, eq=False)
class Piece:
    """A convex piece: vertices (metres, jaw frame) plus its construction faces (outward)."""

    vertices: np.ndarray
    faces: np.ndarray
    role: str  # "pad", "hook", "body", "root", "locator"


def snap(v: float | np.ndarray) -> np.ndarray | float:
    """Round to the 1e-7 m grid that every vertex lives on."""
    return np.round(v, _DEC)


def piece_from_sections(sec_a: np.ndarray, sec_b: np.ndarray, role: str, *, on_grid: bool = True) -> Piece:
    """Hull of two 3D polygons with corresponding vertices (same count, same winding).

    ``on_grid`` rounds the vertices to the 1e-7 m grid; pieces built from grid values stay
    exactly planar-faced. Round pieces (the locator pin) keep their exact vertices instead.
    """
    a = np.asarray(sec_a, dtype=np.float64)
    b = np.asarray(sec_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("sections must be matching (k >= 3, 3) arrays")
    k = len(a)
    verts = np.concatenate([a, b])
    if on_grid:
        verts = np.round(verts, _DEC)
    faces: list[tuple[int, int, int]] = []
    for i in range(1, k - 1):
        faces.append((0, i + 1, i))  # cap a, reversed
        faces.append((k, k + i, k + i + 1))  # cap b
    for i in range(k):
        j = (i + 1) % k
        faces.append((i, j, k + j))
        faces.append((i, k + j, k + i))
    f = np.array(faces, dtype=np.int64)
    if _signed_volume(verts, f) < 0:
        f = f[:, ::-1].copy()
    return Piece(verts, f, role)


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cx = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
    cy = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
    cz = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
    return float((a[:, 0] * cx + a[:, 1] * cy + a[:, 2] * cz).sum() / 6.0)


def piece_volume(piece: Piece) -> float:
    """Volume enclosed by the piece's construction faces (divergence theorem)."""
    return _signed_volume(piece.vertices, piece.faces)


def piece_from_box(lo: tuple[float, float, float], hi: tuple[float, float, float], role: str) -> Piece:
    x0, y0, z0 = (float(v) for v in lo)
    x1, y1, z1 = (float(v) for v in hi)
    sec = lambda x: np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]])
    return piece_from_sections(sec(x0), sec(x1), role)


def _strip_section(x: float, front: float, y_back: float, z0: float, z1: float, chamfer: float) -> np.ndarray:
    """(y, z) section of a pad strip at ``x``, optionally beveled by ``chamfer`` at 45 degrees."""
    if chamfer > 0.0:
        yz = [(front + chamfer, z0), (y_back, z0), (y_back, z1), (front, z1), (front, z0 + chamfer)]
    else:
        yz = [(front, z0), (y_back, z0), (y_back, z1), (front, z1)]
    return np.array([[x, y, z] for y, z in yz])


def pad_strips(
    xk: np.ndarray, yk: np.ndarray, y_back: float, z0: float, z1: float, chamfer: float = 0.0
) -> list[Piece]:
    """One exact convex strip per pad segment (see the module docstring)."""
    if chamfer > 0.0 and (
        y_back - float(np.max(yk)) < chamfer + MIN_LAND - 1e-12 or z1 - z0 < chamfer + MIN_LAND - 1e-12
    ):
        raise ValueError("the chamfer does not fit inside the pad strips")
    out = []
    for j in range(len(xk) - 1):
        if xk[j + 1] - xk[j] <= 0.0:
            continue
        a = _strip_section(float(xk[j]), float(yk[j]), y_back, z0, z1, chamfer)
        b = _strip_section(float(xk[j + 1]), float(yk[j + 1]), y_back, z0, z1, chamfer)
        out.append(piece_from_sections(a, b, "pad"))
    return out


def _hook_section(x: float, front: float, face: float, z0: float, hh: float) -> np.ndarray:
    back = face + HOOK_OVERLAP
    rise = back - front  # the 45-degree top rises one-for-one toward the face
    yz = [(front, z0), (back, z0), (back, z0 + hh + rise), (front, z0 + hh)]
    return np.array([[x, y, z] for y, z in yz])


def hook_pieces(xk: np.ndarray, yk: np.ndarray, hk: np.ndarray, z0: float, hh: float) -> list[Piece]:
    """One hook piece per segment where the hook front ``hk`` protrudes in front of the face ``yk``."""
    out = []
    for j in range(len(xk) - 1):
        if xk[j + 1] - xk[j] <= 0.0:
            continue
        if not _hook_kept(yk, hk, j):
            continue
        a = _hook_section(float(xk[j]), float(hk[j]), float(yk[j]), z0, hh)
        b = _hook_section(float(xk[j + 1]), float(hk[j + 1]), float(yk[j + 1]), z0, hh)
        out.append(piece_from_sections(a, b, "hook"))
    return out


def _hook_kept(yk: np.ndarray, hk: np.ndarray, j: int) -> bool:
    return max(yk[j] - hk[j], yk[j + 1] - hk[j + 1]) >= HOOK_MIN_PROTRUSION


def hooked_front_loft(
    xk: np.ndarray,
    yk: np.ndarray,
    hk: np.ndarray,
    z0: float,
    hh: float,
    far: float,
    z_lo: float,
    z_hi: float,
    ext: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The region behind a hooked jaw's front, lofted along x (for the print; any length unit).

    At every face vertex the section in the (y, z) plane is bounded in front by the hook
    front ``hk`` up to ``z0 + hh``, the hook's 45-degree top up to the face ``yk`` and the face
    above; behind by ``far``; below and above by ``z_lo`` and ``z_hi``. Consecutive sections
    have parallel corresponding edges, so intersecting this with the body gives exactly the
    pad strips plus the hook pieces, as one solid with no faces for a boolean to fuse. The
    loft reaches ``ext`` past the first and last vertex. Returns ``(vertices, faces)``,
    faces outward.
    """
    xs = np.concatenate([[xk[0] - ext], xk, [xk[-1] + ext]])
    fs = np.concatenate([[yk[0]], yk, [yk[-1]]])
    hs = np.concatenate([[hk[0]], hk, [hk[-1]]])
    secs = []
    for x, f, h in zip(xs, fs, hs, strict=True):
        zk = z0 + hh + max(f - h, 1e-6 * max(1.0, abs(far)))  # the top meets the face (never at zero length)
        yz = [(h, z_lo), (far, z_lo), (far, z_hi), (f, z_hi), (f, zk), (h, z0 + hh)]
        secs.append([(x, y, z) for y, z in yz])
    k = 6
    verts = np.array(secs, dtype=np.float64).reshape(-1, 3)
    faces: list[tuple[int, int, int]] = []
    for s in range(len(secs) - 1):
        a, b = k * s, k * (s + 1)
        for i in range(k):
            j = (i + 1) % k
            faces += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    last = k * (len(secs) - 1)
    for i in (2, 3, 4, 5):  # caps fanned from the far bottom corner (the section is star-shaped from it)
        faces += [(1, i + 1 if i < 5 else 0, i), (last + 1, last + i, last + (i + 1 if i < 5 else 0))]
    f = np.array(faces, dtype=np.int64)
    if _signed_volume(verts, f) < 0:
        f = f[:, ::-1].copy()
    return verts, f


def hook_height(tip_size: float) -> float:
    """Hook height at its front: ``min(tip_size, 4 mm)``."""
    return min(tip_size, 0.004)
