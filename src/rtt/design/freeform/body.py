"""The jaw body behind the pad: side profile, collision slices, windows and the root block.

Jaw-local frame, left-jaw convention (see :mod:`rtt.design.freeform.tip`): metres, y away
from the part, z up, root plane ``z = 0``, tip at ``z = z0 < 0``. The pad band is
``[z0, z1]``; the body runs from the band to the root block, which sits just under the root
plane on the carriage interface.

* **Side profile.** The body's inner line ``y_in(z)`` is inactive in the pad band (the pad face
  governs). Above the band it stays :data:`BODY_CLEARANCE` (2 mm) clear of the part wherever
  the part reaches above the band; where it does not, the pad's own shape runs on up the
  finger. The back line ``y_out(z)`` is a smooth S-curve: at least 6 mm behind the deepest
  inner surface at the tip, tapering up to 10 mm at the root, with a gentle belly, every slope
  at most :data:`DESIGN_SLOPE` before smoothing and never above 45 degrees, so it prints
  without supports root-down.
* **Collision.** 8 enclosing convex slices: one behind the pad strips and 7 up the body, each
  the hull of two axis-aligned rectangles grown by the back curve's sagitta. The body is at
  least 2 mm clear of the part, so the coarse slices cost nothing; windows are left out.
* **Print and visual.** A 24-section loft of rounded rectangles (1 mm fillets on the edges that
  cannot touch the part), cut by the pad face, minus the windows and the mount holes.
* **Windows** go through x wherever the side web is at least 8 mm, kept 3 mm behind any
  surface that can touch the part and :data:`WINDOW_SHELL` inside the back.
* **Root block**: the footprint measured on the stock gripper's clamp cut face, 8 mm tall,
  with two M3 holes in its top 6 mm.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from rtt.design.freeform.tip import Piece, piece_from_box, piece_from_sections

ROOT_BLOCK_X = (-0.023, 0.023)  # m; clamp cut-face footprint across the jaw (measured)
ROOT_BLOCK_Y = (0.0004, 0.0164)  # m; from the carriage inner face (0.4 mm at closure), 16 mm deep
ROOT_BLOCK_H = 0.008  # m below the root plane
MOUNT_HOLE_DIAMETER = 0.0032  # m; M3 clearance
MOUNT_HOLE_DEPTH = 0.006  # m from the root plane
MOUNT_HOLE_X = 0.016  # m; +-x of the two holes
MOUNT_HOLE_Y = 0.0084  # m; centre of the footprint depth
BODY_CLEARANCE = 0.002  # m; body above the pad stays this far off the part
THICK_TIP = 0.006  # m behind the deepest inner surface at the tip
THICK_ROOT = 0.010  # m behind it at the root
STRIP_BACK_MARGIN = 0.002  # m; the print reaches at least this far behind the pad strips
DESIGN_SLOPE = 0.6  # dy/dz of the back envelope before the belly and smoothing
MAX_SLOPE = 1.0  # 45 degrees: the printability limit on every body slope
N_BODY_SLICES = 8
N_LOFT = 24
FILLET = 0.001  # m
BODY_OVERLAP = 0.0005  # m; the body loft starts this far inside the pad band (fuses the print)
ROOT_OVERLAP = 0.001  # m; the body loft reaches this far into the root block
WINDOW_MIN_WEB = 0.008  # m; windows only where the side web is at least this thick
WINDOW_FRONT = 0.003  # m behind any surface that can touch the part
WINDOW_MIN_AREA = 4e-6  # m^2; smaller window fragments are dropped


@dataclass(frozen=True)
class WindowStyle:
    shell: float  # m inside the back line
    rib: float  # m rib thickness
    pitch: float  # m between ribs along z
    tilt_deg: float  # rib tilt from horizontal
    alternate: bool  # alternate the tilt (a truss) instead of parallel ribs


WINDOW_STYLES: dict[str, WindowStyle | None] = {
    "solid": None,
    "windowed": WindowStyle(shell=0.0024, rib=0.0022, pitch=0.011, tilt_deg=-18.0, alternate=False),
    "truss": WindowStyle(shell=0.0018, rib=0.0018, pitch=0.009, tilt_deg=40.0, alternate=True),
}


@dataclass(frozen=True, eq=False)
class SideProfile:
    """Sampled side profile of the body (jaw frame, metres)."""

    zs: np.ndarray  # from z0 (tip) up to z_root_join
    y_in: np.ndarray  # body inner line; -inf where the pad face governs
    y_out: np.ndarray  # back line
    z0: float
    z1: float
    face_min: float  # innermost pad point (local y)
    face_max: float  # deepest pad point (local y)
    y_back: float  # back of the pad strips

    def inner(self) -> np.ndarray:
        """The deepest inner surface at each height: ``max(face_max, y_in)``."""
        return np.maximum(self.face_max, self.y_in)

    def front(self) -> np.ndarray:
        """The innermost material at each height: ``max(face_min, y_in)``."""
        return np.maximum(self.face_min, self.y_in)

    def y_out_at(self, z: float | np.ndarray) -> np.ndarray:
        return np.interp(z, self.zs, self.y_out)

    def body_y_in(self) -> float:
        """The constant inner line above the band (``-inf`` if inactive)."""
        above = self.zs > self.z1
        return float(self.y_in[above].max()) if above.any() else -math.inf


def _gauss_smooth(y: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0:
        return y.copy()
    r = math.ceil(3 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.full(r, y[0]), y, np.full(r, y[-1])])
    return np.convolve(pad, k, mode="valid")


def side_profile(
    z0: float,
    z1: float,
    face_min: float,
    face_max: float,
    y_back: float,
    body_y_in: float,
    dz: float = 0.0005,
) -> SideProfile:
    """The body's side profile from the pad band geometry and the inner line above the band.

    ``body_y_in`` is the inner line the body must keep above the band (``-inf`` when the part
    does not reach above the band, or stays narrower than the pad there).
    """
    z_top = -ROOT_BLOCK_H + ROOT_OVERLAP
    n = max(8, math.ceil((z_top - z0) / dz) + 1)
    zs = np.linspace(z0, z_top, n)
    y_in = np.where(zs > z1, body_y_in, -math.inf)
    inner = np.maximum(face_max, y_in)
    s = (zs - z0) / (z_top - z0)
    t = THICK_TIP + (THICK_ROOT - THICK_TIP) * s
    lb = inner + t
    band = zs <= z1 + 1e-12
    lb = np.where(band, np.maximum(lb, y_back + STRIP_BACK_MARGIN), lb)
    lb[-1] = max(lb[-1], ROOT_BLOCK_Y[1])
    step = float(zs[1] - zs[0])
    dist = np.abs(zs[:, None] - zs[None, :])
    env = np.max(lb[None, :] - DESIGN_SLOPE * dist, axis=1)
    length = z_top - z0
    belly = min(0.004, 0.06 * length) * np.sin(math.pi * s)
    smooth = _gauss_smooth(env + belly, 0.002 / step)
    y_out = smooth + max(0.0, float(np.max(lb - smooth)))
    return SideProfile(zs, y_in, y_out, z0, z1, face_min, face_max, y_back)


def max_slope(profile: SideProfile) -> float:
    """Largest |dy/dz| of the back line and of the inner line above the band."""
    dz = np.diff(profile.zs)
    slopes = [float(np.max(np.abs(np.diff(profile.y_out) / dz)))]
    above = profile.zs > profile.z1
    yi = profile.y_in[above]
    if len(yi) > 1 and np.all(np.isfinite(yi)):
        slopes.append(float(np.max(np.abs(np.diff(yi) / np.diff(profile.zs[above])))))
    return max(slopes)


# ----------------------------------------------------------------------------- collision


def _sagitta(zs: np.ndarray, y: np.ndarray, za: float, zb: float) -> float:
    m = (zs >= za) & (zs <= zb)
    if not m.any():
        return 0.0
    ya, yb = np.interp([za, zb], zs, y)
    chord = ya + (yb - ya) * (zs[m] - za) / (zb - za)
    return max(0.0, float(np.max(y[m] - chord)))


def strip_top(profile: SideProfile) -> float:
    """Top of the exact pad strips.

    Where the body's inner line is active above the band, the strips end at the band top and
    the body slices start there, 2 mm clear of the part. Otherwise the pad's own shape runs on
    up the finger; the strips then continue :data:`BODY_CLEARANCE` above the band so the
    coarse body slices (which fill the pad's recesses) start 2 mm above the band top.
    """
    if profile.body_y_in() > profile.face_min:
        return profile.z1
    return min(profile.z1 + BODY_CLEARANCE, -ROOT_BLOCK_H - 0.001)


def body_slices(profile: SideProfile, width: float, chamfer: float = 0.0) -> list[Piece]:
    """The 8 enclosing collision slices: one behind the pad strips, 7 up the body.

    The slice behind the strips overlaps them by 1 mm but never reaches in front of the
    chamfer's bottom edge (``face_max + chamfer``), so the bevel stays clear of the part.
    """
    hw = width / 2.0
    zs, y_out = profile.zs, profile.y_out
    out: list[Piece] = []
    z_strip = strip_top(profile)
    band = zs <= z_strip
    back_band = float(np.max(y_out[band])) if band.any() else float(np.interp(z_strip, zs, y_out))
    behind = max(profile.y_back - 0.001, profile.face_max + chamfer)
    out.append(piece_from_box((-hw, behind, profile.z0), (hw, back_band, z_strip), "body"))
    front = float(np.max(profile.front()[zs > profile.z1])) if (zs > profile.z1).any() else profile.face_min
    edges = np.linspace(z_strip, zs[-1], N_BODY_SLICES)
    for za, zb in itertools.pairwise(edges):
        sag = _sagitta(zs, y_out, za, zb)
        ya, yb = (float(v) + sag for v in np.interp([za, zb], zs, y_out))
        sec = lambda z, yb_: np.array([[-hw, front, z], [hw, front, z], [hw, yb_, z], [-hw, yb_, z]])
        out.append(piece_from_sections(sec(za, ya), sec(zb, yb), "body"))
    return out


def root_block_y_back(profile: SideProfile) -> float:
    return max(ROOT_BLOCK_Y[1], float(profile.y_out[-1]))


def root_block(profile: SideProfile) -> Piece:
    return piece_from_box(
        (ROOT_BLOCK_X[0], ROOT_BLOCK_Y[0], -ROOT_BLOCK_H),
        (ROOT_BLOCK_X[1], root_block_y_back(profile), 0.0),
        "root",
    )


# ----------------------------------------------------------------------------- windows


def window_polygons(profile: SideProfile, style: str) -> list[np.ndarray]:
    """Window outlines in the (y, z) side plane, metres (empty for a solid body)."""
    ws = WINDOW_STYLES.get(style)
    if ws is None:
        return []
    import manifold3d as m3d

    zs = profile.zs
    lo = profile.inner() + WINDOW_FRONT
    hi = profile.y_out - ws.shell
    z_lo = profile.z1 + WINDOW_FRONT
    z_hi = -ROOT_BLOCK_H - 0.002
    ok = (zs >= z_lo) & (zs <= z_hi) & (profile.y_out - profile.inner() >= WINDOW_MIN_WEB)
    if ok.sum() < 3:
        return []
    mm = 1000.0
    regions = []
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        if len(run) < 3:
            continue
        a = np.stack([lo[run], zs[run]], axis=1)
        b = np.stack([hi[run], zs[run]], axis=1)[::-1]
        poly = np.concatenate([a, b]) * mm
        if _area(poly) < 0:
            poly = poly[::-1]
        regions.append(poly)
    if not regions:
        return []
    win = m3d.CrossSection([r for r in regions], m3d.FillRule.Positive)
    z_start, z_end = zs[idx[0]], zs[idx[-1]]
    span = 0.1 * mm
    k = 0
    zc = z_start + ws.pitch / 2.0
    while zc < z_end + ws.pitch:
        tilt = ws.tilt_deg if not (ws.alternate and k % 2) else -ws.tilt_deg
        rib = m3d.CrossSection.square([span, ws.rib * mm], center=True).rotate(tilt)
        rib = rib.translate([float(np.mean(hi[idx])) * mm, zc * mm])
        win = win - rib
        zc += ws.pitch
        k += 1
    r = 0.0009 * mm
    join = m3d.JoinType.Round
    win = win.offset(-r, join, circular_segments=16).offset(r, join, circular_segments=16)
    out = []
    for piece in win.decompose():
        if piece.area() < WINDOW_MIN_AREA * mm * mm:
            continue
        for contour in piece.to_polygons():
            out.append(np.asarray(contour, dtype=np.float64) / mm)
    return out


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ----------------------------------------------------------------------------- loft


def rounded_rect(x0: float, x1: float, y0: float, y1: float, r: float, n_arc: int = 4) -> np.ndarray:
    """CCW rounded rectangle in (x, y), ``4 * (n_arc + 1)`` points."""
    r = max(0.0, min(r, (x1 - x0) / 2.0 - 1e-9, (y1 - y0) / 2.0 - 1e-9))
    a = np.radians(90.0 * np.arange(n_arc + 1) / n_arc)
    centres = np.array([(x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r), (x0 + r, y0 + r)])
    start = np.radians([270.0, 0.0, 90.0, 180.0])[:, None] + a[None, :]
    pts = centres[:, None, :] + r * np.stack([np.cos(start), np.sin(start)], axis=-1)
    return pts.reshape(-1, 2)


def loft_mesh(sections: list[np.ndarray], zs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Closed tube mesh through stacked CCW (x, y) sections of equal size at heights ``zs``."""
    k = len(sections[0])
    m = len(sections)
    verts = np.concatenate([np.column_stack([s, np.full(k, z)]) for s, z in zip(sections, zs, strict=True)])
    i = np.arange(k)
    j = (i + 1) % k
    a0 = (np.arange(m - 1) * k)[:, None]
    b0 = a0 + k
    side = np.stack(
        [np.stack([a0 + i, a0 + j, b0 + j], axis=-1), np.stack([a0 + i, b0 + j, b0 + i], axis=-1)], axis=2
    ).reshape(-1, 3)
    fan = np.arange(1, k - 1)
    top = (m - 1) * k
    bottom = np.stack([np.zeros_like(fan), fan + 1, fan], axis=1)
    cap = np.stack([top + np.zeros_like(fan), top + fan, top + fan + 1], axis=1)
    caps = np.stack([bottom, cap], axis=1).reshape(-1, 3)
    return verts, np.concatenate([side, caps]).astype(np.int64)
