"""Parametric finger generator: ``FingerDesign`` -> exactly 14 oriented boxes.

Frame and units follow :mod:`rtt.design.types`: the left finger's local frame has
its origin at the finger root, the finger extends along -Z, the inner (contact)
face points toward -Y and X is the width direction. The generator additionally
fixes the *inner-face reference plane* at the root to ``y = 0``: the finger body
occupies ``y >= 0`` before the bend, so thickness never changes the grasp gap.
Geometry is emitted in metres.

Box layout (indices into ``FingerGeometry.boxes``; see :data:`BOX_ROLES`)::

    3*i + 0   segment i, flank on the -X side
    3*i + 1   segment i, centre (the core: full width, from the face recess to the back)
    3*i + 2   segment i, flank on the +X side
    12, 13    tip boxes

Segments 0-1 lie before the bend and 2-3 after it (segment 0 is at the root).

Construction, per segment, in a frame ``(x, u, v)`` where ``u`` runs along the
finger toward the tip and ``v`` is the depth from the inner face into the body:

* **Core** (centre element): spans the full width and ``v in [depth, thickness]``.
* **Flanks**: fill the face in front of the core at the two sides.
  ``flat``: no recess, flanks are parked. ``concave_arc``: axis-aligned lips of
  depth ``inner_face_depth`` (a recessed channel; flank angle 0).
  ``v_groove``: flanks rotated about the finger axis so their faces run from the
  side edges down to the centre line at full depth (flank angle
  ``atan(2 * depth / width)``); the outer lip edge is bevelled so the rotated box
  never leaves the finger's width.
* **Taper** scales thickness, width and groove depth linearly from 1 at the root
  to ``taper_ratio`` at the tip; each segment uses the value at its midpoint. The
  inner face stays flat; thickness is removed from the back.
* **Bend**: everything after ``bend_start_frac * length`` is rotated about X by
  ``bend_angle_deg`` (positive bends the tip toward -Y, i.e. inward). The pivot
  is on the convex side of the bend (the back for inward bends, the inner face
  for outward bends), so the two parts meet at a sharp corner with no gap and
  only overlap on the concave side.
* **Tip** (2 boxes; the last part of the finger after the bend):
  ``blunt`` square end cap of length ``tip_size`` (closes any groove);
  ``chamfer`` 45-degree lead-in bevel on the inner corner;
  ``rounded`` both corners cut back into a blunt roof (2-facet rounding);
  ``hook`` a lip protruding inward (along the inner-face normal) by ``tip_size``.

Unused or degenerate elements become tiny "parked" boxes inside the root core, so
every design has exactly 14 boxes. Collision geometry equals visual geometry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rtt.design.schema import FingerDesign
from rtt.design.types import BOXES_PER_FINGER, Box, FingerGeometry

MM = 1e-3
NUM_SEGMENTS = 4
FACE_ELEMENTS = 3

PARK_HALF_MM = 0.1
"""Half size of a parked (unused) box."""
DEGENERATE_MM = 0.05
"""Features smaller than this are treated as absent and their boxes parked."""
HOOK_LIP_MIN_MM = 3.0
"""Minimum hook-lip thickness along the finger (it is 0.5 x tip thickness when larger)."""
LIP_WIDTH_FRAC = 0.2
"""concave_arc lip width as a fraction of the local finger width ..."""
LIP_WIDTH_MIN_MM = 3.0
"""... but at least this wide ..."""
LIP_WIDTH_MAX_FRAC = 0.35
"""... and never wider than this fraction of the local width (the channel keeps >= 30%)."""
TIP_MAX_FRAC_OF_DISTAL = 0.5
"""Tip features occupy at most this fraction of the length after the bend."""
MAX_DEPTH_FRAC = 0.5
"""Face recess is capped at this fraction of the thickness, so even designs that the
validator rejects still produce well-formed boxes."""

BOX_ROLES: tuple[str, ...] = tuple(
    f"segment{i}_{element}"
    for i in range(NUM_SEGMENTS)
    for element in ("flank_neg_x", "centre", "flank_pos_x")
) + ("tip_a", "tip_b")
"""Role of each box index (see the module docstring)."""

CORE_INDICES: tuple[int, ...] = tuple(FACE_ELEMENTS * i + 1 for i in range(NUM_SEGMENTS))
TIP_INDICES: tuple[int, int] = (12, 13)

_E_X = np.array([1.0, 0.0, 0.0])
_SLAB_AXIS_1 = (1.0, 0.0, 0.0)  # box X: finger width
_SLAB_AXIS_2 = (0.0, 0.0, 1.0)  # box Y: depth into the body; box Z = X x Y = toward the root


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------


def quat_from_matrix(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (w, x, y, z) with w >= 0 for a proper rotation matrix."""
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = np.asarray(rot, dtype=float).tolist()
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    norm = math.sqrt(sum(c * c for c in q))
    sign = -1.0 if q[0] < 0.0 else 1.0
    return (sign * q[0] / norm, sign * q[1] / norm, sign * q[2] / norm, sign * q[3] / norm)


def matrix_from_quat(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (not necessarily normalised) quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# Box helpers (public: used by validation, export and tests)
# ---------------------------------------------------------------------------

_CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def _rotations(quats: np.ndarray) -> np.ndarray:
    """Batched rotation matrices, shape (n, 3, 3), from quaternions (n, 4) (w, x, y, z)."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def box_arrays(geom: FingerGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box centres (n, 3), half sizes (n, 3) and rotation matrices (n, 3, 3), metres."""
    centers = np.array([b.center for b in geom.boxes], dtype=float)
    halves = np.array([b.half_size for b in geom.boxes], dtype=float)
    rots = _rotations(np.array([b.quat_wxyz for b in geom.boxes], dtype=float))
    return centers, halves, rots


def box_corners(box: Box) -> np.ndarray:
    """The 8 corners of a box in the finger frame, shape (8, 3), metres."""
    rot = matrix_from_quat(box.quat_wxyz)
    local = _CORNER_SIGNS * np.asarray(box.half_size)
    return np.asarray(box.center) + local @ rot.T


def geometry_corners(geom: FingerGeometry) -> np.ndarray:
    """All box corners of a finger, shape (14 * 8, 3), metres."""
    centers, halves, rots = box_arrays(geom)
    local = _CORNER_SIGNS[None, :, :] * halves[:, None, :]  # (n, 8, 3)
    world = np.einsum("nij,nkj->nki", rots, local) + centers[:, None, :]
    return world.reshape(-1, 3)


def geometry_bounds(geom: FingerGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds (min, max) of the finger in its local frame, metres."""
    corners = geometry_corners(geom)
    return corners.min(axis=0), corners.max(axis=0)


def is_parked(box: Box) -> bool:
    """True for the tiny placeholder boxes that stand in for unused elements."""
    return max(box.half_size) <= PARK_HALF_MM * MM * 1.0001


def point_inside(geom: FingerGeometry, point: Sequence[float], tol: float = 0.0) -> bool:
    """Whether a point (finger frame, metres) lies inside any box (``tol`` shrinks the boxes)."""
    p = np.asarray(point, dtype=float)
    for box in geom.boxes:
        rot = matrix_from_quat(box.quat_wxyz)
        local = rot.T @ (p - np.asarray(box.center))
        if np.all(np.abs(local) <= np.asarray(box.half_size) - tol):
            return True
    return False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    """Segment frame: origin on the inner face; ``e_u`` toward the tip, ``e_v`` into the body."""

    origin: np.ndarray
    e_u: np.ndarray
    e_v: np.ndarray

    @classmethod
    def at(cls, origin: np.ndarray, bend_rad: float) -> _Frame:
        s, c = math.sin(bend_rad), math.cos(bend_rad)
        return cls(np.asarray(origin, dtype=float), np.array([0.0, -s, -c]), np.array([0.0, c, -s]))

    @property
    def basis(self) -> np.ndarray:
        """Columns map (x, u, v) segment coordinates to finger-frame directions."""
        return np.array([_E_X, self.e_u, self.e_v]).T


def _box(
    frame: _Frame,
    center_xuv: Sequence[float],
    axis1_xuv: Sequence[float],
    axis2_xuv: Sequence[float],
    half_mm: Sequence[float],
) -> Box:
    """A box given in segment coordinates; the third axis is ``axis1 x axis2``."""
    basis = frame.basis
    (x1, y1, z1), (x2, y2, z2) = axis1_xuv, axis2_xuv
    axes = np.array(
        [[x1, x2, y1 * z2 - z1 * y2], [y1, y2, z1 * x2 - x1 * z2], [z1, z2, x1 * y2 - y1 * x2]],
        dtype=float,
    )
    rot = basis @ axes
    center = frame.origin + basis @ np.asarray(center_xuv, dtype=float)
    cx, cy, cz = (center * MM).tolist()
    hx, hy, hz = (h * MM for h in half_mm)
    return Box(
        center=(cx, cy, cz), half_size=(float(hx), float(hy), float(hz)), quat_wxyz=quat_from_matrix(rot)
    )


def _slab(frame: _Frame, x0: float, x1: float, u0: float, u1: float, v0: float, v1: float) -> Box:
    """Axis-aligned (in segment coordinates) box spanning the given ranges, mm."""
    center = ((x0 + x1) / 2, (u0 + u1) / 2, (v0 + v1) / 2)
    half = ((x1 - x0) / 2, (v1 - v0) / 2, (u1 - u0) / 2)
    return _box(frame, center, _SLAB_AXIS_1, _SLAB_AXIS_2, half)


def _v_flank(frame: _Frame, side: float, width: float, depth: float, u0: float, u1: float) -> Box:
    """One sloped V-groove flank on the ``side`` (+1 or -1) of the finger.

    The flank face lies on the line from the side edge at the face, ``(side*w/2, 0)``,
    to the centre line at full depth, ``(0, depth)`` in (x, v). The box is just thick
    enough (``depth * cos(alpha)``) to reach the core at the side; its outer end is
    perpendicular to the face, which bevels the lip edge instead of leaving the width.
    """
    half_w = width / 2
    ell = math.hypot(half_w, depth)
    face_dir = np.array([-side * half_w, depth]) / ell  # (x, v): from lip toward centre line
    normal = np.array([side * depth, half_w]) / ell  # (x, v): into the body
    cos_a, sin_a = half_w / ell, depth / ell
    thick = depth * cos_a
    trim = depth * sin_a  # face length lost to the lip bevel
    start = np.array([side * half_w, 0.0]) + trim * face_dir
    apex = np.array([0.0, depth])
    center_xv = (start + apex) / 2 + (thick / 2) * normal
    return _box(
        frame,
        (center_xv[0], (u0 + u1) / 2, center_xv[1]),
        (face_dir[0], 0.0, face_dir[1]),
        (normal[0], 0.0, normal[1]),
        ((ell - trim) / 2, thick / 2, (u1 - u0) / 2),
    )


def _triangle_fill(
    frame: _Frame,
    p1_uv: Sequence[float],
    p2_uv: Sequence[float],
    corner_uv: Sequence[float],
    width: float,
) -> Box:
    """Box whose face is the hypotenuse p1-p2 of a right triangle (right angle at ``corner``).

    The box is as deep as the triangle's altitude, so it covers the triangle exactly;
    the two small extra wedges it adds lie beyond the triangle's legs (inside the
    neighbouring solid by construction of the callers).
    """
    p1, p2, corner = (np.asarray(p, dtype=float) for p in (p1_uv, p2_uv, corner_uv))
    ell = float(np.linalg.norm(p2 - p1))
    t = (p2 - p1) / ell
    n = np.array([-t[1], t[0]])
    if np.dot(corner - p1, n) < 0.0:
        n = -n
    depth = float(np.dot(corner - p1, n))
    center_uv = (p1 + p2) / 2 + (depth / 2) * n
    return _box(
        frame,
        (0.0, center_uv[0], center_uv[1]),
        (1.0, 0.0, 0.0),
        (0.0, n[0], n[1]),
        (width / 2, depth / 2, ell / 2),
    )


@dataclass(frozen=True)
class Section:
    """One material section of the finger whose size is checked against the minimum."""

    size_mm: float
    where: str
    remedy: str


@dataclass(frozen=True)
class FingerBuild:
    """Generated geometry plus the derived quantities that validation reports."""

    geometry: FingerGeometry
    sections: tuple[Section, ...]
    """Material sections (core thickness/width per segment, lips, hook lip)."""
    distal_length_mm: float
    """Length of the finger after the bend."""
    tip_region_mm: float
    """Length along the finger occupied by the tip boxes (0 when they are parked)."""
    tip_feature_mm: float
    """Realised tip feature size (cap length, bevel, rounding, or hook protrusion)."""
    notes: tuple[str, ...]
    """Human-readable notes about parameters that were capped or have no effect."""

    @property
    def min_section(self) -> Section:
        return min(self.sections, key=lambda s: s.size_mm)


def build_finger(design: FingerDesign) -> FingerBuild:
    """Generate the 14 boxes of the left finger plus derived section/tip data."""
    length = design.finger_length_mm
    thick = design.finger_thickness_mm
    width = design.finger_width_mm
    taper = design.taper_ratio
    bend = math.radians(design.bend_angle_deg)
    face = design.inner_face
    depth = 0.0 if face == "flat" else design.inner_face_depth_mm
    notes: list[str] = []
    if depth > MAX_DEPTH_FRAC * thick:
        notes.append(
            f"inner_face_depth_mm={depth:g} is capped at {MAX_DEPTH_FRAC:g} x finger_thickness_mm "
            f"({MAX_DEPTH_FRAC * thick:.2f} mm) in the generated geometry"
        )
        depth = MAX_DEPTH_FRAC * thick

    s_bend = design.bend_start_frac * length
    distal = length - s_bend

    # --- tip region -------------------------------------------------------
    tip = design.tip_profile
    tip_size = design.tip_size_mm
    tip_thick_min = taper * thick  # lower bound on the realised tip thickness
    max_tip_len = TIP_MAX_FRAC_OF_DISTAL * distal
    if tip == "hook":
        has_lip = tip_size >= DEGENERATE_MM
        tip_len = min(max(HOOK_LIP_MIN_MM, 0.5 * tip_thick_min), max_tip_len) if has_lip else 0.0
        feature = tip_size if has_lip else 0.0
        if not has_lip:
            notes.append("tip_profile='hook' with tip_size_mm~0 has no lip; the tip is a plain square end")
    else:
        if tip == "blunt":
            limit, why = max_tip_len, f"half the length after the bend ({max_tip_len:.1f} mm)"
        else:
            limit = min(0.5 * tip_thick_min, max_tip_len)
            why = (
                f"min(half the tip thickness {0.5 * tip_thick_min:.1f} mm, "
                f"half the length after the bend {max_tip_len:.1f} mm)"
            )
        tip_len = min(tip_size, limit)
        if tip_len < DEGENERATE_MM:
            tip_len = 0.0
        feature = tip_len
        if tip_size > limit + 1e-9:
            notes.append(
                f"tip_size_mm={tip_size:g} exceeds what a {tip} tip can use here; "
                f"realised {feature:.1f} mm (capped at {why})"
            )

    # --- segment spans and tapered cross-sections ----------------------------
    distal_body = distal - tip_len
    spans = [(0.0, s_bend / 2), (s_bend / 2, s_bend), (0.0, distal_body / 2), (distal_body / 2, distal_body)]
    mids = [s_bend / 4, 3 * s_bend / 4, s_bend + distal_body / 4, s_bend + 3 * distal_body / 4]
    scale = [1.0 - (1.0 - taper) * m / length for m in mids]
    seg_t = [thick * s for s in scale]
    seg_w = [width * s for s in scale]
    seg_d = [depth * s for s in scale]

    # --- frames: proximal, then distal rotated about a pivot on the convex side
    prox = _Frame.at(np.zeros(3), 0.0)
    joint = prox.origin + s_bend * prox.e_u
    if bend > 0.0:
        pivot = joint + seg_t[2] * prox.e_v  # back corner
        dist = _Frame.at(pivot, bend)
        dist = _Frame(pivot - seg_t[2] * dist.e_v, dist.e_u, dist.e_v)
    else:
        dist = _Frame.at(joint, bend)  # inner-face corner

    park_center = (0.0, s_bend / 4, (seg_d[0] + seg_t[0]) / 2)

    def parked() -> Box:
        return _box(prox, park_center, _SLAB_AXIS_1, _SLAB_AXIS_2, (PARK_HALF_MM,) * 3)

    # --- backbone: 4 segments x (flank -x, centre, flank +x) ----------------
    boxes: list[Box] = []
    sections: list[Section] = []
    for i in range(NUM_SEGMENTS):
        frame = prox if i < 2 else dist
        u0, u1 = spans[i]
        t_i, w_i, d_i = seg_t[i], seg_w[i], seg_d[i]
        recessed = face != "flat" and d_i >= DEGENERATE_MM
        core_v0 = d_i if recessed else 0.0
        core = _slab(frame, -w_i / 2, w_i / 2, u0, u1, core_v0, t_i)
        where = f"segment {i + 1} of 4 from the root"
        if recessed:
            sections.append(
                Section(
                    t_i - d_i,
                    f"{where}: thickness {t_i:.1f} mm minus groove depth {d_i:.1f} mm",
                    "increase finger_thickness_mm or taper_ratio, or reduce inner_face_depth_mm",
                )
            )
        else:
            sections.append(
                Section(
                    t_i, f"{where}: thickness {t_i:.1f} mm", "increase finger_thickness_mm or taper_ratio"
                )
            )
        sections.append(
            Section(w_i, f"{where}: width {w_i:.1f} mm", "increase finger_width_mm or taper_ratio")
        )
        if not recessed:
            boxes += [parked(), core, parked()]
        elif face == "concave_arc":
            lip = min(max(LIP_WIDTH_FRAC * w_i, LIP_WIDTH_MIN_MM), LIP_WIDTH_MAX_FRAC * w_i)
            boxes += [
                _slab(frame, -w_i / 2, -w_i / 2 + lip, u0, u1, 0.0, d_i),
                core,
                _slab(frame, w_i / 2 - lip, w_i / 2, u0, u1, 0.0, d_i),
            ]
            sections.append(
                Section(
                    lip,
                    f"{where}: concave_arc lip width {lip:.1f} mm",
                    "increase finger_width_mm or taper_ratio (lips are 20-35% of the local width)",
                )
            )
        else:  # v_groove
            boxes += [_v_flank(frame, -1.0, w_i, d_i, u0, u1), core, _v_flank(frame, 1.0, w_i, d_i, u0, u1)]

    # --- tip boxes ------------------------------------------------------------
    t3, w3 = seg_t[3], seg_w[3]
    end = distal
    start = distal - tip_len
    if tip_len == 0.0:
        tip_boxes = [parked(), parked()]
    elif tip == "blunt":
        tip_boxes = [_slab(dist, -w3 / 2, w3 / 2, start, end, 0.0, t3), parked()]
    elif tip == "chamfer":
        c = tip_len
        tip_boxes = [
            _slab(dist, -w3 / 2, w3 / 2, start, end, c, t3),
            _triangle_fill(dist, (start, 0.0), (end, c), (start, c), w3),
        ]
    elif tip == "rounded":
        mid = t3 / 2
        tip_boxes = [
            _triangle_fill(dist, (start, 0.0), (end, mid), (start, mid), w3),
            _triangle_fill(dist, (start, t3), (end, mid), (start, mid), w3),
        ]
    else:  # hook
        tip_boxes = [_slab(dist, -w3 / 2, w3 / 2, start, end, -tip_size, t3), parked()]
        sections.append(
            Section(
                tip_len,
                f"hook lip: {tip_len:.1f} mm thick along the finger",
                "lengthen the part after the bend (lower bend_start_frac or increase finger_length_mm)",
            )
        )
    boxes += tip_boxes

    assert len(boxes) == BOXES_PER_FINGER
    return FingerBuild(
        geometry=FingerGeometry(boxes=tuple(boxes)),
        sections=tuple(sections),
        distal_length_mm=distal,
        tip_region_mm=tip_len,
        tip_feature_mm=feature,
        notes=tuple(notes),
    )


def generate_finger(design: FingerDesign) -> FingerGeometry:
    """Deterministically generate the left finger's 14 collision (== visual) boxes."""
    return build_finger(design).geometry


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassProperties:
    """Rigid-body properties of one finger (finger frame, SI units).

    ``principal_moments`` and ``principal_quat_wxyz`` are in MuJoCo's
    ``body_inertia`` / ``body_iquat`` convention (inertia about the COM).
    """

    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_root: np.ndarray
    """3x3 inertia tensor about the finger root origin, kg m^2."""
    inertia_com: np.ndarray
    """3x3 inertia tensor about the centre of mass, kg m^2."""
    principal_moments: tuple[float, float, float]
    principal_quat_wxyz: tuple[float, float, float, float]


def mass_and_inertia(geom: FingerGeometry, density_kg_m3: float | None = None) -> MassProperties:
    """Mass, COM and inertia from the sum of the boxes (overlaps are counted twice).

    Every design is treated the same way, which is what a fair comparison needs; the
    overlaps are small (bend corners, sloped flanks, tip bevels).
    """
    rho = geom.density_kg_m3 if density_kg_m3 is None else density_kg_m3
    centers, halves, rots = box_arrays(geom)
    dims = 2.0 * halves
    masses = rho * np.prod(dims, axis=1)
    sq = dims**2
    local = (masses / 12.0)[:, None] * np.stack(
        [sq[:, 1] + sq[:, 2], sq[:, 0] + sq[:, 2], sq[:, 0] + sq[:, 1]], 1
    )
    rotated = np.einsum("nij,nj,nkj->ik", rots, local, rots)  # sum_n R_n diag(I_n) R_n^T
    eye = np.eye(3)
    r2 = np.einsum("ni,ni->n", centers, centers)
    parallel = np.einsum("n,n->", masses, r2) * eye - np.einsum("n,ni,nj->ij", masses, centers, centers)
    inertia_root = rotated + parallel
    total = float(masses.sum())
    first = masses @ centers
    com = first / total
    inertia_com = inertia_root - total * (np.dot(com, com) * eye - np.outer(com, com))
    inertia_com = 0.5 * (inertia_com + inertia_com.T)
    moments, axes = np.linalg.eigh(inertia_com)
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    inertia_root.setflags(write=False)
    inertia_com.setflags(write=False)
    return MassProperties(
        mass_kg=total,
        com_m=(float(com[0]), float(com[1]), float(com[2])),
        inertia_root=inertia_root,
        inertia_com=inertia_com,
        principal_moments=(float(moments[0]), float(moments[1]), float(moments[2])),
        principal_quat_wxyz=quat_from_matrix(axes),
    )
