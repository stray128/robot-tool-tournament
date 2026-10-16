"""The v2 generator: a recipe (or baseline) compiled against one part into a jaw pair.

Two stages, so the validators can run cheapest first:

* :func:`plan_pair` (a few tens of ms): the part's outline in the pad band for each jaw, the
  strategy face, rib removal, simplification to at most 12 segments, the hook front, the
  locator pin, the seat alignment and the analytic validators.
* :func:`build_from_plan`: the exact convex collision pieces, the side profile, the print
  solid (manifold3d), mass properties, the print STL hash, the build, printability and seat
  validators and the static capture report, assembled into a
  :class:`~rtt.design.freeform.types.FingerPair`.

Frames. Each jaw is planned in the *left-jaw build convention* (the jaw approaches from +u,
its material lies at larger u than its face), the right jaw against the part mirrored across
the closing axis. A planned value ``u`` (part frame, the designed pose) maps to the jaw frame
of :mod:`rtt.design.freeform.types` by ``y = u - gap - Y0`` and ``z = z_table - (tip_height +
finger_length)``, where ``gap`` is the jaw's contact gap and ``Y0 = open_half_gap - q_seat``
the jaw origin's offset from the palm centre at seat. ``Y0`` is shared by both jaws (they are
coupled) and set so that the innermost point of either jaw sits on the carriage inner face
(``y = +0.4 mm``): both pads touch the centred part at the same closure ``q_seat``, and at full
closure without a part the jaws stay 0.8 mm apart. The right jaw is mirrored once (y -> -y)
at the end; nothing downstream mirrors anything.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rtt.design.freeform import body as B
from rtt.design.freeform import validate as V
from rtt.design.freeform.export import (
    JawMesh,
    edge_report,
    inertia6,
    mass_properties,
    stl_sha256,
    weld_print_mesh,
)
from rtt.design.freeform.locator import PinPlan, facing_sockets, pin_sections, plan_pin
from rtt.design.freeform.outline import (
    PartGeometry,
    cell_outline,
    column_max,
    max_y_in_box,
    part_geometry,
    polyline_gap,
)
from rtt.design.freeform.pad import (
    dilate_x,
    face_on_floor,
    lift_polyline,
    pad_face,
    remove_thin_ribs,
    rib_excess,
    simplify_upper,
)
from rtt.design.freeform.tip import (
    HOOK_MIN_PROTRUSION,
    HOOK_OVERLAP,
    MIN_LAND,
    Piece,
    hook_height,
    hook_pieces,
    hooked_front_loft,
    pad_strips,
    piece_from_sections,
)
from rtt.design.freeform.types import (
    CARRIAGE_INNER_Y,
    MAX_PAD_SEGMENTS,
    PLA_DENSITY,
    FingerPair,
    JawGeometry,
)

DX = 1e-4  # m; the pad grid
MIN_STRIP_WALL = 0.003  # m; pad strips reach this far behind the deepest pad point
MIN_LENGTH_OVER_PAD = 0.015  # m; finger length >= pad height + 15 mm
PRINT_ROOT_SETBACK = 5e-5  # m; printed root block front behind the carriage face
_DEC = 7


def _snap(v: float) -> float:
    return float(np.round(v, _DEC))


def _snap_up(v: np.ndarray | float) -> np.ndarray:
    return np.ceil(np.asarray(v, dtype=np.float64) * 10**_DEC - 1e-6) / 10**_DEC


# ----------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class PairSpec:
    """Physical build parameters (metres, radians) of one design, before any part is seen."""

    pad: str
    tip: str
    pad_height: float
    width: float
    depth: float
    draft: float
    clearance: float
    tip_size: float
    grip_height_frac: float | None = 0.5
    tip_height: float | None = None  # explicit pad bottom above the table (overrides the fraction)
    sweep: float = 0.0  # baseline-only uncertainty sweep
    depth_rule: str = "fixed"  # "configurator": depth from the part (imprint baselines)
    locator: str = "none"
    body_style: str = "windowed"

    def id_payload(self) -> str:
        """Content id payload for raw specs (tests and diagnostics; not a search-space point)."""
        import json
        from dataclasses import asdict

        return "spec:" + json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def spec_from_recipe(recipe: Any) -> tuple[PairSpec, tuple[str, ...]]:
    """The canonical recipe as a :class:`PairSpec`, plus the canonicalisation log."""
    from rtt.design.freeform.canon import canonicalize

    canonical, repairs = canonicalize(recipe)
    spec = PairSpec(
        pad=canonical.pad,
        tip=canonical.tip,
        pad_height=canonical.pad_height_mm / 1000.0,
        width=canonical.finger_width_mm / 1000.0,
        depth=float(canonical.pad_depth_mm) / 1000.0,
        draft=math.radians(float(canonical.wall_draft_deg)),
        clearance=float(canonical.clearance_mm) / 1000.0,
        tip_size=float(canonical.tip_size_mm) / 1000.0,
        grip_height_frac=canonical.grip_height_frac,
        locator=canonical.locator,
        body_style=canonical.body_style,
    )
    return spec, repairs


def design_spec(design: Any) -> tuple[PairSpec, tuple[str, ...]]:
    """A :class:`PairSpec` for a recipe, a baseline wrapper or a spec passed through."""
    from rtt.design.freeform.schema import FingerRecipe

    if isinstance(design, PairSpec):
        return design, ()
    if isinstance(design, FingerRecipe):
        return spec_from_recipe(design)
    method = getattr(design, "pair_spec", None)
    if callable(method):
        return method()
    raise TypeError(f"build_pair needs a FingerRecipe or a Baseline, got {type(design).__name__}")


# ----------------------------------------------------------------------------- plans


@dataclass(eq=False)
class JawPlan:
    """One jaw in the build convention at its designed pose (part frame, metres)."""

    side: str
    geometry: PartGeometry
    xs: np.ndarray
    p: np.ndarray  # cell outline over the pad band
    floor: np.ndarray  # what the face must stay at or behind (grown outline, or the outline for flat)
    face: np.ndarray  # strategy face on the grid, after rib removal
    xk: np.ndarray
    yk: np.ndarray  # simplified face (<= 12 segments)
    gap: float  # contact gap: the jaw closes this much past its designed pose to touch
    p_max: float
    feature_excess: float
    hook_k: np.ndarray | None = None
    hook_floor: np.ndarray | None = None
    hook_h: float | None = None
    pin: PinPlan | None = None
    u_min: float = 0.0  # innermost point at seat (part frame u)

    @property
    def face_grid(self) -> np.ndarray:
        return np.interp(self.xs, self.xk, self.yk)

    def front_grid(self) -> np.ndarray:
        """Innermost pad or hook surface at each grid x (designed pose)."""
        f = self.face_grid
        if self.hook_k is not None:
            f = np.minimum(f, np.interp(self.xs, self.xk, self.hook_k))
        return f


@dataclass(eq=False)
class PairPlan:
    spec: PairSpec
    design_id: str
    part: Any
    part_id: str
    task: Any
    height: float
    tip_height: float
    finger_length: float
    left: JawPlan
    right: JawPlan
    y0: float
    q_seat: float
    repairs: list[str] = field(default_factory=list)
    open_clearance: float = 0.0
    pads_gap: float = 0.0

    @property
    def z0(self) -> float:
        return -self.finger_length

    @property
    def z1(self) -> float:
        return -self.finger_length + self.spec.pad_height

    @property
    def z_root_table(self) -> float:
        """Table height of the root plane at the grasp pose."""
        return self.tip_height + self.finger_length


def _grid(width: float) -> np.ndarray:
    n = round(width / DX)
    return np.round(-width / 2.0 + DX * np.arange(n + 1), _DEC)


def _configurator_depth(p_left: np.ndarray, p_right: np.ndarray, task: Any) -> float:
    """``max(4 mm, min(40% of the half-extent along the closing axis, stroke cap))``."""
    pl, pr = float(np.max(p_left[np.isfinite(p_left)])), float(np.max(p_right[np.isfinite(p_right)]))
    half_extent = 0.5 * (pl + pr)
    q_max = V.Q_SEAT_RANGE[1]
    stroke_cap = min(pl, pr) - (task.gripper.open_half_gap - q_max) + CARRIAGE_INNER_Y
    return max(0.004, min(0.4 * half_extent, stroke_cap))


def _floor_gap(gap: float) -> float:
    return float(np.floor(gap * 10**_DEC + 1e-6) / 10**_DEC)  # rounded down: never past contact


def _contact_gap(
    g: PartGeometry, xk: np.ndarray, yk: np.ndarray, band: tuple[float, float], chamfer: float = 0.0
) -> float:
    """How far the face is from first contact with the part, along the closing axis (exact).

    A chamfer bevels the face back by ``(band bottom + chamfer - z)`` below the chamfer top.
    """
    lo, hi = band
    if chamfer <= 0.0:
        return _floor_gap(polyline_gap(g, xk, yk, lo, hi))
    zc = min(lo + chamfer, hi)
    gap = min(polyline_gap(g, xk, yk, zc, hi), polyline_gap(g, xk, yk, lo, zc, kz=-1.0, z_ref=zc))
    return _floor_gap(gap)


def _hook_gap(g: PartGeometry, xk: np.ndarray, hk: np.ndarray, z0: float, hh: float, top: float) -> float:
    """First contact of the hook: its front is vertical up to ``z0 + hh``, then recedes at 45 degrees."""
    zr = z0 + hh
    gap = min(polyline_gap(g, xk, hk, z0, zr), polyline_gap(g, xk, hk, zr, max(zr, top), kz=1.0, z_ref=zr))
    return _floor_gap(gap)


def clearance_floor(
    g: PartGeometry, xs: np.ndarray, z_lo: float, z_hi: float, clearance: float, sweep: float = 0.0
) -> np.ndarray:
    """What a face of the given clearance must stay at or behind, on the grid ``xs``.

    The part's exact outline over the band grown by the clearance in height, taken on a grid
    reaching past the jaw's edges, dilated sideways by the clearance (plus the baseline-only
    sweep) and one more half cell, then moved back by the clearance. A face at or behind it,
    interpolated linearly between grid points, keeps the clearance to the part in every
    direction (its distance is at least the clearance in the max norm, so also in 3D):
    material just above or below the band, or just beyond the jaw's edge, is included.
    """
    reach = clearance + sweep
    k = math.ceil(reach / DX + 0.5 - 1e-9) if reach > 0.0 else 0
    n = len(xs)
    xs_ext = np.round(xs[0] - k * DX + DX * np.arange(n + 2 * k), _DEC)
    p = cell_outline(g, xs_ext, z_lo - clearance, z_hi + clearance)
    return dilate_x(p, k)[k : k + n] + clearance


def _plan_face(
    side: str,
    g: PartGeometry,
    spec: PairSpec,
    xs: np.ndarray,
    p: np.ndarray,
    depth: float,
    band: tuple[float, float],
) -> JawPlan:
    if spec.pad == "flat":  # a plain plane at first contact (contact is allowed)
        floor = p.copy()
        face = pad_face(p, xs, "flat", 0.0, 0.0, 0.0)
    else:
        floor = clearance_floor(g, xs, band[0], band[1], spec.clearance, spec.sweep)
        face = face_on_floor(floor, xs, spec.pad, depth, spec.draft)
    opened = remove_thin_ribs(face, DX)
    xk, yk = simplify_upper(xs, opened, MAX_PAD_SEGMENTS)
    yk = _snap_up(yk)
    fs = np.interp(xs, xk, yk)
    fin = np.isfinite(p)
    chamfer = _snap(spec.tip_size) if spec.tip == "chamfer" else 0.0
    gap = _contact_gap(g, xk, yk, band, chamfer)
    return JawPlan(
        side=side,
        geometry=g,
        xs=xs,
        p=p,
        floor=floor,
        face=opened,
        xk=xk,
        yk=yk,
        gap=gap,
        p_max=float(np.max(p[fin])),
        feature_excess=rib_excess(fs, DX),
    )


def _plan_hook(jaw: JawPlan, spec: PairSpec, tip_height: float) -> None:
    hh = hook_height(spec.tip_size)
    top = tip_height + min(spec.pad_height, hh + spec.tip_size)
    floor = clearance_floor(jaw.geometry, jaw.xs, tip_height, top, spec.clearance)
    hk = lift_polyline(jaw.xs, jaw.xk, jaw.yk - spec.tip_size, floor)
    hk = np.minimum(_snap_up(hk), jaw.yk)
    hk = np.where(jaw.yk - hk < HOOK_MIN_PROTRUSION, jaw.yk, hk)  # no hook piece where it barely protrudes
    jaw.hook_k = hk
    jaw.hook_floor = floor
    jaw.hook_h = hh
    if np.any(hk < jaw.yk):  # the hook can be the first contact (for example under a V)
        reach = tip_height + min(spec.pad_height, hh + spec.tip_size + HOOK_OVERLAP)
        jaw.gap = min(jaw.gap, _hook_gap(jaw.geometry, jaw.xk, hk, tip_height, _snap(hh), reach))


def plan_pair(
    design: Any,
    part: Any,
    task: Any,
    *,
    design_id: str | None = None,
    repairs: tuple[str, ...] = (),
) -> PairPlan:
    """Stage A: outlines, faces, seat alignment and the analytic validators (see the module docstring)."""
    from rtt.design.freeform.hashing import design_id as _design_id

    if design_id is None:
        design_id = _design_id(design)
        spec, repairs = design_spec(design)
    else:
        spec = design if isinstance(design, PairSpec) else design_spec(design)[0]
    part_id = str(getattr(part, "part_id", None) or part.name)
    V.check_width(spec.width, part_id)
    if spec.tip == "hook" and spec.pad_height < hook_height(spec.tip_size) + spec.tip_size - 1e-9:
        raise V.InvalidDesign(
            "schema", "the hook (height + protrusion) does not fit inside the pad band", part_id
        )
    g, gm = part_geometry(part)
    height = float(g.height)
    c = task.controller
    if spec.tip_height is not None:
        tip = float(spec.tip_height)
    else:
        tip = max(float(c.min_tip_clearance), float(spec.grip_height_frac) * height - spec.pad_height / 2.0)
    tip = _snap(tip)
    length = _snap(max(height + float(c.palm_clearance) - tip, spec.pad_height + MIN_LENGTH_OVER_PAD))
    V.check_length(length, part_id)
    xs = _grid(spec.width)
    band = (tip, tip + spec.pad_height)
    p_left = cell_outline(g, xs, band[0], band[1])
    V.check_outline(p_left, part_id, "left", band)
    p_right = cell_outline(gm, xs, band[0], band[1])
    V.check_outline(p_right, part_id, "right", band)
    depth = spec.depth
    notes = list(repairs)
    if spec.depth_rule == "configurator":
        depth = _snap(_configurator_depth(p_left, p_right, task))
        notes.append(f"configurator imprint depth on {part_id}: {depth * 1000:.1f} mm")
    left = _plan_face("left", g, spec, xs, p_left, depth, band)
    right = _plan_face("right", gm, spec, xs, p_right, depth, band)
    if spec.tip == "hook":
        _plan_hook(left, spec, tip)
        _plan_hook(right, spec, tip)
    for jaw in (left, right):
        jaw.u_min = float(np.min(jaw.front_grid())) - jaw.gap
    ohg = float(task.gripper.open_half_gap)
    if spec.locator != "none":
        _plan_pins(spec, part, task, left, right, tip, ohg, notes)
    u_min = min(left.u_min, right.u_min)
    y0 = _snap(u_min - CARRIAGE_INNER_Y)
    q_seat = _snap(ohg - y0)
    V.check_q_seat(q_seat, part_id)
    gap = (left.front_grid() - left.gap) + (right.front_grid() - right.gap)
    V.check_pads_apart(gap, part_id)
    open_clear = min(_open_clearance(j, q_seat, spec.width) for j in (left, right))
    V.check_open_clearance(open_clear, part_id)
    return PairPlan(
        spec=spec,
        design_id=design_id,
        part=part,
        part_id=part_id,
        task=task,
        height=height,
        tip_height=tip,
        finger_length=length,
        left=left,
        right=right,
        y0=y0,
        q_seat=q_seat,
        repairs=notes,
        open_clearance=open_clear,
        pads_gap=float(np.min(gap)),
    )


def _open_clearance(jaw: JawPlan, q_seat: float, width: float) -> float:
    """At full open: the jaw's innermost surface vs the whole part plus the error envelope."""
    e = V.OPEN_ERROR
    k = math.ceil(e / DX - 1e-9)
    xs_ext = np.round(-width / 2.0 - k * DX + DX * np.arange(len(jaw.xs) + 2 * k), _DEC)
    col = column_max(jaw.geometry, xs_ext)
    grown = dilate_x(col, k)[k : k + len(jaw.xs)] + e
    front = jaw.front_grid() - jaw.gap + q_seat
    if jaw.pin is not None:
        front = np.minimum(front, float(jaw.pin.tip[1]) + q_seat)
    fin = np.isfinite(grown)
    if not fin.any():
        return math.inf
    return float(np.min(front[fin] - grown[fin]))


def _plan_pins(
    spec: PairSpec,
    part: Any,
    task: Any,
    left: JawPlan,
    right: JawPlan,
    tip: float,
    ohg: float,
    notes: list[str],
) -> None:
    features = tuple(getattr(part, "features", ()) or ())
    pts = np.concatenate([np.asarray(p, float) for p in part.pieces])
    z_shift = -float(pts[:, 2].min())
    taper = spec.draft if spec.locator == "conical_pin" else 0.0
    clearance = spec.clearance
    u_floor = ohg - V.Q_SEAT_RANGE[1] + CARRIAGE_INNER_Y
    for jaw in (left, right):
        socks = facing_sockets(features, mirrored=jaw.side == "right")
        if not socks:
            notes.append(f"locator inactive on the {jaw.side} jaw: no socket faces it")
            continue
        sock = socks[0]
        mirror_y = -1.0 if jaw.side == "right" else 1.0
        mx = float(sock.origin[0])
        face_u = float(np.interp(mx, jaw.xk, jaw.yk)) - jaw.gap
        mouth_u = float(sock.origin[1]) * mirror_y
        pin, why = plan_pin(
            sock,
            mirrored=jaw.side == "right",
            z_shift=z_shift,
            clearance=clearance,
            taper_rad=taper,
            face_u_at_mouth=face_u,
            max_length=mouth_u - u_floor,
        )
        if pin is None:
            notes.append(f"locator skipped on the {jaw.side} jaw: {why}")
            continue
        r = pin.r_base
        hw = spec.width / 2.0
        mz = float(pin.mouth[2])
        if abs(mx) + r > hw or mz - r < tip or mz + r > tip + spec.pad_height:
            notes.append(
                f"locator skipped on the {jaw.side} jaw: socket {sock.name!r} is not covered by the pad"
            )
            continue
        jaw.pin = pin
        jaw.u_min = min(jaw.u_min, float(pin.tip[1]))


# ----------------------------------------------------------------------------- build


@dataclass(eq=False)
class JawBuild:
    """One jaw built in its own body frame (the right jaw already mirrored)."""

    side: str
    pieces: list[Piece]
    mesh: JawMesh
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    print_sha: str
    profile: B.SideProfile
    y_back: float
    min_strip_wall: float
    min_body_wall: float
    hook_height: float | None
    feature_excess: float
    body_slope: float
    n_window_cuts: int
    xk: np.ndarray
    yk_local: np.ndarray  # face in the build convention, jaw frame
    hook_local: np.ndarray | None

    def jaw_geometry(self) -> JawGeometry:
        return JawGeometry(
            pieces=tuple(p.vertices for p in self.pieces),
            mass=self.mass,
            com=self.com,
            inertia6=inertia6(self.inertia),
            print_mesh_sha=self.print_sha,
        )


@dataclass(eq=False)
class PairBuild:
    pair: FingerPair
    plan: PairPlan
    left: JawBuild
    right: JawBuild
    printability: dict[str, Any]

    @property
    def left_mesh(self) -> JawMesh:
        return self.left.mesh

    @property
    def right_mesh(self) -> JawMesh:
        return self.right.mesh

    def recipe_record(self) -> dict[str, Any]:
        """What ``recipe.json`` in a print pack says about this build."""
        from dataclasses import asdict

        spec = asdict(self.plan.spec)
        return {
            "design_space": "freeform-v2",
            "design_id": self.pair.design_id,
            "part_id": self.pair.part_id,
            "geometry_hash": self.pair.geometry_hash,
            "spec_si": spec,
            "tip_height_m": self.pair.tip_height,
            "finger_length_m": self.pair.finger_length,
            "q_seat_m": self.pair.q_seat,
            "mass_g": {"left": self.left.mass * 1000.0, "right": self.right.mass * 1000.0},
            "print_mesh_sha256": {"left": self.left.print_sha, "right": self.right.print_sha},
            "auto_repairs": list(self.pair.auto_repairs),
            "units": {"stl": "mm", "glb": "m"},
        }


def _local_y(jaw: JawPlan, plan: PairPlan, u: np.ndarray) -> np.ndarray:
    return _snap_up(np.asarray(u) - jaw.gap - plan.y0)


def _build_jaw(plan: PairPlan, jaw: JawPlan) -> tuple[list[Piece], dict[str, Any]]:
    spec = plan.spec
    z0, z1 = _snap(plan.z0), _snap(plan.z1)
    yk = _local_y(jaw, plan, jaw.yk)
    chamfer = _snap(spec.tip_size) if spec.tip == "chamfer" else 0.0
    if chamfer > 0.0 and chamfer > spec.pad_height - MIN_LAND + 1e-12:
        raise V.InvalidDesign("schema", "the chamfer leg is taller than the pad", plan.part_id)
    y_back = _snap(float(np.max(yk)) + max(MIN_STRIP_WALL, chamfer + MIN_LAND))
    band_top = plan.tip_height + spec.pad_height
    body_y_in = -math.inf
    if plan.height > band_top + 1e-9:
        # everything within the body clearance of the body (above the band, across the width
        # and just beyond it) keeps the body 2 mm away in every direction
        reach = spec.width / 2.0 + B.BODY_CLEARANCE
        u_above = max_y_in_box(jaw.geometry, -reach, reach, band_top - B.BODY_CLEARANCE, plan.height)
        if math.isfinite(u_above):
            body_y_in = _snap(u_above - plan.y0 + B.BODY_CLEARANCE)  # at seat
    profile = B.side_profile(z0, z1, float(np.min(yk)), float(np.max(yk)), y_back, body_y_in)
    z_strip = _snap(B.strip_top(profile))
    pieces = pad_strips(jaw.xk, yk, y_back, z0, z_strip, chamfer)
    hook_local = None
    if jaw.hook_k is not None:
        hook_local = np.minimum(_local_y(jaw, plan, jaw.hook_k), yk)
        pieces += hook_pieces(jaw.xk, yk, hook_local, z0, _snap(jaw.hook_h or 0.0))
    pieces += B.body_slices(profile, spec.width, chamfer)
    pieces.append(B.root_block(profile))
    if jaw.pin is not None:
        a, b = pin_sections(jaw.pin)
        shift = np.array([0.0, -plan.y0, -plan.z_root_table])
        pieces.append(piece_from_sections(a + shift, b + shift, "locator", on_grid=False))
    info = {
        "yk": yk,
        "y_back": y_back,
        "chamfer": chamfer,
        "hook_local": hook_local,
        "profile": profile,
        "z_strip": z_strip,
    }
    return pieces, info


def _print_mesh(
    plan: PairPlan, jaw: JawPlan, pieces: list[Piece], info: dict[str, Any]
) -> tuple[JawMesh, int]:
    """The printed solid (manifold3d, built in millimetres), returned in metres.

    One loft from the tip to the root block, cut by the pad face (and the chamfer), cleared in
    front of the body's inner line above the band, minus the windows, plus the hook, the pin
    and the root block, minus the mount holes. Every coordinate is put on a 0.1 um grid in
    millimetres first, so surfaces that coincide by design coincide bit for bit: nearly but
    not exactly coplanar faces are what leave sliver triangles in a boolean result.
    """
    import manifold3d as m3d

    S = 1000.0

    def q(v: Any) -> np.ndarray:
        return np.round(np.asarray(v, dtype=np.float64) * S, 4)

    def qf(v: float) -> float:
        return float(np.round(float(v) * S, 4))

    spec = plan.spec
    hw = qf(spec.width / 2.0)
    profile: B.SideProfile = info["profile"]
    xk, yk = q(jaw.xk), q(info["yk"])
    z0, z1, z_top = qf(profile.z0), qf(profile.z1), qf(profile.zs[-1])

    def face_prism(far: float) -> m3d.Manifold:
        poly = [(xk[0] - 1.0, yk[0]), *zip(xk.tolist(), yk.tolist(), strict=True)]
        poly += [(xk[-1] + 1.0, yk[-1]), (xk[-1] + 1.0, far), (xk[0] - 1.0, far)]
        cs = m3d.CrossSection([np.array(poly, dtype=np.float64)], m3d.FillRule.Positive)
        return m3d.Manifold.extrude(cs, 4.0 - z0).translate([0.0, 0.0, z0 - 2.0])

    far = qf(float(np.max(profile.y_out))) + 50.0
    hook_local = info["hook_local"]
    if hook_local is None:
        face_solid = face_prism(far)
        innermost = profile.face_min
    else:  # the face and the hook as one cutter, so the hook is not a separate solid to fuse
        hv, hf = hooked_front_loft(xk, yk, q(hook_local), z0, qf(jaw.hook_h or 0.0), far, z0 - 2.0, 2.0, 1.0)
        face_solid = m3d.Manifold(m3d.Mesh(vert_properties=hv, tri_verts=hf.astype(np.uint32)))
        innermost = min(profile.face_min, float(np.min(hook_local)))
    front = qf(innermost - 0.003)  # the face cuts the loft clear of its front fillets
    zs = np.concatenate([np.linspace(z0, z1, 4)[:-1], np.linspace(z1, z_top, B.N_LOFT)[1:]])
    zs = np.round(zs, 4)
    y_out = np.round(np.interp(zs / S, profile.zs, profile.y_out) * S, 4)
    secs = [np.round(B.rounded_rect(-hw, hw, front, float(yo), B.FILLET * S), 4) for yo in y_out]
    verts, faces = B.loft_mesh(secs, zs.tolist())
    loft = m3d.Manifold(m3d.Mesh(vert_properties=verts.astype(np.float64), tri_verts=faces.astype(np.uint32)))
    solid = loft ^ face_solid
    if info["chamfer"] > 0.0:
        s = qf(info["chamfer"])
        # a 45-degree bevel: the face moved back by (z0 + s - z) below z0 + s, unchanged above
        shear = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, z0 + s], [0.0, 0.0, 1.0, 0.0]])
        solid = solid ^ face_prism(far + (2.0 - z0) + 10.0).transform(shear)
    y_in = profile.body_y_in()
    if math.isfinite(y_in) and qf(y_in) > front:  # above the band the body stays 2 mm off the part
        clear = m3d.Manifold.cube([2.0 * hw + 4.0, qf(y_in) - front + 2.0, z_top - z1 + 2.0])
        solid = solid - clear.translate([-hw - 2.0, front - 2.0, z1])
    n_win = 0
    win = B.window_polygons(profile, spec.body_style)
    if win:
        cs = m3d.CrossSection([w * S for w in win], m3d.FillRule.EvenOdd)
        n_win = len(cs.decompose())
        perm = np.array([[0.0, 0.0, 1.0, -(hw + 1.0)], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        solid = solid - m3d.Manifold.extrude(cs, 2.0 * hw + 2.0).transform(perm)
    parts = [solid]
    for pc in pieces:
        if pc.role == "locator":
            parts.append(m3d.Manifold.hull_points(pc.vertices * S))
    rx0, rx1 = qf(B.ROOT_BLOCK_X[0]), qf(B.ROOT_BLOCK_X[1])
    # the printed root block's front sits 50 um behind the carriage face, so a pad edge that
    # reaches the carriage face (y = 0.4 mm) never lies exactly in the block's front plane
    ry0, ry1 = qf(B.ROOT_BLOCK_Y[0] + PRINT_ROOT_SETBACK), qf(B.root_block_y_back(profile))
    rh = qf(B.ROOT_BLOCK_H)
    parts.append(m3d.Manifold.cube([rx1 - rx0, ry1 - ry0, rh]).translate([rx0, ry0, -rh]))
    solid = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    holes = []
    r_hole, depth = qf(B.MOUNT_HOLE_DIAMETER) / 2.0, qf(B.MOUNT_HOLE_DEPTH)
    for sx in (-1.0, 1.0):
        h = m3d.Manifold.cylinder(depth + 1.0, r_hole, r_hole, 24)
        holes.append(h.translate([sx * qf(B.MOUNT_HOLE_X), qf(B.MOUNT_HOLE_Y), -depth]))
    solid = solid - m3d.Manifold.batch_boolean(holes, m3d.OpType.Add)
    if solid.status() != m3d.Error.NoError or solid.is_empty():
        raise V.InvalidDesign(
            "build_failed", f"{jaw.side} print solid failed ({solid.status()})", plan.part_id
        )
    mesh = solid.to_mesh64()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3] / S
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return weld_print_mesh(verts, faces), n_win


def _mirror_piece(pc: Piece) -> Piece:
    return Piece(pc.vertices * np.array([1.0, -1.0, 1.0]), pc.faces[:, ::-1].copy(), pc.role)


def _mirror_inertia(t: np.ndarray) -> np.ndarray:
    m = np.diag([1.0, -1.0, 1.0])
    return m @ t @ m


def build_from_plan(plan: PairPlan, *, t_start: float | None = None) -> PairBuild:
    """Stage B: pieces, print, mass, validators and the capture report (module docstring)."""
    from rtt.design.freeform.hashing import geometry_hash
    from rtt.design.freeform.predict import capture_report
    from rtt.design.freeform.printability import check_printability

    t0 = time.perf_counter() if t_start is None else t_start
    jaws: dict[str, JawBuild] = {}
    prints: dict[str, Any] = {}
    for jaw in (plan.left, plan.right):
        try:
            pieces, info = _build_jaw(plan, jaw)
        except V.InvalidDesign:
            raise
        except Exception as exc:  # geometry library failure: report it as a build failure
            raise V.InvalidDesign("build_failed", f"{jaw.side} jaw: {exc}", plan.part_id) from exc
        V.check_piece_count(len(pieces), plan.part_id, jaw.side)
        V.check_convex(pieces, plan.part_id, jaw.side)
        try:
            mesh, n_win = _print_mesh(plan, jaw, pieces, info)
        except V.InvalidDesign:
            raise
        except Exception as exc:
            raise V.InvalidDesign("build_failed", f"{jaw.side} print solid: {exc}", plan.part_id) from exc
        watertight, n_comp = edge_report(mesh.faces)
        V.check_print_manifold(n_comp, watertight, plan.part_id, jaw.side)
        mass, com, inertia = mass_properties(mesh.vertices, mesh.faces, PLA_DENSITY)
        profile: B.SideProfile = info["profile"]
        yk = info["yk"]
        if jaw.side == "right":
            pieces = [_mirror_piece(p) for p in pieces]
            mesh = mesh.mirrored_y()
            com = com * np.array([1.0, -1.0, 1.0])
            inertia = _mirror_inertia(inertia)
        build = JawBuild(
            side=jaw.side,
            pieces=pieces,
            mesh=mesh,
            mass=mass,
            com=com,
            inertia=inertia,
            print_sha=stl_sha256(mesh.vertices, mesh.faces),
            profile=profile,
            y_back=info["y_back"],
            min_strip_wall=float(info["y_back"] - np.max(yk)),
            min_body_wall=float(np.min(profile.y_out - profile.inner())),
            hook_height=jaw.hook_h,
            feature_excess=jaw.feature_excess,
            body_slope=B.max_slope(profile),
            n_window_cuts=n_win,
            xk=jaw.xk,
            yk_local=yk,
            hook_local=info["hook_local"],
        )
        prints[jaw.side] = check_printability(build, plan.part_id, jaw.side)
        jaws[jaw.side] = build
    _seat_checks(plan, jaws)
    left, right = jaws["left"], jaws["right"]
    capture = capture_report(plan, mu=float(plan.task.physics.friction))
    gh = geometry_hash(
        [p.vertices for p in left.pieces], [p.vertices for p in right.pieces], plan.tip_height, plan.q_seat
    )
    pair = FingerPair(
        left=left.jaw_geometry(),
        right=right.jaw_geometry(),
        tip_height=plan.tip_height,
        finger_length=plan.finger_length,
        q_seat=plan.q_seat,
        design_id=plan.design_id,
        part_id=plan.part_id,
        geometry_hash=gh,
        capture=capture,
        build_s=float(time.perf_counter() - t0),
        auto_repairs=tuple(plan.repairs),
    )
    return PairBuild(pair=pair, plan=plan, left=left, right=right, printability=prints)


# ----------------------------------------------------------------------------- seat checks


_PART_MANIFOLDS: dict[int, Any] = {}


def part_manifold(g: PartGeometry) -> Any:
    """The part's pieces as one manifold (mm, table heights), cached per geometry object."""
    import manifold3d as m3d

    key = id(g)
    hit = _PART_MANIFOLDS.get(key)
    if hit is not None and hit[0] is g:
        return hit[1]
    hulls = [m3d.Manifold.hull_points(h.vertices * 1000.0) for h in g.pieces]
    solid = m3d.Manifold.batch_boolean(hulls, m3d.OpType.Add)
    if len(_PART_MANIFOLDS) > 256:
        _PART_MANIFOLDS.clear()
    _PART_MANIFOLDS[key] = (g, solid)
    return solid


def pieces_manifold(pieces: list[Piece], shift: np.ndarray) -> Any:
    import manifold3d as m3d

    hulls = [m3d.Manifold.hull_points((p.vertices + shift) * 1000.0) for p in pieces]
    return m3d.Manifold.batch_boolean(hulls, m3d.OpType.Add)


def _seat_checks(plan: PairPlan, jaws: dict[str, JawBuild]) -> None:
    spec = plan.spec
    for jaw_plan in (plan.left, plan.right):
        V.check_face_behind_outline(jaw_plan.face_grid, jaw_plan.floor, plan.part_id, jaw_plan.side)
        if jaw_plan.hook_k is not None and jaw_plan.hook_floor is not None:
            hook = np.interp(jaw_plan.xs, jaw_plan.xk, jaw_plan.hook_k)
            proud = hook < jaw_plan.face_grid - 1e-9  # only where the hook stands in front of the pad
            floor = np.where(proud, jaw_plan.hook_floor, -np.inf)
            V.check_face_behind_outline(hook, floor, plan.part_id, f"{jaw_plan.side} hook")
    zr = plan.z_root_table
    seat_manifolds = {}
    for side, jaw_plan in (("left", plan.left), ("right", plan.right)):
        build = jaws[side]
        pieces = build.pieces if side == "left" else [_mirror_piece(p) for p in build.pieces]
        # designed pose, part frame (build convention): u = y + Y0 + gap, z_table = z + root height
        designed = pieces_manifold(pieces, np.array([0.0, plan.y0 + jaw_plan.gap, zr]))
        part = part_manifold(jaw_plan.geometry)
        if spec.pad == "flat":
            overlap = (designed ^ part).volume()
            if overlap > 1e-3:  # mm^3
                V.check_seat_distance(-1.0, 0.0, plan.part_id, side)
        else:
            dist = designed.min_gap(part, spec.clearance * 1000.0 + 1.0) / 1000.0
            V.check_seat_distance(dist, spec.clearance, plan.part_id, side)
        seat = pieces_manifold(pieces, np.array([0.0, plan.y0, zr]))
        if side == "right":
            seat = seat.mirror([0.0, 1.0, 0.0])
        seat_manifolds[side] = seat
        min_z = min(float(p.vertices[:, 2].min()) for p in pieces) + zr
        V.check_table(min_z, plan.part_id)
    apart = seat_manifolds["left"].min_gap(seat_manifolds["right"], 5.0) / 1000.0
    V.check_jaws_apart(apart, plan.part_id)


# ----------------------------------------------------------------------------- public


_CACHE: dict[tuple[str, str, str, str], PairBuild] = {}
_CACHE_MAX = 512


def _gripper_digest(task: Any) -> str:
    import hashlib

    g, c = task.gripper, task.controller
    key = (
        g.open_half_gap,
        g.stroke,
        tuple(g.palm_half_size),
        c.palm_clearance,
        c.min_tip_clearance,
        task.physics.friction,
    )
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def build_pair_full(design: Any, part: Any, task: Any, *, use_cache: bool = True) -> PairBuild:
    """Plan and build one (design, part) pair; raises :class:`~rtt.design.freeform.validate.InvalidDesign`.

    Cached per (generator version, design id, part geometry sha, gripper digest).
    """
    from rtt.design.freeform.hashing import GENERATOR_VERSION
    from rtt.design.freeform.hashing import design_id as _design_id

    if design is None or part is None or task is None:
        raise TypeError("build_pair needs a design, a part and a task")
    if getattr(design, "kind", None) == "stock":
        raise TypeError("the stock baseline has no generator build; use build_pair")
    did = _design_id(design)
    key = (GENERATOR_VERSION, did, str(getattr(part, "geometry_sha256", id(part))), _gripper_digest(task))
    if use_cache and key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    spec, repairs = design_spec(design)
    plan = plan_pair(spec, part, task, design_id=did, repairs=repairs)
    build = build_from_plan(plan, t_start=t0)
    if use_cache:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = build
    return build


def build_pair(design: Any, part: Any, task: Any) -> FingerPair:
    """Compile a recipe (or a baseline) against one part (:func:`rtt.design.freeform.build_pair`).

    The stock baseline is not generated: it is the stock blade's committed hulls (or the v1
    box approximation), placed by :func:`rtt.design.freeform.baselines.build_stock_pair`.
    """
    if getattr(design, "kind", None) == "stock":
        from rtt.design.freeform.baselines import build_stock_pair

        return build_stock_pair(design, part, task)
    return build_pair_full(design, part, task).pair
