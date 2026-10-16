"""Validators for v2 designs: identical code for every arm, cheapest first.

1. **Schema**: bounds, strategy rules and quantisation (pydantic, :mod:`schema`).
2. **Analytic**, per training part, before any piece is built: the outline in the pad band is
   not empty; at full open the jaws clear the part by at least 3 mm per side including a
   3 mm placement-error envelope; at seat the pads stay at least 1 mm apart; ``q_seat`` is in
   [2, 40] mm; the derived length is at most 110 mm with the palm at least 10 mm above the
   part; the finger is at most 40 mm wide.
3. **Build**: succeeds on every part; every piece is convex; at most 40 pieces per jaw; the
   print is a closed manifold with one component. Build time is recorded (and reported when it
   exceeds 1 s per pair) but never rejects a design, so validity is deterministic.
4. **Printability** (:mod:`printability`): walls, features, mass, body slope, overhang share.
5. **Seat**: the pad never cuts into the part (face at or behind outline + clearance on the
   grid); the jaw keeps ``clearance - 0.02 mm`` off the part at its designed pose (contact is
   allowed for ``flat`` only); the two jaws do not intersect at seat; the jaws stay at least
   2 mm above the table.
6. **Duplicates**: ``duplicate_geometry``, then the diversity rule (:mod:`distance`).

A rejection raises :class:`InvalidDesign` with a stable ``code`` and a message that names the
part and the numbers, e.g. ``q_seat_range: part 'hex_standoff': seat closure 41.2 mm is
outside [2, 40] mm``. :func:`validate_design` runs the whole battery over a part suite.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

#: Every rejection code, with what it means. Messages always add the part and the numbers.
CODES: Final[dict[str, str]] = {
    "schema": "the recipe breaks a bound, a strategy rule or the schema",
    "outline_empty": "the part has no material in the pad band",
    "open_clearance": "at full open the jaws do not clear the part plus the error envelope",
    "pads_overlap": "at seat the two pads come closer than 1 mm",
    "q_seat_range": "the seat closure is outside the gripper's usable stroke",
    "length": "the derived finger length is too long",
    "width": "the finger is wider than the tool envelope",
    "build_failed": "the generator could not build the jaws",
    "not_convex": "a collision piece is not convex",
    "too_many_pieces": "a jaw has more collision pieces than allowed",
    "print_not_manifold": "the print solid is not one closed manifold",
    "min_wall": "a wall is thinner than 1.2 mm",
    "min_feature": "a pad feature is narrower than 1.6 mm",
    "mass": "a jaw is heavier than 50 g of solid PLA",
    "body_slope": "the body slope is steeper than 45 degrees",
    "pad_cuts_part": "the pad face lies in front of the part outline plus clearance",
    "seat_distance": "the jaw comes closer to the part than its clearance",
    "jaws_intersect": "the two jaws intersect at seat",
    "table_clearance": "a jaw comes within 2 mm of the table",
    "duplicate_geometry": "the compiled geometry equals an earlier design's",
    "too_similar": "the recipe is within the diversity threshold of an earlier design",
}

OPEN_ERROR: Final = 0.003  # m; placement-error envelope for the open-clearance check
OPEN_CLEARANCE: Final = 0.003  # m per side
SEAT_GAP_MIN: Final = 0.001  # m between the pads at seat
Q_SEAT_RANGE: Final = (0.002, 0.040)  # m
MAX_LENGTH: Final = 0.110  # m
MAX_WIDTH: Final = 0.040  # m
MAX_PIECES: Final = 40
MAX_BUILD_S: Final = 1.0
CONVEX_RTOL: Final = 1e-9
SEAT_DISTANCE_TOL: Final = 0.00002  # m
TABLE_CLEARANCE: Final = 0.002  # m
FACE_TOL: Final = 1e-9  # m; grid comparisons


class InvalidDesign(ValueError):
    """A design that fails a validator. ``code`` is one of :data:`CODES`."""

    def __init__(self, code: str, message: str, part_id: str | None = None) -> None:
        if code not in CODES:
            raise ValueError(f"unknown validator code {code!r}")
        self.code = code
        self.part_id = part_id
        where = f"part {part_id!r}: " if part_id else ""
        super().__init__(f"{code}: {where}{message}")


def mm(v: float) -> str:
    return f"{v * 1000.0:.2f} mm"


# ----------------------------------------------------------------------------- analytic


def check_outline(p: np.ndarray, part_id: str, jaw: str, band: tuple[float, float]) -> None:
    if not np.isfinite(p).any():
        raise InvalidDesign(
            "outline_empty",
            f"no material between {mm(band[0])} and {mm(band[1])} above the table on the {jaw} jaw's "
            f"side; move the pad (grip_height_frac, pad_height_mm)",
            part_id,
        )


def check_q_seat(q_seat: float, part_id: str) -> None:
    lo, hi = Q_SEAT_RANGE
    if not lo - 1e-12 <= q_seat <= hi + 1e-12:
        hint = (
            "the pads reach too far in (reduce pad_depth_mm, tip_size_mm, or raise wall_draft_deg)"
            if q_seat > hi
            else "the part is too wide for the opening"
        )
        raise InvalidDesign(
            "q_seat_range", f"seat closure {mm(q_seat)} is outside [{mm(lo)}, {mm(hi)}]: {hint}", part_id
        )


def check_pads_apart(gap: np.ndarray, part_id: str) -> None:
    g = float(np.min(gap))
    if g < SEAT_GAP_MIN - FACE_TOL:
        raise InvalidDesign(
            "pads_overlap",
            f"at seat the pads come {mm(g)} apart (minimum {mm(SEAT_GAP_MIN)}); reduce pad_depth_mm or "
            f"tip_size_mm",
            part_id,
        )


def check_open_clearance(clear: float, part_id: str) -> None:
    if clear < OPEN_CLEARANCE - FACE_TOL:
        raise InvalidDesign(
            "open_clearance",
            f"at full open a jaw clears the part plus a {mm(OPEN_ERROR)} error envelope by {mm(clear)} "
            f"(minimum {mm(OPEN_CLEARANCE)})",
            part_id,
        )


def check_length(length: float, part_id: str) -> None:
    if length > MAX_LENGTH + 1e-12:
        raise InvalidDesign(
            "length",
            f"derived finger length {mm(length)} exceeds {mm(MAX_LENGTH)} (palm 10 mm above the part); "
            f"grip higher",
            part_id,
        )


def check_width(width: float, part_id: str | None) -> None:
    if width > MAX_WIDTH + 1e-12:
        raise InvalidDesign("width", f"finger width {mm(width)} exceeds {mm(MAX_WIDTH)}", part_id)


# ----------------------------------------------------------------------------- build


def check_convex(pieces: Sequence[Any], part_id: str, jaw: str) -> None:
    from scipy.spatial import ConvexHull, QhullError

    from rtt.design.freeform.tip import piece_volume

    for i, pc in enumerate(pieces):
        vol = piece_volume(pc)
        try:
            hull = ConvexHull(pc.vertices).volume
        except QhullError:  # a flat piece has no hull volume: it is not a convex solid
            hull = 0.0
        rtol = CONVEX_RTOL
        if vol <= 0 or abs(hull - vol) > rtol * hull:
            raise InvalidDesign(
                "not_convex",
                f"{jaw} jaw piece {i} ({pc.role}): hull volume {hull:.6e} m^3 vs piece volume {vol:.6e} m^3",
                part_id,
            )


def check_piece_count(n: int, part_id: str, jaw: str) -> None:
    if n > MAX_PIECES:
        raise InvalidDesign(
            "too_many_pieces", f"{jaw} jaw has {n} collision pieces (maximum {MAX_PIECES})", part_id
        )


def check_print_manifold(n_components: int, watertight: bool, part_id: str, jaw: str) -> None:
    if n_components != 1 or not watertight:
        raise InvalidDesign(
            "print_not_manifold",
            f"{jaw} jaw print has {n_components} components (watertight: {watertight}); expected one closed solid",
            part_id,
        )


# ----------------------------------------------------------------------------- seat


def check_face_behind_outline(face: np.ndarray, floor: np.ndarray, part_id: str, jaw: str) -> None:
    fin = np.isfinite(floor)
    if not fin.any():
        return
    worst = float(np.min(face[fin] - floor[fin]))
    if worst < -FACE_TOL:
        raise InvalidDesign(
            "pad_cuts_part",
            f"{jaw} pad face lies {mm(-worst)} in front of the part outline plus clearance",
            part_id,
        )


def check_seat_distance(distance: float, required: float, part_id: str, jaw: str) -> None:
    if distance < required - SEAT_DISTANCE_TOL:
        raise InvalidDesign(
            "seat_distance",
            f"{jaw} jaw comes {mm(distance)} from the part at its designed pose (needs {mm(required)} "
            f"- 0.02 mm)",
            part_id,
        )


def check_jaws_apart(distance: float, part_id: str) -> None:
    if distance <= 0.0:
        raise InvalidDesign("jaws_intersect", "the two jaws intersect at seat", part_id)


def check_table(min_z: float, part_id: str) -> None:
    if min_z < TABLE_CLEARANCE - 1e-12:
        raise InvalidDesign(
            "table_clearance",
            f"a jaw reaches {mm(min_z)} above the table (minimum {mm(TABLE_CLEARANCE)})",
            part_id,
        )


# ----------------------------------------------------------------------------- suite


@dataclass(frozen=True)
class DesignReport:
    """Result of :func:`validate_design` over a part suite."""

    valid: bool
    design_id: str
    code: str | None = None
    message: str | None = None
    part_id: str | None = None
    pairs: tuple[Any, ...] = ()
    geometry_hashes: dict[str, str] = field(default_factory=dict)
    geometry_id: str | None = None
    build_s: float = 0.0
    slow_builds: tuple[str, ...] = ()
    printability: dict[str, Any] = field(default_factory=dict)


def validate_design(
    design: Any,
    parts: Sequence[Any],
    task: Any,
    *,
    seen_geometry_ids: Iterable[str] = (),
) -> DesignReport:
    """Run every validator, cheapest first, over ``parts`` (all must pass).

    ``design`` is a :class:`~rtt.design.freeform.schema.FingerRecipe`, a baseline, or a raw
    mapping of recipe fields (validated by the schema first). Stage order: schema -> analytic on every part -> build, printability and seat on every
    part -> duplicate geometry. The first failure is returned (never raised).
    """
    from rtt.design.freeform.generate import build_from_plan, design_spec, plan_pair
    from rtt.design.freeform.hashing import design_id as _design_id
    from rtt.design.freeform.hashing import geometry_id as _geometry_id

    t0 = time.perf_counter()
    try:
        if isinstance(design, Mapping):  # a raw proposal (for example a tool call's arguments)
            from rtt.design.freeform.schema import FingerRecipe

            design = FingerRecipe(**design)
        did = _design_id(design)
        spec, repairs = design_spec(design)
    except InvalidDesign as exc:
        return DesignReport(False, "", exc.code, str(exc), exc.part_id)
    except (ValueError, TypeError) as exc:
        return DesignReport(False, "", "schema", f"schema: {exc}")
    plans = []
    try:
        for part in parts:
            plans.append(plan_pair(spec, part, task, design_id=did, repairs=repairs))
        builds = []
        slow = []
        for plan in plans:
            tb = time.perf_counter()
            builds.append(build_from_plan(plan))
            if time.perf_counter() - tb > MAX_BUILD_S:
                slow.append(plan.part_id)
    except InvalidDesign as exc:
        return DesignReport(False, did, exc.code, str(exc), exc.part_id, build_s=time.perf_counter() - t0)
    hashes = {b.pair.part_id: b.pair.geometry_hash for b in builds}
    gid = _geometry_id(hashes)
    if gid in set(seen_geometry_ids):
        return DesignReport(
            False,
            did,
            "duplicate_geometry",
            f"duplicate_geometry: geometry id {gid[:12]} equals an earlier design's",
            None,
            geometry_hashes=hashes,
            geometry_id=gid,
            build_s=time.perf_counter() - t0,
        )
    prints = {b.pair.part_id: b.printability for b in builds}
    return DesignReport(
        True,
        did,
        pairs=tuple(b.pair for b in builds),
        geometry_hashes=hashes,
        geometry_id=gid,
        build_s=time.perf_counter() - t0,
        slow_builds=tuple(slow),
        printability=prints,
    )
