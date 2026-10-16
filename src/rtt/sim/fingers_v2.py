"""Freeform v2 jaw pairs in the simulator: how a :class:`FingerPair` becomes two finger bodies.

The frozen contract (:mod:`rtt.design.freeform.types`) describes each jaw in its own mounted
body frame, so the scene never mirrors anything for a v2 design:

* the left jaw's pieces go on ``finger_left`` exactly as given, the right jaw's pieces on
  ``finger_right`` exactly as given (the generator already built the right jaw against the
  mirrored part and mirrored it once);
* each body gets its own jaw's mass, centre of mass and inertia (``JawGeometry.mass``,
  ``com``, ``inertia6``), so asymmetric jaws have different mass properties;
* every piece is one convex mesh geom (MuJoCo takes the convex hull of the vertices, which
  for an exact convex piece is the piece itself).

A v1 entrant carried in a pair (``FingerPair.boxes`` set) keeps the v1 meaning: the left
finger's boxes as authored, the right finger's mirrored ``y -> -y`` by the scene, as box geoms
(MuJoCo's dedicated box colliders). Mass properties still come from the two jaws.

:func:`v1_pair` wraps a v1 :class:`~rtt.design.types.FingerGeometry` as such a pair (for
registration on v1 boxes and the box-approximation fallback of the stock entrant). Its seat
closure ``q_seat`` is probed in the scored scene (:func:`rtt.sim.register.probe_first_contact`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from rtt.design.freeform.types import CaptureReport, FingerPair, JawGeometry
from rtt.design.types import Box, FingerGeometry
from rtt.sim.scene import (
    MAX_FINGER_EXTENT,
    MassProperties,
    box_corners,
    boxes_mass_properties,
    finger_tip_depth,
    mirror_box,
)

V1_DESIGN_SPACE = "box-v1"
_MIRROR = np.diag([1.0, -1.0, 1.0])
_QUANTUM = 1e-7  # m; vertex quantisation of geometry hashes (the generator's rounding)
CONVEX_TOL = 1e-6  # m; a vertex deeper than this inside its piece's hull makes the piece concave


@dataclass(frozen=True)
class JawBody:
    """What the scene adds for one jaw: its collision pieces (or v1 boxes) and mass properties."""

    pieces: tuple[np.ndarray, ...]  # convex vertex sets in the body frame (empty when boxes are used)
    boxes: tuple[Box, ...]  # v1 box geoms in the body frame (empty for mesh pieces)
    props: MassProperties

    @property
    def n_geoms(self) -> int:
        return len(self.boxes) or len(self.pieces)


def jaw_mass_properties(jaw: JawGeometry) -> MassProperties:
    """The jaw's mass properties as a :class:`~rtt.sim.scene.MassProperties` (inertia about the COM)."""
    ixx, iyy, izz, ixy, ixz, iyz = (float(v) for v in jaw.inertia6)
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return MassProperties(mass=float(jaw.mass), com=np.array(jaw.com, dtype=np.float64), inertia=inertia)


def check_pair(pair: FingerPair) -> None:
    """Everything the scene refuses in a pair, as a ``ValueError``: geometry that is not in
    metres (:func:`check_pair_units`) or a collision piece that is not convex
    (:func:`check_pair_pieces`)."""
    check_pair_units(pair)
    check_pair_pieces(pair)


def jaw_bodies(pair: FingerPair) -> tuple[JawBody, JawBody]:
    """``(left, right)`` scene bodies of a pair, after :func:`check_pair`."""
    check_pair(pair)
    left_props, right_props = jaw_mass_properties(pair.left), jaw_mass_properties(pair.right)
    if pair.boxes is not None:
        boxes = tuple(pair.boxes)
        return (
            JawBody(pieces=(), boxes=boxes, props=left_props),
            JawBody(pieces=(), boxes=tuple(mirror_box(b) for b in boxes), props=right_props),
        )
    return (
        JawBody(pieces=pair.left.pieces, boxes=(), props=left_props),
        JawBody(pieces=pair.right.pieces, boxes=(), props=right_props),
    )


def check_pair_units(pair: FingerPair) -> None:
    """Reject jaw geometry that is clearly not in metres (e.g. authored in millimetres)."""
    for side, jaw in (("left", pair.left), ("right", pair.right)):
        extent = float(np.abs(jaw.bounds).max())
        if not np.isfinite(extent) or extent > MAX_FINGER_EXTENT:
            raise ValueError(
                f"{side} jaw reaches {extent:g} from its root; sizes must be in metres "
                f"(at most {MAX_FINGER_EXTENT} m)"
            )


def check_convex_piece(verts: np.ndarray, where: str, tol: float = CONVEX_TOL) -> None:
    """Reject a vertex set that is not a convex piece: flat (no volume), or with a vertex that
    is not on the edges of its own convex hull.

    MuJoCo collides with the convex hull of a mesh geom's vertices. A non-convex piece (an L
    or a notched strip, say) would be filled in silently, and a pad built that way would cut
    into the part it was shaped around. In a convex piece every vertex is a corner of the hull
    or lies on one of its edges (a redundant point). A vertex inside the hull, or inside one
    of its faces (the inner corner of an extruded L lies inside the cap face), means the piece
    was concave there. ``tol`` (m) is the distance within which a point counts as on a plane.
    """
    from scipy.spatial import ConvexHull, QhullError

    pts = np.asarray(verts, dtype=np.float64)
    try:
        hull = ConvexHull(pts)
    except QhullError:
        raise ValueError(f"{where} is flat or degenerate: it has no convex volume") from None
    corners = {int(i) for i in hull.vertices}
    if len(corners) == len(pts):
        return
    # one plane per hull face (Qhull splits a face into coplanar triangles)
    planes: list[np.ndarray] = []
    for eq in hull.equations:
        if not any(abs(float(eq[:3] @ p[:3]) - 1.0) < 1e-9 and abs(eq[3] - p[3]) < tol for p in planes):
            planes.append(eq)
    eqs = np.asarray(planes)
    for i in range(len(pts)):
        if i in corners:
            continue
        signed = pts[i] @ eqs[:, :3].T + eqs[:, 3]  # <= 0 inside
        on = int(np.sum(np.abs(signed) <= tol))
        if on < 2:
            place = (
                f"{-float(signed.max()) * 1000:.3f} mm inside the hull of the others"
                if on == 0
                else "inside a face of the hull of the others"
            )
            raise ValueError(
                f"{where} is not convex: vertex {i} lies {place}, and MuJoCo would collide with "
                "the filled-in hull"
            )


def check_pair_pieces(pair: FingerPair) -> None:
    """Check every collision piece of both jaws with :func:`check_convex_piece`. A pair that
    carries v1 boxes is skipped: its geoms are MuJoCo boxes, convex by construction."""
    if pair.boxes is not None:
        return
    for side, jaw in (("left", pair.left), ("right", pair.right)):
        for i, piece in enumerate(jaw.pieces):
            check_convex_piece(piece, f"{side} jaw piece {i}")


def pair_mass_g(pair: FingerPair) -> float:
    """Mass of both jaws in grams (the v2 tie-break uses the design's mean over parts)."""
    return 1000.0 * (pair.left.mass + pair.right.mass)


def lowest_point(pair: FingerPair) -> float:
    """Depth (m, > 0) of the lowest point of either jaw below the root plane."""
    return float(-min(pair.left.bounds[0, 2], pair.right.bounds[0, 2]))


# --------------------------------------------------------------------------- hashing


def pieces_hash(*jaws: tuple[np.ndarray, ...]) -> str:
    """SHA-256 (hex) of convex pieces quantised to 1e-7 m, jaw by jaw and piece by piece."""
    h = hashlib.sha256()
    for pieces in jaws:
        h.update(np.asarray([len(pieces)], dtype="<i8").tobytes())
        for p in pieces:
            q = np.round(np.asarray(p, dtype=np.float64) / _QUANTUM).astype("<i8")
            h.update(np.asarray(q.shape, dtype="<i8").tobytes())
            h.update(q.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- v1 entrants as pairs


def _mirror_props(props: MassProperties) -> MassProperties:
    return MassProperties(mass=props.mass, com=_MIRROR @ props.com, inertia=_MIRROR @ props.inertia @ _MIRROR)


def _jaw(boxes: tuple[Box, ...], props: MassProperties, sha: str) -> JawGeometry:
    i = props.inertia
    return JawGeometry(
        pieces=tuple(box_corners(b) for b in boxes),
        mass=props.mass,
        com=props.com,
        inertia6=np.array([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]),
        print_mesh_sha=sha,
    )


def v1_design_id(finger: FingerGeometry) -> str:
    """64-hex id of raw v1 finger geometry (boxes and density)."""
    payload = json.dumps(
        {
            "space": V1_DESIGN_SPACE,
            "boxes": [asdict(b) for b in finger.boxes],
            "density": finger.density_kg_m3,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def v1_pair(
    finger: FingerGeometry,
    *,
    part_id: str,
    tip_height: float,
    q_seat: float,
    mu: float = 0.3,
    design_id: str | None = None,
) -> FingerPair:
    """A v1 box finger as a :class:`FingerPair` (boxes kept, right jaw mirrored by the scene).

    ``tip_height`` is where the controller puts the jaws' lowest point (a v1 finger has no
    designed grip height) and ``q_seat`` its seat closure on the part (probe it with
    :func:`rtt.sim.register.probe_first_contact`). There is no printed STL for a v1 finger, so
    ``print_mesh_sha`` is the SHA-256 of the box description, and the capture report is empty.
    """
    did = design_id or v1_design_id(finger)
    left = boxes_mass_properties(finger.boxes, finger.density_kg_m3)
    mirrored = tuple(mirror_box(b) for b in finger.boxes)
    left_jaw = _jaw(finger.boxes, left, did)
    right_jaw = _jaw(mirrored, _mirror_props(left), did)
    return FingerPair(
        left=left_jaw,
        right=right_jaw,
        tip_height=tip_height,
        finger_length=finger_tip_depth(finger),
        q_seat=q_seat,
        design_id=did,
        part_id=part_id,
        geometry_hash=pieces_hash(left_jaw.pieces, right_jaw.pieces),
        capture=CaptureReport(no_jam_mm=0.0, centring_mm=0.0, mu=mu),
        build_s=0.0,
        auto_repairs=("v1 box finger wrapped as a pair; no capture prediction",),
        boxes=finger.boxes,
    )
