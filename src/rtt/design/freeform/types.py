"""Frozen v2 contract between the freeform jaw generator, the simulator and the renderer.

A v2 design is a part-relative recipe. :func:`build_pair` compiles it against one part into
a :class:`FingerPair`: a left and a right :class:`JawGeometry`, each made of exact convex
pieces, plus the numbers the scene, the scorer and the replay need. Nothing here imports
MuJoCo, trimesh or manifold3d, so every consumer can use these types cheaply.

Units are SI throughout (metres, kilograms, seconds); arrays are float64.

Jaw frame
---------
Each jaw is described in its own finger body frame, exactly as it is mounted:

* Axes are parallel to the palm frame: x across the jaw width, y along the closing axis,
  z up. The jaw hangs below its root along -z.
* z = 0 is the root plane, the carriage interface, at palm z = -12 mm
  (:data:`ROOT_PLANE_PALM_Z`, the underside of the 24 mm scored palm box).
* At full closure the body origin lies on the palm centre plane (palm y = 0), and the
  carriage's inner face sits at y = +0.4 mm for the left jaw (:data:`CARRIAGE_INNER_Y`) and
  at y = -0.4 mm for the right jaw. The scene's closure joint ``q`` (0 = fully open, the
  task stroke = fully closed) puts the left origin at palm y = ``open_half_gap - q`` and the
  right origin at ``-(open_half_gap - q)``, as for v1 fingers.
* The left jaw sits on the +y side and its contact side faces -y; the right jaw sits on the
  -y side and faces +y. The generator builds the right jaw against the part mirrored in y and
  mirrors the result once (pieces, centre of mass and inertia) before returning it, so the
  scene, the renderer and the print pack never mirror anything. The right jaw always has its
  own geometry and its own assets.

This differs from the v1 contract in :mod:`rtt.design.types`, where only the left finger is
emitted and the scene mirrors it. v1 entrants carried in a :class:`FingerPair` keep their
v1 boxes in :attr:`FingerPair.boxes` with the v1 meaning.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # implemented by the other v2 modules; never imported at run time here
    from rtt.design.freeform.baselines import Baseline
    from rtt.design.freeform.schema import FingerRecipe
    from rtt.design.types import Box
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

DESIGN_SPACE = "freeform-v2"
ROOT_PLANE_PALM_Z = -0.012  # m; palm z of every jaw's root plane (z = 0 in the jaw frame)
CARRIAGE_INNER_Y = 0.0004  # m; left carriage inner face at full closure (right: -CARRIAGE_INNER_Y)
MAX_PIECES_PER_JAW = 40  # collision pieces per jaw, a build validator limit
MAX_PAD_SEGMENTS = 12  # segments of a simplified pad face
PLA_DENSITY = 1240.0  # kg/m^3; mass and inertia of the printed jaw

CaptureOutcome = Literal["centred", "uncentred", "jammed"]
CAPTURE_OUTCOMES: tuple[CaptureOutcome, ...] = ("centred", "uncentred", "jammed")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _check_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise ValueError(f"{name} must be a lowercase hex sha256 (64 characters), got {value!r}")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _frozen_array(name: str, value: object, shape: tuple[int | None, ...]) -> np.ndarray:
    """A read-only float64 copy of ``value``; ``None`` in ``shape`` means "any length >= 1"."""
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != len(shape) or any(
        want is not None and got != want for got, want in zip(arr.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class JawGeometry:
    """One jaw: exact convex collision pieces plus the mass properties of its printed solid.

    * ``pieces``: convex vertex sets, each ``(n, 3)`` float64 with ``n >= 4``, metres, in the
      jaw frame (module docstring). The simulator builds one convex mesh geom per piece.
    * ``mass``: kg of the printed solid (PLA, windows and mount holes removed).
    * ``com``: ``(3,)`` centre of mass in the jaw frame.
    * ``inertia6``: ``(6,)`` inertia about the centre of mass, axes of the jaw frame, in
      kg m^2, ordered ``Ixx, Iyy, Izz, Ixy, Ixz, Iyz`` (MuJoCo ``fullinertia`` order). Unlike
      :class:`rtt.sim.objects.MeshObject` it is absolute, not per unit density.
    * ``print_mesh_sha``: sha256 of the binary STL (millimetres) that the print pack ships for
      this jaw. The STL is in the same frame as ``pieces``.

    Arrays are copied to read-only float64 on construction. Policy limits (at most
    :data:`MAX_PIECES_PER_JAW` pieces, convexity, printability) are checked by the
    validators, which can report them in words; this class only rejects malformed data.
    """

    pieces: tuple[np.ndarray, ...] = field(repr=False)
    mass: float
    com: np.ndarray
    inertia6: np.ndarray
    print_mesh_sha: str

    def __post_init__(self) -> None:
        if isinstance(self.pieces, np.ndarray) or not isinstance(self.pieces, Sequence) or not self.pieces:
            raise ValueError("a jaw needs a non-empty sequence of convex pieces")
        pieces = tuple(_frozen_array(f"piece {i}", p, (None, 3)) for i, p in enumerate(self.pieces))
        for i, p in enumerate(pieces):
            if len(p) < 4:
                raise ValueError(f"piece {i} has {len(p)} vertices; a convex piece needs at least 4")
        object.__setattr__(self, "pieces", pieces)
        mass = _finite("mass", self.mass)
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "com", _frozen_array("com", self.com, (3,)))
        inertia = _frozen_array("inertia6", self.inertia6, (6,))
        if np.any(inertia[:3] <= 0):
            raise ValueError(f"principal inertia terms Ixx, Iyy, Izz must be positive, got {inertia[:3]}")
        object.__setattr__(self, "inertia6", inertia)
        _check_sha256("print_mesh_sha", self.print_mesh_sha)

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    @property
    def bounds(self) -> np.ndarray:
        """``(2, 3)`` array: the min and max corner over every piece, in the jaw frame."""
        pts = np.concatenate(self.pieces)
        return np.stack([pts.min(axis=0), pts.max(axis=0)])


@dataclass(frozen=True)
class CaptureReport:
    """Static capture prediction for one (design, part), from the generator's predictor.

    * ``no_jam_mm``: the largest lateral placement error (mm, either sign) for which neither
      jaw is predicted to jam.
    * ``centring_mm``: the largest lateral placement error (mm, either sign) for which the
      pair is predicted to end centred.
    * ``by_error``: ``(signed lateral error in mm, predicted outcome)`` at every probed error.
    * ``mu``: the friction coefficient the prediction assumed.

    It is guidance shown on design cards and returned as feedback; it never scores a design.
    """

    no_jam_mm: float
    centring_mm: float
    mu: float
    by_error: tuple[tuple[float, CaptureOutcome], ...] = ()

    def __post_init__(self) -> None:
        for name in ("no_jam_mm", "centring_mm"):
            value = _finite(name, getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            object.__setattr__(self, name, value)
        mu = _finite("mu", self.mu)
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        object.__setattr__(self, "mu", mu)
        rows = []
        for row in self.by_error:
            error_mm, outcome = row
            if outcome not in CAPTURE_OUTCOMES:
                raise ValueError(f"capture outcome must be one of {CAPTURE_OUTCOMES}, got {outcome!r}")
            rows.append((_finite("by_error error_mm", error_mm), outcome))
        object.__setattr__(self, "by_error", tuple(rows))


@dataclass(frozen=True, eq=False)
class FingerPair:
    """A recipe compiled against one part: two jaws plus how the scene and scorer use them.

    * ``left``, ``right``: the two jaws, each in its own body frame (module docstring).
    * ``tip_height``: m above the table of the bottom of the pad band at the designed grip
      height (``grip_height_frac * part height - pad_height / 2``, floored at 3 mm). The
      controller lowers the palm until the jaws' lowest point is at this height.
    * ``finger_length``: m from the root plane down to the lowest point of either jaw.
    * ``q_seat``: the closure ``q`` (per jaw, 0 = fully open) at which both pads first touch
      the centred part. The jaws are seat aligned, so they touch at the same closure.
    * ``design_id``: sha256 of the generator version and the canonical recipe.
    * ``part_id``: :attr:`rtt.parts.types.PartSpec.part_id` of the part it was built for.
    * ``geometry_hash``: sha256 of the quantised convex pieces of both jaws.
    * ``capture``: the static capture prediction for this part.
    * ``build_s``: wall-clock seconds the build took (diagnostic, never hashed).
    * ``auto_repairs``: plain-language notes for every automatic change the build made
      (for example inactive parameters reset by canonicalisation).
    * ``boxes``: v1 entrants only: the left finger's v1 boxes (:class:`rtt.design.types.Box`,
      v1 frame, mirrored by the scene), the same solids as ``left.pieces``, kept so the scene
      can use MuJoCo's box colliders. ``None`` for every v2 design.
    """

    left: JawGeometry
    right: JawGeometry
    tip_height: float
    finger_length: float
    q_seat: float
    design_id: str
    part_id: str
    geometry_hash: str
    capture: CaptureReport
    build_s: float
    auto_repairs: tuple[str, ...] = ()
    boxes: tuple[Box, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            if not isinstance(getattr(self, name), JawGeometry):
                raise TypeError(f"{name} must be a JawGeometry")
        tip_height = _finite("tip_height", self.tip_height)
        finger_length = _finite("finger_length", self.finger_length)
        q_seat = _finite("q_seat", self.q_seat)
        build_s = _finite("build_s", self.build_s)
        if tip_height < 0:
            raise ValueError(f"tip_height must be >= 0, got {tip_height}")
        if finger_length <= 0:
            raise ValueError(f"finger_length must be positive, got {finger_length}")
        if q_seat < 0:
            raise ValueError(f"q_seat must be >= 0, got {q_seat}")
        if build_s < 0:
            raise ValueError(f"build_s must be >= 0, got {build_s}")
        for name, value in (
            ("tip_height", tip_height),
            ("finger_length", finger_length),
            ("q_seat", q_seat),
            ("build_s", build_s),
        ):
            object.__setattr__(self, name, value)
        _check_sha256("design_id", self.design_id)
        _check_sha256("geometry_hash", self.geometry_hash)
        if not isinstance(self.part_id, str) or not self.part_id:
            raise ValueError("part_id must be a non-empty string")
        if not isinstance(self.capture, CaptureReport):
            raise TypeError("capture must be a CaptureReport")
        if isinstance(self.auto_repairs, str) or not all(isinstance(r, str) for r in self.auto_repairs):
            raise ValueError("auto_repairs must be a sequence of strings")
        object.__setattr__(self, "auto_repairs", tuple(self.auto_repairs))
        if self.boxes is not None:
            object.__setattr__(self, "boxes", tuple(self.boxes))


def build_pair(design: FingerRecipe | Baseline, part: PartSpec, task: TaskV2) -> FingerPair:
    """Compile a recipe (or a baseline) against one part into a seat-aligned jaw pair.

    ``design`` is a :class:`~rtt.design.freeform.schema.FingerRecipe` or a
    :class:`~rtt.design.freeform.baselines.Baseline`, ``part`` a
    :class:`~rtt.parts.types.PartSpec` and ``task`` the v2 task configuration (gripper
    stroke, opening and palm geometry; it reads ``task.gripper.open_half_gap``,
    ``task.controller.palm_clearance``, ``task.controller.min_tip_clearance`` and
    ``task.physics.friction``, so a v1 :class:`rtt.sim.task.TaskConfig` works too). The build
    is deterministic: float64, vertices rounded to 1e-7 m, no randomness, cached per
    (generator version, design id, part geometry, id and features, gripper). An invalid design raises
    :class:`~rtt.design.freeform.validate.InvalidDesign` (a ``ValueError``) whose ``code``
    and message say which rule failed.

    This is the frozen signature; the work happens in :mod:`rtt.design.freeform.generate`
    (imported on first call, so importing this module stays numpy-only).
    """
    from rtt.design.freeform.generate import build_pair as _build_pair

    return _build_pair(design, part, task)
