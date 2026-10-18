"""Parametric ISO 9409-1 flange adapter plates (our own work, Apache-2.0).

What it makes. A round plate that bolts to a robot's tool flange and carries a tool. The robot
face sits at z = 0 and the tool face at z = thickness (millimetres, Z up). The robot face has a
centring spigot, counterbored through holes for socket head cap screws and a blind dowel hole.
The tool face has either the tool's pattern (a centring recess, blind pilot holes for heat-set
inserts or tapping, and a dowel hole) or, on a blank plate, a grid of M3 pilot holes so printed
tools can be screwed on. :func:`generate` builds the solid with manifold3d booleans, so it is a
closed manifold. :func:`write_adapter` writes a binary STL, an MJCF fragment and a JSON record.
The same spec always gives byte-identical files.

Provenance. Every dimension the generator uses is reported with a status:

* ``sourced``: read from a published source;
* ``secondary``: from one secondary source only;
* ``assumed``: our engineering assumption, or a standard table value we did not re-read.

Secondary and assumed values are unverified. They are listed in ``unverified`` and must be
measured before the part is used.

Thread depth. A cobot maker's rule says tool-flange screws must not reach more than 8 mm into
the flange (the constant lives in :mod:`rtt.library.mounts`). :func:`bolt_check` encodes it.
Engagement is screw length minus clamp length. It must be at least a minimum engagement
(1.5 x thread diameter, a rule of thumb) and at most the depth limit. An M6 screw cannot meet
1.5 x d (9 mm) inside 8 mm. For such flanges :func:`generate` falls back to 1.0 x d, a rule of
thumb for a steel screw in a steel thread, and says so in the notes.

No conformance claim. The plates use published ISO 9409-1 pattern values where we have them.
They are not checked against the standard and do not claim conformance with it.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import trimesh

from rtt.library.mounts import COBOT_MAX_THREAD_DEPTH_MM as MAX_THREAD_DEPTH_MM
from rtt.library.mounts import CONFIRMED_PATTERNS, ISO_PATTERNS, FlangePattern

__all__ = [
    "CONFIRMED_ADAPTERS",
    "MAX_THREAD_DEPTH_MM",
    "SCREW_LENGTHS_MM",
    "AdapterResult",
    "AdapterSpec",
    "BoltCheck",
    "Feature",
    "bolt_check",
    "default_key",
    "generate",
    "recommend_bolt",
    "write_adapter",
]

Material = Literal["PLA", "aluminium"]
FeatureKind = Literal["spigot", "through", "counterbore", "blind", "recess"]

GENERATOR: Final = "rtt.library.adapters"
LICENCE: Final = "Apache-2.0"

#: Segments per circle. A multiple of 4, so the outer disc touches its bounding box.
SEGMENTS: Final = 64
#: Cutters run this far past a face they open onto, so no boolean face is coplanar.
_OVERCUT_MM: Final = 1.0
#: The spigot reaches this far into the plate before the union, for the same reason.
_WELD_MM: Final = 0.5

WALL_MM: Final = 2.0
SPIGOT_HEIGHT_MM: Final = 2.0
RECESS_DEPTH_MM: Final = 2.5
ROBOT_DOWEL_DEPTH_MM: Final = 6.0
TOOL_DOWEL_DEPTH_MM: Final = 5.0
COUNTERBORE_EXTRA_D_MM: Final = 1.0
COUNTERBORE_EXTRA_DEPTH_MM: Final = 0.5
MIN_CLAMP_MM: Final = 2.0
OUTER_MARGIN_THREADS: Final = 3.0
GRID_PILOT_MM: Final = 2.5
GRID_DEPTH_MM: Final = 6.0
GRID_PITCH_MM: Final = 10.0
MIN_ENGAGEMENT_FACTOR: Final = 1.5
FALLBACK_ENGAGEMENT_FACTOR: Final = 1.0

#: ISO 273 medium series clearance holes (mm).
CLEARANCE_MM: Final = {"M5": 5.5, "M6": 6.6, "M8": 9.0}
#: ISO 4762 socket head cap screw head diameters (mm).
HEAD_MM: Final = {"M5": 8.5, "M6": 10.0, "M8": 13.0}
#: Tap-drill diameters for coarse threads (mm).
TAP_DRILL_MM: Final = {"M5": 4.2, "M6": 5.0, "M8": 6.8}
#: Standard ISO 4762 lengths the recommendation chooses from (mm).
SCREW_LENGTHS_MM: Final = (6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0)
#: Typical datasheet densities (g/cm3).
DENSITY_G_CM3: Final = {"PLA": 1.24, "aluminium": 2.70}

_STL_HEADER: Final = b"rtt.library.adapters binary STL, units mm, Apache-2.0".ljust(80, b" ")
_KEY_RE: Final = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_SRC_TABLE = "standard table value we did not re-read today"
_SRC_RULE = "generator assumption"
_NOTE_ISO = (
    "Not checked against a purchased copy of ISO 9409-1:2004; dimensions marked unverified must "
    "be measured before use."
)
_NOTE_CONFORMANCE = "This part does not claim conformance with ISO 9409-1."


# ----------------------------------------------------------------------------- spec


def _pattern(designation: str) -> FlangePattern:
    try:
        return ISO_PATTERNS[designation]
    except KeyError:
        raise ValueError(
            f"unknown ISO 9409-1 pattern {designation!r}; known: {sorted(ISO_PATTERNS)}"
        ) from None


def _slug(designation: str) -> str:
    body = designation.removeprefix("ISO 9409-1-")
    return "iso" + body.lower().replace(".", "p")


def default_key(robot: str, tool: str | None = None) -> str:
    """The library key for a pattern pair, e.g. ``adapter-iso50-4-m6-to-iso31p5-4-m5``."""
    if tool is None:
        return f"plate-{_slug(robot)}"
    return f"adapter-{_slug(robot)}-to-{_slug(tool)}"


@dataclass(frozen=True)
class AdapterSpec:
    """What to generate.

    ``robot`` and ``tool`` are ISO 9409-1 designations from
    :data:`~rtt.library.mounts.ISO_PATTERNS`; ``tool=None`` makes a blank tool plate.
    ``print_allowance_mm`` is added to every hole and recess and taken off the spigot.
    ``max_thread_depth_mm`` is how deep the robot flange's threads may be used.
    """

    robot: str
    tool: str | None = None
    thickness_mm: float = 10.0
    material: Material = "PLA"
    infill: float = 1.0
    print_allowance_mm: float = 0.1
    key: str = ""
    max_thread_depth_mm: float = MAX_THREAD_DEPTH_MM

    def __post_init__(self) -> None:
        _pattern(self.robot)
        if self.tool is not None:
            _pattern(self.tool)
        if self.material not in DENSITY_G_CM3:
            raise ValueError(f"material must be one of {sorted(DENSITY_G_CM3)}, got {self.material!r}")
        for name in ("thickness_mm", "infill", "print_allowance_mm", "max_thread_depth_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not 6.0 <= self.thickness_mm <= 30.0:
            raise ValueError(f"thickness_mm must be between 6 and 30 mm, got {self.thickness_mm}")
        if not 0.0 < self.infill <= 1.0:
            raise ValueError(f"infill must be in (0, 1], got {self.infill}")
        if not 0.0 <= self.print_allowance_mm <= 1.0:
            raise ValueError(f"print_allowance_mm must be between 0 and 1 mm, got {self.print_allowance_mm}")
        if self.max_thread_depth_mm <= 0.0:
            raise ValueError(f"max_thread_depth_mm must be positive, got {self.max_thread_depth_mm}")
        if not self.key:
            object.__setattr__(self, "key", default_key(self.robot, self.tool))
        if not _KEY_RE.match(self.key):
            raise ValueError(f"key must be a lowercase slug (a-z, 0-9, single hyphens), got {self.key!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------- bolts


@dataclass(frozen=True)
class BoltCheck:
    """One socket head cap screw through the plate into the robot flange (mm).

    ``engagement_mm`` is length minus clamp. ``ok`` is true when it lies between
    ``min_engagement_mm`` and ``max_thread_depth_mm``; ``reasons`` says why it does not.
    """

    bolt: str
    length_mm: float
    clamp_mm: float
    engagement_mm: float
    max_thread_depth_mm: float
    min_engagement_mm: float
    ok: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reasons"] = list(self.reasons)
        return out


def bolt_check(
    pattern_designation: str,
    clamp_mm: float,
    bolt_length_mm: float,
    max_thread_depth_mm: float = MAX_THREAD_DEPTH_MM,
    *,
    min_engagement_factor: float = MIN_ENGAGEMENT_FACTOR,
) -> BoltCheck:
    """Check one screw length against the thread-depth limit and the minimum engagement."""
    pattern = _pattern(pattern_designation)
    if clamp_mm <= 0.0 or bolt_length_mm <= 0.0 or max_thread_depth_mm <= 0.0:
        raise ValueError("clamp_mm, bolt_length_mm and max_thread_depth_mm must be positive")
    if min_engagement_factor <= 0.0:
        raise ValueError("min_engagement_factor must be positive")
    bolt = f"{pattern.thread}x{bolt_length_mm:g}"
    engagement = float(bolt_length_mm) - float(clamp_mm)
    min_engagement = min_engagement_factor * pattern.thread_d
    reasons = []
    if engagement > max_thread_depth_mm:
        reasons.append(
            f"{bolt} reaches {engagement:.1f} mm into the flange thread; "
            f"the limit is {max_thread_depth_mm:.1f} mm."
        )
    if engagement < min_engagement:
        reasons.append(
            f"{bolt} engages only {engagement:.1f} mm of thread; at least {min_engagement:.1f} mm "
            f"({min_engagement_factor:.1f} x d) is wanted."
        )
    return BoltCheck(
        bolt=bolt,
        length_mm=float(bolt_length_mm),
        clamp_mm=float(clamp_mm),
        engagement_mm=engagement,
        max_thread_depth_mm=float(max_thread_depth_mm),
        min_engagement_mm=min_engagement,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def recommend_bolt(
    pattern_designation: str,
    clamp_mm: float,
    max_thread_depth_mm: float = MAX_THREAD_DEPTH_MM,
    *,
    min_engagement_factor: float = MIN_ENGAGEMENT_FACTOR,
) -> BoltCheck:
    """The longest standard ISO 4762 length that passes :func:`bolt_check`."""
    for length in sorted(SCREW_LENGTHS_MM, reverse=True):
        check = bolt_check(
            pattern_designation,
            clamp_mm,
            length,
            max_thread_depth_mm,
            min_engagement_factor=min_engagement_factor,
        )
        if check.ok:
            return check
    pattern = _pattern(pattern_designation)
    lo = min_engagement_factor * pattern.thread_d
    raise ValueError(
        f"no standard {pattern.thread} length gives between {lo:.1f} and {max_thread_depth_mm:.1f} mm "
        f"of engagement through a {clamp_mm:.1f} mm clamp (lengths {list(SCREW_LENGTHS_MM)})"
    )


# ----------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Feature:
    """One cylindrical feature, as cut (mm).

    The spigot adds material; every other kind removes it between ``z_min_mm`` and ``z_max_mm``.
    ``diameter_mm`` already includes the print allowance.
    """

    name: str
    side: Literal["robot", "tool"]
    kind: FeatureKind
    x_mm: float
    y_mm: float
    diameter_mm: float
    z_min_mm: float
    z_max_mm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _polar(radius: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return round(radius * math.cos(a), 9) + 0.0, round(radius * math.sin(a), 9) + 0.0


def _wall(a: Feature, b: Feature) -> float:
    """Material left between two features in plan view."""
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm) - (a.diameter_mm + b.diameter_mm) / 2.0


def _min_wall(f: Feature, others: Sequence[Feature]) -> float:
    return min((_wall(f, o) for o in others), default=math.inf)


def _thread_table(table: dict[str, float], thread: str, what: str) -> float:
    try:
        return table[thread]
    except KeyError:
        raise ValueError(f"no {what} value for {thread}; known: {sorted(table)}") from None


def _dim(value: Any, status: str, source: str) -> dict[str, Any]:
    return {"value": value, "status": status, "source": source}


def _pattern_dims(side: str, p: FlangePattern) -> dict[str, dict[str, Any]]:
    designation_src = f"part of the designation {p.designation}; {p.d1.source}"
    return {
        f"{side}_d1_mm": p.d1.to_dict(),
        f"{side}_d2_mm": p.d2.to_dict(),
        f"{side}_dowel_mm": p.dowel_d.to_dict(),
        f"{side}_thread": _dim(p.thread, p.d1.status, designation_src),
        f"{side}_n_holes": _dim(p.n_holes, p.d1.status, designation_src),
    }


@dataclass(frozen=True)
class _Layout:
    outer_d: float
    features: tuple[Feature, ...]
    dims: dict[str, dict[str, Any]]
    clamp_mm: float
    notes: tuple[str, ...]


def _layout(spec: AdapterSpec) -> _Layout:
    t = spec.thickness_mm
    a = spec.print_allowance_mm
    rp = _pattern(spec.robot)
    tp = _pattern(spec.tool) if spec.tool is not None else None
    dims: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    pitch = max(rp.d1.value, tp.d1.value if tp else 0.0)
    margin = OUTER_MARGIN_THREADS * max(rp.thread_d, tp.thread_d if tp else 0.0)
    outer_d = float(math.ceil(pitch + 2.0 * margin - 1e-9))
    radius = outer_d / 2.0
    dims["outer_diameter_mm"] = _dim(
        outer_d, "assumed", f"{_SRC_RULE}: largest pitch circle + 2 x 3 x thread diameter, rounded up"
    )
    dims["wall_min_mm"] = _dim(WALL_MM, "assumed", f"{_SRC_RULE}: least wall between any two features")
    dims["print_allowance_mm"] = _dim(
        a, "assumed", "spec input; added to every hole and recess, taken off the spigot"
    )

    # Robot side.
    dims.update(_pattern_dims("robot", rp))
    clearance = _thread_table(CLEARANCE_MM, rp.thread, "clearance hole")
    head = _thread_table(HEAD_MM, rp.thread, "screw head")
    cb_d = head + COUNTERBORE_EXTRA_D_MM
    cb_depth = rp.thread_d + COUNTERBORE_EXTRA_DEPTH_MM
    clamp = t - cb_depth
    if clamp < MIN_CLAMP_MM:
        raise ValueError(
            f"a {t:g} mm plate leaves {clamp:.1f} mm under the {rp.thread} screw heads; at least "
            f"{MIN_CLAMP_MM:g} mm is needed, so use a plate of at least {cb_depth + MIN_CLAMP_MM:g} mm"
        )
    robot_dowel_depth = min(ROBOT_DOWEL_DEPTH_MM, t - WALL_MM)
    start = 180.0 / rp.n_holes
    step = 360.0 / rp.n_holes
    r_robot = rp.d1.value / 2.0
    feats: list[Feature] = [
        Feature("robot_spigot", "robot", "spigot", 0.0, 0.0, rp.d2.value - a, -SPIGOT_HEIGHT_MM, 0.0)
    ]
    robot_cut: list[Feature] = []
    for k in range(rp.n_holes):
        x, y = _polar(r_robot, start + k * step)
        robot_cut.append(
            Feature(f"robot_bolt_{k}", "robot", "through", x, y, clearance + a, -SPIGOT_HEIGHT_MM, t)
        )
        robot_cut.append(
            Feature(f"robot_counterbore_{k}", "robot", "counterbore", x, y, cb_d + a, t - cb_depth, t)
        )
    x, y = _polar(r_robot, 0.0)
    robot_cut.append(
        Feature("robot_dowel", "robot", "blind", x, y, rp.dowel_d.value + a, 0.0, robot_dowel_depth)
    )
    feats += robot_cut
    dims["spigot_height_mm"] = _dim(
        SPIGOT_HEIGHT_MM, "assumed", f"{_SRC_RULE}: kept short to fit a shallow centring recess"
    )
    dims["robot_hole_layout"] = _dim(
        f"{rp.n_holes} holes at {start:g} + k x {step:g} deg; dowel on the pitch circle at 0 deg",
        "assumed",
        f"{_SRC_RULE}: positions not read from the standard",
    )
    dims["robot_clearance_hole_mm"] = _dim(clearance, "assumed", f"ISO 273 medium series, {_SRC_TABLE}")
    dims["robot_screw_head_mm"] = _dim(head, "assumed", f"ISO 4762 head diameter, {_SRC_TABLE}")
    dims["robot_counterbore_mm"] = _dim(cb_d, "assumed", f"{_SRC_RULE}: head diameter + 1.0 mm")
    dims["robot_counterbore_depth_mm"] = _dim(cb_depth, "assumed", f"{_SRC_RULE}: thread diameter + 0.5 mm")
    dims["robot_dowel_depth_mm"] = _dim(
        robot_dowel_depth,
        "assumed",
        f"{_SRC_RULE}: 6 mm blind from the robot face, less if the plate is thin",
    )
    dims["min_clamp_mm"] = _dim(MIN_CLAMP_MM, "assumed", f"{_SRC_RULE}: least material under a screw head")

    if tp is None:
        feats += _grid(spec, radius, robot_cut, dims, notes)
    else:
        feats += _tool_side(spec, tp, radius, robot_cut, dims)

    return _Layout(outer_d, tuple(feats), dims, clamp, tuple(notes))


def _tool_side(
    spec: AdapterSpec,
    tp: FlangePattern,
    radius: float,
    robot_cut: Sequence[Feature],
    dims: dict[str, dict[str, Any]],
) -> list[Feature]:
    t = spec.thickness_mm
    a = spec.print_allowance_mm
    dims.update(_pattern_dims("tool", tp))
    recess = Feature("tool_recess", "tool", "recess", 0.0, 0.0, tp.d2.value + a, t - RECESS_DEPTH_MM, t)
    worst = min(robot_cut, key=lambda f: (_wall(recess, f), f.name))
    if _wall(recess, worst) < WALL_MM:
        raise ValueError(
            f"the tool centring recess ({recess.diameter_mm:g} mm) leaves {_wall(recess, worst):.2f} mm of "
            f"wall to {worst.name}; at least {WALL_MM:g} mm is needed. The robot screw heads would crowd or "
            f"break into the tool's recess, and this generator has no layout for that pairing."
        )
    tap_nominal = _thread_table(TAP_DRILL_MM, tp.thread, "tap drill")
    tap = tap_nominal + a
    depth = min(t - WALL_MM, 2.0 * tp.thread_d)
    dowel_d = tp.dowel_d.value + a
    dowel_depth = min(TOOL_DOWEL_DEPTH_MM, t - WALL_MM)
    r_tool = tp.d1.value / 2.0
    if r_tool + max(tap, dowel_d) / 2.0 + WALL_MM > radius:
        raise ValueError(f"the tool holes leave less than {WALL_MM:g} mm of wall to the plate rim")
    n = tp.n_holes
    step = 360.0 / n
    best: tuple[float, int, int] | None = None  # (wall, rotation, dowel slot)
    for rot in range(round(step)):
        holes = [
            Feature("h", "tool", "blind", *_polar(r_tool, rot + step / 2.0 + k * step), tap, t - depth, t)
            for k in range(n)
        ]
        hole_wall = min(_min_wall(h, robot_cut) for h in holes)
        for slot in range(n):
            dowel = Feature(
                "d", "tool", "blind", *_polar(r_tool, rot + slot * step), dowel_d, t - dowel_depth, t
            )
            wall = round(min(hole_wall, _min_wall(dowel, [*robot_cut, *holes])), 9)
            if best is None or wall > best[0]:
                best = (wall, rot, slot)
    assert best is not None
    wall, rot, slot = best
    if wall < WALL_MM:
        raise ValueError(
            f"no rotation of the {tp.designation} tool holes in 1 deg steps keeps {WALL_MM:g} mm of wall "
            f"from the robot holes, counterbores and dowel (best {wall:.2f} mm at {rot} deg)"
        )
    feats = [recess]
    for k in range(n):
        angle = rot + step / 2.0 + k * step
        feats.append(Feature(f"tool_hole_{k}", "tool", "blind", *_polar(r_tool, angle), tap, t - depth, t))
    dowel_angle = (rot + slot * step) % 360.0
    feats.append(
        Feature("tool_dowel", "tool", "blind", *_polar(r_tool, dowel_angle), dowel_d, t - dowel_depth, t)
    )
    dims["tool_recess_depth_mm"] = _dim(
        RECESS_DEPTH_MM, "assumed", f"{_SRC_RULE}: meant to clear the tool's centring spigot"
    )
    dims["tool_tap_drill_mm"] = _dim(
        tap_nominal, "assumed", f"tap-drill size for heat-set inserts or tapping, {_SRC_TABLE}"
    )
    dims["tool_hole_depth_mm"] = _dim(
        depth, "assumed", f"{_SRC_RULE}: blind, the smaller of thickness - 2 mm and 2 x thread diameter"
    )
    dims["tool_dowel_depth_mm"] = _dim(
        dowel_depth, "assumed", f"{_SRC_RULE}: 5 mm blind from the tool face, less if the plate is thin"
    )
    dims["tool_hole_layout"] = _dim(
        f"{n} holes at {rot + step / 2.0:g} + k x {step:g} deg; dowel on the pitch circle at "
        f"{dowel_angle:g} deg, midway between two holes",
        "assumed",
        f"{_SRC_RULE}: rotation chosen in 1 deg steps for the widest wall to the robot features "
        f"({wall:.2f} mm)",
    )
    return feats


def _grid(
    spec: AdapterSpec,
    radius: float,
    robot_cut: Sequence[Feature],
    dims: dict[str, dict[str, Any]],
    notes: list[str],
) -> list[Feature]:
    t = spec.thickness_mm
    d = GRID_PILOT_MM + spec.print_allowance_mm
    depth = min(GRID_DEPTH_MM, t - WALL_MM)
    n = int(radius // GRID_PITCH_MM)
    feats: list[Feature] = []
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = i * GRID_PITCH_MM, j * GRID_PITCH_MM
            hole = Feature(f"grid_{len(feats):03d}", "tool", "blind", x + 0.0, y + 0.0, d, t - depth, t)
            if math.hypot(x, y) + d / 2.0 + WALL_MM > radius:
                continue
            if _min_wall(hole, robot_cut) < WALL_MM:
                continue
            feats.append(hole)
    dims["tool_grid_pilot_mm"] = _dim(GRID_PILOT_MM, "assumed", f"M3 tap-drill size, {_SRC_TABLE}")
    dims["tool_grid_depth_mm"] = _dim(depth, "assumed", f"{_SRC_RULE}: 6 mm blind, less if the plate is thin")
    dims["tool_grid_pitch_mm"] = _dim(GRID_PITCH_MM, "assumed", f"{_SRC_RULE}: centred square grid")
    dims["tool_grid_layout"] = _dim(
        f"{len(feats)} holes on a centred {GRID_PITCH_MM:g} mm grid, clear of the robot holes",
        "assumed",
        f"{_SRC_RULE}: holes kept {WALL_MM:g} mm from every robot feature and the rim",
    )
    notes.append(
        f"Blank tool face: {len(feats)} M3 pilot holes on a {GRID_PITCH_MM:g} mm grid so printed tools "
        f"can be screwed on."
    )
    return feats


# ----------------------------------------------------------------------------- geometry


def _cylinder(f: Feature, thickness: float) -> Any:
    """The manifold for one feature. Cutters run past the faces they open onto."""
    from manifold3d import Manifold

    r = f.diameter_mm / 2.0
    if f.kind == "spigot":
        z0, z1 = f.z_min_mm, f.z_max_mm + _WELD_MM
    else:
        r /= math.cos(math.pi / SEGMENTS)  # circumscribed: the cut is never smaller than the circle
        z0 = -SPIGOT_HEIGHT_MM - _OVERCUT_MM if f.z_min_mm <= 0.0 else f.z_min_mm
        z1 = thickness + _OVERCUT_MM if f.z_max_mm >= thickness else f.z_max_mm
    return Manifold.cylinder(z1 - z0, r, r, SEGMENTS).translate((f.x_mm, f.y_mm, z0))


def _canonical(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Sort vertices and faces so equal geometry always gives equal arrays."""
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    faces = inverse[faces]
    first = np.argmin(faces, axis=1)
    faces = np.take_along_axis(faces, (first[:, None] + np.arange(3)[None, :]) % 3, axis=1)
    faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]
    return trimesh.Trimesh(vertices=vertices[order], faces=faces, process=False)


def _build_mesh(spec: AdapterSpec, layout: _Layout) -> trimesh.Trimesh:
    from manifold3d import Manifold, OpType

    t = spec.thickness_mm
    r = layout.outer_d / 2.0
    parts = [Manifold.cylinder(t, r, r, SEGMENTS)]
    cutters = []
    for f in layout.features:
        (parts if f.kind == "spigot" else cutters).append(_cylinder(f, t))
    body = Manifold.batch_boolean(parts, OpType.Add)
    solid = Manifold.batch_boolean([body, *cutters], OpType.Subtract)
    if solid.is_empty():
        raise ValueError(f"the boolean for {spec.key} produced an empty solid ({solid.status()})")
    mesh = solid.to_mesh64()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return _canonical(vertices, faces)


# ----------------------------------------------------------------------------- result


@dataclass(frozen=True)
class AdapterResult:
    """A generated plate: the mesh (mm) and every record that goes with it."""

    spec: AdapterSpec
    mesh: trimesh.Trimesh
    dimensions: dict[str, dict[str, Any]]
    unverified: tuple[str, ...]
    bolt: BoltCheck
    volume_mm3: float
    mass_kg: float
    notes: tuple[str, ...]
    outer_diameter_mm: float
    features: tuple[Feature, ...]

    @property
    def density_kg_mm3(self) -> float:
        """Effective density (material x infill) in kg per cubic millimetre."""
        return DENSITY_G_CM3[self.spec.material] * 1e-6 * self.spec.infill

    def record(self) -> dict[str, Any]:
        """The JSON record written next to the STL (not yet rounded)."""
        return {
            "generator": GENERATOR,
            "licence": LICENCE,
            "claims_iso_conformance": False,
            "key": self.spec.key,
            "spec": self.spec.to_dict(),
            "files": {"stl": f"{self.spec.key}.stl", "mjcf": f"{self.spec.key}.xml"},
            "units": {"stl": "mm", "mjcf": "m, kg", "record": "mm, kg"},
            "outer_diameter_mm": self.outer_diameter_mm,
            "volume_mm3": self.volume_mm3,
            "mass_kg": self.mass_kg,
            "dimensions": self.dimensions,
            "unverified": list(self.unverified),
            "bolt": self.bolt.to_dict(),
            "features": [f.to_dict() for f in self.features],
            "notes": list(self.notes),
        }


def _robot_bolt(spec: AdapterSpec, clamp: float, dims: dict[str, dict[str, Any]]) -> tuple[BoltCheck, str]:
    p = _pattern(spec.robot)
    limit = spec.max_thread_depth_mm
    note = ""
    try:
        check = recommend_bolt(spec.robot, clamp, limit)
        factor = MIN_ENGAGEMENT_FACTOR
    except ValueError as first:
        try:
            check = recommend_bolt(spec.robot, clamp, limit, min_engagement_factor=FALLBACK_ENGAGEMENT_FACTOR)
        except ValueError as second:
            raise ValueError(
                f"{first}; the {FALLBACK_ENGAGEMENT_FACTOR:.1f} x d fallback fails too ({second}). "
                f"Set max_thread_depth_mm to the flange's real thread depth or change the thickness."
            ) from None
        factor = FALLBACK_ENGAGEMENT_FACTOR
        note = (
            f"No standard {p.thread} length gives {MIN_ENGAGEMENT_FACTOR * p.thread_d:g} mm "
            f"({MIN_ENGAGEMENT_FACTOR:.1f} x d) of engagement within the {limit:g} mm depth limit, so the "
            f"screw was chosen with the {FALLBACK_ENGAGEMENT_FACTOR:.1f} x d rule for a steel screw in a "
            f"steel thread. Check the flange material before use."
        )
    depth_src = (
        "a cobot maker's rule recorded in rtt.library.mounts, not re-read today"
        if limit == MAX_THREAD_DEPTH_MM
        else "spec input"
    )
    dims["max_thread_depth_mm"] = _dim(limit, "assumed", depth_src)
    dims["min_engagement_factor"] = _dim(
        factor, "assumed", "rule of thumb: thread engagement as a multiple of the thread diameter"
    )
    dims["screw_lengths_mm"] = _dim(list(SCREW_LENGTHS_MM), "assumed", f"ISO 4762 lengths, {_SRC_TABLE}")
    return check, note


def generate(spec: AdapterSpec) -> AdapterResult:
    """Build the plate for ``spec``. Raises ``ValueError`` when no layout keeps the wall rule."""
    layout = _layout(spec)
    dims = dict(layout.dims)
    bolt, bolt_note = _robot_bolt(spec, layout.clamp_mm, dims)
    density = DENSITY_G_CM3[spec.material]
    dims["density_g_cm3"] = _dim(density, "assumed", f"typical {spec.material} datasheet value")
    mesh = _build_mesh(spec, layout)
    volume = float(mesh.volume)
    mass = volume * density * 1e-6 * spec.infill

    notes = [_NOTE_ISO, _NOTE_CONFORMANCE]
    rp = _pattern(spec.robot)
    notes.append(
        f"Robot side: fit {rp.n_holes} x {bolt.bolt} socket head cap screws from the tool face; each "
        f"engages {bolt.engagement_mm:g} mm of the flange thread."
    )
    if bolt_note:
        notes.append(bolt_note)
    notes += layout.notes
    if spec.tool is not None:
        notes.append(
            f"Tool side: the {_pattern(spec.tool).thread} pilot holes are blind, for heat-set inserts "
            f"or tapping."
        )
    notes.append(
        f"Mass assumes a uniform {spec.infill:.0%} fill of {spec.material} at {density:g} g/cm3; real "
        f"prints have denser walls."
    )
    if spec.material == "aluminium" and spec.print_allowance_mm > 0.0:
        notes.append("The print allowance still applies; set it to 0 for a machined part.")
    for designation in dict.fromkeys(d for d in (spec.robot, spec.tool) if d is not None):
        notes += [f"{designation}: {n}." for n in _pattern(designation).notes]

    unverified = tuple(sorted(name for name, d in dims.items() if d["status"] != "sourced"))
    return AdapterResult(
        spec=spec,
        mesh=mesh,
        dimensions=dims,
        unverified=unverified,
        bolt=bolt,
        volume_mm3=volume,
        mass_kg=mass,
        notes=tuple(notes),
        outer_diameter_mm=layout.outer_d,
        features=layout.features,
    )


# ----------------------------------------------------------------------------- files


def _stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0.0)
    record = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tri), dtype=record)
    data["normal"] = normals
    data["vertices"] = tri
    return _STL_HEADER + struct.pack("<I", len(tri)) + data.tobytes()


def _num(v: float) -> str:
    return format(float(v) + 0.0, ".9g")


def _mjcf(result: AdapterResult) -> str:
    key = result.spec.key
    rho = result.density_kg_mm3
    props = trimesh.triangles.mass_properties(result.mesh.triangles, density=1.0)
    pos = np.asarray(props.center_mass) * 1e-3
    inertia = np.asarray(props.inertia) * rho * 1e-6  # kg mm^2 -> kg m^2, about the centre of mass
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    tool_z = result.spec.thickness_mm * 1e-3
    return (
        f'<mujoco model="{key}">\n'
        f"  <!-- Generated by {GENERATOR} ({LICENCE}). Mesh in millimetres, scaled to metres. -->\n"
        f"  <!-- Not checked against ISO 9409-1; the JSON record lists the unverified dimensions. -->\n"
        f"  <asset>\n"
        f'    <mesh name="{key}" file="{key}.stl" scale="0.001 0.001 0.001"/>\n'
        f"  </asset>\n"
        f"  <worldbody>\n"
        f'    <body name="{key}">\n'
        f'      <inertial pos="{" ".join(_num(v) for v in pos)}" mass="{_num(result.mass_kg)}" '
        f'fullinertia="{" ".join(_num(v) for v in full)}"/>\n'
        f'      <geom name="{key}" type="mesh" mesh="{key}"/>\n'
        f'      <site name="mount_robot" pos="0 0 0"/>\n'
        f'      <site name="mount_tool" pos="0 0 {_num(tool_z)}"/>\n'
        f"    </body>\n"
        f"  </worldbody>\n"
        f"</mujoco>\n"
    )


def _rounded(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float | np.floating):
        return round(float(value), 6) + 0.0
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    raise TypeError(f"cannot write {type(value).__name__} to the adapter record")


def write_adapter(spec: AdapterSpec, out_dir: Path) -> dict[str, Path]:
    """Write ``<key>.stl`` (mm), ``<key>.xml`` (MJCF, m and kg) and ``<key>.json`` into ``out_dir``."""
    result = generate(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stl": out_dir / f"{spec.key}.stl",
        "xml": out_dir / f"{spec.key}.xml",
        "json": out_dir / f"{spec.key}.json",
    }
    paths["stl"].write_bytes(_stl_bytes(result.mesh))
    paths["xml"].write_text(_mjcf(result), encoding="utf-8")
    text = json.dumps(_rounded(result.record()), sort_keys=True, indent=2) + "\n"
    paths["json"].write_text(text, encoding="utf-8")
    return paths


#: The pre-built set the library ships: blank plates on both confirmed patterns and one adapter.
CONFIRMED_ADAPTERS: Final[tuple[AdapterSpec, ...]] = (
    AdapterSpec(robot=CONFIRMED_PATTERNS[0], key="plate-iso50-4-m6"),
    AdapterSpec(robot=CONFIRMED_PATTERNS[1], key="plate-iso31p5-4-m5"),
    AdapterSpec(
        robot=CONFIRMED_PATTERNS[0],
        tool=CONFIRMED_PATTERNS[1],
        key="adapter-iso50-4-m6-to-iso31p5-4-m5",
    ),
)
