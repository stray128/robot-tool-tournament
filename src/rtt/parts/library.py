"""The procedural parts library: industrial part families built from exact convex pieces.

Each family is a pure function from its dimensions (millimetres, as in the config files) to
exact convex pieces (:mod:`rtt.parts.pieces`), declared features, yaw symmetry and
registration axes. :func:`build_part` turns that into a :class:`~rtt.parts.types.PartSpec`
in the object frame (metres, z up, bounding box centred on the origin, y the closing axis,
the left jaw at +y). A family may put the x/y origin elsewhere (``Shape.origin_xy``): the
origin is where the jaws aim, so the L-bracket's is the centre of the leg the jaws grip.

Training families (``configs/parts_v2.yaml``, the ``train_v2`` suite):

* ``pipe_tee``: a 1/2-inch style slip tee lying flat, run along x, branch toward +y (the left
  jaw). Hubs, three blind sockets (16-gon walls), solid core. The hero part.
* ``block``: a plain rectangular block with sharp edges (no lead-in): the clearest jam case.
* ``hex_standoff``: a standing hex bar with flats facing the jaws and a tapped top hole.
  ``yaw_symmetry`` 6. Twelve full-height sectors around the bore plus a plug under the hole,
  so no internal seam crosses the gripped flats (a horizontal seam gave the pads a ledge).
* ``flanged_bushing``: a standing bushing, flange up, with a through bore. Round: yaw is not
  registered.
* ``l_bracket``: a stamped L-bracket lying flat on one leg, the other leg standing at the
  +x end with its edges toward the jaws. The object frame is centred on the standing leg
  (where the jaws grip it), so the centre of mass sits 9.5 mm off the grip axis.
* ``slug``: a solid round slug (32-gon). Round: yaw is not registered.

Held-out-only families (``rtt.parts.heldout``): ``pipe_elbow`` (a 90 degree slip elbow lying
flat) and ``spur_gear`` (an involute spur gear lying flat, one convex prism per tooth).

Round walls are circumscribed polygons (flats at the nominal radius), so a part is never
smaller than nominal outside and a bore never smaller than nominal inside; walls are tiled
by trapezoidal wedges. Pieces may overlap only where two cores meet (tee, elbow); mass
properties remove overlaps exactly (:func:`rtt.parts.pieces.mass_properties`).

The object frame keeps each part the way it is gripped; it is not re-posed to the
narrowest horizontal extent (``MeshObject`` convention), because a part-relative design
depends on which face meets which jaw.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from rtt.parts import pieces as pc
from rtt.parts.materials import SCAN_CLAY, material
from rtt.parts.types import REGISTER_AXES, Feature, Material, PartSpec, RegisterAxis

if TYPE_CHECKING:
    from rtt.sim.objects import MeshObject

LIBRARY_VERSION = "rtt-parts-v2.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARTS_FILE = REPO_ROOT / "configs" / "parts_v2.yaml"
MM = 1e-3
MIN_WALL_MM = 1.0
ROUND_AXES: tuple[RegisterAxis, ...] = ("x", "y", "tilt")


@dataclass(frozen=True)
class Shape:
    """A family's output in its natural frame (metres): pieces, features and symmetry."""

    pieces: tuple[pc.Piece, ...]
    features: tuple[Feature, ...] = ()
    yaw_symmetry: int = 1
    register_axes: tuple[RegisterAxis, ...] = REGISTER_AXES
    #: the natural-frame (x, y) point that becomes the object frame's x/y origin (where the
    #: jaws aim); ``None`` is the bounding-box centre. z is always bounding-box centred.
    origin_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class Family:
    """One part family: default dimensions (mm or counts), a validator and a builder.

    ``integers`` names the count parameters (segments, teeth); every other parameter is a
    length in millimetres, except ``angles`` (degrees). ``free`` lists the dimensions a
    held-out instance may vary independently (the rest follow the overall scale).
    """

    name: str
    defaults: Mapping[str, float]
    build: Callable[[Mapping[str, float]], Shape] = field(repr=False)
    check: Callable[[Mapping[str, float]], list[str]] = field(repr=False)
    integers: frozenset[str] = frozenset()
    angles: frozenset[str] = frozenset()
    free: tuple[str, ...] = ()
    description: str = ""

    def lengths(self) -> tuple[str, ...]:
        return tuple(k for k in self.defaults if k not in self.integers and k not in self.angles)

    def resolve(self, params: Mapping[str, Any] | None) -> dict[str, float]:
        """Defaults overridden by ``params``; unknown keys and bad values are refused."""
        params = dict(params or {})
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameters {sorted(unknown)}; known {sorted(self.defaults)}"
            )
        out: dict[str, float] = {}
        for key, default in self.defaults.items():
            value = params.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{self.name}.{key} must be a finite number, got {value!r}")
            if key in self.integers:
                if float(value) != int(value):
                    raise ValueError(f"{self.name}.{key} must be an integer, got {value!r}")
                out[key] = int(value)
            else:
                out[key] = float(value)
        problems = self.check(out)
        if problems:
            raise ValueError(f"{self.name}: invalid dimensions: " + "; ".join(problems))
        return out


# --------------------------------------------------------------------------- family builders


def _pipe_check(p: Mapping[str, float]) -> list[str]:
    out = []
    r_b, r_h, r_s = p["body_od"] / 2, p["hub_od"] / 2, p["socket_id"] / 2
    if not r_s + MIN_WALL_MM <= r_b <= r_h:
        out.append(f"need socket_id/2 + {MIN_WALL_MM} <= body_od/2 <= hub_od/2 (got {r_s}, {r_b}, {r_h})")
    if not 0 < p["hub_length"] < p["socket_depth"]:
        out.append("need 0 < hub_length < socket_depth")
    if p["centre_to_end"] - p["socket_depth"] < r_b - 1e-9:
        out.append("need centre_to_end - socket_depth >= body_od/2 (the socket floors clear the core)")
    if p["segments"] < 8 or p["segments"] % 4:
        out.append("segments must be a multiple of 4, at least 8")
    return out


def _socket_rings(
    axis: pc.Axis, sign: float, p: Mapping[str, float], centre: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> list[pc.Piece]:
    """The two wall rings of one socket on ``axis`` (direction ``sign``): body then hub."""
    n = int(p["segments"])
    r_b, r_h, r_s = p["body_od"] / 2 * MM, p["hub_od"] / 2 * MM, p["socket_id"] / 2 * MM
    end, depth, hub = p["centre_to_end"] * MM, p["socket_depth"] * MM, p["hub_length"] * MM
    spans = [(end - depth, end - hub, r_b), (end - hub, end, r_h)]
    out = []
    iw = "xyz".index(axis)
    for a, b, r_out in spans:
        lo, hi = sorted((sign * a + centre[iw], sign * b + centre[iw]))
        out += pc.ring_wedges(r_s, r_out, n, axis, lo, hi, centre=centre)
    return out


def _socket_feature(name: str, axis: tuple[float, float, float], p: Mapping[str, float]) -> Feature:
    end = p["centre_to_end"] * MM
    return Feature(
        "socket",
        name,
        origin=(axis[0] * end, axis[1] * end, axis[2] * end),
        axis=axis,
        radius=p["socket_id"] / 2 * MM,
        depth=p["socket_depth"] * MM,
    )


def _pipe_tee(p: Mapping[str, float]) -> Shape:
    n = int(p["segments"])
    r_b = p["body_od"] / 2 * MM
    core = (p["centre_to_end"] - p["socket_depth"]) * MM
    pieces = [
        pc.prism(pc.ngon(r_b, n), "x", -core, core),  # run core (solid)
        pc.prism(pc.ngon(r_b, n), "y", 0.0, core),  # branch core, runs into the run core
    ]
    pieces += _socket_rings("x", 1.0, p) + _socket_rings("x", -1.0, p) + _socket_rings("y", 1.0, p)
    features = (
        _socket_feature("branch_socket", (0.0, 1.0, 0.0), p),
        _socket_feature("run_socket_pos", (1.0, 0.0, 0.0), p),
        _socket_feature("run_socket_neg", (-1.0, 0.0, 0.0), p),
    )
    return Shape(tuple(pieces), features)


def _pipe_elbow(p: Mapping[str, float]) -> Shape:
    n = int(p["segments"])
    r_b = p["body_od"] / 2 * MM
    core = (p["centre_to_end"] - p["socket_depth"]) * MM
    pieces = [
        pc.prism(pc.ngon(r_b, n), "x", -r_b, core),  # x leg core, closes the outer corner
        pc.prism(pc.ngon(r_b, n), "y", 0.0, core),  # y leg core, runs into the x leg core
    ]
    pieces += _socket_rings("x", 1.0, p) + _socket_rings("y", 1.0, p)
    features = (
        _socket_feature("socket_y", (0.0, 1.0, 0.0), p),
        _socket_feature("socket_x", (1.0, 0.0, 0.0), p),
    )
    return Shape(tuple(pieces), features)


def _block_check(p: Mapping[str, float]) -> list[str]:
    return [f"{k} must be at least 2 mm" for k in ("x", "y", "z") if p[k] < 2.0]


def _block(p: Mapping[str, float]) -> Shape:
    h = np.array([p["x"], p["y"], p["z"]]) * MM / 2
    return Shape((pc.box(-h, h),), yaw_symmetry=2 if abs(h[0] - h[1]) > 1e-12 else 4)


def _hex_check(p: Mapping[str, float]) -> list[str]:
    out = []
    if p["hole_d"] > 0.7 * p["across_flats"]:
        out.append("need hole_d <= 0.7 * across_flats")
    if not 0 < p["hole_depth"] <= p["length"] - 2.0:
        out.append("need 0 < hole_depth <= length - 2 mm")
    if p["hole_d"] <= 0:
        out.append("hole_d must be positive")
    return out


def _hex_standoff(p: Mapping[str, float]) -> Shape:
    hexagon = pc.ngon(p["across_flats"] / 2 * MM, 6, flat_at_deg=90.0)  # flats face +-y
    length, depth = p["length"] * MM, p["hole_depth"] * MM
    r_hole = p["hole_d"] / 2 * MM
    # Twelve full-height sectors between the hexagon and the 12-gon bore, plus a plug under the
    # blind hole. No seam runs across a gripped flat: a solid base under hole-height sectors
    # left a horizontal internal face that MuJoCo gave the pads as a ledge (gate GV2: the part
    # was carried at mu 0.05). The bore's vertices sit 15 deg off the hex corners and flat
    # midpoints, so the seams on the flats facing the jaws are 2.3 mm either side of centre and
    # no seam lies on a hex edge; the 6-fold symmetry is kept. (Two full-height slabs for the
    # gripped flats are cheaper but thin: libccd's MPR then reported 17 mm penetrations.)
    pieces = pc.sector_pieces(hexagon, r_hole, 12, "z", 0.0, length, flat_at_deg=0.0)
    pieces.append(pc.prism(pc.ngon(r_hole, 12, flat_at_deg=0.0), "z", 0.0, length - depth))
    hole = Feature("socket", "tapped_hole", (0.0, 0.0, length), (0.0, 0.0, 1.0), r_hole, depth)
    return Shape(tuple(pieces), (hole,), yaw_symmetry=6)


def _bushing_check(p: Mapping[str, float]) -> list[str]:
    out = []
    if p["bore_d"] / 2 + 1.5 > p["od"] / 2:
        out.append("need a wall of at least 1.5 mm (bore_d/2 + 1.5 <= od/2)")
    if p["flange_od"] < p["od"] + 6.0:
        out.append("need flange_od >= od + 6 mm (3 mm of rim to hook under)")
    if p["flange_thickness"] < 1.5 or p["length"] < 3.0:
        out.append("need flange_thickness >= 1.5 mm and length >= 3 mm")
    if p["segments"] < 8 or p["segments"] % 4:
        out.append("segments must be a multiple of 4, at least 8")
    return out


def _flanged_bushing(p: Mapping[str, float]) -> Shape:
    n = int(p["segments"])
    r_bore, r_od, r_fl = p["bore_d"] / 2 * MM, p["od"] / 2 * MM, p["flange_od"] / 2 * MM
    length, ft = p["length"] * MM, p["flange_thickness"] * MM
    pieces = pc.ring_wedges(r_bore, r_od, n, "z", 0.0, length)
    pieces += pc.ring_wedges(r_bore, r_fl, n, "z", length, length + ft)
    features = (
        Feature("socket", "bore", (0.0, 0.0, length + ft), (0.0, 0.0, 1.0), r_bore, length + ft),
        Feature("flange", "flange", (0.0, 0.0, length), (0.0, 0.0, -1.0), r_fl, ft),
    )
    return Shape(tuple(pieces), features, yaw_symmetry=0, register_axes=ROUND_AXES)


def _bracket_check(p: Mapping[str, float]) -> list[str]:
    out = []
    if p["wall"] < 1.5:
        out.append("wall must be at least 1.5 mm")
    if p["leg_a"] < 2 * p["wall"] + 2 or p["leg_b"] < 2 * p["wall"] + 2:
        out.append("legs must be at least 2 * wall + 2 mm")
    if p["width"] < 4.0:
        out.append("width must be at least 4 mm")
    return out


def _l_bracket(p: Mapping[str, float]) -> Shape:
    """Lying flat on leg ``a`` (along x), leg ``b`` standing at the +x end; ``width`` along y.

    The jaws close on the standing leg's two edges: the frame is centred on that leg, so
    every jaw width meets material in the pad band (bounding-box centred, the leg would sit
    16 mm off the grip axis and a 20 mm wide pad would miss it).
    """
    a, b, w, t = (p[k] * MM for k in ("leg_a", "leg_b", "width", "wall"))
    pieces = (
        pc.box((-a / 2, -w / 2, 0.0), (a / 2 - t, w / 2, t)),  # leg on the table
        pc.box((a / 2 - t, -w / 2, 0.0), (a / 2, w / 2, b)),  # standing leg at +x
    )
    return Shape(pieces, origin_xy=(a / 2 - t / 2, 0.0))


def _slug_check(p: Mapping[str, float]) -> list[str]:
    out = []
    if p["diameter"] < 2.0 or p["length"] < 2.0:
        out.append("diameter and length must be at least 2 mm")
    if p["segments"] < 8 or p["segments"] % 4:
        out.append("segments must be a multiple of 4, at least 8")
    return out


def _slug(p: Mapping[str, float]) -> Shape:
    poly = pc.ngon(p["diameter"] / 2 * MM, int(p["segments"]))
    return Shape((pc.prism(poly, "z", 0.0, p["length"] * MM),), yaw_symmetry=0, register_axes=ROUND_AXES)


def _gear_check(p: Mapping[str, float]) -> list[str]:
    out = []
    z, m = int(p["teeth"]), p["module"]
    if not 8 <= z <= 40:
        out.append("teeth must be in [8, 40]")
    if m <= 0 or p["face_width"] < 2.0:
        out.append("module must be positive and face_width at least 2 mm")
    r_f = m * (z / 2 - 1.25)
    if p["bore_d"] / 2 + 1.5 > r_f:
        out.append("need bore_d/2 + 1.5 mm <= the root radius")
    if not 10.0 <= p["pressure_angle"] <= 30.0:
        out.append("pressure_angle must be in [10, 30] degrees")
    return out


def _involute(alpha: float) -> float:
    return math.tan(alpha) - alpha


def _tooth_polygon(r_f: float, r_base: float, r_p: float, r_a: float, z: int, alpha: float) -> np.ndarray:
    """One tooth centred on +u, from the hub flat (u = r_f) to the tip circle."""
    half_pitch = math.pi / (2 * z)  # half tooth thickness angle at the pitch circle

    def half_angle(r: float) -> float:
        a_r = math.acos(min(1.0, r_base / r))
        return half_pitch + _involute(alpha) - _involute(a_r)

    pts: list[tuple[float, float]] = []
    r_lo = max(r_base, r_f)
    psi_lo = half_angle(r_lo)
    # base corners on the hub flat, on the radial continuation of the flank
    t_base = r_f * math.tan(psi_lo)
    pts += [(r_f, -t_base), (r_f, t_base)]
    for r in np.linspace(r_lo, r_a, 6):
        psi = max(half_angle(float(r)), 0.02 * half_pitch)
        u, v = r * math.cos(psi), r * math.sin(psi)
        if u >= r_f:
            pts += [(u, v), (u, -v)]
    psi_a = max(half_angle(r_a), 0.02 * half_pitch)
    pts.append((r_a, 0.0))
    pts += [
        (r_a * math.cos(psi_a / 2), r_a * math.sin(psi_a / 2)),
        (r_a * math.cos(psi_a / 2), -r_a * math.sin(psi_a / 2)),
    ]
    return pc.convex_polygon(np.array(pts))


def _spur_gear(p: Mapping[str, float]) -> Shape:
    z, m = int(p["teeth"]), p["module"] * MM
    alpha = math.radians(p["pressure_angle"])
    r_p = m * z / 2
    r_a, r_f, r_base = r_p + m, r_p - 1.25 * m, r_p * math.cos(alpha)
    width = p["face_width"] * MM
    hub = pc.ngon(r_f, z)  # one flat under each tooth, the first facing +x
    pieces = pc.sector_pieces(hub, p["bore_d"] / 2 * MM, 16, "z", 0.0, width)
    tooth = _tooth_polygon(r_f, r_base, r_p, r_a, z, alpha)
    for k in range(z):
        th = 2 * math.pi * k / z
        c, s = math.cos(th), math.sin(th)
        rot = np.array([[c, -s], [s, c]])
        pieces.append(pc.prism(tooth @ rot.T, "z", 0.0, width))
    bore = Feature("socket", "bore", (0.0, 0.0, width), (0.0, 0.0, 1.0), p["bore_d"] / 2 * MM, width)
    return Shape(tuple(pieces), (bore,), yaw_symmetry=z)


_PIPE = {
    "body_od": 27.0,
    "hub_od": 28.5,
    "socket_id": 21.5,
    "socket_depth": 16.5,
    "hub_length": 6.0,
    "centre_to_end": 30.0,
    "segments": 16,
}

FAMILIES: MappingProxyType[str, Family] = MappingProxyType(
    {
        f.name: f
        for f in (
            Family(
                "pipe_tee",
                _PIPE,
                _pipe_tee,
                _pipe_check,
                integers=frozenset({"segments"}),
                free=("centre_to_end",),
                description="slip pipe tee lying flat, branch socket toward the left jaw",
            ),
            Family(
                "block",
                {"x": 30.0, "y": 40.0, "z": 60.0},
                _block,
                _block_check,
                free=("x", "y", "z"),
                description="rectangular block, sharp edges",
            ),
            Family(
                "hex_standoff",
                {"across_flats": 17.0, "length": 45.0, "hole_d": 8.5, "hole_depth": 12.0},
                _hex_standoff,
                _hex_check,
                free=("across_flats", "length"),
                description="standing hex standoff, flats toward the jaws, tapped top hole",
            ),
            Family(
                "flanged_bushing",
                {
                    "od": 20.0,
                    "length": 30.0,
                    "flange_od": 36.0,
                    "flange_thickness": 5.0,
                    "bore_d": 12.0,
                    "segments": 16,
                },
                _flanged_bushing,
                _bushing_check,
                integers=frozenset({"segments"}),
                free=("od", "length", "flange_od"),
                description="standing flanged bushing, flange up, through bore",
            ),
            Family(
                "l_bracket",
                {"leg_a": 40.0, "leg_b": 40.0, "width": 30.0, "wall": 4.0},
                _l_bracket,
                _bracket_check,
                free=("leg_a", "leg_b", "width"),
                description="L-bracket lying flat on one leg, gripped on the edges of the standing leg",
            ),
            Family(
                "slug",
                {"diameter": 40.0, "length": 50.0, "segments": 32},
                _slug,
                _slug_check,
                integers=frozenset({"segments"}),
                free=("diameter", "length"),
                description="solid round slug, standing",
            ),
            Family(
                "pipe_elbow",
                _PIPE,
                _pipe_elbow,
                _pipe_check,
                integers=frozenset({"segments"}),
                free=("centre_to_end",),
                description="90 degree slip elbow lying flat, one socket toward the left jaw",
            ),
            Family(
                "spur_gear",
                {"module": 1.5, "teeth": 20, "face_width": 12.0, "bore_d": 8.0, "pressure_angle": 20.0},
                _spur_gear,
                _gear_check,
                integers=frozenset({"teeth"}),
                angles=frozenset({"pressure_angle"}),
                description="involute spur gear lying flat",
            ),
        )
    }
)

TRAINING_FAMILIES = ("pipe_tee", "block", "hex_standoff", "flanged_bushing", "l_bracket", "slug")
HELDOUT_FAMILIES = ("pipe_elbow", "spur_gear")


# --------------------------------------------------------------------------- PartSpec assembly


def _shift_feature(f: Feature, shift: np.ndarray) -> Feature:
    origin = pc.quantize(np.asarray(f.origin) + shift)
    return Feature(
        f.kind,
        f.name,
        tuple(float(v) for v in origin),
        f.axis,
        float(pc.quantize([f.radius])[0]),
        float(pc.quantize([f.depth])[0]),
    )


def assemble(
    name: str,
    family: str,
    shape: Shape,
    mat: Material,
    *,
    source_sha256: str = "",
    visual_asset: str | None = None,
) -> PartSpec:
    """A :class:`PartSpec` from a :class:`Shape`: centred on its bounding box (x/y on
    ``shape.origin_xy`` when the family sets it), with exact mass properties of the union of
    its pieces."""
    pts = np.concatenate([p.verts for p in shape.pieces])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    if shape.origin_xy is not None:
        centre[0], centre[1] = shape.origin_xy
    shift = pc.quantize(-centre)
    pieces = [pc._piece_from_polys(pc.quantize(p.verts + shift), p.polys) for p in shape.pieces]
    props = pc.mass_properties(pieces)
    return PartSpec(
        name=name,
        family=family,
        pieces=tuple(p.verts for p in pieces),
        faces=tuple(p.faces for p in pieces),
        volume=props.volume,
        com=props.com,
        unit_inertia=props.unit_inertia,
        material=mat,
        yaw_symmetry=shape.yaw_symmetry,
        register_axes=shape.register_axes,
        features=tuple(_shift_feature(f, shift) for f in shape.features),
        visual_asset=visual_asset,
        source_sha256=source_sha256,
        decomposition="exact",
    )


def params_sha256(family: str, params: Mapping[str, float]) -> str:
    """SHA-256 of a family name and its resolved dimensions (the part's procedural source)."""
    payload = {"library": LIBRARY_VERSION, "family": family, "params": {k: params[k] for k in sorted(params)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_part(
    name: str, family: str, params: Mapping[str, Any] | None = None, mat: Material | str | None = None
) -> PartSpec:
    """Build one part of ``family`` with ``params`` (millimetres, defaults filled in).

    ``source_sha256`` of the result is :func:`params_sha256` of the resolved dimensions.
    ``mat`` defaults to the family's training material, or SCAN_CLAY for families without one.
    """
    fam = FAMILIES.get(family)
    if fam is None:
        raise KeyError(f"unknown part family {family!r}; known: {sorted(FAMILIES)}")
    resolved = fam.resolve(params)
    if mat is None:
        mat = DEFAULT_MATERIALS.get(family, SCAN_CLAY)
    if isinstance(mat, str):
        mat = material(mat)
    return assemble(name, family, fam.build(resolved), mat, source_sha256=params_sha256(family, resolved))


DEFAULT_MATERIALS: MappingProxyType[str, str] = MappingProxyType(
    {
        "pipe_tee": "pvc_white",
        "block": "anodised_aluminium",
        "hex_standoff": "zinc_plated_steel",
        "flanged_bushing": "brass",
        "l_bracket": "stamped_steel",
        "slug": "ground_steel",
        "pipe_elbow": "pvc_white",
        "spur_gear": "ground_steel",
    }
)


# --------------------------------------------------------------------------- config file


@dataclass(frozen=True, eq=False)
class PartEntry:
    """One part of the library file: its name, family, material and resolved dimensions (mm)."""

    name: str
    family: str
    material: str
    params: Mapping[str, float]
    description: str = ""

    def build(self) -> PartSpec:
        return build_part(self.name, self.family, self.params, self.material)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "material": self.material,
            "params_mm": dict(self.params),
            "description": self.description,
        }


@dataclass(frozen=True)
class PartsLibrary:
    """A parsed ``parts_v2.yaml``: the named parts in file order."""

    name: str
    entries: tuple[PartEntry, ...]
    path: Path | None = field(default=None, compare=False)

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def entry(self, name: str) -> PartEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(f"no part {name!r} in {self.name}; parts: {list(self.names())}")


def parts_library_from_dict(raw: Mapping[str, Any], where: str = "parts") -> PartsLibrary:
    unknown = set(raw) - {"name", "parts"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    entries = []
    for i, item in enumerate(raw.get("parts") or ()):
        if not isinstance(item, Mapping):
            raise TypeError(f"{where}: parts[{i}] must be a mapping")
        extra = set(item) - {"name", "family", "material", "params", "description"}
        if extra:
            raise ValueError(f"{where}: parts[{i}]: unknown keys {sorted(extra)}")
        family = str(item["family"])
        if family not in FAMILIES:
            raise ValueError(f"{where}: parts[{i}]: unknown family {family!r}")
        mat = str(item.get("material") or DEFAULT_MATERIALS[family])
        material(mat)
        params = FAMILIES[family].resolve(item.get("params") or {})
        entries.append(
            PartEntry(
                str(item["name"]), family, mat, MappingProxyType(params), str(item.get("description", ""))
            )
        )
    names = [e.name for e in entries]
    if not entries or len(set(names)) != len(names):
        raise ValueError(f"{where}: needs at least one part and unique names, got {names}")
    return PartsLibrary(str(raw.get("name", "parts")), tuple(entries))


@functools.lru_cache(maxsize=8)
def _load_library(path: str, mtime_ns: int) -> PartsLibrary:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        lib = parts_library_from_dict(yaml.safe_load(fh) or {}, where=str(p))
    return PartsLibrary(lib.name, lib.entries, path=p)


def load_parts_library(path: str | Path | None = None) -> PartsLibrary:
    """Load ``configs/parts_v2.yaml`` (or ``path``)."""
    p = Path(path) if path is not None else DEFAULT_PARTS_FILE
    return _load_library(str(p.resolve()), p.stat().st_mtime_ns)


@functools.lru_cache(maxsize=256)
def _build_cached(name: str, family: str, mat: str, params: tuple[tuple[str, float], ...]) -> PartSpec:
    return build_part(name, family, dict(params), mat)


def _built(entry: PartEntry) -> PartSpec:
    return _build_cached(entry.name, entry.family, entry.material, tuple(sorted(entry.params.items())))


def training_parts(path: str | Path | None = None) -> tuple[PartSpec, ...]:
    """Every part of the library file, built, in file order (cached; parts are immutable)."""
    return tuple(_built(e) for e in load_parts_library(path).entries)


def get_part(name: str, path: str | Path | None = None) -> PartSpec:
    return _built(load_parts_library(path).entry(name))


def part_table(path: str | Path | None = None) -> list[dict[str, Any]]:
    """One row per training part, for prompts and docs: family, dimensions (mm), features,
    material, mass (g), yaw symmetry and the size of its bounding box (mm)."""
    lib = load_parts_library(path)
    rows = []
    for e in lib.entries:
        part = _built(e)
        rows.append(
            {
                "name": e.name,
                "family": e.family,
                "description": e.description or FAMILIES[e.family].description,
                "dimensions_mm": {k: v for k, v in e.params.items() if k not in FAMILIES[e.family].integers},
                "size_mm": [round(1000 * s, 2) for s in part.size],
                "material": e.material,
                "mass_g": round(1000 * part.mass, 1),
                "yaw_symmetry": part.yaw_symmetry,
                "register_axes": list(part.register_axes),
                "features": [
                    {
                        "kind": f.kind,
                        "name": f.name,
                        "faces_jaw": f.facing(),
                        "radius_mm": round(1000 * f.radius, 2),
                        "depth_mm": round(1000 * f.depth, 2),
                    }
                    for f in part.features
                ],
            }
        )
    return rows


def part_digest(part: PartSpec) -> dict[str, Any]:
    """Everything about a part that scoring and visuals depend on, JSON-ready (for digests)."""
    return {
        "name": part.name,
        "family": part.family,
        "geometry_sha256": part.geometry_sha256,
        "material": {
            "name": part.material.name,
            "density": part.material.density,
            "base_color": list(part.material.base_color),
            "metallic": part.material.metallic,
            "roughness": part.material.roughness,
        },
        "yaw_symmetry": part.yaw_symmetry,
        "register_axes": list(part.register_axes),
        "features": [
            {
                "kind": f.kind,
                "name": f.name,
                "origin": list(f.origin),
                "axis": list(f.axis),
                "radius": f.radius,
                "depth": f.depth,
            }
            for f in part.features
        ],
        "source_sha256": part.source_sha256,
        "decomposition": part.decomposition,
    }


def library_digest(path: str | Path | None = None) -> str:
    """SHA-256 of the parts library for PREREG: version, entries and every built part."""
    lib = load_parts_library(path)
    payload = {
        "library_version": LIBRARY_VERSION,
        "name": lib.name,
        "entries": [e.describe() for e in lib.entries],
        "parts": [part_digest(p) for p in training_parts(path)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- scans and meshes


def part_from_mesh_object(
    obj: MeshObject,
    *,
    family: str = "gso",
    mat: Material = SCAN_CLAY,
    yaw_symmetry: int | None = None,
    visual_asset: str | None = None,
    yaw_tol: float = 0.5 * MM,
) -> PartSpec:
    """A scanned or uploaded :class:`~rtt.sim.objects.MeshObject` as a :class:`PartSpec`.

    The pieces, faces, mass properties and provenance carry over unchanged, so
    ``geometry_sha256`` is identical. The density is the material's (an object's own
    ``density``, when set, replaces the material's). Yaw symmetry is detected from the
    pieces' footprint (:func:`rtt.parts.features.detect_yaw_symmetry`) unless given; a
    round footprint does not register yaw.
    """
    from rtt.parts.features import detect_yaw_symmetry

    if obj.density is not None and obj.density != mat.density:
        mat = Material(mat.name, obj.density, mat.base_color, mat.metallic, mat.roughness)
    sym = detect_yaw_symmetry(obj.parts, tol=yaw_tol) if yaw_symmetry is None else int(yaw_symmetry)
    axes = ROUND_AXES if sym == 0 else REGISTER_AXES
    return PartSpec(
        name=obj.name,
        family=family,
        pieces=obj.parts,
        faces=obj.faces,
        volume=obj.volume,
        com=obj.com,
        unit_inertia=obj.unit_inertia,
        material=mat,
        yaw_symmetry=sym,
        register_axes=axes,
        visual_asset=visual_asset,
        source_sha256=obj.source_sha256,
        pose_index=obj.pose_index,
        pose_probability=obj.pose_probability,
        decomposition=obj.decomposition,
    )


def part_pieces(part: PartSpec) -> list[pc.Piece]:
    """The part's pieces as :class:`~rtt.parts.pieces.Piece` (with halfspaces), from its
    vertices: polygon faces are recovered by merging coplanar triangles."""
    out = []
    for verts, tris in zip(part.pieces, part.faces, strict=True):
        out.append(piece_from_mesh(verts, tris))
    return out


def piece_from_mesh(verts: np.ndarray, tris: np.ndarray) -> pc.Piece:
    """A :class:`~rtt.parts.pieces.Piece` from a convex triangle mesh (coplanar triangles
    merged into polygon faces)."""
    import trimesh

    hull = trimesh.Trimesh(np.asarray(verts, dtype=np.float64), np.asarray(tris), process=False)
    normals = np.asarray(hull.face_normals)
    tri_pts = np.asarray(hull.triangles)
    offsets = np.einsum("ij,ij->i", normals, tri_pts[:, 0])
    key = np.round(np.c_[normals, offsets / max(1e-9, float(np.ptp(verts)))], 7)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    polys = []
    for g in range(len(first)):
        idx = np.nonzero(inverse.ravel() == g)[0]
        vids = sorted({int(v) for t in idx for v in tris[t]})
        n = normals[first[g]]
        pts = np.asarray(verts)[vids]
        ordered = pc._order_on_plane(list(pts), n)
        if ordered is None:
            continue
        index = [vids[int(np.argmin(np.linalg.norm(pts - q, axis=1)))] for q in ordered]
        polys.append(tuple(index))
    return pc._piece_from_polys(np.asarray(verts, dtype=np.float64), polys)


def mirror_y(part: PartSpec, name: str | None = None) -> PartSpec:
    """The part mirrored in y (the closing axis), for building the right jaw as a left one.

    Vertices ``y -> -y`` with face winding reversed (faces stay outward), centre of mass and
    the ``Ixy``/``Iyz`` products negated, features mirrored (a socket facing the left jaw
    faces the right one). Exact: mirroring twice gives back the original geometry hash.
    The name defaults to ``<name>.mirror_y``.
    """
    flip = np.array([1.0, -1.0, 1.0])
    pieces = tuple(pc.quantize(v * flip) for v in part.pieces)
    faces = tuple(np.ascontiguousarray(f[:, [0, 2, 1]]) for f in part.faces)
    ixx, iyy, izz, ixy, ixz, iyz = part.unit_inertia
    features = tuple(
        Feature(
            f.kind,
            f.name,
            (f.origin[0], -f.origin[1] + 0.0, f.origin[2]),
            (f.axis[0], -f.axis[1] + 0.0, f.axis[2]),
            f.radius,
            f.depth,
        )
        for f in part.features
    )
    return PartSpec(
        name=name or f"{part.name}.mirror_y",
        family=part.family,
        pieces=pieces,
        faces=faces,
        volume=part.volume,
        com=(part.com[0], -part.com[1] + 0.0, part.com[2]),
        unit_inertia=(ixx, iyy, izz, -ixy + 0.0, ixz, -iyz + 0.0),
        material=part.material,
        yaw_symmetry=part.yaw_symmetry,
        register_axes=part.register_axes,
        features=features,
        visual_asset=None,
        source_sha256=part.source_sha256,
        pose_index=part.pose_index,
        pose_probability=part.pose_probability,
        decomposition=part.decomposition,
    )


def visual_mesh(part: PartSpec) -> Any:
    """The part's watertight visual mesh: the exact union of its collision pieces
    (``trimesh.Trimesh``, metres, object frame). Smooth shading is the renderer's job."""
    return pc.union_mesh(list(part.pieces), list(part.faces))


def parts_by_name(parts: Sequence[PartSpec]) -> dict[str, PartSpec]:
    return {p.name: p for p in parts}
