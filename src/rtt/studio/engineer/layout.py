"""Plan-view footprints of the Studio's objects and the layout check (track S4c).

S2's template builders (``rtt.studio.templates``) put every object's geometry at its pose; this
module gives each object's plan-view **footprint** (what stands on the table) and its **swing**
(the area a lid, a door or a drawer sweeps from closed to its stop), as convex polygons in mm on
the table, from the same params and the same constants as the builders:

* a bin: ``length_mm`` x ``width_mm``; a part: ``size_x_mm`` x ``size_y_mm`` (a cylinder
  ``size_x_mm`` across);
* a latch's plate and a threaded joint's base: ``max(1.6 x diameter, 40 mm)`` square;
* a valve: the wheel (``rim_diameter_mm``) over its 70 x 50 mm body;
* a port: its socket (the bore, a wall of ``max(6 mm, plug/2)`` and the chamfer, over a 16-sided
  polygon) and the plug standing 40 mm beyond it along local +x;
* a drawer: its carcass plus the front and handle toward local -y; its swing is the drawer slid
  out by ``travel_mm``;
* a lid: the panel, its hinge on the +y edge and the back wall behind it; past vertical its swing
  reaches ``depth_mm x -cos(max_angle_deg)`` beyond the hinge;
* a door: the panel along x hinged at ``x = -depth_mm/2`` with its jamb, stop post and handle;
  its swing is the circular sector of radius ``depth_mm`` (plus the handle) about the hinge,
  from closed to ``max_angle_deg`` toward local -y (and a little further for what stands in front of
  the panel, at most a half turn).

:func:`layout_problems` refuses two objects whose footprints or swings come within
:data:`CLEARANCE_MM` of each other [assumed], with two exceptions: a lid at its bin's position
(the lid covers the bin), and the plug and socket of one port (one object). A part a step picks
into a bin therefore starts outside that bin and outside every lid and door.

:func:`fix_layout` is the deterministic layout fixer (track S4e) the Task Engineer runs after its
repair call when layout problems are all that is left: it moves the free objects (every object
but a lid on its bin, which moves with its bin; a port's plug is part of its port) to the nearest
spot that clears every other object, searching a fixed order of offsets (rings of a 10 mm grid
around the object's position, nearest first, then counter-clockwise from +x) inside the pose
limits (the wrist's reach). Bins a step places into keep their place when moving something else
clears the conflict. Each move is reported, so the plan states it as an assumption.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Final

from rtt.studio.types import ObjectInstance, ObjectPose, Step, TemplateKind

__all__ = ["CLEARANCE_MM", "LayoutMove", "Polygon", "Region", "fix_layout", "layout_problems", "regions"]

#: Plan-view gap kept between the footprints and swings of separate objects (mm) [assumed].
CLEARANCE_MM: Final = 20.0

Polygon = list[tuple[float, float]]
Region = tuple[str, Polygon]  # ("footprint" | "swing", convex polygon, counter-clockwise)

# The builders' constants (m in the builders, mm here); tests compare them with the modules.
DOOR_JAMB_MM: Final = (15.0, 5.0, 1.0, 21.0)  # x behind/ahead of the hinge, y from t/2 (start, end)
BAR_DIAMETER_MM: Final = 10.0
DRAWER_FRONT_MM: Final = 13.0  # 1 mm gap plus the 12 mm overlay front
KNOB_HEAD_MM: Final = 6.0
VALVE_BODY_MM: Final = (70.0, 50.0)
LATCH_PLATE_MIN_MM: Final = 40.0
PORT_WALL_MM: Final = 6.0
PORT_SIDES: Final = 16
PLUG_GAP_MM: Final = 40.0
LID_BACK_WALL_MM: Final = 8.0
ARC_STEP_DEG: Final = 15.0
#: The layout fixer's search grid and the farthest it moves an object.
FIX_STEP_MM: Final = 10.0
FIX_MAX_MM: Final = 850.0


def _rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _place(poly: Polygon, o: ObjectInstance) -> Polygon:
    """Local mm (before yaw) to table mm."""
    a = math.radians(o.pose.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return [(o.pose.x_mm + c * x - s * y, o.pose.y_mm + s * x + c * y) for x, y in poly]


def _local_regions(o: ObjectInstance) -> list[Region]:
    p = o.params
    k = o.kind
    if k is TemplateKind.BIN:
        hx, hy = 0.5 * float(p["length_mm"]), 0.5 * float(p["width_mm"])
        return [("footprint", _rect(-hx, -hy, hx, hy))]
    if k is TemplateKind.PART:
        hx = 0.5 * float(p["size_x_mm"])
        hy = hx if p["shape"] == "cylinder" else 0.5 * float(p["size_y_mm"])
        return [("footprint", _rect(-hx, -hy, hx, hy))]
    if k is TemplateKind.QUARTER_TURN_LATCH:
        h = 0.5 * max(1.6 * float(p["head_diameter_mm"]), LATCH_PLATE_MIN_MM)
        return [("footprint", _rect(-h, -h, h, h))]
    if k is TemplateKind.THREADED_JOINT:
        h = 0.5 * max(1.6 * float(p["diameter_mm"]), LATCH_PLATE_MIN_MM)
        return [("footprint", _rect(-h, -h, h, h))]
    if k is TemplateKind.HANDWHEEL_VALVE:
        r = 0.5 * float(p["rim_diameter_mm"])
        hx, hy = max(r, 0.5 * VALVE_BODY_MM[0]), max(r, 0.5 * VALVE_BODY_MM[1])
        return [("footprint", _rect(-hx, -hy, hx, hy))]
    if k is TemplateKind.PORT_PLUG:
        dia = float(p["plug_diameter_mm"])
        chamfer = min(float(p["chamfer_mm"]), 0.8 * float(p["depth_mm"]))
        r_out = 0.5 * (dia + float(p["clearance_mm"])) + max(PORT_WALL_MM, 0.5 * dia) + chamfer
        k_out = r_out / math.cos(math.pi / PORT_SIDES)
        offset = k_out + PLUG_GAP_MM + 0.5 * dia
        hy = max(k_out, 0.5 * dia)
        return [("footprint", _rect(-k_out, -hy, offset + 0.5 * dia, hy))]
    if k is TemplateKind.DRAWER:
        hw, hd = 0.5 * float(p["width_mm"]), 0.5 * float(p["depth_mm"])
        handle = p["handle"]
        ext = (
            float(p["handle_mm"]) + BAR_DIAMETER_MM
            if handle == "bar"
            else (float(p["handle_mm"]) + KNOB_HEAD_MM if handle == "knob" else 0.0)
        )
        front = hd + DRAWER_FRONT_MM + ext
        travel = float(p["travel_mm"])
        return [
            ("footprint", _rect(-hw, -front, hw, hd)),
            ("swing", _rect(-hw, -front - travel, hw, hd - travel)),
        ]
    if k is TemplateKind.HINGED_LID:
        w, d, t = (float(p[n]) for n in ("width_mm", "depth_mm", "thickness_mm"))
        stop = math.radians(float(p["max_angle_deg"]))
        handle = float(p["feature_mm"]) + BAR_DIAMETER_MM if p["feature"] == "handle" else 0.0
        if p["orientation"] == "lid":
            out: list[Region] = [
                ("footprint", _rect(-0.5 * w, -0.5 * d, 0.5 * w, 0.5 * d + t + LID_BACK_WALL_MM))
            ]
            beyond = -math.cos(stop) * (d + handle)
            if beyond > 0.0:
                out.append(("swing", _rect(-0.5 * w, 0.5 * d, 0.5 * w, 0.5 * d + t + beyond)))
            return out
        hx = -0.5 * d
        j_back, _, _, j_y1 = DOOR_JAMB_MM
        front = 0.5 * t + handle
        body = _rect(hx - j_back, -front, 0.5 * d, 0.5 * t + j_y1)
        r = math.hypot(d, front)
        # a handle stands in front of the panel, so it sweeps a little past the stop (at most a half turn)
        sweep = min(math.pi, stop + math.atan2(front, 0.5 * d))
        n = max(1, math.ceil(math.degrees(sweep) / ARC_STEP_DEG))
        arc = [
            (hx + r * math.cos(-sweep * i / n), r * math.sin(-sweep * i / n)) for i in range(n, -1, -1)
        ]  # from the open stop back to closed: counter-clockwise about the hinge
        return [("footprint", body), ("swing", [(hx, 0.0), *arc])]
    return []


def regions(o: ObjectInstance) -> list[Region]:
    """The object's footprint and swing as convex polygons in table mm (counter-clockwise)."""
    return [(name, _place(poly, o)) for name, poly in _local_regions(o)]


# --------------------------------------------------------------------------- geometry


def _axes(poly: Polygon) -> list[tuple[float, float]]:
    out = []
    for i, (x0, y0) in enumerate(poly):
        x1, y1 = poly[(i + 1) % len(poly)]
        out.append((y0 - y1, x1 - x0))
    return out


def _intersect(a: Polygon, b: Polygon) -> bool:
    """Separating-axis test of two convex polygons (touching counts as intersecting)."""
    for ax, ay in _axes(a) + _axes(b):
        pa = [ax * x + ay * y for x, y in a]
        pb = [ax * x + ay * y for x, y in b]
        if max(pa) < min(pb) or max(pb) < min(pa):
            return False
    return True


def _point_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    den = dx * dx + dy * dy
    u = 0.0 if den == 0.0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / den))
    return math.hypot(px - (x0 + u * dx), py - (y0 + u * dy))


def _distance(a: Polygon, b: Polygon) -> float:
    """Plan-view distance between two convex polygons (0 when they meet)."""
    if _intersect(a, b):
        return 0.0
    best = math.inf
    for p, q in ((a, b), (b, a)):
        for px, py in p:
            for i, (x0, y0) in enumerate(q):
                x1, y1 = q[(i + 1) % len(q)]
                best = min(best, _point_segment(px, py, x0, y0, x1, y1))
    return best


# --------------------------------------------------------------------------- the check


def _lid_on_bin(a: ObjectInstance, b: ObjectInstance) -> bool:
    lid, other = (a, b) if a.kind is TemplateKind.HINGED_LID else (b, a)
    return (
        lid.kind is TemplateKind.HINGED_LID
        and lid.params["orientation"] == "lid"
        and other.kind is TemplateKind.BIN
        and abs(lid.pose.x_mm - other.pose.x_mm) < 1.0
        and abs(lid.pose.y_mm - other.pose.y_mm) < 1.0
    )


Conflict = tuple[int, int, str, str, float]  # (i, j, region of i, region of j, distance), i < j


def _bbox(poly: Polygon) -> tuple[float, float, float, float]:
    xs, ys = [x for x, _ in poly], [y for _, y in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _hit(ra: list[Region], rb: list[Region]) -> tuple[str, str, float] | None:
    """The first pair of regions (in order) that comes within :data:`CLEARANCE_MM`."""
    for na, pa in ra:
        ax0, ay0, ax1, ay1 = _bbox(pa)
        for nb, pb in rb:
            bx0, by0, bx1, by1 = _bbox(pb)
            gap = max(bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1)
            if gap >= CLEARANCE_MM:
                continue  # the boxes are already clear, so the polygons are
            dist = _distance(pa, pb)
            if dist < CLEARANCE_MM:
                return na, nb, dist
    return None


def _conflicts(
    objects: Sequence[ObjectInstance],
    placed: Sequence[list[Region]],
    covered: set[tuple[int, int]] | None = None,
) -> list[Conflict]:
    """Every pair (later object ``j``, earlier ``i``) that comes within the clearance; ``covered``
    lists the lid-on-bin pairs to skip (by default the pairs that are a lid on its bin now)."""
    out: list[Conflict] = []
    for j, b in enumerate(objects):
        for i in range(j):
            a = objects[i]
            skip = (i, j) in covered if covered is not None else _lid_on_bin(a, b)
            if a.name == b.name or skip:
                continue
            hit = _hit(placed[i], placed[j])
            if hit is not None:
                out.append((i, j, hit[0], hit[1], hit[2]))
    return out


def layout_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    """Pairs of objects whose footprints or swings come within :data:`CLEARANCE_MM` (one problem
    per pair, on the later object's path)."""
    out: list[str] = []
    placed = [regions(o) for o in objects]
    targets = {(s.object, s.target) for s in steps if s.target is not None}
    for i, j, ra, rb, dist in _conflicts(objects, placed):
        a, b = objects[i], objects[j]
        verb = "overlaps" if dist == 0.0 else f"comes within {CLEARANCE_MM:g} mm of"
        note = ""
        if (b.name, a.name) in targets or (a.name, b.name) in targets:
            note = "; a part picked into a bin starts outside it"
        out.append(
            f"objects[{j}] ({b.name}): its {rb} {verb} the {ra} of objects[{i}] ({a.name}); keep them at "
            f"least {CLEARANCE_MM:g} mm apart{note}"
        )
    return out


# --------------------------------------------------------------------------- the fixer


def _mm(v: float) -> str:
    return f"{round(v, 1) + 0.0:g}"


@dataclass(frozen=True)
class LayoutMove:
    """One object (with any lid on it) the fixer moved, and what that cleared."""

    names: tuple[str, ...]
    before: tuple[float, float]
    after: tuple[float, float]
    clears: str  # "the swing of tote_lid"

    def text(self) -> str:
        """The plan's assumption line for this move."""
        what = self.names[0] + (f" (with {', '.join(self.names[1:])})" if len(self.names) > 1 else "")
        return (
            f"Moved {what} from ({_mm(self.before[0])}, {_mm(self.before[1])}) to ({_mm(self.after[0])}, "
            f"{_mm(self.after[1])}) mm to clear the {self.clears}"
        )


@cache
def _offsets() -> tuple[tuple[float, float], ...]:
    """The search order: grid offsets nearest first, then counter-clockwise from +x."""
    n = int(FIX_MAX_MM // FIX_STEP_MM)
    out = []
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            r = math.hypot(ix, iy)
            if 0 < r * FIX_STEP_MM <= FIX_MAX_MM:
                angle = math.atan2(iy, ix) % (2.0 * math.pi)
                out.append((r, angle, ix * FIX_STEP_MM, iy * FIX_STEP_MM))
    out.sort()
    return tuple((dx, dy) for _, _, dx, dy in out)


def _shift(regs: list[Region], dx: float, dy: float) -> list[Region]:
    return [(n, [(x + dx, y + dy) for x, y in poly]) for n, poly in regs]


def _clear(moving: list[list[Region]], fixed: list[list[Region]]) -> bool:
    return all(_hit(a, b) is None for a in moving for b in fixed)


def fix_layout(
    objects: Sequence[ObjectInstance], steps: Sequence[Step]
) -> tuple[list[ObjectInstance], list[LayoutMove]] | None:
    """Objects moved so that no footprint or swing comes within the clearance of another's, and the
    moves (one per object or bin-with-lid, from its first position to its last), or ``None`` when no
    spot inside the pose limits clears a conflict. Deterministic: the same plan gives the same moves."""
    objs = list(objects)
    n = len(objs)
    group = list(range(n))
    for j, o in enumerate(objs):
        for i, b in enumerate(objs):
            if i != j and _lid_on_bin(o, b) and o.kind is TemplateKind.HINGED_LID:
                group[j] = i  # a lid on its bin moves with the bin
                break
    covered = {(min(i, j), max(i, j)) for j in range(n) for i in range(n) if i != j and group[j] == i}
    members = {g: [k for k in range(n) if group[k] == g] for g in set(group)}
    targets = {s.target for s in steps if s.target is not None}
    anchored = {
        g: any(objs[k].kind is TemplateKind.BIN and objs[k].name in targets for k in ks)
        for g, ks in members.items()
    }
    placed = [regions(o) for o in objs]
    first: dict[int, tuple[tuple[float, float], str]] = {}
    limit = ObjectPose.LIMIT_MM
    for _ in range(4 * n + 4):
        conflicts = _conflicts(objs, placed, covered)
        if not conflicts:
            break
        i, j, ri, rj, _ = conflicts[0]
        options = sorted({group[i], group[j]}, key=lambda g: (anchored[g], 0 if g == group[j] else 1))
        for g in options:
            ks = members[g]
            others = [k for k in range(n) if group[k] != g]
            spot = None
            for dx, dy in _offsets():
                if any(
                    abs(objs[k].pose.x_mm + dx) > limit or abs(objs[k].pose.y_mm + dy) > limit for k in ks
                ):
                    continue
                if _clear([_shift(placed[k], dx, dy) for k in ks], [placed[m] for m in others]):
                    spot = (dx, dy)
                    break
            if spot is None:
                continue
            dx, dy = spot
            if g not in first:
                other = i if g == group[j] else j
                region = ri if g == group[j] else rj
                lead = objs[g]
                first[g] = ((lead.pose.x_mm, lead.pose.y_mm), f"{region} of {objs[other].name}")
            for k in ks:
                o = objs[k]
                objs[k] = replace(o, pose=ObjectPose(o.pose.x_mm + dx, o.pose.y_mm + dy, o.pose.yaw_deg))
                placed[k] = _shift(placed[k], dx, dy)
            break
        else:
            return None
    else:
        return None
    if _conflicts(objs, placed, covered):
        return None
    moves = [
        LayoutMove(
            names=tuple(objs[k].name for k in members[g]),
            before=before,
            after=(objs[g].pose.x_mm, objs[g].pose.y_mm),
            clears=clears,
        )
        for g, (before, clears) in sorted(first.items())
        if (objs[g].pose.x_mm, objs[g].pose.y_mm) != before
    ]
    return objs, moves
