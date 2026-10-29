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
  the panel, at most a half turn);
* Studio v2's job templates: a panel on its rails (the panel, its rails and stops and the clamp's
  post, all on local y about the pose); a tile and its bed (``tile_mm`` + 40 mm square); a push
  button (a wall plate behind its face, or a desk console); a machine with a sliding door (the
  enclosure, long enough along local -y to park the open door, its front leaf toward local -x).
  A wall button also has an **approach**: the plate's width for :data:`FRONT_REACH_MM` in front of
  its face, the way a tool tilted to point at it comes in, which nothing else may stand in (a
  closed door in front of a call button blocks the press: measured, the button is never touched).

:func:`layout_problems` refuses two objects whose footprints or swings come within
:data:`CLEARANCE_MM` of each other [assumed], with three exceptions: a lid at its bin's position
(the lid covers the bin), the plug and socket of one port (one object), and a bin or a part that
stands inside a machine's interior under its loading opening (:func:`inside_machine`: a fixture
bin, a finished part on the machine bed). A part a step picks into a bin therefore starts outside
that bin and outside every lid and door.

It also refuses an object a tool cannot work on from inside the wrist's reach
(:func:`reach_problems`; the palm travels +-350 mm, :data:`WORK_LIMIT_MM` keeps 12 mm of that for
aim and compliance): a panel whose centre or clamp lies past it, a wall button whose face is
pressed by a tool tilted to point at it (which puts the wrist up to :data:`FRONT_REACH_MM` in front
of the face), and a sliding door whose handle slides past it as the door opens. Each is a layout
problem on the object's pose (``objects[2].pose (call_button): ...``), not a plan the simulator
finds it cannot reach.

:func:`fix_layout` is the deterministic layout fixer (track S4e) the Task Engineer runs after its
repair call when layout problems are all that is left: it moves the free objects (every object
but a lid on its bin, which moves with its bin; a port's plug is part of its port) to the nearest
spot that clears every other object, searching a fixed order of offsets (rings of a 10 mm grid
around the object's position, nearest first, then counter-clockwise from +x) inside the pose
limits (the wrist's reach), where the moved object is also within reach. Bins a step places into
keep their place when moving something else clears the conflict; what stands inside a machine
moves with the machine, as a lid with its bin, and is kept apart from the machine's other contents
inside its interior (and, under rule set v2, what the plan's words put inside a machine is placed
there first). Each move is reported, so the plan states it as an assumption.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Final

from rtt.sim.scene import PALM_XY_LIMIT
from rtt.studio.types import ObjectInstance, ObjectPose, Step, TemplateKind

__all__ = [
    "CLEARANCE_MM",
    "FINGER_ROOM_MM",
    "FRONT_REACH_MM",
    "INSIDE",
    "REACH",
    "WORK_LIMIT_MM",
    "LayoutMove",
    "Polygon",
    "Region",
    "fix_layout",
    "inside_machine",
    "layout_problems",
    "panel_reach_mm",
    "picked_inside",
    "reach_problem",
    "reach_problems",
    "regions",
    "room_problems",
]

#: Plan-view gap kept between the footprints and swings of separate objects (mm) [assumed].
CLEARANCE_MM: Final = 20.0

Polygon = list[tuple[float, float]]
Region = tuple[str, Polygon]  # ("footprint" | "swing" | "approach", convex polygon, counter-clockwise)

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
# Studio v2's job templates (the builders' constants in mm; tests compare them with the modules)
PANEL_RAIL_PAST_MM: Final = 10.0  # rail beyond its stop and the clamp's post
PANEL_STOP_MM: Final = 10.0  # an end stop along the rail
PANEL_POST_MM: Final = 20.0  # the clamp's post, square
PANEL_CLAMP_OUT_MM: Final = 25.0  # the clamp's axis beyond the panel's edge
TILE_BORDER_MM: Final = 20.0  # the bed beyond the tile, each side
BUTTON_PLATE_MM: Final = (160.0, 20.0, 80.0)  # a wall plate: width, thickness, width past the button
BUTTON_CONSOLE_MM: Final = (120.0, 40.0)  # a desk console: side, side past the button
MACHINE_WALL_MM: Final = 10.0  # walls and roof
MACHINE_LEAF_MM: Final = 6.0  # the door's leaves
MACHINE_GAP_MM: Final = 2.0  # between the door and the enclosure
MACHINE_OVERLAP_MM: Final = 10.0  # of the door past the opening each side
#: Room kept between what stands inside a machine and its walls and opening edges (mm) [assumed].
INSIDE_MARGIN_MM: Final = 10.0
#: Room a gripper's open fingers need around a part picked inside a machine, from what else stands
#: there, from the machine's walls and from the opening's edges, the open door's included (mm)
#: [measured: the 85 mm adaptive gripper's fingers met a tray and the open door's
#: edge 20 mm from a 50 mm part and cleared both at 30 mm; 10 mm kept spare for other grippers and
#: parts, assumed].
FINGER_ROOM_MM: Final = 40.0
#: The palm's x and y travel either side of the table centre (mm), from ``rtt.sim.scene.PALM_XY_LIMIT``.
PALM_LIMIT_MM: Final = 1000.0 * PALM_XY_LIMIT
#: The reach a layout keeps every work point inside (mm): the palm's travel less 12 mm, the
#: controllers' 10 mm compliance and 2 mm aim [assumed].
WORK_LIMIT_MM: Final = PALM_LIMIT_MM - 12.0
#: How far in front of a face a tool tilted to point at it puts the wrist (mm) [assumed]: the
#: longest Studio tool (355 mm, flange to tip) plus the push's 10 mm standoff and 30 mm approach,
#: rounded up, so the layout lets every tool reach the face and none is ruled out by it.
FRONT_REACH_MM: Final = 400.0
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
    if k is TemplateKind.PANEL_RAIL:
        length, width = float(p["length_mm"]), float(p["width_mm"])
        reach = panel_reach_mm(width)
        y0 = -reach - 0.5 * PANEL_POST_MM - PANEL_RAIL_PAST_MM
        y1 = reach + 0.5 * width + float(p["start_gap_mm"]) + PANEL_STOP_MM + PANEL_RAIL_PAST_MM
        return [("footprint", _rect(-0.5 * length, y0, 0.5 * length, y1))]
    if k is TemplateKind.TILE_BED:
        h = 0.5 * float(p["tile_mm"]) + TILE_BORDER_MM
        return [("footprint", _rect(-h, -h, h, h))]
    if k is TemplateKind.SPRING_BUTTON:
        d = float(p["diameter_mm"])
        if p["mount"] == "wall":  # the plate behind its face (local +y), the button out of it (-y)
            width, thick, past = BUTTON_PLATE_MM
            h = 0.5 * max(width, d + past)
            proud = float(p["proud_mm"])
            # and the way a tool tilted to point at the face comes in: kept clear as far as the wrist
            return [
                ("footprint", _rect(-h, -proud, h, thick)),
                ("approach", _rect(-h, -proud - FRONT_REACH_MM, h, -proud)),
            ]
        side, past = BUTTON_CONSOLE_MM
        h = 0.5 * max(side, d + past)
        return [("footprint", _rect(-h, -h, h, h))]
    if k is TemplateKind.SLIDING_DOOR:
        hx, hw = 0.5 * float(p["depth_mm"]), 0.5 * float(p["width_mm"])
        y0 = -hw - MACHINE_OVERLAP_MM - float(p["travel_mm"]) - MACHINE_WALL_MM
        y1 = hw + MACHINE_OVERLAP_MM + MACHINE_WALL_MM
        return [("footprint", _rect(-hx - MACHINE_GAP_MM - MACHINE_LEAF_MM, y0, hx, y1))]
    return []


def panel_reach_mm(width_mm: float) -> float:
    """How far a panel's top centre and its clamp each lie from the panel_rail's pose (mm, on local
    y): ``(width / 2 + clamp out) / 2`` (``rtt.studio.templates.panel_rail.reach_of``)."""
    return 0.5 * (0.5 * float(width_mm) + PANEL_CLAMP_OUT_MM)


def _interior(o: ObjectInstance) -> tuple[float, float, float, float]:
    """A machine's interior under its loading opening, less :data:`INSIDE_MARGIN_MM` (local mm:
    ``x0, y0, x1, y1``)."""
    hx, hw = 0.5 * float(o.params["depth_mm"]), 0.5 * float(o.params["width_mm"])
    m = INSIDE_MARGIN_MM
    return -hx + MACHINE_WALL_MM + m, -hw + m, hx - MACHINE_WALL_MM - m, hw - m


def _local(o: ObjectInstance, x: float, y: float) -> tuple[float, float]:
    """Table mm to ``o``'s local frame (before its yaw)."""
    a = math.radians(o.pose.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - o.pose.x_mm, y - o.pose.y_mm
    return c * dx + s * dy, -s * dx + c * dy


def inside_machine(machine: ObjectInstance, other: ObjectInstance) -> bool:
    """Whether ``other`` (a bin or a part) stands inside the interior of ``machine`` (a sliding door)
    under its loading opening, every corner of its footprint :data:`INSIDE_MARGIN_MM` clear of the
    walls and the opening's edges, and no taller than the interior less that margin."""
    if machine.kind is not TemplateKind.SLIDING_DOOR or other.kind not in (
        TemplateKind.BIN,
        TemplateKind.PART,
    ):
        return False
    tall = float(other.params["height_mm" if other.kind is TemplateKind.BIN else "size_z_mm"])
    if tall > float(machine.params["height_mm"]) - MACHINE_WALL_MM - INSIDE_MARGIN_MM + 1e-9:
        return False
    x0, y0, x1, y1 = _interior(machine)
    for _, poly in regions(other):
        for x, y in poly:
            lx, ly = _local(machine, x, y)
            if not (x0 - 1e-9 <= lx <= x1 + 1e-9 and y0 - 1e-9 <= ly <= y1 + 1e-9):
                return False
    return True


def _world(o: ObjectInstance, x: float, y: float) -> tuple[float, float]:
    a = math.radians(o.pose.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return o.pose.x_mm + c * x - s * y, o.pose.y_mm + s * x + c * y


def _out(x: float, y: float) -> bool:
    return max(abs(x), abs(y)) > WORK_LIMIT_MM + 1e-9


def _mm_pair(x: float, y: float) -> str:
    return f"({_mm(x)}, {_mm(y)})"


def reach_problem(o: ObjectInstance) -> str | None:
    """Why a tool cannot work on ``o`` from inside the wrist's reach (module docstring), or ``None``:
    its pose (a panel's centre and clamp, a wall button's face and tilted wrist, a sliding door's
    handle along its travel) or the path where a tool works on it (:func:`_work_path`)."""
    lim = _mm(WORK_LIMIT_MM)
    if o.kind is TemplateKind.PANEL_RAIL:
        reach = panel_reach_mm(float(o.params["width_mm"]))
        bound = WORK_LIMIT_MM - reach
        if max(abs(o.pose.x_mm), abs(o.pose.y_mm)) > bound + 1e-9:
            return (
                f"the panel's centre and its clamp lie {_mm(reach)} mm either side of the pose (on local y), so "
                f"keep x_mm and y_mm within -{_mm(bound)} to {_mm(bound)} for the wrist to reach both at any yaw "
                f"(it reaches -{lim} to {lim} mm); it is at {_mm_pair(o.pose.x_mm, o.pose.y_mm)}"
            )
        return None
    if o.kind is TemplateKind.SPRING_BUTTON and o.params["mount"] == "wall":
        ex, ey = _world(o, 0.0, -1.0)
        nx, ny = ex - o.pose.x_mm, ey - o.pose.y_mm  # the face's outward normal on the table
        proud = float(o.params["proud_mm"])
        face = (o.pose.x_mm + proud * nx, o.pose.y_mm + proud * ny)
        wrist = (face[0] + FRONT_REACH_MM * nx, face[1] + FRONT_REACH_MM * ny)
        if _out(*face) or _out(*wrist):
            return (
                f"a wall button is pressed by a tool tilted to point at its face, which puts the wrist up to "
                f"{_mm(FRONT_REACH_MM)} mm in front of it; facing {_mm_pair(nx, ny)} from {_mm_pair(*face)} the "
                f"wrist would reach {_mm_pair(*wrist)}, past -{lim} to {lim} mm; stand it so its face looks toward "
                "the table's centre with that room in front of it"
            )
        return None
    if o.kind is TemplateKind.SLIDING_DOOR:
        travel = float(o.params["travel_mm"])
        end = _world(o, 0.0, -travel)
        if _out(o.pose.x_mm, o.pose.y_mm) or _out(*end):
            return (
                f"its handle slides travel_mm ({_mm(travel)} mm) toward local -y as the door opens, from "
                f"{_mm_pair(o.pose.x_mm, o.pose.y_mm)} to {_mm_pair(*end)}, past -{lim} to {lim} mm; move it "
                "or shorten the travel"
            )
        return None
    path = _work_path(o)
    if path is not None:
        what, motion, points = path
        worst = max(points, key=lambda q: max(abs(q[0]), abs(q[1])))
        if _out(*worst):
            return (
                f"its {what} {motion}, to {_mm_pair(*worst)}, past -{lim} to {lim} mm where the wrist works; move "
                "it toward the table's centre"
            )
    return None


def _work_path(o: ObjectInstance) -> tuple[str, str, list[tuple[float, float]]] | None:
    """Where a tool works on ``o`` as it moves (table mm): a lid's free edge from closed to its stop, a
    door's free edge along its swing, a drawer's handle out to its travel, a port's plug; ``None``
    for kinds a tool works on at the pose (within the pose limits, so within reach)."""
    p, k = o.params, o.kind
    if k is TemplateKind.HINGED_LID:
        d, t = float(p["depth_mm"]), float(p["thickness_mm"])
        stop = math.radians(float(p["max_angle_deg"]))
        steps = max(1, math.ceil(math.degrees(stop) / ARC_STEP_DEG))
        angles = [stop * i / steps for i in range(steps + 1)]
        if p["orientation"] == "lid":  # the free edge's middle, from closed to the stop about the hinge
            hinge = 0.5 * d
            pts = [_world(o, 0.0, hinge - d * math.cos(a)) for a in angles]
            return "free edge (the lid's lip or handle)", "swings up and over toward its hinge", pts
        hx = (
            -0.5 * d
        )  # the hinge; the free edge's front (its handle) at (d, -front) from it, turned by the swing
        handle = float(p["feature_mm"]) + BAR_DIAMETER_MM if p["feature"] == "handle" else 0.0
        front = 0.5 * t + handle
        pts = [
            _world(o, hx + d * math.cos(a) - front * math.sin(a), -d * math.sin(a) - front * math.cos(a))
            for a in angles
        ]
        return "free edge and handle", "swing open toward local -y", pts
    if k is TemplateKind.DRAWER:
        hd = 0.5 * float(p["depth_mm"])
        handle = p["handle"]
        ext = (
            float(p["handle_mm"]) + BAR_DIAMETER_MM
            if handle == "bar"
            else (float(p["handle_mm"]) + KNOB_HEAD_MM if handle == "knob" else 0.0)
        )
        front = hd + DRAWER_FRONT_MM + ext
        travel = float(p["travel_mm"])
        return (
            "handle",
            "pulls out by travel_mm toward local -y",
            [_world(o, 0.0, -front), _world(o, 0.0, -front - travel)],
        )
    if k is TemplateKind.PORT_PLUG:
        dia = float(p["plug_diameter_mm"])
        chamfer = min(float(p["chamfer_mm"]), 0.8 * float(p["depth_mm"]))
        r_out = 0.5 * (dia + float(p["clearance_mm"])) + max(PORT_WALL_MM, 0.5 * dia) + chamfer
        offset = r_out / math.cos(math.pi / PORT_SIDES) + PLUG_GAP_MM + 0.5 * dia
        return "plug", "stands beside its socket along local +x", [_world(o, offset, 0.0)]
    return None


def reach_problems(objects: Sequence[ObjectInstance]) -> list[str]:
    """:func:`reach_problem` of every object, on its pose's path."""
    out = []
    for i, o in enumerate(objects):
        why = reach_problem(o)
        if why is not None:
            out.append(f"objects[{i}].pose ({o.name}): {why}")
    return out


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


def _in_machine(a: ObjectInstance, b: ObjectInstance) -> bool:
    """One of the two stands inside the other, a machine (:func:`inside_machine`)."""
    return inside_machine(a, b) or inside_machine(b, a)


def _exempt(a: ObjectInstance, b: ObjectInstance) -> bool:
    """A pair the clearance rule allows to overlap: a lid on its bin, or something inside a machine."""
    return _lid_on_bin(a, b) or _in_machine(a, b)


Conflict = tuple[int, int, str, str, float]  # (i, j, region of i, region of j, distance), i < j


def _area(regs: list[Region]) -> float:
    """The area of a footprint's bounding box (mm^2): a size to place the largest first."""
    x0, y0, x1, y1 = _bbox(regs[0][1])
    return (x1 - x0) * (y1 - y0)


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
            skip = (i, j) in covered if covered is not None else _exempt(a, b)
            if a.name == b.name or skip:
                continue
            hit = _hit(placed[i], placed[j])
            if hit is not None:
                out.append((i, j, hit[0], hit[1], hit[2]))
    return out


def picked_inside(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> dict[int, tuple[int, int]]:
    """What a pick step picks from inside a machine: the object's index to ``(machine's index, the
    first such step's index)``."""
    from rtt.studio.types import PrimitiveKind

    picks = (PrimitiveKind.GRASP, PrimitiveKind.SUCTION, PrimitiveKind.WRAP)
    index = {o.name: k for k, o in enumerate(objects)}
    out: dict[int, tuple[int, int]] = {}
    for i, s in enumerate(steps):
        k = index.get(s.object)
        if s.primitive not in picks or k is None or k in out:
            continue
        for m, machine in enumerate(objects):
            if inside_machine(machine, objects[k]):
                out[k] = (m, i)
                break
    return out


def _room_bounds(machine: ObjectInstance) -> tuple[float, float]:
    """Where a part picked inside ``machine`` may stand, :data:`FINGER_ROOM_MM` clear of its walls
    and the opening's fixed (+y) edge (local mm: ``|x| <= x_lim``, ``y <= y_lim``; the open door's
    edge, on -y, is the hook pull's travel, :func:`rtt.studio.engineer.checks`)."""
    hx, hw = 0.5 * float(machine.params["depth_mm"]), 0.5 * float(machine.params["width_mm"])
    return hx - MACHINE_WALL_MM - FINGER_ROOM_MM, hw - FINGER_ROOM_MM


def _in_room(machine: ObjectInstance, o: ObjectInstance) -> bool:
    x_lim, y_lim = _room_bounds(machine)
    return all(
        abs(lx) <= x_lim + 1e-9 and ly <= y_lim + 1e-9
        for _, poly in regions(o)
        for lx, ly in (_local(machine, x, y) for x, y in poly)
    )


def _room_pairs(
    objects: Sequence[ObjectInstance], picked: Mapping[int, tuple[int, int]], placed: Sequence[list[Region]]
) -> list[tuple[int, int, float]]:
    """``(picked part, other, distance)`` for every bin or part inside the same machine that stands
    closer than :data:`FINGER_ROOM_MM` to a part a step picks there."""
    out = []
    for k, (m, _) in sorted(picked.items()):
        for j, o in enumerate(objects):
            if (
                j == k
                or o.kind not in (TemplateKind.BIN, TemplateKind.PART)
                or not inside_machine(objects[m], o)
            ):
                continue
            d = min(_distance(pa, pb) for _, pa in placed[k] for _, pb in placed[j])
            if d < FINGER_ROOM_MM - 1e-9:
                out.append((k, j, d))
    return out


def room_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    """Rule set v2's finger room (:data:`FINGER_ROOM_MM`) around a part a step picks inside a machine:
    from the machine's walls and the opening's +y edge, and from every other bin or part inside it."""
    picked = picked_inside(objects, steps)
    placed = [regions(o) for o in objects]
    room = f"{FINGER_ROOM_MM:g}"
    out = []
    for k, (m, i) in sorted(picked.items()):
        o, machine, s = objects[k], objects[m], steps[i]
        if not _in_room(machine, o):
            x_lim, y_lim = _room_bounds(machine)
            out.append(
                f"objects[{k}].pose ({o.name}): steps[{i}] ({s.id}) picks it inside {machine.name}, where a "
                f"gripper's open fingers need {room} mm around it; keep its footprint {room} mm from the walls and "
                f"the opening's +y edge: within {_mm(x_lim)} mm of {machine.name}'s pose along its local x and at "
                f"most {_mm(y_lim)} mm toward its local +y"
            )
    for k, j, d in _room_pairs(objects, picked, placed):
        m, i = picked[k]
        out.append(
            f"objects[{j}] ({objects[j].name}): it stands {_mm(d)} mm from objects[{k}] ({objects[k].name}), which "
            f"steps[{i}] ({steps[i].id}) picks inside {objects[m].name}; a gripper's open fingers need {room} mm "
            f"around it there: move objects[{j}].pose or objects[{k}].pose to keep them {room} mm apart"
        )
    return out


def layout_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step], rules: str = "v2") -> list[str]:
    """Objects out of reach (:func:`reach_problems`, rule set v2), then pairs of objects whose
    footprints or swings come within :data:`CLEARANCE_MM` (one problem per pair, on the later
    object's path), then (rule set v2) the finger room around a part picked inside a machine
    (:func:`room_problems`)."""
    out: list[str] = reach_problems(objects) if rules == "v2" else []
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
    if rules == "v2":
        out += room_problems(objects, steps)
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
    clears: str  # "the swing of tote_lid", REACH: brought within reach, or INSIDE: put inside its machine
    #: the machine it was moved inside of (a move within a machine's interior), else None
    inside: str | None = None

    def text(self) -> str:
        """The plan's assumption line for this move."""
        what = self.names[0] + (f" (with {', '.join(self.names[1:])})" if len(self.names) > 1 else "")
        where = f"({_mm(self.before[0])}, {_mm(self.before[1])}) to ({_mm(self.after[0])}, {_mm(self.after[1])}) mm"
        if self.inside is not None:
            if self.clears == INSIDE:
                why = "as the plan says"
            elif self.clears.startswith("room "):
                why = f"to leave {self.clears}"
            else:
                why = f"to clear the {self.clears}"
            return f"Moved {what} from {where}, inside {self.inside} under its loading opening, {why}"
        why = "bring it within the wrist's reach" if self.clears == REACH else f"clear the {self.clears}"
        return f"Moved {what} from {where} to {why}"


#: :attr:`LayoutMove.clears` of a move that brought an object within reach ...
REACH: Final = "reach"
#: ... and of one that put an object the plan says stands inside a machine there.
INSIDE: Final = "inside"
#: The grid a move within a machine's interior searches (mm, in the machine's frame).
INNER_STEP_MM: Final = 5.0


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


def _moved(o: ObjectInstance, dx: float, dy: float) -> ObjectInstance:
    return replace(o, pose=ObjectPose(o.pose.x_mm + dx, o.pose.y_mm + dy, o.pose.yaw_deg))


def fix_layout(
    objects: Sequence[ObjectInstance],
    steps: Sequence[Step],
    rules: str = "v2",
    inside: Mapping[str, str] | None = None,
) -> tuple[list[ObjectInstance], list[LayoutMove]] | None:
    """Objects moved so that every one is within reach (:func:`reach_problem`) and no footprint or
    swing comes within the clearance of another's, and the moves (one per object, bin-with-lid or
    machine-with-contents, from its first position to its last), or ``None`` when no spot inside the
    pose limits does it. Deterministic: the same plan gives the same moves.

    What stands inside a machine (:func:`inside_machine`), and under rule set v2 what ``inside``
    (thing's name to machine's name: what the plan's words put there) says does, belongs to the
    machine: it moves with it, and it is placed and kept apart from the rest of the machine's
    contents inside its interior under the loading opening (a fixture tray, a finished part, a
    blank), on a :data:`INNER_STEP_MM` grid, nearest spot first; when no single move clears two
    of them, all of them are placed again (step targets first, then bins, then parts)."""
    objs = list(objects)
    n = len(objs)
    parent = list(range(n))
    home: dict[int, int] = {}  # what belongs inside a machine -> the machine

    def out_of_reach(o: ObjectInstance) -> str | None:  # rule set v1 has no reach rules
        return reach_problem(o) if rules == "v2" else None

    for j, o in enumerate(objs):
        for i, b in enumerate(objs):
            if i == j:
                continue
            if o.kind is TemplateKind.HINGED_LID and _lid_on_bin(o, b):
                parent[j] = i  # a lid on its bin moves with the bin
                break
            if inside_machine(b, o):
                parent[j] = i  # what stands inside a machine moves with the machine
                home[j] = i
                break
    if rules == "v2" and inside:
        index = {o.name: k for k, o in enumerate(objs)}
        for name, machine in inside.items():
            j, i = index.get(name), index.get(machine)
            if (
                j is None
                or i is None
                or j in home
                or parent[j] != j
                or objs[i].kind is not TemplateKind.SLIDING_DOOR
                or objs[j].kind not in (TemplateKind.BIN, TemplateKind.PART)
            ):
                continue
            parent[j] = i
            home[j] = i

    def root(k: int) -> int:
        seen = set()
        while parent[k] != k and k not in seen:
            seen.add(k)
            k = parent[k]
        return k

    group = [root(k) for k in range(n)]
    # the pairs allowed to overlap: a lid and its bin, a machine and what stands inside it (not two
    # things inside one machine: they are kept apart inside it)
    covered = {(min(j, parent[j]), max(j, parent[j])) for j in range(n) if parent[j] != j}
    members = {g: [k for k in range(n) if group[k] == g] for g in sorted(set(group))}
    targets = {s.target for s in steps if s.target is not None}
    anchored = {
        g: any(objs[k].kind is TemplateKind.BIN and objs[k].name in targets for k in ks)
        for g, ks in members.items()
    }
    placed = [regions(o) for o in objs]
    first: dict[int, tuple[tuple[float, float], str]] = {}
    inner: dict[int, tuple[tuple[float, float], str]] = {}  # moves within a machine, by object
    limit = ObjectPose.LIMIT_MM

    def spot_for(g: int) -> tuple[float, float] | None:
        ks = members[g]
        others = [k for k in range(n) if group[k] != g]
        for dx, dy in _offsets():
            if any(abs(objs[k].pose.x_mm + dx) > limit or abs(objs[k].pose.y_mm + dy) > limit for k in ks):
                continue
            if any(out_of_reach(_moved(objs[k], dx, dy)) is not None for k in ks):
                continue
            if _clear([_shift(placed[k], dx, dy) for k in ks], [placed[m] for m in others]):
                return dx, dy
        return None

    def move(g: int, dx: float, dy: float, clears: str) -> None:
        if g not in first:
            lead = objs[g]
            first[g] = ((lead.pose.x_mm, lead.pose.y_mm), clears)
        for k in members[g]:
            objs[k] = _moved(objs[k], dx, dy)
            placed[k] = _shift(placed[k], dx, dy)

    def carried(k: int) -> list[int]:
        """``k`` and what moves with it inside its machine (a lid on it)."""
        return [k, *(j for j in range(n) if j != k and parent[j] == k)]

    pick_steps = [s for s in steps if s.primitive.value in ("grasp", "suction", "wrap")]
    pick_names = {s.object for s in pick_steps}
    picked = {k for k in home if objs[k].name in pick_names} if rules == "v2" else set()
    # what a pick works on inside a machine stays where the plan's hook pull uncovers it (rule set v2): the
    # door opens the larger of what the pull asks and what the thing needed where it stood
    involved = pick_names | {s.target for s in pick_steps if s.target is not None}

    def door_need(k: int, o: ObjectInstance) -> float:
        m = objs[home[k]]
        low = min(_local(m, x, y)[1] for _, poly in regions(o) for x, y in poly)
        room = FINGER_ROOM_MM if k in picked else CLEARANCE_MM
        return 0.5 * float(m.params["width_mm"]) + MACHINE_OVERLAP_MM - low + room

    opened: dict[int, float] = {}
    if rules == "v2":
        for k in home:
            if objs[k].name not in involved:
                continue
            m = objs[home[k]]
            asked = [
                c.at_least
                for s in steps
                if s.object == m.name and s.primitive.value == "hook_pull"
                for c in s.success
                if c.quantity == "travel_mm" and c.at_least is not None
            ]
            door = max(asked) if asked else float(m.params["travel_mm"])
            opened[k] = max(door, door_need(k, objs[k]))

    def fits(k: int, dx: float, dy: float, ignore: frozenset[int] = frozenset()) -> bool:
        """``k`` moved by ``(dx, dy)`` stands inside its machine, within reach, clear of everything but
        its machine (and ``ignore``), and with the finger room a part picked there needs (rule set v2)."""
        m = objs[home[k]]
        ks = carried(k)
        if any(abs(objs[j].pose.x_mm + dx) > limit or abs(objs[j].pose.y_mm + dy) > limit for j in ks):
            return False  # past the pose limits
        moved = _moved(objs[k], dx, dy)
        if not inside_machine(m, moved) or out_of_reach(moved) is not None:
            return False
        if k in picked and not _in_room(m, moved):
            return False
        if k in opened and door_need(k, moved) > opened[k] + 1e-9:
            return False  # the plan's hook pull would no longer uncover it
        others = [j for j in range(n) if j not in ks and j != home[k] and j not in ignore]
        regs = [_shift(placed[j], dx, dy) for j in ks]
        if not _clear(regs, [placed[j] for j in others]):
            return False
        roomy = [j for j in others if home.get(j) == home[k] and (k in picked or j in picked)]
        return all(
            _distance(pa, pb) >= FINGER_ROOM_MM - 1e-9
            for j in roomy
            for _, pa in regs[0]
            for _, pb in placed[j]
        )

    def inner_spot(
        k: int, ignore: frozenset[int] = frozenset(), toward: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:
        """The offset nearest ``toward`` (a table point; by default where ``k`` stands) that
        :func:`fits` ``k``, on the :data:`INNER_STEP_MM` grid of its machine's interior."""
        m = objs[home[k]]
        x0, y0, x1, y1 = _interior(m)
        step = INNER_STEP_MM
        cx, cy = objs[k].pose.x_mm, objs[k].pose.y_mm
        tx, ty = toward if toward is not None else (cx, cy)
        spots = []
        for iu in range(math.ceil(x0 / step), math.floor(x1 / step) + 1):
            for iv in range(math.ceil(y0 / step), math.floor(y1 / step) + 1):
                wx, wy = _world(m, iu * step, iv * step)
                dx, dy = round(wx - cx, 6), round(wy - cy, 6)
                if toward is None and dx == 0.0 and dy == 0.0:
                    continue
                ox, oy = wx - tx, wy - ty
                spots.append((math.hypot(ox, oy), math.atan2(oy, ox) % (2.0 * math.pi), dx, dy))
        for _, _, dx, dy in sorted(spots):
            if fits(k, dx, dy, ignore):
                return dx, dy
        return None

    def inner_move(k: int, dx: float, dy: float, clears: str) -> None:
        if k not in inner:
            inner[k] = ((objs[k].pose.x_mm, objs[k].pose.y_mm), clears)
        for j in carried(k):
            objs[j] = _moved(objs[j], dx, dy)
            placed[j] = _shift(placed[j], dx, dy)

    def pinned(k: int) -> bool:
        return objs[k].kind is TemplateKind.BIN and objs[k].name in targets

    def place_all(order: list[int], clears: Callable[[int], str], corner: tuple[float, float] | None) -> bool:
        left = set(order)
        for n_k, k in enumerate(order):
            left.discard(k)
            ignore = frozenset(j for x in left for j in carried(x))
            toward = corner if n_k == 0 else None
            if toward is None and fits(k, 0.0, 0.0, ignore):
                continue  # where it stands is clear of what is placed so far
            spot = inner_spot(k, ignore, toward)
            if spot is None:
                return False
            if spot != (0.0, 0.0):
                inner_move(k, *spot, clears(k))
        return True

    def repack(machine: int, clears: Callable[[int], str]) -> bool:
        """Place every content of ``machine`` again, one by one: step targets, bins, then parts, each
        where it stands or nearest it; failing that, the largest one first in a corner of the interior
        (nearest corner first), the rest nearest where they stand."""
        inside_it = [k for k in sorted(home) if home[k] == machine]
        order = sorted(inside_it, key=lambda k: (not pinned(k), objs[k].kind is not TemplateKind.BIN, k))
        snapshot = (list(objs), [list(r) for r in placed], dict(inner))

        def restore() -> None:
            objs[:], placed[:] = list(snapshot[0]), [list(r) for r in snapshot[1]]
            inner.clear()
            inner.update(snapshot[2])

        if place_all(order, clears, None):
            return True
        restore()
        m = objs[machine]
        x0, y0, x1, y1 = _interior(m)
        big = max(inside_it, key=lambda k: (_area(placed[k]), -k))
        rest = [k for k in order if k != big]
        here = (objs[big].pose.x_mm, objs[big].pose.y_mm)
        corners = sorted(
            (_world(m, u, v) for u in (x0, x1) for v in (y0, y1)),
            key=lambda c: (
                math.hypot(c[0] - here[0], c[1] - here[1]),
                math.atan2(c[1] - here[1], c[0] - here[0]),
            ),
        )
        for corner in corners:
            if place_all([big, *rest], clears, corner):
                return True
            restore()
        return False

    # what the plan says stands inside a machine goes there first
    for k in sorted(home):
        if not inside_machine(objs[home[k]], objs[k]):
            spot = inner_spot(k)
            if spot is None and not repack(
                home[k], lambda x, k=k: INSIDE if x == k else f"room inside {objs[home[k]].name}"
            ):
                return None
            if spot is not None:
                inner_move(k, *spot, INSIDE)

    for _ in range(4 * n + 4):
        far = [g for g, ks in members.items() if any(out_of_reach(objs[k]) is not None for k in ks)]
        if far:
            g = far[0]
            spot = spot_for(g)
            if spot is None:
                return None
            move(g, *spot, REACH)
            continue
        conflicts = _conflicts(objs, placed, covered)
        if not conflicts:
            # the finger room around what a step picks inside a machine (rule set v2)
            cramped = [k for k in sorted(picked) if not _in_room(objs[home[k]], objs[k])]
            pairs = _room_pairs(objs, {k: (home[k], 0) for k in picked}, placed)
            if not cramped and not pairs:
                break
            k, other = (cramped[0], None) if cramped else (pairs[0][0], pairs[0][1])
            why = "room for a gripper's fingers" + (f" from {objs[other].name}" if other is not None else "")
            for x in (
                [k] if other is None else sorted((k, other), key=lambda x: (pinned(x), 0 if x == k else 1))
            ):
                spot = inner_spot(x)
                if spot is not None:
                    inner_move(x, *spot, why)
                    break
            else:
                if not repack(home[k], lambda x, why=why: why):
                    return None
            continue
        i, j, ri, rj, _ = conflicts[0]
        if i in home and j in home and home[i] == home[j]:  # two things inside one machine
            for k in sorted((i, j), key=lambda k: (pinned(k), 0 if k == j else 1)):
                spot = inner_spot(k)
                if spot is not None:
                    other, region = (i, ri) if k == j else (j, rj)
                    inner_move(k, *spot, f"{region} of {objs[other].name}")
                    break
            else:
                if not repack(
                    home[i],
                    lambda x, i=i, j=j, ri=ri, rj=rj: (
                        f"{ri} of {objs[j].name}" if x == i else f"{rj} of {objs[i].name}"
                    ),
                ):
                    return None
            continue
        options = sorted({group[i], group[j]}, key=lambda g: (anchored[g], 0 if g == group[j] else 1))
        for g in options:
            spot = spot_for(g)
            if spot is None:
                continue
            other = i if g == group[j] else j
            region = ri if g == group[j] else rj
            move(g, *spot, f"{region} of {objs[other].name}")
            break
        else:
            return None
    else:
        return None
    if _conflicts(objs, placed, covered) or any(out_of_reach(o) is not None for o in objs):
        return None
    if any(not inside_machine(objs[home[k]], objs[k]) for k in home):
        return None
    if any(not _in_room(objs[home[k]], objs[k]) for k in picked) or _room_pairs(
        objs, {k: (home[k], 0) for k in picked}, placed
    ):
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
    moves += [
        LayoutMove(
            names=tuple(objs[j].name for j in carried(k)),
            before=before,
            after=(objs[k].pose.x_mm, objs[k].pose.y_mm),
            clears=clears,
            inside=objs[home[k]].name,
        )
        for k, (before, clears) in sorted(inner.items())
        if (objs[k].pose.x_mm, objs[k].pose.y_mm) != before
    ]
    return objs, moves
