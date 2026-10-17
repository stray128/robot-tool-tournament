"""Validators for designed recipes, cheapest first. Every problem is a readable sentence.

:func:`validate_recipe` returns ``[]`` for a recipe that may be built for ``part``; otherwise
the problems of the first failing stage, each as ``"<code>: <sentence>"``. Stages, in order:

1. **schema** (``out_of_bounds``): known generator, every parameter present, in bounds, of
   the right type, nothing extra.
2. **structure** (part-free): ``unprintable`` (phalanx slenderness, minimum feature sizes, hook
   lips, pad bumps), ``self_intersecting`` (digit footprints overlap on the palm in the open
   pose), ``too_heavy`` (mass above 1.5 kg), ``over_budget`` (more than 12 servos, more than
   1.5 N m on a joint at the nominal contact lever, more than 50 N in total).
3. **fit** (this part): ``too_narrow`` (in the open pose some digit's innermost point does not
   clear the part's outline in its direction, allowing for the placement error envelope and
   the part's yaw error), ``too_short`` (the digits cannot reach the grip height with the palm
   at least 10 mm above the part), ``above_part`` (the effector's lowest point at the grasp
   is less than 5 mm below the part's top: it would close above it).
4. **closing sweep** (this part, MuJoCo kinematics and collision only, no dynamics): the
   effector is posed at the grasp over the centred part and closed along its synergy in 40
   steps; each digit stops when it touches the part. ``digit_collision`` (two digits
   interpenetrate before both reach the part), ``cannot_reach`` (fewer than two effector
   bodies touch the part when fully closed), ``no_opposition`` (the digits that reach it
   are less than 90 deg apart, so they cannot squeeze it).

Recipes of other parametric families (the soft arm, the suction cup) get stage 1, then a trial
build through their own family (``not_buildable`` carries its ``InvalidEffector`` reason).

Results are cached per (recipe, part geometry, task), so a build right after a validation
costs nothing extra.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import mujoco
import numpy as np

from rtt.effectors.morphology import generator as gen
from rtt.effectors.morphology.layout import (
    MAX_ACTUATORS,
    MAX_EFFECTOR_MASS_KG,
    MAX_JOINT_TORQUE_NM,
    MAX_TOTAL_FORCE_N,
    HandLayout,
    footprint,
    layout_of,
)
from rtt.effectors.morphology.spaces import design_spaces, space_label
from rtt.effectors.scene import MOUNT_Z
from rtt.effectors.types import EffectorFamily, InvalidEffector, MorphologyRecipe

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

__all__ = [
    "CHECK_CODES",
    "ERROR_ENVELOPE_MM",
    "MIN_OPEN_CLEARANCE_MM",
    "PALM_CLEARANCE_MM",
    "YAW_ENVELOPE_DEG",
    "fit_numbers",
    "needed_inner_mm",
    "part_outline",
    "support_mm",
    "validate_recipe",
    "validate_structure",
    "validation_codes",
]

#: Placement error the open effector must clear: the certificate's +-3 mm plus the 0.5 mm jitter.
ERROR_ENVELOPE_MM: Final = 3.5
#: Yaw error of the part the open effector must clear (train_v2 c3 is 6 deg, jitter 1 deg).
YAW_ENVELOPE_DEG: Final = 7.0
#: Extra gap between the open digits and the outline beyond the error envelope.
MIN_OPEN_CLEARANCE_MM: Final = 1.5
#: The palm's underside must stay this far above the part's top at the grasp (the core rule).
PALM_CLEARANCE_MM: Final = 10.0
#: The effector's lowest point must be at least this far below the part's top at the grasp.
MIN_OVERLAP_MM: Final = 5.0
#: Printability (PLA) and machinability (metals): minimum feature and maximum slenderness.
MIN_FEATURE_MM: Final = {"pla": 2.0, "aluminium": 1.0, "steel": 1.0}
MAX_SLENDERNESS: Final = {"pla": 15.0, "aluminium": 25.0, "steel": 30.0}
MIN_PHALANX_MM: Final = 8.0
#: Digits closer than this (m) at their nearest points count as touching in the sweep.
_TOUCH: Final = 0.0003
#: Penetration between two digits that counts as a collision in the sweep (m).
_COLLIDE: Final = 0.0005
_SWEEP_STEPS: Final = 40
_YAW_SAMPLES: Final = 5

CHECK_CODES: Final[tuple[str, ...]] = (
    "out_of_bounds",
    "unprintable",
    "self_intersecting",
    "too_heavy",
    "over_budget",
    "too_narrow",
    "too_short",
    "above_part",
    "digit_collision",
    "cannot_reach",
    "no_opposition",
    "not_buildable",
)

_CACHE: collections.OrderedDict[tuple[str, str, str], tuple[str, ...]] = collections.OrderedDict()
_CACHE_SIZE = 4096


def validation_codes(problems: Sequence[str]) -> list[str]:
    """The ``code`` of each problem string (``"code: sentence"``)."""
    return [p.split(":", 1)[0] for p in problems]


# --------------------------------------------------------------------------- part outline


def part_outline(part: PartSpec) -> np.ndarray:
    """The part's vertices projected on the table plane (object frame, m), shape (n, 2)."""
    return np.concatenate([np.asarray(v)[:, :2] for v in part.pieces])


def support_mm(points: np.ndarray, angle: float) -> float:
    """How far (mm) the outline reaches from the part's centre along direction ``angle`` (rad)."""
    u = np.array([math.cos(angle), math.sin(angle)])
    return 1000.0 * float((points @ u).max())


def needed_inner_mm(part: PartSpec, angle: float) -> float:
    """How far (mm) an open digit pointing along ``angle`` (part frame) must keep its innermost point
    from the centre: the outline's reach within the yaw envelope plus the error and clearance."""
    pts = part_outline(part)
    yaw_err = math.radians(YAW_ENVELOPE_DEG)
    reach = max(
        support_mm(pts, angle - yaw_err + 2.0 * yaw_err * k / (_YAW_SAMPLES - 1)) for k in range(_YAW_SAMPLES)
    )
    return reach + ERROR_ENVELOPE_MM + MIN_OPEN_CLEARANCE_MM


def fit_numbers(layout: HandLayout, part: PartSpec) -> list[dict[str, float]]:
    """Per digit: its direction on the part, its open inner radius and what it must clear (mm)."""
    out = []
    for d in layout.digits:
        phi = d.angle + layout.yaw  # the digit's direction in the part frame
        needed = needed_inner_mm(part, phi)
        out.append(
            {
                "digit": float(d.index + 1),
                "direction_deg": math.degrees(math.remainder(phi, 2.0 * math.pi)),
                "inner_mm": 1000.0 * d.inner_radius,
                "outline_mm": needed - ERROR_ENVELOPE_MM - MIN_OPEN_CLEARANCE_MM,
                "needed_mm": needed,
            }
        )
    return out


# --------------------------------------------------------------------------- stages


def _schema(recipe: MorphologyRecipe) -> list[str]:
    spaces = design_spaces()
    key = (recipe.family, recipe.kind)
    if key not in spaces:
        known = ", ".join(space_label(k) for k in spaces)
        return [f"out_of_bounds: unknown generator {space_label(key)!r} (known: {known})"]
    return [f"out_of_bounds: {p}" for p in recipe.problems(spaces[key])]


def _structure(layout: HandLayout) -> list[str]:
    out: list[str] = []
    mat = layout.material
    feature = MIN_FEATURE_MM[mat]
    slender = MAX_SLENDERNESS[mat]
    for d in layout.digits[:1]:  # every digit of a layout is identical
        for j, link in enumerate(d.links):
            L = 1000.0 * link.length
            if link.shape == "capsule":
                r = 1000.0 * link.radius
                if r < feature:
                    out.append(
                        f"unprintable: segment {j + 1} is {2 * r:.1f} mm across; {mat} needs at least "
                        f"{2 * feature:.1f} mm (raise root_radius_mm or taper)"
                    )
                    break
                if L < 1000.0 * d.links[0].radius + 2.0:
                    out.append(
                        f"unprintable: {L:.1f} mm segments are shorter than the root radius + 2 mm "
                        f"({1000.0 * d.links[0].radius + 2.0:.1f} mm); use fewer segments or a longer digit"
                    )
                    break
                continue
            t = 1000.0 * link.thickness
            if L < max(MIN_PHALANX_MM, t):
                out.append(
                    f"unprintable: phalanx {j + 1} is {L:.1f} mm long; a jointed link needs at least "
                    f"{max(MIN_PHALANX_MM, t):.1f} mm (change proximal_frac or finger_length_mm)"
                )
            if L / t > slender:
                out.append(
                    f"unprintable: phalanx {j + 1} is {L:.0f} mm long and {t:.1f} mm thick (slenderness "
                    f"{L / t:.1f}); {mat} allows at most {slender:g} (thicker or shorter fingers)"
                )
            if link.pad == "round" and link.pad_thickness > link.length / 2.0:
                out.append(
                    f"unprintable: a {1000 * link.pad_thickness:.1f} mm round pad does not fit on a "
                    f"{L:.1f} mm phalanx (it needs twice its thickness)"
                )
            if link.pad == "round" and 2.0 * link.pad_thickness > link.width + 1e-9:
                out.append(
                    f"unprintable: a {1000 * link.pad_thickness:.1f} mm round pad is wider than the "
                    f"{1000 * link.width:.1f} mm finger"
                )
            if link.hook_depth > 0.0:
                lip = link.pad_thickness + link.hook_depth
                if link.hook_height < feature / 1000.0 or lip > 2.5 * link.hook_height + 1e-9:
                    out.append(
                        f"unprintable: a hook lip reaching {1000 * lip:.1f} mm in front of the finger must "
                        f"be at least {1000 * lip / 2.5:.1f} mm thick (hook_height_mm is "
                        f"{1000 * link.hook_height:.1f})"
                    )
                if link.hook_height > 0.5 * link.length:
                    out.append(
                        f"unprintable: the {1000 * link.hook_height:.1f} mm hook lip takes more than half "
                        f"of the {L:.1f} mm finger"
                    )
    if out:
        return out

    # self-intersection in the open pose: digit footprints on the palm
    for d in layout.digits:
        if d.inner_radius < 0.001:
            out.append(
                f"self_intersecting: digit {d.index + 1}'s inner side reaches {1000 * d.inner_radius:.1f} mm from "
                "the centre in the open pose, through the opposite digits' space (larger palm_radius_mm)"
            )
            return out
    polys = [footprint(d, inflate=0.0005) for d in layout.digits]
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            if _convex_overlap(polys[a], polys[b]):
                da, db = layout.digits[a], layout.digits[b]
                gap = math.degrees(abs(math.remainder(da.angle - db.angle, 2 * math.pi)))
                out.append(
                    f"self_intersecting: digits {a + 1} and {b + 1} overlap on the palm in the open pose "
                    f"({gap:.0f} deg apart at a {1000 * layout.palm_radius:.0f} mm radius); spread them, "
                    "narrow them or enlarge the palm"
                )
    if out:
        return out

    mass = layout.mass()
    if mass > MAX_EFFECTOR_MASS_KG:
        out.append(
            f"too_heavy: {mass:.2f} kg of {layout.material} (limit {MAX_EFFECTOR_MASS_KG:g} kg); use a lighter "
            "material, fewer or smaller digits or a thinner palm"
        )
    if layout.n_actuators > MAX_ACTUATORS:
        out.append(
            f"over_budget: {layout.n_actuators} servos ({layout.n_digits} digits x "
            f"{layout.n_joints // max(layout.n_digits, 1)} joints, one per joint); the budget is {MAX_ACTUATORS} "
            "(fewer joints, or tendon actuation)"
        )
    if layout.joint_torque > MAX_JOINT_TORQUE_NM + 1e-9:
        per = layout.grip_force / layout.n_digits
        out.append(
            f"over_budget: {per:.1f} N per digit at the {1000 * layout.lever:.0f} mm contact lever needs "
            f"{layout.joint_torque:.2f} N m at each joint; the limit is {MAX_JOINT_TORQUE_NM:g} N m (less force, "
            "shorter or more digits)"
        )
    if layout.grip_force > MAX_TOTAL_FORCE_N + 1e-9:
        out.append(
            f"over_budget: {layout.grip_force:.1f} N total closing force; the limit is {MAX_TOTAL_FORCE_N:g} N"
        )
    return out


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (vertices in order)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _tip_height_mm(layout: HandLayout, part: PartSpec, task: TaskV2) -> float:
    h = 1000.0 * float(part.size[2])
    return max(1000.0 * task.controller.min_tip_clearance, layout.grip_height_frac * h)


def _fit(layout: HandLayout, part: PartSpec, task: TaskV2) -> list[str]:
    out: list[str] = []
    for row in fit_numbers(layout, part):
        if row["inner_mm"] < row["needed_mm"] - 1e-9:
            out.append(
                f"too_narrow: open, digit {int(row['digit'])} (at {row['direction_deg']:.0f} deg on the part) has "
                f"its inner face {row['inner_mm']:.1f} mm from the centre; the part reaches {row['outline_mm']:.1f} mm "
                f"that way, so it needs {row['needed_mm']:.1f} mm ({ERROR_ENVELOPE_MM:g} mm placement error + "
                f"{MIN_OPEN_CLEARANCE_MM:g} mm clearance; palm_radius_mm sets it)"
            )
    h = 1000.0 * float(part.size[2])
    tip = _tip_height_mm(layout, part, task)
    drop = 1000.0 * layout.finger_drop
    need = h - tip + PALM_CLEARANCE_MM
    if drop < need - 1e-9:
        out.append(
            f"too_short: the digits reach {drop:.1f} mm below the palm; gripping {part.name} ({h:.1f} mm tall) with "
            f"the lowest point at {tip:.1f} mm needs {need:.1f} mm to keep the palm {PALM_CLEARANCE_MM:g} mm above "
            "the part (longer digits or a higher grip)"
        )
    if tip > h - MIN_OVERLAP_MM + 1e-9:
        out.append(
            f"above_part: at grip_height_frac {layout.grip_height_frac:g} the lowest point sits at {tip:.1f} mm, "
            f"less than {MIN_OVERLAP_MM:g} mm below the {h:.1f} mm top of {part.name}; it would close above it"
        )
    return out


# --------------------------------------------------------------------------- closing sweep


def _sweep_model(layout: HandLayout, part: PartSpec, task: TaskV2) -> tuple[mujoco.MjModel, gen.HandNames]:
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    world = spec.worldbody
    h = float(part.size[2])
    for i, (verts, faces) in enumerate(zip(part.pieces, part.faces, strict=True)):
        mesh = f"part{i:02d}"
        spec.add_mesh(
            name=mesh,
            uservert=np.asarray(verts).ravel().tolist(),
            userface=np.asarray(faces).ravel().tolist(),
        )
        world.add_geom(
            name=f"part_{i:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=[0.0, 0.0, h / 2.0]
        )
    tip = _tip_height_mm(layout, part, task) / 1000.0
    wrist_z = tip + layout.reach - MOUNT_Z
    wrist = world.add_body(
        name="wrist",
        pos=[0.0, 0.0, wrist_z],
        quat=[math.cos(layout.yaw / 2), 0.0, 0.0, math.sin(layout.yaw / 2)],
    )
    names = gen.add_hand(spec, wrist, layout, prefix="eff_", z0=MOUNT_Z, wrist=None)
    return spec.compile(), names


def _sweep(layout: HandLayout, part: PartSpec, task: TaskV2) -> list[str]:
    model, names = _sweep_model(layout, part, task)
    data = mujoco.MjData(model)
    part_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    digit_of_body = np.full(model.nbody, -1, dtype=np.int64)
    for di, bodies in enumerate(names.digit_bodies):
        for b in bodies:
            digit_of_body[model.body(b).id] = di
    qadr = [[model.jnt_qposadr[model.joint(j).id] for j in js] for js in names.joints]
    n = layout.n_digits
    frozen = [False] * n
    touching_bodies: set[int] = set()
    q_now = layout.sweep_pose(0.0)
    collision: tuple[int, int, float] | None = None
    for step in range(_SWEEP_STEPS + 1):
        c = step / _SWEEP_STEPS
        target = layout.sweep_pose(c)
        for di in range(n):
            if not frozen[di]:
                q_now[di] = target[di]
            for adr, q in zip(qadr[di], q_now[di], strict=True):
                data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        hit = [False] * n
        for k in range(data.ncon):
            con = data.contact[k]
            g0, g1 = int(con.geom[0]), int(con.geom[1])
            b0, b1 = int(model.geom_bodyid[g0]), int(model.geom_bodyid[g1])
            if (g0 in part_geoms) != (g1 in part_geoms):
                if con.dist <= _TOUCH:
                    body = b1 if g0 in part_geoms else b0
                    di = int(digit_of_body[body])
                    if di >= 0:
                        hit[di] = True
                        touching_bodies.add(body)
                continue
            d0, d1 = int(digit_of_body[b0]), int(digit_of_body[b1])
            between = d0 >= 0 and d1 >= 0 and d0 != d1 and not (frozen[d0] and frozen[d1])
            if between and con.dist < -_COLLIDE and collision is None:
                collision = (min(d0, d1), max(d0, d1), c)
        for di in range(n):
            if hit[di]:
                frozen[di] = True
        if collision is not None or all(frozen):
            break
    out: list[str] = []
    if collision is not None:
        a, b, c = collision
        out.append(
            f"digit_collision: closing over {part.name}, digits {a + 1} and {b + 1} run into each other at "
            f"{100 * c:.0f} % closure before both reach the part (spread them, narrow them or grip a wider section)"
        )
        return out
    reached = [di for di in range(n) if frozen[di]]
    if len(touching_bodies) < 2:
        missing = ", ".join(str(di + 1) for di in range(n) if not frozen[di]) or "none"
        out.append(
            f"cannot_reach: fully closed over {part.name}, only {len(touching_bodies)} effector body touches the "
            f"part (digits that never reach it: {missing}); a smaller palm radius, longer digits or more closing "
            "travel"
        )
        return out
    if len(reached) >= 2:
        spread = max(
            math.degrees(abs(math.remainder(layout.digits[a].angle - layout.digits[b].angle, 2 * math.pi)))
            for i, a in enumerate(reached)
            for b in reached[i + 1 :]
        )
        if spread < 90.0:
            out.append(
                f"no_opposition: the digits that reach {part.name} ({', '.join(str(d + 1) for d in reached)}) are "
                f"at most {spread:.0f} deg apart; at least two must push from directions 90 deg or more apart"
            )
    return out


# --------------------------------------------------------------------------- entry point


def _foreign(recipe: MorphologyRecipe, part: PartSpec, task: TaskV2) -> list[str]:
    from rtt.effectors.registry import get_family
    from rtt.effectors.scene import build_effector_scene

    impl = get_family(recipe.family)
    try:
        spec = impl.spec_from_recipe(recipe)
        build_effector_scene(impl, spec, part, task)
    except InvalidEffector as exc:
        return [f"not_buildable: {exc}"]
    except ValueError as exc:
        return [f"out_of_bounds: {exc}"]
    return []


def validate_structure(recipe: MorphologyRecipe) -> list[str]:
    """The part-free stages only (schema and structure): ``[]`` for a recipe that is a valid
    effector in itself, whether or not it fits a given part."""
    problems = _schema(recipe)
    if problems or recipe.family is not EffectorFamily.DESIGNED:
        return problems
    return _structure(layout_of(recipe))


def validate_recipe(recipe: MorphologyRecipe, part: PartSpec, task: TaskV2 | None = None) -> list[str]:
    """Every problem of the first failing stage (module docstring); ``[]`` when buildable."""
    from rtt.sim.task import default_task_v2, task_digest

    task = task or default_task_v2()
    key = (recipe.recipe_id, part.geometry_sha256, task_digest(task))
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return list(hit)
    problems = _schema(recipe)
    if not problems:
        if recipe.family is not EffectorFamily.DESIGNED:
            problems = _foreign(recipe, part, task)
        else:
            layout = layout_of(recipe)
            problems = _structure(layout) or _fit(layout, part, task) or _sweep(layout, part, task)
    _CACHE[key] = tuple(problems)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return list(problems)
