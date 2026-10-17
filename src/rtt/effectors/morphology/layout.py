"""From a recipe to a concrete effector layout: pure geometry, no MuJoCo.

:func:`layout_of` turns a :class:`~rtt.effectors.types.MorphologyRecipe` of one of our
generators (:data:`~rtt.effectors.morphology.spaces.DESIGNED_KINDS`) into a
:class:`HandLayout`: the palm, every digit's links, pads and hook, its joints with their
ranges and springs, the actuators with their declared limits, and the open and closed control
vectors of the grasp synergy. The generator (:mod:`rtt.effectors.morphology.generator`)
writes it into an ``MjSpec``; the validators read it to check bounds, budgets and fit before
anything is compiled.

Frames. The hand frame is the wrist frame turned by the recipe's ``yaw_deg`` about z; the mount
plane (the palm's top face) is at hand z = ``mount_z``. Digit ``i`` sits at angle
``digits[i].angle`` around z (0 = +x, digit 1 of an even layout at +y). Each digit has its own
frame: origin at its root on the palm's underside, x pointing radially *outward*, z up; its
links hang along -z, its inner (contact) side is -x, and a hinge about +y with a positive angle
swings the digit in toward the centre. A slide digit moves along -x.

Units are SI (m, kg, rad, N, N m) inside a layout; recipes are in mm and deg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Literal

import numpy as np

from rtt.effectors.morphology.spaces import MATERIAL_DENSITY
from rtt.effectors.types import EffectorFamily, MorphologyRecipe

__all__ = [
    "ACTUATION_LIMITS",
    "MAX_ACTUATORS",
    "MAX_EFFECTOR_MASS_KG",
    "MAX_JOINT_TORQUE_NM",
    "MAX_TOTAL_FORCE_N",
    "PAD_DENSITY",
    "Digit",
    "HandLayout",
    "Link",
    "digit_angles",
    "footprint",
    "layout_of",
]

#: Declared actuation budget of every designed effector (docs/EFFECTORS.md, section 2.4).
MAX_TOTAL_FORCE_N: Final = 50.0
MAX_JOINT_TORQUE_NM: Final = 1.5
#: Servos per effector (a fully actuated 5 x 3 hand needs 15; the budget is 12).
MAX_ACTUATORS: Final = 12
#: Track B3's mass limit for a designed effector (the scene refuses anything above 5 kg).
MAX_EFFECTOR_MASS_KG: Final = 1.5
ACTUATION_LIMITS: Final = {
    "total_force_n": MAX_TOTAL_FORCE_N,
    "joint_torque_nm": MAX_JOINT_TORQUE_NM,
    "actuators": MAX_ACTUATORS,
}
#: Contact pads (elastomer); a pad never changes friction (the part's friction governs).
PAD_DENSITY: Final = 1100.0

#: Nominal contact lever: where along a finger its closing force is counted, as a fraction of
#: its length from the knuckle (the force budget's reference point).
_LEVER_FRAC: Final[dict[str, float]] = {
    "fingered_hand": 0.75,
    "hook_gripper": 0.85,
    "segmented_tentacle": 0.6,
}
_PROX_RANGE: Final = (math.radians(-3.0), math.radians(80.0))
_DIST_RANGE: Final = (math.radians(-80.0), math.radians(90.0))
_PALM_MARGIN: Final = 0.001  # m the palm plate overhangs the digit roots
_SLIDE_MIN_RADIUS: Final = 0.002  # m: a slide digit's innermost point stops this far from the centre


@dataclass(frozen=True)
class Link:
    """One rigid link of a digit, in the digit frame (it hangs from its joint along -z).

    ``shape`` is ``"box"`` (``thickness`` radial x ``width`` tangential x ``length``) or
    ``"capsule"`` (``radius``, axis along the link). ``pad`` is on the inner (-x) face: a flat
    block or a round bump of ``pad_thickness``; ``hook_depth``/``hook_height`` describe an
    inward lip at the bottom of the link (hook gripper).
    """

    length: float
    width: float
    thickness: float
    shape: Literal["box", "capsule"] = "box"
    radius: float = 0.0
    pad: Literal["none", "flat", "round"] = "none"
    pad_thickness: float = 0.0
    hook_depth: float = 0.0
    hook_height: float = 0.0

    @property
    def inner_offset(self) -> float:
        """How far the link's innermost point sits inside its axis (m, toward -x)."""
        if self.shape == "capsule":
            return self.radius
        return self.thickness / 2.0 + self.pad_thickness + self.hook_depth

    @property
    def outer_offset(self) -> float:
        return self.radius if self.shape == "capsule" else self.thickness / 2.0

    def volumes(self) -> tuple[float, float]:
        """(structure m^3, pad m^3) of this link (geometry exactly as the generator builds it)."""
        if self.shape == "capsule":
            r, half = self.radius, self.length / 2.0
            return math.pi * r * r * 2.0 * half + 4.0 / 3.0 * math.pi * r**3, 0.0
        body = self.thickness * self.width * self.length
        if self.hook_depth > 0.0:
            body += (self.pad_thickness + self.hook_depth) * self.width * self.hook_height
        pad = 0.0
        if self.pad == "flat":
            pad = self.pad_thickness * (0.9 * self.width) * self.pad_length
        elif self.pad == "round":
            r = self.pad_thickness
            half = self.width / 2.0 - r
            half = half if half > 1e-4 else 0.0  # the generator builds a sphere then
            pad = math.pi * r * r * 2.0 * half + 4.0 / 3.0 * math.pi * r**3
        return body, pad

    @property
    def pad_length(self) -> float:
        """Vertical extent of a flat pad."""
        if self.hook_depth > 0.0:
            return 0.6 * (self.length - self.hook_height)
        return 0.8 * self.length

    @property
    def pad_center_z(self) -> float:
        """z of the pad's centre in the link frame (the link hangs from z = 0 to -length)."""
        if self.pad == "round":
            return -0.6 * self.length
        if self.hook_depth > 0.0:
            return -self.length + self.hook_height + self.pad_length / 2.0
        return -self.length / 2.0


@dataclass(frozen=True)
class Digit:
    """One digit: its place on the palm, its links and its joints (one per link).

    ``torque``: the force limit of each joint's servo or share of the tendon, in N m for a
    hinge (``digit force x contact lever`` measured from that joint, so no link pushes harder
    than the digit's share of the closing force) and in N for a slide.
    """

    index: int
    angle: float  # rad, around z in the hand frame
    radius: float  # m, the root's distance from the hand's axis
    joint: Literal["hinge", "slide"]
    links: tuple[Link, ...]
    lower: tuple[float, ...]  # joint range (rad, or m for a slide)
    upper: tuple[float, ...]
    stiffness: tuple[float, ...]  # joint springs (N m/rad or N/m), springref 0
    torque: tuple[float, ...]

    @property
    def length(self) -> float:
        return sum(link.length for link in self.links)

    @property
    def inner_radius(self) -> float:
        """Radius of the digit's innermost point in the open pose (m from the hand's axis)."""
        return self.radius - max(link.inner_offset for link in self.links)

    def remaining(self, j: int) -> float:
        """Length (m) from joint ``j`` to the digit's tip."""
        return sum(link.length for link in self.links[j:])


@dataclass(frozen=True)
class HandLayout:
    """A concrete designed effector (module docstring)."""

    recipe_id: str
    kind: str
    digits: tuple[Digit, ...]
    palm_radius: float  # m, root circle
    palm_plate_radius: float  # m, the plate's radius
    palm_thickness: float  # m
    density: float  # kg/m^3, structure
    actuation: Literal["per_joint", "tendon"]
    grasp: Literal["power", "pinch"]
    grip_force: float  # N, total, as declared by the recipe
    lever: float  # m, nominal contact lever of a digit from its root joint
    grip_height_frac: float
    yaw: float  # rad
    tendon_coefs: tuple[tuple[float, ...], ...]  # per digit, per joint: moment arm (m), 0 = passive
    material: str

    @property
    def digit_force(self) -> float:
        """N each digit closes with at its nominal contact point (the total shared evenly)."""
        return self.grip_force / len(self.digits)

    @property
    def joint_torque(self) -> float:
        """Largest hinge torque (N m) any joint gets (0 without hinges)."""
        return max((max(d.torque) for d in self.digits if d.joint == "hinge"), default=0.0)

    @property
    def slide_force(self) -> float:
        """N per slide digit (0 without slides)."""
        return max((d.torque[0] for d in self.digits if d.joint == "slide"), default=0.0)

    # ------------------------------------------------------------------ derived

    @property
    def n_digits(self) -> int:
        return len(self.digits)

    @property
    def n_joints(self) -> int:
        return sum(len(d.links) for d in self.digits)

    @property
    def n_actuators(self) -> int:
        return self.n_digits if self.actuation == "tendon" else self.n_joints

    @property
    def reach(self) -> float:
        """m from the mount plane down to the lowest point in the open pose (digits straight)."""
        return self.palm_thickness + max(self._digit_depth(d) for d in self.digits)

    def _digit_depth(self, d: Digit) -> float:
        tip = d.links[-1]
        return d.length + (tip.radius if tip.shape == "capsule" else 0.0)

    @property
    def finger_drop(self) -> float:
        """m from the palm's underside down to the lowest point (the reach below the palm)."""
        return self.reach - self.palm_thickness

    @property
    def max_opening(self) -> float:
        """Diameter of the circle the digits' innermost points clear in the open pose."""
        return 2.0 * min(d.inner_radius for d in self.digits)

    def mass(self) -> float:
        """Analytic mass (kg) of every geom the generator builds."""
        plate = math.pi * self.palm_plate_radius**2 * self.palm_thickness
        total = plate * self.density
        for d in self.digits:
            for link in d.links:
                body, pad = link.volumes()
                total += body * self.density + pad * PAD_DENSITY
        return total

    def actuation_card(self) -> dict[str, float]:
        """The declared limits printed on every card (EffectorBuild.actuation)."""
        out: dict[str, float] = {
            "grip_force_n_total": round(self.grip_force, 3),
            "actuators": float(self.n_actuators),
            "contact_lever_mm": round(1000.0 * self.lever, 2),
        }
        if any(d.joint == "hinge" for d in self.digits):
            out["joint_torque_nm_max"] = round(self.joint_torque, 4)
        if any(d.joint == "slide" for d in self.digits):
            out["slide_force_n_per_finger"] = round(self.slide_force, 3)
        return out

    # ------------------------------------------------------------------ synergy

    def actuator_joints(self) -> list[tuple[int, tuple[int, ...]]]:
        """Per actuator in control order: (digit index, joint indices it drives)."""
        out: list[tuple[int, tuple[int, ...]]] = []
        for d in self.digits:
            if self.actuation == "tendon":
                out.append((d.index, tuple(range(len(d.links)))))
            else:
                out.extend((d.index, (j,)) for j in range(len(d.links)))
        return out

    def ctrl_open(self) -> tuple[float, ...]:
        return tuple(0.0 for _ in self.actuator_joints())

    def ctrl_closed(self) -> tuple[float, ...]:
        """Closed targets: every driven joint to its closing limit (the force limit squeezes).

        A pinch keeps its last phalanx upright with a closed-loop target (the primitive's
        ``upright`` rule), so its closed target here is only a placeholder of 0.
        """
        out = []
        for di, joints in self.actuator_joints():
            d = self.digits[di]
            if self.actuation == "tendon":
                coefs = self.tendon_coefs[di]
                out.append(sum(c * d.upper[j] for j, c in enumerate(coefs)))
            else:
                (j,) = joints
                if self.grasp == "pinch" and d.joint == "hinge" and j > 0:
                    out.append(0.0)
                else:
                    out.append(d.upper[j])
        return tuple(out)

    def upright_rules(self) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
        """Pinch with per-joint servos: (actuator index, (digit, joint) pairs whose angles it cancels).

        The last phalanx of each multi-phalanx finger is driven to minus the sum of the joint
        angles above it, so it hangs upright and its pad meets the part flat.
        """
        if self.grasp != "pinch" or self.actuation != "per_joint":
            return ()
        rules = []
        for a, (di, joints) in enumerate(self.actuator_joints()):
            d = self.digits[di]
            (j,) = joints
            if d.joint == "hinge" and j > 0 and j == len(d.links) - 1:
                rules.append((a, tuple((di, k) for k in range(j))))
        return tuple(rules)

    def sweep_pose(self, closure: float) -> list[tuple[float, ...]]:
        """Kinematic closing path used by the validators: joint positions per digit at ``closure``."""
        c = min(max(float(closure), 0.0), 1.0)
        out = []
        for d in self.digits:
            n = len(d.links)
            if d.joint == "slide":
                out.append((c * d.upper[0],))
                continue
            if self.actuation == "tendon":
                coefs = self.tendon_coefs[d.index]
                q = [c * d.upper[j] if coefs[j] > 0 else 0.0 for j in range(n)]
            elif self.grasp == "pinch":
                q = [c * d.upper[0]] + [0.0] * (n - 1)
                if n > 1:
                    q[-1] = max(d.lower[-1], -sum(q[:-1]))
            else:
                q = [c * d.upper[j] for j in range(n)]
            out.append(tuple(q))
        return out


# --------------------------------------------------------------------------- helpers


def digit_angles(n: int, layout: str, spread_deg: float) -> tuple[float, ...]:
    """Angles (rad) of ``n`` digits: ``even`` from +y, or ``opposed`` (a thumb at -y, the rest
    spread over ``spread_deg`` centred on +y). One digit sits at +y."""
    if n <= 1:
        return (math.pi / 2.0,)
    if layout == "even":
        return tuple(math.pi / 2.0 + 2.0 * math.pi * i / n for i in range(n))
    others = n - 1
    spread = math.radians(spread_deg)
    if others == 1:
        rest = [math.pi / 2.0]
    else:
        rest = [math.pi / 2.0 + spread * (j / (others - 1) - 0.5) for j in range(others)]
    return (-math.pi / 2.0, *rest)


def footprint(d: Digit, inflate: float = 0.0) -> np.ndarray:
    """The digit's footprint on the palm's underside in the open pose (4 corners, hand frame xy).

    Box digits: the rectangle from the innermost point (pad, hook) to the back face, across the
    finger's width. Capsule digits: the root circle's bounding square turned with the digit.
    """
    root = d.links[0]
    if root.shape == "capsule":
        r_in, r_out, half_w = d.radius - root.radius, d.radius + root.radius, root.radius
    else:
        r_in, r_out, half_w = d.inner_radius, d.radius + root.outer_offset, root.width / 2.0
    r_in, r_out, half_w = r_in - inflate, r_out + inflate, half_w + inflate
    local = np.array([[r_in, -half_w], [r_out, -half_w], [r_out, half_w], [r_in, half_w]])
    c, s = math.cos(d.angle), math.sin(d.angle)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T


def _num(params: dict, name: str) -> float:
    return float(params[name])


def _mm(params: dict, name: str) -> float:
    return float(params[name]) / 1000.0


@lru_cache(maxsize=4096)
def _layout_cached(recipe_id: str, kind: str, items: tuple[tuple[str, object], ...]) -> HandLayout:
    p = dict(items)
    builder = {
        "fingered_hand": _fingered_hand,
        "hook_gripper": _hook_gripper,
        "segmented_tentacle": _segmented_tentacle,
    }[kind]
    return builder(recipe_id, p)


def layout_of(recipe: MorphologyRecipe) -> HandLayout:
    """The layout of a designed recipe (cached by recipe id).

    The recipe must already satisfy its parameter space (:meth:`MorphologyRecipe.problems`);
    the validators check that first.
    """
    if recipe.family is not EffectorFamily.DESIGNED:
        raise ValueError(f"layout_of builds designed recipes only, not {recipe.family.value!r}")
    return _layout_cached(recipe.recipe_id, recipe.kind, tuple(sorted(recipe.params.items())))


def _common(p: dict) -> dict:
    return {
        "palm_radius": _mm(p, "palm_radius_mm"),
        "palm_thickness": _mm(p, "palm_thickness_mm"),
        "density": MATERIAL_DENSITY[str(p["material"])],
        "material": str(p["material"]),
        "grip_force": _num(p, "grip_force_n"),
        "grip_height_frac": _num(p, "grip_height_frac"),
        "yaw": math.radians(_num(p, "yaw_deg")),
    }


def _plate_radius(radius: float, digits: tuple[Digit, ...]) -> float:
    return radius + max(d.links[0].outer_offset for d in digits) + _PALM_MARGIN


def _remaining(links: tuple[Link, ...], j: int) -> float:
    return sum(link.length for link in links[j:])


def _hand(kind: str, recipe_id: str, c: dict, digits: tuple[Digit, ...], **kw: object) -> HandLayout:
    return HandLayout(
        recipe_id=recipe_id,
        kind=kind,
        digits=digits,
        palm_radius=c["palm_radius"],
        palm_plate_radius=_plate_radius(c["palm_radius"], digits),
        palm_thickness=c["palm_thickness"],
        density=c["density"],
        grip_force=c["grip_force"],
        grip_height_frac=c["grip_height_frac"],
        yaw=c["yaw"],
        material=c["material"],
        **kw,  # type: ignore[arg-type]
    )


def _fingered_hand(recipe_id: str, p: dict) -> HandLayout:
    c = _common(p)
    n, n_ph = int(p["n_fingers"]), int(p["n_phalanges"])
    length = _mm(p, "finger_length_mm")
    if n_ph == 1:
        lengths = [length]
    else:
        first = _num(p, "proximal_frac") * length
        lengths = [first] + [(length - first) / (n_ph - 1)] * (n_ph - 1)
    pad = str(p["pad"])
    pad_t = 0.0 if pad == "none" else _mm(p, "pad_thickness_mm")
    width, thick = _mm(p, "finger_width_mm"), _mm(p, "finger_thickness_mm")
    links = tuple(Link(L, width, thick, "box", 0.0, pad, pad_t) for L in lengths)  # type: ignore[arg-type]
    actuation, grasp = str(p["actuation"]), str(p["grasp"])
    lam = _LEVER_FRAC["fingered_hand"]
    force = c["grip_force"] / n
    arms = tuple(lam * _remaining(links, j) for j in range(n_ph))  # contact lever from each joint
    torque = tuple(force * a for a in arms)
    lower = (_PROX_RANGE[0],) + (_DIST_RANGE[0],) * (n_ph - 1)
    upper = (_PROX_RANGE[1],) + (_DIST_RANGE[1],) * (n_ph - 1)
    if actuation == "tendon":
        # springs keep the outer phalanges straight until the first one is stopped (30 % of their torque at 90 deg)
        stiffness = (0.0,) + tuple(0.3 * t / (math.pi / 2.0) for t in torque[1:])
        driven = (True,) + ((grasp == "power"),) * (n_ph - 1)
        coefs = tuple(a if on else 0.0 for a, on in zip(arms, driven, strict=True))
    else:
        stiffness = (0.0,) * n_ph
        coefs = ()
    angles = digit_angles(n, str(p["layout"]), _num(p, "spread_deg"))
    digits = tuple(
        Digit(i, a, c["palm_radius"], "hinge", links, lower, upper, stiffness, torque)
        for i, a in enumerate(angles)
    )
    return _hand(
        "fingered_hand",
        recipe_id,
        c,
        digits,
        actuation=actuation,
        grasp=grasp,
        lever=arms[0],
        tendon_coefs=tuple(coefs for _ in digits) if coefs else (),
    )


def _hook_gripper(recipe_id: str, p: dict) -> HandLayout:
    c = _common(p)
    n = int(p["n_fingers"])
    length = _mm(p, "finger_length_mm")
    width, thick = _mm(p, "finger_width_mm"), _mm(p, "finger_thickness_mm")
    hook_d = _mm(p, "hook_depth_mm")
    hook_h = _mm(p, "hook_height_mm") if hook_d > 0.0 else 0.0
    link = Link(length, width, thick, "box", 0.0, "flat", _mm(p, "pad_thickness_mm"), hook_d, hook_h)
    motion = str(p["motion"])
    lever = _LEVER_FRAC["hook_gripper"] * length
    force = c["grip_force"] / n
    radius = c["palm_radius"]
    if motion == "slide":
        stroke = max(radius - link.inner_offset - _SLIDE_MIN_RADIUS, 0.0)
        joint, lower, upper, torque = "slide", (0.0,), (stroke,), (force,)
    else:
        joint, lower, upper, torque = "hinge", (_PROX_RANGE[0],), (_PROX_RANGE[1],), (force * lever,)
    angles = digit_angles(n, str(p["layout"]), _num(p, "spread_deg"))
    digits = tuple(
        Digit(i, a, radius, joint, (link,), lower, upper, (0.0,), torque)  # type: ignore[arg-type]
        for i, a in enumerate(angles)
    )
    return _hand(
        "hook_gripper",
        recipe_id,
        c,
        digits,
        actuation="per_joint",
        grasp="pinch",
        lever=lever,
        tendon_coefs=(),
    )


def _segmented_tentacle(recipe_id: str, p: dict) -> HandLayout:
    c = _common(p)
    n, n_seg = int(p["n_digits"]), int(p["n_segments"])
    length = _mm(p, "digit_length_mm")
    r0 = _mm(p, "root_radius_mm")
    taper = _num(p, "taper")
    seg = length / n_seg
    radii = [r0 * (1.0 - (1.0 - taper) * j / (n_seg - 1)) for j in range(n_seg)]
    links = tuple(Link(seg, 2.0 * r, 2.0 * r, "capsule", r) for r in radii)
    lam = _LEVER_FRAC["segmented_tentacle"]
    force = c["grip_force"] / n
    arms = tuple(lam * _remaining(links, j) for j in range(n_seg))
    torque = tuple(force * a for a in arms)
    limit = min(math.radians(75.0), math.radians(330.0) / n_seg)
    bias = _num(p, "curl_bias")
    # graded springs: the segments nearest curl_bias curl first (softest), the far end last
    stiffness = tuple(t / limit * (0.4 + 1.2 * abs(j / (n_seg - 1) - bias)) for j, t in enumerate(torque))
    lower, upper = (math.radians(-3.0),) * n_seg, (limit,) * n_seg
    angles = digit_angles(n, str(p["layout"]), _num(p, "spread_deg"))
    digits = tuple(
        Digit(i, a, c["palm_radius"], "hinge", links, lower, upper, stiffness, torque)
        for i, a in enumerate(angles)
    )
    return _hand(
        "segmented_tentacle",
        recipe_id,
        c,
        digits,
        actuation="tendon",
        grasp="power",
        lever=arms[0],
        tendon_coefs=tuple(arms for _ in digits),
    )
