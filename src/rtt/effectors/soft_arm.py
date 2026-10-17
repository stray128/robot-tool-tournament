"""Soft arm family (product mode, approximation): an octopus-inspired tentacle curled by a cable.

**Fidelity: approximation.** Read :data:`FIDELITY_CAVEAT` before trusting a soft-arm result.

The arm hangs beside the part from the wrist flange and wraps it in a horizontal plane:

* ``eff_arm_base``: a housing at the mount plane and a rigid vertical **post** that carries the
  arm's root ``stem_offset_mm`` to the side of the wrist axis (+y in the wrist frame), so the
  straight arm clears the part while the wrist lowers it beside the part;
* ``eff_seg_00`` ... ``eff_seg_<n-1>``: ``n_segments`` (at least 12) tapered capsules in a
  chain along +x from the bottom of the post, each on a **hinge about the vertical axis** with a
  torsion spring (rest angle 0: straight) and a damper. The arm bends in one plane, the
  horizontal plane of the grasp; it is rigid out of that plane;
* ``eff_cable``: a spatial tendon routed through sites on the part side (-y) of every segment
  (``cable_offset_frac`` of the local radius from the axis), pulled by ``eff_cable_motor``, a
  motor whose control is the cable tension in N, at most ``max_tension_n`` (<= 50 N, the
  parallel jaw's grip force).

Pulling the cable curls the arm toward -y, round the far side of the part, which it pulls
against the post: the contacts wrap the part. Releasing the cable lets the springs straighten
the arm (:func:`straighten_test` measures how fast). The primitive ``wrap`` puts the arm's
plane at ``grasp_height_frac`` of the part's height (one value per effector, never per part)
and ramps the tension on the shared close ramp; the criterion ``wrap`` is engaged when at least
:data:`MIN_WRAP_SEGMENTS` segments touch the part and the contacts (post included) wrap at least
:data:`MIN_WRAP_DEG` degrees around the part's vertical axis through its centre of mass.

Fairness (``docs/EFFECTORS.md`` sections 2.3 and 2.4): springs, dampers and rotor inertia
(``armature``) on the arm's own joints only; no priority, adhesion, margin or surface velocity
on its geoms; no change to the solver or defaults; the part's friction governs. Segments do
not collide with each other (stated in the caveat); they collide with the part, the table and
the wrist.

Designable by track B3: :meth:`SoftArmFamily.param_space` (``"tentacle"``) and
:meth:`SoftArmFamily.spec_from_recipe`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.primitives import part_height
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import (
    NAME_PREFIX,
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    Engagement,
    InvalidEffector,
    MorphologyRecipe,
    ParamSpec,
    PrimitivePlan,
)
from rtt.sim.controller import Phase

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2

__all__ = [
    "ARM_KIND",
    "FIDELITY_CAVEAT",
    "MAX_TENSION_N",
    "MIN_SEGMENTS",
    "MIN_WRAP_DEG",
    "MIN_WRAP_SEGMENTS",
    "PARAM_SPACE",
    "ArmGeometry",
    "SoftArmFamily",
    "StraightenResult",
    "WrapCriterion",
    "WrapPrimitive",
    "arm_spec",
    "straighten_test",
    "wrap_readout",
]

ARM_KIND: Final = "tentacle"
MAX_TENSION_N: Final = 50.0  # the parallel jaw's grip force (docs/EFFECTORS.md section 2.4)
MIN_SEGMENTS: Final = 12
MIN_WRAP_SEGMENTS: Final = 3
MIN_WRAP_DEG: Final = 180.0
MIN_CONTACT_FORCE: Final = 0.05  # N; a contact counts toward the wrap above this normal force
POST_LENGTH: Final = 0.07  # m, mount plane to the arm's plane
SEGMENT_DENSITY: Final = 1100.0  # kg/m^3, a silicone-like arm
JOINT_RANGE_DEG: Final = (-45.0, 10.0)  # per joint: curl toward the part, a little the other way
DAMPING_TIME: Final = 0.08  # s, at least this damping / stiffness on every joint
#: Each joint is damped at least this fraction of critical for the straight arm beyond it
#: swinging about it, so the released arm returns without ringing (a viscous, octopus-like arm).
DAMPING_RATIO: Final = 1.0

FIDELITY_CAVEAT: Final = (
    "Soft arm (approximation, octopus-inspired): a chain of rigid tapered capsule segments on "
    "spring-damper hinges, curled by a cable (a spatial tendon through sites) with a tension "
    "limit of at most 50 N. It bends in one plane (horizontal) and is rigid out of it; its "
    "segments do not collide with each other; a rigid post carries its root beside the part and "
    "the arm pulls the part against it, so the post takes part of the grip. "
    "Not modelled: continuum mechanics, material non-linearity, suckers, hydrostatic "
    "stiffening. Segment count sets how smooth the wrap is."
)

PARAM_SPACE: Final[tuple[ParamSpec, ...]] = (
    ParamSpec(
        "n_segments",
        "int",
        "Number of spring-jointed segments in the arm (more segments wrap more smoothly).",
        "count",
        MIN_SEGMENTS,
        24,
        default=16,
    ),
    ParamSpec(
        "length_mm", "float", "Length of the arm from its root to its tip.", "mm", 80.0, 240.0, default=160.0
    ),
    ParamSpec(
        "root_radius_mm", "float", "Radius of the arm at its root segment.", "mm", 3.0, 8.0, default=6.0
    ),
    ParamSpec(
        "taper",
        "float",
        "Tip radius as a fraction of the root radius (1 = no taper).",
        "",
        0.3,
        1.0,
        default=0.5,
    ),
    ParamSpec(
        "stiffness_nm_per_rad",
        "float",
        "Spring rate of the root joint; joints toward the tip scale with their radius.",
        "N m/rad",
        0.05,
        1.0,
        default=0.3,
    ),
    ParamSpec(
        "cable_offset_frac",
        "float",
        "Distance of the cable from the segment axis, as a fraction of the local radius.",
        "",
        0.3,
        0.9,
        default=0.7,
    ),
    ParamSpec(
        "max_tension_n",
        "float",
        "Cable tension at full pull (the same limit as the parallel jaw's grip force).",
        "N",
        10.0,
        MAX_TENSION_N,
        default=MAX_TENSION_N,
    ),
    ParamSpec(
        "stem_offset_mm",
        "float",
        "Distance from the wrist axis to the arm's root post (the arm hangs beside the part).",
        "mm",
        20.0,
        60.0,
        default=32.0,
    ),
    ParamSpec(
        "grasp_height_frac",
        "float",
        "Height of the arm's plane as a fraction of the part's height.",
        "",
        0.2,
        0.8,
        default=0.45,
    ),
)
_DEFAULTS: Final = {p.name: p.default for p in PARAM_SPACE}


@dataclass(frozen=True)
class ArmGeometry:
    """Every dimension of one arm (m, N, N m/rad), derived from its params (:data:`PARAM_SPACE`)."""

    n: int
    length: float
    root_radius: float
    taper: float
    stiffness: float
    cable_offset_frac: float
    max_tension: float
    stem_offset: float
    grasp_height_frac: float

    @staticmethod
    def from_params(params: Any) -> ArmGeometry:
        p = {**_DEFAULTS, **dict(params)}
        problems = MorphologyRecipe(EffectorFamily.SOFT_ARM, ARM_KIND, p).problems(PARAM_SPACE)
        if problems:
            raise ValueError("soft arm params: " + "; ".join(problems))
        geo = ArmGeometry(
            n=int(p["n_segments"]),
            length=float(p["length_mm"]) / 1000.0,
            root_radius=float(p["root_radius_mm"]) / 1000.0,
            taper=float(p["taper"]),
            stiffness=float(p["stiffness_nm_per_rad"]),
            cable_offset_frac=float(p["cable_offset_frac"]),
            max_tension=float(p["max_tension_n"]),
            stem_offset=float(p["stem_offset_mm"]) / 1000.0,
            grasp_height_frac=float(p["grasp_height_frac"]),
        )
        if geo.post_radius >= geo.stem_offset - 0.005:
            raise ValueError("soft arm params: the post is too thick for its offset from the wrist axis")
        return geo

    @property
    def segment_length(self) -> float:
        return self.length / self.n

    def radius(self, i: int) -> float:
        """Capsule radius of segment ``i`` (linear taper from the root to the tip)."""
        return self.root_radius * (1.0 + (self.taper - 1.0) * i / max(1, self.n - 1))

    def joint_stiffness(self, i: int) -> float:
        return self.stiffness * self.radius(i) / self.root_radius

    def segment_mass(self, i: int) -> float:
        r, seg = self.radius(i), self.segment_length
        return SEGMENT_DENSITY * (math.pi * r * r * seg + 4.0 / 3.0 * math.pi * r**3)

    def distal_inertia(self, i: int) -> float:
        """kg m^2: the straight arm from segment ``i`` to the tip, about joint ``i``."""
        seg = self.segment_length
        return sum(
            self.segment_mass(j) * (((j - i) + 0.5) ** 2 + 1.0 / 12.0) * seg * seg for j in range(i, self.n)
        )

    def joint_damping(self, i: int) -> float:
        k = self.joint_stiffness(i)
        return max(DAMPING_TIME * k, 2.0 * DAMPING_RATIO * math.sqrt(k * self.distal_inertia(i)))

    def cable_offset(self, i: int) -> float:
        return self.cable_offset_frac * self.radius(i)

    @property
    def post_radius(self) -> float:
        return self.root_radius + 0.001

    @property
    def low(self) -> float:
        """m from the arm's plane down to its lowest point (the post's end or the root segment)."""
        return max(self.post_radius, self.root_radius)

    @property
    def reach(self) -> float:
        return POST_LENGTH + self.low

    def free_curl_deg(self) -> float:
        """Total curl of the free arm at full tension, ignoring joint limits (deg)."""
        return math.degrees(
            sum(self.max_tension * self.cable_offset(i) / self.joint_stiffness(i) for i in range(self.n))
        )


# --------------------------------------------------------------------------- the wrap readout


def wrap_readout(model: mujoco.MjModel, data: mujoco.MjData, scene: Any, segment_bodies: np.ndarray) -> dict:
    """Effector-part contacts above :data:`MIN_CONTACT_FORCE`: segments touching and the wrap angle.

    The wrap angle is 360 deg minus the largest angular gap between the contact points, seen
    from above around the vertical axis through the part's centre of mass (every effector
    body counts, the post included). Returns ``segments``, ``bodies``, ``contacts``,
    ``wrap_deg`` and ``normal_force_n`` (the sum).
    """
    ids = scene.ids
    out = {"segments": 0.0, "bodies": 0.0, "contacts": 0.0, "wrap_deg": 0.0, "normal_force_n": 0.0}
    if not data.ncon:
        return out
    eff = np.zeros(model.ngeom, dtype=bool)
    eff[ids.effector_geoms] = True
    obj = np.zeros(model.ngeom, dtype=bool)
    obj[ids.object_geoms] = True
    con = data.contact
    g = con.geom
    mask = (con.efc_address >= 0) & ((eff[g[:, 0]] & obj[g[:, 1]]) | (obj[g[:, 0]] & eff[g[:, 1]]))
    if not mask.any():
        return out
    centre = data.xipos[ids.object_body]
    wrench = np.zeros(6)
    angles: list[float] = []
    bodies: set[int] = set()
    total = 0.0
    for i in np.flatnonzero(mask):
        mujoco.mj_contactForce(model, data, int(i), wrench)
        if wrench[0] <= MIN_CONTACT_FORCE:
            continue
        g0, g1 = int(g[i, 0]), int(g[i, 1])
        body = int(model.geom_bodyid[g0 if eff[g0] else g1])
        bodies.add(body)
        p = con.pos[i]
        angles.append(math.atan2(p[1] - centre[1], p[0] - centre[0]))
        total += float(wrench[0])
    if not angles:
        return out
    a = np.sort(np.mod(np.asarray(angles), 2.0 * math.pi))
    gaps = np.diff(np.concatenate([a, a[:1] + 2.0 * math.pi]))
    out.update(
        segments=float(len(bodies & {int(b) for b in segment_bodies})),
        bodies=float(len(bodies)),
        contacts=float(len(angles)),
        wrap_deg=float(math.degrees(2.0 * math.pi - gaps.max())),
        normal_force_n=total,
    )
    return out


# --------------------------------------------------------------------------- primitive and criterion


@dataclass(frozen=True)
class WrapPrimitive:
    """``wrap``: lower the arm beside the part, then pull the cable on the shared close ramp.

    The arm's plane goes to ``grasp_height_frac`` of the part's height (its lowest point at
    least the task's minimum tip clearance above the table); the cable tension is
    ``closure x max_tension`` from the start of CLOSE through the lift and the hold.
    """

    geometry: ArmGeometry
    name: str = "wrap"

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        g = self.geometry
        tip = max(task.controller.min_tip_clearance, g.grasp_height_frac * part_height(part) - g.low)
        return PrimitivePlan(tip_height=tip, note="arm plane at a fixed fraction of the part's height")

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        if phase < Phase.CLOSE:
            return np.zeros(1)
        return np.array([closure * self.geometry.max_tension])


@dataclass
class WrapCriterion:
    """Engaged when at least ``min_segments`` segments touch the part and the contacts wrap at
    least ``min_wrap_deg`` around its vertical axis; sub-flag ``no_wrap`` otherwise."""

    geometry: ArmGeometry
    name: str = "wrap"
    min_segments: int = MIN_WRAP_SEGMENTS
    min_wrap_deg: float = MIN_WRAP_DEG
    prefix: str = "eff_"

    def _segments(self, model: mujoco.MjModel) -> np.ndarray:
        return np.array(
            [model.body(f"{self.prefix}seg_{i:02d}").id for i in range(self.geometry.n)], dtype=np.int64
        )

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        return wrap_readout(model, data, scene, self._segments(model))["segments"] >= self.min_segments

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement:
        r = wrap_readout(model, data, scene, self._segments(model))
        ok = r["segments"] >= self.min_segments and r["wrap_deg"] >= self.min_wrap_deg
        return Engagement(ok, () if ok else ("no_wrap",), r)


# --------------------------------------------------------------------------- the family


PRESETS: Final[dict[str, dict[str, Any]]] = {
    "tentacle-12": {"n_segments": 12, "length_mm": 140.0},
    "tentacle-16": {"n_segments": 16, "length_mm": 160.0},
    "tentacle-20-soft": {"n_segments": 20, "length_mm": 180.0, "stiffness_nm_per_rad": 0.2},
}
_PRESET_NAMES: Final = {
    "tentacle-12": "Soft arm, 12 segments, 140 mm",
    "tentacle-16": "Soft arm, 16 segments, 160 mm",
    "tentacle-20-soft": "Soft arm, 20 soft segments, 180 mm",
}


def arm_spec(model: str = "tentacle-16") -> EffectorSpec:
    """A preset arm (``origin = "preset"``): one of :data:`PRESETS`."""
    p = {**_DEFAULTS, **PRESETS[model]}
    ArmGeometry.from_params(p)
    return EffectorSpec(
        family=EffectorFamily.SOFT_ARM,
        model=model,
        display_name=_PRESET_NAMES[model],
        primitive="wrap",
        params=p,
        origin="preset",
    )


class SoftArmFamily(EffectorFamilyImpl):
    """The soft arm family (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.SOFT_ARM
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"
    caveat: ClassVar[str] = FIDELITY_CAVEAT

    def presets(self) -> tuple[EffectorSpec, ...]:
        return tuple(arm_spec(m) for m in PRESETS)

    def recipe_kinds(self) -> tuple[str, ...]:
        return (ARM_KIND,)

    def param_space(self, kind: str) -> tuple[ParamSpec, ...]:
        return PARAM_SPACE if kind == ARM_KIND else ()

    def spec_from_recipe(self, recipe: MorphologyRecipe) -> EffectorSpec:
        if recipe.family != self.family or recipe.kind != ARM_KIND:
            raise ValueError(
                f"not a {self.family.value}/{ARM_KIND} recipe: {recipe.family.value}/{recipe.kind}"
            )
        problems = recipe.problems(PARAM_SPACE)
        if problems:
            raise ValueError("invalid soft arm recipe: " + "; ".join(problems))
        ArmGeometry.from_params(recipe.params)
        n, length = int(recipe.params["n_segments"]), float(recipe.params["length_mm"])
        return EffectorSpec(
            family=EffectorFamily.SOFT_ARM,
            model="designed-tentacle",
            display_name=f"Designed soft arm, {n} segments, {length:.0f} mm",
            primitive="wrap",
            params=dict(recipe.params),
            recipe=recipe,
            origin=recipe.origin,
        )

    # ------------------------------------------------------------------ build

    def build(self, ctx: BuildContext) -> EffectorBuild:
        self.check_spec(ctx.effector)
        if ctx.effector.primitive != "wrap":
            raise ValueError(f"soft arms have one primitive, 'wrap' (got {ctx.effector.primitive!r})")
        g = ArmGeometry.from_params(ctx.effector.params)
        half_width = 0.5 * float(ctx.part.size[1])
        clearance = g.stem_offset - max(g.post_radius, g.root_radius)
        if half_width + 0.002 > clearance:
            raise InvalidEffector(
                f"the part is {2 * half_width * 1000:.0f} mm wide; the arm hangs only "
                f"{clearance * 1000:.0f} mm from the wrist axis"
            )
        p = ctx.prefix
        cap, box = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX
        grey, skin = [0.62, 0.64, 0.68, 1.0], [0.85, 0.42, 0.36, 1.0]
        ys = g.stem_offset

        base = ctx.mount.add_body(name=f"{p}arm_base", pos=[0.0, 0.0, ctx.mount_z])
        y_lo, y_hi = -0.012, ys + g.post_radius
        base.add_geom(
            name=f"{p}arm_housing",
            type=box,
            size=[0.012, 0.5 * (y_hi - y_lo), 0.005],
            pos=[0.0, 0.5 * (y_hi + y_lo), -0.005],
            mass=0.05,
            contype=2,  # the arm's geoms never collide with each other (see the caveat)
            conaffinity=1,
            rgba=grey,
        )
        base.add_geom(
            name=f"{p}arm_post",
            type=cap,
            size=[g.post_radius, 0.0, 0.0],
            fromto=[0.0, ys, -0.010, 0.0, ys, -POST_LENGTH],
            mass=0.03,
            contype=2,
            conaffinity=1,
            rgba=grey,
        )
        base.add_site(name=f"{p}cable_root", pos=[-0.004, ys - g.cable_offset(0), -POST_LENGTH])

        parent = base
        pos = [0.0, ys, -POST_LENGTH]
        lo, hi = (math.radians(v) for v in JOINT_RANGE_DEG)
        sites = [f"{p}cable_root"]
        bodies = [f"{p}arm_base"]
        seg_len = g.segment_length
        for i in range(g.n):
            r = g.radius(i)
            body = parent.add_body(name=f"{p}seg_{i:02d}", pos=pos)
            k = g.joint_stiffness(i)
            body.add_joint(
                name=f"{p}seg_{i:02d}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0.0, 0.0, 1.0],
                range=[lo, hi],
                stiffness=k,
                springref=0.0,
                damping=g.joint_damping(i),
                armature=max(4.0 * k / 250000.0, 1e-7),
            )
            body.add_geom(
                name=f"{p}seg_{i:02d}",
                type=cap,
                size=[r, 0.0, 0.0],
                fromto=[0.0, 0.0, 0.0, seg_len, 0.0, 0.0],
                mass=g.segment_mass(i),
                contype=2,  # segments collide with the part, table and wrist, not with the arm
                conaffinity=1,
                rgba=skin,
            )
            c = g.cable_offset(i)
            for frac, tag in ((0.2, "a"), (0.8, "b")):
                body.add_site(name=f"{p}cable_{i:02d}{tag}", pos=[frac * seg_len, -c, 0.0])
                sites.append(f"{p}cable_{i:02d}{tag}")
            if i == g.n - 1:
                body.add_site(name=f"{p}tip", pos=[seg_len, 0.0, 0.0])
            bodies.append(f"{p}seg_{i:02d}")
            parent = body
            pos = [seg_len, 0.0, 0.0]

        tendon = ctx.spec.add_tendon(name=f"{p}cable")
        for s in sites:
            tendon.wrap_site(s)
        act = ctx.spec.add_actuator(
            name=f"{p}cable_motor", target=f"{p}cable", trntype=mujoco.mjtTrn.mjTRN_TENDON
        )
        act.set_to_motor()
        act.gear = [-1.0, 0, 0, 0, 0, 0]  # positive control pulls the cable (shortens it)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [0.0, g.max_tension]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [0.0, g.max_tension]
        return EffectorBuild(
            spec=ctx.effector,
            bodies=tuple(bodies),
            actuators=(f"{p}cable_motor",),
            ctrl_open=(0.0,),
            reach=g.reach,
            tcp=(0.0, 0.0, ctx.mount_z - POST_LENGTH),
            actuation={
                "cable_tension_n": g.max_tension,
                "segments": g.n,
                "root_joint_stiffness_nm_per_rad": g.stiffness,
                "arm_length_mm": g.length * 1000.0,
            },
            notes=(
                FIDELITY_CAVEAT,
                (
                    f"{g.n} segments, {g.length * 1000:g} mm, radius {g.root_radius * 1000:g} to "
                    f"{g.radius(g.n - 1) * 1000:.1f} mm; cable tension at most {g.max_tension:g} N"
                ),
            ),
            max_opening=None,
        )

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> WrapPrimitive:
        return WrapPrimitive(ArmGeometry.from_params(spec.params))

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> WrapCriterion:
        return WrapCriterion(ArmGeometry.from_params(spec.params))


# --------------------------------------------------------------------------- release test


@dataclass(frozen=True)
class StraightenResult:
    """How the arm returns when the cable is released (:func:`straighten_test`)."""

    curl_deg: float  # total joint curl when the cable is released
    tip_deviation_mm: float  # tip distance from its straight position at release
    settle_time_s: float  # after release, from when the tip stays within the tolerance (inf: never)
    deviation_at_horizon_mm: float  # tip distance from straight at the end of the horizon
    tolerance_mm: float
    horizon_s: float

    @property
    def ok(self) -> bool:
        return self.settle_time_s <= self.horizon_s


def straighten_test(
    spec: EffectorSpec,
    part: PartSpec | None = None,
    *,
    task: TaskV2 | None = None,
    pull_s: float = 1.0,
    horizon_s: float = 1.0,
    tolerance_m: float = 0.005,
) -> StraightenResult:
    """Curl the free arm at full tension for ``pull_s``, release the cable, time the return.

    The arm is built in the shared scene (the wrist parked 0.4 m up, the part far away on the
    table, so nothing is touched) and simulated with the task's timestep; the deviation is the
    tip site's distance from its straight (rest) position.
    """
    from rtt.effectors.scene import build_effector_scene
    from rtt.parts.library import get_part
    from rtt.sim.task import default_task_v2

    impl = _REGISTERED
    task = task or default_task_v2()
    part = part or get_part("block")
    scene = build_effector_scene(impl, spec, part, task)
    g = ArmGeometry.from_params(spec.params)
    model, ids = scene.model, scene.ids
    data = mujoco.MjData(model)
    park = [0.0, 0.0, 0.4, 0.0]
    data.qpos[ids.palm_qpos] = park
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = [0.25, -0.25, 0.5 * part_height(part)]
    data.ctrl[:4] = park
    mujoco.mj_forward(model, data)
    tip = model.site(f"{NAME_PREFIX}tip").id
    straight = data.site_xpos[tip].copy()
    joints = [model.joint(f"{NAME_PREFIX}seg_{i:02d}").qposadr[0] for i in range(g.n)]
    dt = model.opt.timestep
    data.ctrl[4] = g.max_tension
    mujoco.mj_step(model, data, nstep=round(pull_s / dt))
    curl = float(np.degrees(np.abs(data.qpos[joints]).sum()))
    dev0 = float(np.linalg.norm(data.site_xpos[tip] - straight))
    data.ctrl[4] = 0.0
    last_out = 0.0
    dev = dev0
    n = round(horizon_s / dt)
    for k in range(1, n + 1):
        mujoco.mj_step(model, data)
        dev = float(np.linalg.norm(data.site_xpos[tip] - straight))
        if dev > tolerance_m:
            last_out = k * dt
    settle = last_out if dev <= tolerance_m else math.inf
    return StraightenResult(
        curl_deg=curl,
        tip_deviation_mm=1000.0 * dev0,
        settle_time_s=settle,
        deviation_at_horizon_mm=1000.0 * dev,
        tolerance_mm=1000.0 * tolerance_m,
        horizon_s=horizon_s,
    )


_REGISTERED = register_family(SoftArmFamily())
