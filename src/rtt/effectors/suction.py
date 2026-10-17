"""Suction cup family (product mode, approximation): a segmented lip on a bellows, seal-gated vacuum.

**Fidelity: approximation.** Read :data:`FIDELITY_CAVEAT` before trusting a suction result.

The cup hangs from the wrist flange, axis down:

* ``eff_cup_base``: the vacuum fitting, fixed to the mount plane;
* ``eff_cup_bellows``: a slide joint along the cup axis with a spring (the bellows). The lip
  rests at the joint's extended stop; pressing it on a face compresses the spring, which
  presses the lip onto the face with ``stiffness x compression``;
* ``eff_cup``: the cup itself, on two tilt hinges with weak springs (the lip's tilt
  compliance, at most ``max_tilt_deg`` either way), carrying the lip: ``lip_segments`` rigid
  capsules on a circle of the cup diameter, and the cup face above them;
* ``eff_vacuum``: a MuJoCo **adhesion actuator** on the cup body, gain = vacuum x lip area
  (vacuum at most 60 kPa, a typical single-stage ejector; the area is the circle the lip
  encloses), control 0 (off) to 1 (on).

The **seal** (:class:`SealTest`) is geometric: every lip segment is in contact with the part,
and the face under the cup is part material, tested by rays cast along the cup axis at the
centre and on two rings inside the lip (a bore, a tapped hole or a gap under the cup fails it).
The **vacuum switch** (:class:`SealPrimitive`) turns the adhesion on only while the seal
holds: from the start of the shared close ramp the control follows the ramp while sealed and
drops to zero on the control tick the seal is lost, so the part falls.

Fairness (``docs/EFFECTORS.md`` sections 2.3 and 2.4): no geom adhesion, no priority, no
surface velocity, no change to the solver or default classes; the part's friction and contact
parameters govern the lip's contacts; the only force on the part is the actuator's, capped at
vacuum x area. Every effector geom has zero contact margin: a lip segment is "in contact" only
when MuJoCo reports a contact (no force or seal at a distance), and the adhesion acts only on
the cup's active contacts.

Designable by track B3: :meth:`SuctionFamily.param_space` (``"suction_cup"``) and
:meth:`SuctionFamily.spec_from_recipe`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.primitives import part_height
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import (
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    Engagement,
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
    "CUP_KIND",
    "FIDELITY_CAVEAT",
    "PARAM_SPACE",
    "PRESET_DIAMETERS_MM",
    "VACUUM_MAX_KPA",
    "CupGeometry",
    "SealCriterion",
    "SealPrimitive",
    "SealState",
    "SealTest",
    "SuctionFamily",
    "cup_spec",
]

CUP_KIND: Final = "suction_cup"
VACUUM_MAX_KPA: Final = 60.0
#: m: the face under every coverage ray must lie at most this far below the lip plane.
COVERAGE_DEPTH_TOL: Final = 0.001
#: Coverage rays: the centre plus 6 rays on each of these rings (fractions of the lip's inner radius).
COVERAGE_RINGS: Final = (0.5, 0.85)
COVERAGE_RAYS_PER_RING: Final = 6
#: The coverage rays are re-cast when the part has moved relative to the cup by more than these.
COVERAGE_POSE_TOL: Final = 2e-5  # m
COVERAGE_ROT_TOL: Final = 2e-4  # rotation-matrix entries (about 0.01 deg)
#: Fraction of the bellows stroke the primitive presses the lip into the face.
PRESS_FRACTION: Final = 0.5
PRESET_DIAMETERS_MM: Final = (10.0, 16.0, 25.0)

FIDELITY_CAVEAT: Final = (
    "Suction cup (approximation): a compliant lip of rigid segments on a bellows spring with a "
    "limited tilt; the seal is a geometric test (every lip segment touching the face, part "
    "material under 13 rays inside the lip) gating a MuJoCo adhesion actuator capped at vacuum "
    "x lip area (at most 60 kPa). The lip must touch the face: no pull across a gap. Not "
    "modelled: air flow, leak rate, porosity, vacuum build-up time (the vacuum follows the "
    "shared close ramp and is cut the control tick the seal is lost), lip material hysteresis, "
    "peel dynamics beyond the contact distribution. The lip's friction is the part's. Treat "
    "suction results as a screen, then verify on hardware."
)

PARAM_SPACE: Final[tuple[ParamSpec, ...]] = (
    ParamSpec(
        "diameter_mm",
        "float",
        "Diameter of the sealing lip; the vacuum acts on the circle it encloses.",
        "mm",
        8.0,
        40.0,
        default=16.0,
    ),
    ParamSpec(
        "lip_segments",
        "int",
        "Number of rigid segments around the lip; every one must touch the face to seal.",
        "count",
        8,
        24,
        default=12,
    ),
    ParamSpec(
        "bellows_stiffness_n_per_mm",
        "float",
        "Axial spring rate of the bellows that presses the lip onto the face.",
        "N/mm",
        0.2,
        3.0,
        default=1.0,
    ),
    ParamSpec(
        "max_tilt_deg",
        "float",
        "How far the lip can tilt to follow a sloped face before it peels off.",
        "deg",
        3.0,
        20.0,
        default=10.0,
    ),
    ParamSpec(
        "vacuum_kpa",
        "float",
        "Vacuum while the switch is on (a single-stage ejector reaches about 60 kPa).",
        "kPa",
        20.0,
        VACUUM_MAX_KPA,
        default=VACUUM_MAX_KPA,
    ),
)
_DEFAULTS: Final = {p.name: p.default for p in PARAM_SPACE}


@dataclass(frozen=True)
class CupGeometry:
    """Every dimension of one cup (m, kg, N), derived from its params (:data:`PARAM_SPACE`)."""

    diameter: float
    lip_segments: int
    bellows_stiffness: float  # N/m
    max_tilt: float  # rad
    vacuum_pa: float
    lip_radius: float  # m, segment capsule radius
    lip_half_length: float  # m, capsule half-length (tangential)
    base_radius: float
    base_length: float
    bellows_radius: float
    bellows_length: float
    stroke: float
    pivot_height: float  # gimbal above the lip centreline
    face_radius: float
    face_half_thickness: float
    lip_mass: float
    bellows_mass: float
    base_mass: float

    @staticmethod
    def from_params(params: Any) -> CupGeometry:
        p = {**_DEFAULTS, **dict(params)}
        problems = MorphologyRecipe(EffectorFamily.SUCTION, CUP_KIND, p).problems(PARAM_SPACE)
        if problems:
            raise ValueError("suction cup params: " + "; ".join(problems))
        d = float(p["diameter_mm"]) / 1000.0
        n = int(p["lip_segments"])
        r_lip = min(max(0.06 * d, 0.0006), 0.0015)
        chord = d * math.sin(math.pi / n)
        return CupGeometry(
            diameter=d,
            lip_segments=n,
            bellows_stiffness=1000.0 * float(p["bellows_stiffness_n_per_mm"]),
            max_tilt=math.radians(float(p["max_tilt_deg"])),
            vacuum_pa=1000.0 * float(p["vacuum_kpa"]),
            lip_radius=r_lip,
            lip_half_length=max(0.5 * chord - r_lip, 0.0001),
            base_radius=max(0.3 * d, 0.006),
            base_length=0.010,
            bellows_radius=max(0.32 * d, 0.004),
            bellows_length=0.012,
            stroke=0.006,
            pivot_height=0.003,
            face_radius=0.5 * d - r_lip,
            face_half_thickness=0.001,
            lip_mass=0.004 + 0.02 * (d / 0.025) ** 2,  # cup and lip (elastomer on a stiff insert)
            bellows_mass=0.006 + 0.01 * (d / 0.025),
            base_mass=0.03,  # aluminium fitting
        )

    @property
    def area(self) -> float:
        """m^2: the circle the lip encloses (the vacuum's effective area)."""
        return math.pi * (0.5 * self.diameter) ** 2

    @property
    def max_force(self) -> float:
        """N: the holding force cap, vacuum x lip area."""
        return self.vacuum_pa * self.area

    @property
    def press_depth(self) -> float:
        return PRESS_FRACTION * self.stroke

    @property
    def reach(self) -> float:
        """m from the mount plane to the lip's lowest point (extended bellows, no tilt)."""
        return self.base_length + self.bellows_length + self.pivot_height + self.lip_radius


def _name(prefix: str, what: str) -> str:
    return f"{prefix}{what}"


# --------------------------------------------------------------------------- the seal test


@dataclass(frozen=True)
class SealState:
    """The seal test's readout at one instant."""

    lip_touching: int
    lip_total: int
    rays_on_part: int
    rays_total: int
    face_depth_max: float  # m below the lip plane, over the coverage rays (inf if a ray missed)

    @property
    def lip_sealed(self) -> bool:
        return self.lip_touching == self.lip_total

    @property
    def covered(self) -> bool:
        return self.rays_on_part == self.rays_total

    @property
    def sealed(self) -> bool:
        return self.lip_sealed and self.covered


@dataclass
class SealTest:
    """Geometric seal test (module docstring); one instance per episode.

    ``geometry`` is the cup's :class:`CupGeometry`, ``prefix`` the scene's name prefix. The
    readout is cached per simulation time, and the coverage rays (which depend only on where
    the part is relative to the cup) are re-cast only when that relative pose has moved by
    more than :data:`COVERAGE_POSE_TOL` m or :data:`COVERAGE_ROT_TOL` (rotation-matrix entries).
    """

    geometry: CupGeometry
    prefix: str = "eff_"
    _model_id: int = field(default=-1, repr=False)
    _lip_geoms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    _is_lip: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)
    _is_obj: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)
    _cup_body: int = field(default=-1, repr=False)
    _vacuum: int = field(default=-1, repr=False)
    _ray_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    _ray_start: float = field(default=0.0, repr=False)
    _geomid: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int32), repr=False)
    _obj_body: int = field(default=-1, repr=False)
    _cache_t: float = field(default=math.nan, repr=False)
    _cache: SealState | None = field(default=None, repr=False)
    _cov_pose: np.ndarray | None = field(default=None, repr=False)
    _cov: tuple[int, float] = field(default=(0, math.inf), repr=False)

    def _resolve(self, model: mujoco.MjModel, scene: Any) -> None:
        if self._model_id == id(model):
            return
        g = self.geometry
        names = [_name(self.prefix, f"lip_{i:02d}") for i in range(g.lip_segments)]
        self._lip_geoms = np.array([model.geom(n).id for n in names], dtype=np.int64)
        self._is_lip = np.zeros(model.ngeom, dtype=bool)
        self._is_lip[self._lip_geoms] = True
        self._is_obj = np.zeros(model.ngeom, dtype=bool)
        self._is_obj[scene.ids.object_geoms] = True
        self._cup_body = model.body(_name(self.prefix, "cup")).id
        self._vacuum = model.actuator(_name(self.prefix, "vacuum")).id
        self._obj_body = int(scene.ids.object_body)
        self._cov_pose = None
        lip_bottom = -(g.pivot_height + g.lip_radius)  # cup frame
        self._ray_start = 1.5 * g.lip_radius + 0.0005  # above the lip bottom, below the cup face
        inner = 0.5 * g.diameter - g.lip_radius
        pts = [(0.0, 0.0)]
        for ring in COVERAGE_RINGS:
            for k in range(COVERAGE_RAYS_PER_RING):
                a = 2.0 * math.pi * (k + 0.5 * (ring == COVERAGE_RINGS[-1])) / COVERAGE_RAYS_PER_RING
                pts.append((ring * inner * math.cos(a), ring * inner * math.sin(a)))
        self._ray_points = np.array([(x, y, lip_bottom + self._ray_start) for x, y in pts])
        self._model_id = id(model)

    def vacuum_actuator(self, model: mujoco.MjModel, scene: Any) -> int:
        self._resolve(model, scene)
        return self._vacuum

    def lip_contacts(self, model: mujoco.MjModel, data: mujoco.MjData) -> int:
        """How many lip segments are in (active) contact with the part."""
        n = data.ncon
        if not n:
            return 0
        con = data.contact
        g = con.geom
        near = con.efc_address >= 0
        a = self._is_lip[g[:, 0]] & self._is_obj[g[:, 1]] & near
        b = self._is_obj[g[:, 0]] & self._is_lip[g[:, 1]] & near
        if not (a.any() or b.any()):
            return 0
        touching = np.concatenate([g[a, 0], g[b, 1]])
        return int(np.unique(touching).size)

    def _relative_pose(self, data: mujoco.MjData) -> np.ndarray:
        rot = data.xmat[self._cup_body].reshape(3, 3)
        pos = rot.T @ (data.xpos[self._obj_body] - data.xpos[self._cup_body])
        return np.concatenate([pos, (rot.T @ data.xmat[self._obj_body].reshape(3, 3)).ravel()])

    def coverage(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, float]:
        """(rays that meet part material within the depth tolerance, deepest face below the lip)."""
        pose = self._relative_pose(data)
        last = self._cov_pose
        if (
            last is not None
            and np.abs(pose[:3] - last[:3]).max() <= COVERAGE_POSE_TOL
            and np.abs(pose[3:] - last[3:]).max() <= COVERAGE_ROT_TOL
        ):
            return self._cov
        rot = data.xmat[self._cup_body].reshape(3, 3)
        origin = data.xpos[self._cup_body]
        down = -rot[:, 2].copy()
        hits, deepest = 0, 0.0
        for p in self._ray_points:
            start = origin + rot @ p
            dist = mujoco.mj_ray(model, data, start, down, None, False, self._cup_body, self._geomid)
            depth = dist - self._ray_start if dist >= 0 else math.inf
            if dist >= 0 and self._is_obj[int(self._geomid[0])] and depth <= COVERAGE_DEPTH_TOL:
                hits += 1
                deepest = max(deepest, depth)
            else:
                deepest = math.inf
        self._cov_pose, self._cov = pose, (hits, deepest)
        return hits, deepest

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> SealState:
        if data.time == self._cache_t and self._cache is not None:
            return self._cache
        self._resolve(model, scene)
        total = int(self._lip_geoms.size)
        touching = self.lip_contacts(model, data)
        n_rays = len(self._ray_points)
        if touching == total:
            hits, deepest = self.coverage(model, data)
        else:
            hits, deepest = 0, math.inf  # rays are only cast when the lip is on the face
        self._cache = SealState(touching, total, hits, n_rays, deepest)
        self._cache_t = float(data.time)
        return self._cache


# --------------------------------------------------------------------------- primitive and criterion


@dataclass
class SealPrimitive:
    """``seal``: press the lip into the face, switch the vacuum on while sealed, lift.

    The lip's lowest point goes :attr:`CupGeometry.press_depth` below the part's top (the
    bellows takes the overtravel; the same rule on every part). From the start of the shared
    close ramp the vacuum control is ``closure`` while :class:`SealTest` holds and 0 the tick
    it fails, through the lift and the hold.
    """

    geometry: CupGeometry
    seal: SealTest
    name: str = "seal"

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        tip = max(task.controller.min_tip_clearance, part_height(part) - self.geometry.press_depth)
        return PrimitivePlan(tip_height=tip, note="lip pressed into the part's top face")

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        if phase < Phase.CLOSE or closure <= 0.0:
            return np.zeros(1)
        return np.array([closure if self.seal(model, data, scene).sealed else 0.0])


@dataclass
class SealCriterion:
    """Engaged when the seal holds and the vacuum switch is on (sub-flag ``no_seal`` otherwise)."""

    geometry: CupGeometry
    seal: SealTest
    name: str = "seal"

    def _state(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> tuple[bool, dict[str, float]]:
        st = self.seal(model, data, scene)
        act = self.seal.vacuum_actuator(model, scene)
        on = float(data.ctrl[act]) > 0.0
        metrics = {
            "lip_touching": float(st.lip_touching),
            "lip_segments": float(st.lip_total),
            "coverage_rays_on_part": float(st.rays_on_part),
            "coverage_rays": float(st.rays_total),
            "vacuum_on": float(on),
            "holding_force_cap_n": self.geometry.max_force,
            "vacuum_force_n": float(data.actuator_force[act]),
        }
        return st.sealed and on, metrics

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        return self._state(model, data, scene)[0]

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement:
        ok, metrics = self._state(model, data, scene)
        return Engagement(ok, () if ok else ("no_seal",), metrics)


# --------------------------------------------------------------------------- the family


def _display(diameter_mm: float) -> str:
    return f"Suction cup, {diameter_mm:g} mm bellows"


def cup_spec(diameter_mm: float = 16.0, **params: Any) -> EffectorSpec:
    """A preset cup spec (``origin = "preset"``); other params default to :data:`PARAM_SPACE`."""
    p = {**_DEFAULTS, **params, "diameter_mm": float(diameter_mm)}
    CupGeometry.from_params(p)  # validate
    return EffectorSpec(
        family=EffectorFamily.SUCTION,
        model=f"cup-{diameter_mm:g}mm".replace(".", "p"),
        display_name=_display(diameter_mm),
        primitive="seal",
        params=p,
        origin="preset",
    )


class SuctionFamily(EffectorFamilyImpl):
    """The suction family (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.SUCTION
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"
    caveat: ClassVar[str] = FIDELITY_CAVEAT

    def presets(self) -> tuple[EffectorSpec, ...]:
        return tuple(cup_spec(d) for d in PRESET_DIAMETERS_MM)

    def recipe_kinds(self) -> tuple[str, ...]:
        return (CUP_KIND,)

    def param_space(self, kind: str) -> tuple[ParamSpec, ...]:
        return PARAM_SPACE if kind == CUP_KIND else ()

    def spec_from_recipe(self, recipe: MorphologyRecipe) -> EffectorSpec:
        if recipe.family != self.family or recipe.kind != CUP_KIND:
            raise ValueError(
                f"not a {self.family.value}/{CUP_KIND} recipe: {recipe.family.value}/{recipe.kind}"
            )
        problems = recipe.problems(PARAM_SPACE)
        if problems:
            raise ValueError("invalid suction cup recipe: " + "; ".join(problems))
        d = float(recipe.params["diameter_mm"])
        return EffectorSpec(
            family=EffectorFamily.SUCTION,
            model="designed-cup",
            display_name=f"Designed suction cup, {d:.0f} mm",
            primitive="seal",
            params=dict(recipe.params),
            recipe=recipe,
            origin=recipe.origin,
        )

    # ------------------------------------------------------------------ build

    def build(self, ctx: BuildContext) -> EffectorBuild:
        self.check_spec(ctx.effector)
        if ctx.effector.primitive != "seal":
            raise ValueError(f"suction cups have one primitive, 'seal' (got {ctx.effector.primitive!r})")
        g = CupGeometry.from_params(ctx.effector.params)
        p = ctx.prefix
        cyl, cap = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE
        grey, dark, lip_rgba = [0.62, 0.64, 0.68, 1.0], [0.22, 0.23, 0.26, 1.0], [0.12, 0.13, 0.15, 1.0]

        base = ctx.mount.add_body(name=f"{p}cup_base", pos=[0.0, 0.0, ctx.mount_z])
        base.add_geom(
            name=f"{p}cup_base",
            type=cyl,
            size=[g.base_radius, 0.5 * g.base_length, 0.0],
            pos=[0.0, 0.0, -0.5 * g.base_length],
            mass=g.base_mass,
            rgba=grey,
        )
        bellows = base.add_body(name=f"{p}cup_bellows", pos=[0.0, 0.0, -g.base_length])
        m_moving = g.bellows_mass + g.lip_mass
        bellows.add_joint(
            name=f"{p}bellows",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],  # + = compressed (the cup pushed up toward the mount)
            range=[0.0, g.stroke],
            stiffness=g.bellows_stiffness,
            springref=0.0,
            damping=2.0 * math.sqrt(g.bellows_stiffness * m_moving),  # critically damped
            armature=0.002,
        )
        bellows.add_geom(
            name=f"{p}cup_bellows",
            type=cyl,
            size=[g.bellows_radius, 0.5 * g.bellows_length, 0.0],
            pos=[0.0, 0.0, -0.5 * g.bellows_length],
            mass=g.bellows_mass,
            rgba=dark,
        )
        cup = bellows.add_body(name=f"{p}cup", pos=[0.0, 0.0, -g.bellows_length])  # the gimbal
        i_cup = 0.5 * g.lip_mass * (0.5 * g.diameter) ** 2
        k_tilt = 0.004 * (g.diameter / 0.016) ** 2  # N m/rad: a soft lip that follows the face
        for axis_name, axis in (("tilt_x", [1.0, 0.0, 0.0]), ("tilt_y", [0.0, 1.0, 0.0])):
            cup.add_joint(
                name=f"{p}cup_{axis_name}",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=axis,
                pos=[0.0, 0.0, 0.0],
                range=[-g.max_tilt, g.max_tilt],
                stiffness=k_tilt,
                damping=2.0 * math.sqrt(k_tilt * (i_cup + 2e-6)),
                armature=2e-6,
            )
        # the cup face (a thin disc above the lip) and the lip ring
        lip_z = g.pivot_height  # gimbal above the lip centreline
        face_z = -lip_z + g.lip_radius + 0.0002 + g.face_half_thickness
        face_mass = 0.5 * g.lip_mass
        cup.add_geom(
            name=f"{p}cup_face",
            type=cyl,
            size=[g.face_radius, g.face_half_thickness, 0.0],
            pos=[0.0, 0.0, face_z],
            mass=face_mass,
            rgba=dark,
        )
        seg_mass = 0.5 * g.lip_mass / g.lip_segments
        radius = 0.5 * g.diameter
        for i in range(g.lip_segments):
            a = 2.0 * math.pi * i / g.lip_segments
            c, s = math.cos(a), math.sin(a)
            # capsule axis along the tangent (z axis rotated onto (-s, c, 0))
            half = 0.5 * (a + 0.5 * math.pi)
            quat_tangent = [math.cos(half), 0.0, 0.0, math.sin(half)]  # yaw the x axis onto the tangent
            quat = _quat_mul(quat_tangent, [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])  # z onto x
            cup.add_geom(
                name=f"{p}lip_{i:02d}",
                type=cap,
                size=[g.lip_radius, g.lip_half_length, 0.0],
                pos=[radius * c, radius * s, -lip_z],
                quat=quat,
                mass=seg_mass,
                rgba=lip_rgba,
            )
        act = ctx.spec.add_actuator(name=f"{p}vacuum", target=f"{p}cup", trntype=mujoco.mjtTrn.mjTRN_BODY)
        act.set_to_adhesion(g.max_force)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [0.0, 1.0]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [0.0, g.max_force]
        return EffectorBuild(
            spec=ctx.effector,
            bodies=(f"{p}cup_base", f"{p}cup_bellows", f"{p}cup"),
            actuators=(f"{p}vacuum",),
            ctrl_open=(0.0,),
            reach=g.reach,
            tcp=(0.0, 0.0, ctx.mount_z - g.reach),
            actuation={
                "vacuum_kpa": g.vacuum_pa / 1000.0,
                "lip_area_mm2": g.area * 1e6,
                "holding_force_max_n": g.max_force,
                "bellows_stiffness_n_per_mm": g.bellows_stiffness / 1000.0,
                "bellows_stroke_mm": g.stroke * 1000.0,
                "max_tilt_deg": math.degrees(g.max_tilt),
            },
            notes=(
                FIDELITY_CAVEAT,
                (
                    f"{g.lip_segments} lip segments on a {g.diameter * 1000:g} mm circle; "
                    f"holding force at most {g.max_force:.1f} N ({g.vacuum_pa / 1000:g} kPa x "
                    f"{g.area * 1e6:.0f} mm^2)"
                ),
            ),
            max_opening=None,
        )

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> SealPrimitive:
        g = CupGeometry.from_params(spec.params)
        return SealPrimitive(g, SealTest(g))

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> SealCriterion:
        g = CupGeometry.from_params(spec.params)
        return SealCriterion(g, SealTest(g))


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return [float(v) for v in out]


register_family(SuctionFamily())
