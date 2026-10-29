"""Process tools (Studio v2, approximation): a nut runner and a tamper pad.

A grasp holds a part; a **process tool** does something to it: tightens a nut, presses a tile level.
Each tool here is used by the Studio's own primitive (``twist`` for the nut runner, ``push`` for the
tamper), so the same controller, budgets, success predicates and fairness rules judge it as every
other head, and the twist and push primitives are unchanged for every tool. Each tool has one
actuator that is a limiter, not a motor. **Fidelity: approximation.** Read :data:`FIDELITY_CAVEAT`
before trusting a process result.

**Nut runner** (``nut-runner-socket-13``, primitive ``drive``). From the mount plane down: the drive
housing; a **lateral float** (two short slides on weak springs); a **spindle hinge** about the tool
axis; and a **hex socket**. The socket is six steel walls round a hex cavity (the member's flats plus
:data:`SOCKET_CLEARANCE` a flat), a guide funnel at the mouth that lets it find a hex member that is up
to a few millimetres off the aim point (the float lets it centre), and a closed end that bears on the
member's top. It **engages a hex member by shape** (form closure on the flats), so the torque it
passes does not depend on friction.

* **The spindle is the wrist's roll** [assumed approximation]: the twist primitive turns the wrist
  roll, the one rotation every candidate has, at the wrist's speed (at most 180 deg/s) and torque
  (15 N m; ``STUDIO_WRIST``; a test keeps :data:`ROLL_TORQUE_NM` and :data:`ROLL_SPEED_DPS` equal to
  it). The tool has no motor and no speed of its own. What it does have is a **clutch** (the
  actuator): a spring and damper from the housing to the socket's neutral, ``torque cap`` strong at
  most, so the socket turns with the housing and, past the cap, slips (a torque limiter). The cap is
  10 N m for the preset [assumed], under the 15 N m of the wrist and at the most this simulator can
  deliver: past about 10.5 N m a hex socket rides up over a member's corners
  (:data:`TORQUE_CAP_MAX_NM`, measured), so a cap above it is refused.
* **The clutch lets go when the primitive's close ramp falls** (the release): a driven member that the
  wrist then lifted off with the socket would be pulled up its thread (the threaded-joint template
  lets a member lift off the flank freely), so at the release the socket is let go (no spring, no
  damper to the housing, which may still be turning back), turns to the middle of its clearance on
  the member's flats and lifts off without dragging it. While the socket is clear of the member the
  same actuator keeps it centred to the housing, so the next engagement finds its flats square to the
  tool, where the twist primitive aims them.
* **The wall is built from slices.** A soft-contact solver gives every contact point the stiffness of
  the lightest body it touches, here the member, so a hex drive built from six solid pieces rides up
  over a member's corners at 1.6 N m. Each wall is :data:`WALL_SLICES` steel slices stacked along the
  cavity; the torque the hex passes grows with the number of contact points, to 10.5 N m at 24 slices
  [assumed, a solver artifact].
* The **rated torque** is the cap over :data:`RATED_SAFETY_FACTOR` [assumed], 6.7 N m for the preset.
* **Supported members** (measured in the Studio, 13 mm hex, 12 mm socket): a member at least
  :data:`MIN_MEMBER_MM` tall, as tall as the socket is deep. Shorter ones are unsupported in the Studio
  today. Its twist and insert controllers aim the tool's tip 3 mm above the member's bottom, so only
  the member's height less 3 mm enters the socket and the 3 mm funnel takes that much of it; and the
  torque the socket passes grows with how much of its wall is engaged (the slices above). At the rated
  6.7 N m: 11.5 mm and taller pass; 11 mm slips at 6.2 N m (5.9 N m is its limit); 10.5 mm slips at
  5.3 N m; 10 mm and below jam on insert (depth under 0.8 mm). A shallower socket does not help (an
  8 mm or a 6 mm socket starts a 6.8 to 8 mm nut and slips at 1.2 to 3.3 N m).

**Tamper pad** (``tamper-pad-200``, primitive ``press``). The housing on the mount plane, a stage (a
slide) and a flat 200 mm square pad on it. The pad stays level (it has no tilt compliance: a tamper
levels a tile, it does not follow it). The stage is a spring that slips at an overload force: the actuator
applies ``min(stiffness x compression + damping x speed, force cap)`` against the compression, so the
pad never presses harder than the cap (within the stage's travel: a step whose force limit lets the
wrist drive past 60 mm of travel bottoms the stage out and the cap no longer holds). The **rated press
force** is the cap over :data:`RATED_SAFETY_FACTOR`, 40 N for the preset.

Both tools are Studio candidates (their presets) and are never in the per-part pick tournament
(:attr:`~rtt.effectors.types.EntrantGroup.PROCESS`). Tool and primitive follow the effector contract
(``docs/EFFECTORS.md`` sections 2 and 4); fairness (section 2.3): tool geoms have no priority,
adhesion, margin or surface velocity, the part's friction and contact parameters govern every contact.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

import mujoco
import numpy as np

from rtt.effectors.heights import top_height
from rtt.effectors.primitives import object_contacts
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import (
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    Engagement,
    PrimitivePlan,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.controller import Phase
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2

__all__ = [
    "FIDELITY_CAVEAT",
    "MIN_MEMBER_MM",
    "NUT_RUNNER_CAVEAT",
    "RATED_SAFETY_FACTOR",
    "ROLL_SPEED_DPS",
    "ROLL_TORQUE_NM",
    "TAMPER_CAVEAT",
    "TORQUE_CAP_MAX_NM",
    "WALL_SLICES",
    "NutRunnerGeometry",
    "PadCriterion",
    "PadPrimitive",
    "ProcessToolFamily",
    "SocketCriterion",
    "SocketPrimitive",
    "TamperGeometry",
    "clutch_weight",
    "nut_runner_spec",
    "tamper_spec",
]

#: The wrist's roll limits (``rtt.studio.types.STUDIO_WRIST``; ``tests/test_effectors_process.py``
#: keeps these equal to it): the nut runner's spindle is that roll.
ROLL_TORQUE_NM: Final = 15.0
ROLL_SPEED_DPS: Final = 180.0
#: A tool's rated load is its capacity over this factor [assumed: common industry practice; no public
#: source is named, and the vacuum frames' safety factors are assumed too].
RATED_SAFETY_FACTOR: Final = 1.5
#: The most torque the clutch may be set to (N m) [measured]: past about 10.5 N m a socket of
#: :data:`WALL_SLICES` slices rides up over the member's corners in this simulator.
TORQUE_CAP_MAX_NM: Final = 10.0
#: The shortest member the nut runner starts and tightens to its rated torque in the Studio, with the
#: 12 mm socket (mm) [measured; 11.5 mm also passes, with no margin]; see the module docstring.
MIN_MEMBER_MM: Final = 12.0

# --- the nut runner's socket

#: Clearance between a hex socket's flats and the member's, each flat (m) [assumed]. Tighter (0.05 mm)
#: and the socket jams on the member's top edge under the aim error; 0.2 mm is a socket's usual fit.
SOCKET_CLEARANCE: Final = 0.0002
#: The guide funnel at the socket's mouth: how far its mouth is wider than the cavity each side, and
#: how tall it is (m) [assumed]. The member's aim error is at most 4 mm a side (2 mm condition error
#: and 2 mm jitter); the corners of a hex member reach 15 % past its flats, so the mouth takes 6 mm.
FUNNEL_FLARE: Final = 0.006
FUNNEL_HEIGHT: Final = 0.003
#: Wall of the socket's body beyond the funnel's mouth (m).
SOCKET_RIM: Final = 0.002
#: Steel slices per flat (module docstring) [assumed, a solver artifact].
WALL_SLICES: int = 24
STEEL_KG_M3: Final = 7800.0
#: The lateral float: travel either side (m), spring rate (N/m) and the moving mass it is damped for (kg),
#: all [assumed]. 20 N/m is 0.02 N per mm, so a funnel centres it with under 1 N and a socket let go
#: presses on nothing.
FLOAT_RANGE: Final = 0.007
FLOAT_STIFFNESS: Final = 20.0
FLOAT_MASS: Final = 0.16

# --- the nut runner's clutch (the spindle hinge's actuator, SocketPrimitive.command)

#: Added inertia on the spindle hinge (kg m^2) [assumed]: a stability-tuning inertia, not a physical one.
#: It keeps the explicit clutch law (SPINDLE_KP, SPINDLE_KV at the 2 ms control period) stable; a real
#: socket and clutch hub are orders of magnitude lighter. The tool has no motor; the wrist turns it.
SPINDLE_ARMATURE: Final = 0.02
#: Held: spring and damper (N m/rad, N m s/rad) from the housing to the socket's neutral. Stable at
#: the 2 ms control period (``k T^2 / I`` and ``c T / I`` well under 1) [assumed].
SPINDLE_KP: Final = 300.0
SPINDLE_KV: Final = 3.0
#: Clear of the member and not held: softer gains that keep the socket's flats square to the tool.
SPINDLE_CENTRE_KP: Final = 5.0
SPINDLE_CENTRE_KV: Final = 0.4
#: Let go with the member in the socket: a damper on the socket's turning rate relative to the member
#: (N m s/rad; it drags nothing when the member itself turns back as the torque comes off, and has no
#: spring to the housing, which may be moving) stops the kick the stored contact energy gives the
#: socket, and a spring (N m/rad) turns it to the middle of its clearance on the member's flats (the
#: funnel faces' self-aligning action on the member's corners) [assumed].
SPINDLE_FOLLOW: Final = 1.0
SPINDLE_ALIGN: Final = 20.0
#: The socket counts as clear of the member when its rim is this far past every object geom (m).
CLEAR_MARGIN: Final = 0.002
#: The clutch holds from this close ramp value on and is let go below 90 % of it.
CLUTCH_FROM: Final = 0.9

NUT_RUNNER_CAVEAT: Final = (
    "Nut runner (approximation): a rigid hex socket with a guide funnel on a lateral float; it engages a "
    "hex member by shape. The Studio wrist's roll is its spindle [assumed], at the wrist's speed: a "
    "clutch limits the torque to its cap (10 N m, under the wrist's 15 N m) and lets go at the release. "
    "The socket's walls are stacked slices so the simulator's contacts are as stiff as steel (a solver "
    "artifact). Supported in the Studio: members at least 12 mm tall. Not modelled: a motor, a torque "
    "transducer, cross-threading, socket wear, a reaction arm. Simulated, verify on hardware."
)
TAMPER_CAVEAT: Final = (
    "Tamper pad (approximation): a rigid, level 200 mm square pad on a spring stage that slips at 60 N, "
    "so the pad never presses harder than that within the stage's 60 mm of travel. Not modelled: pad "
    "deformation, vibration or impact compaction, mortar flow, the pad's edge catching a joint. "
    "Simulated, verify on hardware."
)
FIDELITY_CAVEAT: Final = (
    "Process tool (approximation): a rigid tool used by the Studio's own twist or push primitive. The "
    "nut runner's spindle is the wrist's roll through a torque-limiting clutch [assumed]; the tamper's "
    "stage is a spring that slips at its overload force. The torque, the force and the contact are "
    "simulated; the process (thread cross-threading, bed compaction) is not modelled beyond them. "
    "Verify on hardware."
)


def _positive(name: str, v: Any, lo: float, hi: float, unit: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    if not lo <= float(v) <= hi:
        raise ValueError(f"{name} must be in [{lo:g}, {hi:g}] {unit}")
    return float(v)


def _check_keys(what: str, params: dict[str, Any], known: tuple[str, ...]) -> None:
    unknown, missing = sorted(set(params) - set(known)), [k for k in known if k not in params]
    if unknown or missing:
        raise ValueError(f"{what} params: unknown {unknown}, missing {missing}")


# --------------------------------------------------------------------------- nut runner


@dataclass(frozen=True)
class NutRunnerGeometry:
    """Every dimension of a nut runner (m, kg, N m), from its params."""

    across_flats: float
    socket_depth: float  # cavity depth from the mouth, funnel included
    torque_cap: float  # N m: the clutch's limit

    KEYS: ClassVar[tuple[str, ...]] = (
        "across_flats_mm",
        "socket_depth_mm",
        "torque_cap_nm",
    )
    housing_radius: ClassVar[float] = 0.022
    housing_height: ClassVar[float] = 0.035
    shank_radius: ClassVar[float] = 0.007
    shank_length: ClassVar[float] = 0.015
    plate_thickness: ClassVar[float] = 0.003

    @staticmethod
    def from_params(params: Any) -> NutRunnerGeometry:
        p = dict(params)
        _check_keys("nut runner", p, NutRunnerGeometry.KEYS)
        return NutRunnerGeometry(
            across_flats=_positive("across_flats_mm", p["across_flats_mm"], 6.0, 24.0, "mm") / 1000.0,
            socket_depth=_positive("socket_depth_mm", p["socket_depth_mm"], 6.0, 40.0, "mm") / 1000.0,
            torque_cap=_positive("torque_cap_nm", p["torque_cap_nm"], 0.1, TORQUE_CAP_MAX_NM, "N m"),
        )

    @property
    def apothem(self) -> float:
        """m from the axis to a flat of the cavity."""
        return 0.5 * self.across_flats + SOCKET_CLEARANCE

    @property
    def mouth_apothem(self) -> float:
        return self.apothem + FUNNEL_FLARE

    @property
    def outer_apothem(self) -> float:
        return self.mouth_apothem + SOCKET_RIM

    @property
    def outer_radius(self) -> float:
        """The socket body's corner radius."""
        return self.outer_apothem / math.cos(math.pi / 6.0)

    @property
    def reach(self) -> float:
        """m from the mount plane to the socket's rim (the tool's lowest point)."""
        return self.housing_height + self.rim_from_stage

    @property
    def rim_from_stage(self) -> float:
        """m from the stage plane (the socket body's origin) down to the socket's rim."""
        return self.shank_length + self.plate_thickness + self.socket_depth

    @property
    def rated_torque(self) -> float:
        return self.torque_cap / RATED_SAFETY_FACTOR


def nut_runner_spec(across_flats_mm: float = 13.0, **params: Any) -> EffectorSpec:
    """A nut runner spec (``origin = "preset"``): a hex socket of ``across_flats_mm``, 12 mm deep,
    its clutch at 10 N m [assumed]; the spindle is the wrist roll, at the wrist's speed."""
    p = {
        "across_flats_mm": float(across_flats_mm),
        "socket_depth_mm": 12.0,  # [assumed]: 12 mm is the measured, supported depth (MIN_MEMBER_MM)
        "torque_cap_nm": TORQUE_CAP_MAX_NM,  # [assumed]: the most the simulator can deliver
        **params,
    }
    NutRunnerGeometry.from_params(p)  # validate
    return EffectorSpec(
        family=EffectorFamily.PROCESS_TOOL,
        model=f"nut-runner-socket-{across_flats_mm:g}".replace(".", "p"),
        display_name=f"Nut runner, {across_flats_mm:g} mm hex socket",
        primitive="drive",
        params=p,
        origin="preset",
    )


def _wall_pieces(g: NutRunnerGeometry, k: int, z_top: float, slices: int) -> list[np.ndarray]:
    """The convex pieces of wall ``k`` of the socket body (the stage frame: z up, the mouth at
    ``z_top - socket_depth``): the funnel piece at the mouth, then the straight wall in ``slices``
    equal slices. Wall ``k`` has its flat normal at 30 + 60 k degrees, so one flat faces the tool's
    +y (the axis the twist primitive aims at a flat's normal)."""
    half = math.pi / 6.0
    a = math.radians(30.0 + 60.0 * k)
    z_mouth = z_top - g.socket_depth
    z_funnel = z_mouth + FUNNEL_HEIGHT

    def ring(levels: list[tuple[float, float]]) -> np.ndarray:
        """Vertices at (apothem, z) pairs on both mitre planes of the wall."""
        pts = []
        for ang in (a - half, a + half):
            c, s = math.cos(ang), math.sin(ang)
            for apothem, z in levels:
                r = apothem / math.cos(half)
                pts.append([r * c, r * s, z])
        return np.array(pts, dtype=np.float64)

    edges = np.linspace(z_funnel, z_top, slices + 1)
    # the funnel and the first straight slice are one convex piece: no ledge for the member's top
    # face to catch on where the funnel meets the straight wall
    first = float(edges[1])
    out = [
        ring(
            [
                (g.mouth_apothem, z_mouth),
                (g.outer_apothem, z_mouth),
                (g.apothem, z_funnel),
                (g.apothem, first),
                (g.outer_apothem, first),
            ]
        )
    ]
    for z0, z1 in itertools.pairwise(edges[1:]):
        a0, a1 = (g.apothem, float(z0)), (g.apothem, float(z1))
        o0, o1 = (g.outer_apothem, float(z0)), (g.outer_apothem, float(z1))
        out.append(ring([a0, o0, a1, o1]))
    return out


def _build_nut_runner(ctx: BuildContext, g: NutRunnerGeometry) -> EffectorBuild:
    p = ctx.prefix
    cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
    steel, grey = [0.55, 0.57, 0.62, 1.0], [0.34, 0.36, 0.40, 1.0]
    housing = ctx.mount.add_body(name=f"{p}nut_housing", pos=[0.0, 0.0, ctx.mount_z])
    housing.add_geom(
        name=f"{p}nut_housing",
        type=cyl,
        size=[g.housing_radius, 0.5 * g.housing_height, 0.0],
        pos=[0.0, 0.0, -0.5 * g.housing_height],
        mass=0.20,
        rgba=grey,
    )
    # the lateral float: x then y, each a short slide on a weak spring, damped critically for the
    # socket's mass (the stage body carries almost none)
    damping = 2.0 * math.sqrt(FLOAT_STIFFNESS * FLOAT_MASS)
    stage = housing.add_body(name=f"{p}nut_stage", pos=[0.0, 0.0, -g.housing_height])
    stage.explicitinertial = True
    stage.mass = 0.01
    stage.fullinertia = [1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0]
    stage.add_joint(
        name=f"{p}nut_float_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1.0, 0.0, 0.0],
        range=[-FLOAT_RANGE, FLOAT_RANGE],
        stiffness=FLOAT_STIFFNESS,
        damping=damping,
        armature=1e-4,
    )
    socket = stage.add_body(name=f"{p}nut_socket", pos=[0.0, 0.0, 0.0])
    socket.add_joint(
        name=f"{p}nut_float_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 1.0, 0.0],
        range=[-FLOAT_RANGE, FLOAT_RANGE],
        stiffness=FLOAT_STIFFNESS,
        damping=damping,
        armature=1e-4,
    )
    socket.add_joint(
        name=f"{p}nut_spindle",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
        damping=0.01,
        armature=SPINDLE_ARMATURE,
    )
    socket.add_geom(
        name=f"{p}nut_shank",
        type=cyl,
        size=[g.shank_radius, 0.5 * g.shank_length, 0.0],
        pos=[0.0, 0.0, -0.5 * g.shank_length],
        density=STEEL_KG_M3,
        rgba=steel,
    )
    z_top = -(g.shank_length + g.plate_thickness)
    socket.add_geom(
        name=f"{p}nut_plate",
        type=cyl,
        size=[g.outer_radius, 0.5 * g.plate_thickness, 0.0],
        pos=[0.0, 0.0, z_top + 0.5 * g.plate_thickness],
        density=STEEL_KG_M3,
        rgba=steel,
    )
    for k in range(6):
        for i, verts in enumerate(_wall_pieces(g, k, z_top, WALL_SLICES)):
            mesh = f"{p}nut_wall_{k}_{i}"
            ctx.spec.add_mesh(name=mesh, uservert=verts.ravel().tolist())
            socket.add_geom(
                name=f"{p}nut_wall_{k}_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                density=STEEL_KG_M3,
                rgba=steel,
            )
    clutch = ctx.spec.add_actuator(
        name=f"{p}nut_clutch", target=f"{p}nut_spindle", trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    clutch.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    clutch.gainprm[0] = 1.0  # the control is the torque (N m): the primitive's own law
    clutch.biastype = mujoco.mjtBias.mjBIAS_NONE
    clutch.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    clutch.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    clutch.forcerange = [-g.torque_cap, g.torque_cap]
    return EffectorBuild(
        spec=ctx.effector,
        bodies=(f"{p}nut_housing", f"{p}nut_stage", f"{p}nut_socket"),
        actuators=(f"{p}nut_clutch",),
        ctrl_open=(0.0,),
        reach=g.reach,
        tcp=(0.0, 0.0, ctx.mount_z - g.reach),
        actuation={
            "torque_cap_nm": g.torque_cap,
            "rated_torque_nm": g.rated_torque,
            "safety_factor": RATED_SAFETY_FACTOR,
            "min_member_mm": MIN_MEMBER_MM,
            "across_flats_mm": 1000.0 * g.across_flats,
            "socket_depth_mm": 1000.0 * g.socket_depth,
            "funnel_flare_mm": 1000.0 * FUNNEL_FLARE,
            "float_range_mm": 1000.0 * FLOAT_RANGE,
            "float_stiffness_n_per_mm": FLOAT_STIFFNESS / 1000.0,
        },
        notes=(
            NUT_RUNNER_CAVEAT,
            (
                f"{1000.0 * g.across_flats:g} mm hex socket, {1000.0 * g.socket_depth:g} mm deep with a "
                f"{1000.0 * FUNNEL_FLARE:g} mm guide funnel; clutch cap {g.torque_cap:g} N m (rated "
                f"{g.rated_torque:.1f} N m at a safety factor of {RATED_SAFETY_FACTOR:g}); the Studio "
                f"turns the socket with the wrist roll, at the wrist's speed; members at least "
                f"{MIN_MEMBER_MM:g} mm tall"
            ),
        ),
        max_opening=None,
    )


def clutch_weight(closure: float) -> float:
    """How firmly the clutch holds the socket to the housing at a close-ramp value: 0 below
    :data:`CLUTCH_FROM`, 1 at the end of the ramp, smooth between."""
    x = min(1.0, max(0.0, (closure - CLUTCH_FROM) / (1.0 - CLUTCH_FROM)))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class SocketPrimitive:
    """``drive``: lower the socket over the top of the member and hold it to the housing.

    The one control is the clutch's torque (N m), by the close ramp (``closure``):

    * **held** (:func:`clutch_weight` 1): a spring and damper to the socket's neutral
      (:data:`SPINDLE_KP`, :data:`SPINDLE_KV`), limited by the actuator to the torque cap;
    * **let go with the member in the socket** (the ramp falling at the release): a damper on the
      socket's turning rate relative to the member and a spring that turns it to the middle of its
      clearance on the member's flats (:data:`SPINDLE_FOLLOW`, :data:`SPINDLE_ALIGN`);
    * **clear of the member** (the rim past every object geom): centred to the housing, softly
      (:data:`SPINDLE_CENTRE_KP`), so the flats are square to the tool for the next engagement.

    The Studio's twist primitive turns the wrist roll; the clutch passes its torque up to the cap and
    slips beyond it.
    """

    geometry: NutRunnerGeometry
    name: str = "drive"
    _body: tuple[int, int] = (-1, -1)  # (id(model), the socket's body id)

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        tip = max(task.controller.min_tip_clearance, top_height(part) - self.geometry.socket_depth)
        return PrimitivePlan(tip_height=tip, note="socket lowered over the top of the member")

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        j = model.joint("eff_nut_spindle")
        q, w = float(data.qpos[j.qposadr[0]]), float(data.qvel[j.dofadr[0]])
        hold = clutch_weight(closure)
        held = -(SPINDLE_KP * q + SPINDLE_KV * w)
        if hold >= 1.0:
            return np.array([held])
        if self._clear(model, data, scene):
            free = -(SPINDLE_CENTRE_KP * q + SPINDLE_CENTRE_KV * w)
        else:
            free = self._follow_member(model, data, scene)
        return np.array([hold * held + (1.0 - hold) * free])

    def _socket(self, model: mujoco.MjModel) -> int:
        if self._body[0] != id(model):
            self._body = (id(model), int(model.body("eff_nut_socket").id))
        return self._body[1]

    def _clear(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        """Whether the socket's rim is past every object geom (along the tool axis): nothing is in the
        socket, so it may turn back to its neutral without dragging a member."""
        geoms = scene.ids.object_geoms
        if not len(geoms):
            return True
        body = self._socket(model)
        axis = -data.xmat[body].reshape(3, 3)[:, 2]  # down the tool axis
        rim = data.xpos[body] + self.geometry.rim_from_stage * axis
        beyond = (data.geom_xpos[geoms] - rim) @ axis  # each object geom's centre past the rim
        return bool(np.all(beyond - model.geom_rbound[geoms] >= CLEAR_MARGIN))

    def _follow_member(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> float:
        """The torque of a socket let go with the member in it (the class docstring)."""
        body = self._socket(model)
        member = int(scene.ids.object_body)
        rot = data.xmat[body].reshape(3, 3)
        axis = rot[:, 2]
        torque = -SPINDLE_FOLLOW * float((data.cvel[body][:3] - data.cvel[member][:3]) @ axis)
        if member > 0:
            x_s, x_m = rot[:, 0], data.xmat[member].reshape(3, 3)[:, 0]
            turn = math.atan2(
                float(axis @ np.cross(x_m, x_s)), float(x_m @ x_s)
            )  # member's x to the socket's
            torque -= SPINDLE_ALIGN * ((turn % (math.pi / 3.0)) - math.pi / 6.0)  # 0 on the flats' middle
        return torque


@dataclass
class SocketCriterion:
    """Engaged when the member stands in the socket: at least four of its six walls, an opposite
    pair among them, are within :attr:`reach_m` of the part (form closure on the flats)."""

    geometry: NutRunnerGeometry
    name: str = "socket"
    reach_m: float = 0.001

    def _walls(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> tuple[int, ...]:
        walls = [model.geom(f"eff_nut_wall_{k}_1").id for k in range(6)]
        parts = [int(g) for g in scene.ids.object_geoms]
        fromto = np.zeros(6)
        near = []
        for k, w in enumerate(walls):
            for g in parts:
                if mujoco.mj_geomDistance(model, data, w, g, self.reach_m, fromto) < self.reach_m:
                    near.append(k)
                    break
        return tuple(near)

    def _state(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> tuple[bool, dict[str, float]]:
        near = self._walls(model, data, scene)
        opposite = any(k in near and (k + 3) % 6 in near for k in range(3))
        metrics = {"walls_near": float(len(near)), "torque_cap_nm": self.geometry.torque_cap}
        return len(near) >= 4 and opposite, metrics

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        return self._state(model, data, scene)[0]

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement:
        ok, metrics = self._state(model, data, scene)
        return Engagement(ok, () if ok else ("few_contacts",), metrics)


# --------------------------------------------------------------------------- tamper


@dataclass(frozen=True)
class TamperGeometry:
    """Every dimension of a tamper (m, kg, N), from its params."""

    pad: float  # side of the square pad
    pad_thickness: float
    stage_stiffness: float  # N/m
    force_cap: float  # N: the stage's overload force
    stage_stroke: float  # m: the stage's whole travel

    KEYS: ClassVar[tuple[str, ...]] = (
        "pad_mm",
        "pad_thickness_mm",
        "stage_stiffness_n_per_mm",
        "press_force_cap_n",
        "stage_stroke_mm",
    )
    housing_radius: ClassVar[float] = 0.028
    housing_height: ClassVar[float] = 0.030
    post_radius: ClassVar[float] = 0.008
    pad_density: ClassVar[float] = 900.0  # kg/m^3: a foam-faced composite pad [assumed]

    @staticmethod
    def from_params(params: Any) -> TamperGeometry:
        p = dict(params)
        _check_keys("tamper", p, TamperGeometry.KEYS)
        g = TamperGeometry(
            pad=_positive("pad_mm", p["pad_mm"], 60.0, 400.0, "mm") / 1000.0,
            pad_thickness=_positive("pad_thickness_mm", p["pad_thickness_mm"], 5.0, 30.0, "mm") / 1000.0,
            stage_stiffness=1000.0
            * _positive("stage_stiffness_n_per_mm", p["stage_stiffness_n_per_mm"], 0.5, 20.0, "N/mm"),
            force_cap=_positive("press_force_cap_n", p["press_force_cap_n"], 5.0, 150.0, "N"),
            stage_stroke=_positive("stage_stroke_mm", p["stage_stroke_mm"], 8.0, 100.0, "mm") / 1000.0,
        )
        if g.stage_stroke < g.spring_travel + 0.005:
            raise ValueError(
                f"stage_stroke_mm must be at least {1000.0 * (g.spring_travel + 0.005):g} mm: the spring "
                f"takes {1000.0 * g.spring_travel:g} mm to reach the overload force and the stage needs 5 mm "
                "of slip beyond it"
            )
        return g

    @property
    def pad_mass(self) -> float:
        return self.pad_density * self.pad * self.pad * self.pad_thickness

    @property
    def spring_travel(self) -> float:
        """m of compression at which the spring reaches the overload force."""
        return self.force_cap / self.stage_stiffness

    @property
    def stage_damping(self) -> float:
        """N s/m: damping of the stage (0.7 of critical for the pad and its post on the spring)."""
        return 2.0 * 0.7 * math.sqrt(self.stage_stiffness * (self.pad_mass + 0.05))

    @property
    def max_force(self) -> float:
        """N: the most the pad can press with, the stage's overload force."""
        return self.force_cap

    @property
    def rated_force(self) -> float:
        return self.max_force / RATED_SAFETY_FACTOR

    @property
    def reach(self) -> float:
        """m from the mount plane to the pad's face with the stage extended."""
        return self.housing_height + self.stage_stroke + self.pad_thickness


def tamper_spec(pad_mm: float = 200.0, **params: Any) -> EffectorSpec:
    """A tamper spec (``origin = "preset"``): a square pad of ``pad_mm`` on a 4 N/mm stage that
    slips at 60 N (60 mm of travel); the stiffness, the cap and the stroke are all [assumed]."""
    p = {
        "pad_mm": float(pad_mm),
        "pad_thickness_mm": 12.0,  # [assumed]
        "stage_stiffness_n_per_mm": 4.0,  # [assumed]
        "press_force_cap_n": 60.0,  # [assumed]
        "stage_stroke_mm": 60.0,  # [assumed]
        **params,
    }
    TamperGeometry.from_params(p)  # validate
    return EffectorSpec(
        family=EffectorFamily.PROCESS_TOOL,
        model=f"tamper-pad-{pad_mm:g}".replace(".", "p"),
        display_name=f"Tamper pad, {pad_mm:g} mm",
        primitive="press",
        params=p,
        origin="preset",
    )


def _build_tamper(ctx: BuildContext, g: TamperGeometry) -> EffectorBuild:
    p = ctx.prefix
    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
    grey, pad_rgba = [0.34, 0.36, 0.40, 1.0], [0.16, 0.17, 0.19, 1.0]
    housing = ctx.mount.add_body(name=f"{p}tamper_housing", pos=[0.0, 0.0, ctx.mount_z])
    housing.add_geom(
        name=f"{p}tamper_housing",
        type=cyl,
        size=[g.housing_radius, 0.5 * g.housing_height, 0.0],
        pos=[0.0, 0.0, -0.5 * g.housing_height],
        mass=0.25,
        rgba=grey,
    )
    pad = housing.add_body(name=f"{p}tamper_pad", pos=[0.0, 0.0, -g.housing_height])
    pad.add_joint(
        name=f"{p}tamper_stage",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, 0.0, 1.0],  # + = compressed (the pad pushed up toward the mount)
        range=[0.0, g.stage_stroke],
        armature=0.002,
    )
    gap = g.stage_stroke
    pad.add_geom(
        name=f"{p}tamper_post",
        type=cyl,
        size=[g.post_radius, 0.5 * gap, 0.0],
        pos=[0.0, 0.0, -0.5 * gap],
        mass=0.05,
        rgba=grey,
    )
    pad.add_geom(
        name=f"{p}tamper_pad",
        type=box,
        size=[0.5 * g.pad, 0.5 * g.pad, 0.5 * g.pad_thickness],
        pos=[0.0, 0.0, -gap - 0.5 * g.pad_thickness],
        mass=g.pad_mass,
        rgba=pad_rgba,
    )
    stage = ctx.spec.add_actuator(
        name=f"{p}tamper_spring", target=f"{p}tamper_stage", trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    stage.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    stage.gainprm[0] = 1.0  # the control is the force (N): the primitive's spring law
    stage.biastype = mujoco.mjtBias.mjBIAS_NONE
    stage.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    stage.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    stage.forcerange = [-g.force_cap, g.force_cap]
    return EffectorBuild(
        spec=ctx.effector,
        bodies=(f"{p}tamper_housing", f"{p}tamper_pad"),
        actuators=(f"{p}tamper_spring",),
        ctrl_open=(0.0,),
        reach=g.reach,
        tcp=(0.0, 0.0, ctx.mount_z - g.reach),
        actuation={
            "press_force_max_n": g.max_force,
            "rated_press_force_n": g.rated_force,
            "safety_factor": RATED_SAFETY_FACTOR,
            "pad_mm": 1000.0 * g.pad,
            "pad_area_cm2": 1e4 * g.pad * g.pad,
            "stage_stiffness_n_per_mm": g.stage_stiffness / 1000.0,
            "spring_travel_mm": 1000.0 * g.spring_travel,
            "stage_stroke_mm": 1000.0 * g.stage_stroke,
        },
        notes=(
            TAMPER_CAVEAT,
            (
                f"{1000.0 * g.pad:g} mm square pad on a {g.stage_stiffness / 1000.0:g} N/mm spring stage "
                f"that slips at {g.force_cap:g} N, so the pad never presses harder; "
                f"{1000.0 * g.stage_stroke:g} mm of travel in all (rated {g.rated_force:.0f} N at a safety "
                f"factor of {RATED_SAFETY_FACTOR:g})"
            ),
        ),
        max_opening=None,
    )


@dataclass
class PadPrimitive:
    """``press``: bring the pad down on the part's top. The one control is the stage's force: the
    spring and damper law ``-min(stiffness x compression + damping x speed, force cap)``, so the stage
    is a spring that slips at the cap and the pad never presses harder than the cap; the Studio's
    push primitive presses with the wrist."""

    geometry: TamperGeometry
    name: str = "press"

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        tip = max(task.controller.min_tip_clearance, top_height(part))
        return PrimitivePlan(tip_height=tip, note="pad face on the part's top")

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        j = model.joint("eff_tamper_stage")
        q, v = float(data.qpos[j.qposadr[0]]), float(data.qvel[j.dofadr[0]])
        g = self.geometry
        return np.array([-min(max(g.stage_stiffness * q + g.stage_damping * v, 0.0), g.force_cap)])


@dataclass
class PadCriterion:
    """Engaged when the pad presses on the part with at least :attr:`min_force` N."""

    geometry: TamperGeometry
    name: str = "pad"
    min_force: float = 1.0

    def _state(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> tuple[bool, dict[str, float]]:
        pad = model.geom("eff_tamper_pad").id
        contacts = [c for c in object_contacts(model, data, scene) if c.geom == pad]
        force = sum(c.normal_force for c in contacts)
        stage = float(data.qpos[model.joint("eff_tamper_stage").qposadr[0]])
        metrics = {
            "press_force_n": force,
            "stage_compression_mm": 1000.0 * stage,
            "press_force_max_n": self.geometry.max_force,
        }
        return force >= self.min_force, metrics

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        return self._state(model, data, scene)[0]

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement:
        ok, metrics = self._state(model, data, scene)
        return Engagement(ok, () if ok else ("few_contacts",), metrics)


# --------------------------------------------------------------------------- the family


class ProcessToolFamily(EffectorFamilyImpl):
    """The process-tool family (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.PROCESS_TOOL
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"
    caveat: ClassVar[str] = FIDELITY_CAVEAT

    def presets(self) -> tuple[EffectorSpec, ...]:
        return (nut_runner_spec(13.0), tamper_spec(200.0))

    def _kind(self, spec: EffectorSpec) -> Literal["nut_runner", "tamper"]:
        if spec.model.startswith("nut-runner-") and spec.primitive == "drive":
            return "nut_runner"
        if spec.model.startswith("tamper-pad-") and spec.primitive == "press":
            return "tamper"
        raise ValueError(f"not a process tool: {spec.model}/{spec.primitive}")

    def build(self, ctx: BuildContext) -> EffectorBuild:
        self.check_spec(ctx.effector)
        if self._kind(ctx.effector) == "nut_runner":
            return _build_nut_runner(ctx, NutRunnerGeometry.from_params(ctx.effector.params))
        return _build_tamper(ctx, TamperGeometry.from_params(ctx.effector.params))

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> SocketPrimitive | PadPrimitive:
        if self._kind(spec) == "nut_runner":
            return SocketPrimitive(NutRunnerGeometry.from_params(spec.params))
        return PadPrimitive(TamperGeometry.from_params(spec.params))

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> SocketCriterion | PadCriterion:
        if self._kind(spec) == "nut_runner":
            return SocketCriterion(NutRunnerGeometry.from_params(spec.params))
        return PadCriterion(TamperGeometry.from_params(spec.params))


register_family(ProcessToolFamily())
