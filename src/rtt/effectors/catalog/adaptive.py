"""Procedural adaptive grippers: two or three underactuated fingers (our own geometry, Apache-2.0).

No permissive open model of a three-finger adaptive gripper exists whose maker can be credited
here (``docs/EFFECTORS.md`` section 9), so these are built from primitives:

* a 16 mm thick base plate under the mount plane;
* fingers of two phalanges (proximal and distal boxes) on hinges whose axes lie along the
  wrist's x, so every finger closes along the parts' y axis. Two fingers face each other; the
  three-finger version puts one finger on the +y side and a pair on the -y side (each pair
  finger at half the tendon tension, so both sides squeeze alike);
* one fixed tendon per finger over both joints (moment arms ``r1`` at the proximal joint, ``r2``
  at the distal one), driven by one position actuator with a force limit (the tendon tension):
  one actuator moves two joints;
* a stiff return spring on the distal joint and a soft one on the proximal joint: in free space
  the finger closes as one straight link; once the proximal phalanx is stopped by the part the
  tendon keeps pulling and the distal phalanx wraps around it (the adaptive behaviour).

Actuation limits (printed on every card): tendon tension ``tension_n`` per single finger,
joint torques ``r1 x tension`` and ``r2 x tension``. With the defaults (40 N, 15 and 10 mm) a
finger presses about 24 N on a part touching the middle of its proximal phalanx, close to the
parallel jaw's 25 N per jaw.

Primitives: ``power`` (the fingertips go low, the proximal phalanges meet the part and the
distal ones wrap) and ``pinch`` (the fingertips meet the part's upper half). One grip-height
fraction each, the same for every part.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mujoco
import numpy as np

from rtt.effectors.catalog.attach import OpenPose, probe
from rtt.effectors.catalog.family import OPENING_MARGIN
from rtt.effectors.primitives import ContactCriterion, SynergyPrimitive
from rtt.effectors.types import (
    NAME_PREFIX,
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    InvalidEffector,
)

__all__ = [
    "ADAPTIVE_MODELS",
    "AdaptiveDesign",
    "adaptive_criterion",
    "adaptive_primitive",
    "adaptive_spec",
    "adaptive_specs",
    "build_adaptive",
]

BASE_HALF_THICKNESS = 0.008  # m
PHALANX_MASS = 0.03  # kg per phalanx
BASE_MASS = 0.25  # kg
Q_OPEN = -0.15  # rad: fingers splayed slightly outward when open
Q1_MAX = 1.4  # rad: proximal flexion limit
Q2_MAX = 1.6  # rad: distal flexion limit


@dataclass(frozen=True)
class AdaptiveDesign:
    """Geometry and actuation of one procedural adaptive gripper (SI units)."""

    key: str
    display_name: str
    n_fingers: int  # 2 or 3
    finger_width: float = 0.018
    thickness: float = 0.008  # phalanx thickness along the closing axis
    proximal: float = 0.050
    distal: float = 0.035
    half_span: float = 0.030  # hinge distance from the wrist axis along y
    pair_offset: float = 0.0095  # three fingers: x offset of each finger of the -y pair (1 mm gap)
    r1: float = 0.0375  # tendon moment arm at the proximal joint (m): 1.5 N m at 40 N
    r2: float = 0.010  # tendon moment arm at the distal joint (m)
    tension_n: float = 40.0  # tendon tension limit of a single finger (N)
    k1: float = 0.05  # proximal return spring (N m/rad)
    k2: float = 0.6  # distal return spring (N m/rad)
    grip: Mapping[str, float] | None = None  # primitive -> grip-height fraction

    def fingers(self) -> list[tuple[str, float, float, float, float]]:
        """(name, x, y, closing sign along y, tension) of every finger."""
        if self.n_fingers == 2:
            return [
                ("f0", 0.0, -self.half_span, 1.0, self.tension_n),
                ("f1", 0.0, self.half_span, -1.0, self.tension_n),
            ]
        if self.n_fingers == 3:
            half = 0.5 * self.tension_n
            return [
                ("f0", -self.pair_offset, -self.half_span, 1.0, half),
                ("f1", self.pair_offset, -self.half_span, 1.0, half),
                ("f2", 0.0, self.half_span, -1.0, self.tension_n),
            ]
        raise ValueError("adaptive grippers have 2 or 3 fingers")

    def as_params(self) -> dict[str, Any]:
        return {
            "source": "procedural",
            "n_fingers": self.n_fingers,
            "finger_width": self.finger_width,
            "thickness": self.thickness,
            "proximal": self.proximal,
            "distal": self.distal,
            "half_span": self.half_span,
            "pair_offset": self.pair_offset,
            "r1": self.r1,
            "r2": self.r2,
            "tension_n": self.tension_n,
            "k1": self.k1,
            "k2": self.k2,
        }

    def tendon_length(self, q1: float, q2: float) -> float:
        return self.r1 * q1 + self.r2 * q2


ADAPTIVE_MODELS: Mapping[str, AdaptiveDesign] = {
    "adaptive-2f": AdaptiveDesign(
        key="adaptive-2f",
        display_name="Two-finger adaptive gripper",
        n_fingers=2,
        grip={"power": 0.2, "pinch": 0.35},
    ),
    "adaptive-3f": AdaptiveDesign(
        key="adaptive-3f",
        display_name="Three-finger adaptive gripper",
        n_fingers=3,
        grip={"power": 0.2, "pinch": 0.35},
    ),
}


def adaptive_spec(design: AdaptiveDesign, primitive: str) -> EffectorSpec:
    grip = (design.grip or {})[primitive]
    label = {"power": "power grasp", "pinch": "pinch"}[primitive]
    return EffectorSpec(
        family=EffectorFamily.CATALOG_GRIPPER,
        model=design.key,
        display_name=f"{design.display_name}, {label}",
        primitive=primitive,
        params={**design.as_params(), "grip_height_frac": grip},
        origin="catalog",
    )


def adaptive_specs() -> tuple[EffectorSpec, ...]:
    return tuple(adaptive_spec(d, p) for d in ADAPTIVE_MODELS.values() for p in ("power", "pinch"))


def _design_of(spec: EffectorSpec) -> AdaptiveDesign:
    """The design a spec names; its params must be that design's (no per-part variants)."""
    design = ADAPTIVE_MODELS[spec.model]
    want = adaptive_spec(design, spec.primitive)
    if spec.effector_id != want.effector_id:
        raise ValueError(f"{spec.model}/{spec.primitive}: params differ from the procedural design")
    return design


def _add(
    spec: mujoco.MjSpec, mount: mujoco.MjsBody, d: AdaptiveDesign, z0: float
) -> tuple[list[str], list[str]]:
    """Author the gripper under ``mount``; returns (bodies, actuators) in compile order."""
    p = NAME_PREFIX
    bodies, actuators = [], []
    xs = [abs(x) for _, x, _, _, _ in d.fingers()]
    base = mount.add_body(name=f"{p}base", pos=[0.0, 0.0, z0])
    base.add_geom(
        name=f"{p}base",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[max(xs) + 0.5 * d.finger_width + 0.004, d.half_span + 0.008, BASE_HALF_THICKNESS],
        pos=[0.0, 0.0, -BASE_HALF_THICKNESS],
        mass=BASE_MASS,
        rgba=[0.35, 0.37, 0.4, 1.0],
    )
    bodies.append(f"{p}base")
    hw, ht = 0.5 * d.finger_width, 0.5 * d.thickness
    for name, x, y, sign, tension in d.fingers():
        axis = [sign, 0.0, 0.0]  # +q swings the tip toward the wrist axis (along +y for sign +1)
        prox = base.add_body(name=f"{p}{name}_proximal", pos=[x, y, -2.0 * BASE_HALF_THICKNESS])
        prox.add_joint(
            name=f"{p}{name}_j1",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis,
            range=[Q_OPEN - 0.05, Q1_MAX],
            stiffness=d.k1,
            damping=0.05,
            armature=0.001,
        )
        prox.add_geom(
            name=f"{p}{name}_proximal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hw, ht, 0.5 * d.proximal],
            pos=[0.0, 0.0, -0.5 * d.proximal],
            mass=PHALANX_MASS,
            rgba=[0.2, 0.45, 0.75, 1.0],
        )
        dist = prox.add_body(name=f"{p}{name}_distal", pos=[0.0, 0.0, -d.proximal])
        dist.add_joint(
            name=f"{p}{name}_j2",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis,
            range=[0.0, Q2_MAX],
            stiffness=d.k2,
            damping=0.02,
            armature=0.001,
        )
        dist.add_geom(
            name=f"{p}{name}_distal",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[hw, ht, 0.5 * d.distal],
            pos=[0.0, 0.0, -0.5 * d.distal],
            mass=PHALANX_MASS,
            rgba=[0.25, 0.55, 0.85, 1.0],
        )
        tendon = spec.add_tendon(name=f"{p}{name}_tendon")
        tendon.wrap_joint(f"{p}{name}_j1", d.r1)
        tendon.wrap_joint(f"{p}{name}_j2", d.r2)
        act = spec.add_actuator(
            name=f"{p}{name}", target=f"{p}{name}_tendon", trntype=mujoco.mjtTrn.mjTRN_TENDON
        )
        act.set_to_position(kp=4000.0, dampratio=1.0)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [d.tendon_length(Q_OPEN, 0.0), d.tendon_length(Q1_MAX, Q2_MAX)]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-tension, tension]
        bodies += [f"{p}{name}_proximal", f"{p}{name}_distal"]
        actuators.append(f"{p}{name}")
    return bodies, actuators


@lru_cache(maxsize=16)
def _open_pose(key: str) -> OpenPose:
    d = ADAPTIVE_MODELS[key]
    names: dict[str, list[str]] = {}

    def attach(spec: mujoco.MjSpec, mount: mujoco.MjsBody) -> None:
        names["bodies"], names["acts"] = _add(spec, mount, d, 0.0)

    def ctrl(model: mujoco.MjModel, acts: list[int]) -> tuple[float, ...]:
        return tuple(d.tendon_length(Q_OPEN, 0.0) for _ in acts)

    plus = tuple(
        f"{NAME_PREFIX}{n}_{seg}" for n, _, _, s, _ in d.fingers() if s < 0 for seg in ("proximal", "distal")
    )
    minus = tuple(
        f"{NAME_PREFIX}{n}_{seg}" for n, _, _, s, _ in d.fingers() if s > 0 for seg in ("proximal", "distal")
    )
    return probe(attach, ctrl, finger_sides=(plus, minus))


def build_adaptive(ctx: BuildContext) -> EffectorBuild:
    """Author the procedural gripper of ``ctx.effector`` under the mount (module docstring)."""
    d = _design_of(ctx.effector)
    pose = _open_pose(d.key)
    assert pose.opening is not None
    width = float(ctx.part.size[1])
    if width > pose.opening - OPENING_MARGIN:
        raise InvalidEffector(
            f"the part is {1000 * width:.1f} mm across the closing axis; the gripper opens "
            f"{1000 * pose.opening:.1f} mm"
        )
    bodies, acts = _add(ctx.spec, ctx.mount, d, ctx.mount_z)
    open_l = d.tendon_length(Q_OPEN, 0.0)
    return EffectorBuild(
        spec=ctx.effector,
        bodies=tuple(bodies),
        actuators=tuple(acts),
        ctrl_open=tuple(open_l for _ in acts),
        reach=pose.reach,
        tcp=(0.0, 0.0, ctx.mount_z - pose.reach + 0.010),
        actuation={
            "n_actuators": len(acts),
            "tendon_tension_n_max": d.tension_n,
            "joint_torque_nm_max": round(d.r1 * d.tension_n, 4),
            "opening_mm": round(1000.0 * pose.opening, 2),
        },
        notes=(
            (
                "procedural adaptive gripper (our own geometry, Apache-2.0): two phalanges per finger "
                "coupled by one tendon and one actuator; a stiff distal spring keeps the finger straight "
                "until the proximal phalanx meets the part, then the distal phalanx wraps"
            ),
            f"{d.n_fingers} fingers; tendon tension limit {d.tension_n:g} N per single finger"
            + (" (half for each finger of the pair)" if d.n_fingers == 3 else ""),
        ),
        max_opening=pose.opening,
    )


def adaptive_primitive(spec: EffectorSpec, build: EffectorBuild) -> SynergyPrimitive:
    d = _design_of(spec)
    closed = d.tendon_length(Q1_MAX, Q2_MAX)
    n = d.n_fingers
    return SynergyPrimitive(
        name=spec.primitive,
        ctrl_open=tuple(d.tendon_length(Q_OPEN, 0.0) for _ in range(n)),
        ctrl_closed=tuple(closed for _ in range(n)),
        grip_height_frac=float(spec.params["grip_height_frac"]),
    )


def adaptive_criterion(spec: EffectorSpec) -> ContactCriterion:
    return ContactCriterion(min_bodies=2)


def joint_coupling(model: mujoco.MjModel, actuator: int) -> int:
    """How many joints one actuator drives (through its tendon)."""
    if int(model.actuator_trntype[actuator]) != mujoco.mjtTrn.mjTRN_TENDON:
        return 1
    t = int(model.actuator_trnid[actuator, 0])
    adr, num = int(model.tendon_adr[t]), int(model.tendon_num[t])
    return int(np.sum(model.wrap_type[adr : adr + num] == mujoco.mjtWrap.mjWRAP_JOINT))
