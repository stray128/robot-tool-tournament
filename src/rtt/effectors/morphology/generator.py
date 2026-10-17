"""Write a :class:`~rtt.effectors.morphology.layout.HandLayout` into an ``MjSpec`` subtree.

:func:`add_hand` adds the palm plate under a parent body (the wrist in the shared scene, a
fixed world body in the validators' sweep model), every digit as a chain of bodies with one
joint per link, the pads and hooks, the tendons and one position servo per joint or per
tendon. Every name starts with the prefix. It returns the names in the order the contract
needs (bodies root first, actuators in compile order).

Contact bookkeeping (allowed by the contract: excludes pair effector bodies and the wrist
only): links of the same digit never collide with each other or with the palm plate (a
curling digit would otherwise rub its own palm), and the effector never collides with the
wrist it is bolted to. Different digits do collide, so fingers that cross stop each other.

Servos: a hinge servo saturates at the declared joint torque within 0.1 rad of its target, a
slide servo at the declared finger force within 5 mm; both are critically damped for the link
inertia plus a fixed armature, which keeps every gain well inside the 1 ms step's stability
limit. Targets are the joint limits, so a digit blocked by the part squeezes it with the full
declared force, and a digit that meets nothing stops at its limit with no force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from rtt.effectors.morphology.layout import PAD_DENSITY, Digit, HandLayout, Link

__all__ = ["HandNames", "add_hand", "body_name", "joint_name"]

_HINGE_ARMATURE = 0.001  # kg m^2
_HINGE_DAMPING = 0.01  # N m s/rad
_HINGE_SATURATION = 0.1  # rad of error that saturates a hinge servo
_SLIDE_ARMATURE = 0.05  # kg
_SLIDE_DAMPING = 1.0  # N s/m
_SLIDE_SATURATION = 0.005  # m
_PALM_RGBA = [0.32, 0.36, 0.42, 1.0]
_LINK_RGBA = [0.86, 0.55, 0.22, 1.0]
_PAD_RGBA = [0.15, 0.15, 0.16, 1.0]
_MIN_HALF = 1e-4  # m: shorter capsules become spheres


@dataclass(frozen=True)
class HandNames:
    """Names of what :func:`add_hand` created (all prefixed)."""

    palm: str
    bodies: tuple[str, ...]  # root first
    joints: tuple[tuple[str, ...], ...]  # per digit, per link
    actuators: tuple[str, ...]  # control order
    digit_bodies: tuple[tuple[str, ...], ...]  # per digit, per link


def body_name(prefix: str, digit: int, link: int) -> str:
    return f"{prefix}d{digit}_l{link}"


def joint_name(prefix: str, digit: int, link: int) -> str:
    return f"{prefix}d{digit}_j{link}"


def _quat_z(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


_QUAT_Z_TO_Y = [math.cos(-math.pi / 4.0), math.sin(-math.pi / 4.0), 0.0, 0.0]  # capsule axis along +y


def _add_link_geoms(body: mujoco.MjsBody, name: str, link: Link, density: float) -> None:
    box = mujoco.mjtGeom.mjGEOM_BOX
    if link.shape == "capsule":
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[link.radius, link.length / 2.0, 0.0],
            pos=[0.0, 0.0, -link.length / 2.0],
            density=density,
            rgba=_LINK_RGBA,
        )
        return
    t, w, L = link.thickness, link.width, link.length
    body.add_geom(
        name=name,
        type=box,
        size=[t / 2.0, w / 2.0, L / 2.0],
        pos=[0.0, 0.0, -L / 2.0],
        density=density,
        rgba=_LINK_RGBA,
    )
    p = link.pad_thickness
    if link.pad == "flat" and p > 0.0:
        body.add_geom(
            name=f"{name}_pad",
            type=box,
            size=[p / 2.0, 0.45 * w, link.pad_length / 2.0],
            pos=[-t / 2.0 - p / 2.0, 0.0, link.pad_center_z],
            density=PAD_DENSITY,
            rgba=_PAD_RGBA,
        )
    elif link.pad == "round" and p > 0.0:
        half = w / 2.0 - p
        if half > _MIN_HALF:  # a bump across the finger
            shape, size = mujoco.mjtGeom.mjGEOM_CAPSULE, [p, half, 0.0]
        else:  # as wide as the finger: a single dome
            shape, size = mujoco.mjtGeom.mjGEOM_SPHERE, [p, 0.0, 0.0]
        body.add_geom(
            name=f"{name}_pad",
            type=shape,
            size=size,
            pos=[-t / 2.0, 0.0, link.pad_center_z],
            quat=_QUAT_Z_TO_Y,
            density=PAD_DENSITY,
            rgba=_PAD_RGBA,
        )
    if link.hook_depth > 0.0:
        span = p + link.hook_depth
        body.add_geom(
            name=f"{name}_hook",
            type=box,
            size=[span / 2.0, w / 2.0, link.hook_height / 2.0],
            pos=[-t / 2.0 - span / 2.0, 0.0, -L + link.hook_height / 2.0],
            density=density,
            rgba=_LINK_RGBA,
        )


def _link_inertia(digit: Digit, j: int, density: float) -> float:
    """Rough moment of inertia (kg m^2) of the links from joint ``j`` out, about joint ``j``."""
    total, depth = 0.0, 0.0
    for link in digit.links[j:]:
        body, pad = link.volumes()
        m = body * density + pad * PAD_DENSITY
        mid = depth + link.length / 2.0
        total += m * (mid * mid + link.length**2 / 12.0)
        depth += link.length
    return total


def _link_mass(digit: Digit, density: float) -> float:
    return sum(link.volumes()[0] * density + link.volumes()[1] * PAD_DENSITY for link in digit.links)


def _position_servo(
    spec: mujoco.MjSpec,
    name: str,
    target: str,
    trntype: mujoco.mjtTrn,
    force: float,
    saturation: float,
    inertia: float,
    ctrlrange: tuple[float, float],
) -> None:
    kp = force / saturation
    kv = 2.0 * math.sqrt(kp * inertia)
    act = spec.add_actuator(name=name, target=target, trntype=trntype)
    act.set_to_position(kp=kp, kv=kv)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    lo, hi = ctrlrange
    act.ctrlrange = [min(lo, hi), max(lo, hi)]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-force, force]


def add_hand(
    spec: mujoco.MjSpec,
    parent: mujoco.MjsBody,
    layout: HandLayout,
    *,
    prefix: str,
    z0: float,
    wrist: str | None = None,
) -> HandNames:
    """Add the effector under ``parent`` with its mount plane at ``z0`` (parent frame).

    ``wrist`` names the body the effector is bolted to (contacts with it are excluded).
    The hand frame is turned by ``layout.yaw`` only where the caller turns ``parent``; in
    the shared scene the wrist itself is yawed by the primitive's ``yaw_offset``.
    """
    p = prefix
    hinge, slide = mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
    palm = parent.add_body(name=f"{p}palm", pos=[0.0, 0.0, z0])
    palm.add_geom(
        name=f"{p}palm",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[layout.palm_plate_radius, layout.palm_thickness / 2.0, 0.0],
        pos=[0.0, 0.0, -layout.palm_thickness / 2.0],
        density=layout.density,
        rgba=_PALM_RGBA,
    )
    bodies = [f"{p}palm"]
    joints: list[tuple[str, ...]] = []
    digit_bodies: list[tuple[str, ...]] = []
    for d in layout.digits:
        c, s = math.cos(d.angle), math.sin(d.angle)
        node = palm
        names, jnames = [], []
        for j, link in enumerate(d.links):
            pos = (
                [d.radius * c, d.radius * s, -layout.palm_thickness]
                if j == 0
                else [0.0, 0.0, -d.links[j - 1].length]
            )
            body = node.add_body(
                name=body_name(p, d.index, j), pos=pos, quat=_quat_z(d.angle) if j == 0 else [1, 0, 0, 0]
            )
            jn = joint_name(p, d.index, j)
            if d.joint == "slide":
                body.add_joint(
                    name=jn,
                    type=slide,
                    axis=[-1.0, 0.0, 0.0],
                    range=[d.lower[j], d.upper[j]],
                    armature=_SLIDE_ARMATURE,
                    damping=_SLIDE_DAMPING,
                )
            else:
                body.add_joint(
                    name=jn,
                    type=hinge,
                    axis=[0.0, 1.0, 0.0],
                    range=[d.lower[j], d.upper[j]],
                    armature=_HINGE_ARMATURE,
                    damping=_HINGE_DAMPING,
                    stiffness=d.stiffness[j],
                    springref=0.0,
                )
            _add_link_geoms(body, body_name(p, d.index, j), link, layout.density)
            names.append(body.name)
            jnames.append(jn)
            node = body
        bodies.extend(names)
        joints.append(tuple(jnames))
        digit_bodies.append(tuple(names))

    # contact excludes: inside a digit, digit-palm, and the effector-wrist mount
    k = 0

    def exclude(a: str, b: str) -> None:
        nonlocal k
        spec.add_exclude(name=f"{p}x{k}", bodyname1=a, bodyname2=b)
        k += 1

    for names in digit_bodies:
        for i, a in enumerate(names):
            for b in names[i + 2 :]:
                exclude(a, b)
            if i > 0:
                exclude(f"{p}palm", a)
    if wrist is not None:
        for b in bodies:
            exclude(wrist, b)

    actuators = []
    trn_joint, trn_tendon = mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_TENDON
    for d, jnames in zip(layout.digits, joints, strict=True):
        if layout.actuation == "tendon":
            # a cable with moment arm c_j (m) at joint j: tension T gives c_j T on the joint
            coefs = layout.tendon_coefs[d.index]
            tname = f"{p}d{d.index}_tendon"
            tendon = spec.add_tendon(name=tname)
            for jn, coef in zip(jnames, coefs, strict=True):
                if coef:
                    tendon.wrap_joint(jn, coef)
            inv = sum(
                coef * coef / (_HINGE_ARMATURE + _link_inertia(d, j, layout.density))
                for j, coef in enumerate(coefs)
                if coef
            )
            lo = sum(c * q for c, q in zip(coefs, d.lower, strict=True))
            hi = sum(c * q for c, q in zip(coefs, d.upper, strict=True))
            _position_servo(
                spec,
                tname,
                tname,
                trn_tendon,
                layout.digit_force,
                _HINGE_SATURATION * coefs[0],
                1.0 / inv,
                (lo, hi),
            )
            actuators.append(tname)
            continue
        for j, jn in enumerate(jnames):
            if d.joint == "slide":
                inertia = _SLIDE_ARMATURE + _link_mass(d, layout.density)
                saturation = _SLIDE_SATURATION
            else:
                inertia = _HINGE_ARMATURE + _link_inertia(d, j, layout.density)
                saturation = _HINGE_SATURATION
            _position_servo(
                spec, jn, jn, trn_joint, d.torque[j], saturation, inertia, (d.lower[j], d.upper[j])
            )
            actuators.append(jn)
    return HandNames(
        palm=f"{p}palm",
        bodies=tuple(bodies),
        joints=tuple(joints),
        actuators=tuple(actuators),
        digit_bodies=tuple(digit_bodies),
    )


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData, bodies: np.ndarray) -> float:
    """World z of the lowest point of the collision geoms of ``bodies`` (boxes, capsules, cylinders)."""
    lowest = math.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies:
            continue
        size = model.geom_size[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g]
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            ext = float(np.abs(rot[2, :]) @ size)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = abs(float(rot[2, 2])) * size[1] + size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ext = abs(float(rot[2, 2])) * size[1] + size[0] * math.sqrt(max(0.0, 1.0 - float(rot[2, 2]) ** 2))
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = size[0]
        else:
            continue
        lowest = min(lowest, float(centre[2]) - ext)
    return lowest
