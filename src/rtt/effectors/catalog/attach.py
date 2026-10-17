"""Mounting a catalog model on the shared wrist, and the open-pose probe that measures it.

:func:`mount_rotation` turns two root-frame axes into the mount rotation: ``down`` to the
wrist's -z (palm-down) and ``forward`` to the wrist's +y (the parts' closing axis).

:func:`probe` compiles the effector alone under a fixed mount (no gravity, the scene's
timestep and integrator), drives its actuators to the open controls for 0.6 s and measures
the open pose: the highest and lowest collision points (exact per geom type), the opening
between the two finger sides of a gripper, and the grasp centre. The levelling offset puts
the highest point on the mount plane; ``reach`` is then the mount plane to the lowest point.
The probe never sees the part, so nothing about the mount depends on it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase

import mujoco
import numpy as np

from rtt.effectors.scene import MOUNT_Z

__all__ = [
    "OPTION_FIELDS",
    "OpenPose",
    "actuator_limits",
    "controls",
    "geom_extent",
    "match_options",
    "mount_rotation",
    "probe",
]

#: Solver options copied from the scene onto a child spec before attaching (the scene's values
#: win anyway; copying them first keeps MuJoCo from reporting a conflict on every build).
OPTION_FIELDS: tuple[str, ...] = (
    "timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "gravity",
    "wind", "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
    "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations",
    "ccd_iterations", "disableflags", "enableflags", "disableactuator", "sdf_initpoints", "sdf_iterations",
)  # fmt: skip

PROBE_TIME = 0.6  # s of settling at the open controls
PROBE_DT = 0.001


def match_options(child: mujoco.MjSpec, parent: mujoco.MjSpec) -> None:
    """Give ``child`` the parent's solver options (module constant :data:`OPTION_FIELDS`)."""
    for name in OPTION_FIELDS:
        try:
            setattr(child.option, name, getattr(parent.option, name))
        except (AttributeError, TypeError, ValueError):
            continue


def mount_rotation(down: Sequence[float], forward: Sequence[float]) -> np.ndarray:
    """Unit quaternion (wxyz) taking ``down`` to -z and ``forward`` (made orthogonal) to +y."""
    u = np.asarray(down, dtype=np.float64)
    u = u / np.linalg.norm(u)
    v = np.asarray(forward, dtype=np.float64)
    v = v - (v @ u) * u
    if np.linalg.norm(v) < 1e-9:
        raise ValueError("forward must not be parallel to down")
    v = v / np.linalg.norm(v)
    w = np.cross(v, u)
    src = np.column_stack([u, v, w])
    dst = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rot = dst @ src.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.ravel())
    return q / np.linalg.norm(q)


def geom_extent(model: mujoco.MjModel, data: mujoco.MjData, g: int, axis: np.ndarray) -> tuple[float, float]:
    """Exact extent (lo, hi) of geom ``g`` along the unit world vector ``axis``."""
    c = float(data.geom_xpos[g] @ axis)
    rot = data.geom_xmat[g].reshape(3, 3)
    a = rot.T @ axis  # the axis in the geom frame
    size = model.geom_size[g]
    gtype = int(model.geom_type[g])
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r = float(size[0] + size[1] * abs(a[2]))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r = float(size[1] * abs(a[2]) + size[0] * math.sqrt(max(0.0, 1.0 - a[2] ** 2)))
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        r = float(np.abs(a) @ size)
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        r = float(math.sqrt(float(((size * a) ** 2).sum())))
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        adr, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        proj = model.mesh_vert[adr : adr + n].astype(np.float64) @ a
        return c + float(proj.min()), c + float(proj.max())
    else:
        raise ValueError(f"unsupported geom type {mujoco.mjtGeom(gtype).name}")
    return c - r, c + r


def controls(
    names: Sequence[str],
    values: Mapping[str, float],
    ctrlrange: np.ndarray,
    rest: float = 0.0,
) -> tuple[float, ...]:
    """A control vector over ``names`` (upstream actuator names): the first pattern in
    ``values`` that matches each name (exact names before glob patterns), else ``rest``;
    clipped to each actuator's ``ctrlrange``."""
    exact = {k: v for k, v in values.items() if not any(ch in k for ch in "*?[")}
    globs = [(k, v) for k, v in values.items() if k not in exact]
    out = []
    for name, (lo, hi) in zip(names, ctrlrange, strict=True):
        if name in exact:
            v = exact[name]
        else:
            v = next((val for pat, val in globs if fnmatchcase(name, pat)), rest)
        out.append(float(min(max(v, lo), hi)))
    return tuple(out)


def actuator_limits(model: mujoco.MjModel, acts: Sequence[int]) -> tuple[float, ...]:
    """Each actuator's peak output: gain x control span, capped by its force range and by the
    driven joint's actuator force range (N for slides and tendons, N m for hinges)."""
    out = []
    for a in acts:
        lo, hi = model.actuator_ctrlrange[a]
        span = float(hi - lo) if model.actuator_ctrllimited[a] else math.inf
        peak = abs(float(model.actuator_gainprm[a, 0])) * span
        if model.actuator_forcelimited[a]:
            flo, fhi = model.actuator_forcerange[a]
            peak = min(peak, max(abs(flo), abs(fhi)))
        if int(model.actuator_trntype[a]) == mujoco.mjtTrn.mjTRN_JOINT:
            j = int(model.actuator_trnid[a, 0])
            if model.jnt_actfrclimited[j]:
                flo, fhi = model.jnt_actfrcrange[j]
                peak = min(peak, max(abs(flo), abs(fhi)))
        out.append(float(peak))
    return tuple(out)


@dataclass(frozen=True)
class OpenPose:
    """What the probe measured (mount frame = wrist frame; metres)."""

    top: float  # highest collision point before levelling
    bottom: float  # lowest collision point before levelling
    centre_xy: tuple[float, float]  # grasp centre: mean of the centre bodies' collision geom centres
    opening: float | None  # gap between the two finger sides along +y (grippers)
    footprint: tuple[float, float, float, float]  # x lo, x hi, y lo, y hi of every collision geom

    def level(self) -> float:
        """The z shift that puts the highest point on the mount plane."""
        return MOUNT_Z - self.top

    @property
    def reach(self) -> float:
        return self.top - self.bottom


def probe(
    attach: Callable[[mujoco.MjSpec, mujoco.MjsBody], None],
    ctrl: Callable[[mujoco.MjModel, list[int]], Sequence[float]],
    *,
    centre_bodies: Sequence[str] = (),
    finger_sides: tuple[Sequence[str], Sequence[str]] | None = None,
) -> OpenPose:
    """Measure the open pose of one effector (module docstring).

    ``attach(spec, mount)`` adds the effector under ``mount``; ``ctrl(model, actuator_ids)``
    returns the open controls. ``centre_bodies`` (attached names) set the grasp centre: the mean,
    over the names (a repeated name weighs more), of each body's collision geom centres (all geoms
    when empty); ``finger_sides`` (grippers) names the bodies of the finger on
    one side and of the other (sorted by their mean y), whose inner faces bound the opening.
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = PROBE_DT
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mount = spec.worldbody.add_body(name="probe_mount")
    # A joint keeps the mount from being welded to the world, as the wrist is in the scene (MuJoCo
    # filters parent-child contacts through welded bodies, but never for bodies welded to the world).
    mount.add_joint(name="probe_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0.0, 0.0, 1.0], armature=1e6)
    attach(spec, mount)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec.compile()
    data = mujoco.MjData(model)
    acts = list(range(model.nu))
    data.ctrl[:] = np.asarray(ctrl(model, acts), dtype=np.float64)
    mujoco.mj_forward(model, data)
    for _ in range(round(PROBE_TIME / PROBE_DT)):
        mujoco.mj_step(model, data)
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    geoms = np.flatnonzero(collides & (model.geom_bodyid != 0) & (model.geom_bodyid != mount.id))
    z = np.array([0.0, 0.0, 1.0])
    ext = np.array([geom_extent(model, data, int(g), z) for g in geoms])
    xs = np.array([geom_extent(model, data, int(g), np.array([1.0, 0.0, 0.0])) for g in geoms])
    ys = np.array([geom_extent(model, data, int(g), np.array([0.0, 1.0, 0.0])) for g in geoms])
    footprint = (float(xs[:, 0].min()), float(xs[:, 1].max()), float(ys[:, 0].min()), float(ys[:, 1].max()))

    def geoms_of(names: Sequence[str]) -> np.ndarray:
        ids = [model.body(n).id for n in names]
        return geoms[np.isin(model.geom_bodyid[geoms], ids)]

    if centre_bodies:  # the mean over names (a repeated name weighs more) of each body's geom centre
        centre_xy = np.mean([data.geom_xpos[geoms_of([n]), :2].mean(axis=0) for n in centre_bodies], axis=0)
    else:
        centre_xy = data.geom_xpos[geoms, :2].mean(axis=0)
    opening = None
    if finger_sides is not None:
        y = np.array([0.0, 1.0, 0.0])
        plus, minus = (geoms_of(side) for side in finger_sides)
        if data.geom_xpos[plus, 1].mean() < data.geom_xpos[minus, 1].mean():
            plus, minus = minus, plus
        inner_plus = min(geom_extent(model, data, int(g), y)[0] for g in plus)
        inner_minus = max(geom_extent(model, data, int(g), y)[1] for g in minus)
        opening = float(inner_plus - inner_minus)
    return OpenPose(
        top=float(ext[:, 1].max()),
        bottom=float(ext[:, 0].min()),
        centre_xy=(float(centre_xy[0]), float(centre_xy[1])),
        opening=opening,
        footprint=footprint,
    )
