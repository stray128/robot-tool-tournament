"""The shared effector scene: the core world and wrist, one family's effector, one part.

Everything outside the effector is the pre-registered v2 scene, built by the frozen code
(:func:`rtt.sim.scene._base_spec`, :func:`rtt.sim.scene._add_object`,
:func:`rtt.sim.scene._add_position_actuator`; ``src/rtt/sim`` is hash-pinned by PREREG, so
these private helpers cannot change under us):

* the task's solver options, contact defaults, lights, table and floor;
* the floating wrist ``palm``: slides ``palm_x``/``palm_y``/``palm_z`` and the ``palm_yaw``
  hinge, gravity compensated, driven by position servos with the task's gains and force limits
  (x/y 100 N, z 150 N, yaw 15 N m in ``task_v2``), identical for every family;
* the part as a free body whose geoms have ``priority = 1``, so the part's (jittered) friction
  and contact parameters govern every contact it makes, whatever the effector's own defaults.

The only change to the wrist is its geometry: the jaw's 48 x 128 x 24 mm gripper-frame box
becomes ``wrist_flange``, a 63 mm diameter disc of the same 24 mm thickness and the same
mass (an ISO 9409-1-50 style flange), so a cup or a tentacle is not judged with a parallel
gripper's housing on top of it. Its underside is the mount plane, wrist z = -12 mm
(:data:`MOUNT_Z`), the plane of the jaw's finger roots.

The family adds its subtree under the wrist (:class:`~rtt.effectors.types.BuildContext`). The
scene then enforces the contract: every new element is named with the prefix, every new body
hangs under the wrist, the declared actuators are exactly the new ones in compile order, and
every effector body is gravity compensated (the arm carries the tool's weight; the part is
not compensated). Mass properties are measured from the compiled model.

Geom groups follow :mod:`rtt.sim.scene`: world geoms ``GROUP_ENV``, the wrist ``GROUP_PALM``,
every effector geom ``GROUP_FINGER_LEFT`` and the part ``GROUP_OBJECT``. With these labels the
frozen :class:`rtt.sim.checks.PhysicsChecks` runs unchanged: effector-table force and
penetration, part speed, ejection, off-table and wrist-part contact during the carry are
flagged exactly as for the jaws, and effector self-contact is not a penetration flag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from rtt.effectors.types import NAME_PREFIX, BuildContext, ContractError, EffectorBuild, EffectorSpec
from rtt.sim.scene import (
    BODY_OBJECT,
    BODY_PALM,
    GROUP_ENV,
    GROUP_FINGER_LEFT,
    GROUP_OBJECT,
    GROUP_OTHER,
    GROUP_PALM,
    PALM_JOINTS,
    PALM_RGBA,
    PALM_XY_LIMIT,
    YAW_LIMIT,
    _add_object,
    _add_position_actuator,
    _base_spec,
)

if TYPE_CHECKING:
    from rtt.effectors.registry import EffectorFamilyImpl
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

__all__ = [
    "FLANGE_GEOM",
    "FLANGE_RADIUS",
    "MOUNT_Z",
    "REPO_ROOT",
    "WRIST_BODY",
    "EffectorScene",
    "EffectorSceneIds",
    "build_effector_scene",
    "build_effector_spec",
]

WRIST_BODY = BODY_PALM
FLANGE_GEOM = "wrist_flange"
FLANGE_RADIUS = 0.0315  # m; 63 mm disc
MOUNT_Z = -0.012  # m; wrist-frame z of the mount plane (rtt.design.freeform.types.ROOT_PLANE_PALM_Z)
REPO_ROOT = Path(__file__).resolve().parents[3]

_ELEMENT_LISTS = ("bodies", "joints", "geoms", "sites", "actuators", "tendons", "equalities", "sensors")


@dataclass(frozen=True)
class EffectorSceneIds:
    """Indices into a compiled effector scene.

    It carries the attributes the frozen helpers read from :class:`rtt.sim.scene.SceneIds`
    (``geom_group``, ``object_body``, ``object_dof``, ``object_geoms``, ``palm_body``,
    ``replay_bodies``), so :class:`~rtt.sim.checks.PhysicsChecks`,
    :func:`~rtt.sim.register.part_pose_in_palm` and the replay recorder accept it.
    ``replay_bodies`` are the wrist, every effector body with a collision geom, then the part,
    named by ``replay_names``.
    """

    palm_body: int
    object_body: int
    palm_qpos: np.ndarray
    object_qpos: int
    object_dof: int
    palm_actuators: np.ndarray
    effector_actuators: np.ndarray
    object_geom: int
    object_geoms: np.ndarray
    effector_bodies: np.ndarray
    effector_geoms: np.ndarray
    geom_group: np.ndarray
    replay_bodies: np.ndarray
    replay_names: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class EffectorScene:
    """A compiled scene for one (effector, part, task), reusable across that pair's episodes."""

    model: mujoco.MjModel
    ids: EffectorSceneIds
    build: EffectorBuild
    part_id: str


def _names(spec: mujoco.MjSpec) -> dict[str, set[str]]:
    return {kind: {e.name for e in getattr(spec, kind)} for kind in _ELEMENT_LISTS}


def _add_wrist(spec: mujoco.MjSpec, palm: mujoco.MjsBody, task: TaskV2) -> None:
    """Swap the jaw's palm box for the flange and add the four wrist servos (core gains)."""
    g, phys = task.gripper, task.physics
    spec.delete(spec.geom(BODY_PALM))
    palm.add_geom(
        name=FLANGE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[FLANGE_RADIUS, -MOUNT_Z, 0.0],
        mass=g.palm_mass,
        rgba=list(PALM_RGBA),
    )
    xy = (-PALM_XY_LIMIT, PALM_XY_LIMIT)
    _add_position_actuator(spec, "palm_x", "palm_x", g.xy_kp, g.xy_kv, g.xy_force, xy)
    _add_position_actuator(spec, "palm_y", "palm_y", g.xy_kp, g.xy_kv, g.xy_force, xy)
    _add_position_actuator(
        spec, "palm_z", "palm_z", g.z_kp, g.z_kv, g.z_force, (phys.table_top_z - 0.05, phys.table_top_z + 0.8)
    )
    _add_position_actuator(
        spec, "palm_yaw", "palm_yaw", g.yaw_kp, g.yaw_kv, g.yaw_torque, (-YAW_LIMIT, YAW_LIMIT)
    )


_OPTION_FIELDS = (
    "timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "gravity",
    "wind", "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
    "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations",
    "ccd_iterations", "disableflags", "enableflags", "disableactuator", "sdf_initpoints", "sdf_iterations",
)  # fmt: skip
_COMPILER_FIELDS = (
    "autolimits", "boundmass", "boundinertia", "settotalmass", "balanceinertia", "fitaabb", "degree",
    "eulerseq", "discardvisual", "fusestatic", "inertiafromgeom", "inertiagrouprange", "alignfree",
)  # fmt: skip
_GEOM_FIELDS = (
    "condim", "contype", "conaffinity", "priority", "friction", "solmix", "solref", "solimp", "margin",
    "gap", "adhesion", "surfacevel", "density",
)  # fmt: skip
_JOINT_FIELDS = (
    "stiffness", "damping", "armature", "frictionloss", "springref", "solref_limit", "solimp_limit",
    "solref_friction", "solimp_friction", "margin",
)  # fmt: skip


def _snapshot(obj: Any, fields: tuple[str, ...]) -> dict[str, str]:
    """The named settings of a spec section, as text (missing fields are skipped)."""
    out = {}
    for name in fields:
        try:
            value = getattr(obj, name)
        except (AttributeError, ValueError):
            continue
        if isinstance(value, np.ndarray):
            out[name] = repr(value.tolist())
        elif isinstance(value, bool | int | float | str):
            out[name] = repr(value)
        else:
            try:
                out[name] = repr(int(value))  # pybind enums
            except (TypeError, ValueError):
                out[name] = repr(list(value)) if hasattr(value, "__iter__") else repr(value)
    return out


def _settings(spec: mujoco.MjSpec) -> dict[str, dict[str, str]]:
    """The scene-wide settings a family must leave alone (solver, compiler, main defaults)."""
    return {
        "option": _snapshot(spec.option, _OPTION_FIELDS),
        "compiler": _snapshot(spec.compiler, _COMPILER_FIELDS),
        "default geom": _snapshot(spec.default.geom, _GEOM_FIELDS),
        "default joint": _snapshot(spec.default.joint, _JOINT_FIELDS),
    }


def _is_descendant(body: mujoco.MjsBody, ancestor: mujoco.MjsBody) -> bool:
    node = body
    while node is not None and node.name != "world":
        if node.name == ancestor.name:
            return True
        node = node.parent
    return False


def build_effector_spec(
    impl: EffectorFamilyImpl, effector: EffectorSpec, part: PartSpec, task: TaskV2
) -> tuple[mujoco.MjSpec, EffectorBuild]:
    """Author (not compile) the scene and check the family's additions (module docstring)."""
    impl.check_spec(effector)
    spec, palm = _base_spec(task)
    spec.modelname = "rtt_effector"
    _add_wrist(spec, palm, task)
    before = _names(spec)
    settings = _settings(spec)
    n_pairs = len(spec.pairs)
    ctx = BuildContext(
        spec=spec, mount=palm, mount_z=MOUNT_Z, effector=effector, part=part, task=task, repo_root=REPO_ROOT
    )
    build = impl.build(ctx)
    if not isinstance(build, EffectorBuild):
        raise ContractError(f"{type(impl).__name__}.build must return an EffectorBuild")
    if build.spec is not effector and build.spec.effector_id != effector.effector_id:
        raise ContractError("the build describes a different effector spec")
    after = _names(spec)
    problems: list[str] = []
    for section, values in _settings(spec).items():
        changed = sorted(k for k, v in values.items() if settings[section].get(k) != v)
        if changed:
            problems.append(f"the family changed the scene's {section} settings: {changed[:5]}")
    if len(spec.pairs) != n_pairs:
        problems.append("explicit contact pairs are not allowed (the part's priority rule governs contacts)")
    for kind in _ELEMENT_LISTS:
        new = after[kind] - before[kind]
        bad = sorted(n for n in new if n and not n.startswith(NAME_PREFIX))
        if bad:
            problems.append(f"new {kind} without the {NAME_PREFIX!r} prefix: {bad[:5]}")
        lost = sorted(before[kind] - after[kind])
        if lost:
            problems.append(f"the family removed scene {kind}: {lost[:5]}")
    new_bodies = [b for b in spec.bodies if b.name in after["bodies"] - before["bodies"]]
    if not new_bodies:
        problems.append("the family added no bodies")
    for b in new_bodies:
        if not _is_descendant(b, palm):
            problems.append(f"body {b.name!r} is not mounted under the wrist")
        b.gravcomp = 1.0
    missing = sorted(set(build.bodies) - {b.name for b in new_bodies})
    if missing:
        problems.append(f"declared bodies that were not added: {missing[:5]}")
    new_acts = [a.name for a in spec.actuators if a.name in after["actuators"] - before["actuators"]]
    if tuple(new_acts) != build.actuators:
        problems.append(f"declared actuators {list(build.actuators)} != added actuators {new_acts}")
    if problems:
        raise ContractError(f"{type(impl).__name__} broke the scene contract: " + "; ".join(problems))
    obj = part.to_mesh_object()
    _add_object(spec, obj, part.density)
    return spec, build


def _check_compiled(model: mujoco.MjModel, ids: EffectorSceneIds) -> list[str]:
    """What the compiled model must satisfy: the effector acts only on itself.

    Effector actuators, equality constraints and tendons may only reference effector bodies
    (no actuator on the wrist, no weld to the part or the world, no tendon anchored in the
    world); effector geoms may not out-rank the part's contact parameters (``priority``), stick
    by themselves (geom ``adhesion``; suction uses an adhesion *actuator*), reach across a gap
    (``margin > 0``: MuJoCo takes the larger margin of the two geoms whatever their priority, so
    an effector margin would make contacts, and an adhesion actuator's pull, act before the
    surfaces touch) or move their surface (``surfacevel``); contact excludes may only pair
    effector bodies and the wrist.
    """
    eff = {int(b) for b in ids.effector_bodies}
    tendons = {
        t
        for t in range(model.ntendon)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, t) or "").startswith(NAME_PREFIX)
    }
    trn, eq, wrap = mujoco.mjtTrn, mujoco.mjtEq, mujoco.mjtWrap
    problems = []

    def name(objtype: mujoco.mjtObj, i: int) -> str:
        return mujoco.mj_id2name(model, objtype, int(i)) or f"#{i}"

    for a in ids.effector_actuators:
        kind, (i, j) = int(model.actuator_trntype[a]), model.actuator_trnid[a]
        if kind in (trn.mjTRN_JOINT, trn.mjTRN_JOINTINPARENT):
            ok = int(model.jnt_bodyid[i]) in eff
        elif kind == trn.mjTRN_TENDON:
            ok = int(i) in tendons
        elif kind in (trn.mjTRN_SITE, trn.mjTRN_SLIDERCRANK):
            ok = int(model.site_bodyid[i]) in eff and (j < 0 or int(model.site_bodyid[j]) in eff)
        elif kind == trn.mjTRN_BODY:
            ok = int(i) in eff
        else:
            ok = False
        if not ok:
            problems.append(f"actuator {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)!r} acts outside the effector")
    for e in range(model.neq):
        kind, o1, o2 = int(model.eq_type[e]), int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if kind in (eq.mjEQ_CONNECT, eq.mjEQ_WELD):
            site = int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE)
            bodies = [int(model.site_bodyid[o]) if site else o for o in (o1, o2)]
            ok = all(b in eff for b in bodies)
        elif kind == eq.mjEQ_JOINT:
            ok = int(model.jnt_bodyid[o1]) in eff and (o2 < 0 or int(model.jnt_bodyid[o2]) in eff)
        elif kind == eq.mjEQ_TENDON:
            ok = o1 in tendons and (o2 < 0 or o2 in tendons)
        else:
            ok = False
        if not ok:
            problems.append(
                f"equality {name(mujoco.mjtObj.mjOBJ_EQUALITY, e)!r} ties the effector to the scene"
            )
    for t in range(model.ntendon):
        if t not in tendons:
            problems.append(f"tendon {name(mujoco.mjtObj.mjOBJ_TENDON, t)!r} is not the effector's")
            continue
        for w in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t]):
            kind, i = int(model.wrap_type[w]), int(model.wrap_objid[w])
            if kind == wrap.mjWRAP_JOINT:
                body = int(model.jnt_bodyid[i])
            elif kind == wrap.mjWRAP_SITE:
                body = int(model.site_bodyid[i])
            elif kind in (wrap.mjWRAP_SPHERE, wrap.mjWRAP_CYLINDER):
                body = int(model.geom_bodyid[i])
            else:
                continue
            if body not in eff:
                problems.append(f"tendon {name(mujoco.mjtObj.mjOBJ_TENDON, t)!r} wraps outside the effector")
                break
    geoms = np.flatnonzero(np.isin(model.geom_bodyid, ids.effector_bodies))
    if np.any(model.geom_priority[geoms] >= 1):
        problems.append("effector geoms may not have priority >= 1 (the part's contact parameters govern)")
    if np.any(model.geom_adhesion[geoms] != 0):
        problems.append("effector geoms may not use geom adhesion (suction uses an adhesion actuator)")
    if np.any(model.geom_margin[geoms] > 0):
        problems.append("effector geoms may not have a contact margin > 0 (contacts would act across a gap)")
    if np.any(model.geom_surfacevel[geoms] != 0):
        problems.append("effector geoms may not have a surface velocity")
    if model.npair:
        problems.append("explicit contact pairs are not allowed")
    allowed = eff | {int(ids.palm_body)}
    for sig in model.exclude_signature:
        b1, b2 = int(sig) >> 16, int(sig) & 0xFFFF
        if b1 not in allowed or b2 not in allowed:
            problems.append(
                f"a contact exclude pairs {name(mujoco.mjtObj.mjOBJ_BODY, b1)!r} and {name(mujoco.mjtObj.mjOBJ_BODY, b2)!r}"
            )
    return problems


def _mass_properties(model: mujoco.MjModel, bodies: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Total mass, centre of mass and inertia about it (wrist frame, reference pose)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    origin = data.xpos[model.body(WRIST_BODY).id].copy()
    masses = model.body_mass[bodies]
    total = float(masses.sum())
    com = (masses[:, None] * (data.xipos[bodies] - origin)).sum(axis=0) / total
    inertia = np.zeros((3, 3))
    for b, m in zip(bodies, masses, strict=True):
        rot = data.ximat[b].reshape(3, 3)
        i_b = rot @ np.diag(model.body_inertia[b]) @ rot.T
        d = data.xipos[b] - origin - com
        inertia += i_b + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return total, com, inertia


def _resolve(model: mujoco.MjModel, build: EffectorBuild) -> EffectorSceneIds:
    def jid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    palm = model.body(WRIST_BODY).id
    obj = model.body(BODY_OBJECT).id
    eff_bodies = np.array(
        [
            b
            for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(NAME_PREFIX)
        ],
        dtype=np.int64,
    )
    is_eff = np.isin(model.geom_bodyid, eff_bodies)
    group = np.full(model.ngeom, GROUP_OTHER, dtype=np.int8)
    group[model.geom_bodyid == 0] = GROUP_ENV
    group[model.geom_bodyid == palm] = GROUP_PALM
    group[is_eff] = GROUP_FINGER_LEFT
    group[model.geom_bodyid == obj] = GROUP_OBJECT
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    drawn = [b for b in eff_bodies if np.any(collides & (model.geom_bodyid == b))]
    replay = [palm, *drawn, obj]
    free = jid("object_free")
    object_geoms = np.flatnonzero(model.geom_bodyid == obj)
    acts = np.array([model.actuator(n).id for n in build.actuators], dtype=np.int64)
    return EffectorSceneIds(
        palm_body=palm,
        object_body=obj,
        palm_qpos=np.array([model.jnt_qposadr[jid(n)] for n in PALM_JOINTS]),
        object_qpos=int(model.jnt_qposadr[free]),
        object_dof=int(model.jnt_dofadr[free]),
        palm_actuators=np.array([model.actuator(n).id for n in PALM_JOINTS], dtype=np.int64),
        effector_actuators=acts,
        object_geom=int(object_geoms[0]),
        object_geoms=object_geoms,
        effector_bodies=eff_bodies,
        effector_geoms=np.flatnonzero(is_eff & collides),
        geom_group=group,
        replay_bodies=np.array(replay, dtype=np.int64),
        replay_names=tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in replay),
    )


def build_effector_scene(
    impl: EffectorFamilyImpl,
    effector: EffectorSpec,
    part: PartSpec,
    task: TaskV2,
    *,
    edit: Callable[[mujoco.MjSpec], Any] | None = None,
) -> EffectorScene:
    """Author, check and compile the scene; measure the effector's mass properties.

    ``edit`` (tests and diagnostics only) may change the authored spec before compiling.
    The wrist actuators come first in the control vector (x, y, z, yaw), then the effector's
    in ``build.actuators`` order.
    """
    spec, build = build_effector_spec(impl, effector, part, task)
    if edit is not None:
        edit(spec)
    model = spec.compile()
    ids = _resolve(model, build)
    if list(ids.palm_actuators) != [0, 1, 2, 3] or list(ids.effector_actuators) != list(
        range(4, 4 + build.n_actuators)
    ):
        raise ContractError("the compiled actuator order is not wrist (x, y, z, yaw) then the effector's")
    if model.nu != 4 + build.n_actuators:
        raise ContractError(
            f"the scene has {model.nu} actuators; expected 4 wrist + {build.n_actuators} effector"
        )
    problems = _check_compiled(model, ids)
    if problems:
        raise ContractError(f"{type(impl).__name__} broke the scene contract: " + "; ".join(problems))
    mass, com, inertia = _mass_properties(model, ids.effector_bodies)
    build = replace(
        build,
        mass=mass,
        com=tuple(float(v) for v in com),
        inertia6=tuple(float(inertia[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))),  # type: ignore[arg-type]
    )
    return EffectorScene(model=model, ids=ids, build=build, part_id=part.name)
