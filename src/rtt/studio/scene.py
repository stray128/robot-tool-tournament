"""The Studio scene: the core world and wrist, a continuous wrist roll, one setup's tools, the objects.

Everything outside the tools is built from the frozen simulator, as in the v1 effector scene
(:mod:`rtt.effectors.scene`): the task's solver options, contact defaults, lights, table and floor
(:func:`rtt.sim.scene._base_spec`) and the four core wrist servos (x, y, z, yaw) with the
``task_v2`` gains and limits (:func:`rtt.sim.scene._add_position_actuator`). The frozen v2 and v1
scenes are untouched; this is the Studio's own builder.

Kinematic chain (names in :data:`WRIST_BODIES`)::

    world -> palm (slides x, y, z and the yaw hinge: frozen core wrist, 0.4 kg, no geometry)
          -> wrist_roll (continuous hinge about the tool axis, STUDIO_WRIST limits)
          -> wrist_flange (the 63 mm x 24 mm flange geom; the tools hang below it)

* **Wrist roll.** A hinge about the palm's z axis through the wrist centre, unlimited
  (continuous), driven by a position servo whose torque saturates at
  ``STUDIO_WRIST.roll_torque_nm``; the control is unlimited so a target can wind several turns.
  Every candidate gets the same roll (:mod:`rtt.studio.fairness` asserts it).
* **single:** the family builds under the flange exactly as in the v1 scene (mount plane at
  wrist z = -12 mm, :data:`~rtt.effectors.scene.MOUNT_Z`), names ``eff_*``.
* **tool_change:** the changer's master and tool plates (``eff_studio_changer``,
  ``ToolChanger.stack_mm`` tall, their masses) sit under the flange and the step's tool builds
  under them. A segment's scene holds one tool; the next tool is a new scene (the mass swap).
* **multi_head:** a generated bracket (``eff_studio_bracket``: a hub under the flange and one
  capsule arm per head, :class:`~rtt.studio.types.BracketSpec` density) carries both heads, each
  on its own adapter plate at its :class:`~rtt.studio.types.HeadMount` pose. Each head is built by
  its family into a scratch spec (the same options and defaults, no visual assets) and attached
  under its mount: the **active** head keeps the family's own names (``eff_*``), so its v1
  primitive and criterion find their elements; the **idle** head is renamed ``eff_h<k>_*``. The
  idle head's bodies, mass and collision geoms are all simulated. To present the active head the
  flange and bracket are tilted, as one rigid assembly about the wrist centre, by
  ``-angle_deg`` of that head (a fixed rotation per segment, between the roll and the flange), so
  the active head points straight down on the roll axis. Which head is active changes only the
  names and the presentation, never the bracket.

Objects are added by their template builders (:mod:`rtt.studio.templates`), each element named
``obj_<instance>_``. The scene then checks the contract on the compiled model (who may act on
what, priorities, margins, excludes) and measures every head's mass properties.

Geom groups follow :mod:`rtt.sim.scene`: world and static geoms ``GROUP_ENV``, the flange
``GROUP_PALM``, every tool-side geom ``GROUP_FINGER_LEFT``, moving object geoms ``GROUP_OBJECT``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import mujoco
import numpy as np

from rtt.effectors.registry import get_family
from rtt.effectors.scene import (
    _ELEMENT_LISTS,
    FLANGE_GEOM,
    FLANGE_RADIUS,
    MOUNT_Z,
    REPO_ROOT,
    _is_descendant,
    _mass_properties,
    _names,
    _settings,
)
from rtt.effectors.types import NAME_PREFIX, BuildContext, ContractError, EffectorBuild, EffectorSpec
from rtt.sim.scene import (
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
    _add_position_actuator,
    _base_spec,
)
from rtt.sim.task import TaskV2, default_task_v2, task_digest
from rtt.studio.templates import (
    OBJ_PREFIX,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    get_template_builder,
    object_prefix,
)
from rtt.studio.types import (
    STUDIO_WRIST,
    PrimitiveKind,
    Setup,
    SetupKind,
    TaskSpec,
    TemplateKind,
    WristLimits,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = [
    "FLANGE_BODY",
    "ROLL_ACTUATOR",
    "ROLL_JOINT",
    "STUDIO_PREFIX",
    "WRIST_BODIES",
    "WRIST_JOINTS",
    "HeadBuild",
    "StudioScene",
    "StudioSceneIds",
    "build_part_for",
    "build_studio_scene",
    "idle_prefix",
    "present_quat",
]

ROLL_JOINT: Final = "wrist_roll"
ROLL_ACTUATOR: Final = "wrist_roll"
ROLL_BODY: Final = "wrist_roll"
FLANGE_BODY: Final = "wrist_flange"
WRIST_BODIES: Final[tuple[str, ...]] = (BODY_PALM, ROLL_BODY, FLANGE_BODY)
#: The control vector starts with these five servos, in this order.
WRIST_JOINTS: Final[tuple[str, ...]] = (*PALM_JOINTS, ROLL_JOINT)
#: Tool-stack elements the Studio itself adds (bracket, adapters, changer plates).
STUDIO_PREFIX: Final = f"{NAME_PREFIX}studio_"
_MOUNT = f"{STUDIO_PREFIX}mount"  # a multi-tool head's adapter body (one per head)


def idle_prefix(tool: int) -> str:
    """Name prefix of an idle head of a multi-tool head (``eff_h<k>_``)."""
    return f"{NAME_PREFIX}h{tool}_"


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _quat(r: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).ravel())
    return [float(v) for v in q]


def present_quat(setup: Setup, tool: int) -> list[float]:
    """Flange orientation in the roll frame that presents ``tool`` (identity unless multi_head)."""
    if setup.kind is not SetupKind.MULTI_HEAD:
        return [1.0, 0.0, 0.0, 0.0]
    return _quat(_rot_x(-setup.mounts[tool].angle_deg))


@dataclass(frozen=True, eq=False)
class HeadBuild:
    """One tool in the scene: its family build, names and where it sits.

    * ``build``: what the family returned (names without ``prefix``), with ``mass``, ``com``
      and ``inertia6`` measured from this scene (wrist frame, roll 0).
    * ``prefix``: ``""`` for the active head (and every single or tool-change tool), else
      ``eff_h<k>_``; :meth:`name` prefixes a family name.
    * ``mount_body``: the body the family built under.
    * ``tcp_wrist``/``axis_wrist``: the grasp centre and the tool axis (pointing out of the
      tool) in the wrist (palm) frame, at roll 0.
    """

    tool: int
    spec: EffectorSpec
    build: EffectorBuild
    prefix: str
    active: bool
    mount_body: str
    tcp_wrist: tuple[float, float, float]
    axis_wrist: tuple[float, float, float]

    def name(self, family_name: str) -> str:
        return f"{self.prefix}{family_name}"

    @property
    def actuators(self) -> tuple[str, ...]:
        return tuple(self.name(a) for a in self.build.actuators)


@dataclass(frozen=True)
class StudioSceneIds:
    """Indices into a compiled Studio scene."""

    palm_body: int
    roll_body: int
    flange_body: int
    palm_qpos: np.ndarray  # x, y, z, yaw
    roll_qpos: int
    roll_dof: int
    wrist_actuators: np.ndarray  # x, y, z, yaw, roll (always 0..4)
    effector_actuators: np.ndarray  # the active head's, in its build order
    idle_actuators: np.ndarray  # every idle head's
    object_actuators: np.ndarray  # the templates' passive actuators
    effector_bodies: np.ndarray  # every tool-side body (heads, bracket, adapters, changer)
    active_bodies: np.ndarray
    idle_bodies: np.ndarray
    stack_bodies: np.ndarray  # bracket, adapters, changer plates
    effector_geoms: np.ndarray  # colliding geoms of every tool-side body
    active_geoms: np.ndarray
    idle_geoms: np.ndarray
    object_geoms: np.ndarray  # every template geom
    geom_group: np.ndarray


@dataclass(frozen=True, eq=False)
class StudioScene:
    """A compiled scene for one (setup, active tool, task spec, task), reusable across episodes."""

    model: mujoco.MjModel
    ids: StudioSceneIds
    setup: Setup
    task_spec: TaskSpec
    tool: int
    heads: tuple[HeadBuild, ...]
    objects: Mapping[str, ResolvedHandles]
    builders: Mapping[str, TemplateBuilder]
    wrist: WristLimits
    stack_mass_kg: float
    tool_mass_kg: float
    task_digest: str
    object_friction: np.ndarray  # nominal sliding friction of ids.object_geoms (seed jitter scales it)

    @property
    def active(self) -> HeadBuild:
        return next(h for h in self.heads if h.active)

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.setup.setup_id, self.tool, self.task_spec.task_id, self.task_digest)


# --------------------------------------------------------------------------- wrist


def _cylinder_inertia(mass: float, radius: float, height: float) -> list[float]:
    ixx = mass * (3.0 * radius**2 + height**2) / 12.0
    return [ixx, ixx, 0.5 * mass * radius**2]


def _add_studio_wrist(
    spec: mujoco.MjSpec, palm: mujoco.MjsBody, task: TaskV2, wrist: WristLimits, quat: list[float]
) -> mujoco.MjsBody:
    """The core wrist without geometry, its four frozen servos, the roll stage and the flange."""
    g, phys = task.gripper, task.physics
    spec.delete(spec.geom(BODY_PALM))
    palm.explicitinertial = True
    palm.mass = g.palm_mass
    palm.ipos = [0.0, 0.0, 0.0]
    palm.inertia = _cylinder_inertia(g.palm_mass, FLANGE_RADIUS, -2.0 * MOUNT_Z)
    xy = (-PALM_XY_LIMIT, PALM_XY_LIMIT)
    _add_position_actuator(spec, "palm_x", "palm_x", g.xy_kp, g.xy_kv, g.xy_force, xy)
    _add_position_actuator(spec, "palm_y", "palm_y", g.xy_kp, g.xy_kv, g.xy_force, xy)
    _add_position_actuator(
        spec, "palm_z", "palm_z", g.z_kp, g.z_kv, g.z_force, (phys.table_top_z - 0.05, phys.table_top_z + 0.8)
    )
    _add_position_actuator(
        spec, "palm_yaw", "palm_yaw", g.yaw_kp, g.yaw_kv, g.yaw_torque, (-YAW_LIMIT, YAW_LIMIT)
    )
    roll = palm.add_body(name=ROLL_BODY, gravcomp=1.0)
    roll.explicitinertial = True
    roll.mass = wrist.rotor_mass_kg
    roll.ipos = [0.0, 0.0, 0.0]
    roll.inertia = _cylinder_inertia(wrist.rotor_mass_kg, 0.5 * FLANGE_RADIUS, 0.01)
    roll.add_joint(
        name=ROLL_JOINT,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
        limited=mujoco.mjtLimited.mjLIMITED_FALSE,
        armature=wrist.roll_armature,
    )
    act = spec.add_actuator(name=ROLL_ACTUATOR, target=ROLL_JOINT, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=wrist.roll_kp, kv=wrist.roll_kv)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-wrist.roll_torque_nm, wrist.roll_torque_nm]
    flange = roll.add_body(name=FLANGE_BODY, quat=quat, gravcomp=1.0)
    flange.add_geom(
        name=FLANGE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[FLANGE_RADIUS, -MOUNT_Z, 0.0],
        mass=wrist.flange_mass_kg,
        rgba=list(PALM_RGBA),
    )
    return flange


# --------------------------------------------------------------------------- family builds


def _family_build(
    spec: mujoco.MjSpec,
    mount: mujoco.MjsBody,
    mount_z: float,
    effector: EffectorSpec,
    part: PartSpec,
    task: TaskV2,
) -> EffectorBuild:
    """Run the family's ``build`` under ``mount`` and check its additions (the v1 rules)."""
    impl = get_family(effector.family)
    impl.check_spec(effector)
    before = _names(spec)
    settings = _settings(spec)
    n_pairs = len(spec.pairs)
    ctx = BuildContext(
        spec=spec, mount=mount, mount_z=mount_z, effector=effector, part=part, task=task, repo_root=REPO_ROOT
    )
    build = impl.build(ctx)
    if not isinstance(build, EffectorBuild):
        raise ContractError(f"{type(impl).__name__}.build must return an EffectorBuild")
    if build.spec.effector_id != effector.effector_id:
        raise ContractError("the build describes a different effector spec")
    after = _names(spec)
    problems: list[str] = []
    for section, values in _settings(spec).items():
        changed = sorted(k for k, v in values.items() if settings[section].get(k) != v)
        if changed:
            problems.append(f"the family changed the scene's {section} settings: {changed[:5]}")
    if len(spec.pairs) != n_pairs:
        problems.append("explicit contact pairs are not allowed")
    for kind in _ELEMENT_LISTS:
        new = after[kind] - before[kind]
        bad = sorted(n for n in new if n and (not n.startswith(NAME_PREFIX) or n.startswith(STUDIO_PREFIX)))
        if bad:
            problems.append(f"new {kind} without the {NAME_PREFIX!r} prefix (or in the Studio's): {bad[:5]}")
        lost = sorted(before[kind] - after[kind])
        if lost:
            problems.append(f"the family removed scene {kind}: {lost[:5]}")
    new_bodies = [b for b in spec.bodies if b.name in after["bodies"] - before["bodies"]]
    if not new_bodies:
        problems.append("the family added no bodies")
    for b in new_bodies:
        if not _is_descendant(b, mount):
            problems.append(f"body {b.name!r} is not mounted under the tool mount")
    missing = sorted(set(build.bodies) - {b.name for b in new_bodies})
    if missing:
        problems.append(f"declared bodies that were not added: {missing[:5]}")
    new_acts = [a.name for a in spec.actuators if a.name in after["actuators"] - before["actuators"]]
    if tuple(new_acts) != build.actuators:
        problems.append(f"declared actuators {list(build.actuators)} != added actuators {new_acts}")
    if problems:
        raise ContractError(f"{type(impl).__name__} broke the Studio scene contract: " + "; ".join(problems))
    return build


def _scratch_spec(task: TaskV2) -> tuple[mujoco.MjSpec, mujoco.MjsBody]:
    """A spec with the scene's options and defaults but no visual assets (attach copies assets)."""
    spec, palm = _base_spec(task)
    for tex in list(spec.textures):
        spec.delete(tex)
    for mat in list(spec.materials):
        spec.delete(mat)
    return spec, palm


def _add_multi_head(
    spec: mujoco.MjSpec,
    flange: mujoco.MjsBody,
    setup: Setup,
    tool: int,
    parts: Mapping[int, PartSpec],
    task: TaskV2,
) -> dict[int, tuple[EffectorBuild, str, str]]:
    """Bracket, adapters and both heads (scratch build and attach); returns tool -> (build, prefix, mount)."""
    br = setup.bracket
    assert br is not None
    hub = br.hub_thickness_mm / 1000.0
    t_ad = br.adapter_thickness_mm / 1000.0
    bracket = flange.add_body(name=f"{STUDIO_PREFIX}bracket", pos=[0.0, 0.0, MOUNT_Z])
    bracket.add_geom(
        name=f"{STUDIO_PREFIX}hub",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[FLANGE_RADIUS, 0.5 * hub, 0.0],
        pos=[0.0, 0.0, -0.5 * hub],
        density=br.density_kg_m3,
        rgba=[0.3, 0.32, 0.36, 1.0],
    )
    out: dict[int, tuple[EffectorBuild, str, str]] = {}
    for k, (effector, m) in enumerate(zip(setup.tools, setup.mounts, strict=True)):
        rot = _rot_x(m.angle_deg)
        # mount plane centre relative to the wrist centre (flange frame), then the bracket frame
        centre = rot @ np.array([m.lateral_mm, 0.0, -m.standoff_mm]) / 1000.0
        in_bracket = centre - np.array([0.0, 0.0, MOUNT_Z])
        spin = math.radians(m.spin_deg)
        r_spin = np.array(
            [[math.cos(spin), -math.sin(spin), 0.0], [math.sin(spin), math.cos(spin), 0.0], [0, 0, 1]]
        )
        start = np.array([0.0, 0.0, -hub])
        if np.linalg.norm(in_bracket - start) > 1e-6:
            bracket.add_geom(
                name=f"{STUDIO_PREFIX}arm{k}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*start.tolist(), *in_bracket.tolist()],
                size=[0.5 * br.arm_width_mm / 1000.0, 0.0, 0.0],
                density=br.density_kg_m3,
                rgba=[0.3, 0.32, 0.36, 1.0],
            )
        scratch, scratch_palm = _scratch_spec(task)
        mount = scratch_palm.add_body(name=_MOUNT)
        mount.add_geom(
            name=f"{STUDIO_PREFIX}adapter",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[FLANGE_RADIUS, 0.5 * t_ad, 0.0],
            pos=[0.0, 0.0, -0.5 * t_ad],
            density=br.density_kg_m3,
            rgba=[0.3, 0.32, 0.36, 1.0],
        )
        build = _family_build(scratch, mount, -t_ad, effector, parts[k], task)
        prefix = "" if k == tool else idle_prefix(k)
        frame = bracket.add_frame(pos=in_bracket.tolist(), quat=_quat(rot @ r_spin))
        frame.attach_body(mount, prefix, "")
        out[k] = (build, prefix, prefix + _MOUNT)
    return out


def _add_changer(spec: mujoco.MjSpec, flange: mujoco.MjsBody, setup: Setup) -> tuple[mujoco.MjsBody, float]:
    ch = setup.changer
    assert ch is not None
    stack = ch.stack_mm / 1000.0
    body = flange.add_body(name=f"{STUDIO_PREFIX}changer", pos=[0.0, 0.0, MOUNT_Z])
    for name, mass, z0 in (
        ("master", ch.master_mass_kg, 0.0),
        ("tool_plate", ch.tool_plate_mass_kg, -0.5 * stack),
    ):
        body.add_geom(
            name=f"{STUDIO_PREFIX}{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[FLANGE_RADIUS, 0.25 * stack, 0.0],
            pos=[0.0, 0.0, z0 - 0.25 * stack],
            mass=mass,
            rgba=[0.45, 0.47, 0.5, 1.0],
        )
    return body, -stack


#: What a tool that picks nothing is built against: a 10 mm cube, so a family's opening and
#: reach checks (sized for a part to grip) cannot refuse a hook, pusher or twisting tool. Its fit
#: on the objects it does touch is checked by the search (``rtt.studio.search``).
PROBE_PART_NAME: Final = "probe_block_10mm"


def build_part_for(setup: Setup, task_spec: TaskSpec, tool: int) -> PartSpec:
    """The part a family sees when it builds ``tool`` (its opening checks use it): the first part
    that tool picks (grasp, suction, wrap), else the 10 mm probe block (:data:`PROBE_PART_NAME`)."""
    from rtt.parts.library import build_part
    from rtt.studio.templates.part import part_spec_of

    picks = (PrimitiveKind.GRASP, PrimitiveKind.SUCTION, PrimitiveKind.WRAP)
    for s in task_spec.steps:
        if setup.assignment[s.id] == tool and s.primitive in picks:
            obj = task_spec.object(s.object)
            if obj.kind is TemplateKind.PART:
                return part_spec_of(obj)
    return build_part(PROBE_PART_NAME, "block", {"x": 10.0, "y": 10.0, "z": 10.0})


# --------------------------------------------------------------------------- compiled checks


def _subtree(model: mujoco.MjModel, root: int) -> list[int]:
    out = []
    for b in range(model.nbody):
        node = b
        while node > 0 and node != root:
            node = int(model.body_parentid[node])
        if node == root and b != root:
            out.append(b)
    return out


def _bodies_of(model: mujoco.MjModel, objtype: int, i: int) -> set[int]:
    """Bodies a transmission or equality target refers to (``-1`` for the world body's)."""
    o = mujoco.mjtObj
    if i < 0:
        return set()
    if objtype == o.mjOBJ_BODY:
        return {int(i)}
    if objtype == o.mjOBJ_SITE:
        return {int(model.site_bodyid[i])}
    if objtype == o.mjOBJ_JOINT:
        return {int(model.jnt_bodyid[i])}
    if objtype == o.mjOBJ_GEOM:
        return {int(model.geom_bodyid[i])}
    return set()


def _tendon_bodies(model: mujoco.MjModel, t: int) -> set[int]:
    w = mujoco.mjtWrap
    out: set[int] = set()
    for k in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t]):
        kind, i = int(model.wrap_type[k]), int(model.wrap_objid[k])
        if kind == w.mjWRAP_JOINT:
            out.add(int(model.jnt_bodyid[i]))
        elif kind == w.mjWRAP_SITE:
            out.add(int(model.site_bodyid[i]))
        elif kind in (w.mjWRAP_SPHERE, w.mjWRAP_CYLINDER):
            out.add(int(model.geom_bodyid[i]))
    return out


def _check_compiled(
    model: mujoco.MjModel, eff: set[int], wrist: set[int], owners: dict[str, set[int]]
) -> list[str]:
    """Tools act only on themselves and objects only on themselves (module docstring).

    ``owners``: template instance -> its bodies (the world body is allowed for static anchors).
    """
    o = mujoco.mjtObj
    problems: list[str] = []

    def nm(objtype: mujoco.mjtObj, i: int) -> str:
        return mujoco.mj_id2name(model, objtype, int(i)) or f"#{i}"

    def owner_ok(name: str, bodies: set[int]) -> bool:
        if name.startswith(NAME_PREFIX):
            return bool(bodies) and bodies <= eff
        matches = [inst for inst in owners if name.startswith(object_prefix(inst))]
        if matches:  # the longest prefix: "obj_tote_lid_x" belongs to tote_lid, not tote
            return bodies <= owners[max(matches, key=len)] | {0}
        return False

    trn = mujoco.mjtTrn
    for a in range(len(WRIST_JOINTS), model.nu):
        kind, (i, j) = int(model.actuator_trntype[a]), model.actuator_trnid[a]
        if kind in (trn.mjTRN_JOINT, trn.mjTRN_JOINTINPARENT):
            bodies = _bodies_of(model, o.mjOBJ_JOINT, i)
        elif kind == trn.mjTRN_TENDON:
            bodies = _tendon_bodies(model, int(i))
        elif kind in (trn.mjTRN_SITE, trn.mjTRN_SLIDERCRANK):
            bodies = _bodies_of(model, o.mjOBJ_SITE, i) | _bodies_of(model, o.mjOBJ_SITE, j)
        elif kind == trn.mjTRN_BODY:
            bodies = {int(i)}
        else:
            bodies = {-1}
        if not owner_ok(nm(o.mjOBJ_ACTUATOR, a), bodies):
            problems.append(f"actuator {nm(o.mjOBJ_ACTUATOR, a)!r} acts outside its owner")
    eq = mujoco.mjtEq
    for e in range(model.neq):
        kind = int(model.eq_type[e])
        if kind in (eq.mjEQ_CONNECT, eq.mjEQ_WELD):
            ot = int(model.eq_objtype[e])
            bodies = _bodies_of(model, ot, int(model.eq_obj1id[e])) | _bodies_of(
                model, ot, int(model.eq_obj2id[e])
            )
        elif kind == eq.mjEQ_JOINT:
            bodies = _bodies_of(model, o.mjOBJ_JOINT, int(model.eq_obj1id[e])) | _bodies_of(
                model, o.mjOBJ_JOINT, int(model.eq_obj2id[e])
            )
        elif kind == eq.mjEQ_TENDON:
            bodies = _tendon_bodies(model, int(model.eq_obj1id[e]))
            if model.eq_obj2id[e] >= 0:
                bodies |= _tendon_bodies(model, int(model.eq_obj2id[e]))
        else:
            bodies = {-1}
        if bodies <= eff and bodies:
            continue
        if not any(bodies <= own | {0} for own in owners.values()):
            problems.append(
                f"equality {nm(o.mjOBJ_EQUALITY, e)!r} ties a tool or an object to something else"
            )
    for t in range(model.ntendon):
        if not owner_ok(nm(o.mjOBJ_TENDON, t), _tendon_bodies(model, t)):
            problems.append(f"tendon {nm(o.mjOBJ_TENDON, t)!r} wraps outside its owner")
    eff_geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(eff)))
    if np.any(model.geom_priority[eff_geoms] >= 1):
        problems.append("tool geoms may not have priority >= 1 (the object's contact parameters govern)")
    if np.any(model.geom_adhesion[eff_geoms] != 0):
        problems.append("tool geoms may not use geom adhesion")
    if np.any(model.geom_margin[eff_geoms] > 0):
        problems.append("tool geoms may not have a contact margin > 0")
    if np.any(model.geom_surfacevel[eff_geoms] != 0):
        problems.append("tool geoms may not have a surface velocity")
    obj_geoms = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, o.mjOBJ_GEOM, g) or "").startswith(OBJ_PREFIX)
    ]
    if obj_geoms and np.any(model.geom_priority[obj_geoms] < 1):
        problems.append("object geoms need priority 1 (their contact parameters govern every tool contact)")
    if model.npair:
        problems.append("explicit contact pairs are not allowed")
    tool_side = eff | wrist
    for sig in model.exclude_signature:
        b1, b2 = int(sig) >> 16, int(sig) & 0xFFFF
        if {b1, b2} <= tool_side or any({b1, b2} <= own | {0} for own in owners.values()):
            continue
        problems.append(f"a contact exclude pairs {nm(o.mjOBJ_BODY, b1)!r} and {nm(o.mjOBJ_BODY, b2)!r}")
    return problems


# --------------------------------------------------------------------------- the builder


def _resolve_handles(model: mujoco.MjModel, h: TemplateHandles) -> ResolvedHandles:
    def jid(n: str) -> int:
        return model.joint(n).id

    return ResolvedHandles(
        handles=h,
        body_ids=tuple(model.body(b).id for b in h.bodies),
        geom_ids=np.array([model.geom(g).id for g in h.geoms], dtype=np.int64),
        joint_qpos={r: int(model.jnt_qposadr[jid(n)]) for r, n in h.joints.items()},
        joint_dof={r: int(model.jnt_dofadr[jid(n)]) for r, n in h.joints.items()},
        site_ids={k: model.site(n).id for k, n in h.sites.items()},
        actuator_ids=np.array([model.actuator(a).id for a in h.actuators], dtype=np.int64),
    )


def build_studio_scene(
    setup: Setup,
    task_spec: TaskSpec,
    tool: int,
    *,
    task: TaskV2 | None = None,
    wrist: WristLimits = STUDIO_WRIST,
    edit: Callable[[mujoco.MjSpec], Any] | None = None,
) -> StudioScene:
    """Author, check and compile the scene for ``setup`` with ``tool`` active (module docstring).

    The control vector is the five wrist servos (x, y, z, yaw, roll), then every head's
    actuators in head order, then the objects' passive actuators. ``edit`` (tests only) may
    change the authored spec before compiling. Raises :class:`~rtt.effectors.types.InvalidEffector`
    if a family cannot build its tool for its part, :class:`~rtt.studio.templates.TemplateUnavailable`
    if an object has no physics yet, and :class:`~rtt.effectors.types.ContractError` on a broken rule.
    """
    task = task or default_task_v2()
    if not isinstance(task, TaskV2):
        raise TypeError("Studio scenes need a TaskV2 (configs/task_v2.yaml)")
    problems = setup.problems(task_spec)
    if problems:
        raise ValueError("this setup cannot run this task: " + "; ".join(problems))
    if not 0 <= tool < len(setup.tools):
        raise ValueError(f"tool {tool} is not one of the setup's {len(setup.tools)} tools")
    spec, palm = _base_spec(task)
    spec.modelname = "rtt_studio"
    flange = _add_studio_wrist(spec, palm, task, wrist, present_quat(setup, tool))
    parts = {k: build_part_for(setup, task_spec, k) for k in range(len(setup.tools))}

    heads_raw: dict[int, tuple[EffectorBuild, str, str]] = {}
    if setup.kind is SetupKind.SINGLE:
        heads_raw[0] = (_family_build(spec, flange, MOUNT_Z, setup.tools[0], parts[0], task), "", FLANGE_BODY)
    elif setup.kind is SetupKind.TOOL_CHANGE:
        mount, mount_z = _add_changer(spec, flange, setup)
        heads_raw[tool] = (
            _family_build(spec, mount, mount_z, setup.tools[tool], parts[tool], task),
            "",
            mount.name,
        )
    else:
        heads_raw = _add_multi_head(spec, flange, setup, tool, parts, task)
    for b in spec.bodies:
        if b.name.startswith(NAME_PREFIX):
            b.gravcomp = 1.0

    handles: dict[str, TemplateHandles] = {}
    builders: dict[str, TemplateBuilder] = {}
    for inst in task_spec.objects:
        builder = get_template_builder(inst.kind)
        before = _names(spec)
        prefix = object_prefix(inst.name)
        h = builder.add(
            TemplateContext(spec=spec, instance=inst, prefix=prefix, task=task, repo_root=REPO_ROOT)
        )
        if not isinstance(h, TemplateHandles) or h.instance != inst.name or h.kind is not inst.kind:
            raise ContractError(f"the {inst.kind.value} builder must return this instance's TemplateHandles")
        after = _names(spec)
        for kind in _ELEMENT_LISTS:
            bad = sorted(n for n in after[kind] - before[kind] if n and not n.startswith(prefix))
            if bad:
                raise ContractError(
                    f"the {inst.kind.value} builder added {kind} without {prefix!r}: {bad[:5]}"
                )
        undeclared = sorted(
            (after["bodies"] - before["bodies"] - set(h.bodies))
            | (after["geoms"] - before["geoms"] - set(h.geoms))
        )
        if undeclared:
            raise ContractError(
                f"the {inst.kind.value} builder did not declare {undeclared[:5]} in its handles"
            )
        for b in spec.bodies:
            if b.name.startswith(prefix) and _is_descendant(b, palm):
                raise ContractError(f"object body {b.name!r} hangs under the wrist")
        handles[inst.name], builders[inst.name] = h, builder
    if edit is not None:
        edit(spec)
    model = spec.compile()

    # ------------------------------------------------------------------ ids and checks
    def bid(n: str) -> int:
        return model.body(n).id

    palm_id, roll_id, flange_id = (bid(n) for n in WRIST_BODIES)
    wrist_acts = np.array([model.actuator(n).id for n in (*PALM_JOINTS, ROLL_ACTUATOR)], dtype=np.int64)
    if list(wrist_acts) != list(range(len(WRIST_JOINTS))):
        raise ContractError("the compiled actuator order does not start with the five wrist servos")
    eff = {
        b
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(NAME_PREFIX)
    }
    owners = {name: {model.body(b).id for b in h.bodies} for name, h in handles.items()}
    problems = _check_compiled(model, eff, {palm_id, roll_id, flange_id}, owners)
    if problems:
        raise ContractError("the Studio scene contract is broken: " + "; ".join(problems))

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    r_palm = data.xmat[palm_id].reshape(3, 3)
    heads: list[HeadBuild] = []
    head_bodies: dict[int, list[int]] = {}
    for k in sorted(heads_raw):
        build, prefix, mount_name = heads_raw[k]
        mount_id = bid(mount_name)
        bodies = [b for b in _subtree(model, mount_id) if b in eff and not _is_stack(model, b)]
        head_bodies[k] = bodies
        mass, com, inertia = _mass_properties(model, np.array(bodies, dtype=np.int64))
        build = replace(
            build,
            mass=mass,
            com=tuple(float(v) for v in com),
            inertia6=tuple(float(inertia[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))),  # type: ignore[arg-type]
        )
        r_m = data.xmat[mount_id].reshape(3, 3)
        tcp = r_palm.T @ (data.xpos[mount_id] + r_m @ np.asarray(build.tcp) - data.xpos[palm_id])
        axis = r_palm.T @ r_m @ np.array([0.0, 0.0, -1.0])
        heads.append(
            HeadBuild(
                tool=k,
                spec=setup.tools[k],
                build=build,
                prefix=prefix,
                active=k == tool,
                mount_body=mount_name,
                tcp_wrist=tuple(float(v) for v in tcp),  # type: ignore[arg-type]
                axis_wrist=tuple(float(v) for v in axis),  # type: ignore[arg-type]
            )
        )
    active_head = next(h for h in heads if h.active)
    stack = sorted(b for b in eff if _is_stack(model, b))
    idle_bodies = sorted(b for k, bs in head_bodies.items() if k != tool for b in bs)
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    eff_arr = np.array(sorted(eff), dtype=np.int64)
    is_eff = np.isin(model.geom_bodyid, eff_arr)
    obj_geom_ids = np.concatenate(
        [np.array([model.geom(g).id for g in h.geoms], dtype=np.int64) for h in handles.values()]
        or [np.zeros(0, np.int64)]
    )
    group = np.full(model.ngeom, GROUP_OTHER, dtype=np.int8)
    group[model.body_weldid[model.geom_bodyid] == 0] = GROUP_ENV
    group[np.isin(model.geom_bodyid, [palm_id, roll_id, flange_id])] = GROUP_PALM
    group[is_eff] = GROUP_FINGER_LEFT
    moving_obj = obj_geom_ids[model.body_weldid[model.geom_bodyid[obj_geom_ids]] != 0]
    group[moving_obj] = GROUP_OBJECT
    obj_acts = [model.actuator(a).id for h in handles.values() for a in h.actuators]
    idle_acts = [model.actuator(a).id for h in heads if not h.active for a in h.actuators]
    ids = StudioSceneIds(
        palm_body=palm_id,
        roll_body=roll_id,
        flange_body=flange_id,
        palm_qpos=np.array([model.jnt_qposadr[model.joint(n).id] for n in PALM_JOINTS]),
        roll_qpos=int(model.jnt_qposadr[model.joint(ROLL_JOINT).id]),
        roll_dof=int(model.jnt_dofadr[model.joint(ROLL_JOINT).id]),
        wrist_actuators=wrist_acts,
        effector_actuators=np.array([model.actuator(a).id for a in active_head.actuators], dtype=np.int64),
        idle_actuators=np.array(idle_acts, dtype=np.int64),
        object_actuators=np.array(obj_acts, dtype=np.int64),
        effector_bodies=eff_arr,
        active_bodies=np.array(head_bodies[tool], dtype=np.int64),
        idle_bodies=np.array(idle_bodies, dtype=np.int64),
        stack_bodies=np.array(stack, dtype=np.int64),
        effector_geoms=np.flatnonzero(is_eff & collides),
        active_geoms=np.flatnonzero(np.isin(model.geom_bodyid, head_bodies[tool]) & collides),
        idle_geoms=np.flatnonzero(np.isin(model.geom_bodyid, idle_bodies) & collides),
        object_geoms=obj_geom_ids,
        geom_group=group,
    )
    stack_mass = float(model.body_mass[stack].sum()) if stack else 0.0
    return StudioScene(
        model=model,
        ids=ids,
        setup=setup,
        task_spec=task_spec,
        tool=tool,
        heads=tuple(heads),
        objects={name: _resolve_handles(model, h) for name, h in handles.items()},
        builders=builders,
        wrist=wrist,
        stack_mass_kg=stack_mass,
        tool_mass_kg=float(model.body_mass[eff_arr].sum()),
        task_digest=task_digest(task),
        object_friction=model.geom_friction[obj_geom_ids, 0].copy(),
    )


def _is_stack(model: mujoco.MjModel, body: int) -> bool:
    """A Studio tool-stack body (bracket, adapter, changer), under any head prefix."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
    return name.startswith(STUDIO_PREFIX) or name.endswith(_MOUNT)
