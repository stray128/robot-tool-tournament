"""MjSpec scene: floating parallel-jaw gripper, table and one free object.

Gripper model (identical for every entrant except the finger boxes):

* ``palm``: a body with slide joints ``palm_x``, ``palm_y``, ``palm_z`` (world
  axes) and a yaw hinge ``palm_yaw``. The joint values are the palm origin's
  world position and yaw. Each joint is driven by a position actuator with a
  finite ``kp``/``kv`` and a force limit (z: +-150 N, yaw: +-15 N m). The palm
  and fingers are gravity compensated, as an arm controller would do; the
  carried object is not.
* ``finger_left`` / ``finger_right``: children of the palm on slide joints of
  the same names. The joint value ``q`` is the closure of that finger,
  ``q = 0`` fully open and ``q = stroke`` (40 mm) fully closed. The left finger
  root sits at palm-local ``y = open_half_gap - q``, the right one is mirrored.
  A joint equality keeps ``q_left == q_right`` (self-centring jaw).
* ``grip``: a fixed tendon ``0.5 q_left + 0.5 q_right`` (mean closure) driven by
  a position servo whose force is clipped to ``+-F`` (``F = 40 N``). In a static
  squeeze each finger therefore presses on the object with ``F / 2``.

Finger collision geoms are the 14 boxes of a :class:`~rtt.design.types.FingerGeometry`
(left finger as authored, right finger mirrored ``y -> -y``). Finger mass and
inertia are computed from the box volumes and the PLA density and set
explicitly, so a batched executor can override them per world.

Contacts: every pair collides except palm-finger (MuJoCo filters parent-child
pairs). Palm-object and finger-finger contacts are real; gripper-table contacts
are simulated and flagged by checks.py. The object geom has ``priority=1`` so
its friction (the jittered mu) governs all of its contacts.

The table top is the plane ``z = table_top_z`` (0 by default); world frame is
Z-up; all angles are radians.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from rtt.design.types import Box, FingerGeometry
from rtt.sim import objects as objs
from rtt.sim.task import TaskConfig

BODY_PALM = "palm"
BODY_FINGER_LEFT = "finger_left"
BODY_FINGER_RIGHT = "finger_right"
BODY_OBJECT = "object"
REPLAY_BODIES: tuple[str, ...] = (BODY_PALM, BODY_FINGER_LEFT, BODY_FINGER_RIGHT, BODY_OBJECT)

PALM_JOINTS: tuple[str, ...] = ("palm_x", "palm_y", "palm_z", "palm_yaw")
GRIP_TENDON = "grip"
GRIP_ACTUATOR = "grip"
YAW_LIMIT = 2.0 * np.pi  # palm yaw range (rad); wide so any placement yaw is reachable without wrapping
PALM_XY_LIMIT = 0.35  # palm x/y travel (m) either side of the table centre
MAX_FINGER_EXTENT = 0.3  # m; a larger finger is almost certainly authored in millimetres

# Geom groups used by the physics checks (see checks.py).
GROUP_OTHER = 0
GROUP_ENV = 1  # table and floor
GROUP_PALM = 2
GROUP_FINGER_LEFT = 3
GROUP_FINGER_RIGHT = 4
GROUP_OBJECT = 5

PALM_RGBA = (0.20, 0.22, 0.27, 1.0)
FINGER_RGBA = (0.96, 0.52, 0.13, 1.0)
OBJECT_RGBA = (0.18, 0.47, 0.86, 1.0)
TABLE_RGBA = (0.80, 0.70, 0.56, 1.0)

_MIRROR = np.diag([1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- geometry helpers


def quat_to_mat(quat_wxyz: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Rotation matrix of a (not necessarily normalised) wxyz quaternion."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def mirror_box(box: Box) -> Box:
    """Reflect a left-finger box across the XZ plane (y -> -y).

    A reflection ``M`` maps the oriented box (c, R) to (M c, M R M); for a
    unit quaternion (w, x, y, z) this is (w, -x, y, -z). Box half sizes are
    unchanged because a box is symmetric in its own frame.
    """
    cx, cy, cz = box.center
    w, x, y, z = box.quat_wxyz
    return Box(center=(cx, -cy, cz), half_size=box.half_size, quat_wxyz=(w, -x, y, -z))


def box_corners(box: Box) -> np.ndarray:
    """The 8 corners of a box in the finger frame, shape (8, 3)."""
    rot = quat_to_mat(box.quat_wxyz)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    return np.asarray(box.center) + (signs * np.asarray(box.half_size)) @ rot.T


def finger_tip_depth(finger: FingerGeometry) -> float:
    """Distance from the finger root down to the lowest point of the finger (> 0)."""
    return float(-min(box_corners(b)[:, 2].min() for b in finger.boxes))


def finger_inner_offset(finger: FingerGeometry) -> float:
    """Smallest local y over all boxes: how far the inner face sits from the root plane."""
    return float(min(box_corners(b)[:, 1].min() for b in finger.boxes))


def open_gap(finger: FingerGeometry, task: TaskConfig) -> float:
    """Clear distance between the two fingers' innermost points when fully open."""
    return 2.0 * (task.gripper.open_half_gap + finger_inner_offset(finger))


@dataclass(frozen=True)
class MassProperties:
    """Mass, centre of mass and inertia tensor (about the COM) in the body frame."""

    mass: float
    com: np.ndarray
    inertia: np.ndarray

    def fullinertia(self) -> list[float]:
        """MuJoCo ``fullinertia`` order: Ixx, Iyy, Izz, Ixy, Ixz, Iyz."""
        i = self.inertia
        return [i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]


def boxes_mass_properties(boxes: tuple[Box, ...], density: float) -> MassProperties:
    """Combine solid boxes (overlaps counted twice) into one rigid body's mass properties."""
    masses = []
    centers = []
    inertias = []
    for b in boxes:
        hx, hy, hz = b.half_size
        m = 8.0 * hx * hy * hz * density
        local = (m / 3.0) * np.diag([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])
        rot = quat_to_mat(b.quat_wxyz)
        masses.append(m)
        centers.append(np.asarray(b.center, dtype=np.float64))
        inertias.append(rot @ local @ rot.T)
    total = float(sum(masses))
    com = sum(m * c for m, c in zip(masses, centers, strict=True)) / total
    inertia = np.zeros((3, 3))
    for m, c, i_c in zip(masses, centers, inertias, strict=True):
        d = c - com
        inertia += i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return MassProperties(mass=total, com=com, inertia=inertia)


def finger_mass_properties(finger: FingerGeometry, mirrored: bool = False) -> MassProperties:
    """Mass properties of the left finger, or of its mirror image when ``mirrored``."""
    props = boxes_mass_properties(finger.boxes, finger.density_kg_m3)
    if not mirrored:
        return props
    return MassProperties(mass=props.mass, com=_MIRROR @ props.com, inertia=_MIRROR @ props.inertia @ _MIRROR)


# --------------------------------------------------------------------------- scene construction


@dataclass(frozen=True)
class SceneIds:
    """Indices into a compiled model, resolved once per model."""

    palm_body: int
    finger_left_body: int
    finger_right_body: int
    object_body: int
    palm_qpos: np.ndarray  # qpos addresses of palm_x, palm_y, palm_z, palm_yaw
    finger_qpos: np.ndarray  # qpos addresses of finger_left, finger_right
    object_qpos: int  # start of the object's 7-dof free-joint qpos block
    object_dof: int  # start of the object's 6-dof velocity block
    palm_actuators: np.ndarray  # actuator ids for x, y, z, yaw
    grip_actuator: int
    object_geom: int
    geom_group: np.ndarray  # (ngeom,) GROUP_* label per geom
    replay_bodies: np.ndarray  # body ids in REPLAY_BODIES order

    @staticmethod
    def resolve(model: mujoco.MjModel) -> SceneIds:
        def body(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        def joint(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def actuator(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        palm, left, right, obj = (body(n) for n in REPLAY_BODIES)
        group = np.full(model.ngeom, GROUP_OTHER, dtype=np.int8)
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            if b == 0:
                group[g] = GROUP_ENV
            elif b == palm:
                group[g] = GROUP_PALM
            elif b == left:
                group[g] = GROUP_FINGER_LEFT
            elif b == right:
                group[g] = GROUP_FINGER_RIGHT
            elif b == obj:
                group[g] = GROUP_OBJECT
        free = joint("object_free")
        return SceneIds(
            palm_body=palm,
            finger_left_body=left,
            finger_right_body=right,
            object_body=obj,
            palm_qpos=np.array([model.jnt_qposadr[joint(n)] for n in PALM_JOINTS]),
            finger_qpos=np.array(
                [model.jnt_qposadr[joint(n)] for n in (BODY_FINGER_LEFT, BODY_FINGER_RIGHT)]
            ),
            object_qpos=int(model.jnt_qposadr[free]),
            object_dof=int(model.jnt_dofadr[free]),
            palm_actuators=np.array([actuator(n) for n in PALM_JOINTS]),
            grip_actuator=actuator(GRIP_ACTUATOR),
            object_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BODY_OBJECT),
            geom_group=group,
            replay_bodies=np.array([palm, left, right, obj]),
        )


def _add_position_actuator(
    spec: mujoco.MjSpec,
    name: str,
    target: str,
    kp: float,
    kv: float,
    force: float,
    ctrlrange: tuple[float, float],
    trntype: mujoco.mjtTrn = mujoco.mjtTrn.mjTRN_JOINT,
) -> None:
    act = spec.add_actuator(name=name, target=target, trntype=trntype)
    act.set_to_position(kp=kp, kv=kv)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = list(ctrlrange)
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-force, force]


def _add_finger(
    palm: mujoco.MjsBody,
    name: str,
    boxes: tuple[Box, ...],
    props: MassProperties,
    side: float,
    task: TaskConfig,
) -> None:
    g = task.gripper
    body = palm.add_body(name=name, pos=[0.0, side * g.open_half_gap, -g.palm_half_size[2]], gravcomp=1.0)
    body.add_joint(
        name=name,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0.0, -side, 0.0],
        range=[0.0, g.stroke],
        armature=g.finger_armature,
        damping=g.finger_damping,
    )
    body.explicitinertial = True
    body.mass = props.mass
    body.ipos = props.com.tolist()
    body.fullinertia = props.fullinertia()
    for i, b in enumerate(boxes):
        body.add_geom(
            name=f"{name}_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(b.half_size),
            pos=list(b.center),
            quat=list(b.quat_wxyz),
            rgba=list(FINGER_RGBA),
        )


def _add_visual_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.93, 0.95, 0.98],
        rgb2=[0.62, 0.70, 0.80],
        width=128,
        height=128,
    )
    spec.add_texture(
        name="floor",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.84, 0.87],
        rgb2=[0.74, 0.76, 0.80],
        width=128,
        height=128,
    )
    floor_mat = spec.add_material(name="floor", texrepeat=[6.0, 6.0], reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    spec.visual.quality.shadowsize = 2048
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960


def check_finger_units(finger: FingerGeometry) -> None:
    """Reject finger geometry that is clearly not in metres (e.g. authored in millimetres)."""
    extent = max(float(np.abs(box_corners(b)).max()) for b in finger.boxes)
    if not np.isfinite(extent) or extent > MAX_FINGER_EXTENT:
        raise ValueError(
            f"finger geometry reaches {extent:g} from its root; sizes must be in metres "
            f"(at most {MAX_FINGER_EXTENT} m)"
        )


def build_spec(finger: FingerGeometry, obj: objs.ObjectSpec, task: TaskConfig) -> mujoco.MjSpec:
    """Author the full scene for one finger design and one object."""
    check_finger_units(finger)
    phys, g = task.physics, task.gripper
    spec = mujoco.MjSpec()
    spec.modelname = "rtt_grasp"
    spec.compiler.degree = False  # all angles in this module are radians

    opt = spec.option
    opt.timestep = phys.timestep
    opt.integrator = {
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    }[phys.integrator]
    opt.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[
        phys.cone
    ]
    opt.impratio = phys.impratio
    opt.iterations = phys.iterations
    opt.ls_iterations = phys.ls_iterations
    if not phys.native_ccd:
        opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    if not phys.multi_ccd:
        opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD

    dgeom = spec.default.geom
    dgeom.condim = phys.condim
    dgeom.friction = [phys.friction, phys.torsional_friction, phys.rolling_friction]
    dgeom.solref = list(phys.contact_solref)
    dgeom.solimp = list(phys.contact_solimp)
    spec.default.joint.solref_limit = list(phys.limit_solref)

    _add_visual_assets(spec)
    world = spec.worldbody
    world.add_light(
        name="key", pos=[0.4, -0.6, 1.4], dir=[-0.3, 0.45, -1.0], castshadow=True, diffuse=[0.6] * 3
    )
    world.add_light(
        name="fill", pos=[-0.6, 0.4, 1.0], dir=[0.5, -0.3, -1.0], castshadow=False, diffuse=[0.25] * 3
    )

    tx, ty, tz = phys.table_half_size
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[tx, ty, tz],
        pos=[0.0, 0.0, phys.table_top_z - tz],
        rgba=list(TABLE_RGBA),
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3.0, 3.0, 0.05],
        pos=[0.0, 0.0, phys.table_top_z - 2 * tz],
        material="floor",
    )

    # Palm: three world-axis slides plus a yaw hinge, all servo driven.
    px, py, pz = g.palm_half_size
    palm = world.add_body(name=BODY_PALM, gravcomp=1.0)
    for name, axis in (("palm_x", [1, 0, 0]), ("palm_y", [0, 1, 0]), ("palm_z", [0, 0, 1])):
        xy_range = [-PALM_XY_LIMIT, PALM_XY_LIMIT]
        rng = [phys.table_top_z - 0.05, phys.table_top_z + 0.8] if name == "palm_z" else xy_range
        palm.add_joint(
            name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, range=rng, armature=g.slide_armature
        )
    palm.add_joint(
        name="palm_yaw",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[-YAW_LIMIT, YAW_LIMIT],
        armature=g.yaw_armature,
    )
    palm.add_geom(
        name="palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[px, py, pz], mass=g.palm_mass, rgba=list(PALM_RGBA)
    )

    _add_finger(palm, BODY_FINGER_LEFT, finger.boxes, finger_mass_properties(finger), +1.0, task)
    mirrored = tuple(mirror_box(b) for b in finger.boxes)
    _add_finger(palm, BODY_FINGER_RIGHT, mirrored, finger_mass_properties(finger, mirrored=True), -1.0, task)

    tendon = spec.add_tendon(name=GRIP_TENDON)
    tendon.wrap_joint(BODY_FINGER_LEFT, 0.5)
    tendon.wrap_joint(BODY_FINGER_RIGHT, 0.5)
    spec.add_equality(
        name="finger_coupling",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1=BODY_FINGER_LEFT,
        name2=BODY_FINGER_RIGHT,
        solref=list(g.equality_solref),
    )

    # Object: a free body; priority 1 makes its friction govern all of its contacts.
    gtype, gsize, gquat = objs.geom_params(obj)
    density = obj.density if obj.density is not None else phys.object_density
    body = world.add_body(name=BODY_OBJECT)
    body.add_freejoint(name="object_free")
    body.add_geom(
        name=BODY_OBJECT,
        type=gtype,
        size=gsize,
        quat=list(gquat),
        density=density,
        priority=1,
        rgba=list(OBJECT_RGBA),
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
    _add_position_actuator(
        spec,
        GRIP_ACTUATOR,
        GRIP_TENDON,
        g.grip_kp,
        g.grip_kv,
        g.grip_force,
        (task.controller.open_target, g.stroke + g.grip_overdrive),
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
    )
    return spec


def build_model(
    finger: FingerGeometry, obj: objs.ObjectSpec, task: TaskConfig
) -> tuple[mujoco.MjModel, SceneIds]:
    """Compile the scene and resolve the ids the episode loop needs."""
    model = build_spec(finger, obj, task).compile()
    return model, SceneIds.resolve(model)
