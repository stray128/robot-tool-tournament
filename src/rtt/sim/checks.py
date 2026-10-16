"""Physics sanity checks evaluated during an episode.

Each check raises a flag (recorded once, with the time and controller phase of
its first occurrence). Any flag makes the episode a failure; see score.py for
how flags map onto failure modes.

* ``gripper_env_force``: palm/finger contact with the table or floor whose
  normal force exceeds ``finger_table_force_max``.
* ``penetration``: any contact deeper than ``penetration_max``, except between the two
  fingers. The fingers are one position-servoed mechanism; when a grasp fails they
  close on nothing and meet at the jaw's 0 mm closed gap, where the soft contact lets
  them overlap ~2.7 mm at the 2 ms timestep. That overlap cannot touch the object and
  only relabelled such failures by phase (e.g. a missed grasp as ``ejected_on_close``),
  so it is tracked as ``jaw_penetration_max`` instead of flagged.
* ``speed_cap``: object linear speed above ``speed_cap`` (numerical blow-up).
* ``ejection``: during CLOSE the object moves more than ``eject_dist``
  horizontally from where it was when closing started, or faster than
  ``eject_speed``.
* ``off_table``: the object's centre drops more than ``off_table_drop`` below the table top.
* ``palm_contact``: while the object is carried (LIFT and HOLD) the palm presses on it
  with a normal force above ``palm_contact_force_max``. Only the fingers may hold the
  object; a part wedged between the finger tips and the palm is not a finger grasp.
  The controller keeps the palm clear of the object's top (see controller.py), so this
  flag means the object was pushed or rotated up into the palm.

* ``bad_qacc`` (task v2 only, ``PhysicsChecks(bad_qacc=True)``): MuJoCo's bad-acceleration,
  bad-velocity or bad-position warning counter went up. MuJoCo then resets the simulation
  state silently (positions back to the model defaults, time to 0), so nothing after that
  moment is physics; the episode is labelled ``physics_invalid`` (score.py). The counters
  are cumulative, so a reset between two checks is never missed. v1 episodes never raise it
  (task_v1 is frozen), but ``bad_qacc_count`` is always available.

Contact normal forces come from ``mj_contactForce``, which is correct for both
friction-cone types (the raw ``efc_force`` rows are pyramid edges, not the
normal force, under a pyramidal cone).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from rtt.sim.controller import Phase
from rtt.sim.scene import (
    GROUP_ENV,
    GROUP_FINGER_LEFT,
    GROUP_FINGER_RIGHT,
    GROUP_OBJECT,
    GROUP_PALM,
    SceneIds,
)
from rtt.sim.task import TaskConfig

FLAG_GRIPPER_ENV = "gripper_env_force"
FLAG_PENETRATION = "penetration"
FLAG_SPEED_CAP = "speed_cap"
FLAG_EJECTION = "ejection"
FLAG_OFF_TABLE = "off_table"
FLAG_PALM_CONTACT = "palm_contact"
FLAG_BAD_QACC = "bad_qacc"

#: MuJoCo warnings that mean the simulation state was reset (NaN, Inf or huge values).
RESET_WARNINGS: tuple[mujoco.mjtWarning, ...] = (
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
)


def bad_state_count(data: mujoco.MjData) -> int:
    """How often MuJoCo has reset ``data`` for NaN / Inf / huge qacc, qvel or qpos."""
    return int(sum(data.warning[w].number for w in RESET_WARNINGS))


@dataclass(frozen=True)
class CheckEvent:
    """First occurrence of a check flag."""

    flag: str
    t: float
    phase: Phase
    detail: str = ""
    gripper_env: bool = False  # the offending contact is between the gripper and the table/floor


@dataclass
class ContactSummary:
    """Aggregate contact forces (N) involving the object and the gripper at one instant.

    Per-finger tuples are ordered (left, right). ``squeeze`` is the component of
    a finger's net normal force on the object along that finger's closing
    direction (left finger toward -palm Y, right finger toward +palm Y), clipped
    at 0. ``press_down`` is the downward (-Z) component of the same force,
    clipped at 0: a finger that landed on top of the object pushes it into the
    table rather than pinching it from the side.
    """

    object_env_normal: float = 0.0
    object_finger_normal: tuple[float, float] = (0.0, 0.0)
    squeeze: tuple[float, float] = (0.0, 0.0)
    press_down: tuple[float, float] = (0.0, 0.0)
    gripper_env_normal: float = 0.0
    object_palm_normal: float = 0.0


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds) -> ContactSummary:
    """Full contact-force breakdown (uses ``mj_contactForce``; call sparingly)."""
    out = ContactSummary()
    groups = ids.geom_group
    palm_y = data.xmat[ids.palm_body].reshape(3, 3)[:, 1]
    finger_normal = [0.0, 0.0]
    squeeze_vec = [np.zeros(3), np.zeros(3)]  # net normal force of each finger on the object (world)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.efc_address < 0:
            continue
        g0, g1 = int(groups[con.geom[0]]), int(groups[con.geom[1]])
        a, b = sorted((g0, g1))
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = float(wrench[0])
        if b == GROUP_OBJECT:
            if a == GROUP_ENV:
                out.object_env_normal += normal
            elif a == GROUP_PALM:
                out.object_palm_normal += normal
            elif a in (GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT):
                side = 0 if a == GROUP_FINGER_LEFT else 1
                finger_normal[side] += normal
                # The contact normal points from geom[0] to geom[1] and pushes geom[1] along
                # it. Geom order follows the collider's geom-type order, so the object can be
                # either geom (a capsule or cylinder object comes before a finger box).
                sign = 1.0 if g1 == GROUP_OBJECT else -1.0
                squeeze_vec[side] += sign * normal * con.frame[:3]
        elif a == GROUP_ENV and b in (GROUP_PALM, GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT):
            out.gripper_env_normal += normal
    out.object_finger_normal = (finger_normal[0], finger_normal[1])
    left, right = squeeze_vec
    out.squeeze = (max(0.0, -float(left @ palm_y)), max(0.0, float(right @ palm_y)))
    out.press_down = (max(0.0, -float(left[2])), max(0.0, -float(right[2])))
    return out


@dataclass
class PhysicsChecks:
    """Accumulates check maxima and first-occurrence flags over an episode."""

    model: mujoco.MjModel
    ids: SceneIds
    task: TaskConfig
    bad_qacc: bool = False  # raise FLAG_BAD_QACC when MuJoCo resets the state (task v2)
    events: list[CheckEvent] = field(default_factory=list)
    gripper_env_force_max: float = 0.0
    penetration_max: float = 0.0
    jaw_penetration_max: float = 0.0  # finger-finger overlap (diagnostic, never flagged)
    object_speed_max: float = 0.0
    close_xy_drift_max: float = 0.0
    palm_object_force_max: float = 0.0  # any phase (diagnostic)
    carry_palm_force_max: float = 0.0  # LIFT and HOLD only (what the palm_contact flag tests)
    bad_qacc_count: int = 0  # MuJoCo state resets of this MjData (its warning counters start at 0)
    _flagged: set[str] = field(default_factory=set, repr=False)
    _close_ref_xy: np.ndarray | None = field(default=None, repr=False)
    _is_gripper: np.ndarray = field(init=False, repr=False)
    _is_env: np.ndarray = field(init=False, repr=False)
    _is_palm: np.ndarray = field(init=False, repr=False)
    _is_object: np.ndarray = field(init=False, repr=False)
    _wrench: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        groups = self.ids.geom_group
        self._is_gripper = np.isin(groups, (GROUP_PALM, GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT))
        self._is_env = groups == GROUP_ENV
        self._is_palm = groups == GROUP_PALM
        self._is_object = groups == GROUP_OBJECT
        self._wrench = np.zeros(6)

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(e.flag for e in self.events)

    def _flag(self, flag: str, t: float, phase: Phase, detail: str = "", gripper_env: bool = False) -> None:
        if flag not in self._flagged:
            self._flagged.add(flag)
            self.events.append(
                CheckEvent(flag=flag, t=t, phase=phase, detail=detail, gripper_env=gripper_env)
            )

    def _normal_force(self, data: mujoco.MjData, contacts: np.ndarray) -> float:
        total = 0.0
        for i in contacts:
            mujoco.mj_contactForce(self.model, data, int(i), self._wrench)
            total += float(self._wrench[0])
        return total

    def update(self, data: mujoco.MjData, t: float, phase: Phase) -> None:
        """Run all checks on the current state."""
        score = self.task.score
        if self.bad_qacc:
            self.bad_qacc_count = bad_state_count(data)
            if self.bad_qacc_count > 0:
                self._flag(FLAG_BAD_QACC, t, phase, f"{self.bad_qacc_count} MuJoCo state reset(s)")
        ncon = data.ncon
        if ncon:
            con = data.contact
            geom = con.geom
            efc = con.efc_address
            active = efc >= 0
            if active.any():
                g = geom[active]
                depth = -con.dist[active]
                jaw = self._is_gripper[g[:, 0]] & self._is_gripper[g[:, 1]]
                if jaw.any():
                    self.jaw_penetration_max = max(self.jaw_penetration_max, float(depth[jaw].max()))
                if not jaw.all():
                    others = np.flatnonzero(~jaw)
                    deepest = int(others[np.argmax(depth[others])])
                    self.penetration_max = max(self.penetration_max, float(depth[deepest]))
                    if depth[deepest] > score.penetration_max:
                        g1, g2 = (int(v) for v in g[deepest])
                        names = "/".join(
                            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, v) or "?"
                            for v in (g1, g2)
                        )
                        env_pair = bool(
                            (self._is_gripper[g1] and self._is_env[g2])
                            or (self._is_env[g1] and self._is_gripper[g2])
                        )
                        self._flag(FLAG_PENETRATION, t, phase, names, gripper_env=env_pair)
                pair = (self._is_gripper[g[:, 0]] & self._is_env[g[:, 1]]) | (
                    self._is_env[g[:, 0]] & self._is_gripper[g[:, 1]]
                )
                if pair.any():
                    force = self._normal_force(data, np.flatnonzero(active)[pair])
                    self.gripper_env_force_max = max(self.gripper_env_force_max, force)
                    if force > score.finger_table_force_max:
                        self._flag(FLAG_GRIPPER_ENV, t, phase, f"{force:.1f} N", gripper_env=True)
                palm = (self._is_palm[g[:, 0]] & self._is_object[g[:, 1]]) | (
                    self._is_object[g[:, 0]] & self._is_palm[g[:, 1]]
                )
                if palm.any():
                    force = self._normal_force(data, np.flatnonzero(active)[palm])
                    self.palm_object_force_max = max(self.palm_object_force_max, force)
                    if phase >= Phase.LIFT:
                        self.carry_palm_force_max = max(self.carry_palm_force_max, force)
                        if force > score.palm_contact_force_max:
                            self._flag(FLAG_PALM_CONTACT, t, phase, f"{force:.2f} N")

        dof = self.ids.object_dof
        speed = float(np.linalg.norm(data.qvel[dof : dof + 3]))
        self.object_speed_max = max(self.object_speed_max, speed)
        if speed > score.speed_cap:
            self._flag(FLAG_SPEED_CAP, t, phase, f"{speed:.2f} m/s")

        pos = data.xpos[self.ids.object_body]
        if pos[2] < self.task.physics.table_top_z - score.off_table_drop:
            self._flag(FLAG_OFF_TABLE, t, phase)

        if phase == Phase.CLOSE:
            if self._close_ref_xy is None:
                self._close_ref_xy = pos[:2].copy()
            drift = float(np.linalg.norm(pos[:2] - self._close_ref_xy))
            self.close_xy_drift_max = max(self.close_xy_drift_max, drift)
            if drift > score.eject_dist:
                self._flag(FLAG_EJECTION, t, phase, f"drift {drift * 1000:.1f} mm")
            elif speed > score.eject_speed:
                self._flag(FLAG_EJECTION, t, phase, f"speed {speed:.2f} m/s")
