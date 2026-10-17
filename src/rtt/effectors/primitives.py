"""Shared building blocks for native families: contact readouts, a synergy primitive, a contact test.

* :func:`object_contacts`: every active effector-part contact with its force on the part.
* :class:`SynergyPrimitive`: the scripted power/pinch primitive. The effector's controls move
  linearly from an open vector to a closed vector over the shared close ramp (the jaw's grip
  target ramps the same way), and stay closed through the lift and hold. A catalog hand's
  power grasp and pinch grasp are two synergies of the same model.
* :class:`ContactCriterion`: engaged when at least ``min_bodies`` effector bodies touch the
  part and their net forces on it oppose each other by at least ``min_opposition_deg``.

Families may use these or bring their own (a vacuum switch, a wrap test), as long as they keep
the :class:`~rtt.effectors.types.GraspPrimitive` and :class:`~rtt.effectors.types.GraspCriterion`
protocols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from rtt.effectors.types import EffectorBuild, Engagement, PrimitivePlan
from rtt.sim import objects as objs

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.controller import Phase
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2

__all__ = ["ContactCriterion", "ObjectContact", "SynergyPrimitive", "object_contacts", "part_height"]


@dataclass(frozen=True)
class ObjectContact:
    """One active contact between an effector geom and the part (forces in N, world frame)."""

    body: int  # effector body id
    geom: int  # effector geom id
    normal_force: float
    force_on_part: np.ndarray  # (3,) normal plus friction force the effector applies to the part
    depth: float  # m, > 0 when the geoms overlap


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> list[ObjectContact]:
    """Active effector-part contacts (uses ``mj_contactForce``; call at check ticks, not every step)."""
    ids = scene.ids
    if not data.ncon:
        return []
    eff = np.zeros(model.ngeom, dtype=bool)
    eff[ids.effector_geoms] = True
    obj = np.zeros(model.ngeom, dtype=bool)
    obj[ids.object_geoms] = True
    out = []
    wrench = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.efc_address < 0:
            continue
        g0, g1 = int(con.geom[0]), int(con.geom[1])
        if eff[g0] and obj[g1]:
            g_eff, sign = g0, 1.0
        elif obj[g0] and eff[g1]:
            g_eff, sign = g1, -1.0
        else:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = con.frame.reshape(3, 3)
        force = sign * (frame.T @ wrench[:3])  # the force on geom[1] is along +normal
        out.append(
            ObjectContact(
                body=int(model.geom_bodyid[g_eff]),
                geom=g_eff,
                normal_force=float(wrench[0]),
                force_on_part=force,
                depth=float(max(0.0, -con.dist)),
            )
        )
    return out


def part_height(part: PartSpec) -> float:
    """The part's height on the table (m)."""
    return float(objs.height(part.to_mesh_object()))


@dataclass(frozen=True)
class SynergyPrimitive:
    """Open-to-closed control synergy on the shared close ramp (module docstring).

    * ``ctrl_open``/``ctrl_closed``: one value per effector actuator (``build.actuators``).
    * ``grip_height_frac``: the effector's lowest point goes to this fraction of the part's
      height (plus ``tip_offset`` m), never below the task's minimum tip clearance.
    * ``yaw_offset``: rad, the wrist turned relative to the part (the part frame still sets
      placement errors and disturbances).
    """

    name: str
    ctrl_open: tuple[float, ...]
    ctrl_closed: tuple[float, ...]
    grip_height_frac: float = 0.5
    tip_offset: float = 0.0
    yaw_offset: float = 0.0

    def __post_init__(self) -> None:
        a = np.asarray(self.ctrl_open, dtype=np.float64)
        b = np.asarray(self.ctrl_closed, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("ctrl_open and ctrl_closed must be finite vectors of the same length")
        if not 0.0 <= self.grip_height_frac <= 1.0:
            raise ValueError("grip_height_frac must be in [0, 1]")
        object.__setattr__(self, "ctrl_open", tuple(float(v) for v in a))
        object.__setattr__(self, "ctrl_closed", tuple(float(v) for v in b))

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        if len(self.ctrl_open) != build.n_actuators:
            raise ValueError(f"{self.name}: {len(self.ctrl_open)} controls for {build.n_actuators} actuators")
        tip = max(
            task.controller.min_tip_clearance, self.grip_height_frac * part_height(part) + self.tip_offset
        )
        return PrimitivePlan(tip_height=tip, yaw_offset=self.yaw_offset)

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        a = np.asarray(self.ctrl_open)
        return a + closure * (np.asarray(self.ctrl_closed) - a)


@dataclass(frozen=True)
class ContactCriterion:
    """Engaged when ``min_bodies`` effector bodies touch the part with opposing net forces.

    ``grasped`` (labels only) asks for the same body count. ``engaged`` reports
    ``few_contacts`` or ``no_opposition`` and the metrics ``bodies``, ``contacts`` and
    ``opposition_deg`` (the largest angle between two bodies' net forces on the part).
    """

    name: str = "contacts"
    min_bodies: int = 2
    min_opposition_deg: float = 90.0
    min_force: float = 0.05  # N; a body counts when its normal forces on the part sum above this

    def _per_body(
        self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> tuple[dict[int, np.ndarray], int]:
        forces: dict[int, np.ndarray] = {}
        normal: dict[int, float] = {}
        contacts = object_contacts(model, data, scene)
        for c in contacts:
            forces[c.body] = forces.get(c.body, np.zeros(3)) + c.force_on_part
            normal[c.body] = normal.get(c.body, 0.0) + c.normal_force
        return {b: f for b, f in forces.items() if normal[b] > self.min_force}, len(contacts)

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool:
        return len(self._per_body(model, data, scene)[0]) >= self.min_bodies

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement:
        per_body, n_contacts = self._per_body(model, data, scene)
        vecs = [f / n for f in per_body.values() if (n := float(np.linalg.norm(f))) > 0]
        best = 0.0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                best = max(best, math.degrees(math.acos(max(-1.0, min(1.0, float(vecs[i] @ vecs[j]))))))
        metrics = {"bodies": float(len(per_body)), "contacts": float(n_contacts), "opposition_deg": best}
        if len(per_body) < self.min_bodies:
            return Engagement(False, ("few_contacts",), metrics)
        if best < self.min_opposition_deg:
            return Engagement(False, ("no_opposition",), metrics)
        return Engagement(True, (), metrics)
