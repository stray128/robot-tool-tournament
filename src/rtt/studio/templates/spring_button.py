"""The ``spring_button`` template: a spring-return push button in a bezel (Studio v2, rigid).

The hotel job: press the elevator call button. A wall button faces sideways and is pressed with
``push`` from the front (the step's ``approach="front"``: the wrist tilts the tool square to it);
a desk button faces up and is pressed from the top.

Button frame: the button's outward normal is its ``+z``. A ``wall`` mount stands a plate on the
table at the pose (its front face through the pose, facing local ``-y``, ``yaw_deg`` turning it),
the button's centre ``height_mm`` above the table; a ``desk`` mount is a console block
(:data:`CONSOLE` square, ``height_mm`` tall) with the button on its top.

* **Bezel**: the plate (:data:`PLATE_W` wide, :data:`PLATE_T` thick, reaching :data:`PLATE_TOP`
  above the button [assumed]) or the console, with a square opening :data:`BEZEL_GAP` wider than
  the button each side [assumed].
* **Button**: a cylinder ``diameter_mm`` across and :data:`CAP_LEN` long [assumed], its face
  ``proud_mm`` out of the bezel at rest, on a slide into the bezel (range ``travel_mm``) with a
  return spring preloaded to :data:`PRELOAD_FRAC` of ``actuation_force_n`` against its rest stop
  and rising to ``actuation_force_n`` at its **actuation point**, :data:`ACTUATE_FRAC` of its
  travel [assumed] (so it does not move under a lighter press, and a press that stops short of the
  actuation point does not call). A button standing out less than its actuation point can only be
  pressed by a tool narrower than the opening (a broad pad stops on the bezel).
* **Latch** (the call, a light slide with no spring and :data:`LATCH_FRICTION` [assumed]): a
  tendon pushes it along once the button passes its actuation point, and it stays, so a press is
  recorded after the button springs back (the template state is read when the step ends).

Sites: ``button`` (the button's face centre: where a push presses) and the auxiliary ``face``
(the bezel's face at the button, static). ``info``: the outward normal in the world
(``normal_x/y/z``), the travel, the actuation point, the sizes and the mount (0 wall, 1 desk).

State: ``travel_mm`` (how far the button is in now) and ``pressed`` (1 once it has been pressed
past its actuation point: the latch moved at least :data:`LATCH_ON`).
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.studio.templates import (
    FRAME_RGBA,
    FRICTION_SOLIMP,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["ACTUATE_FRAC", "SpringButtonBuilder"]

ACTUATE_FRAC: Final = 0.7  # of the travel: where the switch closes [assumed]
PRELOAD_FRAC: Final = 0.3  # of the actuation force: the spring's preload at rest [assumed]
BEZEL_GAP: Final = 0.001  # m between the button and its opening, each side [assumed]
CAP_LEN: Final = 0.015  # m [assumed]
CAP_DENSITY: Final = 1200.0  # kg/m^3 [assumed]
CAP_DAMPING: Final = 0.5  # N s/m [assumed]
PLATE_W: Final = 0.160  # m [assumed]
PLATE_T: Final = 0.020  # m [assumed]
PLATE_TOP: Final = 0.060  # m above the button's centre [assumed]
CONSOLE: Final = 0.120  # m square [assumed]
LATCH_MAX: Final = 0.002  # m [assumed]
LATCH_FRICTION: Final = 0.05  # N [assumed]
LATCH_ON: Final = 0.0002  # m of latch travel that counts as a call [assumed]
MOUNT_CODES: Final = {"wall": 0.0, "desk": 1.0}
BUTTON_RGBA: Final = (0.85, 0.80, 0.55, 1.0)


def _quat(yaw: float, wall: bool) -> list[float]:
    """The button frame in the world: turned by ``yaw``; a wall button's +z along local -y."""
    qz = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
    if not wall:
        return qz.tolist()
    qx = np.array([math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0])  # +90 deg about x
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, qx)
    return out.tolist()


class SpringButtonBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.SPRING_BUTTON

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        wall = prm["mount"] == "wall"
        d = float(prm["diameter_mm"]) / 1000.0
        travel = float(prm["travel_mm"]) / 1000.0
        force = float(prm["actuation_force_n"])
        proud = float(prm["proud_mm"]) / 1000.0
        height = float(prm["height_mm"]) / 1000.0
        yaw = math.radians(pose.yaw_deg)
        quat = _quat(yaw, wall)
        mount = ctx.spec.worldbody.add_body(
            name=f"{p}mount",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z + height],
            quat=quat,
        )
        geoms: list[str] = []

        def box(name: str, lo: list[float], hi: list[float]) -> None:
            mount.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.5 * (b - a) for a, b in zip(lo, hi, strict=True)],
                pos=[0.5 * (a + b) for a, b in zip(lo, hi, strict=True)],
                priority=1,
                rgba=list(FRAME_RGBA),
            )
            geoms.append(f"{p}{name}")

        # the bezel around a square opening (button frame: x, y across the face, z out of it)
        r = 0.5 * d + BEZEL_GAP
        if wall:
            hx, y_lo, y_hi, depth = 0.5 * max(PLATE_W, d + 0.08), -height, PLATE_TOP, PLATE_T
        else:
            hx = 0.5 * max(CONSOLE, d + 0.04)
            y_lo, y_hi, depth = -hx, hx, height
        box("bezel_l", [-hx, y_lo, -depth], [-r, y_hi, 0.0])
        box("bezel_r", [r, y_lo, -depth], [hx, y_hi, 0.0])
        box("bezel_b", [-r, y_lo, -depth], [r, -r, 0.0])
        box("bezel_t", [-r, r, -depth], [r, y_hi, 0.0])
        mount.add_site(name=f"{p}face", pos=[0.0, 0.0, 0.0], size=[0.004, 0.0, 0.0])

        # the button: a preloaded return spring on a slide into the bezel
        x_act = ACTUATE_FRAC * travel
        preload = PRELOAD_FRAC * force
        stiffness = (force - preload) / x_act
        cap = mount.add_body(name=f"{p}cap", pos=[0.0, 0.0, 0.0])
        cap.add_joint(
            name=f"{p}cap",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, -1.0],
            range=[0.0, travel],
            stiffness=stiffness,
            springref=-preload / stiffness,
            damping=CAP_DAMPING,
            solref_limit=list(ctx.task.physics.limit_solref),
            solimp_limit=list(FRICTION_SOLIMP),
        )
        cap.add_geom(
            name=f"{p}cap",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.5 * d, 0.5 * CAP_LEN, 0.0],
            pos=[0.0, 0.0, proud - 0.5 * CAP_LEN],
            density=CAP_DENSITY,
            priority=1,
            rgba=list(BUTTON_RGBA),
        )
        geoms.append(f"{p}cap")
        cap.add_site(name=f"{p}button", pos=[0.0, 0.0, proud], size=[0.004, 0.0, 0.0])

        # the latch: pushed along once the button passes its actuation point, and it stays
        latch = mount.add_body(name=f"{p}latch", pos=[0.0, 0.0, -depth + 0.003])
        latch.add_joint(
            name=f"{p}latch",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, -1.0],
            range=[0.0, LATCH_MAX],
            frictionloss=LATCH_FRICTION,
            damping=0.05,
            armature=0.001,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )
        latch.add_geom(
            name=f"{p}latch",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.001, 0.0, 0.0],
            mass=0.001,
            contype=0,
            conaffinity=0,
            group=3,
            priority=1,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        geoms.append(f"{p}latch")
        tendon = ctx.spec.add_tendon(name=f"{p}latch")
        tendon.wrap_joint(f"{p}latch", 1.0)
        tendon.wrap_joint(f"{p}cap", -1.0)
        tendon.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        tendon.range = [-x_act, 1.0]  # latch >= button - actuation point
        tendon.solref_limit = list(ctx.task.physics.limit_solref)
        tendon.solimp_limit = list(FRICTION_SOLIMP)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.array(quat))
        normal = rot.reshape(3, 3)[:, 2]
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}mount", f"{p}cap", f"{p}latch"),
            geoms=tuple(geoms),
            joints={"cap": f"{p}cap", "latch": f"{p}latch"},
            sites={"button": f"{p}button", "face": f"{p}face"},
            info={
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "travel_m": travel,
                "actuate_m": x_act,
                "diameter_m": d,
                "proud_m": proud,
                "opening_m": 2.0 * r,
                "force_n": force,
                "mount": MOUNT_CODES[str(prm["mount"])],
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        info = obj.handles.info
        travel = min(max(float(data.qpos[obj.joint_qpos["cap"]]), 0.0), info["travel_m"])
        latched = float(data.qpos[obj.joint_qpos["latch"]]) >= LATCH_ON
        return {"travel_mm": 1000.0 * travel, "pressed": 1.0 if latched else 0.0}


register_template_builder(SpringButtonBuilder())
