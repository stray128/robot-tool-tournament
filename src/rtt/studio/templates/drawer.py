"""The ``drawer`` template: a box on a prismatic slide in a fixed carcass (track S2, fidelity rigid).

Instance frame: origin at the pose on the table top, rotated by ``yaw_deg``; the drawer front
faces local ``-y`` and pulls out along ``-y``.

* **Carcass** (static): a box ``width_mm`` x ``depth_mm`` x ``height_mm`` with :data:`WALL`
  walls (bottom on the table, top, sides, back), open at the front.
* **Drawer** (one body on a slide along local ``-y``): an overlay front ``width_mm`` x
  ``height_mm`` x :data:`FRONT` just in front of the carcass, and behind it a box (sides, back,
  floor :data:`BOX_WALL` thick, open on top) in the carcass with :data:`CLEARANCE` all round;
  density :data:`DENSITY`. The slide's range is ``[0, travel_mm]`` (0 closed; stops at both
  ends); its friction loss is ``pull_force_n`` (the force that resists sliding, both ways; a
  near-hard dry friction, :data:`FRICTION_SOLIMP`, so a smaller push does not creep it), plus a
  light damping. The slide holds the drawer level, so a travel longer than the drawer does not
  drop it (the template honours ``travel_mm`` as given).
* **Handle** on the front, centred across [assumed geometry]:

  * ``bar``: a horizontal bar (:data:`BAR_DIAMETER`, :data:`BAR_LENGTH` or 60 % of the width)
    on two posts, ``handle_mm`` clear of the front face, its centre ``min(height / 2, 30 mm)``
    below the top edge (high, so a top-down tool's wrist stays above the carcass);
  * ``knob``: a round knob at the same height: a neck :data:`KNOB_NECK` across and
    ``handle_mm`` long, with a head :data:`KNOB_HEAD` across and 6 mm thick;
  * ``lip``: a plate :data:`LIP_T` thick standing ``handle_mm`` up from the front's top edge,
    flush with its outer face, 80 % of the width: a pull a gripper pinches from above.

Sites: ``drawer`` (centre of the front face: where a push closes it), ``handle`` (the grip: the
bar's, the neck's or the lip plate's centre) and the auxiliary ``front`` (the carcass front at
the drawer's centre, static). ``info``: the pull direction in the world (``pull_x/y``), the
travel, the handle code (0 bar, 1 knob, 2 lip), ``grip_m`` (the grip's thickness across a
gripper's closing axis) and geometry.

State: ``travel_mm``, how far the drawer is out (0 closed), within the slide's stops.
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco

from rtt.sim.scene import OBJECT_RGBA
from rtt.studio.templates import (
    FRAME_RGBA,
    FRICTION_SOLIMP,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    quat_z,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["BAR_DIAMETER", "CLEARANCE", "WALL", "DrawerBuilder"]

WALL: Final = 0.010  # m, carcass walls [assumed]
CLEARANCE: Final = 0.002  # m, drawer to carcass on every side [assumed]
FRONT: Final = 0.012  # m, drawer front thickness [assumed]
BOX_WALL: Final = 0.006  # m, drawer sides, back and bottom [assumed]
DENSITY: Final = 700.0  # kg/m^3 [assumed]
DAMPING: Final = 2.0  # N s/m on the slide [assumed]
BAR_DIAMETER: Final = 0.010
BAR_LENGTH: Final = 0.120
KNOB_NECK: Final = 0.012
KNOB_HEAD: Final = 0.026
LIP_T: Final = 0.004
HANDLE_CODES: Final = {"bar": 0.0, "knob": 1.0, "lip": 2.0}


class DrawerBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.DRAWER

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        w, h, d = (float(prm[k]) / 1000.0 for k in ("width_mm", "height_mm", "depth_mm"))
        travel = float(prm["travel_mm"]) / 1000.0
        hm = float(prm["handle_mm"]) / 1000.0
        kind = str(prm["handle"])
        yaw = math.radians(pose.yaw_deg)
        origin = [pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z]
        carcass = ctx.spec.worldbody.add_body(name=f"{p}carcass", pos=origin, quat=quat_z(yaw))
        geoms: list[str] = []

        def box(body: mujoco.MjsBody, name: str, lo: list[float], hi: list[float], moving: bool) -> None:
            c = [0.5 * (a + b) for a, b in zip(lo, hi, strict=True)]
            half = [0.5 * (b - a) for a, b in zip(lo, hi, strict=True)]
            kw = {"density": DENSITY} if moving else {}
            body.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half,
                pos=c,
                priority=1,
                rgba=list(OBJECT_RGBA if moving else FRAME_RGBA),
                **kw,
            )
            geoms.append(f"{p}{name}")

        # carcass: outer x in +-w/2, y in [-d/2, d/2], z in [0, h]; open at the front (-y)
        hw, hd = 0.5 * w, 0.5 * d
        box(carcass, "bottom", [-hw, -hd, 0.0], [hw, hd, WALL], False)
        box(carcass, "top", [-hw, -hd, h - WALL], [hw, hd, h], False)
        box(carcass, "left", [-hw, -hd, WALL], [-hw + WALL, hd, h - WALL], False)
        box(carcass, "right", [hw - WALL, -hd, WALL], [hw, hd, h - WALL], False)
        box(carcass, "back", [-hw + WALL, hd - WALL, WALL], [hw - WALL, hd, h - WALL], False)

        # the drawer: an overlay front just in front of the carcass and a box in the cavity
        drawer = carcass.add_body(name=f"{p}drawer", pos=[0.0, 0.0, 0.0])
        drawer.add_joint(
            name=f"{p}slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, -1.0, 0.0],
            range=[0.0, travel],
            frictionloss=float(prm["pull_force_n"]),
            damping=DAMPING,
            armature=0.01,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )
        fy = -hd - 0.001 - FRONT  # the front face
        box(drawer, "front", [-hw, fy, 0.001], [hw, fy + FRONT, h], True)
        bx, by0, by1 = hw - WALL - CLEARANCE, fy + FRONT, hd - WALL - CLEARANCE
        bz0, bz1 = WALL + CLEARANCE, h - WALL - CLEARANCE
        box(drawer, "floor", [-bx, by0, bz0], [bx, by1, bz0 + BOX_WALL], True)
        box(drawer, "side_l", [-bx, by0, bz0], [-bx + BOX_WALL, by1, bz1], True)
        box(drawer, "side_r", [bx - BOX_WALL, by0, bz0], [bx, by1, bz1], True)
        box(drawer, "rear", [-bx + BOX_WALL, by1 - BOX_WALL, bz0], [bx - BOX_WALL, by1, bz1], True)

        zc = h - min(0.5 * h, 0.030)  # handle centre height
        if kind == "bar":
            length = min(BAR_LENGTH, 0.6 * w)
            r = 0.5 * BAR_DIAMETER
            yb = fy - hm - r
            drawer.add_geom(
                name=f"{p}bar",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[-0.5 * length, yb, zc, 0.5 * length, yb, zc],
                size=[r, 0.0, 0.0],
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(f"{p}bar")
            for k, x in enumerate((-0.5 * length, 0.5 * length)):
                box(drawer, f"post{k}", [x - 0.004, yb, zc - 0.004], [x + 0.004, fy, zc + 0.004], True)
            handle = [0.0, yb, zc]
            width = BAR_DIAMETER
        elif kind == "knob":
            rn, rh = 0.5 * KNOB_NECK, 0.5 * KNOB_HEAD
            drawer.add_geom(
                name=f"{p}neck",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[0.0, fy - hm, zc, 0.0, fy, zc],
                size=[rn, 0.0, 0.0],
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            drawer.add_geom(
                name=f"{p}knob",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[0.0, fy - hm - 0.006, zc, 0.0, fy - hm, zc],
                size=[rh, 0.0, 0.0],
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms += [f"{p}neck", f"{p}knob"]
            handle = [0.0, fy - 0.5 * hm, zc]
            width = KNOB_NECK
        else:  # lip: a plate standing up from the front's top edge, flush with its outer face
            lw = 0.8 * w
            box(drawer, "lip", [-0.5 * lw, fy, h], [0.5 * lw, fy + LIP_T, h + hm], True)
            handle = [0.0, fy + 0.5 * LIP_T, h + 0.5 * hm]
            width = LIP_T
        drawer.add_site(name=f"{p}handle", pos=handle, size=[0.004, 0.0, 0.0])
        drawer.add_site(name=f"{p}drawer", pos=[0.0, fy, 0.5 * h], size=[0.004, 0.0, 0.0])
        carcass.add_site(name=f"{p}front", pos=[0.0, -hd, 0.5 * h], size=[0.004, 0.0, 0.0])
        c, s = math.cos(yaw), math.sin(yaw)
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}carcass", f"{p}drawer"),
            geoms=tuple(geoms),
            joints={"slide": f"{p}slide"},
            sites={"drawer": f"{p}drawer", "handle": f"{p}handle", "front": f"{p}front"},
            info={
                "pull_x": float(s),  # local -y in the world
                "pull_y": float(-c),
                "travel_m": travel,
                "handle": HANDLE_CODES[kind],
                "handle_m": hm,
                "width_m": w,
                "height_m": h,
                "depth_m": d,
                "bar_radius_m": 0.5 * BAR_DIAMETER,
                "knob_neck_m": KNOB_NECK,
                "knob_head_m": KNOB_HEAD,
                "front_top_z": float(ctx.table_top_z + h),
                "grip_m": width,
                "front_gap_m": FRONT + 0.001,  # the front face ahead of the carcass front when closed
                # from the handle site to the handle's front (a push meets it there)
                "push_out_m": {"bar": 0.5 * BAR_DIAMETER, "knob": 0.5 * hm + 0.006, "lip": 0.5 * LIP_T}[kind],
                "pull_force_n": float(prm["pull_force_n"]),
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        travel = float(data.qpos[obj.joint_qpos["slide"]])  # within the slide's own stops
        return {"travel_mm": 1000.0 * min(max(travel, 0.0), obj.handles.info["travel_m"])}


register_template_builder(DrawerBuilder())
