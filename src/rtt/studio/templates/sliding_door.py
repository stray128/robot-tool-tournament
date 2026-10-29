"""The ``sliding_door`` template: a machine enclosure whose door slides sideways (Studio v2, rigid).

The machine-tending job: slide the door open, load the part into the fixture inside (a ``bin``
placed under the opening, at this template's pose), slide the door shut. The door reuses the
``drawer``'s machinery (:data:`~rtt.studio.types.SLIDE_KINDS`): the same slide, friction and end
stops, the same pull, push and observed travel, and a handle the primitives treat as the drawer's
``lip`` (a plate standing up, gripped across its thickness along the slide).

Instance frame: origin at the pose on the table top, at the centre of the loading opening, turned
by ``yaw_deg``; the door slides along local ``-y`` to open (as a drawer pulls out); the machine's
front faces local ``-x``. The loading is from above, as the Studio's wrist works: the door has a
roof leaf over the roof opening and a front leaf over the doorway (the shape of a machine door
that opens into its roof for overhead loading), so a closed door blocks the load.

* **Enclosure** (static): walls :data:`WALL` thick and a roof :data:`ROOF_T` thick [assumed],
  ``depth_mm`` (x) by ``height_mm``, long enough along y to park the open door over it. The
  opening is ``width_mm`` along y and the interior's depth along x, through the roof and the front
  wall (an L); the interior under it is clear to the table, where a fixture stands.
* **Door** (one body on a slide along local ``-y``, range ``[0, travel_mm]``, 0 closed): a front
  leaf and a roof leaf :data:`LEAF_T` thick, :data:`GAP` clear of the enclosure, :data:`OVERLAP`
  past the opening each side [assumed], density :data:`DENSITY` [assumed]. Its friction loss is
  ``pull_force_n`` (both ways, a near-hard dry friction, so a smaller force does not creep it),
  plus a light damping (the drawer's).
* **Handle**: a fin :data:`FIN_T` thick (along the slide) and :data:`FIN_L` long standing
  ``handle_mm`` up from the roof leaf at the opening's centre [assumed].

Sites: ``door`` (on the door at the roof leaf's top centre: the travel reference; where a push to
close is aimed it uses the handle), ``handle`` (the fin's centre: the grip) and the auxiliaries
``front`` (the same point on the enclosure, static) and ``opening`` (the roof opening's centre).
``info``: the drawer's slide keys (``pull_x``/``pull_y``, ``travel_m``, ``handle`` = 2, ``grip_m``,
``push_out_m``, ``front_gap_m`` = 0, ``pull_force_n``) and the geometry, the interior included.

State: ``travel_mm``, how far the door is open (0 closed), within the slide's stops.
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
from rtt.studio.templates.drawer import HANDLE_CODES
from rtt.studio.types import TemplateKind

__all__ = ["FIN_T", "SlidingDoorBuilder"]

WALL: Final = 0.010  # m [assumed]
ROOF_T: Final = 0.010  # m [assumed]
LEAF_T: Final = 0.006  # m [assumed]
GAP: Final = 0.002  # m between the door and the enclosure [assumed]
OVERLAP: Final = 0.010  # m of door past the opening each side [assumed]
DENSITY: Final = 1200.0  # kg/m^3 [assumed]
DAMPING: Final = 2.0  # N s/m on the slide (the drawer's) [assumed]
FIN_T: Final = 0.006  # m [assumed]
FIN_L: Final = 0.080  # m [assumed]
DOOR_RGBA: Final = (0.62, 0.74, 0.82, 0.85)


class SlidingDoorBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.SLIDING_DOOR

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        w, dx, h = (float(prm[k]) / 1000.0 for k in ("width_mm", "depth_mm", "height_mm"))
        travel = float(prm["travel_mm"]) / 1000.0
        fin_h = float(prm["handle_mm"]) / 1000.0
        yaw = math.radians(pose.yaw_deg)
        frame = ctx.spec.worldbody.add_body(
            name=f"{p}frame", pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z], quat=quat_z(yaw)
        )
        geoms: list[str] = []

        def box(
            body: mujoco.MjsBody,
            name: str,
            lo: list[float],
            hi: list[float],
            rgba: tuple[float, ...] = FRAME_RGBA,
        ) -> None:
            kw = {"density": DENSITY} if body is not frame else {}
            body.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.5 * (b - a) for a, b in zip(lo, hi, strict=True)],
                pos=[0.5 * (a + b) for a, b in zip(lo, hi, strict=True)],
                priority=1,
                rgba=list(rgba),
                **kw,
            )
            geoms.append(f"{p}{name}")

        # the enclosure: long enough along y to park the open door over it
        hx, hw = 0.5 * dx, 0.5 * w
        y0, y1 = -hw - OVERLAP - travel - WALL, hw + OVERLAP + WALL
        z_roof = h - ROOF_T
        box(frame, "roof_a", [-hx, y0, z_roof], [hx, -hw, h])
        box(frame, "roof_b", [-hx, hw, z_roof], [hx, y1, h])
        box(frame, "front_a", [-hx, y0, 0.0], [-hx + WALL, -hw, z_roof])
        box(frame, "front_b", [-hx, hw, 0.0], [-hx + WALL, y1, z_roof])
        box(frame, "back", [hx - WALL, y0, 0.0], [hx, y1, h])
        box(frame, "side_a", [-hx + WALL, y0, 0.0], [hx - WALL, y0 + WALL, z_roof])
        box(frame, "side_b", [-hx + WALL, y1 - WALL, 0.0], [hx - WALL, y1, z_roof])

        # the door: a front leaf and a roof leaf on a slide along local -y (a drawer's slide)
        door = frame.add_body(name=f"{p}door", pos=[0.0, 0.0, 0.0])
        door.add_joint(
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
        ly0, ly1 = -hw - OVERLAP, hw + OVERLAP
        x_out = -hx - GAP - LEAF_T  # the front leaf's outer face
        top = h + GAP + LEAF_T  # the roof leaf's top
        box(door, "roof_leaf", [x_out, ly0, h + GAP], [hx, ly1, top], DOOR_RGBA)
        box(door, "front_leaf", [x_out, ly0, GAP], [-hx - GAP, ly1, top], DOOR_RGBA)
        box(
            door,
            "fin",
            [-0.5 * FIN_L, -0.5 * FIN_T, top],
            [0.5 * FIN_L, 0.5 * FIN_T, top + fin_h],
            OBJECT_RGBA,
        )
        door.add_site(name=f"{p}handle", pos=[0.0, 0.0, top + 0.5 * fin_h], size=[0.004, 0.0, 0.0])
        door.add_site(name=f"{p}door", pos=[0.0, 0.0, top], size=[0.004, 0.0, 0.0])
        frame.add_site(name=f"{p}front", pos=[0.0, 0.0, top], size=[0.004, 0.0, 0.0])
        frame.add_site(name=f"{p}opening", pos=[0.0, 0.0, h], size=[0.004, 0.0, 0.0])
        c, s = math.cos(yaw), math.sin(yaw)
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}frame", f"{p}door"),
            geoms=tuple(geoms),
            joints={"slide": f"{p}slide"},
            sites={
                "door": f"{p}door",
                "handle": f"{p}handle",
                "front": f"{p}front",
                "opening": f"{p}opening",
            },
            info={
                "pull_x": float(s),  # local -y in the world
                "pull_y": float(-c),
                "travel_m": travel,
                "handle": HANDLE_CODES["lip"],  # a plate standing up, gripped across its thickness
                "handle_m": fin_h,
                "grip_m": FIN_T,
                "push_out_m": 0.5 * FIN_T,  # from the handle site to the fin's face a push meets
                "front_gap_m": 0.0,
                "pull_force_n": float(prm["pull_force_n"]),
                "width_m": w,
                "depth_m": dx,
                "height_m": h,
                "roof_top_z": float(ctx.table_top_z + top),
                "interior_x_m": dx - 2.0 * WALL,
                "interior_h_m": z_roof,
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        travel = float(data.qpos[obj.joint_qpos["slide"]])  # within the slide's own stops
        return {"travel_mm": 1000.0 * min(max(travel, 0.0), obj.handles.info["travel_m"])}


register_template_builder(SlidingDoorBuilder())
