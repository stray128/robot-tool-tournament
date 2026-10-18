"""The ``port_plug`` template: a fixed socket with a chamfered bore and a free plug (track S2).

Fidelity: rigid (contact between the plug and the bore decides the insertion).

* **Socket** (static, vertical, opening up), at the instance pose: a floor :data:`FLOOR` thick
  and a wall of :data:`BORE_SIDES` convex pieces around a polygonal bore whose flats are
  ``(plug_diameter_mm + clearance_mm) / 2`` from the axis (the diametral clearance at the flats),
  ``depth_mm`` deep, its top inner edge cut by a 45 deg ``chamfer_mm`` lead-in (at most 80 % of
  the depth). Wall :data:`WALL` thick at least.
* **Plug** (a free body): a cylinder ``plug_diameter_mm`` x ``plug_length_mm`` with a
  :data:`TIP_CHAMFER` chamfer on its tip, standing upright on the table :data:`PLUG_GAP` beside the
  socket's outer edge along the instance's local ``+x`` [assumed start pose and tip chamfer].
  Density :data:`DENSITY`.
* **Insertion friction** (``insert_force_n``): :data:`N_PADS` spring-loaded contact pads in the
  socket wall just below the chamfer (capsules on radial slides, preloaded against an inner stop,
  :data:`PAD_PRELOAD`), so a centred plug pushes each out by 0.4 mm; their springs are set so the
  pads' contact friction on the plug (the task friction, jittered) is ``insert_force_n``. The
  friction is a real contact friction (MuJoCo's, implicit), from the pads' entry to seated.
* **Latch** (``latch``): the plug has a 0.8 mm deep groove where the pads sit when it is seated;
  they snap into it (the click) and hold it: pulling it out has to push them back over the
  groove's edge [assumed geometry].

Sites: ``plug`` (the plug's top centre: where a tool grips or pushes it), ``socket`` (the bore's
mouth centre, static) and the auxiliary ``plug_tip`` (the plug's bottom centre). ``info``: the
bore, depths and heights, the plug's size and start position.

State: ``depth_mm``, how deep the plug's tip is below the socket mouth while it is in the bore
(0 outside); within :data:`SEAT_TOL` of the floor it reads as seated (``depth_mm`` exactly), so
the floor contact's compliance (a few hundredths of a millimetre of springback when a press lets
go) does not decide whether a seated plug passes a full-depth criterion.
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.sim.scene import OBJECT_RGBA
from rtt.studio.templates import (
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["BORE_SIDES", "FLOOR", "N_PADS", "PortPlugBuilder"]

BORE_SIDES: Final = 16
FLOOR: Final = 0.006  # m [assumed]
WALL: Final = 0.006  # m, at least [assumed]
PLUG_GAP: Final = 0.040  # m, plug start beside the socket [assumed]
DENSITY: Final = 1400.0  # kg/m^3, a moulded connector [assumed]
N_PADS: Final = 3
PAD_RADIUS: Final = 0.0015  # m, capsule radius [assumed]
PAD_MASS: Final = 0.002  # kg
PAD_PRELOAD: Final = 0.002  # m: the spring's rest is this far past the pad's inner stop
PAD_SQUEEZE: Final = 0.0004  # m: a centred plug pushes each pad out this far
GROOVE_DEPTH: Final = 0.0008  # m
TIP_CHAMFER: Final = 0.0006  # m, the plug's own tip chamfer [assumed]
SEAT_TOL: Final = 0.0001  # m: a plug this close to the floor reads as seated (full depth)
FRAME_RGBA: Final = (0.55, 0.58, 0.62, 1.0)


def _quat_z(angle: float) -> list[float]:
    return [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]


class PortPlugBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.PORT_PLUG

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        dia, plen = float(prm["plug_diameter_mm"]) / 1000.0, float(prm["plug_length_mm"]) / 1000.0
        clearance = float(prm["clearance_mm"]) / 1000.0
        depth = float(prm["depth_mm"]) / 1000.0
        chamfer = min(float(prm["chamfer_mm"]) / 1000.0, 0.8 * depth)
        r_bore = 0.5 * (dia + clearance)  # apothem of the polygonal bore
        r_out = r_bore + max(WALL, 0.5 * dia) + chamfer
        top = FLOOR + depth
        yaw = math.radians(pose.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        origin = np.array([pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z])
        socket = ctx.spec.worldbody.add_body(name=f"{p}socket", pos=origin.tolist(), quat=_quat_z(yaw))
        socket.add_geom(
            name=f"{p}floor",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r_out, 0.5 * FLOOR, 0.0],
            pos=[0.0, 0.0, 0.5 * FLOOR],
            priority=1,
            rgba=list(FRAME_RGBA),
        )
        geoms = [f"{p}floor"]
        half = math.pi / BORE_SIDES
        for k in range(BORE_SIDES):
            a = 2.0 * math.pi * k / BORE_SIDES
            verts = []
            for b in (a - half, a + half):
                cb, sb = math.cos(b), math.sin(b)
                k_in = r_bore / math.cos(half)  # corner radius of the bore polygon
                k_out = r_out / math.cos(half)
                k_ch = (r_bore + chamfer) / math.cos(half)
                verts += [
                    [k_in * cb, k_in * sb, FLOOR],
                    [k_in * cb, k_in * sb, top - chamfer],
                    [k_ch * cb, k_ch * sb, top],
                    [k_out * cb, k_out * sb, top],
                    [k_out * cb, k_out * sb, FLOOR],
                ]
            mesh = f"{p}wall{k:02d}"
            ctx.spec.add_mesh(name=mesh, uservert=np.ravel(verts).tolist())
            socket.add_geom(
                name=mesh, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, priority=1, rgba=list(FRAME_RGBA)
            )
            geoms.append(mesh)
        socket.add_site(name=f"{p}socket", pos=[0.0, 0.0, top], size=[0.003, 0.0, 0.0])

        bodies = [f"{p}socket"]  # root first, the plug last (the body a pick or insert moves)
        f_ins = float(prm["insert_force_n"])
        latch = bool(prm["latch"])
        hl = max(0.0005, min(0.0025, 0.25 * depth - PAD_RADIUS))
        z_pad = max(FLOOR + hl + PAD_RADIUS + 0.0005, top - chamfer - 0.001 - hl - PAD_RADIUS)
        n_pads = N_PADS if (f_ins > 0.0 or latch) else 0
        mu = float(ctx.task.physics.friction)
        stiffness = (
            max(f_ins, 1.0 if latch else 0.0) / (n_pads * mu * (PAD_PRELOAD + PAD_SQUEEZE)) if n_pads else 0.0
        )
        reach = 0.5 * clearance + PAD_SQUEEZE  # how far a pad at its stop stands into the bore
        for k in range(n_pads):
            a = math.radians(90.0 + 120.0 * k)
            inward = [-math.cos(a), -math.sin(a), 0.0]
            r_axis = r_bore - reach + PAD_RADIUS
            pad = socket.add_body(name=f"{p}pad{k}", pos=[r_axis * math.cos(a), r_axis * math.sin(a), z_pad])
            pad.add_joint(
                name=f"{p}pad{k}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=inward,
                range=[-(reach + 0.002), 0.0],
                stiffness=stiffness,
                springref=PAD_PRELOAD,
                damping=2.0 * math.sqrt(stiffness * PAD_MASS),
                armature=0.0001,
            )
            pad.add_geom(
                name=f"{p}pad{k}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0.0, 0.0, -hl, 0.0, 0.0, hl],
                size=[PAD_RADIUS, 0.0, 0.0],
                mass=PAD_MASS,
                priority=1,
                rgba=list(FRAME_RGBA),
            )
            ctx.spec.add_exclude(bodyname1=f"{p}socket", bodyname2=f"{p}pad{k}")
            geoms.append(f"{p}pad{k}")
            bodies.append(f"{p}pad{k}")

        offset = r_out / math.cos(half) + PLUG_GAP + 0.5 * dia
        start = origin + np.array([c * offset, s * offset, 0.5 * plen])
        plug = ctx.spec.worldbody.add_body(name=f"{p}plug", pos=start.tolist(), quat=_quat_z(yaw))
        plug.add_freejoint(name=f"{p}free")
        bodies.append(f"{p}plug")
        rp = 0.5 * dia
        # the plug, tip at -plen/2; with a latch, a groove where the pads sit when it is seated
        g_mid = (z_pad - FLOOR) - 0.5 * plen
        g_half = hl + PAD_RADIUS + 0.0004
        cuts = [(-0.5 * plen, g_mid - g_half, rp), (g_mid - g_half, g_mid + g_half, rp - GROOVE_DEPTH)]
        cuts.append((g_mid + g_half, 0.5 * plen, rp))
        pieces = (
            cuts
            if latch and g_mid - g_half > -0.5 * plen + TIP_CHAMFER + 0.0005
            else [(-0.5 * plen, 0.5 * plen, rp)]
        )
        z_tip = -0.5 * plen
        pieces[0] = (z_tip + TIP_CHAMFER, pieces[0][1], pieces[0][2])  # the tip is a chamfered frustum below
        ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)]
        cone = [[(rp - TIP_CHAMFER) * cx, (rp - TIP_CHAMFER) * sx, z_tip] for cx, sx in ring]
        cone += [[rp * cx, rp * sx, z_tip + TIP_CHAMFER] for cx, sx in ring]
        ctx.spec.add_mesh(name=f"{p}plug_tip", uservert=np.ravel(cone).tolist())
        plug.add_geom(
            name=f"{p}plug_tip",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{p}plug_tip",
            density=DENSITY,
            priority=1,
            rgba=list(OBJECT_RGBA),
        )
        geoms.append(f"{p}plug_tip")
        for k, (z0, z1, r) in enumerate(pieces):
            name = f"{p}plug" if k == 0 else f"{p}plug{k}"
            plug.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r, 0.5 * (z1 - z0), 0.0],
                pos=[0.0, 0.0, 0.5 * (z0 + z1)],
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(name)
        plug.add_site(name=f"{p}plug", pos=[0.0, 0.0, 0.5 * plen], size=[0.003, 0.0, 0.0])
        plug.add_site(name=f"{p}plug_tip", pos=[0.0, 0.0, -0.5 * plen], size=[0.003, 0.0, 0.0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=tuple(bodies),
            geoms=tuple(geoms),
            joints={"free": f"{p}free", **{f"pad{k}": f"{p}pad{k}" for k in range(n_pads)}},
            sites={"plug": f"{p}plug", "socket": f"{p}socket", "plug_tip": f"{p}plug_tip"},
            info={
                "plug_d_m": dia,
                "plug_len_m": plen,
                "bore_m": r_bore,
                "chamfer_m": chamfer,
                "depth_m": depth,
                "socket_top_z": float(ctx.table_top_z + top),
                "outer_m": float(r_out / math.cos(half)),
                "insert_force_n": f_ins,
                "latch": float(latch),
                "pads": float(n_pads),
                "pad_z": float(ctx.table_top_z + z_pad),
                "pad_stiffness": stiffness,
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "socket_x": float(origin[0]),
                "socket_y": float(origin[1]),
            },
        )

    def _depth(self, obj: ResolvedHandles, data: mujoco.MjData) -> tuple[float, float]:
        """(depth of the plug tip below the mouth, radial offset of the tip from the axis), m."""
        info = obj.handles.info
        tip = data.site_xpos[obj.site_ids["plug_tip"]]
        mouth = data.site_xpos[obj.site_ids["socket"]]
        radial = math.hypot(float(tip[0] - mouth[0]), float(tip[1] - mouth[1]))
        inside = radial <= info["bore_m"] + info["chamfer_m"] + 0.001
        return (max(0.0, float(mouth[2] - tip[2])) if inside else 0.0), radial

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        depth = self._depth(obj, data)[0]
        full = obj.handles.info["depth_m"]
        if depth >= full - SEAT_TOL:  # on the floor: contact compliance does not decide seating
            depth = full
        return {"depth_mm": 1000.0 * depth}


register_template_builder(PortPlugBuilder())
