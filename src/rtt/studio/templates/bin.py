"""The ``bin`` template: an open box, fixed to the table or free (S1 reference builder).

Five box geoms (floor and four walls, ``priority = 1``) at the instance pose. A fixed bin is a
body without joints (welded to the world, so its geoms are static); a free bin is one body on a
free joint. Its handle site ``bin`` is
at the centre of the interior floor. ``info`` carries the interior half sizes and floor height,
which a pick predicate uses for ``in_target``. A bin reports no state quantities.
"""

from __future__ import annotations

import math
from typing import ClassVar

import mujoco

from rtt.studio.templates import (
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["BinBuilder"]

BIN_RGBA = (0.55, 0.58, 0.62, 1.0)
BIN_DENSITY = 950.0  # kg/m^3, a moulded polymer tote [assumed]


class BinBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.BIN

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        lx, ly, h, w = (float(prm[k]) / 1000.0 for k in ("length_mm", "width_mm", "height_mm", "wall_mm"))
        if 2.0 * w >= min(lx, ly) or w >= h:
            raise ValueError(f"{ctx.instance.name}: walls of {1000 * w:g} mm leave no interior")
        yaw = math.radians(pose.yaw_deg)
        pos = [pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z]
        quat = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
        fixed = bool(prm["fixed"])
        if fixed:
            body = ctx.spec.worldbody.add_body(name=f"{p}frame", pos=pos, quat=quat)
        else:
            body = ctx.spec.worldbody.add_body(name=f"{p}body", pos=pos, quat=quat)
            body.add_freejoint(name=f"{p}free")
        hx, hy = 0.5 * lx, 0.5 * ly
        boxes = {
            "floor": ([hx, hy, 0.5 * w], [0.0, 0.0, 0.5 * w]),
            "wall_px": ([0.5 * w, hy, 0.5 * h], [hx - 0.5 * w, 0.0, 0.5 * h]),
            "wall_nx": ([0.5 * w, hy, 0.5 * h], [-hx + 0.5 * w, 0.0, 0.5 * h]),
            "wall_py": ([hx - w, 0.5 * w, 0.5 * h], [0.0, hy - 0.5 * w, 0.5 * h]),
            "wall_ny": ([hx - w, 0.5 * w, 0.5 * h], [0.0, -hy + 0.5 * w, 0.5 * h]),
        }
        geoms = []
        for name, (size, gpos) in boxes.items():
            body.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=gpos,
                priority=1,
                density=BIN_DENSITY,
                rgba=list(BIN_RGBA),
            )
            geoms.append(f"{p}{name}")
        body.add_site(name=f"{p}bin", pos=[0.0, 0.0, w], size=[0.004, 0.0, 0.0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(body.name,),
            geoms=tuple(geoms),
            joints={} if fixed else {"free": f"{p}free"},
            sites={"bin": f"{p}bin"},
            info={"inner_hx": hx - w, "inner_hy": hy - w, "floor_z": ctx.table_top_z + w, "height_m": h},
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {}


register_template_builder(BinBuilder())
