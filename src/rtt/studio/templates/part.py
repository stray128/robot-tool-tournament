"""The ``part`` template: a free rigid part (S1 reference builder).

* ``shape = "box"`` or ``"cylinder"``: built by the frozen parts library's ``block`` and ``slug``
  families at the instance's size, with a density that gives ``mass_g``.
* ``shape = "mesh"``: ``ref = "library:<name>"`` is a parts-library part (the six training parts
  and their siblings). ``upload:`` and ``scan:`` refs are resolved by the intake track (S4)
  through :func:`register_part_resolver`; until then they are :class:`TemplateUnavailable`.

The part is a free body made of its exact convex pieces, each a mesh geom with ``priority = 1``
(the v1 part rule), resting on the table at the instance pose. Its handle site ``part`` is at
the centre of its bounding box. State: ``lift_mm`` (rise of the centre since the start) and
``tilt_deg`` (angle between the part's z axis and the world's).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import mujoco

from rtt.sim.scene import OBJECT_RGBA
from rtt.studio.templates import (
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    TemplateUnavailable,
    register_template_builder,
)
from rtt.studio.types import ObjectInstance, TemplateKind

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["PartBuilder", "part_spec_of", "register_part_resolver"]

_RESOLVERS: dict[str, Callable[[str], PartSpec]] = {}


def register_part_resolver(scheme: str, resolver: Callable[[str], PartSpec]) -> None:
    """Resolve ``<scheme>:<id>`` refs (``upload``, ``scan``) to a :class:`PartSpec` (track S4)."""
    if scheme not in ("upload", "scan"):
        raise ValueError("only upload and scan refs take a resolver; library refs are built in")
    _RESOLVERS[scheme] = resolver


def part_spec_of(instance: ObjectInstance) -> PartSpec:
    """The :class:`PartSpec` a ``part`` instance stands for (module docstring)."""
    from rtt.parts.library import Material, build_part, get_part

    if instance.kind is not TemplateKind.PART:
        raise ValueError(f"{instance.name} is a {instance.kind.value}, not a part")
    p = instance.params
    shape = p["shape"]
    if shape == "mesh":
        assert instance.ref is not None
        scheme, _, key = instance.ref.partition(":")
        if scheme == "library":
            try:
                return get_part(key)
            except KeyError:
                raise TemplateUnavailable(f"{instance.name}: no library part {key!r}") from None
        resolver = _RESOLVERS.get(scheme)
        if resolver is None:
            raise TemplateUnavailable(f"{instance.name}: {scheme} refs are not resolved yet (intake track)")
        return resolver(key)
    x, y, z = (float(p[k]) / 1000.0 for k in ("size_x_mm", "size_y_mm", "size_z_mm"))
    if shape == "box":
        volume = x * y * z
        dims = {"x": 1000.0 * x, "y": 1000.0 * y, "z": 1000.0 * z}
        family = "block"
    else:
        volume = math.pi * (0.5 * x) ** 2 * z
        dims = {"diameter": 1000.0 * x, "length": 1000.0 * z}
        family = "slug"
    density = (float(p["mass_g"]) / 1000.0) / volume
    mat = Material("studio_part", density, (0.18, 0.47, 0.86))
    return build_part(f"studio_{instance.name}", family, dims, mat)


class PartBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.PART

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        part = part_spec_of(ctx.instance)
        p = ctx.prefix
        pose = ctx.instance.pose
        yaw = math.radians(pose.yaw_deg)
        rest_z = ctx.table_top_z + 0.5 * part.size[2]
        body = ctx.spec.worldbody.add_body(
            name=f"{p}body",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, rest_z],
            quat=[math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        )
        body.add_freejoint(name=f"{p}free")
        body.explicitinertial = True
        body.mass = part.mass
        body.ipos = list(part.com)
        body.fullinertia = [v * part.density for v in part.unit_inertia]
        geoms = []
        for i, (verts, faces) in enumerate(zip(part.pieces, part.faces, strict=True)):
            mesh = f"{p}piece{i:02d}"
            ctx.spec.add_mesh(name=mesh, uservert=verts.ravel().tolist(), userface=faces.ravel().tolist())
            body.add_geom(
                name=f"{p}geom{i:02d}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(f"{p}geom{i:02d}")
        body.add_site(name=f"{p}part", pos=[0.0, 0.0, 0.0], size=[0.004, 0.0, 0.0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}body",),
            geoms=tuple(geoms),
            joints={"free": f"{p}free"},
            sites={"part": f"{p}part"},
            info={"rest_z": rest_z, "height_m": part.size[2], "mass_kg": part.mass},
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        b = obj.body_ids[0]
        z = float(data.xpos[b][2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[b][8])))))
        return {"lift_mm": 1000.0 * (z - obj.handles.info["rest_z"]), "tilt_deg": tilt}


register_template_builder(PartBuilder())
