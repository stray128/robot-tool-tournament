"""The ``tile_bed`` template: a floor tile laid on a fresh mortar bed (Studio v2).

The tile-setting job: a vacuum frame lays the tile, a tamper pad presses it to its level.

Fidelity: **approximation** (the template's caveat). The bed is a plastic model that **keeps its
set**: the tamper's press is read after the pad lifts, so an elastic bed would spring back and
hide how hard it was pressed (measured in Studio v2). Each of :data:`GRID` x :data:`GRID`
bed cells under the tile is a short block on a vertical slide driven by its own stiff spring
(:data:`CELL_KP` [assumed], damped by :data:`CELL_KV` [assumed]) to an anchor that
:meth:`TileBedBuilder.passive` moves each control tick only as far as keeps the spring's force
within the cell's share of the bed's yield (the threaded joint's stick-slip friction, applied to
a slide): below the yield a cell holds (it gives elastically, at most a few tenths of a
millimetre), above it the cell sinks and stays sunk when the load comes off. The yield hardens as
the bed is pressed in: the whole bed gives at ``bed_yield_n`` plus ``bed_hardening_n_mm`` per mm
it has sunk (shared equally by the cells) [assumed], so a steady press ``P`` on a tile of weight
``W`` sinks it until ``yield + hardening x sink = P + W``: a harder press seats it deeper, and a
light one leaves it proud. The hardening counts the plastic sink (the anchor), not the elastic
give on top of it. Defaults (2.16 kg tile, 2 mm proud, 30 N, 15 N/mm [assumed]), a steady press at
the tile's centre for 1 s then let go (``tests/test_studio_templates.py``): unpressed it rests
1.85 mm proud (its 21.2 N is under the 30 N yield; the cells and the contacts give 0.15 mm); a 15 N
press leaves it 1.44 mm proud; the tamper's rated 40 N seats it 0.23 mm below its level; 60 N
over-presses it 1.56 mm below. Let go, it springs back 0.09 to 0.31 mm (the cells' elastic give and
the contacts' softness), so the set is kept and a press is read after the pad lifts. The cells
cannot sink more than :data:`OVER` below the level (the substrate). No mortar flow, suction or
curing; a tile is pressed level only by a press on it.

Instance frame: origin at the pose on the table top, turned by ``yaw_deg``.

* **Bed** (static): a substrate slab (:data:`BORDER` wider than the tile each side) under the cells.
* **Cells**: :data:`GRID` x :data:`GRID` blocks on vertical slides, :data:`CELL_FILL` of their pitch
  across and :data:`CELL_H` tall, their tops ``proud_mm`` above the tile's finished bottom
  (:data:`LEVEL_Z` above the table [assumed]) at the start; range ``proud + OVER``.
* **Tile** (a free body, the template's last body: the one a pick lifts): a ``tile_mm`` square,
  ``thickness_mm`` thick, of ``mass_g``, resting on the cells.

Sites: ``part`` (the tile's top centre: where a pick seals and a press presses; a push reads it
there, so the template carries no ``height_m``). ``info``: sizes and mass, the finished level of
the tile's top (``level_top_z``), the bed's numbers, and the pick stand-in (``diameter_m``,
``length_m``: a round slab as thick as the tile, at most :data:`PICK_STANDIN_M` across; only its
height places the tool).

State: ``height_err_mm`` (how far the tile's top centre stands from its finished level, above or
below) and ``tilt_deg`` (its tilt from level).
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.studio.templates import (
    FRAME_RGBA,
    PICK_STANDIN_M,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    quat_z,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["CELL_KP", "GRID", "LEVEL_Z", "TileBedBuilder"]

GRID: Final = 3  # cells a side [assumed]
CELL_FILL: Final = 0.9  # a cell's side over the cell pitch [assumed]
CELL_H: Final = 0.010  # m [assumed]
LEVEL_Z: Final = 0.020  # m: the tile's finished bottom above the table [assumed]
OVER: Final = 0.003  # m a tile can be pressed below its level before the substrate stops it [assumed]
BORDER: Final = 0.020  # m of substrate beyond the tile each side [assumed]
CELL_KP: Final = 1.0e5  # N/m: a cell's elastic stiffness (its anchor spring) [assumed]
CELL_KV: Final = 150.0  # N s/m: damps it (implicit), so the stick-slip anchor does not ratchet [assumed]
CELL_ARMATURE: Final = 0.1  # kg: keeps the stiff cell spring stable at the 1 ms step [assumed]
CELL_DENSITY: Final = 50.0  # kg/m^3; cells are gravity-compensated [assumed]
TILE_RGBA: Final = (0.78, 0.74, 0.66, 1.0)
BED_RGBA: Final = (0.52, 0.50, 0.46, 1.0)


class TileBedBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.TILE_BED

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        side, thick = float(prm["tile_mm"]) / 1000.0, float(prm["thickness_mm"]) / 1000.0
        mass = float(prm["mass_g"]) / 1000.0
        proud = float(prm["proud_mm"]) / 1000.0
        yield_n, hard = float(prm["bed_yield_n"]), float(prm["bed_hardening_n_mm"]) * 1000.0
        yaw = math.radians(pose.yaw_deg)
        x0, y0 = pose.x_mm / 1000.0, pose.y_mm / 1000.0
        bed = ctx.spec.worldbody.add_body(name=f"{p}bed", pos=[x0, y0, ctx.table_top_z], quat=quat_z(yaw))
        geoms: list[str] = []
        sub_top = LEVEL_Z - OVER - CELL_H - 0.001
        half = 0.5 * side + BORDER
        bed.add_geom(
            name=f"{p}substrate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half, half, 0.5 * sub_top],
            pos=[0.0, 0.0, 0.5 * sub_top],
            priority=1,
            rgba=list(FRAME_RGBA),
        )
        geoms.append(f"{p}substrate")
        pitch = side / GRID
        cell = CELL_FILL * pitch
        n = GRID * GRID
        travel = proud + OVER
        cells, joints, actuators = [], {}, []
        for i in range(GRID):
            for j in range(GRID):
                name = f"cell{i}{j}"
                cx, cy = (i - 0.5 * (GRID - 1)) * pitch, (j - 0.5 * (GRID - 1)) * pitch
                body = bed.add_body(name=f"{p}{name}", pos=[cx, cy, LEVEL_Z + proud - 0.5 * CELL_H])
                body.gravcomp = 1.0
                body.add_joint(
                    name=f"{p}{name}",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                    axis=[0.0, 0.0, -1.0],
                    range=[0.0, travel],
                    armature=CELL_ARMATURE,
                    solref_limit=list(ctx.task.physics.limit_solref),
                )
                body.add_geom(
                    name=f"{p}{name}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.5 * cell, 0.5 * cell, 0.5 * CELL_H],
                    density=CELL_DENSITY,
                    priority=1,
                    rgba=list(BED_RGBA),
                )
                act = ctx.spec.add_actuator(
                    name=f"{p}{name}", target=f"{p}{name}", trntype=mujoco.mjtTrn.mjTRN_JOINT
                )
                act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                act.gainprm[0] = CELL_KP
                act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                act.biasprm[0] = 0.0
                act.biasprm[1] = -CELL_KP
                act.biasprm[2] = -CELL_KV
                act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                cap = (yield_n + hard * travel) / n + 1.0
                act.forcerange = [-cap, cap]
                cells.append(f"{p}{name}")
                geoms.append(f"{p}{name}")
                joints[name] = f"{p}{name}"
                actuators.append(f"{p}{name}")

        # the tile, a free body on the cells (the last body: the one a pick lifts)
        rest_z = ctx.table_top_z + LEVEL_Z + proud + 0.5 * thick
        tile = ctx.spec.worldbody.add_body(name=f"{p}tile", pos=[x0, y0, rest_z], quat=quat_z(yaw))
        tile.add_freejoint(name=f"{p}free")
        tile.add_geom(
            name=f"{p}tile",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * side, 0.5 * side, 0.5 * thick],
            mass=mass,
            priority=1,
            rgba=list(TILE_RGBA),
        )
        geoms.append(f"{p}tile")
        tile.add_site(name=f"{p}part", pos=[0.0, 0.0, 0.5 * thick], size=[0.004, 0.0, 0.0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}bed", *cells, f"{p}tile"),
            geoms=tuple(geoms),
            joints={**joints, "free": f"{p}free"},
            sites={"part": f"{p}part"},
            actuators=tuple(actuators),
            info={
                # the pick stand-in (module docstring): a round slab as thick as the tile
                "diameter_m": min(side, PICK_STANDIN_M),
                "length_m": thick,
                "tile_m": side,
                "tile_t_m": thick,
                "mass_kg": mass,
                "proud_m": proud,
                "level_top_z": float(ctx.table_top_z + LEVEL_Z + thick),
                "yield_n": yield_n,
                "hardening_n_m": hard,
                "cells": float(n),
                "period_s": float(ctx.task.controller.control_period),
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        info = obj.handles.info
        tile = obj.body_ids[-1]
        rot = data.xmat[tile].reshape(3, 3)
        top = data.xpos[tile] + rot @ np.array([0.0, 0.0, 0.5 * info["tile_t_m"]])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))
        return {"height_err_mm": 1000.0 * abs(float(top[2]) - info["level_top_z"]), "tilt_deg": tilt}

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Each cell's anchor (module docstring): where it was (at a segment's first tick, the cell
        itself), moved only as far as keeps the spring's force within the cell's yield at the
        middle of the coming tick; the yield hardens with how far the cell has sunk."""
        info = obj.handles.info
        n = info["cells"]
        out = np.zeros(len(obj.actuator_ids))
        for k, act in enumerate(obj.actuator_ids):
            role = f"cell{k // GRID}{k % GRID}"
            q = float(data.qpos[obj.joint_qpos[role]])
            w = float(data.qvel[obj.joint_dof[role]])
            anchor = q if data.time <= 0.0 else float(data.ctrl[act])
            # the yield hardens with the plastic sink (the anchor), not the elastic give on top of it
            limit = (info["yield_n"] + info["hardening_n_m"] * max(anchor, 0.0)) / n
            mid = q + 0.5 * w * info["period_s"]
            reach = limit / CELL_KP
            out[k] = min(max(anchor, mid - reach), mid + reach)
        return out


register_template_builder(TileBedBuilder())
