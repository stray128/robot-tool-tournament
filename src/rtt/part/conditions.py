"""Training conditions for one uploaded part: stable poses x grasp candidates x placements x jitter.

:func:`conditions_for_part` turns a :class:`~rtt.part.intake.PartSpec` into a
:class:`~rtt.sim.conditions.Suite` that the existing search runner evaluates exactly
like the benchmark's training suite (same evaluator, same scoring rule):

* one **family** per resting pose (``pose0``, ``pose1``, ...), so the proposer's
  per-family feedback reads as "how well each way the part can land is held";
* per pose, its **grasp candidates** (jaw axis and grasp point along the part);
* per candidate, the **placements**: ``centred`` (the gripper aims at the grasp point)
  and ``sloppy`` (a coarse pose estimate, as in the benchmark's training suite: 6 mm off
  in both directions and 15 degrees off in yaw, at another spot on the table),
  optionally ``offcentre`` (aiming a quarter of the half-length beside the grasp point).
  The sloppy grasp is deliberately harder than the certificate's +-3 mm sweep: light
  parts held centrally saturate training at 100 % for most designs, which leaves the
  search nothing to compare;
* each condition's **jitter seeds** derive from the part's SHA-256 and the condition id,
  so the suite is identical on every machine (common random numbers).

The condition count is capped (``max_conditions``) by dropping the least likely poses
first; the most likely pose is always kept whole.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal

from rtt.part.intake import GraspCandidate, PartPose, PartSpec
from rtt.sim.conditions import Condition, Suite, derive_seed
from rtt.sim.objects import Placement

__all__ = ["DEFAULT_PLACEMENTS", "PLACEMENTS", "PlacementName", "conditions_for_part", "describe_pose"]

PlacementName = Literal["centred", "sloppy", "offcentre"]
PLACEMENTS: Final[tuple[PlacementName, ...]] = ("centred", "sloppy", "offcentre")
DEFAULT_PLACEMENTS: Final[tuple[PlacementName, ...]] = ("centred", "sloppy")
SLOPPY_ERR_MM: Final = 6.0
SLOPPY_ERR_YAW_DEG: Final = 15.0
SLOPPY_AT: Final = (-0.03, 0.04, -60.0)  # x (m), y (m), yaw (deg) of the sloppy table spot
OFFCENTRE_FRAC: Final = 0.25


def _placement(name: PlacementName, cand: GraspCandidate, half_length_mm: float) -> Placement:
    err_x = -cand.offset_mm / 1000.0  # the part sits -offset from where the gripper aims
    if name == "centred":
        return Placement(err_x=err_x)
    if name == "sloppy":
        x, y, yaw = SLOPPY_AT
        return Placement(
            x=x,
            y=y,
            yaw=math.radians(yaw),
            err_x=err_x + SLOPPY_ERR_MM / 1000.0,
            err_y=SLOPPY_ERR_MM / 1000.0,
            err_yaw=math.radians(SLOPPY_ERR_YAW_DEG),
        )
    if name == "offcentre":
        return Placement(err_x=err_x + OFFCENTRE_FRAC * half_length_mm / 1000.0)
    raise ValueError(f"unknown placement {name!r}; choose from {PLACEMENTS}")


def describe_pose(part: PartSpec, pose: PartPose, cand: GraspCandidate) -> str:
    """Plain words for the proposer prompt (it is the condition's ``size_label``)."""
    lx, wy, hz = pose.extents_mm
    if cand.axis == "wide":
        lx, wy = wy, lx
    where = "at the centre" if abs(cand.offset_mm) < 1e-9 else f"{abs(cand.offset_mm):.0f} mm from the centre"
    return (
        f"uploaded part '{part.name}' resting in pose {pose.rank} (lands this way in "
        f"{100 * pose.probability:.0f}% of drops): {lx:.0f} x {wy:.0f} x {hz:.0f} mm (length x width across the "
        f"jaw x height), {part.mass_g:.0f} g, grasped {where};"
    )


def conditions_for_part(
    part: PartSpec,
    *,
    placements: Sequence[PlacementName] = DEFAULT_PLACEMENTS,
    axes: Literal["narrow", "all"] = "narrow",
    seeds_per_condition: int = 1,
    max_conditions: int = 12,
    base_seed: int | None = None,
    name: str | None = None,
) -> Suite:
    """The part's training suite (see the module docstring).

    Args:
        part: The validated part.
        placements: Which placements every grasp candidate is tried at.
        axes: ``"narrow"``: only grasps across the narrowest side; ``"all"``: also the
            90-degree grasps intake offered.
        seeds_per_condition: Jitter seeds per condition (pose and friction jitter).
        max_conditions: Cap on conditions; whole poses are dropped from the least likely.
        base_seed: Seed base (default: derived from the part file's SHA-256).
        name: Suite name (default ``part_<name>``).
    """
    if not placements or len(set(placements)) != len(placements):
        raise ValueError("placements must be a non-empty list without duplicates")
    for pl in placements:
        if pl not in PLACEMENTS:
            raise ValueError(f"unknown placement {pl!r}; choose from {PLACEMENTS}")
    if seeds_per_condition < 1 or max_conditions < 1:
        raise ValueError("seeds_per_condition and max_conditions must be >= 1")
    suite_name = name or f"part_{part.name}"
    seed_base = base_seed if base_seed is not None else int(part.source_sha256[:15], 16)

    conditions: list[Condition] = []
    for pose in part.poses:
        cands = [c for c in pose.candidates if axes == "all" or c.axis == "narrow"]
        pose_conditions = []
        for cand in cands:
            obj = pose.obj(cand.axis)
            half = obj.size[0] * 1000.0 / 2.0
            for pl in placements:
                cid = f"{suite_name}/pose{pose.rank}-{cand.label}/{pl}"
                seeds = tuple(derive_seed(seed_base, cid, rep) for rep in range(seeds_per_condition))
                pose_conditions.append(
                    Condition(
                        condition_id=cid,
                        family=f"pose{pose.rank}",
                        obj=obj,
                        placement=_placement(pl, cand, half),
                        seeds=seeds,
                        size_label=describe_pose(part, pose, cand),
                        placement_label=pl,
                    )
                )
        if conditions and len(conditions) + len(pose_conditions) > max_conditions:
            break
        conditions.extend(pose_conditions[: max_conditions if not conditions else None])
    return Suite(suite_name, tuple(conditions))
