"""Printability of a compiled jaw (FDM, PLA, printed root-down on its mount face).

Checked (a failure rejects the design, code in parentheses):

* minimum wall 1.2 mm, analytically per pad strip, for the hook and for the body (``min_wall``);
* pad features at least 1.6 mm: trimming every material rib narrower than 1.6 mm would
  move the face back by at most :data:`FEATURE_EXCESS_MAX` (``min_feature``);
* at most 50 g of solid PLA per jaw (``mass``);
* every body slope at most 45 degrees, so the body prints without supports (``body_slope``).

Reported only: the overhang share, the fraction of the print's surface area that faces the
bed at more than 45 degrees when printed root-down (the mount face itself excluded).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from rtt.design.freeform.validate import InvalidDesign, mm

MIN_WALL: Final = 0.0012  # m
MIN_FEATURE: Final = 0.0016  # m
FEATURE_EXCESS_MAX: Final = 0.001  # m
MAX_MASS: Final = 0.050  # kg per jaw
MAX_BODY_SLOPE: Final = 1.0  # dy/dz (45 degrees)
OVERHANG_COS: Final = float(np.cos(np.radians(45.0)))


def overhang_share(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Area share of downward-facing surface (root-down print: normals toward +z of the jaw)."""
    v, f = np.asarray(vertices), np.asarray(faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    nz = np.where(area > 0, n[:, 2] / np.maximum(2.0 * area, 1e-30), 0.0)
    on_bed = np.all(np.abs(np.stack([a[:, 2], b[:, 2], c[:, 2]], axis=1)) < 1e-7, axis=1)
    over = (nz > OVERHANG_COS) & ~on_bed
    total = float(area.sum())
    return float(area[over].sum() / total) if total > 0 else 0.0


def check_printability(jaw: Any, part_id: str, side: str) -> dict[str, float]:
    """Run the printability checks on one jaw build; returns the measured numbers."""
    walls = [jaw.min_strip_wall]
    if jaw.hook_height is not None:
        walls.append(jaw.hook_height)
    walls.append(jaw.min_body_wall)
    wall = float(min(walls))
    if wall < MIN_WALL - 1e-12:
        raise InvalidDesign("min_wall", f"{side} jaw has a {mm(wall)} wall (minimum {mm(MIN_WALL)})", part_id)
    excess = float(jaw.feature_excess)
    if excess > FEATURE_EXCESS_MAX + 1e-12:
        raise InvalidDesign(
            "min_feature",
            f"{side} pad has a rib narrower than {mm(MIN_FEATURE)} ({mm(excess)} of it would be trimmed)",
            part_id,
        )
    if jaw.mass > MAX_MASS + 1e-12:
        raise InvalidDesign(
            "mass",
            f"{side} jaw weighs {jaw.mass * 1000:.1f} g of PLA (maximum {MAX_MASS * 1000:.0f} g)",
            part_id,
        )
    if jaw.body_slope > MAX_BODY_SLOPE + 1e-9:
        deg = float(np.degrees(np.arctan(jaw.body_slope)))
        raise InvalidDesign(
            "body_slope", f"{side} body slope reaches {deg:.1f} degrees (maximum 45)", part_id
        )
    return {
        "min_wall_mm": wall * 1000.0,
        "feature_excess_mm": excess * 1000.0,
        "mass_g": jaw.mass * 1000.0,
        "body_slope_deg": float(np.degrees(np.arctan(jaw.body_slope))),
        "overhang_share": overhang_share(jaw.mesh.vertices, jaw.mesh.faces),
    }
