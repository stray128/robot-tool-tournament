"""Graspable test objects and their placement on the table.

Objects are free rigid bodies of uniform density resting on a static table.
Three families are supported:

* ``box``: ``size = (sx, sy, sz)`` full extents in the object frame.
* ``cylinder``: ``size = (radius, height)``, standing upright (axis along Z).
* ``rod``: a capsule ``size = (radius, length)`` (tip to tip) lying on the table
  with its axis along the object-frame X axis.

The gripper closes along the object-frame Y axis once the palm yaw matches the
placement yaw, so :func:`grip_width` (the Y extent) is the width the fingers see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

ObjectKind = Literal["box", "cylinder", "rod"]
MAX_OBJECT_SIZE = 0.5  # m; larger values almost certainly mean millimetres were passed

_ROD_QUAT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # capsule axis Z -> X


@dataclass(frozen=True)
class ObjectSpec:
    """Shape of one test object; density defaults to the task's object density."""

    kind: ObjectKind
    size: tuple[float, ...]
    density: float | None = None

    def __post_init__(self) -> None:
        expected = {"box": 3, "cylinder": 2, "rod": 2}
        if self.kind not in expected:
            raise ValueError(f"unknown object kind {self.kind!r}")
        if len(self.size) != expected[self.kind]:
            raise ValueError(f"{self.kind} needs {expected[self.kind]} size values, got {self.size}")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"object sizes must be positive, got {self.size}")
        if any(s > MAX_OBJECT_SIZE for s in self.size):
            raise ValueError(f"object sizes are in metres (at most {MAX_OBJECT_SIZE}), got {self.size}")
        if self.kind == "rod" and self.size[1] < 2 * self.size[0]:
            raise ValueError("rod length must be at least its diameter")


@dataclass(frozen=True)
class Placement:
    """Nominal object pose on the table: centre (x, y) in metres and yaw in radians."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def height(obj: ObjectSpec) -> float:
    """Vertical extent of the object when resting on the table."""
    if obj.kind == "box":
        return obj.size[2]
    if obj.kind == "cylinder":
        return obj.size[1]
    return 2.0 * obj.size[0]


def grip_width(obj: ObjectSpec) -> float:
    """Extent along the object-frame Y axis, i.e. the width between the fingers."""
    if obj.kind == "box":
        return obj.size[1]
    return 2.0 * obj.size[0]


def half_extents(obj: ObjectSpec) -> tuple[float, float, float]:
    """Half extents of the object's axis-aligned bounding box in its own frame."""
    if obj.kind == "box":
        sx, sy, sz = obj.size
        return (sx / 2, sy / 2, sz / 2)
    if obj.kind == "cylinder":
        r, h = obj.size
        return (r, r, h / 2)
    r, length = obj.size
    return (length / 2, r, r)


def volume(obj: ObjectSpec) -> float:
    if obj.kind == "box":
        sx, sy, sz = obj.size
        return sx * sy * sz
    if obj.kind == "cylinder":
        r, h = obj.size
        return math.pi * r * r * h
    r, length = obj.size
    return math.pi * r * r * (length - 2 * r) + 4.0 / 3.0 * math.pi * r**3


def geom_params(obj: ObjectSpec) -> tuple[mujoco.mjtGeom, list[float], tuple[float, float, float, float]]:
    """MuJoCo geom type, size vector and local orientation (wxyz) for the object."""
    if obj.kind == "box":
        return mujoco.mjtGeom.mjGEOM_BOX, list(half_extents(obj)), (1.0, 0.0, 0.0, 0.0)
    if obj.kind == "cylinder":
        r, h = obj.size
        return mujoco.mjtGeom.mjGEOM_CYLINDER, [r, h / 2, 0.0], (1.0, 0.0, 0.0, 0.0)
    r, length = obj.size
    return mujoco.mjtGeom.mjGEOM_CAPSULE, [r, length / 2 - r, 0.0], _ROD_QUAT


def yaw_quat(yaw: float) -> np.ndarray:
    """Quaternion (wxyz) for a rotation of ``yaw`` radians about +Z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rest_pose(obj: ObjectSpec, placement: Placement, table_top_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint position and quaternion (wxyz) with the object resting on the table."""
    pos = np.array([placement.x, placement.y, table_top_z + height(obj) / 2])
    return pos, yaw_quat(placement.yaw)
