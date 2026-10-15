"""Perturbations applied while the gripper holds the object (robustness tests).

* :class:`Shake`: the palm oscillates sinusoidally along one gripper axis around its
  hold pose. The commanded displacement is
  ``amplitude * w(s) * sin(2 pi frequency s)`` for ``s`` in ``[0, cycles / frequency]``,
  where the window ``w`` ramps from 0 to 1 (raised cosine) over the first
  ``ramp_cycles`` cycles and back to 0 over the last ones, so the servo targets have no
  velocity step. Peak commanded acceleration is about ``(2 pi frequency)^2 amplitude``.
* :class:`Push`: a constant horizontal force ``impulse / duration`` on the object for
  ``duration`` seconds (applied through ``xfrc_applied``), acting ``height`` metres
  above the object's centre of mass, so it also twists the object in the grasp
  (``height = 0``: a pure push through the centre of mass).

Both start ``delay`` seconds after the HOLD phase begins (the palm has reached the
lift height and stopped). Axes are in the gripper frame: ``x`` along the finger
width, ``y`` along the closing axis, ``z`` up; :func:`axis_vector` maps them to the
world frame for a given palm yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Axis = Literal["x", "y", "z"]
AXES: tuple[Axis, ...] = ("x", "y", "z")


def axis_vector(axis: Axis, yaw: float) -> np.ndarray:
    """World-frame unit vector of a gripper axis when the palm is turned by ``yaw`` (rad)."""
    c, s = math.cos(yaw), math.sin(yaw)
    if axis == "x":
        return np.array([c, s, 0.0])
    if axis == "y":
        return np.array([-s, c, 0.0])
    if axis == "z":
        return np.array([0.0, 0.0, 1.0])
    raise ValueError(f"axis must be one of {AXES}, got {axis!r}")


@dataclass(frozen=True)
class Shake:
    """Sinusoidal palm motion during the hold (see the module docstring)."""

    axis: Axis
    amplitude: float  # m
    frequency: float  # Hz
    cycles: float
    ramp_cycles: float = 0.5
    delay: float = 0.2  # s after HOLD starts

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"shake axis must be one of {AXES}, got {self.axis!r}")
        if self.amplitude < 0 or self.frequency <= 0 or self.cycles <= 0 or self.delay < 0:
            raise ValueError("shake needs amplitude >= 0, frequency > 0, cycles > 0 and delay >= 0")
        if not 0 <= 2 * self.ramp_cycles <= self.cycles:
            raise ValueError("shake ramp_cycles must be in [0, cycles / 2]")

    @property
    def duration(self) -> float:
        return self.cycles / self.frequency

    @property
    def peak_acceleration(self) -> float:
        """Peak commanded acceleration (m/s^2) of the un-windowed sinusoid."""
        return (2.0 * math.pi * self.frequency) ** 2 * self.amplitude

    def displacement(self, s: float) -> float:
        """Commanded offset (m) along the axis, ``s`` seconds after the shake starts."""
        if s <= 0.0 or s >= self.duration:
            return 0.0
        ramp = self.ramp_cycles / self.frequency
        window = 1.0
        if ramp > 0.0:
            u = min(1.0, s / ramp, (self.duration - s) / ramp)
            window = 0.5 * (1.0 - math.cos(math.pi * u))
        return self.amplitude * window * math.sin(2.0 * math.pi * self.frequency * s)


@dataclass(frozen=True)
class Push:
    """A short constant force on the object's centre of mass during the hold."""

    axis: Literal["x", "y"]
    impulse: float  # N s; the sign sets the direction along the axis
    duration: float = 0.01  # s
    delay: float = 0.2  # s after HOLD starts
    height: float = 0.0  # m above the object's centre of mass where the force acts

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"push axis must be 'x' or 'y' (horizontal), got {self.axis!r}")
        if self.duration <= 0 or self.delay < 0:
            raise ValueError("push needs duration > 0 and delay >= 0")

    @property
    def force(self) -> float:
        """Magnitude (signed) of the applied force in newtons."""
        return self.impulse / self.duration

    def wrench(self, yaw: float) -> np.ndarray:
        """World-frame ``[force, torque about the centre of mass]`` for a palm yaw (rad)."""
        f = self.force * axis_vector(self.axis, yaw)
        return np.concatenate([f, np.cross([0.0, 0.0, self.height], f)])


Perturbation = Shake | Push
