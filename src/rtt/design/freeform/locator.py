"""Locator pin (product mode only, never scored): one convex frustum in a declared socket.

A pin is added to a jaw only when the recipe asks for one (``locator`` is ``straight_pin``
or ``conical_pin``) and the part declares a socket facing that jaw. It is coaxial with the
socket, its radius at the socket mouth is the socket radius minus the clearance, and a
conical pin tapers by ``wall_draft_deg`` toward its tip. Its length is
``min(socket depth - clearance, stroke cap)``, where the stroke cap keeps the seat closure
within the usable stroke. The pin's base sinks into the pad behind the socket mouth, so it
fuses with the jaw.

Coordinates are the left-jaw build convention at seat: x across the jaw, ``u`` (= y) away
from the part's centre toward this jaw, table heights for z. A 16-gon frustum is exactly
convex by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PIN_SEGMENTS = 16
PIN_MIN_TIP_RADIUS = 0.001  # m
PIN_BASE_SINK = 0.0015  # m behind the pad face
FACING_TOL_DEG = 10.0


@dataclass(frozen=True)
class PinPlan:
    """A pin in seat coordinates (part frame: x, u, table z)."""

    axis: np.ndarray  # unit vector out of the socket (toward the jaw)
    mouth: np.ndarray  # socket mouth centre
    base: np.ndarray  # pin base centre (inside the pad)
    tip: np.ndarray  # pin tip centre (inside the socket)
    r_base: float
    r_tip: float
    length: float  # from the mouth to the tip
    feature: str


def facing_sockets(features: tuple, mirrored: bool) -> list:
    """Sockets whose axis points at this jaw (+y in the build convention, within 10 degrees)."""
    out = []
    cos_tol = math.cos(math.radians(FACING_TOL_DEG))
    for f in features:
        if getattr(f, "kind", None) != "socket":
            continue
        ay = -f.axis[1] if mirrored else f.axis[1]
        if ay >= cos_tol:
            out.append(f)
    return out


def plan_pin(
    feature: object,
    *,
    mirrored: bool,
    z_shift: float,
    clearance: float,
    taper_rad: float,
    face_u_at_mouth: float,
    max_length: float,
) -> tuple[PinPlan | None, str | None]:
    """The pin for one socket, or ``(None, reason)`` when it cannot be placed."""
    sgn = np.array([1.0, -1.0 if mirrored else 1.0, 1.0])
    axis = np.asarray(feature.axis, dtype=np.float64) * sgn  # type: ignore[attr-defined]
    axis /= np.linalg.norm(axis)
    mouth = np.asarray(feature.origin, dtype=np.float64) * sgn + np.array([0.0, 0.0, z_shift])  # type: ignore[attr-defined]
    radius = float(feature.radius) - clearance  # type: ignore[attr-defined]
    if radius <= PIN_MIN_TIP_RADIUS:
        return None, f"socket {feature.name!r} is too narrow for a pin"  # type: ignore[attr-defined]
    length = min(float(feature.depth) - clearance, max_length)  # type: ignore[attr-defined]
    tan_t = math.tan(taper_rad) if taper_rad > 0 else 0.0
    if tan_t > 0:
        length = min(length, (radius - PIN_MIN_TIP_RADIUS) / tan_t)
    if length <= 0.001:
        return None, f"socket {feature.name!r}: no room for a pin within the stroke"  # type: ignore[attr-defined]
    back = face_u_at_mouth + PIN_BASE_SINK - float(mouth @ axis)
    base = mouth + axis * max(back, PIN_BASE_SINK)
    tip = mouth - axis * length
    r_base = radius + max(back, PIN_BASE_SINK) * tan_t
    r_tip = radius - length * tan_t
    return PinPlan(axis, mouth, base, tip, r_base, r_tip, length, feature.name), None  # type: ignore[attr-defined]


def pin_sections(pin: PinPlan) -> tuple[np.ndarray, np.ndarray]:
    """The two 16-gon rings (base, tip) of the frustum, same winding."""
    a = pin.axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    th = 2.0 * math.pi * np.arange(PIN_SEGMENTS) / PIN_SEGMENTS
    ring = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
    return pin.base + pin.r_base * ring, pin.tip + pin.r_tip * ring
