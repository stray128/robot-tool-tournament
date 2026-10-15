"""A hand-built finger for simulator unit tests.

The simulator runs on fingers from the real design generator (see
:mod:`rtt.sim.m1`); this helper builds a plain straight finger directly from
boxes so scene, contact and determinism tests do not depend on the generator,
and so tests can bend or rescale its 3-element segments freely. It follows the
shared contract in :mod:`rtt.design.types`: left finger, root at the origin,
body along -Z, inner (contact) face on the y = 0 plane facing -Y.
"""

from __future__ import annotations

from rtt.design.types import BOXES_PER_FINGER, Box, FingerGeometry

_SEGMENTS = 4
_FACE_ELEMENTS = 3
_TIP_BOXES = BOXES_PER_FINGER - _SEGMENTS * _FACE_ELEMENTS  # == 2


def straight_finger(
    length: float = 0.08,
    thickness: float = 0.010,
    width: float = 0.020,
    density_kg_m3: float = 1240.0,
) -> FingerGeometry:
    """A flat-faced, unbent, blunt finger: 4 segments x 3 width elements + 2 parked tip boxes.

    The backbone spans ``y in [0, thickness]``, ``x in [-width/2, width/2]`` and
    ``z in [-length, 0]``. The two tip boxes are "parked" (tiny and fully inside
    the last centre element), as the contract prescribes for unused boxes.
    """
    seg_len = length / _SEGMENTS
    elem_w = width / _FACE_ELEMENTS
    boxes: list[Box] = []
    for s in range(_SEGMENTS):
        cz = -(s + 0.5) * seg_len
        for e in range(_FACE_ELEMENTS):
            cx = -width / 2 + (e + 0.5) * elem_w
            boxes.append(
                Box(center=(cx, thickness / 2, cz), half_size=(elem_w / 2, thickness / 2, seg_len / 2))
            )
    park = 0.0005
    park_center = (0.0, thickness / 2, -(_SEGMENTS - 0.5) * seg_len)
    boxes.extend(Box(center=park_center, half_size=(park, park, park)) for _ in range(_TIP_BOXES))
    return FingerGeometry(boxes=tuple(boxes), density_kg_m3=density_kg_m3)
