"""Shared geometry contract between the design generator and the simulator.

Frame convention (all units SI: metres, kilograms, radians):

* The palm frame is Z-up. Fingers hang below the palm, extending along -Z.
* A finger's local frame has its origin at the finger root (where it mounts to
  the finger slide joint). The finger body extends along -Z from the root.
* The *left* finger sits on the +Y side of the palm and closes toward -Y, so its
  contact (inner) face points toward -Y. X is the finger width direction.
* The right finger is the mirror image of the left across the XZ plane
  (y -> -y); the simulator performs the mirroring, generators only emit the
  left finger.

Every finger is exactly ``BOXES_PER_FINGER`` boxes so that different designs can
share one batched simulation model (only sizes/poses change per world). Unused
boxes are emitted as tiny "parked" boxes fully inside another box.
"""

from __future__ import annotations

from dataclasses import dataclass

BOXES_PER_FINGER = 14


@dataclass(frozen=True)
class Box:
    """An oriented box in the left finger's local frame."""

    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FingerGeometry:
    """Collision geometry (== visual geometry) for one left finger."""

    boxes: tuple[Box, ...]
    density_kg_m3: float = 1240.0  # PLA

    def __post_init__(self) -> None:
        if len(self.boxes) != BOXES_PER_FINGER:
            raise ValueError(f"expected {BOXES_PER_FINGER} boxes, got {len(self.boxes)}")
