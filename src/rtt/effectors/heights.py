"""The part height a primitive plans its tool tip from (shared by the cups, the vacuum frames and the
process tools).

``rtt.effectors.primitives.part_height`` (frozen) builds a v1 mesh object, which refuses a part more than
:data:`rtt.sim.objects.MAX_OBJECT_SIZE` across. The Studio plans the same tools on parts that big (a
1000 x 600 mm panel), so a plan reads the height through :func:`top_height`: the same number for every
part a v1 mesh object can hold, the bounding box's height for a part too wide for one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rtt.effectors.primitives import part_height
from rtt.sim import objects as objs

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["top_height"]


def top_height(part: PartSpec) -> float:
    """The part's height on the table (m): :func:`~rtt.effectors.primitives.part_height`, or the
    bounding box's height when the part is wider than a v1 mesh object allows (the size check only: a
    degenerate or otherwise invalid mesh still raises)."""
    if max(float(s) for s in part.size) > objs.MAX_OBJECT_SIZE:
        return float(part.size[2])
    return part_height(part)
