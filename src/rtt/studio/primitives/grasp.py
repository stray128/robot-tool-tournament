"""The ``grasp`` primitive: the v1 family primitive and criterion in the Studio (see :mod:`._pick`)."""

from __future__ import annotations

from rtt.studio.plan import register_primitive
from rtt.studio.primitives._pick import PickController, PickPredicate
from rtt.studio.types import PrimitiveKind

CONTROLLER = PickController(PrimitiveKind.GRASP, name="studio_grasp")
PREDICATE = PickPredicate(PrimitiveKind.GRASP, name="studio_grasp")
register_primitive(CONTROLLER, PREDICATE)
