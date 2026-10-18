"""The ``wrap`` primitive: the v1 family primitive and criterion in the Studio (see :mod:`._pick`)."""

from __future__ import annotations

from rtt.studio.plan import register_primitive
from rtt.studio.primitives._pick import PickController, PickPredicate
from rtt.studio.types import PrimitiveKind

CONTROLLER = PickController(PrimitiveKind.WRAP, name="studio_wrap")
PREDICATE = PickPredicate(PrimitiveKind.WRAP, name="studio_wrap")
register_primitive(CONTROLLER, PREDICATE)
