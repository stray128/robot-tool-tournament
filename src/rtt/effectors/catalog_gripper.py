"""The ``catalog_gripper`` family: open two-finger grippers and procedural adaptive grippers.

* Vendored two-finger parallel grippers (``kind = "gripper"`` in
  :data:`rtt.effectors.catalog.models.CATALOG_MODELS`): mounted fingers-down under the shared
  wrist, closing along the parts' y axis, one ``pinch`` synergy each at the upstream actuator
  gains.
* Procedural adaptive grippers with two and three fingers (:mod:`rtt.effectors.catalog.adaptive`,
  our own geometry): two phalanges per finger coupled by a tendon, one actuator per finger,
  ``power`` and ``pinch`` synergies.

One grip-height fraction per model and primitive (chosen on the c0 condition of all six
training parts together, never per part). Engagement is the shared contact test; registration
is repeatability against the gripper's own nominal episode. Fidelity: rigid.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from rtt.effectors.catalog.adaptive import (
    ADAPTIVE_MODELS,
    adaptive_criterion,
    adaptive_primitive,
    adaptive_specs,
    build_adaptive,
)
from rtt.effectors.catalog.family import build_vendored, contact_criterion, synergy_primitive, vendored_spec
from rtt.effectors.catalog.models import CATALOG_MODELS, catalog_model
from rtt.effectors.catalog.models import card_note as catalog_card_note
from rtt.effectors.primitives import ContactCriterion, SynergyPrimitive
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import BuildContext, EffectorBuild, EffectorFamily, EffectorSpec

__all__ = ["CatalogGripperFamily", "gripper_presets"]


def gripper_presets() -> tuple[EffectorSpec, ...]:
    """Every vendored gripper in the presets x synergy (table order), then the adaptive grippers
    x primitives. A model with ``in_presets = False`` stays buildable but is not an entrant."""
    vendored = tuple(
        vendored_spec(EffectorFamily.CATALOG_GRIPPER, cm, syn)
        for cm in CATALOG_MODELS.values()
        if cm.kind == "gripper" and cm.in_presets
        for syn in cm.synergies
    )
    return vendored + adaptive_specs()


class CatalogGripperFamily(EffectorFamilyImpl):
    """Vendored and procedural grippers (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.CATALOG_GRIPPER
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"

    def presets(self) -> tuple[EffectorSpec, ...]:
        return gripper_presets()

    def card_note(self, model: str) -> str | None:
        """The note every page prints beside this model (a low published grip, for example)."""
        return catalog_card_note(model)

    def _model(self, spec: EffectorSpec):  # type: ignore[no-untyped-def]
        cm = catalog_model(spec.model)
        if cm.kind != "gripper":
            raise ValueError(f"{spec.model} is not a gripper")
        return cm, cm.synergy(spec.primitive)

    def build(self, ctx: BuildContext) -> EffectorBuild:
        if ctx.effector.model in ADAPTIVE_MODELS:
            return build_adaptive(ctx)
        cm, syn = self._model(ctx.effector)
        return build_vendored(ctx, cm, syn)

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> SynergyPrimitive:
        if spec.model in ADAPTIVE_MODELS:
            return adaptive_primitive(spec, build)
        cm, syn = self._model(spec)
        return synergy_primitive(cm, syn)

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> ContactCriterion:
        if spec.model in ADAPTIVE_MODELS:
            return adaptive_criterion(spec)
        cm, _ = self._model(spec)
        return contact_criterion(cm)


register_family(CatalogGripperFamily())
