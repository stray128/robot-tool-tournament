"""The ``catalog_hand`` family: open dexterous hands from vendored MJCF (product mode).

Each vendored hand (:data:`rtt.effectors.catalog.models.CATALOG_MODELS`, ``kind = "hand"``) is
mounted palm-down under the shared wrist and driven by two scripted synergies of the same
model: ``power`` (every finger wraps, the thumb opposes) and ``pinch`` (thumb and index tips
meet). Each synergy has one grip-height fraction, chosen on the c0 condition of all six training
parts together and never tuned per part. Engagement is the shared contact test (at least two
hand bodies on the part, net forces at least 90 deg apart); registration is repeatability against
the hand's own nominal episode. Fidelity: rigid (``docs/EFFECTORS.md`` section 10).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from rtt.effectors.catalog.family import build_vendored, contact_criterion, synergy_primitive, vendored_spec
from rtt.effectors.catalog.models import CATALOG_MODELS, catalog_model
from rtt.effectors.catalog.models import card_note as catalog_card_note
from rtt.effectors.primitives import ContactCriterion, SynergyPrimitive
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import BuildContext, EffectorBuild, EffectorFamily, EffectorSpec

__all__ = ["CatalogHandFamily", "hand_presets"]


def hand_presets() -> tuple[EffectorSpec, ...]:
    """Every hand in the presets x synergy, in table order."""
    return tuple(
        vendored_spec(EffectorFamily.CATALOG_HAND, cm, syn)
        for cm in CATALOG_MODELS.values()
        if cm.kind == "hand" and cm.in_presets
        for syn in cm.synergies
    )


class CatalogHandFamily(EffectorFamilyImpl):
    """Vendored dexterous hands (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.CATALOG_HAND
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"

    def presets(self) -> tuple[EffectorSpec, ...]:
        return hand_presets()

    def card_note(self, model: str) -> str | None:
        """The note every page prints beside this model (none of the hands has one)."""
        return catalog_card_note(model)

    def _model(self, spec: EffectorSpec):  # type: ignore[no-untyped-def]
        cm = catalog_model(spec.model)
        if cm.kind != "hand":
            raise ValueError(f"{spec.model} is not a hand")
        return cm, cm.synergy(spec.primitive)

    def build(self, ctx: BuildContext) -> EffectorBuild:
        cm, syn = self._model(ctx.effector)
        return build_vendored(ctx, cm, syn)

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> SynergyPrimitive:
        cm, syn = self._model(spec)
        return synergy_primitive(cm, syn)

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> ContactCriterion:
        cm, _ = self._model(spec)
        return contact_criterion(cm)


register_family(CatalogHandFamily())
