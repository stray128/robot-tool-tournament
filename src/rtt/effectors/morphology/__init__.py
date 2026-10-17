"""Morphology designer: our own physics-validated effectors, proposed by Nemotron or at random.

Product mode, not pre-registered (docs/EFFECTORS.md, track B3). The interface the tournament
(track B4) uses:

* :func:`design_spaces`: every designable ``(family, kind) -> ParamSpec`` table (our
  ``fingered_hand``, ``hook_gripper`` and ``segmented_tentacle``, plus the soft arm's and the
  suction cup's spaces once those families are built).
* :func:`validate_recipe`: the validators for one part, cheapest first, as readable reasons.
* :func:`random_recipes`: the equal-budget random control (same spaces, same validators).
* :func:`propose_recipes`: one Nemotron design round for one part (forced
  ``submit_morphologies`` call, at most one repair, the diversity rule), as a
  :class:`ProposalBatch`.
* :func:`spec_for`: the :class:`~rtt.effectors.types.EffectorSpec` of a recipe (through its
  family's ``spec_from_recipe``).
* :class:`DesignOutcome`: the physics feedback on a tested design shown in later rounds.
"""

from __future__ import annotations

from rtt.effectors.morphology.diversity import MIN_POINTS, canonical_recipe, distance_points, screen
from rtt.effectors.morphology.proposer import (
    DESIGNER_MODEL,
    DESIGNER_SAMPLING,
    SUBMIT_TOOL,
    CallRecord,
    DesignOutcome,
    ProposalBatch,
    Rejection,
    make_designer_client,
    propose_recipes,
    submit_morphologies_tool,
)
from rtt.effectors.morphology.random_baseline import random_recipes, random_recipes_with_stats
from rtt.effectors.morphology.spaces import DESIGNED_KINDS, SpaceKey, design_spaces, space_label
from rtt.effectors.morphology.validators import CHECK_CODES, validate_recipe, validation_codes
from rtt.effectors.types import EffectorSpec, MorphologyRecipe

__all__ = [
    "CHECK_CODES",
    "DESIGNED_KINDS",
    "DESIGNER_MODEL",
    "DESIGNER_SAMPLING",
    "MIN_POINTS",
    "SUBMIT_TOOL",
    "CallRecord",
    "DesignOutcome",
    "ProposalBatch",
    "Rejection",
    "SpaceKey",
    "canonical_recipe",
    "design_spaces",
    "distance_points",
    "make_designer_client",
    "propose_recipes",
    "random_recipes",
    "random_recipes_with_stats",
    "screen",
    "space_label",
    "spec_for",
    "submit_morphologies_tool",
    "validate_recipe",
    "validation_codes",
]


def spec_for(recipe: MorphologyRecipe) -> EffectorSpec:
    """The effector of ``recipe``: ``get_family(recipe.family).spec_from_recipe(recipe)``."""
    from rtt.effectors.registry import get_family

    return get_family(recipe.family).spec_from_recipe(recipe)
