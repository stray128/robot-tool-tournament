"""Canonicalisation of v2 recipes: identical for every arm, before hashing, building or distances.

1. Inactive parameters are reset to their fixed values (:data:`~rtt.design.freeform.schema.FIXED_VALUES`)
   and every reset that changes a value is logged in plain language.
2. Continuous parameters are quantised: 0.1 mm, 0.5 degrees, 0.01 for fractions.

The canonical form is what :func:`rtt.design.freeform.hashing.design_id` hashes, so two
recipes that differ only in inactive parameters, in quantisation noise or in ``name`` /
``rationale`` get the same id.
"""

from __future__ import annotations

import json
from typing import Any

from rtt.design.freeform.schema import (
    CONTINUOUS_PARAMS,
    FIXED_VALUES,
    PRODUCT_DEFAULTS,
    FingerRecipe,
    inactive_params,
    quantized_values,
)


def canonicalize(recipe: FingerRecipe) -> tuple[FingerRecipe, tuple[str, ...]]:
    """The canonical recipe (every parameter set, quantised) and the log of what changed."""
    if not isinstance(recipe, FingerRecipe):
        raise TypeError(f"expected a FingerRecipe, got {type(recipe).__name__}")
    values = quantized_values(recipe)
    inactive = set(inactive_params(recipe.pad, recipe.tip))
    repairs: list[str] = []
    for name in CONTINUOUS_PARAMS:
        raw = getattr(recipe, name)
        new = values[name]
        if name in inactive:
            if raw is not None and float(raw) != FIXED_VALUES[name]:
                repairs.append(
                    f"{name} is inactive for pad={recipe.pad}, tip={recipe.tip}: {raw:g} reset to {new:g}"
                )
        elif float(raw) != new:
            repairs.append(f"{name} quantised from {raw!r} to {new:g}")
    canonical = FingerRecipe(name=recipe.name, rationale=recipe.rationale, **values)
    return canonical, tuple(repairs)


def canonical_dict(recipe: FingerRecipe) -> dict[str, Any]:
    """The geometry-relevant fields of the canonical recipe (no ``name`` or ``rationale``)."""
    values = quantized_values(recipe)
    out = {k: values[k] for k in ("pad", "tip", *CONTINUOUS_PARAMS)}
    for k in PRODUCT_DEFAULTS:
        out[k] = values[k]
    return out


def canonical_json(recipe: FingerRecipe) -> str:
    """Stable JSON of :func:`canonical_dict`: sorted keys, no spaces, shortest float repr."""
    return json.dumps(canonical_dict(recipe), sort_keys=True, separators=(",", ":"))
