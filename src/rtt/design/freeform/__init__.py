"""Freeform v2 fingers: part-relative recipes compiled into exact convex jaw pairs.

The v1 box-finger package (:mod:`rtt.design`) is untouched and stays the fallback.
:mod:`rtt.design.freeform.types` holds the frozen contract shared with the simulator,
the renderer and the search; import the public names from this package.

Importing the package stays numpy-only: the recipe schema (pydantic), the generator
(scipy, manifold3d), the baselines, the predictor and the exporters load on first use of
their names (``FingerRecipe``, ``Baseline``, ``build_pair_full``, ...).
"""

from __future__ import annotations

import importlib
from typing import Any

from rtt.design.freeform.types import (
    CAPTURE_OUTCOMES,
    CARRIAGE_INNER_Y,
    DESIGN_SPACE,
    MAX_PAD_SEGMENTS,
    MAX_PIECES_PER_JAW,
    PLA_DENSITY,
    ROOT_PLANE_PALM_Z,
    CaptureOutcome,
    CaptureReport,
    FingerPair,
    JawGeometry,
    build_pair,
)

_LAZY = {
    "FingerRecipe": "rtt.design.freeform.schema",
    "recipe_from_unit": "rtt.design.freeform.schema",
    "recipe_json_schema": "rtt.design.freeform.schema",
    "SCORED_PARAMS": "rtt.design.freeform.schema",
    "PARAM_BOUNDS": "rtt.design.freeform.schema",
    "CATEGORICAL_OPTIONS": "rtt.design.freeform.schema",
    "canonicalize": "rtt.design.freeform.canon",
    "canonical_json": "rtt.design.freeform.canon",
    "design_id": "rtt.design.freeform.hashing",
    "geometry_id": "rtt.design.freeform.hashing",
    "GENERATOR_VERSION": "rtt.design.freeform.hashing",
    "build_pair_full": "rtt.design.freeform.generate",
    "PairBuild": "rtt.design.freeform.generate",
    "InvalidDesign": "rtt.design.freeform.validate",
    "validate_design": "rtt.design.freeform.validate",
    "DesignReport": "rtt.design.freeform.validate",
    "Baseline": "rtt.design.freeform.baselines",
    "load_entrants_v2": "rtt.design.freeform.baselines",
    "recipe_distance": "rtt.design.freeform.distance",
    "recipe_cell": "rtt.design.freeform.distance",
    "capture_report": "rtt.design.freeform.predict",
    "export_pair": "rtt.design.freeform.export",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


__all__ = [
    "CAPTURE_OUTCOMES",
    "CARRIAGE_INNER_Y",
    "CATEGORICAL_OPTIONS",
    "DESIGN_SPACE",
    "GENERATOR_VERSION",
    "MAX_PAD_SEGMENTS",
    "MAX_PIECES_PER_JAW",
    "PARAM_BOUNDS",
    "PLA_DENSITY",
    "ROOT_PLANE_PALM_Z",
    "SCORED_PARAMS",
    "Baseline",
    "CaptureOutcome",
    "CaptureReport",
    "DesignReport",
    "FingerPair",
    "FingerRecipe",
    "InvalidDesign",
    "JawGeometry",
    "PairBuild",
    "build_pair",
    "build_pair_full",
    "canonical_json",
    "canonicalize",
    "capture_report",
    "design_id",
    "export_pair",
    "geometry_id",
    "load_entrants_v2",
    "recipe_cell",
    "recipe_distance",
    "recipe_from_unit",
    "recipe_json_schema",
    "validate_design",
]
