"""Procedural v2 parts: exact convex pieces, features, materials and registration symmetry.

:mod:`rtt.parts.types` holds the frozen contract shared with the jaw generator, the scene
and the renderer. This package is separate from :mod:`rtt.part`, the product-mode intake
of uploaded parts.

Modules: :mod:`~rtt.parts.pieces` (exact convex prisms and their mass properties),
:mod:`~rtt.parts.materials`, :mod:`~rtt.parts.library` (part families, the training parts of
``configs/parts_v2.yaml``), :mod:`~rtt.parts.features` (feature checks, yaw symmetry),
:mod:`~rtt.parts.suite` (``train_v2``), :mod:`~rtt.parts.heldout` (``heldout_procedural_v2``)
and :mod:`~rtt.parts.preview` (clay thumbnails).

Importing the package loads only the contract types (NumPy); the conveniences below are
imported on first use.
"""

from __future__ import annotations

from typing import Any

from rtt.parts.types import (
    FEATURE_KINDS,
    REGISTER_AXES,
    Feature,
    FeatureKind,
    JawSide,
    Material,
    PartSpec,
    RegisterAxis,
)

_LAZY = {
    "MATERIALS": "rtt.parts.materials",
    "build_part": "rtt.parts.library",
    "get_part": "rtt.parts.library",
    "library_digest": "rtt.parts.library",
    "part_table": "rtt.parts.library",
    "training_parts": "rtt.parts.library",
    "visual_mesh": "rtt.parts.library",
    "check_features": "rtt.parts.features",
    "train_suite": "rtt.parts.suite",
    "PartCondition": "rtt.parts.suite",
    "PartSuite": "rtt.parts.suite",
    "heldout_procedural_suite": "rtt.parts.heldout",
}

__all__ = [
    "FEATURE_KINDS",
    "MATERIALS",
    "REGISTER_AXES",
    "Feature",
    "FeatureKind",
    "JawSide",
    "Material",
    "PartCondition",
    "PartSpec",
    "PartSuite",
    "RegisterAxis",
    "build_part",
    "check_features",
    "get_part",
    "heldout_procedural_suite",
    "library_digest",
    "part_table",
    "train_suite",
    "training_parts",
    "visual_mesh",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'rtt.parts' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
