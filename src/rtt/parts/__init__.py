"""Procedural v2 parts: exact convex pieces, features, materials and registration symmetry.

:mod:`rtt.parts.types` holds the frozen contract shared with the jaw generator, the scene
and the renderer. This package is separate from :mod:`rtt.part`, the product-mode intake
of uploaded parts.
"""

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

__all__ = [
    "FEATURE_KINDS",
    "REGISTER_AXES",
    "Feature",
    "FeatureKind",
    "JawSide",
    "Material",
    "PartSpec",
    "RegisterAxis",
]
