"""Freeform v2 fingers: part-relative recipes compiled into exact convex jaw pairs.

The v1 box-finger package (:mod:`rtt.design`) is untouched and stays the fallback.
:mod:`rtt.design.freeform.types` holds the frozen contract shared with the simulator,
the renderer and the search; import the public names from this package.
"""

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

__all__ = [
    "CAPTURE_OUTCOMES",
    "CARRIAGE_INNER_Y",
    "DESIGN_SPACE",
    "MAX_PAD_SEGMENTS",
    "MAX_PIECES_PER_JAW",
    "PLA_DENSITY",
    "ROOT_PLANE_PALM_Z",
    "CaptureOutcome",
    "CaptureReport",
    "FingerPair",
    "JawGeometry",
    "build_pair",
]
