"""``box-v1``: today's 10-parameter box finger, wrapped as a :class:`~rtt.search.space.DesignSpace`.

Every method delegates to the existing v1 code (:mod:`rtt.design.schema`,
:mod:`rtt.design.validate`, :mod:`rtt.design.generate`, :mod:`rtt.search.diversity`,
:class:`rtt.search.random_search.DesignSampler`), so a v1 run through the space interface
is byte for byte the run it was before. v1 stays the F1 fallback; nothing here may change
what a v1 run does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.search.space import V1, SpaceCheck

__all__ = ["BoxSpaceV1"]


class BoxSpaceV1:
    """The v1 design space (see module docstring)."""

    name: Final = V1
    n_terms: Final = len(GEOMETRY_PARAMS)
    numeric_params: Final = tuple(p for p in GEOMETRY_PARAMS if p not in CATEGORICAL_OPTIONS)
    cell_params: Final = ("inner_face", "tip_profile")
    param_names: Final = GEOMETRY_PARAMS

    # ------------------------------------------------------------------ identity

    def is_design(self, obj: object) -> bool:
        return isinstance(obj, FingerDesign)

    def design_id(self, design: FingerDesign) -> str:
        return design.design_id()

    def dump(self, design: FingerDesign) -> dict[str, Any]:
        return design.model_dump(mode="json")

    def load(self, data: Mapping[str, Any]) -> FingerDesign:
        return FingerDesign.model_validate(data)

    def label(self, design: FingerDesign) -> str | None:
        return design.name

    # ------------------------------------------------------------------ proposer interface

    def tool_schema(self) -> dict[str, Any]:
        """The 10 parameters as JSON-schema properties, as every v1 slot lists them."""
        properties: dict[str, Any] = {}
        for name in GEOMETRY_PARAMS:
            if name in CATEGORICAL_OPTIONS:
                properties[name] = {"type": "string", "enum": list(CATEGORICAL_OPTIONS[name])}
            else:
                lo, hi = PARAM_BOUNDS[name]
                properties[name] = {"type": "number", "minimum": lo, "maximum": hi}
        return properties

    def raw_params(self, raw: Any) -> dict[str, Any] | None:
        from rtt.search.nemotron import _raw_geometry

        return _raw_geometry(raw)

    def slot_mapping(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        return {k: raw[k] for k in GEOMETRY_PARAMS if k in raw}

    def describe(self) -> str:
        from rtt.search.nemotron import render_parameters

        return render_parameters()

    # ------------------------------------------------------------------ sampling and validity

    def sample_sobol(self, seed: int, context: GripperTask, **kwargs: Any) -> Any:
        from rtt.search.random_search import DesignSampler

        return DesignSampler(seed, context, **kwargs)

    def canonicalize(self, design: Any) -> tuple[FingerDesign, tuple[str, ...]]:
        """v1 has no inactive parameters to reset: the design is already canonical."""
        if not isinstance(design, FingerDesign):
            design = FingerDesign.model_validate(design)
        return design, ()

    def validate(
        self, design: Any, context: GripperTask | None, *, rationale: str | None = None
    ) -> SpaceCheck:
        from rtt.design.validate import validate

        if rationale and isinstance(design, Mapping):
            design = {**design, "rationale": rationale}
        result = validate(design, context)
        return SpaceCheck(
            ok=result.ok,
            design=result.design if result.ok else None,
            errors=tuple(result.errors),
            stage=None if result.ok else "schema",
            metrics=dict(result.metrics),
        )

    def build(self, design: FingerDesign, part: Any = None, context: Any = None) -> Any:
        """v1 fingers do not depend on the part: one left finger, mirrored by the scene."""
        from rtt.design.generate import generate_finger

        return generate_finger(design)

    def geometry_id(self, design: FingerDesign, context: Any) -> str | None:
        return None

    # ------------------------------------------------------------------ metric

    def points(self, design: FingerDesign) -> dict[str, int]:
        from rtt.search.diversity import points_of

        return points_of(design)

    def distance(self, a: FingerDesign, b: FingerDesign) -> float:
        from rtt.design.distance import design_distance

        return design_distance(a, b)

    def cell(self, design: FingerDesign) -> tuple[str, str]:
        return (design.inner_face, design.tip_profile)

    def cells(self) -> list[tuple[str, str]]:
        from rtt.search.diversity import all_cells

        return all_cells()
