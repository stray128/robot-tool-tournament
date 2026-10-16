"""Design spaces: the one interface the search harness uses for everything design-specific.

The harness (budget, rounds, islands, reserves, repair, reflection, fallback, freezing) is
the same for every design space. What differs is what a design *is*:

* ``box-v1`` (:mod:`rtt.search.space_v1`): today's 10-parameter
  :class:`~rtt.design.schema.FingerDesign`, built from 14 boxes. It is the F1 fallback and
  must behave byte for byte as before, so it only wraps the existing functions.
* ``freeform-v2`` (:mod:`rtt.search.space_v2`): a part-relative
  :class:`FingerRecipe` (9 scored parameters) compiled into a separate left and right jaw
  for each training part.

A :class:`DesignSpace` provides, per the v2 spec: ``tool_schema`` (a slot's design
properties for the forced ``submit_designs`` call), ``sample_sobol`` (the random arm's
seeded sampler), ``canonicalize``, ``validate`` (schema rules, then the geometry
validators against the training parts when they are available), ``build`` (one part),
``points``, ``distance``, ``cell`` and ``describe`` (the parameter table shown in prompts),
plus identity and serialisation (``design_id``, ``dump``, ``load``).

Spaces are looked up by name with :func:`get_space`, or from a design object with
:func:`space_of`. Both import their implementation lazily, so importing this module costs
nothing and has no cycles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

__all__ = [
    "SPACE_NAMES",
    "V1",
    "V2",
    "DesignSpace",
    "Sampler",
    "SpaceCheck",
    "get_space",
    "space_of",
]

V1: Final = "box-v1"
V2: Final = "freeform-v2"
SPACE_NAMES: Final[tuple[str, ...]] = (V1, V2)


@dataclass(frozen=True)
class SpaceCheck:
    """Outcome of :meth:`DesignSpace.validate`.

    ``design`` is the canonical design when ``ok``. ``errors`` are plain-language
    messages the proposer is shown verbatim; ``notes`` record what canonicalisation did
    (inactive parameters ignored, values quantised) and are logged with the proposal;
    ``stage`` names the first validator stage that failed (``schema``, ``analytic``,
    ``build``, ``printability``, ``seat``); ``metrics`` are numbers the validators
    measured (for example the fully open gap).
    """

    ok: bool
    design: Any | None
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    stage: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return "valid" if self.ok else "; ".join(self.errors)


@runtime_checkable
class Sampler(Protocol):
    """A seeded source of valid random designs (arm A, round-0 fill, ``fallback_random``)."""

    draws: int
    rejected: int

    def draw(self) -> Any: ...

    def sample(self, n: int, accept: Any = None) -> list[Any]: ...


@runtime_checkable
class DesignSpace(Protocol):
    """Everything the search harness needs to know about one kind of design."""

    name: str
    #: Terms of the normalised distance vector (v1: 10 parameters, v2: 9).
    n_terms: int
    #: Numeric parameters, in canonical order (the points columns).
    numeric_params: tuple[str, ...]
    #: The two categorical parameters; their pair is a design's cell.
    cell_params: tuple[str, str]

    # identity and serialisation
    def is_design(self, obj: object) -> bool: ...
    def design_id(self, design: Any) -> str: ...
    def dump(self, design: Any) -> dict[str, Any]: ...
    def load(self, data: Mapping[str, Any]) -> Any: ...
    def label(self, design: Any) -> str | None: ...

    # proposer interface
    def tool_schema(self) -> dict[str, Any]: ...
    def raw_params(self, raw: Any) -> dict[str, Any] | None: ...
    def slot_mapping(self, raw: Mapping[str, Any]) -> dict[str, Any]: ...
    def describe(self) -> str: ...

    # sampling, canonical form and validity
    def sample_sobol(self, seed: int, context: Any, **kwargs: Any) -> Sampler: ...
    def canonicalize(self, design: Any) -> tuple[Any, tuple[str, ...]]: ...
    def validate(self, design: Any, context: Any, *, rationale: str | None = None) -> SpaceCheck: ...
    def build(self, design: Any, part: Any, context: Any = None) -> Any: ...
    def geometry_id(self, design: Any, context: Any) -> str | None: ...

    # metric
    def points(self, design: Any) -> dict[str, int] | None: ...
    def distance(self, a: Any, b: Any) -> float: ...
    def cell(self, design: Any) -> tuple[str, str] | None: ...
    def cells(self) -> list[tuple[str, str]]: ...


_CACHE: dict[str, DesignSpace] = {}


def get_space(name: str = V1) -> DesignSpace:
    """The design space called ``name`` (``box-v1`` or ``freeform-v2``)."""
    if name not in _CACHE:
        if name == V1:
            from rtt.search.space_v1 import BoxSpaceV1

            _CACHE[name] = BoxSpaceV1()
        elif name == V2:
            from rtt.search.space_v2 import FreeformSpaceV2

            _CACHE[name] = FreeformSpaceV2()
        else:
            raise KeyError(f"unknown design space {name!r}; choose from {list(SPACE_NAMES)}")
    return _CACHE[name]


def space_of(design: object) -> DesignSpace:
    """The design space a design object belongs to (v1 first: it is the common case)."""
    for name in SPACE_NAMES:
        space = get_space(name)
        if space.is_design(design):
            return space
    raise TypeError(f"{type(design).__name__} is not a design of any known design space")


def names_of(designs: Sequence[object]) -> tuple[str, ...]:
    """Design-space names of ``designs`` (for checks that a run never mixes spaces)."""
    return tuple(dict.fromkeys(space_of(d).name for d in designs))
