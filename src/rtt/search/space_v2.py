"""``freeform-v2``: part-relative finger recipes as a :class:`~rtt.search.space.DesignSpace`.

A v2 design is a **recipe** (spec section 4): a pad strategy, a tip strategy and up to 7
numbers. The generator (track G, :mod:`rtt.design.freeform`) compiles one recipe into a
separate left and right jaw for each training part; the search only ever handles recipes.

Scored parameters
-----------------

==================  ==========================  =======================================
parameter           bounds                      inactive when (fixed value)
==================  ==========================  =======================================
pad                 flat | conform | v_funnel   -
tip                 blunt | chamfer | hook      -
grip_height_frac    0.15 - 0.85                 -
pad_height_mm       8 - 30                      -
finger_width_mm     14 - 40                     -
pad_depth_mm        0 - 15                      flat (0)
wall_draft_deg      0 - 70                      flat (0)
clearance_mm        0.2 - 1.5                   flat, v_funnel (0.2)
tip_size_mm         0 - 8                       blunt (0)
==================  ==========================  =======================================

Strategy rules: a ``v_funnel`` needs ``wall_draft_deg`` in [15, 70] and ``pad_depth_mm >= 3``;
a ``conform`` pad needs ``pad_depth_mm >= 1``; a ``hook`` needs ``tip_size_mm >= 2``. Two fit
rules follow track G's schema: a hook needs ``pad_height_mm >= min(tip_size_mm, 4) +
tip_size_mm``, and a chamfer needs ``tip_size_mm <= pad_height_mm - 0.5``.

**Canonicalisation** (identical for every arm, before hashing, building or measuring
distance): inactive parameters are reset to their fixed values (logged), then values are
quantised to 0.1 mm, 0.5 deg and 0.01 for the fraction. ``design_id`` is the SHA-256 of the
generator version and the canonical recipe, so two proposals that differ only in inactive
parameters or below the quantisation step are the same design.

**Distance** is the mean over the 9 terms of the normalised vector: ``|a - b| / range`` for
the 7 numbers and 0/1 for ``pad`` and ``tip`` (the same construction as v1, on 9 terms).
A design's *cell* is its ``(pad, tip)`` pair: 9 cells.

Track G dependency
------------------

Until track G lands, this module carries its own implementation of the section-4 schema
(:class:`FingerRecipeStub`), canonicalisation and ``design_id``, written from the spec.
Each piece switches to track G's module as soon as it imports (:func:`track_g`):

* ``rtt.design.freeform.schema.FingerRecipe``: canonical recipes are G's model (its extra
  fit rules, such as a hook that must fit inside the pad, then apply at the schema stage);
* ``rtt.design.freeform.hashing``: ``design_id`` of G's recipes and baselines, the generator
  version and ``geometry_id``, so search records, built pairs and replays agree;
* ``rtt.design.freeform.validate.validate_design``: the section-13 battery on the training
  parts (its codes map to stages through :data:`G_CODE_STAGES`); its geometry id and pairs
  are cached on the :class:`V2Context`;
* ``rtt.design.freeform.build_pair`` once ``rtt.design.freeform.generate`` exists;
* ``rtt.design.freeform.baselines``: ``Baseline`` and ``load_entrants_v2``.

Without them a design is checked against the schema rules only, and
:meth:`FreeformSpaceV2.geometry_id` returns ``None`` (so ``duplicate_geometry`` is skipped)
unless the context supplies a geometry hash function. Quantisation rounds half-up, as G's
schema does. ``tests/test_search_space.py`` cross-checks the stub against track G as soon as
G's modules import.

Baselines (the stock blade, the configurator and swept imprints) are entrants that are not
recipes. They are carried as track G's ``Baseline`` (or :class:`BaselineEntrant` until G
lands): a name, an optional recipe *point* that places them in the metric (so a proposal
that copies the flat entrant is a duplicate), and what G needs to build them. A baseline
that is only a recipe point (``flat``) builds and hashes as that recipe; the stock blade has
no point and is never run through the recipe validators.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rtt.llm.cache import canonical_json
from rtt.search.space import V2, SpaceCheck

__all__ = [
    "CELL_PARAMS",
    "GENERATOR_VERSION_STUB",
    "INACTIVE",
    "NUMERIC_BOUNDS",
    "NUMERIC_PARAMS",
    "PAD_OPTIONS",
    "POINT_COLUMNS",
    "QUANT_STEPS",
    "RECIPE_KIND",
    "SCORED_PARAMS",
    "TIP_OPTIONS",
    "BaselineEntrant",
    "FingerRecipeStub",
    "FreeformSpaceV2",
    "V2Context",
    "canonical_params",
    "quantise",
    "track_g",
]

PadStrategy = Literal["flat", "conform", "v_funnel"]
TipStrategy = Literal["blunt", "chamfer", "hook"]
PAD_OPTIONS: Final[tuple[str, ...]] = ("flat", "conform", "v_funnel")
TIP_OPTIONS: Final[tuple[str, ...]] = ("blunt", "chamfer", "hook")
CELL_PARAMS: Final[tuple[str, str]] = ("pad", "tip")

#: Inclusive bounds of the 7 numeric parameters, in canonical (Sobol and points) order.
NUMERIC_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "grip_height_frac": (0.15, 0.85),
    "pad_height_mm": (8.0, 30.0),
    "finger_width_mm": (14.0, 40.0),
    "pad_depth_mm": (0.0, 15.0),
    "wall_draft_deg": (0.0, 70.0),
    "clearance_mm": (0.2, 1.5),
    "tip_size_mm": (0.0, 8.0),
}
NUMERIC_PARAMS: Final[tuple[str, ...]] = tuple(NUMERIC_BOUNDS)
#: The 9 scored parameters, in the order prompts and tools list them.
SCORED_PARAMS: Final[tuple[str, ...]] = ("pad", "tip", *NUMERIC_PARAMS)
#: Quantisation step of every numeric parameter (0.1 mm, 0.5 deg, 0.01 for the fraction).
QUANT_STEPS: Final[dict[str, float]] = {
    "grip_height_frac": 0.01,
    "pad_height_mm": 0.1,
    "finger_width_mm": 0.1,
    "pad_depth_mm": 0.1,
    "wall_draft_deg": 0.5,
    "clearance_mm": 0.1,
    "tip_size_mm": 0.1,
}
#: parameter -> (strategy field, strategies where it is inactive, fixed value).
INACTIVE: Final[dict[str, tuple[str, tuple[str, ...], float]]] = {
    "pad_depth_mm": ("pad", ("flat",), 0.0),
    "wall_draft_deg": ("pad", ("flat",), 0.0),
    "clearance_mm": ("pad", ("flat", "v_funnel"), 0.2),
    "tip_size_mm": ("tip", ("blunt",), 0.0),
}
V_FUNNEL_DRAFT: Final = (15.0, 70.0)
V_FUNNEL_MIN_DEPTH: Final = 3.0
CONFORM_MIN_DEPTH: Final = 1.0
HOOK_MIN_TIP: Final = 2.0
#: Fit rules of track G's schema: a hook (height min(tip_size, 4) mm plus its protrusion under a
#: 45 deg top) fits inside the pad, and a chamfer leaves at least 0.5 mm of the pad face.
HOOK_MAX_HEIGHT: Final = 4.0
CHAMFER_LAND: Final = 0.5
#: Short names of the numeric parameters in points tables.
POINT_COLUMNS: Final[dict[str, str]] = {
    "grip_height_frac": "grip",
    "pad_height_mm": "pad_h",
    "finger_width_mm": "width",
    "pad_depth_mm": "depth",
    "wall_draft_deg": "draft",
    "clearance_mm": "clear",
    "tip_size_mm": "tip_size",
}
_MEANING: Final[dict[str, str]] = {
    "pad": (
        "pad face strategy: flat plane; conform = imprint of the part's outline plus clearance, with drafted walls; "
        "v_funnel = a V that just touches the part"
    ),
    "tip": "tip strategy: blunt end; chamfer = 45 deg lead-in bevel; hook = a lip under the part",
    "grip_height_frac": "pad centre height as a fraction of the part height",
    "pad_height_mm": "height of the pad band",
    "finger_width_mm": "jaw width across the closing axis (tool envelope +-20 mm)",
    "pad_depth_mm": "how far the pocket or V reaches past the first-contact plane (wrap depth)",
    "wall_draft_deg": "wall angle from the closing axis: 0 = straight imprint walls; for a V, the flank angle",
    "clearance_mm": "gap between the part's outline and the conform pad",
    "tip_size_mm": "chamfer leg, or hook protrusion (hook height = min(tip_size, 4) mm)",
}
#: Stand-in generator version, used in ``design_id`` until track G's generator version exists.
GENERATOR_VERSION_STUB: Final = "freeform-v2/search-stub-1"
#: Built jaw pairs a :class:`V2Context` keeps (about 40 designs x 6 parts per run, plus margin).
PAIR_CACHE: Final = 2048


# --------------------------------------------------------------------------- schema stub


class FingerRecipeStub(BaseModel):
    """Spec section 4 recipe, as the search needs it until track G's ``FingerRecipe`` lands.

    Strict (``extra="forbid"``, frozen). Bounds and strategy rules are enforced; the
    canonical form (inactive parameters reset, quantised) is produced by
    :func:`canonical_params`, and the space only ever constructs canonical recipes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pad: PadStrategy
    tip: TipStrategy
    grip_height_frac: float = Field(ge=0.15, le=0.85)
    pad_height_mm: float = Field(ge=8.0, le=30.0)
    finger_width_mm: float = Field(ge=14.0, le=40.0)
    pad_depth_mm: float = Field(ge=0.0, le=15.0)
    wall_draft_deg: float = Field(ge=0.0, le=70.0)
    clearance_mm: float = Field(ge=0.2, le=1.5)
    tip_size_mm: float = Field(ge=0.0, le=8.0)

    @model_validator(mode="after")
    def _strategy_rules(self) -> FingerRecipeStub:
        errors = strategy_errors(self.model_dump())
        if errors:
            raise ValueError("; ".join(errors))
        return self


def strategy_errors(params: Mapping[str, Any]) -> list[str]:
    """Violations of the three strategy rules, in words (empty when the recipe obeys them)."""
    errors: list[str] = []
    pad, tip = params.get("pad"), params.get("tip")
    depth, draft, size = params.get("pad_depth_mm"), params.get("wall_draft_deg"), params.get("tip_size_mm")
    if pad == "v_funnel":
        lo, hi = V_FUNNEL_DRAFT
        if isinstance(draft, float | int) and not lo <= draft <= hi:
            errors.append(f"a v_funnel pad needs wall_draft_deg between {lo:g} and {hi:g} (got {draft:g})")
        if isinstance(depth, float | int) and depth < V_FUNNEL_MIN_DEPTH:
            errors.append(f"a v_funnel pad needs pad_depth_mm >= {V_FUNNEL_MIN_DEPTH:g} (got {depth:g})")
    if pad == "conform" and isinstance(depth, float | int) and depth < CONFORM_MIN_DEPTH:
        errors.append(f"a conform pad needs pad_depth_mm >= {CONFORM_MIN_DEPTH:g} (got {depth:g})")
    if tip == "hook" and isinstance(size, float | int) and size < HOOK_MIN_TIP:
        errors.append(f"a hook tip needs tip_size_mm >= {HOOK_MIN_TIP:g} (got {size:g})")
    pad_h = params.get("pad_height_mm")
    if tip == "hook" and isinstance(size, float | int) and isinstance(pad_h, float | int) and not errors:
        need = min(size, HOOK_MAX_HEIGHT) + size
        if pad_h < need - 1e-9:
            errors.append(
                f"a hook with tip_size_mm {size:g} needs pad_height_mm >= {need:g} (hook height "
                f"{min(size, HOOK_MAX_HEIGHT):g} + protrusion {size:g} under its 45 deg top), got {pad_h:g}"
            )
    chamfer = tip == "chamfer" and isinstance(size, float | int) and isinstance(pad_h, float | int)
    if chamfer and size > pad_h - CHAMFER_LAND + 1e-9:
        errors.append(
            f"a chamfer needs tip_size_mm <= pad_height_mm - {CHAMFER_LAND:g} (got {size:g} on a {pad_h:g} mm pad)"
        )
    return errors


def quantise(name: str, value: float) -> float:
    """``value`` rounded half-up to its step, as a short decimal (the same rule as track G's schema)."""
    step = QUANT_STEPS[name]
    steps = math.floor(float(value) / step + 0.5 + 1e-9)
    return round(steps * step, 2 if step < 0.1 else 1) + 0.0


def _short(value: object, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def canonical_params(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, tuple[str, ...], tuple[str, ...]]:
    """Spec section 4 canonicalisation of a raw recipe mapping.

    Returns ``(canonical params or None, notes, errors)``. Unknown keys, unknown strategies,
    missing active parameters and non-numbers are errors; inactive parameters are reset to
    their fixed values whether or not they were given (a given value is noted as ignored);
    active values are quantised; then the bounds and the strategy rules are checked.
    """
    notes: list[str] = []
    errors: list[str] = []
    unknown = sorted(str(k) for k in raw if k not in SCORED_PARAMS)
    if unknown:
        errors.append(f"unknown parameter(s): {', '.join(unknown)}; allowed: {', '.join(SCORED_PARAMS)}")
    pad, tip = raw.get("pad"), raw.get("tip")
    if pad not in PAD_OPTIONS:
        errors.append(f"pad must be one of {' / '.join(PAD_OPTIONS)}, got {_short(pad)}")
    if tip not in TIP_OPTIONS:
        errors.append(f"tip must be one of {' / '.join(TIP_OPTIONS)}, got {_short(tip)}")
    if errors and (pad not in PAD_OPTIONS or tip not in TIP_OPTIONS):
        return None, (), tuple(errors)
    out: dict[str, Any] = {"pad": pad, "tip": tip}
    quantised: list[str] = []
    for name in NUMERIC_PARAMS:
        lo, hi = NUMERIC_BOUNDS[name]
        rule = INACTIVE.get(name)
        if rule is not None and (pad if rule[0] == "pad" else tip) in rule[1]:
            fixed = rule[2]
            if name in raw and raw[name] is not None:
                value = raw[name]
                same = isinstance(value, numbers.Real) and not isinstance(value, bool) and value == fixed
                if not same:
                    owner = f"a {pad} pad" if rule[0] == "pad" else f"a {tip} tip"
                    notes.append(f"{name} is inactive for {owner}: {_short(value)} ignored, set to {fixed:g}")
            out[name] = fixed
            continue
        if name not in raw or raw[name] is None:
            owner = f"pad '{pad}'" if rule is not None and rule[0] == "pad" else f"tip '{tip}'"
            needs = owner if rule is not None else "every design"
            errors.append(f"{needs} needs {name} ({lo:g} to {hi:g})")
            continue
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            errors.append(f"{name} must be a finite number, got {_short(value)}")
            continue
        q = quantise(name, float(value))
        if abs(q - float(value)) > 1e-9:
            quantised.append(f"{name} {float(value):g} -> {q:g}")
        if not lo - 1e-9 <= q <= hi + 1e-9:
            errors.append(f"{name} = {q:g} is outside {lo:g} to {hi:g}")
            continue
        out[name] = q
    if quantised:
        notes.append("quantised: " + ", ".join(quantised))
    if not errors:
        errors.extend(strategy_errors(out))
    if errors:
        return None, tuple(notes), tuple(errors)
    return out, tuple(notes), ()


# --------------------------------------------------------------------------- track G adapters


@dataclass(frozen=True)
class _TrackG:
    """Track G's v2 modules, when they import (each piece ``None`` until G lands).

    * ``recipe_cls``: ``rtt.design.freeform.schema.FingerRecipe``;
    * ``baseline_cls`` / ``load_entrants``: ``rtt.design.freeform.baselines.Baseline`` and
      ``load_entrants_v2`` (``configs/entrants_v2.yaml``);
    * ``validate``: ``rtt.design.freeform.validate.validate_design(design, parts, task)``, the
      section-13 battery, returning a report with ``valid``, ``code``, ``message`` and
      ``geometry_id``;
    * ``build_pair``: the frozen ``rtt.design.freeform.build_pair``, only once the generator
      (``rtt.design.freeform.generate``) exists to implement it;
    * ``design_id`` / ``geometry_id`` / ``generator_version``: ``rtt.design.freeform.hashing``.
    """

    recipe_cls: type | None = None
    baseline_cls: type | None = None
    load_entrants: Callable[..., Any] | None = None
    validate: Callable[..., Any] | None = None
    build_pair: Callable[..., Any] | None = None
    design_id: Callable[[Any], str] | None = None
    geometry_id: Callable[[Mapping[str, str]], str] | None = None
    generator_version: str | None = None


def _g_attr(module: str, *names: str) -> Any:
    """The first of ``names`` that ``module`` defines, or ``None`` when it does not import."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError:
        return None
    for name in names:
        value = getattr(mod, name, None)
        if value is not None:
            return value
    return None


@cache
def track_g() -> _TrackG:
    """Resolve track G's v2 modules once. Every missing piece falls back to this module's stub."""
    build = None
    if _g_attr("rtt.design.freeform.generate", "build_pair") is not None:
        build = _g_attr("rtt.design.freeform", "build_pair")  # the frozen signature, now implemented
    return _TrackG(
        recipe_cls=_g_attr("rtt.design.freeform.schema", "FingerRecipe"),
        baseline_cls=_g_attr("rtt.design.freeform.baselines", "Baseline"),
        load_entrants=_g_attr("rtt.design.freeform.baselines", "load_entrants_v2", "load_entrants"),
        validate=_g_attr("rtt.design.freeform.validate", "validate_design"),
        build_pair=build,
        design_id=_g_attr("rtt.design.freeform.hashing", "design_id"),
        geometry_id=_g_attr("rtt.design.freeform.hashing", "geometry_id"),
        generator_version=_g_attr("rtt.design.freeform.hashing", "GENERATOR_VERSION")
        or _g_attr("rtt.design.freeform.generate", "GENERATOR_VERSION"),
    )


def _generator_version() -> str:
    return track_g().generator_version or GENERATOR_VERSION_STUB


def _is_g_object(obj: object) -> bool:
    """True for track G's own recipe and baseline objects (their ids come from G's hashing)."""
    g = track_g()
    return any(cls is not None and isinstance(obj, cls) for cls in (g.recipe_cls, g.baseline_cls))


#: Validator codes of track G (``rtt.design.freeform.validate.CODES``) -> section-13 stage.
G_CODE_STAGES: Final[dict[str, str]] = {
    "schema": "schema",
    "outline_empty": "analytic",
    "open_clearance": "analytic",
    "pads_overlap": "analytic",
    "q_seat_range": "analytic",
    "length": "analytic",
    "width": "analytic",
    "build_failed": "build",
    "not_convex": "build",
    "too_many_pieces": "build",
    "print_not_manifold": "build",
    "min_wall": "printability",
    "min_feature": "printability",
    "mass": "printability",
    "body_slope": "printability",
    "pad_cuts_part": "seat",
    "seat_distance": "seat",
    "jaws_intersect": "seat",
    "table_clearance": "seat",
    "duplicate_geometry": "duplicate",
    "too_similar": "duplicate",
}


# --------------------------------------------------------------------------- baselines


#: ``BaselineEntrant.kind`` of an entrant that is simply a named recipe (for example ``flat``).
RECIPE_KIND: Final = "recipe"
#: ``source`` of a dumped track-G entrant (``configs/entrants_v2.yaml``), so ``load`` rebuilds G's ``Baseline``.
ENTRANTS_SOURCE: Final = "entrants_v2"
#: Track G ``Baseline`` fields kept in a dump's ``params`` besides its name, kind, recipe and rationale.
_G_BASELINE_FIELDS: Final = ("sweep_mm", "depth_rule", "round0", "reference", "hulls")


@dataclass(frozen=True)
class BaselineEntrant:
    """An entrant that is not a recipe: the stock blade, the configurator or the swept imprint.

    ``point`` is the recipe that places the baseline in the design metric (``None`` for the
    stock blade, which has no recipe); ``params`` are what the generator needs to build it
    (for example ``{"sweep_mm": 2.0}``). Its id hashes the kind, name, params and point,
    except for ``kind="recipe"`` (a named recipe entrant such as ``flat``), whose id is the
    recipe's own id, so a proposal that repeats it is a ``duplicate_design``.
    """

    name: str
    kind: str
    point: Any | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    rationale: str | None = None

    def design_id(self) -> str:
        from rtt.search.space import get_space

        space = get_space(V2)
        if self.kind == RECIPE_KIND and self.point is not None:
            return space.design_id(self.point)  # a named recipe entrant is that recipe
        payload = {
            "generator": _generator_version(),
            "baseline": self.kind,
            "name": self.name,
            "params": dict(self.params),
            "point": space.dump(self.point) if self.point is not None else None,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_baseline(obj: object) -> bool:
    g = track_g().baseline_cls
    return isinstance(obj, BaselineEntrant) or (g is not None and isinstance(obj, g))


def _is_named_recipe(obj: object) -> bool:
    """A baseline that is just a recipe point (``flat``, ``ref_wrap``): it builds and hashes as that recipe."""
    if isinstance(obj, BaselineEntrant):
        return obj.kind == RECIPE_KIND and obj.point is not None
    return _is_baseline(obj) and bool(getattr(obj, "is_point", False))


def _is_stock(obj: object) -> bool:
    """The stock blade: a fixed asset with no recipe (trusted, never run through the recipe validators)."""
    return _is_baseline(obj) and _baseline_point(obj) is None


def _baseline_point(obj: Any) -> Any | None:
    for attr in ("point", "recipe"):
        value = getattr(obj, attr, None)
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------- validation context


@dataclass(frozen=True)
class V2Context:
    """What a v2 validator and builder need: the training parts and the v2 task.

    * ``parts``: training parts (:class:`rtt.parts.types.PartSpec`) in training order.
    * ``task``: the v2 task (``rtt.sim.task.TaskV2``).
    * ``part_ids``: ids of the training parts (taken from ``parts`` when given).
    * ``geometry_hash_fn``: ``(design, part_id) -> geometry hash``; overrides the builder for
      ``geometry_id`` (offline fakes, tests).
    * ``validator``: ``(design, context) -> SpaceCheck | None`` for the geometry stages;
      overrides track G's validators.

    The search keeps it in :attr:`rtt.search.base.RunContext.gripper_task`, where v1 keeps
    its :class:`~rtt.design.schema.GripperTask`. Built pairs and geometry ids are cached on the
    context (they depend on its parts and task), never on the process-wide space.
    """

    parts: tuple[Any, ...] = ()
    task: Any = None
    part_ids: tuple[str, ...] = ()
    geometry_hash_fn: Callable[[Any, str], str] | None = None
    validator: Callable[[Any, V2Context], SpaceCheck | None] | None = None
    label: str = ""
    pairs: OrderedDict[tuple[str, str], Any] = field(default_factory=OrderedDict, compare=False, repr=False)
    """``(design_id, part_id) -> FingerPair`` (least recently used dropped beyond :data:`PAIR_CACHE`)."""
    geometry: dict[str, str | None] = field(default_factory=dict, compare=False, repr=False)
    """``design_id -> geometry_id`` on these parts (``None`` when it cannot be computed)."""

    def __post_init__(self) -> None:
        if self.parts and not self.part_ids:
            object.__setattr__(self, "part_ids", tuple(str(p.part_id) for p in self.parts))

    def remember_pair(self, design_id: str, part_id: str, pair: Any) -> None:
        self.pairs[(design_id, part_id)] = pair
        self.pairs.move_to_end((design_id, part_id))
        while len(self.pairs) > PAIR_CACHE:
            self.pairs.popitem(last=False)

    def to_json(self) -> dict[str, Any]:
        """Manifest summary (part ids only: the parts are hashed into the suite hash)."""
        return {"design_space": V2, "part_ids": list(self.part_ids), "label": self.label}


# --------------------------------------------------------------------------- the space


class FreeformSpaceV2:
    """The v2 design space (see module docstring)."""

    name: Final = V2
    n_terms: Final = len(SCORED_PARAMS)
    numeric_params: Final = NUMERIC_PARAMS
    cell_params: Final = CELL_PARAMS
    param_names: Final = SCORED_PARAMS
    categorical_options: Final = {"pad": PAD_OPTIONS, "tip": TIP_OPTIONS}

    # ------------------------------------------------------------------ identity

    def recipe_cls(self) -> type:
        return track_g().recipe_cls or FingerRecipeStub

    def is_recipe(self, obj: object) -> bool:
        g = track_g().recipe_cls
        return isinstance(obj, FingerRecipeStub) or (g is not None and isinstance(obj, g))

    def is_design(self, obj: object) -> bool:
        return self.is_recipe(obj) or _is_baseline(obj)

    def params(self, design: Any) -> dict[str, Any]:
        """The 9 scored parameters of a recipe in canonical form (``pad``, ``tip`` and the 7 numbers).

        A recipe the space built is canonical already. Track G's recipes may leave inactive
        parameters unset (``None``) or unquantised (for example the entrants read from
        ``configs/entrants_v2.yaml``); they are canonicalised here, so ids, the metric and the
        prompts always see the same values.
        """
        if _is_baseline(design):
            point = _baseline_point(design)
            if point is None:
                raise ValueError(f"baseline {design.name!r} has no recipe parameters")
            design = point
        raw = {name: getattr(design, name) for name in SCORED_PARAMS}
        canon, _, _ = canonical_params({k: v for k, v in raw.items() if v is not None})
        return canon if canon is not None else raw

    def point(self, design: Any) -> Any | None:
        """The recipe that places ``design`` in the metric (itself, a baseline's point, or None)."""
        if self.is_recipe(design):
            return design
        return _baseline_point(design)

    def design_id(self, design: Any) -> str:
        """Track G's ``design_id`` for G's recipes and baselines, so search records and built pairs agree."""
        g_id = track_g().design_id
        if g_id is not None and _is_g_object(design):
            return str(g_id(design))
        if _is_baseline(design):
            ident = design.design_id
            return str(ident() if callable(ident) else ident)
        payload = {"generator": _generator_version(), "recipe": self.params(design)}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def dump(self, design: Any) -> dict[str, Any]:
        if _is_baseline(design):
            point = _baseline_point(design)
            g_baseline = not isinstance(design, BaselineEntrant)
            params = (
                {k: getattr(design, k) for k in _G_BASELINE_FIELDS if hasattr(design, k)}
                if g_baseline
                else dict(getattr(design, "params", {}) or {})
            )
            data = {
                "baseline": str(getattr(design, "kind", design.name)),
                "name": str(design.name),
                "params": params,
                "point": self.dump(point) if point is not None else None,
            }
            if g_baseline:
                data["source"] = ENTRANTS_SOURCE
            if getattr(design, "rationale", None):
                data["rationale"] = design.rationale
            return data
        return self.params(design)

    def load(self, data: Mapping[str, Any]) -> Any:
        if "baseline" in data and data.get("source") == ENTRANTS_SOURCE:
            cls = track_g().baseline_cls
            if cls is None:
                raise ValueError(f"entrant {data.get('name')!r} needs track G's Baseline to load")
            point = data.get("point")
            recipe = None
            if point:
                recipe = self.load(point)
                if hasattr(recipe, "model_copy"):
                    recipe = recipe.model_copy(update={"name": str(data["name"])})
            return cls(
                name=str(data["name"]),
                kind=str(data["baseline"]),
                recipe=recipe,
                rationale=str(data.get("rationale") or ""),
                **dict(data.get("params") or {}),
            )
        if "baseline" in data:
            point = data.get("point")
            return BaselineEntrant(
                name=str(data["name"]),
                kind=str(data["baseline"]),
                point=self.load(point) if point else None,
                params=dict(data.get("params") or {}),
                rationale=data.get("rationale"),
            )
        check = self.validate(dict(data), None)
        if not check.ok:
            raise ValueError(f"not a valid {V2} recipe: {check.summary()}")
        return check.design

    def label(self, design: Any) -> str | None:
        return str(design.name) if _is_baseline(design) else None

    def is_baseline(self, design: Any) -> bool:
        return _is_baseline(design)

    def buildable(self, design: Any) -> Any:
        """What the generator builds: the recipe itself, a named recipe entrant's recipe, or the baseline."""
        if _is_named_recipe(design):
            return _baseline_point(design)
        return design

    def rationale(self, design: Any) -> str | None:
        return getattr(design, "rationale", None) if _is_baseline(design) else None

    def recipe(self, **params: Any) -> Any:
        """A canonical recipe from keyword parameters (``ValueError`` if it is invalid)."""
        check = self.validate(params, None)
        if not check.ok:
            raise ValueError(check.summary())
        return check.design

    # ------------------------------------------------------------------ proposer interface

    def tool_schema(self) -> dict[str, Any]:
        """A slot's design properties: ``pad``, ``tip``, then ``params`` with only the 3 always-active numbers required."""
        numbers_schema: dict[str, Any] = {}
        for name in NUMERIC_PARAMS:
            lo, hi = NUMERIC_BOUNDS[name]
            numbers_schema[name] = {"type": "number", "minimum": lo, "maximum": hi}
        always = [n for n in NUMERIC_PARAMS if n not in INACTIVE]
        return {
            "pad": {"type": "string", "enum": list(PAD_OPTIONS)},
            "tip": {"type": "string", "enum": list(TIP_OPTIONS)},
            "params": {
                "type": "object",
                "description": (
                    "The numbers. Give every parameter that is active for this pad and tip; inactive ones are "
                    "ignored (see the parameter table)."
                ),
                "properties": numbers_schema,
                "required": always,
                "additionalProperties": False,
            },
        }

    def slot_mapping(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """A slot's ``pad``, ``tip`` and ``params`` flattened into one recipe mapping."""
        out: dict[str, Any] = {}
        for key in CELL_PARAMS:
            if key in raw:
                out[key] = raw[key]
        params = raw.get("params")
        if isinstance(params, Mapping):
            for key, value in params.items():
                out[str(key)] = value
        elif "params" in raw:
            out["params"] = params  # reported as an unknown / malformed field
        for key in NUMERIC_PARAMS:  # tolerate numbers written next to pad and tip
            if key in raw and key not in out:
                out[key] = raw[key]
        return out

    def raw_params(self, raw: Any) -> dict[str, Any] | None:
        """A slot's recipe exactly as the model sent it, JSON-safe (``None`` if not an object)."""
        if not isinstance(raw, Mapping):
            return None
        from rtt.search.nemotron import _json_safe

        return _json_safe(self.slot_mapping(raw))

    def describe(self) -> str:
        """Markdown table of the 9 scored parameters (ranges, 10 % steps, inactive rules, meaning)."""
        rows = [
            "| parameter | range / options | 10% of range | inactive (fixed value) | meaning |",
            "|---|---|---|---|---|",
        ]
        for name in SCORED_PARAMS:
            if name == "pad":
                options, step, inactive = " / ".join(PAD_OPTIONS), "-", "-"
            elif name == "tip":
                options, step, inactive = " / ".join(TIP_OPTIONS), "-", "-"
            else:
                lo, hi = NUMERIC_BOUNDS[name]
                options, step = f"{_num(lo)} to {_num(hi)}", _num(0.1 * (hi - lo))
                rule = INACTIVE.get(name)
                inactive = f"{', '.join(rule[1])} ({_num(rule[2])})" if rule else "-"
            rows.append(f"| {name} | {options} | {step} | {inactive} | {_MEANING[name]} |")
        return "\n".join(rows)

    # ------------------------------------------------------------------ sampling and validity

    def from_unit(self, unit: Sequence[float], pad: str, tip: str) -> dict[str, Any]:
        """A raw recipe from a point of the 7-d unit cube plus the two categoricals."""
        raw: dict[str, Any] = {"pad": pad, "tip": tip}
        for value, name in zip(unit, NUMERIC_PARAMS, strict=True):
            lo, hi = NUMERIC_BOUNDS[name]
            raw[name] = float(lo + float(value) * (hi - lo))
        return raw

    def sample_sobol(self, seed: int, context: Any, **kwargs: Any) -> Any:
        from rtt.search.random_search import SpaceSampler

        return SpaceSampler(self, seed, context, **kwargs)

    def canonicalize(self, design: Any) -> tuple[Any, tuple[str, ...]]:
        if _is_baseline(design):
            return design, ()
        raw = self.params(design) if self.is_recipe(design) else dict(design)
        params, notes, errors = canonical_params(raw)
        if params is None:
            raise ValueError("; ".join(errors))
        return self.recipe_cls().model_validate(params), notes

    def validate(self, design: Any, context: Any, *, rationale: str | None = None) -> SpaceCheck:
        """Spec section 13, cheapest first: schema (bounds, strategy rules, quantisation), then geometry.

        The geometry stages (analytic per part, build, printability, seat) run when the
        context carries a validator, or track G's validators and the training parts.
        """
        if _is_baseline(design):
            point = _baseline_point(design)
            if _is_named_recipe(design):
                inner = self.validate(point, context)
                return SpaceCheck(
                    inner.ok,
                    design if inner.ok else None,
                    inner.errors,
                    inner.notes,
                    inner.stage,
                    inner.metrics,
                )
            return self._geometry_check(design, context, ())
        raw = self.params(design) if self.is_recipe(design) else design
        if not isinstance(raw, Mapping):
            return SpaceCheck(
                False, None, ("a design must be an object with pad, tip and params",), stage="schema"
            )
        params, notes, errors = canonical_params(raw)
        if params is None:
            return SpaceCheck(False, None, errors, notes, stage="schema")
        try:
            recipe = self.recipe_cls().model_validate(params)
        except Exception as exc:  # noqa: BLE001 - G's model may add rules the stub lacks
            return SpaceCheck(False, None, (f"recipe rejected: {exc}",), notes, stage="schema")
        return self._geometry_check(recipe, context, notes)

    def _geometry_check(self, design: Any, context: Any, notes: tuple[str, ...]) -> SpaceCheck:
        if isinstance(context, V2Context) and context.validator is not None:
            result = context.validator(design, context)
            if result is not None and not result.ok:
                return SpaceCheck(
                    False, None, result.errors, notes, stage=result.stage, metrics=result.metrics
                )
            metrics = dict(result.metrics) if result is not None else {}
            return SpaceCheck(True, design, (), notes, metrics=metrics)
        validate = track_g().validate
        target = self.buildable(design)
        if (
            validate is None
            or not isinstance(context, V2Context)
            or not context.parts
            or _is_stock(design)  # the fixed stock blade asset is not a recipe
            or not _is_g_object(target)  # a stub entrant that track G cannot build
        ):
            return SpaceCheck(True, design, (), notes)
        if context.task is None:
            raise ValueError("the v2 geometry validators need the task (V2Context.task)")
        did = self.design_id(design)
        try:
            report = validate(target, context.parts, context.task)
        except Exception as exc:  # noqa: BLE001 - a generator bug must not end a run; it is logged as a build failure
            message = f"generator_error: the generator failed on this recipe ({type(exc).__name__}: {exc})"
            return SpaceCheck(
                False, None, (message[:400],), notes, stage="build", metrics={"generator_error": 1.0}
            )
        if getattr(report, "geometry_id", None):
            context.geometry[did] = str(report.geometry_id)
        for pair in getattr(report, "pairs", ()) or ():
            context.remember_pair(did, str(pair.part_id), pair)
        metrics = {"build_s": float(getattr(report, "build_s", 0.0) or 0.0)}
        if not report.valid:
            code = str(report.code or "rejected")
            return SpaceCheck(
                False,
                None,
                (str(report.message or code),),
                notes,
                stage=G_CODE_STAGES.get(code, "geometry"),
                metrics=metrics,
            )
        return SpaceCheck(True, design, (), notes, metrics=metrics)

    def build(self, design: Any, part: Any, context: Any = None) -> Any:
        """Track G's :func:`~rtt.design.freeform.build_pair` for one part (cached on a :class:`V2Context`)."""
        build = track_g().build_pair
        if build is None:
            raise NotImplementedError("the v2 generator (track G) is not available yet")
        task = getattr(context, "task", None)
        if task is None:
            raise ValueError("building a v2 jaw pair needs the task (V2Context.task)")
        key = (self.design_id(design), str(part.part_id))
        cached = context.pairs.get(key) if isinstance(context, V2Context) else None
        if cached is not None:
            context.pairs.move_to_end(key)
            return cached
        pair = build(self.buildable(design), part, task)
        if isinstance(context, V2Context):
            context.remember_pair(key[0], key[1], pair)
        return pair

    def geometry_id(self, design: Any, context: Any) -> str | None:
        """SHA-256 over the per-part geometry hashes on the training parts (``None`` if unknown).

        Track G's ``hashing.geometry_id`` when the generator built the pairs; with the
        context's ``geometry_hash_fn`` (offline fakes), a hash of the hashes in part order.
        """
        if not isinstance(context, V2Context) or not context.part_ids:
            return None
        did = self.design_id(design)
        if did in context.geometry:
            return context.geometry[did]
        value: str | None = None
        if context.geometry_hash_fn is not None:
            hashes = [context.geometry_hash_fn(design, pid) for pid in context.part_ids]
            value = hashlib.sha256(canonical_json({"geometry": hashes}).encode("utf-8")).hexdigest()
        elif (
            track_g().build_pair is not None
            and context.parts
            and context.task is not None
            and (_is_stock(design) or _is_g_object(self.buildable(design)))
        ):
            by_part = {
                str(p.part_id): str(self.build(design, p, context).geometry_hash) for p in context.parts
            }
            g_geometry_id = track_g().geometry_id
            value = (
                g_geometry_id(by_part)
                if g_geometry_id is not None
                else hashlib.sha256(canonical_json({"geometry": by_part}).encode("utf-8")).hexdigest()
            )
        context.geometry[did] = value
        return value

    # ------------------------------------------------------------------ metric

    def unit_vector(self, design: Any) -> tuple[float, ...] | None:
        """The 7 numbers of ``design``'s point on [0, 1] (``None`` for a baseline without a point)."""
        point = self.point(design)
        if point is None:
            return None
        values = self.params(point)
        out = []
        for name in NUMERIC_PARAMS:
            lo, hi = NUMERIC_BOUNDS[name]
            out.append((float(values[name]) - lo) / (hi - lo))
        return tuple(out)

    def points(self, design: Any) -> dict[str, int] | None:
        unit = self.unit_vector(design)
        if unit is None:
            return None
        return {name: round(100.0 * u) for name, u in zip(NUMERIC_PARAMS, unit, strict=True)}

    def distance(self, a: Any, b: Any) -> float:
        """Mean of the 9 normalised terms; ``inf`` when either side has no point (the stock blade)."""
        pa, pb = self.point(a), self.point(b)
        if pa is None or pb is None:
            return math.inf
        va, vb = self.params(pa), self.params(pb)
        total = 0.0
        for name in NUMERIC_PARAMS:
            lo, hi = NUMERIC_BOUNDS[name]
            total += min(1.0, abs(float(va[name]) - float(vb[name])) / (hi - lo))
        total += float(va["pad"] != vb["pad"]) + float(va["tip"] != vb["tip"])
        return total / self.n_terms

    def cell(self, design: Any) -> tuple[str, str] | None:
        point = self.point(design)
        return None if point is None else (str(point.pad), str(point.tip))

    def cells(self) -> list[tuple[str, str]]:
        return [(p, t) for p in PAD_OPTIONS for t in TIP_OPTIONS]

    # ------------------------------------------------------------------ prompt helpers

    def points_row(self, design: Any) -> str:
        """The 7 numbers in points, e.g. ``grip 50 pad_h 55 ... tip_size 0`` (or the baseline's name)."""
        pts = self.points(design)
        if pts is None:
            return f"baseline {self.label(design)} (not a recipe)"
        return " ".join(f"{POINT_COLUMNS[name]} {pts[name]}" for name in NUMERIC_PARAMS)

    def point_differences(self, a: Any, b: Any) -> str:
        """Per-parameter differences in points, largest first (a changed pad or tip counts 100)."""
        pa, pb = self.point(a), self.point(b)
        if pa is None or pb is None:
            return "not comparable (a baseline without a recipe)"
        terms: list[tuple[int, str]] = []
        for name in CELL_PARAMS:
            if getattr(pa, name) != getattr(pb, name):
                terms.append((100, f"{name} 100"))
        ua, ub = self.unit_vector(pa), self.unit_vector(pb)
        assert ua is not None and ub is not None
        for name, va, vb in zip(NUMERIC_PARAMS, ua, ub, strict=True):
            diff = round(abs(100.0 * va - 100.0 * vb))
            if diff:
                terms.append((diff, f"{POINT_COLUMNS[name]} {diff}"))
        terms.sort(key=lambda t: -t[0])
        return ", ".join(text for _, text in terms) or "nothing"

    def params_json(self, design: Any) -> str:
        """Compact JSON of a recipe with only its active numbers (prompt form), or a baseline's name."""
        import json

        if _is_baseline(design):
            point = _baseline_point(design)
            if _is_named_recipe(design):
                return self.params_json(point)  # a named recipe entrant is that recipe
            data: dict[str, Any] = {"baseline": str(design.name)}
            if point is not None:
                data["as_recipe"] = json.loads(self.params_json(point))
            return json.dumps(data, separators=(", ", ": "))
        params = self.params(design)
        out: dict[str, Any] = {"pad": params["pad"], "tip": params["tip"]}
        for name in NUMERIC_PARAMS:
            rule = INACTIVE.get(name)
            if rule is not None and params[rule[0]] in rule[1]:
                continue
            value = round(float(params[name]), 2) + 0.0
            out[name] = int(value) if value.is_integer() else value
        return json.dumps(out, separators=(", ", ": "))


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
