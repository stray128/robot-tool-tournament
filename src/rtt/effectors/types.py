"""Frozen contract of the multi-effector product layer (product mode, not pre-registered).

The pre-registered experiment (``prereg-v1``) compares parallel-jaw *fingers*. This layer
asks a wider product question for one part: which end-effector family should hold it, and how
robust is that grasp? Every family plugs into the same floating wrist, the same part, the same
placements and seeds, the same physics checks and the same lift-and-hold scorer
(``docs/EFFECTORS.md``). Nothing here feeds the pre-registered result.

This module is the data contract. It imports numpy only (MuJoCo, the parts library and the
simulator are referenced for type checking), so every builder track can use it cheaply.

* :class:`EffectorFamily`: the families, each implemented by a module named after it
  (:data:`FAMILY_MODULES`, imported on first use by :func:`rtt.effectors.registry.get_family`).
* :class:`EffectorSpec`: one concrete effector (family, model, primitive, params, optional
  :class:`MorphologyRecipe`), content-addressed by :attr:`EffectorSpec.effector_id`.
* :class:`EffectorBuild`: what a family added to the scene (bodies, actuators, open pose,
  reach, declared actuation limits, licence records, visuals), plus the mass properties the
  scene measures from the compiled model.
* :class:`GraspPrimitive` and :class:`GraspCriterion`: the scripted controller and the
  family's engagement test (seat, contacts, seal, wrap).
* :class:`EffectorEpisodeResult`: one scored episode, in one shape for every family.
* :class:`MorphologyRecipe` and :class:`ParamSpec`: the base schema for effectors designed by
  Nemotron (finger count, link lengths, tentacle segments, pads, cups).

Rules every family follows (enforced where code can):

* Brand names never appear in anything shown on screen: :attr:`EffectorSpec.display_name` is
  checked against :data:`BRAND_TOKENS`. Credits live in :class:`AssetSource` records, which
  only README, NOTICE, ``docs/ATTRIBUTION.md`` and the credits page print.
* Only permissive licences are vendored (:data:`ALLOWED_LICENSES`), each file pinned by its
  upstream commit and sha256.
* Everything carries :data:`PRODUCT_MODE_LABEL`.

Units are SI (metres, kilograms, seconds, newtons, radians) unless a name says ``_mm``/``_deg``.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # referenced in annotations only
    from pathlib import Path

    import mujoco

    from rtt.parts.types import PartSpec
    from rtt.sim.controller import Phase
    from rtt.sim.objects import Placement
    from rtt.sim.replay import Replay
    from rtt.sim.task import TaskV2

__all__ = [
    "ALLOWED_LICENSES",
    "BRAND_TOKENS",
    "CONTRACT_VERSION",
    "ENGAGEMENT_SUB_FLAGS",
    "FAMILY_FIDELITY",
    "FAMILY_MODULES",
    "MAX_EFFECTOR_MASS_KG",
    "MORPHOLOGY_SCHEMA",
    "NAME_PREFIX",
    "PRODUCT_MODE_LABEL",
    "AssetSource",
    "BuildContext",
    "ContractError",
    "EffectorBuild",
    "EffectorEpisodeResult",
    "EffectorFamily",
    "EffectorSpec",
    "Engagement",
    "EntrantGroup",
    "Fidelity",
    "FrozenDict",
    "GraspCriterion",
    "GraspPrimitive",
    "InvalidEffector",
    "MorphologyRecipe",
    "Origin",
    "ParamSpec",
    "PrimitivePlan",
    "RegistrationReference",
    "Requirement",
    "VisualAsset",
    "canonical_json",
    "check_display_name",
    "find_brand_tokens",
    "group_of",
    "params_json_schema",
]

CONTRACT_VERSION: Final = "rtt-effectors/1"
#: Identical to :data:`rtt.agent.recorder.PRODUCT_MODE_LABEL` (a test keeps them equal).
PRODUCT_MODE_LABEL: Final = "Product mode: not part of the pre-registered experiment"
#: Every MuJoCo element a family adds to the scene is named with this prefix.
NAME_PREFIX: Final = "eff_"
MORPHOLOGY_SCHEMA: Final = "rtt-morphology/1"
#: A heavier effector is almost certainly a unit error (grams for kilograms, a whole arm).
MAX_EFFECTOR_MASS_KG: Final = 5.0

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_MAX_DISPLAY = 60


class InvalidEffector(ValueError):
    """This effector cannot be built for this part (too small an opening, no flat face, ...).

    Raised by a family's ``build`` (or the reference family's jaw builder); the episode then
    scores ``invalid_build`` without simulating, as an unbuildable jaw design does.
    """


class ContractError(RuntimeError):
    """A family broke the contract (names, mounting, actuator order, controls): a bug, never a score."""


class EffectorFamily(enum.StrEnum):
    """End-effector families. Each value names the module that implements it."""

    PARALLEL_JAW = "parallel_jaw"  # the reference: the frozen v2 jaw path, unchanged
    CATALOG_HAND = "catalog_hand"  # open dexterous multi-finger hands (vendored MJCF)
    CATALOG_GRIPPER = "catalog_gripper"  # other open grippers: 2-finger industrial, 3-finger adaptive
    SOFT_ARM = "soft_arm"  # octopus-style tentacle: spring-jointed segments + cable (approximation)
    SUCTION = "suction"  # suction cup: compliant lip + seal-gated adhesion (approximation)
    DESIGNED = "designed"  # our own morphologies from MorphologyRecipe (Nemotron or random)


#: ``rtt.effectors.<family value>``: the module that registers a family when imported.
FAMILY_MODULES: Final[Mapping[EffectorFamily, str]] = MappingProxyType(
    {fam: f"rtt.effectors.{fam.value}" for fam in EffectorFamily}
)


class Fidelity(enum.StrEnum):
    """How much to trust a family's physics; printed next to every result."""

    CORE = "core"  # the pre-registered experiment's own scene, checks and scorer
    RIGID = "rigid"  # rigid multibody model with a scripted primitive
    APPROXIMATION = "approximation"  # a labelled approximation (soft arm, suction seal)


FAMILY_FIDELITY: Final[Mapping[EffectorFamily, Fidelity]] = MappingProxyType(
    {
        EffectorFamily.PARALLEL_JAW: Fidelity.CORE,
        EffectorFamily.CATALOG_HAND: Fidelity.RIGID,
        EffectorFamily.CATALOG_GRIPPER: Fidelity.RIGID,
        EffectorFamily.SOFT_ARM: Fidelity.APPROXIMATION,
        EffectorFamily.SUCTION: Fidelity.APPROXIMATION,
        EffectorFamily.DESIGNED: Fidelity.RIGID,
    }
)


class EntrantGroup(enum.StrEnum):
    """Budget groups of the per-part tournament (:func:`group_of`)."""

    REFERENCE = "reference"  # parallel jaw entrants
    CATALOG = "catalog"  # catalog hands and grippers
    SOFT_ARM = "soft_arm"  # tentacle presets
    SUCTION = "suction"  # cup presets
    DESIGNED = "designed"  # recipes proposed by Nemotron, whatever generator builds them
    RANDOM = "random"  # recipes drawn at random from the same space, the designed group's control


class Requirement(enum.StrEnum):
    """What the recommendation optimises: held (lift and hold) or registered (held and in pose)."""

    HELD = "held"
    REGISTERED = "registered"


Origin = Literal["reference", "catalog", "preset", "nemotron", "random", "manual"]
ORIGINS: Final[tuple[str, ...]] = ("reference", "catalog", "preset", "nemotron", "random", "manual")

# --------------------------------------------------------------------------- brands and licences

#: Brand and product tokens that never appear on screen (matched as whole words, any case).
#: Credits (README, NOTICE, docs/ATTRIBUTION.md, the credits page) are the only place for them.
#: Some makers' names are on the project's private leak list, which ``scripts/leak_check.sh``
#: blocks in every file of the repository (credits included); they are deliberately not listed.
BRAND_TOKENS: Final[tuple[str, ...]] = (
    "leap hand",
    "allegro",
    "wonik",
    "simlab",
    "shadow hand",
    "shadow robot",
    "dexee",
    "dex-ee",
    "2f-85",
    "2f85",
    "panda hand",
    "emika",
    "umi gripper",
    "sharpa",
    "tetheria",
    "chestnut robotics",
    "aero hand",
    "so-arm",
    "so-arm100",
    "so-arm102",
    "so-100",
    "so-101",
    "so101",
    "the robot studio",
    "trossen",
    "aloha",
    "viperx",
    "piper",
    "agilex",
    "i2rt",
    "barrett",
    "barretthand",
    "schunk",
    "psyonic",
    "ability hand",
    "kinova",
    "festo",
    "onrobot",
    "piab",
    "schmalz",
    "soft robotics inc",
    "inspire robots",
    "unitree",
    "menagerie",
)
_BRAND_RE = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(t) for t in sorted(BRAND_TOKENS, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)

#: SPDX licences that may be vendored. Anything non-commercial, share-alike or copyleft is not.
ALLOWED_LICENSES: Final[tuple[str, ...]] = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MIT",
    "CC-BY-4.0",
    "CC0-1.0",
)


def find_brand_tokens(text: str) -> list[str]:
    """Every brand token in ``text`` (lower case, in order of appearance)."""
    return [m.group(1).lower() for m in _BRAND_RE.finditer(text)]


def check_display_name(name: object) -> str:
    """``name`` if it can go on screen: a non-empty string of at most 60 characters, no brands."""
    if not isinstance(name, str) or not name.strip() or len(name) > _MAX_DISPLAY:
        raise ValueError(f"display_name must be a non-empty string of at most {_MAX_DISPLAY} characters")
    hits = find_brand_tokens(name)
    if hits:
        raise ValueError(
            f"display_name {name!r} names a brand ({', '.join(hits)}); use a generic name on screen"
        )
    return name


# --------------------------------------------------------------------------- canonical JSON


def _canon(value: Any, where: str) -> Any:
    """A JSON-ready copy of ``value``: finite numbers (integral values as ints, so ``2.0`` and
    ``2`` hash alike), sorted mappings, lists for sequences."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"{where} must be finite, got {value!r}")
        return int(f) if f.is_integer() and abs(f) < 2**53 else f  # one form per number: 2.0 -> 2
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        out = {}
        for k in sorted(value):
            if not isinstance(k, str):
                raise TypeError(f"{where} keys must be strings, got {k!r}")
            out[k] = _canon(value[k], f"{where}.{k}")
        return out
    if isinstance(value, Sequence | np.ndarray):
        return [_canon(v, f"{where}[{i}]") for i, v in enumerate(value)]
    raise ValueError(
        f"{where} must be JSON data (numbers, strings, lists, mappings), got {type(value).__name__}"
    )


class FrozenDict(dict):  # type: ignore[type-arg]
    """A read-only ``dict`` that pickles (``MappingProxyType`` does not, and jobs cross processes)."""

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("this mapping is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly  # type: ignore[assignment]
    __ior__ = _readonly  # type: ignore[assignment]

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (dict(self),))

    def __repr__(self) -> str:
        return f"FrozenDict({dict.__repr__(self)})"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON of JSON-ready data (the hashing form of every contract id)."""
    return json.dumps(_canon(_thaw(value), "value"), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _check_slug(what: str, value: object) -> str:
    if not isinstance(value, str) or not _SLUG.match(value):
        raise ValueError(f"{what} must be a lower-case slug ({_SLUG.pattern}), got {value!r}")
    return value


# --------------------------------------------------------------------------- morphology schema


ParamKind = Literal["float", "int", "choice", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    """One designable parameter of a generator (the rows of a Nemotron parameter table).

    ``float``/``int`` parameters have inclusive bounds ``low``/``high``; ``choice`` parameters
    an ordered tuple of string ``choices``; ``bool`` parameters neither. ``unit`` is shown to
    the proposer (``"mm"``, ``"deg"``, ``"count"``, ``""``) and ``description`` says what the
    parameter does to the physical effector, in one sentence.
    """

    name: str
    kind: ParamKind
    description: str
    unit: str = ""
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    default: float | int | str | bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ValueError(f"parameter name must be snake_case, got {self.name!r}")
        if self.kind not in ("float", "int", "choice", "bool"):
            raise ValueError(f"{self.name}: kind must be float, int, choice or bool, got {self.kind!r}")
        if not self.description:
            raise ValueError(f"{self.name}: needs a description")
        if self.kind in ("float", "int"):
            if (
                self.low is None
                or self.high is None
                or not (math.isfinite(self.low) and math.isfinite(self.high))
            ):
                raise ValueError(f"{self.name}: numeric parameters need finite low and high bounds")
            if self.low > self.high:
                raise ValueError(f"{self.name}: low {self.low} > high {self.high}")
            if self.kind == "int" and (self.low != int(self.low) or self.high != int(self.high)):
                raise ValueError(f"{self.name}: int bounds must be whole numbers")
        elif self.low is not None or self.high is not None:
            raise ValueError(f"{self.name}: only numeric parameters have bounds")
        if self.kind == "choice":
            if not self.choices or len(set(self.choices)) != len(self.choices):
                raise ValueError(f"{self.name}: choice parameters need distinct choices")
            object.__setattr__(self, "choices", tuple(str(c) for c in self.choices))
        elif self.choices:
            raise ValueError(f"{self.name}: only choice parameters have choices")
        if self.default is not None and self.problems(self.default):
            raise ValueError(
                f"{self.name}: default {self.default!r} is invalid: {self.problems(self.default)}"
            )

    def problems(self, value: object) -> list[str]:
        """Why ``value`` is not a valid setting of this parameter (empty when it is)."""
        if self.kind == "bool":
            return [] if isinstance(value, bool) else [f"{self.name} must be true or false"]
        if self.kind == "choice":
            return [] if value in self.choices else [f"{self.name} must be one of {list(self.choices)}"]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            return [f"{self.name} must be a finite number"]
        if self.kind == "int" and float(value) != int(value):
            return [f"{self.name} must be a whole number"]
        assert self.low is not None and self.high is not None
        if not self.low <= float(value) <= self.high:
            return [f"{self.name} must be in [{self.low:g}, {self.high:g}] {self.unit}".rstrip()]
        return []

    def json_schema(self) -> dict[str, Any]:
        """The JSON-schema property for a forced tool call (bounds and choices enumerated)."""
        desc = f"{self.description} ({self.unit})" if self.unit else self.description
        if self.kind == "bool":
            return {"type": "boolean", "description": desc}
        if self.kind == "choice":
            return {"type": "string", "enum": list(self.choices), "description": desc}
        return {
            "type": "integer" if self.kind == "int" else "number",
            "minimum": int(self.low) if self.kind == "int" else self.low,  # type: ignore[arg-type]
            "maximum": int(self.high) if self.kind == "int" else self.high,  # type: ignore[arg-type]
            "description": desc,
        }


def params_json_schema(space: Sequence[ParamSpec]) -> dict[str, Any]:
    """A strict JSON-schema object for one parameter set of ``space`` (every field required)."""
    names = [p.name for p in space]
    if len(set(names)) != len(names):
        raise ValueError("parameter names in a space must be unique")
    return {
        "type": "object",
        "properties": {p.name: p.json_schema() for p in space},
        "required": names,
        "additionalProperties": False,
    }


@dataclass(frozen=True, eq=False)
class MorphologyRecipe:
    """A designed effector: which generator builds it and with which parameters.

    * ``family``: the generator family. :attr:`EffectorFamily.DESIGNED` for our own fingered
      morphologies; a parametric family (for example :attr:`EffectorFamily.SOFT_ARM`) for a
      recipe in that family's :meth:`~rtt.effectors.registry.EffectorFamilyImpl.param_space`.
    * ``kind``: the generator within the family (``"fingered_hand"``, ``"tentacle"``,
      ``"suction_cup"``, ...), a slug.
    * ``params``: one value per :class:`ParamSpec` of that generator's space.
    * ``hypothesis``: the proposer's one-paragraph reason (shown on cards, never hashed).
    * ``origin``: ``"nemotron"``, ``"random"`` (the equal-budget control) or ``"manual"``.
    * ``parents``: recipe ids this one was derived from (empty for a first proposal).

    :attr:`recipe_id` hashes the schema, family, kind and params only, so the same recipe
    proposed twice (or by both arms) is recognised as one design.
    """

    family: EffectorFamily
    kind: str
    params: Mapping[str, float | int | str | bool]
    hypothesis: str = ""
    origin: Literal["nemotron", "random", "manual"] = "manual"
    parents: tuple[str, ...] = ()
    schema: str = MORPHOLOGY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", EffectorFamily(self.family))
        _check_slug("recipe kind", self.kind)
        if self.schema != MORPHOLOGY_SCHEMA:
            raise ValueError(f"recipe schema must be {MORPHOLOGY_SCHEMA!r}, got {self.schema!r}")
        if self.origin not in ("nemotron", "random", "manual"):
            raise ValueError(f"recipe origin must be nemotron, random or manual, got {self.origin!r}")
        if not isinstance(self.hypothesis, str):
            raise TypeError("hypothesis must be a string")
        params = _canon(self.params, "params")
        if not isinstance(params, dict) or any(
            isinstance(v, dict | list) or v is None for v in params.values()
        ):
            raise ValueError("recipe params must map names to numbers, strings or booleans")
        object.__setattr__(self, "params", FrozenDict(params))
        for p in self.parents:
            if not _SHA256.match(str(p)):
                raise ValueError(f"parent ids must be recipe ids (sha256), got {p!r}")
        object.__setattr__(self, "parents", tuple(self.parents))

    def canonical_dict(self) -> dict[str, Any]:
        """The hashed part of the recipe."""
        return {
            "schema": self.schema,
            "family": self.family.value,
            "kind": self.kind,
            "params": dict(self.params),
        }

    @property
    def recipe_id(self) -> str:
        """sha256 (hex) of :meth:`canonical_dict`."""
        return _sha256(canonical_json(self.canonical_dict()))

    def problems(self, space: Sequence[ParamSpec]) -> list[str]:
        """Every way the params miss ``space``: unknown, missing or out-of-range values."""
        by_name = {p.name: p for p in space}
        out = [f"unknown parameter {k!r}" for k in self.params if k not in by_name]
        out += [f"missing parameter {k!r}" for k in by_name if k not in self.params]
        for k, v in self.params.items():
            if k in by_name:
                out += by_name[k].problems(v)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.canonical_dict(),
            "recipe_id": self.recipe_id,
            "hypothesis": self.hypothesis,
            "origin": self.origin,
            "parents": list(self.parents),
        }


# --------------------------------------------------------------------------- the effector


def group_of(origin: str, family: EffectorFamily) -> EntrantGroup:
    """The tournament budget group of an entrant (origin first, then family)."""
    if origin == "nemotron":
        return EntrantGroup.DESIGNED
    if origin == "random":
        return EntrantGroup.RANDOM
    return {
        EffectorFamily.PARALLEL_JAW: EntrantGroup.REFERENCE,
        EffectorFamily.CATALOG_HAND: EntrantGroup.CATALOG,
        EffectorFamily.CATALOG_GRIPPER: EntrantGroup.CATALOG,
        EffectorFamily.SOFT_ARM: EntrantGroup.SOFT_ARM,
        EffectorFamily.SUCTION: EntrantGroup.SUCTION,
        EffectorFamily.DESIGNED: EntrantGroup.DESIGNED,
    }[family]


@dataclass(frozen=True, eq=False)
class EffectorSpec:
    """One concrete end-effector, as the tournament, the runner and the web see it.

    * ``family``: the :class:`EffectorFamily` whose module builds and runs it.
    * ``model``: the family-local model key (a slug), e.g. ``"hand-4f-16dof"``, ``"cup-20mm"``,
      ``"stock"``. Catalog models name the vendored directory; never a brand on screen.
    * ``display_name``: what the screen says, brand-free (:func:`check_display_name`).
    * ``primitive``: the grasp primitive (a slug), e.g. ``"power"``, ``"pinch"``, ``"seal"``,
      ``"wrap"``, ``"jaw"``.
    * ``params``: JSON-ready family parameters (stored canonically, read-only).
    * ``recipe``: the :class:`MorphologyRecipe` of a designed effector (required when
      ``origin`` is ``"nemotron"`` or ``"random"``).
    * ``origin``: where it came from (:data:`ORIGINS`); with ``family`` it sets the
      tournament group (:func:`group_of`).

    :attr:`effector_id` is the sha256 of everything except ``display_name``.
    """

    family: EffectorFamily
    model: str
    display_name: str
    primitive: str
    params: Mapping[str, Any] = field(default_factory=dict)
    recipe: MorphologyRecipe | None = None
    origin: Origin = "preset"

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", EffectorFamily(self.family))
        _check_slug("model", self.model)
        _check_slug("primitive", self.primitive)
        check_display_name(self.display_name)
        if self.origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {self.origin!r}")
        params = _canon(self.params, "params")
        if not isinstance(params, dict):
            raise TypeError("params must be a mapping")
        object.__setattr__(self, "params", _freeze(params))
        if self.recipe is not None and not isinstance(self.recipe, MorphologyRecipe):
            raise TypeError("recipe must be a MorphologyRecipe")
        if self.origin in ("nemotron", "random") and self.recipe is None:
            raise ValueError(f"a {self.origin} effector needs its MorphologyRecipe")
        if self.recipe is not None and self.recipe.origin != "manual" and self.recipe.origin != self.origin:
            raise ValueError(f"origin {self.origin!r} does not match the recipe's {self.recipe.origin!r}")

    @property
    def fidelity(self) -> Fidelity:
        return FAMILY_FIDELITY[self.family]

    @property
    def group(self) -> EntrantGroup:
        return group_of(self.origin, self.family)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "family": self.family.value,
            "model": self.model,
            "primitive": self.primitive,
            "params": _thaw(self.params),
            "recipe": self.recipe.canonical_dict() if self.recipe is not None else None,
            "origin": self.origin,
        }

    @property
    def effector_id(self) -> str:
        """sha256 (hex) of :meth:`canonical_dict` (the display name is not hashed)."""
        return _sha256(canonical_json(self.canonical_dict()))

    @property
    def short_id(self) -> str:
        return self.effector_id[:12]

    def to_dict(self) -> dict[str, Any]:
        out = self.canonical_dict()
        out["recipe"] = self.recipe.to_dict() if self.recipe is not None else None
        out.update(
            effector_id=self.effector_id,
            display_name=self.display_name,
            fidelity=self.fidelity.value,
            group=self.group.value,
            label=PRODUCT_MODE_LABEL,
        )
        return out


# --------------------------------------------------------------------------- builds


@dataclass(frozen=True)
class AssetSource:
    """Licence record of one vendored third-party file (one ``SOURCE.json`` entry).

    * ``path``: repository-relative path of the vendored file, as shipped.
    * ``sha256``: of the vendored file as shipped (after any changes).
    * ``upstream_url``/``upstream_commit``/``upstream_path``: the pinned origin.
    * ``upstream_sha256``: of the upstream file at that commit (equal to ``sha256`` if unchanged).
    * ``license``: SPDX id, one of :data:`ALLOWED_LICENSES`; ``copyright``: the holder line.
    * ``attribution``: the credit line (may name the brand: credits pages only).
    * ``changes``: what we changed (empty only if the file is byte-identical upstream).
    * ``status``: ``"green"`` (permissive chain end to end) or ``"amber"`` (permissive as
      published, with a caveat recorded in ``note``). A red licence is never vendored.
    """

    path: str
    sha256: str
    upstream_url: str
    upstream_commit: str
    upstream_path: str
    upstream_sha256: str
    license: str
    copyright: str
    attribution: str
    changes: tuple[str, ...] = ()
    status: Literal["green", "amber"] = "green"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError(f"path must be repository-relative, got {self.path!r}")
        for name in ("sha256", "upstream_sha256"):
            if not _SHA256.match(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase hex sha256")
        if not self.upstream_url.startswith("https://"):
            raise ValueError("upstream_url must be https")
        if not _COMMIT.match(self.upstream_commit):
            raise ValueError("upstream_commit must be a hex commit id")
        if not _SPDX.match(self.license) or self.license not in ALLOWED_LICENSES:
            raise ValueError(f"licence {self.license!r} may not be vendored; allowed: {ALLOWED_LICENSES}")
        if not self.attribution or not self.copyright:
            raise ValueError("attribution and copyright are required")
        object.__setattr__(self, "changes", tuple(self.changes))
        if self.sha256 != self.upstream_sha256 and not self.changes:
            raise ValueError(f"{self.path}: differs from upstream, so its changes must be listed")
        if self.status not in ("green", "amber"):
            raise ValueError("status must be green or amber")
        if self.status == "amber" and not self.note:
            raise ValueError("an amber licence record needs its caveat in note")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "upstream_url": self.upstream_url,
            "upstream_commit": self.upstream_commit,
            "upstream_path": self.upstream_path,
            "upstream_sha256": self.upstream_sha256,
            "license": self.license,
            "copyright": self.copyright,
            "attribution": self.attribution,
            "changes": list(self.changes),
            "status": self.status,
            "note": self.note,
        }


@dataclass(frozen=True)
class VisualAsset:
    """A visual-only mesh drawn on an effector body by the renderer and the web (never collides).

    ``path`` is repository-relative (GLB preferred; STL/OBJ are converted by the exporter);
    ``pos``/``quat_wxyz``/``scale`` place it in the body frame; scale is positive (no mirrors).
    """

    body: str
    path: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.72, 1.0)
    source: AssetSource | None = None

    def __post_init__(self) -> None:
        if not self.body.startswith(NAME_PREFIX):
            raise ValueError(f"visual body {self.body!r} must be an effector body ({NAME_PREFIX}...)")
        if any(float(s) <= 0 for s in self.scale):
            raise ValueError("visual scale must be positive (mirrors are separate assets)")


@dataclass(frozen=True)
class BuildContext:
    """What a family's ``build`` gets: the scene spec, the wrist body to mount on, and inputs.

    The family adds its subtree under ``mount`` (directly or through ``mount.add_frame`` and
    ``attach_body``), with every name starting ``prefix``; the mount plane, where the
    effector's housing meets the wrist flange, is ``mount_z`` in the wrist frame (the same
    plane as the parallel jaw's finger roots). It must not add bodies to the world body, change
    ``spec.option`` or touch the wrist's joints and actuators.
    """

    spec: mujoco.MjSpec
    mount: mujoco.MjsBody
    mount_z: float
    effector: EffectorSpec
    part: PartSpec
    task: TaskV2
    repo_root: Path
    prefix: str = NAME_PREFIX


@dataclass(frozen=True, eq=False)
class EffectorBuild:
    """What a family added to the scene for one effector (and part).

    Filled by the family:

    * ``bodies``: the effector's bodies (every one prefixed :data:`NAME_PREFIX`), root first.
    * ``actuators``: the effector's actuators, in the order the model compiles them; this is
      the order of the vector a :class:`GraspPrimitive` returns.
    * ``ctrl_open``: one control per actuator: the open (pre-grasp) pose.
    * ``reach``: m from the mount plane down to the effector's lowest point in the open pose.
    * ``tcp``: the grasp centre in the wrist frame (m), for primitives and cards.
    * ``actuation``: the declared actuation limits, printed on every card (for example
      ``grip_force_n``, ``joint_torque_nm_max``, ``cable_tension_n``, ``vacuum_kpa``).
    * ``sources``: licence records of every vendored file used; ``visuals``: visual meshes;
      ``notes``: plain-language notes (approximations, repairs).
    * ``max_opening``: aperture in m where it means something (jaws, grippers), else ``None``.

    Measured by the scene from the compiled model (a family leaves them ``None``): ``mass``
    (kg, every effector body), ``com`` (wrist frame, at the model's reference pose) and
    ``inertia6`` (about ``com``, wrist axes, ``Ixx Iyy Izz Ixy Ixz Iyz``).
    """

    spec: EffectorSpec
    bodies: tuple[str, ...]
    actuators: tuple[str, ...]
    ctrl_open: tuple[float, ...]
    reach: float
    tcp: tuple[float, float, float]
    actuation: Mapping[str, float]
    sources: tuple[AssetSource, ...] = ()
    visuals: tuple[VisualAsset, ...] = ()
    notes: tuple[str, ...] = ()
    max_opening: float | None = None
    mass: float | None = None
    com: tuple[float, float, float] | None = None
    inertia6: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, EffectorSpec):
            raise TypeError("spec must be an EffectorSpec")
        for what in ("bodies", "actuators"):
            names = tuple(getattr(self, what))
            if len(set(names)) != len(names) or any(
                not isinstance(n, str) or not n.startswith(NAME_PREFIX) for n in names
            ):
                raise ValueError(f"{what} must be distinct names starting with {NAME_PREFIX!r}")
            object.__setattr__(self, what, names)
        if not self.bodies:
            raise ValueError("an effector has at least one body")
        ctrl = tuple(float(c) for c in self.ctrl_open)
        if len(ctrl) != len(self.actuators) or not all(math.isfinite(c) for c in ctrl):
            raise ValueError("ctrl_open needs one finite value per actuator")
        object.__setattr__(self, "ctrl_open", ctrl)
        if not (math.isfinite(self.reach) and 0.0 < self.reach < 0.5):
            raise ValueError(f"reach must be in (0, 0.5) m, got {self.reach!r}")
        tcp = tuple(float(v) for v in self.tcp)
        if len(tcp) != 3 or not all(math.isfinite(v) for v in tcp):
            raise ValueError("tcp must be 3 finite numbers")
        object.__setattr__(self, "tcp", tcp)
        act = _canon(self.actuation, "actuation")
        if not isinstance(act, dict) or not act or any(not isinstance(v, float | int) for v in act.values()):
            raise ValueError("actuation must declare at least one numeric limit")
        object.__setattr__(self, "actuation", FrozenDict(act))
        for name in ("sources", "visuals", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.mass is not None and not (0.0 < self.mass <= MAX_EFFECTOR_MASS_KG):
            raise ValueError(f"effector mass {self.mass} kg is outside (0, {MAX_EFFECTOR_MASS_KG}] kg")

    @property
    def n_actuators(self) -> int:
        return len(self.actuators)


# --------------------------------------------------------------------------- control and engagement


@dataclass(frozen=True)
class PrimitivePlan:
    """Where a primitive wants the wrist at the grasp.

    * ``tip_height``: m above the table of the effector's lowest point (in the open pose) at
      the grasp pose. The runner turns it into wrist heights with the task's own approach,
      descent and lift arithmetic (:func:`rtt.sim.controller._plan`), as for the jaws.
    * ``yaw_offset``: rad added to the placement yaw for the wrist (a hand turned 90 deg).
      Placement errors, shakes and pushes stay in the part's nominal frame for every family.
    """

    tip_height: float
    yaw_offset: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tip_height) and self.tip_height >= 0.0):
            raise ValueError(f"tip_height must be finite and >= 0, got {self.tip_height!r}")
        if not math.isfinite(self.yaw_offset):
            raise ValueError("yaw_offset must be finite")


@runtime_checkable
class GraspPrimitive(Protocol):
    """A scripted, open-loop-in-time controller for the effector's own actuators.

    The wrist follows the shared phases (SETTLE, DESCEND, CLOSE, LIFT, HOLD) with the task's
    timings; the primitive only drives the effector. ``closure`` is the shared close ramp,
    0 before CLOSE and rising linearly to 1 over ``controller.close_time``. A primitive may
    read ``data`` (contacts, joint positions) for closed-loop behaviour such as a vacuum
    switch, but must not write to it: it acts only through the returned controls.
    """

    name: str

    def plan(
        self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild
    ) -> PrimitivePlan: ...

    def command(
        self,
        t: float,
        phase: Phase,
        closure: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene: Any,
    ) -> np.ndarray:
        """Controls for every effector actuator (``build.actuators`` order) at time ``t``."""
        ...


#: Engagement sub-flags added to the registration sub-flags of :mod:`rtt.sim.register`.
ENGAGEMENT_SUB_FLAGS: Final[tuple[str, ...]] = (
    "jam_on_close",  # parallel jaw: stopped short of its seat
    "few_contacts",  # hands and grippers: fewer effector bodies on the part than required
    "no_opposition",  # hands and grippers: the contact forces do not oppose each other
    "no_seal",  # suction: the lip is not sealed on the face (gap, edge, hole or curvature)
    "no_wrap",  # soft arm: too few segments on the part or too small a wrap angle
)


@dataclass(frozen=True)
class Engagement:
    """A family's engagement verdict at the registration moment (replaces the jaw's seat)."""

    ok: bool
    sub_flags: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.sub_flags) - set(ENGAGEMENT_SUB_FLAGS)
        if unknown:
            raise ValueError(f"unknown engagement sub-flags {sorted(unknown)}; known: {ENGAGEMENT_SUB_FLAGS}")
        if self.ok and self.sub_flags:
            raise ValueError("an engaged verdict carries no failure sub-flags")
        object.__setattr__(self, "sub_flags", tuple(self.sub_flags))
        object.__setattr__(self, "metrics", FrozenDict({k: float(v) for k, v in self.metrics.items()}))


@runtime_checkable
class GraspCriterion(Protocol):
    """A family's grasp tests. Pass/fail is still the shared scorer's (``docs/EFFECTORS.md``).

    * ``grasped``: at the start of LIFT; it only chooses the failure label (``missed_grasp``
      or a later one), exactly as the jaw's squeeze test does, never success.
    * ``engaged``: at the registration moment; it replaces the jaw's seat test in the
      registered criterion (contacts, seal, wrap).
    """

    name: str

    def grasped(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> bool: ...

    def engaged(self, model: mujoco.MjModel, data: mujoco.MjData, scene: Any) -> Engagement: ...


@dataclass(frozen=True)
class RegistrationReference:
    """The part's pose in the wrist frame that "registered" is measured against.

    ``source = "analytic"``: the wrist-aimed pose (origin on the wrist axis, the part's yaw
    turned by ``-yaw_offset``, upright), which is the pre-registered rule for the jaws.
    ``source = "nominal_episode"``: the pose measured in the same effector's nominal
    certificate episode (no placement error, no jitter): a family that does not centre the
    part on the wrist axis is registered when it puts the part back in that same pose,
    whatever the placement error (repeatability).
    """

    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    source: Literal["analytic", "nominal_episode"]

    def __post_init__(self) -> None:
        pos = tuple(float(v) for v in self.pos)
        quat = np.asarray(self.quat_wxyz, dtype=np.float64)
        if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
            raise ValueError("pos must be 3 finite numbers")
        if quat.shape != (4,) or not np.all(np.isfinite(quat)) or not 0.5 < np.linalg.norm(quat) < 1.5:
            raise ValueError("quat_wxyz must be a (near) unit quaternion")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "quat_wxyz", tuple(float(v) for v in quat / np.linalg.norm(quat)))
        if self.source not in ("analytic", "nominal_episode"):
            raise ValueError("source must be analytic or nominal_episode")

    @staticmethod
    def analytic(yaw_offset: float = 0.0) -> RegistrationReference:
        half = -0.5 * yaw_offset
        return RegistrationReference((0.0, 0.0, 0.0), (math.cos(half), 0.0, 0.0, math.sin(half)), "analytic")


# --------------------------------------------------------------------------- results


def _finite_or_none(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


@dataclass(eq=False)
class EffectorEpisodeResult:
    """One scored episode of any family (one shape for records, lanes and the tournament).

    * ``success``: the **registered** criterion: ``held``, ``engaged`` and the part in the
      reference pose (centred within 0.75 mm, yaw within 2 deg, tilt within 3 deg, upright
      within 5 deg after the hold: the frozen ``task_v2`` registration tolerances).
    * ``held``: the shared lift-and-hold criterion (lift 100 mm, hold 2 s, no physics flag,
      no MuJoCo reset, held penetration <= 1 mm), identical code for every family.
    * ``engaged``: the family's :class:`GraspCriterion` verdict at the registration moment.
    * ``failure_mode``: one of :data:`rtt.sim.score.FAILURE_MODES_V2`, or ``None``.
    * ``registration_flags``: the frozen registration sub-flags plus :data:`ENGAGEMENT_SUB_FLAGS`.
    * ``flags``: physics-check flag names; ``bad_qacc``: MuJoCo resets in the scored part.
    * ``penetration_mm``: deepest effector-part penetration over the scored part.
    * ``reg_pos``/``reg_quat_wxyz``: the measured part pose in the wrist frame at the
      registration moment (``None`` if the episode ended before it); ``reference``: what it
      was compared with.
    * ``traj_sha``: sha256 of the state at every check tick (determinism).
    """

    spec: EffectorSpec
    part_id: str
    seed: int
    success: bool
    held: bool
    engaged: bool
    failure_mode: str | None
    registration_flags: tuple[str, ...]
    flags: tuple[str, ...]
    max_lift: float
    hold_time: float
    bad_qacc: int
    penetration_mm: float
    traj_sha: str
    n_steps: int
    wall_time_s: float
    metrics: dict[str, float]
    reference: Literal["analytic", "nominal_episode"]
    reg_pos: tuple[float, float, float] | None = None
    reg_quat_wxyz: tuple[float, float, float, float] | None = None
    condition_id: str = ""
    placement: Mapping[str, float] = field(default_factory=dict)
    perturbation: str | None = None
    effector_mass_kg: float = math.nan
    label: str = PRODUCT_MODE_LABEL
    replay: Replay | None = field(default=None, repr=False)
    qpos: np.ndarray | None = field(default=None, repr=False)
    overlays: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def fidelity(self) -> Fidelity:
        return self.spec.fidelity

    def reference_pose(self) -> RegistrationReference | None:
        """This episode's measured pose as a reference for the same effector's other episodes."""
        if self.reg_pos is None or self.reg_quat_wxyz is None:
            return None
        return RegistrationReference(self.reg_pos, self.reg_quat_wxyz, "nominal_episode")

    def to_record(self) -> dict[str, Any]:
        """Flat, JSON- and Parquet-ready record (NaN -> ``None``; no replay or arrays)."""
        s = self.spec
        out: dict[str, Any] = {
            "contract": CONTRACT_VERSION,
            "label": self.label,
            "effector_id": s.effector_id,
            "family": s.family.value,
            "model": s.model,
            "display_name": s.display_name,
            "primitive": s.primitive,
            "origin": s.origin,
            "group": s.group.value,
            "fidelity": s.fidelity.value,
            "recipe_id": s.recipe.recipe_id if s.recipe is not None else None,
            "part_id": self.part_id,
            "condition_id": self.condition_id,
            "seed": int(self.seed),
            "perturbation": self.perturbation,
            "success": bool(self.success),
            "held": bool(self.held),
            "engaged": bool(self.engaged),
            "failure_mode": self.failure_mode,
            "registration_flags": list(self.registration_flags),
            "flags": list(self.flags),
            "max_lift": float(self.max_lift),
            "hold_time": float(self.hold_time),
            "bad_qacc": int(self.bad_qacc),
            "penetration_mm": float(self.penetration_mm),
            "reference": self.reference,
            "traj_sha": self.traj_sha,
            "n_steps": int(self.n_steps),
            "wall_time_s": float(self.wall_time_s),
            "effector_mass_kg": float(self.effector_mass_kg),
        }
        for k, v in sorted(self.placement.items()):
            out[f"placement_{k}"] = float(v)
        for k, v in sorted(self.metrics.items()):
            out[f"m_{k}"] = float(v)
        return {k: _finite_or_none(v) for k, v in out.items()}
