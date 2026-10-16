"""v2 test suites over :class:`~rtt.parts.types.PartSpec` parts.

A :class:`PartCondition` is one part at one placement with the jitter seeds of its episodes;
a :class:`PartSuite` is an ordered set of them. Every design sees the same conditions and
seeds (common random numbers). The v1 machinery is reused where it fits: placements are
:class:`rtt.sim.objects.Placement` and seeds come from :func:`rtt.sim.conditions.derive_seed`.
:meth:`PartSuite.to_suite` hands a suite to the v1 evaluation path (each part as a
``MeshObject``), which lets v1 fingers and the stock entrant run on v2 parts unchanged.

**Training suite** ``train_v2`` (:func:`train_suite`, ``configs/suites/train_v2.yaml``): the six
library parts x four placements (c0 centred; c1 +2 mm across the jaw; c2 -3 mm across; c3 +1 mm
across with +6 deg yaw) x 2 jitter seeds = 48 episodes per design. "Across the jaw" is the
object-frame x axis (the finger width); y is the closing axis. Seed jitter (+-0.5 mm, +-1 deg,
+-10 % friction) is a task setting (``task_v2``), not part of the suite.

Held-out conditions may carry a height error ``err_z`` (m): the part sits that much higher
(positive) or lower than the controller assumes. Training conditions have none.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from rtt.parts.library import REPO_ROOT, load_parts_library, part_digest, training_parts
from rtt.parts.types import PartSpec
from rtt.sim import objects as objs
from rtt.sim.conditions import Condition, Suite, _placement_from_mapping, derive_seed

DEFAULT_TRAIN_V2 = REPO_ROOT / "configs" / "suites" / "train_v2.yaml"


@dataclass(frozen=True, eq=False)
class PartCondition:
    """One part at one placement, plus the jitter seeds of its episodes.

    ``size_label`` says how the part was made (``"train"``, or a held-out mode such as
    ``"near"``, ``"down"``, ``"up"``, ``"unseen"``, ``"scan"``); ``params_mm`` are its family
    dimensions when it is procedural.
    """

    condition_id: str
    part: PartSpec
    placement: objs.Placement
    seeds: tuple[int, ...]
    placement_label: str = ""
    size_label: str = ""
    err_z: float = 0.0
    params_mm: Mapping[str, float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError(f"{self.condition_id}: needs at least one jitter seed")
        if not math.isfinite(self.err_z) or abs(self.err_z) > 0.01:
            raise ValueError(f"{self.condition_id}: err_z must be finite and within 10 mm, got {self.err_z}")
        if self.params_mm is not None:
            object.__setattr__(self, "params_mm", MappingProxyType(dict(self.params_mm)))

    @property
    def family(self) -> str:
        return self.part.family

    def describe(self) -> dict[str, Any]:
        """JSON-friendly description (metres, degrees), including the part's digest."""
        p = self.placement
        out: dict[str, Any] = {
            "condition_id": self.condition_id,
            "family": self.part.family,
            "part": part_digest(self.part),
            "size_label": self.size_label,
            "placement_label": self.placement_label,
            "placement": {
                "x": round(p.x, 9),
                "y": round(p.y, 9),
                "yaw_deg": round(math.degrees(p.yaw), 9),
                "err_x": round(p.err_x, 9),
                "err_y": round(p.err_y, 9),
                "err_yaw_deg": round(math.degrees(p.err_yaw), 9),
                "err_z": round(self.err_z, 9),
            },
            "seeds": list(self.seeds),
        }
        if self.params_mm is not None:
            out["params_mm"] = {k: self.params_mm[k] for k in sorted(self.params_mm)}
        return out

    def to_condition(self) -> Condition:
        """The v1 :class:`~rtt.sim.conditions.Condition` (the part as a ``MeshObject``).

        v1 has no height error, so a condition with ``err_z != 0`` is refused."""
        if self.err_z != 0.0:
            raise ValueError(f"{self.condition_id}: the v1 scene has no height error (err_z={self.err_z})")
        return Condition(
            condition_id=self.condition_id,
            family=self.part.family,
            obj=self.part.to_mesh_object(),
            placement=self.placement,
            seeds=self.seeds,
            size_label=self.size_label or self.part.name,
            placement_label=self.placement_label,
        )


@dataclass(frozen=True)
class PartSuite:
    """An ordered set of :class:`PartCondition`."""

    name: str
    conditions: tuple[PartCondition, ...]

    def __post_init__(self) -> None:
        ids = [c.condition_id for c in self.conditions]
        if not ids:
            raise ValueError(f"suite {self.name!r} has no conditions")
        if len(set(ids)) != len(ids):
            raise ValueError(f"suite {self.name!r} has duplicate condition ids")
        by_name: dict[str, str] = {}
        for c in self.conditions:
            sha = by_name.setdefault(c.part.name, c.part.geometry_sha256)
            if sha != c.part.geometry_sha256:
                raise ValueError(f"suite {self.name!r}: two different parts are named {c.part.name!r}")

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self) -> Iterator[PartCondition]:
        return iter(self.conditions)

    @property
    def n_episodes(self) -> int:
        return sum(len(c.seeds) for c in self.conditions)

    def parts(self) -> tuple[PartSpec, ...]:
        """The distinct parts, in order of first appearance."""
        seen: dict[str, PartSpec] = {}
        for c in self.conditions:
            seen.setdefault(c.part.name, c.part)
        return tuple(seen.values())

    @property
    def widest_grip_width(self) -> float:
        """Widest part across the jaw (its y extent, m)."""
        return max(p.size[1] for p in self.parts())

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "conditions": [c.describe() for c in self.conditions]}

    def digest(self) -> str:
        """SHA-256 (hex) of :meth:`describe`, for manifests and pre-registration."""
        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_suite(self) -> Suite:
        """The suite as a v1 :class:`~rtt.sim.conditions.Suite` (parts as ``MeshObject``)."""
        return Suite(self.name, tuple(c.to_condition() for c in self.conditions))


# --------------------------------------------------------------------------- train_v2


@dataclass(frozen=True)
class TrainSpec:
    """A parsed ``train_v2``-style suite file."""

    name: str
    base_seed: int
    jitter_seeds_per_condition: int
    parts: tuple[str, ...]
    placements: Mapping[str, objs.Placement]
    parts_file: Path
    path: Path | None = field(default=None, compare=False)


def train_spec_from_dict(
    raw: Mapping[str, Any], where: str = "suite", base_dir: Path | None = None
) -> TrainSpec:
    allowed = {"name", "base_seed", "jitter_seeds_per_condition", "parts_file", "parts", "placements"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    base = base_dir or DEFAULT_TRAIN_V2.parent
    parts_file = (base / str(raw.get("parts_file", "../parts_v2.yaml"))).resolve()
    library = load_parts_library(parts_file)
    names = tuple(str(n) for n in raw.get("parts") or library.names())
    missing = [n for n in names if n not in library.names()]
    if missing or len(set(names)) != len(names):
        raise ValueError(f"{where}: parts must be unique names from {parts_file.name}; unknown {missing}")
    placements = {
        str(label): _placement_from_mapping(p or {}, f"{where}: placement {label!r}")
        for label, p in (raw.get("placements") or {}).items()
    }
    if not placements:
        raise ValueError(f"{where}: needs at least one placement")
    n_seeds = int(raw.get("jitter_seeds_per_condition", 1))
    if n_seeds < 1:
        raise ValueError(f"{where}: jitter_seeds_per_condition must be >= 1")
    return TrainSpec(
        name=str(raw.get("name", "train")),
        base_seed=int(raw.get("base_seed", 0)),
        jitter_seeds_per_condition=n_seeds,
        parts=names,
        placements=MappingProxyType(placements),
        parts_file=parts_file,
    )


def load_train_spec(path: str | Path = DEFAULT_TRAIN_V2) -> TrainSpec:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        spec = train_spec_from_dict(yaml.safe_load(fh) or {}, where=str(p), base_dir=p.resolve().parent)
    return TrainSpec(
        spec.name,
        spec.base_seed,
        spec.jitter_seeds_per_condition,
        spec.parts,
        spec.placements,
        spec.parts_file,
        p,
    )


def train_suite(spec: TrainSpec | str | Path | None = None) -> PartSuite:
    """The training suite: every part x every placement, parts outermost, in file order."""
    if not isinstance(spec, TrainSpec):
        spec = load_train_spec(spec if spec is not None else DEFAULT_TRAIN_V2)
    library = load_parts_library(spec.parts_file)
    parts = {p.name: p for p in training_parts(spec.parts_file)}
    conditions = []
    for name in spec.parts:
        entry = library.entry(name)
        for label, placement in spec.placements.items():
            cid = f"{spec.name}/{name}/{label}"
            seeds = tuple(
                derive_seed(spec.base_seed, cid, rep) for rep in range(spec.jitter_seeds_per_condition)
            )
            conditions.append(
                PartCondition(
                    condition_id=cid,
                    part=parts[name],
                    placement=placement,
                    seeds=seeds,
                    placement_label=label,
                    size_label="train",
                    params_mm=entry.params,
                )
            )
    return PartSuite(spec.name, tuple(conditions))


def part_suite_from_conditions(name: str, conditions: Sequence[PartCondition]) -> PartSuite:
    return PartSuite(name, tuple(conditions))
