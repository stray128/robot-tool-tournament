"""Freeze each run's top-1 design before any held-out evaluation.

The frozen record holds the design, its controller setting ``k`` (grasp height)
and its training score, the run identity, a SHA-256 over that content and a UTC
timestamp. The held-out job evaluates exactly what was frozen;
:func:`verify_frozen` recomputes the hash so any edit is detected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rtt.llm.cache import canonical_json
from rtt.search.base import EvaluatorInfo, RunHistory
from rtt.search.space import V1, get_space, space_of
from rtt.store.manifest import write_json_atomic

__all__ = ["FrozenDesign", "freeze_top1", "read_frozen", "verify_frozen", "write_frozen"]


@dataclass(frozen=True)
class FrozenDesign:
    """A run's selected design and how it will be evaluated on the held-out suite."""

    arm: str
    run_index: int
    run_seed: int
    label: str
    design_id: str
    design: Any
    """A v1 :class:`~rtt.design.schema.FingerDesign`, or a v2 recipe or baseline."""
    controller_setting: int
    grasp_height_frac: float | None
    training_success: float
    training_hold_fraction: float
    mass_g: float
    origin: str
    content_sha256: str
    frozen_at: str

    @property
    def design_space(self) -> str:
        return space_of(self.design).name

    def content(self) -> dict[str, Any]:
        """The hashed part of the record (everything except the hash and the timestamp)."""
        return _content(
            self.arm,
            self.run_index,
            self.run_seed,
            self.label,
            self.design_id,
            self.design,
            self.controller_setting,
            self.grasp_height_frac,
            self.training_success,
            self.training_hold_fraction,
            self.mass_g,
            self.origin,
        )

    def to_json(self) -> dict[str, Any]:
        return {**self.content(), "content_sha256": self.content_sha256, "frozen_at": self.frozen_at}


def _content_hash(content: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def _content(
    arm: str,
    run_index: int,
    run_seed: int,
    label: str,
    design_id: str,
    design: Any,
    controller_setting: int,
    grasp_height_frac: float | None,
    training_success: float,
    training_hold_fraction: float,
    mass_g: float,
    origin: str,
) -> dict[str, Any]:
    space = space_of(design)
    content = {
        "arm": arm,
        "run_index": run_index,
        "run_seed": run_seed,
        "label": label,
        "design_id": design_id,
        "design": space.dump(design),
        "controller_setting": controller_setting,
        "grasp_height_frac": grasp_height_frac,
        "training_success": training_success,
        "training_hold_fraction": training_hold_fraction,
        "mass_g": mass_g,
        "origin": origin,
    }
    if space.name != V1:
        content["design_space"] = space.name
    return content


def freeze_top1(
    arm: str, run_index: int, run_seed: int, history: RunHistory, info: EvaluatorInfo | None = None
) -> FrozenDesign:
    """Select the run's best design (success, then hold fraction, then lower mass, then earliest).

    In v2 the second key is lift-only success (:meth:`~rtt.search.base.DesignScore.rank_key`), the
    controller setting is always 0 and the grasp height is the recipe's own ``grip_height_frac``.
    """
    best = history.best()
    k = best.score.best_k
    if best.score.is_v2:
        height = getattr(get_space("freeform-v2").point(best.design), "grip_height_frac", None)
    else:
        height = (
            info.controller_settings[k] if info is not None and k < len(info.controller_settings) else None
        )
    content = _content(
        arm,
        run_index,
        run_seed,
        best.label,
        best.score.design_id,
        best.design,
        k,
        height,
        best.score.success,
        best.score.hold_fraction,
        best.score.mass_g,
        best.proposal.origin,
    )
    return FrozenDesign(
        arm=arm,
        run_index=run_index,
        run_seed=run_seed,
        label=best.label,
        design_id=best.score.design_id,
        design=best.design,
        controller_setting=k,
        grasp_height_frac=height,
        training_success=best.score.success,
        training_hold_fraction=best.score.hold_fraction,
        mass_g=best.score.mass_g,
        origin=best.proposal.origin,
        content_sha256=_content_hash(content),
        frozen_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    )


def write_frozen(frozen: FrozenDesign, out_dir: str | Path) -> Path:
    """Write ``<out_dir>/<arm>-r<NN>.json``; an existing frozen record is never overwritten."""
    path = Path(out_dir) / f"{frozen.arm}-r{frozen.run_index:02d}.json"
    if path.exists():
        raise FileExistsError(f"{path} is already frozen")
    return write_json_atomic(path, frozen.to_json())


def read_frozen(path: str | Path) -> FrozenDesign:
    """Load and verify a frozen record."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = FrozenDesign(
        arm=data["arm"],
        run_index=int(data["run_index"]),
        run_seed=int(data["run_seed"]),
        label=data["label"],
        design_id=data["design_id"],
        design=get_space(data.get("design_space", V1)).load(data["design"]),
        controller_setting=int(data["controller_setting"]),
        grasp_height_frac=data.get("grasp_height_frac"),
        training_success=float(data["training_success"]),
        training_hold_fraction=float(data["training_hold_fraction"]),
        mass_g=float(data["mass_g"]),
        origin=data["origin"],
        content_sha256=data["content_sha256"],
        frozen_at=data["frozen_at"],
    )
    if not verify_frozen(frozen):
        raise ValueError(f"{path}: content does not match its sha256")
    return frozen


def verify_frozen(frozen: FrozenDesign) -> bool:
    """True when the content hash matches and the stored design id matches the geometry."""
    return (
        _content_hash(frozen.content()) == frozen.content_sha256
        and space_of(frozen.design).design_id(frozen.design) == frozen.design_id
    )
