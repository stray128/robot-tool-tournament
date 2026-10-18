"""Library items as effector entrants, and the compile and attach gate.

An item with a native simulation model or a valid rigid proxy becomes an
:class:`~rtt.effectors.types.EffectorSpec` of an existing family:

* grippers, hands and tools through the catalog families (``catalog_gripper``,
  ``catalog_hand``), whose loaders read the item's MJCF
  (:data:`rtt.effectors.catalog.library_models.LIBRARY_MODELS`);
* suction cups through the seal-gated ``suction`` family (``sim.family`` in ``SOURCE.json``).

They are **not** default presets of any family: the per-part tournament's catalog group stays
at the founder's 12 candidates. :func:`library_entrants` hands them to the Studio's candidate
search, or to any caller that builds its own candidate list.

The **gate** (:func:`gate_item`) builds every entrant spec in the shared scene on every training
part. It passes when no build raises :class:`~rtt.effectors.types.ContractError` (an
``InvalidEffector`` for one part, such as a part wider than the opening, is a recorded refusal,
not a failure). An item that cannot attach cleanly is dropped with the recorded reason.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rtt.library.records import item_keys, read_item

__all__ = [
    "GATE_PARTS",
    "GateResult",
    "entrant_reason",
    "gate_item",
    "item_specs",
    "library_entrants",
]

#: The six training parts: every entrant is built on each of them.
GATE_PARTS: tuple[str, ...] = (
    "block",
    "steel_slug",
    "hex_standoff",
    "l_bracket",
    "pipe_tee",
    "flanged_bushing",
)


def item_specs(key: str) -> tuple[Any, ...]:
    """The entrant specs of one item (empty when it has none)."""
    from rtt.effectors.catalog.family import vendored_spec
    from rtt.effectors.catalog.library_models import LIBRARY_MODELS
    from rtt.effectors.types import EffectorFamily

    item = read_item(key)
    if item.sim.get("entrant") is False:
        return ()
    if key in LIBRARY_MODELS:
        cm = LIBRARY_MODELS[key]
        fam = EffectorFamily.CATALOG_HAND if cm.kind == "hand" else EffectorFamily.CATALOG_GRIPPER
        return tuple(vendored_spec(fam, cm, syn) for syn in cm.synergies)
    if item.sim.get("family") == "suction":
        from rtt.effectors.suction import cup_spec

        return (cup_spec(float(item.sim["params"]["diameter_mm"])),)
    return ()


def entrant_reason(key: str) -> str | None:
    """Why an item is not an entrant (``None``: it is one)."""
    item = read_item(key)
    if item_specs(key):
        return None
    if item.sim.get("reason"):
        return str(item.sim["reason"])
    if item.sim_status in ("native", "proxy"):
        return "no catalog mounting is defined for it yet"
    return "no simulation model"


def library_entrants() -> tuple[Any, ...]:
    """Every entrant spec of the library, in item order (deterministic)."""
    out = []
    for key in item_keys():
        out.extend(item_specs(key))
    return tuple(out)


def PROBE_PART() -> Any:
    """A 10 mm cube: proves the attach of an effector too small for every training part."""
    from rtt.parts.library import build_part

    return build_part("probe_block_10mm", "block", {"x": 10.0, "y": 10.0, "z": 10.0})


@dataclass
class GateResult:
    """The compile and attach gate of one entrant spec."""

    key: str
    effector_id: str
    family: str
    model: str
    display_name: str
    primitive: str
    built: list[str] = field(default_factory=list)  # parts it built on
    probe: bool = False  # built only on the small probe block (every training part refused)
    refused: dict[str, str] = field(default_factory=dict)  # part -> InvalidEffector reason
    error: str | None = None  # ContractError or any other failure: the item is dropped
    n_actuators: int | None = None
    mass_kg: float | None = None
    reach_mm: float | None = None
    opening_mm: float | None = None
    seconds: float = 0.0
    nominal: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Compiled and attached in the shared scene with no contract error (on a training part,
        or on the probe block when every training part was refused for size)."""
        return self.error is None and (bool(self.built) or self.probe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "effector_id": self.effector_id,
            "family": self.family,
            "model": self.model,
            "display_name": self.display_name,
            "primitive": self.primitive,
            "passed": self.passed,
            "built": self.built,
            "probe_only": self.probe,
            "refused": self.refused,
            "error": self.error,
            "n_actuators": self.n_actuators,
            "mass_kg": None if self.mass_kg is None else round(self.mass_kg, 4),
            "reach_mm": None if self.reach_mm is None else round(self.reach_mm, 1),
            "opening_mm": None if self.opening_mm is None else round(self.opening_mm, 1),
            "nominal": self.nominal,
        }


def gate_item(key: str, *, parts: Sequence[str] = GATE_PARTS, nominal: bool = False) -> list[GateResult]:
    """Build every entrant spec of ``key`` in the shared scene on every part (module docstring).

    With ``nominal``, also run one nominal episode (no placement error, seed 0) per built part
    and record held, engaged and the outcome: information, not a gate.
    """
    from rtt.effectors.registry import get_family
    from rtt.effectors.runner import run_effector_episode
    from rtt.effectors.scene import build_effector_scene
    from rtt.effectors.types import ContractError, InvalidEffector
    from rtt.parts.library import get_part
    from rtt.sim import objects as objs
    from rtt.sim.task import default_task_v2

    task = default_task_v2()
    out = []
    for spec in item_specs(key):
        res = GateResult(
            key=key,
            effector_id=spec.effector_id,
            family=spec.family.value,
            model=spec.model,
            display_name=spec.display_name,
            primitive=spec.primitive,
        )
        t0 = time.perf_counter()
        impl = get_family(spec.family)
        for name in parts:
            part = get_part(name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scene = build_effector_scene(impl, spec, part, task)
            except InvalidEffector as exc:
                res.refused[name] = str(exc)
                continue
            except ContractError as exc:
                res.error = f"ContractError: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - any other failure drops the item, with its reason
                res.error = f"{type(exc).__name__}: {exc}"
                break
            res.built.append(name)
            build = scene.build
            res.n_actuators = build.n_actuators
            res.mass_kg = build.mass
            res.reach_mm = 1000.0 * build.reach
            res.opening_mm = None if build.max_opening is None else 1000.0 * build.max_opening
            if nominal:
                r = run_effector_episode(spec, part, objs.Placement(), 0, task=task)
                res.nominal[name] = {
                    "held": bool(r.held),
                    "engaged": bool(r.engaged),
                    "outcome": r.failure_mode or "held",
                    "penetration_mm": None
                    if r.penetration_mm != r.penetration_mm
                    else round(float(r.penetration_mm), 3),
                }
        if res.error is None and not res.built:  # every part refused: still prove the attach
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scene = build_effector_scene(impl, spec, PROBE_PART(), task)
                res.probe = True
                res.n_actuators = scene.build.n_actuators
                res.mass_kg = scene.build.mass
                res.reach_mm = 1000.0 * scene.build.reach
                res.opening_mm = None if scene.build.max_opening is None else 1000.0 * scene.build.max_opening
            except InvalidEffector as exc:
                res.error = f"refused even the 10 mm probe block: {exc}"
            except Exception as exc:  # noqa: BLE001
                res.error = f"{type(exc).__name__}: {exc}"
        res.seconds = time.perf_counter() - t0
        out.append(res)
    return out
