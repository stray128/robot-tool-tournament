"""Fairness rules of the Studio, read from compiled models and results (``docs/STUDIO.md`` section 5).

Every candidate setup in one comparison must share:

1. **the wrist**: the four frozen core servos (x, y, z, yaw: ranges, armature, damping, gains,
   force limits), the continuous roll (unlimited range, armature, gain, torque limit, the roll
   speed cap) and the limited pitch (range, armature, gains, torque limit, the pitch
   speed cap), and the wrist bodies' masses and gravity compensation (:func:`wrist_signature`),
   plus the solver options;
2. **the Studio constants**: :data:`~rtt.studio.types.STUDIO_WRIST`, the tool changer of every
   tool-change setup, the presentation time of every multi-tool head;
   **the scoring rules** (:func:`scoring_signature`): the success tolerances
   (:data:`~rtt.studio.types.CRITERION_TOLERANCE`), the 1 mm penetration limit and the 10 ms
   transient allowance (:data:`~rtt.studio.plan.PENETRATION_TRANSIENT_MS`);
3. **the task and the conditions**: the same ``task_id``, the same ``(condition_id, seed)``
   list in the same order for every setup (common random numbers), the same controller set.

:func:`fairness_sha` hashes 1 and the constants and scoring rules of 2 for a scene; every
:class:`~rtt.studio.types.TaskResult` carries it. :func:`fairness_problems` lists every
violation in a set of results (empty when fair); :func:`assert_fair` raises on any. The
recommender (track S5) calls :func:`assert_fair` before it ranks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from rtt.effectors.types import ContractError, canonical_json
from rtt.studio.plan import HELD_PENETRATION_MAX_MM, PENETRATION_TRANSIENT_MS
from rtt.studio.scene import WRIST_BODIES, WRIST_JOINTS
from rtt.studio.types import CONTRACT_VERSION, CRITERION_TOLERANCE, Setup, SetupKind, TaskResult

if TYPE_CHECKING:
    import mujoco

    from rtt.studio.scene import StudioScene

__all__ = ["assert_fair", "fairness_problems", "fairness_sha", "scoring_signature", "wrist_signature"]


def _vec(a: Any, digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(a).ravel()]


def wrist_signature(model: mujoco.MjModel) -> dict[str, Any]:
    """The six wrist joints and servos, the wrist bodies and the solver options of a model."""
    out: dict[str, Any] = {}
    for name in WRIST_JOINTS:
        j, a = model.joint(name), model.actuator(name)
        out[name] = {
            "type": int(j.type[0]),
            "range": _vec(j.range),
            "limited": int(model.jnt_limited[j.id]),
            "armature": _vec(j.armature),
            "damping": _vec(j.damping),
            "stiffness": _vec(j.stiffness),
            "frictionloss": _vec(j.frictionloss),
            "gain": _vec(a.gainprm[:3]),
            "bias": _vec(a.biasprm[:3]),
            "forcerange": _vec(a.forcerange),
            "forcelimited": int(model.actuator_forcelimited[a.id]),
            "ctrlrange": _vec(a.ctrlrange),
            "ctrllimited": int(model.actuator_ctrllimited[a.id]),
            "gear": _vec(a.gear[:1]),
        }
    for name in WRIST_BODIES:
        b = model.body(name).id
        out[f"body:{name}"] = {
            "mass": round(float(model.body_mass[b]), 9),
            "gravcomp": float(model.body_gravcomp[b]),
        }
    opt = model.opt
    out["option"] = {
        "timestep": float(opt.timestep),
        "integrator": int(opt.integrator),
        "cone": int(opt.cone),
        "impratio": float(opt.impratio),
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
        "tolerance": float(opt.tolerance),
        "gravity": _vec(opt.gravity),
        "disableflags": int(opt.disableflags),
        "enableflags": int(opt.enableflags),
    }
    return out


def scoring_signature() -> dict[str, Any]:
    """The rules a step is scored by, as constants: the success tolerance of every quantity that has
    one, the penetration limit (mm) and the longest transient above it (ms)."""
    return {
        "criterion_tolerance": {k: float(v) for k, v in sorted(CRITERION_TOLERANCE.items())},
        "penetration_limit_mm": float(HELD_PENETRATION_MAX_MM),
        "penetration_transient_ms": float(PENETRATION_TRANSIENT_MS),
    }


def fairness_sha(scene: StudioScene) -> str:
    """sha256 of the scene's wrist signature, task digest, the Studio wrist constants and the
    scoring rules (:func:`scoring_signature`)."""
    payload = {
        "contract": CONTRACT_VERSION,
        "wrist": wrist_signature(scene.model),
        "wrist_limits": scene.wrist.to_dict(),
        "scoring": scoring_signature(),
        "task_digest": scene.task_digest,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def fairness_problems(results: Iterable[TaskResult], setups: Sequence[Setup] = ()) -> list[str]:
    """Everything that differs between candidates and must not (section docstring)."""
    runs = list(results)
    problems: list[str] = []
    if not runs:
        return problems
    for attr in ("task_id", "fairness_sha", "controllers"):
        values = sorted({str(getattr(r, attr)) for r in runs})
        if len(values) > 1:
            problems.append(f"results differ in {attr}: {[v[:12] for v in values]}")
    grids: dict[str, list[tuple[str, int]]] = {}
    for r in runs:
        grids.setdefault(r.setup_id, []).append((r.condition_id, int(r.seed)))
    first_id, first = next(iter(grids.items()))
    for setup_id, grid in grids.items():
        if grid != first:
            problems.append(f"setup {setup_id[:12]} ran other conditions or seeds than setup {first_id[:12]}")
    changers = {
        canonical_json(s.changer.to_dict()) for s in setups if s.kind is SetupKind.TOOL_CHANGE and s.changer
    }
    if len(changers) > 1:
        problems.append("tool-change setups use different tool changers")
    presents = {s.present_s for s in setups if s.kind is SetupKind.MULTI_HEAD}
    if len(presents) > 1:
        problems.append("multi-tool heads use different presentation times")
    return problems


def assert_fair(results: Iterable[TaskResult], setups: Sequence[Setup] = ()) -> None:
    """Raise :class:`~rtt.effectors.types.ContractError` if :func:`fairness_problems` finds any."""
    problems = fairness_problems(results, setups)
    if problems:
        raise ContractError("unfair comparison: " + "; ".join(problems))
