"""The Studio recommendation: finalists' full-task results in, one setup out (track S5, product mode).

A pure function of the finalists' :class:`~rtt.studio.types.SetupResult` values (no model sets
it; the agent only explains it), in the v1 tournament's pattern (:mod:`rtt.effectors.recommend`):

1. **Evidence.** Results from the S1 reference controllers (``controllers = "reference"``) are
   refused as evidence: they are labelled and listed, never ranked, and there is no
   recommendation. Then :func:`rtt.studio.fairness.assert_fair` runs on every result before any
   ranking (same wrist, constants, task, conditions, seeds and controllers).
2. **Eligible:** the setup completed the task at least once on the grid, and no run reset the
   simulator (``physics_invalid``).
3. **Rank:** :meth:`SetupResult.rank_key`: completion rate, then the weakest step's success
   rate, then median cycle time (with changes), then complexity (tools, actuators, total mass,
   changes), then the setup id (a total order).
4. **Simplicity rule** (as v1): among the setups within :data:`SIMPLICITY_MARGIN` of the best
   remaining completion rate and weakest step, whose cycle time is at most :data:`CYCLE_MARGIN`
   slower [assumed], the simplest (complexity key) is picked.
5. **Fidelity rule** (as v1): an approximation-tier winner carries the best all-rigid eligible
   setup as ``runner_up_rigid`` and the caveat "approximation: verify on hardware".
6. **Baseline:** the best single off-the-shelf effector is always reported next to the
   recommendation, with the difference in completion and cycle time.
7. **No winner:** nothing eligible gives ``recommendation = null`` with the reason and the closest
   setups, their weakest steps and failure modes. A pick is never forced.

:func:`sensitivity` re-ranks the same runs with other tool-change and head-switch times
(orchestrator ruling 3). A change is a fixed clock advance, not simulated physics, so each run's
cycle time is recomputed exactly: ``cycle - change_time + n_tool_changes * c + n_presentations * p``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from rtt.effectors.types import ContractError
from rtt.studio.fairness import assert_fair, fairness_problems
from rtt.studio.types import (
    CONTRACT_VERSION,
    PRESENT_S,
    PRODUCT_MODE_LABEL,
    STUDIO_WRIST,
    TOOL_CHANGE_S,
    FidelityTier,
    SetupResult,
    ToolChanger,
)

__all__ = [
    "APPROXIMATION_CAVEAT",
    "ASSUMED_CONSTANTS",
    "CYCLE_MARGIN",
    "SENSITIVITY_CHANGE_S",
    "SENSITIVITY_PRESENT_S",
    "SIMPLICITY_MARGIN",
    "STUDIO_CAVEATS",
    "eligibility",
    "rank_setups",
    "recommend_setups",
    "sensitivity",
]

#: Completion (and weakest-step) rates closer than this count as a tie for the simplicity rule.
SIMPLICITY_MARGIN: Final = 0.05
#: ... provided the simpler setup is at most this much slower (relative cycle time) [assumed].
CYCLE_MARGIN: Final = 0.10
APPROXIMATION_CAVEAT: Final = "approximation: verify on hardware"
#: Ruling 3: the change times the sensitivity report re-ranks with (s).
SENSITIVITY_CHANGE_S: Final[tuple[float, ...]] = (3.0, 6.0, 12.0)
SENSITIVITY_PRESENT_S: Final[tuple[float, ...]] = (0.75, 1.5, 3.0)
_EPS: Final = 1e-9


def _const(value: float, unit: str, note: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "tag": "assumed", "note": note}


_CHANGER = ToolChanger()
#: Every constant the comparison rests on that has no source: equal for every candidate,
#: printed on every certificate (``[assumed]``).
ASSUMED_CONSTANTS: Final[Mapping[str, dict[str, Any]]] = {
    "roll_torque_nm": _const(STUDIO_WRIST.roll_torque_nm, "N m", "wrist roll torque limit"),
    "roll_speed_max_deg_s": _const(STUDIO_WRIST.roll_speed_max_deg_s, "deg/s", "wrist roll speed cap"),
    "tool_change_s": _const(TOOL_CHANGE_S, "s", "automatic tool change time (a fixed clock advance)"),
    "present_s": _const(PRESENT_S, "s", "multi-tool head switch time (a fixed tilt, not simulated)"),
    "changer_master_kg": _const(
        _CHANGER.master_mass_kg, "kg", "tool changer master plate (stays on the wrist)"
    ),
    "changer_tool_plate_kg": _const(_CHANGER.tool_plate_mass_kg, "kg", "tool changer plate on every tool"),
    "changer_stack_mm": _const(_CHANGER.stack_mm, "mm", "tool changer stack height"),
    "arm_payload_kg": _const(3.0, "kg", "largest tool stack the arm may carry; heavier setups are refused"),
    "bracket_density_kg_m3": _const(1300.0, "kg/m^3", "printed carbon-filled polymer bracket"),
    "placement_error_mm": _const(2.0, "mm", "per-object placement error of the condition grid (uniform)"),
    "simplicity_margin": _const(SIMPLICITY_MARGIN, "rate", "completion-rate tie for the simplicity rule"),
    "cycle_margin": _const(CYCLE_MARGIN, "fraction", "a simpler setup may be this much slower"),
    "sweep_clearance_mm": _const(5.0, "mm", "multi-tool head swept-volume clearance"),
}


#: The Studio's own caveats (``docs/STUDIO.md`` section 6), printed with every result.
STUDIO_CAVEATS: Final[tuple[str, ...]] = (
    "The floating wrist abstracts the arm: there is no arm reach or arm collision model.",
    "Presenting a multi-tool head is a fixed tilt per segment plus a fixed time, not a simulated motion.",
    "A tool change is a fixed time, not a simulated docking.",
    "Controllers are scripted and open-loop on the placement error.",
    "A recommendation is a simulation ranking under the stated conditions, not a hardware guarantee.",
)


# --------------------------------------------------------------------------- rules


def _modes(sr: SetupResult) -> Counter[str]:
    return Counter(s.failure_mode for r in sr.runs for s in r.steps if s.failure_mode)


def eligibility(sr: SetupResult) -> tuple[bool, str | None]:
    """Whether a setup may be recommended, and why not."""
    if sr.completion_rate <= 0.0:
        step, rate = sr.weakest_step
        modes = Counter(
            s.failure_mode for r in sr.runs for s in r.steps if s.step_id == step and s.failure_mode
        )
        why = ", ".join(f"{m} {n}x" for m, n in modes.most_common(3)) or "no run reached it"
        return False, f"never completed the task (weakest step {step}: {100 * rate:.0f}%, {why})"
    if _modes(sr)["physics_invalid"]:
        return False, "a run reset the simulator (physics_invalid)"
    return True, None


def _close(top: SetupResult, other: SetupResult) -> bool:
    return (
        top.completion_rate - other.completion_rate <= SIMPLICITY_MARGIN + _EPS
        and top.weakest_step[1] - other.weakest_step[1] <= SIMPLICITY_MARGIN + _EPS
        and other.cycle_time_s <= top.cycle_time_s * (1.0 + CYCLE_MARGIN) + _EPS
    )


def _base_key(sr: SetupResult) -> tuple[Any, ...]:
    return (*sr.rank_key(), sr.setup.setup_id)


def rank_setups(results: Sequence[SetupResult]) -> list[tuple[SetupResult, SetupResult | None]]:
    """Eligible setups in recommendation order, each with the setup the simplicity rule put it
    ahead of (``None`` when it was first on the base order anyway)."""
    remaining = sorted((r for r in results if eligibility(r)[0]), key=_base_key)
    out: list[tuple[SetupResult, SetupResult | None]] = []
    while remaining:
        top = remaining[0]
        close = [r for r in remaining if _close(top, r)]
        pick = min(close, key=lambda r: (r.complexity.key(), _base_key(r)))
        out.append((pick, top if pick is not top else None))
        remaining.remove(pick)
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.0f}%"


def _simpler(pick: SetupResult, over: SetupResult) -> str:
    a, b = pick.complexity, over.complexity
    if a.n_tools != b.n_tools:
        return f"{a.n_tools} tool{'s' if a.n_tools != 1 else ''} against {b.n_tools}"
    if a.n_actuators != b.n_actuators:
        return f"{a.n_actuators} actuators against {b.n_actuators}"
    if round(a.total_mass_kg, 4) != round(b.total_mass_kg, 4):
        return f"{1000 * a.total_mass_kg:.0f} g against {1000 * b.total_mass_kg:.0f} g"
    return f"{a.n_changes} changes against {b.n_changes}"


def _why(
    pick: SetupResult, over: SetupResult | None, ranked: Sequence[tuple[SetupResult, SetupResult | None]]
) -> str:
    rate = pick.completion_rate
    if over is not None:
        return (
            f"Within {SIMPLICITY_MARGIN:g} of the top completion rate ({_pct(rate)} against "
            f"{_pct(over.completion_rate)}, {over.setup.label or over.setup.kind.value}), at most "
            f"{100 * CYCLE_MARGIN:.0f}% slower, and simpler: {_simpler(pick, over)}."
        )
    if len(ranked) == 1:
        return f"The only setup that completed the task: {_pct(rate)} of the runs, {pick.cycle_time_s:.1f} s cycle."
    nxt = ranked[1][0]
    if rate > nxt.completion_rate + _EPS:
        return (
            f"Highest full-task completion of the {len(ranked)} eligible setups ({_pct(rate)}, next "
            f"{_pct(nxt.completion_rate)})."
        )
    if pick.weakest_step[1] > nxt.weakest_step[1] + _EPS:
        return (
            f"Ties on completion ({_pct(rate)}) with a stronger weakest step ({_pct(pick.weakest_step[1])})."
        )
    if pick.cycle_time_s < nxt.cycle_time_s - 1e-3:
        return (
            f"Ties on completion ({_pct(rate)}) and the weakest step, with the shorter cycle "
            f"({pick.cycle_time_s:.1f} s against {nxt.cycle_time_s:.1f} s)."
        )
    return f"Ties on completion, weakest step and cycle time, and is at least as simple ({_pct(rate)})."


def _tools(sr: SetupResult) -> list[dict[str, Any]]:
    return [
        {
            "effector_id": t.effector_id,
            "display_name": t.display_name,
            "family": t.family.value,
            "fidelity": t.fidelity.value,
        }
        for t in sr.setup.tools
    ]


def _row(
    sr: SetupResult,
    eligible: bool,
    reason: str | None,
    payload: Mapping[str, float],
    best_single_id: str | None,
) -> dict[str, Any]:
    weakest, rate = sr.weakest_step
    return {
        "setup_id": sr.setup.setup_id,
        "kind": sr.setup.kind.value,
        "label": sr.setup.label,
        "tools": _tools(sr),
        "eligible": eligible,
        "reason": reason,
        "completion_rate": round(sr.completion_rate, 4),
        "step_success": {k: round(v, 4) for k, v in sr.step_success.items()},
        "weakest_step": {"step_id": weakest, "success_rate": round(rate, 4)},
        "cycle_time_s": round(sr.cycle_time_s, 3),
        "complexity": sr.complexity.to_dict(),
        "payload_kg": round(payload.get(sr.setup.setup_id, sr.complexity.tool_mass_kg), 4),
        "fidelity": sr.fidelity.value,
        "failure_modes": dict(_modes(sr).most_common()),
        "best_single": sr.setup.setup_id == best_single_id,
        "n_runs": len(sr.runs),
    }


def _card(sr: SetupResult, why: str | None, payload: Mapping[str, float]) -> dict[str, Any]:
    weakest, rate = sr.weakest_step
    return {
        "setup": sr.setup.to_dict(),
        "setup_id": sr.setup.setup_id,
        "kind": sr.setup.kind.value,
        "label": sr.setup.label,
        "tools": _tools(sr),
        "why": why,
        "completion_rate": round(sr.completion_rate, 4),
        "step_success": {k: round(v, 4) for k, v in sr.step_success.items()},
        "weakest_step": {"step_id": weakest, "success_rate": round(rate, 4)},
        "cycle_time_s": round(sr.cycle_time_s, 3),
        "complexity": sr.complexity.to_dict(),
        "payload_kg": round(payload.get(sr.setup.setup_id, sr.complexity.tool_mass_kg), 4),
        "fidelity": sr.fidelity.value,
        "caveats": list(sr.caveats),
        "caveat": APPROXIMATION_CAVEAT if sr.fidelity is not FidelityTier.RIGID else None,
        "n_runs": len(sr.runs),
    }


_RULES: Final = {
    "evidence": "results from the reference controllers are refused; assert_fair runs before ranking",
    "eligible": "completed the task at least once on the grid; no physics_invalid run",
    "rank": "completion rate, weakest step, median cycle time with changes, complexity, setup id",
    "simplicity_margin": SIMPLICITY_MARGIN,
    "cycle_margin": CYCLE_MARGIN,
    "simplicity": "within the margins of the best remaining setup, the simplest (tools, actuators, mass, changes)",
    "fidelity": "an approximation-tier winner carries the best rigid-tier eligible setup as runner-up",
    "baseline": "the best single off-the-shelf effector is always reported",
}


def recommend_setups(
    results: Sequence[SetupResult],
    *,
    best_single_id: str | None = None,
    payload: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The recommendation block (module docstring). ``best_single_id``: the setup id of the best
    single off-the-shelf effector among ``results``; ``payload``: setup id -> carried kg."""
    payload = dict(payload or {})
    out: dict[str, Any] = {
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "evidence": "accepted",
        "reason": None,
        "recommendation": None,
        "runner_up_rigid": None,
        "best_single": None,
        "versus_best_single": None,
        "ranking": [],
        "no_winner": None,
        "rules": dict(_RULES),
    }
    if not results:
        out["no_winner"] = {
            "reason": "no setup could be formed for this task (see the pruning reasons)",
            "closest": [],
        }
        return out
    runs = [r for sr in results for r in sr.runs]
    if any(r.controllers == "reference" for r in runs):
        out["evidence"] = "refused"
        out["reason"] = (
            "these runs used the S1 reference controllers (smoke tests): they are not evidence, "
            "so nothing is ranked or recommended"
        )
        out["reference_results"] = [
            {
                **_row(sr, False, "reference controllers: not evidence", payload, best_single_id),
                "controllers": "reference",
            }
            for sr in sorted(results, key=_base_key)
        ]
        return out
    problems = fairness_problems(runs, [sr.setup for sr in results])
    if problems:
        raise ContractError("unfair comparison: " + "; ".join(problems))
    assert_fair(runs, [sr.setup for sr in results])
    verdicts = {sr.setup.setup_id: eligibility(sr) for sr in results}
    ranked = rank_setups(results)
    out["ranking"] = [_row(sr, True, None, payload, best_single_id) for sr, _ in ranked]
    rest = sorted((sr for sr in results if not verdicts[sr.setup.setup_id][0]), key=_base_key)
    out["ranking"] += [
        _row(sr, False, verdicts[sr.setup.setup_id][1], payload, best_single_id) for sr in rest
    ]
    best_single = next((sr for sr in results if sr.setup.setup_id == best_single_id), None)
    if best_single is not None:
        out["best_single"] = _card(best_single, None, payload)
    if ranked:
        pick, over = ranked[0]
        out["recommendation"] = _card(pick, _why(pick, over, ranked), payload)
        if pick.fidelity is not FidelityTier.RIGID:
            ru = next((sr for sr, _ in ranked if sr.fidelity is FidelityTier.RIGID), None)
            out["runner_up_rigid"] = (
                _card(ru, None, payload)
                if ru is not None
                else {"setup": None, "basis": "no rigid-tier setup is eligible"}
            )
        if best_single is not None:
            out["versus_best_single"] = {
                "same_setup": best_single.setup.setup_id == pick.setup.setup_id,
                "completion_delta": round(pick.completion_rate - best_single.completion_rate, 4),
                "cycle_delta_s": round(pick.cycle_time_s - best_single.cycle_time_s, 3),
                "extra_tools": pick.complexity.n_tools - best_single.complexity.n_tools,
            }
    else:
        closest = sorted(results, key=_base_key)[:3]
        out["no_winner"] = {
            "reason": "no setup completed the task on the condition grid (nothing is forced)",
            "closest": [
                _row(sr, False, verdicts[sr.setup.setup_id][1], payload, best_single_id) for sr in closest
            ],
        }
    return out


# --------------------------------------------------------------------------- sensitivity (ruling 3)


def _retimed(sr: SetupResult, change_s: float, present_s: float) -> SetupResult:
    runs = []
    for r in sr.runs:
        change = r.n_tool_changes * change_s + r.n_presentations * present_s
        runs.append(replace(r, cycle_time_s=r.cycle_time_s - r.change_time_s + change, change_time_s=change))
    return replace(sr, runs=tuple(runs))


def sensitivity(
    results: Sequence[SetupResult],
    change_times: Sequence[float] = SENSITIVITY_CHANGE_S,
    present_times: Sequence[float] = SENSITIVITY_PRESENT_S,
) -> dict[str, Any]:
    """The ranking under every (tool change, head switch) time pair (module docstring)."""
    if not results:
        return {"combos": [], "recommendation_changes": False}
    base = [sr.setup.setup_id for sr, _ in rank_setups(results)]
    combos = []
    positions: dict[str, list[int | None]] = {sr.setup.setup_id: [] for sr in results}
    for c in change_times:
        for p in present_times:
            adj = [_retimed(sr, c, p) for sr in results]
            order = [sr.setup.setup_id for sr, _ in rank_setups(adj)]
            cycles = {sr.setup.setup_id: round(sr.cycle_time_s, 3) for sr in adj}
            for sid, pos in positions.items():
                pos.append(order.index(sid) if sid in order else None)
            combos.append(
                {
                    "tool_change_s": c,
                    "present_s": p,
                    "order": order,
                    "recommended": order[0] if order else None,
                    "changed": order[:1] != base[:1],
                    "order_changed": order != base,
                    "cycle_time_s": cycles,
                }
            )
    return {
        "baseline": {"tool_change_s": TOOL_CHANGE_S, "present_s": PRESENT_S, "order": base},
        "combos": combos,
        "recommendation_changes": any(c["changed"] for c in combos),
        "order_changes": sum(c["order_changed"] for c in combos),
        "positions": positions,
        "kinds": {sr.setup.setup_id: sr.setup.kind.value for sr in results},
        "note": "change times are a fixed clock advance, so cycle times are recomputed exactly from the same runs",
    }
