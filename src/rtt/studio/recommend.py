"""The Studio recommendation: finalists' full-task results in, one setup out (track S5, product mode).

A pure function of the finalists' :class:`~rtt.studio.types.SetupResult` values (no model sets
it; the agent only explains it), in the v1 tournament's pattern (:mod:`rtt.effectors.recommend`):

1. **Evidence.** Results from the S1 reference controllers (``controllers = "reference"``) are
   refused as evidence: they are labelled and listed, never ranked, and there is no
   recommendation. Then :func:`rtt.studio.fairness.assert_fair` runs on every result before any
   ranking (same wrist, constants, task, conditions, seeds and controllers).
2. **Eligible:** the setup completed the task at least once on the grid, and no run reset the
   simulator (``physics_invalid``).
3. **Base order:** :meth:`SetupResult.rank_key`: completion rate, then the weakest step's success
   rate, then median cycle time (with changes), then complexity (tools, actuators, total mass,
   changes), then the setup id (a total order).
4. **Integrator rules** (:mod:`rtt.studio.rules`), one pick at a time from the remaining setups:

   * **R09, R10:** every setup without a soft arm or a dexterous hand ranks first, then those
     with a soft arm, then those with a dexterous hand (the largest complexity penalty):
     eligible, never preferred over a rigid setup.
   * **R01:** within that group, setups are compared by their added pieces
     (:func:`added_pieces`: effectors beyond one, plus the tool changer). The baseline is the
     best single tool (one effector, no changer, no head) on the base order, so completion first.
     It is picked unless (a) its weakest step's success is under
     ``SIMPLICITY_MARGIN["step_floor"]`` (it fails a step outright) and a setup of more pieces
     completes more, or (b) a setup of more pieces completes more than
     ``SIMPLICITY_MARGIN["per_extra_effector"]`` [assumed] per added piece better. Every setup of
     more pieces is tested against the baseline; those that displace it are compared with each
     other the same way (the best of the fewest pieces among them is the next base, per piece of
     difference). Cycle time never lets a more complex setup win (it is still reported).
   * **Same complexity** (the v1 simplicity rule), only among the setups R01 lets through: those
     of the chosen piece count that no more complex setup displaces. Within :data:`TIE_MARGIN` of
     the best one's completion rate and weakest step and at most :data:`CYCLE_MARGIN` slower
     [assumed] they tie, and the simplest (complexity key) is picked.

   The reason names the rule that decided ("Integrator rule R01: one tool finishes the job ...").
5. **Fidelity rule** (as v1): an approximation-tier winner carries the best all-rigid eligible
   setup as ``runner_up_rigid`` and the caveat "approximation: verify on hardware".
6. **Baseline:** the R01 baseline (the best eligible single tool, by completion first) is printed
   beside any pick (``baseline``, ``versus_baseline``: the completion and cycle difference, the added
   pieces, the gain they had to earn and what decided: ``baseline``, ``tie`` when the
   same-complexity tie picked a simpler single over it, ``margin``, ``step_floor``, ``R09``, ``R10``
   or ``ranking``); when no single tool is eligible, ``baseline`` says why. The best single
   off-the-shelf effector of the search is reported too (``best_single``), with the
   recommendation's difference in completion and cycle time.
7. **No winner:** nothing eligible gives ``recommendation = null`` with the reason and the closest
   setups, their weakest steps and failure modes. A pick is never forced.

:func:`sensitivity` re-ranks the same runs with other tool-change and head-switch times
(orchestrator ruling 3; integrator rule R07). A change is a fixed clock advance, not simulated
physics, so each run's cycle time is recomputed exactly:
``cycle - change_time + n_tool_changes * c + n_presentations * p``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final

from rtt.effectors.types import ContractError
from rtt.studio.fairness import assert_fair, fairness_problems
from rtt.studio.rules import (
    HOOK_MIN_MM,
    HOOK_THROAT_MM,
    PROCESS_TORQUE_NM,
    RULE_CLASSES,
    SOCKET_FIT_MM,
    SOFT_ARM_MAX_PART_KG,
    setup_rule_class,
)
from rtt.studio.types import (
    CONTRACT_VERSION,
    PRESENT_S,
    PRODUCT_MODE_LABEL,
    STUDIO_WRIST,
    TOOL_CHANGE_S,
    FidelityTier,
    Setup,
    SetupKind,
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
    "TIE_MARGIN",
    "added_pieces",
    "eligibility",
    "rank_setups",
    "recommend_setups",
    "sensitivity",
    "stops",
]

#: Integrator rule R01 [assumed]: a setup of more pieces must complete more than
#: ``per_extra_effector`` per added piece better than a simpler one; a setup whose weakest step
#: succeeds less than ``step_floor`` fails that step outright and loses to any more complex setup
#: that completes more.
SIMPLICITY_MARGIN: Final[Mapping[str, float]] = MappingProxyType(
    {"per_extra_effector": 0.05, "step_floor": 0.5}
)
#: Among setups of the same complexity, completion (and weakest-step) rates this close tie ...
TIE_MARGIN: Final = 0.05
#: ... provided the simpler setup is at most this much slower (relative cycle time) [assumed].
CYCLE_MARGIN: Final = 0.10
APPROXIMATION_CAVEAT: Final = "approximation: verify on hardware"
#: Ruling 3 and integrator rule R07: the change times the sensitivity report re-ranks with (s),
#: around the contract's :data:`~rtt.studio.types.TOOL_CHANGE_S`.
SENSITIVITY_CHANGE_S: Final[tuple[float, ...]] = (3.0, 6.0, 7.5, 10.0, 12.0)
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
    "pitch_range_deg": _const(
        STUDIO_WRIST.pitch_range_deg, "deg", "wrist pitch limit, +- from straight down"
    ),
    "pitch_speed_dps": _const(STUDIO_WRIST.pitch_speed_dps, "deg/s", "wrist pitch speed cap"),
    "pitch_torque_nm": _const(STUDIO_WRIST.pitch_torque_nm, "N m", "wrist pitch torque limit"),
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
    "simplicity_per_extra_effector": _const(
        SIMPLICITY_MARGIN["per_extra_effector"],
        "rate",
        "integrator rule R01: completion a setup must add per added effector or changer",
    ),
    "simplicity_step_floor": _const(
        SIMPLICITY_MARGIN["step_floor"],
        "rate",
        "integrator rule R01: a weakest step under this fails outright",
    ),
    "tie_margin": _const(TIE_MARGIN, "rate", "completion-rate tie among setups of the same complexity"),
    "cycle_margin": _const(
        CYCLE_MARGIN, "fraction", "among setups of the same complexity, a simpler one may be this much slower"
    ),
    "soft_arm_max_part_kg": _const(
        SOFT_ARM_MAX_PART_KG, "kg", "integrator rule R09: the heaviest part a soft arm may carry"
    ),
    "hook_min_mm": _const(
        HOOK_MIN_MM, "mm", "a rigid tool hooks if its tip reaches this far sideways past the tool and flange"
    ),
    "hook_throat_mm": _const(
        HOOK_THROAT_MM, "mm", "room a lip needs between a hook and the flange, else the flange blocks it"
    ),
    "process_torque_nm": _const(
        PROCESS_TORQUE_NM,
        "N m",
        "integrator rule R05: tightening a threaded member to this torque or more needs a nut runner "
        "(1 N m above hand-tight)",
    ),
    "socket_fit_mm": _const(
        SOCKET_FIT_MM, "mm", "a nut runner's socket drives a hex at most this much under its own size"
    ),
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


def stops(sr: SetupResult) -> dict[str, Any]:
    """Where the runs stop: per step, how many runs failed there (the first failure of a run) and
    with which failure modes; the most common stopping step first."""
    where: dict[str, Counter[str]] = {}
    for r in sr.runs:
        first = next((s for s in r.steps if not s.success and s.failure_mode != "not_reached"), None)
        if first is not None:
            where.setdefault(first.step_id, Counter())[str(first.failure_mode)] += 1
    order = sorted(where, key=lambda k: (-sum(where[k].values()), k))
    return {k: {"runs": sum(where[k].values()), "failure_modes": dict(where[k].most_common())} for k in order}


def eligibility(sr: SetupResult) -> tuple[bool, str | None]:
    """Whether a setup may be recommended, and why not."""
    if sr.completion_rate <= 0.0:
        st = stops(sr)
        if not st:
            return False, "never completed the task"
        step, info = next(iter(st.items()))
        why = ", ".join(f"{m} {n}x" for m, n in info["failure_modes"].items())
        return (
            False,
            f"never completed the task (stops at {step} in {info['runs']}/{len(sr.runs)} runs: {why})",
        )
    if _modes(sr)["physics_invalid"]:
        return False, "a run reset the simulator (physics_invalid)"
    return True, None


def _close(top: SetupResult, other: SetupResult) -> bool:
    return (
        top.completion_rate - other.completion_rate <= TIE_MARGIN + _EPS
        and top.weakest_step[1] - other.weakest_step[1] <= TIE_MARGIN + _EPS
        and other.cycle_time_s <= top.cycle_time_s * (1.0 + CYCLE_MARGIN) + _EPS
    )


def _base_key(sr: SetupResult) -> tuple[Any, ...]:
    return (*sr.rank_key(), sr.setup.setup_id)


def _progress(sr: SetupResult) -> float:
    """Mean per-step success over the runs: how far the setup gets when it never completes."""
    rates = list(sr.step_success.values())
    return sum(rates) / len(rates) if rates else 0.0


def _closest_key(sr: SetupResult) -> tuple[Any, ...]:
    """Order of setups that did not qualify: further along the task first, then the rank key."""
    return (-sr.completion_rate, -round(_progress(sr), 9), *_base_key(sr))


def added_pieces(setup: Setup | SetupResult) -> int:
    """What a setup adds to one tool on the flange (integrator rule R01): every effector beyond
    the first, plus the tool changer. A two-tool head without a changer adds 1; a two-tool change
    adds 2 (the second tool and the changer)."""
    s = setup.setup if isinstance(setup, SetupResult) else setup
    return len(s.tools) - 1 + (1 if s.kind is SetupKind.TOOL_CHANGE else 0)


def _tier(sr: SetupResult) -> int:
    """R09, R10: 0 rigid, 1 with a soft arm, 2 with a dexterous hand."""
    return RULE_CLASSES.index(setup_rule_class(sr.setup))


def _fails_a_step(sr: SetupResult) -> bool:
    """R01 (a): the weakest step succeeds in less than the step floor of the runs."""
    return sr.weakest_step[1] < SIMPLICITY_MARGIN["step_floor"] - _EPS


def _needed(simple: SetupResult, other: SetupResult) -> float:
    """R01 (b): the completion gain ``other`` must beat ``simple`` by (its extra pieces)."""
    return SIMPLICITY_MARGIN["per_extra_effector"] * (added_pieces(other) - added_pieces(simple))


def _displaces(other: SetupResult, simple: SetupResult) -> bool:
    """R01: whether ``other`` (more pieces) is recommended over ``simple``. Cycle time never counts."""
    gain = other.completion_rate - simple.completion_rate
    if _fails_a_step(simple):
        return gain > _EPS
    return gain > _needed(simple, other) + _EPS


def _class_pick(pool: Sequence[SetupResult]) -> SetupResult:
    """Among setups of the same complexity (``pool`` in base order), the v1 simplicity rule."""
    top = pool[0]
    close = [r for r in pool if _close(top, r)]
    return min(close, key=lambda r: (r.complexity.key(), _base_key(r)))


@dataclass(frozen=True)
class _Choice:
    """How R01 chose a setup from one rule class. ``beaten``: the baseline and every later base
    the pick displaced (fewest pieces first); ``tie_over``: the base the same-complexity tie put
    the pick ahead of; ``rivals``: more complex setups ahead of the pick on the base order that R01
    kept it ahead of, each with the setup it failed to displace."""

    pick: SetupResult
    beaten: tuple[SetupResult, ...] = ()
    tie_over: SetupResult | None = None
    rivals: tuple[tuple[SetupResult, SetupResult], ...] = ()


def _choose(pool: Sequence[SetupResult]) -> _Choice:
    """R01 over every setup of one rule class (``pool`` in base order).

    The base is the best setup of the fewest pieces on the base order (the baseline when there are
    single tools: completion first). Every setup of more pieces is tested against it
    (:func:`_displaces`); those that displace it compete again the same way among themselves. When
    none does, the base's class keeps the pick: the same-complexity tie runs among the setups of
    that class that no more complex setup displaces."""
    k0 = min(added_pieces(r) for r in pool)
    base = next(r for r in pool if added_pieces(r) == k0)
    more = [c for c in pool if added_pieces(c) > k0]
    better = [c for c in more if _displaces(c, base)]
    if better:
        inner = _choose(better)
        return replace(inner, beaten=(base, *inner.beaten))
    peers = [r for r in pool if added_pieces(r) == k0 and not any(_displaces(c, r) for c in more)]
    pick = _class_pick(peers)
    return _Choice(pick, tie_over=base if pick is not base else None)


def _rivals(choice: _Choice, pool: Sequence[SetupResult]) -> tuple[tuple[SetupResult, SetupResult], ...]:
    """Setups of more pieces than the pick, ahead of it on the base order, each with the first
    setup of the R01 chain (the bases, then the pick) it does not displace."""
    pick = choice.pick
    out = []
    for c in pool:
        if added_pieces(c) <= added_pieces(pick) or _base_key(c) >= _base_key(pick):
            continue
        ref = next((b for b in (*choice.beaten, pick) if not _displaces(c, b)), pick)
        out.append((c, ref))
    return tuple(out)


@dataclass(frozen=True)
class _Ranked:
    """One place of the ranking: the R01 choice in its rule class, and ``demoted``, a soft-arm or
    hand setup ahead of it on the base order (R09, R10)."""

    choice: _Choice
    over: SetupResult | None = None
    demoted: SetupResult | None = None

    @property
    def sr(self) -> SetupResult:
        return self.choice.pick


def _ranked(results: Sequence[SetupResult]) -> list[_Ranked]:
    remaining = sorted((r for r in results if eligibility(r)[0]), key=_base_key)
    out: list[_Ranked] = []
    while remaining:
        tier = min(_tier(r) for r in remaining)
        pool = [r for r in remaining if _tier(r) == tier]
        choice = _choose(pool)
        choice = replace(choice, rivals=_rivals(choice, pool))
        out.append(
            _Ranked(
                choice,
                over=pool[0] if pool[0] is not choice.pick else None,
                demoted=remaining[0] if _tier(remaining[0]) > tier else None,
            )
        )
        remaining.remove(choice.pick)
    return out


def rank_setups(results: Sequence[SetupResult]) -> list[tuple[SetupResult, SetupResult | None]]:
    """Eligible setups in recommendation order (module docstring, rule 4), each with the setup a
    rule put it ahead of (``None`` when it was first on the base order anyway)."""
    return [(r.sr, r.over or r.demoted) for r in _ranked(results)]


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


def _cycles(a: SetupResult, b: SetupResult) -> str:
    x, y = a.cycle_time_s, b.cycle_time_s
    return f"{x:.2f} s against {y:.2f} s" if abs(x - y) < 0.1 else f"{x:.1f} s against {y:.1f} s"


def _noun(sr: SetupResult) -> str:
    """What a setup is, in words (lower case)."""
    n = len(sr.setup.tools)
    return {
        SetupKind.SINGLE: "one tool",
        SetupKind.MULTI_HEAD: "the two-tool head",
        SetupKind.TOOL_CHANGE: f"the {n}-tool change",
    }[sr.setup.kind]


def _name(sr: SetupResult) -> str:
    return sr.setup.label or sr.setup.kind.value


def _points(x: float) -> str:
    return f"{100.0 * x:g}"


def _fewer(pick: SetupResult, over: SetupResult) -> str:
    """How ``pick`` is simpler than ``over`` (which has more pieces)."""
    if pick.complexity.n_tools != over.complexity.n_tools:
        return _simpler(pick, over)
    k, k_over = added_pieces(pick), added_pieces(over)
    return f"{k} added piece{'s' if k != 1 else ''} against {k_over}"


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _ref(sr: SetupResult, pick: SetupResult) -> str:
    return "it" if sr is pick else f"{_noun(sr)} ({_name(sr)})"


def _kept_lead(pick: SetupResult, rival: SetupResult) -> str:
    """R01 kept the baseline ``pick`` ahead of ``rival`` (more pieces, ahead on the base order)."""
    if _fails_a_step(pick):
        return (
            f"Integrator rule R01: {_noun(rival)} ({_name(rival)}) completes no more of the runs "
            f"({_pct(rival.completion_rate)} against {_pct(pick.completion_rate)}), so the simpler setup "
            f"wins: {_fewer(pick, rival)}."
        )
    return (
        f"Integrator rule R01: {_cap(_noun(pick))} finishes the job ({_pct(pick.completion_rate)} of the "
        f"runs, cycle {_cycles(pick, rival)}); {_noun(rival)} ({_name(rival)}, "
        f"{_pct(rival.completion_rate)}) is not more than {_points(_needed(pick, rival))} points better, "
        f"so the simpler setup wins: {_fewer(pick, rival)}."
    )


def _kept(pick: SetupResult, rival: SetupResult, ref: SetupResult) -> str:
    """``rival`` (more pieces, ahead on the base order) does not displace ``ref``, a setup of the
    R01 chain (the pick or a base it displaced)."""
    if _fails_a_step(ref):
        return (
            f"{_cap(_noun(rival))} ({_name(rival)}, {_pct(rival.completion_rate)}) completes no more "
            f"than {_ref(ref, pick)}, so the simpler setup stays ahead: {_fewer(pick, rival)}."
        )
    return (
        f"{_cap(_noun(rival))} ({_name(rival)}, {_pct(rival.completion_rate)}) is not more than "
        f"{_points(_needed(ref, rival))} points better than {_ref(ref, pick)} (cycle "
        f"{_cycles(pick, rival)}), so the simpler setup stays ahead: {_fewer(pick, rival)}."
    )


def _displaced(pick: SetupResult, beaten: Sequence[SetupResult]) -> str:
    """R01 let ``pick`` displace the baseline and every later base in ``beaten``."""
    bits = []
    for b in beaten:
        if _fails_a_step(b):
            step, rate = b.weakest_step
            bits.append(
                f"{_noun(b)} ({_name(b)}) fails {step} outright ({_pct(rate)} of the runs, under "
                f"{_pct(SIMPLICITY_MARGIN['step_floor'])}) and this setup completes more "
                f"({_pct(pick.completion_rate)} against {_pct(b.completion_rate)})"
            )
        else:
            n = added_pieces(pick) - added_pieces(b)
            bits.append(
                f"it completes {_pct(pick.completion_rate)} of the runs, more than "
                f"{_points(_needed(b, pick))} points better than {_noun(b)} ({_name(b)}, "
                f"{_pct(b.completion_rate)}), the margin for {n} added piece{'s' if n != 1 else ''}"
            )
    return "Integrator rule R01: " + "; and ".join(bits) + "."


def _tie(pick: SetupResult, over: SetupResult) -> str:
    """The v1 simplicity rule among setups of the same complexity."""
    return (
        f"Within {TIE_MARGIN:g} of the top completion rate ({_pct(pick.completion_rate)} against "
        f"{_pct(over.completion_rate)}, {_name(over)}), at most {100 * CYCLE_MARGIN:.0f}% slower, and "
        f"simpler: {_simpler(pick, over)}."
    )


def _class_words(sr: SetupResult) -> tuple[str, str]:
    """(rule id, what) of a soft-arm or hand setup."""
    return ("R10", "a dexterous hand") if setup_rule_class(sr.setup) == "hand" else ("R09", "a soft arm")


def _demotion(pick: SetupResult, demoted: SetupResult) -> str:
    """R09, R10: ``demoted`` (a soft-arm or hand setup ahead of ``pick`` on the base order) ranks
    below it; the sentence names the class of the pick it ranks below."""
    rule, what = _class_words(demoted)
    below = "every rigid setup"
    if _tier(pick) > 0:  # a soft arm picked ahead of a hand (R10)
        below += f" and every setup with {_class_words(pick)[1]}"
    return (
        f"Setups with {what} rank below {below} (integrator rule {rule}), so "
        f"{_name(demoted)} ({_pct(demoted.completion_rate)}) comes later."
    )


def _plain(pick: SetupResult, ranked: Sequence[_Ranked]) -> str:
    """Why the first setup on the base order of its rule class is first (no rule moved it)."""
    rate = pick.completion_rate
    peers = [r.sr for r in ranked[1:] if _tier(r.sr) == _tier(pick)]
    kind = "rigid " if _tier(pick) == 0 and len(peers) < len(ranked) - 1 else ""
    if not peers:
        return (
            f"The only {kind}setup that completed the task: {_pct(rate)} of the runs, "
            f"{pick.cycle_time_s:.1f} s cycle."
        )
    nxt = peers[0]
    if rate > nxt.completion_rate + _EPS:
        return (
            f"Highest full-task completion of the {len(peers) + 1} eligible {kind}setups ({_pct(rate)}, next "
            f"{_pct(nxt.completion_rate)})."
        )
    if pick.weakest_step[1] > nxt.weakest_step[1] + _EPS:
        return (
            f"Ties on completion ({_pct(rate)}) with a stronger weakest step ({_pct(pick.weakest_step[1])})."
        )
    a, b = pick.cycle_time_s, nxt.cycle_time_s
    cycles = _cycles(pick, nxt)
    simpler = pick.complexity.key() < nxt.complexity.key()
    if a < b - 1e-3 and simpler:
        return (
            f"Ties on completion ({_pct(rate)}) and the weakest step, with the shorter cycle ({cycles}), "
            f"and is simpler: {_simpler(pick, nxt)}."
        )
    if a < b - 1e-3:
        return f"Ties on completion ({_pct(rate)}) and the weakest step, with the shorter cycle ({cycles})."
    if simpler:
        return (
            f"Ties on completion ({_pct(rate)}), the weakest step and cycle time, and is simpler: "
            f"{_simpler(pick, nxt)}."
        )
    return f"Ties on completion, weakest step and cycle time, and is at least as simple ({_pct(rate)})."


def _why(r: _Ranked, ranked: Sequence[_Ranked], results: Sequence[SetupResult]) -> str:
    """The recommendation's reason, naming the rule that decided and the setups it compared."""
    c = r.choice
    pick = c.pick
    parts: list[str] = []
    if c.beaten:
        parts.append(_displaced(pick, c.beaten))
    if c.tie_over is not None:
        parts.append(_tie(pick, c.tie_over))
    for k, (rival, ref) in enumerate(c.rivals[:1]):
        if not parts and k == 0 and ref is pick:
            parts.append(_kept_lead(pick, rival))
        else:
            parts.append(_kept(pick, rival, ref))
    if not parts:
        parts.append(_plain(pick, ranked))
    if r.demoted is not None:
        parts.append(_demotion(pick, r.demoted))
    if _tier(pick) > 0:
        rule, what = _class_words(pick)
        if any(_tier(sr) == 0 for sr in results):
            parts.append(
                f"No rigid setup completed the task; {what} is the last resort (integrator rule {rule})."
            )
        else:
            parts.append(
                f"No rigid setup was among the finalists, so none was run; {what} ranks below any rigid "
                f"setup (integrator rule {rule})."
            )
    return " ".join(parts)


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
    baseline_id: str | None = None,
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
        "step_progress": round(_progress(sr), 4),
        "weakest_step": {"step_id": weakest, "success_rate": round(rate, 4)},
        "cycle_time_s": round(sr.cycle_time_s, 3),
        "complexity": sr.complexity.to_dict(),
        "added_pieces": added_pieces(sr),
        "payload_kg": round(payload.get(sr.setup.setup_id, sr.complexity.tool_mass_kg), 4),
        "fidelity": sr.fidelity.value,
        "failure_modes": dict(_modes(sr).most_common()),
        "stops_at": stops(sr),
        "best_single": sr.setup.setup_id == best_single_id,
        "baseline": sr.setup.setup_id == baseline_id,
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


def _versus_baseline(
    pick: SetupResult, baseline: SetupResult, tie_over: SetupResult | None = None
) -> dict[str, Any]:
    """The pick against the R01 baseline: what it adds, what it had to earn and what decided.
    ``tie_over``: the setup the same-complexity tie put the pick ahead of; when that is the baseline
    (a simpler single tool picked over the best one), the basis is ``tie`` and ``note`` says so."""
    gain = pick.completion_rate - baseline.completion_rate
    note = None
    if pick is baseline:
        basis = "baseline"
    elif tie_over is baseline:
        basis = "tie"
        note = (
            f"The same-complexity tie picks {_name(pick)} over the baseline {_name(baseline)}: "
            f"{_tie(pick, baseline)}"
        )
    elif _tier(baseline) > _tier(pick):
        basis = _class_words(baseline)[0]
    elif _fails_a_step(baseline) and gain > _EPS:
        basis = "step_floor"
    elif gain > _needed(baseline, pick) + _EPS:
        basis = "margin"
    else:
        basis = "ranking"
    return {
        "rule": "R01",
        "same_setup": pick is baseline,
        "completion_delta": round(gain, 4),
        "cycle_delta_s": round(pick.cycle_time_s - baseline.cycle_time_s, 3),
        "added_pieces": added_pieces(pick) - added_pieces(baseline),
        # a soft-arm or hand baseline ranks below the pick whatever it completes (R09, R10)
        "needed_gain": None if basis in ("R09", "R10") else round(max(0.0, _needed(baseline, pick)), 4),
        "basis": basis,
        "note": note,
    }


def _baseline_of(first: _Ranked, ranked: Sequence[_Ranked]) -> SetupResult | None:
    """The single tool printed beside the pick (``first``): when R01 kept a single, the best single
    on the base order (completion first), which is the pick unless the same-complexity tie picked a
    simpler single over it; else the best eligible single of the pick's rule class on the base order
    (the R01 baseline the pick displaced), else the best eligible single of any class (ranked below
    the pick by R09 or R10)."""
    pick = first.sr
    if pick.setup.kind is SetupKind.SINGLE:
        over = first.choice.tie_over
        return over if over is not None and over.setup.kind is SetupKind.SINGLE else pick
    singles = sorted((r.sr for r in ranked if r.sr.setup.kind is SetupKind.SINGLE), key=_base_key)
    same = [sr for sr in singles if _tier(sr) == _tier(pick)]
    return (same or singles or [None])[0]


def _no_baseline(results: Sequence[SetupResult], verdicts: Mapping[str, tuple[bool, str | None]]) -> str:
    """Why no single tool can be printed beside the pick."""
    singles = [sr for sr in results if sr.setup.kind is SetupKind.SINGLE]
    if not singles:
        return "no single tool is eligible: no single tool was among the finalists"
    why = "; ".join(
        f"{_name(sr)}: {verdicts[sr.setup.setup_id][1]}" for sr in sorted(singles, key=_closest_key)
    )
    return f"no single tool is eligible: {why}"


_RULES: Final = {
    "evidence": "results from the reference controllers are refused; assert_fair runs before ranking",
    "eligible": "completed the task at least once on the grid; no physics_invalid run",
    "rank": "completion rate, weakest step, median cycle time with changes, complexity, setup id",
    "R09_R10": "setups with a soft arm, then those with a dexterous hand, rank below every rigid setup",
    "R01": (
        "the best single tool is recommended unless it fails a step outright (weakest step under the "
        "step floor) and a more complex setup completes more, or a more complex setup completes more "
        "than the margin per added effector or changer better; every more complex setup is tested; "
        "cycle time never decides it"
    ),
    "simplicity_margin": dict(SIMPLICITY_MARGIN),
    "tie_margin": TIE_MARGIN,
    "cycle_margin": CYCLE_MARGIN,
    "simplicity": (
        "among the setups R01 lets through, of the same complexity, within the tie and cycle margins "
        "of the best, the simplest (tools, actuators, mass, changes)"
    ),
    "fidelity": "an approximation-tier winner carries the best rigid-tier eligible setup as runner-up",
    "baseline": "the R01 baseline (best eligible single tool) and the best single off-the-shelf effector are always reported",
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
        "baseline": None,
        "versus_baseline": None,
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
    ranked = _ranked(results)
    baseline = _baseline_of(ranked[0], ranked) if ranked else None
    base_id = baseline.setup.setup_id if baseline is not None else None
    out["ranking"] = [_row(r.sr, True, None, payload, best_single_id, base_id) for r in ranked]
    rest = sorted((sr for sr in results if not verdicts[sr.setup.setup_id][0]), key=_closest_key)
    out["ranking"] += [
        _row(sr, False, verdicts[sr.setup.setup_id][1], payload, best_single_id, base_id) for sr in rest
    ]
    best_single = next((sr for sr in results if sr.setup.setup_id == best_single_id), None)
    if best_single is not None:
        out["best_single"] = _card(best_single, None, payload)
    if ranked:
        pick = ranked[0].sr
        out["recommendation"] = _card(pick, _why(ranked[0], ranked, results), payload)
        if baseline is not None:
            out["baseline"] = _card(baseline, None, payload)
            out["versus_baseline"] = _versus_baseline(pick, baseline, ranked[0].choice.tie_over)
        else:
            out["baseline"] = {"setup": None, "basis": _no_baseline(results, verdicts)}
        if pick.fidelity is not FidelityTier.RIGID:
            ru = next((r.sr for r in ranked if r.sr.fidelity is FidelityTier.RIGID), None)
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
        closest = sorted(results, key=_closest_key)[:3]
        out["baseline"] = {"setup": None, "basis": _no_baseline(results, verdicts)}
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
