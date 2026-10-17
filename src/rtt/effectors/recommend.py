"""The recommendation: certificates in, one effector out (``docs/EFFECTORS.md`` section 5).

For each requirement R (``held`` and ``registered``, always both; the headline is the requested
one, ``registered`` by default):

1. **Eligible:** certified; the certificate's nominal test passes R; no ``physics_invalid``
   episode (reference, screen or certificate); the nominal test is not ``physics_unreliable``.
2. **Rank:** certificate score for R; then the smaller of the two placement tolerances for R; then
   screen success for R (then ``effector_id``, so the order is total).
3. **Simplicity rule:** among candidates within :data:`SIMPLICITY_MARGIN` of the best remaining
   certificate score with the same tolerance, prefer fewer actuators, then lower effector mass,
   then higher fidelity (core, rigid, approximation).
4. **Fidelity rule:** an ``approximation`` winner (suction, soft arm) always comes with the best
   ``core`` or ``rigid`` candidate as ``runner_up_rigid``, and its card says
   "approximation: verify on hardware".
5. **No winner:** nothing eligible gives ``winner = null``, the reason and the closest candidates
   with their failure modes. A pick is never forced.

Everything here is a pure function of the tournament's candidates; the model that explains a
recommendation (the Grasp Engineer) never sets it.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from rtt.effectors.registry import get_family
from rtt.effectors.types import CONTRACT_VERSION, PRODUCT_MODE_LABEL, Requirement

if TYPE_CHECKING:
    from rtt.effectors.tournament import Candidate, TournamentResult

__all__ = [
    "APPROXIMATION_CAVEAT",
    "SIMPLICITY_MARGIN",
    "Entry",
    "card_note",
    "eligibility",
    "rank_entries",
    "recommend",
    "recommend_for",
]

#: Certificate scores closer than this count as a tie for the simplicity rule.
SIMPLICITY_MARGIN: Final = 0.05
APPROXIMATION_CAVEAT: Final = "approximation: verify on hardware"
_EPS: Final = 1e-9
_FIDELITY_RANK: Final = {"core": 0, "rigid": 1, "approximation": 2}
_RIGID: Final = frozenset({"core", "rigid"})


def card_note(family: str, model: str) -> str | None:
    """The note every card, ranking row and lane must print beside this effector, if its family
    declares one (``card_note(model)``, e.g. a catalog gripper's low grip at its published
    settings); ``None`` otherwise."""
    try:
        impl = get_family(family)
    except (LookupError, ValueError):
        return None
    fn = getattr(impl, "card_note", None)
    return fn(model) if callable(fn) else None


@dataclass(frozen=True)
class Entry:
    """One candidate as the recommendation sees it (built from a tournament ``Candidate``)."""

    candidate: Candidate

    @property
    def id(self) -> str:
        return self.candidate.effector_id

    @property
    def fidelity(self) -> str:
        return self.candidate.fidelity

    @property
    def n_actuators(self) -> int:
        n = self.candidate.n_actuators
        return int(n) if n is not None else 10**6

    @property
    def mass(self) -> float:
        m = self.candidate.mass_kg
        return float(m) if m is not None and math.isfinite(m) else math.inf

    def score(self, req: Requirement) -> float:
        cert = self.candidate.certificate
        if cert is None:
            return -1.0
        return float(cert.held_score if req is Requirement.HELD else cert.registered_score)

    def tolerances(self, req: Requirement) -> dict[str, float | None]:
        cert = self.candidate.certificate
        if cert is None:
            return {"x": None, "y": None}
        tol = cert.held_tolerance_mm if req is Requirement.HELD else cert.registered_tolerance_mm
        return {"x": tol.get("x"), "y": tol.get("y")}

    def tolerance(self, req: Requirement) -> float:
        """The smaller of the two placement tolerances (-1 when the nominal grasp fails)."""
        vals = list(self.tolerances(req).values())
        if any(v is None for v in vals):
            return -1.0
        return float(min(v for v in vals if v is not None))

    def screen_rate(self, req: Requirement) -> float:
        return self.candidate.rate(req)

    def base_key(self, req: Requirement) -> tuple[Any, ...]:
        return (-round(self.score(req), 9), -self.tolerance(req), -round(self.screen_rate(req), 9), self.id)

    def simplicity_key(self, req: Requirement) -> tuple[Any, ...]:
        return (self.n_actuators, self.mass, _FIDELITY_RANK.get(self.fidelity, 3), self.base_key(req))


def _all_failure_modes(cand: Candidate) -> list[str]:
    modes = [r.failure_mode for r in cand.episodes() if r.failure_mode]
    if cand.certificate is not None:
        modes += [t.failure_mode for t in cand.certificate.tests if t.failure_mode]
    return modes


def eligibility(entry: Entry, req: Requirement) -> tuple[bool, str | None]:
    """Whether ``entry`` may be recommended for ``req``, and why not."""
    cand = entry.candidate
    cert = cand.certificate
    if cert is None:
        return False, "not certified (not a finalist)" if cand.status == "screened" else _status_reason(cand)
    nominal = next(t for t in cert.tests if t.name == "nominal")
    passes = nominal.held if req is Requirement.HELD else nominal.registered
    if not passes:
        how = "held" if req is Requirement.HELD else "registered"
        return (
            False,
            f"the certificate's nominal grasp is not {how} ({nominal.failure_mode or 'not engaged'})",
        )
    if "physics_invalid" in _all_failure_modes(cand):
        return False, "an episode reset the simulator (physics_invalid)"
    if nominal.failure_mode == "physics_unreliable":
        return False, "the nominal test penetrated the part by more than 1 mm (physics_unreliable)"
    return True, None


def _status_reason(cand: Candidate) -> str:
    return {
        "invalid_build": f"cannot be built for this part: {cand.reason}",
        "contract_error": f"broke the effector contract: {cand.reason}",
        "not_sampled": "not sampled into the group's screen budget",
    }.get(cand.status, cand.status)


def rank_entries(entries: Iterable[Entry], req: Requirement) -> list[tuple[Entry, Entry | None]]:
    """Eligible entries in recommendation order, each with the entry the simplicity rule put it
    ahead of (``None`` when it was first on the base order anyway)."""
    remaining = sorted(entries, key=lambda e: e.base_key(req))
    out: list[tuple[Entry, Entry | None]] = []
    while remaining:
        top = remaining[0]
        close = [
            e
            for e in remaining
            if top.score(req) - e.score(req) <= SIMPLICITY_MARGIN + _EPS
            and abs(e.tolerance(req) - top.tolerance(req)) <= _EPS
        ]
        pick = min(close, key=lambda e: e.simplicity_key(req))
        out.append((pick, top if pick is not top else None))
        remaining.remove(pick)
    return out


def _fmt_tol(tol: Mapping[str, float | None]) -> str:
    x, y = tol.get("x"), tol.get("y")
    if x is None or y is None:
        return "no placement tolerance"
    if x == y:
        return f"{x:g} mm placement error on both axes"
    return f"{x:g} mm placement error across and {y:g} mm along the closing axis"


def _why(
    pick: Entry, over: Entry | None, ranked: Sequence[tuple[Entry, Entry | None]], req: Requirement
) -> str:
    n = len(ranked)
    score = pick.score(req)
    tol = _fmt_tol(pick.tolerances(req))
    if over is not None:
        simpler = []
        if pick.n_actuators != over.n_actuators:
            simpler.append(
                f"{pick.n_actuators} actuator{'s' if pick.n_actuators != 1 else ''} against {over.n_actuators}"
            )
        elif pick.mass != over.mass:
            simpler.append(f"{1000.0 * pick.mass:.0f} g against {1000.0 * over.mass:.0f} g")
        else:
            simpler.append(f"{pick.fidelity} fidelity against {over.fidelity}")
        if abs(over.score(req) - score) <= _EPS:
            return (
                f"Ties with {over.candidate.spec.display_name} on {req.value} certificate score ({score:.2f}) and "
                f"tolerance ({tol}), and is simpler: {simpler[0]}."
            )
        return (
            f"Within {SIMPLICITY_MARGIN:g} of the top {req.value} certificate score ({score:.2f} against "
            f"{over.score(req):.2f}, {over.candidate.spec.display_name}) with the same tolerance, and simpler: "
            f"{simpler[0]}."
        )
    if n == 1:
        return f"The only eligible candidate: {req.value} certificate score {score:.2f}, tolerating {tol}."
    runner = ranked[1][0]
    if score > runner.score(req) + _EPS:
        return (
            f"Highest {req.value} certificate score of the {n} eligible candidates ({score:.2f}, next "
            f"{runner.score(req):.2f}), tolerating {tol}."
        )
    if pick.tolerance(req) > runner.tolerance(req) + _EPS:
        return f"Ties on {req.value} certificate score ({score:.2f}) and tolerates more: {tol}."
    if pick.screen_rate(req) > runner.screen_rate(req) + _EPS:
        return (
            f"Ties on {req.value} certificate score ({score:.2f}) and tolerance, with the higher screen "
            f"success ({100.0 * pick.screen_rate(req):.0f}% against {100.0 * runner.screen_rate(req):.0f}%)."
        )
    # equal score, tolerance and screen success: first on the base order, so the simplicity rule
    # kept it (it is at least as simple as every tied candidate)
    name = runner.candidate.spec.display_name
    if pick.n_actuators != runner.n_actuators:
        simpler = (
            f"{pick.n_actuators} actuator{'s' if pick.n_actuators != 1 else ''} against {runner.n_actuators}"
        )
    elif pick.mass != runner.mass:
        simpler = f"{1000.0 * pick.mass:.0f} g against {1000.0 * runner.mass:.0f} g"
    elif pick.fidelity != runner.fidelity:
        simpler = f"{pick.fidelity} fidelity against {runner.fidelity}"
    else:
        return (
            f"Ties with {name} on {req.value} certificate score ({score:.2f}), tolerance ({tol}) and screen "
            f"success ({100.0 * pick.screen_rate(req):.0f}%); the order falls to the effector id."
        )
    return (
        f"Ties with {name} on {req.value} certificate score ({score:.2f}), tolerance ({tol}) and screen "
        f"success ({100.0 * pick.screen_rate(req):.0f}%), and is simpler: {simpler}."
    )


def _row(entry: Entry, req: Requirement, eligible: bool, reason: str | None) -> dict[str, Any]:
    cand = entry.candidate
    s = cand.spec
    cert = cand.certificate
    rates = cand.screen_rates() if cand.status == "screened" else None
    return {
        "effector_id": s.effector_id,
        "display_name": s.display_name,
        "family": s.family.value,
        "model": s.model,
        "primitive": s.primitive,
        "group": s.group.value,
        "fidelity": s.fidelity.value,
        "origin": s.origin,
        "status": cand.status,
        "certified": cert is not None,
        "eligible": eligible,
        "reason": reason,
        "score": round(entry.score(req), 4) if cert is not None else None,
        "tolerance_mm": entry.tolerances(req),
        "screen_rate": round(rates[req.value], 4) if rates else None,
        "screen": rates,
        "n_actuators": cand.n_actuators,
        "mass_kg": round(entry.mass, 4) if math.isfinite(entry.mass) else None,
        "actuation": dict(cand.actuation),
        "reference": cand.reference_source,
        "note": card_note(s.family.value, s.model),
    }


def _card(entry: Entry, req: Requirement, why: str | None) -> dict[str, Any]:
    cand = entry.candidate
    rates = cand.screen_rates()
    out = {
        "effector": cand.spec.to_dict(),
        "certificate": cand.certificate.to_dict() if cand.certificate is not None else None,
        "screen": {
            "n": rates["n"],
            "held": round(rates["held"], 4),
            "registered": round(rates["registered"], 4),
        },
        "why": why,
        "fidelity": cand.fidelity,
        "n_actuators": cand.n_actuators,
        "mass_kg": round(entry.mass, 4) if math.isfinite(entry.mass) else None,
        "actuation": dict(cand.actuation),
        "score": round(entry.score(req), 4),
        "tolerance_mm": entry.tolerances(req),
        "caveat": APPROXIMATION_CAVEAT if cand.fidelity == "approximation" else None,
        "note": card_note(cand.spec.family.value, cand.spec.model),
    }
    return out


def _runner_up_rigid(
    ranked: Sequence[tuple[Entry, Entry | None]], entries: Sequence[Entry], req: Requirement
) -> tuple[Entry | None, str]:
    """The best core or rigid candidate: eligible first, else certified, else screened."""
    for e, _ in ranked:
        if e.fidelity in _RIGID:
            return e, "eligible"
    certified = [e for e in entries if e.fidelity in _RIGID and e.candidate.certificate is not None]
    if certified:
        return min(certified, key=lambda e: e.base_key(req)), "certified, not eligible"
    screened = [e for e in entries if e.fidelity in _RIGID and e.candidate.status == "screened"]
    if screened:
        return min(screened, key=lambda e: (-e.screen_rate(req), e.id)), "screened only"
    return None, "no core or rigid candidate was run"


def recommend_for(candidates: Sequence[Candidate], req: Requirement) -> dict[str, Any]:
    """The recommendation block for one requirement (winner, runner-up, ranking, or no winner)."""
    entries = [Entry(c) for c in candidates]
    verdicts = {e.id: eligibility(e, req) for e in entries}
    ranked = rank_entries([e for e in entries if verdicts[e.id][0]], req)
    ranking = [_row(e, req, True, None) for e, _ in ranked]
    rest = sorted(
        (e for e in entries if not verdicts[e.id][0]),
        key=lambda e: (e.candidate.certificate is None, e.base_key(req)),
    )
    ranking += [_row(e, req, False, verdicts[e.id][1]) for e in rest]
    out: dict[str, Any] = {
        "requirement": req.value,
        "winner": None,
        "runner_up_rigid": None,
        "ranking": ranking,
    }
    if ranked:
        pick, over = ranked[0]
        out["winner"] = _card(pick, req, _why(pick, over, ranked, req))
        if pick.fidelity == "approximation":
            ru, basis = _runner_up_rigid(ranked, entries, req)
            out["runner_up_rigid"] = (
                {**_card(ru, req, None), "basis": basis}
                if ru is not None
                else {"effector": None, "basis": basis}
            )
        out["no_winner"] = None
    else:
        closest = sorted(
            (e for e in entries if e.candidate.status == "screened"),
            key=lambda e: (e.candidate.certificate is None, -e.score(req), -e.screen_rate(req), e.id),
        )[:3]
        out["no_winner"] = {
            "reason": (
                f"no candidate is eligible for {req.value}: none was certified with a {req.value} nominal "
                "grasp and clean physics"
                if entries
                else "no candidate was run"
            ),
            "closest": [
                {
                    **_row(e, req, False, verdicts[e.id][1]),
                    "failure_modes": dict(collections.Counter(_all_failure_modes(e.candidate)).most_common()),
                }
                for e in closest
            ],
        }
    return out


def recommend(result: TournamentResult) -> dict[str, Any]:
    """``recommendation.json`` for a finished tournament (the shape is fixed in ``docs/EFFECTORS.md``)."""
    cands = [c for c in result.candidates() if c.status != "pending"]
    head = result.requirement
    blocks = {r.value: recommend_for(cands, r) for r in Requirement}
    main = blocks[head.value]
    other = {
        r.value: (
            {
                "effector_id": blocks[r.value]["winner"]["effector"]["effector_id"],
                "display_name": blocks[r.value]["winner"]["effector"]["display_name"],
                "fidelity": blocks[r.value]["winner"]["fidelity"],
            }
            if blocks[r.value]["winner"] is not None
            else None
        )
        for r in Requirement
        if r is not head
    }
    # the ranking is in recommendation order, so a group's best is its first screened row
    order = {row["effector_id"]: i for i, row in enumerate(main["ranking"])}
    by_group: dict[str, str | None] = {}
    for g, st in result.groups.items():
        ids = [c.effector_id for c in st.candidates if c.status == "screened" and c.effector_id in order]
        by_group[g.value] = min(ids, key=order.__getitem__) if ids else None
    return {
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "part_id": result.part.name,
        "requirement": head.value,
        "winner": main["winner"],
        "runner_up_rigid": main["runner_up_rigid"],
        "no_winner": main.get("no_winner"),
        "alternatives": other,
        "ranking": main["ranking"],
        "by_requirement": blocks,
        "by_group": by_group,
        "groups": {
            g.value: {
                "status": st.status,
                "reason": st.reason,
                "finalists": [c.effector_id for c in st.finalists],
            }
            for g, st in result.groups.items()
        },
        "budget": {g.value: st.budget() for g, st in result.groups.items()},
        "conditions": {
            "suite": result.suite,
            "condition_ids": [c.condition_id for c in result.conditions],
            "seeds": {c.condition_id: [int(s) for s in c.seeds] for c in result.conditions},
            "screen_episodes_per_candidate": sum(len(c.seeds) for c in result.conditions),
            "certificate_tests": 18,
        },
        "rules": {
            "eligible": "certified; nominal test passes the requirement; no physics_invalid episode; nominal not physics_unreliable",
            "rank": "certificate score, then the smaller placement tolerance, then screen success",
            "simplicity_margin": SIMPLICITY_MARGIN,
            "simplicity": "fewer actuators, then lower mass, then higher fidelity (core, rigid, approximation)",
            "fidelity": "an approximation winner carries the best core or rigid candidate as runner-up",
        },
        "provenance": result.provenance(),
    }
