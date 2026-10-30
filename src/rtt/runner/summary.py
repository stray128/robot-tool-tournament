"""The compact summary of one Studio search report (``summary.json`` next to ``report.json``).

What ``GET /runs/{id}`` returns as ``result``: the pick (label, kind, tools), its completion
(``k`` of ``n`` runs), why it was picked, the baseline (the recommender's best single tool and its
completion), how many candidates and setups were ruled out on the way, a per-step table (step,
tool, success rate) and the code version. When no setup completed the task there is no pick; the
closest setup fills the table and ``why`` says why there is no recommendation.

Pure: reads the dict :meth:`rtt.studio.search.TournamentReport.to_dict` writes, imports nothing
heavy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

__all__ = ["SUMMARY_FORMAT", "summarize_report"]

SUMMARY_FORMAT: Final = "effector-lab-run-summary"
PRODUCT_MODE_LABEL: Final = "Product mode: not part of the pre-registered experiment"


def _rate(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def _completion(card: Mapping[str, Any]) -> dict[str, int]:
    n = int(card.get("n_runs") or 0)
    rate = _rate(card.get("completion_rate")) or 0.0
    return {"k": round(rate * n), "n": n}


def _tools(card: Mapping[str, Any]) -> list[str]:
    return [str(t.get("display_name") if isinstance(t, Mapping) else t) for t in card.get("tools") or []]


def _setup_of(report: Mapping[str, Any], card: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(card.get("setup"), Mapping):
        return card["setup"]
    sid = card.get("setup_id")
    for f in report.get("finalists") or []:
        if (f.get("setup") or {}).get("setup_id") == sid:
            return f["setup"]
    return {}


def _steps(report: Mapping[str, Any], card: Mapping[str, Any]) -> list[dict[str, Any]]:
    setup = _setup_of(report, card)
    tools = setup.get("tools") or []
    assignment = setup.get("assignment") or {}
    success = card.get("step_success") or {}
    rows = []
    for s in (report.get("task") or {}).get("steps") or []:
        sid = s.get("id")
        index = assignment.get(sid)
        tool = tools[index].get("display_name") if isinstance(index, int) and index < len(tools) else None
        rows.append({"step": sid, "tool": tool, "success_rate": _rate(success.get(sid))})
    return rows


def _brief(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "setup_id": card.get("setup_id"),
        "label": card.get("label"),
        "kind": card.get("kind"),
        "tools": _tools(card),
    }


def _ruled_out(report: Mapping[str, Any]) -> dict[str, int]:
    screen = report.get("screen") or {}
    pruned: Sequence[Mapping[str, Any]] = screen.get("pruned") or []
    whole = {p.get("effector_id") for p in pruned if p.get("step_id") is None}
    some = {p.get("effector_id") for p in pruned if p.get("step_id") is not None} - whole
    candidates = len(screen.get("candidates") or [])
    ranking = (report.get("recommendation") or {}).get("ranking") or []
    return {
        "candidates": candidates,
        "pruned": len(whole),
        "pruned_for_some_steps": len(some),
        "screened": max(0, candidates - len(whole)),
        "setups_rejected": len((report.get("combinations") or {}).get("rejected") or []),
        "finalists": len(report.get("finalists") or []),
        "finalists_not_eligible": sum(1 for r in ranking if not r.get("eligible")),
    }


def summarize_report(report: Mapping[str, Any], *, run_id: str, code_version: str) -> dict[str, Any]:
    """The summary (module docstring) of ``report``."""
    rec = report.get("recommendation") or {}
    pick = rec.get("recommendation")
    no_winner = rec.get("no_winner") or {}
    closest = (no_winner.get("closest") or [None])[0]
    shown = pick or closest
    base = rec.get("baseline") or {}
    if isinstance(base, Mapping) and base.get("setup_id"):
        baseline: dict[str, Any] | None = {**_brief(base), "completion": _completion(base), "basis": "R01"}
    elif isinstance(rec.get("best_single"), Mapping):
        best = rec["best_single"]
        baseline = {**_brief(best), "completion": _completion(best), "basis": "best_single"}
    else:
        baseline = None
    why = pick.get("why") if pick else (no_winner.get("reason") or rec.get("reason"))
    task = report.get("task") or {}
    return {
        "format": SUMMARY_FORMAT,
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "run_id": run_id,
        "status": "done",
        "task_id": report.get("task_id"),
        "title": task.get("title"),
        "outcome": "recommended" if pick else "no_recommendation",
        "evidence": rec.get("evidence"),
        "pick": _brief(pick) if pick else None,
        "closest": _brief(closest) if (closest and not pick) else None,
        "completion": _completion(shown) if shown else None,
        "why": why,
        "baseline": baseline,
        "baseline_note": None if baseline else (base.get("basis") if isinstance(base, Mapping) else None),
        "ruled_out": _ruled_out(report),
        "steps": _steps(report, shown) if shown else [],
        "conditions": len(report.get("conditions") or []),
        "code_version": code_version,
        "timings": {k: v for k, v in (report.get("timings") or {}).items() if isinstance(v, int | float)},
    }
