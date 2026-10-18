"""Track S4's gate: the gate descriptions through the Task Engineer, and the manifest of what came back.

:func:`run_gate` plans every :data:`~rtt.studio.engineer.examples.GATE_CASES` description (or a
subset) with one client, in order, and :func:`gate_manifest` summarises the results:

* **plans**: the three hero use cases and the five other realistic ones must give a valid
  ``TaskSpec`` after at most one repair, in at least 90 % of the cases;
* **refusals**: the deliberately unsupported ones must be refused, with a reason and at least
  one nearest primitive.

Live (``--live``) the run calls Token Factory under a hard cap on everything recorded in its
ledger; without it the run replays the recorded cache with zero API calls, and the test suite
checks that the replay reproduces the recorded manifest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.llm.cache import wire_json
from rtt.llm.client import LLMClient
from rtt.studio.engineer import prompts
from rtt.studio.engineer.engineer import (
    COMPACT_TOOL,
    ENGINEER_MODEL,
    ENGINEER_SAMPLING,
    PlanResult,
    engineer_tool,
    plan_task,
    tool_sha256,
)
from rtt.studio.engineer.examples import GATE_CASES, UseCase
from rtt.studio.types import CONTRACT_VERSION

__all__ = ["GATE_VALID_RATE", "RECORDED_GATE", "case_row", "gate_manifest", "run_gate"]

#: Share of the plan cases that must give a valid TaskSpec after at most one repair.
GATE_VALID_RATE: Final = 0.9
#: The recorded live gate run (scrubbed cache, ledger, manifest).
RECORDED_GATE: Final = Path(__file__).with_name("recorded") / "gate-2026-10-18-s2"


def run_gate(
    client: LLMClient, cases: Sequence[UseCase] = GATE_CASES, *, compact: bool = COMPACT_TOOL
) -> list[tuple[UseCase, PlanResult]]:
    """Plan every case in order with ``client`` (one repair each)."""
    return [
        (case, plan_task(case.text, client=client, compact=compact, tag=f"gate/{case.case_id}"))
        for case in cases
    ]


def case_row(case: UseCase, result: PlanResult) -> dict[str, Any]:
    """One case of the manifest: the verdict, the steps produced and how it got there."""
    spec = result.spec
    steps = [
        {"id": s.id, "primitive": s.primitive.value, "object": s.object, "target": s.target}
        for s in (spec.steps if spec is not None else ())
    ]
    nearest = [u.nearest_primitive.value for u in spec.unsupported if u.nearest_primitive] if spec else []
    if case.expect == "plan":
        passed = result.status == "valid"
    else:
        passed = result.status == "refused" and spec is not None and bool(spec.refusal) and bool(nearest)
    return {
        "case_id": case.case_id,
        "setting": case.setting,
        "title": case.title,
        "expect": case.expect,
        "text": case.text,
        "status": result.status,
        "passed": passed,
        "first_try": result.first_try,
        "repaired": result.repaired,
        "calls": len(result.attempts),
        "steps": steps,
        "n_objects": len(spec.objects) if spec is not None else 0,
        "refusal": spec.refusal if spec is not None else None,
        "nearest_primitives": nearest,
        "task_id": spec.task_id if spec is not None and spec.steps else None,
        "problems": list(result.problems),
        "first_problems": list(result.attempts[0].problems) if result.attempts else [],
        "warnings": list(result.warnings),
        "notes": list(result.notes),
        "cost_usd": round(result.cost_usd, 6),
        "attempts": [a.to_dict() for a in result.attempts],
        "plan": result.to_dict(),
    }


def gate_manifest(
    results: Sequence[tuple[UseCase, PlanResult]],
    *,
    mode: str,
    budget_usd: float | None = None,
    compact: bool = COMPACT_TOOL,
) -> dict[str, Any]:
    """The gate's manifest (module docstring); ``mode`` is ``live`` or ``replay``."""
    rows = [case_row(c, r) for c, r in results]
    plans = [r for r in rows if r["expect"] == "plan"]
    refusals = [r for r in rows if r["expect"] == "refusal"]
    valid = [r for r in plans if r["passed"]]
    rate = len(valid) / len(plans) if plans else math.nan
    attempts = [a for r in rows for a in r["attempts"]]
    live = [a for a in attempts if not a["cached"]]
    tool = engineer_tool(compact=compact)
    summary = {
        "plans": {
            "n": len(plans),
            "valid": len(valid),
            "valid_first_try": sum(1 for r in valid if r["first_try"]),
            "valid_after_repair": sum(1 for r in valid if r["repaired"]),
            "valid_rate": round(rate, 4) if plans else None,
        },
        "refusals": {
            "n": len(refusals),
            "refused_with_nearest": sum(1 for r in refusals if r["passed"]),
        },
        "all_ok": sum(1 for r in rows if r["passed"]),
        "n": len(rows),
    }
    passed = (not plans or rate >= GATE_VALID_RATE) and all(r["passed"] for r in refusals)
    return {
        "format": "rtt-studio-engineer-gate",
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "contract": CONTRACT_VERSION,
        "model": ENGINEER_MODEL,
        "sampling": dict(ENGINEER_SAMPLING),
        "mode": mode,
        "budget_usd": budget_usd,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tool": {"compact": compact, "sha256": tool_sha256(compact=compact), "bytes": len(wire_json(tool))},
        "template_hashes": prompts.template_hashes(),
        "gate": {"valid_rate_needed": GATE_VALID_RATE, "passed": passed},
        "summary": summary,
        "llm": {
            "calls": len(attempts),
            "live_calls": len(live),
            "cost_usd": round(sum(a["cost_usd"] for a in attempts), 6),
            "prompt_tokens": sum(a["prompt_tokens"] for a in attempts),
            "completion_tokens": sum(a["completion_tokens"] for a in attempts),
        },
        "cases": rows,
    }
