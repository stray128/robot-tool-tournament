"""Server-side checks of a plan submitted to ``POST /runs`` (and again inside the job).

The client's plan is never trusted: it must be a ``TaskSpec`` (its ``to_dict`` form, as
``POST /plan`` returns it) that passes the Task Engineer's own parser and checks
(:func:`rtt.studio.engineer.engineer.judge_args`, the current rule set) with no problem, is not a
refusal, has steps, names no uploaded part (the hosted runner has no part upload), and fits the
runner's size limits. The plan that goes to the job is the parsed one, re-serialised, never the
client's bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from rtt.runner.config import RunnerConfig
from rtt.studio.engineer.engineer import judge_args, normalise_nulls
from rtt.studio.types import CONTRACT_VERSION, TaskSpec

__all__ = ["PLAN_KEYS", "check_run_plan"]

#: The fields of ``TaskSpec.to_dict``; the first five are derived and recomputed here.
DERIVED: Final = ("contract", "label", "task_id", "source", "fidelity")
PLAN_KEYS: Final = frozenset(
    {
        *DERIVED,
        "title",
        "description",
        "objects",
        "steps",
        "assumptions",
        "clarifications",
        "unsupported",
        "refusal",
    }
)
_MAX_PROBLEMS: Final = 20


def check_run_plan(cfg: RunnerConfig, plan: Any) -> tuple[TaskSpec | None, list[str]]:
    """``(spec, [])`` for a plan the runner will run, else ``(None, problems)``."""
    if not isinstance(plan, Mapping):
        return None, ["plan: must be a JSON object (the plan POST /plan returned)"]
    size = len(json.dumps(plan, separators=(",", ":")).encode())
    if size > cfg.max_plan_bytes:
        return None, [f"plan: {size} bytes is over the runner's limit of {cfg.max_plan_bytes}"]
    problems: list[str] = []
    unknown = sorted(str(k) for k in plan if k not in PLAN_KEYS)
    if unknown:
        problems.append(f"plan: unknown fields {unknown[:5]}")
    if plan.get("contract") not in (None, CONTRACT_VERSION):
        problems.append(f"plan.contract: expected {CONTRACT_VERSION}")
    objects, steps = plan.get("objects"), plan.get("steps")
    if not isinstance(objects, list) or not isinstance(steps, list):
        return None, [*problems, "plan: objects and steps must be lists"]
    if len(objects) > cfg.max_objects:
        problems.append(f"plan.objects: {len(objects)} objects; the runner takes at most {cfg.max_objects}")
    if len(steps) > cfg.max_steps:
        problems.append(f"plan.steps: {len(steps)} steps; the runner takes at most {cfg.max_steps}")
    if plan.get("refusal") is not None or not steps:
        problems.append("plan: a refused plan, or one without steps, has nothing to run")
    for i, o in enumerate(objects):
        ref = o.get("ref") if isinstance(o, Mapping) else None
        if isinstance(ref, str) and ref.startswith("upload:"):
            problems.append(f"plan.objects[{i}].ref: uploaded parts cannot run on the hosted runner")
    if problems:
        return None, problems[:_MAX_PROBLEMS]
    args = {k: plan[k] for k in plan if k not in DERIVED}
    args, _ = normalise_nulls(args)
    spec, found, _warnings = judge_args(args, source="manual")
    if found or spec is None:
        return None, (found or ["plan: does not parse into a task"])[:_MAX_PROBLEMS]
    if spec.refusal is not None or not spec.steps:
        return None, ["plan: a refused plan, or one without steps, has nothing to run"]
    return spec, []
