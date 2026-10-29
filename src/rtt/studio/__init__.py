"""Effector Lab Studio: from a use case to a buildable end-effector setup (product mode).

**Product mode, not pre-registered.** The pre-registered finger experiment and the v1 per-part
effector layer (``rtt-effectors/1``, :mod:`rtt.effectors`) are untouched; the Studio reuses them
read-only and labels everything with :data:`PRODUCT_MODE_LABEL`. See ``docs/STUDIO.md``.

Frozen contract ``rtt-effectors/2`` (track S1):

* :mod:`rtt.studio.types`: task specs, steps, object templates, setups, results, build packs and
  the Task Engineer's JSON schema.
* :mod:`rtt.studio.scene`: the Studio scene (core wrist plus a limited pitch and a continuous
  roll; single, multi-tool head and tool-change tool stacks; template objects).
* :mod:`rtt.studio.plan`: the primitive controller and success predicate protocols and registry.
* :mod:`rtt.studio.episode`: segments, carried world state, the tool-change hook, full-task runs.
* :mod:`rtt.studio.fairness`: identical wrist, constants, conditions and seeds across candidates.
* :mod:`rtt.studio.templates`: the template builder protocol and registry (part and bin builders).
* :mod:`rtt.studio.reference`: trivial reference controllers (smoke runs and tests only).

Importing this package loads only the types; the MuJoCo modules load on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import Any

from rtt.studio.types import (
    CONTRACT_VERSION,
    PRIMITIVES,
    PRODUCT_MODE_LABEL,
    STUDIO_WRIST,
    BracketSpec,
    BuildPackManifest,
    FidelityTier,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    ObjectTemplate,
    PrimitiveKind,
    Setup,
    SetupKind,
    SetupResult,
    Step,
    StepConstraints,
    StepOutcome,
    SuccessCriterion,
    TaskResult,
    TaskSpec,
    TaskSpecError,
    TemplateKind,
    ToolChanger,
    get_template,
    submit_task_plan_tool,
    task_plan_json_schema,
    task_spec_from_tool_args,
)

_LAZY = {
    "build_studio_scene": "rtt.studio.scene",
    "StudioScene": "rtt.studio.scene",
    "StudioCondition": "rtt.studio.episode",
    "StudioJob": "rtt.studio.episode",
    "run_task": "rtt.studio.episode",
    "run_task_jobs": "rtt.studio.episode",
    "assert_fair": "rtt.studio.fairness",
    "fairness_problems": "rtt.studio.fairness",
    "get_primitive": "rtt.studio.plan",
    "register_primitive": "rtt.studio.plan",
    "get_template_builder": "rtt.studio.templates",
    "register_template_builder": "rtt.studio.templates",
}

__all__ = [
    "CONTRACT_VERSION",
    "PRIMITIVES",
    "PRODUCT_MODE_LABEL",
    "STUDIO_WRIST",
    "BracketSpec",
    "BuildPackManifest",
    "FidelityTier",
    "HeadMount",
    "ObjectInstance",
    "ObjectPose",
    "ObjectTemplate",
    "PrimitiveKind",
    "Setup",
    "SetupKind",
    "SetupResult",
    "Step",
    "StepConstraints",
    "StepOutcome",
    "StudioCondition",
    "StudioJob",
    "StudioScene",
    "SuccessCriterion",
    "TaskResult",
    "TaskSpec",
    "TaskSpecError",
    "TemplateKind",
    "ToolChanger",
    "assert_fair",
    "build_studio_scene",
    "fairness_problems",
    "get_primitive",
    "get_template",
    "get_template_builder",
    "register_primitive",
    "register_template_builder",
    "run_task",
    "run_task_jobs",
    "submit_task_plan_tool",
    "task_plan_json_schema",
    "task_spec_from_tool_args",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'rtt.studio' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
