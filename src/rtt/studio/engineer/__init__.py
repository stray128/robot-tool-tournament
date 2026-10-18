"""The Studio's Task Engineer (track S4, product mode): a use case in words -> a ``TaskSpec``.

* :func:`plan_task`: one forced ``submit_task_plan`` call on Nemotron 3 Super, judged by the
  contract's parser and :func:`check_plan`, with at most one repair call; refusals carry a
  reason and the nearest primitive (:mod:`rtt.studio.engineer.engineer`).
* :func:`check_plan`: goals, holding, reachability, torque and force, refs, refusals
  (:mod:`rtt.studio.engineer.checks`).
* :class:`UploadRegistry` and :func:`install_upload_resolver`: ``upload:<id>`` parts through
  the existing part intake (:mod:`rtt.studio.engineer.uploads`).
* :func:`web_catalog`: the templates, primitives and limits the intake page edits with.
* :data:`GATE_CASES`: the three hero use cases and the rest of the gate set.

Command line: ``python -m rtt.studio.engineer plan --text "..." [--stl PART.stl]``, ``gate`` and
``export-web`` (see ``docs/STUDIO_ENGINEER.md``). The HTTP route is :func:`studio_router`
(:mod:`rtt.studio.engineer.api`). Every record carries
"Product mode: not part of the pre-registered experiment".
"""

from __future__ import annotations

from typing import Any

from rtt.studio.engineer.catalog import web_catalog
from rtt.studio.engineer.checks import check_plan, nearest_primitive, plan_problems
from rtt.studio.engineer.engineer import (
    COMPACT_TOOL,
    ENGINEER_MODEL,
    ENGINEER_SAMPLING,
    EngineerCache,
    PlanAttempt,
    PlanResult,
    build_messages,
    engineer_tool,
    judge_args,
    make_engineer_client,
    plan_task,
    sanitize_use_case,
)
from rtt.studio.engineer.examples import GATE_CASES, HERO_CASES, UseCase, worked_example
from rtt.studio.engineer.uploads import UploadInfo, UploadRegistry, install_upload_resolver

__all__ = [
    "COMPACT_TOOL",
    "ENGINEER_MODEL",
    "ENGINEER_SAMPLING",
    "GATE_CASES",
    "HERO_CASES",
    "EngineerCache",
    "PlanAttempt",
    "PlanResult",
    "UploadInfo",
    "UploadRegistry",
    "UseCase",
    "build_messages",
    "check_plan",
    "engineer_tool",
    "install_upload_resolver",
    "judge_args",
    "make_engineer_client",
    "nearest_primitive",
    "plan_problems",
    "plan_task",
    "sanitize_use_case",
    "studio_router",
    "web_catalog",
    "worked_example",
]


def __getattr__(name: str) -> Any:
    if name == "studio_router":  # FastAPI is imported only by the API
        from rtt.studio.engineer.api import studio_router

        return studio_router
    raise AttributeError(f"module 'rtt.studio.engineer' has no attribute {name!r}")
