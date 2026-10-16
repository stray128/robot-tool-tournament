"""Tool definitions for the Grasp Engineer and strict validation of the model's arguments.

Each tool has a JSON schema (sent to the model as an OpenAI function tool) and a pydantic
model with the same fields (used to validate what the model sends back). Unknown fields,
missing fields, wrong types, malformed IDs and out-of-range counts are all rejected
before a tool runs; ``tests/test_agent_tools.py`` checks that the two descriptions agree.

IDs have fixed shapes so the model cannot smuggle paths or commands through them:

* part: ``part-`` + 12 hex digits (assigned at upload);
* design: 12 hex digits (:meth:`rtt.design.schema.FingerDesign.design_id`);
* job: ``local-`` + 12 hex digits, or a Nebius ``aijob-...`` ID;
* run: ``run-`` + 12 hex digits (the results of one finished job).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

__all__ = [
    "DESIGN_ID_PATTERN",
    "JOB_ID_PATTERN",
    "PART_ID_PATTERN",
    "RUN_ID_PATTERN",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "ArgsError",
    "ToolSpec",
    "openai_tools",
    "validate_args",
]

PART_ID_PATTERN: Final = r"^part-[0-9a-f]{12}$"
DESIGN_ID_PATTERN: Final = r"^[0-9a-f]{12}$"
JOB_ID_PATTERN: Final = r"^(local-[0-9a-f]{12}|aijob-[a-z0-9]{6,40})$"
RUN_ID_PATTERN: Final = r"^run-[0-9a-f]{12}$"

PartId = Annotated[str, StringConstraints(pattern=PART_ID_PATTERN)]
DesignId = Annotated[str, StringConstraints(pattern=DESIGN_ID_PATTERN)]
JobId = Annotated[str, StringConstraints(pattern=JOB_ID_PATTERN)]
RunId = Annotated[str, StringConstraints(pattern=RUN_ID_PATTERN)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InspectPartArgs(_Args):
    part_id: PartId


class ProposeDesignsArgs(_Args):
    part_id: PartId
    n: int = Field(ge=1, le=4)


class LaunchSimJobArgs(_Args):
    part_id: PartId
    designs: list[DesignId] = Field(min_length=1, max_length=8)
    target: Literal["local", "nebius"] = "local"

    @field_validator("designs")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("design IDs must be unique")
        return value


class PollJobArgs(_Args):
    job_id: JobId


class FetchResultsArgs(_Args):
    job_id: JobId


class StressTestArgs(_Args):
    design_id: DesignId
    part_id: PartId


class ExportPrintPackArgs(_Args):
    run_id: RunId
    design_id: DesignId | None = None


class CostLedgerArgs(_Args):
    pass


def _id(pattern: str, description: str) -> dict[str, Any]:
    return {"type": "string", "pattern": pattern, "description": description}


_PART = _id(PART_ID_PATTERN, "Part ID, e.g. part-0123456789ab.")
_DESIGN = _id(DESIGN_ID_PATTERN, "12-hex-digit design ID from propose_designs or the baselines list.")
_JOB = _id(JOB_ID_PATTERN, "Job ID returned by launch_sim_job.")


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its name, what the model is told, its schema and argument model.

    ``may_spend`` marks tools that can cost money (they need an unlocked session).
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    args_model: type[_Args]
    may_spend: bool = False

    def openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


def _object(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        "inspect_part",
        "Read the uploaded part: size in mm (x along the finger width, y across the jaw, z height), "
        "width the fingers must close on, mass, whether it fits the gripper's opening, the test "
        "conditions it will be simulated under, and the baseline finger designs you can compare against.",
        _object({"part_id": _PART}, ["part_id"]),
        InspectPartArgs,
    ),
    ToolSpec(
        "propose_designs",
        "Ask the design model for n new finger designs for this part (1-4). It sees the part and every "
        "result this session has produced. Returns design IDs, parameters and a one-line hypothesis "
        "each; invalid designs are rejected by the geometry validator and listed with reasons. Costs "
        "a model call.",
        _object(
            {"part_id": _PART, "n": {"type": "integer", "minimum": 1, "maximum": 4, "description": "1-4."}},
            ["part_id", "n"],
        ),
        ProposeDesignsArgs,
        may_spend=True,
    ),
    ToolSpec(
        "launch_sim_job",
        "Simulate up to 8 designs grasping the part in MuJoCo physics (lift 100 mm, hold 2 s, every "
        "test condition and grasp height). target='local' runs on this machine at no cost; "
        "target='nebius' runs a Nebius Serverless AI Job and costs money. At most 2 jobs may be in "
        "flight. Returns a job ID.",
        _object(
            {
                "part_id": _PART,
                "designs": {"type": "array", "items": _DESIGN, "minItems": 1, "maxItems": 8},
                "target": {"type": "string", "enum": ["local", "nebius"]},
            },
            ["part_id", "designs", "target"],
        ),
        LaunchSimJobArgs,
        may_spend=True,
    ),
    ToolSpec(
        "poll_job",
        "Wait (up to about two minutes) for a job to finish and return its state: QUEUED, RUNNING, "
        "COMPLETED, FAILED or CANCELLED. Poll once and only poll again if it is still running.",
        _object({"job_id": _JOB}, ["job_id"]),
        PollJobArgs,
    ),
    ToolSpec(
        "fetch_results",
        "Results of a COMPLETED job: per design, the physics success rate at its best grasp height "
        "(over 'episodes' trials at that height; 'episodes_all_heights' counts every height), the "
        "rate at each height, failure modes, finger mass and the pass/fail verdict from the "
        "simulator. Returns a run ID for export_print_pack.",
        _object({"job_id": _JOB}, ["job_id"]),
        FetchResultsArgs,
    ),
    ToolSpec(
        "stress_test",
        "Robustness certificate for one design on this part: the part placed 0, 1, 2 and 3 mm off "
        "where the gripper aims (both axes), a shake during the hold and a sideways push. Returns "
        "pass rates per category and the placement error tolerated. Runs locally, no cost.",
        _object({"design_id": _DESIGN, "part_id": _PART}, ["design_id", "part_id"]),
        StressTestArgs,
    ),
    ToolSpec(
        "export_print_pack",
        "Build the print pack for a run's best design (or design_id if given): left and right finger "
        "STLs in mm, the design parameters, the physics results and the robustness certificate.",
        _object(
            {"run_id": _id(RUN_ID_PATTERN, "Run ID from fetch_results."), "design_id": _DESIGN},
            ["run_id"],
        ),
        ExportPrintPackArgs,
    ),
    ToolSpec(
        "cost_ledger",
        "What this session has spent so far (model calls and compute), the caps, and the tool calls left.",
        _object({}, []),
        CostLedgerArgs,
    ),
)

TOOL_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in TOOL_SPECS)
_BY_NAME: Final[dict[str, ToolSpec]] = {spec.name: spec for spec in TOOL_SPECS}


class ArgsError(ValueError):
    """The model's arguments for a tool did not validate. ``problems`` lists each issue."""

    def __init__(self, tool: str, problems: list[str]) -> None:
        self.tool = tool
        self.problems = problems
        super().__init__(f"invalid arguments for {tool}: " + "; ".join(problems))


def tool_spec(name: str) -> ToolSpec:
    """The spec of tool ``name`` (``KeyError`` for an unknown tool)."""
    return _BY_NAME[name]


def openai_tools() -> list[dict[str, Any]]:
    """Every tool in OpenAI function-tool format, in a fixed order."""
    return [spec.openai() for spec in TOOL_SPECS]


def validate_args(name: str, args: Any) -> _Args:
    """Validate ``args`` for tool ``name``.

    Raises:
        ArgsError: unknown tool, arguments not an object, or any field problem.
    """
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ArgsError(name, [f"unknown tool {name!r}; tools are {', '.join(TOOL_NAMES)}"])
    if not isinstance(args, Mapping):
        detail = getattr(args, "error", None)  # the agent passes its JSON parse error along
        raise ArgsError(name, [str(detail) if detail else "arguments must be a JSON object"])
    try:
        return spec.args_model.model_validate(dict(args))
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err.get("loc", ())) or "arguments"
            problems.append(f"{where}: {err.get('msg', 'invalid')}")
        raise ArgsError(name, problems) from None
