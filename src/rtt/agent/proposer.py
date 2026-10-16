"""``propose_designs``: Nemotron 3 Super proposes finger designs for an uploaded part.

One forced ``submit_designs`` tool call per request (``analysis`` first, then one slot
per design, each with a ``hypothesis`` before its 10 parameters; the schema bounds every
number and enumerates every category). The prompt carries the part as simulated, the
parameter table and the validator rules shared with the search arms, and every design
this session has tested with its physics result. Each returned design must then pass
the geometry validator for this part (:func:`rtt.design.validate.validate`) and the
diversity rule of the search arms (:data:`rtt.search.diversity.MIN_DISTANCE` to every
known design); failures are reported back to the agent, never simulated. There is no
repair call: the agent decides whether to ask again, within its tool budget.

This is product mode. Nothing here feeds the pre-registered experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from rtt.agent.backends import PartInfo
from rtt.design.distance import design_distance
from rtt.design.schema import GEOMETRY_PARAMS, FingerDesign, GripperTask
from rtt.design.validate import validate
from rtt.llm.client import CallMeta, LLMClient
from rtt.llm.providers import NEMOTRON_SUPER
from rtt.search.diversity import MIN_DISTANCE
from rtt.search.nemotron import (
    SlotSpec,
    assert_prompt_clean,
    render_distance_rule,
    render_parameters,
    render_rules,
    scrub_model_text,
    submit_designs_tool,
)

__all__ = [
    "PROPOSER_SAMPLING",
    "DesignProposer",
    "NemotronDesignProposer",
    "ProposalBatch",
    "ProposedDesign",
    "SimulatedDesign",
    "check_proposal",
]

#: Same sampling as the search proposer (thinking on).
PROPOSER_SAMPLING: Final[dict[str, float | int]] = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}
_MAX_HYPOTHESIS: Final = 400


@dataclass(frozen=True)
class SimulatedDesign:
    """A design this session has simulated, as the proposer is shown it."""

    design: FingerDesign
    label: str
    success: float | None = None
    failure_modes: Mapping[str, int] = field(default_factory=dict)
    certificate_score: float | None = None


@dataclass(frozen=True)
class ProposedDesign:
    design: FingerDesign
    hypothesis: str


@dataclass(frozen=True)
class ProposalBatch:
    """What one ``propose_designs`` call produced."""

    designs: tuple[ProposedDesign, ...]
    rejections: tuple[dict[str, Any], ...]
    analysis: str
    calls: tuple[CallMeta, ...] = ()


class DesignProposer(Protocol):
    """Anything that turns a part plus history into candidate designs."""

    def propose(
        self,
        part: PartInfo,
        n: int,
        *,
        gripper_task: GripperTask,
        history: Sequence[SimulatedDesign],
        known: Sequence[FingerDesign],
    ) -> ProposalBatch: ...


def check_proposal(
    params: Mapping[str, Any],
    hypothesis: str,
    gripper_task: GripperTask,
    reference: Sequence[FingerDesign],
) -> tuple[FingerDesign | None, list[str]]:
    """Validate one proposed design; returns ``(design, [])`` or ``(None, reasons)``."""
    geometry = {k: params.get(k) for k in GEOMETRY_PARAMS}
    result = validate(geometry, gripper_task)
    if not result.ok or result.design is None:
        return None, list(result.errors) or ["invalid design"]
    design = result.design.model_copy(update={"rationale": hypothesis[:1000] or None})
    for other in reference:
        dist = design_distance(design, other)
        if dist < MIN_DISTANCE:
            reason = (
                f"too similar to design {other.design_id()} (distance {dist:.3f} < {MIN_DISTANCE}); "
                "change a category or move parameters further"
            )
            return None, [reason]
    return design, []


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{round(100.0 * value)}%"


def _params_line(design: FingerDesign) -> str:
    parts = []
    for key in GEOMETRY_PARAMS:
        value = getattr(design, key)
        parts.append(f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(parts)


def render_part(part: PartInfo) -> str:
    """The part as the simulator sees it (no file names beyond the user's own label)."""
    x, y, z = part.size_mm
    box = f"{x:g} mm along the finger width (x), {y:g} mm across the jaw (y), {z:g} mm tall (z)"
    mass = f"{part.mass_g:g} g (solid plastic, {part.density_kg_m3:g} kg/m^3)"
    lines = [
        f"- Name: {scrub_model_text(part.name)[:60]}",
        f"- Bounding box in the grasp frame: {box}.",
        f"- The fingers close on {part.grip_width_mm:g} mm; the gripper opens {part.max_opening_mm:g} mm.",
        f"- Mass {mass}; {part.n_convex_pieces} convex pieces in simulation.",
        f"- Test conditions ({part.episodes_per_design} episodes per design, every grasp height):",
    ]
    lines += [f"  - {scrub_model_text(c)}" for c in part.conditions]
    return "\n".join(lines)


def render_history(history: Sequence[SimulatedDesign]) -> str:
    if not history:
        return "No designs have been simulated on this part yet."
    rows = []
    for item in history:
        modes = ", ".join(f"{k} x{v}" for k, v in item.failure_modes.items()) or "none"
        cert = f", certificate {_pct(item.certificate_score)}" if item.certificate_score is not None else ""
        rows.append(
            f"- {item.label}: success {_pct(item.success)}{cert}; failures: {modes}\n  {_params_line(item.design)}"
        )
    return "\n".join(rows)


SYSTEM_PROMPT: Final = (
    "You design 3D-printed fingers for an open-source parallel gripper (two jaws, 50 N grip). "
    "A physics simulator scores every design: the gripper must lift the part 100 mm and hold it for "
    "2 s. Propose designs that are likely to hold THIS part, each one a distinct idea. Think about "
    "the part's size, mass and shape, and about why earlier designs failed."
)


class NemotronDesignProposer:
    """Nemotron 3 Super with a forced ``submit_designs`` call (see the module docstring)."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @classmethod
    def create(
        cls, ledger: Any, *, cache: Any | None = None, transport: Any | None = None
    ) -> NemotronDesignProposer:
        return cls(
            LLMClient(
                NEMOTRON_SUPER,
                temperature=float(PROPOSER_SAMPLING["temperature"]),
                top_p=float(PROPOSER_SAMPLING["top_p"]),
                max_tokens=int(PROPOSER_SAMPLING["max_tokens"]),
                cache=cache,
                ledger=ledger,
                transport=transport,
                timeout_s=180.0,
            )
        )

    def build_messages(
        self, part: PartInfo, n: int, gripper_task: GripperTask, history: Sequence[SimulatedDesign]
    ) -> list[dict[str, str]]:
        plural = "s" if n > 1 else ""
        task = (
            f"Write a short analysis, then propose {n} new design{plural} in the slots of submit_designs. "
            "Each needs a one- or two-sentence hypothesis saying what it changes and why it should hold "
            "this part."
        )
        sections = [
            ("The part", render_part(part)),
            ("Design parameters", render_parameters()),
            (
                "Validator rules (a design that breaks one is rejected without simulation)",
                render_rules(gripper_task),
            ),
            ("Diversity", render_distance_rule(MIN_DISTANCE)),
            ("Designs already simulated on this part", render_history(history)),
            ("Your task", task),
        ]
        user = "\n\n".join(f"## {title}\n{body}" for title, body in sections)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def propose(
        self,
        part: PartInfo,
        n: int,
        *,
        gripper_task: GripperTask,
        history: Sequence[SimulatedDesign],
        known: Sequence[FingerDesign],
    ) -> ProposalBatch:
        if not 1 <= n <= 8:
            raise ValueError("n must be in 1..8")
        messages = self.build_messages(part, n, gripper_task, history)
        assert_prompt_clean(messages)
        slots = [SlotSpec(f"design_{i + 1}", "explore", 0) for i in range(n)]
        args, meta = self.client.call_tool(
            messages, submit_designs_tool(slots), force=True, tag="agent.propose"
        )
        accepted: list[ProposedDesign] = []
        rejections: list[dict[str, Any]] = []
        reference = list(known)
        for slot in slots:
            raw = args.get(slot.name)
            if not isinstance(raw, Mapping):
                rejections.append({"slot": slot.name, "reasons": ["missing from the model's answer"]})
                continue
            hypothesis = str(raw.get("hypothesis") or "")[:_MAX_HYPOTHESIS]
            design, reasons = check_proposal(raw, hypothesis, gripper_task, reference)
            if design is None:
                rejections.append({"slot": slot.name, "reasons": reasons})
                continue
            accepted.append(ProposedDesign(design, hypothesis))
            reference.append(design)
        analysis = str(args.get("analysis") or "")[:2000]
        return ProposalBatch(tuple(accepted), tuple(rejections), analysis, (meta,))
