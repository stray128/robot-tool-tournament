"""Arm C: Nemotron with feedback, plus the LLM proposal machinery shared with zero-shot.

Each round after round 0 a run makes one forced ``submit_designs`` tool call per
island (2 islands x 4 designs). The four slots fix the operator quota in the
schema itself: ``mutate_1``, ``mutate_2`` (one parent each), ``crossover`` (two
parents) and ``explore`` (none). Each slot carries a ``hypothesis`` before its
parameters, and the ``analysis`` field comes before every slot, so the model
analyses the round before it designs.

**Prompt** (:meth:`NemotronProposer.build_messages`, templates in
:mod:`rtt.search.prompts`): task metadata and training conditions, the parameter
table, the validator rules, the island's top-8 designs sorted ascending (integer
success %, per-family success, failure-mode counts, mass, best controller
setting), 4 informative failures, the island's validator rejections from the
last round, and a target of 1.2 x the island's best score. Only this run's
training results ever enter a prompt.

**Islands.** The shared round 0 seeds both islands. Each island's pool then holds
its own designs (FunSearch-style lineages); after round 2 each island receives the
other's best design (one migration). Diversity checks use the whole run.

**Per round** (:meth:`LLMProposer.ask`):

1. Per island: main call, then at most one *repair* call that shows the
   validator's messages for the invalid (or unparseable) slots.
2. Diversity: a valid design closer than 0.08 to any design simulated in this run
   or already accepted this round is rejected; at most one *reflection* call per
   island asks for replacements.
3. Any slot still empty is filled by ``fallback_random`` (the run's own Sobol
   stream, validity- and diversity-checked) and logged with its reason.

Invalid outputs are recorded as :class:`~rtt.search.base.Rejection` rows and cost
no simulator budget. Model calls are capped per run (default 16): main calls
take any call left, while repairs and reflections may only use calls that no
remaining main call of the run will need. Auth and
billing errors and replay-cache misses abort the run; other call failures fall
back to random designs and are logged, and recorded beside the response cache so
that a replay reproduces them instead of missing the cache. Model-authored text
quoted back into a prompt is scrubbed of held-out vocabulary first
(:func:`scrub_model_text`), so the prompt guard only fires on text this code
writes.
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.design.validate import validate
from rtt.llm.cache import CacheMiss, CacheMode, ResponseCache, canonical_json, wire_json
from rtt.llm.client import (
    CallMeta,
    LLMAuthError,
    LLMBillingError,
    LLMClient,
    LLMHTTPError,
    LLMParseError,
    LLMTransportError,
    RetryPolicy,
    TokenBucket,
)
from rtt.llm.ledger import BudgetExceededError, Ledger
from rtt.llm.providers import NEMOTRON_SUPER
from rtt.search import prompts
from rtt.search.base import (
    Evaluated,
    EvaluatorInfo,
    LLMCallRecord,
    Proposal,
    Rejection,
    RunContext,
    SearchStrategy,
    derive_seed,
)
from rtt.search.diversity import MIN_DISTANCE, Nearest, nearest
from rtt.search.random_search import DesignSampler, SamplerExhaustedError
from rtt.store.manifest import write_json_atomic

__all__ = [
    "FAILED_CALLS_DIR",
    "PROPOSER_MODEL",
    "PROPOSER_SAMPLING",
    "PROPOSER_SLOTS",
    "LLMProposer",
    "NemotronProposer",
    "PromptLeakError",
    "ProposerConfig",
    "SlotSpec",
    "assert_prompt_clean",
    "make_proposer_client",
    "prompt_spec_hashes",
    "render_distance_rule",
    "render_parameters",
    "render_rules",
    "render_task",
    "scrub_model_text",
    "submit_designs_tool",
]

PROPOSER_MODEL: Final = NEMOTRON_SUPER
#: Sampling for proposals, repairs and reflections (plan decision 10); thinking stays on.
PROPOSER_SAMPLING: Final[dict[str, float | int]] = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}

_PARAM_MEANING: Final[dict[str, str]] = {
    "finger_length_mm": "root (at the palm) to tip; longer reaches lower on tall objects but is heavier",
    "finger_thickness_mm": "root thickness along the closing direction (contact face to back)",
    "finger_width_mm": "root width across the finger; wider gives a bigger contact patch",
    "bend_angle_deg": "one bend: + tilts the part after the bend inward (toward the other finger), - outward",
    "bend_start_frac": "where the bend starts, as a fraction of the length from the root",
    "inner_face": "contact face: flat pad, concave_arc channel between two lips, or v_groove",
    "inner_face_depth_mm": "depth of the channel or groove (must be 0 for a flat face)",
    "tip_profile": "blunt end cap, rounded corners, chamfer lead-in bevel, or hook lip pointing inward",
    "tip_size_mm": "size of the tip feature (for a hook: how far the lip sticks out inward)",
    "taper_ratio": "tip / root cross-section; thickness and width shrink linearly toward the tip",
}
#: Held-out vocabulary (also catches the seed file name) and paths into a secrets folder.
_FORBIDDEN_PROMPT_TEXT = re.compile(r"held[\s_-]?out|secrets?[/\\]", re.IGNORECASE)
_MAX_ANALYSIS_CHARS: Final = 4000
_MAX_HYPOTHESIS_CHARS: Final = 1000
#: Folder inside the response cache that records calls which failed without a response.
FAILED_CALLS_DIR: Final = "failed_calls"


class PromptLeakError(RuntimeError):
    """A prompt mentioned held-out material; it is never sent."""


def assert_prompt_clean(messages: Sequence[Mapping[str, Any]]) -> None:
    """Refuse any message text that refers to the held-out suite or the secrets folder."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and _FORBIDDEN_PROMPT_TEXT.search(content):
            raise PromptLeakError("prompt refers to held-out material; refusing to send it")


def scrub_model_text(text: str) -> str:
    """Model-authored text made safe to quote in a later prompt.

    The model never sees held-out material, but its own words can still trip the
    prompt guard ("the rod is held outward ..."). Text the model wrote (hypotheses,
    earlier arguments, validator messages that echo its values) passes through here
    before it is quoted back, so only text this code writes can trigger
    :class:`PromptLeakError`.
    """
    return _FORBIDDEN_PROMPT_TEXT.sub("[removed]", text)


# --------------------------------------------------------------------------- tool schema


@dataclass(frozen=True)
class SlotSpec:
    """One design slot of the ``submit_designs`` tool."""

    name: str
    operator: str
    n_parents: int = 0


PROPOSER_SLOTS: Final[tuple[SlotSpec, ...]] = (
    SlotSpec("mutate_1", "mutate", 1),
    SlotSpec("mutate_2", "mutate", 1),
    SlotSpec("crossover", "crossover", 2),
    SlotSpec("explore", "explore", 0),
)


def _slot_schema(slot: SlotSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if slot.n_parents:
        noun = "label" if slot.n_parents == 1 else "labels"
        properties["parents"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": slot.n_parents,
            "maxItems": slot.n_parents,
            "description": f"The {noun} of the tested design{'s' if slot.n_parents > 1 else ''} this builds on.",
        }
    properties["hypothesis"] = {
        "type": "string",
        "description": "One or two sentences: what this design changes and why it should grasp better.",
    }
    for name in GEOMETRY_PARAMS:
        if name in CATEGORICAL_OPTIONS:
            properties[name] = {"type": "string", "enum": list(CATEGORICAL_OPTIONS[name])}
        else:
            lo, hi = PARAM_BOUNDS[name]
            properties[name] = {"type": "number", "minimum": lo, "maximum": hi}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def submit_designs_tool(slots: Sequence[SlotSpec]) -> dict[str, Any]:
    """The forced tool: ``analysis`` first, then one object per slot in order."""
    properties: dict[str, Any] = {
        "analysis": {
            "type": "string",
            "description": "Your analysis of the results so far and your plan, written before the designs.",
        }
    }
    for slot in slots:
        properties[slot.name] = _slot_schema(slot)
    return {
        "type": "function",
        "function": {
            "name": "submit_designs",
            "description": f"Submit {len(slots)} gripper finger designs for simulation, one per slot.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------- rendering helpers


def _num(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pct(fraction: float) -> str:
    return f"{round(100.0 * fraction)}%"


def params_json(design: FingerDesign) -> str:
    """Compact JSON of the 10 parameters with numbers rounded to 0.01 (prompt form)."""
    out: dict[str, Any] = {}
    for key, value in design.geometry_params().items():
        if isinstance(value, float):
            rounded = round(value, 2) + 0.0
            out[key] = int(rounded) if rounded.is_integer() else rounded
        else:
            out[key] = value
    return json.dumps(out, separators=(", ", ": "))


def _raw_params_json(params: Mapping[str, Any] | None) -> str:
    """Parameters exactly as the model sent them (scrubbed: values may be any text)."""
    if not params:
        return "(no parameters)"
    try:
        text = json.dumps({k: params[k] for k in GEOMETRY_PARAMS if k in params}, separators=(", ", ": "))
    except (TypeError, ValueError):
        text = repr(dict(params))[:400]
    return scrub_model_text(text)


def _k_label(info: EvaluatorInfo, k: int) -> str:
    return _num(info.controller_settings[k]) if 0 <= k < len(info.controller_settings) else str(k)


def render_task(info: EvaluatorInfo, gripper_task: GripperTask) -> str:
    """Task metadata and the training conditions (never anything held out)."""
    lift_cm = _num(info.success_lift_m * 100.0)
    heights = ", ".join(_num(h) for h in info.controller_settings)
    lines = [
        (
            f"Pick an object up from a table, lift it at least {lift_cm} cm and hold it there for "
            f"{_num(info.success_hold_s)} s, all within {_num(info.max_time_s)} s."
        ),
        (
            f"- Gripper: two parallel fingers, each sliding 0-{_num(gripper_task.stroke_mm)} mm toward the other "
            f"({_num(gripper_task.max_opening_mm)} mm between the root faces when fully open); grip force "
            f"{_num(info.grip_force_n)} N. Both fingers use your design (mirror images), 3D-printed in PLA "
            f"({_num(info.finger_density_kg_m3)} kg/m^3)."
        ),
        (
            f"- Controller (fixed for every design): approach from above, descend, close with the grip force, lift, "
            f"hold. The palm always stays at least {_num(info.palm_clearance_m * 1000.0)} mm above the object's "
            f"top. It is tried at {len(info.controller_settings)} grasp heights k: the fingertips stop {heights} of "
            "the way up from the lowest height they can reach to the object's top. Fingers that reach at least the "
            f"object height + {_num(info.palm_clearance_m * 1000.0)} mm below the palm can reach the table, so "
            "their tips stop at k x the object height; shorter fingers grasp tall objects higher up. Each design "
            "is scored at its best k."
        ),
        (
            f"- Objects: free rigid bodies, {_num(info.object_density_kg_m3)} kg/m^3, friction {_num(info.friction)}; "
            f"every episode jitters the object pose by up to {_num(info.jitter_pos_mm)} mm and "
            f"{_num(info.jitter_yaw_deg)} deg and its friction by up to {_pct(info.jitter_mu_frac)}."
        ),
        f"- Training conditions ({len(info.conditions)}):",
    ]
    for c in info.conditions:
        lines.append(
            f"  - {c.condition_id} ({c.family}): {c.label}; {_num(c.grip_width_mm)} mm across the grasp"
        )
    return "\n".join(lines)


def render_parameters() -> str:
    """Markdown table of the 10 design parameters, ranges taken from the schema."""
    rows = ["| parameter | range / options | 10% of range | meaning |", "|---|---|---|---|"]
    for name in GEOMETRY_PARAMS:
        if name in CATEGORICAL_OPTIONS:
            options, step = " / ".join(CATEGORICAL_OPTIONS[name]), "-"
        else:
            lo, hi = PARAM_BOUNDS[name]
            options, step = f"{_num(lo)} to {_num(hi)}", _num(0.1 * (hi - lo))
        rows.append(f"| {name} | {options} | {step} | {_PARAM_MEANING[name]} |")
    return "\n".join(rows)


def render_distance_rule(min_distance: float) -> str:
    """How the diversity check works, in the terms a designer needs to clear it."""
    total = min_distance * len(GEOMETRY_PARAMS)
    return (
        f"Every new design must differ from every design already tested in this run, and from your other new "
        f"designs, by a normalised distance of at least {_num(min_distance)}, or it is rejected without being "
        f"simulated. The distance is the sum over the {len(GEOMETRY_PARAMS)} parameters of |change| / range (a "
        f"changed inner_face or tip_profile counts 1), divided by {len(GEOMETRY_PARAMS)}, so the changes must add "
        f"up to at least {_num(total)}: switching inner_face or tip_profile is enough on its own; otherwise numbers "
        f"must move by {_pct(total)} of a range in total, for example three numbers by about {_pct(total / 3)} of "
        'their range each (the "10% of range" column gives the step sizes). Small tweaks of one or two numbers '
        "are always rejected."
    )


def render_rules(task: GripperTask) -> str:
    """The validator's rules with this task's constants."""
    env = task.envelope
    return "\n".join(
        [
            (
                f"- inner_face_depth_mm must be 0 for a flat face; otherwise at most {_num(task.max_depth_to_thickness)}"
                f" x finger_thickness_mm, and a v_groove also needs depth <= {_num(task.max_v_groove_depth_to_width)}"
                " x finger_width_mm."
            ),
            (
                f"- Every section must be at least {_num(task.min_section_mm)} mm thick, including the tapered tip "
                "(taper_ratio x thickness and x width) and the lips beside a groove."
            ),
            f"- A hook may protrude at most finger_thickness_mm + {_num(task.hook_extra_mm)} mm.",
            f"- One finger may weigh at most {_num(task.mass_budget_g)} g.",
            (
                f"- Fully open, the fingers must leave at least {_num(task.required_open_gap_mm)} mm between their "
                f"innermost points (widest object {_num(task.widest_training_object_mm)} mm + "
                f"{_num(task.open_gap_margin_mm)} mm); inward bends and hooks eat into this."
            ),
            (
                f"- Tool envelope: at most {_num(env.max_reach_z_mm)} mm below the mount, {_num(2 * env.half_width_x_mm)}"
                " mm wide, nothing above the mount (an inward bend close to the root lifts the back), and fully open "
                f"the back of each finger within {_num(env.half_span_y_mm)} mm of the gripper centre."
            ),
        ]
    )


def _family_vector(item: Evaluated, families: Sequence[str]) -> str:
    return ", ".join(f"{f} {_pct(item.score.per_family.get(f, 0.0))}" for f in families)


def _failure_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{mode} {n}" for mode, n in counts.items()) or "none"


def render_elites(ranked_pool: Sequence[Evaluated], info: EvaluatorInfo, n: int) -> str:
    """Top ``n`` of an already ranked pool (best first), printed worst first."""
    lines: list[str] = []
    for item in reversed(list(ranked_pool[:n])):
        s = item.score
        by_k = "/".join(_pct(v) for v in s.success_by_k)
        lines.append(
            f"{item.label} [{item.proposal.origin}, round {item.round}] score {_pct(s.success)} at "
            f"k={_k_label(info, s.best_k)} (all k: {by_k}) | {_family_vector(item, info.families)} | "
            f"failures: {_failure_counts(s.failure_modes)} | mass {s.mass_g:.1f} g"
        )
        lines.append(f"  {params_json(item.design)}")
        idea = item.proposal.hypothesis or item.design.rationale
        if idea:
            lines.append(f"  idea: {scrub_model_text(' '.join(idea.split())[:200])}")
    return "\n".join(lines) if lines else "None yet."


def render_failures(ranked_pool: Sequence[Evaluated], info: EvaluatorInfo, n: int) -> str:
    """Up to ``n`` failed (design, condition) pairs that say the most.

    Failures of the best designs come first; distinct (family, failure mode) pairs
    are preferred; each line names a design that did succeed on that condition,
    when there is one.
    """
    candidates: list[tuple[Evaluated, str]] = []
    for item in ranked_pool:
        failed = sorted(
            (cid for cid, rate in item.score.per_condition.items() if rate < 1.0),
            key=lambda cid: (item.score.per_condition[cid], cid),
        )
        candidates.extend((item, cid) for cid in failed)
    chosen: list[tuple[Evaluated, str]] = []
    seen: set[tuple[str, str]] = set()
    for item, cid in candidates:
        modes = item.score.condition_failures.get(cid, {})
        key = (info.condition(cid).family, next(iter(sorted(modes, key=lambda m: -modes[m])), "unknown"))
        if key not in seen:
            seen.add(key)
            chosen.append((item, cid))
        if len(chosen) == n:
            break
    for pair in candidates:
        if len(chosen) >= n:
            break
        if pair not in chosen:
            chosen.append(pair)
    lines = []
    for item, cid in chosen:
        cond = info.condition(cid)
        rate = item.score.per_condition[cid]
        n_eps = sum(
            1
            for e in item.score.episodes
            if e.condition_id == cid and e.controller_setting == item.score.best_k
        )
        tally = f"{round(rate * n_eps)}/{n_eps} succeeded" if n_eps else f"{_pct(rate)} succeeded"
        modes = _failure_counts(item.score.condition_failures.get(cid, {}))
        winner = next(
            (o for o in ranked_pool if o is not item and o.score.per_condition.get(cid, 0.0) >= 1.0), None
        )
        tail = f" {winner.label} succeeded on it." if winner else " No tested design succeeded on it yet."
        lines.append(
            f"- {item.label} (score {_pct(item.score.success)}) on {cid} ({cond.label}) at "
            f"k={_k_label(info, item.score.best_k)}: {tally}; {modes}.{tail}"
        )
    return "\n".join(lines) if lines else "None: every tested design succeeded on every condition."


def render_rejections(rejections: Sequence[Rejection], n: int) -> str:
    if not rejections:
        return "None."
    lines = []
    for r in rejections[:n]:
        errors = scrub_model_text("; ".join(e[:240] for e in r.errors[:3]))
        where = f"{r.slot} ({r.stage} call)" if r.slot else f"{r.stage} call"
        lines.append(f"- {where}, {r.reason}: {_raw_params_json(r.params)} -> {errors}")
    if len(rejections) > n:
        lines.append(f"- ... and {len(rejections) - n} more.")
    return "\n".join(lines)


def render_target(best: float, factor: float) -> str:
    if best <= 0.0:
        return "No tested design has succeeded on any condition yet: any success is progress."
    target = min(1.0, factor * best)
    if target >= 1.0:
        return f"Reach 100% (the current best is {_pct(best)})."
    return f"Beat {_pct(target)} ({_num(factor)} x the current best of {_pct(best)})."


# --------------------------------------------------------------------------- proposer core


@dataclass(frozen=True)
class ProposerConfig:
    """Knobs of the LLM arms (frozen in PREREG)."""

    islands: int = 2
    elites_shown: int = 8
    failures_shown: int = 4
    rejections_shown: int = 6
    target_factor: float = 1.2
    min_distance: float = MIN_DISTANCE
    migrate_after_rounds: tuple[int, ...] = (2,)
    repairs_per_call: int = 1
    reflections_per_call: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "islands": self.islands,
            "elites_shown": self.elites_shown,
            "failures_shown": self.failures_shown,
            "rejections_shown": self.rejections_shown,
            "target_factor": self.target_factor,
            "min_distance": self.min_distance,
            "migrate_after_rounds": list(self.migrate_after_rounds),
            "repairs_per_call": self.repairs_per_call,
            "reflections_per_call": self.reflections_per_call,
        }


@dataclass
class _Slot:
    island: int
    spec: SlotSpec
    proposal: Proposal | None = None
    raw: Mapping[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    reason: str | None = None
    similar: Nearest | None = None
    stage: str = "main"


@dataclass(frozen=True)
class _CallOutcome:
    args: dict[str, Any] | None
    record: LLMCallRecord
    error: str | None


class _FailureLog:
    """Model calls that failed without a response, recorded next to the response cache.

    A call that ends without a response (an HTTP error after retries, no connection,
    the spend cap) leaves nothing in the cache, so replaying the run would miss the
    cache on that request and abort. A live run therefore writes one small record per
    failed request hash to ``<cache>/failed_calls/<hash[:2]>/<hash>.json``, and a
    replay that misses the cache on such a hash reproduces the recorded failure.
    """

    def __init__(self, cache: ResponseCache | None) -> None:
        self.root = cache.root / FAILED_CALLS_DIR if cache is not None else None
        self.writes = cache is not None and cache.writes
        self.replays = cache is not None and cache.mode == "replay"

    def _path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / f"{key}.json"

    def record(self, key: str, kind: str, error: str) -> None:
        if self.root is not None and self.writes:
            write_json_atomic(self._path(key), {"request_hash": key, "kind": kind, "error": error})

    def lookup(self, key: str) -> dict[str, Any] | None:
        if self.root is None or not self.replays:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("request_hash") != key or not isinstance(data.get("error"), str):
            raise ValueError(f"{path} does not hold a failure record for request {key}")
        return data


class LLMProposer(SearchStrategy):
    """Shared machinery of the LLM arms: calls, validation, repair, diversity, fallback."""

    uses_llm = True
    origin: ClassVar[str] = ""
    slots: ClassVar[tuple[SlotSpec, ...]] = PROPOSER_SLOTS

    def __init__(self, ctx: RunContext, client: LLMClient, config: ProposerConfig | None = None) -> None:
        super().__init__(ctx)
        self.client = client
        self.config = config or ProposerConfig()
        if self.config.islands < 1:
            raise ValueError("at least one island is needed")
        self.tool = submit_designs_tool(self.slots)
        self.fallback = DesignSampler(derive_seed(ctx.run_seed, "fallback", ctx.arm), ctx.gripper_task)
        self._disabled: str | None = None
        self._messages: dict[int, list[dict[str, str]]] = {}
        self._failures = _FailureLog(client.cache)

    @property
    def disabled_reason(self) -> str | None:
        """Why this run stopped calling the model (``spend_cap``), or ``None``."""
        return self._disabled

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def build_messages(self, round_index: int, island: int) -> list[dict[str, str]]:
        """System and user messages for this island's main call."""

    # ------------------------------------------------------------------ ask

    @property
    def per_round(self) -> int:
        return self.config.islands * len(self.slots)

    def ask(self, round_index: int, n: int) -> list[Proposal]:
        if n != self.per_round:
            raise ValueError(f"{self.arm} proposes {self.per_round} designs per round, not {n}")
        if round_index < 1:
            raise ValueError("round 0 is shared and filled by the runner")
        islands = self.config.islands
        future_mains = islands * (self.ctx.budget.n_rounds - 1 - round_index)
        slots: list[_Slot] = []
        for island in range(islands):
            later_mains = future_mains + (islands - 1 - island)
            slots.extend(self._propose_island(round_index, island, later_mains))
        accepted = self._diversity_pass(round_index, slots)
        for island in range(islands):
            similar = [s for s in slots if s.island == island and s.reason == "too_similar"]
            if similar:
                self._reflect(round_index, island, slots, similar, accepted, keep=future_mains)
        for slot in slots:
            if slot.proposal is None:
                self._fill_fallback(slot, accepted)
        return [s.proposal for s in slots if s.proposal is not None]

    # ------------------------------------------------------------------ steps

    def _propose_island(self, round_index: int, island: int, later_mains: int) -> list[_Slot]:
        slots = [_Slot(island, spec) for spec in self.slots]
        messages = self.build_messages(round_index, island)
        self._messages[island] = messages
        outcome = self._call("main", round_index, island, messages, keep=0)
        self._absorb(round_index, outcome, slots, stage="main")
        for _ in range(self.config.repairs_per_call):
            broken = [s for s in slots if s.proposal is None and s.reason in ("invalid", "parse_error")]
            if not broken:
                break
            repair = self._repair_messages(messages, slots, broken, outcome)
            outcome = self._call("repair", round_index, island, repair, keep=later_mains)
            if outcome is None:
                break
            self._absorb(round_index, outcome, broken, stage="repair")
        return slots

    def _diversity_pass(self, round_index: int, slots: Sequence[_Slot]) -> list[_Slot]:
        accepted: list[_Slot] = []
        for slot in slots:
            if slot.proposal is not None:
                self._check_diversity(round_index, slot, accepted)
        return accepted

    def _reference(self, accepted: Sequence[_Slot], island: int) -> tuple[list[FingerDesign], list[str]]:
        """Designs a new proposal must keep its distance from, labelled as ``island`` sees them."""
        designs = [item.design for item in self.ctx.history]
        labels = [item.label for item in self.ctx.history]
        for other in accepted:
            assert other.proposal is not None
            designs.append(other.proposal.design)
            if other.island == island:
                labels.append(f"your {other.spec.name} in this submission")
            else:
                labels.append(f"another team's new {other.spec.name} this round")
        return designs, labels

    def _check_diversity(self, round_index: int, slot: _Slot, accepted: list[_Slot]) -> None:
        assert slot.proposal is not None
        designs, labels = self._reference(accepted, slot.island)
        near = nearest(slot.proposal.design, designs, labels)
        if near.distance >= self.config.min_distance:
            accepted.append(slot)
            return
        message = f"distance {near.distance:.3f} to {near.label} (needs >= {self.config.min_distance:g})"
        self.ctx.reject(
            Rejection(
                round=round_index,
                stage=slot.stage,
                reason="too_similar",
                errors=(message,),
                island=slot.island,
                slot=slot.spec.name,
                params=slot.raw,
                request_hash=slot.proposal.request_hash,
            )
        )
        slot.similar = near
        slot.reason = "too_similar"
        slot.errors = [message]
        slot.proposal = None

    def _reflect(
        self,
        round_index: int,
        island: int,
        slots: Sequence[_Slot],
        similar: Sequence[_Slot],
        accepted: list[_Slot],
        keep: int,
    ) -> None:
        for _ in range(self.config.reflections_per_call):
            pending = [s for s in similar if s.proposal is None and s.reason == "too_similar"]
            if not pending:
                return
            designs, _labels = self._reference(accepted, island)
            problems = []
            for s in pending:
                near = s.similar
                distance = near.distance if near else 0.0
                ref = ""
                if near is not None and near.index is not None:
                    ref = f" {near.label} is {params_json(designs[near.index])}."
                shortfall = (self.config.min_distance - distance) * len(GEOMETRY_PARAMS)
                problems.append(
                    f"- {s.spec.name}: {_raw_params_json(s.raw)} is at distance {distance:.3f} from "
                    f"{near.label if near else '?'}; the changes are short by {shortfall:.2f} of a range in "
                    f"total.{ref}"
                )
            keep_names = [s.spec.name for s in slots if s.island == island and s.proposal is not None]
            text = prompts.render(
                "diversity.md",
                min_distance=_num(self.config.min_distance),
                problems="\n".join(problems),
                distance_rule=render_distance_rule(self.config.min_distance),
                keep=", ".join(keep_names) or "none",
            )
            messages = _append_to_user(self._messages[island], text)
            outcome = self._call("reflection", round_index, island, messages, keep=keep)
            if outcome is None:
                return
            self._absorb(round_index, outcome, pending, stage="reflection")
            for s in pending:
                if s.proposal is not None:
                    self._check_diversity(round_index, s, accepted)

    def _fill_fallback(self, slot: _Slot, accepted: list[_Slot]) -> None:
        designs, _ = self._reference(accepted, slot.island)
        min_d = self.config.min_distance
        before = self.fallback.rejected
        try:
            design = self.fallback.sample(1, accept=lambda d: nearest(d, designs).distance >= min_d)[0]
        except SamplerExhaustedError:
            design = self.fallback.sample(1)[0]
        self.ctx.budget.sampler_rejections += self.fallback.rejected - before
        self.ctx.budget.fallbacks += 1
        slot.proposal = Proposal(
            design=design,
            origin="fallback_random",
            island=slot.island,
            slot=slot.spec.name,
            fallback_reason=slot.reason or "unfilled",
        )
        accepted.append(slot)

    # ------------------------------------------------------------------ model calls

    def _call(
        self, kind: str, round_index: int, island: int, messages: list[dict[str, str]], *, keep: int
    ) -> _CallOutcome | None:
        """One forced tool call, or ``None`` when no call may be made (cap or disabled)."""
        if self._disabled is not None or not self.ctx.budget.take_llm_call(keep=keep):
            return None
        assert_prompt_clean(messages)
        tag = f"{self.ctx.run_id}/round{round_index}/island{island}/{kind}"
        sample = derive_seed(self.ctx.run_seed, self.ctx.arm, round_index, island, kind) % (2**31)
        meta: CallMeta | None = None
        replayed_failure = False
        try:
            args, meta = self.client.call_tool(messages, self.tool, force=True, sample=sample, tag=tag)
            error = None
        except (LLMAuthError, LLMBillingError):
            raise
        except CacheMiss as exc:
            recorded = self._failures.lookup(exc.request_hash)
            if recorded is None:
                raise
            args, error, replayed_failure = None, str(recorded["error"]), True
            if recorded.get("kind") == "spend_cap":
                self._disabled = "spend_cap"
        except LLMParseError as exc:
            args, meta, error = None, exc.meta, f"parse_error: {exc}"
        except (LLMHTTPError, LLMTransportError) as exc:
            args, error = None, f"call_failed: {exc}"
            key = exc.request_hash if isinstance(exc, LLMHTTPError) else self.request_key(messages, sample)
            self._failures.record(key, "call_failed", error)
        except BudgetExceededError as exc:
            args, error = None, f"call_failed: spend cap reached ({exc})"
            self._disabled = "spend_cap"
            self._failures.record(self.request_key(messages, sample), "spend_cap", error)
        analysis = args.get("analysis") if isinstance(args, dict) else None
        record = LLMCallRecord(
            tag=tag,
            kind=kind,
            round=round_index,
            island=island,
            request_hash=meta.request_hash if meta else None,
            ok=error is None,
            cached=bool(meta and meta.cached) or replayed_failure,
            error=error,
            prompt_tokens=meta.prompt_tokens if meta else 0,
            completion_tokens=meta.completion_tokens if meta else 0,
            reasoning_tokens=(meta.reasoning_tokens or 0) if meta else 0,
            cost_usd=meta.cost_usd if meta else 0.0,
            latency_s=meta.latency_s if meta else 0.0,
            finish_reason=meta.finish_reason if meta else None,
            notes=meta.notes if meta else (),
            analysis=analysis[:_MAX_ANALYSIS_CHARS] if isinstance(analysis, str) else None,
        )
        self.ctx.llm_calls.append(record)
        return _CallOutcome(args=args, record=record, error=error)

    def request_key(self, messages: Sequence[Mapping[str, str]], sample: int) -> str:
        """The cache key :meth:`LLMClient.call_tool` uses for this forced call of ``self.tool``."""
        name = self.tool["function"]["name"]
        fields = {"tools": [self.tool], "tool_choice": {"type": "function", "function": {"name": name}}}
        return self.client.request_hash_for(self.client.build_payload(messages, fields), sample)

    def _absorb(
        self, round_index: int, outcome: _CallOutcome | None, slots: Sequence[_Slot], *, stage: str
    ) -> None:
        """Turn a call's arguments into proposals for ``slots`` (others are left untouched)."""
        for slot in slots:
            slot.stage = stage
            if outcome is None:
                slot.reason = slot.reason or ("spend_cap" if self._disabled else "llm_call_cap")
                continue
            if outcome.args is None:
                reason = "parse_error" if (outcome.error or "").startswith("parse_error") else "call_failed"
                slot.reason, slot.errors, slot.raw = reason, [outcome.error or reason], None
                self.ctx.reject(
                    Rejection(
                        round=round_index,
                        stage=stage,
                        reason=reason,
                        errors=tuple(slot.errors),
                        island=slot.island,
                        slot=slot.spec.name,
                        request_hash=outcome.record.request_hash,
                    )
                )
                continue
            raw = outcome.args.get(slot.spec.name)
            design, errors, meta = _parse_slot(raw, slot.spec, self.ctx.gripper_task)
            slot.raw = {k: raw[k] for k in GEOMETRY_PARAMS if k in raw} if isinstance(raw, Mapping) else None
            if design is None:
                slot.reason, slot.errors, slot.proposal = "invalid", errors, None
                self.ctx.reject(
                    Rejection(
                        round=round_index,
                        stage=stage,
                        reason="invalid",
                        errors=tuple(errors),
                        island=slot.island,
                        slot=slot.spec.name,
                        params=slot.raw,
                        request_hash=outcome.record.request_hash,
                    )
                )
                continue
            slot.reason, slot.errors = None, []
            slot.proposal = Proposal(
                design=design,
                origin=self.origin,
                operator=slot.spec.operator,
                island=slot.island,
                slot=slot.spec.name,
                parents=meta["parents"],
                hypothesis=meta["hypothesis"],
                via=stage,
                request_hash=outcome.record.request_hash,
            )

    def _repair_messages(
        self,
        messages: list[dict[str, str]],
        slots: Sequence[_Slot],
        broken: Sequence[_Slot],
        outcome: _CallOutcome | None,
    ) -> list[dict[str, str]]:
        if outcome is not None and outcome.args is None:
            problems = f"- The response did not contain a usable submit_designs call ({outcome.error})."
            previous = "(no usable call)"
            keep = "none"
        else:
            problems = "\n".join(f"- {s.spec.name}: " + "; ".join(s.errors) for s in broken)
            previous = (
                wire_json(outcome.args) if outcome is not None and outcome.args is not None else "(none)"
            )
            kept = [s.spec.name for s in slots if s.proposal is not None]
            keep = ", ".join(kept) or "none"
        text = prompts.render(
            "repair.md", problems=scrub_model_text(problems), previous=scrub_model_text(previous), keep=keep
        )
        return _append_to_user(messages, text)


def _append_to_user(messages: Sequence[Mapping[str, str]], text: str) -> list[dict[str, str]]:
    """A copy of ``messages`` whose final user message has ``text`` appended (single-turn follow-up)."""
    out = [dict(m) for m in messages]
    if not out or out[-1].get("role") != "user":
        raise ValueError("the last message must be the user prompt")
    out[-1]["content"] = f"{out[-1]['content']}\n\n{text}"
    return out


def _parse_slot(
    raw: Any, spec: SlotSpec, task: GripperTask
) -> tuple[FingerDesign | None, list[str], dict[str, Any]]:
    """Validate one slot's object: (design or None, error messages, metadata)."""
    meta: dict[str, Any] = {"parents": (), "hypothesis": None}
    if not isinstance(raw, Mapping):
        return None, [f"slot '{spec.name}' is missing or is not an object"], meta
    hypothesis = raw.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis.strip():
        meta["hypothesis"] = " ".join(hypothesis.split())[:_MAX_HYPOTHESIS_CHARS]
    parents = raw.get("parents")
    if isinstance(parents, list):
        meta["parents"] = tuple(str(p) for p in parents if isinstance(p, str | int))[:2]
    mapping: dict[str, Any] = {k: raw[k] for k in GEOMETRY_PARAMS if k in raw}
    if meta["hypothesis"]:
        mapping["rationale"] = meta["hypothesis"]
    result = validate(mapping, task)
    if not result.ok or result.design is None:
        return None, list(result.errors), meta
    return result.design, [], meta


# --------------------------------------------------------------------------- arm C


class NemotronProposer(LLMProposer):
    """Arm C: proposals conditioned on this run's own training results, in 2 islands."""

    arm = "nemotron"
    origin = "nemotron"
    slots = PROPOSER_SLOTS

    def __init__(self, ctx: RunContext, client: LLMClient, config: ProposerConfig | None = None) -> None:
        super().__init__(ctx, client, config)
        self._pools: list[list[int]] = [[] for _ in range(self.config.islands)]

    def tell(self, evaluated: Sequence[Evaluated]) -> None:
        for item in evaluated:
            islands = (
                range(self.config.islands)
                if item.round == 0 or item.proposal.island is None
                else [item.proposal.island]
            )
            for island in islands:
                if item.index not in self._pools[island]:
                    self._pools[island].append(item.index)
        rounds = {item.round for item in evaluated}
        if rounds & set(self.config.migrate_after_rounds) and self.config.islands > 1:
            bests = [self.ranked_pool(i)[0].index for i in range(self.config.islands)]
            for island in range(self.config.islands):
                for other, best in enumerate(bests):
                    if other != island and best not in self._pools[island]:
                        self._pools[island].append(best)

    def pool(self, island: int) -> list[Evaluated]:
        return [self.ctx.history[i - 1] for i in self._pools[island]]

    def ranked_pool(self, island: int) -> list[Evaluated]:
        return self.ctx.history.ranked(self.pool(island))

    def build_messages(self, round_index: int, island: int) -> list[dict[str, str]]:
        cfg = self.config
        info = self.ctx.info
        ranked = self.ranked_pool(island)
        if not ranked:
            raise RuntimeError("the island pool is empty: round 0 was not told to the strategy")
        last_round = [r for r in self.ctx.rejections if r.round == round_index - 1 and r.island == island]
        system = prompts.render("proposer_system.md", n_islands=cfg.islands)
        user = prompts.render(
            "proposer_user.md",
            task=render_task(info, self.ctx.gripper_task),
            parameters=render_parameters(),
            rules=render_rules(self.ctx.gripper_task),
            n_elites=min(cfg.elites_shown, len(ranked)),
            n_pool=len(ranked),
            n_conditions=len(info.conditions),
            elites=render_elites(ranked, info, cfg.elites_shown),
            failures=render_failures(ranked[: cfg.elites_shown], info, cfg.failures_shown),
            rejections=render_rejections(last_round, cfg.rejections_shown),
            target=render_target(ranked[0].score.success, cfg.target_factor),
            round=round_index,
            last_round=self.ctx.budget.n_rounds - 1,
            distance_rule=render_distance_rule(cfg.min_distance),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- client + hashes


def make_proposer_client(
    cache_dir: str | Path | None,
    *,
    cache_mode: CacheMode = "readwrite",
    ledger_path: str | Path | None = None,
    budget_usd: float | None = None,
    model: str = PROPOSER_MODEL,
    transport: Any | None = None,
    timeout_s: float = 300.0,
    limiter: TokenBucket | None = None,
) -> LLMClient:
    """The proposer client with the pre-registered sampling settings.

    ``cache_mode="replay"`` re-runs an arm from its recorded cache with zero API
    calls (and no API key). ``limiter`` defaults to the process-wide token bucket
    for the model (75 % of its tokens-per-minute limit).
    """
    cache = ResponseCache(cache_dir, mode=cache_mode) if cache_dir is not None else None
    return LLMClient(
        model,
        temperature=float(PROPOSER_SAMPLING["temperature"]),
        top_p=float(PROPOSER_SAMPLING["top_p"]),
        max_tokens=int(PROPOSER_SAMPLING["max_tokens"]),
        cache=cache,
        ledger=Ledger(ledger_path, budget_usd=budget_usd),
        transport=transport,
        timeout_s=timeout_s,
        retry=RetryPolicy(max_retries=4, base_delay_s=2.0),
        limiter=limiter,
    )


def prompt_spec_hashes() -> dict[str, str]:
    """Hashes of everything that shapes a prompt besides the run's own data (for PREREG)."""
    import hashlib

    def h(obj: object) -> str:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()

    from rtt.search.zeroshot import ZEROSHOT_SLOTS

    return {
        **{f"template:{k}": v for k, v in prompts.template_hashes().items()},
        "tool:proposer": h(submit_designs_tool(PROPOSER_SLOTS)),
        "tool:zeroshot": h(submit_designs_tool(ZEROSHOT_SLOTS)),
        "parameter_table": h(render_parameters()),
    }
