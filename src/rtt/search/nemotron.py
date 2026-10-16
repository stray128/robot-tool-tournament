"""Arm C: Nemotron with feedback, plus the LLM proposal machinery shared with zero-shot.

Each round after round 0 a run makes one forced ``submit_designs`` tool call per
island (2 islands x 4 designs). The four slots fix the operator quota in the
schema itself: ``mutate_1``, ``mutate_2`` (one parent each), ``crossover`` (two
parents) and ``explore`` (none). Arm C's calls also return ``reserve_1`` ...
``reserve_4`` (:attr:`ProposerConfig.reserves`): extra designs, each declaring its
own operator and parents, that replace rejected slots in order. The ``analysis``
field comes before every slot, and inside each slot the ``hypothesis``, the
``changes`` (every parameter changed from the parent, in points, with the total)
and ``same_cell_distances`` (distances in points to the tested designs in the same
cell) come before the parameters, so the model analyses the round and checks the
distance rule before it writes a design. Design notes and the evidence behind
them are in ``docs/PROPOSER.md``.

**Prompt** (:meth:`NemotronProposer.build_messages`, templates in
:mod:`rtt.search.prompts`): task metadata and training conditions, the parameter
table, the validator rules, the distance rule in *points* (each numeric range is
0-100 points, a changed ``inner_face`` or ``tip_profile`` is 100; see
:mod:`rtt.search.diversity`), a compact table of **every** design tested in this
run grouped by cell (``inner_face`` + ``tip_profile``) with its score and point
positions, plus the cells nobody has tried; the island's top-8 designs sorted
ascending (integer success %, per-family success, failure-mode counts, mass, best
controller setting, nearest tested designs in points) with the crossover-ready
pairs among them; 4 informative failures; the island's rejections from the last
round; a target of 1.2 x the island's best score; and the operator rules. Only
this run's training results ever enter a prompt.

**Islands.** The shared round 0 seeds both islands. Each island's elite pool then
holds its own designs (FunSearch-style lineages); after round 2 each island
receives the other's best design (one migration). The tested-design table and
the diversity checks use the whole run. Islands are asked in turn, and an
island's prompt lists the designs earlier islands already claimed this round
(rows ``t<team>.<slot>``), so two islands do not submit the same idea.

**Per island, per round** (:meth:`LLMProposer.ask`):

1. Main call. Each valid design goes through the diversity check: closer than
   0.08 (80 points) to any design simulated in this run or already accepted this
   round is ``too_similar``; an ``explore`` design must also sit in an untried
   cell or at least 0.2 from every design in its cell (``not_novel``).
2. Holes (invalid or rejected slots) take the call's reserves in order; a reserve
   is validated and diversity-checked when it is tried, and every rejected
   reserve is logged like any other rejection. Unused reserves are dropped.
3. At most one *repair* call for invalid (or unparseable) slots that no reserve
   could fill, showing the validator's messages, and at most one *reflection*
   call for near-copies that are left, showing each one's exact nearest
   neighbours, per-parameter differences in points and shortfall. Each
   follow-up call brings fresh reserves; those its own slots do not need are
   offered to every hole that is left before the next call or the random fill.
4. Any slot still empty is filled by ``fallback_random`` (the run's own Sobol
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
import math
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
    params_key,
)
from rtt.search.diversity import (
    EXPLORE_MIN_DISTANCE,
    MIN_DISTANCE,
    NUMERIC_PARAMS,
    POINTS_PER_RANGE,
    Nearest,
    Screened,
    all_cells,
    cell_name,
    cell_of,
    distance_points,
    explore_check,
    nearest,
    neighbours,
    points_of,
    screen,
    to_points,
)
from rtt.search.random_search import SamplerExhaustedError
from rtt.search.space import V1, DesignSpace
from rtt.store.manifest import write_json_atomic

__all__ = [
    "DIVERSITY_REASONS",
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
    "min_points",
    "points_row",
    "prompt_spec_hashes",
    "render_distance_rule",
    "render_operators",
    "render_parameters",
    "render_points_rule",
    "render_rules",
    "render_task",
    "render_tested",
    "scrub_model_text",
    "submit_designs_tool",
]

PROPOSER_MODEL: Final = NEMOTRON_SUPER
#: Sampling for proposals, repairs and reflections (plan decision 10); thinking stays on.
#: max_tokens leaves room for reasoning plus 8 designs (4 slots and 4 reserves) per call.
PROPOSER_SAMPLING: Final[dict[str, float | int]] = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 16384}

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
_MAX_DECLARED_CHARS: Final = 400
#: Parent labels are model text quoted back into later prompts (table rows, reflection), so they are bounded.
_MAX_PARENT_CHARS: Final = 32
#: Folder inside the response cache that records calls which failed without a response.
FAILED_CALLS_DIR: Final = "failed_calls"
#: Rejection reasons the diversity pass gives (both are answered by the reflection call).
DIVERSITY_REASONS: Final[tuple[str, ...]] = ("too_similar", "not_novel")
#: Short column names of the 8 numeric parameters in the points table.
_POINT_COLUMNS: Final[dict[str, str]] = {
    "finger_length_mm": "len",
    "finger_thickness_mm": "thick",
    "finger_width_mm": "width",
    "bend_angle_deg": "bend",
    "bend_start_frac": "bend_at",
    "inner_face_depth_mm": "depth",
    "tip_size_mm": "tip_size",
    "taper_ratio": "taper",
}
#: Per-slot fields that make the model check the distance rule before it writes parameters.
_EXPLAIN_FIELDS: Final[dict[str, str]] = {
    "changes": (
        "Each parameter you change from the parent (explore: from the nearest tested design in its cell) as "
        "name old -> new with the difference in points, then the total in points."
    ),
    "same_cell_distances": (
        "Distances in points to the tested designs in this design's cell (at least the 3 nearest), nearest "
        'first, e.g. "d25 95, d06 150, d11 210", or "untried cell".'
    ),
}
#: Operators a reserve design may declare.
RESERVE_OPERATORS: Final[tuple[str, ...]] = ("mutate", "crossover", "explore")


class PromptLeakError(RuntimeError):
    """A prompt mentioned held-out material; it is never sent."""


def assert_prompt_clean(
    messages: Sequence[Mapping[str, Any]], pattern: re.Pattern[str] = _FORBIDDEN_PROMPT_TEXT
) -> None:
    """Refuse any message text that refers to the held-out suite or the secrets folder.

    v2 arms pass a stricter ``pattern`` that also refuses the unseen held-out part families.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and pattern.search(content):
            raise PromptLeakError("prompt refers to held-out material; refusing to send it")


def scrub_model_text(text: str, pattern: re.Pattern[str] = _FORBIDDEN_PROMPT_TEXT) -> str:
    """Model-authored text made safe to quote in a later prompt.

    The model never sees held-out material, but its own words can still trip the
    prompt guard ("the rod is held outward ..."). Text the model wrote (hypotheses,
    earlier arguments, validator messages that echo its values) passes through here
    before it is quoted back, so only text this code writes can trigger
    :class:`PromptLeakError`.
    """
    return pattern.sub("[removed]", text)


def _json_safe(value: Any) -> Any:
    """``value`` with every non-finite float (and any non-JSON value) replaced by its text.

    Tool arguments are parsed with :func:`json.loads`, which accepts ``NaN`` and
    ``Infinity``, but the run's logs, cache keys and follow-up prompts are written
    with ``allow_nan=False``. The validator already rejects such a design; this keeps
    its rejection loggable instead of aborting the run. Finite values, key order and
    types are unchanged, so ordinary arguments serialise byte for byte as before.
    """
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | int | float):
        return value
    return repr(value)


def _raw_geometry(raw: Any) -> dict[str, Any] | None:
    """A slot's geometry parameters exactly as the model sent them, made JSON-safe (``None`` if not an object).

    This is what a :class:`~rtt.search.base.Rejection` logs as ``params`` and what
    reserve de-duplication keys on, so it must serialise with ``allow_nan=False``.
    """
    if not isinstance(raw, Mapping):
        return None
    return _json_safe({k: raw[k] for k in GEOMETRY_PARAMS if k in raw})


def _parent_labels(parents: Any, pattern: re.Pattern[str] = _FORBIDDEN_PROMPT_TEXT) -> tuple[str, ...]:
    """The first two parent labels the model gave, scrubbed, single-line and bounded."""
    if not isinstance(parents, list):
        return ()
    labels = (" ".join(str(p).split())[:_MAX_PARENT_CHARS] for p in parents if isinstance(p, str | int))
    return tuple(scrub_model_text(label, pattern) for label in labels)[:2]


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


def reserve_names(n: int) -> list[str]:
    """Names of ``n`` reserve slots: ``reserve_1`` ... ``reserve_n``."""
    return [f"reserve_{i}" for i in range(1, n + 1)]


def _slot_schema(
    slot: SlotSpec, explain: bool, *, reserve: bool = False, space: DesignSpace | None = None
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    if reserve:
        properties["operator"] = {
            "type": "string",
            "enum": list(RESERVE_OPERATORS),
            "description": "How this reserve design was made: mutate (1 parent), crossover (2) or explore (none).",
        }
        properties["parents"] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 2,
            "description": "Labels of the tested designs this builds on (empty for explore).",
        }
    elif slot.n_parents:
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
    if explain:
        for name, description in _EXPLAIN_FIELDS.items():
            properties[name] = {"type": "string", "description": description}
    if space is not None and space.name != V1:
        properties.update(space.tool_schema())
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
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


def submit_designs_tool(
    slots: Sequence[SlotSpec], *, explain: bool = False, reserves: int = 0, space: DesignSpace | None = None
) -> dict[str, Any]:
    """The forced tool: ``analysis`` first, then one object per slot in order, then the reserves.

    With ``explain``, every slot also asks for ``changes`` and ``same_cell_distances``
    between the hypothesis and the parameters (arm C). ``reserves`` adds
    ``reserve_1`` ... ``reserve_n``: extra designs, each with its own operator, that
    replace rejected slots in order. A v2 ``space`` replaces the 10 v1 parameters of each
    slot with ``pad``, ``tip`` and ``params`` (:meth:`~rtt.search.space_v2.FreeformSpaceV2.tool_schema`).
    """
    v2 = space is not None and space.name != V1
    properties: dict[str, Any] = {
        "analysis": {
            "type": "string",
            "description": "Your analysis of the results so far and your plan, written before the designs.",
        }
    }
    for slot in slots:
        properties[slot.name] = _slot_schema(slot, explain, space=space)
    for name in reserve_names(reserves):
        properties[name] = _slot_schema(SlotSpec(name, "reserve"), explain, reserve=True, space=space)
    extra = f", plus {reserves} reserve designs that replace any rejected one" if reserves else ""
    noun = "gripper jaw recipes" if v2 else "gripper finger designs"
    return {
        "type": "function",
        "function": {
            "name": "submit_designs",
            "description": f"Submit {len(slots)} {noun} for simulation, one per slot{extra}.",
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


def min_points(min_distance: float) -> int:
    """A design distance threshold in prompt points (0.08 -> 80)."""
    return math.ceil(min_distance * POINTS_PER_RANGE * len(GEOMETRY_PARAMS) - 1e-9)


def points_row(design: FingerDesign) -> str:
    """The 8 numeric parameters in points, e.g. ``len 50 thick 50 ... taper 100``."""
    pts = points_of(design)
    return " ".join(f"{_POINT_COLUMNS[name]} {pts[name]}" for name in NUMERIC_PARAMS)


def _point_differences(a: FingerDesign, b: FingerDesign) -> str:
    """Per-parameter differences in points, largest first (categoricals count 100)."""
    terms: list[tuple[int, str]] = []
    for name in GEOMETRY_PARAMS:
        va, vb = getattr(a, name), getattr(b, name)
        if name in CATEGORICAL_OPTIONS:
            if va != vb:
                terms.append((POINTS_PER_RANGE, f"{name} {POINTS_PER_RANGE}"))
            continue
        diff = round(abs(to_points(name, va) - to_points(name, vb)))
        if diff:
            terms.append((diff, f"{_POINT_COLUMNS[name]} {diff}"))
    terms.sort(key=lambda t: -t[0])
    return ", ".join(text for _, text in terms) or "nothing"


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
    """How the diversity check works, in range fractions (arm Z's prompt)."""
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


def render_points_rule(min_distance: float) -> str:
    """The diversity check in points and cells (arm C's prompt)."""
    need = min_points(min_distance)
    return (
        "Positions are given in points: each number's range runs from 0 points (its minimum) to 100 points (its "
        'maximum), so 1 point is 1% of the range and the "10% of range" column above is 10 points. A design\'s '
        "cell is its inner_face + tip_profile pair. The distance between two designs in points is the sum over "
        "the 8 numbers of their difference in points, plus 100 if inner_face differs and 100 if tip_profile "
        "differs.\n\n"
        f"Every new design must be at least {need} points from every tested design and from every other new "
        "design this round, or it is rejected without being simulated and its slot goes to a random design. "
        "Designs in different cells are always at least 100 points apart, so only the tested designs in your "
        "design's own cell can be too close, and a design in a cell that no tested design uses always passes. "
        f"Within a cell the 8 numbers must differ by at least {need} points in total from each tested design "
        f"there; aim for {need + 20} or more, because the listed positions are rounded. Tweaking one or two "
        "numbers of a tested design by 10-30 points is always rejected."
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


def _team(island: int) -> str:
    """Model-facing team name of an island (1-based)."""
    return f"team {island + 1}"


def _origin_note(item: Evaluated, island: int | None) -> str:
    """Where a tested design came from, from ``island``'s point of view."""
    p = item.proposal
    if item.round == 0:
        return f"{p.origin}, round 0"
    if p.origin == "fallback_random":
        source = f"random fill, round {item.round}"
    else:
        source = f"{p.operator or p.origin}"
        if p.parents:
            source += " of " + " + ".join(p.parents)
        source += f", round {item.round}"
    if p.island is not None and island is not None:
        source += ", your team" if p.island == island else f", {_team(p.island)}"
    return source


@dataclass(frozen=True)
class _Claim:
    """A design another island already submitted this round (not simulated yet)."""

    label: str
    design: FingerDesign
    note: str


def render_tested(
    items: Sequence[Evaluated], island: int | None = None, claims: Sequence[_Claim] = ()
) -> str:
    """Every tested design of the run grouped by cell, positions in points, plus the untried cells.

    Cells run from the worst best-score to the best and rows inside a cell from the
    worst score to the best (best last, as in OPRO's ascending trajectory).
    """
    legend = (
        "Positions are in points (0-100): "
        + ", ".join(f"{short} = {name}" for name, short in _POINT_COLUMNS.items())
        + ". Cells run from the lowest best score to the highest; inside a cell, rows run from the lowest score "
        "to the highest."
    )
    if claims:
        legend += (
            ' Rows labelled "t<team>.<slot>" were submitted this round by another team and are not simulated '
            "yet; they count as tested for the distance rule."
        )
    groups: dict[tuple[str, str], list[tuple[float, int, str]]] = {}
    for item in items:
        line = f"  {item.label} {_pct(item.score.success)} | {points_row(item.design)} | {_origin_note(item, island)}"
        groups.setdefault(cell_of(item.design), []).append((item.score.success, item.index, line))
    for order, claim in enumerate(claims):
        line = f"  {claim.label} new | {points_row(claim.design)} | {claim.note}"
        groups.setdefault(cell_of(claim.design), []).append((math.inf, len(items) + order + 1, line))
    lines = [legend, ""]

    def cell_key(cell: tuple[str, str]) -> tuple[float, str]:
        scores = [s for s, _, _ in groups[cell] if s != math.inf]
        return (max(scores) if scores else math.inf, cell_name(cell))

    for cell in sorted(groups, key=cell_key):
        rows = sorted(groups[cell], key=lambda t: (t[0], t[1]))
        scores = [s for s, _, _ in rows if s != math.inf]
        tested = len(scores)
        head = f"{cell_name(cell)} ({tested} tested"
        head += f", best {_pct(max(scores))}):" if scores else "):"
        lines.append(head)
        lines.extend(line for _, _, line in rows)
    untried = [cell_name(c) for c in all_cells() if c not in groups]
    lines.append("")
    lines.append(
        "Cells with no tested design yet: "
        + (", ".join(untried) if untried else "none, every cell has at least one tested design")
        + "."
    )
    return "\n".join(lines)


def _family_vector(item: Evaluated, families: Sequence[str]) -> str:
    return ", ".join(f"{f} {_pct(item.score.per_family.get(f, 0.0))}" for f in families)


def _failure_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{mode} {n}" for mode, n in counts.items()) or "none"


def _nearest_line(item: Evaluated, history: Sequence[Evaluated], k: int) -> str:
    others = [o for o in history if o.index != item.index]
    near = neighbours(item.design, [o.design for o in others], [o.label for o in others], k)
    return ", ".join(f"{n.label} {round(1000 * n.distance)}" for n in near) or "none"


def render_elites(
    ranked_pool: Sequence[Evaluated],
    info: EvaluatorInfo,
    n: int,
    *,
    history: Sequence[Evaluated] | None = None,
    near_k: int = 0,
    task: GripperTask | None = None,
) -> str:
    """Top ``n`` of an already ranked pool (best first), printed worst first.

    With ``history`` and ``near_k``, each design also lists its cell and its
    ``near_k`` nearest tested designs in the run with their distances in points;
    with ``task``, also the fully open gap the validator measures (how much room
    is left for an inward bend or a hook).
    """
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
        if history is not None and near_k > 0:
            line = f"  cell {cell_name(cell_of(item.design))}; nearest (points): {_nearest_line(item, history, near_k)}"
            if task is not None:
                gap = validate(item.design, task).metrics.get("open_gap_mm")
                if gap is not None:
                    line += f"; open gap {gap:.1f} mm (needs >= {_num(task.required_open_gap_mm)})"
            lines.append(line)
        idea = item.proposal.hypothesis or item.design.rationale
        if idea:
            lines.append(f"  idea: {scrub_model_text(' '.join(idea.split())[:200])}")
    return "\n".join(lines) if lines else "None yet."


def render_pairs(ranked_pool: Sequence[Evaluated], n: int, need: int, show: int = 6) -> str:
    """Pairs among the top ``n`` designs at least ``need`` points apart (crossover candidates)."""
    top = list(ranked_pool[:n])
    pairs: list[tuple[float, int, str]] = []
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            d = distance_points(a.design, b.design)
            if d >= need:
                pairs.append((-(a.score.success + b.score.success), -d, f"{a.label} + {b.label} ({d})"))
    if not pairs:
        return (
            f"Crossover candidates: no two of your top {len(top)} designs are {need} points apart; take a second "
            "parent from another cell in the table."
        )
    pairs.sort()
    return (
        f"Crossover candidates (pairs of your top {len(top)} designs at least {need} points apart, best pairs "
        f"first): " + ", ".join(text for _, _, text in pairs[:show]) + "."
    )


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


def render_operators(
    untried: Sequence[str],
    min_distance: float,
    explore_min_distance: float,
    *,
    claims: bool = False,
    reserves: int = 0,
) -> str:
    """What each slot must do, with the point budgets of the distance rule."""
    need = min_points(min_distance)
    far = min_points(explore_min_distance) if explore_min_distance > 0 else 0
    cells = ", ".join(untried)
    if untried and far:
        explore_rule = (
            f"Put it in a cell that no tested design uses ({cells}); if you use a cell that already has tested "
            f"designs, it must be at least {far} points from every one of them. An explore that misses this is "
            "rejected like a near-copy."
        )
    elif far:
        explore_rule = (
            f"Every cell already has a tested design, so it must be at least {far} points from every tested design "
            "in its cell. An explore that misses this is rejected like a near-copy."
        )
    elif untried:
        explore_rule = f"Prefer a cell that no tested design uses ({cells})."
    else:
        explore_rule = "Keep it far from every tested design in its cell."
    claims_text = (
        ' and from the designs other teams already submitted this round (rows "t<team>.<slot>" in the table)'
        if claims
        else ""
    )
    return "\n".join(
        [
            (
                f"Call submit_designs with exactly 4 new designs, one per slot, plus {reserves} reserves:"
                if reserves
                else "Call submit_designs with exactly 4 new designs, one per slot:"
            ),
            (
                '- mutate_1 and mutate_2: each starts from ONE tested design (put its label, e.g. "d05", in parents) '
                "and makes one bold change aimed at that design's main failure mode, worth 100-250 points in total: "
                "for example 2-4 numbers moved by 25-60 points each, or a switch of inner_face or tip_profile (100 "
                "points) together with the numbers that switch needs. Give the two mutations different parents or "
                "clearly different changes. Mutations usually fail by landing on an earlier variation of the same "
                "parent, so check every tested design in the cell you end up in."
            ),
            (
                f"- crossover: combines TWO tested designs (both labels in parents) that are at least {2 * need} "
                "points apart. Take each parameter whole from one parent or the other; do not average numbers, "
                f"because an average lands close to both parents. The child must differ from each parent by at "
                f"least {need} points."
            ),
            f"- explore: tests an idea that no tested design covers. {explore_rule}",
            *(
                [
                    (
                        f"- reserve_1 to reserve_{reserves}: {reserves} more designs, each with its own operator "
                        "(mutate, crossover or explore) and parents, following the same rules. Whenever one of the "
                        "4 designs above is rejected (it breaks a validator rule or is too close to a tested design), "
                        "the next reserve that passes takes its place, so make the reserves as good as the first "
                        "four and different from them."
                    )
                ]
                if reserves
                else []
            ),
            "",
            "In each slot, before the parameters:",
            (
                '- "changes": every parameter you change from the parent (explore: from the nearest tested design '
                "in its cell) as name old -> new with the difference in points, then the total, e.g. "
                '"finger_width_mm 15 -> 25 mm (+50), tip_profile hook -> chamfer (+100): 150 points from d06".'
            ),
            (
                '- "same_cell_distances": go through the cell\'s rows in the table above and give the distance in '
                "points from the new design to each tested design in the same cell (at least the 3 nearest), "
                'nearest first, e.g. "d25 95, d06 150, d11 210", or "untried cell". If any is below '
                f"{need}, change the design before you write its parameters."
            ),
            (
                f"Every new design must also be at least {need} points from your other new designs this round"
                f"{claims_text}."
            ),
        ]
    )


# --------------------------------------------------------------------------- proposer core


@dataclass(frozen=True)
class ProposerConfig:
    """Knobs of the LLM arms (frozen in PREREG).

    ``explore_min_distance`` is the within-cell novelty an ``explore`` slot needs
    when its cell is already tried (0 turns the rule off); ``reserves`` is how many
    extra designs each call of an arm that uses reserves (arm C) returns to replace
    rejected slots; ``neighbours_shown`` is how many nearest tested designs each
    elite lists.
    """

    islands: int = 2
    elites_shown: int = 8
    failures_shown: int = 4
    rejections_shown: int = 6
    target_factor: float = 1.2
    min_distance: float = MIN_DISTANCE
    explore_min_distance: float = EXPLORE_MIN_DISTANCE
    reserves: int = 4
    neighbours_shown: int = 2
    crossover_pool: int = 6
    migrate_after_rounds: tuple[int, ...] = (2,)
    repairs_per_call: int = 1
    reflections_per_call: int = 1

    @classmethod
    def for_space(cls, space: str) -> ProposerConfig:
        """Pre-registered defaults: v1's 0.08 / 0.2 bars, or the v2 bars set by the threshold rule."""
        if space == V1:
            return cls()
        from rtt.search.diversity import EXPLORE_MIN_DISTANCE_V2, MIN_DISTANCE_V2

        return cls(min_distance=MIN_DISTANCE_V2, explore_min_distance=EXPLORE_MIN_DISTANCE_V2)

    def to_json(self) -> dict[str, Any]:
        return {
            "islands": self.islands,
            "elites_shown": self.elites_shown,
            "failures_shown": self.failures_shown,
            "rejections_shown": self.rejections_shown,
            "target_factor": self.target_factor,
            "min_distance": self.min_distance,
            "explore_min_distance": self.explore_min_distance,
            "reserves": self.reserves,
            "neighbours_shown": self.neighbours_shown,
            "crossover_pool": self.crossover_pool,
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
    rejected: FingerDesign | None = None
    """The last valid design of this slot that the diversity pass turned down."""
    declared: dict[str, str] = field(default_factory=dict)
    """The model's own ``changes`` / ``same_cell_distances`` text for the current design."""
    parents: tuple[str, ...] = ()
    """Parent labels the model gave for the current design."""
    request_hash: str | None = None
    """Request hash of the call that produced the current content."""
    screened: bool = False
    """True once the current proposal went through the diversity check."""


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


class _Kit:
    """The design-space operations the LLM machinery needs.

    For ``box-v1`` every method is the module-level v1 function it always was, so a v1 run
    is byte for byte unchanged; for v2 the same operations go through the design space.
    """

    def __init__(self, space: DesignSpace, pattern: re.Pattern[str] = _FORBIDDEN_PROMPT_TEXT) -> None:
        self.space = space
        self.v1 = space.name == V1
        self._ds: DesignSpace | None = None if self.v1 else space
        self.pattern = pattern

    def tool(self, slots: Sequence[SlotSpec], explain: bool, reserves: int) -> dict[str, Any]:
        return submit_designs_tool(slots, explain=explain, reserves=reserves, space=self._ds)

    def sampler(self, seed: int, context: Any) -> Any:
        return self.space.sample_sobol(seed, context)

    def raw_geometry(self, raw: Any) -> dict[str, Any] | None:
        return _raw_geometry(raw) if self.v1 else self.space.raw_params(raw)

    def params_key(self, geometry: Mapping[str, Any]) -> str:
        return params_key(geometry, self._ds)

    def parse_slot(self, raw: Any, spec: SlotSpec, context: Any) -> tuple[Any, list[str], dict[str, Any]]:
        if self.v1:
            return _parse_slot(raw, spec, context)
        return _parse_slot_space(raw, spec, context, self.space, self.pattern)

    def nearest(self, design: Any, refs: Sequence[Any], labels: Sequence[str] | None = None) -> Nearest:
        return nearest(design, refs, labels, space=self._ds)

    def neighbours(
        self, design: Any, refs: Sequence[Any], labels: Sequence[str] | None, k: int, **kw: Any
    ) -> list[Nearest]:
        return neighbours(design, refs, labels, k, space=self._ds, **kw)

    def explore_check(
        self, design: Any, refs: Sequence[Any], labels: Sequence[str] | None, min_cell_distance: float
    ) -> tuple[bool, Nearest | None]:
        return explore_check(design, refs, labels, min_cell_distance, space=self._ds)

    def duplicate(
        self, design: Any, refs: Sequence[Any], labels: Sequence[str], context: Any
    ) -> Screened | None:
        """v2 only: ``duplicate_design`` or ``duplicate_geometry`` against ``refs`` (v1: never)."""
        if self.v1:
            return None
        return screen(design, refs, labels, None, space=self.space, context=context)

    def cell_of(self, design: Any) -> tuple[str, str]:
        return cell_of(design, self._ds)

    def all_cells(self) -> list[tuple[str, str]]:
        return all_cells(self._ds)

    def distance_points(self, a: Any, b: Any) -> int:
        return distance_points(a, b, self._ds)

    def min_points(self, min_distance: float) -> int:
        if self.v1:
            return min_points(min_distance)
        return math.ceil(min_distance * POINTS_PER_RANGE * self.space.n_terms - 1e-9)

    def points_row(self, design: Any) -> str:
        return points_row(design) if self.v1 else self.space.points_row(design)

    def point_differences(self, a: Any, b: Any) -> str:
        return _point_differences(a, b) if self.v1 else self.space.point_differences(a, b)


class LLMProposer(SearchStrategy):
    """Shared machinery of the LLM arms: calls, validation, repair, diversity, fallback."""

    uses_llm = True
    origin: ClassVar[str] = ""
    slots: ClassVar[tuple[SlotSpec, ...]] = PROPOSER_SLOTS
    #: Whether slots carry ``changes`` / ``same_cell_distances`` (arm C) or only a hypothesis (arm Z).
    explain_changes: ClassVar[bool] = False
    #: Whether calls also return ``config.reserves`` reserve designs (arm C).
    uses_reserves: ClassVar[bool] = False
    #: Follow-up prompt templates (v2 arms use their own under ``prompts/v2/``).
    repair_template: ClassVar[str] = "repair.md"
    diversity_template: ClassVar[str] = "diversity.md"
    #: Rejection reasons the reflection call answers.
    diversity_reasons: ClassVar[tuple[str, ...]] = DIVERSITY_REASONS
    #: What no prompt may mention; model text quoted back is scrubbed of it first.
    forbidden: ClassVar[re.Pattern[str]] = _FORBIDDEN_PROMPT_TEXT

    def __init__(self, ctx: RunContext, client: LLMClient, config: ProposerConfig | None = None) -> None:
        super().__init__(ctx)
        self.client = client
        self.config = config or ProposerConfig.for_space(ctx.space.name)
        if self.config.islands < 1:
            raise ValueError("at least one island is needed")
        self.n_reserves = self.config.reserves if self.uses_reserves else 0
        if self.n_reserves < 0:
            raise ValueError("reserves must be >= 0")
        self.kit = _Kit(ctx.space, self.forbidden)
        self.tool = self.kit.tool(self.slots, self.explain_changes, self.n_reserves)
        self.fallback = self.kit.sampler(derive_seed(ctx.run_seed, "fallback", ctx.arm), ctx.gripper_task)
        self._disabled: str | None = None
        self._messages: dict[int, list[dict[str, str]]] = {}
        self._failures = _FailureLog(client.cache)
        self._accepted: list[_Slot] = []
        self._accepted_round: int | None = None
        self._seen_reserves: set[str] = set()

    @property
    def disabled_reason(self) -> str | None:
        """Why this run stopped calling the model (``spend_cap``), or ``None``."""
        return self._disabled

    # ------------------------------------------------------------------ hooks

    @abstractmethod
    def build_messages(self, round_index: int, island: int) -> list[dict[str, str]]:
        """System and user messages for this island's main call."""

    def round_claims(self, round_index: int, island: int) -> list[_Claim]:
        """Designs that islands asked before ``island`` already hold this round (accepted or random fill)."""
        if self._accepted_round != round_index:
            return []
        claims = []
        for slot in self._accepted:
            if slot.island >= island or slot.proposal is None:
                continue
            note = f"{slot.spec.operator}, this round, {_team(slot.island)}"
            if slot.proposal.origin == "fallback_random":
                note = f"random fill, this round, {_team(slot.island)}"
            claims.append(_Claim(f"t{slot.island + 1}.{slot.spec.name}", slot.proposal.design, note))
        return claims

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
        self._accepted, self._accepted_round = [], round_index
        slots: list[_Slot] = []
        for island in range(islands):
            later_mains = future_mains + (islands - 1 - island)
            slots.extend(self._run_island(round_index, island, later_mains))
        return [s.proposal for s in slots if s.proposal is not None]

    # ------------------------------------------------------------------ steps

    def _run_island(self, round_index: int, island: int, later_mains: int) -> list[_Slot]:
        """Main call, reserves, at most one repair and one reflection, then random fill, for one island.

        Holes (invalid or rejected slots) are filled first from the call's reserve
        designs, in order; a repair call is made only for invalid slots that no
        reserve could fill, and a reflection only for near-copies that are left.
        Reserves a follow-up call brings but does not need for its own slots are
        offered to every remaining hole before the next call or the random fill.
        """
        slots = [_Slot(island, spec) for spec in self.slots]
        self._seen_reserves = set()
        messages = self.build_messages(round_index, island)
        self._messages[island] = messages
        outcome = self._call("main", round_index, island, messages, keep=0)
        self._absorb(round_index, outcome, slots, stage="main")
        pool = self._absorb_reserves(outcome, island, "main")
        self._screen(round_index, slots, pool)
        for _ in range(self.config.repairs_per_call):
            broken = [s for s in slots if s.proposal is None and s.reason in ("invalid", "parse_error")]
            if not broken:
                break
            repair = self._repair_messages(messages, slots, broken, outcome)
            outcome = self._call("repair", round_index, island, repair, keep=later_mains)
            if outcome is None:
                break
            self._absorb(round_index, outcome, broken, stage="repair")
            pool.extend(self._absorb_reserves(outcome, island, "repair"))
            self._screen(round_index, broken, pool)
        self._screen(round_index, slots, pool)  # leftover repair reserves, before any reflection call
        if any(s.proposal is None and s.reason in self.diversity_reasons for s in slots):
            self._reflect(round_index, island, slots, pool, keep=later_mains)
            self._screen(round_index, slots, pool)  # leftover reflection reserves, before the random fill
        for slot in slots:
            if slot.proposal is None:
                self._fill_fallback(slot, self._accepted)
        return slots

    def _screen(self, round_index: int, slots: Sequence[_Slot], pool: list[_Slot]) -> None:
        """Diversity-check the new proposals in ``slots``, then fill their holes from ``pool``."""
        for slot in slots:
            if slot.proposal is not None and not slot.screened:
                self._check_diversity(round_index, slot, self._accepted)
        for slot in slots:
            if slot.proposal is None:
                self._fill_from_reserves(round_index, slot, pool)

    def _absorb_reserves(self, outcome: _CallOutcome | None, island: int, stage: str) -> list[_Slot]:
        """The call's reserve designs as candidate slots (validated now, checked for diversity when used).

        A follow-up call often copies the previous call's reserves unchanged; a
        reserve whose parameters this island has already received is skipped, so it
        is neither tried nor logged twice.
        """
        if not self.n_reserves or outcome is None or outcome.args is None:
            return []
        out: list[_Slot] = []
        for name in reserve_names(self.n_reserves):
            raw = outcome.args.get(name)
            if raw is None:
                continue
            geometry = self.kit.raw_geometry(raw)
            key = self.kit.params_key(geometry) if geometry is not None else repr(raw)
            if key in self._seen_reserves:
                continue
            self._seen_reserves.add(key)
            declared = raw.get("operator") if isinstance(raw, Mapping) else None
            parents = raw.get("parents") if isinstance(raw, Mapping) else None
            n_parents = len(parents) if isinstance(parents, list) else 0
            if declared not in RESERVE_OPERATORS:
                declared = ("explore", "mutate", "crossover")[min(n_parents, 2)]
            spec = SlotSpec(name, str(declared), n_parents)
            slot = _Slot(island, spec, stage=stage, request_hash=outcome.record.request_hash)
            design, errors, meta = self.kit.parse_slot(raw, spec, self.ctx.gripper_task)
            slot.raw = geometry
            slot.declared, slot.parents = meta["declared"], meta["parents"]
            if design is None:
                slot.reason, slot.errors = "invalid", errors
            else:
                slot.proposal = Proposal(
                    design=design,
                    origin=self.origin,
                    operator=spec.operator,
                    island=island,
                    slot=name,
                    parents=meta["parents"],
                    hypothesis=meta["hypothesis"],
                    via=stage,
                    request_hash=outcome.record.request_hash,
                    notes=meta.get("notes", ()),
                )
            out.append(slot)
        return out

    def _fill_from_reserves(self, round_index: int, hole: _Slot, pool: list[_Slot]) -> bool:
        """Give ``hole`` the next reserve that is valid and passes the diversity check (rejections logged)."""
        while pool:
            reserve = pool.pop(0)
            if reserve.proposal is None:
                self.ctx.reject(
                    Rejection(
                        round=round_index,
                        stage=reserve.stage,
                        reason=reserve.reason or "invalid",
                        errors=tuple(reserve.errors),
                        island=reserve.island,
                        slot=reserve.spec.name,
                        params=reserve.raw,
                        request_hash=reserve.request_hash,
                    )
                )
                continue
            self._check_diversity(round_index, reserve, self._accepted)
            if reserve.proposal is None:
                continue
            hole.proposal, hole.stage, hole.screened = reserve.proposal, reserve.stage, True
            hole.declared, hole.parents = reserve.declared, reserve.parents
            hole.reason, hole.errors = None, []
            return True
        return False

    def _reference(self, accepted: Sequence[_Slot], island: int) -> tuple[list[Any], list[str]]:
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
        """Accept ``slot`` or reject it as ``too_similar`` (any slot) or ``not_novel`` (explore)."""
        assert slot.proposal is not None
        designs, labels = self._reference(accepted, slot.island)
        design = slot.proposal.design
        near = self.kit.nearest(design, designs, labels)
        reason: str | None = None
        duplicate = self.kit.duplicate(design, designs, labels, self.ctx.gripper_task)
        if duplicate is not None:
            reason, message, near = duplicate.reason, duplicate.message, duplicate.near
        elif near.distance < self.config.min_distance:
            reason = "too_similar"
            message = f"distance {near.distance:.3f} to {near.label} (needs >= {self.config.min_distance:g})"
        elif slot.spec.operator == "explore" and self.config.explore_min_distance > 0:
            novel, same_cell = self.kit.explore_check(
                design, designs, labels, self.config.explore_min_distance
            )
            if not novel and same_cell is not None:
                reason, near = "not_novel", same_cell
                message = (
                    f"explore in the tried cell {cell_name(self.kit.cell_of(design))}: distance "
                    f"{near.distance:.3f} to {near.label} (needs an untried cell or >= "
                    f"{self.config.explore_min_distance:g} within the cell)"
                )
        if reason is None:
            slot.screened = True
            accepted.append(slot)
            return
        errors = [message]
        if slot.declared.get("same_cell_distances"):
            errors.append(f"model's same_cell_distances: {slot.declared['same_cell_distances'][:160]}")
        self.ctx.reject(
            Rejection(
                round=round_index,
                stage=slot.stage,
                reason=reason,
                errors=tuple(errors),
                island=slot.island,
                slot=slot.spec.name,
                params=slot.raw,
                request_hash=slot.proposal.request_hash,
            )
        )
        slot.similar = near
        slot.reason = reason
        slot.errors = errors
        slot.rejected = design
        slot.proposal = None

    def _problem(self, slot: _Slot, designs: Sequence[Any], labels: Sequence[str]) -> str:
        """One rejected slot for the reflection prompt: exact neighbours, differences and shortfall."""
        cfg = self.config
        kit = self.kit
        near = slot.similar
        design = slot.rejected
        parents = f" (parent {' + '.join(slot.parents)})" if slot.parents else ""
        if design is None or near is None or near.index is None:
            return f"- {slot.spec.name}: {'; '.join(slot.errors)}"
        d_pts = kit.distance_points(design, designs[near.index])
        if slot.reason == "not_novel":
            same = [labels[i] for i, ref in enumerate(designs) if kit.cell_of(ref) == kit.cell_of(design)]
            far = kit.min_points(cfg.explore_min_distance)
            head = (
                f"- {slot.spec.name}{parents}: its cell {cell_name(kit.cell_of(design))} already holds "
                f"{', '.join(same)}; the nearest, {near.label}, is {d_pts} points away, and an explore needs an "
                f"untried cell or {far} points from each of them (short by {max(1, far - d_pts)})."
            )
        elif slot.reason == "duplicate_design":
            head = (
                f"- {slot.spec.name}{parents}: the same design as {near.label} once inactive parameters are "
                "dropped and values are rounded to the grid (0 points apart)."
            )
        elif slot.reason == "duplicate_geometry":
            head = (
                f"- {slot.spec.name}{parents}: builds exactly the same jaws as {near.label} on every training part "
                f"({d_pts} points apart, but the differences do not change the geometry)."
            )
        else:
            need = kit.min_points(cfg.min_distance)
            head = (
                f"- {slot.spec.name}{parents}: {d_pts} points from {near.label} (needs {need}, short by "
                f"{max(1, need - d_pts)})."
            )
        claimed = slot.declared.get("same_cell_distances")
        if claimed:
            head += f' You wrote same_cell_distances "{self._scrub(claimed[:120])}".'
        lines = [
            head,
            f"    yours: {cell_name(kit.cell_of(design))} | {kit.points_row(design)}",
            (
                f"    {near.label}: {cell_name(kit.cell_of(designs[near.index]))} | "
                f"{kit.points_row(designs[near.index])}; differs by: "
                f"{kit.point_differences(design, designs[near.index])}"
            ),
        ]
        limit = 2.5 * cfg.min_distance
        others = [
            n for n in kit.neighbours(design, designs, labels, 4, max_distance=limit) if n.index != near.index
        ]
        if others:
            lines.append(
                f"    also within {kit.min_points(limit)} points: "
                + ", ".join(f"{n.label} {self._points(n.distance)}" for n in others[:3])
            )
        return "\n".join(lines)

    def _scrub(self, text: str) -> str:
        return scrub_model_text(text, self.forbidden)

    def _points(self, distance: float) -> int:
        """A distance in prompt points (v1: 1000 x distance, as it always printed)."""
        if self.kit.v1:
            return round(1000 * distance)
        return round(distance * POINTS_PER_RANGE * self.ctx.space.n_terms)

    def _follow_up_note(self) -> str:
        """What a repair or reflection call should do with the explain fields and the reserves."""
        note = ""
        if self.explain_changes:
            note += ' Write "changes" and "same_cell_distances" again for every replaced slot.'
        if self.n_reserves:
            note += (
                f" Fill reserve_1 to reserve_{self.n_reserves} with fresh designs that differ from everything above:"
                " they replace, in order, any replaced slot that is still rejected."
            )
        return note

    def _reflect(
        self,
        round_index: int,
        island: int,
        slots: Sequence[_Slot],
        pool: list[_Slot],
        keep: int,
    ) -> None:
        for _ in range(self.config.reflections_per_call):
            pending = [s for s in slots if s.proposal is None and s.reason in self.diversity_reasons]
            if not pending:
                return
            designs, labels = self._reference(self._accepted, island)
            problems = "\n".join(self._problem(s, designs, labels) for s in pending)
            keep_names = [s.spec.name for s in slots if s.proposal is not None]
            used = {self.kit.cell_of(d) for d in designs}
            untried = [cell_name(c) for c in self.kit.all_cells() if c not in used]
            explore_clause = ""
            if any(s.reason == "not_novel" for s in pending):
                explore_clause = (
                    ", or is an explore in a cell that is already tested and less than "
                    f"{self.kit.min_points(self.config.explore_min_distance)} points from a design there"
                )
            text = prompts.render(
                self.diversity_template,
                min_points=self.kit.min_points(self.config.min_distance),
                explore_clause=explore_clause,
                problems=problems,
                untried=", ".join(untried) or "none",
                keep=", ".join(keep_names) or "none",
                fields_note=self._follow_up_note(),
            )
            messages = _append_to_user(self._messages[island], text)
            outcome = self._call("reflection", round_index, island, messages, keep=keep)
            if outcome is None:
                return
            self._absorb(round_index, outcome, pending, stage="reflection")
            pool.extend(self._absorb_reserves(outcome, island, "reflection"))
            self._screen(round_index, pending, pool)

    def _fill_fallback(self, slot: _Slot, accepted: list[_Slot]) -> None:
        designs, _ = self._reference(accepted, slot.island)
        min_d = self.config.min_distance
        before = self.fallback.rejected
        if self.kit.v1:
            try:
                design = self.fallback.sample(1, accept=lambda d: nearest(d, designs).distance >= min_d)[0]
            except SamplerExhaustedError:
                design = self.fallback.sample(1)[0]
        else:  # the shared screen: never a duplicate, and the diversity bar while it can be met
            from rtt.search.random_search import screen_accept

            space, context = self.ctx.space, self.ctx.gripper_task
            try:
                design = self.fallback.sample(1, screen_accept(space, context, lambda: designs, min_d))[0]
            except SamplerExhaustedError:
                design = self.fallback.sample(1, screen_accept(space, context, lambda: designs, None))[0]
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
        assert_prompt_clean(messages, self.forbidden)
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
            slot.stage, slot.screened = stage, False
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
            design, errors, meta = self.kit.parse_slot(raw, slot.spec, self.ctx.gripper_task)
            slot.raw = self.kit.raw_geometry(raw)
            slot.declared, slot.parents = meta["declared"], meta["parents"]
            slot.request_hash = outcome.record.request_hash
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
                notes=meta.get("notes", ()),
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
                # json.loads accepts NaN / Infinity in tool arguments; wire_json refuses them.
                wire_json(_json_safe(outcome.args))
                if outcome is not None and outcome.args is not None
                else "(none)"
            )
            kept = [s.spec.name for s in slots if s.proposal is not None]
            keep = ", ".join(kept) or "none"
        text = prompts.render(
            self.repair_template,
            problems=self._scrub(problems),
            previous=self._scrub(previous),
            keep=keep,
            reserves_note=self._follow_up_note(),
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
    meta: dict[str, Any] = {"parents": (), "hypothesis": None, "declared": {}}
    if not isinstance(raw, Mapping):
        return None, [f"slot '{spec.name}' is missing or is not an object"], meta
    hypothesis = raw.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis.strip():
        meta["hypothesis"] = " ".join(hypothesis.split())[:_MAX_HYPOTHESIS_CHARS]
    meta["parents"] = _parent_labels(raw.get("parents"))
    for name in _EXPLAIN_FIELDS:
        value = raw.get(name)
        if isinstance(value, str) and value.strip():
            meta["declared"][name] = " ".join(value.split())[:_MAX_DECLARED_CHARS]
    mapping: dict[str, Any] = {k: raw[k] for k in GEOMETRY_PARAMS if k in raw}
    if meta["hypothesis"]:
        mapping["rationale"] = meta["hypothesis"]
    result = validate(mapping, task)
    if not result.ok or result.design is None:
        return None, list(result.errors), meta
    return result.design, [], meta


def _parse_slot_space(
    raw: Any,
    spec: SlotSpec,
    context: Any,
    space: DesignSpace,
    pattern: re.Pattern[str] = _FORBIDDEN_PROMPT_TEXT,
) -> tuple[Any, list[str], dict[str, Any]]:
    """:func:`_parse_slot` for a v2 space: ``pad``, ``tip`` and ``params`` through the space's validators."""
    meta: dict[str, Any] = {"parents": (), "hypothesis": None, "declared": {}, "notes": ()}
    if not isinstance(raw, Mapping):
        return None, [f"slot '{spec.name}' is missing or is not an object"], meta
    hypothesis = raw.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis.strip():
        meta["hypothesis"] = " ".join(hypothesis.split())[:_MAX_HYPOTHESIS_CHARS]
    meta["parents"] = _parent_labels(raw.get("parents"), pattern)
    for name in _EXPLAIN_FIELDS:
        value = raw.get(name)
        if isinstance(value, str) and value.strip():
            meta["declared"][name] = " ".join(value.split())[:_MAX_DECLARED_CHARS]
    check = space.validate(space.slot_mapping(raw), context)
    meta["notes"] = tuple(check.notes)
    if not check.ok or check.design is None:
        return None, list(check.errors), meta
    return check.design, [], meta


# --------------------------------------------------------------------------- arm C


class NemotronProposer(LLMProposer):
    """Arm C: proposals conditioned on this run's own training results, in 2 islands."""

    arm = "nemotron"
    origin = "nemotron"
    slots = PROPOSER_SLOTS
    explain_changes = True
    uses_reserves = True

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
        history = list(self.ctx.history)
        claims = self.round_claims(round_index, island)
        used = {cell_of(item.design) for item in history} | {cell_of(c.design) for c in claims}
        untried = [cell_name(c) for c in all_cells() if c not in used]
        last_round = [r for r in self.ctx.rejections if r.round == round_index - 1 and r.island == island]
        elites = render_elites(
            ranked,
            info,
            cfg.elites_shown,
            history=history,
            near_k=cfg.neighbours_shown,
            task=self.ctx.gripper_task,
        )
        pairs = render_pairs(ranked, cfg.crossover_pool, 2 * min_points(cfg.min_distance))
        system = prompts.render(
            "proposer_system.md", n_islands=cfg.islands, island=island + 1, budget=self.ctx.budget.budget
        )
        user = prompts.render(
            "proposer_user.md",
            task=render_task(info, self.ctx.gripper_task),
            parameters=render_parameters(),
            rules=render_rules(self.ctx.gripper_task),
            distance_rule=render_points_rule(cfg.min_distance),
            n_tested=len(history),
            tested=render_tested(history, island, claims),
            n_elites=min(cfg.elites_shown, len(ranked)),
            n_pool=len(ranked),
            n_conditions=len(info.conditions),
            elites=f"{elites}\n\n{pairs}",
            failures=render_failures(ranked[: cfg.elites_shown], info, cfg.failures_shown),
            rejections=render_rejections(last_round, cfg.rejections_shown),
            target=render_target(ranked[0].score.success, cfg.target_factor),
            round=round_index,
            last_round=self.ctx.budget.n_rounds - 1,
            operators=render_operators(
                untried,
                cfg.min_distance,
                cfg.explore_min_distance,
                claims=bool(claims),
                reserves=self.n_reserves,
            ),
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
        "tool:proposer": h(
            submit_designs_tool(
                PROPOSER_SLOTS, explain=NemotronProposer.explain_changes, reserves=ProposerConfig().reserves
            )
        ),
        "tool:zeroshot": h(submit_designs_tool(ZEROSHOT_SLOTS)),
        "parameter_table": h(render_parameters()),
        "points_rule": h(render_points_rule(MIN_DISTANCE)),
    }
