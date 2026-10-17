"""The Nemotron morphology proposer: one forced ``submit_morphologies`` call, at most one repair.

:func:`propose_recipes` asks ``nvidia/nemotron-3-super-120b-a12b`` (through
:class:`rtt.llm.client.LLMClient`: cache, ledger, spend cap, retries) for ``n`` designs for one
part. The prompt (``prompts/``) is part-aware: the task, the part (size, mass, material,
resting pose, features, yaw symmetry, how far its outline reaches in every direction), the
generators' parameter tables, every validator rule with this task's numbers, a few facts about
the simulation, and every design already tested on this part with its measured outcome
(:class:`DesignOutcome`: held and registered rates and failure modes).

**Tool.** ``submit_morphologies`` has ``analysis`` first, then ``designs``: exactly ``n``
objects, each with a ``hypothesis``, the generator in ``kind`` and its ``params``, whose
schema is :func:`~rtt.effectors.types.params_json_schema` of that generator's space (an
``anyOf`` over the generators offered; descriptions live in the prompt's tables).

**Screen.** Every design is canonicalised (:func:`~rtt.effectors.morphology.diversity.canonical_recipe`),
validated for the part (:func:`~rtt.effectors.morphology.validators.validate_recipe`) and
screened by the diversity rule against the designs already tested and the ones accepted in
this batch (:func:`~rtt.effectors.morphology.diversity.screen`). Refused designs, and a call
whose arguments cannot be parsed, get one repair call that lists every problem and asks for
replacements; whatever is still refused after it is reported in
:attr:`ProposalBatch.rejections`, so a batch may hold fewer than ``n`` recipes (the caller fills
the gap, for example from the random control).

**Guards.** No prompt may contain held-out vocabulary, the unseen held-out part families or a
brand token (:class:`~rtt.search.nemotron.PromptLeakError` before anything is sent); model
text quoted back (hypotheses, the previous call) is scrubbed of both first.

Replays are exact: with ``cache_mode="replay"`` the same part, history and code give the same
requests (content-addressed), the same responses and the same recipes, with zero API calls.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from rtt.effectors.morphology import prompts
from rtt.effectors.morphology.diversity import MIN_POINTS, canonical_recipe, screen
from rtt.effectors.morphology.layout import (
    MAX_ACTUATORS,
    MAX_EFFECTOR_MASS_KG,
    MAX_JOINT_TORQUE_NM,
    MAX_TOTAL_FORCE_N,
)
from rtt.effectors.morphology.spaces import SpaceKey, kind_summary, select_spaces, space_label
from rtt.effectors.morphology.validators import (
    ERROR_ENVELOPE_MM,
    MAX_SLENDERNESS,
    MIN_FEATURE_MM,
    MIN_OPEN_CLEARANCE_MM,
    MIN_OVERLAP_MM,
    PALM_CLEARANCE_MM,
    YAW_ENVELOPE_DEG,
    validate_recipe,
)
from rtt.effectors.types import MorphologyRecipe, ParamSpec, find_brand_tokens, params_json_schema
from rtt.llm.cache import CacheEntry, ResponseCache, wire_json
from rtt.llm.client import (
    BudgetExceededError,
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
from rtt.llm.ledger import Ledger, LedgerEntry
from rtt.llm.providers import NEMOTRON_SUPER, OpenAITransport
from rtt.search.nemotron import PromptLeakError
from rtt.search.nemotron_v2 import FORBIDDEN_PROMPT_TEXT_V2

if TYPE_CHECKING:
    from rtt.llm.cache import CacheMode
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

__all__ = [
    "DESIGNER_MODEL",
    "DESIGNER_SAMPLING",
    "SUBMIT_TOOL",
    "CallRecord",
    "DesignOutcome",
    "ProposalBatch",
    "Rejection",
    "ScrubbingCache",
    "assert_clean",
    "build_messages",
    "make_designer_client",
    "propose_recipes",
    "scrub",
    "submit_morphologies_tool",
]

DESIGNER_MODEL: Final = NEMOTRON_SUPER
#: Sampling for the designer (thinking on, the model's default). 8,192 output tokens leave room
#: for reasoning plus 4 designs; with an ~5k-token prompt the worst case is about $0.009 a call.
DESIGNER_SAMPLING: Final[dict[str, float | int]] = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}
SUBMIT_TOOL: Final = "submit_morphologies"
_MAX_ANALYSIS: Final = 2000
_MAX_HYPOTHESIS: Final = 400
_MAX_QUOTED: Final = 220


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class DesignOutcome:
    """Physics feedback on one tested design, shown to the proposer in later rounds.

    ``held_rate``/``registered_rate``: shares of the screen episodes (in [0, 1]);
    ``failure_modes``: counts of each failure mode over those episodes; ``n_episodes``: how
    many episodes the rates are over (0 when unknown); ``per_part``: for a design tested on
    several parts, ``part -> (held rate, registered rate)``.
    """

    recipe: MorphologyRecipe
    held_rate: float
    registered_rate: float
    failure_modes: Mapping[str, int]
    n_episodes: int = 0
    per_part: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("held_rate", "registered_rate"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v!r}")
            object.__setattr__(self, name, v)
        object.__setattr__(
            self, "failure_modes", {str(k): int(v) for k, v in dict(self.failure_modes).items()}
        )
        object.__setattr__(
            self, "per_part", {str(k): (float(v[0]), float(v[1])) for k, v in dict(self.per_part).items()}
        )


@dataclass(frozen=True)
class Rejection:
    """A design (or a whole call) that did not become a recipe, and why."""

    slot: int  # 1-based design index; 0 for the whole call
    stage: str  # "main" | "repair"
    reason: str  # invalid | duplicate | too_similar | parse_error | call_failed | spend_cap
    errors: tuple[str, ...]
    kind: str | None = None
    params: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "stage": self.stage,
            "reason": self.reason,
            "errors": list(self.errors),
            "kind": self.kind,
            "params": _json_safe(dict(self.params)) if self.params is not None else None,
        }


@dataclass(frozen=True)
class CallRecord:
    """One model call of a batch (no message bodies; the request hash points into the cache)."""

    stage: str
    request_hash: str | None
    cached: bool
    ok: bool
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float
    finish_reason: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class ProposalBatch:
    """What :func:`propose_recipes` returns: valid recipes, rejections, calls and their ledger."""

    part_id: str
    n_requested: int
    recipes: tuple[MorphologyRecipe, ...]
    rejections: tuple[Rejection, ...]
    calls: tuple[CallRecord, ...]
    ledger: tuple[LedgerEntry, ...]
    analysis: str | None = None
    kinds: tuple[str, ...] = ()
    template_hashes: Mapping[str, str] = field(default_factory=dict)

    @property
    def shortfall(self) -> int:
        """How many of the requested designs are missing (the caller fills them)."""
        return self.n_requested - len(self.recipes)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls if not c.cached)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "n_requested": self.n_requested,
            "kinds": list(self.kinds),
            "recipes": [r.to_dict() for r in self.recipes],
            "rejections": [r.to_dict() for r in self.rejections],
            "calls": [c.to_dict() for c in self.calls],
            "ledger": [e.to_json() for e in self.ledger],
            "analysis": self.analysis,
            "cost_usd": self.cost_usd,
            "template_hashes": dict(self.template_hashes),
        }


# --------------------------------------------------------------------------- guards


_BRAND_SCRUB = re.compile(r"\S+")


def _scrub_words(text: str) -> str:
    """``text`` with held-out vocabulary and brand tokens replaced by ``[removed]`` (layout kept)."""
    text = FORBIDDEN_PROMPT_TEXT_V2.sub("[removed]", text)
    for token in sorted(set(find_brand_tokens(text)), key=len, reverse=True):
        text = re.sub(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", "[removed]", text, flags=re.IGNORECASE)
    return text


def scrub(text: str) -> str:
    """Model text made safe to quote: held-out vocabulary and brand tokens removed, one line."""
    return _scrub_words(" ".join(str(text).split()))


def assert_clean(messages: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a prompt that names held-out material or a brand (nothing is sent)."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if FORBIDDEN_PROMPT_TEXT_V2.search(content):
            raise PromptLeakError("designer prompt refers to held-out material; refusing to send it")
        if find_brand_tokens(content):
            raise PromptLeakError("designer prompt names a brand; refusing to send it")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


# --------------------------------------------------------------------------- tool


def _compact(schema: dict[str, Any]) -> dict[str, Any]:
    """A params schema without per-property descriptions (they are in the prompt's tables)."""
    props = {
        k: {kk: vv for kk, vv in v.items() if kk != "description"} for k, v in schema["properties"].items()
    }
    return {**schema, "properties": props}


def submit_morphologies_tool(spaces: Mapping[SpaceKey, Sequence[ParamSpec]], n: int) -> dict[str, Any]:
    """The forced tool: ``analysis``, then exactly ``n`` designs (module docstring)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    labels = [space_label(k) for k in spaces]
    schemas = [_compact(params_json_schema(s)) for s in spaces.values()]
    params = schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
    design = {
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "What this design should do on this part and why (under 40 words).",
            },
            "kind": {"type": "string", "enum": labels, "description": "The generator."},
            "params": params,
        },
        "required": ["hypothesis", "kind", "params"],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_TOOL,
            "description": f"Submit {n} effector designs for this part, each a generator and its parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": "Your analysis of the part and the outcomes so far, and your plan (written first).",
                    },
                    "designs": {"type": "array", "items": design, "minItems": n, "maxItems": n},
                },
                "required": ["analysis", "designs"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------- rendering


def _num(value: float, digits: int = 1) -> str:
    """``value`` with at most ``digits`` decimals and no trailing zeros (``100`` stays ``100``)."""
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def render_task(task: TaskV2) -> str:
    c, sc, reg = task.controller, task.score, task.registration
    j = task.jitter
    return "\n".join(
        [
            (
                "Pick the part up from a table with a top-down wrist (it moves in x, y and z and turns about the "
                "vertical axis; it cannot tilt), lift it at least "
                f"{_num(100 * sc.success_lift)} cm and hold it for {_num(sc.success_hold)} s, all within "
                f"{_num(sc.max_time)} s."
            ),
            (
                f"- Sequence, the same for every effector: settle {_num(c.settle_time)} s; descend from above in "
                f"{_num(c.descend_time)} s until the effector's lowest point is at grip_height_frac x part height "
                f"(at least {_num(1000 * c.min_tip_clearance)} mm); close on a {_num(c.close_time)} s ramp plus "
                f"{_num(c.squeeze_time)} s; lift {_num(100 * c.lift_height)} cm in {_num(c.lift_time)} s; hold."
            ),
            (
                "- Held (the main score): lifted and held with no physics flag: the effector pressing the table "
                f"harder than {_num(sc.finger_table_force_max)} N, the effector sinking more than "
                f"{_num(task.checks.held_penetration_max_mm)} mm into the part, the part pushed away while "
                "closing, or a simulation fault."
            ),
            (
                "- Registered: held, at least two effector bodies squeezing the part from directions at least 90 "
                "deg apart, and the part in the same pose as in this design's own centred grasp: within "
                f"{_num(reg.dx_max_mm, 2)} mm, {_num(reg.yaw_max_deg)} deg of yaw (modulo the part's symmetry) and "
                f"{_num(reg.tilt_max_deg)} deg of tilt."
            ),
            (
                "- Tests: the screen grasps the part at 4 placements (c0 centred; c1 2 mm off centre along x; c2 "
                "3 mm off along -x; c3 1 mm off and turned 6 deg) with 2 jitter seeds each (8 episodes); the "
                "controller always aims at the centre and never knows the error. Each episode also jitters the "
                f"part by up to {_num(1000 * j.pos)} mm and {_num(j.yaw_deg)} deg and the friction by "
                f"{_num(100 * j.mu_frac)} %."
            ),
        ]
    )


def _needed_line(part: PartSpec) -> str:
    """Open-clearance radius needed in each direction (outline within +-7 deg + error + clearance)."""
    from rtt.effectors.morphology.validators import needed_inner_mm

    sym = part.yaw_symmetry
    stop = 180 if sym in (0, 2, 4, 6) else 360
    values = ", ".join(f"{a}: {_num(needed_inner_mm(part, math.radians(a)))}" for a in range(0, stop, 15))
    half = "; the opposite direction is the same" if stop == 180 else ""
    return (
        "- Opening needed: in the open pose a digit pointing in direction d (deg around the vertical axis, 0 = +x, "
        f"90 = +y{half}) must have its innermost point at least this far from the centre (mm; the outline within "
        f"+-{_num(YAW_ENVELOPE_DEG)} deg plus {_num(ERROR_ENVELOPE_MM + MIN_OPEN_CLEARANCE_MM)} mm): {values}."
    )


def render_part(part: PartSpec, task: TaskV2) -> str:
    """One part: what it is, size, mass, pose, features, symmetry, opening and reach numbers."""
    from rtt.parts.library import part_table

    row = next((r for r in part_table() if r["name"] == part.name), None)
    sx, sy, sz = (1000.0 * v for v in part.size)
    sym = part.yaw_symmetry
    sym_text = "any yaw" if sym == 0 else ("none" if sym == 1 else f"{sym}-fold")
    lines = [f"- {part.name} ({part.family}): {row['description'] if row else 'an uploaded part'}."]
    lines.append(
        f"- Size {_num(sx)} x {_num(sy)} x {_num(sz)} mm (x, y, z; z up, resting on the table in this pose), "
        f"mass {_num(1000 * part.mass)} g (weight {_num(part.mass * 9.81, 2)} N), material "
        f"{part.material.name.replace('_', ' ')}, yaw symmetry {sym_text}."
    )
    if part.features:
        feats = []
        for f in part.features:
            axis = ", ".join(_num(a, 2) for a in f.axis)
            feats.append(
                f"{f.kind} {f.name.replace('_', ' ')} (radius {_num(1000 * f.radius)} mm, depth "
                f"{_num(1000 * f.depth)} mm, opening toward ({axis}), mouth at height "
                f"{_num(1000 * (f.origin[2] + part.size[2] / 2))} mm)"
            )
        lines.append("- Features: " + "; ".join(feats) + ".")
    else:
        lines.append("- Features: none (plain faces).")
    lines.append(_needed_line(part))
    tip = 1000.0 * task.controller.min_tip_clearance
    lines.append(
        f"- Reach: the digits must reach at least {_num(sz + PALM_CLEARANCE_MM)} mm - grip height below the palm "
        f"(grip height = max({_num(tip)}, grip_height_frac x {_num(sz)}) mm), and grip_height_frac must stay at "
        f"or below {_num(max(0.0, (sz - MIN_OVERLAP_MM) / sz), 2)}."
    )
    return "\n".join(lines)


def render_parts(parts: Sequence[PartSpec], task: TaskV2, min_valid: int) -> str:
    if len(parts) == 1:
        return "Every design is built for this part and tested on it.\n\n" + render_part(parts[0], task)
    head = (
        f"Every design is tested on all {len(parts)} parts below with one recipe (the same effector, the same "
        "grip_height_frac and yaw_deg on every part). A design must pass the validators on at least "
        f"{min_valid} of them; on the others it is not built and scores nothing. The goal is one effector that "
        "holds as many of these parts as possible."
    )
    blocks = [head]
    for part in parts:
        blocks.append(f"## {part.name}\n\n{render_part(part, task)}")
    return "\n\n".join(blocks)


def _range_text(p: ParamSpec) -> str:
    if p.kind == "choice":
        return ", ".join(p.choices)
    if p.kind == "bool":
        return "true, false"
    assert p.low is not None and p.high is not None
    if p.kind == "int":
        return f"{int(p.low)}-{int(p.high)}"
    return f"{_num(p.low, 2)}-{_num(p.high, 2)}"


def render_generators(spaces: Mapping[SpaceKey, Sequence[ParamSpec]]) -> str:
    blocks = []
    for key, space in spaces.items():
        rows = [
            f"## {space_label(key)}",
            "",
            kind_summary(key),
            "",
            "| parameter | range | unit | what it does |",
            "|---|---|---|---|",
        ]
        for p in space:
            rows.append(f"| {p.name} | {_range_text(p)} | {p.unit or '-'} | {p.description} |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def render_rules(task: TaskV2) -> str:
    tip = _num(1000 * task.controller.min_tip_clearance)
    slender = ", ".join(f"{m} {_num(v)}" for m, v in MAX_SLENDERNESS.items())
    feature = ", ".join(f"{m} {_num(v)} mm" for m, v in MIN_FEATURE_MM.items())
    return "\n".join(
        [
            "- Bounds: every parameter of the generator, inside its range or one of its choices, nothing else.",
            (
                f"- Build: minimum feature {feature}; a phalanx at most {slender} times as long as it is thick, at "
                "least 8 mm long and at least as long as it is thick; a round pad at most half its phalanx's length "
                "and half the finger's width; a hook lip at least (pad + hook depth) / 2.5 thick and no taller than "
                "half the finger; tentacle segments at least twice the minimum feature across and at least the root "
                "radius + 2 mm long."
            ),
            (
                "- Open pose: digit footprints may not overlap on the palm (a digit's footprint runs from its "
                "innermost point, pad and hook included, to its back face, across its width). Digit directions: "
                "even layout 90 + 360 i / n deg (i = 0, 1, ...); opposed layout: the thumb at -90 deg and the others "
                "spread over spread_deg centred on 90 deg; yaw_deg is added to all of them."
            ),
            (
                f"- Mass at most {_num(MAX_EFFECTOR_MASS_KG)} kg. At most {MAX_ACTUATORS} servos (per_joint: one per "
                "joint; tendon, tentacles: one per digit; hook gripper: one per finger). Closing force: each digit "
                "gets grip_force_n / digits at its contact lever (0.75 x finger length for hands, 0.85 for hook "
                f"grippers, 0.6 for tentacles); that torque may not exceed {_num(MAX_JOINT_TORQUE_NM)} N m, and the "
                f"total is at most {_num(MAX_TOTAL_FORCE_N)} N."
            ),
            (
                "- Opening: open, every digit's innermost point (palm_radius_mm minus half the finger thickness, the "
                "pad and the hook; tentacles: palm_radius_mm minus the root radius) must be at least the part's "
                "'Opening needed' value for that digit's direction: the outline's reach within "
                f"+-{_num(YAW_ENVELOPE_DEG)} deg of the direction plus {_num(ERROR_ENVELOPE_MM)} mm of placement error "
                f"and {_num(MIN_OPEN_CLEARANCE_MM)} mm. The validators use the exact direction, so between two listed "
                "directions take the larger value."
            ),
            (
                f"- Reach: the lowest point goes to max({tip} mm, grip_height_frac x part height); the digits must "
                "reach that far below the palm with the palm's underside at least "
                f"{_num(PALM_CLEARANCE_MM)} mm above the part's top (finger_length_mm >= part height - grip height + "
                f"{_num(PALM_CLEARANCE_MM)} mm), and the lowest point must be at least {_num(MIN_OVERLAP_MM)} mm below "
                "the part's top."
            ),
            (
                "- Closing: posed over the centred part and closed along its synergy, at least two effector bodies "
                "must touch the part, no two digits may run into each other before both reach it, and the digits "
                "that reach it must include two at least 90 deg apart."
            ),
        ]
    )


def render_notes(parts: Sequence[PartSpec], task: TaskV2) -> str:
    mu = task.physics.friction
    if len(parts) == 1:
        weight = parts[0].mass * 9.81
        squeeze = f"This part weighs {_num(weight, 2)} N, so the digits must squeeze it with at least {_num(weight / mu)} N"
    else:
        squeeze = (
            "The squeeze each part needs (weight / friction): "
            + ", ".join(f"{p.name} {_num(p.mass * 9.81 / mu)} N" for p in parts)
            + "; the digits must squeeze at least that"
        )
    return prompts.render("notes.md", mu=_num(mu, 2), squeeze=squeeze)


def _outcome_text(o: DesignOutcome) -> str:
    n = o.n_episodes
    held = f"{round(o.held_rate * n)}/{n}" if n else f"{_num(100 * o.held_rate, 0)} %"
    reg = f"{round(o.registered_rate * n)}/{n}" if n else f"{_num(100 * o.registered_rate, 0)} %"
    modes = ", ".join(f"{k} {v}" for k, v in sorted(o.failure_modes.items(), key=lambda kv: (-kv[1], kv[0])))
    text = f"held {held}, registered {reg}; failures: {modes or 'none'}"
    if o.per_part:
        text += "; per part (held, registered): " + ", ".join(
            f"{name} {_num(100 * h, 0)} %/{_num(100 * r, 0)} %" for name, (h, r) in o.per_part.items()
        )
    return text


def render_history(history: Sequence[DesignOutcome]) -> str:
    if not history:
        return "None yet: this is the first round."
    lines = []
    for i, o in enumerate(history, start=1):
        r = o.recipe
        params = wire_json(_json_safe(dict(r.params)))
        idea = scrub(r.hypothesis)[:_MAX_QUOTED] if r.hypothesis else ""
        lines.append(f"- t{i} {space_label((r.family, r.kind))} [{r.origin}]: {_outcome_text(o)}")
        lines.append(f"  params {params}")
        if idea:
            lines.append(f"  idea: {idea}")
    return "\n".join(lines)


def render_diversity(min_points: float) -> str:
    return (
        "Distances are in points: within one generator each number's range is 100 points and a different choice "
        "counts 100 points; designs of different generators are always far apart. Unused parameters (spread_deg "
        "of an even layout or of a two-digit design, proximal_frac with one phalanx, pad_thickness_mm without a "
        "pad, hook_height_mm without a hook) are reset to their defaults and numbers are rounded to 0.5 mm, 1 deg, "
        f"0.5 N or 0.01 before comparing. Every new design must be at least {_num(min_points, 0)} points from every "
        "tested design and from the other new designs, or it is refused as too similar; an identical one is "
        "refused as a duplicate."
    )


def build_messages(
    parts: PartSpec | Sequence[PartSpec],
    history: Sequence[DesignOutcome],
    n: int,
    spaces: Mapping[SpaceKey, Sequence[ParamSpec]],
    task: TaskV2,
    *,
    min_points: float = MIN_POINTS,
    min_valid: int | None = None,
) -> list[dict[str, str]]:
    """The system and user messages for one design round on ``parts`` (one part or several)."""
    plist = _as_parts(parts)
    user = prompts.render(
        "user.md",
        parts_title="The part" if len(plist) == 1 else "The parts",
        part=render_parts(plist, task, min_valid or _default_min_valid(len(plist))),
        task=render_task(task),
        generators=render_generators(spaces),
        rules=render_rules(task),
        notes=render_notes(plist, task),
        history=render_history(history),
        diversity=render_diversity(min_points),
        n=n,
    )
    return [{"role": "system", "content": prompts.render("system.md")}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- client


class ScrubbingCache(ResponseCache):
    """A response cache that stores what a replay needs and nothing else.

    Before a live response is written, reasoning text (``reasoning_content``/``reasoning``) is
    dropped (its token count stays in ``usage``), and the free text the model wrote (content,
    the tool call's ``analysis`` and each ``hypothesis``) passes through :func:`scrub`. Params
    are never touched, and :func:`propose_recipes` scrubs the same text before it uses it, so a
    replay from this cache gives exactly the batch the live run gave. Requests are stored as
    sent (they are our own prompts, which :func:`assert_clean` checked).
    """

    def put(self, entry: CacheEntry) -> Path | None:
        return super().put(replace(entry, response=_scrub_response(entry.response)))


def _scrub_args(args: Any) -> Any:
    """Tool arguments with the model's free text (``analysis``, each ``hypothesis``) scrubbed."""
    if not isinstance(args, Mapping):
        return args
    out = dict(args)
    if isinstance(out.get("analysis"), str):
        out["analysis"] = scrub(out["analysis"])
    designs = out.get("designs")
    if isinstance(designs, list):
        out["designs"] = [
            {**d, "hypothesis": scrub(d["hypothesis"])}
            if isinstance(d, Mapping) and isinstance(d.get("hypothesis"), str)
            else d
            for d in designs
        ]
    return out


def _scrub_response(response: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(response))
    for choice in out.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        for key in ("reasoning_content", "reasoning"):
            message.pop(key, None)
        if isinstance(message.get("content"), str):
            message["content"] = _scrub_words(message["content"])
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
                continue
            try:
                args = json.loads(fn["arguments"])
            except json.JSONDecodeError:  # kept as sent (a replay must fail to parse it the same way)
                fn["arguments"] = _scrub_words(fn["arguments"])
                continue
            fn["arguments"] = json.dumps(_scrub_args(args))
    return out


def make_designer_client(
    cache_dir: str | Path | None,
    *,
    cache_mode: CacheMode = "readwrite",
    ledger_path: str | Path | None = None,
    budget_usd: float | None = None,
    transport: Any | None = None,
    env_file: str | Path | None = None,
    limiter: TokenBucket | None = None,
    cache: ResponseCache | None = None,
) -> LLMClient:
    """The designer's client: Nemotron 3 Super, :data:`DESIGNER_SAMPLING`, cache and ledger.

    ``env_file`` names the ``.env`` holding ``NEBIUS_API_KEY`` (read only on the first live
    call); with ``cache_mode="replay"`` no key is needed and a miss is an error.
    """
    if transport is None and env_file is not None:
        transport = _LazyTransport(Path(env_file))
    if cache is None and cache_dir is not None:
        cache = ScrubbingCache(cache_dir, mode=cache_mode)
    return LLMClient(
        DESIGNER_MODEL,
        temperature=float(DESIGNER_SAMPLING["temperature"]),
        top_p=float(DESIGNER_SAMPLING["top_p"]),
        max_tokens=int(DESIGNER_SAMPLING["max_tokens"]),
        cache=cache,
        ledger=Ledger(ledger_path, budget_usd=budget_usd),
        transport=transport,
        timeout_s=300.0,
        retry=RetryPolicy(max_retries=3, base_delay_s=2.0),
        limiter=limiter,
    )


class _LazyTransport:
    """Builds the real transport from an explicit ``.env`` on the first live call."""

    def __init__(self, env_file: Path) -> None:
        self._env_file = env_file
        self._inner: OpenAITransport | None = None

    def _get(self) -> OpenAITransport:
        if self._inner is None:
            self._inner = OpenAITransport.from_env(self._env_file)
        return self._inner

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> Any:
        return self._get().chat_completions(payload, timeout_s)

    def list_models(self, timeout_s: float) -> Any:
        return self._get().list_models(timeout_s)

    def __repr__(self) -> str:
        return "LazyTransport(env_file=<set>)"


# --------------------------------------------------------------------------- proposing


@dataclass
class _Slot:
    index: int  # 1-based
    recipe: MorphologyRecipe | None = None
    reason: str | None = None
    errors: tuple[str, ...] = ()
    kind: str | None = None
    params: Mapping[str, Any] | None = None


def _record(stage: str, meta: CallMeta | None, error: str | None) -> CallRecord:
    return CallRecord(
        stage=stage,
        request_hash=meta.request_hash if meta else None,
        cached=bool(meta and meta.cached),
        ok=error is None,
        error=error,
        prompt_tokens=meta.prompt_tokens if meta else 0,
        completion_tokens=meta.completion_tokens if meta else 0,
        reasoning_tokens=(meta.reasoning_tokens or 0) if meta else 0,
        cost_usd=meta.cost_usd if meta else 0.0,
        latency_s=meta.latency_s if meta else 0.0,
        finish_reason=meta.finish_reason if meta else None,
        notes=meta.notes if meta else (),
    )


def _call(
    client: LLMClient, messages: list[dict[str, str]], tool: dict[str, Any], stage: str, tag: str
) -> tuple[dict[str, Any] | None, CallRecord, str | None]:
    """One forced call; returns (arguments or None, record, failure reason or None)."""
    assert_clean(messages)
    try:
        args, meta = client.call_tool(messages, tool, force=True, sample=0, tag=f"{tag}/{stage}")
    except (LLMAuthError, LLMBillingError):
        raise
    except LLMParseError as exc:
        return None, _record(stage, exc.meta, f"parse_error: {exc}"), "parse_error"
    except BudgetExceededError as exc:
        return None, _record(stage, None, f"spend_cap: {exc}"), "spend_cap"
    except (LLMHTTPError, LLMTransportError) as exc:
        return None, _record(stage, None, f"call_failed: {exc}"), "call_failed"
    return args, _record(stage, meta, None), None


def _parse_design(
    raw: Any, spaces: Mapping[SpaceKey, Sequence[ParamSpec]]
) -> tuple[MorphologyRecipe | None, list[str], str | None, Mapping[str, Any] | None]:
    """(recipe or None, errors, kind label, raw params) of one design object."""
    if not isinstance(raw, Mapping):
        return None, ["the design is not an object with hypothesis, kind and params"], None, None
    label = raw.get("kind")
    params = raw.get("params")
    params_view = _json_safe(dict(params)) if isinstance(params, Mapping) else None
    by_label = {space_label(k): k for k in spaces}
    if not isinstance(label, str) or label not in by_label:
        return (
            None,
            [f"out_of_bounds: unknown generator {label!r} (offered: {', '.join(by_label)})"],
            None,
            params_view,
        )
    if not isinstance(params, Mapping):
        return (
            None,
            ["out_of_bounds: params must be an object with every parameter of the generator"],
            label,
            None,
        )
    hypothesis = raw.get("hypothesis")
    hyp = scrub(hypothesis)[:_MAX_HYPOTHESIS] if isinstance(hypothesis, str) else ""
    fam, kind = by_label[label]
    try:
        recipe = MorphologyRecipe(fam, kind, dict(params), hypothesis=hyp, origin="nemotron")
    except (TypeError, ValueError) as exc:
        return None, [f"out_of_bounds: {exc}"], label, params_view
    return canonical_recipe(recipe), [], label, params_view


def _as_parts(parts: PartSpec | Sequence[PartSpec]) -> list[PartSpec]:
    from rtt.parts.types import PartSpec as _PartSpec

    plist = [parts] if isinstance(parts, _PartSpec) else list(parts)
    if not plist:
        raise ValueError("at least one part is needed")
    return plist


def _default_min_valid(n_parts: int) -> int:
    """A design for several parts must be buildable on at least two thirds of them."""
    return n_parts if n_parts == 1 else math.ceil(2 * n_parts / 3)


def recipe_problems(
    recipe: MorphologyRecipe, parts: Sequence[PartSpec], task: TaskV2, min_valid: int
) -> list[str]:
    """Validator problems of ``recipe`` across ``parts``: ``[]`` when it is buildable on at least
    ``min_valid`` of them, else the first problem on each part where it is not."""
    failing = []
    for part in parts:
        problems = validate_recipe(recipe, part, task)
        if problems:
            failing.append((part.name, problems))
    if len(parts) - len(failing) >= min_valid:
        return []
    if len(parts) == 1:
        return list(failing[0][1])
    head = f"buildable on {len(parts) - len(failing)} of {len(parts)} parts (needs {min_valid})"
    return [head] + [f"{name}: {problems[0]}" for name, problems in failing]


def _judge(
    slot: _Slot,
    raw: Any,
    spaces: Mapping[SpaceKey, Sequence[ParamSpec]],
    parts: Sequence[PartSpec],
    task: TaskV2,
    min_valid: int,
    references: list[MorphologyRecipe],
    labels: list[str],
    min_points: float,
) -> None:
    recipe, errors, kind, params = _parse_design(raw, spaces)
    slot.kind, slot.params = kind, params
    if recipe is None:
        slot.recipe, slot.reason, slot.errors = None, "invalid", tuple(errors)
        return
    problems = recipe_problems(recipe, parts, task, min_valid)
    if problems:
        slot.recipe, slot.reason, slot.errors = None, "invalid", tuple(problems)
        return
    verdict = screen(recipe, references, labels=labels, min_points=min_points)
    if verdict is not None:
        slot.recipe, slot.reason, slot.errors = None, verdict.reason, (verdict.message,)
        return
    slot.recipe, slot.reason, slot.errors = recipe, None, ()


def _designs(args: Mapping[str, Any] | None) -> list[Any]:
    if not isinstance(args, Mapping):
        return []
    designs = args.get("designs")
    return list(designs) if isinstance(designs, list) else []


def propose_recipes(
    part: PartSpec | Sequence[PartSpec],
    history: Sequence[DesignOutcome],
    n: int,
    *,
    client: LLMClient,
    kinds: Sequence[str | SpaceKey] | None = None,
    task: TaskV2 | None = None,
    repairs: int = 1,
    min_points: float = MIN_POINTS,
    min_valid_parts: int | None = None,
    tag: str | None = None,
) -> ProposalBatch:
    """Ask Nemotron for ``n`` designs for ``part`` given the measured ``history`` (module docstring).

    ``part`` is one part (the tournament) or several (one effector for all of them, buildable on
    at least ``min_valid_parts``, by default two thirds of them).
    """
    from rtt.sim.task import default_task_v2

    if n < 1:
        raise ValueError("n must be >= 1")
    if repairs not in (0, 1):
        raise ValueError("at most one repair call per batch")
    parts = _as_parts(part)
    min_valid = min_valid_parts or _default_min_valid(len(parts))
    if not 1 <= min_valid <= len(parts):
        raise ValueError(f"min_valid_parts must be in [1, {len(parts)}]")
    task = task or default_task_v2()
    spaces = select_spaces(kinds)
    tool = submit_morphologies_tool(spaces, n)
    messages = build_messages(parts, history, n, spaces, task, min_points=min_points, min_valid=min_valid)
    part_id = parts[0].name if len(parts) == 1 else "+".join(p.name for p in parts)
    tag = tag or f"designer/{part_id}/after{len(history)}"
    ledger_start = len(client.ledger.session_entries())
    tested = [o.recipe for o in history]
    tested_labels = [f"t{i}" for i in range(1, len(tested) + 1)]

    slots = [_Slot(i) for i in range(1, n + 1)]
    rejections: list[Rejection] = []
    calls: list[CallRecord] = []

    def judge_all(stage: str, args: Mapping[str, Any] | None, which: list[_Slot]) -> None:
        designs = _designs(args)
        for slot in which:
            accepted = [s.recipe for s in slots if s.recipe is not None]
            accepted_labels = [f"design {s.index}" for s in slots if s.recipe is not None]
            raw = designs[slot.index - 1] if slot.index - 1 < len(designs) else None
            if raw is None:
                slot.recipe, slot.reason = None, "invalid"
                slot.errors = (f"the call returned {len(designs)} designs; design {slot.index} is missing",)
            else:
                _judge(
                    slot,
                    raw,
                    spaces,
                    parts,
                    task,
                    min_valid,
                    tested + accepted,
                    tested_labels + accepted_labels,
                    min_points,
                )
            if slot.recipe is None:
                rejections.append(
                    Rejection(
                        slot.index, stage, slot.reason or "invalid", slot.errors, slot.kind, slot.params
                    )
                )

    args, record, failure = _call(client, messages, tool, "main", tag)
    calls.append(record)
    analysis = args.get("analysis") if isinstance(args, Mapping) else None
    if failure is not None:
        rejections.append(Rejection(0, "main", failure, (record.error or failure,)))
        for slot in slots:
            slot.reason, slot.errors = failure, (record.error or failure,)
    else:
        judge_all("main", args, slots)

    broken = [s for s in slots if s.recipe is None]
    if broken and repairs and failure != "spend_cap":
        # the scrubbed text the cache stores, so a replay rebuilds this prompt byte for byte
        previous = (
            wire_json(_json_safe(_scrub_args(args))) if isinstance(args, Mapping) else "(no usable call)"
        )
        problems = "\n".join(
            f"- design {s.index}: " + "; ".join(scrub(e)[:400] for e in s.errors[:7]) for s in broken
        )
        kept = [f"design {s.index}" for s in slots if s.recipe is not None]
        text = prompts.render(
            "repair.md",
            previous=scrub(previous),
            problems=problems,
            n=n,
            keep=(", ".join(kept) if kept else "none of them"),
        )
        repair_messages = [dict(m) for m in messages]
        repair_messages[-1]["content"] = f"{repair_messages[-1]['content']}\n\n{text}"
        args2, record2, failure2 = _call(client, repair_messages, tool, "repair", tag)
        calls.append(record2)
        if failure2 is not None:
            rejections.append(Rejection(0, "repair", failure2, (record2.error or failure2,)))
        else:
            judge_all("repair", args2, broken)
            if isinstance(args2, Mapping) and isinstance(args2.get("analysis"), str) and not analysis:
                analysis = args2["analysis"]

    ledger = tuple(client.ledger.session_entries()[ledger_start:])
    return ProposalBatch(
        part_id=part_id,
        n_requested=n,
        recipes=tuple(s.recipe for s in slots if s.recipe is not None),
        rejections=tuple(rejections),
        calls=tuple(calls),
        ledger=ledger,
        analysis=scrub(analysis)[:_MAX_ANALYSIS] if isinstance(analysis, str) else None,
        kinds=tuple(space_label(k) for k in spaces),
        template_hashes=prompts.template_hashes(),
    )


def outcome_json(o: DesignOutcome) -> dict[str, Any]:
    """A JSON-ready form of a :class:`DesignOutcome` (manifests)."""
    return {
        "recipe": o.recipe.to_dict(),
        "held_rate": o.held_rate,
        "registered_rate": o.registered_rate,
        "failure_modes": dict(o.failure_modes),
        "n_episodes": o.n_episodes,
    }


def dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), indent=2, sort_keys=True)
