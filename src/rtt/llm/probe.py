"""Probe the NVIDIA Nemotron models on Token Factory for the proposer role.

For each model and output mode (forced tool call, strict ``json_schema``) the
probe asks for two gripper finger designs from a short parameter table, a few
times, and records: valid-parse rate, bounds and rule violations, latency,
token usage, whether reasoning comes back in a separate field, and errors.

Every response is cached (``runs/probe_llm/cache``) and every call is ledgered
(``runs/probe_llm/ledger.jsonl``), so re-running the probe regenerates the
report (``docs/PROBE_LLM.md``) from cache at no cost. A hard spend cap covers
all live calls ever recorded in that ledger; a call whose worst case could
cross it is skipped.

Usage::

    rtt probe-llm                       # print the summary of docs/PROBE_LLM.md (no calls)
    rtt probe-llm --replay              # rebuild the report from the cache only (no key needed)
    rtt probe-llm --live [--calls 2] [--cap-usd 0.15] [--thinking-off]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rtt.llm import (
    MODELS,
    NEMOTRON_LIGHTNING,
    NEMOTRON_NANO,
    NEMOTRON_SUPER,
    NEMOTRON_ULTRA,
    BudgetExceededError,
    CacheMiss,
    CallMeta,
    Ledger,
    LLMAuthError,
    LLMBillingError,
    LLMClient,
    LLMError,
    LLMHTTPError,
    LLMParseError,
    ResponseCache,
    RetryPolicy,
)
from rtt.llm.ledger import read_ledger, summarize

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "runs" / "probe_llm"
DEFAULT_DOC = ROOT / "docs" / "PROBE_LLM.md"
MODEL_ORDER = (NEMOTRON_NANO, NEMOTRON_LIGHTNING, NEMOTRON_SUPER, NEMOTRON_ULTRA)
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 2500
N_DESIGNS = 2

# Simplified design space (bounds mirror the build plan's design schema).
NUMERIC: dict[str, tuple[float, float]] = {
    "finger_length_mm": (40, 120),
    "finger_thickness_mm": (4, 16),
    "finger_width_mm": (10, 30),
    "bend_angle_deg": (-15, 45),
    "bend_start_frac": (0.2, 0.9),
    "inner_face_depth_mm": (0, 8),
    "tip_size_mm": (0, 15),
    "taper_ratio": (0.5, 1.0),
}
ENUMS: dict[str, tuple[str, ...]] = {
    "inner_face": ("flat", "concave_arc", "v_groove"),
    "tip_profile": ("blunt", "rounded", "chamfer", "hook"),
}
FIELD_ORDER = (
    "finger_length_mm",
    "finger_thickness_mm",
    "finger_width_mm",
    "bend_angle_deg",
    "bend_start_frac",
    "inner_face",
    "inner_face_depth_mm",
    "tip_profile",
    "tip_size_mm",
    "taper_ratio",
    "rationale",
)


def design_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name in FIELD_ORDER:
        if name in NUMERIC:
            lo, hi = NUMERIC[name]
            properties[name] = {"type": "number", "minimum": lo, "maximum": hi}
        elif name in ENUMS:
            properties[name] = {"type": "string", "enum": list(ENUMS[name])}
        else:
            properties[name] = {"type": "string", "description": "One sentence: why this design."}
    return {
        "type": "object",
        "properties": properties,
        "required": list(FIELD_ORDER),
        "additionalProperties": False,
    }


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string", "description": "At most two sentences on the current results."},
            "designs": {
                "type": "array",
                "items": design_schema(),
                "minItems": N_DESIGNS,
                "maxItems": N_DESIGNS,
            },
        },
        "required": ["analysis", "designs"],
        "additionalProperties": False,
    }


SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_designs",
        "description": f"Submit exactly {N_DESIGNS} gripper finger designs for simulation.",
        "parameters": response_schema(),
    },
}

PARAMETER_TABLE = """| parameter | range / options | meaning |
|---|---|---|
| finger_length_mm | 40-120 | root to tip |
| finger_thickness_mm | 4-16 | along the closing direction |
| finger_width_mm | 10-30 | across the finger |
| bend_angle_deg | -15..45 | inward bend of the distal part (positive = toward the object) |
| bend_start_frac | 0.2-0.9 | where the bend starts, as a fraction of length from the root |
| inner_face | flat / concave_arc / v_groove | contact face profile |
| inner_face_depth_mm | 0-8 | recess depth of the face profile |
| tip_profile | blunt / rounded / chamfer / hook | fingertip shape |
| tip_size_mm | 0-15 | size of the tip feature |
| taper_ratio | 0.5-1.0 | tip thickness / root thickness |"""

SYSTEM_TOOL = (
    "You design fingers for a two-finger parallel gripper that is tested in physics simulation. "
    "Stay strictly inside the parameter ranges. Respond only by calling the submit_designs function."
)
SYSTEM_JSON = (
    "You design fingers for a two-finger parallel gripper that is tested in physics simulation. "
    "Stay strictly inside the parameter ranges. Respond only with a JSON object matching the schema."
)

USER_PROMPT = f"""Task: pick up an object from a table, lift it 10 cm and hold it for 2 s.
Gripper: each finger slides 0-40 mm (80 mm max opening), grip force 40 N, PLA fingers.
Objects: boxes, cylinders and capsules, 20-60 mm wide, 700 kg/m^3, friction 0.8.

Design parameters:
{PARAMETER_TABLE}

Rules: inner_face_depth_mm must be 0 when inner_face is flat, and at most half of finger_thickness_mm
otherwise. A hook tip may not protrude more than finger_thickness_mm + 6 mm.

Results so far (training success over 36 episodes, failure counts):
1. concave: concave_arc face 4 mm deep, 0 deg bend, blunt tip, 80 mm long, 10 mm thick -> 51%;
   slip_during_lift 11, missed_grasp 6
2. straight: flat face, 0 deg bend, blunt tip, 80 mm long, 10 mm thick -> 42%;
   slip_during_lift 14, missed_grasp 7
3. hooked: flat face, 15 deg bend, hook tip 8 mm -> 33%; ejected_on_close 18, missed_grasp 6

Target: beat 61% (1.2 x the current best).
Give a short analysis, then exactly {N_DESIGNS} new designs: one mutation of the best design and one
exploratory design that differs clearly from all three."""


# --------------------------------------------------------------------------- checking


@dataclass
class Check:
    structure_ok: bool
    bounds_violations: list[str] = field(default_factory=list)
    rule_violations: list[str] = field(default_factory=list)


def check_response(value: dict[str, Any]) -> Check:
    """Structural validity, bounds and cross-field rules of a parsed response."""
    designs = value.get("designs")
    if not isinstance(designs, list) or len(designs) != N_DESIGNS:
        return Check(False)
    structure_ok = isinstance(value.get("analysis"), str)
    bounds: list[str] = []
    rules: list[str] = []
    for i, design in enumerate(designs):
        if not isinstance(design, dict) or set(design) != set(FIELD_ORDER):
            structure_ok = False
            if not isinstance(design, dict):
                continue
        for name, (lo, hi) in NUMERIC.items():
            x = design.get(name)
            if not isinstance(x, int | float) or isinstance(x, bool):
                structure_ok = False
            elif not lo <= x <= hi:
                bounds.append(f"d{i}.{name}={x}")
        for name, options in ENUMS.items():
            x = design.get(name)
            if not isinstance(x, str):
                structure_ok = False
            elif x not in options:
                bounds.append(f"d{i}.{name}={x!r}")
        if not isinstance(design.get("rationale"), str):
            structure_ok = False
        face, depth, thick = (
            design.get("inner_face"),
            design.get("inner_face_depth_mm"),
            design.get("finger_thickness_mm"),
        )
        if isinstance(depth, int | float) and isinstance(thick, int | float):
            if face == "flat" and depth != 0:
                rules.append(f"d{i}: flat face with depth {depth}")
            if face != "flat" and depth > 0.5 * thick:
                rules.append(f"d{i}: depth {depth} > thickness/2")
            tip, size = design.get("tip_profile"), design.get("tip_size_mm")
            if tip == "hook" and isinstance(size, int | float) and size > thick + 6:
                rules.append(f"d{i}: hook {size} > thickness+6")
    return Check(structure_ok, bounds, rules)


# --------------------------------------------------------------------------- running


@dataclass
class CallResult:
    model: str
    mode: str
    sample: int
    status: str  # ok | parse_error | http_error | skipped_budget | error
    structure_ok: bool = False
    bounds_violations: list[str] = field(default_factory=list)
    rule_violations: list[str] = field(default_factory=list)
    free_text_content: bool = False  # unparseable answer that did not even start with "{"
    error: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.structure_ok


def run_one(client: LLMClient, mode: str, sample: int) -> CallResult:
    base_mode = "tool" if mode.startswith("tool") else "json_schema"
    system = SYSTEM_TOOL if base_mode == "tool" else SYSTEM_JSON
    messages = [{"role": "system", "content": system}, {"role": "user", "content": USER_PROMPT}]
    tag = f"probe/{mode}/{sample}"
    result = CallResult(client.model, mode, sample, status="error")
    meta: CallMeta | None = None
    try:
        if base_mode == "tool":
            value, meta = client.call_tool(messages, SUBMIT_TOOL, force=True, sample=sample, tag=tag)
        else:
            value, meta = client.call_json_schema(
                messages, response_schema(), name="submit_designs", sample=sample, tag=tag
            )
    except LLMParseError as exc:
        meta = exc.meta
        result.status, result.error = "parse_error", str(exc)
        result.free_text_content = bool(exc.raw_text.strip()) and not exc.raw_text.lstrip().startswith("{")
    except BudgetExceededError as exc:
        result.status, result.error = "skipped_budget", str(exc)
    except (LLMAuthError, LLMBillingError):
        raise
    except LLMHTTPError as exc:
        result.status, result.error = "http_error", str(exc)
    except LLMError as exc:
        result.status, result.error = "error", f"{type(exc).__name__}: {exc}"
    else:
        check = check_response(value)
        result.status = "ok"
        result.structure_ok = check.structure_ok
        result.bounds_violations = check.bounds_violations
        result.rule_violations = check.rule_violations
    if meta is not None:
        result.meta = meta.to_json()
        result.meta["reasoning_chars"] = len(meta.reasoning_text or "")
    return result


def capture_limits(model: str, cache: ResponseCache, ledger: Ledger) -> dict[str, str]:
    """One tiny cached call to read the model's ``x-ratelimit-*`` headers ({} on failure)."""
    client = LLMClient(
        model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        cache=cache,
        ledger=ledger,
        timeout_s=60.0,
        retry=RetryPolicy(max_retries=2, base_delay_s=2.0),
    )
    try:
        _, meta = client.chat([{"role": "user", "content": "Reply with: ok"}], tag="probe/limits")
    except (LLMAuthError, LLMBillingError):
        raise
    except (LLMError, BudgetExceededError, CacheMiss):
        return {}
    return dict(meta.rate_limit_headers)


# --------------------------------------------------------------------------- report


def _short(model: str) -> str:
    return model.split("/", 1)[1]


def _fmt_s(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def _mean_int(values: list[int]) -> str:
    return f"{statistics.fmean(values):.0f}" if values else "-"


def cell_rows(results: list[CallResult]) -> list[dict[str, Any]]:
    """One aggregate row per (model, mode), in model order."""
    rows = []
    keys = sorted({(r.model, r.mode) for r in results}, key=lambda k: (MODEL_ORDER.index(k[0]), k[1]))
    for model, mode in keys:
        cell = [r for r in results if r.model == model and r.mode == mode]
        answered = [r for r in cell if r.meta is not None]
        rows.append(
            {
                "model": model,
                "mode": mode,
                "n": len(cell),
                "measured": sum(r.status != "skipped_budget" for r in cell),
                "skipped": sum(r.status == "skipped_budget" for r in cell),
                "valid": sum(r.valid for r in cell),
                "clean": sum(r.valid and not r.bounds_violations and not r.rule_violations for r in cell),
                "bounds": sum(len(r.bounds_violations) for r in cell),
                "rules": sum(len(r.rule_violations) for r in cell),
                "truncated": sum(r.meta["finish_reason"] == "length" for r in answered),
                "free_text": sum(r.free_text_content for r in cell),
                "http_errors": sum(r.status in ("http_error", "error") for r in cell),
                "latency": [float(r.meta["latency_s"]) for r in answered],
                "prompt": [int(r.meta["prompt_tokens"]) for r in answered],
                "completion": [int(r.meta["completion_tokens"]) for r in answered],
                "reasoning_tokens": [
                    int(r.meta["reasoning_tokens"])
                    for r in answered
                    if r.meta["reasoning_tokens"] is not None
                ],
                "reasoning_separate": sorted({bool(r.meta["reasoning_separate"]) for r in answered}),
                "reasoning_fields": sorted(
                    {r.meta["reasoning_field"] for r in answered if r.meta["reasoning_field"]}
                ),
                "cost": sum(float(r.meta["cost_usd"]) for r in answered),
                "errors": sorted({_short_error(r) for r in cell if r.status not in ("ok", "skipped_budget")}),
                "notes": sorted({n for r in answered for n in r.meta["notes"]}),
                "finish": sorted({str(r.meta["finish_reason"]) for r in answered}),
            }
        )
    return rows


def _short_error(r: CallResult) -> str:
    text = (r.error or r.status).replace("|", "/").replace("\n", " ")
    return f"{r.status}: {text[:140]}"


def _all_clean(row: dict[str, Any]) -> bool:
    return row["measured"] > 0 and row["clean"] == row["measured"]


def _avg_cost(row: dict[str, Any]) -> float:
    return row["cost"] / max(len(row["latency"]), 1)


def _rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    p50 = statistics.median(row["latency"]) if row["latency"] else float("inf")
    return (
        -row["clean"] / max(row["measured"], 1),
        -row["valid"] / max(row["measured"], 1),
        _avg_cost(row),
        p50,
    )


def recommend(rows: list[dict[str, Any]]) -> list[str]:
    """Recommendation paragraphs, derived from the table by fixed rules.

    The proposer must reason about prior results, so only modes with
    thinking left on are eligible. Super (the planned proposer) is chosen if
    one of its modes is fully clean; otherwise the best-ranked Super or Ultra
    cell. Ranking: clean rate, then valid rate, then cost, then latency.
    """
    eligible = [r for r in rows if r["measured"] and r["mode"] in ("tool", "json_schema")]
    if not eligible:
        return ["No model/mode was measured; no recommendation."]
    super_rows = sorted((r for r in eligible if r["model"] == NEMOTRON_SUPER), key=_rank)
    if super_rows and _all_clean(super_rows[0]):
        pick = super_rows[0]
    else:
        large = [r for r in eligible if r["model"] in (NEMOTRON_SUPER, NEMOTRON_ULTRA)] or eligible
        pick = min(large, key=_rank)
    out = [
        (
            f"**Proposer: `{pick['model']}` with `{pick['mode']}` mode, thinking on.** In this probe it was"
            f" {pick['clean']}/{pick['measured']} clean, p50 latency {_fmt_s(pick['latency'])} s, about"
            f" ${_avg_cost(pick):.4f} per call, with at most {max(pick['completion'], default=0)} completion tokens."
        ),
    ]
    others = [r for r in eligible if r is not pick and r["model"] == pick["model"]]
    for other in others:
        out.append(
            f"The other mode on the same model (`{other['mode']}`) was {other['clean']}/{other['measured']} clean"
            f" at ${_avg_cost(other):.4f} per call and p50 {_fmt_s(other['latency'])} s."
        )
    ultra = next(
        (
            r
            for r in sorted(eligible, key=_rank)
            if r["model"] == NEMOTRON_ULTRA and r["mode"] == pick["mode"]
        ),
        None,
    )
    if ultra is not None and pick["model"] != NEMOTRON_ULTRA and _avg_cost(pick) > 0:
        out.append(
            f"Ultra in the same mode was {ultra['clean']}/{ultra['measured']} clean at"
            f" {_avg_cost(ultra) / _avg_cost(pick):.1f}x the cost per call; it is not needed for format reliability."
        )
    no_think = next((r for r in rows if r["model"] == pick["model"] and r["mode"] == "tool+no_think"), None)
    if no_think is not None and no_think["measured"]:
        out.append(
            f"For latency-sensitive calls (the live-bout Director), `tool+no_think` on the same model was"
            f" {no_think['clean']}/{no_think['measured']} clean at p50 {_fmt_s(no_think['latency'])} s."
        )
    out.append(
        "Reasoning tokens count toward `max_tokens`, so production calls should leave generous headroom"
        f" above the largest completion seen here; {MAX_TOKENS} was the probe cap. Validators still run on"
        " every design (bounds, cross-field rules, geometry) with at most one repair call."
    )
    return out


def observations(rows: list[dict[str, Any]]) -> list[str]:
    """Plain-language findings derived from the aggregate rows."""
    out: list[str] = []
    by_key = {(r["model"], r["mode"]): r for r in rows}
    if not any(r["http_errors"] for r in rows):
        out.append(
            "No HTTP errors: every model accepted a forced `tool_choice` naming the function, a strict"
            " `response_format` json_schema"
            + (", and `chat_template_kwargs`" if any(r["mode"] == "tool+no_think" for r in rows) else "")
            + ". Accepted is not the same as enforced; see below."
        )
    for r in rows:
        if r["http_errors"]:
            out.append(f"`{_short(r['model'])}` {r['mode']}: {'; '.join(r['errors'])}.")
        if r["free_text"]:
            inline = r["reasoning_separate"] == [False]
            out.append(
                f"`{_short(r['model'])}` {r['mode']}: {r['free_text']}/{r['measured']} answers began with"
                " free-form text instead of JSON, so the strict schema was not enforced by the decoder for this"
                " model"
                + ("; its reasoning arrived inline in `content`, not in a separate field." if inline else ".")
            )
        if r["truncated"]:
            out.append(
                f"`{_short(r['model'])}` {r['mode']}: {r['truncated']}/{r['measured']} calls hit"
                f" max_tokens={MAX_TOKENS} (finish_reason `length`) before the answer was complete."
            )
    field_models: dict[str, list[str]] = {}
    for r in rows:
        for name in r["reasoning_fields"]:
            models = field_models.setdefault(name, [])
            if _short(r["model"]) not in models:
                models.append(_short(r["model"]))
    if field_models:
        parts = [
            f"`{name}` ({', '.join(f'`{m}`' for m in models)})"
            for name, models in sorted(field_models.items())
        ]
        out.append(
            "With thinking on, reasoning arrived outside `content` in "
            + " and ".join(parts)
            + ". The client reads either field, keeps the text in the cache and never parses it."
        )
    deltas = []
    for model in MODEL_ORDER:
        on, off = by_key.get((model, "tool")), by_key.get((model, "tool+no_think"))
        if on and off and on["completion"] and off["completion"]:
            deltas.append(
                f"`{_short(model)}` {_mean_int(on['completion'])} -> {_mean_int(off['completion'])} completion"
                f" tokens, p50 {_fmt_s(on['latency'])} -> {_fmt_s(off['latency'])} s,"
                f" clean {on['clean']}/{on['measured']} -> {off['clean']}/{off['measured']}"
            )
    if deltas:
        no_sep = all(
            by_key[(m, "tool+no_think")]["reasoning_separate"] == [False]
            for m in MODEL_ORDER
            if (m, "tool+no_think") in by_key
        )
        out.append(
            "`chat_template_kwargs: {enable_thinking: false}` (tool mode, thinking on -> off): "
            + "; ".join(deltas)
            + "."
            + (" No separate reasoning came back with thinking off." if no_sep else "")
        )
    prompt_pairs = [
        (by_key[(m, "tool")]["prompt"], by_key[(m, "json_schema")]["prompt"])
        for m in MODEL_ORDER
        if (m, "tool") in by_key and (m, "json_schema") in by_key
    ]
    prompt_pairs = [(a, b) for a, b in prompt_pairs if a and b]
    if prompt_pairs:
        tool_mean = statistics.fmean(x for a, _ in prompt_pairs for x in a)
        json_mean = statistics.fmean(x for _, b in prompt_pairs for x in b)
        out.append(
            f"Same messages, different billing: tool mode averaged {tool_mean:.0f} prompt tokens and json_schema"
            f" {json_mean:.0f}. The tool definition is rendered into the prompt; the `response_format` schema is not."
        )
    bounds = sum(r["bounds"] for r in rows)
    rules = sum(r["rules"] for r in rows)
    if bounds == 0 and rules == 0:
        out.append(
            "No parsed response broke a bound, an enum or a cross-field rule in any mode. The probe cannot tell"
            " decoder enforcement of min/max apart from plain compliance, so the validators remain mandatory."
        )
    else:
        out.append(f"Parsed responses broke {bounds} bounds/enums and {rules} cross-field rules in total.")
    return out


def write_report(
    path: Path,
    results: list[CallResult],
    limits: dict[str, dict[str, str]],
    ledger_path: Path,
    args: argparse.Namespace,
    stopped: str | None,
) -> None:
    rows = cell_rows(results)
    totals = summarize(read_ledger(ledger_path))
    live_spend = sum(t.cost_usd for t in totals.values())
    probe_spend = sum(r["cost"] for r in rows)
    when = datetime.now(UTC).strftime("%Y-%m-%d")
    has_no_think = any(r["mode"] == "tool+no_think" for r in rows)

    lines = [
        "# Token Factory model probe (proposer output mode)",
        "",
        (
            f"Generated by `rtt probe-llm --live` on {when} from its response cache and ledger;"
            " `rtt probe-llm --replay` rebuilds this file from cache at no cost."
        ),
        "",
        "## Setup",
        "",
        "- Provider: Nebius Token Factory, OpenAI-compatible `chat/completions`.",
        (
            f"- Sampling: temperature {TEMPERATURE}, top_p {TOP_P}, max_tokens {MAX_TOKENS}. There is no seed"
            " parameter, so repeated samples differ only by server-side sampling."
        ),
        (
            f"- Prompt: a short task description, a 10-parameter table, three prior results, and a request for"
            f" {N_DESIGNS} designs (one mutation, one exploratory) after a short analysis."
        ),
        "- Modes: `tool` = forced tool call (`tool_choice` names `submit_designs`); `json_schema` ="
        " `response_format` with `strict: true`. Both use the same JSON schema, with min/max on every numeric"
        " field and enums for the categorical ones."
        + (
            " `tool+no_think` = `tool` plus `chat_template_kwargs: {enable_thinking: false}`."
            if has_no_think
            else ""
        ),
        (
            f"- {args.calls} calls per model and mode, run one at a time. Cost is estimated from returned usage"
            " and catalogue prices."
        ),
        "",
        "## Results",
        "",
        (
            "Valid = parsed into exactly 2 designs with every field present and correctly typed. Clean = valid"
            " with no bounds or rule violations. Bounds = values outside the table ranges or enums. Rules ="
            " the depth/flat-face/hook cross-field rules stated in the prompt. Latency is the median per call;"
            " token counts are means. Reasoning tok is `-` when usage did not report it."
        ),
        "",
        (
            "| model | mode | valid | clean | bounds | rules | p50 latency s | prompt tok | completion tok |"
            " reasoning tok | separate reasoning | finish | est. $ | errors / notes |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sep = "/".join("yes" if s else "no" for s in r["reasoning_separate"]) or "-"
        extra = [*r["errors"], *r["notes"]]
        if r["skipped"]:
            extra.append(f"{r['skipped']} skipped by spend cap")
        lines.append(
            f"| `{_short(r['model'])}` | {r['mode']} | {r['valid']}/{r['measured']} |"
            f" {r['clean']}/{r['measured']} | {r['bounds']} | {r['rules']} | {_fmt_s(r['latency'])} |"
            f" {_mean_int(r['prompt'])} | {_mean_int(r['completion'])} | {_mean_int(r['reasoning_tokens'])} |"
            f" {sep} | {'/'.join(r['finish']) or '-'} | {r['cost']:.4f} | {'; '.join(extra) or '-'} |"
        )
    lines += [
        "",
        (
            f"Cost of the responses in this table: ${probe_spend:.4f}. All live calls in the probe ledger:"
            f" ${live_spend:.4f}, against a hard cap of ${args.cap_usd:.2f}."
        ),
    ]
    if stopped:
        lines += ["", f"**Probe stopped early:** {stopped}"]
    lines += ["", "## Observations", ""]
    lines += [f"- {o}" for o in observations(rows)]
    lines += ["", "## Recommendation", ""]
    lines += [p + "\n" for p in recommend(rows)]
    lines += [
        "## Models and rate limits",
        "",
        (
            "Catalogue prices and TPM, plus the `x-ratelimit-limit-*` values the API returned on one tiny call per"
            " model (16 max_tokens, included in the ledger total). The client also"
            " reads `x-ratelimit-remaining-tokens` and the `x-ratelimit-reset-*` durations (for example `1s`)."
        ),
        "",
        "| model | $/1M in | $/1M out | catalogue TPM | header RPM | header TPM |",
        "|---|---|---|---|---|---|",
    ]
    for model_id in MODEL_ORDER:
        info = MODELS[model_id]
        h = limits.get(model_id, {})
        lines.append(
            f"| `{model_id}` | {info.input_usd_per_mtok:.2f} | {info.output_usd_per_mtok:.2f} |"
            f" {info.tokens_per_minute:,} | {h.get('x-ratelimit-limit-requests', '-')} |"
            f" {h.get('x-ratelimit-limit-tokens', '-')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- summary

SUMMARY_COLUMNS = ("model", "mode", "valid", "clean", "p50 latency s", "est. $", "errors / notes")


def _markdown_sections(text: str) -> dict[str, list[str]]:
    """Lines under each ``## heading``; the preamble is under ``""``."""
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        else:
            sections[current].append(line)
    return sections


def _paragraphs(lines: list[str]) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for line in [*lines, ""]:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    return paras


def _cells(row: str) -> list[str]:
    return [c.strip().replace("`", "") for c in row.strip().strip("|").split("|")]


def report_summary(doc: Path = DEFAULT_DOC, notes_width: int = 60, wrap: int = 110) -> str:
    """A terminal-sized summary of a probe report: results table, spend and recommendation."""
    sections = _markdown_sections(doc.read_text(encoding="utf-8"))
    results = sections.get("Results", [])
    table = [line for line in results if line.startswith("|")]
    if len(table) < 3:
        raise ValueError(f"{doc}: no results table under '## Results'")
    header = _cells(table[0])
    missing = [c for c in SUMMARY_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{doc}: results table lacks columns {missing}")
    idx = [header.index(c) for c in SUMMARY_COLUMNS]
    rows = [["model", "mode", "valid", "clean", "p50 s", "est. $", "errors / notes"]]
    for line in table[2:]:
        cells = _cells(line)
        row = [cells[i] for i in idx]
        if len(row[-1]) > notes_width:
            row[-1] = row[-1][: notes_width - 3] + "..."
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    right = {2, 3, 4, 5}
    try:
        shown = doc.resolve().relative_to(Path.cwd())
    except ValueError:
        shown = doc
    out = [f"Token Factory probe report: {shown}"]
    out += [textwrap.fill(p, wrap) for p in _paragraphs(sections[""]) if not p.startswith("# ")][:1]
    out.append("")
    out += [
        "  ".join(
            c.rjust(w) if i in right else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True))
        ).rstrip()
        for r in rows
    ]
    out += [""] + [
        textwrap.fill(p, wrap) for p in _paragraphs(results) if p.startswith("Cost of the responses")
    ]
    recommendation = _paragraphs(sections.get("Recommendation", []))
    if recommendation:
        out += ["", "Recommendation:"]
        out += [
            textwrap.fill(
                p.replace("**", "").replace("`", ""), wrap, initial_indent="  ", subsequent_indent="  "
            )
            for p in recommendation
        ]
    out += [
        "",
        (
            "Re-run with `rtt probe-llm --live` (cached calls cost nothing; misses are billed under the cap),"
            " or rebuild the report offline with `rtt probe-llm --replay`."
        ),
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------- main


def add_probe_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``rtt probe-llm`` and ``scripts/probe_llm.py`` (not the run mode)."""
    parser.add_argument("--calls", type=int, default=2, help="calls per model and mode")
    parser.add_argument("--cap-usd", type=float, default=0.15, help="hard cap on all probe live spend")
    parser.add_argument("--models", nargs="*", default=list(MODEL_ORDER), choices=list(MODEL_ORDER))
    parser.add_argument("--thinking-off", action="store_true", help="add a tool+no_think mode per model")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="cache, ledger and results directory")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="markdown report path")


def run_probe(args: argparse.Namespace) -> int:
    """Run the probe described by ``args`` (see :func:`add_probe_arguments`; ``args.replay``
    selects cache-only mode) and write the report. Returns a process exit status."""
    ledger_path = args.out / "ledger.jsonl"
    prior = sum(t.cost_usd for t in summarize(read_ledger(ledger_path)).values())
    ledger = Ledger(ledger_path, budget_usd=max(0.0, args.cap_usd - prior))
    cache = ResponseCache(args.out / "cache", mode="replay" if args.replay else "readwrite")
    modes = ["tool", "json_schema"] + (["tool+no_think"] if args.thinking_off else [])
    print(f"probe: prior live spend ${prior:.4f}, cap ${args.cap_usd:.2f}", flush=True)

    results: list[CallResult] = []
    limits: dict[str, dict[str, str]] = {}
    stopped: str | None = None
    for model in [m for m in MODEL_ORDER if m in args.models]:
        try:
            limits[model] = capture_limits(model, cache, ledger)
        except (LLMAuthError, LLMBillingError) as exc:
            stopped = f"{type(exc).__name__} (HTTP {exc.status_code}) on {model} limits call"
            print(f"STOP: {stopped}", file=sys.stderr, flush=True)
            break
    for model in [m for m in MODEL_ORDER if m in args.models and not stopped]:
        for mode in modes:
            extra = {"chat_template_kwargs": {"enable_thinking": False}} if mode == "tool+no_think" else None
            client = LLMClient(
                model,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                cache=cache,
                ledger=ledger,
                timeout_s=240.0,
                retry=RetryPolicy(max_retries=2, base_delay_s=2.0),
                extra_body=extra,
            )
            for sample in range(args.calls):
                try:
                    result = run_one(client, mode, sample)
                except (LLMAuthError, LLMBillingError) as exc:
                    stopped = f"{type(exc).__name__} (HTTP {exc.status_code}) on {model} {mode}"
                    print(f"STOP: {stopped}", file=sys.stderr, flush=True)
                    break
                except CacheMiss as exc:
                    stopped = f"replay cache miss {exc.request_hash[:12]} on {model} {mode}"
                    print(f"STOP: {stopped}", file=sys.stderr, flush=True)
                    break
                results.append(result)
                m = result.meta or {}
                print(
                    f"{model.split('/', 1)[1]:<30} {mode:<14} #{sample} {result.status:<14}"
                    f" valid={result.valid!s:<5} bounds={len(result.bounds_violations)}"
                    f" rules={len(result.rule_violations)} lat={m.get('latency_s', '-')}"
                    f" tok={m.get('prompt_tokens', '-')}/{m.get('completion_tokens', '-')}"
                    f" cached={m.get('cached', '-')} spent=${ledger.spent_usd:.4f}",
                    flush=True,
                )
            if stopped:
                break
        if stopped:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps([r.__dict__ | {"valid": r.valid} for r in results], indent=1, sort_keys=True) + "\n"
    )
    write_report(args.doc, results, limits, ledger_path, args, stopped)
    print(f"report: {args.doc}  session live spend ${ledger.spent_usd:.4f}", flush=True)
    return 1 if stopped else 0


def main(argv: list[str] | None = None) -> int:
    """Script entry point: runs the probe live (cache first) unless ``--replay`` is given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_probe_arguments(parser)
    parser.add_argument("--replay", action="store_true", help="cache only; error on any miss")
    return run_probe(parser.parse_args(argv))
