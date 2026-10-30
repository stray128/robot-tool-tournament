"""The Task Engineer: a use case in words -> a validated :class:`~rtt.studio.types.TaskSpec`.

:func:`plan_task` asks ``nvidia/nemotron-3-super-120b-a12b`` (through
:class:`rtt.llm.client.LLMClient`: cache, ledger, spend cap, retries) for one forced
``submit_task_plan`` call. The prompt (``prompts/``) carries the use case, any uploaded part,
every object template and primitive with its bounded params (:mod:`rtt.studio.engineer.catalog`),
the quantities, the rules the checks enforce, the action-to-primitive mapping, the wrist and
table limits, the kinds of tool the search tries (what exists, never which wins) and one worked
example of a different use case.

**Judge.** The arguments are parsed by :func:`~rtt.studio.types.task_spec_from_tool_args`
(structure, bounds, cross-references; every problem with its path) and then checked by
:func:`~rtt.studio.engineer.checks.check_plan` (goals, holding, reachability, torque and force,
refs, refusals). A plan with problems gets **one repair call** that quotes the previous call
and lists every problem; whatever is still wrong after it is reported, never silently fixed,
with two stated exceptions. A refusal whose items name no nearest primitive after the repair
gets one from :func:`~rtt.studio.engineer.checks.nearest_primitive`. And when layout problems
(overlaps, swings, clearances) are all that is left after the repair, the deterministic
**layout fixer** (:func:`~rtt.studio.engineer.layout.fix_layout`) moves the free objects to the
nearest spot that clears them; each move becomes an assumption of the plan and the result is
``layout_fixed`` ("valid after layout fix"). Both say so in the result's notes.

**Tool.** :func:`engineer_tool` is :func:`~rtt.studio.types.submit_task_plan_tool`, the
contract's strict, bounded schema (closed objects, every property required, ``anyOf`` per kind
and primitive with const discriminators), descriptions included. ``compact=True`` drops every
``description`` (a third smaller, same structure); a live trial left out the steps'
``description`` field with it, so the full tool is the default (``docs/STUDIO_ENGINEER.md``).
The model sometimes writes null as text (``"None"``); :func:`normalise_nulls` reads that as
null in nullable fields, and the plan notes it.

**Guards.** No prompt may contain held-out vocabulary or a brand token (``PromptLeakError``
before anything is sent): the user's text is scrubbed first (the plan notes it). The model's
free text is scrubbed the same way before it is judged and before it is cached
(:class:`EngineerCache`, which also drops reasoning text), so a replay from the cache gives
exactly the plan the live run gave.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
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
from rtt.studio.engineer import prompts
from rtt.studio.engineer.catalog import (
    render_limits,
    render_mapping,
    render_primitives,
    render_quantities,
    render_rules,
    render_templates,
    render_tools,
)
from rtt.studio.engineer.checks import (
    check_parts,
    check_plan,
    claim_problems,
    claimed_inside,
    nearest_primitive,
)
from rtt.studio.engineer.examples import worked_example
from rtt.studio.engineer.layout import fix_door_bars, fix_layout, layout_problems
from rtt.studio.engineer.uploads import UploadInfo
from rtt.studio.types import (
    CONTRACT_VERSION,
    ObjectInstance,
    ObjectPose,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
    TaskSpecError,
    UnsupportedItem,
    submit_task_plan_tool,
    task_plan_json_schema,
    task_spec_from_tool_args,
)

if TYPE_CHECKING:
    from rtt.llm.cache import CacheMode

__all__ = [
    "COMPACT_TOOL",
    "ENGINEER_MODEL",
    "ENGINEER_SAMPLING",
    "MAX_USE_CASE_CHARS",
    "EngineerCache",
    "PlanAttempt",
    "PlanResult",
    "assert_clean",
    "build_messages",
    "engineer_tool",
    "explain_problems",
    "judge_args",
    "make_engineer_client",
    "normalise_nulls",
    "plan_task",
    "sanitize_use_case",
    "scrub_text",
    "tool_sha256",
]

ENGINEER_MODEL: Final = NEMOTRON_SUPER
#: Thinking on (the model's default). The largest complete call recorded (83 of them, S4 to S4d) was
#: 3,150 completion tokens, reasoning (at most 1,791) included; 5,000 leaves 1.6 x that and bounds a
#: call that keeps generating after a complete tool call (7 recorded calls ran on to the cap). With
#: a ~20k-token prompt the worst case is about $0.011.
ENGINEER_SAMPLING: Final[dict[str, float | int]] = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 5000}
#: Retries of a call: the client's statuses and Token Factory's 529 "System is overloaded", which a
#: draft re-record met on 7 of 12 calls (not part of a request, so recordings are unaffected).
ENGINEER_RETRY: Final = RetryPolicy(
    max_retries=4, base_delay_s=5.0, retry_statuses=RetryPolicy().retry_statuses | {529}
)
#: The default tool is the contract's own schema, descriptions included (module docstring).
COMPACT_TOOL: Final = False
MAX_USE_CASE_CHARS: Final = 2000
_MAX_PREVIOUS_CHARS: Final = 14000
_MAX_PROBLEMS: Final = 40

Status = Literal["valid", "refused", "invalid", "failed"]


# --------------------------------------------------------------------------- guards


def scrub_text(text: str) -> str:
    """``text`` with held-out vocabulary and brand tokens replaced by ``[removed]`` (layout kept)."""
    text = FORBIDDEN_PROMPT_TEXT_V2.sub("[removed]", text)
    for token in sorted(set(find_brand_tokens(text)), key=len, reverse=True):
        text = re.sub(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", "[removed]", text, flags=re.IGNORECASE)
    return text


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, Mapping):
        return {k: _scrub_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    return value


def assert_clean(messages: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a prompt that names held-out material or a brand (nothing is sent)."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if FORBIDDEN_PROMPT_TEXT_V2.search(content):
            raise PromptLeakError("task engineer prompt refers to held-out material; refusing to send it")
        if find_brand_tokens(content):
            raise PromptLeakError("task engineer prompt names a brand; refusing to send it")


def sanitize_use_case(text: str) -> tuple[str, tuple[str, ...]]:
    """The user's text as the prompt quotes it, and notes on what was changed.

    Whitespace is collapsed per line, the text is cut at :data:`MAX_USE_CASE_CHARS`, and
    held-out vocabulary and brand names are replaced by ``[removed]``.
    """
    if not isinstance(text, str):
        raise TypeError("the use case must be text")
    lines = [" ".join(line.split()) for line in text.strip().splitlines()]
    clean = "\n".join(line for line in lines if line)
    notes: list[str] = []
    if not clean:
        raise ValueError("describe the use case first")
    if len(clean) > MAX_USE_CASE_CHARS:
        clean = clean[:MAX_USE_CASE_CHARS].rstrip()
        notes.append(f"The description was cut to its first {MAX_USE_CASE_CHARS} characters.")
    scrubbed = scrub_text(clean)
    if scrubbed != clean:
        notes.append("Brand or reserved words in the description were replaced by [removed].")
    return scrubbed, tuple(notes)


# --------------------------------------------------------------------------- tool and prompt


def _strip_descriptions(schema: Any) -> Any:
    if isinstance(schema, Mapping):
        return {k: _strip_descriptions(v) for k, v in schema.items() if k != "description"}
    if isinstance(schema, list):
        return [_strip_descriptions(v) for v in schema]
    return schema


def engineer_tool(*, compact: bool = COMPACT_TOOL, version: str = prompts.CURRENT) -> dict[str, Any]:
    """The forced tool (module docstring): the contract's, optionally without schema descriptions;
    a frozen prompt's (v1, v2) is its tool as it was sent (:func:`prompts.frozen`)."""
    if version in prompts.FROZEN:
        return copy.deepcopy(prompts.frozen(version)["tool_compact" if compact else "tool"])
    tool = submit_task_plan_tool()
    if not compact:
        return tool
    fn = tool["function"]
    return {
        "type": "function",
        "function": {
            "name": fn["name"],
            "description": fn["description"],
            "parameters": _strip_descriptions(fn["parameters"]),
        },
    }


def tool_sha256(*, compact: bool = COMPACT_TOOL, version: str = prompts.CURRENT) -> str:
    return hashlib.sha256(wire_json(engineer_tool(compact=compact, version=version)).encode()).hexdigest()


def _schema_order(value: Any, schema: Mapping[str, Any]) -> Any:
    """``value`` with its keys in the order ``schema`` lists them (what the model emits)."""
    if isinstance(value, Mapping):
        branches = schema.get("anyOf")
        if branches:
            kind = value.get("kind", value.get("primitive"))
            schema = next(
                (
                    b
                    for b in branches
                    if kind
                    in (
                        b["properties"].get("kind", {}).get("const"),
                        b["properties"].get("primitive", {}).get("const"),
                    )
                ),
                branches[0],
            )
        props = schema.get("properties", {})
        keys = [k for k in props if k in value] + [k for k in value if k not in props]
        return {k: _schema_order(value[k], props.get(k, {})) for k in keys}
    if isinstance(value, list):
        return [_schema_order(v, schema.get("items", {})) for v in value]
    return value


def render_example() -> str:
    """The worked example as the JSON arguments of a submit_task_plan call (one entry per line),
    keys in schema order."""
    schema = task_plan_json_schema()
    args = _schema_order(worked_example().to_tool_args(), schema)
    lines = ["{"]
    lines.append(f'"title": {json.dumps(args["title"])}, "description": {json.dumps(args["description"])},')
    lines.append('"objects": [')
    lines.append(",\n".join(wire_json(o) for o in args["objects"]))
    lines.append("],")
    lines.append('"steps": [')
    lines.append(",\n".join(wire_json(s) for s in args["steps"]))
    lines.append("],")
    rest = {k: args[k] for k in ("assumptions", "clarifications", "unsupported", "refusal")}
    lines.append(wire_json(rest)[1:-1])
    lines.append("}")
    return "\n".join(lines)


def _uploads_text(uploads: Sequence[UploadInfo]) -> str:
    if not uploads:
        return "No part was uploaded: model every part as a box or a cylinder (ref null)."
    lines = ["# Uploaded parts", ""]
    lines += [u.prompt_line() for u in uploads]
    return "\n".join(lines)


def build_messages(
    use_case: str, uploads: Sequence[UploadInfo] = (), *, version: str = prompts.CURRENT
) -> list[dict[str, str]]:
    """The system and user messages for one use case (already sanitised, :func:`sanitize_use_case`),
    in prompt ``version`` (v1, v2: their frozen sections, :func:`prompts.frozen`)."""
    if version in prompts.FROZEN:
        sections = dict(prompts.frozen(version)["sections"])
    else:
        sections = {
            "templates": render_templates(),
            "primitives": render_primitives(),
            "quantities": render_quantities(),
            "rules": render_rules(),
            "mapping": render_mapping(),
            "limits": render_limits(),
            "tools": render_tools(),
            "example": render_example(),
        }
    user = prompts.render("user.md", version, use_case=use_case, uploads=_uploads_text(uploads), **sections)
    return [
        {"role": "system", "content": prompts.render("system.md", version)},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- client


class EngineerCache(ResponseCache):
    """A response cache that stores what a replay needs and nothing else.

    Before a live response is written, reasoning text is dropped (its token count stays in
    ``usage``) and every string the model wrote (content and every string in the tool call's
    arguments) passes through :func:`scrub_text`. :func:`plan_task` scrubs the same way before
    it judges, so a replay gives exactly the plan the live run gave.
    """

    def put(self, entry: CacheEntry) -> Path | None:
        return super().put(replace(entry, response=_scrub_response(entry.response)))


def _scrub_response(response: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(response))
    for choice in out.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            continue
        for key in ("reasoning_content", "reasoning"):
            message.pop(key, None)
        if isinstance(message.get("content"), str):
            message["content"] = scrub_text(message["content"])
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
                continue
            try:
                args = json.loads(fn["arguments"])
            except json.JSONDecodeError:  # kept as sent, scrubbed (a replay fails to parse it the same way)
                fn["arguments"] = scrub_text(fn["arguments"])
                continue
            fn["arguments"] = json.dumps(_scrub_value(args), ensure_ascii=False)
    return out


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


def make_engineer_client(
    cache_dir: str | Path | None,
    *,
    cache_mode: CacheMode = "readwrite",
    ledger: Ledger | None = None,
    ledger_path: str | Path | None = None,
    budget_usd: float | None = None,
    transport: Any | None = None,
    env_file: str | Path | None = None,
    limiter: TokenBucket | None = None,
) -> LLMClient:
    """The Task Engineer's client: Nemotron 3 Super, :data:`ENGINEER_SAMPLING`, cache and ledger.

    ``env_file`` names the ``.env`` holding ``NEBIUS_API_KEY`` (read only on the first live
    call); with ``cache_mode="replay"`` no key is needed and a miss is an error. A given
    ``ledger`` (for example the API's spend-guarded one) wins over ``ledger_path``/``budget_usd``.
    """
    if transport is None and env_file is not None:
        transport = _LazyTransport(Path(env_file))
    cache = EngineerCache(cache_dir, mode=cache_mode) if cache_dir is not None else None
    return LLMClient(
        ENGINEER_MODEL,
        temperature=float(ENGINEER_SAMPLING["temperature"]),
        top_p=float(ENGINEER_SAMPLING["top_p"]),
        max_tokens=int(ENGINEER_SAMPLING["max_tokens"]),
        cache=cache,
        ledger=ledger if ledger is not None else Ledger(ledger_path, budget_usd=budget_usd),
        transport=transport,
        timeout_s=300.0,
        retry=ENGINEER_RETRY,
        limiter=limiter,
    )


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class PlanAttempt:
    """One model call of a plan and its verdict (no message bodies; the hash points into the cache)."""

    stage: Literal["main", "repair"]
    outcome: Literal["valid", "refused", "invalid", "parse_error", "call_failed", "spend_cap", "truncated"]
    problems: tuple[str, ...]
    request_hash: str | None
    cached: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    finish_reason: str | None = None
    notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def truncated(self) -> bool:
        """The call was cut off at ``max_tokens`` (finish_reason ``length``): under rule set v2 its
        plan, should it parse, is never judged (``truncated``); the recorder marks it either way."""
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class PlanResult:
    """What :func:`plan_task` returns: the plan (or why there is none) and how it was reached.

    ``status``: ``valid`` (a runnable plan), ``refused`` (a valid refusal), ``invalid`` (still
    has problems after the repair; ``problems`` lists them and ``spec`` is the parsed plan when
    it parsed), ``failed`` (no usable call: transport, parse or spend cap). ``layout_moves``:
    the layout fixer's moves (the assumption lines it added) when it made the plan valid.
    """

    use_case: str
    status: Status
    spec: TaskSpec | None
    attempts: tuple[PlanAttempt, ...]
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    uploads: tuple[UploadInfo, ...] = ()
    template_hashes: Mapping[str, str] = field(default_factory=dict)
    tool_sha256: str = ""
    ledger: tuple[LedgerEntry, ...] = ()
    layout_moves: tuple[str, ...] = ()
    version: str = prompts.CURRENT

    @property
    def ok(self) -> bool:
        return self.status in ("valid", "refused")

    @property
    def first_try(self) -> bool:
        """The first call already gave a valid plan or refusal."""
        return bool(self.attempts) and self.attempts[0].outcome in ("valid", "refused")

    @property
    def layout_fixed(self) -> bool:
        """Valid only after the layout fixer moved objects (after the repair call)."""
        return self.status == "valid" and bool(self.layout_moves)

    @property
    def repaired(self) -> bool:
        """Valid (or refused) after the repair call, without the layout fixer."""
        return self.ok and not self.first_try and not self.layout_fixed

    @property
    def result(self) -> str:
        """``valid_first_call``, ``valid_after_repair``, ``valid_after_layout_fix``, ``refused`` or
        ``failed`` (the gate table's column)."""
        if self.status == "valid":
            if self.first_try:
                return "valid_first_call"
            return "valid_after_layout_fix" if self.layout_fixed else "valid_after_repair"
        return "refused" if self.status == "refused" else "failed"

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts if not a.cached)

    @property
    def truncated(self) -> tuple[str, ...]:
        """The stages whose call was cut off at ``max_tokens`` (:attr:`PlanAttempt.truncated`)."""
        return tuple(a.stage for a in self.attempts if a.truncated)

    def to_dict(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "format": "rtt-studio-plan",
            "version": 1,
            "contract": CONTRACT_VERSION,
            "label": PRODUCT_MODE_LABEL,
            "model": ENGINEER_MODEL,
            "use_case": self.use_case,
            "status": self.status,
            "first_try": self.first_try,
            "repaired": self.repaired,
            "layout_fixed": self.layout_fixed,
            "result": self.result,
            "layout_moves": list(self.layout_moves),
            "problems": list(self.problems),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "uploads": [u.to_json() for u in self.uploads],
            "plan": spec.to_dict() if spec is not None else None,
            "attempts": [a.to_dict() for a in self.attempts],
            "cost_usd": round(self.cost_usd, 6),
            "template_hashes": dict(self.template_hashes),
            "tool_sha256": self.tool_sha256,
            "prompt_version": self.version,
        }


# --------------------------------------------------------------------------- judging


_NULL_TEXT: Final = frozenset({"none", "null", "nil", "n/a", "na", ""})
_MISSING_KEY = re.compile(r"^(?P<list>objects|steps)\[(?P<i>\d+)\]: '(?P<key>[A-Za-z_]+)'$")


def _nullish(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _NULL_TEXT


def normalise_nulls(args: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """``args`` with null written as text (``"None"``, ``"null"``) in a nullable field turned into
    JSON null, and a note when that happened. Nothing else is changed."""
    out = copy.deepcopy(dict(args))
    changed: list[str] = []

    def fix(container: Any, key: Any, where: str) -> None:
        if isinstance(container, dict) and key in container and _nullish(container[key]):
            container[key] = None
            changed.append(where)

    fix(out, "refusal", "refusal")
    for i, o in enumerate(out.get("objects") if isinstance(out.get("objects"), list) else []):
        fix(o, "ref", f"objects[{i}].ref")
    for i, st in enumerate(out.get("steps") if isinstance(out.get("steps"), list) else []):
        fix(st, "target", f"steps[{i}].target")
        cons = st.get("constraints") if isinstance(st, dict) else None
        for k in list(cons) if isinstance(cons, dict) else []:
            fix(cons, k, f"steps[{i}].constraints.{k}")
        crit = st.get("success") if isinstance(st, dict) else None
        for j, c in enumerate(crit if isinstance(crit, list) else []):
            fix(c, "at_least", f"steps[{i}].success[{j}].at_least")
            fix(c, "at_most", f"steps[{i}].success[{j}].at_most")
    for i, u in enumerate(out.get("unsupported") if isinstance(out.get("unsupported"), list) else []):
        fix(u, "nearest_primitive", f"unsupported[{i}].nearest_primitive")
    note = (f"Null written as text was read as null ({', '.join(changed[:6])}).",) if changed else ()
    return out, note


def _field_hint(list_name: str, key: str, item: Any, version: str = prompts.CURRENT) -> str:
    """The full schema's description of ``key`` in an object or step (``""`` if none), in the tool
    of prompt ``version``."""
    params = (
        task_plan_json_schema() if version != "v1" else engineer_tool(version="v1")["function"]["parameters"]
    )
    schema = params["properties"][list_name]["items"]["anyOf"]
    disc = "kind" if list_name == "objects" else "primitive"
    value = item.get(disc) if isinstance(item, Mapping) else None
    branch = next((b for b in schema if b["properties"][disc].get("const") == value), schema[0])
    return str(branch["properties"].get(key, {}).get("description", ""))


def explain_problems(
    problems: Sequence[str], args: Mapping[str, Any], version: str = prompts.CURRENT
) -> list[str]:
    """The contract parser's problems, reworded where its text is terse (a bare missing key, a
    refusal next to steps), with the entry's name in the path."""
    out: list[str] = []
    for p in problems:
        m = _MISSING_KEY.match(p)
        if m is not None:
            items = args.get(m["list"]) if isinstance(args.get(m["list"]), list) else []
            i = int(m["i"])
            item = items[i] if i < len(items) else None
            name = item.get("name" if m["list"] == "objects" else "id") if isinstance(item, Mapping) else None
            hint = _field_hint(m["list"], m["key"], item, version)
            label = f" ({name})" if isinstance(name, str) else ""
            out.append(f"{m['list']}[{i}]{label}: missing {m['key']!r}" + (f": {hint}" if hint else ""))
        elif p == "a refused task has no steps":
            out.append(
                "refusal: must be null (JSON null, not text) when the plan has steps; a refused use case "
                "has no steps"
            )
        else:
            out.append(p)
    return out


def judge_args(
    args: Mapping[str, Any],
    *,
    uploads: Sequence[UploadInfo] = (),
    source: Literal["nemotron", "manual", "recorded", "test"] = "nemotron",
    rules: str = prompts.CURRENT_RULES,
    paths: bool = True,
) -> tuple[TaskSpec | None, list[str], list[str]]:
    """``(spec or None, problems, warnings)`` of a plan's tool arguments, under rule set ``rules``
    (:mod:`rtt.studio.engineer.checks`).

    ``spec`` is ``None`` when the arguments do not parse into a :class:`TaskSpec`. The uploads
    given are the ones that exist and the ones the plan must use.
    """
    ids = [u.upload_id for u in uploads]
    try:
        spec = task_spec_from_tool_args(args, source=source)
    except TaskSpecError as exc:
        problems = explain_problems(exc.problems, args, rules)
        if args.get("refusal") is None:
            objects, steps = _parse_parts(args)
            extra = check_parts(objects, steps, uploads=ids, required_uploads=ids, rules=rules, paths=paths)
            problems += [p for p in extra if p not in problems]
        return None, problems, []
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return (
            None,
            [f"$: the arguments do not match the submit_task_plan schema ({type(exc).__name__}: {exc})"],
            [],
        )
    problems, warnings = check_plan(spec, uploads=ids, required_uploads=ids, rules=rules, paths=paths)
    return spec, problems, warnings


def _parse_parts(args: Mapping[str, Any]) -> tuple[list[ObjectInstance], list[Step]]:
    """The objects and steps of ``args`` that construct on their own (the rest are skipped; the
    contract's parser has reported them), built exactly as :func:`task_spec_from_tool_args` does."""
    objects: list[ObjectInstance] = []
    steps: list[Step] = []
    raw_objects = args.get("objects")
    raw_steps = args.get("steps")
    for o in raw_objects if isinstance(raw_objects, list) else []:
        try:
            objects.append(
                ObjectInstance(
                    name=o["name"],
                    kind=o["kind"],
                    label=o.get("label", o.get("name", "")),
                    params=o["params"],
                    pose=ObjectPose(**(o.get("pose") or {})),
                    ref=o.get("ref"),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    for st in raw_steps if isinstance(raw_steps, list) else []:
        try:
            steps.append(
                Step(
                    id=st["id"],
                    primitive=st["primitive"],
                    object=st["object"],
                    handle=st["handle"],
                    description=st["description"],
                    params=st["params"],
                    success=tuple(SuccessCriterion(**c) for c in st["success"]),
                    constraints=StepConstraints(**(st.get("constraints") or {})),
                    target=st.get("target"),
                    after=tuple(st.get("after") or ()),
                    hold_after=bool(st.get("hold_after", False)),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return objects, steps


def _attempt(
    stage: Literal["main", "repair"],
    outcome: Any,
    problems: Sequence[str],
    meta: CallMeta | None,
    error: str | None = None,
) -> PlanAttempt:
    return PlanAttempt(
        stage=stage,
        outcome=outcome,
        problems=tuple(problems),
        request_hash=meta.request_hash if meta else None,
        cached=bool(meta and meta.cached),
        prompt_tokens=meta.prompt_tokens if meta else 0,
        completion_tokens=meta.completion_tokens if meta else 0,
        reasoning_tokens=(meta.reasoning_tokens or 0) if meta else 0,
        cost_usd=meta.cost_usd if meta else 0.0,
        latency_s=meta.latency_s if meta else 0.0,
        finish_reason=meta.finish_reason if meta else None,
        notes=meta.notes if meta else (),
        error=error,
    )


def _call(
    client: LLMClient, messages: list[dict[str, str]], tool: dict[str, Any], stage: str, tag: str
) -> tuple[dict[str, Any] | None, CallMeta | None, str | None, str | None]:
    """One forced call: ``(scrubbed arguments or None, meta, failure kind, error text)``."""
    assert_clean(messages)
    try:
        args, meta = client.call_tool(messages, tool, force=True, sample=0, tag=f"{tag}/{stage}")
    except (LLMAuthError, LLMBillingError):
        raise
    except LLMParseError as exc:
        return None, exc.meta, "parse_error", f"parse_error: {exc}"
    except BudgetExceededError as exc:
        return None, None, "spend_cap", f"spend_cap: {exc}"
    except (LLMHTTPError, LLMTransportError) as exc:
        return None, None, "call_failed", f"call_failed: {exc}"
    return _scrub_value(args), meta, None, None


def _previous_text(args: Mapping[str, Any] | None, error: str | None) -> str:
    if args is None:
        return f"(no usable submit_task_plan call: {scrub_text(error or 'unknown error')[:300]})"
    text = wire_json(_json_safe(args))
    if len(text) > _MAX_PREVIOUS_CHARS:
        text = text[:_MAX_PREVIOUS_CHARS] + " ... (cut)"
    return text


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


def _fill_nearest(spec: TaskSpec, use_case: str) -> TaskSpec:
    """A refusal whose items name no nearest primitive, with one from the keyword table."""
    if spec.unsupported:
        items = tuple(
            replace(
                u, nearest_primitive=nearest_primitive(f"{u.what} {u.reason}") or nearest_primitive(use_case)
            )
            if u.nearest_primitive is None
            else u
            for u in spec.unsupported
        )
    else:
        what = scrub_text(use_case)[:117].rstrip() + ("..." if len(use_case) > 117 else "")
        reason = (spec.refusal or "not simulated")[:200]
        items = (UnsupportedItem(what, reason, nearest_primitive(use_case)),)
    return replace(spec, unsupported=items)


#: The contract's cap on a plan's assumptions (the layout fixer's moves are added to them).
_MAX_ASSUMPTIONS: Final = 10


def _fix_layout(
    spec: TaskSpec,
    problems: Sequence[str],
    uploads: Sequence[UploadInfo],
    rules: str = prompts.CURRENT_RULES,
    paths: bool = True,
) -> tuple[TaskSpec, tuple[str, ...], tuple[str, ...]] | str:
    """The layout fixer on a plan whose only problems are layout problems (and, under rule set v2,
    words that put a thing inside a machine it does not stand inside, which the fixer places there):
    ``(spec, warnings, moves)`` when the moved plan passes every check, else a note (empty when it
    did not apply)."""
    layout = layout_problems(spec.objects, spec.steps, rules=rules)
    claims = claim_problems(spec) if rules == "v2" else []
    if not problems or not (layout or claims) or not set(problems) <= {*layout, *claims}:
        return ""
    inside = claimed_inside(spec) if rules == "v2" else None
    fixed = fix_layout(spec.objects, spec.steps, rules=rules, inside=inside)
    if fixed is None:
        return (
            "The layout fixer found no spot within reach that clears every object; the layout problems stand."
        )
    objects, moves = fixed
    lines = tuple(m.text() for m in moves)
    if len(spec.assumptions) + len(lines) > _MAX_ASSUMPTIONS:
        return (
            f"The layout fixer would move {len(lines)} objects, more than the plan's assumptions can list; "
            "the layout problems stand."
        )
    try:
        moved = replace(spec, objects=tuple(objects), assumptions=(*spec.assumptions, *lines))
    except TaskSpecError as exc:
        return f"The layout fixer's moves do not construct ({'; '.join(exc.problems)[:200]}); the layout problems stand."
    ids = [u.upload_id for u in uploads]
    left, warnings = check_plan(moved, uploads=ids, required_uploads=ids, rules=rules, paths=paths)
    if left:
        return "The layout fixer's moves left other problems; the layout problems stand."
    return moved, tuple(warnings), lines


def _fix_door_bars(
    spec: TaskSpec, uploads: Sequence[UploadInfo], paths: bool = True
) -> tuple[TaskSpec, tuple[str, ...], tuple[str, ...]] | str:
    """The layout fixer's door-bar rule (rule set v2, :func:`~rtt.studio.engineer.layout.fix_door_bars`)
    on a valid plan: ``(spec, warnings, moves)`` when it moved something and the moved plan passes
    every check, else a note (empty when no door needed it)."""
    fixed = fix_door_bars(spec.objects, spec.steps, inside=claimed_inside(spec))
    if isinstance(fixed, str):
        return fixed
    objects, moves = fixed
    if not moves:
        return ""
    lines = tuple(m.text() for m in moves)
    if len(spec.assumptions) + len(lines) > _MAX_ASSUMPTIONS:
        return (
            f"The layout fixer would move {len(lines)} objects so a door's bar is gripped from the front, more "
            "than the plan's assumptions can list; the door stays where it is."
        )
    try:
        moved = replace(spec, objects=tuple(objects), assumptions=(*spec.assumptions, *lines))
    except TaskSpecError as exc:
        return f"The layout fixer's door moves do not construct ({'; '.join(exc.problems)[:200]}); the door stays where it is."
    ids = [u.upload_id for u in uploads]
    left, warnings = check_plan(moved, uploads=ids, required_uploads=ids, rules="v2", paths=paths)
    if left:
        return "The layout fixer's door moves left other problems; the door stays where it is."
    return moved, tuple(warnings), lines


@dataclass(frozen=True)
class _Verdict:
    spec: TaskSpec | None
    outcome: str | None
    problems: tuple[str, ...]
    warnings: tuple[str, ...]
    failure: str | None
    args: dict[str, Any] | None
    error: str | None


def plan_task(
    text: str,
    *,
    client: LLMClient,
    uploads: Sequence[UploadInfo] = (),
    repairs: int = 1,
    compact: bool = COMPACT_TOOL,
    tag: str | None = None,
    version: str = prompts.CURRENT,
) -> PlanResult:
    """Turn a use case into a validated task plan (module docstring), with prompt ``version`` and
    its rule set (:data:`prompts.RULES_OF`; v1 and v2 rebuild their recordings' requests byte for
    byte)."""
    if repairs not in (0, 1):
        raise ValueError("at most one repair call per plan")
    use_case, notes_in = sanitize_use_case(text)
    notes = list(notes_in)
    rules = prompts.rules_of(version)
    paths = prompts.field_paths(version)
    tool = engineer_tool(compact=compact, version=version)
    messages = build_messages(use_case, uploads, version=version)
    digest = hashlib.sha256(use_case.encode()).hexdigest()[:12]
    tag = tag or f"studio-engineer/{digest}"
    ledger_start = len(client.ledger.session_entries())
    attempts: list[PlanAttempt] = []

    def judge(stage: Literal["main", "repair"], msgs: list[dict[str, str]]) -> _Verdict:
        args, meta, failure, error = _call(client, msgs, tool, stage, tag)
        if failure is not None:
            attempts.append(_attempt(stage, failure, [error or failure], meta, error))
            return _Verdict(None, failure, (error or failure,), (), failure, args, error)
        assert args is not None
        if rules == "v2" and meta is not None and meta.finish_reason == "length":
            # cut off at max_tokens: what parsed is a prefix of the plan the model meant, never judged
            error = (
                "truncated: the call was cut off at max_tokens (finish_reason length); its plan is not judged"
            )
            attempts.append(_attempt(stage, "truncated", [error], meta, error))
            return _Verdict(None, "truncated", (error,), (), "truncated", args, error)
        args, null_note = normalise_nulls(args)
        for n in null_note:
            if n not in notes:
                notes.append(n)
        spec, problems, warnings = judge_args(args, uploads=uploads, rules=rules, paths=paths)
        outcome = "invalid" if problems or spec is None else ("refused" if spec.refusal else "valid")
        attempts.append(_attempt(stage, outcome, problems, meta))
        return _Verdict(spec, outcome, tuple(problems), tuple(warnings), None, args, None)

    final = main = judge("main", messages)
    if main.outcome not in ("valid", "refused") and repairs and main.failure != "spend_cap":
        repair_text = prompts.render(
            "repair.md",
            version,
            previous=_previous_text(main.args, main.error),
            problems="\n".join(f"- {scrub_text(p)[:300]}" for p in main.problems[:_MAX_PROBLEMS]),
        )
        repair_messages = [dict(m) for m in messages]
        repair_messages[-1]["content"] = f"{repair_messages[-1]['content']}\n\n{repair_text}"
        repair = judge("repair", repair_messages)
        if repair.failure is None or main.failure is not None:
            final = repair
        else:
            notes.append(f"The repair call failed ({repair.failure}); the first plan's problems stand.")

    spec, problems, warnings = final.spec, list(final.problems), final.warnings
    layout_moves: tuple[str, ...] = ()
    if final.failure is None and final.outcome == "invalid" and spec is not None and spec.refusal is None:
        fixed = _fix_layout(spec, problems, uploads, rules, paths)
        if isinstance(fixed, str):
            if fixed:
                notes.append(fixed)
        else:
            spec, warnings, layout_moves = fixed
            problems = []
            notes.append(
                f"The layout fixer moved {len(layout_moves)} object{'s' if len(layout_moves) != 1 else ''} in code "
                "after the repair call; each move is an assumption of the plan."
            )
    if (
        rules == "v2"
        and final.failure is None
        and spec is not None
        and spec.refusal is None
        and (final.outcome == "valid" or layout_moves)
    ):
        doors = _fix_door_bars(spec, uploads, paths)
        if isinstance(doors, str):
            if doors:
                notes.append(doors)
        else:
            spec, warnings, more = doors
            layout_moves = (*layout_moves, *more)
            notes.append(
                f"The layout fixer moved {len(more)} object{'s' if len(more) != 1 else ''} in code so a "
                "two-finger gripper grips the door's bar from the front; each move is an assumption of the plan."
            )
    status: Status
    if final.failure is not None:
        status = "failed"
    elif final.outcome in ("valid", "refused"):
        status = final.outcome  # type: ignore[assignment]
    elif layout_moves:
        status = "valid"
    elif (
        spec is not None and spec.refusal is not None and all(p.startswith("unsupported:") for p in problems)
    ):
        spec = _fill_nearest(spec, use_case)
        problems = []
        status = "refused"
        notes.append(
            "The nearest primitive of the refusal was suggested by the keyword table, not the model."
        )
    else:
        status = "invalid"
    ledger = tuple(client.ledger.session_entries()[ledger_start:])
    return PlanResult(
        use_case=use_case,
        status=status,
        spec=spec,
        attempts=tuple(attempts),
        problems=tuple(problems),
        warnings=tuple(warnings),
        notes=tuple(notes),
        uploads=tuple(uploads),
        template_hashes=prompts.template_hashes(version),
        tool_sha256=tool_sha256(compact=compact, version=version),
        ledger=ledger,
        layout_moves=layout_moves,
        version=version,
    )
