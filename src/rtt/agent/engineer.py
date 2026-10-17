"""The Grasp Engineer: a bounded Nemotron 3 Super tool-calling loop (product mode).

A plain OpenAI-SDK-style loop over :class:`rtt.llm.LLMClient` (Token Factory, with the
client's ledger, retries and rate limiting): the model sees the eight tools of
:mod:`rtt.agent.schemas` (nine when the toolbox has an effector runner: ``recommend_effector``)
with ``tool_choice="auto"``; each tool call it makes is run by
:class:`rtt.agent.tools.Toolbox` under the hard caps and answered with a ``tool``
message; the loop ends when the model answers in plain text, when the stop flag is set,
or when a cap ends it. Once the tool budget is spent, the next turn is sent with
``tool_choice="none"`` so the model must write its final answer.

Nemotron 3.5 Lightning (thinking off) writes the short summaries shown to the user
after key steps and at the end (:mod:`rtt.agent.summarizer`).

Everything is recorded (:class:`rtt.agent.recorder.SessionRecorder`): every model call
with tokens and cost, every message, every tool call with validated arguments and its
result, job state changes, summaries, and the code-computed physics verdicts.

This agent is a product feature. It is not part of the pre-registered experiment and
its sessions are labelled so (:data:`rtt.agent.recorder.PRODUCT_MODE_LABEL`).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from rtt.agent.caps import AgentCaps
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SCRIPTED_MODEL, SessionRecorder, redact_paths
from rtt.agent.schemas import openai_tools
from rtt.agent.summarizer import LightningSummarizer
from rtt.agent.tools import Toolbox, ToolOutcome
from rtt.llm.cache import CacheMiss
from rtt.llm.client import CallMeta, LLMClient, LLMError, LLMParseError, split_inline_reasoning
from rtt.llm.ledger import BudgetExceededError
from rtt.llm.providers import NEMOTRON_SUPER

__all__ = [
    "AGENT_MODEL",
    "AGENT_SAMPLING",
    "DEFAULT_GOAL",
    "AgentLLM",
    "AssistantTurn",
    "GraspEngineer",
    "SessionResult",
    "ToolCallRequest",
    "system_prompt",
]

AGENT_MODEL: Final = NEMOTRON_SUPER
#: Agent sampling: moderate temperature, thinking on, room for reasoning plus tool calls.
AGENT_SAMPLING: Final[dict[str, float | int]] = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096}
DEFAULT_GOAL: Final = (
    "Find gripper fingers that reliably lift and hold my part, check how robust the best one is, "
    "and give me a print pack."
)
#: Tools whose results get a Lightning summary for the user.
SUMMARIZED_TOOLS: Final = frozenset({"inspect_part", "fetch_results", "stress_test", "export_print_pack"})
_MAX_REASONING_CHARS: Final = 2000


# --------------------------------------------------------------------------- model client


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call in an assistant turn. ``args`` is ``None`` when the arguments were not JSON."""

    call_id: str
    name: str
    arguments: str
    args: dict[str, Any] | None
    parse_error: str | None = None

    def message(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class AssistantTurn:
    """The parsed assistant message of one agent call."""

    content: str
    tool_calls: tuple[ToolCallRequest, ...]
    reasoning: str | None = None

    def message(self) -> dict[str, Any]:
        """The assistant message to append to the history (OpenAI format)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [c.message() for c in self.tool_calls]
        return msg


def _parse_turn(response: Mapping[str, Any]) -> tuple[AssistantTurn, tuple[str, ...]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise LLMParseError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise LLMParseError("response choice has no message")
    raw_content = message.get("content")
    content, inline = split_inline_reasoning(raw_content if isinstance(raw_content, str) else "")
    reasoning = None
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            reasoning = value
            break
    reasoning = reasoning or inline
    calls: list[ToolCallRequest] = []
    notes: list[str] = []
    for i, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        call_id = str(call.get("id") or f"call_{i}")
        if isinstance(arguments, Mapping):
            calls.append(ToolCallRequest(call_id, name, json.dumps(arguments), dict(arguments)))
            continue
        text = arguments if isinstance(arguments, str) else ""
        try:
            value = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            notes.append("tool_args_not_json")
            calls.append(
                ToolCallRequest(call_id, name, text, None, f"arguments are not valid JSON: {exc.msg}")
            )
            continue
        if isinstance(value, dict):
            calls.append(ToolCallRequest(call_id, name, text, value))
        else:
            calls.append(ToolCallRequest(call_id, name, text, None, "arguments must be a JSON object"))
    if inline is not None:
        notes.append("think_tags_stripped")
    return AssistantTurn(content, tuple(calls), reasoning), tuple(notes)


class AgentLLM(LLMClient):
    """:class:`LLMClient` plus multi-tool calls with ``tool_choice`` ``auto`` or ``none``.

    The request goes through the client's own pipeline (cache, ledger reservation
    against the session's spend guard, rate limiting, retries), so the agent's calls
    are capped and recorded exactly like the proposer's.
    """

    @classmethod
    def create(cls, ledger: Any, *, cache: Any | None = None, transport: Any | None = None) -> AgentLLM:
        return cls(
            AGENT_MODEL,
            temperature=float(AGENT_SAMPLING["temperature"]),
            top_p=float(AGENT_SAMPLING["top_p"]),
            max_tokens=int(AGENT_SAMPLING["max_tokens"]),
            cache=cache,
            ledger=ledger,
            transport=transport,
            timeout_s=180.0,
        )

    def call_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        tool_choice: str = "auto",
        tag: str | None = None,
    ) -> tuple[AssistantTurn, CallMeta]:
        """One agent turn: the model may answer in text or call any number of ``tools``."""
        if tool_choice not in ("auto", "none"):
            raise ValueError("tool_choice must be 'auto' or 'none'")
        fields = {"tools": [dict(t) for t in tools], "tool_choice": tool_choice}
        return self._run("tool", messages, fields, _parse_turn, sample=0, tag=tag)


# --------------------------------------------------------------------------- prompts


def system_prompt(caps: AgentCaps, targets: Sequence[str], *, effectors: bool = False) -> str:
    target_text = "'local' (this machine, free)" + (
        " or 'nebius' (Nebius Serverless AI Job, costs money)" if "nebius" in targets else ""
    )
    paragraphs = (
        (
            "You are the Grasp Engineer. A user uploaded a part; you find 3D-printable fingers for an "
            "open-source parallel gripper (two jaws, 50 N grip) that lift it 100 mm and hold it for 2 s."
        ),
        (
            "Physics decides. A MuJoCo simulator scores every design and returns a pass/fail verdict. You "
            "orchestrate tools and explain; you never claim a design works without simulator results, and "
            "you report verdicts exactly as the tools give them."
        ),
        (
            "A good plan: inspect_part; propose_designs (3-4); launch_sim_job with those designs plus one "
            f"valid baseline, target {target_text}; poll_job until COMPLETED; fetch_results; stress_test the "
            "best design; export_print_pack for the run; then answer."
        ),
        (
            "If no design passes, or the best one fails its stress test, and enough tool calls remain for "
            "another round (propose, launch, poll, fetch, stress_test, export: 6 calls), propose 1-2 designs "
            "aimed at the failure modes you saw, simulate them, and stress-test the best. Export the print "
            "pack from the run that holds the design you recommend, and keep your last tool call for it. "
            "Skip cost_ledger unless asked."
        ),
        (
            f"Hard limits, enforced by code: {caps.max_tool_calls} tool calls in total (every call counts, "
            f"including failed ones), at most {caps.max_concurrent_jobs} jobs in flight, and a "
            f"${caps.session_usd:.2f} spend cap for the session. Use IDs exactly as tools return them."
        ),
        (
            "Final answer: at most 120 words, plain text. Name the recommended design ID, its simulated "
            "success and verdict, the stress-test result, and the print pack ID. If nothing passed, say so "
            "and say what to try next."
        ),
    )
    if effectors:
        paragraphs += (
            (
                "If the user asks which kind of end effector suits the part (jaw, hand, gripper, suction cup, "
                "soft arm), call recommend_effector (local, free; design=true asks Nemotron for new morphologies "
                "and costs money), poll_job until COMPLETED, then fetch_results. Report the recommended effector, "
                "its certificate score and verdict, and for an approximation the rigid runner-up and that it must "
                "be verified on hardware, exactly as the tool gives them. It is product mode, not the "
                "pre-registered experiment."
            ),
        )
    return "\n\n".join(paragraphs)


def user_prompt(part_id: str, goal: str) -> str:
    return f"Part: {part_id}\nGoal: {goal.strip() or DEFAULT_GOAL}"


# --------------------------------------------------------------------------- the loop


@dataclass(frozen=True)
class SessionResult:
    status: str  # completed | stopped | failed
    answer: str
    reason: str | None
    turns: int
    tool_calls: int
    verdicts: list[dict[str, Any]]
    spend: dict[str, Any]
    summary: str | None = None


class GraspEngineer:
    """Runs one session (see the module docstring).

    Args:
        llm: The agent's model client (Nemotron 3 Super).
        toolbox: The session's tools (bound to its part, caps and spend guard).
        recorder: Where every step is recorded.
        caps: Session limits (the same object the toolbox enforces).
        goal: The user's goal in their words.
        summarizer: Lightning summaries (``None``: no summaries).
        stop: Set it to end the session after the current step.
        offline: The model is the scripted offline transport, not Nemotron; the session
            file then names ``scripted-offline`` as every model.
    """

    def __init__(
        self,
        llm: AgentLLM,
        toolbox: Toolbox,
        recorder: SessionRecorder,
        *,
        caps: AgentCaps,
        goal: str = DEFAULT_GOAL,
        summarizer: LightningSummarizer | None = None,
        stop: threading.Event | None = None,
        offline: bool = False,
    ) -> None:
        self.offline = offline
        self.llm = llm
        self.toolbox = toolbox
        self.recorder = recorder
        self.caps = caps
        self.goal = goal
        self.summarizer = summarizer
        self.stop = stop or threading.Event()
        self.messages: list[dict[str, Any]] = []
        self.turns = 0

    def _models(self) -> dict[str, str | None]:
        models = {
            "agent": self.llm.model,
            "proposer": getattr(getattr(self.toolbox.proposer, "client", None), "model", None),
            "summarizer": self.summarizer.client.model if self.summarizer else None,
        }
        if self.offline:
            return {k: (SCRIPTED_MODEL if v else None) for k, v in models.items()}
        return models

    def run(self) -> SessionResult:
        effectors = getattr(self.toolbox, "effectors", None) is not None
        tools = openai_tools(effectors=effectors)
        self.messages = [
            {
                "role": "system",
                "content": system_prompt(self.caps, sorted(self.toolbox.runners), effectors=effectors),
            },
            {"role": "user", "content": user_prompt(self.toolbox.part_id, self.goal)},
        ]
        self.recorder.emit(
            "session.started",
            part_id=self.toolbox.part_id,
            goal=self.goal,
            label=PRODUCT_MODE_LABEL,
            models=self._models(),
            offline=self.offline,
            caps=self.caps.to_json(),
            unlocked=self.toolbox.unlocked,
            targets=sorted(self.toolbox.runners),
            **({"effectors": True} if effectors else {}),
        )
        self.recorder.set_messages(self.messages)
        status, reason, answer = "completed", None, ""
        try:
            status, reason, answer = self._loop(tools)
        except (BudgetExceededError, CacheMiss) as exc:
            status, reason = "stopped", redact_paths(f"{type(exc).__name__}: {exc}")
        except LLMError as exc:
            status, reason = "failed", redact_paths(f"{type(exc).__name__}: {str(exc)[:300]}")
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            status, reason = "failed", redact_paths(f"{type(exc).__name__}: {str(exc)[:300]}")
            self.recorder.emit("error", message=reason)
        finally:
            self.toolbox.close()
        verdicts = self.toolbox.verdicts()
        spend = self.toolbox.guard.snapshot()
        summary = None
        if self.summarizer is not None and status != "failed":
            summary = self._summarize(
                "session",
                {"status": status, "answer": answer, "verdicts": verdicts, "spend_usd": spend["spent_usd"]},
            )
            spend = self.toolbox.guard.snapshot()
        self.recorder.set_messages(self.messages)
        self.recorder.update(spend=spend, verdicts=verdicts, answer=answer, summary=summary)
        self.recorder.finish(
            status,
            reason=reason,
            answer=answer,
            summary=summary,
            verdicts=verdicts,
            spend=spend,
            turns=self.turns,
            tool_calls=self.toolbox.calls_used,
        )
        return SessionResult(
            status, answer, reason, self.turns, self.toolbox.calls_used, verdicts, spend, summary
        )

    def _loop(self, tools: list[dict[str, Any]]) -> tuple[str, str | None, str]:
        while True:
            if self.stop.is_set():
                return "stopped", "stopped by the user", ""
            if self.turns >= self.caps.max_llm_turns:
                return "stopped", f"turn cap ({self.caps.max_llm_turns} model calls) reached", ""
            choice = "auto" if self.toolbox.calls_left > 0 else "none"
            turn, meta = self.llm.call_tools(self.messages, tools, tool_choice=choice, tag="agent.turn")
            self.turns += 1
            self.toolbox.emit_llm_call("agent", meta)
            self.messages.append(turn.message())
            self.recorder.emit(
                "agent.message",
                turn=self.turns,
                content=turn.content,
                tool_calls=[{"name": c.name, "call_id": c.call_id} for c in turn.tool_calls],
                reasoning=(turn.reasoning or "")[:_MAX_REASONING_CHARS] or None,
            )
            self.recorder.set_messages(self.messages)
            if not turn.tool_calls:
                return "completed", None, turn.content
            for call in turn.tool_calls:
                self._run_tool(call)
                self.recorder.set_messages(self.messages)

    def _run_tool(self, call: ToolCallRequest) -> ToolOutcome:
        raw: Any = call.args
        if call.args is None:
            raw = _Unparsed(call.parse_error or "invalid arguments")
        self.recorder.emit(
            "tool.call", call_id=call.call_id, name=call.name, args=call.args, n=self.toolbox.calls_used + 1
        )
        outcome = self.toolbox.call(call.name, raw)
        event = "tool.refused" if outcome.refused else "tool.result"
        self.recorder.emit(event, call_id=call.call_id, **outcome.to_event())
        self.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": outcome.for_model()})
        if outcome.ok and call.name in SUMMARIZED_TOOLS and self.summarizer is not None:
            self._summarize(call.name, outcome.result or {})
        return outcome

    def _summarize(self, kind: str, payload: Mapping[str, Any]) -> str | None:
        assert self.summarizer is not None
        try:
            text = self.summarizer.summarize(kind, payload)
        except Exception as exc:  # noqa: BLE001 - a summary is decoration; the session must still finish
            self.recorder.emit(
                "error", message=redact_paths(f"summary failed: {type(exc).__name__}: {exc}")[:300]
            )
            return None
        meta = self.summarizer.last_meta
        if meta is not None:
            self.toolbox.emit_llm_call("summarizer", meta)
        if text:
            model = SCRIPTED_MODEL if self.offline else self.summarizer.client.model
            self.recorder.emit("summary", step=kind, text=text, model=model)
        return text


class _Unparsed:
    """Stands in for tool arguments that were not JSON (validation reports ``parse_error``)."""

    def __init__(self, error: str) -> None:
        self.error = error

    def __repr__(self) -> str:
        return f"<unparsed: {self.error}>"
