"""Short user-facing summaries written by Nemotron 3.5 Lightning (thinking off).

The summarizer turns a tool result (or the session's outcome) into one or two plain
sentences for the UI and the video captions. It is told to use only the numbers it is
given and to repeat the simulator's verdicts, never to judge a grasp itself; when the
step has no verdict, sentences claiming a pass or fail are removed in code
(:func:`drop_verdict_claims`), and when no verdict of the step is a pass, sentences
claiming a pass are removed (:func:`drop_unsupported_passes`), so a summary can never
upgrade a fail. Calls are
small (a few hundred tokens) and go through the session's spend guard like every other
model call. Any failure returns ``None``: summaries are decoration, not results.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final

from rtt.llm.cache import CacheMiss
from rtt.llm.client import CallMeta, LLMClient, LLMError
from rtt.llm.ledger import BudgetExceededError
from rtt.llm.providers import NEMOTRON_LIGHTNING

__all__ = [
    "SUMMARY_SYSTEM",
    "LightningSummarizer",
    "drop_unsupported_passes",
    "drop_verdict_claims",
    "verdicts_in",
]

SUMMARY_SYSTEM: Final = (
    "You write one or two short sentences (at most 45 words) for a mechanical engineer watching an "
    "AI agent design gripper fingers for their part. Use only the facts and numbers in the JSON you "
    "are given. Pass/fail comes from the physics simulator: mention a verdict only if the JSON has a "
    "'verdict' field, repeat it exactly, and never upgrade it. Plain text, no markdown, no lists."
)
_MAX_PAYLOAD_CHARS: Final = 3500


class LightningSummarizer:
    """Lightning with thinking off (``chat_template_kwargs.enable_thinking = false``)."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.last_meta: CallMeta | None = None

    @classmethod
    def create(
        cls, ledger: Any, *, cache: Any | None = None, transport: Any | None = None
    ) -> LightningSummarizer:
        return cls(
            LLMClient(
                NEMOTRON_LIGHTNING,
                temperature=0.3,
                top_p=0.95,
                max_tokens=200,
                cache=cache,
                ledger=ledger,
                transport=transport,
                timeout_s=60.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        )

    def summarize(self, kind: str, payload: Mapping[str, Any]) -> str | None:
        """One summary of ``payload`` (a tool result or the session outcome), or ``None``."""
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        if len(text) > _MAX_PAYLOAD_CHARS:
            text = text[:_MAX_PAYLOAD_CHARS] + " ...(truncated)"
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Step: {kind}\nJSON:\n{text}"},
        ]
        self.last_meta = None
        try:
            answer, meta = self.client.chat(messages, tag=f"agent.summary.{kind}")
        except (LLMError, BudgetExceededError, CacheMiss):
            return None
        self.last_meta = meta
        answer = " ".join(answer.split())
        verdicts = verdicts_in(payload)
        if not verdicts:
            answer = drop_verdict_claims(answer)
        elif "pass" not in verdicts:
            answer = drop_unsupported_passes(answer)
        return answer[:400] or None


_SENTENCE_RE: Final = re.compile(r"(?<=[.!?])\s+")
_PASS_RE: Final = re.compile(r"\b(pass|passed|passes|passing)\b", re.IGNORECASE)
_VERDICT_KEYS: Final = frozenset({"verdict", "certificate_verdict"})
_VERDICT_RE: Final = re.compile(r"\b(verdict|passed|passes|pass|failed|fails)\b", re.IGNORECASE)


def drop_verdict_claims(text: str) -> str:
    """Remove sentences that state a pass/fail when the step produced no simulator verdict.

    Code, not the summarizer, decides what may be called a pass: a summary of a step
    without a ``verdict`` (for example ``inspect_part``) must not announce one.
    """
    return " ".join(s for s in _SENTENCE_RE.split(text) if not _VERDICT_RE.search(s)).strip()


def verdicts_in(payload: Any) -> set[str]:
    """Every code-computed verdict in ``payload`` (values of ``verdict``/``certificate_verdict`` keys)."""
    found: set[str] = set()
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in _VERDICT_KEYS and isinstance(value, str):
                found.add(value.strip().lower())
            else:
                found |= verdicts_in(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found |= verdicts_in(item)
    return found


def drop_unsupported_passes(text: str) -> str:
    """Remove sentences that announce a pass when no verdict of the step is a pass.

    The prompt says never to upgrade a verdict; this enforces the dangerous direction in
    code (a sentence such as "the pass threshold is 80%" goes too, which is the safe side).
    """
    return " ".join(s for s in _SENTENCE_RE.split(text) if not _PASS_RE.search(s)).strip()
