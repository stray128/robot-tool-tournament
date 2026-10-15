"""Nemotron client for Nebius Token Factory: forced tool calls, strict JSON schema, plain chat.

:class:`LLMClient` wraps one model with fixed sampling parameters and adds the
machinery the search needs around every request:

* **Caching.** Requests are content-addressed (:mod:`rtt.llm.cache`). In replay
  mode a miss is an error, so a cached search arm re-runs with zero API calls
  and without needing an API key.
* **Ledger and spend cap.** Every call lands in a :class:`~rtt.llm.ledger.Ledger`
  with tokens and estimated cost. A ledger budget refuses live calls whose
  worst case (prompt estimate plus ``max_tokens`` at output price) could
  exceed it.
* **Rate limiting.** A token bucket throttles to a fraction of the model's
  tokens-per-minute limit and is pulled down to the server's
  ``x-ratelimit-remaining-tokens`` when that header is present.
* **Retries.** 408/409/425/429/5xx and connection errors are retried with
  exponential backoff and jitter, honouring ``retry-after`` and
  ``x-ratelimit-reset-*``. Auth (401/403) and billing (402, "insufficient
  balance") errors fail fast with distinct exception types.
* **Reasoning.** Reasoning returned in a separate field (``reasoning_content``
  or ``reasoning``), or inline in ``<think>`` tags, is kept in
  :attr:`CallMeta.reasoning_text` and never parsed as the answer.

Token Factory has no ``seed`` parameter, so repeated samples of one prompt are
distinguished in the cache key by the ``sample`` index, which is not sent to
the API.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, TypeVar

from rtt.llm.cache import CacheEntry, CacheMiss, ResponseCache, request_hash, wire_json
from rtt.llm.ledger import BudgetExceededError, Ledger, LedgerEntry, utc_now_iso
from rtt.llm.providers import PROVIDER, OpenAITransport, Transport, TransportError, model_info

__all__ = [
    "BudgetExceededError",
    "CacheMiss",
    "CallMeta",
    "LLMAuthError",
    "LLMBillingError",
    "LLMClient",
    "LLMError",
    "LLMHTTPError",
    "LLMParseError",
    "LLMTransportError",
    "RetryPolicy",
    "TokenBucket",
    "extract_json_object",
    "parse_duration_s",
    "shared_limiter",
]

Mode = Literal["tool", "json_schema", "chat"]
Message = Mapping[str, Any]
T = TypeVar("T")

# Whole words only: "balance" must not match "load balancer". "quota" is left out on
# purpose: per-minute rate-limit 429s often say "quota" and must stay retryable.
_BILLING_RE = re.compile(r"\b(insufficient|balance|billing|payment|credits?)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- errors


class LLMError(RuntimeError):
    """Base class for client errors."""


class LLMHTTPError(LLMError):
    """The provider answered with an error status (after any retries)."""

    def __init__(self, message: str, *, status_code: int, request_hash: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_hash = request_hash


class LLMAuthError(LLMHTTPError):
    """401/403: the key is missing, wrong, or not allowed to use the model."""


class LLMBillingError(LLMHTTPError):
    """402 or an "insufficient balance" style refusal: stop spending."""


class LLMTransportError(LLMError):
    """No HTTP response after all retries (timeouts, connection failures)."""


class LLMParseError(LLMError):
    """The response arrived but did not contain the expected structured output.

    ``meta`` describes the call (it was paid for, and is cached) and
    ``raw_text`` holds the text that failed to parse, for repair prompts.
    """

    def __init__(self, message: str, *, raw_text: str = "", notes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.notes = tuple(notes)
        self.meta: CallMeta | None = None


# --------------------------------------------------------------------------- metadata


@dataclass(frozen=True)
class CallMeta:
    """Facts about one call, suitable for manifests (no secrets, no message bodies).

    For cache hits, ``latency_s``, token counts, ``cost_usd`` and
    ``rate_limit_headers`` are those of the original live call; ``cached``
    tells them apart.
    """

    model: str
    mode: str
    request_hash: str
    cached: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int
    finish_reason: str | None = None
    reasoning_tokens: int | None = None
    reasoning_separate: bool = False
    reasoning_field: str | None = None
    reasoning_text: str | None = None
    response_id: str | None = None
    rate_limit_headers: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    provider: str = PROVIDER

    def to_json(self, *, include_reasoning: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["rate_limit_headers"] = dict(self.rate_limit_headers)
        data["notes"] = list(self.notes)
        if not include_reasoning:
            data.pop("reasoning_text")
        return data


# --------------------------------------------------------------------------- retry + rate limit


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter.

    Delay before retry ``n`` (1-based) is ``base_delay_s * 2**(n-1)``, capped at
    ``max_delay_s``, then scaled by ``1 +/- jitter_frac``. Server hints
    (``retry-after``, ``x-ratelimit-reset-*``) replace the exponential term and
    are only ever jittered upwards, so a retry never fires before the server
    said it may.
    """

    max_retries: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    jitter_frac: float = 0.25
    retry_statuses: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def delay_s(self, retry_number: int, rng: random.Random, hint_s: float | None = None) -> float:
        base = hint_s if hint_s is not None else self.base_delay_s * 2 ** (retry_number - 1)
        base = min(max(base, 0.0), self.max_delay_s)
        if self.jitter_frac > 0:
            low = 0.0 if hint_s is not None else -self.jitter_frac
            base *= 1.0 + rng.uniform(low, self.jitter_frac)
        return max(base, 0.0)


class TokenBucket:
    """Tokens-per-minute limiter shared by the threads of one process.

    The bucket holds up to ``tokens_per_minute`` tokens and refills linearly.
    :meth:`acquire` blocks until enough tokens are available; the level may go
    negative after :meth:`adjust` if a call used more than it reserved.
    """

    def __init__(
        self,
        tokens_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        self.capacity = float(tokens_per_minute)
        self._rate_per_s = self.capacity / 60.0
        self._clock = clock
        self._sleep = sleep
        self._level = self.capacity
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._level = min(self.capacity, self._level + (now - self._updated) * self._rate_per_s)
        self._updated = now

    @property
    def level(self) -> float:
        with self._lock:
            self._refill()
            return self._level

    def acquire(self, tokens: float) -> float:
        """Take ``tokens`` (clipped to capacity), sleeping as needed; returns seconds waited."""
        need = min(float(tokens), self.capacity)
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._level >= need:
                    self._level -= need
                    return waited
                wait = (need - self._level) / self._rate_per_s
            self._sleep(wait)
            waited += wait

    def adjust(self, delta: float) -> None:
        """Return unused tokens (``delta > 0``) or charge extra ones (``delta < 0``)."""
        with self._lock:
            self._refill()
            self._level = min(self.capacity, self._level + delta)

    def clamp(self, remaining: float) -> None:
        """Never believe we have more tokens than the server says remain."""
        with self._lock:
            self._refill()
            self._level = min(self._level, float(remaining))


_SHARED_LIMITERS: dict[str, TokenBucket] = {}
_SHARED_LIMITERS_LOCK = threading.Lock()


def shared_limiter(model: str, tpm_fraction: float = 0.75) -> TokenBucket | None:
    """Process-wide token bucket for ``model``, so concurrent clients share one TPM budget.

    The bucket is created on first use at ``tpm_fraction`` of the catalogue
    limit; later callers get the same bucket. Unknown models get ``None``.
    """
    info = model_info(model)
    if info is None:
        return None
    with _SHARED_LIMITERS_LOCK:
        bucket = _SHARED_LIMITERS.get(model)
        if bucket is None:
            bucket = TokenBucket(info.tokens_per_minute * tpm_fraction)
            _SHARED_LIMITERS[model] = bucket
        return bucket


_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration_s(value: str | None) -> float | None:
    """Parse ``retry-after`` / ``x-ratelimit-reset-*`` values into seconds.

    Accepts plain seconds (``"2"``, ``"0.5"``) and Go-style durations
    (``"250ms"``, ``"1s"``, ``"6m0s"``, ``"1h2m3.5s"``). Returns ``None`` for
    anything else, including HTTP dates.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    try:
        return max(float(text), 0.0)
    except ValueError:
        pass
    parts = _DURATION_RE.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[u] for n, u in parts)


# --------------------------------------------------------------------------- parsing helpers

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)


def split_inline_reasoning(text: str) -> tuple[str, str | None]:
    """Separate inline reasoning from the answer.

    Handles closed ``<think>...</think>`` blocks, a dangling ``</think>`` (the
    opening tag was part of the prompt template) and a dangling ``<think>``
    (reasoning cut off by ``max_tokens``): everything after an unclosed
    ``<think>`` is reasoning, never answer.
    """
    thoughts = [m.strip() for m in _THINK_RE.findall(text)]
    answer = _THINK_RE.sub("", text)
    lowered = answer.lower()
    if "</think>" in lowered:  # opening tag was in the prompt template
        cut = lowered.rindex("</think>")
        thoughts.append(answer[:cut].strip())
        answer = answer[cut + len("</think>") :]
        lowered = answer.lower()
    if "<think>" in lowered:  # reasoning never closed: truncated mid-thought
        cut = lowered.index("<think>")
        thoughts.append(answer[cut + len("<think>") :].strip())
        answer = answer[:cut]
    reasoning = "\n\n".join(t for t in thoughts if t) or None
    return answer.strip(), reasoning


def extract_json_object(text: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Parse a JSON object from model text, tolerating fences and inline reasoning.

    Returns the object and notes describing any cleanup that was needed.
    Candidates are tried in order: each fenced code block, last first (models
    put drafts before the final answer); the whole text; the span from the
    first ``{`` to the last ``}``.

    Raises:
        LLMParseError: if no JSON object can be recovered.
    """
    notes: list[str] = []
    body, inline = split_inline_reasoning(text)
    if inline is not None:
        notes.append("think_tags_stripped")
    candidates: list[tuple[str, str | None]] = [
        (block.strip(), "code_fence_stripped") for block in reversed(_FENCE_RE.findall(body))
    ]
    stripped = body.strip()
    candidates.append((stripped, None))
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start and (start, end) != (0, len(stripped) - 1):
        candidates.append((stripped[start : end + 1], "json_substring"))
    for candidate, note in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if note is not None:
                notes.append(note)
            return value, tuple(notes)
    if any(note == "code_fence_stripped" for _, note in candidates):
        notes.append("code_fence_stripped")
    raise LLMParseError("response did not contain a JSON object", raw_text=text, notes=notes)


def _first_choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise LLMParseError("response has no choices")
    return choices[0]


def _message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    message = _first_choice(response).get("message")
    if not isinstance(message, Mapping):
        raise LLMParseError("response choice has no message")
    return message


def _separate_reasoning(message: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(field name, text)`` of reasoning returned outside ``content``, if any."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    return None, None


def _normalize_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Accept ``{"type": "function", "function": {...}}`` or the bare function dict."""
    function = tool.get("function", tool) if tool.get("type", "function") == "function" else None
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise TypeError("tool must be an OpenAI function tool with a name")
    if "parameters" not in function:
        raise ValueError("tool function must define a JSON schema in 'parameters'")
    return {"type": "function", "function": dict(function)}


# --------------------------------------------------------------------------- client


class LLMClient:
    """One model, fixed sampling parameters, cached and ledgered calls.

    Args:
        model: Token Factory model ID, e.g. ``nvidia/nemotron-3-super-120b-a12b``.
        temperature: Sampling temperature (``None`` leaves the server default).
        top_p: Nucleus sampling mass (``None`` leaves the server default).
        max_tokens: Completion budget per call, reasoning included.
        cache: Response cache; ``None`` disables caching.
        ledger: Call ledger (and budget); ``None`` uses a private in-memory one.
        transport: Backend; defaults to :meth:`OpenAITransport.from_env`,
            created on the first live call so cache replays need no key.
        timeout_s: Per-attempt HTTP timeout.
        retry: Backoff policy.
        limiter: Token bucket; defaults to the process-wide bucket for this
            model from :func:`shared_limiter` (no limit for unknown models).
        tpm_fraction: Share of the model's TPM the shared bucket allows when
            this client is the first to create it.
        extra_body: Provider extensions merged into every request (for
            example ``{"chat_template_kwargs": {...}}``); part of the cache key.
        sleep, clock, rng: Injection points for tests.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = 0.6,
        top_p: float | None = 0.95,
        max_tokens: int = 4096,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
        *,
        transport: Transport | None = None,
        timeout_s: float = 120.0,
        retry: RetryPolicy | None = None,
        limiter: TokenBucket | None = None,
        tpm_fraction: float = 0.75,
        extra_body: Mapping[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cache = cache
        self.ledger = ledger if ledger is not None else Ledger()
        self.timeout_s = timeout_s
        self.retry = retry if retry is not None else RetryPolicy()
        self.extra_body = dict(extra_body or {})
        self.info = model_info(model)
        self.limiter = limiter if limiter is not None else shared_limiter(model, tpm_fraction)
        self._transport = transport
        self._transport_lock = threading.Lock()
        self._sleep = sleep
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model!r}, temperature={self.temperature!r}, top_p={self.top_p!r}, "
            f"max_tokens={self.max_tokens!r})"
        )

    # ------------------------------------------------------------------ public API

    def call_tool(
        self,
        messages: Sequence[Message],
        tool: Mapping[str, Any],
        force: bool = True,
        *,
        sample: int = 0,
        tag: str | None = None,
    ) -> tuple[dict[str, Any], CallMeta]:
        """Ask the model to call ``tool`` and return the parsed arguments.

        With ``force=True`` the request sets ``tool_choice`` to the named
        function. If the model answers in plain content instead of a tool
        call, a JSON object in the content is accepted and the meta notes
        ``tool_args_from_content``.
        """
        spec = _normalize_tool(tool)
        name = spec["function"]["name"]
        choice: Any = {"type": "function", "function": {"name": name}} if force else "auto"
        fields = {"tools": [spec], "tool_choice": choice}

        def parse(response: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
            message = _message(response)
            calls = [c for c in message.get("tool_calls") or [] if isinstance(c, Mapping)]
            notes: list[str] = []
            picked = next((c for c in calls if (c.get("function") or {}).get("name") == name), None)
            if picked is None and calls:
                picked = calls[0]
                notes.append("tool_name_mismatch")
            if picked is not None:
                arguments = (picked.get("function") or {}).get("arguments")
                if isinstance(arguments, Mapping):
                    return dict(arguments), tuple(notes)
                if isinstance(arguments, str):
                    try:
                        value = json.loads(arguments)
                    except json.JSONDecodeError:
                        value, extra = extract_json_object(arguments)
                        notes.extend(extra)
                        notes.append("tool_args_lenient")
                    if isinstance(value, dict):
                        return value, tuple(notes)
                raise LLMParseError(
                    "tool call arguments are not a JSON object", raw_text=str(arguments), notes=notes
                )
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                value, extra = extract_json_object(content)
                return value, (*notes, *extra, "tool_args_from_content")
            raise LLMParseError(f"model did not call tool {name!r}", notes=notes)

        return self._run("tool", messages, fields, parse, sample=sample, tag=tag)

    def call_json_schema(
        self,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        *,
        name: str = "response",
        strict: bool = True,
        sample: int = 0,
        tag: str | None = None,
    ) -> tuple[dict[str, Any], CallMeta]:
        """Request ``response_format={"type": "json_schema", ...}`` and parse the content."""
        fields = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": dict(schema), "strict": strict},
            }
        }

        def parse(response: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
            content = _message(response).get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMParseError("response has no content")
            return extract_json_object(content)

        return self._run("json_schema", messages, fields, parse, sample=sample, tag=tag)

    def chat(
        self, messages: Sequence[Message], *, sample: int = 0, tag: str | None = None
    ) -> tuple[str, CallMeta]:
        """Plain completion; returns the answer text with any inline ``<think>`` block removed."""

        def parse(response: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
            content = _message(response).get("content")
            text = content if isinstance(content, str) else ""
            answer, inline = split_inline_reasoning(text)
            return answer, ("think_tags_stripped",) if inline is not None else ()

        return self._run("chat", messages, {}, parse, sample=sample, tag=tag)

    def request_hash_for(self, payload: Mapping[str, Any], sample: int = 0) -> str:
        """Cache key of ``payload`` (exactly what would be sent) at ``sample``."""
        return request_hash({"payload": payload, "sample": sample})

    # ------------------------------------------------------------------ internals

    def build_payload(self, messages: Sequence[Message], fields: Mapping[str, Any]) -> dict[str, Any]:
        """The exact JSON body sent to ``/chat/completions``."""
        normalized = []
        for message in messages:
            if "role" not in message:
                raise ValueError("every message needs a 'role'")
            normalized.append(dict(message))
        if not normalized:
            raise ValueError("messages must not be empty")
        payload: dict[str, Any] = {"model": self.model, "messages": normalized, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        payload.update(self.extra_body)
        payload.update(fields)
        return payload

    def _run(
        self,
        mode: Mode,
        messages: Sequence[Message],
        fields: Mapping[str, Any],
        parse: Callable[[Mapping[str, Any]], tuple[T, tuple[str, ...]]],
        *,
        sample: int,
        tag: str | None,
    ) -> tuple[T, CallMeta]:
        payload = self.build_payload(messages, fields)
        key = self.request_hash_for(payload, sample)
        fetched = self._fetch(mode, payload, key, sample, tag)
        meta = self._meta(mode, key, fetched)
        try:
            value, notes = parse(fetched.response)
        except LLMParseError as exc:
            notes = (*meta.notes, *exc.notes)
            if fetched.finish_reason == "length":
                notes = (*notes, "truncated")
            exc.meta = _replace_notes(meta, notes)
            self._record(exc.meta, ok=False, error=f"LLMParseError: {exc}", tag=tag)
            raise
        meta = _replace_notes(meta, (*meta.notes, *notes))
        self._record(meta, ok=True, error=None, tag=tag)
        return value, meta

    def _get_transport(self) -> Transport:
        with self._transport_lock:
            if self._transport is None:
                self._transport = OpenAITransport.from_env()
            return self._transport

    def _worst_case_cost(self, prompt_tokens: int) -> float:
        if self.info is None:
            if self.ledger.budget_usd is not None:
                raise BudgetExceededError(f"no price known for {self.model!r}; refusing a capped live call")
            return 0.0
        return self.info.cost_usd(prompt_tokens, self.max_tokens)

    def _fetch(self, mode: Mode, payload: dict[str, Any], key: str, sample: int, tag: str | None) -> _Fetched:
        if self.cache is not None:
            entry = self.cache.get(key)  # raises CacheMiss in replay mode
            if entry is not None:
                return _Fetched(
                    entry.response, entry.latency_s, attempts=0, headers=entry.headers, cached=True
                )

        prompt_estimate = estimate_prompt_tokens(payload)
        worst_case = self._worst_case_cost(prompt_estimate)
        self.ledger.reserve(worst_case)
        try:
            reserved_tokens = prompt_estimate + self.max_tokens
            if self.limiter is not None:
                self.limiter.acquire(reserved_tokens)
            fetched = self._send_with_retries(mode, payload, key, tag)
            if self.limiter is not None:
                used = fetched.prompt_tokens + fetched.completion_tokens
                self.limiter.adjust(reserved_tokens - used if used else 0.0)
                self._sync_limiter(fetched.headers)
        finally:
            self.ledger.release(worst_case)

        if self.cache is not None:
            self.cache.put(
                CacheEntry(
                    request_hash=key,
                    request={"payload": payload, "sample": sample},
                    response=fetched.response,
                    latency_s=fetched.latency_s,
                    created_at=utc_now_iso(),
                    headers=dict(fetched.headers),
                )
            )
        return fetched

    def _send_with_retries(self, mode: Mode, payload: dict[str, Any], key: str, tag: str | None) -> _Fetched:
        transport = self._get_transport()
        attempts = 0
        while True:
            attempts += 1
            retries_left = attempts <= self.retry.max_retries
            started = self._clock()
            try:
                result = transport.chat_completions(payload, self.timeout_s)
            except TransportError as exc:
                if retries_left:
                    self._sleep(self.retry.delay_s(attempts, self._rng))
                    continue
                self._record_failure(mode, key, attempts, f"transport: {exc}", tag)
                raise LLMTransportError(
                    f"no response from provider after {attempts} attempts ({exc})"
                ) from None
            latency = self._clock() - started
            headers = {k.lower(): v for k, v in result.headers.items()}
            rate_headers = {
                k: v for k, v in headers.items() if k.startswith("x-ratelimit") or k == "retry-after"
            }
            status = result.status_code
            if 200 <= status < 300 and _is_completion(result.body):
                return _Fetched(result.body, latency, attempts=attempts, headers=rate_headers, cached=False)
            self._sync_limiter(rate_headers)

            detail = _error_detail(result.body, result.error_text)
            message = f"HTTP {status} from {PROVIDER} for {self.model}: {detail}"
            if status == 402 or (status in (403, 429) and _mentions_billing(detail)):
                self._record_failure(mode, key, attempts, f"billing: HTTP {status}", tag)
                raise LLMBillingError(message, status_code=status, request_hash=key)
            if status in (401, 403):
                self._record_failure(mode, key, attempts, f"auth: HTTP {status}", tag)
                raise LLMAuthError(message, status_code=status, request_hash=key)
            retryable = status in self.retry.retry_statuses or (200 <= status < 300)
            if retryable and retries_left:
                hint = _retry_hint_s(headers) if status in (429, 503) else None
                self._sleep(self.retry.delay_s(attempts, self._rng, hint))
                continue
            self._record_failure(mode, key, attempts, f"HTTP {status}", tag)
            raise LLMHTTPError(message, status_code=status, request_hash=key)

    def _sync_limiter(self, headers: Mapping[str, str]) -> None:
        """Pull the bucket down to the server-reported remaining token count."""
        remaining = headers.get("x-ratelimit-remaining-tokens")
        if self.limiter is None or remaining is None:
            return
        try:
            self.limiter.clamp(float(remaining))
        except ValueError:
            pass

    def _meta(self, mode: Mode, key: str, fetched: _Fetched) -> CallMeta:
        response = fetched.response
        notes: list[str] = []
        reasoning_field: str | None = None
        reasoning_text: str | None = None
        try:
            message = _message(response)
            reasoning_field, reasoning_text = _separate_reasoning(message)
            content = message.get("content")
            if reasoning_text is None and isinstance(content, str):
                reasoning_text = split_inline_reasoning(content)[1]
        except LLMParseError:
            pass
        cost = self.info.cost_usd(fetched.prompt_tokens, fetched.completion_tokens) if self.info else 0.0
        if fetched.finish_reason == "length":
            notes.append("finish_reason_length")
        return CallMeta(
            model=self.model,
            mode=mode,
            request_hash=key,
            cached=fetched.cached,
            latency_s=round(fetched.latency_s, 4),
            prompt_tokens=fetched.prompt_tokens,
            completion_tokens=fetched.completion_tokens,
            cost_usd=cost,
            attempts=fetched.attempts,
            finish_reason=fetched.finish_reason,
            reasoning_tokens=fetched.reasoning_tokens,
            reasoning_separate=reasoning_field is not None,
            reasoning_field=reasoning_field,
            reasoning_text=reasoning_text,
            response_id=response.get("id") if isinstance(response.get("id"), str) else None,
            rate_limit_headers=fetched.headers,
            notes=tuple(notes),
        )

    def _record(self, meta: CallMeta, *, ok: bool, error: str | None, tag: str | None) -> None:
        self.ledger.record(
            LedgerEntry(
                model=meta.model,
                mode=meta.mode,
                request_hash=meta.request_hash,
                cached=meta.cached,
                ok=ok,
                latency_s=meta.latency_s,
                prompt_tokens=meta.prompt_tokens,
                completion_tokens=meta.completion_tokens,
                cost_usd=meta.cost_usd,
                attempts=meta.attempts,
                provider=meta.provider,
                error=error,
                tag=tag,
            )
        )

    def _record_failure(self, mode: Mode, key: str, attempts: int, error: str, tag: str | None) -> None:
        self.ledger.record(
            LedgerEntry(
                model=self.model,
                mode=mode,
                request_hash=key,
                cached=False,
                ok=False,
                latency_s=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                attempts=attempts,
                error=error,
                tag=tag,
            )
        )


@dataclass(frozen=True)
class _Fetched:
    response: dict[str, Any]
    latency_s: float
    attempts: int
    headers: Mapping[str, str]
    cached: bool

    @property
    def _usage(self) -> Mapping[str, Any]:
        usage = self.response.get("usage")
        return usage if isinstance(usage, Mapping) else {}

    @property
    def prompt_tokens(self) -> int:
        return int(self._usage.get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self._usage.get("completion_tokens") or 0)

    @property
    def reasoning_tokens(self) -> int | None:
        details = self._usage.get("completion_tokens_details")
        if isinstance(details, Mapping) and details.get("reasoning_tokens") is not None:
            return int(details["reasoning_tokens"])
        return None

    @property
    def finish_reason(self) -> str | None:
        try:
            reason = _first_choice(self.response).get("finish_reason")
        except LLMParseError:
            return None
        return reason if isinstance(reason, str) else None


def _is_completion(body: Mapping[str, Any] | None) -> bool:
    """True for a body with at least one choice; error bodies sent with 200 are not cached."""
    if not isinstance(body, Mapping):
        return False
    choices = body.get("choices")
    return isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], Mapping)


def _replace_notes(meta: CallMeta, notes: Sequence[str]) -> CallMeta:
    return replace(meta, notes=tuple(dict.fromkeys(notes)))


def estimate_prompt_tokens(payload: Mapping[str, Any]) -> int:
    """Conservative prompt-size estimate (2 characters per token plus template overhead).

    Counts messages, tools and response format; used only for rate limiting
    and worst-case budget checks before the real usage is known. Chat
    templates render tool schemas more verbosely than compact JSON (the probe
    measured about 2.5 characters per prompt token in tool mode), so 3
    characters per token under-reserved; 2 keeps the estimate an upper bound
    for JSON- and number-heavy prompts.
    """
    parts = {k: payload[k] for k in ("messages", "tools", "response_format") if k in payload}
    overhead = 16 * len(payload.get("messages", ())) + 64 * len(payload.get("tools", ())) + 32
    return len(wire_json(parts)) // 2 + overhead


def _retry_hint_s(headers: Mapping[str, str]) -> float | None:
    for name in (
        "retry-after",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset",
    ):
        hint = parse_duration_s(headers.get(name))
        if hint is not None:
            return hint
    return None


def _error_detail(body: Mapping[str, Any] | None, error_text: str) -> str:
    if isinstance(body, Mapping):
        error = body.get("error", body.get("detail", body.get("message")))
        if isinstance(error, Mapping):
            error = error.get("message") or error.get("code") or json.dumps(error)[:300]
        if error:
            return str(error)[:300]
    return (error_text or "no detail")[:300]


def _mentions_billing(detail: str) -> bool:
    return _BILLING_RE.search(detail) is not None
