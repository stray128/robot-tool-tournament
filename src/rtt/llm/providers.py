"""Model catalogue, credentials and HTTP transport for Nebius Token Factory.

Token Factory exposes an OpenAI-compatible API, so the default transport is a
thin wrapper around the ``openai`` SDK. The transport never retries and never
raises on HTTP status codes: it hands status, headers and body back to
:class:`rtt.llm.client.LLMClient`, which owns retries, rate limiting, caching
and the ledger. Tests swap in a fake transport that implements the same
:class:`Transport` protocol.

Secret hygiene: the API key is read from the environment or a ``.env`` file and
handed to the SDK client. The transport keeps a private copy only to scrub it
from error text; it is never logged, shown in ``repr`` or included in exception
messages, and ambient ``OPENAI_ORG_ID`` / ``OPENAI_PROJECT_ID`` values are not
forwarded to Token Factory.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

PROVIDER = "nebius-tokenfactory"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
API_KEY_ENV = "NEBIUS_API_KEY"
BASE_URL_ENV = "NEBIUS_BASE_URL"

NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
NEMOTRON_NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
NEMOTRON_ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"


@dataclass(frozen=True)
class ModelInfo:
    """Static pricing and rate-limit facts for one hosted model.

    Prices are USD per one million tokens, taken from the Token Factory model
    catalogue on 2026-10-15.
    """

    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    tokens_per_minute: int
    requests_per_minute: int

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of a call with the given token counts."""
        return (
            prompt_tokens * self.input_usd_per_mtok + completion_tokens * self.output_usd_per_mtok
        ) / 1_000_000


MODELS: dict[str, ModelInfo] = {
    info.model_id: info
    for info in (
        ModelInfo(NEMOTRON_SUPER, 0.30, 0.90, tokens_per_minute=200_000, requests_per_minute=300),
        ModelInfo(NEMOTRON_LIGHTNING, 0.06, 0.24, tokens_per_minute=400_000, requests_per_minute=600),
        ModelInfo(NEMOTRON_NANO, 0.06, 0.24, tokens_per_minute=800_000, requests_per_minute=100),
        ModelInfo(NEMOTRON_ULTRA, 1.00, 3.00, tokens_per_minute=200_000, requests_per_minute=300),
    )
}


def model_info(model_id: str) -> ModelInfo | None:
    """Return catalogue facts for ``model_id``, or ``None`` if it is unknown."""
    return MODELS.get(model_id)


_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s\"',}]+")


class MissingCredentialsError(RuntimeError):
    """Raised when no API key can be found."""


def load_api_key(env_file: str | Path | None = None) -> str:
    """Read the Token Factory API key without exporting or printing it.

    The process environment wins; otherwise ``env_file`` (or the nearest
    ``.env`` found from the current directory upward) is parsed.
    """
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return key
    from dotenv import dotenv_values, find_dotenv

    path = str(env_file) if env_file is not None else find_dotenv(usecwd=True)
    if path:
        key = (dotenv_values(path).get(API_KEY_ENV) or "").strip()
    if not key:
        raise MissingCredentialsError(f"{API_KEY_ENV} is not set in the environment or in a .env file")
    return key


def load_base_url(env_file: str | Path | None = None) -> str:
    """Base URL from ``NEBIUS_BASE_URL`` (environment or ``.env``), else the default."""
    url = os.environ.get(BASE_URL_ENV, "").strip()
    if not url:
        from dotenv import dotenv_values, find_dotenv

        path = str(env_file) if env_file is not None else find_dotenv(usecwd=True)
        if path:
            url = (dotenv_values(path).get(BASE_URL_ENV) or "").strip()
    return url or DEFAULT_BASE_URL


@dataclass(frozen=True)
class TransportResponse:
    """Status, headers and decoded JSON body of one HTTP exchange."""

    status_code: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    error_text: str = ""


class TransportError(Exception):
    """A request that produced no HTTP response (timeout, DNS, reset, ...)."""


class Transport(Protocol):
    """Minimal interface :class:`rtt.llm.client.LLMClient` needs from a backend."""

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        """POST ``payload`` to ``/chat/completions``. Must not raise on HTTP errors."""
        ...

    def list_models(self, timeout_s: float) -> TransportResponse:
        """GET ``/models``. Must not raise on HTTP errors."""
        ...


def redact(text: str, secret: str | None) -> str:
    """Remove ``secret`` and any ``Bearer`` token value from ``text``."""
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return _BEARER_RE.sub(r"\1[REDACTED]", text)


class OpenAITransport:
    """Transport backed by the ``openai`` SDK, pointed at Token Factory.

    SDK retries are disabled (``max_retries=0``) because the client implements
    its own backoff that understands the ``x-ratelimit-*`` headers.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        http_client: Any | None = None,
    ) -> None:
        from openai import OpenAI

        self._sdk = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client)
        # The SDK picks up OPENAI_ORG_ID / OPENAI_PROJECT_ID from the environment and
        # would send them to Token Factory as headers; they belong to another provider.
        self._sdk.organization = None
        self._sdk.project = None
        # Kept only so error text can be scrubbed; never exposed or logged.
        self._redact_value = api_key
        self.base_url = base_url

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> OpenAITransport:
        """Build a transport from ``NEBIUS_API_KEY`` / ``NEBIUS_BASE_URL``."""
        return cls(load_api_key(env_file), load_base_url(env_file))

    def __repr__(self) -> str:
        return f"OpenAITransport(base_url={self.base_url!r})"

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        body = dict(payload)
        model = body.pop("model")
        messages = body.pop("messages")
        # Everything else goes through ``extra_body`` so the SDK forwards it
        # verbatim (including provider extensions such as chat_template_kwargs).
        return self._call(
            lambda: self._sdk.chat.completions.with_raw_response.create(
                model=model, messages=messages, extra_body=body, timeout=timeout_s
            )
        )

    def list_models(self, timeout_s: float) -> TransportResponse:
        return self._call(lambda: self._sdk.models.with_raw_response.list(timeout=timeout_s))

    def _call(self, send: Any) -> TransportResponse:
        import openai

        try:
            raw = send()
        except openai.APIStatusError as exc:
            body = exc.body if isinstance(exc.body, dict) else None
            text = redact(str(getattr(exc, "message", "") or ""), self._redact_value)
            return TransportResponse(
                status_code=exc.status_code,
                headers=_lower_headers(exc.response.headers),
                body=_scrub_body(body, self._redact_value),
                error_text=text[:2000],
            )
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransportError(type(exc).__name__) from None
        text = raw.text
        try:
            decoded = json.loads(text) if text else None
        except json.JSONDecodeError:
            decoded = None
        return TransportResponse(
            status_code=raw.status_code,
            headers=_lower_headers(raw.headers),
            body=decoded if isinstance(decoded, dict) else None,
            error_text="" if isinstance(decoded, dict) else redact(text[:2000], self._redact_value),
        )


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items() if str(k).lower() != "authorization"}


def _scrub_body(body: dict[str, Any] | None, secret: str) -> dict[str, Any] | None:
    if body is None:
        return None
    return json.loads(redact(json.dumps(body), secret))
