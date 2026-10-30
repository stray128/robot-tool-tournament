"""The Task Engineer behind the runner's three providers (``POST /plan``).

:func:`rtt.studio.engineer.plan_task` does the planning exactly as the Studio does: the current
prompt (:data:`rtt.studio.engineer.prompts.CURRENT`) and its rule set, one forced call, one repair
call, the layout fixer and the refusal fallback. This module only chooses the endpoint and key:

* ``nvidia``: NVIDIA's API with the visitor's key;
* ``tokenfactory``: Token Factory with the visitor's key;
* ``free``: Token Factory with the runner's own key (read from Secrets Manager by the caller).

Key hygiene. The key reaches the OpenAI-compatible transport and nothing else: no cache (the
client has none), no file, no log line. :class:`PlanOutcome` is built from the plan result only,
and :func:`redact` removes the key from every string of it in case a provider echoed it (the
transport already scrubs it from error text). A rejected key raises :class:`ProviderRejected`,
which carries a kind and no text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rtt.llm.client import LLMAuthError, LLMBillingError, LLMClient, RetryPolicy, TokenBucket
from rtt.llm.ledger import Ledger
from rtt.llm.providers import OpenAITransport, Transport
from rtt.runner.config import ProviderSpec, RunnerConfig
from rtt.studio.engineer import prompts
from rtt.studio.engineer.engineer import ENGINEER_RETRY, ENGINEER_SAMPLING, PlanResult, plan_task

__all__ = ["PlanOutcome", "Planner", "ProviderRejected", "TransportFactory", "default_transport", "redact"]

TransportFactory = Callable[[ProviderSpec, str], Transport]


class ProviderRejected(Exception):
    """The provider refused the key (``auth``) or its account (``billing``). No provider text."""

    def __init__(self, kind: Literal["auth", "billing"]) -> None:
        super().__init__(kind)
        self.kind = kind


def default_transport(provider: ProviderSpec, key: str) -> Transport:
    return OpenAITransport(key, provider.base_url)


def redact(value: Any, secret: str | None) -> Any:
    """``value`` with every occurrence of ``secret`` in its strings replaced by ``[redacted]``."""
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[redacted]")
    if isinstance(value, Mapping):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v, secret) for v in value]
    return value


@dataclass(frozen=True)
class PlanOutcome:
    """The ``POST /plan`` body: ``status`` is ``valid``, ``refused`` or ``failed`` (the Engineer's
    ``invalid``, a plan still wrong after the repair call, counts as failed)."""

    status: Literal["valid", "failed", "refused"]
    plan: dict[str, Any] | None
    problems: list[str]
    calls: int
    cost_usd: float
    model: str
    task_id: str | None = None

    def body(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan,
            "problems": self.problems,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "model": self.model,
        }

    @classmethod
    def of(cls, result: PlanResult, model: str, key: str | None) -> PlanOutcome:
        status: Literal["valid", "failed", "refused"]
        if result.status == "valid":
            status = "valid"
        elif result.status == "refused":
            status = "refused"
        else:
            status = "failed"
        spec = result.spec if status != "failed" else None
        plan = spec.to_dict() if spec is not None else None
        problems = list(result.problems)
        if status == "failed" and not problems:
            problems = [a.error or a.outcome for a in result.attempts if a.outcome != "valid"]
        return cls(
            status=status,
            plan=redact(plan, key),
            problems=redact(problems, key)[:40],
            calls=len(result.attempts),
            cost_usd=round(result.cost_usd, 6),
            model=model,
            task_id=spec.task_id if spec is not None and status == "valid" else None,
        )


class Planner:
    """Plans a description on one provider (module docstring).

    ``transport_factory`` builds the transport from the provider and key (tests pass fakes);
    ``limiter`` overrides the shared token bucket (tests pass an unlimited one).
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        transport_factory: TransportFactory = default_transport,
        limiter: TokenBucket | None = None,
    ) -> None:
        self.cfg = cfg
        self.transport_factory = transport_factory
        self.limiter = limiter

    def client(self, provider: ProviderSpec, key: str) -> LLMClient:
        retry = RetryPolicy(
            max_retries=self.cfg.call_retries,
            base_delay_s=2.0,
            max_delay_s=10.0,
            retry_statuses=ENGINEER_RETRY.retry_statuses,
        )
        return LLMClient(
            provider.model,
            temperature=float(ENGINEER_SAMPLING["temperature"]),
            top_p=float(ENGINEER_SAMPLING["top_p"]),
            max_tokens=int(ENGINEER_SAMPLING["max_tokens"]),
            cache=None,
            ledger=Ledger(None, budget_usd=self.cfg.caps.plan_budget_usd),
            transport=self.transport_factory(provider, key),
            timeout_s=self.cfg.call_timeout_s,
            retry=retry,
            limiter=self.limiter,
        )

    def plan(self, description: str, provider: ProviderSpec, key: str) -> PlanOutcome:
        """Plan ``description`` with the current prompt; raises :class:`ProviderRejected`."""
        client = self.client(provider, key)
        try:
            result = plan_task(description, client=client, version=prompts.CURRENT)
        except LLMAuthError:
            raise ProviderRejected("auth") from None
        except LLMBillingError:
            raise ProviderRejected("billing") from None
        return PlanOutcome.of(result, provider.model, key)
