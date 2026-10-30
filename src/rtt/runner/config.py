"""Settings of the hosted runner: caps, providers, the Nebius job and where state lives.

Everything is plain configuration with a default; nothing here is a secret. In AWS the values
arrive as Lambda environment variables set by the CDK stack (``infra/runner``); locally,
``rtt runner serve`` builds them from its flags. Account IDs (the Nebius project, subnet, bucket
and image) are never written in the repository: they come from the environment at run time.

Money is kept in integer micro-dollars (1 USD = 1,000,000) wherever it is stored or compared.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import date
from typing import Final

from rtt.llm.providers import DEFAULT_BASE_URL, NEMOTRON_SUPER

__all__ = [
    "DEFAULT_ORIGINS",
    "MICROS",
    "NVIDIA_BASE_URL",
    "PROVIDERS",
    "Caps",
    "NebiusJobConfig",
    "ProviderSpec",
    "RunnerConfig",
    "to_micros",
    "to_usd",
]

MICROS: Final = 1_000_000
#: The site and its local dev server; nothing else may call the runner from a browser.
DEFAULT_ORIGINS: Final = ("https://stray128.github.io", "http://localhost:4341")
#: NVIDIA's hosted API (docs.api.nvidia.com, model reference for nemotron-3-super-120b-a12b,
#: read 2026-10-28: ``POST https://integrate.api.nvidia.com/v1/chat/completions``).
NVIDIA_BASE_URL: Final = "https://integrate.api.nvidia.com/v1"
PROVIDERS: Final = ("nvidia", "tokenfactory", "free")


def to_micros(usd: float) -> int:
    return round(usd * MICROS)


def to_usd(micros: int) -> float:
    return round(micros / MICROS, 6)


@dataclass(frozen=True)
class Caps:
    """Hard caps, checked before any money is spent. Days are UTC days."""

    free_plans_per_day: int = 20
    free_plans_per_ip_per_day: int = 3
    #: Plans on the visitor's own key cost us nothing but Lambda time; still bounded, per visitor
    #: and site-wide.
    keyed_plans_per_ip_per_day: int = 30
    keyed_plans_per_day: int = 300
    runs_per_ip_per_day: int = 2
    spend_per_day_usd: float = 2.0
    spend_total_usd: float = 10.0
    #: The most one plan may spend on the model (both calls), as the Engineer's API does.
    plan_budget_usd: float = 0.05
    #: Creation to finish of one job; past it the job is cancelled.
    job_wall_limit_s: int = 1200
    #: What a cancelled job may still run before it stops (Nebius cancels slowly), reserved too.
    cancel_grace_s: int = 300


@dataclass(frozen=True)
class ProviderSpec:
    """An OpenAI-compatible endpoint and the model the Task Engineer calls there."""

    name: str
    base_url: str
    model: str


@dataclass(frozen=True)
class NebiusJobConfig:
    """Where and on what each run's job goes. IDs are account settings, never shown."""

    project_id: str | None = None
    subnet_id: str | None = None
    bucket_id: str | None = None
    bucket_name: str | None = None
    s3_endpoint: str = "https://storage.uk-south2.nebius.cloud"
    s3_region: str = "uk-south2"
    image: str | None = None  # registry path @sha256:digest
    region: str = "uk-south2"
    platform: str = "cpu-d3"
    preset: str = "32vcpu-128gb"
    job_timeout: str = "1h"  # the platform's own backstop; the runner cancels at the wall limit
    #: The search stops itself this long after the container starts: the wall limit (counted from
    #: job creation) less about 4 minutes for provisioning, the image pull and publishing.
    deadline_s: int = 960
    cli: str = "nebius"

    def __repr__(self) -> str:
        shown = ("region", "platform", "preset", "job_timeout", "deadline_s")
        parts = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if f.name in shown]
        hidden = [f"{f.name}=<set>" for f in fields(self) if f.name not in shown and getattr(self, f.name)]
        return "NebiusJobConfig(" + ", ".join(parts + hidden) + ")"

    def missing(self) -> list[str]:
        return [
            n
            for n in ("project_id", "subnet_id", "bucket_id", "bucket_name", "image")
            if not getattr(self, n)
        ]


@dataclass(frozen=True)
class RunnerConfig:
    caps: Caps = field(default_factory=Caps)
    #: The runner serves through this UTC day and answers 503 from the next one on.
    auto_off: date = date(2026, 12, 15)
    kill_switch: bool = False
    #: SSM parameter holding "on" or "off" (read every ``kill_switch_ttl_s``); ``None``: env only.
    kill_switch_param: str | None = None
    kill_switch_ttl_s: float = 30.0
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS
    #: Behind a Lambda Function URL the URL's own CORS config sends the headers (sending them
    #: twice breaks browsers), so the function only checks the origin; ``rtt runner serve`` sends them.
    emit_cors: bool = False
    max_body_bytes: int = 64 * 1024
    max_description_chars: int = 4000
    max_key_chars: int = 512
    max_plan_bytes: int = 48 * 1024
    max_objects: int = 6
    max_steps: int = 8
    eta_s: int = 480
    #: A run's job is polled at most this often by GET /runs/{id} (the sweeper polls too).
    poll_interval_s: float = 10.0
    report_url_ttl_s: int = 900
    run_ttl_days: int = 30
    #: Per call to the model: the HTTP timeout and retries. Two calls (plan and repair) must fit
    #: in the Lambda timeout (360 s in the stack).
    call_timeout_s: float = 75.0
    call_retries: int = 1
    #: The function's own timeout (the stack sets both). A run with no job id yet is left alone for
    #: this long plus 2 minutes: until then the request that is submitting it may still be alive.
    lambda_timeout_s: int = 360
    providers: Mapping[str, ProviderSpec] = field(
        default_factory=lambda: {
            "nvidia": ProviderSpec("nvidia", NVIDIA_BASE_URL, NEMOTRON_SUPER),
            "tokenfactory": ProviderSpec("tokenfactory", DEFAULT_BASE_URL, NEMOTRON_SUPER),
            "free": ProviderSpec("free", DEFAULT_BASE_URL, NEMOTRON_SUPER),
        }
    )
    nebius: NebiusJobConfig = field(default_factory=NebiusJobConfig)
    table_name: str | None = None
    tokenfactory_secret: str = "effector-lab/runner/tokenfactory"
    nebius_secret: str = "effector-lab/runner/nebius-sa"

    @property
    def unsubmitted_grace_s(self) -> int:
        """How long a run with no job id is left alone: the function timeout plus 2 minutes."""
        return self.lambda_timeout_s + 120

    def job_env(self) -> dict[str, str]:
        """The limits the job re-checks the plan against (the job reads them with ``from_env``)."""
        return {
            "RUNNER_MAX_OBJECTS": str(self.max_objects),
            "RUNNER_MAX_STEPS": str(self.max_steps),
            "RUNNER_MAX_PLAN_BYTES": str(self.max_plan_bytes),
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RunnerConfig:
        """Read ``RUNNER_*`` variables (unset ones keep their defaults)."""
        env = os.environ if environ is None else environ
        base = cls()

        def get(name: str) -> str | None:
            value = env.get(f"RUNNER_{name}")
            return value if value not in (None, "") else None

        caps = base.caps
        for f in fields(Caps):
            raw = get(f.name.upper())
            if raw is not None:
                caps = replace(caps, **{f.name: type(getattr(caps, f.name))(raw)})
        nebius = replace(base.nebius, deadline_s=max(300, caps.job_wall_limit_s - 240))
        for f in fields(NebiusJobConfig):
            raw = get(f"NEBIUS_{f.name.upper()}")
            if raw is not None:
                nebius = replace(nebius, **{f.name: int(raw) if f.name == "deadline_s" else raw})
        providers = dict(base.providers)
        for name in ("nvidia", "tokenfactory"):
            url, model = get(f"{name.upper()}_BASE_URL"), get(f"{name.upper()}_MODEL")
            spec = providers[name]
            providers[name] = ProviderSpec(name, url or spec.base_url, model or spec.model)
        providers["free"] = ProviderSpec(
            "free", providers["tokenfactory"].base_url, providers["tokenfactory"].model
        )
        out = replace(base, caps=caps, nebius=nebius, providers=providers)
        simple = {
            "AUTO_OFF": ("auto_off", date.fromisoformat),
            "KILL_SWITCH": ("kill_switch", _truthy),
            "KILL_SWITCH_PARAM": ("kill_switch_param", str),
            "KILL_SWITCH_TTL_S": ("kill_switch_ttl_s", float),
            "ALLOWED_ORIGINS": (
                "allowed_origins",
                lambda v: tuple(o.strip() for o in v.split(",") if o.strip()),
            ),
            "EMIT_CORS": ("emit_cors", _truthy),
            "MAX_BODY_BYTES": ("max_body_bytes", int),
            "MAX_DESCRIPTION_CHARS": ("max_description_chars", int),
            "MAX_PLAN_BYTES": ("max_plan_bytes", int),
            "MAX_OBJECTS": ("max_objects", int),
            "MAX_STEPS": ("max_steps", int),
            "ETA_S": ("eta_s", int),
            "POLL_INTERVAL_S": ("poll_interval_s", float),
            "REPORT_URL_TTL_S": ("report_url_ttl_s", int),
            "RUN_TTL_DAYS": ("run_ttl_days", int),
            "CALL_TIMEOUT_S": ("call_timeout_s", float),
            "CALL_RETRIES": ("call_retries", int),
            "LAMBDA_TIMEOUT_S": ("lambda_timeout_s", int),
            "TABLE_NAME": ("table_name", str),
            "TOKENFACTORY_SECRET": ("tokenfactory_secret", str),
            "NEBIUS_SECRET": ("nebius_secret", str),
        }
        for var, (name, parse) in simple.items():
            raw = get(var)
            if raw is not None:
                out = replace(out, **{name: parse(raw)})
        return out


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "on", "yes")
