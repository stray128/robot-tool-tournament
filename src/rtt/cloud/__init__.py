"""Nebius AI Cloud: Serverless AI Jobs, Object Storage and job spend.

* :mod:`rtt.cloud.config`: account settings from env vars and ``~/.config/rtt/*.env``,
  per region (``NebiusConfig.for_region``).
* :mod:`rtt.cloud.nebius_jobs`: submit/poll/logs/cancel/list over the ``nebius`` CLI,
  region profiles (``uks2-cpu`` default, ``eun1-gpu`` fallback), capacity advice, vCPU
  quota and platform fallback.
* :mod:`rtt.cloud.storage`: the results bucket over S3 (boto3, named profile).
* :mod:`rtt.cloud.cost`: price table, estimates and the spend ledger.
* :mod:`rtt.cloud.runs`: run IDs, the job spec for a run, and checks of fetched runs.
* :mod:`rtt.cloud.entry`: what runs inside the job container (``python -m rtt.cloud.entry``).

Laptop CLI: ``python -m rtt.cloud --help``. Nothing here imports boto3 or MuJoCo at
import time.
"""

from rtt.cloud.config import ConfigError, NebiusConfig, load_config
from rtt.cloud.cost import (
    CPU_PRICES,
    PRICES,
    JobSpend,
    SpendCapError,
    SpendLedger,
    estimate_usd,
    hourly_usd,
)
from rtt.cloud.nebius_jobs import (
    DEFAULT_CANDIDATES,
    DEFAULT_TARGET,
    FALLBACK_TARGET,
    TARGETS,
    Candidate,
    CapacityAdvice,
    Job,
    JobSpec,
    NebiusCLI,
    NebiusCLIError,
    NebiusJobs,
    NoCapacityError,
    ProvisioningTimeoutError,
    RunOutcome,
    Target,
    cpu_preset,
    get_target,
    rank_candidates,
)
from rtt.cloud.runs import job_spec_for, new_run_id, verify_run
from rtt.cloud.storage import BucketStore

__all__ = [
    "CPU_PRICES",
    "DEFAULT_CANDIDATES",
    "DEFAULT_TARGET",
    "FALLBACK_TARGET",
    "PRICES",
    "TARGETS",
    "BucketStore",
    "Candidate",
    "CapacityAdvice",
    "ConfigError",
    "Job",
    "JobSpec",
    "JobSpend",
    "NebiusCLI",
    "NebiusCLIError",
    "NebiusConfig",
    "NebiusJobs",
    "NoCapacityError",
    "ProvisioningTimeoutError",
    "RunOutcome",
    "SpendCapError",
    "SpendLedger",
    "Target",
    "cpu_preset",
    "estimate_usd",
    "get_target",
    "hourly_usd",
    "job_spec_for",
    "load_config",
    "new_run_id",
    "rank_candidates",
    "verify_run",
]
