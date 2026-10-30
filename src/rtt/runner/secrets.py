"""The runner's two secrets, read at run time and kept in memory for the container's lifetime.

* ``effector-lab/runner/tokenfactory``: a plain string, the Token Factory key of the free tier.
* ``effector-lab/runner/nebius-sa``: JSON, :class:`NebiusCredentials`::

      {
        "service_account_credentials": {"subject-credentials": {...}},
        "s3_access_key_id": "...",
        "s3_secret_access_key": "..."
      }

  ``service_account_credentials`` is the file ``nebius iam auth-public-key generate
  --output-format service-account-json`` writes for the runner's service account; the S3 pair is
  an access key of that account for the results bucket.

Values never appear in a ``repr``, an exception message or a log line; errors name the secret
and the missing field only.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["NebiusCredentials", "SecretError", "SecretSource", "SecretsManagerSource", "StaticSecrets"]


class SecretError(RuntimeError):
    """A secret is missing or malformed (the message never holds a value)."""


class SecretSource(Protocol):
    def get(self, secret_id: str) -> str: ...


class SecretsManagerSource:
    """Secrets Manager through boto3, each secret fetched once per container."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"SecretsManagerSource(cached={len(self._cache)})"

    def get(self, secret_id: str) -> str:
        with self._lock:
            if secret_id in self._cache:
                return self._cache[secret_id]
            if self._client is None:
                import boto3

                self._client = boto3.client("secretsmanager")
            try:
                resp = self._client.get_secret_value(SecretId=secret_id)
            except Exception as exc:  # noqa: BLE001 - the SDK's text may echo request details
                raise SecretError(f"secret {secret_id} could not be read ({type(exc).__name__})") from None
            value = resp.get("SecretString")
            if not isinstance(value, str) or not value.strip():
                raise SecretError(f"secret {secret_id} has no string value")
            self._cache[secret_id] = value.strip()
            return self._cache[secret_id]


class StaticSecrets:
    """Secrets from a mapping (tests and ``rtt runner serve``)."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    def __repr__(self) -> str:
        return f"StaticSecrets(names={sorted(self._values)!r})"

    def get(self, secret_id: str) -> str:
        if secret_id not in self._values:
            raise SecretError(f"secret {secret_id} is not set")
        return self._values[secret_id]


@dataclass(frozen=True)
class NebiusCredentials:
    service_account_credentials: Mapping[str, Any] = field(repr=False)
    s3_access_key_id: str = field(repr=False)
    s3_secret_access_key: str = field(repr=False)

    def __repr__(self) -> str:
        return "NebiusCredentials(<hidden>)"

    @classmethod
    def parse(cls, text: str, secret_id: str = "nebius-sa") -> NebiusCredentials:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise SecretError(f"secret {secret_id} is not JSON") from None
        if not isinstance(data, Mapping):
            raise SecretError(f"secret {secret_id} must be a JSON object")
        sa = data.get("service_account_credentials")
        if not isinstance(sa, Mapping) or not isinstance(sa.get("subject-credentials"), Mapping):
            raise SecretError(
                f"secret {secret_id}: service_account_credentials must hold the service-account JSON "
                "(with subject-credentials)"
            )
        missing = [
            k
            for k in ("s3_access_key_id", "s3_secret_access_key")
            if not isinstance(data.get(k), str) or not data[k]
        ]
        if missing:
            raise SecretError(f"secret {secret_id}: missing {', '.join(missing)}")
        return cls(dict(sa), data["s3_access_key_id"], data["s3_secret_access_key"])
