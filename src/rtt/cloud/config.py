"""Nebius AI Cloud settings, read from the environment and local env files.

Resource IDs (tenant, project, subnet, bucket, registry) are private to the account
that owns them, so none is written in this repository. They come from, in order of
precedence:

1. process environment variables (``NEBIUS_PROJECT_ID=... rtt ...``);
2. ``$RTT_NEBIUS_ENV`` if set, else ``~/.config/rtt/nebius.env``;
3. ``~/.config/rtt/image.env``, written by ``jobs/nebius/push_image.sh`` with the
   digest reference of the last pushed image (``RTT_IMAGE``).

The env files use ``KEY=value`` lines (``export`` prefixes, quotes and ``#`` comments
are accepted). :class:`NebiusConfig` never shows ID values in its ``repr``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

ENV_FILE_VAR = "RTT_NEBIUS_ENV"
DEFAULT_ENV_FILE = Path("~/.config/rtt/nebius.env")
DEFAULT_IMAGE_FILE = Path("~/.config/rtt/image.env")
DEFAULT_CLI = Path("~/.nebius/bin/nebius")

DEFAULT_REGION = "eu-north1"
DEFAULT_PLATFORM = "gpu-h100-sxm"
DEFAULT_PRESET = "1gpu-16vcpu-200gb"
DEFAULT_SECRET = "tf-api-key"
DEFAULT_S3_PROFILE = "nebius-eun1"
IMAGE_NAME = "rtt-sim"


class ConfigError(RuntimeError):
    """A required Nebius setting is missing or malformed."""


# field name -> environment variable
ENV_VARS: dict[str, str] = {
    "profile": "NEBIUS_PROFILE",
    "tenant_id": "NEBIUS_TENANT_ID",
    "project_id": "NEBIUS_PROJECT_ID",
    "subnet_id": "NEBIUS_SUBNET_ID",
    "bucket_name": "NEBIUS_BUCKET_NAME",
    "bucket_id": "NEBIUS_BUCKET_ID",
    "registry_id": "NEBIUS_REGISTRY_ID",
    "secret_name": "NEBIUS_SECRET_NAME",
    "s3_profile": "NEBIUS_S3_PROFILE",
    "s3_endpoint": "NEBIUS_S3_ENDPOINT",
    "region": "NEBIUS_REGION",
    "gpu_platform": "NEBIUS_GPU_PLATFORM",
    "gpu_preset": "NEBIUS_GPU_PRESET",
    "image": "RTT_IMAGE",
    "cli": "NEBIUS_CLI",
}

# fields whose values are account-specific resource IDs and are never displayed
_PRIVATE = frozenset(
    {"tenant_id", "project_id", "subnet_id", "bucket_name", "bucket_id", "registry_id", "image"}
)


def parse_env_file(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines; blank lines and ``#`` comments are skipped."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def default_env_files(environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """The env files read by :func:`load_config`, lowest precedence last."""
    env = os.environ if environ is None else environ
    main = Path(env[ENV_FILE_VAR]) if env.get(ENV_FILE_VAR) else DEFAULT_ENV_FILE
    return (main.expanduser(), DEFAULT_IMAGE_FILE.expanduser())


@dataclass(frozen=True)
class NebiusConfig:
    """Everything the cloud helpers need to address the account's resources."""

    profile: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    subnet_id: str | None = None
    bucket_name: str | None = None
    bucket_id: str | None = None
    registry_id: str | None = None
    secret_name: str = DEFAULT_SECRET
    s3_profile: str = DEFAULT_S3_PROFILE
    s3_endpoint: str | None = None
    region: str = DEFAULT_REGION
    gpu_platform: str = DEFAULT_PLATFORM
    gpu_preset: str = DEFAULT_PRESET
    image: str | None = None
    cli: str = str(DEFAULT_CLI)

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _PRIVATE:
                parts.append(f"{f.name}={'<set>' if value else None}")
            else:
                parts.append(f"{f.name}={value!r}")
        return "NebiusConfig(" + ", ".join(parts) + ")"

    def require(self, *names: str) -> None:
        """Raise :class:`ConfigError` naming the env vars of every unset field in ``names``."""
        missing = [ENV_VARS[n] for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError(
                "missing Nebius settings: "
                + ", ".join(missing)
                + f" (set them in the environment or in {DEFAULT_ENV_FILE})"
            )

    @property
    def registry_host(self) -> str:
        """Container Registry endpoint for :attr:`region`, e.g. ``cr.eu-north1.nebius.cloud``."""
        return f"cr.{self.region}.nebius.cloud"

    @property
    def registry_path(self) -> str:
        """The registry ID without its ``registry-`` prefix (the first path segment of image refs)."""
        self.require("registry_id")
        assert self.registry_id is not None
        return self.registry_id.removeprefix("registry-")

    def image_repository(self, name: str = IMAGE_NAME) -> str:
        """``cr.<region>.nebius.cloud/<registry path>/<name>`` (no tag or digest)."""
        return f"{self.registry_host}/{self.registry_path}/{name}"


def resolve_cli(candidate: str | None) -> str:
    """Path of the ``nebius`` binary: ``candidate``, ``~/.nebius/bin/nebius``, or ``nebius`` on PATH."""
    if candidate:
        return str(Path(candidate).expanduser())
    default = DEFAULT_CLI.expanduser()
    if default.is_file():
        return str(default)
    return shutil.which("nebius") or "nebius"


def load_config(
    environ: Mapping[str, str] | None = None, env_files: Iterable[str | Path] | None = None
) -> NebiusConfig:
    """Build a :class:`NebiusConfig` from env files overlaid with environment variables.

    Args:
        environ: Environment to read (default ``os.environ``).
        env_files: Files to read, highest precedence first (default
            :func:`default_env_files`). Missing files are skipped.
    """
    env = dict(os.environ if environ is None else environ)
    files = default_env_files(env) if env_files is None else tuple(Path(p) for p in env_files)
    merged: dict[str, str] = {}
    for path in reversed(tuple(files)):  # lowest precedence first, so later files win
        p = Path(path).expanduser()
        if p.is_file():
            merged.update(parse_env_file(p.read_text(encoding="utf-8")))
    merged.update({k: v for k, v in env.items() if k in ENV_VARS.values() and v})

    values: dict[str, str] = {}
    for name, var in ENV_VARS.items():
        if merged.get(var):
            values[name] = merged[var]
    values["cli"] = resolve_cli(values.get("cli"))
    return NebiusConfig(**values)
