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

Regions. The unsuffixed variables describe the *home* region (``NEBIUS_REGION``, default
``eu-north1``). Resources in another region use the same variable names with the region
as a suffix, upper-cased without dashes: ``NEBIUS_PROJECT_ID_UKSOUTH2``,
``NEBIUS_SUBNET_ID_UKSOUTH2``, ``NEBIUS_BUCKET_ID_UKSOUTH2``, ``NEBIUS_S3_PROFILE_UKSOUTH2``
and so on. :meth:`NebiusConfig.for_region` returns the settings for one region. A config
for a region other than home never inherits the home region's project, subnet, bucket or
S3 profile, so a job can never be submitted into one region with another region's IDs.
The image and its registry are region-bound too: a job pulls only from a registry in its
own project (uk-south2 jobs failed to pull from the eu-north1 registry with
``StartFailed``), so uk-south2 has its own registry (``NEBIUS_REGISTRY_ID_UKSOUTH2``) and
image reference (``RTT_IMAGE_UKSOUTH2``, written by ``jobs/nebius/push_image.sh`` with
the same digest as ``RTT_IMAGE``). The secret name, CLI profile and tenant carry over.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
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

# fields whose values are account-specific resource IDs and are never displayed. The secret
# selector is one too: another region references the home secret by its MysteryBox ID
# (``NEBIUS_SECRET_NAME_UKSOUTH2=mbsec-…``), so it is hidden even when it is a plain name.
_PRIVATE = frozenset(
    {
        "tenant_id",
        "project_id",
        "subnet_id",
        "bucket_name",
        "bucket_id",
        "registry_id",
        "secret_name",
        "image",
    }
)

REGIONAL_FIELDS: tuple[str, ...] = (
    "project_id",
    "subnet_id",
    "bucket_name",
    "bucket_id",
    "registry_id",
    "secret_name",
    "s3_profile",
    "s3_endpoint",
    "image",
)
"""Fields that can be overridden per region with a ``<VAR>_<SUFFIX>`` variable."""

REGION_BOUND: frozenset[str] = frozenset(
    {
        "project_id",
        "subnet_id",
        "bucket_name",
        "bucket_id",
        "s3_profile",
        "s3_endpoint",
        "registry_id",
        "image",
    }
)
"""Fields naming resources of one region; another region's config never inherits them."""

_SUFFIX = re.compile(r"^[A-Z]+[0-9]+$")


def region_suffix(region: str) -> str:
    """``uk-south2`` -> ``UKSOUTH2``, the suffix of that region's variables."""
    suffix = region.replace("-", "").upper()
    if not _SUFFIX.match(suffix):
        raise ValueError(f"not a Nebius region name: {region!r}")
    return suffix


def _regional_key(key: str) -> tuple[str, str] | None:
    """``NEBIUS_SUBNET_ID_UKSOUTH2`` -> ``("subnet_id", "UKSOUTH2")``, else ``None``.

    Only suffixes shaped like a region (letters then digits) count, so variables such as
    ``RTT_IMAGE_DIGEST`` are not mistaken for overrides.
    """
    for name in REGIONAL_FIELDS:
        base = ENV_VARS[name] + "_"
        if key.startswith(base) and _SUFFIX.match(key[len(base) :]):
            return name, key[len(base) :]
    return None


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
    """Everything the cloud helpers need to address the account's resources.

    A loaded config describes the home region; :meth:`for_region` derives the config of
    any region (including home) with its ``_<SUFFIX>`` overrides applied.
    """

    profile: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    subnet_id: str | None = None
    bucket_name: str | None = None
    bucket_id: str | None = None
    registry_id: str | None = None
    secret_name: str = DEFAULT_SECRET
    s3_profile: str | None = DEFAULT_S3_PROFILE
    s3_endpoint: str | None = None
    region: str = DEFAULT_REGION
    gpu_platform: str = DEFAULT_PLATFORM
    gpu_preset: str = DEFAULT_PRESET
    image: str | None = None
    cli: str = str(DEFAULT_CLI)
    # suffix (``UKSOUTH2``) -> {field: value}, from ``<VAR>_<SUFFIX>`` variables
    region_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict, compare=False, hash=False)
    # set on a config that :meth:`for_region` made for a region other than home
    env_suffix: str | None = None

    def __repr__(self) -> str:
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _PRIVATE:
                parts.append(f"{f.name}={'<set>' if value else None}")
            elif f.name == "region_overrides":
                parts.append(f"{f.name}={sorted(value)!r}")  # suffixes only, never values
            else:
                parts.append(f"{f.name}={value!r}")
        return "NebiusConfig(" + ", ".join(parts) + ")"

    def env_var(self, name: str) -> str:
        """The variable that sets field ``name`` for this config's region.

        Region-bound fields of a non-home region come only from the suffixed variable;
        shared fields (secret name, profile) from the unsuffixed one.
        """
        if self.env_suffix and name in REGION_BOUND:
            return f"{ENV_VARS[name]}_{self.env_suffix}"
        return ENV_VARS[name]

    def require(self, *names: str) -> None:
        """Raise :class:`ConfigError` naming the env vars of every unset field in ``names``."""
        missing = [self.env_var(n) for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError(
                "missing Nebius settings: "
                + ", ".join(missing)
                + f" (set them in the environment or in {DEFAULT_ENV_FILE})"
            )

    @property
    def regions(self) -> tuple[str, ...]:
        """Suffixes that have overrides, e.g. ``("UKSOUTH2",)``."""
        return tuple(sorted(self.region_overrides))

    def for_region(self, region: str) -> NebiusConfig:
        """The settings for jobs and storage in ``region``.

        The home region keeps its unsuffixed values with any ``_<SUFFIX>`` override on top.
        Any other region starts with its :data:`REGION_BOUND` fields unset and takes them
        from its own overrides only; the secret name, CLI profile, tenant and CLI path
        carry over.

        Raises:
            ValueError: called on a config that is already for another region, or
                ``region`` is not a region name.
        """
        if self.env_suffix is not None:
            raise ValueError("for_region() must be called on the home-region config")
        suffix = region_suffix(region)
        over: dict[str, str | None] = dict(self.region_overrides.get(suffix, {}))
        if region == self.region:
            return replace(self, **over)  # type: ignore[arg-type]
        values: dict[str, str | None] = dict.fromkeys(REGION_BOUND)
        values.update(over)
        return replace(self, region=region, env_suffix=suffix, **values)  # type: ignore[arg-type]

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
    """Build the home-region :class:`NebiusConfig` from env files overlaid with environment variables.

    ``<VAR>_<SUFFIX>`` variables of :data:`REGIONAL_FIELDS` are collected into
    :attr:`NebiusConfig.region_overrides` (see :meth:`NebiusConfig.for_region`).

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
    merged.update({k: v for k, v in env.items() if v and (k in ENV_VARS.values() or _regional_key(k))})

    values: dict[str, str] = {}
    for name, var in ENV_VARS.items():
        if merged.get(var):
            values[name] = merged[var]
    values["cli"] = resolve_cli(values.get("cli"))
    overrides: dict[str, dict[str, str]] = {}
    for key, value in merged.items():
        hit = _regional_key(key)
        if hit is not None and value:
            overrides.setdefault(hit[1], {})[hit[0]] = value
    return NebiusConfig(**values, region_overrides=overrides)  # type: ignore[arg-type]
