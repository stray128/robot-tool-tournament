"""rtt.cloud.config: env-file parsing, precedence, redaction and registry naming."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtt.cloud.config import ConfigError, NebiusConfig, load_config, parse_env_file

ENV_TEXT = """
# Nebius resources (fake values for tests)
NEBIUS_PROFILE=rtt
export NEBIUS_PROJECT_ID=project-test0001
NEBIUS_SUBNET_ID="vpcsubnet-test0001"
NEBIUS_BUCKET_ID='storagebucket-test0001'
NEBIUS_BUCKET_NAME=results-bucket  # trailing comment
NEBIUS_REGISTRY_ID=registry-abc123
not a line
=novalue
"""


def test_parse_env_file_handles_export_quotes_and_comments() -> None:
    env = parse_env_file(ENV_TEXT)
    assert env == {
        "NEBIUS_PROFILE": "rtt",
        "NEBIUS_PROJECT_ID": "project-test0001",
        "NEBIUS_SUBNET_ID": "vpcsubnet-test0001",
        "NEBIUS_BUCKET_ID": "storagebucket-test0001",
        "NEBIUS_BUCKET_NAME": "results-bucket",
        "NEBIUS_REGISTRY_ID": "registry-abc123",
    }


def test_load_config_precedence_environment_then_files(tmp_path: Path) -> None:
    main = tmp_path / "nebius.env"
    image = tmp_path / "image.env"
    main.write_text(ENV_TEXT)
    image.write_text("RTT_IMAGE=cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:00\nNEBIUS_PROFILE=other\n")
    cfg = load_config(
        environ={"NEBIUS_SUBNET_ID": "vpcsubnet-from-env", "UNRELATED": "x"}, env_files=[main, image]
    )
    assert cfg.subnet_id == "vpcsubnet-from-env"  # environment wins
    assert cfg.profile == "rtt"  # the first file wins over the second
    assert cfg.image == "cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:00"
    assert cfg.secret_name == "tf-api-key" and cfg.region == "eu-north1"  # defaults


def test_missing_files_are_skipped(tmp_path: Path) -> None:
    cfg = load_config(environ={}, env_files=[tmp_path / "absent.env"])
    assert cfg.project_id is None and cfg.gpu_platform == "gpu-h100-sxm"


def test_repr_hides_resource_ids() -> None:
    cfg = NebiusConfig(project_id="project-secretish", bucket_id="storagebucket-xyz", profile="rtt")
    text = repr(cfg)
    assert "project-secretish" not in text and "storagebucket-xyz" not in text
    assert "project_id=<set>" in text and "subnet_id=None" in text and "profile='rtt'" in text


def test_require_names_env_vars_not_values() -> None:
    cfg = NebiusConfig(project_id="project-1")
    cfg.require("project_id")
    with pytest.raises(ConfigError, match="NEBIUS_SUBNET_ID, NEBIUS_BUCKET_ID") as err:
        cfg.require("project_id", "subnet_id", "bucket_id")
    assert "project-1" not in str(err.value)


def test_registry_path_and_image_repository() -> None:
    cfg = NebiusConfig(registry_id="registry-e00abc", region="eu-north1")
    assert cfg.registry_host == "cr.eu-north1.nebius.cloud"
    assert cfg.registry_path == "e00abc"
    assert cfg.image_repository() == "cr.eu-north1.nebius.cloud/e00abc/rtt-sim"
    with pytest.raises(ConfigError, match="NEBIUS_REGISTRY_ID"):
        _ = NebiusConfig().registry_path


def test_cli_path_override(tmp_path: Path) -> None:
    cfg = load_config(environ={"NEBIUS_CLI": "/opt/nebius/bin/nebius"}, env_files=[])
    assert cfg.cli == "/opt/nebius/bin/nebius"
