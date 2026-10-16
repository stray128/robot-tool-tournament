"""Region profiles: per-region config, CPU presets and prices, vCPU quota, and the
uk-south2 CPU default with the eu-north1 GPU fallback (fake ``nebius`` CLI, no network)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.__main__ import main as cloud_main
from rtt.cloud.config import ConfigError, NebiusConfig, load_config, region_suffix
from rtt.cloud.cost import (
    CPU_PRICES,
    estimate_usd,
    hourly_usd,
    preset_resources,
    price_label,
    price_tag,
    read_spend,
)
from rtt.cloud.nebius_jobs import (
    CPU_VCPU_QUOTA,
    DEFAULT_TARGET,
    FALLBACK_TARGET,
    GPU_CANDIDATES,
    TARGETS,
    Candidate,
    CommandResult,
    NebiusCLI,
    NebiusJobs,
    NoCapacityError,
    QuotaAllowance,
    cpu_preset,
    get_target,
    rank_candidates,
    target_for_platform,
)
from rtt.cloud.runs import job_spec_for
from rtt.cloud.storage import BucketStore

IMAGE = "cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:" + "2" * 64
IMAGE_UKS2 = "cr.uk-south2.nebius.cloud/uks2/rtt-sim@sha256:" + "2" * 64
ENV = f"""
NEBIUS_PROFILE=rtt
NEBIUS_TENANT_ID=tenant-test
NEBIUS_PROJECT_ID=project-eun1
NEBIUS_SUBNET_ID=vpcsubnet-eun1
NEBIUS_BUCKET_NAME=bucket-eun1
NEBIUS_BUCKET_ID=storagebucket-eun1
NEBIUS_REGISTRY_ID=registry-abc123
NEBIUS_PROJECT_ID_UKSOUTH2=project-uks2
NEBIUS_SUBNET_ID_UKSOUTH2=vpcsubnet-uks2
NEBIUS_BUCKET_NAME_UKSOUTH2=bucket-uks2
NEBIUS_BUCKET_ID_UKSOUTH2=storagebucket-uks2
NEBIUS_S3_PROFILE_UKSOUTH2=nebius-uks2
NEBIUS_S3_ENDPOINT_UKSOUTH2=https://storage.uk-south2.nebius.cloud
NEBIUS_SECRET_NAME_UKSOUTH2=mbsec-shared
NEBIUS_REGISTRY_ID_UKSOUTH2=registry-uks2
RTT_IMAGE={IMAGE}
RTT_IMAGE_UKSOUTH2={IMAGE_UKS2}
RTT_IMAGE_DIGEST=sha256:{"2" * 64}
"""


def _config(tmp_path: Path, extra: str = "", environ: dict[str, str] | None = None) -> NebiusConfig:
    path = tmp_path / "nebius.env"
    path.write_text(ENV + extra)
    return load_config(environ=environ or {}, env_files=[path])


# --------------------------------------------------------------------------- config


def test_region_suffix() -> None:
    assert region_suffix("uk-south2") == "UKSOUTH2" and region_suffix("eu-north1") == "EUNORTH1"
    with pytest.raises(ValueError):
        region_suffix("not a region")


def test_load_config_collects_region_overrides(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    assert cfg.region == "eu-north1" and cfg.project_id == "project-eun1"
    assert cfg.regions == ("UKSOUTH2",)  # RTT_IMAGE_DIGEST is not mistaken for a region
    assert cfg.region_overrides["UKSOUTH2"]["subnet_id"] == "vpcsubnet-uks2"
    text = repr(cfg)
    assert "region_overrides=['UKSOUTH2']" in text and "project-uks2" not in text


def test_environment_override_wins_for_suffixed_variables(tmp_path: Path) -> None:
    cfg = _config(tmp_path, environ={"NEBIUS_SUBNET_ID_UKSOUTH2": "vpcsubnet-env"})
    assert cfg.for_region("uk-south2").subnet_id == "vpcsubnet-env"


def test_for_region_swaps_region_bound_ids_and_keeps_shared_ones(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    uks2 = cfg.for_region("uk-south2")
    assert uks2.region == "uk-south2" and uks2.env_suffix == "UKSOUTH2"
    assert (uks2.project_id, uks2.subnet_id, uks2.bucket_id, uks2.bucket_name) == (
        "project-uks2",
        "vpcsubnet-uks2",
        "storagebucket-uks2",
        "bucket-uks2",
    )
    assert uks2.s3_profile == "nebius-uks2" and uks2.s3_endpoint == "https://storage.uk-south2.nebius.cloud"
    assert uks2.secret_name == "mbsec-shared"
    # a job pulls only from its own project's registry, so the image is per region too
    assert uks2.image == IMAGE_UKS2 and uks2.registry_id == "registry-uks2"
    assert uks2.registry_host == "cr.uk-south2.nebius.cloud" and uks2.registry_path == "uks2"
    assert cfg.registry_host == "cr.eu-north1.nebius.cloud" and cfg.image == IMAGE
    home = cfg.for_region("eu-north1")
    assert home.project_id == "project-eun1" and home.s3_profile == "nebius-eun1" and home.env_suffix is None
    with pytest.raises(ValueError, match="home-region"):
        uks2.for_region("eu-north1")


def test_a_region_without_overrides_never_inherits_home_ids(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    other = cfg.for_region("us-central1")
    assert other.project_id is None and other.subnet_id is None and other.bucket_id is None
    assert other.s3_profile is None  # never the eu-north1 keys, never the default AWS profile
    assert other.image is None and other.registry_id is None  # the eu-north1 image is unpullable there
    assert other.secret_name == "tf-api-key" and other.profile == "rtt" and other.tenant_id == "tenant-test"
    with pytest.raises(ConfigError, match="NEBIUS_SUBNET_ID_USCENTRAL1, NEBIUS_BUCKET_ID_USCENTRAL1"):
        other.require("subnet_id", "bucket_id")
    with pytest.raises(ConfigError, match="RTT_IMAGE_USCENTRAL1"):
        job_spec_for(other, "m1", "m1-x")
    with pytest.raises(ConfigError, match="NEBIUS_S3_PROFILE_USCENTRAL1"):
        BucketStore.from_config(replace(other, bucket_name="b"))


def test_image_digest_must_match_across_regions(tmp_path: Path, monkeypatch, capsys) -> None:
    other = "cr.uk-south2.nebius.cloud/uks2/rtt-sim@sha256:" + "3" * 64
    env = tmp_path / "nebius.env"
    env.write_text(ENV + f"RTT_IMAGE_UKSOUTH2={other}\n")
    monkeypatch.setenv("RTT_NEBIUS_ENV", str(env))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cloud_main(["run", "m1", "--dry-run"], runner=FakeCloud({})) == 2
    assert "is not the home image" in capsys.readouterr().err


# --------------------------------------------------------------------------- prices


def test_cpu_prices_are_per_vcpu_and_gib() -> None:
    assert preset_resources("16vcpu-64gb") == (16, 64)
    assert preset_resources("1gpu-16vcpu-200gb") == (16, 200)
    assert preset_resources(None) is None
    # 16 x $0.015 + 64 x $0.0045
    assert hourly_usd("cpu-d3", "16vcpu-64gb") == pytest.approx(0.528)
    # the marketing page's "from $0.13" is the 4 vCPU / 16 GiB preset at the same rates
    assert hourly_usd("cpu-d3", "4vcpu-16gb") == pytest.approx(0.132)
    assert hourly_usd("cpu-d3", "32vcpu-128gb") == pytest.approx(2 * 0.528)
    assert estimate_usd("cpu-d3", "16vcpu-64gb", 3600) == pytest.approx(0.528)
    assert CPU_PRICES["cpu-d3"].tag == "sourced" and price_tag("cpu-d3") == "sourced"
    assert price_tag("gpu-l40s-a") == "assumed" and price_tag("nope") == "unpriced"
    assert price_label("cpu-d3", "16vcpu-64gb").startswith("$0.528/h (16 x $0.015/vCPU-h")
    assert price_label("gpu-h100-sxm", "1gpu-16vcpu-200gb").startswith("$4.500/h (1 x $4.50/GPU-h)")


# --------------------------------------------------------------------------- targets


def test_cpu_preset_and_targets() -> None:
    assert cpu_preset(16) == "16vcpu-64gb" and cpu_preset(32) == "32vcpu-128gb"
    with pytest.raises(ValueError, match="no CPU preset"):
        cpu_preset(12)
    assert DEFAULT_TARGET == "uks2-cpu" and FALLBACK_TARGET == "eun1-gpu"
    cpu = TARGETS[DEFAULT_TARGET]
    assert cpu.kind == "cpu" and cpu.region == "uk-south2"
    assert cpu.candidates == (Candidate("cpu-d3", "16vcpu-64gb", "uk-south2"),)
    assert cpu.candidates[0].is_cpu and cpu.candidates[0].vcpus == 16
    assert TARGETS[FALLBACK_TARGET].candidates == GPU_CANDIDATES
    assert get_target("uks2-cpu", 32).candidates[0].preset == "32vcpu-128gb"
    assert get_target("eun1-gpu", 32) == TARGETS["eun1-gpu"]  # vCPUs only size CPU presets
    with pytest.raises(ValueError, match="unknown target"):
        get_target("mars-cpu")
    assert target_for_platform("gpu-h200-sxm").name == "eun1-gpu"
    assert target_for_platform("cpu-d3").name == "uks2-cpu"
    with pytest.raises(ValueError):
        target_for_platform("gpu-made-up")
    only = TARGETS["eun1-gpu"].only("gpu-h200-sxm")
    assert only.candidates == (Candidate("gpu-h200-sxm", "1gpu-16vcpu-200gb", "eu-north1"),)
    with pytest.raises(ValueError, match="pass a preset"):
        TARGETS["uks2-cpu"].only("cpu-e2")


def test_quota_allowance_parsing() -> None:
    data = {"spec": {"limit": "200", "region": "uk-south2"}, "status": {"usage_percentage": "0.00"}}
    quota = QuotaAllowance.from_json(data, CPU_VCPU_QUOTA, "uk-south2")
    assert (quota.limit, quota.usage, quota.free) == (200, 0, 200)
    used = {"spec": {"limit": "200"}, "status": {"usage": "48"}}
    assert QuotaAllowance.from_json(used, CPU_VCPU_QUOTA, "uk-south2").free == 152
    pct = {"spec": {"limit": "200"}, "status": {"usage_percentage": "25.00"}}
    assert QuotaAllowance.from_json(pct, CPU_VCPU_QUOTA, "uk-south2").usage == 50


def test_rank_candidates_uses_quota_for_cpu() -> None:
    cpu16 = Candidate("cpu-d3", "16vcpu-64gb", "uk-south2")
    cpu64 = Candidate("cpu-d3", "64vcpu-256gb", "uk-south2")
    assert rank_candidates([], [cpu16, cpu64], cpu_free_vcpus={"uk-south2": 40}) == [cpu16]
    assert rank_candidates([], [cpu16]) == []  # no quota known: not usable


# --------------------------------------------------------------------------- fake CLI


class FakeCloud:
    """Plays ``nebius`` for both regions: jobs finish on their second poll."""

    def __init__(self, vcpu_limit: dict[str, int], states: dict[str, list[str]] | None = None) -> None:
        self.vcpu_limit = vcpu_limit
        self.states = states or {}
        self.calls: list[list[str]] = []
        self.jobs: dict[str, dict[str, Any]] = {}

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        words = argv[1:4]
        if words == ["quotas", "quota-allowance", "get-by-name"]:
            region = argv[argv.index("--region") + 1]
            limit = self.vcpu_limit.get(region)
            if limit is None:
                return CommandResult(1, "", "Resource not found: quota allowance")
            return CommandResult(
                0, json.dumps({"spec": {"limit": str(limit), "region": region}, "status": {}}), ""
            )
        if argv[1:4] == ["capacity", "resource-advice", "list"]:
            item = {
                "spec": {
                    "region": "eu-north1",
                    "compute_instance": {"platform": "gpu-h100-sxm", "preset": {"name": "1gpu-16vcpu-200gb"}},
                },
                "status": {"on_demand": {"availability_level": "AVAILABILITY_LEVEL_HIGH", "available": 8}},
            }
            return CommandResult(0, json.dumps({"items": [item]}), "")
        if words == ["ai", "job", "create"]:
            if "--dry-run" in argv:
                return CommandResult(0, "Job validation succeeded; no resource was created.\n", "")
            job_id = f"aijob-{len(self.jobs) + 1:04d}"
            platform = argv[argv.index("--platform") + 1]
            self.jobs[job_id] = {
                "name": argv[argv.index("--name") + 1],
                "platform": platform,
                "preset": argv[argv.index("--preset") + 1],
                "parent": argv[argv.index("--parent-id") + 1],
                "states": list(self.states.get(platform, ["RUNNING", "COMPLETED"])),
            }
            return CommandResult(0, json.dumps({"resource_id": job_id}), "")
        if words == ["ai", "job", "get"]:
            job_id = argv[argv.index("--id") + 1]
            job = self.jobs[job_id]
            state = job["states"].pop(0) if len(job["states"]) > 1 else job["states"][0]
            status: dict[str, str] = {}
            if state in ("RUNNING", "COMPLETED"):
                status["started_at"] = "2026-10-15T10:01:00Z"
            if state in ("COMPLETED", "ERROR", "FAILED", "CANCELLED"):
                status["finished_at"] = "2026-10-15T10:02:00Z"
            if state == "ERROR":
                status["state_details"] = {"code": "StartFailed", "message": "Could not pull the image."}  # type: ignore[assignment]
            body = {
                "metadata": {"id": job_id, "name": job["name"], "created_at": "2026-10-15T10:00:00Z"},
                "spec": {"platform": job["platform"], "preset": job["preset"]},
                "status": {"state": state, **status},
            }
            return CommandResult(0, json.dumps(body), "")
        if words == ["ai", "job", "logs"]:
            return CommandResult(0, "RTT_SUMMARY {}\n", "")
        if words == ["ai", "job", "cancel"]:
            return CommandResult(0, "{}", "")
        return CommandResult(1, "", f"unexpected {argv[1:4]}")


def test_run_on_cpu_target_checks_quota_not_advice(tmp_path: Path) -> None:
    cfg = _config(tmp_path).for_region("uk-south2")
    fake = FakeCloud({"uk-south2": 200})
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", fake), cfg, sleep=lambda s: None, clock=lambda: 0.0)
    spec = job_spec_for(cfg, "m1", "m1-t", probe_warp=False, candidate=TARGETS["uks2-cpu"].candidates[0])
    outcome = jobs.run(spec, TARGETS["uks2-cpu"].candidates, poll_s=1)
    assert outcome.job.succeeded and outcome.candidate.platform == "cpu-d3"
    assert not any(c[1:3] == ["capacity", "resource-advice"] for c in fake.calls)
    (attempt,) = outcome.attempts
    assert attempt.region == "uk-south2" and attempt.usd_per_hour == pytest.approx(0.528)
    assert attempt.est_usd == pytest.approx(0.528 * 120 / 3600, abs=1e-4)
    create = next(c for c in fake.calls if c[1:4] == ["ai", "job", "create"])
    assert create[create.index("--parent-id") + 1] == "project-uks2"
    assert create[create.index("--subnet-id") + 1] == "vpcsubnet-uks2"
    assert "storagebucket-uks2:/out:rw" in create and "--env-secret" not in create
    assert "RTT_REGION=uk-south2" in create
    # not enough quota for a 256-vCPU preset
    big = (Candidate("cpu-d3", cpu_preset(256), "uk-south2"),)
    with pytest.raises(NoCapacityError, match="no capacity in uk-south2"):
        jobs.run(spec, big)


def _cli_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    env = tmp_path / "nebius.env"
    env.write_text(ENV)
    monkeypatch.setenv("RTT_NEBIUS_ENV", str(env))
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/rtt/image.env
    return tmp_path


def test_cli_run_defaults_to_uk_south2_cpu(tmp_path: Path, monkeypatch, capsys) -> None:
    home = _cli_home(tmp_path, monkeypatch)
    fake = FakeCloud({"uk-south2": 200})
    argv = ["run", "m1", "--run-id", "m1-cli", "--no-fetch", "--poll-s", "1"]
    argv += ["--fetch-dir", str(home / "runs"), "--ledger", str(home / "spend.jsonl")]
    assert cloud_main(argv, runner=fake) == 0
    record = json.loads((home / "runs" / "m1-cli" / "job.json").read_text())
    assert record["target"] == "uks2-cpu" and record["region"] == "uk-south2"
    assert record["platform"] == "cpu-d3" and record["fallback_from"] is None
    assert record["price"].startswith("$0.528/h")
    (spend,) = read_spend(home / "spend.jsonl")
    assert spend.platform == "cpu-d3" and spend.region == "uk-south2"
    create = next(c for c in fake.calls if c[1:4] == ["ai", "job", "create"])
    assert "--probe-warp" not in " ".join(create)  # no Warp probe without a GPU


def test_cli_run_falls_back_to_h100_without_cpu_quota(tmp_path: Path, monkeypatch, capsys) -> None:
    home = _cli_home(tmp_path, monkeypatch)
    fake = FakeCloud({"uk-south2": 0})
    argv = ["run", "m1", "--run-id", "m1-fb", "--no-fetch", "--poll-s", "1"]
    argv += ["--fetch-dir", str(home / "runs"), "--ledger", str(home / "spend.jsonl")]
    assert cloud_main(argv, runner=fake) == 0
    assert "falling back to eun1-gpu" in capsys.readouterr().out
    record = json.loads((home / "runs" / "m1-fb" / "job.json").read_text())
    assert record["target"] == "eun1-gpu" and record["region"] == "eu-north1"
    assert record["fallback_from"] == "uks2-cpu" and record["platform"] == "gpu-h100-sxm"
    create = next(c for c in fake.calls if c[1:4] == ["ai", "job", "create"])
    assert create[create.index("--parent-id") + 1] == "project-eun1"
    # ...and not at all with --no-fallback
    assert (
        cloud_main([*argv[:3], "m1-nofb", *argv[4:], "--no-fallback"], runner=FakeCloud({"uk-south2": 0}))
        == 2
    )
    assert "no capacity in uk-south2" in capsys.readouterr().err


def test_cli_explicit_vcpus_never_falls_back(tmp_path: Path, monkeypatch, capsys) -> None:
    """A size the quota cannot fit is refused, not swapped for a much dearer H100."""
    home = _cli_home(tmp_path, monkeypatch)
    argv = ["run", "m1", "--run-id", "m1-big", "--no-fetch", "--poll-s", "1", "--vcpus", "32"]
    argv += ["--fetch-dir", str(home / "runs"), "--ledger", str(home / "spend.jsonl")]
    fake = FakeCloud({"uk-south2": 0})
    assert cloud_main(argv, runner=fake) == 2
    assert not any(c[1:4] == ["ai", "job", "create"] for c in fake.calls)
    assert "falling back" not in capsys.readouterr().out


def test_cli_platform_flag_picks_its_target(tmp_path: Path, monkeypatch, capsys) -> None:
    _cli_home(tmp_path, monkeypatch)
    fake = FakeCloud({})
    assert cloud_main(["run", "m1", "--dry-run", "--platform", "gpu-h100-sxm"], runner=fake) == 0
    out = capsys.readouterr().out
    assert out.startswith("eun1-gpu (eu-north1)") and "uks2-cpu" not in out  # no fallback chain
    create = next(c for c in fake.calls if c[1:4] == ["ai", "job", "create"])
    assert create[create.index("--platform") + 1] == "gpu-h100-sxm"
    assert "--probe-warp" in create[create.index("--args") + 1]
    # a dry run validates the default target and its fallback
    fake = FakeCloud({})
    assert (
        cloud_main(
            ["run", "search", "--arm", "nemotron", "--runs", "1", "--budget", "4", "--dry-run"], runner=fake
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("uks2-cpu (uk-south2)") and "eun1-gpu (eu-north1)" in out
    creates = [c for c in fake.calls if c[1:4] == ["ai", "job", "create"]]
    assert creates[0][creates[0].index("--env-secret") + 1] == "NEBIUS_API_KEY=mbsec-shared"
    assert creates[1][creates[1].index("--env-secret") + 1] == "NEBIUS_API_KEY=tf-api-key"
    assert creates[0][creates[0].index("--preset") + 1] == "16vcpu-64gb"


def test_cli_capacity_lists_both_targets(tmp_path: Path, monkeypatch, capsys) -> None:
    _cli_home(tmp_path, monkeypatch)
    assert cloud_main(["capacity", "--vcpus", "32"], runner=FakeCloud({"uk-south2": 200})) == 0
    out = capsys.readouterr().out
    assert "uks2-cpu (default)" in out and "eun1-gpu (fallback)" in out
    assert "usable  cpu-d3         32vcpu-128gb" in out and "vCPU quota 0/200 used" in out
    assert "usable  gpu-h100-sxm" in out and "skip    gpu-h200-sxm" in out


def test_cli_run_falls_back_when_the_cpu_job_never_starts(tmp_path: Path, monkeypatch, capsys) -> None:
    home = _cli_home(tmp_path, monkeypatch)
    fake = FakeCloud({"uk-south2": 200}, states={"cpu-d3": ["IMAGE_PULLING", "ERROR"]})
    argv = ["run", "m1", "--run-id", "m1-sf", "--no-fetch", "--poll-s", "1"]
    argv += ["--fetch-dir", str(home / "runs"), "--ledger", str(home / "spend.jsonl")]
    assert cloud_main(argv, runner=fake) == 0
    assert "ERROR before start" in capsys.readouterr().out
    record = json.loads((home / "runs" / "m1-sf" / "job.json").read_text())
    assert record["target"] == "eun1-gpu" and record["fallback_from"] == "uks2-cpu"
    # both attempts are on the record and in the ledger, so the site's cost includes the failed one
    assert [a["platform"] for a in record["attempts"]] == ["cpu-d3", "gpu-h100-sxm"]
    assert [e.state for e in read_spend(home / "spend.jsonl")] == ["ERROR", "COMPLETED"]


def test_cli_run_falls_back_when_uk_south2_is_not_configured(tmp_path: Path, monkeypatch, capsys) -> None:
    env = tmp_path / "nebius.env"
    env.write_text("\n".join(line for line in ENV.splitlines() if "UKSOUTH2" not in line))
    monkeypatch.setenv("RTT_NEBIUS_ENV", str(env))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cloud_main(["run", "m1", "--dry-run"], runner=FakeCloud({})) == 0
    out = capsys.readouterr().out
    assert "uks2-cpu: not configured" in out and "NEBIUS_PROJECT_ID_UKSOUTH2" in out
    assert "eun1-gpu (eu-north1)" in out
