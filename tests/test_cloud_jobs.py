"""rtt.cloud.nebius_jobs against a scripted fake of the ``nebius`` CLI (no network)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.config import NebiusConfig
from rtt.cloud.cost import SpendCapError, SpendLedger, read_spend
from rtt.cloud.nebius_jobs import (
    DEFAULT_CANDIDATES,
    Candidate,
    CommandResult,
    Job,
    JobSpec,
    NebiusCLI,
    NebiusCLIError,
    NebiusJobs,
    ProvisioningTimeoutError,
    _extract_job_id,
    parse_capacity,
    rank_candidates,
)

DIGEST_IMAGE = "cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:" + "0" * 64
CONFIG = NebiusConfig(
    profile="rtt",
    tenant_id="tenant-test",
    project_id="project-test",
    subnet_id="vpcsubnet-test",
    bucket_id="storagebucket-test",
    bucket_name="bucket",
    cli="nebius",
)


def _job_json(job_id: str, name: str, state: str, platform: str = "gpu-h100-sxm", **status: str) -> dict:
    return {
        "metadata": {"id": job_id, "name": name, "created_at": "2026-10-15T09:16:47.829267Z"},
        "spec": {"image": DIGEST_IMAGE, "platform": platform, "preset": "1gpu-16vcpu-200gb"},
        "status": {"state": state, **status},
    }


def _advice(platform: str, preset: str, level: str, available: int | None, region: str = "eu-north1") -> dict:
    on_demand: dict[str, Any] = {"availability_level": f"AVAILABILITY_LEVEL_{level}", "limit": 8}
    if available is not None:
        on_demand["available"] = available
    return {
        "spec": {
            "region": region,
            "fabric": "fabric-2",
            "compute_instance": {"platform": platform, "preset": {"name": preset}},
        },
        "status": {"on_demand": on_demand, "preemptible": {"availability_level": "AVAILABILITY_LEVEL_LOW"}},
    }


ADVICE = {
    "items": [
        _advice("gpu-h100-sxm", "1gpu-16vcpu-200gb", "LOW", None),
        _advice("gpu-h200-sxm", "1gpu-16vcpu-200gb", "HIGH", 8),
        _advice("gpu-l40s-a", "1gpu-8vcpu-32gb", "LOW", 1),
        _advice("gpu-h100-sxm", "1gpu-16vcpu-200gb", "HIGH", 8, region="us-central1"),
        {"spec": {"region": "eu-north1"}, "status": {}},  # non-compute row
    ]
}


class FakeNebius:
    """Plays the CLI: jobs advance through scripted state lists, one step per ``get``."""

    def __init__(self, scripts: dict[str, list[str]], advice: dict | None = None) -> None:
        self.scripts = scripts  # platform -> states returned by successive gets
        self.advice = advice or {"items": []}
        self.calls: list[list[str]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.cancelled: list[str] = []

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        words = argv[1:4]
        if words == ["ai", "job", "create"]:
            name = argv[argv.index("--name") + 1]
            platform = argv[argv.index("--platform") + 1]
            job_id = f"aijob-{len(self.jobs) + 1:04d}"
            self.jobs[job_id] = {"name": name, "platform": platform, "states": list(self.scripts[platform])}
            if "--dry-run" in argv:
                return CommandResult(0, "Job validation succeeded; no resource was created.\n", "")
            return CommandResult(0, json.dumps({"id": "op-1", "resource_id": job_id, "done": False}), "")
        if words == ["ai", "job", "get"]:
            job_id = argv[argv.index("--id") + 1]
            job = self.jobs[job_id]
            state = job["states"].pop(0) if len(job["states"]) > 1 else job["states"][0]
            if job_id in self.cancelled and state not in ("COMPLETED", "FAILED"):
                state = "CANCELLED"
            status = {}
            if state in ("RUNNING", "COMPLETED", "FAILED"):
                status["started_at"] = "2026-10-15T09:18:39.585569020Z"
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                status["finished_at"] = "2026-10-15T09:20:39.585569020Z"
            return CommandResult(
                0, json.dumps(_job_json(job_id, job["name"], state, job["platform"], **status)), ""
            )
        if words == ["ai", "job", "cancel"]:
            self.cancelled.append(argv[argv.index("--id") + 1])
            return CommandResult(0, "{}", "")
        if words == ["ai", "job", "list"]:
            items = [_job_json(i, j["name"], j["states"][0]) for i, j in self.jobs.items()]
            return CommandResult(0, json.dumps({"items": items}), "")
        if words == ["ai", "job", "logs"]:
            return CommandResult(0, "line 1\nRTT_SUMMARY {}\n", "")
        if argv[1:4] == ["capacity", "resource-advice", "list"]:
            return CommandResult(0, json.dumps(self.advice), "")
        return CommandResult(1, "", f"unknown command {argv[1:4]}")


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _service(fake: FakeNebius) -> tuple[NebiusJobs, FakeClock]:
    clock = FakeClock()
    return NebiusJobs(NebiusCLI("nebius", "rtt", fake), CONFIG, sleep=clock.sleep, clock=clock), clock


def _spec(**overrides: Any) -> JobSpec:
    base: dict[str, Any] = {
        "name": "rtt-m1-test",
        "image": DIGEST_IMAGE,
        "command": "/app/.venv/bin/python",
        "args": ["-m", "rtt.cloud.entry", "m1", "--out", "/out/runs/m1 x"],
        "env": {"RTT_RUN_ID": "m1-test"},
        "env_secrets": {"NEBIUS_API_KEY": "tf-api-key"},
        "volumes": ["storagebucket-test:/out:rw"],
        "subnet_id": "vpcsubnet-test",
        "parent_id": "project-test",
    }
    base.update(overrides)
    return JobSpec(**base)


# --------------------------------------------------------------------------- JobSpec


def test_create_args_render_every_flag() -> None:
    argv = _spec(disk_size="100Gi", preemptible=True).create_args()
    pairs = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--")}
    assert argv[:3] == ["ai", "job", "create"]
    assert pairs["--image"] == DIGEST_IMAGE
    assert pairs["--container-command"] == "/app/.venv/bin/python"
    assert pairs["--args"] == "-m rtt.cloud.entry m1 --out '/out/runs/m1 x'"  # shell-quoted
    assert pairs["--env-secret"] == "NEBIUS_API_KEY=tf-api-key"
    assert pairs["--env"] == "RTT_RUN_ID=m1-test"
    assert pairs["--volume"] == "storagebucket-test:/out:rw"
    assert pairs["--platform"] == "gpu-h100-sxm" and pairs["--preset"] == "1gpu-16vcpu-200gb"
    assert pairs["--timeout"] == "1h" and pairs["--restart-policy"] == "never"
    assert pairs["--subnet-id"] == "vpcsubnet-test" and pairs["--parent-id"] == "project-test"
    assert "--preemptible" in argv and "--follows-spot-price" in argv
    assert argv[-1] == "--async"
    assert _spec().create_args(dry_run=True)[-1] == "--dry-run"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image": "nvidia/cuda:13.1.1-base-ubuntu24.04"}, "pinned by digest"),
        ({"env": {"NEBIUS_API_KEY": "sk-123"}}, "looks like a credential"),
        ({"env": {"AWS_SECRET_ACCESS_KEY": "x"}}, "looks like a credential"),
        ({"timeout": "30m"}, "between 1h and 168h"),
        ({"timeout": "200h"}, "between 1h and 168h"),
        ({"restart_policy": "always"}, "restart_policy"),
        ({"name": "Bad_Name"}, "job name"),
        ({"volumes": ["storagebucket-test"]}, "SOURCE:/container/path"),
    ],
)
def test_spec_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(**overrides).create_args()


def test_tagged_image_allowed_only_when_asked() -> None:
    _spec(image="nvidia/cuda:13.1.1-base-ubuntu24.04", allow_tag=True).validate()


# --------------------------------------------------------------------------- CLI plumbing


def test_cli_adds_format_profile_and_reports_errors_without_values() -> None:
    seen: list[list[str]] = []

    def runner(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        seen.append(list(argv))
        return CommandResult(3, "", "permission denied")

    cli = NebiusCLI("/bin/nebius", "rtt", runner)
    with pytest.raises(NebiusCLIError) as err:
        cli.run_json(["ai", "job", "get", "--id", "aijob-secret"])
    assert seen[0][:1] == ["/bin/nebius"]
    assert seen[0][-5:] == ["--format", "json", "--profile", "rtt", "--no-check-update"]
    assert err.value.command == "ai job get" and err.value.returncode == 3
    assert "aijob-secret" not in str(err.value).split(":")[0]


def test_unparseable_json_is_an_error() -> None:
    cli = NebiusCLI("nebius", None, lambda argv, t: CommandResult(0, "not json", ""))
    with pytest.raises(NebiusCLIError, match="unparseable JSON"):
        cli.run_json(["ai", "job", "list"])


def test_extract_job_id_from_operation_shapes() -> None:
    assert _extract_job_id({"id": "op-1", "resource_id": "aijob-1"}) == "aijob-1"
    assert _extract_job_id({"metadata": {"id": "aijob-2"}}) == "aijob-2"
    assert _extract_job_id({"operation": [{"resource": {"id": "aijob-3"}}]}) == "aijob-3"
    assert _extract_job_id({"id": "op-1"}) is None


# --------------------------------------------------------------------------- Job model


def test_job_from_json_timings_and_nanosecond_timestamps() -> None:
    job = Job.from_json(
        _job_json(
            "aijob-1",
            "rtt-smoke",
            "completed",
            started_at="2026-10-15T09:18:39.585569020Z",
            finished_at="2026-10-15T09:18:39.988273721Z",
        )
    )
    assert job.state == "COMPLETED" and job.terminal and job.succeeded
    assert job.provisioning_s == pytest.approx(111.756, abs=1e-3)
    assert job.run_s == pytest.approx(0.403, abs=1e-3)
    assert job.wall_s() == pytest.approx(112.159, abs=1e-3)
    running = Job.from_json(_job_json("aijob-2", "x", "RUNNING"))
    assert not running.terminal and running.run_s is None
    assert running.wall_s(datetime(2026, 10, 15, 9, 17, 47, 829267, tzinfo=UTC)) == pytest.approx(60.0)


# --------------------------------------------------------------------------- capacity


def test_parse_and_rank_capacity() -> None:
    rows = parse_capacity(ADVICE)
    assert len(rows) == 4  # the non-compute row is dropped
    h200 = next(r for r in rows if r.platform == "gpu-h200-sxm")
    assert h200.on_demand.level == "HIGH" and h200.on_demand.available == 8
    # H100 in eu-north1 is LOW with nothing free; us-central1 does not count; L40S LOW is below MEDIUM
    assert rank_candidates(rows) == [Candidate("gpu-h200-sxm", "1gpu-16vcpu-200gb")]
    assert rank_candidates(rows, min_level="LOW") == [
        Candidate("gpu-h200-sxm", "1gpu-16vcpu-200gb"),
        Candidate("gpu-l40s-a", "1gpu-8vcpu-32gb"),
    ]
    assert rank_candidates(rows, preemptible=True) == []


def test_capacity_needs_tenant() -> None:
    fake = FakeNebius({}, ADVICE)
    jobs, _ = _service(fake)
    assert len(jobs.capacity()) == 4
    assert fake.calls[0][1:7] == [
        "capacity",
        "resource-advice",
        "list",
        "--parent-id",
        "tenant-test",
        "--all",
    ]


# --------------------------------------------------------------------------- service


def test_submit_get_logs_cancel_list() -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING", "RUNNING", "COMPLETED"]})
    jobs, _ = _service(fake)
    job = jobs.submit(_spec())
    assert job.id == "aijob-0001" and job.state == "PROVISIONING"
    assert jobs.status(job.id) == "RUNNING"
    assert "RTT_SUMMARY" in jobs.logs(job.id, tail=50)
    assert fake.calls[-1][1:7] == ["ai", "job", "logs", "aijob-0001", "--since", "24h"]
    jobs.cancel(job.id)
    assert fake.calls[-1][4:7] == ["--id", "aijob-0001", "--async"]
    assert [j.id for j in jobs.list_jobs()] == ["aijob-0001"]
    assert jobs.dry_run(_spec()).startswith("Job validation succeeded")


def test_wait_reports_transitions_and_returns_terminal_job() -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING", "PROVISIONING", "STARTING", "RUNNING", "COMPLETED"]})
    jobs, clock = _service(fake)
    job = jobs.submit(_spec())
    seen: list[str] = []
    final = jobs.wait(job.id, poll_s=10, on_update=lambda u: seen.append(u.job.state))
    assert final.state == "COMPLETED"
    assert seen == ["PROVISIONING", "STARTING", "RUNNING", "COMPLETED"]
    assert clock.t == 30.0  # submit saw the first PROVISIONING; three polls sleep


def test_wait_provisioning_timeout() -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING"]})
    jobs, _ = _service(fake)
    job = jobs.submit(_spec())
    with pytest.raises(ProvisioningTimeoutError) as err:
        jobs.wait(job.id, poll_s=60, provisioning_timeout_s=300)
    assert err.value.waited_s == 300 and err.value.job.state == "PROVISIONING"
    assert fake.cancelled == []  # the caller decides


def test_wait_cancels_when_budget_runs_out() -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["RUNNING"]})
    jobs, _ = _service(fake)
    job = jobs.submit(_spec())
    with pytest.raises(SpendCapError, match="cancelled"):
        jobs.wait(job.id, poll_s=60, max_wall_s=120)
    assert fake.cancelled == [job.id]


def test_run_falls_back_after_provisioning_timeout(tmp_path: Path) -> None:
    fake = FakeNebius(
        {
            "gpu-h100-sxm": ["PROVISIONING"],
            "gpu-h200-sxm": ["PROVISIONING", "RUNNING", "COMPLETED"],
        }
    )
    jobs, _ = _service(fake)
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=10.0)
    outcome = jobs.run(
        _spec(),
        DEFAULT_CANDIDATES[:2],
        ledger=ledger,
        provisioning_timeout_s=900,
        poll_s=300,
        check_capacity=False,
        run_id="m1-test",
    )
    assert outcome.job.state == "COMPLETED" and outcome.candidate.platform == "gpu-h200-sxm"
    assert [a.state for a in outcome.attempts] == ["CANCELLED", "COMPLETED"]
    assert outcome.attempts[0].note and outcome.attempts[0].note.startswith("cancelled")
    # the retry got its own name, platform and RTT_* env
    creates = [c for c in fake.calls if c[1:4] == ["ai", "job", "create"]]
    assert creates[1][creates[1].index("--name") + 1] == "rtt-m1-test-r1"
    assert "RTT_PLATFORM=gpu-h200-sxm" in creates[1]
    assert [e.platform for e in read_spend(ledger.path)] == ["gpu-h100-sxm", "gpu-h200-sxm"]
    assert outcome.est_usd == pytest.approx(sum(e.est_usd for e in read_spend(ledger.path)))


def test_run_uses_capacity_advice_and_refuses_when_none(tmp_path: Path) -> None:
    fake = FakeNebius({"gpu-h200-sxm": ["COMPLETED"]}, ADVICE)
    jobs, _ = _service(fake)
    outcome = jobs.run(_spec(), poll_s=1)
    assert outcome.candidate.platform == "gpu-h200-sxm"  # H100 had no capacity in the advice
    empty = FakeNebius({}, {"items": []})
    jobs, _ = _service(empty)
    with pytest.raises(RuntimeError, match="no capacity"):
        jobs.run(_spec())


def test_run_refuses_submission_over_the_cap(tmp_path: Path) -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["COMPLETED"]})
    jobs, _ = _service(fake)
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=0.5)
    with pytest.raises(SpendCapError):
        jobs.run(_spec(), DEFAULT_CANDIDATES[:1], ledger=ledger, check_capacity=False)
    assert not any(c[1:4] == ["ai", "job", "create"] for c in fake.calls)


@pytest.mark.parametrize("stuck", ["STARTING", "IMAGE_PULLING"])
def test_wait_times_out_in_every_pre_start_state(stuck: str) -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING", stuck]})
    jobs, _ = _service(fake)
    job = jobs.submit(_spec())
    with pytest.raises(ProvisioningTimeoutError) as err:
        jobs.wait(job.id, poll_s=60, provisioning_timeout_s=300)
    assert err.value.job.state == stuck


class FlakyGets:
    """Wraps a fake CLI so that the given (1-based) ``ai job get`` calls fail."""

    def __init__(self, fake: FakeNebius, failing: set[int]) -> None:
        self.fake = fake
        self.failing = failing
        self.gets = 0

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        if list(argv[1:4]) == ["ai", "job", "get"]:
            self.gets += 1
            if self.gets in self.failing:
                return CommandResult(1, "", "transport: connection reset")
        return self.fake(argv, timeout_s)


def test_wait_rides_out_transient_poll_errors() -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING", "RUNNING", "COMPLETED"]})
    flaky = FlakyGets(fake, {2, 3})  # get #1 is submit's; two polls then fail
    clock = FakeClock()
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", flaky), CONFIG, sleep=clock.sleep, clock=clock)
    job = jobs.submit(_spec())
    assert jobs.wait(job.id, poll_s=10).state == "COMPLETED"
    # ...but not forever
    fake = FakeNebius({"gpu-h100-sxm": ["RUNNING"]})
    flaky = FlakyGets(fake, set(range(2, 100)))
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", flaky), CONFIG, sleep=clock.sleep, clock=clock)
    job = jobs.submit(_spec())
    with pytest.raises(NebiusCLIError):
        jobs.wait(job.id, poll_s=10, max_poll_errors=3)
    assert flaky.gets == 4


def test_run_records_spend_of_a_job_it_stops_watching(tmp_path: Path) -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["RUNNING"]})
    flaky = FlakyGets(fake, set(range(2, 7)))  # five polls in a row fail, then gets work again
    clock = FakeClock()
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", flaky), CONFIG, sleep=clock.sleep, clock=clock)
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=10.0)
    with pytest.raises(NebiusCLIError):
        jobs.run(_spec(), DEFAULT_CANDIDATES[:1], ledger=ledger, check_capacity=False, run_id="m1-test")
    (entry,) = read_spend(ledger.path)
    assert entry.state == "RUNNING" and entry.run_id == "m1-test"
    assert entry.note and entry.note.startswith("abandoned while RUNNING (NebiusCLIError)")
    assert fake.cancelled == []  # left running for the caller to decide


def test_retry_attempt_renames_the_job_in_its_env(tmp_path: Path) -> None:
    fake = FakeNebius({"gpu-h100-sxm": ["PROVISIONING"], "gpu-h200-sxm": ["RUNNING", "COMPLETED"]})
    jobs, _ = _service(fake)
    spec = _spec(env={"RTT_RUN_ID": "m1-test", "RTT_JOB_NAME": "rtt-m1-test"})
    jobs.run(spec, DEFAULT_CANDIDATES[:2], provisioning_timeout_s=60, poll_s=60, check_capacity=False)
    creates = [c for c in fake.calls if c[1:4] == ["ai", "job", "create"]]
    assert "RTT_JOB_NAME=rtt-m1-test" in creates[0]
    assert "RTT_JOB_NAME=rtt-m1-test-r1" in creates[1]
