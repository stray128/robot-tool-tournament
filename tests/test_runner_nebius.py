"""The runner's Nebius backend: the job it submits (pinned image, cpu-d3 32 vCPU, the bucket mount,
no secret), the CLI profile from the service-account secret, and reading results back. The CLI and
the S3 client are fakes; nothing reaches Nebius."""

from __future__ import annotations

import io
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.nebius_jobs import CommandResult, NebiusCLI
from rtt.runner.config import NebiusJobConfig
from rtt.runner.nebius import NebiusBackend, job_name
from rtt.runner.secrets import NebiusCredentials, SecretError, SecretsManagerSource, StaticSecrets

DIGEST = "sha256:" + "ab" * 32
CFG = NebiusJobConfig(
    project_id="project-e00test",
    subnet_id="vpcsubnet-e00test",
    bucket_id="storagebucket-e00test",
    bucket_name="rtt-runs-test",
    image=f"cr.uk-south2.nebius.cloud/e00registry/rtt-sim@{DIGEST}",
    cli="/opt/nebius/nebius",
)
RUN = "abcdefghijklmnopqrstuvwxyz234567"
SA = {
    "subject-credentials": {
        "alg": "RS256",
        "private-key": "PRIVATE-SENTINEL",
        "kid": "k",
        "iss": "i",
        "sub": "s",
    }
}
SECRET_JSON = json.dumps(
    {"service_account_credentials": SA, "s3_access_key_id": "AKID", "s3_secret_access_key": "S3-SENTINEL"}
)


class FakeCLI:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        self.calls.append(list(argv))
        if argv[1:4] == ["ai", "job", "create"]:
            return CommandResult(0, json.dumps({"resource_id": "aijob-e00xyz"}), "")
        if argv[1:4] == ["ai", "job", "get"]:
            return CommandResult(
                0,
                json.dumps(
                    {
                        "metadata": {
                            "id": "aijob-e00xyz",
                            "name": job_name(RUN),
                            "created_at": "2026-10-28T09:00:00Z",
                        },
                        "status": {"state": "PROVISIONING"},
                    }
                ),
                "",
            )
        return CommandResult(0, "{}", "")


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.order: list[str] = []

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body
        self.order.append(f"put {Key}")

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            return {"Body": io.BytesIO(self.objects[f"{Bucket}/{Key}"])}
        except KeyError:
            raise RuntimeError("NoSuchKey") from None

    def generate_presigned_url(self, op: str, Params: dict[str, str], ExpiresIn: int) -> str:
        return f"https://storage.test/{Params['Bucket']}/{Params['Key']}?X-Amz-Expires={ExpiresIn}"


def backend(cli: FakeCLI | None = None, s3: FakeS3 | None = None) -> NebiusBackend:
    return NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", cli or FakeCLI()), s3 or FakeS3())


def test_job_spec_is_the_pinned_cpu_job_with_no_secret() -> None:
    spec = backend().job_spec(RUN, "f" * 64)
    args = spec.create_args()
    assert spec.name == f"elab-{RUN}" and spec.image.endswith(DIGEST)
    assert spec.platform == "cpu-d3" and spec.preset == "32vcpu-128gb"
    assert spec.env_secrets == {} and "--env-secret" not in args
    assert spec.volumes == ["storagebucket-e00test:/out:rw"]
    assert spec.restart_policy == "never" and spec.timeout == "1h"
    assert spec.command == "/app/.venv/bin/python"
    assert list(spec.args) == [
        "-m",
        "rtt.runner.job",
        "--run-dir",
        f"/out/runs/{RUN}",
        "--plan-sha256",
        "f" * 64,
        "--deadline-s",
        "960",
    ]
    assert spec.env["RTT_IMAGE_DIGEST"] == DIGEST and spec.env["RTT_RUN_ID"] == RUN
    assert "--async" in args and args[args.index("--parent-id") + 1] == "project-e00test"
    # account IDs stay out of the repr that a traceback could print
    assert "project-e00test" not in repr(spec) and "e00registry" not in repr(spec)


def test_a_tag_instead_of_a_digest_is_refused() -> None:
    cfg = NebiusJobConfig(**{**CFG.__dict__, "image": "cr.example/reg/rtt-sim:latest"})
    with pytest.raises(ValueError, match="digest"):
        NebiusBackend(cfg, NebiusCLI("nebius", "runner", FakeCLI()), FakeS3()).job_spec(RUN, "0" * 64)


def test_incomplete_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="image"):
        NebiusBackend(NebiusJobConfig(project_id="p"), NebiusCLI("nebius", "runner", FakeCLI()), FakeS3())


def test_submit_uploads_the_plan_then_creates_the_job() -> None:
    cli, s3 = FakeCLI(), FakeS3()
    b = backend(cli, s3)
    plan = b'{"title": "x"}'
    sub = b.submit(RUN, plan)
    assert sub.job_id == "aijob-e00xyz" and sub.job_name == f"elab-{RUN}"
    assert s3.objects[f"rtt-runs-test/runs/{RUN}/input/plan.json"] == plan
    create = next(c for c in cli.calls if c[1:4] == ["ai", "job", "create"])
    assert "--profile" in create and create[create.index("--profile") + 1] == "runner"
    import hashlib

    assert f"--plan-sha256 {hashlib.sha256(plan).hexdigest()}" in create[create.index("--args") + 1]


def test_status_cancel_and_results() -> None:
    cli, s3 = FakeCLI(), FakeS3()
    b = backend(cli, s3)
    st = b.status("aijob-e00xyz")
    assert st.state == "PROVISIONING" and not st.terminal
    b.cancel("aijob-e00xyz")
    assert ["ai", "job", "cancel", "--id", "aijob-e00xyz", "--async"] == cli.calls[-1][1:7]
    assert b.read_result(RUN) == (None, None)
    s3.objects[f"rtt-runs-test/runs/{RUN}/manifest.json"] = b'{"exit_status": 0}'
    s3.objects[f"rtt-runs-test/runs/{RUN}/summary.json"] = b"not json"
    assert b.read_result(RUN) == ({"exit_status": 0}, None)
    s3.objects[f"rtt-runs-test/runs/{RUN}/summary.json"] = b"x" * (600 * 1024)
    assert b.read_result(RUN)[1] is None
    url = b.report_url(RUN, 900)
    assert url == f"https://storage.test/rtt-runs-test/runs/{RUN}/report.json?X-Amz-Expires=900"


def test_from_credentials_writes_a_private_key_file_and_a_profile(tmp_path: Path) -> None:
    cli = FakeCLI()
    creds = NebiusCredentials.parse(SECRET_JSON)
    b = NebiusBackend.from_credentials(CFG, creds, tmp_path / "home", s3_client=FakeS3(), runner=cli)
    sa = tmp_path / "home" / ".nebius" / "runner-sa.json"
    assert json.loads(sa.read_text()) == SA
    assert stat.S_IMODE(os.stat(sa).st_mode) == 0o600
    profile = cli.calls[0]
    assert profile[1:4] == ["profile", "create", "runner"]
    assert "--skip-auth" in profile and profile[profile.index("--service-account-file-path") + 1] == str(sa)
    # nothing secret on any command line
    assert all("PRIVATE-SENTINEL" not in " ".join(c) and "S3-SENTINEL" not in " ".join(c) for c in cli.calls)
    assert b.cli.profile == "runner"


def test_credentials_parse_and_hide() -> None:
    creds = NebiusCredentials.parse(SECRET_JSON)
    assert "SENTINEL" not in repr(creds) and creds.s3_access_key_id == "AKID"
    for bad in (
        "nope",
        "[]",
        json.dumps({"s3_access_key_id": "a"}),
        json.dumps({"service_account_credentials": SA}),
    ):
        with pytest.raises(SecretError) as info:
            NebiusCredentials.parse(bad)
        assert "SENTINEL" not in str(info.value)


class FakeSM:
    def __init__(self) -> None:
        self.calls = 0

    def get_secret_value(self, SecretId: str) -> dict[str, str]:
        self.calls += 1
        if SecretId == "missing":
            raise RuntimeError(f"AccessDenied for {SecretId} with token XYZ")
        return {"SecretString": "  value  "}


def test_secrets_are_read_once_and_errors_carry_no_detail() -> None:
    sm = FakeSM()
    src = SecretsManagerSource(sm)
    assert src.get("a") == "value" and src.get("a") == "value" and sm.calls == 1
    with pytest.raises(SecretError) as info:
        src.get("missing")
    assert "XYZ" not in str(info.value) and "RuntimeError" in str(info.value)
    assert "value" not in repr(src)
    with pytest.raises(SecretError):
        StaticSecrets({}).get("x")


def test_find_is_none_only_when_the_platform_says_not_found() -> None:
    from rtt.cloud.nebius_jobs import NebiusCLIError

    def cli_with(result: CommandResult) -> NebiusBackend:
        return NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", lambda argv, t: result), FakeS3())

    assert cli_with(CommandResult(1, "", "rpc error: code = NotFound desc = job not found")).find(RUN) is None
    for flaky in (
        CommandResult(124, "", "timed out after 60.0 s"),
        CommandResult(1, "", "code = Unavailable"),
    ):
        with pytest.raises(NebiusCLIError):
            cli_with(flaky).find(RUN)
    job = {"metadata": {"id": "aijob-e00xyz", "name": job_name(RUN)}, "status": {"state": "RUNNING"}}
    assert cli_with(CommandResult(0, json.dumps(job), "")).find(RUN).job_id == "aijob-e00xyz"  # type: ignore[union-attr]


def test_a_failure_before_create_is_marked_not_started() -> None:
    from rtt.runner.nebius import SubmitNotStarted

    class BrokenS3(FakeS3):
        def put_object(self, **kw: Any) -> None:
            raise RuntimeError("S3 unavailable")

    cli = FakeCLI()
    with pytest.raises(SubmitNotStarted):
        backend(cli, BrokenS3()).submit(RUN, b"{}")
    assert not any(c[1:4] == ["ai", "job", "create"] for c in cli.calls)

    def create_times_out(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        return CommandResult(124, "", "timed out")

    b = NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", create_times_out), FakeS3())
    with pytest.raises(Exception) as info:
        b.submit(RUN, b"{}")
    assert not isinstance(info.value, SubmitNotStarted)


def test_the_job_spec_carries_the_runners_plan_limits() -> None:
    env = {"RUNNER_MAX_OBJECTS": "4", "RUNNER_MAX_STEPS": "5", "RUNNER_MAX_PLAN_BYTES": "20000"}
    b = NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", FakeCLI()), FakeS3(), env)
    spec = b.job_spec(RUN, "f" * 64)
    assert {k: spec.env[k] for k in env} == env
    spec.validate()  # none of them looks like a credential


def test_the_not_found_answer_is_read_from_the_grpc_status_only() -> None:
    from rtt.cloud.nebius_jobs import NebiusCLIError
    from rtt.runner.nebius import grpc_status

    assert grpc_status("rpc error: code = NotFound desc = job not found") == "notfound"
    assert grpc_status('{"code": "NOT_FOUND", "message": "x"}') == "notfound"
    assert grpc_status("Resource not found: quota allowance") is None
    assert grpc_status("attempt 1: code = Unavailable\nattempt 2: code = NotFound") == "notfound"  # the last
    outage = (
        "warning: cached token not found, re-authenticating\nrpc error: code = Unavailable desc = upstream"
    )

    def cli_with(result: CommandResult) -> NebiusBackend:
        return NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", lambda argv, t: result), FakeS3())

    with pytest.raises(NebiusCLIError):
        cli_with(CommandResult(1, "", outage)).find(RUN)
    with pytest.raises(NebiusCLIError):  # a timeout is never an answer, whatever it printed
        cli_with(CommandResult(124, "", "code = NotFound")).find(RUN)
    assert cli_with(CommandResult(1, "", "rpc error: code = NotFound desc = job not found")).find(RUN) is None


def test_only_a_refusal_of_create_is_a_refusal() -> None:
    from rtt.cloud.nebius_jobs import NebiusCLIError
    from rtt.runner.nebius import SubmitRefused

    def create_fails(result: CommandResult) -> BaseException:
        def cli(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
            return result if argv[1:4] == ["ai", "job", "create"] else CommandResult(0, "{}", "")

        with pytest.raises(Exception) as info:
            NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", cli), FakeS3()).submit(RUN, b"{}")
        return info.value

    for code in ("ResourceExhausted", "InvalidArgument", "PermissionDenied", "FailedPrecondition"):
        err = create_fails(CommandResult(1, "", f"rpc error: code = {code} desc = no"))
        assert isinstance(err, SubmitRefused) and err.code == code.lower()
    for text, rc in (
        ("rpc error: code = Unavailable desc = reset", 1),
        ("rpc error: code = DeadlineExceeded desc = slow", 1),
        ("rpc error: code = AlreadyExists desc = exists", 1),
        ("rpc error: code = ResourceExhausted desc = no", 124),
        ("something broke, quota not found", 1),
    ):
        err = create_fails(CommandResult(rc, "", text))
        assert isinstance(err, NebiusCLIError) and not isinstance(err, SubmitRefused), text


def test_absence_needs_every_grpc_status_to_be_not_found() -> None:
    from rtt.cloud.nebius_jobs import NebiusCLIError
    from rtt.runner.nebius import grpc_statuses

    nested = (
        "rpc error: code = Unavailable desc = gateway: upstream said rpc error: code = NotFound desc = route"
    )
    retried = "retrying after rpc error: code = Unavailable desc = reset\nError: rpc error: code = NotFound desc = gone"
    assert grpc_statuses(nested) == ["unavailable", "notfound"]

    def cli_with(text: str) -> NebiusBackend:
        return NebiusBackend(
            CFG, NebiusCLI(CFG.cli, "runner", lambda argv, t: CommandResult(1, "", text)), FakeS3()
        )

    for text in (nested, retried):
        with pytest.raises(NebiusCLIError):  # unknown, not absent
            cli_with(text).find(RUN)
    twice = "rpc error: code = NotFound desc = a\nrpc error: code = NotFound desc = b"
    assert cli_with(twice).find(RUN) is None


def test_a_refusal_needs_every_grpc_status_to_be_a_refusal() -> None:
    from rtt.cloud.nebius_jobs import NebiusCLIError
    from rtt.runner.nebius import SubmitRefused

    def create_fails(text: str) -> BaseException:
        def cli(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
            return (
                CommandResult(1, "", text)
                if argv[1:4] == ["ai", "job", "create"]
                else CommandResult(0, "{}", "")
            )

        with pytest.raises(Exception) as info:
            NebiusBackend(CFG, NebiusCLI(CFG.cli, "runner", cli), FakeS3()).submit(RUN, b"{}")
        return info.value

    # a first attempt that failed otherwise may have created the job (the retry then finds the
    # quota used up): not a refusal
    retried = "attempt 1: rpc error: code = Unavailable desc = reset\nrpc error: code = ResourceExhausted desc = quota"
    err = create_fails(retried)
    assert isinstance(err, NebiusCLIError) and not isinstance(err, SubmitRefused)
    both = "rpc error: code = InvalidArgument desc = a\nrpc error: code = ResourceExhausted desc = b"
    assert isinstance(create_fails(both), SubmitRefused)
