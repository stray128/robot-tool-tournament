"""Agent job runners: local threads, Nebius through a scripted CLI, the part job spec, the remote entry."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import trimesh

from rtt.agent import remote
from rtt.agent.fakes import FakePartBackend
from rtt.agent.jobs import LocalJobRunner, NebiusJobRunner, UnknownJobError, part_job_factory
from rtt.cloud.config import NebiusConfig
from rtt.cloud.nebius_jobs import CommandResult, NebiusCLI, NebiusJobs
from rtt.design.schema import FingerDesign, load_entrants
from rtt.search.base import DesignScore

REPO = Path(__file__).resolve().parents[1]
ENTRANTS = load_entrants(REPO / "configs" / "entrants.yaml")
PART = "part-0123456789ab"
IMAGE = "cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:" + "0" * 64
CONFIG = NebiusConfig(
    profile="rtt",
    project_id="project-test",
    subnet_id="vpcsubnet-test",
    bucket_id="storagebucket-test",
    image=IMAGE,
    cli="nebius",
)


@pytest.fixture
def backend(tmp_path: Path) -> FakePartBackend:
    fake = FakePartBackend(tmp_path / "parts")
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=(40.0, 30.0, 60.0)).export(path)
    fake.intake(path, part_id=PART, name="box")
    return fake


# --------------------------------------------------------------------------- local


def test_local_runner_runs_and_returns_scores(backend: FakePartBackend) -> None:
    runner = LocalJobRunner(backend.evaluate)
    record = runner.launch(PART, ENTRANTS[:2])
    assert record.job_id.startswith("local-") and record.target == "local"
    done = runner.wait(record.job_id, 10.0)
    assert done.state == "COMPLETED" and done.finished_at is not None
    scores = runner.results(record.job_id)
    assert [s.design_id for s in scores] == [d.design_id() for d in ENTRANTS[:2]]
    assert runner.estimate_usd(5) == 0.0
    with pytest.raises(UnknownJobError):
        runner.status("local-000000000000")
    runner.shutdown()


def test_local_runner_reports_failures() -> None:
    def broken(part_id: str, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        raise RuntimeError("simulator exploded")

    runner = LocalJobRunner(broken)
    record = runner.wait(runner.launch(PART, ENTRANTS[:1]).job_id, 10.0)
    assert record.state == "FAILED" and "simulator exploded" in (record.error or "")
    with pytest.raises(RuntimeError, match="FAILED"):
        runner.results(record.job_id)


def test_local_runner_cancels_queued_jobs(backend: FakePartBackend) -> None:
    backend.gate.clear()
    runner = LocalJobRunner(backend.evaluate, max_workers=1)
    first = runner.launch(PART, ENTRANTS[:1])
    second = runner.launch(PART, ENTRANTS[1:2])
    assert runner.cancel(second.job_id).state == "CANCELLED"
    backend.gate.set()
    assert runner.wait(first.job_id, 10.0).state == "COMPLETED"


# --------------------------------------------------------------------------- nebius


class ScriptedCLI:
    """Plays ``nebius ai job create/get/cancel``; the job advances one state per ``get``."""

    def __init__(self, states: Sequence[str]) -> None:
        self.states = list(states)
        self.calls: list[list[str]] = []
        self.gets = 0
        self.lock = threading.Lock()

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        argv = list(argv)
        with self.lock:
            self.calls.append(argv)
        words = argv[1:4]
        if words == ["ai", "job", "create"]:
            return CommandResult(0, json.dumps({"resource_id": "aijob-e00test000001"}), "")
        if words == ["ai", "job", "get"]:
            state = self.states[min(self.gets, len(self.states) - 1)]
            self.gets += 1
            status: dict[str, Any] = {"state": state}
            if state in ("RUNNING", "COMPLETED"):
                status["started_at"] = "2026-10-16T10:02:00+00:00"
            if state == "COMPLETED":
                status["finished_at"] = "2026-10-16T10:05:00+00:00"
            return CommandResult(
                0,
                json.dumps(
                    {
                        "metadata": {
                            "id": "aijob-e00test000001",
                            "name": "rtt-agent",
                            "created_at": "2026-10-16T10:00:00+00:00",
                        },
                        "spec": {"platform": "cpu-d3", "preset": "16vcpu-64gb"},
                        "status": status,
                    }
                ),
                "",
            )
        if words == ["ai", "job", "cancel"]:
            return CommandResult(0, "{}", "")
        return CommandResult(1, "", f"unexpected {argv}")


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}

    def upload_file(self, local: Path, key: str) -> str:
        self.objects[key] = Path(local).read_bytes()
        return key

    def put_json(self, key: str, data: Any) -> str:
        self.objects[key] = json.loads(json.dumps(data))
        return key

    def get_json(self, key: str) -> Any:
        return self.objects[key]


def test_nebius_runner_launch_wait_results_and_cost(backend: FakePartBackend) -> None:
    cli = ScriptedCLI(["PROVISIONING", "RUNNING", "COMPLETED"])
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", cli), CONFIG, sleep=lambda s: None)
    specs: list[Any] = []

    def factory(run_name: str, part_id: str, designs: Sequence[FingerDesign]) -> Any:
        from rtt.cloud.nebius_jobs import JobSpec

        spec = JobSpec(name=f"rtt-{run_name}", image=IMAGE, platform="cpu-d3", preset="16vcpu-64gb")
        specs.append((run_name, spec))
        return spec

    fetched: list[str] = []

    def fetch(run_name: str) -> list[DesignScore]:
        fetched.append(run_name)
        return backend.evaluate(PART, ENTRANTS[:1])

    now = [1000.0]
    runner = NebiusJobRunner(
        jobs,
        spec_factory=factory,
        fetch_results=fetch,
        usd_per_hour=1.2,
        max_wall_s=600.0,
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert runner.estimate_usd(3) == pytest.approx(0.2)
    record = runner.launch(PART, ENTRANTS[:1])
    assert record.job_id == "aijob-e00test000001" and record.state == "QUEUED"
    assert record.run_name == specs[0][0] and record.platform == "cpu-d3"
    done = runner.wait(record.job_id, 120.0)
    assert done.state == "COMPLETED"
    assert done.est_usd == pytest.approx(1.2 * 300 / 3600)  # wall time 10:00 -> 10:05
    assert [s.design_id for s in runner.results(record.job_id)] == [ENTRANTS[0].design_id()]
    assert fetched == [record.run_name]
    assert runner.cancel(record.job_id).state == "COMPLETED"  # terminal jobs are not cancelled
    assert not any(c[1:4] == ["ai", "job", "cancel"] for c in cli.calls)


def test_nebius_runner_cancel(backend: FakePartBackend) -> None:
    cli = ScriptedCLI(["PROVISIONING"])
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", cli), CONFIG, sleep=lambda s: None)
    from rtt.cloud.nebius_jobs import JobSpec

    runner = NebiusJobRunner(
        jobs,
        spec_factory=lambda r, p, d: JobSpec(name=f"rtt-{r}", image=IMAGE),
        fetch_results=lambda r: [],
        usd_per_hour=4.5,
    )
    record = runner.launch(PART, ENTRANTS[:1])
    assert runner.cancel(record.job_id).state == "CANCELLED"
    assert any(c[1:4] == ["ai", "job", "cancel"] for c in cli.calls)


def test_part_job_factory_stages_inputs_and_builds_a_safe_spec(tmp_path: Path) -> None:
    part_dir = tmp_path / PART
    part_dir.mkdir()
    (part_dir / "source.stl").write_bytes(b"solid x\nendsolid x\n")
    (part_dir / "part.json").write_text(
        json.dumps({"intake": {"units": "mm", "name": "box", "density": None}})
    )
    store = FakeStore()
    factory = part_job_factory(CONFIG, store, tmp_path, platform="cpu-d3", preset="16vcpu-64gb")
    spec = factory("agent-0123456789ab", PART, ENTRANTS[:2])
    spec.validate()
    prefix = "runs/agent-0123456789ab/inputs"
    assert set(store.objects) == {f"{prefix}/part.stl", f"{prefix}/part.json", f"{prefix}/designs.json"}
    assert store.objects[f"{prefix}/part.json"]["part_id"] == PART
    assert [d["design_id"] for d in store.objects[f"{prefix}/designs.json"]] == [
        d.design_id() for d in ENTRANTS[:2]
    ]
    assert list(spec.args) == ["-m", "rtt.agent.remote", "--run-dir", "/out/runs/agent-0123456789ab"]
    assert spec.platform == "cpu-d3" and spec.restart_policy == "never"
    assert not spec.env_secrets  # the job needs no model key
    assert all("KEY" not in k and "TOKEN" not in k for k in spec.env)
    assert "abc123" not in json.dumps(dict(spec.env))  # the registry path never reaches the container env


def test_remote_entry_publishes_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "bucket" / "runs" / "agent-0123456789ab"
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    trimesh.creation.box(extents=(40.0, 30.0, 60.0)).export(inputs / "part.stl")
    (inputs / "part.json").write_text(json.dumps({"part_id": PART, "units": "mm", "name": "box"}))
    designs = [{"design_id": d.design_id(), **d.model_dump()} for d in ENTRANTS[:2]]
    (inputs / "designs.json").write_text(json.dumps(designs))
    monkeypatch.setattr("rtt.agent.backends.default_part_backend", lambda root, **kw: FakePartBackend(root))
    assert remote.main(["--run-dir", str(run_dir), "--stage", str(tmp_path / "stage")]) == 0
    results = json.loads((run_dir / "results.json").read_text())
    assert results["status"] == "complete"
    scores = [DesignScore.from_json(s) for s in results["scores"]]
    assert [s.design_id for s in scores] == [d["design_id"] for d in designs]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["exit_status"] == 0 and manifest["kind"] == "agent-part"
    assert "results.json" in manifest["files"]


def test_remote_entry_records_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "agent-broken"
    (run_dir / "inputs").mkdir(parents=True)
    assert remote.main(["--run-dir", str(run_dir), "--stage", str(tmp_path / "stage")]) == 2
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["exit_status"] == 2 and "FileNotFoundError" in manifest["error"]


# --------------------------------------------------------------------------- verifier additions


def test_nebius_reservation_covers_the_job_timeout(backend: FakePartBackend) -> None:
    # A Nebius timeout is whole hours (>= 1h): the reservation must cover it plus provisioning,
    # and a job still running after that is cancelled, so the charge never outgrows it.
    from rtt.agent.jobs import JOB_TIMEOUT, PROVISIONING_ALLOWANCE_S, job_timeout_s
    from rtt.cloud.nebius_jobs import JobSpec

    cli = ScriptedCLI(["RUNNING"])
    jobs = NebiusJobs(NebiusCLI("nebius", "rtt", cli), CONFIG, sleep=lambda s: None)
    now = [0.0]  # moved past the scripted created_at after launch
    runner = NebiusJobRunner(
        jobs,
        spec_factory=lambda r, p, d: JobSpec(name=f"rtt-{r}", image=IMAGE),
        fetch_results=lambda r: [],
        usd_per_hour=0.132,
        poll_s=60.0,
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert runner.max_wall_s == job_timeout_s(JOB_TIMEOUT) + PROVISIONING_ALLOWANCE_S == 4500.0
    reserved = runner.estimate_usd(4)
    assert reserved == pytest.approx(0.132 * 4500 / 3600)
    record = runner.launch(PART, ENTRANTS[:1])
    now[0] = record.created_at + 600.0
    assert runner.status(record.job_id).state == "RUNNING"
    done = runner.wait(record.job_id, 10_000.0)
    assert done.state == "CANCELLED" and "reserved wall time" in (done.error or "")
    assert any(c[1:4] == ["ai", "job", "cancel"] for c in cli.calls)
    assert done.est_usd <= reserved + 0.132 * runner.poll_s / 3600 + 1e-9


def test_job_timeout_parsing() -> None:
    from rtt.agent.jobs import job_timeout_s

    assert job_timeout_s("1h") == 3600.0 and job_timeout_s("3h") == 10800.0
    for bad in ("0h", "90m", "1.5h", ""):
        with pytest.raises(ValueError):
            job_timeout_s(bad)


def test_nebius_from_config_defaults_to_the_cpu_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rtt.cloud.cost import hourly_usd

    uk_image = "cr.uk-south2.nebius.cloud/uk000/rtt-sim@sha256:" + "1" * 64
    config = NebiusConfig(
        profile="rtt",
        project_id="project-home",
        subnet_id="vpcsubnet-home",
        bucket_id="storagebucket-home",
        bucket_name="bucket-home",
        image=IMAGE,
        cli="nebius",
        region="eu-north1",
        region_overrides={
            "UKSOUTH2": {
                "project_id": "project-uk",
                "subnet_id": "vpcsubnet-uk",
                "bucket_id": "storagebucket-uk",
                "bucket_name": "bucket-uk",
                "s3_profile": "rtt-uk",
                "image": uk_image,
            }
        },
    )
    stores: list[Any] = []

    def fake_store(cfg: Any) -> FakeStore:
        stores.append(cfg)
        return FakeStore()

    monkeypatch.setattr("rtt.cloud.storage.BucketStore.from_config", fake_store)
    runner = NebiusJobRunner.from_config(tmp_path, config=config)
    assert runner.usd_per_hour == pytest.approx(hourly_usd("cpu-d3", "4vcpu-16gb"))
    assert runner.max_wall_s == 4500.0
    assert runner.estimate_usd(8) * 2 <= 0.50  # two worst-case jobs fit the default session cap
    assert stores[0].region == "uk-south2" and stores[0].bucket_name == "bucket-uk"
    part_dir = tmp_path / PART
    part_dir.mkdir()
    (part_dir / "source.stl").write_bytes(b"solid x\nendsolid x\n")
    (part_dir / "part.json").write_text(json.dumps({"intake": {"units": "mm", "name": "box"}}))
    spec = runner.spec_factory("agent-0123456789ab", PART, ENTRANTS[:1])
    spec.validate()
    assert (spec.platform, spec.preset, spec.timeout) == ("cpu-d3", "4vcpu-16gb", "1h")
    assert spec.image == uk_image and spec.parent_id == "project-uk" and spec.subnet_id == "vpcsubnet-uk"
    assert spec.volumes[0].startswith("storagebucket-uk:")


def test_local_job_errors_do_not_leak_local_paths(tmp_path: Path) -> None:
    def evaluate(part_id: str, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        raise RuntimeError(f"XML Error: could not open {tmp_path}/a/b/part_0.stl")

    runner = LocalJobRunner(evaluate)
    try:
        record = runner.launch(PART, ENTRANTS[:1])
        done = runner.wait(record.job_id, 10.0)
        assert done.state == "FAILED" and "part_0.stl" in (done.error or "")
        assert str(tmp_path) not in (done.error or "")
    finally:
        runner.shutdown()


def test_nebius_runner_refuses_a_non_finite_price() -> None:
    # a NaN price would make every reservation NaN, and NaN never exceeds a cap
    for kwargs in ({"usd_per_hour": float("nan")}, {"usd_per_hour": 0.1, "max_wall_s": float("inf")}):
        with pytest.raises(ValueError):
            NebiusJobRunner(object(), spec_factory=lambda r, p, d: None, fetch_results=lambda r: [], **kwargs)
