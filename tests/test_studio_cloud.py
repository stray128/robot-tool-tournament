"""Track S6: the Nebius fan-out of the Studio search and the use-case benchmark v1 (product mode).

No Nebius: the fan-out runs on the local fake cloud (the job entry point in subprocesses) and on a
scripted fake of the ``nebius`` CLI that executes each job against local folders standing for the
regions' buckets. Physics uses S1's ``part`` and ``bin`` templates with the S1 reference controllers
(fast, deterministic) except where a test says otherwise. The benchmark's statistics are tested on
synthetic rows.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import shlex
import shutil
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.config import load_config as load_nebius_config
from rtt.cloud.cost import SpendCapError, SpendLedger, read_spend
from rtt.cloud.nebius_jobs import Candidate, CommandResult, NebiusCLI, NebiusJobs
from rtt.cloud.runs import verify_run
from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.studio import cloud
from rtt.studio.cloud import (
    CloudPool,
    InProcessExecutor,
    JobPlan,
    NebiusExecutor,
    SubprocessExecutor,
    dumps,
    pack_jobs,
    plan_shards,
    safe_loads,
    studio_job_spec,
)
from rtt.studio.episode import StudioJob, run_task
from rtt.studio.runner import SegmentJob, WorkerPool, condition_grid, run_full_tasks, run_segment_job
from rtt.studio.search import run_task_tournament
from rtt.studio.types import Setup
from studio_fixtures import cup, gripper, two_press_task
from test_cloud_jobs import FakeClock, FakeNebius

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "usecase_v1.yaml"
DOC = REPO / "docs" / "USECASE_V1.md"


def _script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "studio_usecase_v1", REPO / "scripts" / "studio_usecase_v1.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["studio_usecase_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


uc = _script()


def _strip(report: dict[str, Any]) -> dict[str, Any]:
    """A tournament report without its timings (wall times differ between any two runs)."""
    d = json.loads(json.dumps(report))
    d.pop("timings")
    d.pop("workers")
    d["screen"].pop("wall_s")
    for f in d["finalists"]:
        for r in f["runs"]:
            r.pop("wall_time_s")
    return d


def _items(task: Any, controllers: str = "reference") -> list[Any]:
    setups = [
        Setup.single(gripper(), task.step_ids),
        Setup.tool_change({"press_carton": cup(), "press_flat": gripper()}),
    ]
    return [StudioJob(s, task, c, controllers=controllers) for s in setups for c in condition_grid(task, 2)]


# --------------------------------------------------------------------------- shards, jobs, pickles


def test_shards_group_a_candidate_per_chain_and_a_run_per_setup_and_condition() -> None:
    task = two_press_task()
    grid = condition_grid(task, 3)
    screen = [
        SegmentJob(Setup.single(spec, task.step_ids), task, c, tag=(spec.effector_id, 1, c.condition_id))
        for spec in (gripper(), cup())
        for c in grid
    ]
    shards = plan_shards(screen)
    assert [s.n for s in shards] == [3, 3] and shards[0].key == ("screen", gripper().effector_id, "chain1")
    assert sorted(i for s in shards for i in s.indices) == list(range(6))
    runs = plan_shards(_items(task))
    assert len(runs) == 4 and all(s.n == 1 and s.key[0] == "run" for s in runs)
    assert cloud.phase_kind(screen) == "screen-c1" and cloud.phase_kind(_items(task)) == "runs"
    jobs = pack_jobs("usecase-dev-h1-tote-b", "p00-screen-c0", runs, 3)
    assert [j.shards for j in jobs] == [("s0000", "s0003"), ("s0001",), ("s0002",)]
    assert all(cloud._JOB_NAME.match(j.name) and len(j.name) <= 58 for j in jobs)
    long = cloud.job_name("x" * 59, "eval-p00-screen-c0-1-2-3", "j00")
    assert cloud._JOB_NAME.match(long) and len(long) <= 58


def test_safe_loads_restores_work_items_and_refuses_any_other_global() -> None:
    task = two_press_task()
    items = _items(task)
    back = safe_loads(dumps(items))
    assert [(i.setup.setup_id, i.condition.condition_id) for i in back] == [
        (i.setup.setup_id, i.condition.condition_id) for i in items
    ]

    class Boom:
        def __reduce__(self) -> tuple[Any, ...]:
            import os

            return (os.system, ("true",))

    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        safe_loads(pickle.dumps(Boom()))
    with pytest.raises(pickle.UnpicklingError, match="refused global"):
        safe_loads(pickle.dumps(run_task))  # a function, even a Studio one, is not a work item


def test_job_entry_publishes_a_verified_run_and_refuses_a_changed_input(tmp_path: Path) -> None:
    task = two_press_task()
    items = _items(task)
    ex = InProcessExecutor(tmp_path / "bucket")
    pool = CloudPool(ex, "entry-test", session_dir=tmp_path / "session")
    results = pool.map(run_task, items)
    (phase,) = pool.phases
    job_dir = phase.jobs[0].out_dir
    check = verify_run(job_dir)
    assert check.ok, check.describe()
    manifest = check.manifest or {}
    assert manifest["kind"] == "studio-shards" and manifest["label"] == PRODUCT_MODE_LABEL
    assert manifest["files"][-1] == "manifest.json" and manifest["result"]["runs"] == len(items)
    assert [r.traj_sha for r in results] == [run_task(i).traj_sha for i in items]
    # a changed input is refused before anything runs
    phase_dir = ex.phase_dir("entry-test", phase.phase_id)
    (phase_dir / "inputs" / "s0000.pkl").write_bytes(dumps(items[1:2]))
    out, status = cloud.run_shards(phase_dir, "j09", ["s0000"], workers=1)
    assert status == 2 and "does not match its SHA-256" in out["error"]


def test_a_tampered_or_misplaced_result_is_refused_at_the_merge(tmp_path: Path) -> None:
    task = two_press_task()
    items = _items(task)
    ex = InProcessExecutor(tmp_path / "bucket")
    pool = CloudPool(ex, "merge-test", session_dir=tmp_path / "session")
    pool.map(run_task, items)
    (phase,) = pool.phases
    phase_dir = ex.phase_dir("merge-test", phase.phase_id)
    job = phase.jobs[0]
    shard = job.out_dir / "shards" / "s0001.pkl"
    good = shard.read_bytes()
    shard.write_bytes(good + b"\0")
    with pytest.raises(RuntimeError, match="does not match the job manifest"):
        cloud.merge_phase(phase_dir, items, phase.jobs)
    shard.write_bytes(good)
    swapped = list(items)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(RuntimeError, match="does not belong to its item"):
        cloud.merge_phase(phase_dir, swapped, phase.jobs)


def test_g3_the_fake_cloud_tournament_equals_the_local_pool_run_for_run(tmp_path: Path) -> None:
    """G3: S5's whole tournament through the fan-out (screen phases, finalist phase, several jobs
    per phase, job subprocesses) gives the local pool's report, every traj_sha included."""
    task = two_press_task()
    kw: dict[str, Any] = {
        "candidates": [gripper(), cup()],
        "n_conditions": 2,
        "n_screen": 1,
        "controllers": "reference",
    }
    with WorkerPool(1) as p:
        local = run_task_tournament(task, pool=p, **kw).to_dict()
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    ex = SubprocessExecutor(tmp_path / "bucket", parallel=2, workers=1, ledger=ledger)
    pool = CloudPool(ex, "g3-test", session_dir=tmp_path / "session")
    fanned = run_task_tournament(task, pool=pool, **kw).to_dict()
    assert _strip(fanned) == _strip(local)
    runs = sorted(
        (f["setup"]["setup_id"], r["condition_id"], r["traj_sha"])
        for f in fanned["finalists"]
        for r in f["runs"]
    )
    assert runs == sorted(
        (f["setup"]["setup_id"], r["condition_id"], r["traj_sha"])
        for f in local["finalists"]
        for r in f["runs"]
    )
    assert [p.phase_id for p in pool.phases] == ["p00-screen-c0", "p01-screen-c1", "p02-runs"]
    assert sum(len(p.jobs) for p in pool.phases) >= 4
    # every row: its job id, which keys a ledger line in the existing format, and its dollar share
    spend = {e.job_id: e for e in read_spend(tmp_path / "spend.jsonl")}
    assert len(spend) == sum(len(p.jobs) for p in pool.phases)
    for row in pool.rows:
        c = row["cloud"]
        assert c["job_id"] in spend and c["est_usd"] >= 0 and row["label"] == PRODUCT_MODE_LABEL
        assert spend[c["job_id"]].platform == "cpu-d3" and spend[c["job_id"]].region == "local"
    total = sum(r["cloud"]["est_usd"] for r in pool.rows)
    assert total == pytest.approx(sum(e.est_usd for e in spend.values()), rel=1e-4, abs=1e-7)
    assert len(pool.rows) == local["screen"]["runs"] + sum(len(f["runs"]) for f in local["finalists"])
    rows = [json.loads(line) for line in (tmp_path / "session" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == len(pool.rows) and not find_brand_tokens(json.dumps(rows))


def test_g3_real_controllers_on_a_hero_setup_match_in_process(tmp_path: Path) -> None:
    """G3 with S2's real primitives: a hero task's first step pair on two conditions, through job
    subprocesses, equals two in-process runs."""
    specs = uc.hero_specs(uc.load_config(CONFIG))
    task = specs["h2_coupling"]
    from rtt.studio.search import default_candidates

    spec = next(
        c.spec
        for c in default_candidates(library=False, designed=False, task_spec=task)
        if c.source == "preset"
    )
    setup = Setup.single(spec, task.step_ids)
    items = [StudioJob(setup, task, c) for c in condition_grid(task, 2)]
    one = [run_task(i).traj_sha for i in items]
    two = [run_task(i).traj_sha for i in items]
    pool = CloudPool(
        SubprocessExecutor(tmp_path / "bucket", parallel=2), "g3-real", session_dir=tmp_path / "s"
    )
    fanned = [r.traj_sha for r in pool.map(run_task, items)]
    assert one == two == fanned


def test_cloud_pool_is_a_worker_pool_for_run_full_tasks(tmp_path: Path) -> None:
    task = two_press_task()
    setups = [Setup.single(gripper(), task.step_ids), Setup.single(cup(), task.step_ids)]
    grid = condition_grid(task, 2)
    with WorkerPool(1) as p:
        local = run_full_tasks(setups, task, grid, pool=p, controllers="reference")
    with CloudPool(InProcessExecutor(tmp_path / "b"), "rft-test", session_dir=tmp_path / "s") as pool:
        fanned = run_full_tasks(setups, task, grid, pool=pool, controllers="reference")
    assert {k: [r.traj_sha for r in v] for k, v in fanned.items()} == {
        k: [r.traj_sha for r in v] for k, v in local.items()
    }
    with pytest.raises(FileExistsError):
        CloudPool(InProcessExecutor(tmp_path / "b"), "rft-test", session_dir=tmp_path / "s").map(
            run_task, [StudioJob(setups[0], task, grid[0], controllers="reference")]
        )
    assert pool.map(run_segment_job, []) == []
    with pytest.raises(ValueError, match="not a Studio work function"):
        pool.map(len, [1])


# --------------------------------------------------------------------------- Nebius (scripted fake)

IMAGE = "cr.eu-north1.nebius.cloud/abc123/rtt-sim@sha256:" + "3" * 64
IMAGE_UKS2 = "cr.uk-south2.nebius.cloud/uks2/rtt-sim@sha256:" + "3" * 64
ENV = f"""
NEBIUS_PROFILE=rtt
NEBIUS_TENANT_ID=tenant-test
NEBIUS_PROJECT_ID=project-eun1
NEBIUS_SUBNET_ID=vpcsubnet-eun1
NEBIUS_BUCKET_NAME=bucket-eun1
NEBIUS_BUCKET_ID=storagebucket-eun1
NEBIUS_S3_PROFILE=nebius-eun1
NEBIUS_PROJECT_ID_UKSOUTH2=project-uks2
NEBIUS_SUBNET_ID_UKSOUTH2=vpcsubnet-uks2
NEBIUS_BUCKET_NAME_UKSOUTH2=bucket-uks2
NEBIUS_BUCKET_ID_UKSOUTH2=storagebucket-uks2
NEBIUS_S3_PROFILE_UKSOUTH2=nebius-uks2
RTT_IMAGE={IMAGE}
RTT_IMAGE_UKSOUTH2={IMAGE_UKS2}
"""
REGION_OF = {"cpu-d3": "uk-south2", "gpu-h100-sxm": "eu-north1", "gpu-h200-sxm": "eu-north1"}


class LocalStore:
    """A bucket as a local folder (the same folder the fake job mounts at /out)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def upload_file(self, local: Path, key: str) -> str:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)
        return key

    def upload_dir(self, local_dir: Path, prefix: str) -> list[str]:
        return [
            self.upload_file(p, f"{prefix}/{p.relative_to(local_dir).as_posix()}")
            for p in sorted(Path(local_dir).rglob("*"))
            if p.is_file()
        ]

    def download_prefix(self, prefix: str, dest_dir: Path) -> list[Path]:
        src = self.root / prefix
        out = []
        for p in sorted(src.rglob("*")):
            if p.is_file():
                dst = Path(dest_dir) / p.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
                out.append(dst)
        return out


class ExecutingNebius(FakeNebius):
    """The scripted CLI, plus: a job whose script completes runs its container command in-process
    against its region's bucket folder (``/out`` -> that folder)."""

    def __init__(self, scripts: dict[str, list[str]], buckets: Path) -> None:
        super().__init__(scripts)
        self.buckets = buckets
        self.ran: list[list[str]] = []

    _lock = threading.Lock()  # jobs run in the test process: one at a time (a real job is a process)

    def __call__(self, argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        res = super().__call__(argv, timeout_s)
        argv = list(argv)
        if argv[1:4] == ["ai", "job", "get"]:
            data = json.loads(res.stdout)
            if data["status"]["state"] == "ERROR":  # ended before start: a finish time, no start
                data["status"]["finished_at"] = "2026-10-15T09:18:39.585569020Z"
                return CommandResult(0, json.dumps(data), "")
        if argv[1:4] == ["ai", "job", "create"] and "--dry-run" not in argv:
            platform = argv[argv.index("--platform") + 1]
            if "COMPLETED" in self.scripts[platform]:
                args = shlex.split(argv[argv.index("--args") + 1])
                assert args[:3] == ["-m", "rtt.studio.cloud", "shards"]
                mount = self.buckets / REGION_OF[platform]
                local = [a.replace("/out", str(mount), 1) if a.startswith("/out/") else a for a in args[2:]]
                local[local.index("--workers") + 1] = "1"
                self.ran.append(local)
                with self._lock:
                    assert cloud.main(local) == 0
        return res


def _nebius(tmp_path: Path, fake: ExecutingNebius, cap: float = 5.0, **kw: Any) -> NebiusExecutor:
    env = tmp_path / "nebius.env"
    env.write_text(ENV)
    cfg = load_nebius_config(environ={}, env_files=[env])
    clock = FakeClock()
    return NebiusExecutor(
        cfg,
        staging=tmp_path / "staging",
        ledger=SpendLedger(tmp_path / "bench-spend.jsonl", cap_usd=cap),
        project_ledger=SpendLedger(tmp_path / "project-spend.jsonl"),
        jobs_for=lambda c: NebiusJobs(NebiusCLI("nebius", "rtt", fake), c, sleep=clock.sleep, clock=clock),
        store_for=lambda c: LocalStore(tmp_path / "buckets" / c.region),
        check_capacity=False,
        poll_s=1.0,
        **kw,
    )


def _creates(fake: FakeNebius) -> list[list[str]]:
    return [c for c in fake.calls if c[1:4] == ["ai", "job", "create"]]


def test_nebius_executor_runs_cpu_jobs_without_any_secret_and_records_spend(tmp_path: Path) -> None:
    task = two_press_task()
    items = _items(task)
    fake = ExecutingNebius({"cpu-d3": ["PROVISIONING", "RUNNING", "COMPLETED"]}, tmp_path / "buckets")
    ex = _nebius(tmp_path, fake, jobs_per_phase=2)
    pool = CloudPool(ex, "nebius-test", session_dir=tmp_path / "session", prefix="eval-")
    results = pool.map(run_task, items)
    assert [r.traj_sha for r in results] == [run_task(i).traj_sha for i in items]
    creates = _creates(fake)
    assert len(creates) == 2
    for argv in creates:
        flags = {argv[i]: argv[i + 1] for i in range(4, len(argv) - 1) if argv[i].startswith("--")}
        assert flags["--platform"] == "cpu-d3" and flags["--preset"] == "32vcpu-128gb"
        assert flags["--image"] == IMAGE_UKS2 and flags["--volume"] == "storagebucket-uks2:/out:rw"
        assert flags["--parent-id"] == "project-uks2" and flags["--restart-policy"] == "never"
        assert "--env-secret" not in argv and "--inject-file" not in argv
        assert "/out/runs/nebius-test/eval-p00-runs" in flags["--args"]
    bench = read_spend(tmp_path / "bench-spend.jsonl")
    assert {e.job_id for e in bench} == {r["cloud"]["job_id"] for r in pool.rows} == set(fake.jobs)
    assert [e.job_id for e in read_spend(tmp_path / "project-spend.jsonl")] == [e.job_id for e in bench]
    assert all(e.region == "uk-south2" and e.platform == "cpu-d3" and e.est_usd > 0 for e in bench)
    assert all(r["cloud"]["executor"] == "nebius" for r in pool.rows)


def test_nebius_executor_moves_to_the_h100_only_when_uk_south2_cannot_start(tmp_path: Path) -> None:
    task = two_press_task()
    items = _items(task)[:2]
    fake = ExecutingNebius(
        {"cpu-d3": ["PROVISIONING", "ERROR"], "gpu-h100-sxm": ["RUNNING", "COMPLETED"]}, tmp_path / "buckets"
    )
    ex = _nebius(tmp_path, fake)
    pool = CloudPool(ex, "fallback-test", session_dir=tmp_path / "session")
    results = pool.map(run_task, items)
    assert [r.traj_sha for r in results] == [run_task(i).traj_sha for i in items]
    platforms = [c[c.index("--platform") + 1] for c in _creates(fake)]
    assert platforms == ["cpu-d3", "gpu-h100-sxm"]
    assert (tmp_path / "buckets" / "eu-north1" / "runs" / "fallback-test").is_dir()  # inputs went there too
    regions = [e.region for e in read_spend(tmp_path / "bench-spend.jsonl")]
    assert regions == ["uk-south2", "eu-north1"]
    (job,) = pool.phases[0].jobs
    assert [a.platform for a in job.attempts] == ["cpu-d3", "gpu-h100-sxm"]
    assert {r["cloud"]["platform"] for r in pool.rows} == {"gpu-h100-sxm"}


def test_nebius_executor_needs_a_cap_and_refuses_a_job_the_cap_cannot_pay_for(tmp_path: Path) -> None:
    fake = ExecutingNebius({"cpu-d3": ["COMPLETED"]}, tmp_path / "buckets")
    env = tmp_path / "nebius.env"
    env.write_text(ENV)
    with pytest.raises(ValueError, match="capped spend ledger"):
        NebiusExecutor(
            load_nebius_config(environ={}, env_files=[env]),
            staging=tmp_path / "s",
            ledger=SpendLedger(tmp_path / "x.jsonl"),
        )
    ex = _nebius(tmp_path, fake, cap=0.05)
    pool = CloudPool(ex, "cap-test", session_dir=tmp_path / "session")
    with pytest.raises(SpendCapError):
        pool.map(run_task, _items(two_press_task())[:1])
    assert _creates(fake) == []


def test_studio_job_spec_is_a_cpu_job_with_no_secret(tmp_path: Path) -> None:
    env = tmp_path / "nebius.env"
    env.write_text(ENV)
    cfg = load_nebius_config(environ={}, env_files=[env]).for_region("uk-south2")
    job = JobPlan("j00", cloud.job_name("usecase-v1-eval", "eval-p00-runs", "j00"), ("s0000", "s0001"))
    spec = studio_job_spec(
        cfg, "usecase-v1-eval", "eval-p00-runs", job, Candidate("cpu-d3", "32vcpu-128gb", "uk-south2")
    )
    argv = spec.create_args()
    assert spec.env_secrets == {} and not spec.inject_files and "--env-secret" not in argv
    assert spec.args[:3] == ["-m", "rtt.studio.cloud", "shards"] and spec.command == cloud.CONTAINER_PYTHON
    assert "abc123" not in repr(spec) and "uks2" not in repr(spec)  # no registry path in logs


# --------------------------------------------------------------------------- the benchmark


def test_the_draft_config_pins_the_heroes_and_holds_a_regenerable_unsecreted_frozen_block() -> None:
    """The draft's frozen block is the script's output (complete and self-consistent: every setup
    hashes to its id, the primary rule follows from the arms, the arms were verified on the code),
    with no secret yet, so ``freeze-arms --refreeze`` can still regenerate it."""
    from test_studio_engineer import FROZEN_HERO_TASK_IDS

    cfg = uc.load_config(CONFIG)
    assert cfg["label"] == PRODUCT_MODE_LABEL and cfg["status"] == "draft"
    assert cfg["heroes"]["cases"] == FROZEN_HERO_TASK_IDS
    assert cfg["heldout"]["conditions"] == 32 and cfg["heldout"]["placement_mm"] == 2.0
    assert cfg["development"]["conditions"] == 8 and cfg["development"]["screen"] == 3
    assert cfg["analysis"]["alpha"] == 0.05 and cfg["analysis"]["multiplicity"] == "holm"
    assert cfg["executor"]["cap_usd"] == 10.0 and cfg["executor"]["token_factory_usd"] == 0.0
    assert cfg["heldout"]["secret_file"].startswith("secrets/")
    frozen = cfg["frozen"]
    assert frozen is not None and frozen["commitment"] is None and frozen["label"] == PRODUCT_MODE_LABEL
    assert uc.frozen_problems(cfg) == []
    verified = frozen["code"]["verified"]
    assert verified["runs"] > 0 and verified["equal"] == verified["runs"] and not verified["unequal"]
    for case, h in frozen["heroes"].items():
        assert set(h["arms"]) == set(uc.ARMS) and h["dev_reports"]["b"]["sha256"]
        assert "/" not in str(h["dev_reports"]["b"]["path"]).split(case, 1)[0]  # relative, no private path
    for path in (CONFIG, DOC, REPO / "scripts" / "studio_usecase_v1.py", Path(cloud.__file__)):
        assert not find_brand_tokens(path.read_text(encoding="utf-8")), path
    assert PRODUCT_MODE_LABEL in DOC.read_text(encoding="utf-8")


def test_heldout_conditions_are_deterministic_domain_separated_and_bounded() -> None:
    cfg = uc.load_config(CONFIG)
    specs = uc.hero_specs(cfg)
    h1 = specs["h1_tote"]
    a = uc.heldout_conditions("ab" * 32, h1, cfg["heldout"])
    assert a == uc.heldout_conditions("ab" * 32, h1, cfg["heldout"])
    assert len(a) == 32 and [c.condition_id for c in a][:2] == ["heldout00", "heldout01"]
    assert len({c.seed for c in a}) == 32 and min(c.seed for c in a) >= 1_000_000
    assert all(set(c.aim_offsets_mm) == {o.name for o in h1.objects} for c in a)
    assert all(abs(v) <= 2.0 for c in a for xy in c.aim_offsets_mm.values() for v in xy)
    assert a != uc.heldout_conditions("cd" * 32, h1, cfg["heldout"])
    other = uc.heldout_conditions("ab" * 32, specs["h2_coupling"], cfg["heldout"])
    assert {c.seed for c in other}.isdisjoint({c.seed for c in a})
    dev = uc.dev_conditions(h1, cfg)
    assert {c.seed for c in dev} == set(range(8)) and {c.seed for c in dev}.isdisjoint({c.seed for c in a})


def test_exact_paired_test_holm_and_intervals_on_known_values() -> None:
    t = uc.exact_paired_one_sided([True] * 5 + [False] * 3, [False] * 5 + [False] * 3)
    assert (t["n10"], t["n01"], t["ties"]) == (5, 0, 3) and t["p"] == pytest.approx(1 / 32)
    t = uc.exact_paired_one_sided([True, True, True, False, True], [False, False, False, True, True])
    assert (t["n10"], t["n01"]) == (3, 1) and t["p"] == pytest.approx(5 / 16)
    assert uc.exact_paired_one_sided([False] * 4, [False] * 4)["p"] == 1.0
    h = uc.holm({"h1": 0.01, "h2": 0.04, "h3": 0.03}, 0.05)
    assert h["h1"]["reject"] and not h["h3"]["reject"] and not h["h2"]["reject"]
    assert [h[k]["p_holm"] for k in ("h1", "h3", "h2")] == pytest.approx([0.03, 0.06, 0.06])
    assert uc.holm({"h1": 0.001, "h2": 0.02}, 0.05)["h2"]["reject"]
    assert uc.clopper_pearson(0, 10) == pytest.approx([0.0, 0.3085], abs=1e-4)
    assert uc.clopper_pearson(10, 10)[1] == 1.0 and uc.clopper_pearson(0, 0) is None


def _arm(sid: str | None, status: str) -> dict[str, Any]:
    if sid is None:
        return {"setup_id": None, "status": "missing", "basis": "none"}
    return {
        "setup_id": sid,
        "status": status,
        "kind": "single",
        "label": f"setup {sid}",
        "tools": ["Tool"],
        "complexity": {
            "n_tools": 1,
            "n_actuators": 1,
            "tool_mass_kg": 0.5,
            "total_mass_kg": 0.5,
            "n_changes": 0,
        },
        "dev": {
            "completion_rate": 0.5,
            "runs": [{"condition_id": "nominal", "seed": 0, "completed": True, "traj_sha": f"dev-{sid}"}],
        },
    }


def _row(
    hero: str, arm: str, cond: dict[str, Any], completed: bool, which: str = "heldout", sha: str = "x"
) -> dict[str, Any]:
    step = {
        "step_id": "s1",
        "primitive": "grasp",
        "success": completed,
        "failure_mode": None if completed else "slip",
        "fidelity": "rigid",
    }
    return {
        "hero": hero, "arm": arm, "set": which, "condition_id": cond["condition_id"], "seed": cond["seed"],
        "completed": completed, "controllers": "default", "fairness_sha": "f" * 64, "cycle_time_s": 12.0,
        "tool_mass_kg": [0.5], "fidelity": "rigid", "steps": [step], "traj_sha": sha,
        "cloud": {"job_id": "aijob-1", "job_est_usd": 0.05, "est_usd": 0.001},
    }  # fmt: skip


def _synthetic(n: int = 16) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    conds = {
        h: [{"condition_id": f"heldout{i:02d}", "seed": 1_000_000 + i} for i in range(n)]
        for h in ("h1", "h2")
    }
    cfg = {
        "analysis": {"alpha": 0.05, "test": "exact one-sided paired", "multiplicity": "holm"},
        "frozen": {
            "commitment": "c" * 64,
            "heroes": {
                "h1": {"task_id": "1" * 64, "primary": {"tested": True, "reason": None},
                       "arms": {"a": _arm("sa", "tested"), "b": _arm("sb", "tested"), "c": _arm("sc", "descriptive")}},
                "h2": {"task_id": "2" * 64, "primary": {"tested": False, "reason": "arm b has no recommendation"},
                       "arms": {"a": _arm("ta", "tested"), "b": _arm("tb", "descriptive"), "c": _arm(None, "missing")}},
            },
        },
    }  # fmt: skip
    rows = []
    for i, c in enumerate(conds["h1"]):
        rows += [_row("h1", "a", c, i < 2), _row("h1", "b", c, i < 12), _row("h1", "c", c, i < 6)]
    for i, c in enumerate(conds["h2"]):
        rows += [_row("h2", "a", c, False), _row("h2", "b", c, i < 3)]
    dev = {"condition_id": "nominal", "seed": 0}
    rows += [_row("h1", "a", dev, True, "parity", "dev-sa"), _row("h1", "b", dev, True, "parity", "other")]
    return cfg, rows, conds


def test_analysis_on_synthetic_rows_tests_only_the_declared_heroes() -> None:
    cfg, rows, conds = _synthetic()
    out = uc.analyse(rows, cfg, conditions=conds, mode="dry_run")
    assert out["label"] == PRODUCT_MODE_LABEL and out["family"] == ["h1"] and out["problems"] == []
    assert out["excluded"] == {"h2": "arm b has no recommendation"}
    p = out["heroes"]["h1"]["primary"]
    assert (p["n10"], p["n01"], p["ties"]) == (10, 0, 6) and p["p"] == pytest.approx(2**-10)
    assert p["p_holm"] == pytest.approx(2**-10) and p["reject"] is True
    arms = out["heroes"]["h1"]["arms"]
    assert (arms["a"]["completed"], arms["b"]["completed"], arms["c"]["completed"]) == (2, 12, 6)
    assert arms["b"]["steps"]["s1"] == {"primitive": "grasp", "n": 16, "successes": 12, "rate": 0.75,
                                        "fidelity": {"rigid": 16}, "failure_modes": {"slip": 4}}  # fmt: skip
    assert (
        arms["b"]["cycle_time_s"]["median_completed"] == 12.0 and arms["b"]["tools_and_mass"]["n_tools"] == 1
    )
    sec = out["heroes"]["h1"]["secondary_b_vs_c"]
    assert sec["descriptive"] and (sec["b_only"], sec["c_only"]) == (6, 0) and sec["difference"] == 0.375
    assert (
        out["heroes"]["h2"]["arms"]["b"]["completed"] == 3
        and out["heroes"]["h2"]["primary"]["tested"] is False
    )
    parity = {x["arm"]: x["equal"] for x in out["heroes"]["h1"]["parity"]}
    assert parity == {"a": 1, "b": 0, "c": 0}
    assert out["spend"] == {"jobs": 1, "est_usd": 0.05}
    text = uc.summary_text(out)
    assert "primary b > a" in text and "excluded (arm b has no recommendation)" in text


def test_a_pairing_problem_takes_a_hero_out_of_the_family() -> None:
    cfg, rows, conds = _synthetic()
    drop = next(
        i
        for i, r in enumerate(rows)
        if r["hero"] == "h1" and r["arm"] == "b" and r["condition_id"] == "heldout05"
    )
    rows.pop(drop)
    out = uc.analyse(rows, cfg, conditions=conds, mode="confirmatory")
    assert out["family"] == [] and out["problems"] and "h1" in out["excluded"]
    assert "pairing problems" in out["excluded"]["h1"]
    cfg, rows, conds = _synthetic()
    rows[0] = dict(rows[0], fairness_sha="0" * 64)
    assert "one fairness hash" in " ".join(
        uc.analyse(rows, cfg, conditions=conds, mode="dry_run")["problems"]
    )


def test_arm_rules_follow_the_declared_exclusions() -> None:
    def report(rec: str | None, closest: str | None, best: str | None) -> dict[str, Any]:
        fin = [
            {"setup": {"setup_id": s, "kind": "single", "label": s, "tools": [{"display_name": "Tool"}]},
             "result": {"complexity": {}, "fidelity": "rigid", "completion_rate": 0.0},
             "runs": [{"condition_id": "nominal", "seed": 0, "completed": False, "traj_sha": "t"}]}
            for s in {x for x in (rec, closest, best) if x}
        ]  # fmt: skip
        return {
            "finalists": fin,
            "best_single": best,
            "recommendation": {
                "recommendation": {"setup_id": rec} if rec else None,
                "no_winner": None if rec else {"closest": [{"setup_id": closest}] if closest else []},
            },
        }

    a = uc.arm_from_report(report("r1", None, "s1"), "best_single")
    b = uc.arm_from_report(report("r1", None, "s1"), "recommendation")
    assert (a["status"], b["status"]) == ("tested", "tested") and uc.primary_rule({"a": a, "b": b})["tested"]
    b2 = uc.arm_from_report(report(None, "c1", "s1"), "recommendation")
    assert b2["status"] == "descriptive" and b2["setup_id"] == "c1"
    assert "no recommendation" in uc.primary_rule({"a": a, "b": b2})["reason"]
    same = uc.arm_from_report(report("s1", None, "s1"), "recommendation")
    assert "identical by construction" in uc.primary_rule({"a": a, "b": same})["reason"]
    gone = uc.arm_from_report(report("r1", None, None), "best_single")
    assert (
        gone["status"] == "missing" and "arm a is missing" in uc.primary_rule({"a": gone, "b": b})["reason"]
    )
    assert uc.arm_from_report(None, "recommendation")["status"] == "missing"


def test_commit_reveal_order_and_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    shutil.copyfile(CONFIG, cfg_path)
    uc.write_frozen(cfg_path, None)
    with pytest.raises(ValueError, match="freeze the arms first"):
        uc.commit_secret(cfg_path, secret_file=tmp_path / "secrets" / "seed.txt")
    frozen = {"label": PRODUCT_MODE_LABEL, "heroes": {"h1_tote": {"task_id": "x"}}, "commitment": None}
    uc.write_frozen(cfg_path, frozen)
    assert uc.load_config(cfg_path)["frozen"]["heroes"]["h1_tote"]["task_id"] == "x"
    with pytest.raises(ValueError, match="not ready"):
        uc.finalize(cfg_path, secret_file=tmp_path / "secrets" / "seed.txt")
    commitment = uc.commit_secret(cfg_path, secret_file=tmp_path / "secrets" / "seed.txt")
    from rtt.store.vault import HeldoutVault

    assert HeldoutVault(tmp_path / "secrets" / "seed.txt").sha256() == commitment
    assert uc.load_config(cfg_path)["frozen"]["commitment"] == commitment
    with pytest.raises(ValueError, match="never replaced"):
        uc.commit_secret(cfg_path, secret_file=tmp_path / "secrets" / "seed.txt")
    with pytest.raises(ValueError, match="a secret is committed"):
        uc.freeze_arms(cfg_path, executor="local", out=tmp_path / "dev", allow_dirty=True, refreeze=True)
    # confirmatory evaluation: refused before the status is frozen and before the tag
    with pytest.raises(ValueError, match="not frozen"):
        uc.evaluate(
            cfg_path, executor="fake", out=tmp_path / "r", secret_file=tmp_path / "secrets" / "seed.txt"
        )
    uc.set_status(cfg_path, "frozen")
    with pytest.raises(ValueError, match="no tag|differs"):
        uc.evaluate(
            cfg_path, executor="fake", out=tmp_path / "r", secret_file=tmp_path / "secrets" / "seed.txt"
        )
    # a dry run never reads the real secret, nor any secret inside the repository
    with pytest.raises(ValueError, match="never reads the real secret"):
        uc.evaluate(
            cfg_path, executor="fake", out=tmp_path / "r", dry_run=True, secret_file=uc.real_secret_path()
        )
    with pytest.raises(ValueError, match="outside the repository"):
        uc.dry_run(heroes=["h1_tote"], n_conditions=2, out=REPO / "runs" / "never")


def test_end_to_end_on_a_small_task_through_the_fake_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """freeze-arms, commit-secret, evaluate (dry run) and analyse on a two-step task, with the
    reference controllers' stand-in pool replaced by two presets and S2's real push primitive."""
    from rtt.studio.search import candidate_of

    task = two_press_task()
    monkeypatch.setattr(uc, "hero_specs", lambda cfg: {"t1": task})
    monkeypatch.setattr(
        uc,
        "_candidates",
        lambda spec, pool: [
            candidate_of(gripper(), "preset"),
            candidate_of(cup(), "library" if pool == "library" else "preset"),
        ],
    )
    cfg_path = tmp_path / "config.yaml"
    text = CONFIG.read_text(encoding="utf-8").split(uc.FROZEN_MARKER)[0]
    import yaml

    cfg = yaml.safe_load(text)
    cfg["heroes"]["cases"] = {"t1": task.task_id}
    cfg["development"].update(conditions=2, screen=1, finalists=2)
    cfg["heldout"].update(conditions=3, secret_file=str(tmp_path / "secrets" / "usecase_v1_seed.txt"))
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False) + "\n" + uc.FROZEN_MARKER + "\nfrozen: null\n")
    frozen = uc.freeze_arms(cfg_path, executor="fake", out=tmp_path / "dev", parallel=2, allow_dirty=True)
    arms = frozen["heroes"]["t1"]["arms"]
    assert arms["a"]["status"] == "tested" and all(
        arms[k]["status"] in ("tested", "descriptive", "missing") for k in "abc"
    )
    assert (tmp_path / "dev" / "t1" / "b" / "report.json").is_file()
    assert (tmp_path / "dev" / "t1" / "b" / "rows.jsonl").is_file()
    assert frozen["code"]["verified"]["equal"] == frozen["code"]["verified"]["runs"] > 0
    assert uc.frozen_problems(uc.load_config(cfg_path)) == []
    # the block is regenerated by the script, from the reports the freeze kept, until a secret exists
    with pytest.raises(ValueError, match="pass refreeze"):
        uc.freeze_arms(cfg_path, executor="fake", out=tmp_path / "dev2", allow_dirty=True)
    again = uc.freeze_arms(
        cfg_path,
        executor="fake",
        out=tmp_path / "dev2",
        parallel=2,
        allow_dirty=True,
        reuse={"b": tmp_path / "dev", "c": tmp_path / "dev"},
        refreeze=True,
    )
    assert again["heroes"]["t1"]["arms"] == arms
    assert again["heroes"]["t1"]["dev_reports"]["b"]["source"] == "reused"
    report = json.loads((tmp_path / "dev" / "t1" / "b" / "report.json").read_text())
    report["conditions"] = report["conditions"][:1]
    assert "not the development conditions" in uc.check_reused_report(
        report, task, "full", uc.load_config(cfg_path)
    )
    uc.commit_secret(cfg_path)
    with pytest.raises(ValueError, match="a secret is committed"):
        uc.freeze_arms(cfg_path, executor="fake", out=tmp_path / "dev3", allow_dirty=True, refreeze=True)
    res = uc.evaluate(cfg_path, executor="fake", out=tmp_path / "results", dry_run=True, parallel=2)
    a = res["analysis"]
    assert a["mode"] == "dry_run" and a["note"] == uc.DRY_RUN_NOTE and a["problems"] == []
    assert a["secret_revealed"] and len(a["heroes"]["t1"]["arms"]["a"]["steps"]) == 2
    assert all(x["equal"] == x["expected"] for x in a["heroes"]["t1"]["parity"])  # same code, same hashes
    rows = [json.loads(line) for line in (tmp_path / "results" / "rows.jsonl").read_text().splitlines()]
    assert all(r["cloud"]["job_id"].startswith("local-") for r in rows)
    assert {r["condition_id"] for r in rows if r["set"] == "heldout"} == {
        "heldout00",
        "heldout01",
        "heldout02",
    }
    again = uc.analyse_results(cfg_path, tmp_path / "results")
    assert again["heroes"] == a["heroes"]
    with pytest.raises(FileExistsError):
        uc.evaluate(cfg_path, executor="fake", out=tmp_path / "results", dry_run=True)


def test_the_image_must_hold_the_frozen_job_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def diff(*files: str) -> Any:
        return lambda *args: subprocess.CompletedProcess(args, 0, "\n".join(files) + "\n", "")

    monkeypatch.setattr(uc, "_git", diff("docs/USECASE_V1.md", "tests/test_x.py", "configs/usecase_v1.yaml"))
    assert uc.same_code("a", "b")
    for changed in ("src/rtt/studio/episode.py", "configs/task_v2.yaml", "uv.lock", "assets/x.stl"):
        monkeypatch.setattr(uc, "_git", diff("docs/x.md", changed))
        assert not uc.same_code("a", "b"), changed
    monkeypatch.setattr(uc, "_git", lambda *args: subprocess.CompletedProcess(args, 128, "", "bad revision"))
    assert not uc.same_code("a", "b")
