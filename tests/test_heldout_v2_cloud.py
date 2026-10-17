"""The held-out job: the secret reaches it only, the laptop checks before submitting, the
in-container entry checks the picks before it reads the secret, and spend is recorded.

No Nebius: a scripted fake of the ``nebius`` CLI (tests/test_cloud_jobs.py) and local
directories in place of the bucket. Physics is replaced by invalid-build records where a
test only checks plumbing (the runners' physics is covered in tests/test_heldout_v2.py).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from heldout_v2_support import COMMITMENT, SECRET, fake_simulate, mirror_config, write_search
from rtt.cloud import entry
from rtt.cloud.__main__ import Context, build_parser, cmd_run
from rtt.cloud.config import NebiusConfig
from rtt.cloud.cost import read_spend
from rtt.cloud.nebius_jobs import INJECT_FILE_MAX_BYTES, NebiusCLI, NebiusJobs, check_inject_file
from rtt.cloud.runs import (
    KINDS,
    SEED_MOUNT,
    HeldoutJobArgs,
    SearchJobArgs,
    entry_args,
    job_spec_for,
    verify_run,
)
from rtt.heldout_v2 import common, real
from rtt.sim.gso import LocalMirror
from test_cloud_jobs import FakeClock, FakeNebius

DIGEST = "sha256:" + "1" * 64
IMAGE = "cr.uk-south2.nebius.cloud/abc123/rtt-sim@" + DIGEST
CONFIG = NebiusConfig(
    project_id="project-test",
    subnet_id="vpcsubnet-test",
    bucket_id="storagebucket-test",
    bucket_name="bucket",
    image=IMAGE,
    region="uk-south2",
)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two fetched search runs (cloud layout), the scan mirror, the throwaway secret, a selection."""
    base = tmp_path_factory.mktemp("heldout_cloud")
    fetch = base / "runs"
    fetch.mkdir()
    runs = {arm: write_search(fetch, arm, layout="cloud") for arm in ("random", "nemotron")}
    scans = mirror_config(base / "scans")
    selection = base / "selection"
    real.select_real_v2(
        selection,
        mode="scored",
        seed_file=scans["secret"],
        commitment=COMMITMENT,
        config=scans["config"],
        workers=1,
        cache_root=scans["cache"],
        fetch=LocalMirror(scans["mirror"]),
    )
    digest = common.frozen_digest(common.collect_frozen_v2(runs.values()))
    return {"base": base, "fetch": fetch, "runs": runs, "selection": selection, "digest": digest, **scans}


def _heldout_args(digest: str = "f" * 64) -> HeldoutJobArgs:
    return HeldoutJobArgs(frozen_runs=("search-random", "search-nemotron"), frozen_digest=digest)


def _no_secret_in(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


# --------------------------------------------------------------------------- the job spec


def test_seed_reaches_only_the_heldout_job_spec(world: dict[str, Any]) -> None:
    inject = [f"{world['secret']}:{SEED_MOUNT}"]
    search = SearchJobArgs(arm="nemotron", runs=32, budget=40, space="freeform-v2")
    for kind in KINDS:
        if kind == "heldout":
            continue
        kw: dict[str, Any] = {"search": search} if kind == "search" else {}
        spec = job_spec_for(CONFIG, kind, f"{kind}-x", **kw)
        argv = spec.create_args(dry_run=True)
        assert spec.inject_files == () and "--inject-file" not in argv, kind
        assert SEED_MOUNT not in " ".join(argv) and "--seed-file" not in spec.args, kind
        with pytest.raises(ValueError, match="only the heldout job"):
            job_spec_for(CONFIG, kind, f"{kind}-y", inject_files=inject, **kw)
    with pytest.raises(ValueError, match="needs HeldoutJobArgs"):
        job_spec_for(CONFIG, "heldout", "h-x", inject_files=inject)
    with pytest.raises(ValueError, match="exactly one injected file"):
        job_spec_for(CONFIG, "heldout", "h-x", heldout=_heldout_args())
    with pytest.raises(ValueError, match="exactly one injected file"):
        job_spec_for(
            CONFIG, "heldout", "h-x", heldout=_heldout_args(), inject_files=[f"{world['secret']}:/tmp/x"]
        )
    spec = job_spec_for(CONFIG, "heldout", "h-x", heldout=_heldout_args(), inject_files=inject)
    argv = spec.create_args()
    assert argv[argv.index("--inject-file") + 1] == f"{world['secret']}:{SEED_MOUNT}"
    assert argv.count("--inject-file") == 1
    assert spec.args[spec.args.index("--seed-file") + 1] == SEED_MOUNT
    assert spec.env_secrets == {}  # no Token Factory key for the held-out job
    # the secret's value travels as a file only: never in the arguments, env or repr
    assert SECRET not in " ".join(argv) and SECRET not in repr(spec) and SECRET not in json.dumps(spec.env)
    assert "inject_files=1" in repr(spec) and str(world["secret"]) not in repr(spec)


def test_inject_file_values_are_checked(tmp_path: Path) -> None:
    good = tmp_path / "heldout_seed"
    good.write_text(SECRET + "\n")
    check_inject_file(f"{good}:{SEED_MOUNT}")
    with pytest.raises(ValueError, match="absolute"):
        check_inject_file(f"{good}:secret/heldout_seed")
    with pytest.raises(ValueError, match="absolute"):
        check_inject_file(f"{good}:/secret/../etc/x")
    with pytest.raises(ValueError, match="not a file"):
        check_inject_file(f"{tmp_path / 'missing'}:{SEED_MOUNT}")
    big = tmp_path / "big"
    big.write_bytes(b"0" * (INJECT_FILE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="larger"):
        check_inject_file(f"{big}:{SEED_MOUNT}")


def test_heldout_entry_args() -> None:
    args = entry_args("heldout", "heldout-x", heldout=_heldout_args("ab" * 32))
    assert args == [
        "-m", "rtt.cloud.entry", "heldout", "--out", "/out/runs/heldout-x",
        "--seed-file", "/secret/heldout_seed", "--expect-frozen-digest", "ab" * 32,
        "--suites", "procedural,real",
        "--frozen", "/out/runs/search-random/search", "/out/runs/search-nemotron/search",
    ]  # fmt: skip
    dev = HeldoutJobArgs(("s-a",), "d", suites=("real",), selection_run="heldout-y", workers=32, dev=True)
    assert dev.argv()[-5:] == [
        "--selection",
        "/out/runs/heldout-y/real/selection",
        "--workers",
        "32",
        "--dev",
    ]
    assert dev.argv()[dev.argv().index("--suites") + 1] == "real"
    with pytest.raises(ValueError, match="suites"):
        HeldoutJobArgs(("s-a",), "d", suites=("training",)).argv()
    only = HeldoutJobArgs(("s-a",), "d", suites=("real", "selection")).argv()
    assert only[only.index("--suites") + 1] == "selection,real"
    with pytest.raises(ValueError, match="reuses a selection"):
        HeldoutJobArgs(("s-a",), "d", suites=("selection",), selection_run="heldout-y").argv()
    with pytest.raises(ValueError, match="search runs"):
        HeldoutJobArgs((), "d").argv()
    with pytest.raises(ValueError, match="run ID"):
        HeldoutJobArgs(("../x",), "d").argv()
    with pytest.raises(ValueError, match="needs HeldoutJobArgs"):
        entry_args("heldout", "h")


# --------------------------------------------------------------------------- laptop side


def _ctx(fake: FakeNebius) -> Context:
    clock = FakeClock()
    return Context(
        CONFIG, lambda c: NebiusJobs(NebiusCLI("nebius", "rtt", fake), c, sleep=clock.sleep, clock=clock)
    )


def _run_args(world: dict[str, Any], *extra: str, seed: Path | None = None) -> Any:
    tmp = world["base"]
    argv = ["run", "heldout", "--target", "uks2-cpu", "--vcpus", "32", "--no-fallback",
            "--frozen-runs", "search-random,search-nemotron", "--seed-file", str(seed or world["secret"]),
            "--commitment-file", str(world["commitment_file"]), "--fetch-dir", str(world["fetch"]),
            "--ledger", str(tmp / "spend.jsonl"), "--no-capacity-check", "--no-fetch", "--poll-s", "1",
            "--cap-usd", "5", *extra]  # fmt: skip
    return build_parser().parse_args(argv)


def _creates(fake: FakeNebius) -> list[list[str]]:
    return [c for c in fake.calls if c[1:4] == ["ai", "job", "create"]]


def test_cloud_run_heldout_checks_then_injects_and_records_spend(world: dict[str, Any]) -> None:
    fake = FakeNebius({"cpu-d3": ["PROVISIONING", "RUNNING", "COMPLETED"]})
    assert cmd_run(_ctx(fake), _run_args(world, "--run-id", "heldout-test")) == 0
    (create,) = _creates(fake)
    assert create[create.index("--inject-file") + 1] == f"{world['secret'].resolve()}:{SEED_MOUNT}"
    job_args = create[create.index("--args") + 1]
    assert f"--expect-frozen-digest {world['digest']}" in job_args
    assert "/out/runs/search-random/search" in job_args and "--seed-file /secret/heldout_seed" in job_args
    assert create[create.index("--platform") + 1] == "cpu-d3"
    assert "--env-secret" not in create
    assert SECRET not in " ".join(c for call in fake.calls for c in call)
    spend = read_spend(world["base"] / "spend.jsonl")
    assert [e.name for e in spend] == ["rtt-heldout-test"] and spend[0].region == "uk-south2"
    record = json.loads((world["fetch"] / "heldout-test" / "job.json").read_text())
    assert record["target"] == "uks2-cpu" and record["fallback_from"] is None


def test_cloud_run_heldout_refuses_before_submitting(world: dict[str, Any], tmp_path: Path) -> None:
    fake = FakeNebius({"cpu-d3": ["COMPLETED"]})
    ctx = _ctx(fake)
    wrong = tmp_path / "heldout_seed.txt"
    wrong.write_text("ab" * 32 + "\n")
    with pytest.raises(ValueError, match="commitment"):
        cmd_run(ctx, _run_args(world, seed=wrong))
    args = _run_args(world)
    args.frozen_runs = "search-random,search-missing"
    with pytest.raises(ValueError, match="does not verify"):
        cmd_run(ctx, args)
    args = _run_args(world, "--suites", "procedural,training")
    with pytest.raises(ValueError, match="suites"):
        cmd_run(ctx, args)
    other = world["fetch"] / "search-random" / "job.json"
    saved = other.read_text()
    other.write_text(json.dumps({"region": "eu-north1"}))
    try:
        with pytest.raises(ValueError, match="must run there too"):
            cmd_run(ctx, _run_args(world))
    finally:
        other.write_text(saved)
    search = build_parser().parse_args(
        [
            "run",
            "search",
            "--arm",
            "random",
            "--runs",
            "1",
            "--budget",
            "40",
            "--seed-file",
            str(world["secret"]),
        ]
    )
    with pytest.raises(ValueError, match="heldout runs only"):
        cmd_run(ctx, search)
    assert fake.calls == []  # nothing reached the CLI


def test_cloud_run_heldout_dev_sends_a_throwaway_secret(world: dict[str, Any]) -> None:
    fake = FakeNebius({"cpu-d3": ["COMPLETED"]})
    args = _run_args(world, "--dev", "--dry-run")
    args.seed_file = None
    assert cmd_run(_ctx(fake), args) == 0
    (create,) = _creates(fake)
    local = create[create.index("--inject-file") + 1].rpartition(":")[0]
    assert Path(local).resolve() != world["secret"].resolve() and not Path(local).exists()  # removed after
    assert "--dry-run" in create and "--dev" in create[create.index("--args") + 1]


# --------------------------------------------------------------------------- in the container


@pytest.fixture
def no_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "simulate", fake_simulate)


def test_heldout_entry_writes_both_reports(world: dict[str, Any], tmp_path: Path, no_physics: None) -> None:
    out = tmp_path / "out"
    manifest, status = entry.run_heldout_job(
        out,
        world["secret"],
        list(world["runs"].values()),
        expect_frozen_digest=world["digest"],
        selection=world["selection"],
        workers=1,
        commitment_file=world["commitment_file"],
    )
    assert status == 0, manifest.get("error")
    assert manifest["kind"] == "heldout" and manifest["inputs"]["frozen_digest"] == world["digest"]
    assert manifest["inputs"]["commitment"] == COMMITMENT
    assert manifest["inputs"]["heldout_code"] == common.code_listing()
    assert set(manifest["result"]) == {"procedural", "real"}
    assert {"procedural/heldout_v2.json", "real/heldout_real_v2.json", "procedural/episodes.parquet"} <= set(
        manifest["files"]
    )
    h1 = json.loads((out / "procedural" / "heldout_v2.json").read_text())
    assert not h1["exploratory"] and h1["suite"]["episodes_per_design"] == 200
    check = verify_run(out)
    # the test's scan draw (a mirror, 3 objects) is not the frozen draw: flagged, the procedural is not
    assert check.problems == (
        f"real/heldout_real_v2.json: exploratory ({manifest['result']['real']['exploratory_reasons']})",
    )
    _no_secret_in(out)


def test_heldout_entry_draws_the_scans_itself(
    world: dict[str, Any], tmp_path: Path, no_physics: None
) -> None:
    out = tmp_path / "out"
    manifest, status = entry.run_heldout_job(
        out,
        world["secret"],
        list(world["runs"].values()),
        suites=("real",),
        expect_frozen_digest=world["digest"],
        workers=1,
        select_workers=1,
        commitment_file=world["commitment_file"],
        gso_cache=world["cache"],
        fetch=LocalMirror(world["mirror"]),
        real_config=world["config"],
    )
    assert status == 0, manifest.get("error")
    sel = manifest["result"]["selection"]
    assert sel["mode"] == "scored" and sel["objects"] == sel["requested"] == 3
    assert sel["digest"] == json.loads((world["selection"] / "selection.json").read_text())["digest"]
    assert (out / "real" / "selection" / real.HULLS).is_file() and "procedural" not in manifest["result"]
    _no_secret_in(out)


def test_heldout_entry_checks_the_picks_before_reading_the_secret(
    world: dict[str, Any], tmp_path: Path, no_physics: None
) -> None:
    opened: list[str] = []
    armed = [True]
    secret = os.path.realpath(world["secret"])

    def spy(event: str, args: tuple) -> None:
        path = args[0] if event == "open" and args else None
        if (
            armed[0]
            and isinstance(path, str | bytes | os.PathLike)
            and os.path.realpath(os.fsdecode(path)) == secret
        ):
            opened.append(event)

    sys.addaudithook(spy)
    try:
        manifest, status = entry.run_heldout_job(
            tmp_path / "out",
            world["secret"],
            list(world["runs"].values()),
            expect_frozen_digest="0" * 64,
            selection=world["selection"],
            commitment_file=world["commitment_file"],
        )
    finally:
        armed[0] = False
    assert status == 2 and "the secret was not read" in manifest["error"]
    assert opened == []
    assert not (tmp_path / "out" / "procedural").exists()


def test_heldout_entry_dev_is_exploratory_and_verifies(
    world: dict[str, Any], tmp_path: Path, no_physics: None
) -> None:
    throwaway = tmp_path / "secret" / "heldout_seed"
    throwaway.parent.mkdir()
    throwaway.write_text("ef" * 32 + "\n")
    out = tmp_path / "out"
    code = entry.main(
        ["heldout", "--out", str(out), "--seed-file", str(throwaway), "--frozen",
         *(str(p) for p in world["runs"].values()), "--suites", "procedural", "--dev", "--workers", "1"]
    )  # fmt: skip
    assert code == 0
    data = json.loads((out / "manifest.json").read_text())
    assert data["inputs"]["dev"] and data["inputs"]["commitment"] is None
    h1 = json.loads((out / "procedural" / "heldout_v2.json").read_text())
    assert h1["exploratory"] and h1["suite"]["n_conditions"] == 4
    assert verify_run(out).ok


def test_heldout_entry_selection_step_scores_nothing(world: dict[str, Any], tmp_path: Path) -> None:
    """``--suites selection``: the scan draw and its hull hashes, before any held-out score exists."""
    out = tmp_path / "out"
    manifest, status = entry.run_heldout_job(
        out,
        world["secret"],
        list(world["runs"].values()),
        suites=("selection",),
        expect_frozen_digest=world["digest"],
        select_workers=1,
        commitment_file=world["commitment_file"],
        gso_cache=world["cache"],
        fetch=LocalMirror(world["mirror"]),
        real_config=world["config"],
    )
    assert status == 0, manifest.get("error")
    assert set(manifest["result"]) == {"selection"}
    assert not (out / "procedural").exists() and not (out / "real" / real.REPORT).exists()
    assert not any(name.endswith(".parquet") for name in manifest["files"])
    assert verify_run(out).ok
    hulls = json.loads((out / "real" / "selection" / real.HULLS).read_text())
    assert hulls["selection_digest"] == manifest["result"]["selection"]["digest"]
    with pytest.raises(ValueError, match="selection was given"):
        entry.run_heldout_job(
            tmp_path / "again",
            world["secret"],
            [],
            suites=("selection",),
            selection=out / "real" / "selection",
        )
