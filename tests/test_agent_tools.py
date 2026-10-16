"""Grasp Engineer tools: schemas, argument validation and every hard cap (fake physics, no network)."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import trimesh

from rtt.agent.backends import PartInfo
from rtt.agent.caps import AgentCaps, DailySpendBook, SpendGuard
from rtt.agent.fakes import SCRIPTED_DESIGNS, FakePartBackend
from rtt.agent.jobs import JobRecord, LocalJobRunner
from rtt.agent.proposer import ProposalBatch, ProposedDesign, SimulatedDesign, check_proposal
from rtt.agent.recorder import SessionRecorder
from rtt.agent.schemas import TOOL_NAMES, TOOL_SPECS, ArgsError, openai_tools, validate_args
from rtt.agent.store import AgentStore
from rtt.agent.tools import PASS_THRESHOLD, Toolbox
from rtt.design.schema import FingerDesign, GripperTask, load_entrants
from rtt.search.base import DesignScore

REPO = Path(__file__).resolve().parents[1]
ENTRANTS = load_entrants(REPO / "configs" / "entrants.yaml")
PART = "part-0123456789ab"


# --------------------------------------------------------------------------- schemas


def test_eight_tools_in_a_fixed_order() -> None:
    assert TOOL_NAMES == (
        "inspect_part",
        "propose_designs",
        "launch_sim_job",
        "poll_job",
        "fetch_results",
        "stress_test",
        "export_print_pack",
        "cost_ledger",
    )
    for tool in openai_tools():
        assert tool["type"] == "function"
        assert tool["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
def test_json_schema_matches_the_argument_model(spec: Any) -> None:
    props = set(spec.parameters["properties"])
    fields = set(spec.args_model.model_fields)
    assert props == fields
    required_by_model = {n for n, f in spec.args_model.model_fields.items() if f.is_required()}
    assert required_by_model <= set(spec.parameters["required"]) <= fields


@pytest.mark.parametrize(
    ("name", "args", "problem"),
    [
        ("inspect_part", {"part_id": "../../etc/passwd"}, "part_id"),
        ("inspect_part", {"part_id": PART, "extra": 1}, "extra"),
        ("inspect_part", {}, "part_id"),
        ("propose_designs", {"part_id": PART, "n": 9}, "n"),
        ("propose_designs", {"part_id": PART, "n": 0}, "n"),
        ("launch_sim_job", {"part_id": PART, "designs": [], "target": "local"}, "designs"),
        ("launch_sim_job", {"part_id": PART, "designs": ["abc"], "target": "local"}, "designs"),
        ("launch_sim_job", {"part_id": PART, "designs": ["0" * 12, "0" * 12], "target": "local"}, "unique"),
        ("launch_sim_job", {"part_id": PART, "designs": ["0" * 12], "target": "aws"}, "target"),
        ("poll_job", {"job_id": "local-xyz"}, "job_id"),
        ("export_print_pack", {"run_id": "run-; rm -rf /"}, "run_id"),
        ("cost_ledger", {"x": 1}, "x"),
        ("no_such_tool", {}, "unknown tool"),
    ],
)
def test_validate_args_rejects(name: str, args: dict[str, Any], problem: str) -> None:
    with pytest.raises(ArgsError) as info:
        validate_args(name, args)
    assert problem in str(info.value)


def test_validate_args_accepts_good_arguments() -> None:
    args = validate_args("launch_sim_job", {"part_id": PART, "designs": ["0123456789ab"], "target": "nebius"})
    assert args.model_dump() == {"part_id": PART, "designs": ["0123456789ab"], "target": "nebius"}
    assert validate_args("poll_job", {"job_id": "aijob-e00abc123def"}).job_id == "aijob-e00abc123def"
    with pytest.raises(ArgsError, match="JSON object"):
        validate_args("poll_job", "not a dict")


# --------------------------------------------------------------------------- fixtures


class StaticProposer:
    """Hands out SCRIPTED_DESIGNS (validated like the real proposer); no model calls."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.histories: list[list[SimulatedDesign]] = []
        self.client = None

    def propose(
        self,
        part: PartInfo,
        n: int,
        *,
        gripper_task: GripperTask,
        history: Sequence[SimulatedDesign],
        known: Sequence[FingerDesign],
    ) -> ProposalBatch:
        self.calls.append(n)
        self.histories.append(list(history))
        accepted, rejected, reference = [], [], list(known)
        for i in range(n):
            raw = SCRIPTED_DESIGNS[(len(self.calls) - 1 + i) % len(SCRIPTED_DESIGNS)]
            design, reasons = check_proposal(raw, f"idea {i}", gripper_task, reference)
            if design is None:
                rejected.append({"slot": f"design_{i + 1}", "reasons": reasons})
            else:
                accepted.append(ProposedDesign(design, f"idea {i}"))
                reference.append(design)
        return ProposalBatch(tuple(accepted), tuple(rejected), "static")


class FakeNebiusRunner:
    """A paid runner: fixed estimate, jobs complete on the first poll."""

    target = "nebius"

    def __init__(self, backend: FakePartBackend, estimate: float = 0.05) -> None:
        self.backend = backend
        self.estimate = estimate
        self.records: dict[str, JobRecord] = {}
        self.designs: dict[str, list[FingerDesign]] = {}
        self.cancelled: list[str] = []

    def estimate_usd(self, n_designs: int) -> float:
        return self.estimate

    def launch(self, part_id: str, designs: Sequence[FingerDesign]) -> JobRecord:
        job_id = f"aijob-e00fake{len(self.records):06d}"
        rec = JobRecord(job_id, "nebius", part_id, tuple(d.design_id() for d in designs), "QUEUED", 0.0)
        self.records[job_id] = rec
        self.designs[job_id] = list(designs)
        return rec

    def status(self, job_id: str) -> JobRecord:
        return self.records[job_id]

    def wait(self, job_id: str, timeout_s: float) -> JobRecord:
        rec = self.records[job_id]
        if not rec.terminal:
            rec = JobRecord(**{**rec.__dict__, "state": "COMPLETED", "finished_at": 100.0, "est_usd": 0.02})
            self.records[job_id] = rec
        return rec

    def results(self, job_id: str) -> list[DesignScore]:
        rec = self.records[job_id]
        return self.backend.evaluate(rec.part_id, self.designs[job_id])

    def cancel(self, job_id: str) -> JobRecord:
        self.cancelled.append(job_id)
        rec = JobRecord(
            **{**self.records[job_id].__dict__, "state": "CANCELLED", "finished_at": 50.0, "est_usd": 0.01}
        )
        self.records[job_id] = rec
        return rec


def box_stl(path: Path, extents: tuple[float, float, float] = (40.0, 30.0, 60.0)) -> Path:
    trimesh.creation.box(extents=extents).export(path)
    return path


@pytest.fixture
def backend(tmp_path: Path) -> FakePartBackend:
    fake = FakePartBackend(tmp_path / "parts")
    fake.intake(box_stl(tmp_path / "box.stl"), part_id=PART, name="box")
    return fake


def make_toolbox(
    tmp_path: Path,
    backend: FakePartBackend,
    *,
    caps: AgentCaps | None = None,
    unlocked: bool = True,
    nebius: FakeNebiusRunner | None = None,
    session_usd: float = 0.5,
) -> Toolbox:
    caps = caps or AgentCaps(session_usd=session_usd, poll_wait_s=5.0)
    runners: dict[str, Any] = {"local": LocalJobRunner(backend.evaluate)}
    if nebius is not None:
        runners["nebius"] = nebius
    guard = SpendGuard(caps.session_usd, DailySpendBook(None, caps.day_usd), "sess-000000000001")
    return Toolbox(
        session_id="sess-000000000001",
        part_id=PART,
        backend=backend,
        proposer=StaticProposer(),
        runners=runners,
        store=AgentStore(tmp_path / "agent"),
        caps=caps,
        guard=guard,
        unlocked=unlocked,
        recorder=SessionRecorder("sess-000000000001", None),
        baselines=ENTRANTS,
    )


def ok(outcome: Any) -> dict[str, Any]:
    assert outcome.ok, (outcome.code, outcome.error)
    return outcome.result


# --------------------------------------------------------------------------- tools


def test_full_tool_chain_and_code_verdicts(tmp_path: Path, backend: FakePartBackend) -> None:
    box = make_toolbox(tmp_path, backend)
    part = ok(box.call("inspect_part", {"part_id": PART}))
    assert part["part"]["part_id"] == PART and "backend" not in part["part"]
    baseline = next(b["design_id"] for b in part["baselines"] if b["valid_for_part"])
    proposed = ok(box.call("propose_designs", {"part_id": PART, "n": 3}))
    ids = [d["design_id"] for d in proposed["designs"]]
    assert len(ids) == 3
    job = ok(box.call("launch_sim_job", {"part_id": PART, "designs": [*ids, baseline], "target": "local"}))
    polled = ok(box.call("poll_job", {"job_id": job["job_id"]}))
    assert polled["state"] == "COMPLETED"
    results = ok(box.call("fetch_results", {"job_id": job["job_id"]}))
    for row in results["designs"]:
        assert row["verdict"] == ("pass" if row["success_pct"] >= 100 * PASS_THRESHOLD else "fail")
    saved = box.store.load_run(results["run_id"])
    assert saved["label"].startswith("Product mode") and saved["designs"][0]["episodes"]
    best = results["best_design_id"]
    cert = ok(box.call("stress_test", {"design_id": best, "part_id": PART}))
    assert cert["verdict"] in ("pass", "fail") and 0 <= cert["score_pct"] <= 100
    pack = ok(box.call("export_print_pack", {"run_id": results["run_id"]}))
    assert pack["design_id"] == best and pack["certificate_included"]
    with zipfile.ZipFile(box.store.pack_zip(pack["pack_id"])) as zf:
        names = zf.namelist()
    assert any(n.endswith("finger_left.stl") for n in names)
    ledger = ok(box.call("cost_ledger", {}))
    assert ledger["tool_calls_used"] == 8 and ledger["tool_calls_left"] == 4
    verdicts = {v["design_id"]: v for v in box.verdicts()}
    assert verdicts[best]["certificate_verdict"] == cert["verdict"]
    # a second proposal sees the session's results
    ok(box.call("propose_designs", {"part_id": PART, "n": 1}))
    assert {h.label for h in box.proposer.histories[-1]} >= {best}


def test_tool_budget_is_hard(tmp_path: Path, backend: FakePartBackend) -> None:
    box = make_toolbox(tmp_path, backend, caps=AgentCaps(max_tool_calls=3))
    assert box.call("inspect_part", {"part": PART}).code == "invalid_args"  # invalid calls count
    ok(box.call("cost_ledger", {}))
    ok(box.call("inspect_part", {"part_id": PART}))
    refused = box.call("cost_ledger", {})
    assert refused.refused and refused.code == "tool_budget_exhausted" and refused.calls_left == 0
    assert box.calls_used == 3  # refusals are not counted


def test_other_parts_are_off_limits(tmp_path: Path, backend: FakePartBackend) -> None:
    box = make_toolbox(tmp_path, backend)
    out = box.call("inspect_part", {"part_id": "part-ffffffffffff"})
    assert out.code == "invalid_args" and "works on" in out.error


def test_spending_tools_need_the_passcode(tmp_path: Path, backend: FakePartBackend) -> None:
    nebius = FakeNebiusRunner(backend)
    box = make_toolbox(tmp_path, backend, unlocked=False, nebius=nebius)
    out = box.call("propose_designs", {"part_id": PART, "n": 2})
    assert out.refused and out.code == "passcode_required"
    assert box.proposer.calls == []
    stock = ENTRANTS[0].design_id()
    paid = box.call("launch_sim_job", {"part_id": PART, "designs": [stock], "target": "nebius"})
    assert paid.code == "passcode_required" and not nebius.records
    free = box.call("launch_sim_job", {"part_id": PART, "designs": [stock], "target": "local"})
    assert free.ok


def test_concurrency_cap(tmp_path: Path, backend: FakePartBackend) -> None:
    backend.gate.clear()  # jobs block until released
    box = make_toolbox(tmp_path, backend, caps=AgentCaps(poll_wait_s=0.0))
    ids = [e.design_id() for e in ENTRANTS[:3]]
    jobs = [
        ok(box.call("launch_sim_job", {"part_id": PART, "designs": [i], "target": "local"})) for i in ids[:2]
    ]
    third = box.call("launch_sim_job", {"part_id": PART, "designs": [ids[2]], "target": "local"})
    assert third.refused and third.code == "too_many_jobs"
    early = box.call("fetch_results", {"job_id": jobs[0]["job_id"]})
    assert early.code == "error" and "poll_job" in early.error
    backend.gate.set()
    box.caps = AgentCaps(poll_wait_s=5.0)
    assert ok(box.call("poll_job", {"job_id": jobs[0]["job_id"]}))["state"] == "COMPLETED"
    assert ok(box.call("poll_job", {"job_id": jobs[1]["job_id"]}))["state"] == "COMPLETED"
    ok(box.call("launch_sim_job", {"part_id": PART, "designs": [ids[2]], "target": "local"}))


def test_paid_jobs_reserve_then_charge(tmp_path: Path, backend: FakePartBackend) -> None:
    nebius = FakeNebiusRunner(backend, estimate=0.05)
    box = make_toolbox(tmp_path, backend, nebius=nebius)
    stock = ENTRANTS[0].design_id()
    job = ok(box.call("launch_sim_job", {"part_id": PART, "designs": [stock], "target": "nebius"}))
    assert box.guard.reserved_usd == pytest.approx(0.05)
    ok(box.call("poll_job", {"job_id": job["job_id"]}))
    assert box.guard.reserved_usd == 0.0
    assert box.guard.snapshot()["by_kind_usd"] == {"compute": 0.02}
    results = ok(box.call("fetch_results", {"job_id": job["job_id"]}))
    assert results["target"] == "nebius" and results["job_id"].startswith("aijob-")


def test_paid_job_refused_over_the_session_cap(tmp_path: Path, backend: FakePartBackend) -> None:
    nebius = FakeNebiusRunner(backend, estimate=0.40)
    box = make_toolbox(tmp_path, backend, nebius=nebius, session_usd=0.30)
    out = box.call(
        "launch_sim_job", {"part_id": PART, "designs": [ENTRANTS[0].design_id()], "target": "nebius"}
    )
    assert out.refused and out.code == "spend_cap" and not nebius.records


def test_close_cancels_jobs_in_flight(tmp_path: Path, backend: FakePartBackend) -> None:
    nebius = FakeNebiusRunner(backend, estimate=0.05)
    box = make_toolbox(tmp_path, backend, nebius=nebius)
    job = ok(
        box.call(
            "launch_sim_job", {"part_id": PART, "designs": [ENTRANTS[0].design_id()], "target": "nebius"}
        )
    )
    cancelled = box.close()
    assert [c["job_id"] for c in cancelled] == [job["job_id"]] == nebius.cancelled
    assert box.guard.reserved_usd == 0.0
    assert box.guard.snapshot()["by_kind_usd"] == {"compute": 0.01}


def test_unknown_ids_are_not_found(tmp_path: Path, backend: FakePartBackend) -> None:
    box = make_toolbox(tmp_path, backend)
    assert box.call("poll_job", {"job_id": "local-000000000000"}).code == "not_found"
    assert box.call("export_print_pack", {"run_id": "run-000000000000"}).code == "not_found"
    assert box.call("stress_test", {"design_id": "0" * 12, "part_id": PART}).code == "not_found"
    out = box.call("launch_sim_job", {"part_id": PART, "designs": ["0" * 12], "target": "local"})
    assert out.code == "not_found"
    assert box.call("launch_sim_job", {"part_id": PART, "designs": ["0" * 12], "target": "nebius"}).code == (
        "invalid_args"
    )


def test_designs_must_fit_the_part(tmp_path: Path) -> None:
    fake = FakePartBackend(tmp_path / "parts")
    fake.intake(box_stl(tmp_path / "wide.stl", (90.0, 75.0, 40.0)), part_id=PART, name="wide")
    box = make_toolbox(tmp_path, fake)
    info = ok(box.call("inspect_part", {"part_id": PART}))
    hooked = next(b for b in info["baselines"] if b["name"] == "hooked")
    assert hooked["valid_for_part"] is False and hooked["why_not"]
    out = box.call("launch_sim_job", {"part_id": PART, "designs": [hooked["design_id"]], "target": "local"})
    assert out.code == "invalid_args" and "does not fit" in out.error


def test_tool_messages_are_compact_json(tmp_path: Path, backend: FakePartBackend) -> None:
    box = make_toolbox(tmp_path, backend)
    body = json.loads(box.call("cost_ledger", {}).for_model())
    assert body["ok"] is True and body["tool_calls_left"] == 11
    body = json.loads(box.call("poll_job", {"job_id": "nope"}).for_model())
    assert body["ok"] is False and body["code"] == "invalid_args"


# --------------------------------------------------------------------------- verifier additions


def test_success_and_episode_count_describe_the_same_trials(tmp_path: Path, backend: FakePartBackend) -> None:
    # success_pct is the rate at the best grasp height, so "episodes" must count that height only
    box = make_toolbox(tmp_path, backend)
    ids = [e.design_id() for e in ENTRANTS[:2]]
    job = ok(box.call("launch_sim_job", {"part_id": PART, "designs": ids, "target": "local"}))
    ok(box.call("poll_job", {"job_id": job["job_id"]}))
    results = ok(box.call("fetch_results", {"job_id": job["job_id"]}))
    saved = {d["design_id"]: d for d in box.store.load_run(results["run_id"])["designs"]}
    for row in results["designs"]:
        full = saved[row["design_id"]]
        at_best = [e for e in full["episodes"] if e["controller_setting"] == full["best_k"]]
        assert row["episodes"] == len(at_best) == backend.seeds
        assert row["episodes_all_heights"] == full["n_episodes"] == backend.seeds * backend.settings
        assert row["success_pct"] == round(100.0 * sum(e["success"] for e in at_best) / len(at_best), 1)


def test_tool_errors_do_not_leak_local_paths(tmp_path: Path, backend: FakePartBackend) -> None:
    secret_dir = tmp_path / "private" / "folder"

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory", str(secret_dir / "mesh_0.stl"))

    backend.certify = broken  # type: ignore[method-assign]
    box = make_toolbox(tmp_path, backend)
    part = ok(box.call("inspect_part", {"part_id": PART}))
    baseline = next(b["design_id"] for b in part["baselines"] if b["valid_for_part"])
    out = box.call("stress_test", {"design_id": baseline, "part_id": PART})
    assert out.code == "error" and "mesh_0.stl" in out.error
    assert str(tmp_path) not in out.error and "private" not in out.error


def test_export_reports_whether_the_design_was_stress_tested(
    tmp_path: Path, backend: FakePartBackend
) -> None:
    box = make_toolbox(tmp_path, backend)
    ids = [e.design_id() for e in ENTRANTS[:1]]
    job = ok(box.call("launch_sim_job", {"part_id": PART, "designs": ids, "target": "local"}))
    ok(box.call("poll_job", {"job_id": job["job_id"]}))
    results = ok(box.call("fetch_results", {"job_id": job["job_id"]}))
    pack = ok(box.call("export_print_pack", {"run_id": results["run_id"]}))
    assert pack["stress_tested"] is False and pack["certificate_verdict"] is None
    assert pack["certificate_included"] is any(f["name"] == "certificate.json" for f in pack["files"])
