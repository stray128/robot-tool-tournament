"""Record web replays of scored episodes.

Scoring runs with recording off (it is the hot path). To show an episode, it is run
again with the same finger, condition, controller setting and jitter seed and
recording on; the simulator is deterministic on one machine, so the recorded episode
is the scored one. :func:`record_replays` checks that: a recorded outcome that differs
from the expected one (when given) raises instead of publishing a misleading replay.

Each replay is written as ``<stem>.bin`` + ``<stem>.json``
(:func:`rtt.sim.replay.write_replay`) with extra ``meta``: the design id and
parameters, the condition id and description, and whatever the caller adds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtt.design.generate import generate_finger
from rtt.design.schema import FingerDesign
from rtt.sim.conditions import Condition
from rtt.sim.episode import EpisodeResult
from rtt.sim.exec_cpu import CpuExecutor, EpisodeJob
from rtt.sim.replay import write_replay
from rtt.sim.task import TaskConfig, default_task

__all__ = ["RecordedReplay", "ReplayMismatchError", "ReplayRequest", "record_replays"]


class ReplayMismatchError(RuntimeError):
    """A recorded episode did not reproduce the outcome it was recorded to show."""


@dataclass(frozen=True)
class ReplayRequest:
    """One episode to record.

    Args:
        stem: File stem (no extension) relative to the output directory.
        design: The finger design.
        condition: Object, placement and jitter seeds.
        controller_setting: Index into the task's grasp heights.
        seed: Jitter seed (default: the condition's first seed, as scored).
        expect_success: The scored outcome; a recorded episode that differs is an error.
        expect_failure_mode: The scored failure mode (``None`` for a success).
        meta: Extra replay-header metadata.
    """

    stem: str
    design: FingerDesign
    condition: Condition
    controller_setting: int
    seed: int | None = None
    expect_success: bool | None = None
    expect_failure_mode: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def episode_seed(self) -> int:
        return int(self.seed if self.seed is not None else self.condition.seeds[0])


@dataclass(frozen=True)
class RecordedReplay:
    """A written replay and the outcome of the recorded episode."""

    request: ReplayRequest
    bin_path: Path
    json_path: Path
    success: bool
    failure_mode: str | None
    max_lift_m: float
    hold_time_s: float
    flags: tuple[str, ...]
    n_frames: int
    n_steps: int
    wall_time_s: float


def _meta(req: ReplayRequest, task: TaskConfig) -> dict[str, Any]:
    fracs = task.controller.grasp_height_fracs
    return {
        "design_id": req.design.design_id(),
        "design": req.design.model_dump(mode="json", exclude={"rationale"}),
        "condition_id": req.condition.condition_id,
        "condition": req.condition.describe(),
        "family": req.condition.family,
        "grasp_height_frac": fracs[req.controller_setting],
        **dict(req.meta),
    }


def _check(req: ReplayRequest, res: EpisodeResult) -> None:
    if req.expect_success is None:
        return
    if res.success != req.expect_success or (not res.success and res.failure_mode != req.expect_failure_mode):
        got = "success" if res.success else res.failure_mode
        want = "success" if req.expect_success else req.expect_failure_mode
        raise ReplayMismatchError(
            f"{req.stem}: recorded episode gave {got}, the scored episode gave {want} "
            f"(design {req.design.design_id()}, {req.condition.condition_id}, k={req.controller_setting}, "
            f"seed {req.episode_seed}); not publishing a replay that disagrees with the score"
        )


def record_replays(
    requests: Sequence[ReplayRequest],
    out_dir: str | Path,
    *,
    task: TaskConfig | None = None,
    executor: CpuExecutor | None = None,
    workers: int | None = None,
) -> list[RecordedReplay]:
    """Run every request with recording on and write its replay under ``out_dir``.

    Episodes run on ``executor`` (or a temporary pool of ``workers``); replays are
    written only after every episode matched its expected outcome.
    """
    task = task or default_task()
    out = Path(out_dir)
    stems = [r.stem for r in requests]
    if len(set(stems)) != len(stems):
        raise ValueError("replay stems must be unique")
    jobs = [
        EpisodeJob(
            generate_finger(r.design),
            r.condition.obj,
            r.condition.placement,
            r.controller_setting,
            r.episode_seed,
            record=True,
            task=task,
        )
        for r in requests
    ]
    own = executor is None
    ex = executor or CpuExecutor(workers)
    try:
        results = ex.run(jobs).results if jobs else []
    finally:
        if own:
            ex.close()
    for req, res in zip(requests, results, strict=True):
        _check(req, res)
    recorded = []
    for req, res in zip(requests, results, strict=True):
        if res.replay is None:
            raise RuntimeError(f"{req.stem}: the episode was not recorded")
        res.replay.header["meta"].update(_meta(req, task))
        bin_path, json_path = write_replay(res.replay, out / req.stem)
        recorded.append(
            RecordedReplay(
                request=req,
                bin_path=bin_path,
                json_path=json_path,
                success=res.success,
                failure_mode=res.failure_mode,
                max_lift_m=float(res.max_lift),
                hold_time_s=float(res.hold_time),
                flags=tuple(f.split("@", 1)[0] for f in res.flags),
                n_frames=res.replay.n_frames,
                n_steps=res.n_steps,
                wall_time_s=float(res.wall_time_s),
            )
        )
    return recorded
