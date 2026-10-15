"""Grasp simulation: MjSpec scene, controller, scoring, replays and the CPU executor.

Typical use::

    from rtt.sim import ObjectSpec, Placement, run_episode

    result = run_episode(finger, ObjectSpec("box", (0.05, 0.05, 0.08)), Placement(), 1, seed=0)

Evaluating designs on a suite (one record per episode, Parquet-ready)::

    from rtt.sim import evaluate_designs, rank_designs, training_suite

    records = evaluate_designs(designs, training_suite(), workers=16)
    best = rank_designs(records)[0]

Mesh objects (:mod:`rtt.sim.mesh`), the held-out generator (:mod:`rtt.sim.conditions`)
and the robustness certificate (:mod:`rtt.sim.certify`) import trimesh or extra
machinery and are imported from their modules.
"""

from rtt.sim.conditions import Condition, Suite, heldout_suite, training_suite
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.evaluate import (
    DesignScore,
    EpisodeRecord,
    evaluate_design,
    evaluate_designs,
    rank_designs,
    score_designs,
    write_parquet,
)
from rtt.sim.exec_cpu import BatchResult, CpuExecutor, EpisodeJob, run_jobs
from rtt.sim.objects import AnyObject, MeshObject, ObjectSpec, Placement
from rtt.sim.perturb import Push, Shake
from rtt.sim.score import FAILURE_MODES, FailureMode
from rtt.sim.task import TaskConfig, default_task, load_task

__all__ = [
    "FAILURE_MODES",
    "AnyObject",
    "BatchResult",
    "Condition",
    "CpuExecutor",
    "DesignScore",
    "EpisodeJob",
    "EpisodeRecord",
    "EpisodeResult",
    "FailureMode",
    "MeshObject",
    "ObjectSpec",
    "Placement",
    "Push",
    "Shake",
    "Suite",
    "TaskConfig",
    "default_task",
    "evaluate_design",
    "evaluate_designs",
    "heldout_suite",
    "load_task",
    "rank_designs",
    "run_episode",
    "run_jobs",
    "score_designs",
    "training_suite",
    "write_parquet",
]
