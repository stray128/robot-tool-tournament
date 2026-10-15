"""Grasp simulation: MjSpec scene, controller, scoring, replays and the CPU executor.

Typical use::

    from rtt.sim import ObjectSpec, Placement, run_episode

    result = run_episode(finger, ObjectSpec("box", (0.05, 0.05, 0.08)), Placement(), 1, seed=0)
"""

from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.exec_cpu import BatchResult, EpisodeJob, run_jobs
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.score import FAILURE_MODES, FailureMode
from rtt.sim.task import TaskConfig, default_task, load_task

__all__ = [
    "FAILURE_MODES",
    "BatchResult",
    "EpisodeJob",
    "EpisodeResult",
    "FailureMode",
    "ObjectSpec",
    "Placement",
    "TaskConfig",
    "default_task",
    "load_task",
    "run_episode",
    "run_jobs",
]
