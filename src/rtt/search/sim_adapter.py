"""Evaluator adapter for the MuJoCo simulator's batch API.

:class:`SimEvaluator` implements the search :class:`~rtt.search.base.Evaluator`
protocol on top of :func:`rtt.sim.evaluate.evaluate_designs`: every design is run on
every condition of a training :class:`~rtt.sim.conditions.Suite` (default
``configs/suites/train_v1.yaml``), at every controller setting and with each
condition's own jitter seeds (common random numbers), on the chosen executor
(``"cpu"``, ``"mjwarp"``, a reusable :class:`~rtt.sim.exec_cpu.CpuExecutor` or any
callable with the ``run_jobs(jobs, workers)`` contract). The simulator's episode
records are aggregated with :meth:`~rtt.search.base.DesignScore.from_episodes`, the
one scoring rule every arm uses.

The adapter only ever receives a training suite. The held-out suite is built by the
held-out job from the vault; nothing here can reach it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rtt.design.schema import FingerDesign, GripperTask
from rtt.search.base import ConditionInfo, DesignScore, EpisodeOutcome, EvaluatorInfo

__all__ = ["SimEvaluator", "describe_condition"]


def _mm(value: float) -> str:
    return f"{1000.0 * value:.1f}".rstrip("0").rstrip(".")


def describe_condition(condition: Any) -> str:
    """Prompt label for a :class:`~rtt.sim.conditions.Condition` (millimetres and degrees)."""
    obj, p = condition.obj, condition.placement
    size = f"{condition.size_label} " if getattr(condition, "size_label", "") else ""
    if obj.kind == "box":
        shape = "box " + " x ".join(_mm(s) for s in obj.size) + " mm (length x width across the jaw x height)"
    elif obj.kind == "cylinder":
        shape = f"upright cylinder, radius {_mm(obj.size[0])} mm, height {_mm(obj.size[1])} mm"
    elif obj.kind == "rod":
        shape = f"rod lying flat, radius {_mm(obj.size[0])} mm, length {_mm(obj.size[1])} mm"
    else:
        shape = f"{obj.kind} object {getattr(obj, 'name', '')}".rstrip()
    if p.x == 0.0 and p.y == 0.0 and p.yaw == 0.0:
        where = "centred"
    else:
        where = f"at ({_mm(p.x)}, {_mm(p.y)}) mm, yaw {math.degrees(p.yaw):.0f} deg"
    err = (getattr(p, "err_x", 0.0), getattr(p, "err_y", 0.0), getattr(p, "err_yaw", 0.0))
    if any(err):
        where += (
            f", sitting ({_mm(err[0])}, {_mm(err[1])}) mm and {math.degrees(err[2]):.0f} deg away from where "
            "the controller aims"
        )
    return f"{size}{shape}, {where}"


class SimEvaluator:
    """Training-suite evaluator backed by :func:`rtt.sim.evaluate.evaluate_designs`.

    Args:
        suite: Training suite (default :func:`rtt.sim.conditions.training_suite`).
        controller_settings: Indices into ``task.controller.grasp_height_fracs``
            (default: all). Scores index settings in this order.
        task: Simulator task (default ``configs/task_v1.yaml``).
        executor: ``"cpu"``, ``"mjwarp"``, a ``CpuExecutor`` or a ``run_jobs`` callable.
        workers: Worker processes for the CPU executor (default: all CPUs).
        gripper_task: Validator constants (default: ``suite.gripper_task(task)``).
    """

    def __init__(
        self,
        suite: Any | None = None,
        *,
        controller_settings: Sequence[int] | None = None,
        task: Any | None = None,
        executor: Any = "cpu",
        workers: int | None = None,
        gripper_task: GripperTask | None = None,
    ) -> None:
        from rtt.sim import evaluate as sim_evaluate
        from rtt.sim.conditions import training_suite
        from rtt.sim.task import default_task

        self._sim = sim_evaluate
        self.suite = suite if suite is not None else training_suite()
        self.task = task or default_task()
        n_settings = len(self.task.controller.grasp_height_fracs)
        self.settings = (
            tuple(controller_settings) if controller_settings is not None else tuple(range(n_settings))
        )
        if not self.settings or any(not 0 <= k < n_settings for k in self.settings):
            raise ValueError(f"controller settings must index the task's {n_settings} grasp heights")
        seed_counts = {len(c.seeds) for c in self.suite}
        if len(seed_counts) != 1:
            raise ValueError("every training condition must have the same number of jitter seeds")
        self.seeds_per_condition = seed_counts.pop()
        self.executor = executor
        self.executor_name = (
            executor if isinstance(executor, str) else "cpu" if hasattr(executor, "run") else "custom"
        )
        self.workers = workers
        self.gripper_task = gripper_task or self.suite.gripper_task(self.task)
        self.last_records: list[Any] = []

    def describe(self) -> EvaluatorInfo:
        from rtt.sim.objects import grip_width
        from rtt.sim.task import task_digest

        fracs = self.task.controller.grasp_height_fracs
        jitter = self.task.jitter
        return EvaluatorInfo(
            executor=self.executor_name,
            conditions=tuple(
                ConditionInfo(c.condition_id, c.family, describe_condition(c), grip_width(c.obj) * 1000.0)
                for c in self.suite
            ),
            controller_settings=tuple(fracs[k] for k in self.settings),
            seeds_per_condition=self.seeds_per_condition,
            gripper_task=self.gripper_task,
            task_hash=task_digest(self.task),
            suite_hash=self.suite.digest(),
            grip_force_n=self.task.gripper.grip_force,
            success_lift_m=self.task.score.success_lift,
            success_hold_s=self.task.score.success_hold,
            max_time_s=self.task.score.max_time,
            friction=self.task.physics.friction,
            object_density_kg_m3=self.task.physics.object_density,
            jitter_pos_mm=jitter.pos * 1000.0,
            jitter_yaw_deg=jitter.yaw_deg,
            jitter_mu_frac=jitter.mu_frac,
            palm_clearance_m=self.task.controller.palm_clearance,
        )

    def evaluate(self, designs: Sequence[FingerDesign]) -> list[DesignScore]:
        if not designs:
            return []
        unique = list({d.design_id(): d for d in designs}.values())  # identical geometry: simulate once
        records = self._sim.evaluate_designs(
            unique,
            self.suite,
            controller_settings=self.settings,
            executor=self.executor,
            workers=self.workers,
            task=self.task,
        )
        self.last_records = list(records)
        index = {k: i for i, k in enumerate(self.settings)}
        grouped: dict[str, list[Any]] = {}
        for record in records:
            grouped.setdefault(record.design_id, []).append(record)
        scores = []
        for design in designs:
            design_id = design.design_id()
            recs = grouped.get(design_id)
            if not recs:
                raise RuntimeError(f"simulator returned no episodes for design {design_id}")
            episodes = [
                EpisodeOutcome(
                    condition_id=r.condition_id,
                    family=r.family,
                    controller_setting=index[r.k],
                    seed=int(r.seed),
                    success=bool(r.success),
                    failure_mode=r.failure_mode,
                    max_lift_m=float(r.lift),
                    hold_frac=float(r.hold_frac),
                )
                for r in recs
            ]
            scores.append(
                DesignScore.from_episodes(design_id, recs[0].finger_mass_g, episodes, len(self.settings))
            )
        return scores
