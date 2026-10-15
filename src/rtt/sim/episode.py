"""One grasp episode on the CPU MuJoCo backend.

``run_episode`` builds the scene for a finger design and object, applies the
seed's jitter (object pose +-2 mm / +-3 deg and friction +-10 %, drawn from a
deterministic PCG64 stream), runs the controller, the physics checks and the
lift tracker, and returns an :class:`EpisodeResult`.

The episode ends as soon as the outcome is decided (success confirmed, a
physics flag raised, or the object provably below the success height after the
lift), or at ``max_time``. When recording, ``record_tail`` extra seconds are
simulated for the replay only; they never change the outcome or metrics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from rtt.design.types import FingerGeometry
from rtt.sim import objects as objs
from rtt.sim.checks import PhysicsChecks, contact_summary
from rtt.sim.controller import GraspController, Phase, grasp_fraction, plan_grasp
from rtt.sim.replay import Replay, ReplayRecorder
from rtt.sim.scene import build_model, finger_mass_properties
from rtt.sim.score import FailureMode, LiftTracker, classify
from rtt.sim.task import JitterConfig, TaskConfig, default_task

_PHASE_METRICS = (
    "rest_table_support",
    "rest_finger_force",
    "preclose_table_support",
    "preclose_finger_force",
    "squeeze_left",
    "squeeze_right",
    "press_down_left",
    "press_down_right",
)


def _ratio(period: float, base: float, name: str) -> int:
    """Integer number of ``base`` intervals in ``period`` (raises if not a whole multiple)."""
    n = round(period / base)
    if n < 1 or abs(n * base - period) > 1e-9:
        raise ValueError(f"{name}={period} must be a positive multiple of {base}")
    return n


@dataclass(frozen=True)
class Jitter:
    """Per-episode perturbation of the object's pose and friction."""

    dx: float
    dy: float
    dyaw: float
    mu_scale: float


def sample_jitter(seed: int, cfg: JitterConfig) -> Jitter:
    """Draw the jitter for ``seed`` (same seed, same jitter, on every platform)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    u = rng.uniform(-1.0, 1.0, size=4)
    return Jitter(
        dx=float(u[0] * cfg.pos),
        dy=float(u[1] * cfg.pos),
        dyaw=float(math.radians(u[2] * cfg.yaw_deg)),
        mu_scale=float(1.0 + u[3] * cfg.mu_frac),
    )


@dataclass
class EpisodeResult:
    """Outcome, diagnostics and (optionally) the recording of one episode."""

    success: bool
    failure_mode: FailureMode | None
    max_lift: float
    hold_time: float
    metrics: dict[str, float]
    flags: tuple[str, ...]
    seed: int
    controller_setting: int
    n_steps: int  # physics steps simulated, including any recording tail
    wall_time_s: float
    replay: Replay | None = None
    qpos: np.ndarray | None = field(default=None, repr=False)  # (n_frames, nq) for video rendering

    def summary(self) -> dict[str, Any]:
        """Deterministic, JSON-friendly view (excludes wall time and recordings; NaN -> None)."""
        return {
            "success": self.success,
            "failure_mode": self.failure_mode,
            "max_lift": self.max_lift,
            "hold_time": self.hold_time,
            "flags": list(self.flags),
            "seed": self.seed,
            "controller_setting": self.controller_setting,
            "n_steps": self.n_steps,
            "metrics": {k: None if math.isnan(v) else v for k, v in self.metrics.items()},
        }


def run_episode(
    finger_geom: FingerGeometry,
    object_spec: objs.ObjectSpec,
    placement: objs.Placement,
    controller_setting: int,
    seed: int,
    record: bool = True,
    task: TaskConfig | None = None,
) -> EpisodeResult:
    """Simulate one grasp attempt and score it."""
    wall0 = time.perf_counter()
    task = task or default_task()
    sc = task.score
    model, ids = build_model(finger_geom, object_spec, task)
    data = mujoco.MjData(model)

    jitter = sample_jitter(seed, task.jitter)
    mu = task.physics.friction * jitter.mu_scale
    model.geom_friction[ids.object_geom, 0] = mu

    plan = plan_grasp(finger_geom, object_spec, placement, controller_setting, task)
    controller = GraspController(plan)
    actual = objs.Placement(placement.x + jitter.dx, placement.y + jitter.dy, placement.yaw + jitter.dyaw)
    obj_pos, obj_quat = objs.rest_pose(object_spec, actual, task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = plan.palm_start
    data.qpos[ids.finger_qpos] = 0.0
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = obj_pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = obj_quat
    data.ctrl[:] = controller.command(0.0)
    mujoco.mj_forward(model, data)

    checks = PhysicsChecks(model=model, ids=ids, task=task)
    tracker = LiftTracker(cfg=sc)
    recorder = ReplayRecorder(model, ids, task.replay.fps) if record else None
    obj_mass = float(model.body_mass[ids.object_body])
    weight = obj_mass * float(np.linalg.norm(model.opt.gravity))
    min_squeeze = sc.grasp_min_squeeze_frac * task.gripper.grip_force / 2.0

    # Phase-entry metrics stay NaN when the episode ends before that phase.
    metrics: dict[str, float] = dict.fromkeys(_PHASE_METRICS, math.nan)

    def support_metrics(prefix: str) -> None:
        cs = contact_summary(model, data, ids)
        metrics[f"{prefix}_table_support"] = cs.object_env_normal / weight
        metrics[f"{prefix}_finger_force"] = sum(cs.object_finger_normal)

    steps_per_tick = _ratio(task.controller.control_period, model.opt.timestep, "control_period")
    ticks_per_check = _ratio(sc.check_period, task.controller.control_period, "check_period")
    max_ticks = round(sc.max_time / task.controller.control_period)
    tail_ticks = round(sc.record_tail / task.controller.control_period) if record else 0
    grasped = False
    stage = Phase.SETTLE  # last phase whose entry bookkeeping has run
    decided_tick: int | None = None
    tick = 0
    while tick < max_ticks:
        if recorder is not None:
            recorder.maybe_record(data)
        data.ctrl[:] = controller.command(data.time)
        mujoco.mj_step(model, data, nstep=steps_per_tick)
        tick += 1
        if decided_tick is not None:
            if tick >= decided_tick + tail_ticks:
                break
            continue
        if tick % ticks_per_check:
            continue

        t = data.time
        phase = controller.phase(t)
        if phase > stage:
            if stage < Phase.DESCEND <= phase:
                tracker.set_reference(float(data.xpos[ids.object_body][2]))
                support_metrics("rest")
            if stage < Phase.CLOSE <= phase:
                support_metrics("preclose")
            if stage < Phase.LIFT <= phase:
                cs = contact_summary(model, data, ids)
                metrics["squeeze_left"], metrics["squeeze_right"] = cs.squeeze
                metrics["press_down_left"], metrics["press_down_right"] = cs.press_down
                # Both fingers must pinch the object from the side: a finger that landed on
                # top of it (pressing down harder than it squeezes) is not holding it.
                grasped = all(
                    s >= min_squeeze and d <= sc.grasp_max_press_down_ratio * s
                    for s, d in zip(cs.squeeze, cs.press_down, strict=True)
                )
            stage = phase
        checks.update(data, t, phase)
        tracker.update(t, float(data.xpos[ids.object_body][2]))

        settled_after_lift = t >= plan.t_hold + sc.fail_settle_time and not tracker.above
        out_of_time = not tracker.above and t > sc.max_time - sc.success_hold
        if checks.events or tracker.success_at is not None or settled_after_lift or out_of_time:
            decided_tick = tick
            metrics["decided_at"] = t
            if tail_ticks == 0:
                break

    if decided_tick is None:
        metrics["decided_at"] = data.time
    success, mode = classify(checks.events, tracker, grasped)

    finger_mass = finger_mass_properties(finger_geom).mass
    metrics.update(
        mu=mu,
        jitter_dx=jitter.dx,
        jitter_dy=jitter.dy,
        jitter_dyaw_deg=math.degrees(jitter.dyaw),
        object_mass=obj_mass,
        finger_mass=finger_mass,
        grasp_fraction=grasp_fraction(controller_setting, task),
        tip_height=plan.tip_height,
        z0=tracker.z0 if tracker.z0 is not None else float("nan"),
        final_lift=tracker.lift,
        success_time=tracker.success_at if tracker.success_at is not None else float("nan"),
        grasped=float(grasped),
        gripper_env_force_max=checks.gripper_env_force_max,
        penetration_max=checks.penetration_max,
        object_speed_max=checks.object_speed_max,
        close_xy_drift_max=checks.close_xy_drift_max,
    )

    replay = qpos = None
    if recorder is not None:
        replay, qpos = recorder.finish(
            meta={
                "success": success,
                "failure_mode": mode,
                "seed": seed,
                "controller_setting": controller_setting,
                "object": {"kind": object_spec.kind, "size": list(object_spec.size)},
                "placement": {"x": placement.x, "y": placement.y, "yaw": placement.yaw},
                "task": task.name,
                "flags": [e.flag for e in checks.events],
            }
        )
    return EpisodeResult(
        success=success,
        failure_mode=mode,
        max_lift=tracker.max_lift,
        hold_time=tracker.hold_time,
        metrics=metrics,
        flags=tuple(f"{e.flag}@{e.t:.2f}s:{e.phase.name.lower()} {e.detail}".rstrip() for e in checks.events),
        seed=seed,
        controller_setting=controller_setting,
        n_steps=tick * steps_per_tick,
        wall_time_s=time.perf_counter() - wall0,
        replay=replay,
        qpos=qpos,
    )
