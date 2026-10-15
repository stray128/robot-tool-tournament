"""One grasp episode on the CPU MuJoCo backend.

``run_episode`` builds the scene for a finger design and object, places the
object at the placement's true pose (its pose error plus the seed's jitter:
object pose +-2 mm / +-3 deg and friction +-10 %, drawn from a deterministic
PCG64 stream), runs the controller, the physics checks and the lift tracker,
and returns an :class:`EpisodeResult`.

An optional :class:`~rtt.sim.perturb.Shake` or :class:`~rtt.sim.perturb.Push`
disturbs the hold; the episode then only succeeds if the object is still lifted
``post_perturb_hold`` seconds after the disturbance ends.

The episode ends as soon as the outcome is decided (success confirmed, a
physics flag raised, or the object provably below the success height after the
lift), or at ``max_time``. When recording, ``record_tail`` extra seconds are
simulated for the replay only; they never change the outcome or metrics.
"""

from __future__ import annotations

import dataclasses
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
from rtt.sim.perturb import Perturbation, Push, Shake
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


def _object_meta(obj: objs.AnyObject) -> dict[str, Any]:
    meta: dict[str, Any] = {"kind": obj.kind, "size": list(obj.size)}
    if isinstance(obj, objs.MeshObject):
        meta.update(
            name=obj.name,
            source_sha256=obj.source_sha256,
            pose_index=obj.pose_index,
            decomposition=obj.decomposition,
            geometry_sha256=obj.geometry_sha256,
        )
    return meta


def _perturbation_meta(perturbation: Perturbation | None) -> dict[str, Any] | None:
    if perturbation is None:
        return None
    return {"type": type(perturbation).__name__.lower(), **dataclasses.asdict(perturbation)}


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
    object_spec: objs.AnyObject,
    placement: objs.Placement,
    controller_setting: int,
    seed: int,
    record: bool = True,
    task: TaskConfig | None = None,
    perturbation: Perturbation | None = None,
) -> EpisodeResult:
    """Simulate one grasp attempt and score it."""
    wall0 = time.perf_counter()
    task = task or default_task()
    sc = task.score
    model, ids = build_model(finger_geom, object_spec, task)
    data = mujoco.MjData(model)

    jitter = sample_jitter(seed, task.jitter)
    mu = task.physics.friction * jitter.mu_scale
    model.geom_friction[ids.object_geoms, 0] = mu

    plan = plan_grasp(finger_geom, object_spec, placement, controller_setting, task)
    shake = perturbation if isinstance(perturbation, Shake) else None
    push = perturbation if isinstance(perturbation, Push) else None
    controller = GraspController(plan, shake=shake)
    actual = placement.actual(jitter.dx, jitter.dy, jitter.dyaw)
    obj_pos, obj_quat = objs.rest_pose(object_spec, actual, task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = plan.palm_start
    data.qpos[ids.finger_qpos] = 0.0
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = obj_pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = obj_quat
    data.ctrl[:] = controller.command(0.0)
    mujoco.mj_forward(model, data)

    checks = PhysicsChecks(model=model, ids=ids, task=task)
    tracker = LiftTracker(cfg=sc)
    cp = task.controller.control_period
    perturb_end = math.nan
    push_ticks = range(0)
    push_wrench = np.zeros(6)
    if perturbation is not None:
        start = plan.t_hold + perturbation.delay
        duration = perturbation.duration
        perturb_end = start + duration
        tracker.min_success_time = perturb_end + task.robustness.post_perturb_hold
        if push is not None:
            first = round(start / cp)
            push_ticks = range(first, first + max(1, round(duration / cp)))
            push_wrench = push.wrench(plan.yaw)
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

    steps_per_tick = _ratio(cp, model.opt.timestep, "control_period")
    ticks_per_check = _ratio(sc.check_period, cp, "check_period")
    max_ticks = round(sc.max_time / cp)
    tail_ticks = round(sc.record_tail / cp) if record else 0
    xfrc = data.xfrc_applied[ids.object_body]
    grasped = False
    stage = Phase.SETTLE  # last phase whose entry bookkeeping has run
    decided_tick: int | None = None
    tick = 0
    while tick < max_ticks:
        if recorder is not None:
            recorder.maybe_record(data)
        data.ctrl[:] = controller.command(data.time)
        if push_ticks:
            xfrc[:] = push_wrench if tick in push_ticks else 0.0
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
        reach_floor=plan.reach_floor,
        palm_gap=plan.palm_gap,
        err_x=placement.err_x,
        err_y=placement.err_y,
        err_yaw_deg=math.degrees(placement.err_yaw),
        perturb_end=perturb_end,
        z0=tracker.z0 if tracker.z0 is not None else float("nan"),
        final_lift=tracker.lift,
        success_time=tracker.success_at if tracker.success_at is not None else float("nan"),
        grasped=float(grasped),
        gripper_env_force_max=checks.gripper_env_force_max,
        penetration_max=checks.penetration_max,
        jaw_penetration_max=checks.jaw_penetration_max,
        object_speed_max=checks.object_speed_max,
        close_xy_drift_max=checks.close_xy_drift_max,
        palm_object_force_max=checks.palm_object_force_max,
        carry_palm_force_max=checks.carry_palm_force_max,
    )

    replay = qpos = None
    if recorder is not None:
        replay, qpos = recorder.finish(
            meta={
                "success": success,
                "failure_mode": mode,
                "seed": seed,
                "controller_setting": controller_setting,
                "object": _object_meta(object_spec),
                "placement": dataclasses.asdict(placement),
                "perturbation": _perturbation_meta(perturbation),
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
