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

Task v2 (:func:`run_episode_v2`) runs a freeform :class:`~rtt.design.freeform.types.FingerPair`
on a part with the same controller phases, checks and lift tracker, and adds:

* registration (:mod:`rtt.sim.register`), measured at an exact control tick 0.25 s after the
  grip ramp ends and before the lift, plus the tilt when the hold is confirmed;
* the ``bad_qacc`` check (a MuJoCo state reset makes the episode ``physics_invalid``);
* ``hold_creep_mm`` (logged, never scored), object-finger penetration, a SHA-256 of the
  state at every check tick (``traj_sha``, for the determinism gates), and when recording,
  per-frame overlay channels for the replay;
* the v2 outcome (:func:`rtt.sim.score.classify_v2`): ``success`` is the registered grasp,
  ``lift_success`` the v1 criterion.

The v2 loop keeps its own clock (``tick * control_period``) so a MuJoCo reset, which sets
``data.time`` back to 0, cannot restart the controller. :func:`run_episode` hands a
``FingerPair`` to :func:`run_episode_v2`, so :class:`~rtt.sim.exec_cpu.EpisodeJob` and the CPU
executor run v2 jobs unchanged (the controller setting is ignored).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from rtt.design.freeform.types import DESIGN_SPACE, FingerPair
from rtt.design.types import FingerGeometry
from rtt.sim import objects as objs
from rtt.sim.checks import PhysicsChecks, contact_summary
from rtt.sim.controller import GraspController, Phase, grasp_fraction, plan_grasp, plan_grasp_v2
from rtt.sim.perturb import Perturbation, Push, Shake
from rtt.sim.register import (
    Registration,
    assess,
    mean_closure,
    not_measured,
    part_pose_in_palm,
    registration_spec,
)
from rtt.sim.replay import Replay, ReplayRecorder
from rtt.sim.scene import (
    GROUP_FINGER_LEFT,
    GROUP_FINGER_RIGHT,
    GROUP_OBJECT,
    build_model,
    build_model_v2,
    finger_mass_properties,
)
from rtt.sim.score import FailureMode, LiftTracker, classify, classify_v2
from rtt.sim.task import JitterConfig, TaskConfig, TaskV2, default_task, default_task_v2

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

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
    finger_geom: FingerGeometry | FingerPair,
    object_spec: objs.AnyObject | PartSpec,
    placement: objs.Placement,
    controller_setting: int,
    seed: int,
    record: bool = True,
    task: TaskConfig | None = None,
    perturbation: Perturbation | None = None,
) -> EpisodeResult:
    """Simulate one grasp attempt and score it (a ``FingerPair`` goes to :func:`run_episode_v2`)."""
    if isinstance(finger_geom, FingerPair):
        return run_episode_v2(
            finger_geom, object_spec, placement, seed, record=record, task=task, perturbation=perturbation
        )
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


# --------------------------------------------------------------------------- task v2

OVERLAY_CHANNELS: tuple[str, ...] = ("offset_mm", "seat_short_mm", "hold_s", "n_contacts")


@dataclass
class EpisodeResultV2(EpisodeResult):
    """A v2 episode: ``success`` is the registered grasp (``success_v2``).

    ``lift_success`` is the v1 lift-and-hold criterion; ``registration`` the verdict of
    :mod:`rtt.sim.register`; ``traj_sha`` a SHA-256 of the state at every check tick (empty when
    ``task.checks.trajectory_hash`` is off); ``overlays`` the per-frame replay channels
    (:data:`OVERLAY_CHANNELS`, float32, one value per replay frame) when recording.
    ``controller_setting`` is -1: task v2 has no grasp-height grid.
    """

    lift_success: bool = False
    registration: Registration | None = None
    traj_sha: str = ""
    design_id: str = ""
    part_id: str = ""
    geometry_hash: str = ""
    bad_qacc: int = 0
    overlays: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def registered(self) -> bool:
        return self.registration is not None and self.registration.registered

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update(
            lift_success=self.lift_success,
            registered=self.registered,
            registration_flags=list(self.registration.sub_flags) if self.registration else [],
            traj_sha=self.traj_sha,
            design_id=self.design_id,
            part_id=self.part_id,
            geometry_hash=self.geometry_hash,
            bad_qacc=self.bad_qacc,
        )
        return out


def _part_meta(part: Any, obj: objs.AnyObject) -> dict[str, Any]:
    meta = _object_meta(obj)
    for key in ("part_id", "family", "yaw_symmetry"):
        if hasattr(part, key):
            meta[key] = getattr(part, key)
    if hasattr(part, "register_axes"):
        meta["register_axes"] = list(part.register_axes)
    if hasattr(part, "material"):
        meta["material"] = part.material.name
    if getattr(part, "visual_asset", None):
        meta["visual_asset"] = part.visual_asset
    return meta


def _as_object(part: Any) -> objs.AnyObject:
    """The simulator object for a part: a ``PartSpec`` becomes its ``MeshObject``."""
    if isinstance(part, objs.ObjectSpec | objs.MeshObject):
        return part
    to_mesh = getattr(part, "to_mesh_object", None)
    if to_mesh is None:
        raise TypeError(f"expected a PartSpec, MeshObject or ObjectSpec, got {type(part).__name__}")
    return to_mesh()


def _part_id(part: Any, obj: objs.AnyObject) -> str:
    pid = getattr(part, "part_id", None) or getattr(obj, "name", None)
    return str(pid) if pid else obj.label


def run_episode_v2(
    pair: FingerPair,
    part: PartSpec | objs.AnyObject,
    placement: objs.Placement,
    seed: int,
    record: bool = True,
    task: TaskV2 | None = None,
    perturbation: Perturbation | None = None,
    *,
    check_part: bool = True,
) -> EpisodeResultV2:
    """Simulate one v2 grasp of ``part`` by the jaw pair ``pair`` and score it (module docstring).

    ``part`` is normally the :class:`~rtt.parts.types.PartSpec` the pair was built for; a v1
    object works too (registration then uses :func:`rtt.sim.register.registration_spec`).
    With ``check_part`` a pair built for a different part is refused; the deliberate-failure
    gate cases pass ``check_part=False`` to grasp a part the jaws were not built for.
    """
    wall0 = time.perf_counter()
    task = task or default_task_v2()
    if not isinstance(task, TaskV2):
        raise TypeError("run_episode_v2 needs a TaskV2 (configs/task_v2.yaml); got a v1 TaskConfig")
    sc, reg_cfg = task.score, task.registration
    obj = _as_object(part)
    part_id = _part_id(part, obj)
    if check_part and pair.part_id != part_id:
        raise ValueError(f"the jaw pair was built for part {pair.part_id!r}, not {part_id!r}")
    spec = registration_spec(part)
    model, ids = build_model_v2(pair, obj, task)
    data = mujoco.MjData(model)

    jitter = sample_jitter(seed, task.jitter)
    mu = task.physics.friction * jitter.mu_scale
    model.geom_friction[ids.object_geoms, 0] = mu

    plan = plan_grasp_v2(pair, obj, placement, task)
    shake = perturbation if isinstance(perturbation, Shake) else None
    push = perturbation if isinstance(perturbation, Push) else None
    controller = GraspController(plan, shake=shake)
    actual = placement.actual(jitter.dx, jitter.dy, jitter.dyaw)
    obj_pos, obj_quat = objs.rest_pose(obj, actual, task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = plan.palm_start
    data.qpos[ids.finger_qpos] = 0.0
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = obj_pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = obj_quat
    data.ctrl[:] = controller.command(0.0)
    mujoco.mj_forward(model, data)

    checks = PhysicsChecks(model=model, ids=ids, task=task, bad_qacc=task.checks.bad_qacc)
    tracker = LiftTracker(cfg=sc)
    cp = task.controller.control_period
    perturb_start = perturb_end = math.nan
    push_ticks = range(0)
    push_wrench = np.zeros(6)
    if perturbation is not None:
        perturb_start = plan.t_hold + perturbation.delay
        perturb_end = perturb_start + perturbation.duration
        tracker.min_success_time = perturb_end + task.robustness.post_perturb_hold
        if push is not None:
            first = round(perturb_start / cp)
            push_ticks = range(first, first + max(1, round(perturbation.duration / cp)))
            push_wrench = push.wrench(plan.yaw)
    recorder = ReplayRecorder(model, ids, task.replay.fps) if record else None
    overlay_rows: list[tuple[float, float, float, float]] = []
    next_overlay_t = 0.0
    half_step = 0.5 * model.opt.timestep
    obj_mass = float(model.body_mass[ids.object_body])
    weight = obj_mass * float(np.linalg.norm(model.opt.gravity))
    min_squeeze = sc.grasp_min_squeeze_frac * task.gripper.grip_force / 2.0
    is_jaw = np.isin(ids.geom_group, (GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT))
    is_obj = ids.geom_group == GROUP_OBJECT

    metrics: dict[str, float] = dict.fromkeys(_PHASE_METRICS, math.nan)

    def support_metrics(prefix: str) -> None:
        cs = contact_summary(model, data, ids)
        metrics[f"{prefix}_table_support"] = cs.object_env_normal / weight
        metrics[f"{prefix}_finger_force"] = sum(cs.object_finger_normal)

    def jaw_object_contacts() -> tuple[int, float]:
        """Active jaw-object contacts and their deepest penetration (m)."""
        if not data.ncon:
            return 0, 0.0
        con = data.contact
        g = con.geom
        mask = (con.efc_address >= 0) & (
            (is_jaw[g[:, 0]] & is_obj[g[:, 1]]) | (is_obj[g[:, 0]] & is_jaw[g[:, 1]])
        )
        if not mask.any():
            return 0, 0.0
        return int(mask.sum()), float(max(0.0, (-con.dist[mask]).max()))

    steps_per_tick = _ratio(cp, model.opt.timestep, "control_period")
    ticks_per_check = _ratio(sc.check_period, cp, "check_period")
    max_ticks = round(sc.max_time / cp)
    tail_ticks = round(sc.record_tail / cp) if record else 0
    reg_tick = round((plan.t_close + plan.close_time + reg_cfg.settle_time) / cp)
    creep_tick = round((plan.t_hold + reg_cfg.creep_delay) / cp)
    creep_end_tick = round(perturb_start / cp) if perturbation is not None else max_ticks
    xfrc = data.xfrc_applied[ids.object_body]
    sha = hashlib.sha256() if task.checks.trajectory_hash else None
    grasped = False
    stage = Phase.SETTLE
    decided_tick: int | None = None
    reg_pose = None
    reg_q = math.nan
    hold_tilt = math.nan
    creep_ref: np.ndarray | None = None
    creep_max = math.nan
    pen_max = pen_hold = 0.0
    hold_contacts: list[int] = []
    tick = 0
    while tick < max_ticks:
        if recorder is not None:
            recorder.maybe_record(data)
            if data.time + half_step >= next_overlay_t:
                pose = part_pose_in_palm(data, ids)
                n_con, _ = jaw_object_contacts()
                overlay_rows.append(
                    (
                        1000.0 * math.hypot(pose.dx, pose.dy),
                        1000.0 * (pair.q_seat - mean_closure(data, ids)),
                        tracker.hold_time if tracker.above else 0.0,
                        float(n_con),
                    )
                )
                next_overlay_t = len(overlay_rows) / task.replay.fps
        data.ctrl[:] = controller.command(tick * cp)
        if push_ticks:
            xfrc[:] = push_wrench if tick in push_ticks else 0.0
        mujoco.mj_step(model, data, nstep=steps_per_tick)
        tick += 1
        if decided_tick is not None:
            if tick >= decided_tick + tail_ticks:
                break
            continue
        if tick == reg_tick:
            reg_pose = part_pose_in_palm(data, ids)
            reg_q = mean_closure(data, ids)
        if tick % ticks_per_check:
            continue

        t = tick * cp
        if sha is not None:
            sha.update(data.qpos.tobytes())
            sha.update(data.qvel.tobytes())
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
                grasped = all(
                    s >= min_squeeze and d <= sc.grasp_max_press_down_ratio * s
                    for s, d in zip(cs.squeeze, cs.press_down, strict=True)
                )
            stage = phase
        checks.update(data, t, phase)
        had_success = tracker.success_at is not None
        tracker.update(t, float(data.xpos[ids.object_body][2]))
        if not had_success and tracker.success_at is not None:
            hold_tilt = part_pose_in_palm(data, ids).tilt
        n_con, depth = jaw_object_contacts()
        pen_max = max(pen_max, depth)
        if creep_tick <= tick < creep_end_tick:
            pen_hold = max(pen_hold, depth)
            hold_contacts.append(n_con)
            if tracker.above:
                rel = _palm_xyz(data, ids)
                if creep_ref is None:
                    creep_ref, creep_max = rel, 0.0
                else:
                    creep_max = max(creep_max, float(np.linalg.norm(rel - creep_ref)))

        settled_after_lift = t >= plan.t_hold + sc.fail_settle_time and not tracker.above
        out_of_time = not tracker.above and t > sc.max_time - sc.success_hold
        if checks.events or tracker.success_at is not None or settled_after_lift or out_of_time:
            decided_tick = tick
            metrics["decided_at"] = t
            if tail_ticks == 0:
                break

    if decided_tick is None:
        metrics["decided_at"] = tick * cp
    if reg_pose is None:
        registration = not_measured(pair.q_seat)
    else:
        registration = assess(reg_pose, reg_q, pair.q_seat, spec, reg_cfg, hold_tilt)
    checks_bad = checks.bad_qacc_count
    success, lift_success, mode = classify_v2(checks.events, tracker, grasped, registration.registered)

    metrics.update(
        mu=mu,
        jitter_dx=jitter.dx,
        jitter_dy=jitter.dy,
        jitter_dyaw_deg=math.degrees(jitter.dyaw),
        object_mass=obj_mass,
        finger_mass=pair.left.mass + pair.right.mass,
        finger_mass_left=pair.left.mass,
        finger_mass_right=pair.right.mass,
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
        lift_success=float(lift_success),
        gripper_env_force_max=checks.gripper_env_force_max,
        penetration_max=checks.penetration_max,
        jaw_penetration_max=checks.jaw_penetration_max,
        obj_finger_pen_max=pen_max,
        obj_finger_pen_hold=pen_hold,
        hold_contacts_mean=float(np.mean(hold_contacts)) if hold_contacts else math.nan,
        object_speed_max=checks.object_speed_max,
        close_xy_drift_max=checks.close_xy_drift_max,
        palm_object_force_max=checks.palm_object_force_max,
        carry_palm_force_max=checks.carry_palm_force_max,
        hold_creep_mm=1000.0 * creep_max if math.isfinite(creep_max) else math.nan,
        bad_qacc=float(checks_bad),
        **registration.as_metrics(),
    )

    replay = qpos = None
    overlays = None
    if recorder is not None:
        replay, qpos = recorder.finish(
            meta={
                "success": success,
                "lift_success": lift_success,
                "failure_mode": mode,
                "registered": registration.registered,
                "registration_flags": list(registration.sub_flags),
                "seed": seed,
                "design_space": DESIGN_SPACE if pair.boxes is None else "box-v1",
                "design_id": pair.design_id,
                "geometry_hash": pair.geometry_hash,
                "part_id": part_id,
                "object": _part_meta(part, obj),
                "placement": dataclasses.asdict(placement),
                "perturbation": _perturbation_meta(perturbation),
                "task": task.name,
                "flags": [e.flag for e in checks.events],
            }
        )
        rows = np.asarray(overlay_rows, dtype=np.float32).reshape(-1, len(OVERLAY_CHANNELS))
        n = replay.n_frames
        rows = rows[:n] if len(rows) >= n else np.vstack([rows, np.repeat(rows[-1:], n - len(rows), axis=0)])
        overlays = {name: rows[:, i].copy() for i, name in enumerate(OVERLAY_CHANNELS)}
    return EpisodeResultV2(
        success=success,
        failure_mode=mode,  # type: ignore[arg-type]  # v2 modes extend the v1 Literal
        max_lift=tracker.max_lift,
        hold_time=tracker.hold_time,
        metrics=metrics,
        flags=tuple(f"{e.flag}@{e.t:.2f}s:{e.phase.name.lower()} {e.detail}".rstrip() for e in checks.events),
        seed=seed,
        controller_setting=-1,
        n_steps=tick * steps_per_tick,
        wall_time_s=time.perf_counter() - wall0,
        replay=replay,
        qpos=qpos,
        lift_success=lift_success,
        registration=registration,
        traj_sha=sha.hexdigest() if sha is not None else "",
        design_id=pair.design_id,
        part_id=part_id,
        geometry_hash=pair.geometry_hash,
        bad_qacc=checks_bad,
        overlays=overlays,
    )


def _palm_xyz(data: mujoco.MjData, ids: Any) -> np.ndarray:
    """The object's origin in the palm frame (m)."""
    r = data.xmat[ids.palm_body].reshape(3, 3)
    return r.T @ (data.xpos[ids.object_body] - data.xpos[ids.palm_body])
