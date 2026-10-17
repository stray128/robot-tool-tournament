"""The shared episode runner: one scorer for every effector family (product mode).

:func:`run_effector_episode` dispatches to the family (:func:`rtt.effectors.registry.get_family`).
Native families run :func:`run_native_episode`, which mirrors the frozen v2 episode
(:func:`rtt.sim.episode.run_episode_v2`) step for step, with the effector in place of the jaws:

* the same seeded jitter (part pose +-0.5 mm / +-1 deg, friction +-10 %, PCG64 from the seed),
  the same task, timestep, control period, check period and decision rules;
* the wrist follows the frozen controller's phases and heights (:func:`rtt.sim.controller._plan`
  and :class:`~rtt.sim.controller.GraspController`), aimed at the nominal placement;
  disturbances (:mod:`rtt.sim.perturb`) act along the part's nominal axes for every family;
* the family's :class:`~rtt.effectors.types.GraspPrimitive` drives only the effector;
* the frozen :class:`~rtt.sim.checks.PhysicsChecks` (including the MuJoCo reset check) and
  :class:`~rtt.sim.score.LiftTracker` run unchanged, and :func:`rtt.sim.score.classify_v2`
  labels the outcome, with the 1 mm held-penetration rule on effector-part contacts;
* registration is measured at the same tick (0.25 s after the close ramp ends) with the
  frozen tolerances (:func:`rtt.sim.register.assess`), against a
  :class:`~rtt.effectors.types.RegistrationReference`, and the family's
  :class:`~rtt.effectors.types.GraspCriterion` replaces the jaw's seat test.

Physics decides pass/fail; no family can change the scorer. A build that raises
:class:`~rtt.effectors.types.InvalidEffector` scores ``invalid_build`` without simulating.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from rtt.effectors.scene import EffectorScene, EffectorSceneIds, build_effector_scene
from rtt.effectors.types import (
    PRODUCT_MODE_LABEL,
    ContractError,
    EffectorEpisodeResult,
    EffectorSpec,
    Engagement,
    InvalidEffector,
    RegistrationReference,
)
from rtt.sim import objects as objs
from rtt.sim.checks import PhysicsChecks
from rtt.sim.controller import GraspController, Phase, _check_reach, _plan, reach_floor
from rtt.sim.episode import OVERLAY_CHANNELS, sample_jitter
from rtt.sim.perturb import Perturbation, Push, Shake, axis_vector
from rtt.sim.register import PalmPose, assess, not_measured, registration_spec
from rtt.sim.replay import Replay, ReplayRecorder, _geom_entry, wxyz_to_xyzw
from rtt.sim.score import LiftTracker, classify_v2
from rtt.sim.task import TaskV2, default_task_v2, task_digest

if TYPE_CHECKING:
    from rtt.effectors.registry import EffectorFamilyImpl
    from rtt.parts.types import PartSpec

__all__ = [
    "EffectorJob",
    "EffectorReplayRecorder",
    "effector_scene_header",
    "invalid_build_result",
    "perturbation_name",
    "run_effector_episode",
    "run_effector_job",
    "run_effector_jobs",
    "run_native_episode",
]

_REPLAY_TYPES = {
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_MESH,
    mujoco.mjtGeom.mjGEOM_PLANE,
}


# --------------------------------------------------------------------------- replay


def _effector_geom_entry(model: mujoco.MjModel, g: int) -> dict[str, Any]:
    """A replay geom; spheres become zero-length capsules and ellipsoids boxes (overlay only)."""
    gtype = mujoco.mjtGeom(model.geom_type[g])
    if gtype in _REPLAY_TYPES:
        return _geom_entry(model, g)
    entry: dict[str, Any] = {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"}
    size = model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        entry.update(type="capsule", radius=round(float(size[0]), 6), half_length=0.0)
    elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        entry.update(type="box", half_size=[round(float(v), 6) for v in size])
    else:
        raise ValueError(f"replay export does not support geom type {gtype.name}")
    entry.update(
        pos=[round(float(v), 6) for v in model.geom_pos[g]],
        quat_xyzw=[round(float(v), 6) for v in wxyz_to_xyzw(model.geom_quat[g])],
        rgba=[round(float(v), 3) for v in model.geom_rgba[g]],
    )
    return entry


def effector_scene_header(model: mujoco.MjModel, ids: EffectorSceneIds, fps: int) -> dict[str, Any]:
    """Version 1 header of an effector replay: the wrist, every effector body that collides, the
    part, each with its collision geoms (what the physics saw), plus the static table and floor.
    Lift it to version 2 with :func:`rtt.sim.replay.upgrade_header`."""
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    bodies = []
    for name, body_id in zip(ids.replay_names, ids.replay_bodies, strict=True):
        geoms = [
            _effector_geom_entry(model, g)
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body_id and collides[g]
        ]
        bodies.append({"name": name, "geoms": geoms})
    static = [_effector_geom_entry(model, g) for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    return {
        "format": "rtt-replay",
        "version": 1,
        "fps": fps,
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
        "up_axis": "z",
        "units": "m",
        "bodies": bodies,
        "static": static,
    }


class EffectorReplayRecorder(ReplayRecorder):
    """:class:`~rtt.sim.replay.ReplayRecorder` for any number of effector bodies."""

    def finish(self, meta: dict[str, Any] | None = None) -> tuple[Replay, np.ndarray]:
        frames = np.stack(self._poses).astype(np.float32, copy=False)
        header = effector_scene_header(self._model, self._ids, self.fps)  # type: ignore[arg-type]
        header["n_frames"] = int(frames.shape[0])
        header["n_bodies"] = int(frames.shape[1])
        header["meta"] = meta or {}
        return Replay(frames=frames, header=header, overlays=None), np.stack(self._qpos)


# --------------------------------------------------------------------------- helpers


def perturbation_name(perturbation: Perturbation | None) -> str | None:
    if perturbation is None:
        return None
    return f"{type(perturbation).__name__.lower()}_{perturbation.axis}"


def _ratio(period: float, base: float, name: str) -> int:
    n = round(period / base)
    if n < 1 or abs(n * base - period) > 1e-9:
        raise ValueError(f"{name}={period} must be a positive multiple of {base}")
    return n


def _placement_dict(placement: objs.Placement) -> dict[str, float]:
    return {k: float(v) for k, v in dataclasses.asdict(placement).items()}


def _quat_to_mat(q: Sequence[float]) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def _mat_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).ravel())
    return tuple(float(v) for v in q)  # type: ignore[return-value]


def _pose_in_wrist(data: mujoco.MjData, ids: EffectorSceneIds) -> tuple[np.ndarray, np.ndarray]:
    """The part's origin and rotation in the wrist frame."""
    r_w = data.xmat[ids.palm_body].reshape(3, 3)
    pos = r_w.T @ (data.xpos[ids.object_body] - data.xpos[ids.palm_body])
    rot = r_w.T @ data.xmat[ids.object_body].reshape(3, 3)
    return pos, rot


def _relative(pos: np.ndarray, rot: np.ndarray, ref: RegistrationReference) -> PalmPose:
    """The measured pose relative to ``ref``, as the frozen rule's :class:`PalmPose`."""
    r_rel = _quat_to_mat(ref.quat_wxyz).T @ rot
    d = pos - np.asarray(ref.pos)
    yaw = math.atan2(r_rel[1, 0], r_rel[0, 0])
    tilt = math.acos(max(-1.0, min(1.0, float(r_rel[2, 2]))))
    return PalmPose(float(d[0]), float(d[1]), float(d[2]), math.remainder(yaw, 2 * math.pi), tilt)


def invalid_build_result(
    spec: EffectorSpec,
    part_id: str,
    seed: int,
    reason: str,
    *,
    condition_id: str = "",
    placement: objs.Placement | None = None,
    perturbation: Perturbation | None = None,
    reference: str = "analytic",
) -> EffectorEpisodeResult:
    """The result of an effector that could not be built for this part (no simulation)."""
    nan = math.nan
    return EffectorEpisodeResult(
        spec=spec,
        part_id=part_id,
        seed=seed,
        success=False,
        held=False,
        engaged=False,
        failure_mode="invalid_build",
        registration_flags=(),
        flags=(f"invalid_build: {reason}"[:200],),
        max_lift=0.0,
        hold_time=0.0,
        bad_qacc=0,
        penetration_mm=nan,
        traj_sha="",
        n_steps=0,
        wall_time_s=0.0,
        metrics={},
        reference=reference,  # type: ignore[arg-type]
        condition_id=condition_id,
        placement=_placement_dict(placement) if placement is not None else {},
        perturbation=perturbation_name(perturbation),
    )


# --------------------------------------------------------------------------- the native runner


def run_native_episode(
    impl: EffectorFamilyImpl,
    spec: EffectorSpec,
    part: PartSpec,
    placement: objs.Placement,
    seed: int,
    *,
    task: TaskV2 | None = None,
    perturbation: Perturbation | None = None,
    record: bool = False,
    reference: RegistrationReference | None = None,
    condition_id: str = "",
    scene: EffectorScene | None = None,
) -> EffectorEpisodeResult:
    """Simulate and score one episode of a native family (module docstring).

    ``reference`` defaults to the analytic wrist-aimed pose; the certificate and the tournament
    pass the nominal episode's pose for ``reference_mode = "nominal_episode"`` families.
    ``scene`` reuses a compiled :class:`~rtt.effectors.scene.EffectorScene` of the same effector
    and part (built with the same task).
    """
    wall0 = time.perf_counter()
    task = task or default_task_v2()
    if not isinstance(task, TaskV2):
        raise TypeError("effector episodes need a TaskV2 (configs/task_v2.yaml)")
    sc, reg_cfg = task.score, task.registration
    if scene is None:
        try:
            scene = build_effector_scene(impl, spec, part, task)
        except InvalidEffector as exc:
            return invalid_build_result(
                spec,
                part.name,
                seed,
                str(exc) or type(exc).__name__,
                condition_id=condition_id,
                placement=placement,
                perturbation=perturbation,
                reference=reference.source if reference else "analytic",
            )
    elif scene.build.spec.effector_id != spec.effector_id or scene.part_id != part.name:
        raise ContractError("the scene was built for a different effector or part")
    model, ids, build = scene.model, scene.ids, scene.build
    data = mujoco.MjData(model)
    obj = part.to_mesh_object()

    jitter = sample_jitter(seed, task.jitter)
    mu = task.physics.friction * jitter.mu_scale
    model.geom_friction[ids.object_geoms, 0] = mu

    primitive = impl.primitive(spec, build)
    criterion = impl.criterion(spec, build)
    pplan = primitive.plan(part, placement, task, build)
    _check_reach(placement)
    h = objs.height(obj)
    aim = replace(placement, yaw=placement.yaw + pplan.yaw_offset)
    plan = _plan(task, aim, h, build.reach, pplan.tip_height, reach_floor(build.reach, h, task))
    wrist = GraspController(plan)
    ref = reference or RegistrationReference.analytic(pplan.yaw_offset)
    part_yaw = placement.yaw
    shake = perturbation if isinstance(perturbation, Shake) else None
    push = perturbation if isinstance(perturbation, Push) else None
    shake_dir = axis_vector(shake.axis, part_yaw) if shake is not None else np.zeros(3)
    shake_start = plan.t_hold + shake.delay if shake is not None else math.inf
    n_act = build.n_actuators
    ctrl = np.zeros(model.nu)

    def command(t: float) -> np.ndarray:
        ctrl[:4] = wrist.command(t)[:4]
        if shake is not None:
            ctrl[:3] += shake.displacement(t - shake_start) * shake_dir
        closure = min(max((t - plan.t_close) / plan.close_time, 0.0), 1.0)
        u = np.asarray(primitive.command(t, wrist.phase(t), closure, model, data, scene), dtype=np.float64)
        if u.shape != (n_act,) or not np.all(np.isfinite(u)):
            raise ContractError(
                f"primitive {primitive.name!r} returned {u.shape} controls; needs {n_act} finite"
            )
        ctrl[4:] = u
        return ctrl

    actual = placement.actual(jitter.dx, jitter.dy, jitter.dyaw)
    obj_pos, obj_quat = objs.rest_pose(obj, actual, task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = plan.palm_start
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = obj_pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = obj_quat
    mujoco.mj_forward(model, data)
    data.ctrl[:] = command(0.0)
    mujoco.mj_forward(model, data)

    checks = PhysicsChecks(model=model, ids=ids, task=task, bad_qacc=task.checks.bad_qacc)  # type: ignore[arg-type]
    tracker = LiftTracker(cfg=sc)
    cp = task.controller.control_period
    perturb_end = math.nan
    push_ticks = range(0)
    push_wrench = np.zeros(6)
    if perturbation is not None:
        perturb_start = plan.t_hold + perturbation.delay
        perturb_end = perturb_start + perturbation.duration
        tracker.min_success_time = perturb_end + task.robustness.post_perturb_hold
        if push is not None:
            first = round(perturb_start / cp)
            push_ticks = range(first, first + max(1, round(perturbation.duration / cp)))
            push_wrench = push.wrench(part_yaw)
    recorder = EffectorReplayRecorder(model, ids, task.replay.fps) if record else None  # type: ignore[arg-type]
    overlay_rows: list[tuple[float, float, float, float]] = []
    next_overlay_t = 0.0
    half_step = 0.5 * model.opt.timestep
    is_eff = np.zeros(model.ngeom, dtype=bool)
    is_eff[ids.effector_geoms] = True
    is_obj = np.zeros(model.ngeom, dtype=bool)
    is_obj[ids.object_geoms] = True

    def effector_part_contacts() -> tuple[int, float]:
        if not data.ncon:
            return 0, 0.0
        con = data.contact
        g = con.geom
        mask = (con.efc_address >= 0) & (
            (is_eff[g[:, 0]] & is_obj[g[:, 1]]) | (is_obj[g[:, 0]] & is_eff[g[:, 1]])
        )
        if not mask.any():
            return 0, 0.0
        return int(mask.sum()), float(max(0.0, (-con.dist[mask]).max()))

    steps_per_tick = _ratio(cp, model.opt.timestep, "control_period")
    ticks_per_check = _ratio(sc.check_period, cp, "check_period")
    max_ticks = round(sc.max_time / cp)
    tail_ticks = round(sc.record_tail / cp) if record else 0
    reg_tick = round((plan.t_close + plan.close_time + reg_cfg.settle_time) / cp)
    xfrc = data.xfrc_applied[ids.object_body]
    others = np.array([b for b in range(model.nbody) if b != ids.object_body], dtype=np.int64)
    sha = hashlib.sha256() if task.checks.trajectory_hash else None
    grasped = False
    stage = Phase.SETTLE
    decided_tick: int | None = None
    reg_pose: tuple[np.ndarray, np.ndarray] | None = None
    engagement: Engagement | None = None
    hold_rot: np.ndarray | None = None
    pen_max = 0.0
    tilt_cos_min = 1.0
    hold_contacts: list[int] = []
    tick = 0
    while tick < max_ticks:
        if recorder is not None:
            recorder.maybe_record(data)
            if data.time + half_step >= next_overlay_t:
                pos, rot = _pose_in_wrist(data, ids)
                rel = _relative(pos, rot, ref)
                n_con, _ = effector_part_contacts()
                overlay_rows.append(
                    (
                        1000.0 * math.hypot(rel.dx, rel.dy),
                        math.nan,
                        tracker.hold_time if tracker.above else 0.0,
                        float(n_con),
                    )
                )
                next_overlay_t = len(overlay_rows) / task.replay.fps
        data.ctrl[:] = command(tick * cp)
        if data.qfrc_applied.any() or data.xfrc_applied[others].any():
            raise ContractError(
                f"primitive {primitive.name!r} applied forces directly; act through actuators"
            )
        xfrc[:] = push_wrench if tick in push_ticks else 0.0
        mujoco.mj_step(model, data, nstep=steps_per_tick)
        tick += 1
        if decided_tick is not None:
            if tick >= decided_tick + tail_ticks:
                break
            continue
        if tick == reg_tick:
            reg_pose = _pose_in_wrist(data, ids)
            engagement = criterion.engaged(model, data, scene)
            tilt_cos_min = min(tilt_cos_min, float(data.xmat[ids.object_body][8]))
        if tick % ticks_per_check:
            continue

        t = tick * cp
        if sha is not None:
            sha.update(data.qpos.tobytes())
            sha.update(data.qvel.tobytes())
        phase = wrist.phase(t)
        if phase > stage:
            if stage < Phase.DESCEND <= phase:
                tracker.set_reference(float(data.xpos[ids.object_body][2]))
            if stage < Phase.LIFT <= phase:
                grasped = bool(criterion.grasped(model, data, scene))
            stage = phase
        checks.update(data, t, phase)
        had_success = tracker.success_at is not None
        tracker.update(t, float(data.xpos[ids.object_body][2]))
        if not had_success and tracker.success_at is not None:
            hold_rot = _pose_in_wrist(data, ids)[1]
        tilt_cos_min = min(tilt_cos_min, float(data.xmat[ids.object_body][8]))
        n_con, depth = effector_part_contacts()
        pen_max = max(pen_max, depth)
        if stage >= Phase.HOLD:
            hold_contacts.append(n_con)

        settled_after_lift = t >= plan.t_hold + sc.fail_settle_time and not tracker.above
        out_of_time = not tracker.above and t > sc.max_time - sc.success_hold
        if checks.events or tracker.success_at is not None or settled_after_lift or out_of_time:
            decided_tick = tick
            if tail_ticks == 0:
                break

    decided_at = (decided_tick if decided_tick is not None else tick) * cp
    rspec = registration_spec(part)
    if reg_pose is None or engagement is None:
        registration = not_measured(0.0)
        engaged = False
        eng_flags: tuple[str, ...] = ()
        eng_metrics: dict[str, float] = {}
    else:
        hold_tilt = math.nan
        if hold_rot is not None:
            r_rel = _quat_to_mat(ref.quat_wxyz).T @ hold_rot
            hold_tilt = math.acos(max(-1.0, min(1.0, float(r_rel[2, 2]))))
        pose = _relative(reg_pose[0], reg_pose[1], ref)
        registration = assess(pose, 0.0, 0.0, rspec, reg_cfg, hold_tilt)  # q = q_seat: no seat test here
        engaged = engagement.ok
        eng_flags = engagement.sub_flags
        eng_metrics = dict(engagement.metrics)
    registered = registration.registered and engaged
    success, held, mode = classify_v2(
        checks.events,
        tracker,
        grasped,
        registered,
        penetration_mm=1000.0 * pen_max,
        held_penetration_max_mm=task.checks.held_penetration_max_mm,
    )

    reg_metrics = registration.as_metrics()
    for k in ("q_seat_mm", "reg_q_mm", "seat_short_mm"):
        reg_metrics.pop(k, None)
    reg_metrics["registered"] = float(registered)
    metrics: dict[str, float] = {
        "mu": mu,
        "jitter_dx": jitter.dx,
        "jitter_dy": jitter.dy,
        "jitter_dyaw_deg": math.degrees(jitter.dyaw),
        "object_mass": float(model.body_mass[ids.object_body]),
        "effector_mass": float(build.mass) if build.mass is not None else math.nan,
        "tip_height": plan.tip_height,
        "reach": build.reach,
        "yaw_offset_deg": math.degrees(pplan.yaw_offset),
        "perturb_end": perturb_end,
        "decided_at": decided_at,
        "z0": tracker.z0 if tracker.z0 is not None else math.nan,
        "final_lift": tracker.lift,
        "success_time": tracker.success_at if tracker.success_at is not None else math.nan,
        "grasped": float(grasped),
        "held": float(held),
        "engaged": float(engaged),
        "gripper_env_force_max": checks.gripper_env_force_max,
        "penetration_max": checks.penetration_max,
        "self_penetration_max": checks.jaw_penetration_max,
        "obj_effector_pen_max": pen_max,
        "hold_contacts_mean": float(np.mean(hold_contacts)) if hold_contacts else math.nan,
        "object_speed_max": checks.object_speed_max,
        "close_xy_drift_max": checks.close_xy_drift_max,
        "palm_object_force_max": checks.palm_object_force_max,
        "carry_palm_force_max": checks.carry_palm_force_max,
        "tilt_max_deg": math.degrees(math.acos(max(-1.0, min(1.0, tilt_cos_min)))),
        "bad_qacc": float(checks.bad_qacc_count),
        **reg_metrics,
        **{f"eng_{k}": v for k, v in eng_metrics.items()},
    }
    reg_flags = tuple(f for f in registration.sub_flags) + tuple(eng_flags)
    flag_names = tuple(e.flag for e in checks.events)

    replay = qpos = None
    overlays = None
    if recorder is not None:
        replay, qpos = recorder.finish(
            meta={
                "success": success,
                "held": held,
                "lift_success": held,
                "failure_mode": mode,
                "registered": registered,
                "registration_flags": list(reg_flags),
                "seed": seed,
                "design_space": f"effector-{spec.family.value}",
                "design_id": spec.effector_id,
                "geometry_hash": "",
                "part_id": part.name,
                "object": {
                    "kind": "part",
                    "part_id": part.name,
                    "family": part.family,
                    "size": list(part.size),
                },
                "placement": _placement_dict(placement),
                "perturbation": perturbation_name(perturbation),
                "task": task.name,
                "task_digest": task_digest(task),
                "flags": list(flag_names),
                "effector": {
                    "family": spec.family.value,
                    "model": spec.model,
                    "display_name": spec.display_name,
                    "primitive": spec.primitive,
                    "origin": spec.origin,
                    "fidelity": spec.fidelity.value,
                    "label": PRODUCT_MODE_LABEL,
                },
            }
        )
        rows = np.asarray(overlay_rows, dtype=np.float32).reshape(-1, len(OVERLAY_CHANNELS))
        n = replay.n_frames
        rows = rows[:n] if len(rows) >= n else np.vstack([rows, np.repeat(rows[-1:], n - len(rows), axis=0)])
        overlays = {name: rows[:, i].copy() for i, name in enumerate(OVERLAY_CHANNELS)}

    return EffectorEpisodeResult(
        spec=spec,
        part_id=part.name,
        seed=seed,
        success=success,
        held=held,
        engaged=engaged,
        failure_mode=mode,
        registration_flags=reg_flags,
        flags=flag_names,
        max_lift=tracker.max_lift,
        hold_time=tracker.hold_time,
        bad_qacc=checks.bad_qacc_count,
        penetration_mm=1000.0 * pen_max,
        traj_sha=sha.hexdigest() if sha is not None else "",
        n_steps=tick * steps_per_tick,
        wall_time_s=time.perf_counter() - wall0,
        metrics=metrics,
        reference=ref.source,
        reg_pos=tuple(float(v) for v in reg_pose[0]) if reg_pose is not None else None,  # type: ignore[arg-type]
        reg_quat_wxyz=_mat_to_quat(reg_pose[1]) if reg_pose is not None else None,
        condition_id=condition_id,
        placement=_placement_dict(placement),
        perturbation=perturbation_name(perturbation),
        effector_mass_kg=float(build.mass) if build.mass is not None else math.nan,
        replay=replay,
        qpos=qpos,
        overlays=overlays,
    )


# --------------------------------------------------------------------------- dispatch and jobs


def run_effector_episode(
    spec: EffectorSpec,
    part: PartSpec,
    placement: objs.Placement,
    seed: int,
    *,
    task: TaskV2 | None = None,
    perturbation: Perturbation | None = None,
    record: bool = False,
    reference: RegistrationReference | None = None,
    condition_id: str = "",
) -> EffectorEpisodeResult:
    """One scored episode of ``spec`` on ``part`` (dispatched to its family)."""
    from rtt.effectors.registry import get_family

    return get_family(spec.family).run_episode(
        spec,
        part,
        placement,
        seed,
        task=task,
        perturbation=perturbation,
        record=record,
        reference=reference,
        condition_id=condition_id,
    )


@dataclass(frozen=True)
class EffectorJob:
    """Everything needed to reproduce one effector episode (picklable for worker processes)."""

    spec: EffectorSpec
    part: Any  # PartSpec
    placement: objs.Placement
    seed: int
    task: TaskV2 | None = None
    perturbation: Perturbation | None = None
    record: bool = False
    reference: RegistrationReference | None = None
    condition_id: str = ""


_SCENES: collections.OrderedDict[tuple[str, str, str, str], EffectorScene] = collections.OrderedDict()
_SCENE_CACHE_SIZE = 8


def _cached_scene(impl: EffectorFamilyImpl, job: EffectorJob, task: TaskV2) -> EffectorScene:
    key = (job.spec.effector_id, job.part.name, job.part.geometry_sha256, task_digest(task))
    scene = _SCENES.get(key)
    if scene is None:
        scene = build_effector_scene(impl, job.spec, job.part, task)
        _SCENES[key] = scene
        while len(_SCENES) > _SCENE_CACHE_SIZE:
            _SCENES.popitem(last=False)
    else:
        _SCENES.move_to_end(key)
    return scene


def run_effector_job(job: EffectorJob) -> EffectorEpisodeResult:
    """Run one job; native families reuse a compiled scene per (effector, part, task) in-process."""
    from rtt.effectors.registry import EffectorFamilyImpl, get_family

    impl = get_family(job.spec.family)
    kwargs: dict[str, Any] = {
        "task": job.task,
        "perturbation": job.perturbation,
        "record": job.record,
        "reference": job.reference,
        "condition_id": job.condition_id,
    }
    if type(impl).run_episode is EffectorFamilyImpl.run_episode:
        task = job.task or default_task_v2()
        try:
            kwargs["scene"] = _cached_scene(impl, job, task)
        except InvalidEffector as exc:
            return invalid_build_result(
                job.spec,
                job.part.name,
                job.seed,
                str(exc) or type(exc).__name__,
                condition_id=job.condition_id,
                placement=job.placement,
                perturbation=job.perturbation,
                reference=job.reference.source if job.reference else "analytic",
            )
        kwargs["task"] = task
    return impl.run_episode(job.spec, job.part, job.placement, job.seed, **kwargs)


def run_effector_jobs(jobs: Sequence[EffectorJob], workers: int | None = 1) -> list[EffectorEpisodeResult]:
    """Run jobs in order: in-process for ``workers == 1``, else on a spawned process pool.

    Results come back in job order and are identical either way (each episode is a pure
    function of its job). Family modules are imported in each worker on first use.
    """
    jobs = list(jobs)
    if workers is None:
        workers = max(1, (mp.cpu_count() or 2) - 1)
    if workers <= 1 or len(jobs) <= 1:
        return [run_effector_job(j) for j in jobs]
    with mp.get_context("spawn").Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(run_effector_job, jobs, chunksize=1)
