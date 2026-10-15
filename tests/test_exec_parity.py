"""MJWarp executor vs the CPU executor (gate G3 criteria (a) and (b), small versions).

The host-side scoring port is checked without a GPU: a CPU MuJoCo episode is
stepped exactly as ``run_episode`` steps it, the same per-check quantities the GPU
kernels record are computed from it, and :class:`SeriesScorer` must then reproduce
``run_episode``'s outcome, flags and metrics. The remaining tests need CUDA and
compare whole episodes on both executors (the full 288-episode parity set runs in
``python -m rtt.sim.exec_mjwarp parity``; see docs/BENCH_MJWARP.md).
"""

from __future__ import annotations

import math
import os

import mujoco
import numpy as np
import pytest

from rtt.sim import objects as objs
from rtt.sim.controller import GraspController
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.exec_cpu import EpisodeJob, run_jobs
from rtt.sim.exec_mjwarp import (
    F_FN_L,
    F_GE_FORCE,
    F_NEFC,
    F_OBJ_ENV,
    F_PALM_OBJ,
    F_PALM_Y,
    F_PEN,
    F_PEN_GE,
    F_PEN_JAW,
    F_SPEED,
    F_SQ_L,
    N_FEATURES,
    P_PUSH_END,
    P_PUSH_FIRST,
    P_PUSH_FORCE,
    P_PUSH_TORQUE,
    BatchedScene,
    BatchIncompatible,
    SeriesScorer,
    WarpBatch,
    WarpConfig,
    _episode_result,
    batch_key,
    contact_widths,
    entrant_fingers,
    outcome,
    run_jobs_mjwarp,
    stack_model_fields,
    tick_times,
    world_setup,
)
from rtt.sim.m1 import G1_BOX, G1_SETTING, g1_cases
from rtt.sim.perturb import Push, Shake
from rtt.sim.scene import (
    GROUP_ENV,
    GROUP_FINGER_LEFT,
    GROUP_FINGER_RIGHT,
    GROUP_OBJECT,
    GROUP_PALM,
    SceneIds,
    build_model,
)
from rtt.sim.task import load_task

TASK = load_task()
FINGERS = entrant_fingers()
CYLINDER = objs.ObjectSpec("cylinder", (0.02, 0.09))
ROD = objs.ObjectSpec("rod", (0.012, 0.14))
OFFSET = objs.Placement(0.04, -0.03, math.radians(35.0), 0.003, 0.003, math.radians(8.0))


def _gpu_available() -> bool:
    if os.environ.get("RTT_SKIP_GPU_TESTS"):
        return False
    try:
        import mujoco_warp  # noqa: F401
        import warp as wp

        wp.config.quiet = True
        return wp.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - any import or driver failure means "no GPU here"
        return False


needs_gpu = pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA GPU with mujoco_warp")


# --------------------------------------------------------------------------- host scoring (no GPU)


GRIPPER = (GROUP_PALM, GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT)


def _features(model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds) -> np.ndarray:
    """The per-check sample the GPU kernels record, computed from a CPU MjData."""
    f = np.zeros(N_FEATURES)
    f[0:3] = data.xpos[ids.object_body]
    f[F_SPEED] = np.linalg.norm(data.qvel[ids.object_dof : ids.object_dof + 3])
    wrench = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        if con.efc_address < 0:
            continue
        g0, g1 = (int(ids.geom_group[g]) for g in con.geom)
        a, b = sorted((g0, g1))
        depth = -con.dist
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = wrench[0]
        if a in GRIPPER and b in GRIPPER:
            f[F_PEN_JAW] = max(f[F_PEN_JAW], depth)
            continue
        f[F_PEN] = max(f[F_PEN], depth)
        if b == GROUP_OBJECT:
            if a == GROUP_ENV:
                f[F_OBJ_ENV] += normal
            elif a == GROUP_PALM:
                f[F_PALM_OBJ] += normal
            elif a in (GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT):
                side = a - GROUP_FINGER_LEFT
                f[F_FN_L + side] += normal
                sign = 1.0 if g1 == GROUP_OBJECT else -1.0
                f[F_SQ_L + 3 * side : F_SQ_L + 3 * side + 3] += sign * normal * con.frame[:3]
        elif a == GROUP_ENV and b in GRIPPER:
            f[F_GE_FORCE] += normal
            f[F_PEN_GE] = max(f[F_PEN_GE], depth)
    f[F_PALM_Y : F_PALM_Y + 3] = data.xmat[ids.palm_body].reshape(3, 3)[:, 1]
    f[F_NEFC] = data.nefc
    return f


def _replay_on_cpu(job: EpisodeJob) -> EpisodeResult:
    """Step the CPU model like run_episode, but score it through SeriesScorer."""
    s = world_setup(job)
    task = s.task
    model, ids = build_model(job.finger, job.obj, task)
    model.geom_friction[ids.object_geoms, 0] = s.mu
    data = mujoco.MjData(model)
    pos, quat = objs.rest_pose(job.obj, s.actual, task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = s.plan.palm_start
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = quat
    shake = job.perturbation if isinstance(job.perturbation, Shake) else None
    controller = GraspController(s.plan, shake=shake)
    data.ctrl[:] = controller.command(0.0)
    mujoco.mj_forward(model, data)
    weight = model.body_mass[ids.object_body] * float(np.linalg.norm(model.opt.gravity))
    scorer = SeriesScorer([s], np.array([weight]))
    cp = task.controller.control_period
    spt = round(cp / model.opt.timestep)
    tpc = round(task.score.check_period / cp)
    max_ticks = round(task.score.max_time / cp)
    first, end = s.plan_row[P_PUSH_FIRST], s.plan_row[P_PUSH_END]
    # The wrench exactly as the GPU controller kernel applies it: force, then torque.
    wrench = np.concatenate(
        [s.plan_row[P_PUSH_FORCE : P_PUSH_FORCE + 3], s.plan_row[P_PUSH_TORQUE : P_PUSH_TORQUE + 3]]
    )
    tick = check = 0
    while tick < max_ticks and not scorer.all_decided:
        data.ctrl[:] = controller.command(data.time)
        if end > first:
            data.xfrc_applied[ids.object_body] = wrench if first <= tick < end else 0.0
        mujoco.mj_step(model, data, nstep=spt)
        tick += 1
        if tick % tpc == 0:
            scorer.feed(data.time, check, _features(model, data, ids)[None, :])
            check += 1
    times = tick_times(model.opt.timestep, max_ticks * spt)
    return _episode_result(job, s, model, ids, scorer, 0, tick * spt, float(times[tick * spt]))


def _flag_names(result: EpisodeResult) -> list[str]:
    return [f.split("@")[0] for f in result.flags]


def _assert_same_episode(cpu: EpisodeResult, other: EpisodeResult) -> None:
    assert outcome(other) == outcome(cpu)
    assert _flag_names(other) == _flag_names(cpu)
    assert other.n_steps == cpu.n_steps
    assert set(other.metrics) == set(cpu.metrics)
    for key, value in cpu.metrics.items():
        got = other.metrics[key]
        assert (math.isnan(value) and math.isnan(got)) or got == pytest.approx(value, rel=1e-9, abs=1e-12), (
            key
        )


REPLAY_CASES = {
    "straight_box": EpisodeJob(FINGERS["straight"], G1_BOX, objs.Placement(), G1_SETTING, 3, task=TASK),
    "concave_cylinder_offset": EpisodeJob(FINGERS["concave"], CYLINDER, OFFSET, 0, 11, task=TASK),
    "hooked_rod": EpisodeJob(FINGERS["hooked"], ROD, objs.Placement(0.02, 0.01, 0.4), 2, 5, task=TASK),
    "shake": EpisodeJob(
        FINGERS["straight"], G1_BOX, objs.Placement(), G1_SETTING, 1, task=TASK,
        perturbation=Shake("y", 0.01, 3.0, 4, 0.5, 0.2),
    ),
    "push": EpisodeJob(
        FINGERS["straight"], G1_BOX, objs.Placement(0.0, 0.0, 0.5), G1_SETTING, 2, task=TASK,
        perturbation=Push("x", 0.1, 0.01, 0.2),
    ),
    # The certificate's push acts at the object's top: a force plus a torque about the CoM.
    "push_top": EpisodeJob(
        FINGERS["straight"], G1_BOX, objs.Placement(0.0, 0.0, 0.3), G1_SETTING, 4, task=TASK,
        perturbation=Push("y", 0.3, 0.02, 0.2, height=0.04),
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(REPLAY_CASES))
def test_series_scorer_reproduces_run_episode(name: str) -> None:
    """Same physics, scored by the GPU executor's host replay: identical outcome and metrics."""
    job = REPLAY_CASES[name]
    cpu = run_episode(
        job.finger, job.obj, job.placement, job.controller_setting, job.seed,
        record=False, task=job.task, perturbation=job.perturbation,
    )  # fmt: skip
    _assert_same_episode(cpu, _replay_on_cpu(job))


def test_failure_cases_replay_identically() -> None:
    """The G1 deliberate failures (missed grasp, slip) go through the same labels."""
    for case in g1_cases(TASK):
        if case.expect_success:
            continue
        job = EpisodeJob(case.finger, case.obj, objs.Placement(), G1_SETTING, 0, task=case.task)
        cpu = run_episode(
            case.finger, case.obj, objs.Placement(), G1_SETTING, 0, record=False, task=case.task
        )
        assert not cpu.success
        _assert_same_episode(cpu, _replay_on_cpu(job))


def test_batch_key_groups_by_family_and_task() -> None:
    a = EpisodeJob(FINGERS["straight"], G1_BOX, objs.Placement(), 0, 0, task=TASK)
    b = EpisodeJob(FINGERS["hooked"], objs.ObjectSpec("box", (0.05, 0.03, 0.07)), OFFSET, 2, 9, task=TASK)
    c = EpisodeJob(FINGERS["straight"], CYLINDER, objs.Placement(), 0, 0, task=TASK)
    low_mu = TASK.with_overrides(physics={"friction": 0.05})
    d = EpisodeJob(FINGERS["straight"], G1_BOX, objs.Placement(), 0, 0, task=low_mu)
    assert batch_key(a) == batch_key(b)
    assert batch_key(a) != batch_key(c)
    assert batch_key(a) != batch_key(d)


def test_stacked_fields_hold_each_worlds_compiled_values() -> None:
    jobs = [
        EpisodeJob(FINGERS[name], obj, objs.Placement(), 0, seed, task=TASK)
        for name, obj, seed in (
            ("straight", G1_BOX, 0),
            ("hooked", objs.ObjectSpec("box", (0.05, 0.03, 0.07)), 1),
            ("concave", G1_BOX, 2),
        )
    ]
    setups = [world_setup(j) for j in jobs]
    scene = BatchedScene.build(jobs, [s.mu for s in setups])
    for name in (
        "geom_size",
        "geom_pos",
        "geom_aabb",
        "geom_rbound",
        "body_mass",
        "body_inertia",
        "geom_friction",
    ):
        assert name in scene.batched, name
    for w, job in enumerate(jobs):
        model, ids = build_model(job.finger, job.obj, TASK)
        model.geom_friction[ids.object_geoms, 0] = setups[w].mu
        for name, stacked in scene.batched.items():
            np.testing.assert_array_equal(stacked[w], getattr(model, name), err_msg=name)


def test_different_families_cannot_share_a_model() -> None:
    box, _ = build_model(FINGERS["straight"], G1_BOX, TASK)
    cyl, _ = build_model(FINGERS["straight"], CYLINDER, TASK)
    with pytest.raises(BatchIncompatible, match="geom_type"):
        stack_model_fields([box, cyl])


# --------------------------------------------------------------------------- GPU vs CPU


@needs_gpu
def test_g1_cases_agree_with_cpu() -> None:
    jobs = [
        EpisodeJob(c.finger, c.obj, objs.Placement(), G1_SETTING, seed, task=c.task)
        for c in g1_cases(TASK)
        for seed in range(2)
    ]
    cpu = run_jobs(jobs, workers=4).results
    gpu = run_jobs_mjwarp(jobs).results
    assert [outcome(r) for r in gpu] == [outcome(r) for r in cpu]
    for c, g in zip(cpu, gpu, strict=True):
        assert set(g.metrics) == set(c.metrics)
        assert g.max_lift == pytest.approx(c.max_lift, abs=2e-3)
        if c.success:  # a failure may be decided earlier by a GPU-only flag, with the same label
            assert g.metrics["success_time"] == pytest.approx(c.metrics["success_time"], abs=0.05)


@needs_gpu
def test_mixed_families_and_perturbations_agree_with_cpu() -> None:
    jobs = list(REPLAY_CASES.values()) + [
        EpisodeJob(FINGERS["stock" if "stock" in FINGERS else "straight"], ROD, OFFSET, 1, 4, task=TASK)
    ]
    cpu = run_jobs(jobs, workers=4).results
    res = run_jobs_mjwarp(jobs)
    assert len(res.batches) == 3  # box, cylinder and rod each get their own model
    assert [r.seed for r in res.results] == [j.seed for j in jobs]  # job order is kept
    assert [outcome(r) for r in res.results] == [outcome(r) for r in cpu]


@needs_gpu
def test_per_world_geometry_sets_contact_width() -> None:
    """Criterion (a): fingers 10 and 30 mm wide in one batched model touch the box over 10 and 30 mm."""
    base = next(iter(FINGERS))
    from rtt.design.generate import generate_finger
    from rtt.design.schema import load_entrants
    from rtt.sim.m1 import DEFAULT_ENTRANTS

    design = next(d for d in load_entrants(DEFAULT_ENTRANTS) if (d.name or "") == base)
    fingers = [generate_finger(design.model_copy(update={"finger_width_mm": w})) for w in (10.0, 30.0)]
    out = contact_widths(fingers, objs.ObjectSpec("box", (0.06, 0.04, 0.10)), TASK)
    assert out["gpu_widths_mm"] == pytest.approx([10.0, 30.0], abs=0.5)
    assert out["gpu_widths_mm"] == pytest.approx(out["cpu_widths_mm"], abs=0.5)
    assert out["aabb_per_world_matches_compiler"]
    assert out["rbound_per_world_matches_compiler"]
    assert out["aabb_rbound_match_box_formula"]
    assert {"geom_size", "geom_aabb", "geom_rbound", "body_mass", "body_inertia"} <= set(
        out["batched_fields"]
    )


@needs_gpu
def test_recording_is_refused() -> None:
    job = EpisodeJob(FINGERS["straight"], G1_BOX, objs.Placement(), 0, 0, record=True, task=TASK)
    with pytest.raises(ValueError, match="record"):
        run_jobs_mjwarp([job])


@needs_gpu
def test_push_wrench_matches_cpu_inside_the_push_window() -> None:
    """The GPU applies Push.wrench (force and the torque of a push above the CoM) on the CPU's ticks."""
    job = REPLAY_CASES["push_top"]
    assert isinstance(job.perturbation, Push) and job.perturbation.height > 0
    batch = WarpBatch([job], WarpConfig(graph=False))
    setup = batch.setups[0]
    first, end = int(setup.plan_row[P_PUSH_FIRST]), int(setup.plan_row[P_PUSH_END])
    tpc = batch.ticks_per_check
    # A check period whose last control tick lies inside the push window, then one after it.
    inside = next(c for c in range(batch.n_checks) if first <= (c + 1) * tpc - 1 < end)
    after = next(c for c in range(batch.n_checks) if (c + 1) * tpc - 1 >= end)
    batch.advance(inside + 1)
    applied = batch.d.xfrc_applied.numpy()[0, batch.ids.object_body].astype(np.float64)
    expected = job.perturbation.wrench(setup.plan.yaw)
    assert np.abs(expected[3:]).max() > 0.1  # a real torque, not just a force
    np.testing.assert_allclose(applied, expected, rtol=1e-6, atol=1e-6)
    batch.advance(after - inside)
    np.testing.assert_array_equal(batch.d.xfrc_applied.numpy()[0, batch.ids.object_body], 0.0)
