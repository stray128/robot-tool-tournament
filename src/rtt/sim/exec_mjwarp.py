"""MJWarp executor: many grasp episodes advanced in lockstep as one batched GPU simulation.

This is the GPU counterpart of :mod:`rtt.sim.exec_cpu`. It takes the same
:class:`~rtt.sim.exec_cpu.EpisodeJob` list and returns one
:class:`~rtt.sim.episode.EpisodeResult` per job, in job order, so the two
executors can be compared episode by episode (gate G3, plan section 3.6).
:func:`run_jobs` has the ``run_jobs(jobs, workers)`` signature that
:mod:`rtt.sim.evaluate` expects of the ``"mjwarp"`` executor.

How a batch is built
--------------------
* **Same model.** Every world's scene is compiled by the CPU executor's own
  :func:`rtt.sim.scene.build_model` (cached per finger, object and task) and gets
  the episode's jittered object friction. The first world's model is uploaded with
  ``mujoco_warp.put_model``; every model field that differs between worlds (the
  finger box sizes and poses, the per-world ``geom_aabb`` / ``geom_rbound`` the
  MuJoCo compiler derives from them, finger and object mass properties, the inverse
  weights derived from those, the object friction) becomes an MJWarp per-world
  batched field holding each world's compiled values. Every other field MJWarp
  reads must be identical across the batch, otherwise :class:`BatchIncompatible`
  is raised. :func:`batch_key` groups jobs so that only same-topology scenes share
  a batch (one batch per object family and simulator task; one per mesh object).
* **Same controller.** The open-loop state machine of :mod:`rtt.sim.controller`
  (including a :class:`~rtt.sim.perturb.Shake`) is a Warp kernel that evaluates
  every world's :class:`~rtt.sim.controller.GraspPlan` in float64 at the CPU
  executor's exact tick times; a :class:`~rtt.sim.perturb.Push` is applied through
  ``xfrc_applied`` on the same ticks as on the CPU, as its full
  :meth:`~rtt.sim.perturb.Push.wrench` (the force and, for a push above the centre
  of mass, its torque).
* **Same scoring.** Every ``check_period`` a kernel reduces each world's active
  contacts to the quantities :mod:`rtt.sim.checks` uses (deepest penetration and
  whether it is a gripper-environment pair, gripper-environment and palm-object
  normal force, object support, each finger's net normal force on the object) and
  records them with the object's pose and speed. The host replays these series
  through :class:`SeriesScorer`, a vectorised port of the
  :func:`rtt.sim.episode.run_episode` check loop (phase-entry metrics,
  :class:`~rtt.sim.checks.PhysicsChecks` thresholds,
  :class:`~rtt.sim.score.LiftTracker` rules), and labels every episode with
  :func:`rtt.sim.score.classify`.

All worlds of a batch advance together until every world is decided (checked
every ``chunk_checks`` check periods) or ``max_time``; a world's outcome uses only
the samples up to its own decision, as the CPU loop stops there.

Remaining differences from the CPU executor: MJWarp computes in float32, always
uses its own convex collision (``native_ccd: false`` selects libccd on the CPU
only), and builds contacts and constraint rows with atomics, so their order (and
float rounding) varies between runs. Gate G3 (``docs/BENCH_MJWARP.md``) found ~97 %
outcome agreement with the CPU and ~1 % same-seed rerun disagreement, so this is
**not the executor of record**: it is a cross-check, and its numbers are never
published where they disagree with the CPU executor. Replays are not recorded here:
re-run an episode on the CPU executor for one.

``python -m rtt.sim.exec_mjwarp`` runs the spike criteria (a) to (e); see :func:`main`.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import warp as wp

from rtt.sim import objects as objs
from rtt.sim.checks import (
    FLAG_EJECTION,
    FLAG_GRIPPER_ENV,
    FLAG_OFF_TABLE,
    FLAG_PALM_CONTACT,
    FLAG_PENETRATION,
    FLAG_SPEED_CAP,
    CheckEvent,
)
from rtt.sim.controller import GraspPlan, Phase, grasp_fraction, plan_grasp
from rtt.sim.episode import EpisodeResult, Jitter, _ratio, sample_jitter
from rtt.sim.exec_cpu import BatchResult, EpisodeJob
from rtt.sim.perturb import Push, Shake, axis_vector
from rtt.sim.scene import (
    GROUP_ENV,
    GROUP_FINGER_LEFT,
    GROUP_FINGER_RIGHT,
    GROUP_OBJECT,
    GROUP_PALM,
    SceneIds,
    build_model,
    finger_mass_properties,
)
from rtt.sim.score import LiftTracker, classify
from rtt.sim.task import TaskConfig, default_task, task_digest

# --------------------------------------------------------------------------- layouts

# Per-world contact reductions, zeroed every check period (see _contact_kernel). Maxima and
# sums live in separate arrays (Warp's deterministic mode cannot mix them on one array).
M_PEN = 0  # deepest penetration among contacts that are not finger-finger (>= 0)
M_PEN_GE = 1  # deepest penetration among gripper-environment contacts (>= 0)
M_PEN_JAW = 2  # deepest finger-finger (jaw) penetration (>= 0)
N_MAX = 3
S_GE_FORCE = 0  # summed normal force of gripper-environment contacts
S_OBJ_ENV = 1  # summed normal force of object-environment contacts
S_FN_L = 2  # summed normal force of the left finger on the object (right: S_FN_L + 1)
S_SQ_L = 4  # 3: left finger's net normal force on the object, world frame (right: S_SQ_L + 3)
S_PALM_OBJ = 10  # summed normal force of palm-object contacts
N_SUM = 11

# Per-world sample recorded every check period: object pose and speed, the maxima, the
# sums, the palm's closing axis, the constraint count and the overflow bits.
F_OBJ_X, F_OBJ_Y, F_OBJ_Z = 0, 1, 2
F_SPEED = 3
F_MAX = 4  # N_MAX maxima follow
F_SUM = F_MAX + N_MAX  # N_SUM sums follow
F_PEN, F_PEN_GE, F_PEN_JAW = F_MAX + M_PEN, F_MAX + M_PEN_GE, F_MAX + M_PEN_JAW
F_GE_FORCE, F_OBJ_ENV, F_PALM_OBJ = F_SUM + S_GE_FORCE, F_SUM + S_OBJ_ENV, F_SUM + S_PALM_OBJ
F_FN_L, F_FN_R = F_SUM + S_FN_L, F_SUM + S_FN_L + 1
F_SQ_L, F_SQ_R = F_SUM + S_SQ_L, F_SUM + S_SQ_L + 3
F_PALM_Y = F_SUM + N_SUM  # 3: palm Y axis (the closing axis) in world coordinates
F_NEFC = F_PALM_Y + 3  # constraint rows in use
F_OVERFLOW = F_NEFC + 1  # the world's sticky mujoco_warp OverflowType bitmask so far
N_FEATURES = F_OVERFLOW + 1

# The same indices as Warp compile-time constants for the kernels.
_M_PEN, _M_PEN_GE, _M_PEN_JAW, _N_MAX = (wp.constant(v) for v in (M_PEN, M_PEN_GE, M_PEN_JAW, N_MAX))
_S_GE_FORCE, _S_OBJ_ENV, _S_FN_L, _S_SQ_L, _S_PALM_OBJ, _N_SUM = (
    wp.constant(v) for v in (S_GE_FORCE, S_OBJ_ENV, S_FN_L, S_SQ_L, S_PALM_OBJ, N_SUM)
)
_F_MAX, _F_SUM, _F_PALM_Y = wp.constant(F_MAX), wp.constant(F_SUM), wp.constant(F_PALM_Y)
_F_NEFC, _F_OVERFLOW = wp.constant(F_NEFC), wp.constant(F_OVERFLOW)
# Geom groups (rtt.sim.scene.GROUP_*).
_G_ENV, _G_PALM, _G_LEFT, _G_RIGHT, _G_OBJECT = (
    wp.constant(v) for v in (GROUP_ENV, GROUP_PALM, GROUP_FINGER_LEFT, GROUP_FINGER_RIGHT, GROUP_OBJECT)
)

# Per-world controller rows (float64), see _plan_row().
P_X, P_Y, P_YAW, P_APPROACH, P_GRASP, P_LIFT = 0, 1, 2, 3, 4, 5
P_T_DESCEND, P_T_CLOSE, P_T_LIFT, P_T_HOLD, P_CLOSE_TIME, P_OPEN, P_CLOSE = 6, 7, 8, 9, 10, 11, 12
P_SHAKE_START, P_SHAKE_DURATION, P_SHAKE_RAMP, P_SHAKE_AMPLITUDE, P_SHAKE_FREQUENCY = 13, 14, 15, 16, 17
P_SHAKE_DIR = 18  # 3
P_PUSH_FIRST, P_PUSH_END = 21, 22  # push tick range [first, end)
P_PUSH_FORCE = 23  # 3: world-frame force on the object's centre of mass
P_PUSH_TORQUE = 26  # 3: world-frame torque about the centre of mass (Push.height lever arm)
N_PLAN = 29

# Check flags in the order PhysicsChecks.update raises them (same-time ties keep this order).
FLAG_ORDER: tuple[str, ...] = (
    FLAG_PENETRATION,
    FLAG_GRIPPER_ENV,
    FLAG_PALM_CONTACT,
    FLAG_SPEED_CAP,
    FLAG_OFF_TABLE,
    FLAG_EJECTION,
)
_FI = {name: i for i, name in enumerate(FLAG_ORDER)}

# Overflow bits (mujoco_warp.OverflowType). Per-world capacities (constraint rows) only spoil
# the world that overflows them; the contact buffers are shared by every world of a batch.
_WORLD_OVERFLOW = ("NEFC", "NJMAX_NNZ")
_SHARED_OVERFLOW = ("BROADPHASE", "NARROWPHASE", "CCD", "EPA_HORIZON")

# put_model folds actuator damping into these, so per-world values cannot be copied verbatim.
_TRANSFORMED_FIELDS = frozenset({"dof_damping", "dof_dampingpoly", "tendon_damping", "tendon_dampingpoly"})

# Shared fields MJWarp reads only while sleeping is enabled (mujoco_warp/_src/sleep.py); with
# sleeping off (as in every task here) they may differ between worlds.
_SLEEP_ONLY_FIELDS = frozenset({"dof_length"})


def _mjw() -> Any:
    """``mujoco_warp``, imported on first use (keeps CPU-only importers light)."""
    import mujoco_warp

    return mujoco_warp


# --------------------------------------------------------------------------- kernels


@wp.func
def _min_jerk(s: wp.float64) -> wp.float64:
    s = wp.clamp(s, wp.float64(0.0), wp.float64(1.0))
    return s * s * s * (wp.float64(10.0) + s * (wp.float64(-15.0) + wp.float64(6.0) * s))


@wp.func
def _shake_displacement(p: wp.array2d(dtype=wp.float64), w: int, t: wp.float64) -> wp.float64:
    """perturb.Shake.displacement at time ``t`` (0 without a shake: amplitude 0)."""
    s = t - p[w, 13]
    duration = p[w, 14]
    if s <= wp.float64(0.0) or s >= duration:
        return wp.float64(0.0)
    freq = p[w, 17]
    ramp = p[w, 15]
    window = wp.float64(1.0)
    if ramp > wp.float64(0.0):
        u = wp.min(wp.float64(1.0), wp.min(s / ramp, (duration - s) / ramp))
        window = wp.float64(0.5) * (wp.float64(1.0) - wp.cos(wp.float64(wp.pi) * u))
    return p[w, 16] * window * wp.sin(wp.float64(2.0 * wp.pi) * freq * s)


@wp.kernel
def _ctrl_kernel(
    tick: wp.array(dtype=wp.int32),
    times: wp.array(dtype=wp.float64),
    steps_per_tick: wp.int32,
    plan: wp.array2d(dtype=wp.float64),
    act_ids: wp.array(dtype=wp.int32),
    object_body: wp.int32,
    ctrl: wp.array2d(dtype=wp.float32),
    xfrc: wp.array2d(dtype=wp.spatial_vector),
):
    """GraspController.command (and a Push wrench) for every world at this tick's CPU time."""
    w = wp.tid()
    k = tick[0]
    t = times[k * steps_per_tick]
    t_descend = plan[w, 6]
    t_close = plan[w, 7]
    approach = plan[w, 3]
    grasp = plan[w, 4]
    z = grasp
    if t < t_close:
        s = _min_jerk((t - t_descend) / (t_close - t_descend))
        z = approach + s * (grasp - approach)
    else:
        s = _min_jerk((t - plan[w, 8]) / (plan[w, 9] - plan[w, 8]))
        z = grasp + s * (plan[w, 5] - grasp)
    g = wp.clamp((t - t_close) / plan[w, 10], wp.float64(0.0), wp.float64(1.0))
    grip = plan[w, 11] + g * (plan[w, 12] - plan[w, 11])
    d = _shake_displacement(plan, w, t)
    ctrl[w, act_ids[0]] = wp.float32(plan[w, 0] + d * plan[w, 18])
    ctrl[w, act_ids[1]] = wp.float32(plan[w, 1] + d * plan[w, 19])
    ctrl[w, act_ids[2]] = wp.float32(z + d * plan[w, 20])
    ctrl[w, act_ids[3]] = wp.float32(plan[w, 2])
    ctrl[w, act_ids[4]] = wp.float32(grip)
    kf = wp.float64(k)
    f = wp.vec3(0.0, 0.0, 0.0)
    tq = wp.vec3(0.0, 0.0, 0.0)
    if kf >= plan[w, 21] and kf < plan[w, 22]:
        f = wp.vec3(wp.float32(plan[w, 23]), wp.float32(plan[w, 24]), wp.float32(plan[w, 25]))
        tq = wp.vec3(wp.float32(plan[w, 26]), wp.float32(plan[w, 27]), wp.float32(plan[w, 28]))
    # xfrc_applied is (force, torque about the body's centre of mass), as Push.wrench returns.
    xfrc[w, object_body] = wp.spatial_vector(f[0], f[1], f[2], tq[0], tq[1], tq[2])


@wp.kernel
def _increment_kernel(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _contact_kernel(
    nacon: wp.array(dtype=wp.int32),
    con_dist: wp.array(dtype=wp.float32),
    con_frame: wp.array(dtype=wp.mat33),
    con_geom: wp.array(dtype=wp.vec2i),
    con_dim: wp.array(dtype=wp.int32),
    con_efc: wp.array2d(dtype=wp.int32),
    con_world: wp.array(dtype=wp.int32),
    efc_force: wp.array2d(dtype=wp.float32),
    geom_group: wp.array(dtype=wp.int32),
    pyramidal: wp.int32,
    acc_max: wp.array2d(dtype=wp.float32),
    acc_sum: wp.array2d(dtype=wp.float32),
):
    """Reduce active contacts into the per-world quantities of checks.py (cf. contact_summary)."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    adr = con_efc[c, 0]
    if adr < 0:
        return
    w = con_world[c]
    g0 = geom_group[con_geom[c][0]]
    g1 = geom_group[con_geom[c][1]]
    a = wp.min(g0, g1)
    b = wp.max(g0, g1)
    depth = -con_dist[c]
    # Normal force: the first row of an elliptic cone; the sum of the pyramid edges otherwise.
    normal = efc_force[w, adr]
    if pyramidal != 0:
        dim = con_dim[c]
        if dim > 1:
            normal = wp.float32(0.0)
            for i in range(2 * (dim - 1)):
                normal += efc_force[w, con_efc[c, i]]
    gripper_a = a >= _G_PALM and a <= _G_RIGHT
    gripper_b = b >= _G_PALM and b <= _G_RIGHT
    if gripper_a and gripper_b:  # finger against finger: tracked, never flagged
        wp.atomic_max(acc_max, w, _M_PEN_JAW, depth)
        return
    wp.atomic_max(acc_max, w, _M_PEN, depth)
    if b == _G_OBJECT:
        if a == _G_ENV:
            wp.atomic_add(acc_sum, w, _S_OBJ_ENV, normal)
        elif a == _G_PALM:
            wp.atomic_add(acc_sum, w, _S_PALM_OBJ, normal)
        elif a == _G_LEFT or a == _G_RIGHT:
            side = a - _G_LEFT
            wp.atomic_add(acc_sum, w, _S_FN_L + side, normal)
            # The normal points from geom[0] to geom[1] and pushes geom[1] along it.
            sign = wp.float32(-1.0)
            if g1 == _G_OBJECT:
                sign = wp.float32(1.0)
            fr = con_frame[c]
            base = _S_SQ_L + 3 * side
            wp.atomic_add(acc_sum, w, base + 0, sign * normal * fr[0, 0])
            wp.atomic_add(acc_sum, w, base + 1, sign * normal * fr[0, 1])
            wp.atomic_add(acc_sum, w, base + 2, sign * normal * fr[0, 2])
    elif a == _G_ENV and gripper_b:
        wp.atomic_add(acc_sum, w, _S_GE_FORCE, normal)
        wp.atomic_max(acc_max, w, _M_PEN_GE, depth)


@wp.kernel
def _record_kernel(
    check: wp.array(dtype=wp.int32),
    acc_max: wp.array2d(dtype=wp.float32),
    acc_sum: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    qvel: wp.array2d(dtype=wp.float32),
    nefc: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
    object_body: wp.int32,
    palm_body: wp.int32,
    object_dof: wp.int32,
    hist: wp.array3d(dtype=wp.float32),
):
    """Write one check sample per world into ``hist[check]`` (layout: the F_* constants)."""
    w = wp.tid()
    k = check[0]
    p = xpos[w, object_body]
    hist[k, w, 0] = p[0]
    hist[k, w, 1] = p[1]
    hist[k, w, 2] = p[2]
    v = wp.vec3(qvel[w, object_dof], qvel[w, object_dof + 1], qvel[w, object_dof + 2])
    hist[k, w, 3] = wp.length(v)
    for i in range(_N_MAX):
        hist[k, w, _F_MAX + i] = acc_max[w, i]
    for i in range(_N_SUM):
        hist[k, w, _F_SUM + i] = acc_sum[w, i]
    m = xmat[w, palm_body]
    hist[k, w, _F_PALM_Y + 0] = m[0, 1]
    hist[k, w, _F_PALM_Y + 1] = m[1, 1]
    hist[k, w, _F_PALM_Y + 2] = m[2, 1]
    hist[k, w, _F_NEFC] = wp.float32(nefc[w])
    hist[k, w, _F_OVERFLOW] = wp.float32(overflow[w])


# --------------------------------------------------------------------------- configuration and results


class BatchIncompatible(ValueError):
    """Jobs whose compiled models differ in a field MJWarp cannot batch per world."""


class CapacityOverflow(RuntimeError):
    """A world needed more contacts or constraint rows than reserved; its physics is invalid."""


@dataclass(frozen=True)
class WarpConfig:
    """Capacity and scheduling knobs of the MJWarp executor (none of them change the physics).

    ``nconmax`` contacts and ``njmax`` constraint rows are reserved per world. A batch
    in which any world overflows either is rerun from the start with both doubled, up
    to ``overflow_retries`` times, then :class:`CapacityOverflow` is raised (a world's
    physics does not depend on the reserved capacity, only on not overflowing it).
    ``chunk_checks`` check periods run between host synchronisations, where the batch
    stops early once every world is decided.
    """

    max_worlds: int = 960
    nconmax: int = 128
    njmax: int = 512
    overflow_retries: int = 2
    chunk_checks: int = 50
    device: str | None = None  # Warp device; None = Warp's default (cuda:0 when present)
    graph: bool = True  # replay one captured check period as a CUDA graph
    quiet: bool = True  # silence Warp's per-module compile messages (sets wp.config.quiet)


@dataclass
class BatchTiming:
    """Wall-clock breakdown of one batch (seconds)."""

    n_worlds: int
    build_s: float  # jitter, plans and MuJoCo compiles of the per-world scenes (host)
    upload_s: float  # put_model / put_data, per-world fields, first forward (loads forward kernels)
    first_period_s: float  # first check period eagerly (Warp JIT or kernel-cache load) + graph capture
    sim_s: float  # the remaining GPU simulation, including host synchronisation
    score_s: float  # host replay of the recorded series and result assembly
    sim_time: float  # simulated seconds (every world of the batch)
    steps: int  # physics steps per world
    nconmax: int = 0  # contacts reserved per world in the run that succeeded
    njmax: int = 0  # constraint rows reserved per world in the run that succeeded
    failed_attempts_s: float = 0.0  # wall time of earlier runs discarded for overflowing

    @property
    def total_s(self) -> float:
        parts = (self.build_s, self.upload_s, self.first_period_s, self.sim_s, self.score_s)
        return sum(parts) + self.failed_attempts_s


@dataclass
class WarpBatchResult(BatchResult):
    """:class:`~rtt.sim.exec_cpu.BatchResult` plus per-batch timing and solver diagnostics.

    ``total_steps`` counts the steps that decided the episodes (as on the CPU), while
    :attr:`world_steps` counts every step simulated, including worlds idling after
    their decision while the rest of their batch runs on.
    """

    batches: list[BatchTiming] = field(default_factory=list)
    iteration_capped_worlds: int = 0  # worlds whose solver hit its iteration cap at least once

    @property
    def world_steps(self) -> int:
        return sum(b.n_worlds * b.steps for b in self.batches)


# --------------------------------------------------------------------------- batched model


def _model_field_specs(model: mujoco.MjModel) -> tuple[frozenset[str], frozenset[str]]:
    """(batchable, shared) names of MJWarp ``Model`` array fields that mirror an MjModel field."""
    from mujoco_warp._src import types as mjw_types
    from mujoco_warp._src import warp_util

    batchable: set[str] = set()
    shared: set[str] = set()
    for f in dataclasses.fields(mjw_types.Model):
        if not warp_util.is_array_spec(f.type) or not isinstance(getattr(model, f.name, None), np.ndarray):
            continue
        (batchable if getattr(f.type, "shape", ())[:1] == ("*",) else shared).add(f.name)
    return frozenset(batchable), frozenset(shared)


def batch_key(job: EpisodeJob) -> Hashable:
    """Jobs with equal keys can share one batched model (same topology, solver and timing).

    Primitive objects of one family share a batch (their sizes are per-world fields);
    a mesh object's vertices are not, so each mesh object gets its own batch.
    """
    task = job.task or default_task()
    obj_key: Hashable = job.obj if isinstance(job.obj, objs.MeshObject) else job.obj.kind
    return (
        obj_key,
        task.physics,
        task.gripper,
        task.controller.control_period,
        task.controller.open_target,
        task.score.check_period,
        task.score.max_time,
    )


def stack_model_fields(models: Sequence[mujoco.MjModel]) -> dict[str, np.ndarray]:
    """Per-world stacks of every MJWarp-batchable field that differs between ``models``.

    Raises :class:`BatchIncompatible` when the models differ in a field MJWarp shares
    across worlds (topology, solver settings, collision filters, mesh data and so on).
    """
    ref = models[0]
    batchable, shared = _model_field_specs(ref)
    if not ref.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
        shared -= _SLEEP_ONLY_FIELDS
    for name in sorted(shared):
        a = getattr(ref, name)
        for i, m in enumerate(models[1:], start=1):
            b = getattr(m, name)
            if a.shape != b.shape or not np.array_equal(a, b):
                raise BatchIncompatible(f"world {i} differs from world 0 in shared model field {name!r}")
    out: dict[str, np.ndarray] = {}
    for name in sorted(batchable):
        a = getattr(ref, name)
        values = [getattr(m, name) for m in models]
        if any(v.shape != a.shape for v in values):
            raise BatchIncompatible(f"model field {name!r} changes shape between worlds")
        if all(np.array_equal(v, a) for v in values[1:]):
            continue
        if name in _TRANSFORMED_FIELDS:
            raise BatchIncompatible(f"model field {name!r} differs between worlds; MJWarp rewrites it")
        out[name] = np.stack(values)
    return out


@dataclass
class BatchedScene:
    """The compiled per-world scenes of one batch, ready for MJWarp.

    ``models[i]`` is world ``i``'s MjModel exactly as the CPU executor builds it (worlds
    with the same finger, object and task share one instance); ``batched`` holds the
    stacked per-world values of every batchable field that differs between worlds,
    including ``geom_friction`` with each episode's jittered object friction.
    """

    models: list[mujoco.MjModel]
    ids: SceneIds
    batched: dict[str, np.ndarray]
    meaninertia: np.ndarray  # (n_worlds,) MuJoCo's stat.meaninertia per world

    @property
    def reference(self) -> mujoco.MjModel:
        return self.models[0]

    @classmethod
    def build(cls, jobs: Sequence[EpisodeJob], frictions: Sequence[float]) -> BatchedScene:
        """Compile every distinct (finger, object, task) scene once and stack the per-world values."""
        cache: dict[Hashable, int] = {}
        unique: list[mujoco.MjModel] = []
        index: list[int] = []
        ids: SceneIds | None = None
        for job in jobs:
            task = job.task or default_task()
            key = (job.finger, job.obj, task)
            if key not in cache:
                model, world_ids = build_model(job.finger, job.obj, task)
                cache[key] = len(unique)
                unique.append(model)
                ids = ids or world_ids
            index.append(cache[key])
        if ids is None:
            raise ValueError("a batch needs at least one job")
        idx = np.asarray(index)
        batched = {name: values[idx] for name, values in stack_model_fields(unique).items()}
        friction = batched.get("geom_friction", np.stack([unique[i].geom_friction for i in idx]))
        friction[:, ids.object_geoms, 0] = np.asarray(frictions, dtype=np.float64)[:, None]
        if np.all(friction == unique[0].geom_friction):  # put_model's reference values already
            batched.pop("geom_friction", None)
        else:
            batched["geom_friction"] = friction
        return cls(
            models=[unique[i] for i in idx],
            ids=ids,
            batched=batched,
            meaninertia=np.array([unique[i].stat.meaninertia for i in idx]),
        )

    def model_with_friction(self, world: int) -> mujoco.MjModel:
        """A copy of world ``world``'s model carrying its jittered object friction (diagnostics)."""
        model = copy.copy(self.models[world])
        if "geom_friction" in self.batched:
            model.geom_friction[:] = self.batched["geom_friction"][world]
        return model


def _assign(dst: wp.array, host: np.ndarray) -> None:
    """Copy a host array into a Warp array with the same number of scalars (vector dtypes too)."""
    scalar = getattr(dst.dtype, "_wp_scalar_type_", dst.dtype)
    tail = tuple(getattr(dst.dtype, "_shape_", ()))
    src = np.asarray(host).reshape(tuple(dst.shape) + tail)
    dst.assign(np.ascontiguousarray(src, dtype=wp.dtype_to_numpy(scalar)))


# --------------------------------------------------------------------------- per-world episode setup


@dataclass(frozen=True)
class WorldSetup:
    """What run_episode derives from a job before stepping, for one world."""

    task: TaskConfig
    jitter: Jitter
    mu: float
    plan: GraspPlan
    actual: objs.Placement  # the object's true resting pose
    perturb_end: float  # nan without a perturbation
    min_success_time: float
    plan_row: np.ndarray  # (N_PLAN,) controller kernel inputs


def world_setup(job: EpisodeJob) -> WorldSetup:
    """Jitter, plan, true placement and perturbation timing, exactly as run_episode computes them."""
    task = job.task or default_task()
    jitter = sample_jitter(job.seed, task.jitter)
    plan = plan_grasp(job.finger, job.obj, job.placement, job.controller_setting, task)
    row = np.zeros(N_PLAN)
    row[[P_X, P_Y, P_YAW]] = plan.x, plan.y, plan.yaw
    row[[P_APPROACH, P_GRASP, P_LIFT]] = plan.approach_z, plan.grasp_z, plan.lift_z
    row[[P_T_DESCEND, P_T_CLOSE, P_T_LIFT, P_T_HOLD]] = plan.t_descend, plan.t_close, plan.t_lift, plan.t_hold
    row[[P_CLOSE_TIME, P_OPEN, P_CLOSE]] = plan.close_time, plan.open_target, plan.close_target
    row[P_SHAKE_DURATION] = 1.0  # amplitude 0: no shake
    perturb_end = math.nan
    min_success_time = 0.0
    p = job.perturbation
    if p is not None:
        start = plan.t_hold + p.delay
        perturb_end = start + p.duration
        min_success_time = perturb_end + task.robustness.post_perturb_hold
        if isinstance(p, Shake):
            row[[P_SHAKE_START, P_SHAKE_DURATION]] = start, p.duration
            row[[P_SHAKE_RAMP, P_SHAKE_AMPLITUDE, P_SHAKE_FREQUENCY]] = (
                p.ramp_cycles / p.frequency,
                p.amplitude,
                p.frequency,
            )
            row[P_SHAKE_DIR : P_SHAKE_DIR + 3] = axis_vector(p.axis, plan.yaw)
        elif isinstance(p, Push):
            cp = task.controller.control_period
            first = round(start / cp)
            row[[P_PUSH_FIRST, P_PUSH_END]] = first, first + max(1, round(p.duration / cp))
            wrench = p.wrench(plan.yaw)  # the force and its torque about the CoM (push at a height)
            row[P_PUSH_FORCE : P_PUSH_FORCE + 3] = wrench[:3]
            row[P_PUSH_TORQUE : P_PUSH_TORQUE + 3] = wrench[3:]
        else:
            raise TypeError(f"unknown perturbation {type(p).__name__}")
    return WorldSetup(
        task=task,
        jitter=jitter,
        mu=task.physics.friction * jitter.mu_scale,
        plan=plan,
        actual=job.placement.actual(jitter.dx, jitter.dy, jitter.dyaw),
        perturb_end=perturb_end,
        min_success_time=min_success_time,
        plan_row=row,
    )


# --------------------------------------------------------------------------- host-side scoring


def _col(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


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


class SeriesScorer:
    """Vectorised replay of :func:`rtt.sim.episode.run_episode`'s check loop over many worlds.

    Feed it one sample per world per check period (:meth:`feed`), in time order, with
    the time the CPU executor has at that check; a world ignores samples after its
    decision. :meth:`events` and :meth:`tracker` rebuild what
    :func:`rtt.sim.score.classify` needs.
    """

    def __init__(self, setups: Sequence[WorldSetup], weights: np.ndarray) -> None:
        n = len(setups)
        self.n = n
        self.setups = list(setups)
        tasks = [s.task for s in setups]
        sc = [t.score for t in tasks]
        self.success_lift = _col(s.success_lift for s in sc)
        self.success_hold = _col(s.success_hold for s in sc)
        self.max_time = _col(s.max_time for s in sc)
        self.min_success_time = _col(s.min_success_time for s in setups)
        self.pen_thr = _col(s.penetration_max for s in sc)
        self.ge_thr = _col(s.finger_table_force_max for s in sc)
        self.palm_thr = _col(s.palm_contact_force_max for s in sc)
        self.speed_cap = _col(s.speed_cap for s in sc)
        self.eject_dist = _col(s.eject_dist for s in sc)
        self.eject_speed = _col(s.eject_speed for s in sc)
        self.off_z = _col(t.physics.table_top_z - t.score.off_table_drop for t in tasks)
        self.fail_settle = _col(s.fail_settle_time for s in sc)
        self.press_ratio = _col(s.grasp_max_press_down_ratio for s in sc)
        self.min_squeeze = _col(t.score.grasp_min_squeeze_frac * t.gripper.grip_force / 2.0 for t in tasks)
        self.bounds = np.array(
            [[s.plan.t_descend, s.plan.t_close, s.plan.t_lift, s.plan.t_hold] for s in setups]
        )
        self.weight = np.asarray(weights, dtype=np.float64)

        nan = np.full(n, np.nan)
        self.stage = np.zeros(n, dtype=np.int64)
        self.decided = np.zeros(n, dtype=bool)
        self.decided_at = nan.copy()
        self.decided_check = np.full(n, -1, dtype=np.int64)
        self.z0 = nan.copy()
        self.lift = np.zeros(n)
        self.max_lift = np.zeros(n)
        self.hold_time = np.zeros(n)
        self.success_at = nan.copy()
        self.above_since = nan.copy()
        self.grasped = np.zeros(n, dtype=bool)
        self.metrics = {k: nan.copy() for k in _PHASE_METRICS}
        self.pen_max = np.zeros(n)
        self.jaw_pen_max = np.zeros(n)
        self.ge_force_max = np.zeros(n)
        self.palm_force_max = np.zeros(n)
        self.carry_palm_force_max = np.zeros(n)
        self.speed_max = np.zeros(n)
        self.drift_max = np.zeros(n)
        self.close_ref = np.full((n, 2), np.nan)
        nf = len(FLAG_ORDER)
        self.flag_t = np.full((n, nf), np.nan)
        self.flag_phase = np.zeros((n, nf), dtype=np.int64)
        self.flag_env = np.zeros((n, nf), dtype=bool)
        self.flag_detail: list[dict[int, str]] = [{} for _ in range(n)]

    @property
    def all_decided(self) -> bool:
        return bool(self.decided.all())

    def _flag(
        self,
        name: str,
        mask: np.ndarray,
        t: float,
        phase: np.ndarray,
        detail: Callable[[int], str],
        env: np.ndarray | None = None,
    ) -> None:
        j = _FI[name]
        new = mask & np.isnan(self.flag_t[:, j])
        if not new.any():
            return
        self.flag_t[new, j] = t
        self.flag_phase[new, j] = phase[new]
        if env is not None:
            self.flag_env[new, j] = env[new]
        for w in np.flatnonzero(new):
            self.flag_detail[w][j] = detail(int(w))

    def feed(self, t: float, check: int, f: np.ndarray) -> None:
        """Consume one sample per world (``f``: ``(n, N_FEATURES)`` float64) taken at time ``t``."""
        act = ~self.decided
        if not act.any():
            return
        phase = (t >= self.bounds).sum(axis=1)
        z = f[:, F_OBJ_Z]

        # Phase-entry bookkeeping, in run_episode's order.
        enter = act & (phase > self.stage)
        if enter.any():
            e = enter & (self.stage < Phase.DESCEND) & (phase >= Phase.DESCEND)
            self.z0[e] = z[e]
            self.metrics["rest_table_support"][e] = f[e, F_OBJ_ENV] / self.weight[e]
            self.metrics["rest_finger_force"][e] = f[e, F_FN_L] + f[e, F_FN_R]
            e = enter & (self.stage < Phase.CLOSE) & (phase >= Phase.CLOSE)
            self.metrics["preclose_table_support"][e] = f[e, F_OBJ_ENV] / self.weight[e]
            self.metrics["preclose_finger_force"][e] = f[e, F_FN_L] + f[e, F_FN_R]
            e = enter & (self.stage < Phase.LIFT) & (phase >= Phase.LIFT)
            if e.any():
                palm_y = f[:, F_PALM_Y : F_PALM_Y + 3]
                left, right = f[:, F_SQ_L : F_SQ_L + 3], f[:, F_SQ_R : F_SQ_R + 3]
                sq_l = np.maximum(0.0, -(left * palm_y).sum(axis=1)) + 0.0
                sq_r = np.maximum(0.0, (right * palm_y).sum(axis=1)) + 0.0
                pd_l = np.maximum(0.0, -left[:, 2]) + 0.0
                pd_r = np.maximum(0.0, -right[:, 2]) + 0.0
                for key, val in (
                    ("squeeze_left", sq_l),
                    ("squeeze_right", sq_r),
                    ("press_down_left", pd_l),
                    ("press_down_right", pd_r),
                ):
                    self.metrics[key][e] = val[e]
                ok = (sq_l >= self.min_squeeze) & (pd_l <= self.press_ratio * sq_l)
                ok &= (sq_r >= self.min_squeeze) & (pd_r <= self.press_ratio * sq_r)
                self.grasped[e] = ok[e]
            self.stage[enter] = phase[enter]

        # PhysicsChecks.update, flag by flag in its order.
        pen, pen_ge = f[:, F_PEN], f[:, F_PEN_GE]
        self.jaw_pen_max[act] = np.maximum(self.jaw_pen_max[act], f[act, F_PEN_JAW])
        self.pen_max[act] = np.maximum(self.pen_max[act], pen[act])
        self._flag(
            FLAG_PENETRATION,
            act & (pen > self.pen_thr),
            t,
            phase,
            lambda w: f"{pen[w] * 1000:.2f} mm" + (" gripper/env" if pen_ge[w] >= pen[w] else ""),
            env=(pen_ge >= pen) & (pen_ge > 0.0),
        )
        ge = f[:, F_GE_FORCE]
        self.ge_force_max[act] = np.maximum(self.ge_force_max[act], ge[act])
        self._flag(
            FLAG_GRIPPER_ENV,
            act & (ge > self.ge_thr),
            t,
            phase,
            lambda w: f"{ge[w]:.1f} N",
            env=np.ones(self.n, dtype=bool),
        )
        palm = f[:, F_PALM_OBJ]
        self.palm_force_max[act] = np.maximum(self.palm_force_max[act], palm[act])
        carrying = act & (phase >= Phase.LIFT)
        self.carry_palm_force_max[carrying] = np.maximum(self.carry_palm_force_max[carrying], palm[carrying])
        self._flag(
            FLAG_PALM_CONTACT, carrying & (palm > self.palm_thr), t, phase, lambda w: f"{palm[w]:.2f} N"
        )
        speed = f[:, F_SPEED]
        self.speed_max[act] = np.maximum(self.speed_max[act], speed[act])
        self._flag(FLAG_SPEED_CAP, act & (speed > self.speed_cap), t, phase, lambda w: f"{speed[w]:.2f} m/s")
        self._flag(FLAG_OFF_TABLE, act & (z < self.off_z), t, phase, lambda w: "")
        closing = act & (phase == Phase.CLOSE)
        if closing.any():
            xy = f[:, F_OBJ_X : F_OBJ_Y + 1]
            unset = closing & np.isnan(self.close_ref[:, 0])
            self.close_ref[unset] = xy[unset]
            drift = np.where(closing, np.linalg.norm(xy - self.close_ref, axis=1), 0.0)
            self.drift_max[closing] = np.maximum(self.drift_max[closing], drift[closing])
            far = closing & (drift > self.eject_dist)
            self._flag(FLAG_EJECTION, far, t, phase, lambda w: f"drift {drift[w] * 1000:.1f} mm")
            fast = closing & ~far & (speed > self.eject_speed)
            self._flag(FLAG_EJECTION, fast, t, phase, lambda w: f"speed {speed[w]:.2f} m/s")

        # LiftTracker.update.
        ref = act & ~np.isnan(self.z0)
        self.lift[ref] = z[ref] - self.z0[ref]
        self.max_lift[ref] = np.maximum(self.max_lift[ref], self.lift[ref])
        above = ref & (self.lift >= self.success_lift)
        self.above_since[above & np.isnan(self.above_since)] = t
        held = t - self.above_since
        self.hold_time[above] = np.maximum(self.hold_time[above], held[above])
        win = (
            above
            & np.isnan(self.success_at)
            & (held >= self.success_hold)
            & (t >= self.min_success_time)
            & (t <= self.max_time)
        )
        self.success_at[win] = t
        self.above_since[ref & ~above] = np.nan

        # The episode's stopping rule.
        is_above = ~np.isnan(self.above_since)
        settled = (t >= self.bounds[:, 3] + self.fail_settle) & ~is_above
        out_of_time = ~is_above & (t > self.max_time - self.success_hold)
        flagged = ~np.isnan(self.flag_t).all(axis=1)
        newly = act & (flagged | ~np.isnan(self.success_at) | settled | out_of_time)
        self.decided[newly] = True
        self.decided_at[newly] = t
        self.decided_check[newly] = check

    def events(self, w: int) -> list[CheckEvent]:
        """World ``w``'s check events in the order PhysicsChecks would have raised them."""
        raised = np.flatnonzero(~np.isnan(self.flag_t[w]))
        return [
            CheckEvent(
                flag=FLAG_ORDER[j],
                t=float(self.flag_t[w, j]),
                phase=Phase(int(self.flag_phase[w, j])),
                detail=self.flag_detail[w].get(int(j), ""),
                gripper_env=bool(self.flag_env[w, j]),
            )
            for j in sorted(raised, key=lambda j: (self.flag_t[w, j], j))
        ]

    def tracker(self, w: int) -> LiftTracker:
        """World ``w``'s final :class:`~rtt.sim.score.LiftTracker` state."""

        def opt(v: float) -> float | None:
            return None if math.isnan(v) else float(v)

        return LiftTracker(
            cfg=self.setups[w].task.score,
            z0=opt(self.z0[w]),
            lift=float(self.lift[w]),
            max_lift=float(self.max_lift[w]),
            hold_time=float(self.hold_time[w]),
            success_at=opt(self.success_at[w]),
            min_success_time=float(self.min_success_time[w]),
            _above_since=opt(self.above_since[w]),
        )


# --------------------------------------------------------------------------- the executor


def tick_times(timestep: float, n_steps: int) -> np.ndarray:
    """``data.time`` after each of ``n_steps`` CPU steps (MuJoCo adds ``timestep`` in float64)."""
    times = np.empty(n_steps + 1)
    t = 0.0
    times[0] = t
    for i in range(n_steps):
        t += timestep
        times[i + 1] = t
    return times


def _overflow_names(bits: int) -> list[str]:
    return [o.name or "?" for o in _mjw().OverflowType if o.value and bits & o.value]


class WarpBatch:
    """One batch of same-topology jobs on the GPU: model and data uploaded, ready to step.

    :meth:`advance` simulates whole check periods (recording one sample per world at
    the end of each), :meth:`samples` copies recorded samples to the host and
    :meth:`check_overflow` validates the run. :func:`run_jobs_mjwarp` drives it to
    completion; diagnostics such as :func:`contact_widths` step it part way.
    """

    def __init__(self, jobs: Sequence[EpisodeJob], cfg: WarpConfig) -> None:
        mjw = _mjw()
        if any(job.record for job in jobs):
            raise ValueError("the mjwarp executor does not record replays; run recorded episodes on cpu")
        t0 = time.perf_counter()
        self.jobs = list(jobs)
        self.cfg = cfg
        self.n = n = len(jobs)
        self.setups = [world_setup(job) for job in jobs]
        self.scene = BatchedScene.build(jobs, [s.mu for s in self.setups])
        ids, ref = self.scene.ids, self.scene.reference
        self.ids = ids
        qpos = np.tile(ref.qpos0, (n, 1))
        for w, (job, s) in enumerate(zip(jobs, self.setups, strict=True)):
            pos, quat = objs.rest_pose(job.obj, s.actual, s.task.physics.table_top_z)
            qpos[w, ids.palm_qpos] = s.plan.palm_start
            qpos[w, ids.finger_qpos] = 0.0
            qpos[w, ids.object_qpos : ids.object_qpos + 3] = pos
            qpos[w, ids.object_qpos + 3 : ids.object_qpos + 7] = quat
        gravity = float(np.linalg.norm(ref.opt.gravity))
        self.weights = np.array([m.body_mass[ids.object_body] * gravity for m in self.scene.models])
        plan_rows = np.stack([s.plan_row for s in self.setups])
        act_ids = np.array([*ids.palm_actuators, ids.grip_actuator], dtype=np.int32)
        ctrl0 = np.zeros((n, ref.nu))
        ctrl0[:, act_ids] = plan_rows[:, [P_X, P_Y, P_APPROACH, P_YAW, P_OPEN]]

        task0 = self.setups[0].task
        cp = task0.controller.control_period
        self.steps_per_tick = _ratio(cp, ref.opt.timestep, "control_period")
        self.ticks_per_check = _ratio(task0.score.check_period, cp, "check_period")
        self.n_checks = round(task0.score.max_time / cp) // self.ticks_per_check
        self.times = tick_times(ref.opt.timestep, self.n_checks * self.ticks_per_check * self.steps_per_tick)
        self.build_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.device = wp.get_device(cfg.device)
        with wp.ScopedDevice(self.device):
            m = mjw.put_model(ref, batch_sizes=dict.fromkeys(self.scene.batched, n))
            for name, values in self.scene.batched.items():
                _assign(getattr(m, name), values)
            if np.any(self.scene.meaninertia != self.scene.meaninertia[0]):
                m.stat.meaninertia = wp.array(self.scene.meaninertia.astype(np.float32), dtype=wp.float32)
            m.opt.warn_overflow = 0  # overflows are read from d.overflow (check_overflow)
            d = mjw.put_data(ref, mujoco.MjData(ref), nworld=n, nconmax=cfg.nconmax, njmax=cfg.njmax)
            _assign(d.qpos, qpos)
            _assign(d.ctrl, ctrl0)
            self.m, self.d = m, d
            self._plan = wp.array(plan_rows, dtype=wp.float64)
            self._times = wp.array(self.times, dtype=wp.float64)
            self._act = wp.array(act_ids, dtype=wp.int32)
            self._group = wp.array(ids.geom_group.astype(np.int32), dtype=wp.int32)
            self._tick = wp.zeros(1, dtype=wp.int32)
            self._check = wp.zeros(1, dtype=wp.int32)
            self._acc_max = wp.zeros((n, N_MAX), dtype=wp.float32)
            self._acc_sum = wp.zeros((n, N_SUM), dtype=wp.float32)
            self._hist = wp.zeros((self.n_checks, n, N_FEATURES), dtype=wp.float32)
            self._pyramidal = int(ref.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL)
            mjw.forward(m, d)
            wp.synchronize_device(self.device)
        self.upload_s = time.perf_counter() - t1
        self.first_period_s = 0.0
        self.done = 0  # check periods simulated
        self._graph: Any = None

    def _period(self) -> None:
        """Launch one check period: ticks of (controller, physics steps), then one sample."""
        mjw, d, n = _mjw(), self.d, self.n
        for _ in range(self.ticks_per_check):
            wp.launch(
                _ctrl_kernel,
                dim=n,
                inputs=[
                    self._tick,
                    self._times,
                    self.steps_per_tick,
                    self._plan,
                    self._act,
                    self.ids.object_body,
                    d.ctrl,
                    d.xfrc_applied,
                ],
            )
            for _ in range(self.steps_per_tick):
                mjw.step(self.m, d)
            wp.launch(_increment_kernel, dim=1, inputs=[self._tick])
        self._acc_max.zero_()
        self._acc_sum.zero_()
        con = d.contact
        wp.launch(
            _contact_kernel,
            dim=d.naconmax,
            inputs=[
                d.nacon,
                con.dist,
                con.frame,
                con.geom,
                con.dim,
                con.efc_address,
                con.worldid,
                d.efc.force,
                self._group,
                self._pyramidal,
                self._acc_max,
                self._acc_sum,
            ],
        )
        wp.launch(
            _record_kernel,
            dim=n,
            inputs=[
                self._check,
                self._acc_max,
                self._acc_sum,
                d.xpos,
                d.xmat,
                d.qvel,
                d.nefc,
                d.overflow,
                self.ids.object_body,
                self.ids.palm_body,
                self.ids.object_dof,
                self._hist,
            ],
        )
        wp.launch(_increment_kernel, dim=1, inputs=[self._check])

    def advance(self, periods: int) -> int:
        """Simulate up to ``periods`` more check periods (never past max_time); returns how many ran.

        With ``cfg.graph`` the first period runs eagerly (Warp JIT and module loads cannot
        be captured), then one period is captured as a CUDA graph and replayed.
        """
        periods = min(periods, self.n_checks - self.done)
        with wp.ScopedDevice(self.device):
            for _ in range(periods):
                if not self.cfg.graph:
                    self._period()
                elif self._graph is None:
                    ts = time.perf_counter()
                    self._period()
                    wp.synchronize_device(self.device)
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._period()
                    self._graph = capture.graph
                    self.first_period_s = time.perf_counter() - ts
                else:
                    wp.capture_launch(self._graph)
                self.done += 1
        return periods

    def time_after(self, check: int) -> float:
        """The CPU executor's ``data.time`` at check ``check`` (0-based)."""
        return float(self.times[(check + 1) * self.ticks_per_check * self.steps_per_tick])

    def samples(self, start: int, stop: int) -> np.ndarray:
        """Recorded samples of checks ``[start, stop)``: ``(stop - start, n, N_FEATURES)`` float64."""
        with wp.ScopedDevice(self.device):
            return self._hist[start:stop].numpy().astype(np.float64)

    def contacts(self) -> dict[str, np.ndarray]:
        """Host copy of the current active contacts (world, geoms, position, depth, normal force)."""
        with wp.ScopedDevice(self.device):
            d = self.d
            nacon = int(d.nacon.numpy()[0])
            efc = d.contact.efc_address.numpy()[:nacon, 0]
            world = d.contact.worldid.numpy()[:nacon]
            force = d.efc.force.numpy()
            active = efc >= 0
            return {
                "world": world[active],
                "geom": d.contact.geom.numpy()[:nacon][active],
                "pos": d.contact.pos.numpy()[:nacon][active].astype(np.float64),
                "dist": d.contact.dist.numpy()[:nacon][active].astype(np.float64),
                "normal": force[world[active], efc[active]].astype(np.float64),
            }

    def check_overflow(self, decided_check: np.ndarray) -> int:
        """Raise :class:`CapacityOverflow` if an overflow could have changed any outcome.

        A shared contact-buffer overflow is always fatal. A world that overflows its own
        constraint rows only spoils itself, so that is fatal only at or before the check
        that decided the world (``decided_check``, -1 if undecided); afterwards nothing
        it does is used. Returns the number of worlds whose solver hit its iteration cap.
        """
        mjw = _mjw()
        ot = mjw.OverflowType

        def mask(names: Sequence[str]) -> int:
            return sum(int(getattr(ot, name, 0)) for name in names)

        if self.done == 0:
            return 0
        bits = self.samples(0, self.done)[:, :, F_OVERFLOW].astype(np.int64)  # (checks, worlds), sticky
        last = np.where(decided_check >= 0, decided_check, self.done - 1)
        upto = bits[last, np.arange(self.n)]
        fatal = ((bits[-1] & mask(_SHARED_OVERFLOW)) != 0) | ((upto & mask(_WORLD_OVERFLOW)) != 0)
        if fatal.any():
            bad = int(np.flatnonzero(fatal)[0])
            max_nefc = int(self.samples(0, self.done)[:, :, F_NEFC].max())
            raise CapacityOverflow(
                f"MJWarp overflow in {int(fatal.sum())} of {self.n} worlds (first: world {bad}, "
                f"{_overflow_names(int(bits[-1, bad]))}, max nefc seen {max_nefc}); raise "
                f"WarpConfig.nconmax (now {self.cfg.nconmax}) or njmax (now {self.cfg.njmax})"
            )
        return int(((bits[-1] & int(ot.ITERATIONS)) != 0).sum())


def _run_batch(jobs: Sequence[EpisodeJob], cfg: WarpConfig) -> tuple[list[EpisodeResult], BatchTiming, int]:
    """Simulate one batch to completion; returns results, timing and iteration-capped worlds."""
    t0 = time.perf_counter()
    batch = WarpBatch(jobs, cfg)
    scorer = SeriesScorer(batch.setups, batch.weights)
    replay_s = 0.0
    t2 = time.perf_counter()
    fed = 0
    while True:
        if batch.done > fed:
            chunk = batch.samples(fed, batch.done)
            ts = time.perf_counter()
            for i, sample in enumerate(chunk):
                scorer.feed(batch.time_after(fed + i), fed + i, sample)
            replay_s += time.perf_counter() - ts
            fed = batch.done
        if batch.done >= batch.n_checks or scorer.all_decided:
            break
        batch.advance(cfg.chunk_checks if batch.done else 1)
    capped = batch.check_overflow(scorer.decided_check)
    t3 = time.perf_counter()

    steps_per_check = batch.ticks_per_check * batch.steps_per_tick
    end_time = batch.time_after(batch.done - 1) if batch.done else 0.0
    results = []
    for w, (job, s) in enumerate(zip(jobs, batch.setups, strict=True)):
        checks = int(scorer.decided_check[w]) + 1 if scorer.decided[w] else batch.done
        results.append(
            _episode_result(
                job, s, batch.scene.models[w], batch.ids, scorer, w, checks * steps_per_check, end_time
            )
        )
    t4 = time.perf_counter()
    for r in results:
        r.wall_time_s = (t4 - t0) / batch.n  # amortised share of the batch's wall time
    timing = BatchTiming(
        n_worlds=batch.n,
        build_s=batch.build_s,
        upload_s=batch.upload_s,
        first_period_s=batch.first_period_s,
        sim_s=t3 - t2 - batch.first_period_s - replay_s,
        score_s=replay_s + (t4 - t3),
        sim_time=end_time,
        steps=batch.done * steps_per_check,
        nconmax=cfg.nconmax,
        njmax=cfg.njmax,
    )
    return results, timing, capped


def _episode_result(
    job: EpisodeJob,
    s: WorldSetup,
    model: mujoco.MjModel,
    ids: SceneIds,
    scorer: SeriesScorer,
    w: int,
    n_steps: int,
    end_time: float,
) -> EpisodeResult:
    """Assemble world ``w``'s EpisodeResult with the same fields and metrics as run_episode."""
    events = scorer.events(w)
    tracker = scorer.tracker(w)
    grasped = bool(scorer.grasped[w])
    success, mode = classify(events, tracker, grasped)
    metrics = {k: float(v[w]) for k, v in scorer.metrics.items()}
    metrics["decided_at"] = float(scorer.decided_at[w]) if scorer.decided[w] else end_time
    p = job.placement
    metrics.update(
        mu=s.mu,
        jitter_dx=s.jitter.dx,
        jitter_dy=s.jitter.dy,
        jitter_dyaw_deg=math.degrees(s.jitter.dyaw),
        object_mass=float(model.body_mass[ids.object_body]),
        finger_mass=finger_mass_properties(job.finger).mass,
        grasp_fraction=grasp_fraction(job.controller_setting, s.task),
        tip_height=s.plan.tip_height,
        reach_floor=s.plan.reach_floor,
        palm_gap=s.plan.palm_gap,
        err_x=p.err_x,
        err_y=p.err_y,
        err_yaw_deg=math.degrees(p.err_yaw),
        perturb_end=s.perturb_end,
        z0=tracker.z0 if tracker.z0 is not None else float("nan"),
        final_lift=tracker.lift,
        success_time=tracker.success_at if tracker.success_at is not None else float("nan"),
        grasped=float(grasped),
        gripper_env_force_max=float(scorer.ge_force_max[w]),
        penetration_max=float(scorer.pen_max[w]),
        jaw_penetration_max=float(scorer.jaw_pen_max[w]),
        object_speed_max=float(scorer.speed_max[w]),
        close_xy_drift_max=float(scorer.drift_max[w]),
        palm_object_force_max=float(scorer.palm_force_max[w]),
        carry_palm_force_max=float(scorer.carry_palm_force_max[w]),
    )
    return EpisodeResult(
        success=success,
        failure_mode=mode,
        max_lift=tracker.max_lift,
        hold_time=tracker.hold_time,
        metrics=metrics,
        flags=tuple(f"{e.flag}@{e.t:.2f}s:{e.phase.name.lower()} {e.detail}".rstrip() for e in events),
        seed=job.seed,
        controller_setting=job.controller_setting,
        n_steps=n_steps,
        wall_time_s=0.0,
    )


def run_jobs_mjwarp(jobs: Sequence[EpisodeJob], config: WarpConfig | None = None) -> WarpBatchResult:
    """Run ``jobs`` as batched MJWarp simulations; results come back in job order.

    Jobs are grouped by :func:`batch_key` and every group is split into batches of at
    most ``config.max_worlds`` worlds.
    """
    cfg = config or WarpConfig()
    if cfg.quiet:
        wp.config.quiet = True
    t0 = time.perf_counter()
    groups: dict[Hashable, list[int]] = {}
    for i, job in enumerate(jobs):
        groups.setdefault(batch_key(job), []).append(i)
    results: list[EpisodeResult | None] = [None] * len(jobs)
    timings: list[BatchTiming] = []
    capped = 0
    for idx in groups.values():
        for start in range(0, len(idx), cfg.max_worlds):
            part = idx[start : start + cfg.max_worlds]
            batch_cfg = cfg
            t_try = time.perf_counter()
            for attempt in range(cfg.overflow_retries + 1):
                try:
                    res, timing, cap = _run_batch([jobs[i] for i in part], batch_cfg)
                    break
                except CapacityOverflow:
                    if attempt == cfg.overflow_retries:
                        raise
                    batch_cfg = dataclasses.replace(
                        batch_cfg, nconmax=2 * batch_cfg.nconmax, njmax=2 * batch_cfg.njmax
                    )
            timing.failed_attempts_s = max(0.0, time.perf_counter() - t_try - timing.total_s)
            for i, r in zip(part, res, strict=True):
                results[i] = r
            timings.append(timing)
            capped += cap
    done = [r for r in results if r is not None]
    if len(done) != len(jobs):
        raise RuntimeError("internal error: a job has no result")
    return WarpBatchResult(
        results=done,
        wall_time_s=time.perf_counter() - t0,
        total_steps=sum(r.n_steps for r in done),
        workers=1,
        batches=timings,
        iteration_capped_worlds=capped,
    )


def run_jobs(jobs: Sequence[EpisodeJob], workers: int | None = None) -> WarpBatchResult:
    """The ``"mjwarp"`` executor for :mod:`rtt.sim.evaluate` (``workers`` is ignored: one GPU)."""
    del workers
    return run_jobs_mjwarp(jobs)


# --------------------------------------------------------------------------- spike benchmark (gate G3, plan 3.6)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTRANTS = REPO_ROOT / "configs" / "entrants.yaml"


def entrant_fingers(path: str | Path = DEFAULT_ENTRANTS) -> dict[str, Any]:
    """The manual entrants' fingers, keyed by entrant name, in file order."""
    from rtt.design.generate import generate_finger
    from rtt.design.schema import load_entrants

    return {(d.name or d.design_id()): generate_finger(d) for d in load_entrants(path)}


def parity_jobs(
    task: TaskConfig | None = None, seeds_per_condition: int = 2, entrants: str | Path = DEFAULT_ENTRANTS
) -> tuple[list[EpisodeJob], list[str]]:
    """Manual entrants x training suite x controller settings x jitter seeds, and each job's entrant.

    The first seed of each condition is the suite's own; further seeds are derived the
    same way (``derive_seed(base_seed, condition_id, rep)``). With 4 entrants, 12
    conditions, 3 settings and 2 seeds this is 288 episodes.
    """
    from rtt.sim.conditions import derive_seed, load_suite_spec, training_suite

    task = task or default_task()
    spec = load_suite_spec()
    suite = training_suite(spec)
    jobs: list[EpisodeJob] = []
    labels: list[str] = []
    for name, finger in entrant_fingers(entrants).items():
        for cond in suite:
            for k in range(len(task.controller.grasp_height_fracs)):
                for rep in range(seeds_per_condition):
                    seed = derive_seed(spec.base_seed, cond.condition_id, rep)
                    jobs.append(EpisodeJob(finger, cond.obj, cond.placement, k, seed, task=task))
                    labels.append(name)
    return jobs, labels


def sobol_jobs(n: int, family: str, seed: int = 0, task: TaskConfig | None = None) -> list[EpisodeJob]:
    """``n`` training-suite episodes of one object family on valid Sobol designs (round-like batches)."""
    from rtt.design.generate import generate_finger
    from rtt.search.random_search import DesignSampler
    from rtt.sim.conditions import training_suite

    task = task or default_task()
    suite = training_suite()
    conds = [c for c in suite if c.family == family]
    settings = range(len(task.controller.grasp_height_fracs))
    per_design = len(conds) * len(settings)
    designs = DesignSampler(seed, suite.gripper_task(task)).sample(-(-n // per_design))
    jobs = [
        EpisodeJob(generate_finger(d), c.obj, c.placement, k, c.seeds[0], task=task)
        for d in designs
        for c in conds
        for k in settings
    ]
    return jobs[:n]


def outcome(result: EpisodeResult) -> tuple[bool, str | None]:
    """The flag pair compared across executors: success and failure mode."""
    return result.success, result.failure_mode


def agreement(a: Sequence[EpisodeResult], b: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Agreement between two equally ordered result lists.

    The G3 criteria (plan sections 3.5 and 3.6) are stated on the success flag
    (``success_agreement``, ``success_mismatches``); the stricter outcome agreement
    (success and failure mode, ``mismatches``) is reported alongside.
    """
    same = [outcome(x) == outcome(y) for x, y in zip(a, b, strict=True)]
    success_same = [x.success == y.success for x, y in zip(a, b, strict=True)]
    return {
        "episodes": len(same),
        "outcome_agreement": sum(same) / len(same),
        "success_agreement": sum(success_same) / len(same),
        "mismatches": len(same) - sum(same),
        "success_mismatches": len(same) - sum(success_same),
    }


def entrant_scores(
    labels: Sequence[str], jobs: Sequence[EpisodeJob], results: Sequence[EpisodeResult]
) -> dict[str, float]:
    """Each entrant's score: its best controller setting's mean success (the plan's design score)."""
    by: dict[tuple[str, int], list[bool]] = {}
    for label, job, res in zip(labels, jobs, results, strict=True):
        by.setdefault((label, job.controller_setting), []).append(res.success)
    scores: dict[str, float] = {}
    for (label, _k), hits in by.items():
        scores[label] = max(scores.get(label, 0.0), sum(hits) / len(hits))
    return scores


def ranking(scores: dict[str, float]) -> list[list[str]]:
    """Entrants grouped by equal score, best first (ties share a rank)."""
    out: list[list[str]] = []
    for value in sorted(set(scores.values()), reverse=True):
        out.append(sorted(k for k, v in scores.items() if v == value))
    return out


def contact_widths(
    fingers: Sequence[Any], obj: objs.ObjectSpec, task: TaskConfig | None = None, setting: int = 1
) -> dict[str, Any]:
    """Criterion (a): per-world finger geometry inside one batched model.

    Each finger squeezes ``obj`` (centred, no jitter) in its own world of ONE batched
    MJWarp model; at the start of LIFT the extent along the finger-width axis (world x)
    of the left finger's contacts with the object is measured, and the same quantity
    is measured on the CPU executor's model. Also checks that the per-world
    ``geom_aabb`` / ``geom_rbound`` uploaded to MJWarp equal both the MuJoCo compiler's
    values and the box formula (aabb = (0, half sizes), rbound = |half sizes|).
    """
    task = (task or default_task()).with_overrides(jitter={"pos": 0.0, "yaw_deg": 0.0, "mu_frac": 0.0})
    jobs = [EpisodeJob(f, obj, objs.Placement(), setting, 0, task=task) for f in fingers]
    batch = WarpBatch(jobs, WarpConfig(graph=False))
    ids = batch.ids
    left = np.flatnonzero(ids.geom_group == GROUP_FINGER_LEFT)
    lift_check = round(batch.setups[0].plan.t_lift / task.score.check_period) - 1
    batch.advance(lift_check + 1)
    con = batch.contacts()
    gpu_widths, cpu_widths = [], []
    for w, job in enumerate(jobs):
        pick = (
            (con["world"] == w)
            & np.isin(con["geom"], left).any(axis=1)
            & (ids.geom_group[con["geom"]] == GROUP_OBJECT).any(axis=1)
        )
        xs = con["pos"][pick, 0]
        gpu_widths.append(float(xs.max() - xs.min()) if xs.size else 0.0)
        cpu_widths.append(
            _cpu_contact_width(job, ids, batch.steps_per_tick * batch.ticks_per_check * (lift_check + 1))
        )
    with wp.ScopedDevice(batch.device):
        aabb = batch.m.geom_aabb.numpy().reshape(-1, batch.scene.reference.ngeom, 6)
        rbound = batch.m.geom_rbound.numpy().reshape(-1, batch.scene.reference.ngeom)
    aabb_ok = rbound_ok = formula_ok = True
    for w, model in enumerate(batch.scene.models):
        a = aabb[w % aabb.shape[0]]
        r = rbound[w % rbound.shape[0]]
        aabb_ok &= bool(np.allclose(a, model.geom_aabb, atol=1e-7))
        rbound_ok &= bool(np.allclose(r, model.geom_rbound, atol=1e-7))
        half = model.geom_size[left]
        formula_ok &= bool(
            np.allclose(a[left, 3:], half, atol=1e-7) and np.allclose(a[left, :3], 0.0, atol=1e-7)
        )
        formula_ok &= bool(np.allclose(r[left], np.linalg.norm(half, axis=1), rtol=1e-6))
    return {
        "gpu_widths_mm": [1000 * x for x in gpu_widths],
        "cpu_widths_mm": [1000 * x for x in cpu_widths],
        "aabb_per_world_matches_compiler": aabb_ok,
        "rbound_per_world_matches_compiler": rbound_ok,
        "aabb_rbound_match_box_formula": formula_ok,
        "batched_fields": sorted(batch.scene.batched),
    }


def _cpu_contact_width(job: EpisodeJob, ids: SceneIds, n_steps: int) -> float:
    """The left finger's contact extent along world x after ``n_steps`` CPU steps (same controller)."""
    from rtt.sim.controller import GraspController

    s = world_setup(job)
    model, _ = build_model(job.finger, job.obj, s.task)
    model.geom_friction[ids.object_geoms, 0] = s.mu
    data = mujoco.MjData(model)
    pos, quat = objs.rest_pose(job.obj, s.actual, s.task.physics.table_top_z)
    data.qpos[ids.palm_qpos] = s.plan.palm_start
    data.qpos[ids.object_qpos : ids.object_qpos + 3] = pos
    data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = quat
    controller = GraspController(s.plan)
    data.ctrl[:] = controller.command(0.0)
    mujoco.mj_forward(model, data)
    spt = _ratio(s.task.controller.control_period, model.opt.timestep, "control_period")
    for _ in range(n_steps // spt):
        data.ctrl[:] = controller.command(data.time)
        mujoco.mj_step(model, data, nstep=spt)
    xs = [
        data.contact[i].pos[0]
        for i in range(data.ncon)
        if data.contact[i].efc_address >= 0
        and GROUP_FINGER_LEFT in ids.geom_group[data.contact[i].geom]
        and GROUP_OBJECT in ids.geom_group[data.contact[i].geom]
    ]
    return float(max(xs) - min(xs)) if xs else 0.0


def _timing_dict(res: WarpBatchResult) -> dict[str, Any]:
    return {
        "wall_s": res.wall_time_s,
        "episodes": len(res.results),
        "episodes_per_s": res.episodes_per_sec,
        "decided_steps_per_s": res.steps_per_sec,
        "world_steps_per_s": res.world_steps / res.wall_time_s if res.wall_time_s else 0.0,
        "iteration_capped_worlds": res.iteration_capped_worlds,
        "batches": [dataclasses.asdict(b) | {"total_s": b.total_s} for b in res.batches],
    }


def _cpu_run(jobs: Sequence[EpisodeJob], workers: int) -> tuple[list[EpisodeResult], dict[str, Any]]:
    from rtt.sim.exec_cpu import run_jobs as run_jobs_cpu

    res = run_jobs_cpu(jobs, workers=workers)
    return res.results, {
        "wall_s": res.wall_time_s,
        "episodes": len(res.results),
        "episodes_per_s": res.episodes_per_sec,
        "steps_per_s": res.steps_per_sec,
        "workers": res.workers,
    }


def bench_geometry(task: TaskConfig) -> dict[str, Any]:
    """(a) Two straight fingers, 10 mm and 30 mm wide, squeeze a 60 mm (x) box in one batch."""
    from rtt.design.generate import generate_finger
    from rtt.design.schema import load_entrants

    base = next(iter(load_entrants(DEFAULT_ENTRANTS)))
    fingers = [generate_finger(base.model_copy(update={"finger_width_mm": w})) for w in (10.0, 30.0)]
    out = contact_widths(fingers, objs.ObjectSpec("box", (0.06, 0.04, 0.10)), task)
    out["finger_widths_mm"] = [10.0, 30.0]
    g = out["gpu_widths_mm"]
    out["pass"] = bool(
        abs(g[0] - 10.0) < 1.0
        and abs(g[1] - 30.0) < 1.0
        and np.allclose(g, out["cpu_widths_mm"], atol=0.5)
        and out["aabb_per_world_matches_compiler"]
        and out["rbound_per_world_matches_compiler"]
        and out["aabb_rbound_match_box_formula"]
    )
    return out


def bench_parity(
    task: TaskConfig, workers: int, seeds: int = 2
) -> tuple[dict[str, Any], list[EpisodeResult]]:
    """(b) The parity set on both executors: outcome agreement and the entrants' ranking."""
    jobs, labels = parity_jobs(task, seeds)
    gpu = run_jobs_mjwarp(jobs)
    cpu, cpu_t = _cpu_run(jobs, workers)
    agree = agreement(cpu, gpu.results)
    cpu_scores = entrant_scores(labels, jobs, cpu)
    gpu_scores = entrant_scores(labels, jobs, gpu.results)
    by_family: dict[str, list[bool]] = {}
    mism = []
    for job, label, c, g in zip(jobs, labels, cpu, gpu.results, strict=True):
        by_family.setdefault(job.obj.kind, []).append(outcome(c) == outcome(g))
        if outcome(c) != outcome(g):
            mism.append(
                {
                    "entrant": label,
                    "object": job.obj.label,
                    "placement": dataclasses.asdict(job.placement),
                    "setting": job.controller_setting,
                    "seed": job.seed,
                    "cpu": [c.success, c.failure_mode, round(c.max_lift, 4), list(c.flags)],
                    "gpu": [g.success, g.failure_mode, round(g.max_lift, 4), list(g.flags)],
                }
            )
    out = {
        **agree,
        "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
        "cpu_success_rate": sum(r.success for r in cpu) / len(cpu),
        "gpu_success_rate": sum(r.success for r in gpu.results) / len(cpu),
        "cpu_entrant_scores": cpu_scores,
        "gpu_entrant_scores": gpu_scores,
        "cpu_ranking": ranking(cpu_scores),
        "gpu_ranking": ranking(gpu_scores),
        "mismatch_details": mism,
        "cpu_timing": cpu_t,
        "gpu_timing": _timing_dict(gpu),
    }
    # Plan 3.6 (b): success flags agree on >= 95 % and the manual entrants rank in the same order.
    out["pass"] = bool(out["success_agreement"] >= 0.95 and out["cpu_ranking"] == out["gpu_ranking"])
    return out, gpu.results


def bench_determinism(
    task: TaskConfig, reruns: int = 3, first: Sequence[EpisodeResult] | None = None
) -> dict[str, Any]:
    """(c) Same jobs, same GPU, ``reruns`` runs: disagreement with the first run.

    The gate (plan 3.5: "success-flag disagreement <= 1 %") is judged on the success
    flag; the stricter outcome disagreement (success and failure mode) is reported too.
    """
    jobs, _ = parity_jobs(task)
    runs = [list(first)] if first is not None else []
    while len(runs) < reruns:
        runs.append(run_jobs_mjwarp(jobs).results)
    worst = worst_outcome = 0.0
    pairs = []
    for i in range(1, len(runs)):
        a = agreement(runs[0], runs[i])
        exact = sum(x.summary() == y.summary() for x, y in zip(runs[0], runs[i], strict=True))
        pairs.append({**a, "bitwise_identical_summaries": exact})
        worst = max(worst, 1.0 - a["success_agreement"])
        worst_outcome = max(worst_outcome, 1.0 - a["outcome_agreement"])
    return {
        "reruns": len(runs),
        "pairs": pairs,
        "max_disagreement": worst,
        "max_outcome_disagreement": worst_outcome,
        "pass": worst <= 0.01,
    }


def bench_invariance(task: TaskConfig, n_batch: int = 288, singles: int = 3) -> dict[str, Any]:
    """(d) One design's episodes alone vs inside a batch of ``n_batch`` worlds (one object family).

    The design is the first manual entrant on the box family (4 conditions x 3 settings x
    2 seeds = 24 episodes). "Alone" is a batch of just those 24 worlds, and the first
    ``singles`` of them also run as 1-world batches; "in batch" pads them with Sobol
    designs to ``n_batch`` worlds.
    """
    jobs, labels = parity_jobs(task)
    target = [j for j, lab in zip(jobs, labels, strict=True) if lab == labels[0] and j.obj.kind == "box"]
    filler = sobol_jobs(n_batch - len(target), "box", seed=7, task=task)
    alone = run_jobs_mjwarp(target).results
    mixed = run_jobs_mjwarp(filler[: len(filler) // 2] + target + filler[len(filler) // 2 :])
    inside = mixed.results[len(filler) // 2 : len(filler) // 2 + len(target)]
    single = [run_jobs_mjwarp([j]).results[0] for j in target[:singles]]
    a = agreement(alone, inside)
    s = agreement(single, inside[:singles])
    exact = sum(x.summary() == y.summary() for x, y in zip(alone, inside, strict=True))
    return {
        "design_episodes": len(target),
        "batch_worlds": len(mixed.results),
        "alone_vs_batch": {**a, "bitwise_identical_summaries": exact},
        "single_world_vs_batch": s,
        # Plan 3.5 / 3.6 (d): identical success flags alone and inside the batch.
        "pass": a["success_mismatches"] == 0 and s["success_mismatches"] == 0,
    }


def bench_throughput(
    task: TaskConfig, sizes: Sequence[int], workers: int, cpu_episodes: int
) -> dict[str, Any]:
    """(e) Wall time per batch size, first (cold: includes JIT or cache load) and warm, vs the CPU pool."""
    out: dict[str, Any] = {"gpu": {}, "cpu": None}
    for n in sizes:
        jobs = sobol_jobs(n, "box", seed=11, task=task)
        first = run_jobs_mjwarp(jobs, WarpConfig(max_worlds=n))
        warm = run_jobs_mjwarp(jobs, WarpConfig(max_worlds=n))
        out["gpu"][str(n)] = {"first": _timing_dict(first), "warm": _timing_dict(warm)}
    if cpu_episodes:
        jobs = sobol_jobs(cpu_episodes, "box", seed=11, task=task)
        _, out["cpu"] = _cpu_run(jobs, workers)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Spike benchmark for gate G3: ``python -m rtt.sim.exec_mjwarp {geometry,parity,...}``.

    Prints one JSON document (and writes it to ``--out`` when given). Set
    ``WARP_CACHE_PATH`` to an empty directory to measure a cold Warp kernel cache.
    """
    parser = argparse.ArgumentParser(prog="python -m rtt.sim.exec_mjwarp", description=main.__doc__)
    parser.add_argument(
        "what", nargs="+", choices=("geometry", "parity", "determinism", "invariance", "throughput", "all")
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="CPU executor processes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 288, 960], help="throughput batch sizes")
    parser.add_argument("--cpu-episodes", type=int, default=288, help="CPU throughput sample (0: skip)")
    parser.add_argument("--out", type=Path, help="also write the JSON here")
    args = parser.parse_args(argv)
    wp.config.quiet = True
    what = set(args.what)
    if "all" in what:
        what = {"geometry", "parity", "determinism", "invariance", "throughput"}
    task = default_task()
    report: dict[str, Any] = {
        "task": task.name,
        "task_digest": task_digest(task),
        "timestep": task.physics.timestep,
        "device": str(wp.get_device()),
        "gpu": wp.get_device().name,
        "mujoco": mujoco.__version__,
        "mujoco_warp": _mjw().__version__ if hasattr(_mjw(), "__version__") else "3.15.0",
        "warp": wp.__version__,
        "cpu_count": os.cpu_count(),
        "load_avg_start": os.getloadavg(),
    }
    first = None
    if "geometry" in what:
        report["a_geometry"] = bench_geometry(task)
    if "parity" in what:
        report["b_parity"], first = bench_parity(task, args.workers)
    if "determinism" in what:
        report["c_determinism"] = bench_determinism(task, first=first)
    if "invariance" in what:
        report["d_invariance"] = bench_invariance(task)
    if "throughput" in what:
        report["e_throughput"] = bench_throughput(task, args.sizes, args.workers, args.cpu_episodes)
    report["load_avg_end"] = os.getloadavg()
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
