"""Gates GV1 (fidelity and fairness) and GV2 (physics) for the task v2 simulator path.

Both gates take the jaw builder, the training parts and the designs as arguments, so they
run on any implementation of the frozen contracts: the test fixtures today, the real
generator (:func:`rtt.design.freeform.build_pair`) and part library
(:func:`rtt.parts.library.training_parts`) once they land. Each returns a JSON-friendly
report with every number and a ``passed`` flag per criterion (spec section 14).

GV1:

* **clearance**: in the compiled scored scene, MuJoCo's own distance between each imprint jaw
  and its centred part at closure ``q_seat - clearance`` equals the designed clearance within
  0.05 mm, and each jaw's first contact is at ``q_seat`` within 0.05 mm (the pads conform to
  the collision pieces the simulator sees);
* **zero-error registration**: both imprints register in >= 95 % of zero-error episodes on
  every part. Zero error means no placement error and no position or yaw jitter (the friction
  jitter stays on); the same episodes with the full task jitter are reported as a diagnostic.

GV2:

* the strategy matrix (each design x each part x the train_v2 placements x 2 seeds): no MuJoCo
  resets, object-jaw penetration <= 1 mm in held episodes and p99 <= 2 mm overall, hold creep
  p95 <= 0.5 mm in held episodes;
* deliberate failures fail: the tips 30 mm above the part's top, mu 0.05 on a copy of every
  part made heavier than the friction capacity, and the part widened past the open jaws;
* 1 ms vs 2 ms: >= 99 % identical outcomes (registered and held) over >= 200 v2 episodes;
* ``native_ccd`` (spec section 6): outcome agreement on and off on the v1 parity set (200
  episodes) and on the same >= 200 v2 episodes, the lip-edge-normal regression (reproduced on
  v1 box lips with a primitive cylinder; v2 analogues with a mesh cylinder on box lips and on
  convex-mesh lips), and the rule's verdict; the task's setting is the recorded decision;
* asymmetric jaws compile with their own mass properties;
* determinism: identical trajectory hashes for 10 designs over 3 fresh processes;
* cost: the v2 median episode CPU time is <= 1.5x the v1 median (interleaved, same process).

Command line, with builder, parts and designs given as ``module:function``::

    python -m rtt.sim.gates_v2 gv2 --builder freeform_fixtures:build_fixture_pair \\
        --parts freeform_fixtures:training_parts --designs freeform_fixtures:gv2_designs \\
        --det-designs freeform_fixtures:determinism_designs

With the generator and the part library: ``--builder rtt.design.freeform:build_pair --parts
rtt.parts.library:training_parts --designs rtt.sim.gates_v2:generator_gv2_designs
--det-designs rtt.sim.gates_v2:generator_determinism_designs`` (gv1:
``--designs rtt.sim.gates_v2:generator_gv1_designs``). The determinism criterion needs at
least 10 designs, so a gv2 run without ``--det-designs`` (3 designs) fails it.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import multiprocessing as mp
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rtt.sim import objects as objs
from rtt.sim.conditions import derive_seed
from rtt.sim.controller import plan_grasp_v2
from rtt.sim.episode import EpisodeResult, run_episode, run_episode_v2
from rtt.sim.exec_cpu import CpuExecutor, EpisodeJob
from rtt.sim.fingers_v2 import check_pair, jaw_mass_properties, lowest_point
from rtt.sim.register import seat_clearance
from rtt.sim.scene import build_model_v2
from rtt.sim.task import TaskConfig, TaskV2, default_task_v2

Builder = Callable[[Any, Any, TaskConfig], Any]
FIDELITY_TOL_MM = 0.05
IMPRINT_REGISTER_MIN = 0.95
PEN_HELD_MAX_MM = 1.0
PEN_P99_MAX_MM = 2.0
CREEP_P95_MAX_MM = 0.5
PARITY_MIN = 0.99
PARITY_MIN_EPISODES = 200
DETERMINISM_MIN_DESIGNS = 10  # spec GV2: 10 designs x 3 fresh processes
COST_RATIO_MAX = 1.5
TOO_HIGH = 0.03  # m above the part's top
LOW_MU = 0.05
HEAVY_FACTOR = 2.0  # the low-friction copy weighs this many times the friction capacity
WIDE_MARGIN = 0.010  # m wider than the open jaws
GRAVITY = 9.81
GATE_BASE_SEED = 20261016  # public; gate episodes only, never a scored suite
GV1_SEEDS = 8
#: The train_v2 placements (spec section 8): err_x is across the jaw (the finger width).
GV2_PLACEMENTS: dict[str, objs.Placement] = {
    "c0": objs.Placement(),
    "c1": objs.Placement(err_x=0.002),
    "c2": objs.Placement(err_x=-0.003),
    "c3": objs.Placement(err_x=0.001, err_yaw=math.radians(6.0)),
}


# --------------------------------------------------------------------------- helpers


def scale_part(part: Any, scale: Sequence[float], name: str) -> Any:
    """``part`` scaled by ``(sx, sy, sz)`` (positive), with exact mass properties (no features)."""
    d = np.asarray(scale, dtype=np.float64)
    if d.shape != (3,) or np.any(d <= 0):
        raise ValueError("scale must be three positive factors")
    i6 = np.asarray(part.unit_inertia)
    inertia = np.array([[i6[0], i6[3], i6[4]], [i6[3], i6[1], i6[5]], [i6[4], i6[5], i6[2]]])
    second = np.trace(inertia) / 2.0 * np.eye(3) - inertia  # int r r^T dV about the COM
    det = float(np.prod(d))
    second = det * (np.diag(d) @ second @ np.diag(d))
    scaled = np.trace(second) * np.eye(3) - second
    return dataclasses.replace(
        part,
        name=name,
        pieces=tuple(p * d for p in part.pieces),
        volume=part.volume * det,
        com=tuple(float(v) for v in np.asarray(part.com) * d),
        unit_inertia=(scaled[0, 0], scaled[1, 1], scaled[2, 2], scaled[0, 1], scaled[0, 2], scaled[1, 2]),
        features=(),
        visual_asset=None,
    )


def with_density(part: Any, density: float) -> Any:
    """``part`` with its material's density replaced (same geometry, so the same jaws fit)."""
    return dataclasses.replace(part, material=dataclasses.replace(part.material, density=float(density)))


def _outcome(r: EpisodeResult) -> tuple[bool, bool]:
    return (bool(r.success), bool(getattr(r, "lift_success", r.success)))


def _worst_abs(values: Sequence[float]) -> float:
    """Largest ``|v|``, infinite if any value is not finite."""
    return max((abs(v) if math.isfinite(v) else math.inf for v in values), default=math.inf)


def _pct(values: Sequence[float], q: float) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.percentile(vals, q)) if vals else math.nan


def _run(jobs: Sequence[EpisodeJob], workers: int | None) -> list[EpisodeResult]:
    if not jobs:
        return []
    with CpuExecutor(workers) as ex:
        return ex.run(jobs).results


def _with_physics(jobs: Sequence[EpisodeJob], **physics: Any) -> list[EpisodeJob]:
    return [dataclasses.replace(j, task=j.task.with_overrides(physics=physics)) for j in jobs]


def _agreement(a: Sequence[EpisodeResult], b: Sequence[EpisodeResult]) -> dict[str, Any]:
    n = len(a)
    if not n:
        return {"n": 0, "outcome_agreement": math.nan, "failure_mode_agreement": math.nan}
    pairs = list(zip(a, b, strict=True))
    diff = [i for i, (x, y) in enumerate(pairs) if _outcome(x) != _outcome(y)]
    same_mode = sum(x.failure_mode == y.failure_mode for x, y in pairs)
    return {
        "n": n,
        "outcome_agreement": 1.0 - len(diff) / n,
        "registered_agreement": sum(_outcome(x)[0] == _outcome(y)[0] for x, y in pairs) / n,
        "held_agreement": sum(_outcome(x)[1] == _outcome(y)[1] for x, y in pairs) / n,
        "failure_mode_agreement": same_mode / n,
        "success_rate": [sum(x.success for x in a) / n, sum(y.success for y in b) / n],
        "disagreements": diff,
    }


def build_pairs(
    builder: Builder, parts: Sequence[Any], designs: Mapping[str, Any], task: TaskConfig
) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], str]]:
    """``({(design, part_id): pair}, {(design, part_id): build error})``; a pair the scene
    refuses (:func:`rtt.sim.fingers_v2.check_pair`, e.g. a non-convex piece) is a build error."""
    pairs: dict[tuple[str, str], Any] = {}
    invalid: dict[tuple[str, str], str] = {}
    for dname, design in designs.items():
        for part in parts:
            try:
                pair = builder(design, part, task)
                check_pair(pair)
            except ValueError as exc:
                invalid[(dname, part.part_id)] = str(exc)
                continue
            pairs[(dname, part.part_id)] = pair
    return pairs, invalid


def matrix_jobs(
    pairs: Mapping[tuple[str, str], Any],
    parts: Sequence[Any],
    designs: Sequence[str],
    task: TaskV2,
    seeds_per_condition: int,
    placements: Mapping[str, objs.Placement] = GV2_PLACEMENTS,
    base_seed: int = GATE_BASE_SEED,
) -> tuple[list[tuple[str, str, str, int]], list[EpisodeJob]]:
    """``(keys, jobs)``: design x part x placement x seed, keys ``(design, part_id, label, rep)``.

    Seeds are derived per (part, placement, rep), so every design sees the same draws and a
    run with more seeds per condition extends one with fewer.
    """
    keys, jobs = [], []
    for dname in designs:
        for part in parts:
            pair = pairs.get((dname, part.part_id))
            if pair is None:
                continue
            for label, placement in placements.items():
                cid = f"gate_v2/{part.part_id}/{label}"
                for rep in range(seeds_per_condition):
                    seed = derive_seed(base_seed, cid, rep)
                    keys.append((dname, part.part_id, label, rep))
                    jobs.append(EpisodeJob(pair, part, placement, -1, seed, task=task))
    return keys, jobs


def matrix_stats(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Penetration, creep, resets and rates of a set of v2 episodes."""
    held = [r for r in results if r.lift_success]
    pen_held = [r.metrics["obj_finger_pen_max"] * 1000.0 for r in held]
    pen_all = [r.metrics["obj_finger_pen_max"] * 1000.0 for r in results]
    creep = [r.metrics["hold_creep_mm"] for r in held]
    modes: dict[str, int] = {}
    for r in results:
        if r.failure_mode:
            modes[r.failure_mode] = modes.get(r.failure_mode, 0) + 1
    n = len(results)
    return {
        "episodes": n,
        "bad_qacc": int(sum(r.bad_qacc for r in results)),
        "registered_rate": sum(r.success for r in results) / n if n else math.nan,
        "held_rate": len(held) / n if n else math.nan,
        "pen_held_max_mm": max(pen_held) if pen_held else math.nan,
        "pen_held_over_1mm": sum(v > PEN_HELD_MAX_MM for v in pen_held),
        "pen_registered_over_1mm": sum(
            r.metrics["obj_finger_pen_max"] * 1000.0 > PEN_HELD_MAX_MM for r in results if r.success
        ),
        "pen_p99_mm": _pct(pen_all, 99),
        "pen_max_mm": max(pen_all) if pen_all else math.nan,
        "creep_p95_mm": _pct(creep, 95),
        "creep_max_mm": max((c for c in creep if math.isfinite(c)), default=math.nan),
        "failure_modes": modes,
    }


def _cells(keys: Sequence[tuple[str, str, str, int]], results: Sequence[EpisodeResult]) -> dict[str, Any]:
    by_cell: dict[str, dict[str, Any]] = {}
    for (dname, part_id, _, _), r in zip(keys, results, strict=True):
        cell = by_cell.setdefault(
            f"{dname}/{part_id}",
            {"episodes": 0, "registered": 0, "held": 0, "modes": {}, "pen_max_mm": 0.0, "creep_max_mm": 0.0},
        )
        cell["episodes"] += 1
        cell["registered"] += int(r.success)
        cell["held"] += int(r.lift_success)
        cell["pen_max_mm"] = max(cell["pen_max_mm"], round(r.metrics["obj_finger_pen_max"] * 1000.0, 3))
        creep = r.metrics["hold_creep_mm"]
        if r.lift_success and math.isfinite(creep):
            cell["creep_max_mm"] = max(cell["creep_max_mm"], round(creep, 3))
        if r.failure_mode:
            cell["modes"][r.failure_mode] = cell["modes"].get(r.failure_mode, 0) + 1
    return by_cell


# --------------------------------------------------------------------------- GV1


def run_gv1(
    builder: Builder,
    parts: Sequence[Any],
    imprints: Mapping[str, tuple[Any, float]],
    task: TaskV2 | None = None,
    *,
    seeds: int = GV1_SEEDS,
    workers: int | None = None,
) -> dict[str, Any]:
    """GV1 (module docstring). ``imprints``: name -> (design, designed clearance in m)."""
    task = task or default_task_v2()
    report: dict[str, Any] = {
        "gate": "GV1",
        "task": task.name,
        "native_ccd": task.physics.native_ccd,
        "clearance": {},
        "registration": {},
        "invalid_builds": {},
    }
    zero = task.with_overrides(jitter={"pos": 0.0, "yaw_deg": 0.0})
    jobs, keys, full_jobs = [], [], []
    clearance_ok = True
    worst_gap = worst_seat = 0.0
    for name, (design, clearance) in imprints.items():
        for part in parts:
            cell = f"{name}/{part.part_id}"
            try:
                pair = builder(design, part, task)
            except ValueError as exc:
                report["invalid_builds"][cell] = str(exc)
                clearance_ok = False
                continue
            obj = part.to_mesh_object()
            model, ids = build_model_v2(pair, obj, task)
            plan = plan_grasp_v2(pair, obj, objs.Placement(), task)
            probe = seat_clearance(model, ids, plan, obj, pair.q_seat, [clearance])
            gaps = [probe["left"][0], probe["right"][0]]
            contacts = probe["first_contact"]
            # a jaw that never touches the part has a NaN first contact: an infinite error
            # (max() would silently drop a NaN that is not its first argument)
            gap_err = _worst_abs([g - clearance for g in gaps]) * 1000.0
            seat_err = _worst_abs([q - pair.q_seat for q in contacts]) * 1000.0
            worst_gap, worst_seat = max(worst_gap, gap_err), max(worst_seat, seat_err)
            ok = gap_err <= FIDELITY_TOL_MM and seat_err <= FIDELITY_TOL_MM
            clearance_ok &= ok
            report["clearance"][cell] = {
                "designed_mm": clearance * 1000.0,
                "simulated_mm": [round(g * 1000.0, 4) for g in gaps],
                "first_contact_minus_q_seat_mm": [round((q - pair.q_seat) * 1000.0, 4) for q in contacts],
                "passed": ok,
            }
            for s in range(seeds):
                seed = derive_seed(GATE_BASE_SEED, f"gv1/{part.part_id}", s)
                keys.append((name, part.part_id))
                jobs.append(EpisodeJob(pair, part, objs.Placement(), -1, seed, task=zero))
                full_jobs.append(EpisodeJob(pair, part, objs.Placement(), -1, seed, task=task))
    results = _run(jobs + full_jobs, workers)
    zero_res, full_res = results[: len(jobs)], results[len(jobs) :]
    reg_ok = True
    cells: dict[tuple[str, str], list[tuple[EpisodeResult, EpisodeResult]]] = {}
    for key, rz, rf in zip(keys, zero_res, full_res, strict=True):
        cells.setdefault(key, []).append((rz, rf))
    pooled: dict[str, list[bool]] = {}
    for (name, part_id), rows in cells.items():
        rate = sum(rz.success for rz, _ in rows) / len(rows)
        rate_jit = sum(rf.success for _, rf in rows) / len(rows)
        pooled.setdefault(name, []).extend(rz.success for rz, _ in rows)
        ok = rate >= IMPRINT_REGISTER_MIN
        reg_ok &= ok
        report["registration"][f"{name}/{part_id}"] = {
            "zero_error_registered": rate,
            "with_task_jitter_registered": rate_jit,
            "episodes": len(rows),
            "failure_modes": sorted({rz.failure_mode for rz, _ in rows if rz.failure_mode}),
            "pen_max_mm": round(max(rz.metrics["obj_finger_pen_max"] for rz, _ in rows) * 1000.0, 3),
            "passed": ok,
        }
    report["summary"] = {
        "worst_clearance_error_mm": round(worst_gap, 4),
        "worst_first_contact_error_mm": round(worst_seat, 4),
        "zero_error_registered_pooled": {k: sum(v) / len(v) for k, v in pooled.items()},
    }
    report["passed"] = {
        "clearance": clearance_ok and bool(report["clearance"]),
        "zero_error_registration": reg_ok and bool(cells) and not report["invalid_builds"],
    }
    report["passed"]["all"] = all(report["passed"].values())
    return report


# --------------------------------------------------------------------------- GV2 pieces


def part_rests_on_jaws(
    pair: Any, part: Any, task: TaskConfig, drop: float = 0.005, backoff: float = 0.0002
) -> bool:
    """Can ``part`` rest on the jaws without friction (a flange on a pad's top, say)?

    The jaws stand ``backoff`` short of their seat, so they do not touch the centred part
    sideways, and the part is lowered by ``drop``: if it then overlaps a jaw (MuJoCo's own
    distance), the jaws can carry it by geometry and a low-friction case need not slip.
    """
    from rtt.sim.register import ProbeScene

    obj = part.to_mesh_object()
    model, ids = build_model_v2(pair, obj, task)
    plan = plan_grasp_v2(pair, obj, objs.Placement(), task)
    nominal = objs.Placement(plan.x, plan.y, plan.yaw)
    scene = ProbeScene.pose(model, ids, plan, obj, nominal, task.physics.table_top_z)
    d = scene.data
    d.qpos[ids.finger_qpos] = max(0.0, pair.q_seat - backoff)
    d.qpos[ids.object_qpos + 2] -= drop
    mujoco.mj_kinematics(scene.model, d)
    for g1 in np.concatenate(scene.jaw_geoms):
        for g2 in ids.object_geoms:
            if mujoco.mj_geomDistance(scene.model, d, int(g1), int(g2), 0.01, None) < 0.0:
                return True
    return False


def deliberate_failures(
    builder: Builder, parts: Sequence[Any], design: Any, task: TaskV2, workers: int | None = None
) -> dict[str, Any]:
    """The three GV2 deliberate failures on every part (``design`` should be a flat pad).

    * ``too_high``: the tips 30 mm above the part's top (``grasp_z_offset``): ``missed_grasp``.
    * ``low_friction``: mu 0.05 on a copy of the part ``HEAVY_FACTOR`` times heavier than what
      both jaws can carry by friction at that mu (the same geometry, so the same jaws): it
      must slip, unless the part can rest on the jaws (:func:`part_rests_on_jaws`; then the
      case is reported as ``supported`` and not counted). At least half the parts must be
      tested.
    * ``too_wide``: the part widened along the closing axis so it is 5 mm wider than the open
      jaws on each side: the jaws come down on it.
    """
    jobs, labels, expect = [], [], []
    capacity = 2.0 * LOW_MU * (1.0 + task.jitter.mu_frac) * task.gripper.grip_force / 2.0  # N
    supported: list[str] = []
    for part in parts:
        try:
            pair = builder(design, part, task)
        except ValueError:
            continue
        h = part.size[2]
        high = task.with_overrides(controller={"grasp_z_offset": h - pair.tip_height + TOO_HIGH})
        jobs.append(EpisodeJob(pair, part, objs.Placement(), -1, 0, task=high))
        labels.append(f"too_high/{part.part_id}")
        expect.append({"missed_grasp"})
        heavy_density = max(part.density, HEAVY_FACTOR * capacity / (GRAVITY * part.volume))
        slippery = task.with_overrides(physics={"friction": LOW_MU})
        jobs.append(
            EpisodeJob(pair, with_density(part, heavy_density), objs.Placement(), -1, 0, task=slippery)
        )
        labels.append(f"low_friction/{part.part_id}")
        if part_rests_on_jaws(pair, part, task):
            supported.append(part.part_id)
            expect.append(set())
        else:
            expect.append({"slip_during_lift", "missed_grasp", "unstable_hold"})
        width = part.size[1]
        wide = scale_part(part, (1.0, (width + 2.0 * pair.q_seat + WIDE_MARGIN) / width, 1.0), part.name)
        jobs.append(EpisodeJob(pair, wide, objs.Placement(), -1, 0, task=task))
        labels.append(f"too_wide/{part.part_id}")
        expect.append({"missed_grasp", "finger_env_collision", "ejected_on_close"})
    results = _run(jobs, workers)
    cases = {}
    for label, want, r in zip(labels, expect, results, strict=True):
        row: dict[str, Any] = {"failure_mode": r.failure_mode, "lift_success": bool(r.lift_success)}
        if not want:
            row.update(supported=True, passed=True)
        else:
            row.update(
                expected=sorted(want),
                passed=(not r.success) and (not r.lift_success) and r.failure_mode in want,
            )
        cases[label] = row
    n_parts = sum(label.startswith("too_high/") for label in labels)
    tested = n_parts - len(supported)
    return {
        "cases": cases,
        "low_friction_supported": supported,
        "passed": bool(cases) and all(c["passed"] for c in cases.values()) and 2 * tested >= n_parts,
    }


def _lips_finger() -> Any:
    """The v1 lip-edge case: a concave finger of box lips (three boxes per row, four rows)."""
    from rtt.design.types import Box, FingerGeometry

    boxes = []
    for s in range(4):
        cz = -0.01 - 0.02 * s
        boxes.append(Box((-0.008, 0.002, cz), (0.002, 0.002, 0.01)))
        boxes.append(Box((0.0, 0.007, cz), (0.01, 0.003, 0.01)))
        boxes.append(Box((0.008, 0.002, cz), (0.002, 0.002, 0.01)))
    boxes += [Box((0.0, 0.007, -0.01), (1e-4, 1e-4, 1e-4))] * 2
    return FingerGeometry(tuple(boxes))


def mesh_cylinder(radius: float = 0.025, height: float = 0.09, n: int = 32) -> Any:
    """A standing ``n``-gon cylinder as a one-piece PartSpec (the lip case's object)."""
    import scipy.spatial

    from rtt.parts.types import Material, PartSpec

    ring = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.array(
        [[radius * math.cos(a), radius * math.sin(a), z] for z in (-height / 2, height / 2) for a in ring]
    )
    hull = scipy.spatial.ConvexHull(pts)
    faces = []
    for tri in hull.simplices:
        a, b, c = pts[tri]
        if np.dot(np.cross(b - a, c - a), a) < 0:
            tri = tri[[0, 2, 1]]
        faces.append(tri)
    vol = n * 0.5 * radius**2 * math.sin(2 * math.pi / n) * height
    ixx = vol * (3 * radius**2 + height**2) / 12
    return PartSpec(
        name="lip_mesh_cylinder",
        family="cylinder",
        pieces=(pts,),
        faces=(np.asarray(faces, dtype=np.int64),),
        volume=vol,
        com=(0.0, 0.0, 0.0),
        unit_inertia=(ixx, ixx, vol * radius**2 / 2, 0.0, 0.0, 0.0),
        material=Material("plastic", 700.0, (0.5, 0.5, 0.5)),
        yaw_symmetry=0,
        register_axes=("x", "y", "tilt"),
    )


def _lip_row(r: EpisodeResult, pen_key: str) -> dict[str, Any]:
    return {
        "success": bool(r.success),
        "lift_success": bool(getattr(r, "lift_success", r.success)),
        "failure_mode": r.failure_mode,
        "squeeze_n": [round(r.metrics["squeeze_left"], 2), round(r.metrics["squeeze_right"], 2)],
        "penetration_mm": round(r.metrics[pen_key] * 1000.0, 3),
    }


def lip_regression(task: TaskV2) -> dict[str, Any]:
    """The v1 lip-edge-normal regression with native CCD on and off, and its v2 analogues.

    v1: a primitive cylinder pressed against box lips (what made task_v1 turn native CCD off:
    the contact normals ran along the finger width, so the squeeze along the closing axis read
    0). v2 parts are always meshes, so the analogues are a 32-gon mesh cylinder on the same box
    lips (``boxes`` kept, MuJoCo box geoms) and on the lips as convex mesh pieces (how the v2
    generator's jaws are made). "Reproduced": the v1 case keeps its squeeze and succeeds with
    native CCD off, and loses the squeeze or the grasp with it on. "Fixed": both v2 analogues
    squeeze at least the grasp threshold on each jaw and hold the part with native CCD on.
    """
    from rtt.sim.fingers_v2 import v1_pair
    from rtt.sim.m1 import G1_SETTING
    from rtt.sim.register import probe_first_contact
    from rtt.sim.task import load_task

    lips = _lips_finger()
    cyl = objs.ObjectSpec("cylinder", (0.025, 0.09), 700.0)
    out: dict[str, Any] = {}
    v1 = load_task()
    for native in (False, True):
        t = v1.with_overrides(physics={"native_ccd": native})
        r = run_episode(lips, cyl, objs.Placement(), G1_SETTING, 0, record=False, task=t)
        out[f"v1_primitive_cylinder_native_{native}"] = _lip_row(r, "penetration_max")
    part = mesh_cylinder()
    obj = part.to_mesh_object()
    min_squeeze = task.score.grasp_min_squeeze_frac * task.gripper.grip_force / 2.0
    for variant in ("box_lips", "mesh_lips"):
        for native in (False, True):
            t = task.with_overrides(physics={"native_ccd": native})
            rough = v1_pair(lips, part_id=part.part_id, tip_height=0.045, q_seat=0.01)
            if variant == "mesh_lips":
                rough = dataclasses.replace(rough, boxes=None)
            model, ids = build_model_v2(rough, obj, t)
            q = min(probe_first_contact(model, ids, plan_grasp_v2(rough, obj, objs.Placement(), t), obj))
            pair = dataclasses.replace(rough, q_seat=q)
            r = run_episode_v2(pair, part, objs.Placement(), 0, record=False, task=t)
            out[f"v2_{variant}_mesh_cylinder_native_{native}"] = _lip_row(r, "obj_finger_pen_max")

    def squeezed(row: dict[str, Any]) -> bool:
        return all(math.isfinite(s) and s >= min_squeeze for s in row["squeeze_n"])

    off, on = out["v1_primitive_cylinder_native_False"], out["v1_primitive_cylinder_native_True"]
    out["v1_regression_reproduced"] = bool(
        squeezed(off) and off["success"] and (not squeezed(on) or not on["success"])
    )
    out["v2_analogues_fixed_with_native"] = all(
        out[f"v2_{v}_mesh_cylinder_native_True"]["lift_success"]
        and squeezed(out[f"v2_{v}_mesh_cylinder_native_True"])
        for v in ("box_lips", "mesh_lips")
    )
    return out


def _hash_job(job: EpisodeJob) -> str:
    return run_episode(job.finger, job.obj, job.placement, -1, job.seed, record=False, task=job.task).traj_sha


def fresh_process_hashes(jobs: Sequence[EpisodeJob], processes: int = 3) -> dict[str, Any]:
    """Trajectory hashes of ``jobs`` in this process and in ``processes`` fresh spawned
    interpreters (one at a time, each running every job)."""
    ctx = mp.get_context("spawn")
    here = [_hash_job(j) for j in jobs]
    runs = []
    for _ in range(processes):
        with ctx.Pool(1) as pool:
            runs.append(pool.map(_hash_job, jobs, chunksize=max(1, len(jobs))))
    mismatches = [i for i in range(len(jobs)) if any(r[i] != here[i] for r in runs)]
    return {
        "jobs": len(jobs),
        "processes": processes,
        "identical": not mismatches and all(here),
        "hashes": here,
        "mismatches": mismatches,
    }


def episode_cpu_seconds(jobs: Sequence[EpisodeJob]) -> list[float]:
    """CPU seconds of each job run in this process (``time.process_time``)."""
    out = []
    for j in jobs:
        t0 = time.process_time()
        run_episode(j.finger, j.obj, j.placement, j.controller_setting, j.seed, record=False, task=j.task)
        out.append(time.process_time() - t0)
    return out


def v1_reference_jobs(n: int = 24) -> list[EpisodeJob]:
    """v1 entrants on the v1 training suite at k = 0.5 (the cost reference)."""
    from rtt.design.generate import generate_finger
    from rtt.design.schema import load_entrants
    from rtt.sim.conditions import training_suite
    from rtt.sim.m1 import DEFAULT_ENTRANTS
    from rtt.sim.task import default_task

    fingers = [generate_finger(e) for e in load_entrants(DEFAULT_ENTRANTS)]
    jobs = []
    for i, cond in enumerate(training_suite()):
        for seed in cond.seeds:
            jobs.append(
                EpisodeJob(fingers[i % len(fingers)], cond.obj, cond.placement, 1, seed, task=default_task())
            )
    step = max(1, len(jobs) // n)
    return jobs[::step][:n]


def cost_ratio(v2_jobs: Sequence[EpisodeJob], n: int = 24) -> dict[str, Any]:
    """Median CPU time per episode, v2 over v1, the two sets interleaved in this process."""
    v1_jobs = v1_reference_jobs(n)
    step = max(1, len(v2_jobs) // n)
    v2_sample = list(v2_jobs[::step][:n])
    v1_cpu: list[float] = []
    v2_cpu: list[float] = []
    for a, b in zip(v1_jobs, v2_sample, strict=False):
        v1_cpu += episode_cpu_seconds([a])
        v2_cpu += episode_cpu_seconds([b])
    v1_med, v2_med = statistics.median(v1_cpu), statistics.median(v2_cpu)
    return {
        "episodes": [len(v1_cpu), len(v2_cpu)],
        "v1_median_cpu_s": v1_med,
        "v2_median_cpu_s": v2_med,
        "ratio": v2_med / v1_med,
    }


def determinism_jobs(
    builder: Builder, parts: Sequence[Any], designs: Mapping[str, Any], task: TaskV2
) -> tuple[list[str], list[EpisodeJob]]:
    """One episode per design (cycling through the parts and the placements)."""
    labels, jobs = [], []
    places = list(GV2_PLACEMENTS.items())
    for i, (dname, design) in enumerate(designs.items()):
        for k in range(len(parts)):
            part = parts[(i + k) % len(parts)]
            try:
                pair = builder(design, part, task)
            except ValueError:
                continue
            label, placement = places[i % len(places)]
            labels.append(f"{dname}/{part.part_id}/{label}")
            seed = derive_seed(GATE_BASE_SEED, labels[-1], 0)
            jobs.append(EpisodeJob(pair, part, placement, -1, seed, task=task))
            break
    return labels, jobs


def native_ccd_rule(
    v1_agreement: float,
    v2_agreement: float,
    lips: Mapping[str, Any],
    v1_n: int = PARITY_MIN_EPISODES,
    v2_n: int = PARITY_MIN_EPISODES,
) -> dict[str, Any]:
    """Spec section 6: switch to native CCD only if both parities are >= 99 % (the v1 parity
    set and >= 200 v2 episodes) and the lip-edge regression is reproduced and shown fixed;
    otherwise keep it off. ``v1_n`` and ``v2_n`` are the episode counts behind the parities."""
    checks = {
        "v1_parity": bool(v1_agreement >= PARITY_MIN and v1_n >= PARITY_MIN_EPISODES),
        "v2_parity": bool(v2_agreement >= PARITY_MIN and v2_n >= PARITY_MIN_EPISODES),
        "lip_regression_reproduced": bool(lips["v1_regression_reproduced"]),
        "lip_regression_fixed_on_v2_geometry": bool(lips["v2_analogues_fixed_with_native"]),
    }
    return {**checks, "switch_to_native": all(checks.values())}


def run_gv2(
    builder: Builder,
    parts: Sequence[Any],
    designs: Mapping[str, Any],
    task: TaskV2 | None = None,
    *,
    det_designs: Mapping[str, Any] | None = None,
    seeds_per_condition: int = 2,
    parity_seeds_per_condition: int = 3,
    workers: int | None = None,
    parity_timestep: bool = True,
    native_ccd_parity: bool = True,
    v1_parity_n: int = 200,
    fresh_processes: int = 3,
    cost_samples: int = 24,
) -> dict[str, Any]:
    """GV2 (module docstring). ``designs``: name -> design, one per pad strategy, the first a
    flat pad (it runs the deliberate failures); ``det_designs``: the determinism designs
    (default ``designs``). The parity checks run on the matrix extended to
    ``parity_seeds_per_condition`` seeds (3 x 6 parts x 4 placements x 3 designs = 216)."""
    task = task or default_task_v2()
    t0 = time.perf_counter()
    pairs, invalid = build_pairs(builder, parts, designs, task)
    reps = max(seeds_per_condition, parity_seeds_per_condition)
    keys, jobs = matrix_jobs(pairs, parts, list(designs), task, reps)
    results = _run(jobs, workers)
    in_matrix = [i for i, k in enumerate(keys) if k[3] < seeds_per_condition]
    m_keys = [keys[i] for i in in_matrix]
    m_res = [results[i] for i in in_matrix]
    stats = matrix_stats(m_res)
    lowest = [abs(lowest_point(p) - p.finger_length) * 1000.0 for p in pairs.values()]
    report: dict[str, Any] = {
        "gate": "GV2",
        "task": task.name,
        "native_ccd": task.physics.native_ccd,
        "machine": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "cpus": mp.cpu_count(),
        },
        "matrix": {
            "designs": list(designs),
            "parts": [p.part_id for p in parts],
            "placements": list(GV2_PLACEMENTS),
            "seeds_per_condition": seeds_per_condition,
            "invalid_builds": {f"{d}/{p}": v for (d, p), v in invalid.items()},
            "finger_length_vs_lowest_point_max_mm": max(lowest) if lowest else math.nan,
            **stats,
            "by_cell": _cells(m_keys, m_res),
        },
    }
    passed = {
        "matrix_builds": not invalid and bool(m_res),
        "bad_qacc_zero": stats["bad_qacc"] == 0,
        "penetration_held": stats["pen_held_over_1mm"] == 0,
        "penetration_p99": bool(stats["pen_p99_mm"] <= PEN_P99_MAX_MM),
        "creep_p95": bool(stats["creep_p95_mm"] <= CREEP_P95_MAX_MM),
    }
    first = next(iter(designs.values()))
    report["deliberate_failures"] = deliberate_failures(builder, parts, first, task, workers)
    passed["deliberate_failures"] = report["deliberate_failures"]["passed"]

    asym = {}
    for (dname, part_id), pair in pairs.items():
        lp, rp = jaw_mass_properties(pair.left), jaw_mass_properties(pair.right)
        if abs(lp.mass - rp.mass) > 1e-6 * lp.mass:
            part = next(p for p in parts if p.part_id == part_id)
            model, ids = build_model_v2(pair, part.to_mesh_object(), task)
            lb, rb = ids.finger_left_body, ids.finger_right_body
            compiled = [float(model.body_mass[lb]), float(model.body_mass[rb])]
            asym[f"{dname}/{part_id}"] = {
                "mass_g": [round(lp.mass * 1000, 4), round(rp.mass * 1000, 4)],
                "compiled_mass_g": [round(m * 1000, 4) for m in compiled],
                "com_y_mm": [round(float(lp.com[1]) * 1000, 3), round(float(rp.com[1]) * 1000, 3)],
                "own": bool(
                    np.allclose(compiled, [lp.mass, rp.mass], rtol=1e-9)
                    and np.allclose(model.body_ipos[lb], lp.com, atol=1e-12)
                    and np.allclose(model.body_ipos[rb], rp.com, atol=1e-12)
                ),
            }
    report["asymmetric_jaws"] = {"pairs": asym, "own_mass_properties": all(v["own"] for v in asym.values())}
    passed["asymmetric_jaws"] = bool(asym) and report["asymmetric_jaws"]["own_mass_properties"]

    if parity_timestep:
        fine = _run(_with_physics(jobs, timestep=0.001), workers)
        report["timestep_parity"] = {
            "timesteps_s": [0.001, task.physics.timestep],
            **_agreement(fine, results),
        }
        passed["timestep_parity"] = (
            len(jobs) >= PARITY_MIN_EPISODES and report["timestep_parity"]["outcome_agreement"] >= PARITY_MIN
        )
    if native_ccd_parity:
        other = _run(_with_physics(jobs, native_ccd=not task.physics.native_ccd), workers)
        off, on = (other, results) if task.physics.native_ccd else (results, other)
        lips = lip_regression(task)
        nccd: dict[str, Any] = {
            "v2_off_vs_on": _agreement(off, on),
            f"matrix_with_native_{not task.physics.native_ccd}": matrix_stats([other[i] for i in in_matrix]),
            "lip_regression": lips,
        }
        v1_agree, v1_n = math.nan, 0
        if v1_parity_n:
            from rtt.sim.bench import parity_jobs
            from rtt.sim.task import default_task

            v1_jobs = [j for _, _, j in parity_jobs(default_task(), v1_parity_n)]
            v1_off = _run(v1_jobs, workers)
            v1_on = _run(_with_physics(v1_jobs, native_ccd=True), workers)
            nccd["v1_parity_set_off_vs_on"] = _agreement(v1_off, v1_on)
            v1_agree = nccd["v1_parity_set_off_vs_on"]["outcome_agreement"]
            v1_n = nccd["v1_parity_set_off_vs_on"]["n"]
        v2 = nccd["v2_off_vs_on"]
        nccd["rule"] = native_ccd_rule(v1_agree, v2["outcome_agreement"], lips, v1_n, v2["n"])
        nccd["decision"] = task.physics.native_ccd
        report["native_ccd"] = nccd
        passed["native_ccd_decision_recorded"] = True
    if fresh_processes:
        dd = det_designs or designs
        labels, det_jobs = determinism_jobs(builder, parts, dd, task)
        report["determinism"] = {"episodes": labels, **fresh_process_hashes(det_jobs, fresh_processes)}
        passed["determinism"] = (
            report["determinism"]["identical"]
            and len(det_jobs) == len(dd)
            and len(det_jobs) >= DETERMINISM_MIN_DESIGNS
        )
    if cost_samples:
        report["cost"] = cost_ratio(jobs, cost_samples)
        passed["cost"] = report["cost"]["ratio"] <= COST_RATIO_MAX
    passed["all"] = all(passed.values())
    report["passed"] = passed
    report["wall_s"] = time.perf_counter() - t0
    return report


# --------------------------------------------------------------------------- generator inputs


def generator_gv1_designs() -> dict[str, tuple[Any, float]]:
    """GV1 imprints from ``configs/entrants_v2.yaml`` (needs the generator track's code)."""
    from rtt.design.freeform.baselines import load_entrants_v2

    out = {}
    for b in load_entrants_v2():
        if b.name in ("imprint_configurator", "imprint_swept"):
            out[b.name] = (b, float(b.recipe.clearance_mm) / 1000.0)
    if len(out) != 2:
        raise ValueError(f"expected both imprints in the entrants file, got {sorted(out)}")
    return out


def generator_gv2_designs() -> dict[str, Any]:
    """One FingerRecipe per pad strategy, 40 mm wide, blunt (needs the generator track's code)."""
    from rtt.design.freeform.schema import FingerRecipe

    common = {"tip": "blunt", "grip_height_frac": 0.5, "pad_height_mm": 20.0, "finger_width_mm": 40.0}
    return {
        "flat": FingerRecipe(pad="flat", **common),
        "conform": FingerRecipe(
            pad="conform", pad_depth_mm=4.0, wall_draft_deg=30.0, clearance_mm=0.4, **common
        ),
        "v_funnel": FingerRecipe(pad="v_funnel", pad_depth_mm=6.0, wall_draft_deg=45.0, **common),
    }


def generator_determinism_designs() -> dict[str, Any]:
    """GV2 determinism: ten distinct FingerRecipes covering every pad strategy, the chamfer
    and hook tips, straight and drafted walls and several widths and clearances (needs the
    generator track's code)."""
    from rtt.design.freeform.schema import FingerRecipe

    c = {"tip": "blunt", "grip_height_frac": 0.5, "pad_height_mm": 20.0, "finger_width_mm": 40.0}
    return {
        **generator_gv2_designs(),
        "conform_d0": FingerRecipe(
            pad="conform", pad_depth_mm=6.0, wall_draft_deg=0.0, clearance_mm=0.4, **c
        ),
        "conform_d45": FingerRecipe(
            pad="conform", pad_depth_mm=5.0, wall_draft_deg=45.0, clearance_mm=0.6, **c
        ),
        "v_funnel_30": FingerRecipe(
            pad="v_funnel", pad_depth_mm=8.0, wall_draft_deg=30.0, **{**c, "finger_width_mm": 36.0}
        ),
        "flat_narrow_high": FingerRecipe(
            pad="flat", **{**c, "finger_width_mm": 16.0, "grip_height_frac": 0.7, "pad_height_mm": 10.0}
        ),
        "chamfer": FingerRecipe(
            pad="conform",
            pad_depth_mm=4.0,
            wall_draft_deg=20.0,
            clearance_mm=0.4,
            tip="chamfer",
            tip_size_mm=3.0,
            grip_height_frac=0.4,
            pad_height_mm=16.0,
            finger_width_mm=30.0,
        ),
        "hook": FingerRecipe(
            pad="conform",
            pad_depth_mm=2.0,
            wall_draft_deg=30.0,
            clearance_mm=0.4,
            tip="hook",
            tip_size_mm=6.0,
            grip_height_frac=0.5,
            pad_height_mm=20.0,
            finger_width_mm=30.0,
        ),
        "wide_clear": FingerRecipe(
            pad="conform", pad_depth_mm=8.0, wall_draft_deg=30.0, clearance_mm=1.2, **c
        ),
    }


# --------------------------------------------------------------------------- CLI


def _load(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected module:function, got {spec!r}")
    return getattr(importlib.import_module(module), attr)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating | np.bool_):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o).__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rtt.sim.gates_v2", description=__doc__.split("\n")[0])
    parser.add_argument("gate", choices=("gv1", "gv2"))
    parser.add_argument("--builder", required=True, help="module:function (design, part, task) -> FingerPair")
    parser.add_argument("--parts", required=True, help="module:function () -> parts")
    parser.add_argument(
        "--designs",
        required=True,
        help="module:function () -> {name: design} (gv1: {name: (design, clearance)})",
    )
    parser.add_argument("--det-designs", default=None, help="gv2: module:function () -> {name: design}")
    parser.add_argument("--task", default=None, help="task_v2 YAML (default configs/task_v2.yaml)")
    parser.add_argument(
        "--native-ccd", choices=("true", "false"), default=None, help="override physics.native_ccd"
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--seeds", type=int, default=None, help="gv1: seeds per cell; gv2: seeds per condition"
    )
    parser.add_argument("--quick", action="store_true", help="gv2: matrix, failures and mass checks only")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    from rtt.sim.task import load_task_v2

    task = load_task_v2(args.task)
    if args.native_ccd is not None:
        task = task.with_overrides(physics={"native_ccd": args.native_ccd == "true"})
    builder, parts, designs = _load(args.builder), _load(args.parts)(), _load(args.designs)()
    if args.gate == "gv1":
        report = run_gv1(builder, parts, designs, task, seeds=args.seeds or GV1_SEEDS, workers=args.workers)
    else:
        quick: dict[str, Any] = {}
        if args.quick:
            quick = {
                "parity_timestep": False,
                "native_ccd_parity": False,
                "fresh_processes": 0,
                "cost_samples": 0,
                "parity_seeds_per_condition": 0,
            }
        report = run_gv2(
            builder,
            parts,
            designs,
            task,
            det_designs=_load(args.det_designs)() if args.det_designs else None,
            seeds_per_condition=args.seeds or 2,
            workers=args.workers,
            **quick,
        )
    text = json.dumps(report, indent=1, default=_json_default)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps(report["passed"], indent=1))
    return 0 if report["passed"]["all"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
