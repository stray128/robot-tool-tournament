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

:class:`SimEvaluatorV2` does the same for ``freeform-v2`` recipes on top of track S's
:func:`rtt.sim.evaluate.evaluate_designs_v2`: every recipe is compiled against every training
part (track G's ``build_pair``, cached on the run's :class:`~rtt.search.space_v2.V2Context`),
run at every placement and jitter seed of the ``train_v2`` suite, and aggregated with
:meth:`~rtt.search.base.DesignScore.from_episodes_v2` (registered-grasp score, lift-only
tie-break).

The adapter only ever receives a training suite. The held-out suite is built by the
held-out job from the vault; nothing here can reach it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rtt.design.schema import FingerDesign, GripperTask
from rtt.search.base import ConditionInfo, DesignScore, EpisodeOutcome, EvaluatorInfo

__all__ = [
    "SimEvaluator",
    "SimEvaluatorV2",
    "describe_condition",
    "episode_from_record_v2",
    "part_info_from_spec",
    "v2_info_from_task",
]


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


# --------------------------------------------------------------------------- freeform-v2


def _attr(obj: Any, path: str, default: Any) -> Any:
    """``obj.a.b`` for ``path="a.b"``, or ``default`` when any step is missing."""
    for name in path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return default
    return obj


def _feature_text(f: Any) -> str:
    """A part feature in words: kind, name, which way it opens and its size (``mm``)."""
    facing = f.facing() if hasattr(f, "facing") else None
    if facing:
        where = f" facing the {facing} jaw"
    else:
        ax = [float(v) for v in f.axis]
        i = max(range(3), key=lambda k: abs(ax[k]))
        sign = "+" if ax[i] > 0 else "-"
        where = (
            (" opening up" if ax[i] > 0 else " opening down")
            if i == 2
            else f" opening toward {sign}{'xy'[i]}"
        )
    if f.kind == "flange":
        side = (
            " at the bottom" if float(f.axis[2]) < 0 else (" at the top" if float(f.axis[2]) > 0 else where)
        )
        return f"flange '{f.name}'{side}, radius {_mm(f.radius)} mm, {_mm(f.depth)} mm thick"
    return f"{f.kind} '{f.name}'{where}, radius {_mm(f.radius)} mm, {_mm(f.depth)} mm deep"


#: Heights the jaws can grip from (the controller's minimum tip clearance) and the shortest pad band.
_REACH_FROM_M = 0.003
_REACH_BAND_M = 0.008


def part_reach(pieces: Sequence[Any], step: float = 0.0005) -> tuple[float, float] | None:
    """How far a part reaches toward the left (+y) and right (-y) jaw at its narrowest band, in m.

    The part's convex pieces are sliced every ``step`` from the table up; within each 8 mm band
    (the shortest pad) starting 3 mm or more above the table the reach is the widest slice, and
    the result is the narrowest band. A side with no material in some band (the jaw would have
    to cross the centre plane) gives a negative reach. ``None`` when the pieces are degenerate.
    """
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError

    hulls = []
    for piece in pieces:
        v = np.asarray(piece, dtype=np.float64)
        try:
            simplices = ConvexHull(v).simplices
        except (QhullError, ValueError):
            continue
        edges = {
            tuple(sorted((int(a), int(b))))
            for tri in simplices
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2]))
        }
        hulls.append((v, np.array(sorted(edges))))
    if not hulls:
        return None
    z0 = min(float(v[:, 2].min()) for v, _ in hulls)
    z1 = max(float(v[:, 2].max()) for v, _ in hulls)
    zs = np.arange(z0 + _REACH_FROM_M, z1 + 1e-12, step)
    left = np.full(len(zs), -np.inf)
    right = np.full(len(zs), -np.inf)
    for v, edges in hulls:
        a, b = v[edges[:, 0]], v[edges[:, 1]]
        for k, z in enumerate(zs):
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            hit = (lo <= z) & (hi >= z)
            if not np.any(hit):
                continue
            aa, bb = a[hit], b[hit]
            dz = bb[:, 2] - aa[:, 2]
            t = np.where(np.abs(dz) > 1e-15, (z - aa[:, 2]) / np.where(np.abs(dz) > 1e-15, dz, 1.0), 0.0)
            ys = np.concatenate(
                [aa[:, 1] + t * (bb[:, 1] - aa[:, 1]), aa[np.abs(dz) <= 1e-15, 1], bb[np.abs(dz) <= 1e-15, 1]]
            )
            left[k] = max(left[k], float(ys.max()))
            right[k] = max(right[k], float(-ys.min()))
    n = max(1, round(_REACH_BAND_M / step))
    if len(zs) <= n:
        return float(left.max()), float(right.max())
    band_left = [float(left[i : i + n + 1].max()) for i in range(len(zs) - n)]
    band_right = [float(right[i : i + n + 1].max()) for i in range(len(zs) - n)]
    return min(band_left), min(band_right)


def part_info_from_spec(spec: Any) -> Any:
    """A :class:`~rtt.search.base.PartInfo` (what a prompt may say) from a track-O ``PartSpec``."""
    from rtt.search.base import PartInfo

    features = [_feature_text(f) for f in getattr(spec, "features", ())]
    size = tuple(1000.0 * float(v) for v in spec.size)
    reach = part_reach(getattr(spec, "pieces", ()) or ())
    return PartInfo(
        part_id=str(spec.part_id),
        family=str(spec.family),
        description=str(getattr(spec, "description", "") or str(spec.family).replace("_", " ")),
        size_mm=(round(size[0], 1), round(size[1], 1), round(size[2], 1)),
        material=str(spec.material.name).replace("_", " "),
        mass_g=round(1000.0 * float(spec.mass), 1),
        yaw_symmetry=int(spec.yaw_symmetry),
        features=tuple(features),
        reach_mm=None if reach is None else (round(1000.0 * reach[0], 1), round(1000.0 * reach[1], 1)),
    )


def v2_info_from_task(
    task: Any, parts: Sequence[Any], placements: Sequence[Any], conditions: Sequence[Any]
) -> Any:
    """:class:`~rtt.search.base.V2Info` with the gripper and registration numbers of a ``TaskV2``."""
    from rtt.search.base import RegistrationInfo, V2Info

    reg = getattr(task, "registration", None)
    registration = (
        RegistrationInfo(
            seat_short_mm=float(reg.seat_short_max_mm),
            dx_mm=float(reg.dx_max_mm),
            dy_mm=float(reg.dy_max_mm),
            yaw_deg=float(reg.yaw_max_deg),
            tilt_deg=float(reg.tilt_max_deg),
            tilt_after_hold_deg=float(reg.hold_tilt_max_deg),
            settle_s=float(reg.settle_time),
        )
        if reg is not None
        else RegistrationInfo()
    )
    return V2Info(
        parts=tuple(parts),
        placements=tuple(placements),
        conditions=tuple(conditions),
        registration=registration,
        opening_mm=round(2000.0 * float(_attr(task, "gripper.open_half_gap", 0.042)), 3),
        stroke_mm=round(1000.0 * float(_attr(task, "gripper.stroke", 0.042)), 3),
        grip_force_per_jaw_n=float(_attr(task, "gripper.grip_force", 50.0)) / 2.0,
        palm_clearance_mm=round(1000.0 * float(_attr(task, "controller.palm_clearance", 0.01)), 3),
        min_tip_height_mm=round(1000.0 * float(_attr(task, "controller.min_tip_clearance", 0.003)), 3),
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def episode_from_record_v2(
    record: Any, part_of: dict[str, str], placement_of: dict[str, str]
) -> EpisodeOutcome:
    """One v2 simulator record (``EpisodeRecordV2``, ``RECORD_VERSION`` 2) as a search episode.

    ``success`` is the registered grasp (``success_v2``); ``lift_success`` is the v1
    lift-and-hold flag. ``jam_on_close`` / ``off_centre`` come from the record's
    ``registration_flags`` (or its own fields, else ``seat_short_mm``). An episode whose
    ``bad_qacc`` counter rose is ``physics_invalid`` and never a success. Registration values
    the simulator could not measure (NaN) become ``None``.
    """
    cid = str(_get(record, "condition_id"))
    lift = bool(_get(record, "lift_success", _get(record, "success")))
    registered = bool(_get(record, "registered", False))
    success_v2 = _get(record, "success_v2")
    if success_v2 is None and _get(record, "lift_success") is not None:
        success_v2 = _get(record, "success")  # an EpisodeRecordV2's success is success_v2
    success = bool(success_v2) if success_v2 is not None else (lift and registered)
    flags = tuple(_get(record, "registration_flags", ()) or ())
    seat_short = _finite_or_none(_get(record, "seat_short_mm"))
    jam = _get(record, "jam_on_close")
    if jam is None:
        jam = "jam_on_close" in flags if flags else (seat_short is not None and seat_short > 1.0)
    off = _get(record, "off_centre")
    if off is None:
        off = "off_centre" in flags if flags else (not registered and not jam)
    mode = _get(record, "failure_mode")
    if bool(_get(record, "bad_qacc", 0)):  # spec section 6: a MuJoCo state reset invalidates the episode
        mode, success = "physics_invalid", False
    part_id = str(_get(record, "part_id") or part_of.get(cid, _get(record, "family", "")))
    placement = placement_of.get(cid) or _get(record, "placement_label")

    def opt(name: str) -> float | None:
        return _finite_or_none(_get(record, name))

    return EpisodeOutcome(
        condition_id=cid,
        family=part_id,
        controller_setting=0,
        seed=int(_get(record, "seed", 0)),
        success=success,
        failure_mode=None if success else (mode or ("not_registered" if lift else "unknown")),
        max_lift_m=float(_finite_or_none(_get(record, "lift", _get(record, "max_lift_m", 0.0))) or 0.0),
        hold_frac=float(_finite_or_none(_get(record, "hold_frac", 0.0)) or 0.0),
        lift_success=lift,
        registered=registered,
        jam_on_close=bool(jam) and not success,
        off_centre=bool(off) and not success and not bool(jam),
        seat_short_mm=seat_short,
        reg_dx_mm=opt("reg_dx_mm"),
        reg_dy_mm=opt("reg_dy_mm"),
        reg_dyaw_deg=opt("reg_dyaw_deg"),
        reg_tilt_deg=opt("reg_tilt_deg"),
        part_id=part_id,
        placement=None if placement is None else str(placement),
    )


class SimEvaluatorV2:
    """Training-suite evaluator for ``freeform-v2`` recipes (spec sections 7, 8 and 12).

    Args:
        suite: The ``train_v2`` suite: track S's :class:`~rtt.sim.conditions.SuiteV2` (or any
            object with ``name``, ``parts`` and conditions that carry ``condition_id``,
            ``part_id``, ``placement_label``, ``placement`` with ``err_x`` / ``err_yaw``, and
            ``seeds``). Its parts are the training parts, in training order.
        task: The v2 task (default: ``rtt.sim.task.default_task_v2()``).
        evaluate_records: ``(designs, suite, *, builder, executor, workers, task) -> records``,
            records ordered by design, condition, then seed (default: track S's
            :func:`rtt.sim.evaluate.evaluate_designs_v2`).
        builder: ``(design, part, task) -> FingerPair`` (default: the space's builder, which
            calls track G's ``build_pair`` and caches pairs on the run's context).
        executor, workers: Passed through to ``evaluate_records``.
        task_hash, suite_hash: Identity for manifests (default: S's ``task_digest`` and the
            suite's ``digest()`` when they exist).
        validator, geometry_hash_fn: Overrides for the run's
            :class:`~rtt.search.space_v2.V2Context` (tests and dry runs; by default track G's
            validators and geometry hashes are used).
    """

    def __init__(
        self,
        suite: Any,
        task: Any = None,
        *,
        evaluate_records: Any = None,
        builder: Any = None,
        executor: Any = "cpu",
        workers: int | None = None,
        task_hash: str | None = None,
        suite_hash: str | None = None,
        validator: Any = None,
        geometry_hash_fn: Any = None,
    ) -> None:
        from rtt.search.space import V2, get_space
        from rtt.search.space_v2 import V2Context

        if task is None:
            from rtt.sim.task import default_task_v2

            task = default_task_v2()
        self.space = get_space(V2)
        self.suite = suite
        self.task = task
        self.parts = tuple(suite.parts)
        if not self.parts:
            raise ValueError("a v2 evaluator needs the training parts")
        self.conditions = tuple(suite)
        self.n_per_design = sum(len(c.seeds) for c in self.conditions)
        self._evaluate = evaluate_records
        self._builder = builder
        self.executor = executor
        self.workers = workers
        if task_hash is None:
            try:
                from rtt.sim.task import task_digest

                task_hash = task_digest(task)
            except (ImportError, TypeError, AttributeError):
                task_hash = ""
        digest = getattr(suite, "digest", None)
        self.task_hash = task_hash
        self.suite_hash = suite_hash if suite_hash is not None else (digest() if callable(digest) else "")
        self.context = V2Context(
            parts=self.parts,
            task=task,
            label=str(getattr(suite, "name", "train_v2")),
            validator=validator,
            geometry_hash_fn=geometry_hash_fn,
        )
        self.last_records: list[Any] = []

    def describe(self) -> EvaluatorInfo:
        from rtt.search.base import ConditionV2, PlacementInfo

        infos = [part_info_from_spec(p) for p in self.parts]
        width = {i.part_id: i.width_across_mm for i in infos}
        conditions, links, placements = [], [], {}
        for c in self.conditions:
            dx = round(1000.0 * float(getattr(c.placement, "err_x", 0.0)), 3) + 0.0
            dyaw = round(math.degrees(float(getattr(c.placement, "err_yaw", 0.0))), 3) + 0.0
            label = str(getattr(c, "placement_label", "") or c.condition_id)
            placements.setdefault(label, PlacementInfo(label, dx, dyaw))
            conditions.append(
                ConditionInfo(
                    c.condition_id,
                    c.part_id,
                    f"{c.part_id}, {dx:+g} mm across, {dyaw:g} deg",
                    width[c.part_id],
                )
            )
            links.append(ConditionV2(c.condition_id, c.part_id, label))
        seeds = {len(c.seeds) for c in self.conditions}
        if len(seeds) != 1:
            raise ValueError("every v2 training condition must have the same number of jitter seeds")
        task = self.task
        return EvaluatorInfo(
            executor=self.executor if isinstance(self.executor, str) else type(self.executor).__name__,
            conditions=tuple(conditions),
            controller_settings=(),
            seeds_per_condition=seeds.pop(),
            gripper_task=self.context,
            task_hash=self.task_hash,
            suite_hash=self.suite_hash,
            grip_force_n=float(_attr(task, "gripper.grip_force", 50.0)),
            success_lift_m=float(_attr(task, "score.success_lift", 0.10)),
            success_hold_s=float(_attr(task, "score.success_hold", 2.0)),
            max_time_s=float(_attr(task, "score.max_time", 12.0)),
            friction=float(_attr(task, "physics.friction", 0.3)),
            object_density_kg_m3=float(_attr(task, "physics.object_density", 1200.0)),
            jitter_pos_mm=round(1000.0 * float(_attr(task, "jitter.pos", 0.0005)), 3),
            jitter_yaw_deg=float(_attr(task, "jitter.yaw_deg", 1.0)),
            jitter_mu_frac=float(_attr(task, "jitter.mu_frac", 0.10)),
            palm_clearance_m=float(_attr(task, "controller.palm_clearance", 0.01)),
            v2=v2_info_from_task(task, infos, tuple(placements.values()), links),
        )

    def _build(self, design: Any, part: Any, task: Any) -> Any:
        if self._builder is not None:
            return self._builder(self.space.buildable(design), part, task)
        return self.space.build(design, part, self.context)

    def _records(self, designs: Sequence[Any]) -> list[Any]:
        run = self._evaluate
        if run is None:
            try:
                from rtt.sim.evaluate import evaluate_designs_v2 as run
            except ImportError as exc:
                raise NotImplementedError(
                    "the v2 simulator (track S, rtt.sim.evaluate.evaluate_designs_v2) is not available yet"
                ) from exc
        return list(
            run(
                list(designs),
                self.suite,
                builder=self._build,
                executor=self.executor,
                workers=self.workers,
                task=self.task,
            )
        )

    def evaluate(self, designs: Sequence[Any]) -> list[DesignScore]:
        if not designs:
            return []
        unique: dict[str, Any] = {}
        for d in designs:
            unique.setdefault(self.space.design_id(d), d)
        records = self._records(list(unique.values()))
        if len(records) != len(unique) * self.n_per_design:
            raise RuntimeError(
                f"the v2 simulator returned {len(records)} records for {len(unique)} designs x "
                f"{self.n_per_design} episodes"
            )
        self.last_records = records
        part_of = {c.condition_id: c.part_id for c in self.conditions}
        placement_of = {
            c.condition_id: str(getattr(c, "placement_label", "") or c.condition_id) for c in self.conditions
        }
        scores: dict[str, DesignScore] = {}
        for i, (design_id, design) in enumerate(unique.items()):
            chunk = records[i * self.n_per_design : (i + 1) * self.n_per_design]
            episodes = [episode_from_record_v2(r, part_of, placement_of) for r in chunk]
            pairs = {}
            for part in self.parts:
                try:
                    pairs[str(part.part_id)] = self._build(design, part, self.task)
                except ValueError:  # this part scored invalid_build; it has no pair
                    continue
            masses = [500.0 * (pair.left.mass + pair.right.mass) for pair in pairs.values()]
            capture = {
                pid: (float(pair.capture.no_jam_mm), float(pair.capture.centring_mm))
                for pid, pair in pairs.items()
                if getattr(pair, "capture", None) is not None
            }
            pieces = [max(pair.left.n_pieces, pair.right.n_pieces) for pair in pairs.values()]
            scores[design_id] = DesignScore.from_episodes_v2(
                design_id,
                sum(masses) / len(masses) if masses else 0.0,
                episodes,
                placements=placement_of,
                capture=capture or None,
                max_pieces=max(pieces) if pieces else None,
            )
        return [scores[self.space.design_id(d)] for d in designs]
