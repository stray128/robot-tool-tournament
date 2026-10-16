"""SimEvaluatorV2: compiles each recipe per part, runs the v2 episodes, scores registered grasps.

Track G's builder and track S's ``evaluate_designs_v2`` are injected here (they land
separately): a fake suite and a fake simulator with S's shapes (``SuiteV2`` conditions,
``EpisodeRecordV2`` fields, records ordered by design, condition, then seed). The test pins how
the adapter turns their outputs into search scores.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from rtt.design.freeform.types import CaptureReport, FingerPair, JawGeometry
from rtt.parts import Feature, Material, PartSpec
from rtt.search.fakes_v2 import stub_entrants
from rtt.search.runner import run_search
from rtt.search.sim_adapter import SimEvaluatorV2, episode_from_record_v2, part_info_from_spec
from rtt.search.space import V2, get_space

SPACE = get_space(V2)
PVC = Material("white_pvc", density=1400.0, base_color=(0.92, 0.92, 0.9))


def _corners(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _part(name: str, size: tuple[float, float, float], **kw: Any) -> PartSpec:
    sx, sy, sz = size
    corners = _corners((-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2))
    v = sx * sy * sz
    return PartSpec(
        name=name,
        family=name.split("_")[0],
        pieces=[corners],
        faces=[ConvexHull(corners).simplices],
        volume=v,
        com=(0.0, 0.0, 0.0),
        unit_inertia=(
            v * (sy**2 + sz**2) / 12,
            v * (sx**2 + sz**2) / 12,
            v * (sx**2 + sy**2) / 12,
            0.0,
            0.0,
            0.0,
        ),
        material=PVC,
        decomposition="exact",
        **kw,
    )


PARTS = (
    _part("block_a", (0.04, 0.03, 0.06), yaw_symmetry=2),
    _part(
        "tee_a",
        (0.05, 0.035, 0.023),
        features=(Feature("socket", "branch_socket", (0.0, 0.0175, 0.0), (0.0, 1.0, 0.0), 0.0105, 0.012),),
    ),
)


def _jaw(n: int, mass: float) -> JawGeometry:
    pieces = [
        _corners((-0.01, 0.0004 + 0.001 * i, -0.07), (0.01, 0.0014 + 0.001 * i, -0.05)) for i in range(n)
    ]
    return JawGeometry(pieces, mass, [0.0, 0.0, -0.03], [2e-5, 2e-5, 1e-6, 0, 0, 0], "a" * 64)


def fake_build(design: Any, part: PartSpec, task: Any) -> FingerPair:
    ident = SPACE.design_id(design)
    n = 2 + int(ident[0], 16) % 5
    return FingerPair(
        left=_jaw(n, 0.030),
        right=_jaw(n + 1, 0.034),
        tip_height=0.006,
        finger_length=0.07,
        q_seat=0.02,
        design_id=ident,
        part_id=part.part_id,
        geometry_hash=hashlib.sha256(f"{ident}:{part.part_id}".encode()).hexdigest(),
        capture=CaptureReport(no_jam_mm=3.0, centring_mm=2.0 if part.part_id == "tee_a" else 0.5, mu=0.3),
        build_s=0.1,
    )


@dataclass(frozen=True)
class Placement:
    err_x: float = 0.0
    err_yaw: float = 0.0


@dataclass(frozen=True)
class Condition:
    """Track O's PartCondition shape (the part id comes from the part)."""

    condition_id: str
    part: PartSpec
    placement: Placement
    seeds: tuple[int, ...]
    placement_label: str


@dataclass(frozen=True)
class Suite:
    """Track O's PartSuite shape: every part x c0-c3 x 2 seeds; ``parts()`` is a method."""

    name: str
    conditions: tuple[Condition, ...]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.conditions)

    def parts(self) -> tuple[PartSpec, ...]:
        return tuple({c.part.part_id: c.part for c in self.conditions}.values())

    def digest(self) -> str:
        return "suite-digest"


PLACEMENTS = {"c0": (0.0, 0.0), "c1": (0.002, 0.0), "c2": (-0.003, 0.0), "c3": (0.001, math.radians(6.0))}
SUITE = Suite(
    "train_v2",
    tuple(
        Condition(f"train_v2/{p.part_id}/{label}", p, Placement(dx, dyaw), (11, 12), label)
        for p in PARTS
        for label, (dx, dyaw) in PLACEMENTS.items()
    ),
)
TEE_C3 = "train_v2/tee_a/c3"


def fake_simulator(designs, suite, *, builder, executor, workers, task):  # type: ignore[no-untyped-def]
    """S's evaluate_designs_v2 contract: one record per (design, condition, seed), in that order."""
    assert executor == "cpu" and task == "task-v2"
    out = []
    for design in designs:
        pairs = {p.part_id: builder(design, p, task) for p in suite.parts()}
        for cond in suite:
            pid = cond.part.part_id
            for seed in cond.seeds:
                did = pairs[pid].design_id
                u = (
                    int(hashlib.sha256(f"{did}:{cond.condition_id}:{seed}".encode()).hexdigest()[:8], 16)
                    / 2**32
                )
                lift = u < 0.8
                registered = lift and abs(cond.placement.err_x) < 0.0025 and u < 0.6
                jam = lift and not registered and pid == "block_a"
                bad = seed == 12 and cond.condition_id == TEE_C3
                out.append(
                    {
                        "design_id": did,
                        "part_id": pid,
                        "condition_id": cond.condition_id,
                        "placement_label": cond.placement_label,
                        "seed": seed,
                        "success": registered and not bad,
                        "lift_success": lift and not bad,
                        "registered": registered,
                        "registration_flags": ["jam_on_close"]
                        if jam
                        else ([] if registered else ["off_centre"]),
                        "failure_mode": "physics_invalid"
                        if bad
                        else (None if registered else ("not_registered" if lift else "slip")),
                        "lift": 0.12 if lift else 0.01,
                        "hold_frac": 1.0 if lift else 0.2,
                        "seat_short_mm": 2.0 if jam else (float("nan") if not lift else 0.1),
                        "reg_dx_mm": 1000.0 * abs(cond.placement.err_x),
                        "bad_qacc": int(bad),
                        "finger_mass_g": 64.0,
                    }
                )
    return out


def fake_geometry(design: Any, part_id: str) -> str:
    return hashlib.sha256(f"{SPACE.design_id(design)}:{part_id}".encode()).hexdigest()


def evaluator() -> SimEvaluatorV2:
    return SimEvaluatorV2(
        SUITE,
        "task-v2",
        evaluate_records=fake_simulator,
        builder=fake_build,
        task_hash="t",
        suite_hash=None,
        validator=lambda design, ctx: None,  # schema only: the fake parts are not G's to build
        geometry_hash_fn=fake_geometry,
    )


def test_describe_exposes_parts_placements_and_conditions_only() -> None:
    info = evaluator().describe()
    assert info.design_space == V2 and info.episodes_per_design == 2 * 4 * 2
    assert info.task_hash == "t" and info.suite_hash == "suite-digest"
    assert [p.part_id for p in info.v2.parts] == ["block_a", "tee_a"]
    tee = info.v2.part("tee_a")
    assert tee.size_mm == (50.0, 35.0, 23.0) and tee.material == "white pvc"
    assert tee.features == ("socket 'branch_socket' facing the left jaw, radius 10.5 mm, 12 mm deep",)
    assert [(p.placement_id, p.dx_mm, p.dyaw_deg) for p in info.v2.placements] == [
        ("c0", 0.0, 0.0),
        ("c1", 2.0, 0.0),
        ("c2", -3.0, 0.0),
        ("c3", 1.0, 6.0),
    ]
    assert (
        info.v2.placement_of("train_v2/tee_a/c2") == "c2" and info.v2.part_of("train_v2/tee_a/c2") == "tee_a"
    )
    assert info.gripper_task.part_ids == ("block_a", "tee_a") and info.gripper_task.task == "task-v2"
    # A task without the v2 sections falls back to the spec's numbers.
    assert info.v2.opening_mm == 84.0 and info.v2.registration.dx_mm == 0.75 and info.friction == 0.3


def test_scores_aggregate_registration_capture_pieces_and_mass() -> None:
    ev = evaluator()
    recipes = SPACE.sample_sobol(2, None).sample(3)
    scores = ev.evaluate([recipes[0], recipes[1], recipes[0]])
    assert scores[0] == scores[2] and len(ev.last_records) == 2 * 16
    s = scores[0]
    assert s.n_episodes == 16 and s.is_v2 and s.best_k == 0
    assert s.capture == {"block_a": (3.0, 0.5), "tee_a": (3.0, 2.0)}
    assert s.mass_g == 32.0 and s.max_pieces >= 3
    assert set(s.per_placement) == {"c0", "c1", "c2", "c3"} and set(s.per_family) == {"block_a", "tee_a"}
    eps = [e for e in s.episodes if e.condition_id == TEE_C3 and e.seed == 12]
    assert eps[0].failure_mode == "physics_invalid" and not eps[0].success
    assert all(e.jam_on_close == (e.seat_short_mm is not None and e.seat_short_mm > 1.0) for e in s.episodes)
    assert all(not (e.jam_on_close and e.off_centre) for e in s.episodes)
    assert s.success == sum(e.success for e in s.episodes) / 16
    assert s.lift_success == sum(bool(e.lift_success) for e in s.episodes) / 16
    # NaN registration values (not measured) become None, so every score serialises as JSON.
    assert any(e.seat_short_mm is None for e in s.episodes)
    import json

    json.dumps(s.to_json(), allow_nan=False)


def test_a_record_count_mismatch_is_refused() -> None:
    import pytest

    def short(*args, **kwargs):  # type: ignore[no-untyped-def]
        return fake_simulator(*args, **kwargs)[:-1]

    ev = SimEvaluatorV2(SUITE, "task-v2", evaluate_records=short, builder=fake_build, task_hash="t")
    with pytest.raises(RuntimeError, match="records"):
        ev.evaluate(SPACE.sample_sobol(2, None).sample(1))


def test_records_in_another_design_order_are_refused() -> None:
    """Scores are matched to designs by position, so records that name another design must stop the run."""
    import pytest

    def reversed_designs(designs, suite, **kwargs):  # type: ignore[no-untyped-def]
        return fake_simulator(list(reversed(designs)), suite, **kwargs)

    ev = SimEvaluatorV2(
        SUITE, "task-v2", evaluate_records=reversed_designs, builder=fake_build, task_hash="t"
    )
    with pytest.raises(RuntimeError, match="not in design order"):
        ev.evaluate(SPACE.sample_sobol(3, None).sample(2))


def test_record_conversion_prefers_success_v2_and_derives_flags() -> None:
    record = {
        "condition_id": "block_a_c1",
        "success": True,
        "registered": False,
        "seat_short_mm": 1.6,
        "seed": 0,
    }
    ep = episode_from_record_v2(record, {"block_a_c1": "block_a"}, {"block_a_c1": "c1"})
    assert not ep.success and ep.lift_success and ep.jam_on_close and not ep.off_centre
    assert ep.failure_mode == "not_registered" and ep.part_id == "block_a" and ep.placement == "c1"
    ok = episode_from_record_v2({**record, "success_v2": True, "registered": True}, {}, {})
    assert ok.success and ok.failure_mode is None and not ok.jam_on_close
    # Track S's record: success is success_v2, lift_success the v1 flag, sub-flags in registration_flags.
    s_record = {
        "condition_id": "c",
        "seed": 3,
        "success": False,
        "lift_success": True,
        "registered": False,
        "registration_flags": ["off_centre"],
        "failure_mode": "not_registered",
        "seat_short_mm": 0.2,
        "reg_dx_mm": float("nan"),
        "placement_label": "c2",
    }
    ep = episode_from_record_v2(s_record, {"c": "tee_a"}, {})
    assert not ep.success and ep.lift_success and ep.off_centre and not ep.jam_on_close
    assert ep.reg_dx_mm is None and ep.placement == "c2" and ep.failure_mode == "not_registered"


def test_the_runner_drives_a_v2_search_on_the_adapter(tmp_path: Path) -> None:
    out = run_search("random", 1, 16, [4], evaluator(), tmp_path / "r", entrants=stub_entrants())
    ctx = out.runs[0].ctx
    assert len(ctx.history) == 16 and out.manifest["design_space"] == V2
    assert out.manifest["evaluator"]["episodes_per_design"] == 16
    assert out.runs[0].frozen.design_space == V2


def test_part_reach_slices_the_pieces_band_by_band() -> None:
    from rtt.search.sim_adapter import part_reach

    box = _corners((-0.02, -0.015, 0.0), (0.02, 0.015, 0.06))
    assert part_reach([box]) == pytest.approx((0.015, 0.015))
    # A flanged body: the 10 mm body above a 5 mm flange of radius 18 mm sets the reach.
    flange = _corners((-0.018, -0.018, 0.0), (0.018, 0.018, 0.005))
    body = _corners((-0.01, -0.01, 0.005), (0.01, 0.01, 0.035))
    assert part_reach([flange, body]) == pytest.approx((0.01, 0.01))
    # An L lying flat with its 4 mm leg at the +y edge: the right jaw finds no material above the base.
    base = _corners((-0.015, -0.02, 0.0), (0.015, 0.016, 0.004))
    leg = _corners((-0.015, 0.016, 0.0), (0.015, 0.02, 0.04))
    left, right = part_reach([base, leg])
    assert left == pytest.approx(0.02) and right == pytest.approx(-0.016)
    info = part_info_from_spec(PARTS[0])
    assert info.reach_mm == (15.0, 15.0) and info.min_reach_mm == 15.0
