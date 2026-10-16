"""Gate GV2 pieces (physics, G1-v2): deliberate failures, the matrix statistics, the
native-CCD rule and the lip-edge regression, mass properties of asymmetric jaws.

The full gate (``python -m rtt.sim.gates_v2 gv2 ...``) takes minutes; these tests run its
parts on one or two fixture parts so the suite stays fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from freeform_fixtures import (
    block,
    build_fixture_pair,
    flanged_bushing,
    flat,
    gv2_designs,
    imprint_configurator,
    l_bracket,
    pipe_tee,
    steel_slug,
)
from rtt.sim.episode import EpisodeResultV2
from rtt.sim.gates_v2 import (
    GV2_PLACEMENTS,
    HEAVY_FACTOR,
    LOW_MU,
    build_pairs,
    deliberate_failures,
    lip_regression,
    matrix_jobs,
    matrix_stats,
    native_ccd_rule,
    part_rests_on_jaws,
    run_gv2,
    scale_part,
    with_density,
)
from rtt.sim.task import load_task_v2

TASK = load_task_v2()


def test_deliberate_failures_fail_for_the_right_reason() -> None:
    report = deliberate_failures(
        build_fixture_pair, [block(), steel_slug()], flat(finger_width_mm=40.0), TASK
    )
    assert set(report["cases"]) == {
        f"{case}/{p}"
        for case in ("too_high", "low_friction", "too_wide")
        for p in ("block_30x40x60", "steel_slug_40")
    }
    for label, case in report["cases"].items():
        assert case["passed"], (label, case)
    assert report["cases"]["too_high/block_30x40x60"]["failure_mode"] == "missed_grasp"
    assert report["low_friction_supported"] == []  # prisms: nothing to rest on
    assert report["passed"]


def test_a_flange_resting_on_flat_pads_is_not_a_deliberate_failure() -> None:
    bushing = flanged_bushing()
    pair = build_fixture_pair(flat(finger_width_mm=40.0), bushing, TASK)
    assert part_rests_on_jaws(pair, bushing, TASK)  # the flange sits 2.5 mm above the pads
    assert not part_rests_on_jaws(
        build_fixture_pair(flat(finger_width_mm=40.0), block(), TASK), block(), TASK
    )
    report = deliberate_failures(build_fixture_pair, [bushing], flat(finger_width_mm=40.0), TASK)
    assert report["low_friction_supported"] == ["flanged_bushing_20"]
    assert report["cases"]["low_friction/flanged_bushing_20"]["supported"]
    assert not report["passed"]  # one part, and it was not tested: fewer than half


def test_the_low_friction_copy_is_heavier_than_friction_can_carry() -> None:
    blk = block()
    capacity = 2.0 * LOW_MU * (1.0 + TASK.jitter.mu_frac) * TASK.gripper.grip_force / 2.0
    heavy = with_density(blk, HEAVY_FACTOR * capacity / (9.81 * blk.volume))
    assert heavy.mass * 9.81 == pytest.approx(HEAVY_FACTOR * capacity)
    assert heavy.geometry_sha256 == blk.geometry_sha256  # same geometry: the same jaws fit
    assert heavy.to_mesh_object().density == heavy.material.density


def test_scale_part_keeps_exact_mass_properties() -> None:
    part = l_bracket()
    scaled = scale_part(part, (1.0, 1.5, 0.8), "wide")
    hull_sum = trimesh.util.concatenate([trimesh.convex.convex_hull(p) for p in scaled.pieces])
    assert scaled.volume == pytest.approx(part.volume * 1.2)
    assert scaled.size == pytest.approx((part.size[0], 1.5 * part.size[1], 0.8 * part.size[2]))
    # the bracket's two legs do not overlap, so the hull sum is the solid (the fixture's
    # volume comes from a float32 mesh union, hence the tolerance)
    assert scaled.volume == pytest.approx(hull_sum.volume, rel=1e-5)
    hull_sum.density = 1.0
    i = hull_sum.moment_inertia
    want = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
    assert np.allclose(scaled.unit_inertia, want, rtol=1e-4, atol=1e-14)
    with pytest.raises(ValueError, match="positive"):
        scale_part(part, (1.0, -1.0, 1.0), "bad")


def test_matrix_jobs_share_seeds_across_designs_and_extend() -> None:
    parts = [block()]
    designs = gv2_designs()
    pairs, invalid = build_pairs(build_fixture_pair, parts, designs, TASK)
    assert not invalid
    keys2, jobs2 = matrix_jobs(pairs, parts, list(designs), TASK, 2)
    keys3, jobs3 = matrix_jobs(pairs, parts, list(designs), TASK, 3)
    assert len(jobs2) == len(designs) * len(GV2_PLACEMENTS) * 2
    seeds = {}
    for (d, p, label, rep), job in zip(keys3, jobs3, strict=True):
        seeds.setdefault((p, label, rep), set()).add(job.seed)
    assert all(len(s) == 1 for s in seeds.values())  # common random numbers
    two = {(k, j.seed) for k, j in zip(keys2, jobs2, strict=True)}
    assert two <= {(k, j.seed) for k, j in zip(keys3, jobs3, strict=True)}


def _fake(success: bool, lift: bool, pen_mm: float, creep_mm: float, bad: int = 0) -> EpisodeResultV2:
    return EpisodeResultV2(
        success=success,
        failure_mode=None if success else ("not_registered" if lift else "slip_during_lift"),
        max_lift=0.15 if lift else 0.0,
        hold_time=2.0 if lift else 0.0,
        metrics={"obj_finger_pen_max": pen_mm / 1000.0, "hold_creep_mm": creep_mm},
        flags=(),
        seed=0,
        controller_setting=-1,
        n_steps=1,
        wall_time_s=0.0,
        lift_success=lift,
        bad_qacc=bad,
    )


def test_matrix_stats_counts_only_held_episodes_for_creep_and_held_penetration() -> None:
    rows = [_fake(True, True, 0.2, 0.1), _fake(False, True, 1.2, 0.3), _fake(False, False, 3.0, math.nan, 1)]
    s = matrix_stats(rows)
    assert s["episodes"] == 3 and s["bad_qacc"] == 1
    assert s["held_rate"] == pytest.approx(2 / 3) and s["registered_rate"] == pytest.approx(1 / 3)
    assert s["pen_held_over_1mm"] == 1 and s["pen_held_max_mm"] == pytest.approx(1.2)
    assert s["pen_max_mm"] == pytest.approx(3.0)
    assert s["creep_max_mm"] == pytest.approx(0.3)
    assert s["failure_modes"] == {"not_registered": 1, "slip_during_lift": 1}


def test_native_ccd_rule_needs_every_condition() -> None:
    lips = {"v1_regression_reproduced": True, "v2_analogues_fixed_with_native": True}
    assert native_ccd_rule(1.0, 0.995, lips)["switch_to_native"]
    assert not native_ccd_rule(0.98, 1.0, lips)["switch_to_native"]
    assert not native_ccd_rule(1.0, 0.98, lips)["switch_to_native"]
    assert not native_ccd_rule(math.nan, 1.0, lips)["switch_to_native"]
    assert not native_ccd_rule(1.0, 1.0, {**lips, "v2_analogues_fixed_with_native": False})[
        "switch_to_native"
    ]


def test_the_lip_edge_regression_is_reproduced_on_v1_primitives() -> None:
    out = lip_regression(TASK)
    off, on = out["v1_primitive_cylinder_native_False"], out["v1_primitive_cylinder_native_True"]
    assert off["success"] and min(off["squeeze_n"]) > 20.0  # radial normals: the squeeze is F/2
    assert out["v1_regression_reproduced"]
    assert not on["success"]
    for variant in ("box_lips", "mesh_lips"):
        for native in (False, True):
            row = out[f"v2_{variant}_mesh_cylinder_native_{native}"]
            assert row["lift_success"], (variant, native, row)
    assert out["v2_analogues_fixed_with_native"]


def test_gv2_quick_run_on_two_parts() -> None:
    parts = [block(), flanged_bushing()]
    designs = {"flat": flat(finger_width_mm=40.0), "conform": gv2_designs()["conform"]}
    report = run_gv2(
        build_fixture_pair,
        parts,
        designs,
        TASK,
        seeds_per_condition=1,
        parity_seeds_per_condition=0,
        workers=2,
        parity_timestep=False,
        native_ccd_parity=False,
        fresh_processes=0,
        cost_samples=0,
    )
    m = report["matrix"]
    assert m["episodes"] == 2 * 2 * len(GV2_PLACEMENTS)
    assert m["bad_qacc"] == 0 and not m["invalid_builds"]
    assert m["finger_length_vs_lowest_point_max_mm"] < 1e-3
    assert set(m["by_cell"]) == {f"{d}/{p.part_id}" for d in designs for p in parts}
    assert report["passed"]["deliberate_failures"]
    assert set(report["passed"]) >= {"bad_qacc_zero", "penetration_held", "creep_p95", "asymmetric_jaws"}


def test_asymmetric_pairs_are_checked_for_their_own_mass_properties() -> None:
    parts = [pipe_tee(), l_bracket()]
    report = run_gv2(
        build_fixture_pair,
        parts,
        {"imprint": imprint_configurator()},
        TASK,
        seeds_per_condition=1,
        parity_seeds_per_condition=0,
        workers=2,
        parity_timestep=False,
        native_ccd_parity=False,
        fresh_processes=0,
        cost_samples=0,
    )
    asym = report["asymmetric_jaws"]
    assert asym["pairs"] and asym["own_mass_properties"]
    for row in asym["pairs"].values():
        assert row["mass_g"] == row["compiled_mass_g"] and row["mass_g"][0] != row["mass_g"][1]


def test_agreement_splits_registered_and_held() -> None:
    from rtt.sim.gates_v2 import _agreement

    a = [_fake(True, True, 0.1, 0.1), _fake(False, True, 0.1, 0.1), _fake(False, False, 0.1, 0.1)]
    b = [_fake(True, True, 0.1, 0.1), _fake(True, True, 0.1, 0.1), _fake(False, True, 0.1, 0.1)]
    out = _agreement(a, b)
    assert out["n"] == 3 and out["disagreements"] == [1, 2]
    assert out["outcome_agreement"] == pytest.approx(1 / 3)
    assert out["registered_agreement"] == pytest.approx(2 / 3)
    assert out["held_agreement"] == pytest.approx(2 / 3)
    assert _agreement([], [])["n"] == 0
