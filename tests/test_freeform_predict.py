"""Static capture predictor: reproduces the 2.5D prototype's simulated tee table (GV0: 24 / 24).

Reference: the prototype simulated six jaw pairs on the 3/8-inch style tee (``mvp_tee``) at
lateral placement errors 0, 1, 2 and 3 mm (CPU MuJoCo 3.15, task_v1 physics, jitter off) and
labelled each episode *registered* when, at the end of closing, the part sat within 0.75 mm
across the jaw, within 2 degrees of yaw and at most 1.0 mm short of the designed seat. The
labels below are those simulation results; the three-way label adds *jammed* (more than 1 mm
short) and *uncentred* (seated but off-centre).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from freeform_parts import block, flanged_bushing, mvp_tee
from rtt.design.freeform.baselines import load_entrants_v2
from rtt.design.freeform.generate import PairSpec, build_pair_full, plan_pair
from rtt.design.freeform.predict import capture_ranges, pair_outcomes, predict, slide_limit_deg
from rtt.design.freeform.schema import FingerRecipe
from rtt.sim.task import default_task

TASK = default_task()
ERRORS_MM = (0.0, 1.0, 2.0, 3.0)

#: Simulated three-way outcome per prototype design at 0 / 1 / 2 / 3 mm (c = centred = registered).
SIMULATED = {
    "flat": ("c", "u", "u", "u"),
    "imprint_tight": ("c", "j", "j", "j"),
    "imprint_swept3": ("c", "u", "u", "u"),
    "conform_d30": ("c", "c", "c", "c"),
    "vfunnel_45": ("c", "c", "c", "c"),
    "conform_d75": ("c", "u", "u", "u"),
}


def _mvp_spec(
    pad: str, *, c: float = 0.0004, d: float = 0.010, draft: float = 0.0, sweep: float = 0.0
) -> PairSpec:
    tee_h = mvp_tee().size[2]
    return PairSpec(
        pad=pad,
        tip="blunt",
        pad_height=0.020,
        width=0.040,
        depth=0.0 if pad == "flat" else d,
        draft=math.radians(draft),
        clearance=c,
        tip_size=0.0,
        tip_height=max(0.003, 0.3 * tee_h),  # the prototype's tip height (controller setting 0)
        sweep=sweep,
    )


#: The prototype's six designs as exact build specs (some lie outside the v2 search bounds).
MVP_DESIGNS = {
    "flat": _mvp_spec("flat"),
    "imprint_tight": _mvp_spec("conform"),
    "imprint_swept3": _mvp_spec("conform", sweep=0.003),
    "conform_d30": _mvp_spec("conform", draft=30.0),
    "vfunnel_45": _mvp_spec("v_funnel", c=0.0, d=0.015, draft=45.0),
    "conform_d75": _mvp_spec("conform", draft=75.0),
}


def _short(outcome: str) -> str:
    return {"centred": "c", "uncentred": "u", "jammed": "j"}[outcome]


def test_predictor_reproduces_the_mvp_tee_table_24_of_24() -> None:
    tee = mvp_tee()
    registered_ok = three_way_ok = 0
    for name, spec in MVP_DESIGNS.items():
        plan = plan_pair(spec, tee, TASK, design_id="0" * 64)
        outs = [_short(o) for o in pair_outcomes(plan.left, plan.right, np.array(ERRORS_MM) / 1000.0, 0.3)]
        for got, want in zip(outs, SIMULATED[name], strict=True):
            registered_ok += (got == "c") == (want == "c")
            three_way_ok += got == want
    assert registered_ok == 24
    # the prototype's predictor also called the 75-degree conform "jammed" where the simulation
    # only left it off-centre: the friction rule's margin is conservative
    assert three_way_ok == 21


def test_predictor_on_v2_canonical_recipes_keeps_the_registered_labels() -> None:
    """The same designs expressed through the v2 schema and baselines (draft capped at 70 degrees,
    V-funnel clearance at its fixed 0.2 mm, the swept imprint at the baseline's 2 mm)."""
    tee = mvp_tee()
    base = {"tip": "blunt", "grip_height_frac": 0.73, "pad_height_mm": 20.0, "finger_width_mm": 40.0}
    recipes = {
        "flat": FingerRecipe(pad="flat", **base),
        "imprint_tight": FingerRecipe(
            pad="conform", pad_depth_mm=10.0, wall_draft_deg=0.0, clearance_mm=0.4, **base
        ),
        "conform_d30": FingerRecipe(
            pad="conform", pad_depth_mm=10.0, wall_draft_deg=30.0, clearance_mm=0.4, **base
        ),
        "vfunnel_45": FingerRecipe(pad="v_funnel", pad_depth_mm=15.0, wall_draft_deg=45.0, **base),
        "conform_d75": FingerRecipe(
            pad="conform", pad_depth_mm=10.0, wall_draft_deg=70.0, clearance_mm=0.4, **base
        ),
    }
    for name, recipe in recipes.items():
        b = build_pair_full(recipe, tee, TASK)
        outs = [_short(predict(b.plan, e)) for e in ERRORS_MM]
        assert [o == "c" for o in outs] == [s == "c" for s in SIMULATED[name]], name


def test_friction_rule_limit() -> None:
    assert slide_limit_deg(0.3) == pytest.approx(90.0 - math.degrees(math.atan(0.3)) - 5.0)
    assert 68.0 < slide_limit_deg(0.3) < 68.5


def test_capture_report_fields_and_ranges() -> None:
    entrants = {b.name: b for b in load_entrants_v2()}
    recipe = FingerRecipe(
        pad="conform",
        tip="blunt",
        grip_height_frac=0.5,
        pad_height_mm=20.0,
        finger_width_mm=40.0,
        pad_depth_mm=6.0,
        wall_draft_deg=35.0,
        clearance_mm=0.4,
    )
    good = build_pair_full(recipe, block(), TASK).pair.capture
    imprint = build_pair_full(entrants["imprint_configurator"], block(), TASK).pair.capture
    assert [e for e, _ in good.by_error] == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert dict(good.by_error)[0.0] == "centred"
    assert good.centring_mm >= 3.0  # drafted walls pull a 3 mm error to centre
    assert imprint.no_jam_mm < 1.0  # the straight-walled imprint jams on the block's edge
    assert dict(imprint.by_error)[2.0] == "jammed"
    assert good.mu == pytest.approx(0.3)


def test_capture_is_cheap() -> None:
    b = build_pair_full(
        FingerRecipe(
            pad="v_funnel",
            tip="blunt",
            grip_height_frac=0.5,
            pad_height_mm=20.0,
            finger_width_mm=30.0,
            pad_depth_mm=6.0,
            wall_draft_deg=40.0,
        ),
        flanged_bushing(),
        TASK,
    )
    t = time.perf_counter()
    for _ in range(20):
        capture_ranges(b.plan.left, b.plan.right, 0.3)
    assert (time.perf_counter() - t) / 20 < 0.02
