"""Use-case descriptions: the three hero examples, the gate set and the prompt's worked example.

* :data:`HERO_CASES`: the intake page's three example prompts (warehouse tote, pipe coupling,
  field-maintenance cabinet), written the way a user would type them.
* :data:`GATE_CASES`: the heroes plus five more realistic descriptions (warehouse,
  construction, maintenance, lab, retail) and two the simulator cannot do (cloth, liquid),
  which must be refused with a reason and a nearest primitive (track S4's gate).
* :func:`worked_example`: the format example in the prompt, a different use case (a drawer and
  a small box), built from the contract so it is always valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from rtt.studio.types import ObjectInstance, ObjectPose, Step, StepConstraints, SuccessCriterion, TaskSpec

__all__ = ["GATE_CASES", "HERO_CASES", "UPLOAD_CASE", "UPLOAD_EXAMPLE_STL", "UseCase", "worked_example"]


@dataclass(frozen=True)
class UseCase:
    """One description of the gate set (``expect``: a plan, or a refusal)."""

    case_id: str
    setting: str
    title: str
    text: str
    expect: Literal["plan", "refusal"] = "plan"


HERO_CASES: Final[tuple[UseCase, ...]] = (
    UseCase(
        "h1_tote",
        "warehouse",
        "Returns tote",
        "Warehouse returns: a plastic tote on the packing bench has a hinged lid. Open the lid, put a "
        "small rigid carton (about 120 x 80 x 60 mm, 300 g) and a flat shrink-wrapped pack (about "
        "150 x 100 x 20 mm, 150 g) from the bench into the tote, then close the lid.",
    ),
    UseCase(
        "h2_coupling",
        "construction",
        "Pipe coupling",
        "Pipe prefabrication: pick a threaded steel coupling (1 inch pipe size, about 40 mm across "
        "and 45 mm long) from the bench, start it on the threaded stub of a pipe spool that stands "
        "upright, and turn it on by hand until it is snug, without going over 8 N m.",
    ),
    UseCase(
        "h3_cabinet",
        "maintenance",
        "Equipment cabinet",
        "Field maintenance on an equipment cabinet: turn the quarter-turn latch to unlock the door, "
        "swing the door open, turn the valve handwheel inside a quarter turn to open it, then plug "
        "the test connector into the service port.",
    ),
)

GATE_CASES: Final[tuple[UseCase, ...]] = (
    *HERO_CASES,
    UseCase(
        "e1_drawer_kitting",
        "warehouse",
        "Drawer kitting",
        "Kitting at a parts shelf: pull open the drawer of the shelf unit, take a 50 mm cube part off "
        "the bench and a short steel rod (20 mm across, 60 mm long) next to it, drop both into the "
        "kit tray, then push the drawer closed again.",
    ),
    UseCase(
        "e2_rod_nut",
        "construction",
        "Nut on a threaded rod",
        "On a framing jig, take an M10 hex nut from the tray and run it down an upright threaded rod "
        "three full turns. Keep the torque under 3 N m so the thread is not damaged.",
    ),
    UseCase(
        "e3_isolation_valve",
        "maintenance",
        "Isolation valve",
        "Plant maintenance round: open the hatch of a valve box, close the isolation valve inside by "
        "turning its handwheel one full turn clockwise against about 4 N m, then close the hatch.",
    ),
    UseCase(
        "e4_vial_holder",
        "lab",
        "Vial into a holder",
        "Lab automation: lift the hinged cover of a benchtop instrument, pick a sample vial (12 mm "
        "across, 60 mm tall) from the bench and press it into the 12.4 mm holder of the instrument, "
        "then close the cover.",
    ),
    UseCase(
        "e5_fridge_restock",
        "retail",
        "Fridge restock",
        "Store back room: open the glass door of a drinks fridge, take a can (66 mm across, 115 mm "
        "tall, 380 g) from the crate on the floor stand and put it in the basket on the fridge shelf, "
        "then close the door.",
    ),
    UseCase(
        "u1_fold_shirt",
        "unsupported",
        "Fold a shirt",
        "Take a T-shirt from the laundry pile, fold it in thirds on the table and put it in the box.",
        expect="refusal",
    ),
    UseCase(
        "u2_pour",
        "unsupported",
        "Pour coffee",
        "Pour hot coffee from the jug into three cups without spilling, filling each about 200 ml.",
        expect="refusal",
    ),
)


#: An example part of the repository (CC0, ``assets/parts``) for the recorded upload plan.
UPLOAD_EXAMPLE_STL: Final = "assets/parts/rounded_tub.stl"
#: A use case that comes with an STL (the recorded upload plan, ``recorded/upload-2026-10-18``).
UPLOAD_CASE: Final = UseCase(
    "x1_upload_tub",
    "warehouse",
    "Uploaded tub",
    "Kitting: open the hinged lid of the kit tote, pick the small plastic tub I uploaded off the bench "
    "and put it in the tote, then close the lid.",
)


def worked_example() -> TaskSpec:
    """The prompt's format example (module docstring)."""
    objects = (
        ObjectInstance.of(
            "parts_drawer",
            "drawer",
            "Parts drawer",
            ObjectPose(-150.0, 120.0, 0.0),
            width_mm=300,
            height_mm=90,
            depth_mm=350,
            travel_mm=220,
            pull_force_n=8,
            handle="bar",
            handle_mm=25,
        ),
        ObjectInstance.of(
            "small_box",
            "part",
            "Small box",
            ObjectPose(60.0, -80.0, 0.0),
            shape="box",
            size_x_mm=60,
            size_y_mm=40,
            size_z_mm=30,
            mass_g=120,
        ),
        ObjectInstance.of(
            "tote",
            "bin",
            "Tote",
            ObjectPose(180.0, 60.0, 0.0),
            length_mm=300,
            width_mm=200,
            height_mm=120,
            wall_mm=4,
            fixed=True,
        ),
    )
    steps = (
        Step.of(
            "open_drawer",
            "hook_pull",
            "parts_drawer",
            "handle",
            "Hook the drawer's bar handle and pull the drawer open.",
            [SuccessCriterion("travel_mm", at_least=180.0), SuccessCriterion("force_n", at_most=40.0)],
            params={"speed_mm_s": 60.0, "approach": "front"},
            constraints=StepConstraints(max_force_n=40.0),
        ),
        Step.of(
            "pick_box",
            "grasp",
            "small_box",
            "part",
            "Pick the small box off the bench and drop it in the tote.",
            [SuccessCriterion("in_target", at_least=1.0), SuccessCriterion("tilt_deg", at_most=30.0)],
            target="tote",
            params={"lift_mm": 150.0, "hold_s": 1.0},
            after=("open_drawer",),
        ),
        Step.of(
            "close_drawer",
            "push",
            "parts_drawer",
            "drawer",
            "Push the drawer front until the drawer is shut.",
            [SuccessCriterion("travel_mm", at_most=5.0), SuccessCriterion("force_n", at_most=40.0)],
            params={"speed_mm_s": 60.0},
            constraints=StepConstraints(max_force_n=40.0),
            after=("pick_box",),
        ),
    )
    return TaskSpec(
        "Drawer and tote",
        "Pull a parts drawer open, move a small box from the bench into a tote, and push the drawer shut.",
        objects,
        steps,
        assumptions=(
            "The drawer slides with about 8 N of friction.",
            "The box is 60 x 40 x 30 mm and 120 g.",
        ),
        clarifications=(),
        source="manual",
    )
