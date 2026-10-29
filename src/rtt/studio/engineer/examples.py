"""Use-case descriptions: the hero examples, the gate sets and the prompt's worked example.

* :data:`HERO_CASES`: Studio v2's six hero jobs, written the way a user would type them, with the
  sizes and limits they would know: a warehouse returns tote, CNC machine tending, a solar panel
  install, floor tile setting, a pipe coupling and a hotel delivery. They are the v2 gate set
  (:data:`GATE_CASES`), recorded live in ``recorded/gate-2026-10-27-v2c`` (prompt and rules v2).
* :data:`V1_HERO_CASES` and :data:`V1_GATE_CASES`: the v1 descriptions exactly as recorded in
  ``recorded/gate-2026-10-18-s4e`` (the three v1 heroes, five more realistic ones and two the
  simulator cannot do). The pre-registered use-case benchmark v1 froze those plans; they stay
  replayable from their recorded responses (:func:`rtt.studio.engineer.gate.replay_recorded`).
* :func:`worked_example`: the format example in the prompt, a different use case (a drawer and
  a small box), built from the contract so it is always valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from rtt.studio.types import ObjectInstance, ObjectPose, Step, StepConstraints, SuccessCriterion, TaskSpec

__all__ = [
    "GATE_CASES",
    "HERO_CASES",
    "UPLOAD_CASE",
    "UPLOAD_EXAMPLE_STL",
    "V1_GATE_CASES",
    "V1_HERO_CASES",
    "UseCase",
    "worked_example",
]


@dataclass(frozen=True)
class UseCase:
    """One description of the gate set (``expect``: a plan, or a refusal)."""

    case_id: str
    setting: str
    title: str
    text: str
    expect: Literal["plan", "refusal"] = "plan"


#: The v1 heroes exactly as recorded (their plans are frozen by ``configs/usecase_v1.yaml``).
V1_HERO_CASES: Final[tuple[UseCase, ...]] = (
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
        "Pipe prefabrication: a threaded steel coupling (1 inch pipe size, about 40 mm across and 45 mm "
        "long) sits loosely on the threaded stub of a pipe spool that stands upright. Push it onto the "
        "thread start and turn it on until it is hand-tight (about 3 N m), never over 8 N m.",
    ),
    UseCase(
        "h3_cabinet",
        "maintenance",
        "Equipment cabinet",
        "Field maintenance on an equipment cabinet: turn the quarter-turn latch to unlock the door, "
        "swing the door open, turn the valve handwheel inside a quarter turn (90 degrees) to open it, then plug "
        "the 8 mm round test connector into the service port.",
    ),
)

#: Studio v2's hero jobs (module docstring), in the order the product shows them.
HERO_CASES: Final[tuple[UseCase, ...]] = (
    UseCase(
        "h1_tote",
        "warehouse",
        "Returns tote",
        "Warehouse returns: a plastic tote (300 x 200 mm, 120 mm tall) stands on the packing bench with its "
        "hinged lid shut; the lid overhangs the tote by a 15 mm lip all round. Open the lid, put a small rigid "
        "carton (about 120 x 80 x 60 mm, 300 g) and a flat shrink-wrapped pack (about 150 x 100 x 20 mm, 150 g) "
        "from the bench into the tote, then close the lid.",
    ),
    UseCase(
        "h4_cnc",
        "manufacturing",
        "CNC machine tending",
        "CNC machine tending: the machine on the bench has a guard door that slides sideways over its loading "
        "opening (the opening is 250 mm wide; the machine is 300 mm deep and 250 mm tall; the door slides 300 mm, "
        "takes about 10 N to move and has a fin handle on top). Slide the door open, take the finished part (a "
        "50 x 50 x 30 mm aluminium block, 200 g) off the machine bed beside the fixture and drop it in the out "
        "bin next to the machine, put the next blank (the same size and weight), which waits on the bench in "
        "front of the machine, into the fixture tray inside the machine (120 x 100 mm, 40 mm tall), then slide "
        "the door fully shut.",
    ),
    UseCase(
        "h5_solar",
        "construction",
        "Solar panel install",
        "Solar panel installation: a solar module (1000 x 600 mm, 35 mm thick, 8 kg) waits on its stand beside "
        "the two mounting rails. Lift it onto the rails and set it down against the rail end stops (within "
        "5 mm), then start the clamp nut (13 mm hex) at the middle of its long edge on its thread and tighten "
        "it to 6 N m.",
    ),
    UseCase(
        "h6_tile",
        "construction",
        "Floor tile setting",
        "Floor tiling: a 300 x 300 mm porcelain floor tile (10 mm thick, 2.2 kg) goes on the fresh adhesive "
        "bed, where it sits about 2 mm proud of its finished height until it is pressed. Lay it on the bed, "
        "then tamp it down level: within 1 mm of its finished height and no more than 0.5 degrees out of "
        "level, pressing no harder than 60 N.",
    ),
    UseCase(
        "h2_coupling",
        "construction",
        "Pipe coupling",
        "Pipe prefabrication: a threaded steel coupling (1 inch pipe size, about 40 mm across and 45 mm "
        "long) sits loosely on the threaded stub of a pipe spool that stands upright. Push it onto the "
        "thread start and turn it on until it is hand-tight (about 3 N m), never over 8 N m.",
    ),
    UseCase(
        "h7_hotel",
        "hospitality",
        "Hotel delivery",
        "Hotel delivery, on a bench mock-up: press the elevator call button on the wall (a 40 mm round button, "
        "about 5 N to press, its centre 150 mm up the wall, nothing standing in front of it), then open the room "
        "door beside it (a hinged door 300 mm wide and 400 mm tall with a pull handle, hinged on its left) and put "
        "the parcel (a 160 x 120 x 80 mm box, 500 g) from the bench into the delivery tray behind the door.",
    ),
)

#: The v2 gate set: the six heroes (each recorded live once; a failed one stays recorded as failed).
GATE_CASES: Final[tuple[UseCase, ...]] = HERO_CASES

#: The v1 gate set exactly as recorded (``recorded/gate-2026-10-18-s4e``).
V1_GATE_CASES: Final[tuple[UseCase, ...]] = (
    *V1_HERO_CASES,
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
        "On a framing jig, an M10 hex nut sits loose on top of an upright threaded rod. Run it down the "
        "rod three full turns, keeping the torque under 3 N m so the thread is not damaged.",
    ),
    UseCase(
        "e3_isolation_valve",
        "maintenance",
        "Isolation valve",
        "Plant maintenance round: open the hatch of the meter box, open the isolation valve that stands "
        "beside the box by turning its handwheel one full turn counter-clockwise against about 4 N m, "
        "then close the hatch.",
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
        "tall, 380 g) from the bench beside the fridge and put it in the basket that stands behind the "
        "door, then close the door.",
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
            ObjectPose(60.0, -140.0, 0.0),
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
