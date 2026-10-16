"""Episode scoring: lift tracking, success criterion and failure-mode labels.

Success: the object's centre rises at least ``success_lift`` (0.10 m) above
its settled rest height and stays there for a continuous ``success_hold``
(2.0 s) that completes before ``max_time`` (12 s), with no physics-check flag
raised before that moment.

Every failure gets exactly one label, decided in this order (only flags raised
before a confirmed success count):

1. A physics-check flag raised before LIFT (see checks.py): gripper-table/floor
   force or penetration -> ``finger_env_collision``; otherwise by phase:
   SETTLE/DESCEND -> ``missed_grasp``, CLOSE -> ``ejected_on_close``.
2. No grasp at the start of LIFT (either finger presses on the object with less
   than ``grasp_min_squeeze_frac * F / 2`` along the closing axis, or pushes it
   down harder than ``grasp_max_press_down_ratio`` times that squeeze, i.e. the
   finger rests on top of the object instead of pinching it) -> ``missed_grasp``. A flag raised later (for example the fingers snapping shut
   once they clear the top of an object they were resting on) does not hide
   the missed grasp.
3. A flag raised during LIFT -> ``slip_during_lift``, during HOLD ->
   ``unstable_hold`` (gripper/environment contacts -> ``finger_env_collision``).
   This includes ``palm_contact`` (the palm pressing on the carried object): the
   object moved up out of the finger grasp into the palm, so a palm-supported lift
   never counts as a success; the flag itself stays in the episode's flags.
4. Grasped but the object never reached ``success_lift`` -> ``slip_during_lift``.
5. Reached ``success_lift`` but never held it for ``success_hold`` -> ``unstable_hold``.

Task v2 (:func:`classify_v2`, spec section 7) scores a *registered grasp*: the v1 lift-and-hold
criterion above (reported as ``lift_success``) and the registration rule of
:mod:`rtt.sim.register`. Three failure modes are added (:data:`FAILURE_MODES_V2`), and the
order of precedence is: the physics flags, ``physics_invalid`` (a MuJoCo state reset in the
scored part of the episode) then ``physics_unreliable`` (a held episode whose jaw-part
penetration went above ``checks.held_penetration_max_mm``, 1 mm; founder ruling 4 of
2026-10-16); then the v1 labels (missed, ejected, slip, unstable); then ``not_registered``
(lifted and held, but not seated, centred or upright). Both physics flags fail the registered
and the lift-only criterion alike. ``invalid_build`` marks a design whose jaws could not be
built for a part (it scores as a failure there, without a simulation).

The scored part of an episode runs up to the check tick at which the outcome is decided; the
checks never run in the replay-only tail, so every flag and maximum the scorer sees, and the
``bad_qacc`` count in the record, belong to the scored part. One rule therefore holds in the
simulator, the search harness and the analysis: the reset counter rose -> ``physics_invalid``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rtt.sim.checks import FLAG_BAD_QACC, FLAG_GRIPPER_ENV, CheckEvent
from rtt.sim.controller import Phase
from rtt.sim.task import ScoreConfig

FailureMode = Literal[
    "missed_grasp", "slip_during_lift", "ejected_on_close", "unstable_hold", "finger_env_collision"
]
FAILURE_MODES: tuple[FailureMode, ...] = (
    "missed_grasp",
    "slip_during_lift",
    "ejected_on_close",
    "unstable_hold",
    "finger_env_collision",
)

FailureModeV2 = Literal[
    "missed_grasp",
    "slip_during_lift",
    "ejected_on_close",
    "unstable_hold",
    "finger_env_collision",
    "not_registered",
    "physics_invalid",
    "physics_unreliable",
    "invalid_build",
]
FAILURE_MODES_V2: tuple[FailureModeV2, ...] = (
    *FAILURE_MODES,
    "not_registered",
    "physics_invalid",
    "physics_unreliable",
    "invalid_build",
)

_PHASE_MODE: dict[Phase, FailureMode] = {
    Phase.SETTLE: "missed_grasp",
    Phase.DESCEND: "missed_grasp",
    Phase.CLOSE: "ejected_on_close",
    Phase.LIFT: "slip_during_lift",
    Phase.HOLD: "unstable_hold",
}


@dataclass
class LiftTracker:
    """Tracks object lift and the longest continuous time above the success height."""

    cfg: ScoreConfig
    z0: float | None = None
    lift: float = 0.0
    max_lift: float = 0.0
    hold_time: float = 0.0  # longest continuous stretch above the success height
    success_at: float | None = None
    # Success is only confirmed at or after this time (a perturbation test sets it to the
    # end of the perturbation plus a settling hold).
    min_success_time: float = 0.0
    _above_since: float | None = None

    def set_reference(self, z0: float) -> None:
        self.z0 = z0

    def update(self, t: float, z: float) -> None:
        if self.z0 is None:
            return
        self.lift = z - self.z0
        self.max_lift = max(self.max_lift, self.lift)
        if self.lift >= self.cfg.success_lift:
            if self._above_since is None:
                self._above_since = t
            held = t - self._above_since
            self.hold_time = max(self.hold_time, held)
            if (
                self.success_at is None
                and held >= self.cfg.success_hold
                and t >= self.min_success_time
                and t <= self.cfg.max_time
            ):
                self.success_at = t
        else:
            self._above_since = None

    @property
    def above(self) -> bool:
        return self._above_since is not None


def mode_for_event(event: CheckEvent) -> FailureMode:
    """Failure mode implied by a physics-check flag."""
    if event.flag == FLAG_GRIPPER_ENV or event.gripper_env:
        return "finger_env_collision"
    return _PHASE_MODE[event.phase]


def classify(
    events: Sequence[CheckEvent], tracker: LiftTracker, grasped: bool
) -> tuple[bool, FailureMode | None]:
    """Return ``(success, failure_mode)`` for a finished episode."""
    cutoff = tracker.success_at if tracker.success_at is not None else math.inf
    flagged = sorted((e for e in events if e.t <= cutoff), key=lambda e: e.t)
    if flagged and flagged[0].phase < Phase.LIFT:
        return False, mode_for_event(flagged[0])
    if not flagged and tracker.success_at is not None:
        return True, None
    if not grasped:
        return False, "missed_grasp"
    if flagged:
        return False, mode_for_event(flagged[0])
    if tracker.max_lift < tracker.cfg.success_lift:
        return False, "slip_during_lift"
    return False, "unstable_hold"


def classify_v2(
    events: Sequence[CheckEvent],
    tracker: LiftTracker,
    grasped: bool,
    registered: bool,
    *,
    penetration_mm: float = 0.0,
    held_penetration_max_mm: float = math.inf,
) -> tuple[bool, bool, FailureModeV2 | None]:
    """Return ``(success_v2, lift_success, failure_mode)`` for a finished v2 episode.

    ``lift_success`` is the v1 criterion (:func:`classify`, ignoring the v2-only
    ``bad_qacc`` flag); ``success_v2`` additionally needs ``registered``.

    * Any MuJoCo state reset among ``events`` makes the whole episode ``physics_invalid``:
      after a reset the trajectory is not physics, so neither success counts. ``events`` hold
      the scored part of the episode only (the checks stop when the outcome is decided), which
      is the same window the record's ``bad_qacc`` count covers.
    * A held episode (the v1 criterion met) whose deepest jaw-part penetration
      ``penetration_mm`` is above ``held_penetration_max_mm`` is ``physics_unreliable``: the
      grasp leans on interpenetrating geometry, so neither success counts (ruling 4).
    """
    if any(e.flag == FLAG_BAD_QACC for e in events):
        return False, False, "physics_invalid"
    lift_success, mode = classify([e for e in events if e.flag != FLAG_BAD_QACC], tracker, grasped)
    if not lift_success:
        return False, False, mode
    if not penetration_mm <= held_penetration_max_mm:  # NaN penetration is unreliable too
        return False, False, "physics_unreliable"
    if not registered:
        return False, True, "not_registered"
    return True, True, None
