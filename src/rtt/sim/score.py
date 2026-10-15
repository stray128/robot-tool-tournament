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
4. Grasped but the object never reached ``success_lift`` -> ``slip_during_lift``.
5. Reached ``success_lift`` but never held it for ``success_hold`` -> ``unstable_hold``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rtt.sim.checks import FLAG_GRIPPER_ENV, CheckEvent
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
            if self.success_at is None and held >= self.cfg.success_hold and t <= self.cfg.max_time:
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
