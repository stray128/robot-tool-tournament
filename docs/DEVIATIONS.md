# Deviations from the pre-registration

Every change to the protocol after the `prereg-v1` tag, and every rerun, is recorded here
with its date, what changed, why, and which results it can affect. Entries are added
before the affected result is looked at whenever that is possible. Results are never
dropped; a run that cannot complete is reported as failed.

Only infrastructure failures are rerun, with the same seed and the failed job's ID logged.

`docs/PREREG.md` holds two protocols: Part A (v2, part-shaped fingers) and Part B
(fallback F1, the v1 protocol). Exactly one runs. The go/no-go decision is recorded below
before the tag.

## Go/no-go decision (before the tag)

- **Decision:** [Mon Oct 19, 22:00 IST: Part A (v2) or F1 (Part B)]
- **Gates:** GV0 [ ], GV1 [ ], GV2 [ ], GV3 [ ], GV4 [ ]; GV5 by Tue Oct 20, 22:00 [ ]
- **Decided by, and when:**

## Declared at registration

### Changes from the build plan, adopted before any v2 scored run (2026-10-16)

The v2 protocol (PREREG Part A) differs from the build plan's visual-upgrade addendum and
from the v1 protocol as follows. All were decided before any v2 design was simulated on
the scored suite.

| Change | From | To | Why |
|---|---|---|---|
| Primary success | Lift 100 mm and hold 2 s | Registered grasp: lift and hold, plus seated and centred 0.25 s after closing (`docs/TASK_V2.md`). Lift-only success is reported for every design and endpoint. | Under lift-only scoring a jammed imprint passes. |
| Finger collision | Convex decomposition (CoACD) of every design, as the addendum said | Exact convex pieces built by the generator. CoACD only for scanned objects and the fixed stock blade. | At the repository's default threshold CoACD removes a 0.5 mm clearance; at a fine threshold it costs tens of seconds per finger. |
| Training suite | Primitive objects (`train_v1`) | 6 procedural industrial parts. Scanned objects are held-out only. | Exact, unbranded, and regenerable at new sizes for the held-out suite. |
| Round 0 | The 4 v1 entrants plus 4 Sobol designs | Stock blade, flat pad, configurator imprint and swept imprint, plus 4 Sobol designs. The wrap and hook entrants become reference entrants, scored outside the budget. | The imprint is the core comparator. |
| Grip height | 3 controller settings (k = 0.3, 0.5, 0.7), best one scored | A design parameter, with one controller setting | A part-relative pad has a designed height. |
| tau and R | The v1 rules (Part B) | Written rules for v2 (PREREG A1.1 and A7), fixed before the v2 pilot | So neither can be tuned after the fact. |
| Fallback | None | F1: if the v2 gates miss Oct 19, 22:00 (or GV5 misses Oct 20, 22:00), Part B runs and v2 is product mode and visuals only | Protects the Oct 21 registration. |
| Product-only options | None | The locator pin, body styles and offline renders never enter the scored space. Clips that show them are labelled "product mode". | Keeps the scored experiment small. |
| Stock entrant | Box approximation of the open-source gripper's finger | The real blade mesh, as convex hulls. Fallback: the box approximation. | What viewers see is what is simulated. |

### Gate G3 near-miss (Part B only)

`task_v1` was frozen although calibration gate G3 failed on two criteria by narrow
margins (the concave entrant at 75 % against a 70 % limit; 4 of 64 random designs above
the best entrant against 5 required). See `docs/TASK.md` and `docs/PREREG.md` section B9.
This applies only if F1 runs.

### Gate GV3 calibration iterations (Part A only)

At most 3 iterations, suite-level knobs only, and tolerances changed in at most one of
them (`docs/PREREG.md` section A9). One row per iteration, copied from
`calibration_log.json`:

| Iteration | Date | Changes | Checks failing | Decision |
|---|---|---|---|---|
| 1 | | (the task as drafted) | | |

Declared misses, if any: [at freeze].

### Registration tolerances (Part A only)

The tolerances in `docs/TASK_V2.md` may change once, during GV3, on the screening sample.
If they change, record the old and new values, the iteration and the reason here.
[at freeze]

## After the tag

None yet.

<!--
Template:

### YYYY-MM-DD: short title

- **What changed:**
- **Why:**
- **Affects:** (endpoints, runs, job IDs)
- **Known before or after seeing the affected result:**

Entries required under Part A: a run with more than 1 % physics_invalid episodes; a job
that waited for uk-south2 capacity under --no-fallback; any rerun after an infrastructure
failure.
-->
