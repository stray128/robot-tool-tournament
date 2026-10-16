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

### Implementation choices in the v2 build, adopted before any v2 scored run (2026-10-16)

Where the code that implements Part A differs from the visual-upgrade spec. Each was decided
by the track that owns the code, before any v2 design was simulated on the scored suite.

| Area | Spec | Implemented | Why |
|---|---|---|---|
| Part outline (generator) | Sampled at 0.1 mm in x and 25 rows in z | Exact for convex pieces, and the clearance is kept in height as well as sideways | Sampled rows let a 0.4 mm imprint come 0.25 mm from a socket rim |
| Build time (validator) | At most 1 s per pair | Recorded per build, slow builds listed, never a rejection | Validity must not depend on the machine |
| Extra schema rules | None | A hook (height plus protrusion) must fit in the pad; a chamfer must leave 0.5 mm of face | Otherwise the tip geometry is undefined |
| Printed root block | At the carriage face | 50 um behind it, in the print only | Keeps the print a single solid; collision unchanged |
| Generator version | | `rtt-freeform-gen/2`: a face that only grazes a pad band's boundary plane no longer counts as the seat | The earlier version seated some jaws up to 8 mm early; every design ID changed with it |
| Inactive parameters | Ignored and logged | Their bounds apply only when active (the tool schema still shows them) | Spec section 12 |
| Pipe tee | Prototype dimensions | Socket depth 16.5 mm (17), solid core without a flow bore: 39.9 g against about 30 g | The outline the jaws see is identical |
| `heldout_real_v2` | Only the decomposition changes | Also `max_parts` 16 to 32, and 8 more names in `exclude_names` | The finer threshold needs more pieces; the added names are branded or too wide |
| Penetration check | | Sampled every 10 ms (the check period), not every physics step | Cost |
| Height error, held-out | +-1.5 mm | The 3 mm tip floor stops a +1.5 mm error from lowering the jaws further | Keeps the jaws off the table |
| MuJoCo Warp | Cut for v2 | The Warp executor has no v2 path; a v2 job sent there fails loudly | Never scored silently |

New files beyond the spec's track table: `src/rtt/parts/{pieces,suite,preview}.py`,
`src/rtt/sim/gates_v2.py`, `src/rtt/analysis/{gv3_run,gvh_run}.py`, `scripts/v2_bout.py`.

### Open before the go/no-go (integration, 2026-10-16)

Recorded so the decision is visible; each is settled, or the miss declared, before the tag:

- The L-bracket's training pose cannot be gripped by the entrants (every one fails the
  seat-closure rule on it), and the hex standoff's internal seam carries the part at low
  friction. Both are parts-library fixes, re-measured on GV0 to GV2.
- `native_ccd`: the rule of PREREG Part A keeps it off; the GV2 evidence favours on.
- GV2 criteria missed with either collider: penetration in held episodes, 1 ms vs 2 ms
  parity, and CPU cost per episode.
- The stock-blade hulls bulge 0.69 and 0.75 mm over the whole inner face (0.0003 mm on the
  contact pad) against the 0.2 mm limit: decision D9 or its box fallback.

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
