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

- **Decision:** Part A (v2), with the stricter rules declared below; F1 is not taken. GV4
  (Nebius parity) and GV5 (harness) passed on 2026-10-16, before their deadlines, so the
  decision stands and Part B is marked "not run" in `docs/PREREG.md`.
- **Gates:** GV0 [x], GV1 [x], GV2 [x] (cost miss declared), GV3 [x] (spread miss declared,
  frozen), GV-H [x] (0 heroes: offset-contrast fallback), GV4 [x] (2026-10-16: 384 of 384
  episodes agree on outcome, every geometry and hull hash equal), GV5 [x] (2026-10-16:
  harness tests, diversity bars on the training parts, probe 100 % valid after reserves,
  one local B = 40 Nemotron run). Numbers in `docs/BENCH_V2.md`.
- **Decided by, and when:** the founder, 2026-10-16, after round 4 integration (rulings
  below); gates re-run and GV3 frozen the same night.

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
| MuJoCo resets | "the counter went up -> `physics_invalid`" | One rule in the scorer, the search harness and the analysis: a reset in the scored part of the episode (up to the decision) makes it `physics_invalid`; the replay-only tail is never checked, so it can change neither the label nor the record's `bad_qacc` count | The scorer used to ignore resets after a confirmed success while the harness and analysis counted every one; the two can only differ in the tail, which the spec says never changes an outcome |
| Hero rule | `physics_invalid` episodes count against qualification | `physics_unreliable` episodes too (digest draft `8bbdb111...` -> `dda0ec35...`, changed before calibration) | A failure caused by the penetration rule is not a failure a viewer sees; a failure is never staged |
| Stock entrant | Validators apply to recipes | The stock blade must also seat inside the 2-40 mm closure range (`q_seat_range`) | One validity rule for every entrant (integration ruling, track G) |
| GV2 low-friction case | Must slip or drop | Any failure mode | An ejection or a physics flag at mu 0.05 also means the part was not carried (integration ruling, track S) |
| Diversity bars (rule D6) | t from 10k random valid Sobol pairs | Applied with the full section-13 validators on the six patched training parts (`configs/diversity_v2.json`, `python -m rtt.search.threshold_v2`): t 0.070, explore bar 0.15; the draft constants 0.080 / 0.17 came from the schema rules only | The rule asks for valid designs, and validity includes the geometry stages (gate GV5, 2026-10-16, before any pilot) |
| GV5 probe | 5 calls of Super with the v2 tool | Each call is arm C's real round-1 prompt (training parts, v2 entrants, round 0 scored by the simulator) and the slots pass the full validators (`probe_v2 --real`) | Track P's probe used synthetic parts in the prompt and schema checks only: 100 % valid, 35 % under the geometry validators |
| GV4 episode set | 100 v2 episodes | 384: the six v2 entrants and two Sobol recipes (seed 20261024) on all 48 `train_v2` episodes, jaws built in the job (`rtt.sim.gv4`) | More than the minimum, and every entrant geometry is covered |
| tau rule input | Scores on the 1/48 grid | The GV3 report rounds scores to 6 decimals; `load_calibration_v2` snaps them back to their grid points (within 5e-7) | Without it the tau rule refused the frozen GV3 report (merge fallout, fixed 2026-10-17 before tau was computed) |

New files beyond the spec's track table: `src/rtt/parts/{pieces,suite,preview}.py`,
`src/rtt/sim/gates_v2.py`, `src/rtt/analysis/{gv3_run,gvh_run}.py`, `scripts/v2_bout.py`.

### Founder rulings after round 4 integration (2026-10-16), adopted before any v2 scored run

Spec Addendum D. Each was decided before the official GV3 calibration and before any search
or pilot run; GV0 to GV3 and GV-H were re-run under them the same night.

| # | Ruling | From | To | Why | Affects |
|---|---|---|---|---|---|
| 1 | Parts fix | L-bracket lying on one leg with the other at +y; hex standoff a solid base under hole-height ring sectors | L-bracket lying flat, gripped on the standing leg's edges, its frame centred on that leg; hex standoff as 12 full-height sectors plus a plug under the hole (track O's patch) | Every entrant failed the seat-closure rule on the old L-bracket; the hex's horizontal internal seam gave the pads a ledge that carried the part at mu 0.05 | `parts_v2`, `train_v2`, GV0-GV3 |
| 2 | Native collision on | `native_ccd: false` by the rule of spec section 6 (parity on/off at least 99 % on the v1 set and on 200 v2 episodes) | `native_ccd: true` | The rule's parities missed (98.5 % v1, 97.2 % v2 at 1 ms), but the v1 misses are the primitive-cylinder lip-edge regression that v2 meshes never meet, and the v2 misses were mostly libccd at fault (2-3 mm mesh penetrations at 2 ms) | Every v2 episode |
| 3 | 1 ms step | 2 ms, with a 1 ms vs 2 ms parity criterion (99 %) in GV2 | 1 ms for every v2 episode; the parity criterion is replaced (the 2 ms re-run is a diagnostic: 94.4 % agreement) | Outcomes at 2 ms changed with the step on mesh contacts (88-96 % parity) | Every v2 episode; compute (below) |
| 3a | Cost accepted | GV2: a v2 episode at most 1.5x the v1 median CPU time | Declared miss: 3.57x measured (0.791 s against 0.221 s per episode) | The founder accepts the cost of the 1 ms step | Compute: about 21-29 CPU-h for the pilot, both arms at R = 10 and the held-out suites [derived], against the spec's 8 |
| 4 | Penetration rule | GV2 criterion: penetration at most 1 mm in held episodes | Scoring rule: a held episode above 1 mm jaw-part penetration is `physics_unreliable`, a failure on both the registered and the lift-only criterion, for every arm and design | A grasp that leans on interpenetrating geometry is not evidence about the fingers | Every v2 score; 1.70 % of the GV3 screening episodes (57 of 3,360; 7 of them the stock blade on the bushing) |
| 5 | D9 on the contact zone | Stock-blade inner-face bulge at most 0.2 mm, else the box fallback | Measured on the pad contact zone only: 0.0003 mm, pass | The 0.69/0.75 mm whole-face bulge lies on recessed arms at least 2.5 mm behind the contact plane | The stock entrant (real blade kept) |
| 6 | GV3 run officially; spread miss declarable | Only the headroom miss could be declared | A miss on the Sobol SD target is declared in PREREG, as G3 was for v1 | | GV3 decision (below) |

Integration rulings requested with them (same date): the stock path applies the
seat-closure range check (track G); the GV2 low-friction case accepts any failure mode
(track S); one reset rule across the scorer, the harness and the analysis (tracks S, P, Q;
the implementation-choices table above).

**Settled from the integration list:** the L-bracket and hex standoff (ruling 1, re-measured:
every entrant builds on all six parts, the hex now slips at mu 0.05); `native_ccd` (ruling 2);
the GV2 misses (penetration by ruling 4, timestep parity by ruling 3, cost declared under
ruling 3); the stock-blade bulge (ruling 5).

**Noted, not a deviation:** on the flanged bushing the stock blade's held episodes are mostly
`physics_unreliable` (7 of 8 in GV3, 16 of 21 in the hero screen; 1.3-1.9 mm penetration,
and the bushing tilts in the blade, up to 20 deg in the hero screen). The rule applies to the stock entrant as to every
design; without it 2 more stock episodes would register (25.0 % -> 29.2 %).

### Gate G3 near-miss (Part B only)

`task_v1` was frozen although calibration gate G3 failed on two criteria by narrow
margins (the concave entrant at 75 % against a 70 % limit; 4 of 64 random designs above
the best entrant against 5 required). See `docs/TASK.md` and `docs/PREREG.md` section B9.
This applies only if F1 runs.

### Gate GV3 calibration iterations (Part A only)

At most 3 iterations, suite-level knobs only, and tolerances changed in at most one of
them (`docs/PREREG.md` section A9). One row per iteration, copied from
`calibration_log.json`:

| Iteration | Date (UTC) | Changes | Checks failing | Decision |
|---|---|---|---|---|
| 1 | 2026-10-16 17:07 | (the task as drafted) | sobol_sd (SD 0.113) | iterate |
| 2 | 2026-10-16 17:20 | `part:block.x_mm` 30.0 -> 20.0 | sobol_sd (0.109), headroom (3 of 64) | iterate |
| 3 | 2026-10-16 17:28 | `part:block.x_mm` 20.0 -> 30.0 | sobol_sd (0.113) | declare the miss and freeze |

Iteration 2 tried the one part that separated no designs (every Sobol design scored 3 of 8
on the 30 mm block; no finger in the space reaches its sides); the narrower block lowered the
spread and the headroom, so iteration 3 restored the drafted suite and reproduced iteration 1
record for record. Re-scoring iteration 1's records under other tolerances moved the SD only
between 0.113 and 0.126, so the tolerances were not used.

Declared misses: **GV3 Sobol spread**, SD 0.113 against the 0.15 target (founder ruling 6);
the other GV3 targets are met (entrants 25.0-58.3 %, 5 of 64 above the best entrant, none
at 100 %). **GV2 cost**, 3.57x the v1 median against 1.5x (ruling 3). **`native_ccd`** on
against the spec's rule (ruling 2). Not a miss: GV-H found 0 heroes, so the video hook uses
the imprint-vs-designed offset contrast only, as the hero rule prescribes.

### Registration tolerances (Part A only)

The tolerances in `docs/TASK_V2.md` may change once, during GV3, on the screening sample.
If they change, record the old and new values, the iteration and the reason here.
Unchanged at the freeze: seat 1.0 mm, dx and dy 0.75 mm, yaw 2 deg, tilt 3 deg, tilt after
the hold 5 deg.

### Pilot (before the tag)

- **2026-10-17, 00:00 IST: the pilot ran on the laptop, not on Nebius (provisional).** The
  Nebius CLI's browser session expired at about 23:53 IST on Oct 16, so the image of the
  merge commit could not be pushed and no job could be submitted. The three random pilots
  (seeds 101-103) and the two Nemotron pilots (101-102) ran locally on the merged code
  (`main` `24234cb`, then `f80e58f`, which only adds site data) with the same commands as
  the jobs (`rtt search --space freeform-v2`). tau and the R rule's q in PREREG A1.1 and A7
  are therefore provisional: the protocol's pilot (A12) is the same seeds on `uks2-cpu` from
  the image of the tagged commit, and its tau and q are the ones frozen. Gate GV4 found the
  laptop and `cpu-d3` identical in outcome on 384 of 384 v2 episodes, so the values are
  expected to carry over. Known before any confirmatory run; no confirmatory seed was used.
- **2026-10-17, 00:50-01:05 IST: the pilot on the executor of record.** With the CLI session
  renewed, the image of `main` `53e9f57` was built (`make image`, `git_dirty: false`) and
  pushed to both registries (`rtt-sim@sha256:c6162fa01d41beaab64e1431bc10e07535ce7bd59045fcef7bf0e074857b4b89`),
  and the protocol's pilot ran on `uks2-cpu` (`cpu-d3`, `32vcpu-128gb`): random seeds
  101-103 (job `aijob-e05zsdh89e1dj65gc2`) and Nemotron seeds 101-102 (job
  `aijob-e05eecwevax4wcdjn4`). The random arm reproduced the laptop pilot exactly (all 120
  design scores, each run's fingerprint and frozen top-1 hash), so tau (37/48) and q (6/172)
  are unchanged, and the values frozen in PREREG A1 and A7 are the Nebius ones. The Nemotron
  runs differ from the laptop's, as expected without a model seed; they are descriptive only
  (PREREG A12). The laptop pilot is superseded.
- **Held-out procedural secret: kept (founder, 2026-10-17).** PREREG A6 (draft) said a new
  secret would be drawn for `heldout_procedural_v2`. The code draws the suite from the
  committed secret (`configs/suites/heldout_v1.sha256`, `073055ff...`), domain-separated by
  `rtt-heldout-procedural-v2`, so the v1 and v2 suites are uncorrelated. The secret has not
  been used by any run we can find (the real-object development selections use the public
  dev seed 20261016). The founder approved keeping it; no new secret is drawn. The
  commitment is the SHA-256 of the secret string without its trailing newline
  (`rtt.store.vault.seed_sha256`).

### Founder approvals (2026-10-17), before the tag and before any confirmatory run

| Approval | From | To | Why | Affects |
|---|---|---|---|---|
| R | The R rule's result, R = 12 (candidates {8, 10, 12}, fixed before the pilot) | R = 32 per arm, seeds 1001-1032 (all of `configs/seeds_main.yaml`) | No candidate reaches 0.8 power at 2x (0.40 at R = 12). R = 32 gives 0.75 at 2x and a minimum detectable speed-up of 2.25x against 3.25x, for about $1.85-2.90 of Nebius and $3.40-3.90 of Token Factory (PREREG A4.1) | S1 and every secondary endpoint; the job size. Decided after the laptop pilot of both arms had been reported in the PREREG draft; it uses only the random arm's q and the cost |
| tau source | The laptop pilot (provisional) | The Nebius pilot, the executor of record | The protocol says the pilot runs on `uks2-cpu` | tau and q (equal in value) |
| Held-out secret | A new secret (PREREG draft) | The committed secret, domain-separated (above) | Unused; uncorrelated with v1 by the domain | H1 |
| H2 width | Undecided (options in `docs/HELDOUT_REAL.md`) | Option A: the 79 mm limit kept; H2 tested on the whole draw; success also reported at most 50 mm and above 50 mm as a planned split, outside the Holm family | Keeps the 64-object draw; reports the size effect openly | H2 |
| Release | A private repository | Public at the `prereg-v1` tag | Anyone can check the hashes against the files | The pre-publication checks below |

### Image of record built from `53e9f57`, not from the tagged commit (2026-10-17)

- **What changed:** PREREG A4 (draft) said the image is built from the tagged commit. The
  image of record is the pilot's, built from `53e9f57`. The tagged commit adds only
  documentation to it (`docs/PREREG.md`, `docs/DEVIATIONS.md`, `README.md`), so the image is
  not rebuilt, and the confirmatory manifests will record `git_sha` `53e9f57`.
- **Why:** the pilot ran this image, and a rebuild would differ only in the documents it
  carries.
- **Check:** `git diff --stat 53e9f57 prereg-v1 -- . ':!docs' ':!README.md'` prints nothing.
- **Affects:** provenance of every confirmatory job. Known before any confirmatory run.

### Open at the tag: the v2 held-out runners (found 2026-10-17)

- **What:** the image of record has no command that scores a frozen v2 recipe on
  `heldout_procedural_v2` or on the `heldout_real_v2` draw. `rtt heldout` scores v1
  `FingerDesign`s at a controller setting on the v1 suite, `rtt heldout-real run` scores under
  `task_v1` with v1 width bands, the job entry point has no held-out kind, and nothing outside
  the tests calls `rtt.parts.heldout.heldout_procedural_suite`.
- **Affects:** H1 and H2 (PREREG A6). S1 and S2 are computed from the search outputs and are
  not affected.
- **Status:** open, blocks the tag (PREREG freeze checklist item 10). The founder chooses
  between building the runners before the tag (a new image, A4 and A5 re-recorded) and
  declaring them here as written after the tag, to the six rules in PREREG A6, with their
  code hash and image digest logged before the held-out jobs.

**Founder decision (2026-10-17), before the tag:** tag now, and build the v2 held-out runners after the tag. They follow the six rules in PREREG A6. Their code is committed and its hash, and the image digest that contains them, are logged here before any held-out job runs. No held-out job runs on code written after any held-out outcome has been seen. The confirmatory training-suite runs (S1) need no new code and start right after the tag.

### Pre-publication checks (2026-10-17), before the repository is made public

- `scripts/leak_check.sh` (26 private patterns from outside the repository): clean in
  work-tree, `--staged` and `--history` mode (158 commits, every branch and tag) and in path
  mode on `docs`, `web/public`, `data`, `configs`, `src`, `scripts`, `tests`, `README.md` and
  `NOTICE`. On the git-ignored pilot outputs under `runs/` it reports 4 hits of one short
  numeric pattern, all inside the digits of floating-point values (false positives); those
  outputs are not in the repository, and a release of them needs a fresh check.
- A regex sweep of all 1,006 blobs reachable from any ref, and of the work tree, for key
  shapes (the Token Factory key prefix, AWS access-key IDs, PEM private-key headers,
  OpenAI-style, GitHub, Slack and Google tokens, JWTs, bearer tokens, secret-looking
  assignments, credentials or tokens in URLs),
  private paths and names, AWS account, instance and IP forms, and the value of every ID in
  the Nebius settings file: no secret and no private content. The only hits are made-up keys
  and IDs in tests, the public default names of the Token Factory secret and the S3
  profiles, and Nebius job IDs in site data (approved). Commits are authored under a GitHub
  noreply address; commit messages are clean.
- `.gitignore` covers `.env`, `.env.*`, `secrets/` and `runs/`; none of them is tracked.
  `LICENSE` is Apache-2.0 at the top level; `NOTICE` and the README credit Google Scanned
  Objects (CC BY 4.0), the SO-ARM-102 meshes (CC BY 4.0, vendored with their licence) and the
  software dependencies. No file in the tree or in history is over 5 MB (the largest is
  0.30 MB).

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
