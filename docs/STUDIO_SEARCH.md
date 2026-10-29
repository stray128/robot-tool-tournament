# Studio search, full-task runner, recommender and build pack (track S5)

**Product mode. Not part of the pre-registered experiment.** Every record, page and file with
text carries "Product mode: not part of the pre-registered experiment". The site never renders
this file.

S5 turns a `TaskSpec` (track S4) into a ranked set of tested **setups** (one effector, a two-tool
head, or a tool-change sequence) and the files to build the recommended one. It runs on the
frozen Studio contract (`rtt-effectors/2`, [STUDIO.md](STUDIO.md)) and S2's primitives
([STUDIO_PRIMITIVES.md](STUDIO_PRIMITIVES.md)), and changes neither.

| What | Where |
|---|---|
| Candidates, capability pruning, per-step screen, success matrix, combination search, bracket geometry search, swept-volume check, `run_task_tournament` | `src/rtt/studio/search.py` |
| Full-task runs on the condition grid (on top of `rtt.studio.episode`), worker pool, carried-state runs | `src/rtt/studio/runner.py` |
| Ranking, simplicity rule, best single baseline, sensitivity report | `src/rtt/studio/recommend.py` |
| Build pack: printable parts, MJCF export, documents, manifest, `export_build_pack` | `src/rtt/studio/buildpack/` |
| Tests | `tests/test_studio_search.py`, `tests/test_studio_buildpack.py` |

## Contents

1. [Pipeline](#1-pipeline)
2. [Candidates and capability pruning](#2-candidates-and-capability-pruning)
3. [Per-step screen and the success matrix](#3-per-step-screen-and-the-success-matrix)
4. [Combination search](#4-combination-search)
5. [Finalists and the recommendation](#5-finalists-and-the-recommendation)
6. [Sensitivity to the change times](#6-sensitivity-to-the-change-times)
7. [Build pack](#7-build-pack)
8. [Orchestrator rulings applied](#8-orchestrator-rulings-applied)
9. [Assumed constants](#9-assumed-constants)
10. [Agent tools and command line](#10-agent-tools-and-command-line)
11. [Hero gate](#11-hero-gate)
12. [Open issues](#12-open-issues)

## 1. Pipeline

```
TaskSpec
  -> candidates: family presets + library entrants (when installed) + designed effectors
  -> capability pruning, a reason per (candidate, step)                       search.prune
  -> hold chains; per-chain screen on the first 3 conditions, carried state    search.screen
  -> success matrix (candidates x steps)                                       Screen.matrix
  -> single / tool_change / multi_head setups scored from the matrix,
     bracket geometry search + swept-volume check, payload limit               search.combine
  -> 3 to 5 finalists, always with the best single off-the-shelf effector     search.choose_finalists
  -> full-task runs of the finalists on the whole condition grid               runner.run_full_tasks
  -> assert_fair, ranking with the simplicity rule, best single baseline       recommend.recommend_setups
  -> sensitivity to the tool-change and head-switch times                      recommend.sensitivity
  -> build pack of the recommendation                                          buildpack.export_build_pack
```

`run_task_tournament(task_spec, ...)` runs everything up to the sensitivity report and returns a
`TournamentReport`; `report.to_dict()` is the JSON record (`format: rtt-studio-tournament`).

## 2. Candidates and capability pruning

**Candidates** (`default_candidates`): every preset of every effector family that loads (in
family order), the open library's green entrants when `rtt.library` is installed (track S3;
capability tags from each item), and the designed effectors: the designed family's hand-written
reference recipes plus the Nemotron designs of the recorded morphology run
(`src/rtt/effectors/morphology/recorded/smoke-2026-10-17/manifest.json`, replayed from the
record: no model call, no spend). Callers can pass their own `EffectorSpec`s. Designed effectors
are never "off the shelf".

**Pruning** (`prune`) leaves a candidate out of a step, or out of everything, and records why:

| Rule | Reason shown |
|---|---|
| the family has no Studio build (the frozen parallel-jaw reference runs only inside the v2 episode) | "this family runs only inside the frozen v2 episode ..." |
| an amber licence record | "an amber licence: never competes or ships" |
| the tool alone is heavier than the arm payload limit (ruling 5) | "it weighs X kg; the arm carries at most 3 kg [assumed]" |
| the mechanism cannot serve the step's primitive (table below) | e.g. "a suction cup has no hook or fingers to engage a handle or lip" |
| the tool does not fit an object it touches (ruling 4, every object, not just the first part) | the family's own build refusal, or the cup, opening or feature check |
| a tool that picks nothing in a chain cannot be built on the part the scene gives it (`scene.build_part_for`: the 10 mm probe cube if the family builds on it, else the library block; contracts CR-S5-1 and CR-S5-2) | "built against the block: ..." |
| integrator rule R05: a process step for any tool but the process tool of its kind (below) | 'integrator rule R05: tightening "Clamp nut" to 5 N m is a process step for a nut runner ...' |
| integrator rule R03: a soft arm, a tentacle (the soft-arm family, a designed segmented tentacle) or a dexterous hand (a catalog hand, a designed fingered hand) on a step whose object is a lid, door, drawer or button (`rules.ARTICULATED_TEMPLATES`, by template value) | 'integrator rule R03: "Lid" is a mechanism to open or press ...' (the object's label, quoted as written) |
| integrator rule R09: a soft arm on an insert step, or on a box or cylinder part over 2.5 kg [assumed] | "integrator rule R09: ..." |

`rule_ineligible(cand, step, task_spec)` gives the integrator-rule reason (the step names its
object only, so the task spec is passed). **Readmission (R03, "unless every rigid option
fails"):** when a hold chain has no candidate left after pruning and some were kept off it only by
R03, they come back for that chain, and the screen records it in `screen.rule_notes` (rule, chain,
steps, the readmitted effector ids). R05 and R09 refusals are never readmitted.

**Process steps (R05).** A step is a process step (`rules.process_need`) when it is:

* a `twist` or `insert` on a threaded member (`rules.THREADED_TEMPLATES`: the threaded joint and a
  panel's clamp on its rail, by template value) whose criteria require a torque (`torque_nm`, or a
  clamp's `clamp_torque_nm` once that quantity exists) of at least `PROCESS_TORQUE_NM` = 4 N m
  [assumed]: it needs a **nut runner**;
* a `push` on a tile on its bed (`rules.PRESS_TEMPLATES`: `tile_bed`, by value, covered as soon as
  that template exists): it needs a **tamper pad**.

Only the process tool of the step's kind is eligible; every other tool, the other process tool
included, is refused with the R05 reason. The threshold: hand-tight, what a person reaches with
the fingers, is about 3 N m (the pipe coupling hero H2 asks for at least 3 N m and at most 8 N m);
4 N m sits 1 N m above it, so H2 stays a gripper's job and a 5 N m clamp does not. The rule makes
one exception, for light work at low torque: an ordinary hand tool held in a gripper, marked as a
makeshift setup. The Studio does not model hand tools held in a gripper, so under the threshold
the gripper turns the member directly and no mark is shown. For anyone writing a job: today the
only nut runner is the 13 mm socket, so a tightening to 4 N m or more on anything but a 13 mm
across-flats hex nut at least 12 mm tall has no eligible tool, and the search reports no setup for
that step.

Capabilities by mechanism [assumed]. A library item tagged with a grip (pinch, power, suction,
wrap) adds its tags to its mechanism's; an item without one is a tool (hook, pusher, driver)
and serves only its tags, except `hook_pull`, which is measured, not tagged:

| Mechanism | Families | Serves |
|---|---|---|
| fingers | catalog grippers and hands, designed | grasp, hook_pull, twist, insert, push |
| suction | suction cups and vacuum frames | suction, twist (seals on a flat top face), insert (holds by its seal), push |
| soft_arm | soft arms | wrap, hook_pull (curls around a handle), twist |
| process | the process tools: a nut runner (`drive`), a tamper pad (`press`) | by the tool's primitive: the nut runner twist and insert, the tamper push; no acquire class, no hook |

A process tool does its own process only: "a nut runner turns a fastener; it cannot pick or hold
a part", "a tamper pad presses a face; it is not a hook, so it cannot pull a lip or handle". It
never counts as a hook: the nut runner measures none, and the tamper's 200 mm pad overhangs its
post like a T (its measured `hook_mm` is about 102 mm), but that pad is its press face, rated to
press, not to pull a lip. So a process tool is never the rigid option that
keeps R03 from readmitting a soft arm or a hand to a lid, a door or a drawer.

**Hooks by geometry.** A rigid tool without a grip (mechanism `fingers`, no grip tag) may do a
`hook_pull` step when its measured hook (`ToolInfo.hook_mm`, `measured_hook_mm`) is at least 5 mm
[assumed]: from the calibrated footprint of the open head (the hook pull's own calibration), over 72
headings, how far the geoms wholly within 15 mm of the tip reach sideways past every other geom of
the tool, and past the flange when the flange face is less than 6 mm [assumed] above the hook (no
lip fits between them): the hook pull's own `Footprint.hook` with its `flange` argument. Its
tags do not count either way. The library's button pusher (tagged
`push`) has an offset pad 24 mm past its shank and past the flange, so it may hook; the
drawer-pull hook (tagged `hook`) is a plate flush under the flange, so it does not. Grippers and
hands hook by gripping and soft arms by curling (their mechanism); a cup never hooks.

`grasp`, `suction` and `wrap` are one **acquire** class (ruling 6): any acquire-capable tool is a
candidate on a step with any of them, and S2's pick controllers drive the head's own v1
primitive whatever the step names ("the pick's mechanism is the tool's"). The task keeps one
`task_id` (the step is not rewritten); the screen records the primitive each candidate actually
used (`Cell.primitive_used`: `grasp` for fingers, `suction` for cups, `wrap` for soft arms), and
the certificate prints it next to the step ("grasp (as suction)"). The fidelity tier is the worst
of the step's primitive, its templates and the effector's family, so a cup on a grasp step is an
approximation with the suction family's caveat.

Fit checks (ruling 4): a pick of a part runs the family's own build against that part; a pick of
a plug or threaded member runs it against a cylinder of the member's size (what S2's pick plans
with); a cup must be no wider than the face it seals on (part top, plug top, a cap; a coupling or
nut has an open top; a slot head breaks the seal); an insert of a port plug picks the plug, so it
gets the same checks as a pick of it; a gripper must open 5 mm wider than the threaded member it
turns or holds in place for an insert; a pre-shaped gripper needs the handle plus 6 mm each side.
A **vacuum frame** fits a face only when the face covers at least the frame's engagement count of
its cups (`FrameGeometry.min_sealed`: half of them rounded up, at least three, never more than the
frame has): with the frame centred on the face and its x along the face's x, as the pick aims it,
a cup counts when its lip circle lies on the face (a box's top, a cylinder's, a plug's or a cap's
round top, a latch's wing). A 100 mm part under the 2 x 2 frame covers none of its 4 cups and is
refused before any physics; the 300 x 300 mm tile covers all 4. A **nut runner** fits a threaded
joint only when the member is a hex nut (a cap or coupling is round) of its socket's size (13 mm
across flats, within 0.5 mm [assumed], no bigger than the cavity) and at least `MIN_MEMBER_MM`
tall (12 mm, measured: a shorter one jams or slips in the socket). It drives only threaded
members (a threaded joint, a panel's clamp: `rules.THREADED_TEMPLATES`), so it is not screened on
a plug, a latch or a valve ("a nut runner drives only a threaded nut or clamp; ...").

## 3. Per-step screen and the success matrix

Steps are grouped into **hold chains**: consecutive steps joined by `hold_after`, which one tool
must do (`Setup.problems` refuses a change while holding). Each surviving candidate runs each
chain as a single-tool setup on the chain's **sub-task** (every object kept, only the chain's
steps), on the first `n_screen` (3) conditions of the grid.

**Carried context.** Chain *j* starts from the world state that the best tool of chain *j - 1*
left behind under the same condition (`runner.run_segments(..., initial=WorldState)`). That is
exactly the contract's tool-change model (STUDIO.md 4.4: the wrist starts at home with the new
tool, the objects keep their joint positions), so a screen cell is what that step would see in a
tool-change setup after the best earlier tools. `run_segments` reuses the episode's own step
runner, scene cache and outcome builder; with no initial state it gives `run_task`'s result
exactly (a test compares `traj_sha`). The best tool of a chain is the highest chain success rate,
then fewer actuators, lower mass, shorter time, effector id. Cells record `context`
(`start`/`carried`) and how many conditions had all earlier chains succeed in that context.

**The success matrix** (`Screen.matrix()`): candidates x steps; each cell is the success rate and
the primitive used, or the pruning reason (a step pruned with its chain says at which step).
Cells also keep failure modes, median durations, fidelity, caveats, the controller and the
trajectory hashes.

## 4. Combination search

Every setup is scored from the matrix: predicted completion = product of its chains' rates,
weakest chain rate, cycle time = the chains' median durations plus every change (6 s per tool
change, 1.5 s per head switch), and complexity (`ComplexityMetrics`: tools, actuators, total
mass, changes, bracket). Ordering: completion, weakest chain, cycle time, complexity.

* **single:** every candidate that survived pruning on every chain.
* **tool_change:** for each chain its top 3 candidates; every combination with 2 to 4 distinct
  tools (consecutive chains on the same tool merge); the shared default `ToolChanger`.
* **multi_head:** pairs of the chains' top candidates, each chain on the better head; the 6 best
  pairs get a **bracket geometry search**: layouts (head angles) -30/30, -45/45, -60/60,
  -75/75, 0/60 and 0/75 deg, standoffs 50, 70, 100 and 150 mm, lateral 0 (a twist needs the head
  on the roll axis), the default `BracketSpec`; the lightest layout that passes the swept-volume
  check is kept (shortest standoff, then smallest separation, then largest clearance).

**Swept-volume check** (`swept_volume_problems`, ruling 2). The scene of the setup is compiled
(head 0 presented) and every tool-side collision geom is covered by spheres (its oriented
bounding box split into near-cubes). Objects are their geoms' boxes at the start and, for hinges
and slides, also at the joint's far limit (an opened lid or drawer). A layout is rejected, with
the reason, when:

1. the two heads overlap;
2. a head or an adapter enters the arm's last link: a cylinder of the flange's radius plus 10 mm
   upward from the flange's mounting face [assumed keep-out; the floating wrist has no arm];
3. switching heads at the home pose (the contract's segment start, 300 mm above the table
   centre) swings either head or the bracket within 5 mm of an object or the table;
4. at any step's engagement pose (the active tool's centre at the handle's site) the idle head or
   the bracket comes within 5 mm of another object or the table, at every quarter heading
   (the controller chooses the heading), or over the whole turn for a twist (the roll turns the
   bracket). A step whose tool tilts the wrist (`tilted_poses`) is checked at its pitched poses
   instead: a lid's lip pinched by a two-finger gripper whose opening fits it (the same rules as
   the controller, `two_finger` and `lip_pinch_fits`, its apertures calibrated on the head's
   probe scene) at the lip carried along the
   arc from closed to the pull's goal, the pitch 90 deg less the lid's angle and the yaw that sets
   the pitch axis along the hinge; a push from the front at its handle, tilted 90 deg square to
   the face. The roll is 0 (a segment starts there) unless a twist earlier in the segment turned
   it (then a pinch's either half turn, a push's every quarter turn). A tilted approach the palm
   cannot reach is not taken, by the check as by the controllers;
5. while a step tilts the wrist, from straight down through every pitch it uses (5 deg steps), a
   head, an adapter or a bracket arm enters the arm's link before the pitch joint: a cylinder of
   the flange's radius plus 10 mm from 50 mm above the pitch axis upward, in that link's frame
   (the flange's at pitch 0, so this adds nothing straight down), which does not tilt with the
   tool [assumed keep-out: room for the flange stage to turn through +-90 deg]. The pitch turns
   about the same axis as a bracket's head angles, so a bracket whose idle head sits on the side
   the tilt turns toward swings it up through the arm (a two-finger head at -75 deg and a cup at
   75 deg, pinching a lip at a 90 deg tilt, is refused; a cup straight down with the gripper at
   -120 deg passes). Item 4 catches the other half: with the cup 60 deg from the gripper (100 mm
   standoff) the tilt brings the idle cup down to the table.

Setups whose carried stack (heads, bracket and adapters, or tool plus changer plates) exceeds
the payload limit are refused (ruling 5), with the reason.

## 5. Finalists and the recommendation

**Finalists** (3 to 5, default 4), in rigid-first order (integrator rules R09, R10): setups
predicted to complete at least once go rigid first, then those with a soft arm, then those with a
dexterous hand, and setups predicted never to complete come last, so a soft-arm or hand setup
takes a slot only once the viable rigid setups have one. In that order: the best predicted setup,
the best rigid single tool from any source (the R01 baseline candidate: designed and part-shaped
jaws count), the best single off-the-shelf effector (always), the best tool change and the best
multi-tool head, then the next best. They
run the whole condition grid (default 8: `nominal`, then `place01`... with per-object placement
errors uniform in +-2 mm and their own seeds; the seed also draws the v1 friction and aim jitter)
through `rtt.studio.episode.run_task`, every setup on the same conditions in the same order.

**Recommendation** (`recommend_setups`), a pure function of the finalists' `SetupResult`s:

1. **Evidence.** Results from the S1 reference controllers are refused: listed and labelled
   ("reference controllers: not evidence"), never ranked, no recommendation. Then
   `rtt.studio.fairness.assert_fair` on every run (same wrist, constants, task, conditions,
   seeds, controllers, changer and presentation time); an unfair set raises `ContractError`.
2. **Eligible:** completed the task at least once; no run reset the simulator.
3. **Base order:** `SetupResult.rank_key()` (completion, weakest step, median cycle with changes,
   complexity), then the setup id.
4. **Integrator rules** (`rtt.studio.rules`), one pick at a time from the remaining setups:
   * **R09, R10:** setups with no soft arm and no dexterous hand first ("rigid" in the rules'
     sense: a cup counts), then setups with a soft arm, then setups with a dexterous hand (the
     largest complexity penalty). Eligible, never preferred: a hand is recommended only when no
     rigid or soft-arm setup completed the task, and the reason says so.
   * **R01 (a single gripper is the default):** setups are compared by their added pieces
     (`added_pieces`: effectors beyond one, plus 1 for a tool changer; a two-tool head adds 1, a
     two-tool change adds 2). The baseline is the best single tool on the base order (so completion
     first). It is picked unless (a) its weakest step succeeds in under half the runs
     (`SIMPLICITY_MARGIN["step_floor"]` = 0.5 [assumed]) and a more complex setup completes more,
     or (b) a more complex setup completes more than 0.05 per added piece better
     (`SIMPLICITY_MARGIN["per_extra_effector"]` [assumed]: more than 5 points for a two-tool head,
     more than 10 for a two-tool change). Every more complex setup is tested against the baseline.
     The ones that displace it compete again the same way: the best of the fewest pieces among
     them is the next base, and a setup of more pieces must beat it by the margin per piece of
     difference. Cycle time never lets a more complex setup win; it is still reported.
   * **Same complexity** (the v1 simplicity rule), only among the setups R01 lets through: those
     of the chosen piece count that no more complex setup displaces. Within 0.05 (`TIE_MARGIN`) of
     the best one's completion and weakest step and at most 10 % slower [assumed] they tie, and
     the simplest (tools, actuators, total mass, changes) wins. So a simpler single that a
     two-tool head beats by more than the margin is not picked over the best single, and a simpler
     head that does not clear the margin over the single is not picked over the head that does.

   The reason names the rule and the setups it compared ("Integrator rule R01: One tool finishes
   the job (100% of the runs, cycle 80.0 s against 62.0 s); the 2-tool change (..., 100%) is not
   more than 10 points better, so the simpler setup wins: 1 tool against 2."; "Integrator rule
   R01: it completes 80% of the runs, more than 5 points better than one tool (S, 70%), the margin
   for 1 added piece. The 2-tool change (T, 85%) is not more than 5 points better than it ...").
   A soft-arm or hand pick says "No rigid setup completed the task" only when rigid setups ran and
   failed; when none was among the finalists it says so.
5. **Fidelity rule:** an approximation-tier winner carries the best rigid-tier eligible setup
   (`runner_up_rigid`) and "approximation: verify on hardware".
6. **Baseline:** the R01 baseline (the best eligible single tool of the pick's rule class, which
   may be a designed or part-shaped one; the pick itself when a single tool wins; a soft-arm or
   hand single when no rigid one is eligible) is printed beside every pick (`baseline`, and
   `versus_baseline`: completion and cycle differences, the added pieces, the gain they had to
   earn (`null` for an R09/R10 baseline) and what decided: `baseline`, `tie`, `margin`,
   `step_floor`, `R09`/`R10` or `ranking`; `note` says which simpler single the tie picked);
   ranking rows carry `baseline` and `added_pieces`. When R01 keeps one tool and the
   same-complexity tie picks a simpler single over the best one, the baseline stays the best single
   by completion and the basis is `tie`. When no single tool
   is eligible, `baseline` is `{"setup": null, "basis": "no single tool is eligible: ..."}` with
   each single's reason (or "no single tool was among the finalists"). The best single off-the-shelf effector of the search is reported too
   (`best_single`), with the recommendation's difference in completion, cycle time and tools.
7. **No winner:** `recommendation: null`, the reason, and the three closest setups with where
   their runs stop (`stops_at`: the first failing step of each run and its failure modes).

## 6. Sensitivity to the change times

Ruling 3 and integrator rule R07: the 15 N m roll torque, 180 deg/s roll speed, 7.5 s tool change
and 1.5 s head switch stay [assumed]. `sensitivity(results)` re-ranks the same runs with tool
changes of 3, 6, 7.5, 10 and 12 s and head switches of 0.75, 1.5 and 3 s (15 pairs). Under R01 a
change time never overturns the single tool against a more complex setup; it can reorder setups
of the same complexity (two tool changes with different numbers of changes). A change is a fixed clock advance, not simulated
physics, so each run's cycle time is recomputed exactly
(`cycle - change_time + n_tool_changes * c + n_presentations * p`). The report gives each pair's
order, recommended setup, whether the recommendation or the order changed, and each setup's
position per pair. The certificate summarises it.

## 7. Build pack

`export_build_pack(report, out_dir, setup_id=None)` packs one finalist (the recommendation by
default; a report without one needs `setup_id`). Contents:

| File | Kind | Licence | Notes |
|---|---|---|---|
| `bracket.stl` (multi-tool head) | bracket | CC0-1.0, generated | hub, two arms (the simulation's capsules, trimmed flush with the flange face and the tool faces), two adapters at the `HeadMount` poses; ISO 9409-1-50-4-M6 holes [assumed]; one watertight solid, below the flange face |
| `changer_plate_tool<k>.stl` (tool change) | changer_plate | CC0-1.0, generated | 63 mm x half the changer stack, both bolt patterns (turned 45 deg) and a 20 mm bore [assumed]; one watertight solid |
| `mjcf/setup_tool<k>.xml`, `mjcf/meshes/*.stl` | mjcf | scene CC0-1.0 combined with the vendored models' licence (the most demanding is stated); meshes keep their source licence | the exact tested scene for each tool configuration; reloaded from the pack and compared with the tested model (bodies, geoms, joints, actuators, masses, poses, contact and actuator parameters) |
| `cad/tool<k>_reference.stl` (procedural tools) | effector_cad | CC0-1.0, generated | the simulated collision geometry, for sizing and clearance (not a manufacturing drawing) |
| `licences/tool<k>_LICENSE.txt` (vendored tools) | licence | the source's | the upstream licence text |
| `bom.csv` | bom | CC0-1.0 | tools (with their simulated limits), vacuum or cable actuators, printed parts with mass, the changer, fasteners |
| `ASSEMBLY.md`, `README.txt` | notes | CC0-1.0 | mounting, head angles or tool order, the task on the setup, what to check first |
| `certificate.html` | certificate | CC0-1.0 | self-contained (no scripts, no external files): result, setup, per-step success with primitive used, failure modes, fidelity tiers and caveats, every condition with its run and trajectory hash, the compared setups and the best single, the sensitivity summary, assumed constants, provenance |
| `CREDITS.txt` | licence | CC0-1.0 | every third-party credit line, copyright, upstream and change (the only file where upstream names may appear) |
| `manifest.json` | | | `BuildPackManifest`: every file with sha256, kind, format, SPDX licence, source (`library`, `generated`, `studio`), source ref, brand-free attribution, watertight; upstream links for vendored tools; the bill of materials; caveats |

A deterministic zip (`<folder>.zip`: sorted entries, fixed timestamps) sits beside the folder.

* **Licences.** Every file's licence is on the permissive list (`ALLOWED_LICENSES`); amber or red
  items are pruned before the search and refused by the export. Library and vendored files carry
  their licence record's attribution; credit lines with upstream names go to `CREDITS.txt`, and
  upstream URLs (which may contain a maker's name) are data in the manifest's `links`, rendered
  as link targets, as on the library page.
* **Formats.** STL only for meshes. **STEP is not provided**: no CAD kernel (cadquery, OCP,
  build123d) is installed. 3MF and GLB are not provided either: the contract's build pack formats
  are `stl, step, xml, csv, html, md, txt, json`. Vendored OBJ meshes are converted to binary STL
  (geometry only, noted per file).
* **MJCF export workaround.** `MjSpec.to_xml` crashes (segfault, MuJoCo 3.15.0) on scenes with
  some attached vendored hands because of their default classes. The export therefore points
  every element at the main default class before writing (element attributes already hold their
  resolved values, so the compiled model is unchanged; the reload check proves it) and names the
  unnamed default classes an attached scratch spec brings along.

## 8. Orchestrator rulings applied

| # | Ruling | Where |
|---|---|---|
| 1 | Force-feedback search within a step through wrist and effector commands only, with an equal per-step time and attempt budget | S2's controllers (`BUDGET_S`, STUDIO_PRIMITIVES.md 4); S5 runs every candidate on the same steps, conditions and seeds and the planned durations include the budgets |
| 2 | Multi-tool head presentation stays a fixed tilt plus `present_s`; a swept-volume check before accepting a multi-head setup, rejected with a reason | `swept_volume_problems`, `bracket_search` (section 4) |
| 3 | 15 N m, 180 deg/s, 7.5 s and 1.5 s stay [assumed]; a sensitivity report for 3/6/7.5/10/12 s and 0.75/1.5/3 s | `ASSUMED_CONSTANTS`, `sensitivity` (section 6) |
| 4 | Each tool's opening and fit checked against every object it touches | `fit_problem` (section 2) |
| 5 | An arm payload limit (every head, bracket or changer plates) in the complexity; over the limit refused; the limit [assumed] | `ARM_PAYLOAD_KG = 3.0`, `prune`, `combine`, `bracket_search` |
| 6 | grasp, suction and wrap are one acquire class; every acquire-capable effector competes under its own adapter; the primitive used is recorded | `ACQUIRE`, `capability_problem`, `Cell.primitive_used` (section 2) |

## 9. Assumed constants

Equal for every candidate; none is sourced from a target arm or changer. They are printed on every
certificate and in `report.constants` (each tagged `assumed`).

| Constant | Value |
|---|---|
| wrist roll torque limit | 15 N m |
| wrist roll speed cap | 180 deg/s |
| wrist pitch limit, speed cap, torque limit | +-90 deg, 180 deg/s, 15 N m |
| tool change time | 7.5 s |
| multi-tool head switch time | 1.5 s |
| changer master plate, tool plate, stack | 0.25 kg, 0.10 kg, 52 mm |
| arm payload limit | 3.0 kg |
| bracket density | 1300 kg/m^3 (printed carbon-filled polymer) |
| placement error of the condition grid | uniform +-2 mm per object |
| R01: margin per added piece; step floor | 0.05; 0.5 |
| same-complexity tie margin; cycle-time margin | 0.05; 10 % |
| R09: heaviest part a soft arm carries | 2.5 kg |
| R05: torque from which tightening needs a nut runner | 4 N m (1 N m above hand-tight) |
| nut runner socket fit | within 0.5 mm of its size across flats |
| hook: least sideways reach; room under the flange | 5 mm; 6 mm |
| swept-volume clearance; arm last-link keep-out | 5 mm; flange radius + 10 mm |
| arm keep-out before the wrist pitch | flange radius + 10 mm, from 50 mm above the pitch axis |
| bolt pattern | ISO 9409-1-50-4-M6 |

## 10. Agent tools and command line

```python
from rtt.studio.search import run_task_tournament
from rtt.studio.buildpack import export_build_pack

report = run_task_tournament(task_spec_or_its_dict, workers=6)   # TournamentReport
data = report.to_dict()                                          # JSON: screen, matrix, finalists, runs,
                                                                 # recommendation, sensitivity, constants
pack = export_build_pack(data, "runs/studio/h1/build_pack")      # or setup_id=... for any finalist
pack.to_dict()                                                   # manifest, zip path, MJCF checks
```

`run_task_tournament` options: `candidates` (specs or `Candidate`s), `library`, `n_conditions`
(8), `n_screen` (3), `n_finalists` (4), `top_k` (3), `multi_head_pairs` (6), `workers` (at most
6 processes: the host is shared), `controllers` (`"reference"` runs everything but is never
evidence), `payload_kg`, `kinds`, `pool` (reuse one `WorkerPool`).

```
python -m rtt.studio.search TASK.json --out DIR [--workers 6] [--conditions 8] [--pack]
```

## 11. Hero gate

Run 2026-10-18 on the final S4b plans (`src/rtt/studio/engineer/recorded/gate-2026-10-18-s2/`,
task ids checked against the manifest), S2's real primitives, every candidate (30: 21 family
presets, 3 of them the parallel-jaw reference that cannot build here, and 9 designed
effectors), 3 screen conditions, 8 full-run conditions, 4 finalists, 6 worker processes on a
16-core host shared with two other jobs.

| | H1 returns tote | H2 pipe coupling | H3 equipment cabinet |
|---|---|---|---|
| task id | `0bf0b255bbe7` | `c5bee032ddeb` | `67155d5f1d53` |
| steps (chains) | 4 (4) | 2 (2) | 4 (4) |
| screen runs; wall time | 192; 130 s | 138; 110 s | 255; 321 s |
| total wall time (CPU use) | 192 s (4.4 cores) | 125 s (5.0 cores) | 372 s (4.6 cores) |
| step with no passing candidate | open_lid 0/23 (slip, force over 20 N), close_lid 0/23 | insert_coupling 0/23 (depth 4.0002 mm against a [4.0, 4.0] mm window) | unlatch 0/23, open_door 0/23 (penetration), open_valve 0/26, plug_in 0/13 (jam) |
| steps some candidate passes in the screen | pick_carton 9/9, pick_pack 9/9 (hands and cups) | tighten_coupling 6/23 (designed hands) | none |
| setups scored (single, tool change, multi-head) | 6, 81, 0 of 6 geometry searches | 20, 9, 6 | 13, 81, 0 of 6 |
| multi-head rejections | the idle head comes within 5 mm of the open lid during the picks; the switch swing at home | none needed (one accepted) | the switch swing at home goes through the door (at the table centre) |
| finalists' completion | 0/8 each; all stop at open_lid | 0/8 each; all stop at insert_coupling | 0/8 each; all stop at unlatch |
| recommendation | none (nothing completed) | none | none |
| best single off-the-shelf | five-finger 22-joint hand, power grasp | two-finger gripper, rigid pads A | five-finger 18-actuator hand, power grasp |

The top finalist's runs on two conditions reproduced their `traj_sha` twice in-process against
the spawned pool. The packs of the closest finalists (and of every tool-change and multi-head finalist:
changer plates and one bracket) are complete, every printed part is one watertight solid, every
exported MJCF reloads as the tested scene, every licence is on the permissive list.

What the failures are (no parameter was changed):

* **H1.** A lid with a lip defeats every tool from the top (S2's known limit); the designed
  sliding hook gripper is refused at build time against the library block (section 12), the
  swinging one exceeds the 20 N limit. The parts start inside the tote's footprint, under the
  closed lid (the layout). The push to close never reaches 10 deg in the screen's carried state.
* **H2.** The insert's success window is zero wide (`depth_mm` at least 4.0 and at most 4.0);
  every tool seats the coupling at the thread start, measured 4.0001 to 4.0002 mm, and fails by
  0.2 micrometres. Six designed effectors (hands, tentacles, a hook gripper) then tighten it to 3
turns in at least one screen condition, three of them in all three.
* **H3.** The latch sits inside the door's footprint (latch x -120 to -80 mm, door x -165 to
  150 mm, 410 mm tall), so the twist meets the door (roll torque at the 10 N m limit with no
  force on the latch); the door pull penetrates more than 1 mm; the 8 mm connector jams (S2: the
  12 mm default already does).

### Rerun on the S4c plans (2026-10-18)

Plans from `recorded/gate-2026-10-18-s4c/` (no overlaps, no zero-width windows), contract
CR-S5-1 (tools that pick nothing are built against a 10 mm probe), the open library merged
(49 candidates: 21 presets, 19 library entrants, 9 designed), 4 worker processes. S2b's fixes
(small-plug insert, lid-lip hook pull, door penetration) were not merged yet. No setup
completes any hero task.

| | H1 `b75435602453` | H2 `59ba912a5fa8` | H3 `4f607694742c` |
|---|---|---|---|
| wall time (4 workers) | 257 s | 266 s | 267 s |
| steps no candidate passes | open_lid 0/29 (slip; the sliding hook gripper 24 N and an angular jaw 32 N against the 20 N limit); pick_carton 0/20, pick_pack 0/18 (the lid stays shut) | tighten_coupling 0/29 (slip: an angular jaw reaches 3.4 turns with 33 deg of slip; hands slip at 0.2 to 0.5 N m) | unlock_latch 0/30 (tools turn about 174 deg over the wing, which stays at 0 deg; contact force 52 to 909 N against a 10 N limit); open_valve 0/34 (the sliding hook gripper turns it 84 deg at 2.9 N m but presses 14 N against a 10 N limit); insert_plug 0/25 (jam, depth 0) |
| steps some candidate passes | close_lid 29/33 (the lid was never opened) | insert_coupling 22/29 (depth 4.0 mm) | open_door 11/29 (the sliding hook gripper 89 deg at 28 N; a force-controlled library gripper 73 deg) |
| best single off-the-shelf | a library angular-jaw gripper (0.11 kg, 1 actuator) | a library force-controlled gripper | a library force-controlled gripper |

With the lid set open (diagnostic only: hinge at 100 deg, as if open_lid had worked), a single
suction cup (16, 25 or 40 mm) does both picks into the tote and closes the lid on all three
screen conditions, so H1 waits only on the lid-lip hook pull. CR-S5-1 lets the designed sliding
hook gripper compete, but the designed family's own validators refuse 8 of the 9 designed
effectors against the 10 mm probe (digits collide, or the lowest point is not 5 mm below the
probe's top), so they no longer compete on steps that pick nothing; in the first gate three of
them tightened H2's coupling.

## 12. Open issues

* **A tool that picks nothing** was built against the library block until contract CR-S5-1
  (now the 10 mm probe cube), which kept the designed sliding hook gripper out of hook pulls in
  the first hero gate (section 11). The fit of such a tool on the objects it touches is checked
  by the search's fit rules, not by its family's grip checks.
* **Screening is greedy in context.** Chain *j* is screened after the best tool of the earlier
  chains; a setup whose earlier tool leaves a different state is only measured in the full runs.
  Predictions multiply chain rates (independence), so finalists, not predictions, are the
  evidence.
* **The head switch happens at the home pose** (the contract's segment start). A long tool on a
  two-tool head can be rejected because its swing reaches a tall object or the table there,
  although an arm could present it higher up.
* **Objects in the swept-volume check** are their start and fully-open boxes; parts that earlier
  steps moved (into a tote) are checked where they started.
* **Rigid packs, not drawings.** Bolt patterns, changer plates and bracket material are
  [assumed]; the BOM names simulated limits, not part numbers.
