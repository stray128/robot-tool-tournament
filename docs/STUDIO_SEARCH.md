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
| a tool that picks nothing in a chain is built against the library block (S1 rule) and its build refuses it | "built against the block: ..." |

Capabilities by mechanism [assumed], unless the library item carries its own tags:

| Mechanism | Families | Serves |
|---|---|---|
| fingers | catalog grippers and hands, designed | grasp, hook_pull, twist, insert, push |
| suction | suction cups | suction, twist (seals on a flat top face), insert (holds by its seal), push |
| soft_arm | soft arms | wrap, hook_pull (curls around a handle), twist |

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
   bracket).

Setups whose carried stack (heads, bracket and adapters, or tool plus changer plates) exceeds
the payload limit are refused (ruling 5), with the reason.

## 5. Finalists and the recommendation

**Finalists** (3 to 5, default 4): the best predicted setup, the best single off-the-shelf
effector (always), the best tool change and the best multi-tool head, then the next best. They
run the whole condition grid (default 8: `nominal`, then `place01`... with per-object placement
errors uniform in +-2 mm and their own seeds; the seed also draws the v1 friction and aim jitter)
through `rtt.studio.episode.run_task`, every setup on the same conditions in the same order.

**Recommendation** (`recommend_setups`), a pure function of the finalists' `SetupResult`s:

1. **Evidence.** Results from the S1 reference controllers are refused: listed and labelled
   ("reference controllers: not evidence"), never ranked, no recommendation. Then
   `rtt.studio.fairness.assert_fair` on every run (same wrist, constants, task, conditions,
   seeds, controllers, changer and presentation time); an unfair set raises `ContractError`.
2. **Eligible:** completed the task at least once; no run reset the simulator.
3. **Rank:** `SetupResult.rank_key()` (completion, weakest step, median cycle with changes,
   complexity), then the setup id.
4. **Simplicity rule** (as v1): among the setups within 0.05 of the best remaining completion
   rate and weakest step, and at most 10 % slower [assumed], the simplest (tools, actuators,
   total mass, changes). The reason is written out ("Within 0.05 of the top completion rate
   ... and simpler: 1 tool against 2").
5. **Fidelity rule:** an approximation-tier winner carries the best rigid-tier eligible setup
   (`runner_up_rigid`) and "approximation: verify on hardware".
6. **Baseline:** the best single off-the-shelf effector is always reported (`best_single`), with
   the recommendation's difference in completion, cycle time and tools.
7. **No winner:** `recommendation: null`, the reason, and the three closest setups with where
   their runs stop (`stops_at`: the first failing step of each run and its failure modes).

## 6. Sensitivity to the change times

Ruling 3: the 15 N m roll torque, 180 deg/s roll speed, 6 s tool change and 1.5 s head switch stay
[assumed]. `sensitivity(results)` re-ranks the same runs with tool changes of 3, 6 and 12 s and
head switches of 0.75, 1.5 and 3 s (9 pairs). A change is a fixed clock advance, not simulated
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
| 3 | 15 N m, 180 deg/s, 6 s and 1.5 s stay [assumed]; a sensitivity report for 3/6/12 s and 0.75/1.5/3 s | `ASSUMED_CONSTANTS`, `sensitivity` (section 6) |
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
| tool change time | 6 s |
| multi-tool head switch time | 1.5 s |
| changer master plate, tool plate, stack | 0.25 kg, 0.10 kg, 25 mm |
| arm payload limit | 3.0 kg |
| bracket density | 1300 kg/m^3 (printed carbon-filled polymer) |
| placement error of the condition grid | uniform +-2 mm per object |
| simplicity margin; cycle-time margin | 0.05; 10 % |
| swept-volume clearance; arm last-link keep-out | 5 mm; flange radius + 10 mm |
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

Pending the re-recorded hero plans (the recorded 2026-10-18 plans have planner mismatches that
S4 is fixing: h3's latch and valve twisted clockwise, h2 lifting a threaded coupling off its
stub). Results will be recorded here when the plans land.

## 12. Open issues

* **A tool that picks nothing is built against the library block** (S1 rule, STUDIO.md open
  question 4). A designed hook gripper whose validators refuse the block cannot be screened on a
  hook pull at all ("built against the block: ..."), although it might open a lip lid. A contract
  change (build such tools against a neutral probe, or against the object they touch) would let it
  compete.
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
