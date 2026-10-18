# Studio Task Engineer and intake page (track S4)

**Product mode. Not part of the pre-registered experiment.** Every record, page and API answer
carries "Product mode: not part of the pre-registered experiment".

The Task Engineer turns a use case written in plain words (plus an optional STL) into a
`TaskSpec` of the frozen Studio contract (`rtt-effectors/2`, [STUDIO.md](STUDIO.md) section 3):
the objects as parametric templates and the ordered steps as primitive skills with bounded
params, constraints and success criteria. The user checks and edits the step list on the intake
page before anything runs.

| What | Where |
|---|---|
| Task Engineer (prompts, forced call, checks, repair, refusals, uploads, gate, CLI, HTTP routes) | `src/rtt/studio/engineer/` |
| Recorded live runs (scrubbed cache, ledger, manifest) | `src/rtt/studio/engineer/recorded/` |
| Tests (offline: scripted or recorded responses only) | `tests/test_studio_engineer.py`, `web/tests/studio_intake.test.ts` |
| Intake page `/studio/new/` | `web/app/studio/new/`, `web/components/studio/intake/` |
| Page smoke test | `web/scripts/smoke_studio_intake.mjs` |

## 1. How a plan is made

1. **Text.** The use case is trimmed, cut at 2,000 characters, and held-out vocabulary and brand
   names are replaced by `[removed]` (the plan notes it). No prompt is sent that names either
   (`PromptLeakError` before any network traffic).
2. **Prompt** (`prompts/system.md`, `user.md`; every table rendered from the contract by
   `catalog.py`, so it cannot drift from the validator): the use case, any uploaded part, every
   object template (params with range and default, handles, state quantities, fidelity and
   caveat, and how the simulator models it, section 1a), every primitive (what it acts on, through which handles, its "how" params, what it
   observes, whether it may hold), the quantities, every rule the checks enforce, a table mapping
   everyday verbs to primitives (deformables, liquids and processes go to `unsupported` or a
   refusal), the wrist and table limits, and one worked example of a *different* use case (a
   drawer and a small box, built from the contract, so it is always valid).
3. **One forced call** of `submit_task_plan` on `nvidia/nemotron-3-super-120b-a12b` through
   `rtt.llm.LLMClient` (cache, ledger, spend cap, retries; temperature 0.6, top_p 0.95,
   max_tokens 5,000, thinking on; section 6 has the token counts behind that number).
4. **Judge.** `task_spec_from_tool_args` (structure, bounds, cross-references, every problem
   with its path), then `check_plan` (`checks.py`):
   * every step has a goal criterion (a state quantity of its object, `lift_mm`, `hold_s` or
     `in_target` for a pick, or `torque_nm` at_least for a twist that tightens until snug), not
     only limits;
   * `in_target` needs a target, and a step that places its object releases it;
   * directions (section 1a): a twist on a latch or a valve is `ccw`, on a thread `cw`; the other
     way only turns back an object an earlier twist turned the working way; a twist on a latch or
     a valve takes no `torque_nm` at_least (the twist stops at that torque, which a latch or valve
     reaches before it turns);
   * a threaded member: no pick acts on it; a twist on it comes after an insert of it; that
     insert ends with `hold_after` false and pushes it no deeper than the thread start
     (`chamfer_mm` + 3 mm);
   * criteria are reachable: a lid within `max_angle_deg`, a drawer within `travel_mm`, a latch
     within `turn_deg`, a valve within `travel_deg`, a thread within `engage_turns` (depth within
     `chamfer_mm` + 3 mm + `engage_turns x pitch_mm` when seated), a plug within the socket's
     `depth_mm`, a pick within its own `lift_mm` and `hold_s`;
   * a twist's torque (a latch's or valve's `torque_nm`; a thread's torque where the step stops,
     on its ramp from `start_torque_nm` to `final_torque_nm`; any `torque_nm` at_least) fits what
     the twist allows itself, 90 % of `max_torque_nm`, else 95 % of the wrist roll's 15 N m
     (14.25 N m), and any `torque_nm` at_most; a hook pull on a drawer needs its `pull_force_n`
     within `max_force_n`, a push on a drawer and an insert of a plug push with at most 80 % of
     `max_force_n`, else 30 N;
   * refs: `upload:<id>` names an upload that exists, and an uploaded part is used; `library:`
     names a training part; `scan:` is not resolvable here;
   * a lid over a bin (the same position) needs the bin 120 mm tall, the lid's fixed hinge height;
     a use case that states another height has that lid-on-bin combination in `unsupported`;
   * a lid with feature `lip` over a bin overhangs it by `feature_mm` on every side (S2b's hook
     slides under the free edge, CR-S2b-1): `width_mm` at least the bin's x extent + 2 x
     `feature_mm`, `depth_mm` at least its y extent + 2 x `feature_mm` (the extents of the bin as
     turned relative to the lid); a lid with feature `handle` may match its bin. The check and the
     prompt state the rule; neither picks the feature for the planner;
   * a pick (grasp, suction or wrap) into a bin under a lid needs an earlier `hook_pull` of that
     lid with `angle_deg` at_least 95 (past vertical: a lid opened less falls shut when the tool
     lets go), and no `push` on that lid between that hook pull and the pick (S4e);
   * layout (`layout.py`, section 1b): plan-view footprints and swings at least 20 mm apart
     [assumed], except a lid at its bin's position and the plug of a port (one object); so a part
     picked into a bin starts outside it, and nothing stands in a lid's, door's or drawer's swing;
   * windows: a criterion with both bounds is at least 1 mm (`depth_mm`, `travel_mm`, `lift_mm`,
     `slip_mm`, `penetration_mm`), 5 deg (`angle_deg`, `tilt_deg`, `slip_deg`), 0.25 turns,
     0.2 N m, 1 N (`force_n`, `lateral_force_n`) or 0.2 s (`hold_s`) wide [assumed]; the flags
     `in_target` and `jam` are exempt (S5 found a [4.0, 4.0] mm window that every tool missed by
     0.2 micrometres);
   * a refusal lists what is not simulated with at least one nearest primitive.
   Warnings never refuse a plan: objects nothing touches, a lid whose stop is at or below
   vertical (it falls shut). When the spec as a whole does not
   construct, the checks still run on every object and step that does, so one repair prompt
   lists every problem.
5. **One repair call** if anything is wrong: the same messages plus the previous call (scrubbed)
   and every problem. Whatever is still wrong after it is reported (`status: invalid`), with two
   stated exceptions: a refusal whose items still name no nearest primitive gets one from a
   keyword table (`nearest_primitive`), and the layout fixer (step 6). Both are in the plan's
   notes.
6. **Layout fix** (S4e, `layout.fix_layout`, Python only). When the plan after the repair call
   has layout problems (overlaps, swings, clearances) and nothing else, code moves the free
   objects to the nearest spot that clears every other object within the pose limits (+-300 mm,
   inside the wrist's reach). A lid on its bin moves only with its bin, and a port's plug is part
   of the port. A bin a step places into keeps its place when moving the other object of the
   conflict clears it. The search is a fixed order: offsets on a 10 mm grid, nearest first, ties
   counter-clockwise from +x, out to 850 mm. It takes one conflict at a time (in the order the
   check lists them) and moves the later object of the pair unless that one is such a bin. So the
   same plan always gives the same moves. Each moved object adds one assumption to the plan, for
   example "Moved test_port from (0, -150) to (0, -240) mm to clear the swing of cabinet_door".
   The moved plan is checked again in full, and the result is `valid` with `layout_fixed` true
   ("valid after layout fix"). If no spot clears a conflict, or the moves would need more than
   the 10 assumption lines, the layout problems stand. The page shows the moved positions, the
   moves and the assumptions. It has no "Fix layout" button; edited plans are only validated.
7. **Status**: `valid`, `refused` (a valid refusal), `invalid`, or `failed` (no usable call:
   transport, unreadable answer, spend cap). `result` says how a plan got there:
   `valid_first_call`, `valid_after_repair`, `valid_after_layout_fix`, `refused` or `failed`.

### 1a. The simulator's conventions (track S2)

The prompt, the checks and the intake page follow S2's models ([STUDIO_PRIMITIVES.md](STUDIO_PRIMITIVES.md));
`catalog.py` holds the numbers and a test compares them with the template and primitive modules.

| Template | How S2 models it | What a plan must do |
|---|---|---|
| `hinged_lid` | lid: closed underside 120 mm above the table (no hinge-height parameter), stays open only past vertical; door: vertical, hinged on its left edge; `angle_deg` 0 closed, positive open | open with `hook_pull`, close with `push` (`angle_deg` at_most); a bin under a lid is 120 mm tall; a `lip` lid overhangs its bin by `feature_mm` on every side, a `handle` lid need not; a pick into a bin under a lid comes after a `hook_pull` of the lid to `angle_deg` at_least 95 |
| `drawer` | starts closed; `travel_mm` positive open | `hook_pull` out, `push` shut |
| `quarter_turn_latch` | starts locked at its stop; opens counter-clockwise from above; `angle_deg` ccw from locked | twist `ccw`; no `torque_nm` at_least |
| `handwheel_valve` | starts closed at its stop; opens counter-clockwise; `angle_deg` ccw from closed | twist `ccw`; closing a valve that starts closed is not modelled |
| `threaded_joint` | the member rests loose on its stub, captive on the stub's axis; a lead-in of `chamfer_mm` + 3 mm to the thread start; tightens clockwise, `engage_turns` to seated, torque ramp `start_torque_nm` to `final_torque_nm`; `depth_mm` down from the stub top | no pick; insert (`hold_after` false) to the thread start, then twist `cw` |
| `port_plug` | one object: the socket and the plug standing 40 mm beside it; `depth_mm` the tip below the mouth | `insert` (it picks the plug itself; a grasp with `hold_after` before it is optional) |

**Not modelled (S2 limitation):** picking a loose coupling, cap or nut and starting it on a
stub (the member starts on its stub), closing a valve or locking a latch that starts closed. The
planner puts such actions in `unsupported` (nearest primitive `insert` or `twist`) or, when they are
the whole use case, refuses. H2's description was reworded to the modelled task (the coupling
sits loosely on the stub; push it onto the thread start and turn it on until snug, under
8 N m); see section 6 for E2, E3 and H3.

### 1b. Footprints and the layout check

`layout.py` gives every object a plan-view footprint and, for lids, doors and drawers, a swing,
as convex polygons in table mm, from the template's params and S2's builder constants (a test
builds every template variant, closed and at its stop, at two yaws, and checks that every geom
lies inside them, within 5 mm for a door's arc drawn as chords):

| Template | Footprint (centred on the pose, before yaw) | Swing |
|---|---|---|
| `bin` | `length_mm` x `width_mm` | |
| `part` | `size_x_mm` x `size_y_mm` (a cylinder `size_x_mm` across) | |
| `quarter_turn_latch`, `threaded_joint` | the plate or base, `max(1.6 x diameter, 40 mm)` square | |
| `handwheel_valve` | `max(rim_diameter_mm, 70) x max(rim_diameter_mm, 50)` | |
| `port_plug` | the socket (16-sided, bore + `max(6, plug/2)` wall + chamfer) and the plug 40 mm beyond it along +x | |
| `drawer` | `width_mm` x `depth_mm`, plus the front and handle toward -y | the drawer slid out by `travel_mm` |
| `hinged_lid` (lid) | `width_mm` x `depth_mm`, plus the back wall on +y | past vertical: `depth_mm x -cos(max_angle_deg)` beyond the hinge |
| `hinged_lid` (door) | the panel along x hinged at `x = -depth_mm/2`, with its jamb, stop post and handle | the sector of radius `depth_mm` (plus the handle) about the hinge to `max_angle_deg` toward -y |

Two objects whose footprints or swings come within 20 mm [assumed] are a problem on the later
object's path ("its footprint overlaps the swing of objects[1] (door)"), one per pair. The
prompt states the footprints and the rule, so the planner can lay a scene out; in the recorded
run it still placed parts inside their totes, a latch on its door and a valve inside its box on
the first call, and the repair moved them (section 6).

Object parameters stay what the description says or a stated assumption: nothing is tuned toward
what passes in physics (S2's default 0.8 N m latch and 12 mm plug slip or jam with catalog
grippers; the plans keep them where the description implies them).

**Null written as text.** Nemotron writes `"refusal": "None"` (Python's null) in most plans; a
first trial lost an otherwise valid cabinet plan to it. `normalise_nulls` reads `"None"`,
`"null"`, `"n/a"` and `""` as JSON null in nullable fields (refusal, target, ref, criterion
bounds, constraints, nearest primitive), and the plan notes it. Nothing else is rewritten.

**The tool schema.** The default tool is the contract's own `submit_task_plan_tool()`
(33 KB, descriptions included). `--compact-tool` drops every `description` from the schema
(22 KB, the same strict structure). In a live trial on the cabinet case the compact schema made
the model leave out the steps' required `description` field in both calls, so the full schema
is the default; the prompt tables already carry the descriptions, so the extra 11 KB is the only
cost (about 2,800 prompt tokens, $0.0008 a call).

**Guards and replays.** Reasoning text is dropped before a response is cached
(`EngineerCache`), and every string the model wrote is scrubbed (held-out words, brands) before
it is cached and before it is judged, so a replay from the cache gives exactly the plan the live
run gave, with zero calls and no key.

## 2. Uploaded parts (`upload:<id>`)

`UploadRegistry.add_stl` runs the existing part intake (`rtt.part.intake.intake`: units, repair,
mass, resting poses, convex pieces) and saves it with `save_part` as `<root>/<id>/part/`, the
layout the API's part backend already writes for `POST /parts`. Ids are `stl-<sha256[:12]>` on
the command line (content addressed) and the API's `part-<hex>`. The prompt gives the part's
size as it rests, its mass and material, and the exact `part` params and ref to use.
`install_upload_resolver` registers the registries with
`rtt.studio.templates.part.register_part_resolver("upload", ...)`; the `part` template then
builds the most likely resting pose from its convex pieces (family `upload`). The intake's own
limits apply: some resting pose must fit the 79 mm parallel-gripper opening, and the part must
weigh at most 2 kg.

## 3. Command line

```
python -m rtt.studio.engineer plan --case h1_tote                 # replay a recorded case
python -m rtt.studio.engineer plan --text "..." [--stl PART.stl] --live --env-file PATH/.env --budget 0.05
python -m rtt.studio.engineer gate --out DIR                       # replay the recorded gate run
python -m rtt.studio.engineer gate --out DIR --live --env-file PATH/.env --budget 0.50
python -m rtt.studio.engineer export-web                           # regenerate the intake page's data
```

Without `--live` every call is replayed from a cache (by default the recorded gate run); a
description that is not in it stops with exit code 2. `--live` reads `NEBIUS_API_KEY` from
`--env-file` on the first call (never printed) and caps everything recorded in the run's ledger
at `--budget`.

## 4. HTTP routes

`studio_router(svc)` (`src/rtt/studio/engineer/api.py`) is an `APIRouter` over the existing
`AgentService`: its passcode gate, its daily spend book and its part store. `create_app` in
`src/rtt/api/app.py` mounts it after the existing routes:

```python
from rtt.studio.engineer.api import studio_router

...
app.include_router(studio_router(svc))
```

* `POST /studio/plan` `{"text", "part_id"?}` (passcode in the body or `X-RTT-Passcode`). A
  description of the recorded gate run is answered from its cache, free and without the passcode.
  Anything else calls Nemotron: the passcode is required (403; 429 when locked out), one plan
  runs at a time (429), each plan spends at most `RTT_STUDIO_PLAN_USD` (default $0.05, both
  calls) inside the server's daily cap (`RTT_AGENT_DAY_USD`; 429 when a cap refuses it). An
  offline server plans recorded descriptions only (409). `part_id` names a part uploaded with
  `POST /parts` (404 if unknown, 422 without an intake record). Answer:
  `{"label", "source": "recorded"|"live", "result": PlanResult.to_dict(), "spend"}`.
* `POST /studio/validate` `{"plan", "part_ids"?}` -> `{"status", "problems", "warnings",
  "task_id", "fidelity"}`; no model call, no passcode. Bodies over 256 KB get 413.

The existing CORS settings already allow these calls (POST, `Content-Type`, `X-RTT-Passcode`).
The agent tool `plan_task` (S7) can call `plan_task(text, client=make_engineer_client(...))`
directly with the session's guarded ledger.

## 5. The intake page, `/studio/new/`

1. **Describe the use case**: three example prompts (the hero use cases: warehouse returns tote,
   pipe coupling, equipment cabinet), the rest of the recorded test set in a list, a text box, and
   an optional STL drop (read in the browser; size and triangle count shown).
2. **Plan it**: on the public static site (no `NEXT_PUBLIC_API_URL`) Plan replays the recorded
   live plan of an example, call by call (a repaired plan shows its first call's problems), and
   says so; other text gets an honest "this static site has no model to call". In live mode Plan
   sends the STL through `POST /parts` first, then `POST /studio/plan` with the passcode in a
   header, and shows the spend.
3. **Check and edit the steps**: objects (collapsed to a summary; every param within its bounds,
   pose, label) and steps (reorder, delete, add; object, handle, target, params, up to four
   criteria, task limits, holding, order dependencies), each with its fidelity tier and caveat.
   Problems appear on the entry they name; warnings in amber. A refusal shows its reason and the
   nearest primitive, with a button that starts a plan from it. **Download TaskSpec JSON** (and
   Copy) export `TaskSpec.to_dict()`: contract, label, `task_id`, source (`recorded`,
   `nemotron` or `manual` once edited), the plan and the per-step fidelity; only a valid plan
   exports. In live mode "Check with the API" asks `POST /studio/validate`.

The page validates in the browser (`taskspec.ts`), message for message the same as
`judge_args`: `export-web` writes `parity.json` (46 versions of the recorded plans, 41 of them edited, every
kind of problem) and `web/tests/studio_intake.test.ts` checks identical problems, warnings,
statuses, task ids (the same canonical JSON and SHA-256), fidelity and caveats. Brand names are
not checked in the browser (the list is not shipped to it); the API and the Python parser check
them. `catalog.json` and `recorded.json` are generated the same way; a Python test regenerates
all three files and compares them with the committed copies.

## 6. Gate (design S4) and spend

The gate is judged on **one** full live run (S4e). A case that fails stays recorded as failed and
is not drawn again. The page's summary line and this table report that run's numbers exactly.

Recorded live on 2026-10-18 with every rule above in the prompt (`recorded/gate-2026-10-18-s4e/`:
one run, 17 calls, $0.133, the run's cap $0.24):

| case | setting | result | calls | steps produced | $ |
|---|---|---|---|---|---|
| h1_tote | warehouse | valid after repair | 2 | hook_pull tote_lid, grasp carton -> tote_bin, grasp pack -> tote_bin, push tote_lid | 0.0159 |
| h2_coupling | construction | valid after repair | 2 | insert pipe_coupling, twist pipe_coupling (cw) | 0.0152 |
| h3_cabinet | maintenance | valid after layout fix | 2 | twist qt_latch (ccw), hook_pull cabinet_door, twist service_valve (ccw), insert test_port | 0.0169 |
| e1_drawer_kitting | warehouse | valid after repair | 2 | hook_pull shelf_drawer, grasp cube_part -> kit_tray, grasp rod_part -> kit_tray, push shelf_drawer | 0.0182 |
| e2_rod_nut | construction | valid after repair | 2 | insert threaded_rod_nut, twist threaded_rod_nut (cw) | 0.0141 |
| e3_isolation_valve | maintenance | valid after repair | 2 | hook_pull meter_hatch, twist iso_valve (ccw), push meter_hatch | 0.0155 |
| e4_vial_holder | lab | valid first call | 1 | hook_pull instrument_lid, grasp sample_holder, insert sample_holder, push instrument_lid | 0.0088 |
| e5_fridge_restock | retail | valid after layout fix | 2 | hook_pull fridge_door, grasp drink_can -> storage_basket, push fridge_door | 0.0152 |
| u1_fold_shirt | unsupported | refused (nearest: grasp) | 1 | none | 0.0065 |
| u2_pour | unsupported | refused (nearest: grasp) | 1 | none | 0.0065 |

Plans: 8 of 8 valid in that one run: 1 on the first call, 5 after the repair call, 2 after the
layout fix, 0 failed. Unsupported: 2 of 2 refused with a reason and a nearest primitive. The gate
(at least 90 % valid, every unsupported case refused) passed.

What the repair calls fixed: parts inside the tote, the tray or the drawer (h1, e1), the latch and
the valve against the door (h3), the valve against the hatch (e3), steps left holding while the
next acted on another object (h3, e3), an insert of a thread left holding (h2), a twist on a nut
before its insert (e2). What the layout fixer moved after the repair: h3's test port, from (0, -150)
to (0, -240), out of the door's swing; e5's fridge door, from (0, 0) to (60, -60), off the basket
that the can goes into (the basket is a target, so it kept its place).

### 6a. The hero plans, frozen for S7

The three hero plans below are frozen for the recorded Studio sessions (S7). A test pins their
`task_id`s (`FROZEN_HERO_TASK_IDS` in `tests/test_studio_engineer.py`), so any later change to
them is deliberate. Positions are in mm; the brackets give each footprint's and swing's plan-view
extent in x and y.

* **h1_tote** (task `e4586f36...`):
  * Objects:
    * tote: 300 x 200 x 120 mm at (0, 0) [x -150..150, y -100..100];
    * lid: 340 x 240 mm, lip 20 mm (20 mm over the tote on every side), stop 110 deg, at (0, 0)
      [footprint x -170..170, y -120..134; swing y 120..208];
    * carton: 120 x 80 x 60 mm, 300 g, on the bench at (-200, -200) [x -260..-140, y -240..-160];
    * pack: 150 x 100 x 20 mm, 150 g, on the bench at (200, -200) [x 125..275, y -250..-150].
  * Steps:
    1. hook_pull the lid (lip, from the top, 80 mm/s, at most 20 N): `angle_deg` at_least 95,
       past vertical;
    2. grasp the carton into the tote (lift 150 mm, hold 1 s): `in_target` at_least 1,
       `tilt_deg` at_most 30;
    3. grasp the pack into the tote (the same);
    4. push the lid shut (60 mm/s, at most 20 N): `angle_deg` at_most 5.
* **h2_coupling** (task `33599e99...`):
  * Object: the coupling, 40 mm across, 45 mm long, pitch 1.5 mm, 2 engaged turns, 1 mm chamfer,
    torque 0.2 rising to 8 N m when seated, at (0, 0) [x and y -32..32].
  * Steps:
    1. insert it to the thread start (10 mm/s, 5 mm standoff, at most 30 N): `depth_mm` at_most 4
       (the thread starts at 4 mm), `jam` at_most 0; releases;
    2. twist it `cw` (60 deg/s, `max_torque_nm` 10): `torque_nm` 3 to 8 (hand-tight, never
       over 8), `turns` at_least 0.25.
* **h3_cabinet** (task `3297a111...`):
  * Objects:
    * latch: wing head 20 mm, 0.5 N m, 90 deg, at (-100, 0) [x -120..-80, y -20..20];
    * door: 200 mm wide, 400 mm tall, handle, stop 120 deg, at (100, 0) [footprint x -15..200,
      y -32..24; swing x -151..203, y -201..0];
    * valve: 3 spokes, 100 mm wheel, 2 N m, 90 deg travel, at (0, 150) [x -50..50, y 100..200];
    * port: the 8 mm round connector (30 mm long) in a 15 mm socket, 0.2 mm clearance, at
      (0, -240) after the layout fix [x -11..59, y -251..-229].
  * Steps:
    1. twist the latch `ccw` (60 deg/s, at most 5 N m): `angle_deg` 80 to 100;
    2. hook_pull the door (handle, from the front, 100 mm/s, at most 20 N): `angle_deg` 100 to
       130;
    3. twist the valve `ccw` (60 deg/s, at most 5 N m): `angle_deg` 80 to 100;
    4. insert the connector (10 mm/s, 5 mm standoff, at most 30 N): `depth_mm` 12 to 18.

The hero descriptions were made specific in S4e where a user would know the detail: h2 says
"hand-tight (about 3 N m), never over 8 N m", and h3 says "the 8 mm round test connector". The h1
parts stay on the bench.

One more recorded plan (`recorded/upload-2026-10-18-s4e/`, one run, $0.0157, valid after the
repair) plans a use case with an uploaded STL (`assets/parts/rounded_tub.stl`) and uses it as
`upload:stl-575c1d026461`.

**max_tokens.** Through S4d, 83 complete calls had at most 3,150 completion tokens (reasoning,
at most 1,791, included). 7 calls kept generating after a complete tool call and ran to the cap
(8,000 or 12,000), which was billed. S4e lowered the cap to 5,000. In the S4e run no call ran to
the cap, but one complete call (e1's repair) used 4,403 tokens. So the margin is thinner than the
earlier counts suggested; section 7 has more.

Token Factory spend: S4 $0.149 (cap $0.50); S4b $0.175 (cap $0.20); S4c $0.196 (cap $0.25);
S4d $0.224 (cap $0.25); S4e $0.148 (cap $0.30): $0.133 for the one gate run and $0.016 for the
upload plan.

## 7. Open issues

* **First calls rarely lay a scene out right.** 1 of 8 plans passed on the first call in S4c,
  S4d and S4e alike. The repair call does not always clear the layout: S4d's run had two plans
  still overlapping after it, and in S4e the layout fixer cleared two (h3, e5). The fixer moves
  objects in code and states each move, but it does not know which placement makes physical sense:
  h3's test port now stands in front of the door's swing, not behind the door. A worked example
  with a lid on a bin and a door, or a layout helper the planner calls, could raise the
  first-call rate.
* **The lip rule grows h1's lid.** The planner meets the overhang by enlarging the lid. h1's lid is
  340 x 240 mm over a 300 x 200 mm tote, so the parts need more bench room around the tote.
* **max_tokens 5,000 has a thin margin.** One complete S4e call used 4,403 tokens, against 3,150 at
  most before. A longer call than that would be cut off and need the repair call. The next
  re-record should raise the cap to about 6,000. Changing it changes every request, so the frozen
  recording keeps 5,000.
* **Nested scenes are not modelled.** The templates stand side by side on one table: a valve in
  a box under a hatch, cans in a crate, a latch on a door are separate objects that must stand
  20 mm apart. The h3 latch now stands beside the door's hinge side, where a real latch would sit
  at the free edge; nothing checks which side.
* **Semantics beyond the rules are not checked**: the model may still misread a quantity or a
  limit, or fill in one the use case does not give. With "hand-tight (about 3 N m)" h2 now stops
  at 3 N m; h3's latch torque (0.5 N m) is the planner's own guess. The user edits these on the
  page.
* **Recorded replays are tied to the prompt.** Any change to the prompt templates, the catalog
  rendering (including `layout.py`'s numbers and the S2 conventions), the worked example, the
  tool schema or the sampling is a different request: the replay tests then fail with a cache
  miss, and the gate must be re-recorded (`gate --out ... --live`, about $0.15), then `export-web`.
  The re-record is one run, judged as it comes; the hero plans' pinned `task_id`s then need a
  deliberate update.
* The upload path inherits the part intake's gripper-sized limits (79 mm, 2 kg); larger parts in
  a Studio use case need a Studio intake.
* S2's limits stay limits of the plans: a loose member cannot be started on a stub, a valve or
  latch that starts closed cannot be closed, a lid's hinge height is fixed at 120 mm, and an insert
  held into a twist on a thread is not exercised by S2's tests (so the checks forbid it).
