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
   max_tokens 8,000, thinking on).
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
   and every problem. Whatever is still wrong after it is reported (`status: invalid`), never
   silently fixed. The one fallback: a refusal whose items still name no nearest primitive gets
   one from a keyword table (`nearest_primitive`), and the plan says so.
6. **Status**: `valid`, `refused` (a valid refusal), `invalid`, or `failed` (no usable call:
   transport, unreadable answer, spend cap).

### 1a. The simulator's conventions (track S2)

The prompt, the checks and the intake page follow S2's models ([STUDIO_PRIMITIVES.md](STUDIO_PRIMITIVES.md));
`catalog.py` holds the numbers and a test compares them with the template and primitive modules.

| Template | How S2 models it | What a plan must do |
|---|---|---|
| `hinged_lid` | lid: closed underside 120 mm above the table (no hinge-height parameter), stays open only past vertical; door: vertical, hinged on its left edge; `angle_deg` 0 closed, positive open | open with `hook_pull`, close with `push` (`angle_deg` at_most); a bin under a lid is 120 mm tall |
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

Recorded live on 2026-10-18 with S2's conventions, the layout, window and lid-on-bin rules in the
prompt (`recorded/gate-2026-10-18-s4c/`, 19 calls, $0.143):

| case | setting | result | calls | steps produced | $ |
|---|---|---|---|---|---|
| h1_tote | warehouse | valid after repair | 2 | hook_pull tote_lid, grasp carton -> tote_bin, grasp pack -> tote_bin, push tote_lid | 0.0156 |
| h2_coupling | construction | valid after repair | 2 | insert pipe_coupling, twist pipe_coupling (cw) | 0.0150 |
| h3_cabinet | maintenance | valid after repair | 2 | twist latch (ccw), hook_pull door, twist valve (ccw), insert port | 0.0160 |
| e1_drawer_kitting | warehouse | valid after repair | 2 | hook_pull shelf_drawer, grasp cube_part -> kit_tray, grasp rod_part -> kit_tray, push shelf_drawer | 0.0156 |
| e2_rod_nut | construction | valid after repair | 2 | insert m10_nut_on_rod, twist m10_nut_on_rod (cw) | 0.0143 |
| e3_isolation_valve | maintenance | valid after repair | 2 | hook_pull meter_hatch, twist iso_valve (ccw), push meter_hatch | 0.0153 |
| e4_vial_holder | lab | valid after repair | 2 | hook_pull instrument_cover, grasp instrument_holder, insert instrument_holder, push instrument_cover | 0.0172 |
| e5_fridge_restock | retail | valid first try | 1 | hook_pull fridge_door, grasp drink_can -> basket, push fridge_door | 0.0074 |
| u1_fold_shirt | unsupported | refused (nearest: grasp) | 2 | none | 0.0133 |
| u2_pour | unsupported | refused (nearest: grasp) | 2 | none | 0.0129 |

Plans valid 8/8 (1 on the first call, 7 after the repair); unsupported 2/2 refused with a reason
and a nearest primitive. What the repairs fixed: parts inside their tote or tray (h1, e1), the
latch and the valve inside the door's footprint and zero-wide windows (h3), the valve inside the
meter box with a `torque_nm` at_least (e3), the vial and holder against the cover (e4), a step
left holding while the next acted on another object (h3, e3, e4), inserts of a thread left
holding (h2, e2), refusals with no nearest primitive (u1, u2).

How the run was assembled (`manifest.json` `runs`): a first live run of all ten cases ($0.120 of
the recorded calls) left e3 and e5 invalid after the repair, both nesting one object in another
(a valve inside the valve box under its hatch, cans inside a crate under the fridge door's
swing), which the templates cannot represent; e3 and e5 were reworded to say where things stand
(the isolation valve **beside** the meter box; the can on the **bench beside** the fridge, the
basket **behind** the door) and re-run live ($0.023). The other eight cases are the first run's
calls.

The three hero plans, as recorded (positions in mm, plan-view extents x and y of each
footprint and swing):

* **h1_tote**: tote 300 x 200 x 120 mm at (0, 0) [x -150..150, y -100..100]; lid 300 x 200 mm,
  lip, stop 110 deg, at (0, 0) [footprint y -100..114, swing y 100..174]; carton 120 x 80 x 60 mm,
  300 g at (250, 0) [x 190..310]; pack 150 x 100 x 20 mm, 150 g at (-250, 0) [x -325..-175].
  Steps: hook_pull the lid (lip, top, `angle_deg` at_least 90, 20 N); grasp the carton into the
  tote (lift 150 mm, `in_target` at_least 1, `tilt_deg` at_most 20); grasp the pack into the tote
  (the same); push the lid shut (`angle_deg` at_most 10, 20 N).
* **h2_coupling**: coupling 40 mm across, 45 mm long, pitch 1.5 mm, 3 turns, 0.2 to 8 N m, at
  (0, 0) [base x and y -32..32]. Steps: insert to the thread start (`depth_mm` 3.5 to 4.5 around
  the 4 mm start, 30 N, releases); twist `cw` at 60 deg/s until snug (`torque_nm` 2 to 8,
  `max_torque_nm` 15). The planner set the thread's seated torque (`final_torque_nm`) to the
  8 N m limit the description gives, and the twist stops at 2 N m, before the thread seats.
* **h3_cabinet**: latch (wing, 25 mm head, 0.8 N m, 90 deg) at (-200, 0) [x -220..-180, y
  -20..20], beside the door's hinge side; door 200 mm wide, 200 mm tall, handle, stop 110 deg, at
  (0, 0) [footprint x -115..100, y -35..26; swing x -169..103, y -202..0]; valve (3 spokes,
  120 mm, 3 N m, 90 deg) at (200, 0) [x 140..260]; port with its 8 mm connector at (300, 0)
  [x 289..359]. Steps: twist the latch `ccw` (`angle_deg` 85 to 95, 5 N m); hook_pull the door
  (handle, front, `angle_deg` 70 to 110, 30 N); twist the valve `ccw` (`angle_deg` 80 to 100,
  5 N m); insert the connector (`depth_mm` 13 to 17 of the 15 mm socket, 20 N).

One more recorded plan (`recorded/upload-2026-10-18-s4c/`, $0.0157, valid after the repair) plans
a use case with an uploaded STL (`assets/parts/rounded_tub.stl`) and uses it as
`upload:stl-575c1d026461`.

Token Factory spend: S4 $0.149 (cap $0.50); S4b $0.175 (cap $0.20); S4c $0.196 (cap $0.25):
$0.143 of recorded gate calls, $0.038 of first-run calls that the e3 and e5 rewording made stale,
and $0.016 for the upload plan.

## 7. Open issues

* **First calls rarely lay a scene out right.** With the footprints and the rule in the prompt,
  1 of 8 plans passed on the first call; the repair, which names each overlapping pair, fixed
  every one. A worked example with a lid and a door, or a layout helper the planner calls, could
  raise the first-call rate.
* **Nested scenes are not modelled.** The templates stand side by side on one table: a valve in
  a box under a hatch, cans in a crate, a latch on a door are separate objects that must stand
  20 mm apart. The h3 latch now stands beside the door's hinge side, where a real latch would sit
  at the free edge; nothing checks which side.
* **Semantics beyond the rules are not checked**: the model may still misread a quantity or a
  limit (h2 takes "snug, without going over 8 N m" as a seated torque of 8 N m and a stop at
  2 N m). The user edits these on the page.
* **Recorded replays are tied to the prompt.** Any change to the prompt templates, the catalog
  rendering (including `layout.py`'s numbers and the S2 conventions), the worked example, the
  tool schema or the sampling is a different request: the replay tests then fail with a cache
  miss, and the gate must be re-recorded (`gate --out ... --live`, about $0.15), then `export-web`.
* The upload path inherits the part intake's gripper-sized limits (79 mm, 2 kg); larger parts in
  a Studio use case need a Studio intake.
* S2's limits stay limits of the plans: a loose member cannot be started on a stub, a valve or
  latch that starts closed cannot be closed, a lid's hinge height is fixed at 120 mm, and an insert
  held into a twist on a thread is not exercised by S2's tests (so the checks forbid it).
