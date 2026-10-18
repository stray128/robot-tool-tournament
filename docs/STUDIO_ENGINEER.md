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
   * a refusal lists what is not simulated with at least one nearest primitive.
   * a lid over a bin (the same position) needs the bin 120 mm tall, the lid's fixed hinge height;
     a use case that states another height has that lid-on-bin combination in `unsupported`.
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

Recorded live on 2026-10-18 with S2's conventions in the prompt (`recorded/gate-2026-10-18-s2/`,
17 calls, $0.130):

| case | setting | result | calls | steps produced | $ |
|---|---|---|---|---|---|
| h1_tote | warehouse | valid after repair | 2 | hook_pull tote_lid, grasp carton -> tote_bin, grasp pack -> tote_bin, push tote_lid | 0.0157 |
| h2_coupling | construction | valid after repair | 2 | insert pipe_coupling, twist pipe_coupling (cw) | 0.0145 |
| h3_cabinet | maintenance | valid after repair | 2 | twist qtr_latch (ccw), hook_pull cab_door, twist handwheel (ccw), insert service_port | 0.0212 |
| e1_drawer_kitting | warehouse | valid first try | 1 | hook_pull shelf_drawer, grasp cube_part -> kit_tray, grasp rod_part -> kit_tray, push shelf_drawer | 0.0075 |
| e2_rod_nut | construction | valid after repair | 2 | insert m10_nut_on_rod, twist m10_nut_on_rod (cw) | 0.0145 |
| e3_isolation_valve | maintenance | valid after repair | 2 | hook_pull valve_hatch, twist isolation_valve (ccw), push valve_hatch | 0.0150 |
| e4_vial_holder | lab | valid after repair | 2 | hook_pull instrument_cover, insert vial_holder, push instrument_cover | 0.0154 |
| e5_fridge_restock | retail | valid first try | 1 | hook_pull fridge_door, grasp can -> basket, push fridge_door | 0.0074 |
| u1_fold_shirt | unsupported | refused (nearest: grasp) | 1 | none | 0.0063 |
| u2_pour | unsupported | refused (nearest: grasp) | 2 | none | 0.0127 |

Plans valid 8/8 (2 on the first call, 6 after the repair); unsupported 2/2 refused with a reason
and a nearest primitive. What the repairs fixed: h1 put its lid on a 200 mm tote (the lid's hinge
is fixed at 120 mm, now an error; the repair made the tote 120 mm and says so in its assumptions);
h2 and e2 ended the insert holding the member;
h3 and e3 asked a latch or valve twist for `torque_nm` at_least; e4 ended a pick holding the vial
before inserting another object; u2's refusal named no nearest primitive.

How the run was assembled (`manifest.json` `runs`): a first live run of all ten cases ($0.065
of the recorded calls; it refused e2 and e3 as then worded); e2 and e3 were reworded to the
modelled tasks (an M10 nut **sits loose on top of** the rod; the isolation valve is **opened**
one full turn counter-clockwise) and h3 got "(90 degrees)" after "a quarter turn", because the
model read a quarter turn as 22 to 45 degrees; those were re-run live, and h2 was repaired live
after the insert-release rule was added ($0.037); h3 and e3 were repaired live after the rule on
`torque_nm` at_least ($0.021); h1 was repaired live after a lid over a bin of another height
became an error, with the prompt unchanged so that the other recordings stay byte-identical
($0.008). The unchanged requests of the earlier runs are reused, so the recorded cache holds
exactly the 17 calls the manifest replays.

The three hero plans, as recorded:

* **h1_tote**: a 400 x 300 x 120 mm fixed tote with a 380 x 280 mm lid at the same position
  (lip, stop 110 deg); hook_pull the lid (lip, top, `angle_deg` 90 to 110, 20 N), grasp the carton
  (120 x 80 x 60 mm, 300 g) into the tote, grasp the pack (150 x 100 x 20 mm, 150 g) into the tote,
  push the lid shut (`angle_deg` at_most 10, 20 N). Assumption stated by the plan: "The tote bin
  height is 120 mm to match the lid hinge height."
* **h2_coupling**: insert the coupling (40 mm, 45 mm long, pitch 1.5 mm, 3 turns, 0.2 to 6 N m) to
  the thread start (`depth_mm` 4 = 1 mm chamfer + 3 mm, 30 N, releases), twist `cw` at 60 deg/s to
  `turns` at_least 3 with `torque_nm` at_most 8 (`max_torque_nm` 10).
* **h3_cabinet**: twist the wing latch (0.8 N m, 90 deg) `ccw` to `angle_deg` at_least 80;
  hook_pull the door (handle, front) to at_least 80 deg (30 N); twist the 3-spoke 120 mm handwheel
  (3 N m, 90 deg travel) `ccw` to at_least 80 deg; insert the 8 mm connector into the 15 mm port
  to `depth_mm` at_least 15 (20 N).

One more recorded plan (`recorded/upload-2026-10-18-s2/`, $0.0073) plans a use case with an
uploaded STL (`assets/parts/rounded_tub.stl`) and uses it as `upload:stl-575c1d026461`.

Token Factory spend: S4 $0.149 (cap $0.50); S4b $0.175 (cap $0.20): $0.065 + $0.037 + $0.021 +
$0.008 of recorded gate calls, $0.038 of first-run calls that the rewording made stale (old e2 and
e3, the first h3 attempt chain), and $0.007 for the upload plan.

## 7. Open issues

* **First-call rule deferred to the next full re-record.** The lid-on-bin height is enforced by
  the checks and its repair message, not yet stated in the prompt's rules, because any prompt
  change re-records every case (about $0.11) and S5 records against these exact prompts. The rule
  text is ready: "Lids on bins: a lid's hinge height is fixed (120 mm, its closed underside), so a
  bin a lid covers (at the lid's position) is 120 mm tall; when the use case states another bin
  height, put that lid on that bin in unsupported (not modelled) instead of changing the height."
  Until then a lid-on-tote plan gets it right after the one repair (h1 did).

* **Semantics beyond the rules are not checked.** Layouts are not checked for overlapping
  footprints; the model may still misread a quantity (it read "a quarter turn" as 22 to 45 deg
  before h3 said 90 degrees). The user edits these on the page.
* **Runaway completions.** S4's run had two calls that kept generating after a complete tool call
  up to the 12,000-token limit; `max_tokens` is now 8,000 (the largest useful completion was
  2,700 tokens). No S4b call hit the limit.
* **Recorded replays are tied to the prompt.** Any change to the prompt templates, the catalog
  rendering (including the S2 conventions in `catalog.py`), the worked example, the tool schema
  or the sampling is a different request: the replay tests then fail with a cache miss, and the
  gate must be re-recorded (`gate --out ... --live`, about $0.10), then `export-web`.
* The upload path inherits the part intake's gripper-sized limits (79 mm, 2 kg); larger parts in
  a Studio use case need a Studio intake.
* S2's limits stay limits of the plans: a loose member cannot be started on a stub, a valve or
  latch that starts closed cannot be closed, a lid's hinge height is fixed at 120 mm, and an insert
  held into a twist on a thread is not exercised by S2's tests (so the checks forbid it).
