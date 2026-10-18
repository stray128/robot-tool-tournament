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
   caveat), every primitive (what it acts on, through which handles, its "how" params, what it
   observes, whether it may hold), the quantities, every rule the checks enforce, a table mapping
   everyday verbs to primitives (deformables, liquids and processes go to `unsupported` or a
   refusal), the wrist and table limits, and one worked example of a *different* use case (a
   drawer and a small box, built from the contract, so it is always valid).
3. **One forced call** of `submit_task_plan` on `nvidia/nemotron-3-super-120b-a12b` through
   `rtt.llm.LLMClient` (cache, ledger, spend cap, retries; temperature 0.6, top_p 0.95,
   max_tokens 12,000, thinking on).
4. **Judge.** `task_spec_from_tool_args` (structure, bounds, cross-references, every problem
   with its path), then `check_plan` (`checks.py`):
   * every step has a goal criterion (a state quantity of its object, or `lift_mm`, `hold_s`,
     `in_target`), not only force or torque limits;
   * `in_target` needs a target, and a step that places its object releases it;
   * an `insert` directly follows a step that holds the same object;
   * criteria are reachable: a lid within `max_angle_deg`, a drawer within `travel_mm`, a latch
     within `turn_deg`, a valve within `travel_deg`, a thread within `engage_turns` (depth within
     `engage_turns x pitch_mm`), a plug within the socket's `depth_mm`, a pick within its own
     `lift_mm` and `hold_s`;
   * a twist's torque (latch or valve `torque_nm`; a thread's `final_torque_nm` when the step
     seats it, else `start_torque_nm`) is within the wrist roll's 15 N m, the step's
     `max_torque_nm` and any `torque_nm` bound; a drawer's `pull_force_n` and a plug's
     `insert_force_n` are within the step's `max_force_n`;
   * refs: `upload:<id>` names an upload that exists, and an uploaded part is used; `library:`
     names a training part; `scan:` is not resolvable here;
   * a refusal lists what is not simulated with at least one nearest primitive.
   Objects nothing touches are warnings, never problems. When the spec as a whole does not
   construct, the checks still run on every object and step that does, so one repair prompt
   lists every problem.
5. **One repair call** if anything is wrong: the same messages plus the previous call (scrubbed)
   and every problem. Whatever is still wrong after it is reported (`status: invalid`), never
   silently fixed. The one fallback: a refusal whose items still name no nearest primitive gets
   one from a keyword table (`nearest_primitive`), and the plan says so.
6. **Status**: `valid`, `refused` (a valid refusal), `invalid`, or `failed` (no usable call:
   transport, unreadable answer, spend cap).

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

## 4. HTTP routes (for the integrator)

`studio_router(svc)` (`src/rtt/studio/engineer/api.py`) is an `APIRouter` over the existing
`AgentService`: its passcode gate, its daily spend book and its part store. **Integrator: add one
line to `create_app` in `src/rtt/api/app.py`, after the existing routes:**

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

Recorded live on 2026-10-18 (`recorded/gate-2026-10-18/`, 11 calls, $0.098):

| case | setting | result | calls | steps produced | $ |
|---|---|---|---|---|---|
| h1_tote | warehouse | valid first try | 1 | hook_pull tote_lid, grasp carton -> tote_bin, grasp shrink_pack -> tote_bin, push tote_lid | 0.0079 |
| h2_coupling | construction | valid first try | 1 | grasp coupling, insert coupling, twist coupling | 0.0074 |
| h3_cabinet | maintenance | valid after repair | 2 | twist latch_qtr, hook_pull door_cab, twist valve_hw, grasp port_test, insert port_test | 0.0241 |
| e1_drawer_kitting | warehouse | valid first try | 1 | hook_pull parts_drawer, grasp cube_part -> kit_tray, grasp rod_part -> kit_tray, push parts_drawer | 0.0073 |
| e2_rod_nut | construction | valid first try | 1 | grasp m10_nut, twist m10_nut, grasp m10_nut -> tray | 0.0076 |
| e3_isolation_valve | maintenance | valid first try | 1 | hook_pull valve_hatch, twist iso_valve, push valve_hatch | 0.0164 |
| e4_vial_holder | lab | valid first try | 1 | hook_pull hinged_cover, grasp instrument_holder, insert instrument_holder, push hinged_cover | 0.0080 |
| e5_fridge_restock | retail | valid first try | 1 | hook_pull fridge_door, grasp drink_can -> basket, push fridge_door | 0.0073 |
| u1_fold_shirt | unsupported | refused (nearest: wrap) | 1 | none | 0.0059 |
| u2_pour | unsupported | refused (nearest: grasp) | 1 | none | 0.0060 |

Plans valid 8/8 (7 on the first call, 1 after the repair); unsupported 2/2 refused with a reason
and a nearest primitive. The cabinet's first call ended two steps with `hold_after: true` while
the next step acted on another object; the repair fixed both. One more recorded plan
(`recorded/upload-2026-10-18/`, $0.0068) plans a use case with an uploaded STL
(`assets/parts/rounded_tub.stl`) and uses it as `upload:stl-575c1d026461`.

Token Factory spend of the whole track: $0.044 (three tuning calls on the cabinet, cloth and nut
cases) + $0.098 (gate) + $0.007 (upload plan) = **$0.149**, under the $0.50 cap.

## 7. Conventions proposed to S2, and open issues

* **Signs of state quantities** (the contract says "from its start" only). The prompt tells the
  model: a lid's or door's `angle_deg` and a drawer's `travel_mm` count from the start, positive
  toward open (so a closing step bounds them from above); for a twist, `angle_deg` and `turns`
  are positive in the step's own direction (a quarter turn is `angle_deg at_least 85`, either
  way); `depth_mm` is positive going in. S2's predicates should report them that way, or the
  prompt changes (and the gate is re-recorded). The checks compare magnitudes, so they hold
  either way.
* **Semantics beyond the rules are not checked.** The nut plan (e2) ends with a grasp that puts
  the nut back in its tray, because its twist ended holding and the last step must release;
  layouts are not checked for overlapping footprints (S2 decides each template's footprint). The
  user edits these on the page.
* **Runaway completions.** Two of the eleven recorded calls (h3 main, e3) ran to the 12,000-token
  limit after a complete tool call (about $0.009 each). A lower `max_tokens` (6,000; the largest
  useful completion was 2,700 tokens) would save that, at the price of re-recording the gate.
* **Recorded replays are tied to the prompt.** Any change to the prompt templates, the catalog
  rendering, the worked example, the tool schema or the sampling is a different request: the
  replay tests then fail with a cache miss, and the gate must be re-recorded
  (`gate --out ... --live`, about $0.10), then `export-web`.
* The upload path inherits the part intake's gripper-sized limits (79 mm, 2 kg); larger parts in
  a Studio use case need a Studio intake.
* The API's `plan_task` agent tool and the Studio pages' navigation entry belong to S7.
