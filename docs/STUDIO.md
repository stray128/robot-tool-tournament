# Effector Lab Studio (product mode, not pre-registered)

**Product mode. Not part of the pre-registered experiment.** The pre-registered finger
experiment (`prereg-v1`) and the per-part effector layer (`rtt-effectors/1`,
[EFFECTORS.md](EFFECTORS.md)) stay exactly as frozen. The Studio is built beside them, reads them
only, and labels every record, page and agent answer with
"Product mode: not part of the pre-registered experiment" (`rtt.studio.PRODUCT_MODE_LABEL`).

The Studio turns a **use case** (a few interactions with real objects) into a **setup** (one
effector, a two-tool head, or a tool-change sequence) tested in physics, plus the files to build
it. This file is the contract and the build plan. The site never renders it.

Contract version: `rtt-effectors/2`. Code: `src/rtt/studio/`. Contract tests:
`tests/test_studio_contract.py` (data) and `tests/test_studio_scene.py` (physics).

## Contents

1. [What is frozen, what is new](#1-what-is-frozen-what-is-new)
2. [Architecture](#2-architecture)
3. [Data contracts](#3-data-contracts)
4. [Scene and runner](#4-scene-and-runner)
5. [Fairness rules](#5-fairness-rules)
6. [Fidelity tiers](#6-fidelity-tiers)
7. [Build plan: tracks S2 to S7](#7-build-plan-tracks-s2-to-s7)
8. [Open questions](#8-open-questions)

## 1. What is frozen, what is new

| Area | Status | Rule |
|---|---|---|
| `src/rtt/sim/`, `src/rtt/design/freeform/`, `src/rtt/parts/`, `src/rtt/search/`, `src/rtt/analysis/`, `configs/task_v2.yaml`, `uv.lock` and the other PREREG A5 inputs | frozen by PREREG A5 | never edited; `test_prereg_a5_frozen_inputs_are_unchanged` (and the v1 guard test) recompute the hashes |
| the v1 effector contract (`src/rtt/effectors/{__init__,types,registry,scene,runner,certificate,primitives,parallel_jaw}.py`) | frozen (`rtt-effectors/1`) | unchanged; the per-part tournament keeps running on it; `test_the_v1_effector_contract_is_unchanged` |
| `src/rtt/studio/{__init__,types,scene,plan,episode,fairness,reference}.py`, `src/rtt/studio/templates/{__init__,part,bin}.py`, `tests/test_studio_contract.py`, `tests/test_studio_scene.py`, `tests/studio_fixtures.py`, this file | the frozen Studio contract (`rtt-effectors/2`, track S1) | changed only through a contract change request (section 7.1) |
| everything else under `src/rtt/studio/`, the Studio and library web pages, `assets/library/` | built by tracks S2 to S7 | exclusive file ownership (section 7) |

The Studio imports these private helpers of hash-pinned or frozen modules, which therefore
cannot change under it: `rtt.sim.scene._base_spec`, `_add_position_actuator`;
`rtt.effectors.scene._names`, `_settings`, `_is_descendant`, `_mass_properties`,
`_ELEMENT_LISTS`; `rtt.effectors.types._canon`, `_freeze`, `_thaw`.

Like the v1 layer it never touches the held-out suites or the vault (a contract test scans the
package).

## 2. Architecture

```
use case (text, optional STL)
  -> Task Engineer (S4): forced tool call submit_task_plan -> TaskSpec (objects x steps)
  -> the user edits the step list
  -> candidate tools (S3 library + designed effectors) pruned by capability tags (S5)
  -> per-step screening, Nebius fan-out (S5, S6)
  -> combination search: single / multi_head / tool_change Setups (S5)
  -> full-task runs of the finalists: rtt.studio.episode.run_task (S1 core, S5 runner)
  -> SetupResult ranking + best single off-the-shelf baseline (S5)
  -> build pack: BuildPackManifest (S5)  -> Studio UI and agent tools (S7)
```

Modules of the contract (track S1):

| Module | What it holds |
|---|---|
| `rtt.studio.types` | `TaskSpec`, `Step`, `ObjectTemplate` and registry, `Setup`, results, `BuildPackManifest`, the Task Engineer's JSON schema and parser. numpy only. |
| `rtt.studio.scene` | `build_studio_scene(setup, task_spec, tool)`: the wrist with its continuous roll, the tool stack of the setup kind, the template objects, the compiled-model contract checks, measured head masses. |
| `rtt.studio.templates` | the `TemplateBuilder` protocol and registry (auto-import of `rtt.studio.templates.<kind>`); S1 ships `part` and `bin`. |
| `rtt.studio.plan` | the `PrimitiveController` and `SuccessPredicate` protocols, `StepContext`, `criteria_verdict`, `V1SceneView`, the controller registry. |
| `rtt.studio.episode` | segments, carried `WorldState`, the tool-change hook, `run_task` and worker fan-out. |
| `rtt.studio.fairness` | `wrist_signature`, `fairness_sha`, `fairness_problems`, `assert_fair`. |
| `rtt.studio.reference` | trivial reference controllers for hook pull, twist, insert and push (tests and smoke runs only). |

## 3. Data contracts

All contract types are frozen dataclasses that validate on construction (the repository's
style, as in `rtt.effectors.types`). Params are stored canonically and read-only; ids are
sha256 of a canonical JSON form. Params, poses and criteria are in the units their names say
(mm, deg, N, N m, s), which is what the Task Engineer reads and writes; physics code converts.

### 3.1 `TaskSpec`

| Field | Meaning |
|---|---|
| `title`, `description` | the use case on screen (brand-free, at most 80 and 600 characters) |
| `objects` | 1 to 8 `ObjectInstance`: `name` (snake_case id), `kind` (a template), `label`, `params` (the template's bounded params, all present), `pose` (`x_mm`, `y_mm` within +-300 mm, `yaw_deg`), `ref` (`library:<part>`, `upload:<id>` or `scan:<id>` for a mesh part, else null) |
| `steps` | 1 to 12 `Step`, in execution order |
| `assumptions`, `clarifications` | what the Task Engineer assumed; questions with the answer assumed until the user answers |
| `unsupported` | what is not simulated (deformables, for example), why, and the nearest primitive |
| `refusal` | set (with no steps) when the whole use case cannot be simulated |
| `source` | `nemotron`, `manual`, `recorded` or `test` |

`Step`: `id`, `primitive` (`grasp`, `suction`, `wrap`, `hook_pull`, `twist`, `insert`, `push`),
`object` and `handle` (which feature of it), `target` (the destination bin of a pick), `params`
(the primitive's bounded "how"), `constraints` (`max_force_n`, `max_torque_nm`, `precision_mm`,
`precision_deg`, each optional), `success` (1 to 4 `SuccessCriterion`: a quantity with
`at_least` and/or `at_most`), `after` (order dependencies), `hold_after` (the step ends still
holding its object), `description`.

Validation (all problems are listed at once in `TaskSpecError.problems`): object and step
names unique; every step's object exists and is a kind its primitive acts on; the handle
belongs to both the template and the primitive; every criterion's quantity is measured there
(template state or primitive observation); targets are bins; dependencies are earlier steps;
a held object is the next step's object; the last step releases.

`task_id` hashes objects and steps only, so editing the title does not change the task.
`fidelity(step_id)` and `caveats(step_id)` give the step's tier from its primitive and
templates (the runner adds the effector's).

### 3.2 Object templates

| Kind | Model | Handles | State quantities | Tier | Builder |
|---|---|---|---|---|---|
| `hinged_lid` | revolute joint, damping, stop, opening torque; lid or door | `lid`, `handle` | `angle_deg` | rigid | S2 |
| `drawer` | prismatic joint with friction and stops | `drawer`, `handle` | `travel_mm` | rigid | S2 |
| `quarter_turn_latch` | revolute joint with a torque to open, detent and stop; slot or wing head | `head` | `angle_deg` | approximation | S2 |
| `threaded_joint` | revolute coupled to prismatic (helical equality), friction torque ramp; cap, coupling or nut | `member`, `seat` | `turns`, `depth_mm`, `angle_deg` | approximation | S2 |
| `handwheel_valve` | revolute joint with a resistive torque curve; vertical stem | `wheel` | `angle_deg` | approximation | S2 |
| `port_plug` | rigid socket with chamfer and clearance, free plug, optional latch click; vertical | `plug`, `socket` | `depth_mm` | rigid | S2 |
| `bin` | static or free open box | `bin` | none | rigid | S1 |
| `part` | free rigid part: a library or uploaded mesh, or a box or cylinder | `part` | `lift_mm`, `tilt_deg` | rigid | S1 |

Every param of every template is a bounded `ParamSpec` row with a default
(`rtt.studio.types.get_template(kind).params`). `register_template` changes a descriptor (a
contract change).

### 3.3 Primitives

| Primitive | Acts on | Handles | Target | "How" params | Observed | Roll | May hold | Status, tier |
|---|---|---|---|---|---|---|---|---|
| `grasp` | part, port_plug, threaded_joint | part, plug, member | bin | `lift_mm`, `hold_s` | lift, hold, in_target, tilt, force, penetration | | yes | existing (v1 synergies), rigid |
| `suction` | part | part | bin | `lift_mm`, `hold_s` | same | | yes | existing, approximation |
| `wrap` | part | part | bin | `lift_mm`, `hold_s` | same | | yes | existing, approximation |
| `hook_pull` | hinged_lid, drawer | handle | | `speed_mm_s`, `approach` | `slip_mm`, force, penetration | | | new, rigid |
| `twist` | latch, threaded_joint, valve | head, member, wheel | | `direction`, `speed_deg_s` | `slip_deg`, `torque_nm`, force, penetration | yes | yes | new, rigid |
| `insert` | port_plug, threaded_joint | plug, member | | `speed_mm_s`, `standoff_mm` | `lateral_force_n`, force, `jam`, penetration | | yes | new, rigid |
| `push` | hinged_lid, drawer, part, port_plug | lid, drawer, part, plug | | `speed_mm_s` | force, penetration | | | new, rigid |

Quantities (`rtt.studio.types.QUANTITIES`, each with a unit and bounds): `angle_deg`,
`travel_mm`, `depth_mm`, `turns`, `lift_mm`, `hold_s`, `in_target`, `tilt_deg`, `slip_deg`,
`slip_mm`, `force_n`, `lateral_force_n`, `torque_nm`, `jam`, `penetration_mm`.

### 3.4 `Setup`

| Kind | Tools | Geometry | Change cost |
|---|---|---|---|
| `single` | one `EffectorSpec` for every step | on the flange, as in v1 | none |
| `multi_head` | two, one `HeadMount` each, a `BracketSpec` | `HeadMount`: `angle_deg` (tilt from straight down about the bracket's x axis, +-120), `standoff_mm` (wrist centre to the head's mount plane along its axis, 40 to 200), `lateral_mm`, `spin_deg`; heads at least 45 deg apart. `BracketSpec`: hub, arm and adapter sizes, density (a printed polymer by default) | `present_s` (1.5 s) per head switch |
| `tool_change` | two to four distinct effectors, one per step | a `ToolChanger`: `stack_mm` (25), `master_mass_kg` (stays on the wrist), `tool_plate_mass_kg` (travels with each tool) | `changeover_s` (6 s) per change |

`assignment` maps every step id to a tool. `setup_id` hashes everything but the label.
`problems(task)` refuses: unassigned or unknown steps, unused tools, a change or head switch
while an object is held, a twist with a multi-tool head whose head is off the roll axis
(`lateral_mm != 0`). `segments(task)` returns the episodes (section 4.4).

### 3.5 Results

* `StepOutcome`: step, primitive, object, tool, effector id, `success`, `failure_mode`
  (`criterion_not_met`, `constraint_exceeded`, `slip`, `jam`, `dropped`, `physics_invalid`,
  `physics_unreliable`, `invalid_build`, `not_simulated`, `not_reached`), measured quantities,
  `fidelity`, `caveats` (required for an approximation; a `not_simulated` step never succeeds),
  start time, duration, `traj_sha`, controller name.
* `TaskResult`: one full-task run of one setup under one condition and seed: per-step outcomes,
  `completed` (every step succeeded), `cycle_time_s` (step durations plus every change),
  `change_time_s`, tool changes and presentations, the tool mass carried in each segment (the
  mass swap), `traj_sha`, `fairness_sha`, `controllers` (`default` or `reference`).
* `SetupResult`: a setup's runs on one task with `ComplexityMetrics` (tools, actuators, carried
  and total mass, changes, bracket) and `CertificateRef`s (v1 grasp certificates or step
  certificates). Aggregates: completion rate, per-step success, weakest step, median cycle time
  of completed runs. `rank_key()` orders by completion rate, weakest step, cycle time, then
  complexity (design section 5.3); the simplicity margin rule is the recommender's (S5).

### 3.6 Build pack manifest

`BuildPackManifest` (S5 writes `manifest.json`, S7 renders it; `buildpack_manifest_json_schema()`):
task and setup ids, title, `files` (`BuildPackFile`: relative path, sha256, kind
`effector_cad|adapter|bracket|changer_plate|mjcf|bom|certificate|notes|licence`, format, SPDX
licence from the permissive list, source `library|generated|studio`, source ref, attribution
(required for library files), `watertight` (required for printable meshes)), `links` (items
whose files may not be redistributed: upstream URL and reason), `bom`, `caveats`. A pack must
contain the MJCF, the certificate and the bill of materials.

### 3.7 The Task Engineer's tool

`submit_task_plan_tool()` is the forced tool call; its parameters are `task_plan_json_schema()`:
every object closed (`additionalProperties: false`) with every property required, every number
and string bounded, objects and steps as one `anyOf` branch per template kind and per
primitive (const discriminators). The schema stays inside the repository's schema subset
(`tests/schema_lite.py`). `task_spec_from_tool_args(args)` parses and validates, and reports every
problem with its path (`objects[2]: ...`, `steps[1]: ...`) for a repair prompt.
`TaskSpec.to_tool_args()` round-trips.

## 4. Scene and runner

### 4.1 The wrist and its continuous roll

```
world -> palm        slides x, y, z and the yaw hinge: the frozen core wrist servos (task_v2 gains:
                     x/y 100 N, z 150 N, yaw 15 N m), 0.4 kg, no geometry
      -> wrist_roll  continuous hinge about the tool axis through the wrist centre
      -> wrist_flange  the 63 mm x 24 mm flange geom; tools hang below it
```

The roll (`STUDIO_WRIST`, the same for every candidate, [assumed] values): unlimited joint,
position servo kp 20 N m/rad, kv 1 N m s/rad (the core yaw gains), armature 0.001 kg m^2, torque
limit 15 N m (the core yaw servo's), unlimited control so a target can wind several turns,
roll speed cap 180 deg/s enforced by the runner. Rotor 0.05 kg, flange 0.1 kg. Every wrist
body and every tool body is gravity compensated (the arm carries the tool's weight). The control
vector starts with the five wrist servos (x, y, z, yaw, roll), then the heads' actuators, then
the objects' passive actuators.

### 4.2 Tool stacks

* **single:** the family builds under the flange exactly as in the v1 scene (mount plane at wrist
  z = -12 mm). A contract test checks that every tool body sits where the v1 scene puts it and
  that the measured mass is the v1 mass.
* **tool_change:** the changer's master and tool plates (`eff_studio_changer`) sit under the flange
  and the step's tool builds under them, `stack_mm` lower. A segment's scene holds only its tool.
* **multi_head:** a generated bracket (`eff_studio_bracket`: a hub and one capsule arm per head)
  carries both heads, each on its own adapter plate. Each head is built by its family into a
  scratch spec (the scene's options and defaults, no visual assets) and attached under its mount.
  The **active** head keeps its family's element names (`eff_*`), so the family's v1 primitive
  and criterion run unchanged (`V1SceneView`); the **idle** head is renamed `eff_h<k>_*`. The idle
  head's bodies, collision geoms and mass are simulated, and its actuators are held open. To
  present a head, the flange and bracket tilt together, as one rigid assembly about the wrist
  centre, by minus that head's `angle_deg`: the presented head points straight down on the roll
  axis. A contract test checks that the bracket and both heads have the same poses relative to the
  flange whichever head is active (only names and the presentation change).

Every tool-side element is named `eff_` (Studio stack elements `eff_studio_`). The scene refuses,
on the authored spec and the compiled model: unprefixed names, bodies outside the mount, changed
solver or default settings, explicit contact pairs, tool actuators, tendons or equalities that
reach outside the tools, tool geoms with priority >= 1, geom adhesion, a positive margin or a
surface velocity, and contact excludes outside the tool side.

### 4.3 Objects

A template builder (`rtt.studio.templates.<kind>`) adds an instance with every element named
`obj_<instance>_`, under the world body, and declares its bodies, geoms, joints (by role), one
site per handle and its passive actuators (`TemplateHandles`). Object geoms carry priority 1, so
the object's contact parameters govern every tool contact (the v1 part rule). `state()` reports
the template's quantities; `passive()` drives its own actuators from its own state (detents,
torque ramps). A kind without a builder raises `TemplateUnavailable`: its steps are
`not_simulated`.

The family that builds a tool sees the first part that tool picks, so a gripper whose opening is
too small for its part is an `invalid_build`, as in v1. A tool that picks nothing (hook pull, twist,
insert, push only) is built against a 10 mm probe cube (`PROBE_PART_NAME`), so a family's
grip-sized reach and opening checks cannot refuse a hook or pusher; its fit on the objects it
touches is the search's check (S5, ruling 4). Changed from the library `block` in CR-S5-1.

### 4.4 Segments and the tool-change hook

`Setup.segments(task)` groups consecutive steps with the same tool into one **segment**. A
segment is one continuous episode in its own compiled scene: steps run back to back on one
`MjData`, so a part grasped in one step is still in the tool for the next.

A **tool change or head switch** is modelled as the end of one episode and the start of the next,
with the **world state carried** (`WorldState`: every object joint's position by name, at rest) and
the **task clock advanced** by the change time. The new segment's scene contains the new tool (for
a tool change only it, so the carried mass swaps) and the wrist starts at home, 0.30 m above the
table centre. Nothing is held across a change (`Setup.problems`). This is the simplest correct
model: swapping a MuJoCo subtree mid-episode would recompile the model and reset contact and
actuator state anyway, while carrying named joint positions is exact for objects at rest.

`run_task(StudioJob(setup, task_spec, condition, controllers=...))` runs every segment and returns
a `TaskResult`; `run_task_jobs(jobs, workers)` fans out on a spawned pool with identical results.
The task stops at the first failed step (the rest are `not_reached`).

### 4.5 Primitive plan protocol

Per step: `PrimitiveController.plan(ctx, model, data) -> StepPlan` (duration up to 30 s and the
controller's own values), then `command(t, ctx, plan, model, data) -> PrimitiveCommand` every
2 ms control tick: targets for the five wrist servos and controls for the active head's
actuators. `SuccessPredicate.observe(...)` accumulates peaks and last values at every check tick
and at the step's end; `verdict(ctx, acc) -> StepVerdict`. `criteria_verdict(step, measured)` is the
shared final rule: penetration above 1 mm is `physics_unreliable`, a force or torque above the
step's limit is `constraint_exceeded`, an unmet or unmeasured criterion is `criterion_not_met`.

The runner enforces: finite commands of the right shapes, the roll speed cap, no forces applied
directly (`qfrc_applied`, `xfrc_applied`), MuJoCo state resets (`physics_invalid`).

Registry: S2's modules `rtt.studio.primitives.<primitive>` call `register_primitive(controller,
predicate)` on import; `get_primitive(kind)` imports them on first use. Missing means
`PrimitiveNotSimulated`, never a pass. The trivial S1 references (`rtt.studio.reference`) are a
separate set (`controllers="reference"`), and their results say so.

### 4.6 Conditions and what a controller may see

`StudioCondition(condition_id, seed, aim_offsets_mm)`. The seed draws the v1 jitter: every object
geom's nominal friction is scaled by `mu_scale` (+-10 %) and `dx`/`dy` (+-0.5 mm) is added to every
object's aim offset; `aim_offsets_mm` adds per-object placement errors (S6's grid).
Controllers locate objects only through `StepContext.observe_site(data, handle)`, the true handle
position plus the object's aim offset, so the placement error stays unknown to them as in v1,
also for objects an earlier step moved. Predicates read the true state.

## 5. Fairness rules

Every candidate setup in one comparison runs with (documented here, asserted in code):

1. **The same wrist**: the four frozen core servos and the continuous roll (range, limit flags,
   armature, damping, gains, force and control limits), the wrist bodies' masses and gravity
   compensation, and the solver options (`rtt.studio.fairness.wrist_signature`). The roll speed
   cap is enforced by the runner for every controller.
2. **The same Studio constants**: `STUDIO_WRIST`; one `ToolChanger` for every tool-change setup;
   one presentation time for every multi-tool head.
3. **The same task, conditions and seeds**: one `task_id`; the same `(condition_id, seed)` list, in
   the same order, for every setup (common random numbers); one controller set.
4. **The same physics for objects**: object geoms carry priority 1, so no tool buys grip with its
   own contact parameters; tool geoms may not have priority, adhesion, margin or surface velocity.
5. **Effectors at their own documented limits** (v1 rule, [EFFECTORS.md](EFFECTORS.md) section 2.4),
   printed on every card; the task's own limits (`StepConstraints`) apply to every candidate alike.

Every `TaskResult` carries `fairness_sha` (wrist signature, wrist constants and task digest).
`fairness_problems(results, setups)` lists every violation; `assert_fair` raises
`ContractError`. The recommender (S5) calls `assert_fair` before ranking; the contract tests
check that the three setup kinds share one `fairness_sha` and that a mismatched seed grid, a
different changer or mixed controller sets are caught.

## 6. Fidelity tiers

| Tier | Meaning | Shown |
|---|---|---|
| `rigid` | rigid multibody model with a scripted controller | the result |
| `approximation` | a labelled approximation: suction seal, soft arm, latches, threads, valves | the result and its caveat, always (a step outcome without its caveat cannot be constructed) |
| `not_simulated` | no physics model yet (deformables, a missing builder or controller, a primitive whose S2 fidelity gate failed) | "not simulated yet", never scored as a pass |

A step's tier is the worst of its primitive's, its templates' and its effector's (v1 `core` and
`rigid` map to `rigid`). The v1 caveats ([EFFECTORS.md](EFFECTORS.md) section 10) still apply, plus:
the floating wrist abstracts the arm (no arm reach or collision model); presenting a multi-tool
head is a fixed tilt per segment, not a simulated motion; a tool change is a fixed time, not a
simulated docking; controllers are scripted and open-loop on the aim offset. A recommendation is a
simulation ranking under the stated conditions, not a hardware guarantee.

## 7. Build plan: tracks S2 to S7

### 7.1 Rules for every track

* Work in your own worktree from the integration branch that holds this contract:
  `git worktree add -b studio-<track> /home/ashwith/rtt-worktrees/studio-<track> <base>`;
  `PYTHONPATH=<worktree>/src`; the shared venv at `/home/ashwith/robot-tool-tournament/.venv`.
* Edit only the files you own (below). Never edit the frozen contract files (section 1), the v1
  effector contract, the PREREG-frozen code or configs, `pyproject.toml` or `uv.lock`. Never push,
  never spend (Nebius and Token Factory runs are S6's and need the founder's go). Plain
  `git commit`, no co-author trailers.
* Need a contract change? Write it down (what, why, which test proves it) and hand it to the
  contract owner, who changes the contract, updates the tests and asks every track to rebase.
* Common gates (all must hold before merge):
  * **G1 scope:** `git diff --name-only <base>...` lists only owned files.
  * **G2 frozen:** `tests/test_studio_contract.py`, `tests/test_studio_scene.py` and
    `tests/test_effectors_contract.py` pass unchanged.
  * **G3 determinism:** two in-process runs and one spawned-worker run give the same `traj_sha`.
  * **G4 labels, brands and fidelity:** every record and page carries the label and every
    approximation its caveat; no brand token on screen (`find_brand_tokens`).
  * **G5 hygiene:** `ruff check` and `ruff format --check` clean; full `pytest` green;
    `scripts/leak_check.sh` clean.

### 7.2 Track S2: templates and primitives in physics

**Owns:** `src/rtt/studio/templates/{hinged_lid,drawer,quarter_turn_latch,threaded_joint,handwheel_valve,port_plug}.py`,
`src/rtt/studio/primitives/` (package: one module per primitive, `grasp`, `suction`, `wrap`,
`hook_pull`, `twist`, `insert`, `push`, plus shared helpers), `tests/test_studio_templates.py`,
`tests/test_studio_primitives.py`, `docs/STUDIO_PRIMITIVES.md`.

**Builds:** the six articulated template builders (section 3.2 models; every param honoured,
handle sites where a tool engages, `state()` quantities, passive torque curves through their own
actuators); real controllers and predicates for all seven primitives. Grasp, suction and wrap
adapt the v1 family primitives and criteria through `V1SceneView` (approach, close, lift, place in
the target, release); `in_target` from the bin's interior box. Twist drives the roll under the
torque limit and reports `slip_deg`; insert reports lateral force and `jam`; hook pull reports
`slip_mm`. Controllers aim through `StepContext.observe_site` only.

**Gate (design S2):** each primitive succeeds with at least one reference effector and fails on a
deliberate wrong one; deterministic; at most 1 mm penetration on successful holds; fidelity tier
labelled. Fallback: a primitive that fails its fidelity gate after fix attempts is marked
"not simulated yet" and the hero tasks run with the steps that pass.

### 7.3 Track S3: library ingest and catalog

**Owns:** `src/rtt/studio/library/` (item schema, ingest, licence gate, hashing, simulation proxies,
mount metadata, capability tags, ISO 9409-1 flange adapter generator), `assets/library/**` (green
items with their `SOURCE.json` records), `scripts/ingest_library.py`, `web/app/library/**`,
`web/components/library/**`, `web/public/data/library/**`, `web/tests/library.test.ts`,
`tests/test_studio_library.py`, `docs/LIBRARY.md`.

**Builds:** the ingest pipeline: source URL and commit or version; licence gate (allowed SPDX list;
amber excluded and listed with its reason, founder ruling; red excluded); files hashed; simulation
model (existing MJCF or URDF, an automatic CoACD proxy for rigid items, or "visual only" for
articulated items without a model); mount metadata (ISO 9409 pattern, offsets); specs with sources
(stroke, force, mass, actuation); capability tags per primitive. Library items become
`EffectorSpec`s through a registered effector family or an existing one, and the excluded list
is published with reasons. The catalog page: search and filter by capability, licence and mount,
3D viewer, specs, licence and attribution, download (the redistributed file where allowed,
otherwise an upstream link).

**Gate (design S3):** every item re-hashes to its source; licence gate passes; sim proxy or
"visual only" status explicit; page builds.

### 7.4 Track S4: Task Engineer and intake UI

**Owns:** `src/rtt/studio/engineer/` (prompts, the forced `submit_task_plan` call through the
existing LLM client, validators on top of `TaskSpec`, one repair round, refusals, the
`upload:` part resolver via `rtt.studio.templates.part.register_part_resolver`), recorded
responses under `src/rtt/studio/engineer/recorded/`, `tests/test_studio_engineer.py`,
`web/app/studio/new/**`, `web/components/studio/intake/**`.

**Builds:** Nemotron 3 Super with `submit_task_plan_tool()`; parse with `task_spec_from_tool_args`
and feed its listed problems to the repair prompt; unsupported primitives map to the nearest one
with a note, deformables go to `unsupported`, a whole unsupported use case is a `refusal` with a
reason. Intake: text plus an optional STL (photo intake is a stretch item after S7, founder ruling);
the user edits the step list before anything runs. The agent tool `plan_task` is a thin wrapper S7
wires.

**Gate (design S4):** 3 hero descriptions and 5 extra test descriptions produce valid TaskSpecs (at
least 90 % valid after one repair); unsupported tasks are refused with a reason.

### 7.5 Track S5: combination search, full-task runner, recommender, build pack

**Owns:** `src/rtt/studio/search.py` (capability pruning, per-step screening, the success matrix,
combination search over single, multi-head and tool-change setups, bracket geometry search within
`HeadMount` bounds), `src/rtt/studio/runner.py` (full-task runs of the top 3 to 5 setups on the
condition grid, on top of `rtt.studio.episode`), `src/rtt/studio/recommend.py`,
`src/rtt/studio/buildpack/` (bracket and changer-plate STL/STEP generator, full-setup MJCF export,
bill of materials, assembly notes, certificate HTML, `BuildPackManifest`),
`tests/test_studio_search.py`, `tests/test_studio_buildpack.py`.

**Builds:** ranking by `SetupResult.rank_key` with the simplicity rule within a small margin, as in
the v1 tournament; the best single off-the-shelf effector always reported next to the
recommendation; `assert_fair` before any ranking; reference-controller results refused as
evidence. The functions behind the agent tools `run_task_tournament` and `export_build_pack`.

**Gate (design S5):** the 3 hero tasks run end to end locally; build pack files watertight,
licences complete.

### 7.6 Track S6: Nebius fan-out and the use-case benchmark v1

**Owns:** `src/rtt/studio/cloud.py` (job specs for per-step and per-candidate shards on the
existing `rtt.cloud` jobs, cost ledger entries), `configs/usecase_v1.yaml`, `docs/USECASE_V1.md`,
`jobs/studio/**`, `tests/test_studio_cloud.py`.

**Builds:** fan-out of S5's screening and finalist runs as Nebius Serverless Jobs (uk-south2 CPU,
H100 as fallback), with job ids and dollars on every result, inside the existing caps. The
use-case benchmark v1, lightly pre-registered (tag `usecase-v1` before any run, founder ruling):
per hero use case, across N placements (`StudioCondition.aim_offsets_mm`) and seeds, compare
(a) the best single off-the-shelf library effector, (b) the Studio's recommended setup, (c) the
Studio restricted to library items. Metric: full-task completion rate with per-step breakdowns
and fidelity tiers per step. Results are published whatever they show.

**Gate (design S6):** jobs complete; results published.

### 7.7 Track S7: Studio UI, recorded hero sessions, agent tools

**Owns:** `web/app/studio/**` (except S4's `new/`), `web/components/studio/**` (except
`intake/`), `web/public/data/studio/**`, `web/tests/studio.test.ts`, `web/scripts/smoke_studio.mjs`,
`src/rtt/agent/studio_tool.py` and the Studio entries in `src/rtt/agent/tools.py`.

**Builds:** the Studio pages on the public site: use case, editable steps, candidates, combination
results with the best single baseline, the recommendation with per-step fidelity and caveats, the
build pack download; recorded sessions of the three hero use cases (H1 warehouse returns and
kitting, H2 pipe coupling make-up, H3 equipment cabinet maintenance); the Grasp Engineer gains
`plan_task`, `run_task_tournament` and `export_build_pack`, with the existing passcode and dollar
caps.

**Gate (design S7):** Playwright click-through at desktop and 375 px; leak check clean.

### 7.8 Order and cuts

S2, S3 and S4 start when this contract is merged; S5 builds against the reference controllers and
S1's `part` and `bin` templates until S2 lands; S6 starts when S5's runner exists; S7 starts with
recorded fixtures. The video and submission (M12, M13) must not be squeezed: M12 starts no later
than S7. If time runs short, cut in this order: photo intake; the benchmark pre-registration (run
descriptively instead); H3; the multi-tool head (keep single plus tool change).

## 8. Open questions

1. **Perception rule.** S1 makes controllers aim through `observe_site` (true pose plus a hidden
   per-object offset), so placement errors stay unknown, as in v1. Should S2's insert and hook
   pull be allowed contact-based search (force feedback within a step), which v1 allowed only for
   effector controls? Default: yes, through the effector and wrist commands only.
2. **Presentation as motion.** Presenting a multi-tool head is a fixed tilt per segment plus
   `present_s`. If the swing of the idle head through the work matters (H3's cabinet), S5 could
   add a swept-volume check before accepting a multi-head setup.
3. **Wrist numbers.** Roll torque 15 N m, 180 deg/s, a 6 s tool change and a 1.5 s presentation
   are [assumed]. They are equal for every candidate, but the cycle-time ranking depends on the
   change times; the founder may want them sourced from a target arm and changer.
4. **Which part a family sees.** A tool is built against the first part it picks; a tool used only
   on articulated objects is built against a 10 mm probe cube (CR-S5-1; it was the library
   `block`, which refused a 60 mm hook gripper as too short). S5 checks fit against every object
   the tool touches.
5. **Gravity compensation.** As in v1 the arm carries the tool's weight, so the idle head's mass
   shows only through inertia, not as payload. S5 may add an arm payload limit to complexity.
