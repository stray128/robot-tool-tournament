# Effector layer (product mode, not pre-registered)

**Product mode. Not part of the pre-registered experiment.** The pre-registered parallel-jaw
finger experiment (`prereg-v1`, S1 primary, R = 32) stays exactly as frozen. This layer is
built beside it. For one part it tries every end-effector family, scores each with the same
physics, and recommends one with a robustness certificate. Every record, certificate, replay,
page and agent answer it produces carries the label
"Product mode: not part of the pre-registered experiment"
(`rtt.effectors.PRODUCT_MODE_LABEL`, equal to the Grasp Engineer's label).

This file is documentation. The site never renders it. It names upstream projects in the
licence audit (section 9) because the audit needs exact sources; on screen and in the video
every effector has a generic name (section 8).

Contract version: `rtt-effectors/1`. Code: `src/rtt/effectors/`. Contract tests:
`tests/test_effectors_contract.py`.

## Contents

1. [What is frozen, what is new](#1-what-is-frozen-what-is-new)
2. [The Effector abstraction](#2-the-effector-abstraction)
3. [How families plug into the runner and the certificate](#3-how-families-plug-into-the-runner-and-the-certificate)
4. [The per-part tournament](#4-the-per-part-tournament)
5. [The recommendation](#5-the-recommendation)
6. [Web: lanes and pages](#6-web-lanes-and-pages)
7. [Agent tool](#7-agent-tool)
8. [Labels and brand rules](#8-labels-and-brand-rules)
9. [Licence audit of catalog models](#9-licence-audit-of-catalog-models)
10. [Fidelity caveats](#10-fidelity-caveats)
11. [Build plan: four parallel tracks](#11-build-plan-four-parallel-tracks)
12. [Risks](#12-risks)

## 1. What is frozen, what is new

| Area | Status | Rule |
|---|---|---|
| `src/rtt/sim/*.py`, `src/rtt/design/freeform/*.py`, `src/rtt/parts/*.py`, `src/rtt/search/**`, `src/rtt/analysis/*.py`, `configs/task_v2.yaml`, `uv.lock` | frozen by PREREG A5 | never edited. `test_prereg_frozen_code_and_files_are_unchanged` recomputes the C-locale listings and file hashes of PREREG A5 and fails on any change. No new dependencies (`uv.lock` is pinned). |
| `src/rtt/effectors/{__init__,types,registry,scene,runner,certificate,primitives,parallel_jaw}.py`, `tests/test_effectors_contract.py`, `tests/effector_toy.py`, this file | the frozen effector contract | only the contract owner changes them, through a contract change request (section 11.6). `test_contract_files_are_frozen` pins their hashes. |
| everything else under `src/rtt/effectors/`, `assets/effectors/`, the web effector pages | built by the four tracks | exclusive file ownership (section 11). |

The layer reads the frozen code; it never changes it. It imports these private helpers of
the hash-pinned simulator, which therefore cannot change under it:
`rtt.sim.scene._base_spec`, `_add_object`, `_add_position_actuator`;
`rtt.sim.controller._plan`, `_check_reach`; `rtt.sim.replay._geom_entry`;
`rtt.sim.certify._tolerance_by`.

The layer runs only on the six public training parts (`train_v2`) and on parts users
upload. It never touches the held-out suites or the vault (a contract test scans the package).

## 2. The Effector abstraction

```
EffectorSpec ──get_family(spec.family)──▶ EffectorFamilyImpl
   │                                        │
   │   native families                      │   reference family (parallel_jaw)
   ▼                                        ▼
build_effector_scene(impl, spec, part, task)    pair_for(spec, part, task)  (frozen generator)
   core world + wrist (frozen code)              run_episode_v2(...)         (frozen episode)
   impl.build(ctx) adds the effector             from_v2(...)                (field mapping)
   contract checks, mass properties
   ▼                                        │
run_native_episode: frozen controller phases,│
 impl.primitive → effector controls,         │
 frozen PhysicsChecks + LiftTracker,         │
 classify_v2, registration via assess(),     │
 impl.criterion → engaged                    │
   ▼                                        ▼
            EffectorEpisodeResult (one shape for every family)
                         ▼
     certify_effector (frozen battery, 18 tests)  ──▶ tournament ──▶ recommendation
                         ▼                                    ▼
                 replay v2 lanes (web)                  agent tool, pages
```

### 2.1 Families

| `EffectorFamily` | Module | Fidelity | Primitives | Engagement test | Registration reference | Track |
|---|---|---|---|---|---|---|
| `parallel_jaw` | `rtt.effectors.parallel_jaw` | `core` | `jaw` | the jaw's seat (`jam_on_close`) | analytic (jaw centre line) | contract (done) |
| `catalog_hand` | `rtt.effectors.catalog_hand` | `rigid` | `power`, `pinch` | contacts (`few_contacts`, `no_opposition`) | nominal episode | B1 |
| `catalog_gripper` | `rtt.effectors.catalog_gripper` | `rigid` | `pinch` (vendored 2-finger), `power`, `pinch` (procedural adaptive, 2 and 3 fingers) | contacts | nominal episode | B1 |
| `soft_arm` | `rtt.effectors.soft_arm` | `approximation` | `wrap` | wrap (`no_wrap`) | nominal episode | B2 |
| `suction` | `rtt.effectors.suction` | `approximation` | `seal` | seal (`no_seal`) | nominal episode | B2 |
| `designed` | `rtt.effectors.designed` | `rigid` | `power`, `pinch` | contacts | nominal episode | B3 |

A family module registers itself on import (`register_family(Impl())`). `get_family`
imports `rtt.effectors.<family value>` on first use, so a spawned worker finds every
family without a shared list. A family module may be a package (`catalog_hand/__init__.py`).

### 2.2 `EffectorSpec`

One concrete effector (`rtt.effectors.types.EffectorSpec`):

* `family`, `model` (family-local slug, never a brand: catalog models use generic keys such
  as `hand-4f-16dof`), `display_name` (on-screen, brand-free, at most 60 characters,
  checked against `BRAND_TOKENS`), `primitive` (slug), `params` (JSON data, stored
  canonically and read-only; integral numbers become ints so `2.0` and `2` hash alike),
  `recipe` (a `MorphologyRecipe` for designed effectors), `origin` (`reference`, `catalog`,
  `preset`, `nemotron`, `random`, `manual`).
* `effector_id`: sha256 of the canonical form (everything but the display name).
* `group`: the tournament budget group, from origin then family (`reference`, `catalog`,
  `soft_arm`, `suction`, `designed` for every Nemotron recipe whatever builds it, `random`
  for the equal-budget random control).
* Specs pickle (worker processes) and their params cannot be mutated.

### 2.3 Building into the shared scene

`build_effector_scene(impl, spec, part, task)` builds everything outside the effector with
the frozen code, then calls `impl.build(ctx)`:

* **World and part:** the task's solver options (1 ms, implicitfast, elliptic cone,
  impratio 10, 100 iterations), contact defaults, lights, table, floor. The part is a free
  body whose geoms have `priority = 1`, so the part's jittered friction (mu 0.3 +-10 %) and
  contact parameters govern every contact it makes, whatever the effector's own defaults.
  No family can buy grip with stickier pads.
* **Wrist:** the core floating wrist (`palm`): slides x, y, z and a yaw hinge, gravity
  compensated, position servos with the core gains and force limits (x/y 100 N, z 150 N,
  yaw 15 N m). Its geometry is a 63 mm, 24 mm thick flange of the core palm's mass
  (0.4 kg), so a cup or a tentacle is not judged with a parallel gripper's housing on top.
  The flange underside is the **mount plane**, wrist z = -12 mm, the plane of the jaw's
  finger roots (`MOUNT_Z`). It is a top-down, 4-DoF wrist for every family.
* **`BuildContext`:** `spec` (the `MjSpec`), `mount` (the wrist body), `mount_z`,
  `effector`, `part`, `task`, `repo_root`, `prefix = "eff_"`.
* **What a family may do:** add bodies, joints, geoms, sites, tendons, equalities,
  actuators, meshes and materials under `mount`, all named `eff_...` (unnamed geoms are
  allowed). Catalog MJCF is attached with `MjSpec.from_file(...)` and
  `mount.add_frame(pos=..., quat=...).attach_body(child.body(...), "eff_", "")`, which
  prefixes every attached name; drop the child's keyframes and lights first.
* **What it may not do** (checked, `ContractError`, so fairness is enforced by code):
  * leave a new element without the prefix, mount a body anywhere but under the wrist,
    remove scene elements, or declare actuators other than exactly the ones it added, in
    compile order;
  * change the solver options, the compiler settings or the main default geom and joint
    classes (they would change the part's contacts too);
  * act outside itself: an actuator on a wrist joint, an equality that welds or connects
    the effector to the part or the world, a tendon that wraps a non-effector site, joint or
    geom, a contact exclude with the table or the part, an explicit contact pair;
  * out-rank or stick to the part by geometry: effector geoms with `priority >= 1`, geom
    `adhesion`, a contact `margin > 0` (MuJoCo uses the larger margin of the two geoms whatever
    their priority, so contacts, and an adhesion actuator's pull, would act across a gap), or a
    surface velocity (suction uses an adhesion *actuator* gated by its seal);
  * apply forces directly: a primitive that writes `xfrc_applied` or `qfrc_applied` stops
    the episode (the runner owns the only external force, the certificate's push).
* **Gravity:** the scene sets `gravcomp = 1` on every effector body (the arm carries the
  tool's weight); the part is never compensated.
* **Measured, not declared:** effector mass, centre of mass and inertia (wrist frame,
  reference pose) are measured from the compiled model and stored in the returned
  `EffectorBuild`. A mass above 5 kg is refused as a unit error.
* **Cannot be built for this part** (an opening narrower than the part, no flat face under
  the cup): raise `InvalidEffector("reason")`. The episode scores `invalid_build` without
  simulating, like an unbuildable jaw design.

`EffectorBuild` (returned by `build`): `bodies` (root first), `actuators` (control order),
`ctrl_open`, `reach` (m from the mount plane to the lowest point in the open pose; the
controller uses it exactly as the jaw's finger length), `tcp` (grasp centre, wrist frame),
`actuation` (declared limits, printed on every card), `sources` (`AssetSource` licence
records), `visuals`, `notes`, `max_opening`.

### 2.4 Actuation limits (fairness)

Every effector runs at its own documented limits, and every limit is printed on its card:

| Family | Limit |
|---|---|
| parallel jaw | 50 N grip on the mean-closure tendon (25 N per jaw), as frozen |
| catalog hands and grippers | the upstream model's own actuator gains and `forcerange`, unmodified (changes would be listed in `SOURCE.json`) |
| suction | vacuum 60 kPa maximum (a typical single-stage ejector); holding force = vacuum x lip area |
| soft arm | cable tension at most 50 N (the jaw's grip force) |
| designed | at most 50 N total closing force across fingers and 1.5 N m per joint |

The wrist, its servos, the phase timings, the lift (150 mm), the hold (2 s) and every
physics check are identical for all families.

### 2.5 Grasp primitive

`GraspPrimitive` (protocol): `name`, `plan(part, placement, task, build) -> PrimitivePlan`
and `command(t, phase, closure, model, data, scene) -> controls`.

* The wrist always follows the frozen phases: SETTLE 0.3 s, DESCEND 1.0 s (minimum jerk),
  CLOSE 1.0 s ramp plus 0.3 s squeeze, LIFT 150 mm in 1.5 s, HOLD. The primitive drives
  only the effector's actuators.
* `PrimitivePlan.tip_height`: m above the table of the effector's lowest point at the grasp
  pose. The runner converts it into wrist heights with the frozen arithmetic
  (`rtt.sim.controller._plan`), including the approach clearance (20 mm above the part's
  top at least). `yaw_offset` turns the wrist relative to the part; placement errors,
  shakes and pushes stay in the part's nominal frame for every family.
* `closure` is the shared close ramp: 0 before CLOSE, rising linearly to 1 over 1.0 s.
* A primitive may read `data` (contacts, joint positions) for closed-loop behaviour such as
  a vacuum switch, but acts only through the returned controls (one finite value per
  actuator, else `ContractError`).
* `SynergyPrimitive` (shared): controls move linearly from `ctrl_open` to `ctrl_closed` on
  the close ramp and stay closed. A catalog hand's power and pinch grasps are two synergies
  of one model; the grip height is a fraction of the part height (one value per model and
  primitive, never tuned per part).

### 2.6 Success predicate

One scorer for every family. Pass/fail is physics:

* **held** (the shared lift-and-hold criterion, frozen code): the part rises 100 mm and
  stays there for 2 s before 12 s, with no physics flag (effector-table force or
  penetration, part speed, ejection during close, off table, wrist pressing on the carried
  part), no MuJoCo state reset (`physics_invalid`), and an effector-part penetration of
  at most 1 mm over the scored part of the episode (`physics_unreliable` otherwise).
* **engaged** (the family's `GraspCriterion.engaged`, at the registration moment, 0.25 s
  after the close ramp ends, before the lift):

  | Family | Engaged when | Sub-flag when not |
  |---|---|---|
  | parallel jaw | seated: `q_seat - q <= 1 mm` (frozen) | `jam_on_close` |
  | hands, grippers, designed | at least 2 effector bodies touch the part with net forces at least 90 deg apart (`ContactCriterion`) | `few_contacts`, `no_opposition` |
  | suction | every lip segment is in contact, the face under the cup centre is part material (no bore), and the vacuum switch is on | `no_seal` |
  | soft arm | at least k segments touch the part and the contacts wrap at least 180 deg around its vertical axis | `no_wrap` |

* **registered** = held and engaged and the part in its reference pose at the registration
  moment, with the frozen tolerances (`rtt.sim.register.assess`): centred within 0.75 mm
  in x and y, yaw within 2 deg (modulo the part's symmetry), tilt within 3 deg, and
  upright within 5 deg when the hold is confirmed.
* **Reference pose** (`RegistrationReference`):
  * `analytic` (the jaws): the wrist-aimed pose, part origin on the wrist axis. This is the
    pre-registered rule.
  * `nominal_episode` (every other family): the pose the same effector produced in its own
    nominal certificate episode (no placement error, no jitter). A cup or a hand does not
    centre a part on the wrist axis; it is registered when it puts the part back in the
    same pose whatever the placement error (repeatability). This is the property a
    downstream placement needs.
* `grasped` (`GraspCriterion.grasped`, at the start of LIFT) only chooses the failure label
  (`missed_grasp` or a later one), as the jaw's squeeze test does. It never decides success.
* Failure modes are the frozen v2 set (`missed_grasp`, `slip_during_lift`,
  `ejected_on_close`, `unstable_hold`, `finger_env_collision`, `not_registered`,
  `physics_invalid`, `physics_unreliable`, `invalid_build`). Engagement failures appear as
  `not_registered` with the family sub-flag.

### 2.7 Results

`EffectorEpisodeResult` carries `success` (registered), `held`, `engaged`, `failure_mode`,
`registration_flags` (frozen sub-flags plus engagement sub-flags), `flags`, `max_lift`,
`hold_time`, `bad_qacc`, `penetration_mm`, `traj_sha` (sha256 of the state at every check
tick), `reg_pos`/`reg_quat_wxyz` (the measured pose in the wrist frame), `reference`,
`metrics`, `effector_mass_kg`, the label, and optionally the replay, qpos and overlays.
`to_record()` flattens it for Parquet and JSON (NaN becomes `null`; metrics prefixed `m_`).

`MorphologyRecipe` (designed effectors): `family` (the generator's family), `kind` (the
generator, e.g. `fingered_hand`, `tentacle`, `suction_cup`), `params` (one value per
`ParamSpec` of that generator), `hypothesis`, `origin` (`nemotron`, `random`, `manual`),
`parents`. `recipe_id` hashes schema, family, kind and params only, so the same recipe from
both arms is one design. `ParamSpec` rows (`float`, `int`, `choice`, `bool` with bounds,
units and a one-sentence description) generate the strict JSON schema of the forced tool
call (`params_json_schema`).

## 3. How families plug into the runner and the certificate

`run_effector_episode(spec, part, placement, seed, task=..., perturbation=..., record=...,
reference=...)` dispatches to the family. The native runner (`run_native_episode`) mirrors
`run_episode_v2` step for step:

1. Same seeded jitter as the core (PCG64 from the seed: part x/y +-0.5 mm, yaw +-1 deg,
   friction +-10 %); the controller aims at the nominal placement and never sees the error.
2. Frozen phases and heights; the family's primitive drives the effector every control tick
   (500 Hz, two 1 ms physics steps).
3. Frozen `PhysicsChecks` and `LiftTracker` every 10 ms; `classify_v2` labels the outcome,
   with the 1 mm held-penetration rule measured on effector-part contacts.
4. Registration at the frozen tick through `assess()`, against the reference; the family's
   criterion gives `engaged`.
5. Disturbances: a shake moves the wrist target along the part's nominal axes; a push acts
   on the part along those axes at its top.
6. The trajectory hash and the replay never change the physics (recording runs a 0.5 s
   tail after the decision, as in the core).

Jobs: `EffectorJob` and `run_effector_jobs(jobs, workers)` (in-process for 1 worker,
otherwise a spawned process pool; results in job order, identical either way). Native
families reuse one compiled scene per (effector, part, task) inside a worker. A long run (the
tournament) may keep one pool open and map `run_effector_job` (the picklable unit) over it.

**Certificate** (`certify_effector(spec, part)`): the frozen battery
(`rtt.sim.certify.plan_tests`, jitter off): the nominal grasp; placement errors 0, +-1,
+-2, +-3 mm along the part's x and y; a 10 mm, 5 Hz (about 1 g) shake along x, y and z
after the lift; a 0.5 N s push at the part's top along x and y: 18 episodes. Each test
reports held and registered; scores are per-category pass rates and their mean; the
placement tolerance per axis is the largest error up to which every test passed. For a
`nominal_episode` family the nominal episode runs first and becomes the reference (19
episodes); if it does not hold, nothing can be registered (`reference = "none"`). For the
reference family the certificate equals `certify_v2` test for test (a contract test checks
it).

## 4. The per-part tournament

Implemented by track B4 (`rtt.effectors.tournament`); the rules are fixed here.

**Inputs:** one part (a training part, or an upload converted with
`rtt.parts.library.part_from_mesh_object`), the requirement (`held` or `registered`), the
groups to run, a seed.

**Conditions, the same for every candidate (common random numbers):**

* **Screen:** the part's four `train_v2` placements (c0 centred; c1 +2 mm across; c2 -3 mm
  across; c3 +1 mm across and +6 deg) x the suite's 2 jitter seeds: 8 episodes per candidate.
  For an uploaded part, `rtt.part.conditions.conditions_for_part`.
* **Certificate:** the 18-test battery above, for the finalists.

**Stages:**

1. **Build check:** `InvalidEffector` candidates are listed with their reason and score 0.
2. **Screen:** every candidate in every group, 8 episodes each.
3. **Finalists:** the top 2 per group by screen success for the requirement (ties: held
   rate, then fewer actuators, then lower mass, then `effector_id`).
4. **Certificate:** each finalist (2 per group).
5. **Recommendation** (section 5).

**Budget per group, equal for every group:** at most 96 screen episodes (12 candidates x 8)
and 2 certificates. A group with more candidates samples them (fixed seed); a group with
fewer uses fewer. The designed group's candidates are Nemotron proposals in 2 rounds of 4
(the second round sees the first round's physics feedback). The random control is matched to
them (integration decision, 2026-10-17): after the rounds it draws, from the spaces Nemotron
was offered and with the same validators, exactly as many buildable recipes as Nemotron
designs were screened on that part, and screens them. Without the Nemotron rounds
(`design = false`) it draws 8 recipes from the same spaces.

**Fairness rules (asserted by B4's tests):** same part, placements, seeds and certificate
battery for all; same wrist, servo gains, force and torque limits, phases, lift and hold; the
part's friction governs every contact; every family at its own declared actuation limits
(section 2.4); no per-part tuning of catalog synergies or presets; one scorer.

**Determinism:** the tournament is a pure function of (part, groups, requirement, seed, code,
the cached Nemotron responses). Re-running it gives the same records, the same `traj_sha`
values and the same recommendation.

**Cost:** about 6 groups x (96 + 2 x 19) = 804 episodes per part, at about 1-2 CPU-s each:
roughly 15-25 CPU-minutes per part, 2-3 CPU-hours for all six training parts, about 10
minutes on a 16-core laptop. CPU MuJoCo 3.15.0 is the executor of record. Nemotron: 2 forced
calls per part, each at most about $0.01 (an 8k-token prompt plus 8,192 output tokens at
$0.30 and $0.90 per million tokens).

**Outputs** (`runs/effectors/<part>/`, git-ignored): `records.parquet` (every
`to_record()`), `certificates/*.json`, `tournament.json` (the stages, budgets, finalists),
`recommendation.json`.

## 5. The recommendation

Per requirement R (`held` and `registered` are both always computed; the headline uses the
requested one, `registered` by default, the experiment's definition of a good grasp):

1. **Eligible:** certified, the nominal test passes R, no `physics_invalid` episode and no
   `physics_unreliable` nominal test.
2. **Rank:** certificate score for R; then the smaller of the two placement tolerances for R;
   then screen success for R.
3. **Simplicity rule:** when two candidates differ by at most 0.05 in certificate score and
   have equal tolerances, prefer fewer actuators, then lower effector mass, then higher
   fidelity (core, rigid, approximation).
4. **Fidelity rule:** if the winner is an `approximation` (suction, soft arm), the
   recommendation also names the best `core` or `rigid` candidate as the runner-up, and the
   card says "approximation: verify on hardware".
5. **No winner:** if nothing is eligible, the recommendation says so and lists the closest
   candidates with their failure modes. It never forces a pick.

`recommendation.json` (shape fixed here; B4 writes it):

```json
{
  "contract": "rtt-effectors/1",
  "label": "Product mode: not part of the pre-registered experiment",
  "part_id": "block",
  "requirement": "registered",
  "winner": {"effector": {}, "certificate": {}, "screen": {"n": 8, "held": 0.875, "registered": 0.75}, "why": "one plain sentence"},
  "runner_up_rigid": null,
  "alternatives": {"held": {"effector_id": "...", "display_name": "..."}},
  "ranking": [{"effector_id": "...", "display_name": "...", "family": "...", "group": "...", "fidelity": "...",
               "certified": true, "score": 0.9, "tolerance_mm": {"x": 3.0, "y": 2.0},
               "screen_rate": 0.875, "n_actuators": 2, "mass_kg": 0.31, "actuation": {}}],
  "by_group": {"reference": "...", "catalog": "...", "soft_arm": "...", "suction": "...", "designed": "...", "random": "..."},
  "budget": {"reference": {"screen_episodes": 24, "certificates": 2}},
  "conditions": {"suite": "train_v2", "condition_ids": ["..."], "certificate_tests": 18},
  "provenance": {"code_version": "...", "task_digest": "...", "engine": "MuJoCo 3.15.0", "created_utc": "..."}
}
```

## 6. Web: lanes and pages

**Lanes.** One replay per lane, version 2 header (`src/rtt/sim/replay_v2.schema.json`,
unchanged), written from the runner's recording:

* `EffectorReplayRecorder` writes a version 1 header; lift it with
  `rtt.sim.replay.upgrade_header(v1, visuals={}, assets={}, meta={"geometry_id": None},
  provenance=provenance(run_id=..., code_version=..., product_mode=True))`. `check_header`
  and the JSON schema must pass (a contract test does this for the toy family).
* Bodies: the wrist, every effector body that has a collision geom, the part. Each body is
  drawn from its collision geoms (what the physics saw); spheres become zero-length
  capsules, ellipsoids boxes. GLB visuals for effector links are out of scope for this
  round: the schema's asset roles cannot name them without editing the frozen schema.
* `meta.design_space = "effector-<family>"`, `meta.design_id = effector_id`,
  `meta.effector = {family, model, display_name, primitive, origin, fidelity, label}`;
  `provenance.product_mode = true`.
* Overlays: `offset_mm` (from the reference), `hold_s`, `n_contacts`; `seat_short_mm` is
  NaN for every family but the jaw.
* **Payload:** at most 300 KB per lane as served (header, poses, overlays and any GLB it
  alone uses), the existing hero budget (`HERO_PAYLOAD_BYTES`). Mesh collision geoms carry
  their vertices inline, so catalog models favour primitive collision geoms; decimate or
  drop to 20 fps if needed.
* Every lane of one part plays the same condition on one clock (default: c2, the part 3 mm
  off centre, first seed), one lane per group's best candidate, with the recommendation
  first.

**Pages** (B4): `/effectors/` (all parts, the winner per requirement, the product-mode
banner) and `/effectors/<part>/` (lanes, certificate per finalist, ranking table, budget,
fidelity badges, caveats). Data in `web/public/data/effectors/`. The credits page lists every
vendored model from its `SOURCE.json` records.

## 7. Agent tool

B4 adds one tool to the Grasp Engineer: `recommend_effector(part_id, requirement, groups,
design)`. It starts a tournament job (local CPU) and returns a job id; `poll_job` and
`fetch_results` return the recommendation (the existing job runner). Caps as for every
tool: it counts toward the 12 tool calls, at most 2 concurrent jobs, `design = true` (the
Nemotron morphology proposals) needs the passcode and reserves its worst case ($0.05: 2 rounds
of a call and a repair) against the session and day caps, within a per-session design cap of
$0.10. The API service gives live (passcode) sessions the designer, its client capped at the
reservation; offline sessions stay designer-free. The verdict is computed by code; the model
only explains it. Results carry the product-mode label.

## 8. Labels and brand rules

* **Label:** every artifact carries `PRODUCT_MODE_LABEL`. Pages show the banner
  "Product mode: not part of the pre-registered experiment" above any effector result.
  Effector results never appear on the benchmark, results or PREREG pages.
* **Brands:** never on screen or in the video. `display_name` is checked against
  `BRAND_TOKENS` (whole words, any case). On screen a catalog model is described by its
  form: "Four-finger 16-joint hand", "Three-finger 12-joint hand", "Five-finger 22-joint
  hand", "Two-finger parallel gripper, fin-ray pads", "Two-finger parallel gripper, rigid
  pads", "Two-finger adaptive gripper", "Three-finger adaptive gripper".
* **Private leak list:** some makers' names are private leak patterns. `scripts/leak_check.sh`
  refuses them in every file, credits included, so those models are not used at all and the
  names are not in `BRAND_TOKENS` either.
* **Where brands may appear:** README, `NOTICE`, `docs/ATTRIBUTION.md`, the credits page,
  this file's licence audit, vendored provenance under `assets/effectors/` (each model's
  `SOURCE.json` and upstream `LICENSE`), `src/rtt/effectors/types.py` (the token list) and
  tests. `test_brand_tokens_appear_only_in_credit_files` scans every tracked and untracked
  file, including `web/public/data/`, so a lane header that leaks a brand fails the suite.
  Source URLs are ignored by the scan (a pinned URL is provenance, not an on-screen credit,
  the rule `tests/test_gvr.py` already applies to the gripper).
  Vendored directories use generic model keys (`assets/effectors/catalog/hand-4f-16dof/`);
  attached element names are prefixed and must not carry brand words.

## 9. Licence audit of catalog models

Audited 2026-10-17 with the GitHub API (`gh api`) against MuJoCo Menagerie
`https://github.com/google-deepmind/mujoco_menagerie` at commit
`62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35` (main, 2026-10-15). Hashes are sha256 of the
files at that commit. Status: **green** (permissive end to end), **amber** (permissive as
published, with a caveat: vendor only with founder sign-off), **red** (do not vendor).

| Candidate (screen name) | Menagerie dir (last commit) | Model file, sha256 | `LICENSE` sha256 | Licence | Upstream chain | Status |
|---|---|---|---|---|---|---|
| Four-finger 16-joint hand A | `wonik_allegro` (`71f066ad`) | `right_hand.xml` `cec3efe34be49c8220c23afdae0cfacbce10164a5bc7c34075b6668ee0a862a6` | `09c187439ecece8b00330e71eb988d3d30f29373c580dc31ab54200d0aa4deb4` | BSD-2-Clause (SimLab 2016) | URDF from `https://github.com/simlabrobotics/allegro_hand_ros` (commit `296ef6bae4b535ffb91420237cdc8a87f6e9f6df`): BSD declared in `stack.xml` and `manifest.xml`, no LICENSE file upstream | green |
| Five-finger 20-actuator hand | `shadow_hand` (`71f066ad`) | `right_hand.xml` `ac14d09dd12b62c3ca62233d0ac646101cc718146dda76e26d0b993a91abf399` | `d42cae9a0f292b7a713597ba26e1e1e23d50df906c488f0d36fd3538af200bc1` | Apache-2.0 | README: URDF and assets "provided directly by" the maker under Apache-2.0; `https://github.com/shadow-robot/sr_common` is BSD-3-Clause on GitHub (a third-party table calls it GPL-3.0; the direct grant governs the Menagerie copy) | green |
| Three-finger 12-joint hand | `shadow_dexee` (`71f066ad`) | `shadow_dexee.xml` `7d7ad518b01451f58d3e5f3e59e58719741efead2d7730c9cf800f7d766f267a` | `d42cae9a0f292b7a713597ba26e1e1e23d50df906c488f0d36fd3538af200bc1` | Apache-2.0 | provided by the maker under Apache-2.0. Technical risk: its 12 actuators are `plugin` actuators | green |
| Five-finger 22-joint hand | `sharpa_wave` (`c1a4eeb8`, 2026-08-04) | `right_hand.xml` `22208af2ee00bcb62a197dbba8e5ed6577ec41f30ba1fa13fdee375a299ab27e` | `8d3d877d33afe58c5146e4304db9cc01e7550f843b025a461ea2804e90a3c6d3` | Apache-2.0 | `https://github.com/sharpa-robotics/sharpa-urdf-usd-xml` commit `6eea427eb24189519f32b9f21674cd534d3f973c`, `LICENSE.txt` Apache-2.0 | green |
| Four-finger 16-joint hand B | `leap_hand` (`71f066ad`) | `right_hand.xml` `9d8f25c38bdb60617f378c07b7e71f97306471353c81bf281eb9943f3f2340d8` | `5a91c3d709c2ee3a71f166b46e3208cebd83c0c7a870746c729b31a01f156984` | MIT (Ananye Agarwal 2023) | URDF from `https://github.com/dexsuite/dex-urdf` (MIT, commit `2ee2f70cfd549857f56ffe13c61fa939f4cb50ba`) from `https://github.com/leap-hand/LEAP_Hand_Sim` (MIT); but `https://github.com/leap-hand/LEAP_Hand_API` is CC BY-NC 4.0 and the meshes derive from the maker's CAD | amber |
| Five-finger tendon hand | `tetheria_aero_hand_open` (`71f066ad`) | `right_hand.xml` `f9e136acec1c5e51265bd2d6f21f346449dc7506bb293ee5ebf698d16d4e9692` | `d566efb455e2699e94cb728d00084f911ce4548c61049129ba81121f0fc44bc4` | Apache-2.0 in Menagerie | upstream `https://github.com/TetherIA/aero-hand-open` `LICENSE.md`: hardware files (CAD, STEP, STL) are CC BY-NC-SA 4.0; the meshes derive from them | **red** |
| Two-finger adaptive linkage gripper | withheld | | | BSD-2-Clause in Menagerie | the maker's name is on the project's private leak list (`scripts/leak_check.sh`), and the model's paths, files and credit lines carry it | **excluded** (it could not be credited in this repository); build an adaptive gripper procedurally instead |
| Two-finger parallel gripper, fin-ray pads | `umi_gripper` (`71f066ad`) | `umi_gripper.xml` `12b6bbd5e94758164141d54417cad9cde6acdc073311c7c9657dca1ae3fde6ec` | `304e99a48271e10820f71a1b86ef395c06c100d7f68dd1d308790d1064fc4d41` | MIT (Columbia AI and Robotics Lab 2023) | the project's published CAD | green |
| Two-finger parallel gripper, rigid pads A | `i2rt_yam` (`f054586a`) | `yam.xml` `5ae901629db944abd3e6cbb0eaf28ea78bbda3483c840d6db1167e999cc0348c` (arm and gripper; vendor the gripper subtree) | `dc46c3a7dc48221aa783e634d15db0f816ba83ba5cdb77bbaa91525c61c7552f` | MIT (i2rt robotics 2025) | URDF from `https://github.com/i2rt-robotics/i2rt` (MIT) | green |
| Two-finger parallel gripper, rigid pads B | `agilex_piper` (`71f066ad`) | `piper.xml` `a7b5b5d3b2a68d5c553b2ee9665d54a422bd8bf1fa6f3251bc11834993d37098` (arm and gripper; vendor the gripper subtree) | `e32d78fd3761052ab0b2bb42fef8a7a9ad3f298d315d305333c349d2aff1f134` | MIT (RosenYin 2024) | `https://github.com/agilexrobotics/Piper_ros` (MIT) | green |
| Two-finger parallel gripper, rigid pads (an arm maker's hand) | withheld | | | Apache-2.0 in Menagerie | the maker's name is on the private leak list | **excluded** |
| Open parallel gripper (the core experiment's) | `robotstudio_so101` (`3abca2e3`) | `so101.xml` `982c02c027ec0c62d3459d0a6eb00dfb7dc13a451992e72ce99addc71f209805` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | Apache-2.0 | already represented by `parallel_jaw` | green, not needed |
| Three-finger adaptive gripper | not in Menagerie | the only open model found is a URDF by the excluded linkage gripper's maker | | | | build it procedurally (below) |

Other permissive grippers seen (not needed): `aloha` (BSD-3-Clause, `aloha.xml`
`68430b29719bda1b75e63f540953f81991bc3fd136bdf0a43bbe3e04393b78d3`, `LICENSE`
`21868447315ac2b10e592d08958c1035b19a32cf39e9165b23033f68390098cb`) and `trs_so_arm100`
(Apache-2.0).

Two candidates (an adaptive linkage gripper and an arm maker's parallel hand, both permissive
in Menagerie) are withheld: their makers' names match the project's private leak patterns,
which `scripts/leak_check.sh` refuses in every file, so they cannot be vendored and credited
here. The full audit rows for them are kept outside the repository.

**Decision for B1:** vendor the green set: the 16-joint hand A, the 12-joint three-finger
hand (if its plugin actuators load in the pinned MuJoCo, else the 20-actuator hand), the
22-joint hand, the fin-ray gripper and the rigid-pad gripper A (B as the spare). Build the
adaptive grippers procedurally, with two and with three fingers (our own geometry,
Apache-2.0: two phalanges per finger coupled by a tendon, so one actuator closes each finger
and the distal phalanx wraps when the proximal one is stopped). The amber hand (hand B) only
with the founder's sign-off. Never the red hand or the withheld models.

**Founder decisions at integration (2026-10-17):** the three vendored hands stay in as honest
entrants at their upstream gains (no gain changes); the rigid-pad gripper B (1.4 N at its
published settings) is vendored and credited but not a preset (`in_presets = False`), so the
catalog group is exactly 12 candidates and is never sampled; the rigid-pad gripper A keeps its
upstream gains (4.1 N) and every page that shows it prints "Low grip at published settings"
(`CatalogModel.card_note`). Results: `docs/EFFECTORS_RESULTS.md`.

**Vendoring rules:** files under `assets/effectors/catalog/<generic-model-key>/`, the
upstream `LICENSE` beside them, a `SOURCE.json` with one `AssetSource` per file (path,
sha256 as shipped, upstream URL, commit `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`, upstream
path and sha256, SPDX licence, copyright line, credit line, changes). Vendor the model XML
and the meshes the collision geoms need; visual meshes only if the payload allows. A test
re-hashes every vendored file against its `SOURCE.json`.

## 10. Fidelity caveats

These are stated on every card and in the README section that introduces the layer.

* **Suction cup (approximation).** A compliant lip of rigid segments on a bellows spring; the
  seal is a geometric test (every lip segment in contact, no bore under the cup) gating a
  MuJoCo adhesion actuator capped at vacuum x lip area. Not modelled: air flow, leak rate,
  porosity, vacuum build-up time (the switch is instantaneous), lip material hysteresis,
  peel dynamics beyond the contact distribution. The lip's friction is the part's (the
  priority rule). Treat suction results as a screen, then verify on hardware.
* **Soft arm (approximation, octopus-inspired).** A chain of rigid capsule segments with
  spring-damper joints, curled by a cable (a spatial tendon through sites) with a tension
  limit. Not modelled: continuum mechanics, material non-linearity, suckers, hydrostatic
  stiffening. Segment count sets how smooth the wrap is.
* **Catalog hands and grippers (rigid).** Simplified upstream models ("simplified robot
  description" in the upstream READMEs) with rigid links and no skin compliance, driven by
  scripted open-to-closed synergies: no tactile feedback, no regrasp. Their actuator limits
  are the upstream model's, which can differ from the hardware. Rubber pads would grip
  better than the part's mu 0.3 that governs every contact here, so results are conservative
  for padded hands.
* **Every family.** A top-down 4-DoF wrist (no roll or pitch), open-loop timing, no
  perception (the placement error is unknown to the controller), soft contacts (4 ms
  solref, the 1 mm held-penetration rule), 1 ms timestep. The recommendation is a
  simulation ranking under the stated conditions, not a hardware guarantee.
* **Comparisons.** The jaw's registration reference is analytic; every other family's is
  repeatability against its own nominal episode (section 2.6). Families run at different
  actuation limits (section 2.4). Both are printed with every result.

## 11. Build plan: four parallel tracks

### 11.1 Rules for every track

* Work in your own worktree from `effectors-contracts`:
  `git worktree add -b effectors-<track> /home/ashwith/rtt-worktrees/effectors-<track> effectors-contracts`;
  `PYTHONPATH=<worktree>/src`; the shared venv at `/home/ashwith/robot-tool-tournament/.venv`.
* Edit only the files you own (below). Never edit the frozen contract files, the
  PREREG-frozen code or configs, `pyproject.toml` or `uv.lock`. Never touch Nebius, never
  push. Commit with plain `git commit`, no co-author trailers.
* Import the contract from `rtt.effectors` (`types`, `registry`, `scene`, `runner`,
  `certificate`, `primitives`). Need a contract change? File a contract change request
  (section 11.6); do not work around it.
* Common acceptance gates (all must hold before merge):
  * **C1 scope:** `git diff --name-only effectors-contracts...` lists only owned files.
  * **C2 frozen:** `tests/test_effectors_contract.py` passes unchanged (it pins the PREREG
    listings and the contract file hashes).
  * **C3 determinism:** for every preset, two in-process runs and one spawned-worker run
    of one episode give the same `traj_sha`.
  * **C4 physics validity:** 0 `bad_qacc` (no MuJoCo resets) across every gate episode;
    `physics_unreliable` in at most 5 % of held episodes; no effector-table flag during
    SETTLE or DESCEND in any nominal episode.
  * **C5 labels and brands:** every record and page carries the label; no brand token in
    any display string or lane (`find_brand_tokens` returns nothing; the brand scan test).
  * **C6 runtime:** median episode wall time at most 3 s on one laptop core (5 s for the
    soft arm), measured on the six parts' c0 episodes.
  * **C7 hygiene:** `ruff check` and `ruff format --check` clean; full `pytest` green;
    `scripts/leak_check.sh` clean.

### 11.2 Track B1: catalog hands and grippers

**Owns:** `src/rtt/effectors/catalog_hand.py` (or package), `src/rtt/effectors/catalog_gripper.py`
(or package), `src/rtt/effectors/catalog/` (shared vendoring, attach and synergy helpers),
`assets/effectors/catalog/**`, `scripts/vendor_effectors.py`, `tests/test_effectors_catalog.py`,
and the credit lines in `README.md`, `NOTICE` and `docs/ATTRIBUTION.md` (B1 is the only track
that vendors third-party files).

**Uses:** `EffectorFamilyImpl`, `BuildContext`, `EffectorBuild`, `AssetSource`,
`SynergyPrimitive`, `ContactCriterion`, `InvalidEffector`, `certify_effector`,
`run_effector_jobs`.

**Build:** vendor the green set of section 9 at the pinned commit with `SOURCE.json`;
attach each model palm-down under the mount with prefixed names (drop keyframes, lights,
visual-only geoms the payload cannot afford); generic model keys and display names; two
synergies per hand (`power`: all fingers wrap, thumb opposed; `pinch`: thumb and index tips
meet), one per two-finger gripper (`pinch`), two (`power`, `pinch`) for each procedural
adaptive gripper. One grip-height fraction per model and primitive, chosen on the c0 condition of
all six parts together, never per part. `presets()` returns every model x primitive.

**Acceptance gates:**

* every vendored file re-hashes to its `SOURCE.json` entry; licences in `ALLOWED_LICENSES`;
  no red model; amber only with recorded sign-off;
* every preset compiles in the shared scene with no `ContractError` and the declared
  actuator count; plugin actuators either load or the model is dropped;
* held at nominal (certificate nominal test): each hand, best primitive, on at least 4 of
  the 6 training parts, at least one hand on 5; each gripper on at least 5 of 6;
* the procedural adaptive grippers are underactuated (one actuator per finger drives at
  least 2 joints); the three-finger one engages at least 3 bodies on `pipe_tee` and
  `flanged_bushing`;
* the c2 lane of every preset is at most 300 KB;
* common gates C1 to C7.

**Effort:** about 8 agent-hours (vendoring 1.5 h, attach and naming 2 h, synergies and
heights 3 h, tests and lanes 1.5 h).

### 11.3 Track B2: soft arm and suction

**Owns:** `src/rtt/effectors/soft_arm.py`, `src/rtt/effectors/suction.py`,
`tests/test_effectors_soft_suction.py`.

**Uses:** the same contract types; `object_contacts` for the seal and wrap tests;
`ParamSpec`, `MorphologyRecipe` (both families are designable by B3).

**Build:**

* **Suction:** a cup on a short bellows (a prismatic joint with a spring along the cup
  axis and a small ball-joint tilt compliance), a lip of N rigid segments around the rim,
  an adhesion actuator on the cup body with gain = vacuum x lip area (vacuum at most
  60 kPa), switched on by the primitive only while the seal test holds (a vacuum switch).
  Presets: 3 cup diameters (for example 10, 16, 25 mm). Primitive `seal`: descend until
  the lip compresses, switch on, lift. Criterion: `no_seal` when any lip segment is off
  the face or the ray under the cup centre misses the part (a bore).
* **Soft arm:** at least 12 capsule segments with spring-damper hinges (two bending axes
  or one plane, documented), a cable as a spatial tendon routed through sites on one side,
  a motor or position actuator with tension at most 50 N; the primitive lowers the arm
  beside the part and pulls the cable; the springs straighten it when released. Presets: 3
  (segment count or stiffness). Criterion: `no_wrap` below k contacting segments or a wrap
  angle under 180 deg.
* Both: `param_space(kind)`, `recipe_kinds()` and `spec_from_recipe` (cup diameter, lip
  segments, bellows stiffness; segment count, taper, stiffness, cable offset) for B3.

**Acceptance gates:**

* suction holds at nominal on `block` and `steel_slug`, and reports `no_seal` (not held, not
  engaged) at nominal on `flanged_bushing` (bore), `pipe_tee` (curved run) and
  `hex_standoff` (tapped top hole): 3 of 3 honest refusals;
* suction seal loss drops the part: a unit scene that tilts the face past the lip
  compliance releases it within 0.2 s; the holding force never exceeds vacuum x area;
* soft arm: released cable returns the tip within 5 mm of straight within 1 s; held at
  nominal on at least 2 of 6 parts; `no_wrap` reported where it fails to wrap;
* held penetration at most 1 mm in every held episode of both families (the adhesion must
  not pull the lip into the part);
* `param_space` rows validate and round-trip through `MorphologyRecipe` and
  `spec_from_recipe`;
* common gates C1 to C7 (C6 at 5 s for the soft arm).

**Effort:** about 8 agent-hours (suction 3.5 h, soft arm 3.5 h, tests 1 h).

### 11.4 Track B3: Nemotron morphology designer

**Owns:** `src/rtt/effectors/designed.py` (the `designed` family: fingered morphologies),
`src/rtt/effectors/morphology/` (spaces, generator, validators, proposer, random baseline,
`prompts/`), `tests/test_effectors_designed.py`, `tests/test_effectors_morphology.py`.

**Uses:** `MorphologyRecipe`, `ParamSpec`, `params_json_schema`, `EffectorSpec`,
`SynergyPrimitive`, `ContactCriterion`, `InvalidEffector`, the registry (`param_space` and
`spec_from_recipe` of any designable family, so the proposer can also design B2's cups and
tentacles once they land), `rtt.llm.client.LLMClient` (forced tool call, cache, ledger,
retries) with `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory.

**Interface it provides to B4** (`rtt.effectors.morphology`):

```python
def design_spaces() -> dict[tuple[EffectorFamily, str], tuple[ParamSpec, ...]]
def validate_recipe(recipe: MorphologyRecipe, part: PartSpec, task: TaskV2) -> list[str]  # cheapest first
def random_recipes(part: PartSpec, n: int, seed: int, *, kinds=None) -> list[MorphologyRecipe]
def propose_recipes(part: PartSpec, history: Sequence[DesignOutcome], n: int, *, client: LLMClient,
                    kinds=None) -> ProposalBatch   # valid recipes, rejections with reasons, ledger entries
def spec_for(recipe: MorphologyRecipe) -> EffectorSpec   # get_family(recipe.family).spec_from_recipe
@dataclass(frozen=True)
class DesignOutcome:  # physics feedback shown to Nemotron
    recipe: MorphologyRecipe; held_rate: float; registered_rate: float; failure_modes: dict[str, int]
```

**Build:** the `fingered_hand` generator (finger count 2-5, phalanges per finger 1-3,
phalanx lengths, palm width, finger spread angles, thumb opposition, pad shape and
thickness) as an `MjSpec` subtree with one position actuator per joint (or tendon-coupled
fingers); the `designed` family's synergies (`power`, `pinch`) derived from the recipe;
validators (bounds, self-collision at open and closed poses, opening wider than the part,
reach budget, mass at most 1.5 kg, actuation budget); the forced `submit_morphologies`
tool call (analysis first, then per design a hypothesis and its params, the schema from
`params_json_schema`); a near-duplicate filter in normalised parameter space; the random
control with the same validators and budget. Prompts live in
`src/rtt/effectors/morphology/prompts/`, never in the frozen search prompts directory.

**Acceptance gates:**

* every `ParamSpec` space round-trips: 256 random valid recipes per kind build (or raise
  `InvalidEffector` with a reason) on all six parts with 0 `ContractError`, build time at
  most 200 ms each;
* validators reject 100 % of a fixed adversarial set (out-of-bounds, self-intersecting,
  too narrow, too heavy, over budget), each with a readable reason;
* `recipe_id` is stable across processes and ignores hypothesis and origin;
* the proposer runs from cached responses in tests (zero live calls; a fake transport as in
  `tests/test_llm_client.py`); one live smoke call on one part with a spend cap of $0.05,
  its manifest kept;
* prompts contain no held-out words (the existing guard test pattern) and no brand tokens;
* physics: on the six training parts, the best Nemotron design (2 rounds of 4) is held at
  nominal on at least 4 of 6; the random control's result on the same budget is reported
  next to it, with no superiority claim;
* common gates C1 to C7.

**Effort:** about 10 agent-hours (spaces and generator 4 h, validators 2 h, proposer and
prompts 2.5 h, random and tests 1.5 h).

### 11.5 Track B4: tournament, recommendation, agent tool, web

**Owns:** `src/rtt/effectors/tournament.py`, `src/rtt/effectors/recommend.py`,
`src/rtt/effectors/lanes.py`, `src/rtt/effectors/__main__.py` (CLI), `src/rtt/agent/`
changes for `recommend_effector` (`schemas.py`, `tools.py`, `engineer.py`, a new
`effector_tool.py`; existing agent tests must stay green), `web/app/effectors/**`,
`web/components/effectors/**`, `web/lib/effectors.ts`, `web/public/data/effectors/**`,
`web/public/data/replays/effectors-*/**`, the effector section of `web/app/credits/page.tsx`
(rendered from `SOURCE.json`), `docs/AGENT.md` (the tool row), `tests/test_effectors_tournament.py`,
`tests/test_effectors_lanes.py`, `web/tests/effectors*.test.ts`.

**Uses:** `presets_of`, `get_family`, `run_effector_jobs`, `certify_effector`,
`EffectorEpisodeResult.to_record`, `EffectorReplayRecorder` output, `upgrade_header`,
`check_header`, `provenance`; B3's `rtt.effectors.morphology` interface (stub it with the
contract's types until B3 lands; the reference family and test fakes are enough to build
and test everything).

**Build:** sections 4 to 7: stages, equal budgets, finalists, certificates, the
recommendation rules, `recommendation.json`, lanes, pages, the agent tool, the CLI
(`python -m rtt.effectors tournament --part block --requirement registered --workers 8`).

**Acceptance gates:**

* fairness tests: every candidate sees the identical condition ids and seeds; the wrist
  servo constants and phase timings are identical across families (from the compiled
  models); each group uses at most 96 screen episodes and 2 certificates;
* the tournament runs end to end on all six training parts with every family available at
  merge time and writes the four outputs; a re-run gives identical records, `traj_sha`
  values and recommendations;
* recommendation tests: ranking, the simplicity rule, the fidelity rule (an approximation
  winner always carries a rigid runner-up), the no-winner case;
* agent tool: argument schema validated, counts toward the 12 calls, passcode and spend
  caps for `design = true`, at most 2 concurrent jobs; the session records the code verdict
  and the label;
* web: `pnpm build` (static export) and `vitest` green; every lane at most 300 KB; the brand
  scan finds nothing in `web/public/data/effectors*` and the rendered pages; the product
  mode banner on every effector page;
* common gates C1 to C7.

**Effort:** about 10 agent-hours (tournament and recommendation 4 h, lanes 1.5 h, agent
tool 1.5 h, web 3 h).

### 11.6 Integration and contract changes

* **Order:** B1, B2, B3 and B4 start together. B4 builds against the reference family and
  test fakes; B3's designed family needs no other track; B3's proposer picks up B2's spaces
  through the registry when B2 lands. Merge order: B2 and B1 (families), then B3, then B4
  (it runs the full tournament last).
* **Contract change request:** a track that needs a contract change writes it down (what,
  why, which test proves it) and hands it to the contract owner, who changes the contract on
  `effectors-contracts`, updates the pinned hashes in `test_contract_files_are_frozen`, and
  every track rebases. No track edits a contract file itself.
* **Contract changes after integration** (the integration branch owns the contract now):
  1. `margin` (requested by B2): `_check_compiled` refuses effector geoms with a contact
     margin above 0; `test_scene_refuses_a_family_that_breaks_the_contract["margin"]` proves
     it. No B1 to B3 preset or random recipe used a margin, so no family changed. The
     contract version stays `rtt-effectors/1` (the record shapes are unchanged).
* **Merge gate:** on the integration branch, the full suite, ruff, the leak check and the
  six-part tournament all pass.

## 12. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Catalog hands are sized for human objects; top-down scripted synergies miss 17-60 mm parts | few catalog wins, an empty-looking lane | two primitives per hand, a grip height per model (not per part); report failures honestly; the gate asks for 4 of 6 |
| MJCF attach problems (defaults, keyframes, plugin actuators, mesh paths, contact excludes) | a model cannot be used | B1 gate on compile; drop the model rather than hack the contract; the green list has spares |
| Adhesion and soft contacts push penetration past 1 mm | suction or soft arm fails as `physics_unreliable` | the 1 mm rule is shared and stays; tune lip segments, adhesion gain and solref of the effector side; the gate is 5 % |
| Many-body soft arm and 22-joint hands are slow | the tournament exceeds its budget | runtime gate; fewer segments; scene reuse per worker |
| Licence chains are messier than the published licence (the red and amber rows) | a takedown or a credibility hit | red never vendored, amber only with sign-off, every file pinned by commit and sha256 with its licence text |
| Two permissive candidates are withheld by the private leak list | fewer grippers, no off-the-shelf adaptive linkage gripper | procedural two- and three-finger adaptive grippers (our own geometry); two MIT rigid-pad grippers |
| Brand words leak through upstream element names into lane headers | the brand rule breaks on screen | prefixed and renamed elements; the brand scan test covers `web/public/data` |
| Different registration references and actuation limits invite "unfair" critiques | the recommendation is doubted | both are fixed rules, printed on every card; `held` and `registered` are both reported |
| The product layer is read as part of the pre-registered result | the experiment's credibility | labels everywhere, separate pages, never on the benchmark or PREREG pages |
| Over-claiming: "our designs beat the catalog" from six parts | an unsupported claim | product mode, no statistics, certificate numbers only, the random control shown beside Nemotron |
| A track edits a contract or frozen file to get unblocked | silent drift from the experiment | `test_prereg_frozen_code_and_files_are_unchanged` and `test_contract_files_are_frozen`; the change-request path |
| Nemotron cost or latency spikes, malformed tool calls | the designed group comes up short | forced tool call, strict schema, validators, cached replay in tests, a spend cap per part, the random control fills the group's slots |
| Payload of mesh collision geoms in lanes | slow pages | primitive collision geoms, decimation, 20 fps, the 300 KB gate |
