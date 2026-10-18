# Part-shaped jaws in the Studio (track S5d)

**Product mode. Not part of the pre-registered experiment.** Every record, card and file with text
carries "Product mode: not part of the pre-registered experiment". The site never renders this file.

The pre-registered v2 fingers (the stock blade, the flat pad and the configurator imprint on the
open 84 mm parallel gripper, family `parallel_jaw`) run only inside the frozen v2 episode, and that
is unchanged. S5d gives the same jaws a separate life as **Studio tools**: the gripper with two v2
fingers, built under the Studio's tool mount and built against the object a step touches. Their
numbers are Studio simulation results under S2's controllers, never the experiment's.

| What | Where |
|---|---|
| Studio build path, gripped features, specs, print files | `src/rtt/effectors/part_jaw.py` |
| Candidate pool and pruning | `src/rtt/studio/search.py` (`part_jaw_candidates`, `part_jaw_problem`, `default_candidates(task_spec=...)`) |
| Printed jaws in a build pack | `src/rtt/studio/buildpack/part_jaw.py` (`add_jaw_prints`, `export_part_jaw_pack`) |
| Tests | `tests/test_studio_partjaw.py` |

## Contents

1. [The build](#1-the-build)
2. [Built against the step's object](#2-built-against-the-steps-object)
3. [Search integration](#3-search-integration)
4. [Build pack](#4-build-pack)
5. [Evidence](#5-evidence)
6. [Open issues and change requests](#6-open-issues-and-change-requests)

## 1. The build

A Studio jaw is an `EffectorSpec` of the `parallel_jaw` family whose `params` hold one key,
`studio_part_jaw`: the entrant (`imprint_configurator`, the default; `flat` and `stock` as
baselines) and the target object (name, template kind, gripped feature, params, part reference).
`part_jaw_spec(obj, entrant)` makes one; its model is `studio-<entrant>` and its display name says
which object it is shaped for. Different objects (or sizes) give different effector ids.

Under the mount (wrist z = -12 mm for a single tool; the changer's or the adapter's plane otherwise):

| Element | Model (all `task_v2`, the v2 scene's values) |
|---|---|
| `eff_pj_housing` | the gripper frame, a 47.6 x 128 x 24 mm box with the v2 palm mass (0.4 kg): in the v2 scene the palm is the gripper frame. The Studio wrist keeps its own 0.4 kg (equal for every candidate), so the housing is the tool's |
| `eff_pj_left`, `eff_pj_right` | the generated jaws on closing slides rooted on the housing's underside, `q` 0 (open, 84.8 mm between the innermost faces) to the 42 mm stroke, armature 0.02, damping 2; one convex mesh geom per piece, each jaw with its own mass, centre of mass and inertia |
| `eff_pj_grip` | the mean-closure tendon, driven by a position servo, kp 4000, kv 100, force limit 50 N (25 N per jaw: the sustained value of the gripper's protected servo), control range `open_target` (-2 mm) to the stroke plus 10 mm of overdrive |
| `eff_pj_coupling` | the joint equality that couples the jaws (self-centring) |

The primitive is the v2 grip ramp as a v1 synergy (the grip target moves linearly with the shared
closure), and the jaws' lowest point goes to the pair's designed tip height. The criterion is the
shared contact test. Card actuation: one actuator, 25 N per jaw (50 N on the servo), 84.8 mm
opening, 42 mm stroke. Card notes: the label, "Studio build of the v2 ... not the pre-registered
result", the jaw force, the generator's notes (the imprint depth).

**Fidelity.** The family's fidelity is the v2 fingers' (`core`); the Studio maps it to `rigid`
(`tier_of_effector`), and the candidate record shows `rigid`. A step on a threaded joint or a
latch is an approximation through its template, with the template's caveat.

**Registration.** The registry holds one implementation per family. Importing
`rtt.effectors.part_jaw` (which `rtt.studio.search` does) replaces the frozen reference in the
`parallel_jaw` slot with `StudioPartJaw`, a subclass that inherits every frozen behaviour
unchanged (presets, the delegated v2 episode, `pair_for`, fidelity, analytic registration) and
adds the Studio build path only for specs with `studio_part_jaw`. A pre-registered entrant still
refuses the Studio scene with the frozen message, and a Studio jaw is refused by the v2 episode
("needs params.entrant or params.recipe"): neither can stand in for the other. Spawned worker
processes import `rtt.studio.runner`, not the search, so a Studio jaw spec pickled for
`multiprocessing` restores through `part_jaw._restore_spec`, which imports the module first
(`ForkingPickler.register`); every other spec pickles exactly as before. Without the hook a
spawned worker fails with the frozen `NotImplementedError` (checked with a main module that
imports nothing).

**Licences.** Imprint and flat jaws are our generated geometry (no licence records; the pack's
files are CC0-1.0). The stock blade is the open-source gripper's own part: its build carries the
licence records of the committed blade hulls and the licence text (CC BY 4.0, green), so a pack
links upstream and ships the licence instead of printing it.

## 2. Built against the step's object

The jaw pair is compiled by the frozen generator (`rtt.design.freeform`) against a `PartSpec` of
the **gripped feature** (`gripped_feature`), in the jaws' frame (closing axis = y):

| Object | Gripped feature | How it is made | Turned | Grip height (pad band bottom above the feature's bottom) |
|---|---|---|---|---|
| `part` | the part | `part_spec_of` (exact pieces) | no | the recipe's own (grip fraction 0.5); the pick lowers the jaws to the pair's tip height |
| `threaded_joint` (coupling, nut, cap) | the member | the template's own builder run alone in a scratch scene; the member body's collision geoms read from the compiled model in its frame | +90 deg about z: S2 closes along a flat's normal, the member's +x (`flat` site) | 3 mm (S2's twist and insert: tips 3 mm above the member's bottom) |
| `port_plug` | the plug | as above (cylinders as the circumscribed 64-gon prism, the tip frustum exactly) | no | `rtt.studio.primitives.insert.grip_tip` (the insert's pick: below the tip-over height) |
| `quarter_turn_latch` | the head (wing or slot) | as above (boss and fin, or core and cheeks) | no (the twist closes across the fin, local y) | 2 mm above the panel (S2's twist), at least the generator's 2 mm table clearance |
| `handwheel_valve` | none | | | S2's twist turns a spoked wheel with open fingers in the gaps (nothing is gripped); a solid disc is round, so an imprint gives no hold about its axis |
| lid, drawer, bin | none | | | hooked or pushed, or a destination |

Meshes and boxes are exact; a cylinder becomes the circumscribed 64-gon prism, at most 0.12 % of
its radius outside the simulated surface (inside the imprint's 0.4 mm clearance). A test checks
the coupling feature against the template's own piece function and the nut's turn (its flats face
the jaws).

**Why the grip height.** The v2 generator puts the pad band at the recipe's grip fraction of a part
standing on the table and the root block 2 mm over the part's top (10 mm palm clearance). S2's
twist and insert grip a threaded member with the jaws' lowest point 3 mm above its bottom, which
for H2's 45 mm coupling is 9.5 mm lower than the recipe's band. Built at the recipe's band, the
imprint jaw's root block landed on the coupling's top and the jaws stopped on the flats at 22.4 mm
closure, 8 mm short of the imprint's seat (30.4 mm): the imprint never engaged. So a generated
entrant on a template feature is compiled from the entrant's own `PairSpec` with `tip_height` set
to where the step grips (everything else is the entrant's: pad, depth rule, clearance, width).
This is a design input, not a tuned value; a test checks that the jaws then close into the
imprint, within 1 mm of its seat, at the start of the turn. The stock blade has no design and is
built as it is.

**Which build is refused.** The generator's own validators decide, and their reason is the pruning
reason: on H1 the carton (80 mm across the closing axis) fails the open-clearance rule for the
imprint and the flat pad (3 mm error envelope at full open: -0.6 mm) and the pack (100 mm) fails
the seat range for all three; the default latch wing (5 mm fin) and the default 12 mm plug fail
the seat range for the imprint (the 4 mm configurator depth reaches past the seat); the 20 mm
plug, the slot head, the nut, the cap and every coupling build.

## 3. Search integration

* **Pool.** `default_candidates(task_spec=...)` (what `run_task_tournament` now calls) replaces the
  parallel-jaw presets (the pre-registered entrants) by `part_jaw_candidates(task_spec)`: for
  every object a step touches that has a gripped feature, the imprint (source `designed`, not
  off the shelf: it is shaped for its object) and the flat pad and the stock blade (source
  `preset`, off the shelf). Without a task the pool is as before, and a pre-registered entrant
  passed explicitly is still pruned with "this family runs only inside the frozen v2 episode".
* **Pruning.** Every S5 rule still applies (the build on the probe parts: the generator's
  refusal; the payload; the mechanism; the fit against every object; the scene's part for a tool
  that picks nothing), plus one: a part-shaped jaw serves only the steps on its own object ("its
  jaws are shaped and sized for X; Y gets part-shaped jaws of its own"). Measured on H2's coupling:
  0.41 to 0.45 kg (housing 0.40 kg plus two jaws), one actuator, 84.8 mm opening.

## 4. Build pack

`export_part_jaw_pack(report, out_dir, setup_id=...)` is `export_build_pack` plus
`add_jaw_prints`: for each imprint or flat jaw tool `k` of the packed setup it writes, through the
v2 print pack (`rtt.part.pack.make_print_pack_v2`, the same solids and checks),
`jaws/tool<k>_left.stl` and `_right.stl` (each jaw upright for printing, one watertight solid; the
right jaw is its own part) and `jaws/tool<k>_mount_left.stl` and `_mount_right.stl` (each jaw fused
with its carriage block), plus `jaws/JAWS.txt`, all CC0-1.0, generated; the manifest is rewritten
and the zip rebuilt. The pack's own `cad/tool<k>_reference.stl` (the simulated collision geometry)
and the MJCF (reloaded and checked) are written as for any procedural tool. A stock-blade tool
gets the upstream link and licence text instead (the blade is never printed); its hull geometry
is written inline into the MJCF of every scene that holds it, which `export_build_pack` labels
CC0-1.0 (it reads licences from mesh files only), so `add_jaw_prints` relabels those MJCF entries
CC-BY-4.0 with the blade's brand-free attribution.

## 5. Evidence

Run on `main` b2d0c77 plus this track, 2026-10-18, S2's default controllers, nothing tuned.

### 5.1 H2 tighten (the recorded S4c plan)

`src/rtt/studio/engineer/recorded/gate-2026-10-18-s4c/manifest.json`, case `h2_coupling`, task
`59ba912a5fa8`: insert the 40 mm 12-sided coupling (45 mm long, 1.5 mm pitch, 3 turns to seat,
ramp 0.2 to 8 N m) to the thread start (3.5 to 4.5 mm), then tighten clockwise to a roll torque of
2 to 8 N m (15 N m limit). One tool for both steps, the plan's first three conditions.

| Jaw (on the coupling member) | Condition | Insert | Tighten | Turns | Depth (mm) | Peak roll torque (N m) | Slip (deg) | Penetration (mm) |
|---|---|---|---|---|---|---|---|---|
| configurator imprint (q_seat 30.4 mm, 8 mm deep) | nominal | 4.0 mm | physics_unreliable | 2.34 | 7.17 | 1.67 | 372 | 1.13 |
| | place01 | 4.0 mm | physics_unreliable | 2.43 | 7.34 | 1.86 | 283 | 1.03 |
| | place02 | 4.0 mm | physics_unreliable | 2.47 | 7.58 | 1.63 | 343 | 1.05 |
| flat pad | nominal | 4.0 mm | physics_unreliable | 2.41 | 7.43 | 0.54 | 372 | 1.05 |
| | place01 | 4.0 mm | physics_unreliable | 2.53 | 7.61 | 0.60 | 313 | 1.26 |
| | place02 | 4.0 mm | physics_unreliable | 2.53 | 7.49 | 0.54 | 342 | 1.08 |
| stock blade | nominal | 4.0 mm | physics_unreliable | 3.20 | 5.02 | 1.46 | 41 | 1.74 |
| | place01 | 4.0 mm | physics_unreliable | 3.31 | 5.24 | 1.45 | 41 | 1.75 |
| | place02 | 4.0 mm | physics_unreliable | 3.28 | 5.08 | 1.43 | 33 | 2.06 |

Every jaw inserts the coupling (3/3); no jaw tightens it (0/9): no run reaches 2 N m, and every
run passes 1 mm of penetration. The imprint's peaks (1.6 to 1.9 N m) are three times the flat
pad's, but they come with corner impacts (>1 mm penetration), not with a hold.

### 5.2 Why form closure does not transmit the torque here

1. **The jaws back off.** Diagnostic (not a record): the same plan with the coupling's hinge locked
   by a 50 N m dry friction and its ramp off, asking for 12 N m, so the peak roll torque is what the
   grip transmits. Imprint 0.55, 0.60, 0.55 N m; flat pad 0.55, 0.60, 0.55 N m; stock blade 0.74,
   1.75, 0.63 N m; all slip. The imprint is seated (closure 30.45 mm at the turn start against a
   30.41 mm seat), and both the imprint and the flat pad are pushed open by the same 0.72 mm once
   every 30 deg of turn: a corner of the 12-gon stands 0.7 mm proud of its flats (corner radius
   20.7 mm), and a flat pad on a flat is already that cam. With the same torque and the same
   excursion for both, the imprint's other walls (0.4 mm clear) carry no load before the 25 N jaws
   back off. The measured peaks fit
   `T = 2 F (R_c cos 75 deg + mu r)` = 2 x 25 N x (5.4 + 6.0) mm = 0.57 N m: for a twelve-sided
   member the cam lever (5.4 mm) is no longer than the friction lever (6 mm), so the shape adds
   about as much as friction. By that law 2 N m needs about 88 N per jaw (by friction alone,
   170 N): the gripper's unprotected stall regime, which `task_v2` lists as a robustness point, not
   its 25 N default. Not run (no tuning toward passing).
2. **The thread's friction ramp does not hold the member.** Diagnostic: the coupling alone, put at
   half engagement (ramp 4.1 N m), a steady tightening torque applied to its hinge, the template's
   `passive()` run every 2 ms tick as the runner does. 1 N m and 3 N m (both below the ramp) turn
   it 380 deg and 1295 deg in 2 s; at seat (ramp 8 N m) 6 N m turns it 2342 deg; the motor's sign
   flips on 25 to 74 % of the ticks. The ramp is an explicit velocity-saturated motor on a member with 0.0005 kg m^2 of
   armature: at 4 N m one 2 ms tick changes its speed by 16 rad/s against a 0.3 rad/s saturation, so
   it chatters and a load below the ramp turns the member anyway. This is why every jaw above
   turns the coupling 2.3 to 3.3 turns with peak torques below 2 N m, and why the "2 N m" criterion
   is not a clean test of grip in this physics (S2's template; see section 6).

### 5.3 H1 picks (the recorded S4c plan)

Case `h1_tote`, task `b75435602453`: `pick_carton` (120 x 80 x 60 mm, 300 g) and `pick_pack`
(150 x 100 x 20 mm, 150 g) into the tote.

| Pick | Imprint | Flat pad | Stock blade |
|---|---|---|---|
| carton | refused: open clearance (3 mm envelope at full open: -0.6 mm) | refused: open clearance | builds (seat 2.4 mm); as planned (lid closed) 0/3: lifted 160 mm, not in the tote, 2 runs over 1 mm penetration; with the lid set open (diagnostic only, as S5 did) 3/3 into the tote, 0.13 to 0.17 mm penetration, 50 N |
| pack | refused: too wide for the opening (seat -7.6 mm) | refused | refused |

An 84 mm gripper cannot open around H1's pack, and only the stock blade (which keeps no 3 mm
envelope) opens around the carton.

### 5.4 Odd parts (beyond the hero plans)

Not a hero plan: whether part-shaped fingers help a plain pick of an odd part. Library parts at
(100, 0) mm, pick, lift 100 mm, hold 2 s (criteria: lift at least 80 mm, hold at least 2 s, tilt
at most 10 deg), the first three conditions of each task's grid (placement errors up to 2 mm).

| Part | Imprint | Flat pad | Stock blade |
|---|---|---|---|
| pipe tee (60 x 44 x 29 mm) | 3/3, 0.10 to 0.16 mm | 3/3, 0.19 mm | 3/3, 0.15 to 0.21 mm |
| L bracket (40 x 30 x 40 mm) | 3/3, 0.13 to 0.17 mm | 3/3, 0.13 to 0.21 mm | 3/3, 0.15 to 0.16 mm |
| hex standoff (20 x 17 x 45 mm) | 3/3, 0.08 to 0.17 mm | 3/3, 0.10 to 0.12 mm | 3/3, 0.10 to 0.18 mm |
| flanged bushing (36 x 36 x 35 mm) | 3/3, 0.08 to 0.17 mm | 3/3, 0.10 to 0.12 mm | 0/3, physics_unreliable (1.35 to 1.57 mm, 178 to 381 N) |

(success, deepest jaw penetration). On these lifts the imprint and the flat pad both pass 12/12:
the Studio pick asks for a lift and a hold, which a 25 N squeeze already gives these light parts,
so this test does not separate them (the v2 experiment's registered-grasp and perturbation tests
are where an imprint was meant to differ). The stock blade presses into the bushing (178 to 381 N,
more than 1 mm deep).

### 5.5 Search, pack and determinism

**H2, part jaws only** (`run_task_tournament` with `part_jaw_candidates`, 3 screen and 8 full-run
conditions, 3 worker processes, 158 s): the three jaws on the coupling all screen `insert_coupling`
3/3 and `tighten_coupling` 0/3 (imprint: 2 physics_unreliable, 1 slip; flat pad and stock blade:
physics_unreliable); four finalists (stock single, stock-to-flat tool change, flat single, imprint
single) complete 0/8 each, all stopping at the tighten; no recommendation. The imprint single
inserts 8/8 and tightens 0/8 (6 physics_unreliable, 2 slip). Its build pack (`export_part_jaw_pack`)
holds `jaws/tool0_left.stl`, `_right.stl` (23,989 mm^3 each), `_mount_left.stl`, `_mount_right.stl`
(36,399 mm^3 each), each one watertight solid, CC0-1.0, plus `jaws/JAWS.txt`; the MJCF reloads as
tested. The stock single's pack has no `jaws/`, the upstream link, the licence text, and its MJCF
labelled CC-BY-4.0. The nine H2 runs of section 5.1 (in process) and the same setups' runs among
the finalists (spawned workers) have equal `traj_sha`.

**The default pool** (`run_task_tournament` as the agent tool calls it, 4 worker processes):

| | H2 `59ba912a5fa8` | H1 `b75435602453` |
|---|---|---|
| candidates | 49: 18 presets, the 3 part jaws on the coupling (in place of the 3 pre-registered entrants), 19 library entrants, 9 designed | 52: 18 presets, 6 part jaws (carton, pack), 19 library, 9 designed |
| "frozen v2 episode" prunes | none | none |
| part jaws pruned | none | imprint and flat for the carton (open clearance), all three for the pack (too wide); the carton's stock jaw on the steps on other objects |
| part jaws screened | insert 3/3 each; tighten 0/3 each (physics_unreliable) | stock jaw on `pick_carton` 0/3 (the lid stays shut in the carried state; 2 physics_unreliable) |
| finalists | the stock-blade jaw single (the best single off-the-shelf effector), a cup-to-screwdriver tool change and two-tool head, a force-controlled gripper; 0/8 each, all stop at the tighten | four, none with a part jaw; 0/8 each |
| wall time | 317 s | 202 s |

No setup completes either task, as before S5d (STUDIO_SEARCH.md section 11): H2 waits on the
tighten (section 5.2), H1 on the lid.

Two in-process runs and one spawned worker give the same `traj_sha`
(`tests/test_studio_partjaw.py`); the spawned worker finds the build path through the pickling hook.

## 6. Open issues and change requests

* **S2, threaded joint (affects H2 for every tool):** the friction ramp is an explicit
  velocity-saturated motor that chatters at the 2 ms control tick on the member's 0.0005 kg m^2
  armature (section 5.2): a member turns under a torque well below its ramp, and a tool's peak roll
  torque is not the thread's torque. Suggested: an implicit friction (the hinge's `frictionloss`
  set each tick from the ramp, as the slide already uses), or a velocity saturation wide enough for
  the tick.
* **S5, build pack:** `export_build_pack` (the agent tool) does not call `add_jaw_prints`; the
  printed jaws ship through `export_part_jaw_pack`. One call after step 3 of `export_build_pack`
  would make every pack carry them.
* **S2, grip heights:** the part-jaw build copies S2's grip heights (3 mm above a member's bottom,
  2 mm above a latch panel, `grip_tip` for a plug); a test checks the imprint seats in a twist, but
  a named constant in S2's modules would remove the copy.
* **The jaw force.** 25 N per jaw is the v2 default (the protected servo, sustained). A
  twelve-sided coupling needs about 88 N per jaw to hold 2 N m by its flats; a hexagon (the nut)
  has a longer cam lever (corner radius x cos 60 deg). The founder may want the unprotected regime
  as a declared Studio sensitivity, labelled as such.
* **A part-shaped jaw serves one object.** A single setup with one jaw cannot do a task on two
  objects (H1); the tool-change and two-tool-head combinations can.
* `docs/STUDIO_SEARCH.md` section 2 still describes the pool without the task's part jaws (S5's
  file; this file is the reference).
