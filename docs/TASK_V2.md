# Task v2: what every `freeform-v2` design is scored on

**Status: frozen at gate GV3 (2026-10-16, 23:05 IST).** This file describes `task_v2` and the
`train_v2` suite of the part-shaped finger design space, as frozen by the official GV3
calibration (screening seed 20261020, three logged iterations; the Sobol spread miss is
declared, see below). Any later change is a deviation and goes in `docs/DEVIATIONS.md`.
The founder rulings of 2026-10-16 (1 ms step, native collision, the `physics_unreliable`
rule, the stock-blade bulge on the contact zone, the parts fix) are in force and declared
there.

If the v2 gates are not green by the go/no-go call (Mon Oct 19, 22:00 IST), the
pre-registered experiment runs on the frozen v1 task instead (`docs/TASK.md`, fallback F1
in `docs/PREREG.md`), and v2 is used for product mode and visuals only, labelled
"product mode".

Items marked **[GV3]** could change during calibration, under the rules in
`docs/PREREG.md` section A9; none did.

## Frozen files (GV3, 2026-10-16)

| File | SHA-256 |
|---|---|
| `configs/task_v2.yaml` | `bef7297f8e1ff2a6270cd3f1ffb6de8690928f46485828a58fd1ce2313d87487` |
| `configs/suites/train_v2.yaml` | `ebd292b98fb46d220589a96e1d18d465e9b3152657401b11c3efd323e2625c63` |
| `configs/entrants_v2.yaml` | `7d9ac3922a34f3904e9ffb1cd868442197c3b94e475682cad2bbba3851faf421` |
| `configs/parts_v2.yaml` | `43b35533a91746d3333714a648f0d3ca300b7dc2d88297749f6bdbd07dbfff70` |
| Stock-blade hulls (`stock_blade_hulls.npz` under `assets/gripper/`) | `501782c6b1cc47db419fd47d82448b43a7385a0dc0a89c7938cd7f93d51aee0b` |
| Generator code (`src/rtt/design/freeform/*.py`, `GENERATOR_VERSION` `rtt-freeform-gen/2`) | `c3ab4e3751f4fb46b3b2006096b340b24a359d88f58fc2419bb60d7d0ef57268` |
| Parts library code (`src/rtt/parts/*.py`) | `62bb2cb129ff8cc5b70a6111bf1d86f2d23cbc56e09ccc53c4d18629fd1ad392` |

`rtt.analysis.calibrate_v2.freeze_hashes` computes the first four rows (they are in every
iteration of the calibration log). The two code rows are the SHA-256 of the `sha256sum *.py`
listing of each directory, as `(cd src/rtt/design/freeform && sha256sum *.py | sha256sum)`
prints it; task digest `task_digest(load_task_v2())` is in the GV3 report.

## What changed from v1

| | v1 (`task_v1`, `train_v1`) | v2 (`task_v2`, `train_v2`) |
|---|---|---|
| Finger | 10 parameters, 14 boxes, left finger mirrored by the scene | A part-relative recipe (9 scored parameters, 2 categoricals) compiled against each part into a left and a right jaw made of exact convex pieces |
| Success | Lift 100 mm and hold 2 s | **Registered grasp**: lift and hold as in v1, plus the part seated and centred at the end of closing. Lift-only success is still reported everywhere. |
| Controller settings | 3 grasp heights (k = 0.3, 0.5, 0.7), best one scored | One setting. Grip height is a design parameter. |
| Score | Best-k mean over 16 conditions (1/16 grid) | Mean registered success over 48 episodes (1/48 grid) |
| Training suite | 16 conditions of plastic and aluminium blocks, bars, cylinders and rods | 6 procedural industrial parts x 4 placements x 2 jitter seeds |
| Jitter | +-2 mm, +-3 deg, +-10 % friction | +-0.5 mm, +-1 deg, +-10 % friction |
| Stock entrant | Box approximation of the open-source gripper's finger | The real blade mesh of the same gripper, cut at the carriage, as convex hulls. Fallback: the v1 box approximation |
| Physics | 2 ms step, libccd for convex pairs | 1 ms step, MuJoCo's native collision (founder rulings 3 and 2); everything else as v1 |
| New checks | | `physics_invalid` (MuJoCo's bad-acceleration warning), `physics_unreliable` (held with jaw-part penetration above 1 mm, ruling 4), per-episode hold creep |

Gripper force and opening are unchanged. The 1 ms step and the mesh contacts make a v2
episode cost 3.6 times a v1 episode (gate GV2, `docs/BENCH_V2.md`), so the v1 compute
figures do not carry over: a design costs about 38 to 66 CPU-s.

## Gripper

The floating parallel gripper of v1, matched to an open-source parallel gripper
(attribution in the README):

- two rack-driven jaws, 42 mm travel each: an 84 mm maximum opening;
- grip force F = 50 N, i.e. 25 N squeeze per jaw;
- the scored palm is the 47.6 x 128 x 24 mm box. The gripper's CAD is drawn for show only
  and never collides (`contype = conaffinity = 0`, mass 0).

**Jaw frames** (the frozen contract in `src/rtt/design/freeform/types.py`):

- Each jaw is described in its own body frame, axes parallel to the palm: x across the
  jaw width, y along the closing axis, z up.
- z = 0 is the root plane, the carriage interface, at palm z = -12 mm.
- The left jaw sits on +y and faces -y; its carriage inner face is at y = +0.4 mm at full
  closure. The right jaw sits on -y, faces +y, and is at y = -0.4 mm.
- The right jaw is built against the part mirrored in y and has its own geometry, mass
  properties and assets. The scene, the renderer and the print pack never mirror a jaw.
- Both jaws are shifted so that their pads touch the centred part at the same closure,
  `q_seat`. The jaws are coupled, so without this one jaw would push the part sideways.

## Physics

As `task_v1`, except the step and the collider (founder rulings 3 and 2 of 2026-10-16,
declared in `docs/DEVIATIONS.md`):

- MuJoCo 3.15.0 on the CPU, **1 ms timestep** for every v2 episode (the control period stays
  2 ms), `implicitfast`, elliptic cone, `condim` 4, `impratio` 10;
- **`native_ccd: true`**: MuJoCo's native convex collision;
- sliding friction mu = 0.3 (the same mu for pad and part), torsional friction 0.001 m;
- contact `solref`/`solimp` as in v1.

The spec's rule would have kept `native_ccd` off: outcomes on and off agree on 98.5 % of the
v1 parity set and 97.2 % of 216 v2 episodes, below its 99 %. The founder switched it on: the
v1 disagreements are the primitive-cylinder lip-edge regression, which v2's mesh parts never
meet (it is reproduced on v1 primitives and absent on both v2 analogues), and at 2 ms libccd
gave 2 to 3 mm penetrations on v2 mesh contacts where native collision and a 1 ms step did
not. The 1 ms step replaces the spec's 1 ms vs 2 ms parity criterion (the 2 ms re-run agrees
on 94.4 % of 216 episodes).

**Collision geometry.**

| Geometry | Method |
|---|---|
| Designed jaws | Exact convex pieces built by the generator, at most 40 per jaw; pad strips are 8-vertex prisms. No convex decomposition. |
| Stock blade | The gripper's blade mesh cut at palm z = -12 mm, decomposed once with CoACD 1.0.14 (threshold 0.02, seed 0); 40 hulls per jaw, committed with their SHA-256. Bulge on the pad contact zone 0.0003 mm against the 0.2 mm limit (founder ruling 5, decision D9); 0.75 mm over the whole inner face, on recessed arms at least 2.5 mm behind the contact plane (reported). |
| Training and procedural held-out parts | Exact convex pieces (boxes, prisms, 16-segment wedges for rings and sockets, a hub plus tooth prisms for the gear). |
| Scanned held-out objects | CoACD 1.0.14, threshold 0.02, seed 0, at least 6 pieces (else retried at 0.01), cached, hashes in PREREG. |

Pads are built against the part's own collision pieces, so a pad matches exactly what the
physics engine sees. Visual meshes never enter the scored model: the scored model's hash
is the same with or without render assets.

## Controller

Open loop, as in v1: settle, descend, close (force ramp to F), lift 150 mm, hold. The
fingertip height comes from the design (`tip_height = grip_height_frac x part height -
pad_height / 2`, floored at 3 mm). Registration is measured 0.25 s after the close phase
ends, before the lift starts, so hold creep cannot affect it.

## Success: the registered grasp (`success_v2`)

All of the following must hold:

1. **Lift and hold, as in v1:** the part rises at least 100 mm and holds for a
   continuous 2.0 s before t = 12 s.
2. **Seated**, at the end of closing + 0.25 s: `seat_short_mm` = `q_seat` - q at most
   **1.0 mm** [GV3].
3. **Centred**, at the same moment, with the part pose in the palm frame:
   - |dx| (across the jaw) at most **0.75 mm** [GV3];
   - |dy| (along the closing axis) at most **0.75 mm** [GV3];
   - yaw at most **2 deg**, taken modulo the part's declared `yaw_symmetry` [GV3];
   - tilt at most **3 deg** [GV3].
4. **Upright after the hold:** tilt at most **5 deg** [GV3].

The criterion is identical for every design. The tolerances may change only during GV3,
on the screening sample, in at most one iteration; they are then frozen here.

**Failure modes.** Each failed episode gets one mode, by precedence: the physics flags
(`physics_invalid`, then `physics_unreliable`, then the v1 flags), then `missed_grasp`,
`ejected_on_close`, `slip_during_lift`, `unstable_hold`, `not_registered`. `not_registered`
carries the diagnostic sub-flags `jam_on_close` (shortfall above 1 mm) and `off_centre`.

**Physics validity.** If MuJoCo's bad-acceleration warning counter rises during the scored
part of an episode (up to the moment its outcome is decided; the replay-only tail is never
checked), the episode is labelled `physics_invalid` and counts as a failure. If more than
1 % of a run's episodes are flagged, a deviation is logged. The simulator's scorer, the
search harness and the analysis apply this one rule.

**Penetration (`physics_unreliable`, founder ruling 4).** A held episode (the v1
lift-and-hold criterion met) whose deepest jaw-part penetration, sampled at every 10 ms
check up to the decision, is above **1 mm** (`checks.held_penetration_max_mm`) is
`physics_unreliable`: it fails both the registered and the lift-only criterion, for every
arm and design alike. Its share is reported per run (1.70 % of the GV3 screening episodes).

**Hold creep** (`hold_creep_mm`, the part's slide in the jaws during the hold) is logged
per episode and never scored.

## Score

A design's score is its mean `success_v2` over the 48 episodes of `train_v2`. Ties break
on lift-only success, then lower mass. Lift-only success is reported for every design and
every endpoint.

**Records** (`RECORD_VERSION = 2`) add `registered`, `seat_short_mm`, `reg_dx_mm`,
`reg_dy_mm`, `reg_dyaw_deg`, `reg_tilt_deg`, `hold_creep_mm`, `bad_qacc`, `part_id`,
`geometry_hash` and `design_space`.

## Training suite `train_v2`

6 parts x 4 placements x 2 jitter seeds = **48 episodes per design**. The parts are
procedural (generated by this repository, Apache-2.0), unbranded, and built from exact
convex pieces, so they can be regenerated at new sizes for the held-out suite.

| Part | Material look | Role |
|---|---|---|
| 1/2" slip pipe tee, lying flat, branch toward the left jaw | white PVC | Hero. A straight imprint jams on the branch; a drafted pad keys it. |
| 30 x 40 x 60 mm block, no chamfer | anodised aluminium | The clearest jam case: a prismatic part with no lead-in |
| Hex standoff, standing, 17 mm across flats, `yaw_symmetry` = 6 | zinc-plated steel | Yaw registration |
| Flanged bushing, 20 mm x 30 mm with a 36 x 5 mm flange | brass | A hook under the flange |
| L-bracket 40 x 40 mm, 30 mm wide, 4 mm wall, lying flat; the jaws grip the standing leg's edges | stamped steel | Reach: a 4 mm leg gripped 9.5 mm off the centre of mass |
| Steel slug, 40 mm x 50 mm, 490 g | ground steel | The honest null: a round part at mu 0.3 does not jam |

Exact dimensions, densities and features are in `configs/parts_v2.yaml` (frozen above).
The L-bracket and hex standoff are as patched on 2026-10-16 (founder ruling 1): the
L-bracket's frame is centred on its standing leg, and the hex standoff is twelve full-height
sectors around the bore plus a plug under the hole, so no internal seam crosses a gripped
flat.

**Placements:** c0 centred; c1 +2 mm across the jaw; c2 -3 mm across; c3 +1 mm across
with +6 deg yaw.

**Jitter:** +-0.5 mm, +-1 deg, +-10 % mu, with common random numbers: every design sees
the same jitter draws. At v1's +-2 mm, jitter alone would decide registration.

## Validity

Validators run cheapest first and are identical for every arm. A recipe must be valid on
all 6 training parts:

1. **Schema:** bounds, strategy rules (v_funnel: draft 15-70 deg and depth at least 3 mm;
   conform: depth at least 1 mm; hook: tip size at least 2 mm), quantisation.
2. **Analytic, per part:** a non-empty outline in the pad band; at full open the jaws clear
   the part by at least 3 mm per side including a 3 mm error envelope; pads at least 1 mm
   apart at seat; `q_seat` in [2, 40] mm; derived length at most 110 mm with the palm at
   least 10 mm above the part; width at most 40 mm.
3. **Build:** succeeds on all 6 parts; every piece convex; at most 40 pieces per jaw; the
   print is one closed manifold; at most 1 s per pair.
4. **Printability:** minimum wall 1.2 mm; features at least 1.6 mm; at most 50 g of PLA per
   jaw at the worst part; body slopes at most 45 deg.
5. **Seat:** the pad never cuts into the part; jaw-to-part distance at least the clearance
   less 0.02 mm (contact allowed for `flat` only); the jaws do not intersect; at least
   2 mm above the table.
6. **Duplicates:** a design with the same `geometry_id` as an earlier one is rejected as
   `duplicate_geometry`, then the diversity rule applies.

A recipe that builds invalid on a held-out part scores as a failure there.

## Entrants (`configs/entrants_v2.yaml`)

| Name | Recipe | Round 0 |
|---|---|---|
| `stock` | The open-source gripper's own blade (real mesh, convex hulls); the seat-closure range applies to it as to every recipe | yes |
| `flat` | flat pad, blunt tip, grip 0.5, pad 20 mm, width 20 mm | yes |
| `imprint_configurator` | conform, draft 0, clearance 0.4 mm, depth = min(40 % of the part's half-extent along the closing axis in the pad band, the stroke cap) and at least 4 mm, grip 0.5, pad 20 mm, width 40 mm, blunt | yes |
| `imprint_swept` | the same with a +-2.0 mm sweep, as in published imprint-and-sweep jaw design (Gebauer et al., Production Engineering, 2025), through a baseline wrapper outside the search space | yes |
| `ref_wrap` | conform, depth 4 mm, draft 30 deg, clearance 0.4 mm | reference, outside the budget |
| `ref_hook` | conform, depth 2 mm, draft 30 deg, hook, tip 6 mm | reference, outside the budget |

Round 0 of every run is these 4 entrants plus 4 Sobol designs, byte-identical across arms
for a given run seed. The reference entrants are scored outside the budget and shown in
every comparison.

**Anti-strawman gate (GV1).** Both imprints register in at least 95 % of zero-error
episodes on all 6 parts, and the simulated clearance at seat equals the designed clearance
within 0.05 mm. Otherwise the imprint is fixed before anything is scored.

## Held-out suites

**Procedural (`heldout_procedural_v2`).** Commit-reveal with a new secret seed; 100
conditions x 2 jitter seeds.

- 70 % are new instances of the 6 training families, scaled 0.7-1.3, at least 30 % of
  them outside the training dimensions.
- 30 % come from 2 unseen families: a 90 deg pipe elbow and a spur gear (module 1.5-2,
  16-24 teeth, lying flat).
- Placement: uniform in a +-3 mm disc, +-8 deg yaw and +-1.5 mm height, all beyond the
  training ranges.
- Recipes are recompiled against each held-out part. No search process sees these parts.

**Real objects (`heldout_real_v2`).** The `heldout_real_v1` selection rule unchanged;
only the decomposition changes (CoACD 0.02, see above). Secondary endpoint.

## What we will and will not claim

The claim is fixed now, before any result:

> "On parts with flat or socket features, the configurator imprint stops short under
> +-1-3 mm placement error and holds the part where it landed. Drafted designs pull it to
> centre."

- Round parts at mu 0.3 do not jam, and we say so.
- Friction sweeps at mu 0.6 and 1.0 are exploratory and shown as a limit.
- Everything is simulated at mu 0.3 and 25 N per jaw. No claim of real-world centring.
- Rib-windowed bodies are rigid trusses, never presented as compliant.

## Calibration (gate GV3)

Run on a screening sample of 64 Sobol designs (public seed 20261020, distinct from every
gate and search seed) plus the round-0 and reference entrants. Targets: every round-0
entrant 20-70 % registered; Sobol SD at least 0.15; at least 5 of 64 beat the best entrant
(or the miss is declared, as with v1's G3); at most 10 % of Sobol designs at 100 %. At
most 3 logged iterations; only part dimensions, density or material, placements and (in
one iteration) the registration tolerances may change. Friction, grip force and the
physics settings stay as above. The procedure is `rtt.analysis.calibrate_v2` and
`docs/PREREG.md` section A9.

### Result (official, 2026-10-16)

3,360 episodes per iteration on the laptop (16 cores, 14 workers), logged in
`runs/gates-v2/gv3/calibration_log.json` (git-ignored; it re-creates bit-identically).

| Iteration | Change | Sobol SD | Beat the best entrant | Decision |
|---|---|---|---|---|
| 1 | none (the task as drafted) | 0.113 | 5 / 64 | iterate (spread) |
| 2 | block 30 -> 20 mm across the jaws | 0.109 | 3 / 64 | iterate (spread, headroom) |
| 3 | block back to 30 mm | 0.113 | 5 / 64 | declare the spread miss and freeze |

| Design | Registered | Lift-only |
|---|---|---|
| stock | 25.0 % | 85.4 % |
| flat | 31.2 % | 100.0 % |
| imprint_configurator | 58.3 % | 95.8 % |
| imprint_swept | 31.2 % | 100.0 % |
| ref_wrap (reference) | 70.8 % | 97.9 % |
| ref_hook (reference) | 56.2 % | 95.8 % |
| Sobol-64 | mean 36.6 %, SD 0.113, median 31.2 %, max 77.1 %, none at 100 % | mean 97.6 % |

Every target is met except the Sobol spread (SD 0.113 against 0.15), which is declared
under founder ruling 6 (`docs/DEVIATIONS.md`). Two parts separate almost no designs: on the
block every Sobol design scores 3 of 8, and on the tee nearly every design registers only
centred. `physics_invalid`: 0 episodes; `physics_unreliable`: 57 of 3,360 (1.70 %). The
registration tolerances are unchanged. The numbers by part, placement and iteration are in
`docs/BENCH_V2.md`.
