# Task v2: what every `freeform-v2` design is scored on (DRAFT)

**Status: draft, not frozen.** This file describes `task_v2` and the `train_v2` suite as
specified for the part-shaped finger design space. It is frozen at gate GV3
(calibration, Mon Oct 19), when the hashes below are filled in. Any later change is a
deviation and goes in `docs/DEVIATIONS.md`.

If the v2 gates are not green by the go/no-go call (Mon Oct 19, 22:00 IST), the
pre-registered experiment runs on the frozen v1 task instead (`docs/TASK.md`, fallback F1
in `docs/PREREG.md`), and v2 is used for product mode and visuals only, labelled
"product mode".

Items marked **[at freeze]** are filled in by the GV3 freeze. Items marked **[GV3]** may
change during calibration, under the rules in `docs/PREREG.md` section A9.

## Frozen files [at freeze]

| File | SHA-256 |
|---|---|
| `configs/task_v2.yaml` | [at freeze] |
| `configs/suites/train_v2.yaml` | [at freeze] |
| `configs/entrants_v2.yaml` | [at freeze] |
| `configs/parts_v2.yaml` | [at freeze] |
| Stock-blade hulls (`stock_blade_hulls.npz` under `assets/gripper/`) | [at freeze] |
| Generator version and code (`src/rtt/design/freeform/`) | [at freeze] |

`rtt.analysis.calibrate_v2.freeze_hashes` computes the first four rows.

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
| Physics | as listed below | unchanged |
| New checks | | `physics_invalid` (MuJoCo's bad-acceleration warning), per-episode hold creep |

Physics, gripper force and opening are unchanged, so v1 compute figures carry over.

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

As `task_v1`:

- MuJoCo 3.15.0 on the CPU, 2 ms timestep, `implicitfast`, elliptic cone, `condim` 4,
  `impratio` 10;
- sliding friction mu = 0.3 (the same mu for pad and part), torsional friction 0.001 m;
- contact `solref`/`solimp` as in v1.

`native_ccd` stays `false` unless gate GV2 shows at least 99 % identical outcomes on the
v1 parity set and on 200 v2 episodes, and the v1 lip-edge-normal regression is reproduced
and shown fixed. The decision is recorded here at freeze: **[at freeze]**.

**Collision geometry.**

| Geometry | Method |
|---|---|
| Designed jaws | Exact convex pieces built by the generator, at most 40 per jaw; pad strips are 8-vertex prisms. No convex decomposition. |
| Stock blade | The gripper's blade mesh cut at palm z = -12 mm, decomposed once with CoACD 1.0.14 (threshold 0.02, seed 0); hulls committed with their SHA-256; inner-face bulge at most 0.2 mm (reported). |
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

**Failure modes.** Each failed episode gets one mode, by precedence: physics flags
(`physics_invalid` and the v1 flags), then `missed_grasp`, `ejected_on_close`,
`slip_during_lift`, `unstable_hold`, `not_registered`. `not_registered` carries the
diagnostic sub-flags `jam_on_close` (shortfall above 1 mm) and `off_centre`.

**Physics validity.** If MuJoCo's bad-acceleration warning counter rises during an
episode, the episode is labelled `physics_invalid` and counts as a failure. If more than
1 % of a run's episodes are flagged, a deviation is logged.

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
| L-bracket 40 x 40 x 30 mm, 4 mm wall, lying flat | stamped steel | Reach and asymmetric jaws |
| Steel slug, 40 mm x 50 mm, 490 g | ground steel | The honest null: a round part at mu 0.3 does not jam |

Exact dimensions, densities and features are in `configs/parts_v2.yaml` [at freeze].

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
| `stock` | The open-source gripper's own blade (real mesh, convex hulls) | yes |
| `flat` | flat pad, blunt tip, grip 0.5, pad 20 mm, width 20 mm | yes |
| `imprint_configurator` | conform, draft 0, clearance 0.4 mm, depth = min(40 % of the part's half-extent along the closing axis in the pad band, the stroke cap) and at least 4 mm, grip 0.5, pad 20 mm, width 40 mm, blunt | yes |
| `imprint_swept` | the same with a +-2.0 mm sweep, as in published imprint-and-sweep jaw design (Gebauer et al., 2025), through a baseline wrapper outside the search space | yes |
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
most 3 logged iterations; task-level knobs only. The procedure is
`rtt.analysis.calibrate_v2` and `docs/PREREG.md` section A9. Results: **[at freeze]**.
