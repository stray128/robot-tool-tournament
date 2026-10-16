# Task v2 physics bench: gates GV0 to GV3 and GV-H

Measured on 2026-10-16 with MuJoCo 3.15.0 on the CPU of the 16-core laptop. The gate code is
`rtt.sim.gates_v2` (GV1, GV2), `tests/test_freeform_hypothesis.py` (GV0),
`rtt.analysis.gv3_run` (GV3) and `rtt.analysis.gvh_run` (GV-H); every number below is in the
JSON report it writes (under `runs/gates-v2/`, git-ignored; each re-creates bit-identically).

The first section is the gate record under the founder rulings of 2026-10-16 (task v2 at a
1 ms step with native collision, the `physics_unreliable` rule, the patched parts library).
The sections after it are the measurements that led to those rulings, kept as history; they
ran at 2 ms under heavy machine load (load average 16 to 90), so their timings are ratios only.

## How to run

```
make gv0 gv1 gv2            # GATES_OUT=runs/gates-v2 WORKERS=12 by default
make gv3 LOG=runs/gates-v2/gv3/calibration_log.json
make gvh

# test fixtures (tests/ on the path)
python -m rtt.sim.gates_v2 gv1 --builder freeform_fixtures:build_fixture_pair \
    --parts freeform_fixtures:training_parts --designs freeform_fixtures:gv1_designs --out gv1.json
python -m rtt.sim.gates_v2 gv2 --builder freeform_fixtures:build_fixture_pair \
    --parts freeform_fixtures:training_parts --designs freeform_fixtures:gv2_designs \
    --det-designs freeform_fixtures:determinism_designs --out gv2.json

# the generator and the part library
python -m rtt.sim.gates_v2 gv1 --builder rtt.design.freeform:build_pair \
    --parts rtt.parts.library:training_parts --designs rtt.sim.gates_v2:generator_gv1_designs
python -m rtt.sim.gates_v2 gv2 --builder rtt.design.freeform:build_pair \
    --parts rtt.parts.library:training_parts --designs rtt.sim.gates_v2:generator_gv2_designs \
    --det-designs rtt.sim.gates_v2:generator_determinism_designs
```

The determinism criterion needs 10 designs, so a gv2 run without `--det-designs` fails it.
GV1 checks the stock-blade hulls too (`--skip-stock-blade` leaves that out, for fixtures).

`task_v2` now runs MuJoCo's native collision; add `--native-ccd false` to run a gate with
libccd instead (the history sections below were measured with libccd as the task setting).

## Gate record under the founder rulings (2026-10-16, 22:27 to 23:05 IST)

Code at `v2-look/integ` commits `5467184` (GV1, GV2) and `7d7cffe` to `9b03486` (GV3, GV-H), task
`task_v2` with `timestep: 0.001`, `native_ccd: true`, `checks.held_penetration_max_mm: 1.0`;
parts library with track O's patch (the L-bracket gripped on its standing leg, the hex
standoff without a horizontal seam). Machine load average 3 to 12 (the gates' own workers).

| Gate | Result |
|---|---|
| GV0 geometry | pass |
| GV1 fidelity and fairness | pass (every criterion, all six parts) |
| GV2 physics | pass; the cost criterion misses (3.57x against 1.5x), a declared deviation |
| GV3 calibration | the Sobol spread misses (SD 0.113 against 0.15), declared under ruling 6; every other target met; frozen |
| GV-H hero screen | 0 heroes, so the hook uses the offset contrast only (the rule's fallback) |
| GV4 Nebius parity | pass: 384 of 384 episodes agree on outcome, every part, jaw-pair, design and hull hash equal |
| GV5 search harness | pass: harness tests, the diversity bars on the training parts, the probe (100 % valid after reserves, 0 repairs) and a local B = 40 Nemotron run |

### GV0: geometry (2,048 Sobol recipes x 6 parts)

| Criterion | Measured |
|---|---|
| Builds or a specific rejection | 1,287 recipes valid on all six parts; 462 schema, 288 `q_seat_range`, 11 `mass` first rejections; 0 crashes, 0 unspecific |
| Pieces convex (hull volume = piece volume) | worst relative difference 6.2e-13 |
| Print watertight | 0 failures over 9,035 pairs |
| Pad-zone print = collision (Hausdorff) | 2.1e-6 mm (limit 0.05 mm) |
| Pad never cuts into the part | deepest early cut 2.0e-13 mm^3 |
| Build time | p50 0.13 s, p95 0.26 s per pair (limit p95 0.5 s); max 1.05 s |
| Pieces per jaw, PLA per jaw | at most 33 (limit 40); at most 49.7 g (limit 50 g) |

### GV1: fidelity and fairness

| Criterion | Limit | Measured |
|---|---|---|
| Simulated clearance at seat, both imprints, six parts | designed +-0.05 mm | 0.000 mm worst error (0.4 mm designed, 0.4 mm simulated on every jaw) |
| First contact vs `q_seat` | +-0.05 mm | 0.000 mm |
| Configurator imprint, zero-error registered | >= 95 % per part | 100 % on all six parts (with the task jitter: 87.5 % on the tee, 100 % elsewhere) |
| Swept imprint, zero-error registered | >= 95 % per part | 100 % on all six parts |
| Stock-blade hulls on the pad contact zone (ruling 5, D9) | <= 0.2 mm | 0.0003 mm; 40 convex pieces per jaw, inputs not stale |
| Stock-blade hulls, whole inner face (reported) | | 0.75 mm, on recessed arms at least 2.5 mm behind the contact plane |
| Side clearance across the jaw (`lateral_slide_mm`, diagnostic) | | configurator 0.49 to 0.55 mm, swept 2.49 to 2.55 mm (0.4 and 2.4 mm designed) |

Deepest object-jaw penetration in the zero-error episodes: 0.07 to 0.12 mm.

### GV2: physics

3 designs (flat, conform 4 mm at 30 deg, V-funnel 6 mm at 45 deg; 40 mm wide) x 6 parts x
4 placements x 2 seeds = 144 matrix episodes; the parity runs extend it to 3 seeds (216).

| Criterion | Limit | Measured |
|---|---|---|
| Matrix builds | all | all 18 pairs |
| MuJoCo resets (`bad_qacc`) | 0 | 0 |
| Scored successes with penetration above 1 mm (ruling 4) | 0 | 0; the rule caught 2 held episodes (1.26 and 1.60 mm: V-funnel on the hex, conform on the block) and scored them `physics_unreliable` |
| Penetration p99, all episodes | <= 2 mm | 1.10 mm (max 1.60 mm) |
| Hold creep p95, held episodes | <= 0.5 mm | 0.40 mm (max 0.71 mm, flat pad on the bushing) |
| Deliberate failures | all fail | pass: too high -> `missed_grasp` on all six; mu 0.05 -> `slip_during_lift` on five, the bushing rests on the jaws by its flange (not counted); too wide -> `missed_grasp` on all six |
| Timestep | 1 ms (ruling 3) | 1 ms; the 2 ms re-run agrees on 94.4 % of 216 episodes (diagnostic; 12 differ, 5 of them the V-funnel on the hex) |
| `native_ccd` decision recorded | | on, by ruling 2 (the rule says off: v1 parity 98.5 %, v2 parity 97.2 %) |
| Asymmetric jaws, own mass properties | yes | yes (3 asymmetric pairs) |
| 10 designs x 3 fresh processes, identical hashes | yes | yes |
| v2 / v1 median CPU per episode | <= 1.5 (declared) | 3.57: 0.791 s against 0.221 s, 16 interleaved episodes each, one process |

Held (lift-only) 98.6 % of the matrix, registered 55.6 %; failure modes 62 `not_registered`,
2 `physics_unreliable`. With libccd (the same matrix): registered 56.9 %, held 97.9 %,
penetration p99 1.54 mm and max 2.90 mm, creep p95 0.14 mm, 2 `physics_unreliable`, 1 slip.

**The rulings fixed what the history sections measured.** The hex standoff no longer carries
at mu 0.05 (the patch removed its seam), the tee slips at mu 0.05 instead of being ejected,
and libccd-on-native parity on v2 episodes rose from 84-93 % to 97.2 % at the 1 ms step.

**Native vs libccd on the v1 parity set** is unchanged at 98.5 % (the same three primitive
cylinders); the lip-edge regression is still reproduced on v1 primitives (native ejects the
cylinder) and absent on both v2 analogues (25.0 N squeeze per jaw, held).

### GV3: calibration (official, screening seed 20261020)

The four round-0 entrants, the two reference entrants and 64 Sobol designs (91 draws, 70.3 %
valid) on the 48 episodes of `train_v2`: 3,360 episodes per iteration, 6.6 to 7.4 min on 14
workers. Every iteration is logged in `runs/gates-v2/gv3/calibration_log.json`.

| Iteration | Change | Sobol SD | Beat best entrant | Failing | Decision |
|---|---|---|---|---|---|
| 1 | none (the task as drafted) | 0.113 | 5 / 64 | spread | iterate |
| 2 | block 30 -> 20 mm across the jaws | 0.109 | 3 / 64 | spread, headroom | iterate |
| 3 | block back to 30 mm | 0.113 | 5 / 64 | spread | declare the miss and freeze |

Iteration 3 reproduces iteration 1 record for record (3,360 identical records, wall time
aside).

Result at the freeze (iterations 1 and 3):

| Design | Registered | Lift-only | `jam_on_close` | `off_centre` | `physics_unreliable` |
|---|---|---|---|---|---|
| stock | 25.0 % | 85.4 % | 0 | 34 | 7 (all on the bushing) |
| flat | 31.2 % | 100.0 % | 0 | 33 | 0 |
| imprint_configurator | 58.3 % | 95.8 % | 20 | 20 | 1 |
| imprint_swept | 31.2 % | 100.0 % | 8 | 33 | 0 |
| ref_wrap (reference) | 70.8 % | 97.9 % | 5 | 14 | 0 |
| ref_hook (reference) | 56.2 % | 95.8 % | 10 | 20 | 0 |
| Sobol-64 | mean 36.6 %, SD 0.113, median 31.2 %, max 77.1 % | mean 97.6 % | | | |

| Target | Measured | |
|---|---|---|
| Every round-0 entrant 20-70 % registered | 25.0 to 58.3 % | pass |
| Sobol-64 SD >= 0.15 | 0.113 | miss, declared (ruling 6) |
| >= 5 of 64 beat the best entrant (configurator, 58.3 %) | 5 | pass |
| <= 10 % of Sobol designs at 100 % | 0 | pass |
| `physics_invalid` share (reported) | 0 of 3,360 | |
| `physics_unreliable` share (reported) | 57 of 3,360 (1.70 %): 28 on the L-bracket, 16 on the hex standoff, 7 stock on the bushing, 5 on the tee (1 of them the configurator imprint), 1 on the bushing | |

**Why the spread is narrow.** Two of the six parts separate almost no designs: on the block
(30 mm across the jaws) every Sobol design scores the same 3 of 8, since no finger in the
space (at most 40 mm wide) reaches its side faces; on the tee, whose run is 60 mm long across
the jaws, nearly every design registers only at c0. Per part, the Sobol SD is 0.00 on the
block and 0.09 on the tee against 0.21 to 0.26 on the other four. Iteration 2 made the block
narrower: 13 Sobol episodes on it then lost registration (10 off centre, 3 jammed) and 6
gained it, so the spread fell. Re-scoring iteration 1's records under other registration tolerances (dx and dy 0.5 to
1.5 mm, seat 0.5 to 1.5 mm) moves the SD only between 0.113 and 0.126, so the tolerances were
left as drafted.

### GV-H: hero screen (hero rule digest `dda0ec35...`)

The stock blade, the configurator imprint and 7 designed candidates (the two references and
the five best Sobol designs of GV3) at the seven hero conditions, 3 jitter seeds each.

| Part | (a) stock fails / visibly | (b) imprint short or off | (c) best design passes | Verdict |
|---|---|---|---|---|
| pipe_tee | 6 / 0 | 6 of 6 | 1 of 7 | no |
| block | 6 / 0 | 6 of 6 | 1 of 7 | no |
| hex_standoff | 6 / 0 | 0 of 6 | 7 of 7 | no |
| flanged_bushing | 1 / 0 | 0 of 6 | 7 of 7 | no |
| l_bracket | 7 / 0 | 6 of 6 | 7 of 7 | no |
| steel_slug | 6 / 0 | 0 of 6 | 7 of 7 | no |

No part qualifies: the stock blade fails registration on most conditions but never visibly
(no slip, drop or tilt above 10 deg; it holds the part where it landed), and on the bushing
16 of its 21 episodes are `physics_unreliable`, which count against qualification. The hook
therefore uses the imprint-vs-designed offset contrast only (`hook_mode
offset_contrast_only`). The L-bracket shows that contrast in full: the imprint is short or
off-centre on 6 of 6 errors and the best design passes 7 of 7.

### GV4: Nebius parity (2026-10-16, 23:50 IST)

`rtt.sim.gv4` builds and simulates a fixed set of v2 designs exactly as a search scores them
(`SimEvaluatorV2`): the six entrants of `configs/entrants_v2.yaml` and two Sobol recipes
(seed 20261024, used by nothing else) on the 48 episodes of `train_v2`, 384 episodes. The
jaws are built where the episodes run. The same code ran on the laptop and as a Nebius job
(`python -m rtt.cloud run gv4 --no-fallback --vcpus 32`), and `python -m rtt.sim.gv4 compare`
matched the two reports episode by episode.

| Criterion | Limit | Measured |
|---|---|---|
| Episodes | at least 100 | 384, the same on both sides |
| Outcome agreement (`success_v2`, lift-only, failure mode) | at least 99 % | 384 of 384 (100 %) |
| Records bit for bit (every field but wall time) | reported | 257 of 384; the other 127 differ by at most 8.9e-16 mm (hold creep) and 4.4e-16 mm (`reg_dy`) |
| Training-part geometry hashes | equal | 6 of 6 |
| Jaw-pair geometry hash of every episode | equal | 384 of 384 |
| Design `geometry_id` | equal | 8 of 8 |
| Stock-blade hull file | equal | `501782c6...` on both |
| Task and suite digests | equal | yes |

Scores match the GV3 freeze for all six entrants (stock 25.0 %, configurator 58.3 %, ...).
Since the hashes agree, jobs build the jaws in-job; the spec's fallback (build once, score
stored geometry) is not needed.

| | Laptop | Nebius job |
|---|---|---|
| Machine | AMD Ryzen 7 5800H, 14 workers | AMD EPYC Genoa, `cpu-d3` `32vcpu-128gb`, uk-south2 |
| Evaluate (build + 384 episodes) | 44.6 s | 12.9 s |
| Job | | `aijob-e05ppqtwhtxanj89ze` (`gv4-20261016-181945`): provisioning 119 s, run 16 s, wall 135 s, $0.040 wall-based ($0.005 run-only) |
| Image | | `rtt-sim@sha256:74a3abc58ce7...`, built from `032a39f` (clean) |

### GV5: search harness (2026-10-16, 23:20 to 23:55 IST)

| Criterion | Measured |
|---|---|
| `test_budget_equal` (v2) | pass: every run of every arm (random, Nemotron with 40 % invalid, 30 % near-copy and 20 % garbage output, a hostile proposer) simulates exactly 40 valid, distinct designs; invalid output costs no budget (12 tests) |
| Round 0 byte-identical across arms | pass (`test_v2_round0_is_byte_identical_across_arms`) |
| `test_llm_cache_replay` | pass: a v2 run replays with 0 API calls; a changed notes block misses the cache instead of calling (8 tests). On real data: the B = 40 run below, replayed with `rtt replay-llm`: 0 live API calls, the same run fingerprint (40 designs, 1,920 episodes re-simulated) |
| De-duplication and canonicalisation | pass: `test_search_dedup` (25), `test_freeform_canon` (11) |
| Leak test | pass: `test_no_heldout_in_prompts` (8, including the v2 part names and dimensions), `test_vault_isolation` (6), `test_leak_check` (8); `scripts/leak_check.sh` clean |
| Diversity bars (rule D6) on the patched training parts | t = 0.070 (63 points), explore bar 0.15; see below |
| Probe: 5 calls of Super with the v2 tool, at least 90 % valid after reserves, at most 1 repair | pass: 80 % valid primary, **100 % valid after reserves**, 0 calls needing a repair, 80 % accepted after the screen; $0.030 |
| One local B = 40 Nemotron run completes | pass: 40 designs, 1,920 episodes, 15 calls, $0.114, 3 of 32 slots `fallback_random`, 8.3 min |

**Diversity bars.** `python -m rtt.search.threshold_v2` applies the threshold rule with the
full section-13 validators on the six training parts under `task_v2` (seed 20261016): 2,000
valid designs from 3,179 Sobol draws (62.9 % valid), 10,000 pairs. Pair-distance quantiles
p0.5 / p1 / p5 / p50 = 0.057 / 0.070 / 0.137 / 0.405, so t = **0.070** (it was 0.080 with the
schema rules only) and the explore bar, the median same-cell distance, is **0.15** (was
0.17). Recorded with its input hashes in `configs/diversity_v2.json`; the arms use these bars
from now on, for the shared screen of every arm as well as the Nemotron prompt (63 points).

**Probe on the real prompt.** Track P's probe of 2026-10-16 put synthetic parts in the prompt
and checked the slots with the schema rules only: 100 % valid, but re-checked with the
geometry validators only 35 % valid after reserves (4 calls needing a repair, 32 analytic
rejections). The GV5 probe (`python -m rtt.search.probe_v2 --live --real`) sends each call
arm C's real round-1 prompt: the six training parts and `task_v2`, the four v2 entrants and
four screened Sobol recipes in round 0, scored by the CPU simulator on `train_v2`. Five calls,
12.6k prompt and 2.1-3.2k completion tokens each, 10.6-15.0 s; the 4 invalid primary slots
were 3 seat-closure misses on the hex standoff (41.9-47.4 mm against 40) and draft outside a
V-funnel's 15-70 deg range, all filled by the call's own reserves. The pass rule keeps 4 slots
plus 4 reserves.

**Local B = 40 Nemotron run** (`rtt search --arm nemotron --space freeform-v2`, seed 20261015,
a development seed, laptop, 14 workers): best-so-far after 8 / 16 / 24 / 32 / 40 designs
0.625 / 0.646 / 0.708 / 0.750 / 0.750 (the best round-0 design, a Sobol recipe, 62.5 %).
29 of the 32 arm slots are Nemotron designs (14 of them reserves; 22 from main calls, 3 from repairs, 4 from reflections); 52 rejections: 25 invalid,
20 too similar, 4 not novel, 3 duplicate designs. 15 calls: 225k prompt and 51.6k completion
tokens (17.4k reasoning), p50 latency 15.8 s, max 31.1 s. Evaluator 202 s of the 8.3 min wall
time; the rest is the model and the geometry validators. One development run, not evidence
about S1.

### Cost

| Measure | v1 | v2 (1 ms, native) | Ratio |
|---|---|---|---|
| GV2: median CPU per episode, interleaved in one process | 0.221 s | 0.791 s | 3.57 |
| GV3: CPU per episode in the 14-worker pool (user + system, including building) | 0.51 s (spec estimate) | 1.37 s | 2.7 |

So a design costs about 38 to 66 CPU-s (48 episodes), not 25, and the spec's 8 CPU-h for the
pilot, 2 arms x R = 10 and the held-out suites becomes about 21 to 29 CPU-h [derived].

## History: measurements before the rulings

### Inputs

- **A, fixtures.** The jaw builder and the six parts in `tests/freeform_fixtures.py`: a port of
  the 2.5D prototype with plain slab bodies. Its pad strips are split at 5 mm, and straight
  walls leave 0.02 mm sliver strips. Both are fixture traits, not the generator's.
- **B, preview.** The generator and the part library as they stood on 2026-10-16, before
  either was merged, run in a scratch copy. The L-bracket is left out, because the generator
  rejects it ("seat closure 58.40 mm is outside [2.00 mm, 40.00 mm]"). So B has 5 parts and
  a 120-episode matrix.

GV2 designs: one flat pad, one drafted conform pad (depth 4 mm, draft 30 deg, clearance
0.4 mm) and one V-funnel (depth 6 mm, draft 45 deg). All three are 40 mm wide, blunt, grip
0.5, pad 20 mm. Placements are the train_v2 ones (c0 to c3) with 2 seeds; the parity checks
add a third seed (216 episodes for A, 180 for B).

### GV1: fidelity and fairness

| | A, libccd | A, native | B, libccd | B, native |
|---|---|---|---|---|
| Worst simulated clearance error (limit 0.05 mm) | 0.019 mm | 0.019 mm | 0.000 mm | 0.000 mm |
| Worst first-contact error vs `q_seat` | 0.010 mm | 0.010 mm | 0.000 mm | 0.000 mm |
| Configurator imprint, zero-error registered | 100 % | 100 % | 100 % | 100 % |
| Swept imprint, zero-error registered | 85 % (tee 25 %, L-bracket 88 %) | 100 % | 100 % | 100 % |
| **GV1** | fail | pass | pass | pass |

The fixture's swept imprint on the tee fails on spurious penetration flags during closing
(about 2.3 mm reported, 0.16 mm with native collision). The generator's jaws do not show it.

### GV2: physics

| Criterion | A, libccd | A, native | B, libccd | B, native |
|---|---|---|---|---|
| MuJoCo resets (`bad_qacc`) | 0 | 0 | 0 | 0 |
| Held (lift-only) | 97.2 % | 98.6 % | 87.5 % | 98.3 % |
| Registered | 56.9 % | 56.3 % | 48.3 % | 52.5 % |
| Held episodes with penetration above 1 mm (limit 0) | 5 (max 1.88 mm) | 7 (max 1.69 mm) | 1 (max 1.73 mm) | 3 (max 1.58 mm) |
| Penetration p99, all episodes (limit 2 mm) | 6.21 mm | 1.95 mm | 4.54 mm | 2.001 mm |
| Hold creep p95 (limit 0.5 mm) | 0.44 mm | 0.19 mm | 0.18 mm | 0.49 mm |
| Deliberate failures | pass | not run | fail (hex, see below) | fail (hex) |
| 1 ms vs 2 ms outcome parity (limit 99 %) | 96.3 % | not run | 87.8 % | 93.9 % |
| Asymmetric jaws carry their own mass properties | yes | | yes | yes |
| 10 designs x 3 fresh processes, identical hashes | yes | | yes | yes |
| v2 / v1 median CPU per episode (limit 1.5) | 2.33 | | 1.42 | 1.58 |

"Not run": the A native column is the matrix re-run inside the libccd gate, not a full gate.

### `native_ccd`: the rule and the decision

Spec section 6 allows `native_ccd: true` only if outcomes agree on and off in at least 99 %
of the v1 parity set and of 200 v2 episodes, and the lip-edge regression is reproduced and
shown fixed.

| Condition | Result |
|---|---|
| v1 parity set, 200 episodes | 98.5 % (3 disagree) |
| v2 episodes, A (216) and B (180) | 92.6 % and 83.9 % |
| Lip-edge regression reproduced on v1 primitives | yes: libccd squeezes 24/25 N and holds; native ejects the cylinder on closing |
| Fixed on v2 geometry (mesh cylinder on box lips and on mesh lips) | yes: both hold with native, squeeze 25.0/25.0 N |

**Decision: `native_ccd` stays `false`** in `configs/task_v2.yaml`, because both parity
conditions fail. The evidence points the other way, so a ruling is needed:

- All three v1 disagreements are primitive cylinders, the known regression class. v2 parts
  are always meshes, so v2 never has that pair.
- Most v2 disagreements are libccd failing where native and a 1 ms step agree. At a 2 ms
  step, libccd reports 2 to 3 mm penetrations on mesh-mesh contacts. The penetration check
  then fires and the episode fails: flat pads drop the hex standoff at mu 0.3, and the tee
  is "ejected" on closing. The episodes checked are clean (0.1 to 0.3 mm) at 1 ms with
  libccd and at either step with native. In B this costs 11 points of lift-only success
  (87.5 % vs 98.3 %).
- The rule's parity test assumes both colliders are mostly right. On v2 geometry the
  disagreements measure libccd's error, not native's.

### Other findings

1. **Seams between part pieces catch the pads.** The library's hex standoff is a solid base
   plus 12 ring wedges above it, around the tapped hole. When the part slides in the jaws,
   the pad's top edge meets the horizontal seam, and MuJoCo gives that contact a vertical
   normal: the jaw then carries the part on an internal face. At mu 0.05 the heavy copy is
   held with either collider; a one-piece hex of the same outline slips, as it must. Any
   part whose pieces meet in a seam across a gripped face can do this, such as the tee's
   wedge rings. Fix in the part library: keep gripped faces on single pieces, or leave
   product-only features (the tapped hole) out of the scored collision pieces.
2. **V-funnel on the hex.** The V touches the hex on its vertical edges. Penetration reaches
   1.3 to 2.8 mm and 1 ms vs 2 ms outcomes differ (6 of B's 11 native disagreements).
3. **Timestep parity below 99 % with either collider.** Besides the V-funnel cells, the
   remaining disagreements are borderline cases: c3 leaves a flat pad 0.7 to 0.8 mm off
   centre against the 0.75 mm tolerance, and c2 (-3 mm) sits at the edge of the drafted
   pad's capture range (0.4 + 4 tan 30 deg = 2.7 mm). The gate report now gives held and
   registered agreement separately as well.
4. **Cost scales with collision pieces.** On the fixture parts, 2-piece flat jaws ran at
   0.8 to 1.4 times the v1 cost per step, and 14 to 17-piece jaws at 1.1 to 2.7 times. The
   library's tee has 98 pieces. Fewer, wider pad strips and fewer part pieces near the jaws
   bring the ratio down.

### Re-run on the current generator and part library (verification, 2026-10-16 evening)

The S branch with its checks above, merged in a scratch copy with the generator and the part
library at their latest commits, all six parts. The generator still rejects the L-bracket, so
`matrix_builds` and GV1's builds fail on it; every other number is over the five parts that
build (120-episode matrix, 180 parity episodes).

| | libccd (the task) | native (matrix re-run) |
|---|---|---|
| GV1 closing-axis clearance error, first-contact error | 0.000 mm, 0.000 mm | |
| GV1 zero-error registration, both imprints | 100 % on 5 parts | |
| Held (lift-only), registered | 89.2 %, 47.5 % | 98.3 %, 53.3 % |
| Held episodes over 1 mm penetration | 7 (max 1.79 mm) | 3 (max 1.92 mm) |
| Penetration p99 | 6.77 mm | 2.06 mm |
| Creep p95 | 0.18 mm | 0.46 mm (max 0.79 mm) |
| Deliberate failures | fail: hex held at mu 0.05, tee ejected on closing at mu 0.05 | |
| 1 ms vs 2 ms parity | 88.3 % | |
| Native on/off: v2, v1 parity set | 87.8 %, 98.5 % | |
| Determinism, 10 designs x 3 processes | identical | |
| Cost v2 / v1 | 1.48 | |

Two more findings:

- **The imprint's side clearance is looser than designed.** Sliding the part across the jaw
  at the seat (`lateral_slide_mm`, now in the GV1 report) gives 0.49 to 0.55 mm for the
  configurator imprint against 0.40 mm designed: 0.50 mm on the block, whose sides run along
  the closing axis, so 0.10 mm extra. The swept imprint gives 2.49 to 2.55 mm against 2.40 mm.
  The closing-axis criterion cannot see this. The 0.1 mm outline grid rounding outward would
  explain it. A looser imprint jams less, so it favours the baseline.
- **libccd errors go both ways.** The stock blade on the tee at -3 mm across, with seed 0's
  friction (mu 0.27), slides the tee 3 mm to the centre under libccd and registers; with native
  collision the tee stays 2.36 mm off centre and does not register. So libccd at 2 ms also makes
  false registrations, not only the false failures above.
