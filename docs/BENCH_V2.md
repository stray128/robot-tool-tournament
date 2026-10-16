# Task v2 physics bench: gates GV1 and GV2

Measured on 2026-10-16 with MuJoCo 3.15.0 on the CPU, 16 cores, under heavy machine load
(load average 16 to 90). Timings are ratios, not absolute speeds. The gate code is
`rtt.sim.gates_v2`; every number below is in the JSON report it writes.

## How to run

```
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

Add `--native-ccd true` to run a gate with MuJoCo's native collision instead of libccd.

## Inputs

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

## GV1: fidelity and fairness

| | A, libccd | A, native | B, libccd | B, native |
|---|---|---|---|---|
| Worst simulated clearance error (limit 0.05 mm) | 0.019 mm | 0.019 mm | 0.000 mm | 0.000 mm |
| Worst first-contact error vs `q_seat` | 0.010 mm | 0.010 mm | 0.000 mm | 0.000 mm |
| Configurator imprint, zero-error registered | 100 % | 100 % | 100 % | 100 % |
| Swept imprint, zero-error registered | 85 % (tee 25 %, L-bracket 88 %) | 100 % | 100 % | 100 % |
| **GV1** | fail | pass | pass | pass |

The fixture's swept imprint on the tee fails on spurious penetration flags during closing
(about 2.3 mm reported, 0.16 mm with native collision). The generator's jaws do not show it.

## GV2: physics

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

## `native_ccd`: the rule and the decision

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

## Other findings

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

## Re-run on the current generator and part library (verification, 2026-10-16 evening)

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
