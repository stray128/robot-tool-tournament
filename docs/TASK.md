# Task v1: what every design is scored on

This file freezes the training task after calibration gate G3 (2026-10-15). Every arm,
every manual entrant and every random design is scored on exactly this task and suite.
Any later change is a deviation and goes in `docs/DEVIATIONS.md`.

## Frozen files

| File | SHA-256 |
|---|---|
| `configs/task_v1.yaml` | `1fb8fd4d06998f317616623ed1ba6f88c661b58b29aadc55c387bfb90f378081` |
| `configs/suites/train_v1.yaml` | `7e62796db3ffaee53eb553cb5343322a5c42453a7ae982f60234b8e94a701501` |
| `configs/entrants.yaml` | `8c1f50a7cf1ac4470ff91b7a5356f31327a70755f95a4bfa6dc49fa1fb64204d` |

Content digests, recorded in every run manifest:

- task: `task_digest` = `823e58a28a93`;
- training suite: `Suite.digest()` = `2dc46d1f79d82059cbe108193ae2ba29159a6cea25f75745bb4bbf5895eaff6a`.

Check them with `sha256sum configs/task_v1.yaml configs/suites/train_v1.yaml configs/entrants.yaml`.

## The task in one page

**Gripper.** A floating parallel gripper matched to an open-source 6-DoF arm's parallel
gripper (attribution in the README).

- Two rack-driven jaws, 42 mm travel each: an 84 mm maximum opening and a 0 mm closed gap.
- Grip force F = 50 N, i.e. 25 N squeeze per jaw. This is the servo's sustained regime
  after overload protection.
- The palm is the gripper frame, 47.6 x 128 x 24 mm.

The sources and the [S]/[D]/[A] tags are in the comments of `configs/task_v1.yaml`.

**Physics.**

- MuJoCo 3.15.0 on the CPU, with a 2 ms timestep, the `implicitfast` integrator, an elliptic cone, `condim` 4 and `impratio` 10.
- Sliding friction μ = 0.3. This is the low end of the 0.3–0.5 range assumed for bare PETG-CF pads.
- Torsional friction is 0.001 m, which is μ times an effective contact-patch radius of about 3 mm.
- Objects without their own material are solid plastic at 1,200 kg/m³.

**Controller.** Open loop, with five phases: settle, descend, close (force ramp to F), lift 150 mm, then hold.

- There are three controller settings, k = 0.3, 0.5 and 0.7. Each places the fingertips at that fraction of the band they can reach while the palm stays 10 mm above the object.
- The same three settings apply to every design.

**Success.** The object's centre rises at least 100 mm and stays there for a continuous 2.0 s before t = 12 s.

- Each failure gets exactly one mode: `missed_grasp`, `slip_during_lift`, `ejected_on_close`, `unstable_hold` or `finger_env_collision`.
- Physics checks flag penetration over 2 mm, blow-ups, ejection and palm contact.

**Score.** A design's score is the mean success at its best controller setting over the
training suite. Ties break on mean hold fraction, then on lower finger mass.

**Training suite** (`configs/suites/train_v1.yaml`): 16 parts-handling conditions × 3 settings × 1 public jitter seed, giving 48 episodes per design.

| Family | Condition | Object | Material | Placement |
|---|---|---|---|---|
| box | large / sloppy | 80 × 50 × 110 mm housing | plastic, 0.53 kg | aimed 6 mm and 15° off |
| box | long / handle20, 35, 50 | 140 × 40 × 60 mm block | aluminium, 0.91 kg | grasped 20, 35 or 50 mm from its centre |
| box | bar / handle50 | 160 × 30 × 30 mm bar | aluminium, 0.39 kg | grasped 50 mm from its centre |
| box | slab / handle30, 40, 45 | 120 × 30 × 80 mm slab | aluminium, 0.78 kg | grasped 30, 40 or 45 mm from its centre |
| cylinder | large / sloppy | Ø50 × 110 mm, standing | plastic, 0.26 kg | 6 mm and 15° off |
| cylinder | thin / sloppy | Ø12 × 100 mm pen, standing | plastic, 0.014 kg | 6 mm and 15° off |
| rod | wire / sloppy | Ø6 × 120 mm, lying flat | aluminium, 0.009 kg | 6 mm and 15° off |
| rod | peg / centred, handle20 | Ø14 × 140 mm, lying flat | aluminium, 0.056 kg | centred, or 20 mm from its centre |
| rod | shaft / handle20, 35, 50 | Ø28 × 180 mm, lying flat | aluminium, 0.28 kg | 20, 35 or 50 mm from its centre |

**Seeds.** Every condition's jitter seed is derived from the public `base_seed` and the condition ID. Jitter adds ±2 mm and ±3° to the pose and ±10 % to μ, and is the same for every candidate (common random numbers).

**Validity.** The validator requires a 55 mm opening at full open, which is the widest object (50 mm) plus a 5 mm margin.

**Held-out suite** (commit-reveal, `rtt.sim.conditions.heldout_suite`). It has 100 conditions, each run with 2 jitter seeds, all drawn from the secret seed.

- **Sizes:** new sizes, with 30 % extrapolated 15–30 % beyond the training range in each direction.
- **Placements:** new placements and yaws.
- **Materials:** each condition takes a material its family uses in training.
- **Off-centre grasps:** they appear with the same per-family frequency as in training, at 15–60 % of the part's half-length.

  These last two rules were added during calibration, so the held-out suite tests the same kinds of difficulty as the training suite.

## Calibration (gate G3)

Run with `rtt calibrate`. It scores the 4 manual entrants plus 64 scrambled-Sobol valid
designs (public seed 20261018) on the training suite, on the CPU executor with 16 workers.
The full record is in `runs/calibrate/g3-2026-10-15/calibration.json` (git-ignored; it
re-creates bit-identically, and the final run reproduced the exploratory run exactly).

### Final result (frozen task and suite)

| Design | Score | Best k | Box | Cylinder | Rod | Failure modes at best k |
|---|---|---|---|---|---|---|
| straight | 69 % | 0.3 | 50 % | 100 % | 83 % | 4 slip during lift, 1 unstable hold |
| concave | 75 % | 0.3 | 50 % | 100 % | 100 % | 4 slip during lift |
| hooked | 50 % | 0.3 | 50 % | 50 % | 50 % | 4 missed grasp, 4 slip during lift |
| stock | 69 % | 0.3 | 50 % | 100 % | 83 % | 4 slip during lift, 1 missed grasp |
| Sobol-64, mean | 56 % | | 73 % | 85 % | 22 % | |

Sobol-64 scores:

| Statistic | Value |
|---|---|
| Mean | 0.556 |
| SD (ddof=1) | 0.156 |
| Min | 0.125 |
| Quartiles | 0.500 / 0.562 / 0.625 |
| Max | 0.938 |
| Designs at 100 % | 0 |
| Designs at 0 % | 0 |

| G3 criterion | Target | Measured | Verdict |
|---|---|---|---|
| Entrants in range | every entrant 30–70 % | 3/4: concave scores 75 % | **fail** (by 5 points) |
| Family gap | ≥ 15 points on one family | 50 points (cylinder and rod) | pass |
| Sobol SD | ≥ 0.15 | 0.156 | pass |
| Headroom | ≥ 5 of 64 beat the best entrant | 4 of 64 beat concave's 75 % (94, 88, 81, 81 %) | **fail** (by 1) |
| Kill trigger 4 | SD ≥ 0.10 | 0.156 | not triggered |

**Gate G3 formally fails on two criteria, each by a narrow margin.** Kill trigger 4 does
not fire, because the score does respond to geometry (SD 0.156, up from 0.050). The
decision on how to proceed is below.

### How we got here

Each row below is one gate run with 64 Sobol designs. Only task-level knobs were changed:
friction, density, sizes and placements.

| Run | Change | Entrants (straight / concave / hooked / stock) | Sobol SD | Beat best |
|---|---|---|---|---|
| v0 | Retargeted task: μ 0.8, 700 kg/m³, 12 light objects | 100 / 100 / 100 / 100 | 0.050 | 0 |
| v1 | μ 0.4 | 100 / 100 / 75 / 100 | 0.062 | 0 |
| v2 | μ 0.4, 1,200 kg/m³ | 100 / 100 / 100 / 100 | 0.062 | 0 |
| A | 12 parts conditions, μ 0.3, torsion 0.001 | 83 / 92 / 58 / 83 | 0.130 | 0 |
| A′ | Same suite, μ 0.4 | 92 / 100 / 67 / 92 | 0.142 | 0 |
| B | 14 conditions (+ long/handle50, aluminium cylinder) | 79 / 86 / 50 / 79 | 0.123 | 3 |
| D | 14 conditions (light anchors swapped for slab handles) | 71 / 79 / 50 / 71 | 0.157 | 4 |
| **E (frozen)** | 16 conditions (+ slab/handle40, shaft/handle50) | **69 / 75 / 50 / 69** | **0.156** | **4** |
| E2 | E with 2 jitter seeds per condition | 72 / 72 / 50 / 75 | 0.151 | 4 |

Between v2 and A, three screens put 4 entrants and 16–20 Sobol designs through about 250
candidate conditions. The screens used a separate design sample (seed 20261019), so the
conditions were not picked on the gate's own designs. They found three things:

1. **Friction capacity is not the bottleneck.** μF = 15–40 N holds any part that fits in
   the opening. Mass and μ alone barely separate designs.
2. **Thin parts lying flat separate designs strongly.** A rod of radius 3–7 mm can only be
   pinched below its equator, so tip shape, bend and taper decide. Straight and concave
   fingers succeed; most random designs, and the hooked entrant, miss.
3. **Heavy parts grasped off-centre separate designs too.** Here contact-patch size
   (finger width and length) and grasp height decide, and the flat entrants slip.

Runs A to E were then scored on the gate's own 64-design sample, so the final suite was
chosen after seeing those designs. E is the last of five iterations, and we stopped there
rather than tune further against the gate sample.

The two kinds of condition pull the grasp height in opposite directions: rods want
k = 0.3, off-centre blocks want 0.5–0.7. A design therefore has to make one setting work
for both, which is what produces the spread.

E2 shows that a second jitter seed changes nothing material and doubles the cost, so the
frozen suite keeps one seed per condition.

### Decision and what to watch

- **Proceed with the frozen task, and record the near-miss in PREREG as a declared deviation from G3.**
  - The benchmark separates designs: SD is 0.156, the Sobol range is 13–94 %, and per-family gaps reach 50 points.
  - It leaves room above every entrant: 4/64 random designs beat the best entrant, and 1 reaches 94 %.
- **Headroom above concave is thin.**
  - Random designs beat concave's 75 % at a rate of 4/64. A random run of B = 40 (36 random designs after the 4 entrants) finds at least one such design with probability about 0.9.
  - So H1 (Nemotron beats random on held-out) may show a small effect.
  - Kill trigger 4's fallback, making sample efficiency (S1) the primary endpoint, is the founder's call before the PREREG tag. Our recommendation is to keep H1 primary and S1 as the first secondary endpoint, since the score is not flat.
- **The live slice is an early signal, not a result.** Its first Nemotron round proposed a v-groove mutation of the concave finger that scored 75 %, tying the best entrant, on its first call (`runs/slice/slice-20261015-a`).
- **Plan deviation on entrants.** Addendum A said to *replace* one invented entrant with the stock finger. We *added* it, giving 4 entrants.
  - Concave and hooked are each defined as one-idea changes from straight, so dropping straight would lose that control.
  - Round 0 is therefore 4 entrants plus 4 Sobol designs.
  - This needs founder sign-off before PREREG.
- **Approximations that affect the absolute numbers, but not the comparison between arms:**
  - one μ for both pad and part;
  - box-approximated fingers;
  - an open-loop controller;
  - mesh-object contact artefacts (see `docs/BENCH.md`).

## Reproduce

```bash
rtt calibrate                           # 3,264 episodes, about 105 s on 16 cores
sha256sum configs/task_v1.yaml configs/suites/train_v1.yaml configs/entrants.yaml
```
