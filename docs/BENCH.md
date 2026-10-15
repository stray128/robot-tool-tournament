# Simulator benchmarks

Measured numbers for the CPU MuJoCo executor (`rtt.sim`), with the commands that
reproduce them. Every command prints JSON; `--out FILE` also saves it.

```
python -m rtt.sim.bench parity      --workers 16   # timestep parity + gate G1 at each timestep
python -m rtt.sim.bench throughput  --worker-counts 16 8 4 1
python -m rtt.sim.bench entrants    --workers 16   # manual entrants on the training suite
```

## Setup (2026-10-15)

| Item | Value |
|---|---|
| Machine | Laptop, AMD Ryzen 7 5800H (8 cores, 16 threads), 27 GB RAM |
| Software | Python 3.12.3, `mujoco` 3.15.0, one single-threaded simulation per process (`spawn` pool) |
| Task | `configs/task_v1.yaml` after the gripper retarget: 84 mm opening, F = 50 N, palm clearance 10 mm |
| Load | **Shared machine.** Other jobs kept the 1-minute load average at 35-42 during every run below (16 hardware threads), so all throughput numbers are lower bounds; single-worker physics ran at ~4,200 steps/s here vs 10,700-13,900 steps/s (G1 episodes) on a quieter machine earlier the same day. |

## Timestep: 2 ms adopted (was 1 ms)

Rule (set before measuring): adopt 2 ms if gate G1 passes and outcomes on a
200-episode random parity set are identical on at least 99 % of episodes. The rule
was applied to the success flag, the plan's agreement metric for executor parity;
failure-mode labels agreed on 98 %, which is below 99 %.

The parity set is fixed and public (`rtt.sim.bench.parity_jobs`): 100 random
conditions (named `parity_v1/...`; they are not the held-out suite) from the held-out
generator run with a public, non-secret seed text (box,
cylinder and rod; sizes interpolated and extrapolated +-15-30 % from the training
suite; random placements, yaws and pose errors up to 3 mm / 8 deg), 2 jitter seeds
each; every episode gets a random design (the 4 manual entrants plus 46 random valid
designs) and a random controller setting. The random designs are validated against
the parity set's own widest object (64.8 mm, so a 69.8 mm opening), which is stricter
than training validity (55 mm): this sample's bends span -14 to +18 deg and 9 of 46
have hooks, so strongly bent fingers (up to 45 deg, valid for training) are not in it.

| | 1 ms (reference) | 2 ms |
|---|---|---|
| Success rate | 168/200 (84.0 %) | 167/200 (83.5 %) |
| Success flag agrees with 1 ms | - | **199/200 (99.5 %)** |
| Failure mode agrees with 1 ms | - | 196/200 (98.0 %) |
| Penetration flags | 2 | 2 |
| p99 contact penetration | 1.92 mm | 1.90 mm |
| Gate G1, seeds 0-7 | 8/8 runs pass (56/56 cases) | 8/8 runs pass (56/56 cases) |
| Episodes/s, 16 workers (loaded) | 6.9 | 11.8 (1.7x) |

Decision: **2 ms** (`physics.timestep: 0.002`). The one flipped outcome is a rod
(`parity_v1/007-rod-interp`, k = 1) that succeeds at 1 ms and ends `unstable_hold`
at 2 ms; the other three mode changes are failures that change label
(`slip_during_lift` <-> `unstable_hold` / `missed_grasp`).

One check changed with this decision (`src/rtt/sim/checks.py`): contact between the
two fingers no longer raises the `penetration` flag. When a grasp fails the fingers
close on nothing and meet at the jaw's 0 mm closed gap; at 2 ms they overlapped
~2.7 mm there (for example the G1 `low_friction` case), which relabelled those
failures by phase (a missed grasp became `ejected_on_close`). The overlap is now
reported as the `jaw_penetration_max` metric. A first parity run made before this
change gave the same 199/200 success agreement, 195/200 failure-mode agreement and
11 vs 5 penetration flags.

## Throughput (2 ms, training suite)

`evaluate_designs` with 8 random valid designs x 12 training conditions x 3
controller settings = 288 episodes per row, on a warm `CpuExecutor` (pool start-up,
about 1 s, excluded). Episodes end when their outcome is decided: 5.42 s of
simulated time on average (2,710 physics steps).

| Workers | Wall (s) | Episodes/s | Physics steps/s |
|---|---|---|---|
| 1 | 187.5 | 1.54 | 4,159 |
| 4 | 62.2 | 4.63 | 12,534 |
| 8 | 41.6 | 6.92 | 18,742 |
| **16** | **26.0** | **11.1** | **30,003** |

All rows ran with a load average of 35-39 from other jobs. Earlier, on a quieter
machine, a single worker ran 2.6-3.3x faster (10,700-13,900 vs 4,159 steps/s); scaling
the measured rows by that factor puts an idle 16-thread run at roughly 30-45
episodes/s. That is an estimate, not a measurement; re-measure on an idle machine or a
Nebius job. One training evaluation of a design (36 episodes) takes ~3.2 s at the
measured 16-worker rate.

Mesh objects cost more: at 1 ms and under the same load, a test mug decomposed into 16
convex pieces ran 1.5-2.2 s per episode (its single convex hull 1.3-1.7 s) against
0.7-1.3 s for the G1 box.

## Manual entrants on `train_v1` (2 ms)

`python -m rtt.sim.bench entrants`: 12 conditions x k = 3 = 36 episodes per entrant.

| Entrant | Score (best k) | k = 0.3 / 0.5 / 0.7 | Families at best k |
|---|---|---|---|
| stock | 1.00 (k = 0.3) | 1.00 / 1.00 / 0.75 | box 1.0, cylinder 1.0, rod 1.0 |
| concave | 1.00 (k = 0.3) | 1.00 / 1.00 / 0.75 | box 1.0, cylinder 1.0, rod 1.0 |
| straight | 1.00 (k = 0.3) | 1.00 / 1.00 / 0.67 | box 1.0, cylinder 1.0, rod 1.0 |
| hooked | 1.00 (k = 0.3) | 1.00 / 0.75 / 0.58 | box 1.0, cylinder 1.0, rod 1.0 |

The training suite is **not yet calibrated**: every entrant saturates, and 23 of 24
random valid designs also score 1.0 (mean 0.997, SD 0.017), far from the G3 targets
(entrants at 30-70 %, Sobol-64 SD >= 0.15). Lowering mu to 0.4 or 0.3 only dropped
the hooked entrant (to 0.75). Calibration (task knobs only: object sizes and
density, mu, F, placements and pose errors) is the D4 work item.

## Robustness-certificate magnitudes

Chosen on the 4 entrants x the 6 centred training objects x k = 0.3 / 0.5 (48
grasps, jitter off, run at the 1 ms timestep before the switch):

| Disturbance | Grasps failed (of 48) |
|---|---|
| None | 1 |
| Shake 10 mm @ 3 Hz (x, y or z) | 2 |
| Shake 10 mm @ 5 Hz, 15 mm @ 6 Hz, 20 mm @ 6 Hz | 1 |
| Push at the object's top, 0.1 N s over 20 ms (x or y) | 1 |
| Push at the object's top, 0.3 N s | 1-2 |
| Push at the object's top, 0.6 N s | 7 (x), 11 (y) |

At mu = 0.8 and F = 50 N the grasps hold far more than these accelerations, so
shakes barely discriminate; a push that also twists the object does. Defaults
(`robustness` in `task_v1.yaml`): shake 10 mm at 5 Hz for 5 cycles (peak commanded
9.9 m/s^2, ~1 g) along x, y and z; push 0.5 N s over 20 ms (25 N) at the object's
top along x and y; placement errors 0, +-1, +-2, +-3 mm along x and y. 18 episodes
per certificate.

## Independent re-check (2026-10-15, same laptop, load average 8-21)

- **Parity reproduced exactly**: `python -m rtt.sim.bench parity --workers 12` gave the
  same 199/200 success-flag and 196/200 failure-mode agreement, the same four
  mismatches, 2 vs 2 penetration flags, p99 penetration 1.92 / 1.90 mm and G1 56/56 at
  both timesteps (8.6 vs 14.2 episodes/s, 1.65x).
- **Throughput at lower load** (`throughput --worker-counts 16 1`, 288 episodes, load
  15 at start): 16 workers 25.8 episodes/s (69,800 steps/s); 1 worker 4.06 episodes/s
  (10,990 steps/s). Idle numbers are still unmeasured.
- **Entrants**: all four again score 1.00 on `train_v1`; 24 other random valid designs
  (seed 99) gave 23 x 1.00 and one 0.92 (mean 0.997, SD 0.017).
- **Certificate magnitudes at 2 ms** (4 entrants x 6 centred training objects x
  k = 0.3 / 0.5, 48 grasps, jitter off): undisturbed 1/48 failed; each default shake
  (x, y, z) 1/48, the same grasp; push 0.5 N s at the top 12/48 (x) and 15/48 (y). The
  push separates designs; the shake does not.

### Mesh objects: contact artifacts on thin or concave parts

8 test meshes decomposed with CoACD x 4 entrants x k = 0..2 x 2 seeds (24 grasps
each), with the task's collision setting (`native_ccd: false`, libccd) and with
MuJoCo's native collider:

| Mesh (pieces) | libccd: success / penetration flags / max depth | native: success / flags / max depth |
|---|---|---|
| bracket, L 60 x 40 mm, 6 mm walls (2) | 12 / 12 / 11.2 mm | 24 / 0 / 0.6 mm |
| cup, 3 mm wall (16) | 24 / 0 / 1.0 mm | 13 / 10 / 11.3 mm |
| mug = cup + handle (16) | 16 / 4 / 3.1 mm | 16 / 4 / 3.8 mm |
| tee, 12 mm bars (2) | 19 / 3 / 5.5 mm | 20 / 1 / 2.3 mm |
| capsule mesh (1) | 14 / 1 / 5.3 mm | 14 / 1 / 2.2 mm |
| box, cylinder, cone meshes (1 each) | 24 / 0 / <= 0.5 mm | 24 / 0 / <= 0.5 mm |

Single convex pieces behave like the primitives. On multi-piece or thin-walled parts a
finger box can sink 2-11 mm into a piece, which raises the `penetration` flag and fails
the grasp; which collider does this depends on the part, so neither setting is safe
for every upload. Flags are failures, so this under-rates parts rather than inflating
them; treat a penetration-flagged mesh grasp as unreliable, not as a design verdict.

## Gate G3 calibration (2026-10-15, CPU executor, 16 workers)

The full record, method and decision are in [TASK.md](TASK.md#calibration-gate-g3). `rtt
calibrate` scores the 4 manual entrants plus 64 Sobol designs (seed 20261018) on the frozen
training suite: 16 conditions x 3 controller settings x 1 seed = 48 episodes per design,
3,264 episodes in total.

| Design | Score | Box | Cylinder | Rod |
|---|---|---|---|---|
| straight | 69 % | 50 % | 100 % | 83 % |
| concave | 75 % | 50 % | 100 % | 100 % |
| hooked | 50 % | 50 % | 50 % | 50 % |
| stock | 69 % | 50 % | 100 % | 83 % |
| Sobol-64 mean | 56 % | 73 % | 85 % | 22 % |

- **Sobol spread:** SD 0.156, range 12.5-93.8 %. 4 of 64 designs beat the best entrant.
- **Gate verdicts:** spread and family gap pass. Entrants-in-range and headroom fail narrowly:
  concave scores 75 % against a 70 % limit, and 4 designs beat it where 5 are required.
- **Throughput:** 3,264 episodes in 104 s, i.e. 31.4 episodes/s, at a load average of about
  15 on the shared laptop.
- **Determinism:** the run reproduced the exploratory run on the same configuration
  bit-for-bit (identical scores for all 68 designs).
- **Timestep check:** after the friction change, gate G1 still passes 7/7 at seeds 0-2 with
  2 ms steps. The deepest entrant penetration is now 1.76 mm (was about 1.1 mm at μ 0.8);
  the flag fires at 2 mm.
