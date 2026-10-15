# MJWarp spike: gate G3 criteria (a) to (e)

**Verdict: MJWarp does not become the executor of record. The `cpu` executor (MuJoCo
3.15.0, one process per episode) stays the executor of record.** Criterion (c)
(determinism) fails, (b) (parity) fails on the ranking clause, and (d) (batch
invariance) failed in the independent re-run (3 / 24 success flips). All three trace back to
the same two properties of MJWarp 3.15.0 that the spike could not change: its
physics is not bit-reproducible run to run, and its float32 contact handling differs
from the CPU's on the hardest grasps. Per plan section 3.6, the GPU executor is kept
only as a reported cross-check on finalists, and GPU numbers that disagree with the
CPU are never published.

| Criterion | Threshold | Measured (final run) | Verdict |
|---|---|---|---|
| (a) Per-world geometry | different designs in one model give the right, different contact widths; `geom_aabb` / `geom_rbound` per world | 10.0 / 30.0 mm wide fingers touch over 10.000 / 30.000 mm (CPU: 10.000 / 30.000 mm); per-world `geom_aabb` and `geom_rbound` equal the MuJoCo compiler's and the box formula | **pass** |
| (b) Parity, 288 episodes | ≥ 95 % success-flag agreement **and** same entrant ranking | 96.9 % outcome agreement (279 / 288), 97.2 % success agreement; ranking differs: CPU ties all four entrants at 1.000, GPU drops `hooked` to 0.917. Re-run: 96.5 % success agreement (278 / 288), same ranking split | **fail** |
| (c) Determinism, 3 same-seed runs | ≤ 1 % success-flag disagreement | 1.04 % outcome disagreement (3 / 288 in each rerun pair); 0 / 288 episode summaries bit-identical. Two independent re-runs: 1.39 % success-flag disagreement (4 / 288) in the worst pair | **fail** |
| (d) Batch invariance | identical success flags alone vs in a batch of ≥ 288 worlds | 24 / 24 identical alone vs inside 288 worlds, 3 / 3 single-world batches identical; an earlier run had 1 / 24 differ. Independent re-runs: 24 / 24, then **3 / 24 differ** (single-world batches 3 / 3 both times) | **fail**: not reproducible; holds only within the (c) noise |
| (e) Throughput incl. JIT | ≥ 3× the CPU executor | warm GPU vs the 16-process CPU pool on the same episodes: 2.0× (96 worlds), 2.9× (288), 6.5× (960); re-run at lower load: 2.1× / 4.4× / 6.2× (48 / 100 / 140 vs 22.5 ep/s); a cold kernel cache adds ~87 s of JIT once per process | conditional: met at 960 worlds only with a persisted kernel cache; not met including a cold JIT, and not yet measured on the Nebius job size |

Machine: laptop, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, sm_86, driver 595.99,
CUDA 13.2), AMD Ryzen 7 5800H (8 cores, 16 threads), mujoco 3.15.0, mujoco_warp
3.15.0, warp-lang 1.18.0. Task `task_v1`, digest `9fb0d88d9a04` (2 ms timestep,
Newton 100 / 50 iterations, elliptic cones, condim 4, F = 50 N, 84 mm opening). The
laptop was shared with other build jobs: 1-minute load average 17 to 30 during the
final (a) to (d) run, 10 to 20 during the throughput run in (e), up to 50 in earlier runs. Raw JSON: `python -m rtt.sim.exec_mjwarp all --out <file>`.

## What was built

`src/rtt/sim/exec_mjwarp.py` is a complete second executor, not a benchmark stub:
`run_jobs(jobs, workers)` takes the same `EpisodeJob` list as `exec_cpu` and returns
the same `EpisodeResult` per job, so `rtt.sim.evaluate` can use `executor="mjwarp"`.

- **Same model.** Each world's scene is compiled by `rtt.sim.scene.build_model`
  (the CPU executor's builder, unmodified). Fields that differ between worlds are
  uploaded as MJWarp per-world batched fields; in a box batch with different
  designs these are `geom_size`, `geom_pos`, `geom_quat`, `geom_aabb`,
  `geom_rbound`, `geom_friction` (the jittered μ), `body_mass`, `body_inertia`,
  `body_ipos`, `body_iquat`, `body_subtreemass`, `body_invweight0`,
  `dof_invweight0`, `tendon_invweight0`, `actuator_acc0` and `stat.meaninertia`.
  Every other field MJWarp reads must match across the batch or the build raises
  `BatchIncompatible`; the AABB and bounding radius are therefore the compiler's own
  per-world values, not a separate recomputation. One batch per object family
  (different geom types cannot share a model) and per mesh object.
- **Same controller.** A Warp kernel evaluates every world's `GraspPlan` in float64
  at the CPU executor's exact `data.time` for each control tick (minimum-jerk
  descent and lift, grip ramp, and the robustness shake); a push goes through
  `xfrc_applied` on the same ticks as on the CPU, as its full `Push.wrench`: the
  force and, for the certificate's push at the object's top, its torque about the
  centre of mass (see "Independent verification" below).
- **Same scoring.** Every 10 ms check a kernel reduces each world's active contacts
  (deepest non-jaw penetration and whether it is gripper-environment, jaw
  penetration, gripper-environment and palm-object normal force, object support,
  each finger's net normal force vector on the object) and records them with the
  object's pose and speed. `SeriesScorer` replays the samples through a vectorised
  port of `run_episode`'s check loop and labels each world with
  `rtt.sim.score.classify`. Tested without a GPU: stepping CPU MuJoCo and scoring
  it through `SeriesScorer` reproduces `run_episode` exactly (outcome, flags, step
  count and every metric) on box, cylinder, rod, pose-error, shake, push and the G1
  failure cases (`tests/test_exec_parity.py`).
- **Lockstep batches.** One check period (5 control ticks of physics, the contact
  reduction and the sample) is captured as a CUDA graph and replayed; the host
  pulls samples every 50 checks and stops the batch when every world is decided.
  Contact and constraint capacity is reserved per world (128 contacts, 512 rows);
  a batch whose contact buffers overflow, or whose world overflows its rows before
  its own decision, is rerun with doubled capacity.

## (a) Per-world geometry: pass

Two fingers from the `straight` entrant, 10 mm and 30 mm wide, squeeze a 60 × 40 ×
100 mm box (no jitter) in two worlds of one batched model. At the start of LIFT the
left finger's contacts with the box span 10.000 mm and 30.000 mm along the finger
width on the GPU, and 10.000 mm and 30.000 mm on the CPU. The uploaded per-world
`geom_aabb` and `geom_rbound` equal both the MuJoCo compiler's values for each
world's model and the box formula (centre 0, half sizes; radius = |half sizes|).

## (b) Parity: fail (ranking)

Parity set: 4 manual entrants (`straight`, `concave`, `hooked`, `stock`) × the 12
training conditions (`configs/suites/train_v1.yaml`) × 3 controller settings × 2
jitter seeds = 288 episodes, each on both executors.

| | CPU | GPU |
|---|---|---|
| Success rate | 86.8 % | 85.4 % |
| Outcome agreement (success + failure mode) | | 96.9 % (box 94.8 %, cylinder 96.9 %, rod 99.0 %) |
| Entrant scores (best setting's mean success) | all 1.000 | `hooked` 0.917, others 1.000 |

The flag-agreement clause passes; the ranking clause fails. The CPU ties all four
entrants, so any single GPU-only failure in an entrant's best setting changes the
ranking; that fragility is itself a finding (the scores need the calibration of
gate G3 to separate the entrants), but the GPU-only failures are real physics
differences, not scoring noise:

- 9 of 9 mismatches are the hard cases: 8 at the offset placement (35° yaw plus a 3
  mm / 8° pose error) and 6 with the `hooked` finger, whose bent tip is made of
  rotated boxes that touch the object edge-on.
- 6 of 9 are GPU-only failures preceded by a penetration flag (2.1 to 5.7 mm) that
  the CPU never raised, and one of them is a tunnelling blow-up: the hooked tip
  reported 589 mm of penetration into a box during DESCEND.
- 1 of 9 fails on both executors with a different label (`unstable_hold` after an
  `off_table` flag on the GPU, `slip_during_lift` on the CPU).
- 2 of 9 go the other way (the GPU lifts a cylinder that the CPU misses).

Likely causes, in order: float32 contact solving at a stiff contact timescale
(solref 0.004 s at a 2 ms step); MJWarp's own box-box and GJK/EPA collision instead
of the CPU's box-box and libccd paths (`native_ccd: false` has no effect in MJWarp);
and the run-to-run variation of (c), which alone flips about 1 % of outcomes.

## (c) Determinism: fail

The parity set was run three times on the same GPU with the same seeds: 3 of 288
outcomes differ between run 1 and run 2, and 3 of 288 between run 1 and run 3
(1.04 %, threshold 1 %; an earlier run of the same measurement gave 1.7 %). No
episode summary was bit-identical between runs.

Root cause, measured on 48 box worlds (two identical runs): the physics state
already differs in the last bits after the first 10 ms check (max |Δqpos| 1e-10);
by t = 4 s the difference has grown to 0.18 m in object height in some worlds.
MJWarp builds contacts and constraint rows with atomic counters and reduces them
with atomic adds, so their order, and hence float rounding, changes from run to run;
grasping is chaotic enough near the decision thresholds to turn that into outcome
flips. The executor's own contact-reduction kernels also sum with atomics, but they
only feed the scoring thresholds and cannot cause the state divergence. The flips concentrate in a few near-chaotic episodes: for the
`straight` design on boxes, 3 of its 12 offset-placement episodes flipped at least
once in 5 reruns, and none of its 12 centred ones.

Tried and rejected within the time box: Warp 1.18's
`wp.config.deterministic = RUN_TO_RUN` (sort-based deterministic atomics). With the
`smooth` module's record bound raised to 8 it compiles and makes the first second
bit-identical (0 / 48 worlds differ after 100 checks), but by t = 4 s every world
diverges and some fly off to ~1.5 km, i.e. some data-dependent atomic loop is being
truncated. A global record bound of 256 instead runs out of the 6 GB of GPU memory.
Making this work means auditing every MJWarp kernel's atomic bounds; that is
upstream work, not a spike.

## (d) Batch invariance: fail (not reproducible)

One design (`straight`) on the box family, 4 conditions × 3 settings × 2 seeds = 24
episodes, was run as its own 24-world batch and again inside a 288-world batch padded
with 264 Sobol designs. Final run: 24 / 24 outcomes identical, and the first 3 also
as 1-world batches: 3 / 3 identical. An earlier run of the same test had 1 / 24
differ. MJWarp's solver iterates each world to its own convergence and its
collision filter depends only on topology, so batch composition should not matter;
what remains is the run-to-run variation of (c), since the other worlds in a batch
change the order in which contacts and constraint rows are assembled. Invariance
therefore holds to within the (c) noise floor, not bit for bit. An independent re-run of the same
measurement had 3 / 24 success flags differ, so the criterion fails.

## (e) Throughput: conditional (960 worlds with a persisted kernel cache only)

Same box-family training episodes for both executors (Sobol designs × 4 conditions ×
3 settings). The GPU wall time includes the per-world scene compiles on the host, the
upload, the simulation and the host-side scoring. "First" is the first batch of that
size in the process (kernel cache warm on disk); "warm" is a second identical batch.
Measured in the quietest window (load average 10 to 20).

| Batch | GPU first | GPU warm | Wall (warm) | Split (build / upload / sim / score), warm | vs CPU pool |
|---|---|---|---|---|---|
| 96 worlds | 24.8 ep/s | 45.6 ep/s | 2.1 s | 0.2 / 0.1 / 1.6 / 0.1 s | 2.0× |
| 288 worlds | 78.5 ep/s | 64.6 ep/s | 4.5 s | 0.5 / 0.1 / 3.4 / 0.3 s | 2.9× |
| 960 worlds | 127.1 ep/s | 146.2 ep/s | 6.6 s | 1.5 / 0.3 / 3.9 / 0.7 s | 6.5× |
| CPU, 16 processes | 22.4 ep/s (960 episodes in 42.8 s) | | | | 1× |

- Contention-free CPU check: one episode costs 268 ms of CPU time run serially at
  load 11 (567 ms at load 41, so contention roughly doubles it). That bounds 16
  logical CPUs at 16 / 0.268 = 59.7 ep/s with perfect SMT scaling, or about 37 ep/s
  for 8 cores with ~25 % from SMT. Against those the 960-world batch is 2.4× to
  3.9× faster: a pass against the realistic estimate, not against the upper bound.
- An earlier, more loaded run (load 17 to 30) gave 40.6 / 65.8 / 102.9 ep/s at 96 /
  288 / 960 worlds against 13.0 ep/s for the CPU pool (3.1× / 5.1× / 7.9×).
- Cold kernel cache (`WARP_CACHE_PATH` pointing at an empty directory, as in a fresh
  job without a persisted cache): the first 960-world batch took 95.1 s, of which
  ~87 s was JIT (81.5 s while uploading and running the first forward pass, 5.8 s in
  the first check period), i.e. 10.1 ep/s, below the CPU pool. The second batch took
  7.0 s (136.9 ep/s). The compiled cache is 20 MB, so persisting it (the plan's
  `WARP_CACHE_PATH` on the bucket) makes the JIT a one-off; otherwise a job needs
  about a dozen 960-world batches before it beats the CPU pool by 3×.
- A batch runs until its slowest world is decided (6.0 s simulated here), and
  decided worlds keep stepping, so about 8 % of world-steps are idle.
- Other start-up costs, once per process: importing `mujoco_warp` (5 to 20 s on the
  loaded laptop) and loading kernels for each new specialisation (object family,
  batch size, capacity) from the cache, about 4 s each.
- At 960 worlds the host side (scene compiles 1.5 s, scoring 0.7 s) is a third of
  the wall time and single-threaded, so the GPU path still has headroom.

## Independent verification (adversarial re-run)

An independent verifier re-ran the tests and the benchmark on the same laptop
(task digest `9fb0d88d9a04`, load average 3 to 13) and read the executor against
`run_episode`, `PhysicsChecks`, `LiftTracker` and the perturbations.

- **Defect fixed: push torque.** `Push.wrench` returns the force *and* its torque
  about the centre of mass, and the robustness certificate pushes at the object's
  top (`robustness.push_at_top`, a 0.875 N·m torque for a 25 N push 35 mm above
  the CoM). The GPU controller kernel applied only the force. On 48 certificate push
  episodes (4 entrants × 2 boxes × 2 axes × 3 settings, jitter off) CPU and GPU
  agreed on 30 / 48 outcomes before the fix and 47 / 48 after it. The plan row now
  carries the torque (`P_PUSH_TORQUE`). `test_exec_parity.py` has a `push_top`
  replay case (a push 40 mm above the CoM) and a GPU test that reads
  `xfrc_applied` inside and after the push window against `Push.wrench`.
- **Gate metrics.** Plan sections 3.5 and 3.6 state (b), (c) and (d) on the success
  flag. The benchmark now judges them on it (`success_agreement`,
  `success_mismatches`) and still reports the stricter outcome agreement (success
  plus failure mode). No verdict changed.
- **Re-measured.** (a) 10.000 / 30.000 mm, pass. (b) 96.5 % success agreement
  (10 mismatches: 8 GPU-only failures, each after a GPU-only penetration flag,
  two of them 554 mm and 589 mm tunnelling blow-ups; and 2 hooked-finger cylinder
  grasps that only the GPU lifts); ranking split as above, fail. (c) 1.39 %
  success-flag disagreement in two runs, fail. (d) 24 / 24, then 3 / 24 differing
  alone vs inside 288 worlds, fail. (e) 48.2 / 99.9 / 140.3 ep/s warm at 96 / 288 /
  960 worlds vs 22.5 ep/s for the 16-process CPU pool.
- **Checked and found correct.** Per-world geometry: after `forward`, every geom's
  world position and orientation on the GPU equals CPU `mj_forward` of that world's
  own model to float32 precision (max 2e-8 m, 3e-7 in rotation) for all four
  entrants, including the hooked finger's rotated boxes, on all three families.
  Mesh objects, which had not run on a GPU before: 24 / 24 episodes (two convex-hull
  meshes × 4 entrants × 3 settings) agree with the CPU, one batch per mesh. The
  phase boundaries, check order, flag-to-mode labels, palm-contact rule, lift
  tracker and stopping rule match `run_episode` line for line, and the shake, push
  and new `push_top` cases replay exactly through `SeriesScorer`.

## Recommendation

- Executor of record: `cpu`, per plan section 3.6. Write it in `docs/PREREG.md`.
- Nebius: run scored jobs on the CPU executor. Plan Addendum A ("honest compute")
  asks for CPU job presets when CPU MuJoCo is the executor of record, but CPU-only
  jobs have zero quota in eu-north1. Either get CPU quota there (or a bucket in
  uk-south2, which has CPU quota), or, as a fallback, keep the `1gpu-16vcpu-200gb`
  preset and use its 16 vCPUs for `exec_cpu`. On that preset the GPU is billed
  (about $4.50 per GPU-hour, `docs/NEBIUS.md`) while it sits idle or runs the
  cross-check below, so say so in the write-up.
- Keep `exec_mjwarp` as the reported cross-check on finalists (Kendall τ of the
  ranking under the other executor, plan section 3.5), clearly labelled; it agrees
  with the CPU on about 97 % of outcomes.
- If revisited: (1) Warp deterministic atomics with audited per-kernel record
  bounds; (2) rerun (b) after calibration, when the entrants no longer tie; (3)
  move scene compiles to a process pool and park decided worlds to cut the idle 8 %.

## Reproduce

```bash
source .venv/bin/activate
python -m rtt.sim.exec_mjwarp all --out runs/bench/mjwarp.json      # (a) to (e), ~15 min on the laptop (runs/ is gitignored)
python -m rtt.sim.exec_mjwarp parity determinism                    # just (b) and (c)
WARP_CACHE_PATH=$(mktemp -d) python -m rtt.sim.exec_mjwarp throughput --sizes 960 --cpu-episodes 0  # cold cache
pytest tests/test_exec_parity.py tests/test_batch_invariance.py     # GPU tests skip without CUDA or with RTT_SKIP_GPU_TESTS=1
```
