# Use-case benchmark v1 (pre-registration draft) and the Nebius fan-out (track S6)

**Product mode: not part of the pre-registered experiment.** The pre-registered finger experiment
(`prereg-v1`, [PREREG.md](PREREG.md)) is unchanged. This benchmark asks a product question of the
Effector Lab Studio ([STUDIO.md](STUDIO.md), [STUDIO_SEARCH.md](STUDIO_SEARCH.md)) and borrows the
experiment's commit-reveal pattern in a lighter form.

**Status: DRAFT.** Nothing here has run on Nebius. The founder rules on this draft (section 1)
before the tag `usecase-v1`. From the tag on, every change goes in section 12.

| What | Where |
|---|---|
| Parameters, and the frozen block the freeze writes | `configs/usecase_v1.yaml` |
| Driver: freeze, commit, evaluate, analyse, dry run, estimate | `scripts/studio_usecase_v1.py` (wrapper `jobs/studio/usecase_v1.sh`) |
| Fan-out of S5's search onto Serverless Jobs | `src/rtt/studio/cloud.py` |
| Tests | `tests/test_studio_cloud.py` |

## Contents

1. [What the founder approves](#1-what-the-founder-approves)
2. [Question](#2-question)
3. [Heroes and development conditions](#3-heroes-and-development-conditions)
4. [Arms](#4-arms)
5. [Held-out evaluation](#5-held-out-evaluation)
6. [Commit-reveal](#6-commit-reveal)
7. [Endpoints and analysis](#7-endpoints-and-analysis)
8. [Executor, fan-out and determinism](#8-executor-fan-out-and-determinism)
9. [Budget](#9-budget)
10. [Dry run](#10-dry-run)
11. [Procedure after the go](#11-procedure-after-the-go)
12. [Deviations](#12-deviations)

## 1. What the founder approves

Each line is a default set by the orchestrator. The founder accepts or changes each one before the
freeze; none may change after the tag.

| # | Item | Draft value |
|---|---|---|
| 1 | Heroes | the three frozen plans of `src/rtt/studio/engineer/recorded/gate-2026-10-18-s4e/` (H1 `e4586f3612e1`, H2 `33599e99ac27`, H3 `3297a111d7c6`) |
| 2 | Development conditions | S5's grid: 8 full-run conditions (`nominal` and `place01` to `place07`, seeds 0 to 7, aim offsets uniform in +-2 mm), the first 3 screen the steps; search and selection use only these |
| 3 | Arm a | S5's best single off-the-shelf effector of tournament b, on the development conditions |
| 4 | Arm b | S5's recommendation, full pool (family presets, library entrants, designed effectors, part-shaped jaws) |
| 5 | Arm c | S5's recommendation with the pool limited to the library entrants |
| 6 | No recommendation | the arm is its closest setup (S5's `no_winner.closest[0]`), reported descriptively; the hero is excluded from the test. Also excluded: a hero with no arm a, or whose arm b is arm a's setup |
| 7 | Held-out conditions | 32 per hero from a secret seed; each: a seed of at least 1,000,000 (it draws the friction jitter, +-10 %, and the +-0.5 mm aim jitter), and per object a hidden aim offset uniform in +-2 mm on x and y |
| 8 | Commit-reveal | the secret drawn after the arms are frozen, its SHA-256 in the config before the tag, the secret in `secrets/` (git-ignored), revealed with the results |
| 9 | Primary endpoint | per tested hero: full-task completion of arm b against arm a on its 32 held-out conditions; exact one-sided paired sign-flip permutation test (exact McNemar), b > a; Holm across the tested heroes at alpha 0.05 |
| 10 | Secondary (descriptive) | b against c; per-step success with fidelity tiers and failure modes; cycle time; tool count and mass; parity of the development runs |
| 11 | Publication | results are published whatever they show |
| 12 | Executor | Nebius `uks2-cpu`, `cpu-d3` `32vcpu-128gb`; H100 only as fallback; one image for the freeze and the evaluation |
| 13 | Freeze executor | Nebius (one executor for everything), or the laptop (free, the same code); see section 9 |
| 14 | Caps | $10 Nebius over every benchmark job (its own capped ledger), $0 Token Factory |

## 2. Question

Does the Studio's recommended setup complete the hero use cases more often than the best single
off-the-shelf effector, on placements and frictions it was not selected on?

The Studio searches single tools, two-tool heads and tool changes, and recommends a setup from
full-task runs on 8 development conditions (S5). The benchmark freezes each arm's setup, then runs
every arm on 32 new conditions per hero that nobody could see during the search. The held-out
conditions come from the same distribution as the development ones (placement error +-2 mm,
friction +-10 %): this tests whether the selection holds up out of sample, not a distribution shift.

## 3. Heroes and development conditions

| Hero | Use case | Task id | Steps (hold chains) |
|---|---|---|---|
| H1 `h1_tote` | warehouse returns and kitting: open a tote lid, pick a carton and a pack into it, close the lid | `e4586f3612e11b0eb490349c52a109bec75f43fdb0bcf9f1d91127f9e01a4ad5` | 4 (4) |
| H2 `h2_coupling` | pipe coupling make-up: insert the coupling, tighten it | `33599e99ac277cf20e73d6acd349b9422330a255b3fe5ea48c6442a26c3decd6` | 2 (2) |
| H3 `h3_cabinet` | equipment cabinet maintenance: unlatch, open the door, open a valve, plug in a connector | `3297a111d7c61392e9a99a08d3a80a8e14c7046b80d277925ed3a78788ff651d` | 4 (4) |

The plans are read from the recorded gate and checked against these ids
(`tests/test_studio_engineer.py` pins them). Development conditions: S5's defaults
(`run_task_tournament`: 3 screen conditions, 8 full-run conditions, 4 finalists, the top 3 per
chain, 6 two-tool pairs, the 3 kg arm payload, the default controllers).

## 4. Arms

`freeze-arms` runs two S5 tournaments per hero on the development conditions only:

* **tournament b**: the full candidate pool (`default_candidates(library=True, task_spec=...)`:
  every family preset, part-shaped jaws in place of the parallel-jaw entrants, the 19 green library
  entrants, the designed effectors);
* **tournament c**: the same search, the pool limited to the library entrants.

| Arm | Setup | Status |
|---|---|---|
| a | tournament b's `best_single` (the best single off-the-shelf effector, always a finalist) | `tested`, or `missing` when none survived pruning |
| b | tournament b's recommendation | `tested`; with no recommendation, the closest setup, `descriptive` |
| c | tournament c's recommendation | as b |

A hero is in the primary family when a and b are both `tested` and b is not a's setup. This is
decided and written into the frozen block at the freeze, before any secret exists (`primary_rule`).
The frozen block holds, per hero and arm: the setup id, the whole setup (rebuilt and checked
against its id at evaluation), its status and basis, kind, tools, complexity and fidelity, its
development completion rate and the trajectory hash of each development run; and per tournament
the SHA-256 of its report. The freeze runs inside the vault guard (`rtt.store.vault.VaultGuard`).

What the development runs looked like on `main` `ee0fdec` (S5's hero gate 4, 2026-10-19; the freeze
recomputes them on the final code):

| Hero | Arm b on development | Arm a on development | In the family? |
|---|---|---|---|
| H1 | two-tool head (soft arm and 16 mm suction cup), 8/8, approximation tier | 110 mm servo parallel gripper, 0/8 | yes |
| H2 | no recommendation (every finalist stops at the insert) | a stock-blade part-shaped jaw, 0/8 | no: reported descriptively |
| H3 | tool change (three grippers), 4/8 | a cobot linkage gripper, 0/8 | yes |

Two caveats the founder should weigh. First, arm b is chosen for completing the task while arm a
is S5's best single by predicted rank; on development, no single tool completes any hero, because
some step (the lid, the latch) defeats every single tool. The primary test therefore mostly asks
whether b's development completion survives new placements and frictions. Second, H1's
recommendation is approximation-tier (soft arm, suction seal): every result carries its caveat, and
the per-step tiers are reported.

## 5. Held-out evaluation

For hero *h* with task id *t*, the generator (`heldout_conditions`) seeds a PCG64 stream with
`sha256("rtt-usecase-v1|<t>|<secret>")` and draws, for `heldout00` to `heldout31`: a seed uniform in
[1,000,000, 2^31 - 1) (distinct within the hero), then for every object in task order an aim offset
uniform in [-2, 2] mm on x and on y (rounded to 0.001 mm). The seed draws S5's v1 jitter as on the
development grid: friction scaled by 1 +- 10 %, and +-0.5 mm on every aim. The offsets are hidden
from the controllers (`StepContext.observe_site`), as on the development grid. Every arm of a hero
runs the same 32 conditions in the same order (common random numbers). The development seeds are
0 to 7, so no held-out condition repeats one.

Every arm's setup also re-runs its 8 development conditions in the same job (**parity runs**): their
trajectory hashes are compared with the frozen ones (descriptive; it checks that the job runs the
frozen code and physics).

A run that fails any way (a criterion, a limit, a slip, `physics_invalid`, `physics_unreliable`,
an invalid build) is not a completion. Nothing is rerun except after an infrastructure failure
(section 12).

## 6. Commit-reveal

A lighter form of [PREREG.md](PREREG.md) A6:

1. `freeze-arms` fixes every arm (setup ids in the config). No secret exists yet.
2. `commit-secret` draws the secret with `secrets.token_hex(32)` into `secrets/usecase_v1_seed.txt`
   (git-ignored, mode 600, never replaced; `HeldoutVault.create`) and writes its SHA-256 (of the
   string without the newline, `seed_sha256`) into the frozen block. It refuses before the arms are
   frozen.
3. `finalize` checks the frozen block and the secret against the commitment, then sets
   `status: frozen`; the commit is tagged `usecase-v1`.
4. `evaluate` refuses unless the tag exists and the config equals the tagged copy, and reads the
   secret through the vault against the commitment. **The secret never leaves the laptop**: the jobs
   receive the derived conditions as work items. No job gets any secret or the Token Factory key.
5. The results folder holds the revealed secret (`usecase_v1_seed.revealed.txt`) and the
   conditions; `analyse` checks the secret against the commitment and that it regenerates exactly
   the recorded conditions.

## 7. Endpoints and analysis

**Primary, per tested hero.** Full-task completion rate of arm b against arm a on the hero's 32
held-out conditions. Test: the exact one-sided paired sign-flip permutation test of the
per-condition differences, which on binary outcomes is the exact McNemar test: with *n10*
conditions where only b completes and *n01* where only a does,
p = P(Binomial(n10 + n01, 1/2) >= n10) (p = 1 when there is no discordant condition). The pairing
is the design (both arms run every condition), so the paired test is used rather than Fisher's
unpaired test. **Multiplicity:** Holm across the tested heroes, each rejected when its adjusted
p-value is at most 0.05. With 32 conditions, the smallest possible p-value is 2^-32; one hero needs
at least 5 discordant conditions, all for b, to reach p <= 0.05 alone (p = 1/32).

**Secondary, descriptive** (no test, not in the family):

* arm b against arm c: completion rates, their difference, and the discordant counts;
* per arm and step: success rate, fidelity tiers and failure modes;
* cycle time: the median of the completed runs, and of all runs;
* tools, actuators, the carried and total mass (the measured carried stack per run);
* `physics_invalid` and `physics_unreliable` counts;
* parity of the development runs (section 5);
* each arm's completion with its exact (Clopper-Pearson) 95 % interval.

**Checks before any test** (`pairing_problems`): every arm ran exactly the hero's held-out
conditions in order, with the default controllers and one fairness hash. A tested hero that fails a
check leaves the family, with the reason, and is reported descriptively.

`analyse` writes `analysis.json` and `summary.txt`. It was tested on synthetic rows
(`tests/test_studio_cloud.py`: known p-values, Holm, the declared exclusions, a pairing problem).
**Results are published whatever they show.**

## 8. Executor, fan-out and determinism

**Executor of record:** Nebius Serverless Jobs, `uks2-cpu` (`cpu-d3` `32vcpu-128gb`, uk-south2),
CPU MuJoCo, one image built from the freeze commit (`build_image.sh --ref HEAD`) and pushed to both
registries. The H100 (`eun1-gpu`) is used only when uk-south2 cannot start a job; the job then runs
the same digest (`check_same_image`).

**Fan-out** (`rtt.studio.cloud`). S5's search does all its physics through one pool's `map`.
`CloudPool` stands in for S5's `WorkerPool`, so `run_task_tournament(task, pool=CloudPool(...))`
runs S5's search unchanged:

| Unit | What it is |
|---|---|
| phase | one `map` call: one hold chain of the screen (each chain starts from the state the best tool of the previous chain left, so phases run in order), or the finalists' runs, or the evaluation |
| shard | one candidate on one hold chain (its 3 screen conditions), or one setup under one condition (a finalist, held-out or parity run); the exact work items, pickled |
| job | `python -m rtt.studio.cloud shards --phase-dir /out/runs/<run>/<phase> --job jNN --shards ...` on its own process pool; one job per phase on Nebius (provisioning dominates), several in the fake cloud |

Layout in the bucket (and in the local stand-in): `runs/<run_id>/<phase>/phase.json` (the shards,
their items in JSON and the SHA-256 of each input), `inputs/<shard>.pkl`, and per job
`jobs/<jNN>/shards/<shard>.{pkl,json}` and `manifest.json`, written last through
`rtt.cloud.entry`'s staging (the bucket mount rejects seeks and renames). Each job records its CPU
seconds and the runs' wall times.

**Merge.** Each result goes back to its item's index after three checks: the result file's SHA-256
against the job manifest, the count per shard, and each result's task, setup, condition and seed
against its item. A tampered or misplaced result is refused (tested).

**Records.** Every run is a row: its `TaskResult.to_record()` plus `cloud`: the job id (which keys
that job's `JobSpend` line in the spend ledger, the existing format), the platform, preset, region
and price, and `est_usd`, the run's share of the job's wall-based estimate (split by the runs' own
wall times). The benchmark's jobs go to their own capped ledger (`runs/cloud/usecase-v1-spend.jsonl`,
$10) and every line is also appended to the project ledger (`runs/cloud/spend.jsonl`).

**Executors.** `SubprocessExecutor`, the local fake cloud, runs the job entry point in subprocesses
(at most 4 at once) against a local folder that stands for the bucket, with the arguments a Nebius
job gets. `NebiusExecutor` uploads each phase to the region's bucket, submits each job through
`NebiusJobs.run` (capacity and quota checks, the provisioning timeout, the ledger's cap before
submission and while running), moves to the H100 under the same conditions as `python -m rtt.cloud
run`, fetches and verifies each job's output. At most 3 jobs are in flight (96 of the 200 vCPU
quota). It is tested against a scripted CLI that executes each job on local folders.

**Trust boundary.** Work items and results are pickles (multiprocessing's pickler, so part-shaped
jaws restore their build path as in S5's spawned pool), written and read only by this code in the
project's private bucket. Every file's SHA-256 is checked before loading, and the unpickler resolves
only Studio, effector and simulator classes (and the part-jaw restore hook).

**Determinism (G3).** The fan-out runs S5's own functions on S5's own objects, so a sharded run gives
the local run's `traj_sha`:

* tests: S5's whole tournament through the fake cloud (two screen phases and the finalists, 2 jobs per
  phase, job subprocesses) equals the local pool's report field for field, every `traj_sha`
  included; real S2 primitives on a hero task give the same hashes in-process twice and through job
  subprocesses; the scripted Nebius path gives the in-process hashes;
* the dry run (section 10): tournament b of H1 through the fake cloud against S5's hero-gate report
  of the same code.

Laptop and Genoa are different CPUs: the parity runs measure whether the hashes also agree across
machines (in GV4 most v2 records did, the rest to 1e-15 mm). A parity mismatch is reported, not a
failure; the held-out results are what the job computes.

## 9. Budget

`scripts/studio_usecase_v1.py estimate` builds the table from the dry run's measured seconds per
run, the run counts S5's pruning gives for each hero and pool (no physics), and the uk-south2 timings
of [NEBIUS.md](NEBIUS.md). Price: `cpu-d3` `32vcpu-128gb` at $1.056 per hour [sourced].

| Stage | Jobs | Runs | CPU-s | Wall, one after another | Est. $ (wall-based) | Est. $ (run-only) |
|---|---|---|---|---|---|---|
| development search: `freeze-arms --executor nebius`, tournaments b and c of the three heroes (one job per hold chain, one per finalist phase) | 26 | 1,614 | 5,765 | 65 min | 1.144 | 0.229 |
| held-out evaluation: 9 arms x (32 held-out + 8 parity) runs, one job | 1 | 360 | 3,093 | 4 min | 0.075 | 0.040 |
| contingency: two attempts cancelled at the 15 min provisioning timeout | 2 | 0 | 0 | 32 min | 0.563 | 0 |
| image build from the commit (cached layers) and push to both registries | 0 | 0 | 0 | 3 min | 0 | 0 |
| **Total** | | | | | **1.78** | |

How the rows are built:

* **Seconds per run** [measured, dry run]: 2.89 s per screen run (501 runs) and 8.59 s per full-task
  run (84 runs): each run's own wall time in a one-process job on the laptop (Ryzen 7 5800H, 16
  threads, shared, load about 12). In GV4 a Genoa vCPU was about 1.3 times faster than this laptop,
  so these are upper bounds.
* **Runs** [derived]: S5's pruning, without physics, gives the screen runs per hold chain: tournament
  b 351 (H1), 240 (H2), 438 (H3), exactly S5's hero-gate counts; tournament c 150, 78 and 165; plus 4
  finalists x 8 conditions per tournament. The evaluation counts 3 arms per hero; arms that share a
  setup run once, so 360 is an upper bound.
* **Job wall** [derived and assumed]: 120 s from creation to container start (111 to 119 s measured
  on uk-south2, [NEBIUS.md](NEBIUS.md)), 15 s of container overhead [assumed], then the phase's CPU
  seconds over 32 vCPUs at 80 % efficiency [assumed], at least 15 s (the slowest single run). The
  wall-based estimate is the upper bound the cap is enforced on; the run-only one leaves out
  provisioning.
* **Image** [measured, NEBIUS.md]: 15 s to build from a commit with cached dependency layers, 2 min
  36 s to push to both registries; no compute is billed for either. Object storage for a few MB of
  runs is well under $0.01.
* **Token Factory**: $0. No model is called.

The development search can run on the laptop instead (`--executor local`, about 30 min with 4
processes, $0 Nebius): the total then drops to about $0.64. Either way the evaluation, the result
of record, runs on Nebius.

**Caps.** $10 Nebius over every benchmark job (the benchmark ledger's cap, enforced before each
submission and while a job runs, across processes), $0 Token Factory (no job attaches the key;
`studio_job_spec` has no secret, tested; the hero plans are recorded and the designed effectors are
replayed from the recorded morphology run).

## 10. Dry run

Run on 2026-10-19 on `main` `ee0fdec` plus this track's files (uncommitted at the time, so the
frozen block says `git_dirty: true`; the real freeze refuses a dirty tree), with the local fake
cloud (4 job subprocesses, 1 process each), a throwaway seed in a scratch folder (never the real
secret), H1 only and 2 held-out conditions:

```bash
scripts/studio_usecase_v1.py dry-run --hero h1_tote --conditions 2 --out <scratch>/dry \
    --reference <S5 hero gate 4>/h1_tote/report.json
scripts/studio_usecase_v1.py estimate --dry <scratch>/dry
```

| Step | Result |
|---|---|
| freeze, tournament b | 425 s; 5 phases (4 hold chains, the finalists), 20 jobs, 383 runs (351 screen, 32 finalist), 1,523 job CPU-s |
| G3 against S5's hero gate 4 report (same code, S5's own 6-process pool) | identical: all 32 finalist `traj_sha`, every screen cell, the combinations, the recommendation and the best single |
| freeze, tournament c (19 library entrants) | 146 s; 150 screen runs; no setup completes (nothing completes without opening the lid) |
| frozen arms | a: the 110 mm servo parallel gripper (`d136ab4824c1`, 0/8 on development, rigid); b: the two-tool head, soft arm and 16 mm suction cup (`9d7829e0d886`, 8/8, approximation tier); c: no recommendation, closest setup the same servo gripper (descriptive); H1 in the family |
| commit-secret | throwaway secret and its commitment written to the scratch config |
| evaluate | 20 unique runs (a and c share a setup) for 30 rows (2 held-out + 8 parity per arm), 4 jobs, 58 s |
| analysis | a 0/2, b 2/2, c 0/2; primary b > a: 2 discordant conditions, p = 0.25, Holm 0.25, not rejected (two conditions cannot reject); parity 8/8 for every arm; no pairing problem; the revealed seed matches its commitment and regenerates the conditions |
| records | every row carries its job id, the job's ledger line (`JobSpend`) and its dollar share; 44 jobs in all, 2,204 job CPU-s |

The dry run is an infrastructure check on a throwaway seed. Its numbers are not a result.

## 11. Procedure after the go

From the repository root, on `main` with every track that changes physics merged (S2d's templates
and primitives in particular), a clean tree, and a fresh Nebius CLI session (it lasts about 12 hours).

```bash
# 1. image from the freeze commit, pushed to both registries (digest saved to ~/.config/rtt/image.env)
make image REF=HEAD && make push
python -m rtt.cloud capacity                                  # uk-south2 vCPU quota (needs 96 free)

# 2. freeze the arms: tournaments b and c of every hero on the development conditions
jobs/studio/usecase_v1.sh freeze-arms --executor nebius      # or --executor local (free, about 30 min)

# 3. draw the secret (secrets/usecase_v1_seed.txt) and commit its SHA-256, then check and freeze
jobs/studio/usecase_v1.sh commit-secret
jobs/studio/usecase_v1.sh finalize
scripts/leak_check.sh && git add configs/usecase_v1.yaml docs/USECASE_V1.md
git commit -m "Use-case benchmark v1: frozen arms and the held-out commitment"

# 4. the founder approves the frozen block; tag (and push the tag when the founder says so)
git tag -a usecase-v1 -m "Use-case benchmark v1 of the Studio: pre-registration"

# 5. submit, wait, fetch, merge and analyse: the held-out and parity runs, one cpu-d3 job
jobs/studio/usecase_v1.sh evaluate --executor nebius --out runs/usecase_v1/results
#    recovery only, if the laptop lost the job: jobs/studio/usecase_v1.sh fetch --run-id <run_id> --out DIR

# 6. re-run the analysis from the results folder at any time
jobs/studio/usecase_v1.sh analyse --results runs/usecase_v1/results
python -m rtt.cloud spend --ledger runs/cloud/usecase-v1-spend.jsonl
```

Then publish `runs/usecase_v1/results/{analysis.json,summary.txt,rows.jsonl,conditions.json,usecase_v1_seed.revealed.txt}`
and the development reports, whatever they show.

## 12. Deviations

As [PREREG.md](PREREG.md) A11, lighter: only infrastructure failures (a job that never started, a
lost fetch, a failed upload) are rerun, with the same inputs and a new run id, and the failed job's
id is logged here. A result is never dropped. Any change after the tag is logged here, with the
reason, before the affected result is looked at where possible.

| Date | Change | Reason |
|---|---|---|
| | none yet | |
