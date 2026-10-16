# Pre-registration (DRAFT, not frozen)

**Status: draft for approval. Not tagged.** This file becomes the protocol of record when
the commit that holds it is tagged `prereg-v1`. The tag must precede the first
confirmatory job. Until then every number here may change; after it, every change goes in
`docs/DEVIATIONS.md`.

**Scope.** This draft covers the v1 design space: 14-box fingers, `task_v1`, `train_v1`.
A richer design space is in development. If it passes its gates in time, a v2
pre-registration replaces this file before any confirmatory run. If not, this protocol
runs as written.

Items marked **[decide]** need a decision before the tag. Items marked **[at freeze]** are
filled in by the freeze steps at the end.

---

## 1. Question and primary endpoint (S1)

**Question.** Does NVIDIA Nemotron 3 Super, reading its own run's simulator results, reach
a good gripper-finger design in fewer simulated designs than random search with the same
budget?

**S1: evaluations to tau.**

- Each run simulates exactly B = 40 valid designs, numbered 1 to 40 in evaluation order.
  Designs 1-8 are the shared round 0: the four manual entrants, then a Sobol fill seeded
  only by the run seed. For a given seed, round 0 is byte-identical in every arm.
- A design's **training score** is the success rate of its best controller setting over the
  16 conditions of `train_v1` (48 episodes per design; `rtt.sim.evaluate.score_designs`).
  Scores lie on a 1/16 grid.
- **T** is the index of the first design **proposed by the arm** (index 9 or later) whose
  training score is at least tau. A run that never gets there is censored at T = 41.
- Round-0 designs never count toward T. They are the same in both arms, so they say
  nothing about the arm. (With T counted from design 1, any seed whose round 0 already
  holds a tau-level design gives both arms the same T.) The arm may still use round 0:
  arm C reads its scores, which is the point of feedback.

**tau = 13/16 = 0.8125** (provisional, by the rule below; confirmed by the Nebius pilot at
freeze).

**Hypotheses.** H0: the arms' T values are exchangeable. H1: arm C needs fewer
evaluations than arm A.

**Test.** One-sided exact permutation test of the restricted mean of T, alpha = 0.05.
Every relabelling of the 2R runs into two groups of R is counted. With equal arm sizes
the statistic mean(T_A) - mean(T_C) is a function of the arm-C sum, and T is an integer,
so the exact null distribution is a subset-sum count (`permutation_test` in
`src/rtt/analysis/s1.py`). Ties count against H1.

Why this test:

- All censoring is administrative, at the same budget for every run, so the restricted
  mean is unbiased and reads directly as "designs saved".
- It needs no proportional-hazards assumption and no large-sample approximation, which
  matters at R = 12.
- A permutation log-rank test is reported as a sensitivity analysis (section 6).

**Effect estimate.** mean(T_A) - mean(T_C) with a 95 % percentile bootstrap interval
(10,000 resamples of runs within each arm, seed 0; `restricted_mean_ci`). Also reported:
the share of runs that reached tau by each evaluation count, per arm (`reach_curve`).

### The tau rule (fixed before the pilot)

1. Pool every random-arm design score available before the confirmatory runs: the
   Sobol-64 calibration sample (`runs/calibrate/g3-2026-10-15/calibration.json`) and every
   non-entrant design of the three random pilot runs.
2. Bootstrap 20,000 random runs of B = 40: the four entrants' scores plus 36 scores drawn
   with replacement from the pool. Best-at-B is the maximum (seed 20261016).
3. tau_raw is the median of the bootstrapped best-at-B. tau is tau_raw rounded to the 1/16
   grid, halves up.
4. tau is raised, if needed, to one grid step above the best manual entrant (the fixed
   part of round 0), and capped at 1.
5. The Nemotron pilots never enter the rule; they are used only for valid and fallback
   rates.

Result on the pilot data available on 2026-10-16 (`runs/analysis/s1-pilot-20261016/s1.json`):

| Input | Value |
|---|---|
| Pool | 172 designs (64 calibration + 3 x 36 pilot) |
| Bootstrap best-at-B quantiles (10/25/50/75/90 %) | 0.750 / 0.750 / 0.8125 / 0.9375 / 0.9375 |
| tau_raw (median) | 0.8125 |
| Best entrant (concave) | 0.75, so the floor is 0.8125 |
| **tau** | **0.8125 (13 of 16 conditions)** |
| Pilot random runs' best-at-B | 0.9375, 0.8125, 0.75 |

Independent check: 20 more random runs (seeds 201-220, used only for power) have a median
best-at-B of 0.8125 too.

The three random pilots were run on the development laptop. CPU MuJoCo on that laptop
differed from the Nebius CPU host in 1 of 1,920 success flags for seed 101 (no design score
changed). The freeze reruns the same three seeds on the executor of record and recomputes
tau. Because the floor binds, tau can only change if the bootstrap median moves above
0.84375.

## 2. Arms

| Arm | What it is | Status |
|---|---|---|
| **A: random** | Scrambled Sobol over the 8 continuous parameters, uniform categoricals, rejection-sampled until the validator accepts (`rtt.search.random_search`). | confirmatory |
| **C: Nemotron with feedback** | `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory. Settings below. | confirmatory |
| **Z: zero-shot Nemotron** | The same model proposing 40 designs without feedback. | **[decide]** recommended: exploratory only (see section 10) |

**Arm C settings** (from `ProposerConfig` and the run manifests):

- Sampling: temperature 1.0, top_p 0.95, max_tokens 16,384; reasoning at the model's
  default. Token Factory exposes no seed parameter, so model calls are not repeatable;
  every request and response is cached by SHA-256 and the run replays with zero API calls
  (`rtt replay-llm`).
- Output: a forced `submit_designs` tool call; 4 slots plus 4 reserves per call.
- Search: 2 islands; per call 2 mutate, 1 crossover, 1 explore; elites shown 8, failures
  4, rejections 6; target 1.2 x the current best; migration after round 2.
- Diversity: a design within 0.08 of a tested design is rejected (`min_distance`);
  explore designs need an untried cell or 0.2 from every design in their cell.
- Caps: at most 16 model calls per run, 1 repair and 1 reflection per call. A slot still
  empty after reserves, repair and reflection is filled at random and logged
  `fallback_random`. Invalid proposals cost no simulator budget.
- Spend cap per job: `--spend-cap` in USD, checked before every call.

**Budget.** B = 40 valid simulated designs per run: round 0 (8) then 4 rounds of 8. Equal
budget means equal simulator evaluations. Tokens, dollars and wall time are reported
separately.

## 3. Runs and seeds

- **R = 12 runs per arm** (section 7; **[decide]** whether to raise R to 32).
- Confirmatory seeds: the first R entries of `configs/seeds_main.yaml` (1001 onwards).
  Run r of every arm uses the same seed, so arms share round 0 run by run.
- Pilot and development seeds (17, 101-103, 201-220, 20261015) are excluded from the
  confirmatory analysis (`configs/seeds_pilot.yaml`).
- No search may use a confirmatory seed before the tag.

## 4. Executor, image and compute

- **Executor of record:** CPU MuJoCo 3.15.0 (`--executor cpu`). MuJoCo Warp failed the
  determinism gate and is a reported cross-check only (`docs/BENCH_MJWARP.md`).
- **Platform:** Nebius Serverless AI Jobs, target `uks2-cpu` (uk-south2, `cpu-d3`, AMD
  EPYC Genoa, preset `16vcpu-64gb`, 16 workers). Confirmatory jobs run with
  `--no-fallback`: if uk-south2 has no capacity the job waits, and the wait is logged in
  DEVIATIONS.md, rather than moving to another CPU type. (The H100 host gave byte-identical
  search tables on 2026-10-15, but that was one comparison.)
- **Image:** built from the tagged commit with `make image` (`build_image.sh --ref HEAD`,
  `git_dirty: false`), pushed to both regional registries, referenced by digest.
  Digest **[at freeze]**.
- **Job order:** all search jobs first; the held-out job last, and it is the only job given
  the held-out seed (`--inject-file`).

## 5. Frozen inputs

Hashes as of this draft; the freeze recomputes every row.

| Input | SHA-256 / digest |
|---|---|
| `configs/task_v1.yaml` | `1fb8fd4d06998f317616623ed1ba6f88c661b58b29aadc55c387bfb90f378081` (task digest `823e58a28a93`) |
| `configs/suites/train_v1.yaml` | `7e62796db3ffaee53eb553cb5343322a5c42453a7ae982f60234b8e94a701501` (suite digest `2dc46d1f79d8...`) |
| `configs/entrants.yaml` | `8c1f50a7cf1ac4470ff91b7a5356f31327a70755f95a4bfa6dc49fa1fb64204d` |
| `configs/seeds_main.yaml` | `e09bc94071e4ceff97acf3cd1b068454feafc2e553febbcb24c18f593830cb9f` |
| `src/rtt/search/prompts/proposer_system.md` | `29a39e0b43a48595bff495d912932c7f9896552c55fce39f561467cd0c5429a6` |
| `src/rtt/search/prompts/proposer_user.md` | `d026107898ef4f602305ed3a09d3792daa78ffa507834296550078ed27efcd24` |
| `src/rtt/search/prompts/repair.md` | `fd1a59d529337afef8a5163c5cca38a8a5ae6e9836112eb47990293d5b494405` |
| `src/rtt/search/prompts/diversity.md` | `6706c224fb722a064003a6ff4ea39277ba8178aa8b85292163efb4f700287422` |
| `src/rtt/search/prompts/zeroshot_{system,user}.md` | `69c263f3...`, `ca29938d...` |
| Tool schema / parameter table / points rule (manifest `prompt_hashes`) | `c559e2d3...`, `58def034...`, `a3ea8087...` |
| `src/rtt/search/nemotron.py` | `ab2f9bb8a22ca5b27af7e45a6d2c7a39415fddbea26d593baf4d166313dfa517` |
| Analysis script `src/rtt/analysis/s1.py` | `d6f4e3b1611bdc1a682cdf800f411aaddc4e57262906adcdc5f00abfead6209e` |
| `uv.lock` | `35ab03fc301c96a223590c5ea17dac525b2432346cce3a1e1a6c62ac9deb8259` |
| Held-out seed commitment `configs/suites/heldout_v1.sha256` | `073055ffc30dd3a66a87be9bd5d6d7d639ce5bd258115cfd58d0e97fb86057d7` |
| Real-object rule `configs/suites/heldout_real_v1.yaml` | file `79137740e1435a559242449e0b8c76a440dc0588cb9e6c6dd72ec0bd1679ca3c`, parsed digest `ce272cfb376a7048396d360e291ae669e5794f0c3bc85c172729bfafab23be82` |
| Git commit, image digest | **[at freeze]** |

## 6. Secondary and exploratory endpoints

**Secondary family** (Holm step-down over the members that run, alpha = 0.05; `holm` in
`s1.py`):

| ID | Endpoint | Test |
|---|---|---|
| **H1** | Procedural held-out success: run-level mean success of each run's frozen top-1 design on the commit-reveal suite (100 unseen conditions x 2 jitter seeds), C > A | one-sided permutation of run-level means (`permutation_test_means`; exact up to 10^6 splits, else 200,000 seeded relabellings) |
| **H2** | Real-object held-out success on the scored Google Scanned Objects draw (64 objects x 2 jitter seeds, `docs/HELDOUT_REAL.md`), C > A | as H1 |
| **S2** | Area under the best-so-far training curve over designs 9-40 (`best_so_far_area`), C > A | as H1 |
| **S3** | S1 for C vs Z, only if Z runs as confirmatory | as S1 |

**Held-out mechanics.**

- The procedural suite's seed was drawn with `secrets.token_hex(32)` into
  `secrets/heldout_seed.txt` (gitignored, mode 600). Only its SHA-256 is in the repository
  (`configs/suites/heldout_v1.sha256`). Search runs execute inside `VaultGuard`, which
  refuses any read of it. The seed is revealed with the results.
- The real-object draw uses the same secret, domain-separated, and reads the commitment
  from the committed file. The scored selection runs after the tag (it names the objects).
  Objects simulated during development are excluded by name (`draw.exclude_names`).
- **[decide] Width rule for H2.** Nine of the ten development objects are wider than the
  widest training part (50 mm), so H2 partly measures how far a design opens.
  Option A: keep the 79 mm limit and report success by width band (at most 50 mm, above
  50 mm) as a pre-registered split. Option B: cap object width at 50 mm (about 5 % of the
  pool qualifies, so `n_objects` may fall to 50 or fewer). Recommended: A.

**Exploratory** (labelled as such wherever shown):

- S1 counted from design 1 (round 0 included), and a permutation log-rank test of T.
- Fallback, validity and rejection rates per arm; model calls, tokens and dollars.
- Score-noise check: each run's first tau-level design re-scored on fresh jitter seeds.
- Best Nemotron design vs the best manual entrant on paired held-out episodes (exact
  McNemar, Wilson intervals).
- Per-family and per-condition success; certificate scores of finalists.

## 7. Power and the choice of R

**Rule (fixed before the pilot):** R is the smallest value in {8, 10, 12} with Monte
Carlo power of at least 0.8 for S1, assuming arm C is 2x faster **[assumed]**. If none
qualifies, R = 12 and the minimum detectable effect is declared.

**Model.** After round 0 each random design reaches tau independently with probability
q. q is fitted to the run-level T of 23 random runs (3 pilot, 20 analysis-only; censored
geometric MLE, 20 events in 353 designs): **q = 0.057**. A speed-up s gives arm C
q_C = 1 - (1 - q)^s, so each C design is worth s random designs. 4,000 simulated
experiments per cell, each analysed with the exact test above. Under q = 0.057, a random
run reaches tau within the budget 85 % of the time, with mean T 23.1.

| R per arm | s = 1 (type-I error) | s = 1.5 | s = 2 | s = 3 |
|---|---|---|---|---|
| 8 | 0.059 | 0.158 | 0.326 | 0.605 |
| 10 | 0.047 | 0.198 | 0.385 | 0.701 |
| 12 | 0.051 | 0.211 | 0.442 | 0.790 |
| 16 | 0.042 | 0.257 | 0.551 | 0.878 |
| 24 | 0.047 | 0.354 | 0.721 | 0.975 |
| 32 | 0.048 | 0.426 | 0.825 | 0.994 |

Sensitivity (same file): drawing q from its posterior changes power by at most 0.03;
run-to-run variation in arm C's speed-up (log-SD 0.5) lowers power at s = 2 by 0.06-0.13;
using the per-design hit rate (q = 0.047, 892 designs) instead of the run-level fit gives
0.43 at R = 12 and 0.79 at R = 32.

**Outcome of the rule:** no R in {8, 10, 12} reaches 0.8 at s = 2, so **R = 12**, with a
declared minimum detectable speed-up of **3.25x at 80 % power** (2.75x at R = 16).

**[decide] Raise R to 32.** Compute is cheap on CPU jobs, and R = 32 gives 0.83 power at
2x. Estimates from measured rates: a random run is 1,920 episodes in 18.7 s on `cpu-d3`
16 vCPU; a Nemotron run made 15 calls of about 17,400 tokens each for $0.11; the client
throttles Super to 150,000 tokens per minute.

| R | Nebius CPU, both arms [derived] | Token Factory, arm C [derived] | Arm C job wall time [derived] |
|---|---|---|---|
| 12 | about $0.30 | about $1.32 | at least 21 min (3.1M tokens) |
| 32 | about $0.70 | about $3.52 | at least 55 min (8.3M tokens); needs `--timeout 2h` |

The Token Factory figures exceed a $1 trial-key limit if one still applies.

## 8. Analysis plan

`rtt analyze-s1 --random <A dirs> --nemotron <C dirs> --calibration <calibration.json>
--tau 0.8125 --compare --out <dir>` produces the primary test, the effect interval, the
reach curves and both sensitivity analyses from the fetched job outputs. The held-out
endpoints come from `rtt heldout` and `rtt heldout-real run`, tested with
`permutation_test_means` and adjusted with `holm`. Every figure is rebuilt from the
released Parquet tables.

## 9. Deviations policy

- Only infrastructure failures are rerun, with the same seed. A rerun is logged with the
  failed job's ID.
- Every change after the tag, and every rerun, goes in `docs/DEVIATIONS.md` with its date
  and reason, before the affected result is looked at where possible.
- A result is never dropped. A run that cannot complete is reported as failed.

**Declared at registration: gate G3 near-miss.** `task_v1` was frozen although gate G3
formally failed on two criteria by narrow margins: the concave entrant scores 75 %
(limit 70 %), and 4 of 64 random designs beat the best entrant (5 required). The score
does respond to geometry (Sobol SD 0.156). Details: `docs/TASK.md`.

## 10. Pilot (before the tag)

| Run | Where | Seeds | Role |
|---|---|---|---|
| Random x 3 | development laptop, CPU (`runs/proposer_dev/random-101-103b`) | 101, 102, 103 | tau rule (provisional) |
| Random x 1 | Nebius `cpu-d3`, job `aijob-e05te2wrh14djpf6d0`; and H100 host, job `aijob-e00xx86cr7b6ycwkwm` | 101 | executor check: identical scores on both hosts |
| Nemotron x 2 (current proposer, before its last fixes) | development laptop (`runs/proposer_dev/nemotron-v3-101-102`) | 101, 102 | valid and fallback rates: 2/32 and 7/32 slots filled at random |
| Nemotron x 1 (old proposer) | Nebius H100 host, job `aijob-e00p6shj7m77w4dfnd` | 101 | superseded (14/32 fallbacks) |
| Random x 20 | development laptop (`runs/analysis/random-201-220`) | 201-220 | power model only |

Still to run before the tag, on `uks2-cpu` from the freeze image: 3 random runs (seeds
101-103) to confirm tau, and 2 Nemotron runs (seeds 101-102) with the final proposer code,
for valid and fallback rates only.

**[decide] Arm Z.** `zeroshot.py` has no reserves, so Z would fill far more slots at random
than C, which muddies "priors vs feedback". Recommended: Z stays exploratory unless it gets
the same reserves before the tag.

## Freeze steps

1. Resolve every **[decide]** item above.
2. Commit; build the image from that commit (`make image`), push it to both registries,
   record the digest.
3. Run the pilot jobs listed in section 10 on `uks2-cpu`; rerun `rtt analyze-s1` on them
   and record tau.
4. Recompute every hash in section 5 and fill in the commit and digest.
5. Tag `prereg-v1`. Its time must precede the first confirmatory job.
