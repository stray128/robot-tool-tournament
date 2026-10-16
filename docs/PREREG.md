# Pre-registration (DRAFT v2, not frozen)

**Status: draft for approval. Not tagged.** This file becomes the protocol of record when
the commit that holds it is tagged `prereg-v1`. The tag must precede the first
confirmatory job. Until then every number here may change; after it, every change goes in
`docs/DEVIATIONS.md`.

This file holds two protocols. Exactly one of them runs:

- **Part A, the v2 protocol** (part-shaped fingers, `freeform-v2`, `task_v2`,
  `train_v2`). It runs if the v2 gates pass in time (section A0).
- **Part B, fallback F1** (14-box fingers, `task_v1`, `train_v1`). It is the v1 draft of
  2026-10-16, unchanged apart from heading levels. It runs if the v2 gates miss their
  deadlines.

The go/no-go decision is written into section A0 and `docs/DEVIATIONS.md` before the tag,
and the protocol that does not run is kept here, marked "not run".

Items marked **[decide]** need a founder decision before the tag. Items marked
**[at freeze]** are filled in by the freeze steps at the end of each part.

---

# Part A. The v2 protocol

## A0. Which protocol runs

| Deadline (IST) | Condition | If not met |
|---|---|---|
| Mon Oct 19, 22:00 | Gates GV0 (geometry), GV1 (fidelity and fairness), GV2 (physics), GV3 (calibration) and GV4 (Nebius parity) green | F1: Part B runs |
| Tue Oct 20, 22:00 | Gate GV5 (search harness) green | F1: Part B runs |

Under F1, `freeform-v2` is used for product mode and visuals only, and every clip that
shows it is labelled "product mode".

**Decision:** [at freeze] (date, time, which gates, who decided).

## A1. Question and primary endpoint (S1)

**Question.** Does NVIDIA Nemotron 3 Super, reading its own run's simulator results, reach
a good part-shaped finger recipe in fewer simulated designs than random search with the
same budget?

**Design score.** A design is a part-relative recipe, compiled against each training part
into a left and a right jaw (`docs/TASK_V2.md`). Its **training score** is its mean
registered-grasp success (`success_v2`) over the 48 episodes of `train_v2` (6 parts x 4
placements x 2 jitter seeds). There is no best-of-k step. Scores lie on a 1/48 grid.

`success_v2` requires lift and hold as in v1, plus, 0.25 s after closing and before the
lift: seat shortfall at most 1.0 mm, |dx| and |dy| at most 0.75 mm, yaw at most 2 deg
(modulo the part's symmetry), tilt at most 3 deg; and tilt at most 5 deg after the hold.
These tolerances may change once, during GV3, and are frozen in `docs/TASK_V2.md`.

**S1: evaluations to tau.** As in v1:

- Each run simulates exactly B = 40 valid designs, numbered 1 to 40 in evaluation order.
  Designs 1-8 are the shared round 0: the four round-0 entrants (`stock`, `flat`,
  `imprint_configurator`, `imprint_swept`), then 4 Sobol designs seeded only by the run
  seed. For a given seed, round 0 is byte-identical in every arm.
- **T** is the index of the first design **proposed by the arm** (index 9 or later) whose
  training score is at least tau. A run that never gets there is censored at T = 41.
  Round-0 designs never count toward T.

**tau = [at freeze]**, set by the rule in A1.1 from the pilot.

**Hypotheses.** H0: the arms' T values are exchangeable. H1: arm C needs fewer
evaluations than arm A.

**Test.** One-sided exact permutation test of the restricted mean of T, alpha = 0.05,
over every relabelling of the 2R runs into two groups of R (`permutation_test` in
`src/rtt/analysis/s1.py`). Ties count against H1. The reasons for this test are those of
Part B, section B1.

**Effect estimate.** mean(T_A) - mean(T_C) with a 95 % percentile bootstrap interval
(10,000 resamples of runs within each arm, seed 0; `restricted_mean_ci`). Also reported:
the share of runs that reached tau by each evaluation count, per arm (`reach_curve`).

### A1.1 The tau rule (fixed before the pilot)

Implemented by `derive_tau_v2` in `src/rtt/analysis/tau.py`; constants in `TauRuleV2`.

1. Inputs: exactly three random-arm pilot runs (B = 40, seeds 101-103) and the Sobol-64
   screening sample of gate GV3 (`calibration_v2.json`). The Nemotron pilots never enter
   the rule; they are used only for valid and fallback rates.
2. The pool is every Sobol-64 score plus every non-entrant design score of the three
   random pilots.
3. Bootstrap 20,000 random runs of B = 40: the four round-0 entrants' scores plus 36
   scores drawn with replacement from the pool (seed 20261021). Best-at-B is the maximum.
4. tau_raw is the median of the bootstrapped best-at-B. tau is tau_raw rounded to the
   1/48 grid, halves up.
5. tau is raised, if needed, to 2/48 above the best round-0 score seen before the tag:
   the maximum over the four round-0 entrants and the round-0 designs of the three random
   pilots. tau is capped at 1; a capped tau is reported as degenerate.
6. A pilot that used a confirmatory seed (`configs/seeds_main.yaml`) is refused.

Rule digest (`TauRuleV2().digest()`, draft): `e4db295272c0818d8a5148b0d1b32b2104a1db6018947a050480253c22b9b695`.
The freeze recomputes it with the file hash.

Result on the pilot: [at freeze] (tau_raw, floor, binding, pool size, per-design hit rate
at tau, pilot best-at-B).

## A2. Arms

| Arm | What it is | Status |
|---|---|---|
| **A: random** | Scrambled Sobol over the 7 continuous recipe parameters, uniform over the 9 pad x tip cells, canonicalised, then rejection-sampled until the validators accept (`rtt.search.space_v2`). The valid rate is logged. | confirmatory |
| **C: Nemotron with feedback** | `nvidia/nemotron-3-super-120b-a12b` on Nebius Token Factory, `submit_designs` v2 tool. Settings below. | confirmatory |
| **Z: zero-shot Nemotron** | The same model and tool, 40 designs, no feedback. | **[decide]** at freeze, from measured cost. Recommended: exploratory unless it gets the same reserves as C. |
| **T: TPE** | Tree-structured Parzen estimator over the same space (`optuna`, extra `tpe`). | **[decide]** at freeze, from measured cost. Recommended: not run. |

**Arm C settings** [at freeze, from `ProposerConfig` and the v2 run manifests]:

- Sampling as in v1 (temperature 1.0, top_p 0.95, max_tokens 16,384, reasoning at the
  model's default). Token Factory has no seed parameter, so every request and response is
  cached by SHA-256 and the run replays with zero API calls.
- Output: a forced `submit_designs` v2 call. Each design gives `pad`, `tip`, `params`,
  `hypothesis`, `changes` and `same_cell_distances`. Only active parameters are required;
  inactive ones are ignored and logged. **4 slots plus 4 reserves** per call, or 3 plus 3
  if the probe (5 calls) is below 90 % valid after reserves or needs more than one repair.
- The prompt adds the training parts table (family, dimensions, features, material,
  mass), the placement conditions and the registration tolerances, per-part and
  per-placement success vectors, `jam_on_close` and `off_centre` counts, and for each
  elite its capture report, mass and piece count. Held-out parts never appear in a
  prompt.
- **Engineering-notes block**, shared by C and Z and hashed: the friction rule (draft at
  most 90 deg - atan(mu) - 5 deg, about 68 deg at mu 0.3), the capture rule (clearance +
  depth x tan(draft) + chamfer at least the error band), the stroke cap and print rules.
  It is guidance, never a validator.
- The capture report (the static predictor in `rtt.design.freeform.predict`) is returned
  in feedback. It never rejects a design.
- Caps: at most 16 model calls per run, 1 repair and 1 reflection per call. A slot still
  empty after reserves, repair and reflection is filled at random and logged
  `fallback_random`. Invalid proposals cost no simulator budget.

**Canonicalisation, IDs and duplicates** (identical for every arm, before hashing,
building or distance): inactive parameters reset to their fixed values (logged); values
quantised to 0.1 mm, 0.5 deg and 0.01. `design_id` = SHA-256 of the generator version and
the canonical recipe; `geometry_id` = SHA-256 over the 6 training-part geometry hashes. A
design whose `geometry_id` matches an earlier design's is rejected as
`duplicate_geometry`, in every arm.

**Diversity threshold** (fixed before the pilot): t is the smallest distance such that at
most 1 % of 10,000 random valid Sobol pairs lie within t on the normalised 9-parameter
vector. A design within t of a tested design is rejected. Explore designs keep the v1
rule (an untried cell, or a margin from every design in their cell), with the margin
rescaled to the v2 vector. Values: [at freeze] (computed by the search code,
`rtt.search.space_v2`).

**Budget.** B = 40 valid simulated designs per run: round 0 (8) then 4 rounds of 8. Equal
budget means equal simulator evaluations. Tokens, dollars and wall time are reported
separately. The reference entrants `ref_wrap` and `ref_hook` are scored outside the budget
and shown in every comparison.

## A3. Runs and seeds

- **R runs per arm, set by the rule in A7.** [at freeze]
- Confirmatory seeds: the first R entries of `configs/seeds_main.yaml` (1001 onwards).
  Run r of every arm uses the same seed. No v2 search may use them before the tag.
- Pilot seeds (`configs/seeds_pilot.yaml`): random 101-103, Nemotron 101-102, Z (if
  piloted) 101. The GV3 screening sample uses Sobol seed 20261020, distinct from every
  gate and search seed (`check_screening_seed`).

## A4. Executor, image and compute

- **Executor of record:** CPU MuJoCo 3.15.0, as in Part B.
- **Platform:** Nebius Serverless AI Jobs, target `uks2-cpu` (uk-south2, `cpu-d3`), with
  `--no-fallback`; waits are logged in DEVIATIONS.md.
- **Geometry:** jaws are built inside the job. Gate GV4 requires at least 99 % of 100 v2
  episodes to match the laptop and every geometry and hull hash to be equal; otherwise
  geometry is built once, stored by hash, and the jobs score the stored geometry.
- **Physics validity:** an episode whose MuJoCo bad-acceleration counter rises is
  `physics_invalid` and counts as a failure. More than 1 % of a run's episodes flagged is a
  deviation.
- **Image:** built from the tagged commit with `make image`, pushed to both regional
  registries, referenced by digest. Digest [at freeze].
- **Job order:** all search jobs first; the held-out jobs last, and only they are given
  the held-out seed.
- **Compute estimate [derived]:** about 25 CPU-s per design and about 17 CPU-min per run;
  about 6 CPU-h for 2 arms x R = 10, about 8 CPU-h with the pilot and held-out. Gate GV2
  caps a v2 episode at 1.5x the v1 median CPU time.

## A5. Frozen inputs

All rows are filled in at freeze.

| Input | SHA-256 / digest |
|---|---|
| Design-space version and generator code (`src/rtt/design/freeform/*.py`) | [at freeze] |
| Parts library (`src/rtt/parts/*.py`, `configs/parts_v2.yaml`) | [at freeze] |
| `configs/task_v2.yaml` (including the registration tolerances) | [at freeze] |
| `configs/suites/train_v2.yaml` | [at freeze] |
| `configs/entrants_v2.yaml` | [at freeze] |
| CoACD version, seed and threshold | coacd 1.0.14, seed 0, threshold 0.02 |
| Stock-blade hulls (`stock_blade_hulls.npz`) | [at freeze] |
| Scanned-object hull hashes (`heldout_real_v2`) | [at freeze] |
| Hero rule (`src/rtt/analysis/hero_rule.py`; `HeroRule().digest()`) | [at freeze]; digest draft `8bbdb11175e350a9e7b64834e994827b014069823d5981eb8a82862ba0c8e875` |
| v2 prompts and the engineering-notes block (`src/rtt/search/prompts/v2/*`) | [at freeze] |
| Tool schema, parameter table, points rule (manifest `prompt_hashes`) | [at freeze] |
| tau rule (`src/rtt/analysis/tau.py`; `TauRuleV2().digest()`) | [at freeze]; digest draft `e4db295272c0818d8a5148b0d1b32b2104a1db6018947a050480253c22b9b695` |
| R rule and GV6 checks (`src/rtt/analysis/power.py`; `RRuleV2().digest()`) | [at freeze]; digest draft `903634035020013c3ee5f68839f283001aa78a7e76bb36455a8061de3beba8c4` |
| Diversity rule (`src/rtt/search/space_v2.py` or its diversity module) | [at freeze] |
| Analysis scripts (`src/rtt/analysis/s1.py`, `tau.py`, `power.py`, `calibrate_v2.py`) | [at freeze] |
| `configs/seeds_main.yaml`, `configs/seeds_pilot.yaml` | [at freeze] |
| Held-out seed commitment (`configs/suites/heldout_procedural_v2.sha256`) | [at freeze] |
| Real-object rule (`configs/suites/heldout_real_v2.yaml`) | [at freeze] |
| `uv.lock` | [at freeze] |
| Git commit, image digest | [at freeze] |

`make prereg-rules` prints the three rule digests.

## A6. Secondary and exploratory endpoints

**Secondary family** (Holm step-down over the members that run, alpha = 0.05; `holm` in
`s1.py`):

| ID | Endpoint | Test |
|---|---|---|
| **H1** | Procedural held-out registered success: run-level mean `success_v2` of each run's frozen top-1 recipe, recompiled against each part of `heldout_procedural_v2` (100 conditions x 2 jitter seeds), C > A | one-sided permutation of run-level means (`permutation_test_means`) |
| **H2** | Real-object held-out registered success on the scored scan draw (`heldout_real_v2`), C > A | as H1 |
| **S2** | Area under the best-so-far training curve over designs 9-40, C > A | as H1 |
| **S3** | S1 for C vs Z, only if Z runs as confirmatory | as S1 |

A recipe that builds invalid on a held-out part scores as a failure on that part.

**Held-out mechanics.** As in Part B, section B6, with a new secret seed for
`heldout_procedural_v2`: drawn with `secrets.token_hex(32)` into a git-ignored file before
the tag, only its SHA-256 committed, revealed with the results. Searches run inside the
vault guard. The real-object draw reuses the v1 selection rule (and the width-band
decision of Part B, section B6, **[decide]**), with CoACD 0.02 decomposition.

**Lift-only.** Every endpoint above is also computed on lift-only success (exploratory).
`python -m rtt.analysis.tau analyze` does this for S1 at the frozen tau.

**Exploratory** (labelled as such wherever shown):

- everything listed as exploratory in Part B, section B6;
- per-part and per-placement success; `jam_on_close` and `off_centre` counts per arm;
- agreement of the capture-report predictor with the simulator;
- the certificate battery on finalists: 0, +-1, +-2 and +-3 mm, push and shake, reporting
  both "held" and "registered" per placement;
- hold creep and `physics_invalid` rates per arm;
- the friction sweeps at mu 0.6 and 1.0, shown as a limit.

## A7. Power and the choice of R

**Rule (fixed before the pilot):** R is the smallest value in {8, 10, 12} with Monte Carlo
power of at least 0.8 for S1, assuming arm C is 2x faster **[assumed]**. If none
qualifies, R = 12 and the minimum detectable speed-up at R = 12 is declared.
Implemented by `apply_r_rule` in `src/rtt/analysis/power.py`.

**Model.** The geometric model of Part B, section B7. The per-design hit rate q is the
pooled hit rate at tau of the random pool that set tau (Sobol-64 plus the three random
pilots' non-entrant designs, about 170 designs). With three pilot runs a run-level fit
has at most three events, so it is reported as a sensitivity analysis, as is power with q
drawn from its Jeffreys posterior.

**What the rule will almost certainly return [derived, Monte Carlo, written before any v2
data].** At 2x, power at R = 12 is 0.33-0.47 for every q from 0.02 to 0.25 (400-1,000
experiments per cell), so no R in {8, 10, 12} reaches 0.8 and the rule returns R = 12. The
declared minimum detectable speed-up at R = 12 is then about 3.0-3.75x (3.25x at q = 0.04).
This matches Part B: S1 at this budget is powered for large effects only.

**[decide] Raise R before the pilot.** The rule's candidate set is the founder's choice
and must be fixed before any pilot result is seen. Part B found R = 32 gives 0.83 power
at 2x. Cost at v1 rates [derived]: CPU about $0.30 (R = 12) or $0.70 (R = 32) for both
arms; Token Factory about $1.32 or $3.52 for arm C, and more if the v2 prompts are longer
**[assumed]**.

Result: R = [at freeze], q = [at freeze], MDE = [at freeze].

**GV6 checks on the analysis** (before the tag; `python -m rtt.analysis.power gv6`):

- Null calibration: at the fitted q and the chosen R, randomised exact p-values under
  s = 1 are uniform (Kolmogorov-Smirnov p at least 0.01), and the conservative test's
  rejection rate at alpha is at most alpha plus 3 Monte Carlo standard errors.
- Planted effect: synthetic runs with a planted speed-up, written as score sequences on
  the 1/48 grid and analysed by the same code as real runs, recover the planted
  difference in restricted means without bias (within 3 Monte Carlo standard errors), and
  the 95 % intervals cover it in at least 90 % of experiments (less 3 standard errors).
  The end-to-end test in `tests/test_analysis_v2.py` runs the same check through run
  directories on disk and the command line.

Results: [at freeze].

## A8. Hero rule (hashed before calibration)

Implemented by `src/rtt/analysis/hero_rule.py`; applied at gate GV-H (Mon Oct 19) to the
training parts and the unbranded scanned objects. Every input is a recorded simulation
episode. **A failure is never staged.**

Screen conditions: +-1, +-2 and +-3 mm across the jaw, and 5 deg yaw at zero offset (7
conditions). A part qualifies as a **hero** when:

- **(a)** the stock blade fails `success_v2` on at least 4 of 7, and fails **visibly** (a
  slip during the lift, an ejection on closing, a drop during the hold, or a tilt above
  10 deg) on at least 2 of 7; and
- **(c)** the best designed recipe passes `success_v2` with a planar offset of at most
  0.5 mm on at least 6 of 7.

It is also an **imprint hero** when:

- **(b)** the configurator imprint shows a seat shortfall of at least 2 mm or a planar
  offset of at least 1 mm on at least 4 of the 6 non-zero translation errors.

Scans with visible branding never qualify. With several episodes per condition, a
condition shows a property only if a strict majority of its episodes do; ties count
against qualification. A `physics_invalid` episode never counts as a stock or imprint
failure, and never as a designed pass. A value that was not measured (no registration
pose, shortfall or tilt in the record) counts against qualification too: it is never a
designed pass, an imprint jam or a visible failure.

Heroes are labelled "showcase parts chosen because the stock finger fails". Expected: the
tee, the block, the steel slug and one unbranded scanned object. If fewer than 3 qualify,
the video hook uses the imprint-vs-designed offset contrast only.

Result: [GV-H] (`hero_screen.json`, its SHA-256, the heroes and the hook mode).

## A9. Calibration (gate GV3) rules

Implemented by `src/rtt/analysis/calibrate_v2.py`. The screening sample is the round-0
and reference entrants plus 64 valid Sobol designs (seed 20261020), scored on `train_v2`.

**Targets** (on the screening sample):

- every round-0 entrant scores 20-70 % registered;
- the Sobol-64 SD (ddof = 1) is at least 0.15;
- at least 5 of 64 beat the best round-0 entrant outright (or the miss is declared, as
  with v1's G3);
- at most 10 % of the Sobol designs score 100 %.

Reported, not gating: lift-only scores, `jam_on_close` and `off_centre` counts, and the
`physics_invalid` share (above 1 %: a deviation).

**Procedure:**

- At most 3 iterations, each logged in `calibration_log.json` with its file hashes,
  changes and checks. Iteration 1 is the task as drafted.
- Only suite-level knobs may change: part dimensions, density or material; placements;
  the registration tolerances. Never anything per design, never an entrant, never a
  physics or task setting. Friction stays 0.3 and the grip force 25 N per jaw, because the
  claims in A13 and the hashed engineering notes are stated at those values.
- The registration tolerances change in at most one iteration.
- Every iteration scores the same screening sample.
- After the last iteration: pass, freeze; only the headroom criterion failing, declare the
  miss and freeze; anything else, F1 (section A0).
- The freeze records the SHA-256 of `configs/task_v2.yaml`, `configs/suites/train_v2.yaml`,
  `configs/entrants_v2.yaml` and `configs/parts_v2.yaml` in `docs/TASK_V2.md` and here.

Results: [at freeze].

## A10. Analysis plan

```bash
python -m rtt.analysis.tau derive --random <3 pilot dirs> --calibration calibration_v2.json --power --out <dir>
python -m rtt.analysis.power gv6 --q <q> --runs <R>
python -m rtt.analysis.tau analyze --random <A dirs> --nemotron <C dirs> --tau <frozen tau> --out <dir>
```

The first command sets tau and R by rule from the pilot. The last produces the primary
test, the effect interval, the reach curves, both sensitivity analyses (permutation
log-rank; T counted from design 1) and the lift-only pass, from the fetched job outputs.
The held-out endpoints are tested with `permutation_test_means` and adjusted with `holm`.
Every figure is rebuilt from the released Parquet tables.

## A11. Deviations policy

As Part B, section B9: only infrastructure failures are rerun, with the same seed and the
failed job's ID logged; every change after the tag goes in `docs/DEVIATIONS.md` before the
affected result is looked at where possible; a result is never dropped. In addition, a run
with more than 1 % `physics_invalid` episodes is logged as a deviation, and is still
reported.

## A12. Pilot (Wed Oct 21, before the tag)

| Run | Where | Seeds | Role |
|---|---|---|---|
| Random x 3 | Nebius `uks2-cpu`, freeze image | 101, 102, 103 | tau rule (A1.1) and q for the R rule (A7) |
| Nemotron x 2 | Nebius `uks2-cpu`, freeze image | 101, 102 | valid and fallback rates only |
| Zero-shot x 1 (optional) | Nebius `uks2-cpu`, freeze image | 101 | cost and fallback rate, for the Z decision |

Pilot bouts may be shown, labelled "pilot". Calibration bouts rendered on Oct 20 are
labelled "pilot" too.

## A13. Claims fixed now

> "On parts with flat or socket features, the configurator imprint stops short under
> +-1-3 mm placement error and holds the part where it landed. Drafted designs pull it to
> centre."

- Round parts at mu 0.3 do not jam, and we say so.
- Everything is simulated at mu 0.3 and 25 N per jaw. No claim of real-world centring.
- Product-mode options (the locator pin and body styles) are never in the scored space,
  and every clip that shows them is labelled "product mode".
- Rib-windowed bodies are rigid trusses, never presented as compliant.

## Part A freeze steps

1. Record the go/no-go decision in A0 and in DEVIATIONS.md.
2. Resolve every **[decide]** item in Part A.
3. Freeze `task_v2`, `train_v2`, `entrants_v2` and `parts_v2` at GV3; record the hero
   screen (GV-H).
4. Commit; build the image from that commit (`make image`), push it to both registries,
   record the digest.
5. Run the pilot (A12); apply the tau and R rules; run the GV6 checks.
6. Draw the procedural held-out seed and commit its SHA-256.
7. Recompute every hash in A5 and fill in the commit and digest.
8. Mark Part B "not run".
9. Tag `prereg-v1`. Its time must precede the first confirmatory job.

---

# Part B. Fallback F1: the v1 protocol

This part runs only if F1 is triggered (section A0). It is the v1 draft as written on
2026-10-16, unchanged except that its headings are now B1-B10. Section numbers inside
Part B refer to Part B. Its scope note: it covers the v1 design space (14-box fingers,
`task_v1`, `train_v1`).

## B1. Question and primary endpoint (S1)

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

## B2. Arms

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

## B3. Runs and seeds

- **R = 12 runs per arm** (section 7; **[decide]** whether to raise R to 32).
- Confirmatory seeds: the first R entries of `configs/seeds_main.yaml` (1001 onwards).
  Run r of every arm uses the same seed, so arms share round 0 run by run.
- Pilot and development seeds (17, 101-103, 201-220, 20261015) are excluded from the
  confirmatory analysis (`configs/seeds_pilot.yaml`).
- No search may use a confirmatory seed before the tag.

## B4. Executor, image and compute

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

## B5. Frozen inputs

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

## B6. Secondary and exploratory endpoints

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

## B7. Power and the choice of R

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

## B8. Analysis plan

`rtt analyze-s1 --random <A dirs> --nemotron <C dirs> --calibration <calibration.json>
--tau 0.8125 --compare --out <dir>` produces the primary test, the effect interval, the
reach curves and both sensitivity analyses from the fetched job outputs. The held-out
endpoints come from `rtt heldout` and `rtt heldout-real run`, tested with
`permutation_test_means` and adjusted with `holm`. Every figure is rebuilt from the
released Parquet tables.

## B9. Deviations policy

- Only infrastructure failures are rerun, with the same seed. A rerun is logged with the
  failed job's ID.
- Every change after the tag, and every rerun, goes in `docs/DEVIATIONS.md` with its date
  and reason, before the affected result is looked at where possible.
- A result is never dropped. A run that cannot complete is reported as failed.

**Declared at registration: gate G3 near-miss.** `task_v1` was frozen although gate G3
formally failed on two criteria by narrow margins: the concave entrant scores 75 %
(limit 70 %), and 4 of 64 random designs beat the best entrant (5 required). The score
does respond to geometry (Sobol SD 0.156). Details: `docs/TASK.md`.

## B10. Pilot (before the tag)

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

## Part B freeze steps

1. Resolve every **[decide]** item above.
2. Commit; build the image from that commit (`make image`), push it to both registries,
   record the digest.
3. Run the pilot jobs listed in section 10 on `uks2-cpu`; rerun `rtt analyze-s1` on them
   and record tau.
4. Recompute every hash in section 5 and fill in the commit and digest.
5. Tag `prereg-v1`. Its time must precede the first confirmatory job.
