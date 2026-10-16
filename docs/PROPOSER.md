# Proposer (arm C): prompt design and the near-copy fix

**Verdict.** The Nebius pilot lost 14 of 32 Nemotron slots to random fallbacks, and
28 of its 33 rejections were near-copies of designs already tried. Adding more
information to the prompt did not fix this: Nemotron 3 Super cannot reliably
check its own designs against the run's history. What did fix it is structural.
Each call now returns 4 **reserve designs** besides its 4 slots, and the harness
uses them, in order, to fill any slot it rejects. Islands are asked in turn, so
the second island sees what the first already claimed. In two local B = 40 runs
the fallback rate fell from 44 % (pilot) and 41 % (v2, prompt-only fix) to
**6 % and 22 % (9 of 64 slots, 14 %)**. Every simulated design is still written
by the model or is a logged `fallback_random`; the harness validates and filters
and never edits a design. The diversity threshold stays at **0.08**.

All numbers below are development runs. They were made on a shared laptop CPU
(the executor of record, MuJoCo 3.15.0, full 16-condition training suite), with
one or two seeds each. They are not pre-registered results.

## 1. What went wrong in the pilot

Run `nemotron-r00`, seed 101, B = 40 (8 shared round-0 designs, then 4 rounds of
2 islands x 4 slots), 16 calls, $0.058.

The 28 `too_similar` rejections, classified by what the rejected design was
close to:

| near-copy of | count | what it shows |
|---|---|---|
| a design shown in the island's top-8 (usually the parent itself, at 0.000 to 0.017) | 16 | "mutations" that change one or two numbers by a few percent, or re-propose a parent + categorical switch that an earlier round already tested |
| a tested design that island never saw (outside its top-8, or from the other island) | 6 | the prompt hid most of the run's history |
| the other island's new design in the same round | 6 | both islands got the same prompt in round 1 and proposed the same ideas |

Crossovers were mostly parameter-wise averages of two close parents, which land
within 0.06 of both.

## 2. What changed

| change | why | precedent |
|---|---|---|
| **Table of every tested design in the run**, grouped by *cell* (`inner_face` + `tip_profile`), with its score and its 8 numbers as integer **points** (each range is 0-100 points; a categorical change counts 100), plus the list of untried cells | Nothing is hidden. Points turn the 0.08 rule into integer arithmetic (80 points). Cells make the rule local: designs in different cells are always at least 100 points apart, so only same-cell rows can collide, and an untried cell always passes | OPRO's trajectory of (solution, score) pairs sorted ascending; EvoLLM's integer-discretised solution encoding; MAP-Elites-style niches |
| Per slot: `hypothesis`, then `changes` (every changed parameter, old -> new, in points, with the total), then `same_cell_distances` (distances to the same-cell designs), then the parameters | Forces an explicit plan and parent delta before the numbers | EoH's "thought before code"; chain-of-thought placement in the schema |
| Operator rules with point budgets: mutations 100-250 points from one parent; crossover from parents at least 160 points apart, taking whole parameters (no averaging), at least 80 points from each parent; explore in an untried cell, or at least 200 points from every same-cell design | Quantifies "bold change" and fixes averaging crossovers | ELM / LMX (LLM as mutation and crossover operator); FunSearch / EoH operator mixes |
| **Explore novelty rule, enforced** (`not_novel`): untried cell, or at least 0.2 from every design in its cell | Keeps one slot per call far from all tried designs | Novelty search |
| Elites show their cell, their 2 nearest tested designs in points, and the fully open gap the validator measures; crossover-ready pairs among the top 6 are listed | Shows how crowded each region is; opening violations were the most common invalid reason | |
| **Islands asked in turn**: an island's prompt lists the designs earlier islands claimed this round (rows `t<team>.<slot>`), and they count for the distance rule | Removes same-round cross-island duplicates | FunSearch islands keep separate elite pools (kept here), not separate novelty checks |
| **Reflection** shows, for each rejected slot, its points row, the nearest design's points row, per-parameter differences in points, the shortfall, other designs within 200 points, the untried cells, and the model's own (wrong) distance claim | Exact, actionable feedback | ReEvo-style reflection |
| **Reserves**: `reserve_1` ... `reserve_4` in every call, each with its own operator and parents; they fill rejected slots in order before any repair or reflection call is made | The model's own checks are unreliable (section 4), so oversample and let the validator and diversity check filter. This is the measured fix | FunSearch / AlphaEvolve-style generate-and-filter |
| Temperature 1.0, top_p 0.95 (unchanged); `max_tokens` 8192 -> 16384 | Room for reasoning plus 8 designs; you pay only for tokens generated | |

Unchanged: at most 16 calls per run, at most one repair and one reflection per
island per round, the 0.08 threshold, `fallback_random` with logged reasons, and
the rule that prompts contain only this run's training results.

## 3. Final prompt and call flow

Templates live in `src/rtt/search/prompts/` and are hashed into the manifest
(`prompt_spec_hashes`, which now also hashes the points rule and the tool with
reserves). The user message sections, in order:

1. Task and the 16 training conditions
2. Design parameters (ranges, 10 % steps)
3. Validator rules
4. Distance rule (points, cells, 80-point bar)
5. All N designs tested in this run, by cell (cells and rows in ascending score, best last), plus the untried cells
6. Your team's best designs (top 8 of the island pool, physical parameters, per-family success, failure modes, mass, best k; cell, nearest designs, open gap; crossover candidates)
7. Informative failures (4)
8. Validator rejections in your last round
9. Target (1.2 x the island's best)
10. Round r: what to submit (operator rules, reserves, `changes` / `same_cell_distances` instructions)

The forced tool `submit_designs` has `analysis`, then `mutate_1`, `mutate_2`,
`crossover`, `explore`, then `reserve_1` ... `reserve_4`.

Per island, per round (`LLMProposer._run_island`):

1. Main call. Valid slots go through the diversity check (`too_similar` below
   0.08 from any tested or already accepted design; `not_novel` for an explore
   that fails the novelty rule).
2. Holes take reserves in order. A reserve is validated and checked when it is
   tried; rejected reserves are logged like any rejection, and unused reserves
   are dropped. A reserve copied unchanged into a later call is not tried or
   logged twice.
3. Repair call, only for invalid slots that no reserve filled. Reflection call,
   only for near-copies that are left. Both bring fresh reserves. Reserves a
   follow-up call does not need for its own slots go to every remaining hole
   (near-copy or invalid) before the next call is made or a slot falls back.
4. `fallback_random` for anything still empty.

The zero-shot arm Z shares this machinery but keeps its old prompt and schema:
no table, no explain fields and no reserves (`uses_reserves = False`). It does
get the new points-based reflection text and the island-in-turn flow.

## 4. Evaluation

Seeds 101 (the pilot's seed, whose round 0 already contains a 94 % design, so
it measures the fallback rate only) and 102. Random arm on the same seeds, same
round 0. Runs: `runs/proposer_dev/` (not committed).

| version | seed | fallback | primary slots rejected at the main call | rejections (too_similar / invalid / not_novel) | reserves used / rejected | accepted via main / repair / reflection | calls | $ |
|---|---|---|---|---|---|---|---|---|
| pilot (Nebius) | 101 | **14/32** | 21/32 | 28 / 5 / - | - | 11 / 1 / 6 | 16 | 0.058 |
| v2: table, points, explain fields, islands in turn | 101 | **13/32** | 20/32 | 19 / 8 / - | - | 12 / 2 / 5 | 16 | 0.082 |
| v3 = v2 + reserves (final) | 101 | **2/32** | 19/32 | 41 / 13 / 9 | 15 / 33 | 19 / 0 / 11 | 15 | 0.110 |
| v3 = v2 + reserves (final) | 102 | **7/32** | 17/32 | 40 / 12 / 5 | 7 / 31 | 20 / 1 / 4 | 15 | 0.110 |

The v3 counts exclude 12 duplicate reserve rejections that the final code no
longer logs (section 3, step 2). Final fallback reasons: v3 seed 101 had 1
`too_similar` and 1 `invalid`; seed 102 had 4 `too_similar` and 3 `invalid`.

**Prompt information alone did not reduce near-copies at the source.** Between
53 % and 66 % of primary slots are rejected at the main call in every version. The
model does not use the table to check itself. In v3, its `same_cell_distances`
named the true nearest same-cell design first in only 56 of 194 slots, and it
overstated the nearest distance by 50 points or more in 100 of 194. 17 of its
42 "untried cell" claims were for cells that were already tested. The explain
fields stay because they force a written plan and they make the failure
measurable, but reserves are what turn the extra candidates into simulated LLM
designs.

**Best-so-far versus random.** Best training success after 8 / 16 / 24 / 32 / 40
designs:

| seed | arm | 8 | 16 | 24 | 32 | 40 | first design at >= 81 % / 88 % / 94 % |
|---|---|---|---|---|---|---|---|
| 101 | random | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 6 / 6 / 6 (round 0) |
| 101 | Nemotron v3 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 6 / 6 / 6 (round 0) |
| 102 | random | 0.75 | 0.75 | 0.81 | 0.81 | 0.81 | 18 / - / - |
| 102 | Nemotron v3 | 0.75 | 0.75 | 0.81 | 0.81 | 0.94 | 17 / 35 / 40 |

This is one informative seed. It is not evidence for or against the S1 endpoint.
Seed 103 (random: best 75 %, never reached 81 %) was not run with the
Nemotron arm, to stay inside the $0.40 development cap.

**Cost and latency per call** (mean prompt tokens / completion tokens / p50
latency): pilot 7.7k / 1.5k / 7.1 s; v2 10.2k / 2.3k / 10.1 s; v3 13.8k / 3.5k
/ 13.9 s. A B = 40 run costs about $0.11 in Token Factory (Super, $0.30 / $0.90
per 1M tokens). The model writes about 2 candidates per simulated design.

**Replay determinism.** `rtt replay-llm` on the v3 recording (2 runs, 30
calls) reproduced both runs exactly (same fingerprints) with 0 live API calls,
with the code as it was when the recording was made. The final code does
**not** replay that recording: not logging duplicate reserve rejections changes
the "... and N more" count under "Validator rejections in your last round", so
the seed-101 prompt for the second island in round 2 already differs and the replay stops
on a cache miss. Within a round the final code picks the same designs from the
same responses; from round 2 on, a live run of the final code would ask
different questions. The final code therefore has no live recording yet; the v3
numbers above come from the code before that change and before two later
fixes: leftover follow-up reserves are offered to every hole (section 3), and
NaN or Infinity in tool arguments is rejected as invalid instead of crashing
the repair prompt. `tests/test_llm_cache_replay.py` covers replay of the final code on
scripted responses.

Total live Token Factory spend for this work: $0.302 (v2 $0.082, v3 $0.220).

## 5. Is 0.08 the right threshold?

Kept at 0.08. Two checks:

- **Random designs never trip it.** Take each run's round 0 plus 32 Sobol draws
  (30 seeds, 960 draws) and measure each draw's distance to its nearest earlier
  design in the run. The 1st / 5th / 50th / 90th percentiles are 0.105 / 0.124 /
  0.212 / 0.287, and 0.4 % fall below 0.08. The bar only rejects deliberate
  near-copies. It is far below the novelty of a typical random draw, which is
  why the explore rule uses 0.2, roughly the median.
  `tests/test_diversity.py::test_threshold_never_binds_on_random_search` pins
  this property.
- **Below 0.08 the training score is mostly noise.** I perturbed 4 good designs
  (the best of each random run and the concave entrant) by exact distances,
  with 5 perturbations per level, 80 designs in all, keeping the categoricals.
  The score changed in 90 % / 45 % / 70 % / 90 % of cases at 0.02 / 0.04 /
  0.08 / 0.16, with mean |change| of 0.16 / 0.08 / 0.07 / 0.15 (on average 2 to 4 of the
  16 conditions flip). The contact dynamics are rugged at every scale, so a 0.02 copy is
  a re-roll of its parent's score, not a design step. A lower bar would let an
  arm improve its training best by re-sampling noise around its best design.
  That would inflate S1 through the winner's curse rather than better design.
  This sample is small (n = 20 per level), but the direction is clear.

## 6. Why this is still a fair "Nemotron with feedback" arm

- **Information.** Only this run's training results enter a prompt, as before.
  The table adds the run's own designs and scores, nothing from other runs and
  nothing held out. The prompt guard (`assert_prompt_clean`) still runs on every
  message.
- **Authorship.** Every simulated design in the arm is either written by the
  model (a slot or one of its reserves, `origin: nemotron`, with its operator,
  parents, hypothesis and `via`) or is a `fallback_random` with its reason. The
  harness never moves, repairs or interpolates a design. I considered pushing
  near-copies out along the model's change direction and rejected it, because
  the harness would then co-author designs.
- **Budget.** The simulator budget is unchanged (exactly 40 designs per run, 8
  per round), and so is the call cap (16). Reserves cost output tokens only.
  The extra candidates the model writes are reported (`slot = reserve_k`, and
  rejected reserves appear in `rejections.jsonl`).
- **The random arm** also rejection-samples for validity, so filtering
  candidates through the validator is symmetric.

## 7. Knobs (`ProposerConfig`, frozen in PREREG)

`islands 2`, `elites_shown 8`, `failures_shown 4`, `rejections_shown 6`,
`target_factor 1.2`, `min_distance 0.08`, `explore_min_distance 0.2` (0
disables the explore rule), `reserves 4` (arm C only), `neighbours_shown 2`,
`crossover_pool 6`, `migrate_after_rounds (2,)`, `repairs_per_call 1`,
`reflections_per_call 1`. Sampling: temperature 1.0, top_p 0.95, max_tokens
16384, thinking on.

## 8. Open issues

- **More seeds.** The fallback and best-so-far numbers rest on 1 to 2 local
  seeds. The Nebius pilot of the final code should report fallback rate,
  rejections by reason, reserves used and best-so-far against random on the
  pre-registered seeds before PREREG.
- **Arm Z symmetry.** Arm Z does not get reserves or the table. If Z stays in
  the pre-registration, decide whether it should get reserves
  (`ZeroShotProposer.uses_reserves = True` plus one sentence in
  `zeroshot_user.md`), so the two LLM arms differ only in feedback.
- **Invalid designs** are now about a fifth of all rejections (25 of 120 in v3), mostly the open
  gap (inward bend plus hook) and `depth > 0.5 x thickness`. The open-gap margin
  shown on each elite targets the first; a follow-up could show the validator's
  depth cap per elite.
- **Web labels.** A slot filled by a reserve has `slot = "reserve_k"`, so
  `export_web` currently labels it "Nemotron reserve k". A label built from the
  operator (for example "Nemotron crossover (reserve)") would read better.
- **Machine load.** One local random-arm run hung in the CPU executor's
  `multiprocessing.Pool.map` at a load average of about 45 (all workers idle,
  results never returned). A rerun with 8 workers completed. The scored Nebius
  jobs run alone, but a timeout on `Pool.map` would turn a hang into a clean
  failure.

## References

- Yang et al., *Large Language Models as Optimizers* (OPRO), ICLR 2024.
- Lange, Tian and Tang, *Large Language Models As Evolution Strategies* (EvoLLM), GECCO 2024 companion.
- Romera-Paredes et al., *Mathematical discoveries from program search with large language models* (FunSearch), Nature 2024.
- Lehman et al., *Evolution through Large Models* (ELM), 2022.
- Meyerson et al., *Language Model Crossover: Variation through Few-Shot Prompting* (LMX), ACM TELO 2024.
- Liu et al., *Evolution of Heuristics* (EoH), ICML 2024.
- Ye et al., *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution*, NeurIPS 2024.
- Mouret and Clune, *Illuminating search spaces by mapping elites* (MAP-Elites), 2015.
- Lehman and Stanley, *Abandoning Objectives: Evolution Through the Search for Novelty Alone*, Evolutionary Computation 2011.
