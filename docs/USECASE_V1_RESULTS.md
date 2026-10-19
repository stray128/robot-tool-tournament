# Results: use-case benchmark v1 of the Studio

**Product mode: not part of the pre-registered experiment.** The pre-registered finger experiment
and its result ([RESULTS.md](RESULTS.md)) are unchanged. This benchmark asks a product question of
the Effector Lab Studio. Protocol: [USECASE_V1.md](USECASE_V1.md), frozen and tagged `usecase-v1`
before any held-out run. Every number below comes from `scripts/studio_usecase_v1.py analyse` on the
fetched job output, which is committed in [`results/usecase_v1/`](../results/usecase_v1/).

## Question

For each of the three recorded use cases, does the setup the Studio recommends complete the whole
task more often than the best single off-the-shelf effector, on placements and frictions neither was
chosen on?

## Arms

Frozen on `main` `67c0824` from the development search (8 conditions per hero), before the held-out
secret was drawn:

* **a**: the best single off-the-shelf effector the search found.
* **b**: the Studio's recommendation from the full pool (catalog, open library, designed effectors,
  part-shaped jaws).
* **c**: the Studio's recommendation from the open library only (descriptive).

## Held-out results (32 conditions per hero)

Each condition is a new seed (friction and aim jitter) plus a hidden placement error of up to
±2 mm per object, drawn from the committed secret. All arms of a hero ran the same 32 conditions.

| Use case | a: best single | b: Studio recommendation | c: library only | Primary test, b over a |
|---|---|---|---|---|
| H1 warehouse returns tote | 0/32 (95 % CI 0 to 11 %) | **32/32** (89 to 100 %), two-tool head: soft arm and suction cup | 0/32 (no library-only setup completes; its closest is arm a) | p = 2.3e-10, Holm 7.0e-10 |
| H2 pipe coupling | 0/32 (0 to 11 %) | **32/32** (89 to 100 %), two-tool head: fin-ray and linkage grippers | 16/32 (32 to 68 %) | p = 2.3e-10, Holm 7.0e-10 |
| H3 equipment cabinet | 0/32 (0 to 11 %) | **10/32** (16 to 50 %), tool change of three grippers | 12/32 (21 to 56 %) | p = 0.00098, Holm 0.00098 |

Test: exact one-sided paired sign-flip permutation test (exact McNemar) of b over a, Holm across
the three heroes at α 0.05. All three reject. The best single effector completed none of the 96
held-out runs; the recommendation completed 74.

### Where runs stop (held-out)

| Use case, arm | Step success |
|---|---|
| H1 a | open lid 0/32 (25 slip, 7 over the force limit); nothing after it runs |
| H1 b | open lid, pick carton, pick pack, close lid: 32/32 each |
| H2 a | insert 26/32, tighten 0/32 (26 slip) |
| H2 b | insert 32/32, tighten 32/32 |
| H2 c | insert 16/32, tighten 16/32 |
| H3 a | latch 32/32, door 32/32, valve 0/32 (slip) |
| H3 b | latch 32/32, door 22/32 (10 slip), valve 11/32 (11 penetration over 1 mm), plug 10/32 |
| H3 c | latch 32/32, door 32/32, valve 16/32 (16 penetration over 1 mm), plug 12/32 (4 jams) |

Median cycle of completed runs: H1 b 62.6 s; H2 b 33.9 s, c 38.4 s; H3 b 71.7 s, c 67.9 s.

## What this says, and what it does not

* On H1 and H2, no single off-the-shelf effector finishes the task, and the two-tool head the Studio
  recommends finishes every held-out run. That is the product claim, in simulation.
* On H3 the recommendation finishes about a third of held-out runs, below its 4 of 8 on the
  development conditions. The library-only setup did slightly better here (12 against 10, not
  tested). The cabinet's valve step is the weak point for every arm.
* Arm a completed 0 of 8 development runs on every hero, so the primary test mostly asks whether
  the recommendation's development success holds up on new conditions.
* Every H2 and H3 arm, and H1's recommendation, is approximation tier (soft arm, suction seal,
  threaded joint, latch, valve). These are simulation rankings under stated conditions, not hardware
  results.

## Laptop and Nebius agree

Each arm also re-ran its 8 development conditions in the same job (parity runs). The outcome matched
the laptop's on 72 of 72 runs. The trajectory hash matched on 41 of 72. The most likely cause is
floating-point rounding that differs between the laptop's CPU and the Nebius AMD EPYC Genoa CPUs;
the drift never changed an outcome. The protocol declared parity descriptive in advance.

## Held-out secret, revealed

```
a418e29f0a0cb819298bc0f3e7de328d3f743b7e8f470f415f5e7322933a33a7
```

Check: `printf '%s' '<secret>' | sha256sum` prints the commitment in the tagged config,
`8011661730eb0c5805065ca6ef5b9b4948dbe58db86e26c8333bd30f4e31196a`. The analysis also checks that
the secret regenerates the recorded conditions.

## Compute

One Nebius Serverless Job on uk-south2 (`cpu-d3`, 32 vCPU), `aijob-e05e55n905qj9va2fh`, 206 s wall,
about $0.06 (wall-based estimate). Image built from the tagged commit, digest
`sha256:6f0ebaf9ae7147464ed44845b0d12b5ca656d27fdc527f2f6afee494f54e3976`. No Token Factory spend.
