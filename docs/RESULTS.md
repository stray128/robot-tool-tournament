# Results: pre-registered experiment (v2, part-shaped fingers)

Protocol: [`docs/PREREG.md`](PREREG.md), frozen and tagged `prereg-v1` before any confirmatory run. Deviations: [`docs/DEVIATIONS.md`](DEVIATIONS.md). Every number below comes from the scripts named in PREREG A10 run on the fetched job outputs. Nothing was re-run or tuned after the results were seen.

## Question

Given the same simulated budget of 40 finger designs per run, does NVIDIA Nemotron 3 Super, reading the measured results of earlier designs, reach a finger that seats and centres the part in at least 37 of 48 training grasps (tau = 37/48) in fewer designs than equal-budget random search?

## Primary endpoint (S1): designs needed to reach tau

32 runs per arm, seeds 1001 to 1032, executed as Nebius Serverless Jobs on uk-south2 (cpu-d3, 32 vCPU) from image `sha256:c6162fa01d41beaab64e1431bc10e07535ce7bd59045fcef7bf0e074857b4b89`.

| | Random search (arm A) | Nemotron with feedback (arm C) |
|---|---|---|
| Runs that reached tau within 40 designs | 11 of 32 | 23 of 32 |
| Median designs to tau | not reached (above 40) | 22.5 |
| Mean designs to tau (not reached counted as 41) | 34.5 | 25.6 |

- Difference in means (A minus C): **8.97 designs**, 95% interval 3.53 to 14.16.
- One-sided exact permutation test over all C(64,32) splits: **p = 0.0011**.
- Sensitivity, permutation log-rank: p = 0.0011.
- Exploratory, lift-only scoring (the part only has to come off the table): no difference between arms (p = 1.0). Both arms find a finger that lifts the part almost at once. The advantage is in seating and centring the part, which is what the next step in a work cell depends on.

Jobs: arm A `aijob-e05yxb1awgg2d68hq1` (28 min), arm C `aijob-e05s3f8ka6r508p4mk` (84 min). Output: `runs/analysis/s1-confirmatory/s1_v2.json` (released with the run data).

## Secondary endpoints (Holm family)

Each run's frozen top-1 design was scored on suites neither arm saw during search. Held-out job `aijob-e05nvfme46ss0q4wp4`, image `sha256:b7d54d7514ed87121cf657907995b582d5e25c6a78a28d081f5aabda5b1dfb41`.

| Endpoint | Random (A) | Nemotron (C) | Difference | p | Holm-adjusted p |
|---|---|---|---|---|---|
| H1: registered success on 100 unseen procedural conditions x 2 seeds | 60.5% | 72.5% | +12.0 points | 7e-05 | 0.00017 |
| H2: registered success on 64 real scanned objects | 22.0% | 27.5% | +5.5 points | 0.00077 | 0.00077 |
| S2: area under the best-so-far curve | 0.703 | 0.788 | +0.085 | 5.5e-05 | 0.00017 |

All three reject the null at alpha 0.05 after Holm adjustment.

H2 by object width (planned split, reported outside the Holm family): objects 50 mm or narrower across the jaw (18 objects) 19.9% vs 24.1% (p = 0.012); wider than 50 mm (46 objects) 22.8% vs 28.8% (p = 0.001).

Real scanned objects are hard for every design in this space: the best arm seats them about a quarter of the time. The held-out real-object result says the Nemotron arm's picks generalise better, not that the fingers are ready for arbitrary objects.

## Held-out seed, revealed

The procedural held-out suite was generated from a secret seed whose SHA-256 was committed before the tag. It is revealed here so anyone can check the commitment and regenerate the suite.

```
1fef4e226b828acaee6901b28f51fcec1ef3612f10cbff65e44910161af96dbd
```

Check: `printf '%s' '<seed>' | sha256sum` must print the commitment recorded in PREREG A5 (`073055ffc30dd3a66a87be9bd5d6d7d639ce5bd258115cfd58d0e97fb86057d7`).

## Limits

- Everything is simulated in MuJoCo 3.15.0 with rigid contacts. No finger was printed or tested on hardware for this result.
- One gripper geometry (an open-source 84 mm parallel gripper), six training parts, one task (seat, lift, hold).
- Calibration missed its spread target (Sobol SD 0.113 against 0.15). That was declared before the tag. Other declared deviations are listed in DEVIATIONS.
- The multi-effector product layer (dexterous hands, soft arm, suction, Nemotron-designed morphologies) is product mode and not part of this pre-registered result.

## Cost

Nebius AI Cloud for the whole project to this point: about $4.12 (ledger, wall-time estimate). Token Factory for the confirmatory Nemotron arm: about $3.5.
