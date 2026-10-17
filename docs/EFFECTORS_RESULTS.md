# Effector tournament results (product mode, not pre-registered)

**Product mode: not part of the pre-registered experiment.** Nothing here is a pre-registered
endpoint, and nothing here changes or adds to `docs/RESULTS.md`. The pre-registered code,
configs and lock file are unchanged by the effector layer (`docs/PREREG.md` A5: every listing
and file hash recomputed on the integration branch, identical to `main`). These are
simulation rankings under stated conditions, from one tournament per part, with no
statistics. The rules are in `docs/EFFECTORS.md` (sections 4, 5 and 10).

Run on 2026-10-17, branch `effectors-integ`, contract `rtt-effectors/1`, CPU MuJoCo 3.15.0,
task digest `8168db801332`. The published outputs are those of the last re-run, at commit
`8c1011c` with a clean tree. The site's pages under `/effectors/` show the same data.

## 1. What ran

The six public training parts (`train_v2`). For each part, every family this build has,
on the same wrist, servos, phases, placements, seeds and scorer:

| Budget group | Candidates per part | Fidelity | What they are |
|---|---|---|---|
| Reference | 3 | core | the parallel jaw with the stock, flat-pad and configurator-imprint fingers (the frozen v2 path) |
| Catalog | 12 | rigid | three vendored hands, two synergies each (6); the fin-ray and rigid-pad A two-finger grippers (2); the procedural two- and three-finger adaptive grippers, two synergies each (4) |
| Soft arm | 3 | approximation | 12, 16 and 20 segment tentacles |
| Suction | 3 | approximation | 10, 16 and 25 mm cups |
| Designed (Nemotron) | 4 to 7 | rigid or approximation | Nemotron 3 Super proposals, 2 rounds of 4 requested; round 2 sees round 1's physics |
| Random control | 8 | rigid or approximation | recipes drawn from the same spaces, with the same validators |

* Screen: 4 placements (centred, +2 mm, -3 mm, +1 mm and 6 deg) x 2 jitter seeds = 8 episodes
  per candidate. The non-jaw families first run a nominal reference episode.
* Certificate: the top 2 of each group by screen success, on the frozen 18-test battery
  (nominal, placement errors to 3 mm on both axes, a 1 g shake on three axes, a 0.5 N s push
  on two axes).
* Per part: 510 to 537 episodes, 3,141 in all (1,845 screen and reference, 1,296 certificate).
* Headline requirement: **registered** (held, and back in the reference pose within the
  frozen tolerances). **Held** (lifted and held) is computed too.

Founder decisions applied before the run: the hands stay in at their upstream gains (no gain
changes); the rigid-pad gripper B (1.4 N at its published settings) is not an entrant, so the
catalog group is exactly 12 and is never sampled; the rigid-pad gripper A keeps its upstream
4.1 N and every page that shows it prints "Low grip at published settings".

## 2. Recommendation per part

Certificate score: the share of the 18 tests passed for the requirement (passes in brackets).
Tolerance: the largest placement error, across x / along y, up to which every placement test
passed.

| Part | Registered (headline): winner, certificate, tolerance | Held: winner, certificate, tolerance | Rigid runner-up |
|---|---|---|---|
| Block | Parallel jaw, stock blade (reference, core): 0.68 (11/18), ±0 / ±3 mm | Parallel jaw, flat pad (reference, core): 1.00 (18/18), ±3 / ±3 mm | not needed |
| Flanged bushing | Parallel jaw, configurator imprint (reference, core): 1.00 (18/18), ±3 / ±3 mm | Parallel jaw, configurator imprint (reference, core): 1.00 (18/18), ±3 / ±3 mm | not needed |
| Hex standoff | Designed four-finger hand, 2 joints per finger e900 (Nemotron, rigid): 0.85 (12/18), ±2 / ±1 mm | the same design: 1.00 (18/18), ±3 / ±3 mm | not needed |
| L-bracket | Designed three-finger sliding hook gripper 4a1d (Nemotron, rigid): 0.56 (12/18), ±1 / ±3 mm | Parallel jaw, stock blade (reference, core): 0.67 (16/18), ±3 / ±3 mm | not needed |
| Pipe tee | Parallel jaw, flat pad (reference, core): 0.68 (11/18), ±0 / ±3 mm | Parallel jaw, flat pad (reference, core): 0.83 (17/18), ±3 / ±3 mm | not needed |
| Steel slug | Parallel jaw, configurator imprint (reference, core): 1.00 (18/18), ±3 / ±3 mm | Parallel jaw, configurator imprint (reference, core): 1.00 (18/18), ±3 / ±3 mm | not needed |

"Rigid runner-up: not needed": the fidelity rule names one only when the winner is an
approximation (suction or soft arm). No approximation won on any part, for either
requirement. Every part has a winner for both requirements.

Ties settled by the simplicity rule (equal certificate score and tolerance):

* Block, registered: the stock-blade jaw, the flat-pad jaw and the two-finger adaptive gripper
  all score 0.68 at ±0 / ±3 mm; a jaw wins on actuators (1 against 2).
* Flanged bushing, held: the imprint jaw and the 16-segment soft arm both pass 18/18; the jaw
  wins on mass (62 g against 99 g).
* Steel slug, both: the imprint jaw and a Nemotron four-finger hand (789b) both pass 18/18 at
  ±3 / ±3 mm with 8/8 on the screen; the jaw wins on actuators (1 against 4).

## 3. Best certificate per family

Best registered / best held certificate score among a family's certified finalists on each
part. A family with no finalist shows its best screen result instead.

| Part | Parallel jaw | Catalog hands | Catalog grippers | Soft arm | Suction | Nemotron designs | Random control |
|---|---|---|---|---|---|---|---|
| Block | 0.68 / 1.00 | not certified (screen 3/8 registered, 8/8 held) | 0.68 / 0.83 | 0.27 / 0.56 | 0.14 / 0.44 | 0.56 / 0.67 | 0.14 / 0.44 |
| Flanged bushing | 1.00 / 1.00 | not certified (0/8, 8/8) | 0.40 / 0.83 | 0.43 / 1.00 | 0.00 / 0.00 | 0.78 / 0.97 | 0.68 / 0.83 |
| Hex standoff | 0.78 / 0.83 | not certified (0/8, 6/8) | 0.68 / 0.83 | 0.56 / 0.67 | 0.00 / 0.00 | 0.85 / 1.00 | 0.00 / 0.00 |
| L-bracket | 0.46 / 0.67 | not certified (0/8, 6/8) | 0.51 / 0.67 | 0.00 / 0.67 | 0.00 / 0.00 | 0.56 / 0.67 | 0.00 / 0.00 |
| Pipe tee | 0.68 / 0.83 | not certified (0/8, 0/8) | 0.56 / 0.67 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.28 | 0.00 / 0.00 |
| Steel slug | 1.00 / 1.00 | not certified (0/8, 0/8) | 0.58 / 0.83 | 0.65 / 0.68 | 0.14 / 0.44 | 1.00 / 1.00 | 0.14 / 0.44 |

The random control's best entries are suction cups on block, steel slug and pipe tee, a hook
gripper on flanged bushing, and nothing that held on hex standoff or L-bracket.

## 4. Nemotron designs and the random control

| Part | Nemotron designs screened | Best Nemotron registered / held | Nominal held | Random designs screened | Best random registered / held | Nominal held |
|---|---|---|---|---|---|---|
| Block | 6 | 0.56 / 0.67 | yes | 8 | 0.14 / 0.44 | yes |
| Flanged bushing | 4 | 0.78 / 0.97 | yes | 8 | 0.68 / 0.83 | yes |
| Hex standoff | 7 | 0.85 / 1.00 | yes | 8 | 0.00 / 0.00 | no |
| L-bracket | 5 | 0.56 / 0.67 | yes | 8 | 0.00 / 0.00 | no |
| Pipe tee | 4 | 0.00 / 0.28 | no | 8 | 0.00 / 0.00 | no |
| Steel slug | 7 | 1.00 / 1.00 | yes | 8 | 0.14 / 0.44 | yes |

"Nominal held": a certified design of the group holds the part in the certificate's nominal
test. B3's physics gate (the best Nemotron design held at nominal on at least 4 of the 6
parts) is met: 5 of 6, with each part's own designs.

On these numbers the Nemotron designs score at least as high as the random control on every
part and higher on five. **We make no superiority claim.** The documented rule for this layer
(`docs/EFFECTORS.md` sections 11.4 and 12) is product mode, no statistics, certificate numbers
only, the random control shown beside Nemotron. The comparison is also confounded:

* One tournament per part, one seed, six parts.
* The groups are not the same size: 15 of Nemotron's 48 requested designs did not become
  candidates (refused by the validators, mostly the opening rule `too_narrow`, or by the
  duplicate and too-similar screens), so it screened 33 designs to the control's 48.
* The two arms drew from the shared spaces very differently. The random control drew 24
  suction cups, 12 soft-arm tentacles, 7 hook grippers and 5 fingered hands (the validators
  refuse more random fingered designs, and the suction and soft-arm spaces joined the shared
  space when those families landed). Nemotron chose 14 fingered hands, 5 segmented tentacles,
  3 hook grippers, 6 cups and 5 soft-arm tentacles. Much of the gap is "fingered designs grip
  these parts better than cups", not "Nemotron's parameters beat random ones".
* Nemotron's prompt describes the part; the random draw is blind to it. That is the point of
  the designer, but it is not the comparison the pre-registered experiment makes.

Two parts are won by a Nemotron design (hex standoff for both requirements, L-bracket for
registered), and a third ties the best jaw (steel slug). Treat these as design leads to verify,
not as evidence about the designer.

## 5. Honest notes

**The dexterous hands lose with scripted grasps.** No hand reached its group's top 2 on any
part, so none was certified. Their best screens: block 8/8 held, 3/8 registered (the
four-finger hand's power grasp, tied with five grippers on screen success and behind them on
the finalist tie-break, 16 actuators against 1 to 3); flanged bushing 8/8 held, 0/8 registered;
hex standoff and L-bracket 6/8 held; pipe tee and steel slug 0/8. The hand episodes mostly end
in a slip during the lift, an ejection on closing or a missed grasp. The hands run open-loop
scripted synergies with one grip height per model, no tactile feedback and no regrasp, from a
top-down 4-DoF wrist, at their upstream actuator limits, on parts whose friction (mu 0.3)
governs every contact. That is a test of these grasps, not of what the hands could do with a
controller. Founder decision: they stay in at upstream gains as honest entrants.

**The rigid-pad gripper A has low grip at its published settings.** Its upstream model
limits the actuator to 4.1 N, left unmodified. It slips during the lift on every screen
episode on block, hex standoff, L-bracket and steel slug, holds 3/8 on pipe tee, and holds
8/8 (3/8 registered) on flanged bushing, where it is a certified catalog finalist (0.40
registered, 9/18; 0.67 held, 16/18). Every page that shows it prints the note.

**Soft arm and suction are approximations.** Neither won anything, so no rigid runner-up was
needed.

* Suction: a geometric seal test gating an adhesion actuator capped at vacuum x lip area (no
  air flow, leaks, porosity or vacuum build-up). It seals and holds on the two parts with a
  flat face under the cup, block and steel slug (certified 0.44 held, 0.14 registered; the
  part ends off centre), and reports no seal on flanged bushing, hex standoff, L-bracket and
  pipe tee (every screen episode), as the B2 gate expects for bores, holes and curved faces.
* Soft arm: rigid capsules on spring-damper joints curled by a 50 N cable, hung beside the
  part from a rigid post that takes part of the grip. It holds well on flanged bushing (18/18
  held for the 16 and 20 segment arms) and steel slug, registers poorly (the part ends off
  centre or tilted), and is pushed off the pipe tee as it closes.

**Physics validity.** 0 simulator resets in 3,141 episodes. 42 episodes (1.3 %) were
`physics_unreliable` (more than 1 mm effector-part penetration while held), spread over the
jaw, hands, grippers, soft arm and designed morphologies; they count as failures. The fairness
signature (wrist servos and force limits, solver, the part's contact parameters, phase
timings, conditions and seeds, read from every compiled model) is identical across all
candidates on every part.

**Determinism.** The tournament was run five times. Run 1 (live Nemotron calls) and run 2
(`--compare`, from the response cache) gave identical digests on all six parts. A wording fix
to the recommendation text (a full tie was explained as "higher screen success" when the
screens were equal) changed the digests of flanged bushing and steel slug only; their records
and `traj_sha` values were unchanged. Runs 3 to 5 then reproduced each other on all six parts.

| Part | Digest (published) |
|---|---|
| Block | `6e038abe37b6` |
| Flanged bushing | `fb63184da51e` |
| Hex standoff | `3258c6c652b8` |
| L-bracket | `4f4fb34ba42e` |
| Pipe tee | `55a852de6a39` |
| Steel slug | `8b444988647c` |

## 6. Costs

* **Token Factory** (Nemotron 3 Super, `nvidia/nemotron-3-super-120b-a12b`, temperature 1.0,
  8,192 output tokens): 23 live calls in run 1 (12 proposals, 11 repairs), 188,427 prompt and
  25,864 completion tokens, **$0.0798** against the $0.50 allowed. Per part $0.011 to $0.015.
  Runs 2 to 5 replayed the cache: 0 live calls, $0. Every call succeeded.
* **Compute:** this laptop's CPU, 8 worker processes: 60 to 122 s per part, about 10 minutes
  for the six parts. No cloud compute.

## 7. Reproduce

```bash
python -m rtt.effectors tournament --part all --workers 8 --design --spend-usd 0.5
python -m rtt.effectors tournament --part all --workers 8 --design --spend-usd 0.5 --compare
python -m rtt.effectors export-web --runs runs/effectors --web web/public/data
```

`--design` spends money on Token Factory (`NEBIUS_API_KEY`); a re-run with the same response
cache (`runs/effectors/llm_cache/`) spends nothing. Outputs: `runs/effectors/<part>/`
(`records.parquet`, `certificates/`, `tournament.json`, `recommendation.json`), git-ignored.
