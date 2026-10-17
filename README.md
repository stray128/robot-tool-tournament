# Effector Lab

[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![Pre-registered: prereg-v1](https://img.shields.io/badge/pre--registered-prereg--v1-2da44e.svg)](https://github.com/stray128/robot-tool-tournament/tree/prereg-v1)
[![Live site](https://img.shields.io/badge/live%20site-stray128.github.io-24292f.svg)](https://stray128.github.io/robot-tool-tournament/)

**Bring your part. NVIDIA Nemotron designs gripper fingers and other end-effectors for it,
Nebius runs the physics simulation that keeps or rejects each design, and you get a
robustness certificate and a print pack.**

Live site: https://stray128.github.io/robot-tool-tournament/ (static: recorded bouts,
designs, certificates and effector results, replayed in the browser). Hackathon track:
Physical AI.
Everything here is simulated in MuJoCo. Nothing has been printed or tested on hardware yet.

## The problem

When a robot cell gets a new part, its gripper usually needs new fingers. Integrators and
cell builders find them by drawing, printing, testing and drawing again. The common
shortcut is an imprint: a jaw with the part's outline cut into it. It fits when the part
sits exactly where it should. It can jam when the part arrives a few millimetres off.

Effector Lab moves that loop into simulation. One simulated design stands in for one
print-and-test cycle. Every design is tested with the part placed off centre, shaken and
pushed, before anyone prints it.

## What it does

1. **Bring Your Part.** Upload an STL (also OBJ, PLY or OFF, 5 mm to 0.5 m, up to 2 kg).
   The intake finds the units, mass, convex pieces and resting poses, then builds the test
   conditions.
2. **Nemotron designs.** NVIDIA Nemotron 3 Super on Nebius Token Factory proposes finger
   designs through a forced tool call. It reads the measured results of its earlier
   designs before it proposes the next round. In the pre-registered space a design is a
   part-shaped recipe (a pad strategy, a tip and 7 numbers) that compiles into its own pair
   of jaws for each part.
3. **Physics decides.** Validators check geometry and printability before any simulation.
   MuJoCo then scores every design under placement error. A grasp counts only if the part
   is seated and centred before the lift ("registered") and still held 2 s later.
4. **The effector tournament** (product mode). For one part, every end-effector family
   competes on the same placements, seeds and scorer: parallel-jaw fingers, open-source
   multi-finger hands and grippers, a soft arm (tentacle), a suction cup, and effectors
   that Nemotron designs (fingered hands, hook grippers, segmented tentacles). The top two
   of each group are certified and one effector is recommended. Results:
   [`docs/EFFECTORS_RESULTS.md`](docs/EFFECTORS_RESULTS.md).
5. **Robustness certificate.** 18 tests: placement errors up to 3 mm on both axes, a 1 g
   shake on three axes and a push at the part's top.
6. **Print pack.** Left and right finger STLs, a printable jaw, the certificate as HTML and
   JSON, and a manifest with a checksum per file ([`docs/PRINT_PACK.md`](docs/PRINT_PACK.md)).

The **Grasp Engineer** agent (Nemotron 3 Super, tool-calling) runs the same steps for an
uploaded part within hard caps: 12 tool calls, 2 jobs in flight, a dollar cap per session
and per day, and a passcode for anything that spends ([`docs/AGENT.md`](docs/AGENT.md)).

## Result: the pre-registered experiment

The question was fixed and tagged (`prereg-v1`) before any confirmatory job ran: with the
same budget of 40 simulated designs per run, does Nemotron with simulator feedback reach a
finger that registers the part in at least 37 of 48 training grasps in fewer designs than
equal-budget random search? 32 runs per arm, seeds 1001 to 1032, all on Nebius Serverless
AI Jobs. Full report: [`docs/RESULTS.md`](docs/RESULTS.md). Protocol:
[`docs/PREREG.md`](docs/PREREG.md). Every change after the tag:
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

| Primary endpoint (S1) | Random search | Nemotron with feedback |
|---|---|---|
| Runs that reached the target within 40 designs | 11 of 32 | 23 of 32 |
| Mean designs to the target (not reached counted as 41) | 34.5 | 25.6 |

**Nemotron needed 8.97 fewer designs on average (95% interval 3.53 to 14.16). One-sided
exact permutation test: p = 0.0011.**

Held out: each run's best design was frozen, then scored on suites that no search process
could read.

| Secondary endpoint (Holm family) | Random | Nemotron | Difference | Holm-adjusted p |
|---|---|---|---|---|
| H1: registered success, 100 unseen procedural parts x 2 seeds | 60.5% | 72.5% | +12.0 points | 0.00017 |
| H2: registered success, 64 real scanned objects | 22.0% | 27.5% | +5.5 points | 0.00077 |
| S2: area under the best-so-far curve | 0.703 | 0.788 | +0.085 | 0.00017 |

What this does not say: with lift-only scoring (the part only has to leave the table) the
arms do not differ (p = 1.0). Both find a finger that lifts at once. The gain is in seating
and centring the part. Real scanned objects are hard for every design in this space: the
better arm registers about a quarter of them.

## How NVIDIA Nemotron and Nebius are used

![Effector Lab architecture: a part goes through intake, Nemotron on Nebius Token Factory proposes designs, validators check them, Nebius Serverless AI Jobs simulate them on CPU, and measured results feed the next round. Outputs are a certificate, a print pack and a replay site.](docs/architecture.svg)

| Piece | What it does here | Evidence |
|---|---|---|
| **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) on **Nebius Token Factory** | The design proposer, through a forced `submit_designs` tool call with reserves and one repair call; the Grasp Engineer's tool-calling loop; the effector designer | Confirmatory arm C: 473 calls from inside one job, 0 errors, 6.98M prompt and 1.64M completion tokens, **$3.57**. Effector designer: 23 calls, $0.080 |
| **NVIDIA Nemotron 3.5 Lightning** (`nvidia/Nemotron-3_5-Lightning`) | Short plain-language summaries of agent steps. A summary that claims a pass the code did not compute is dropped | about $0.00005 per summary ([`docs/FEEDBACK.md`](docs/FEEDBACK.md)) |
| NVIDIA Nemotron 3 Nano and 3 Ultra | Probed beside Super and Lightning before choosing the proposer (output modes, latency, cost) | 28 calls, $0.039 ([`docs/PROBE_LLM.md`](docs/PROBE_LLM.md)) |
| **Nebius AI Cloud: Serverless AI Jobs** | Every scored batch: uk-south2, `cpu-d3`, 32 vCPU, CPU MuJoCo as the executor of record, one image pinned by digest. `rtt job` submits, waits, records the spend, then fetches and verifies the outputs | Arm A `aijob-e05yxb1awgg2d68hq1` (wall 30 min, $0.52), arm C `aijob-e05s3f8ka6r508p4mk` (wall 86 min, $1.51), held-out `aijob-e05nvfme46ss0q4wp4` (wall 30 min, $0.53): 132,320 simulated grasps |
| **Object Storage** | A bucket per region, mounted at `/out` in each job: tables, replays, manifests | every job's outputs fetched and verified ([`docs/NEBIUS.md`](docs/NEBIUS.md)) |
| **Container Registry** | One registry per region, holding the same image digest | search image `sha256:c6162fa0...`, held-out image `sha256:b7d54d75...` |
| **MysteryBox** | The Token Factory key reaches jobs as a secret by reference (`--env-secret`), never on the command line or in the image | arm C's 473 calls used it |

The client in `src/rtt/llm/` caches every Token Factory request and response by SHA-256
and checks a dollar cap before each call. A cached run replays with zero API calls
(`rtt replay-llm`). The whole project's AI Cloud ledger is about **$4.12** (17 jobs,
wall-time estimates). The per-job table is in [`docs/NEBIUS.md`](docs/NEBIUS.md), and the
platform feedback is in [`docs/FEEDBACK.md`](docs/FEEDBACK.md). MuJoCo Warp on NVIDIA Warp
was benchmarked as a GPU cross-check. It failed the determinism gate, so it never scores
([`docs/BENCH_MJWARP.md`](docs/BENCH_MJWARP.md)).

## Quick start

Requires Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node 20.9+ with pnpm.

**The site, on your laptop** (no key, no GPU, no cloud account):

```bash
git clone https://github.com/stray128/robot-tool-tournament.git && cd robot-tool-tournament
(cd web && pnpm install --frozen-lockfile)
make demo                      # builds the static site and serves it at http://127.0.0.1:4173/
```

**Python setup, tests and Bring Your Part:**

```bash
make sync                      # locked dependencies plus pytest, hypothesis and ruff (uv)
source .venv/bin/activate
make test                      # full test suite on CPU
rtt part run assets/parts/l_bracket.stl --arm random --budget 16 --out runs/part/demo
                               # intake, search, certificate, print pack; no key, $0
```

**A search on your laptop** (the pre-registered design space, one run of 40 designs):

```bash
rtt search --space freeform-v2 --arm random --runs 1 --budget 40 --seeds 101 --out runs/search/my-random
rtt search --space freeform-v2 --arm nemotron --runs 1 --budget 40 --seeds 101 --spend-cap 0.25 \
  --out runs/search/my-nemotron  # needs NEBIUS_API_KEY in .env (see .env.example); about $0.11
rtt replay-llm --from runs/search/my-nemotron --out runs/search/my-nemotron-replay   # 0 API calls
```

**A Nebius Serverless AI Job.** The one-time setup (CLI profile, bucket, registry,
MysteryBox secret) is in [`docs/NEBIUS.md`](docs/NEBIUS.md). Account settings live in
`~/.config/rtt/`, never in the repository.

```bash
make image && make push        # build from a clean commit; push the same digest to each region
rtt job search --space freeform-v2 --arm random --runs 1 --budget 40 --seeds 101 --vcpus 32 --dry-run
rtt job search --space freeform-v2 --arm random --runs 1 --budget 40 --seeds 101 --vcpus 32
```

**The effector tournament** on the six training parts:

```bash
python -m rtt.effectors tournament --part all --workers 8              # no Nemotron rounds, $0
python -m rtt.effectors tournament --part all --workers 8 --design --spend-usd 0.5
python -m rtt.effectors export-web --runs runs/effectors --web web/public/data
```

## Reproduce the results

1. **Check the protocol.** `git checkout prereg-v1`. The frozen inputs and their SHA-256
   values are in PREREG A5, with the commands that recompute them. `make prereg-rules`
   prints the rule digests.
2. **Check the held-out commitment.** The seed is revealed in `docs/RESULTS.md`.
   `printf '%s' '<seed>' | sha256sum` must print
   `073055ffc30dd3a66a87be9bd5d6d7d639ce5bd258115cfd58d0e97fb86057d7` (PREREG A5).
3. **Re-run the jobs.** The exact submissions are in PREREG A4 (both search arms) and in
   `docs/DEVIATIONS.md` under "Open at the tag: the v2 held-out runners" (held-out).
   Token Factory has no seed parameter, so a new arm C run is a new draw from the model.
   The response cache in a fetched run replays it exactly with `rtt replay-llm`.
4. **Re-run the analysis** on the fetched outputs (`<A>`, `<C>` and `<H>` are run
   directories under `runs/cloud/`):

```bash
python -m rtt.analysis.tau analyze --random runs/cloud/<A> --nemotron runs/cloud/<C> \
  --tau 0.7708333333333334 --out runs/analysis/s1
python -m rtt.heldout_v2.secondary --random runs/cloud/<A> --nemotron runs/cloud/<C> \
  --heldout runs/cloud/<H>/procedural --heldout-real runs/cloud/<H>/real --out runs/analysis/secondary
```

The analysis refuses anything but the registered design: 32 runs per arm, the seeds of
`configs/seeds_main.yaml` in order, and the frozen tau.

## Limitations

- **Simulation only.** MuJoCo 3.15 on CPU, rigid contacts, friction 0.3 and 25 N per jaw
  assumed. No finger has been printed or tested on a real gripper.
- **One gripper and one task** in the pre-registered experiment: an open-source 84 mm
  parallel gripper, six training parts, seat, lift and hold.
- **Real objects are hard.** On 64 scanned household objects the better arm registers
  27.5% of grasps.
- **The soft arm and the suction cup are approximations.** The soft arm is rigid capsules
  on spring-damper joints. The suction cup is a geometric seal test gating an adhesion
  force capped at vacuum times lip area, with no air flow and no leaks.
- **The dexterous hands lose with scripted grasps.** They run open-loop synergies with no
  tactile feedback and no regrasp, and none reached a certificate on any part. That tests
  these grasps, not what the hands could do with a controller.
- **The effector tournament is product mode.** One tournament per part, one seed, no
  statistics. Nemotron's designs won the headline requirement on two of six parts and tied
  a third. We treat those as leads to verify, not as evidence about the designer.

## Roadmap (not built)

- **Train an open NVIDIA model on simulator rewards.** Fine-tune an open Nemotron model
  with reinforcement learning on the simulator's scores, on Nebius GPUs, served from
  Nebius Serverless Endpoints, so it designs end-effectors for any task: fingers first,
  then general attachments, then other task types.
- **Print and test with a partner.** Print the top designs and test them on a real gripper
  against the simulated certificate.

## Command line

| Command | What it does |
|---|---|
| `rtt part {intake,run,certify,printpack}` | Bring Your Part: intake, search, certificate, print pack |
| `rtt agent run --stl PART [--offline]` | One Grasp Engineer session in the terminal; `--offline` is a scripted agent with real physics and no model calls |
| `rtt serve` | The Grasp Engineer API on 127.0.0.1:8000; the site's `/part` page talks to it in live mode |
| `rtt search --arm {random,nemotron,zeroshot} --space freeform-v2 --runs R --budget B --out DIR` | R search runs in lockstep, each simulating exactly B valid designs; manifests, histories, Parquet tables and a frozen top design per run |
| `rtt replay-llm --from DIR --out DIR2` | Re-runs an LLM search from its response cache with zero API calls and checks that every run reproduces |
| `rtt job {smoke,m1,search,gv4}`, `python -m rtt.cloud run ...` | The same work as a Nebius Serverless AI Job: capacity check, submit by digest, wait, record the spend, fetch and verify |
| `rtt cloud {capacity,status,logs,list,fetch,spend}` | The rest of the job tooling |
| `python -m rtt.effectors {tournament,export-web,families}` | The effector tournament (product mode) |
| `python -m rtt.analysis.tau {derive,analyze,rule}`, `rtt analyze-secondary` | The pre-registered analysis |
| `rtt certify`, `rtt heldout`, `rtt heldout-real` | Certificates and the held-out suites |
| `rtt probe-llm` | The Token Factory output-mode probe report |
| `rtt export-web RUN_DIR...` | Publishes run directories into `web/public/data` for the site |

`make help` lists the gates (G1, GV0 to GV6) and the other developer targets.

## Repository layout

- `src/rtt/design/`, `src/rtt/design/freeform/`: finger schemas, generators, validators, STL and GLB export.
- `src/rtt/parts/`: the procedural industrial parts library.
- `src/rtt/sim/`: MuJoCo scene, controller, scoring, conditions, executors, certificate.
- `src/rtt/search/`: search arms, lockstep runner, the Nemotron proposer and its prompts.
- `src/rtt/llm/`: Token Factory client with response cache, spend ledger and rate limiting.
- `src/rtt/cloud/`: Nebius Serverless AI Jobs, Object Storage, cost ledger, in-container entry point.
- `src/rtt/analysis/`, `src/rtt/heldout_v2/`: the pre-registered analysis and the held-out runners.
- `src/rtt/part/`, `src/rtt/agent/`, `src/rtt/api/`: Bring Your Part, the Grasp Engineer and its API.
- `src/rtt/effectors/`: the effector layer (catalog, soft arm, suction, designed morphologies, tournament).
- `configs/`, `assets/`: tasks, suites, seeds, example parts (CC0), vendored effector models.
- `docker/`, `jobs/nebius/`: the job image and its build and push scripts.
- `web/`: the static Next.js site ([`web/README.md`](web/README.md)), deployed to GitHub Pages by `.github/workflows/pages.yml`.
- `docs/`: protocol, results, benchmarks, Nebius notes, platform feedback, Devpost draft.

## Secrets and leak hygiene

- **API keys** live only in `.env` (git-ignored, mode 600) and in Nebius MysteryBox for jobs.
- **Account settings** (project, subnet, bucket and registry IDs) live in `~/.config/rtt/`.
- **The held-out seed** stayed in a git-ignored file until the reveal. Only its SHA-256 was
  committed before the tag.
- **Leak check.** `scripts/leak_check.sh` checks the work tree, staged files or the full
  history against private patterns kept outside the repository, and reports only file and
  line. `make hooks` runs it before every commit and push.

## Credits and licences

Effector Lab is Apache-2.0 ([`LICENSE`](LICENSE)). Full third-party notices are in
[`NOTICE`](NOTICE) and [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md), and the site has a
credits page. On screen, every third-party robot model carries a generic name.

- **Gripper.** The simulated parallel gripper's opening, stroke, frame and stock finger
  were measured from the open-source SO-ARM-102 parallel gripper by Robonine
  (https://github.com/roboninecom/SO-ARM-102, commit 190eb162; simulation files CC BY 4.0,
  hardware sources CERN-OHL-P-2.0). The site draws its CC BY 4.0 simulation meshes,
  modified (cut, re-posed, simplified, re-encoded).
- **Effector catalog** (product mode), vendored from MuJoCo Menagerie (Google DeepMind,
  commit 62ab00f2), each with its licence and a `SOURCE.json` of pinned hashes in
  `assets/effectors/catalog/`: the Allegro Hand v3 by Wonik Robotics (BSD-2-Clause), the
  Shadow Hand E3M5 by Shadow Robot Company (Apache-2.0), the Sharpa Wave hand by Sharpa
  Robotics (Apache-2.0), the UMI Gripper by the Columbia Artificial Intelligence and
  Robotics Lab (MIT), the YAM arm's linear gripper by I2RT Robotics (MIT) and the PiPER
  arm's gripper by AgileX Robotics (MIT).
- **Real-object held-out set.** Google Scanned Objects (Google Research; Downs et al.,
  ICRA 2022, arXiv:2204.11918), CC BY 4.0, via Kevin Zakka's `mujoco_scanned_objects`
  (MIT, commit 6ff8d275). Scans are fetched at run time, and each selection writes
  per-object credits.
- **Software.** MuJoCo and MuJoCo Warp (Google DeepMind, Apache-2.0), NVIDIA Warp
  (Apache-2.0), CoACD (MIT), FastAPI (MIT), Next.js, React Three Fiber and three.js (MIT).
- **Models.** NVIDIA Nemotron 3 Super, Nemotron 3.5 Lightning, Nemotron 3 Nano and
  Nemotron 3 Ultra, served by Nebius Token Factory.

Related work: Fit2Form learns gripper-finger shapes with a 3D generative model (Ha, Agrawal
and Song, CoRL 2020, arXiv:2011.06498); imprint-based jaw design with an uncertainty sweep
for electrical connectors (Gebauer et al., Production Engineering 19:15-27, 2025,
doi:10.1007/s11740-024-01287-x); VLMgineer co-designs tools and their use with
vision-language models and evolutionary search (Gao et al., 2025, arXiv:2507.12644);
Text2Robot evolves robot designs from text (Ringel et al., 2024, arXiv:2406.19963).
