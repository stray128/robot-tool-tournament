# Effector Lab

[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![Pre-registered: prereg-v1](https://img.shields.io/badge/pre--registered-prereg--v1-2da44e.svg)](https://github.com/stray128/robot-tool-tournament/tree/prereg-v1)
[![Pre-registered: usecase-v1](https://img.shields.io/badge/pre--registered-usecase--v1-2da44e.svg)](https://github.com/stray128/robot-tool-tournament/tree/usecase-v1)
[![Live site](https://img.shields.io/badge/live%20site-stray128.github.io-24292f.svg)](https://stray128.github.io/robot-tool-tournament/)

Effector Lab picks the tools for a robot job before anyone buys one. You describe a multi-step
job in plain words, such as "open the tote lid, put a carton and a flat pack in, close the lid".
NVIDIA Nemotron 3 Super, served by Nebius Token Factory, plans it as objects, steps and pass
criteria. MuJoCo physics then tests every end effector it can model (the tool at the end of the
arm: grippers, hands, suction cups and frames, soft arms, a nut runner, a tamper pad) on every
step, alone and in combinations, under varied conditions. The search can send its heavy runs to
Nebius Serverless Jobs, where the pre-registered benchmarks ran. The tool recommends a setup the way a robot integrator would, one tool by default, and the
website shows the recorded replays and the evidence.

- **Live site:** https://stray128.github.io/robot-tool-tournament/ (static, no login, no key).
- **Hackathon track:** Physical AI, Nebius x NVIDIA Global AI Hackathon.
- **Everything is simulated** (MuJoCo 3.15 on CPU). No setup has been built or tested on hardware.
  Verify on hardware.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The site needs Node 20.9+ with pnpm.
Docker is optional.

**Browse, with nothing installed.** Open the [Studio](https://stray128.github.io/robot-tool-tournament/studio/)
and pick a job. To plan and test a new job, run the same search on your own computer, as below.

**Install.**

```bash
git clone https://github.com/stray128/robot-tool-tournament.git && cd robot-tool-tournament
make sync                      # uv sync --extra dev: the locked dependencies, pytest and ruff
source .venv/bin/activate
make test                      # optional: the full test suite on CPU, no network
```

`pip install -e .` in a Python 3.12 virtual environment also works, without the lock file's
pinned versions of the dependencies' own dependencies.

**Plan a job with the Task Engineer** (the planner's command line). A recorded plan replays with
no key and costs nothing:

```bash
mkdir -p runs/studio
python -m rtt.studio.engineer plan --case h1_tote --out runs/studio/h1_plan.json    # replayed, $0
python -c "import json; d = json.load(open('runs/studio/h1_plan.json')); json.dump(d['plan'], open('runs/studio/h1_task.json', 'w'))"
```

To plan your own words live, put a Token Factory key in `.env` (see `.env.example`) and cap the
spend:

```bash
python -m rtt.studio.engineer plan --text "Open the drawer, put the cube in the tray, close it." \
  --live --env-file .env --budget 0.05 --out runs/studio/my_plan.json
```

On the site, [`/studio/new/`](https://stray128.github.io/robot-tool-tournament/studio/new/) edits a
plan and downloads it as a task file (`taskspec-<id>.json`) for the next step.

**Test every tool and get a recommendation.** No key, no cloud:

```bash
python -m rtt.studio.search runs/studio/h1_task.json --out runs/studio/h1 --pack
# a first look runs faster on fewer conditions and finalists:
python -m rtt.studio.search runs/studio/h1_task.json --out runs/studio/h1-quick --conditions 3 --finalists 2 --pack
```

It writes `report.json` (the plan, every candidate and why it was ruled out, the success matrix,
the combinations, the finalists, the recommendation and its reason) and, with `--pack`, the build
pack in `build_pack/` and `build_pack.zip`.

**The local API.** `rtt serve` serves the Task Engineer and the Grasp Engineer agent on
`http://127.0.0.1:8000` (`GET /health`, `POST /studio/plan`, `POST /studio/validate`, parts and
sessions). A recorded description is answered for free; anything else needs a passcode you set
(`RTT_AGENT_PASSCODE_SHA256`) and stays inside the spend caps in
[`docs/STUDIO_ENGINEER.md`](docs/STUDIO_ENGINEER.md).

**Docker.** The simulation image holds the same code and the locked dependencies (about 2 GB):

```bash
jobs/nebius/build_image.sh rtt-sim:dev
docker run --rm rtt-sim:dev m1 --no-files      # a quick check: gate G1 passes in seconds
docker run --rm -v "$PWD:/work" --entrypoint python rtt-sim:dev \
  -m rtt.studio.search /work/runs/studio/h1_task.json --out /work/runs/studio/h1-docker --pack
```

Files the container writes belong to root.

**The site on your laptop** (no key, no GPU, no cloud account):

```bash
(cd web && pnpm install --frozen-lockfile)
make demo                      # builds the static site and serves it at http://127.0.0.1:4173/
```

## How it works

Three layers: understand the job, test the tools, recommend a setup.

**1. Understand.** The Task Engineer is Nemotron 3 Super on Nebius Token Factory. Think of it as a
planner who writes a work order that a machine can check. It reads the description and returns,
in one forced tool call, a `TaskSpec`: the objects (each a template with bounded sizes and a
pose), the steps (each a primitive with its parameters) and the pass criteria of each step (a
measured quantity with bounds, such as "the lid angle at least 95 degrees"). Code checks every
rule and lists each problem with its path: objects that overlap, a wall object out of the wrist's
reach, a criterion the physics cannot measure, a step left holding a part while the next one acts
on another. One repair call gets the problems back, and a layout fixer moves objects that still
overlap. You can edit any step before anything runs. Every request and response is cached by
SHA-256, so recorded plans replay with zero calls.
([`docs/STUDIO_ENGINEER.md`](docs/STUDIO_ENGINEER.md))

**2. Test.** The search screens every candidate tool on every step it could do, on the first 3
conditions. Tools that cannot do a step are ruled out first, each with a reason. From the success
matrix it builds single tools, two-tool heads (two tools on one bracket, with a bracket geometry
search and a swept-volume check) and tool changes (the wrist swaps tools between steps). The
finalists run the whole job on the condition grid: by default 8 conditions, each with placement
errors up to 2 mm per object, friction varied by up to 10 % and its own seed, the same for every
setup. A fairness contract gives every candidate the same wrist, constants and seeds, checked by
hash. On Nebius, the search sends one Serverless Job per phase.
([`docs/STUDIO_SEARCH.md`](docs/STUDIO_SEARCH.md), [`docs/STUDIO.md`](docs/STUDIO.md))

**3. Recommend.** The ranking is completion first, then the weakest step, cycle time and
simplicity, under integrator rules (`src/rtt/studio/rules.py`; margins are assumed):

| Rule | What it says |
|---|---|
| R01 | One tool is the default. A more complex setup wins only when the single tool's weakest step succeeds in under half the runs, or it completes more than 5 points better per added piece. Cycle time never makes a more complex setup win. |
| R03 | Soft arms, tentacles and dexterous hands are kept off lids, doors, drawers and buttons, unless every rigid option fails. |
| R05 | Tightening to 4 N m or more is a nut runner's job; pressing a tile is a tamper's job. |
| R07 | The ranking is re-run with tool changes of 3 to 12 s and head switches of 0.75 to 3 s, and the report says whether the pick changes. |
| R09, R10 | Rigid setups first, then soft arms, then hands. No soft arm on an insert or on a part over 2.5 kg. |

The best single off-the-shelf tool is always shown beside the pick, with the difference. Every
step carries its fidelity tier: rigid, approximation ("verify on hardware") or not simulated. The
build pack holds the bracket or changer plates as STL, the exact tested scene (MJCF), a bill of
materials, assembly notes, a certificate and the licences.

**Replays.** Each replay re-runs a scored run with recording on, at 100 Hz, with a cut at every
tool change. The exporter's quality gate re-runs the physics at 500 Hz and refuses any lane that
strays more than 2 mm from it, jumps, or shows an overlap the physics did not have
(`src/rtt/studio/replay_qa.py`). The site draws a "Changing tool" card over the cut, because a tool
change is a fixed time and is not simulated.

## Building blocks

**12 object templates** (`src/rtt/studio/templates/`):

| Template | What it models | Tier |
|---|---|---|
| `hinged_lid` | a lid or a door on a hinge, with damping, a stop and an opening torque | rigid |
| `drawer` | a drawer on a slide with friction and stops | rigid |
| `quarter_turn_latch` | a latch that turns a quarter turn past a detent | approximation |
| `threaded_joint` | a cap, coupling or nut on a thread, with a torque that rises as it seats | approximation |
| `handwheel_valve` | a valve wheel with a resistive torque | approximation |
| `port_plug` | a plug in a socket with a chamfer and a clearance | rigid |
| `bin` | an open box, fixed or free | rigid |
| `part` | a free rigid part: a library or uploaded mesh, a box or a cylinder | rigid |
| `panel_rail` (new) | a 1000 x 600 x 35 mm, 8 kg panel on two rails with end stops, and a threaded clamp | approximation |
| `tile_bed` (new) | a 300 x 300 x 10 mm tile on a mortar bed that keeps its set | approximation |
| `spring_button` (new) | a push button on a preloaded spring, on a wall or a desk, that latches a call | rigid |
| `sliding_door` (new) | a machine guard door that slides sideways over its loading opening | rigid |

**7 primitives** (`src/rtt/studio/primitives/`), the basic moves a step can use:

| Primitive | What it does |
|---|---|
| `grasp`, `suction`, `wrap` | pick a part and set it down in a bin or back in place. They form one class: the tool's own mechanism decides how it holds (fingers, a seal, a curl). |
| `hook_pull` | pull a lid, drawer or door open by its handle or lip; a two-finger gripper tilts and pinches a lid's lip from the front |
| `twist` | turn a latch, a threaded member, a valve or a clamp with the wrist's roll |
| `insert` | push a plug or a threaded member to depth |
| `push` | push a lid, drawer, door, part, plug, tile or button, from the top or the front |

**The wrist.** A floating wrist with six servos: x, y and z slides, a yaw, a pitch of plus or
minus 90 degrees and a continuous roll (180 degrees per second each, 15 N m; assumed values, the
same for every tool). The palm travels plus or minus 350 mm; a step that would take it further
fails as "out of reach".

**Tool families** (`src/rtt/effectors/`):

| Family | Examples | Tier |
|---|---|---|
| Catalog grippers | parallel, linkage and adaptive two- and three-finger grippers, hook and pusher tools | rigid |
| Catalog hands | four- and five-finger hands | rigid |
| Suction | cups of 10 to 40 mm, and two vacuum frames (2 x 2 cups rated 4.15 kg, 2 x 3 cups rated 8.47 kg on a smooth face) | approximation |
| Soft arms | tentacle-like chains of segments curled by a cable | approximation |
| Designed effectors | hand-written recipes and Nemotron's designs: fingered hands, hook grippers, segmented tentacles | rigid |
| Part-shaped jaws | the parallel gripper with jaws cut to the part a step touches | rigid |
| Process tools (new) | a nut runner with a 13 mm socket (rated 6.7 N m, clutch at 10 N m), a 200 mm tamper pad (rated 40 N, capped at 60 N) | approximation |
| Open library | 19 entrants from 17 items of the open effector library | by item |

**What is not simulated.**

- **The arm.** The wrist floats. There is no arm reach or collision model beyond the palm's
  travel and an assumed keep-out for the arm's last link.
- **Tool changes and head switches.** They are a fixed 7.5 s and 1.5 s on the clock, with a
  sensitivity report (R07), and no docking motion.
- **Air and soft matter.** Suction is a geometric seal test that gates a capped holding force,
  with no air flow, leak or vacuum build-up. A soft arm is rigid segments on spring-damper joints.
  The tile bed is a grid of plastic cells.
- **Process details.** The nut runner's spindle is the wrist's roll. The panel clamp is a tall
  13 mm hex so the socket can seat; a standard M8 nut is shorter.
- **Deformables, liquids and perception.** Cloth, pouring and similar jobs are refused with the
  nearest primitive. Controllers are scripted and open-loop on the planned positions.

## Evidence

**This version's six jobs** (development search on 8 conditions, simulated, verify on hardware):

| Job | Steps | Recommended setup | Completes |
|---|---|---|---|
| [Returns tote](https://stray128.github.io/robot-tool-tournament/studio/h1_tote/) | open the lid, put a carton and a flat pack in, close the lid | One tool: an industrial servo parallel gripper, 110 mm stroke (also the best single off-the-shelf tool) | 8 of 8 |
| CNC machine tending | slide the guard door open, unload the part, load a blank, slide the door shut | Next: its plan failed the plan gate, so it was not searched | not run |
| [Solar panel install](https://stray128.github.io/robot-tool-tournament/studio/h5_solar/) | lift the panel onto its rails, start the clamp, tighten it to 6 N m | Tool change: vacuum area frame (2 x 3 cups, 600 mm), then a 10 mm suction cup, then a nut runner (13 mm hex socket) | 7 of 8 |
| [Floor tile setting](https://stray128.github.io/robot-tool-tournament/studio/h6_tile/) | lay the tile on the bed, tamp it level | Two-tool head: vacuum area frame (2 x 2 cups, 300 mm) and a 200 mm tamper pad | 8 of 8 |
| Pipe coupling | push the coupling onto its thread, turn it hand-tight | Not shown: replay held back (its replay failed the quality gate) | not shown |
| [Hotel delivery](https://stray128.github.io/robot-tool-tournament/studio/h7_hotel/) | press the call button, open the door, place the parcel | Tool change: a drawer-pull hook tool (button), a force-controlled 48 mm parallel gripper that grips the door's handle from the front (door), and a 25 mm suction cup (parcel); the best single tool finishes 0 of 8 | 8 of 8 |

The v2 benchmark, with held-out conditions for these jobs, comes after this release.

### 1. Use-case benchmark v1 (product mode, tag `usecase-v1`)

Does the setup the Studio recommends complete the whole job more often than the best single
off-the-shelf effector, on placements and frictions neither was chosen on? The arms were frozen
and the protocol tagged before any held-out run; the held-out secret was committed by SHA-256 and
revealed with the results. Each use case ran 32 new conditions (new seeds, friction jitter, hidden
placement errors up to 2 mm per object). Results:
[`docs/USECASE_V1_RESULTS.md`](docs/USECASE_V1_RESULTS.md). Protocol:
[`docs/USECASE_V1.md`](docs/USECASE_V1.md).

| Use case | a: best single off-the-shelf | b: Studio recommendation | c: library only | Primary test, b over a |
|---|---|---|---|---|
| H1 returns tote | 0/32 | **32/32**, two-tool head: soft arm and suction cup | 0/32 | p = 2.3e-10, Holm 7.0e-10 |
| H2 pipe coupling | 0/32 | **32/32**, two-tool head: fin-ray and linkage grippers | 16/32 | p = 2.3e-10, Holm 7.0e-10 |
| H3 equipment cabinet | 0/32 | **10/32**, tool change of three grippers | 12/32 | p = 0.00098, Holm 0.00098 |

The best single effector completed 0 of 96 held-out runs and the recommendation 74. Exact
one-sided paired test (exact McNemar), Holm across the three use cases: all three reject at
alpha 0.05. The evaluation ran as one Nebius Serverless Job on uk-south2 CPU
(`aijob-e05e55n905qj9va2fh`, 206 s, about $0.06), and its outcomes matched the laptop's on 72 of
72 development re-runs.

What this does not say. On H3 the recommendation finishes about a third of the runs, and the
library-only setup did slightly better (12 against 10, not tested). Every recommendation here is
approximation tier. **Dated note, 26 Oct 2026:** v1 used a wrist that could only point straight
down, so only soft arms could open the tote lid, and the soft arm in that recommendation says as
much about the wrist as about the job. The coupling's insert counted as done at "at most 4.0 mm",
the exact depth where the thread starts, and the published pass sat 6 micrometres inside it. The
tagged protocol, code and numbers stand as published.

### 2. The finger experiment (tag `prereg-v1`)

With the same budget of 40 simulated designs per run, does Nemotron with simulator feedback
reach a gripper finger that seats the part in at least 37 of 48 training grasps in fewer designs
than equal-budget random search? 32 runs per arm, seeds 1001 to 1032, all on Nebius Serverless
Jobs. Report: [`docs/RESULTS.md`](docs/RESULTS.md). Protocol: [`docs/PREREG.md`](docs/PREREG.md).
Every change after the tag: [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

| Primary endpoint (S1) | Random search | Nemotron with feedback |
|---|---|---|
| Runs that reached the target within 40 designs | 11 of 32 | 23 of 32 |
| Mean designs to the target (not reached counted as 41) | 34.5 | 25.6 |

Nemotron needed 8.97 fewer designs on average (95 % interval 3.53 to 14.16). One-sided exact
permutation test: p = 0.0011.

| Secondary endpoint (Holm family) | Random | Nemotron | Holm-adjusted p |
|---|---|---|---|
| H1: registered success, 100 unseen procedural parts x 2 seeds | 60.5 % | 72.5 % | 0.00017 |
| H2: registered success, 64 real scanned objects | 22.0 % | 27.5 % | 0.00077 |
| S2: area under the best-so-far curve | 0.703 | 0.788 | 0.00017 |

With lift-only scoring the arms do not differ (p = 1.0): the gain is in seating and centring the
part. Real scanned objects are hard for every design; the better arm registers about a quarter of
them.

## The open effector library

[`/library/`](https://stray128.github.io/robot-tool-tournament/library/) lists **29 items**:
open-source grippers, hands, suction grippers, tools and ISO 9409-1 mounting adapters whose
licences allow redistribution ([`docs/LIBRARY.md`](docs/LIBRARY.md)).

- Every item is pinned to an upstream commit, hashed (`lock.json`) and licence-gated. Green items
  ship with attribution. Amber (share-alike, copyleft) and red (non-commercial, no licence) items
  are excluded and listed with the reason (23 amber, 26 red).
- Each item says how it simulates: 16 native models, 5 generated proxies, 8 visual only.
  19 effector entrants from 17 items compete in the Studio's search.
- Our own ISO 9409-1 adapter generator makes the tool plates and adapters (deterministic STL, MJCF
  and JSON, every dimension tagged sourced, secondary or assumed).
- Makers are credited in [`NOTICE`](NOTICE), [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) and the
  site's credits page; on screen every item has a generic name.

## How Nebius and NVIDIA are used

![Effector Lab architecture: a task goes to NVIDIA Nemotron 3 Super on Nebius Token Factory, which plans it as objects and steps; code checks the plan and the user edits it; CPU MuJoCo screens every candidate tool and builds single tools, two-tool heads and tool changes; the search fans out as Nebius Serverless Jobs; outputs are a recommendation, a build pack and a static site.](docs/architecture.svg)

**Nebius Token Factory** (OpenAI-compatible API, NVIDIA open models):

| Model | Role | Evidence |
|---|---|---|
| **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) | **Planner:** the Task Engineer's forced `submit_task_plan` call plus one repair call | First Studio's recorded gate: 8 of 8 use cases planned validly, 2 of 2 unsupported ones refused with the nearest primitive; 17 calls, $0.133. This version's six jobs: 61 calls, $0.560558, two draft recordings included; gate: 5 of 6 plans valid (3 after one repair call, 2 more after the layout fix; the CNC plan failed) |
| | **Designer:** finger recipes through a forced `submit_designs` call, reading the measured results of earlier rounds; effector morphologies (hands, hook grippers, tentacles) | Finger experiment arm C: 473 calls from inside one job, 0 errors, $3.57. Effector designer: 23 calls, $0.080 |
| | **Agent:** the Grasp Engineer's tool loop (`plan_task`, `run_task_tournament`, `export_build_pack`) | 3 recorded Studio sessions, 3 tool calls each, $0.0037 to $0.0041 per session |
| **NVIDIA Nemotron 3.5 Lightning** (`nvidia/Nemotron-3_5-Lightning`) | Short plain-language summaries of agent steps; a summary that claims a pass the code did not compute is dropped | about $0.00005 per summary |
| NVIDIA Nemotron 3 Nano and 3 Ultra | Probed beside Super and Lightning before choosing (output modes, latency, cost) | 28 calls, $0.039 ([`docs/PROBE_LLM.md`](docs/PROBE_LLM.md)) |

**Nebius AI Cloud:**

| Service | Use | Evidence |
|---|---|---|
| **Serverless Jobs** | Scored simulation batches on uk-south2 `cpu-d3` (CPU MuJoCo is the executor of record), with an H100 in eu-north1 as the fallback only. `rtt job` and the Studio's fan-out (`rtt.studio.cloud`) check capacity, submit one image by digest, wait, record the spend, then fetch and verify the outputs | Finger experiment: arms `aijob-e05yxb1awgg2d68hq1` (29 min 43 s, $0.523) and `aijob-e05s3f8ka6r508p4mk` (85 min 56 s, $1.512), held-out `aijob-e05nvfme46ss0q4wp4` (30 min, $0.528): 132,320 simulated grasps. Use-case benchmark v1: `aijob-e05e55n905qj9va2fh`. This version's job searches ran on a local machine, not on Nebius |
| **Object Storage** | A bucket per region, mounted at `/out` in each job: tables, replays, manifests | every job's outputs fetched and verified ([`docs/NEBIUS.md`](docs/NEBIUS.md)) |
| **Container Registry** | One registry per region holding the same image digest | each scored image built from a tagged commit |
| **MysteryBox** | The Token Factory key reaches jobs as a secret by reference, never on the command line or in the image | used by arm C's 473 calls |

The client in `src/rtt/llm/` caches every Token Factory request and response by SHA-256 and checks
a dollar cap before each call, so a cached run replays with zero API calls. Before this version the
AI Cloud ledger came to about $4.18 for 18 jobs (wall-time estimates; per-job table in
[`docs/NEBIUS.md`](docs/NEBIUS.md), every published job on the site's
[`/compute/`](https://stray128.github.io/robot-tool-tournament/compute/) page). Platform feedback:
[`docs/FEEDBACK.md`](docs/FEEDBACK.md). MuJoCo Warp on NVIDIA Warp was benchmarked as a GPU
cross-check; it failed the determinism gate, so it never scores
([`docs/BENCH_MJWARP.md`](docs/BENCH_MJWARP.md)).

## Reproduce the results

**Use-case benchmark v1.** `git checkout usecase-v1` shows the frozen arms and the commitment in
`configs/usecase_v1.yaml`. The secret is revealed in
[`docs/USECASE_V1_RESULTS.md`](docs/USECASE_V1_RESULTS.md):
`printf '%s' '<secret>' | sha256sum` must print the committed
`8011661730eb0c5805065ca6ef5b9b4948dbe58db86e26c8333bd30f4e31196a`. The fetched output is committed
in [`results/usecase_v1/`](results/usecase_v1/) (`analysis.json`, `summary.txt`, `rows.jsonl`,
`conditions.json`). The whole procedure, including the evaluation job, is section 11 of
[`docs/USECASE_V1.md`](docs/USECASE_V1.md). This version changes the physics contract, so v1's
numbers reproduce at the tag, not on `main`.

**Finger experiment.**

1. `git checkout prereg-v1`. The frozen inputs and their SHA-256 values are in PREREG A5;
   `make prereg-rules` prints the rule digests.
2. The held-out seed is revealed in `docs/RESULTS.md`. `printf '%s' '<seed>' | sha256sum` must
   print `073055ffc30dd3a66a87be9bd5d6d7d639ce5bd258115cfd58d0e97fb86057d7` (PREREG A5).
3. The exact submissions are in PREREG A4 (both search arms) and in `docs/DEVIATIONS.md` under
   "Open at the tag: the v2 held-out runners" (held-out). Token Factory has no seed parameter, so
   a new arm C run is a new draw from the model; the response cache in a fetched run replays it
   exactly with `rtt replay-llm`.
4. Re-run the analysis on the fetched outputs (`<A>`, `<C>` and `<H>` are run directories under
   `runs/cloud/`):

```bash
python -m rtt.analysis.tau analyze --random runs/cloud/<A> --nemotron runs/cloud/<C> \
  --tau 0.7708333333333334 --out runs/analysis/s1
python -m rtt.heldout_v2.secondary --random runs/cloud/<A> --nemotron runs/cloud/<C> \
  --heldout runs/cloud/<H>/procedural --heldout-real runs/cloud/<H>/real --out runs/analysis/secondary
```

**A Nebius Serverless Job.** The one-time setup (CLI profile, bucket, registry, MysteryBox
secret) is in [`docs/NEBIUS.md`](docs/NEBIUS.md). Account settings live in `~/.config/rtt/`,
never in the repository.

```bash
make image && make push        # build from a clean commit; push the same digest to each region
rtt job search --space freeform-v2 --arm random --runs 1 --budget 40 --seeds 101 --vcpus 32 --dry-run
```

## Limitations

- **Simulation only.** MuJoCo 3.15 on CPU, rigid contacts, scripted open-loop controllers. No
  setup has been built or tested on hardware.
- **Approximations are labelled.** Suction seals, soft arms, vacuum frames, process tools,
  latches, threads, valves, the panel clamp and the tile bed are approximation templates or
  families. Every such step says "verify on hardware".
- **The wrist abstracts the arm.** See "What is not simulated" above.
- **Six jobs, development conditions.** This version's recommendations come from 8 development
  conditions per job; their held-out test comes next.
- **Fingers:** one gripper and one task in the pre-registered experiment; real scanned objects
  are hard (27.5 % for the better arm).

## Command line

| Command | What it does |
|---|---|
| `python -m rtt.studio.engineer {plan,gate,export-web}` | The Task Engineer: plan a use case (replayed, or `--live` with a spend cap), the recorded gate, the intake page's data |
| `python -m rtt.studio.search TASK.json --out DIR [--pack]` | The Studio search: screen, combinations, finalists, recommendation, build pack |
| `python -m rtt.agent.studio_tool record --case h1_tote --out DIR` | Record a Grasp Engineer session of a hero use case |
| `scripts/studio_usecase_v1.py`, `jobs/studio/usecase_v1.sh` | The use-case benchmark v1: freeze, commit, evaluate on Nebius, analyse |
| `scripts/export_studio_web.py` | Export search reports and sessions to the site, through the replay quality gate |
| `python -m rtt.library {check,gate,export-web}`, `scripts/vendor_library.py [--check\|--drift]` | The open effector library: verify, gate the entrants, export the catalog page, vendor at pinned commits |
| `rtt part {intake,run,certify,printpack}` | Bring Your Part: intake, finger search, certificate, print pack |
| `rtt agent run --stl PART [--offline]`, `rtt serve` | The Grasp Engineer in the terminal, or the local API on 127.0.0.1:8000 |
| `rtt search ...`, `rtt replay-llm ...` | Finger search runs in lockstep; replay an LLM search from its cache |
| `rtt job {smoke,m1,search,gv4}`, `python -m rtt.cloud {run,capacity,status,logs,list,fetch,spend}` | Nebius Serverless Jobs and their tooling |
| `python -m rtt.effectors {tournament,export-web,families}` | The per-part effector tournament |
| `python -m rtt.analysis.tau {derive,analyze,rule}`, `rtt analyze-secondary` | The pre-registered finger analysis |
| `scripts/video/{capture.mjs,assemble.sh,voiceover.py}` | The demo video kit: screen capture, AI voice-over, assembly ([`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)) |

`make help` lists the gates and the other developer targets.

## Repository layout

- `src/rtt/studio/`: the Studio: contract and scene (`types`, `scene`, `episode`, `fairness`),
  object templates and primitives, integrator rules (`rules`), the Task Engineer (`engineer/`), the
  search, runner, recommender and build pack, replays and their quality gate, and the Nebius
  fan-out (`cloud.py`).
- `src/rtt/library/`, `assets/effectors/library/`: the open effector library.
- `src/rtt/effectors/`: the effector families (catalog, soft arm, suction and vacuum frames,
  designed morphologies, part-shaped jaws, process tools) and the per-part tournament.
- `src/rtt/design/`, `src/rtt/parts/`, `src/rtt/sim/`, `src/rtt/search/`, `src/rtt/analysis/`,
  `src/rtt/heldout_v2/`: the pre-registered finger experiment (frozen at `prereg-v1`).
- `src/rtt/llm/`: Token Factory client with response cache, spend ledger and rate limiting.
- `src/rtt/cloud/`: Nebius Serverless Jobs, Object Storage, cost ledger, in-container entry point.
- `src/rtt/part/`, `src/rtt/agent/`, `src/rtt/api/`: Bring Your Part, the Grasp Engineer and its API.
- `configs/`, `assets/`, `results/`: tasks, suites, seeds, example parts, vendored models,
  committed results.
- `docker/`, `jobs/`: the job image and the Nebius job scripts.
- `web/`: the static Next.js site ([`web/README.md`](web/README.md)), deployed to GitHub Pages.
- `scripts/`: export, library vendoring, the benchmark driver, the leak check and the video kit.
- `docs/`: protocols, results, design notes, Nebius notes, platform feedback, Devpost draft, video
  script.

## Secrets and leak hygiene

- **API keys** live only in `.env` and `api_keys.json` (both git-ignored) and in Nebius MysteryBox
  for jobs. Scripts read them at run time and never print them.
- **Account settings** (project, subnet, bucket and registry IDs) live in `~/.config/rtt/`.
- **Held-out secrets** stayed in git-ignored files until the reveal; only their SHA-256 was
  committed before each tag.
- **Leak check.** `scripts/leak_check.sh` checks the work tree, staged files or the full history
  against private patterns kept outside the repository, and reports only file and line.
  `make hooks` runs it before every commit and push.

## Licence and credits

Effector Lab is Apache-2.0 ([`LICENSE`](LICENSE)). Full third-party notices are in
[`NOTICE`](NOTICE) and [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md), and the site has a credits
page. On screen, every third-party robot model carries a generic name.

- **Gripper of the finger experiment.** The simulated parallel gripper's opening, stroke, frame and
  stock finger were measured from the open-source SO-ARM-102 parallel gripper: Robonine,
  SO-ARM 102, https://github.com/roboninecom/SO-ARM-102, CC BY 4.0, at commit 190eb162. Its
  simulation files are CC BY 4.0 and its hardware sources CERN-OHL-P-2.0. The site draws its
  CC BY 4.0 simulation meshes, modified (cut, re-posed, simplified, re-encoded).
- **Effector catalog**, vendored from MuJoCo Menagerie (Google DeepMind, commit 62ab00f2), each with
  its licence and a `SOURCE.json` of pinned hashes in `assets/effectors/catalog/`: the Allegro Hand
  v3 by Wonik Robotics (BSD-2-Clause), the Shadow Hand E3M5 by Shadow Robot Company (Apache-2.0),
  the Sharpa Wave hand by Sharpa Robotics (Apache-2.0), the UMI Gripper by the Columbia Artificial
  Intelligence and Robotics Lab (MIT), the YAM arm's linear gripper by I2RT Robotics (MIT) and the
  PiPER arm's gripper by AgileX Robotics (MIT).
- **Open effector library**, 26 third-party items pinned by commit in `assets/effectors/library/`
  (plus our own three ISO 9409-1 plates and adapters), each with its licence texts and a
  `SOURCE.json` of hashes (`lock.json` pins the whole library). Simulation models from MuJoCo
  Menagerie (Google DeepMind, commit 62ab00f2): the Robotiq 2F-85 (v4) gripper by ROS-Industrial
  (BSD-2-Clause), the Franka Emika Panda hand (Apache-2.0), the Shadow DEX-EE hand by The Shadow
  Robot Company (Apache-2.0), the ALOHA gripper by Trossen Robotics (BSD-3-Clause), the SO-ARM100
  gripper by The Robot Studio (Apache-2.0), and the UFACTORY xArm and Lite 6 grippers
  (BSD-3-Clause). Open hardware: the SSG-48 gripper by Source Robotics (Apache-2.0), the ORCA hand
  by ORCA (MIT description, CC-BY-4.0 hardware), the MIDAS hand by its contributors (MIT), the
  SO-ARM100/101 parallel gripper by Robonine (CERN-OHL-P-2.0, Apache-2.0), the Schunk WSG 50 model
  by Toyota Research Institute (BSD-3-Clause), the xArm vacuum gripper description by UFACTORY
  (BSD-3-Clause), the robosuite suction gripper (MIT), Stretch Tool Share tools by Hello Robot
  (Apache-2.0), assembler0 parts by Jack Vial (MIT), ALOHA 2 fingers by Tony Z. Zhao (MIT),
  Grip4SO101 by Xiujin Liu (MIT), the norma-core pgripper (Apache-2.0), the XLeRobot soft fin
  (Apache-2.0), the co-designed soft gripper (MIT), the BCN3D Moveo gripper (MIT) and the
  AmazingHand by Pollen Robotics (CC-BY-4.0, Apache-2.0). Copyright licences cover the files, not
  the makers' patents or trademarks.
- **Real-object held-out set.** Google Scanned Objects (Google Research; Downs et al., ICRA 2022,
  arXiv:2204.11918), CC BY 4.0, via Kevin Zakka's `mujoco_scanned_objects` (MIT, commit 6ff8d275).
  Scans are fetched at run time, and each selection writes per-object credits.
- **Software.** MuJoCo and MuJoCo Warp (Google DeepMind, Apache-2.0), NVIDIA Warp (Apache-2.0),
  CoACD (MIT), FastAPI (MIT), Next.js, React Three Fiber and three.js (MIT).
- **Models.** NVIDIA Nemotron 3 Super, Nemotron 3.5 Lightning, Nemotron 3 Nano and Nemotron 3 Ultra,
  served by Nebius Token Factory. The use-case atlas was drafted from public sources with a Claude
  model and is labelled as research estimates. The demo video's voice-over is AI (OpenAI
  text-to-speech).

Related work: Fit2Form learns gripper-finger shapes with a 3D generative model (Ha, Agrawal and
Song, CoRL 2020, arXiv:2011.06498); imprint-based jaw design with an uncertainty sweep for
electrical connectors (Gebauer et al., Production Engineering 19:15-27, 2025,
doi:10.1007/s11740-024-01287-x); VLMgineer co-designs tools and their use with vision-language
models and evolutionary search (Gao et al., 2025, arXiv:2507.12644); Text2Robot evolves robot
designs from text (Ringel et al., 2024, arXiv:2406.19963).
</content>
</invoke>
