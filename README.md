# Robot Tool Tournament

**Bring Your Part:** upload the STL of a part and NVIDIA Nemotron designs printable
gripper fingers for it. Every design is validated, scored in MuJoCo physics, stress-tested
(placement error, shake, push) and exported as a print pack with a robustness
certificate.

Behind the product sits **the tournament**: a pre-registered benchmark in which
Nemotron with simulator feedback, random search and hand-designed fingers get the same
budget of simulated designs on the same grasping tasks. Scored batches run as Nebius
Serverless AI Jobs from one pinned container image, and a static web app replays every
bout in the browser.

Status: work in progress (hackathon build, October 2026). The protocol is pre-registered
before any scored run: the draft is `docs/PREREG.md` (not yet frozen), the frozen task is
`docs/TASK.md`, and changes after the freeze go in `docs/DEVIATIONS.md`.

## Bring Your Part (quick start)

```bash
make sync                                         # locked dependencies (uv)
rtt part intake assets/parts/l_bracket.stl        # what the simulator sees: units, mass, resting poses, grasps
rtt part run assets/parts/l_bracket.stl --arm random --budget 16 --out runs/part/demo
                                                  # intake -> search -> certificate -> print pack; no key, $0
rtt part run assets/parts/l_bracket.stl --arm nemotron --budget 16 --out runs/part/demo-nemo
                                                  # the same with Nemotron proposing (key in .env, spend cap $1)
rtt part certify runs/part/demo                   # re-run the robustness certificate on the run's best design
rtt part printpack runs/part/demo                 # rebuild print_pack.zip: finger + jaw STLs, certificate.html, manifest
```

Parts may be STL, OBJ, PLY or OFF, 5 mm to 0.5 m across and up to 2 kg; the narrow side
must fit the 84 mm opening. Example parts (CC0) are in `assets/parts/`. The print pack is
described in `docs/PRINT_PACK.md`. All results are simulated and untested on hardware.

**Grasp Engineer** (product mode, not part of the pre-registered experiment): a
Nemotron 3 Super agent that inspects the part, proposes designs, launches simulation
jobs, stress-tests the best one and exports the print pack, within hard caps (12 tool
calls, 2 jobs in flight, a dollar cap per session and per day, a passcode). See
`docs/AGENT.md`.

```bash
rtt agent run --stl assets/parts/l_bracket.stl --offline   # scripted agent, real physics, no model calls
RTT_AGENT_PASSCODE=... rtt agent run --stl assets/parts/l_bracket.stl
                                                            # live Nemotron (passcode checked against RTT_AGENT_PASSCODE_SHA256)
rtt serve                                                   # the API on 127.0.0.1:8000; the web /part page talks to it
```

## How Nebius and NVIDIA are used

| Piece | What it does here |
|---|---|
| **Nebius Token Factory** | Serves the NVIDIA models over an OpenAI-compatible API. Every request and response is cached by SHA-256, so a run replays with zero API calls (`rtt replay-llm`), and a spend ledger enforces a dollar cap before each call. |
| **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`) | The design proposer: reads its own run's simulator results and returns designs through a forced `submit_designs` tool call (`docs/PROPOSER.md`). Also the Grasp Engineer agent's tool-calling loop. |
| **NVIDIA Nemotron 3.5 Lightning** (`nvidia/Nemotron-3_5-Lightning`) | Short plain-language summaries of agent steps. |
| **Nebius AI Cloud Serverless AI Jobs** | Scored searches run in one pinned image, referenced by digest, on the uk-south2 CPU platform (`cpu-d3`): CPU MuJoCo is the executor of record. An H100 in eu-north1 is the fallback only. `rtt job ...` submits, waits, records the spend and fetches the outputs. |
| **Nebius Object Storage, Container Registry, MysteryBox** | One bucket and one registry per region for outputs and images; the Token Factory key reaches jobs as a MysteryBox secret. No account IDs live in this repository. |
| **MuJoCo Warp / NVIDIA Warp** | Batched GPU physics, benchmarked as a cross-check. It failed the determinism gate, so it never scores (`docs/BENCH_MJWARP.md`). |

Platform feedback, kept as we go, is in `docs/FEEDBACK.md`; job setup and prices in
`docs/NEBIUS.md`.

## How it works

1. **Design space.** A finger has 10 parameters: length, thickness, width, bend angle and
   position, inner face (flat, concave arc or V-groove) and its depth, tip profile (blunt,
   rounded, chamfer or hook) and tip size, and taper. It is generated as 14 boxes. Collision
   geometry equals visual geometry, and the STL you would print is the union of the same
   boxes.
2. **Task.** A floating parallel gripper with an 84 mm opening and 50 N grip lifts and
   holds parts. The training suite has 16 conditions: plastic housings, aluminium blocks
   and bars grasped off-centre, a standing pen, and thin rods lying flat, under 3 grasp
   heights. Success means lifting the part 100 mm and holding it for 2 s. See
   `docs/TASK.md`.
3. **Search arms.** All arms get the same budget of simulated designs:
   - random (scrambled Sobol);
   - Nemotron with feedback: Nemotron 3 Super proposes designs with a forced
     `submit_designs` tool call after reading its own run's results;
   - zero-shot Nemotron.
4. **Held-out evaluation.** Each run's top design is frozen. It is then scored on a
   commit-reveal held-out suite that no search process can read.

## Quick start (development)

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Node 20.9+ with pnpm for the web app.

```bash
make sync            # uv sync --extra dev: locked dependencies plus pytest, hypothesis, ruff
make test            # full test suite (CPU; GPU tests skip without CUDA or with RTT_SKIP_GPU_TESTS=1)
make lint            # ruff check + format check
make hooks           # install the leak check as git pre-commit / pre-push hooks
make m1              # gate G1: entrants + planted failures, replays and MP4s in data/demo/m1/
make calibrate       # gate G3: entrants + 64 random designs on the training suite (~2 min on 16 cores)
make web && make demo  # build the static site and serve it at http://127.0.0.1:4173/ (no key, no GPU)
```

A live Nemotron round needs a Token Factory key in `.env` (`NEBIUS_API_KEY=...`, never
committed; see `.env.example`):

```bash
rtt slice            # Super proposes 4 designs -> validate/repair -> simulate -> score ->
                     # manifest -> replays -> web/public/data (live spend capped at $0.20)
cd web && pnpm build # the new bout appears at /bout/<slice id>
```

## Command line

| Command | What it does |
|---|---|
| `rtt m1` | Gate G1. Each manual entrant must lift and hold the G1 box; three planted failures must fail with the right label. Writes replays and MP4s. |
| `rtt calibrate` | Gate G3. Scores the entrants plus 64 Sobol designs on the training suite. Prints per-family success and Sobol statistics against the targets, and writes `calibration.json` and an episode Parquet file. |
| `rtt search --arm {random,nemotron,zeroshot} --runs R --budget B --out DIR` | Runs R independent search runs in lockstep, each simulating exactly B valid designs on the training suite. Writes a manifest, per-run histories, Parquet tables and a frozen top-1 per run. The LLM arms cache every response and run under `--spend-cap` (USD). `--executor mjwarp` is available but not the executor of record (`docs/BENCH_MJWARP.md`). |
| `rtt replay-llm --from DIR --out DIR2` | Re-runs an LLM search from its response cache with zero API calls, and checks that every run reproduces. |
| `rtt heldout --seed-file PATH --frozen DIR... --out DIR` | Scores frozen designs and the entrants on the commit-reveal held-out suite. The seed must match the committed SHA-256. This is the only command that reads the seed. |
| `rtt certify --entrant NAME \| --design FILE [--object box:80x50x110]` | Robustness certificate: placement error of 0–3 mm, shake during hold, and a push at the part's top. |
| `rtt slice` | The vertical slice described above. |
| `rtt export-web RUN_DIR...` | Publishes slice, search and held-out run directories into `web/public/data`. Replays, print files (STL/GLB), provenance and jobs are included. |
| `rtt job {smoke,m1,search}` | Runs the same work as a Nebius Serverless AI Job in the pinned image. The default target is `uks2-cpu` (uk-south2 CPU, the executor of record); `--target eun1-gpu` picks the H100, `--vcpus N` the CPU preset size (and turns off the fallback), `--no-fallback` never moves to the H100. It waits, records the spend, then fetches and verifies the output. Settings come from `~/.config/rtt/`; no account IDs live in the repository. |
| `rtt part {intake,run,certify,printpack}` | Bring Your Part, as above. |
| `rtt agent run --stl PART [--offline]` | One Grasp Engineer session in the terminal; the session file, results and print pack land in `runs/agent/`. |
| `rtt serve` | The Grasp Engineer API on 127.0.0.1 (uploads, sessions with a live event stream, print packs). Binding beyond this machine requires the passcode for every upload. |
| `rtt heldout-real {select,run}` | The real-object held-out set from Google Scanned Objects: a deterministic, hashed selection rule, and scoring of frozen designs on it (`docs/HELDOUT_REAL.md`). |
| `rtt analyze-s1 --random DIR... --calibration FILE [--nemotron DIR...] [--power] [--compare]` | The S1 analysis: best-so-far curves, the tau rule, evaluations to tau per run, the exact permutation test and Monte Carlo power (`src/rtt/analysis/s1.py`). |
| `rtt cloud ...` | The rest of the job tooling: `capacity`, `status`, `logs`, `list`, `fetch`, `spend`. |
| `rtt probe-llm` | The Token Factory output-mode probe report (`docs/PROBE_LLM.md`). |
| `rtt export-geom NAME --out finger.stl` | One entrant's finger as an STL in millimetres, or a gripper preview as GLB. |

## Repository layout

- `src/rtt/design/`: finger schema, 14-box generator, validators, STL/GLB export.
- `src/rtt/sim/`: MuJoCo scene, controller, scoring, physics checks, replays, conditions (training suite and commit-reveal held-out), CPU and MJWarp executors, robustness certificate, mesh objects.
- `src/rtt/search/`: search arms, lockstep runner, Nemotron proposer (prompts, repair, diversity check), top-1 freezing.
- `src/rtt/llm/`: Token Factory client with response cache, spend ledger and rate limiting.
- `src/rtt/store/`: manifests, run tables and the held-out vault guard.
- `src/rtt/cloud/`: Nebius Serverless AI Jobs, Object Storage, cost ledger, in-container entry point.
- `src/rtt/pipeline/`: slice, held-out scoring (procedural and real-object), replay recording, web export.
- `src/rtt/analysis/`: calibration (gate G3) and the S1 analysis (tau rule, test, power).
- `src/rtt/part/`: Bring Your Part (intake, per-part conditions, search, certificate, jaw, print pack).
- `src/rtt/agent/`, `src/rtt/api/`: the Grasp Engineer agent and its local API.
- `configs/`: task, training suite, real-object held-out rule, manual entrants, seeds.
- `assets/parts/`: example parts (CC0).
- `docker/`, `jobs/nebius/`: the job image and its build/push scripts.
- `web/`: the static Next.js site (see `web/README.md`).
- `docs/`: task, benchmarks, Nebius notes, LLM probe, platform feedback.

## Secrets and leak hygiene

- **API keys** live only in `.env` (git-ignored, mode 600) and in Nebius MysteryBox for
  jobs.
- **Account settings** (project, subnet, bucket and registry IDs) live outside the
  repository, in `~/.config/rtt/`.
- **The held-out seed** lives in a git-ignored file. Only its SHA-256 is committed.
- **Leak check.** `scripts/leak_check.sh` checks the work tree, staged files or the full
  history. It uses private patterns kept outside the repository, plus the ID and key
  values from those files, and reports only file and line. `make hooks` runs it before
  every commit and push.

## Acknowledgements

The simulated gripper's opening, stroke, frame size and stock finger were measured from
the open-source SO-ARM-102 parallel gripper. That is Robonine, SO-ARM 102,
https://github.com/roboninecom/SO-ARM-102, at commit 190eb162. Its documentation and
simulation files are licensed CC BY 4.0, and its hardware sources CERN-OHL-P-2.0. The
`stock` entrant is our box approximation of its finger. This repository contains no files
from that project.

The real-object held-out set uses 3D models from Google Scanned Objects (Google Research;
Downs et al., "Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household
Items", ICRA 2022, arXiv:2204.11918), licensed CC BY 4.0, via Kevin Zakka's
`mujoco_scanned_objects` (MIT, commit 6ff8d275). The scans are fetched at run time and not
redistributed here; every selection writes per-object credits.

MuJoCo and MuJoCo Warp are by Google DeepMind (Apache-2.0); NVIDIA Warp is Apache-2.0;
CoACD is MIT. Full third-party notices are in `NOTICE`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
