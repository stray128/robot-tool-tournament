# Robot Tool Tournament

NVIDIA Nemotron models on Nebius Token Factory design parallel-gripper fingers. Every
design is built from 14 boxes per finger, validated, and scored in MuJoCo physics on the
same grasping tasks as hand-designed fingers and randomly sampled ones. Scored batches run
as Nebius Serverless AI Jobs from one pinned container image, and a static web app replays
every bout in the browser.

Status: work in progress (hackathon build, October 2026). The protocol is pre-registered
before any scored run; see `docs/TASK.md` for the frozen task.

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
| `rtt job {smoke,m1,search}` | Runs the same work as a Nebius Serverless AI Job in the pinned image. It picks a platform with capacity, waits, records the spend, then fetches and verifies the output. Settings come from `~/.config/rtt/`; no account IDs live in the repository. |
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
- `src/rtt/pipeline/`: slice, held-out scoring, replay recording, web export.
- `src/rtt/analysis/`: calibration (gate G3).
- `configs/`: task, training suite, manual entrants.
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

MuJoCo and MuJoCo Warp are by Google DeepMind (Apache-2.0).

## License

Apache-2.0. See `LICENSE`.
