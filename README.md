# Robot Tool Tournament

NVIDIA Nemotron models on Nebius Token Factory design parallel-gripper fingers; every design
is built from 14 boxes, validated, and tested in MuJoCo physics against the same grasping
tasks as hand-designed and randomly sampled fingers.

Work in progress.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make sync        # uv sync --extra dev: locked dependencies plus pytest, hypothesis and ruff
make test        # full test suite; CPU only, no network
make m1          # gate G1: manual entrants plus deliberate failures, writes data/demo/m1/
```

## Command line

| command | what it does |
|---|---|
| `rtt m1` | Runs every manual entrant in `configs/entrants.yaml` and three deliberate failures on the G1 box, prints a results table, and writes a replay (`.bin` + `.json`) and an MP4 per case to `data/demo/m1/`. Options: `--no-video`, `--no-files`, `--json`, `--seed N`. |
| `rtt probe-llm` | Prints a summary of `docs/PROBE_LLM.md` and makes no API calls. `--live` re-runs the Token Factory probe (needs `NEBIUS_API_KEY` in `.env`; cached calls are free and live spend is capped); `--replay` rebuilds the report from the response cache without a key. |
| `rtt export-geom <entrant> --out finger.stl` | Validates a manual entrant and writes its finger as an STL in millimetres (`--mirror` for the right finger), or a palm-and-fingers preview for a `.glb` path. |

## Layout

- `src/rtt/design/`: the 10-parameter finger schema, 14-box generator, validators, STL/GLB export.
- `src/rtt/sim/`: MuJoCo scene, grasp controller, scoring, physics checks, replays, MP4 rendering,
  CPU process-pool executor and the G1 gate.
- `src/rtt/llm/`: Token Factory client with response cache, spend ledger and rate limiting, plus
  the output-mode probe.
- `configs/`: the simulation task (`task_v1.yaml`) and the manual entrants.
- `docs/`: probe results and platform feedback.

## License

Apache-2.0. See `LICENSE`.
