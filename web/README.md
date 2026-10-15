# Web app

The public, static site: recorded bouts replayed in the browser, finger designs with
their print files, results and compute provenance. Next.js static export
(`output: "export"`), React Three Fiber for the replays, Tailwind CSS 4. No server:
everything it shows comes from `public/data/`.

## Commands

Run from `web/` with Node 20.9 or newer and pnpm.

| Command | What it does |
|---|---|
| `pnpm install` | Install the pinned dependencies (`pnpm-lock.yaml`) |
| `pnpm import:demo` | With the repo's `.venv`: export the entrants' STL/GLB files and run the G1 gate under the current `configs/task_v1.yaml`, recording each case's replay in that same run into `public/data/replays/m1/`, then write `bouts.json`, `designs.json` and `jobs.json`. Fails, publishing nothing, if the gate does not pass. `--src DIR` copies an existing replay set instead; `--no-python` copies `data/demo/m1` and derives lift and hold from the replay frames. Copied replays recorded under a different gripper config (palm size, opening, frame rate) or an older entrant design are refused |
| `pnpm dev` | Development server |
| `pnpm build` | Static export to `out/`; fails if any data file is malformed, a lane's replay is missing or does not match its header, or a design links a missing file |
| `pnpm start` | Serve `out/` locally at http://127.0.0.1:4173/ |
| `pnpm test` | Unit tests (replay decoding, overlays, data validation, import script) |
| `pnpm typecheck` | TypeScript check |
| `pnpm smoke` | After `pnpm build`: open every page in headless Chromium at 1280 px and 375 px; fail on console errors, horizontal scroll, replays that do not load or missing canvases; screenshots go to `.smoke/` |

For a GitHub Pages project site, build with `NEXT_PUBLIC_BASE_PATH=/<repo-name>`.
`NEXT_PUBLIC_REPO_URL` adds a source link to every page footer.

The product name is a placeholder. Change `APP_NAME` in `lib/site.ts`; nothing else
hard-codes it.

## Data contract

`lib/types.ts` defines every file; `lib/validate.ts` checks them at build time.
Importers own entries by their `source` field and replace only their own, so the
demo importer and a later Nebius export can share the same files.

| File | Holds |
|---|---|
| `public/data/bouts.json` | Bouts: task constants (phases, success height and hold), compute provenance, and one lane per replay with the scorer's outcome and design provenance |
| `public/data/designs.json` | Parameter bounds, and per design: parameters, rationale, validation, mass, print-file paths, robustness certificate |
| `public/data/jobs.json` | Every simulation job: local or Nebius, platform, preset, region, times, cost |
| `public/data/results.json` | Forest-plot and best-so-far data; `status` is `synthetic`, `pilot` or `final`. The committed file is synthetic layout data from `scripts/make-synthetic-results.mjs`, stamped as such on the page |
| `public/data/replays/**/<stem>.{json,bin}` | Replays as written by `src/rtt/sim/replay.py`: float32 `[frames x bodies x 7]` poses `(px, py, pz, qx, qy, qz, qw)` at 30 fps, MuJoCo Z-up. The viewer rotates its root group by -90 degrees about X once; nothing else converts axes |
| `public/data/designs/<id>/` | `finger-left.stl`, `finger-right.stl` (mm) and `gripper.glb` (m, already Y-up, so not rotated) |

Provenance chips render only what the data says: a lane from a local run is labelled
as a local run, not a Nebius job.
