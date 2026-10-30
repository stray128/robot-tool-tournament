# Web app

The public, static site: the "Bring your part" flow with the Grasp Engineer agent's
session timeline, recorded bouts replayed in the browser, finger designs with their print
files and robustness certificates, results and compute provenance. Next.js static export
(`output: "export"`), React Three Fiber for the 3D views, Tailwind CSS 4. No server:
everything it shows comes from `public/data/`, except in live mode (below), where `/part`
talks to a local API.

## Pages

| Route | What it shows |
|---|---|
| `/` | Hook, the three-step strip and live numbers computed from the published data (`lib/stats.ts`) |
| `/part` | Bring your part: STL drop zone, in-browser preview against the jaw opening, size readout, passcode, Run, and the agent timeline |
| `/results` | S1, the primary endpoint (best-so-far curves, designs-to-threshold dot plot), H1 held-out success as secondary; stamped `PILOT` or `SYNTHETIC` until the final analysis |
| `/certificate/<id>` | One design's robustness certificate: placement-error curve, every shake and push, provenance |
| `/benchmark`, `/bout/<id>`, `/design/<id>`, `/compute`, `/about` | Bouts, designs (now linking their certificate, with studio, x-ray, cutaway and turntable views of the jaws on the part), jobs, method |
| `/credits` | Third-party models (from `public/assets/manifest.json`), scans, software and fonts, with licences and what was changed. The short credits that assets and replays carry (`... (full credit: /credits#gripper)`, `#gso`) link to these anchors, and every clip that draws a third-party model shows its short credit with that link |

## Commands

Run from `web/` with Node 20.9 or newer and pnpm.

| Command | What it does |
|---|---|
| `pnpm install` | Install the pinned dependencies (`pnpm-lock.yaml`) |
| `pnpm import:demo` | With the repo's `.venv`: export the entrants' STL/GLB files and run the G1 gate under the current `configs/task_v1.yaml`, recording each case's replay in that same run into `public/data/replays/m1/`, then write `bouts.json`, `designs.json` and `jobs.json`. Fails, publishing nothing, if the gate does not pass. `--src DIR` copies an existing replay set instead; `--no-python` copies `data/demo/m1` and derives lift and hold from the replay frames. Copied replays recorded under a different gripper config (palm size, opening, frame rate) or an older entrant design are refused |
| `pnpm import:results -- --status final --threshold TAU --s1-analysis S1_JSON --secondary SECONDARY_JSON RUN_DIR...` | Build `results.json` (version 2) from search run directories: each run's `history.jsonl` becomes a best-so-far curve of `score.success` (in v2 the registered success: seated and centred, not lift alone), grouped into arms by the manifest's `arm`; with a threshold each run also gets S1's T, counted from one past round 0 (the manifest's first round size), with the run length plus one when never reached. `--s1-analysis` adds the confirmatory comparison from `s1_v2.json` after checking its tau, budget and every run's T against the histories; `--secondary` adds the Holm family and the planned H2 width split from `secondary_v2.json` (`--secondary-job` names its held-out job). The Nebius job id, platform, preset, region, wall and run time come from the run's `job.json`, the job cost from the published `jobs.json`, and the run's model-call spend (Token Factory, billed separately) from its `llm_calls.jsonl`. `--status pilot` without the analysis files still works. Refuses to overwrite a `final` file without `--force` |
| `pnpm import:certificate -- --in FILE.json [--engine "MuJoCo 3.15.0" --wall-s 13] [--job-id ID --platform P --preset P --region R --cost-usd X]` | Publish an `rtt certify --out` file as `public/data/certificates/<design_id>.json` with a `provenance` block (a local run, or a Nebius job when `--job-id` is given) |
| `pnpm import:session -- --in SESSION.json --stl PART.stl [--pack PACK_DIR] [--id ID --title T --note N]` | Publish a recorded agent session, optionally with its print-pack files (see "Agent sessions") |
| `pnpm make:session [--force]` | Write the illustrative demo session `public/data/sessions/demo-dumbbell.json` and its example part `sessions/parts/dumbbell-handle.stl`; refuses to overwrite a recorded session |
| `pnpm dev` | Development server |
| `pnpm build` | Static export to `out/`; fails if any data file is malformed, a lane's replay is missing or does not match its header, a design links a missing file, a session's part STL is missing, or a certificate does not parse |
| `pnpm start` | Serve `out/` locally at http://127.0.0.1:4173/ |
| `pnpm test` | Unit tests (replay decoding, overlays, data validation, session/SSE/STL/certificate/S1 parsers, import scripts). The GLB compression test runs only where the glTF Transform modules are available: in `web/node_modules`, or in the `node_modules` directory named by `RTT_GLTF_MODULES` |
| `pnpm typecheck` | TypeScript check |
| `node scripts/compress_glb.mjs SPEC.json [--map MAP.json]` | Meshopt-compress the GLBs a spec lists (WebP textures, optional simplification for scans) and add them to `public/assets/` as `<sha256[:16]>.glb` with their manifest entries; Identical output under a different role, licence, credit, change list or source is refused, and nothing is written unless every input passes. `--check` verifies the manifest against the files without any dependency. Compression needs `@gltf-transform/{core,extensions,functions}` and `meshoptimizer` (and `sharp` for textures), from `web/node_modules` or `--modules DIR` |
| `node scripts/smoke_v2.mjs` | After `pnpm build`: the v2 look in headless Chromium. A v1 bout still plays; the published bout of v2 lanes (`--v2-bout ID`, default the first bout of v2 parts) plays every lane under its own label and replay (design and part checked), with GLB decoding, captions, offset readouts, x-ray and collision toggles, the 300 000-byte payload per lane, credit links and no maker name on the page; then the design views and the credits page. Screenshots go to `.smoke/v2/` |
| `node scripts/smoke_library.mjs` | After `pnpm build`: the effector library pages in headless Chromium at 1280 px and 375 px: HTTP 200, no console errors, no horizontal scroll, the product-mode banner, no maker name and no URL in the visible text, one 3D viewer per item page, the catalog search filtering cards, and the library section of `/credits/`. Screenshots go to `.smoke/library/` |
| `node scripts/mock_runner.mjs [--port 8765]` | A tiny mock of the hosted runner's API (health, plan, runs, report; the runner's own body shapes) for the browser smoke tests and for trying the "Plan and run it live" panel by hand. Test data only |
| `node scripts/smoke_runner.mjs` | After a build with `NEXT_PUBLIC_RUNNER_URL=http://127.0.0.1:8765` (or `http://localhost:8765`): starts the mock and drives the hosted panel at 1440 px and 375 px through plan, run, result, a reload that resumes the run, a run with no recommendation, a failed run and each error, and proves the key is only in the body of `POST /plan`. Screenshots go to `.smoke/runner/` |
| `node scripts/smoke_site.mjs` | After `pnpm build`: every main route at 1440 px and 375 px. In a build without `NEXT_PUBLIC_RUNNER_URL` it also checks there is no hosted panel, no hosted docs section, the old home lede and no request to another origin; with it set it starts the mock and checks the panel, the docs section and the lede. `smoke_studio.mjs` and `smoke_studio_intake.mjs` start the mock the same way |
| `pnpm smoke` | After `pnpm build`: open every page in headless Chromium at 1280 px and 375 px; fail on console errors, horizontal scroll, replays that do not load or missing canvases. On `/part` it also drops the example STL (a preview canvas and size readout must appear) and plays the first demo session to its end (an illustrative session must be labelled illustrative, never recorded). Screenshots go to `.smoke/` |

For a GitHub Pages project site, build with `NEXT_PUBLIC_BASE_PATH=/<repo-name>`;
`.github/workflows/pages.yml` does this for https://stray128.github.io/robot-tool-tournament/.
`NEXT_PUBLIC_REPO_URL` adds a source link to every page footer.

The product name, Effector Lab, is `APP_NAME` in `lib/site.ts`; nothing else
hard-codes it.

## Data contract

`lib/types.ts` defines the index files and `lib/validate.ts` checks them at build time;
sessions and certificates have their own parsers (`lib/session.ts`, `lib/certificate.ts`).
Importers own entries by their `source` field and replace only their own, so the demo
importer and a later Nebius export can share the same files.

| File | Holds |
|---|---|
| `public/data/bouts.json` | Bouts: task constants (phases, success height and hold, jaw opening), compute provenance, and one lane per replay with the scorer's outcome and design provenance |
| `public/data/designs.json` | Parameter bounds, and per design: parameters, rationale, validation, mass, print-file paths, robustness certificate summary |
| `public/data/jobs.json` | Every simulation job: local or Nebius, platform, preset, region, times, cost |
| `public/data/results.json` | Version 2: `s1` (primary: threshold, budget B, `count_from` (the first design S1 counts), trials per design, the leading designs paired runs share, a best-so-far curve and T per run with its Nebius job id, and in a final file the confirmatory `analysis`), `secondary` (final file: H1, H2, S2 with raw and Holm-adjusted p, and the H2 width split) or the older `h1` forest block, `protocol` (the prereg tag and documents), `sources` (where each run ran). `status` is `synthetic`, `pilot` or `final` and is stamped on every chart until it is `final`. A version 1 file (`forest` + `curves`) is still read and upgraded. The committed file is the pre-registered confirmatory result (32 runs per arm, tau 37/48), imported from the two confirmatory search jobs with the S1 and secondary analysis outputs; a unit test checks that every number the page shows appears in `docs/RESULTS.md` |
| `public/data/certificates/<id>.json` | `rtt certify --out` documents (`rtt.certificates/1`: one certificate per object, each the nominal grasp, placement errors of ±1/±2/±3 mm along x and y, shakes and pushes) plus a `provenance` block. One `/certificate/<id>` page per file; a design page links its certificate. The committed four are the entrants on the nine training objects, run locally |
| `public/data/sessions/<id>.json` | Grasp Engineer sessions played by `/part`, in the agent's own format (see below); `kind` is `recorded` or `illustrative`, and an illustrative one is stamped as such. The committed `demo-dumbbell.json` is illustrative |
| `public/data/sessions/parts/*.stl` | The example part of each session (binary STL, mm) |
| `public/data/replays/**/<stem>.{json,bin}` | Replays as written by `src/rtt/sim/replay.py`: float32 `[frames x bodies x 7]` poses `(px, py, pz, qx, qy, qz, qw)` at 30 fps, MuJoCo Z-up. The viewer rotates its root group by -90 degrees about X once; nothing else converts axes |
| `public/data/replays/**/<stem>.json` (version 2) | The header of `src/rtt/sim/replay_v2.schema.json`, validated by `lib/replay.ts`: the same pose payload, bodies drawn from GLB assets (`bodies[].visuals`, positive scale only, so a right jaw is its own asset), optional v1-style geoms as the collision overlay, `meta` (design space, design, geometry and part ids), `provenance` (the caption) and an optional `<stem>.ovl.bin` of float16 readouts. The viewer reads versions 1 and 2 |
| `public/assets/<id>.glb`, `public/assets/manifest.json` | Content-addressed GLB assets (`id` = sha256[:16]), meshopt-compressed, Z-up metres in the body frame (no rotation, no mirroring), shared by every replay; the manifest format is `docs/ASSET_MANIFEST.md`. The build checks that every asset a replay draws is served with the listed size and sha256. Draco-compressed files are refused: no decoder is shipped and none is fetched from a CDN |
| `public/data/designs/<id>/` | `finger-left.stl`, `finger-right.stl` (mm) and `gripper.glb` (m, already Y-up, so not rotated) |

Provenance chips render only what the data says: a lane or certificate from a local run
is labelled as a local run, not a Nebius job.

## Agent sessions

`/part` plays sessions in the agent's own recording format (`src/rtt/agent/recorder.py`,
`format: "rtt-agent-session"`), and live mode reads the same events from the API's SSE
stream, so one normaliser (`lib/session.ts`) serves both. It reads defensively: unknown
fields are ignored, aliases are accepted (`event`/`kind` for `type`, underscores for
dots, `tool` for `name`, `arguments`/`input` for `args`, JSON-encoded strings), an
OpenAI-style `messages` transcript is used when there are no events, and an unreadable
event is dropped and counted instead of breaking the page.

Publish a recorded session with
`pnpm import:session -- --in runs/agent/sessions/sess-<id>.json --stl part.stl`: it
copies the part to `public/data/sessions/parts/<id>.stl`, adds `kind: "recorded"` and
`part.stl`, drops the chat transcript and refuses anything that looks like a local path, a Nebius
resource ID other than an AI job ID, or a container registry path (`import:certificate`
applies the same checks). With `--pack runs/agent/printpacks/pack-<id>/print_pack`, the
files the session's `export_print_pack` result lists are copied to
`public/data/printpacks/<pack_id>/` and linked from the print-pack card (the zip itself is
served only by the live API). Run `scripts/leak_check.sh` before committing either.
Recorded sessions are listed before illustrative ones; an `offline: true` session (the
scripted agent) is stamped as such.

A published session file:

```json
{
  "format": "rtt-agent-session", "version": 1, "session_id": "sess-82d7e038aa5b",
  "id": "sess-82d7e038aa5b", "kind": "recorded", "title": "dumbbell-handle", "note": "",
  "recorded_at": "2026-10-15T17:31:37.871Z", "offline": false,
  "part": { "name": "dumbbell-handle.stl", "stl": "sessions/parts/sess-82d7e038aa5b.stl", "units": "mm" },
  "events": [ { "seq": 1, "ts": "...", "t_s": 0.0, "type": "session.started" } ]
}
```

Events (`t_s` is seconds since the session started; without it, `ts` relative to the
first event):

| `type` | Fields read | Drawn as |
|---|---|---|
| `session.started` | `models` (`proposer`/`agent`), `caps` (`max_tool_calls`, `max_concurrent_jobs`, `session_usd`), `offline` | The meter's caps |
| `llm.call` | `role`, `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `latency_s`, `cached` | Spend; a proposer call is charged to its propose card |
| `agent.message` | `content` (empty turns are skipped) | Agent narration |
| `summary` | `text` (Nemotron Lightning's step summary) | Highlighted narration |
| `tool.call` | `call_id`, `name`, `args` | A card; `poll_job` folds into its job's card |
| `tool.result`, `tool.refused` | `call_id`, `name`, `ok`, `result`, `error`, `code` | The card's body; a refusal names the cap |
| `job.update` | `job_id`, `target`, `state`, `elapsed_s`, `est_usd`, `platform`, `preset`, `designs`, `error` | The job chip's live status (cost marked "est.") |
| `session.finished` | `status` (`completed`, `stopped`, `failed`), `reason`, `answer`, `summary`, `spend` | The outcome and final spend |
| `error` | `message` | An error line |

Tool results (`src/rtt/agent/tools.py`):

| Tool | Card | Result fields read |
|---|---|---|
| `inspect_part` | Inspect | `part` (`size_mm`, `grip_width_mm`, `fits_gripper`, `max_opening_mm`, `volume_cm3`, `mass_g`, `n_convex_pieces`, `stable_pose_probability`, `conditions`, `episodes_per_design`, `notes`), `pass_threshold_pct`, `baselines` |
| `propose_designs` | Nemotron hypotheses | `designs: [{design_id, hypothesis, params}]`, `rejected: [{slot, reasons}]`, `analysis` |
| `launch_sim_job`, `poll_job` | Job chip with live status | the job record, as in `job.update` |
| `fetch_results` | Results | `run_id`, `pass_threshold_pct`, `best_design_id`, `designs: [{design_id, label, success_pct, verdict, failure_modes, episodes}]`; a `replay` stem that a published bout contains links to it |
| `stress_test` | Stress test | `score_pct`, `verdict`, `category_pct`, `placement_tolerance_mm`, `passed`, `failed_tests`; links `/certificate/<id>` when the design has a published certificate |
| `export_print_pack` | Print pack | `pack_id`, `verdict`, `certificate_included`, `files: [{name, bytes}]`, `zip_bytes`, `download` |
| `cost_ledger` | Ledger | `spend`, `llm_by_model`, `jobs`, `tool_calls_used`, `caps` |

Any other tool gets a generic card with its raw result. `tests/fixtures/` holds a real
offline session (`python -m rtt.agent run --offline --fake-physics`) that the tests play.

## Live mode

Build or run with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` and start the API
(`python -m rtt.api`, with `RTT_AGENT_CORS` set to the site's origin if it is not one of
the API's default local dev origins). `/part` then runs your own STL through the agent.
Without the variable, which is how the public site is built, Run plays a recorded
session and says so; dropped files never leave the browser.

`lib/api.ts` follows `src/rtt/api/app.py`:

| Request | Sent | Used |
|---|---|---|
| `GET /health` | | mode, physics backend, targets, whether a passcode is configured |
| `POST /parts?name=&units=mm\|m` | the STL as the raw body (a file in cm or inches is re-encoded in mm first) | `part.part_id` |
| `POST /sessions` | JSON `{part_id, passcode}` and the `X-RTT-Passcode` header | `session_id` |
| `GET /sessions/{id}/events` | `Last-Event-ID` when resuming | SSE until `event: end`; a dropped stream resumes up to 3 times |
| `POST /sessions/{id}/stop` | | the Stop button (the agent stops after its current step) |
| `GET /sessions/{id}` | | read once after `event: end`, for any event the stream missed (the API can mark a session done just before it appends `session.finished`) |
| `GET /runs/{id}`, `GET /printpack/{id}` | | plain links from the results and print pack cards |

The passcode goes only to `POST /sessions`, never into a URL. Checked end to end against
the API in offline mode with synthetic physics: upload, session start, the streamed
timeline to `session.finished`, and the print pack link.

## Hosted runner (Plan and run it live)

An optional panel on the Studio planner plans a description with Nemotron and tests the plan in physics
on a hosted runner (the API is in `docs/RUNNER.md`). It is off unless the build has
`NEXT_PUBLIC_RUNNER_URL=https://<runner origin>` (plain `http` is accepted only for a loopback address,
so a key never travels in clear text). With it unset, which is how the public site is built, nothing
changes: no panel, no docs section, the home lede says what it always said, and no request leaves the site.
With it set, the panel shows only when `GET /health` answers `ok`; the quickstart and the plan-a-task
tutorial gain a "Plan and run it on this site" section, and the nine sentences that said live planning runs
only on your own machine, or that the site has no backend (home lede and Plan card, the question box, the quickstart
lede and note, the Studio's Explore mode tip, its Plan box and its custom-text message, the plan-a-task note) say that
you can also plan live here with your own key or the free allowance. Those sentences live in `lib/live_copy.ts`, each
as an approved `local` text and a `hosted` text, and are read through `liveCopy()`, the one place that checks the flag.

| Request | Sent | Used |
|---|---|---|
| `GET /health` | | whether to show the panel (`ok`), whether it is off (`kill_switch`, `open`), the plans and runs left today across the site (the panel says "across the site": a visitor's own limits are lower), the auto-off date |
| `POST /plan` | JSON `{description, provider: nvidia\|tokenfactory\|free, key}` | `status`, `plan` (loaded into the step editor like a replayed plan), `problems`, `calls`, `model` |
| `POST /runs` | JSON `{plan}` (the editor's plan, edits included) | `run_id` (kept in the URL hash as `#run-<id>`, so a reload resumes), `eta_s` |
| `GET /runs/{id}` | every 10 s, with backoff, until `done` or `failed` | `status`, `created_at`, `error`, `result` (the summary card) |
| `GET /runs/{id}/report` | the "Download the full report" link; redirects to a fresh presigned link | the full report JSON |

The runner still answers `/health` with 200 and `ok: true` after its auto-off date, so the panel shows its off notice when
`kill_switch` is true, or `open` is false, or the runner sends no `open` field (an older one) and `auto_off` is before today
in UTC (it serves through that date). When `open` is there it is the runner's word, whatever the date says. The time limits
are 8 s for `/health`, 330 s for a plan and 210 s for submitting a run (a plan the runner has not seen can take two to three
minutes to be checked before it answers 202), and the panel says so while it waits. A 429 that names a cap says which cap and
when it resets; a 429 with no `cap` is the hosting layer throttling a burst and reads "The runner is busy. Try again in a
minute." Every call is made with `redirect: "error"`, so a redirect of `POST /plan` can never carry the key to another address.

The key is in the JSON body of one `POST /plan` and nowhere else: it is kept in component memory only, is
cleared as soon as that request settles unless "Forget the key after each request" is unticked, and is
never put in storage, a URL, a header or the console. The key field is a password field, so a screen reader is
given bullets and the field cannot be copied from; it has autocomplete off (not `new-password`, which makes browsers
offer to generate a password), no name or id, no autocorrect or spell check, and the opt-out markers of the password
managers and of Grammarly and LanguageTool (`components/studio/runner/keyfield.ts`). A browser's own password
manager has no marker and may still offer to save it. A text field drawn as dots was tried and dropped: it exposes the
typed key to assistive technology. Run it against the mock:

```
NEXT_PUBLIC_RUNNER_URL=http://127.0.0.1:8765 pnpm build
node scripts/mock_runner.mjs --port 8765 &   # then: pnpm start
```

The code is `lib/runner.ts` (the client and the plain error messages), `components/studio/runner/`
(`controller.ts`: the state machine, tested without a browser; `HostedRunner.tsx`: the panel;
`keyfield.ts`: the key field's attributes; `handoff.ts`: the seam with the step editor), `components/docs/HostedDocs.tsx` and `lib/live_copy.ts`.
