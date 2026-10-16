# Grasp Engineer (product mode)

The Grasp Engineer is a tool-calling agent. A user uploads a part, and the agent looks
for 3D-printable fingers that hold it. Nemotron 3 Super runs the loop and picks the tools.
MuJoCo decides what works. Nemotron 3.5 Lightning writes the short summaries the user
reads.

**This is product mode. It is not part of the pre-registered experiment.** Every session
file, run file, print pack and API response carries the label "Product mode: not part of
the pre-registered experiment". No session result feeds the search arms, the held-out
evaluation or any number in `docs/TASK.md`.

## How a session runs

1. **Upload.** `POST /parts` takes an STL. The `rtt.part` intake rests the part in its
   likely stable poses, splits it into convex pieces, picks grip candidates and builds
   the test conditions. The intake path is described in `src/rtt/part/`.
2. **Loop.** The agent (`nvidia/nemotron-3-super-120b-a12b`, thinking on, `tool_choice="auto"`)
   sees eight tools. It calls them through a plain OpenAI-style loop, which runs on the
   project's Token Factory client (`rtt.llm`). That client handles the cache, the ledger,
   retries and rate limits.
3. **Tools.** Each tool call is checked against the hard caps, then calls real code:

   | Tool | What it calls | Costs |
   |---|---|---|
   | `inspect_part(part_id)` | the part as simulated: size, grip width, mass, poses, conditions; the entrant fingers it can compare against | free |
   | `propose_designs(part_id, n)` | Nemotron 3 Super, one forced `submit_designs` call (1-4 designs); then the geometry validator and the diversity rule of the search arms | Token Factory |
   | `launch_sim_job(part_id, designs, target)` | `local`: CPU MuJoCo in a thread on this machine; `nebius`: a Serverless AI Job (`rtt.agent.remote` in the pinned image) | `nebius` only |
   | `poll_job(job_id)` | blocks up to 2 minutes for the job to finish | free |
   | `fetch_results(job_id)` | per design: success at the best grasp height with the number of trials behind it (`episodes`; `episodes_all_heights` counts every height), the rate at each height, failure modes, mass, and the **verdict computed by code** | free |
   | `stress_test(design_id, part_id)` | the robustness certificate on every kept pose: placement error of 0 to ±3 mm on both axes, a shake and a push | free |
   | `export_print_pack(run_id)` | the `rtt.part` print pack: finger STLs, a printable jaw, the certificate as HTML and JSON, the results, and a manifest with a checksum for each file and the product-mode label, all zipped. `stress_tested` says whether the design was stress-tested in the session; a certificate made at export has no verdict in the session | free |
   | `cost_ledger()` | session spend by kind and by model, the caps, and the tool calls left | free |

4. **Answer.** The loop ends when the model answers in plain text. Lightning then
   summarises the outcome. The session file records the verdicts computed by code next
   to the model's answer.

**Physics decides pass/fail.** A design's verdict is `pass` when its simulated success at
its best grasp height is at least 80%. A certificate's verdict is `pass` when the nominal
grasp holds in every certified pose and the certificate score is at least 75%. Both rules
live in `rtt.agent.tools`. The agent can only quote these verdicts. It cannot set them.
Lightning's summaries are checked in code too: a step with no verdict cannot be summarised
as a pass or a fail, and a step with no `pass` verdict cannot be summarised as a pass
(`rtt.agent.summarizer`). The agent's own final answer is model text; the session file
records the code verdicts next to it.

## Hard caps (enforced in code)

| Cap | Value | Where |
|---|---|---|
| Tool calls per session | ≤ 12. Invalid and refused calls count, except the refusal of a call past the budget. Once the budget is spent, the next turn is sent with `tool_choice="none"`, so the model must answer. | `AgentCaps`, `Toolbox.call` |
| Jobs in flight | ≤ 2 per session | `Toolbox._launch_sim_job` |
| Model turns | ≤ 16 per session | `GraspEngineer` |
| Dollars per session | default $0.50 (`RTT_AGENT_SESSION_USD`). Covers Token Factory calls plus estimated compute. Caps, prices, reservations and charges must be finite: a NaN would compare false with every limit and switch the caps off, so it is refused. | `SpendGuard` |
| Dollars per UTC day | default $2.00 (`RTT_AGENT_DAY_USD`). Persisted in `runs/agent/daily_spend.json` (`RTT_AGENT_DAY_BOOK` to move it) and shared by every session, whatever `--root` or `RTT_AGENT_ROOT` says. | `DailySpendBook` |
| Passcode | Anything that spends needs it. Live sessions need it at start, because every agent turn costs money. It is checked against `RTT_AGENT_PASSCODE_SHA256`, and with no digest set nothing can spend. After 5 wrong tries in 10 minutes, the passcode is locked. It is checked before the part is looked up, so a request without it cannot make the server reload a part. | `PasscodeGate` |
| Passcode off this machine | When `python -m rtt.api` listens on anything but a loopback address, uploads (`X-RTT-Passcode` header) and offline sessions need the passcode too: part intake is "local or passcode only". `RTT_AGENT_REQUIRE_PASSCODE=1` forces this. | `ApiSettings.require_passcode` |
| Tool arguments | Must match the JSON schema: no extra fields, fixed ID shapes, the session's own part only, bounded counts | `rtt.agent.schemas` |
| Sessions per server | 1 at a time (default). The check and the slot claim are one atomic step, so concurrent requests cannot both start. | `AgentService` |
| Upload | STL only (binary STL with the size matching its triangle count, or ASCII STL), 20 MB maximum, sent with a binary content type (`application/octet-stream` or `model/stl`; anything else gets 415). A flat or almost flat mesh (convex hull `2 x volume / area` under 0.1 mm) gets 422 before intake: the stable-pose search would loop forever on it. A failed intake leaves no files behind. | `rtt.api` |
| Host header | Only loopback names (`RTT_AGENT_ALLOWED_HOSTS` to change; `*` for any), so a DNS-rebinding page cannot reach the API | `rtt.api` |
| Paid targets offline | Offline sessions need no passcode, so they are never offered `nebius` | `AgentService.runners` |

Spend works like this. Every model call reserves its worst case against the session cap
and the day cap before anything is sent: the prompt estimate plus `max_tokens`, at the
output price. When the call finishes, the actual cost is charged. A Nebius job works the
same way, at its true worst case. Its timeout must be whole hours (at least `1h`), so it
reserves `usd_per_hour × (timeout + 15 min of provisioning)` before it is submitted
(`max_wall_s`, 4,500 s by default). A job still running after that long is cancelled, and
when a job ends it is charged the wall-time estimate from creation to end, so the charge
cannot outgrow the reservation by more than one poll interval. Jobs still in flight when a
session ends are cancelled. On the default CPU preset (`cpu-d3`, `4vcpu-16gb`, $0.132/h)
one job reserves about $0.165, so two fit the default $0.50 session cap.

Error messages from libraries often name local files. Tool errors, job errors, the
session's stop reason and upload errors are cut to the file name (`.../source.stl`)
before they are shown to the model, recorded or returned (`rtt.agent.recorder.redact_paths`).

## Recording and playback

Each session is one JSON file in `runs/agent/sessions/<session_id>.json`
(`format: rtt-agent-session`). The file is rewritten atomically after every event. It
holds:

- the session facts: part, goal, models, caps and the label;
- the ordered `events`;
- the full chat transcript (`messages`): system prompt, model turns with their tool calls,
  and tool results;
- the code-computed `verdicts`;
- the spend.

The event types are:

- `session.started`
- `llm.call`: role, model, tokens, cost, latency, whether it was cached
- `agent.message`: text, tool calls requested, and up to 2,000 characters of reasoning
- `tool.call`
- `tool.result`
- `tool.refused`: names the cap that refused the call
- `job.update`
- `summary`: from Lightning
- `session.finished`

No local path or credential is recorded. Nebius job IDs are recorded on purpose, as
public provenance. The static site and the video play sessions back from these files.

Offline sessions (`--offline`) use a scripted policy (`rtt.agent.fakes.ScriptedAgentTransport`)
instead of Nemotron. They make no model calls and spend nothing, and they are recorded with
`scripted-offline` as every model name and $0 for every call and for the session, so they
can never pass for Nemotron output or for real spend.

## API

Start the server with `python -m rtt.api`. It binds to 127.0.0.1:8000. CORS allows the
local web dev servers on ports 3000 and 4173.

| Endpoint | |
|---|---|
| `POST /parts?name=&units=mm\|m&density_kg_m3=` | Raw STL body (`application/octet-stream`). Returns 201 with the part. Returns 413 if too large and 422 if the file is not an STL or cannot be simulated. On a server that listens beyond loopback it also needs `X-RTT-Passcode` (403 without it). |
| `GET /parts/{part_id}` | The part as the agent sees it. |
| `POST /sessions` | Body `{"part_id", "passcode", "goal"}`. The passcode can also go in the `X-RTT-Passcode` header. Returns 201 `{session_id, events}`. Returns 403 for a wrong or missing passcode, 429 when locked out or busy, and 503 when the Nebius target is enabled but cannot be set up (no session file is written then). |
| `GET /sessions/{id}/events` | SSE: `id` is the event `seq` and `event` is its type. Resumes with `Last-Event-ID` or `?after=`, and ends with `event: end`. |
| `GET /sessions/{id}` | The session file, live or finished. |
| `POST /sessions/{id}/stop` | Stops the session after the current step. |
| `GET /runs/{run_id}` | One job's results: every design and every episode. |
| `GET /printpack/{pack_id}` | The print pack as a zip. |

The upload takes a raw body rather than multipart, because `python-multipart` is not a
dependency.

From the terminal:

```bash
export RTT_AGENT_PASSCODE_SHA256=$(python -c "import hashlib;print(hashlib.sha256(b'<passcode>').hexdigest())")
RTT_AGENT_PASSCODE=<passcode> python -m rtt.agent run --stl assets/parts/spur_gear.stl --session-usd 0.25
python -m rtt.agent run --stl assets/parts/l_bracket.stl --offline   # no model calls, real physics
python -m rtt.api --offline --fake-physics                          # UI work: no models, no MuJoCo
```

## Nebius target

`launch_sim_job(target="nebius")` is offered only when the server runs with `--nebius`
(or `RTT_AGENT_NEBIUS=1`), and never to an offline session. The flow is:

1. `NebiusJobRunner.from_config` reads the account settings from `~/.config/rtt/` and
   takes the settings of the job's region (`NebiusConfig.for_region`).
2. The part STL, its intake options and the designs are uploaded to
   `runs/agent-<id>/inputs/` in the results bucket.
3. A job is submitted that runs `python -m rtt.agent.remote --run-dir /out/runs/agent-<id>`
   in the pinned image. The job needs no model key, so it gets no `--env-secret`.
4. The job writes `results.json` and `manifest.json`, which the runner reads back.

Jobs go to the CPU target by default: `cpu-d3` in uk-south2, the executor of record, with
the `4vcpu-16gb` preset (`AGENT_CPU_VCPUS`), using that region's project, subnet, bucket and
image. `target=` picks another target and `platform=`/`preset=` pin one platform; a GPU
platform is used only when asked for. A platform with no listed price needs
`usd_per_hour`, because the spend cap depends on it. In this track, Nebius launches are
tested only against a scripted CLI. The image must be rebuilt to include `rtt.agent.remote`
before the first live agent job.

## Files

- `src/rtt/agent/`:
  - `engineer.py`: the loop.
  - `tools.py`: the tools and caps.
  - `schemas.py`: the tool schemas and argument validation.
  - `caps.py`: the session caps, spend tracking and passcode check.
  - `backends.py`: part intake and physics, through the `rtt.part` adapter or `rtt.sim` directly.
  - `jobs.py`: the local and Nebius job runners.
  - `remote.py`: the job entry point.
  - `proposer.py`: Super.
  - `summarizer.py`: Lightning.
  - `recorder.py`: the session file.
  - `store.py`: where outputs are written.
  - `fakes.py`: the offline stand-ins.
  - `__main__.py`: the CLI.
- `src/rtt/api/`:
  - `app.py`: the FastAPI app.
  - `service.py`: parts and sessions.
  - `__main__.py`: the server.
- Tests:
  - `tests/test_agent_caps.py`
  - `tests/test_agent_tools.py`
  - `tests/test_agent_jobs.py`
  - `tests/test_agent_engineer.py`
  - `tests/test_api.py`
