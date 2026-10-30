# Hosted runner

The site at https://stray128.github.io/robot-tool-tournament/ replays recorded runs. The hosted
runner lets a visitor plan and run a new job live, under hard caps:

1. The visitor describes a robot job. `POST /plan` sends it to the Task Engineer
   (`rtt.studio.engineer`, prompt v2.3, rule set v2-r4), which asks NVIDIA Nemotron 3 Super for a
   plan, repairs it once if the checks find problems, and returns a validated `TaskSpec`.
2. The visitor starts a run. `POST /runs` checks the plan again on the server and sends it to one
   Nebius Serverless Job, which runs the Studio search (`rtt.studio.search`) with the product's
   default settings on 32 vCPUs.
3. The site polls `GET /runs/{id}` and shows the pick, the baseline and the per-step table when
   the run is done.

Product mode: not part of the pre-registered experiment.

## How it is built

| Piece | What it does |
|---|---|
| AWS Lambda (container image) behind a Function URL | The API (`rtt.runner.app`). A Function URL, not API Gateway: a plan takes 17 to 40 s, past API Gateway's 29 s limit. HTTPS only; CORS for the site and its dev server only. |
| DynamoDB (on demand) | Daily counters, per-IP limits, run records and the spend ledger. Every item has a TTL. |
| EventBridge schedule, every 2 minutes | Polls runs in flight, cancels jobs past the wall-time limit and settles their cost. |
| SSM parameter | The kill switch. |
| Reserved concurrency (10) | At most 10 requests run at once, whatever arrives. |
| AWS Budgets | A monthly cost budget ($20 by default) that mails an alert address. |
| Nebius Serverless Jobs (uk-south2, `cpu-d3`, 32 vCPUs) | The physics. One job per run, from the simulation image pinned by digest. Nothing simulates on AWS. |
| Nebius Object Storage | The job writes `report.json`, `summary.json` and `manifest.json` under `runs/<run_id>/`. |

Code: `src/rtt/runner/` (API, caps, store, Engineer adapter, Nebius client, job entry point) and
`infra/runner/` (the CDK app in Python, the Lambda Dockerfile and a template checker).

## API

All bodies are JSON. Errors are `{"error": "..."}`.

### `GET /health`

```json
{"ok": true, "kill_switch": false, "auto_off": "2026-12-15", "open": true,
 "plans_left_today": 20, "runs_left_today": 4,
 "spend_today_usd": 0.0, "spend_total_usd": 0.0}
```

`open` is false while the kill switch is on or after the auto-off date: then plans and runs answer
503. `plans_left_today` counts the free tier, site-wide. `runs_left_today` is how many more runs
the compute budget can reserve today. Spend is the Nebius compute spend, including the
reservations of runs still in flight. Both counts read 0 while the runner is closed.

### `POST /plan`

```json
{"description": "Open the drawer and put the cup in it.", "provider": "free", "key": null}
```

`provider` is `nvidia` (NVIDIA's API with your key), `tokenfactory` (Nebius Token Factory with
your key) or `free` (our Token Factory key, capped; send `"key": null`). The description is at
most 4,000 characters (the Engineer reads the first 2,000).

```json
{"status": "valid", "plan": {"...": "TaskSpec"}, "problems": [], "calls": 1,
 "cost_usd": 0.0089, "model": "nvidia/nemotron-3-super-120b-a12b"}
```

`status` is `valid`, `refused` (a valid refusal: the job cannot be simulated, with the nearest
supported action) or `failed` (still wrong after the repair call, or the model could not be
reached; `problems` says why). `cost_usd` is priced at Token Factory list prices.

| Status | When |
|---|---|
| 400 | Bad input: missing or long description, unknown provider, a key that is missing or malformed, unknown fields. |
| 401 | The provider rejected your key, or refused it for billing. The message never contains the key. |
| 429 | A cap was reached: the body names the cap, its limit and when it resets. |
| 503 | The kill switch is on, the runner is past its auto-off date, or the free tier is unavailable. |

### `POST /runs`

```json
{"plan": {"...": "the TaskSpec POST /plan returned"}}
```

The server never trusts the plan it receives. It must parse and pass the Engineer's checks with
no problem, have steps and no refusal, name no uploaded part, stay under 48 KiB, and have at most
6 objects and 8 steps. The job receives the server's own parse of it.

```json
{"run_id": "63xyzebd7okmshvd6vrvkrpckmz2d6hd", "status": "queued", "eta_s": 480}
```

The run id is 160 random bits in lowercase base32. Anyone holding it can read the run. Errors:
400 (the plan failed the checks; `problems` lists why), 429 (a cap), 503 (switched off; the
simulation backend unavailable, checked before anything is reserved; or the job could not be
started, or the platform refused it and reports it absent: in all these cases nothing counts
against your runs).

When the platform's answer to a submission is lost (a timeout after the job was requested), the
run is accepted as `queued` and keeps its reservation. The scheduled sweep looks the job up by
name once the submitting request can no longer be alive (the function timeout plus 2 minutes):
a job it finds is tracked and capped like any other; a job the platform says does not exist is
released at no cost. A job past the wall limit (or with the kill switch on) is cancelled even
while its status cannot be read. Whenever the platform cannot be asked about a run for 75 minutes
after its creation (the lookup or the status call fails, or the backend cannot be built), the run
is settled: a run with a job first has its published result read from Object Storage, and a
complete one finishes the run as `done` at its real cost (job creation to the manifest's finish
time); otherwise the job is cancelled (best effort) and the run is charged its cost from creation
to then; a run without a job id is charged its full reservation. A job whose run record was lost
to a race is cancelled at once. "Does not exist" is read only from the platform's gRPC status,
and only when every status in the error is `NotFound` (`code = NotFound`), never from free text.

### `GET /runs/{id}`

```json
{"status": "done", "created_at": "2026-10-28T09:00:00Z", "cost_estimate_usd": 0.176,
 "error": null, "result": {"...": "summary"}}
```

`status` moves from `queued` to `running` to `done` or `failed`. The result, once done:

| Field | Meaning |
|---|---|
| `pick` | The recommended setup: `label`, `kind` (`single`, `tool_change`, `multi_head`), `tools`. `null` when nothing completed the task; `closest` then names the setup that got furthest. |
| `completion` | `{"k": 8, "n": 8}`: the pick (or the closest setup) completed the task in k of n runs. |
| `why` | Why it was picked, or why there is no recommendation. |
| `baseline` | The recommender's best single tool and its completion. |
| `ruled_out` | Candidates, pruned tools, screened tools, rejected setups, finalists and ineligible finalists. |
| `steps` | One row per step: `step`, `tool`, `success_rate`. |
| `code_version` | The commit the job ran. |
| `report_url` | A presigned link to the full `report.json`, valid for 15 minutes. |

Unknown or malformed ids are 404. `GET /runs/{id}/report` redirects (302) to a fresh
`report_url`.

## Caps

All caps are checked and taken in one DynamoDB transaction before any money is spent. Days are
UTC days. Each value is configurable (`RUNNER_<NAME>` in the function's environment, or
`-c cap.<name>=...` at deploy time).

| Cap | Default |
|---|---|
| Free-tier plans, whole site | 20 per day |
| Free-tier plans, per visitor | 3 per day |
| Plans with your own key, per visitor | 30 per day |
| Plans with your own key, whole site | 300 per day |
| Most one plan may spend on the model | $0.05 |
| Runs, per visitor | 2 per day |
| Nebius compute spend | $2 per day, $10 in total (the total never resets) |
| Job wall time | 20 minutes from job creation; the job is then cancelled, and cancelled again if it still runs 5 minutes later |
| Concurrent requests | 10 (the function's reserved concurrency) |
| Auto-off | The runner serves through 2026-12-15 (UTC) and answers 503 from 2026-12-16 |

How a run's cost is counted: before a job is submitted, the run reserves its worst case, the
wall-time limit plus 5 minutes for a slow cancel, at the `cpu-d3` 32-vCPU list price of $1.056 per
hour (`docs/NEBIUS.md`): $0.44. When the job ends, the reservation is replaced by its estimated
cost: wall time from job creation to finish times the hourly price [assumed: the billed time is
expected to be shorter, so this is an upper bound]. A typical run of 8 to 10 minutes costs about
$0.14 to $0.18.

The search inside the job also stops itself 16 minutes after its container starts (the wall
limit less about 4 minutes for provisioning and publishing), so a slow search ends with its own
error before the runner cancels it.

## Keys and privacy

- **Your key is used for one request.** It arrives only in the JSON body of `POST /plan`, over
  HTTPS. It goes to the provider you chose and nowhere else: it is never logged, stored, cached
  or written to disk, and never echoed in a response or an error. If a provider's error message
  contains your key, it is removed before anything is returned. The tests run every path a key
  can take with every logger at debug level and check the key appears nowhere
  (`tests/test_runner_keys.py`).
- **No request bodies are logged.** Each request logs one JSON line: request id, route template
  (`/runs/{id}`, never the id), status and duration. An internal error logs its exception type,
  never its message.
- **IP addresses are never stored or logged.** Per-visitor limits use an HMAC of the address
  and the day, keyed with a server secret, so a stored hash cannot be reversed by trying
  addresses and two days' hashes of one address do not match. An IPv6 visitor counts by their
  /64 prefix (one subscriber usually holds a whole /64).
- **Runner secrets stay out of the image and the environment.** The function reads its two
  secrets (the free tier's key, and the Nebius service account and storage key) from AWS Secrets
  Manager at run time, keeps them in memory for the container's lifetime, and may read only
  those two.
- **The job sees no secret.** It only reads a plan and writes results.
- **Descriptions and plans** you send are processed by the model provider you choose. Plans you
  run are stored with their results in the results bucket.

## Run it locally

Against fakes (no network, no cost; the fake model always answers with the Engineer's worked
example, and fake jobs finish after 15 s):

```bash
rtt runner serve --fake            # http://127.0.0.1:8787, CORS for http://localhost:4341
curl -s localhost:8787/health
curl -s localhost:8787/plan -H 'content-type: application/json' \
  -d '{"description": "Open a drawer.", "provider": "free", "key": null}'
```

With real keys (this calls Token Factory and starts real Nebius jobs, under the same caps):

```bash
# NEBIUS_API_KEY in .env (the free tier); Nebius settings for uk-south2 in ~/.config/rtt/nebius.env
rtt runner serve --env-file .env
```

State is in memory unless `--table` names a DynamoDB table. The job itself can run on a laptop
against a folder that stands for the bucket:

```bash
python -m rtt.runner.job --run-dir /tmp/bucket/runs/<run_id> --plan-sha256 <sha256 of input/plan.json>
```

## Deploy

`infra/runner/` holds a CDK app in Python (the rest of the repository is Python, and a CDK app in
the same language needs no build step). The account, the region, the Nebius project, subnet and
bucket, and the job image's digest are passed as context at deploy time; none is committed.

```bash
cd infra/runner
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
npx aws-cdk@2.1145.0 synth -c alertEmail=<address> \
  -c nebiusProjectId=... -c nebiusSubnetId=... -c nebiusBucketId=... \
  -c nebiusBucketName=... -c jobImage=cr.uk-south2.nebius.cloud/<registry>/rtt-sim@sha256:...
python3 check_template.py cdk.out/EffectorLabRunner.template.json
```

`alertEmail` is required: synth fails without it. The stack creates a monthly AWS cost budget
(`-c budgetUsd=20` by default) on the resources tagged `project=effector-lab`, and mails that
address at 80 % of actual and 100 % of forecast spend. The tag scope needs `project` activated as
a cost allocation tag in the Billing console (`-c budgetScope=account` budgets the whole account
instead).

The function reserves 10 concurrent executions (`-c reservedConcurrency=<n>`, or `none`). AWS
lets a function reserve only what leaves at least 100 unreserved executions in the account, so
the account's Lambda concurrency limit must be at least 110; new accounts with a lower limit need
a quota increase or `-c reservedConcurrency=none` (and then the template check fails on purpose).

`-c jobPreset=4vcpu-16gb` runs jobs on a smaller `cpu-d3` preset (the caps price it the same
way), which makes a cheap smoke test; the default is `32vcpu-128gb`.

`check_template.py` fails if the template breaks a rule: CORS wider than the two origins, a
public invoke permission not limited to the Function URL, more than the two secrets, any wildcard
action or resource, a table without TTL or point-in-time recovery, a log group without retention,
an environment variable that looks like a secret, a role with a managed policy or log access beyond
its own log group, no reserved concurrency, or no cost budget.

The job image is the simulation image (`docker/Dockerfile.sim`) built from a commit that contains
`rtt.runner.job` (`jobs/nebius/build_image.sh --ref HEAD`, then `jobs/nebius/push_image.sh`).

The kill switch is the SSM parameter `/effector-lab/runner/kill-switch`: `on` stops new plans and
runs within 30 s, and the next sweep (up to 2 minutes later) cancels the jobs in flight. If the
parameter cannot be read, the runner treats the switch as on. The stack owns the parameter, so a
stack update can set it back to `off`: after any deploy, check it and set it again if needed.

### Before going live

- **A Nebius budget alert on the uk-south2 project is mandatory.** The runner's caps bound its own
  spend, but the Nebius service account it uses holds the `editor` role on that project (the
  narrowest built-in role that can create jobs, pull images and use the bucket), so the platform's
  budget alert is the backstop for anything outside the runner. Nebius budgets are set in the
  console.
- The AWS budget above covers the AWS side.

Nebius jobs keep `--timeout 1h`: the CLI accepts whole hours only (1h to 168h), so the runner's
20-minute cancel and the job's own deadline do the bounding.

## Limits and known gaps

- The NVIDIA endpoint is assumed to accept the Engineer's forced tool call with its strict
  schema, as Token Factory does. If it does not, a plan comes back `failed` with the provider's
  message in `problems`.
- The hosted runner has no part upload: plans that name an uploaded part are refused.
- `report_url` points at Nebius Object Storage. Opening it in a browser works as a link; fetching
  it from page JavaScript needs a CORS rule on the bucket.
