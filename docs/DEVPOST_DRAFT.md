# Devpost submission draft

Draft text for the Nebius x NVIDIA Global AI Hackathon form. Every number comes from
`docs/RESULTS.md`, `docs/EFFECTORS_RESULTS.md`, `docs/NEBIUS.md` or `docs/FEEDBACK.md`.
Paste each block into its field; the founder submits.

## Project name

Effector Lab

## Tagline

Bring your part: NVIDIA Nemotron designs gripper fingers and other end-effectors for it, and Nebius Serverless Jobs test every design in physics simulation before you print.

## Track

Physical AI

## Links

| What | URL |
|---|---|
| Code repository (Apache-2.0) | https://github.com/stray128/robot-tool-tournament |
| Live site (static, no login) | https://stray128.github.io/robot-tool-tournament/ |
| Pre-registered protocol (git tag) | https://github.com/stray128/robot-tool-tournament/tree/prereg-v1 |
| Results | https://github.com/stray128/robot-tool-tournament/blob/main/docs/RESULTS.md |
| Platform feedback | https://github.com/stray128/robot-tool-tournament/blob/main/docs/FEEDBACK.md |
| Demo video (YouTube, under 3 minutes) | to be added by the founder |

New during the submission period: the repository's first commit is dated 2026-10-15.

## Inspiration

When a robot cell gets a new part, its gripper usually needs new fingers. Integrators find
them by drawing, printing and testing until one works. The common shortcut is an imprint:
the part's outline cut into the jaw. It fits when the part sits exactly where it should,
and it can jam when the part arrives a few millimetres off.

We wanted to move that loop into simulation, and to find out honestly whether an NVIDIA
Nemotron model that reads simulator results designs better fingers than random search. So
we wrote the test down and tagged it before running it.

## What it does

- **Bring Your Part.** Upload an STL. The intake finds the part's units, mass, convex pieces
  and resting poses, and builds the test conditions.
- **Nemotron designs.** Nemotron 3 Super on Nebius Token Factory proposes designs through a
  forced tool call, reads the measured results of its earlier designs, and proposes the
  next round. A design is a part-shaped recipe that compiles into its own pair of jaws.
- **Physics decides.** Validators check geometry and printability first. MuJoCo then scores
  every design with the part placed off centre. A grasp counts only if the part is seated
  and centred before the lift, and still held 2 s later.
- **Effector tournament.** For one part, parallel-jaw fingers, open-source hands and
  grippers, a soft arm, a suction cup and Nemotron-designed effectors compete on the same
  placements, seeds and scorer. One effector is recommended with its certificate.
- **Certificate and print pack.** 18 robustness tests (placement error up to 3 mm, shake,
  push), then finger and jaw STLs, the certificate and a manifest in one zip.
- **Grasp Engineer.** A Nemotron 3 Super agent runs these steps for an uploaded part within
  hard caps on tool calls, jobs and dollars. Nemotron 3.5 Lightning writes the step
  summaries.

The pre-registered result: with 40 simulated designs per run and 32 runs per arm, Nemotron
with feedback reached the target in 23 of 32 runs against 11 of 32 for random search. It
needed 8.97 fewer designs on average (95% interval 3.53 to 14.16; one-sided exact
permutation p = 0.0011). On held-out parts its frozen designs registered 72.5% of grasps
against 60.5% (procedural parts) and 27.5% against 22.0% (64 real scanned objects), both
significant after Holm adjustment. Everything is simulated; nothing has been printed yet.

## How we built it

- **Physics:** MuJoCo 3.15 on CPU is the executor of record. Gates checked geometry,
  physics, calibration and machine parity before any scored run.
- **Nemotron on Token Factory:** the OpenAI SDK against the Token Factory endpoint. We
  probed Nano, Super, Ultra and Lightning, then chose Super with a forced tool call. The
  proposer asks for reserve designs, repairs rejected ones once, and screens near-copies.
  Every request and response is cached by SHA-256, so a run replays with zero API calls,
  and a spend ledger caps every call.
- **Nebius AI Cloud:** every scored batch is a Serverless AI Job on uk-south2 `cpu-d3`,
  from one image pinned by digest in Container Registry. Outputs go to an Object Storage
  bucket mounted at `/out`. The Token Factory key reaches the job from MysteryBox. Our
  `rtt job` command checks capacity, submits, waits, records the spend, then fetches and
  verifies the outputs.
- **The experiment:** a pre-registration frozen at the git tag `prereg-v1`, a commit-reveal
  held-out suite whose seed hash was committed first, and a deviations log for every change
  after the tag.
- **Product:** a FastAPI service for the Grasp Engineer agent, and a static Next.js site
  with React Three Fiber that replays every recorded bout from the simulator's pose logs.

## Challenges we ran into

- **Near-copy proposals.** Super often proposed designs almost identical to ones already
  tested, and it could not audit itself against a long history. Asking for 4 reserve
  designs per call fixed most of it.
- **Determinism.** Our laptop differed from the cloud in 1 of 1,920 success flags for the
  same seed, so we pinned the executor of record to the Nebius CPU platform. Results there
  were byte-identical across hosts. The GPU physics cross-check failed our determinism gate,
  so it never scores.
- **Platform surprises:** a bucket mount that is not a POSIX disk, a cross-region image
  pull that failed after the dry run passed, a default CPU quota of 0 in one region, and a
  CLI that hangs after its session expires.
- **Honest labels.** Our hero screen found no part where the stock finger visibly fails, so
  the video uses the imprint against a designed finger instead. Product-mode results are
  kept apart from the pre-registered ones and labelled on every page.

## Accomplishments that we're proud of

- A pre-registered win for Nemotron with feedback (p = 0.0011), confirmed on held-out parts
  that no search process could read.
- The whole project's compute on Nebius AI Cloud cost about $4.12. The Nemotron arm's 473
  Token Factory calls cost $3.57, with no errors.
- The confirmatory batch simulated 132,320 grasps from one pinned image, and every LLM run
  replays from its cache for free.
- The effector tournament ran 3,006 episodes across every family with 0 simulator resets.

## What we learned

- The metric decides the answer. With lift-only scoring the arms do not differ (p = 1.0);
  the gain is in seating and centring the part.
- Nemotron is a good proposer and a weak self-auditor. Checks in code (validators,
  diversity screens, verdicts computed by physics) are what make it safe to use.
- Real objects are hard: the better arm registers about a quarter of scanned objects.
- Jobs, not endpoints, suit batch simulation: a fixed image runs, writes its results and
  stops.

## What's next for Effector Lab

- Fine-tune an open NVIDIA Nemotron model with reinforcement learning on simulator rewards,
  on Nebius GPUs, served from Nebius Serverless Endpoints, so it designs end-effectors for
  any task: fingers first, then general attachments, then other task types.
- Print the top designs and test them on a real gripper with a partner, against the
  simulated certificate.

## Built With

nebius-token-factory, nebius-ai-cloud, nebius-serverless-jobs, nebius-object-storage,
nebius-container-registry, nebius-mysterybox, nvidia-nemotron, mujoco, nvidia-warp,
mujoco-warp, python, numpy, trimesh, coacd, fastapi, next.js, react, react-three-fiber,
three.js, tailwindcss, typescript, docker, github-pages

## Testing instructions

- Open the live site: https://stray128.github.io/robot-tool-tournament/ (no login). Pages
  show recorded runs; the one illustrative agent session is labelled as such.
- Or run it locally: clone the repository, `cd web && pnpm install`, then `make demo` from
  the repository root, and open http://127.0.0.1:4173/.
- Live Nemotron calls need a Token Factory key in `.env` (see `.env.example`). Nebius jobs
  need the one-time setup in `docs/NEBIUS.md`.

## Feedback (Token Factory, AI Cloud, NVIDIA models)

Copied from `docs/FEEDBACK.md`, section "Feedback for the Devpost field". That file is the
source; re-copy it if it changes.

**What we built and how much we used.** Effector Lab designs gripper fingers and other
end-effectors for an uploaded part. NVIDIA Nemotron 3 Super proposes designs through Nebius
Token Factory, and Nebius Serverless AI Jobs score them in MuJoCo physics on CPU. Our
pre-registered experiment ran 32 search runs per arm, one job per arm. The Nemotron arm made
473 Token Factory calls from inside a Serverless Job: 6.98M prompt and 1.64M completion
tokens, 0 errors, no retries, $3.57. The whole project used 17 Serverless Jobs for about
$4.12 (wall-time estimate), and the confirmatory batch simulated 132,320 grasps.

#### Token Factory

What worked:

- The stock OpenAI SDK worked unchanged against `https://api.tokenfactory.nebius.com/v1/`.
  The first chat call returned in 0.85 s.
- Rate-limit headers matched the published limits exactly, in OpenAI's header format, so
  our existing throttling code carried over.
- All four NVIDIA models accepted a forced `tool_choice` naming a function. Given enough
  token budget, it was the most reliable output mode we tested.
- It was dependable at scale: 32 search runs called it in lockstep from one job, 473 calls,
  not one error or retry. p50 latency 15.1 s, max 65.8 s, with 16,384 max tokens.

What needs work:

- **Catalogue feature flags do not match API behaviour.** `GET /v1/models?verbose=true` lists
  only `tools` and `reasoning` for the NVIDIA models. Even so, all four accepted a strict
  `response_format` json_schema with HTTP 200. Nano, Super and Ultra returned valid JSON.
  Nemotron 3.5 Lightning ignored the schema: it wrote free text in `content` until it hit
  `max_tokens` (2 of 2 calls). Please return a 400 or a warning header when a request asks
  for a feature the model does not support, and document which models enforce schemas.
- **Reasoning output differs by model.** Nano puts it in `message.reasoning`, the others in
  `message.reasoning_content`. Nano's usage never reports reasoning tokens. In Lightning's
  json_schema calls, usage counted all 2,500 tokens as reasoning while the text came back in
  `content`. One field name and one accounting rule would help.
- **Reasoning shares the `max_tokens` budget.** Nano spent its whole 2,500-token budget
  thinking in tool mode and returned no answer. A separate reasoning budget, or a documented
  safe ratio per model, would prevent this. Super's reasoning length also varied: one
  reflection call reasoned 4.7x longer than the main call on a similar prompt.
- **No `seed` parameter.** We cache every response so our runs replay exactly, but nobody
  else can re-draw our model answers. A seed would make LLM experiments reproducible.
- **Small things.** Tool definitions are billed as prompt tokens (about 1,330 tokens against
  600 for the same messages with json_schema); that deserves a line in the pricing docs.
  Model IDs mix casing and separators (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
  `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`) and are
  case-sensitive.

#### AI Cloud: Serverless Jobs, Object Storage, Container Registry, MysteryBox, CLI

What worked:

- **Serverless Jobs fit batch simulation.** One pinned image runs, writes its results and
  stops. CPU jobs on uk-south2 `cpu-d3` cost 8-10x less than the H100 preset for the same
  CPU physics ($0.019 against $0.182 for one search), with byte-identical outputs across
  Nebius hosts. A 384-episode parity check matched our laptop in 384 of 384 outcomes.
- **`nebius ai job create --dry-run`** validated the full job spec on the server. We ran it
  before every submission.
- **MysteryBox secrets by reference** (`--env-secret`) delivered the Token Factory key with
  nothing on the command line or in the image.
- **The registry credential helper** (`nebius registry docker-credential`) worked from a
  throwaway `DOCKER_CONFIG`, so no token touched disk. Code-only re-pushes took about 10 s.

What needs work:

- **Capacity is invisible at submit time.** A job on a platform with no capacity sat in
  `PROVISIONING` for over 10 minutes. The logs said only "Waiting before retrying resource
  provisioning", while `capacity resource-advice list` already showed L40S at 0. Please fail
  fast or warn at submit. Capacity advice also has no CPU rows.
- **The default CPU quota in eu-north1 is 0** (`compute.instance.non-gpu.vcpu`). CPU-only
  jobs there are impossible without a request, so our first jobs ran CPU physics on an idle
  H100 at $4.50/h. uk-south2 has 200 by default.
- **The dry run misses two late failures.** A uk-south2 job could not pull from the
  eu-north1 registry: the dry run passed, then the job spent about 64 s pulling and ended
  `StartFailed`. A new uk-south2 bucket defaulted to a storage class with 0 quota, so the
  first upload failed with `QuotaLimitExceeded`. The dry run could check pull access, and
  the errors could name the registry's region and the storage class.
- **The bucket mount is not a POSIX disk, and the docs do not say so.** An MP4 muxer's seek
  failed with `EINVAL` and left a truncated file. Write-then-rename left a 0-byte target. We
  now stage files locally and copy them front to back. One paragraph listing the supported
  operations would save others a day.
- **CLI details.** `--volume <bucket name>` fails with `unsupported volume source type`;
  only the bucket ID works, and the message could say so. `ai job cancel` blocked for more
  than 2 minutes on a job that never started. `--dry-run` prints plain text even with
  `--format json`. `ai job get-by-name` requires `--parent-id` even when the profile sets
  one. The `IMAGE_PULLING` state is missing from the jobs docs.
- **The CLI session expires after about 12 hours, and unattended commands then hang.** The
  credential helper printed `exit status 7`, and `docker push` waited out a 10-minute
  timeout instead of failing. `nebius iam whoami` waited for a browser login. Please fail
  fast when no terminal is attached, and document service-account profiles next to the
  quick start.
- **Budgets are console-only.** There is no CLI or API for budgets or alerts, so we enforce
  our own spend cap from a ledger of wall-time estimates.
- **About 2 minutes of overhead per job** (provisioning, starting, image pull) before the
  container runs. Warm pools or pre-pulled images for a repeated digest would help short
  jobs. There is also no job ID inside the container; a `NEBIUS_JOB_ID` variable would let
  outputs name the job that made them.
- **CPU prices are split across two pages** (per vCPU-hour and per GiB-hour). A per-preset
  table like the GPU one would help.

#### NVIDIA Nemotron models

- **Nemotron 3 Super** was the right default. In our probe it was clean in every output
  mode, and a forced tool call cost about $0.0008. On the real design prompt (12.6k
  tokens), 16 of 20 primary designs passed the full geometry validators. The pre-registered
  result says its feedback loop works on this task: it reached the target in 25.6 designs
  on average, against 34.5 for equal-budget random search (p = 0.0011).
- **Super's weak spot is diversity.** It cannot reliably audit its own proposals against a
  long history: asked for the nearest tested design, it named the true nearest in 56 of 194
  slots, and many proposals were near-copies. Asking for 4 reserve designs per call fixed
  most of this. In the confirmatory arm, 131 of 1,024 model slots (13%) still fell back to
  random designs.
- **Nemotron 3 Ultra** was clean in every mode at 8.4x Super's cost per call. We did not
  need it for format reliability.
- **Nemotron 3 Nano** was the slowest by wall clock (p50 30-44 s with thinking on), and in
  tool mode it ran out of budget while thinking. With thinking off it was clean at 12.8 s.
- **Nemotron 3.5 Lightning** was fast in tool mode (p50 2.0 s with thinking off) and cheap
  for step summaries (about $0.00005 each). It ignored strict json_schema (above). Once it
  wrote that a design passed when the step had no verdict, so we now drop such sentences in
  code. Guidance for small summarisers on "no verdict" inputs would help.
- Turning thinking off (`chat_template_kwargs: {enable_thinking: false}`) cut median tool
  latency to about 2 s on Lightning, Super and Ultra, and every answer stayed valid.

#### Onboarding: zero to hello world

- Token Factory: a key in `.env`, the OpenAI SDK with the base URL, `GET /v1/models`, one
  chat call. It worked with a trial key within minutes.
- AI Cloud: the CLI install and browser login took about 2 minutes, and zero to the first
  job took minutes.
- Gotcha: the OpenAI SDK sends `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment
  as headers to any base URL. A developer who also uses OpenAI would send those IDs to
  Token Factory. One line in the quick start would warn them.

#### Would we build on it again?

Yes. Token Factory gave us four NVIDIA models behind one OpenAI-compatible key, with limits
that match the docs, and it served a 473-call batch from inside a job without an error.
Serverless Jobs suit a pre-registered experiment: every scored number comes from one
pinned image, and the whole project's compute cost about $4. What cost us time were the
silent failures: capacity waits, image-pull and quota errors that the dry run let through,
the non-POSIX bucket mount, the CLI hang after the session expired, and schema options that
a model accepts and then ignores. Fix those first and the platform is excellent for this
kind of work.
