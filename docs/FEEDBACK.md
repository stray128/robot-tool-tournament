# Platform feedback

A running log of what worked, what surprised us and what we would change while building
Robot Tool Tournament on Nebius. Entries are dated and stick to things we measured or hit
ourselves; numbers come from the files named in each entry. We add to it every day.

## Token Factory

### 2026-10-15: first contact and the output-mode probe

Source: `docs/PROBE_LLM.md` (rebuild offline with `rtt probe-llm --replay`), its ledger in
`runs/probe_llm/`, and `scripts/tf_smoke.py`. The probe made 28 live calls (24 design requests
plus 4 tiny calls to read rate-limit headers) for $0.0391 in total.

What worked:

- The OpenAI SDK worked unchanged against `https://api.tokenfactory.nebius.com/v1/`.
  `GET /v1/models` listed 25 models, including the 4 NVIDIA Nemotron models, and the first
  chat call (Nano, 64 max tokens) returned in 0.85 s for an estimated $0.000015.
- All 28 calls returned HTTP 200. Every NVIDIA model accepted a forced `tool_choice` that
  names a function, a strict `response_format` json_schema, and
  `chat_template_kwargs: {enable_thinking: false}`.
- The rate-limit headers match the published limits exactly (requests per minute / tokens
  per minute): Nano 100 / 800,000, Lightning 600 / 400,000, Super 300 / 200,000,
  Ultra 300 / 200,000. Our client throttles from `x-ratelimit-remaining-tokens`.
- Turning thinking off cut median latency in tool mode to about 2.0 s on Lightning, Super and
  Ultra (from 5.0, 2.9 and 5.8 s), with every answer still valid.

What surprised us:

- **Catalogue feature flags do not match API behaviour.** The verbose model catalogue
  (`GET /v1/models?verbose=true`, read 2026-10-15) lists only `["tools", "reasoning"]` under
  `supported_features` for all four NVIDIA models. `json_mode` and `structured_outputs` appear
  only on non-NVIDIA models. Even so, a strict `json_schema` request was accepted with HTTP 200
  by all four:
  - Nano, Super and Ultra returned schema-valid JSON in 2/2 calls each. We cannot tell whether
    the decoder enforced the schema or the model simply complied.
  - Lightning returned free-form reasoning text in `content` in 2/2 calls. Both calls hit
    `max_tokens` (2,500) before any JSON appeared. A decoder that enforced the schema could not
    have produced that output, so the parameter was silently ignored.
  - *Suggestion:* return a 400 (or at least a warning header) when `response_format` asks for
    a feature the model does not list, and say in the docs which models enforce schemas.
- **Reasoning arrives under different field names.** With thinking on, Nano puts it in
  `message.reasoning`; Lightning, Super and Ultra use `message.reasoning_content`. In the
  Lightning json_schema calls, usage reported all 2,500 completion tokens as reasoning tokens
  while the reasoning text itself came back inside `content`. Nano's usage never reports
  reasoning tokens. *Suggestion:* one field name and consistent usage accounting.
- **Reasoning shares the `max_tokens` budget.** In tool mode Nano spent the whole 2,500-token
  budget thinking in 2/2 calls and returned no answer (`finish_reason: length`).
  *Suggestion:* a separate reasoning budget parameter, or documenting a safe ratio per model.
- **Tool definitions are billed as prompt tokens; `response_format` schemas are not.** The same
  messages cost about 1,330 prompt tokens in tool mode and 600 with json_schema. That is
  reasonable, but worth one sentence in the pricing docs.
- **No `seed` parameter.** Repeat samples cannot be reproduced on the server, so we cache every
  response keyed by the exact request body and a sample index. Replays need no key and cost
  nothing. A seed would let others re-run our probe and get the same answers.
- **Nano was the slowest model by wall clock.** Median latency was 30.2 s (json_schema) and
  43.5 s (tool) with thinking on, and 12.8 s with thinking off (single calls took 5.5 s and
  20.1 s). Lightning, Super and Ultra all came in around 2 s with thinking off. For answers of
  300-420 tokens, the medians work out to roughly 30 output tokens/s for Nano against 150-210
  for the others (wall clock, including prompt processing and network).

### 2026-10-15: the first live design round (`rtt slice`)

Source: `runs/slice/slice-20261015-a/slice.json` (gitignored; the response cache replays it
for free with `rtt slice --replay --cache runs/slice/slice-20261015-a/llm_cache`). Nemotron 3 Super, forced `submit_designs` tool call,
thinking on, temperature 1.0, top_p 0.95, under a $0.20 cap.

- One round cost **$0.0074**: 2 calls (the main proposal and one diversity reflection),
  14,257 prompt and 3,473 completion tokens, of which 2,003 were reasoning tokens.
  Latency was 6.9 s and 11.7 s; both ended with `finish_reason: tool_calls`.
- **The reflection call reasoned 4.7x longer than the main call** (1,650 vs 353 reasoning
  tokens) for a similar prompt. That is fine for quality, but it makes the cost of repair
  paths hard to budget. A per-request reasoning budget would help.
- What the model did with the results: 2 of its 4 first-call designs were near-copies of
  the best entrant (design distance 0.013 and 0.023 against a required 0.08), even though
  the prompt states the rule in practical terms. After one reflection, 3 of 4 slots were
  usable and one fell back to random. One of the accepted designs (a V-groove variant of
  the concave finger) tied the best hand-designed finger on its first try.
- Earlier dev runs of the full search loop (16 calls each) averaged about 6.9K prompt
  tokens per call, with a 13.9 s maximum latency and $0.05 per 40-design run.

## AI Cloud / Serverless Jobs

### 2026-10-15: first jobs, image pipeline and the bucket mount

Sources: `docs/NEBIUS.md`, the spend ledger `runs/cloud/spend.jsonl` (gitignored) and the
fetched run manifests. Every job ran on `gpu-h100-sxm` / `1gpu-16vcpu-200gb` in eu-north1
from one image pinned by digest.

What worked:

- **Zero to first job in minutes.** The CLI install plus browser login took about 2 minutes.
  `nebius ai job create --dry-run` validated a full job spec (image digest, secret, bucket
  volume, subnet) server-side without creating anything; we run it before every real
  submission.
- **Secrets by reference.** `--env-secret NEBIUS_API_KEY=<mysterybox secret>` delivered the
  Token Factory key into the container with nothing on the command line or in the image.
- **The registry credential helper.** `nebius registry docker-credential` works as a Docker
  credential helper from a throwaway `DOCKER_CONFIG`, so pushing needs no stored token. A
  code-only re-push took about 10 s (the 1.5 GB of dependency layers were cached).
- **The H100 image environment just worked:** driver 580.173.02 (CUDA 13.1), Warp found
  `cuda:0`, and EGL rendering used the NVIDIA driver, so MP4s render on the job.
- **Capacity advice is accurate.** `nebius capacity resource-advice list` showed L40S at 0
  while H100 was HIGH, which matched what jobs experienced. We now read it before every
  submission and fall back across platforms.

What surprised us:

- **About 2 minutes of overhead on short jobs.** For the 8 s G1 gate, a job spent about
  54-67 s `PROVISIONING`, 43-47 s `STARTING` and 22-31 s `IMAGE_PULLING` (717 MB
  compressed) before the container ran: 114-127 s from creation to start in three runs.
  *Suggestion:* warm pools or image pre-pulls for repeated jobs from the same digest.
- **CPU-only jobs are impossible in eu-north1 by default.** The default quota for
  `compute.instance.non-gpu.vcpu` there is 0 (uk-south2 has 200). Our physics runs on CPU
  MuJoCo, so it runs on the 16 vCPUs of an H100 preset at $4.50/h with the GPU idle.
  *Suggestion:* a small default CPU quota in every region with Serverless Jobs, or
  CPU-only presets listed beside the GPU ones.
- **The bucket mount is not a POSIX disk, and nothing says so.** On `/out`:
  - an MP4 muxer's seek-back failed with `EINVAL`, leaving a truncated file (the Python
    call did not even raise);
  - write-to-temp-then-rename left a 0-byte target next to the full temp file.
  We now stage every file on local disk and copy it to `/out` front to back.
  *Suggestion:* one paragraph in the jobs docs listing the supported operations.
- **Capacity is not used at submit time.** A job on a platform with no capacity sat in
  `PROVISIONING` for over 10 minutes, with logs saying only "Waiting before retrying
  resource provisioning". *Suggestion:* fail fast or warn when capacity advice for the
  requested platform is 0.
- **An undocumented state.** Jobs pass through `IMAGE_PULLING`, which the jobs docs do not
  list; our poller had to learn it from the CLI's enum.
- **No job ID inside the container.** The container gets no variable with its own job ID,
  so run manifests cannot name the job that produced them; the laptop records it beside
  the fetched run instead. *Suggestion:* inject `NEBIUS_JOB_ID`.
- **Small CLI inconsistencies:**
  - `--dry-run` prints plain text even with `--format json`;
  - `ai job get-by-name` requires `--parent-id` even when the profile sets one;
  - `--volume <bucket name>` fails with `unsupported volume source type`; only the bucket ID
    works, and the message could say so;
  - `ai job cancel` blocked for more than 2 minutes on a job that never started.
- **Budgets are console-only.** There is no CLI command for budgets or alerts, so our
  spend cap is enforced client-side from a ledger of wall-time estimates.

## NVIDIA Nemotron models

### 2026-10-15: proposer output-mode probe (2 calls per model and mode)

Task: propose two gripper-finger designs as JSON over a 10-parameter schema, after reading
three prior results. "Clean" means parsed, complete, within every bound and passing the
cross-field rules. Full table: `docs/PROBE_LLM.md`.

| model | forced tool call | strict json_schema | tool call, thinking off |
|---|---|---|---|
| Nano 30B-A3B | 0/2 (thinking used the whole budget) | 2/2, p50 30.2 s | 2/2, p50 12.8 s |
| Lightning | 2/2, p50 5.0 s | 0/2 (free text, budget exhausted) | 2/2, p50 2.0 s |
| Super 120B-A12B | 2/2, p50 2.9 s | 2/2, p50 5.5 s | 2/2, p50 2.0 s |
| Ultra 550B-A55B | 2/2, p50 5.8 s | 2/2, p50 4.6 s | 2/2, p50 2.0 s |

- No parsed answer broke a numeric bound, an enum or a cross-field rule. Our validators still
  run on every design.
- Super in tool mode was the cheapest clean option for the proposer (about $0.0008 per call,
  at most 469 completion tokens). Ultra was also clean, at 8.4x Super's cost per call.
- Super's reasoning was short in tool mode (103-169 reasoning tokens) and longer under
  json_schema (475-663) for the same prompt.
- Model IDs mix casing and separators (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
  `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`). They are
  case-sensitive, so we copy them from `GET /v1/models` rather than typing them.

Decision so far: Super with a forced tool call and thinking on for the design proposer, and
Super with thinking off for latency-sensitive calls. The pre-registered runs will measure this
at scale.

### 2026-10-15: first full Nemotron search run, inside a Nebius job

One run of the Nemotron arm used a budget of 40 designs: 8 shared round-0 designs, then 4
rounds of 2 islands x 4 slots. It ran as a Serverless job on the calibrated task, with
Super reached through `--env-secret`.

- All 16 calls succeeded (16 is the per-run cap). They cost $0.058 in total: 123,007
  prompt and 23,912 completion tokens, of which 11,331 were reasoning. The maximum latency
  was 10.5 s.
- **Diversity is the model's weak spot.**
  - 28 of 33 rejected proposals were near-duplicates of designs already tried (design
    distance below 0.08). Only 5 broke a geometry rule.
  - After repairs and reflections, 14 of 32 model slots still had to be filled at random.
  - In this run the best design came from the shared random round 0 (94 %), and the model
    never beat it.
- *Suggestion for us, not the platform:* say the diversity rule in the tool schema itself
  (per-parameter minimum moves), or loosen it. This has to be decided before
  pre-registration; it is not a Token Factory issue.

## Onboarding: zero to hello world

### 2026-10-15

- Hello world was a key in `.env`, the stock OpenAI SDK with the Token Factory base URL,
  `GET /v1/models`, then one chat call (`scripts/tf_smoke.py`). It worked with a trial key.
- **Gotcha:** the OpenAI SDK reads `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment
  and sends them as headers to whatever base URL it is given. A developer who also uses OpenAI
  would leak those identifiers to Token Factory. Our transport clears them. *Suggestion:* add a
  line to the quick-start.
- Response bodies and rate-limit headers follow the OpenAI shapes (`x-ratelimit-limit-*`,
  `x-ratelimit-remaining-*`, and `x-ratelimit-reset-*` as durations such as `1s`), so existing
  retry and throttling code carried over.

## Would we build again?

After day 1: yes for Token Factory, and yes for AI Cloud with caveats. An OpenAI-compatible
endpoint, four NVIDIA models behind one key, and limits that match the docs got us from zero
to a 28-call probe for under four cents. The mismatch between feature flags and `response_format` behaviour is the one
thing that would have bitten us silently without a probe. AI Cloud ran every job we
submitted once it had capacity, and secrets by reference, the registry helper and
`--dry-run` were excellent. Two things cost us real time: the undocumented
bucket-mount semantics, and a default CPU quota of 0 that puts CPU physics on an idle
H100.
