# Platform feedback: Nebius Token Factory, Nebius AI Cloud, NVIDIA Nemotron

The section "Feedback for the Devpost field" is the text we paste into the submission. The
dated log under it is the evidence: every number there names the file it came from.

## Feedback for the Devpost field

**What we built and how much we used.** Effector Lab turns a multi-step robot job, described in
plain words, into a tested setup: one tool, a two-tool head or a tool change, with replays and a
build pack. NVIDIA Nemotron 3 Super on Nebius Token Factory plans the job as objects, steps and
pass criteria (a forced tool call with the task contract's schema), designs fingers and other
effectors, and runs the agent's tool loop; Nemotron 3.5 Lightning writes step summaries. Nebius
Serverless Jobs run the scored MuJoCo physics on CPU. Usage: the finger experiment's Nemotron arm
made 473 Token Factory calls from inside one Serverless Job (6.98M prompt and 1.64M completion
tokens, 0 errors, no retries, $3.57); the first Studio's Task Engineer made 17 calls for its
recorded gate ($0.133); re-planning the six jobs of this version took 61 calls ($0.56), two
draft recordings included. The project ran 18 Serverless Jobs for about $4.18 (wall-time estimate) before this
version: 17 through the finger experiment, which simulated 132,320 grasps in its confirmatory
batch, and one 206 s job for the Studio's pre-registered use-case benchmark. This version's job
searches ran on a local machine, not on Nebius.

### Token Factory

What worked:

- The stock OpenAI SDK worked unchanged against `https://api.tokenfactory.nebius.com/v1/`.
  The first chat call returned in 0.85 s.
- Rate-limit headers matched the published limits exactly, in OpenAI's header format, so
  our existing throttling code carried over.
- All four NVIDIA models accepted a forced `tool_choice` naming a function. Given enough
  token budget, it was the most reliable output mode we tested.
- It was dependable at scale: 32 search runs called it in lockstep from one job, 473 calls,
  not one error or retry. p50 latency 15.1 s, max 65.8 s, with 16,384 max tokens.
- **A forced tool call with a 33 KB schema just worked.** The Studio's Task Engineer sends the
  whole task contract as one `submit_task_plan` tool (every object and step a closed `anyOf`
  branch, about 21k prompt tokens per call). In the recorded gate all 17 calls returned a
  usable tool call, with 0 errors and no retries, for $0.133.

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
- **Tool schemas guide, but do not bind.** With a compact copy of the same strict schema
  (descriptions removed, 22 KB), Nemotron 3 Super left out a required field in both calls of a
  live trial. The full schema fixed it, but a constrained decoder would not have let it happen.
  Please document whether `tools` parameters are enforced while decoding, or offer a strict
  mode for forced tool calls.
- **"None" written as text in nullable fields.** In most plans the model wrote
  `"refusal": "None"` (Python's null as a string) where the schema allows null, and one
  otherwise valid plan was lost to it before we read "None", "null", "n/a" and "" as null in
  nullable fields. Schema enforcement would prevent this; until then, a line in the model
  guidance would help.
- **Completions run on after a complete tool call, and are billed.** In our first four
  recording rounds, 7 calls with a forced tool call kept generating after the call was complete
  and ran to `max_tokens` (8,000 or 12,000); every token was billed, while the other 83 calls
  finished within 3,150 completion tokens. We cut `max_tokens` to 5,000; one complete call later
  used 4,403, so the margin is thin. Ending the response once a forced tool call is complete
  would make the cap a safety net instead of a cost.
- **Latency per call is long for large tool prompts, with no progress signal.** Task Engineer
  calls (about 21k prompt tokens, up to 4,403 completion tokens) took 17 to 40 s, p50 23.7 s.
  Short agent turns took 0.7 to 2.4 s and Lightning summaries 0.4 to 1.0 s. Streaming the
  tool-call arguments, or a documented latency per model and prompt size, would let a UI show
  progress during the long calls.
- **HTTP 529 without guidance.** In one recording run of this version, 8 of 12 planning calls
  came back HTTP 529 and no answer. Our client did not retry 529 at first; it does now, and a
  second pass got every answer. Please document what 529 means on Token Factory and send a
  `Retry-After` header with it.
- **A cut-off call is billed and gives nothing.** One repair call ran to its 5,000-token cap
  (`finish_reason: length`) with no tool call that parsed, and its $0.013 was billed. We now treat
  such a call as failed and never judge it. Ending a forced tool call once its arguments are
  complete, as above, would prevent most of these.
- **Small things.** Tool definitions are billed as prompt tokens (about 1,330 tokens against
  600 for the same messages with json_schema); that deserves a line in the pricing docs.
  Model IDs mix casing and separators (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
  `nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`) and are
  case-sensitive.

### AI Cloud: Serverless Jobs, Object Storage, Container Registry, MysteryBox, CLI

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
- **One CPU job carried a whole pre-registered evaluation.** The Studio's use-case benchmark
  ran every frozen arm of three use cases in one uk-south2 `cpu-d3` 32-vCPU job: 206 s wall,
  about $0.06, and its outcomes matched our laptop's on 72 of 72 development re-runs.

What needs work:

- **Capacity is invisible at submit time.** A job on a platform with no capacity sat in
  `PROVISIONING` for over 10 minutes. The logs said only "Waiting before retrying resource
  provisioning", while `capacity resource-advice list` already showed L40S at 0. Please fail
  fast or warn at submit. Capacity advice also has no CPU rows.
- **The default CPU quota in eu-north1 is 0** (`compute.instance.non-gpu.vcpu`). CPU-only
  jobs there are impossible without a request, so our first jobs ran CPU physics on an idle
  H100 at $4.50/h. uk-south2 has 200 by default. It still shapes the fallback: when uk-south2
  cannot start a CPU job, our fallback is eu-north1's GPU target, so the CPU physics runs on an
  idle H100 at $4.50/h against $1.056/h for the 32-vCPU CPU preset.
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
  only the bucket ID works, and the message could say so. So every region's bucket ID has to
  live in our private config next to its name; accepting a name or an `s3://` URI would remove
  that step. `ai job cancel` blocked for more
  than 2 minutes on a job that never started. `--dry-run` prints plain text even with
  `--format json`. `ai job get-by-name` requires `--parent-id` even when the profile sets
  one. The `IMAGE_PULLING` state is missing from the jobs docs.
- **The CLI session expires after about 12 hours, and unattended commands then hang.** The
  credential helper printed `exit status 7`, and `docker push` waited out a 10-minute
  timeout instead of failing. `nebius iam whoami` waited for a browser login. Please fail
  fast when no terminal is attached, and document service-account profiles next to the
  quick start. The docs show how to make a service account's S3 access key, but not how to
  point the CLI itself at a service account, so our procedure starts every scored batch from a
  freshly refreshed browser session.
- **Budgets are console-only.** There is no CLI or API for budgets or alerts, so we enforce
  our own spend cap from a ledger of wall-time estimates.
- **About 2 minutes of overhead per job** before the container runs: 111 to 128 s from
  creation to container start on uk-south2, of which the 717 MB image pull took 21 to 32 s. In
  the use-case benchmark's job that was 119 s of overhead against 88 s of work, so we fan the
  Studio's search out one job per phase, not per shard. Warm pools or pre-pulled images for a
  repeated digest would help short jobs. There is also no job ID inside the container; a
  `NEBIUS_JOB_ID` variable would let outputs name the job that made them.
- **CPU prices are split across two pages** (per vCPU-hour and per GiB-hour). A per-preset
  table like the GPU one would help.

### NVIDIA Nemotron models

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
- **Super as a planner is good after one repair, rarely on the first call.** On the Studio's
  recorded gate, 8 of 8 use cases ended with a valid plan (1 on the first call, 5 after one
  repair call, 2 after our layout fixer), and 2 of 2 unsupported tasks (folding a shirt,
  pouring) were refused with the nearest primitive. The first-call rate was 1 of 8 in three
  recording rounds in a row. The failures were spatial and sequential: parts placed inside
  their totes, a latch and a valve against a door, a step left holding an object while the next
  acted on another. Given every problem with its path, the repair call fixed them.
- **Super as a planner of real jobs needs physics in the loop.** For the six jobs of this version
  (a tote, machine tending, a solar panel, floor tiles, a pipe coupling, a hotel delivery), the
  prompt grew to about 27,000 tokens and a call cost about $0.01. In the recorded run of record,
  no plan was valid on the first call: 3 were valid after one repair, 2 after our layout fixer, and
  1 failed. Plans that passed every check could still not run in physics: a pass criterion asking
  for 4 mm of overlap, a door edge 350 mm out against a 338 mm safe reach, a lift shorter than the
  vacuum frame gives under an 8 kg panel. Each became a check with a plain message, and the repair
  call then fixed most of them. One repair raised a pass criterion together with the lift it was
  told to raise, because our message named the quantity and not the field. Naming the exact field
  path fixed that for us.
- **The same model at two providers.** Nemotron 3 Super is also listed on the NVIDIA API catalog,
  behind an OpenAI-compatible endpoint. One model at two providers means a planner can take either
  an NVIDIA key or a Token Factory key, with only the base URL and the model string to change.
- **Super as a tool-calling agent was crisp.** Each recorded Studio session called
  `plan_task`, `run_task_tournament` and `export_build_pack` in order, 3 tool calls in 4 turns,
  for $0.0037 to $0.0041. Its final answers use typographic characters (U+2011 non-breaking
  hyphens, U+202F narrow spaces) even inside setup IDs, which breaks a plain-text search for
  the ID; we read IDs from the tool results, never from the prose.
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

### Onboarding: zero to hello world

- Token Factory: a key in `.env`, the OpenAI SDK with the base URL, `GET /v1/models`, one
  chat call. It worked with a trial key within minutes.
- AI Cloud: the CLI install and browser login took about 2 minutes, and zero to the first
  job took minutes.
- Gotcha: the OpenAI SDK sends `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment
  as headers to any base URL. A developer who also uses OpenAI would send those IDs to
  Token Factory. One line in the quick start would warn them.

### Would we build on it again?

Yes. Token Factory gave us four NVIDIA models behind one OpenAI-compatible key, with limits
that match the docs, and it served a 473-call batch from inside a job without an error.
Serverless Jobs suit a pre-registered experiment: every scored number comes from one
pinned image, and the whole project's compute cost about $4. What cost us time were the
silent failures: capacity waits, image-pull and quota errors that the dry run let through,
the non-POSIX bucket mount, the CLI hang after the session expired, and schema options that
a model accepts and then ignores. The Studio added two of the same kind: tool schemas that a
model may step outside, and completions billed past a complete tool call. This version added one more: HTTP 529 answers with no
guidance on when to retry. Fix those first and
the platform is excellent for this kind of work.

---

## Appendix: dated log (evidence)

### Token Factory

#### 2026-10-15: first contact and the output-mode probe

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

#### 2026-10-15: the first live design round (`rtt slice`)

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

#### 2026-10-17: the confirmatory batch

Sources: `docs/NEBIUS.md` ("Confirmatory and held-out jobs"), the Nemotron arm's
`search/llm_ledger.jsonl` and `search/manifest.json` in `runs/cloud/search-20261017-062726/`.

- 32 runs of the Nemotron arm called Super in lockstep from one `cpu-d3` job: 473 calls,
  0 errors, every call on its first attempt. 6,980,047 prompt and 1,638,651 completion
  tokens (584,902 reasoning), $3.569. p50 latency 15.1 s, max 65.8 s, `max_tokens` 16,384.
- Main proposals, reflections and repairs: 256, 130 and 87 calls. 1,766 proposals were
  rejected (910 too similar, 499 invalid, 214 not novel, 143 duplicates), and 131 of the
  1,024 model slots were filled at random [derived: 32 runs x 32 model slots].

#### 2026-10-18: the Studio's Task Engineer (a 33 KB forced tool call)

Sources: `docs/STUDIO_ENGINEER.md` (sections 1, 6 and 7) and the recorded gate run,
`src/rtt/studio/engineer/recorded/gate-2026-10-18-s4e/` (`manifest.json`, `ledger.jsonl`).
Nemotron 3 Super, forced `submit_task_plan` call, thinking on, temperature 0.6, top_p 0.95,
`max_tokens` 5,000.

What worked:

- The tool is the contract's own schema, 33,196 bytes with descriptions: every object and step
  is a closed `anyOf` branch with const discriminators and bounded fields. Token Factory took it
  as a forced `tool_choice` without complaint. The one gate run made 17 live calls: 352,696
  prompt and 29,998 completion tokens, 0 errors, no retries, $0.133 (cap $0.24).
- Latency: p50 23.7 s, 17.2 to 40.1 s, for 20.3k to 21.7k prompt tokens per call.

What surprised us:

- **A compact schema let the model skip required fields.** `--compact-tool` drops every
  description (22 KB, the same strict structure). In a live trial on the cabinet case the model
  then left out the steps' required `description` field in both calls. The full schema is the
  default; its extra 11 KB costs about 2,800 prompt tokens, $0.0008 a call.
- **"None" as a string.** Nemotron writes `"refusal": "None"` in most plans; a first trial lost an
  otherwise valid cabinet plan to it. `normalise_nulls` now reads `"None"`, `"null"`, `"n/a"` and
  `""` as null in nullable fields (refusal, target, ref, criterion bounds, constraints, nearest
  primitive), and the plan notes it. Nothing else is rewritten.
- **Run-on completions.** Through the fourth recording round, 83 complete calls used at most
  3,150 completion tokens (reasoning included), but 7 calls kept generating after a complete
  tool call and ran to the cap (8,000 or 12,000), which was billed. The fifth round lowered the
  cap to 5,000; no call ran to it, but one complete call used 4,403 tokens.
- **First calls.** 1 of 8 plans was valid on the first call in each of the last three recording
  rounds (see the Nemotron section below).
- *Suggestions:* say whether tool parameters are constrained while decoding (or offer a strict
  mode); stop generating once a forced tool call is complete; stream tool-call arguments so a
  20 to 40 s call can show progress.

#### 2026-10-26 and 27: re-planning six real jobs (Studio v2)

Source: the Studio v2 build's Task Engineer recordings (`recorded/gate-2026-10-27-v2*`, their
`manifest.json` and `ledger.jsonl`) and the task report's spend tables. Nemotron 3 Super, forced
`submit_task_plan` call, `max_tokens` 5,000.

- **Spend.** 54 live calls in three rounds to 27 October, $0.4845 in all (first round 15 calls,
  $0.1528; second round 35 calls, $0.2903; third round 4 calls, $0.0414). Prompts ran 27,129 to
  29,320 tokens and completions 928 to 3,561 tokens on complete calls; a call cost $0.009 to
  $0.013.
- **HTTP 529.** In one run (v2b), 8 of 12 calls returned HTTP 529 with no answer. The client did
  not retry 529 then; after the fix, the four cases without an answer were re-run once and all
  ended valid. That run was later superseded, because the rules changed.
- **A call cut at the cap.** In the run of record (v2c), the hotel job's repair call used all
  5,000 completion tokens and returned no tool call that parsed ($0.013). The plan came from the
  main call plus our layout fixer. Under the v2 rule set, a call with `finish_reason: length` is
  recorded as truncated and never judged.
- **First calls.** In v2c no plan was valid on the first call (0 of 6): 3 after one repair, 2 after
  the layout fixer, 1 failed (the machine-tending layout, inside the machine). The first version's
  gate had 1 of 8 on the first call.
- **What a physics probe found after the checks passed.** A solar plan asking for at least 4 mm of
  `penetration_mm` together with a gap bound below its start gap; a hotel door whose free edge sat
  350 mm out (the safe reach is 338 mm); a machine tray beside the machine where the words put it
  inside; a lift of 20 mm under a vacuum frame that gives about 23 mm under 8 kg before the panel
  rises. None was hand-edited. Each became a check and a line in the prompt, and the case was
  planned again.

### AI Cloud / Serverless Jobs

#### 2026-10-15: first jobs, image pipeline and the bucket mount

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

#### 2026-10-16: CPU jobs in uk-south2, and a second region

Sources: `docs/NEBIUS.md` (targets section), the spend ledger and the fetched runs
`runs/cloud/{m1-20261015-171556,search-20261015-172651,probe-20261015-173010}`.

What worked:

- **CPU Serverless Jobs are the right tool for CPU physics.** The same 40-design random
  search took 132 s wall on `cpu-d3` 16 vCPU ($0.019) against 146 s on the H100 preset
  ($0.182). Physics ran at 30.1k vs 31.0k simulation steps/s. The `cpu-d3` 16-vCPU preset
  is $0.528/h against $4.50/h, so a scored job costs 8-10x less.
- **Reproducible across Nebius hosts.** The search's episode, design and history tables
  from the AMD EPYC Genoa CPU host and from the H100 host were byte-identical; only a
  timestamp differed. (Our laptop, a Zen 3 CPU, differed in 1 of 1,920 success flags for
  the same seed, so the experiment's executor of record is pinned to the Nebius CPU
  platform.)
- **Secrets by ID across projects.** A uk-south2 job referenced the eu-north1 MysteryBox
  secret by its ID and received the key; a second copy of the secret was not needed.

What surprised us:

- **Storage class quota trap.** The tenant had no quota for the standard storage class in
  uk-south2, so the first upload to a new bucket failed with `QuotaLimitExceeded`. Forcing
  the enhanced-throughput class fixed it. *Suggestion:* name the storage class in the
  error, and show per-class quota next to the bucket create command.
- **Cross-region image pulls fail late.** A uk-south2 job could not pull its image from the
  eu-north1 registry. The dry run passed; the job then spent about 64 s pulling and ended
  `StartFailed`. We now push the same digest to a registry in each region. *Suggestion:*
  have `--dry-run` check that the job's region can pull the image, or say in the error
  that the registry is in another region.
- **Capacity advice is GPU-only.** `capacity resource-advice list` has no CPU rows, so our
  CPU capacity check reads the vCPU quota instead.
- **CPU preset prices are split across two pages** (per vCPU-hour and per GiB-hour). We had
  to add them up to price a preset ($0.015 x 16 + $0.0045 x 64 = $0.528/h). *Suggestion:* a
  per-preset price table like the GPU one.
- **Provisioning overhead is the same on CPU:** about 110 s from creation to container
  start on both platforms, against 21-22 s of actual work for these jobs.

#### 2026-10-17: the v2 parity job, and an expired CLI session

Sources: `docs/NEBIUS.md` (v2 section), `docs/BENCH_V2.md` (GV4), the spend ledger and
`runs/cloud/gv4-20261016-181945/`.

What worked:

- **Bit-for-bit physics across machines, again.** 384 v2 episodes, with every finger
  generated inside the job, gave the same outcome as on our laptop in 384 of 384 cases;
  257 records matched bit for bit and the rest differed by at most 1e-15 mm. Every
  geometry hash matched. The job (`cpu-d3`, 32 vCPU, uk-south2) cost $0.040 and ran the
  episodes 3.5x faster than the laptop.
- **A 32-vCPU CPU job is as easy as a 16-vCPU one:** same image, same 119 s from creation to
  start, $1.056/h.

What surprised us:

- **The CLI session ends after about 12 hours, and unattended commands then hang.** The CLI
  profile signs in through the browser. When that session expired, the Docker credential
  helper (`nebius registry docker-credential`) printed `Error getting credentials - err:
  exit status 7` and `docker push` waited for the whole 10-minute timeout without pushing;
  `nebius iam whoami` waited for a browser login. An overnight batch started from a script
  would stall the same way. *Suggestion:* fail fast with "session expired, run `nebius
  profile ...`" when no terminal is attached, have the credential helper exit non-zero
  quickly, and document service-account profiles for unattended use next to the federation
  quick-start.

#### 2026-10-19: the use-case benchmark on uk-south2 CPU

Sources: `docs/USECASE_V1_RESULTS.md` (Compute), `docs/USECASE_V1.md` (sections 8, 9 and 11)
and the benchmark's own spend ledger `runs/cloud/usecase-v1-spend.jsonl` (git-ignored).

- **One job, one evaluation.** `aijob-e05e55n905qj9va2fh` ran every frozen arm of the three
  use cases, 32 held-out and 8 development conditions each, on `cpu-d3` `32vcpu-128gb` in
  uk-south2: 119 s from creation to container start, 88 s running, 206 s wall, about $0.06
  (wall-based; $0.026 run-only). Its outcomes matched the laptop's on 72 of 72 development
  re-runs; trajectory hashes matched on 41 of 72, the rest differing by floating-point rounding
  between CPUs, never in an outcome.
- **Provisioning shaped the design.** With about 2 minutes before any container runs, the
  Studio's fan-out (`rtt.studio.cloud`) sends one job per phase of the search, not one per
  shard, and the benchmark budgeted two attempts cancelled at the 15-minute provisioning
  timeout as contingency. Neither was needed.
- **Bucket mount, again.** Phase inputs and results go through the same local staging as
  every other job, because the mount rejects seeks and renames; the bucket is mounted by ID.
- **The secret stayed on the laptop.** The job received only the derived conditions as work
  items; no job got the held-out secret or the Token Factory key.
- **The CLI session.** The procedure starts from a fresh browser-federated CLI session (it
  lasts about 12 hours), because we have no documented way to run the CLI as a service account.

### NVIDIA Nemotron models

#### 2026-10-18: Super as the Studio's planner and agent

Sources: `docs/STUDIO_ENGINEER.md` (section 6), the recorded Studio sessions
`web/public/data/studio/sessions/h{1,2,3}.json`, and `docs/EFFECTORS_RESULTS.md` (sections 4
and 6) for the effector designer.

- **Planner.** In the one judged gate run, 8 of 8 use cases ended valid: 1 on the first call,
  5 after one repair call, 2 after the code's layout fixer; 0 failed. 2 of 2 unsupported tasks
  (fold a shirt, pour) were refused with a nearest primitive. What the repair calls fixed: parts
  inside the tote, the tray or the drawer; the latch and the valve against the door; a valve
  against a hatch; steps left holding while the next acted on another object; an insert of a
  thread left holding; a twist on a nut before its insert. The first-call rate was 1 of 8 in
  the last three recording rounds alike.
- **Agent.** Each recorded session ran `plan_task`, `run_task_tournament` and
  `export_build_pack` in order: 4 agent turns, 3 tool calls, $0.0037 to $0.0041 per session.
  Agent turns took 0.66 to 2.42 s; Lightning summaries 0.38 to 1.0 s. The final answers carry
  U+2011 (non-breaking hyphen) and U+202F (narrow no-break space) characters, also inside setup
  IDs such as "setup 72c634ec4fa3"; IDs are read from the tool results, never from the prose.
- **Effector designer.** 23 live calls (12 proposals, 11 repairs), 188,427 prompt and 25,864
  completion tokens, $0.0798, every call successful. 15 of the 48 designs requested did not
  become candidates: refused by the validators (mostly the opening rule `too_narrow`) or by the
  duplicate and too-similar screens.

#### 2026-10-16: the v2 design tool on the real prompt

Sources: `docs/BENCH_V2.md` (GV5), `runs/gates-v2/gv5/` (probe and run ledgers).

- **Five calls of Super with the v2 `submit_designs` tool** (4 slots plus 4 reserves, each slot
  with a hypothesis, a change list and distances), on the real round-1 prompt of 12.6k tokens
  with six parts, the placements, the registration tolerances and an engineering-notes block:
  16 of 20 primary slots valid under the full geometry validators, and every call filled its
  4 slots from its own reserves, so no repair call was needed. $0.030 in total, 10.6-15.0 s
  per call, 2.1-3.2k completion tokens.
- The invalid slots were near misses a person would also make without a calculator: a seat
  closure 2-7 mm beyond the 40 mm stroke on the hex standoff, and a funnel draft outside its
  15-70 deg range.
- **A full 40-design v2 run** (15 calls, the cap is 16) cost $0.114: 225k prompt and 51.6k
  completion tokens, 17.4k of them reasoning; p50 latency 15.8 s, max 31.1 s. 29 of 32 slots
  were model designs (14 from reserves) and 3 were filled at random. Replaying it from the
  response cache made 0 API calls and reproduced the run exactly.
- An earlier probe whose prompt described synthetic stand-in parts looked perfect under
  schema checks (100 % valid), but only 35 % of its slots passed the geometry validators on
  the real parts. Testing a tool-calling model on a stand-in prompt overstates it; the real
  prompt is cheap to build.

#### 2026-10-15: proposer output-mode probe (2 calls per model and mode)

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

#### 2026-10-15: first full Nemotron search run, inside a Nebius job

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

#### 2026-10-16: fixing near-copy proposals, and the Grasp Engineer agent

Sources: `docs/PROPOSER.md`, `runs/proposer_dev/` (ledgers and recordings) and the agent's
session files in `runs/agent/sessions/`.

- **Super cannot audit its own proposals against a long history.** We asked it to name the
  nearest already-tested design in the same cell and its distance before proposing. Over
  194 slots its "nearest" design was the true nearest in only 56, it overstated the
  distance by more than 50 points (of 100) in 100, and 17 of its 42 "untried cell" claims
  were wrong. Prompt changes alone moved random fallbacks from 14/32 to 13/32.
- **Reserves fixed it.** Asking for 4 reserve designs per call, used in order when a slot is
  rejected, brought fallbacks to 2/32 and 7/32 on two development seeds, at $0.11 per
  40-design run (max_tokens 16,384, median call about 14 s). The simulator still never
  sees a design the model did not write.
- **Super as a tool-calling agent** (`tool_choice="auto"`, 8 tools) ran 7-12 tool calls per
  session. Each Super turn cost $0.0007-0.0016 and took about 1-2 s; whole sessions cost
  $0.011-0.027. Lightning step summaries cost about $0.00005 each.
- **Lightning once invented a verdict.** At a step that produced no pass/fail result, its
  summary said the design passed. We now drop, in code, any summary sentence that claims a
  pass when the step's computed verdict is not a pass. *Suggestion:* small summarisers
  need an explicit "no verdict" signal; worth a line in model guidance.

### Onboarding: zero to hello world

#### 2026-10-15

- Hello world was a key in `.env`, the stock OpenAI SDK with the Token Factory base URL,
  `GET /v1/models`, then one chat call (`scripts/tf_smoke.py`). It worked with a trial key.
- **Gotcha:** the OpenAI SDK reads `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment
  and sends them as headers to whatever base URL it is given. A developer who also uses OpenAI
  would leak those identifiers to Token Factory. Our transport clears them. *Suggestion:* add a
  line to the quick-start.
- Response bodies and rate-limit headers follow the OpenAI shapes (`x-ratelimit-limit-*`,
  `x-ratelimit-remaining-*`, and `x-ratelimit-reset-*` as durations such as `1s`), so existing
  retry and throttling code carried over.

### Would we build again?

After day 2: still yes. Moving the physics to CPU jobs in uk-south2 cut the price of a
scored job by 8-10x with identical results. What it cost us: a storage-class quota error,
and a cross-region image pull that failed only after the dry run passed.

After day 1: yes for Token Factory, and yes for AI Cloud with caveats. An OpenAI-compatible
endpoint, four NVIDIA models behind one key, and limits that match the docs got us from zero
to a 28-call probe for under four cents. The mismatch between feature flags and `response_format` behaviour is the one
thing that would have bitten us silently without a probe. AI Cloud ran every job we
submitted once it had capacity, and secrets by reference, the registry helper and
`--dry-run` were excellent. Two things cost us real time: the undocumented
bucket-mount semantics, and a default CPU quota of 0 that puts CPU physics on an idle
H100.
