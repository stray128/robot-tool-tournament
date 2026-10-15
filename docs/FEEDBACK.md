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

## AI Cloud / Serverless Jobs

Not exercised yet: the Nebius bring-up (CLI profile, Object Storage, first CPU job) is
scheduled after the local simulator gate. Entries will follow the same format: what we ran, how
long it took, and what we would change.

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

Preliminary, after day 1: yes for Token Factory. An OpenAI-compatible endpoint, four NVIDIA
models behind one key, and limits that match the docs got us from zero to a 28-call probe for
under four cents. The mismatch between feature flags and `response_format` behaviour is the one
thing that would have bitten us silently without a probe. AI Cloud: to be decided after the
first jobs run.
