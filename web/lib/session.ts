/**
 * Grasp Engineer sessions: the agent's event stream, recorded or live, and the
 * timeline the /part page draws from it.
 *
 * The primary format is the agent's own (src/rtt/agent/recorder.py, `format:
 * "rtt-agent-session"`): events with `seq`, `ts`, `t_s` and a dotted `type`
 * (`session.started`, `llm.call`, `agent.message`, `tool.call`, `tool.result`,
 * `tool.refused`, `job.update`, `summary`, `session.finished`, `error`), and tool results
 * shaped by src/rtt/agent/tools.py. The API streams the same events over SSE.
 *
 * Reading is defensive (web/README.md, "Agent sessions"): unknown fields are ignored,
 * common aliases are accepted (`event`/`kind` for `type`, underscores for dots,
 * `tool`/`tool_name` for `name`, `arguments`/`input` for `args`, JSON-encoded strings),
 * an OpenAI-style `messages` transcript is converted, and an event that cannot be read
 * is dropped and counted instead of breaking the page.
 */

import type { LengthUnit } from "./stl";

// ---------------------------------------------------------------- event model

export type ToolKind = "inspect" | "propose" | "job" | "results" | "stress" | "printpack" | "ledger" | "other";

/** The Grasp Engineer's tools (build plan, Addendum A, Move 2) and how each is drawn. */
export const TOOL_KINDS: Readonly<Record<string, ToolKind>> = {
  inspect_part: "inspect",
  propose_designs: "propose",
  launch_sim_job: "job",
  poll_job: "job",
  fetch_results: "results",
  stress_test: "stress",
  export_print_pack: "printpack",
  cost_ledger: "ledger",
};

export const TOOL_TITLES: Readonly<Record<ToolKind, string>> = {
  inspect: "Inspect the part",
  propose: "Nemotron proposes designs",
  job: "Simulation job",
  results: "Results",
  stress: "Stress test",
  printpack: "Print pack",
  ledger: "Cost ledger",
  other: "Tool call",
};

export type JobStatus = "queued" | "provisioning" | "running" | "completed" | "failed" | "cancelled" | "unknown";

export interface JobState {
  id: string;
  status: JobStatus;
  /** Where it runs: a Nebius Serverless Job, or the local executor. */
  backend: "nebius" | "local" | null;
  platform: string | null;
  preset: string | null;
  region: string | null;
  elapsed_s: number | null;
  cost_usd: number | null;
  /** The cost is the agent's wall-time estimate (`est_usd`), not a billed amount. */
  cost_estimated: boolean;
  designs: number | null;
  episodes: number | null;
  error: string | null;
  /** Session time (s) of the last update. */
  updated_t: number;
}

export interface Caps {
  max_tool_calls: number | null;
  max_concurrent_jobs: number | null;
  budget_usd: number | null;
}

export interface Ledger {
  llm_usd: number | null;
  compute_usd: number | null;
  total_usd: number | null;
  budget_usd: number | null;
  tool_calls: number | null;
  max_tool_calls: number | null;
}

export interface LlmTotals {
  calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  latency_s: number;
  model: string | null;
}

export type SessionEndStatus = "completed" | "failed" | "stopped" | "capped" | "cancelled";

export type AgentEvent =
  | {
      type: "session_started";
      t: number;
      session_id: string | null;
      model: string | null;
      caps: Caps | null;
      part_name: string | null;
      offline: boolean;
    }
  | { type: "message"; t: number; role: "agent" | "system" | "summary"; text: string }
  | {
      type: "llm_call";
      t: number;
      role: string | null;
      model: string | null;
      tokens_in: number;
      tokens_out: number;
      cost_usd: number;
      latency_s: number;
      cached: boolean;
    }
  | { type: "tool_call"; t: number; call_id: string; tool: string; args: Record<string, unknown> }
  | {
      type: "tool_result";
      t: number;
      call_id: string;
      tool: string | null;
      ok: boolean;
      /** A cap (tool budget, passcode, dollars, concurrency) refused the call. */
      refused: boolean;
      result: unknown;
      error: string | null;
      cost_usd: number | null;
    }
  | { type: "job_update"; t: number; call_id: string | null; job: JobState }
  | { type: "ledger"; t: number; ledger: Ledger }
  | {
      type: "session_ended";
      t: number;
      status: SessionEndStatus;
      summary: string | null;
      answer: string | null;
      reason: string | null;
      printpack_id: string | null;
      ledger: Ledger | null;
    }
  | { type: "error"; t: number; message: string };

// ---------------------------------------------------------------- loose readers

type Obj = Record<string, unknown>;

export function isObj(v: unknown): v is Obj {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** A JSON string becomes its value; anything else is returned unchanged. */
export function maybeJson(v: unknown): unknown {
  if (typeof v !== "string") return v;
  const s = v.trim();
  if (!(s.startsWith("{") || s.startsWith("["))) return v;
  try {
    return JSON.parse(s);
  } catch {
    return v;
  }
}

export function pick(o: Obj, ...keys: string[]): unknown {
  for (const k of keys) if (o[k] !== undefined && o[k] !== null) return o[k];
  return undefined;
}

export function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  return null;
}

export function str(v: unknown): string | null {
  if (typeof v === "string") return v;
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  return null;
}

function objOf(v: unknown): Obj {
  const parsed = maybeJson(v);
  return isObj(parsed) ? parsed : {};
}

/** A percentage field (`success_pct: 75`) or a fraction field (`success: 0.75`) as a fraction. */
function fraction(o: Obj, pctKeys: string[], fracKeys: string[]): number | null {
  const pct = num(pick(o, ...pctKeys));
  if (pct !== null) return pct / 100;
  const f = num(pick(o, ...fracKeys));
  if (f === null) return null;
  return f > 1 && f <= 100 ? f / 100 : f;
}

/** Absolute time in epoch seconds from an ISO string or epoch s/ms number, else null. */
export function epochSeconds(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v > 1e11 ? v / 1000 : v > 1e8 ? v : null;
  if (typeof v === "string") {
    const ms = Date.parse(v);
    return Number.isNaN(ms) ? null : ms / 1000;
  }
  return null;
}

// ---------------------------------------------------------------- normalisation

type RawType =
  | AgentEvent["type"]
  | "tool_refused"
  | "summary";

const TYPE_ALIASES: Record<string, RawType> = {
  session_started: "session_started",
  session_start: "session_started",
  started: "session_started",
  message: "message",
  agent_message: "message",
  assistant: "message",
  narration: "message",
  summary: "summary",
  llm_call: "llm_call",
  tool_call: "tool_call",
  tool_use: "tool_call",
  tool_result: "tool_result",
  tool_output: "tool_result",
  tool_response: "tool_result",
  tool_refused: "tool_refused",
  job_update: "job_update",
  job_status: "job_update",
  ledger: "ledger",
  cost: "ledger",
  session_finished: "session_ended",
  session_ended: "session_ended",
  session_end: "session_ended",
  finished: "session_ended",
  done: "session_ended",
  error: "error",
};

/** `tool.result` -> `tool_result`, `Tool-Call` -> `tool_call`. */
function typeKey(name: string): string {
  return name.trim().toLowerCase().replace(/[.\s-]+/g, "_");
}

const JOB_STATUS: Record<string, JobStatus> = {
  queued: "queued",
  pending: "queued",
  submitted: "queued",
  created: "queued",
  provisioning: "provisioning",
  starting: "provisioning",
  scheduling: "provisioning",
  running: "running",
  active: "running",
  in_progress: "running",
  completed: "completed",
  complete: "completed",
  succeeded: "completed",
  success: "completed",
  done: "completed",
  finished: "completed",
  failed: "failed",
  error: "failed",
  errored: "failed",
  cancelled: "cancelled",
  canceled: "cancelled",
  stopped: "cancelled",
  deleting: "cancelled",
};

export function jobStatus(v: unknown): JobStatus {
  const s = str(v)?.toLowerCase().replace(/[\s-]+/g, "_");
  return (s && JOB_STATUS[s]) || "unknown";
}

export function isJobActive(status: JobStatus): boolean {
  return status === "queued" || status === "provisioning" || status === "running";
}

/**
 * A job record from a tool result or a job update: `{job: {...}}` or the job itself
 * (the agent's `JobRecord.to_json()`: `job_id`, `target`, `state`, `elapsed_s`,
 * `est_usd`, `platform`, `preset`, `designs`). Returns null without an id.
 */
export function readJob(v: unknown, t: number): JobState | null {
  const outer = objOf(v);
  const o = isObj(outer.job) ? outer.job : outer;
  const id = str(pick(o, "job_id", "id"));
  if (!id) return null;
  const minutes = num(pick(o, "minutes", "wall_min"));
  const backendRaw = str(pick(o, "target", "backend", "kind", "launcher"))?.toLowerCase() ?? "";
  const platform = str(pick(o, "platform"));
  const backend: JobState["backend"] = backendRaw.includes("nebius")
    ? "nebius"
    : backendRaw.includes("local")
      ? "local"
      : platform || str(pick(o, "region"))
        ? "nebius"
        : null;
  const billed = num(pick(o, "cost_usd", "usd", "cost"));
  const estimate = num(pick(o, "est_usd", "estimated_usd"));
  const designs = pick(o, "designs", "n_designs");
  return {
    id,
    status: jobStatus(pick(o, "status", "state")),
    backend,
    platform,
    preset: str(pick(o, "preset")),
    region: str(pick(o, "region")),
    elapsed_s: num(pick(o, "elapsed_s", "wall_s", "run_s", "duration_s")) ?? (minutes !== null ? minutes * 60 : null),
    cost_usd: billed ?? estimate,
    cost_estimated: billed === null && estimate !== null,
    designs: Array.isArray(designs) ? designs.length : num(designs),
    episodes: num(pick(o, "episodes", "n_episodes")),
    error: str(pick(o, "error")),
    updated_t: t,
  };
}

/** Merge a newer job record into an older one, keeping known fields the newer one omits. */
export function mergeJob(old: JobState | undefined, next: JobState): JobState {
  if (!old) return next;
  const keep = <T>(a: T | null, b: T | null) => (b === null ? a : b);
  return {
    id: next.id,
    status: next.status === "unknown" ? old.status : next.status,
    backend: keep(old.backend, next.backend),
    platform: keep(old.platform, next.platform),
    preset: keep(old.preset, next.preset),
    region: keep(old.region, next.region),
    elapsed_s: keep(old.elapsed_s, next.elapsed_s),
    cost_usd: keep(old.cost_usd, next.cost_usd),
    cost_estimated: next.cost_usd === null ? old.cost_estimated : next.cost_estimated,
    designs: keep(old.designs, next.designs),
    episodes: keep(old.episodes, next.episodes),
    error: keep(old.error, next.error),
    updated_t: Math.max(old.updated_t, next.updated_t),
  };
}

function readCaps(v: unknown): Caps | null {
  if (!isObj(v)) return null;
  return {
    max_tool_calls: num(pick(v, "max_tool_calls", "tool_calls")),
    max_concurrent_jobs: num(pick(v, "max_concurrent_jobs", "concurrent_jobs")),
    budget_usd: num(pick(v, "session_usd", "budget_usd", "usd_cap", "cap_usd", "budget")),
  };
}

/**
 * Spend from a `cost_ledger` result, a `session.finished` event or a plain ledger:
 * the agent's `spend` snapshot (`spent_usd`, `session_cap_usd`, `by_kind_usd`) with
 * `llm_by_model`, `jobs`, `tool_calls_used` and `caps`, or flat `llm_usd`/`compute_usd`.
 */
export function readLedger(v: unknown): Ledger {
  const o = objOf(v);
  const inner = isObj(o.ledger) ? o.ledger : o;
  const snap = isObj(inner.spend) ? inner.spend : inner;
  const byKind = objOf(snap.by_kind_usd);
  const kinds = Object.entries(byKind).flatMap(([k, x]) => (num(x) !== null ? [[k, num(x)!] as const] : []));
  const byModel = Object.values(objOf(inner.llm_by_model)).map((m) => num(objOf(m).usd) ?? 0);
  const jobs = Array.isArray(inner.jobs) ? inner.jobs.map((j) => num(pick(objOf(j), "cost_usd", "est_usd")) ?? 0) : [];
  const llm =
    num(pick(inner, "llm_usd", "tokens_usd", "model_usd")) ??
    (kinds.length ? kinds.filter(([k]) => k === "llm").reduce((s, [, x]) => s + x, 0) : null) ??
    (byModel.length ? byModel.reduce((a, b) => a + b, 0) : null);
  const compute =
    num(pick(inner, "compute_usd", "jobs_usd", "sim_usd")) ??
    (jobs.length ? jobs.reduce((a, b) => a + b, 0) : null) ??
    (kinds.length ? kinds.filter(([k]) => k !== "llm").reduce((s, [, x]) => s + x, 0) : null);
  const total =
    num(pick(snap, "spent_usd", "total_usd", "usd")) ??
    num(pick(inner, "total_usd", "spent_usd")) ??
    (llm !== null || compute !== null ? (llm ?? 0) + (compute ?? 0) : null);
  const caps = objOf(inner.caps);
  return {
    llm_usd: llm,
    compute_usd: compute,
    total_usd: total,
    budget_usd: num(pick(snap, "session_cap_usd", "budget_usd", "cap_usd", "usd_cap")) ?? num(caps.session_usd),
    tool_calls: num(pick(inner, "tool_calls_used", "tool_calls", "calls")),
    max_tool_calls: num(pick(inner, "max_tool_calls")) ?? num(caps.max_tool_calls),
  };
}

const END_STATUS: Record<string, SessionEndStatus> = {
  completed: "completed",
  complete: "completed",
  done: "completed",
  ok: "completed",
  success: "completed",
  succeeded: "completed",
  failed: "failed",
  error: "failed",
  stopped: "stopped",
  capped: "capped",
  cap_reached: "capped",
  budget_exceeded: "capped",
  cancelled: "cancelled",
  canceled: "cancelled",
  aborted: "cancelled",
};

/**
 * One raw event (an SSE `data:` payload or a recorded entry) as an AgentEvent, or null
 * when it cannot be read or carries nothing to draw (an `agent.message` with no text,
 * whose tool calls arrive as their own events). `t` is session seconds, computed by
 * the caller when the event carries no time. `sseEvent` is the SSE `event:` field,
 * used when the payload has no type of its own.
 */
export function normalizeEvent(raw: unknown, t: number, sseEvent?: string): AgentEvent | null {
  const o = objOf(raw);
  const typeName = str(pick(o, "type", "event", "kind")) ?? sseEvent ?? "";
  const type = TYPE_ALIASES[typeKey(typeName)];
  if (!type) return null;
  switch (type) {
    case "session_started": {
      const part = pick(o, "part");
      const models = objOf(o.models);
      const offline = o.offline === true;
      return {
        type,
        t,
        session_id: str(pick(o, "session_id", "id")),
        model: str(pick(o, "model", "model_id")) ?? str(pick(models, "proposer", "agent")),
        caps: readCaps(pick(o, "caps", "limits")),
        part_name: str(pick(o, "part_name")) ?? (isObj(part) ? str(pick(part, "name", "filename")) : str(part)),
        offline,
      };
    }
    case "message":
    case "summary": {
      const text = str(pick(o, "text", "content", "message"));
      if (!text || !text.trim()) return null;
      const role = type === "summary" ? "summary" : str(o.role) === "system" ? "system" : "agent";
      return { type: "message", t, role, text };
    }
    case "llm_call":
      return {
        type,
        t,
        role: str(o.role),
        model: str(pick(o, "model", "model_id")),
        tokens_in: num(pick(o, "prompt_tokens", "tokens_in")) ?? 0,
        tokens_out: num(pick(o, "completion_tokens", "tokens_out")) ?? 0,
        cost_usd: o.cached === true ? 0 : (num(o.cost_usd) ?? 0),
        latency_s: num(o.latency_s) ?? 0,
        cached: o.cached === true,
      };
    case "tool_call": {
      const tool = str(pick(o, "name", "tool", "tool_name"));
      const call_id = str(pick(o, "call_id", "tool_call_id", "id")) ?? (tool ? `${tool}@${t}` : null);
      if (!tool || !call_id) return null;
      return { type, t, call_id, tool, args: objOf(pick(o, "args", "arguments", "input", "params")) };
    }
    case "tool_result":
    case "tool_refused": {
      const call_id = str(pick(o, "call_id", "tool_call_id", "id"));
      if (!call_id) return null;
      const error = str(pick(o, "error", "error_message"));
      const okRaw = pick(o, "ok", "success");
      const status = str(o.status)?.toLowerCase();
      const refused = type === "tool_refused" || o.refused === true;
      const ok = refused
        ? false
        : typeof okRaw === "boolean"
          ? okRaw
          : status
            ? !["error", "failed"].includes(status)
            : error === null;
      const code = str(o.code);
      return {
        type: "tool_result",
        t,
        call_id,
        tool: str(pick(o, "name", "tool", "tool_name")),
        ok,
        refused,
        result: maybeJson(pick(o, "result", "output", "content", "data")),
        error: error ?? (ok ? null : code ? `Refused: ${code}` : "The tool reported an error."),
        cost_usd: num(pick(o, "cost_usd", "usd")),
      };
    }
    case "job_update": {
      const job = readJob(isObj(o.job) ? o.job : o, t);
      if (!job) return null;
      return { type, t, call_id: str(pick(o, "call_id", "tool_call_id")), job };
    }
    case "ledger":
      return { type, t, ledger: readLedger(o) };
    case "session_ended": {
      const s = str(pick(o, "status", "reason"))?.toLowerCase() ?? "completed";
      return {
        type,
        t,
        status: END_STATUS[s] ?? "completed",
        summary: str(pick(o, "summary", "text")),
        answer: str(pick(o, "answer", "message")),
        reason: str(pick(o, "reason")),
        printpack_id: str(pick(o, "printpack_id", "print_pack_id", "pack_id")),
        ledger: isObj(o.spend) || isObj(o.ledger) ? readLedger(o) : null,
      };
    }
    case "error": {
      const message = str(pick(o, "message", "error", "detail", "text"));
      return { type, t, message: message ?? "Unknown error" };
    }
  }
}

/** Event time from the raw payload: `t`/`t_s` (session seconds) or an absolute timestamp. */
export function rawTime(raw: unknown): { rel: number | null; abs: number | null } {
  const o = objOf(raw);
  const rel = num(pick(o, "t_s", "t", "elapsed_s"));
  const abs = epochSeconds(pick(o, "ts", "time", "timestamp", "at", "created_at"));
  return { rel, abs };
}

/**
 * Normalise a recorded event list. Times come from `t_s`/`t`, else from absolute
 * timestamps relative to the first one, else events are spaced one second apart.
 * Returns the events in time order (stable) and how many raw entries were dropped.
 */
export function normalizeEvents(raws: unknown[]): { events: AgentEvent[]; dropped: number } {
  let t0: number | null = null;
  let last = 0;
  const out: AgentEvent[] = [];
  let dropped = 0;
  for (const raw of raws) {
    const { rel, abs } = rawTime(raw);
    let t: number;
    if (rel !== null) t = rel;
    else if (abs !== null) {
      t0 ??= abs;
      t = abs - t0;
    } else t = out.length === 0 ? 0 : last + 1;
    t = Math.max(0, t);
    const ev = normalizeEvent(raw, t);
    if (ev) {
      out.push(ev);
      last = t;
    } else if (!isSilent(raw)) dropped++;
  }
  const order = out.map((e, i) => [e, i] as const).sort((a, b) => a[0].t - b[0].t || a[1] - b[1]);
  return { events: order.map(([e]) => e), dropped };
}

/** Raw events that are skipped on purpose (an agent turn with no text), not counted as dropped. */
function isSilent(raw: unknown): boolean {
  const o = objOf(raw);
  const key = typeKey(str(pick(o, "type", "event", "kind")) ?? "");
  return (TYPE_ALIASES[key] === "message" || TYPE_ALIASES[key] === "summary") && !str(pick(o, "text", "content", "message"))?.trim();
}

/**
 * Events from an OpenAI-style chat transcript: assistant text becomes a message,
 * assistant `tool_calls` become tool calls, `role: "tool"` replies become results.
 * Chat transcripts carry no times, so events are spaced one second apart.
 */
export function eventsFromChatMessages(messages: unknown[]): unknown[] {
  const out: unknown[] = [];
  let t = 0;
  for (const m of messages) {
    if (!isObj(m)) continue;
    const role = str(m.role);
    if (role === "assistant") {
      const text = str(m.content);
      if (text && text.trim()) out.push({ type: "message", t: t++, text });
      const calls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
      for (const c of calls) {
        if (!isObj(c)) continue;
        const fn = isObj(c.function) ? c.function : c;
        out.push({ type: "tool_call", t: t++, call_id: str(c.id), tool: str(fn.name), args: maybeJson(fn.arguments) });
      }
    } else if (role === "tool") {
      const content = objOf(m.content);
      // The agent's tool messages wrap the payload as {ok, result | error, ...}.
      const wrapped = typeof content.ok === "boolean" && ("result" in content || "error" in content);
      out.push({
        type: "tool_result",
        t: t++,
        call_id: str(m.tool_call_id),
        tool: str(m.name),
        ...(wrapped ? { ok: content.ok, result: content.result, error: content.error } : { result: maybeJson(m.content) }),
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------- timeline

export type StepKind = ToolKind | "message" | "error";

export interface Step {
  /** call_id for tool steps, "msg-<n>" / "err-<n>" otherwise. */
  id: string;
  kind: StepKind;
  tool: string | null;
  status: "running" | "ok" | "error";
  t_start: number;
  t_end: number | null;
  args: Record<string, unknown>;
  result: unknown;
  error: string | null;
  refused: boolean;
  cost_usd: number | null;
  /** Jobs this step launched or polled. */
  job_ids: string[];
  /** poll_job calls folded into this job card. */
  polls: number;
  /** Model calls made while this step ran (the proposer's, for propose_designs). */
  llm: LlmTotals | null;
  /** Message text; `role` says whose. */
  text: string | null;
  role: "agent" | "system" | "summary" | null;
}

export type SessionStatus = "idle" | "running" | SessionEndStatus;

export interface Timeline {
  session_id: string | null;
  model: string | null;
  part_name: string | null;
  caps: Caps | null;
  /** A scripted session with no model calls (`--offline`). */
  offline: boolean;
  steps: Step[];
  jobs: Record<string, JobState>;
  ledger: Ledger | null;
  /** Every model call of the session (agent, proposer, summarizer). */
  llm: LlmTotals;
  status: SessionStatus;
  summary: string | null;
  answer: string | null;
  reason: string | null;
  printpack_id: string | null;
  /** Tool calls made, poll calls included (the agent's cap counts every call). */
  tool_calls: number;
  /** call_id of a folded poll -> the job step it updates. */
  aliases: Record<string, string>;
  t_last: number;
}

function emptyLlm(): LlmTotals {
  return { calls: 0, tokens_in: 0, tokens_out: 0, cost_usd: 0, latency_s: 0, model: null };
}

function addLlm(a: LlmTotals | null, ev: Extract<AgentEvent, { type: "llm_call" }>): LlmTotals {
  const base = a ?? emptyLlm();
  return {
    calls: base.calls + 1,
    tokens_in: base.tokens_in + ev.tokens_in,
    tokens_out: base.tokens_out + ev.tokens_out,
    cost_usd: base.cost_usd + ev.cost_usd,
    latency_s: base.latency_s + ev.latency_s,
    model: ev.model ?? base.model,
  };
}

export function emptyTimeline(): Timeline {
  return {
    session_id: null,
    model: null,
    part_name: null,
    caps: null,
    offline: false,
    steps: [],
    jobs: {},
    ledger: null,
    llm: emptyLlm(),
    status: "idle",
    summary: null,
    answer: null,
    reason: null,
    printpack_id: null,
    tool_calls: 0,
    aliases: {},
    t_last: 0,
  };
}

export function toolKind(tool: string | null): ToolKind {
  return (tool && TOOL_KINDS[tool]) || "other";
}

function jobIdFrom(args: Record<string, unknown>): string | null {
  return str(pick(args, "job_id"));
}

function replaceStep(tl: Timeline, id: string, update: (s: Step) => Step): Step[] {
  return tl.steps.map((s) => (s.id === id ? update(s) : s));
}

function blankStep(id: string, kind: StepKind, tool: string | null, t: number, extra: Partial<Step>): Step {
  return {
    id,
    kind,
    tool,
    status: "running",
    t_start: t,
    t_end: null,
    args: {},
    result: null,
    error: null,
    refused: false,
    cost_usd: null,
    job_ids: [],
    polls: 0,
    llm: null,
    text: null,
    role: null,
    ...extra,
  };
}

/** Apply one event; pure, so React state and tests share it. */
export function applyEvent(tl: Timeline, ev: AgentEvent): Timeline {
  const next: Timeline = { ...tl, t_last: Math.max(tl.t_last, ev.t), status: tl.status === "idle" ? "running" : tl.status };
  switch (ev.type) {
    case "session_started":
      return {
        ...next,
        session_id: ev.session_id ?? tl.session_id,
        model: ev.model ?? tl.model,
        caps: ev.caps ?? tl.caps,
        part_name: ev.part_name ?? tl.part_name,
        offline: ev.offline || tl.offline,
      };
    case "message":
      return {
        ...next,
        steps: [
          ...tl.steps,
          blankStep(`msg-${tl.steps.length}`, "message", null, ev.t, { text: ev.text, role: ev.role, status: "ok", t_end: ev.t }),
        ],
      };
    case "error":
      return {
        ...next,
        steps: [...tl.steps, blankStep(`err-${tl.steps.length}`, "error", null, ev.t, { text: ev.message, status: "error", t_end: ev.t })],
      };
    case "llm_call": {
      // A proposer call belongs to the propose_designs card that is waiting for it.
      const owner = ev.role === "proposer" ? [...tl.steps].reverse().find((s) => s.kind === "propose" && s.status === "running") : undefined;
      return {
        ...next,
        llm: addLlm(tl.llm, ev),
        model: ev.role === "proposer" && ev.model ? ev.model : tl.model,
        steps: owner ? replaceStep(tl, owner.id, (s) => ({ ...s, llm: addLlm(s.llm, ev) })) : tl.steps,
      };
    }
    case "tool_call": {
      const kind = toolKind(ev.tool);
      const calls = tl.tool_calls + 1;
      if (ev.tool === "poll_job") {
        const jobId = jobIdFrom(ev.args);
        const target = jobId ? [...tl.steps].reverse().find((s) => s.job_ids.includes(jobId)) : undefined;
        if (target) {
          return {
            ...next,
            tool_calls: calls,
            aliases: { ...tl.aliases, [ev.call_id]: target.id },
            steps: replaceStep(tl, target.id, (s) => ({ ...s, polls: s.polls + 1 })),
          };
        }
      }
      if (tl.steps.some((s) => s.id === ev.call_id)) return { ...next, tool_calls: calls };
      const jobId = kind === "job" ? jobIdFrom(ev.args) : null;
      return {
        ...next,
        tool_calls: calls,
        steps: [...tl.steps, blankStep(ev.call_id, kind, ev.tool, ev.t, { args: ev.args, job_ids: jobId ? [jobId] : [] })],
      };
    }
    case "tool_result": {
      const alias = tl.aliases[ev.call_id];
      const stepId = alias ?? ev.call_id;
      let steps = tl.steps;
      if (!steps.some((s) => s.id === stepId)) {
        // A result without its call (stream joined late): give it a card of its own.
        steps = [...steps, blankStep(stepId, toolKind(ev.tool), ev.tool, ev.t, {})];
      }
      const owner = steps.find((s) => s.id === stepId);
      const kind = toolKind(ev.tool ?? owner?.tool ?? null);
      const resultObj = objOf(ev.result);
      const job = ev.ok && (kind === "job" || alias || isObj(resultObj.job)) ? readJob(ev.result, ev.t) : null;
      const jobs = job ? { ...tl.jobs, [job.id]: mergeJob(tl.jobs[job.id], job) } : tl.jobs;
      steps = steps.map((s) => {
        if (s.id !== stepId) return s;
        const job_ids = job && !s.job_ids.includes(job.id) ? [...s.job_ids, job.id] : s.job_ids;
        if (alias) return { ...s, job_ids };
        return {
          ...s,
          status: ev.ok ? "ok" : "error",
          t_end: ev.t,
          result: ev.result,
          error: ev.error,
          refused: ev.refused,
          cost_usd: ev.cost_usd,
          job_ids,
        };
      });
      const ledger = kind === "ledger" && ev.ok ? readLedger(ev.result) : tl.ledger;
      const pack = kind === "printpack" && ev.ok ? str(pick(resultObj, "pack_id", "printpack_id", "print_pack_id", "id")) : null;
      return { ...next, steps, jobs, ledger, printpack_id: pack ?? tl.printpack_id };
    }
    case "job_update": {
      const jobs = { ...tl.jobs, [ev.job.id]: mergeJob(tl.jobs[ev.job.id], ev.job) };
      let steps = tl.steps;
      const owner = ev.call_id ? (tl.aliases[ev.call_id] ?? ev.call_id) : null;
      if (owner && steps.some((s) => s.id === owner)) {
        steps = replaceStep(tl, owner, (s) => (s.job_ids.includes(ev.job.id) ? s : { ...s, job_ids: [...s.job_ids, ev.job.id] }));
      } else if (!steps.some((s) => s.job_ids.includes(ev.job.id))) {
        // The agent reports a new job just before its launch call returns: give it to
        // the launch (or stress test) still waiting for a job.
        const waiting = [...steps]
          .reverse()
          .find((s) => (s.kind === "job" || s.kind === "stress") && s.status === "running" && s.job_ids.length === 0);
        if (waiting) steps = replaceStep(tl, waiting.id, (s) => ({ ...s, job_ids: [ev.job.id] }));
      }
      return { ...next, jobs, steps };
    }
    case "ledger":
      return { ...next, ledger: ev.ledger };
    case "session_ended":
      return {
        ...next,
        status: ev.status,
        summary: ev.summary ?? tl.summary,
        answer: ev.answer ?? tl.answer,
        reason: ev.reason ?? tl.reason,
        printpack_id: ev.printpack_id ?? tl.printpack_id,
        ledger: ev.ledger ?? tl.ledger,
        steps: tl.steps.map((s) => (s.status === "running" ? { ...s, status: ev.status === "completed" ? "ok" : "error", t_end: ev.t } : s)),
      };
  }
}

export function buildTimeline(events: AgentEvent[]): Timeline {
  return events.reduce(applyEvent, emptyTimeline());
}

/** Jobs that are queued, provisioning or running. */
export function activeJobs(tl: Timeline): JobState[] {
  return Object.values(tl.jobs).filter((j) => isJobActive(j.status));
}

/**
 * Spend so far: the latest ledger the agent reported, else what the events say (model
 * calls plus job costs or estimates). `estimated` is true when any job cost is an estimate.
 */
export function spend(tl: Timeline): { llm_usd: number; compute_usd: number; total_usd: number; estimated: boolean } {
  const jobs = Object.values(tl.jobs);
  const llm = tl.llm.calls > 0 ? tl.llm.cost_usd : tl.steps.reduce((s, st) => s + (st.kind === "propose" ? (st.cost_usd ?? num(objOf(st.result).cost_usd) ?? 0) : 0), 0);
  const compute = jobs.reduce((s, j) => s + (j.cost_usd ?? 0), 0);
  const l = tl.ledger?.llm_usd ?? llm;
  const c = tl.ledger?.compute_usd ?? compute;
  return {
    llm_usd: l,
    compute_usd: c,
    total_usd: tl.ledger?.total_usd ?? l + c,
    estimated: jobs.some((j) => j.cost_estimated && (j.cost_usd ?? 0) > 0),
  };
}

// ---------------------------------------------------------------- tool views

export interface InspectView {
  name: string | null;
  extents_mm: [number, number, number] | null;
  grip_width_mm: number | null;
  height_mm: number | null;
  volume_cm3: number | null;
  mass_g: number | null;
  watertight: boolean | null;
  convex_pieces: number | null;
  stable_poses: { probability: number | null; label: string | null; width_mm: number | null }[];
  fits_opening: boolean | null;
  opening_mm: number | null;
  conditions: number | null;
  episodes_per_design: number | null;
  pass_threshold: number | null;
  baselines: { name: string; ok: boolean; why: string | null }[];
  notes: string[];
}

function vec3(v: unknown): [number, number, number] | null {
  if (!Array.isArray(v) || v.length !== 3) return null;
  const n = v.map(num);
  return n.every((x) => x !== null) ? (n as [number, number, number]) : null;
}

function strList(v: unknown): string[] {
  return Array.isArray(v) ? v.map(str).filter((s): s is string => s !== null) : typeof v === "string" ? [v] : [];
}

/** inspect_part: the agent returns `{part: PartInfo, pass_threshold_pct, baselines, targets}`. */
export function inspectView(result: unknown): InspectView {
  const outer = objOf(result);
  const o = isObj(outer.part) ? outer.part : outer;
  const poses = pick(o, "stable_poses", "poses");
  const fits = pick(o, "fits_gripper", "fits_opening", "fits");
  const singlePose = num(o.stable_pose_probability);
  const grip = num(pick(o, "grip_width_mm"));
  const conditions = pick(o, "conditions");
  return {
    name: str(o.name),
    extents_mm: vec3(pick(o, "size_mm", "extents_mm", "bbox_mm")),
    grip_width_mm: grip,
    height_mm: num(o.height_mm),
    volume_cm3: num(pick(o, "volume_cm3")) ?? (num(o.volume_mm3) !== null ? num(o.volume_mm3)! / 1000 : null),
    mass_g: num(pick(o, "mass_g")),
    watertight: typeof o.watertight === "boolean" ? o.watertight : null,
    convex_pieces: num(pick(o, "n_convex_pieces", "convex_pieces", "n_convex", "hulls", "n_hulls")),
    stable_poses: Array.isArray(poses)
      ? poses.map((p) => {
          const po = objOf(p);
          return {
            probability: num(pick(po, "probability", "prob", "p")),
            label: str(pick(po, "label", "name")),
            width_mm: num(pick(po, "grasp_width_mm", "width_mm")),
          };
        })
      : num(poses) !== null
        ? Array.from({ length: num(poses)! }, () => ({ probability: null, label: null, width_mm: null }))
        : singlePose !== null
          ? [{ probability: singlePose, label: "Most likely resting pose, used for the trials", width_mm: grip }]
          : [],
    fits_opening: typeof fits === "boolean" ? fits : null,
    opening_mm: num(pick(o, "max_opening_mm", "opening_mm")),
    conditions: Array.isArray(conditions) ? conditions.length : num(conditions),
    episodes_per_design: num(o.episodes_per_design),
    pass_threshold: fraction(outer, ["pass_threshold_pct"], ["pass_threshold"]),
    baselines: (Array.isArray(outer.baselines) ? outer.baselines : []).flatMap((b) => {
      const bo = objOf(b);
      const name = str(pick(bo, "name", "label", "design_id"));
      return name ? [{ name, ok: bo.valid_for_part !== false, why: strList(bo.why_not)[0] ?? null }] : [];
    }),
    notes: strList(pick(o, "notes", "warnings")),
  };
}

export interface ProposedDesign {
  label: string;
  design_id: string | null;
  hypothesis: string | null;
  params: Record<string, number | string>;
  origin: string | null;
}

export interface ProposeView {
  model: string | null;
  analysis: string | null;
  designs: ProposedDesign[];
  tokens_in: number | null;
  tokens_out: number | null;
  latency_s: number | null;
  cost_usd: number | null;
  rejected: number | null;
  /** Why proposals were rejected (near-copies, invalid geometry), one line each. */
  rejections: string[];
}

function params(v: unknown): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const [k, x] of Object.entries(objOf(v))) {
    if (typeof x === "number" && Number.isFinite(x)) out[k] = x;
    else if (typeof x === "string") out[k] = x;
  }
  return out;
}

/** propose_designs: `{designs: [{design_id, hypothesis, params}], rejected: [...], analysis}`; tokens from the step's model calls. */
export function proposeView(result: unknown, step: Pick<Step, "cost_usd" | "llm"> | null = null): ProposeView {
  const o = objOf(result);
  const list = pick(o, "designs", "proposals", "candidates");
  const designs = (Array.isArray(list) ? list : []).map((d, i): ProposedDesign => {
    const dobj = objOf(d);
    const inner = isObj(dobj.design) ? dobj.design : dobj;
    const p = params(isObj(dobj.params) ? dobj.params : inner);
    for (const k of ["name", "rationale", "hypothesis", "label", "design_id"]) delete p[k];
    const id = str(pick(dobj, "design_id", "id"));
    return {
      label: str(pick(dobj, "label", "name")) ?? str(inner.name) ?? id ?? `Design ${i + 1}`,
      design_id: id,
      hypothesis: str(pick(dobj, "hypothesis", "rationale")) ?? str(pick(inner, "rationale")),
      params: p,
      origin: str(pick(dobj, "origin")),
    };
  });
  const usage = objOf(pick(o, "usage", "llm_call"));
  const rejectedRaw = o.rejected;
  const rejections = Array.isArray(rejectedRaw)
    ? rejectedRaw.map((r) => {
        const ro = objOf(r);
        const reasons = strList(ro.reasons);
        return [str(ro.slot), reasons.join("; ") || str(r)].filter(Boolean).join(": ");
      })
    : [];
  const llm = step?.llm ?? null;
  return {
    model: str(pick(o, "model", "model_id")) ?? llm?.model ?? null,
    analysis: str(pick(o, "analysis", "reasoning_summary", "summary")),
    designs,
    tokens_in: num(pick(o, "tokens_in", "prompt_tokens")) ?? num(pick(usage, "prompt_tokens", "tokens_in")) ?? llm?.tokens_in ?? null,
    tokens_out: num(pick(o, "tokens_out", "completion_tokens")) ?? num(pick(usage, "completion_tokens", "tokens_out")) ?? llm?.tokens_out ?? null,
    latency_s: num(pick(o, "latency_s")) ?? num(usage.latency_s) ?? llm?.latency_s ?? null,
    cost_usd: num(pick(o, "cost_usd")) ?? num(usage.cost_usd) ?? step?.cost_usd ?? llm?.cost_usd ?? null,
    rejected: Array.isArray(rejectedRaw) ? rejectedRaw.length : num(pick(o, "rejected", "n_rejected")),
    rejections,
  };
}

export interface ResultRow {
  label: string;
  design_id: string | null;
  /** Training success, 0..1. */
  success: number | null;
  /** The agent's physics verdict against its pass threshold. */
  verdict: "pass" | "fail" | null;
  episodes: number | null;
  failure_modes: Record<string, number>;
  /** Replay stem relative to /data, when the session published one. */
  replay: string | null;
}

export interface ResultsView {
  job_id: string | null;
  run_id: string | null;
  pass_threshold: number | null;
  rows: ResultRow[];
  best: ResultRow | null;
}

/** fetch_results: `{run_id, job_id, pass_threshold_pct, best_design_id, designs: [{design_id, label, success_pct, verdict, failure_modes, episodes}]}`. */
export function resultsView(result: unknown): ResultsView {
  const o = objOf(result);
  const list = pick(o, "designs", "results", "scores");
  const rows = (Array.isArray(list) ? list : []).map((r, i): ResultRow => {
    const ro = objOf(r);
    const fm: Record<string, number> = {};
    for (const [k, v] of Object.entries(objOf(pick(ro, "failure_modes", "failures")))) {
      const n = num(v);
      if (n !== null) fm[k] = n;
    }
    const verdict = str(ro.verdict);
    return {
      label: str(pick(ro, "label", "name")) ?? str(ro.design_id) ?? `Design ${i + 1}`,
      design_id: str(pick(ro, "design_id", "id")),
      success: fraction(ro, ["success_pct"], ["success", "success_rate", "score"]),
      verdict: verdict === "pass" || verdict === "fail" ? verdict : null,
      episodes: num(pick(ro, "episodes", "n_episodes")),
      failure_modes: fm,
      replay: str(pick(ro, "replay")),
    };
  });
  const bestId = str(o.best_design_id);
  const best =
    (bestId ? rows.find((r) => r.design_id === bestId) : undefined) ??
    rows.reduce<ResultRow | null>((b, r) => ((r.success ?? -1) > (b?.success ?? -1) ? r : b), null);
  return {
    job_id: str(pick(o, "job_id")),
    run_id: str(pick(o, "run_id")),
    pass_threshold: fraction(o, ["pass_threshold_pct"], ["pass_threshold"]),
    rows,
    best,
  };
}

export interface StressView {
  design_id: string | null;
  label: string | null;
  score: number | null;
  verdict: "pass" | "fail" | null;
  passed: string | null;
  tolerance_mm: { x: number | null; y: number | null };
  categories: { placement: number | null; shake: number | null; push: number | null };
  failed: { test: string; failure_mode: string | null }[];
  certificate_id: string | null;
}

/** stress_test: `{design_id, score_pct, verdict, category_pct, placement_tolerance_mm, passed, failed_tests}`. */
export function stressView(result: unknown): StressView {
  const o = objOf(result);
  const cert = isObj(o.certificate) ? o.certificate : o;
  const tol = objOf(pick(cert, "placement_tolerance_mm", "tolerance_mm", "tolerance"));
  const pct = objOf(pick(cert, "category_pct"));
  const cats = objOf(pick(cert, "category_scores", "categories"));
  const cat = (k: "placement" | "shake" | "push") => (num(pct[k]) !== null ? num(pct[k])! / 100 : num(cats[k]));
  const verdict = str(pick(cert, "verdict")) ?? str(o.verdict);
  return {
    design_id: str(pick(o, "design_id")) ?? str(pick(cert, "design_id")),
    label: str(pick(o, "label", "name")),
    score: fraction(cert, ["score_pct"], ["score"]),
    verdict: verdict === "pass" || verdict === "fail" ? verdict : null,
    passed: str(pick(cert, "passed")),
    tolerance_mm: { x: num(tol.x), y: num(tol.y) },
    categories: { placement: cat("placement"), shake: cat("shake"), push: cat("push") },
    failed: (Array.isArray(cert.failed_tests) ? cert.failed_tests : []).flatMap((f) => {
      const fo = objOf(f);
      const test = str(pick(fo, "test", "name"));
      return test ? [{ test, failure_mode: str(fo.failure_mode) }] : [];
    }),
    certificate_id: str(pick(o, "certificate_id")) ?? str(pick(cert, "certificate_id")),
  };
}

export interface PrintPackFile {
  name: string;
  /** Path relative to /data for recorded sessions; null when only the API serves it. */
  path: string | null;
  bytes: number | null;
}

export interface PrintPackView {
  printpack_id: string | null;
  design_id: string | null;
  verdict: "pass" | "fail" | null;
  certificate_included: boolean | null;
  files: PrintPackFile[];
  zip_bytes: number | null;
  /** API path of the zip, e.g. "/printpack/pack-...". */
  url: string | null;
}

/** The only download path the API serves for a print pack (src/rtt/api/app.py). */
const PRINTPACK_PATH = /^\/printpack\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

/**
 * A data-relative file path that stays inside /data: no scheme, no leading slash, no
 * `..` segment, no backslash. Anything else is dropped (the file is then listed
 * without a link).
 */
export function safeDataPath(p: string | null): string | null {
  if (p === null) return null;
  if (/^[a-z][a-z0-9+.-]*:/i.test(p) || p.startsWith("/") || p.includes("\\") || p.split("/").includes("..")) return null;
  return p;
}

/** export_print_pack: `{pack_id, design_id, verdict, certificate_included, files: [{name, bytes}], zip_bytes, download}`. */
export function printpackView(result: unknown): PrintPackView {
  const o = objOf(result);
  const files = pick(o, "files", "contents");
  const verdict = str(o.verdict);
  // The page appends `url` to the API origin, so only the API's own path shape is
  // kept: "@other.host/x" or "//x" appended to "http://127.0.0.1:8000" would leave it.
  const url = str(pick(o, "download", "url", "download_url"));
  return {
    printpack_id: str(pick(o, "pack_id", "printpack_id", "print_pack_id", "id")),
    design_id: str(o.design_id),
    verdict: verdict === "pass" || verdict === "fail" ? verdict : null,
    certificate_included: typeof o.certificate_included === "boolean" ? o.certificate_included : null,
    files: (Array.isArray(files) ? files : []).flatMap((f): PrintPackFile[] => {
      if (typeof f === "string") return [{ name: f.split("/").pop() ?? f, path: null, bytes: null }];
      const fo = objOf(f);
      const name = str(pick(fo, "name", "filename"));
      return name ? [{ name, path: safeDataPath(str(pick(fo, "path"))), bytes: num(pick(fo, "bytes", "size")) }] : [];
    }),
    zip_bytes: num(o.zip_bytes),
    url: url !== null && PRINTPACK_PATH.test(url) ? url : null,
  };
}

// ---------------------------------------------------------------- recorded sessions

export interface RecordedSession {
  id: string;
  /** `recorded`: a real agent run. `illustrative`: layout data, stamped as such on the page. */
  kind: "recorded" | "illustrative";
  /** A scripted run with no model calls (`python -m rtt.agent run --offline`). */
  offline: boolean;
  title: string;
  note: string;
  recorded_at: string | null;
  part: { name: string; stl: string | null; units: LengthUnit };
  model: string | null;
  events: AgentEvent[];
  /** Raw entries that could not be read. */
  dropped: number;
}

export class SessionError extends Error {
  constructor(file: string, detail: string) {
    super(`${file}: ${detail}`);
    this.name = "SessionError";
  }
}

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;
const UNITS: readonly LengthUnit[] = ["mm", "cm", "m", "in"];

/**
 * Read a session file: the agent's own (`format: "rtt-agent-session"`, as written by
 * src/rtt/agent/recorder.py and published by scripts/import-session.mjs), a bare
 * `{events: [...]}`, or a chat-style `{messages: [...]}` transcript (used only when
 * there are no events). The id comes from `id`/`session_id`, else the file name.
 * Throws SessionError naming the file.
 */
export function parseRecordedSession(raw: unknown, file = "session.json"): RecordedSession {
  if (!isObj(raw)) throw new SessionError(file, "expected a JSON object");
  const format = pick(raw, "format", "schema");
  if (format !== undefined && format !== "rtt-agent-session") {
    throw new SessionError(file, `unknown format ${JSON.stringify(format)}`);
  }
  if (raw.version !== undefined && raw.version !== 1) throw new SessionError(file, `unsupported version ${String(raw.version)}`);
  const fallbackId = file.replace(/^.*\//, "").replace(/\.json$/, "").toLowerCase();
  const id = str(pick(raw, "id", "session_id")) ?? fallbackId;
  if (!SLUG.test(id)) throw new SessionError(file, `id ${JSON.stringify(id)} is not URL-safe`);
  const rawEvents =
    Array.isArray(raw.events) && raw.events.length > 0
      ? raw.events
      : Array.isArray(raw.messages)
        ? eventsFromChatMessages(raw.messages)
        : Array.isArray(raw.events)
          ? raw.events
          : null;
  if (!rawEvents) throw new SessionError(file, "expected an `events` (or `messages`) array");
  const { events, dropped } = normalizeEvents(rawEvents);
  if (events.length === 0) throw new SessionError(file, "no readable events");
  const part = objOf(raw.part);
  const stl = str(pick(part, "stl", "path"));
  if (stl !== null && (stl.startsWith("/") || stl.includes("..") || /^[a-z]+:/i.test(stl))) {
    throw new SessionError(file, `part.stl must be a relative data path, got ${JSON.stringify(stl)}`);
  }
  const unit = str(part.units) as LengthUnit | null;
  const kind = raw.kind === "illustrative" ? "illustrative" : "recorded";
  const started = events.find((e): e is Extract<AgentEvent, { type: "session_started" }> => e.type === "session_started");
  const name = str(pick(part, "name", "filename")) ?? started?.part_name ?? "part";
  return {
    id,
    kind,
    offline: raw.offline === true || started?.offline === true,
    title: str(raw.title) ?? name,
    note: str(raw.note) ?? "",
    recorded_at: str(pick(raw, "recorded_at", "created_at")),
    part: { name, stl, units: unit && UNITS.includes(unit) ? unit : "mm" },
    model: str(raw.model) ?? buildTimeline(events).model,
    events,
    dropped,
  };
}

/** What the /part page lists for each recorded session (the events are fetched on demand). */
export interface SessionSummary {
  id: string;
  kind: RecordedSession["kind"];
  offline: boolean;
  title: string;
  note: string;
  recorded_at: string | null;
  part: RecordedSession["part"];
  model: string | null;
  tool_calls: number;
  duration_s: number;
}

export function summarizeSession(s: RecordedSession): SessionSummary {
  const tl = buildTimeline(s.events);
  return {
    id: s.id,
    kind: s.kind,
    offline: s.offline,
    title: s.title,
    note: s.note,
    recorded_at: s.recorded_at,
    part: s.part,
    model: s.model,
    tool_calls: tl.tool_calls,
    duration_s: tl.t_last,
  };
}

// ---------------------------------------------------------------- playback

/**
 * Playback times for recorded events: real gaps are kept up to `maxGap` seconds and
 * longer waits (a Nebius job running for minutes) are cut to `maxGap`, so a session
 * replays in about a minute while keeping its rhythm. Never shorter than `minGap`
 * between distinct events, except for events recorded at the same instant.
 */
export function playbackTimes(events: AgentEvent[], maxGap = 2.4, minGap = 0.25): number[] {
  const out: number[] = [];
  let clock = 0;
  events.forEach((e, i) => {
    if (i > 0) {
      const gap = e.t - events[i - 1]!.t;
      clock += gap <= 0 ? 0 : Math.min(Math.max(gap, minGap), maxGap);
    }
    out.push(clock);
  });
  return out;
}

/** How many events have played by playback time `clock`. */
export function playedCount(times: number[], clock: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid]! <= clock) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
