/**
 * Client of the hosted runner: plan a description with Nemotron and test the plan in physics on
 * a hosted runner. The runner is optional. It is on only when NEXT_PUBLIC_RUNNER_URL is set at
 * build time; the public site is otherwise built without it and reads exactly as it did before.
 *
 *   GET  /health        -> {ok, kill_switch, open, auto_off, plans_left_today, runs_left_today, spend_today_usd, spend_total_usd}
 *                          (`open` is false once the runner has switched itself off; an older runner has no such field and
 *                          still answers 200 after its auto-off date, so the page also reads `auto_off`: see `offReason`)
 *   POST /plan          {description, provider, key} -> {status, plan, problems, calls, cost_usd, model}
 *   POST /runs          {plan} -> {run_id, status: "queued", eta_s}
 *   GET  /runs/{id}     -> {status, created_at, cost_estimate_usd, error, result}
 *
 * Errors (bodies are `{"error": "..."}`): 400 (a /runs 400 also has `problems`), 401 (the provider rejected
 * the key), 429 ({cap, limit, resets_at}; resets_at is null for a cap that never resets; a 429 with no `cap` is the
 * hosting layer throttling, so the runner is busy), 503 (the runner is off).
 * docs/RUNNER.md describes them; the summary's tools may be plain names.
 *
 * The key goes only in the JSON body of POST /plan, over HTTPS. It is never put in a URL, a header,
 * storage or a log, and no message in this file repeats it.
 */

import { countOf } from "./format";
import { isObj, num, pick, str } from "./session";

export type Provider = "nvidia" | "tokenfactory" | "free";

/** The plain privacy line, shown next to the key field and in the docs. */
export const KEY_PRIVACY_LINE = "Your key is sent once over HTTPS for this request and is never stored or logged.";

/** What the daily spend cap is, in the words shown beside the Run button. */
export const RUN_CAP_LINE = "Runs on Nebius Serverless Jobs, capped at $2 a day";

/*
 * The counts /health gives are for the whole site, not for this visitor: the free allowance is shared by everyone, and the
 * runs left are how many the day's compute budget can still reserve. The labels say so. Each visitor also has limits of their
 * own, lower ones, which the page reports in words when one is reached (`capSentence`), so a count above zero is not a promise.
 */

/** The free allowance's label, with the plans left across the site today when the runner said how many. */
export function freeAllowanceLabel(plansLeft: number | null): string {
  return plansLeft === null ? "Use the free allowance" : `Use the free allowance (${countOf(plansLeft, "free plan")} left today across the site)`;
}

/** The sentence after the run cap line, led by a space; empty when the runner gave no count. */
export function runsLeftLine(runsLeft: number | null): string {
  return runsLeft === null ? "" : ` ${countOf(runsLeft, "run")} left today across the site.`;
}

/** The longest description POST /plan accepts. */
export const MAX_DESCRIPTION_CHARS = 4000;

const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]"]);

/**
 * The runner origin from NEXT_PUBLIC_RUNNER_URL (inlined at build time), without a trailing slash;
 * null when it is unset or unusable. A key must not travel in clear text, so plain http is accepted
 * only for a loopback host (a runner on your own machine, or the local mock used by the smoke test).
 */
export function runnerBase(raw: string | undefined = process.env.NEXT_PUBLIC_RUNNER_URL): string | null {
  const v = raw?.trim();
  if (!v) return null;
  try {
    const url = new URL(v);
    const local = LOOPBACK.has(url.hostname) || url.hostname.endsWith(".localhost");
    if (!(url.protocol === "https:" || (url.protocol === "http:" && local))) return null;
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
}

/** Whether this build offers the hosted runner (docs, home and Studio all gate on this one flag). */
export function runnerConfigured(): boolean {
  return runnerBase() !== null;
}

// ---------------------------------------------------------------- the run id in the URL hash

const RUN_HASH = /^#run-([A-Za-z0-9_-]{1,80})$/;

/** The run id a URL hash names (`#run-<id>`), or null. */
export function runIdFromHash(hash: string): string | null {
  return RUN_HASH.exec(hash)?.[1] ?? null;
}

/** The hash that names a run: a bare token, no path and no query. */
export function hashForRun(id: string): string {
  return `#run-${id}`;
}

// ---------------------------------------------------------------- errors

export type RunnerErrorKind =
  | "key_missing"
  | "key_rejected"
  | "cap"
  | "off"
  | "bad_request"
  | "not_found"
  | "busy"
  | "network"
  | "server"
  | "bad_reply";

/** A failed call, with the plain message the page shows. The message never holds a raw status code. */
export class RunnerError extends Error {
  constructor(
    readonly kind: RunnerErrorKind,
    message: string,
    /** The cap that was reached (429), in the server's words. */
    readonly cap: string | null = null,
    /** When the cap resets (an ISO time), when the server said. */
    readonly resetsAt: string | null = null,
    /** The checks a refused plan failed (a 400 on POST /runs lists them). */
    readonly problems: string[] = [],
  ) {
    super(message);
    this.name = "RunnerError";
  }

  /** Whether trying the same call again can help. */
  get retryable(): boolean {
    return this.kind === "network" || this.kind === "server" || this.kind === "busy";
  }
}

/** What each of the runner's caps says when it is reached, in plain words (the names are the runner's own). */
const CAP_TEXT: Record<string, string> = {
  free_plans_per_day: "The free allowance for the whole site is used up for today.",
  free_plans_per_ip_per_day: "You have used your free plans for today.",
  keyed_plans_per_ip_per_day: "You have reached the daily limit on plans with your own key.",
  runs_per_ip_per_day: "You have reached the daily limit on physics runs.",
  spend_per_day_usd: "The hosted runner's compute budget for today is used up.",
  spend_total_usd: "The hosted runner's total compute budget is used up.",
};

/** A cap reached, in plain words; a name this page does not know is shown as words ("some_cap" reads "The some cap limit was reached."). */
export function capSentence(cap: string): string {
  const known = CAP_TEXT[cap.trim().toLowerCase()];
  if (known) return known;
  return `The ${cap.trim().replace(/[_-]+/g, " ").toLowerCase().slice(0, 60)} limit was reached.`;
}

/** An ISO time as "29 October 2026, 00:00 UTC" (fixed zone and locale, so it reads the same everywhere); null when unreadable. */
export function formatReset(iso: string | null): string | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return null;
  const d = new Date(ms);
  const date = d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: "UTC" });
  return `${date}, ${time} UTC`;
}

export const MESSAGES = {
  key_missing: "Paste your key first, or use the free allowance.",
  key_rejected: "The provider rejected that key.",
  off: "The hosted runner is off. Run it on your own machine:",
  bad_request: "The runner could not use that request. Check the description and the key, then try again.",
  plan_checks: "The runner checked the plan again and found problems:",
  not_found: "The runner has no record of that run. It may be too old, or the link is wrong.",
  busy: "The runner is busy. Try again in a minute.",
  network: "Could not reach the hosted runner. Check your connection and try again.",
  server: "The hosted runner had a problem. Try again in a minute.",
  bad_reply: "The hosted runner sent an answer this page could not read.",
} as const;

/** Which cap was reached and when it resets. `resets_at: null` is how the runner says a cap never resets. */
function capMessage(cap: string, resetsAt: string | null): string {
  const when = formatReset(resetsAt);
  return when ? `${capSentence(cap)} It resets at ${when}.` : `${capSentence(cap)} It does not reset.`;
}

/** The JSON body of an error response, or {} (a body that is not JSON is dropped). */
async function errorBody(res: Response): Promise<Record<string, unknown>> {
  try {
    const body: unknown = await res.json();
    return isObj(body) ? body : {};
  } catch {
    return {};
  }
}

/**
 * An error response as a RunnerError. The runner's bodies are `{"error": ...}`; a 429 also names the
 * cap and when it resets, and a 400 on POST /runs lists the problems. A 429 that names no cap did not come
 * from the runner's own caps but from the hosting layer throttling a burst (a Function URL with reserved
 * concurrency answers that way), so it reads "busy", not "a daily limit". Nothing else is read, and the
 * rest of the body is dropped so the connection is released (an unread body keeps the request open).
 */
export async function errorFromResponse(res: Response): Promise<RunnerError> {
  const reads = res.status === 429 || res.status === 400;
  if (!reads) void res.body?.cancel().catch(() => {});
  const body = reads ? await errorBody(res) : {};
  switch (res.status) {
    case 401:
      return new RunnerError("key_rejected", MESSAGES.key_rejected);
    case 429: {
      const o = isObj(body.detail) ? body.detail : isObj(body.error) ? body.error : body;
      const cap = str(o.cap);
      if (!cap) return new RunnerError("busy", MESSAGES.busy);
      const resetsAt = str(o.resets_at);
      return new RunnerError("cap", capMessage(cap, resetsAt), cap, resetsAt);
    }
    case 503:
      return new RunnerError("off", MESSAGES.off);
    case 404:
      return new RunnerError("not_found", MESSAGES.not_found);
    case 400: {
      const problems = strings(body.problems);
      return new RunnerError("bad_request", problems.length ? MESSAGES.plan_checks : MESSAGES.bad_request, null, null, problems);
    }
    case 413:
    case 422:
      return new RunnerError("bad_request", MESSAGES.bad_request);
    default:
      return new RunnerError("server", MESSAGES.server);
  }
}

// ---------------------------------------------------------------- the replies

export interface RunnerHealth {
  ok: boolean;
  killSwitch: boolean;
  /** Whether the runner is taking requests: false once it has switched itself off; null from a runner that has no such field. */
  open: boolean | null;
  /** The date the runner switches itself off (YYYY-MM-DD), when it says. */
  autoOff: string | null;
  plansLeft: number | null;
  runsLeft: number | null;
  spendTodayUsd: number | null;
  spendTotalUsd: number | null;
}

export interface PlanReply {
  status: "valid" | "failed" | "refused";
  /** The TaskSpec document, as the Studio's recorded plans carry it. */
  plan: Record<string, unknown> | null;
  problems: string[];
  calls: number;
  costUsd: number | null;
  model: string | null;
}

export interface RunTicket {
  runId: string;
  etaS: number | null;
}

export interface Completion {
  k: number;
  n: number;
}

/** A setup as the summary names it. Tools come as plain names, or as {label, family}. */
export interface SetupBrief {
  label: string;
  kind: string;
  tools: { label: string; family: string }[];
}

export interface StepRow {
  step: string;
  /** The step in words (the runner sends the step id; "open_lid" reads "Open lid"). */
  label: string;
  /** The tool the setup uses for the step; null when none does. */
  tool: string | null;
  /** 0 to 1; null when the run did not measure it. */
  successRate: number | null;
}

export interface RunSummary {
  /** The recommended setup; null when no setup completed the task (then `closest` may name the nearest). */
  pick: SetupBrief | null;
  closest: SetupBrief | null;
  /** How often the pick (or the closest setup) completed the task; null when there is nothing to count. */
  completion: Completion | null;
  why: string;
  /** The best single tool the recommender compared against. */
  baseline: { label: string; completion: Completion } | null;
  /** Tools ruled out before the full runs, of how many were considered (null when the runner did not say). */
  ruledOut: { pruned: number; candidates: number | null } | null;
  steps: StepRow[];
  codeVersion: string | null;
  reportUrl: string | null;
}

export type RunStatus = "queued" | "running" | "done" | "failed";

export interface RunRecord {
  status: RunStatus;
  createdAt: string | null;
  costEstimateUsd: number | null;
  error: string | null;
  result: RunSummary | null;
}

function bad(): RunnerError {
  return new RunnerError("bad_reply", MESSAGES.bad_reply);
}

export function parseHealth(body: unknown): RunnerHealth {
  const o = isObj(body) ? body : {};
  return {
    ok: o.ok === true,
    killSwitch: o.kill_switch === true,
    open: typeof o.open === "boolean" ? o.open : null,
    autoOff: str(o.auto_off),
    plansLeft: num(o.plans_left_today),
    runsLeft: num(o.runs_left_today),
    spendTodayUsd: num(o.spend_today_usd),
    spendTotalUsd: num(o.spend_total_usd),
  };
}

// ---------------------------------------------------------------- when the runner is off

export type OffReason = "kill_switch" | "closed" | "auto_off";

/** The calendar day (YYYY-MM-DD) an ISO date or date-time names, or null when it is not a date. */
function dayOf(iso: string | null): string | null {
  const day = iso ? /^(\d{4}-\d{2}-\d{2})/.exec(iso.trim())?.[1] : undefined;
  return day && Number.isFinite(Date.parse(`${day}T00:00:00Z`)) ? day : null;
}

/** Whether the auto-off date is before today in UTC. The runner serves through that date, so on the day itself it is not past. */
export function autoOffPassed(autoOff: string | null, nowMs: number = Date.now()): boolean {
  const day = dayOf(autoOff);
  return day !== null && day < new Date(nowMs).toISOString().slice(0, 10);
}

/**
 * Why the panel should say the runner is off, or null when it is on: the kill switch is on, the runner says
 * `open: false`, or the runner has no `open` field (an older one, which still answers 200 after its auto-off
 * date) and the auto-off date has passed. When `open` is there it is the runner's own word and wins over the date.
 */
export function offReason(health: RunnerHealth, nowMs: number = Date.now()): OffReason | null {
  if (health.killSwitch) return "kill_switch";
  if (health.open === false) return "closed";
  if (health.open === null && autoOffPassed(health.autoOff, nowMs)) return "auto_off";
  return null;
}

function strings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

export function parsePlanReply(body: unknown): PlanReply {
  if (!isObj(body)) throw bad();
  const status = body.status;
  if (status !== "valid" && status !== "failed" && status !== "refused") throw bad();
  const plan = isObj(body.plan) ? body.plan : null;
  if (status === "valid" && !plan) throw bad();
  return {
    status,
    plan,
    problems: strings(body.problems),
    calls: num(body.calls) ?? 0,
    costUsd: num(body.cost_usd),
    model: str(body.model),
  };
}

export function parseTicket(body: unknown): RunTicket {
  if (!isObj(body)) throw bad();
  const id = str(body.run_id);
  if (!id || !/^[A-Za-z0-9_-]{1,80}$/.test(id)) throw bad();
  return { runId: id, etaS: num(body.eta_s) };
}

function completion(v: unknown): Completion | null {
  if (!isObj(v)) return null;
  const k = num(v.k);
  const n = num(v.n);
  return k === null || n === null ? null : { k, n };
}

/** "open_lid" as "Open lid". */
function stepWords(id: string): string {
  const s = id.replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function tools(v: unknown): SetupBrief["tools"] {
  if (!Array.isArray(v)) return [];
  return v.flatMap((t) => {
    if (typeof t === "string") return t ? [{ label: t, family: "" }] : [];
    if (isObj(t) && str(t.label ?? t.display_name)) return [{ label: str(t.label ?? t.display_name)!, family: str(t.family ?? t.family_title) ?? "" }];
    return [];
  });
}

function setupBrief(v: unknown): SetupBrief | null {
  if (!isObj(v)) return null;
  const label = str(v.label);
  if (!label) return null;
  return { label, kind: str(v.kind) ?? "single", tools: tools(v.tools) };
}

/** `ruled_out` is an object of counts (candidates, pruned, ...), or a bare number of tools. */
function ruledOut(v: unknown): RunSummary["ruledOut"] {
  if (typeof v === "number" && Number.isFinite(v)) return { pruned: v, candidates: null };
  if (!isObj(v)) return null;
  const pruned = num(v.pruned);
  return pruned === null ? null : { pruned, candidates: num(v.candidates) };
}

/** The run's result. A run with no recommendation is still a result: `pick` is null and `why` says why. */
export function parseSummary(v: unknown): RunSummary | null {
  if (!isObj(v)) return null;
  const base = isObj(v.baseline) ? v.baseline : null;
  const baseDone = base ? completion(base.completion) : null;
  const steps = Array.isArray(v.steps)
    ? v.steps.flatMap((s): StepRow[] => {
        if (!isObj(s)) return [];
        const id = str(s.step);
        if (!id) return [];
        return [{ step: id, label: str(s.label) ?? stepWords(id), tool: str(s.tool), successRate: num(s.success_rate) }];
      })
    : [];
  return {
    pick: setupBrief(v.pick),
    closest: setupBrief(v.closest),
    completion: completion(v.completion),
    why: str(v.why) ?? "",
    baseline: base && baseDone && str(base.label) ? { label: str(base.label)!, completion: baseDone } : null,
    ruledOut: ruledOut(v.ruled_out),
    steps,
    codeVersion: str(v.code_version),
    reportUrl: str(v.report_url),
  };
}

export function parseRun(body: unknown): RunRecord {
  if (!isObj(body)) throw bad();
  const status = body.status;
  if (status !== "queued" && status !== "running" && status !== "done" && status !== "failed") throw bad();
  const result = parseSummary(pick(body, "result"));
  if (status === "done" && !result) throw bad();
  return {
    status,
    createdAt: str(body.created_at),
    costEstimateUsd: num(body.cost_estimate_usd),
    error: str(body.error),
    result,
  };
}

/**
 * The link to a run's full report. The summary's `report_url` is a presigned link that expires after
 * 15 minutes, so a card left open would go stale; the runner's `GET /runs/{id}/report` redirects to a fresh
 * one, so that is what the link points to. Null when the run gave no report, or gave one that is not http(s).
 */
export function reportLink(base: string, runId: string, reportUrl: string | null): string | null {
  if (!reportUrl) return null;
  try {
    const given = new URL(reportUrl, `${base}/`);
    if (given.protocol !== "https:" && given.protocol !== "http:") return null;
  } catch {
    return null;
  }
  return `${base}/runs/${encodeURIComponent(runId)}/report`;
}

// ---------------------------------------------------------------- the calls

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

async function call(base: string, path: string, init: RequestInit, fetchImpl?: FetchLike): Promise<unknown> {
  const doFetch: FetchLike = fetchImpl ?? ((input, i) => fetch(input, i));
  let res: Response;
  try {
    // redirect "error": the runner's API never redirects, and a redirect of POST /plan (a 307 or 308 repeats the body)
    // would carry the key to another address, so any redirect fails the call as a network error.
    res = await doFetch(`${base}${path}`, { ...init, cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer", redirect: "error" });
  } catch (err) {
    if (init.signal?.aborted) throw err;
    throw new RunnerError("network", MESSAGES.network);
  }
  if (!res.ok) throw await errorFromResponse(res);
  try {
    return await res.json();
  } catch {
    throw bad();
  }
}

export async function getHealth(base: string, signal?: AbortSignal, fetchImpl?: FetchLike): Promise<RunnerHealth> {
  return parseHealth(await call(base, "/health", { method: "GET", signal }, fetchImpl));
}

/** Plan a description. The key goes in the JSON body, once; `key` is null for the free allowance. */
export async function postPlan(
  base: string,
  request: { description: string; provider: Provider; key: string | null },
  signal?: AbortSignal,
  fetchImpl?: FetchLike,
): Promise<PlanReply> {
  const body = JSON.stringify({ description: request.description, provider: request.provider, key: request.key });
  return parsePlanReply(await call(base, "/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body, signal }, fetchImpl));
}

export async function postRun(base: string, plan: Record<string, unknown>, signal?: AbortSignal, fetchImpl?: FetchLike): Promise<RunTicket> {
  const body = JSON.stringify({ plan });
  return parseTicket(await call(base, "/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body, signal }, fetchImpl));
}

export async function getRun(base: string, id: string, signal?: AbortSignal, fetchImpl?: FetchLike): Promise<RunRecord> {
  return parseRun(await call(base, `/runs/${encodeURIComponent(id)}`, { method: "GET", signal }, fetchImpl));
}

/** Text from the server with the key struck out, in case a message ever echoes it. */
export function redact(text: string, key: string): string {
  const k = key.trim();
  return k.length >= 4 ? text.split(k).join("[key]") : text;
}
