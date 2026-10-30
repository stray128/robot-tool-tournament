/**
 * The hosted runner's state machine: the gate, the providers, what happens to the key, the plan
 * hand-off, the run lifecycle, each error, the hash resume and the base path. fetch is scripted and
 * every place a key could leak (storage, history, the console, the URL) is a spy.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { browserHash, type Fault, PLAN_TIMEOUT_MS, POLL_MS, RunnerController, SUBMIT_TIMEOUT_MS } from "@/components/studio/runner/controller";
import { MESSAGES, type PlanReply, type Provider } from "@/lib/runner";

import { BASE, HEALTH, json, leakSpies, leaked, networkError, REFUSAL_PLAN, scriptedFetch, SUMMARY, VALID_PLAN } from "./helpers/runner_mock";

const KEY = "nvapi-SECRET-KEY-0123456789";
const DESCRIPTION = "Open the lid of the tote, put the carton in it, close the lid.";

/** What the runner's 503s say (src/rtt/runner/app.py). Only the first reads as the runner being off. */
const OFF_WORDS = "the runner is switched off for now; recorded runs still replay";
const FREE_TIER_DOWN = "the free tier is unavailable right now; try your own key";
const JOB_NOT_STARTED = "the simulation job could not be started; try again later";

const validReply = { status: "valid", plan: VALID_PLAN, problems: [], calls: 1, cost_usd: 0.01, model: "nemotron-mock" };
const queued = { run_id: "r1", status: "queued", eta_s: 480 };
const rec = (status: string, extra: Record<string, unknown> = {}) => ({
  status,
  created_at: "2026-10-28T10:00:00Z",
  cost_estimate_usd: 0.18,
  error: null,
  result: null,
  ...extra,
});

let spies: ReturnType<typeof leakSpies>;
beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-10-28T10:00:05Z"));
  spies = leakSpies();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function make(routes: Parameters<typeof scriptedFetch>[0], opts: { onPlan?: (r: PlanReply) => void; hash?: string } = {}) {
  const fetchImpl = scriptedFetch({ "GET /health": json(HEALTH), ...routes });
  if (opts.hash !== undefined) spies.win.location.hash = opts.hash;
  const c = new RunnerController({ base: BASE, fetchImpl, onPlan: opts.onPlan });
  return { c, fetchImpl };
}

const bodyOf = (call: unknown[]): Record<string, unknown> => JSON.parse(String((call[1] as RequestInit).body));
const callsTo = (f: ReturnType<typeof scriptedFetch>, method: string, path: string) =>
  f.mock.calls.filter((c) => c[0] === `${BASE}${path}` && ((c[1] as RequestInit | undefined)?.method ?? "GET") === method);

describe("the gate: the panel shows only when /health answers ok", () => {
  it("is checking before the answer, then ready with the numbers", async () => {
    const { c } = make({});
    expect(c.getSnapshot().gate).toBe("checking");
    await c.start();
    const s = c.getSnapshot();
    expect(s.gate).toBe("ready");
    expect(s.health).toMatchObject({ plansLeft: 3, runsLeft: 9, autoOff: "2026-12-15" });
  });

  it("is hidden when /health cannot be reached", async () => {
    const { c } = make({ "GET /health": [networkError()] });
    await c.start();
    expect(c.getSnapshot().gate).toBe("hidden");
  });

  it("is hidden when /health says it is not ok, or answers with an error", async () => {
    const a = make({ "GET /health": json({ ...HEALTH, ok: false }) });
    await a.c.start();
    expect(a.c.getSnapshot().gate).toBe("hidden");
    const b = make({ "GET /health": json({ detail: "off" }, 503) });
    await b.c.start();
    expect(b.c.getSnapshot().gate).toBe("hidden");
    const d = make({ "GET /health": new Response("<html>", { status: 200 }) });
    await d.c.start();
    expect(d.c.getSnapshot().gate).toBe("hidden");
  });

  it("is hidden when /health hangs", async () => {
    const hang = vi.fn((_u: string, init?: RequestInit) => new Promise<Response>((_res, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")))));
    const c = new RunnerController({ base: BASE, fetchImpl: hang });
    const started = c.start();
    await vi.advanceTimersByTimeAsync(8_000);
    await started;
    expect(c.getSnapshot().gate).toBe("hidden");
  });

  it("shows the off notice when the kill switch is on, and never plans or runs", async () => {
    const { c, fetchImpl } = make({ "GET /health": json({ ...HEALTH, kill_switch: true }) }, { hash: "#run-abc" });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "off", offReason: "kill_switch" });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});

describe("the gate after the auto-off date: /health still answers 200 with ok: true", () => {
  const { open: _open, ...OLDER } = HEALTH; // a runner with no `open` field

  it("shows the off notice when the runner says open is false, and never plans, runs or resumes", async () => {
    const { c, fetchImpl } = make({ "GET /health": json({ ...HEALTH, open: false }) }, { hash: "#run-abc" });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "off", offReason: "closed", health: { ok: true, open: false } });
    // one call, /health: a run named in the address is not resumed
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(c.getSnapshot().runPhase).toBe("idle");
  });

  it("is off for open: false even when the auto-off date is still ahead and the kill switch is off", async () => {
    const { c } = make({ "GET /health": json({ ...HEALTH, open: false, kill_switch: false, auto_off: "2099-01-01" }) });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "off", offReason: "closed" });
  });

  it("is ready for open: true, even when the date has passed (the runner's word wins)", async () => {
    const { c } = make({ "GET /health": json({ ...HEALTH, open: true, auto_off: "2026-01-01" }) });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "ready", offReason: null });
  });

  it("with no open field, is off once the auto-off date is before today in UTC", async () => {
    const { c, fetchImpl } = make({ "GET /health": json({ ...OLDER, auto_off: "2026-10-27" }) }, { hash: "#run-abc" });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "off", offReason: "auto_off", health: { open: null, autoOff: "2026-10-27" } });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("with no open field, is ready through the auto-off day itself, and before it", async () => {
    // the clock in these tests is 2026-10-28T10:00:05Z
    for (const day of ["2026-10-28", "2026-10-29", "2026-12-15"]) {
      const { c } = make({ "GET /health": json({ ...OLDER, auto_off: day }) });
      await c.start();
      expect(c.getSnapshot(), day).toMatchObject({ gate: "ready", offReason: null });
    }
  });

  it("with no open field and no date, is ready", async () => {
    const { auto_off: _a, ...noDate } = OLDER;
    const { c } = make({ "GET /health": json(noDate) });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "ready", offReason: null });
  });

  it("reads today from the clock it is given", async () => {
    const fetchImpl = scriptedFetch({ "GET /health": json({ ...OLDER, auto_off: "2026-12-15" }) });
    const open = new RunnerController({ base: BASE, fetchImpl, now: () => Date.parse("2026-12-15T23:59:59Z") });
    await open.start();
    expect(open.getSnapshot().gate).toBe("ready");
    const closed = new RunnerController({ base: BASE, fetchImpl: scriptedFetch({ "GET /health": json({ ...OLDER, auto_off: "2026-12-15" }) }), now: () => Date.parse("2026-12-16T00:00:00Z") });
    await closed.start();
    expect(closed.getSnapshot()).toMatchObject({ gate: "off", offReason: "auto_off" });
  });

  it("a runner that is not ok stays hidden, not off, whatever its date says", async () => {
    const { c } = make({ "GET /health": json({ ...OLDER, ok: false, auto_off: "2026-01-01" }) });
    await c.start();
    expect(c.getSnapshot()).toMatchObject({ gate: "hidden", offReason: null });
  });
});

describe("each provider choice works", () => {
  const cases: [Provider, string | null][] = [
    ["nvidia", KEY],
    ["tokenfactory", KEY],
    ["free", null],
  ];
  for (const [provider, key] of cases) {
    it(`${provider}: sends the provider${key ? " and the key" : " and no key"} in one POST /plan`, async () => {
      const { c, fetchImpl } = make({ "POST /plan": json(validReply) });
      await c.start();
      c.setProvider(provider);
      if (key) c.setKey(key);
      await c.plan(DESCRIPTION);
      const plans = callsTo(fetchImpl, "POST", "/plan");
      expect(plans).toHaveLength(1);
      expect(bodyOf(plans[0]!)).toEqual({ description: DESCRIPTION, provider, key });
      expect((plans[0]![1] as RequestInit).headers).toEqual({ "Content-Type": "application/json" });
      expect(c.getSnapshot().planPhase).toBe("done");
    });
  }

  it("the free allowance ignores a key that was typed before switching", async () => {
    const { c, fetchImpl } = make({ "POST /plan": json(validReply) });
    await c.start();
    c.setKey(KEY);
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    expect(bodyOf(callsTo(fetchImpl, "POST", "/plan")[0]!).key).toBeNull();
    expect(JSON.stringify(fetchImpl.mock.calls)).not.toContain(KEY);
  });

  it("a key provider with no key sends nothing and asks for the key", async () => {
    const { c, fetchImpl } = make({ "POST /plan": json(validReply) });
    await c.start();
    c.setProvider("tokenfactory");
    await c.plan(DESCRIPTION);
    expect(callsTo(fetchImpl, "POST", "/plan")).toHaveLength(0);
    expect(c.getSnapshot().planFault).toMatchObject({ kind: "key_missing", message: MESSAGES.key_missing });
    expect(c.canPlan(DESCRIPTION)).toBe(false);
    c.setKey("x");
    expect(c.canPlan(DESCRIPTION)).toBe(true);
    expect(c.getSnapshot().planFault).toBeNull();
  });

  it("an empty description sends nothing", async () => {
    const { c, fetchImpl } = make({ "POST /plan": json(validReply) });
    await c.start();
    c.setProvider("free");
    await c.plan("   ");
    expect(callsTo(fetchImpl, "POST", "/plan")).toHaveLength(0);
    expect(c.canPlan("  ")).toBe(false);
  });
});

describe("the key stays in memory, once", () => {
  async function planWithKey(routes: Parameters<typeof scriptedFetch>[0], forget = true) {
    const made = make({ "POST /runs": json(queued, 202), ...routes });
    await made.c.start();
    made.c.setProvider("nvidia");
    made.c.setKey(KEY);
    made.c.setForget(forget);
    await made.c.plan(DESCRIPTION);
    return made;
  }

  it("is never written to storage, the URL, history or the console", async () => {
    const { c, fetchImpl } = await planWithKey({ "POST /plan": json(validReply), "GET /runs/r1": json(rec("done", { result: SUMMARY })) });
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(c.getSnapshot().runPhase).toBe("done");
    // the spies: nothing was stored, and nothing carried the key
    expect(spies.localStorage.setItem).not.toHaveBeenCalled();
    expect(spies.sessionStorage.setItem).not.toHaveBeenCalled();
    expect(spies.history.pushState).not.toHaveBeenCalled();
    for (const spy of spies.consoles) expect(spy).not.toHaveBeenCalled();
    expect(leaked(spies)).not.toContain(KEY);
    expect(spies.win.location.hash).not.toContain(KEY);
    // the history entry written is the run token and nothing else
    expect(spies.history.replaceState.mock.calls).toEqual([[null, "", "#run-r1"]]);
    // fetch: the key is in the body of the one POST /plan, in no URL and no header, and in no other call
    const withKey = fetchImpl.mock.calls.filter((call) => JSON.stringify(call).includes(KEY));
    expect(withKey).toHaveLength(1);
    expect(withKey[0]![0]).toBe(`${BASE}/plan`);
    expect(String((withKey[0]![1] as RequestInit).body)).toContain(KEY);
    expect(JSON.stringify((withKey[0]![1] as RequestInit).headers)).not.toContain(KEY);
    for (const call of fetchImpl.mock.calls) expect(String(call[0])).not.toContain(KEY);
  });

  it("is cleared after the request when forget is on (the default)", async () => {
    const { c } = await planWithKey({ "POST /plan": json(validReply) });
    expect(c.getSnapshot().forget).toBe(true);
    expect(c.getSnapshot().key).toBe("");
    expect(JSON.stringify(c.getSnapshot())).not.toContain(KEY);
  });

  it("is cleared whatever the outcome: refused key, cap, runner off, network failure, failed plan", async () => {
    const outcomes: Parameters<typeof scriptedFetch>[0][] = [
      { "POST /plan": json({ detail: "no" }, 401) },
      { "POST /plan": json({ cap: "free_plans_per_ip_per_day", resets_at: "2026-10-29T00:00:00Z" }, 429) },
      { "POST /plan": json({ error: OFF_WORDS }, 503) },
      { "POST /plan": json({ error: FREE_TIER_DOWN }, 503) },
      { "POST /plan": [networkError()] },
      { "POST /plan": json({ status: "failed", plan: null, problems: ["no"], calls: 2, cost_usd: 0, model: "m" }) },
      { "POST /plan": json({ status: "maybe" }) },
    ];
    for (const routes of outcomes) {
      const { c } = await planWithKey(routes);
      expect(c.getSnapshot().key, JSON.stringify(Object.keys(routes))).toBe("");
      expect(JSON.stringify(c.getSnapshot())).not.toContain(KEY);
    }
  });

  it("stays in the field when forget is off, and still reaches nothing but the field", async () => {
    const { c } = await planWithKey({ "POST /plan": json(validReply) }, false);
    const s = c.getSnapshot();
    expect(s.key).toBe(KEY);
    expect(JSON.stringify({ ...s, key: undefined })).not.toContain(KEY);
    expect(leaked(spies)).not.toContain(KEY);
  });

  it("is cleared when the panel goes away", async () => {
    const made = make({ "POST /plan": json(validReply) });
    await made.c.start();
    made.c.setKey(KEY);
    made.c.setForget(false);
    made.c.stop();
    expect(made.c.getSnapshot().key).toBe("");
  });

  it("is struck out of anything the server sends back", async () => {
    const { c } = await planWithKey({
      "POST /plan": json({ status: "failed", plan: null, problems: [`the provider says ${KEY} is not valid`], calls: 1, cost_usd: 0, model: "m" }),
    });
    expect(c.getSnapshot().planReply?.problems).toEqual(["the provider says [key] is not valid"]);
    expect(JSON.stringify(c.getSnapshot())).not.toContain(KEY);
  });

  it("a second request needs the key again", async () => {
    const { c, fetchImpl } = await planWithKey({ "POST /plan": json(validReply) });
    await c.plan(DESCRIPTION);
    expect(callsTo(fetchImpl, "POST", "/plan")).toHaveLength(1);
    expect(c.getSnapshot().planFault?.kind).toBe("key_missing");
  });
});

describe("plan to editor hand-off", () => {
  it("passes the plan document to the page when the plan is valid", async () => {
    const onPlan = vi.fn();
    const { c } = make({ "POST /plan": json(validReply) }, { onPlan });
    await c.start();
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    expect(onPlan).toHaveBeenCalledTimes(1);
    expect(onPlan.mock.calls[0]![0]).toMatchObject({ status: "valid", plan: VALID_PLAN, calls: 1, model: "nemotron-mock" });
    expect(c.getSnapshot().planReply?.status).toBe("valid");
  });

  it("shows a failed plan with its problems and loads nothing", async () => {
    const onPlan = vi.fn();
    const { c } = make({ "POST /plan": json({ status: "failed", plan: null, problems: ["steps[1]: the hinge limit is too high"], calls: 2, cost_usd: 0.01, model: "m" }) }, { onPlan });
    await c.start();
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    expect(onPlan).not.toHaveBeenCalled();
    expect(c.getSnapshot()).toMatchObject({ planPhase: "done", planReply: { status: "failed", problems: ["steps[1]: the hinge limit is too high"] } });
  });

  it("shows a refusal with its reason, and hands over the refusal so the editor can offer the nearest skill", async () => {
    const onPlan = vi.fn();
    const { c } = make({ "POST /plan": json({ status: "refused", plan: REFUSAL_PLAN, problems: ["Folding cloth is not simulated."], calls: 1, cost_usd: 0, model: "m" }) }, { onPlan });
    await c.start();
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    expect(c.getSnapshot().planReply).toMatchObject({ status: "refused", problems: ["Folding cloth is not simulated."] });
    expect(onPlan).toHaveBeenCalledWith(expect.objectContaining({ status: "refused", plan: REFUSAL_PLAN }));
  });

  it("sends a long description cut to the contract's 4000 characters", async () => {
    const { c, fetchImpl } = make({ "POST /plan": json(validReply) });
    await c.start();
    c.setProvider("free");
    await c.plan("x".repeat(5000));
    expect((bodyOf(callsTo(fetchImpl, "POST", "/plan")[0]!).description as string).length).toBe(4000);
  });
});

describe("the run lifecycle", () => {
  it("queued, running, done: renders the summary, stops polling and refreshes the counts", async () => {
    const { c, fetchImpl } = make({
      "POST /runs": json(queued, 202),
      "GET /runs/r1": [json(rec("running")), json(rec("done", { result: SUMMARY }))],
    });
    await c.start();
    await c.run(VALID_PLAN);
    let s = c.getSnapshot();
    expect(s).toMatchObject({ runPhase: "queued", runId: "r1", runEtaS: 480 });
    expect(bodyOf(callsTo(fetchImpl, "POST", "/runs")[0]!)).toEqual({ plan: VALID_PLAN });
    expect(spies.history.replaceState).toHaveBeenCalledWith(null, "", "#run-r1");

    await vi.advanceTimersByTimeAsync(POLL_MS - 1);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    s = c.getSnapshot();
    expect(s.runPhase).toBe("running");
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(POLL_MS);
    s = c.getSnapshot();
    expect(s.runPhase).toBe("done");
    expect(s.runRecord?.result?.pick?.label).toBe("Suction cup plus parallel jaw");
    expect(s.runRecord?.result?.completion).toEqual({ k: 7, n: 8 });
    expect(s.runRecord?.result?.reportUrl).toContain("report.json");
    expect(s.runFault).toBeNull();

    // polling has stopped, and the daily counts were read again
    const polls = callsTo(fetchImpl, "GET", "/runs/r1").length;
    await vi.advanceTimersByTimeAsync(5 * POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(polls);
    expect(callsTo(fetchImpl, "GET", "/health").length).toBeGreaterThanOrEqual(2);
  });

  it("starts the clock at the submit, so elapsed time counts from the click", async () => {
    const { c } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": json(rec("running")) });
    await c.start();
    const t0 = Date.now();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot().runStartedAt).toBe(t0);
  });

  it("a failed run shows its error and stops polling", async () => {
    const { c, fetchImpl } = make({
      "POST /runs": json(queued, 202),
      "GET /runs/r1": [json(rec("failed", { error: "The physics worker ran out of time on step 3." }))],
    });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    const s = c.getSnapshot();
    expect(s.runPhase).toBe("failed");
    expect(s.runRecord?.error).toBe("The physics worker ran out of time on step 3.");
    await vi.advanceTimersByTimeAsync(3 * POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(1);
  });

  it("backs off on a flaky connection, then stops with a retry", async () => {
    const { c, fetchImpl } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": [networkError()] });
    await c.start();
    await c.run(VALID_PLAN);
    const gaps: number[] = [];
    let last = Date.now();
    for (let i = 0; i < 5; i++) {
      const before = callsTo(fetchImpl, "GET", "/runs/r1").length;
      while (callsTo(fetchImpl, "GET", "/runs/r1").length === before) await vi.advanceTimersByTimeAsync(500);
      gaps.push(Date.now() - last);
      last = Date.now();
    }
    // 10 s, then 15 s, 22.5 s, 33.75 s: each gap longer than the one before
    expect(gaps[0]).toBeGreaterThanOrEqual(POLL_MS);
    for (let i = 1; i < gaps.length; i++) expect(gaps[i]!).toBeGreaterThan(gaps[i - 1]!);
    const s = c.getSnapshot();
    expect(s.runFault).toMatchObject({ kind: "network", retry: "poll" });
    const stopped = callsTo(fetchImpl, "GET", "/runs/r1").length;
    await vi.advanceTimersByTimeAsync(10 * 60_000);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(stopped);
  });

  it("a retry after a network failure picks the run back up", async () => {
    const { c } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": [networkError(), networkError(), networkError(), networkError(), networkError(), json(rec("done", { result: SUMMARY }))] });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(5 * 60_000);
    expect(c.getSnapshot().runFault?.retry).toBe("poll");
    c.poll();
    await vi.advanceTimersByTimeAsync(0);
    expect(c.getSnapshot().runPhase).toBe("done");
    expect(c.getSnapshot().runFault).toBeNull();
  });

  it("a run the server has forgotten says so and does not retry", async () => {
    const { c } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": json({ detail: "gone" }, 404) });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(c.getSnapshot().runFault).toMatchObject({ kind: "not_found", message: MESSAGES.not_found, retry: null });
  });

  it("stop() ends the polling", async () => {
    const { c, fetchImpl } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": json(rec("running")) });
    await c.start();
    await c.run(VALID_PLAN);
    c.stop();
    await vi.advanceTimersByTimeAsync(5 * POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(0);
  });
});

describe("hash resume", () => {
  it("reads #run-<id> on load and shows the finished run without a new POST", async () => {
    const { c, fetchImpl } = make({ "GET /runs/abc123": json(rec("done", { result: SUMMARY })) }, { hash: "#run-abc123" });
    await c.start();
    await vi.advanceTimersByTimeAsync(0);
    const s = c.getSnapshot();
    expect(s).toMatchObject({ gate: "ready", runId: "abc123", runPhase: "done" });
    expect(s.runRecord?.result?.completion).toEqual({ k: 7, n: 8 });
    expect(callsTo(fetchImpl, "POST", "/runs")).toHaveLength(0);
    expect(callsTo(fetchImpl, "GET", "/runs/abc123")).toHaveLength(1);
  });

  it("counts the elapsed time from the run's own created_at after a reload", async () => {
    const { c } = make({ "GET /runs/abc123": json(rec("running", { created_at: "2026-10-28T10:00:00Z" })) }, { hash: "#run-abc123" });
    await c.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(c.getSnapshot().runStartedAt).toBe(Date.parse("2026-10-28T10:00:00Z"));
    expect(c.getSnapshot().runPhase).toBe("running");
  });

  it("keeps polling a resumed run that is still going", async () => {
    const { c, fetchImpl } = make({ "GET /runs/abc123": [json(rec("running")), json(rec("done", { result: SUMMARY }))] }, { hash: "#run-abc123" });
    await c.start();
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(c.getSnapshot().runPhase).toBe("done");
    expect(callsTo(fetchImpl, "GET", "/runs/abc123")).toHaveLength(2);
  });

  it("ignores a hash that is not a run token", async () => {
    for (const hash of ["", "#plan", "#run-a/b", "#run-"]) {
      const { c, fetchImpl } = make({}, { hash });
      await c.start();
      expect(c.getSnapshot().runPhase, hash).toBe("idle");
      expect(fetchImpl.mock.calls.filter((call) => String(call[0]).includes("/runs"))).toHaveLength(0);
    }
  });

  it("does not resume when the runner is hidden", async () => {
    const { c, fetchImpl } = make({ "GET /health": json({ ...HEALTH, ok: false }) }, { hash: "#run-abc123" });
    await c.start();
    expect(fetchImpl.mock.calls.filter((call) => String(call[0]).includes("/runs"))).toHaveLength(0);
  });

  it("clearing a run takes its token out of the address and keeps the page's own path", async () => {
    const { c } = make({ "GET /runs/abc123": json(rec("done", { result: SUMMARY })) }, { hash: "#run-abc123" });
    await c.start();
    await vi.advanceTimersByTimeAsync(0);
    c.reset();
    expect(spies.history.replaceState).toHaveBeenCalledWith(null, "", "/robot-tool-tournament/studio/");
    expect(c.getSnapshot()).toMatchObject({ runPhase: "idle", runId: null, runRecord: null });
  });

  it("clearing leaves a hash that is not a run token alone", () => {
    spies.win.location.hash = "#plan";
    browserHash.write(null);
    expect(spies.history.replaceState).not.toHaveBeenCalled();
  });
});

describe("base path", () => {
  it("writes the run token as a fragment-only URL, so it keeps whatever path the site is served under", async () => {
    for (const pathname of ["/studio/", "/robot-tool-tournament/studio/"]) {
      spies.history.replaceState.mockClear();
      spies.win.location.pathname = pathname;
      const { c } = make({ "POST /runs": json(queued, 202) });
      await c.start();
      await c.run(VALID_PLAN);
      expect(spies.history.replaceState).toHaveBeenCalledWith(null, "", "#run-r1");
      c.reset();
      expect(spies.history.replaceState).toHaveBeenLastCalledWith(null, "", pathname);
    }
  });

  it("calls the runner at its own base, whatever the site's base path", async () => {
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /runs": json(queued, 202), "GET /runs/r1": json(rec("running")) });
    const c = new RunnerController({ base: "https://runner.example/api", fetchImpl: f });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(f.mock.calls.map((call) => call[0])).toEqual(["https://runner.example/api/health", "https://runner.example/api/runs", "https://runner.example/api/runs/r1"]);
  });
});

describe("each error has a plain message", () => {
  const planWith = async (routes: Parameters<typeof scriptedFetch>[0]) => {
    const { c } = make(routes);
    await c.start();
    c.setKey(KEY);
    await c.plan(DESCRIPTION);
    return c.getSnapshot();
  };

  it("401: the provider rejected the key", async () => {
    const s = await planWith({ "POST /plan": json({ detail: "bad" }, 401) });
    expect(s.planPhase).toBe("error");
    expect(s.planFault).toEqual<Fault>({ kind: "key_rejected", message: "The provider rejected that key.", retry: null });
  });

  it("429: names the cap that was reached and when it resets", async () => {
    const s = await planWith({ "POST /plan": json({ error: "cap reached", cap: "free_plans_per_ip_per_day", limit: 3, resets_at: "2026-10-29T00:00:00Z" }, 429) });
    expect(s.planFault).toEqual<Fault>({ kind: "cap", message: "You have used your free plans for today. It resets at 29 October 2026, 00:00 UTC.", retry: null });
  });

  it("429 on a run says the runs cap", async () => {
    const { c } = make({ "POST /runs": json({ error: "cap reached", cap: "runs_per_ip_per_day", limit: 2, resets_at: "2026-10-29T00:00:00Z" }, 429) });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot().runFault?.message).toBe("You have reached the daily limit on physics runs. It resets at 29 October 2026, 00:00 UTC.");
    expect(c.getSnapshot().runPhase).toBe("idle");
  });

  it("429 on the total budget does not reset: the fault says so, so the panel can point to running it yourself", async () => {
    const body = { error: "cap reached: the runner's total compute budget; it does not reset", cap: "spend_total_usd", limit: 10, resets_at: null };
    const { c } = make({ "POST /runs": json(body, 429) });
    await c.start();
    await c.run(VALID_PLAN);
    const s = c.getSnapshot();
    expect(s.runFault).toEqual<Fault>({ kind: "cap", message: "The hosted runner's total compute budget is used up. It does not reset.", retry: null, neverResets: true });
    expect(s.runPhase).toBe("idle");
    // and on a plan, the same
    const planned = await planWith({ "POST /plan": json(body, 429) });
    expect(planned.planFault).toEqual<Fault>({ kind: "cap", message: "The hosted runner's total compute budget is used up. It does not reset.", retry: null, neverResets: true });
  });

  it("429 on a cap that resets carries no such mark", async () => {
    const { c } = make({ "POST /runs": json({ error: "cap reached", cap: "spend_per_day_usd", limit: 2, resets_at: "2026-10-29T00:00:00Z" }, 429) });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot().runFault).toEqual<Fault>({ kind: "cap", message: "The hosted runner's compute budget for today is used up. It resets at 29 October 2026, 00:00 UTC.", retry: null });
    expect(c.getSnapshot().runFault).not.toHaveProperty("neverResets");
  });

  it("503: a runner that says it is off reads as off, with the tutorial to point to, on a plan and on a run", async () => {
    for (const words of [OFF_WORDS, "the hosted runner closed after 2026-12-15; recorded runs still replay"]) {
      const s = await planWith({ "POST /plan": json({ error: words }, 503) });
      expect(s.planFault, words).toEqual<Fault>({ kind: "off", message: "The hosted runner is off. Run it on your own machine:", retry: null });
      const { c } = make({ "POST /runs": json({ error: words }, 503) });
      await c.start();
      await c.run(VALID_PLAN);
      expect(c.getSnapshot().runFault, words).toEqual<Fault>({ kind: "off", message: "The hosted runner is off. Run it on your own machine:", retry: null });
    }
  });

  it("503 on a plan with the free allowance: the free tier is down, with no retry that would fail the same way", async () => {
    const { c, fetchImpl } = make({ "POST /plan": json({ error: FREE_TIER_DOWN }, 503) });
    await c.start();
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    const s = c.getSnapshot();
    expect(s.planPhase).toBe("error");
    expect(s.planFault).toEqual<Fault>({ kind: "free_unavailable", message: "The free tier is not available right now. Use your own key, or try later.", retry: null });
    expect(s.gate).toBe("ready");
    // the visitor can switch to a key of their own and plan again
    c.setProvider("nvidia");
    c.setKey(KEY);
    expect(c.canPlan(DESCRIPTION)).toBe(true);
    expect(callsTo(fetchImpl, "POST", "/plan")).toHaveLength(1);
  });

  it("503 on a run: the simulator could not start a job, nothing was charged, and Try again submits again", async () => {
    const { c, fetchImpl } = make({ "POST /runs": [json({ error: JOB_NOT_STARTED }, 503), json(queued, 202)] });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot()).toMatchObject({
      runPhase: "idle",
      runId: null,
      runFault: { kind: "sim_unavailable", message: "The simulator could not start a job right now. Nothing was charged; try again in a few minutes.", retry: "run" },
    });
    expect(spies.history.replaceState).not.toHaveBeenCalled();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot()).toMatchObject({ runPhase: "queued", runFault: null, runId: "r1" });
    expect(callsTo(fetchImpl, "POST", "/runs")).toHaveLength(2);
  });

  it("503 with words that are not about being off reads as unavailable, with a retry, and the panel stays open", async () => {
    for (const error of ["the runner is not configured (free-tier key)", "runs are not configured on this runner", "the run could not be recorded, so its job was cancelled; try again", ""]) {
      const { c } = make({ "POST /runs": json(error ? { error } : {}, 503) });
      await c.start();
      await c.run(VALID_PLAN);
      expect(c.getSnapshot().runFault, error).toEqual<Fault>({ kind: "unavailable", message: MESSAGES.unavailable, retry: "run" });
      expect(c.getSnapshot().gate, error).toBe("ready");
    }
    const s = await planWith({ "POST /plan": new Response("<html>Service Unavailable</html>", { status: 503, headers: { "content-type": "text/html" } }) });
    expect(s.planFault).toEqual<Fault>({ kind: "unavailable", message: MESSAGES.unavailable, retry: "plan" });
  });

  it("503 while watching a run is a passing trouble: it backs off and keeps going, and never says off", async () => {
    const { c, fetchImpl } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": [json({}, 503), json({}, 503), json(rec("running"))] });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(1);
    expect(c.getSnapshot().runFault).toBeNull();
    await vi.advanceTimersByTimeAsync(10 * POLL_MS);
    expect(c.getSnapshot().runPhase).toBe("running");
    expect(c.getSnapshot().runFault).toBeNull();
  });

  it("400 on a run lists the problems the runner's own checks found", async () => {
    const { c } = make({ "POST /runs": json({ error: "plan failed the checks", problems: ["steps: at most 8 steps"] }, 400) });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot().runFault).toEqual<Fault>({ kind: "bad_request", message: "The runner checked the plan again and found problems:", retry: null, problems: ["steps: at most 8 steps"] });
    expect(c.getSnapshot().runPhase).toBe("idle");
  });

  it("a network failure on a plan offers a retry that works", async () => {
    const { c, fetchImpl } = make({ "POST /plan": [networkError(), json(validReply)] });
    await c.start();
    c.setKey(KEY);
    await c.plan(DESCRIPTION);
    expect(c.getSnapshot().planFault).toMatchObject({ kind: "network", retry: "plan" });
    // the key was forgotten with the failed request, so the retry needs it typed again
    expect(c.canPlan(DESCRIPTION)).toBe(false);
    c.setKey(KEY);
    await c.plan(DESCRIPTION);
    expect(c.getSnapshot().planPhase).toBe("done");
    expect(c.getSnapshot().planFault).toBeNull();
    expect(callsTo(fetchImpl, "POST", "/plan")).toHaveLength(2);
  });

  it("a network failure on a run offers a retry that works", async () => {
    const { c } = make({ "POST /runs": [networkError(), json(queued, 202)] });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot().runFault).toMatchObject({ kind: "network", retry: "run" });
    await c.run(VALID_PLAN);
    expect(c.getSnapshot()).toMatchObject({ runPhase: "queued", runFault: null, runId: "r1" });
  });

  it("a server fault is retryable and reads plainly", async () => {
    const s = await planWith({ "POST /plan": json({ detail: "boom" }, 500) });
    expect(s.planFault).toEqual<Fault>({ kind: "server", message: MESSAGES.server, retry: "plan" });
  });

  it("a plan that takes too long stops with a retry", async () => {
    const hang = (_u: string, init?: RequestInit) => new Promise<Response>((_res, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError"))));
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /plan": hang });
    const c = new RunnerController({ base: BASE, fetchImpl: f });
    await c.start();
    c.setProvider("free");
    const p = c.plan(DESCRIPTION);
    await vi.advanceTimersByTimeAsync(PLAN_TIMEOUT_MS);
    await p;
    expect(c.getSnapshot().planFault).toMatchObject({ kind: "server", retry: "plan", message: "The planner took too long. Try again." });
  });

  it("no message carries a raw status code", async () => {
    for (const status of [400, 401, 413, 429, 500, 503]) {
      const s = await planWith({ "POST /plan": json({}, status) });
      expect(s.planFault?.message, String(status)).not.toMatch(/\b(400|401|413|429|500|503)\b/);
    }
  });
});

describe("the timeouts leave room for a slow runner", () => {
  const hang = (_u: string, init?: RequestInit) =>
    new Promise<Response>((_res, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError"))));

  it("are 330 s to plan and 210 s to submit a run", () => {
    expect(PLAN_TIMEOUT_MS).toBe(330_000);
    expect(SUBMIT_TIMEOUT_MS).toBe(210_000);
  });

  it("a plan keeps waiting past the old 120 s, and gives up at 330 s with the plain message", async () => {
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /plan": hang });
    const c = new RunnerController({ base: BASE, fetchImpl: f });
    await c.start();
    c.setProvider("free");
    const p = c.plan(DESCRIPTION);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(c.getSnapshot().planPhase).toBe("planning");
    await vi.advanceTimersByTimeAsync(PLAN_TIMEOUT_MS - 120_000 - 1);
    expect(c.getSnapshot().planPhase).toBe("planning");
    expect(c.getSnapshot().planFault).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    await p;
    expect(c.getSnapshot()).toMatchObject({ planPhase: "error", planFault: { kind: "server", retry: "plan", message: "The planner took too long. Try again." } });
  });

  it("a plan that answers at 300 s is used, not dropped", async () => {
    const slow = (_u: string, init?: RequestInit) =>
      new Promise<Response>((res, rej) => {
        const t = setTimeout(() => res(json(validReply)), 300_000);
        init?.signal?.addEventListener("abort", () => {
          clearTimeout(t);
          rej(new DOMException("aborted", "AbortError"));
        });
      });
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /plan": slow });
    const onPlan = vi.fn();
    const c = new RunnerController({ base: BASE, fetchImpl: f, onPlan });
    await c.start();
    c.setProvider("free");
    const p = c.plan(DESCRIPTION);
    await vi.advanceTimersByTimeAsync(300_000);
    await p;
    expect(c.getSnapshot()).toMatchObject({ planPhase: "done", planFault: null });
    expect(onPlan).toHaveBeenCalledTimes(1);
  });

  it("a submit keeps waiting past the old 30 s, and gives up at 210 s with a retry", async () => {
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /runs": hang });
    const c = new RunnerController({ base: BASE, fetchImpl: f });
    await c.start();
    const r = c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(c.getSnapshot().runPhase).toBe("submitting");
    await vi.advanceTimersByTimeAsync(SUBMIT_TIMEOUT_MS - 30_000 - 1);
    expect(c.getSnapshot().runPhase).toBe("submitting");
    expect(c.getSnapshot().runFault).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    await r;
    expect(c.getSnapshot()).toMatchObject({ runPhase: "idle", runFault: { kind: "server", retry: "run", message: MESSAGES.server } });
  });

  it("a submit the runner accepts after 2 to 3 minutes is queued and watched as usual", async () => {
    const slow = (_u: string, init?: RequestInit) =>
      new Promise<Response>((res, rej) => {
        const t = setTimeout(() => res(json(queued, 202)), 170_000);
        init?.signal?.addEventListener("abort", () => {
          clearTimeout(t);
          rej(new DOMException("aborted", "AbortError"));
        });
      });
    const f = scriptedFetch({ "GET /health": json(HEALTH), "POST /runs": slow, "GET /runs/r1": json(rec("running")) });
    const c = new RunnerController({ base: BASE, fetchImpl: f });
    await c.start();
    const r = c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(170_000);
    await r;
    expect(c.getSnapshot()).toMatchObject({ runPhase: "queued", runId: "r1", runFault: null });
    expect(c.getSnapshot().runStartedAt).toBe(Date.parse("2026-10-28T10:00:05Z"));
  });

  it("the health check keeps its short 8 s limit", async () => {
    const c = new RunnerController({ base: BASE, fetchImpl: hang });
    const started = c.start();
    await vi.advanceTimersByTimeAsync(8_000);
    await started;
    expect(c.getSnapshot().gate).toBe("hidden");
  });
});

describe("a 429 with no cap is the runner being busy", () => {
  // what a throttled Function URL (reserved concurrency) answers: no cap, no reset time
  const throttled = () => json({ message: "Too Many Requests" }, 429);

  it("on a plan: a plain busy message and a retry, not a daily limit", async () => {
    const { c } = make({ "POST /plan": throttled() });
    await c.start();
    c.setKey(KEY);
    await c.plan(DESCRIPTION);
    expect(c.getSnapshot().planPhase).toBe("error");
    expect(c.getSnapshot().planFault).toEqual<Fault>({ kind: "busy", message: "The runner is busy. Try again in a minute.", retry: "plan" });
    // the key was forgotten with the request, so the retry needs it typed again
    expect(c.canPlan(DESCRIPTION)).toBe(false);
  });

  it("on a run: busy, back to idle, and Try again submits again", async () => {
    const { c, fetchImpl } = make({ "POST /runs": [throttled(), json(queued, 202)] });
    await c.start();
    await c.run(VALID_PLAN);
    expect(c.getSnapshot()).toMatchObject({ runPhase: "idle", runFault: { kind: "busy", message: "The runner is busy. Try again in a minute.", retry: "run" } });
    await c.run(VALID_PLAN);
    expect(c.getSnapshot()).toMatchObject({ runPhase: "queued", runFault: null, runId: "r1" });
    expect(callsTo(fetchImpl, "POST", "/runs")).toHaveLength(2);
  });

  it("while watching a run: backs off and keeps going, like any other passing trouble", async () => {
    const { c, fetchImpl } = make({ "POST /runs": json(queued, 202), "GET /runs/r1": [throttled(), throttled(), json(rec("running"))] });
    await c.start();
    await c.run(VALID_PLAN);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(1);
    // no fault is shown for one busy answer, and the next check comes later than the usual 10 s
    expect(c.getSnapshot().runFault).toBeNull();
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(callsTo(fetchImpl, "GET", "/runs/r1")).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(10 * POLL_MS);
    expect(c.getSnapshot().runPhase).toBe("running");
    expect(c.getSnapshot().runFault).toBeNull();
  });

  it("a 429 that names a cap is still a cap with no retry", async () => {
    const { c } = make({ "POST /plan": json({ error: "cap reached", cap: "free_plans_per_ip_per_day", limit: 3, resets_at: "2026-10-29T00:00:00Z" }, 429) });
    await c.start();
    c.setProvider("free");
    await c.plan(DESCRIPTION);
    expect(c.getSnapshot().planFault).toMatchObject({ kind: "cap", retry: null });
  });
});
