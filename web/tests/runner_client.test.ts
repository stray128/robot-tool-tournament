/**
 * The hosted runner's client (lib/runner.ts): the build flag, the URL hash token, the plain error
 * messages, the reply parsers and the base path. No network: fetch is scripted.
 */

import { readFileSync } from "node:fs";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  autoOffPassed,
  capSentence,
  errorFromResponse,
  formatReset,
  getHealth,
  getRun,
  hashForRun,
  KEY_PRIVACY_LINE,
  kindOf503,
  MESSAGES,
  offReason,
  parseHealth,
  parseRun,
  postPlan,
  postRun,
  redact,
  reportLink,
  retryableKind,
  RUN_YOURSELF_LEAD,
  RunnerError,
  runIdFromHash,
  runnerBase,
  runnerConfigured,
  type UnavailableKind,
} from "@/lib/runner";

import { BASE, HEALTH, json, scriptedFetch, SUMMARY, SUMMARY_NO_PICK, SUMMARY_OBJECTS, VALID_PLAN } from "./helpers/runner_mock";

afterEach(() => vi.unstubAllEnvs());

describe("the build flag", () => {
  it("is off when NEXT_PUBLIC_RUNNER_URL is unset, empty or blank", () => {
    expect(runnerBase(undefined)).toBeNull();
    expect(runnerBase("")).toBeNull();
    expect(runnerBase("   ")).toBeNull();
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    expect(runnerConfigured()).toBe(false);
  });

  it("is on for an https origin, trimming the trailing slash, query and hash", () => {
    expect(runnerBase("https://runner.example")).toBe("https://runner.example");
    expect(runnerBase("https://runner.example/")).toBe("https://runner.example");
    expect(runnerBase("https://runner.example/api/?x=1#y")).toBe("https://runner.example/api");
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "https://runner.example");
    expect(runnerConfigured()).toBe(true);
  });

  it("takes plain http only for a loopback host, so a key never travels in clear text", () => {
    expect(runnerBase("http://127.0.0.1:8765")).toBe("http://127.0.0.1:8765");
    expect(runnerBase("http://localhost:8765")).toBe("http://localhost:8765");
    expect(runnerBase("http://runner.example")).toBeNull();
    expect(runnerBase("ftp://runner.example")).toBeNull();
    expect(runnerBase("not a url")).toBeNull();
  });
});

describe("the run id in the URL hash", () => {
  it("is a bare token", () => {
    expect(hashForRun("abc-123_X")).toBe("#run-abc-123_X");
    expect(runIdFromHash("#run-abc-123_X")).toBe("abc-123_X");
  });

  it("rejects anything that is not a bare token", () => {
    for (const h of ["", "#", "#plan", "#run-", "#run-a/b", "#run-a?b", "#run-a b", "#run-../x", "#run-" + "a".repeat(81), "run-abc"]) expect(runIdFromHash(h), h).toBeNull();
  });
});

describe("plain error messages", () => {
  const read = (status: number, body?: unknown) => errorFromResponse(json(body ?? {}, status));

  it("401 says the provider rejected the key", async () => {
    const e = await read(401);
    expect(e.kind).toBe("key_rejected");
    expect(e.message).toBe("The provider rejected that key.");
  });

  it("429 names the cap and when it resets, with the body the runner sends", async () => {
    const e = await read(429, { error: "cap reached: ...", cap: "free_plans_per_ip_per_day", limit: 3, resets_at: "2026-10-29T00:00:00Z" });
    expect(e.kind).toBe("cap");
    expect(e.message).toBe("You have used your free plans for today. It resets at 29 October 2026, 00:00 UTC.");
    expect((await read(429, { cap: "runs_per_ip_per_day", resets_at: "2026-10-29T13:45:00+00:00" })).message).toBe(
      "You have reached the daily limit on physics runs. It resets at 29 October 2026, 13:45 UTC.",
    );
    expect((await read(429, { cap: "free_plans_per_day", resets_at: "2026-10-29T00:00:00Z" })).message).toContain("The free allowance for the whole site is used up for today.");
    expect((await read(429, { cap: "keyed_plans_per_ip_per_day", resets_at: "2026-10-29T00:00:00Z" })).message).toContain("plans with your own key");
    expect((await read(429, { cap: "spend_per_day_usd", resets_at: "2026-10-29T00:00:00Z" })).message).toContain("compute budget for today");
  });

  it("429 says a cap that never resets does not reset (the total budget has no reset time)", async () => {
    const e = await read(429, { error: "x", cap: "spend_total_usd", limit: 10, resets_at: null });
    expect(e.message).toBe("The hosted runner's total compute budget is used up. It does not reset.");
  });

  it("429 on the total budget is a cap that does not reset, and no other refusal is", async () => {
    const total = await read(429, { error: "x", cap: "spend_total_usd", limit: 10, resets_at: null });
    expect(total.kind).toBe("cap");
    expect(total.neverResets).toBe(true);
    // a cap with a reset time, however it is worded, resets
    for (const cap of ["free_plans_per_day", "free_plans_per_ip_per_day", "keyed_plans_per_ip_per_day", "runs_per_ip_per_day", "spend_per_day_usd", "some_new_cap"]) {
      expect((await read(429, { cap, resets_at: "2026-10-29T00:00:00Z" })).neverResets, cap).toBe(false);
    }
    // the hosting layer's throttle, the 503s and the other statuses are not a cap that does not reset
    expect((await read(429, {})).neverResets).toBe(false);
    expect((await read(503, { error: "the runner is switched off for now; recorded runs still replay" })).neverResets).toBe(false);
    expect((await read(401)).neverResets).toBe(false);
  });

  it("the off message and the cap that does not reset share the words that lead to the tutorial link", () => {
    expect(RUN_YOURSELF_LEAD).toBe("Run it on your own machine:");
    expect(MESSAGES.off).toBe("The hosted runner is off. Run it on your own machine:");
    expect(MESSAGES.off.endsWith(RUN_YOURSELF_LEAD)).toBe(true);
  });

  it("429 reads a cap nested under detail", async () => {
    expect((await read(429, { detail: { cap: "spend_per_day_usd", resets_at: "2026-10-29T00:00:00Z" } })).message).toContain("compute budget for today");
    expect((await read(429, { error: { cap: "runs_per_ip_per_day", resets_at: "2026-10-29T00:00:00Z" } })).message).toContain("daily limit on physics runs");
  });

  it("429 with no cap is the runner being busy, not a daily limit (a throttled Function URL sends no cap)", async () => {
    const bodies: Response[] = [
      new Response("busy", { status: 429 }),
      new Response("", { status: 429 }),
      new Response("<html>Too Many Requests</html>", { status: 429, headers: { "content-type": "text/html" } }),
      json({}, 429),
      json({ message: "Too Many Requests" }, 429),
      json({ error: "Too Many Requests" }, 429),
      json({ cap: null, resets_at: "2026-10-29T00:00:00Z" }, 429),
      json({ cap: "", resets_at: null }, 429),
      json({ detail: "slow down" }, 429),
    ];
    for (const res of bodies) {
      const e = await errorFromResponse(res);
      expect(e.kind).toBe("busy");
      expect(e.message).toBe("The runner is busy. Try again in a minute.");
      expect(e.message).toBe(MESSAGES.busy);
      expect(e.message).not.toMatch(/daily|limit|resets/i);
      expect(e.cap).toBeNull();
      expect(e.resetsAt).toBeNull();
      expect(e.retryable).toBe(true);
    }
  });

  it("429 with a cap is still a cap, which is not retried (the busy reading does not swallow it)", async () => {
    const e = await read(429, { error: "cap reached", cap: "runs_per_ip_per_day", limit: 2, resets_at: "2026-10-29T00:00:00Z" });
    expect(e.kind).toBe("cap");
    expect(e.cap).toBe("runs_per_ip_per_day");
    expect(e.retryable).toBe(false);
    expect(e.message).not.toContain("busy");
  });

  it("429 puts a cap this page does not know into words", async () => {
    expect((await read(429, { cap: "some_new_cap", resets_at: "2026-10-29T00:00:00Z" })).message).toBe("The some new cap limit was reached. It resets at 29 October 2026, 00:00 UTC.");
  });

  it("400 lists the problems the runner's checks found, and is plain without them", async () => {
    const withProblems = await read(400, { error: "plan failed the checks", problems: ["steps: at most 8 steps", "objects[2]: width_mm must be in [60, 600] mm"] });
    expect(withProblems.kind).toBe("bad_request");
    expect(withProblems.message).toBe("The runner checked the plan again and found problems:");
    expect(withProblems.problems).toEqual(["steps: at most 8 steps", "objects[2]: width_mm must be in [60, 600] mm"]);
    const plain = await read(400, { error: "unknown provider" });
    expect(plain.message).toBe("The runner could not use that request. Check the description and the key, then try again.");
    expect(plain.problems).toEqual([]);
  });

  it("never shows a raw status code", async () => {
    for (const status of [400, 401, 404, 413, 422, 429, 500, 502, 503]) {
      const e = await read(status);
      expect(e.message, String(status)).not.toMatch(/\b(400|401|404|413|422|429|500|502|503)\b/);
    }
  });

  it("only a network failure, a server fault, a busy runner or a simulator that could not start a job can be retried", () => {
    expect(new RunnerError("network", "x").retryable).toBe(true);
    expect(new RunnerError("server", "x").retryable).toBe(true);
    expect(new RunnerError("busy", "x").retryable).toBe(true);
    expect(new RunnerError("sim_unavailable", "x").retryable).toBe(true);
    expect(new RunnerError("unavailable", "x").retryable).toBe(true);
    expect(new RunnerError("key_rejected", "x").retryable).toBe(false);
    expect(new RunnerError("cap", "x").retryable).toBe(false);
    expect(new RunnerError("off", "x").retryable).toBe(false);
    expect(new RunnerError("free_unavailable", "x").retryable).toBe(false);
    expect(retryableKind("sim_unavailable")).toBe(true);
    expect(retryableKind("not_found")).toBe(false);
  });

  it("names caps in plain words", () => {
    expect(capSentence("runs_per_ip_per_day")).toBe("You have reached the daily limit on physics runs.");
    expect(capSentence("some_cap")).toBe("The some cap limit was reached.");
    expect(formatReset("garbage")).toBeNull();
    expect(formatReset(null)).toBeNull();
  });
});

/** What the runner's 503s say (src/rtt/runner/app.py): the first two are the only ones that mean it is off. */
const OFF_WORDS = ["the runner is switched off for now; recorded runs still replay", "the hosted runner closed after 2026-12-15; recorded runs still replay"];
const FREE_WORDS = ["the free tier is unavailable right now; try your own key"];
const SIM_WORDS = [
  "the simulation backend is unavailable; nothing was reserved, try again later",
  "the simulation job could not be started; try again later",
  "the platform refused the simulation job; this did not count against your runs",
];
const OTHER_WORDS = [
  "the runner is not configured (free-tier key)",
  "runs are not configured on this runner",
  "the run could not be recorded, so its job was cancelled; try again",
  "the report link could not be made; try again",
];

describe("a 503 reads as the runner being off only when the runner says it is off", () => {
  const read503 = (error?: string) => errorFromResponse(json(error === undefined ? {} : { error }, 503));

  it("the kill switch and the auto-off date read as off, with the tutorial to point to and no retry", async () => {
    for (const error of OFF_WORDS) {
      const e = await read503(error);
      expect(e.kind, error).toBe("off");
      expect(e.message).toBe("The hosted runner is off. Run it on your own machine:");
      expect(e.retryable).toBe(false);
    }
  });

  it("the free tier being down says so, points to the visitor's own key, and is not retried as it was", async () => {
    for (const error of FREE_WORDS) {
      const e = await read503(error);
      expect(e.kind, error).toBe("free_unavailable");
      expect(e.message).toBe("The free tier is not available right now. Use your own key, or try later.");
      expect(e.message).not.toMatch(/\boff\b/i);
      expect(e.retryable).toBe(false);
    }
  });

  it("a simulator that could not start a job says nothing was charged, and can be tried again", async () => {
    for (const error of SIM_WORDS) {
      const e = await read503(error);
      expect(e.kind, error).toBe("sim_unavailable");
      expect(e.message).toBe("The simulator could not start a job right now. Nothing was charged; try again in a few minutes.");
      expect(e.message).not.toMatch(/\boff\b/i);
      expect(e.retryable).toBe(true);
    }
  });

  it("any other 503 is plainly unavailable, not off: the runner's other words, a body with no words, and the hosting layer's", async () => {
    const bodies: Response[] = [
      ...OTHER_WORDS.map((error) => json({ error }, 503)),
      json({}, 503),
      json({ error: "" }, 503),
      json({ error: 503 }, 503),
      json({ message: "Service Unavailable" }, 503),
      json({ detail: "off" }, 503),
      // not the runner's words (it says "switched off" or "closed after"): only those count as the runner saying it is off
      json({ error: "the runner is off" }, 503),
      new Response("", { status: 503 }),
      new Response("not json", { status: 503 }),
      new Response("<html>Service Unavailable</html>", { status: 503, headers: { "content-type": "text/html" } }),
    ];
    for (const res of bodies) {
      const e = await errorFromResponse(res);
      expect(e.kind).toBe("unavailable");
      expect(e.message).toBe("The hosted runner cannot take this request right now. Try again in a few minutes.");
      expect(e.message).not.toMatch(/\boff\b/i);
      expect(e.retryable).toBe(true);
    }
  });

  it("reads the words from whichever field a body puts them in, as the runner's own bodies do in `error`", async () => {
    expect((await errorFromResponse(json({ error: OFF_WORDS[0] }, 503))).kind).toBe("off");
    expect((await errorFromResponse(json({ message: OFF_WORDS[1] }, 503))).kind).toBe("off");
    expect((await errorFromResponse(json({ detail: FREE_WORDS[0] }, 503))).kind).toBe("free_unavailable");
    // a capital letter or the words inside a longer sentence still read the same
    expect(kindOf503("The Runner Is Switched Off For Now")).toBe("off");
    expect(kindOf503("503: the simulation job could not be started; try again later")).toBe("sim_unavailable");
    expect(kindOf503(null)).toBe("unavailable");
  });

  it("no message of them shows a status code or a phrase of the runner's own", async () => {
    for (const error of [...OFF_WORDS, ...FREE_WORDS, ...SIM_WORDS, ...OTHER_WORDS]) {
      const e = await read503(error);
      expect(e.message, error).not.toMatch(/\b503\b|recorded runs|reserved|platform|backend|HTTP/i);
    }
  });

  it("every 503 in the runner's source is one this page has a reading for, and only the two it says are off read as off", () => {
    const source = readFileSync(path.join(__dirname, "..", "..", "src", "rtt", "runner", "app.py"), "utf8");
    // `HTTPError(503, "words")` and the same over several lines; an f-string's {fields} are filled with a sample
    const sites = source.match(/HTTPError\(\s*503\b/g) ?? [];
    const found = [...source.matchAll(/HTTPError\(\s*503,\s*f?"((?:[^"\\]|\\.)*)"/g)].map((m) => m[1]!.replace(/\{[^}]*\}/g, "2026-12-15"));
    expect(found, "a 503 with a body this test cannot read").toHaveLength(sites.length);
    const known: Record<string, UnavailableKind> = {
      ...Object.fromEntries(OFF_WORDS.map((w) => [w, "off"])),
      ...Object.fromEntries(FREE_WORDS.map((w) => [w, "free_unavailable"])),
      ...Object.fromEntries(SIM_WORDS.map((w) => [w, "sim_unavailable"])),
      ...Object.fromEntries(OTHER_WORDS.map((w) => [w, "unavailable"])),
    };
    // a new or reworded 503 fails here until it is read on purpose
    expect([...new Set(found)].sort()).toEqual(Object.keys(known).sort());
    for (const words of found) expect(kindOf503(words), words).toBe(known[words]);
    expect(found.filter((w) => kindOf503(w) === "off")).toHaveLength(OFF_WORDS.length);
  });
});

describe("the calls", () => {
  it("build every URL from the runner base, path included", async () => {
    const f = scriptedFetch({
      "GET /health": json(HEALTH),
      "POST /plan": json({ status: "valid", plan: VALID_PLAN, problems: [], calls: 1, cost_usd: 0.01, model: "m" }),
      "POST /runs": json({ run_id: "r1", status: "queued", eta_s: 60 }, 202),
      "GET /runs/r1": json({ status: "running", created_at: "2026-10-28T10:00:00Z", cost_estimate_usd: 0, error: null, result: null }),
    });
    await getHealth(BASE, undefined, f);
    await postPlan(BASE, { description: "d", provider: "free", key: null }, undefined, f);
    await postRun(BASE, VALID_PLAN, undefined, f);
    await getRun(BASE, "r1", undefined, f);
    expect(f.mock.calls.map((c) => c[0])).toEqual([`${BASE}/health`, `${BASE}/plan`, `${BASE}/runs`, `${BASE}/runs/r1`]);
  });

  it("send no cookies, no referrer and no stored response, and follow no redirect", async () => {
    const f = scriptedFetch({
      "GET /health": json(HEALTH),
      "POST /plan": json({ status: "valid", plan: VALID_PLAN, problems: [], calls: 1, cost_usd: 0.01, model: "m" }),
      "POST /runs": json({ run_id: "r1", status: "queued", eta_s: 60 }, 202),
      "GET /runs/r1": json({ status: "running", created_at: null, cost_estimate_usd: 0, error: null, result: null }),
    });
    await getHealth(BASE, undefined, f);
    await postPlan(BASE, { description: "d", provider: "nvidia", key: "k-123456" }, undefined, f);
    await postRun(BASE, VALID_PLAN, undefined, f);
    await getRun(BASE, "r1", undefined, f);
    expect(f.mock.calls).toHaveLength(4);
    for (const call of f.mock.calls) {
      expect(call[1], String(call[0])).toMatchObject({ cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer", redirect: "error" });
    }
  });

  it("encode a run id into the path", async () => {
    const f = scriptedFetch({});
    await getRun(BASE, "a/b", undefined, f).catch(() => {});
    expect(f.mock.calls[0]![0]).toBe(`${BASE}/runs/a%2Fb`);
  });

  it("turn a thrown fetch into a retryable network error", async () => {
    const f = vi.fn(async () => {
      throw new TypeError("fetch failed");
    });
    await expect(getHealth(BASE, undefined, f)).rejects.toMatchObject({ kind: "network", message: "Could not reach the hosted runner. Check your connection and try again." });
  });

  it("refuse a reply it cannot read", async () => {
    const f = scriptedFetch({ "POST /plan": json({ status: "maybe" }), "POST /runs": json({ run_id: "../x" }) });
    await expect(postPlan(BASE, { description: "d", provider: "free", key: null }, undefined, f)).rejects.toMatchObject({ kind: "bad_reply" });
    await expect(postRun(BASE, VALID_PLAN, undefined, f)).rejects.toMatchObject({ kind: "bad_reply" });
  });
});

describe("the replies", () => {
  it("reads a finished run as the runner builds it: tools as names, ruled_out as counts, steps without labels", () => {
    const run = parseRun({ status: "done", created_at: "2026-10-28T10:00:00Z", cost_estimate_usd: 0.18, error: null, result: SUMMARY });
    const r = run.result!;
    expect(r.pick?.label).toBe("Suction cup plus parallel jaw");
    expect(r.pick?.kind).toBe("multi_head");
    expect(r.pick?.tools).toEqual([
      { label: "Suction cup, 40 mm", family: "" },
      { label: "Parallel jaw, 90 mm stroke", family: "" },
    ]);
    expect(r.completion).toEqual({ k: 7, n: 8 });
    expect(r.baseline).toEqual({ label: "Parallel jaw, 90 mm stroke", completion: { k: 2, n: 8 } });
    expect(r.ruledOut).toEqual({ pruned: 31, candidates: 56 });
    expect(r.steps.map((s) => [s.step, s.label, s.tool, s.successRate])).toEqual([
      ["open_lid", "Open lid", "Parallel jaw, 90 mm stroke", 1],
      ["lift_carton", "Lift carton", "Suction cup, 40 mm", 0.875],
    ]);
    expect(r.reportUrl).toContain("X-Signature");
  });

  it("also reads the first contract's shapes: tools with a family, a number for ruled_out, step labels", () => {
    const r = parseRun({ status: "done", created_at: null, cost_estimate_usd: null, error: null, result: SUMMARY_OBJECTS }).result!;
    expect(r.pick?.tools[1]).toEqual({ label: "Parallel jaw, 90 mm stroke", family: "Parallel jaw" });
    expect(r.ruledOut).toEqual({ pruned: 5, candidates: null });
    expect(r.steps[0]).toMatchObject({ label: "Open the lid", successRate: 1 });
  });

  it("reads a run with no recommendation: no pick, the closest setup, a step with no tool", () => {
    const r = parseRun({ status: "done", created_at: null, cost_estimate_usd: null, error: null, result: SUMMARY_NO_PICK }).result!;
    expect(r.pick).toBeNull();
    expect(r.closest?.label).toBe("Parallel jaw, 90 mm stroke");
    expect(r.completion).toEqual({ k: 0, n: 8 });
    expect(r.baseline).toBeNull();
    expect(r.steps[1]).toMatchObject({ step: "close_lid", tool: null, successRate: null });
    expect(r.why).toContain("no recommendation");
  });

  it("accepts a run with no baseline and a run that measured nothing", () => {
    const run = parseRun({ status: "done", created_at: null, cost_estimate_usd: null, error: null, result: { ...SUMMARY, baseline: null } });
    expect(run.result?.baseline).toBeNull();
    const none = parseRun({ status: "done", result: { pick: null, closest: null, completion: null, why: "Nothing ran.", baseline: null, ruled_out: {}, steps: [] } }).result!;
    expect(none).toMatchObject({ pick: null, closest: null, completion: null, why: "Nothing ran.", ruledOut: null, steps: [] });
  });

  it("refuses a done run with no result object", () => {
    expect(() => parseRun({ status: "done", result: null })).toThrow();
  });

  it("links the report through the runner, so the link never goes stale, and keeps http(s) only", () => {
    // report_url is a presigned link that lasts 15 minutes; GET /runs/{id}/report redirects to a fresh one
    expect(reportLink(BASE, "abc234", "https://files.example/r.json?sig=1")).toBe("https://runner.example/api/runs/abc234/report");
    expect(reportLink(BASE, "abc234", "/reports/r1.json")).toBe("https://runner.example/api/runs/abc234/report");
    expect(reportLink(BASE, "abc234", "javascript:alert(1)")).toBeNull();
    expect(reportLink(BASE, "abc234", "data:text/plain,x")).toBeNull();
    expect(reportLink(BASE, "abc234", null)).toBeNull();
  });

  it("strikes a key out of a message", () => {
    expect(redact("rejected nvapi-SECRET-123 as a key", "nvapi-SECRET-123")).toBe("rejected [key] as a key");
    expect(redact("short key", "key")).toBe("short key");
  });
});

describe("the health answer", () => {
  it("reads the numbers, the kill switch and the new open field", () => {
    expect(parseHealth(HEALTH)).toEqual({
      ok: true,
      killSwitch: false,
      open: true,
      autoOff: "2026-12-15",
      plansLeft: 3,
      runsLeft: 9,
      spendTodayUsd: 0.4,
      spendTotalUsd: 3.1,
    });
    expect(parseHealth({ ...HEALTH, open: false }).open).toBe(false);
  });

  it("has open as null when the runner sends none (an older runner) or sends something that is not a boolean", () => {
    const { open: _open, ...older } = HEALTH;
    expect(parseHealth(older).open).toBeNull();
    for (const v of ["false", "no", 0, 1, null, {}, []]) expect(parseHealth({ ...HEALTH, open: v }).open, JSON.stringify(v)).toBeNull();
    expect(parseHealth("nonsense")).toMatchObject({ ok: false, open: null, killSwitch: false });
  });
});

describe("when the runner is off", () => {
  const at = (iso: string) => Date.parse(iso);
  const health = (over: Record<string, unknown>) => parseHealth({ ...HEALTH, ...over });
  const older = (over: Record<string, unknown>) => {
    const { open: _open, ...rest } = HEALTH;
    return parseHealth({ ...rest, ...over });
  };
  const NOW = at("2026-10-28T10:00:00Z");

  it("is on for a runner that is open", () => {
    expect(offReason(health({}), NOW)).toBeNull();
  });

  it("is off when the kill switch is on, whatever else it says", () => {
    expect(offReason(health({ kill_switch: true }), NOW)).toBe("kill_switch");
    expect(offReason(health({ kill_switch: true, open: true }), NOW)).toBe("kill_switch");
    expect(offReason(older({ kill_switch: true }), NOW)).toBe("kill_switch");
  });

  it("is off when the runner says open is false, even with a future auto-off date and the kill switch off", () => {
    expect(offReason(health({ open: false }), NOW)).toBe("closed");
    expect(offReason(health({ open: false, auto_off: "2099-01-01" }), NOW)).toBe("closed");
  });

  it("takes the runner's word when it says open, even past the auto-off date", () => {
    expect(offReason(health({ open: true, auto_off: "2026-01-01" }), NOW)).toBeNull();
  });

  it("with no open field, is off from the day after the auto-off date (UTC), and on through the date itself", () => {
    expect(offReason(older({ auto_off: "2026-10-27" }), NOW)).toBe("auto_off");
    expect(offReason(older({ auto_off: "2026-10-28" }), NOW)).toBeNull();
    expect(offReason(older({ auto_off: "2026-10-29" }), NOW)).toBeNull();
    // the same date either side of UTC midnight, not the visitor's local midnight
    expect(offReason(older({ auto_off: "2026-12-15" }), at("2026-12-15T23:59:59Z"))).toBeNull();
    expect(offReason(older({ auto_off: "2026-12-15" }), at("2026-12-16T00:00:00Z"))).toBe("auto_off");
    expect(offReason(older({ auto_off: "2026-12-15" }), at("2026-12-16T00:00:00+05:00"))).toBeNull();
  });

  it("with no open field, reads a date-time auto-off by its day, and ignores a date it cannot read", () => {
    expect(offReason(older({ auto_off: "2026-10-27T23:59:59Z" }), NOW)).toBe("auto_off");
    expect(offReason(older({ auto_off: "2026-10-28T00:00:00Z" }), NOW)).toBeNull();
    for (const bad of [null, "", "soon", "15 December 2026", "2026-13-45", "2026-1-5"]) expect(offReason(older({ auto_off: bad }), NOW), String(bad)).toBeNull();
  });

  it("uses the clock it is given, and the real one when it is given none", () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(at("2027-01-01T00:00:00Z"));
      expect(offReason(older({ auto_off: "2026-12-15" }))).toBe("auto_off");
      expect(autoOffPassed("2026-12-15")).toBe(true);
      vi.setSystemTime(at("2026-12-15T12:00:00Z"));
      expect(offReason(older({ auto_off: "2026-12-15" }))).toBeNull();
      expect(autoOffPassed("2026-12-15")).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("a redirect is never followed (the key rides in the body of POST /plan)", () => {
  const servers: Server[] = [];
  afterEach(async () => {
    await Promise.all(servers.splice(0).map((s) => new Promise<void>((done) => s.close(() => done()))));
  });

  const listen = (handler: (req: IncomingMessage, body: string) => { status: number; headers?: Record<string, string>; body?: string }): Promise<{ origin: string; hits: { method: string; url: string; body: string }[] }> =>
    new Promise((resolve) => {
      const hits: { method: string; url: string; body: string }[] = [];
      const server = createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          hits.push({ method: req.method ?? "", url: req.url ?? "", body });
          const out = handler(req, body);
          res.writeHead(out.status, { "content-type": "application/json", ...(out.headers ?? {}) });
          res.end(out.body ?? "{}");
        });
      });
      servers.push(server);
      server.listen(0, "127.0.0.1", () => resolve({ origin: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, hits }));
    });

  const KEY = "nvapi-SECRET-KEY-0123456789";

  it("POST /plan answered with a 307 fails as a network error and the other address gets nothing", async () => {
    const elsewhere = await listen(() => ({ status: 200, body: JSON.stringify({ status: "valid", plan: VALID_PLAN, problems: [], calls: 1, cost_usd: 0, model: "m" }) }));
    for (const status of [301, 302, 307, 308]) {
      const runner = await listen(() => ({ status, headers: { location: `${elsewhere.origin}/plan` } }));
      const err = await postPlan(runner.origin, { description: "d", provider: "nvidia", key: KEY }).catch((e: unknown) => e);
      expect(err, String(status)).toBeInstanceOf(RunnerError);
      expect(err, String(status)).toMatchObject({ kind: "network", message: MESSAGES.network });
      expect(String((err as Error).message)).not.toContain(KEY);
      // the runner saw the request once, with the key in its body; nothing followed it
      expect(runner.hits.map((h) => `${h.method} ${h.url}`), String(status)).toEqual(["POST /plan"]);
    }
    expect(elsewhere.hits).toEqual([]);
  });

  it("the other runner calls do not follow a redirect either", async () => {
    const elsewhere = await listen(() => ({ status: 200, body: JSON.stringify(HEALTH) }));
    const runner = await listen(() => ({ status: 302, headers: { location: `${elsewhere.origin}/health` } }));
    await expect(getHealth(runner.origin)).rejects.toMatchObject({ kind: "network" });
    await expect(postRun(runner.origin, VALID_PLAN)).rejects.toMatchObject({ kind: "network" });
    await expect(getRun(runner.origin, "r1")).rejects.toMatchObject({ kind: "network" });
    expect(elsewhere.hits).toEqual([]);
  });

  it("a plain answer still works through the real fetch, with the key in the one POST body only", async () => {
    const runner = await listen((req) =>
      req.url === "/health"
        ? { status: 200, body: JSON.stringify(HEALTH) }
        : { status: 200, body: JSON.stringify({ status: "valid", plan: VALID_PLAN, problems: [], calls: 1, cost_usd: 0.01, model: "m" }) },
    );
    expect((await getHealth(runner.origin)).open).toBe(true);
    expect((await postPlan(runner.origin, { description: "d", provider: "nvidia", key: KEY })).status).toBe("valid");
    expect(runner.hits.filter((h) => h.body.includes(KEY)).map((h) => `${h.method} ${h.url}`)).toEqual(["POST /plan"]);
    expect(runner.hits.some((h) => h.url.includes(KEY))).toBe(false);
  });
});

describe("the privacy line", () => {
  it("is the plain sentence", () => {
    expect(KEY_PRIVACY_LINE).toBe("Your key is sent once over HTTPS for this request and is never stored or logged.");
  });
});
