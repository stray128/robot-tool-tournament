#!/usr/bin/env node
/**
 * A tiny mock of the hosted runner's API, for the browser smoke tests and for trying the panel by hand.
 * It follows the contract the Studio's "Plan and run it live" panel is written against:
 *
 *   GET  /health        {ok, kill_switch, open, auto_off, plans_left_today, runs_left_today, spend_today_usd, spend_total_usd}
 *                       (`open` is false once the runner has switched itself off, and the answer is still 200;
 *                       an older runner has no `open` field: see the `health` switch below)
 *   POST /plan          {description, provider, key} -> {status, plan, problems, calls, cost_usd, model}
 *   POST /runs          {plan} -> {run_id, status: "queued", eta_s}
 *   GET  /runs/{id}     -> {status, created_at, cost_estimate_usd, error, result}
 *   GET  /runs/{id}/report    302 to a fresh presigned link of the full report (here: /reports/{id}.json)
 *   OPTIONS *           CORS preflight (the site is on another origin)
 *
 * Bodies, caps and the result's shape follow docs/RUNNER.md and src/rtt/runner/: errors are
 * {"error": ...}, a 429 names the cap, its limit and when it resets, a 503 says in the runner's own
 * words which of its 503s it is (see ERRORS_503), POST /runs answers 202 with
 * eta_s 480, a result's tools are plain names, `ruled_out` is an object of counts and a step row
 * has no label.
 *
 * A run is queued when posted, then "running" on its first status check and "done" (or "failed") on
 * the second, so a test with the panel's 10 s polling sees every state.
 *
 *   node scripts/mock_runner.mjs [--port 8765]
 *
 * `POST /__mock` with a JSON object merges it into the behaviour switches (`state`), for example
 * {"plan": "429"} or {"health": "off"}. In-process users (smoke_runner.mjs) change `state` directly.
 * Everything is test data: the plans are the recorded planner plans, the results are made up.
 *
 * The mock never prints or stores keys beyond `state.requests`, which a test reads to prove the key
 * was only ever in a /plan body.
 */

import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const here = path.dirname(fileURLToPath(import.meta.url));
const recorded = JSON.parse(readFileSync(path.join(here, "..", "components", "studio", "intake", "recorded.json"), "utf8"));
const planDoc = (expect) => recorded.examples.find((e) => e.expect === expect && e.plan)?.plan ?? null;

/**
 * The behaviour switches. Values:
 *   health ok | off (kill switch) | closed (open: false, after the auto-off date) | lapsed (an older runner: no `open` field, an auto-off date in the past)
 *          | legacy (an older runner: no `open` field, an auto-off date in the future) | bad | down | drop;
 *   plan valid|failed|refused|401|429|busy (a 429 with no cap: the hosting layer throttling)|400|drop
 *        | 503 (the kill switch: off) | 503_closed (the auto-off date: off) | 503_free (the free tier is down)
 *        | 503_unconfigured | 503_bare (no words: the hosting layer);
 *   run ok|400|429|busy|drop
 *        | 503 (the kill switch: off) | 503_closed | 503_job (the simulator could not start a job)
 *        | 503_backend | 503_platform | 503_unconfigured | 503_bare;
 *   poll ok|drop|404; ending done|failed|no_pick.
 */
export function defaultState() {
  return {
    health: "ok",
    plan: "valid",
    run: "ok",
    poll: "ok",
    /** How the run ends: done (a recommendation), no_pick (done, nothing completed the task) or failed. */
    ending: "done",
    planDelayMs: 0,
    plansLeft: 3,
    runsLeft: 9,
    runs: new Map(),
    requests: [],
  };
}

const RESET = "2026-10-29T00:00:00Z";

/** The runner's own 503 bodies (src/rtt/runner/app.py), by the switch value that makes the mock send one. Only the first two say it is off. */
const ERRORS_503 = {
  503: { error: "the runner is switched off for now; recorded runs still replay" },
  "503_closed": { error: "the hosted runner closed after 2026-12-15; recorded runs still replay" },
  "503_free": { error: "the free tier is unavailable right now; try your own key" },
  "503_job": { error: "the simulation job could not be started; try again later" },
  "503_backend": { error: "the simulation backend is unavailable; nothing was reserved, try again later" },
  "503_platform": { error: "the platform refused the simulation job; this did not count against your runs" },
  "503_unconfigured": { error: "runs are not configured on this runner" },
  // the hosting layer answers with no words of the runner's
  "503_bare": {},
};
const error503 = (value) => (Object.hasOwn(ERRORS_503, value) ? ERRORS_503[value] : null);

/** The result as the runner builds it (src/rtt/runner/summary.py). */
function summary(id, origin, { noPick = false } = {}) {
  const jaw = "Parallel jaw, 90 mm stroke";
  const base = {
    pick: {
      setup_id: "d136ab4824c1",
      label: "Suction cup plus parallel jaw",
      kind: "multi_head",
      tools: ["Suction cup, 40 mm", jaw],
    },
    closest: null,
    completion: { k: 7, n: 8 },
    why: "The suction cup lifts the carton and the jaw closes the lid; no single tool does both steps.",
    baseline: { setup_id: "a1b2c3d4e5f6", label: jaw, kind: "single", tools: [jaw], completion: { k: 2, n: 8 }, basis: "R01" },
    ruled_out: { candidates: 56, pruned: 31, pruned_for_some_steps: 4, screened: 25, setups_rejected: 12, finalists: 4, finalists_not_eligible: 1 },
    steps: [
      { step: "open_lid", tool: jaw, success_rate: 1 },
      { step: "pick_carton", tool: "Suction cup, 40 mm", success_rate: 0.875 },
      { step: "pick_pack", tool: "Suction cup, 40 mm", success_rate: 0.875 },
      { step: "close_lid", tool: jaw, success_rate: 1 },
    ],
    code_version: "3f9c2d7a41be",
    report_url: `${origin}/reports/${id}.json?X-Signature=mock`,
  };
  if (!noPick) return base;
  return {
    ...base,
    pick: null,
    closest: { setup_id: "f00dfeed0001", label: jaw, kind: "single", tools: [jaw] },
    completion: { k: 0, n: 8 },
    why: "No setup closed the lid in any run, so there is no recommendation. The closest setup is shown.",
    baseline: null,
    steps: [
      { step: "open_lid", tool: jaw, success_rate: 1 },
      { step: "pick_carton", tool: jaw, success_rate: 0.5 },
      { step: "pick_pack", tool: jaw, success_rate: 0 },
      { step: "close_lid", tool: null, success_rate: null },
    ],
  };
}

export async function startMockRunner({ port = 0, state = defaultState() } = {}) {
  let counter = 0;
  const send = (res, status, body, extra = {}) => {
    res.writeHead(status, {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "content-type",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "cache-control": "no-store",
      ...extra,
    });
    res.end(body === null ? "" : JSON.stringify(body));
  };
  const readBody = (req) =>
    new Promise((resolve) => {
      const chunks = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://mock");
    const body = req.method === "POST" ? await readBody(req) : "";
    if (req.method === "OPTIONS") return send(res, 204, null);
    if (url.pathname === "/__mock" && req.method === "POST") {
      Object.assign(state, JSON.parse(body || "{}"));
      return send(res, 200, { ok: true });
    }
    state.requests.push({ method: req.method, url: req.url, headers: { ...req.headers }, body });

    if (url.pathname === "/health") {
      if (state.health === "down") return send(res, 503, ERRORS_503[503]);
      if (state.health === "drop") return req.socket.destroy();
      // The date is fixed test data. Only the two older-runner values depend on it, and they use dates that are far from today.
      const older = state.health === "lapsed" || state.health === "legacy";
      return send(res, 200, {
        ok: state.health !== "bad",
        kill_switch: state.health === "off",
        ...(older ? {} : { open: state.health !== "off" && state.health !== "closed" }),
        auto_off: state.health === "lapsed" ? "2026-01-01" : state.health === "legacy" ? "2099-12-31" : "2026-12-15",
        plans_left_today: state.plansLeft,
        runs_left_today: state.runsLeft,
        spend_today_usd: 0.42,
        spend_total_usd: 3.1,
      });
    }

    if (url.pathname === "/plan" && req.method === "POST") {
      if (state.planDelayMs) await sleep(state.planDelayMs);
      let input;
      try {
        input = JSON.parse(body);
      } catch {
        return send(res, 400, { error: "bad json" });
      }
      if (state.plan === "drop") return req.socket.destroy();
      if (error503(state.plan)) return send(res, 503, error503(state.plan));
      if (state.plan === "busy") return send(res, 429, { message: "Too Many Requests" });
      if (state.plan === "429") return send(res, 429, { error: "cap reached: the free planning tier's daily limit per visitor; it resets at " + RESET, cap: "free_plans_per_ip_per_day", limit: 3, resets_at: RESET });
      if (state.plan === "400") return send(res, 400, { error: "description too long" });
      if (!["nvidia", "tokenfactory", "free"].includes(input.provider) || typeof input.description !== "string") return send(res, 400, { error: "bad request" });
      const needsKey = input.provider !== "free";
      if (state.plan === "401" || (needsKey && (typeof input.key !== "string" || !input.key || input.key === "bad-key"))) return send(res, 401, { error: "the provider rejected the key" });
      if (state.plan === "failed") {
        return send(res, 200, { status: "failed", plan: null, problems: ["steps[1]: the tote lid hinge limit is above 120 degrees"], calls: 2, cost_usd: 0.012, model: "nemotron-mock" });
      }
      if (state.plan === "refused") {
        return send(res, 200, { status: "refused", plan: planDoc("refusal"), problems: ["Folding cloth is not simulated."], calls: 1, cost_usd: 0.006, model: "nemotron-mock" });
      }
      if (!needsKey) state.plansLeft = Math.max(0, state.plansLeft - 1);
      return send(res, 200, { status: "valid", plan: planDoc("plan"), problems: [], calls: 1, cost_usd: 0.008, model: "nemotron-mock" });
    }

    if (url.pathname === "/runs" && req.method === "POST") {
      if (state.run === "drop") return req.socket.destroy();
      if (error503(state.run)) return send(res, 503, error503(state.run));
      if (state.run === "busy") return send(res, 429, { message: "Too Many Requests" });
      if (state.run === "429") return send(res, 429, { error: "cap reached: the daily limit on runs per visitor; it resets at " + RESET, cap: "runs_per_ip_per_day", limit: 2, resets_at: RESET });
      let input;
      try {
        input = JSON.parse(body);
      } catch {
        return send(res, 400, { error: "bad json" });
      }
      if (!input || typeof input.plan !== "object" || input.plan === null) return send(res, 400, { error: "plan missing" });
      if (state.run === "400" || (input.plan.steps ?? []).length > 8) return send(res, 400, { error: "the plan failed the checks", problems: ["steps: a hosted run takes at most 8 steps"] });
      const id = `r${Date.now().toString(36)}${(++counter).toString(36)}`;
      state.runs.set(id, { polls: 0, created_at: new Date().toISOString() });
      state.runsLeft = Math.max(0, state.runsLeft - 1);
      return send(res, 202, { run_id: id, status: "queued", eta_s: 480 });
    }

    const run = /^\/runs\/([A-Za-z0-9_-]+)$/.exec(url.pathname);
    if (run && req.method === "GET") {
      if (state.poll === "drop") return req.socket.destroy();
      if (state.poll === "404") return send(res, 404, { error: "no such run" });
      const rec = state.runs.get(run[1]) ?? (run[1].startsWith("r") ? { polls: 1, created_at: new Date(Date.now() - 65_000).toISOString() } : null);
      if (!rec) return send(res, 404, { error: "no such run" });
      state.runs.set(run[1], rec);
      rec.polls += 1;
      const base = { created_at: rec.created_at, cost_estimate_usd: 0.18, error: null, result: null };
      if (rec.polls < 2) return send(res, 200, { ...base, status: "running" });
      if (state.ending === "failed") return send(res, 200, { ...base, status: "failed", error: "The physics worker ran out of time on step 3." });
      return send(res, 200, { ...base, status: "done", result: summary(run[1], `http://${req.headers.host}`, { noPick: state.ending === "no_pick" }) });
    }

    const redirect = /^\/runs\/([A-Za-z0-9_-]+)\/report$/.exec(url.pathname);
    if (redirect && req.method === "GET") {
      if (!state.runs.has(redirect[1])) return send(res, 404, { error: "no such run" });
      return send(res, 302, null, { location: `/reports/${redirect[1]}.json?X-Signature=fresh` });
    }
    const report = /^\/reports\/([A-Za-z0-9_-]+)\.json$/.exec(url.pathname);
    if (report && req.method === "GET") return send(res, 200, { run_id: report[1], note: "mock report", summary: summary(report[1], `http://${req.headers.host}`) });

    return send(res, 404, { error: "not found" });
  });

  await new Promise((resolve) => server.listen(port, "127.0.0.1", resolve));
  const actual = server.address().port;
  return {
    server,
    state,
    port: actual,
    url: `http://127.0.0.1:${actual}`,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { port: { type: "string", default: "8765" } } });
  const mock = await startMockRunner({ port: Number(values.port) });
  console.log(`mock runner at ${mock.url} (build the site with NEXT_PUBLIC_RUNNER_URL=${mock.url})`);
}
