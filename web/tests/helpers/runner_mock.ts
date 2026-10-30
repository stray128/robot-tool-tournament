/** A scripted fetch for the hosted runner tests: no network, every call recorded. */

import { vi } from "vitest";

import recordedJson from "@/components/studio/intake/recorded.json";

export const BASE = "https://runner.example/api";

interface Recorded {
  examples: { case_id: string; expect: string; status: string; plan: Record<string, unknown> | null }[];
}
const examples = (recordedJson as unknown as Recorded).examples;

/** A valid plan document and a refusal, as the planner's recorded runs carry them. */
export const VALID_PLAN = examples.find((e) => e.expect === "plan" && e.status === "valid" && e.plan)!.plan!;
export const REFUSAL_PLAN = examples.find((e) => e.expect === "refusal" && e.plan)!.plan!;

/** The result as the runner builds it (src/rtt/runner/summary.py): tools are plain names, `ruled_out` is an object of counts, a step row has no label. */
export const SUMMARY = {
  pick: {
    setup_id: "d136ab48",
    label: "Suction cup plus parallel jaw",
    kind: "multi_head",
    tools: ["Suction cup, 40 mm", "Parallel jaw, 90 mm stroke"],
  },
  closest: null,
  completion: { k: 7, n: 8 },
  why: "The suction cup lifts the carton and the jaw closes the lid.",
  baseline: { setup_id: "a1b2c3d4", label: "Parallel jaw, 90 mm stroke", kind: "single", tools: ["Parallel jaw, 90 mm stroke"], completion: { k: 2, n: 8 }, basis: "R01" },
  ruled_out: { candidates: 56, pruned: 31, pruned_for_some_steps: 4, screened: 25, setups_rejected: 12, finalists: 4, finalists_not_eligible: 1 },
  steps: [
    { step: "open_lid", tool: "Parallel jaw, 90 mm stroke", success_rate: 1 },
    { step: "lift_carton", tool: "Suction cup, 40 mm", success_rate: 0.875 },
  ],
  code_version: "3f9c2d7a41be",
  report_url: "https://files.example/runs/r1/report.json?X-Signature=abc",
};

/** The same result in the shape the first contract described: tools with a family, a number for `ruled_out`, step labels. */
export const SUMMARY_OBJECTS = {
  ...SUMMARY,
  pick: { label: "Suction cup plus parallel jaw", kind: "two_tool_head", tools: [{ label: "Suction cup, 40 mm", family: "Suction cup" }, { label: "Parallel jaw, 90 mm stroke", family: "Parallel jaw" }] },
  baseline: { label: "Parallel jaw, 90 mm stroke", completion: { k: 2, n: 8 } },
  ruled_out: 5,
  steps: [{ step: "open_lid", label: "Open the lid", tool: "Parallel jaw, 90 mm stroke", success_rate: 1 }],
};

/** A run in which no setup completed the task: no pick, the closest setup, and why. */
export const SUMMARY_NO_PICK = {
  ...SUMMARY,
  pick: null,
  closest: { setup_id: "f00dfeed", label: "Parallel jaw, 90 mm stroke", kind: "single", tools: ["Parallel jaw, 90 mm stroke"] },
  completion: { k: 0, n: 8 },
  why: "No setup closed the lid in any run, so there is no recommendation.",
  baseline: null,
  steps: [
    { step: "open_lid", tool: "Parallel jaw, 90 mm stroke", success_rate: 1 },
    { step: "close_lid", tool: null, success_rate: null },
  ],
};

export const HEALTH = {
  ok: true,
  kill_switch: false,
  open: true,
  auto_off: "2026-12-15",
  plans_left_today: 3,
  runs_left_today: 9,
  spend_today_usd: 0.4,
  spend_total_usd: 3.1,
};

export type Handler = (url: string, init: RequestInit | undefined) => Response | Promise<Response>;

export const json = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/** A fetch mock that answers by `METHOD /path` (after the base); anything else is a 404. A route may be a function or a queue of replies. */
export function scriptedFetch(routes: Record<string, Handler | Response | (Response | Error)[]>) {
  const queues = new Map<string, (Response | Error)[]>();
  const fn = vi.fn(async (input: string, init?: RequestInit) => {
    const path = input.startsWith(BASE) ? input.slice(BASE.length) : input;
    const key = `${init?.method ?? "GET"} ${path}`;
    const route = routes[key];
    if (route === undefined) return json({ detail: "no route" }, 404);
    if (Array.isArray(route)) {
      const q = queues.get(key) ?? [...route];
      queues.set(key, q);
      const next = q.length > 1 ? q.shift()! : q[0]!;
      if (next instanceof Error) throw next;
      return next.clone();
    }
    if (route instanceof Response) return route.clone();
    return route(input, init);
  });
  return fn;
}

export const networkError = () => new TypeError("fetch failed");

/** Every place the page could leak a key, as spies. */
export function leakSpies() {
  const store = () => ({ getItem: vi.fn(() => null), setItem: vi.fn(), removeItem: vi.fn(), clear: vi.fn(), key: vi.fn(), length: 0 });
  const localStorage = store();
  const sessionStorage = store();
  // replaceState moves the stub's hash the way a browser does, so a clear after a write is real
  const history = {
    state: null,
    pushState: vi.fn(),
    replaceState: vi.fn((_state: unknown, _title: string, url?: string) => {
      if (typeof url === "string") win.location.hash = url.startsWith("#") ? url : "";
    }),
  };
  const win = { location: { hash: "", pathname: "/robot-tool-tournament/studio/", search: "" }, history, localStorage, sessionStorage };
  vi.stubGlobal("window", win);
  vi.stubGlobal("localStorage", localStorage);
  vi.stubGlobal("sessionStorage", sessionStorage);
  vi.stubGlobal("history", history);
  const consoles = (["log", "info", "warn", "error", "debug", "trace"] as const).map((m) => vi.spyOn(console, m).mockImplementation(() => {}));
  return { localStorage, sessionStorage, history, win, consoles };
}

/** Everything the spies saw, as one string. */
export function leaked(spies: ReturnType<typeof leakSpies>): string {
  const calls = [
    spies.localStorage.setItem.mock.calls,
    spies.sessionStorage.setItem.mock.calls,
    spies.history.pushState.mock.calls,
    spies.history.replaceState.mock.calls,
    ...spies.consoles.map((c) => c.mock.calls),
  ];
  return JSON.stringify(calls);
}
