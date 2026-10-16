import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { findLocalPaths, publishableSession, withPackPaths } from "../scripts/import-session.mjs";
import { demoSession } from "../scripts/make-demo-session.mjs";
import { getSessions } from "@/lib/data";
import {
  activeJobs,
  applyEvent,
  buildTimeline,
  emptyTimeline,
  eventsFromChatMessages,
  inspectView,
  jobStatus,
  normalizeEvent,
  normalizeEvents,
  parseRecordedSession,
  playbackTimes,
  playedCount,
  printpackView,
  proposeView,
  readJob,
  readLedger,
  resultsView,
  SessionError,
  spend,
  stressView,
  summarizeSession,
} from "@/lib/session";

const DATA = path.join(__dirname, "..", "public", "data");
const fixture = (name: string) => JSON.parse(readFileSync(path.join(__dirname, "fixtures", name), "utf8"));

const kindsOf = (events: ReturnType<typeof normalizeEvents>["events"]) =>
  buildTimeline(events)
    .steps.filter((s) => s.kind !== "message")
    .map((s) => s.kind);

describe("normalizeEvent: the agent's own format", () => {
  it("reads dotted types, `name` and `t_s`-style fields", () => {
    expect(normalizeEvent({ type: "tool.call", call_id: "call_1", name: "inspect_part", args: { part_id: "p" }, n: 1 }, 2)).toEqual({
      type: "tool_call",
      t: 2,
      call_id: "call_1",
      tool: "inspect_part",
      args: { part_id: "p" },
    });
    expect(normalizeEvent({ type: "tool.result", call_id: "call_1", name: "inspect_part", ok: true, result: { part: {} } }, 3)).toMatchObject({
      type: "tool_result",
      ok: true,
      refused: false,
      error: null,
    });
  });

  it("marks refusals and names the cap", () => {
    const ev = normalizeEvent({ type: "tool.refused", call_id: "c", name: "launch_sim_job", ok: false, code: "concurrency" }, 1);
    expect(ev).toMatchObject({ type: "tool_result", ok: false, refused: true, error: "Refused: concurrency" });
  });

  it("reads llm calls, summaries and the finish event with its spend", () => {
    expect(normalizeEvent({ type: "llm.call", role: "proposer", model: "m", prompt_tokens: 10, completion_tokens: 5, cost_usd: 0.01, latency_s: 2, cached: true }, 0)).toMatchObject({
      type: "llm_call",
      role: "proposer",
      tokens_in: 10,
      cost_usd: 0, // a cached call costs nothing
    });
    expect(normalizeEvent({ type: "summary", step: "inspect_part", text: "Fits." }, 1)).toEqual({ type: "message", t: 1, role: "summary", text: "Fits." });
    expect(normalizeEvent({ type: "agent.message", content: "", tool_calls: [{ name: "x" }] }, 1)).toBeNull();
    const end = normalizeEvent(
      { type: "session.finished", status: "stopped", reason: "BudgetExceededError", answer: "a", spend: { spent_usd: 0.2, session_cap_usd: 0.5, by_kind_usd: { llm: 0.05, nebius: 0.15 } } },
      9,
    );
    expect(end).toMatchObject({ type: "session_ended", status: "stopped", answer: "a", reason: "BudgetExceededError" });
    expect(end && end.type === "session_ended" ? end.ledger : null).toMatchObject({ llm_usd: 0.05, compute_usd: 0.15, total_usd: 0.2, budget_usd: 0.5 });
  });

  it("reads flat job records with target, state and an estimated cost", () => {
    const ev = normalizeEvent({ type: "job.update", job_id: "j", target: "nebius", state: "RUNNING", designs: ["a", "b"], elapsed_s: 30, est_usd: 0.02, platform: "cpu-d3" }, 4);
    expect(ev && ev.type === "job_update" ? ev.job : null).toMatchObject({
      id: "j",
      backend: "nebius",
      status: "running",
      designs: 2,
      cost_usd: 0.02,
      cost_estimated: true,
    });
    expect(readJob({ job_id: "j", target: "local", state: "COMPLETED" }, 0)).toMatchObject({ backend: "local", status: "completed", cost_estimated: false });
  });
});

describe("normalizeEvent: aliases", () => {
  it("accepts other spellings, JSON-encoded arguments and the SSE event name", () => {
    const ev = normalizeEvent({ tool: "propose_designs", tool_call_id: "x", arguments: '{"n": 4}' }, 1, "tool_use");
    expect(ev).toMatchObject({ type: "tool_call", call_id: "x", tool: "propose_designs", args: { n: 4 } });
    const res = normalizeEvent({ event: "tool_output", id: "x", output: '{"designs": []}', status: "error" }, 3);
    expect(res).toMatchObject({ type: "tool_result", call_id: "x", ok: false, result: { designs: [] } });
    expect(normalizeEvent({ type: "done", reason: "budget_exceeded" }, 9)).toMatchObject({ type: "session_ended", status: "capped" });
  });

  it("drops what it cannot read instead of throwing", () => {
    expect(normalizeEvent(null, 0)).toBeNull();
    expect(normalizeEvent("not json", 0)).toBeNull();
    expect(normalizeEvent({ type: "telemetry" }, 0)).toBeNull();
    expect(normalizeEvent({ type: "tool.call" }, 0)).toBeNull();
    expect(jobStatus("SUCCEEDED")).toBe("completed");
    expect(jobStatus("in progress")).toBe("running");
    expect(jobStatus("weird")).toBe("unknown");
  });
});

describe("normalizeEvents", () => {
  it("prefers t_s, then absolute timestamps, and keeps order stable", () => {
    const { events, dropped } = normalizeEvents([
      { type: "summary", text: "a", ts: "2026-10-15T10:00:00Z" },
      { type: "bogus" },
      { type: "summary", text: "b", ts: "2026-10-15T10:00:05Z" },
      { type: "summary", text: "c", ts: "2026-10-15T10:00:05Z", t_s: 5 },
      { type: "agent.message", content: "" },
    ]);
    expect(dropped).toBe(1); // the empty agent turn is skipped on purpose, not dropped
    expect(events.map((e) => [e.t, e.type === "message" ? e.text : ""])).toEqual([
      [0, "a"],
      [5, "b"],
      [5, "c"],
    ]);
  });

  it("converts the agent's chat transcript, unwrapping {ok, result}", () => {
    const { events } = normalizeEvents(eventsFromChatMessages(fixture("agent-chat-offline.json").messages));
    const tl = buildTimeline(events);
    const inspect = tl.steps.find((s) => s.kind === "inspect");
    expect(inspect?.status).toBe("ok");
    expect(inspectView(inspect?.result).extents_mm).toEqual([84, 40, 40]);
  });
});

describe("a recorded offline session (src/rtt/agent, --offline --fake-physics)", () => {
  const session = parseRecordedSession(fixture("agent-session-offline.json"), "sessions/x.json");
  const tl = buildTimeline(session.events);

  it("parses with nothing dropped", () => {
    expect(session).toMatchObject({ id: "sess-82d7e038aa5b", kind: "recorded", offline: true, part: { name: "dumbbell-handle" } });
    expect(session.dropped).toBe(0);
    expect(session.recorded_at).toBe("2026-10-15T17:31:37.871Z");
  });

  it("builds the timeline the agent ran", () => {
    expect(kindsOf(session.events)).toEqual(["inspect", "propose", "job", "results", "stress", "printpack", "ledger"]);
    expect(tl).toMatchObject({ status: "completed", tool_calls: 7, offline: true, printpack_id: "pack-5037bcc6f6f5" });
    expect(tl.llm.calls).toBe(14);
    expect(Object.values(tl.jobs)).toEqual([expect.objectContaining({ id: "local-037a898ea474", backend: "local", status: "completed" })]);
    expect(tl.answer).toMatch(/Recommended design c84efeac4d3c/);
    expect(spend(tl).total_usd).toBeCloseTo(0.005595, 6);
  });

  it("reads every tool's result", () => {
    const step = (kind: string) => tl.steps.find((s) => s.kind === kind)!;
    expect(inspectView(step("inspect").result)).toMatchObject({ grip_width_mm: 40, fits_opening: true, opening_mm: 84, convex_pieces: 1, pass_threshold: 0.8 });
    expect(inspectView(step("inspect").result).baselines.length).toBeGreaterThan(0);
    const p = proposeView(step("propose").result, step("propose"));
    expect(p.designs).toHaveLength(3);
    expect(p.tokens_in).toBe(1159); // from the proposer's llm.call
    const r = resultsView(step("results").result);
    expect(r.best?.design_id).toBe("c84efeac4d3c");
    expect(r.best?.success).toBe(1);
    expect(r.pass_threshold).toBe(0.8);
    expect(stressView(step("stress").result)).toMatchObject({ score: 0.483, verdict: "fail", tolerance_mm: { x: 2, y: 1 }, passed: "10/18" });
    expect(printpackView(step("printpack").result)).toMatchObject({ printpack_id: "pack-5037bcc6f6f5", url: "/printpack/pack-5037bcc6f6f5", verdict: "pass" });
    expect(readLedger(step("ledger").result)).toMatchObject({ total_usd: 0.004718, budget_usd: 0.5, tool_calls: 7, max_tool_calls: 12 });
  });
});

describe("timeline", () => {
  const ev = (raw: object, t: number) => normalizeEvent(raw, t)!;

  it("gives a job update to the launch still waiting for it, and folds polls", () => {
    const tl = buildTimeline([
      ev({ type: "tool.call", call_id: "c1", name: "launch_sim_job", args: {} }, 0),
      ev({ type: "job.update", job_id: "j1", state: "QUEUED", target: "nebius", platform: "cpu-d3" }, 0.5),
      ev({ type: "tool.result", call_id: "c1", name: "launch_sim_job", ok: true, result: { job_id: "j1", state: "QUEUED" } }, 1),
      ev({ type: "tool.call", call_id: "c2", name: "poll_job", args: { job_id: "j1" } }, 5),
      ev({ type: "tool.result", call_id: "c2", name: "poll_job", ok: true, result: { job_id: "j1", state: "RUNNING", elapsed_s: 30 } }, 6),
    ]);
    expect(tl.steps).toHaveLength(1);
    expect(tl.steps[0]).toMatchObject({ kind: "job", status: "ok", polls: 1, job_ids: ["j1"] });
    expect(tl.tool_calls).toBe(2);
    expect(tl.jobs.j1).toMatchObject({ status: "running", platform: "cpu-d3", elapsed_s: 30, backend: "nebius" });
    expect(activeJobs(tl).map((j) => j.id)).toEqual(["j1"]);
  });

  it("charges proposer calls to the waiting propose card", () => {
    const tl = buildTimeline([
      ev({ type: "llm.call", role: "agent", model: "m", prompt_tokens: 5, completion_tokens: 1, cost_usd: 0.001 }, 0),
      ev({ type: "tool.call", call_id: "p", name: "propose_designs", args: {} }, 1),
      ev({ type: "llm.call", role: "proposer", model: "nemo", prompt_tokens: 100, completion_tokens: 20, cost_usd: 0.003, latency_s: 4 }, 2),
      ev({ type: "tool.result", call_id: "p", name: "propose_designs", ok: true, result: { designs: [] } }, 3),
    ]);
    expect(tl.steps[0]!.llm).toMatchObject({ calls: 1, tokens_in: 100, cost_usd: 0.003, model: "nemo" });
    expect(tl.llm.calls).toBe(2);
    expect(tl.model).toBe("nemo");
    expect(spend(tl).llm_usd).toBeCloseTo(0.004, 6);
  });

  it("closes running steps when the session ends", () => {
    let tl = emptyTimeline();
    tl = applyEvent(tl, ev({ type: "tool.call", call_id: "s1", name: "stress_test", args: {} }, 0));
    tl = applyEvent(tl, ev({ type: "session.finished", status: "failed" }, 3));
    expect(tl.status).toBe("failed");
    expect(tl.steps[0]!.status).toBe("error");
  });

  it("gives a result without its call a card of its own, and never reads a pack id as a job", () => {
    const tl = buildTimeline([
      ev({ type: "tool.result", call_id: "late", name: "fetch_results", ok: true, result: { designs: [] } }, 4),
      ev({ type: "tool.call", call_id: "p", name: "export_print_pack", args: {} }, 5),
      ev({ type: "tool.result", call_id: "p", name: "export_print_pack", ok: true, result: { pack_id: "pack-1", files: ["a.stl"] } }, 6),
    ]);
    expect(tl.steps[0]).toMatchObject({ id: "late", kind: "results", status: "ok" });
    expect(tl.jobs).toEqual({});
    expect(tl.printpack_id).toBe("pack-1");
  });
});

describe("tool views with other spellings", () => {
  it("read inspect, results, stress and print pack aliases", () => {
    const v = inspectView({ bbox_mm: [84, 40, 40], volume_mm3: 42000, n_hulls: 3, stable_poses: [{ prob: 0.9, label: "flat" }], fits: true });
    expect(v).toMatchObject({ extents_mm: [84, 40, 40], volume_cm3: 42, convex_pieces: 3, fits_opening: true });
    expect(inspectView("garbage").extents_mm).toBeNull();
    const r = resultsView({ results: [{ label: "a", success: 75 }, { label: "b", success_rate: 0.9, failure_modes: { slip_during_lift: 2, x: "n" } }] });
    expect(r.rows.map((x) => x.success)).toEqual([0.75, 0.9]);
    expect(r.rows[1]!.failure_modes).toEqual({ slip_during_lift: 2 });
    expect(r.best?.label).toBe("b");
    expect(stressView({ certificate: { score: 0.8, tolerance_mm: { x: 3, y: null }, category_scores: { shake: 1 } } })).toMatchObject({
      score: 0.8,
      tolerance_mm: { x: 3, y: null },
      categories: { placement: null, shake: 1, push: null },
    });
    expect(printpackView({ id: "p", files: ["out/finger-left.stl", { name: "c.html", bytes: 10 }] }).files).toEqual([
      { name: "finger-left.stl", path: null, bytes: null },
      { name: "c.html", path: null, bytes: 10 },
    ]);
    // The download path is appended to the API origin, so only "/printpack/<id>" is kept.
    for (const bad of ["@evil.example/x", "//evil.example/x", "/printpack/../runs/x", "javascript:alert(1)", "/printpack/a b"]) {
      expect(printpackView({ pack_id: "p", download: bad }).url).toBeNull();
    }
    expect(printpackView({ pack_id: "p", download: "/printpack/pack-1" }).url).toBe("/printpack/pack-1");
    // A file link must stay inside /data.
    const linked = printpackView({ files: [{ name: "a", path: "../../x" }, { name: "b", path: "https://x" }, { name: "c", path: "packs/p/c.stl" }] });
    expect(linked.files.map((f) => f.path)).toEqual([null, null, "packs/p/c.stl"]);
    const p = proposeView({ proposals: [{ design: { name: "d1", rationale: "why", finger_length_mm: 60 } }], rejected: 2 });
    expect(p.designs[0]).toMatchObject({ label: "d1", hypothesis: "why", params: { finger_length_mm: 60 } });
    expect(p.rejected).toBe(2);
  });
});

describe("session files", () => {
  it("validate the wrapper", () => {
    expect(() => parseRecordedSession([], "x.json")).toThrow(SessionError);
    expect(() => parseRecordedSession({ format: "other", events: [] }, "x.json")).toThrow(/unknown format/);
    expect(() => parseRecordedSession({ id: "Bad Id", events: [{ type: "summary", text: "a" }] }, "x.json")).toThrow(/URL-safe/);
    expect(() => parseRecordedSession({ id: "a", events: [{ type: "nope" }] }, "x.json")).toThrow(/no readable events/);
    expect(() => parseRecordedSession({ id: "a", part: { stl: "../secret.stl" }, events: [{ type: "summary", text: "a" }] }, "x.json")).toThrow(
      /relative data path/,
    );
    expect(parseRecordedSession({ events: [{ type: "summary", text: "hi" }] }, "sessions/run-7.json")).toMatchObject({ id: "run-7", kind: "recorded" });
  });

  it("the published sessions load, and the demo plays end to end within the caps", () => {
    expect(getSessions(DATA).length).toBeGreaterThan(0);
    const demo = parseRecordedSession(demoSession(), "demo.json");
    expect(demo).toMatchObject({ id: "demo-dumbbell", kind: "illustrative", offline: false, dropped: 0 });
    expect(kindsOf(demo.events)).toEqual(["inspect", "propose", "job", "results", "propose", "job", "results", "stress", "printpack"]);
    const tl = buildTimeline(demo.events);
    expect(tl.status).toBe("completed");
    expect(tl.tool_calls).toBeLessThanOrEqual(tl.caps?.max_tool_calls ?? 0);
    expect(Object.values(tl.jobs).every((j) => j.status === "completed" && j.cost_estimated)).toBe(true);
    // the finish event's spend is the sum of the session's own model calls and job estimates
    const s = spend(tl);
    expect(s.total_usd).toBeCloseTo(tl.llm.cost_usd + 0.04, 6);
    expect(summarizeSession(demo)).toMatchObject({ id: "demo-dumbbell", tool_calls: 12 });
    const committed = JSON.parse(readFileSync(path.join(DATA, "sessions", "demo-dumbbell.json"), "utf8"));
    if (committed.kind === "illustrative") expect(committed).toEqual(JSON.parse(JSON.stringify(demoSession())));
  });
});

describe("import-session", () => {
  it("adds the part file, drops the transcript and refuses local paths", () => {
    const doc = { ...fixture("agent-session-offline.json"), messages: [{ role: "system", content: "..." }] };
    const out = publishableSession(doc, { id: "sess-82d7e038aa5b", stlPath: "sessions/parts/sess-82d7e038aa5b.stl" });
    expect(out).toMatchObject({ id: "sess-82d7e038aa5b", kind: "recorded", part: { name: "dumbbell-handle.stl", stl: "sessions/parts/sess-82d7e038aa5b.stl" } });
    expect("messages" in out).toBe(false);
    expect(parseRecordedSession(out, "sessions/sess-82d7e038aa5b.json").part.stl).toBe("sessions/parts/sess-82d7e038aa5b.stl");
    const leaky = { ...doc, goal: "use /home/someone/part.stl" };
    expect(() => publishableSession(leaky, { id: "a", stlPath: "sessions/parts/a.stl" })).toThrow(/local paths/);
    expect(findLocalPaths({ a: ["C:\\Users\\x"], b: "fine/relative/path", c: "~/x", d: "file:///home/x/a.stl" })).toHaveLength(3);
    // a failed Nebius job's error can name the project or the image's registry path
    const failedJob = { ...doc, events: [...doc.events, { type: "job.update", job_id: "aijob-e00abcdefghijklmn", error: "image cr.eu-north1.nebius.cloud/abc/rtt not found" }] };
    expect(() => publishableSession(failedJob, { id: "a", stlPath: "sessions/parts/a.stl" })).toThrow(/registry paths at:\n {2}\$\.events\[\d+\]\.error/);
    const project = { ...doc, goal: "run it in project-e00abcdefghijklmnopq" };
    expect(() => publishableSession(project, { id: "a", stlPath: "sessions/parts/a.stl" })).toThrow(/Nebius resource IDs/);
    expect(() => publishableSession({ format: "rtt-agent-session", events: [] }, { id: "a", stlPath: "x" })).toThrow(/no events/);
  });
});

describe("import-session --pack", () => {
  it("links only the listed files of the exported pack, inside /data", () => {
    const events = [
      { type: "tool.call", name: "export_print_pack", args: {} },
      {
        type: "tool.result",
        name: "export_print_pack",
        ok: true,
        result: { pack_id: "pack-0123456789ab", files: [{ name: "finger_left.stl", bytes: 10 }, { name: "README.txt", bytes: 5 }] },
      },
      { type: "tool.result", name: "stress_test", ok: true, result: { files: [{ name: "finger_left.stl" }] } },
    ];
    const out = withPackPaths(events, { "pack-0123456789ab": { "finger_left.stl": "printpacks/pack-0123456789ab/finger_left.stl" } });
    const files = (out[1] as { result: { files: { name: string; path?: string }[] } }).result.files;
    expect(files[0]?.path).toBe("printpacks/pack-0123456789ab/finger_left.stl");
    expect(files[1]?.path).toBeUndefined();
    expect(out[2]).toBe(events[2]); // other tools untouched
    expect(printpackView((out[1] as { result: unknown }).result).files[0]?.path).toBe("printpacks/pack-0123456789ab/finger_left.stl");
    // another pack's map does nothing
    expect(withPackPaths(events, { "pack-ffffffffffff": { "finger_left.stl": "x" } })[1]).toBe(events[1]);
  });
});

describe("playback", () => {
  it("cuts long waits and keeps simultaneous events together", () => {
    const evs = [0, 0.1, 0.1, 100, 101].map((t) => normalizeEvent({ type: "summary", text: "x" }, t)!);
    expect(playbackTimes(evs, 2, 0.25)).toEqual([0, 0.25, 0.25, 2.25, 3.25]);
    const times = [0, 0.25, 0.25, 2.25];
    expect(playedCount(times, -1)).toBe(0);
    expect(playedCount(times, 0.25)).toBe(3);
    expect(playedCount(times, 99)).toBe(4);
  });
});
