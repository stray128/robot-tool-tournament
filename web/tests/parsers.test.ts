import { readFileSync } from "node:fs";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

import { apiBase, errorMessage, PASSCODE_HEADER, startSession, streamEvents, uploadPart } from "@/lib/api";
import { describeDisturbance, disturbanceRows, parseCertificateDoc, placementCurve, summarizeCertificate } from "@/lib/certificate";
import { getCertificates } from "@/lib/data";
import { formatP, formatPercent } from "@/lib/format";
import {
  bestSoFar,
  designsToThreshold,
  medianCurve,
  s1Outcome,
  secondaryDiff,
  secondaryValue,
  stepPoints,
  summarizeArm,
} from "@/lib/results";
import { createSseParser, type SseMessage } from "@/lib/sse";
import { binaryTriangleCount, checkStlBuffer, checkStlFile, encodeBinaryStl, fitAgainstOpening, guessUnit, measurePositions, toMillimetres } from "@/lib/stl";
import { parseResults } from "@/lib/validate";

const DATA = path.join(__dirname, "..", "public", "data");

describe("sse parser", () => {
  const collect = (chunks: string[], flush = false) => {
    const out: SseMessage[] = [];
    const p = createSseParser((m) => out.push(m));
    chunks.forEach((c) => p.push(c));
    if (flush) p.flush();
    return { out, p };
  };

  it("parses events split across chunks with CRLF endings", () => {
    const { out } = collect(['event: tool_call\r\ndata: {"a":', "1}\r", "\n\r\n", "data: x\n\n"]);
    expect(out).toEqual([
      { event: "tool_call", data: '{"a":1}', id: null },
      { event: "message", data: "x", id: null },
    ]);
  });

  it("joins multi-line data, ignores comments and remembers ids and retry", () => {
    const { out, p } = collect([": keep-alive\n", "id: 7\nretry: 1500\ndata: a\ndata: b\n\n", "data: c\n\n"]);
    expect(out).toEqual([
      { event: "message", data: "a\nb", id: "7" },
      { event: "message", data: "c", id: "7" },
    ]);
    expect(p.retryMs).toBe(1500);
  });

  it("dispatches a trailing message on flush and skips empty ones", () => {
    expect(collect(["data: last"], true).out).toEqual([{ event: "message", data: "last", id: null }]);
    expect(collect(["event: ping\n\n"]).out).toEqual([]);
  });
});

describe("stl helpers", () => {
  it("measures vertex triples and skips non-finite values", () => {
    const e = measurePositions(new Float32Array([0, 0, 0, 10, 20, 5, NaN, 0, 0, -2, 1, 1]));
    expect(e).toEqual({ min: [-2, 0, 0], max: [10, 20, 5], size: [12, 20, 5], vertices: 3 });
    expect(measurePositions([])).toBeNull();
  });

  it("guesses metres for sub-unit parts and converts", () => {
    expect(guessUnit([0.06, 0.03, 0.05])).toBe("m");
    expect(guessUnit([60, 30, 50])).toBe("mm");
    expect(toMillimetres([0.06, 0.03, 0.05], "m").map((v) => Number(v.toFixed(6)))).toEqual([60, 30, 50]);
    expect(toMillimetres([2, 1, 1], "in")).toEqual([50.8, 25.4, 25.4]);
  });

  it("judges fit against the opening on the narrowest side", () => {
    expect(fitAgainstOpening([84, 40, 40], 84)).toMatchObject({ verdict: "fits", narrowest_mm: 40, margin_mm: 44 });
    expect(fitAgainstOpening([120, 82, 90], 84).verdict).toBe("tight");
    expect(fitAgainstOpening([120, 90, 100], 84)).toMatchObject({ verdict: "too_wide", margin_mm: -6 });
    expect(fitAgainstOpening([0.06, 0.03, 0.05], 84).verdict).toBe("too_small");
    expect(fitAgainstOpening([0, 0, 0], 84).verdict).toBe("empty");
  });

  it("checks file names and sizes", () => {
    expect(checkStlFile("part.STL", 1000).ok).toBe(true);
    expect(checkStlFile("part.obj", 1000).problem).toMatch(/not an .stl/);
    expect(checkStlFile("part.stl", 10).problem).toMatch(/too small/);
    expect(checkStlFile("part.stl", 60 * 1048576).problem).toMatch(/limit/);
  });

  it("reads the demo part with three's STLLoader, in Node", () => {
    const buf = readFileSync(path.join(DATA, "sessions", "parts", "dumbbell-handle.stl"));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    expect(binaryTriangleCount(ab)).toBe(576);
    const geom = new STLLoader().parse(ab);
    const e = measurePositions(geom.getAttribute("position").array);
    expect(e?.size.map((v) => Math.round(v))).toEqual([84, 40, 40]);
    expect(binaryTriangleCount(new TextEncoder().encode("solid x\nendsolid x\n").buffer)).toBeNull();
    expect(checkStlBuffer(ab)).toBeNull();
  });

  it("refuses a binary STL whose header claims more triangles than the file holds", () => {
    const forged = new ArrayBuffer(84 + 50);
    new DataView(forged).setUint32(80, 0xffffffff, true);
    expect(checkStlBuffer(forged)).toMatch(/declares 4,294,967,295 triangles.*at most 1/);
    const empty = new ArrayBuffer(84);
    expect(checkStlBuffer(empty)).toMatch(/no triangles/);
    // ASCII files and binary files with trailing bytes still go to the loader.
    const ascii = new TextEncoder().encode(`solid x\n${" ".repeat(90)}\nendsolid x\n`);
    expect(checkStlBuffer(ascii.buffer.slice(0) as ArrayBuffer)).toBeNull();
    const padded = new ArrayBuffer(84 + 50 + 7);
    new DataView(padded).setUint32(80, 1, true);
    expect(checkStlBuffer(padded)).toBeNull();
    expect(checkStlBuffer(encodeBinaryStl([0, 0, 0, 1, 0, 0, 0, 1, 0]))).toBeNull();
  });

  it("re-encodes a part in millimetres", () => {
    const tri = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1];
    const buf = encodeBinaryStl(tri, 25.4);
    expect(binaryTriangleCount(buf)).toBe(2);
    const e = measurePositions(new STLLoader().parse(buf).getAttribute("position").array);
    expect(e?.size.map((v) => Number(v.toFixed(3)))).toEqual([25.4, 25.4, 25.4]);
  });
});

describe("s1", () => {
  it("computes best-so-far and designs to threshold", () => {
    expect(bestSoFar([0.2, 0.1, 0.5, 0.4])).toEqual([0.2, 0.2, 0.5, 0.5]);
    expect(designsToThreshold([0.2, 0.2, 0.5, 0.5], 0.5)).toBe(3);
    expect(designsToThreshold([0.2, 0.2], 0.5)).toBeNull();
  });

  it("censors at the budget, or at a short run's length", () => {
    expect(s1Outcome({ run: "r", best: [0.1, 0.9], job_id: null }, 0.8, 40)).toMatchObject({ designs: 2, reached: true, final_best: 0.9 });
    expect(s1Outcome({ run: "r", best: Array(40).fill(0.5), job_id: null }, 0.8, 40)).toMatchObject({ designs: 40, reached: false });
    expect(s1Outcome({ run: "r", best: [0.5, 0.5], job_id: null }, 0.8, 40)).toMatchObject({ designs: 2, reached: false });
  });

  it("uses a recorded T, so round 0 does not count", () => {
    // round 0 (designs 1-2) already reached 0.8, but T counts from design 3
    const run = { run: "r", best: [0.9, 0.9, 0.9, 0.9], job_id: null, t: 5 };
    expect(s1Outcome(run, 0.8, 4)).toMatchObject({ designs: 4, reached: false, t: 5 });
    expect(s1Outcome({ ...run, t: 3 }, 0.8, 4)).toMatchObject({ designs: 3, reached: true, t: 3 });
    const arm = summarizeArm({ id: "a", label: "A", runs: [{ ...run, t: 3 }, run, { ...run, t: 4 }] }, 0.8, 4);
    expect(arm).toMatchObject({ reached: 2, median_designs: 4 });
    expect(arm.mean_designs).toBe(4);
  });

  it("reports a censored median as not reached instead of imputing it", () => {
    const run = (best: number[]) => ({ run: "r", best, job_id: null });
    const reached = summarizeArm({ id: "a", label: "A", runs: [run([0.9]), run([0.1, 0.9]), run([0.1, 0.1])] }, 0.8, 2);
    expect(reached).toMatchObject({ reached: 2, runs: 3, median_designs: 2 });
    const censored = summarizeArm({ id: "a", label: "A", runs: [run([0.9]), run([0.1, 0.1]), run([0.1, 0.1])] }, 0.8, 2);
    expect(censored.median_designs).toBeNull();
    expect(summarizeArm({ id: "a", label: "A", runs: [] }, 0.8, 2).median_designs).toBeNull();
  });

  it("draws step curves and medians", () => {
    expect(stepPoints([0.5, 1], 2)).toBe("0,50 100,50 100,0");
    expect(medianCurve([{ best: [0.1, 0.2, 0.3] }, { best: [0.3, 0.4] }, { best: [0.2, 0.6, 0.9] }])).toEqual([0.2, 0.4, 0.6]);
  });
});

describe("results.json", () => {
  const v2 = () => JSON.parse(readFileSync(path.join(DATA, "results.json"), "utf8"));

  it("the published file is version 2 and labelled", () => {
    const r = parseResults(v2());
    expect(r.version).toBe(2);
    expect(["synthetic", "pilot", "final"]).toContain(r.status);
    expect(r.s1.arms.length).toBeGreaterThan(0);
  });

  it("a final file shows exactly the numbers of docs/RESULTS.md", () => {
    const r = parseResults(v2());
    if (r.status !== "final") return;
    const doc = readFileSync(path.join(__dirname, "..", "..", "docs", "RESULTS.md"), "utf8");
    const shown: string[] = [];
    const { s1 } = r;
    const [a, c] = s1.arms.map((arm) => summarizeArm(arm, s1.threshold!, s1.budget));
    expect(s1.count_from).toBe(9);
    expect(`${Math.round(s1.threshold! * s1.trials!)}/${s1.trials}`).toBe("37/48");
    shown.push(`${a!.reached} of ${a!.runs}`, `${c!.reached} of ${c!.runs}`);
    expect([a!.reached, c!.reached]).toEqual([11, 23]);
    expect(a!.median_designs).toBeNull();
    shown.push(`not reached (above ${s1.budget})`, String(c!.median_designs));
    shown.push(`counted as ${s1.budget + 1}) | ${a!.mean_designs!.toFixed(1)} | ${c!.mean_designs!.toFixed(1)} |`);
    const an = s1.analysis!;
    shown.push(`${an.effect.diff.toFixed(2)} designs`, `interval ${an.effect.ci_low.toFixed(2)} to ${an.effect.ci_high.toFixed(2)}`);
    shown.push(`p = ${formatP(an.primary.p_value)}`, `log-rank: p = ${formatP(an.logrank.p_value)}`);
    shown.push(`(p = ${formatP(an.lift_only!.p_value)})`);
    for (const t of r.secondary!.tests) {
      shown.push(`| ${secondaryValue(t, t.mean_a)} | ${secondaryValue(t, t.mean_c)} | ${secondaryDiff(t)} | ${formatP(t.p_value)} | ${formatP(t.p_holm!)} |`);
    }
    for (const b of r.secondary!.width_bands!.bands) {
      shown.push(`(${b.objects} objects) ${formatPercent(b.mean_a, 1)} vs ${formatPercent(b.mean_c, 1)} (p = ${formatP(b.p_value)})`);
    }
    for (const s of shown) expect(doc, s).toContain(s);
  });

  it("rejects a decreasing best-so-far curve and an out-of-range shared prefix", () => {
    const doc = v2();
    doc.s1.arms[0].runs[0].best = [0.5, 0.4];
    expect(() => parseResults(doc)).toThrow(/decreases at design 2/);
    const doc2 = v2();
    doc2.s1.shared_prefix = doc2.s1.budget + 1;
    expect(() => parseResults(doc2)).toThrow(/shared_prefix/);
    const doc3 = v2();
    doc3.h1 = { endpoint: "H1", status: "not_run", arms: [{ id: "a", label: "A", runs: [0.5], mean: 0.5, ci95: [0.4, 0.6] }], references: [] };
    expect(() => parseResults(doc3)).toThrow(/not_run/);
  });

  it("upgrades a version 1 file", () => {
    const v1 = {
      schema: "rtt-web-results",
      version: 1,
      status: "synthetic",
      note: "",
      forest: { endpoint: "Held-out", arms: [{ id: "a", label: "A", runs: [0.5], mean: 0.5, ci95: [0.4, 0.6] }], references: [] },
      curves: { endpoint: "Best so far", budget: 3, arms: [{ id: "a", label: "A", runs: [{ run: "r1", best: [0.1, 0.2] }] }] },
    };
    const r = parseResults(v1);
    expect(r.s1).toMatchObject({ threshold: null, budget: 3, shared_prefix: 0 });
    expect(r.s1.arms[0]!.runs[0]).toEqual({ run: "r1", best: [0.1, 0.2], job_id: null });
    expect(r.h1?.status).toBe("available");
  });
});

describe("certificates", () => {
  const test = (name: string, category: string, params: object, success: boolean) => ({
    name,
    category,
    params,
    success,
    failure_mode: success ? null : "slip_during_lift",
    max_lift_m: 0.1,
    hold_time_s: success ? 2 : 0.3,
    flags: [],
  });
  const cert = (object: string, okAt3: boolean) => ({
    object,
    controller_setting: 0,
    task: "task_v1",
    task_digest: "abc",
    score: 0.9,
    category_scores: { placement: 1, shake: 1, push: 0.5 },
    tolerance_mm: { x: okAt3 ? 3 : 2, y: 3 },
    passed: "17/18",
    tests: [
      test("nominal", "placement", { axis: null, error_mm: 0 }, true),
      test("offset_x_+3mm", "placement", { axis: "x", error_mm: 3 }, okAt3),
      test("offset_x_-3mm", "placement", { axis: "x", error_mm: -3 }, true),
      test("offset_y_+3mm", "placement", { axis: "y", error_mm: 3 }, true),
      test("shake_z", "shake", { axis: "z", amplitude_m: 0.01, frequency_hz: 5, cycles: 5, peak_accel_m_s2: 9.87 }, true),
      test("push_x", "push", { axis: "x", impulse_ns: 0.5, duration_s: 0.02, force_n: 25, height_above_com_m: 0.055 }, false),
    ],
  });

  it("parse a single certificate or a list, and reject bad files", () => {
    const one = parseCertificateDoc({ design_id: "d", ...cert("box", true) }, "d");
    expect(one.certificates).toHaveLength(1);
    expect(one.provenance).toBeNull();
    expect(() => parseCertificateDoc({ schema: "nope", design_id: "d", certificates: [] }, "d")).toThrow(/schema/);
    const noNominal = cert("box", true);
    noNominal.tests = noNominal.tests.slice(1);
    expect(() => parseCertificateDoc({ design_id: "d", certificates: [noNominal] }, "d")).toThrow(/no nominal/);
  });

  it("build the placement curve, disturbance rows and summary", () => {
    const doc = parseCertificateDoc({ design_id: "d", certificates: [cert("box", true), cert("rod", false)] }, "d");
    expect(placementCurve(doc.certificates, "x")).toEqual([
      { error_mm: -3, held: 2, total: 2, rate: 1 },
      { error_mm: 0, held: 2, total: 2, rate: 1 },
      { error_mm: 3, held: 1, total: 2, rate: 0.5 },
    ]);
    expect(placementCurve(doc.certificates, "y").map((p) => p.error_mm)).toEqual([0, 3]);
    const shakes = disturbanceRows(doc.certificates, "shake");
    expect(shakes[0]).toMatchObject({ object: "box", axis: "z", success: true });
    expect(shakes[0]!.detail).toBe("10 mm at 5 Hz, 5 cycles, peak 9.9 m/s²");
    const pushes = disturbanceRows(doc.certificates, "push");
    expect(pushes[0]).toMatchObject({ detail: "0.50 N·s (25 N for 20 ms)", push_height_mm: 55 });
    expect(describeDisturbance(pushes)).toBe("0.50 N·s (25 N for 20 ms), 55 mm above the centre of mass");
    expect(describeDisturbance(shakes)).toBe("10 mm at 5 Hz, 5 cycles, peak 9.9 m/s²");
    // The push is at each object's top, so its height differs by object: a range, never one object's value.
    const varied = [pushes[0]!, { ...pushes[0]!, object: "rod", push_height_mm: 3 }];
    expect(describeDisturbance(varied)).toBe(
      "0.50 N·s (25 N for 20 ms), at the object's top: 3 to 55 mm above its centre of mass, depending on the object",
    );
    expect(describeDisturbance([shakes[0]!, { ...shakes[0]!, detail: "20 mm at 5 Hz" }])).toBeNull();
    expect(describeDisturbance([])).toBeNull();
    const s = summarizeCertificate(doc);
    expect(s).toMatchObject({ objects: 2, episodes: 12, passed: 9, tolerance_mm: { x: 2, y: 3 } });
    expect(s.category.push).toEqual({ passed: 0, total: 2 });
  });

  it("the published certificates load with their provenance", () => {
    const certs = getCertificates(DATA);
    expect(certs.length).toBeGreaterThan(0);
    for (const c of certs) {
      expect(c.certificates.length).toBeGreaterThan(0);
      expect(c.provenance).not.toBeNull();
    }
  });
});

describe("api client against the FastAPI contract", () => {
  afterEach(() => vi.unstubAllGlobals());

  const sse = (chunks: string[]) =>
    new Response(
      new ReadableStream({
        start(c) {
          for (const ch of chunks) c.enqueue(new TextEncoder().encode(ch));
          c.close();
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } },
    );
  const frame = (seq: number, type: string, extra = {}) =>
    `id: ${seq}\nevent: ${type}\ndata: ${JSON.stringify({ seq, type, t_s: seq, ...extra })}\n\n`;

  it("uploads the raw STL with name and units in the query, and the passcode only to /sessions", async () => {
    const calls: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      if (url.includes("/parts")) return new Response(JSON.stringify({ part: { part_id: "part-abc", name: "x" } }), { status: 201 });
      return new Response(JSON.stringify({ session_id: "sess-1" }), { status: 201 });
    });
    const part = await uploadPart("http://api", new Uint8Array([1, 2, 3]).buffer, "bracket.stl", "mm");
    expect(part.part_id).toBe("part-abc");
    expect(calls[0]!.url).toBe("http://api/parts?name=bracket&units=mm");
    expect((calls[0]!.init.headers as Record<string, string>)["Content-Type"]).toBe("application/octet-stream");
    expect(JSON.stringify(calls[0]!.init.headers)).not.toContain(PASSCODE_HEADER);
    expect(await startSession("http://api", "part-abc", "s3cret")).toBe("sess-1");
    expect((calls[1]!.init.headers as Record<string, string>)[PASSCODE_HEADER]).toBe("s3cret");
    expect(JSON.parse(String(calls[1]!.init.body))).toEqual({ part_id: "part-abc", passcode: "s3cret" });
    expect(calls.every((c) => !c.url.includes("s3cret"))).toBe(true);
  });

  it("explains a refused passcode", async () => {
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ detail: "wrong passcode" }), { status: 403 }));
    await expect(startSession("http://api", "part-abc", "nope")).rejects.toThrow(/wrong passcode\. Check the passcode/);
  });

  it("streams events until `end`, resuming a dropped stream from the last event id", async () => {
    const seen: string[] = [];
    const resumedFrom: (string | null)[] = [];
    let n = 0;
    const doc = { events: [1, 2, 3].map((seq) => ({ seq, type: seq === 3 ? "session.finished" : "x" })) };
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (!url.endsWith("/events")) return new Response(JSON.stringify(doc)); // the catch-up read
      resumedFrom.push((init.headers as Record<string, string>)["Last-Event-ID"] ?? null);
      n++;
      if (n === 1) return sse(["retry: 2000\n\n", frame(1, "session.started"), frame(2, "tool.call", { name: "inspect_part", call_id: "c1" })]);
      return sse([": keep-alive\n\n", frame(3, "session.finished", { status: "completed" }), 'event: end\ndata: {"last_seq":3}\n\n', frame(4, "never")]);
    });
    vi.useFakeTimers({ toFake: ["setTimeout"] });
    const done = streamEvents("http://api", "sess-1", (payload, name) => seen.push(`${name}:${(payload as { seq: number }).seq}`));
    await vi.runAllTimersAsync();
    await done;
    vi.useRealTimers();
    expect(seen).toEqual(["session.started:1", "tool.call:2", "session.finished:3"]);
    expect(resumedFrom).toEqual([null, "2"]);
  });

  it("counts only consecutive failed reconnects against the retry budget", async () => {
    // Five drops, each after one new event, then the end: more drops than `retries`,
    // but every connection made progress, so the stream must still finish.
    const seen: number[] = [];
    let n = 0;
    vi.stubGlobal("fetch", async (url: string) => {
      if (!url.endsWith("/events")) return new Response(JSON.stringify({ events: [] }));
      n++;
      if (n <= 5) return sse([frame(n, "x")]);
      return sse([frame(6, "session.finished", { status: "completed" }), 'event: end\ndata: {"last_seq":6}\n\n']);
    });
    vi.useFakeTimers({ toFake: ["setTimeout"] });
    const done = streamEvents("http://api", "sess-1", (payload) => seen.push((payload as { seq: number }).seq), undefined, 2);
    await vi.runAllTimersAsync();
    await done;
    expect(seen).toEqual([1, 2, 3, 4, 5, 6]);

    // A stream that keeps dropping with nothing new gives up after `retries` reconnects.
    let empty = 0;
    vi.stubGlobal("fetch", async () => {
      empty++;
      return sse([": keep-alive\n\n"]);
    });
    const failed = streamEvents("http://api", "sess-2", () => {}, undefined, 2).then(
      () => null,
      (err: unknown) => err,
    );
    await vi.runAllTimersAsync();
    vi.useRealTimers();
    expect(String(await failed)).toMatch(/closed before the session ended/);
    expect(empty).toBe(3);
  });

  it("recovers session.finished when the stream ends just before the server appends it", async () => {
    // The API marks a session done, then appends session.finished; a stream that checks
    // in between sends `end` without it. The session document has it.
    const seen: string[] = [];
    const urls: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      urls.push(url);
      if (url.endsWith("/events")) return sse([frame(1, "session.started"), frame(2, "tool.call"), 'event: end\ndata: {"last_seq":2}\n\n']);
      return new Response(JSON.stringify({ events: [1, 2, 3].map((seq) => ({ seq, type: seq === 3 ? "session.finished" : "x", status: "completed" })) }));
    });
    await streamEvents("http://api", "sess-1", (payload, name) => seen.push(`${name}:${(payload as { seq: number }).seq}`));
    expect(seen).toEqual(["session.started:1", "tool.call:2", "session.finished:3"]);
    expect(urls).toEqual(["http://api/sessions/sess-1/events", "http://api/sessions/sess-1"]);
  });
});

describe("api helpers", () => {
  it("reads the API base from the env value", () => {
    expect(apiBase(undefined)).toBeNull();
    expect(apiBase("  ")).toBeNull();
    expect(apiBase("http://127.0.0.1:8000/")).toBe("http://127.0.0.1:8000");
    expect(apiBase("javascript:alert(1)")).toBeNull();
    expect(apiBase("not a url")).toBeNull();
  });

  it("extracts FastAPI error details", async () => {
    expect(await errorMessage(new Response(JSON.stringify({ detail: "bad passcode" }), { status: 401 }))).toBe("bad passcode");
    expect(await errorMessage(new Response(JSON.stringify({ detail: [{ msg: "field required" }] }), { status: 422 }))).toBe("field required");
    expect(await errorMessage(new Response("", { status: 500, statusText: "Server Error" }))).toBe("500 Server Error");
  });
});

describe("site stats", () => {
  it("sum the published data", async () => {
    const { siteStats } = await import("@/lib/stats");
    const { parseDesigns, parseJobs } = await import("@/lib/validate");
    const read = (name: string) => JSON.parse(readFileSync(path.join(DATA, name), "utf8"));
    const jobs = parseJobs(read("jobs.json"));
    const stats = siteStats({ jobs, designs: parseDesigns(read("designs.json")), results: parseResults(read("results.json")), certificates: getCertificates(DATA) });
    expect(stats.map((s) => s.label)[0]).toBe("Grasp trials simulated");
    const nebius = jobs.jobs.filter((j) => j.kind === "nebius").length;
    expect(stats.find((s) => s.label === "Nebius jobs")?.value).toBe(String(nebius));
    for (const s of stats) expect(s.value).not.toMatch(/NaN|undefined/);
  });
});
