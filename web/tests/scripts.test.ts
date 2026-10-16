import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import {
  buildBouts,
  buildLane,
  certificateChecks,
  checkReplaysCurrent,
  frameOutcome,
  mergeBySource,
  readReplays,
  replayConfigMismatches,
  SOURCE,
  taskSummary,
} from "../scripts/import-demo.mjs";
import { withProvenance } from "../scripts/import-certificate.mjs";
import { findPrivateIds } from "../scripts/import-session.mjs";
import { bestSoFar, buildResults, readHistory, scanRunDir, sharedPrefix } from "../scripts/import-results.mjs";
import { syntheticResults } from "../scripts/make-synthetic-results.mjs";
import { resolveFile } from "../scripts/serve.mjs";
import { parseResults } from "@/lib/validate";

const WEB = path.join(__dirname, "..");
const REPLAYS = path.join(WEB, "public", "data", "replays", "m1");

const TASK_YAML = `
name: task_test
controller: { settle_time: 0.3, descend_time: 1.0, close_time: 1.0, squeeze_time: 0.3, lift_time: 1.5, grasp_height_fracs: [0.3, 0.5, 0.7] }
score: { success_lift: 0.1, success_hold: 2.0, record_tail: 0.5, max_time: 12.0 }
gripper: { grip_force: 50.0, open_half_gap: 0.042, palm_half_size: [0.024, 0.064, 0.012] }
replay: { fps: 30 }
robustness: { placement_errors_mm: [1.0, 2.0], shake_amplitude: 0.01, shake_frequency: 3.0, shake_cycles: 4, shake_axes: [x, z], push_impulse: 0.1, push_duration: 0.01, push_axes: [x, y] }
`;

describe("import-demo", () => {
  const { summary, graspHeightFracs, certificateChecks: checks } = taskSummary(parse(TASK_YAML));

  it("derives phase spans and constants from the task YAML", () => {
    expect(summary.phases.map((p) => [p.name, p.start_s, p.end_s])).toEqual([
      ["settle", 0, 0.3],
      ["descend", 0.3, 1.3],
      ["close", 1.3, 2.6],
      ["lift", 2.6, 4.1],
      ["hold", 4.1, null],
    ]);
    expect(summary.max_opening_mm).toBe(84);
    expect(graspHeightFracs).toEqual([0.3, 0.5, 0.7]);
  });

  it("lists the configured robustness checks", () => {
    expect(checks.map((c) => c.name)).toEqual([
      "Placement error 0 mm",
      "Placement error 1 mm",
      "Placement error 2 mm",
      "Shake during hold",
      "Lateral push",
    ]);
    expect(certificateChecks(undefined).length).toBeGreaterThan(0);
  });

  it("reads replays and derives outcomes from frames", () => {
    const replays = readReplays(REPLAYS);
    expect([...replays.keys()]).toEqual(["concave", "hooked", "low_friction", "stock", "straight", "too_high", "too_wide"]);
    expect(frameOutcome(replays.get("straight"), summary).maxLift * 1000).toBeCloseTo(149.1, 0);
    const lane = buildLane(replays.get("low_friction"), "Slippery", summary, graspHeightFracs, undefined);
    expect(lane.outcome).toMatchObject({ success: false, failure_mode: "slip_during_lift", source: "replay-frames" });
    expect(lane.grasp_height_frac).toBe(0.5);
  });

  it("refuses replays that disagree with the re-run scorer", () => {
    const replays = readReplays(REPLAYS);
    const gateCase = { success: true, failure_mode: null, max_lift_m: 0.149, hold_time_s: 2, flags: [], wall_time_s: 0.4, sim_steps: 1 };
    expect(() => buildLane(replays.get("too_high"), "x", summary, graspHeightFracs, gateCase)).toThrow(/stale/);
    const liftGap = { ...gateCase, max_lift_m: 0.12 };
    expect(() => buildLane(replays.get("straight"), "x", summary, graspHeightFracs, liftGap)).toThrow(/peak lift/);
  });

  it("accepts replays recorded with the task's gripper and refuses others", () => {
    const replays = readReplays(REPLAYS);
    const doc = parse(TASK_YAML);
    for (const replay of replays.values()) expect(replayConfigMismatches(replay, doc)).toEqual([]);
    expect(() => checkReplaysCurrent(replays, doc)).not.toThrow();
    // the pre-retarget gripper: 80 mm opening and a larger palm
    const old = { ...doc, gripper: { ...doc.gripper, open_half_gap: 0.04, palm_half_size: [0.03, 0.065, 0.015] } };
    const problems = replayConfigMismatches(replays.get("straight"), old);
    expect(problems.join("; ")).toMatch(/palm half size .*finger_left opens 42 mm from the palm, task YAML 40 mm/);
    expect(() => checkReplaysCurrent(replays, old)).toThrow(/recorded under a different config/);
    expect(replayConfigMismatches(replays.get("straight"), { ...doc, replay: { fps: 60 } })).toHaveLength(1);
    // a replay of an entrant whose design has since changed
    expect(() => checkReplaysCurrent(replays, doc, new Map([["straight", "000000000000"]]))).toThrow(/entrants YAML gives/);
  });

  it("groups the recorded cases into an entrants bout and a featured failures bout", () => {
    const replays = readReplays(REPLAYS);
    const order = ["straight", "concave", "hooked", "stock"];
    const { bouts, unused } = buildBouts(replays, summary, graspHeightFracs, null, undefined, order, "2026-10-15T10:00:00.000Z");
    expect(unused).toEqual([]);
    expect(bouts.map((b: { id: string }) => b.id)).toEqual(["m1-entrants", "m1-failures"]);
    const entrants = bouts[0]!;
    const failures = bouts[1]!;
    expect(entrants.lanes.map((l: { entrant: string }) => l.entrant)).toEqual(order);
    expect(entrants.title).toBe("Four baseline fingers, one box");
    expect(entrants.recorded_at).toBe("2026-10-15T10:00:00.000Z");
    expect(entrants.compute.note).toMatch(/Recorded at import/);
    const copied = buildBouts(replays, summary, graspHeightFracs, null, undefined, order).bouts[0]!;
    expect(copied.recorded_at).toBeNull();
    expect(copied.compute.note).toMatch(/Recorded during development/);
    expect(failures.featured).toBe(true);
    expect(failures.lanes[0]?.label).toBe("Baseline grasp");
    expect(failures.lanes.slice(1).every((l: { outcome: { success: boolean } }) => !l.outcome.success)).toBe(true);
    expect(failures.title).toBe("One good grasp and three planted failures");
  });

  it("replaces only its own entries when merging", () => {
    const merged = mergeBySource(
      [
        { id: "a", source: "nebius" },
        { id: "b", source: SOURCE },
      ],
      [{ id: "c", source: SOURCE }],
    );
    expect(merged.map((x: { id: string }) => x.id)).toEqual(["a", "c"]);
  });
});

describe("synthetic results", () => {
  it("validate and are marked synthetic", () => {
    const doc = syntheticResults();
    expect(parseResults(doc).status).toBe("synthetic");
    expect(doc.s1.arms[0]?.runs[0]?.best).toHaveLength(40);
  });

  it("match the committed file", () => {
    const committed = JSON.parse(readFileSync(path.join(WEB, "public", "data", "results.json"), "utf8"));
    if (committed.status === "synthetic") expect(committed).toEqual(JSON.parse(JSON.stringify(syntheticResults())));
  });
});

describe("import-results", () => {
  const history = (scores: number[], ids: string[]) =>
    scores.map((s, i) => JSON.stringify({ index: i + 1, proposal: { design_id: ids[i], origin: "x" }, score: { success: s } })).join("\n");

  function runDir(arm: string, scores: number[], ids: string[], job?: object) {
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-results-"));
    mkdirSync(path.join(dir, "search", "runs", `${arm}-r00`), { recursive: true });
    writeFileSync(path.join(dir, "search", "manifest.json"), JSON.stringify({ kind: "search", arm, budget: 4, executor: "cpu", job: { region: "uk-south2" } }));
    writeFileSync(path.join(dir, "search", "runs", `${arm}-r00`, "history.jsonl"), history(scores, ids));
    if (job) writeFileSync(path.join(dir, "job.json"), JSON.stringify(job));
    return dir;
  }

  it("reads histories in index order and finds the shared prefix", () => {
    const rows = readHistory('{"index":2,"proposal":{"design_id":"b"},"score":{"success":0.5}}\n{"index":1,"proposal":{"design_id":"a"},"score":{"success":0.25}}\n');
    expect(rows.map((r: { design_id: string }) => r.design_id)).toEqual(["a", "b"]);
    expect(bestSoFar([0.25, 0.5, 0.25])).toEqual([0.25, 0.5, 0.5]);
    const a = [{ design_id: "e1" }, { design_id: "e2" }, { design_id: "x" }];
    const b = [{ design_id: "e1" }, { design_id: "e2" }, { design_id: "y" }];
    expect(sharedPrefix([a, b])).toBe(2);
    expect(sharedPrefix([a])).toBe(0);
  });

  it("builds a valid pilot file with Nebius provenance and costs from jobs.json", () => {
    const dirs = [
      runDir("nemotron", [0.5, 0.6, 0.9, 0.7], ["e1", "e2", "n1", "n2"], { job_id: "aijob-n", platform: "cpu-d3", preset: "16vcpu-64gb", wall_s: 100.123456, secret: "x" }),
      runDir("random", [0.5, 0.6, 0.65, 0.85], ["e1", "e2", "r1", "r2"]),
    ];
    writeFileSync(
      path.join(dirs[0]!, "search", "runs", "nemotron-r00", "llm_calls.jsonl"),
      `${JSON.stringify({ cost_usd: 0.0031, prompt_tokens: 10 })}\n${JSON.stringify({ cost_usd: 0.0012 })}\n`,
    );
    writeFileSync(path.join(dirs[1]!, "search", "runs", "random-r00", "llm_calls.jsonl"), "");
    try {
      const doc = buildResults(dirs.map(scanRunDir), {
        status: "pilot",
        threshold: 0.8,
        thresholdNote: "provisional",
        note: "pilot",
        jobsIndex: { jobs: [{ id: "aijob-n", cost_usd: 0.05 }] },
      });
      const parsed = parseResults(doc);
      expect(parsed.s1.arms.map((a) => a.id)).toEqual(["a-random", "c-nemotron"]);
      expect(parsed.s1.shared_prefix).toBe(2);
      expect(parsed.s1.arms[1]!.runs[0]).toEqual({ run: "nemotron-r00", best: [0.5, 0.6, 0.9, 0.9], job_id: "aijob-n" });
      expect(parsed.h1.status).toBe("not_run");
      const nebius = parsed.sources.find((s) => s.kind === "nebius")!;
      expect(nebius).toMatchObject({ job_id: "aijob-n", platform: "cpu-d3", region: "uk-south2", wall_s: 100.1235, cost_usd: 0.05 });
      // model calls are billed separately from the job and reported next to it
      expect(nebius).toMatchObject({ llm_calls: 2, llm_usd: 0.0043 });
      expect(parsed.sources.find((x) => x.arm === "a-random")).toMatchObject({ llm_calls: 0, llm_usd: 0 });
      expect(JSON.stringify(doc)).not.toContain("secret");
    } finally {
      for (const d of dirs) rmSync(d, { recursive: true, force: true });
    }
  });
});

describe("import-certificate", () => {
  const doc = { schema: "rtt.certificates/1", design_id: "abc", certificates: [{}] };

  it("labels a local run and requires full Nebius facts", () => {
    expect(withProvenance(doc, { engine: "MuJoCo 3.15.0", wallS: "12.5" }).provenance).toEqual({
      kind: "local",
      executor: "cpu",
      engine: "MuJoCo 3.15.0",
      host: "workstation",
      wall_s: 12.5,
      sim_steps: null,
      note: null,
    });
    expect(() => withProvenance(doc, { jobId: "aijob-e00abc" })).toThrow(/--platform/);
    expect(withProvenance(doc, { jobId: "aijob-e00abc", platform: "p", preset: "s", region: "r", costUsd: "0.01" }).provenance).toMatchObject({
      kind: "nebius",
      job_id: "aijob-e00abc",
      cost_usd: 0.01,
    });
    expect(() => withProvenance({ design_id: "x", certificates: [] }, {})).toThrow(/no certificates/);
  });

  it("refuses local paths, Nebius resource ids and registry paths, but not the AI job id", () => {
    const full = { platform: "p", preset: "s", region: "r" };
    expect(() => withProvenance(doc, { jobId: "project-e00abcdefghijklmnop", ...full })).toThrow(/AI job id/);
    expect(() => withProvenance({ ...doc, note: "ran in /home/someone/rtt" }, {})).toThrow(/local paths/);
    expect(() => withProvenance({ ...doc, note: "bucket storagebucket-e00abcdefghijklmnopq" }, {})).toThrow(/Nebius resource IDs/);
    expect(() => withProvenance({ ...doc, image: "cr.eu-north1.nebius.cloud/x/rtt:1" }, {})).toThrow(/registry/);
    expect(withProvenance(doc, { jobId: "aijob-e00xx86cr7b6ycwkwm", ...full }).provenance).toMatchObject({ job_id: "aijob-e00xx86cr7b6ycwkwm" });
    // agent ids (prefix + 12 hex) are not mistaken for Nebius ids
    expect(findPrivateIds({ a: "sess-e0a1b2c3d4e5", b: "pack-e00000000000", c: "aijob-e00p6shj7m77w4dfnd" })).toEqual([]);
  });

  it("the published certificates pass the importer's checks", () => {
    const dir = path.join(__dirname, "..", "public", "data", "certificates");
    for (const name of readdirSync(dir)) {
      const { provenance, ...cert } = JSON.parse(readFileSync(path.join(dir, name), "utf8"));
      expect(() => withProvenance(cert, { note: provenance?.note ?? undefined })).not.toThrow();
    }
  });
});

describe("serve", () => {
  it("never resolves outside the site root", () => {
    const root = path.join(WEB, "public");
    expect(resolveFile(root, "/../package.json")).toBeNull();
    expect(resolveFile(root, "/%E0%A4%A")).toBeNull(); // malformed escape: no file, no throw
    expect(resolveFile(root, "/data/bouts.json")).toBe(path.join(root, "data", "bouts.json"));
  });

  it("does not serve a sibling directory that shares the root's name as a prefix", () => {
    const base = mkdtempSync(path.join(os.tmpdir(), "rtt-serve-"));
    try {
      mkdirSync(path.join(base, "out"));
      mkdirSync(path.join(base, "out-private"));
      writeFileSync(path.join(base, "out-private", "secret.txt"), "x");
      writeFileSync(path.join(base, "out", "index.html"), "ok");
      const root = path.join(base, "out");
      expect(resolveFile(root, "/../out-private/secret.txt")).toBeNull();
      expect(resolveFile(root, "/")).toBe(path.join(root, "index.html"));
    } finally {
      rmSync(base, { recursive: true, force: true });
    }
  });
});
