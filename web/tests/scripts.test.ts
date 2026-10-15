import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
    expect(doc.curves.arms[0]?.runs[0]?.best).toHaveLength(40);
  });

  it("match the committed file", () => {
    const committed = JSON.parse(readFileSync(path.join(WEB, "public", "data", "results.json"), "utf8"));
    if (committed.status === "synthetic") expect(committed).toEqual(JSON.parse(JSON.stringify(syntheticResults())));
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
