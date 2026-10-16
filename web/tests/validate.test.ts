import { copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, truncateSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { checkDesignFiles, checkReplayFiles } from "@/lib/data";
import { parseAnyReplayHeader } from "@/lib/replay";
import { DataError, parseBouts, parseDesigns, parseJobs, parseResults } from "@/lib/validate";

const DATA = path.join(__dirname, "..", "public", "data");
/** Served GLB assets of version 2 replays, next to the data directory. */
const ASSETS = path.join(__dirname, "..", "public", "assets");
const read = (name: string) => JSON.parse(readFileSync(path.join(DATA, name), "utf8"));

describe("published data", () => {
  it("passes validation", () => {
    const bouts = parseBouts(read("bouts.json"));
    const designs = parseDesigns(read("designs.json"));
    parseJobs(read("jobs.json"));
    parseResults(read("results.json"));
    expect(bouts.bouts.length).toBeGreaterThan(0);
    // every lane's design id resolves to a design page
    const ids = new Set(designs.designs.map((d) => d.id));
    for (const lane of bouts.bouts.flatMap((b) => b.lanes)) expect(ids.has(lane.design_id)).toBe(true);
  });

  it("only links files that exist", () => {
    for (const d of parseDesigns(read("designs.json")).designs) {
      for (const file of Object.values(d.files)) {
        expect(() => readFileSync(path.join(DATA, file!))).not.toThrow();
      }
    }
    for (const lane of parseBouts(read("bouts.json")).bouts.flatMap((b) => b.lanes)) {
      // Published lanes may be replay version 1 or 2 (the viewer reads both).
      const header = parseAnyReplayHeader(read(`${lane.replay}.json`));
      expect(header.n_frames).toBe(lane.n_frames);
    }
  });
});

describe("validation errors", () => {
  it("name the file and JSON path", () => {
    const doc = read("bouts.json");
    doc.bouts[0].lanes[0].outcome.hold_s = "2";
    expect(() => parseBouts(doc)).toThrow(/bouts\.json: \$\.bouts\[0\]\.lanes\[0\]\.outcome\.hold_s: expected a finite number/);
  });

  it("reject a success that carries a failure mode", () => {
    const doc = read("bouts.json");
    doc.bouts[0].lanes[0].outcome.success = true;
    doc.bouts[0].lanes[0].outcome.failure_mode = "missed_grasp";
    expect(() => parseBouts(doc)).toThrow(DataError);
  });

  it("reject duplicate ids, unsafe paths and unknown schemas", () => {
    const doc = read("bouts.json");
    doc.bouts.push(doc.bouts[0]);
    expect(() => parseBouts(doc)).toThrow(/duplicate bout id/);
    const designs = read("designs.json");
    designs.designs[0].files.stl_left = "../../etc/passwd";
    expect(() => parseDesigns(designs)).toThrow(/relative data path/);
    expect(() => parseJobs({ schema: "nope", version: 1, jobs: [] })).toThrow(/schema/);
  });

  it("reject a job that ends before it starts", () => {
    const jobs = {
      schema: "rtt-web-jobs",
      version: 1,
      jobs: [
        {
          id: "j",
          source: "t",
          name: "j",
          purpose: "",
          kind: "nebius",
          executor: "cpu",
          platform: "p",
          preset: "q",
          region: "r",
          status: "completed",
          started_at: "2026-10-15T10:00:00Z",
          ended_at: "2026-10-15T09:00:00Z",
          wall_s: 1,
          cost_usd: 0.1,
          episodes: 1,
          sim_steps: 1,
          bouts: [],
        },
      ],
    };
    expect(() => parseJobs(jobs)).toThrow(/ended_at is before started_at/);
  });

  it("check mesh geoms' face indices", () => {
    const h = read("replays/m1/straight.json");
    h.bodies[3].geoms[0] = { ...h.bodies[3].geoms[0], type: "mesh", vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0], faces: [0, 1, 3] };
    expect(() => parseAnyReplayHeader(h)).toThrow(/face index outside/);
  });
});

describe("build-time file checks", () => {
  it("pass on the published data", () => {
    expect(() => checkReplayFiles(parseBouts(read("bouts.json")), DATA)).not.toThrow();
    expect(() => checkDesignFiles(parseDesigns(read("designs.json")), DATA)).not.toThrow();
  });

  it("fail on a missing, truncated or miscounted replay and a missing print file", () => {
    // A site tree like public/: data/ plus the assets/ that version 2 replays draw.
    const site = mkdtempSync(path.join(os.tmpdir(), "rtt-data-"));
    const dir = path.join(site, "data");
    mkdirSync(dir);
    try {
      cpSync(path.join(DATA, "replays"), path.join(dir, "replays"), { recursive: true });
      if (existsSync(ASSETS)) cpSync(ASSETS, path.join(site, "assets"), { recursive: true });
      const bouts = parseBouts(read("bouts.json"));
      const lane = bouts.bouts[0]!.lanes[0]!;
      const bin = path.join(dir, `${lane.replay}.bin`);

      truncateSync(bin, 100);
      expect(() => checkReplayFiles(bouts, dir)).toThrow(/bouts\.json: \$\.bouts\[0\]\.lanes\[0\]\.replay: .* is 100 bytes/);
      copyFileSync(path.join(DATA, `${lane.replay}.bin`), bin);
      expect(() => checkReplayFiles(bouts, dir)).not.toThrow();

      const miscounted = structuredClone(bouts);
      miscounted.bouts[0]!.lanes[0]!.n_frames += 1;
      expect(() => checkReplayFiles(miscounted, dir)).toThrow(/frames; the lane says/);

      rmSync(path.join(dir, `${lane.replay}.json`));
      expect(() => checkReplayFiles(bouts, dir)).toThrow(/missing replay header/);

      expect(() => checkDesignFiles(parseDesigns(read("designs.json")), dir)).toThrow(/designs\.json: .*files\.stl_left: missing file/);
    } finally {
      rmSync(site, { recursive: true, force: true });
    }
  });
});
