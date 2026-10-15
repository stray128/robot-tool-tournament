import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { decodeFrames, geomPoints, isParkedGeom, phaseAt, Replay, ReplayFormatError, rotate, slerp } from "@/lib/replay";
import type { ReplayHeader, TaskSummary } from "@/lib/types";
import { parseReplayHeader } from "@/lib/validate";

const REPLAYS = path.join(__dirname, "..", "public", "data", "replays", "m1");

function loadDemo(stem: string): Replay {
  const header = parseReplayHeader(JSON.parse(readFileSync(path.join(REPLAYS, `${stem}.json`), "utf8")));
  const buf = readFileSync(path.join(REPLAYS, header.bin));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return new Replay(header, decodeFrames(ab, header));
}

const TASK: Pick<TaskSummary, "phases" | "success_lift_m" | "record_tail_s"> = {
  phases: [
    { name: "settle", start_s: 0, end_s: 0.3 },
    { name: "descend", start_s: 0.3, end_s: 1.3 },
    { name: "close", start_s: 1.3, end_s: 2.6 },
    { name: "lift", start_s: 2.6, end_s: 4.1 },
    { name: "hold", start_s: 4.1, end_s: null },
  ],
  success_lift_m: 0.1,
  record_tail_s: 0.5,
};

function header(nFrames: number, bodies: string[]): ReplayHeader {
  return {
    format: "rtt-replay",
    version: 1,
    fps: 10,
    dtype: "float32",
    byte_order: "little",
    up_axis: "z",
    units: "m",
    bodies: bodies.map((name) => ({ name, geoms: [] })),
    static: [],
    n_frames: nFrames,
    n_bodies: bodies.length,
    bin: "x.bin",
    meta: {},
  };
}

describe("decodeFrames", () => {
  it("rejects a payload whose size does not match the header", () => {
    expect(() => decodeFrames(new ArrayBuffer(28 * 3), { n_frames: 2, n_bodies: 2 })).toThrow(ReplayFormatError);
  });

  it("decodes little-endian float32 poses", () => {
    const data = new Float32Array([1, 2, 3, 0, 0, 0, 1]);
    expect(Array.from(decodeFrames(data.buffer, { n_frames: 1, n_bodies: 1 }))).toEqual([1, 2, 3, 0, 0, 0, 1]);
  });
});

describe("Replay sampling", () => {
  it("interpolates position linearly and rotation spherically between frames", () => {
    const s = Math.SQRT1_2;
    // frame 0: origin, identity; frame 1: (1, 0, 0), 90 degrees about z
    const frames = new Float32Array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, s, s]);
    const replay = new Replay(header(2, ["object"]), frames);
    const pos = [0, 0, 0];
    const quat = [0, 0, 0, 0];
    replay.sample(0, 0.5, pos, quat);
    expect(pos[0]).toBeCloseTo(0.5, 6);
    const half = Math.sin(Math.PI / 8);
    expect(quat[2]).toBeCloseTo(half, 5); // 45 degrees about z
    expect(Math.hypot(...quat)).toBeCloseTo(1, 6);
    expect(replay.duration).toBeCloseTo(0.1, 6);
    expect(replay.frameAt(99)).toBe(1);
  });

  it("slerp takes the short way round for opposite-sign quaternions", () => {
    const out = [0, 0, 0, 0];
    slerp([0, 0, 0, 1, 0, 0, 0, -1], 0, 4, 0.5, out);
    expect(out.map((v) => v + 0)).toEqual([0, 0, 0, 1]);
  });

  it("rotates vectors by a quaternion", () => {
    const out = [0, 0, 0];
    rotate([0, 0, Math.SQRT1_2, Math.SQRT1_2], [1, 0, 0], out);
    expect(out[0]).toBeCloseTo(0, 6);
    expect(out[1]).toBeCloseTo(1, 6);
  });
});

describe("demo replays", () => {
  it("parse, decode and reproduce the scorer's peak lift and hold for the straight entrant", () => {
    const replay = loadDemo("straight");
    expect(replay.header.bodies.map((b) => b.name)).toEqual(["palm", "finger_left", "finger_right", "object"]);
    const tracks = replay.tracks(TASK);
    expect(Math.max(...tracks.lift) * 1000).toBeCloseTo(149.1, 0);
    expect(Math.max(...tracks.hold)).toBeGreaterThanOrEqual(2.0);
    expect(tracks.decidedAt).toBeCloseTo(replay.duration - 0.5, 6);
    const ref = replay.objectReference(TASK);
    expect(ref?.[2]).toBeCloseTo(0.06, 2); // a 120 mm box rests with its centre 60 mm up
  });

  it("keeps a missed grasp on the table", () => {
    const tracks = loadDemo("too_high").tracks(TASK);
    expect(Math.max(...tracks.lift)).toBeLessThan(0.002);
  });

  it("bounds every drawn body over the recording, skipping parked generator boxes", () => {
    const replay = loadDemo("straight");
    const parked = replay.header.bodies[1]!.geoms.filter(isParkedGeom).length;
    expect(parked).toBeGreaterThan(0);
    const b = replay.bounds();
    expect(b.min[2]).toBeGreaterThanOrEqual(-0.001); // nothing below the table top
    expect(b.max[2]).toBeGreaterThan(0.3); // the palm lifts well above the box
    expect(b.max[1] - b.min[1]).toBeLessThan(0.3);
  });
});

describe("geomPoints", () => {
  it("places box corners with the geom's offset and rotation", () => {
    const pts = geomPoints({
      name: "b",
      type: "box",
      half_size: [0.1, 0.2, 0.3],
      pos: [1, 0, 0],
      quat_xyzw: [0, 0, 0, 1],
      rgba: [1, 1, 1, 1],
    });
    expect(pts).toHaveLength(8);
    expect(Math.max(...pts.map((p) => p[0]!))).toBeCloseTo(1.1, 6);
  });

  it("uses mesh vertices", () => {
    const pts = geomPoints({
      name: "m",
      type: "mesh",
      vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
      faces: [0, 1, 2, 0, 1, 3],
      pos: [0, 0, 0],
      quat_xyzw: [0, 0, 0, 1],
      rgba: [1, 1, 1, 1],
    });
    expect(pts).toHaveLength(4);
  });
});

describe("phaseAt", () => {
  it("returns the last phase that has started", () => {
    expect(phaseAt(0, TASK.phases)?.name).toBe("settle");
    expect(phaseAt(2.6, TASK.phases)?.name).toBe("lift");
    expect(phaseAt(11, TASK.phases)?.name).toBe("hold");
    expect(phaseAt(-1, TASK.phases)).toBeNull();
  });
});
