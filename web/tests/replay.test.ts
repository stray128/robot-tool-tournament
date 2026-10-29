import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  decodeFrames,
  decodeQuantised,
  encodeQuantised,
  geomPoints,
  isParkedGeom,
  parseReplayHeaderV2,
  phaseAt,
  poseBytes,
  Replay,
  ReplayFormatError,
  rotate,
  slerp,
} from "@/lib/replay";
import { replayPayload } from "@/lib/assets";
import type { ReplayHeader, ReplayHeaderV2, TaskSummary } from "@/lib/types";
import { DataError, parseReplayHeader } from "@/lib/validate";

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

// ---------------------------------------------------------------- cut-aware lanes (Studio v2)

/** The published v1 Studio lanes, kept as fixtures (tests/fixtures/studio_v1) when v2 replaced them. */
const STUDIO_REPLAYS = path.join(__dirname, "fixtures", "studio_v1", "data", "studio", "replays");

/** A real v2 Studio lane header, re-sized to `nFrames` frames of one body, with `extra` keys. */
function laneHeader(nFrames: number, extra: Record<string, unknown> = {}): ReplayHeaderV2 {
  const raw = JSON.parse(readFileSync(path.join(STUDIO_REPLAYS, "h1", "recommended.json"), "utf8"));
  const body = raw.bodies.find((b: { visuals: unknown[] }) => b.visuals.length > 0);
  const assets = Object.fromEntries(body.visuals.map((v: { asset: string }) => [v.asset, raw.assets[v.asset]]));
  // A keyframed lane says how it was keyframed, at the recording's rate (rtt.studio.replay).
  const fps = (extra.fps as number | undefined) ?? raw.fps;
  const meta = "t" in extra ? { ...raw.meta, keyframes: { recorded_fps: fps, samples: 10_000, tolerance_mm: 1, max_error_vs_recording_mm: 0.9 } } : raw.meta;
  return parseReplayHeaderV2({ ...raw, n_frames: nFrames, n_bodies: 1, bodies: [body], static_visuals: [], assets, meta, ...extra });
}

/** Frames of one body moving along x, identity rotation: positions in metres. */
function alongX(xs: number[]): Float32Array {
  const out = new Float32Array(xs.length * 7);
  xs.forEach((x, i) => out.set([x, 0, 0, 0, 0, 0, 1], i * 7));
  return out;
}

function xAt(replay: Replay, t: number): number {
  const pos = [0, 0, 0];
  replay.sample(0, replay.frameAt(t), pos, [0, 0, 0, 1]);
  return pos[0]!;
}

describe("cut-aware sampling", () => {
  it("sample does not interpolate across a cut", () => {
    // Frames 0 and 1 end segment 1; frame 2 starts segment 2 one metre away (a tool change).
    const frames = alongX([0, 0.1, 1.0, 1.1]);
    const cut = new Replay(laneHeader(4, { cuts: [2] }), frames);
    const pos = [0, 0, 0];
    const quat = [0, 0, 0, 1];
    cut.sample(0, 1.5, pos, quat);
    expect(pos[0]).toBeCloseTo(0.1, 6); // holds the last pose of segment 1
    cut.sample(0, 1.999, pos, quat);
    expect(pos[0]).toBeCloseTo(0.1, 6);
    cut.sample(0, 2, pos, quat);
    expect(pos[0]).toBeCloseTo(1.0, 6); // then jumps
    cut.sample(0, 2.5, pos, quat);
    expect(pos[0]).toBeCloseTo(1.05, 6); // and interpolates inside segment 2
    cut.sample(0, 0.5, pos, quat);
    expect(pos[0]).toBeCloseTo(0.05, 6);
    expect(cut.isCut(2)).toBe(true);
    expect(cut.isCut(1)).toBe(false);

    // The same frames without the cut interpolate straight through (today's lanes).
    const plain = new Replay(laneHeader(4), frames);
    plain.sample(0, 1.5, pos, quat);
    expect(pos[0]).toBeCloseTo(0.55, 6);
  });

  it("reads a non-uniform timeline and holds through the change time", () => {
    // Keys at 0, 0.1 s; a 7.5 s tool change; keys at 7.6, 7.7 s.
    const t = [0, 0.1, 7.6, 7.7];
    const replay = new Replay(laneHeader(4, { fps: 100, t, cuts: [2] }), alongX([0, 0.1, 1.0, 1.1]));
    expect(replay.duration).toBeCloseTo(7.7, 9);
    expect(replay.frameAt(0.05)).toBeCloseTo(0.5, 9);
    expect(replay.frameAt(7.65)).toBeCloseTo(2.5, 9);
    expect(replay.timeOf(2)).toBe(7.6);
    expect(xAt(replay, 0.05)).toBeCloseTo(0.05, 6);
    for (const s of [0.1, 1, 4, 7.59]) expect(xAt(replay, s)).toBeCloseTo(0.1, 6);
    expect(xAt(replay, 7.6)).toBeCloseTo(1.0, 6);
    expect(xAt(replay, 7.65)).toBeCloseTo(1.05, 6);
    expect(replay.cutGapAt(4)).toEqual({ cut: 2, start_s: 0.1, end_s: 7.6 });
    expect(replay.cutGapAt(7.65)).toBeNull();
    expect(replay.cutGapAt(0.05)).toBeNull();
  });

  it("validates timelines and cuts", () => {
    expect(() => laneHeader(4, { t: [0, 0.1, 0.2] })).toThrow(DataError); // one time per frame
    expect(() => laneHeader(3, { t: [0, 0.2, 0.1] })).toThrow(DataError); // never decreasing
    expect(() => laneHeader(3, { cuts: [3] })).toThrow(DataError); // inside the recording
    expect(() => laneHeader(3, { cuts: [2, 1] })).toThrow(DataError); // increasing
    expect(() => laneHeader(3, { t: [0, 0.5, 0.5] })).toThrow(DataError); // strictly increasing
    const ok = laneHeader(3, { t: [0, 0.5, 0.6], cuts: [0, 2] });
    expect(ok.t).toEqual([0, 0.5, 0.6]);
    expect(ok.cuts).toEqual([0, 2]);
    expect(ok.quantised).toBe(false);
  });

  it("keeps reading the published v1 uniform, unquantised Studio lanes", () => {
    for (const hero of ["h1", "h2", "h3"]) {
      for (const lane of ["recommended", "best_single"]) {
        const h = parseReplayHeaderV2(JSON.parse(readFileSync(path.join(STUDIO_REPLAYS, hero, `${lane}.json`), "utf8")));
        expect(h.t).toBeNull();
        expect(h.cuts).toEqual([]);
        expect(h.quantised).toBe(false);
        expect(poseBytes(h)).toBe(readFileSync(path.join(STUDIO_REPLAYS, hero, h.bin)).byteLength);
        const buf = readFileSync(path.join(STUDIO_REPLAYS, hero, h.bin));
        const replay = new Replay(h, decodeFrames(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), h));
        expect(replay.duration).toBeCloseTo((h.n_frames - 1) / h.fps, 9);
        expect(replay.frameAt(1)).toBeCloseTo(h.fps, 9);
        expect(replay.cutGapAt(1)).toBeNull();
      }
    }
  });
});

/** A deterministic pseudo-random stream (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random poses: positions within +-2 m, unit quaternions (with `halfTurns`, all within |w| < 0.001). */
function randomPoses(n: number, seed: number, halfTurns = false): Float32Array {
  const r = rng(seed);
  const out = new Float32Array(n * 7);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  for (let i = 0; i < n; i++) {
    const q = [gauss(), gauss(), gauss(), halfTurns ? 0 : gauss()];
    if (halfTurns) q[3] = (r() - 0.5) * 0.002 * Math.hypot(q[0]!, q[1]!, q[2]!);
    const len = Math.hypot(...q);
    out.set([(r() - 0.5) * 4, (r() - 0.5) * 4, r() * 2, ...q.map((c) => c / len)], i * 7);
  }
  return out;
}

/** Round-trip error of one pose: the distance between the origins (m) and the rotation angle between them (rad). */
function poseError(a: ArrayLike<number>, b: ArrayLike<number>, o: number): { pos: number; rad: number } {
  const dot = Math.abs(a[o + 3]! * b[o + 3]! + a[o + 4]! * b[o + 4]! + a[o + 5]! * b[o + 5]! + a[o + 6]! * b[o + 6]!);
  return {
    pos: Math.hypot(a[o]! - b[o]!, a[o + 1]! - b[o + 1]!, a[o + 2]! - b[o + 2]!),
    rad: 2 * Math.acos(Math.min(1, dot)),
  };
}

describe("quantised lanes", () => {
  /** Worst position (m) and angle (rad) error of a round trip through the int16 payload. */
  function roundTrip(poses: Float32Array): { pos: number; rad: number } {
    const n = poses.length / 7;
    const decoded = decodeQuantised(encodeQuantised(poses), n);
    let pos = 0;
    let rad = 0;
    for (let i = 0; i < n; i++) {
      const e = poseError(poses, decoded, i * 7);
      pos = Math.max(pos, e.pos);
      rad = Math.max(rad, e.rad);
    }
    return { pos, rad };
  }

  /** A pose at `p` rotated `deg` degrees about the unit axis `axis`. */
  function about(axis: [number, number, number], deg: number, p: [number, number, number] = [0.1, -0.2, 0.3]): number[] {
    const h = (deg * Math.PI) / 360;
    const n = Math.hypot(...axis);
    return [...p, (axis[0] / n) * Math.sin(h), (axis[1] / n) * Math.sin(h), (axis[2] / n) * Math.sin(h), Math.cos(h)];
  }

  it("quantised lane round-trips within 0.1 mm and 0.002 rad for every orientation", () => {
    // Uniform random rotations, plus rotations within 0.1 degrees of a half turn (w close to 0).
    for (const [seed, halfTurns] of [
      [7, false],
      [11, true],
    ] as const) {
      const e = roundTrip(randomPoses(4000, seed, halfTurns));
      expect(e.pos).toBeLessThan(1e-4);
      expect(e.rad).toBeLessThan(0.002);
    }
  });

  it("holds at a half turn about x, 1 degree away from it, and at the identity", () => {
    const cases = [
      about([1, 0, 0], 180), // a tool pointing straight down
      about([1, 0, 0], 179), // 1 degree away
      about([1, 0, 0], 181),
      about([1, 0.02, 0], 180), // a half turn about a slightly tilted axis
      about([0, 0, 1], 180),
      about([0, 1, 0], 90),
      about([1, 1, 1], 120), // all four components equal in size
      about([1, 0, 0], 0), // the identity
    ];
    for (const c of cases) {
      const e = roundTrip(new Float32Array(c));
      expect(e.pos, JSON.stringify(c)).toBeLessThan(1e-4);
      expect(e.rad, JSON.stringify(c)).toBeLessThan(0.002);
    }
  });

  it("stores seven int16 words: position as round(mm x 10), three components and the dropped index", () => {
    // a quarter turn about z written with a negative w: the largest is z (index 2) or w; after the
    // sign flip the largest is positive
    const s = Math.SQRT1_2;
    const words = new Int16Array(encodeQuantised(new Float32Array([0.12345, -0.5, 1.0, 0, 0, -s * 1.0001, -s])));
    expect(words.length).toBe(7);
    expect(Array.from(words.slice(0, 3))).toEqual([1235, -5000, 10000]);
    expect(words[6]).toBe(2); // z is the largest magnitude
    const w = s / Math.hypot(s * 1.0001, s); // w after normalising, flipped positive with z
    expect(Array.from(words.slice(3, 6))).toEqual([0, 0, Math.round((w / 0.70710678) * 32767)]); // x, y, w
    const back = decodeQuantised(words.buffer, 1);
    expect(back[0]).toBeCloseTo(0.1235, 7);
    expect(back[5]).toBeCloseTo(s, 3);
    expect(back[6]).toBeCloseTo(s, 3);
    expect(encodeQuantised(new Float32Array(14)).byteLength).toBe(28);
  });

  it("decodes a quantised payload through the header into the float layout", () => {
    const n = 40;
    const poses = randomPoses(n, 3);
    const header = laneHeader(n, { quantised: true, dtype: "int16", pose_fields: ["px", "py", "pz", "qa", "qb", "qc", "qi"], cuts: [20] });
    expect(header.quantised).toBe(true);
    const buf = encodeQuantised(poses);
    expect(buf.byteLength).toBe(poseBytes(header));
    expect(poseBytes(header)).toBe(n * 14);
    const replay = new Replay(header, decodeFrames(buf, header));
    const pos = [0, 0, 0];
    const quat = [0, 0, 0, 1];
    for (let i = 0; i < n; i++) {
      replay.sample(0, i, pos, quat);
      expect(Math.hypot(pos[0]! - poses[i * 7]!, pos[1]! - poses[i * 7 + 1]!, pos[2]! - poses[i * 7 + 2]!)).toBeLessThan(1e-4);
    }
    // Still never interpolated across the cut.
    replay.sample(0, 19.5, pos, quat);
    expect(pos[0]).toBeCloseTo(replay.frames[19 * 7]!, 6);
    expect(() => decodeFrames(new ArrayBuffer(n * 28), header)).toThrow(ReplayFormatError);
    // the same lane unquantised is float32, 28 bytes a pose
    expect(poseBytes({ ...header, quantised: false })).toBe(n * 28);
  });

  it("a quantised header names exactly its seven int16 words", () => {
    const words = ["px", "py", "pz", "qa", "qb", "qc", "qi"];
    expect(laneHeader(2, { quantised: true, dtype: "int16", pose_fields: words }).quantised).toBe(true);
    expect(laneHeader(2, { quantised: false }).quantised).toBe(false);
    expect(() => laneHeader(2, { quantised: true })).toThrow(DataError); // the float32 field names
    expect(() => laneHeader(2, { quantised: true, pose_fields: ["px", "py", "pz", "q0", "q1", "q2", "q3"] })).toThrow(DataError);
    expect(() => laneHeader(2, { quantised: true, pose_fields: [...words].reverse() })).toThrow(DataError);
    expect(() => laneHeader(2, { dtype: "int16" })).toThrow(DataError); // int16 only with the flag
    expect(() => laneHeader(2, { quantised: true, dtype: "float32", pose_fields: words })).toThrow(DataError); // and the flag only with int16
    expect(() => laneHeader(2, { pose_fields: words })).toThrow(DataError); // int16 words only with the flag
  });

  it("the asset payload counts quantised poses at 14 bytes", () => {
    const header = laneHeader(10, { quantised: true, dtype: "int16", pose_fields: ["px", "py", "pz", "qa", "qb", "qc", "qi"] });
    expect(replayPayload(header, 100).poses).toBe(10 * 14);
    expect(replayPayload(laneHeader(10), 100).poses).toBe(10 * 28);
  });
});
