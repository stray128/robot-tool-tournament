import { describe, expect, it } from "vitest";

import { holdTrack, liftTrack, referenceFrame, summarizeTracks } from "@/lib/track.mjs";

/** Frames for `n` frames of `bodies` bodies with body `b` at height z(f). */
function frames(n: number, bodies: number, b: number, z: (f: number) => number): Float32Array {
  const out = new Float32Array(n * bodies * 7);
  for (let f = 0; f < n; f++) {
    for (let k = 0; k < bodies; k++) out[(f * bodies + k) * 7 + 6] = 1; // identity quaternions
    out[(f * bodies + b) * 7 + 2] = z(f);
  }
  return out;
}

describe("overlay tracks", () => {
  it("places the reference at the start of DESCEND and clamps it to the recording", () => {
    expect(referenceFrame(30, 0.3, 181)).toBe(9);
    expect(referenceFrame(30, 100, 50)).toBe(49);
    expect(referenceFrame(30, -1, 50)).toBe(0);
  });

  it("measures lift of one body relative to its reference frame", () => {
    const fr = frames(5, 3, 2, (f) => 0.06 + 0.01 * f);
    const lift = liftTrack(fr, 3, 2, 1);
    expect(Array.from(lift).map((v) => Number(v.toFixed(4)))).toEqual([-0.01, 0, 0.01, 0.02, 0.03]);
  });

  it("counts continuous time above the threshold and resets when the object drops", () => {
    const lift = new Float32Array([0, 0.11, 0.12, 0.12, 0.05, 0.11, 0.11]);
    const hold = holdTrack(lift, 10, 0.1);
    expect(Array.from(hold).map((v) => Number(v.toFixed(3)))).toEqual([0, 0, 0.1, 0.2, 0, 0, 0.1]);
    const { maxLift, maxHold } = summarizeTracks(lift, hold);
    expect(maxLift).toBeCloseTo(0.12, 6);
    expect(maxHold).toBeCloseTo(0.2, 6);
  });

  it("returns empty tracks for an empty recording", () => {
    expect(liftTrack(new Float32Array(0), 4, 3, 0)).toHaveLength(0);
  });
});
