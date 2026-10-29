/**
 * The Studio lane format on the site's side, read from lanes the current Python exporter wrote
 * (tests/fixtures/studio_lanes/, made and shape-checked by tests/test_web_lane_contract.py): a tool
 * change with a cut, as float32 and as int16, and their page entries. Plus what JSON Schema
 * cannot say: frame times strictly increase, cuts are sorted, unique and inside the lane, the
 * clock ends on the last frame time, and a keyframed lane's fps is the recording's rate.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { changeCardAt, laneFramesText, laneHeaderProblems, parseStudioLane, type StudioLane } from "@/components/studio/schema";
import { decodeFrames, parseReplayHeaderV2, Replay } from "@/lib/replay";
import type { ReplayHeaderV2 } from "@/lib/types";
import { DataError } from "@/lib/validate";

const DIR = path.join(__dirname, "fixtures", "studio_lanes");
const HERO = "fixture";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

const rawHeader = (stem: string): Json => JSON.parse(readFileSync(path.join(DIR, `${stem}.json`), "utf8"));
const rawEntry = (stem: string): Json => JSON.parse(readFileSync(path.join(DIR, "entries.json"), "utf8"))[stem];
const bin = (stem: string): ArrayBuffer => {
  const buf = readFileSync(path.join(DIR, `${stem}.bin`));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
};

function load(stem: string): { header: ReplayHeaderV2; lane: StudioLane; replay: Replay } {
  const header = parseReplayHeaderV2(rawHeader(stem), `${stem}.json`);
  const lane = parseStudioLane(rawEntry(stem), HERO, "entries.json");
  return { header, lane, replay: new Replay(header, decodeFrames(bin(stem), header)) };
}

describe("lanes written by the current Python exporter", () => {
  for (const stem of ["recommended", "recommended_int16"]) {
    it(`${stem}: the header, the page entry and the poses agree`, () => {
      const { header, lane, replay } = load(stem);
      expect(header.quantised).toBe(stem.endsWith("int16"));
      expect(lane.quantised).toBe(header.quantised);
      expect(header.t).not.toBeNull();
      expect(header.cuts.length).toBeGreaterThan(0);
      const kf = header.meta.keyframes!;
      expect(kf.recorded_fps).toBe(header.fps);
      expect([1, 1.5, 2]).toContain(kf.tolerance_mm);
      expect(lane.max_error_vs_recording_mm).toBe(kf.max_error_vs_recording_mm);
      expect(laneHeaderProblems(lane, header)).toEqual([]);
      // the clock is the frame times, not the frames at fps
      expect(replay.duration).toBeCloseTo(header.t!.at(-1)!, 9);
      expect(lane.duration_s).toBeCloseTo(header.t!.at(-1)!, 3);
      expect((header.n_frames - 1) / header.fps).toBeLessThan(replay.duration / 2);
      expect(laneFramesText(lane)).toMatch(new RegExp(`^${header.n_frames} keyframes from a 100 Hz recording, within \\d\\.\\d mm of it`));
    });

    it(`${stem}: never interpolates across the cut, and shows the card in the gap`, () => {
      const { header, lane, replay } = load(stem);
      const cut = header.cuts[0]!;
      const before = header.t![cut - 1]!;
      const after = header.t![cut]!;
      expect(after - before).toBeGreaterThan(1); // the tool change is not simulated
      const pos = [0, 0, 0];
      const quat = [0, 0, 0, 0];
      const held = [0, 0, 0];
      for (let b = 0; b < header.n_bodies; b++) {
        replay.sample(b, cut - 1, held, quat);
        replay.sample(b, replay.frameAt(0.5 * (before + after)), pos, quat);
        expect(pos).toEqual(held);
      }
      const mid = 0.5 * (before + after);
      expect(replay.cutGapAt(mid)).toEqual({ cut, start_s: before, end_s: after });
      const card = changeCardAt(lane, mid, replay.cutGapAt(mid));
      expect(card?.title).toBe("Changing tool");
      expect(card?.next).toMatch(/^Next: tool 2, /);
      expect(changeCardAt(lane, before - 0.5, replay.cutGapAt(before - 0.5))).toBeNull();
    });
  }

  it("the int16 lane shows the float32 lane's run: each is within 1 mm of the recording", () => {
    const wide = load("recommended");
    const narrow = load("recommended_int16");
    const a = [0, 0, 0];
    const b = [0, 0, 0];
    const q = [0, 0, 0, 0];
    for (const t of [0.5, 2, 5, 14, 18]) {
      for (let body = 0; body < wide.header.n_bodies; body++) {
        wide.replay.sample(body, wide.replay.frameAt(t), a, q);
        narrow.replay.sample(body, narrow.replay.frameAt(t), b, q);
        // two keyframings of one recording, each within its tolerance of it
        expect(Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!)).toBeLessThan(2.1e-3);
      }
    }
  });
});

describe("what the site checks that JSON Schema cannot", () => {
  const header = (mutate: (h: Json) => void): Json => {
    const h = structuredClone(rawHeader("recommended"));
    mutate(h);
    return h;
  };
  const refused: [string, (h: Json) => void][] = [
    ["a frame time that does not increase", (h) => (h.t[5] = h.t[4])],
    ["a frame time that goes back", (h) => (h.t[5] = h.t[3])],
    ["one time too few", (h) => h.t.pop()],
    ["cuts out of order", (h) => (h.cuts = [h.cuts[0], 3])],
    ["a repeated cut", (h) => (h.cuts = [h.cuts[0], h.cuts[0]])],
    ["a cut past the last frame", (h) => (h.cuts = [h.n_frames])],
    ["keyframes under the old name", (h) => (h.meta.keyframes = { ...h.meta.keyframes, max_error_mm: 0.5 })],
    ["the old name alone", (h) => (h.meta.keyframes = { recorded_fps: 100, samples: 9999, tolerance_mm: 1, max_error_mm: 0.5 })],
    ["a tolerance off the ladder", (h) => (h.meta.keyframes.tolerance_mm = 3)],
    ["an error over the tolerance", (h) => (h.meta.keyframes.max_error_vs_recording_mm = 1.2)],
    ["fps that is not the recording's", (h) => (h.fps = 30)],
    ["more keys than samples", (h) => (h.meta.keyframes.samples = 3)],
    ["frame times without keyframes", (h) => delete h.meta.keyframes],
    ["keyframes without frame times", (h) => delete h.t],
  ];
  it.each(refused)("refuses %s", (_name, mutate) => {
    expect(() => parseReplayHeaderV2(header(mutate))).toThrow(DataError);
  });

  const entry = (mutate: (e: Json) => void): Json => {
    const e = structuredClone(rawEntry("recommended"));
    mutate(e);
    return e;
  };
  it("refuses a page entry under the old field name", () => {
    expect(() => parseStudioLane(entry((e) => (e.max_error_mm = e.max_error_vs_recording_mm)), HERO)).toThrow(/renamed max_error_vs_recording_mm/);
  });
  it("refuses page entry times that do not increase", () => {
    expect(() => parseStudioLane(entry((e) => (e.t = [0, 1, 1])), HERO)).toThrow(DataError);
  });

  const problems = (mutate: (e: Json) => void): string[] =>
    laneHeaderProblems(parseStudioLane(entry(mutate), HERO), parseReplayHeaderV2(rawHeader("recommended")));
  it("names every disagreement between the page entry and its replay", () => {
    expect(problems(() => {})).toEqual([]);
    expect(problems((e) => (e.duration_s = (e.n_frames - 1) / e.fps))).toEqual([expect.stringMatching(/^lasts .* last frame is at/)]);
    expect(problems((e) => (e.fps = 30))[0]).toMatch(/frames at 100 fps; the page says .* at 30/);
    expect(problems((e) => (e.quantised = true))).toEqual([expect.stringMatching(/poses are float32; the page says int16/)]);
    expect(problems((e) => (e.max_error_vs_recording_mm = 0.5))).toEqual([expect.stringMatching(/error against the recording/)]);
    expect(problems((e) => delete e.max_error_vs_recording_mm)).toEqual([expect.stringMatching(/states its error against the recording/)]);
    expect(problems((e) => (e.segments[1].start_s += 0.5)).join(" ")).toMatch(/is not at segment 1's start/);
    expect(problems((e) => e.segments.pop()).join(" ")).toMatch(/1 cuts for 1 segments/);
  });
});
