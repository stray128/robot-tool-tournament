/**
 * Round trip on the real v2 data: the pages and lanes scripts/export_studio_web.py wrote from the
 * Studio searches of the v2 heroes (public/data/studio, the recorded v2 plans run locally) are read
 * by the site's own parsers: every page by parseStudioHero, every lane entry by parseStudioLane, every
 * replay header by parseReplayHeaderV2 and laneHeaderProblems, every pose file decoded. What the v2
 * export adds is then checked on the pages that have it: a rigid pick with no rigid runner-up, the
 * baseline beside the pick, Studio-only tools with their rating and caveat, a closest setup whose
 * steps carry their tool's caveat, the blocker of a task nothing completes, and pruning reasons in
 * plain words.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { brandTokens } from "@/app/studio/data";
import { brandPattern, changeCardAt, type HeroPage, laneHeaderProblems, parseStudioHero, parseStudioIndex, parseStudioLane, PRODUCT_MODE_LABEL } from "@/components/studio/schema";
import { buildStory, focusOf } from "@/components/studio/story";
import { decodeFrames, parseReplayHeaderV2, poseBytes } from "@/lib/replay";

const WEB = path.join(__dirname, "..");
const DATA = path.join(WEB, "public", "data");
const INDEX = path.join(DATA, "studio", "index.json");
const exported = existsSync(INDEX);
const BRAND = brandPattern(brandTokens(path.join(WEB, "..")));

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;
const json = (rel: string): Json => JSON.parse(readFileSync(path.join(DATA, rel), "utf8"));

describe.skipIf(!exported)("the real v2 export, read by the site", () => {
  const index = parseStudioIndex(json("studio/index.json"), "studio/index.json", { brand: BRAND });
  const raw = new Map<string, Json>(index.heroes.map((h) => [h.hero, json(h.page)]));
  const pages = new Map<string, HeroPage>(index.heroes.map((h) => [h.hero, parseStudioHero(raw.get(h.hero), h.page, { brand: BRAND })]));

  it("lists the v2 heroes by case id, each page from the v2 recording", () => {
    expect(index.heroes.length).toBeGreaterThan(0);
    for (const h of index.heroes) {
      expect(h.hero).toBe(h.case_id);
      const page = pages.get(h.hero)!;
      expect(page.label).toBe(PRODUCT_MODE_LABEL);
      expect(page.plan.task_id).toBe(h.task_id);
    }
  });

  it("parses every lane entry, its replay header and its poses", () => {
    for (const [hero, page] of pages) {
      expect(page.lanes.length, hero).toBeGreaterThan(0);
      raw.get(hero).lanes.forEach((entry: Json, i: number) => {
        const lane = parseStudioLane(entry, hero, `studio/${hero}.json`);
        expect(lane).toEqual(page.lanes[i]);
        const header = parseReplayHeaderV2(json(`${lane.replay}.json`), `${lane.replay}.json`);
        expect(laneHeaderProblems(lane, header), `${hero}/${lane.id}`).toEqual([]);
        const buf = readFileSync(path.join(DATA, path.dirname(lane.replay), header.bin));
        expect(buf.byteLength).toBe(poseBytes(header));
        const frames = decodeFrames(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), header);
        expect(frames.every((v) => Number.isFinite(v))).toBe(true);
        // a lane that changes tool shows the card for its whole gap
        for (const seg of lane.segments.filter((s) => s.change !== "none")) {
          expect(changeCardAt(lane, 0.5 * (seg.change_start_s + seg.start_s))?.title).toBe("Changing tool");
        }
      });
    }
  });

  it("a rigid pick names no rigid runner-up; an approximation pick says what its runner-up is", () => {
    for (const [hero, page] of pages) {
      const rec = page.recommendation;
      if (!rec) continue;
      if (rec.fidelity === "rigid") expect(rec.runner_up_rigid, hero).toBeNull();
      else expect(rec.runner_up_rigid?.basis, hero).toBeTruthy();
    }
  });

  it("the baseline is beside the pick, or the page says why there is none", () => {
    for (const [hero, page] of pages) {
      if (!page.baseline) continue;
      if (page.baseline.row) expect(page.finalists.some((f) => f.setup_id === page.baseline!.row!.setup_id), hero).toBe(true);
      else expect(page.baseline.basis, hero).toBeTruthy();
      const vs = page.recommendation?.versus_baseline;
      if (vs && page.baseline.row) expect(vs.same_setup).toBe(page.baseline.row.setup_id === page.recommendation!.setup_id);
    }
  });

  it("Studio-only tools carry their rating and caveat wherever they appear", () => {
    for (const [hero, page] of pages) {
      for (const c of page.screen.candidates.filter((x) => x.studio_only)) {
        expect(c.caveat, `${hero}: ${c.display_name}`).toBeTruthy();
      }
      for (const t of page.recommendation?.tools ?? []) if (t.studio_only) expect(t.caveat && t.rating, `${hero}: ${t.display_name}`).toBeTruthy();
    }
  });

  it("a task nothing completes shows its closest setup, step by step with its tools' caveats, and the blocker", () => {
    for (const [hero, page] of pages) {
      if (page.recommendation) continue;
      const focus = focusOf(page)!;
      expect(focus.role, hero).toBe("closest");
      expect(page.no_winner?.blocker, hero).not.toBeNull();
      for (const step of focus.steps) {
        const tool = focus.tools[step.tool];
        const caveat = tool && typeof tool !== "string" ? tool.caveat : null;
        if (caveat) expect(step.caveats, `${hero}/${step.step_id}`).toContain(caveat);
      }
      const story = buildStory(page);
      expect(story.answer.startsWith("No setup finishes this task.")).toBe(true);
    }
  });

  it("writes pruning reasons in plain words, with no internal code", () => {
    for (const [hero, page] of pages) {
      const reasons = [
        ...page.screen.candidates.flatMap((c) => [c.pruned_reason, ...c.pruned_steps.map((p) => p.reason)]),
        ...page.screen.pruned_reasons.map((r) => r.reason),
        ...page.matrix.rows.flatMap((r) => Object.values(r.cells).map((c) => ("reason" in c ? c.reason : null))),
      ].filter((r): r is string => typeof r === "string");
      for (const r of reasons) expect(r, hero).not.toMatch(/[a-z0-9]_[a-z0-9]/i);
    }
  });
});
