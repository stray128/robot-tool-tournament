/**
 * Tool families: process tools (nut runner, tamper pad) and vacuum area frames are Studio-only.
 * The per-part tournament pages refuse them, as rtt.effectors.tournament PART_FAMILIES and
 * per_part_presets keep them out; Studio pages show them with their rating and caveat.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ScreenView } from "@/components/studio/ScreenView";
import { parseStudioHero } from "@/components/studio/schema";
import { buildStory } from "@/components/studio/story";
import { StoryPage } from "@/components/story/StoryPage";
import { ALL_FAMILIES, ALL_FAMILY_TITLES, FAMILIES, isStudioOnlyTool, isVacuumFrame, parseEffectorIndex, parseEffectorPage } from "@/lib/effectors";
import { DataError } from "@/lib/validate";

import { STUDIO_TOOLS, sixHeroes } from "./helpers/heroes";

const REPO = path.join(__dirname, "..", "..");
const EFFECTORS = path.join(__dirname, "..", "public", "data", "effectors");
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

/** The values of rtt.effectors.types.EffectorFamily, read from the source. */
function pythonFamilies(): string[] {
  const src = readFileSync(path.join(REPO, "src", "rtt", "effectors", "types.py"), "utf8");
  const block = /class EffectorFamily\(enum\.StrEnum\):([\s\S]*?)\n\n\n/.exec(src)![1]!;
  return [...block.matchAll(/^\s+[A-Z_]+ = "([a-z_]+)"/gm)].map((m) => m[1]!);
}

describe("the family lists", () => {
  it("are the Python families, and the per-part ones leave the process tools out", () => {
    expect([...ALL_FAMILIES]).toEqual(pythonFamilies());
    expect([...FAMILIES]).toEqual(pythonFamilies().filter((f) => f !== "process_tool"));
    expect(ALL_FAMILY_TITLES.process_tool).toBe("Process tool");
  });

  it("know a Studio-only tool", () => {
    expect(isVacuumFrame("vacuum-frame-2x2-300")).toBe(true);
    expect(isVacuumFrame("cup-16mm")).toBe(false);
    expect(isStudioOnlyTool("process_tool", "nut-runner-socket-13")).toBe(true);
    expect(isStudioOnlyTool("suction", "vacuum-frame-2x3-600")).toBe(true);
    expect(isStudioOnlyTool("suction", "cup-25mm")).toBe(false);
    expect(isStudioOnlyTool("soft_arm", "vacuum-frame-x")).toBe(false);
  });
});

describe("per-part tournament pages keep Studio-only tools out", () => {
  const raw = (): Json => JSON.parse(readFileSync(path.join(EFFECTORS, "pipe_tee.json"), "utf8"));

  it("today's pages parse", () => {
    expect(() => parseEffectorPage(raw())).not.toThrow();
    expect(() => parseEffectorIndex(JSON.parse(readFileSync(path.join(EFFECTORS, "index.json"), "utf8")))).not.toThrow();
  });

  it("refuse a process tool", () => {
    const p = raw();
    p.recommendation.ranking[0].family = "process_tool";
    expect(() => parseEffectorPage(p)).toThrow(/process tool is Studio-only/);
    const idx = JSON.parse(readFileSync(path.join(EFFECTORS, "index.json"), "utf8"));
    idx.families.push({ family: "process_tool", title: "Process tool", fidelity: "approximation", available: true });
    expect(() => parseEffectorIndex(idx)).toThrow(DataError);
  });

  it("refuse a vacuum area frame", () => {
    const p = raw();
    p.recommendation.ranking[0].display_name = "Vacuum area frame, 2 x 2 cups, 300 mm";
    expect(() => parseEffectorPage(p)).toThrow(/vacuum area frame is Studio-only/);
    const q = raw();
    q.candidates[0].model = "vacuum-frame-2x2-300";
    expect(() => parseEffectorPage(q)).toThrow(/vacuum area frame is Studio-only/);
    const r = raw();
    r.lanes[0].effector.model = "vacuum-frame-2x3-600";
    expect(() => parseEffectorPage(r)).toThrow(/vacuum area frame is Studio-only/);
  });
});

describe("Studio pages show them with their rating and caveat", () => {
  const { pages, entries } = sixHeroes();
  /** The Studio-only tools the tile fixture lists (heroes.ts). */
  const TILE = STUDIO_TOOLS.filter((t) => ["tamper-pad-200", "vacuum-frame-2x2-300"].includes(t.model));

  it("the fixtures carry the Studio-only tools as the exporter writes them", () => {
    const tile = pages.find((p) => p.hero === "h6_tile")!;
    const studio = tile.screen.candidates.filter((c) => c.studio_only);
    expect(studio.map((c) => c.display_name)).toEqual(TILE.map((t) => t.display_name));
    for (const c of studio) {
      expect(c.rating).toMatch(/^rated /);
      expect(c.caveat).toMatch(/\(approximation\)/);
    }
  });

  it("the candidate list and every step card name the rating; the caveat opens", () => {
    const tile = pages.find((p) => p.hero === "h6_tile")!;
    const screen = renderToStaticMarkup(createElement(ScreenView, { screen: tile.screen, plan: tile.plan }));
    for (const t of TILE) {
      expect(screen).toContain(t.display_name);
      expect(screen).toContain(t.rating);
    }
    expect(screen.match(/data-tool-caveat/g)?.length).toBe(TILE.length);
    expect(screen).toContain("Studio only");
    const story = buildStory(tile);
    for (const s of story.steps) expect(s.ruledOut.filter((o) => o.rating).map((o) => o.rating)).toEqual(TILE.map((t) => t.rating));
    const html = renderToStaticMarkup(createElement(StoryPage, { page: tile, session: null, held: null, others: entries }));
    for (const t of TILE) expect(html).toContain(t.rating);
  });

  it("a Studio-only tool must say so and carry its caveat", () => {
    const raw: Json = JSON.parse(readFileSync(path.join(__dirname, "fixtures", "studio_v1", "data", "studio", "h2.json"), "utf8"));
    const c = { ...raw.screen.candidates[0], effector_id: "f".repeat(64), display_name: "Nut runner, 13 mm hex socket", family: "process_tool", family_title: "Process tool", status: "pruned", pruned_reason: "a test", pruned_steps: [] };
    const page = (cand: Json) => ({ ...raw, screen: { ...raw.screen, n_candidates: raw.screen.n_candidates + 1, candidates: [...raw.screen.candidates, cand] } });
    expect(() => parseStudioHero(page(c))).toThrow(/Studio-only tool and says so/);
    expect(() => parseStudioHero(page({ ...c, studio_only: true }))).toThrow(/carries its caveat/);
    expect(() => parseStudioHero(page({ ...c, studio_only: true, caveat: "Nut runner (approximation): x" }))).not.toThrow();
    const cup = raw.screen.candidates.find((x: Json) => x.family === "suction");
    if (cup) expect(() => parseStudioHero({ ...raw, screen: { ...raw.screen, candidates: raw.screen.candidates.map((x: Json) => (x === cup ? { ...x, rating: "rated 1 kg" } : x)) } })).toThrow(/only a Studio-only tool states a rating/);
  });
});
