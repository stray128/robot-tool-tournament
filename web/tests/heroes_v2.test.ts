/**
 * The six v2 heroes' story pages (fixtures: tests/helpers/heroes.ts until the v2 export lands):
 * each shows a recommendation or the closest setup, the baseline beside it, its caveats in plain
 * words and the "Changing tool" card wherever its replay changes tool; on disk they pass every
 * build-time check, and a hero the export leaves out (the cut order is the hotel, then the tile)
 * leaves no broken page or link behind.
 */

import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, describe, expect, it } from "vitest";

import { heroLinks } from "@/app/atlas/data";
import {
  type StudioDirs,
  featuredHero,
  getStudioHero,
  getStudioHeroIds,
  getStudioIndex,
  getStudioSession,
  legacyHeroRoutes,
} from "@/app/studio/data";
import { StoryPage } from "@/components/story/StoryPage";
import { changeCardAt, type HeroPage, parseStudioHero } from "@/components/studio/schema";
import { baselineStory, buildStory, focusOf } from "@/components/studio/story";
import { simStatus } from "@/lib/atlas";
import { getAtlas } from "@/app/atlas/data";

import { LID_FALL_CAVEAT, rawHero, sixHeroes, V1_FIXTURE, V2_HEROES, writeHeroFixtures } from "./helpers/heroes";

const REPO = path.join(__dirname, "..", "..");
const tmp: string[] = [];
afterAll(() => {
  for (const d of tmp) rmSync(d, { recursive: true, force: true });
});

/** A data folder with the v2 hero fixtures on disk (the v1 fixture's site root serves the GLBs their v1 lanes draw). */
function fixtureDirs(omit: string[] = []): StudioDirs {
  const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-heroes-"));
  tmp.push(dir);
  writeHeroFixtures(dir, { omit });
  return { data: dir, site: V1_FIXTURE.site, repo: REPO };
}

const render = (page: HeroPage, others = sixHeroes().entries.filter((e) => e.hero !== page.hero)) =>
  renderToStaticMarkup(createElement(StoryPage, { page, session: null, held: null, others }));

describe("the six story pages", () => {
  const { pages } = sixHeroes();

  for (const page of pages) {
    it(`${page.hero}: the baseline sits beside the ${page.recommendation ? "recommendation" : "closest setup"}`, () => {
      const html = render(page);
      const b = baselineStory(page)!;
      expect(b).not.toBeNull();
      expect(html).toContain("The baseline: the best single tool");
      expect(html).toContain(`data-baseline="${b.row ? b.row.setup_id : "none"}"`);
      if (page.recommendation) {
        expect(b.row?.setup_id).toBe(page.best_single!.setup_id);
        expect(b.offTheShelf).toBe(true);
        expect(html).toContain("also the best single off-the-shelf tool");
        expect(b.lines.join(" ")).toMatch(/runs against \d+ of \d+/);
        expect(html).toContain("data-head-to-head");
      } else {
        expect(b.row).toBeNull();
        expect(html).toContain("No single tool is eligible");
        expect(b.lines[0]).toBe("No single tool is eligible: none finished the task on the condition grid.");
        expect(b.lines.slice(1).every((l) => / finishes 0 of \d+ runs: it /.test(l))).toBe(true);
      }
      // the honest labels on every story page, and no code on screen
      expect(html).toContain("Simulated, verify on hardware");
      expect(html).toContain("Product mode: not part of the pre-registered experiment");
      expect(html).not.toMatch(/>[^<]*\b(?:out_of_reach|physics_unreliable|criterion_not_met|constraint_exceeded|not_reached)\b/);
    });

    it(`${page.hero}: every step shows its tier and, off the rigid tier, its caveat`, () => {
      const story = buildStory(page);
      for (const s of story.steps) {
        expect(s.pick).not.toBeNull();
        if (s.pick!.fidelity !== "rigid") expect(s.notes.some((n) => n.kind === "model"), `${page.hero}/${s.step.id}`).toBe(true);
      }
      const html = render(page);
      expect(html.match(/data-step-card=/g)?.length).toBe(page.plan.steps.length);
      expect((html.match(/data-fidelity=/g)?.length ?? 0) >= page.plan.steps.length).toBe(true);
    });

    it(`${page.hero}: the replay shows "Changing tool" through every change`, () => {
      for (const lane of page.lanes) {
        for (const seg of lane.segments.slice(1)) {
          const mid = 0.5 * (seg.change_start_s + seg.start_s);
          const card = changeCardAt(lane, mid);
          expect(card?.title, `${page.hero}/${lane.id} segment ${seg.index}`).toBe("Changing tool");
          expect(card?.next).toBe(`Next: tool ${seg.tool + 1}, ${seg.tool_label}`);
          expect(changeCardAt(lane, seg.start_s + 0.01)).toBeNull();
        }
      }
    });
  }

  it("the tote's lid that fell shut, the CNC's stiff button and the hotel's reach read in words", () => {
    const tote = pages.find((p) => p.hero === "h1_tote")!;
    const close = buildStory(tote).steps.find((s) => s.step.id === "close_lid")!;
    // the run's own measurement, not the caveat's rounded words (the caveat is the same event)
    expect(close.notes.filter((n) => n.kind === "run").map((n) => n.label)).toEqual(["lid fell shut from 21° at 317°/s"]);
    expect(render(tote)).toContain("lid fell shut from 21° at 317°/s");
    // the caveat's own rounded "about 320" is the same event, so it is not listed twice
    expect(render(tote)).not.toContain("lid fell shut from 21° at about 320°/s");
    expect(tote.recommendation!.steps.find((s) => s.step_id === "close_lid")!.caveats).toContain(LID_FALL_CAVEAT);
    const cnc = pages.find((p) => p.hero === "h4_cnc")!;
    const stop = cnc.best_single!.stops_at!;
    const baseLane = cnc.lanes.find((l) => l.id === "best_single")!;
    expect(baseLane.steps.find((s) => s.step_id === stop)!.failure_mode).toBe("criterion_not_met");
    const hotel = pages.find((p) => p.hero === "h7_hotel")!;
    const last = buildStory(hotel).steps.at(-1)!;
    expect(last.why[0]).toBe("This is where it stops: in 8 of 8 runs it was out of the arm's reach.");
    expect(last.notes.map((n) => n.label)).toContain("out of reach: the palm would need x 364 mm, y -12 mm (travel ±350 mm)");
    const solar = pages.find((p) => p.hero === "h5_solar")!;
    expect(render(solar)).toContain("stopped 41 mm above the bin floor, not let go");
  });

  it("words every way the pick can stand against the baseline", () => {
    const base = parseStudioHero(rawHero(V2_HEROES[0]!), "studio/h1_tote.json");
    const vs = base.recommendation!.versus_baseline!;
    const words = (basis: typeof vs.basis, extra: Partial<typeof vs> = {}) =>
      baselineStory({ ...base, recommendation: { ...base.recommendation!, versus_baseline: { ...vs, basis, ...extra } } })!.lines.join(" ");
    expect(words("margin", { completion_delta: 0.125, needed_gain: 0.05, added_pieces: 1 })).toMatch(
      /finishes 12\.5 points more of the runs \(8 of 8 runs against 0 of 8\), more than the 5 points its 1 added piece must earn/,
    );
    expect(words("step_floor")).toMatch(/It fails a step outright \(open the lid: 0% of runs, under half\), so the two-tool head, soft arm plus suction cup is recommended: 8 of 8 runs against 0 of 8\./);
    expect(words("R09")).toMatch(/has a soft arm, and setups with a soft arm rank below every rigid setup \(integrator rule R09\)/);
    expect(words("R10")).toMatch(/dexterous hand .* \(integrator rule R10\)/);
    expect(words("tie", { note: "The same-complexity tie picks A over the baseline B: lighter." })).toContain("The same-complexity tie picks A");
    expect(words("ranking")).toMatch(/The ranking put the two-tool head/);
    expect(words("baseline", { same_setup: true })).toContain("It is the recommendation: no second tool or tool change earned its place");
  });

  it("a page exported before the baseline existed renders as before", () => {
    const raw = rawHero(V2_HEROES[0]!);
    delete raw.baseline;
    delete raw.recommendation.versus_baseline;
    raw.finalists = raw.finalists.map((f: { roles: string[] }) => ({ ...f, roles: f.roles.filter((r) => r !== "baseline") }));
    raw.best_single.roles = ["best_single"];
    const page = parseStudioHero(raw, "studio/h1_tote.json");
    expect(baselineStory(page)).toBeNull();
    const html = render(page);
    expect(html).not.toContain("The baseline: the best single tool");
    expect(html).toContain("data-head-to-head");
  });

  it("the parser holds the baseline to its finalist row", () => {
    const raw = rawHero(V2_HEROES[0]!);
    expect(() => parseStudioHero({ ...raw, baseline: { setup_id: "e".repeat(64), basis: null } })).toThrow(/not among the finalists/);
    const multi = raw.finalists.find((f: { kind: string }) => f.kind !== "single");
    expect(() => parseStudioHero({ ...raw, baseline: { setup_id: multi.setup_id, basis: null } })).toThrow(/baseline is a single tool|baseline role/);
    expect(() => parseStudioHero({ ...raw, baseline: { setup_id: null, basis: null } })).toThrow(/says why/);
    const same = structuredClone(raw);
    same.recommendation.versus_baseline.same_setup = true;
    expect(() => parseStudioHero(same)).toThrow(/same_setup disagrees/);
    const basis = structuredClone(raw);
    basis.recommendation.versus_baseline.basis = "vibes";
    expect(() => parseStudioHero(basis)).toThrow();
  });
});

describe("the six heroes on disk", () => {
  it("pass every build-time check: pages, lanes against their replays, GLBs, packs and sessions", () => {
    const dirs = fixtureDirs();
    expect(getStudioHeroIds(dirs)).toEqual(V2_HEROES.map((h) => h.hero));
    for (const h of V2_HEROES) {
      const page = getStudioHero(h.hero, dirs)!;
      expect(page.hero).toBe(h.hero);
      expect(focusOf(page)).not.toBeNull();
      if (page.session) expect(getStudioSession(page, dirs)).not.toBeNull();
    }
    expect(featuredHero(dirs)?.hero).toBe("h1_tote");
    // the published v1 routes redirect to the hero of the same case, or to the examples
    expect(legacyHeroRoutes(dirs)).toEqual([
      { hero: "h1", to: "../h1_tote/", title: "Returns tote" },
      { hero: "h2", to: "../h2_coupling/", title: "Pipe coupling" },
      { hero: "h3", to: "../#examples", title: null },
    ]);
  });

  it("leave nothing broken when the export cuts the hotel, then the tile", () => {
    for (const omit of [["h7_hotel"], ["h7_hotel", "h6_tile"]]) {
      const dirs = fixtureDirs(omit);
      const ids = getStudioHeroIds(dirs);
      expect(ids).toEqual(V2_HEROES.map((h) => h.hero).filter((h) => !omit.includes(h)));
      for (const h of omit) expect(getStudioHero(h, dirs)).toBeUndefined();
      const index = getStudioIndex(dirs)!;
      // story pages link only to the heroes that exist
      for (const id of ids) {
        const page = getStudioHero(id, dirs)!;
        const html = render(page, index.heroes.filter((e) => e.hero !== id));
        for (const h of omit) expect(html).not.toContain(`/studio/${h}`);
      }
      // the atlas reads a cut hero's job as researched, not simulated, and links nowhere
      const links = heroLinks(dirs);
      const atlas = getAtlas();
      for (const t of atlas.tasks.filter((x) => x.sim.hero && omit.includes(x.sim.hero))) {
        expect(simStatus(t, links).kind).toBe("researched");
      }
      for (const t of atlas.tasks.filter((x) => x.sim.hero && ids.includes(x.sim.hero))) {
        expect(simStatus(t, links).kind).toBe("simulated");
      }
    }
  });

  it("feature the first recommendation when the tote is cut", () => {
    const dirs = fixtureDirs(["h1_tote"]);
    expect(featuredHero(dirs)?.hero).toBe("h4_cnc");
  });

  it("redirect every v1 route the published data renamed or left out", () => {
    // the published v1 data itself had none
    expect(legacyHeroRoutes({ ...V1_FIXTURE, repo: REPO })).toEqual([]);
    const ids = getStudioHeroIds();
    if (ids.includes("h1")) return expect(legacyHeroRoutes()).toEqual([]);
    // the coupling's old route leads to its v2 page when the export publishes it, else to the examples
    const coupling = ids.includes("h2_coupling") ? { to: "../h2_coupling/", title: "Pipe coupling" } : { to: "../#examples", title: null };
    expect(legacyHeroRoutes()).toEqual([
      { hero: "h1", to: "../h1_tote/", title: "Returns tote" },
      { hero: "h2", ...coupling },
      { hero: "h3", to: "../#examples", title: null },
    ]);
  });
});
