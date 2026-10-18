/**
 * The Studio pages' data (product mode): the strict parsers of components/studio/schema.ts, the
 * build-time file checks of app/studio/data.ts (replay headers and payloads, GLB assets, the
 * build pack, the session), the brand and URL rules, and the pure helpers the pages and lanes use.
 * The contract fixture lives in tests/fixtures/studio (make_fixture.mjs); the published data under
 * public/data/studio is checked too when it exists.
 */

import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  brandTokens,
  checkBuildPack,
  checkLaneFiles,
  DEFAULT_DIRS,
  getStudioHero,
  getStudioHeroIds,
  getStudioIndex,
  getStudioSession,
  type StudioDirs,
} from "@/app/studio/data";
import { briefFields } from "@/components/studio/SessionView";
import {
  blockerRange,
  brandPattern,
  changeText,
  constantValue,
  type HeroPage,
  laneAt,
  type MatrixRow,
  orderRows,
  outcomeText,
  paramText,
  parseStudioHero,
  parseStudioIndex,
  PRODUCT_MODE_LABEL,
  prunedNotes,
  quantityName,
  runEnd,
  runsText,
  stepTitle,
  targetText,
  VERIFY_CAVEAT,
} from "@/components/studio/schema";
import { DataError } from "@/lib/validate";

const WEB = path.join(__dirname, "..");
const REPO = path.join(WEB, "..");
const FIXTURE = path.join(__dirname, "fixtures", "studio");
const FIX: StudioDirs = { data: path.join(FIXTURE, "data"), site: path.join(FIXTURE, "site"), repo: REPO };
const published = existsSync(path.join(DEFAULT_DIRS.data, "studio", "index.json"));
const BRAND = brandPattern(brandTokens(REPO));
/** Two tokens of the repository's lists (read from the Python sources, never written here). */
const [TOKEN_A = "", TOKEN_B = ""] = brandTokens(REPO);
const tmp: string[] = [];

afterEach(() => {
  for (const d of tmp.splice(0)) rmSync(d, { recursive: true, force: true });
});

function raw(name: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path.join(FIX.data, "studio", name), "utf8"));
}

/** A writable copy of the fixture (data and site), for checks that read files. */
function copyFixture(): StudioDirs {
  const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-studio-"));
  tmp.push(dir);
  cpSync(FIXTURE, dir, { recursive: true });
  return { data: path.join(dir, "data"), site: path.join(dir, "site"), repo: REPO };
}

function heroPage(): HeroPage {
  return parseStudioHero(raw("f1.json"), "studio/f1.json", { brand: BRAND });
}

/** A modified copy of the f1 page through `edit`, then parsed. */
function parseEdited(edit: (doc: Record<string, any>) => void, name = "f1.json") {
  const doc = structuredClone(raw(name)) as Record<string, any>;
  edit(doc);
  return () => parseStudioHero(doc, `studio/${name}`, { brand: BRAND });
}

describe("the contract fixture", () => {
  it("parses, with every file it points at", () => {
    const index = getStudioIndex(FIX)!;
    expect(index.label).toBe(PRODUCT_MODE_LABEL);
    expect(getStudioHeroIds(FIX)).toEqual(["f1", "f2"]);
    const f1 = getStudioHero("f1", FIX)!;
    expect(f1.outcome).toBe("recommended");
    expect(f1.recommendation?.caveat).toBe(VERIFY_CAVEAT);
    expect(f1.lanes.map((l) => l.id)).toEqual(["recommended", "best_single"]);
    expect(f1.lanes[0]!.segments[1]!.change_start_s).toBe(2.5);
    expect(f1.build_pack?.for).toBe("recommended");
    // v1 fidelity "core" reads as the Studio's rigid tier.
    expect(f1.best_single?.tool_details[0]?.fidelity).toBe("rigid");
    // index tools are objects for a recommendation and names elsewhere; both read as tools.
    expect(index.heroes[0]!.recommended!.tools[1]!.caveat).toMatch(/verify on hardware/);
    expect(index.heroes[0]!.best_single!.tools[0]!.display_name).toBe("Two-finger parallel gripper, flat pads");
    const session = getStudioSession(f1, FIX)!;
    expect(session.events.length).toBeGreaterThan(3);
    const f2 = getStudioHero("f2", FIX)!;
    expect(f2.recommendation).toBeNull();
    expect(f2.no_winner?.blocker?.families).toHaveLength(2);
    expect(f2.lanes).toEqual([]);
    expect(f2.build_pack_note).toMatch(/nothing to pack/);
  });

  it("returns null without an index and undefined for an unknown hero", () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-studio-empty-"));
    tmp.push(dir);
    const dirs = { data: dir, site: dir, repo: null };
    expect(getStudioIndex(dirs)).toBeNull();
    expect(getStudioHeroIds(dirs)).toEqual([]);
    expect(getStudioHero("f1", FIX)?.hero).toBe("f1");
    expect(getStudioHero("nope", FIX)).toBeUndefined();
  });
});

describe("the parsers reject a broken export", () => {
  it("a missing or changed label", () => {
    expect(parseEdited((d) => delete d.label)).toThrow(DataError);
    expect(parseEdited((d) => (d.label = "Product mode"))).toThrow(/label/);
    const index = structuredClone(raw("index.json")) as Record<string, unknown>;
    delete index.label;
    expect(() => parseStudioIndex(index)).toThrow(/\$\.label/);
  });

  it("an unknown outcome, or one its blocks do not match", () => {
    expect(parseEdited((d) => (d.outcome = "maybe"))).toThrow(/outcome: expected one of recommended, no_recommendation/);
    expect(parseEdited((d) => (d.recommendation = null))).toThrow(/recommended outcome/);
    expect(parseEdited((d) => (d.outcome = "no_recommendation"), "f2.json")).not.toThrow();
    expect(parseEdited((d) => (d.no_winner = null), "f2.json")).toThrow(/no-winner/);
  });

  it("an approximation result without its caveat or rigid runner-up", () => {
    expect(parseEdited((d) => (d.recommendation.caveat = null))).toThrow(/verify on hardware/);
    expect(parseEdited((d) => (d.recommendation.verify_on_hardware = false))).toThrow(/verify on hardware/);
    expect(parseEdited((d) => (d.recommendation.runner_up_rigid = null))).toThrow(/rigid runner-up/);
    expect(parseEdited((d) => (d.plan.steps[1].caveats = []))).toThrow(/caveat/);
    expect(parseEdited((d) => (d.recommendation.steps[1].caveats = []))).toThrow(/caveat/);
  });

  it("a matrix, step or lane that does not fit the plan", () => {
    expect(parseEdited((d) => d.matrix.steps.reverse())).toThrow(/plan's steps/);
    expect(parseEdited((d) => delete d.matrix.rows[0].cells.twist_cap)).toThrow(/no cell for step twist_cap/);
    expect(parseEdited((d) => (d.plan.steps[0].object = "lid"))).toThrow(/does not list/);
    expect(parseEdited((d) => (d.lanes[0].steps[0].step_id = "open_lid"))).toThrow(/no step open_lid/);
    expect(parseEdited((d) => (d.lanes[0].setup_id = d.best_single.setup_id))).toThrow(/not the page's recommended setup/);
    expect(parseEdited((d) => (d.lanes[0].replay = "studio/replays/f2/recommended"))).toThrow(/under studio\/replays\/f1/);
    expect(parseEdited((d) => (d.lanes[0].segments[1].change = "none"))).toThrow(/tool change or a head switch/);
    expect(parseEdited((d) => (d.lanes[0].duration_s = 30))).toThrow(/frames at/);
  });

  it("counts, rates and packs that do not add up", () => {
    expect(parseEdited((d) => (d.screen.n_screened = 3))).toThrow(/n_screened/);
    expect(parseEdited((d) => (d.matrix.rows[1].cells.twist_cap.successes = 2))).toThrow(/rate/);
    expect(parseEdited((d) => (d.finalists[0].completed_runs = 7))).toThrow(/completion rate/);
    expect(parseEdited((d) => (d.build_pack.bytes = 2_000_001))).toThrow(/at most|1 to 2000000/);
    expect(parseEdited((d) => (d.build_pack.zip = "studio/f2/build_pack.zip"))).toThrow(/studio\/f1\/build_pack.zip/);
  });

  it("a URL outside a url field; a URL inside one is fine", () => {
    expect(parseEdited((d) => (d.use_case = "See https://example.org for the tote"))).toThrow(/URL may appear only/);
    expect(parseEdited((d) => (d.build_pack.links[0].reason = "upstream at www.example.org"))).toThrow(/URL/);
    expect(parseEdited((d) => (d.build_pack.links[0].url = "https://example.org/another"))).not.toThrow();
    expect(parseEdited((d) => (d.build_pack.links[0].url = "http://example.org"))).toThrow(/https URL/);
  });

  it("a brand token in a display name", () => {
    expect(BRAND).not.toBeNull();
    expect(parseEdited((d) => (d.screen.candidates[0].display_name = `${TOKEN_A} gripper`))).toThrow(/names a brand/);
    expect(parseEdited((d) => (d.matrix.rows[0].display_name = `${TOKEN_B} hand`))).toThrow(/names a brand/);
    // Upstream URLs may name a maker; they are link targets, never text.
    expect(parseEdited((d) => (d.build_pack.links[0].url = `https://github.com/example/${TOKEN_A.replace(/[^A-Za-z0-9]+/g, "-")}-model`))).not.toThrow();
  });
});

describe("the build-time file checks", () => {
  it("a lane whose bytes differ from its files", () => {
    const page = heroPage();
    page.lanes[0]!.bytes += 1;
    expect(() => checkLaneFiles(page, "studio/f1.json", FIX)).toThrow(/serves \d+ bytes; the page says/);
  });

  it("a lane whose asset bytes differ, or a GLB that does not match its sha256", () => {
    const page = heroPage();
    page.lanes[1]!.asset_bytes -= 1;
    expect(() => checkLaneFiles(page, "studio/f1.json", FIX)).toThrow(/GLB assets/);
    const dirs = copyFixture();
    const header = JSON.parse(readFileSync(path.join(dirs.data, "studio", "replays", "f1", "recommended.json"), "utf8"));
    const glb = path.join(dirs.site, Object.values<{ uri: string }>(header.assets)[0]!.uri);
    const bytes = readFileSync(glb);
    bytes[bytes.length - 1] = bytes[bytes.length - 1]! ^ 0xff;
    writeFileSync(glb, bytes);
    expect(() => checkLaneFiles(heroPage(), "studio/f1.json", dirs)).toThrow(/does not match its sha256/);
  });

  it("a replay header of another setup, hero or frame count", () => {
    const dirs = copyFixture();
    const file = path.join(dirs.data, "studio", "replays", "f1", "best_single.json");
    const header = JSON.parse(readFileSync(file, "utf8"));
    writeFileSync(file, JSON.stringify({ ...header, meta: { ...header.meta, design_id: "0".repeat(64) } }));
    expect(() => checkLaneFiles(heroPage(), "studio/f1.json", dirs)).toThrow(/shows setup 000000000000/);
    writeFileSync(file, JSON.stringify({ ...header, meta: { ...header.meta, part_id: "h9" } }));
    expect(() => checkLaneFiles(heroPage(), "studio/f1.json", dirs)).toThrow(/belongs to h9/);
    writeFileSync(file, JSON.stringify({ ...header, provenance: { ...header.provenance, product_mode: false } }));
    expect(() => checkLaneFiles(heroPage(), "studio/f1.json", dirs)).toThrow(/product mode/);
    rmSync(file);
    expect(() => checkLaneFiles(heroPage(), "studio/f1.json", dirs)).toThrow(/missing replay header/);
  });

  it("a build pack whose bytes or hash differ", () => {
    const page = heroPage();
    page.build_pack!.sha256 = "0".repeat(64);
    expect(() => checkBuildPack(page, "studio/f1.json", FIX)).toThrow(/sha256/);
    const other = heroPage();
    other.build_pack!.bytes += 1;
    expect(() => checkBuildPack(other, "studio/f1.json", FIX)).toThrow(/bytes; the page says/);
  });

  it("a page the index does not list, or one that differs from its entry", () => {
    const dirs = copyFixture();
    writeFileSync(path.join(dirs.data, "studio", "f3.json"), "{}");
    expect(() => getStudioHeroIds(dirs)).toThrow(/not listed in studio\/index.json/);
    const dirs2 = copyFixture();
    const f1 = JSON.parse(readFileSync(path.join(dirs2.data, "studio", "f1.json"), "utf8"));
    writeFileSync(path.join(dirs2.data, "studio", "f1.json"), JSON.stringify({ ...f1, headline: "Another sentence." }));
    expect(() => getStudioHero("f1", dirs2)).toThrow(/headline: differs from the index/);
  });
});

describe.skipIf(!published)("published Studio data", () => {
  it("parses and passes every file check", () => {
    const index = getStudioIndex()!;
    expect(index.label).toBe(PRODUCT_MODE_LABEL);
    const ids = getStudioHeroIds();
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) {
      const page = getStudioHero(id)!;
      expect(page.label).toBe(PRODUCT_MODE_LABEL);
      expect(page.matrix.steps).toEqual(page.plan.steps.map((s) => s.id));
      if (page.recommendation?.verify_on_hardware) expect(page.recommendation.caveat).toBe(VERIFY_CAVEAT);
      if (page.outcome === "no_recommendation") expect(page.no_winner?.closest.length).toBeGreaterThan(0);
      for (const lane of page.lanes) {
        expect(runEnd(lane)).toBeGreaterThan(0);
        expect(laneAt(lane, runEnd(lane) + 1).kind).toBe("ended");
      }
      if (page.session) expect(getStudioSession(page)).not.toBeNull();
    }
  });
});

// ---------------------------------------------------------------- helpers

describe("lanes on the clock", () => {
  const lane = heroPage().lanes[0]!;
  const plan = heroPage().plan;

  it("names the step, the change gap and the end", () => {
    const a = laneAt(lane, 1);
    expect(a.kind === "step" && a.step.step_id).toBe("pick_part");
    expect(a.kind === "step" && a.segment?.tool).toBe(0);
    const gap = laneAt(lane, 3);
    expect(gap.kind).toBe("change");
    expect(gap.kind === "change" && changeText(gap.change, gap.change_s)).toBe("Tool change: a fixed 1 s, not simulated");
    const b = laneAt(lane, 4);
    expect(b.kind === "step" && b.step.step_id).toBe("twist_cap");
    expect(b.kind === "step" && b.segment?.tool_label).toMatch(/Suction cup/);
    expect(laneAt(lane, 6.01).kind).toBe("ended");
    expect(changeText("present", 1.5)).toBe("Head switch: a fixed 1.5 s, not simulated");
  });

  it("says how a run ends", () => {
    expect(outcomeText(lane, plan)).toBe("Completed in 6.0 s");
    expect(outcomeText(heroPage().lanes[1]!, plan)).toBe("Stopped at 2. Twist cap: slip");
    expect(runEnd(heroPage().lanes[1]!)).toBeCloseTo(5.9);
  });
});

describe("the matrix and the words", () => {
  const cell = (rate: number, n = 3) => ({ state: "run" as const, rate, n, successes: Math.round(rate * n), primitive_used: null, failure_modes: {}, context: null });
  const row = (name: string, cells: MatrixRow["cells"]) => ({ display_name: name, cells });

  it("orders rows by passing steps, then summed rate, keeping ties in export order", () => {
    const rows = [
      row("a", { s1: cell(0), s2: cell(0) }),
      row("b", { s1: cell(1), s2: { state: "pruned", reason: "x" } }),
      row("c", { s1: cell(1), s2: cell(0.333) }),
      row("d", { s1: cell(1), s2: { state: "pruned", reason: "y" } }),
    ];
    expect(orderRows(rows).map((r) => r.display_name)).toEqual(["c", "b", "d", "a"]);
  });

  it("numbers pruned reasons in order of first appearance", () => {
    const notes = prunedNotes(heroPage().matrix);
    expect([...notes.entries()]).toEqual([["the jar is narrower than the cup", 1]]);
  });

  it("writes runs, steps, parameters, targets and constants", () => {
    expect(runsText(8, 8)).toBe("8 of 8 runs");
    expect(runsText(1, 1)).toBe("1 of 1 run");
    expect(stepTitle(heroPage().plan, "twist_cap")).toBe("2. Twist cap");
    expect(paramText("speed_mm_s", 80)).toEqual(["speed", "80 mm/s"]);
    expect(paramText("width_mm", 340)).toEqual(["width", "340 mm"]);
    expect(paramText("open_torque_nm", 0.3)).toEqual(["open torque", "0.3 N m"]);
    expect(paramText("approach", "top")).toEqual(["approach", "top"]);
    expect(quantityName("torque_nm")).toBe("torque");
    expect(targetText({ at_least: 3, at_most: null, unit: "N m" })).toBe("at least 3 N m");
    const blocker = getStudioHero("f2", FIX)!.no_winner!.blocker!;
    expect(blockerRange(blocker)).toBe("0.09 to 1.15 N m against at least 3 N m, at most 8 N m");
    expect(blockerRange({ ...blocker, at_most: null })).toBe("0.09 to 1.15 N m against at least 3 N m");
    expect(constantValue({ key: "c", value: 0.1, unit: "fraction", note: "", tag: "assumed" })).toBe("10%");
    expect(constantValue({ key: "t", value: 6, unit: "s", note: "", tag: "assumed" })).toBe("6 s");
  });

  it("summarises a session's tool calls without ids, paths or URLs", () => {
    expect(briefFields({ steps: 2, task_id: "abc", url: "https://x", note: "see https://example.org", items: [1, 2], ok: true })).toEqual([
      "steps: 2",
      "items: 2 items",
      "ok: yes",
    ]);
  });

  it("builds the brand pattern from the repository's lists", () => {
    expect(TOKEN_A).not.toBe("");
    expect(BRAND?.test(`a ${TOKEN_A} gripper`)).toBe(true);
    expect(BRAND?.test("Two-finger adaptive linkage gripper, 85 mm")).toBe(false);
    expect(brandPattern([])).toBeNull();
  });
});

describe("the server-rendered sections", () => {
  it("render the fixture with its words, and no URL or brand as text", async () => {
    const { createElement: h } = await import("react");
    const { renderToStaticMarkup } = await import("react-dom/server");
    const { SessionView } = await import("@/components/studio/SessionView");
    const { RecommendationCard, NoWinnerBlock } = await import("@/components/studio/Outcome");
    const { PlanView } = await import("@/components/studio/PlanView");
    const { BuildPackView } = await import("@/components/studio/BuildPackView");
    const { ProvenanceView } = await import("@/components/studio/ProvenanceView");
    const { FinalistsTable } = await import("@/components/studio/FinalistsTable");
    const { ScreenView } = await import("@/components/studio/ScreenView");
    const f1 = getStudioHero("f1", FIX)!;
    const f2 = getStudioHero("f2", FIX)!;
    const text = (html: string) => html.replace(/<[^>]+>/g, " ").replace(/&[a-z#0-9]+;/g, " ");
    const html = [
      renderToStaticMarkup(h(SessionView, { block: f1.session!, session: getStudioSession(f1, FIX)! })),
      renderToStaticMarkup(h(RecommendationCard, { rec: f1.recommendation!, plan: f1.plan, bestSingle: f1.best_single })),
      renderToStaticMarkup(h(NoWinnerBlock, { noWinner: f2.no_winner!, plan: f2.plan })),
      renderToStaticMarkup(h(PlanView, { plan: f1.plan })),
      renderToStaticMarkup(h(BuildPackView, { pack: f1.build_pack, note: null })),
      renderToStaticMarkup(h(BuildPackView, { pack: null, note: f2.build_pack_note })),
      renderToStaticMarkup(h(ProvenanceView, { page: f1 })),
      renderToStaticMarkup(h(FinalistsTable, { rows: f1.finalists, plan: f1.plan, baselineId: f1.best_single!.setup_id })),
      renderToStaticMarkup(h(ScreenView, { screen: f1.screen, plan: f1.plan })),
    ];
    const all = html.map(text).join("\n");
    expect(all).toContain("Plan the task");
    expect(all).toContain("Run the task tournament");
    expect(all).toContain("Use the tool change: 6 of 8 runs.");
    expect(all).toContain("No setup completes this task");
    expect(all).toContain("0.09 to 1.15 N m against at least 3 N m, at most 8 N m");
    expect(all).toContain("verify on hardware");
    expect(all).toContain("Edit this plan on /studio/new/");
    expect(all).toContain("nothing to pack");
    expect(all).toMatch(/Download the build pack \(zip, \d/);
    expect(all).not.toMatch(/http|www\./i);
    expect(all.match(BRAND!)).toBeNull();
    // URLs only as link targets.
    expect(html[4]).toContain('href="https://example.org/upstream/gripper"');
    expect(html[4]).toContain('href="/data/studio/f1/build_pack.zip"');
    // A caveat shared by later steps is written out once (the published two-tool head repeats one).
    const h1 = published ? getStudioHero("h1") : undefined;
    if (h1?.recommendation) {
      const card = text(renderToStaticMarkup(h(RecommendationCard, { rec: h1.recommendation, plan: h1.plan, bestSingle: h1.best_single })));
      const caveats = h1.recommendation.steps.flatMap((s) => s.caveats);
      if (new Set(caveats).size < caveats.length) expect(card).toContain("The same caveat as");
    }
  });
});
