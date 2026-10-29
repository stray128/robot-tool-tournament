import { readFileSync } from "node:fs";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AtlasExplorer, TaskCard } from "@/components/atlas/AtlasExplorer";
import { atlasScreen, filterTasks, kindCounts, NO_FILTER, parseAtlas, simStatus, sortTasks } from "@/lib/atlas";
import { DataError } from "@/lib/validate";

import { atlasTokens } from "../app/atlas/data";
import { brandTokens } from "../app/studio/data";

const WEB = path.join(__dirname, "..");
const REPO = path.join(WEB, "..");
const RAW = JSON.parse(readFileSync(path.join(WEB, "public", "data", "atlas.json"), "utf8"));
const TOKENS = atlasTokens(REPO);
const screen = atlasScreen(TOKENS);
const atlas = parseAtlas(RAW, "atlas.json", screen);
/** A token of the contract's list and an ordinary-word name of the research, read from the lists (never typed here). */
const CONTRACT = brandTokens(REPO);
const BRAND = CONTRACT.find((t) => /^[a-z]+$/.test(t))!;
const WORD = TOKENS.cs.find((t) => /^[A-Z][a-z]+$/.test(t))!;

/** Visible text of rendered markup: tags dropped, entities decoded. */
function visible(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/&#x27;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ");
}

describe("the use-case atlas", () => {
  it("screens with the contract, the library and the research's own names", () => {
    for (const t of CONTRACT) expect(TOKENS.ci).toContain(t); // the contract's and the library's lists
    expect(TOKENS.ci.length).toBeGreaterThan(CONTRACT.length + 400); // research names, any case
    expect(TOKENS.cs.length).toBeGreaterThan(50); // research names made of ordinary words, as written
    expect(screen(`a gripper from ${BRAND.toUpperCase()}`)).toEqual([BRAND.toUpperCase()]);
    expect(screen(`${WORD} builds robots`)).toEqual([WORD]);
    expect(screen(`the ${WORD.toLowerCase()} of it`)).toEqual([]);
  });

  it("atlas renders 95 tasks with no brand token", () => {
    expect(atlas.tasks).toHaveLength(95);
    const html = renderToStaticMarkup(createElement(AtlasExplorer, { tasks: atlas.tasks, domains: atlas.domains, heroes: {} }));
    expect(html.match(/data-atlas-task=/g)).toHaveLength(95);
    const text = visible(html);
    expect(screen(text)).toEqual([]);
    expect(text).not.toMatch(/https?:\/\/|www\./);
    expect(text).toContain("Showing 95 of 95 jobs");
  });

  it("covers the three research catalogues and every setup kind", () => {
    expect(atlas.domains.map((d) => d.id)).toEqual(["construction", "manufacturing", "lab_food_service"]);
    const counts = Object.fromEntries(atlas.domains.map((d) => [d.id, atlas.tasks.filter((t) => t.domain === d.id).length]));
    expect(counts).toEqual({ construction: 27, manufacturing: 31, lab_food_service: 37 });
    expect(kindCounts(atlas.tasks)).toEqual({ single: 69, tool_change: 16, two_tool_head: 10 });
    expect(atlas.rules).toHaveLength(23);
    for (const t of atlas.tasks) {
      expect(t.subtasks.length).toBeGreaterThan(0);
      expect(t.setup.tools.length).toBeGreaterThan(0);
      expect(t.why.length).toBeLessThanOrEqual(420);
      expect(t.sources.every((s) => /^https?:\/\//.test(s.url))).toBe(true);
    }
  });

  it("links source URLs only as targets, with generic text", () => {
    const task = atlas.tasks.find((t) => t.sources.length >= 3)!;
    const html = renderToStaticMarkup(createElement(TaskCard, { task, heroes: {}, domainTitle: "Construction" }));
    for (const [i, s] of task.sources.entries()) {
      expect(html).toContain(`href="${s.url.replace(/&/g, "&amp;")}"`);
      expect(html).toContain(`aria-label="Source ${i + 1} for`);
    }
    expect(visible(html)).not.toMatch(/https?:\/\//);
  });

  it("links the simulated heroes to their story pages", () => {
    const heroTasks = atlas.tasks.filter((t) => t.sim.hero);
    expect(heroTasks.map((t) => t.sim.hero).sort()).toEqual(["h1_tote", "h2_coupling", "h4_cnc", "h5_solar", "h6_tile", "h7_hotel"]);
    // Today's index lists the tote and the coupling (as h1 and h2); the others link once their pages exist.
    const heroes = { h1_tote: { hero: "h1", title: "Returns tote" }, h2_coupling: { hero: "h2", title: "Pipe coupling" } };
    const linked = atlas.tasks.filter((t) => simStatus(t, heroes).kind === "simulated");
    expect(linked.map((t) => t.sim.hero).sort()).toEqual(["h1_tote", "h2_coupling"]);
    const html = renderToStaticMarkup(createElement(AtlasExplorer, { tasks: atlas.tasks, domains: atlas.domains, heroes }));
    expect(html).toMatch(/href="\/studio\/h1\/?"/);
    expect(html).toMatch(/href="\/studio\/h2\/?"/);
    expect(html.match(/data-hero-link=/g)).toHaveLength(2);
    // with all six pages exported, all six link
    const all = Object.fromEntries(heroTasks.map((t) => [t.sim.hero!, { hero: t.sim.hero!, title: t.title }]));
    expect(filterTasks(atlas.tasks, { ...NO_FILTER, status: "simulated" }, all)).toHaveLength(6);
    expect(sortTasks(atlas.tasks, "simulated", all).slice(0, 6).every((t) => t.sim.hero)).toBe(true);
  });

  it("filters by text, domain, setup and simulation status", () => {
    const vacuum = filterTasks(atlas.tasks, { ...NO_FILTER, q: "vacuum" }, {});
    expect(vacuum.length).toBeGreaterThan(5);
    expect(filterTasks(atlas.tasks, { ...NO_FILTER, domain: "construction", kind: "two_tool_head" }, {}).every((t) => t.domain === "construction" && t.setup.kind === "two_tool_head")).toBe(true);
    expect(filterTasks(atlas.tasks, { ...NO_FILTER, status: "out_of_scope" }, {})).toHaveLength(9);
    expect(filterTasks(atlas.tasks, { ...NO_FILTER, q: "no such words anywhere" }, {})).toHaveLength(0);
  });

  it("refuses a brand name, or a URL shown as text", () => {
    const branded = structuredClone(RAW);
    branded.tasks[0].why = `Works like the ${BRAND} dual gripper.`;
    expect(() => parseAtlas(branded, "atlas.json", screen)).toThrow(DataError);
    const linked = structuredClone(RAW);
    linked.tasks[0].subtasks[0].note = "see https://example.org";
    expect(() => parseAtlas(linked, "atlas.json", screen)).toThrow(DataError);
    const research = structuredClone(RAW);
    research.tasks[1].title = `Picking with ${TOKENS.ci.find((t) => t.length > 8 && /^[a-z ]+$/.test(t))}`;
    expect(() => parseAtlas(research, "atlas.json", screen)).toThrow(DataError);
  });
});
