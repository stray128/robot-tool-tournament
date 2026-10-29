#!/usr/bin/env node
/**
 * Headless check of the product site's main routes on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_site.mjs [--out out] [--base /robot-tool-tournament] [--port 4196] [--shots DIR]
 *
 * Serves web/out (under --base, as GitHub Pages does) and opens Home, the Studio, every story page
 * the served data/studio/index.json lists, the use-case atlas, Docs with the quickstart and the
 * three tutorials, Evidence, Compute and the Library in Chromium at 1440 x 900 and 375 x 812.
 * Fails on: a non-200 page, a console or page error, horizontal page scroll, a brand token in the
 * visible text (the contract's and the library's lists everywhere but the credits; on the atlas also
 * the research names of tests/data/atlas_brand_tokens.json), an internal link that does not answer
 * 200, and per page:
 * story pages show the answer, "Simulated, verify on hardware" and the product-mode label and
 * draw a WebGL canvas per lane; the atlas renders every task, filters, and shows no URL as text;
 * a docs code block's copy button answers. Saves a screenshot per route and width to --shots.
 *
 * Controls (no placeholder controls: every control works or is not there): no "Prototype" label
 * and no link to "#"; a link that says Run or "Open the Studio" leads to the recorded runs, a story
 * page or the run-it-yourself tutorial (the site has no hosted runner); the home question box hands
 * its words to the planner; on a story page "Watch this step" lights that step's card, the replay
 * shows the "Changing tool" card inside a tool change, and the baseline and the step caveats show.
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.join(webDir, "..");
const { values } = parseArgs({
  options: {
    out: { type: "string", default: path.join(webDir, "out") },
    base: { type: "string", default: process.env.NEXT_PUBLIC_BASE_PATH ?? "" },
    port: { type: "string", default: "4196" },
    shots: { type: "string", default: path.join(webDir, ".smoke", "site") },
  },
});
const LABEL = "Product mode: not part of the pre-registered experiment";
const outDir = path.resolve(values.out);
const base = values.base.replace(/\/$/, "");
const index = JSON.parse(readFileSync(path.join(outDir, "data", "studio", "index.json"), "utf8"));
const atlas = JSON.parse(readFileSync(path.join(outDir, "data", "atlas.json"), "utf8"));

function pythonTokens(rel, name) {
  const src = readFileSync(path.join(repoDir, rel), "utf8");
  const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
  if (!block) throw new Error(`${name} not found in ${rel}`);
  return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}
const escape = (t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const alt = (list) => [...new Set(list)].sort((a, b) => b.length - a.length).map(escape).join("|");
const research = JSON.parse(readFileSync(path.join(repoDir, "tests", "data", "atlas_brand_tokens.json"), "utf8"));
const siteTokens = [...pythonTokens("src/rtt/effectors/types.py", "BRAND_TOKENS"), ...pythonTokens("src/rtt/library/brands.py", "LIBRARY_ONLY_TOKENS")];
const siteRe = new RegExp(`(?<![A-Za-z0-9])(${alt(siteTokens)})(?![A-Za-z0-9])`, "i");
// The research's names screen the atlas, whose text comes from that research ("MIT" is a licence
// elsewhere on the site, and "d3" a cloud platform).
const researchRe = new RegExp(`(?<![A-Za-z0-9])(${alt(research.ci)})(?![A-Za-z0-9])`, "i");
const researchCsRe = research.cs.length ? new RegExp(`(?<![A-Za-z0-9])(${alt(research.cs)})(?![A-Za-z0-9])`) : null;
const brandIn = (text, route) =>
  text.match(siteRe)?.[1] ??
  (route === "/atlas/" ? (text.match(researchRe)?.[1] ?? (researchCsRe ? text.match(researchCsRe)?.[1] : null)) : null) ??
  null;

const heroes = index.heroes.map((h) => h.hero);
const routes = [
  "/",
  "/studio/",
  ...heroes.map((h) => `/studio/${h}/`),
  "/atlas/",
  "/docs/",
  "/docs/quickstart/",
  "/docs/tutorials/plan-a-task/",
  "/docs/tutorials/read-a-recommendation/",
  "/docs/tutorials/run-it-yourself/",
  "/evidence/",
  "/compute/",
  "/library/",
];
const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

const shotName = (route, vp) => `${route === "/" ? "home" : route.replace(/^\/|\/$/g, "").replace(/\//g, "-")}-${vp}.png`;

/** Where a link that says Run or "Open the Studio" may lead: recorded runs, a story page, the tutorial. */
function runTargetOk(href) {
  const rel = href.startsWith(base) ? href.slice(base.length) : href;
  return (
    rel === "/studio/#examples" ||
    rel === "/docs/tutorials/run-it-yourself/" ||
    heroes.some((h) => rel === `/studio/${h}/` || rel.startsWith(`/studio/${h}/#`))
  );
}

async function controlChecks(page) {
  const problems = [];
  if ((await page.$("[data-prototype]")) || /\bPrototype\b/.test(await page.evaluate(() => document.body.innerText))) problems.push('a "Prototype" label');
  const links = await page.$$eval("a", (as) => as.map((a) => ({ href: a.getAttribute("href"), text: (a.textContent ?? "").trim().replace(/\s+/g, " ") })));
  for (const l of links) {
    if (l.href === null || l.href === "#" || l.href === "") problems.push(`a link to nowhere: "${l.text}"`);
    // an in-page anchor ("On this page: Run it on your machine") is navigation, not an action
    else if (/^(run\b|open the studio)/i.test(l.text) && !l.href.startsWith("#") && !runTargetOk(l.href)) problems.push(`"${l.text}" leads to ${l.href}, not to the recorded runs or the run-it-yourself tutorial`);
  }
  return problems;
}

/** Set the story page's shared clock through its scrubber. */
async function scrubTo(page, t) {
  await page.locator("[data-studio-transport] input[type=range]").first().evaluate((el, at) => {
    const input = /** @type {HTMLInputElement} */ (el);
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, String(at));
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }, t);
  await page.waitForTimeout(300);
}

async function storyControls(page, data) {
  const problems = [];
  // the step timeline and the step cards' seek buttons drive the shared clock
  const seeks = page.locator("[data-step-card] [data-seek]");
  if ((await seeks.count()) > 1) {
    const second = seeks.nth(1);
    const card = await second.evaluate((b) => b.closest("[data-step-card]")?.getAttribute("data-step-card"));
    await second.scrollIntoViewIfNeeded();
    await second.click();
    await page.waitForTimeout(700);
    if (!(await page.$(`[data-step-card="${card}"][data-active]`))) problems.push(`"Watch this step" does not light step ${card}`);
  }
  // inside a tool change the replay shows the "Changing tool" card
  const lane = data.lanes.find((l) => l.segments.some((s) => s.change !== "none"));
  if (lane) {
    const seg = lane.segments.find((s) => s.change !== "none");
    await scrubTo(page, 0.5 * (seg.change_start_s + seg.start_s));
    const card = page.locator("[data-change-card]").first();
    if ((await card.count()) === 0 || !(await card.innerText()).includes("Changing tool")) problems.push("no \"Changing tool\" card inside a tool change");
  }
  if (data.baseline && !(await page.$("[data-baseline]"))) problems.push("the page's baseline is not shown");
  const approx = (data.recommendation?.steps ?? []).filter((s) => s.fidelity !== "rigid").length;
  if (approx && (await page.$$("[data-step-notes]")).length < approx) problems.push("a step off the rigid tier shows no caveat");
  return problems;
}

async function homeControls(page) {
  const problems = [];
  // the question box hands the words to the planner (this tab's storage, never the URL)
  const words = "Open the lid of a tote on the bench and put a carton in it.";
  await page.locator("[data-hero-ask] textarea").fill(words);
  await page.locator("[data-hero-ask] button[type=submit]").click();
  await page.waitForURL(/\/studio\/#plan$/, { timeout: 15000 }).catch(() => problems.push(`the question box went to ${page.url()}`));
  if (page.url().includes("q=")) problems.push("the question box put the words in the URL");
  await page.waitForTimeout(800);
  const got = await page.locator("#plan textarea").first().inputValue().catch(() => "");
  if (!got.includes(words)) problems.push("the planner did not receive the question box's words");
  return problems;
}

async function pageChecks(page, route) {
  const problems = [];
  const text = await page.evaluate(() => document.body.innerText);
  problems.push(...(await controlChecks(page)));
  if (route.startsWith("/studio/") && route !== "/studio/") {
    for (const must of ["Simulated, verify on hardware", LABEL]) if (!text.includes(must)) problems.push(`missing "${must}"`);
    if (!(await page.$("[data-answer]"))) problems.push("no answer sentence");
    const hero = index.heroes.find((h) => route === `/studio/${h.hero}/`);
    const data = JSON.parse(readFileSync(path.join(outDir, "data", hero.page), "utf8"));
    if (data.lanes.length) {
      await page.locator("[data-lane] canvas").first().waitFor({ timeout: 30000 }).catch(() => problems.push("no replay canvas"));
      const n = await page.$$eval("[data-lane] canvas", (c) => c.length);
      if (n < data.lanes.length) problems.push(`${n} replay canvases for ${data.lanes.length} lanes`);
    }
    if (/https?:\/\//.test(text)) problems.push("a URL in the visible text");
  }
  if (route === "/atlas/") {
    const n = await page.$$eval("[data-atlas-task]", (e) => e.length);
    if (n !== atlas.tasks.length) problems.push(`${n} atlas tasks rendered, the data has ${atlas.tasks.length}`);
    if (/https?:\/\//.test(text)) problems.push("a URL in the visible text");
    await page.getByRole("button", { name: /Two-tool head/ }).first().click();
    const shown = Number(await page.getAttribute("[data-atlas-count]", "data-atlas-count"));
    const expected = atlas.tasks.filter((t) => t.setup.kind === "two_tool_head").length;
    if (shown !== expected) problems.push(`the two-tool filter shows ${shown}, expected ${expected}`);
    await page.getByRole("button", { name: /Any/ }).first().click();
  }
  if (route.startsWith("/docs/") && route !== "/docs/") {
    const copy = page.getByRole("button", { name: /^Copy/ }).first();
    if ((await copy.count()) > 0) {
      await copy.scrollIntoViewIfNeeded();
      await copy.click();
      await page.waitForTimeout(400);
      const label = (await copy.textContent())?.trim();
      if (!["Copied", "Select and copy"].includes(label)) problems.push(`the copy button says "${label}" after a click`);
    }
  }
  if (!route.startsWith("/credits")) {
    const hit = brandIn(text, route);
    if (hit) problems.push(`names a brand in the visible text: ${hit}`);
  }
  return problems;
}

/** The checks that drive the page (after its screenshot): the story's clock, the home hand-over. */
async function interactiveChecks(page, route) {
  if (route === "/") return homeControls(page);
  const hero = index.heroes.find((h) => route === `/studio/${h.hero}/`);
  if (!hero) return [];
  return storyControls(page, JSON.parse(readFileSync(path.join(outDir, "data", hero.page), "utf8")));
}

async function main() {
  if (!existsSync(outDir)) throw new Error(`no export at ${outDir}; run pnpm build first`);
  mkdirSync(values.shots, { recursive: true });
  const port = Number(values.port);
  const server = await startServer({ dir: outDir, port, base });
  const origin = `http://127.0.0.1:${port}${base}`;
  const browser = await chromium.launch();
  const failures = [];
  const links = new Set();
  try {
    for (const vp of viewports) {
      for (const route of routes) {
        const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, colorScheme: "dark" });
        const page = await ctx.newPage();
        const errors = [];
        page.on("console", (m) => m.type() === "error" && errors.push(m.text().slice(0, 200)));
        page.on("pageerror", (e) => errors.push(`pageerror: ${String(e).slice(0, 200)}`));
        const res = await page.goto(origin + route, { waitUntil: "networkidle", timeout: 90000 });
        await page.waitForTimeout(800);
        const problems = [];
        if (res?.status() !== 200) problems.push(`status ${res?.status()}`);
        const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
        if (sw > iw) problems.push(`horizontal scroll (${sw} > ${iw})`);
        problems.push(...(await pageChecks(page, route)));
        for (const href of await page.$$eval("a[href]", (as) => as.map((a) => a.getAttribute("href")))) {
          if (href && href.startsWith(`${base}/`) && !href.startsWith("//")) links.add(href.split("#")[0]);
        }
        await page.waitForTimeout(300);
        await page.screenshot({ path: path.join(values.shots, shotName(route, vp.name)), fullPage: true });
        problems.push(...(await interactiveChecks(page, route)));
        problems.push(...errors.map((e) => `console: ${e}`));
        console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${route}${problems.length ? `\n       ${problems.join("\n       ")}` : ""}`);
        if (problems.length) failures.push(`${vp.name} ${route}: ${problems.join("; ")}`);
        await ctx.close();
      }
    }
    for (const href of [...links].sort()) {
      const r = await fetch(`http://127.0.0.1:${port}${href}`, { method: "HEAD" });
      if (r.status !== 200) failures.push(`link ${href}: ${r.status}`);
    }
    console.log(`${links.size} internal links checked`);
  } finally {
    await browser.close();
    server.close();
  }
  if (failures.length) {
    console.error(`\n${failures.length} failure(s)`);
    process.exit(1);
  }
  console.log("site smoke: all routes pass");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
