#!/usr/bin/env node
/**
 * Headless check of the effector library pages on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_library.mjs [--shots DIR] [--port 4177]
 *
 * Serves web/out and, in Chromium at 1280 and 375 px, checks /library/ and every
 * /library/<key>/ page: HTTP 200, no console or page error, no horizontal scroll, the
 * product-mode banner word for word, and rendered text that names no brand (the tokens of
 * src/rtt/effectors/types.py and src/rtt/library/brands.py, read from the source) and shows no
 * URL (no "http" in the visible text: URLs stay in href attributes). Each detail
 * page with a preview must load its viewer into one WebGL canvas. On /library/ the search must
 * filter the cards (type a word: fewer cards; clear: all cards again). /credits/ must carry the
 * library section with an anchor per item. Screenshots go to --shots (default web/.smoke/library).
 */

import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.join(webDir, "..");
const { values } = parseArgs({
  options: {
    shots: { type: "string", default: path.join(webDir, ".smoke", "library") },
    port: { type: "string", default: "4177" },
  },
});
const LABEL = "Product mode: not part of the pre-registered experiment";
const index = JSON.parse(readFileSync(path.join(webDir, "public", "data", "library", "index.json"), "utf8"));
const items = index.items;

/** One tuple of tokens from a Python source file (one list each, never copied by hand). */
function pythonTokens(rel, name) {
  const src = readFileSync(path.join(repoDir, rel), "utf8");
  const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
  if (!block) throw new Error(`${name} not found in ${rel}`);
  return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}
const tokens = [
  ...new Set([...pythonTokens("src/rtt/effectors/types.py", "BRAND_TOKENS"), ...pythonTokens("src/rtt/library/brands.py", "LIBRARY_ONLY_TOKENS")]),
];
const brandRe = new RegExp(
  `(?<![a-z0-9])(${tokens
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|")})(?![a-z0-9])`,
  "i",
);

/** A search word that matches some cards but not all, with the count the page should show. */
function searchProbe() {
  const hay = items.map((i) => `${i.display_name} ${i.summary} ${i.kind_title} ${i.kind.replace(/_/g, " ")}`.toLowerCase());
  const words = [...new Set(items.flatMap((i) => i.display_name.toLowerCase().split(/[^a-z]+/)))].filter((w) => w.length >= 5);
  for (const w of words) {
    const n = hay.filter((h) => h.includes(w)).length;
    if (n > 0 && n < items.length) return { word: w, expected: n };
  }
  throw new Error("no search word narrows the catalog");
}
const probe = searchProbe();

const routes = ["/library/", ...items.map((i) => `/library/${i.key}/`), "/credits/"];
const widths = [
  { name: "desktop", width: 1280, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

async function exerciseSearch(page) {
  const problems = [];
  const input = page.locator("[data-library-search]");
  const count = () => page.locator("[data-library-card]").count();
  const before = await count();
  if (before !== items.length) problems.push(`catalog shows ${before} cards for ${items.length} items`);
  await input.fill(probe.word);
  await page
    .waitForSelector(`[data-library-count="${probe.expected}"]`, { timeout: 5000 })
    .catch(() => problems.push(`search "${probe.word}" did not narrow to ${probe.expected}`));
  const narrowed = await count();
  if (!(narrowed < before && narrowed === probe.expected)) problems.push(`search "${probe.word}": ${narrowed} cards, expected ${probe.expected}`);
  if (!page.url().includes(`q=${encodeURIComponent(probe.word)}`)) problems.push("search not kept in the URL query");
  await input.fill("");
  await page
    .waitForSelector(`[data-library-count="${items.length}"]`, { timeout: 5000 })
    .catch(() => problems.push("clearing the search did not restore every card"));
  const restored = await count();
  if (restored !== before) problems.push(`after clearing: ${restored} cards, expected ${before}`);
  if (page.url().includes("q=")) problems.push("cleared search still in the URL query");
  return { problems, note: `search "${probe.word}" ${before}->${narrowed}->${restored}` };
}

mkdirSync(values.shots, { recursive: true });
const server = await startServer({ dir: path.join(webDir, "out"), port: Number(values.port) });
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const failures = [];

for (const vp of widths) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, reducedMotion: "reduce" });
  for (const route of routes) {
    const page = await context.newPage();
    const errors = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(String(err)));
    const res = await page.goto(origin + route, { waitUntil: "networkidle" });
    const problems = [...errors];
    let note = "";
    if ((res?.status() ?? 0) !== 200) problems.push(`HTTP ${res?.status()}`);
    const key = route.match(/^\/library\/([^/]+)\/$/)?.[1];
    const item = key ? items.find((i) => i.key === key) : null;
    if (route.startsWith("/library/")) {
      const banner = await page.locator("[data-product-mode]").first().innerText().catch(() => "");
      if (!banner.toLowerCase().includes(LABEL.toLowerCase())) problems.push("no product-mode banner");
    }
    if (item?.preview) {
      await page
        .waitForSelector('[data-viewer-state="ready"], [data-viewer-state="error"]', { timeout: 30000 })
        .catch(() => problems.push("viewer still loading after 30 s"));
      await page.waitForTimeout(300);
      const state = await page.locator("[data-viewer-state]").first().getAttribute("data-viewer-state").catch(() => null);
      if (state !== "ready") problems.push(`viewer state ${state}`);
    }
    if (route === "/library/") {
      const r = await exerciseSearch(page);
      problems.push(...r.problems);
      note = r.note;
    }
    const metrics = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      canvases: document.querySelectorAll("canvas").length,
      text: document.body.innerText,
      failed: [...document.querySelectorAll("p")].filter((p) => p.textContent?.includes("could not be loaded")).length,
      librarySection: !!document.getElementById("library"),
      credits: [...document.querySelectorAll("[data-library-credit]")].map((e) => e.id),
    }));
    if (metrics.scrollWidth > metrics.innerWidth) problems.push(`horizontal scroll: ${metrics.scrollWidth} > ${metrics.innerWidth}`);
    if (metrics.failed) problems.push("the 3D preview failed to load");
    if (item?.preview && metrics.canvases !== 1) problems.push(`${metrics.canvases} canvases, expected 1`);
    if (route.startsWith("/library/")) {
      const hit = metrics.text.replace(/https?:\/\/\S+/g, " ").match(brandRe);
      if (hit) problems.push(`names a brand: ${hit[1]}`);
      const url = metrics.text.match(/\S*http\S*/i);
      if (url) problems.push(`shows a URL as text: ${url[0].slice(0, 60)}`);
    }
    if (route === "/credits/") {
      if (!metrics.librarySection) problems.push("no library credits section");
      const missing = items.filter((i) => !metrics.credits.includes(i.credit_anchor)).map((i) => i.key);
      if (missing.length) problems.push(`no credit anchor for ${missing.join(", ")}`);
      note = `credits=${metrics.credits.length}`;
    }
    const shot = path.join(values.shots, `${vp.name}${route.replace(/\/$/, "").replace(/\//g, "-")}.png`);
    await page.screenshot({ path: shot, fullPage: true });
    console.log(
      `${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${route.padEnd(48)} canvases=${metrics.canvases} ${note} ${problems.join("; ")}`,
    );
    if (problems.length) failures.push({ route, viewport: vp.name, problems });
    await page.close();
  }
  await context.close();
}

await browser.close();
server.close();
console.log(
  failures.length ? `${failures.length} page checks failed` : `all ${routes.length * widths.length} page checks passed; screenshots in ${values.shots}`,
);
process.exit(failures.length ? 1 : 0);
