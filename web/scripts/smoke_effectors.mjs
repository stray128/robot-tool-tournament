#!/usr/bin/env node
/**
 * Headless check of the effector pages on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_effectors.mjs [--shots DIR] [--port 4176]
 *
 * Serves web/out and, in Chromium at 1280 and 375 px, checks /effectors/ and every
 * /effectors/<part>/ page: HTTP 200, no console or page error, no horizontal scroll, the
 * product-mode banner word for word, every lane loads (one WebGL canvas per lane shown) and
 * downloads at most 300 000 bytes, the rendered text names no brand (the tokens of
 * src/rtt/effectors/types.py), an approximation winner shows its caveat and its rigid
 * runner-up. /part/ must link to /effectors/ and /credits/ must carry the effector section.
 * Screenshots go to --shots (default web/.smoke/effectors).
 */

import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    shots: { type: "string", default: path.join(webDir, ".smoke", "effectors") },
    port: { type: "string", default: "4176" },
  },
});
const LABEL = "Product mode: not part of the pre-registered experiment";
const BUDGET = 300_000;
const DATA = path.join(webDir, "public", "data");
const index = JSON.parse(readFileSync(path.join(DATA, "effectors", "index.json"), "utf8"));
const pages = Object.fromEntries(
  index.parts.map((p) => [p.part_id, JSON.parse(readFileSync(path.join(DATA, p.page), "utf8"))]),
);

/** BRAND_TOKENS from the contract (one list, never copied by hand). */
function brandTokens() {
  const src = readFileSync(path.join(webDir, "..", "src", "rtt", "effectors", "types.py"), "utf8");
  const block = src.match(/BRAND_TOKENS: Final\[tuple\[str, \.\.\.\]\] = \(([\s\S]*?)\n\)/);
  if (!block) throw new Error("BRAND_TOKENS not found in src/rtt/effectors/types.py");
  return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}
const tokens = brandTokens();
const brandRe = new RegExp(
  `(?<![a-z0-9])(${tokens
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|")})(?![a-z0-9])`,
  "i",
);

const routes = ["/effectors/", ...index.parts.map((p) => `/effectors/${p.part_id}/`), "/part/", "/credits/"];
const widths = [
  { name: "desktop", width: 1280, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

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
    const bytes = new Map();
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(String(err)));
    page.on("response", async (res) => {
      const url = new URL(res.url());
      if (!url.pathname.includes("/data/replays/effectors-")) return;
      const stem = url.pathname.replace(/\.(ovl\.bin|bin|json)$/, "");
      const body = await res.body().catch(() => Buffer.alloc(0));
      bytes.set(stem, (bytes.get(stem) ?? 0) + body.byteLength);
    });
    const res = await page.goto(origin + route, { waitUntil: "networkidle" });
    const problems = [...errors];
    if ((res?.status() ?? 0) !== 200) problems.push(`HTTP ${res?.status()}`);
    const partId = route.match(/^\/effectors\/([^/]+)\/$/)?.[1];
    if (route.startsWith("/effectors/")) {
      await page
        .waitForFunction(() => !document.body.innerText.includes("Loading replay"), null, { timeout: 20000 })
        .catch(() => problems.push("lanes still loading after 20 s"));
      await page.waitForTimeout(500);
      const banner = await page.locator("[data-product-mode]").first().innerText().catch(() => "");
      if (!banner.toLowerCase().includes(LABEL.toLowerCase())) problems.push("no product-mode banner");
    }
    const metrics = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      canvases: document.querySelectorAll("canvas").length,
      lanes: document.querySelectorAll("[data-lane]").length,
      text: document.body.innerText,
      failed: [...document.querySelectorAll("p")].filter((p) => p.textContent?.includes("could not be loaded")).length,
      effectorsLink: !!document.querySelector('a[href$="/effectors/"]'),
      creditsSection: !!document.getElementById("effectors"),
    }));
    if (metrics.scrollWidth > metrics.innerWidth) problems.push(`horizontal scroll: ${metrics.scrollWidth} > ${metrics.innerWidth}`);
    if (metrics.failed) problems.push(`${metrics.failed} lanes failed to load`);
    if (partId) {
      const data = pages[partId];
      if (metrics.lanes === 0 || metrics.canvases < metrics.lanes) problems.push(`lanes ${metrics.lanes}, canvases ${metrics.canvases}`);
      for (const [stem, n] of bytes) if (n > BUDGET) problems.push(`${stem} downloaded ${n} bytes`);
      if (bytes.size < metrics.lanes) problems.push(`only ${bytes.size} lane downloads for ${metrics.lanes} lanes`);
      const win = data.recommendation.winner;
      if (win && win.effector.fidelity === "approximation") {
        if (!metrics.text.includes("approximation: verify on hardware")) problems.push("approximation winner without its caveat");
        if (!metrics.text.includes("Rigid runner-up")) problems.push("approximation winner without a rigid runner-up");
      }
    }
    if (route.startsWith("/effectors/")) {
      const hit = metrics.text.match(brandRe);
      if (hit) problems.push(`names a brand: ${hit[1]}`);
    }
    if (route === "/part/" && !metrics.effectorsLink) problems.push("no link to /effectors/");
    if (route === "/credits/" && !metrics.creditsSection) problems.push("no effector credits section");
    const shot = path.join(values.shots, `${vp.name}${route.replace(/\/$/, "").replace(/\//g, "-")}.png`);
    await page.screenshot({ path: shot, fullPage: true });
    console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${route.padEnd(34)} lanes=${metrics.lanes} canvases=${metrics.canvases} ${problems.join("; ")}`);
    if (problems.length) failures.push({ route, viewport: vp.name, problems });
    await page.close();
  }
  await context.close();
}

await browser.close();
server.close();
console.log(failures.length ? `${failures.length} page checks failed` : `all ${routes.length * widths.length} page checks passed; screenshots in ${values.shots}`);
process.exit(failures.length ? 1 : 0);
