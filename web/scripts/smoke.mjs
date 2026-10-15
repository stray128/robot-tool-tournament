#!/usr/bin/env node
/**
 * Headless smoke test of the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke.mjs [--shots DIR] [--port 4174]
 *
 * Serves web/out, opens every route in Chromium at desktop (1280 px) and phone
 * (375 px) widths, and fails on: a non-200 response, a console error or page
 * error, horizontal page scroll, a replay lane that does not finish loading, or a
 * missing WebGL canvas on replay pages. Saves a full-page screenshot per route and
 * width to --shots (default web/.smoke).
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
    shots: { type: "string", default: path.join(webDir, ".smoke") },
    port: { type: "string", default: "4174" },
  },
});

const bouts = JSON.parse(readFileSync(path.join(webDir, "public/data/bouts.json"), "utf8")).bouts;
const designs = JSON.parse(readFileSync(path.join(webDir, "public/data/designs.json"), "utf8")).designs;
const routes = [
  "/",
  "/benchmark/",
  "/results/",
  "/compute/",
  "/about/",
  ...bouts.map((b) => `/bout/${b.id}/`),
  ...designs.map((d) => `/design/${d.id}/`),
];
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
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(String(err)));
    const res = await page.goto(origin + route, { waitUntil: "networkidle" });
    const status = res?.status() ?? 0;
    if (route.startsWith("/bout/") || route === "/") {
      await page
        .waitForFunction(() => !document.body.innerText.includes("Loading replay"), null, { timeout: 15000 })
        .catch(() => errors.push("replay lanes still loading after 15 s"));
      await page.waitForTimeout(500);
    }
    const needsCanvas = route === "/" || route.startsWith("/bout/") || route.startsWith("/design/");
    if (needsCanvas) await page.waitForSelector("canvas", { timeout: 15000 }).catch(() => {});
    const metrics = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      canvases: document.querySelectorAll("canvas").length,
      title: document.title,
      replayErrors: [...document.querySelectorAll("p")].filter((p) => p.textContent?.includes("could not be loaded")).length,
    }));
    const problems = [...errors];
    if (status !== 200) problems.push(`HTTP ${status}`);
    if (metrics.scrollWidth > metrics.innerWidth) problems.push(`horizontal scroll: ${metrics.scrollWidth} > ${metrics.innerWidth}`);
    if (metrics.replayErrors) problems.push(`${metrics.replayErrors} replay/preview load errors`);
    if (needsCanvas && metrics.canvases === 0) problems.push("no WebGL canvas");
    const shot = path.join(values.shots, `${vp.name}${route === "/" ? "-home" : route.replace(/\/$/, "").replace(/\//g, "-")}.png`);
    await page.screenshot({ path: shot, fullPage: true });
    console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${route.padEnd(28)} canvases=${metrics.canvases} ${problems.join("; ")}`);
    if (problems.length) failures.push({ route, viewport: vp.name, problems });
    await page.close();
  }
  await context.close();
}

await browser.close();
server.close();
console.log(failures.length ? `${failures.length} page checks failed` : `all ${routes.length * widths.length} page checks passed; screenshots in ${values.shots}`);
process.exit(failures.length ? 1 : 0);
