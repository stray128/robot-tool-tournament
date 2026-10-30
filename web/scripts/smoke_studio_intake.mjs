#!/usr/bin/env node
/**
 * Headless smoke test of the Studio planner, on the Studio page (run `pnpm build` first; the old
 * /studio/new/ route redirects there).
 *
 *   node scripts/smoke_studio_intake.mjs [--shots DIR] [--port 4175]
 *
 * Serves web/out and, at 1280 px and 375 px, fails on: a non-200 response, a console or page
 * error, horizontal page scroll, a missing product-mode label, or a brand token on screen
 * (BRAND_TOKENS read from src/rtt/effectors/types.py). It then drives the page: plays the
 * hero examples' recorded plans (the v2 tile one shows its repair call), breaks and fixes
 * an object parameter, reorders, deletes and adds steps, downloads the TaskSpec JSON (contract,
 * label, a 64-hex task id), plays a refused example (nearest primitive shown), plans a custom
 * text (the static site says it runs in Explore mode), drops an STL (size readout) and builds a plan by hand.
 * Screenshots go to --shots (default web/.smoke/studio-intake).
 */

import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { runnerUrlFromEnv, startMockFromEnv } from "./runner_env.mjs";
import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    shots: { type: "string", default: path.join(webDir, ".smoke", "studio-intake") },
    port: { type: "string", default: "4175" },
  },
});

const LABEL = "Product mode: not part of the pre-registered experiment";
const HOSTED = runnerUrlFromEnv() !== null;
const STL = path.join(webDir, "public", "data", "sessions", "parts", "dumbbell-handle.stl");

function brandPattern() {
  const src = readFileSync(path.join(webDir, "..", "src", "rtt", "effectors", "types.py"), "utf8");
  const block = src.match(/BRAND_TOKENS: Final\[tuple\[str, \.\.\.\]\] = \(([\s\S]*?)\n\)/);
  if (!block) throw new Error("BRAND_TOKENS not found in src/rtt/effectors/types.py");
  const tokens = [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  const alt = tokens.sort((a, b) => b.length - a.length).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`(?<![a-z0-9])(${alt.join("|")})(?![a-z0-9])`, "i");
}
const brandRe = brandPattern();

async function mainText(page) {
  return page.locator("main").innerText();
}

async function waitText(page, needle, problems, what, timeout = 10000) {
  await page
    .waitForFunction((n) => document.querySelector("main")?.innerText.includes(n), needle, { timeout })
    .catch(() => problems.push(`${what}: "${needle}" never appeared`));
}

async function playExample(page, title) {
  await page.getByRole("button", { name: new RegExp(title) }).click();
  await page.getByRole("button", { name: "Plan", exact: true }).click();
}

/** Drive the page; returns the problems found. */
async function exercise(page, vp) {
  const problems = [];
  // 1. a hero example: replay, edit, export
  await playExample(page, "Returns tote");
  await waitText(page, "Plan is valid", problems, "tote");
  const steps = await page.locator('article[aria-label^="Step "]').count();
  const objects = await page.locator('article[aria-label^="Object "]').count();
  if (steps !== 4 || objects < 3) problems.push(`tote plan: ${steps} steps, ${objects} objects`);
  // the header counts in the plural for several, and in the singular for one (see the hand-built plan below)
  if (!/Plan is valid: \d+ objects, 4 steps/.test(await mainText(page))) problems.push("tote plan: the header does not read \"Plan is valid: N objects, 4 steps\"");
  await page.getByRole("button", { name: "Edit object 1" }).click();
  const width = page.getByLabel("Object 1 width_mm", { exact: true });
  const before = await width.inputValue();
  await width.fill("900");
  await waitText(page, "width_mm must be in [60, 600] mm", problems, "bound check");
  if (!(await page.getByRole("button", { name: "Download TaskSpec JSON" }).isDisabled())) problems.push("download enabled for an invalid plan");
  await width.fill(before);
  await waitText(page, "Plan is valid", problems, "bound restored");
  await page.getByRole("button", { name: "Move step 4 up" }).click();
  await waitText(page, "which is not an earlier step", problems, "reorder check");
  await page.getByRole("button", { name: "Move step 3 down" }).click();
  await waitText(page, "Plan is valid", problems, "reorder restored");
  await page.getByRole("button", { name: "Delete step 3" }).click();
  await waitText(page, "no step acts on it", problems, "delete warning");
  await page.getByRole("button", { name: "Add a step" }).click();
  await waitText(page, "Plan is valid", problems, "added step");
  if ((await page.locator('article[aria-label^="Step "]').count()) !== 4) problems.push("add step did not add a step");
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 10000 }), page.getByRole("button", { name: "Download TaskSpec JSON" }).click()]);
  const doc = JSON.parse(readFileSync(await download.path(), "utf8"));
  if (doc.contract !== "rtt-effectors/2" || doc.label !== LABEL || !/^[0-9a-f]{64}$/.test(doc.task_id ?? "") || doc.source !== "manual") {
    problems.push(`exported document: ${JSON.stringify({ contract: doc.contract, label: doc.label, task_id: doc.task_id, source: doc.source })}`);
  }
  if (doc.steps?.length !== 4 || Object.keys(doc.fidelity ?? {}).length !== 4) problems.push("exported document has the wrong steps");
  await page.screenshot({ path: path.join(values.shots, `${vp}-tote.png`), fullPage: true });

  // 2. the other heroes (the Studio v2 hero examples; the tile plan needed one repair call)
  await playExample(page, "Pipe coupling");
  await waitText(page, "Plan is valid", problems, "coupling");
  await playExample(page, "Floor tile setting");
  await waitText(page, "Call 2 (repair)", problems, "tile repair");
  await waitText(page, "Plan is valid", problems, "tile");
  const tile = await mainText(page);
  if (!/APPROXIMATION/.test(tile)) problems.push("the tile plan shows no approximation tier");
  await page.screenshot({ path: path.join(values.shots, `${vp}-tile.png`), fullPage: true });

  // 3. a refusal
  await page.locator("select").first().selectOption({ label: "unsupported: Fold a shirt (not simulated)" });
  await page.getByRole("button", { name: "Plan", exact: true }).click();
  await waitText(page, "REFUSED", problems, "refusal");
  await waitText(page, "Nearest primitive", problems, "refusal nearest primitive");
  if (await page.getByRole("button", { name: "Download TaskSpec JSON" }).count()) problems.push("a refusal offers an export");
  await page.screenshot({ path: path.join(values.shots, `${vp}-refusal.png`), fullPage: true });
  await page.getByRole("button", { name: "Start a plan with it" }).first().click();
  await waitText(page, "Plan is valid", problems, "plan from the nearest primitive");

  // 4. custom text on the static site
  await page.getByLabel("Use case", { exact: true }).fill("Juggle three balls on the bench.");
  await page.getByRole("button", { name: "Plan", exact: true }).click();
  // a build with the hosted runner (lib/live_copy.ts) says Plan only replays and points to the live panel
  await waitText(page, HOSTED ? "This public site runs in Explore mode, so Plan can only replay" : "This public site runs in Explore mode: it has no model to call", problems, "custom text");
  const customText = await mainText(page);
  if (HOSTED && !customText.includes("Plan and run it live")) problems.push("custom text: a build with the runner does not point to the live panel");
  if (HOSTED && customText.includes("has no model to call")) problems.push("custom text: a build with the runner still says it has no model to call");

  // 5. an STL, then a plan by hand
  await page.setInputFiles('input[type="file"]', STL);
  await waitText(page, "triangles", problems, "STL readout");
  await page.getByRole("button", { name: "Build a plan by hand" }).click();
  await waitText(page, "a task needs at least one step (or a refusal)", problems, "empty plan");
  await page.getByRole("combobox", { name: "Add an object" }).selectOption("part");
  await page.getByRole("button", { name: "Add an object" }).click();
  await page.getByRole("combobox", { name: "Add a step" }).selectOption("grasp");
  await page.getByRole("button", { name: "Add a step" }).click();
  await waitText(page, "Plan is valid", problems, "hand-built plan");
  // one object and one step read in the singular: "1 object, 1 step", never "1 objects" or "1 steps"
  const built = await mainText(page);
  if (!/Plan is valid: 1 object, 1 step(?!s)/.test(built)) problems.push(`hand-built plan: the header reads "${built.match(/Plan is valid:[^\n]*/)?.[0]}", not "Plan is valid: 1 object, 1 step"`);
  if (/\b1 (objects|steps)\b/.test(built)) problems.push('hand-built plan: "1 objects" or "1 steps" on screen');
  return problems;
}

const widths = [
  { name: "desktop", width: 1280, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

mkdirSync(values.shots, { recursive: true });
// A build with NEXT_PUBLIC_RUNNER_URL asks the runner for /health; answer it. Unset: nothing starts.
const mock = await startMockFromEnv();
const server = await startServer({ dir: path.join(webDir, "out"), port: Number(values.port) });
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();
const failures = [];

for (const vp of widths) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, reducedMotion: "reduce", acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  const res = await page.goto(`${origin}/studio/`, { waitUntil: "networkidle" });
  const problems = [];
  if ((res?.status() ?? 0) !== 200) problems.push(`HTTP ${res?.status()}`);
  const first = await mainText(page);
  if (!first.includes(LABEL)) problems.push("no product-mode label");
  problems.push(...(await exercise(page, vp.name)));
  const metrics = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth, text: document.body.innerText }));
  if (metrics.scrollWidth > metrics.innerWidth) problems.push(`horizontal scroll: ${metrics.scrollWidth} > ${metrics.innerWidth}`);
  const hit = metrics.text.match(brandRe);
  if (hit) problems.push(`names a brand: ${hit[1]}`);
  problems.push(...errors);
  await page.screenshot({ path: path.join(values.shots, `${vp.name}-final.png`), fullPage: true });
  console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} /studio/ ${problems.join("; ")}`);
  if (problems.length) failures.push({ viewport: vp.name, problems });
  await context.close();
}

await browser.close();
server.close();
await mock?.close();
console.log(failures.length ? `${failures.length} intake checks failed` : `intake checks passed at ${widths.length} widths; screenshots in ${values.shots}`);
process.exit(failures.length ? 1 : 0);
