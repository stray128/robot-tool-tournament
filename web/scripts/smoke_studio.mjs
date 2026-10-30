#!/usr/bin/env node
/**
 * Headless check of the Studio pages on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_studio.mjs [--port 4191] [--shots DIR] [--out DIR]
 *
 * Serves web/out (or --out) and, in Chromium at 1280 x 900 and 375 x 812, checks /studio/, every
 * /studio/<hero>/ listed in the served data/studio/index.json, and /studio/new/ (a redirect to the
 * planner on the Studio page): HTTP 200, no
 * console or page error, no horizontal page scroll, and a primary nav that links to the Studio.
 * On the Studio's own pages also: the product-mode banner word for word, no brand token in the
 * visible text (BRAND_TOKENS of src/rtt/effectors/types.py and LIBRARY_ONLY_TOKENS of
 * src/rtt/library/brands.py, read from the source), no "http" in the visible text (URLs stay in
 * href attributes), and, on a hero page: every lane loads into a WebGL canvas, the success matrix
 * gives a pruned cell's reason on keyboard focus, a no-recommendation page says "No setup
 * completes this task", an approximation recommendation shows "verify on hardware", and the build
 * pack link answers 200 with the bytes the page states. Screenshots go to --shots (default
 * web/.smoke/studio): studio-desktop.png, h1-desktop.png, h1-phone.png, new-phone.png, ...
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startMockFromEnv } from "./runner_env.mjs";
import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoDir = path.join(webDir, "..");
const { values } = parseArgs({
  options: {
    shots: { type: "string", default: path.join(webDir, ".smoke", "studio") },
    port: { type: "string", default: "4191" },
    out: { type: "string", default: path.join(webDir, "out") },
  },
});
const KEY_LINE = "Your key is sent once over HTTPS for this request and is never stored or logged.";
const LABEL = "Product mode: not part of the pre-registered experiment";
const outDir = path.resolve(values.out);
const indexFile = path.join(outDir, "data", "studio", "index.json");
const index = existsSync(indexFile) ? JSON.parse(readFileSync(indexFile, "utf8")) : { heroes: [] };
const pages = Object.fromEntries(index.heroes.map((h) => [h.hero, JSON.parse(readFileSync(path.join(outDir, "data", h.page), "utf8"))]));

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

const routes = ["/studio/", ...index.heroes.map((h) => `/studio/${h.hero}/`), "/studio/new/"];
const widths = [
  { name: "desktop", width: 1280, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

function shotName(route, vp) {
  const id = route === "/studio/" ? "studio" : route.replace(/^\/studio\//, "").replace(/\/$/, "").replace(/\//g, "-");
  return `${id}-${vp}.png`;
}

/** Hero-page checks: lanes, matrix focus, outcome wording, build pack download. */
async function heroChecks(page, hero, vp, origin) {
  const problems = [];
  const data = pages[hero];
  const notes = [];
  const nLanes = data.lanes.length;
  if (nLanes > 0) {
    await page
      .waitForFunction(() => !document.body.innerText.includes("Loading replay"), null, { timeout: 30000 })
      .catch(() => problems.push("lanes still loading after 30 s"));
    await page.waitForTimeout(500);
    const m = await page.evaluate(() => ({
      lanes: document.querySelectorAll("[data-lane]").length,
      canvases: document.querySelectorAll("[data-lane] canvas").length,
      webgl: [...document.querySelectorAll("[data-lane] canvas")].filter((c) => {
        const gl = c.getContext("webgl2") ?? c.getContext("webgl");
        return !!gl;
      }).length,
      failed: [...document.querySelectorAll("[data-lane] p")].filter((p) => p.textContent?.includes("could not be loaded")).map((p) => p.textContent),
      // The story page draws one step timeline for every lane (a track each) instead of a strip per lane.
      strips: document.querySelectorAll("[data-step-strip]").length + document.querySelectorAll("[data-step-timeline] [data-track]").length,
    }));
    if (m.lanes !== nLanes) problems.push(`${m.lanes} lanes on the page, the data has ${nLanes}`);
    if (m.canvases < nLanes || m.webgl < nLanes) problems.push(`${m.canvases} canvases (${m.webgl} WebGL) for ${nLanes} lanes`);
    if (m.failed.length) problems.push(`lanes failed: ${m.failed.join(" | ").slice(0, 200)}`);
    if (m.strips !== nLanes) problems.push(`${m.strips} step strips for ${nLanes} lanes`);
    // Seek to the end: every lane shows its outcome stamp.
    await page.locator("[data-studio-transport] input[type=range]").evaluate((el) => {
      const input = /** @type {HTMLInputElement} */ (el);
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
      setter.call(input, input.max);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForTimeout(300);
    const stamps = await page.locator("[data-lane] [data-outcome]").count();
    if (stamps !== nLanes) problems.push(`${stamps} outcome stamps at the end for ${nLanes} lanes`);
    notes.push(`lanes=${m.lanes} canvases=${m.canvases}`);
  } else if ((await page.locator("[data-no-lanes]").count()) !== 1) {
    problems.push("no lanes, and no note saying the replays are not exported yet");
  }
  // The matrix (under "Show all data" on a story page): a pruned cell gives its reason on keyboard focus.
  const allData = page.locator("#all-data > summary");
  if ((await allData.count()) > 0) await allData.click();
  const pruned = page.locator('[data-cell="pruned"] [tabindex="0"]').first();
  if ((await pruned.count()) > 0) {
    await pruned.focus();
    const reason = await pruned.getAttribute("data-reason");
    const detail = await page.locator("[data-matrix-detail]").innerText();
    if (!reason || !detail.includes(reason)) problems.push("focusing a pruned cell does not show its reason");
    await page.locator("body").click({ position: { x: 1, y: 1 } });
  }
  if (data.outcome === "no_recommendation") {
    const text = await page.locator("[data-no-winner]").innerText().catch(() => "");
    if (!text.includes("No setup completes this task")) problems.push('no "No setup completes this task" block');
  } else if (data.recommendation?.verify_on_hardware) {
    const text = await page
      .locator("[data-recommendation], [data-verify]")
      .first()
      .innerText()
      .catch(() => "");
    if (!/verify on hardware/i.test(text)) problems.push('an approximation recommendation without "verify on hardware"');
  }
  if (data.build_pack) {
    const href = await page.locator("[data-build-pack-link]").getAttribute("href").catch(() => null);
    if (!href) problems.push("no build pack link");
    else {
      const res = await page.request.get(origin + href);
      const body = await res.body();
      if (res.status() !== 200) problems.push(`build pack ${href}: HTTP ${res.status()}`);
      else if (body.byteLength !== data.build_pack.bytes) problems.push(`build pack ${href}: ${body.byteLength} bytes, the page says ${data.build_pack.bytes}`);
      notes.push(`pack=${body.byteLength}B`);
    }
  } else if ((await page.locator('[data-build-pack="none"]').count()) !== 1) {
    problems.push("no build pack and no note saying why");
  }
  if (vp === "phone") {
    const box = await page.locator("[data-matrix] [role=region]").first().evaluate((el) => ({ sw: el.scrollWidth, cw: el.clientWidth }));
    notes.push(`matrix scroll ${box.sw}/${box.cw}`);
  }
  return { problems, note: notes.join(" ") };
}

/**
 * A full-page screenshot, in parts of at most 8000 px: software WebGL caps the capture texture at
 * 8192 px, and a taller capture repeats the page top. The first part keeps the plain name
 * (h1-desktop.png), later parts are numbered (h1-desktop-2.png, ...).
 */
async function screenshot(page, file) {
  const CHUNK = 8000;
  const { width, height } = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight }));
  for (let i = 0, y = 0; y < height; i++, y += CHUNK) {
    const name = i === 0 ? file : file.replace(/\.png$/, `-${i + 1}.png`);
    const opts = { path: name, fullPage: true, clip: { x: 0, y, width, height: Math.min(CHUNK, height - y) } };
    // A capture of a tall WebGL page can time out on a loaded host: retry once with more time.
    await page.screenshot(opts).catch(() => page.screenshot({ ...opts, timeout: 90_000 }));
  }
}

mkdirSync(values.shots, { recursive: true });
// A build with NEXT_PUBLIC_RUNNER_URL asks the runner for /health; answer it. Unset: nothing starts.
const mock = await startMockFromEnv();
const server = await startServer({ dir: outDir, port: Number(values.port) });
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
    const own = route !== "/studio/new/";
    const hero = route.match(/^\/studio\/([^/]+)\/$/)?.[1];
    if (hero && pages[hero]) {
      const r = await heroChecks(page, hero, vp.name, origin);
      problems.push(...r.problems);
      note = r.note;
    }
    const metrics = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      text: document.body.innerText,
      banner: document.querySelector("[data-product-mode]")?.textContent ?? "",
      nav: [...document.querySelectorAll('nav[aria-label="Primary"] a')].map((a) => [a.textContent?.trim(), a.getAttribute("href")]),
      heroes: document.querySelectorAll("[data-hero]").length,
      // The planner lives on the Studio page (/studio/new/ redirects there).
      planOwn: !!document.querySelector('a[href$="/studio/new/"], a[href$="/studio/#plan"], a[href$="/studio/"], #plan'),
    }));
    if (metrics.scrollWidth > metrics.innerWidth) problems.push(`horizontal scroll: ${metrics.scrollWidth} > ${metrics.innerWidth}`);
    if (!metrics.nav.some(([label, href]) => label === "Studio" && href?.endsWith("/studio/"))) problems.push("the nav has no Studio link");
    if (own) {
      if (!metrics.banner.includes(LABEL)) problems.push("no product-mode banner, word for word");
      const hit = metrics.text.match(brandRe);
      if (hit) problems.push(`names a brand: ${hit[1]}`);
      // The hosted runner's privacy line says "HTTPS" in words; it is not a URL.
      const url = metrics.text.replace(KEY_LINE, "").match(/\S*http\S*/i);
      if (url) problems.push(`shows a URL as text: ${url[0].slice(0, 60)}`);
    }
    if (route === "/studio/") {
      if (metrics.heroes !== index.heroes.length) problems.push(`${metrics.heroes} use-case cards for ${index.heroes.length} heroes`);
      if (!metrics.planOwn) problems.push("no planner on the Studio page");
      if (index.heroes.length === 0 && !metrics.text.includes("No use case has been exported yet")) problems.push("empty index without its note");
      note = `cards=${metrics.heroes}`;
    }
    if (hero && pages[hero] && !metrics.planOwn) problems.push("no link back to the planner");
    await screenshot(page, path.join(values.shots, shotName(route, vp.name)));
    console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${route.padEnd(16)} ${note} ${problems.join("; ")}`);
    if (problems.length) failures.push({ route, viewport: vp.name, problems });
    await page.close();
  }
  await context.close();
}

await browser.close();
server.close();
await mock?.close();
console.log(
  failures.length ? `${failures.length} page checks failed` : `all ${routes.length * widths.length} page checks passed; screenshots in ${values.shots}`,
);
process.exit(failures.length ? 1 : 0);
