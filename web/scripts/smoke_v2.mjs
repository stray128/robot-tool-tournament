#!/usr/bin/env node
/**
 * Headless check of the v2 look on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_v2.mjs [--fixture DIR] [--stem sample] [--bout m1-entrants]
 *                             [--lane replays/m1/straight] [--shots DIR] [--port 4175]
 *
 * Serves web/out and, in Chromium, checks that:
 *   - a bout of version 1 replays still plays in the drawing look, and in the studio look;
 *   - the same bout plays with one lane swapped for a version 2 replay (the fixture's
 *     <stem>.json, .bin, .ovl.bin and assets/, served in place of --lane by request
 *     interception): GLBs decode, the caption and the offset readout show, x-ray and
 *     collision toggles work, and the v2 lane's download stays within the hero payload
 *     budget (300 000 bytes);
 *   - design pages show the studio views (x-ray, cutaway, turntable) without errors;
 *   - /credits/ lists the third-party credits;
 *   - no page has console errors or horizontal scroll at 1280 and 375 px.
 * The default fixture is tests/fixtures/replay_v2 (Apache-2.0 geometry made from a
 * recorded v1 episode). Screenshots go to --shots (default web/.smoke/v2).
 */

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    fixture: { type: "string", default: path.join(webDir, "tests", "fixtures", "replay_v2") },
    stem: { type: "string", default: "sample" },
    bout: { type: "string", default: "m1-entrants" },
    lane: { type: "string", default: "replays/m1/straight" },
    shots: { type: "string", default: path.join(webDir, ".smoke", "v2") },
    port: { type: "string", default: "4175" },
  },
});
const HERO_BUDGET = 300_000;
const fixtureDir = path.resolve(values.fixture);
const header = JSON.parse(readFileSync(path.join(fixtureDir, `${values.stem}.json`), "utf8"));
const laneDir = values.lane.slice(0, values.lane.lastIndexOf("/") + 1);
const designs = JSON.parse(readFileSync(path.join(webDir, "public/data/designs.json"), "utf8")).designs;
const bouts = JSON.parse(readFileSync(path.join(webDir, "public/data/bouts.json"), "utf8")).bouts;
const designWithBout = designs.find((d) => d.bouts.length > 0);

mkdirSync(values.shots, { recursive: true });
const server = await startServer({ dir: path.join(webDir, "out"), port: Number(values.port) });
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const failures = [];
const report = {};

/** Serve the v2 fixture in place of one v1 lane; returns the byte counter for the swapped lane. */
async function swapLane(page) {
  const counted = { bytes: 0, files: [] };
  const serveFile = async (route, file, type) => {
    const body = readFileSync(file);
    counted.bytes += body.byteLength;
    counted.files.push(path.basename(file));
    await route.fulfill({ status: 200, body, contentType: type });
  };
  await page.route(`**/data/${values.lane}.json`, (r) => serveFile(r, path.join(fixtureDir, `${values.stem}.json`), "application/json"));
  await page.route(`**/data/${laneDir}${header.bin}`, (r) => serveFile(r, path.join(fixtureDir, header.bin), "application/octet-stream"));
  if (header.overlays) {
    await page.route(`**/data/${laneDir}${header.overlays.bin}`, (r) =>
      serveFile(r, path.join(fixtureDir, header.overlays.bin), "application/octet-stream"),
    );
  }
  await page.route("**/assets/*.glb", (r) => {
    const name = new URL(r.request().url()).pathname.split("/").pop();
    const file = path.join(fixtureDir, "assets", name);
    return existsSync(file) ? serveFile(r, file, "model/gltf-binary") : r.continue();
  });
  return counted;
}

async function open(context, route, { swap = false } = {}) {
  const page = await context.newPage();
  const errors = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  const counted = swap ? await swapLane(page) : null;
  const res = await page.goto(origin + route, { waitUntil: "networkidle" });
  if (res?.status() !== 200) errors.push(`HTTP ${res?.status()}`);
  return { page, errors, counted };
}

async function settle(page, errors) {
  await page
    .waitForFunction(() => !document.body.innerText.includes("Loading replay"), null, { timeout: 20000 })
    .catch(() => errors.push("replay lanes still loading after 20 s"));
  await page.waitForTimeout(800);
}

async function metrics(page) {
  return page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    canvases: document.querySelectorAll("canvas").length,
    loadErrors: [...document.querySelectorAll("p")].filter((p) => p.textContent?.includes("could not be loaded")).length,
    text: document.body.innerText,
  }));
}

/**
 * PNG of an element; a blank or uniform canvas compresses to almost nothing. Fails
 * when the element keeps changing size (a resize feedback loop).
 */
async function shot(page, locator, name) {
  await locator.scrollIntoViewIfNeeded();
  const boxes = [];
  for (let i = 0; i < 3; i++) {
    boxes.push(await locator.boundingBox());
    await page.waitForTimeout(150);
  }
  const [a, , c] = boxes;
  if (!a || !c || Math.abs(a.width - c.width) > 1 || Math.abs(a.height - c.height) > 1) {
    throw new Error(`${name}: element size is not stable: ${JSON.stringify(boxes)}`);
  }
  const buf = await page.screenshot({ path: path.join(values.shots, `${name}.png`), clip: c, animations: "disabled" });
  return buf.byteLength;
}

function record(name, problems, extra = "") {
  console.log(`${problems.length ? "FAIL" : "ok  "} ${name.padEnd(44)} ${extra} ${problems.join("; ")}`);
  if (problems.length) failures.push({ name, problems });
}

async function press(page, label) {
  await page.getByRole("button", { name: label, exact: true }).first().click();
  await page.waitForTimeout(600);
}

for (const vp of [
  { name: "desktop", width: 1280, height: 900 },
  { name: "phone", width: 375, height: 812 },
]) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, reducedMotion: "reduce" });

  // 1. v1 bout, drawing look (unchanged) and studio look.
  {
    const { page, errors } = await open(context, `/bout/${values.bout}/`);
    await settle(page, errors);
    const m = await metrics(page);
    const problems = [];
    if (m.canvases === 0) problems.push("no canvas");
    if (m.loadErrors) problems.push(`${m.loadErrors} load errors`);
    if (m.scrollWidth > m.innerWidth) problems.push(`horizontal scroll ${m.scrollWidth} > ${m.innerWidth}`);
    const lane = page.locator("figure").first();
    const drawingPng = await shot(page, lane, `${vp.name}-v1-drawing`);
    await press(page, "Studio");
    const studioPng = await shot(page, lane, `${vp.name}-v1-studio`);
    if (studioPng < 8000) problems.push(`studio lane looks blank (${studioPng} B png)`);
    const after = await metrics(page);
    if (after.canvases !== m.canvases) problems.push(`canvas count changed ${m.canvases} -> ${after.canvases}`);
    record(`${vp.name} v1 bout drawing + studio`, [...errors, ...problems], `canvases=${m.canvases} png=${drawingPng}/${studioPng}`);
    await page.close();
  }

  // 2. Same bout with one lane served as a v2 replay.
  {
    const { page, errors, counted } = await open(context, `/bout/${values.bout}/`, { swap: true });
    await settle(page, errors);
    const m = await metrics(page);
    const problems = [];
    if (m.loadErrors) problems.push(`${m.loadErrors} load errors`);
    if (m.scrollWidth > m.innerWidth) problems.push(`horizontal scroll ${m.scrollWidth} > ${m.innerWidth}`);
    if (!m.text.includes("simulation replay · MuJoCo")) problems.push("no v2 caption");
    if (header.overlays?.channels.includes("offset_mm") && !m.text.includes("Offset")) problems.push("no offset readout");
    if (counted.bytes > HERO_BUDGET) problems.push(`v2 lane payload ${counted.bytes} B > ${HERO_BUDGET}`);
    if (!counted.files.some((f) => f.endsWith(".glb"))) problems.push("no GLB was requested");
    // The swapped lane, wherever it sits in the bout.
    const lane = page.locator(`figure[data-lane="${values.lane}"]`).first();
    if ((await lane.count()) === 0) problems.push(`no lane figure for ${values.lane}`);
    const plain = await shot(page, lane, `${vp.name}-v2-studio`);
    await press(page, "X-ray");
    const xray = await shot(page, lane, `${vp.name}-v2-xray`);
    await press(page, "Collision");
    const coll = await shot(page, lane, `${vp.name}-v2-collision`);
    await press(page, "Angled");
    const angled = await shot(page, lane, `${vp.name}-v2-angled`);
    if (plain < 8000) problems.push(`v2 lane looks blank (${plain} B png)`);
    if (xray === plain && coll === plain) problems.push("x-ray and collision did not change the image");
    report[`${vp.name}_v2_payload_bytes`] = counted.bytes;
    report[`${vp.name}_v2_files`] = counted.files.length;
    const detail = `payload=${counted.bytes}B files=${counted.files.length} png=${plain}/${xray}/${coll}/${angled}`;
    record(`${vp.name} v2 lane in a v1 bout`, [...errors, ...problems], detail);
    await page.close();
  }

  // 3. Design page: studio views.
  if (designWithBout) {
    const { page, errors } = await open(context, `/design/${designWithBout.id}/`);
    await page.waitForSelector("[data-design-studio] canvas", { timeout: 20000 }).catch(() => errors.push("no design studio canvas"));
    await page.waitForTimeout(1200);
    const problems = [];
    const stage = page.locator("[data-design-studio]").first();
    const sizes = {};
    for (const mode of ["Studio", "X-ray", "Cutaway"]) {
      await stage.getByRole("button", { name: mode, exact: true }).click();
      await page.waitForTimeout(700);
      sizes[mode] = await shot(page, stage.locator("[data-stage]"), `${vp.name}-design-${mode.toLowerCase()}`);
    }
    await stage.getByRole("button", { name: "Turntable", exact: true }).click();
    await page.waitForTimeout(1200);
    sizes.turntable = await shot(page, stage.locator("[data-stage]"), `${vp.name}-design-turntable`);
    const m = await metrics(page);
    if (m.scrollWidth > m.innerWidth) problems.push(`horizontal scroll ${m.scrollWidth} > ${m.innerWidth}`);
    if (m.loadErrors) problems.push(`${m.loadErrors} load errors`);
    if (Object.values(sizes).some((s) => s < 8000)) problems.push(`a design view looks blank ${JSON.stringify(sizes)}`);
    if (new Set(Object.values(sizes)).size < 3) problems.push(`views look identical ${JSON.stringify(sizes)}`);
    record(`${vp.name} design studio views`, [...errors, ...problems], JSON.stringify(sizes));
    await page.close();
  }

  // 4. Design page with the v2 fixture swapped in for one of its lanes.
  if (designWithBout) {
    const bout = bouts.find((b) => designWithBout.bouts.includes(b.id));
    const lane = bout?.lanes.find((l) => l.design_id === designWithBout.id);
    if (lane && lane.replay === values.lane) {
      const { page, errors } = await open(context, `/design/${designWithBout.id}/`, { swap: true });
      await page.waitForSelector("[data-design-studio] canvas", { timeout: 20000 }).catch(() => errors.push("no design studio canvas"));
      await page.waitForTimeout(1500);
      const stage = page.locator("[data-design-studio]").first();
      const png = await shot(page, stage.locator("[data-stage]"), `${vp.name}-design-v2`);
      const problems = png < 8000 ? [`v2 design view looks blank (${png} B)`] : [];
      record(`${vp.name} design studio, v2 lane`, [...errors, ...problems], `png=${png}`);
      await page.close();
    }
  }

  // 5. Credits.
  {
    const { page, errors } = await open(context, "/credits/");
    const m = await metrics(page);
    const problems = [];
    for (const needle of ["Google Scanned Objects", "CC BY 4.0", "MuJoCo", "three.js", "CERN-OHL-P-2.0", "Nemotron"]) {
      if (!m.text.includes(needle)) problems.push(`credits page lacks "${needle}"`);
    }
    if (m.scrollWidth > m.innerWidth) problems.push(`horizontal scroll ${m.scrollWidth} > ${m.innerWidth}`);
    await page.screenshot({ path: path.join(values.shots, `${vp.name}-credits.png`), fullPage: true });
    record(`${vp.name} credits`, [...errors, ...problems]);
    await page.close();
  }
  await context.close();
}

await browser.close();
server.close();
console.log(JSON.stringify(report));
console.log(failures.length ? `${failures.length} v2 checks failed` : `all v2 checks passed; screenshots in ${values.shots}`);
process.exit(failures.length ? 1 : 0);
