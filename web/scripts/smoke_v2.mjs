#!/usr/bin/env node
/**
 * Headless check of the v2 look on the exported site (run `pnpm build` first).
 *
 *   node scripts/smoke_v2.mjs [--bout m1-entrants] [--v2-bout ID] [--shots DIR] [--port 4175]
 *
 * Serves web/out and, in Chromium, checks that:
 *   - a bout of version 1 replays still plays in the drawing look, and in the studio look;
 *   - a published bout of version 2 lanes (--v2-bout, default: the first bout whose lanes are
 *     all v2 parts) plays each lane under its own label: the label on every lane is the
 *     bout's, and the replay that lane loads shows that lane's design and part (no lane is
 *     swapped in from another bout). GLBs decode, captions and the offset readout show,
 *     x-ray and collision toggles work, every lane's download fits the hero payload budget
 *     (300 000 bytes), third-party models link their short credits to the credits page,
 *     and the page never names the gripper's maker (that credit lives on /credits/);
 *   - design pages show the studio views (x-ray, cutaway, turntable) without errors;
 *   - /credits/ lists the third-party credits, links the CC BY 4.0 licence, carries the
 *     gripper's copyright notice and has the anchors short credits point at;
 *   - no page has console errors or horizontal scroll at 1280 and 375 px.
 * Screenshots go to --shots (default web/.smoke/v2).
 */

import { mkdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    bout: { type: "string", default: "m1-entrants" },
    "v2-bout": { type: "string" },
    shots: { type: "string", default: path.join(webDir, ".smoke", "v2") },
    port: { type: "string", default: "4175" },
  },
});
const HERO_BUDGET = 300_000;
/** The gripper's maker is credited on /credits/ (and in README, NOTICE, docs/ATTRIBUTION.md), never on a replay page. */
const MAKER = "Robonine";
const DATA = path.join(webDir, "public", "data");
const designs = JSON.parse(readFileSync(path.join(DATA, "designs.json"), "utf8")).designs;
const bouts = JSON.parse(readFileSync(path.join(DATA, "bouts.json"), "utf8")).bouts;
const designWithBout = designs.find((d) => d.bouts.length > 0);
const v2Bout = values["v2-bout"]
  ? bouts.find((b) => b.id === values["v2-bout"])
  : bouts.find((b) => b.lanes.length > 0 && b.lanes.every((l) => l.object.kind === "part"));
if (!v2Bout) {
  console.error(`smoke_v2: no published bout of v2 lanes${values["v2-bout"] ? ` called ${values["v2-bout"]}` : ""}`);
  process.exit(1);
}

/** A lane's download as published: header, poses, overlays and each GLB it draws (lib/assets.ts replayPayload). */
function lanePayload(lane) {
  const file = path.join(DATA, `${lane.replay}.json`);
  const h = JSON.parse(readFileSync(file, "utf8"));
  const dir = path.dirname(file);
  let bytes = statSync(file).size + statSync(path.join(dir, h.bin)).size;
  if (h.overlays) bytes += statSync(path.join(dir, h.overlays.bin)).size;
  for (const a of Object.values(h.assets)) bytes += a.bytes;
  return { header: h, bytes };
}

mkdirSync(values.shots, { recursive: true });
const server = await startServer({ dir: path.join(webDir, "out"), port: Number(values.port) });
const origin = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const failures = [];
const report = {};

async function open(context, route) {
  const page = await context.newPage();
  const errors = [];
  const glbs = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("response", (r) => {
    if (r.url().endsWith(".glb")) glbs.push(new URL(r.url()).pathname.split("/").pop());
  });
  const res = await page.goto(origin + route, { waitUntil: "networkidle" });
  if (res?.status() !== 200) errors.push(`HTTP ${res?.status()}`);
  return { page, errors, glbs };
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

  // 2. The published v2 bout: every lane under its own label, playing its own replay.
  {
    const { page, errors, glbs } = await open(context, `/bout/${v2Bout.id}/`);
    await settle(page, errors);
    const m = await metrics(page);
    const problems = [];
    if (m.loadErrors) problems.push(`${m.loadErrors} load errors`);
    if (m.scrollWidth > m.innerWidth) problems.push(`horizontal scroll ${m.scrollWidth} > ${m.innerWidth}`);
    if (!m.text.includes("simulation replay · MuJoCo")) problems.push("no v2 caption");
    if (m.text.includes(MAKER)) problems.push(`the bout page names the gripper's maker ("${MAKER}"); that credit belongs on /credits/`);
    if (!glbs.length) problems.push("no GLB was requested");
    if (new Set(glbs).size !== glbs.length) problems.push(`a GLB was fetched twice: ${glbs.join(", ")}`);
    const sizes = [];
    for (const lane of v2Bout.lanes) {
      const { header, bytes } = lanePayload(lane);
      sizes.push(bytes);
      report[`${vp.name}_${lane.id}_payload_bytes`] = bytes;
      if (bytes > HERO_BUDGET) problems.push(`lane ${lane.id} downloads ${bytes} B > ${HERO_BUDGET}`);
      // the swapped-label check: the lane's label belongs to the replay it plays
      if (header.meta.design_id !== lane.design_id) problems.push(`lane ${lane.id} plays design ${header.meta.design_id}, labelled ${lane.design_id}`);
      if (header.meta.part_id !== lane.object.part_id) problems.push(`lane ${lane.id} plays part ${header.meta.part_id}, labelled ${lane.object.part_id}`);
      const fig = page.locator(`figure[data-lane="${lane.replay}"]`).first();
      if ((await fig.count()) === 0) {
        problems.push(`no lane figure for ${lane.replay}`);
        continue;
      }
      const shown = (await fig.locator(".heading").first().innerText()).trim();
      if (shown !== lane.label) problems.push(`lane ${lane.id} shows the label "${shown}", the bout says "${lane.label}"`);
      const caption = (await fig.locator("figcaption").innerText()).trim();
      if (!caption.toLowerCase().startsWith(lane.condition.toLowerCase())) problems.push(`lane ${lane.id} caption "${caption}" is not its condition`);
      if (header.overlays?.channels.includes("offset_mm") && !(await fig.innerText()).includes("Offset")) problems.push(`lane ${lane.id}: no offset readout`);
      const thirdParty = Object.values(header.assets).some((a) => a.license !== "Apache-2.0");
      if (thirdParty && !(await fig.locator('[data-credits] a[href*="/credits/#"]').count())) {
        problems.push(`lane ${lane.id} draws third-party models without a credit link`);
      }
    }
    const lane = page.locator(`figure[data-lane="${v2Bout.lanes[0].replay}"]`).first();
    const plain = await shot(page, lane, `${vp.name}-v2-studio`);
    await press(page, "X-ray");
    const xray = await shot(page, lane, `${vp.name}-v2-xray`);
    await press(page, "Collision");
    const coll = await shot(page, lane, `${vp.name}-v2-collision`);
    await press(page, "Angled");
    const angled = await shot(page, lane, `${vp.name}-v2-angled`);
    await page.screenshot({ path: path.join(values.shots, `${vp.name}-v2-bout.png`), fullPage: true });
    if (plain < 8000) problems.push(`v2 lane looks blank (${plain} B png)`);
    if (xray === plain && coll === plain) problems.push("x-ray and collision did not change the image");
    report[`${vp.name}_v2_glbs`] = glbs.length;
    const detail = `lanes=${v2Bout.lanes.length} payload=${sizes.join("/")}B glbs=${glbs.length} png=${plain}/${xray}/${coll}/${angled}`;
    record(`${vp.name} v2 bout ${v2Bout.id}`, [...errors, ...problems], detail);
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

  // 4. Credits.
  {
    const { page, errors } = await open(context, "/credits/");
    const m = await metrics(page);
    const problems = [];
    for (const needle of ["Google Scanned Objects", "CC BY 4.0", "MuJoCo", "three.js", "CERN-OHL-P-2.0", "Nemotron"]) {
      if (!m.text.includes(needle)) problems.push(`credits page lacks "${needle}"`);
    }
    // Short credits in manifests and replays point at these anchors.
    for (const anchor of ["gripper", "gso"]) {
      if (!(await page.locator(`#${anchor}`).count())) problems.push(`credits page has no #${anchor} anchor`);
    }
    // CC BY 4.0 credits must carry a link to the licence and the upstream copyright notice.
    if (!(await page.locator('a[href="https://creativecommons.org/licenses/by/4.0/"]').count())) {
      problems.push("credits page does not link the CC BY 4.0 licence");
    }
    if (!m.text.includes("Copyright (c) 2026 Robonine")) problems.push("credits page lacks the gripper's copyright notice");
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
