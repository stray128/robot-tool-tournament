#!/usr/bin/env node
/**
 * Record every shot of the video script as a 1920 x 1080 clip of the site.
 *
 *   node scripts/video/capture.mjs --out DIR                       # the live site
 *   node scripts/video/capture.mjs --serve web/out --out DIR        # a local build made with
 *                                                                   # NEXT_PUBLIC_BASE_PATH=/robot-tool-tournament
 *   options: --shots scripts/video/shots.json  --only s01,s09  --url URL  --port 4320  --gl egl|swiftshader
 *            --flag NAME (repeatable: a flagged shot's own version; without it, its fallback)
 *
 * shots.json is the machine copy of docs/VIDEO_SCRIPT.md, resolved for the flags given by
 * scripts/video/shots.mjs (the same rules assemble.sh and voiceover.py use). For each shot this script opens a fresh
 * Chromium context recording video, runs the shot's `setup` actions (navigate, wait for the page
 * and its replays to be ready, scroll into place), notes the moment the shot starts, runs the
 * `timeline` actions on the clock, and keeps recording until the shot's duration has passed.
 * Every wait is on a page condition (a selector, a text, the replay lanes loaded), never on a
 * guess; `pause` is only used inside the timeline, where it is the shot's pacing.
 *
 * Output in --out: <id>.webm per shot, <id>.png (the last frame, for a quick look) and
 * clips.json, which records each clip's `trim_s` (seconds of setup before the shot starts) and the
 * flags it was recorded with, for scripts/video/assemble.sh. Nothing is written inside the repository.
 *
 * Presentation: like browser zoom at 150 % (shots.json `presentation`): a larger root font size
 * and a container as wide as the zoomed one, with the scrollbar hidden. A shot may override any of
 * these for itself (its own `presentation`), for example a narrower container so a replay's
 * transport and its whole canvas fit in the frame. Layout only; the page's content, numbers and
 * labels are untouched. WebGL replays render on the GPU through EGL
 * (`--gl egl`, the default) so they play in real time; `--gl swiftshader` is the software fallback.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const web = path.join(repo, "web");
const { chromium } = await import(pathToFileURL(path.join(web, "node_modules", "playwright", "index.mjs")).href);
const { loadShots } = await import(pathToFileURL(path.join(here, "shots.mjs")).href);

const { values } = parseArgs({
  options: {
    shots: { type: "string", default: path.join(here, "shots.json") },
    out: { type: "string" },
    url: { type: "string" },
    serve: { type: "string" },
    port: { type: "string", default: "4320" },
    only: { type: "string" },
    gl: { type: "string", default: "egl" },
    flag: { type: "string", multiple: true, default: [] },
  },
});
if (!values.out) {
  console.error("capture: --out DIR is required (keep media outside the repository)");
  process.exit(2);
}
const outDir = path.resolve(values.out);
if (outDir === repo || outDir.startsWith(repo + path.sep)) {
  console.error(`capture: --out must be outside the repository (${repo})`);
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

const spec = loadShots(values.shots, values.flag);
const flagsOn = Object.entries(spec.flags).filter(([, v]) => v).map(([k]) => k).sort();
const only = values.only ? new Set(values.only.split(",").map((s) => s.trim())) : null;
const shots = spec.shots.filter((s) => !only || only.has(s.id));
const W = spec.width;
const H = spec.height;

let server = null;
let base = values.url ?? spec.site.live;
if (values.serve) {
  const { startServer } = await import(pathToFileURL(path.join(web, "scripts", "serve.mjs")).href);
  server = await startServer({ dir: path.resolve(values.serve), port: Number(values.port), base: spec.site.base });
  base = `http://127.0.0.1:${server.address().port}${spec.site.base}`;
}
base = base.replace(/\/$/, "");

const GL = {
  egl: ["--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu"],
  swiftshader: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
};
if (!GL[values.gl]) throw new Error(`--gl must be one of ${Object.keys(GL).join(", ")}`);

/** The presentation CSS of a shot: the list's `presentation`, with the shot's own keys over it. */
const presentationCss = (shot) => {
  const p = { ...spec.presentation, ...(shot.presentation ?? {}) };
  return `
  html { font-size: ${p.root_font_pct}% !important; scrollbar-width: none !important; }
  html::-webkit-scrollbar, body::-webkit-scrollbar { display: none !important; }
  main#main, header > div { max-width: ${p.container_px}px !important; }
  html { scroll-behavior: auto !important; }
`;
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * The sync mark: a magenta square in the top-left corner, shown for a moment in the pre-roll just
 * before a shot starts (the cut never reaches it). The recording can start a second or two after
 * the page exists on a loaded machine, so the clock alone cannot place the shot in the clip; the
 * first frame that shows the mark can.
 */
const SYNC_ID = "__video_sync_mark";
const SYNC_FPS = 50;
async function showSyncMark(page, on) {
  await page.evaluate(
    ([id, show]) => {
      document.getElementById(id)?.remove();
      if (!show) return;
      const d = document.createElement("div");
      d.id = id;
      d.style.cssText = "position:fixed;left:0;top:0;width:64px;height:64px;background:#ff00ff;z-index:2147483647;pointer-events:none";
      document.documentElement.appendChild(d);
    },
    [SYNC_ID, on],
  );
}

/** Seconds into `file` of the first frame showing the sync mark, or null (ffmpeg missing, mark not found). */
function syncMarkSeconds(file) {
  const r = spawnSync(
    "ffmpeg",
    ["-v", "error", "-i", file, "-vf", `fps=${SYNC_FPS},crop=32:32:16:16,scale=1:1:flags=area,format=rgb24`, "-f", "rawvideo", "pipe:1"],
    { maxBuffer: 1 << 26 },
  );
  if (r.status !== 0 || !r.stdout) return null;
  const px = r.stdout;
  for (let i = 0; i + 2 < px.length; i += 3) {
    if (px[i] > 190 && px[i + 1] < 90 && px[i + 2] > 190) return i / 3 / SYNC_FPS;
  }
  return null;
}

/** The end card: plain HTML, no external files. */
function endCardHtml(card) {
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    html, body { margin: 0; height: 100%; background: #f4f3ee; color: #17181c; font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif; }
    .wrap { height: 100%; display: flex; flex-direction: column; justify-content: center; padding: 0 160px; box-sizing: border-box; }
    h1 { font-size: 112px; margin: 0 0 28px; letter-spacing: -2px; }
    p { font-size: 40px; margin: 0 0 14px; color: #3b3d45; max-width: 1500px; }
    .links { margin-top: 44px; font-family: "DejaVu Sans Mono", monospace; font-size: 38px; color: #2a3fd6; }
    .links div { margin-bottom: 10px; }
    .credit { margin-top: 56px; font-size: 30px; color: #5b5d66; }
  </style></head><body><div class="wrap">
    <h1>${esc(card.title)}</h1>
    ${card.lines.map((l) => `<p>${esc(l)}</p>`).join("")}
    <div class="links">${card.links.map((l) => `<div>${esc(l)}</div>`).join("")}</div>
    <div class="credit">${esc(card.credit)}</div>
  </div></body></html>`;
}

/**
 * Scroll so `selector` sits `offset` px below the top (instant). With `text`, the first element
 * matching `selector` whose text contains `text` (for example one row of a table).
 */
async function scrollTo(page, selector, offset = 0, text = null) {
  await page.waitForSelector(selector, { state: "attached", timeout: 30_000 });
  await page.evaluate(
    ([sel, off, txt]) => {
      const el = txt ? [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(txt)) : document.querySelector(sel);
      if (!el) throw new Error(`no ${sel} holds "${txt}"`);
      window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY + off);
    },
    [selector, offset, text],
  );
}

/**
 * Scroll smoothly over `s` seconds (ease in-out), by `by` px or to `to` (a selector) plus `offset`;
 * with `text`, to the first element matching `to` whose text contains it.
 */
async function smoothScroll(page, { by, to, offset = 0, s, text = null }) {
  await page.evaluate(
    ([byPx, sel, off, ms, txt]) =>
      new Promise((done, fail) => {
        const y0 = window.scrollY;
        const el = sel ? (txt ? [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(txt)) : document.querySelector(sel)) : null;
        if (sel && !el) return fail(new Error(`no ${sel}${txt ? ` holds "${txt}"` : ""}`));
        const target = el ? el.getBoundingClientRect().top + y0 + off : y0 + byPx;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        const y1 = Math.max(0, Math.min(max, target));
        const t0 = performance.now();
        const step = (now) => {
          const f = Math.min(1, (now - t0) / ms);
          const e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
          window.scrollTo(0, y0 + (y1 - y0) * e);
          if (f < 1) requestAnimationFrame(step);
          else done();
        };
        requestAnimationFrame(step);
      }),
    [by ?? 0, to ?? null, offset, s * 1000, text],
  );
}

/**
 * Seek the replays to a step's start in the recommended run plus `offset_s` (negative: before it),
 * through the scrubber. The step is the n-th step card (1-based); its "Watch this step" button holds
 * the start (`data-seek`). Used to show a tool change: a few seconds before the step after it.
 */
async function replaySeek(page, step, offset) {
  const start = await page.evaluate((n) => {
    const card = document.querySelectorAll("[data-step-card]")[n - 1];
    const seek = card?.querySelector("[data-seek]");
    return seek ? Number(seek.getAttribute("data-seek")) : null;
  }, step);
  if (start === null || !Number.isFinite(start)) throw new Error(`step card ${step} has no "Watch this step" time`);
  const at = Math.max(0, start + offset);
  await page.evaluate((t) => {
    const input = document.querySelector("[data-studio-transport] input[type=range]");
    if (!input) throw new Error("no replay scrubber");
    const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    set.call(input, String(Math.min(t, Number(input.max))));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }, at);
  await sleep(300);
}

/** The Studio lanes loaded, paused at 0 at the given speed, ready for replayPlay. */
async function replayReady(page, speed) {
  await page.waitForSelector("[data-studio-lanes]", { timeout: 30_000 });
  await page.locator("[data-studio-transport]").scrollIntoViewIfNeeded();
  await page.waitForFunction(
    () => {
      const lanes = document.querySelectorAll("[data-lane]");
      return lanes.length > 0 && !document.body.innerText.includes("Loading replay") && document.querySelectorAll("[data-lane] canvas").length >= lanes.length;
    },
    null,
    { timeout: 60_000 },
  );
  const transport = page.locator("[data-studio-transport]");
  const toggle = transport.locator("button").first();
  if ((await toggle.innerText()).includes("Pause")) await toggle.click();
  await transport.getByRole("group", { name: "Playback speed" }).getByRole("button", { name: `${speed}x`, exact: true }).click();
  await transport.getByRole("button", { name: "Restart" }).click();
  await page.waitForFunction(() => /t\s*00[.,]\s*00/.test(document.querySelector("[data-studio-transport] p")?.textContent ?? ""), null, { timeout: 10_000 });
}

function locate(page, a) {
  if (a.role) return page.getByRole(a.role, { name: a.name, exact: a.exact ?? false });
  return page.locator(a.selector).first();
}

async function run(page, a, card) {
  switch (a.do) {
    case "goto":
      await page.goto(base + a.path, { waitUntil: "networkidle", timeout: 90_000 });
      await page.evaluate(() => document.fonts.ready);
      return;
    case "endCard":
      await page.setContent(endCardHtml(card), { waitUntil: "load" });
      await page.evaluate(() => document.fonts.ready);
      return;
    case "waitSelector":
      await page.waitForSelector(a.selector, { timeout: (a.timeout_s ?? 30) * 1000 });
      return;
    case "waitText":
      await page.waitForFunction((t) => document.body.innerText.includes(t), a.text, { timeout: (a.timeout_s ?? 30) * 1000 });
      return;
    case "scrollTo":
      await scrollTo(page, a.selector, a.offset ?? 0, a.text ?? null);
      return;
    case "smoothScroll":
      await smoothScroll(page, a);
      return;
    case "pause":
      await sleep(a.s * 1000);
      return;
    case "click":
      await locate(page, a).click();
      return;
    case "focus":
      await locate(page, a).focus();
      return;
    case "type": {
      const el = locate(page, a);
      await el.fill("");
      await el.pressSequentially(a.text, { delay: a.delay_ms ?? 100 });
      return;
    }
    case "replayReady":
      await replayReady(page, a.speed ?? 1);
      return;
    case "replaySeek":
      await replaySeek(page, a.step, a.offset_s ?? 0);
      return;
    case "replayPlay": {
      const toggle = page.locator("[data-studio-transport] button").first();
      if ((await toggle.innerText()).includes("Play")) await toggle.click();
      return;
    }
    case "checkLink": {
      const href = await locate(page, a).getAttribute("href");
      const res = await page.request.get(new URL(href, page.url()).href);
      if (res.status() !== 200) throw new Error(`${a.selector} answers HTTP ${res.status()}`);
      return;
    }
    default:
      throw new Error(`unknown action ${a.do}`);
  }
}

const browser = await chromium.launch({ args: GL[values.gl] });
const manifest = { format: "rtt-video-clips/2", base, width: W, height: H, gl: values.gl, flags: flagsOn, clips: [] };
const manifestFile = path.join(outDir, "clips.json");
if (only && existsSync(manifestFile)) {
  const old = JSON.parse(readFileSync(manifestFile, "utf8"));
  if (JSON.stringify([...(old.flags ?? [])].sort()) !== JSON.stringify(flagsOn)) {
    console.error(`capture: ${manifestFile} was recorded with flags [${old.flags ?? []}]; re-record every shot or pass the same --flag`);
    process.exit(2);
  }
  manifest.clips = old.clips.filter((c) => !only.has(c.id));
}
let failed = 0;

for (const shot of shots) {
  const tmp = path.join(outDir, `.rec-${shot.id}`);
  rmSync(tmp, { recursive: true, force: true });
  const context = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir: tmp, size: { width: W, height: H } },
    colorScheme: "light",
    reducedMotion: "no-preference",
  });
  await context.addInitScript((css) => {
    const add = () => {
      const s = document.createElement("style");
      s.dataset.videoPresentation = "";
      s.textContent = css;
      document.head.appendChild(s);
    };
    if (document.head) add();
    else document.addEventListener("DOMContentLoaded", add);
  }, presentationCss(shot));
  const page = await context.newPage();
  const t0 = Date.now();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  try {
    for (const a of shot.setup) await run(page, a, spec.end_card);
    await sleep(150); // let the last scroll paint
    await showSyncMark(page, true);
    const tMark = Date.now();
    await sleep(300);
    await showSyncMark(page, false);
    await sleep(300); // the mark is gone well before the shot starts
    const tStart = Date.now();
    for (const a of shot.timeline) await run(page, a, spec.end_card);
    const left = shot.duration * 1000 - (Date.now() - tStart);
    if (left < -250) throw new Error(`the timeline ran ${(-left / 1000).toFixed(1)} s past the shot's ${shot.duration} s`);
    await sleep(Math.max(0, left) + 600);
    await page.screenshot({ path: path.join(outDir, `${shot.id}.png`) });
    const video = page.video();
    await context.close();
    const file = path.join(outDir, `${shot.id}.webm`);
    renameSync(await video.path(), file);
    rmSync(tmp, { recursive: true, force: true });
    // The trim is the sync mark's place in the clip plus the time from the mark to the shot's start;
    // without ffmpeg (or if the mark is not found) it falls back to the clock, which runs early by
    // however late the recording started.
    const markS = syncMarkSeconds(file);
    if (markS === null) console.log(`warn ${shot.id}: no sync mark found in the clip; the trim is from the clock`);
    const trim = markS !== null ? markS + (tStart - tMark) / 1000 : (tStart - t0) / 1000;
    manifest.clips.push({ id: shot.id, file: path.basename(file), trim_s: Number(trim.toFixed(3)), trim_from: markS !== null ? "sync mark" : "clock", duration_s: shot.duration, page: shot.page, variant: shot.variant ?? null, errors });
    console.log(`ok   ${shot.id} ${String(shot.page ?? "end card").padEnd(28)} trim ${trim.toFixed(2)} s, ${shot.duration} s${errors.length ? `, ${errors.length} page errors` : ""}`);
  } catch (err) {
    failed += 1;
    console.log(`FAIL ${shot.id} ${shot.page ?? "end card"}: ${err instanceof Error ? err.message : String(err)}`);
    await page.screenshot({ path: path.join(outDir, `${shot.id}-failed.png`) }).catch(() => {});
    await context.close().catch(() => {});
    rmSync(tmp, { recursive: true, force: true });
  }
}

manifest.clips.sort((a, b) => a.id.localeCompare(b.id));
writeFileSync(manifestFile, JSON.stringify(manifest, null, 1) + "\n");
await browser.close();
server?.close();
console.log(failed ? `${failed} shots failed` : `${shots.length} shots recorded in ${outDir}`);
process.exit(failed ? 1 : 0);
