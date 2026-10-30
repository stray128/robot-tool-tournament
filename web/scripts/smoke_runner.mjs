#!/usr/bin/env node
/**
 * Headless smoke test of the Studio's "Plan and run it live" panel against a tiny local mock of the
 * hosted runner (scripts/mock_runner.mjs). Build the site for it first:
 *
 *   NEXT_PUBLIC_BASE_PATH=/robot-tool-tournament NEXT_PUBLIC_RUNNER_URL=http://127.0.0.1:8765 pnpm build
 *   node scripts/smoke_runner.mjs [--base /robot-tool-tournament] [--port 4198] [--shots DIR]
 *
 * (http://localhost:8765 works as well: the mock is the same, and the report link is checked for the name the
 * build was given, not for one spelling of it.)
 *
 * It starts the mock on the port NEXT_PUBLIC_RUNNER_URL names (or --runner-port), serves web/out under
 * --base, and at 1440 x 900 and 375 x 812 drives the panel:
 *
 *   - idle; planning (progress and elapsed seconds); a valid plan loaded into the step editor;
 *     Run in physics, with the run id in the URL hash; running; the result card (recommended setup,
 *     "k of n runs", the best single tool, the step table, the report link, "Simulated, verify on
 *     hardware", the product-mode label); a reload that resumes the run without a new POST;
 *   - a run with no recommendation (the closest setup, a step with no tool); a failed run; a failed plan;
 *     a refused plan; a plan the runner would turn down (an uploaded part), whose Run button says why;
 *   - each error in plain words: 401, 429 (plans and runs), 503 (plan and run), 400 with the runner's
 *     problems, a dropped connection with Try again that works; and the panel's absence when /health
 *     fails, its off notice when the kill switch is on;
 *   - the key: it is only ever in the body of POST /plan (not a URL, header, storage, cookie, the console
 *     or the address bar), and the field is empty after the request.
 *
 * Fails on a console error nobody expected, a page error, horizontal scroll, a raw status code or a
 * brand name on screen. The panel's 10 s polling is stepped with the browser's fake clock. Screenshots of
 * the panel in every state go to --shots (default web/.smoke/runner).
 */

import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

import { startMockRunner } from "./mock_runner.mjs";
import { isReportRedirect, runnerUrlFromEnv } from "./runner_env.mjs";
import { startServer } from "./serve.mjs";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    base: { type: "string", default: process.env.NEXT_PUBLIC_BASE_PATH ?? "" },
    port: { type: "string", default: "4198" },
    "runner-port": { type: "string" },
    out: { type: "string", default: path.join(webDir, "out") },
    shots: { type: "string", default: path.join(webDir, ".smoke", "runner") },
  },
});
const base = values.base.replace(/\/$/, "");
const runnerRaw = runnerUrlFromEnv();
const runnerPort = Number(values["runner-port"] ?? (runnerRaw ? new URL(runnerRaw).port : 8765));

const KEY = "nvapi-smoke-SECRET-key-0123456789";
const TOTE = "Open the hinged lid of the returns tote, put the carton from the bench in it, close the lid.";
const RAW_CODE = /\b(400|401|403|404|413|422|429|500|502|503)\b/;
const PRODUCT = "Product mode: not part of the pre-registered experiment";
const PRIVACY = "Your key is sent once over HTTPS for this request and is never stored or logged.";

function brandPattern() {
  const src = readFileSync(path.join(webDir, "..", "src", "rtt", "effectors", "types.py"), "utf8");
  const block = src.match(/BRAND_TOKENS: Final\[tuple\[str, \.\.\.\]\] = \(([\s\S]*?)\n\)/);
  if (!block) throw new Error("BRAND_TOKENS not found in src/rtt/effectors/types.py");
  const alt = [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]).sort((a, b) => b.length - a.length).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`(?<![a-z0-9])(${alt.join("|")})(?![a-z0-9])`, "i");
}
const brandRe = brandPattern();

const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "phone", width: 375, height: 812 },
];

mkdirSync(values.shots, { recursive: true });
const mock = await startMockRunner({ port: runnerPort });
const server = await startServer({ dir: path.resolve(values.out), port: Number(values.port), base });
const origin = `http://127.0.0.1:${values.port}${base}`;
const browser = await chromium.launch();
const failures = [];

/** One page with everything a key could leak through watched. */
async function openPage(vp, { allow = [], fakeClock = false } = {}) {
  const context = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, colorScheme: "dark", reducedMotion: "reduce" });
  const page = await context.newPage();
  const watch = { errors: [], consoleText: [], requests: [], problems: [], allow };
  page.on("console", (m) => {
    watch.consoleText.push(m.text());
    if (m.type() === "error" && !allow.some((re) => re.test(m.text()))) watch.errors.push(m.text().slice(0, 200));
  });
  page.on("pageerror", (e) => watch.errors.push(`pageerror: ${String(e).slice(0, 200)}`));
  page.on("request", (r) => watch.requests.push({ method: r.method(), url: r.url(), headers: r.headers(), body: r.postData() ?? "" }));
  if (fakeClock) await page.clock.install();
  mock.state.requests.length = 0;
  return { context, page, watch };
}

const panelOf = (page) => page.locator("[data-hosted-runner]");

async function shot(vp, page, name, { full = false } = {}) {
  const file = path.join(values.shots, `${vp.name}-${name}.png`);
  if (full) {
    await page.screenshot({ path: file, fullPage: true });
    return;
  }
  // The sticky site header would cover the top of a tall panel in an element screenshot.
  await page.addStyleTag({ content: "header.sticky { position: static !important; }" });
  await panelOf(page).screenshot({ path: file });
}

/** Open the Studio and wait for the panel (it appears when /health answers). */
async function openStudio(page, watch, hash = "") {
  const res = await page.goto(`${origin}/studio/${hash}`, { waitUntil: "networkidle" });
  if (res?.status() !== 200) watch.problems.push(`Studio answered ${res?.status()}`);
}

async function chooseProvider(page, name) {
  await panelOf(page).getByRole("radio", { name }).check();
}

async function planLive(page, { key = KEY, text = TOTE } = {}) {
  await page.getByLabel("Use case", { exact: true }).fill(text);
  await chooseProvider(page, /Use my NVIDIA API key/);
  if (key !== null) await panelOf(page).locator("[data-hosted-key]").fill(key);
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
}

async function panelText(page) {
  return panelOf(page).innerText();
}

async function waitIn(page, selector, what, watch, timeout = 15000) {
  await panelOf(page)
    .locator(selector)
    .first()
    .waitFor({ timeout })
    .catch(() => watch.problems.push(`${what}: ${selector} never appeared`));
}

/** Layout, wording and brand checks of whatever the panel shows now. */
async function checkPanel(vp, page, watch, what) {
  const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  if (sw > iw) watch.problems.push(`${what}: horizontal scroll (${sw} > ${iw})`);
  const text = await panelText(page);
  const code = text.match(RAW_CODE);
  if (code) watch.problems.push(`${what}: a raw status code on screen (${code[1]})`);
  const brand = text.match(brandRe);
  if (brand) watch.problems.push(`${what}: names a brand (${brand[1]})`);
  if (/—/.test(text)) watch.problems.push(`${what}: an em dash`);
  return text;
}

function expectText(watch, text, needle, what) {
  if (!text.includes(needle)) watch.problems.push(`${what}: missing "${needle}"`);
}

/**
 * A plan that came back from the hosted planner loads into the step editor below the panel; the page brings the editor into
 * view (smoothly, but these contexts ask for reduced motion, so at once). Its heading must be on screen, in the upper part of
 * the viewport and below the sticky header, and the page must have scrolled to get it there.
 */
async function expectEditorInView(vp, page, watch, what, scrollBefore) {
  const read = () =>
    page.evaluate(() => {
      const el = document.querySelector('section[aria-labelledby="edit-title"]');
      return { top: el ? Math.round(el.getBoundingClientRect().top) : null, y: Math.round(window.scrollY), focused: el !== null && document.activeElement === el };
    });
  let now = await read();
  for (let i = 0; i < 40 && !(now.top !== null && now.top >= 0 && now.top < vp.height * 0.4); i++) {
    await page.waitForTimeout(100);
    now = await read();
  }
  if (now.top === null || now.top < 0 || now.top >= vp.height * 0.4) watch.problems.push(`${what}: the step editor was not brought into view (its top is at ${now.top}px in a ${vp.height}px viewport)`);
  else if (now.top < 56) watch.problems.push(`${what}: the step editor was scrolled under the sticky header (its top is at ${now.top}px)`);
  if (scrollBefore !== undefined && now.y === scrollBefore) watch.problems.push(`${what}: the page did not scroll when the plan loaded (scrollY stayed at ${now.y})`);
  if (!now.focused) watch.problems.push(`${what}: the step editor does not hold the focus`);
}

/** Every place the key could leak, after a scenario that used it. */
async function keyProof(page, watch, key, sends = 1) {
  const dump = await page.evaluate(() => ({
    local: JSON.stringify({ ...localStorage }),
    session: JSON.stringify({ ...sessionStorage }),
    cookie: document.cookie,
    href: location.href,
  }));
  for (const [where, value] of Object.entries(dump)) if (value.includes(key)) watch.problems.push(`the key is in ${where}`);
  for (const r of watch.requests) {
    const hasKey = (s) => s.includes(key) || s.includes(encodeURIComponent(key));
    const isPlan = r.method === "POST" && new URL(r.url).pathname === "/plan";
    if (hasKey(r.url)) watch.problems.push(`the key is in a URL: ${r.method} ${new URL(r.url).pathname}`);
    if (hasKey(JSON.stringify(r.headers))) watch.problems.push(`the key is in a header of ${r.method} ${new URL(r.url).pathname}`);
    if (hasKey(r.body) && !isPlan) watch.problems.push(`the key is in the body of ${r.method} ${new URL(r.url).pathname}`);
  }
  const sent = watch.requests.filter((r) => r.method === "POST" && new URL(r.url).pathname === "/plan" && r.body.includes(key));
  if (sent.length !== sends) watch.problems.push(`the key was sent in ${sent.length} POST /plan bodies, expected ${sends}`);
  if (watch.consoleText.some((t) => t.includes(key))) watch.problems.push("the key is in the console");
  const field = panelOf(page).locator("[data-hosted-key]");
  if ((await field.count()) && (await field.inputValue()) !== "") watch.problems.push("the key field still holds the key after the request");
}

async function finish(vp, name, ctx) {
  const { context, watch } = ctx;
  const unexpected = watch.errors.filter(Boolean);
  const problems = [...watch.problems, ...unexpected.map((e) => `console: ${e}`)];
  console.log(`${problems.length ? "FAIL" : "ok  "} ${vp.name.padEnd(7)} ${name}${problems.length ? `\n       ${problems.join("\n       ")}` : ""}`);
  if (problems.length) failures.push(`${vp.name} ${name}: ${problems.join("; ")}`);
  await context.close();
}

const resetMock = () => Object.assign(mock.state, { health: "ok", plan: "valid", run: "ok", poll: "ok", ending: "done", planDelayMs: 0, plansLeft: 3, runsLeft: 9 });

/**
 * The key field is a password field, so a screen reader is given bullets and not the typed key. It asks not to be filled, corrected,
 * spell-checked or read (autocomplete off, not new-password, which makes browsers offer to generate a password), carries the opt-out
 * markers, and has no name, id or form. It is not drawn as dots by a style: that exposes the typed key to assistive technology.
 */
async function checkKeyFieldAttributes(panel, watch) {
  const field = panel.locator("[data-hosted-key]");
  const attr = (name) => field.getAttribute(name);
  if ((await attr("type")) !== "password") watch.problems.push(`the key field's type is "${await attr("type")}", expected password`);
  if ((await attr("style")) !== null && /text-security/i.test((await attr("style")) ?? "")) watch.problems.push("the key field is drawn as dots by a style");
  for (const [name, expected] of [["autocomplete", "off"], ["autocorrect", "off"], ["autocapitalize", "off"], ["spellcheck", "false"], ["data-lpignore", "true"], ["data-form-type", "other"], ["data-gramm", "false"], ["data-gramm_editor", "false"], ["data-enable-grammarly", "false"], ["data-lt-active", "false"]]) {
    if ((await attr(name)) !== expected) watch.problems.push(`the key field's ${name} is "${await attr(name)}", expected ${expected}`);
  }
  for (const name of ["data-1p-ignore", "data-bwignore"]) if ((await attr(name)) === null) watch.problems.push(`the key field has no ${name}`);
  for (const name of ["name", "id", "form"]) if ((await attr(name)) !== null) watch.problems.push(`the key field has a ${name} attribute`);
  if (await field.evaluate((el) => el.form !== null)) watch.problems.push("the key field is inside a form");
}

// ------------------------------------------------------------------ scenarios

/** idle, plan, load, run, running, result, reload-resume, with the key proof. */
async function happyPath(vp) {
  resetMock();
  const ctx = await openPage(vp, { fakeClock: true });
  const { page, watch } = ctx;
  await openStudio(page, watch);
  await waitIn(page, "h2", "panel", watch, 20000);
  const idle = await checkPanel(vp, page, watch, "idle");
  for (const t of ["Plan and run it live", "Use my NVIDIA API key", "Use my Nebius Token Factory key", "Use the free allowance (3 free plans left today across the site)", "Forget the key after each request", PRIVACY, "Runs on Nebius Serverless Jobs, capped at $2 a day", "9 runs left today across the site"]) {
    expectText(watch, idle, t, "idle");
  }
  const field = panelOf(page).locator("[data-hosted-key]");
  await checkKeyFieldAttributes(panelOf(page), watch);
  if (!(await panelOf(page).getByLabel("Forget the key after each request").isChecked())) watch.problems.push("forget after use is not on by default");
  if (!(await panelOf(page).getByRole("button", { name: "Run in physics" }).isDisabled())) watch.problems.push("Run in physics is enabled before there is a plan");
  if (!(await panelOf(page).getByRole("button", { name: "Plan live" }).isDisabled())) watch.problems.push("Plan live is enabled with no description and no key");
  await shot(vp, page, "idle");

  // the free allowance needs no key
  await chooseProvider(page, /free allowance/);
  if ((await field.count()) !== 0) watch.problems.push("the free allowance still asks for a key");
  await chooseProvider(page, /Token Factory/);
  await chooseProvider(page, /NVIDIA/);

  // plan live, with the key
  mock.state.planDelayMs = 1500;
  await page.getByRole("button", { name: /Returns tote/ }).click();
  await panelOf(page).locator("[data-hosted-key]").fill(KEY);
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
  await waitIn(page, "[data-hosted-progress]:has-text('Nemotron is planning, usually 15–45 s')", "planning progress", watch, 5000);
  const planning = await checkPanel(vp, page, watch, "planning");
  expectText(watch, planning, "Nemotron is planning, usually 15–45 s. Planning can take up to five minutes.", "planning");
  if (!/\d+ s so far/.test(planning)) watch.problems.push("planning shows no elapsed seconds");
  await shot(vp, page, "planning");
  const scrollBeforePlan = await page.evaluate(() => Math.round(window.scrollY));
  await waitIn(page, '[data-hosted-plan-status="valid"]', "valid plan", watch);
  await expectEditorInView(vp, page, watch, "plan loaded", scrollBeforePlan);
  const loaded = await checkPanel(vp, page, watch, "plan loaded");
  expectText(watch, loaded, "VALID PLAN", "plan loaded");
  const main = await page.locator("main").innerText();
  expectText(watch, main, "Plan is valid", "the step editor");
  const steps = await page.locator('article[aria-label^="Step "]').count();
  if (steps !== 4) watch.problems.push(`the loaded plan has ${steps} steps in the editor, expected 4`);
  if (await panelOf(page).getByRole("button", { name: "Run in physics" }).isDisabled()) watch.problems.push("Run in physics is disabled for a valid plan");
  await shot(vp, page, "plan-loaded");
  await shot(vp, page, "plan-loaded-page", { full: true });

  // run in physics
  await panelOf(page).getByRole("button", { name: "Run in physics" }).click();
  await waitIn(page, '[data-run-status="queued"]', "queued", watch, 10000);
  const hash = new URL(page.url()).hash;
  if (!/^#run-[A-Za-z0-9_-]+$/.test(hash)) watch.problems.push(`the run id is not a bare token in the URL hash: "${hash}"`);
  if (!new URL(page.url()).pathname.startsWith(`${base}/studio/`)) watch.problems.push(`the run changed the page path: ${page.url()}`);
  const queued = await checkPanel(vp, page, watch, "queued");
  expectText(watch, queued, "QUEUED", "queued");
  expectText(watch, queued, "about 8 min 00 s expected", "queued");
  await shot(vp, page, "queued");
  await page.clock.fastForward(10_000);
  await waitIn(page, '[data-run-status="running"]', "running", watch, 10000);
  const running = await checkPanel(vp, page, watch, "running");
  expectText(watch, running, "RUNNING", "running");
  expectText(watch, running, "Runs on Nebius Serverless Jobs, capped at $2 a day", "running");
  await shot(vp, page, "running");
  await page.clock.fastForward(10_000);
  await waitIn(page, "[data-run-result]", "the result card", watch, 10000);
  const result = await checkPanel(vp, page, watch, "result");
  for (const t of ["Recommended setup", "Suction cup plus parallel jaw", "Two-tool head", "Suction cup, 40 mm", "7 of 8 runs", "Best single tool", "Parallel jaw, 90 mm stroke", "2 of 8 runs", "31 of 56 tools were ruled out before the full runs.", "Simulated, verify on hardware", PRODUCT, "Download the full report (JSON)", "Replays are shown for the recorded examples; hosted runs show the summary.", "Open lid", "Pick carton", "100%", "88%"]) {
    expectText(watch, result, t, "result");
  }
  const stepRows = await panelOf(page).locator("[data-run-steps] tbody tr").count();
  if (stepRows !== 4) watch.problems.push(`the step table has ${stepRows} rows, expected 4`);
  // the link goes through the runner's redirect (a presigned link would expire after 15 minutes)
  const report = await panelOf(page).locator("[data-run-report]").getAttribute("href");
  const runId = new URL(page.url()).hash.replace(/^#run-/, "");
  if (!isReportRedirect(report, runId, mock.port)) watch.problems.push(`the report link is ${report}`);
  else {
    // asked at the mock's own address, so the check does not depend on how this machine resolves "localhost"
    const res = await fetch(new URL(new URL(report).pathname, mock.url));
    if (res.status !== 200 || !res.redirected || (await res.json()).run_id !== runId) watch.problems.push("the report link does not lead to the report");
  }
  await shot(vp, page, "result");
  await keyProof(page, watch, KEY);
  const posts = mock.state.requests.filter((r) => r.method === "POST" && r.url === "/runs").length;
  if (posts !== 1) watch.problems.push(`${posts} POST /runs, expected 1`);

  // a reload resumes the run: the result is back, no new POST
  await page.reload({ waitUntil: "networkidle" });
  await waitIn(page, "[data-run-result]", "the resumed result card", watch, 20000);
  if (mock.state.requests.filter((r) => r.method === "POST" && r.url === "/runs").length !== 1) watch.problems.push("the reload posted the run again");
  if (new URL(page.url()).hash !== hash) watch.problems.push("the reload lost the run hash");
  const box = await panelOf(page).boundingBox();
  if (box && box.y > vp.height) watch.problems.push("the resumed run is below the fold; the panel was not brought into view");
  await shot(vp, page, "resumed");

  // clear: the hash goes, the page path stays
  await panelOf(page).getByRole("button", { name: "Clear this run" }).click();
  if (new URL(page.url()).hash !== "" || !new URL(page.url()).pathname.startsWith(`${base}/studio/`)) watch.problems.push(`clearing left ${page.url()}`);
  await finish(vp, "plan, run, result, resume", ctx);
}

/** The free allowance: no key sent; a failed run; a failed plan and a refusal. */
async function otherOutcomes(vp) {
  resetMock();
  const ctx = await openPage(vp, { fakeClock: true });
  const { page, watch } = ctx;
  await openStudio(page, watch);
  await waitIn(page, "h2", "panel", watch, 20000);

  // free allowance, valid plan, then a run that fails
  await page.getByLabel("Use case", { exact: true }).fill(TOTE);
  await chooseProvider(page, /free allowance/);
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
  await waitIn(page, '[data-hosted-plan-status="valid"]', "free plan", watch);
  await expectEditorInView(vp, page, watch, "free plan loaded");
  const body = JSON.parse(mock.state.requests.find((r) => r.url === "/plan")?.body ?? "{}");
  if (body.provider !== "free" || body.key !== null) watch.problems.push(`the free allowance sent ${JSON.stringify({ provider: body.provider, key: body.key })}`);
  await waitIn(page, "text=2 free plans left today across the site", "the plan count falling", watch, 5000);
  // a run in which nothing completed the task: no pick, the closest setup, a step with no tool
  mock.state.ending = "no_pick";
  await panelOf(page).getByRole("button", { name: "Run in physics" }).click();
  await waitIn(page, '[data-run-status="queued"]', "queued (no pick)", watch);
  await page.clock.fastForward(10_000);
  await waitIn(page, '[data-run-status="running"]', "running (no pick)", watch);
  await page.clock.fastForward(10_000);
  await waitIn(page, '[data-run-outcome="none"]', "the no-recommendation card", watch);
  const nopick = await checkPanel(vp, page, watch, "no recommendation");
  for (const t of ["No setup completed the task", "The closest setup was:", "0 of 8 runs", "No single-tool comparison came back for this run.", "No tool", "n/a", "Simulated, verify on hardware", "there is no recommendation"]) expectText(watch, nopick, t, "no recommendation");
  if (nopick.includes("Recommended setup")) watch.problems.push("a run with no pick still says Recommended setup");
  await shot(vp, page, "result-no-pick");
  await panelOf(page).getByRole("button", { name: "Clear this run" }).click();

  mock.state.ending = "failed";
  await panelOf(page).getByRole("button", { name: "Run in physics" }).click();
  await waitIn(page, '[data-run-status="queued"]', "queued", watch);
  await page.clock.fastForward(10_000);
  await waitIn(page, '[data-run-status="running"]', "running", watch);
  await page.clock.fastForward(10_000);
  await waitIn(page, "[data-run-failed]", "the failed run", watch);
  const failed = await checkPanel(vp, page, watch, "failed run");
  expectText(watch, failed, "The physics worker ran out of time on step 3.", "failed run");
  if ((await panelOf(page).locator("[data-run-result]").count()) !== 0) watch.problems.push("a failed run shows a result card");
  await shot(vp, page, "run-failed");

  // a failed plan: the problems in plain words, nothing loaded
  mock.state.plan = "failed";
  await panelOf(page).getByRole("button", { name: "Clear this run" }).click();
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
  await waitIn(page, '[data-hosted-plan-status="failed"]', "failed plan", watch);
  const noPlan = await checkPanel(vp, page, watch, "failed plan");
  expectText(watch, noPlan, "NO PLAN", "failed plan");
  expectText(watch, noPlan, "the tote lid hinge limit is above 120 degrees", "failed plan");
  await shot(vp, page, "plan-failed");

  // a refusal
  mock.state.plan = "refused";
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
  await waitIn(page, '[data-hosted-plan-status="refused"]', "refused plan", watch);
  const refused = await checkPanel(vp, page, watch, "refused plan");
  expectText(watch, refused, "REFUSED", "refused plan");
  expectText(watch, refused, "Folding cloth is not simulated.", "refused plan");
  await shot(vp, page, "plan-refused");
  await finish(vp, "free allowance, no recommendation, failed run, failed plan, refusal", ctx);
}

/** The key field: a password field whose typed key the accessibility tree shows as bullets, and that no storage, URL or console message holds. */
async function keyField(vp) {
  resetMock();
  const ctx = await openPage(vp);
  const { page, watch, context } = ctx;
  await openStudio(page, watch);
  await waitIn(page, "h2", "panel", watch, 20000);
  await checkKeyFieldAttributes(panelOf(page), watch);
  const field = panelOf(page).locator("[data-hosted-key]");
  await field.fill(KEY);
  if ((await field.inputValue()) !== KEY) watch.problems.push("the key field does not hold what was typed");
  // what a screen reader is given: the textbox with bullets for a value, and no part of the key anywhere in the tree
  const client = await context.newCDPSession(page);
  const { nodes } = await client.send("Accessibility.getFullAXTree");
  if (JSON.stringify(nodes).includes(KEY)) watch.problems.push("the accessibility tree holds the typed key");
  const box = nodes.find((n) => n.role?.value === "textbox" && /API key/i.test(n.name?.value ?? ""));
  if (!box) watch.problems.push("the accessibility tree has no textbox named for the key");
  else if (!/^[^A-Za-z0-9]+$/.test(String(box.value?.value ?? ""))) watch.problems.push(`the accessibility tree gives the key field the value "${String(box.value?.value ?? "").slice(0, 12)}", expected bullets`);
  await shot(vp, page, "key-field");
  // and nothing puts the key anywhere else
  const stored = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage }, cookie: document.cookie, href: location.href }));
  if (stored.includes(KEY)) watch.problems.push("the key is in storage, a cookie or the URL after typing it");
  if (watch.consoleText.some((t) => t.includes(KEY))) watch.problems.push("the key is in the console after typing it");
  await finish(vp, "the key field is a password field, shown to assistive technology as bullets", ctx);
}

/** A recorded plan that uses an uploaded part: the runner would refuse it, so Run in physics is off and says why. */
async function limitReason(vp) {
  resetMock();
  const ctx = await openPage(vp);
  const { page, watch } = ctx;
  await openStudio(page, watch);
  await waitIn(page, "h2", "panel", watch, 20000);
  await page.locator("select").first().selectOption("x1_upload_tub");
  await page.getByRole("button", { name: "Plan", exact: true }).click();
  await page.locator("main").getByText("Plan is valid").first().waitFor({ timeout: 15000 }).catch(() => watch.problems.push("the upload example never became a valid plan"));
  if (!(await panelOf(page).getByRole("button", { name: "Run in physics" }).isDisabled())) watch.problems.push("Run in physics is on for a plan that names an uploaded part");
  expectText(watch, await checkPanel(vp, page, watch, "limit"), "The hosted runner has no part upload, so it cannot run a plan that uses one.", "limit");
  await shot(vp, page, "run-limit");
  await finish(vp, "a plan the runner would turn down", ctx);
}

/** Each plan error, in plain words. */
async function planErrors(vp) {
  const cases = [
    { name: "401", set: { plan: "valid" }, key: "bad-key", allow: /status of 401/, kind: "key_rejected", text: "The provider rejected that key." },
    { name: "429", set: { plan: "429" }, allow: /status of 429/, kind: "cap", text: "You have used your free plans for today. It resets at 29 October 2026, 00:00 UTC." },
    { name: "503", set: { plan: "503" }, allow: /status of 503/, kind: "off", text: "The hosted runner is off. Run it on your own machine:" },
    // a 429 with no cap is the hosting layer throttling a burst: busy, not a daily limit
    { name: "busy", set: { plan: "busy" }, allow: /status of 429/, kind: "busy", text: "The runner is busy. Try again in a minute.", retry: true },
    { name: "network", set: { plan: "drop" }, allow: /Failed to load resource|ERR_/, kind: "network", text: "Could not reach the hosted runner. Check your connection and try again.", retry: true },
  ];
  for (const c of cases) {
    resetMock();
    Object.assign(mock.state, c.set);
    const ctx = await openPage(vp, { allow: [c.allow] });
    const { page, watch } = ctx;
    await openStudio(page, watch);
    await waitIn(page, "h2", "panel", watch, 20000);
    const key = c.key ?? KEY;
    await planLive(page, { key });
    await waitIn(page, `[data-hosted-error="${c.kind}"]`, `the ${c.name} message`, watch);
    const text = await checkPanel(vp, page, watch, `plan error ${c.name}`);
    expectText(watch, text, c.text, `plan error ${c.name}`);
    if (c.kind === "off") {
      const href = await panelOf(page).getByRole("link", { name: "Run it yourself" }).getAttribute("href");
      if (href !== `${base}/docs/tutorials/run-it-yourself/`) watch.problems.push(`the run-it-yourself link is ${href}`);
    }
    if (c.name === "busy" && /daily limit/i.test(text)) watch.problems.push("a 429 with no cap says a daily limit was reached");
    if (c.retry) {
      const retry = panelOf(page).getByRole("button", { name: "Try again" });
      if (!(await retry.isVisible())) watch.problems.push(`a ${c.name} failure shows no Try again`);
      await shot(vp, page, `error-${c.name}`);
      // the key was forgotten with the failed request: type it again, the server is back, Try again works
      mock.state.plan = "valid";
      await panelOf(page).locator("[data-hosted-key]").fill(KEY);
      await retry.click();
      await waitIn(page, '[data-hosted-plan-status="valid"]', "the plan after Try again", watch);
    } else {
      await shot(vp, page, `error-${c.name}`);
    }
    // one POST /plan per press of Plan live or Try again, each with its own typed key
    await keyProof(page, watch, key, c.retry ? 2 : 1);
    await finish(vp, `plan error: ${c.name}`, ctx);
  }
}

/** Each run error, from a valid plan. */
async function runErrors(vp) {
  resetMock();
  const ctx = await openPage(vp, { allow: [/status of 429/, /status of 503/, /Failed to load resource|ERR_/] });
  const { page, watch } = ctx;
  await openStudio(page, watch);
  await waitIn(page, "h2", "panel", watch, 20000);
  await page.getByLabel("Use case", { exact: true }).fill(TOTE);
  await chooseProvider(page, /free allowance/);
  await panelOf(page).getByRole("button", { name: "Plan live" }).click();
  await waitIn(page, '[data-hosted-plan-status="valid"]', "plan", watch);
  const run = panelOf(page).getByRole("button", { name: "Run in physics" });

  mock.state.run = "429";
  await run.click();
  await waitIn(page, '[data-hosted-error="cap"]', "the runs cap message", watch);
  expectText(watch, await checkPanel(vp, page, watch, "run error 429"), "You have reached the daily limit on physics runs. It resets at 29 October 2026, 00:00 UTC.", "run error 429");
  await shot(vp, page, "error-run-429");

  mock.state.run = "busy";
  await run.click();
  await waitIn(page, '[data-hosted-error="busy"]', "the busy message", watch);
  const busy = await checkPanel(vp, page, watch, "run error busy");
  expectText(watch, busy, "The runner is busy. Try again in a minute.", "run error busy");
  if (/daily limit/i.test(busy)) watch.problems.push("a 429 with no cap on a run says a daily limit was reached");
  await shot(vp, page, "error-run-busy");

  mock.state.run = "400";
  await run.click();
  await waitIn(page, '[data-hosted-error="bad_request"]', "the plan problems message", watch);
  const problems = await checkPanel(vp, page, watch, "run error 400");
  expectText(watch, problems, "The runner checked the plan again and found problems:", "run error 400");
  expectText(watch, problems, "steps: a hosted run takes at most 8 steps", "run error 400");
  await shot(vp, page, "error-run-400");

  mock.state.run = "503";
  await run.click();
  await waitIn(page, '[data-hosted-error="off"]', "the off message", watch);
  expectText(watch, await checkPanel(vp, page, watch, "run error 503"), "The hosted runner is off. Run it on your own machine:", "run error 503");
  await shot(vp, page, "error-run-503");

  mock.state.run = "drop";
  await run.click();
  await waitIn(page, '[data-hosted-error="network"]', "the network message", watch);
  await checkPanel(vp, page, watch, "run error network");
  await shot(vp, page, "error-run-network");
  mock.state.run = "ok";
  await panelOf(page).getByRole("button", { name: "Try again" }).click();
  await waitIn(page, '[data-run-status="queued"]', "the run after Try again", watch);
  await finish(vp, "run errors: 429, busy, 400 with problems, 503, network with Try again", ctx);
}

/** The panel follows /health. */
async function healthStates(vp) {
  // The runner answers 200 with ok: true after its auto-off date. `open: false` ("closed"), or an auto-off date in the past from an
  // older runner with no `open` field ("lapsed"), shows the off notice; an older runner with a future date ("legacy") is on.
  for (const [state, allow, expectPanel] of [
    ["bad", [], false],
    ["down", [/status of 503/], false],
    ["drop", [/Failed to load resource|ERR_/], false],
    ["off", [], true],
    ["closed", [], true],
    ["lapsed", [], true],
    ["legacy", [], true],
  ]) {
    resetMock();
    mock.state.health = state;
    const ctx = await openPage(vp, { allow });
    const { page, watch } = ctx;
    await openStudio(page, watch, "#plan");
    await page.waitForTimeout(600);
    const count = await panelOf(page).count();
    if (expectPanel !== (count > 0)) watch.problems.push(`health ${state}: the panel is ${count ? "shown" : "hidden"}, expected ${expectPanel ? "shown" : "hidden"}`);
    const [sw, iw] = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    if (sw > iw) watch.problems.push(`health ${state}: horizontal scroll`);
    if (!(await page.locator("main").innerText()).includes("Describe the use case")) watch.problems.push(`health ${state}: the planner itself is gone`);
    if (state === "off" || state === "closed" || state === "lapsed") {
      const text = await checkPanel(vp, page, watch, state);
      expectText(watch, text, "The hosted runner is off. Run it on your own machine:", state);
      if ((await panelOf(page).getByRole("button", { name: "Plan live" }).count()) !== 0) watch.problems.push(`the ${state} panel still offers Plan live`);
      if ((await panelOf(page).getByRole("button", { name: "Run in physics" }).count()) !== 0) watch.problems.push(`the ${state} panel still offers Run in physics`);
      if (state === "lapsed") expectText(watch, text, "switched itself off on 1 January 2026", state);
      await shot(vp, page, state === "off" ? "off" : `off-${state}`);
    } else if (state === "legacy") {
      const text = await checkPanel(vp, page, watch, state);
      expectText(watch, text, "Plan live", state);
      if (text.includes("The hosted runner is off")) watch.problems.push("an older runner with a future auto-off date shows the off notice");
    } else if (count === 0) {
      const text = await page.locator("main").innerText();
      for (const t of ["Plan and run it live", "Run in physics"]) if (text.includes(t)) watch.problems.push(`health ${state}: "${t}" is on the page`);
    }
    await finish(vp, `health ${state}`, ctx);
  }
}

try {
  for (const vp of viewports) {
    await happyPath(vp);
    await otherOutcomes(vp);
    await limitReason(vp);
    await keyField(vp);
    await planErrors(vp);
    await runErrors(vp);
    await healthStates(vp);
  }
} finally {
  await browser.close();
  server.close();
  await mock.close();
}

if (failures.length) {
  console.error(`\n${failures.length} scenario(s) failed`);
  process.exit(1);
}
console.log(`runner smoke: every scenario passes at ${viewports.length} widths; screenshots in ${values.shots}`);
