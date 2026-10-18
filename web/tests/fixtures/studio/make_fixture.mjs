#!/usr/bin/env node
/**
 * Regenerate the Studio test fixture (web/tests/studio.test.ts), following the S7 data contract.
 *
 *   node tests/fixtures/studio/make_fixture.mjs
 *
 * Output, next to this script:
 *   data/studio/index.json                 two heroes: f1 (recommended), f2 (no recommendation)
 *   data/studio/f1.json, f2.json           hero pages
 *   data/studio/replays/f1/<lane>.{json,bin}  replay v2 lanes, poses from the replay v2 sample
 *   data/studio/f1/build_pack.zip          a small stored zip
 *   data/studio/sessions/f1.json           a recorded session (rtt-agent-session)
 *   site/assets/<id>.glb                   the GLBs the lanes draw (copied from the replay v2 sample)
 *
 * The poses and GLBs are the recorded M1 sample's (tests/fixtures/replay_v2); everything else is
 * made up for the tests. The fixture is never published on the site.
 */

import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const sampleDir = path.join(here, "..", "replay_v2");
const sample = JSON.parse(readFileSync(path.join(sampleDir, "sample.json"), "utf8"));
const LABEL = "Product mode: not part of the pre-registered experiment";
const data = path.join(here, "data");
const studio = path.join(data, "studio");
const site = path.join(here, "site");

const sha = (s) => createHash("sha256").update(s).digest("hex");
const id = (name) => sha(`fixture:${name}`);

rmSync(data, { recursive: true, force: true });
rmSync(site, { recursive: true, force: true });
for (const d of [path.join(studio, "replays", "f1"), path.join(studio, "f1"), path.join(studio, "sessions"), path.join(site, "assets")]) {
  mkdirSync(d, { recursive: true });
}
for (const a of Object.keys(sample.assets)) copyFileSync(path.join(sampleDir, "assets", `${a}.glb`), path.join(site, "assets", `${a}.glb`));
const assetBytes = Object.values(sample.assets).reduce((s, a) => s + a.bytes, 0);

const json = (file, doc) => writeFileSync(file, `${JSON.stringify(doc, null, 1)}\n`);

// ---------------------------------------------------------------- setups

const TASK = id("task-f1");
const TASK2 = id("task-f2");
const GRIP = { effector_id: id("grip"), display_name: "Two-finger parallel gripper, flat pads", family: "catalog_gripper", family_title: "Industrial gripper" };
const CUP = { effector_id: id("cup"), display_name: "Suction cup, 25 mm bellows", family: "suction", family_title: "Suction cup" };
const CUP_CAVEAT = "Suction cup (approximation): the seal is a geometric test. Treat suction results as a screen, then verify on hardware.";
const TWIST_CAVEAT = "Threads are an approximation: a helical joint with a torque ramp.";

const REC_ID = id("setup-rec");
const BEST_ID = id("setup-best");
const recBrief = {
  setup_id: REC_ID,
  kind: "tool_change",
  kind_title: "Tool change",
  label: "Tool change: Two-finger > Suction cup",
  tools: [
    { index: 0, ...GRIP, fidelity: "rigid", off_the_shelf: true, caveat: null },
    { index: 1, ...CUP, fidelity: "approximation", off_the_shelf: true, caveat: CUP_CAVEAT },
  ],
  completion_rate: 0.75,
  completed_runs: 6,
  n_runs: 8,
  fidelity: "approximation",
  stops_at: null,
  cycle_time_s: 6.03,
};
const bestBrief = {
  setup_id: BEST_ID,
  kind: "single",
  kind_title: "Single tool",
  label: GRIP.display_name,
  tools: [GRIP.display_name],
  completion_rate: 0,
  completed_runs: 0,
  n_runs: 8,
  fidelity: "approximation",
  stops_at: "twist_cap",
  cycle_time_s: null,
};
const finalist = (brief, extra) => ({
  ...brief,
  tools: brief.tools.map((t) => (typeof t === "string" ? t : t.display_name)),
  eligible: brief.completed_runs > 0,
  reason: brief.completed_runs > 0 ? null : "never completed the task (stops at twist_cap in 8/8 runs: slip 8x)",
  ...extra,
});
const recRow = finalist(recBrief, {
  roles: ["recommended"],
  tool_details: recBrief.tools.map(({ caveat, ...t }) => t),
  step_success: { pick_part: 1, twist_cap: 0.75 },
  weakest_step: { step_id: "twist_cap", success_rate: 0.75 },
  complexity: { n_tools: 2, n_actuators: 2, total_mass_kg: 0.71, n_changes: 1, bracket: false },
  payload_kg: 0.71,
  failure_modes: { slip: 2 },
  stops_at_detail: { twist_cap: { runs: 2, failure_modes: { slip: 2 } } },
});
const bestRow = finalist(bestBrief, {
  roles: ["best_single"],
  tool_details: [{ index: 0, ...GRIP, fidelity: "core", off_the_shelf: true }],
  step_success: { pick_part: 1, twist_cap: 0 },
  weakest_step: { step_id: "twist_cap", success_rate: 0 },
  complexity: { n_tools: 1, n_actuators: 1, total_mass_kg: 0.41, n_changes: 0, bracket: false },
  payload_kg: 0.41,
  failure_modes: { slip: 8 },
  stops_at_detail: { twist_cap: { runs: 8, failure_modes: { slip: 8 } } },
});

// ---------------------------------------------------------------- plan, screen, matrix

const plan = (task, title) => ({
  task_id: task,
  title,
  description: "Pick a jar from the bench and twist its cap a quarter turn.",
  source: "recorded",
  result: "valid_first_call",
  calls: 1,
  model: "fixture-model",
  layout_moves: [],
  assumptions: ["The jar is rigid."],
  clarifications: [{ question: "How tight is the cap?", assumed: "0.5 N m" }],
  unsupported: [],
  objects: [
    { name: "jar", kind: "part", kind_title: "Rigid part", label: "Jar", params: { width_mm: 60, shape: "cylinder", fixed: false }, pose: { x_mm: 0, y_mm: 0, yaw_deg: 0 }, ref: null },
    { name: "cap", kind: "threaded_joint", kind_title: "Threaded joint", label: "Cap", params: { turns: 0.25 }, pose: { x_mm: 0, y_mm: 0, yaw_deg: 0 }, ref: "jar" },
  ],
  steps: [
    {
      id: "pick_part",
      primitive: "grasp",
      primitive_title: "Grasp",
      object: "jar",
      handle: "part",
      target: null,
      params: { lift_mm: 50, speed_mm_s: 80 },
      constraints: { max_force_n: 20, max_torque_nm: null, precision_mm: null, precision_deg: null },
      success: [{ quantity: "lift_mm", unit: "mm", at_least: 40, at_most: null, text: "lift at least 40 mm" }],
      after: [],
      hold_after: true,
      description: "Pick the jar.",
      fidelity: "rigid",
      caveats: [],
    },
    {
      id: "twist_cap",
      primitive: "twist",
      primitive_title: "Twist",
      object: "cap",
      handle: "head",
      target: null,
      params: { turns: 0.25 },
      constraints: { max_force_n: null, max_torque_nm: 2, precision_mm: null, precision_deg: null },
      success: [{ quantity: "torque_nm", unit: "N m", at_least: 0.5, at_most: 2, text: "torque 0.5 to 2 N m" }],
      after: ["pick_part"],
      hold_after: false,
      description: "Twist the cap a quarter turn.",
      fidelity: "approximation",
      caveats: [TWIST_CAVEAT],
    },
  ],
});

const screen = {
  n_candidates: 3,
  n_screened: 2,
  n_conditions: 3,
  runs: 12,
  wall_s: 4.2,
  candidates: [
    { ...GRIP, source: "library", off_the_shelf: true, mechanism: "fingers", fidelity: "rigid", built_for: null, status: "screened", pruned_reason: null, pruned_steps: [] },
    {
      ...CUP,
      source: "preset",
      off_the_shelf: true,
      mechanism: "suction",
      fidelity: "approximation",
      built_for: null,
      status: "screened",
      pruned_reason: null,
      pruned_steps: [{ step_id: "pick_part", reason: "the jar is narrower than the cup" }],
    },
    {
      effector_id: id("arm"),
      display_name: "Soft arm, 12 segments, 140 mm",
      family: "soft_arm",
      family_title: "Soft arm",
      source: "preset",
      off_the_shelf: true,
      mechanism: "soft_arm",
      fidelity: "approximation",
      built_for: null,
      status: "pruned",
      pruned_reason: null,
      pruned_steps: [
        { step_id: "pick_part", reason: "the part is wider than the soft arm reaches around" },
        { step_id: "twist_cap", reason: "a slack soft arm cannot twist" },
      ],
    },
  ],
  pruned_reasons: [
    { reason: "The part is wider than the soft arm reaches around", count: 1 },
    { reason: "A slack soft arm cannot twist", count: 1 },
    { reason: "The jar is narrower than the cup", count: 1 },
  ],
};

const matrix = {
  steps: ["pick_part", "twist_cap"],
  rows: [
    {
      effector_id: CUP.effector_id,
      display_name: CUP.display_name,
      family_title: CUP.family_title,
      fidelity: "approximation",
      off_the_shelf: true,
      cells: {
        pick_part: { state: "pruned", reason: "the jar is narrower than the cup" },
        twist_cap: { state: "run", rate: 1, n: 3, successes: 3, primitive_used: "twist", failure_modes: {}, context: "carried" },
      },
    },
    {
      effector_id: GRIP.effector_id,
      display_name: GRIP.display_name,
      family_title: GRIP.family_title,
      fidelity: "rigid",
      off_the_shelf: true,
      cells: {
        pick_part: { state: "run", rate: 1, n: 3, successes: 3, primitive_used: "grasp", failure_modes: {}, context: "start" },
        twist_cap: { state: "run", rate: 0.333, n: 3, successes: 1, primitive_used: "twist", failure_modes: { slip: 2 }, context: "carried" },
      },
    },
  ],
};

// ---------------------------------------------------------------- lanes

const DURATION = (sample.n_frames - 1) / sample.fps;

function laneFiles(name, setupId, kind, traj) {
  const header = structuredClone(sample);
  header.bin = `${name}.bin`;
  delete header.overlays;
  header.meta = {
    design_space: `studio-${kind}`,
    design_id: setupId,
    geometry_id: null,
    part_id: "f1",
    label: LABEL,
    lane: name,
    task_id: TASK,
    traj_sha: traj,
    tool_stand: "display only: a tool that is not on the wrist rests here; nothing about it is simulated",
  };
  header.provenance = { ...header.provenance, run_id: `studio-f1-${setupId.slice(0, 12)}`, product_mode: true, code_version: "ee0fdec3a1b2" };
  const headerFile = path.join(studio, "replays", "f1", `${name}.json`);
  writeFileSync(headerFile, JSON.stringify(header));
  copyFileSync(path.join(sampleDir, "sample.bin"), path.join(studio, "replays", "f1", `${name}.bin`));
  return statSync(headerFile).size + statSync(path.join(studio, "replays", "f1", `${name}.bin`)).size;
}

const lane = (laneId, title, brief, completed, segments, steps, credits = []) => {
  const traj = id(`traj-${laneId}`);
  return {
    id: laneId,
    title,
    setup_id: brief.setup_id,
    kind: brief.kind,
    label: brief.label,
    replay: `studio/replays/f1/${laneId}`,
    condition_id: "nominal",
    seed: 0,
    condition_text: "nominal placement, seed 0",
    traj_sha: traj,
    completed,
    cycle_time_s: completed ? 6.03 : 5.9,
    fps: sample.fps,
    n_frames: sample.n_frames,
    duration_s: Number(DURATION.toFixed(3)),
    bytes: laneFiles(laneId, brief.setup_id, brief.kind, traj),
    asset_bytes: assetBytes,
    segments,
    steps,
    credits,
  };
};

const lanes = [
  lane(
    "recommended",
    "Recommended",
    recBrief,
    true,
    [
      { index: 0, tool: 0, tool_label: GRIP.display_name, start_s: 0, end_s: 2.5, change: "none", change_s: 0, change_start_s: 0, steps: ["pick_part"] },
      { index: 1, tool: 1, tool_label: CUP.display_name, start_s: 3.5, end_s: 6.03, change: "tool_change", change_s: 1, change_start_s: 2.5, steps: ["twist_cap"] },
    ],
    [
      { step_id: "pick_part", primitive: "grasp", tool: 0, start_s: 0, duration_s: 2.5, success: true, failure_mode: null, fidelity: "rigid", quantities: { lift_mm: 52 } },
      { step_id: "twist_cap", primitive: "twist", tool: 1, start_s: 3.5, duration_s: 2.5, success: true, failure_mode: null, fidelity: "approximation", quantities: { torque_nm: 0.8 } },
    ],
    [{ text: "Simulation model of an open-source industrial gripper, BSD-3-Clause, modified", anchor: "library" }],
  ),
  lane(
    "best_single",
    "Best single off-the-shelf",
    bestBrief,
    false,
    [{ index: 0, tool: 0, tool_label: GRIP.display_name, start_s: 0, end_s: 5.9, change: "none", change_s: 0, change_start_s: 0, steps: ["pick_part", "twist_cap"] }],
    [
      { step_id: "pick_part", primitive: "grasp", tool: 0, start_s: 0, duration_s: 2.5, success: true, failure_mode: null, fidelity: "rigid", quantities: {} },
      { step_id: "twist_cap", primitive: "twist", tool: 0, start_s: 2.5, duration_s: 3.4, success: false, failure_mode: "slip", fidelity: "approximation", quantities: { torque_nm: 0.2 } },
    ],
  ),
];

// ---------------------------------------------------------------- build pack (a stored zip)

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function storedZip(files) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const [name, text] of files) {
    const body = Buffer.from(text);
    const fname = Buffer.from(name);
    const crc = crc32(body);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt32LE(0x00210000, 10); // 1980-01-01 00:00
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(body.length, 22);
    local.writeUInt16LE(fname.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(0x00210000, 12);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(body.length, 20);
    central.writeUInt32LE(body.length, 24);
    central.writeUInt16LE(fname.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, fname, body);
    centrals.push(central, fname);
    offset += local.length + fname.length + body.length;
  }
  const dir = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(dir.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, dir, end]);
}
const packFiles = [
  ["README.txt", "Fixture build pack for the Studio tests.\n"],
  ["bom.csv", "item,qty\nchanger plate,2\n"],
];
const zip = storedZip(packFiles);
writeFileSync(path.join(studio, "f1", "build_pack.zip"), zip);
const buildPack = {
  zip: "studio/f1/build_pack.zip",
  bytes: zip.length,
  sha256: sha(zip),
  setup_id: REC_ID,
  label: recBrief.label,
  for: "recommended",
  files: packFiles.map(([p, t]) => ({ path: p, kind: p.endsWith(".csv") ? "bom" : "notes", format: p.split(".").pop(), license: "CC0-1.0", bytes: Buffer.byteLength(t) })),
  links: [{ name: "Tool 0: Two-finger parallel gripper, flat pads", url: "https://example.org/upstream/gripper", license: "MIT", reason: "the tool's own CAD is upstream" }],
  caveats: [CUP_CAVEAT],
};

// ---------------------------------------------------------------- session

const t0 = Date.parse("2026-10-18T12:00:00Z");
const ts = (s) => new Date(t0 + s * 1000).toISOString();
const events = [
  { seq: 1, ts: ts(0), t_s: 0, type: "session.started", session_id: "studio-f1-fixture", models: { agent: "fixture-model" } },
  { seq: 2, ts: ts(1), t_s: 1, type: "agent.message", content: "I will plan the task, then run the tournament." },
  { seq: 3, ts: ts(1.5), t_s: 1.5, type: "tool.call", call_id: "c1", name: "plan_task", args: { text: "Pick a jar and twist its cap.", task_id: TASK } },
  { seq: 4, ts: ts(3), t_s: 3, type: "tool.result", call_id: "c1", name: "plan_task", ok: true, result: { steps: 2, result: "valid_first_call", url: "https://example.org/never-shown" } },
  { seq: 5, ts: ts(4), t_s: 4, type: "tool.call", call_id: "c2", name: "run_task_tournament", args: { task_id: TASK, conditions: 8 } },
  { seq: 6, ts: ts(60), t_s: 60, type: "tool.result", call_id: "c2", name: "run_task_tournament", ok: true, result: { recommended: recBrief.label, completion: 0.75, finalists: [1, 2] } },
  { seq: 7, ts: ts(61), t_s: 61, type: "session.finished", status: "completed", answer: "Use the tool change: 6 of 8 runs.", turns: 3, tool_calls: 2, spend: { spent_usd: 0.004 } },
];
json(path.join(studio, "sessions", "f1.json"), {
  format: "rtt-agent-session",
  version: 1,
  session_id: "studio-f1-fixture",
  label: LABEL,
  kind: "recorded",
  events,
});

// ---------------------------------------------------------------- pages

const common = (hero, caseId, title, setting, useCase, outcome, headline) => ({
  schema: "rtt-web-studio-hero",
  version: 1,
  label: LABEL,
  contract: "rtt-effectors/2",
  hero,
  case_id: caseId,
  title,
  setting,
  use_case: useCase,
  outcome,
  headline,
});
const constants = [
  { key: "tool_change_s", value: 6, unit: "s", note: "automatic tool change time", tag: "assumed" },
  { key: "cycle_margin", value: 0.1, unit: "fraction", note: "a simpler setup may be this much slower", tag: "assumed" },
];
const provenance = (task) => ({
  code_version: "ee0fdec3a1b2",
  report_sha256: id("report"),
  task_id: task,
  controllers: "default",
  evidence: "accepted",
  conditions: [
    { condition_id: "nominal", seed: 0, aim_offsets_mm: {} },
    { condition_id: "place01", seed: 1, aim_offsets_mm: { jar: [1.2, -0.4] } },
  ],
  fairness_sha: id("fairness"),
  workers: 2,
  timings: { screen_s: 4.2, total_s: 9.8 },
  determinism: [{ condition: "nominal", same: true }],
});
const sensitivity = { recommendation_changes: false, order_changes: 0, n_pairs: 9, tool_change_s: [3, 6, 12], present_s: [0.75, 1.5, 3], note: "change times are a fixed clock advance" };

const F1_HEADLINE = "A tool-change sequence completes the task in 6 of 8 runs; the best single off-the-shelf tool in 0 of 8 runs.";
const f1 = {
  ...common("f1", "f1_jar", "Jar cap", "lab", "Pick a jar from the bench and twist its cap a quarter turn.", "recommended", F1_HEADLINE),
  plan: plan(TASK, "Jar cap quarter turn"),
  screen,
  matrix,
  combinations: { counts: { single: 2, tool_change: 2, multi_head: 0, multi_head_geometry_searched: 1 }, rejected: [{ label: "", reason: "no bracket layout clears" }] },
  finalists: [recRow, bestRow],
  recommendation: {
    ...recBrief,
    why: "Highest full-task completion of the 1 eligible setup (75%).",
    caveat: "approximation: verify on hardware",
    verify_on_hardware: true,
    geometry: { changer: { stack_mm: 25, changeover_s: 6 } },
    steps: [
      { step_id: "pick_part", primitive: "grasp", primitive_used: "grasp", tool: 0, tool_label: GRIP.display_name, success_rate: 1, fidelity: "rigid", caveats: [], verify_on_hardware: false },
      { step_id: "twist_cap", primitive: "twist", primitive_used: "twist", tool: 1, tool_label: CUP.display_name, success_rate: 0.75, fidelity: "approximation", caveats: [TWIST_CAVEAT, CUP_CAVEAT], verify_on_hardware: true },
    ],
    runner_up_rigid: { label: null, basis: "no rigid-tier setup is eligible" },
    versus_best_single: { same_setup: false, completion_delta: 0.75, cycle_delta_s: null, extra_tools: 1 },
  },
  best_single: bestRow,
  no_winner: null,
  sensitivity,
  constants,
  fixed_caveats: ["A tool change is a fixed time, not a simulated docking."],
  lanes,
  build_pack: buildPack,
  build_pack_note: null,
  session: { file: "studio/sessions/f1.json", session_id: "studio-f1-fixture", status: "completed", model: "fixture-model", turns: 3, tool_calls: 2, spend_usd: 0.004, answer: "Use the tool change: 6 of 8 runs." },
  provenance: provenance(TASK),
};

const closestRow = { ...bestRow, setup_id: id("setup-closest"), label: "Suction cup, 25 mm bellows", tools: [CUP.display_name], roles: ["closest"] };
const F2_HEADLINE = "No setup completes the task; the closest setup stops at step 2, twist the cap.";
const f2 = {
  ...common("f2", "f2_jar", "Tight jar cap", "lab", "Pick a jar and turn its tight cap open.", "no_recommendation", F2_HEADLINE),
  plan: plan(TASK2, "Tight jar cap"),
  screen,
  matrix,
  combinations: { counts: { single: 2, tool_change: 2 }, rejected: [] },
  finalists: [closestRow, bestRow],
  recommendation: null,
  best_single: bestRow,
  no_winner: {
    reason: "no setup completed the task on the condition grid (nothing is forced)",
    closest: [closestRow],
    blocker: {
      step_id: "twist_cap",
      primitive: "twist",
      quantity: "torque_nm",
      unit: "N m",
      at_least: 3,
      at_most: 8,
      text: "Every tool slips before the cap reaches 3 N m.",
      conditions: ["nominal", "place01"],
      families: [
        { family: "catalog_gripper", family_title: "Industrial gripper", n_candidates: 1, n_runs: 3, peak: 1.15, best: GRIP.display_name, failure_modes: { slip: 3 } },
        { family: "suction", family_title: "Suction cup", n_candidates: 1, n_runs: 3, peak: 0.09, best: CUP.display_name, failure_modes: { slip: 3 } },
      ],
      max_peak: 1.15,
    },
  },
  sensitivity: null,
  constants,
  fixed_caveats: ["A tool change is a fixed time, not a simulated docking."],
  lanes: [],
  build_pack: null,
  build_pack_note: "No setup was run on the full grid, so there is nothing to pack.",
  session: null,
  provenance: provenance(TASK2),
};

const strip = (brief) => (brief ? { ...brief } : null);
const entry = (page, recommended, best, closest) => ({
  hero: page.hero,
  case_id: page.case_id,
  title: page.title,
  setting: page.setting,
  use_case: page.use_case,
  page: `studio/${page.hero}.json`,
  task_id: page.plan.task_id,
  n_steps: page.plan.steps.length,
  n_candidates: page.screen.n_candidates,
  outcome: page.outcome,
  headline: page.headline,
  recommended: strip(recommended),
  best_single: strip(best),
  closest: strip(closest),
});

json(path.join(studio, "f1.json"), f1);
json(path.join(studio, "f2.json"), f2);
json(path.join(studio, "index.json"), {
  schema: "rtt-web-studio",
  version: 1,
  label: LABEL,
  contract: "rtt-effectors/2",
  code_version: "ee0fdec3a1b2",
  heroes: [entry(f1, recBrief, bestBrief, null), entry(f2, null, bestBrief, { ...bestBrief, setup_id: closestRow.setup_id, label: closestRow.label })],
});
console.log(`wrote ${path.relative(process.cwd(), here)}/data and site`);
