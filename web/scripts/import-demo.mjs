#!/usr/bin/env node
/**
 * Import the milestone-1 demo replays into the web app's static data.
 *
 *   node scripts/import-demo.mjs [--repo DIR] [--src DIR] [--out DIR] [--python BIN] [--no-python]
 *
 * 1. Unless --no-python, runs scripts/demo_assets.py with the repository's
 *    virtualenv to export each manual entrant's STLs and GLB into
 *    <out>/designs/<design_id>/ and to run the G1 gate under the current task
 *    config. The gate records every case's replay in the same run, so the replays,
 *    the scorer's numbers and the task constants the site prints beside them all
 *    describe one simulation. The import fails if the gate does not pass.
 * 2. With --src DIR, or with --no-python (default DIR: <repo>/data/demo/m1), the
 *    replays are copied from DIR instead. Every replay is checked against the task
 *    YAML (palm size, full-open finger gap, frame rate) and, with Python, against
 *    the gate's outcomes and the entrants' current design ids; a replay recorded
 *    under another config fails the import. Without Python, lift and hold are
 *    derived from the 30 fps replay frames (marked `source: "replay-frames"`).
 * 3. Writes the replays to <out>/replays/m1/ and bouts.json, designs.json and
 *    jobs.json under <out>. Entries this importer produced earlier (source
 *    "demo-m1") are replaced; entries from other producers (for example Nebius
 *    exports) are kept.
 *
 * Nothing under <out> changes until every step has succeeded, so a failed import
 * (for example a stale replay set) leaves the published data intact. Replays are
 * deterministic for a given sim and config; the gate's timestamps and wall times
 * in bouts.json and jobs.json change on every import.
 */

import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { parse as parseYaml } from "yaml";

import { holdTrack, liftTrack, POSE_STRIDE, referenceFrame, summarizeTracks } from "../lib/track.mjs";

export const SOURCE = "demo-m1";
const REPLAY_SUBDIR = "replays/m1";

/** The gate's baseline entrant (src/rtt/sim/m1.py BASELINE_ENTRANT). */
export const BASELINE = "straight";

/** Lane labels for the planted-failure cases; other cases fall back to their humanized name. */
export const CASE_LABELS = {
  too_high: "Closes too high",
  low_friction: "Slippery box",
  too_wide: "Box turned wide",
};

const NUMBER_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"];
const numberWord = (n) => NUMBER_WORDS[n] ?? String(n);
const humanize = (s) => {
  const t = String(s).replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
};

/**
 * How the recorded gate cases are grouped into bouts, decided from each replay's
 * metadata so new entrants or cases appear without editing this file:
 *
 * - entrants: every case that is an entrant's own grasp (case == entrant), in the
 *   entrants YAML order;
 * - failures (featured): the baseline grasp first, then every case expected to fail.
 *
 * Text placeholders are filled from the task and the lanes.
 */
export const DEMO_BOUTS = [
  {
    id: "m1-entrants",
    kind: "gate",
    featured: false,
    title: "{count} baseline fingers, one box",
    summary:
      "Each hand-designed entrant tries to lift a {object} and hold it for {hold}. Same gripper, same controller, same {grip} grip: only the fingers differ.",
    select: (meta) => meta.case === meta.entrant,
    label: (meta) => humanize(meta.entrant),
  },
  {
    id: "m1-failures",
    kind: "gate",
    featured: true,
    title: "One good grasp and {failures} planted failures",
    summary:
      "Before any search runs, the judge has to tell a grasp from a miss. The same {baseline} fingers are set up to fail in {failures} different ways, and each failure has to get its own label.",
    select: (meta) => meta.case === BASELINE || meta.expect_success === false,
    label: (meta) => (meta.case === BASELINE ? "Baseline grasp" : CASE_LABELS[meta.case] ?? humanize(meta.case)),
  },
];

/** Fallback certificate checks (Addendum A) when the task YAML has no `robustness` section. */
export const DEFAULT_CERTIFICATE_CHECKS = [
  { name: "Placement error 0 mm", condition: "Object at the aimed pose" },
  { name: "Placement error 1-3 mm", condition: "Object offset by 1, 2 and 3 mm along x and y" },
  { name: "Shake during hold", condition: "Palm shakes while holding the object" },
  { name: "Lateral push", condition: "Sideways push on the object during the hold" },
];

/** Max allowed gap between the scorer's peak lift and the one read from replay frames. */
export const STALE_LIFT_TOLERANCE_MM = 5;

/** Max difference, metres, between a replay's gripper geometry and the task YAML's. */
export const CONFIG_TOLERANCE_M = 1e-4;

/** Compute notes for the two ways a demo bout's replays reach the site. */
const NOTE_RECORDED_AT_IMPORT =
  "Recorded at import, in the same run as the scorer's numbers. Scored runs execute on Nebius Serverless Jobs.";
const NOTE_COPIED = "Recorded during development. Scored runs execute on Nebius Serverless Jobs.";

const r = (v, digits = 6) => Number(Number(v).toFixed(digits));

function fail(message) {
  console.error(`import-demo: ${message}`);
  process.exit(1);
}

/** Task constants the viewer needs, read from the task YAML (single source of truth). */
export function taskSummary(doc) {
  const c = doc.controller;
  const s = doc.score;
  const g = doc.gripper;
  for (const [section, value] of [["controller", c], ["score", s], ["gripper", g]]) {
    if (!value || typeof value !== "object") throw new Error(`task YAML has no ${section} section`);
  }
  const tDescend = c.settle_time;
  const tClose = tDescend + c.descend_time;
  const tLift = tClose + c.close_time + c.squeeze_time;
  const tHold = tLift + c.lift_time;
  return {
    summary: {
      name: String(doc.name),
      phases: [
        { name: "settle", start_s: 0, end_s: r(tDescend) },
        { name: "descend", start_s: r(tDescend), end_s: r(tClose) },
        { name: "close", start_s: r(tClose), end_s: r(tLift) },
        { name: "lift", start_s: r(tLift), end_s: r(tHold) },
        { name: "hold", start_s: r(tHold), end_s: null },
      ],
      success_lift_m: s.success_lift,
      success_hold_s: s.success_hold,
      record_tail_s: s.record_tail,
      max_time_s: s.max_time,
      grip_force_n: g.grip_force,
      max_opening_mm: r(2 * g.open_half_gap * 1000, 3),
    },
    graspHeightFracs: c.grasp_height_fracs ?? [],
    certificateChecks: certificateChecks(doc.robustness),
  };
}

/** The robustness certificate's checks, as configured in the task YAML's `robustness` section. */
export function certificateChecks(rb) {
  if (!rb || typeof rb !== "object") return DEFAULT_CERTIFICATE_CHECKS;
  const checks = [{ name: "Placement error 0 mm", condition: "Object at the aimed pose" }];
  for (const e of rb.placement_errors_mm ?? []) {
    checks.push({ name: `Placement error ${e} mm`, condition: `Object offset by ${e} mm from the aimed pose, along x and along y` });
  }
  if (rb.shake_amplitude) {
    const axes = (rb.shake_axes ?? []).join(", ");
    checks.push({
      name: "Shake during hold",
      condition: `Palm shakes ${r(rb.shake_amplitude * 1000, 1)} mm at ${rb.shake_frequency} Hz for ${rb.shake_cycles} cycles, along ${axes}`,
    });
  }
  if (rb.push_impulse) {
    const axes = (rb.push_axes ?? []).join(" and ");
    checks.push({
      name: "Lateral push",
      condition: `A ${rb.push_impulse} N s push on the object over ${r(rb.push_duration * 1000, 1)} ms, along ${axes}`,
    });
  }
  return checks;
}

/** Read and check every <stem>.json/<stem>.bin pair in `dir`. */
export function readReplays(dir) {
  const replays = new Map();
  for (const name of readdirSync(dir).sort()) {
    if (!name.endsWith(".json")) continue;
    const stem = name.slice(0, -5);
    const header = JSON.parse(readFileSync(path.join(dir, name), "utf8"));
    if (header.format !== "rtt-replay" || header.version !== 1) continue;
    const binPath = path.join(dir, header.bin ?? `${stem}.bin`);
    if (!existsSync(binPath)) throw new Error(`${name}: missing payload ${path.basename(binPath)}`);
    const expected = header.n_frames * header.n_bodies * POSE_STRIDE * 4;
    const size = statSync(binPath).size;
    if (size !== expected) throw new Error(`${name}: payload is ${size} bytes, header implies ${expected}`);
    if (header.bodies.length !== header.n_bodies) throw new Error(`${name}: n_bodies does not match the body list`);
    const buf = readFileSync(binPath);
    const frames = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    replays.set(stem, { stem, header, binPath, jsonPath: path.join(dir, name), frames });
  }
  return replays;
}

/**
 * Ways a replay disagrees with the task YAML's gripper, i.e. signs that it was
 * recorded under another config: the palm box's half size, the fingers' full-open
 * offset from the palm in the first frame (`open_half_gap`) and the frame rate.
 * Returns human-readable mismatches; empty when the replay is consistent.
 */
export function replayConfigMismatches(replay, taskDoc) {
  const { header, frames } = replay;
  const g = taskDoc?.gripper ?? {};
  const out = [];
  const fps = taskDoc?.replay?.fps;
  if (fps !== undefined && header.fps !== fps) out.push(`frame rate ${header.fps} fps, task YAML ${fps} fps`);
  const palm = header.bodies.findIndex((b) => b.name === "palm");
  const palmBox = palm >= 0 ? header.bodies[palm].geoms.find((x) => x.type === "box") : undefined;
  if (palmBox && Array.isArray(g.palm_half_size)) {
    const gap = Math.max(...g.palm_half_size.map((v, i) => Math.abs(v - (palmBox.half_size?.[i] ?? Number.NaN))));
    if (!(gap <= CONFIG_TOLERANCE_M)) {
      out.push(`palm half size [${palmBox.half_size}] m, task YAML [${g.palm_half_size}] m`);
    }
  }
  if (palm >= 0 && typeof g.open_half_gap === "number" && header.n_frames > 0) {
    for (const name of ["finger_left", "finger_right"]) {
      const b = header.bodies.findIndex((x) => x.name === name);
      if (b < 0) continue;
      const dx = frames[b * POSE_STRIDE] - frames[palm * POSE_STRIDE];
      const dy = frames[b * POSE_STRIDE + 1] - frames[palm * POSE_STRIDE + 1];
      const offset = Math.hypot(dx, dy);
      if (!(Math.abs(offset - g.open_half_gap) <= CONFIG_TOLERANCE_M)) {
        out.push(`${name} opens ${r(offset * 1000, 2)} mm from the palm, task YAML ${r(g.open_half_gap * 1000, 2)} mm`);
      }
    }
  }
  return out;
}

/**
 * Check every replay against the task YAML and, when known, the entrants' current
 * design ids (`designIds`: entrant name -> id). Throws on the first stale replay.
 *
 * @param {Map<string, any>} replays
 * @param {any} taskDoc
 * @param {Map<string, string> | null} [designIds]
 */
export function checkReplaysCurrent(replays, taskDoc, designIds = null) {
  for (const replay of replays.values()) {
    const problems = replayConfigMismatches(replay, taskDoc);
    const meta = replay.header.meta ?? {};
    const current = designIds?.get(meta.entrant);
    if (current !== undefined && meta.design_id !== current) {
      problems.push(`design ${meta.design_id} of entrant ${meta.entrant}, entrants YAML gives ${current}`);
    }
    if (problems.length) {
      throw new Error(
        `${replay.stem}: recorded under a different config (${problems.join("; ")}); re-run \`rtt m1\` or import without --src`,
      );
    }
  }
}

/** Lift and hold derived from the replay frames (used when the scorer's numbers are unavailable). */
export function frameOutcome(replay, task) {
  const { header, frames } = replay;
  const object = header.bodies.findIndex((b) => b.name === "object");
  if (object < 0) return { maxLift: 0, maxHold: 0 };
  const descend = task.phases.find((p) => p.name === "descend")?.start_s ?? 0;
  const lift = liftTrack(frames, header.n_bodies, object, referenceFrame(header.fps, descend, header.n_frames));
  return summarizeTracks(lift, holdTrack(lift, header.fps, task.success_lift_m));
}

function fill(template, values) {
  return template.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? `{${key}}`);
}

function objectLabel(obj) {
  const dims = obj.size_mm.map((v) => Number(v.toFixed(1))).join(" x ");
  return `${dims} mm ${obj.kind}`;
}

/** Build one lane from a replay, the task and (optionally) the re-run gate's case. */
export function buildLane(replay, label, task, graspHeightFracs, gateCase) {
  const { header, stem } = replay;
  const meta = header.meta ?? {};
  const setting = Number(meta.controller_setting ?? 0);
  let outcome;
  if (gateCase) {
    if (gateCase.success !== meta.success || (gateCase.failure_mode ?? null) !== (meta.failure_mode ?? null)) {
      throw new Error(
        `${stem}: the re-run gate gives ${gateCase.success ? "success" : gateCase.failure_mode} but the replay recorded ` +
          `${meta.success ? "success" : meta.failure_mode}; the demo replays are stale, re-run \`rtt m1\``,
      );
    }
    const { maxLift } = frameOutcome(replay, task);
    const gapMm = Math.abs(maxLift * 1000 - gateCase.max_lift_m * 1000);
    if (gapMm > STALE_LIFT_TOLERANCE_MM) {
      throw new Error(
        `${stem}: the re-run scorer's peak lift (${(gateCase.max_lift_m * 1000).toFixed(1)} mm) differs from the replay's ` +
          `(${(maxLift * 1000).toFixed(1)} mm) by more than ${STALE_LIFT_TOLERANCE_MM} mm; re-run \`rtt m1\``,
      );
    }
    outcome = {
      success: gateCase.success,
      failure_mode: gateCase.failure_mode ?? null,
      max_lift_mm: r(gateCase.max_lift_m * 1000, 1),
      hold_s: r(gateCase.hold_time_s, 3),
      flags: gateCase.flags ?? [],
      source: "scorer",
      wall_s: r(gateCase.wall_time_s, 3),
      sim_steps: gateCase.sim_steps,
    };
  } else {
    const { maxLift, maxHold } = frameOutcome(replay, task);
    outcome = {
      success: Boolean(meta.success),
      failure_mode: meta.failure_mode ?? null,
      max_lift_mm: r(maxLift * 1000, 1),
      hold_s: r(maxHold, 3),
      flags: meta.flags ?? [],
      source: "replay-frames",
      wall_s: null,
      sim_steps: null,
    };
  }
  const obj = meta.object ?? { kind: "box", size: [] };
  return {
    id: stem.replace(/_/g, "-"),
    replay: `${REPLAY_SUBDIR}/${stem}`,
    label,
    condition: String(meta.description ?? ""),
    entrant: String(meta.entrant ?? "unknown"),
    design_id: String(meta.design_id ?? "unknown"),
    object: { kind: obj.kind, size_mm: obj.size.map((v) => r(v * 1000, 3)) },
    controller_setting: setting,
    grasp_height_frac: graspHeightFracs[setting] ?? null,
    seed: Number(meta.seed ?? 0),
    n_frames: header.n_frames,
    duration_s: r((header.n_frames - 1) / header.fps, 4),
    outcome,
    design_provenance: { kind: "manual", note: "Hand-designed baseline entrant" },
  };
}

/** Order replays for a bout: the baseline first, then entrants in YAML order, then the case table order. */
function laneOrder(entrantOrder) {
  const caseOrder = Object.keys(CASE_LABELS);
  const rank = ({ header }) => {
    const meta = header.meta ?? {};
    if (meta.case === BASELINE) return -1;
    const e = entrantOrder.indexOf(meta.case);
    if (e >= 0) return e;
    const c = caseOrder.indexOf(meta.case);
    return 1000 + (c >= 0 ? c : caseOrder.length);
  };
  return (a, b) => rank(a) - rank(b) || a.stem.localeCompare(b.stem);
}

/**
 * Group replays into the demo bouts. `recordedAt` (ISO time) is set when the replays
 * were recorded by this import's gate run, which also selects the compute note.
 *
 * @param {Map<string, any>} replays
 * @param {any} task
 * @param {number[]} graspHeightFracs
 * @param {any} gate
 * @param {any} versions
 * @param {string[]} [entrantOrder]
 * @param {string | null} [recordedAt]
 */
export function buildBouts(replays, task, graspHeightFracs, gate, versions, entrantOrder = [], recordedAt = null) {
  const used = new Set();
  const bouts = [];
  for (const spec of DEMO_BOUTS) {
    const chosen = [...replays.values()].filter((rep) => spec.select(rep.header.meta ?? {})).sort(laneOrder(entrantOrder));
    if (chosen.length === 0) continue;
    const lanes = chosen.map((rep) => {
      used.add(rep.stem);
      return buildLane(rep, spec.label(rep.header.meta ?? {}), task, graspHeightFracs, gate?.cases?.[rep.stem]);
    });
    const failures = chosen.filter((rep) => rep.header.meta?.expect_success === false).length;
    const values = {
      count: numberWord(lanes.length),
      failures: numberWord(failures).toLowerCase(),
      baseline: BASELINE,
      grip: `${task.grip_force_n} N`,
      hold: `${task.success_hold_s} s`,
      opening: `${task.max_opening_mm} mm`,
      object: objectLabel(lanes[0].object),
    };
    const walls = lanes.map((l) => l.outcome.wall_s);
    const steps = lanes.map((l) => l.outcome.sim_steps);
    bouts.push({
      id: spec.id,
      source: SOURCE,
      kind: spec.kind,
      title: fill(spec.title, values),
      summary: fill(spec.summary, values),
      featured: spec.featured,
      recorded_at: recordedAt,
      task,
      compute: {
        kind: "local",
        executor: "cpu",
        engine: versions?.mujoco ? `MuJoCo ${versions.mujoco}` : "MuJoCo",
        host: "Developer laptop",
        wall_s: walls.every((w) => w !== null) ? r(walls.reduce((a, b) => a + b, 0), 3) : null,
        sim_steps: steps.every((s) => s !== null) ? steps.reduce((a, b) => a + b, 0) : null,
        note: recordedAt ? NOTE_RECORDED_AT_IMPORT : NOTE_COPIED,
      },
      lanes,
    });
  }
  const unused = [...replays.keys()].filter((s) => !used.has(s));
  return { bouts, unused };
}

export function buildDesigns(assets, entrantsDoc, replays, bouts, checks = DEFAULT_CERTIFICATE_CHECKS) {
  const boutsByDesign = new Map();
  for (const bout of bouts) {
    for (const lane of bout.lanes) {
      const list = boutsByDesign.get(lane.design_id) ?? [];
      if (!list.includes(bout.id)) list.push(bout.id);
      boutsByDesign.set(lane.design_id, list);
    }
  }
  const certificate = {
    status: "pending",
    issued_at: null,
    checks: checks.map((c) => ({ ...c, success_rate: null, episodes: 0 })),
  };
  if (assets) {
    return assets.designs.map((d) => ({
      id: d.id,
      source: SOURCE,
      name: d.name,
      origin: "manual",
      rationale: d.rationale ?? null,
      params: d.params,
      valid: d.valid,
      errors: d.errors,
      warnings: d.warnings,
      mass_printed_g: d.mass_printed_g,
      mass_sim_g: d.mass_sim_g,
      bbox_mm: d.bbox_mm,
      watertight: d.watertight,
      n_boxes: d.n_boxes,
      files: Object.fromEntries(Object.entries(d.files).map(([k, f]) => [k, `designs/${d.id}/${f}`])),
      provenance: { kind: "manual", note: null },
      bouts: boutsByDesign.get(d.id) ?? [],
      certificate,
    }));
  }
  // Without Python: parameters from the replays' design metadata, rationale from the entrants YAML.
  const rationale = new Map((entrantsDoc?.entrants ?? []).map((e) => [e.name, e.rationale ?? null]));
  const seen = new Map();
  for (const { header } of replays.values()) {
    const meta = header.meta ?? {};
    if (!meta.design_id || !meta.design || seen.has(meta.design_id)) continue;
    const { name, ...params } = meta.design;
    seen.set(meta.design_id, {
      id: meta.design_id,
      source: SOURCE,
      name: name ?? meta.entrant,
      origin: "manual",
      rationale: rationale.get(name) ?? null,
      params,
      valid: true,
      errors: [],
      warnings: [],
      mass_printed_g: null,
      mass_sim_g: null,
      bbox_mm: null,
      watertight: null,
      n_boxes: null,
      files: {},
      provenance: { kind: "manual", note: null },
      bouts: boutsByDesign.get(meta.design_id) ?? [],
      certificate,
    });
  }
  return [...seen.values()];
}

export function buildGateJob(gate, bouts) {
  return {
    id: "local-m1-gate",
    source: SOURCE,
    name: "Milestone 1 gate",
    purpose: gate.recorded
      ? "Ran the milestone 1 gate at import; the demo bouts' replays and the scorer's numbers come from this run."
      : "Re-ran the milestone 1 gate at import to check the copied replays and read the scorer's own numbers.",
    kind: "local",
    executor: "cpu",
    platform: null,
    preset: null,
    region: null,
    status: "completed",
    started_at: gate.started_at,
    ended_at: gate.ended_at,
    wall_s: gate.wall_time_s,
    cost_usd: null,
    episodes: Object.keys(gate.cases).length,
    sim_steps: gate.sim_steps,
    bouts: bouts.map((b) => b.id),
  };
}

function readIndex(file, schema, key) {
  if (!existsSync(file)) return { schema, version: 1, [key]: [] };
  const doc = JSON.parse(readFileSync(file, "utf8"));
  if (doc.schema !== schema) throw new Error(`${file}: expected schema ${schema}, found ${doc.schema}`);
  return doc;
}

/** Replace this importer's entries, keep everyone else's (theirs first). */
export function mergeBySource(existing, fresh) {
  return [...existing.filter((x) => x.source !== SOURCE), ...fresh];
}

function writeJson(file, doc) {
  writeFileSync(file, `${JSON.stringify(doc, null, 1)}\n`, "utf8");
}

function runAssets(python, webDir, repo, outDesigns, entrantsPath, taskPath, replaysDir) {
  const script = path.join(webDir, "scripts", "demo_assets.py");
  const args = ["-I", script, "--entrants", entrantsPath, "--task", taskPath, "--out", outDesigns];
  if (replaysDir) args.push("--replays", replaysDir);
  const res = spawnSync(python, args, { cwd: repo, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (res.error) throw new Error(`cannot run ${python}: ${res.error.message}`);
  if (res.status !== 0) throw new Error(`demo_assets.py exited with ${res.status}:\n${res.stderr.trim()}`);
  return JSON.parse(res.stdout);
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({
    options: {
      repo: { type: "string", default: path.resolve(webDir, "..") },
      src: { type: "string" },
      out: { type: "string", default: path.join(webDir, "public", "data") },
      python: { type: "string" },
      "no-python": { type: "boolean", default: false },
    },
  });
  const repo = path.resolve(values.repo);
  const usePython = !values["no-python"];
  // With Python the gate records the replays itself unless --src names a set to copy.
  const copySrc =
    values.src !== undefined || !usePython ? path.resolve(values.src ?? path.join(repo, "data", "demo", "m1")) : null;
  const out = path.resolve(values.out);
  const taskPath = path.join(repo, "configs", "task_v1.yaml");
  const entrantsPath = path.join(repo, "configs", "entrants.yaml");
  const python = values.python ?? path.join(repo, ".venv", "bin", "python");

  if (copySrc && !existsSync(copySrc)) fail(`no replay directory at ${copySrc}; run \`rtt m1\` in the repository first`);
  const taskDoc = parseYaml(readFileSync(taskPath, "utf8"));
  const { summary: task, graspHeightFracs, certificateChecks: checks } = taskSummary(taskDoc);
  const replayOut = path.join(out, REPLAY_SUBDIR);

  // Everything is computed (and every check run) before anything under <out> changes,
  // so a failed import leaves the published data as it was.
  const staging = mkdtempSync(path.join(os.tmpdir(), "rtt-import-"));
  try {
    // 1. print files, scorer numbers and (unless copying) replays, into a staging directory
    let assets = null;
    if (usePython) {
      if (!existsSync(python)) throw new Error(`no Python at ${python}; pass --python or --no-python`);
      const recordDir = copySrc ? null : path.join(staging, "replays");
      assets = runAssets(python, webDir, repo, path.join(staging, "designs"), entrantsPath, taskPath, recordDir);
      if (!assets.gate?.passed) {
        const bad = Object.entries(assets.gate?.cases ?? {})
          .filter(([, c]) => !c.passed)
          .map(([name]) => name);
        throw new Error(`the G1 gate did not pass (${bad.join(", ") || "invalid entrants"}); nothing was published`);
      }
    }
    const replaySrc = copySrc ?? path.join(staging, "replays");
    const replays = readReplays(replaySrc);
    if (replays.size === 0) throw new Error(`no rtt-replay files in ${replaySrc}`);
    const designIds = assets ? new Map(assets.designs.map((d) => [d.name, d.id])) : null;
    checkReplaysCurrent(replays, taskDoc, designIds);

    // 2. indexes
    const gate = assets?.gate ?? null;
    const recordedAt = gate && !copySrc ? gate.started_at : null;
    const entrantsDoc = parseYaml(readFileSync(entrantsPath, "utf8"));
    const entrantOrder = (entrantsDoc?.entrants ?? []).map((e) => e.name);
    const { bouts, unused } = buildBouts(
      replays,
      task,
      graspHeightFracs,
      gate,
      assets?.versions,
      entrantOrder,
      recordedAt,
    );
    if (unused.length) console.warn(`import-demo: warning: replays not in any bout: ${unused.join(", ")}`);
    const designs = buildDesigns(assets, entrantsDoc, replays, bouts, checks);

    const boutsFile = path.join(out, "bouts.json");
    const designsFile = path.join(out, "designs.json");
    const jobsFile = path.join(out, "jobs.json");
    const boutsDoc = readIndex(boutsFile, "rtt-web-bouts", "bouts");
    const designsDoc = readIndex(designsFile, "rtt-web-designs", "designs");
    const jobsDoc = readIndex(jobsFile, "rtt-web-jobs", "jobs");
    // Without Python no gate ran, so no local job backs these bouts.
    const jobs = gate ? [buildGateJob(gate, bouts)] : [];

    // 3. publish: replays, print files, then the indexes that point at them
    rmSync(replayOut, { recursive: true, force: true });
    mkdirSync(replayOut, { recursive: true });
    for (const { binPath, jsonPath } of replays.values()) {
      copyFileSync(binPath, path.join(replayOut, path.basename(binPath)));
      copyFileSync(jsonPath, path.join(replayOut, path.basename(jsonPath)));
    }
    if (assets) {
      const designsOut = path.join(out, "designs");
      for (const d of designsDoc.designs.filter((x) => x.source === SOURCE)) {
        rmSync(path.join(designsOut, d.id), { recursive: true, force: true });
      }
      mkdirSync(designsOut, { recursive: true });
      for (const d of assets.designs) {
        const from = path.join(staging, "designs", d.id);
        if (existsSync(from)) cpSync(from, path.join(designsOut, d.id), { recursive: true });
      }
    }
    writeJson(boutsFile, { schema: "rtt-web-bouts", version: 1, bouts: mergeBySource(boutsDoc.bouts, bouts) });
    writeJson(designsFile, {
      schema: "rtt-web-designs",
      version: 1,
      params: assets?.schema ?? designsDoc.params ?? {},
      designs: mergeBySource(designsDoc.designs, designs),
    });
    writeJson(jobsFile, { schema: "rtt-web-jobs", version: 1, jobs: mergeBySource(jobsDoc.jobs, jobs) });
    const lanes = bouts.reduce((n, b) => n + b.lanes.length, 0);
    console.log(
      `import-demo: ${replays.size} replays (${copySrc ? `copied from ${copySrc}` : "recorded by the gate"}) -> ` +
        `${path.relative(webDir, replayOut)}/, ${bouts.length} bouts (${lanes} lanes), ${designs.length} designs, ` +
        `outcomes from ${gate ? "the scorer" : "replay frames"}`,
    );
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (err) {
    fail(err instanceof Error ? err.message : String(err));
  }
}
