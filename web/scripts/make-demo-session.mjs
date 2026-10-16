#!/usr/bin/env node
/**
 * Write the ILLUSTRATIVE Grasp Engineer session the /part page plays when no live
 * backend is configured, plus the example part it is about.
 *
 *   node scripts/make-demo-session.mjs [--force]
 *
 * Outputs:
 *   public/data/sessions/demo-dumbbell.json      (`kind: "illustrative"`)
 *   public/data/sessions/parts/dumbbell-handle.stl
 *
 * The session is layout data, not a recorded agent run: its job ids say
 * "illustrative", and the page stamps it "Illustrative session, not a recorded run".
 * It uses the agent's own session format (src/rtt/agent/recorder.py), so a recorded
 * session published with scripts/import-session.mjs plays through the same code.
 * Its timings and costs are kept in line with the measured pilot jobs in
 * docs/NEBIUS.md (about two minutes before a container starts, about 100 episodes/s
 * on 16 vCPUs, about $0.003 per Nemotron call) so the layout shows realistic values.
 * A recorded session (`kind: "recorded"`) from the agent replaces it; this script
 * refuses to overwrite a recorded file.
 *
 * The part is a dumbbell handle (a 16 mm bar, 60 mm long, between two 40 mm discs
 * 12 mm thick; CC0), the same shape as src/rtt/part/examples.py `dumbbell_handle`,
 * built here as one watertight surface of revolution and written as binary STL in mm.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

/** Profile (radius, axial position) in mm, revolved about the part's long axis. */
const PROFILE = [
  [0, 0],
  [20, 0],
  [20, 12],
  [8, 12],
  [8, 72],
  [20, 72],
  [20, 84],
  [0, 84],
];

/**
 * Binary STL of a surface of revolution: the profile is swept about the x axis in
 * `segments` steps, then placed resting on z = 0 and centred in x and y.
 */
export function revolvedStl(profile = PROFILE, segments = 48) {
  const length = Math.max(...profile.map((p) => p[1]));
  const radius = Math.max(...profile.map((p) => p[0]));
  const point = (r, a, theta) => [a - length / 2, r * Math.cos(theta), r * Math.sin(theta) + radius];
  const tris = [];
  for (let i = 0; i + 1 < profile.length; i++) {
    const [r0, a0] = profile[i];
    const [r1, a1] = profile[i + 1];
    for (let s = 0; s < segments; s++) {
      const t0 = (2 * Math.PI * s) / segments;
      const t1 = (2 * Math.PI * (s + 1)) / segments;
      const p00 = point(r0, a0, t0);
      const p01 = point(r0, a0, t1);
      const p10 = point(r1, a1, t0);
      const p11 = point(r1, a1, t1);
      // Outward winding: the profile runs from x- to x+ around the outside.
      if (r0 > 0) tris.push([p00, p01, p11]);
      if (r1 > 0) tris.push([p00, p11, p10]);
    }
  }
  const buf = Buffer.alloc(84 + tris.length * 50);
  buf.write("rtt example part: dumbbell handle (CC0), millimetres", 0, "ascii");
  buf.writeUInt32LE(tris.length, 80);
  let o = 84;
  for (const [a, b, c] of tris) {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const len = Math.hypot(...n) || 1;
    for (const x of [...n.map((k) => k / len), ...a, ...b, ...c]) {
      buf.writeFloatLE(x, o);
      o += 4;
    }
    buf.writeUInt16LE(0, o);
    o += 2;
  }
  return buf;
}

const SUPER = "nvidia/nemotron-3-super-120b-a12b";
const LIGHTNING = "nvidia/Nemotron-3_5-Lightning";
const PART_ID = "part-illustrative";
const STRAIGHT = "462150ca1475";

/** Finger parameters (src/rtt/design/schema.py geometry_params). */
const P = (length, thickness, width, face, depth, tip, tipSize, taper, bend = 0, bendAt = 0.5) => ({
  finger_length_mm: length,
  finger_thickness_mm: thickness,
  finger_width_mm: width,
  bend_angle_deg: bend,
  bend_start_frac: bendAt,
  inner_face: face,
  inner_face_depth_mm: depth,
  tip_profile: tip,
  tip_size_mm: tipSize,
  taper_ratio: taper,
});

const D = {
  saddle: { design_id: "r1-saddle", hypothesis: "A v-groove centres the 16 mm bar; a short finger keeps the tips clear of the 40 mm discs.", params: P(58, 10, 18, "v_groove", 4, "chamfer", 3, 0.85) },
  hook: { design_id: "r1-hook", hypothesis: "A hooked tip wraps under the bar so the lift cannot pull it out of a weak pinch.", params: P(66, 9, 16, "flat", 0, "hook", 6, 0.9) },
  cradle: { design_id: "r1-cradle", hypothesis: "A concave channel cradles the bar on two lines and resists it rolling under a push.", params: P(62, 12, 20, "concave_arc", 5, "rounded", 3, 1) },
  hookGroove: { design_id: "r2-hook-groove", hypothesis: "Combine the two best: the v-groove for centring and a 5 mm hook against slip during the lift.", params: P(62, 10, 18, "v_groove", 4, "hook", 5, 0.9) },
  wideGroove: { design_id: "r2-wide-groove", hypothesis: "A wider v-groove finger resists the bar twisting when it is pushed sideways at the top.", params: P(60, 11, 24, "v_groove", 5, "chamfer", 3, 0.85) },
};

/** A Nebius job record as src/rtt/agent/jobs.py JobRecord.to_json() writes it. */
const job = (id, designs, state, elapsed, est) => ({
  job_id: id,
  target: "nebius",
  part_id: PART_ID,
  designs,
  state,
  elapsed_s: elapsed,
  est_usd: est,
  platform: "cpu-d3",
  preset: "16vcpu-64gb",
});

/** One fetch_results row (tools.py _compact_run). */
const row = (d, label, pct, byK, hold, mass, failures) => ({
  design_id: d,
  label,
  success_pct: pct,
  verdict: pct >= 80 ? "pass" : "fail",
  success_by_grasp_height_pct: byK,
  hold_fraction: hold,
  mass_g: mass,
  failure_modes: failures,
  episodes: 24,
});

/**
 * The illustrative session document (see the module comment), in the agent's own
 * format (src/rtt/agent/recorder.py): dotted event types, `t_s`, tool results shaped
 * by src/rtt/agent/tools.py. Spend totals are summed from its own events.
 */
export function demoSession() {
  const J1 = "illustrative-job-1";
  const J2 = "illustrative-job-2";
  const r1 = [D.saddle.design_id, D.hook.design_id, D.cradle.design_id, STRAIGHT];
  const r2 = [D.hookGroove.design_id, D.wideGroove.design_id];
  // The session and day dollar caps are the API's defaults (src/rtt/agent/caps.py AgentCaps,
  // RTT_AGENT_SESSION_USD / RTT_AGENT_DAY_USD), so the meter shows the cap a live run has.
  const caps = { max_tool_calls: 12, max_concurrent_jobs: 2, session_usd: 0.5, day_usd: 2, max_llm_turns: 16, max_designs_per_job: 8, max_designs_per_proposal: 4, poll_wait_s: 90 };
  let calls = 0;
  const call = (t, name, args) => ({ t_s: t, type: "tool.call", call_id: `call_${++calls}`, name, args, n: calls });
  const result = (t, name, res, duration) => ({ t_s: t, type: "tool.result", call_id: `call_${calls}`, name, ok: true, duration_s: duration, result: res, tool_calls_left: caps.max_tool_calls - calls });
  const llm = (t, role, tin, tout, usd, lat) => ({ t_s: t, type: "llm.call", role, model: role === "summarizer" ? LIGHTNING : SUPER, prompt_tokens: tin, completion_tokens: tout, reasoning_tokens: null, cost_usd: usd, latency_s: lat, cached: false, finish_reason: role === "agent" ? "tool_calls" : "stop" });
  const say = (t, text) => ({ t_s: t, type: "agent.message", content: text, tool_calls: [], reasoning: null });
  const summary = (t, step, text) => ({ t_s: t, type: "summary", step, text, model: LIGHTNING });

  const events = [
    { t_s: 0, type: "session.started", part_id: PART_ID, goal: "Find gripper fingers that reliably lift and hold my part, check how robust the best one is, and give me a print pack.", label: "Product mode: not part of the pre-registered experiment", models: { agent: SUPER, proposer: SUPER, summarizer: LIGHTNING }, offline: false, caps, unlocked: true, targets: ["local", "nebius"] },
    llm(1.5, "agent", 1240, 96, 0.0005, 1.4),
    say(1.6, "I will measure the part and find how it rests, then design fingers for the grasp it allows."),
    call(1.7, "inspect_part", { part_id: PART_ID }),
    result(6.3, "inspect_part", {
      part: {
        part_id: PART_ID,
        name: "dumbbell-handle",
        size_mm: [84, 40, 40],
        grip_width_mm: 40,
        height_mm: 40,
        volume_cm3: 42.1,
        mass_g: 50.5,
        density_kg_m3: 1200,
        max_opening_mm: 84,
        fits_gripper: true,
        n_convex_pieces: 3,
        stable_pose_probability: 0.94,
        conditions: ["centred", "offset 5 mm", "rotated 10 degrees", "low friction"],
        episodes_per_design: 24,
        notes: ["The bar is 16 mm across and sits 20 mm above the table, between discs 60 mm apart."],
      },
      pass_threshold_pct: 80,
      baselines: [
        { design_id: STRAIGHT, name: "straight", valid_for_part: true },
        { design_id: "9a0624079454", name: "concave", valid_for_part: true },
        { design_id: "732d575caf85", name: "hooked", valid_for_part: false, why_not: ["the hook tip would strike a disc at full close"] },
        { design_id: "be6496fa15e2", name: "stock", valid_for_part: true },
      ],
      targets: ["local", "nebius"],
    }, 4.6),
    llm(7.0, "summarizer", 540, 46, 0.00005, 0.6),
    summary(7.1, "inspect_part", "The handle rests on its two discs nearly every time. Its 40 mm width fits the 84 mm opening with room to spare; the hard part is the thin bar between the discs."),
    llm(8.6, "agent", 1890, 140, 0.0008, 1.5),
    say(8.7, "Lying down, the bar is 16 mm across and the fingers must reach it without touching the discs. I will try four ways to hold a thin bar."),
    call(8.8, "propose_designs", { part_id: PART_ID, n: 4 }),
    llm(17.0, "proposer", 6120, 1004, 0.0028, 8.1),
    result(17.2, "propose_designs", {
      designs: [D.saddle, D.hook, D.cradle],
      rejected: [{ slot: "d4", reasons: ["too similar to r1-saddle: every parameter within 5% of its range"] }],
      analysis: "Thin bar between two discs: the risks are slip during the lift and the fingertips striking the discs. Explore centring (v-groove), wrapping (hook) and cradling (concave channel).",
    }, 8.4),
    llm(18.5, "agent", 2650, 118, 0.0009, 1.2),
    call(18.6, "launch_sim_job", { part_id: PART_ID, designs: r1, target: "nebius" }),
    { t_s: 19.9, type: "job.update", ...job(J1, r1, "QUEUED", 0, 0.021) },
    result(20.0, "launch_sim_job", job(J1, r1, "QUEUED", 0, 0.021), 1.4),
    llm(21.2, "agent", 2980, 64, 0.0010, 1.1),
    call(21.3, "poll_job", { job_id: J1 }),
    { t_s: 111.3, type: "job.update", ...job(J1, r1, "RUNNING", 91.4, 0.021) },
    result(111.4, "poll_job", job(J1, r1, "RUNNING", 91.4, 0.021), 90.1),
    llm(112.5, "agent", 3120, 64, 0.0010, 1.0),
    call(112.6, "poll_job", { job_id: J1 }),
    { t_s: 141.2, type: "job.update", ...job(J1, r1, "COMPLETED", 121.3, 0.021) },
    result(141.3, "poll_job", job(J1, r1, "COMPLETED", 121.3, 0.021), 28.7),
    llm(142.4, "agent", 3260, 60, 0.0010, 1.0),
    call(142.5, "fetch_results", { job_id: J1 }),
    result(143.9, "fetch_results", {
      run_id: "run-illustrative-1",
      job_id: J1,
      target: "nebius",
      pass_threshold_pct: 80,
      best_design_id: D.hook.design_id,
      designs: [
        row(D.hook.design_id, D.hook.design_id, 75, [62.5, 75, 50], 0.79, 14.2, { finger_env_collision: 4, missed_grasp: 2 }),
        row(D.saddle.design_id, D.saddle.design_id, 70.8, [70.8, 58.3, 41.7], 0.74, 12.9, { slip_during_lift: 5, unstable_hold: 2 }),
        row(D.cradle.design_id, D.cradle.design_id, 58.3, [58.3, 45.8, 33.3], 0.63, 17.6, { slip_during_lift: 8, ejected_on_close: 2 }),
        row(STRAIGHT, "straight", 41.7, [41.7, 37.5, 25], 0.48, 19.8, { slip_during_lift: 11, unstable_hold: 3 }),
      ],
    }, 1.4),
    llm(144.8, "summarizer", 760, 52, 0.00006, 0.7),
    summary(144.9, "fetch_results", "The hook finger held 75% of trials, the best so far but below the 80% pass mark; it failed when its tip clipped a disc. The grooved finger slipped during the lift."),
    llm(146.3, "agent", 3880, 152, 0.0013, 1.6),
    say(146.4, "The groove centres the bar but it slips on the way up; the hook holds but its tip clips a disc. Next round: both ideas in one shorter finger, and a wider groove."),
    call(146.5, "propose_designs", { part_id: PART_ID, n: 2 }),
    llm(153.7, "proposer", 7480, 912, 0.0031, 7.0),
    result(153.9, "propose_designs", {
      designs: [D.hookGroove, D.wideGroove],
      rejected: [],
      analysis: "Round 1: slip_during_lift dominates the grooved and cradling fingers, finger_env_collision the hook. Keep the groove, add a short hook, and shorten the finger so the tip clears the discs.",
    }, 7.4),
    llm(155.0, "agent", 4320, 104, 0.0014, 1.1),
    call(155.1, "launch_sim_job", { part_id: PART_ID, designs: r2, target: "nebius" }),
    { t_s: 156.3, type: "job.update", ...job(J2, r2, "QUEUED", 0, 0.019) },
    result(156.4, "launch_sim_job", job(J2, r2, "QUEUED", 0, 0.019), 1.3),
    llm(157.5, "agent", 4510, 64, 0.0014, 1.0),
    call(157.6, "poll_job", { job_id: J2 }),
    { t_s: 247.6, type: "job.update", ...job(J2, r2, "RUNNING", 91.2, 0.019) },
    { t_s: 271.0, type: "job.update", ...job(J2, r2, "COMPLETED", 114.6, 0.019) },
    result(271.1, "poll_job", job(J2, r2, "COMPLETED", 114.6, 0.019), 113.5),
    llm(272.2, "agent", 4640, 60, 0.0014, 1.0),
    call(272.3, "fetch_results", { job_id: J2 }),
    result(273.6, "fetch_results", {
      run_id: "run-illustrative-2",
      job_id: J2,
      target: "nebius",
      pass_threshold_pct: 80,
      best_design_id: D.hookGroove.design_id,
      designs: [
        row(D.hookGroove.design_id, D.hookGroove.design_id, 91.7, [91.7, 83.3, 66.7], 0.93, 13.8, { slip_during_lift: 1, unstable_hold: 1 }),
        row(D.wideGroove.design_id, D.wideGroove.design_id, 79.2, [79.2, 70.8, 54.2], 0.82, 16.4, { slip_during_lift: 4, unstable_hold: 1 }),
      ],
    }, 1.3),
    llm(274.5, "summarizer", 690, 48, 0.00005, 0.6),
    summary(274.6, "fetch_results", "The combined hook and groove finger held 22 of 24 trials and passes. The wider groove fell just short at 79%."),
    llm(275.8, "agent", 5210, 92, 0.0016, 1.2),
    call(275.9, "stress_test", { part_id: PART_ID, design_id: D.hookGroove.design_id }),
    result(290.4, "stress_test", {
      design_id: D.hookGroove.design_id,
      score_pct: 80.8,
      verdict: "pass",
      category_pct: { placement: 92.3, shake: 100, push: 50 },
      placement_tolerance_mm: { x: 3, y: 2 },
      passed: "16/18",
      failed_tests: [
        { test: "offset_y_+3mm", failure_mode: "slip_during_lift" },
        { test: "push_x", failure_mode: "unstable_hold" },
      ],
    }, 14.5),
    llm(291.3, "summarizer", 610, 50, 0.00005, 0.6),
    summary(291.4, "stress_test", "It tolerates the part being placed 3 mm off along the bar and 2 mm across it, survives all three shakes, and drops once when pushed sideways along the bar."),
    llm(292.6, "agent", 5690, 88, 0.0017, 1.2),
    call(292.7, "export_print_pack", { run_id: "run-illustrative-2", design_id: D.hookGroove.design_id }),
    result(296.2, "export_print_pack", {
      pack_id: "pack-illustrative",
      design_id: D.hookGroove.design_id,
      run_id: "run-illustrative-2",
      verdict: "pass",
      certificate_included: true,
      files: [
        { name: "certificate.html", bytes: 41210 },
        { name: "certificate.json", bytes: 6122 },
        { name: "design.json", bytes: 512 },
        { name: "finger_left.stl", bytes: 18484 },
        { name: "finger_right.stl", bytes: 18484 },
        { name: "mount_plate.stl", bytes: 9884 },
      ],
      zip_bytes: 31876,
      download: "/printpack/pack-illustrative",
    }, 3.5),
    llm(297.9, "agent", 6040, 146, 0.0019, 1.7),
    say(298.0, "Recommended design r2-hook-groove: 92% simulated success (pass) and a robustness score of 81% (pass). The print pack has both fingers, the mount plate and the certificate."),
  ];
  const llmUsd = Number(events.filter((e) => e.type === "llm.call").reduce((s, e) => s + e.cost_usd, 0).toFixed(6));
  const jobUsd = 0.021 + 0.019;
  const spend = {
    session_cap_usd: caps.session_usd,
    spent_usd: Number((llmUsd + jobUsd).toFixed(6)),
    by_kind_usd: { llm: llmUsd, nebius: jobUsd },
    reserved_usd: 0,
    session_remaining_usd: Number((caps.session_usd - llmUsd - jobUsd).toFixed(6)),
    day_cap_usd: caps.day_usd,
    day_spent_usd: Number((llmUsd + jobUsd).toFixed(6)),
  };
  const verdicts = [
    { design_id: STRAIGHT, label: "straight", origin: "baseline", success_pct: 41.7, verdict: "fail" },
    { design_id: D.saddle.design_id, label: D.saddle.design_id, origin: "proposed", success_pct: 70.8, verdict: "fail" },
    { design_id: D.hook.design_id, label: D.hook.design_id, origin: "proposed", success_pct: 75, verdict: "fail" },
    { design_id: D.cradle.design_id, label: D.cradle.design_id, origin: "proposed", success_pct: 58.3, verdict: "fail" },
    { design_id: D.hookGroove.design_id, label: D.hookGroove.design_id, origin: "proposed", success_pct: 91.7, verdict: "pass", certificate_score_pct: 80.8, certificate_verdict: "pass" },
    { design_id: D.wideGroove.design_id, label: D.wideGroove.design_id, origin: "proposed", success_pct: 79.2, verdict: "fail" },
  ];
  const finalSummary =
    "r2-hook-groove held the bar in 92% of 24 trials and passed 16 of 18 stress checks: every placement error up to 3 mm along the bar and 2 mm across it, all three shakes, and one of two pushes.";
  events.push(
    llm(299.0, "summarizer", 820, 64, 0.00006, 0.7),
    {
      t_s: 299.2,
      type: "session.finished",
      status: "completed",
      reason: null,
      answer: events.at(-1).content,
      summary: finalSummary,
      verdicts,
      spend: { ...spend },
      turns: 13,
      tool_calls: calls,
    },
  );
  // Totals again with the last summarizer call included.
  const llmAll = Number(events.filter((e) => e.type === "llm.call").reduce((s, e) => s + e.cost_usd, 0).toFixed(6));
  const finished = events.at(-1);
  finished.spend.by_kind_usd = { llm: llmAll, nebius: jobUsd };
  finished.spend.spent_usd = Number((llmAll + jobUsd).toFixed(6));
  finished.spend.session_remaining_usd = Number((caps.session_usd - llmAll - jobUsd).toFixed(6));
  finished.spend.day_spent_usd = finished.spend.spent_usd;
  events.forEach((e, i) => {
    e.seq = i + 1;
  });
  return {
    format: "rtt-agent-session",
    version: 1,
    session_id: "demo-dumbbell",
    kind: "illustrative",
    mode: "product",
    label: "Product mode: not part of the pre-registered experiment",
    status: "completed",
    title: "Dumbbell handle",
    note: "Illustrative session for layout, in the agent's own event format: the job ids, timings and scores are examples in line with the measured pilot jobs, not a recorded agent run.",
    recorded_at: null,
    part: { name: "dumbbell-handle.stl", stl: "sessions/parts/dumbbell-handle.stl", units: "mm" },
    offline: false,
    spend: finished.spend,
    verdicts,
    answer: finished.answer,
    summary: finalSummary,
    events: events.map(({ seq, ...e }) => ({ seq, ...e })),
  };
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({ options: { force: { type: "boolean", default: false } } });
  const dir = path.join(webDir, "public", "data", "sessions");
  const sessionFile = path.join(dir, "demo-dumbbell.json");
  const stlFile = path.join(dir, "parts", "dumbbell-handle.stl");
  if (existsSync(sessionFile)) {
    const current = JSON.parse(readFileSync(sessionFile, "utf8"));
    if (current.kind !== "illustrative") {
      console.error(`make-demo-session: ${sessionFile} is a recorded session; refusing to overwrite`);
      process.exit(1);
    }
    if (!values.force) {
      console.log("make-demo-session: files exist (pass --force to regenerate)");
      return;
    }
  }
  mkdirSync(path.dirname(stlFile), { recursive: true });
  writeFileSync(stlFile, revolvedStl());
  writeFileSync(sessionFile, `${JSON.stringify(demoSession(), null, 1)}\n`, "utf8");
  console.log(`make-demo-session: wrote ${path.relative(webDir, sessionFile)} and ${path.relative(webDir, stlFile)}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
