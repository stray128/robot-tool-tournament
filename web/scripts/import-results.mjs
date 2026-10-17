#!/usr/bin/env node
/**
 * Build public/data/results.json (version 2) from search run directories.
 *
 *   node scripts/import-results.mjs --status pilot --threshold 0.8 \
 *        --threshold-note "Provisional ..." --note "..." RUN_DIR [RUN_DIR ...]
 *
 *   node scripts/import-results.mjs --status final --threshold 0.7708333333333334 \
 *        --s1-analysis ../runs/analysis/s1-confirmatory/s1_v2.json \
 *        --secondary ../runs/analysis/secondary-confirmatory/secondary_v2.json \
 *        ../runs/cloud/search-<random> ../runs/cloud/search-<nemotron>
 *
 * A RUN_DIR is a search output (`rtt search`), or a cloud run directory holding one
 * under `search/` plus the Nebius `job.json`. For every `runs/<arm>-rNN/history.jsonl`
 * it takes each design's training success (`score.success`), in evaluation order, and
 * writes the best-so-far curve. In the v2 design space `score.success` is the
 * registered success (the part lifted, held, seated and centred), the score S1 is
 * defined on; the lift-only rate is `score.lift_success` and is not read. Runs are
 * grouped into arms by the manifest's `arm`. Leading designs that paired runs (the
 * same run slot, so the same seed, in every arm) share are counted as `shared_prefix`
 * so the chart can mark them.
 *
 * With a threshold, every run also gets S1's T (`t`): the 1-based index of the first
 * design from `count_from` on whose own score reaches the threshold, or the run length
 * plus one when none does. `count_from` is one past round 0 (the manifest's first
 * round size): round 0 holds the entrants and seed designs, which no arm chose, so
 * they do not count. This is the rule of `rtt.analysis.s1.evals_to_tau`.
 *
 * --s1-analysis FILE adds the confirmatory S1 comparison from the pre-registered
 * analysis (`s1_v2.json`) after checking that its tau, budget and every run's T match
 * what this script computed from the histories. --secondary FILE adds the Holm family
 * (H1, H2, S2) and the planned H2 width split from `secondary_v2.json`.
 *
 * Only the job id, platform, preset, region and wall time are read from job.json;
 * the job cost comes from the already-published public/data/jobs.json, and each run's
 * model-call spend (Token Factory, billed separately from the job) from its
 * `llm_calls.jsonl`. H1 (held-out) is left
 * `not_run` unless --h1 FILE supplies the `h1` block. Refuses to overwrite a `final`
 * file unless --force.
 */

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

export const ARM_LABELS = {
  random: { id: "a-random", label: "Random search (A)", letter: "A", order: 0 },
  nemotron: { id: "c-nemotron", label: "Nemotron with feedback (C)", letter: "C", order: 1 },
  zeroshot: { id: "z-zeroshot", label: "Zero-shot Nemotron (Z)", letter: "Z", order: 2 },
};

/** Tolerance of a score reaching the threshold (as in rtt.analysis.s1). */
const EPS = 1e-9;

const round = (x) => Number(x.toFixed(4));

/** Running maximum. */
export function bestSoFar(scores) {
  let best = -Infinity;
  return scores.map((s) => (best = Math.max(best, s)));
}

/** Parse a history.jsonl body into per-design records in evaluation order. */
export function readHistory(text) {
  return text
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l))
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
    .map((row) => ({
      index: row.index,
      design_id: row.proposal?.design_id ?? row.score?.design_id ?? null,
      origin: row.proposal?.origin ?? null,
      // v2: registered success (seated and centred), not score.lift_success
      success: Number(row.score?.success),
      episodes: Number.isInteger(row.score?.n_episodes) ? row.score.n_episodes : null,
    }));
}

/**
 * S1's T for one run: the 1-based index of the first design at or after `countFrom`
 * whose own score reaches `threshold`; the run length plus one when none does.
 */
export function designsToTarget(scores, threshold, countFrom = 1) {
  for (let i = Math.max(countFrom, 1) - 1; i < scores.length; i++) {
    if (scores[i] >= threshold - EPS) return i + 1;
  }
  return scores.length + 1;
}

/** Leading positions where every run evaluated the same design. */
export function sharedPrefix(runs) {
  if (runs.length < 2) return 0;
  const len = Math.min(...runs.map((r) => r.length));
  let n = 0;
  while (n < len && runs.every((r) => r[n].design_id && r[n].design_id === runs[0][n].design_id)) n++;
  return n;
}

/**
 * Leading designs shared by paired runs: runs in the same slot (`<arm>-rNN`, so the
 * same seed) of different arms. The smallest prefix over all pairs; with no pairs, the
 * prefix every run shares. `runs` is a list of `{ name, history }`.
 */
export function pairedSharedPrefix(runs) {
  const slots = new Map();
  for (const r of runs) {
    const slot = /-(r\d+)$/.exec(r.name)?.[1] ?? r.name;
    slots.set(slot, [...(slots.get(slot) ?? []), r.history]);
  }
  const pairs = [...slots.values()].filter((h) => h.length > 1);
  if (pairs.length === 0) return sharedPrefix(runs.map((r) => r.history));
  return Math.min(...pairs.map(sharedPrefix));
}

function readJson(file) {
  return JSON.parse(readFileSync(file, "utf8"));
}

/**
 * Model calls of one run from its `llm_calls.jsonl` (Token Factory spend, which the
 * Nebius job cost does not include): `{calls, usd}`, or null when the run has no such
 * file. A random-search run has the file with no rows, so it reports 0 calls, $0.
 */
export function readLlmCalls(file) {
  if (!existsSync(file)) return null;
  const rows = readFileSync(file, "utf8")
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
  const usd = rows.reduce((s, r) => s + (Number.isFinite(Number(r.cost_usd)) ? Number(r.cost_usd) : 0), 0);
  return { calls: rows.length, usd: round(usd) };
}

/** Locate the search directory and its run histories inside a run directory. */
export function scanRunDir(dir) {
  const searchDir = existsSync(path.join(dir, "search", "manifest.json")) ? path.join(dir, "search") : dir;
  const manifestFile = path.join(searchDir, "manifest.json");
  if (!existsSync(manifestFile)) throw new Error(`${dir}: no search manifest.json`);
  const manifest = readJson(manifestFile);
  if (manifest.kind !== undefined && manifest.kind !== "search") throw new Error(`${dir}: manifest kind is ${manifest.kind}, not search`);
  const runsDir = path.join(searchDir, "runs");
  const runs = readdirSync(runsDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && existsSync(path.join(runsDir, d.name, "history.jsonl")))
    .map((d) => ({
      name: d.name,
      history: readHistory(readFileSync(path.join(runsDir, d.name, "history.jsonl"), "utf8")),
      llm: readLlmCalls(path.join(runsDir, d.name, "llm_calls.jsonl")),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
  const jobFile = path.join(dir, "job.json");
  const job = existsSync(jobFile) ? readJson(jobFile) : null;
  const sizes = Array.isArray(manifest.round_sizes) ? manifest.round_sizes : [];
  const round0 = Number.isInteger(sizes[0]) ? sizes[0] : null;
  return { manifest, runs, job, round0 };
}

/** One test of a pre-registered comparison, as the page shows it. */
function readTest(t) {
  return { statistic: Number(t.statistic), p_value: Number(t.p_value), method: String(t.method) };
}

/**
 * The confirmatory S1 block from `s1_v2.json`, after checking it against the runs this
 * script read: the same tau and budget, and every run's T equal to the analysis's
 * `t_arm`. Throws on any difference, so the page cannot show numbers the analysis
 * did not produce.
 */
export function s1AnalysisBlock(analysis, { threshold, budget, arms, source }) {
  if (analysis.format !== "rtt-s1-analysis") throw new Error(`s1 analysis: format is ${analysis.format}`);
  if (threshold === null || Math.abs(Number(analysis.tau) - threshold) > 1e-12) {
    throw new Error(`s1 analysis: tau ${analysis.tau} differs from --threshold ${threshold}`);
  }
  if (Number(analysis.budget) !== budget) throw new Error(`s1 analysis: budget ${analysis.budget} differs from ${budget}`);
  const means = {};
  for (const arm of arms) {
    const rows = analysis.runs?.[arm.letter];
    if (!Array.isArray(rows) || rows.length !== arm.runs.length) {
      throw new Error(`s1 analysis: arm ${arm.letter} has ${rows?.length ?? 0} runs, the histories ${arm.runs.length}`);
    }
    for (const run of arm.runs) {
      const row = rows.find((r) => r.run_id === run.run);
      if (!row) throw new Error(`s1 analysis: no row for ${run.run}`);
      if (row.t_arm !== run.t) throw new Error(`s1 analysis: ${run.run} has T ${row.t_arm}, the history gives ${run.t}`);
    }
    means[arm.letter] = arm.runs.reduce((s, r) => s + r.t, 0) / arm.runs.length;
  }
  const cmp = analysis.comparison;
  if (means.A !== undefined && means.C !== undefined && Math.abs(means.A - means.C - Number(cmp.primary.statistic)) > 1e-9) {
    throw new Error(`s1 analysis: mean difference ${means.A - means.C} differs from the statistic ${cmp.primary.statistic}`);
  }
  const lift = analysis.lift_only?.comparison;
  return {
    source,
    tau_source: String(analysis.tau_source ?? ""),
    primary: readTest(cmp.primary),
    effect: { diff: Number(cmp.effect.diff), ci_low: Number(cmp.effect.ci_low), ci_high: Number(cmp.effect.ci_high), level: Number(cmp.effect.level) },
    logrank: readTest(cmp.sensitivity.logrank),
    lift_only: lift
      ? { status: String(analysis.lift_only.status), diff: Number(lift.effect.diff), p_value: Number(lift.primary.p_value), method: String(lift.primary.method) }
      : null,
  };
}

const SECONDARY_LABELS = {
  H1: { label: "H1: seated and centred, 100 unseen procedural conditions x 2 seeds", unit: "rate" },
  H2: { label: "H2: seated and centred, 64 real scanned objects", unit: "rate" },
  S2: { label: "S2: area under the best-so-far curve", unit: "score" },
};

function secondaryTest(id, t) {
  const meta = SECONDARY_LABELS[id] ?? { label: id, unit: "rate" };
  return {
    id,
    label: meta.label,
    unit: meta.unit,
    mean_a: Number(t.mean_a),
    mean_c: Number(t.mean_c),
    diff: Number(t.diff_c_minus_a),
    p_value: Number(t.p),
    p_holm: t.p_holm === undefined ? null : Number(t.p_holm),
    rejected: t.rejected === undefined ? null : Boolean(t.rejected),
    method: String(t.method),
    n_a: Number(t.n_a),
    n_c: Number(t.n_c),
  };
}

/** The Holm family and the planned H2 width split from `secondary_v2.json`. */
export function secondaryBlock(doc, { source, job = null }) {
  if (doc.schema !== "rtt.secondary_v2/1") throw new Error(`secondary: schema is ${doc.schema}`);
  if (doc.confirmatory !== true) throw new Error("secondary: not marked confirmatory");
  const bands = doc.exploratory?.H2_width_bands;
  return {
    source,
    alpha: Number(doc.alpha),
    family: doc.family.map(String),
    tests: doc.family.map((id) => {
      if (!doc.tests[id]) throw new Error(`secondary: no test ${id}`);
      return secondaryTest(id, doc.tests[id]);
    }),
    width_bands: bands
      ? {
          note: String(bands.note),
          edge_mm: Number(bands.band_edge_mm),
          bands: [
            { id: "le", label: `${bands.band_edge_mm} mm or narrower across the jaw`, objects: Number(bands.objects_by_band.width_le_50mm), ...pick(bands.width_le_50mm) },
            { id: "gt", label: `wider than ${bands.band_edge_mm} mm`, objects: Number(bands.objects_by_band.width_gt_50mm), ...pick(bands.width_gt_50mm) },
          ],
        }
      : null,
    job,
  };
}

function pick(t) {
  return { mean_a: Number(t.mean_a), mean_c: Number(t.mean_c), p_value: Number(t.p) };
}

/**
 * Assemble the results document from scanned run directories.
 * `jobsIndex` is public/data/jobs.json (for costs), or null.
 *
 * @param {ReturnType<typeof scanRunDir>[]} scans
 * @param {{
 *   status: "synthetic" | "pilot" | "final",
 *   threshold: number | null,
 *   thresholdNote: string,
 *   note: string,
 *   jobsIndex?: { jobs: { id: string, cost_usd: number | null }[] } | null,
 *   h1?: object | null,
 *   s1Analysis?: { doc: object, source: string } | null,
 *   secondary?: { doc: object, source: string, job?: object | null } | null,
 *   protocol?: { tag: string, results_doc: string, protocol_doc: string } | null,
 * }} opts
 */
export function buildResults(
  scans,
  { status, threshold, thresholdNote, note, jobsIndex = null, h1 = null, s1Analysis = null, secondary = null, protocol = null },
) {
  const arms = new Map();
  const sources = [];
  const allRuns = [];
  const episodes = new Set();
  let budget = 0;
  const round0s = new Set(scans.map((s) => s.round0 ?? null));
  const round0 = round0s.size === 1 ? [...round0s][0] : null;
  const countFrom = round0 !== null && round0 !== undefined ? round0 + 1 : 1;
  for (const { manifest, runs, job } of scans) {
    const armKey = String(manifest.arm ?? "").toLowerCase();
    const meta = ARM_LABELS[armKey] ?? { id: armKey.replace(/[^a-z0-9._-]/g, "-") || "arm", label: armKey || "Arm", letter: null, order: 9 };
    budget = Math.max(budget, Number(manifest.budget) || 0);
    const nebiusJob = job?.job_id ? job : null;
    const publishedJob = nebiusJob && jobsIndex ? jobsIndex.jobs.find((j) => j.id === nebiusJob.job_id) : null;
    for (const run of runs) {
      const scores = run.history.map((h) => h.success);
      if (scores.some((s) => !Number.isFinite(s))) throw new Error(`${run.name}: a design has no numeric score.success`);
      for (const h of run.history) episodes.add(h.episodes);
      allRuns.push(run);
      const entry = arms.get(meta.id) ?? { id: meta.id, label: meta.label, letter: meta.letter, order: meta.order, runs: [] };
      const row = { run: run.name, best: bestSoFar(scores), job_id: nebiusJob ? nebiusJob.job_id : null };
      if (threshold !== null) row.t = designsToTarget(scores, threshold, countFrom);
      entry.runs.push(row);
      arms.set(meta.id, entry);
      sources.push({
        arm: meta.id,
        run: run.name,
        kind: nebiusJob ? "nebius" : "local",
        job_id: nebiusJob ? nebiusJob.job_id : null,
        platform: nebiusJob?.platform ?? manifest.job?.platform ?? null,
        preset: nebiusJob?.preset ?? manifest.job?.preset ?? null,
        region: manifest.job?.region ?? null,
        executor: manifest.executor ?? null,
        wall_s: typeof nebiusJob?.wall_s === "number" ? round(nebiusJob.wall_s) : null,
        ...(typeof nebiusJob?.run_s === "number" ? { run_s: round(nebiusJob.run_s) } : {}),
        ...(manifest.job?.image_digest ? { image_digest: String(manifest.job.image_digest) } : {}),
        cost_usd: typeof publishedJob?.cost_usd === "number" ? publishedJob.cost_usd : null,
        llm_calls: run.llm ? run.llm.calls : null,
        llm_usd: run.llm ? run.llm.usd : null,
      });
    }
  }
  if (arms.size === 0) throw new Error("no search runs found");
  const sorted = [...arms.values()].sort((a, b) => a.order - b.order);
  const s1Arms = sorted.map(({ order, letter, ...arm }) => arm);
  const trials = episodes.size === 1 && [...episodes][0] !== null ? [...episodes][0] : null;
  const s1 = {
    endpoint: "S1: simulated designs needed to reach the training-success threshold",
    threshold,
    threshold_note: thresholdNote,
    budget,
    shared_prefix: pairedSharedPrefix(allRuns),
    ...(countFrom > 1 ? { count_from: countFrom } : {}),
    ...(trials !== null ? { trials } : {}),
    arms: s1Arms,
  };
  if (s1Analysis) s1.analysis = s1AnalysisBlock(s1Analysis.doc, { threshold, budget, arms: sorted, source: s1Analysis.source });
  const doc = { schema: "rtt-web-results", version: 2, status, note };
  if (protocol) doc.protocol = protocol;
  doc.s1 = s1;
  if (secondary) doc.secondary = secondaryBlock(secondary.doc, { source: secondary.source, job: secondary.job ?? null });
  else doc.h1 = h1 ?? { endpoint: "H1: run-level held-out success rate", status: "not_run", arms: [], references: [] };
  doc.sources = sources;
  return doc;
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      status: { type: "string", default: "pilot" },
      threshold: { type: "string" },
      "threshold-note": { type: "string", default: "Provisional" },
      note: { type: "string", default: "" },
      h1: { type: "string" },
      "s1-analysis": { type: "string" },
      secondary: { type: "string" },
      "secondary-job": { type: "string" },
      "prereg-tag": { type: "string", default: "prereg-v1" },
      force: { type: "boolean", default: false },
      out: { type: "string", default: path.join(webDir, "public", "data", "results.json") },
    },
  });
  if (!["synthetic", "pilot", "final"].includes(values.status)) {
    console.error(`import-results: --status must be synthetic, pilot or final`);
    process.exit(2);
  }
  if (positionals.length === 0) {
    console.error("import-results: give at least one search run directory");
    process.exit(2);
  }
  const threshold = values.threshold === undefined ? null : Number(values.threshold);
  if (threshold !== null && !(threshold >= 0 && threshold <= 1)) {
    console.error("import-results: --threshold must be in [0, 1]");
    process.exit(2);
  }
  if (existsSync(values.out) && !values.force) {
    const current = readJson(values.out);
    if (current.status === "final") {
      console.error(`import-results: ${values.out} holds final results; pass --force to replace them`);
      process.exit(1);
    }
  }
  const jobsFile = path.join(webDir, "public", "data", "jobs.json");
  const repoDir = path.resolve(webDir, "..");
  const rel = (file) => path.relative(repoDir, path.resolve(file)).split(path.sep).join("/");
  const secondaryJob = values["secondary-job"] ? readJson(values["secondary-job"]) : null;
  const doc = buildResults(positionals.map(scanRunDir), {
    status: values.status,
    threshold,
    thresholdNote: values["threshold-note"],
    note: values.note,
    jobsIndex: existsSync(jobsFile) ? readJson(jobsFile) : null,
    h1: values.h1 ? readJson(values.h1).h1 : null,
    s1Analysis: values["s1-analysis"] ? { doc: readJson(values["s1-analysis"]), source: rel(values["s1-analysis"]) } : null,
    secondary: values.secondary
      ? {
          doc: readJson(values.secondary),
          source: rel(values.secondary),
          job: secondaryJob?.job_id
            ? { job_id: secondaryJob.job_id, platform: secondaryJob.platform ?? null, region: secondaryJob.region ?? null, run_s: typeof secondaryJob.run_s === "number" ? round(secondaryJob.run_s) : null }
            : null,
        }
      : null,
    protocol: values.status === "final" ? { tag: values["prereg-tag"], results_doc: "docs/RESULTS.md", protocol_doc: "docs/PREREG.md" } : null,
  });
  writeFileSync(values.out, `${JSON.stringify(doc, null, 1)}\n`, "utf8");
  const runs = doc.s1.arms.map((a) => `${a.id} x${a.runs.length}`).join(", ");
  console.log(`import-results: wrote ${path.relative(webDir, values.out)} (${values.status}; ${runs}; shared prefix ${doc.s1.shared_prefix})`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
