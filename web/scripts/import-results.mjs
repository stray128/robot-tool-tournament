#!/usr/bin/env node
/**
 * Build public/data/results.json (version 2) from search run directories.
 *
 *   node scripts/import-results.mjs --status pilot --threshold 0.8 \
 *        --threshold-note "Provisional ..." --note "..." RUN_DIR [RUN_DIR ...]
 *
 * A RUN_DIR is a search output (`rtt search`), or a cloud run directory holding one
 * under `search/` plus the Nebius `job.json`. For every `runs/<arm>-rNN/history.jsonl`
 * it takes each design's training success (`score.success`), in evaluation order, and
 * writes the best-so-far curve: S1, the primary endpoint. Runs are grouped into arms
 * by the manifest's `arm`. Leading designs that every run shares (entrants and seed
 * designs) are counted as `shared_prefix` so the chart can mark them.
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
  random: { id: "a-random", label: "Random search (A)", order: 0 },
  nemotron: { id: "c-nemotron", label: "Nemotron with feedback (C)", order: 1 },
  zeroshot: { id: "z-zeroshot", label: "Zero-shot Nemotron (Z)", order: 2 },
};

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
      success: Number(row.score?.success),
    }));
}

/** Leading positions where every run evaluated the same design. */
export function sharedPrefix(runs) {
  if (runs.length < 2) return 0;
  const len = Math.min(...runs.map((r) => r.length));
  let n = 0;
  while (n < len && runs.every((r) => r[n].design_id && r[n].design_id === runs[0][n].design_id)) n++;
  return n;
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
  return { manifest, runs, job };
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
 * }} opts
 */
export function buildResults(scans, { status, threshold, thresholdNote, note, jobsIndex = null, h1 = null }) {
  const arms = new Map();
  const sources = [];
  const allHistories = [];
  let budget = 0;
  for (const { manifest, runs, job } of scans) {
    const armKey = String(manifest.arm ?? "").toLowerCase();
    const meta = ARM_LABELS[armKey] ?? { id: armKey.replace(/[^a-z0-9._-]/g, "-") || "arm", label: armKey || "Arm", order: 9 };
    budget = Math.max(budget, Number(manifest.budget) || 0);
    const nebiusJob = job?.job_id ? job : null;
    const publishedJob = nebiusJob && jobsIndex ? jobsIndex.jobs.find((j) => j.id === nebiusJob.job_id) : null;
    for (const run of runs) {
      const scores = run.history.map((h) => h.success);
      if (scores.some((s) => !Number.isFinite(s))) throw new Error(`${run.name}: a design has no numeric score.success`);
      allHistories.push(run.history);
      const entry = arms.get(meta.id) ?? { id: meta.id, label: meta.label, order: meta.order, runs: [] };
      entry.runs.push({ run: run.name, best: bestSoFar(scores).map(round), job_id: nebiusJob ? nebiusJob.job_id : null });
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
        cost_usd: typeof publishedJob?.cost_usd === "number" ? publishedJob.cost_usd : null,
        llm_calls: run.llm ? run.llm.calls : null,
        llm_usd: run.llm ? run.llm.usd : null,
      });
    }
  }
  if (arms.size === 0) throw new Error("no search runs found");
  const s1Arms = [...arms.values()].sort((a, b) => a.order - b.order).map(({ order, ...arm }) => arm);
  return {
    schema: "rtt-web-results",
    version: 2,
    status,
    note,
    s1: {
      endpoint: "S1: simulated designs needed to reach the training-success threshold",
      threshold,
      threshold_note: thresholdNote,
      budget,
      shared_prefix: sharedPrefix(allHistories),
      arms: s1Arms,
    },
    h1: h1 ?? { endpoint: "H1: run-level held-out success rate", status: "not_run", arms: [], references: [] },
    sources,
  };
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
  const doc = buildResults(positionals.map(scanRunDir), {
    status: values.status,
    threshold,
    thresholdNote: values["threshold-note"],
    note: values.note,
    jobsIndex: existsSync(jobsFile) ? readJson(jobsFile) : null,
    h1: values.h1 ? readJson(values.h1).h1 : null,
  });
  writeFileSync(values.out, `${JSON.stringify(doc, null, 1)}\n`, "utf8");
  const runs = doc.s1.arms.map((a) => `${a.id} x${a.runs.length}`).join(", ");
  console.log(`import-results: wrote ${path.relative(webDir, values.out)} (${values.status}; ${runs}; shared prefix ${doc.s1.shared_prefix})`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
