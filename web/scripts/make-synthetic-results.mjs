#!/usr/bin/env node
/**
 * Write a synthetic public/data/results.json so the /results charts can be laid out
 * before any real result exists.
 *
 *   node scripts/make-synthetic-results.mjs [--force] [--out FILE]
 *
 * The file is marked `status: "synthetic"` and the page stamps every chart with
 * "Synthetic layout data, not a result". Both arms are drawn from the SAME
 * distribution (no effect), so the layout implies no finding. The real file comes
 * from the pre-registered analysis (`make analyze` -> export-web) and replaces this
 * one; this script refuses to overwrite a non-synthetic file even with --force.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

/** Small deterministic PRNG (mulberry32). */
export function prng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rand) {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const round = (x) => Number(x.toFixed(4));

function bootstrapCi(values, rand, draws = 2000) {
  const means = [];
  for (let d = 0; d < draws; d++) {
    let s = 0;
    for (let i = 0; i < values.length; i++) s += values[Math.floor(rand() * values.length)];
    means.push(s / values.length);
  }
  means.sort((a, b) => a - b);
  return [means[Math.floor(0.025 * draws)], means[Math.floor(0.975 * draws)]];
}

export function syntheticResults({ runs = 10, budget = 40, seed = 20261015 } = {}) {
  const rand = prng(seed);
  const arms = [
    { id: "a-random", label: "Random search (A)" },
    { id: "c-nemotron", label: "Nemotron with feedback (C)" },
  ];
  const forestArms = arms.map((arm) => {
    const values = Array.from({ length: runs }, () => round(clamp01(0.5 + 0.12 * normal(rand))));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const [lo, hi] = bootstrapCi(values, rand);
    return { ...arm, runs: values, mean: round(mean), ci95: [round(lo), round(hi)] };
  });
  const curveArms = arms.map((arm) => ({
    ...arm,
    runs: Array.from({ length: runs }, (_, i) => {
      let best = 0;
      const curve = [];
      for (let e = 0; e < budget; e++) {
        best = Math.max(best, clamp01(0.3 + 0.18 * normal(rand)));
        curve.push(round(best));
      }
      return { run: `run-${String(i + 1).padStart(2, "0")}`, best: curve };
    }),
  }));
  return {
    schema: "rtt-web-results",
    version: 1,
    status: "synthetic",
    note: "Both arms are drawn from one distribution, so the layout implies no finding. The pre-registered analysis replaces this file.",
    forest: {
      endpoint: "Run-level held-out success rate",
      arms: forestArms,
      references: [
        { label: "Straight entrant", value: round(clamp01(0.45 + 0.1 * normal(rand))) },
        { label: "Concave entrant", value: round(clamp01(0.45 + 0.1 * normal(rand))) },
        { label: "Hooked entrant", value: round(clamp01(0.45 + 0.1 * normal(rand))) },
      ],
    },
    curves: { endpoint: "Best training success so far", budget, arms: curveArms },
  };
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({
    options: {
      force: { type: "boolean", default: false },
      out: { type: "string", default: path.join(webDir, "public", "data", "results.json") },
    },
  });
  if (existsSync(values.out)) {
    const current = JSON.parse(readFileSync(values.out, "utf8"));
    if (current.status !== "synthetic") {
      console.error(`make-synthetic-results: ${values.out} holds ${current.status} results; refusing to overwrite`);
      process.exit(1);
    }
    if (!values.force) {
      console.log(`make-synthetic-results: ${values.out} exists (pass --force to regenerate)`);
      return;
    }
  }
  writeFileSync(values.out, `${JSON.stringify(syntheticResults(), null, 1)}\n`, "utf8");
  console.log(`make-synthetic-results: wrote ${values.out}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
