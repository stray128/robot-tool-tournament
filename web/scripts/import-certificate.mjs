#!/usr/bin/env node
/**
 * Publish a robustness certificate written by `rtt certify --out FILE.json` to
 * public/data/certificates/<id>.json, adding where it ran.
 *
 *   node scripts/import-certificate.mjs --in FILE.json [--id ID]
 *        [--executor cpu] [--engine "MuJoCo 3.15.0"] [--host workstation] [--wall-s 13.2]
 *        [--job-id ID --platform P --preset P --region R --cost-usd 0.01]
 *
 * Without --job-id the certificate is labelled a local run; with it, a Nebius job
 * (platform, preset and region are then required). The id defaults to the design id.
 * The /certificate/<id> page is generated for every file in that directory.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { findLocalPaths, findPrivateIds } from "./import-session.mjs";

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;

/**
 * The certificate document with a `provenance` block (ComputeProvenance, lib/types.ts).
 * Throws on a document that is not an `rtt certify` output or on incomplete Nebius facts.
 */
export function withProvenance(doc, opts) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc)) throw new Error("certificate: expected a JSON object");
  if (doc.schema !== undefined && doc.schema !== "rtt.certificates/1") throw new Error(`certificate: unknown schema ${doc.schema}`);
  if (typeof doc.design_id !== "string" || !doc.design_id) throw new Error("certificate: missing design_id");
  if (!Array.isArray(doc.certificates) || doc.certificates.length === 0) throw new Error("certificate: no certificates");
  const num = (v) => (v === undefined || v === null || v === "" ? null : Number(v));
  let provenance;
  if (opts.jobId) {
    for (const k of ["platform", "preset", "region"]) {
      if (!opts[k]) throw new Error(`certificate: --${k} is required with --job-id`);
    }
    provenance = {
      kind: "nebius",
      job_id: opts.jobId,
      platform: opts.platform,
      preset: opts.preset,
      region: opts.region,
      executor: opts.executor ?? "cpu",
      engine: opts.engine ?? "MuJoCo",
      wall_s: num(opts.wallS) ?? 0,
      cost_usd: num(opts.costUsd) ?? 0,
      image_digest: opts.imageDigest ?? null,
    };
  } else {
    provenance = {
      kind: "local",
      executor: opts.executor ?? "cpu",
      engine: opts.engine ?? "MuJoCo",
      host: opts.host ?? "workstation",
      wall_s: num(opts.wallS),
      sim_steps: null,
      note: opts.note ?? null,
    };
  }
  const out = { ...doc, provenance };
  // Same refusals as a published session: no local paths, no Nebius resource IDs other
  // than the AI job ID (which is shown as provenance), no registry paths.
  if (opts.jobId && !/^aijob-[a-z0-9]+$/i.test(opts.jobId)) throw new Error("certificate: --job-id must be a Nebius AI job id (aijob-...)");
  const leaks = findLocalPaths(out);
  if (leaks.length) throw new Error(`certificate: refusing to publish local paths:\n  ${leaks.join("\n  ")}`);
  const ids = findPrivateIds(out);
  if (ids.length) throw new Error(`certificate: refusing to publish Nebius resource IDs or registry paths at:\n  ${ids.join("\n  ")}`);
  return out;
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({
    options: {
      in: { type: "string" },
      id: { type: "string" },
      executor: { type: "string" },
      engine: { type: "string" },
      host: { type: "string" },
      note: { type: "string" },
      "wall-s": { type: "string" },
      "job-id": { type: "string" },
      platform: { type: "string" },
      preset: { type: "string" },
      region: { type: "string" },
      "cost-usd": { type: "string" },
      "image-digest": { type: "string" },
      out: { type: "string", default: path.join(webDir, "public", "data", "certificates") },
    },
  });
  if (!values.in) {
    console.error("import-certificate: --in FILE.json is required");
    process.exit(2);
  }
  const doc = JSON.parse(readFileSync(values.in, "utf8"));
  const out = withProvenance(doc, {
    executor: values.executor,
    engine: values.engine,
    host: values.host,
    note: values.note,
    wallS: values["wall-s"],
    jobId: values["job-id"],
    platform: values.platform,
    preset: values.preset,
    region: values.region,
    costUsd: values["cost-usd"],
    imageDigest: values["image-digest"],
  });
  const id = values.id ?? doc.design_id;
  if (!SLUG.test(id)) {
    console.error(`import-certificate: id ${id} is not URL-safe`);
    process.exit(2);
  }
  mkdirSync(values.out, { recursive: true });
  const file = path.join(values.out, `${id}.json`);
  writeFileSync(file, `${JSON.stringify(out, null, 1)}\n`, "utf8");
  console.log(`import-certificate: wrote ${path.relative(webDir, file)}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
