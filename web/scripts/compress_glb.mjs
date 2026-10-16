#!/usr/bin/env node
/**
 * Compress GLBs for the site and content-address them (docs/ASSET_MANIFEST.md).
 *
 *   node scripts/compress_glb.mjs SPEC.json [--out-dir DIR] [--manifest FILE] [--map FILE]
 *                                 [--no-webp] [--modules DIR]
 *   node scripts/compress_glb.mjs --check [--out-dir DIR] [--manifest FILE]
 *
 * SPEC.json lists the uncompressed GLBs (written by the render track's exporter) and
 * the manifest fields for each:
 *
 *   { "inputs": [ { "key": "tee/left", "file": "raw/tee_left.glb", "role": "finger_left",
 *                   "license": "Apache-2.0", "attribution": "Robot Tool Tournament (Apache-2.0)",
 *                   "changes": [], "label": "pipe tee, left jaw",
 *                   "source": { "kind": "generated", "generator": "rtt.render.glb" },
 *                   "simplify": 0.5, "brand_visible": false } ] }
 *
 * `file` is relative to SPEC.json. `simplify` (optional, 0..1) keeps that share of the
 * triangles (scans use 0.5); omit it for exact CAD and jaw geometry. Each input is
 * deduplicated, welded, optionally simplified, given WebP textures and
 * meshopt-compressed (EXT_meshopt_compression with KHR_mesh_quantization). Geometry
 * stays in the body frame, Z up, metres: nothing is rotated or mirrored. The output is
 * written to `<out-dir>/<id>.glb`, where id is the first 16 hex characters of its
 * sha256, and its entry is added to the manifest. `--map` writes {key: id} so the
 * replay writer can refer to assets by stable key.
 *
 * `--check` needs no dependencies: it verifies that every manifest entry's file exists
 * with the listed size and sha256, that ids and uris follow the content-addressing
 * rule, that third-party assets list their changes, and that every GLB in the output
 * directory is listed. It exits 1 on any problem.
 *
 * Compression needs @gltf-transform/core, @gltf-transform/extensions,
 * @gltf-transform/functions and meshoptimizer (and sharp for WebP textures). They are
 * looked up in web/node_modules, or in `--modules DIR` (a node_modules directory) when
 * they are not dependencies of web/ yet.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROLES = ["finger_left", "finger_right", "gripper_frame", "carriage", "object", "prop"];
const OWN_LICENSE = "Apache-2.0";
const ID_RE = /^[0-9a-f]{16}$/;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "out-dir": { type: "string", default: path.join(webDir, "public", "assets") },
    manifest: { type: "string" },
    map: { type: "string" },
    "no-webp": { type: "boolean", default: false },
    modules: { type: "string" },
    check: { type: "boolean", default: false },
  },
});
const outDir = path.resolve(values["out-dir"]);
const manifestPath = path.resolve(values.manifest ?? path.join(outDir, "manifest.json"));

function fail(message) {
  console.error(`compress_glb: ${message}`);
  process.exit(1);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function readManifest() {
  if (!existsSync(manifestPath)) return { format: "rtt-asset-manifest", version: 1, assets: {} };
  const m = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (m.format !== "rtt-asset-manifest" || m.version !== 1) fail(`${manifestPath} is not an rtt-asset-manifest v1`);
  return m;
}

/** Stable output: ids sorted, keys in the manifest doc's order, two-space indent. */
function writeManifest(m) {
  const order = ["uri", "sha256", "bytes", "role", "license", "attribution", "changes", "label", "media_type", "brand_visible", "source"];
  const assets = {};
  for (const id of Object.keys(m.assets).sort()) {
    const e = m.assets[id];
    assets[id] = Object.fromEntries(order.filter((k) => e[k] !== undefined).map((k) => [k, e[k]]));
  }
  writeFileSync(manifestPath, `${JSON.stringify({ format: "rtt-asset-manifest", version: 1, assets }, null, 2)}\n`);
}

/** Problems with a manifest entry, independent of the file. */
function entryProblems(id, e) {
  const p = [];
  if (!ID_RE.test(id)) p.push(`${id}: id is not 16 lowercase hex characters`);
  if (e.uri !== `assets/${id}.glb`) p.push(`${id}: uri must be assets/${id}.glb, got ${e.uri}`);
  if (typeof e.sha256 !== "string" || !e.sha256.startsWith(id)) p.push(`${id}: sha256 does not start with the id`);
  if (!ROLES.includes(e.role)) p.push(`${id}: unknown role ${e.role}`);
  if (typeof e.license !== "string" || !/^[A-Za-z0-9][A-Za-z0-9.+-]*$/.test(e.license)) p.push(`${id}: license must be an SPDX id`);
  if (typeof e.attribution !== "string" || !e.attribution) p.push(`${id}: attribution is missing`);
  if (!Array.isArray(e.changes) || e.changes.some((c) => typeof c !== "string" || !c)) p.push(`${id}: changes must be a list of strings`);
  else if (e.license !== OWN_LICENSE && e.changes.length === 0) p.push(`${id}: third-party asset (${e.license}) must list its changes`);
  if (!Number.isInteger(e.bytes) || e.bytes < 1) p.push(`${id}: bytes must be a positive integer`);
  return p;
}

function check() {
  const m = readManifest();
  const problems = [];
  for (const [id, e] of Object.entries(m.assets)) {
    problems.push(...entryProblems(id, e));
    const file = path.join(outDir, `${id}.glb`);
    if (!existsSync(file)) {
      problems.push(`${id}: ${path.relative(webDir, file)} is missing`);
      continue;
    }
    const bytes = readFileSync(file);
    if (bytes.byteLength !== e.bytes) problems.push(`${id}: file is ${bytes.byteLength} bytes, entry says ${e.bytes}`);
    if (sha256(bytes) !== e.sha256) problems.push(`${id}: file does not match its sha256`);
  }
  if (existsSync(outDir)) {
    for (const f of readdirSync(outDir).filter((f) => f.endsWith(".glb"))) {
      if (!(f.slice(0, -4) in m.assets)) problems.push(`${f}: served but not listed in the manifest`);
    }
  }
  for (const p of problems) console.error(`compress_glb --check: ${p}`);
  console.log(`compress_glb --check: ${Object.keys(m.assets).length} assets, ${problems.length} problems`);
  process.exit(problems.length ? 1 : 0);
}

/** Import a package from web/node_modules, or from --modules DIR. */
async function load(name) {
  try {
    if (values.modules) {
      const require = createRequire(path.join(path.resolve(values.modules), "noop.js"));
      return await import(pathToFileURL(require.resolve(name)).href);
    }
    return await import(name);
  } catch (err) {
    fail(
      `cannot load ${name} (${err.code ?? err.message}). Compression needs @gltf-transform/core, ` +
        "@gltf-transform/extensions, @gltf-transform/functions and meshoptimizer (sharp for WebP textures) " +
        "as devDependencies of web/, or --modules pointing at a node_modules directory that has them.",
    );
  }
}

async function compress() {
  if (positionals.length !== 1) fail("usage: compress_glb.mjs SPEC.json [options] (or --check)");
  const specPath = path.resolve(positionals[0]);
  const spec = JSON.parse(readFileSync(specPath, "utf8"));
  if (!Array.isArray(spec.inputs) || spec.inputs.length === 0) fail(`${specPath}: expected a non-empty "inputs" list`);

  const core = await load("@gltf-transform/core");
  const { ALL_EXTENSIONS } = await load("@gltf-transform/extensions");
  const fn = await load("@gltf-transform/functions");
  const meshoptimizer = await load("meshoptimizer");
  const { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } = meshoptimizer.default ?? meshoptimizer;
  await Promise.all([MeshoptEncoder.ready, MeshoptDecoder.ready, MeshoptSimplifier.ready]);
  const io = new (core.NodeIO ?? core.default.NodeIO)()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({ "meshopt.decoder": MeshoptDecoder, "meshopt.encoder": MeshoptEncoder });
  let sharp = null;

  const manifest = readManifest();
  const map = {};
  mkdirSync(outDir, { recursive: true });
  for (const input of spec.inputs) {
    const where = `input ${JSON.stringify(input.key ?? input.file)}`;
    if (!input.key || !input.file) fail(`${where}: needs "key" and "file"`);
    const raw = readFileSync(path.resolve(path.dirname(specPath), input.file));
    const doc = await io.readBinary(new Uint8Array(raw));
    const steps = [fn.dedup(), fn.weld()];
    if (input.simplify !== undefined) {
      if (!(input.simplify > 0 && input.simplify <= 1)) fail(`${where}: simplify must be in (0, 1]`);
      steps.push(fn.simplify({ simplifier: MeshoptSimplifier, ratio: input.simplify, error: 0.001 }));
    }
    const hasTextures = doc.getRoot().listTextures().length > 0;
    if (hasTextures && !values["no-webp"]) {
      sharp ??= (await load("sharp")).default;
      steps.push(fn.textureCompress({ encoder: sharp, targetFormat: "webp", quality: 85 }));
    }
    steps.push(fn.prune(), fn.meshopt({ encoder: MeshoptEncoder, level: "medium" }));
    await doc.transform(...steps);
    const out = await io.writeBinary(doc);
    const sha = sha256(out);
    const id = sha.slice(0, 16);
    writeFileSync(path.join(outDir, `${id}.glb`), out);
    const entry = {
      uri: `assets/${id}.glb`,
      sha256: sha,
      bytes: out.byteLength,
      role: input.role,
      license: input.license,
      attribution: input.attribution,
      changes: [...(input.changes ?? [])],
      label: input.label,
      media_type: "model/gltf-binary",
      brand_visible: input.brand_visible,
      source: input.source,
    };
    // Re-encoding is a change from the upstream file; say so for third-party assets.
    if (entry.license !== OWN_LICENSE && !entry.changes.some((c) => /re-encoded/i.test(c))) entry.changes.push("re-encoded as GLB");
    if (input.simplify !== undefined && input.simplify < 1 && entry.license !== OWN_LICENSE && !entry.changes.some((c) => /simplif/i.test(c))) {
      entry.changes.push(`simplified to ${Math.round(input.simplify * 100)}%`);
    }
    const problems = entryProblems(id, entry);
    if (problems.length) fail(`${where}: ${problems.join("; ")}`);
    manifest.assets[id] = entry;
    map[input.key] = id;
    console.log(`${input.key}: ${raw.byteLength} -> ${out.byteLength} bytes, assets/${id}.glb`);
  }
  writeManifest(manifest);
  if (values.map) writeFileSync(path.resolve(values.map), `${JSON.stringify(map, null, 2)}\n`);
}

if (values.check) check();
else await compress();
