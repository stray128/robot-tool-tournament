#!/usr/bin/env node
/**
 * Regenerate the replay v2 test fixture from a recorded v1 replay.
 *
 *   node tests/fixtures/replay_v2/make_fixture.mjs [--compress] [--modules DIR]
 *
 * Input: public/data/replays/m1/straight.{json,bin}, the recorded M1 episode in which
 * the straight entrant lifts and holds the box. Output, next to this script:
 *   sample.json        replay header, version 2 (src/rtt/sim/replay_v2.schema.json)
 *   sample.bin         the v1 pose payload, byte for byte (the layout did not change)
 *   sample.ovl.bin     float16 overlays derived from those poses: offset_mm (the
 *                      object's horizontal distance from the palm axis) and hold_s
 *   assets/<id>.glb    one GLB per body, made from exactly that body's collision geoms,
 *                      content addressed (id = sha256[:16]), plus assets/manifest.json
 *
 * With --compress the GLBs go through scripts/compress_glb.mjs (meshopt), as served
 * assets do; --modules is passed on to it. Without it they are written uncompressed.
 *
 * The poses are real simulation output; only the container changes. The fixture is
 * for tests and the headless page check (scripts/smoke_v2.mjs), never for the site.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { boxMesh, flatMesh, hexToLinear, writeGlb } from "../../helpers/glb_writer.mjs";

const { values: opts } = parseArgs({ options: { compress: { type: "boolean", default: false }, modules: { type: "string" } } });
const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(here, "..", "..", "..");
const srcDir = path.join(web, "public", "data", "replays", "m1");
const v1 = JSON.parse(readFileSync(path.join(srcDir, "straight.json"), "utf8"));
const poses = readFileSync(path.join(srcDir, v1.bin));

const PARKED = 2e-4;
const parked = (g) => g.type === "box" && g.half_size.every((h) => Math.abs(h) <= PARKED);

const MATERIALS = {
  finger: { name: "pla_orange", baseColorFactor: hexToLinear("#e8742a"), metallicFactor: 0, roughnessFactor: 0.55 },
  palm: { name: "anodised_dark", baseColorFactor: hexToLinear("#3a3f47"), metallicFactor: 0.3, roughnessFactor: 0.45 },
  object: { name: "aluminium", baseColorFactor: hexToLinear("#c3c7cc"), metallicFactor: 0.25, roughnessFactor: 0.4 },
};
const ROLES = { palm: "gripper_frame", finger_left: "finger_left", finger_right: "finger_right", object: "object" };
const LABELS = {
  palm: "palm block",
  finger_left: "straight entrant, left jaw",
  finger_right: "straight entrant, right jaw",
  object: "box 90 x 40 x 120 mm",
};

function geomPrimitive(g) {
  if (g.type === "box") return boxMesh(g.half_size, g.pos, g.quat_xyzw);
  if (g.type === "mesh") {
    const v = [];
    for (let i = 0; i < g.vertices.length; i += 3) v.push(g.vertices.slice(i, i + 3));
    const f = [];
    for (let i = 0; i < g.faces.length; i += 3) f.push(g.faces.slice(i, i + 3));
    return flatMesh(v, f, g.pos, g.quat_xyzw);
  }
  throw new Error(`fixture: geom type ${g.type} is not converted`);
}

rmSync(path.join(here, "assets"), { recursive: true, force: true });
mkdirSync(path.join(here, "assets"), { recursive: true });
const raw = mkdtempSync(path.join(os.tmpdir(), "rtt-fixture-"));

const inputs = v1.bodies.map((body) => {
  const geoms = body.geoms.filter((g) => !parked(g));
  const kind = body.name.startsWith("finger") ? "finger" : body.name === "palm" ? "palm" : "object";
  const glb = new Uint8Array(
    writeGlb({
      meshes: [{ name: body.name, primitives: geoms.map((g) => ({ ...geomPrimitive(g), material: 0 })) }],
      materials: [MATERIALS[kind]],
    }),
  );
  writeFileSync(path.join(raw, `${body.name}.glb`), glb);
  return {
    key: body.name,
    file: `${body.name}.glb`,
    role: ROLES[body.name],
    license: "Apache-2.0",
    attribution: "Robot Tool Tournament (Apache-2.0)",
    changes: [],
    label: LABELS[body.name],
    source: { kind: "generated", generator: "tests/fixtures/replay_v2/make_fixture.mjs" },
  };
});

const manifestPath = path.join(here, "assets", "manifest.json");
let ids;
if (opts.compress) {
  writeFileSync(path.join(raw, "spec.json"), JSON.stringify({ inputs }));
  const args = [path.join(web, "scripts", "compress_glb.mjs"), path.join(raw, "spec.json"), "--out-dir", path.join(here, "assets")];
  args.push("--map", path.join(raw, "map.json"), ...(opts.modules ? ["--modules", opts.modules] : []));
  execFileSync(process.execPath, args, { stdio: "inherit" });
  ids = JSON.parse(readFileSync(path.join(raw, "map.json"), "utf8"));
} else {
  ids = {};
  const assets = {};
  for (const input of inputs) {
    const glb = readFileSync(path.join(raw, input.file));
    const sha = createHash("sha256").update(glb).digest("hex");
    const id = sha.slice(0, 16);
    writeFileSync(path.join(here, "assets", `${id}.glb`), glb);
    const { key, file, ...fields } = input;
    assets[id] = { uri: `assets/${id}.glb`, sha256: sha, bytes: glb.byteLength, ...fields, media_type: "model/gltf-binary" };
    ids[key] = id;
  }
  writeFileSync(manifestPath, `${JSON.stringify({ format: "rtt-asset-manifest", version: 1, assets }, null, 2)}\n`);
}
rmSync(raw, { recursive: true, force: true });
const assets = JSON.parse(readFileSync(manifestPath, "utf8")).assets;
const bodies = v1.bodies.map((body) => ({
  name: body.name,
  visuals: [{ asset: ids[body.name], pos: [0, 0, 0], quat_xyzw: [0, 0, 0, 1], scale: [1, 1, 1] }],
  geoms: body.geoms.filter((g) => !parked(g)),
}));

// Overlays from the poses: offset of the object from the palm axis, and time held above the success height.
const nB = v1.n_bodies;
const nF = v1.n_frames;
const f32 = new Float32Array(poses.buffer.slice(poses.byteOffset, poses.byteOffset + poses.byteLength));
const palm = v1.bodies.findIndex((b) => b.name === "palm");
const obj = v1.bodies.findIndex((b) => b.name === "object");
const at = (f, b, k) => f32[(f * nB + b) * 7 + k];
const ref = Math.round(0.3 * v1.fps); // start of DESCEND, the scorer's reference frame
const z0 = at(ref, obj, 2);
const channels = ["offset_mm", "hold_s"];
const half = new Uint16Array(nF * channels.length);
const toHalf = (x) => {
  // round-to-nearest float16 for the non-negative values used here
  if (x < 2 ** -14) return 0;
  const e = Math.floor(Math.log2(x));
  const exp = Math.min(Math.max(e + 15, 1), 30);
  const frac = Math.round((x / 2 ** (exp - 15) - 1) * 1024);
  return frac === 1024 ? ((exp + 1) << 10) : (exp << 10) | frac;
};
let held = 0;
for (let f = 0; f < nF; f++) {
  const dx = at(f, obj, 0) - at(f, palm, 0);
  const dy = at(f, obj, 1) - at(f, palm, 1);
  const lift = at(f, obj, 2) - z0;
  held = lift >= 0.1 ? held + 1 / v1.fps : 0;
  half[f * 2] = toHalf(Math.hypot(dx, dy) * 1000);
  half[f * 2 + 1] = toHalf(held);
}

const header = {
  format: "rtt-replay",
  version: 2,
  fps: v1.fps,
  dtype: "float32",
  byte_order: "little",
  layout: ["frames", "bodies", "pose"],
  pose_fields: ["px", "py", "pz", "qx", "qy", "qz", "qw"],
  up_axis: "z",
  units: "m",
  n_frames: nF,
  n_bodies: nB,
  bin: "sample.bin",
  assets,
  bodies,
  static: v1.static.filter((g) => g.type !== "plane"),
  meta: {
    design_space: "v1-boxes",
    design_id: v1.meta.design_id,
    geometry_id: null,
    part_id: "box_90x40x120",
    success: v1.meta.success,
    failure_mode: v1.meta.failure_mode,
    entrant: v1.meta.entrant,
  },
  provenance: {
    source: "simulation",
    engine: "mujoco",
    engine_version: "3.15.0",
    run_id: "demo-m1",
    code_version: "b184c81",
    product_mode: false,
    episode: "straight",
    seed: v1.meta.seed,
  },
  overlays: {
    bin: "sample.ovl.bin",
    dtype: "float16",
    byte_order: "little",
    layout: ["frames", "channels"],
    channels,
  },
};

writeFileSync(path.join(here, "sample.json"), `${JSON.stringify(header, null, 1)}\n`);
writeFileSync(path.join(here, "sample.bin"), poses);
writeFileSync(path.join(here, "sample.ovl.bin"), new Uint8Array(half.buffer));
console.log(`wrote sample.json (${Object.keys(assets).length} assets, ${nF} frames) to ${path.relative(web, here)}`);
