/**
 * Replay decoding and sampling (format rtt-replay, written by src/rtt/sim/replay.py).
 *
 * `<stem>.bin` is little-endian float32 [frames x bodies x 7] holding each body's
 * world pose (px, py, pz, qx, qy, qz, qw) at `fps`. Coordinates are MuJoCo's Z-up
 * world; quaternions are already xyzw (three.js order). The viewer rotates its
 * root group by -90 degrees about X to show Z-up data in three.js's Y-up world,
 * and applies nothing else, so there is exactly one up-axis conversion.
 *
 * Two header versions are read. Version 1 draws each body from primitive geoms.
 * Version 2 (src/rtt/sim/replay_v2.schema.json) keeps the same pose payload and draws
 * bodies from content-addressed GLB assets (lib/assets.ts); its v1-style geoms are an
 * optional collision overlay, and an optional `<stem>.ovl.bin` sidecar carries
 * per-frame readouts as float16.
 *
 * Version 2 lanes may also carry (all optional, so uniform float32 lanes read as before):
 *
 * - `t`: the time in seconds of each stored frame, strictly increasing and non-uniform (a
 *   keyframed lane); without it frame i is at i / fps. A keyframed lane's `fps` is nominal: the
 *   rate its run was recorded at (`meta.keyframes.recorded_fps`), not the rate of its frames, so
 *   every time comes from `t`. `meta.keyframes` (recorded_fps, samples, tolerance_mm of 1, 1.5 or
 *   2, max_error_vs_recording_mm) is present exactly when `t` is.
 * - `cuts`: frame indices where a new segment starts (after a tool change or a head switch).
 *   There is a frame exactly at each segment's start. Poses are never interpolated from frame
 *   cuts[i] - 1 to cuts[i]: between those two frames' times the replay holds the earlier pose,
 *   and the player covers the gap with a "Changing tool" card.
 * - `quantised: true`: each pose is seven little-endian int16 words, 14 bytes: position as
 *   round(mm x 10), then the quaternion as "smallest three" (the three components other than the
 *   largest, scaled by 32767 / 0.70710678, and the largest one's index). lib/poses.ts has the exact
 *   layout. Decoding yields the same float32 [frames x bodies x 7] layout as an unquantised lane.
 *   Without the flag the payload is float32 as before.
 *
 * Lanes without `t` or `cuts` behave as before: uniform at the stated fps, no cuts.
 */

import { matching, onlyKeys, readAssetId, readAssetMap, referencedAssets } from "./assets";
import { decodeQuantised, poseBytes, type PoseLayout, QUANTISED_STRIDE } from "./poses";
import { holdTrack, liftTrack, POSE_STRIDE, referenceFrame } from "./track.mjs";
import type {
  AnyReplayHeader,
  LaneKeyframes,
  OverlayChannel,
  PhaseSpan,
  ReplayBodyV2,
  ReplayGeom,
  ReplayHeaderV2,
  ReplayMetaV2,
  ReplayOverlays,
  ReplayProvenance,
  ReplayVisual,
  TaskSummary,
  Vec3,
  Quat,
} from "./types";
import { KEY_TOLERANCES_MM } from "./types";
import { DataError, Node, parseReplayHeader, readGeom } from "./validate";

/** Geoms smaller than this in every half extent are "parked" generator slots (not drawn). */
export const PARKED_HALF_SIZE_M = 2e-4;

const LITTLE_ENDIAN_HOST = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class ReplayFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReplayFormatError";
  }
}

export { decodeQuantised, encodeQuantised, poseBytes, type PoseLayout, QUANT_POS_M, QUANTISED_STRIDE } from "./poses";

/**
 * Decode the `.bin` payload into a flat Float32Array [frames x bodies x 7] (px, py, pz, qx, qy,
 * qz, qw), checking its size against the header. Quantised payloads are expanded to the same layout.
 */
export function decodeFrames(buffer: ArrayBuffer, header: PoseLayout): Float32Array {
  const expected = poseBytes(header);
  if (buffer.byteLength !== expected) {
    const what = header.quantised ? `${QUANTISED_STRIDE} int16` : `${POSE_STRIDE} float32`;
    throw new ReplayFormatError(
      `replay payload is ${buffer.byteLength} bytes; header says ${header.n_frames} frames x ${header.n_bodies} bodies x ${what} = ${expected} bytes`,
    );
  }
  if (header.quantised) return decodeQuantised(buffer, header.n_frames * header.n_bodies);
  if (LITTLE_ENDIAN_HOST) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const out = new Float32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

// ---------------------------------------------------------------- header version 2

/** Overlay channels a version 2 header may declare, in the schema's order. */
export const OVERLAY_CHANNELS: readonly OverlayChannel[] = ["offset_mm", "seat_short_mm", "hold_s", "n_contacts"];

const V2_KEYS = [
  "format",
  "version",
  "fps",
  "dtype",
  "byte_order",
  "layout",
  "pose_fields",
  "up_axis",
  "units",
  "n_frames",
  "n_bodies",
  "bin",
  "assets",
  "bodies",
  "static",
  "static_visuals",
  "meta",
  "provenance",
  "overlays",
  "t",
  "cuts",
  "quantised",
] as const;
/** Pose fields a header may name for a quantised payload (its seven int16 words per body and frame). */
export const QUANTISED_POSE_FIELDS = ["px", "py", "pz", "qa", "qb", "qc", "qi"] as const;
const PROVENANCE_KEYS = [
  "source",
  "engine",
  "engine_version",
  "run_id",
  "code_version",
  "product_mode",
  "episode",
  "suite",
  "condition",
  "seed",
  "task_sha256",
  "record_version",
  "created_utc",
] as const;
const SLUG_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;

function slugOf(n: Node): string {
  return matching(n, SLUG_RE, "an id of letters, digits, '.', '_' or '-'");
}

/** An array that must equal `expected` exactly (the schema's `const` arrays). */
function constList(n: Node, expected: readonly string[]): void {
  const got = n.strs();
  if (got.length !== expected.length || got.some((v, i) => v !== expected[i])) {
    n.fail(`expected [${expected.join(", ")}], got [${got.join(", ")}]`);
  }
}

function readVisual(n: Node): ReplayVisual {
  onlyKeys(n, ["asset", "pos", "quat_xyzw", "scale"]);
  const scale = n.get("scale").nums(3);
  // A negative scale would draw a mirrored copy; right jaws are their own assets.
  scale.forEach((s, i) => {
    if (!(s > 0)) n.get("scale").fail(`scale[${i}] must be positive (a right jaw is its own asset, never a mirror), got ${s}`);
  });
  return {
    asset: readAssetId(n.get("asset")),
    pos: n.get("pos").nums(3) as Vec3,
    quat_xyzw: n.get("quat_xyzw").nums(4) as Quat,
    scale: scale as Vec3,
  };
}

function readGeomV2(n: Node): ReplayGeom {
  const geom = readGeom(n);
  geom.rgba.forEach((c, i) => {
    if (c < 0 || c > 1) n.get("rgba").fail(`rgba[${i}] must be in [0, 1], got ${c}`);
  });
  return geom;
}

function readBody(n: Node): ReplayBodyV2 {
  onlyKeys(n, ["name", "visuals", "geoms"]);
  const body: ReplayBodyV2 = {
    name: slugOf(n.get("name")),
    visuals: n.get("visuals").items().map(readVisual),
    geoms: n.has("geoms") ? n.get("geoms").items().map(readGeomV2) : [],
  };
  if (body.visuals.length === 0 && body.geoms.length === 0) n.fail(`body ${body.name} has nothing to draw (no visuals, no geoms)`);
  return body;
}

function readMeta(n: Node): ReplayMetaV2 {
  const raw = n.object();
  const geometry = n.get("geometry_id");
  const meta: ReplayMetaV2 = {
    ...raw,
    design_space: slugOf(n.get("design_space")),
    design_id: n.get("design_id").nonEmpty(),
    geometry_id: geometry.value === null ? null : matching(geometry, SHA256, "a sha256 or null"),
    part_id: slugOf(n.get("part_id")),
  };
  if (n.has("keyframes")) meta.keyframes = readKeyframes(n.get("keyframes"));
  else delete meta.keyframes;
  return meta;
}

/** A keyframed lane's `meta.keyframes` (rtt.studio.replay._lane_replay), exactly its four fields. */
function readKeyframes(n: Node): LaneKeyframes {
  // The old name hid what the error is measured against; the exporter renamed it.
  if (n.has("max_error_mm")) n.get("max_error_mm").fail("renamed max_error_vs_recording_mm (the error against the recording, not the simulation)");
  onlyKeys(n, ["recorded_fps", "samples", "tolerance_mm", "max_error_vs_recording_mm"]);
  const kf: LaneKeyframes = {
    recorded_fps: n.get("recorded_fps").int(),
    samples: n.get("samples").int(),
    tolerance_mm: n.get("tolerance_mm").num(),
    max_error_vs_recording_mm: n.get("max_error_vs_recording_mm").num(),
  };
  if (!KEY_TOLERANCES_MM.some((t) => Math.abs(t - kf.tolerance_mm) < 1e-9)) n.get("tolerance_mm").fail(`a lane is keyframed within ${KEY_TOLERANCES_MM.join(", ")} mm, got ${kf.tolerance_mm}`);
  if (!(kf.max_error_vs_recording_mm >= 0) || kf.max_error_vs_recording_mm > kf.tolerance_mm + 1e-9) {
    n.get("max_error_vs_recording_mm").fail(`expected 0 to the ${kf.tolerance_mm} mm tolerance, got ${kf.max_error_vs_recording_mm}`);
  }
  return kf;
}

function readProvenance(n: Node): ReplayProvenance {
  onlyKeys(n, PROVENANCE_KEYS);
  const p: ReplayProvenance = {
    source: n.get("source").literal("simulation"),
    engine: n.get("engine").literal("mujoco"),
    engine_version: matching(n.get("engine_version"), /^[0-9]+\.[0-9]+\.[0-9]+$/, "a version like 3.15.0"),
    run_id: slugOf(n.get("run_id")),
    code_version: matching(n.get("code_version"), /^[0-9a-f]{7,40}(\+dirty)?$/, "a git commit, optionally +dirty"),
    product_mode: n.get("product_mode").bool(),
  };
  if (n.has("episode")) p.episode = n.get("episode").nonEmpty();
  if (n.has("suite")) p.suite = slugOf(n.get("suite"));
  if (n.has("condition")) p.condition = n.get("condition").nonEmpty();
  if (n.has("seed")) {
    p.seed = n.get("seed").int();
    if (p.seed < 0) n.get("seed").fail(`expected a seed >= 0, got ${p.seed}`);
  }
  if (n.has("task_sha256")) p.task_sha256 = matching(n.get("task_sha256"), SHA256, "a sha256");
  if (n.has("record_version")) {
    p.record_version = n.get("record_version").int();
    if (p.record_version < 1) n.get("record_version").fail(`expected a record version >= 1, got ${p.record_version}`);
  }
  if (n.has("created_utc")) {
    p.created_utc = matching(
      n.get("created_utc"),
      /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$/,
      "a UTC time like 2026-10-17T10:00:00Z",
    );
  }
  return p;
}

function readOverlays(n: Node): ReplayOverlays {
  onlyKeys(n, ["bin", "dtype", "byte_order", "layout", "channels"]);
  n.get("dtype").literal("float16");
  n.get("byte_order").literal("little");
  constList(n.get("layout"), ["frames", "channels"]);
  const channels = n
    .get("channels")
    .items()
    .map((c) => c.oneOf(OVERLAY_CHANNELS));
  if (channels.length === 0) n.get("channels").fail("expected at least one channel");
  if (new Set(channels).size !== channels.length) n.get("channels").fail("channels must not repeat");
  return {
    bin: matching(n.get("bin"), /^[A-Za-z0-9][A-Za-z0-9_.-]*\.ovl\.bin$/, "a file name ending .ovl.bin"),
    dtype: "float16",
    byte_order: "little",
    channels,
  };
}

/**
 * Validate a version 2 replay header against src/rtt/sim/replay_v2.schema.json and the
 * cross-field rules in its `$comment`: n_bodies matches the body list, every visual
 * names a listed asset, and every asset is content addressed.
 */
export function parseReplayHeaderV2(raw: unknown, file = "replay.json"): ReplayHeaderV2 {
  const root = new Node(file, "$", raw);
  onlyKeys(root, V2_KEYS);
  root.get("format").literal("rtt-replay");
  root.get("version").literal(2);
  const quantised = root.has("quantised") ? root.get("quantised").bool() : false;
  // A quantised payload is int16 and says so; an unquantised one is float32.
  const dtype = root.get("dtype").literal(quantised ? "int16" : "float32");
  root.get("byte_order").literal("little");
  root.get("up_axis").literal("z");
  root.get("units").literal("m");
  constList(root.get("layout"), ["frames", "bodies", "pose"]);
  // A quantised header names its seven int16 words exactly; a float32 one its seven floats.
  constList(root.get("pose_fields"), quantised ? QUANTISED_POSE_FIELDS : ["px", "py", "pz", "qx", "qy", "qz", "qw"]);
  const fps = root.get("fps").int();
  if (fps < 1 || fps > 1000) root.get("fps").fail(`fps must be in 1..1000, got ${fps}`);
  const n_frames = root.get("n_frames").int();
  if (n_frames < 1) root.get("n_frames").fail(`expected at least 1 frame, got ${n_frames}`);
  const n_bodies = root.get("n_bodies").int();
  if (n_bodies < 1) root.get("n_bodies").fail(`expected at least 1 body, got ${n_bodies}`);
  const bin = matching(root.get("bin"), /^[A-Za-z0-9][A-Za-z0-9_.-]*\.bin$/, "a file name ending .bin, next to the header");
  const assets = readAssetMap(root.get("assets"));
  const bodies = root.get("bodies").items().map(readBody);
  if (bodies.length === 0) root.get("bodies").fail("expected at least one body");
  if (n_bodies !== bodies.length) root.get("n_bodies").fail(`header lists ${bodies.length} bodies, n_bodies is ${n_bodies}`);
  const static_visuals = root.has("static_visuals") ? root.get("static_visuals").items().map(readVisual) : [];
  const t = root.has("t") ? readTimeline(root.get("t"), n_frames) : null;
  const cuts = root.has("cuts") ? readCuts(root.get("cuts"), n_frames) : [];
  const meta = readMeta(root.get("meta"));
  // A keyframed lane says how it was keyframed, and its fps is the recording's (nominal: its
  // frames are at t, not at i / fps); a uniform lane has neither.
  const kf = meta.keyframes;
  if (t && !kf) root.get("meta").fail("a keyframed lane (with t) says how it was keyframed in meta.keyframes");
  if (!t && kf) root.get("meta").get("keyframes").fail("keyframes describe a lane with frame times t, and this one has none");
  if (kf) {
    const where = root.get("meta").get("keyframes");
    if (kf.recorded_fps !== fps) where.get("recorded_fps").fail(`the header's fps is the recording's rate, ${fps}, not ${kf.recorded_fps}`);
    if (kf.samples < n_frames) where.get("samples").fail(`${n_frames} keys cannot come from ${kf.samples} samples`);
  }
  const unknown = (where: Node, v: ReplayVisual) => {
    if (!Object.hasOwn(assets, v.asset)) where.fail(`visual draws asset ${v.asset}, which the header's assets map does not list`);
  };
  root
    .get("bodies")
    .items()
    .forEach((b, i) => bodies[i]!.visuals.forEach((v, j) => unknown(b.get("visuals").items()[j]!, v)));
  static_visuals.forEach((v, j) => unknown(root.get("static_visuals").items()[j]!, v));
  // The map holds exactly the entries the visuals draw (rtt.sim.replay.check_header), so a
  // header's credits and payload never count an asset that is not on screen.
  const drawn = new Set(referencedAssets({ bodies, static_visuals }));
  for (const id of Object.keys(assets)) {
    if (!drawn.has(id)) root.get("assets").get(id).fail(`asset ${id} is listed, but no visual draws it`);
  }
  return {
    format: "rtt-replay",
    version: 2,
    fps,
    dtype,
    byte_order: "little",
    up_axis: "z",
    units: "m",
    bodies,
    static: root.has("static") ? root.get("static").items().map(readGeomV2) : [],
    static_visuals,
    assets,
    n_frames,
    n_bodies,
    bin,
    meta,
    provenance: readProvenance(root.get("provenance")),
    overlays: root.has("overlays") ? readOverlays(root.get("overlays")) : null,
    t,
    cuts,
    quantised,
  };
}

/** A lane's frame times: one finite, non-negative time per frame, strictly increasing. */
function readTimeline(n: Node, nFrames: number): number[] {
  const t = n.items().map((x) => x.num());
  if (t.length !== nFrames) n.fail(`expected one time per frame (${nFrames}), got ${t.length}`);
  t.forEach((v, i) => {
    if (!Number.isFinite(v) || v < 0) n.items()[i]!.fail(`a frame time must be a finite number >= 0, got ${v}`);
    if (i > 0 && v <= t[i - 1]!) n.items()[i]!.fail(`frame times must increase (${t[i - 1]} then ${v})`);
  });
  return t;
}

/** Segment starts: increasing frame indices inside the recording. */
function readCuts(n: Node, nFrames: number): number[] {
  const cuts = n.items().map((x) => x.int());
  cuts.forEach((c, i) => {
    if (c < 0 || c >= nFrames) n.items()[i]!.fail(`a cut is a frame index in 0..${nFrames - 1}, got ${c}`);
    if (i > 0 && c <= cuts[i - 1]!) n.items()[i]!.fail("cuts must increase");
  });
  return cuts;
}

/** Header versions this viewer reads. */
export const SUPPORTED_REPLAY_VERSIONS = [1, 2] as const;

/** Validate a replay header of either version (dispatching on `version`). */
export function parseAnyReplayHeader(raw: unknown, file = "replay.json"): AnyReplayHeader {
  const root = new Node(file, "$", raw);
  const version = root.get("version").value;
  if (version === 1) return parseReplayHeader(raw, file);
  if (version === 2) return parseReplayHeaderV2(raw, file);
  throw new DataError(file, "$.version", `unsupported replay version ${JSON.stringify(version)}; this viewer reads 1 and 2`);
}

export function isV2(header: AnyReplayHeader): header is ReplayHeaderV2 {
  return header.version === 2;
}

/** The asset files a version 2 header draws (for build-time checks). */
export function referencedAssetFiles(header: ReplayHeaderV2): { id: string; uri: string; bytes: number; sha256: string }[] {
  return referencedAssets(header).map((id) => {
    const a = header.assets[id]!;
    return { id, uri: a.uri, bytes: a.bytes, sha256: a.sha256 };
  });
}

/** IEEE 754 half-precision bits to a number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 0x1f) return frac ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export type OverlayTracks = Partial<Record<OverlayChannel, Float32Array>>;

/** Decode `<stem>.ovl.bin` (float16 [frames x channels]) into one track per channel. */
export function decodeOverlays(
  buffer: ArrayBuffer,
  header: Pick<ReplayHeaderV2, "n_frames"> & { overlays: ReplayOverlays },
): OverlayTracks {
  const { channels } = header.overlays;
  const expected = header.n_frames * channels.length * 2;
  if (buffer.byteLength !== expected) {
    throw new ReplayFormatError(
      `overlay payload is ${buffer.byteLength} bytes; header says ${header.n_frames} frames x ${channels.length} channels = ${expected} bytes`,
    );
  }
  const view = new DataView(buffer);
  const out: OverlayTracks = {};
  channels.forEach((name, c) => {
    const track = new Float32Array(header.n_frames);
    for (let f = 0; f < header.n_frames; f++) track[f] = halfToFloat(view.getUint16((f * channels.length + c) * 2, true));
    out[name] = track;
  });
  return out;
}

/**
 * The caption every clip carries: "simulation replay · MuJoCo 3.15.0 · run <id> ·
 * design <id>", plus "product mode" for product-mode episodes. Version 1 headers have
 * no run provenance, so they name the design only.
 */
export function replayCaption(header: AnyReplayHeader): string {
  const short = (id: string) => (SHA256.test(id) ? id.slice(0, 12) : id);
  if (isV2(header)) {
    const p = header.provenance;
    const parts = ["simulation replay", `MuJoCo ${p.engine_version}`, `run ${p.run_id}`, `design ${short(header.meta.design_id)}`];
    if (p.product_mode) parts.push("product mode");
    return parts.join(" · ");
  }
  const design = header.meta.design_id;
  return ["simulation replay", "MuJoCo", ...(typeof design === "string" && design ? [`design ${short(design)}`] : [])].join(
    " · ",
  );
}

/** True for a generator's zero-size placeholder box, which is not drawn. */
export function isParkedGeom(geom: ReplayGeom): boolean {
  return geom.type === "box" && (geom.half_size ?? []).every((h) => Math.abs(h) <= PARKED_HALF_SIZE_M);
}

/** Per-frame overlay values for one lane. */
export interface LaneTracks {
  /** Object lift above its reference height, metres. */
  lift: Float32Array;
  /** Continuous seconds at or above the success height. */
  hold: Float32Array;
  /** Time at which the scorer decided the outcome (recording continues for the tail). */
  decidedAt: number;
}

/** Axis-aligned box in MuJoCo world coordinates (Z-up, metres). */
export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
}

/** Rotate vector `v` by unit quaternion `q` (x, y, z, w); writes into `out`. */
export function rotate(q: ArrayLike<number>, v: readonly number[], out: number[]): void {
  const qx = q[0]!, qy = q[1]!, qz = q[2]!, qw = q[3]!;
  const vx = v[0]!, vy = v[1]!, vz = v[2]!;
  // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
}

/**
 * Points that bound a geom, in its body's frame: box corners, the corners of a
 * cylinder's or capsule's bounding box, or a mesh's vertices. Planes give none.
 */
export function geomPoints(geom: ReplayGeom): number[][] {
  let local: number[][] = [];
  const corners = (hx: number, hy: number, hz: number) => {
    const out: number[][] = [];
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) out.push([sx * hx, sy * hy, sz * hz]);
    return out;
  };
  switch (geom.type) {
    case "box": {
      const [hx = 0, hy = 0, hz = 0] = geom.half_size ?? [];
      local = corners(hx, hy, hz);
      break;
    }
    case "cylinder":
      local = corners(geom.radius ?? 0, geom.radius ?? 0, geom.half_height ?? 0);
      break;
    case "capsule": {
      const r = geom.radius ?? 0;
      local = corners(r, r, (geom.half_length ?? 0) + r);
      break;
    }
    case "mesh": {
      const v = geom.vertices ?? [];
      for (let i = 0; i + 2 < v.length; i += 3) local.push([v[i]!, v[i + 1]!, v[i + 2]!]);
      break;
    }
    default:
      return [];
  }
  const out = [0, 0, 0];
  return local.map((pt) => {
    rotate(geom.quat_xyzw, pt, out);
    return [out[0]! + geom.pos[0], out[1]! + geom.pos[1], out[2]! + geom.pos[2]];
  });
}

/** Union of two bounds (either may be null). */
export function unionBounds(a: Bounds | null, b: Bounds | null): Bounds | null {
  if (!a) return b;
  if (!b) return a;
  return {
    min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
    max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
  };
}

/** Bytes fetched to load a replay (for the payload readout). */
export interface ReplayBytes {
  header: number;
  poses: number;
  overlays: number;
}

/** A decoded replay with pose sampling. */
export class Replay {
  readonly frames: Float32Array;
  readonly duration: number;
  /** Seconds of every frame for a keyframed lane (header `t`); null when frames sit at 1 / fps. */
  readonly times: Float64Array | null;
  /** Frame indices where a segment starts (header `cuts`), in order. */
  readonly cuts: readonly number[];
  private readonly stride: number;
  /** 1 at a frame that starts a segment: never interpolated into from the frame before. */
  private readonly cutAt: Uint8Array;

  constructor(
    readonly header: AnyReplayHeader,
    frames: Float32Array,
    /** Per-frame readouts from the `.ovl.bin` sidecar (version 2 only); null when absent. */
    readonly overlays: OverlayTracks | null = null,
    readonly bytes: ReplayBytes | null = null,
  ) {
    this.frames = frames;
    this.stride = header.n_bodies * POSE_STRIDE;
    const v2 = header.version === 2 ? header : null;
    this.times = v2?.t ? Float64Array.from(v2.t) : null;
    this.cuts = v2?.cuts ?? [];
    this.cutAt = new Uint8Array(header.n_frames);
    for (const c of this.cuts) if (c > 0 && c < header.n_frames) this.cutAt[c] = 1;
    this.duration = this.times ? Math.max(0, this.times[this.times.length - 1] ?? 0) : Math.max(0, (header.n_frames - 1) / header.fps);
  }

  /** Seconds of frame `i` (clamped to the recording). */
  timeOf(i: number): number {
    const k = Math.min(Math.max(Math.round(i), 0), this.header.n_frames - 1);
    return this.times ? this.times[k]! : k / this.header.fps;
  }

  /** True when frame `i` starts a segment (a cut). */
  isCut(i: number): boolean {
    return this.cutAt[i] === 1;
  }

  /**
   * The jump at time `t`, when `t` lies between the last frame of a segment and the first frame
   * of the next (the time the replay holds a pose instead of animating a tool change), or null.
   */
  cutGapAt(t: number): { cut: number; start_s: number; end_s: number } | null {
    for (const c of this.cuts) {
      if (c <= 0) continue;
      const start = this.timeOf(c - 1);
      const end = this.timeOf(c);
      if (t >= start && t < end) return { cut: c, start_s: start, end_s: end };
      if (start > t) break;
    }
    return null;
  }

  get version(): 1 | 2 {
    return this.header.version;
  }

  /** Value of an overlay channel at fractional frame `f` (nearest frame), or null without that channel. */
  overlayAt(channel: OverlayChannel, f: number): number | null {
    const track = this.overlays?.[channel];
    if (!track || track.length === 0) return null;
    const i = Math.min(Math.max(Math.round(f), 0), track.length - 1);
    return track[i]!;
  }

  /**
   * Tight box containing every drawn body over the whole recording (static geoms
   * excluded): each geom's corner points, moved by the body pose of every frame.
   * `extra(b)` adds body-frame points for body `b`, such as the corners of the GLB
   * assets a version 2 body draws. `frames` limits the sweep to [first, last].
   */
  bounds(extra?: (body: number) => number[][], frames?: [number, number]): Bounds {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const { n_frames, bodies } = this.header;
    const p = [0, 0, 0];
    bodies.forEach((body, b) => {
      const local = body.geoms.filter((g) => !isParkedGeom(g)).flatMap(geomPoints);
      if (extra) local.push(...extra(b));
      if (local.length === 0) return;
      const first = Math.max(0, Math.floor(frames?.[0] ?? 0));
      const last = Math.min(n_frames - 1, Math.ceil(frames?.[1] ?? n_frames - 1));
      for (let f = first; f <= last; f++) {
        const o = f * this.stride + b * POSE_STRIDE;
        const q = this.frames.subarray(o + 3, o + 7);
        for (const point of local) {
          rotate(q, point, p);
          for (let k = 0; k < 3; k++) {
            const v = this.frames[o + k]! + p[k]!;
            if (v < min[k]!) min[k] = v;
            if (v > max[k]!) max[k] = v;
          }
        }
      }
    });
    if (!Number.isFinite(min[0])) return { min: [0, 0, 0], max: [0, 0, 0] };
    return { min, max };
  }

  bodyIndex(name: string): number {
    return this.header.bodies.findIndex((b) => b.name === name);
  }

  /**
   * Fractional frame index for time `t` (seconds), clamped to the recording. A keyframed lane
   * finds the frames around `t` in its timeline; frame i + a means a of the way to frame i + 1.
   */
  frameAt(t: number): number {
    const last = this.header.n_frames - 1;
    const times = this.times;
    if (!times) return Math.min(Math.max(t * this.header.fps, 0), last);
    if (!(t > times[0]!)) return 0;
    if (t >= times[last]!) return last;
    let lo = 0;
    let hi = last;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (times[mid]! <= t) lo = mid;
      else hi = mid;
    }
    const span = times[hi]! - times[lo]!;
    return span > 0 ? lo + (t - times[lo]!) / span : lo;
  }

  /**
   * Interpolated pose of body `body` at fractional frame `f`: linear position,
   * shortest-path spherical quaternion. Writes into `pos` (3) and `quat` (4, xyzw).
   * Never interpolates into a cut: up to the cut frame itself the earlier frame's pose holds.
   */
  sample(body: number, f: number, pos: Float64Array | number[], quat: Float64Array | number[]): void {
    const last = this.header.n_frames - 1;
    const i0 = Math.min(Math.max(Math.floor(f), 0), last);
    const i1 = Math.min(i0 + 1, last);
    const a = this.cutAt[i1] === 1 && i1 !== i0 ? 0 : Math.min(Math.max(f - i0, 0), 1);
    const p = this.frames;
    const o0 = i0 * this.stride + body * POSE_STRIDE;
    const o1 = i1 * this.stride + body * POSE_STRIDE;
    for (let k = 0; k < 3; k++) pos[k] = p[o0 + k]! + (p[o1 + k]! - p[o0 + k]!) * a;
    slerp(p, o0 + 3, o1 + 3, a, quat);
  }

  /**
   * Object position (x, y, z) at the scorer's reference frame (start of DESCEND), or
   * null without an object body. The success datum sits `success_lift_m` above it.
   */
  objectReference(task: Pick<TaskSummary, "phases">, objectName = "object"): [number, number, number] | null {
    const obj = this.bodyIndex(objectName);
    if (obj < 0) return null;
    const descend = task.phases.find((ph) => ph.name === "descend")?.start_s ?? 0;
    // A keyframed lane's fps is nominal: its frame at a time comes from its timeline.
    const f = this.times ? Math.round(this.frameAt(descend)) : referenceFrame(this.header.fps, descend, this.header.n_frames);
    const o = f * this.stride + obj * POSE_STRIDE;
    return [this.frames[o]!, this.frames[o + 1]!, this.frames[o + 2]!];
  }

  /**
   * Lift and hold tracks for the object body, using the task's reference and thresholds. The hold
   * counts frames at `fps`, so a keyframed lane (frames at `t`, fps nominal) is refused.
   */
  tracks(task: Pick<TaskSummary, "phases" | "success_lift_m" | "record_tail_s">, objectName = "object"): LaneTracks {
    if (this.times) throw new ReplayFormatError("hold tracks need frames at a fixed rate; this lane is keyframed (its frames are at t)");
    const obj = this.bodyIndex(objectName);
    const n = this.header.n_frames;
    if (obj < 0) {
      return { lift: new Float32Array(n), hold: new Float32Array(n), decidedAt: this.duration };
    }
    const descend = task.phases.find((ph) => ph.name === "descend")?.start_s ?? 0;
    const lift = liftTrack(this.frames, this.header.n_bodies, obj, referenceFrame(this.header.fps, descend, n));
    return {
      lift,
      hold: holdTrack(lift, this.header.fps, task.success_lift_m),
      decidedAt: Math.max(0, this.duration - task.record_tail_s),
    };
  }
}

/** Spherical interpolation between quaternions stored in `src` at offsets `o0` and `o1`. */
export function slerp(src: ArrayLike<number>, o0: number, o1: number, a: number, out: Float64Array | number[]): void {
  const x0 = src[o0]!, y0 = src[o0 + 1]!, z0 = src[o0 + 2]!, w0 = src[o0 + 3]!;
  let x1 = src[o1]!, y1 = src[o1 + 1]!, z1 = src[o1 + 2]!, w1 = src[o1 + 3]!;
  let cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;
  if (cos < 0) {
    cos = -cos;
    x1 = -x1;
    y1 = -y1;
    z1 = -z1;
    w1 = -w1;
  }
  let s0: number;
  let s1: number;
  if (cos > 0.9995) {
    s0 = 1 - a;
    s1 = a;
  } else {
    const theta = Math.acos(cos);
    const sin = Math.sin(theta);
    s0 = Math.sin((1 - a) * theta) / sin;
    s1 = Math.sin(a * theta) / sin;
  }
  const x = s0 * x0 + s1 * x1;
  const y = s0 * y0 + s1 * y1;
  const z = s0 * z0 + s1 * z1;
  const w = s0 * w0 + s1 * w1;
  const len = Math.hypot(x, y, z, w) || 1;
  out[0] = x / len;
  out[1] = y / len;
  out[2] = z / len;
  out[3] = w / len;
}

/** Phase active at time `t` (the last phase whose start is <= t). */
export function phaseAt(t: number, phases: PhaseSpan[]): PhaseSpan | null {
  let current: PhaseSpan | null = null;
  for (const ph of phases) if (t >= ph.start_s) current = ph;
  return current;
}

/**
 * Fetch and decode `<stem>.json` + `<stem>.bin` (and, for a version 2 header that
 * declares one, the `.ovl.bin` overlay sidecar) for a stem URL without extension.
 * Reads header versions 1 and 2; GLB assets are loaded separately (lib/assets.ts).
 */
export async function loadReplay(stemUrl: string, init?: RequestInit): Promise<Replay> {
  const headRes = await fetch(`${stemUrl}.json`, init);
  if (!headRes.ok) throw new ReplayFormatError(`replay header ${stemUrl}.json: HTTP ${headRes.status}`);
  const text = await headRes.text();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (err) {
    throw new ReplayFormatError(`replay header ${stemUrl}.json: ${err instanceof Error ? err.message : String(err)}`);
  }
  const header = parseAnyReplayHeader(raw, `${stemUrl}.json`);
  const dir = stemUrl.slice(0, stemUrl.lastIndexOf("/") + 1);
  const fetchBin = async (name: string, what: string) => {
    const res = await fetch(`${dir}${name}`, init);
    if (!res.ok) throw new ReplayFormatError(`replay ${what} ${dir}${name}: HTTP ${res.status}`);
    return res.arrayBuffer();
  };
  const poses = await fetchBin(header.bin, "frames");
  let overlays: OverlayTracks | null = null;
  let overlayBytes = 0;
  if (isV2(header) && header.overlays) {
    const buf = await fetchBin(header.overlays.bin, "overlays");
    overlayBytes = buf.byteLength;
    overlays = decodeOverlays(buf, { n_frames: header.n_frames, overlays: header.overlays });
  }
  const bytes = { header: new TextEncoder().encode(text).byteLength, poses: poses.byteLength, overlays: overlayBytes };
  return new Replay(header, decodeFrames(poses, header), overlays, bytes);
}
