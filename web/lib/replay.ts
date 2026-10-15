/**
 * Replay decoding and sampling (format rtt-replay v1, written by src/rtt/sim/replay.py).
 *
 * `<stem>.bin` is little-endian float32 [frames x bodies x 7] holding each body's
 * world pose (px, py, pz, qx, qy, qz, qw) at `fps`. Coordinates are MuJoCo's Z-up
 * world; quaternions are already xyzw (three.js order). The viewer rotates its
 * root group by -90 degrees about X to show Z-up data in three.js's Y-up world,
 * and applies nothing else, so there is exactly one up-axis conversion.
 */

import { holdTrack, liftTrack, POSE_STRIDE, referenceFrame } from "./track.mjs";
import type { PhaseSpan, ReplayGeom, ReplayHeader, TaskSummary } from "./types";
import { parseReplayHeader } from "./validate";

/** Geoms smaller than this in every half extent are "parked" generator slots (not drawn). */
export const PARKED_HALF_SIZE_M = 2e-4;

const LITTLE_ENDIAN_HOST = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export class ReplayFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ReplayFormatError";
  }
}

/** Decode the `.bin` payload into a flat Float32Array, checking its size against the header. */
export function decodeFrames(buffer: ArrayBuffer, header: Pick<ReplayHeader, "n_frames" | "n_bodies">): Float32Array {
  const expected = header.n_frames * header.n_bodies * POSE_STRIDE * 4;
  if (buffer.byteLength !== expected) {
    throw new ReplayFormatError(
      `replay payload is ${buffer.byteLength} bytes; header says ${header.n_frames} frames x ${header.n_bodies} bodies = ${expected} bytes`,
    );
  }
  if (LITTLE_ENDIAN_HOST) return new Float32Array(buffer);
  const view = new DataView(buffer);
  const out = new Float32Array(buffer.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
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

/** A decoded replay with pose sampling. */
export class Replay {
  readonly frames: Float32Array;
  readonly duration: number;
  private readonly stride: number;

  constructor(
    readonly header: ReplayHeader,
    frames: Float32Array,
  ) {
    this.frames = frames;
    this.stride = header.n_bodies * POSE_STRIDE;
    this.duration = Math.max(0, (header.n_frames - 1) / header.fps);
  }

  /**
   * Tight box containing every drawn body over the whole recording (static geoms
   * excluded): each geom's corner points, moved by the body pose of every frame.
   */
  bounds(): Bounds {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const { n_frames, bodies } = this.header;
    const p = [0, 0, 0];
    bodies.forEach((body, b) => {
      const local = body.geoms.filter((g) => !isParkedGeom(g)).flatMap(geomPoints);
      if (local.length === 0) return;
      for (let f = 0; f < n_frames; f++) {
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

  /** Fractional frame index for time `t` (seconds), clamped to the recording. */
  frameAt(t: number): number {
    return Math.min(Math.max(t * this.header.fps, 0), this.header.n_frames - 1);
  }

  /**
   * Interpolated pose of body `body` at fractional frame `f`: linear position,
   * shortest-path spherical quaternion. Writes into `pos` (3) and `quat` (4, xyzw).
   */
  sample(body: number, f: number, pos: Float64Array | number[], quat: Float64Array | number[]): void {
    const last = this.header.n_frames - 1;
    const i0 = Math.min(Math.floor(f), last);
    const i1 = Math.min(i0 + 1, last);
    const a = Math.min(Math.max(f - i0, 0), 1);
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
    const f = referenceFrame(this.header.fps, descend, this.header.n_frames);
    const o = f * this.stride + obj * POSE_STRIDE;
    return [this.frames[o]!, this.frames[o + 1]!, this.frames[o + 2]!];
  }

  /** Lift and hold tracks for the object body, using the task's reference and thresholds. */
  tracks(task: Pick<TaskSummary, "phases" | "success_lift_m" | "record_tail_s">, objectName = "object"): LaneTracks {
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

/** Fetch and decode `<stem>.json` + `<stem>.bin` from `baseUrl` (no trailing slash). */
export async function loadReplay(stemUrl: string, init?: RequestInit): Promise<Replay> {
  const headRes = await fetch(`${stemUrl}.json`, init);
  if (!headRes.ok) throw new ReplayFormatError(`replay header ${stemUrl}.json: HTTP ${headRes.status}`);
  const header = parseReplayHeader(await headRes.json(), `${stemUrl}.json`);
  const dir = stemUrl.slice(0, stemUrl.lastIndexOf("/") + 1);
  const binRes = await fetch(`${dir}${header.bin}`, init);
  if (!binRes.ok) throw new ReplayFormatError(`replay frames ${dir}${header.bin}: HTTP ${binRes.status}`);
  return new Replay(header, decodeFrames(await binRes.arrayBuffer(), header));
}
