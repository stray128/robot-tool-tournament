/**
 * Pose payloads of a replay lane (`<stem>.bin`), shared by the decoder (lib/replay.ts) and every
 * size check (lib/assets.ts, the build-time loaders), so one rule sizes them everywhere.
 *
 * Unquantised: little-endian float32, seven per body and frame: px, py, pz, qx, qy, qz, qw
 * (metres; quaternion xyzw).
 *
 * Quantised (`quantised: true` in the header): little-endian int16, seven per body and frame,
 * 14 bytes: [px, py, pz, q_a, q_b, q_c, qi].
 *  - px, py, pz = round(mm x 10): units of 0.1 mm, so +-3.2767 m.
 *  - qi = the index (0..3 = x, y, z, w) of the quaternion's largest-magnitude component, after
 *    flipping the quaternion's sign so that component is positive.
 *  - q_a, q_b, q_c = the other three components in x, y, z, w order (qi skipped), each
 *    round(c / 0.70710678 x 32767).
 *  - Decode: c = int / 32767 x 0.70710678; the dropped one = sqrt(max(0, 1 - sum c^2)) at qi.
 */

/** Values per body and frame in either payload. */
export const POSE_VALUES = 7;
/** int16 words per body and frame of a quantised payload. */
export const QUANTISED_STRIDE = 7;
/** Metres per position unit of a quantised payload (0.1 mm). */
export const QUANT_POS_M = 1e-4;
/** The largest magnitude a non-largest unit-quaternion component can have (1 / sqrt 2). */
export const QUAT_RANGE = 0.70710678;

/** The header fields that decide the pose payload's size and encoding. */
export interface PoseLayout {
  n_frames: number;
  n_bodies: number;
  quantised?: boolean;
}

/** Bytes of the `.bin` pose payload a header describes. */
export function poseBytes(header: PoseLayout): number {
  return header.quantised ? header.n_frames * header.n_bodies * QUANTISED_STRIDE * 2 : header.n_frames * header.n_bodies * POSE_VALUES * 4;
}

const clamp16 = (v: number) => Math.max(-32767, Math.min(32767, Math.round(v)));

/** Expand `poses` quantised poses (7 int16 each) to float32 poses (px, py, pz, qx, qy, qz, qw). */
export function decodeQuantised(buffer: ArrayBuffer, poses: number): Float32Array {
  const view = new DataView(buffer);
  const out = new Float32Array(poses * POSE_VALUES);
  const q = [0, 0, 0, 0];
  for (let i = 0; i < poses; i++) {
    const o = i * QUANTISED_STRIDE * 2;
    const qi = view.getInt16(o + 12, true);
    if (qi < 0 || qi > 3) throw new RangeError(`pose ${i}: the dropped-component index is ${qi}, not 0 to 3`);
    let sum = 0;
    let k = 0;
    for (let j = 0; j < 4; j++) {
      if (j === qi) continue;
      const c = (view.getInt16(o + 6 + 2 * k, true) / 32767) * QUAT_RANGE;
      q[j] = c;
      sum += c * c;
      k += 1;
    }
    q[qi] = Math.sqrt(Math.max(0, 1 - sum));
    const len = Math.hypot(q[0]!, q[1]!, q[2]!, q[3]!) || 1;
    const p = i * POSE_VALUES;
    out[p] = view.getInt16(o, true) * QUANT_POS_M;
    out[p + 1] = view.getInt16(o + 2, true) * QUANT_POS_M;
    out[p + 2] = view.getInt16(o + 4, true) * QUANT_POS_M;
    for (let j = 0; j < 4; j++) out[p + 3 + j] = q[j]! / len;
  }
  return out;
}

/**
 * Quantise float32 poses (7 values each) to the int16 payload: the inverse of decodeQuantised and
 * the same words the exporter writes. For tests and tools.
 */
export function encodeQuantised(frames: ArrayLike<number>): ArrayBuffer {
  const poses = Math.floor(frames.length / POSE_VALUES);
  const buf = new ArrayBuffer(poses * QUANTISED_STRIDE * 2);
  const view = new DataView(buf);
  for (let i = 0; i < poses; i++) {
    const p = i * POSE_VALUES;
    const o = i * QUANTISED_STRIDE * 2;
    for (let k = 0; k < 3; k++) view.setInt16(o + 2 * k, clamp16(frames[p + k]! * 1000 * 10), true);
    const len = Math.hypot(frames[p + 3]!, frames[p + 4]!, frames[p + 5]!, frames[p + 6]!) || 1;
    const q = [0, 1, 2, 3].map((j) => frames[p + 3 + j]! / len);
    let qi = 0;
    for (let j = 1; j < 4; j++) if (Math.abs(q[j]!) > Math.abs(q[qi]!)) qi = j;
    const sign = q[qi]! < 0 ? -1 : 1;
    let k = 0;
    for (let j = 0; j < 4; j++) {
      if (j === qi) continue;
      view.setInt16(o + 6 + 2 * k, clamp16(((sign * q[j]!) / QUAT_RANGE) * 32767), true);
      k += 1;
    }
    view.setInt16(o + 12, qi, true);
  }
  return buf;
}
