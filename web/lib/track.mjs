/**
 * Per-frame overlay tracks derived from a replay's pose frames.
 *
 * Plain ESM JavaScript so that both the Next.js app (via lib/track.d.mts) and the
 * Node import script (scripts/import-demo.mjs) share one implementation.
 *
 * Frames are a flat little-endian float32 array laid out [frames x bodies x 7] with
 * (px, py, pz, qx, qy, qz, qw) per body, MuJoCo Z-up world coordinates, so the
 * object's height is component 2.
 */

export const POSE_STRIDE = 7;

/**
 * Frame index at which the scorer fixes the object's reference height: the start of
 * the DESCEND phase (see src/rtt/sim/episode.py, tracker.set_reference).
 *
 * @param {number} fps
 * @param {number} descendStartS
 * @param {number} nFrames
 * @returns {number}
 */
export function referenceFrame(fps, descendStartS, nFrames) {
  return Math.max(0, Math.min(nFrames - 1, Math.round(descendStartS * fps)));
}

/**
 * Lift of one body's origin above its height at `refFrame`, per frame, in metres.
 *
 * @param {Float32Array} frames
 * @param {number} nBodies
 * @param {number} bodyIndex
 * @param {number} refFrame
 * @returns {Float32Array}
 */
export function liftTrack(frames, nBodies, bodyIndex, refFrame) {
  const stride = nBodies * POSE_STRIDE;
  const n = Math.floor(frames.length / stride);
  const out = new Float32Array(n);
  if (n === 0) return out;
  const ref = frames[Math.min(refFrame, n - 1) * stride + bodyIndex * POSE_STRIDE + 2];
  for (let f = 0; f < n; f++) out[f] = frames[f * stride + bodyIndex * POSE_STRIDE + 2] - ref;
  return out;
}

/**
 * Continuous time, in seconds, that `lift` has stayed at or above `threshold`
 * at each frame (0 below the threshold). This mirrors the scorer's hold timer at
 * frame resolution.
 *
 * @param {Float32Array} lift
 * @param {number} fps
 * @param {number} threshold
 * @returns {Float32Array}
 */
export function holdTrack(lift, fps, threshold) {
  const out = new Float32Array(lift.length);
  let since = -1;
  for (let f = 0; f < lift.length; f++) {
    if (lift[f] >= threshold) {
      if (since < 0) since = f;
      out[f] = (f - since) / fps;
    } else {
      since = -1;
    }
  }
  return out;
}

/**
 * Peak lift (m) and longest continuous hold (s) of a pair of tracks.
 *
 * @param {Float32Array} lift
 * @param {Float32Array} hold
 * @returns {{ maxLift: number, maxHold: number }}
 */
export function summarizeTracks(lift, hold) {
  let maxLift = 0;
  let maxHold = 0;
  for (let f = 0; f < lift.length; f++) {
    if (lift[f] > maxLift) maxLift = lift[f];
    if (hold[f] > maxHold) maxHold = hold[f];
  }
  return { maxLift, maxHold };
}
