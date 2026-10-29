/**
 * Camera framing for replays that should fill their frame: the task's objects over the whole run
 * (bodies named object*, plus the static visuals such as a tote or a cabinet) rather than every
 * body's swept volume, with the top raised so lifts stay in view.
 */

import { placedCorners, type LoadedAsset } from "@/lib/assets";
import { type Bounds, isV2, type Replay } from "@/lib/replay";

import { assetPoints } from "./lanes";

function rotate(q: ArrayLike<number>, v: number[]): number[] {
  const [x, y, z, w] = [q[0]!, q[1]!, q[2]!, q[3]!];
  const [px, py, pz] = [v[0]!, v[1]!, v[2]!];
  const tx = 2 * (y * pz - z * py);
  const ty = 2 * (z * px - x * pz);
  const tz = 2 * (x * py - y * px);
  return [px + w * tx + (y * tz - z * ty), py + w * ty + (z * tx - x * tz), pz + w * tz + (x * ty - y * tx)];
}

/**
 * Framing for the hero: the task's objects over the whole run (the tote, its lid, the parts), not
 * the tool's full swept volume, so the scene fills the frame; the top is raised so lifts stay in view.
 */
export function objectBounds(replay: Replay, assets: Map<string, LoadedAsset> | null): Bounds {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const add = (p: number[]) => {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k]!, p[k]!);
      max[k] = Math.max(max[k]!, p[k]!);
    }
  };
  const header = replay.header;
  const points = assetPoints(replay, assets);
  const pos = [0, 0, 0];
  const quat = [0, 0, 0, 1];
  header.bodies.forEach((body, b) => {
    if (!body.name.startsWith("object")) return;
    const local = points(b);
    for (let f = 0; f < header.n_frames; f += 5) {
      replay.sample(b, f, pos, quat);
      for (const pt of local) {
        const r = rotate(quat, pt);
        add([pos[0]! + r[0]!, pos[1]! + r[1]!, pos[2]! + r[2]!]);
      }
    }
  });
  if (isV2(header) && assets) {
    for (const v of header.static_visuals) {
      const loaded = assets.get(v.asset);
      if (loaded) placedCorners(loaded.box, v.pos, v.quat_xyzw, v.scale).forEach(add);
    }
  }
  if (!Number.isFinite(min[0]!)) return replay.bounds(points);
  return {
    min: [min[0]! - 0.03, min[1]! - 0.03, Math.min(min[2]!, 0) - 0.01],
    max: [max[0]! + 0.03, max[1]! + 0.03, max[2]! + 0.12],
  };
}

