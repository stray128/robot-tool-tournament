/** Loading helpers shared by the bout player, the design studio and the hero. */

import { placedCorners, sharedAssets, type LoadedAsset } from "@/lib/assets";
import { isV2, loadReplay, type Replay } from "@/lib/replay";

/** Body-frame corner points of each body's GLB visuals, for framing a version 2 replay. */
export function assetPoints(replay: Replay, assets: Map<string, LoadedAsset> | null): (body: number) => number[][] {
  const { header } = replay;
  return (b) => {
    if (!assets || !isV2(header)) return [];
    return header.bodies[b]!.visuals.flatMap((v) => {
      const loaded = assets.get(v.asset);
      return loaded ? placedCorners(loaded.box, v.pos, v.quat_xyzw, v.scale) : [];
    });
  };
}

/** Load a replay and, for version 2, every asset it draws (shared across lanes and views). */
export async function loadLane(
  stemUrl: string,
  signal?: AbortSignal,
): Promise<{ replay: Replay; assets: Map<string, LoadedAsset> | null }> {
  const replay = await loadReplay(stemUrl, { signal });
  const assets = isV2(replay.header) ? await sharedAssets.loadAll(replay.header) : null;
  return { replay, assets };
}
