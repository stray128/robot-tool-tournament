/**
 * Build-time loaders of the effector pages (server only: reads public/data from disk).
 *
 * Every page file is validated (lib/effectors.ts) and every lane it lists is checked against
 * its replay files: the header parses as replay version 2 in product mode, shows the lane's
 * effector and part, and its pose and overlay payloads match the header; the bytes served are the
 * bytes the page states, at most the 300 KB lane budget. A broken export fails `pnpm build`.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import {
  type EffectorIndex,
  type EffectorPage,
  LANE_BUDGET_BYTES,
  parseEffectorIndex,
  parseEffectorPage,
} from "@/lib/effectors";
import { isV2, parseAnyReplayHeader, poseBytes } from "@/lib/replay";
import { DataError } from "@/lib/validate";

const DATA_DIR = path.join(process.cwd(), "public", "data");
const cache = new Map<string, unknown>();

function memo<T>(key: string, load: () => T): T {
  if (!cache.has(key)) cache.set(key, load());
  return cache.get(key) as T;
}

/** Check one page's lanes against their files under `dataDir`. */
export function checkLaneFiles(page: EffectorPage, file: string, dataDir: string = DATA_DIR): void {
  page.lanes.forEach((lane, i) => {
    const where = `$.lanes[${i}]`;
    const headerFile = path.join(dataDir, `${lane.replay}.json`);
    if (!existsSync(headerFile)) throw new DataError(file, where, `missing replay header ${lane.replay}.json`);
    const header = parseAnyReplayHeader(JSON.parse(readFileSync(headerFile, "utf8")), `${lane.replay}.json`);
    if (!isV2(header)) throw new DataError(file, where, "an effector lane must be replay version 2");
    if (!header.provenance.product_mode) throw new DataError(file, where, "an effector lane must be labelled product mode");
    if (header.meta.design_id !== lane.effector.effector_id) {
      throw new DataError(file, where, `lane ${lane.id} shows design ${header.meta.design_id}, not ${lane.effector.effector_id}`);
    }
    if (header.meta.part_id !== page.part.part_id) {
      throw new DataError(file, where, `lane ${lane.id} shows part ${header.meta.part_id}, not ${page.part.part_id}`);
    }
    if (header.n_frames !== lane.n_frames || header.fps !== lane.fps) {
      throw new DataError(file, where, `lane ${lane.id} frames or fps differ from its header`);
    }
    const dir = path.dirname(headerFile);
    const bin = path.join(dir, header.bin);
    if (!existsSync(bin) || statSync(bin).size !== poseBytes(header)) {
      throw new DataError(file, where, `${header.bin} is missing or does not match its header`);
    }
    let served = statSync(headerFile).size + statSync(bin).size;
    if (header.overlays) {
      const ovl = path.join(dir, header.overlays.bin);
      if (!existsSync(ovl) || statSync(ovl).size !== header.n_frames * header.overlays.channels.length * 2) {
        throw new DataError(file, where, `${header.overlays.bin} is missing or does not match its header`);
      }
      served += statSync(ovl).size;
    }
    if (Object.keys(header.assets).length > 0) throw new DataError(file, where, "effector lanes draw collision geoms only");
    if (served !== lane.bytes) throw new DataError(file, where, `lane ${lane.id} serves ${served} bytes; the page says ${lane.bytes}`);
    if (served > LANE_BUDGET_BYTES) throw new DataError(file, where, `lane ${lane.id} serves ${served} bytes, over ${LANE_BUDGET_BYTES}`);
  });
}

/** The index, or null when no tournament has been exported yet. */
export function getEffectorIndex(dataDir: string = DATA_DIR): EffectorIndex | null {
  return memo(`index:${dataDir}`, () => {
    const file = path.join(dataDir, "effectors", "index.json");
    if (!existsSync(file)) return null;
    return parseEffectorIndex(JSON.parse(readFileSync(file, "utf8")), "effectors/index.json");
  });
}

/** One part's page data, validated with its lanes. */
export function getEffectorPage(partId: string, dataDir: string = DATA_DIR): EffectorPage | undefined {
  return memo(`page:${dataDir}:${partId}`, () => {
    const index = getEffectorIndex(dataDir);
    const entry = index?.parts.find((p) => p.part_id === partId);
    if (!entry) return undefined;
    const rel = entry.page;
    const page = parseEffectorPage(JSON.parse(readFileSync(path.join(dataDir, rel), "utf8")), rel);
    if (page.part.part_id !== partId) throw new DataError(rel, "$.part.part_id", `expected ${partId}`);
    checkLaneFiles(page, rel, dataDir);
    return page;
  });
}

/** Part ids with a page (for generateStaticParams); every page file must be listed in the index. */
export function getEffectorPartIds(dataDir: string = DATA_DIR): string[] {
  const index = getEffectorIndex(dataDir);
  const ids = index ? index.parts.map((p) => p.part_id) : [];
  const dir = path.join(dataDir, "effectors");
  if (existsSync(dir)) {
    for (const name of readdirSync(dir)) {
      if (name.endsWith(".json") && name !== "index.json" && !ids.includes(name.slice(0, -5))) {
        throw new DataError(`effectors/${name}`, "$", "this page is not listed in effectors/index.json");
      }
    }
  }
  return ids;
}
