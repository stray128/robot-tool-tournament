/**
 * Build-time data access for server components and generateStaticParams.
 *
 * Reads public/data/*.json from disk (the same files the browser can fetch) and
 * validates them; a missing or malformed file fails the build with the file and
 * JSON path in the message. The files the indexes point at are checked too: every
 * lane's replay header must parse and match its payload size and frame count, and
 * every design's print files must exist. Only import this from server code.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import { POSE_STRIDE } from "./track.mjs";
import type { Bout, BoutsIndex, Design, DesignsIndex, JobsIndex, ResultsIndex } from "./types";
import { DataError, parseBouts, parseDesigns, parseJobs, parseReplayHeader, parseResults } from "./validate";

const DATA_DIR = path.join(process.cwd(), "public", "data");

function readJson(name: string): unknown {
  const file = path.join(DATA_DIR, name);
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    throw new Error(`cannot read public/data/${name}: ${reason}. Run \`pnpm import:demo\` to generate the demo data.`);
  }
}

const cache = new Map<string, unknown>();

function memo<T>(key: string, load: () => T): T {
  if (!cache.has(key)) cache.set(key, load());
  return cache.get(key) as T;
}

/**
 * Check that every lane's replay exists under `dataDir`, that its header is valid and
 * that the payload holds exactly the frames the header and the lane claim. Throws a
 * DataError naming the lane.
 */
export function checkReplayFiles(index: BoutsIndex, dataDir: string = DATA_DIR): void {
  index.bouts.forEach((bout, b) =>
    bout.lanes.forEach((lane, l) => {
      const where = `$.bouts[${b}].lanes[${l}].replay`;
      const headerFile = path.join(dataDir, `${lane.replay}.json`);
      if (!existsSync(headerFile)) throw new DataError("bouts.json", where, `missing replay header ${lane.replay}.json`);
      const header = parseReplayHeader(JSON.parse(readFileSync(headerFile, "utf8")), `${lane.replay}.json`);
      const binFile = path.join(path.dirname(headerFile), header.bin);
      if (!existsSync(binFile)) throw new DataError("bouts.json", where, `missing replay payload ${header.bin}`);
      const expected = header.n_frames * header.n_bodies * POSE_STRIDE * 4;
      const size = statSync(binFile).size;
      if (size !== expected) {
        throw new DataError("bouts.json", where, `${header.bin} is ${size} bytes; its header implies ${expected}`);
      }
      if (header.n_frames !== lane.n_frames) {
        throw new DataError("bouts.json", where, `replay has ${header.n_frames} frames; the lane says ${lane.n_frames}`);
      }
    }),
  );
}

/** Check that every print file a design links exists under `dataDir`. */
export function checkDesignFiles(index: DesignsIndex, dataDir: string = DATA_DIR): void {
  index.designs.forEach((design, d) => {
    for (const [key, rel] of Object.entries(design.files)) {
      if (rel && !existsSync(path.join(dataDir, rel))) {
        throw new DataError("designs.json", `$.designs[${d}].files.${key}`, `missing file ${rel}`);
      }
    }
  });
}

export function getBouts(): BoutsIndex {
  return memo("bouts", () => {
    const index = parseBouts(readJson("bouts.json"));
    checkReplayFiles(index);
    return index;
  });
}

export function getDesigns(): DesignsIndex {
  return memo("designs", () => {
    const index = parseDesigns(readJson("designs.json"));
    checkDesignFiles(index);
    return index;
  });
}

export function getJobs(): JobsIndex {
  return memo("jobs", () => parseJobs(readJson("jobs.json")));
}

export function getResults(): ResultsIndex {
  return memo("results", () => parseResults(readJson("results.json")));
}

export function getBout(id: string): Bout | undefined {
  return getBouts().bouts.find((b) => b.id === id);
}

export function getDesign(id: string): Design | undefined {
  return getDesigns().designs.find((d) => d.id === id);
}

/** The bout shown on the landing page: the first one marked featured, else the first. */
export function getFeaturedBout(): Bout | undefined {
  const { bouts } = getBouts();
  return bouts.find((b) => b.featured) ?? bouts[0];
}

/**
 * Short content hash of the published data files, shown in each page's title block
 * so a screenshot can be matched to the exact data it shows.
 */
export function getDataRevision(): string {
  return memo("revision", () => {
    const hash = createHash("sha256");
    for (const name of ["bouts.json", "designs.json", "jobs.json", "results.json"]) {
      hash.update(name);
      hash.update(readFileSync(path.join(DATA_DIR, name)));
    }
    return hash.digest("hex").slice(0, 8);
  });
}
