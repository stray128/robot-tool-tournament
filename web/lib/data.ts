/**
 * Build-time data access for server components and generateStaticParams.
 *
 * Reads public/data/*.json from disk (the same files the browser can fetch) and
 * validates them; a missing or malformed file fails the build with the file and
 * JSON path in the message. The files the indexes point at are checked too: every
 * lane's replay header must parse and match its payload size and frame count (a
 * version 2 header also its overlay sidecar and every GLB asset it draws, which must
 * be a GLB the browser can decode), and every design's print files must exist. Only import this from server code.
 */

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import { readGlb, unsupportedExtensions } from "./assets";
import { type CertificateDoc, parseCertificateDoc } from "./certificate";
import { parseRecordedSession, type RecordedSession, SessionError, type SessionSummary, summarizeSession } from "./session";
import { isV2, parseAnyReplayHeader, referencedAssetFiles } from "./replay";
import { POSE_STRIDE } from "./track.mjs";
import type { Bout, BoutsIndex, Design, DesignsIndex, JobsIndex, ResultsIndex } from "./types";
import { DataError, parseBouts, parseDesigns, parseJobs, parseResults } from "./validate";

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
      const header = parseAnyReplayHeader(JSON.parse(readFileSync(headerFile, "utf8")), `${lane.replay}.json`);
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
      if (isV2(header)) {
        if (header.overlays) {
          const ovl = path.join(path.dirname(headerFile), header.overlays.bin);
          const want = header.n_frames * header.overlays.channels.length * 2;
          if (!existsSync(ovl)) throw new DataError("bouts.json", where, `missing replay overlays ${header.overlays.bin}`);
          if (statSync(ovl).size !== want) {
            throw new DataError("bouts.json", where, `${header.overlays.bin} is ${statSync(ovl).size} bytes; its header implies ${want}`);
          }
        }
        // Assets are served from the site root (public/assets), next to public/data.
        for (const { id, uri, bytes, sha256 } of referencedAssetFiles(header)) {
          const file = path.join(dataDir, "..", uri);
          if (!existsSync(file)) throw new DataError("bouts.json", where, `replay draws asset ${id}, but ${uri} is missing`);
          const content = readFileSync(file);
          if (content.byteLength !== bytes || createHash("sha256").update(content).digest("hex") !== sha256) {
            throw new DataError("bouts.json", where, `${uri} does not match the size and sha256 its replay lists`);
          }
          // The same container and extension checks the browser runs (lib/assets.ts verifyAsset),
          // so a file the page would refuse (Draco, not a GLB) fails the build instead.
          let missing: string[];
          try {
            missing = unsupportedExtensions(readGlb(content.buffer.slice(content.byteOffset, content.byteOffset + content.byteLength)).json);
          } catch (err) {
            throw new DataError("bouts.json", where, `${uri}: ${err instanceof Error ? err.message : String(err)}`);
          }
          if (missing.length > 0) {
            throw new DataError("bouts.json", where, `${uri} needs ${missing.join(", ")}, which this site does not decode`);
          }
        }
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

/** JSON files directly inside public/data/<dir>, sorted by name; [] when the directory is absent. */
function listJson(dir: string, dataDir: string = DATA_DIR): string[] {
  const full = path.join(dataDir, dir);
  if (!existsSync(full)) return [];
  return readdirSync(full, { withFileTypes: true })
    .filter((e) => e.isFile() && e.name.endsWith(".json"))
    .map((e) => e.name)
    .sort();
}

/**
 * Recorded Grasp Engineer sessions (public/data/sessions/*.json), validated, each
 * part STL checked to exist. Recorded runs sort before illustrative ones.
 */
export function getSessions(dataDir: string = DATA_DIR): RecordedSession[] {
  return memo(`sessions:${dataDir}`, () => {
    const sessions = listJson("sessions", dataDir).map((name) => {
      const rel = `sessions/${name}`;
      const session = parseRecordedSession(JSON.parse(readFileSync(path.join(dataDir, rel), "utf8")), rel);
      if (session.part.stl && !existsSync(path.join(dataDir, session.part.stl))) {
        throw new SessionError(rel, `missing part file ${session.part.stl}`);
      }
      return session;
    });
    const ids = new Set<string>();
    for (const s of sessions) {
      if (ids.has(s.id)) throw new SessionError("sessions/", `duplicate session id ${s.id}`);
      ids.add(s.id);
    }
    return sessions.sort((a, b) => (a.kind === b.kind ? a.id.localeCompare(b.id) : a.kind === "recorded" ? -1 : 1));
  });
}

export function getSessionSummaries(): SessionSummary[] {
  return getSessions().map(summarizeSession);
}

/** Robustness certificates (public/data/certificates/<id>.json), validated. */
export function getCertificates(dataDir: string = DATA_DIR): CertificateDoc[] {
  return memo(`certificates:${dataDir}`, () =>
    listJson("certificates", dataDir).map((name) => {
      const id = name.replace(/\.json$/, "");
      if (!/^[a-z0-9][a-z0-9._-]*$/.test(id)) throw new DataError(`certificates/${name}`, "$", "file name is not a URL-safe id");
      return parseCertificateDoc(JSON.parse(readFileSync(path.join(dataDir, "certificates", name), "utf8")), id);
    }),
  );
}

export function getCertificate(id: string): CertificateDoc | undefined {
  return getCertificates().find((c) => c.id === id);
}

/** Certificate ids by design id, for linking a design to its certificate page. */
export function certificateIdsByDesign(): Map<string, string> {
  const out = new Map<string, string>();
  for (const c of getCertificates()) if (!out.has(c.design_id)) out.set(c.design_id, c.id);
  return out;
}

/**
 * Short content hash of the published data files, shown in each page's title block
 * so a screenshot can be matched to the exact data it shows.
 */
export function getDataRevision(): string {
  return memo("revision", () => {
    const hash = createHash("sha256");
    const names = [
      "bouts.json",
      "designs.json",
      "jobs.json",
      "results.json",
      ...listJson("sessions").map((n) => `sessions/${n}`),
      ...listJson("certificates").map((n) => `certificates/${n}`),
    ];
    for (const name of names) {
      hash.update(name);
      hash.update(readFileSync(path.join(DATA_DIR, name)));
    }
    return hash.digest("hex").slice(0, 8);
  });
}
