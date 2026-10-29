/**
 * Build-time loaders of the Studio pages (server only: reads public/data from disk).
 *
 * Every document is validated (components/studio/schema.ts): the product-mode label, the
 * contract's fields and cross-field rules, URLs only in `url` fields and, when the repository's
 * brand lists can be read, no brand token in any shown string. Then every file a page points at
 * is checked on disk: each lane's replay header (version 2, product mode, the lane's setup and
 * hero, its frames and fps) and payloads (the bytes served are the bytes the page states), each
 * GLB it draws (present under public/assets with the listed size and sha256), the build pack zip
 * (present, its bytes and sha256, at most 2 MB) and the recorded session (it parses). A broken
 * export fails `pnpm build` with a DataError naming the file and the JSON path.
 */

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import {
  brandPattern,
  type HeroEntry,
  type HeroPage,
  laneHeaderProblems,
  PACK_CAP_BYTES,
  parseStudioHero,
  parseStudioIndex,
  type StudioIndex,
} from "@/components/studio/schema";
import { isV2, parseAnyReplayHeader, poseBytes, referencedAssetFiles } from "@/lib/replay";
import { parseRecordedSession, type RecordedSession } from "@/lib/session";
import { DataError } from "@/lib/validate";

/** Where the site's data and assets live (public/data and public/assets of web/). */
export interface StudioDirs {
  data: string;
  /** The site root the GLBs' `uri` (assets/<id>.glb) is relative to: web/public. */
  site: string;
  /** The repository root, for the brand lists in the Python sources (null skips the brand rule). */
  repo: string | null;
}

const PUBLIC = path.join(process.cwd(), "public");
export const DEFAULT_DIRS: StudioDirs = { data: path.join(PUBLIC, "data"), site: PUBLIC, repo: path.join(process.cwd(), "..") };

const cache = new Map<string, unknown>();

function memo<T>(key: string, load: () => T): T {
  if (!cache.has(key)) cache.set(key, load());
  return cache.get(key) as T;
}

function readJson(file: string, rel: string): unknown {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch (err) {
    throw new DataError(rel, "$", `not readable JSON: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function sha256(file: string): string {
  return createHash("sha256").update(readFileSync(file)).digest("hex");
}

/** One tuple of tokens from a Python source file, as the smoke checks read them (never copied by hand). */
function pythonTokens(file: string, name: string): string[] {
  if (!existsSync(file)) return [];
  const src = readFileSync(file, "utf8");
  const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
  return block ? [...block[1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!) : [];
}

/**
 * The brand tokens of the contract and the library (src/rtt/effectors/types.py BRAND_TOKENS and
 * src/rtt/library/brands.py LIBRARY_ONLY_TOKENS), or an empty list outside the repository.
 */
export function brandTokens(repo: string | null): string[] {
  if (!repo) return [];
  return [
    ...pythonTokens(path.join(repo, "src", "rtt", "effectors", "types.py"), "BRAND_TOKENS"),
    ...pythonTokens(path.join(repo, "src", "rtt", "library", "brands.py"), "LIBRARY_ONLY_TOKENS"),
  ];
}

function brandOf(dirs: StudioDirs): RegExp | null {
  return memo(`brand:${dirs.repo}`, () => brandPattern(brandTokens(dirs.repo)));
}

/** Check every lane of a page against its replay files and the GLB assets it draws. */
export function checkLaneFiles(page: HeroPage, file: string, dirs: StudioDirs = DEFAULT_DIRS): void {
  page.lanes.forEach((lane, i) => {
    const where = `$.lanes[${i}]`;
    const fail = (detail: string): never => {
      throw new DataError(file, where, `lane ${lane.id}: ${detail}`);
    };
    const headerFile = path.join(dirs.data, `${lane.replay}.json`);
    if (!existsSync(headerFile)) fail(`missing replay header ${lane.replay}.json`);
    const header = parseAnyReplayHeader(readJson(headerFile, `${lane.replay}.json`), `${lane.replay}.json`);
    if (!isV2(header)) return fail("a Studio lane must be replay version 2");
    if (!header.provenance.product_mode) fail("a Studio lane must be labelled product mode");
    if (header.meta.design_id !== lane.setup_id) fail(`the replay shows setup ${header.meta.design_id.slice(0, 12)}, not ${lane.setup_id.slice(0, 12)}`);
    if (header.meta.part_id !== page.hero) fail(`the replay belongs to ${header.meta.part_id}, not ${page.hero}`);
    const metaSha = (header.meta as Record<string, unknown>).traj_sha;
    if (typeof metaSha === "string" && metaSha !== lane.traj_sha) fail("the replay's traj_sha differs from the lane's");
    // Frames, fps, encoding, keyframing error, clock and cuts (a keyframed lane's clock is its `t`).
    const problems = laneHeaderProblems(lane, header);
    if (problems.length) fail(problems.join("; "));
    const dir = path.dirname(headerFile);
    const bin = path.join(dir, header.bin);
    const poses = poseBytes(header);
    if (!existsSync(bin) || statSync(bin).size !== poses) fail(`${header.bin} is missing or is not ${poses} bytes`);
    let served = statSync(headerFile).size + poses;
    if (header.overlays) {
      const ovl = path.join(dir, header.overlays.bin);
      const size = header.n_frames * header.overlays.channels.length * 2;
      if (!existsSync(ovl) || statSync(ovl).size !== size) fail(`${header.overlays.bin} is missing or is not ${size} bytes`);
      served += size;
    }
    if (served !== lane.bytes) fail(`serves ${served} bytes; the page says ${lane.bytes}`);
    let assetBytes = 0;
    for (const a of referencedAssetFiles(header)) {
      const glb = path.join(dirs.site, a.uri);
      if (!existsSync(glb)) fail(`draws ${a.uri}, which is not in the asset store`);
      const size = statSync(glb).size;
      if (size !== a.bytes) fail(`${a.uri} is ${size} bytes; the header says ${a.bytes}`);
      if (sha256(glb) !== a.sha256) fail(`${a.uri} does not match its sha256`);
      assetBytes += size;
    }
    if (assetBytes !== lane.asset_bytes) fail(`draws ${assetBytes} bytes of GLB assets; the page says ${lane.asset_bytes}`);
  });
}

/** Check the build pack zip a page links to. */
export function checkBuildPack(page: HeroPage, file: string, dirs: StudioDirs = DEFAULT_DIRS): void {
  const pack = page.build_pack;
  if (!pack) return;
  const zip = path.join(dirs.data, pack.zip);
  const where = "$.build_pack";
  if (!existsSync(zip)) throw new DataError(file, where, `missing ${pack.zip}`);
  const size = statSync(zip).size;
  if (size !== pack.bytes) throw new DataError(file, where, `${pack.zip} is ${size} bytes; the page says ${pack.bytes}`);
  if (size > PACK_CAP_BYTES) throw new DataError(file, where, `${pack.zip} is ${size} bytes, over the ${PACK_CAP_BYTES} the site serves`);
  if (sha256(zip) !== pack.sha256) throw new DataError(file, where, `${pack.zip} does not match its sha256`);
}

/** The index, or null when no use case has been exported yet. */
export function getStudioIndex(dirs: StudioDirs = DEFAULT_DIRS): StudioIndex | null {
  return memo(`index:${dirs.data}`, () => {
    const file = path.join(dirs.data, "studio", "index.json");
    if (!existsSync(file)) return null;
    return parseStudioIndex(readJson(file, "studio/index.json"), "studio/index.json", { brand: brandOf(dirs) });
  });
}

/** One hero's page, validated with every file it points at; undefined for an unknown hero. */
export function getStudioHero(hero: string, dirs: StudioDirs = DEFAULT_DIRS): HeroPage | undefined {
  return memo(`hero:${dirs.data}:${hero}`, () => {
    const entry = getStudioIndex(dirs)?.heroes.find((h) => h.hero === hero);
    if (!entry) return undefined;
    const rel = entry.page;
    const file = path.join(dirs.data, rel);
    if (!existsSync(file)) throw new DataError("studio/index.json", "$.heroes", `${rel} is listed but missing`);
    const page = parseStudioHero(readJson(file, rel), rel, { brand: brandOf(dirs) });
    const same = (key: keyof typeof entry & keyof HeroPage) => {
      if (page[key] !== entry[key]) throw new DataError(rel, `$.${key}`, `differs from the index (${JSON.stringify(entry[key])})`);
    };
    for (const key of ["hero", "case_id", "title", "setting", "use_case", "outcome", "headline"] as const) same(key);
    if (page.plan.task_id !== entry.task_id) throw new DataError(rel, "$.plan.task_id", "differs from the index");
    if (page.plan.steps.length !== entry.n_steps) throw new DataError(rel, "$.plan.steps", `${page.plan.steps.length} steps; the index says ${entry.n_steps}`);
    if (page.screen.n_candidates !== entry.n_candidates) throw new DataError(rel, "$.screen.n_candidates", "differs from the index");
    if ((page.recommendation?.setup_id ?? null) !== (entry.recommended?.setup_id ?? null)) {
      throw new DataError(rel, "$.recommendation", "the recommended setup differs from the index");
    }
    if ((page.best_single?.setup_id ?? null) !== (entry.best_single?.setup_id ?? null)) {
      throw new DataError(rel, "$.best_single", "the best single setup differs from the index");
    }
    checkLaneFiles(page, rel, dirs);
    checkBuildPack(page, rel, dirs);
    if (page.session) getStudioSession(page, dirs);
    return page;
  });
}

/** The recorded session a page names, parsed (lib/session.ts), or null. */
export function getStudioSession(page: HeroPage, dirs: StudioDirs = DEFAULT_DIRS): RecordedSession | null {
  const block = page.session;
  if (!block) return null;
  return memo(`session:${dirs.data}:${block.file}`, () => {
    const file = path.join(dirs.data, block.file);
    if (!existsSync(file)) throw new DataError(`studio/${page.hero}.json`, "$.session.file", `missing ${block.file}`);
    try {
      return parseRecordedSession(readJson(file, block.file), block.file);
    } catch (err) {
      if (err instanceof DataError) throw err;
      throw new DataError(block.file, "$", err instanceof Error ? err.message : String(err));
    }
  });
}

/** Hero ids with a page (for generateStaticParams); every page file must be listed in the index. */
export function getStudioHeroIds(dirs: StudioDirs = DEFAULT_DIRS): string[] {
  const index = getStudioIndex(dirs);
  const ids = index ? index.heroes.map((h) => h.hero) : [];
  const dir = path.join(dirs.data, "studio");
  if (existsSync(dir)) {
    for (const name of readdirSync(dir)) {
      if (name.endsWith(".json") && name !== "index.json" && !ids.includes(name.slice(0, -5))) {
        throw new DataError(`studio/${name}`, "$", "this page is not listed in studio/index.json");
      }
    }
  }
  return ids;
}

/**
 * The published v1 hero routes (/studio/<hero>/) and the case each showed. When an export renames
 * a hero (v2 heroes are named by their case id) or leaves one out, its old route stays as a
 * redirect: to the hero with the same case, else to the worked examples.
 */
export const V1_HERO_ROUTES: Readonly<Record<string, string>> = { h1: "h1_tote", h2: "h2_coupling", h3: "h3_cabinet" };

/** Old routes the index no longer lists, and where each now leads (relative to /studio/<old>/). */
export function legacyHeroRoutes(dirs: StudioDirs = DEFAULT_DIRS): { hero: string; to: string; title: string | null }[] {
  const heroes = getStudioIndex(dirs)?.heroes ?? [];
  return Object.entries(V1_HERO_ROUTES)
    .filter(([old]) => !heroes.some((h) => h.hero === old))
    .map(([old, caseId]) => {
      const now = heroes.find((h) => h.case_id === caseId);
      return { hero: old, to: now ? `../${now.hero}/` : "../#examples", title: now?.title ?? null };
    });
}

/** The case the home page and the footer feature: the returns tote when it is listed, else the first hero with a recommendation. */
export const FEATURED_CASE = "h1_tote";

/** The featured hero's index entry, or null when no use case is exported. */
export function featuredHero(dirs: StudioDirs = DEFAULT_DIRS): HeroEntry | null {
  const heroes = getStudioIndex(dirs)?.heroes ?? [];
  return heroes.find((h) => h.case_id === FEATURED_CASE && h.recommended) ?? heroes.find((h) => h.recommended) ?? heroes[0] ?? null;
}
