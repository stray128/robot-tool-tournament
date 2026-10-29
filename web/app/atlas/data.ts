/**
 * Build-time loader of the use-case atlas (server only: reads public/data/atlas.json).
 *
 * The atlas is screened again here, so a brand that slipped past the exporter fails `pnpm build`:
 * the effector contract's and the library's brand lists (read from the Python sources, as the
 * Studio pages do) plus the company and product names of the research, which
 * scripts/build_atlas_data.py writes to tests/data/atlas_brand_tokens.json. Outside the repository
 * only the file's own rules apply.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { type Atlas, atlasScreen, type HeroLinks, parseAtlas } from "@/lib/atlas";
import { DataError } from "@/lib/validate";

import { brandTokens, DEFAULT_DIRS, getStudioIndex, type StudioDirs } from "../studio/data";

const PUBLIC = path.join(process.cwd(), "public");
const REPO = path.join(process.cwd(), "..");
export const ATLAS_FILE = "atlas.json";
export const ATLAS_TOKENS = path.join("tests", "data", "atlas_brand_tokens.json");

/** The names the atlas must not show: the repository's brand lists and the research's names. */
export function atlasTokens(repo: string | null = REPO): { ci: string[]; cs: string[] } {
  if (!repo) return { ci: [], cs: [] };
  const ci = brandTokens(repo);
  const cs: string[] = [];
  const file = path.join(repo, ATLAS_TOKENS);
  if (existsSync(file)) {
    const doc = JSON.parse(readFileSync(file, "utf8")) as { ci?: string[]; cs?: string[] };
    ci.push(...(doc.ci ?? []));
    cs.push(...(doc.cs ?? []));
  }
  return { ci, cs };
}

let cached: Atlas | null = null;

/** The atlas, parsed and screened; a missing or brand-naming file fails the build. */
export function getAtlas(dataDir: string = path.join(PUBLIC, "data"), repo: string | null = REPO): Atlas {
  if (cached && dataDir === path.join(PUBLIC, "data")) return cached;
  const file = path.join(dataDir, ATLAS_FILE);
  if (!existsSync(file)) throw new DataError(ATLAS_FILE, "$", "the atlas data is missing (run scripts/build_atlas_data.py)");
  const atlas = parseAtlas(JSON.parse(readFileSync(file, "utf8")), ATLAS_FILE, atlasScreen(atlasTokens(repo)));
  if (dataDir === path.join(PUBLIC, "data")) cached = atlas;
  return atlas;
}

/**
 * The Studio heroes with a story page, by case id, for the atlas's links. A hero the export left
 * out has no entry, so its atlas task reads as researched, not simulated, and links nowhere.
 */
export function heroLinks(dirs: StudioDirs = DEFAULT_DIRS): HeroLinks {
  const out: HeroLinks = {};
  for (const h of getStudioIndex(dirs)?.heroes ?? []) out[h.case_id] = { hero: h.hero, title: h.title };
  return out;
}
