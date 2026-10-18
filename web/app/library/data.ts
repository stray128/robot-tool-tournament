/**
 * Build-time loader of the library pages (server only: reads public/data from disk).
 *
 * The index is validated (lib/library.ts) and checked against the files it names: every
 * preview GLB is served from public/ with the stated size, and, when the repository's
 * assets/effectors/library/ is present, every download exists there with the stated size.
 * A broken export fails `pnpm build`.
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

import { getEffectorIndex } from "@/app/effectors/data";
import { type LibraryIndex, type LibraryItem, parseLibraryIndex, partTitleFallback } from "@/lib/library";
import { DataError } from "@/lib/validate";

const PUBLIC_DIR = path.join(process.cwd(), "public");
const REPO_DIR = path.join(process.cwd(), "..");
const FILE = "library/index.json";
const cache = new Map<string, LibraryIndex | null>();

/** Check the previews and downloads an index names against the files on disk. */
export function checkLibraryFiles(index: LibraryIndex, publicDir: string = PUBLIC_DIR, repoDir: string = REPO_DIR): void {
  const assets = path.join(repoDir, "assets", "effectors", "library");
  const haveAssets = existsSync(assets);
  index.items.forEach((item, i) => {
    const where = `$.items[${i}]`;
    if (item.preview) {
      const glb = path.join(publicDir, item.preview.url);
      if (!existsSync(glb)) throw new DataError(FILE, `${where}.preview`, `missing ${item.preview.url} under public/`);
      const size = statSync(glb).size;
      if (size !== item.preview.bytes) throw new DataError(FILE, `${where}.preview`, `${item.preview.url} is ${size} bytes; the index says ${item.preview.bytes}`);
    }
    if (!haveAssets) return;
    item.downloads.forEach((d, j) => {
      const file = path.join(repoDir, d.path);
      if (!existsSync(file)) throw new DataError(FILE, `${where}.downloads[${j}]`, `missing ${d.path}`);
      const size = statSync(file).size;
      if (size !== d.bytes) throw new DataError(FILE, `${where}.downloads[${j}]`, `${d.path} is ${size} bytes; the index says ${d.bytes}`);
    });
  });
}

/** The library index, or null when it has not been exported yet. */
export function getLibraryIndex(publicDir: string = PUBLIC_DIR, repoDir: string = REPO_DIR): LibraryIndex | null {
  const memo = `${publicDir}\n${repoDir}`;
  if (!cache.has(memo)) {
    const file = path.join(publicDir, "data", "library", "index.json");
    if (!existsSync(file)) cache.set(memo, null);
    else {
      const index = parseLibraryIndex(JSON.parse(readFileSync(file, "utf8")), FILE);
      checkLibraryFiles(index, publicDir, repoDir);
      cache.set(memo, index);
    }
  }
  return cache.get(memo) ?? null;
}

/** Item keys with a page (for generateStaticParams). */
export function getLibraryKeys(): string[] {
  return getLibraryIndex()?.items.map((i) => i.key) ?? [];
}

export function getLibraryItem(key: string): LibraryItem | undefined {
  return getLibraryIndex()?.items.find((i) => i.key === key);
}

/** Titles of the training parts the gate builds on (from the effector index, else from the ids). */
export function getPartTitles(ids: Iterable<string>): Record<string, string> {
  let titles = new Map<string, string>();
  try {
    titles = new Map((getEffectorIndex()?.parts ?? []).map((p) => [p.part_id, p.title]));
  } catch {
    // A broken effector export fails its own pages; here the ids in words are enough.
  }
  return Object.fromEntries([...new Set(ids)].map((id) => [id, titles.get(id) ?? partTitleFallback(id)]));
}
