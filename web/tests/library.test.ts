/**
 * The effector library's pages (product mode): the index parser and its cross-checks, the file
 * checks run at build time, the catalog filter, the credits reader, and the brand rule on the
 * published library data and on the rendered /library pages (when web/out exists).
 */

import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { checkLibraryFiles, getLibraryIndex, getLibraryKeys, getPartTitles } from "@/app/library/data";
import {
  filterCards,
  filtersFromQuery,
  filtersToQuery,
  LIBRARY_LABEL,
  licenceText,
  mountText,
  NO_FILTERS,
  parseLibraryIndex,
  partTitleFallback,
  repoFileUrl,
  shortCommit,
  toCard,
  type LibraryCard,
} from "@/lib/library";
import { DataError } from "@/lib/validate";

const WEB = path.join(__dirname, "..");
const REPO = path.join(WEB, "..");
const PUBLIC = path.join(WEB, "public");
const INDEX = path.join(PUBLIC, "data", "library", "index.json");
const OUT = path.join(WEB, "out");
const published = existsSync(INDEX);
const tmp: string[] = [];

afterEach(() => {
  for (const d of tmp.splice(0)) rmSync(d, { recursive: true, force: true });
});

/** One tuple of tokens from a Python source file (never copied by hand). */
function pythonTokens(rel: string, name: string): string[] {
  const src = readFileSync(path.join(REPO, rel), "utf8");
  const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
  expect(block, `${name} in ${rel}`).not.toBeNull();
  return [...block![1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!);
}

/** The contract's BRAND_TOKENS plus the library's own tokens, whole words, any case. */
function brandPattern(): RegExp {
  const contract = pythonTokens("src/rtt/effectors/types.py", "BRAND_TOKENS");
  const library = pythonTokens("src/rtt/library/brands.py", "LIBRARY_ONLY_TOKENS");
  expect(contract.length).toBeGreaterThan(20);
  expect(library.length).toBeGreaterThan(20);
  const alt = [...new Set([...contract, ...library])]
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  return new RegExp(`(?<![a-z0-9])(${alt})(?![a-z0-9])`, "i");
}

/** URLs out (in attributes, text and the page's data payload), then tags out. */
function screenText(html: string): string {
  return html
    .replace(/https?:\/\/[^\s"'<>\\]+/g, " ")
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&");
}

/** The text a reader sees: scripts, styles and tags (with their attributes) out. */
function visibleText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]*>/g, " ");
}

function files(dir: string, ext: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    return e.isDirectory() ? files(full, ext) : e.name.endsWith(ext) ? [full] : [];
  });
}

function raw(): Record<string, any> {
  return JSON.parse(readFileSync(INDEX, "utf8"));
}

/** The JSON chunk of a binary glTF (node, mesh and material names live here). */
function glbJson(file: string): string {
  const buf = readFileSync(file);
  expect(buf.readUInt32LE(0)).toBe(0x46546c67); // "glTF"
  const len = buf.readUInt32LE(12);
  expect(buf.readUInt32LE(16)).toBe(0x4e4f534a); // "JSON"
  return buf.subarray(20, 20 + len).toString("utf8");
}

describe("catalog filter and labels", () => {
  const card = (over: Partial<LibraryCard>): LibraryCard => ({
    key: "x",
    display_name: "Item",
    summary: "",
    kind: "parallel_gripper",
    kind_title: "Parallel gripper",
    capabilities: ["pinch"],
    licences: ["MIT"],
    sim_status: "native",
    sim_title: "Native simulation model",
    iso_pattern: null,
    entrant: false,
    ...over,
  });
  const cards = [
    card({ key: "a", display_name: "Two-finger gripper", capabilities: ["pinch"], licences: ["MIT"], iso_pattern: "ISO 9409-1-50-4-M6", entrant: true }),
    card({ key: "b", display_name: "Five-finger hand", kind: "multi_finger_hand", kind_title: "Multi-finger hand", capabilities: ["power", "pinch"], licences: ["Apache-2.0"], sim_status: "proxy" }),
    card({ key: "c", display_name: "Drawer hook", summary: "Pulls a handle", kind: "passive_tool", kind_title: "Passive tool", capabilities: ["hook"], licences: ["CC-BY-4.0"], sim_status: "visual_only" }),
  ];
  const keys = (f: Partial<typeof NO_FILTERS>) => filterCards(cards, { ...NO_FILTERS, ...f }).map((c) => c.key);

  it("filters by text, capability (any of), licence, mount, sim status and entrants", () => {
    expect(keys({})).toEqual(["a", "b", "c"]);
    expect(keys({ query: "FIVE" })).toEqual(["b"]);
    expect(keys({ query: "hand" })).toEqual(["b", "c"]); // "handle" too: words match as substrings
    expect(keys({ query: "pulls handle" })).toEqual(["c"]);
    expect(keys({ query: "multi finger" })).toEqual(["b"]);
    expect(keys({ query: "no such thing" })).toEqual([]);
    expect(keys({ capabilities: ["power", "hook"] })).toEqual(["b", "c"]);
    expect(keys({ capabilities: ["pinch"] })).toEqual(["a", "b"]);
    expect(keys({ licence: "Apache-2.0" })).toEqual(["b"]);
    expect(keys({ mount: "iso" })).toEqual(["a"]);
    expect(keys({ mount: "own" })).toEqual(["b", "c"]);
    expect(keys({ sim: "visual_only" })).toEqual(["c"]);
    expect(keys({ entrantOnly: true })).toEqual(["a"]);
    expect(keys({ capabilities: ["pinch"], mount: "own" })).toEqual(["b"]);
  });

  it("keeps the filters in the URL query and ignores unknown values", () => {
    const known = { capabilities: ["pinch", "hook"], licences: ["MIT"] };
    const f = { query: "hand", capabilities: ["pinch", "hook"], licence: "MIT", mount: "iso" as const, sim: "proxy" as const, entrantOnly: true };
    const q = filtersToQuery(f);
    expect(q).toBe("?q=hand&cap=pinch%2Chook&licence=MIT&mount=iso&sim=proxy&entrant=1");
    expect(filtersFromQuery(q, known)).toEqual(f);
    expect(filtersToQuery(NO_FILTERS)).toBe("");
    expect(filtersFromQuery("?cap=laser,pinch&licence=GPL&mount=x&sim=y", known)).toEqual({ ...NO_FILTERS, capabilities: ["pinch"] });
  });

  it("speaks in plain words", () => {
    expect(mountText(null)).toBe("own interface");
    expect(mountText("ISO 9409-1-50-4-M6")).toBe("ISO 9409-1-50-4-M6");
    expect(licenceText("NONE")).toBe("none (no licence)");
    expect(licenceText("NOASSERTION")).toBe("not stated");
    expect(licenceText("LicenseRef-platform-eula")).toBe("custom terms (platform eula)");
    expect(licenceText("MIT")).toBe("MIT");
    expect(partTitleFallback("steel_slug")).toBe("Steel slug");
    expect(shortCommit("62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35")).toBe("62ab00f2");
    expect(repoFileUrl(null, "assets/x.stl")).toBeNull();
    expect(repoFileUrl("https://example.org/r/", "assets/a b/x.stl")).toBe("https://example.org/r/raw/main/assets/a%20b/x.stl");
    const titles = getPartTitles(["block", "steel_slug", "no_such_part"]);
    expect(titles.no_such_part).toBe("No such part");
    expect(titles.block).toBe("Block");
  });
});

describe.skipIf(!published)("published library data", () => {
  it("parses, labels every page and gives every item a route and a preview under public/", () => {
    const index = getLibraryIndex()!;
    expect(index).not.toBeNull();
    expect(index.label).toBe(LIBRARY_LABEL);
    expect(LIBRARY_LABEL).toBe("Product mode: not part of the pre-registered experiment");
    const keys = getLibraryKeys();
    expect(keys.length).toBe(index.counts.items);
    expect(new Set(keys).size).toBe(keys.length);
    for (const item of index.items) {
      expect(item.key).toMatch(/^[a-z0-9][a-z0-9-]*$/);
      expect(item.credit_anchor).toBe(`library-${item.key}`);
      expect(item.preview, item.key).not.toBeNull();
      expect(existsSync(path.join(PUBLIC, item.preview!.url)), item.preview!.url).toBe(true);
      if (item.mount.adapter) expect(keys).toContain(item.mount.adapter);
      expect(toCard(item).licences.length).toBeGreaterThan(0);
      if (item.sim.entrant) expect(item.sim.entrant_reason).toBeNull();
      else expect(item.sim.entrant_reason).toBeTruthy();
    }
  });

  it("refuses a malformed index", () => {
    const bad = (mutate: (d: Record<string, any>) => void, match: RegExp) => {
      const d = structuredClone(raw());
      mutate(d);
      expect(() => parseLibraryIndex(d, "x.json")).toThrow(DataError);
      expect(() => parseLibraryIndex(d, "x.json")).toThrow(match);
    };
    expect(() => parseLibraryIndex(raw(), "x.json")).not.toThrow();
    bad((d) => (d.label = "Benchmark result"), /label/);
    bad((d) => (d.format = "rtt-web-effectors"), /format/);
    bad((d) => (d.version = 2), /version/);
    bad((d) => d.items.push(structuredClone(d.items[0])), /duplicate item key|counts/);
    bad((d) => (d.counts.items += 1), /counts\.items/);
    bad((d) => (d.counts.excluded_red += 1), /excluded_red/);
    bad((d) => (d.items[0].credit_anchor = "elsewhere"), /credit_anchor/);
    bad((d) => (d.items[0].key = "Has Spaces"), /URL-safe key/);
    bad((d) => (d.items[0].preview = { url: "../../secret.glb", bytes: 1 }), /data\/library\/glb/);
    bad((d) => (d.items[0].capabilities = ["laser"]), /unknown capability/);
    bad((d) => (d.items[0].mount.adapter = "no-such-adapter"), /not a library item/);
    bad((d) => (d.items[0].mount.status = "certain"), /expected one of/);
    bad((d) => (d.items[0].sim.status = "perfect"), /expected one of/);
    bad((d) => (d.items[0].licences = []), /at least one licence/);
    bad((d) => (d.excluded[0].rating = "green"), /expected one of amber, red/);
    bad((d) => (d.listed[0].url = "ftp://example.org"), /http\(s\) URL/);
    bad((d) => {
      const i = d.items.find((x: any) => x.downloads.length > 0);
      i.downloads[0].path = "/etc/passwd";
    }, /repository path/);
    bad((d) => {
      const i = d.items.find((x: any) => x.specs.some((s: any) => s.tag === "sourced"));
      i.specs.find((s: any) => s.tag === "sourced").source = "";
    }, /source URL/);
    bad((d) => {
      const i = d.items.find((x: any) => x.sim.entrant);
      i.sim.entrant_reason = "because";
    }, /entrant/);
    bad((d) => {
      const i = d.items.find((x: any) => x.gate.ran);
      i.gate.passed = !i.gate.passed;
    }, /every entrant spec passes/);
    bad((d) => delete d.items[0].summary, /summary: expected a string/);
  });

  it("fails the build when a preview or a download does not match the index", () => {
    const index = parseLibraryIndex(raw());
    const item = index.items.find((i) => i.preview && i.downloads.length > 0)!;
    const pub = mkdtempSync(path.join(os.tmpdir(), "lib-pub-"));
    const repo = mkdtempSync(path.join(os.tmpdir(), "lib-repo-"));
    tmp.push(pub, repo);
    const one = { ...index, items: [item] };
    const glb = path.join(pub, item.preview!.url);
    mkdirSync(path.dirname(glb), { recursive: true });
    expect(() => checkLibraryFiles(one, pub, repo)).toThrow(/missing data\/library\/glb/);
    copyFileSync(path.join(PUBLIC, item.preview!.url), glb);
    expect(() => checkLibraryFiles(one, pub, repo)).not.toThrow(); // no assets/ here: downloads not checked
    const d = item.downloads[0]!;
    mkdirSync(path.join(repo, "assets", "effectors", "library"), { recursive: true });
    expect(() => checkLibraryFiles(one, pub, repo)).toThrow(/missing assets\/effectors\/library/);
    mkdirSync(path.dirname(path.join(repo, d.path)), { recursive: true });
    writeFileSync(path.join(repo, d.path), Buffer.alloc(d.bytes + 1));
    expect(() => checkLibraryFiles(one, pub, repo)).toThrow(/bytes; the index says/);
    writeFileSync(glb, Buffer.alloc(3));
    expect(() => checkLibraryFiles(one, pub, repo)).toThrow(DataError);
  });

  it("names no brand in the published library data or the previews' node names", () => {
    const re = brandPattern();
    const json = files(path.join(PUBLIC, "data", "library"), ".json");
    expect(json.length).toBeGreaterThan(0);
    for (const f of json) expect(readFileSync(f, "utf8").replace(/https?:\/\/[^\s"]+/g, " ").match(re)?.[1] ?? null, f).toBeNull();
    const glbs = files(path.join(PUBLIC, "data", "library", "glb"), ".glb");
    expect(glbs.length).toBe(getLibraryIndex()!.items.filter((i) => i.preview).length);
    for (const f of glbs) expect(glbJson(f).match(re)?.[1] ?? null, f).toBeNull();
  });

  it.skipIf(!existsSync(path.join(OUT, "library")))("names no brand and shows no URL on the rendered library pages, and anchors every credit", () => {
    const re = brandPattern();
    const index = getLibraryIndex()!;
    const pages = files(path.join(OUT, "library"), ".html");
    expect(pages.length).toBe(index.items.length + 1);
    for (const item of index.items) expect(existsSync(path.join(OUT, "library", item.key, "index.html")), item.key).toBe(true);
    for (const html of pages) {
      const text = readFileSync(html, "utf8");
      expect(screenText(text).match(re)?.[1] ?? null, html).toBeNull();
      // URLs stay in href attributes: no visible link text or prose shows one (it could name a maker).
      expect(visibleText(text).match(/\S*http\S*/i)?.[0] ?? null, html).toBeNull();
      expect(text, html).toContain(LIBRARY_LABEL);
    }
    const credits = readFileSync(path.join(OUT, "credits", "index.html"), "utf8");
    expect(credits).toContain('id="library"');
    for (const item of index.items) expect(credits, item.key).toContain(`id="${item.credit_anchor}"`);
  });
});

describe("library credits", () => {
  it("reads every item's SOURCE.json, and the effector credits skip the library", async () => {
    const { libraryCredits } = await import("@/components/library/LibraryCredits");
    const { effectorCredits } = await import("@/components/effectors/EffectorCredits");
    const root = mkdtempSync(path.join(os.tmpdir(), "lib-credits-"));
    tmp.push(root);
    const file = (over: Record<string, unknown>) => ({
      path: "assets/effectors/library/x/sim/x.xml",
      sha256: "1".repeat(64),
      size: 10,
      role: "derived",
      upstream_url: "https://example.org/models",
      upstream_commit: "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35",
      license: "MIT",
      copyright: "Copyright (c) 2026 Example Lab",
      attribution: "Example Lab gripper model, MIT",
      changes: ["converted from URDF"],
      status: "green",
      note: "",
      ...over,
    });
    const mk = (rel: string, body: unknown) => {
      mkdirSync(path.join(root, rel), { recursive: true });
      writeFileSync(path.join(root, rel, "SOURCE.json"), JSON.stringify(body));
    };
    mk("library/gripper-x", {
      key: "gripper-x",
      display_name: "Two-finger gripper",
      class: "open_hardware",
      upstream: { repository: "https://example.org/models", commit: "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35", extra: [{ repository: "https://example.org/cad", commit: "abc" }] },
      licence: [{ spdx: "MIT", covers: "the model" }],
      copyright: "Copyright (c) 2026 Example Lab",
      attribution: "Example Lab gripper, https://example.org/models, MIT",
      files: [file({}), file({ path: "b", changes: ["converted from URDF", "convex hulls of collision meshes"] }), file({ path: "c", changes: [] })],
      preview: { changes: ["re-encoded as GLB"] },
    });
    mk("library/plate-y", {
      key: "plate-y",
      display_name: "Tool plate",
      class: "own",
      upstream: { repository: "https://example.org/us", commit: "generated" },
      licence: [{ spdx: "Apache-2.0", covers: "our own work" }],
      copyright: "Copyright 2026 contributors",
      attribution: "Tool plate, generated, Apache-2.0",
      files: [file({ changes: [] })],
    });
    mk("catalog/hand-z", [file({ path: "assets/effectors/catalog/hand-z/h.xml" })]);
    const credits = libraryCredits(path.join(root, "library"));
    expect(credits.map((c) => [c.key, c.own])).toEqual([
      ["gripper-x", false],
      ["plate-y", true],
    ]);
    expect(credits[0]!.changes).toEqual([
      { text: "converted from URDF", files: 2 },
      { text: "convex hulls of collision meshes", files: 1 },
    ]);
    expect(credits[0]!.extra).toEqual([{ repository: "https://example.org/cad", commit: "abc" }]);
    expect(credits[0]!.preview_changes).toEqual(["re-encoded as GLB"]);
    expect(credits[1]!.upstream).toBeNull();
    expect(libraryCredits(path.join(root, "missing"))).toEqual([]);
    expect(effectorCredits(root).map((c) => [c.family, c.model])).toEqual([["catalog", "hand-z"]]);
  });

  it.skipIf(!existsSync(path.join(REPO, "assets", "effectors", "library")) || !published)(
    "credits every published item under its anchor",
    async () => {
      const { libraryCredits } = await import("@/components/library/LibraryCredits");
      const credited = new Set(libraryCredits().map((c) => c.key));
      for (const item of getLibraryIndex()!.items) expect(credited.has(item.key), item.key).toBe(true);
    },
  );
});
