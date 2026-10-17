/**
 * The effector pages' data (product mode): parsers, the lane checks run at build time, the labels,
 * and the brand rule on everything published under public/data/effectors* and on the rendered
 * effector pages (when web/out exists).
 */

import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { checkLaneFiles, getEffectorIndex, getEffectorPage, getEffectorPartIds } from "@/app/effectors/data";
import {
  defaultLanes,
  LANE_BUDGET_BYTES,
  outcomeLabel,
  parseEffectorIndex,
  parseEffectorPage,
  PRODUCT_MODE_LABEL,
  roleLabel,
  toleranceText,
  type EffectorPage,
} from "@/lib/effectors";
import { DataError } from "@/lib/validate";

const WEB = path.join(__dirname, "..");
const DATA = path.join(WEB, "public", "data");
const INDEX = path.join(DATA, "effectors", "index.json");
const published = existsSync(INDEX);
const tmp: string[] = [];

afterEach(() => {
  for (const d of tmp.splice(0)) rmSync(d, { recursive: true, force: true });
});

/** BRAND_TOKENS of the contract, read from src/rtt/effectors/types.py (never copied by hand). */
function brandPattern(): RegExp {
  const src = readFileSync(path.join(WEB, "..", "src", "rtt", "effectors", "types.py"), "utf8");
  const block = src.match(/BRAND_TOKENS: Final\[tuple\[str, \.\.\.\]\] = \(([\s\S]*?)\n\)/);
  expect(block).not.toBeNull();
  const tokens = [...block![1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!);
  expect(tokens.length).toBeGreaterThan(20);
  const alt = tokens
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  return new RegExp(`(?<![a-z0-9])(${alt})(?![a-z0-9])`, "i");
}

function files(dir: string, ext: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    return e.isDirectory() ? files(full, ext) : e.name.endsWith(ext) ? [full] : [];
  });
}

function firstPage(): { id: string; raw: Record<string, unknown> } {
  const index = JSON.parse(readFileSync(INDEX, "utf8"));
  const id = index.parts[0].part_id as string;
  return { id, raw: JSON.parse(readFileSync(path.join(DATA, "effectors", `${id}.json`), "utf8")) };
}

describe("labels and formatting", () => {
  it("speaks in effector words", () => {
    expect(outcomeLabel(true, true, null)).toBe("Registered");
    expect(outcomeLabel(false, true, "not_registered")).toBe("Held, not registered");
    expect(outcomeLabel(false, false, "physics_unreliable")).toBe("Penetration over 1 mm");
    expect(toleranceText({ x: 3, y: 3 })).toBe("±3 mm both axes");
    expect(toleranceText({ x: 0, y: 3 })).toBe("0 mm across, ±3 mm along");
    expect(toleranceText({ x: null, y: 3 })).toBe("none");
    expect(roleLabel("best_suction")).toBe("Best Suction");
    expect(roleLabel("runner_up")).toBe("Rigid runner-up");
  });

  it("shows the recommendation and group bests first, other finalists on request", () => {
    const lane = (id: string, roles: string[]) => ({ id, roles }) as unknown as EffectorPage["lanes"][number];
    const lanes = [lane("a", ["recommended", "finalist"]), lane("b", ["finalist"]), lane("c", ["best_catalog", "finalist"])];
    expect(defaultLanes(lanes).map((l) => l.id)).toEqual(["a", "c"]);
    expect(defaultLanes([lane("b", ["finalist"])]).map((l) => l.id)).toEqual(["b"]);
  });
});

describe.skipIf(!published)("published effector data", () => {
  it("parses, lists every page and passes the lane checks", () => {
    const index = getEffectorIndex();
    expect(index?.label).toBe(PRODUCT_MODE_LABEL);
    const ids = getEffectorPartIds();
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) {
      const page = getEffectorPage(id)!;
      expect(page.label).toBe(PRODUCT_MODE_LABEL);
      expect(page.fairness.identical).toBe(true);
      expect(page.families.every((f) => f.label === PRODUCT_MODE_LABEL)).toBe(true);
      for (const lane of page.lanes) expect(lane.bytes).toBeLessThanOrEqual(LANE_BUDGET_BYTES);
      expect(page.lanes[0]!.roles[0]).toMatch(/^(recommended|best_|finalist)/);
      const winner = page.recommendation.winner;
      if (winner?.effector.fidelity === "approximation") {
        expect(winner.caveat).toBe("approximation: verify on hardware");
        expect(page.recommendation.runner_up).not.toBeNull();
      }
      for (const g of page.groups) {
        expect(g.budget.screen_episodes).toBeLessThanOrEqual(96);
        expect(g.budget.certificates).toBeLessThanOrEqual(2);
      }
    }
  });

  it("prints the low-grip note wherever the rigid-pad gripper A appears, and parses older data without notes", () => {
    const LOW_GRIP = /^Low grip at published settings/;
    let seen = 0;
    for (const id of getEffectorPartIds()) {
      const page = getEffectorPage(id)!;
      const rigidA = new Set(page.candidates.filter((c) => c.model === "gripper-2f-rigid-a").map((c) => c.effector_id));
      const rows = [...page.by_requirement.held.ranking, ...page.by_requirement.registered.ranking];
      for (const r of rows) {
        if (rigidA.has(r.effector_id)) {
          expect(r.note).toMatch(LOW_GRIP);
          seen += 1;
        } else expect(r.note).toBeNull();
      }
      for (const f of page.families) if (f.best && rigidA.has(f.best.effector_id)) expect(f.best.note).toMatch(LOW_GRIP);
      for (const lane of page.lanes) expect(lane.effector.note === null || rigidA.has(lane.effector.effector_id)).toBe(true);
    }
    expect(seen).toBeGreaterThan(0);
    const { raw } = firstPage();
    const old = structuredClone(raw) as Record<string, any>;
    for (const row of old.recommendation.ranking) delete row.note;
    delete old.recommendation.winner?.effector?.note;
    const parsed = parseEffectorPage(old, "old.json");
    expect(parsed.recommendation.ranking.every((r) => r.note === null)).toBe(true);
  });

  it("names no brand in any published effector file", () => {
    const re = brandPattern();
    const roots = [path.join(DATA, "effectors"), ...readdirSync(path.join(DATA, "replays")).filter((d) => d.startsWith("effectors-")).map((d) => path.join(DATA, "replays", d))];
    const hits = roots.flatMap((r) => files(r, ".json")).filter((f) => re.test(readFileSync(f, "utf8").replace(/https?:\/\/\S+/g, " ")));
    expect(hits).toEqual([]);
  });

  it.skipIf(!existsSync(path.join(WEB, "out", "effectors")))("names no brand on the rendered effector pages", () => {
    const re = brandPattern();
    const pages = files(path.join(WEB, "out", "effectors"), ".html");
    expect(pages.length).toBeGreaterThan(0);
    for (const html of pages) {
      const text = readFileSync(html, "utf8");
      expect(text.match(re)?.[1] ?? null, html).toBeNull();
      expect(text, html).toContain(PRODUCT_MODE_LABEL);
    }
  });

  it("refuses a page with the wrong label, an over-budget lane or an unaccompanied approximation", () => {
    const { raw } = firstPage();
    const bad = (mutate: (p: Record<string, any>) => void, match: RegExp) => {
      const p = structuredClone(raw) as Record<string, any>;
      mutate(p);
      expect(() => parseEffectorPage(p, "x.json")).toThrow(match);
    };
    bad((p) => (p.label = "Benchmark result"), /label/);
    bad((p) => (p.lanes[0].bytes = LANE_BUDGET_BYTES + 1), /at most 300000 bytes/);
    bad((p) => (p.lanes[0].replay = "replays/m1/straight"), /under replays\/effectors-/);
    bad((p) => (p.lanes = []), /at least one lane/);
    bad((p) => {
      p.recommendation.winner = null;
      p.recommendation.no_winner = null;
    }, /winner or a no-winner/);
    bad((p) => {
      p.recommendation.winner.effector.fidelity = "approximation";
      p.recommendation.runner_up_rigid = null;
    }, /rigid runner-up/);
    bad((p) => (p.groups[0].budget.screen_episodes = 97), /more than its budget/);
    const index = JSON.parse(readFileSync(INDEX, "utf8"));
    index.parts[0].page = "effectors/elsewhere.json";
    expect(() => parseEffectorIndex(index)).toThrow(/points at/);
  });

  it("fails the build when a lane's files do not match the page", () => {
    const { id, raw } = firstPage();
    const page = parseEffectorPage(raw, "p.json");
    const dir = mkdtempSync(path.join(os.tmpdir(), "eff-"));
    tmp.push(dir);
    cpSync(path.join(DATA, "replays", `effectors-${id}`), path.join(dir, "replays", `effectors-${id}`), { recursive: true });
    expect(() => checkLaneFiles(page, "p.json", dir)).not.toThrow();
    const lane = page.lanes[0]!;
    const header = path.join(dir, `${lane.replay}.json`);
    const doc = JSON.parse(readFileSync(header, "utf8"));
    writeFileSync(header, JSON.stringify({ ...doc, meta: { ...doc.meta, design_id: "f".repeat(64) } }));
    expect(() => checkLaneFiles(page, "p.json", dir)).toThrow(DataError);
    writeFileSync(header, JSON.stringify(doc));
    const bin = path.join(dir, `${lane.replay}.bin`);
    writeFileSync(bin, Buffer.alloc(statSync(bin).size + 28));
    expect(() => checkLaneFiles(page, "p.json", dir)).toThrow(/does not match its header/);
  });
});

describe("effector credits", () => {
  it("renders every vendored model's SOURCE.json, in either layout", async () => {
    const { effectorCredits } = await import("@/components/effectors/EffectorCredits");
    const root = mkdtempSync(path.join(os.tmpdir(), "eff-credits-"));
    tmp.push(root);
    const record = {
      path: "assets/effectors/catalog/hand-4f-16dof/right_hand.xml",
      sha256: "1".repeat(64),
      upstream_url: "https://example.org/models",
      upstream_commit: "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35",
      upstream_path: "hand/right_hand.xml",
      upstream_sha256: "2".repeat(64),
      license: "BSD-2-Clause",
      copyright: "Copyright (c) 2016 Example Lab",
      attribution: "Example Lab four-finger hand model, BSD-2-Clause",
      changes: ["prefixed names", "dropped keyframes"],
      status: "amber",
      note: "meshes derive from the maker's CAD",
    };
    const mk = (rel: string, body: unknown) => {
      const dir = path.join(root, rel);
      mkdirSync(dir, { recursive: true });
      writeFileSync(path.join(dir, "SOURCE.json"), JSON.stringify(body));
    };
    mk("catalog/hand-4f-16dof", [record]);
    mk("catalog/gripper-2f-finray", { files: [{ ...record, status: "green", note: "", license: "MIT" }] });
    const credits = effectorCredits(root);
    expect(credits.map((c) => [c.family, c.model])).toEqual([
      ["catalog", "gripper-2f-finray"],
      ["catalog", "hand-4f-16dof"],
    ]);
    expect(credits[1]!.records[0]!.status).toBe("amber");
    expect(credits[0]!.records[0]!.license).toBe("MIT");
    expect(effectorCredits(path.join(root, "missing"))).toEqual([]);
  });
});
