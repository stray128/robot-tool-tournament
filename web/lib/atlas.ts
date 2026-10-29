/**
 * The use-case atlas (web/public/data/atlas.json, written by scripts/build_atlas_data.py): the
 * researched jobs with the setup an experienced integrator would expect, brand-free, plus the
 * integrator rules. Types, the strict parser (a malformed or brand-naming file fails the build)
 * and the pure helpers the page and its explorer share. Node-free.
 *
 * Text rules, the same as the exporter's: URLs only in fields named `url` (shown as link targets
 * with generic text, never as text), and no brand token in any other string.
 */

import { Node } from "./validate";

export const ATLAS_SCHEMA = "rtt-web-atlas";
export const SETUP_KINDS = ["single", "tool_change", "two_tool_head"] as const;
export type AtlasKind = (typeof SETUP_KINDS)[number];
export const FEASIBILITY = ["rigid", "approximation", "out_of_scope"] as const;
export type Feasibility = (typeof FEASIBILITY)[number];

export const KIND_LABELS: Record<AtlasKind, string> = {
  single: "One tool",
  tool_change: "Tool change",
  two_tool_head: "Two-tool head",
};

export const KIND_NOTES: Record<AtlasKind, string> = {
  single: "One tool on the arm does every step.",
  tool_change: "The arm swaps tools between steps on a tool changer.",
  two_tool_head: "Two tools share one head, so the arm never stops to swap.",
};

export const FEASIBILITY_LABELS: Record<Feasibility, string> = {
  rigid: "Rigid models would do",
  approximation: "Needs approximation models",
  out_of_scope: "Out of scope for now",
};

export const FEASIBILITY_NOTES: Record<Feasibility, string> = {
  rigid: "Research estimate: the objects and skills are rigid bodies the simulator can model as they are.",
  approximation: "Research estimate: some part (a seal, a soft object, a process) would need a labelled approximation.",
  out_of_scope: "Research estimate: cloth, liquids, food or other deformables the simulator does not model.",
};

export interface AtlasStep {
  verb: string;
  object: string;
  note: string;
}

export interface AtlasTask {
  id: string;
  domain: string;
  setting: string;
  title: string;
  subtasks: AtlasStep[];
  setup: { kind: AtlasKind; tools: string[] };
  why: string;
  sim: { feasibility: Feasibility; hero: string | null; match: "same" | "related" | null };
  demo_score: number;
  sources: { url: string }[];
}

export interface AtlasRule {
  id: string;
  title: string;
  text: string;
  sources: { url: string }[];
}

export interface Atlas {
  schema: typeof ATLAS_SCHEMA;
  version: 1;
  generated: string;
  note: string;
  domains: { id: string; title: string }[];
  tasks: AtlasTask[];
  rules: AtlasRule[];
}

/** Names that must not appear: `ci` in any case, `cs` (ordinary words) only as written. */
export interface AtlasTokens {
  ci: readonly string[];
  cs: readonly string[];
}

const URL_TEXT = /https?:\/\/|www\./i;

function alternation(list: readonly string[]): string {
  return [...new Set(list.map((t) => t.trim()).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
}

/** A function returning every brand name in a string (whole words), or null for no tokens. */
export function atlasScreen(tokens: AtlasTokens): (text: string) => string[] {
  const ci = tokens.ci.length ? new RegExp(`(?<![A-Za-z0-9])(${alternation(tokens.ci)})(?![A-Za-z0-9])`, "gi") : null;
  const cs = tokens.cs.length ? new RegExp(`(?<![A-Za-z0-9])(${alternation(tokens.cs)})(?![A-Za-z0-9])`, "g") : null;
  return (text: string) => {
    const out = new Set<string>();
    for (const re of [ci, cs]) {
      if (!re) continue;
      re.lastIndex = 0;
      for (const m of text.matchAll(re)) out.add(m[1]!);
    }
    return [...out];
  };
}

function checkText(root: Node, screen: ((text: string) => string[]) | null): void {
  const walk = (n: Node, key: string): void => {
    const v = n.value;
    if (typeof v === "string") {
      if (key === "url") {
        if (!/^https?:\/\/\S+$/.test(v)) n.fail("expected an http(s) URL");
        return;
      }
      if (URL_TEXT.test(v)) n.fail("a URL outside a `url` field (shown text must not be a link)");
      const hits = screen ? screen(v) : [];
      if (hits.length) n.fail(`names a brand (${hits.join(", ")}); the atlas is brand-free`);
      return;
    }
    if (Array.isArray(v)) n.items().forEach((x) => walk(x, key));
    else if (v && typeof v === "object") for (const [k, x] of n.entries()) walk(x, k);
  };
  walk(root, "");
}

function sources(n: Node): { url: string }[] {
  return n.items().map((s) => ({ url: s.get("url").nonEmpty() }));
}

/** Parse and check atlas.json. `screen` (atlasScreen plus the contract's lists) rejects brand names. */
export function parseAtlas(raw: unknown, file = "atlas.json", screen: ((text: string) => string[]) | null = null): Atlas {
  const root = new Node(file, "$", raw);
  root.get("schema").literal(ATLAS_SCHEMA);
  root.get("version").literal(1);
  checkText(root, screen);
  const domains = root
    .get("domains")
    .items()
    .map((d) => ({ id: d.get("id").nonEmpty(), title: d.get("title").nonEmpty() }));
  const domainIds = new Set(domains.map((d) => d.id));
  const ids = new Set<string>();
  const tasks = root
    .get("tasks")
    .items()
    .map((t): AtlasTask => {
      const id = t.get("id").nonEmpty();
      if (ids.has(id)) t.get("id").fail(`task ${id} repeats`);
      ids.add(id);
      const domain = t.get("domain").nonEmpty();
      if (!domainIds.has(domain)) t.get("domain").fail(`unknown domain ${domain}`);
      const demo = t.get("demo_score").int();
      if (demo < 1 || demo > 5) t.get("demo_score").fail("a demo score is 1 to 5");
      const sim = t.get("sim");
      const hero = sim.get("hero").nullable((h) => h.nonEmpty());
      const match = sim.get("match").nullable((m) => m.oneOf(["same", "related"] as const));
      if ((hero === null) !== (match === null)) sim.fail("a hero and its match come together");
      const subtasks = t
        .get("subtasks")
        .items()
        .map((s) => ({ verb: s.get("verb").nonEmpty(), object: s.get("object").nonEmpty(), note: s.get("note").str() }));
      if (subtasks.length === 0) t.get("subtasks").fail("a task has at least one step");
      const tools = t.get("setup").get("tools").strs();
      if (tools.length === 0) t.get("setup").get("tools").fail("a setup has at least one tool");
      return {
        id,
        domain,
        setting: t.get("setting").nonEmpty(),
        title: t.get("title").nonEmpty(),
        subtasks,
        setup: { kind: t.get("setup").get("kind").oneOf(SETUP_KINDS), tools },
        why: t.get("why").nonEmpty(),
        sim: { feasibility: sim.get("feasibility").oneOf(FEASIBILITY), hero, match },
        demo_score: demo,
        sources: sources(t.get("sources")),
      };
    });
  const rules = root
    .get("rules")
    .items()
    .map((r) => ({ id: r.get("id").nonEmpty(), title: r.get("title").nonEmpty(), text: r.get("text").nonEmpty(), sources: sources(r.get("sources")) }));
  return {
    schema: ATLAS_SCHEMA,
    version: 1,
    generated: root.get("generated").nonEmpty(),
    note: root.get("note").nonEmpty(),
    domains,
    tasks,
    rules,
  };
}

// ---------------------------------------------------------------- helpers

/** "lab_food_service" -> "Lab food service"; settings read as words. */
export function settingLabel(setting: string): string {
  const s = setting.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** "R08_dual_head_for_every_cycle_pairs" -> "R08". */
export function ruleNumber(id: string): string {
  return id.split("_")[0] ?? id;
}

/** Tasks per setup kind, in KIND order. */
export function kindCounts(tasks: readonly AtlasTask[]): Record<AtlasKind, number> {
  const out: Record<AtlasKind, number> = { single: 0, tool_change: 0, two_tool_head: 0 };
  for (const t of tasks) out[t.setup.kind] += 1;
  return out;
}

/** What a task's simulation status is on this site: a linked worked example, or the research estimate. */
export type SimStatus = { kind: "simulated"; hero: string; title: string; match: "same" | "related" } | { kind: "researched"; feasibility: Feasibility };

/** The Studio heroes the site has pages for, by case id (from the Studio data index). */
export type HeroLinks = Record<string, { hero: string; title: string }>;

export function simStatus(task: AtlasTask, heroes: HeroLinks): SimStatus {
  const link = task.sim.hero ? heroes[task.sim.hero] : undefined;
  if (link && task.sim.match) return { kind: "simulated", hero: link.hero, title: link.title, match: task.sim.match };
  return { kind: "researched", feasibility: task.sim.feasibility };
}

export interface AtlasFilter {
  q: string;
  domain: string | null;
  kind: AtlasKind | null;
  /** "simulated" for worked examples only, or a feasibility estimate. */
  status: "simulated" | Feasibility | null;
}

export const NO_FILTER: AtlasFilter = { q: "", domain: null, kind: null, status: null };

/** Tasks matching every set filter; the text query matches title, setting, steps, tools and why. */
export function filterTasks(tasks: readonly AtlasTask[], f: AtlasFilter, heroes: HeroLinks): AtlasTask[] {
  const words = f.q.toLowerCase().split(/\s+/).filter(Boolean);
  return tasks.filter((t) => {
    if (f.domain && t.domain !== f.domain) return false;
    if (f.kind && t.setup.kind !== f.kind) return false;
    if (f.status) {
      const s = simStatus(t, heroes);
      if (f.status === "simulated" ? s.kind !== "simulated" : t.sim.feasibility !== f.status) return false;
    }
    if (words.length) {
      const hay = [t.title, t.setting, t.why, ...t.setup.tools, ...t.subtasks.flatMap((s) => [s.verb, s.object, s.note])].join(" ").toLowerCase();
      if (!words.every((w) => hay.includes(w))) return false;
    }
    return true;
  });
}

export type AtlasSort = "demo" | "title" | "simulated";

export function sortTasks(tasks: readonly AtlasTask[], sort: AtlasSort, heroes: HeroLinks): AtlasTask[] {
  const sim = (t: AtlasTask) => (simStatus(t, heroes).kind === "simulated" ? 0 : 1);
  const list = [...tasks];
  if (sort === "title") return list.sort((a, b) => a.title.localeCompare(b.title));
  if (sort === "simulated") return list.sort((a, b) => sim(a) - sim(b) || b.demo_score - a.demo_score || a.title.localeCompare(b.title));
  return list.sort((a, b) => b.demo_score - a.demo_score || sim(a) - sim(b) || a.title.localeCompare(b.title));
}
