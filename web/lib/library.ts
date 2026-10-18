/**
 * The effector library's web data (product mode, not part of the pre-registered experiment).
 *
 * Producer: `rtt.library.web.export_web` (src/rtt/library/web.py), which writes
 * public/data/library/index.json and the item previews under public/data/library/glb/. Every
 * string that reaches the screen comes from that file, which the producer checks against the
 * library's brand list; makers are named only on the credits page, which reads each item's
 * SOURCE.json. This module holds the types, the parser (a malformed index fails `pnpm build`),
 * the labels and the catalog filter; it imports nothing from Node, so client components can use
 * it. The build-time loader lives in app/library/data.ts.
 */

import { FIDELITIES, type Fidelity, PRODUCT_MODE_LABEL } from "./effectors";
import { DataError, Node } from "./validate";

export const LIBRARY_FORMAT = "rtt-web-library";
/** Shown above every library page, word for word (rtt.library.web.LABEL). */
export const LIBRARY_LABEL = PRODUCT_MODE_LABEL;

export const SIM_STATUSES = ["native", "proxy", "visual_only", "none"] as const;
export type SimStatus = (typeof SIM_STATUSES)[number];
export const SPEC_TAGS = ["sourced", "derived", "assumed"] as const;
export type SpecTag = (typeof SPEC_TAGS)[number];
export const EXCLUDED_RATINGS = ["amber", "red"] as const;
export type ExcludedRating = (typeof EXCLUDED_RATINGS)[number];
export const MOUNT_STATUSES = ["sourced", "assumed", "unknown"] as const;
export type MountStatus = (typeof MOUNT_STATUSES)[number];
export const DIM_STATUSES = ["sourced", "secondary", "assumed"] as const;
export type DimStatus = (typeof DIM_STATUSES)[number];

/** Short sim-status words for badges and filters (the index carries the long titles). */
export const SIM_STATUS_LABELS: Record<SimStatus, string> = {
  native: "Native",
  proxy: "Proxy",
  visual_only: "Visual only",
  none: "Link only",
};

export const SIM_STATUS_NOTES: Record<SimStatus, string> = {
  native: "An upstream simulation model, or one converted to MJCF by us.",
  proxy: "No upstream model: a generated rigid body from the item's own mesh, or a mapping onto an existing family.",
  visual_only: "Files and a 3D preview, no physics yet.",
  none: "Listed with a link only.",
};

export const SPEC_TAG_LABELS: Record<SpecTag, string> = {
  sourced: "Sourced",
  derived: "Derived",
  assumed: "Assumed",
};

export const SPEC_TAG_NOTES: Record<SpecTag, string> = {
  sourced: "Read from a published source, linked.",
  derived: "Worked out from the files or the model, not stated by the source.",
  assumed: "Our engineering assumption: check it before you rely on it.",
};

export const MOUNT_STATUS_LABELS: Record<MountStatus, string> = {
  sourced: "sourced",
  assumed: "assumed, not read today",
  unknown: "unknown",
};

export const DIM_STATUS_LABELS: Record<DimStatus, string> = {
  sourced: "sourced",
  secondary: "one secondary source, unverified",
  assumed: "assumed",
};

export const RATING_LABELS: Record<ExcludedRating, string> = {
  amber: "Amber",
  red: "Red",
};

// ---------------------------------------------------------------- types

export interface LibraryLicence {
  spdx: string;
  covers: string;
  url: string | null;
}

export interface LibraryMount {
  interface: string;
  iso_pattern: string | null;
  status: MountStatus;
  adapter: string | null;
  source: string;
  notes: string[];
}

export interface LibrarySpec {
  name: string;
  value: string;
  unit: string;
  tag: SpecTag;
  source: string;
}

export interface LibrarySim {
  status: SimStatus;
  title: string;
  source: string;
  fidelity: Fidelity | null;
  note: string;
  reason: string;
  family: string | null;
  n_actuators: number | null;
  mass_kg: number | null;
  entrant: boolean;
  entrant_reason: string | null;
}

export interface NominalEpisode {
  held: boolean;
  engaged: boolean;
  outcome: string | null;
  penetration_mm: number | null;
}

export interface GateEntrant {
  display_name: string;
  primitive: string;
  family: string;
  passed: boolean;
  built: string[];
  /** Built only on the 10 mm probe block, because every training part was refused for size. */
  probe_only: boolean;
  refused: Record<string, string>;
  error: string | null;
  n_actuators: number | null;
  mass_kg: number | null;
  reach_mm: number | null;
  opening_mm: number | null;
  nominal: Record<string, NominalEpisode>;
  nominal_held: string[];
}

export interface LibraryGate {
  ran: boolean;
  passed: boolean | null;
  entrants: GateEntrant[];
}

export interface LibraryDownload {
  path: string;
  name: string;
  role: string;
  role_title: string;
  bytes: number;
  license: string;
  sha256: string;
  modified: boolean;
}

export interface LibraryLink {
  label: string;
  reason: string;
  url: string;
  bytes: number;
}

export interface UpstreamPin {
  repository: string;
  commit: string;
  url: string;
}

export type LibraryUpstream = { generated: true } | (UpstreamPin & { generated: false; extra: UpstreamPin[] });

export interface LibraryItem {
  key: string;
  display_name: string;
  summary: string;
  kind: string;
  kind_title: string;
  class: string;
  class_title: string;
  capabilities: string[];
  licences: LibraryLicence[];
  rating: string;
  mount: LibraryMount;
  specs: LibrarySpec[];
  sim: LibrarySim;
  gate: LibraryGate;
  preview: { url: string; bytes: number } | null;
  downloads: LibraryDownload[];
  links: LibraryLink[];
  upstream: LibraryUpstream;
  notes: string[];
  credit_anchor: string;
}

export interface Dimension {
  value: number;
  status: DimStatus;
  source: string;
}

export interface IsoPattern {
  designation: string;
  d1_mm: Dimension;
  n_holes: number;
  thread: string;
  d2_mm: Dimension;
  dowel_mm: Dimension;
  seen_on: string;
  unverified: string[];
  notes: string[];
  confirmed: boolean;
}

export interface ListedItem {
  name: string;
  kind: string;
  capabilities: string[];
  license: string;
  url: string;
  reason: string;
}

export interface ExcludedItem {
  name: string;
  rating: ExcludedRating;
  license: string;
  url: string | null;
  reason: string;
}

export interface VendoredItem {
  name: string;
  where: string;
  key: string;
}

export interface LibraryCounts {
  items: number;
  entrants: number;
  entrant_items: number;
  by_sim_status: Record<SimStatus, number>;
  listed: number;
  excluded_amber: number;
  excluded_red: number;
}

export interface LibraryIndex {
  format: typeof LIBRARY_FORMAT;
  version: 1;
  label: typeof LIBRARY_LABEL;
  audit: { date: string; note: string };
  counts: LibraryCounts;
  capabilities: { tag: string; note: string }[];
  green_licences: string[];
  iso_patterns: IsoPattern[];
  items: LibraryItem[];
  listed: ListedItem[];
  excluded: ExcludedItem[];
  already_vendored: VendoredItem[];
  provenance: { manifest_sha256: string; lock: string };
}

// ---------------------------------------------------------------- parser

const KEY = /^[a-z0-9][a-z0-9-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;
const URL_RE = /^https?:\/\/\S+$/;

function key(n: Node): string {
  const s = n.nonEmpty();
  if (!KEY.test(s)) n.fail(`expected a URL-safe key (lowercase letters, digits and hyphens), got ${JSON.stringify(s)}`);
  return s;
}

function url(n: Node): string {
  const s = n.nonEmpty();
  if (!URL_RE.test(s)) n.fail(`expected an http(s) URL, got ${JSON.stringify(s.slice(0, 60))}`);
  return s;
}

function optUrl(n: Node): string | null {
  return n.nullable(url);
}

function optNum(n: Node): number | null {
  return n.nullable((m) => m.num());
}

function optStr(n: Node): string | null {
  return n.nullable((m) => m.str());
}

function strOr(n: Node, parent: Node, name: string): string {
  return parent.has(name) ? n.str() : "";
}

/** A repository-relative path: no scheme, no leading slash, no "..". */
function repoPath(n: Node, prefix: string): string {
  const s = n.nonEmpty();
  if (s.startsWith("/") || s.split("/").includes("..") || /^[a-z]+:/i.test(s)) n.fail(`expected a repository path, got ${JSON.stringify(s)}`);
  if (!s.startsWith(prefix)) n.fail(`expected a path under ${prefix}, got ${JSON.stringify(s)}`);
  return s;
}

function readLicence(n: Node): LibraryLicence {
  return { spdx: n.get("spdx").nonEmpty(), covers: n.get("covers").str(), url: optUrl(n.get("url")) };
}

function readMount(n: Node): LibraryMount {
  return {
    interface: n.get("interface").nonEmpty(),
    iso_pattern: optStr(n.get("iso_pattern")),
    status: n.get("status").oneOf(MOUNT_STATUSES),
    adapter: n.has("adapter") ? n.get("adapter").nullable(key) : null,
    source: strOr(n.get("source"), n, "source"),
    notes: n.has("notes") ? n.get("notes").strs() : [],
  };
}

function readSpec(n: Node): LibrarySpec {
  const spec = {
    name: n.get("name").nonEmpty(),
    value: n.get("value").str(),
    unit: n.get("unit").str(),
    tag: n.get("tag").oneOf(SPEC_TAGS),
    source: n.get("source").str(),
  };
  if (spec.tag === "sourced" && !URL_RE.test(spec.source)) n.get("source").fail("a sourced value needs its source URL");
  return spec;
}

function readSim(n: Node): LibrarySim {
  const status = n.get("status").oneOf(SIM_STATUSES);
  const entrant = n.get("entrant").bool();
  const reason = optStr(n.get("entrant_reason"));
  if (entrant && reason !== null) n.get("entrant_reason").fail("an entrant has no reason for not being one");
  if (!entrant && !reason) n.get("entrant_reason").fail("an item that is not an entrant needs the reason");
  return {
    status,
    title: n.get("title").nonEmpty(),
    source: n.get("source").str(),
    fidelity: n.get("fidelity").nullable((f) => f.oneOf(FIDELITIES)),
    note: n.get("note").str(),
    reason: n.get("reason").str(),
    family: optStr(n.get("family")),
    n_actuators: n.get("n_actuators").nullable((m) => m.int()),
    mass_kg: optNum(n.get("mass_kg")),
    entrant,
    entrant_reason: reason,
  };
}

function readEntrant(n: Node): GateEntrant {
  const nominal: Record<string, NominalEpisode> = {};
  for (const [part, ep] of n.get("nominal").entries()) {
    nominal[part] = {
      held: ep.get("held").bool(),
      engaged: ep.get("engaged").bool(),
      outcome: optStr(ep.get("outcome")),
      penetration_mm: optNum(ep.get("penetration_mm")),
    };
  }
  const refused: Record<string, string> = {};
  for (const [part, why] of n.get("refused").entries()) refused[part] = why.str();
  const built = n.get("built").strs();
  const passed = n.get("passed").bool();
  const error = optStr(n.get("error"));
  const probeOnly = n.has("probe_only") ? n.get("probe_only").bool() : false;
  if (probeOnly && built.length > 0) n.get("probe_only").fail("a probe-only build builds on no training part");
  if (passed && ((built.length === 0 && !probeOnly) || error !== null)) {
    n.get("passed").fail("a passed gate builds on a training part (or the probe block), without an error");
  }
  const held = n.get("nominal_held").strs();
  for (const p of held) if (!nominal[p]?.held) n.get("nominal_held").fail(`part ${p} is listed as held but its nominal episode was not`);
  return {
    display_name: n.get("display_name").nonEmpty(),
    primitive: n.get("primitive").nonEmpty(),
    family: n.get("family").nonEmpty(),
    passed,
    built,
    probe_only: probeOnly,
    refused,
    error,
    n_actuators: n.get("n_actuators").nullable((m) => m.int()),
    mass_kg: optNum(n.get("mass_kg")),
    reach_mm: optNum(n.get("reach_mm")),
    opening_mm: optNum(n.get("opening_mm")),
    nominal,
    nominal_held: held,
  };
}

function readGate(n: Node): LibraryGate {
  const ran = n.get("ran").bool();
  if (!ran) return { ran, passed: null, entrants: [] };
  const entrants = n.get("entrants").items().map(readEntrant);
  const passed = n.get("passed").bool();
  if (passed !== entrants.every((e) => e.passed)) n.get("passed").fail("the gate passes only when every entrant spec passes");
  return { ran, passed, entrants };
}

function readDownload(n: Node, itemKey: string): LibraryDownload {
  const path = repoPath(n.get("path"), `assets/effectors/library/${itemKey}/`);
  const name = n.get("name").nonEmpty();
  if (path.split("/").at(-1) !== name) n.get("name").fail(`expected the file name of ${path}`);
  const sha = n.get("sha256").nonEmpty();
  if (!SHA256.test(sha)) n.get("sha256").fail("expected a sha256 hex digest");
  return {
    path,
    name,
    role: n.get("role").nonEmpty(),
    role_title: n.get("role_title").nonEmpty(),
    bytes: n.get("bytes").int(),
    license: n.get("license").nonEmpty(),
    sha256: sha,
    modified: n.get("modified").bool(),
  };
}

function readPin(n: Node): UpstreamPin {
  return { repository: url(n.get("repository")), commit: n.get("commit").nonEmpty(), url: url(n.get("url")) };
}

function readUpstream(n: Node): LibraryUpstream {
  if (n.has("generated")) {
    if (!n.get("generated").bool()) n.get("generated").fail("expected true, or a pinned repository instead");
    return { generated: true };
  }
  return { generated: false, ...readPin(n), extra: n.has("extra") ? n.get("extra").items().map(readPin) : [] };
}

function readItem(n: Node): LibraryItem {
  const k = key(n.get("key"));
  const preview = n.get("preview").nullable((p) => {
    const rel = p.get("url").nonEmpty();
    if (!/^data\/library\/glb\/[a-z0-9][a-z0-9._-]*\.glb$/.test(rel)) p.get("url").fail(`expected data/library/glb/<file>.glb, got ${JSON.stringify(rel)}`);
    return { url: rel, bytes: p.get("bytes").int() };
  });
  const anchor = n.get("credit_anchor").nonEmpty();
  if (anchor !== `library-${k}`) n.get("credit_anchor").fail(`expected library-${k}`);
  const licences = n.get("licences").items().map(readLicence);
  if (licences.length === 0) n.get("licences").fail("an item needs at least one licence");
  const sim = readSim(n.get("sim"));
  const gate = readGate(n.get("gate"));
  if (!sim.entrant && gate.entrants.length > 0) n.get("gate").fail("an item that is not an entrant has no gate entrants");
  return {
    key: k,
    display_name: n.get("display_name").nonEmpty(),
    summary: n.get("summary").nonEmpty(),
    kind: n.get("kind").nonEmpty(),
    kind_title: n.get("kind_title").nonEmpty(),
    class: n.get("class").nonEmpty(),
    class_title: n.get("class_title").nonEmpty(),
    capabilities: n.get("capabilities").strs(),
    licences,
    rating: n.get("rating").nonEmpty(),
    mount: readMount(n.get("mount")),
    specs: n.get("specs").items().map(readSpec),
    sim,
    gate,
    preview,
    downloads: n.get("downloads").items().map((d) => readDownload(d, k)),
    links: n
      .get("links")
      .items()
      .map((l) => ({ label: l.get("label").nonEmpty(), reason: l.get("reason").str(), url: url(l.get("url")), bytes: l.get("bytes").int() })),
    upstream: readUpstream(n.get("upstream")),
    notes: n.get("notes").strs(),
    credit_anchor: anchor,
  };
}

function readDimension(n: Node): Dimension {
  return { value: n.get("value").num(), status: n.get("status").oneOf(DIM_STATUSES), source: n.get("source").str() };
}

function readIsoPattern(n: Node): IsoPattern {
  return {
    designation: n.get("designation").nonEmpty(),
    d1_mm: readDimension(n.get("d1_mm")),
    n_holes: n.get("n_holes").int(),
    thread: n.get("thread").nonEmpty(),
    d2_mm: readDimension(n.get("d2_mm")),
    dowel_mm: readDimension(n.get("dowel_mm")),
    seen_on: n.get("seen_on").str(),
    unverified: n.get("unverified").strs(),
    notes: n.get("notes").strs(),
    confirmed: n.get("confirmed").bool(),
  };
}

function readCounts(n: Node): LibraryCounts {
  const by = n.get("by_sim_status");
  const by_sim_status = Object.fromEntries(SIM_STATUSES.map((s) => [s, by.has(s) ? by.get(s).int() : 0])) as Record<SimStatus, number>;
  return {
    items: n.get("items").int(),
    entrants: n.get("entrants").int(),
    entrant_items: n.get("entrant_items").int(),
    by_sim_status,
    listed: n.get("listed").int(),
    excluded_amber: n.get("excluded_amber").int(),
    excluded_red: n.get("excluded_red").int(),
  };
}

/** Parse and cross-check public/data/library/index.json (throws DataError on any problem). */
export function parseLibraryIndex(raw: unknown, file = "library/index.json"): LibraryIndex {
  const root = new Node(file, "$", raw);
  root.get("format").literal(LIBRARY_FORMAT);
  root.get("version").literal(1);
  root.get("label").literal(LIBRARY_LABEL);
  const capabilities = root
    .get("capabilities")
    .items()
    .map((c) => ({ tag: c.get("tag").nonEmpty(), note: c.get("note").nonEmpty() }));
  const tags = new Set(capabilities.map((c) => c.tag));
  const isoPatterns = root.get("iso_patterns").items().map(readIsoPattern);
  const designations = new Set(isoPatterns.map((p) => p.designation));
  const itemsNode = root.get("items");
  const items = itemsNode.items().map(readItem);
  const keys = new Set<string>();
  items.forEach((item, i) => {
    const where = `$.items[${i}]`;
    if (keys.has(item.key)) throw new DataError(file, where, `duplicate item key ${item.key}`);
    keys.add(item.key);
    for (const c of item.capabilities) if (!tags.has(c)) throw new DataError(file, `${where}.capabilities`, `unknown capability ${c}`);
    if (item.mount.iso_pattern && !designations.has(item.mount.iso_pattern)) {
      throw new DataError(file, `${where}.mount.iso_pattern`, `unknown ISO pattern ${item.mount.iso_pattern}`);
    }
  });
  items.forEach((item, i) => {
    if (item.mount.adapter && !keys.has(item.mount.adapter)) {
      throw new DataError(file, `$.items[${i}].mount.adapter`, `adapter ${item.mount.adapter} is not a library item`);
    }
  });
  const listed = root
    .get("listed")
    .items()
    .map((l) => ({
      name: l.get("name").nonEmpty(),
      kind: l.get("kind").str(),
      capabilities: l.get("capabilities").strs(),
      license: l.get("license").nonEmpty(),
      url: url(l.get("url")),
      reason: l.get("reason").nonEmpty(),
    }));
  const excluded = root
    .get("excluded")
    .items()
    .map((e) => ({
      name: e.get("name").nonEmpty(),
      rating: e.get("rating").oneOf(EXCLUDED_RATINGS),
      license: e.get("license").nonEmpty(),
      url: optUrl(e.get("url")),
      reason: e.get("reason").nonEmpty(),
    }));
  const counts = readCounts(root.get("counts"));
  const expect = (name: string, stated: number, actual: number) => {
    if (stated !== actual) throw new DataError(file, `$.counts.${name}`, `states ${stated}, the lists give ${actual}`);
  };
  expect("items", counts.items, items.length);
  expect("entrants", counts.entrants, items.reduce((n, i) => n + i.gate.entrants.length, 0));
  expect("entrant_items", counts.entrant_items, items.filter((i) => i.sim.entrant).length);
  for (const s of SIM_STATUSES) expect(`by_sim_status.${s}`, counts.by_sim_status[s], items.filter((i) => i.sim.status === s).length);
  expect("listed", counts.listed, listed.length);
  expect("excluded_amber", counts.excluded_amber, excluded.filter((e) => e.rating === "amber").length);
  expect("excluded_red", counts.excluded_red, excluded.filter((e) => e.rating === "red").length);
  const audit = root.get("audit");
  const provenance = root.get("provenance");
  return {
    format: LIBRARY_FORMAT,
    version: 1,
    label: LIBRARY_LABEL,
    audit: { date: audit.get("date").nonEmpty(), note: audit.get("note").str() },
    counts,
    capabilities,
    green_licences: root.get("green_licences").strs(),
    iso_patterns: isoPatterns,
    items,
    listed,
    excluded,
    already_vendored: root
      .get("already_vendored")
      .items()
      .map((v) => ({ name: v.get("name").nonEmpty(), where: v.get("where").str(), key: v.get("key").nonEmpty() })),
    provenance: { manifest_sha256: provenance.get("manifest_sha256").nonEmpty(), lock: provenance.get("lock").nonEmpty() },
  };
}

// ---------------------------------------------------------------- catalog cards and filters

/** What a catalog card shows, and what the filters match against (no URLs). */
export interface LibraryCard {
  key: string;
  display_name: string;
  summary: string;
  kind: string;
  kind_title: string;
  capabilities: string[];
  licences: string[];
  sim_status: SimStatus;
  sim_title: string;
  iso_pattern: string | null;
  entrant: boolean;
}

export function toCard(item: LibraryItem): LibraryCard {
  return {
    key: item.key,
    display_name: item.display_name,
    summary: item.summary,
    kind: item.kind,
    kind_title: item.kind_title,
    capabilities: item.capabilities,
    licences: [...new Set(item.licences.map((l) => l.spdx))],
    sim_status: item.sim.status,
    sim_title: item.sim.title,
    iso_pattern: item.mount.iso_pattern,
    entrant: item.sim.entrant,
  };
}

export type MountFilter = "any" | "iso" | "own";

export interface LibraryFilters {
  query: string;
  /** Capability tags; a card matches when it has any of them (none selected: every card). */
  capabilities: string[];
  /** One SPDX id, or "" for any licence. */
  licence: string;
  mount: MountFilter;
  sim: SimStatus | "any";
  entrantOnly: boolean;
}

export const NO_FILTERS: LibraryFilters = { query: "", capabilities: [], licence: "", mount: "any", sim: "any", entrantOnly: false };

export function isFiltered(f: LibraryFilters): boolean {
  return (
    f.query.trim() !== "" || f.capabilities.length > 0 || f.licence !== "" || f.mount !== "any" || f.sim !== "any" || f.entrantOnly
  );
}

/** Cards matching every filter; the text search needs every word in the name, summary or kind. */
export function filterCards(cards: readonly LibraryCard[], f: LibraryFilters): LibraryCard[] {
  const words = f.query.toLowerCase().split(/\s+/).filter(Boolean);
  return cards.filter((c) => {
    if (words.length > 0) {
      const text = `${c.display_name} ${c.summary} ${c.kind_title} ${c.kind.replace(/_/g, " ")}`.toLowerCase();
      if (!words.every((w) => text.includes(w))) return false;
    }
    if (f.capabilities.length > 0 && !f.capabilities.some((t) => c.capabilities.includes(t))) return false;
    if (f.licence && !c.licences.includes(f.licence)) return false;
    if (f.mount === "iso" && !c.iso_pattern) return false;
    if (f.mount === "own" && c.iso_pattern) return false;
    if (f.sim !== "any" && c.sim_status !== f.sim) return false;
    if (f.entrantOnly && !c.entrant) return false;
    return true;
  });
}

/** Filters as a URL query (only the ones set), e.g. "?q=hand&cap=pinch,power". */
export function filtersToQuery(f: LibraryFilters): string {
  const q = new URLSearchParams();
  if (f.query.trim()) q.set("q", f.query.trim());
  if (f.capabilities.length) q.set("cap", f.capabilities.join(","));
  if (f.licence) q.set("licence", f.licence);
  if (f.mount !== "any") q.set("mount", f.mount);
  if (f.sim !== "any") q.set("sim", f.sim);
  if (f.entrantOnly) q.set("entrant", "1");
  const s = q.toString();
  return s ? `?${s}` : "";
}

/** The inverse of filtersToQuery; unknown values fall back to no filter. */
export function filtersFromQuery(search: string, known: { capabilities: readonly string[]; licences: readonly string[] }): LibraryFilters {
  const q = new URLSearchParams(search);
  const mount = q.get("mount");
  const sim = q.get("sim");
  return {
    query: q.get("q") ?? "",
    capabilities: (q.get("cap") ?? "").split(",").filter((t) => known.capabilities.includes(t)),
    licence: known.licences.includes(q.get("licence") ?? "") ? (q.get("licence") as string) : "",
    mount: mount === "iso" || mount === "own" ? mount : "any",
    sim: (SIM_STATUSES as readonly string[]).includes(sim ?? "") ? (sim as SimStatus) : "any",
    entrantOnly: q.get("entrant") === "1",
  };
}

// ---------------------------------------------------------------- formatting

/** An SPDX id for the screen: NONE, NOASSERTION and LicenseRef ids in words. */
export function licenceText(spdx: string): string {
  if (spdx === "NONE") return "none (no licence)";
  if (spdx === "NOASSERTION") return "not stated";
  const ref = spdx.match(/^LicenseRef-(.+)$/);
  return ref ? `custom terms (${ref[1]!.replace(/-/g, " ")})` : spdx;
}

/** The mount in a few words: the ISO pattern, or "own interface". */
export function mountText(isoPattern: string | null): string {
  return isoPattern ?? "own interface";
}

/** A training part id in words, when no title is known ("steel_slug" -> "Steel slug"). */
export function partTitleFallback(id: string): string {
  const s = id.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Download link of a repository file, or null while the repository URL is unset. */
export function repoFileUrl(repo: string | null, path: string): string | null {
  if (!repo) return null;
  return `${repo.replace(/\/+$/, "")}/raw/main/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/** The first 8 characters of a commit (or the whole string when shorter). */
export function shortCommit(commit: string): string {
  return /^[0-9a-f]{9,}$/.test(commit) ? commit.slice(0, 8) : commit;
}

/** Capitalise the first letter of a sentence from the data. */
export function sentence(text: string): string {
  const t = text.trim();
  if (!t) return t;
  const s = t.charAt(0).toUpperCase() + t.slice(1);
  return /[.!?]$/.test(s) ? s : `${s}.`;
}
