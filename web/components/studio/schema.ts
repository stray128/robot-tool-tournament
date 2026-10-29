/**
 * The Studio's web data (product mode, not part of the pre-registered experiment).
 *
 * Producer: `scripts/export_studio_web.py`, which writes public/data/studio/index.json,
 * studio/<hero>.json, the replay v2 lanes under studio/replays/<hero>/, the build pack zips under
 * studio/<hero>/ and the recorded sessions under studio/sessions/. This module holds the types,
 * the strict parsers (a malformed file fails `pnpm build`), the labels and the pure helpers the
 * pages and the replay lanes share. It imports nothing from Node, so client components can use
 * it; the build-time loaders that check the files on disk live in app/studio/data.ts.
 *
 * Rules every document follows: it carries the product-mode label word for word; URLs appear only
 * in fields named `url` (rendered as link targets, never as text); and, when a brand pattern is
 * given, no other string names a brand.
 */

import { PRODUCT_MODE_LABEL } from "@/lib/effectors";
import { STEP_FAILURE_LABELS, STEP_FAILURE_MODES } from "@/lib/labels";
import type { ReplayHeaderV2 } from "@/lib/types";
import { Node } from "@/lib/validate";

export { PRODUCT_MODE_LABEL };
export const STUDIO_INDEX_SCHEMA = "rtt-web-studio";
export const STUDIO_HERO_SCHEMA = "rtt-web-studio-hero";
/** The largest build pack the site serves (scripts/export_studio_web.py PACK_CAP_BYTES). */
export const PACK_CAP_BYTES = 2_000_000;
/** The caveat every approximation-tier recommendation carries, word for word. */
export const VERIFY_CAVEAT = "approximation: verify on hardware";

export const TIERS = ["rigid", "approximation", "not_simulated"] as const;
export type Tier = (typeof TIERS)[number];
export const SETUP_KINDS = ["single", "multi_head", "tool_change"] as const;
export type SetupKind = (typeof SETUP_KINDS)[number];
export const OUTCOMES = ["recommended", "no_recommendation"] as const;
export type Outcome = (typeof OUTCOMES)[number];
export const ROLES = ["recommended", "best_single", "baseline", "closest", "runner_up_rigid", "finalist"] as const;
export type Role = (typeof ROLES)[number];
export const CHANGES = ["none", "tool_change", "present"] as const;
export type Change = (typeof CHANGES)[number];
export const SPEC_TAGS = ["sourced", "derived", "assumed"] as const;
export type SpecTag = (typeof SPEC_TAGS)[number];

export const TIER_LABELS: Record<Tier, string> = {
  rigid: "Rigid",
  approximation: "Approximation",
  not_simulated: "Not simulated yet",
};

export const TIER_NOTES: Record<Tier, string> = {
  rigid: "A rigid multibody model with a scripted controller.",
  approximation: "A labelled approximation (suction seal, soft arm, latches, threads, valves): verify on hardware.",
  not_simulated: "No physics model yet: never scored as a pass.",
};

/**
 * Step failure modes (rtt.studio.types.STEP_FAILURE_MODES) in plain words (lib/labels.ts). The
 * parsers refuse any other mode, so a code never reaches the screen.
 */
export const FAILURE_LABELS: Record<string, string> = STEP_FAILURE_LABELS;
export { STEP_FAILURE_MODES };

export const ROLE_LABELS: Record<Role, string> = {
  recommended: "Recommended",
  best_single: "Best single off-the-shelf",
  baseline: "Best single tool (baseline)",
  closest: "Closest",
  runner_up_rigid: "Rigid runner-up",
  finalist: "Finalist",
};

export const PLAN_RESULT_LABELS: Record<string, string> = {
  valid_first_call: "valid on the first call",
  valid_after_repair: "valid after one repair call",
  valid_after_layout_fix: "valid after the layout fix",
};

// ---------------------------------------------------------------- types

export type Scalar = number | string | boolean;

/** One tool of a setup. The index lists tool objects for a recommendation and names elsewhere. */
export interface ToolBrief {
  index: number;
  display_name: string;
  effector_id: string | null;
  family: string | null;
  family_title: string | null;
  fidelity: Tier | null;
  off_the_shelf: boolean | null;
  caveat: string | null;
  /** A Studio-only tool (a process tool or a vacuum area frame): never in a per-part tournament. */
  studio_only: boolean;
  /** Its rated load, in words ("rated 6.7 N m ..."), for a Studio-only tool; else null. */
  rating: string | null;
}

export interface SetupBrief {
  setup_id: string;
  kind: SetupKind;
  kind_title: string;
  label: string;
  tools: ToolBrief[];
  completion_rate: number;
  completed_runs: number;
  n_runs: number;
  fidelity: Tier;
  stops_at: string | null;
  cycle_time_s: number | null;
}

export interface HeroEntry {
  hero: string;
  case_id: string;
  title: string;
  setting: string;
  use_case: string;
  page: string;
  task_id: string;
  n_steps: number;
  n_candidates: number;
  outcome: Outcome;
  headline: string;
  recommended: SetupBrief | null;
  best_single: SetupBrief | null;
  closest: SetupBrief | null;
}

export interface StudioIndex {
  schema: typeof STUDIO_INDEX_SCHEMA;
  version: 1;
  label: typeof PRODUCT_MODE_LABEL;
  contract: string;
  code_version: string;
  heroes: HeroEntry[];
}

export interface PlanObject {
  name: string;
  kind: string;
  kind_title: string;
  label: string;
  params: Record<string, Scalar>;
  pose: { x_mm: number; y_mm: number; yaw_deg: number };
  ref: string | null;
}

export interface Criterion {
  quantity: string;
  unit: string;
  at_least: number | null;
  at_most: number | null;
  text: string;
}

export interface PlanStep {
  id: string;
  primitive: string;
  primitive_title: string;
  object: string;
  handle: string | null;
  target: string | null;
  params: Record<string, Scalar>;
  constraints: { max_force_n: number | null; max_torque_nm: number | null; precision_mm: number | null; precision_deg: number | null };
  success: Criterion[];
  after: string[];
  hold_after: boolean;
  description: string;
  fidelity: Tier;
  caveats: string[];
}

export interface Plan {
  task_id: string;
  title: string;
  description: string;
  source: string;
  result: string;
  calls: number | null;
  model: string | null;
  layout_moves: string[];
  assumptions: string[];
  clarifications: { question: string; assumed: string }[];
  unsupported: { what: string; why: string; nearest: string | null }[];
  objects: PlanObject[];
  steps: PlanStep[];
}

export interface Candidate {
  effector_id: string;
  display_name: string;
  family: string;
  family_title: string;
  source: string;
  off_the_shelf: boolean;
  mechanism: string;
  fidelity: Tier;
  built_for: string | null;
  status: "screened" | "pruned";
  pruned_reason: string | null;
  pruned_steps: { step_id: string; reason: string }[];
  /** A Studio-only tool (a process tool or a vacuum area frame), with its rating and caveat. */
  studio_only: boolean;
  rating: string | null;
  caveat: string | null;
}

export interface Screen {
  n_candidates: number;
  n_screened: number;
  n_conditions: number;
  runs: number;
  wall_s: number;
  candidates: Candidate[];
  pruned_reasons: { reason: string; count: number }[];
}

export type MatrixCell =
  | {
      state: "run";
      rate: number;
      n: number;
      successes: number;
      primitive_used: string | null;
      failure_modes: Record<string, number>;
      context: string | null;
    }
  | { state: "pruned"; reason: string }
  | { state: "not_simulated"; reason: string | null };

export interface MatrixRow {
  effector_id: string;
  display_name: string;
  family_title: string;
  fidelity: Tier;
  off_the_shelf: boolean;
  cells: Record<string, MatrixCell>;
}

export interface Matrix {
  steps: string[];
  rows: MatrixRow[];
}

export interface FinalistRow extends SetupBrief {
  roles: Role[];
  eligible: boolean;
  reason: string | null;
  tool_details: ToolBrief[];
  step_success: Record<string, number>;
  weakest_step: { step_id: string; success_rate: number } | null;
  complexity: { n_tools: number; n_actuators: number; total_mass_kg: number; n_changes: number; bracket: boolean };
  payload_kg: number | null;
  failure_modes: Record<string, number>;
  stops_at_detail: Record<string, { runs: number; failure_modes: Record<string, number> }>;
}

export interface RecStep {
  step_id: string;
  primitive: string;
  primitive_used: string | null;
  tool: number;
  tool_label: string;
  success_rate: number;
  fidelity: Tier;
  caveats: string[];
  verify_on_hardware: boolean;
}

export type Geometry =
  | { kind: "head"; mounts: { angle_deg: number; standoff_mm: number; lateral_mm: number; spin_deg: number }[]; clearance_mm: number | null; present_s: number | null }
  | { kind: "changer"; stack_mm: number | null; changeover_s: number | null };

export interface Recommendation extends SetupBrief {
  why: string;
  caveat: string | null;
  verify_on_hardware: boolean;
  geometry: Geometry | null;
  steps: RecStep[];
  runner_up_rigid: { label: string | null; basis: string } | null;
  versus_best_single: { same_setup: boolean; completion_delta: number; cycle_delta_s: number | null; extra_tools: number } | null;
  /** The pick against the baseline, the best eligible single tool (rtt.studio.recommend, rule R01); null in older pages. */
  versus_baseline: VersusBaseline | null;
}

/**
 * What decided the pick against the baseline (rtt.studio.recommend._versus_baseline): it is the
 * baseline; a same-complexity tie; the baseline has a soft arm (R09) or a hand (R10), which rank
 * below rigid setups; the baseline fails a step outright; the pick clears the margin its added
 * pieces must earn; or the ranking.
 */
export const BASELINE_BASES = ["baseline", "tie", "R09", "R10", "step_floor", "margin", "ranking"] as const;
export type BaselineBasis = (typeof BASELINE_BASES)[number];

export interface VersusBaseline {
  same_setup: boolean;
  completion_delta: number;
  cycle_delta_s: number | null;
  added_pieces: number;
  /** The completion gain the added pieces had to earn; null when a rule ranked the baseline lower. */
  needed_gain: number | null;
  basis: BaselineBasis;
  note: string | null;
}

/**
 * The baseline printed beside the pick: the best eligible single tool (its finalist row), or,
 * when no single tool is eligible, the recommender's reason (kept for the record; the page words
 * it from the single tools' own rows).
 */
export interface Baseline {
  row: FinalistRow | null;
  basis: string | null;
}

export interface BlockerFamily {
  family: string;
  family_title: string;
  n_candidates: number;
  n_runs: number;
  peak: number | null;
  best: string | null;
  failure_modes: Record<string, number>;
}

export interface Blocker {
  step_id: string;
  primitive: string;
  quantity: string;
  unit: string;
  at_least: number | null;
  at_most: number | null;
  text: string;
  conditions: string[];
  families: BlockerFamily[];
  max_peak: number | null;
}

export interface NoWinner {
  reason: string;
  closest: FinalistRow[];
  blocker: Blocker | null;
}

export interface LaneSegment {
  index: number;
  tool: number;
  tool_label: string;
  start_s: number;
  end_s: number;
  change: Change;
  change_s: number;
  /** When the change before this segment starts (the replay holds the last pose until start_s). */
  change_start_s: number;
  steps: string[];
}

export interface LaneStep {
  step_id: string;
  primitive: string;
  tool: number;
  start_s: number;
  duration_s: number;
  success: boolean;
  failure_mode: string | null;
  fidelity: Tier;
  quantities: Record<string, number | null>;
}

export interface StudioLane {
  id: string;
  title: string;
  setup_id: string;
  kind: SetupKind;
  label: string;
  replay: string;
  condition_id: string;
  seed: number;
  condition_text: string;
  traj_sha: string;
  completed: boolean;
  cycle_time_s: number | null;
  fps: number;
  n_frames: number;
  duration_s: number;
  bytes: number;
  asset_bytes: number;
  segments: LaneSegment[];
  steps: LaneStep[];
  credits: { text: string; anchor: string | null }[];
  /** Seconds of every stored frame of a keyframed lane (its replay header's `t`), or null. */
  t: number[] | null;
  /** Frame indices where a segment starts (its replay header's `cuts`); empty for a uniform lane. */
  cuts: number[];
  /** Its poses are int16 (the replay header's `quantised`). */
  quantised: boolean;
  /**
   * A keyframed lane's farthest vertex from the recording its keys were chosen from, in mm (its
   * header's `meta.keyframes.max_error_vs_recording_mm`); null for a uniform lane.
   */
  max_error_vs_recording_mm: number | null;
}

export interface BuildPack {
  zip: string;
  bytes: number;
  sha256: string;
  setup_id: string;
  label: string;
  for: "recommended" | "closest";
  files: { path: string; kind: string; format: string; license: string; bytes: number }[];
  links: { name: string; url: string; license: string; reason: string }[];
  caveats: string[];
}

export interface SessionBlock {
  file: string;
  session_id: string | null;
  status: string | null;
  model: string | null;
  turns: number;
  tool_calls: number;
  spend_usd: number;
  answer: string;
}

export interface Sensitivity {
  recommendation_changes: boolean;
  order_changes: number;
  n_pairs: number;
  tool_change_s: number[];
  present_s: number[];
  note: string;
}

export interface Constant {
  key: string;
  value: number | string | number[];
  unit: string | null;
  note: string;
  tag: SpecTag;
}

export interface Provenance {
  code_version: string;
  report_sha256: string;
  task_id: string;
  controllers: string;
  evidence: string;
  conditions: { condition_id: string; seed: number; aim_offsets_mm: Record<string, number[]> }[];
  fairness_sha: string | null;
  workers: number | null;
  timings: Record<string, number>;
  determinism: { condition: string; same: boolean }[];
}

export interface HeroPage {
  schema: typeof STUDIO_HERO_SCHEMA;
  version: 1;
  label: typeof PRODUCT_MODE_LABEL;
  contract: string;
  hero: string;
  case_id: string;
  title: string;
  setting: string;
  use_case: string;
  outcome: Outcome;
  headline: string;
  plan: Plan;
  screen: Screen;
  matrix: Matrix;
  combinations: { counts: Record<string, number>; rejected: { label: string; reason: string }[] };
  finalists: FinalistRow[];
  recommendation: Recommendation | null;
  best_single: FinalistRow | null;
  /** The recommender's baseline (null in pages exported before it had one). */
  baseline: Baseline | null;
  no_winner: NoWinner | null;
  sensitivity: Sensitivity | null;
  constants: Constant[];
  fixed_caveats: string[];
  lanes: StudioLane[];
  build_pack: BuildPack | null;
  build_pack_note: string | null;
  session: SessionBlock | null;
  provenance: Provenance;
}

// ---------------------------------------------------------------- text rules

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;
const CODE_VERSION = /^[0-9a-f]{7,40}(\+dirty)?$/;
/** A URL anywhere in a string: only fields named `url` may hold one. */
const URL_TEXT = /https?:\/\/|www\./i;
const URL_KEYS: ReadonlySet<string> = new Set(["url"]);

/**
 * A brand pattern from a token list (whole words, any case), the same construction as
 * rtt.effectors.types.find_brand_tokens and rtt.library.brands.find_library_brands.
 */
export function brandPattern(tokens: readonly string[]): RegExp | null {
  const list = [...new Set(tokens.map((t) => t.trim()).filter(Boolean))];
  if (list.length === 0) return null;
  const alt = list
    .sort((a, b) => b.length - a.length)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join("|");
  return new RegExp(`(?<![a-z0-9])(${alt})(?![a-z0-9])`, "i");
}

/** Every string under `root` with its JSON path and the key it sits under. */
function* strings(n: Node, key: string | null = null): Generator<[Node, string | null]> {
  const v = n.value;
  if (typeof v === "string") yield [n, key];
  else if (Array.isArray(v)) for (const item of n.items()) yield* strings(item, key);
  else if (v && typeof v === "object") for (const [k, child] of n.entries()) yield* strings(child, k);
}

/**
 * Fails on a URL outside a `url` field, and (with `brand`) on a brand token in any string that is
 * not a URL. Shown text is brand free and URLs are link targets only.
 */
export function checkText(root: Node, brand: RegExp | null = null): void {
  for (const [n, key] of strings(root)) {
    const s = n.value as string;
    if (key !== null && URL_KEYS.has(key)) continue;
    if (URL_TEXT.test(s)) n.fail(`a URL may appear only in a field named "url", got ${JSON.stringify(s.slice(0, 60))}`);
    const hit = brand?.exec(s);
    if (hit) n.fail(`names a brand (${JSON.stringify(hit[1])}); every string shown on the site is brand free`);
  }
}

// ---------------------------------------------------------------- readers

function slug(n: Node): string {
  const s = n.nonEmpty();
  if (!SLUG.test(s)) n.fail(`expected a URL-safe id (lowercase letters, digits, . _ -), got ${JSON.stringify(s)}`);
  return s;
}

function sha(n: Node): string {
  const s = n.str();
  if (!SHA256.test(s)) n.fail(`expected a sha256 (64 lowercase hex), got ${JSON.stringify(s.slice(0, 20))}`);
  return s;
}

function codeVersion(n: Node): string {
  const s = n.str();
  if (!CODE_VERSION.test(s)) n.fail(`expected a git commit (optionally +dirty), got ${JSON.stringify(s)}`);
  return s;
}

/** A path under /data: relative, no "..", no scheme. */
function dataPath(n: Node): string {
  const s = n.nonEmpty();
  if (s.startsWith("/") || s.includes("..") || /^[a-z]+:/i.test(s)) n.fail(`expected a relative data path, got ${JSON.stringify(s)}`);
  return s;
}

function optNum(n: Node): number | null {
  return n.nullable((x) => x.num());
}

function optStr(n: Node): string | null {
  return n.nullable((x) => x.str());
}

/** An optional field: absent and null both read as null. */
function opt<T>(n: Node, key: string, read: (x: Node) => T): T | null {
  return n.has(key) ? n.get(key).nullable(read) : null;
}

function count(n: Node): number {
  const v = n.int();
  if (v < 0) n.fail(`expected a count >= 0, got ${v}`);
  return v;
}

function label(n: Node): typeof PRODUCT_MODE_LABEL {
  return n.literal(PRODUCT_MODE_LABEL);
}

function countMap(n: Node): Record<string, number> {
  return Object.fromEntries(n.entries().map(([k, v]) => [k, count(v)]));
}

/** A failure mode: one of STEP_FAILURE_MODES, each of which has its plain label. */
function failureMode(n: Node): string {
  return n.oneOf(STEP_FAILURE_MODES);
}

/** Runs per failure mode, keyed only by known modes. */
function modeCounts(n: Node): Record<string, number> {
  return Object.fromEntries(
    n.entries().map(([k, v]) => {
      if (!(STEP_FAILURE_MODES as readonly string[]).includes(k)) v.fail(`unknown failure mode ${JSON.stringify(k)} (rtt.studio.types.STEP_FAILURE_MODES)`);
      return [k, count(v)];
    }),
  );
}

function rateMap(n: Node): Record<string, number> {
  return Object.fromEntries(n.entries().map(([k, v]) => [k, v.unit()]));
}

function scalars(n: Node): Record<string, Scalar> {
  return Object.fromEntries(
    n.entries().map(([k, v]) => {
      const x = v.value;
      if (typeof x === "number" && Number.isFinite(x)) return [k, x];
      if (typeof x === "string" || typeof x === "boolean") return [k, x];
      return v.fail("expected a number, string or boolean");
    }),
  );
}

/** A Studio tier; v1 effector fidelities map onto it (`core` is rigid, as STUDIO.md 6 says). */
function tier(n: Node): Tier {
  const s = n.oneOf([...TIERS, "core"] as const);
  return s === "core" ? "rigid" : s;
}

function strList(n: Node): string[] {
  return n.items().map((x) => x.nonEmpty());
}

function readTool(n: Node, i: number): ToolBrief {
  if (typeof n.value === "string") {
    return { index: i, display_name: n.nonEmpty(), effector_id: null, family: null, family_title: null, fidelity: null, off_the_shelf: null, caveat: null, studio_only: false, rating: null };
  }
  const tool: ToolBrief = {
    index: n.has("index") ? count(n.get("index")) : i,
    display_name: n.get("display_name").nonEmpty(),
    effector_id: opt(n, "effector_id", sha),
    family: opt(n, "family", slug),
    family_title: opt(n, "family_title", (x) => x.nonEmpty()),
    fidelity: opt(n, "fidelity", tier),
    off_the_shelf: opt(n, "off_the_shelf", (x) => x.bool()),
    caveat: opt(n, "caveat", (x) => x.nonEmpty()),
    studio_only: n.has("studio_only") ? n.get("studio_only").bool() : false,
    rating: opt(n, "rating", (x) => x.nonEmpty()),
  };
  studioOnlyRules(n, tool);
  return tool;
}

/**
 * A process tool or a vacuum area frame says so (`studio_only`) and carries its caveat; only such
 * a tool states a rating (scripts/export_studio_web.py studio_tool).
 */
function studioOnlyRules(n: Node, t: { family: string | null; display_name: string; studio_only: boolean; rating: string | null; caveat: string | null }): void {
  const kind = t.family === "process_tool" || /^Vacuum area frame\b/.test(t.display_name);
  if (kind && !t.studio_only) n.fail(`${t.display_name} is a Studio-only tool and says so (studio_only)`);
  if (t.studio_only && !t.caveat) n.fail(`${t.display_name} is a Studio-only tool and carries its caveat`);
  if (t.rating !== null && !t.studio_only) n.get("rating").fail("only a Studio-only tool states a rating");
}

function readBrief(n: Node): SetupBrief {
  const completed = count(n.get("completed_runs"));
  const runs = count(n.get("n_runs"));
  const rate = n.get("completion_rate").unit();
  if (completed > runs) n.get("completed_runs").fail(`${completed} completed runs of ${runs}`);
  if (runs > 0 && Math.abs(rate - completed / runs) > 1e-3) {
    n.get("completion_rate").fail(`completion rate ${rate} is not ${completed} of ${runs} runs`);
  }
  const tools = n
    .get("tools")
    .items()
    .map((t, i) => readTool(t, i));
  if (tools.length === 0) n.get("tools").fail("a setup has at least one tool");
  return {
    setup_id: sha(n.get("setup_id")),
    kind: n.get("kind").oneOf(SETUP_KINDS),
    kind_title: n.get("kind_title").nonEmpty(),
    label: n.get("label").nonEmpty(),
    tools,
    completion_rate: rate,
    completed_runs: completed,
    n_runs: runs,
    fidelity: tier(n.get("fidelity")),
    stops_at: optStr(n.get("stops_at")),
    cycle_time_s: optNum(n.get("cycle_time_s")),
  };
}

function readEntry(n: Node): HeroEntry {
  const hero = slug(n.get("hero"));
  const page = dataPath(n.get("page"));
  if (page !== `studio/${hero}.json`) n.get("page").fail(`expected studio/${hero}.json`);
  const entry: HeroEntry = {
    hero,
    case_id: slug(n.get("case_id")),
    title: n.get("title").nonEmpty(),
    setting: n.get("setting").nonEmpty(),
    use_case: n.get("use_case").nonEmpty(),
    page,
    task_id: sha(n.get("task_id")),
    n_steps: count(n.get("n_steps")),
    n_candidates: count(n.get("n_candidates")),
    outcome: n.get("outcome").oneOf(OUTCOMES),
    headline: n.get("headline").nonEmpty(),
    recommended: n.get("recommended").nullable(readBrief),
    best_single: n.get("best_single").nullable(readBrief),
    closest: opt(n, "closest", readBrief),
  };
  if (entry.outcome === "recommended" && !entry.recommended) n.get("recommended").fail("a recommended outcome names its setup");
  if (entry.outcome === "no_recommendation" && entry.recommended) n.get("recommended").fail("no recommendation, but a recommended setup");
  return entry;
}

export interface ParseOptions {
  /** Brand pattern for the shown text (brandPattern of the contract's token lists); null skips it. */
  brand?: RegExp | null;
}

/** Validate public/data/studio/index.json. */
export function parseStudioIndex(raw: unknown, file = "studio/index.json", opts: ParseOptions = {}): StudioIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal(STUDIO_INDEX_SCHEMA);
  root.get("version").literal(1);
  checkText(root, opts.brand ?? null);
  const heroes = root.get("heroes").items().map(readEntry);
  const seen = new Set<string>();
  heroes.forEach((h, i) => {
    if (seen.has(h.hero)) root.get("heroes").items()[i]!.get("hero").fail(`duplicate hero ${h.hero}`);
    seen.add(h.hero);
  });
  return {
    schema: STUDIO_INDEX_SCHEMA,
    version: 1,
    label: label(root.get("label")),
    contract: root.get("contract").nonEmpty(),
    code_version: codeVersion(root.get("code_version")),
    heroes,
  };
}

function readObject(n: Node): PlanObject {
  const pose = n.get("pose");
  return {
    name: n.get("name").nonEmpty(),
    kind: slug(n.get("kind")),
    kind_title: n.get("kind_title").nonEmpty(),
    label: n.get("label").nonEmpty(),
    params: scalars(n.get("params")),
    pose: { x_mm: pose.get("x_mm").num(), y_mm: pose.get("y_mm").num(), yaw_deg: pose.get("yaw_deg").num() },
    ref: opt(n, "ref", (x) => x.nonEmpty()),
  };
}

function readStep(n: Node): PlanStep {
  const c = n.get("constraints");
  const step: PlanStep = {
    id: n.get("id").nonEmpty(),
    primitive: slug(n.get("primitive")),
    primitive_title: n.get("primitive_title").nonEmpty(),
    object: n.get("object").nonEmpty(),
    handle: optStr(n.get("handle")),
    target: optStr(n.get("target")),
    params: scalars(n.get("params")),
    constraints: {
      max_force_n: optNum(c.get("max_force_n")),
      max_torque_nm: optNum(c.get("max_torque_nm")),
      precision_mm: optNum(c.get("precision_mm")),
      precision_deg: optNum(c.get("precision_deg")),
    },
    success: n
      .get("success")
      .items()
      .map((s) => ({
        quantity: s.get("quantity").nonEmpty(),
        unit: s.get("unit").str(),
        at_least: optNum(s.get("at_least")),
        at_most: optNum(s.get("at_most")),
        text: s.get("text").nonEmpty(),
      })),
    after: n.get("after").strs(),
    hold_after: n.get("hold_after").bool(),
    description: n.get("description").str(),
    fidelity: tier(n.get("fidelity")),
    caveats: strList(n.get("caveats")),
  };
  if (step.fidelity === "approximation" && step.caveats.length === 0) n.get("caveats").fail("an approximation-tier step carries its caveat");
  return step;
}

function readPlan(n: Node): Plan {
  const plan: Plan = {
    task_id: sha(n.get("task_id")),
    title: n.get("title").nonEmpty(),
    description: n.get("description").str(),
    source: n.get("source").nonEmpty(),
    result: n.get("result").nonEmpty(),
    calls: opt(n, "calls", count),
    model: opt(n, "model", (x) => x.nonEmpty()),
    layout_moves: n.has("layout_moves") ? strList(n.get("layout_moves")) : [],
    assumptions: n.has("assumptions") ? strList(n.get("assumptions")) : [],
    clarifications: n.has("clarifications")
      ? n
          .get("clarifications")
          .items()
          .map((c) => ({ question: c.get("question").nonEmpty(), assumed: c.get("assumed").nonEmpty() }))
      : [],
    unsupported: n.has("unsupported")
      ? n
          .get("unsupported")
          .items()
          .map((u) => ({ what: u.get("what").nonEmpty(), why: u.get("why").nonEmpty(), nearest: opt(u, "nearest", (x) => x.nonEmpty()) }))
      : [],
    objects: n.get("objects").items().map(readObject),
    steps: n.get("steps").items().map(readStep),
  };
  if (plan.steps.length === 0) n.get("steps").fail("a plan has at least one step");
  const names = new Set(plan.objects.map((o) => o.name));
  const ids = new Set<string>();
  n.get("steps")
    .items()
    .forEach((s, i) => {
      const step = plan.steps[i]!;
      if (ids.has(step.id)) s.get("id").fail(`duplicate step id ${step.id}`);
      if (!names.has(step.object)) s.get("object").fail(`step ${step.id} acts on ${step.object}, which the plan does not list`);
      if (step.target !== null && !names.has(step.target)) s.get("target").fail(`step ${step.id} targets ${step.target}, which the plan does not list`);
      for (const a of step.after) if (!ids.has(a)) s.get("after").fail(`step ${step.id} comes after ${a}, which is not an earlier step`);
      ids.add(step.id);
    });
  return plan;
}

function readCandidate(n: Node): Candidate {
  const c: Candidate = {
    effector_id: sha(n.get("effector_id")),
    display_name: n.get("display_name").nonEmpty(),
    family: slug(n.get("family")),
    family_title: n.get("family_title").nonEmpty(),
    source: n.get("source").nonEmpty(),
    off_the_shelf: n.get("off_the_shelf").bool(),
    mechanism: n.get("mechanism").nonEmpty(),
    fidelity: tier(n.get("fidelity")),
    built_for: opt(n, "built_for", (x) => x.nonEmpty()),
    status: n.get("status").oneOf(["screened", "pruned"] as const),
    pruned_reason: opt(n, "pruned_reason", (x) => x.nonEmpty()),
    pruned_steps: n.has("pruned_steps")
      ? n
          .get("pruned_steps")
          .items()
          .map((p) => ({ step_id: p.get("step_id").nonEmpty(), reason: p.get("reason").nonEmpty() }))
      : [],
    studio_only: n.has("studio_only") ? n.get("studio_only").bool() : false,
    rating: opt(n, "rating", (x) => x.nonEmpty()),
    caveat: opt(n, "caveat", (x) => x.nonEmpty()),
  };
  studioOnlyRules(n, c);
  // Pruned from everything: one reason, or a reason per step (the export gives either).
  if (c.status === "pruned" && c.pruned_reason === null && c.pruned_steps.length === 0) n.get("pruned_reason").fail("a pruned candidate says why");
  return c;
}

function readScreen(n: Node): Screen {
  const s: Screen = {
    n_candidates: count(n.get("n_candidates")),
    n_screened: count(n.get("n_screened")),
    n_conditions: count(n.get("n_conditions")),
    runs: count(n.get("runs")),
    wall_s: n.get("wall_s").nonNeg(),
    candidates: n.get("candidates").items().map(readCandidate),
    pruned_reasons: n
      .get("pruned_reasons")
      .items()
      .map((r) => ({ reason: r.get("reason").nonEmpty(), count: count(r.get("count")) })),
  };
  if (s.candidates.length !== s.n_candidates) n.get("candidates").fail(`lists ${s.candidates.length} candidates, n_candidates is ${s.n_candidates}`);
  const screened = s.candidates.filter((c) => c.status === "screened").length;
  if (screened !== s.n_screened) n.get("n_screened").fail(`${screened} candidates are screened, n_screened is ${s.n_screened}`);
  return s;
}

function readCell(n: Node): MatrixCell {
  const state = n.get("state").oneOf(["run", "pruned", "not_simulated"] as const);
  if (state === "pruned") return { state, reason: n.get("reason").nonEmpty() };
  if (state === "not_simulated") return { state, reason: opt(n, "reason", (x) => x.nonEmpty()) };
  const cell = {
    state,
    rate: n.get("rate").unit(),
    n: count(n.get("n")),
    successes: count(n.get("successes")),
    primitive_used: opt(n, "primitive_used", slug),
    failure_modes: n.has("failure_modes") ? modeCounts(n.get("failure_modes")) : {},
    context: opt(n, "context", (x) => x.nonEmpty()),
  };
  if (cell.successes > cell.n) n.get("successes").fail(`${cell.successes} successes of ${cell.n}`);
  if (cell.n > 0 && Math.abs(cell.rate - cell.successes / cell.n) > 1e-3) n.get("rate").fail(`rate ${cell.rate} is not ${cell.successes} of ${cell.n}`);
  return cell;
}

function readMatrix(n: Node, steps: string[]): Matrix {
  const ids = n.get("steps").strs();
  if (ids.length !== steps.length || ids.some((s, i) => s !== steps[i])) {
    n.get("steps").fail(`expected the plan's steps [${steps.join(", ")}], got [${ids.join(", ")}]`);
  }
  const rows = n
    .get("rows")
    .items()
    .map((r) => {
      const cellsNode = r.get("cells");
      for (const key of Object.keys(cellsNode.object())) if (!ids.includes(key)) cellsNode.get(key).fail(`no step ${key} in the plan`);
      const cells: Record<string, MatrixCell> = {};
      for (const id of ids) {
        if (!cellsNode.has(id)) cellsNode.fail(`no cell for step ${id}`);
        cells[id] = readCell(cellsNode.get(id));
      }
      return {
        effector_id: sha(r.get("effector_id")),
        display_name: r.get("display_name").nonEmpty(),
        family_title: r.get("family_title").nonEmpty(),
        fidelity: tier(r.get("fidelity")),
        off_the_shelf: r.get("off_the_shelf").bool(),
        cells,
      };
    });
  return { steps: ids, rows };
}

function readFinalist(n: Node): FinalistRow {
  const brief = readBrief(n);
  const c = n.get("complexity");
  return {
    ...brief,
    roles: n
      .get("roles")
      .items()
      .map((r) => r.oneOf(ROLES)),
    eligible: n.get("eligible").bool(),
    reason: optStr(n.get("reason")),
    tool_details: n.has("tool_details")
      ? n
          .get("tool_details")
          .items()
          .map((t, i) => readTool(t, i))
      : brief.tools,
    step_success: rateMap(n.get("step_success")),
    weakest_step: n.get("weakest_step").nullable((w) => ({ step_id: w.get("step_id").nonEmpty(), success_rate: w.get("success_rate").unit() })),
    complexity: {
      n_tools: count(c.get("n_tools")),
      n_actuators: count(c.get("n_actuators")),
      total_mass_kg: c.get("total_mass_kg").nonNeg(),
      n_changes: count(c.get("n_changes")),
      bracket: c.get("bracket").bool(),
    },
    payload_kg: opt(n, "payload_kg", (x) => x.nonNeg()),
    failure_modes: modeCounts(n.get("failure_modes")),
    stops_at_detail: Object.fromEntries(
      n
        .get("stops_at_detail")
        .entries()
        .map(([k, v]) => [k, { runs: count(v.get("runs")), failure_modes: modeCounts(v.get("failure_modes")) }]),
    ),
  };
}

function readGeometry(n: Node): Geometry {
  if (n.has("changer")) {
    const ch = n.get("changer");
    return { kind: "changer", stack_mm: opt(ch, "stack_mm", (x) => x.num()), changeover_s: opt(ch, "changeover_s", (x) => x.nonNeg()) };
  }
  return {
    kind: "head",
    mounts: n
      .get("mounts")
      .items()
      .map((m) => ({
        angle_deg: m.get("angle_deg").num(),
        standoff_mm: m.get("standoff_mm").num(),
        lateral_mm: m.get("lateral_mm").num(),
        spin_deg: m.get("spin_deg").num(),
      })),
    clearance_mm: opt(n, "clearance_mm", (x) => x.num()),
    present_s: opt(n, "present_s", (x) => x.nonNeg()),
  };
}

function readRecommendation(n: Node, stepIds: string[]): Recommendation {
  const brief = readBrief(n);
  const steps: RecStep[] = n
    .get("steps")
    .items()
    .map((s) => {
      const step: RecStep = {
        step_id: s.get("step_id").nonEmpty(),
        primitive: slug(s.get("primitive")),
        primitive_used: opt(s, "primitive_used", slug),
        tool: count(s.get("tool")),
        tool_label: s.get("tool_label").nonEmpty(),
        success_rate: s.get("success_rate").unit(),
        fidelity: tier(s.get("fidelity")),
        caveats: strList(s.get("caveats")),
        verify_on_hardware: s.get("verify_on_hardware").bool(),
      };
      if (!stepIds.includes(step.step_id)) s.get("step_id").fail(`no step ${step.step_id} in the plan`);
      if (step.fidelity === "approximation" && (step.caveats.length === 0 || !step.verify_on_hardware)) {
        s.fail("an approximation-tier step carries its caveat and asks to verify on hardware");
      }
      return step;
    });
  const rec: Recommendation = {
    ...brief,
    why: n.get("why").nonEmpty(),
    caveat: optStr(n.get("caveat")),
    verify_on_hardware: n.get("verify_on_hardware").bool(),
    geometry: opt(n, "geometry", readGeometry),
    steps,
    runner_up_rigid: opt(n, "runner_up_rigid", (r) => ({ label: optStr(r.get("label")), basis: r.get("basis").nonEmpty() })),
    versus_best_single: opt(n, "versus_best_single", (v) => ({
      same_setup: v.get("same_setup").bool(),
      completion_delta: v.get("completion_delta").num(),
      cycle_delta_s: optNum(v.get("cycle_delta_s")),
      extra_tools: v.get("extra_tools").int(),
    })),
    versus_baseline: opt(n, "versus_baseline", (v) => {
      if (v.has("rule")) v.get("rule").literal("R01");
      return {
        same_setup: v.get("same_setup").bool(),
        completion_delta: v.get("completion_delta").num(),
        cycle_delta_s: optNum(v.get("cycle_delta_s")),
        added_pieces: v.get("added_pieces").int(),
        needed_gain: optNum(v.get("needed_gain")),
        basis: v.get("basis").oneOf(BASELINE_BASES),
        note: opt(v, "note", (x) => x.nonEmpty()),
      };
    }),
  };
  const approx = steps.some((s) => s.fidelity === "approximation") || rec.fidelity === "approximation";
  if (approx && !rec.verify_on_hardware) n.get("verify_on_hardware").fail("an approximation-tier recommendation asks to verify on hardware");
  if (rec.verify_on_hardware && rec.caveat !== VERIFY_CAVEAT) n.get("caveat").fail(`expected ${JSON.stringify(VERIFY_CAVEAT)}`);
  if (approx && rec.runner_up_rigid === null) n.get("runner_up_rigid").fail("an approximation-tier recommendation carries its rigid runner-up (or why there is none)");
  return rec;
}

/** The page's baseline: a finalist single tool by id (carrying the "baseline" role), or why there is none. */
function readBaseline(n: Node, finalists: FinalistRow[], rec: Recommendation | null): Baseline {
  const id = n.get("setup_id").nullable(sha);
  const basis = opt(n, "basis", (x) => x.nonEmpty());
  if (id === null) {
    if (basis === null) n.get("basis").fail("no baseline: the page says why");
    if (finalists.some((f) => f.roles.includes("baseline"))) n.fail("a finalist carries the baseline role, but the page names no baseline");
    if (rec?.versus_baseline) n.fail("the recommendation is compared with a baseline the page does not name");
    return { row: null, basis };
  }
  const row = finalists.find((f) => f.setup_id === id);
  if (!row) return n.get("setup_id").fail("the baseline is not among the finalists");
  if (row.kind !== "single") n.get("setup_id").fail("the baseline is a single tool");
  const marked = finalists.filter((f) => f.roles.includes("baseline")).map((f) => f.setup_id);
  if (marked.length !== 1 || marked[0] !== id) n.fail("exactly the baseline's finalist row carries the baseline role");
  const vs = rec?.versus_baseline;
  if (vs && vs.same_setup !== (rec!.setup_id === id)) n.fail("versus_baseline.same_setup disagrees with the baseline's id");
  return { row, basis: null };
}

function readBlocker(n: Node): Blocker {
  return {
    step_id: n.get("step_id").nonEmpty(),
    primitive: slug(n.get("primitive")),
    quantity: n.get("quantity").nonEmpty(),
    unit: n.get("unit").str(),
    at_least: optNum(n.get("at_least")),
    at_most: optNum(n.get("at_most")),
    text: n.get("text").nonEmpty(),
    conditions: n.has("conditions") ? n.get("conditions").strs() : [],
    families: n
      .get("families")
      .items()
      .map((f) => ({
        family: slug(f.get("family")),
        family_title: f.get("family_title").nonEmpty(),
        n_candidates: count(f.get("n_candidates")),
        n_runs: count(f.get("n_runs")),
        peak: optNum(f.get("peak")),
        best: opt(f, "best", (x) => x.nonEmpty()),
        failure_modes: f.has("failure_modes") ? modeCounts(f.get("failure_modes")) : {},
      })),
    max_peak: opt(n, "max_peak", (x) => x.num()),
  };
}

function readSegment(n: Node): LaneSegment {
  const seg: LaneSegment = {
    index: count(n.get("index")),
    tool: count(n.get("tool")),
    tool_label: n.get("tool_label").nonEmpty(),
    start_s: n.get("start_s").nonNeg(),
    end_s: n.get("end_s").nonNeg(),
    change: n.get("change").oneOf(CHANGES),
    change_s: n.get("change_s").nonNeg(),
    change_start_s: 0,
    steps: n.get("steps").strs(),
  };
  seg.change_start_s = n.has("change_start_s") ? n.get("change_start_s").nonNeg() : Math.max(0, seg.start_s - seg.change_s);
  if (seg.end_s < seg.start_s) n.get("end_s").fail(`segment ends at ${seg.end_s} s before it starts at ${seg.start_s} s`);
  if (seg.change_start_s > seg.start_s + 1e-6) n.get("change_start_s").fail("the change starts after its segment");
  return seg;
}

function readLaneStep(n: Node): LaneStep {
  const step: LaneStep = {
    step_id: n.get("step_id").nonEmpty(),
    primitive: slug(n.get("primitive")),
    tool: count(n.get("tool")),
    start_s: n.get("start_s").nonNeg(),
    duration_s: n.get("duration_s").nonNeg(),
    success: n.get("success").bool(),
    failure_mode: n.get("failure_mode").nullable(failureMode),
    fidelity: tier(n.get("fidelity")),
    quantities: n.has("quantities") ? Object.fromEntries(n.get("quantities").entries().map(([k, v]) => [k, optNum(v)])) : {},
  };
  if (step.success === (step.failure_mode !== null)) n.fail("a step succeeds exactly when it has no failure mode");
  return step;
}

function readLane(n: Node, hero: string): StudioLane {
  // The exporter renamed it: the error is measured against the recording, not the simulation.
  if (n.has("max_error_mm")) n.get("max_error_mm").fail("renamed max_error_vs_recording_mm");
  const replay = dataPath(n.get("replay"));
  if (!replay.startsWith(`studio/replays/${hero}/`)) n.get("replay").fail(`expected a lane under studio/replays/${hero}/`);
  const lane: StudioLane = {
    id: slug(n.get("id")),
    title: n.get("title").nonEmpty(),
    setup_id: sha(n.get("setup_id")),
    kind: n.get("kind").oneOf(SETUP_KINDS),
    label: n.get("label").nonEmpty(),
    replay,
    condition_id: n.get("condition_id").nonEmpty(),
    seed: count(n.get("seed")),
    condition_text: "",
    traj_sha: sha(n.get("traj_sha")),
    completed: n.get("completed").bool(),
    cycle_time_s: optNum(n.get("cycle_time_s")),
    fps: n.get("fps").int(),
    n_frames: n.get("n_frames").int(),
    duration_s: n.get("duration_s").nonNeg(),
    bytes: count(n.get("bytes")),
    asset_bytes: n.has("asset_bytes") ? count(n.get("asset_bytes")) : 0,
    segments: n.get("segments").items().map(readSegment),
    steps: n.get("steps").items().map(readLaneStep),
    t: null,
    cuts: [],
    quantised: n.has("quantised") ? n.get("quantised").bool() : false,
    max_error_vs_recording_mm: opt(n, "max_error_vs_recording_mm", (x) => x.nonNeg()),
    credits: n.has("credits")
      ? n
          .get("credits")
          .items()
          .map((c) => {
            const anchor = opt(c, "anchor", (x) => x.nonEmpty());
            if (anchor !== null && !/^[A-Za-z0-9_-]+$/.test(anchor)) c.get("anchor").fail(`expected a credits-page anchor, got ${JSON.stringify(anchor)}`);
            return { text: c.get("text").nonEmpty(), anchor };
          })
      : [],
  };
  // The contract's condition_text is optional in the export; fall back to the id and seed.
  lane.condition_text = opt(n, "condition_text", (x) => x.nonEmpty()) ?? `${lane.condition_id === "nominal" ? "nominal placement" : `condition ${lane.condition_id}`}, seed ${lane.seed}`;
  if (lane.fps < 1 || lane.n_frames < 1) n.fail("a lane has at least one frame at a positive fps");
  if (n.has("t") && n.get("t").value !== null) {
    const t = n.get("t").nums();
    if (t.length !== lane.n_frames) n.get("t").fail(`expected one time per frame (${lane.n_frames}), got ${t.length}`);
    t.forEach((v, i) => {
      if (!(v >= 0) || (i > 0 && v <= t[i - 1]!)) n.get("t").items()[i]!.fail("frame times are non-negative and strictly increasing");
    });
    lane.t = t;
  }
  if (n.has("cuts")) {
    lane.cuts = n.get("cuts").items().map((c) => count(c));
    lane.cuts.forEach((c, i) => {
      if (c >= lane.n_frames || (i > 0 && c <= lane.cuts[i - 1]!)) n.get("cuts").items()[i]!.fail(`cuts are increasing frame indices below ${lane.n_frames}`);
    });
  }
  // A lane with its own frame times must end on its last one. Every other lane's clock is checked
  // against its replay header at build time (app/studio/data.ts), since a keyframed lane's times may
  // live only in the header.
  if (lane.t && Math.abs(lane.duration_s - lane.t[lane.t.length - 1]!) > 1e-3) {
    n.get("duration_s").fail(`${lane.duration_s} s, but its last frame is at ${lane.t[lane.t.length - 1]!.toFixed(3)} s`);
  }
  if (lane.segments.length === 0) n.get("segments").fail("a lane has at least one segment");
  lane.segments.forEach((seg, i) => {
    if (i === 0 && seg.change !== "none") n.get("segments").items()[i]!.get("change").fail("the first segment has no change before it");
    if (i > 0 && seg.change === "none") n.get("segments").items()[i]!.get("change").fail("every later segment starts with a tool change or a head switch");
    if (i > 0 && seg.change_start_s + 1e-6 < lane.segments[i - 1]!.end_s) n.get("segments").items()[i]!.get("start_s").fail("segments overlap");
  });
  return lane;
}

/**
 * How a lane stores its run, in words: "56 keyframes from a 100 Hz recording, within 1.0 mm of it"
 * (a keyframed lane: its fps is the recording's, not its frames'), or "601 frames at 30 fps".
 */
export function laneFramesText(lane: Pick<StudioLane, "n_frames" | "fps" | "quantised" | "max_error_vs_recording_mm" | "t">): string {
  if (lane.max_error_vs_recording_mm === null && !lane.t) return `${lane.n_frames} frames at ${lane.fps} fps`;
  const within = lane.max_error_vs_recording_mm !== null ? `, within ${lane.max_error_vs_recording_mm.toFixed(1)} mm of it` : "";
  return `${lane.n_frames} keyframes from a ${lane.fps} Hz recording${within}${lane.quantised ? ", int16 poses" : ""}`;
}

/** Validate one lane entry of a page (`$.lanes[i]` of studio/<hero>.json) on its own. */
export function parseStudioLane(raw: unknown, hero: string, file = "studio/hero.json"): StudioLane {
  return readLane(new Node(file, "$", raw), hero);
}

/**
 * What a lane's page entry and its replay header disagree on (empty when they agree): frames and
 * fps, the int16 flag, the keyframing error, frame times and cuts, the clock's end, and a frame
 * exactly at each segment's start (scripts/export_studio_web.py check_lane, on the site's side).
 * A keyframed lane's fps is nominal (its recording's rate), so its clock comes from `t` only.
 */
export function laneHeaderProblems(lane: StudioLane, header: ReplayHeaderV2): string[] {
  const out: string[] = [];
  if (header.n_frames !== lane.n_frames || header.fps !== lane.fps) {
    out.push(`the header has ${header.n_frames} frames at ${header.fps} fps; the page says ${lane.n_frames} at ${lane.fps}`);
  }
  if (header.quantised !== lane.quantised) out.push(`the header's poses are ${header.quantised ? "int16" : "float32"}; the page says ${lane.quantised ? "int16" : "float32"}`);
  const kf = header.meta.keyframes;
  if (kf && lane.max_error_vs_recording_mm === null) out.push("a keyframed lane states its error against the recording (max_error_vs_recording_mm)");
  if (lane.max_error_vs_recording_mm !== null && (!kf || Math.abs(kf.max_error_vs_recording_mm - lane.max_error_vs_recording_mm) > 1e-9)) {
    out.push(`the page's error against the recording (${lane.max_error_vs_recording_mm} mm) is not its replay's (${kf ? `${kf.max_error_vs_recording_mm} mm` : "none"})`);
  }
  if (lane.t && (!header.t || header.t.length !== lane.t.length || lane.t.some((v, k) => Math.abs(v - header.t![k]!) > 1e-9))) {
    out.push("the page's frame times differ from its replay's");
  }
  if (lane.cuts.length && (lane.cuts.length !== header.cuts.length || lane.cuts.some((c, k) => c !== header.cuts[k]))) {
    out.push(`the page's cuts [${lane.cuts.join(", ")}] differ from its replay's [${header.cuts.join(", ")}]`);
  }
  const clock = laneTimelineEnd(lane, header.t);
  if (Math.abs(lane.duration_s - clock.end) > clock.tolerance) out.push(`lasts ${lane.duration_s} s, but its replay's ${clock.why}`);
  if (header.t) {
    // A keyframed lane cuts exactly where a segment starts, so a player never interpolates across a change.
    if (header.cuts.length !== lane.segments.length - 1) out.push(`${header.cuts.length} cuts for ${lane.segments.length} segments`);
    header.cuts.forEach((c, k) => {
      const seg = lane.segments[k + 1];
      if (seg && Math.abs(header.t![c]! - seg.start_s) > 1e-3) out.push(`the frame at cut ${c} (${header.t![c]} s) is not at segment ${seg.index}'s start (${seg.start_s} s)`);
    });
  }
  return out;
}

/**
 * Where a lane's clock ends: the last of its frame times (the lane's `t`, else its replay header's),
 * or, for a uniform lane, its frames at `fps`.
 */
export function laneTimelineEnd(
  lane: Pick<StudioLane, "t" | "n_frames" | "fps">,
  headerT: readonly number[] | null = null,
): { end: number; tolerance: number; why: string } {
  const t = lane.t ?? headerT;
  if (t && t.length) return { end: t[t.length - 1]!, tolerance: 1e-3, why: `last frame is at ${t[t.length - 1]!.toFixed(3)} s` };
  const span = (lane.n_frames - 1) / lane.fps;
  return { end: span, tolerance: 2 / lane.fps + 1e-6, why: `${lane.n_frames} frames at ${lane.fps} fps span ${span.toFixed(3)} s` };
}

function readBuildPack(n: Node, hero: string): BuildPack {
  const zip = dataPath(n.get("zip"));
  if (zip !== `studio/${hero}/build_pack.zip`) n.get("zip").fail(`expected studio/${hero}/build_pack.zip`);
  const bytes = count(n.get("bytes"));
  if (bytes < 1 || bytes > PACK_CAP_BYTES) n.get("bytes").fail(`a build pack is 1 to ${PACK_CAP_BYTES} bytes, this one ${bytes}`);
  return {
    zip,
    bytes,
    sha256: sha(n.get("sha256")),
    setup_id: sha(n.get("setup_id")),
    label: n.get("label").nonEmpty(),
    for: n.get("for").oneOf(["recommended", "closest"] as const),
    files: n
      .get("files")
      .items()
      .map((f) => {
        const p = f.get("path").nonEmpty();
        if (p.startsWith("/") || p.includes("..")) f.get("path").fail("expected a path inside the pack");
        return { path: p, kind: f.get("kind").nonEmpty(), format: f.get("format").nonEmpty(), license: f.get("license").nonEmpty(), bytes: count(f.get("bytes")) };
      }),
    links: n
      .get("links")
      .items()
      .map((l) => {
        const url = l.get("url").str();
        if (!/^https:\/\/\S+$/.test(url)) l.get("url").fail("expected an https URL");
        return { name: l.get("name").nonEmpty(), url, license: l.get("license").nonEmpty(), reason: l.get("reason").str() };
      }),
    caveats: strList(n.get("caveats")),
  };
}

function readSession(n: Node, hero: string): SessionBlock {
  const file = dataPath(n.get("file"));
  if (file !== `studio/sessions/${hero}.json`) n.get("file").fail(`expected studio/sessions/${hero}.json`);
  return {
    file,
    session_id: opt(n, "session_id", (x) => x.nonEmpty()),
    status: opt(n, "status", (x) => x.nonEmpty()),
    model: opt(n, "model", (x) => x.nonEmpty()),
    turns: n.has("turns") ? count(n.get("turns")) : 0,
    tool_calls: n.has("tool_calls") ? count(n.get("tool_calls")) : 0,
    spend_usd: n.has("spend_usd") ? n.get("spend_usd").nonNeg() : 0,
    answer: n.has("answer") ? n.get("answer").str() : "",
  };
}

function readConstant(n: Node): Constant {
  const v = n.get("value");
  const value = Array.isArray(v.value) ? v.nums() : typeof v.value === "string" ? v.nonEmpty() : v.num();
  return {
    key: n.get("key").nonEmpty(),
    value,
    unit: opt(n, "unit", (x) => x.str()),
    note: n.get("note").str(),
    tag: n.get("tag").oneOf(SPEC_TAGS),
  };
}

function readProvenance(n: Node): Provenance {
  return {
    code_version: codeVersion(n.get("code_version")),
    report_sha256: sha(n.get("report_sha256")),
    task_id: sha(n.get("task_id")),
    controllers: n.get("controllers").nonEmpty(),
    evidence: n.get("evidence").nonEmpty(),
    conditions: n
      .get("conditions")
      .items()
      .map((c) => ({
        condition_id: c.get("condition_id").nonEmpty(),
        seed: count(c.get("seed")),
        aim_offsets_mm: Object.fromEntries(
          c
            .get("aim_offsets_mm")
            .entries()
            .map(([k, v]) => [k, v.nums()]),
        ),
      })),
    fairness_sha: opt(n, "fairness_sha", sha),
    workers: opt(n, "workers", count),
    timings: n.has("timings") ? Object.fromEntries(n.get("timings").entries().map(([k, v]) => [k, v.nonNeg()])) : {},
    determinism: n.has("determinism")
      ? n
          .get("determinism")
          .items()
          .map((d) => ({ condition: d.get("condition").nonEmpty(), same: d.get("same").bool() }))
      : [],
  };
}

/** Validate public/data/studio/<hero>.json and its cross-field rules. */
export function parseStudioHero(raw: unknown, file = "studio/hero.json", opts: ParseOptions = {}): HeroPage {
  const root = new Node(file, "$", raw);
  root.get("schema").literal(STUDIO_HERO_SCHEMA);
  root.get("version").literal(1);
  checkText(root, opts.brand ?? null);
  const hero = slug(root.get("hero"));
  const plan = readPlan(root.get("plan"));
  const stepIds = plan.steps.map((s) => s.id);
  const outcome = root.get("outcome").oneOf(OUTCOMES);
  const recommendation = root.get("recommendation").nullable((n) => readRecommendation(n, stepIds));
  const nw = root.get("no_winner");
  const no_winner: NoWinner | null = nw.nullable((n) => ({
    reason: n.get("reason").nonEmpty(),
    closest: n.get("closest").items().map(readFinalist),
    blocker: opt(n, "blocker", readBlocker),
  }));
  if (outcome === "recommended" && (!recommendation || no_winner)) root.get("outcome").fail("a recommended outcome has a recommendation and no no-winner report");
  if (outcome === "no_recommendation" && (recommendation || !no_winner)) root.get("outcome").fail("no recommendation: a no-winner report and no recommendation");
  const finalists = root.get("finalists").items().map(readFinalist);
  const best_single = root.get("best_single").nullable(readFinalist);
  if (best_single && best_single.kind !== "single") root.get("best_single").get("kind").fail("the best single off-the-shelf setup is a single tool");
  const baseline = opt(root, "baseline", (b) => readBaseline(b, finalists, recommendation));
  if (recommendation && finalists[0]?.setup_id !== recommendation.setup_id) root.get("finalists").fail("the recommended setup ranks first among the finalists");
  const lanes = root
    .get("lanes")
    .items()
    .map((l) => readLane(l, hero));
  const laneIds = new Set<string>();
  root
    .get("lanes")
    .items()
    .forEach((l, i) => {
      const lane = lanes[i]!;
      if (laneIds.has(lane.id)) l.get("id").fail(`duplicate lane ${lane.id}`);
      laneIds.add(lane.id);
      const expected =
        lane.id === "recommended"
          ? recommendation?.setup_id
          : lane.id === "best_single"
            ? best_single?.setup_id
            : lane.id === "closest"
              ? no_winner?.closest.find((c) => c.setup_id === lane.setup_id)?.setup_id
              : lane.setup_id;
      if (expected !== lane.setup_id) l.get("setup_id").fail(`lane ${lane.id} shows a setup that is not the page's ${lane.id} setup`);
      for (const s of lane.steps) if (!stepIds.includes(s.step_id)) l.get("steps").fail(`no step ${s.step_id} in the plan`);
    });
  const page: HeroPage = {
    schema: STUDIO_HERO_SCHEMA,
    version: 1,
    label: label(root.get("label")),
    contract: root.get("contract").nonEmpty(),
    hero,
    case_id: slug(root.get("case_id")),
    title: root.get("title").nonEmpty(),
    setting: root.get("setting").nonEmpty(),
    use_case: root.get("use_case").nonEmpty(),
    outcome,
    headline: root.get("headline").nonEmpty(),
    plan,
    screen: readScreen(root.get("screen")),
    matrix: readMatrix(root.get("matrix"), stepIds),
    combinations: {
      counts: countMap(root.get("combinations").get("counts")),
      rejected: root
        .get("combinations")
        .get("rejected")
        .items()
        .map((r) => ({ label: r.get("label").str(), reason: r.get("reason").nonEmpty() })),
    },
    finalists,
    recommendation,
    best_single,
    baseline,
    no_winner,
    sensitivity: opt(root, "sensitivity", (s) => ({
      recommendation_changes: s.get("recommendation_changes").bool(),
      order_changes: count(s.get("order_changes")),
      n_pairs: count(s.get("n_pairs")),
      tool_change_s: s.get("tool_change_s").nums(),
      present_s: s.get("present_s").nums(),
      note: s.get("note").str(),
    })),
    constants: root.get("constants").items().map(readConstant),
    fixed_caveats: strList(root.get("fixed_caveats")),
    lanes,
    build_pack: root.get("build_pack").nullable((n) => readBuildPack(n, hero)),
    build_pack_note: opt(root, "build_pack_note", (x) => x.nonEmpty()),
    session: opt(root, "session", (n) => readSession(n, hero)),
    provenance: readProvenance(root.get("provenance")),
  };
  if (page.provenance.task_id !== plan.task_id) root.get("provenance").get("task_id").fail("differs from the plan's task id");
  return page;
}

// ---------------------------------------------------------------- formatting

export function pct(rate: number | null | undefined, digits = 0): string {
  return rate === null || rate === undefined ? "n/a" : `${(rate * 100).toFixed(digits)}%`;
}

/** "8 of 8 runs", "1 of 3 runs", "1 run". */
export function runsText(completed: number, n: number): string {
  return `${completed} of ${n} ${n === 1 ? "run" : "runs"}`;
}

export function seconds(s: number | null | undefined, digits = 1): string {
  return s === null || s === undefined ? "n/a" : `${s.toFixed(digits)} s`;
}

export function massText(kg: number | null | undefined): string {
  if (kg === null || kg === undefined) return "n/a";
  return kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`;
}

/** "open_lid" -> "Open lid". */
export function humanize(code: string): string {
  const s = code.replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * A reason from the recommender in plain words: a failure-mode code inside it becomes its label
 * ("a run reset the simulator (physics_invalid)" reads "... (physics reset)").
 */
export function plainReason(text: string): string {
  return text.replace(/\b[a-z]+(?:_[a-z]+)+\b/g, (code) => FAILURE_LABELS[code] ?? code);
}

/** A failure mode in plain words (the parsers admit only labelled modes; anything else reads "failed"). */
export function failureLabel(mode: string): string {
  return FAILURE_LABELS[mode] ?? "failed";
}

/** "slip 5, limit exceeded 3", most frequent first; "none" for an empty map. */
export function failureText(modes: Record<string, number>, skip: readonly string[] = []): string {
  const list = Object.entries(modes)
    .filter(([m, n]) => n > 0 && !skip.includes(m))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  return list.length ? list.map(([m, n]) => `${failureLabel(m)} ${n}`).join(", ") : "none";
}

/** A step's title for headings: "2. Pick carton". */
export function stepTitle(plan: Pick<Plan, "steps">, id: string): string {
  const i = plan.steps.findIndex((s) => s.id === id);
  return i < 0 ? humanize(id) : `${i + 1}. ${humanize(id)}`;
}

export function scalarText(v: Scalar): string {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(3)));
  return v.replace(/_/g, " ");
}

/** "width_mm: 340" -> ["width", "340 mm"]; other keys are humanized as they are. */
export function paramText(key: string, v: Scalar): [string, string] {
  const m = /^(.+?)_(mm_s|deg_s|nms|nm|mm|deg|kpa|kg|n|s|g)$/.exec(key);
  const units: Record<string, string> = { mm: "mm", deg: "deg", n: "N", nm: "N m", nms: "N m s", s: "s", kg: "kg", g: "g", mm_s: "mm/s", deg_s: "deg/s", kpa: "kPa" };
  if (m && typeof v === "number") return [m[1]!.replace(/_/g, " "), `${scalarText(v)} ${units[m[2]!]}`];
  return [key.replace(/_/g, " "), scalarText(v)];
}

export function constantValue(c: Constant): string {
  if (c.unit === "fraction" && typeof c.value === "number") return `${Number((c.value * 100).toFixed(1))}%`;
  const v = Array.isArray(c.value) ? c.value.map((x) => scalarText(x)).join(", ") : scalarText(c.value);
  return c.unit && c.unit !== "rate" && c.unit !== "fraction" && c.unit !== "flag" ? `${v} ${c.unit}` : v;
}

/** "Tool change: a fixed 6 s, not simulated", "Head switch: a fixed 1.5 s, not simulated". */
export function changeText(change: Change, s: number): string {
  const t = Number.isInteger(s) ? String(s) : String(Number(s.toFixed(2)));
  if (change === "tool_change") return `Tool change: a fixed ${t} s, not simulated`;
  if (change === "present") return `Head switch: a fixed ${t} s, not simulated`;
  return "";
}

function num(v: number): string {
  return Math.abs(v) >= 10 ? v.toFixed(1).replace(/\.0$/, "") : String(Number(v.toFixed(2)));
}

/** A quantity's name without its unit suffix: "torque_nm" -> "torque". */
export function quantityName(q: string): string {
  return q.replace(/_(mm_s|deg_s|nms|nm|mm|deg|kpa|kg|n|s|g)$/, "").replace(/_/g, " ");
}

/** A blocker's target in words: "at least 3 N m", "3 to 8 N m". */
export function targetText(b: Pick<Blocker, "at_least" | "at_most" | "unit">): string {
  const u = b.unit ? ` ${b.unit}` : "";
  if (b.at_least !== null && b.at_most !== null) return `at least ${num(b.at_least)}${u}, at most ${num(b.at_most)}${u}`;
  if (b.at_least !== null) return `at least ${num(b.at_least)}${u}`;
  if (b.at_most !== null) return `at most ${num(b.at_most)}${u}`;
  return "no bound";
}

/** "0.09 to 1.15 N m against at least 3 N m": the peaks every family reached against the target. */
export function blockerRange(b: Blocker): string {
  const peaks = b.families.map((f) => f.peak).filter((p): p is number => p !== null);
  const u = b.unit ? ` ${b.unit}` : "";
  if (peaks.length === 0) return `no peak recorded against ${targetText(b)}`;
  const lo = Math.min(...peaks);
  const hi = Math.max(...peaks);
  const span = lo === hi ? `${num(hi)}${u}` : `${num(lo)} to ${num(hi)}${u}`;
  return `${span} against ${targetText(b)}`;
}

// ---------------------------------------------------------------- lanes on the clock

/** What a lane shows at time `t` (replay time is the task clock, changes included). */
export type LaneMoment =
  | { kind: "step"; step: LaneStep; segment: LaneSegment | null }
  | { kind: "change"; change: Exclude<Change, "none">; change_s: number; to: LaneSegment }
  | { kind: "idle"; segment: LaneSegment | null }
  | { kind: "ended" };

/** When the run stops: the end of its last step (the task clock at the end of the run). */
export function runEnd(lane: Pick<StudioLane, "steps" | "duration_s">): number {
  if (lane.steps.length === 0) return lane.duration_s;
  return Math.max(...lane.steps.map((s) => s.start_s + s.duration_s));
}

/** Segment containing time `t`, or null. */
export function segmentAt(lane: Pick<StudioLane, "segments">, t: number): LaneSegment | null {
  return lane.segments.find((s) => t >= s.start_s && t < s.end_s) ?? null;
}

/** The step, change gap or idle stretch at time `t`; after the last step, "ended". */
export function laneAt(lane: Pick<StudioLane, "segments" | "steps" | "duration_s">, t: number): LaneMoment {
  for (let k = 1; k < lane.segments.length; k++) {
    const seg = lane.segments[k]!;
    if (seg.change === "none") continue;
    if (t >= seg.change_start_s && t < seg.start_s) return { kind: "change", change: seg.change, change_s: seg.change_s, to: seg };
  }
  const step = lane.steps.find((s) => t >= s.start_s && t < s.start_s + s.duration_s);
  if (step) {
    const segment = lane.segments.find((s) => s.steps.includes(step.step_id) && t >= s.start_s - 1e-9 && t <= s.end_s + 1e-9) ?? segmentAt(lane, t);
    return { kind: "step", step, segment };
  }
  if (t >= runEnd(lane)) return { kind: "ended" };
  return { kind: "idle", segment: segmentAt(lane, t) };
}

/** What the "Changing tool" card over a replay says while the replay holds a pose at a cut. */
export interface ChangeCard {
  title: "Changing tool";
  /** "Tool change: a fixed 7.5 s, not simulated" or the head-switch equivalent. */
  detail: string;
  /** "Next: tool 2, Suction cup, 40 mm bellows", when the lane names the next tool. */
  next: string | null;
  start_s: number;
  end_s: number;
  /** How far through the change the clock is, 0 to 1. */
  progress: number;
}

/**
 * The card to show at time `t`, or null. The lane's segments give the change window and the
 * next tool; a keyframed replay's own cut gap (Replay.cutGapAt) covers a jump the lane does not
 * describe. Every gap at a cut shows the card (`minGapS` can hide very short ones).
 */
export function changeCardAt(
  lane: Pick<StudioLane, "segments" | "steps" | "duration_s">,
  t: number,
  gap: { start_s: number; end_s: number } | null = null,
  minGapS = 0,
): ChangeCard | null {
  const m = laneAt(lane, t);
  const progress = (a: number, b: number) => (b > a ? Math.min(Math.max((t - a) / (b - a), 0), 1) : 1);
  if (m.kind === "change") {
    const start = m.to.change_start_s;
    const end = m.to.start_s;
    const s = Number.isInteger(m.change_s) ? String(m.change_s) : String(Number(m.change_s.toFixed(2)));
    return {
      title: "Changing tool",
      detail:
        m.change === "present"
          ? `Head switch: the wrist turns the next tool to the work, a fixed ${s} s, not simulated`
          : `Tool change: a fixed ${s} s, not simulated`,
      next: `Next: tool ${m.to.tool + 1}, ${m.to.tool_label}`,
      start_s: start,
      end_s: end,
      progress: progress(start, end),
    };
  }
  if (gap && gap.end_s - gap.start_s >= minGapS && t >= gap.start_s && t < gap.end_s) {
    const next = lane.segments.find((s) => s.start_s >= gap.end_s - 1e-6) ?? null;
    return {
      title: "Changing tool",
      detail: `Not simulated: the replay holds for ${(gap.end_s - gap.start_s).toFixed(1)} s, then cuts to the next segment`,
      next: next ? `Next: tool ${next.tool + 1}, ${next.tool_label}` : null,
      start_s: gap.start_s,
      end_s: gap.end_s,
      progress: progress(gap.start_s, gap.end_s),
    };
  }
  return null;
}

/** The run's outcome stamp: "Completed in 58.1 s" or "Stopped at 1. Open lid: slip". */
export function outcomeText(lane: Pick<StudioLane, "completed" | "cycle_time_s" | "steps">, plan: Pick<Plan, "steps">): string {
  if (lane.completed) return lane.cycle_time_s !== null ? `Completed in ${lane.cycle_time_s.toFixed(1)} s` : "Completed";
  const failed = lane.steps.find((s) => !s.success);
  if (!failed) return "Stopped";
  return `Stopped at ${stepTitle(plan, failed.step_id)}: ${failed.failure_mode ? failureLabel(failed.failure_mode) : "failed"}`;
}

// ---------------------------------------------------------------- the matrix

/** Steps a candidate passes at least once in the screen. */
export function passingSteps(row: Pick<MatrixRow, "cells">): number {
  return Object.values(row.cells).filter((c) => c.state === "run" && c.successes > 0).length;
}

/** Sum of the success rates of the steps it was screened on. */
function rateSum(row: Pick<MatrixRow, "cells">): number {
  return Object.values(row.cells).reduce((sum, c) => sum + (c.state === "run" ? c.rate : 0), 0);
}

/**
 * Rows with the most passing steps first, then the higher summed success rate; ties keep the
 * export's order (a stable sort).
 */
export function orderRows<T extends Pick<MatrixRow, "cells">>(rows: readonly T[]): T[] {
  return rows
    .map((row, i) => ({ row, i, p: passingSteps(row), r: rateSum(row) }))
    .sort((a, b) => b.p - a.p || b.r - a.r || a.i - b.i)
    .map((x) => x.row);
}

/** Pruned reasons of the matrix as numbered footnotes, in order of first appearance. */
export function prunedNotes(matrix: Pick<Matrix, "rows" | "steps">): Map<string, number> {
  const notes = new Map<string, number>();
  for (const row of matrix.rows) {
    for (const id of matrix.steps) {
      const c = row.cells[id];
      const reason = c && c.state !== "run" ? c.reason : null;
      if (reason && !notes.has(reason)) notes.set(reason, notes.size + 1);
    }
  }
  return notes;
}

/** Finalists with the best single appended when the export lists it on its own. */
export function finalistsWithBaseline(page: Pick<HeroPage, "finalists" | "best_single">): FinalistRow[] {
  const out = [...page.finalists];
  if (page.best_single && !out.some((f) => f.setup_id === page.best_single!.setup_id)) out.push(page.best_single);
  return out;
}

export function roleText(roles: readonly Role[]): string {
  const shown = roles.filter((r) => r !== "finalist");
  return shown.length ? shown.map((r) => ROLE_LABELS[r]).join(" · ") : ROLE_LABELS.finalist;
}
