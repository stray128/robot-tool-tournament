/**
 * The effector tournament's web data (product mode, not part of the pre-registered experiment).
 *
 * Producer: `python -m rtt.effectors export-web` (src/rtt/effectors/lanes.py), which writes
 * public/data/effectors/index.json, public/data/effectors/<part>.json and the replay v2 lanes
 * under public/data/replays/effectors-<part>/. This module holds the types, the parsers (a
 * malformed file fails `pnpm build`) and the labels; it imports nothing from Node, so client
 * components can use it. The build-time loaders live in app/effectors/data.ts.
 */

import { DataError, Node } from "./validate";

/** Shown above every effector result, word for word (rtt.effectors.PRODUCT_MODE_LABEL). */
export const PRODUCT_MODE_LABEL = "Product mode: not part of the pre-registered experiment";
export const INDEX_SCHEMA = "rtt-web-effectors";
export const PAGE_SCHEMA = "rtt-web-effector-part";
/** Per-lane download budget, the hero payload budget (lib/assets.ts). */
export const LANE_BUDGET_BYTES = 300_000;

/** Every effector family (rtt.effectors.types.EffectorFamily). */
export const ALL_FAMILIES = ["parallel_jaw", "catalog_hand", "catalog_gripper", "soft_arm", "suction", "designed", "process_tool"] as const;
export type AnyFamily = (typeof ALL_FAMILIES)[number];
/**
 * The families of the per-part tournament, its lanes and its pages (rtt.effectors.tournament
 * PART_FAMILIES): every family but the process tools (nut runner, tamper), which only the Studio's
 * process steps use. The per-part parsers accept these only.
 */
export const FAMILIES = ["parallel_jaw", "catalog_hand", "catalog_gripper", "soft_arm", "suction", "designed"] as const;
export const PART_FAMILIES = FAMILIES;
export type Family = (typeof FAMILIES)[number];
/** Families that never enter a per-part pick: only Studio pages show them. */
export const STUDIO_ONLY_FAMILIES: readonly AnyFamily[] = ["process_tool"];
/** Per-part budget groups (rtt.effectors.tournament GROUP_ORDER: the process group is Studio-only). */
export const GROUPS = ["reference", "catalog", "soft_arm", "suction", "designed", "random"] as const;
export type Group = (typeof GROUPS)[number];

/**
 * Whether a suction preset is a vacuum area frame (its model, rtt.effectors.suction): an area tool
 * for a large flat part, kept out of the per-part tournament (per_part_presets) and shown only on
 * Studio pages.
 */
export function isVacuumFrame(model: string): boolean {
  return /^vacuum-frame-/.test(model);
}

/** Whether a tool is Studio-only: a process tool, or a vacuum area frame. */
export function isStudioOnlyTool(family: string, model: string): boolean {
  return (STUDIO_ONLY_FAMILIES as readonly string[]).includes(family) || (family === "suction" && isVacuumFrame(model));
}
export const FIDELITIES = ["core", "rigid", "approximation"] as const;
export type Fidelity = (typeof FIDELITIES)[number];
export type Requirement = "held" | "registered";

export const FAMILY_TITLES: Record<Family, string> = {
  parallel_jaw: "Parallel jaw",
  catalog_hand: "Multi-finger hand",
  catalog_gripper: "Industrial gripper",
  soft_arm: "Soft arm",
  suction: "Suction cup",
  designed: "Designed morphology",
};

/** Every family's title, the Studio-only process tools included (rtt.effectors.lanes FAMILY_TITLES). */
export const ALL_FAMILY_TITLES: Record<AnyFamily, string> = { ...FAMILY_TITLES, process_tool: "Process tool" };

export const FIDELITY_LABELS: Record<Fidelity, string> = {
  core: "Core",
  rigid: "Rigid model",
  approximation: "Approximation",
};

export const FIDELITY_NOTES: Record<Fidelity, string> = {
  core: "The core experiment's own scene, checks and scorer.",
  rigid: "A rigid multibody model driven by a scripted grasp.",
  approximation: "A labelled approximation of a soft or vacuum effector: verify on hardware.",
};

export const REQUIREMENT_LABELS: Record<Requirement, string> = {
  registered: "Registered",
  held: "Held",
};

export const REQUIREMENT_NOTES: Record<Requirement, string> = {
  registered: "lifted, held 2 s and back in the same pose (within 0.75 mm, 2 deg) whatever the placement error",
  held: "lifted 100 mm and held 2 s",
};

/** Scorer outcomes in effector words (the jaw-centric labels of lib/labels.ts say "jaws" and "seat"). */
export const OUTCOME_LABELS: Record<string, string> = {
  missed_grasp: "Missed grasp",
  slip_during_lift: "Slipped during lift",
  ejected_on_close: "Pushed away on close",
  unstable_hold: "Dropped during hold",
  finger_env_collision: "Hit the table",
  not_registered: "Held, not registered",
  physics_invalid: "Physics reset",
  physics_unreliable: "Penetration over 1 mm",
  invalid_build: "Cannot be built",
};

export const FLAG_LABELS: Record<string, string> = {
  off_centre: "off centre",
  yawed: "turned",
  tilted: "tilted",
  not_upright: "not upright",
  jam_on_close: "jaw short of its seat",
  few_contacts: "too few contacts",
  no_opposition: "contacts do not oppose",
  no_seal: "no seal",
  no_wrap: "no wrap",
};

export const ROLE_LABELS: Record<string, string> = {
  recommended: "Recommended",
  runner_up: "Rigid runner-up",
  recommended_held: "Best for held",
  recommended_registered: "Best for registered",
  finalist: "Finalist",
};

export function roleLabel(role: string): string {
  if (role.startsWith("best_")) return `Best ${GROUP_TITLES[role.slice(5) as Group] ?? role.slice(5)}`;
  return ROLE_LABELS[role] ?? role;
}

export const GROUP_TITLES: Record<Group, string> = {
  reference: "Reference",
  catalog: "Catalog",
  soft_arm: "Soft arm",
  suction: "Suction",
  designed: "Designed (Nemotron)",
  random: "Random control",
};

export function outcomeLabel(registered: boolean, held: boolean, mode: string | null): string {
  if (registered) return "Registered";
  if (mode) return OUTCOME_LABELS[mode] ?? mode.replace(/_/g, " ");
  return held ? "Held" : "Failed";
}

// ---------------------------------------------------------------- types

export interface EffectorBrief {
  effector_id: string;
  display_name: string;
  family: Family;
  model: string;
  primitive: string;
  origin: string;
  group: Group;
  fidelity: Fidelity;
  hypothesis: string | null;
  /** A note every card, row and lane naming this effector prints (e.g. low grip at published settings). */
  note: string | null;
}

export interface CertTest {
  name: string;
  category: "placement" | "shake" | "push";
  held: boolean;
  registered: boolean;
  failure_mode: string | null;
  registration_flags: string[];
  offset_mm: number | null;
}

export type Tolerance = { x: number | null; y: number | null };

export interface CertSummary {
  reference: string;
  held_score: number;
  registered_score: number;
  held_scores: Record<string, number>;
  registered_scores: Record<string, number>;
  held_tolerance_mm: Tolerance;
  registered_tolerance_mm: Tolerance;
  held: string;
  registered: string;
  effector_mass_kg: number | null;
  tests: CertTest[];
}

export interface ScreenRates {
  n: number;
  held: number;
  registered: number;
}

export interface Card {
  effector: EffectorBrief;
  certificate: CertSummary | null;
  screen: ScreenRates;
  why: string | null;
  score: number | null;
  tolerance_mm: Tolerance | null;
  n_actuators: number | null;
  mass_kg: number | null;
  actuation: Record<string, number>;
  caveat: string | null;
  basis?: string;
}

export interface RunnerUp {
  card: Card | null;
  basis: string | null;
}

export interface RankingRow {
  effector_id: string;
  display_name: string;
  family: Family;
  group: Group;
  fidelity: Fidelity;
  status: string;
  certified: boolean;
  eligible: boolean;
  reason: string | null;
  score: number | null;
  tolerance_mm: Tolerance;
  screen_rate: number | null;
  n_actuators: number | null;
  mass_kg: number | null;
  actuation: Record<string, number>;
  note: string | null;
}

export interface NoWinner {
  reason: string;
  closest: { effector_id: string; display_name: string; family: Family; fidelity: Fidelity; score: number | null; screen_rate: number | null; reason: string | null; failure_modes: Record<string, number> }[];
}

export interface RequirementBlock {
  winner: Card | null;
  runner_up: RunnerUp | null;
  no_winner: NoWinner | null;
  ranking: RankingRow[];
}

export interface FamilyCard {
  family: Family;
  title: string;
  blurb: string;
  fidelity: Fidelity;
  available: boolean;
  status: string;
  candidates: number;
  screened: number;
  invalid: number;
  best: { effector_id: string; display_name: string; screen: ScreenRates | null; certified: boolean; note: string | null } | null;
  caveat: string;
  label: string;
}

export interface GroupBudget {
  candidates: number;
  buildable: number;
  screened: number;
  screen_episodes: number;
  reference_episodes: number;
  certificates: number;
  certificate_episodes: number;
  max_screen_episodes: number;
  max_certificates: number;
}

export interface GroupRow {
  group: Group;
  title: string;
  status: "run" | "empty" | "skipped";
  reason: string | null;
  /** How the group was drawn when that is not the plain rule (the matched random control). */
  note: string | null;
  finalists: string[];
  budget: GroupBudget;
  sampled_from: number;
}

export interface CandidateRow {
  effector_id: string;
  display_name: string;
  family: Family;
  model: string;
  group: Group;
  fidelity: Fidelity;
  status: string;
  reason: string | null;
  n_actuators: number | null;
  mass_kg: number | null;
  finalist: boolean;
  certified: boolean;
  screen: (ScreenRates & { failure_modes: Record<string, number> }) | null;
  note: string | null;
}

export interface LaneOutcome {
  registered: boolean;
  held: boolean;
  engaged: boolean;
  failure_mode: string | null;
  registration_flags: string[];
  max_lift_mm: number;
  hold_s: number;
  penetration_mm: number | null;
  sim_steps: number;
  traj_sha: string;
}

export interface EffectorLane {
  id: string;
  /** Replay stem relative to /data, e.g. "replays/effectors-block/recommended-0123456789ab". */
  replay: string;
  roles: string[];
  effector: EffectorBrief;
  condition_id: string;
  condition: string;
  seed: string;
  fps: number;
  n_frames: number;
  duration_s: number;
  bytes: number;
  reference: string;
  outcome: LaneOutcome;
}

export interface EffectorTask {
  name: string;
  phases: { name: "settle" | "descend" | "close" | "lift" | "hold"; start_s: number; end_s: number | null }[];
  success_lift_m: number;
  success_hold_s: number;
  record_tail_s: number;
  max_time_s: number;
}

export interface ScreenConditionDoc {
  condition_id: string;
  label: string;
  placement: { err_x_mm: number; err_y_mm: number; err_yaw_deg: number };
  seeds: string[];
}

export interface EffectorPage {
  schema: typeof PAGE_SCHEMA;
  version: 1;
  label: typeof PRODUCT_MODE_LABEL;
  part: { part_id: string; title: string; family: string; size_mm: number[]; mass_kg: number };
  requirement: Requirement;
  recommendation: RequirementBlock;
  by_requirement: Record<Requirement, RequirementBlock>;
  alternatives: Partial<Record<Requirement, { effector_id: string; display_name: string; fidelity: Fidelity } | null>>;
  families: FamilyCard[];
  groups: GroupRow[];
  candidates: CandidateRow[];
  certificates: Record<string, CertSummary>;
  lanes: EffectorLane[];
  task: EffectorTask;
  conditions: ScreenConditionDoc[];
  fairness: { identical: boolean; problems: string[]; signature_sha256: string | null; conditions: string[] };
  physics: { episodes: number; certificate_episodes: number; bad_qacc: number; physics_invalid: number; physics_unreliable: number; held_episodes: number };
  caveats: Record<string, string>;
  provenance: { code_version: string; task_digest: string; engine: string; created_utc: string; run_id: string };
}

export interface IndexWinner {
  effector_id: string;
  display_name: string;
  family: Family;
  fidelity: Fidelity;
  score: number | null;
  tolerance_mm: Tolerance | null;
  caveat: string | null;
}

export interface EffectorIndexPart {
  part_id: string;
  title: string;
  family: string;
  size_mm: number[];
  page: string;
  requirement: Requirement;
  winner: IndexWinner | null;
  by_requirement: Record<Requirement, IndexWinner | null>;
  groups_run: Group[];
  lanes: number;
  candidates: number;
  episodes: number;
  code_version: string;
}

export interface EffectorIndex {
  schema: typeof INDEX_SCHEMA;
  version: 1;
  label: typeof PRODUCT_MODE_LABEL;
  generated_utc: string;
  families: { family: Family; title: string; fidelity: Fidelity; available: boolean }[];
  parts: EffectorIndexPart[];
}

// ---------------------------------------------------------------- parsers

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;
const SHA256 = /^[0-9a-f]{64}$/;

function slug(n: Node): string {
  const s = n.nonEmpty();
  if (!SLUG.test(s)) n.fail(`expected a URL-safe id, got ${JSON.stringify(s)}`);
  return s;
}

function sha(n: Node): string {
  const s = n.str();
  if (!SHA256.test(s)) n.fail("expected a sha256");
  return s;
}

function optNum(n: Node): number | null {
  return n.nullable((x) => x.num());
}

function optStr(n: Node): string | null {
  return n.nullable((x) => x.str());
}

function numMap(n: Node): Record<string, number> {
  return Object.fromEntries(n.entries().map(([k, v]) => [k, v.num()]));
}

function label(n: Node): typeof PRODUCT_MODE_LABEL {
  return n.literal(PRODUCT_MODE_LABEL);
}

function tolerance(n: Node): Tolerance {
  return { x: optNum(n.get("x")), y: optNum(n.get("y")) };
}

/** The optional card note (absent in data exported before notes existed). */
function optNote(n: Node): string | null {
  return n.has("note") ? optStr(n.get("note")) : null;
}

/** A family on a per-part page (PART_FAMILIES: never a process tool). */
function partFamily(n: Node): Family {
  if (n.value === "process_tool") n.fail("a process tool is Studio-only: it is never in a per-part tournament (rtt.effectors.tournament.PART_FAMILIES)");
  return n.oneOf(FAMILIES);
}

/** No vacuum area frame on a per-part page (rtt.effectors.tournament.per_part_presets). */
function notAFrame(n: Node, model: string, name: string): void {
  if (isVacuumFrame(model) || /^Vacuum area frame\b/.test(name)) n.fail("a vacuum area frame is Studio-only: it is never in a per-part tournament (rtt.effectors.tournament.per_part_presets)");
}

function readBrief(n: Node): EffectorBrief {
  const model = slug(n.get("model"));
  const display_name = n.get("display_name").nonEmpty();
  notAFrame(n.get("model"), model, display_name);
  return {
    effector_id: sha(n.get("effector_id")),
    display_name,
    family: partFamily(n.get("family")),
    model,
    primitive: slug(n.get("primitive")),
    origin: n.get("origin").nonEmpty(),
    group: n.get("group").oneOf(GROUPS),
    fidelity: n.get("fidelity").oneOf(FIDELITIES),
    hypothesis: n.has("hypothesis") ? optStr(n.get("hypothesis")) : null,
    note: optNote(n),
  };
}

function readCert(n: Node): CertSummary {
  return {
    reference: n.get("reference").oneOf(["analytic", "nominal_episode", "none"]),
    held_score: n.get("held_score").unit(),
    registered_score: n.get("registered_score").unit(),
    held_scores: numMap(n.get("held_scores")),
    registered_scores: numMap(n.get("registered_scores")),
    held_tolerance_mm: tolerance(n.get("held_tolerance_mm")),
    registered_tolerance_mm: tolerance(n.get("registered_tolerance_mm")),
    held: n.get("held").str(),
    registered: n.get("registered").str(),
    effector_mass_kg: n.has("effector_mass_kg") ? optNum(n.get("effector_mass_kg")) : null,
    tests: n
      .get("tests")
      .items()
      .map((t) => ({
        name: t.get("name").nonEmpty(),
        category: t.get("category").oneOf(["placement", "shake", "push"] as const),
        held: t.get("held").bool(),
        registered: t.get("registered").bool(),
        failure_mode: optStr(t.get("failure_mode")),
        registration_flags: t.get("registration_flags").strs(),
        offset_mm: optNum(t.get("offset_mm")),
      })),
  };
}

function readScreen(n: Node): ScreenRates {
  return { n: n.get("n").int(), held: n.get("held").unit(), registered: n.get("registered").unit() };
}

function readCard(n: Node): Card {
  return {
    effector: readBrief(n.get("effector")),
    certificate: n.get("certificate").nullable(readCert),
    screen: readScreen(n.get("screen")),
    why: optStr(n.get("why")),
    score: optNum(n.get("score")),
    tolerance_mm: n.get("tolerance_mm").nullable(tolerance),
    n_actuators: n.get("n_actuators").nullable((x) => x.int()),
    mass_kg: optNum(n.get("mass_kg")),
    actuation: numMap(n.get("actuation")),
    caveat: optStr(n.get("caveat")),
    ...(n.has("basis") ? { basis: n.get("basis").str() } : {}),
  };
}

function readRunnerUp(n: Node): RunnerUp | null {
  if (n.value === null || n.value === undefined) return null;
  const basis = n.has("basis") ? optStr(n.get("basis")) : null;
  const card = n.get("effector").value === null ? null : readCard(n);
  return { card, basis };
}

function readRow(n: Node): RankingRow {
  const display_name = n.get("display_name").nonEmpty();
  notAFrame(n.get("display_name"), "", display_name);
  return {
    effector_id: sha(n.get("effector_id")),
    display_name,
    family: partFamily(n.get("family")),
    group: n.get("group").oneOf(GROUPS),
    fidelity: n.get("fidelity").oneOf(FIDELITIES),
    status: n.get("status").nonEmpty(),
    certified: n.get("certified").bool(),
    eligible: n.get("eligible").bool(),
    reason: optStr(n.get("reason")),
    score: optNum(n.get("score")),
    tolerance_mm: tolerance(n.get("tolerance_mm")),
    screen_rate: optNum(n.get("screen_rate")),
    n_actuators: n.get("n_actuators").nullable((x) => x.int()),
    mass_kg: optNum(n.get("mass_kg")),
    actuation: numMap(n.get("actuation")),
    note: optNote(n),
  };
}

function readBlock(n: Node): RequirementBlock {
  const nw = n.get("no_winner");
  const block: RequirementBlock = {
    winner: n.get("winner").nullable(readCard),
    runner_up: readRunnerUp(n.get("runner_up_rigid")),
    no_winner: nw.nullable((x) => ({
      reason: x.get("reason").nonEmpty(),
      closest: x
        .get("closest")
        .items()
        .map((c) => ({
          effector_id: sha(c.get("effector_id")),
          display_name: c.get("display_name").nonEmpty(),
          family: partFamily(c.get("family")),
          fidelity: c.get("fidelity").oneOf(FIDELITIES),
          score: optNum(c.get("score")),
          screen_rate: optNum(c.get("screen_rate")),
          reason: optStr(c.get("reason")),
          failure_modes: numMap(c.get("failure_modes")),
        })),
    })),
    ranking: n.get("ranking").items().map(readRow),
  };
  if (block.winner === null && block.no_winner === null) n.fail("a requirement block needs a winner or a no-winner report");
  if (block.winner?.effector.fidelity === "approximation" && block.runner_up === null) {
    n.get("runner_up_rigid").fail("an approximation winner must carry a rigid runner-up");
  }
  return block;
}

function readLane(n: Node, partId: string): EffectorLane {
  const replay = n.get("replay").nonEmpty();
  if (!replay.startsWith(`replays/effectors-${partId}/`) || replay.includes("..")) {
    n.get("replay").fail(`expected a lane under replays/effectors-${partId}/`);
  }
  const bytes = n.get("bytes").int();
  if (bytes > LANE_BUDGET_BYTES) n.get("bytes").fail(`a lane may download at most ${LANE_BUDGET_BYTES} bytes, this one ${bytes}`);
  const o = n.get("outcome");
  return {
    id: slug(n.get("id")),
    replay,
    roles: n.get("roles").strs(),
    effector: readBrief(n.get("effector")),
    condition_id: n.get("condition_id").nonEmpty(),
    condition: n.get("condition").nonEmpty(),
    seed: n.get("seed").nonEmpty(),
    fps: n.get("fps").int(),
    n_frames: n.get("n_frames").int(),
    duration_s: n.get("duration_s").nonNeg(),
    bytes,
    reference: n.get("reference").nonEmpty(),
    outcome: {
      registered: o.get("registered").bool(),
      held: o.get("held").bool(),
      engaged: o.get("engaged").bool(),
      failure_mode: optStr(o.get("failure_mode")),
      registration_flags: o.get("registration_flags").strs(),
      max_lift_mm: o.get("max_lift_mm").num(),
      hold_s: o.get("hold_s").nonNeg(),
      penetration_mm: optNum(o.get("penetration_mm")),
      sim_steps: o.get("sim_steps").int(),
      traj_sha: sha(o.get("traj_sha")),
    },
  };
}

/** Validate public/data/effectors/<part>.json. */
export function parseEffectorPage(raw: unknown, file = "effectors/part.json"): EffectorPage {
  const root = new Node(file, "$", raw);
  root.get("schema").literal(PAGE_SCHEMA);
  root.get("version").literal(1);
  const part = root.get("part");
  const partId = slug(part.get("part_id"));
  const requirement = root.get("requirement").oneOf(["held", "registered"] as const);
  const by = root.get("by_requirement");
  const page: EffectorPage = {
    schema: PAGE_SCHEMA,
    version: 1,
    label: label(root.get("label")),
    part: {
      part_id: partId,
      title: part.get("title").nonEmpty(),
      family: part.get("family").nonEmpty(),
      size_mm: part.get("size_mm").nums(3),
      mass_kg: part.get("mass_kg").nonNeg(),
    },
    requirement,
    recommendation: readBlock(root.get("recommendation")),
    by_requirement: { held: readBlock(by.get("held")), registered: readBlock(by.get("registered")) },
    alternatives: Object.fromEntries(
      root
        .get("alternatives")
        .entries()
        .map(([k, v]) => [
          k,
          v.nullable((x) => ({
            effector_id: sha(x.get("effector_id")),
            display_name: x.get("display_name").nonEmpty(),
            fidelity: x.get("fidelity").oneOf(FIDELITIES),
          })),
        ]),
    ),
    families: root
      .get("families")
      .items()
      .map((f) => ({
        family: partFamily(f.get("family")),
        title: f.get("title").nonEmpty(),
        blurb: f.get("blurb").nonEmpty(),
        fidelity: f.get("fidelity").oneOf(FIDELITIES),
        available: f.get("available").bool(),
        status: f.get("status").nonEmpty(),
        candidates: f.get("candidates").int(),
        screened: f.get("screened").int(),
        invalid: f.get("invalid").int(),
        best: f.get("best").nullable((b) => ({
          effector_id: sha(b.get("effector_id")),
          display_name: b.get("display_name").nonEmpty(),
          screen: b.get("screen").nullable(readScreen),
          certified: b.get("certified").bool(),
          note: optNote(b),
        })),
        caveat: f.get("caveat").nonEmpty(),
        label: f.get("label").literal(PRODUCT_MODE_LABEL),
      })),
    groups: root
      .get("groups")
      .items()
      .map((g) => {
        const b = g.get("budget");
        const budget: GroupBudget = {
          candidates: b.get("candidates").int(),
          buildable: b.get("buildable").int(),
          screened: b.get("screened").int(),
          screen_episodes: b.get("screen_episodes").int(),
          reference_episodes: b.get("reference_episodes").int(),
          certificates: b.get("certificates").int(),
          certificate_episodes: b.get("certificate_episodes").int(),
          max_screen_episodes: b.get("max_screen_episodes").int(),
          max_certificates: b.get("max_certificates").int(),
        };
        if (budget.screen_episodes > budget.max_screen_episodes || budget.certificates > budget.max_certificates) {
          b.fail("the group spent more than its budget");
        }
        return {
          group: g.get("group").oneOf(GROUPS),
          title: g.get("title").nonEmpty(),
          status: g.get("status").oneOf(["run", "empty", "skipped"] as const),
          reason: optStr(g.get("reason")),
          note: optNote(g),
          finalists: g.get("finalists").items().map(sha),
          budget,
          sampled_from: g.get("sampled_from").int(),
        };
      }),
    candidates: root
      .get("candidates")
      .items()
      .map((c) => {
        const model = slug(c.get("model"));
        const display_name = c.get("display_name").nonEmpty();
        notAFrame(c.get("model"), model, display_name);
        return {
        effector_id: sha(c.get("effector_id")),
        display_name,
        family: partFamily(c.get("family")),
        model,
        group: c.get("group").oneOf(GROUPS),
        fidelity: c.get("fidelity").oneOf(FIDELITIES),
        status: c.get("status").nonEmpty(),
        reason: optStr(c.get("reason")),
        n_actuators: c.get("n_actuators").nullable((x) => x.int()),
        mass_kg: optNum(c.get("mass_kg")),
        finalist: c.get("finalist").bool(),
        certified: c.get("certified").bool(),
        screen: c.has("screen")
          ? c.get("screen").nullable((s) => ({ ...readScreen(s), failure_modes: numMap(s.get("failure_modes")) }))
          : null,
        note: optNote(c),
        };
      }),
    certificates: Object.fromEntries(
      root
        .get("certificates")
        .entries()
        .map(([k, v]) => [k, readCert(v)]),
    ),
    lanes: root
      .get("lanes")
      .items()
      .map((l) => readLane(l, partId)),
    task: {
      name: root.get("task").get("name").nonEmpty(),
      phases: root
        .get("task")
        .get("phases")
        .items()
        .map((p) => ({
          name: p.get("name").oneOf(["settle", "descend", "close", "lift", "hold"] as const),
          start_s: p.get("start_s").nonNeg(),
          end_s: optNum(p.get("end_s")),
        })),
      success_lift_m: root.get("task").get("success_lift_m").num(),
      success_hold_s: root.get("task").get("success_hold_s").num(),
      record_tail_s: root.get("task").get("record_tail_s").num(),
      max_time_s: root.get("task").get("max_time_s").num(),
    },
    conditions: root
      .get("conditions")
      .items()
      .map((c) => ({
        condition_id: c.get("condition_id").nonEmpty(),
        label: c.get("label").nonEmpty(),
        placement: {
          err_x_mm: c.get("placement").get("err_x_mm").num(),
          err_y_mm: c.get("placement").get("err_y_mm").num(),
          err_yaw_deg: c.get("placement").get("err_yaw_deg").num(),
        },
        seeds: c.get("seeds").strs(),
      })),
    fairness: {
      identical: root.get("fairness").get("identical").bool(),
      problems: root.get("fairness").get("problems").strs(),
      signature_sha256: optStr(root.get("fairness").get("signature_sha256")),
      conditions: root.get("fairness").get("conditions").strs(),
    },
    physics: {
      episodes: root.get("physics").get("episodes").int(),
      certificate_episodes: root.get("physics").get("certificate_episodes").int(),
      bad_qacc: root.get("physics").get("bad_qacc").int(),
      physics_invalid: root.get("physics").get("physics_invalid").int(),
      physics_unreliable: root.get("physics").get("physics_unreliable").int(),
      held_episodes: root.get("physics").get("held_episodes").int(),
    },
    caveats: Object.fromEntries(
      root
        .get("caveats")
        .entries()
        .map(([k, v]) => [k, v.nonEmpty()]),
    ),
    provenance: {
      code_version: root.get("provenance").get("code_version").nonEmpty(),
      task_digest: root.get("provenance").get("task_digest").nonEmpty(),
      engine: root.get("provenance").get("engine").nonEmpty(),
      created_utc: root.get("provenance").get("created_utc").isoTime(),
      run_id: root.get("provenance").get("run_id").nonEmpty(),
    },
  };
  if (page.recommendation.winner?.effector.effector_id !== page.by_requirement[requirement].winner?.effector.effector_id) {
    root.get("recommendation").fail("the headline recommendation differs from its requirement's block");
  }
  if (page.lanes.length === 0) root.get("lanes").fail("a part page needs at least one lane");
  const ids = new Set<string>();
  for (const lane of page.lanes) {
    if (ids.has(lane.id)) root.get("lanes").fail(`duplicate lane id ${lane.id}`);
    ids.add(lane.id);
  }
  return page;
}

function readWinner(n: Node): IndexWinner | null {
  return n.nullable((w) => ({
    effector_id: sha(w.get("effector_id")),
    display_name: w.get("display_name").nonEmpty(),
    family: partFamily(w.get("family")),
    fidelity: w.get("fidelity").oneOf(FIDELITIES),
    score: optNum(w.get("score")),
    tolerance_mm: w.get("tolerance_mm").nullable(tolerance),
    caveat: optStr(w.get("caveat")),
  }));
}

/** Validate public/data/effectors/index.json. */
export function parseEffectorIndex(raw: unknown, file = "effectors/index.json"): EffectorIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal(INDEX_SCHEMA);
  root.get("version").literal(1);
  const parts = root
    .get("parts")
    .items()
    .map((p) => ({
      part_id: slug(p.get("part_id")),
      title: p.get("title").nonEmpty(),
      family: p.get("family").nonEmpty(),
      size_mm: p.get("size_mm").nums(3),
      page: p.get("page").nonEmpty(),
      requirement: p.get("requirement").oneOf(["held", "registered"] as const),
      winner: readWinner(p.get("winner")),
      by_requirement: {
        held: readWinner(p.get("by_requirement").get("held")),
        registered: readWinner(p.get("by_requirement").get("registered")),
      },
      groups_run: p.get("groups_run").items().map((g) => g.oneOf(GROUPS)),
      lanes: p.get("lanes").int(),
      candidates: p.get("candidates").int(),
      episodes: p.get("episodes").int(),
      code_version: p.get("code_version").nonEmpty(),
    }));
  const seen = new Set<string>();
  for (const p of parts) {
    if (seen.has(p.part_id)) throw new DataError(file, "$.parts", `duplicate part ${p.part_id}`);
    seen.add(p.part_id);
    if (p.page !== `effectors/${p.part_id}.json`) throw new DataError(file, "$.parts", `part ${p.part_id} points at ${p.page}`);
  }
  return {
    schema: INDEX_SCHEMA,
    version: 1,
    label: label(root.get("label")),
    generated_utc: root.get("generated_utc").isoTime(),
    families: root
      .get("families")
      .items()
      .map((f) => ({
        family: partFamily(f.get("family")),
        title: f.get("title").nonEmpty(),
        fidelity: f.get("fidelity").oneOf(FIDELITIES),
        available: f.get("available").bool(),
      })),
    parts,
  };
}

// ---------------------------------------------------------------- formatting

/**
 * A part's title as a noun inside a sentence: "Hex standoff" -> "hex standoff", while a
 * leading letter name or initialism keeps its case ("L-bracket", "U bolt", "PCB mount").
 */
export function partNoun(title: string): string {
  const first = title.split(/\s/)[0] ?? "";
  if (/^[A-Z](?:$|[-A-Z0-9])/.test(first)) return title;
  return title.charAt(0).toLowerCase() + title.slice(1);
}

export function pct(v: number | null | undefined, digits = 0): string {
  return v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(digits)}%`;
}

export function score(v: number | null | undefined): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(2);
}

/** "3 mm on both axes", "0 mm across, 3 mm along" or "none" (the nominal grasp fails). */
export function toleranceText(t: Tolerance | null | undefined): string {
  if (!t || t.x === null || t.y === null) return "none";
  const mm = (v: number) => (v === 0 ? "0 mm" : `±${v} mm`);
  if (t.x === t.y) return `${mm(t.x)} both axes`;
  return `${mm(t.x)} across, ${mm(t.y)} along`;
}

export function massText(kg: number | null | undefined): string {
  return kg === null || kg === undefined ? "n/a" : `${(1000 * kg).toFixed(0)} g`;
}

const ACTUATION_LABELS: Record<string, [string, string]> = {
  grip_force_n: ["Grip force", "N"],
  grip_force_n_per_pad: ["Force per pad", "N"],
  joint_torque_nm_max: ["Joint torque", "N m"],
  cable_tension_n: ["Cable tension", "N"],
  vacuum_kpa: ["Vacuum", "kPa"],
  holding_force_n: ["Holding force", "N"],
};

/** Declared actuation limits in words, e.g. "Grip force 50 N". */
export function actuationText(a: Record<string, number>): string[] {
  return Object.entries(a).map(([k, v]) => {
    const [name, unit] = ACTUATION_LABELS[k] ?? [k.replace(/_/g, " "), ""];
    return `${name} ${Number(v.toFixed(3))}${unit ? ` ${unit}` : ""}`;
  });
}

/** The lane order shown first: the recommendation, the rigid runner-up, then one lane per group. */
export function defaultLanes(lanes: EffectorLane[]): EffectorLane[] {
  const shown = lanes.filter((l) => l.roles.some((r) => r !== "finalist"));
  return shown.length > 0 ? shown : lanes.slice(0, 1);
}
