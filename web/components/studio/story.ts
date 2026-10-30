/**
 * The story of one Studio use case, derived from its exported page (components/studio/schema.ts)
 * and nothing else: the one-sentence answer, and per step how many tools could do it, which tool
 * the recommendation uses and why, and why the others failed. Every number here is counted from
 * the page's success matrix, candidates, finalists, recommendation and lanes; the wording around
 * the numbers is fixed per failure mode and per pruning rule.
 *
 * Pure and Node-free, so pages and client components can both use it.
 */

import {
  failureLabel,
  type FinalistRow,
  type HeroPage,
  humanize,
  type SetupKind,
  type ToolBrief,
  type LaneStep,
  type MatrixCell,
  type MatrixRow,
  type Plan,
  type PlanStep,
  type RecStep,
  type Recommendation,
  type StudioLane,
  type Tier,
  TIER_NOTES,
  VERIFY_CAVEAT,
} from "./schema";
import { caveatNote, runNotes } from "@/lib/labels";

// ---------------------------------------------------------------- words

const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

export function numberWord(n: number): string {
  return NUMBER_WORDS[n] ?? String(n);
}

/** "open_lid" -> "open the lid", "tighten_coupling" -> "tighten the coupling". */
export function stepVerb(stepId: string): string {
  const [verb, ...rest] = stepId.split("_").filter(Boolean);
  if (!verb) return stepId;
  return rest.length ? `${verb} the ${rest.join(" ")}` : verb;
}

/** "Open the lid". */
export function stepName(stepId: string): string {
  const v = stepVerb(stepId);
  return v.charAt(0).toUpperCase() + v.slice(1);
}

/** "Soft arm" -> "soft arms", "Suction cup" -> "suction cups", "Multi-finger hand" -> "multi-finger hands". */
export function plural(title: string): string {
  const t = title.charAt(0).toLowerCase() + title.slice(1);
  if (/(s|sh|ch|x)$/.test(t)) return `${t}es`;
  return `${t}s`;
}

function lower(s: string): string {
  return s.charAt(0).toLowerCase() + s.slice(1);
}

export function massShort(kg: number): string {
  return kg < 1 ? `${Math.round(kg * 1000)} g` : `${kg.toFixed(2)} kg`;
}

export function secondsShort(s: number): string {
  return `${s.toFixed(s >= 100 ? 0 : 1)} s`;
}

// ---------------------------------------------------------------- failure words

/** A step's limit in words: "the 20 N limit", "the 8 N m limit", or null. */
export function limitText(step: PlanStep): string | null {
  const c = step.constraints;
  if (c.max_force_n !== null) return `the ${trim(c.max_force_n)} N limit`;
  if (c.max_torque_nm !== null) return `the ${trim(c.max_torque_nm)} N m limit`;
  if (c.precision_mm !== null) return `the ${trim(c.precision_mm)} mm precision`;
  return null;
}

function trim(v: number): string {
  return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2)));
}

/** The handle feature of the step's object, e.g. "lip" or "handle". */
function featureOf(plan: Plan, step: PlanStep): string | null {
  const obj = plan.objects.find((o) => o.name === step.object);
  const f = obj?.params.feature;
  return typeof f === "string" && f ? f.replace(/_/g, " ") : null;
}

/** What a failure mode means on this step, as a short past-tense phrase ("slipped off the lip"). */
export function failurePhrase(mode: string, step: PlanStep, plan: Plan): string {
  switch (mode) {
    case "slip":
      if (step.primitive === "hook_pull") return `slipped off the ${featureOf(plan, step) ?? "handle"}`;
      if (step.primitive === "twist") return "slipped while turning";
      if (step.primitive === "insert") return "slipped while pushing in";
      return "slipped";
    case "constraint_exceeded": {
      const lim = limitText(step);
      return lim ? `pushed over ${lim}` : "went over a limit";
    }
    case "criterion_not_met":
      return "fell short of the goal";
    case "jam":
      return "jammed";
    case "dropped":
      return "dropped the part";
    case "physics_unreliable":
      return "made the physics unreliable (parts overlapped by over 1 mm)";
    case "physics_invalid":
      return "made the physics reset";
    case "not_simulated":
      return "can't be simulated yet";
    case "not_reached":
      return "never reached this step";
    case "out_of_reach":
      return "was out of the arm's reach";
    case "invalid_build":
      return "could not be built for this task";
    default:
      return failureLabel(mode);
  }
}

/**
 * Pruning reasons in short words, by a phrase of the reason (the exporter's REASON_GROUPS in
 * scripts/export_studio_web.py, in the same order: the first phrase found wins). The exporter now
 * writes the checks' codes in plain words (plain_reason: "too narrow:", "digits collide:"); the raw
 * codes stay for pages exported before it did.
 */
const RULED_OUT_GROUPS: readonly [string, string][] = [
  ["closing axis; the gripper opens", "opens narrower than the part"],
  ["too_narrow", "digits stand too close for the part"],
  ["too narrow: open", "digits stand too close for the part"],
  ["capability tags do not include", "not built for this kind of step"],
  ["no hook or fingers", "a cup has nothing to hook with"],
  ["cannot pick or hold a part", "can't pick or hold a part"],
  ["digit_collision", "digits collide when they close"],
  ["digits collide:", "digits collide when they close"],
  ["cannot hold a member stiffly", "a soft arm can't hold it stiffly"],
  ["cup cannot seal", "no closed face for a cup to seal on"],
  ["cup is wider than", "cup is wider than the face"],
  ["the arm hangs only", "the part is wider than the arm reaches"],
  ["no stiff face to push with", "a slack arm has no stiff face to push with"],
  ["the arm carries at most", "heavier than the arm can carry"],
  ["shaped and sized for", "jaws shaped for a different object"],
  ["pre-shaping around", "opens too little for the feature"],
  ["cannot be built", "can't be built for this task"],
  ["built against the block", "can't be built on the probe part"],
  ["amber licence", "licence not cleared"],
  ["frozen v2 episode", "runs only in the frozen experiment"],
  ["opens", "opens narrower than the object"],
];

export function ruledOutGroup(reason: string | null): string {
  if (!reason) return "not screened on this step";
  for (const [phrase, label] of RULED_OUT_GROUPS) if (reason.includes(phrase)) return label;
  return reason.length > 60 ? `${reason.slice(0, 57)}...` : reason;
}

// ---------------------------------------------------------------- per-step outcomes

export type ToolStatus = "passed" | "partial" | "failed" | "ruled_out";

export interface ToolOutcome {
  effector_id: string;
  display_name: string;
  family_title: string;
  fidelity: Tier | null;
  status: ToolStatus;
  successes: number;
  n: number;
  failure_modes: Record<string, number>;
  /** The failure mode most of its failed runs had (ties: the more severe, in FAILURE_ORDER). */
  dominant: string | null;
  /** Why it was not run on this step (pruned or never screened), in short words. */
  group: string | null;
  reason: string | null;
  /** A Studio-only tool's rating ("rated 40 N ..."), else null. */
  rating: string | null;
}

/** Ties between failure modes go to the one earlier in this list. */
const FAILURE_ORDER = ["constraint_exceeded", "slip", "jam", "dropped", "out_of_reach", "criterion_not_met", "physics_unreliable", "physics_invalid", "not_simulated"];

function dominantMode(modes: Record<string, number>): string | null {
  const list = Object.entries(modes).filter(([, n]) => n > 0);
  if (!list.length) return null;
  const rank = (m: string) => {
    const i = FAILURE_ORDER.indexOf(m);
    return i < 0 ? FAILURE_ORDER.length : i;
  };
  list.sort((a, b) => b[1] - a[1] || rank(a[0]) - rank(b[0]));
  return list[0]![0];
}

function outcomeOf(row: MatrixRow, cell: MatrixCell | undefined, ratings: ReadonlyMap<string, string>): ToolOutcome {
  const base = { effector_id: row.effector_id, display_name: row.display_name, family_title: row.family_title, fidelity: row.fidelity, rating: ratings.get(row.effector_id) ?? null };
  if (!cell || cell.state !== "run") {
    const reason = cell ? cell.reason : null;
    const group = cell?.state === "not_simulated" ? "can't be simulated yet" : ruledOutGroup(reason);
    return { ...base, status: "ruled_out", successes: 0, n: 0, failure_modes: {}, dominant: null, group, reason };
  }
  const status: ToolStatus = cell.n > 0 && cell.successes === cell.n ? "passed" : cell.successes > 0 ? "partial" : "failed";
  return {
    ...base,
    status,
    successes: cell.successes,
    n: cell.n,
    failure_modes: cell.failure_modes,
    dominant: status === "passed" ? null : dominantMode(cell.failure_modes),
    group: null,
    reason: null,
  };
}

export interface BarSegment {
  key: string;
  kind: "passed" | "partial" | "failed" | "ruled_out";
  label: string;
  tools: number;
}

export interface StepStory {
  step: PlanStep;
  index: number;
  name: string;
  /** Every candidate of the use case. */
  n_candidates: number;
  /** Candidates run on this step in physics. */
  tried: number;
  passed: ToolOutcome[];
  partial: ToolOutcome[];
  failed: ToolOutcome[];
  ruledOut: ToolOutcome[];
  /** Screen runs on this step and their failure modes (every tried tool, every condition). */
  runs: number;
  runFailures: Record<string, number>;
  /** The whole field as one bar: passed, failed by dominant mode, ruled out before physics. */
  bar: BarSegment[];
  /** The recommendation's tool on this step. */
  pick: (RecStep & { family_title: string | null; effector_id: string | null; rating: string | null }) | null;
  /** Why that tool, one sentence per reason. */
  why: string[];
  /** What the page data cannot explain about the choice, or null. */
  whyGap: string | null;
  /** The step (index) where this step's tool was chosen, when it is already on the wrist. */
  reusedFrom: number | null;
  /** How the best single off-the-shelf tool did on this step. */
  baseline: { text: string; tone: "pass" | "fail" | "muted" } | null;
  /** The recommended and best-single lanes' run of this step (for the replay seek). */
  laneStep: LaneStep | null;
  baselineLaneStep: LaneStep | null;
  /**
   * The step's caveats in short words: what the shown run itself recorded (a lid that fell shut,
   * a part held above its support, a button too stiff, a target out of reach), then the tool's
   * modelling caveat, or the tier's when the page has no caveat text for it.
   */
  notes: StepNote[];
}

export interface StepNote {
  label: string;
  /** The full caveat, when it says more than the label. */
  detail: string | null;
  kind: "run" | "model";
  /**
   * For a run note: "lane" when the replayed run measured it, "recorded" when it comes from the
   * caveat of a recorded run of the setup (which may be another condition than the replay's).
   */
  source: "lane" | "recorded" | null;
}

/** What a step note is headed with on its card. */
export function noteLead(n: Pick<StepNote, "kind" | "source">): string {
  if (n.kind === "model") return "Caveat: ";
  return n.source === "lane" ? "In this run: " : "In a recorded run: ";
}

/** The notes of one step (StepStory.notes): run notes first, each caveat once. */
export function stepNotes(pick: Pick<RecStep, "caveats" | "fidelity"> | null, laneStep: LaneStep | null): StepNote[] {
  const out: StepNote[] = [];
  const byKey = new Map<string, StepNote>();
  const add = (n: StepNote) => {
    // One note per kind of event: the run's own measurement wins over the caveat's rounded words,
    // and keeps the caveat's full explanation.
    const key = n.kind === "run" ? n.label.split(/[:\d]/)[0]!.trim() : n.label;
    const had = byKey.get(key);
    if (had) {
      if (!had.detail && n.detail) had.detail = n.detail;
      return;
    }
    byKey.set(key, n);
    out.push(n);
  };
  if (laneStep) for (const label of runNotes(laneStep.failure_mode, laneStep.quantities)) add({ label, detail: null, kind: "run", source: "lane" });
  for (const c of pick?.caveats ?? []) {
    const note = caveatNote(c);
    add({ label: note.label, detail: note.detail !== note.label ? note.detail : null, kind: note.kind, source: note.kind === "run" ? "recorded" : null });
  }
  if (pick && pick.fidelity !== "rigid" && !out.some((n) => n.kind === "model")) {
    add({ label: pick.fidelity === "approximation" ? VERIFY_CAVEAT : "not simulated yet: never scored as a pass", detail: TIER_NOTES[pick.fidelity], kind: "model", source: null });
  }
  return out;
}

export interface UseCaseStory {
  answer: string;
  /** Why it takes this kind of setup (more than one tool), or null for a single tool. */
  whySetup: string[];
  steps: StepStory[];
  /** The recommendation's ranking reason, word for word. */
  rankingReason: string | null;
}

/** Candidates that never reached the matrix (pruned from every step), as ruled-out outcomes. */
function unscreened(page: HeroPage): ToolOutcome[] {
  const inMatrix = new Set(page.matrix.rows.map((r) => r.effector_id));
  return page.screen.candidates
    .filter((c) => !inMatrix.has(c.effector_id))
    .map((c) => ({
      effector_id: c.effector_id,
      display_name: c.display_name,
      family_title: c.family_title,
      fidelity: c.fidelity,
      status: "ruled_out" as const,
      successes: 0,
      n: 0,
      failure_modes: {},
      dominant: null,
      reason: c.pruned_reason,
      group: ruledOutGroup(c.pruned_reason ?? c.pruned_steps[0]?.reason ?? null),
      rating: c.rating,
    }));
}

/** Ratings of the Studio-only tools among the candidates, by effector id. */
function ratingsOf(page: HeroPage): Map<string, string> {
  return new Map(page.screen.candidates.filter((c) => c.rating !== null).map((c) => [c.effector_id, c.rating!]));
}

function cellOf(page: HeroPage, effectorId: string, stepId: string): MatrixCell | undefined {
  return page.matrix.rows.find((r) => r.effector_id === effectorId)?.cells[stepId];
}

function fullPass(cell: MatrixCell | undefined): boolean {
  return !!cell && cell.state === "run" && cell.n > 0 && cell.successes === cell.n;
}

/** The finalist row of the recommendation (its complexity), when the export lists it. */
function recRow(page: HeroPage): FinalistRow | null {
  const id = page.recommendation?.setup_id;
  return page.finalists.find((f) => f.setup_id === id) ?? null;
}

function toolIds(row: { tool_details: { effector_id: string | null }[] }): string[] {
  return row.tool_details.map((t) => t.effector_id ?? "");
}

function stepRange(indices: number[]): string {
  const n = [...new Set(indices.map((i) => i + 1))].sort((a, b) => a - b);
  if (n.length === 1) return `step ${n[0]}`;
  if (n.length === 2) return `steps ${n[0]} and ${n[1]}`;
  const consecutive = n.every((v, i) => i === 0 || v === n[i - 1]! + 1);
  if (consecutive) return `steps ${n[0]} to ${n[n.length - 1]}`;
  return `steps ${n.slice(0, -1).join(", ")} and ${n[n.length - 1]}`;
}

/** A tool's name inside a sentence, quoted so its own commas do not break the sentence. */
export function quoted(name: string): string {
  return `\u201c${name}\u201d`;
}

function timeCompare(a: number, b: number): string {
  const digits = Math.abs(a - b) < 1 ? 2 : 1;
  return `${a.toFixed(digits)} s against ${b.toFixed(digits)} s`;
}

function listNames(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** Group tools by family for a compact phrase: "3 suction cups and an industrial gripper". */
function familyCounts(tools: { family_title: string }[]): string {
  const counts = new Map<string, number>();
  for (const t of tools) counts.set(t.family_title, (counts.get(t.family_title) ?? 0) + 1);
  const parts = [...counts.entries()].map(([fam, n]) => (n === 1 ? `${/^[aeiou]/i.test(fam) ? "an" : "a"} ${lower(fam)}` : `${numberWord(n)} ${plural(fam)}`));
  return listNames(parts);
}

const TIE_BREAK =
  "the Studio's ranking chose between them (screen success, then fewer actuators, less mass, a shorter step), and the page data does not record those values per tool";

function whyForStep(
  page: HeroPage,
  rec: Recommendation,
  story: Omit<StepStory, "why" | "whyGap" | "pick" | "baseline" | "laneStep" | "baselineLaneStep" | "reusedFrom" | "notes">,
  recStep: RecStep,
): { reasons: string[]; gap: string | null } {
  const out: string[] = [];
  const tool = rec.tools[recStep.tool] ?? null;
  const toolId = tool?.effector_id ?? null;
  const family = tool?.family_title ?? null;
  const passers = story.passed;
  const tried = story.tried;
  const pickPassed = passers.some((p) => p.effector_id === toolId);
  const name = stepVerb(story.step.id);
  const served = rec.steps.filter((s) => s.tool === recStep.tool).map((s) => s.step_id);

  // A later step of a tool already on the wrist: the choice was made at its first step.
  if (served.length > 1 && served[0] !== story.step.id) {
    const first = page.plan.steps.findIndex((s) => s.id === served[0]);
    return { reasons: [`Same tool as step ${first + 1}, already on the wrist; it was chosen there for the steps it has to cover.`], gap: null };
  }

  /** Tools still in the running for this choice after each explanation. */
  let field = passers.length;
  let leftover: ToolOutcome[] = passers;

  if (passers.length === 0) {
    out.push(`No tool passed every screen run of this step; in the full task this one managed to ${name} in ${Math.round(recStep.success_rate * 100)}% of runs.`);
  } else if (passers.length === 1 && pickPassed) {
    out.push(`It is the only tool that could ${name}: 1 of ${tried} tried in physics.`);
    field = 1;
  } else if (family && passers.every((p) => p.family_title === family) && pickPassed) {
    out.push(`Only ${plural(family)} could ${name}: ${passers.length} of ${tried} tools tried, all ${plural(family)}.`);
  } else if (!pickPassed) {
    const cell = toolId ? cellOf(page, toolId, story.step.id) : undefined;
    if (cell && cell.state === "run") out.push(`In the screen it passed ${cell.successes} of ${cell.n} runs; in the full task it passed ${Math.round(recStep.success_rate * 100)}%.`);
  }

  // A tool that serves several steps has to pass all of them.
  if (served.length > 1) {
    const indices = served.map((id) => page.plan.steps.findIndex((s) => s.id === id));
    const allRounders = page.matrix.rows.filter((r) => served.every((sid) => fullPass(r.cells[sid])));
    const which = rec.kind === "multi_head" ? `The head's ${recStep.tool === 0 ? "first" : "second"} tool` : "This tool";
    const all = served.length === 2 ? "both" : `all ${numberWord(served.length)}`;
    out.push(`${which} does ${stepRange(indices)}, so it has to pass ${all}: ${allRounders.length} of the ${page.screen.n_candidates} tools did (${familyCounts(allRounders)}).`);
    leftover = passers.filter((p) => allRounders.some((r) => r.effector_id === p.effector_id));
    field = allRounders.length;
    // The baseline's own tool among them, heavier on its own than the whole recommendation.
    const base = page.best_single;
    const rr = recRow(page);
    if (base && rr && base.tool_details.length === 1) {
      const baseTool = base.tool_details[0]!;
      if (allRounders.some((r) => r.effector_id === baseTool.effector_id) && baseTool.effector_id !== toolId && base.complexity.total_mass_kg > rr.complexity.total_mass_kg) {
        out.push(
          `The ${quoted(baseTool.display_name)} passed them too, but it weighs ${massShort(base.complexity.total_mass_kg)} on its own; the whole recommended setup weighs ${massShort(rr.complexity.total_mass_kg)}.`,
        );
        leftover = leftover.filter((p) => p.effector_id !== baseTool.effector_id);
        field -= 1;
      }
    }
  }

  // A finalist that differs from the recommendation in this tool only.
  const rr = recRow(page);
  let compared = 0;
  if (rr && toolId) {
    const recIds = toolIds(rr);
    for (const f of page.finalists) {
      if (f.setup_id === rr.setup_id || f.roles.includes("best_single")) continue;
      const ids = toolIds(f);
      if (ids.length !== recIds.length) continue;
      const missing = recIds.filter((id) => !ids.includes(id));
      const extra = ids.filter((id) => !recIds.includes(id));
      if (missing.length !== 1 || extra.length !== 1 || missing[0] !== toolId) continue;
      const other = f.tool_details.find((t) => t.effector_id === extra[0]);
      if (!other) continue;
      compared += 1;
      if (f.completed_runs === rr.completed_runs) {
        const bits: string[] = [];
        if (rr.cycle_time_s !== null && f.cycle_time_s !== null && rr.cycle_time_s < f.cycle_time_s) bits.push(`faster (${timeCompare(rr.cycle_time_s, f.cycle_time_s)})`);
        if (rr.complexity.total_mass_kg < f.complexity.total_mass_kg) bits.push(`lighter (${massShort(rr.complexity.total_mass_kg)} against ${massShort(f.complexity.total_mass_kg)})`);
        out.push(`With the ${quoted(other.display_name)} in its place the setup also finished ${f.completed_runs} of ${f.n_runs} runs${bits.length ? `; this one is ${listNames(bits)}` : ""}.`);
      } else {
        out.push(`A finalist with the ${quoted(other.display_name)} in its place finished ${f.completed_runs} of ${f.n_runs} runs, against ${rr.completed_runs} of ${rr.n_runs} with this one.`);
      }
    }
  }

  // What the data cannot explain: tools still in the running that no finalist compares.
  const remaining = field - 1 - compared;
  let gap: string | null = null;
  if (remaining > 0) {
    if (compared > 0) gap = `${remaining === 1 ? "One other passing tool was" : `${numberWord(remaining)} other passing tools were`} not in a finalist: ${TIE_BREAK}.`;
    else {
      const fams = [...new Set(leftover.map((p) => p.family_title))];
      const what = fams.length === 1 && leftover.length > 1 ? `${numberWord(leftover.length)} ${plural(fams[0]!)}` : `${field} tools`;
      gap = `${out.length ? "That leaves" : "This step has"} ${what} that pass: ${TIE_BREAK}.`;
    }
    gap = gap.charAt(0).toUpperCase() + gap.slice(1);
  }
  return { reasons: out, gap };
}

function baselineOn(page: HeroPage, step: PlanStep, index: number): StepStory["baseline"] {
  const base = page.best_single;
  if (!base) return null;
  const rate = base.step_success[step.id];
  const stop = base.stops_at_detail[step.id];
  if (stop && stop.runs > 0) {
    const modes = Object.entries(stop.failure_modes)
      .filter(([, n]) => n > 0)
      .sort((a, b) => b[1] - a[1])
      .map(([m, n]) => `${failurePhrase(m, step, page.plan)} in ${n}`);
    return { text: `Stops here in ${stop.runs} of ${base.n_runs} runs: ${listNames(modes)}.`, tone: "fail" };
  }
  const earlier = page.plan.steps.slice(0, index).some((s) => (base.stops_at_detail[s.id]?.runs ?? 0) >= base.n_runs);
  if (earlier) return { text: "Never gets here: every run stopped at an earlier step.", tone: "muted" };
  if (rate !== undefined) return { text: `Passes in ${Math.round(rate * base.n_runs)} of ${base.n_runs} runs.`, tone: rate >= 1 ? "pass" : "fail" };
  return null;
}

/**
 * A tool's name for a sentence, in lower case: the name the lanes show without the spec after its first comma
 * ("Force-controlled parallel gripper, 48 mm stroke" is "force-controlled parallel gripper").
 */
function toolWords(tool: Pick<ToolBrief, "display_name">): string {
  return lower(tool.display_name.split(",")[0]!.trim() || tool.display_name);
}

/** A name with its article: "a suction cup", "an industrial servo parallel gripper", "a one-finger hand". */
function withArticle(words: string): string {
  return `${/^[aeiou]/i.test(words) && !/^(?:one|uni|use|eu)/i.test(words) ? "an" : "a"} ${words}`;
}

/**
 * The setup in words for the answer, naming its tools by their own names as the lanes do ("a drawer-pull hook tool", not their family,
 * "industrial gripper", which the hotel's first two tools share):
 *
 *   single industrial servo parallel gripper, 110 mm stroke
 *   two-tool head (a soft arm plus a suction cup)
 *   tool change (a drawer-pull hook tool, then a force-controlled parallel gripper, then a suction cup)
 *   two-tool head with two suction cups / tool change between two suction cups   (every tool the same kind)
 *
 * A single tool keeps its full name, as the one tool to read about. In a list the spec after the first comma is left out, unless
 * that would make two of the tools read the same ("a two-finger gripper, then a two-finger gripper"); those keep their full names.
 * The kind comes first, up to the first comma or bracket, so a caption can quote just that (the home hero does).
 */
export function setupPhrase(rec: Pick<Recommendation, "kind" | "tools" | "label">): string {
  if (rec.tools.length === 0) return lower(rec.label);
  if (rec.kind === "single") return `single ${lower(rec.tools[0]?.display_name ?? rec.label)}`;
  const short = rec.tools.map(toolWords);
  const head = rec.kind === "multi_head" ? "two-tool head" : "tool change";
  if (short.every((w) => w === short[0])) return `${head} ${rec.kind === "multi_head" ? "with" : "between"} ${numberWord(rec.tools.length)} ${plural(short[0]!)}`;
  const names = rec.tools.map((t, i) => (short.filter((w) => w === short[i]).length > 1 ? lower(t.display_name) : short[i]!));
  return `${head} (${names.map(withArticle).join(rec.kind === "multi_head" ? " plus " : ", then ")})`;
}

/** A phrase with a comma of its own is set off by commas in a sentence ("A single gripper, 110 mm stroke, finishes ..."); commas only inside brackets are not. */
function inSentence(phrase: string): string {
  return phrase.replace(/\([^)]*\)/g, "").includes(", ") ? `${phrase},` : phrase;
}

// ---------------------------------------------------------------- the focus setup

/**
 * The setup a story page centres on: the recommendation, or, when no setup completes the task,
 * the closest one (the first of the no-winner report). Its per-step tool and success rate come
 * from the recommendation, or for the closest setup from its finalist row and its replay lane.
 */
export interface FocusSetup {
  role: "recommended" | "closest";
  setup_id: string;
  kind: SetupKind;
  label: string;
  tools: ToolBrief[];
  completed_runs: number;
  n_runs: number;
  fidelity: Tier;
  cycle_time_s: number | null;
  /** Its finalist row (complexity, stops), when the page lists one. */
  row: FinalistRow | null;
  /** Its replay lane ("recommended" or "closest"), when exported. */
  lane: StudioLane | null;
  steps: RecStep[];
}

export function focusOf(page: HeroPage): FocusSetup | null {
  const rec = page.recommendation;
  if (rec) {
    return {
      role: "recommended",
      setup_id: rec.setup_id,
      kind: rec.kind,
      label: rec.label,
      tools: rec.tools,
      completed_runs: rec.completed_runs,
      n_runs: rec.n_runs,
      fidelity: rec.fidelity,
      cycle_time_s: rec.cycle_time_s,
      row: page.finalists.find((f) => f.setup_id === rec.setup_id) ?? null,
      lane: page.lanes.find((l) => l.id === "recommended") ?? null,
      steps: rec.steps,
    };
  }
  const c = page.no_winner?.closest[0];
  if (!c) return null;
  const lane = page.lanes.find((l) => l.id === "closest" && l.setup_id === c.setup_id) ?? null;
  const tools = c.tool_details.length ? c.tool_details : c.tools;
  const steps: RecStep[] = page.plan.steps.map((step) => {
    const ls = lane?.steps.find((x) => x.step_id === step.id);
    const tool = ls ? ls.tool : 0;
    const brief = tools[tool] ?? null;
    const fidelity = ls?.fidelity ?? brief?.fidelity ?? c.fidelity;
    return {
      step_id: step.id,
      primitive: step.primitive,
      primitive_used: ls?.primitive ?? null,
      tool,
      tool_label: tools.length > 1 && !ls ? c.label : (brief?.display_name ?? c.label),
      success_rate: c.step_success[step.id] ?? 0,
      fidelity,
      // the tool's own caveat (a Studio-only tool's, or its family's for an approximation)
      caveats: brief?.caveat ? [brief.caveat] : [],
      verify_on_hardware: fidelity !== "rigid",
    };
  });
  return {
    role: "closest",
    setup_id: c.setup_id,
    kind: c.kind,
    label: c.label,
    tools,
    completed_runs: c.completed_runs,
    n_runs: c.n_runs,
    fidelity: c.fidelity,
    cycle_time_s: c.cycle_time_s,
    row: c,
    lane,
    steps,
  };
}

/** Where each lane stops, by lane id: its own first failed step and why, or null for a lane that finishes. */
export function laneStops(page: Pick<HeroPage, "lanes" | "plan">): Record<string, { step: number; phrase: string } | null> {
  const out: Record<string, { step: number; phrase: string } | null> = {};
  for (const lane of page.lanes) {
    const fail = lane.completed ? undefined : lane.steps.find((s) => !s.success);
    const index = fail ? page.plan.steps.findIndex((s) => s.id === fail.step_id) : -1;
    const planStep = index >= 0 ? page.plan.steps[index]! : null;
    out[lane.id] = fail && planStep ? { step: index + 1, phrase: fail.failure_mode ? failurePhrase(fail.failure_mode, planStep, page.plan) : "failed" } : null;
  }
  return out;
}

/** A lane in a sentence: "recommended setup", "closest setup", "best single tool". */
export function laneNoun(lane: Pick<StudioLane, "id" | "title">): string {
  if (lane.id === "recommended") return "recommended setup";
  if (lane.id === "closest") return "closest setup";
  if (lane.id === "best_single") return "best single tool";
  return lane.title.toLowerCase();
}

/** A lane's name on the story page: "Recommended", "Closest setup", "Best single tool". */
export function laneName(lane: Pick<StudioLane, "id" | "title">): string {
  if (lane.id === "recommended") return "Recommended";
  if (lane.id === "closest") return "Closest setup";
  if (lane.id === "best_single") return "Best single tool";
  return lane.title;
}

/** How a setup fails, for the answer: "it can't open the lid", "it stops at step 3 (...) in 5 runs". */
function stopsPhrase(page: HeroPage, row: FinalistRow): string {
  const stops = Object.entries(row.stops_at_detail)
    .filter(([, d]) => d.runs > 0)
    .sort((a, b) => b[1].runs - a[1].runs);
  if (!stops.length) return "it does not finish";
  if (stops.length === 1 && stops[0]![1].runs === row.n_runs) return `it can't ${stepVerb(stops[0]![0])}`;
  const at = (id: string) => page.plan.steps.findIndex((s) => s.id === id) + 1;
  const parts = stops.map(([id, d], i) => `at step ${at(id)} (${stepVerb(id)}) in ${d.runs}${i === 0 ? (d.runs === 1 ? " run" : " runs") : ""}`);
  return `it stops ${listNames(parts)}`;
}

/** How the best single tool fails, for the answer: "it can't open the lid". */
function baselineFailure(page: HeroPage): string {
  return stopsPhrase(page, page.best_single!);
}

export function answerSentence(page: HeroPage): string {
  const rec = page.recommendation;
  const base = page.best_single;
  if (!rec) {
    const focus = focusOf(page);
    if (!focus?.row) return page.no_winner ? `No setup finishes this task. ${capital(page.no_winner.reason)}.` : page.headline;
    const phrase = setupPhrase(focus);
    const closest = `The closest, a ${phrase}, finishes ${focus.completed_runs} of ${focus.n_runs} runs: ${stopsPhrase(page, focus.row)}`;
    return `No setup finishes this task. ${closest}.`;
  }
  const phrase = setupPhrase(rec);
  const head = `A ${inSentence(phrase)} finishes the task in ${rec.completed_runs} of ${rec.n_runs} runs`;
  if (!base) return `${head}.`;
  if (base.setup_id === rec.setup_id) return `${head}; it is also the best single off-the-shelf tool.`;
  const tail = `the best single off-the-shelf tool finishes ${base.completed_runs} of ${base.n_runs}${base.completed_runs === 0 ? `: ${baselineFailure(page)}` : ""}`;
  return `${head}; ${tail}.`;
}

function capital(s: string): string {
  const t = s.trim().replace(/\.$/, "");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

// ---------------------------------------------------------------- the baseline

export interface BaselineStory {
  /** The best eligible single tool's finalist row, or null when no single tool is eligible. */
  row: FinalistRow | null;
  /** It is also the best single off-the-shelf tool (the replay's second lane). */
  offTheShelf: boolean;
  /** The pick against it, or why there is none, in sentences. */
  lines: string[];
}

function points(delta: number): string {
  const p = Math.round(Math.abs(delta) * 1000) / 10;
  return `${Number.isInteger(p) ? p.toFixed(0) : p.toFixed(1)} ${p === 1 ? "point" : "points"}`;
}

/**
 * The recommender's baseline in words (rule R01: one tool by default): what the best single tool
 * does, and why the pick is it, or beats it; or, when no single tool is eligible, how each single
 * tool among the finalists stops. Null for a page exported before the recommender had a baseline.
 */
export function baselineStory(page: HeroPage): BaselineStory | null {
  const b = page.baseline;
  if (!b) return null;
  const rec = page.recommendation;
  if (!b.row) {
    const singles = finalistsOf(page).filter((f) => f.kind === "single");
    const lines = singles.length
      ? singles.map((f) => `${capital(f.label)} finishes ${f.completed_runs} of ${f.n_runs} runs: ${stopsPhrase(page, f)}.`)
      : ["No single tool reached the finalists."];
    return { row: null, offTheShelf: false, lines: ["No single tool is eligible: none finished the task on the condition grid.", ...lines] };
  }
  const row = b.row;
  const offTheShelf = page.best_single?.setup_id === row.setup_id;
  const vs = rec?.versus_baseline ?? null;
  const lines: string[] = [];
  const runs = `It finishes ${row.completed_runs} of ${row.n_runs} runs${row.completed_runs < row.n_runs ? `: ${stopsPhrase(page, row)}` : ""}.`;
  lines.push(runs);
  if (rec && vs) {
    const setup = `the ${setupPhrase(rec)}`;
    const against = `${rec.completed_runs} of ${rec.n_runs} runs against ${row.completed_runs} of ${row.n_runs}`;
    switch (vs.basis) {
      case "baseline":
        lines.push("It is the recommendation: no second tool or tool change earned its place (rule R01, one tool by default).");
        break;
      case "tie":
        lines.push(vs.note ?? "A same-complexity tie picked a simpler single tool over it.");
        break;
      case "R09":
      case "R10":
        lines.push(
          `It has ${vs.basis === "R09" ? "a soft arm" : "a dexterous hand"}, and setups with ${vs.basis === "R09" ? "a soft arm" : "a dexterous hand"} rank below every rigid setup (integrator rule ${vs.basis}), whatever they complete.`,
        );
        break;
      case "step_floor": {
        // the step it fails outright: where most of its runs stop, else its weakest step
        const stop = Object.entries(row.stops_at_detail).sort((a, b) => b[1].runs - a[1].runs)[0];
        const id = stop ? stop[0] : (row.weakest_step?.step_id ?? null);
        const rate = id !== null ? (row.step_success[id] ?? row.weakest_step?.success_rate ?? 0) : null;
        const at = id !== null ? ` (${stepVerb(id)}: ${Math.round((rate ?? 0) * 100)}% of runs, under half)` : "";
        lines.push(`It fails a step outright${at}, so ${setup} is recommended: ${against}.`);
        break;
      }
      case "margin":
        lines.push(
          `${capital(setup)} finishes ${points(vs.completion_delta)} more of the runs (${against}), more than the ${points(vs.needed_gain ?? 0)} its ${vs.added_pieces} added ${vs.added_pieces === 1 ? "piece" : "pieces"} must earn (rule R01: 5 points for each tool or changer added, an assumed margin).`,
        );
        break;
      case "ranking":
        lines.push(`The ranking put ${setup} first: completion, then the weakest step, then the cycle time and the simplest setup.`);
        break;
    }
    if (!vs.same_setup && vs.cycle_delta_s !== null && Math.abs(vs.cycle_delta_s) >= 0.05) {
      lines.push(`Its cycle is ${Math.abs(vs.cycle_delta_s).toFixed(1)} s ${vs.cycle_delta_s > 0 ? "shorter" : "longer"} than the pick's; cycle time never decides between them.`);
    }
  } else if (!rec) {
    lines.push("No setup finishes the task, so nothing is compared with it.");
  }
  return { row, offTheShelf, lines };
}

/** Every finalist row on the page, the best single and the closest included, once. */
function finalistsOf(page: HeroPage): FinalistRow[] {
  const out = [...page.finalists];
  for (const r of [page.best_single, ...(page.no_winner?.closest ?? [])]) if (r && !out.some((f) => f.setup_id === r.setup_id)) out.push(r);
  return out;
}

/** For the closest setup: how it did on this step, and where it stops. */
function whyClosest(page: HeroPage, focus: FocusSetup, story: { step: PlanStep; passed: ToolOutcome[]; tried: number }, recStep: RecStep): string[] {
  const out: string[] = [];
  const row = focus.row;
  const name = stepVerb(story.step.id);
  const n = focus.n_runs;
  const stop = row?.stops_at_detail[story.step.id];
  if (stop && stop.runs > 0) {
    const list = Object.entries(stop.failure_modes)
      .filter(([, k]) => k > 0)
      .sort((a, b) => b[1] - a[1]);
    const how =
      list.length === 1 && list[0]![1] === stop.runs
        ? ` it ${failurePhrase(list[0]![0], story.step, page.plan)}`
        : list.length
          ? `: ${listNames(list.map(([m, k]) => `${failurePhrase(m, story.step, page.plan)} in ${k}`))}`
          : "";
    out.push(`This is where it stops: in ${stop.runs} of ${n} runs${how}.`);
  } else {
    const earlier = page.plan.steps.slice(0, Math.max(0, page.plan.steps.findIndex((s) => s.id === story.step.id)));
    const before = earlier.reduce((k, s) => k + (row?.stops_at_detail[s.id]?.runs ?? 0), 0);
    if (row && before >= n) out.push("It never gets here: every run stopped at an earlier step.");
    else out.push(`It passed this step in ${Math.round(recStep.success_rate * n)} of ${n} full-task runs.`);
  }
  if (story.passed.length === 0) out.push(`No tool passed every screen run of this step (${story.tried} tried in physics).`);
  else out.push(`${story.passed.length} of ${story.tried} tools tried in physics passed every screen run of this step; the closest setup is the one that got furthest through the whole task.`);
  return out;
}

/** Why it takes more than one tool: the bottleneck step's tools and what they cannot do. */
function whySetup(page: HeroPage, stories: Omit<StepStory, "why" | "whyGap" | "pick" | "baseline" | "laneStep" | "baselineLaneStep" | "reusedFrom" | "notes">[]): string[] {
  const rec = page.recommendation;
  if (!rec || rec.kind === "single") return [];
  const out: string[] = [];
  const everywhere = page.matrix.rows.filter((r) => page.plan.steps.every((s) => fullPass(r.cells[s.id])));
  out.push(
    everywhere.length === 0
      ? `No single tool passed every step in the screen (${page.screen.n_candidates} tools, ${page.screen.n_conditions} conditions each).`
      : `${everywhere.length} single tool${everywhere.length === 1 ? "" : "s"} passed every step in the screen, but none finished the full task as often.`,
  );
  const hardest = [...stories].sort((a, b) => a.passed.length - b.passed.length || a.index - b.index)[0];
  if (hardest && hardest.passed.length > 0) {
    const n = hardest.passed.length;
    const fams = [...new Set(hardest.passed.map((p) => p.family_title))];
    const who = n === 1 ? `the ${quoted(hardest.passed[0]!.display_name)}` : fams.length === 1 ? `${numberWord(n)} ${plural(fams[0]!)}` : `${n} tools`;
    // Other steps none of those tools can pass, grouped by what stops them there.
    const byReason = new Map<string, number[]>();
    let nBlocked = 0;
    for (const s of stories) {
      if (s.step.id === hardest.step.id) continue;
      if (hardest.passed.some((p) => s.passed.some((q) => q.effector_id === p.effector_id))) continue;
      nBlocked += 1;
      const reasons = new Map<string, number>();
      for (const p of hardest.passed) {
        const o = [...s.failed, ...s.ruledOut, ...s.partial].find((x) => x.effector_id === p.effector_id);
        if (!o) continue;
        const why =
          o.status === "ruled_out"
            ? (o.group ?? "ruled out")
            : o.status === "partial"
              ? `passed only ${o.successes} of ${o.n} screen runs`
              : failurePhrase(o.dominant ?? "criterion_not_met", s.step, page.plan);
        reasons.set(why, (reasons.get(why) ?? 0) + 1);
      }
      const top = [...reasons.entries()].sort((a, b) => b[1] - a[1])[0];
      if (top) byReason.set(top[0], [...(byReason.get(top[0]) ?? []), s.index]);
    }
    const blocked = [...byReason.entries()].map(([why, idx]) => `${stepRange(idx)}: ${why}`);
    out.push(
      `Step ${hardest.index + 1} is the bottleneck: only ${who} could ${stepVerb(hardest.step.id)}${
        blocked.length ? `, and ${n === 1 ? "it fails" : "none of them passes"} ${nBlocked === stories.length - 1 ? (n === 1 ? "every other step" : "any other step") : "the rest"} (${blocked.join("; ")})` : ""
      }.`,
    );
  }
  return out;
}

function laneStepOf(lane: StudioLane | undefined, stepId: string): LaneStep | null {
  return lane?.steps.find((s) => s.step_id === stepId) ?? null;
}

/** Everything the story page shows, derived from the page data. */
export function buildStory(page: HeroPage): UseCaseStory {
  const rec = page.recommendation;
  const focus = focusOf(page);
  const extra = unscreened(page);
  const ratings = ratingsOf(page);
  const recLane = focus?.lane ?? page.lanes.find((l) => l.id !== "best_single") ?? page.lanes[0];
  const baseLane = page.lanes.find((l) => l.id === "best_single");

  const partials = page.plan.steps.map((step, index) => {
    const outcomes = [...page.matrix.rows.map((r) => outcomeOf(r, r.cells[step.id], ratings)), ...extra];
    const passed = outcomes.filter((o) => o.status === "passed");
    const partial = outcomes.filter((o) => o.status === "partial");
    const failed = outcomes.filter((o) => o.status === "failed");
    const ruledOut = outcomes.filter((o) => o.status === "ruled_out");
    const runFailures: Record<string, number> = {};
    let runs = 0;
    for (const o of [...passed, ...partial, ...failed]) {
      runs += o.n;
      for (const [m, k] of Object.entries(o.failure_modes)) runFailures[m] = (runFailures[m] ?? 0) + k;
    }
    const bar: BarSegment[] = [];
    if (passed.length) bar.push({ key: "passed", kind: "passed", label: "passed", tools: passed.length });
    if (partial.length) bar.push({ key: "partial", kind: "partial", label: "passed some runs", tools: partial.length });
    const byMode = new Map<string, number>();
    for (const o of failed) byMode.set(o.dominant ?? "criterion_not_met", (byMode.get(o.dominant ?? "criterion_not_met") ?? 0) + 1);
    for (const [mode, k] of [...byMode.entries()].sort((a, b) => b[1] - a[1])) {
      bar.push({ key: `failed:${mode}`, kind: "failed", label: failurePhrase(mode, step, page.plan), tools: k });
    }
    if (ruledOut.length) bar.push({ key: "ruled_out", kind: "ruled_out", label: "ruled out before physics", tools: ruledOut.length });
    return {
      step,
      index,
      name: stepName(step.id),
      n_candidates: page.screen.n_candidates,
      tried: passed.length + partial.length + failed.length,
      passed,
      partial,
      failed,
      ruledOut,
      runs,
      runFailures,
      bar,
    };
  });

  const steps: StepStory[] = partials.map((p) => {
    const recStep = focus?.steps.find((s) => s.step_id === p.step.id) ?? null;
    const tool = recStep && focus ? (focus.tools[recStep.tool] ?? null) : null;
    return {
      ...p,
      pick: recStep ? { ...recStep, family_title: tool?.family_title ?? null, effector_id: tool?.effector_id ?? null, rating: tool?.rating ?? null } : null,
      ...(() => {
        if (rec && recStep) {
          const w = whyForStep(page, rec, p, recStep);
          return { why: w.reasons, whyGap: w.gap };
        }
        return { why: focus && recStep ? whyClosest(page, focus, p, recStep) : [], whyGap: null };
      })(),
      reusedFrom: (() => {
        if (!focus || !recStep) return null;
        const first = focus.steps.find((s) => s.tool === recStep.tool);
        return first && first.step_id !== p.step.id ? page.plan.steps.findIndex((s) => s.id === first.step_id) : null;
      })(),
      baseline: baselineOn(page, p.step, p.index),
      laneStep: laneStepOf(recLane, p.step.id),
      baselineLaneStep: laneStepOf(baseLane, p.step.id),
      notes: stepNotes(recStep, laneStepOf(recLane, p.step.id)),
    };
  });

  return {
    answer: answerSentence(page),
    whySetup: whySetup(page, partials),
    steps,
    rankingReason: rec?.why ?? null,
  };
}

/** Ruled-out tools grouped by reason, most common first. */
export function ruledOutGroups(tools: ToolOutcome[]): { group: string; tools: ToolOutcome[] }[] {
  const map = new Map<string, ToolOutcome[]>();
  for (const t of tools) {
    const g = t.group ?? "ruled out";
    map.set(g, [...(map.get(g) ?? []), t]);
  }
  return [...map.entries()].map(([group, list]) => ({ group, tools: list })).sort((a, b) => b.tools.length - a.tools.length || a.group.localeCompare(b.group));
}

export { humanize };
