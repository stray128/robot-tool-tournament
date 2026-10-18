/**
 * Task plans in the browser: types, the validator and the edit operations of the intake page.
 *
 * The validator mirrors the Python side message for message: `task_spec_from_tool_args` and
 * `TaskSpec.problems` (src/rtt/studio/types.py), then `check_plan` (src/rtt/studio/engineer/
 * checks.py), as `judge_args` (src/rtt/studio/engineer/engineer.py) runs them. Every number it
 * checks against comes from catalog.json, which Python writes from the contract
 * (`python -m rtt.studio.engineer export-web`). web/tests/studio_intake.test.ts replays the
 * parity cases Python wrote (parity.json) and asserts identical problems, warnings, task ids and
 * fidelity. Brand names are not checked here (the list is not shipped to the browser); the API
 * and the Python parser check them.
 */

/* ------------------------------------------------------------------ catalog */

export interface ParamDesc {
  name: string;
  kind: "float" | "int" | "choice" | "bool";
  description: string;
  unit: string;
  low: number | null;
  high: number | null;
  choices: string[];
  default: number | string | boolean;
}

export interface TemplateDesc {
  kind: string;
  display_name: string;
  description: string;
  note: string;
  model: string;
  fidelity: Tier;
  caveat: string;
  params: ParamDesc[];
  handles: string[];
  quantities: string[];
  needs_ref: boolean;
}

export interface PrimitiveDesc {
  kind: string;
  description: string;
  fidelity: Tier;
  caveat: string;
  objects: string[];
  handles: string[];
  targets: string[];
  params: ParamDesc[];
  quantities: string[];
  uses_roll: boolean;
  may_hold: boolean;
}

export interface QuantityDesc {
  name: string;
  unit: string;
  low: number;
  high: number;
  description: string;
}

export interface Catalog {
  format: string;
  contract: string;
  label: string;
  templates: TemplateDesc[];
  primitives: PrimitiveDesc[];
  quantities: QuantityDesc[];
  handles: Record<string, string>;
  library_parts: string[];
  /** The simulator's conventions (track S2): directions, captive threads, force and torque shares. */
  sim: {
    twist_sense: Record<string, "cw" | "ccw">;
    no_pick: string[];
    picks: string[];
    thread_lead_extra_mm: number;
    force_share: number;
    default_push_n: number;
    torque_share: number;
    wrist_torque_share: number;
    lid_hinge_mm: number;
    depth_tolerance_mm: number;
    window_min: Record<string, number>;
    layout: {
      clearance_mm: number;
      door_jamb_mm: [number, number, number, number];
      bar_diameter_mm: number;
      drawer_front_mm: number;
      knob_head_mm: number;
      valve_body_mm: [number, number];
      plate_min_mm: number;
      port_wall_mm: number;
      port_sides: number;
      plug_gap_mm: number;
      lid_back_wall_mm: number;
      arc_step_deg: number;
    };
  };
  limit_quantities: string[];
  goal_extra: string[];
  limits: {
    max_objects: number;
    max_steps: number;
    pose_mm: number;
    roll_torque_nm: number;
    constraints: Record<string, [number, number]>;
    text: { title: number; description: number; label: number; line: number; note: number; what: number };
    max_assumptions: number;
    max_clarifications: number;
    max_unsupported: number;
    refusal: number;
  };
}

export type Tier = "rigid" | "approximation" | "not_simulated";

/* ------------------------------------------------------------------ plan (tool arguments) */

export type ParamValue = number | string | boolean;

export interface Pose {
  x_mm: number;
  y_mm: number;
  yaw_deg: number;
}

export interface PlanObject {
  name: string;
  kind: string;
  label: string;
  params: Record<string, ParamValue>;
  pose: Pose;
  ref: string | null;
}

export interface Criterion {
  quantity: string;
  at_least: number | null;
  at_most: number | null;
}

export interface Constraints {
  max_force_n: number | null;
  max_torque_nm: number | null;
  precision_mm: number | null;
  precision_deg: number | null;
}

export interface PlanStep {
  id: string;
  primitive: string;
  object: string;
  handle: string;
  target: string | null;
  description: string;
  params: Record<string, ParamValue>;
  constraints: Constraints;
  success: Criterion[];
  after: string[];
  hold_after: boolean;
}

export interface Clarification {
  question: string;
  assumed_answer: string;
}

export interface Unsupported {
  what: string;
  reason: string;
  nearest_primitive: string | null;
}

export interface PlanArgs {
  title: string;
  description: string;
  objects: PlanObject[];
  steps: PlanStep[];
  assumptions: string[];
  clarifications: Clarification[];
  unsupported: Unsupported[];
  refusal: string | null;
}

export const CONSTRAINT_KEYS = ["max_force_n", "max_torque_nm", "precision_mm", "precision_deg"] as const;

/* ------------------------------------------------------------------ Python-compatible text */

/** Python's `repr` of a string, number, boolean, null or list (enough for the messages). */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return pyFloat(value);
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  if (typeof value === "string") {
    const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
    let body = value.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
    if (quote === "'") body = body.replace(/'/g, "\\'");
    return `${quote}${body}${quote}`;
  }
  return String(value);
}

/** Python's `format(v, "g")`: 6 significant digits, exponent outside 1e-4 <= |v| < 1e6. */
export function fmtG(v: number): string {
  if (!Number.isFinite(v)) return v > 0 ? "inf" : v < 0 ? "-inf" : "nan";
  if (v === 0) return Object.is(v, -0) ? "-0" : "0";
  const rounded = Number(v.toPrecision(6));
  const exp = Math.floor(Math.log10(Math.abs(rounded)));
  const strip = (s: string) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  if (exp >= -4 && exp < 6) return strip(rounded.toFixed(Math.max(0, 5 - exp)));
  const mantissa = strip((rounded / 10 ** exp).toFixed(5));
  return `${mantissa}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
}

/** Python's `float.__repr__` / `json.dumps` of a number (integral values printed as integers). */
export function pyFloat(v: number): string {
  if (Number.isInteger(v)) return String(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e16)) {
    const [m, e] = v.toExponential().split("e");
    const n = Number(e);
    return `${m}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  return String(v);
}

/** `rtt.studio.engineer.catalog.num`: at most two decimals, no trailing zeros. */
export function num(v: number, digits = 2): string {
  let s = v.toFixed(digits);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "-0" || s === "" ? "0" : s;
}

/** Python's code-point length. */
function pyLen(s: string): number {
  return [...s].length;
}

/* ------------------------------------------------------------------ catalog lookups */

export function template(catalog: Catalog, kind: string): TemplateDesc | undefined {
  return catalog.templates.find((t) => t.kind === kind);
}

export function primitive(catalog: Catalog, kind: string): PrimitiveDesc | undefined {
  return catalog.primitives.find((p) => p.kind === kind);
}

export function quantity(catalog: Catalog, name: string): QuantityDesc | undefined {
  return catalog.quantities.find((q) => q.name === name);
}

/** The handles `prim` may address on an object of `kind` (template order). */
export function handlesFor(catalog: Catalog, prim: string, kind: string): string[] {
  const t = template(catalog, kind);
  const p = primitive(catalog, prim);
  if (!t || !p) return [];
  return t.handles.filter((h) => p.handles.includes(h));
}

/** Quantities a step of `prim` on an object of `kind` measures (sorted, as Python lists them). */
export function measurable(catalog: Catalog, prim: string, kind: string): string[] {
  const t = template(catalog, kind);
  const p = primitive(catalog, prim);
  return [...new Set([...(t?.quantities ?? []), ...(p?.quantities ?? [])])].sort();
}

function goalQuantities(catalog: Catalog, t: TemplateDesc): Set<string> {
  return new Set([...t.quantities, ...catalog.goal_extra]);
}

/* ------------------------------------------------------------------ parsing (mirrors the Python constructors) */

const SLUG = /^[a-z][a-z0-9_]{0,31}$/;
const REF = /^(library|upload|scan):[a-z0-9][a-z0-9_.-]{0,63}$/;
const NULL_TEXT = new Set(["none", "null", "nil", "n/a", "na", ""]);

class PyError extends Error {}

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function slug(what: string, value: unknown): string {
  if (typeof value !== "string" || !SLUG.test(value)) {
    throw new PyError(`${what} must be a short snake_case id (^[a-z][a-z0-9_]{0,31}$), got ${pyRepr(value)}`);
  }
  return value;
}

function text(what: string, value: unknown, max: number, required = true): string {
  if (typeof value !== "string") throw new PyError(`${what} must be a string`);
  if (required && !value.trim()) throw new PyError(`${what} must not be empty`);
  if (pyLen(value) > max) throw new PyError(`${what} is longer than ${max} characters`);
  return value;
}

function number(what: string, value: unknown, low: number, high: number): number {
  if (!isNum(value)) throw new PyError(`${what} must be a finite number, got ${pyRepr(value)}`);
  if (value < low || value > high) throw new PyError(`${what} must be in [${fmtG(low)}, ${fmtG(high)}], got ${fmtG(value)}`);
  return value;
}

function optional(what: string, value: unknown, low: number, high: number): number | null {
  return value === null || value === undefined ? null : number(what, value, low, high);
}

function paramProblems(p: ParamDesc, value: unknown): string[] {
  if (p.kind === "bool") return typeof value === "boolean" ? [] : [`${p.name} must be true or false`];
  if (p.kind === "choice") return typeof value === "string" && p.choices.includes(value) ? [] : [`${p.name} must be one of ${pyRepr(p.choices)}`];
  if (!isNum(value)) return [`${p.name} must be a finite number`];
  if (p.kind === "int" && !Number.isInteger(value)) return [`${p.name} must be a whole number`];
  if (value < (p.low as number) || value > (p.high as number)) {
    return [`${p.name} must be in [${fmtG(p.low as number)}, ${fmtG(p.high as number)}] ${p.unit}`.trimEnd()];
  }
  return [];
}

function paramsProblems(space: ParamDesc[], params: Record<string, unknown>): string[] {
  const byName = new Map(space.map((p) => [p.name, p]));
  const keys = Object.keys(params).sort();
  const out = keys.filter((k) => !byName.has(k)).map((k) => `unknown param ${pyRepr(k)}`);
  for (const p of space) if (!(p.name in params)) out.push(`missing param ${pyRepr(p.name)}`);
  for (const k of keys) {
    const p = byName.get(k);
    if (p) out.push(...paramProblems(p, params[k]));
  }
  return out;
}

function need(o: Record<string, unknown>, key: string): unknown {
  if (!(key in o)) throw new PyError(pyRepr(key)); // a Python KeyError prints the key
  return o[key];
}

/** One object as `ObjectInstance(...)` builds it (throws the first problem, as Python does). */
function parseObject(catalog: Catalog, o: unknown): PlanObject {
  if (!isObj(o)) throw new PyError("'list' object has no attribute 'get'");
  const pose = isObj(o.pose) ? o.pose : {};
  const name = need(o, "name");
  const kind = need(o, "kind");
  const label = "label" in o ? o.label : (o.name ?? "");
  const params = need(o, "params");
  const lim = catalog.limits.pose_mm;
  const parsedPose: Pose = {
    x_mm: number("pose.x_mm", "x_mm" in pose ? pose.x_mm : 0, -lim, lim),
    y_mm: number("pose.y_mm", "y_mm" in pose ? pose.y_mm : 0, -lim, lim),
    yaw_deg: number("pose.yaw_deg", "yaw_deg" in pose ? pose.yaw_deg : 0, -180, 180),
  };
  slug("object name", name);
  const t = template(catalog, String(kind));
  if (!t) throw new PyError(`${pyRepr(kind)} is not a valid TemplateKind`);
  text(`object ${name as string} label`, label, catalog.limits.text.label);
  if (!isObj(params)) throw new PyError("object params must be a mapping");
  const problems = paramsProblems(t.params, params);
  if (problems.length) throw new PyError(`object ${name as string} (${t.kind}): ${problems.join("; ")}`);
  const ref = o.ref ?? null;
  if (ref !== null && (typeof ref !== "string" || !REF.test(ref))) {
    throw new PyError(`object ${name as string}: ref must match ^(library|upload|scan):[a-z0-9][a-z0-9_.-]{0,63}$, got ${pyRepr(ref)}`);
  }
  const needs = t.kind === "part" && params.shape === "mesh";
  if (needs && ref === null) throw new PyError(`object ${name as string}: a mesh part needs its ref`);
  if (!needs && ref !== null) throw new PyError(`object ${name as string}: only a mesh part has a ref`);
  return { name: name as string, kind: t.kind, label: label as string, params: params as Record<string, ParamValue>, pose: parsedPose, ref: ref as string | null };
}

function parseCriterion(catalog: Catalog, c: unknown): Criterion {
  if (!isObj(c)) throw new PyError("argument after ** must be a mapping");
  const q = quantity(catalog, String(c.quantity));
  if (!q) throw new PyError(`unknown quantity ${pyRepr(c.quantity)}; known: ${pyRepr(catalog.quantities.map((x) => x.name).sort())}`);
  const lo = optional(`${q.name}.at_least`, c.at_least, q.low, q.high);
  const hi = optional(`${q.name}.at_most`, c.at_most, q.low, q.high);
  if (lo === null && hi === null) throw new PyError(`criterion on ${q.name} needs at_least or at_most`);
  if (lo !== null && hi !== null && lo > hi) throw new PyError(`criterion on ${q.name}: at_least ${fmtG(lo)} > at_most ${fmtG(hi)}`);
  return { quantity: q.name, at_least: lo, at_most: hi };
}

/** One step as `Step(...)` builds it (throws the first problem, as Python does). */
function parseStep(catalog: Catalog, s: unknown): PlanStep {
  if (!isObj(s)) throw new PyError("'list' object has no attribute 'get'");
  const id = need(s, "id");
  const prim = need(s, "primitive");
  const object = need(s, "object");
  const handle = need(s, "handle");
  const description = need(s, "description");
  const params = need(s, "params");
  const rawSuccess = need(s, "success");
  if (!Array.isArray(rawSuccess)) throw new PyError("'success' must be a list");
  const success = rawSuccess.map((c) => parseCriterion(catalog, c));
  const rawCons = isObj(s.constraints) ? s.constraints : {};
  const constraints = {} as Constraints;
  for (const k of CONSTRAINT_KEYS) {
    const [lo, hi] = catalog.limits.constraints[k] as [number, number];
    constraints[k] = optional(`constraints.${k}`, rawCons[k], lo, hi);
  }
  const target = s.target ?? null;
  const after = Array.isArray(s.after) ? (s.after as unknown[]) : [];
  const holdAfter = Boolean(s.hold_after ?? false);
  slug("step id", id);
  const info = primitive(catalog, String(prim));
  if (!info) throw new PyError(`${pyRepr(prim)} is not a valid PrimitiveKind`);
  slug(`step ${id as string} object`, object);
  slug(`step ${id as string} handle`, handle);
  if (target !== null) slug(`step ${id as string} target`, target);
  text(`step ${id as string} description`, description, catalog.limits.text.note);
  if (!isObj(params)) throw new PyError("step params must be a mapping");
  const problems = paramsProblems(info.params, params);
  if (problems.length) throw new PyError(`step ${id as string} (${info.kind}): ${problems.join("; ")}`);
  if (success.length < 1 || success.length > 4) throw new PyError(`step ${id as string}: 1 to 4 SuccessCriterion rows`);
  for (const a of after) slug(`step ${id as string} after`, a);
  if (new Set(after).size !== after.length || after.includes(id)) throw new PyError(`step ${id as string}: after must list distinct other steps`);
  if (holdAfter && !info.may_hold) throw new PyError(`step ${id as string}: a ${info.kind} step cannot end holding its object`);
  return {
    id: id as string,
    primitive: info.kind,
    object: object as string,
    handle: handle as string,
    target: target as string | null,
    description: description as string,
    params: params as Record<string, ParamValue>,
    constraints,
    success,
    after: after as string[],
    hold_after: holdAfter,
  };
}

/* ------------------------------------------------------------------ the validator */

export interface Verdict {
  status: "valid" | "refused" | "invalid";
  problems: string[];
  warnings: string[];
  notes: string[];
  /** The parsed plan when it constructs (problems may still exist, from the plan checks). */
  objects: PlanObject[];
  steps: PlanStep[];
  constructed: boolean;
}

/** `rtt.studio.engineer.engineer.normalise_nulls`: null written as text, read as null. */
export function normaliseNulls(args: PlanArgs): { args: PlanArgs; notes: string[] } {
  const out = structuredClone(args) as PlanArgs & Record<string, unknown>;
  const changed: string[] = [];
  const nullish = (v: unknown) => typeof v === "string" && NULL_TEXT.has(v.trim().toLowerCase());
  const fix = (c: unknown, key: string, where: string) => {
    if (isObj(c) && key in c && nullish(c[key])) {
      c[key] = null;
      changed.push(where);
    }
  };
  fix(out, "refusal", "refusal");
  (Array.isArray(out.objects) ? out.objects : []).forEach((o, i) => fix(o, "ref", `objects[${i}].ref`));
  (Array.isArray(out.steps) ? out.steps : []).forEach((st, i) => {
    fix(st, "target", `steps[${i}].target`);
    const cons = isObj(st) && isObj(st.constraints) ? st.constraints : null;
    if (cons) for (const k of Object.keys(cons)) fix(cons, k, `steps[${i}].constraints.${k}`);
    const crit = isObj(st) && Array.isArray(st.success) ? st.success : [];
    crit.forEach((c, j) => {
      fix(c, "at_least", `steps[${i}].success[${j}].at_least`);
      fix(c, "at_most", `steps[${i}].success[${j}].at_most`);
    });
  });
  (Array.isArray(out.unsupported) ? out.unsupported : []).forEach((u, i) => fix(u, "nearest_primitive", `unsupported[${i}].nearest_primitive`));
  const notes = changed.length ? [`Null written as text was read as null (${changed.slice(0, 6).join(", ")}).`] : [];
  return { args: out, notes };
}

const STEP_HINT = "One plain sentence for the step list";

/** `explain_problems`: a bare missing key and a refusal next to steps, reworded. */
function explain(problems: string[], args: PlanArgs): string[] {
  return problems.map((p) => {
    const m = /^(objects|steps)\[(\d+)\]: '([A-Za-z_]+)'$/.exec(p);
    if (m) {
      const list = (m[1] === "objects" ? args.objects : args.steps) as unknown[];
      const item = Array.isArray(list) ? list[Number(m[2])] : undefined;
      const name = isObj(item) ? item[m[1] === "objects" ? "name" : "id"] : undefined;
      const label = typeof name === "string" ? ` (${name})` : "";
      const hint = m[1] === "steps" && m[3] === "description" ? `: ${STEP_HINT}` : "";
      return `${m[1]}[${m[2]}]${label}: missing ${pyRepr(m[3])}${hint}`;
    }
    if (p === "a refused task has no steps") {
      return "refusal: must be null (JSON null, not text) when the plan has steps; a refused use case has no steps";
    }
    return p;
  });
}

function parseParts(catalog: Catalog, args: PlanArgs): { objects: PlanObject[]; steps: PlanStep[]; problems: string[] } {
  const problems: string[] = [];
  const objects: PlanObject[] = [];
  const steps: PlanStep[] = [];
  if (!("objects" in args)) problems.push("$: missing 'objects'");
  (Array.isArray(args.objects) ? args.objects : []).forEach((o, i) => {
    try {
      objects.push(parseObject(catalog, o));
    } catch (e) {
      problems.push(`objects[${i}]: ${(e as Error).message}`);
    }
  });
  if (!("steps" in args)) problems.push("$: missing 'steps'");
  (Array.isArray(args.steps) ? args.steps : []).forEach((s, i) => {
    try {
      steps.push(parseStep(catalog, s));
    } catch (e) {
      problems.push(`steps[${i}]: ${(e as Error).message}`);
    }
  });
  const lim = catalog.limits.text;
  (args.clarifications ?? []).forEach((c, i) => {
    try {
      text("clarification question", c.question, lim.line);
      text("clarification assumed_answer", c.assumed_answer, lim.line);
    } catch (e) {
      problems.push(`clarifications[${i}]: ${(e as Error).message}`);
    }
  });
  (args.unsupported ?? []).forEach((u, i) => {
    try {
      text("unsupported what", u.what, lim.what);
      text("unsupported reason", u.reason, lim.line);
      if (u.nearest_primitive !== null && u.nearest_primitive !== undefined && !primitive(catalog, u.nearest_primitive)) {
        throw new PyError(`${pyRepr(u.nearest_primitive)} is not a valid PrimitiveKind`);
      }
    } catch (e) {
      problems.push(`unsupported[${i}]: ${(e as Error).message}`);
    }
  });
  if (!("title" in args)) problems.push("$: missing 'title'");
  return { objects, steps, problems };
}

/** `TaskSpec.problems` on parsed parts. */
function specProblems(catalog: Catalog, args: PlanArgs, objects: PlanObject[], steps: PlanStep[]): string[] {
  const out: string[] = [];
  const lim = catalog.limits;
  try {
    text("title", args.title, lim.text.title);
    text("description", args.description ?? "", lim.text.description, false);
    if (args.refusal !== null && args.refusal !== undefined) text("refusal", args.refusal, lim.refusal);
  } catch (e) {
    out.push((e as Error).message);
  }
  const assumptions = args.assumptions ?? [];
  if (assumptions.length > lim.max_assumptions) out.push(`at most ${lim.max_assumptions} assumptions`);
  assumptions.forEach((a, i) => {
    try {
      text(`assumptions[${i}]`, a, lim.text.line);
    } catch (e) {
      out.push((e as Error).message);
    }
  });
  if ((args.clarifications ?? []).length > lim.max_clarifications || (args.unsupported ?? []).length > lim.max_unsupported) {
    out.push("at most 6 clarifications and 6 unsupported items");
  }
  if (args.refusal !== null && args.refusal !== undefined) {
    if (steps.length) out.push("a refused task has no steps");
    return out;
  }
  if (!steps.length) out.push("a task needs at least one step (or a refusal)");
  if (objects.length < 1 || objects.length > lim.max_objects) out.push(`a task has 1 to ${lim.max_objects} objects`);
  if (steps.length > lim.max_steps) out.push(`a task has at most ${lim.max_steps} steps`);
  const byName = new Map<string, PlanObject>(objects.map((o) => [o.name, o])); // the last of a name wins, as in Python
  if (byName.size !== objects.length) out.push("object names must be unique");
  if (new Set(steps.map((s) => s.id)).size !== steps.length) out.push("step ids must be unique");
  const seen = new Set<string>();
  let held: string | null = null;
  steps.forEach((s, i) => {
    const where = `step ${s.id}`;
    const info = primitive(catalog, s.primitive) as PrimitiveDesc;
    const obj = byName.get(s.object);
    if (!obj) out.push(`${where}: unknown object ${pyRepr(s.object)}`);
    else {
      const t = template(catalog, obj.kind) as TemplateDesc;
      if (!info.objects.includes(obj.kind)) out.push(`${where}: ${s.primitive} cannot act on a ${obj.kind} (it acts on ${info.objects.join(", ")})`);
      if (!t.handles.includes(s.handle) || !info.handles.includes(s.handle)) {
        out.push(`${where}: handle ${pyRepr(s.handle)} is not one of ${pyRepr(t.handles.filter((h) => info.handles.includes(h)))}`);
      }
      const meas = new Set([...t.quantities, ...info.quantities]);
      for (const c of s.success) {
        if (!meas.has(c.quantity)) out.push(`${where}: ${pyRepr(c.quantity)} is not measured here (use ${pyRepr([...meas].sort())})`);
      }
    }
    if (s.target !== null) {
      const tgt = byName.get(s.target);
      if (!tgt) out.push(`${where}: unknown target ${pyRepr(s.target)}`);
      else if (!info.targets.includes(tgt.kind)) out.push(`${where}: target ${pyRepr(s.target)} must be one of ${pyRepr(info.targets)}`);
      else if (s.target === s.object) out.push(`${where}: the target is the object itself`);
    }
    for (const a of s.after) if (!seen.has(a)) out.push(`${where}: depends on ${pyRepr(a)}, which is not an earlier step`);
    if (held !== null && s.object !== held) out.push(`${where}: acts on ${pyRepr(s.object)} while ${pyRepr(held)} is still held by the previous step`);
    held = s.hold_after ? s.object : null;
    seen.add(s.id);
    if (i === steps.length - 1 && s.hold_after) out.push(`${where}: the last step must release its object`);
  });
  return out;
}

function demand(c: Criterion): number | null {
  let out: number | null = null;
  if (c.at_least !== null && c.at_least > 0) out = c.at_least;
  if (c.at_most !== null && c.at_most < 0) out = Math.max(out ?? 0, -c.at_most);
  return out;
}

const TWIST_NAMES: Record<string, string> = { quarter_turn_latch: "latch", handwheel_valve: "valve", threaded_joint: "threaded member" };

function lead(catalog: Catalog, p: Record<string, ParamValue>): number {
  return Number(p.chamfer_mm) + catalog.sim.thread_lead_extra_mm;
}

function ceiling(catalog: Catalog, kind: string, p: Record<string, ParamValue>, q: string, prim: string): [number, string] | null {
  const n = (k: string) => Number(p[k]);
  if (kind === "hinged_lid" && q === "angle_deg") return [n("max_angle_deg"), "its hinge stop max_angle_deg"];
  if (kind === "drawer" && q === "travel_mm") return [n("travel_mm"), "its travel_mm"];
  if (kind === "quarter_turn_latch" && q === "angle_deg") return [n("turn_deg"), "its turn_deg"];
  if (kind === "handwheel_valve" && q === "angle_deg") return [n("travel_deg"), "its travel_deg"];
  if (kind === "threaded_joint" && q === "turns") return [n("engage_turns"), "its engage_turns"];
  if (kind === "threaded_joint" && q === "depth_mm") {
    if (prim === "insert") return [lead(catalog, p), "the thread start, chamfer_mm + 3 mm; a twist runs it further"];
    const seated = lead(catalog, p) + n("engage_turns") * n("pitch_mm");
    return [seated + catalog.sim.depth_tolerance_mm, "seated, chamfer_mm + 3 mm + engage_turns x pitch_mm"];
  }
  if (kind === "threaded_joint" && q === "angle_deg") return [360 * n("engage_turns"), "its engage_turns x 360 deg"];
  if (kind === "port_plug" && q === "depth_mm") return [n("depth_mm"), "the socket's depth_mm"];
  return null;
}

function torqueNeeded(catalog: Catalog, kind: string, p: Record<string, ParamValue>, s: PlanStep): [number, string] | null {
  if (s.primitive !== "twist") return null;
  if (kind === "quarter_turn_latch") return [Number(p.torque_nm), "the latch's torque_nm"];
  if (kind === "handwheel_valve") return [Number(p.torque_nm), "the valve's torque_nm"];
  if (kind === "threaded_joint") {
    const engage = Number(p.engage_turns);
    const seat = engage * Number(p.pitch_mm);
    let share = 0;
    for (const c of s.success) {
      const d = demand(c);
      if (d === null) continue;
      if (c.quantity === "turns") share = Math.max(share, d / engage);
      else if (c.quantity === "angle_deg") share = Math.max(share, d / (360 * engage));
      else if (c.quantity === "depth_mm") share = Math.max(share, (d - lead(catalog, p)) / seat);
    }
    share = Math.min(1, Math.max(0, share));
    const start = Number(p.start_torque_nm);
    const final = Number(p.final_torque_nm);
    return [start + (final - start) * share, "the thread's torque there, start_torque_nm rising to final_torque_nm"];
  }
  return null;
}

function forceNeeded(kind: string, p: Record<string, ParamValue>, s: PlanStep): [number, string, boolean] | null {
  if (kind === "drawer" && s.primitive === "hook_pull") return [Number(p.pull_force_n), "the drawer's pull_force_n", false];
  if (kind === "drawer" && s.primitive === "push") return [Number(p.pull_force_n), "the drawer's pull_force_n", true];
  if (kind === "port_plug" && s.primitive === "insert") return [Number(p.insert_force_n), "the plug's insert_force_n", true];
  return null;
}

/** `check_parts`: refs and the per-step plan checks. */
function checkParts(catalog: Catalog, objects: PlanObject[], steps: PlanStep[], uploads: string[]): string[] {
  const out: string[] = [];
  const used = new Set<string>();
  objects.forEach((o, i) => {
    if (o.ref === null) return;
    const [scheme, ...rest] = o.ref.split(":");
    const key = rest.join(":");
    const where = `objects[${i}] (${o.name})`;
    if (scheme === "upload") {
      used.add(key);
      if (!uploads.includes(key)) {
        const known = uploads.map((u) => `upload:${u}`).join(", ") || "none";
        out.push(`${where}: ${o.ref} was not uploaded (uploads: ${known}); use a box or cylinder`);
      }
    } else if (scheme === "scan") out.push(`${where}: scan refs cannot be resolved here; use a box or cylinder (ref null)`);
    else if (scheme === "library" && !catalog.library_parts.includes(key)) {
      out.push(`${where}: no library part ${pyRepr(key)} (known: ${catalog.library_parts.join(", ")}); use a box or cylinder`);
    }
  });
  for (const key of uploads) {
    if (!used.has(key)) out.push(`objects: the uploaded part upload:${key} is not used; model it as a part with shape mesh and ref upload:${key}`);
  }
  out.push(...lidProblems(catalog, objects));
  out.push(...layoutProblems(catalog, objects, steps));
  const byName = new Map<string, PlanObject>(objects.map((o) => [o.name, o]));
  const sim = catalog.sim;
  const wrist = sim.wrist_torque_share * catalog.limits.roll_torque_nm;
  steps.forEach((s, i) => {
    const where = `steps[${i}] (${s.id})`;
    const obj = byName.get(s.object);
    const info = primitive(catalog, s.primitive) as PrimitiveDesc;
    if (!obj || !info.objects.includes(obj.kind)) return;
    const t = template(catalog, obj.kind) as TemplateDesc;
    const kind = obj.kind;
    const twist = s.primitive === "twist";
    const goals = goalQuantities(catalog, t);
    const snug = twist && s.success.some((c) => c.quantity === "torque_nm" && c.at_least !== null);
    if (!snug && !s.success.some((c) => goals.has(c.quantity))) {
      const usable = [...goals].filter((q) => t.quantities.includes(q) || info.quantities.includes(q)).sort();
      if (twist) usable.push("torque_nm at_least");
      out.push(`${where}: add a criterion on what the step achieves (${usable.join(", ")}); force, torque, slip and penetration limits alone do not say when it succeeded`);
    }
    for (const c of s.success) {
      const least = sim.window_min[c.quantity];
      const lo = c.at_least;
      const hi = c.at_most;
      if (least === undefined || lo === null || hi === null || hi - lo >= least - 1e-9) continue;
      out.push(`${where}: the ${c.quantity} window ${num(lo)} to ${num(hi)} is narrower than ${num(least)} ${quantity(catalog, c.quantity)?.unit}; a measurement never lands exactly, so widen it or keep one bound`);
    }
    if (s.success.some((c) => c.quantity === "in_target") && s.target === null) out.push(`${where}: in_target needs a target bin`);
    if (s.target !== null && s.hold_after) out.push(`${where}: placing ${pyRepr(s.object)} in ${pyRepr(s.target)} releases it; set hold_after false`);
    if (sim.picks.includes(s.primitive) && sim.no_pick.includes(kind)) {
      out.push(`${where}: a threaded member lives on its stub and cannot be picked up or moved; leave the pick out (insert and twist grip the member in place)`);
    }
    const sense = sim.twist_sense[kind];
    if (twist && sense && s.params.direction !== sense) {
      const back = steps.slice(0, i).some((p) => p.primitive === "twist" && p.object === s.object && p.params.direction === sense);
      if (!back) {
        const name = TWIST_NAMES[kind] as string;
        out.push(
          sense === "ccw"
            ? `${where}: a ${name} starts closed at its stop and opens counter-clockwise; twist it ccw (cw meets the stop)`
            : `${where}: a ${name} tightens clockwise from its start; twist it cw (ccw meets the thread start)`,
        );
      }
    }
    if (s.primitive === "insert" && kind === "threaded_joint" && s.hold_after) {
      out.push(`${where}: end the insert of a threaded member with hold_after false; the twist grips it again in place`);
    }
    if (twist && (kind === "quarter_turn_latch" || kind === "handwheel_valve") && s.success.some((c) => c.quantity === "torque_nm" && c.at_least !== null)) {
      out.push(`${where}: a twist stops as soon as the roll torque reaches a torque_nm at_least, and a ${TWIST_NAMES[kind]} resists most before it turns; drop torque_nm at_least here (it tightens a thread until snug)`);
    }
    if (twist && kind === "threaded_joint" && !steps.slice(0, i).some((p) => p.primitive === "insert" && p.object === s.object)) {
      out.push(`${where}: a threaded member starts on its lead-in, free of the thread; insert ${pyRepr(s.object)} to the thread start before this twist`);
    }
    for (const c of s.success) {
      const d = demand(c);
      if (d === null) continue;
      if ((c.quantity === "lift_mm" || c.quantity === "hold_s") && c.quantity in s.params) {
        const own = Number(s.params[c.quantity]);
        if (d > own) out.push(`${where}: ${c.quantity} at_least ${num(d)} is more than the step's own ${c.quantity} param (${num(own)}); raise the param or lower the criterion`);
        continue;
      }
      const ceil = ceiling(catalog, kind, obj.params, c.quantity, s.primitive);
      if (ceil && d > ceil[0] + 1e-9) out.push(`${where}: ${c.quantity} ${num(d)} cannot be reached; ${pyRepr(s.object)} stops at ${num(ceil[0])} (${ceil[1]})`);
    }
    const tq = torqueNeeded(catalog, kind, obj.params, s);
    if (tq) {
      let [torque, why] = tq;
      for (const c of s.success) {
        if (c.quantity === "torque_nm" && c.at_least !== null && c.at_least > torque) {
          torque = c.at_least;
          why = "its torque_nm at_least criterion";
        }
      }
      const cap = s.constraints.max_torque_nm;
      const allowed = cap !== null ? Math.min(wrist, sim.torque_share * cap) : wrist;
      if (torque > allowed + 1e-9) {
        if (cap !== null && sim.torque_share * cap < wrist) {
          out.push(`${where}: max_torque_nm ${num(cap)} lets the twist turn up to ${num(sim.torque_share * cap)} N m (${num(100 * sim.torque_share)} %), below the ${num(torque)} N m it takes (${why})`);
        } else {
          out.push(`${where}: it takes ${num(torque)} N m (${why}) but a twist turns at most ${num(wrist)} N m (95 % of the wrist roll's 15 N m); lower the torque or refuse the step`);
        }
      }
      for (const c of s.success) {
        if (c.quantity === "torque_nm" && c.at_most !== null && c.at_most < tq[0]) {
          out.push(`${where}: torque_nm at_most ${num(c.at_most)} is below the ${num(tq[0])} N m it takes (${tq[1]})`);
        }
      }
    }
    const fq = forceNeeded(kind, obj.params, s);
    if (fq) {
      const [force, why, shared] = fq;
      const cap = s.constraints.max_force_n;
      if (!shared) {
        if (cap !== null && cap < force) out.push(`${where}: max_force_n ${num(cap)} is below the ${num(force)} N it takes (${why})`);
      } else if (cap !== null && sim.force_share * cap < force) {
        out.push(`${where}: max_force_n ${num(cap)} lets the push reach ${num(sim.force_share * cap)} N (${num(100 * sim.force_share)} %), below the ${num(force)} N it takes (${why})`);
      } else if (cap === null && sim.default_push_n < force) {
        out.push(`${where}: without a max_force_n the push stops at ${num(sim.default_push_n)} N, below the ${num(force)} N it takes (${why}); set max_force_n`);
      }
    }
  });
  return out;
}

/* ------------------------------------------------------------------ layout (rtt/studio/engineer/layout.py) */

export type Polygon = [number, number][];
export type Region = [string, Polygon];

function rect(x0: number, y0: number, x1: number, y1: number): Polygon {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}

function place(poly: Polygon, o: PlanObject): Polygon {
  const a = o.pose.yaw_deg * (Math.PI / 180); // as Python's math.radians
  const c = Math.cos(a);
  const s = Math.sin(a);
  return poly.map(([x, y]) => [o.pose.x_mm + c * x - s * y, o.pose.y_mm + s * x + c * y]);
}

function localRegions(catalog: Catalog, o: PlanObject): Region[] {
  const L = catalog.sim.layout;
  const p = o.params;
  const n = (k: string) => Number(p[k]);
  switch (o.kind) {
    case "bin": {
      const hx = 0.5 * n("length_mm");
      const hy = 0.5 * n("width_mm");
      return [["footprint", rect(-hx, -hy, hx, hy)]];
    }
    case "part": {
      const hx = 0.5 * n("size_x_mm");
      const hy = p.shape === "cylinder" ? hx : 0.5 * n("size_y_mm");
      return [["footprint", rect(-hx, -hy, hx, hy)]];
    }
    case "quarter_turn_latch": {
      const h = 0.5 * Math.max(1.6 * n("head_diameter_mm"), L.plate_min_mm);
      return [["footprint", rect(-h, -h, h, h)]];
    }
    case "threaded_joint": {
      const h = 0.5 * Math.max(1.6 * n("diameter_mm"), L.plate_min_mm);
      return [["footprint", rect(-h, -h, h, h)]];
    }
    case "handwheel_valve": {
      const r = 0.5 * n("rim_diameter_mm");
      const hx = Math.max(r, 0.5 * L.valve_body_mm[0]);
      const hy = Math.max(r, 0.5 * L.valve_body_mm[1]);
      return [["footprint", rect(-hx, -hy, hx, hy)]];
    }
    case "port_plug": {
      const dia = n("plug_diameter_mm");
      const chamfer = Math.min(n("chamfer_mm"), 0.8 * n("depth_mm"));
      const rOut = 0.5 * (dia + n("clearance_mm")) + Math.max(L.port_wall_mm, 0.5 * dia) + chamfer;
      const kOut = rOut / Math.cos(Math.PI / L.port_sides);
      const offset = kOut + L.plug_gap_mm + 0.5 * dia;
      const hy = Math.max(kOut, 0.5 * dia);
      return [["footprint", rect(-kOut, -hy, offset + 0.5 * dia, hy)]];
    }
    case "drawer": {
      const hw = 0.5 * n("width_mm");
      const hd = 0.5 * n("depth_mm");
      const ext = p.handle === "bar" ? n("handle_mm") + L.bar_diameter_mm : p.handle === "knob" ? n("handle_mm") + L.knob_head_mm : 0;
      const front = hd + L.drawer_front_mm + ext;
      const travel = n("travel_mm");
      return [
        ["footprint", rect(-hw, -front, hw, hd)],
        ["swing", rect(-hw, -front - travel, hw, hd - travel)],
      ];
    }
    case "hinged_lid": {
      const w = n("width_mm");
      const d = n("depth_mm");
      const t = n("thickness_mm");
      const stop = n("max_angle_deg") * (Math.PI / 180);
      const handle = p.feature === "handle" ? n("feature_mm") + L.bar_diameter_mm : 0;
      if (p.orientation === "lid") {
        const out: Region[] = [["footprint", rect(-0.5 * w, -0.5 * d, 0.5 * w, 0.5 * d + t + L.lid_back_wall_mm)]];
        const beyond = -Math.cos(stop) * (d + handle);
        if (beyond > 0) out.push(["swing", rect(-0.5 * w, 0.5 * d, 0.5 * w, 0.5 * d + t + beyond)]);
        return out;
      }
      const hx = -0.5 * d;
      const [jBack, , , jY1] = L.door_jamb_mm;
      const front = 0.5 * t + handle;
      const body = rect(hx - jBack, -front, 0.5 * d, 0.5 * t + jY1);
      const r = Math.hypot(d, front);
      const sweep = Math.min(Math.PI, stop + Math.atan2(front, 0.5 * d));
      const steps = Math.max(1, Math.ceil((sweep * (180 / Math.PI)) / L.arc_step_deg));
      const arc: Polygon = [];
      for (let i = steps; i >= 0; i--) arc.push([hx + r * Math.cos((-sweep * i) / steps), r * Math.sin((-sweep * i) / steps)]);
      return [
        ["footprint", body],
        ["swing", [[hx, 0], ...arc]],
      ];
    }
    default:
      return [];
  }
}

/** `regions`: the object's footprint and swing as convex polygons in table mm. */
export function regions(catalog: Catalog, o: PlanObject): Region[] {
  return localRegions(catalog, o).map(([name, poly]) => [name, place(poly, o)]);
}

function axes(poly: Polygon): [number, number][] {
  return poly.map(([x0, y0], i) => {
    const [x1, y1] = poly[(i + 1) % poly.length] as [number, number];
    return [y0 - y1, x1 - x0];
  });
}

function intersect(a: Polygon, b: Polygon): boolean {
  for (const [ax, ay] of [...axes(a), ...axes(b)]) {
    const pa = a.map(([x, y]) => ax * x + ay * y);
    const pb = b.map(([x, y]) => ax * x + ay * y);
    if (Math.max(...pa) < Math.min(...pb) || Math.max(...pb) < Math.min(...pa)) return false;
  }
  return true;
}

function pointSegment(px: number, py: number, x0: number, y0: number, x1: number, y1: number): number {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const den = dx * dx + dy * dy;
  const u = den === 0 ? 0 : Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / den));
  return Math.hypot(px - (x0 + u * dx), py - (y0 + u * dy));
}

function distance(a: Polygon, b: Polygon): number {
  if (intersect(a, b)) return 0;
  let best = Number.POSITIVE_INFINITY;
  for (const [p, q] of [
    [a, b],
    [b, a],
  ] as [Polygon, Polygon][]) {
    for (const [px, py] of p) {
      q.forEach(([x0, y0], i) => {
        const [x1, y1] = q[(i + 1) % q.length] as [number, number];
        best = Math.min(best, pointSegment(px, py, x0, y0, x1, y1));
      });
    }
  }
  return best;
}

function lidOnBin(a: PlanObject, b: PlanObject): boolean {
  const [lid, other] = a.kind === "hinged_lid" ? [a, b] : [b, a];
  return (
    lid.kind === "hinged_lid" &&
    lid.params.orientation === "lid" &&
    other.kind === "bin" &&
    Math.abs(lid.pose.x_mm - other.pose.x_mm) < 1 &&
    Math.abs(lid.pose.y_mm - other.pose.y_mm) < 1
  );
}

/** `layout_problems`: pairs whose footprints or swings come within the clearance. */
function layoutProblems(catalog: Catalog, objects: PlanObject[], steps: PlanStep[]): string[] {
  const out: string[] = [];
  const clear = catalog.sim.layout.clearance_mm;
  const placed = objects.map((o) => regions(catalog, o));
  const targets = new Set(steps.filter((s) => s.target !== null).map((s) => `${s.object}\u0000${s.target}`));
  objects.forEach((b, j) => {
    for (let i = 0; i < j; i++) {
      const a = objects[i] as PlanObject;
      if (a.name === b.name || lidOnBin(a, b)) continue;
      let hit: [string, string, number] | null = null;
      for (const [ra, pa] of placed[i] as Region[]) {
        for (const [rb, pb] of placed[j] as Region[]) {
          const dist = distance(pa, pb);
          if (dist < clear) {
            hit = [ra, rb, dist];
            break;
          }
        }
        if (hit) break;
      }
      if (!hit) continue;
      const [ra, rb, dist] = hit;
      const verb = dist === 0 ? "overlaps" : `comes within ${fmtG(clear)} mm of`;
      const note = targets.has(`${b.name}\u0000${a.name}`) || targets.has(`${a.name}\u0000${b.name}`) ? "; a part picked into a bin starts outside it" : "";
      out.push(`objects[${j}] (${b.name}): its ${rb} ${verb} the ${ra} of objects[${i}] (${a.name}); keep them at least ${fmtG(clear)} mm apart${note}`);
    }
  });
  return out;
}

/** `_layout_warnings`: lids the simulator cannot keep open. */
function layoutWarnings(catalog: Catalog, objects: PlanObject[], steps: PlanStep[]): string[] {
  const out: string[] = [];
  const opened = new Set(steps.filter((s) => s.primitive === "hook_pull").map((s) => s.object));
  objects.forEach((o, i) => {
    if (o.kind !== "hinged_lid" || o.params.orientation !== "lid") return;
    const where = `objects[${i}] (${o.name})`;
    const stop = Number(o.params.max_angle_deg);
    if (opened.has(o.name) && stop <= 90) out.push(`${where}: a lid stays open only past vertical; with max_angle_deg ${num(stop)} it falls shut when the tool lets go`);
  });
  return out;
}

/** `_lid_problems`: a lid over a bin (same position) whose height is not the lid's fixed hinge height. */
function lidProblems(catalog: Catalog, objects: PlanObject[]): string[] {
  const out: string[] = [];
  const hinge = catalog.sim.lid_hinge_mm;
  objects.forEach((o, i) => {
    if (o.kind !== "hinged_lid" || o.params.orientation !== "lid") return;
    for (const b of objects) {
      if (b.kind !== "bin") continue;
      const same = Math.abs(b.pose.x_mm - o.pose.x_mm) < 1 && Math.abs(b.pose.y_mm - o.pose.y_mm) < 1;
      const height = Number(b.params.height_mm);
      if (same && Math.abs(height - hinge) > 1) {
        out.push(
          `objects[${i}] (${o.name}): the lid template's hinge height is fixed at ${num(hinge)} mm, so the bin ${pyRepr(b.name)} it covers must be ${num(hinge)} mm tall (it is ${num(height)} mm); if the use case states another height, put that lid on that bin in unsupported (not modelled) instead of changing it`,
        );
      }
    }
  });
  return out;
}

/**
 * `judge_args` on a plan: every problem (paths name the entry) and every warning. `uploads` are
 * the upload ids that exist, which the plan must also use.
 */
export function validatePlan(catalog: Catalog, raw: PlanArgs, uploads: string[] = []): Verdict {
  const { args, notes } = normaliseNulls(raw);
  const parsed = parseParts(catalog, args);
  const refused = args.refusal !== null && args.refusal !== undefined;
  const merge = (base: string[], extra: string[]) => [...base, ...extra.filter((p) => !base.includes(p))];
  if (parsed.problems.length) {
    let problems = explain(parsed.problems, args);
    if (!refused) problems = merge(problems, checkParts(catalog, parsed.objects, parsed.steps, uploads));
    return { status: "invalid", problems, warnings: [], notes, objects: parsed.objects, steps: parsed.steps, constructed: false };
  }
  const structural = specProblems(catalog, args, parsed.objects, parsed.steps);
  if (structural.length) {
    let problems = explain(structural, args);
    if (!refused) problems = merge(problems, checkParts(catalog, parsed.objects, parsed.steps, uploads));
    return { status: "invalid", problems, warnings: [], notes, objects: parsed.objects, steps: parsed.steps, constructed: false };
  }
  const problems: string[] = [];
  const warnings: string[] = [];
  if (refused) {
    const items = args.unsupported ?? [];
    if (!items.length) problems.push("unsupported: a refusal lists what is not simulated, why, and the nearest primitive");
    else if (items.every((u) => u.nearest_primitive === null || u.nearest_primitive === undefined)) {
      problems.push("unsupported: name the nearest primitive for at least one item (what the simulator could do instead)");
    }
  } else {
    problems.push(...checkParts(catalog, parsed.objects, parsed.steps, uploads));
    const touched = new Set<string>([...parsed.steps.map((s) => s.object), ...parsed.steps.flatMap((s) => (s.target ? [s.target] : []))]);
    parsed.objects.forEach((o, i) => {
      if (!touched.has(o.name)) warnings.push(`objects[${i}] (${o.name}): no step acts on it or places into it`);
    });
    warnings.push(...layoutWarnings(catalog, parsed.objects, parsed.steps));
  }
  const status = problems.length ? "invalid" : refused ? "refused" : "valid";
  return { status, problems, warnings, notes, objects: parsed.objects, steps: parsed.steps, constructed: true };
}

/* ------------------------------------------------------------------ fidelity, task id, export */

const TIER_ORDER: Record<Tier, number> = { rigid: 0, approximation: 1, not_simulated: 2 };

/** `TaskSpec.fidelity` and `TaskSpec.caveats` of one step. */
export function stepFidelity(catalog: Catalog, plan: { objects: PlanObject[] }, step: PlanStep): { tier: Tier; caveats: string[] } {
  const info = primitive(catalog, step.primitive);
  const names = [step.object, ...(step.target ? [step.target] : [])];
  const temps = names.map((n) => template(catalog, plan.objects.find((o) => o.name === n)?.kind ?? ""));
  const tiers: Tier[] = [info?.fidelity ?? "not_simulated", ...temps.map((t) => t?.fidelity ?? ("not_simulated" as Tier))];
  const tier = tiers.reduce<Tier>((a, b) => (TIER_ORDER[b] > TIER_ORDER[a] ? b : a), "rigid");
  const texts = [info?.caveat ?? "", ...temps.map((t) => t?.caveat ?? "")].filter(Boolean);
  return { tier, caveats: [...new Set(texts)] };
}

function canon(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canon);
  if (isObj(value)) {
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(value).sort()) out[k] = canon(value[k]);
    return out;
  }
  return value;
}

/** Python's `canonical_json`: sorted keys, compact separators, Python number forms, ASCII only. */
export function canonicalJson(value: unknown): string {
  const enc = (v: unknown): string => {
    if (v === null || v === undefined) return "null";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v === "number") return pyFloat(v);
    if (typeof v === "string") return JSON.stringify(v).replace(/[\u007f-￿]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
    if (Array.isArray(v)) return `[${v.map(enc).join(",")}]`;
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o)
      .map((k) => `${enc(k)}:${enc(o[k])}`)
      .join(",")}}`;
  };
  return enc(canon(value));
}

/** The text `TaskSpec.task_id` hashes (objects and steps only, no free text). */
export function taskIdText(contract: string, objects: PlanObject[], steps: PlanStep[]): string {
  return canonicalJson({
    contract,
    objects: objects.map((o) => ({ name: o.name, kind: o.kind, params: o.params, pose: o.pose, ref: o.ref })),
    steps: steps.map((s) => ({
      id: s.id,
      primitive: s.primitive,
      object: s.object,
      handle: s.handle,
      target: s.target,
      params: s.params,
      constraints: s.constraints,
      success: s.success,
      after: s.after,
      hold_after: s.hold_after,
    })),
  });
}

/** SHA-256 (hex) of `text` with the Web Crypto API (browsers and Node 20). */
export async function sha256Hex(text: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function taskId(catalog: Catalog, verdict: Verdict): Promise<string | null> {
  if (verdict.status !== "valid") return null;
  return sha256Hex(taskIdText(catalog.contract, verdict.objects, verdict.steps));
}

/** The exported document: `TaskSpec.to_dict()` (contract, label, task id, source, plan, fidelity). */
export function exportDocument(catalog: Catalog, plan: PlanArgs, id: string | null, source: string): Record<string, unknown> {
  const fidelity: Record<string, Tier> = {};
  for (const s of plan.steps) fidelity[s.id] = stepFidelity(catalog, plan, s).tier;
  return {
    contract: catalog.contract,
    label: catalog.label,
    task_id: id,
    source,
    title: plan.title,
    description: plan.description,
    objects: plan.objects,
    steps: plan.steps,
    assumptions: plan.assumptions,
    clarifications: plan.clarifications,
    unsupported: plan.unsupported,
    refusal: plan.refusal,
    fidelity,
  };
}

/* ------------------------------------------------------------------ edits */

/** Which entry a problem points at: `objects[2]`, `steps[1]`, `step <id>`, or the whole plan. */
export function problemTarget(problem: string, plan: PlanArgs): { kind: "object" | "step" | "plan"; index: number } {
  let m = /^objects\[(\d+)\]/.exec(problem);
  if (m) return { kind: "object", index: Number(m[1]) };
  m = /^steps\[(\d+)\]/.exec(problem);
  if (m) return { kind: "step", index: Number(m[1]) };
  m = /^step ([a-z][a-z0-9_]*)[: ]/.exec(problem);
  if (m) {
    const i = plan.steps.findIndex((s) => s.id === m![1]);
    if (i >= 0) return { kind: "step", index: i };
  }
  m = /^object ([a-z][a-z0-9_]*)[: ]/.exec(problem);
  if (m) {
    const i = plan.objects.findIndex((o) => o.name === m![1]);
    if (i >= 0) return { kind: "object", index: i };
  }
  return { kind: "plan", index: -1 };
}

function uniqueName(base: string, taken: string[]): string {
  const stem = base.replace(/[^a-z0-9_]/g, "_").replace(/^[^a-z]+/, "").slice(0, 26) || "item";
  if (!taken.includes(stem)) return stem;
  for (let i = 2; ; i++) {
    const name = `${stem}_${i}`;
    if (!taken.includes(name)) return name;
  }
}

function defaults(space: ParamDesc[]): Record<string, ParamValue> {
  return Object.fromEntries(space.map((p) => [p.name, p.default]));
}

export function moveStep(plan: PlanArgs, index: number, delta: -1 | 1): PlanArgs {
  const j = index + delta;
  if (j < 0 || j >= plan.steps.length) return plan;
  const steps = [...plan.steps];
  const [s] = steps.splice(index, 1);
  steps.splice(j, 0, s as PlanStep);
  return { ...plan, steps };
}

export function deleteStep(plan: PlanArgs, index: number): PlanArgs {
  const gone = plan.steps[index]?.id;
  const steps = plan.steps.filter((_, i) => i !== index).map((s) => (gone && s.after.includes(gone) ? { ...s, after: s.after.filter((a) => a !== gone) } : s));
  return { ...plan, steps };
}

/** A new object of `kind` with its template defaults, placed clear of the others. */
export function addObject(catalog: Catalog, plan: PlanArgs, kind: string): PlanArgs {
  const t = template(catalog, kind);
  if (!t) return plan;
  const name = uniqueName(kind, plan.objects.map((o) => o.name));
  const params = defaults(t.params);
  if (kind === "part") params.shape = "box";
  const slot = plan.objects.length;
  const base: PlanObject = { name, kind, label: t.display_name.slice(0, 60), params, pose: { x_mm: -240 + 120 * (slot % 5), y_mm: slot < 5 ? -220 : 220, yaw_deg: 0 }, ref: null };
  // the first spot on a 60 mm grid where its footprint and swing keep the clearance from every other object
  const clear = catalog.sim.layout.clearance_mm;
  const others = plan.objects.flatMap((o) => regions(catalog, o).map(([, poly]) => poly));
  let pose: Pose = base.pose;
  search: for (let y = -240; y <= 240; y += 60) {
    for (let x = -240; x <= 240; x += 60) {
      const mine = regions(catalog, { ...base, pose: { x_mm: x, y_mm: y, yaw_deg: 0 } }).map(([, poly]) => poly);
      if (mine.every((a) => others.every((b) => distance(a, b) >= clear))) {
        pose = { x_mm: x, y_mm: y, yaw_deg: 0 };
        break search;
      }
    }
  }
  return { ...plan, objects: [...plan.objects, { ...base, pose }], refusal: null };
}

export function deleteObject(plan: PlanArgs, index: number): PlanArgs {
  return { ...plan, objects: plan.objects.filter((_, i) => i !== index) };
}

/** A new step of `prim` on the first object it can act on, with a goal criterion. */
export function addStep(catalog: Catalog, plan: PlanArgs, prim: string): PlanArgs {
  const info = primitive(catalog, prim);
  if (!info) return plan;
  const allowed = info.objects.filter((k) => !(catalog.sim.picks.includes(prim) && catalog.sim.no_pick.includes(k)));
  const obj = plan.objects.find((o) => allowed.includes(o.kind));
  const kind = obj?.kind ?? allowed[0] ?? "part";
  const t = template(catalog, kind);
  const goal = [...(t?.quantities ?? []), ...catalog.goal_extra].find((q) => measurable(catalog, prim, kind).includes(q)) ?? info.quantities[0] ?? "force_n";
  const q = quantity(catalog, goal);
  const success: Criterion[] = [{ quantity: goal, at_least: q && q.low < 1 && q.high >= 1 ? 1 : null, at_most: q && !(q.low < 1 && q.high >= 1) ? q.high : null }];
  const id = uniqueName(`${prim}_${obj?.name ?? "step"}`, plan.steps.map((s) => s.id));
  const step: PlanStep = {
    id,
    primitive: prim,
    object: obj?.name ?? "",
    handle: handlesFor(catalog, prim, kind)[0] ?? info.handles[0] ?? "",
    target: null,
    description: `${prim.replace("_", " ")} the ${(obj?.label ?? "object").toLowerCase()}.`.replace(/^./, (c) => c.toUpperCase()),
    params: { ...defaults(info.params), ...(prim === "twist" && catalog.sim.twist_sense[kind] ? { direction: catalog.sim.twist_sense[kind] } : {}) },
    constraints: { max_force_n: null, max_torque_nm: null, precision_mm: null, precision_deg: null },
    success,
    after: [],
    hold_after: false,
  };
  return { ...plan, steps: [...plan.steps, step], refusal: null };
}

export function emptyPlan(): PlanArgs {
  return { title: "New use case", description: "", objects: [], steps: [], assumptions: [], clarifications: [], unsupported: [], refusal: null };
}

/** The tool arguments of a recorded or returned plan document (`TaskSpec.to_dict()`). */
export function planArgsOf(doc: Record<string, unknown>): PlanArgs {
  const keys = ["title", "description", "objects", "steps", "assumptions", "clarifications", "unsupported", "refusal"] as const;
  return structuredClone(Object.fromEntries(keys.map((k) => [k, doc[k] ?? (k === "refusal" ? null : k === "title" || k === "description" ? "" : [])]))) as unknown as PlanArgs;
}
