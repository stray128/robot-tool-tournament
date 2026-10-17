/**
 * Runtime validation of the site's JSON data (see lib/types.ts).
 *
 * Every parser takes `unknown` and either returns a typed value or throws a
 * DataError naming the JSON path that is wrong, e.g.
 * `bouts.json: $.bouts[2].lanes[0].outcome.hold_s: expected a number, got "2"`.
 * The build calls these on every file, so bad data never reaches a page.
 */

import type {
  ArmEstimate,
  Bout,
  BoutsIndex,
  Certificate,
  ComputeProvenance,
  Design,
  DesignProvenance,
  DesignsIndex,
  FailureMode,
  Job,
  JobsIndex,
  Lane,
  ObjectSummary,
  ParamSpec,
  PhaseSpan,
  ReplayGeom,
  ReplayHeader,
  ResultSource,
  ResultsIndex,
  S1Analysis,
  S1Arm,
  S1Run,
  S1Test,
  SecondaryBlock,
  SecondaryTest,
  TaskSummary,
  Vec3,
} from "./types";

export class DataError extends Error {
  constructor(
    readonly file: string,
    readonly path: string,
    detail: string,
  ) {
    super(`${file}: ${path}: ${detail}`);
    this.name = "DataError";
  }
}

const FAILURE_MODES: readonly FailureMode[] = [
  "missed_grasp",
  "slip_during_lift",
  "ejected_on_close",
  "unstable_hold",
  "finger_env_collision",
  "not_registered",
  "physics_invalid",
  "invalid_build",
];

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "an array";
  if (typeof v === "string") return JSON.stringify(v.length > 40 ? `${v.slice(0, 40)}...` : v);
  return typeof v === "object" ? "an object" : String(v);
}

/** A cursor over one JSON value that knows its path, for precise error messages. */
export class Node {
  constructor(
    readonly file: string,
    readonly path: string,
    readonly value: unknown,
  ) {}

  fail(detail: string): never {
    throw new DataError(this.file, this.path, detail);
  }

  private expect(what: string): never {
    this.fail(`expected ${what}, got ${describe(this.value)}`);
  }

  get(key: string): Node {
    const obj = this.object();
    return new Node(this.file, `${this.path}.${key}`, obj[key]);
  }

  has(key: string): boolean {
    return Object.prototype.hasOwnProperty.call(this.object(), key) && this.object()[key] !== undefined;
  }

  object(): Record<string, unknown> {
    if (typeof this.value !== "object" || this.value === null || Array.isArray(this.value)) this.expect("an object");
    return this.value as Record<string, unknown>;
  }

  items(): Node[] {
    if (!Array.isArray(this.value)) this.expect("an array");
    return (this.value as unknown[]).map((v, i) => new Node(this.file, `${this.path}[${i}]`, v));
  }

  entries(): [string, Node][] {
    return Object.keys(this.object()).map((k) => [k, this.get(k)]);
  }

  str(): string {
    if (typeof this.value !== "string") this.expect("a string");
    return this.value as string;
  }

  nonEmpty(): string {
    const s = this.str();
    if (s.length === 0) this.fail("expected a non-empty string");
    return s;
  }

  num(): number {
    if (typeof this.value !== "number" || !Number.isFinite(this.value)) this.expect("a finite number");
    return this.value as number;
  }

  int(): number {
    const n = this.num();
    if (!Number.isInteger(n)) this.fail(`expected an integer, got ${n}`);
    return n;
  }

  nonNeg(): number {
    const n = this.num();
    if (n < 0) this.fail(`expected a number >= 0, got ${n}`);
    return n;
  }

  unit(): number {
    const n = this.num();
    if (n < 0 || n > 1) this.fail(`expected a number in [0, 1], got ${n}`);
    return n;
  }

  bool(): boolean {
    if (typeof this.value !== "boolean") this.expect("a boolean");
    return this.value as boolean;
  }

  oneOf<T extends string>(options: readonly T[]): T {
    const s = this.str();
    if (!options.includes(s as T)) this.fail(`expected one of ${options.join(", ")}, got ${describe(s)}`);
    return s as T;
  }

  literal<T extends string | number>(expected: T): T {
    if (this.value !== expected) this.fail(`expected ${JSON.stringify(expected)}, got ${describe(this.value)}`);
    return expected;
  }

  nullable<T>(read: (n: Node) => T): T | null {
    return this.value === null || this.value === undefined ? null : read(this);
  }

  nums(length?: number): number[] {
    const out = this.items().map((n) => n.num());
    if (length !== undefined && out.length !== length) this.fail(`expected ${length} numbers, got ${out.length}`);
    return out;
  }

  strs(): string[] {
    return this.items().map((n) => n.str());
  }

  isoTime(): string {
    const s = this.str();
    if (Number.isNaN(Date.parse(s))) this.fail(`expected an ISO 8601 time, got ${describe(s)}`);
    return s;
  }
}

function uniqueIds<T extends { id: string }>(root: Node, list: T[], what: string): T[] {
  const seen = new Set<string>();
  for (const item of list) {
    if (seen.has(item.id)) root.fail(`duplicate ${what} id ${JSON.stringify(item.id)}`);
    seen.add(item.id);
  }
  return list;
}

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;

function slug(n: Node): string {
  const s = n.nonEmpty();
  if (!SLUG.test(s)) n.fail(`expected a URL-safe id (lowercase letters, digits, . _ -), got ${describe(s)}`);
  return s;
}

/** A path under /data: relative, no "..", no scheme. */
function dataPath(n: Node): string {
  const s = n.nonEmpty();
  if (s.startsWith("/") || s.includes("..") || /^[a-z]+:/i.test(s)) n.fail(`expected a relative data path, got ${describe(s)}`);
  return s;
}

// ---------------------------------------------------------------- provenance

function readCompute(n: Node): ComputeProvenance {
  const kind = n.get("kind").oneOf(["local", "nebius"] as const);
  if (kind === "local") {
    return {
      kind,
      executor: n.get("executor").nonEmpty(),
      engine: n.get("engine").nonEmpty(),
      host: n.get("host").nonEmpty(),
      wall_s: n.get("wall_s").nullable((x) => x.nonNeg()),
      sim_steps: n.get("sim_steps").nullable((x) => x.int()),
      note: n.get("note").nullable((x) => x.str()),
    };
  }
  return {
    kind,
    job_id: n.get("job_id").nonEmpty(),
    platform: n.get("platform").nonEmpty(),
    preset: n.get("preset").nonEmpty(),
    region: n.get("region").nonEmpty(),
    executor: n.get("executor").nonEmpty(),
    engine: n.get("engine").nonEmpty(),
    wall_s: n.get("wall_s").nonNeg(),
    cost_usd: n.get("cost_usd").nonNeg(),
    image_digest: n.get("image_digest").nullable((x) => x.str()),
  };
}

function readDesignProvenance(n: Node): DesignProvenance {
  const kind = n.get("kind").oneOf(["manual", "random", "llm"] as const);
  if (kind === "manual") return { kind, note: n.get("note").nullable((x) => x.str()) };
  if (kind === "random") return { kind, sampler: n.get("sampler").nonEmpty(), seed: n.get("seed").int() };
  return {
    kind,
    model: n.get("model").nonEmpty(),
    provider: n.get("provider").nonEmpty(),
    tokens_in: n.get("tokens_in").int(),
    tokens_out: n.get("tokens_out").int(),
    latency_s: n.get("latency_s").nonNeg(),
    cost_usd: n.get("cost_usd").nonNeg(),
    round: n.get("round").nullable((x) => x.int()),
  };
}

// ---------------------------------------------------------------- bouts

const PHASES = ["settle", "descend", "close", "lift", "hold"] as const;

function readTask(n: Node): TaskSummary {
  const phases: PhaseSpan[] = n
    .get("phases")
    .items()
    .map((p) => ({
      name: p.get("name").oneOf(PHASES),
      start_s: p.get("start_s").nonNeg(),
      end_s: p.get("end_s").nullable((x) => x.nonNeg()),
    }));
  if (phases.length === 0) n.get("phases").fail("expected at least one phase");
  return {
    name: n.get("name").nonEmpty(),
    phases,
    success_lift_m: n.get("success_lift_m").nonNeg(),
    success_hold_s: n.get("success_hold_s").nonNeg(),
    record_tail_s: n.get("record_tail_s").nonNeg(),
    max_time_s: n.get("max_time_s").nonNeg(),
    grip_force_n: n.get("grip_force_n").nonNeg(),
    max_opening_mm: n.get("max_opening_mm").nonNeg(),
  };
}

function readObject(n: Node): ObjectSummary {
  const kind = n.get("kind").oneOf(["box", "cylinder", "capsule", "part"] as const);
  const size_mm = n.get("size_mm").nums(kind === "box" || kind === "part" ? 3 : 2);
  if (kind !== "part") return { kind, size_mm };
  // A v2 part: the id must be the replay header's meta.part_id (checked against the file by lib/data.ts).
  return { kind, size_mm, part_id: n.get("part_id").nonEmpty(), family: n.get("family").nonEmpty() };
}

function readLane(n: Node): Lane {
  const o = n.get("outcome");
  const success = o.get("success").bool();
  const failure_mode = o.get("failure_mode").nullable((x) => x.oneOf(FAILURE_MODES));
  if (success === (failure_mode !== null)) o.fail("a success has no failure_mode and a failure has exactly one");
  return {
    id: slug(n.get("id")),
    replay: dataPath(n.get("replay")),
    label: n.get("label").nonEmpty(),
    condition: n.get("condition").str(),
    entrant: n.get("entrant").nonEmpty(),
    design_id: slug(n.get("design_id")),
    object: readObject(n.get("object")),
    controller_setting: n.get("controller_setting").nullable((x) => x.int()),
    grasp_height_frac: n.get("grasp_height_frac").nullable((x) => x.unit()),
    seed: n.get("seed").int(),
    n_frames: n.get("n_frames").int(),
    duration_s: n.get("duration_s").nonNeg(),
    outcome: {
      success,
      failure_mode,
      max_lift_mm: o.get("max_lift_mm").num(),
      hold_s: o.get("hold_s").nonNeg(),
      flags: o.get("flags").strs(),
      source: o.get("source").oneOf(["scorer", "replay-frames"] as const),
      wall_s: o.get("wall_s").nullable((x) => x.nonNeg()),
      sim_steps: o.get("sim_steps").nullable((x) => x.int()),
    },
    design_provenance: readDesignProvenance(n.get("design_provenance")),
  };
}

function readBout(n: Node): Bout {
  const lanes = n.get("lanes").items().map(readLane);
  if (lanes.length === 0) n.get("lanes").fail("a bout needs at least one lane");
  uniqueIds(n.get("lanes"), lanes, "lane");
  return {
    id: slug(n.get("id")),
    source: n.get("source").nonEmpty(),
    kind: n.get("kind").oneOf(["gate", "training", "heldout", "live"] as const),
    title: n.get("title").nonEmpty(),
    summary: n.get("summary").str(),
    featured: n.get("featured").bool(),
    recorded_at: n.get("recorded_at").nullable((x) => x.isoTime()),
    task: readTask(n.get("task")),
    compute: readCompute(n.get("compute")),
    lanes,
  };
}

export function parseBouts(raw: unknown, file = "bouts.json"): BoutsIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal("rtt-web-bouts");
  root.get("version").literal(1);
  const bouts = uniqueIds(root.get("bouts"), root.get("bouts").items().map(readBout), "bout");
  return { schema: "rtt-web-bouts", version: 1, bouts };
}

// ---------------------------------------------------------------- designs

function readParamSpec(n: Node): ParamSpec {
  const spec: ParamSpec = { description: n.get("description").str() };
  if (n.has("options")) spec.options = n.get("options").strs();
  if (n.has("min")) spec.min = n.get("min").num();
  if (n.has("max")) spec.max = n.get("max").num();
  return spec;
}

function readCertificate(n: Node): Certificate {
  return {
    status: n.get("status").oneOf(["pending", "issued"] as const),
    issued_at: n.get("issued_at").nullable((x) => x.isoTime()),
    checks: n
      .get("checks")
      .items()
      .map((c) => ({
        name: c.get("name").nonEmpty(),
        condition: c.get("condition").str(),
        success_rate: c.get("success_rate").nullable((x) => x.unit()),
        episodes: c.get("episodes").int(),
      })),
  };
}

function readDesign(n: Node): Design {
  const params: Record<string, number | string> = {};
  for (const [k, v] of n.get("params").entries()) {
    params[k] = typeof v.value === "string" ? v.str() : v.num();
  }
  const files: Design["files"] = {};
  const f = n.get("files");
  for (const key of ["stl_left", "stl_right", "glb", "mount_plate", "certificate_html"] as const) {
    if (f.has(key)) files[key] = dataPath(f.get(key));
  }
  return {
    id: slug(n.get("id")),
    source: n.get("source").nonEmpty(),
    name: n.get("name").nonEmpty(),
    origin: n.get("origin").oneOf(["manual", "nemotron", "random", "zeroshot"] as const),
    rationale: n.get("rationale").nullable((x) => x.str()),
    params,
    valid: n.get("valid").bool(),
    errors: n.get("errors").strs(),
    warnings: n.get("warnings").strs(),
    mass_printed_g: n.get("mass_printed_g").nullable((x) => x.nonNeg()),
    mass_sim_g: n.get("mass_sim_g").nullable((x) => x.nonNeg()),
    bbox_mm: n.get("bbox_mm").nullable((x) => x.nums(3) as Vec3),
    watertight: n.get("watertight").nullable((x) => x.bool()),
    n_boxes: n.get("n_boxes").nullable((x) => x.int()),
    files,
    provenance: readDesignProvenance(n.get("provenance")),
    bouts: n.get("bouts").strs(),
    certificate: n.get("certificate").nullable(readCertificate),
  };
}

export function parseDesigns(raw: unknown, file = "designs.json"): DesignsIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal("rtt-web-designs");
  root.get("version").literal(1);
  const params: Record<string, ParamSpec> = {};
  for (const [k, v] of root.get("params").entries()) params[k] = readParamSpec(v);
  const designs = uniqueIds(root.get("designs"), root.get("designs").items().map(readDesign), "design");
  return { schema: "rtt-web-designs", version: 1, params, designs };
}

// ---------------------------------------------------------------- jobs

function readJob(n: Node): Job {
  const started_at = n.get("started_at").isoTime();
  const ended_at = n.get("ended_at").isoTime();
  if (Date.parse(ended_at) < Date.parse(started_at)) n.fail("ended_at is before started_at");
  return {
    id: n.get("id").nonEmpty(),
    source: n.get("source").nonEmpty(),
    name: n.get("name").nonEmpty(),
    purpose: n.get("purpose").str(),
    kind: n.get("kind").oneOf(["local", "nebius"] as const),
    executor: n.get("executor").nonEmpty(),
    platform: n.get("platform").nullable((x) => x.str()),
    preset: n.get("preset").nullable((x) => x.str()),
    region: n.get("region").nullable((x) => x.str()),
    status: n.get("status").oneOf(["completed", "failed", "cancelled", "running"] as const),
    started_at,
    ended_at,
    wall_s: n.get("wall_s").nonNeg(),
    cost_usd: n.get("cost_usd").nullable((x) => x.nonNeg()),
    episodes: n.get("episodes").nullable((x) => x.int()),
    sim_steps: n.get("sim_steps").nullable((x) => x.int()),
    bouts: n.get("bouts").strs(),
  };
}

export function parseJobs(raw: unknown, file = "jobs.json"): JobsIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal("rtt-web-jobs");
  root.get("version").literal(1);
  const jobs = uniqueIds(root.get("jobs"), root.get("jobs").items().map(readJob), "job");
  return { schema: "rtt-web-jobs", version: 1, jobs };
}

// ---------------------------------------------------------------- results

function readArm(n: Node): ArmEstimate {
  const ci = n.get("ci95").nums(2) as [number, number];
  if (ci[0] > ci[1]) n.get("ci95").fail("lower bound is above the upper bound");
  return {
    id: slug(n.get("id")),
    label: n.get("label").nonEmpty(),
    runs: n.get("runs").items().map((x) => x.unit()),
    mean: n.get("mean").unit(),
    ci95: ci,
  };
}

/** A best-so-far curve: values in [0, 1], never decreasing, at most `budget` long. */
function readBest(n: Node, budget: number): number[] {
  const best = n.items().map((x) => x.unit());
  if (best.length > budget) n.fail(`more points (${best.length}) than the budget (${budget})`);
  best.forEach((v, i) => {
    if (i > 0 && v < best[i - 1]!) n.fail(`best-so-far decreases at design ${i + 1} (${best[i - 1]} -> ${v})`);
  });
  return best;
}

/**
 * S1's T of one run: between `countFrom` and the run length plus one, and consistent
 * with the curve (a run that reached the threshold at design T has a best-so-far at or
 * above it there).
 */
function readT(n: Node, best: number[], countFrom: number, threshold: number | null): number {
  const t = n.int();
  if (t < countFrom || t > best.length + 1) n.fail(`expected ${countFrom}..${best.length + 1}, got ${t}`);
  if (threshold === null) n.fail("a run has T but the threshold is not set");
  if (t <= best.length && best[t - 1]! < threshold - 1e-9) n.fail(`best-so-far at design ${t} is below the threshold`);
  return t;
}

function readS1Arm(n: Node, budget: number, version: 1 | 2, countFrom = 1, threshold: number | null = null): S1Arm {
  return {
    id: slug(n.get("id")),
    label: n.get("label").nonEmpty(),
    runs: n
      .get("runs")
      .items()
      .map((r) => {
        const best = readBest(r.get("best"), budget);
        const run: S1Run = {
          run: r.get("run").nonEmpty(),
          best,
          job_id: version === 2 && r.has("job_id") ? r.get("job_id").nullable((x) => x.nonEmpty()) : null,
        };
        if (version === 2 && r.has("t")) run.t = readT(r.get("t"), best, countFrom, threshold);
        return run;
      }),
  };
}

function readS1Test(n: Node): S1Test {
  return { statistic: n.get("statistic").num(), p_value: n.get("p_value").unit(), method: n.get("method").nonEmpty() };
}

function readS1Analysis(n: Node): S1Analysis {
  const effect = n.get("effect");
  const ci = { low: effect.get("ci_low").num(), high: effect.get("ci_high").num() };
  if (ci.low > ci.high) effect.fail("ci_low is above ci_high");
  const lift = n.get("lift_only");
  return {
    source: n.get("source").nonEmpty(),
    tau_source: n.get("tau_source").str(),
    primary: readS1Test(n.get("primary")),
    effect: { diff: effect.get("diff").num(), ci_low: ci.low, ci_high: ci.high, level: effect.get("level").unit() },
    logrank: readS1Test(n.get("logrank")),
    lift_only: lift.nullable((x) => ({
      status: x.get("status").nonEmpty(),
      diff: x.get("diff").num(),
      p_value: x.get("p_value").unit(),
      method: x.get("method").nonEmpty(),
    })),
  };
}

function readSecondaryTest(n: Node): SecondaryTest {
  const unit = n.get("unit").oneOf(["rate", "score"] as const);
  return {
    id: n.get("id").nonEmpty(),
    label: n.get("label").nonEmpty(),
    unit,
    mean_a: n.get("mean_a").unit(),
    mean_c: n.get("mean_c").unit(),
    diff: n.get("diff").num(),
    p_value: n.get("p_value").unit(),
    p_holm: n.get("p_holm").nullable((x) => x.unit()),
    rejected: n.get("rejected").nullable((x) => x.bool()),
    method: n.get("method").nonEmpty(),
    n_a: n.get("n_a").int(),
    n_c: n.get("n_c").int(),
  };
}

function readSecondary(n: Node): SecondaryBlock {
  const tests = uniqueIds(n, n.get("tests").items().map(readSecondaryTest), "test");
  const family = n.get("family").strs();
  for (const id of family) if (!tests.some((t) => t.id === id)) n.get("family").fail(`no test ${id}`);
  const job = n.get("job");
  return {
    source: n.get("source").nonEmpty(),
    alpha: n.get("alpha").unit(),
    family,
    tests,
    width_bands: n.get("width_bands").nullable((w) => ({
      note: w.get("note").str(),
      edge_mm: w.get("edge_mm").nonNeg(),
      bands: w
        .get("bands")
        .items()
        .map((b) => ({
          id: b.get("id").nonEmpty(),
          label: b.get("label").nonEmpty(),
          objects: b.get("objects").int(),
          mean_a: b.get("mean_a").unit(),
          mean_c: b.get("mean_c").unit(),
          p_value: b.get("p_value").unit(),
        })),
    })),
    job: job.nullable((j) => ({
      job_id: j.get("job_id").nonEmpty(),
      platform: j.get("platform").nullable((x) => x.str()),
      region: j.get("region").nullable((x) => x.str()),
      run_s: j.get("run_s").nullable((x) => x.nonNeg()),
    })),
  };
}

function readReferences(n: Node): { label: string; value: number }[] {
  return n.items().map((r) => ({ label: r.get("label").nonEmpty(), value: r.get("value").unit() }));
}

function readSource(n: Node): ResultSource {
  return {
    arm: n.get("arm").nonEmpty(),
    run: n.get("run").nonEmpty(),
    kind: n.get("kind").oneOf(["nebius", "local"] as const),
    job_id: n.get("job_id").nullable((x) => x.nonEmpty()),
    platform: n.get("platform").nullable((x) => x.str()),
    preset: n.get("preset").nullable((x) => x.str()),
    region: n.get("region").nullable((x) => x.str()),
    executor: n.get("executor").nullable((x) => x.str()),
    wall_s: n.get("wall_s").nullable((x) => x.nonNeg()),
    cost_usd: n.get("cost_usd").nullable((x) => x.nonNeg()),
    llm_calls: n.has("llm_calls") ? n.get("llm_calls").nullable((x) => x.nonNeg()) : null,
    llm_usd: n.has("llm_usd") ? n.get("llm_usd").nullable((x) => x.nonNeg()) : null,
    run_s: n.has("run_s") ? n.get("run_s").nullable((x) => x.nonNeg()) : null,
    image_digest: n.has("image_digest") ? n.get("image_digest").nullable((x) => x.nonEmpty()) : null,
  };
}

/**
 * Validate results.json. Version 2 is current (S1 primary, H1 secondary); a version 1
 * file (`forest` + `curves`, from before the endpoint change) is upgraded: its curves
 * become S1 with no threshold and its forest becomes H1. A final file may carry the
 * confirmatory S1 comparison (`s1.analysis`) and, in place of `h1`, the pre-registered
 * secondary endpoints (`secondary`).
 */
export function parseResults(raw: unknown, file = "results.json"): ResultsIndex {
  const root = new Node(file, "$", raw);
  root.get("schema").literal("rtt-web-results");
  const version = root.get("version").value;
  const status = root.get("status").oneOf(["synthetic", "pilot", "final"] as const);
  const note = root.get("note").str();
  if (version === 1) {
    const forest = root.get("forest");
    const curves = root.get("curves");
    const budget = curves.get("budget").int();
    return {
      schema: "rtt-web-results",
      version: 2,
      status,
      note,
      s1: {
        endpoint: curves.get("endpoint").nonEmpty(),
        threshold: null,
        threshold_note: "No threshold in this version 1 file",
        budget,
        shared_prefix: 0,
        count_from: 1,
        arms: uniqueIds(curves, curves.get("arms").items().map((a) => readS1Arm(a, budget, 1)), "arm"),
      },
      h1: {
        endpoint: forest.get("endpoint").nonEmpty(),
        status: "available",
        arms: uniqueIds(forest, forest.get("arms").items().map(readArm), "arm"),
        references: readReferences(forest.get("references")),
      },
      sources: [],
    };
  }
  root.get("version").literal(2);
  const s1 = root.get("s1");
  const budget = s1.get("budget").int();
  if (budget < 1) s1.get("budget").fail("budget must be at least 1");
  const shared = s1.get("shared_prefix").int();
  if (shared < 0 || shared > budget) s1.get("shared_prefix").fail(`expected 0..${budget}, got ${shared}`);
  const countFrom = s1.has("count_from") ? s1.get("count_from").int() : 1;
  if (countFrom < 1 || countFrom > budget) s1.get("count_from").fail(`expected 1..${budget}, got ${countFrom}`);
  const threshold = s1.get("threshold").nullable((x) => x.unit());
  const secondary = root.has("secondary") ? readSecondary(root.get("secondary")) : null;
  if (secondary && status !== "final") root.get("secondary").fail("secondary endpoints belong in a final file");
  let h1: ResultsIndex["h1"] = null;
  if (root.has("h1") || !secondary) {
    const h = root.get("h1");
    const h1Status = h.get("status").oneOf(["not_run", "available"] as const);
    const h1Arms = uniqueIds(h, h.get("arms").items().map(readArm), "arm");
    if (h1Status === "not_run" && h1Arms.length > 0) h.get("arms").fail("arms must be empty while status is not_run");
    h1 = { endpoint: h.get("endpoint").nonEmpty(), status: h1Status, arms: h1Arms, references: readReferences(h.get("references")) };
  }
  const doc: ResultsIndex = {
    schema: "rtt-web-results",
    version: 2,
    status,
    note,
    s1: {
      endpoint: s1.get("endpoint").nonEmpty(),
      threshold,
      threshold_note: s1.get("threshold_note").str(),
      budget,
      shared_prefix: shared,
      count_from: countFrom,
      arms: uniqueIds(s1, s1.get("arms").items().map((a) => readS1Arm(a, budget, 2, countFrom, threshold)), "arm"),
    },
    h1,
    sources: root.has("sources") ? root.get("sources").items().map(readSource) : [],
  };
  if (s1.has("trials")) {
    const trials = s1.get("trials").int();
    if (trials < 1) s1.get("trials").fail("expected at least 1");
    doc.s1.trials = trials;
  }
  if (s1.has("analysis")) {
    if (status !== "final") s1.get("analysis").fail("the confirmatory analysis belongs in a final file");
    if (doc.s1.arms.some((a) => a.runs.some((r) => r.t === undefined))) s1.get("analysis").fail("every run needs its T");
    doc.s1.analysis = readS1Analysis(s1.get("analysis"));
  }
  if (secondary) doc.secondary = secondary;
  if (root.has("protocol")) {
    const p = root.get("protocol");
    doc.protocol = { tag: p.get("tag").nonEmpty(), results_doc: dataPath(p.get("results_doc")), protocol_doc: dataPath(p.get("protocol_doc")) };
  }
  return doc;
}

// ---------------------------------------------------------------- replay headers

/** Validate one v1-style replay geom (shared with the version 2 header parser in lib/replay.ts). */
export function readGeom(n: Node): ReplayGeom {
  const type = n.get("type").oneOf(["box", "cylinder", "capsule", "plane", "mesh"] as const);
  const geom: ReplayGeom = {
    name: n.get("name").str(),
    type,
    pos: n.get("pos").nums(3) as ReplayGeom["pos"],
    quat_xyzw: n.get("quat_xyzw").nums(4) as ReplayGeom["quat_xyzw"],
    rgba: n.get("rgba").nums(4) as ReplayGeom["rgba"],
  };
  if (type === "box") geom.half_size = n.get("half_size").nums(3);
  if (type === "plane") geom.half_size = n.get("half_size").nums(2);
  if (type === "cylinder") {
    geom.radius = n.get("radius").nonNeg();
    geom.half_height = n.get("half_height").nonNeg();
  }
  if (type === "capsule") {
    geom.radius = n.get("radius").nonNeg();
    geom.half_length = n.get("half_length").nonNeg();
  }
  if (type === "mesh") {
    const vertices = n.get("vertices").nums();
    const faces = n.get("faces").items().map((x) => x.int());
    if (vertices.length % 3 !== 0) n.get("vertices").fail(`expected x, y, z triples, got ${vertices.length} numbers`);
    if (faces.length % 3 !== 0) n.get("faces").fail(`expected index triples, got ${faces.length} indices`);
    const nv = vertices.length / 3;
    if (faces.some((i) => i < 0 || i >= nv)) n.get("faces").fail(`face index outside 0..${nv - 1}`);
    geom.vertices = vertices;
    geom.faces = faces;
  }
  return geom;
}

/** Validate a replay header (`<stem>.json`, format rtt-replay v1). */
export function parseReplayHeader(raw: unknown, file = "replay.json"): ReplayHeader {
  const root = new Node(file, "$", raw);
  root.get("format").literal("rtt-replay");
  root.get("version").literal(1);
  root.get("dtype").literal("float32");
  root.get("byte_order").literal("little");
  root.get("up_axis").literal("z");
  root.get("units").literal("m");
  const fields = root.get("pose_fields").strs().join(",");
  if (fields !== "px,py,pz,qx,qy,qz,qw") root.get("pose_fields").fail(`unsupported pose layout ${fields}`);
  const bodies = root
    .get("bodies")
    .items()
    .map((b) => ({ name: b.get("name").nonEmpty(), geoms: b.get("geoms").items().map(readGeom) }));
  const n_bodies = root.get("n_bodies").int();
  if (n_bodies !== bodies.length) root.get("n_bodies").fail(`header lists ${bodies.length} bodies, n_bodies is ${n_bodies}`);
  const fps = root.get("fps").num();
  if (fps <= 0) root.get("fps").fail("fps must be positive");
  return {
    format: "rtt-replay",
    version: 1,
    fps,
    dtype: "float32",
    byte_order: "little",
    up_axis: "z",
    units: "m",
    bodies,
    static: root.get("static").items().map(readGeom),
    n_frames: root.get("n_frames").int(),
    n_bodies,
    bin: root.get("bin").nonEmpty(),
    meta: root.has("meta") ? root.get("meta").object() : {},
  };
}
