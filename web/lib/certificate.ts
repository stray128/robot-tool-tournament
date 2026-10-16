/**
 * Robustness certificates as written by `rtt certify --out` (schema
 * `rtt.certificates/1`, src/rtt/sim/certify.py), published under
 * public/data/certificates/<id>.json, and the views the /certificate page draws.
 *
 * A file holds one design's certificates, one per object. Each certificate is a fixed
 * battery: the nominal grasp, placement errors of +-1, +-2 and +-3 mm along x (finger
 * width) and y (closing axis), a shake along each axis and a push along x and y. The
 * importer (scripts/import-certificate.mjs) may add a `provenance` block saying where
 * it ran; the page says so plainly when it is missing.
 */

import type { ComputeProvenance } from "./types";
import { Node } from "./validate";

export type CertCategory = "placement" | "shake" | "push";

export interface CertTest {
  name: string;
  category: CertCategory;
  /** Placement: axis "x"/"y" (null for nominal) and signed error in mm. Shake/push: their settings. */
  params: Record<string, number | string | boolean | null>;
  success: boolean;
  failure_mode: string | null;
  max_lift_m: number;
  hold_time_s: number;
  flags: string[];
}

export interface ObjectCertificate {
  object: string;
  controller_setting: number;
  task: string;
  task_digest: string;
  score: number;
  category_scores: Partial<Record<CertCategory, number>>;
  tolerance_mm: { x: number | null; y: number | null };
  passed: string;
  settings: Record<string, unknown>;
  tests: CertTest[];
}

export interface CertificateDoc {
  id: string;
  design_id: string;
  design_name: string | null;
  design: Record<string, number | string>;
  issued_at: string | null;
  provenance: ComputeProvenance | null;
  certificates: ObjectCertificate[];
}

const CATEGORIES: readonly CertCategory[] = ["placement", "shake", "push"];

function readParams(n: Node): Record<string, number | string | boolean | null> {
  const out: Record<string, number | string | boolean | null> = {};
  for (const [k, v] of n.entries()) {
    const x = v.value;
    if (x === null || typeof x === "number" || typeof x === "string" || typeof x === "boolean") out[k] = x;
  }
  return out;
}

function readTest(n: Node): CertTest {
  const category = n.get("category").oneOf(CATEGORIES);
  const params = readParams(n.get("params"));
  if (category === "placement" && typeof params.error_mm !== "number") n.get("params").fail("placement test needs a numeric error_mm");
  const success = n.get("success").bool();
  return {
    name: n.get("name").nonEmpty(),
    category,
    params,
    success,
    failure_mode: n.get("failure_mode").nullable((x) => x.str()),
    max_lift_m: n.get("max_lift_m").num(),
    hold_time_s: n.get("hold_time_s").nonNeg(),
    flags: n.has("flags") ? n.get("flags").strs() : [],
  };
}

function readObjectCert(n: Node): ObjectCertificate {
  const scores: Partial<Record<CertCategory, number>> = {};
  for (const [k, v] of n.get("category_scores").entries()) {
    if ((CATEGORIES as readonly string[]).includes(k)) scores[k as CertCategory] = v.unit();
  }
  const tol = n.get("tolerance_mm");
  const tests = n.get("tests").items().map(readTest);
  if (!tests.some((t) => t.name === "nominal")) n.get("tests").fail("no nominal test");
  return {
    object: n.get("object").nonEmpty(),
    controller_setting: n.get("controller_setting").int(),
    task: n.get("task").nonEmpty(),
    task_digest: n.get("task_digest").str(),
    score: n.get("score").unit(),
    category_scores: scores,
    tolerance_mm: { x: tol.get("x").nullable((x) => x.nonNeg()), y: tol.get("y").nullable((x) => x.nonNeg()) },
    passed: n.get("passed").str(),
    settings: n.has("settings") ? n.get("settings").object() : {},
    tests,
  };
}

function readProvenance(n: Node): ComputeProvenance {
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

/**
 * Validate a certificate file. Accepts the `rtt certify` document (`certificates`
 * list) or a single certificate object (as `Certificate.to_dict()` writes it).
 * Throws a DataError naming the JSON path.
 */
export function parseCertificateDoc(raw: unknown, id: string, file = `certificates/${id}.json`): CertificateDoc {
  const root = new Node(file, "$", raw);
  root.object();
  if (root.has("schema")) root.get("schema").literal("rtt.certificates/1");
  const list = root.has("certificates") ? root.get("certificates").items().map(readObjectCert) : [readObjectCert(root)];
  if (list.length === 0) root.get("certificates").fail("expected at least one certificate");
  const design: Record<string, number | string> = {};
  if (root.has("design")) {
    for (const [k, v] of root.get("design").entries()) {
      if (typeof v.value === "number" || typeof v.value === "string") design[k] = v.value;
    }
  }
  return {
    id,
    design_id: root.get("design_id").nonEmpty(),
    design_name: typeof design.name === "string" ? design.name : null,
    design,
    issued_at: root.has("issued_at") ? root.get("issued_at").nullable((x) => x.isoTime()) : null,
    provenance: root.has("provenance") ? root.get("provenance").nullable(readProvenance) : null,
    certificates: list,
  };
}

// ---------------------------------------------------------------- views

export interface CurvePoint {
  /** Signed placement error, mm (0 is the nominal grasp). */
  error_mm: number;
  /** Objects whose grasp held at this error. */
  held: number;
  /** Objects tested at this error. */
  total: number;
  rate: number;
}

/**
 * The placement-error curve along one axis: at each signed error, the share of
 * objects whose grasp still held. The nominal grasp (0 mm) is shared by both axes.
 */
export function placementCurve(certs: ObjectCertificate[], axis: "x" | "y"): CurvePoint[] {
  const by = new Map<number, { held: number; total: number }>();
  for (const c of certs) {
    for (const t of c.tests) {
      if (t.category !== "placement") continue;
      const err = Number(t.params.error_mm);
      const isNominal = t.name === "nominal" || err === 0;
      if (!isNominal && t.params.axis !== axis) continue;
      const key = isNominal ? 0 : err;
      const cell = by.get(key) ?? { held: 0, total: 0 };
      cell.held += t.success ? 1 : 0;
      cell.total += 1;
      by.set(key, cell);
    }
  }
  return [...by.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([error_mm, c]) => ({ error_mm, held: c.held, total: c.total, rate: c.total ? c.held / c.total : 0 }));
}

export interface DisturbanceRow {
  object: string;
  axis: string;
  success: boolean;
  failure_mode: string | null;
  hold_time_s: number;
  /**
   * One line of the disturbance's settings that do not depend on the object, e.g.
   * "10 mm at 5 Hz, 5 cycles, peak 9.9 m/s^2" or "0.50 N·s (25 N for 20 ms)".
   */
  detail: string;
  /** Push only: how far above the object's centre of mass it was pushed, mm (the push is at the top, so this varies by object). */
  push_height_mm: number | null;
}

function fmtNum(v: unknown, digits = 1): string {
  return typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(digits)) : "?";
}

/** Shake or push results, one row per object and axis. */
export function disturbanceRows(certs: ObjectCertificate[], category: "shake" | "push"): DisturbanceRow[] {
  return certs.flatMap((c) =>
    c.tests
      .filter((t) => t.category === category)
      .map((t) => {
        const p = t.params;
        const detail =
          category === "shake"
            ? `${fmtNum(typeof p.amplitude_m === "number" ? p.amplitude_m * 1000 : null, 0)} mm at ${fmtNum(p.frequency_hz)} Hz, ${fmtNum(p.cycles)} cycles, peak ${fmtNum(p.peak_accel_m_s2)} m/s²`
            : `${fmtNum(p.impulse_ns, 2)} N·s (${fmtNum(p.force_n, 0)} N for ${fmtNum(typeof p.duration_s === "number" ? p.duration_s * 1000 : null, 0)} ms)`;
        return {
          object: c.object,
          axis: String(p.axis ?? "?"),
          success: t.success,
          failure_mode: t.failure_mode,
          hold_time_s: t.hold_time_s,
          detail,
          push_height_mm:
            category === "push" && typeof p.height_above_com_m === "number" ? Number((p.height_above_com_m * 1000).toFixed(1)) : null,
        };
      }),
  );
}

/**
 * The settings every row shares, as one line for "Each: ...", or null when the rows'
 * settings differ (then no single description is true of all of them). A push's
 * height above the centre of mass follows the object's size, so it is given as a
 * range when it varies.
 */
export function describeDisturbance(rows: DisturbanceRow[]): string | null {
  if (rows.length === 0) return null;
  if (new Set(rows.map((r) => r.detail)).size !== 1) return null;
  const detail = rows[0]!.detail;
  const heights = rows.map((r) => r.push_height_mm).filter((h): h is number => h !== null);
  if (heights.length === 0) return detail;
  const lo = Math.min(...heights);
  const hi = Math.max(...heights);
  return lo === hi
    ? `${detail}, ${fmtNum(lo, 0)} mm above the centre of mass`
    : `${detail}, at the object's top: ${fmtNum(lo, 0)} to ${fmtNum(hi, 0)} mm above its centre of mass, depending on the object`;
}

export interface CertificateSummary {
  objects: number;
  episodes: number;
  passed: number;
  /** Mean of the per-object scores (each the unweighted mean of its category pass rates). */
  score: number;
  category: Record<CertCategory, { passed: number; total: number }>;
  /** Smallest per-object tolerance along each axis; null if any object failed its nominal grasp. */
  tolerance_mm: { x: number | null; y: number | null };
}

export function summarizeCertificate(doc: CertificateDoc): CertificateSummary {
  const category: Record<CertCategory, { passed: number; total: number }> = {
    placement: { passed: 0, total: 0 },
    shake: { passed: 0, total: 0 },
    push: { passed: 0, total: 0 },
  };
  let episodes = 0;
  let passed = 0;
  for (const c of doc.certificates) {
    for (const t of c.tests) {
      episodes++;
      passed += t.success ? 1 : 0;
      category[t.category].total++;
      category[t.category].passed += t.success ? 1 : 0;
    }
  }
  const minTol = (axis: "x" | "y") => {
    const vals = doc.certificates.map((c) => c.tolerance_mm[axis]);
    return vals.some((v) => v === null) ? null : Math.min(...(vals as number[]));
  };
  const n = doc.certificates.length;
  return {
    objects: n,
    episodes,
    passed,
    score: n ? doc.certificates.reduce((s, c) => s + c.score, 0) / n : 0,
    category,
    tolerance_mm: { x: minTol("x"), y: minTol("y") },
  };
}
