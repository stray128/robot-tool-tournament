/**
 * S1, the primary endpoint: how many simulated designs a search needs before its
 * best-so-far training success reaches the pre-registered threshold. A run that never
 * reaches it within the budget B is right-censored at B (it needed "more than B").
 * When the file records each run's T (counted from `s1.count_from`, so the shared
 * round 0 does not count), T is used as it is; older files are read from the curve.
 * Pure functions, shared by the /results charts, the landing-page stats and the tests.
 */

import { formatPercent } from "./format";
import type { S1Arm, S1Run, SecondaryTest } from "./types";

/** Running maximum of per-design scores: the best-so-far curve. */
export function bestSoFar(scores: number[]): number[] {
  const out: number[] = [];
  let best = -Infinity;
  for (const s of scores) {
    best = Math.max(best, s);
    out.push(best);
  }
  return out;
}

/** 1-based index of the first design whose best-so-far reaches `threshold`, or null. */
export function designsToThreshold(best: number[], threshold: number): number | null {
  const i = best.findIndex((v) => v >= threshold - 1e-9);
  return i < 0 ? null : i + 1;
}

export interface S1Outcome {
  run: string;
  job_id: string | null;
  /** Designs needed, or the censoring point when not reached. */
  designs: number;
  reached: boolean;
  /** S1's T as the analysis counts it: `designs` when reached, else one past the censoring point. */
  t: number;
  /** Best training success at the end of the run. */
  final_best: number | null;
}

/**
 * One run's S1 value. Not reached: censored at the budget, or at the run's own length
 * if it stopped early (it is then known only to need more than that many designs).
 */
export function s1Outcome(run: S1Run, threshold: number, budget: number): S1Outcome {
  const limit = Math.min(budget, run.best.length);
  const n = run.t !== undefined ? (run.t <= limit ? run.t : null) : designsToThreshold(run.best, threshold);
  return {
    run: run.run,
    job_id: run.job_id,
    designs: n ?? limit,
    reached: n !== null,
    t: n ?? limit + 1,
    final_best: run.best.at(-1) ?? null,
  };
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

export interface ArmS1Summary {
  id: string;
  label: string;
  outcomes: S1Outcome[];
  reached: number;
  runs: number;
  /**
   * Median designs to threshold, treating a censored run as "more than its value".
   * null when the median itself is censored (half or more of the runs never reached it).
   */
  median_designs: number | null;
  /** Mean of T, a run that never reached the threshold counted as B + 1; null with no runs. */
  mean_designs: number | null;
  /** Median best-so-far at the end of the budget. */
  median_final_best: number | null;
}

/**
 * Per-arm S1 summary. Censored runs sort after every reached run, so the median is
 * exact while fewer than half the runs are censored and reported as not reached
 * (null) otherwise; no value is imputed.
 */
export function summarizeArm(arm: S1Arm, threshold: number, budget: number): ArmS1Summary {
  const outcomes = arm.runs.map((r) => s1Outcome(r, threshold, budget));
  const keyed = outcomes.map((o) => (o.reached ? o.designs : Infinity)).sort((a, b) => a - b);
  let med: number | null = null;
  if (keyed.length > 0) {
    const m = Math.floor(keyed.length / 2);
    const value = keyed.length % 2 ? keyed[m]! : (keyed[m - 1]! + keyed[m]!) / 2;
    med = Number.isFinite(value) ? value : null;
  }
  const finals = outcomes.map((o) => o.final_best).filter((v): v is number => v !== null);
  return {
    id: arm.id,
    label: arm.label,
    outcomes,
    reached: outcomes.filter((o) => o.reached).length,
    runs: outcomes.length,
    median_designs: med,
    mean_designs: outcomes.length ? outcomes.reduce((s, o) => s + o.t, 0) / outcomes.length : null,
    median_final_best: finals.length ? median(finals) : null,
  };
}

/** Median best-so-far curve across runs, design by design (runs of unequal length allowed). */
export function medianCurve(runs: { best: number[] }[]): number[] {
  const length = Math.max(0, ...runs.map((r) => r.best.length));
  const out: number[] = [];
  for (let e = 0; e < length; e++) {
    const vals = runs.flatMap((r) => (e < r.best.length ? [r.best[e]!] : []));
    if (vals.length) out.push(median(vals));
  }
  return out;
}

/** SVG polyline points for a step curve on a 0..100 box: design 1 at x=0, design `budget` at x=100. */
export function stepPoints(values: number[], budget: number): string {
  const span = Math.max(budget - 1, 1);
  const pts: string[] = [];
  values.forEach((v, i) => {
    const x = (i / span) * 100;
    const y = (1 - v) * 100;
    if (i > 0) pts.push(`${x},${(1 - values[i - 1]!) * 100}`);
    pts.push(`${x},${y}`);
  });
  return pts.join(" ");
}

/** A secondary endpoint's arm value as docs/RESULTS.md gives it: "60.5%" for a rate, "0.703" for a score. */
export function secondaryValue(t: SecondaryTest, v: number): string {
  return t.unit === "rate" ? formatPercent(v, 1) : v.toFixed(3);
}

/** C minus A, signed: "+12.0 points" for a rate, "+0.085" for a score. */
export function secondaryDiff(t: SecondaryTest): string {
  const sign = t.diff >= 0 ? "+" : "-";
  return t.unit === "rate" ? `${sign}${Math.abs(t.diff * 100).toFixed(1)} points` : `${sign}${Math.abs(t.diff).toFixed(3)}`;
}
