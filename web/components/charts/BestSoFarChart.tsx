import { formatPercent } from "@/lib/format";
import type { ResultsIndex } from "@/lib/types";

import { ARM_COLORS } from "./ForestPlot";

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** Median best-so-far curve across runs, evaluation by evaluation. */
export function medianCurve(runs: { best: number[] }[]): number[] {
  const length = Math.max(0, ...runs.map((r) => r.best.length));
  const out: number[] = [];
  for (let e = 0; e < length; e++) {
    const vals = runs.flatMap((r) => (e < r.best.length ? [r.best[e]!] : []));
    if (vals.length) out.push(median(vals));
  }
  return out;
}

function points(values: number[], budget: number): string {
  const span = Math.max(budget - 1, 1);
  return values.map((v, i) => `${(i / span) * 100},${(1 - v) * 100}`).join(" ");
}

const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

/**
 * Best-so-far training success against evaluations: thin lines per run, a heavy
 * line for each arm's median. The SVG stretches to the box; strokes and labels
 * do not scale with it.
 */
export function BestSoFarChart({ curves }: { curves: ResultsIndex["curves"] }) {
  const xTicks = [1, ...Array.from({ length: 4 }, (_, i) => Math.round(((i + 1) * curves.budget) / 4))];
  const span = Math.max(curves.budget - 1, 1);
  return (
    <div>
      <div className="grid grid-cols-[2.75rem_minmax(0,1fr)]">
        <div className="relative h-64">
          {Y_TICKS.map((t) => (
            <span key={t} className="num absolute right-2 -translate-y-1/2 text-[0.6875rem] text-ink-3" style={{ top: `${(1 - t) * 100}%` }}>
              {formatPercent(t)}
            </span>
          ))}
        </div>
        <div className="relative h-64 border-b border-l border-rule-strong">
          {Y_TICKS.slice(1).map((t) => (
            <div key={t} className="absolute inset-x-0 border-t border-rule" style={{ top: `${(1 - t) * 100}%` }} />
          ))}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible" aria-hidden="true">
            {curves.arms.map((arm, i) => (
              <g key={arm.id} stroke={ARM_COLORS[i % ARM_COLORS.length]} fill="none">
                {arm.runs.map((run) => (
                  <polyline key={run.run} points={points(run.best, curves.budget)} strokeWidth={1} opacity={0.28} vectorEffect="non-scaling-stroke" />
                ))}
                <polyline points={points(medianCurve(arm.runs), curves.budget)} strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
              </g>
            ))}
          </svg>
        </div>
        <div />
        <div className="relative h-6">
          {xTicks.map((x) => (
            <span key={x} className="num absolute top-1 -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: `${((x - 1) / span) * 100}%` }}>
              {x}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-3 pl-11 text-xs text-ink-3">
        <span>Designs evaluated (budget {curves.budget})</span>
        <ul className="flex flex-wrap gap-4">
          {curves.arms.map((arm, i) => (
            <li key={arm.id} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-5" style={{ background: ARM_COLORS[i % ARM_COLORS.length] }} />
              <span className="text-ink-2">{arm.label}, median of {arm.runs.length} runs</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
