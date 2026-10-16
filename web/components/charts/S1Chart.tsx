import { formatPercent } from "@/lib/format";
import { type ArmS1Summary, medianCurve, stepPoints } from "@/lib/results";
import type { ResultsIndex } from "@/lib/types";

import { ARM_COLORS } from "./ForestPlot";

const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

function xTicks(budget: number): number[] {
  return Array.from(new Set([1, ...Array.from({ length: 4 }, (_, i) => Math.round(((i + 1) * budget) / 4))])).sort((a, b) => a - b);
}

/**
 * S1 curves: best training success so far after each simulated design. Thin step
 * lines are runs, the heavy line each arm's median; the dashed line is the threshold
 * and the hatched band the designs every arm shares before the arms diverge.
 */
export function S1Chart({ s1 }: { s1: ResultsIndex["s1"] }) {
  const span = Math.max(s1.budget - 1, 1);
  const x = (design: number) => ((design - 1) / span) * 100;
  const shared = s1.shared_prefix;
  const described = s1.arms
    .map((arm) => {
      const med = medianCurve(arm.runs);
      return `${arm.label}: ${med.length ? formatPercent(med.at(-1)!) : "no data"} after ${med.length} designs`;
    })
    .join("; ");
  return (
    <figure>
      <p className="sr-only">
        Median best training success so far. {described}.
        {s1.threshold !== null ? ` Threshold ${formatPercent(s1.threshold)}.` : ""}
      </p>
      <div className="grid grid-cols-[2.75rem_minmax(0,1fr)]">
        <div className="relative h-64">
          {Y_TICKS.map((t) => (
            <span key={t} className="num absolute right-2 -translate-y-1/2 text-[0.6875rem] text-ink-3" style={{ top: `${(1 - t) * 100}%` }}>
              {formatPercent(t)}
            </span>
          ))}
        </div>
        <div className="relative h-64 border-b border-l border-rule-strong">
          {shared > 1 ? (
            <div
              className="absolute inset-y-0 left-0 border-r border-dashed border-rule-strong"
              style={{
                width: `${x(shared)}%`,
                backgroundImage: "repeating-linear-gradient(135deg, var(--grid) 0 4px, transparent 4px 8px)",
              }}
            >
              <span className="absolute bottom-1 left-1 bg-paper/80 px-0.5 text-[0.6875rem] leading-tight text-ink-3">Shared seeds</span>
            </div>
          ) : null}
          {Y_TICKS.slice(1).map((t) => (
            <div key={t} className="absolute inset-x-0 border-t border-rule" style={{ top: `${(1 - t) * 100}%` }} />
          ))}
          {s1.threshold !== null ? (
            <div className="absolute inset-x-0 border-t-2 border-dashed border-pending" style={{ top: `${(1 - s1.threshold) * 100}%` }}>
              <span className="num absolute right-1 -top-5 bg-paper/90 px-1 text-[0.6875rem] text-pending">
                threshold {formatPercent(s1.threshold)}
              </span>
            </div>
          ) : null}
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible" aria-hidden="true">
            {s1.arms.map((arm, i) => (
              <g key={arm.id} stroke={ARM_COLORS[i % ARM_COLORS.length]} fill="none">
                {arm.runs.length > 1
                  ? arm.runs.map((run) => (
                      <polyline key={run.run} points={stepPoints(run.best, s1.budget)} strokeWidth={1} opacity={0.3} vectorEffect="non-scaling-stroke" />
                    ))
                  : null}
                <polyline
                  points={stepPoints(medianCurve(arm.runs), s1.budget)}
                  strokeWidth={2 + 1.75 * (s1.arms.length - 1 - i)}
                  strokeDasharray={i % 2 === 1 ? "6 3" : undefined}
                  vectorEffect="non-scaling-stroke"
                />
              </g>
            ))}
          </svg>
        </div>
        <div />
        <div className="relative h-6">
          {xTicks(s1.budget).map((t) => (
            <span key={t} className="num absolute top-1 -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: `${x(t)}%` }}>
              {t}
            </span>
          ))}
        </div>
      </div>
      <figcaption className="mt-1 flex flex-wrap items-center justify-between gap-3 pl-11 text-xs text-ink-3">
        <span>Simulated designs (budget B = {s1.budget})</span>
        <ul className="flex flex-wrap gap-4">
          {s1.arms.map((arm, i) => (
            <li key={arm.id} className="flex items-center gap-1.5">
              <svg width="20" height="4" aria-hidden="true">
                <line x1="0" y1="2" x2="20" y2="2" stroke={ARM_COLORS[i % ARM_COLORS.length]} strokeWidth="2.5" strokeDasharray={i % 2 === 1 ? "6 3" : undefined} />
              </svg>
              <span className="text-ink-2">
                {arm.label}, {arm.runs.length > 1 ? `median of ${arm.runs.length} runs` : `${arm.runs.length} run`}
              </span>
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}

/**
 * Designs needed to reach the threshold, one dot per run on a shared axis. A hollow
 * dot with an arrow past the budget is a run that never reached it (censored at B).
 */
export function S1DotPlot({ summaries, budget }: { summaries: ArmS1Summary[]; budget: number }) {
  // The axis runs past the budget so censored runs sit visibly beyond it.
  const max = budget * 1.25;
  const censoredAt = budget * 1.12;
  const left = (d: number) => `${(Math.min(Math.max(d, 0), max) / max) * 100}%`;
  const ticks = xTicks(budget);
  return (
    <figure>
      <div className="grid grid-cols-[7.5rem_minmax(0,1fr)] gap-x-4 sm:grid-cols-[12rem_minmax(0,1fr)]">
        {summaries.map((s, i) => (
          <div key={s.id} className="contents">
            <div className="flex flex-col justify-center border-b border-rule py-3 text-sm">
              <span className="text-ink">{s.label}</span>
              <span className="num text-xs text-ink-3">
                {s.reached} of {s.runs} reached · median {s.median_designs === null ? `> ${budget}` : s.median_designs}
              </span>
            </div>
            <div className="relative h-14 border-b border-rule">
              <div className="absolute inset-y-0 border-l border-dashed border-ink-3" style={{ left: left(budget) }} aria-hidden="true" />
              {s.outcomes.map((o, j) => {
                const color = ARM_COLORS[i % ARM_COLORS.length];
                const top = `${s.outcomes.length > 1 ? 22 + ((j * 37) % 56) : 50}%`;
                return o.reached ? (
                  <span
                    key={o.run}
                    className="absolute size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
                    style={{ left: left(o.designs), top, background: color }}
                    title={`${o.run}: ${o.designs} designs`}
                  />
                ) : (
                  <span
                    key={o.run}
                    className="absolute flex -translate-y-1/2 items-center"
                    style={{ left: left(censoredAt), top, color }}
                    title={`${o.run}: not reached within ${o.designs} designs`}
                  >
                    <span className="size-2.5 -translate-x-1/2 rounded-full border-2 border-current bg-paper" />
                    <svg viewBox="0 0 8 8" className="size-2" aria-hidden="true">
                      <path d="M1 1l4 3-4 3" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                  </span>
                );
              })}
              {s.median_designs !== null ? (
                <span
                  className="absolute top-1.5 bottom-1.5 w-0.5 -translate-x-1/2"
                  style={{ left: left(s.median_designs), background: ARM_COLORS[i % ARM_COLORS.length] }}
                  aria-hidden="true"
                />
              ) : null}
            </div>
          </div>
        ))}
        <div />
        <div className="relative h-6">
          {ticks.map((t) => (
            <span key={t} className="num absolute top-1 -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: left(t) }}>
              {t}
            </span>
          ))}
          <span className="absolute right-0 top-1 text-[0.6875rem] text-ink-3">not reached</span>
        </div>
      </div>
      <figcaption className="mt-1 text-xs text-ink-3">
        Each dot is one run; the bar marks the arm&apos;s median. Fewer designs is better. Hollow dots past the dashed line
        never reached the threshold within B = {budget} (censored).
      </figcaption>
    </figure>
  );
}
