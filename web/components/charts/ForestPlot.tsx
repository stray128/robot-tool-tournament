import { formatPercent } from "@/lib/format";
import type { ArmEstimate } from "@/lib/types";

/** Series colours by arm order; the second arm (the Nemotron arm in the protocol) takes the accent. */
export const ARM_COLORS = ["var(--ink-2)", "var(--accent)", "var(--pending)", "var(--pass)"] as const;

const TICKS = [0, 0.25, 0.5, 0.75, 1];

/** Closest two reference labels may sit on one line, as a fraction of the axis. */
const LABEL_GAP = 0.16;

/** Line index (0 = nearest the plot) for each reference label, so close labels do not overlap. */
export function labelRows(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const lastOnRow: number[] = [];
  const rows = new Array<number>(values.length).fill(0);
  for (const [v, i] of order) {
    let row = lastOnRow.findIndex((last) => v - last >= LABEL_GAP);
    if (row < 0) row = lastOnRow.length;
    lastOnRow[row] = v;
    rows[i] = row;
  }
  return rows;
}

/**
 * Run-level estimates per arm: each run as a dot, the mean as a diamond, the 95%
 * interval as a line; manual entrants as dashed reference lines. HTML, not SVG, so
 * the text stays legible at any width.
 */
export function ForestPlot({ forest }: { forest: { arms: ArmEstimate[]; references: { label: string; value: number }[] } }) {
  const rows = forest.arms.length;
  const refRows = labelRows(forest.references.map((r) => r.value));
  const labelLines = Math.max(0, ...refRows.map((r) => r + 1));
  const pct = (v: number) => `${Math.min(Math.max(v, 0), 1) * 100}%`;
  return (
    <div
      className="grid grid-cols-[7.5rem_minmax(0,1fr)] gap-x-4 sm:grid-cols-[12rem_minmax(0,1fr)]"
      style={{ paddingTop: `${labelLines * 1.1}rem` }}
    >
      {forest.arms.map((arm, i) => (
        <div key={arm.id} className="contents">
          <div className="flex flex-col justify-center border-b border-rule py-3 text-sm" style={{ gridRow: i + 1 }}>
            <span className="text-ink">{arm.label}</span>
            <span className="num text-xs text-ink-3">
              {formatPercent(arm.mean)} [{formatPercent(arm.ci95[0])}, {formatPercent(arm.ci95[1])}]
            </span>
          </div>
          <div className="relative h-14 border-b border-rule" style={{ gridRow: i + 1, gridColumn: 2 }}>
            <div
              className="absolute top-1/2 h-0.5 -translate-y-1/2"
              style={{ left: pct(arm.ci95[0]), width: `calc(${pct(arm.ci95[1])} - ${pct(arm.ci95[0])})`, background: ARM_COLORS[i % ARM_COLORS.length] }}
            />
            {arm.runs.map((v, j) => (
              <span
                key={j}
                className="absolute size-1.5 -translate-x-1/2 rounded-full opacity-60"
                style={{
                  left: pct(v),
                  top: `${18 + ((j * 37) % 64)}%`,
                  background: ARM_COLORS[i % ARM_COLORS.length],
                }}
              />
            ))}
            <span
              className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rotate-45 border border-paper"
              style={{ left: pct(arm.mean), background: ARM_COLORS[i % ARM_COLORS.length] }}
            />
          </div>
        </div>
      ))}
      <div className="pointer-events-none relative" style={{ gridColumn: 2, gridRow: `1 / span ${rows}` }}>
        {forest.references.map((ref, i) => (
          <div
            key={ref.label}
            className="absolute bottom-0 border-l border-dashed border-ink-3"
            style={{ left: pct(ref.value), top: `${-1.1 * (refRows[i] ?? 0) - 1.1}rem` }}
          >
            <span className={`absolute top-0 z-10 whitespace-nowrap bg-paper px-0.5 text-[0.6875rem] leading-4 text-ink-3 ${ref.value > 0.8 ? "right-1" : "left-1"}`}>
              {ref.label}
            </span>
          </div>
        ))}
      </div>
      <div className="relative h-6" style={{ gridColumn: 2, gridRow: rows + 1 }}>
        {TICKS.map((t) => (
          <span key={t} className="num absolute top-1 -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: pct(t) }}>
            {formatPercent(t)}
          </span>
        ))}
      </div>
    </div>
  );
}
