import type { CurvePoint } from "@/lib/certificate";
import { formatPercent } from "@/lib/format";

const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

export const AXIS_SERIES = {
  x: { label: "Along x, the finger width", color: "var(--ink)", marker: "rounded-full" },
  y: { label: "Along y, the closing axis", color: "var(--accent)", marker: "" },
} as const;

/**
 * Placement-error curve: the share of objects still held at each signed placement
 * error, one line per axis. The band marks the errors every object tolerated on both
 * axes. HTML axes and markers around a stretched SVG, so text and dots stay crisp.
 */
export function PlacementCurve({
  curves,
  toleranceMm,
}: {
  curves: { x: CurvePoint[]; y: CurvePoint[] };
  toleranceMm: number | null;
}) {
  const all = [...curves.x, ...curves.y].map((p) => p.error_mm);
  const maxAbs = Math.max(1, ...all.map(Math.abs));
  const xTicks = Array.from(new Set([...all, 0])).sort((a, b) => a - b);
  const left = (e: number) => ((e + maxAbs) / (2 * maxAbs)) * 100;
  const top = (r: number) => (1 - r) * 100;
  const pts = (series: CurvePoint[]) => series.map((p) => `${left(p.error_mm)},${top(p.rate)}`).join(" ");

  return (
    <figure>
      <div className="grid grid-cols-[2.75rem_minmax(0,1fr)]">
        <div className="relative h-56">
          {Y_TICKS.map((t) => (
            <span key={t} className="num absolute right-2 -translate-y-1/2 text-[0.6875rem] text-ink-3" style={{ top: `${top(t)}%` }}>
              {formatPercent(t)}
            </span>
          ))}
        </div>
        <div className="relative mx-2 h-56 border-b border-l border-rule-strong">
          {toleranceMm !== null && toleranceMm > 0 ? (
            <div
              className="absolute inset-y-0 bg-pass/10"
              style={{ left: `${left(-toleranceMm)}%`, width: `${left(toleranceMm) - left(-toleranceMm)}%` }}
            >
              <span className="absolute left-1 top-1 text-[0.6875rem] leading-tight text-pass">Every object held</span>
            </div>
          ) : null}
          {Y_TICKS.slice(1).map((t) => (
            <div key={t} className="absolute inset-x-0 border-t border-rule" style={{ top: `${top(t)}%` }} />
          ))}
          <div className="absolute inset-y-0 border-l border-dashed border-rule-strong" style={{ left: `${left(0)}%` }} />
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible" aria-hidden="true">
            {(["x", "y"] as const).map((axis) => (
              <polyline
                key={axis}
                points={pts(curves[axis])}
                fill="none"
                stroke={AXIS_SERIES[axis].color}
                strokeWidth={axis === "x" ? 2 : 1.5}
                strokeDasharray={axis === "y" ? "5 3" : undefined}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>
          {(["x", "y"] as const).flatMap((axis) =>
            curves[axis].map((p) => (
              <span
                key={`${axis}${p.error_mm}`}
                className={`absolute size-2 -translate-x-1/2 -translate-y-1/2 border-2 bg-paper ${AXIS_SERIES[axis].marker}`}
                style={{ left: `${left(p.error_mm)}%`, top: `${top(p.rate)}%`, borderColor: AXIS_SERIES[axis].color }}
                title={`${axis} ${p.error_mm > 0 ? "+" : ""}${p.error_mm} mm: ${p.held} of ${p.total} held`}
                aria-hidden="true"
              />
            )),
          )}
        </div>
        <div />
        <div className="relative mx-2 h-6">
          {xTicks.map((e) => (
            <span key={e} className="num absolute top-1 -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: `${left(e)}%` }}>
              {e > 0 ? `+${e}` : e}
            </span>
          ))}
        </div>
      </div>
      <figcaption className="mt-1 flex flex-wrap items-center justify-between gap-3 pl-11 text-xs text-ink-3">
        <span>Placement error, mm (0 = where the controller aims)</span>
        <ul className="flex flex-wrap gap-4">
          {(["x", "y"] as const).map((axis) => (
            <li key={axis} className="flex items-center gap-1.5">
              <span
                className={`inline-block size-2 border-2 ${AXIS_SERIES[axis].marker}`}
                style={{ borderColor: AXIS_SERIES[axis].color }}
                aria-hidden="true"
              />
              <span className="text-ink-2">{AXIS_SERIES[axis].label}</span>
            </li>
          ))}
        </ul>
      </figcaption>
      {/* The text equivalent, for screen readers. A table ignores the 1 px width of sr-only and
          would widen the page on phones, so a wrapper carries it. */}
      <div className="sr-only">
        <table>
          <caption>Share of objects held at each placement error</caption>
          <thead>
            <tr>
              <th scope="col">Error (mm)</th>
              <th scope="col">Along x</th>
              <th scope="col">Along y</th>
            </tr>
          </thead>
          <tbody>
            {xTicks.map((e) => {
              const x = curves.x.find((p) => p.error_mm === e);
              const y = curves.y.find((p) => p.error_mm === e);
              return (
                <tr key={e}>
                  <th scope="row">{e}</th>
                  <td>{x ? `${x.held} of ${x.total}` : "not tested"}</td>
                  <td>{y ? `${y.held} of ${y.total}` : "not tested"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
