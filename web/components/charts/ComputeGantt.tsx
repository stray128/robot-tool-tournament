import { formatDuration, formatUtc } from "@/lib/format";
import type { Job } from "@/lib/types";

const HATCH = "repeating-linear-gradient(135deg, var(--ink-3) 0 2px, transparent 2px 5px)";

/** Tick labels: elapsed time for short spans, UTC clock times for long ones. */
function ticks(start: number, end: number): { at: number; label: string }[] {
  const span = Math.max(end - start, 1);
  return [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const ms = start + f * span;
    const label =
      span < 10 * 60_000
        ? `+${formatDuration((f * span) / 1000)}`
        : new Date(ms).toISOString().slice(span > 36 * 3_600_000 ? 5 : 11, 16).replace("T", " ");
    return { at: f, label };
  });
}

/**
 * One bar per job on a shared time axis. Nebius jobs are solid accent; local runs
 * are hatched so the two read apart without colour.
 */
export function ComputeGantt({ jobs }: { jobs: Job[] }) {
  if (jobs.length === 0) return null;
  const sorted = [...jobs].sort((a, b) => Date.parse(a.started_at) - Date.parse(b.started_at));
  const start = Math.min(...sorted.map((j) => Date.parse(j.started_at)));
  const end = Math.max(...sorted.map((j) => Date.parse(j.ended_at)));
  const span = Math.max(end - start, 1);

  return (
    <div>
      <ol className="grid gap-y-1.5">
        {sorted.map((job) => {
          const left = ((Date.parse(job.started_at) - start) / span) * 100;
          const width = ((Date.parse(job.ended_at) - Date.parse(job.started_at)) / span) * 100;
          return (
            <li key={job.id} className="grid grid-cols-[8rem_minmax(0,1fr)] items-center gap-3 text-sm sm:grid-cols-[14rem_minmax(0,1fr)]">
              <span className="truncate text-ink" title={job.name}>
                {job.name}
              </span>
              <span className="relative h-5 bg-paper-2/70">
                <span
                  className="absolute inset-y-0 min-w-[3px]"
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    background: job.kind === "nebius" ? "var(--accent)" : HATCH,
                  }}
                  title={`${job.name}: ${formatUtc(job.started_at)}, ${formatDuration(job.wall_s)}`}
                />
              </span>
            </li>
          );
        })}
      </ol>
      <div className="mt-1 grid grid-cols-[8rem_minmax(0,1fr)] gap-3 sm:grid-cols-[14rem_minmax(0,1fr)]">
        <span className="text-xs text-ink-3">{formatUtc(new Date(start).toISOString())}</span>
        <span className="relative h-5">
          {ticks(start, end).map((t) => (
            <span
              key={t.at}
              className="num absolute top-0 whitespace-nowrap text-[0.6875rem] text-ink-3"
              style={{ left: `${t.at * 100}%`, transform: t.at === 1 ? "translateX(-100%)" : t.at === 0 ? undefined : "translateX(-50%)" }}
            >
              {t.label}
            </span>
          ))}
        </span>
      </div>
      <ul className="mt-3 flex flex-wrap gap-4 text-xs text-ink-2">
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 bg-accent" /> Nebius Serverless Job
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-5" style={{ background: HATCH }} /> Local run (not scored)
        </li>
      </ul>
    </div>
  );
}
