import { formatDuration, formatUsd } from "@/lib/format";
import { isJobActive, type JobState, type JobStatus } from "@/lib/session";

const STATUS_TEXT: Record<JobStatus, string> = {
  queued: "Queued",
  provisioning: "Starting container",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
  unknown: "Status unknown",
};

/** Status with a shape as well as a colour: a pulsing dot while active, a tick or a cross when done. */
export function JobStatusPill({ status }: { status: JobStatus }) {
  const active = isJobActive(status);
  const tone =
    status === "completed" ? "text-pass border-pass" : status === "failed" || status === "cancelled" ? "text-fail border-fail" : "text-pending border-pending";
  return (
    <span className={`inline-flex items-center gap-1.5 border px-1.5 py-0.5 text-xs leading-4 ${tone}`}>
      {active ? (
        <span className="relative flex size-2" aria-hidden="true">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60 motion-reduce:hidden" />
          <span className="relative inline-flex size-2 rounded-full bg-current" />
        </span>
      ) : status === "completed" ? (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M1.5 6.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
      ) : (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M2 2l8 8M10 2l-8 8" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
      )}
      {STATUS_TEXT[status]}
    </span>
  );
}

/**
 * A simulation job as a provenance chip: where it runs, its id, platform, preset,
 * region, minutes and dollars, and its live status. Only the fields the agent reported
 * are shown; an illustrative session says so in the subject cell.
 */
export function JobChip({ job, illustrative = false }: { job: JobState; illustrative?: boolean }) {
  // Only a job the agent reported as Nebius is labelled one; an unknown backend is not.
  const nebius = job.backend === "nebius";
  const where = nebius ? "Nebius job" : job.backend === "local" ? "Local job" : "Simulation job";
  const facts = [
    job.platform,
    job.preset,
    job.region,
    job.designs !== null ? `${job.designs} designs` : null,
    job.episodes !== null ? `${job.episodes} episodes` : null,
    job.elapsed_s !== null ? formatDuration(job.elapsed_s) : null,
    job.cost_usd !== null ? `${formatUsd(job.cost_usd)}${job.cost_estimated && job.cost_usd > 0 ? " est." : ""}` : null,
  ].filter((f): f is string => Boolean(f));
  return (
    <div
      className={`inline-grid max-w-full grid-cols-[auto_minmax(0,1fr)] border text-sm ${
        nebius && !illustrative ? "border-accent" : "border-dashed border-rule-strong"
      }`}
    >
      <span className={`px-2 py-1 text-xs leading-5 ${nebius && !illustrative ? "bg-accent text-accent-ink" : "bg-paper-2 text-ink-2"}`}>
        {where}
        {illustrative ? <span className="block">illustrative</span> : null}
      </span>
      <span className="min-w-0 px-2 py-1">
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="num break-all text-ink">{job.id}</span>
          <JobStatusPill status={job.status} />
        </span>
        {facts.length ? (
          <span className="num mt-1 flex flex-wrap gap-x-3 text-xs text-ink-2">
            {facts.map((f) => (
              <span key={f}>{f}</span>
            ))}
          </span>
        ) : null}
        {job.error ? <span className="mt-1 block text-xs text-fail">{job.error}</span> : null}
      </span>
    </div>
  );
}
