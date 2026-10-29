"use client";

import { usePlayback, usePlaybackStore } from "@/components/replay/playback";
import { changeText, failureLabel, laneAt, type Plan, runEnd, type StudioLane } from "@/components/studio/schema";
import { laneName } from "@/components/studio/story";
import StudioLanes from "@/components/studio/StudioLanes";

import { seekTo } from "./StoryPlayback";

export interface TimelineStep {
  id: string;
  index: number;
  name: string;
}

function pctOf(s: number, span: number): string {
  return `${(Math.min(Math.max(s, 0), span) / span) * 100}%`;
}

function Track({
  lane,
  steps,
  span,
  tone,
  stopLabel,
}: {
  lane: StudioLane;
  steps: TimelineStep[];
  span: number;
  tone: "accent" | "baseline";
  /** Why this lane's own run stops, or null. */
  stopLabel: string | null;
}) {
  const store = usePlaybackStore();
  const t = usePlayback((s) => s.t);
  const m = laneAt(lane, t);
  const end = runEnd(lane);
  const gaps = lane.segments.slice(1).filter((seg) => seg.change !== "none");
  const ring = tone === "accent" ? "ring-accent" : "ring-baseline";
  return (
    <div className="relative h-11 rounded-md bg-paper-2/70" data-track={lane.id}>
      {lane.steps.map((s) => {
        const meta = steps.find((x) => x.id === s.step_id);
        const active = m.kind === "step" && m.step.step_id === s.step_id;
        const left = pctOf(s.start_s, span);
        const width = `calc(${pctOf(s.start_s + s.duration_s, span)} - ${left})`;
        return (
          <button
            key={s.step_id}
            type="button"
            onClick={() => seekTo(store, s.start_s)}
            aria-label={`Step ${(meta?.index ?? 0) + 1}, ${meta?.name ?? s.step_id}: ${s.start_s.toFixed(1)} to ${(s.start_s + s.duration_s).toFixed(1)} s, ${
              s.success ? "passed" : `failed (${s.failure_mode ? failureLabel(s.failure_mode) : "failed"})`
            }. Seek both replays here.`}
            aria-current={active ? "step" : undefined}
            className={`absolute inset-y-1 flex min-w-0 items-center gap-1.5 overflow-hidden rounded border px-1.5 text-left text-xs transition-shadow ${
              s.success ? "border-pass/50 bg-pass-soft text-ink" : "border-fail/70 bg-fail-soft text-ink"
            } ${active ? `ring-2 ${ring}` : "hover:ring-1 hover:ring-rule-strong"}`}
            style={{ left, width }}
            data-timeline-step={s.step_id}
          >
            <span className="num shrink-0 font-bold">{(meta?.index ?? 0) + 1}</span>
            <span className="hidden truncate sm:inline">{meta?.name ?? s.step_id}</span>
            {!s.success ? (
              <svg viewBox="0 0 10 10" className="ml-auto size-2.5 shrink-0 text-fail" aria-hidden="true">
                <path d="M1.5 1.5l7 7M8.5 1.5l-7 7" stroke="currentColor" strokeWidth="1.8" />
              </svg>
            ) : null}
          </button>
        );
      })}
      {gaps.map((g) => (
        <span
          key={`gap-${g.index}`}
          className="absolute inset-y-2 rounded-sm bg-[repeating-linear-gradient(135deg,var(--ink-3)_0_1.5px,transparent_1.5px_5px)] opacity-50"
          style={{ left: pctOf(g.change_start_s, span), width: `calc(${pctOf(g.start_s, span)} - ${pctOf(g.change_start_s, span)})` }}
          title={changeText(g.change, g.change_s)}
        />
      ))}
      {!lane.completed ? (
        <>
          {end < span - 0.05 ? (
            <span
              className="absolute inset-y-1 rounded-r bg-[repeating-linear-gradient(135deg,var(--rule-strong)_0_1px,transparent_1px_6px)]"
              style={{ left: pctOf(end, span), right: 0 }}
              aria-hidden="true"
            />
          ) : null}
          {/* The stop line; a run that stops at the end of the clock keeps it inside the track. */}
          <span className="absolute -inset-y-1 w-0.5 bg-fail" style={{ left: `calc(${pctOf(end, span)} - ${end >= span - 0.05 ? "2px" : "0px"})` }} aria-hidden="true" />
          {stopLabel ? (
            <span
              className="pointer-events-none absolute top-1/2 hidden max-w-[45%] -translate-y-1/2 truncate rounded bg-surface px-1.5 py-0.5 text-xs text-fail sm:block"
              // Right of the stop line when there is room, left of it near the end of the clock.
              style={end / span > 0.55 ? { right: `calc(${100 - (Math.min(end, span) / span) * 100}% + 0.5rem)` } : { left: `calc(${pctOf(end, span)} + 0.5rem)` }}
              data-stop-marker
            >
              Stops here: {stopLabel}
            </span>
          ) : null}
        </>
      ) : null}
      <span className="pointer-events-none absolute -inset-y-1 w-0.5 rounded-full bg-ink" style={{ left: pctOf(t, span) }} aria-hidden="true" />
    </div>
  );
}

function Axis({ span }: { span: number }) {
  const step = span > 50 ? 10 : 5;
  const ticks: number[] = [];
  for (let s = 0; s <= span + 1e-6; s += step) ticks.push(s);
  return (
    <div className="relative mt-1 h-4" aria-hidden="true">
      {ticks.map((s) => (
        <span key={s} className="num absolute -translate-x-1/2 text-[0.6875rem] text-ink-3" style={{ left: pctOf(s, span) }}>
          {s}
          {s === 0 ? " s" : ""}
        </span>
      ))}
    </div>
  );
}

/** The step timeline under both replays: each step is a segment on the shared clock; click to seek. */
export function StepTimeline({ lanes, steps, stopLabels }: { lanes: StudioLane[]; steps: TimelineStep[]; stopLabels: Record<string, string | null> }) {
  const span = Math.max(1, ...lanes.map((l) => Math.max(l.duration_s, runEnd(l))));
  return (
    <div className="panel p-3 sm:p-4" data-step-timeline>
      <div className="grid grid-cols-1 gap-x-4 gap-y-1.5 sm:grid-cols-[8.5rem_minmax(0,1fr)] sm:items-center">
        {lanes.map((lane) => (
          <div key={lane.id} className="contents">
            <p className={`text-sm font-semibold ${lane.id === "best_single" ? "text-baseline" : "text-accent"}`}>
              {laneName(lane)}
              <span className="ml-2 font-normal text-ink-3 sm:ml-0 sm:block sm:text-xs">
                {lane.completed ? `done at ${(lane.cycle_time_s ?? runEnd(lane)).toFixed(1)} s` : `stops at ${runEnd(lane).toFixed(1)} s`}
              </span>
            </p>
            <Track lane={lane} steps={steps} span={span} tone={lane.id === "best_single" ? "baseline" : "accent"} stopLabel={lane.completed ? null : (stopLabels[lane.id] ?? null)} />
          </div>
        ))}
        <span className="hidden sm:block" />
        <Axis span={span} />
      </div>
      <p className="mt-2 text-xs text-ink-3">Click a step to jump both replays to it. Hatched gaps are a head switch or tool change: a fixed time, not simulated.</p>
    </div>
  );
}

/** The side-by-side replays on the page's shared clock, with the step timeline under them. */
export default function StoryReplays({
  lanes,
  plan,
  steps,
  stopLabels,
}: {
  lanes: StudioLane[];
  plan: Pick<Plan, "steps">;
  steps: TimelineStep[];
  /** Why each lane's run stops, by lane id (each lane's own first failed step). */
  stopLabels: Record<string, string | null>;
}) {
  const store = usePlaybackStore();
  return (
    <div className="grid gap-4">
      <StudioLanes lanes={lanes} plan={plan} store={store} strips={false} framing="objects" zoom={1.35} />
      <StepTimeline lanes={lanes} steps={steps} stopLabels={stopLabels} />
    </div>
  );
}
