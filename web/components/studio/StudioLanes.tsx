"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { prefersReducedMotion, useClock } from "@/components/replay/clock";
import { assetPoints, loadLane } from "@/components/replay/lanes";
import { createPlaybackStore, PlaybackContext, usePlayback, usePlaybackStore } from "@/components/replay/playback";
import { useThemeColors } from "@/components/replay/useThemeColors";
import { clayAssets, type LoadedAsset } from "@/lib/assets";
import { dataUrl } from "@/lib/paths";
import { isV2, Replay, replayCaption, unionBounds, type Bounds } from "@/lib/replay";

import { changeText, humanize, laneAt, outcomeText, type Plan, runEnd, stepTitle, type StudioLane } from "./schema";
import { StudioTransport } from "./StudioTransport";
import { TierBadge } from "./ui";

const StudioReplayCanvas = dynamic(() => import("@/components/replay/StudioReplayCanvas"), { ssr: false });

/** Screen space kept free above and below a lane for the HTML readouts, CSS px. */
const PAD = { top: 76, bottom: 64 };
/** Margin around the bodies' swept volume and the table band kept in view, metres. */
const VIEW_MARGIN_M = 0.02;
const TABLE_BAND_M = 0.015;
const NO_CLAY: ReadonlySet<string> = new Set();

type LaneState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      replay: Replay;
      assets: Map<string, LoadedAsset> | null;
      /** Asset ids drawn untextured (brand screen). */
      clay: ReadonlySet<string>;
      /** The display-only tool stand's note (header meta.tool_stand), when the scene has one. */
      toolStand: string | null;
    };

/**
 * Tool bodies drawn from their geoms get the tool colour. The shared renderer colours a geom by
 * its body's name (`finger*` is the tool); effector bodies are named `eff_*`, so they are aliased
 * for drawing only (the data is not changed). Bodies with GLB visuals keep their own materials.
 */
function paintTools(replay: Replay): Replay {
  const header = replay.header;
  const bodies = header.bodies.map((b) => (/(^|_)eff_/.test(b.name) && !b.name.startsWith("finger") ? { ...b, name: `finger_${b.name}` } : b));
  return new Replay({ ...header, bodies } as typeof header, replay.frames, replay.overlays, replay.bytes);
}

type StepsOnly = Pick<Plan, "steps">;

function Readout({ lane, plan }: { lane: StudioLane; plan: StepsOnly }) {
  const t = usePlayback((s) => s.t);
  const m = laneAt(lane, t);
  const n = plan.steps.length;
  // The run ends at its last step; the scrubber moves in 1/30 s steps and the recording may stop
  // a frame before the task clock, so the stamp shows from the last frame on.
  const ended = lane.steps.length > 0 && t >= Math.min(runEnd(lane), lane.duration_s) - 0.04;
  return (
    <>
      <div className="pointer-events-none absolute right-2 top-2 flex max-w-[48%] flex-col items-end gap-1 text-right" data-moment={m.kind}>
        {m.kind === "step" ? (
          <div className="bg-paper/85 px-1.5 py-0.5 leading-tight">
            <div className="text-[0.6875rem] text-ink-3">
              Step {plan.steps.findIndex((s) => s.id === m.step.step_id) + 1} of {n}
            </div>
            <div className="text-xs text-ink sm:text-sm">{humanize(m.step.step_id)}</div>
            <div className="text-[0.6875rem] text-ink-2">
              {humanize(m.step.primitive)} · tool {m.step.tool + 1}
              {m.segment ? `, ${m.segment.tool_label}` : ""}
            </div>
          </div>
        ) : null}
        {m.kind === "change" ? (
          <div className="border border-pending bg-paper/90 px-1.5 py-0.5 text-[0.6875rem] leading-tight text-pending" data-change>
            {changeText(m.change, m.change_s)}
            <div className="text-ink-2">
              next: tool {m.to.tool + 1}, {m.to.tool_label}
            </div>
          </div>
        ) : null}
        {m.kind === "idle" && m.segment ? (
          <div className="bg-paper/85 px-1.5 py-0.5 text-[0.6875rem] leading-tight text-ink-2">
            tool {m.segment.tool + 1}, {m.segment.tool_label}
          </div>
        ) : null}
      </div>
      {ended ? (
        <span
          className={`stamp pointer-events-none absolute bottom-9 left-2 max-w-[90%] bg-paper ${lane.completed ? "text-pass" : "text-fail"}`}
          data-outcome={lane.completed ? "completed" : "stopped"}
        >
          {outcomeText(lane, plan)}
        </span>
      ) : null}
    </>
  );
}

/** The lane's steps and change gaps on its clock, with the playhead. */
function StepStrip({ lane, plan }: { lane: StudioLane; plan: StepsOnly }) {
  const t = usePlayback((s) => s.t);
  const span = Math.max(lane.duration_s, runEnd(lane)) || 1;
  const at = (s: number) => `${(Math.min(Math.max(s, 0), span) / span) * 100}%`;
  const gaps = lane.segments.slice(1).flatMap((seg) => (seg.change === "none" ? [] : [{ seg, start: seg.change_start_s, end: seg.start_s }]));
  return (
    <div className="relative h-5 w-full border border-rule bg-paper-2" aria-hidden="true" data-step-strip>
      {lane.steps.map((s) => (
        <span
          key={s.step_id}
          className={`absolute inset-y-0 overflow-hidden border-x border-paper px-0.5 text-[0.625rem] leading-[1.125rem] ${s.success ? "bg-pass/25 text-ink" : "bg-fail/25 text-fail"}`}
          style={{ left: at(s.start_s), width: `calc(${at(s.start_s + s.duration_s)} - ${at(s.start_s)})` }}
          title={`${stepTitle(plan, s.step_id)}: ${s.success ? "passed" : (s.failure_mode ?? "failed").replace(/_/g, " ")}`}
        >
          {plan.steps.findIndex((p) => p.id === s.step_id) + 1}
        </span>
      ))}
      {gaps.map((g) => (
        <span
          key={`gap-${g.seg.index}`}
          className="absolute inset-y-0 bg-[repeating-linear-gradient(135deg,var(--pending)_0_2px,transparent_2px_5px)] opacity-60"
          style={{ left: at(g.start), width: `calc(${at(g.end)} - ${at(g.start)})` }}
          title={changeText(g.seg.change, g.seg.change_s)}
        />
      ))}
      <span className="absolute -inset-y-0.5 w-0.5 bg-accent" style={{ left: at(t) }} />
    </div>
  );
}

function LaneView({ lane, state, bounds, plan }: { lane: StudioLane; state: LaneState; bounds: Bounds | null; plan: StepsOnly }) {
  const store = usePlaybackStore();
  const view = usePlayback((s) => s.view);
  const colors = useThemeColors();
  const clay = state.status === "ready" ? state.clay : NO_CLAY;
  const options = useMemo(() => ({ xray: false, collision: false, clay, cut: null }), [clay]);
  const stand = state.status === "ready" && lane.kind === "tool_change" ? state.toolStand : null;
  const caption = state.status === "ready" ? replayCaption(state.replay.header) : null;
  const tierOf = lane.steps.some((s) => s.fidelity === "approximation") ? "approximation" : lane.steps.some((s) => s.fidelity === "not_simulated") ? "not_simulated" : "rigid";
  return (
    <figure className="flex min-w-0 flex-col" data-lane={lane.id} data-replay={lane.replay}>
      <div className="crop-marks relative aspect-square bg-paper-2/60 sm:aspect-[4/3]">
        {state.status === "ready" && bounds ? (
          <StudioReplayCanvas
            replay={state.replay}
            assets={state.assets}
            store={store}
            bounds={bounds}
            view={view}
            colors={colors}
            options={options}
            pad={PAD}
          />
        ) : null}
        <div className="pointer-events-none absolute left-2 top-2 flex max-w-[46%] flex-col items-start gap-1">
          <span className="bg-paper/85 px-1.5 py-0.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-accent">{lane.title}</span>
          <span className="heading bg-paper/85 px-1.5 py-0.5 text-xs leading-tight text-ink sm:text-sm">{lane.label}</span>
        </div>
        {state.status === "loading" ? <p className="absolute inset-0 grid place-items-center text-sm text-ink-3">Loading replay</p> : null}
        {state.status === "error" ? (
          <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">This replay could not be loaded: {state.message}</p>
        ) : null}
        <Readout lane={lane} plan={plan} />
        {caption ? (
          <p className="num pointer-events-none absolute inset-x-2 bottom-2 truncate bg-paper/85 px-1.5 py-0.5 text-[0.625rem] text-ink-3" title={caption} data-caption>
            {caption}
          </p>
        ) : null}
        <p className="sr-only">
          {lane.title}: {lane.label}, {lane.condition_text}. {outcomeText(lane, plan)}.
        </p>
      </div>
      <figcaption className="mt-2 grid gap-1.5 text-sm leading-snug text-ink-2">
        <StepStrip lane={lane} plan={plan} />
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>{lane.condition_text}</span>
          <TierBadge tier={tierOf} />
          <span className="num text-xs text-ink-3">trajectory {lane.traj_sha.slice(0, 12)}</span>
        </span>
        {stand ? (
          <span className="block text-xs text-ink-3" data-tool-stand>
            The tools not on the wrist rest on a tool stand behind the table ({stand.replace(/\.$/, "")}).
          </span>
        ) : null}
        {lane.credits.length ? (
          <span className="block text-xs text-ink-3" data-credits>
            {lane.credits.map((c, i) => (
              <span key={c.text}>
                {i > 0 ? "; " : "Models: "}
                {c.text}{" "}
                <Link href={`/credits/#${c.anchor ?? "models"}`} className="link whitespace-nowrap">
                  (full credit)
                </Link>
              </span>
            ))}
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}

/**
 * The recommended (or closest) setup and the best single off-the-shelf tool on the same
 * condition, side by side on one clock. Each lane shows its active step and tool, a tool change
 * or head switch as a labelled gap (a fixed time, not simulated), and its outcome at the end.
 */
export default function StudioLanes({ lanes, plan }: { lanes: StudioLane[]; plan: StepsOnly }) {
  const store = useMemo(() => createPlaybackStore({ look: "studio", view: "iso" }), []);
  const [states, setStates] = useState<Record<string, LaneState>>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const visible = useRef(true);
  useClock(store, visible);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    Promise.all(
      lanes.map(async (lane): Promise<[string, LaneState]> => {
        try {
          const loaded = await loadLane(dataUrl(lane.replay), controller.signal);
          const header = loaded.replay.header;
          const stand = isV2(header) ? (header.meta as Record<string, unknown>).tool_stand : null;
          return [
            lane.id,
            {
              status: "ready",
              replay: paintTools(loaded.replay),
              assets: loaded.assets,
              clay: isV2(header) ? clayAssets(header.assets) : NO_CLAY,
              toolStand: typeof stand === "string" && stand ? stand : null,
            },
          ];
        } catch (err) {
          return [lane.id, { status: "error", message: err instanceof Error ? err.message : String(err) }];
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      setStates(Object.fromEntries(entries));
      const ready = entries.flatMap(([, s]) => (s.status === "ready" ? [s.replay.duration] : []));
      store.getState().setDuration(Math.max(0, ...ready, ...lanes.map((l) => runEnd(l))));
      if (ready.length > 0 && !prefersReducedMotion()) store.getState().play();
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [lanes, store]);

  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => {
      visible.current = entry?.isIntersecting ?? true;
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  /** One camera framing for every lane, so the lanes compare like for like. */
  const bounds = useMemo(() => {
    let b: Bounds | null = null;
    for (const s of Object.values(states)) if (s.status === "ready") b = unionBounds(b, s.replay.bounds(assetPoints(s.replay, s.assets)));
    if (!b) return null;
    const m = VIEW_MARGIN_M;
    return {
      min: [b.min[0] - m, b.min[1] - m, Math.min(b.min[2], 0) - TABLE_BAND_M],
      max: [b.max[0] + m, b.max[1] + m, b.max[2] + m],
    } satisfies Bounds;
  }, [states]);

  const ready = Object.values(states).some((s) => s.status === "ready");
  return (
    <PlaybackContext.Provider value={store}>
      <div ref={rootRef} className="flex flex-col gap-4" data-studio-lanes>
        <StudioTransport disabled={!ready} />
        <div className={`grid grid-cols-1 gap-x-4 gap-y-6 ${lanes.length >= 2 ? "md:grid-cols-2" : ""}`}>
          {lanes.map((lane) => (
            <LaneView key={lane.id} lane={lane} state={states[lane.id] ?? { status: "loading" }} bounds={bounds} plan={plan} />
          ))}
        </div>
      </div>
    </PlaybackContext.Provider>
  );
}
