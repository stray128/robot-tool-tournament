"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { capitalizeFirst, formatLift } from "@/lib/format";
import { FAILURE_EXPLANATIONS, FAILURE_LABELS } from "@/lib/labels";
import { dataUrl } from "@/lib/paths";
import { loadReplay, unionBounds, type Bounds, type LaneTracks, type Replay } from "@/lib/replay";
import type { Bout, Lane } from "@/lib/types";

import { createPlaybackStore, PlaybackContext, usePlayback, usePlaybackStore, type PlaybackStore } from "./playback";
import { Transport } from "./Transport";
import { useThemeColors } from "./useThemeColors";

const ReplayCanvas = dynamic(() => import("./ReplayCanvas"), { ssr: false });

/** Seconds the last frame stays on screen before a looping player restarts. */
const LOOP_PAUSE_S = 1.6;
/** Margin added around the bodies' swept volume, metres. */
const VIEW_MARGIN_M = 0.008;
/** Table depth kept in view below its top surface, metres. */
const TABLE_BAND_M = 0.015;

type LaneState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; replay: Replay; tracks: LaneTracks; datum: { center: [number, number]; z: number } | null };

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Drives the shared clock with requestAnimationFrame while playing and visible. */
function useClock(store: PlaybackStore, visible: React.RefObject<boolean>) {
  useEffect(() => {
    let raf = 0;
    let last = 0;
    let restart: ReturnType<typeof setTimeout> | undefined;
    const frame = (now: number) => {
      const dt = last ? Math.min((now - last) / 1000, 0.1) : 0;
      last = now;
      if (visible.current) {
        const ended = store.getState().advance(dt);
        if (ended) {
          const { loop } = store.getState();
          store.getState().pause();
          if (loop) {
            restart = setTimeout(() => {
              store.getState().seek(0);
              store.getState().play();
            }, LOOP_PAUSE_S * 1000);
          }
          return;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    const unsubscribe = store.subscribe((s, prev) => {
      if (s.playing && !prev.playing) {
        last = 0;
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(frame);
      } else if (!s.playing && prev.playing) {
        cancelAnimationFrame(raf);
      }
    });
    if (store.getState().playing) raf = requestAnimationFrame(frame);
    return () => {
      unsubscribe();
      cancelAnimationFrame(raf);
      if (restart) clearTimeout(restart);
    };
  }, [store, visible]);
}

function Readout({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="bg-paper/85 px-1.5 py-0.5 text-right leading-tight">
      <div className="text-[0.6875rem] text-ink-3">{label}</div>
      <div className="num text-sm text-ink">
        {value}
        <span className="ml-0.5 text-ink-3">{unit}</span>
      </div>
    </div>
  );
}

/** Lift readout, hold timer and verdict for one lane at the current playback time. */
function LaneOverlay({ lane, state, holdTarget }: { lane: Lane; state: LaneState; holdTarget: number }) {
  const t = usePlayback((s) => s.t);
  if (state.status !== "ready") return null;
  const { replay, tracks } = state;
  const f = Math.round(replay.frameAt(t));
  const lift = tracks.lift[f] ?? 0;
  const hold = tracks.hold[f] ?? 0;
  const decided = t >= tracks.decidedAt;
  const { success, failure_mode } = lane.outcome;
  const holdFrac = Math.min(hold / holdTarget, 1);

  return (
    <>
      <div className="pointer-events-none absolute right-2 top-2">
        <Readout label="Lift" value={formatLift(lift)} unit="mm" />
      </div>
      <div className="pointer-events-none absolute inset-x-2 bottom-2 flex flex-col gap-1.5">
        {decided ? (
          <span
            className={`stamp self-start bg-paper ${success ? "text-pass" : "text-fail"}`}
            title={failure_mode ? FAILURE_EXPLANATIONS[failure_mode] : undefined}
          >
            {success ? `Held ${lane.outcome.hold_s.toFixed(1)} s` : failure_mode ? FAILURE_LABELS[failure_mode] : "Failed"}
          </span>
        ) : null}
        <div className="bg-paper/85 px-1.5 py-1">
          <div className="flex items-baseline justify-between text-[0.6875rem] text-ink-3">
            <span>Hold above target</span>
            <span className="num text-ink">
              {hold.toFixed(2)}
              <span className="text-ink-3"> / {holdTarget.toFixed(1)} s</span>
            </span>
          </div>
          <div className="mt-1 h-1.5 w-full bg-paper-3" aria-hidden="true">
            <div
              className={`h-full ${holdFrac >= 1 ? "bg-pass" : "bg-accent"}`}
              style={{ width: `${holdFrac * 100}%` }}
            />
          </div>
        </div>
      </div>
    </>
  );
}

function LaneView({
  lane,
  state,
  bounds,
  bout,
  linkDesign,
}: {
  lane: Lane;
  state: LaneState;
  bounds: Bounds | null;
  bout: Bout;
  linkDesign: boolean;
}) {
  const store = usePlaybackStore();
  const view = usePlayback((s) => s.view);
  const colors = useThemeColors();
  const outcome = lane.outcome.success
    ? `held for ${lane.outcome.hold_s.toFixed(1)} s`
    : lane.outcome.failure_mode
      ? FAILURE_LABELS[lane.outcome.failure_mode].toLowerCase()
      : "failed";

  return (
    <figure className="flex min-w-0 flex-col">
      <div className="crop-marks relative aspect-square bg-paper-2/60 sm:aspect-[4/5]">
        {state.status === "ready" && bounds ? (
          <ReplayCanvas
            replay={state.replay}
            store={store}
            bounds={bounds}
            view={view}
            colors={colors}
            datum={state.datum}
          />
        ) : null}
        <div className="pointer-events-none absolute left-2 top-2 max-w-[60%] bg-paper/85 px-1.5 py-0.5">
          <span className="heading text-sm text-ink">{lane.label}</span>
        </div>
        {state.status === "loading" ? (
          <p className="absolute inset-0 grid place-items-center text-sm text-ink-3">Loading replay</p>
        ) : null}
        {state.status === "error" ? (
          <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">
            This replay could not be loaded: {state.message}
          </p>
        ) : null}
        <LaneOverlay lane={lane} state={state} holdTarget={bout.task.success_hold_s} />
        <p className="sr-only">
          {lane.label}: {outcome}. Peak lift {lane.outcome.max_lift_mm.toFixed(1)} mm.
        </p>
      </div>
      <figcaption className="mt-2 text-sm leading-snug text-ink-2">
        {capitalizeFirst(lane.condition)}
        {linkDesign ? (
          <>
            {" "}
            <Link href={`/design/${lane.design_id}/`} className="link whitespace-nowrap">
              {lane.entrant} finger
            </Link>
          </>
        ) : null}
      </figcaption>
    </figure>
  );
}

export interface BoutPlayerProps {
  bout: Bout;
  /** Start playing once loaded (skipped when the visitor prefers reduced motion). */
  autoPlay?: boolean;
  loop?: boolean;
  /** Link each lane's caption to its design page. */
  linkDesigns?: boolean;
  /** Design ids that have a design page. */
  designIds?: string[];
}

/** Synchronised multi-lane replay of one bout with a shared transport. */
export default function BoutPlayer({ bout, autoPlay = false, loop = false, linkDesigns = true, designIds }: BoutPlayerProps) {
  const store = useMemo(() => createPlaybackStore({ loop }), [loop]);
  const [lanes, setLanes] = useState<Record<string, LaneState>>(() =>
    Object.fromEntries(bout.lanes.map((l) => [l.id, { status: "loading" } as LaneState])),
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const visible = useRef(true);
  useClock(store, visible);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    Promise.all(
      bout.lanes.map(async (lane): Promise<[string, LaneState]> => {
        try {
          const replay = await loadReplay(dataUrl(lane.replay), { signal: controller.signal });
          const ref = replay.objectReference(bout.task);
          return [
            lane.id,
            {
              status: "ready",
              replay,
              tracks: replay.tracks(bout.task),
              datum: ref ? { center: [ref[0], ref[1]], z: ref[2] + bout.task.success_lift_m } : null,
            },
          ];
        } catch (err) {
          return [lane.id, { status: "error", message: err instanceof Error ? err.message : String(err) }];
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      setLanes(Object.fromEntries(entries));
      const durations = entries.flatMap(([, s]) => (s.status === "ready" ? [s.replay.duration] : []));
      store.getState().setDuration(Math.max(0, ...durations));
      if (autoPlay && !prefersReducedMotion()) store.getState().play();
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [bout, store, autoPlay]);

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
    for (const s of Object.values(lanes)) if (s.status === "ready") b = unionBounds(b, s.replay.bounds());
    if (!b) return null;
    const m = VIEW_MARGIN_M;
    return {
      min: [b.min[0] - m, b.min[1] - m, Math.min(b.min[2], 0) - TABLE_BAND_M],
      max: [b.max[0] + m, b.max[1] + m, b.max[2] + m],
    } satisfies Bounds;
  }, [lanes]);

  const cols =
    bout.lanes.length >= 4
      ? "sm:grid-cols-2 lg:grid-cols-4"
      : bout.lanes.length === 3
        ? "sm:grid-cols-3"
        : bout.lanes.length === 2
          ? "sm:grid-cols-2"
          : "";
  const ready = Object.values(lanes).some((s) => s.status === "ready");
  const known = designIds ? new Set(designIds) : null;

  return (
    <PlaybackContext.Provider value={store}>
      <div ref={rootRef} className="flex flex-col gap-4">
        <Transport phases={bout.task.phases} disabled={!ready} />
        <div className={`grid grid-cols-1 gap-x-4 gap-y-6 ${cols}`}>
          {bout.lanes.map((lane) => (
            <LaneView
              key={lane.id}
              lane={lane}
              state={lanes[lane.id] ?? { status: "loading" }}
              bounds={bounds}
              bout={bout}
              linkDesign={linkDesigns && (!known || known.has(lane.design_id))}
            />
          ))}
        </div>
      </div>
    </PlaybackContext.Provider>
  );
}
