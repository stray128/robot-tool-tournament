"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { clayAssets, isThirdParty, type LoadedAsset } from "@/lib/assets";
import { capitalizeFirst, formatLift } from "@/lib/format";
import { FAILURE_EXPLANATIONS, FAILURE_LABELS } from "@/lib/labels";
import { dataUrl } from "@/lib/paths";
import { isV2, replayCaption, unionBounds, type Bounds, type LaneTracks, type Replay } from "@/lib/replay";
import type { Bout, Lane, PhaseSpan } from "@/lib/types";

import { prefersReducedMotion, useClock } from "./clock";
import { assetPoints, loadLane } from "./lanes";
import { createPlaybackStore, PlaybackContext, usePlayback, usePlaybackStore } from "./playback";
import { Transport } from "./Transport";
import { useThemeColors } from "./useThemeColors";

const ReplayCanvas = dynamic(() => import("./ReplayCanvas"), { ssr: false });
const StudioReplayCanvas = dynamic(() => import("./StudioReplayCanvas"), { ssr: false });

/** Space kept free above and below a studio lane for the HTML readouts, CSS px. */
const STUDIO_PAD = { top: 52, bottom: 76 };
/** Margin added around the bodies' swept volume, metres. */
const VIEW_MARGIN_M = 0.008;
/** Table depth kept in view below its top surface, metres. */
const TABLE_BAND_M = 0.015;

type LaneState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      replay: Replay;
      tracks: LaneTracks;
      datum: { center: [number, number]; z: number } | null;
      /** GLB assets of a version 2 replay (null for version 1). */
      assets: Map<string, LoadedAsset> | null;
      /** Asset ids drawn untextured (brand screen). */
      clay: ReadonlySet<string>;
    };

const NO_CLAY: ReadonlySet<string> = new Set();

/** Start of the close phase, where a placement offset is read before the jaws act on it. */
function closeStart(phases: PhaseSpan[]): number {
  return phases.find((p) => p.name === "close")?.start_s ?? 0;
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
function LaneOverlay({
  lane,
  state,
  holdTarget,
  phases,
}: {
  lane: Lane;
  state: LaneState;
  holdTarget: number;
  phases: PhaseSpan[];
}) {
  const t = usePlayback((s) => s.t);
  if (state.status !== "ready") return null;
  const { replay, tracks } = state;
  const f = Math.round(replay.frameAt(t));
  const lift = tracks.lift[f] ?? 0;
  const hold = tracks.hold[f] ?? 0;
  const decided = t >= tracks.decidedAt;
  const { success, failure_mode } = lane.outcome;
  const holdFrac = Math.min(hold / holdTarget, 1);
  const offset = replay.overlayAt("offset_mm", f);
  const offsetFrom = offset === null ? null : replay.overlayAt("offset_mm", replay.frameAt(closeStart(phases)));

  return (
    <>
      <div className="pointer-events-none absolute right-2 top-2 flex flex-col items-end gap-1">
        <Readout label="Lift" value={formatLift(lift)} unit="mm" />
        {offset !== null && offsetFrom !== null ? (
          <Readout label="Offset" value={`${offsetFrom.toFixed(1)} \u2192 ${offset.toFixed(1)}`} unit="mm" />
        ) : null}
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
        {isV2(replay.header) ? (
          <p
            className="num truncate bg-paper/85 px-1.5 py-0.5 text-[0.625rem] text-ink-3"
            title={replayCaption(replay.header)}
            data-caption
          >
            {replayCaption(replay.header)}
          </p>
        ) : null}
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
  const look = usePlayback((s) => s.look);
  const xray = usePlayback((s) => s.xray);
  const collision = usePlayback((s) => s.collision);
  const colors = useThemeColors();
  const studio = state.status === "ready" && (state.assets !== null || look === "studio");
  const clay = state.status === "ready" ? state.clay : NO_CLAY;
  const options = useMemo(() => ({ xray, collision, clay, cut: null }), [xray, collision, clay]);
  const thirdParty =
    state.status === "ready" && isV2(state.replay.header) && Object.values(state.replay.header.assets).some(isThirdParty);
  const outcome = lane.outcome.success
    ? `held for ${lane.outcome.hold_s.toFixed(1)} s`
    : lane.outcome.failure_mode
      ? FAILURE_LABELS[lane.outcome.failure_mode].toLowerCase()
      : "failed";

  return (
    <figure className="flex min-w-0 flex-col" data-lane={lane.replay}>
      <div className="crop-marks relative aspect-square bg-paper-2/60 sm:aspect-[4/5]">
        {state.status === "ready" && bounds && studio ? (
          <StudioReplayCanvas
            replay={state.replay}
            assets={state.assets}
            store={store}
            bounds={bounds}
            view={view}
            colors={colors}
            datum={state.datum}
            options={options}
            pad={STUDIO_PAD}
          />
        ) : null}
        {state.status === "ready" && bounds && !studio ? (
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
        <LaneOverlay lane={lane} state={state} holdTarget={bout.task.success_hold_s} phases={bout.task.phases} />
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
        {thirdParty ? (
          <span className="block text-xs text-ink-3">
            Drawn on an open-source parallel gripper.{" "}
            <Link href="/credits/" className="link">
              Credits
            </Link>
          </span>
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
          const { replay, assets } = await loadLane(dataUrl(lane.replay), controller.signal);
          const ref = replay.objectReference(bout.task);
          return [
            lane.id,
            {
              status: "ready",
              replay,
              tracks: replay.tracks(bout.task),
              datum: ref ? { center: [ref[0], ref[1]], z: ref[2] + bout.task.success_lift_m } : null,
              assets,
              clay: isV2(replay.header) ? clayAssets(replay.header.assets) : NO_CLAY,
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
      // Version 2 replays draw GLB assets, which only the studio look renders.
      if (entries.some(([, s]) => s.status === "ready" && s.assets !== null)) store.getState().setLook("studio");
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
    for (const s of Object.values(lanes)) if (s.status === "ready") b = unionBounds(b, s.replay.bounds(assetPoints(s.replay, s.assets)));
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
  const readyLanes = Object.values(lanes).filter((s) => s.status === "ready");
  const features = {
    // The drawing look exists for version 1 lanes only; a bout with v2 lanes is always studio.
    look: readyLanes.length > 0 && readyLanes.every((s) => s.assets === null),
    collision: readyLanes.some((s) => s.replay.header.bodies.some((b) => b.geoms.length > 0)),
  };

  return (
    <PlaybackContext.Provider value={store}>
      <div ref={rootRef} className="flex flex-col gap-4">
        <Transport phases={bout.task.phases} disabled={!ready} features={features} />
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
