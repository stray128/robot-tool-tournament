"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

import { prefersReducedMotion, useClock } from "@/components/replay/clock";
import { loadLane } from "@/components/replay/lanes";
import { createPlaybackStore, PlaybackContext, usePlayback, usePlaybackStore } from "@/components/replay/playback";
import { Transport } from "@/components/replay/Transport";
import { useThemeColors } from "@/components/replay/useThemeColors";
import {
  defaultLanes,
  FAMILY_TITLES,
  FLAG_LABELS,
  outcomeLabel,
  REQUIREMENT_LABELS,
  roleLabel,
  type EffectorLane,
  type EffectorTask,
} from "@/lib/effectors";
import { formatLift } from "@/lib/format";
import { dataUrl } from "@/lib/paths";
import { Replay, replayCaption, unionBounds, type Bounds, type LaneTracks } from "@/lib/replay";

import { FidelityBadge } from "./FidelityBadge";

const ReplayCanvas = dynamic(() => import("@/components/replay/ReplayCanvas"), { ssr: false });
const StudioReplayCanvas = dynamic(() => import("@/components/replay/StudioReplayCanvas"), { ssr: false });

const STUDIO_PAD = { top: 64, bottom: 84 };
const VIEW_MARGIN_M = 0.008;
const TABLE_BAND_M = 0.015;
const NO_CLAY: ReadonlySet<string> = new Set();

type LaneState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; replay: Replay; tracks: LaneTracks; datum: { center: [number, number]; z: number } | null };

/**
 * The effector's bodies drawn in the designed-part colour. The shared renderers colour a body by
 * its name (`finger*` is the designed tool); an effector's bodies are named `eff_*` in the
 * published lane, so they are aliased here for drawing only (the data is not changed).
 */
function paintEffector(replay: Replay): Replay {
  const header = replay.header;
  const bodies = header.bodies.map((b) => (b.name.startsWith("eff_") ? { ...b, name: `finger_${b.name}` } : b));
  return new Replay({ ...header, bodies } as typeof header, replay.frames, replay.overlays, replay.bytes);
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

function LaneOverlay({ lane, state, task }: { lane: EffectorLane; state: LaneState; task: EffectorTask }) {
  const t = usePlayback((s) => s.t);
  if (state.status !== "ready") return null;
  const { replay, tracks } = state;
  const f = Math.round(replay.frameAt(t));
  const lift = tracks.lift[f] ?? 0;
  const hold = tracks.hold[f] ?? 0;
  const decided = t >= tracks.decidedAt;
  const offset = replay.overlayAt("offset_mm", f);
  const contacts = replay.overlayAt("n_contacts", f);
  const holdFrac = Math.min(hold / task.success_hold_s, 1);
  const o = lane.outcome;
  const flags = o.registration_flags.map((x) => FLAG_LABELS[x] ?? x.replace(/_/g, " "));
  return (
    <>
      <div className="pointer-events-none absolute right-2 top-2 flex flex-col items-end gap-1">
        <Readout label="Lift" value={formatLift(lift)} unit="mm" />
        {offset !== null && Number.isFinite(offset) ? <Readout label="Offset" value={offset.toFixed(1)} unit="mm" /> : null}
        {contacts !== null && Number.isFinite(contacts) ? <Readout label="Contacts" value={String(Math.round(contacts))} unit="" /> : null}
      </div>
      <div className="pointer-events-none absolute inset-x-2 bottom-2 flex flex-col gap-1.5">
        {decided ? (
          <span
            className={`stamp self-start bg-paper ${o.registered ? "text-pass" : o.held ? "text-pending" : "text-fail"}`}
            title={flags.length ? flags.join(", ") : undefined}
          >
            {outcomeLabel(o.registered, o.held, o.failure_mode)}
          </span>
        ) : null}
        <div className="bg-paper/85 px-1.5 py-1">
          <div className="flex items-baseline justify-between text-[0.6875rem] text-ink-3">
            <span>Hold above target</span>
            <span className="num text-ink">
              {hold.toFixed(2)}
              <span className="text-ink-3"> / {task.success_hold_s.toFixed(1)} s</span>
            </span>
          </div>
          <div className="mt-1 h-1.5 w-full bg-paper-3" aria-hidden="true">
            <div className={`h-full ${holdFrac >= 1 ? "bg-pass" : "bg-accent"}`} style={{ width: `${holdFrac * 100}%` }} />
          </div>
        </div>
        <p className="num truncate bg-paper/85 px-1.5 py-0.5 text-[0.625rem] text-ink-3" title={replayCaption(replay.header)} data-caption>
          {replayCaption(replay.header)}
        </p>
      </div>
    </>
  );
}

function LaneView({ lane, state, bounds, task }: { lane: EffectorLane; state: LaneState; bounds: Bounds | null; task: EffectorTask }) {
  const store = usePlaybackStore();
  const view = usePlayback((s) => s.view);
  const look = usePlayback((s) => s.look);
  const colors = useThemeColors();
  const options = useMemo(() => ({ xray: false, collision: false, clay: NO_CLAY, cut: null }), []);
  const o = lane.outcome;
  return (
    <figure className="flex min-w-0 flex-col" data-lane={lane.replay} data-effector={lane.effector.effector_id}>
      <div className="crop-marks relative aspect-square bg-paper-2/60 sm:aspect-[4/5]">
        {state.status === "ready" && bounds && look === "studio" ? (
          <StudioReplayCanvas
            replay={state.replay}
            assets={null}
            store={store}
            bounds={bounds}
            view={view}
            colors={colors}
            datum={state.datum}
            options={options}
            pad={STUDIO_PAD}
          />
        ) : null}
        {state.status === "ready" && bounds && look === "drawing" ? (
          <ReplayCanvas replay={state.replay} store={store} bounds={bounds} view={view} colors={colors} datum={state.datum} />
        ) : null}
        <div className="pointer-events-none absolute left-2 top-2 flex max-w-[62%] flex-col items-start gap-1">
          <span className="bg-paper/85 px-1.5 py-0.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-accent">
            {lane.roles.filter((r) => r !== "finalist").map(roleLabel).join(" · ") || "Finalist"}
          </span>
          <span className="heading bg-paper/85 px-1.5 py-0.5 text-sm text-ink">{lane.effector.display_name}</span>
        </div>
        {state.status === "loading" ? <p className="absolute inset-0 grid place-items-center text-sm text-ink-3">Loading replay</p> : null}
        {state.status === "error" ? (
          <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">This replay could not be loaded: {state.message}</p>
        ) : null}
        <LaneOverlay lane={lane} state={state} task={task} />
        <p className="sr-only">
          {lane.effector.display_name}: {outcomeLabel(o.registered, o.held, o.failure_mode)}. Peak lift {o.max_lift_mm.toFixed(1)} mm.
        </p>
      </div>
      <figcaption className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm leading-snug text-ink-2">
        <span>{FAMILY_TITLES[lane.effector.family]}</span>
        <FidelityBadge fidelity={lane.effector.fidelity} />
        <span className="text-ink-3">
          {REQUIREMENT_LABELS.held}: {o.held ? "yes" : "no"} · {REQUIREMENT_LABELS.registered}: {o.registered ? "yes" : "no"}
        </span>
        {lane.effector.note ? (
          <span className="basis-full text-xs text-pending" data-note>
            {lane.effector.note}
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}

export interface EffectorLanesProps {
  lanes: EffectorLane[];
  task: EffectorTask;
}

/** Every finalist's replay of the same condition on one clock, the recommendation first. */
export default function EffectorLanes({ lanes, task }: EffectorLanesProps) {
  const first = useMemo(() => defaultLanes(lanes), [lanes]);
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? lanes : first;
  const store = useMemo(() => createPlaybackStore({ look: "studio" }), []);
  const [states, setStates] = useState<Record<string, LaneState>>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const visible = useRef(true);
  useClock(store, visible);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    const missing = shown.filter((l) => !states[l.id] || states[l.id]!.status === "error");
    if (missing.length === 0) return;
    Promise.all(
      missing.map(async (lane): Promise<[string, LaneState]> => {
        try {
          const loaded = await loadLane(dataUrl(lane.replay), controller.signal);
          const replay = paintEffector(loaded.replay);
          const ref = replay.objectReference(task);
          return [
            lane.id,
            {
              status: "ready",
              replay,
              tracks: replay.tracks(task),
              datum: ref ? { center: [ref[0], ref[1]], z: ref[2] + task.success_lift_m } : null,
            },
          ];
        } catch (err) {
          return [lane.id, { status: "error", message: err instanceof Error ? err.message : String(err) }];
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      setStates((prev) => ({ ...prev, ...Object.fromEntries(entries) }));
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
    // `states` is read only to skip lanes already loaded; re-running on its change would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown, task]);

  const ready = useMemo(
    () => shown.map((l) => states[l.id]).filter((s): s is Extract<LaneState, { status: "ready" }> => s?.status === "ready"),
    [shown, states],
  );
  useEffect(() => {
    if (ready.length === 0) return;
    store.getState().setDuration(Math.max(0, ...ready.map((s) => s.replay.duration)));
  }, [ready, store]);
  const autoplayed = useRef(false);
  useEffect(() => {
    if (!autoplayed.current && ready.length === shown.length && ready.length > 0) {
      autoplayed.current = true;
      if (!prefersReducedMotion()) store.getState().play();
    }
  }, [ready.length, shown.length, store]);

  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => {
      visible.current = entry?.isIntersecting ?? true;
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const bounds = useMemo(() => {
    let b: Bounds | null = null;
    for (const s of ready) b = unionBounds(b, s.replay.bounds());
    if (!b) return null;
    const m = VIEW_MARGIN_M;
    return {
      min: [b.min[0] - m, b.min[1] - m, Math.min(b.min[2], 0) - TABLE_BAND_M],
      max: [b.max[0] + m, b.max[1] + m, b.max[2] + m],
    } satisfies Bounds;
  }, [ready]);

  const cols =
    shown.length >= 4 ? "sm:grid-cols-2 lg:grid-cols-4" : shown.length === 3 ? "sm:grid-cols-3" : shown.length === 2 ? "sm:grid-cols-2" : "";

  return (
    <PlaybackContext.Provider value={store}>
      <div ref={rootRef} className="flex flex-col gap-4" data-effector-lanes>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0 flex-1">
            <Transport phases={task.phases} disabled={ready.length === 0} features={{ look: true, collision: false }} />
          </div>
          {lanes.length > first.length ? (
            <button
              type="button"
              className="num border border-rule px-2 py-1 text-xs text-ink-2 hover:text-ink"
              aria-pressed={showAll}
              onClick={() => setShowAll((v) => !v)}
            >
              {showAll ? `Group bests (${first.length})` : `All finalists (${lanes.length})`}
            </button>
          ) : null}
        </div>
        <div className={`grid grid-cols-1 gap-x-4 gap-y-6 ${cols}`}>
          {shown.map((lane) => (
            <LaneView key={lane.id} lane={lane} state={states[lane.id] ?? { status: "loading" }} bounds={bounds} task={task} />
          ))}
        </div>
      </div>
    </PlaybackContext.Provider>
  );
}
