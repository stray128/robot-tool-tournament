"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

import { prefersReducedMotion, useClock } from "@/components/replay/clock";
import { objectBounds } from "@/components/replay/framing";
import { loadLane } from "@/components/replay/lanes";
import { createPlaybackStore, watchPlayback } from "@/components/replay/playback";
import { useThemeColors } from "@/components/replay/useThemeColors";
import { type StudioLane } from "@/components/studio/schema";
import { clayAssets, type LoadedAsset } from "@/lib/assets";
import { dataUrl } from "@/lib/paths";
import { type Bounds, isV2, Replay } from "@/lib/replay";

import { heroReadout, type HeroStep, sameReadout } from "./heroReadout";

export type { HeroStep } from "./heroReadout";

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; replay: Replay; assets: Map<string, LoadedAsset> | null; clay: ReadonlySet<string> };

/** three.js and the renderer load only in the browser, after the poster and the text. */
const StudioReplayCanvas = dynamic(() => import("@/components/replay/StudioReplayCanvas"), { ssr: false });

/** One turn every ~105 s while playing. */
const SPIN_RAD_S = 0.06;
/** Screen space kept for the label above and the step readout below, CSS px. */
const HERO_PAD = { top: 40, bottom: 92 };

/** Tool bodies are named eff_*; the shared renderer colours finger* as the tool (drawing only). */
function paintTools(replay: Replay): Replay {
  const header = replay.header;
  const bodies = header.bodies.map((b) => (/(^|_)eff_/.test(b.name) && !b.name.startsWith("finger") ? { ...b, name: `finger_${b.name}` } : b));
  return new Replay({ ...header, bodies } as typeof header, replay.frames, replay.overlays, replay.bytes);
}

/**
 * The home page's hero: the recorded run of the featured use case's recommendation (the returns
 * tote), looping with a slow turn of the camera. The poster is in the page's HTML; the replay data and the 3D renderer load
 * after the page has painted, when the browser is idle. The overlay names the step on screen, in
 * sync with the replay clock.
 */
export default function HeroScene({
  lane,
  steps,
  setup,
  useCase,
  posterDark,
  posterLight,
}: {
  lane: StudioLane;
  /** The use case's title, e.g. "Returns tote". */
  useCase: string;
  /** The setup in words, e.g. "two-tool head (a soft arm plus a suction cup)": its kind first, up to the first comma or bracket. */
  setup: string;
  steps: HeroStep[];
  posterDark: string;
  posterLight: string;
}) {
  const colors = useThemeColors();
  const store = useMemo(() => createPlaybackStore({ loop: true, look: "studio", view: "iso" }), []);
  const visible = useRef(true);
  const root = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<State>({ status: "loading" });
  const [readout, setReadout] = useState(() => heroReadout(lane, steps, 0));
  const [playing, setPlaying] = useState(false);
  const [shown, setShown] = useState(false);
  useClock(store, visible);

  // React state follows the clock only when the readout or the play state changes, never per frame.
  useEffect(() => watchPlayback(store, (s) => heroReadout(lane, steps, s.t), setReadout, sameReadout), [store, lane, steps]);
  useEffect(() => watchPlayback(store, (s) => s.playing, setPlaying), [store]);

  // Development only: lets the poster script pause the hero on a chosen frame.
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    (window as Window & { __heroStore?: typeof store }).__heroStore = store;
  }, [store]);

  // Load after first paint, when the browser is idle, so the 3D never delays the page's text.
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    const start = () => {
      loadLane(dataUrl(lane.replay), controller.signal)
        .then(({ replay, assets }) => {
          if (cancelled) return;
          store.getState().setDuration(replay.duration);
          setState({ status: "ready", replay: paintTools(replay), assets, clay: isV2(replay.header) ? clayAssets(replay.header.assets) : new Set() });
          const second = lane.steps[1];
          if (prefersReducedMotion()) store.getState().seek(second ? second.start_s + second.duration_s * 0.5 : 0);
          else store.getState().play();
        })
        .catch((err) => {
          if (!cancelled && !controller.signal.aborted) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
        });
    };
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number; cancelIdleCallback?: (id: number) => void };
    const id = w.requestIdleCallback ? w.requestIdleCallback(start, { timeout: 1200 }) : window.setTimeout(start, 300);
    return () => {
      cancelled = true;
      controller.abort();
      if (w.cancelIdleCallback) w.cancelIdleCallback(id);
      else window.clearTimeout(id);
    };
  }, [lane, store]);

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => {
      visible.current = entry?.isIntersecting ?? true;
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const ready = state.status === "ready" ? state : null;
  useEffect(() => {
    if (!ready) return;
    const id = window.setTimeout(() => setShown(true), 250);
    return () => window.clearTimeout(id);
  }, [ready]);

  const bounds = useMemo<Bounds | null>(() => (ready ? objectBounds(ready.replay, ready.assets) : null), [ready]);
  const options = useMemo(() => ({ xray: false, collision: false, clay: ready?.clay ?? new Set<string>(), cut: null }), [ready?.clay]);

  const { current, done, moment, title, detail } = readout;

  return (
    <figure ref={root} className="relative" data-hero-scene>
      <div className="crop-marks relative aspect-[4/3] overflow-hidden bg-paper-2">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={posterDark} alt="" aria-hidden="true" className="poster-dark absolute inset-0 size-full object-cover" fetchPriority="high" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={posterLight} alt="" aria-hidden="true" className="poster-light absolute inset-0 size-full object-cover" />
        {ready && bounds ? (
          <div className={`absolute inset-0 transition-opacity duration-700 ${shown ? "opacity-100" : "opacity-0"}`}>
            <StudioReplayCanvas
              replay={ready.replay}
              assets={ready.assets}
              store={store}
              bounds={bounds}
              view="iso"
              colors={colors}
              options={options}
              pad={HERO_PAD}
              spin={SPIN_RAD_S}
              zoom={1.5}
            />
          </div>
        ) : null}

        <div className="pointer-events-none absolute inset-x-3 top-3 flex items-start justify-between gap-3">
          <span className="rounded bg-paper/85 px-2 py-1 text-xs text-ink-2">Recorded physics run, not an animation</span>
          {state.status === "error" ? <span className="rounded bg-paper/85 px-2 py-1 text-xs text-fail">3D unavailable: {state.message}</span> : null}
        </div>

        <div className="absolute inset-x-3 bottom-3 flex items-end gap-3">
          <div className="pointer-events-none min-w-0 flex-1 rounded-md bg-paper/90 px-3 py-2 backdrop-blur-sm" aria-live="off" data-hero-step={moment}>
            <p className="text-sm font-semibold text-ink">{title || `The recommended ${setup.split(/\s*[,(]/)[0]}, on the ${useCase.toLowerCase()}`}</p>
            <p className="mt-0.5 line-clamp-2 text-xs text-ink-2">{detail || "Loading the recorded run."}</p>
            <ol className="mt-2 flex gap-1" aria-hidden="true">
              {steps.map((s, i) => (
                <li
                  key={s.id}
                  className={`h-1 flex-1 rounded-full ${i < current || done ? "bg-pass" : i === current ? "bg-accent" : "bg-muted-bar"}`}
                />
              ))}
            </ol>
          </div>
          <button
            type="button"
            onClick={() => store.getState().toggle()}
            disabled={!ready}
            className="inline-flex size-10 shrink-0 items-center justify-center rounded-md border border-rule-strong bg-paper/90 text-ink hover:border-accent disabled:opacity-50"
            aria-label={playing ? "Pause the replay" : "Play the replay"}
            data-hero-toggle
          >
            <svg viewBox="0 0 12 12" className="size-3.5" aria-hidden="true">
              {playing ? <path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" /> : <path d="M2.5 1l8 5-8 5z" fill="currentColor" />}
            </svg>
          </button>
        </div>
      </div>
      <figcaption className="mt-3 text-sm text-ink-3" data-text-equivalent>
        What it shows: the recommendation for the {useCase.toLowerCase()}, a {setup}, on its recorded run ({lane.condition_text}).{" "}
        {steps.map((s, i) => `${i + 1}. ${s.name}`).join("; ")}.{" "}
        {lane.completed && lane.cycle_time_s !== null ? `Done in ${lane.cycle_time_s.toFixed(1)} s.` : ""}
      </figcaption>
    </figure>
  );
}
