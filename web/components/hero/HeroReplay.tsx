"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { prefersReducedMotion, useClock } from "@/components/replay/clock";
import { assetPoints, loadLane } from "@/components/replay/lanes";
import { createPlaybackStore, type ViewName } from "@/components/replay/playback";
import StudioReplayCanvas from "@/components/replay/StudioReplayCanvas";
import { useThemeColors } from "@/components/replay/useThemeColors";
import { clayAssets, isThirdParty, replayPayload, type LoadedAsset } from "@/lib/assets";
import { dataUrl } from "@/lib/paths";
import { isV2, replayCaption, type Bounds, type Replay } from "@/lib/replay";
import type { PhaseSpan } from "@/lib/types";

const MARGIN_M = 0.01;

type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; replay: Replay; assets: Map<string, LoadedAsset> | null; clay: ReadonlySet<string> };

export interface HeroReplayProps {
  /** Replay stem relative to /data, without extension. */
  stem: string;
  /** The task's controller phases (the offset readout starts at the close phase). */
  phases: PhaseSpan[];
  /** Short title over the clip, e.g. "Designed jaws, tee placed 2 mm off". */
  title?: string;
  view?: ViewName;
}

/**
 * One replay in the studio look, looping, with the burned-in caption every clip
 * carries ("simulation replay · MuJoCo · run <id> · design <id>", plus "product mode")
 * and, when the replay has overlays, the offset readout ("2.0 -> 0.1 mm"). Branded
 * or unscreened scans are drawn in clay. Visitors who prefer reduced motion get the
 * seated frame and a play button instead of autoplay.
 */
export default function HeroReplay({ stem, phases, title, view = "iso" }: HeroReplayProps) {
  const colors = useThemeColors();
  const store = useMemo(() => createPlaybackStore({ loop: true, view }), [view]);
  const visible = useRef(true);
  const root = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<State>({ status: "loading" });
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  useClock(store, visible);

  useEffect(() => store.subscribe((s) => (setT(s.t), setPlaying(s.playing))), [store]);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    loadLane(dataUrl(stem), controller.signal)
      .then(({ replay, assets }) => {
        if (cancelled) return;
        store.getState().setDuration(replay.duration);
        setState({ status: "ready", replay, assets, clay: isV2(replay.header) ? clayAssets(replay.header.assets) : new Set() });
        const seat = phases.find((p) => p.name === "lift")?.start_s ?? 0;
        if (prefersReducedMotion()) store.getState().seek(seat);
        else store.getState().play();
      })
      .catch((err) => {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [stem, store, phases]);

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
  const bounds = useMemo<Bounds | null>(() => {
    if (!ready) return null;
    const b = ready.replay.bounds(assetPoints(ready.replay, ready.assets));
    return {
      min: [b.min[0] - MARGIN_M, b.min[1] - MARGIN_M, Math.min(b.min[2], 0) - MARGIN_M],
      max: [b.max[0] + MARGIN_M, b.max[1] + MARGIN_M, b.max[2] + MARGIN_M],
    };
  }, [ready]);
  const options = useMemo(() => ({ xray: false, collision: false, clay: ready?.clay ?? new Set<string>(), cut: null }), [ready?.clay]);

  const f = ready ? ready.replay.frameAt(t) : 0;
  const offset = ready?.replay.overlayAt("offset_mm", f) ?? null;
  const closeStart = phases.find((p) => p.name === "close")?.start_s ?? 0;
  const offsetFrom = ready && offset !== null ? ready.replay.overlayAt("offset_mm", ready.replay.frameAt(closeStart)) : null;
  const payload =
    ready && isV2(ready.replay.header) && ready.replay.bytes
      ? replayPayload(ready.replay.header, ready.replay.bytes.header).total
      : null;
  const thirdParty = ready && isV2(ready.replay.header) && Object.values(ready.replay.header.assets).some(isThirdParty);

  return (
    <figure ref={root} className="flex flex-col" data-hero data-payload-bytes={payload ?? undefined}>
      <div className="relative aspect-video overflow-hidden bg-paper-2/60">
        {ready && bounds ? (
          <StudioReplayCanvas
            replay={ready.replay}
            assets={ready.assets}
            store={store}
            bounds={bounds}
            view={view}
            colors={colors}
            options={options}
          />
        ) : null}
        {title ? (
          <p className="pointer-events-none absolute left-3 top-3 max-w-[70%] bg-paper/85 px-2 py-1 text-sm text-ink">{title}</p>
        ) : null}
        {offset !== null && offsetFrom !== null ? (
          <p className="num pointer-events-none absolute right-3 top-3 bg-paper/85 px-2 py-1 text-sm text-ink">
            <span className="text-ink-3">Offset </span>
            {offsetFrom.toFixed(1)} {"→"} {offset.toFixed(2)}
            <span className="text-ink-3"> mm</span>
          </p>
        ) : null}
        {ready ? (
          <p className="num pointer-events-none absolute bottom-2 left-3 right-3 truncate text-[0.6875rem] text-ink-2" data-caption>
            {replayCaption(ready.replay.header)}
          </p>
        ) : null}
        {state.status === "loading" ? <p className="absolute inset-0 grid place-items-center text-sm text-ink-3">Loading replay</p> : null}
        {state.status === "error" ? (
          <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">
            This replay could not be loaded: {state.message}
          </p>
        ) : null}
        {ready ? (
          <button
            type="button"
            onClick={() => store.getState().toggle()}
            className="absolute bottom-7 right-3 bg-paper/85 px-2 py-1 text-xs text-ink-2 hover:text-ink"
          >
            {playing ? "Pause" : "Play"}
          </button>
        ) : null}
      </div>
      {thirdParty ? (
        <figcaption className="mt-1 text-xs text-ink-3">
          Drawn on an open-source parallel gripper.{" "}
          <Link href="/credits/" className="link">
            Credits
          </Link>
        </figcaption>
      ) : null}
    </figure>
  );
}
