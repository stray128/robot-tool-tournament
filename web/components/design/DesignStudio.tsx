"use client";

import { useEffect, useId, useMemo, useState } from "react";
import * as THREE from "three";

import { assetPoints, loadLane } from "@/components/replay/lanes";
import { createPlaybackStore } from "@/components/replay/playback";
import { ReplayCredits } from "@/components/replay/ReplayCredits";
import StudioReplayCanvas, { type StudioOptions } from "@/components/replay/StudioReplayCanvas";
import { useThemeColors } from "@/components/replay/useThemeColors";
import { clayAssets, type LoadedAsset } from "@/lib/assets";
import { dataUrl } from "@/lib/paths";
import { geomPoints, isParkedGeom, isV2, replayCaption, rotate, type Bounds, type Replay } from "@/lib/replay";
import type { PhaseSpan } from "@/lib/types";

/** One compiled finger pair to show: a replay of this design on one part. */
export interface DesignStudioEntry {
  id: string;
  /** Button label, e.g. "Box 90 x 40 x 120 mm". */
  label: string;
  /** Replay stem relative to /data, without extension. */
  stem: string;
}

type Mode = "studio" | "xray" | "cutaway";

const MODES: { id: Mode; label: string; help: string }[] = [
  { id: "studio", label: "Studio", help: "The jaws seated on the part, as simulated." },
  { id: "xray", label: "X-ray", help: "The part at 35% opacity, so the pads show where they touch it." },
  { id: "cutaway", label: "Cutaway", help: "A horizontal section through the part and the jaws; drag the slider to move it." },
];

/** Camera directions (three.js coordinates, target to camera). */
const ANGLED = new THREE.Vector3(1, 0.55, 0.62).normalize();
const FROM_ABOVE = new THREE.Vector3(0.35, 1.0, 0.45).normalize();

const MARGIN_M = 0.006;

type Loaded = {
  replay: Replay;
  assets: Map<string, LoadedAsset> | null;
  clay: ReadonlySet<string>;
};

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Height range (MuJoCo z) at frame `f` of the named bodies, or null when none of them is drawn. */
function heightRange(
  replay: Replay,
  assets: Map<string, LoadedAsset> | null,
  f: number,
  names: (n: string) => boolean,
): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  const pos = [0, 0, 0];
  const quat = [0, 0, 0, 1];
  const out = [0, 0, 0];
  const extra = assetPoints(replay, assets);
  replay.header.bodies.forEach((body, b) => {
    if (!names(body.name)) return;
    const local = [...body.geoms.filter((g) => !isParkedGeom(g)).flatMap(geomPoints), ...extra(b)];
    if (local.length === 0) return;
    replay.sample(b, f, pos, quat);
    for (const p of local) {
      rotate(quat, p, out);
      const z = out[2]! + pos[2]!;
      lo = Math.min(lo, z);
      hi = Math.max(hi, z);
    }
  });
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/** Default section height: the middle of the band where the jaws overlap the part (else mid-part), as a fraction of the part. */
function gripBandFraction(part: [number, number], jaws: [number, number] | null): number {
  const span = part[1] - part[0];
  if (!jaws || span <= 0) return 0.5;
  const lo = Math.max(part[0], jaws[0]);
  const hi = Math.min(part[1], jaws[1]);
  if (hi <= lo) return 0.5;
  return Math.min(Math.max(((lo + hi) / 2 - part[0]) / span, 0.05), 0.95);
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <span role="group" aria-label={label} className="inline-flex flex-wrap border border-rule">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          aria-pressed={o.id === value}
          onClick={() => onChange(o.id)}
          className={`px-2 py-1 text-xs ${o.id === value ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </span>
  );
}

/**
 * The design page's 3D views of a finger pair seated on a part, from a recorded
 * replay: studio, x-ray (part at 35% opacity), cutaway (a horizontal section with a
 * movable height) and a turntable. With several entries it switches between the
 * parts the same recipe was compiled for. Reads replay versions 1 and 2.
 */
export default function DesignStudio({ entries, phases, title }: { entries: DesignStudioEntry[]; phases: PhaseSpan[]; title: string }) {
  const colors = useThemeColors();
  const [index, setIndex] = useState(0);
  const [mode, setMode] = useState<Mode>("studio");
  const [turntable, setTurntable] = useState(false);
  const [state, setState] = useState<{ status: "loading" } | { status: "error"; message: string } | ({ status: "ready" } & Loaded)>({
    status: "loading",
  });
  const close = phases.find((p) => p.name === "close");
  const lift = phases.find((p) => p.name === "lift");
  const [closure, setClosure] = useState(1);
  const [cutFrac, setCutFrac] = useState(0.5);
  const sliderId = useId();
  const cutId = useId();
  const entry = entries[Math.min(index, entries.length - 1)];
  const store = useMemo(() => createPlaybackStore(), []);

  useEffect(() => setTurntable(!prefersReducedMotion()), []);

  useEffect(() => {
    if (!entry) return;
    const controller = new AbortController();
    let cancelled = false;
    setState({ status: "loading" });
    loadLane(dataUrl(entry.stem), controller.signal)
      .then(({ replay, assets }) => {
        if (cancelled) return;
        store.getState().setDuration(replay.duration);
        setState({ status: "ready", replay, assets, clay: isV2(replay.header) ? clayAssets(replay.header.assets) : new Set() });
      })
      .catch((err) => {
        if (!cancelled) setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [entry, store]);

  // Seat time: end of the close phase (registration is measured there), one frame before the lift.
  const ready = state.status === "ready" ? state : null;
  const fps = ready?.replay.header.fps ?? 30;
  const tOpen = close?.start_s ?? 0;
  const tSeat = Math.max(tOpen, (lift?.start_s ?? close?.end_s ?? 0) - 1 / fps);
  const t = tOpen + (tSeat - tOpen) * closure;

  useEffect(() => {
    if (ready) store.getState().seek(Math.min(t, ready.replay.duration));
  }, [ready, store, t]);

  const seatFrame = ready ? Math.round(ready.replay.frameAt(tSeat)) : 0;
  const bounds = useMemo<Bounds | null>(() => {
    if (!ready) return null;
    const f0 = Math.round(ready.replay.frameAt(tOpen));
    const b = ready.replay.bounds(assetPoints(ready.replay, ready.assets), [f0, seatFrame]);
    return {
      min: [b.min[0] - MARGIN_M, b.min[1] - MARGIN_M, Math.min(b.min[2], 0) - MARGIN_M],
      max: [b.max[0] + MARGIN_M, b.max[1] + MARGIN_M, b.max[2] + MARGIN_M],
    };
  }, [ready, tOpen, seatFrame]);
  const partZ = useMemo(
    () => (ready ? heightRange(ready.replay, ready.assets, seatFrame, (n) => n === "object") : null),
    [ready, seatFrame],
  );
  const jawZ = useMemo(
    () => (ready ? heightRange(ready.replay, ready.assets, seatFrame, (n) => n.startsWith("finger")) : null),
    [ready, seatFrame],
  );
  // Each new part starts with the section through the grip band.
  useEffect(() => {
    if (partZ) setCutFrac(gripBandFraction(partZ, jawZ));
  }, [partZ, jawZ]);
  const cut = mode === "cutaway" && partZ ? partZ[0] + (partZ[1] - partZ[0]) * cutFrac : null;
  const options = useMemo<StudioOptions>(
    () => ({ xray: mode === "xray", collision: false, clay: ready?.clay ?? new Set(), cut }),
    [mode, ready?.clay, cut],
  );
  const orbit = useMemo(() => ({ autoRotate: turntable }), [turntable]);

  if (!entry) return null;
  return (
    <figure className="flex flex-col" data-design-studio>
      <div className="crop-marks relative aspect-[4/3] bg-paper-2/60" data-stage>
        {ready && bounds ? (
          <StudioReplayCanvas
            replay={ready.replay}
            assets={ready.assets}
            store={store}
            bounds={bounds}
            view={mode === "cutaway" ? FROM_ABOVE : ANGLED}
            colors={colors}
            options={options}
            orbit={orbit}
          />
        ) : null}
        {state.status === "loading" ? <p className="absolute inset-0 grid place-items-center text-sm text-ink-3">Loading replay</p> : null}
        {state.status === "error" ? (
          <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">
            This replay could not be loaded: {state.message}
          </p>
        ) : null}
        {ready ? (
          <p className="num pointer-events-none absolute inset-x-2 bottom-2 truncate text-[0.625rem] text-ink-3">
            {replayCaption(ready.replay.header)}
          </p>
        ) : null}
        <p className="sr-only">
          {title} on {entry.label}, {MODES.find((m) => m.id === mode)?.help}
        </p>
      </div>
      <figcaption className="mt-2 grid gap-2 text-sm text-ink-3">
        <div className="flex flex-wrap items-center gap-2">
          <Segmented label="View" options={MODES} value={mode} onChange={setMode} />
          <button
            type="button"
            aria-pressed={turntable}
            onClick={() => setTurntable((v) => !v)}
            className={`border border-rule px-2 py-1 text-xs ${turntable ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"}`}
          >
            Turntable
          </button>
          <span className="text-xs">Drag to turn.</span>
        </div>
        <p>{MODES.find((m) => m.id === mode)?.help}</p>
        <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
          <label htmlFor={sliderId} className="flex flex-col gap-1 text-xs">
            <span>
              Jaws: <span className="text-ink">{closure >= 0.999 ? "seated (end of closing)" : closure <= 0.001 ? "open" : "closing"}</span>
            </span>
            <input
              id={sliderId}
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={closure}
              onChange={(e) => setClosure(Number(e.currentTarget.value))}
              className="w-full accent-[var(--accent)]"
            />
          </label>
          {mode === "cutaway" && partZ ? (
            <label htmlFor={cutId} className="flex flex-col gap-1 text-xs">
              <span>
                Section height: <span className="num text-ink">{(((cut ?? 0) - partZ[0]) * 1000).toFixed(1)} mm</span> above the
                part&apos;s base
              </span>
              <input
                id={cutId}
                type="range"
                min={0.05}
                max={0.95}
                step={0.01}
                value={cutFrac}
                onChange={(e) => setCutFrac(Number(e.currentTarget.value))}
                className="w-full accent-[var(--accent)]"
              />
            </label>
          ) : null}
        </div>
        {entries.length > 1 ? (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs">Same recipe, {entries.length} parts:</span>
            <Segmented
              label="Part"
              options={entries.map((e, i) => ({ id: String(i), label: e.label }))}
              value={String(index)}
              onChange={(v) => setIndex(Number(v))}
            />
          </div>
        ) : null}
        {ready && isV2(ready.replay.header) ? <ReplayCredits assets={ready.replay.header.assets} /> : null}
      </figcaption>
    </figure>
  );
}
