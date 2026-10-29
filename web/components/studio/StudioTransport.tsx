"use client";

import { useId } from "react";

import { SPEEDS, usePlayback, usePlaybackStore, type ViewName } from "@/components/replay/playback";
import { formatClock } from "@/lib/format";

const VIEWS: { id: ViewName; label: string }[] = [
  { id: "iso", label: "Angled" },
  { id: "front", label: "Side" },
];

function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
  disabled,
}: {
  label: string;
  options: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div role="group" aria-label={label} className="inline-flex overflow-hidden rounded-md border border-rule">
      {options.map((o) => (
        <button
          key={String(o.id)}
          type="button"
          aria-pressed={o.id === value}
          disabled={disabled}
          onClick={() => onChange(o.id)}
          className={`num px-2 py-1 text-xs disabled:opacity-50 ${o.id === value ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Play, restart, the scrubber, speed and camera for the Studio's lanes, on the shared playback
 * clock (components/replay/playback.ts). The bout Transport's phase bands are grasp phases; a
 * Studio run is steps, so each lane draws its own step strip under its replay instead.
 */
export function StudioTransport({ disabled = false }: { disabled?: boolean }) {
  const store = usePlaybackStore();
  const t = usePlayback((s) => s.t);
  const duration = usePlayback((s) => s.duration);
  const playing = usePlayback((s) => s.playing);
  const speed = usePlayback((s) => s.speed);
  const view = usePlayback((s) => s.view);
  const scrubId = useId();
  const span = duration || 1;
  return (
    <div className="rounded-lg border border-rule bg-surface px-3 py-2.5" data-studio-transport>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => store.getState().toggle()}
            disabled={disabled}
            className="inline-flex h-8 items-center gap-1.5 rounded-md bg-accent px-3 text-sm font-semibold text-accent-ink disabled:opacity-50"
          >
            <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
              {playing ? <path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" /> : <path d="M2 1l9 5-9 5z" fill="currentColor" />}
            </svg>
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            onClick={() => store.getState().seek(0)}
            disabled={disabled}
            className="inline-flex h-8 items-center rounded-md border border-rule px-2.5 text-sm text-ink-2 hover:text-ink disabled:opacity-50"
          >
            Restart
          </button>
        </div>
        <p className="num text-sm text-ink" aria-live="off">
          <span className="text-ink-3">t </span>
          {formatClock(t)}
          <span className="text-ink-3"> / {formatClock(duration)} s</span>
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
          <Segmented
            label="Playback speed"
            options={SPEEDS.map((s) => ({ id: s, label: `${s}x` }))}
            value={speed}
            onChange={(s) => store.getState().setSpeed(s)}
            disabled={disabled}
          />
          <Segmented label="Camera view" options={VIEWS} value={view} onChange={(v) => store.getState().setView(v)} disabled={disabled} />
        </div>
      </div>
      <label htmlFor={scrubId} className="sr-only">
        Playback position in seconds
      </label>
      <input
        id={scrubId}
        type="range"
        min={0}
        max={span}
        step={1 / 30}
        value={Math.min(t, span)}
        disabled={disabled}
        aria-valuetext={`${t.toFixed(1)} seconds of ${duration.toFixed(1)}`}
        onChange={(e) => {
          store.getState().pause();
          store.getState().seek(Number(e.currentTarget.value));
        }}
        className="mt-2 block w-full accent-[var(--accent)]"
      />
    </div>
  );
}
