"use client";

import { useId } from "react";

import { formatClock } from "@/lib/format";
import { PHASE_LABELS } from "@/lib/labels";
import { phaseAt } from "@/lib/replay";
import type { PhaseSpan } from "@/lib/types";

import { SPEEDS, usePlayback, usePlaybackStore, type LookName, type ViewName } from "./playback";

const VIEWS: { id: ViewName; label: string }[] = [
  { id: "front", label: "Side" },
  { id: "iso", label: "Angled" },
];

const LOOKS: { id: LookName; label: string }[] = [
  { id: "drawing", label: "Drawing" },
  { id: "studio", label: "Studio" },
];

/** Which optional controls a bout offers. */
export interface TransportFeatures {
  /** The drawing/studio switch (version 1 lanes only; v2 lanes are always studio). */
  look: boolean;
  /** The collision-geometry outline (some lane carries v1-style geoms). */
  collision: boolean;
}

function Toggle({
  label,
  pressed,
  onChange,
  disabled,
}: {
  label: string;
  pressed: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      disabled={disabled}
      onClick={() => onChange(!pressed)}
      className={`num border border-rule px-2 py-1 text-xs disabled:opacity-50 ${
        pressed ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}

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
    <div role="group" aria-label={label} className="inline-flex border border-rule">
      {options.map((o) => (
        <button
          key={String(o.id)}
          type="button"
          aria-pressed={o.id === value}
          disabled={disabled}
          onClick={() => onChange(o.id)}
          className={`num px-2 py-1 text-xs disabled:opacity-50 ${
            o.id === value ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Play/pause, scrubber with phase bands, speed, view and look controls for a bout's shared clock. */
export function Transport({
  phases,
  disabled = false,
  features = { look: false, collision: false },
}: {
  phases: PhaseSpan[];
  disabled?: boolean;
  features?: TransportFeatures;
}) {
  const store = usePlaybackStore();
  const t = usePlayback((s) => s.t);
  const duration = usePlayback((s) => s.duration);
  const playing = usePlayback((s) => s.playing);
  const speed = usePlayback((s) => s.speed);
  const view = usePlayback((s) => s.view);
  const look = usePlayback((s) => s.look);
  const xray = usePlayback((s) => s.xray);
  const collision = usePlayback((s) => s.collision);
  const scrubId = useId();
  const phase = phaseAt(t, phases);
  const span = duration || 1;

  return (
    <div className="border border-rule bg-paper/80 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => store.getState().toggle()}
            disabled={disabled}
            className="inline-flex h-8 items-center gap-1.5 bg-accent px-3 text-sm text-accent-ink disabled:opacity-50"
          >
            <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
              {playing ? (
                <path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor" />
              ) : (
                <path d="M2 1l9 5-9 5z" fill="currentColor" />
              )}
            </svg>
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            onClick={() => store.getState().seek(0)}
            disabled={disabled}
            className="inline-flex h-8 items-center border border-rule px-2.5 text-sm text-ink-2 hover:text-ink disabled:opacity-50"
          >
            Restart
          </button>
        </div>
        <p className="num text-sm text-ink" aria-live="off">
          <span className="text-ink-3">t </span>
          {formatClock(t)}
          <span className="text-ink-3"> / {formatClock(duration)} s</span>
        </p>
        <p className="text-sm text-ink-2">
          Phase <span className="text-ink">{phase ? PHASE_LABELS[phase.name] : "n/a"}</span>
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
          <Segmented
            label="Playback speed"
            options={SPEEDS.map((s) => ({ id: s, label: `${s}x` }))}
            value={speed}
            onChange={(s) => store.getState().setSpeed(s)}
            disabled={disabled}
          />
          <Segmented
            label="Camera view"
            options={VIEWS}
            value={view}
            onChange={(v) => store.getState().setView(v)}
            disabled={disabled}
          />
          {features.look ? (
            <Segmented label="Look" options={LOOKS} value={look} onChange={(l) => store.getState().setLook(l)} disabled={disabled} />
          ) : null}
          {look === "studio" ? (
            <Toggle label="X-ray" pressed={xray} onChange={(v) => store.getState().setXray(v)} disabled={disabled} />
          ) : null}
          {look === "studio" && features.collision ? (
            <Toggle label="Collision" pressed={collision} onChange={(v) => store.getState().setCollision(v)} disabled={disabled} />
          ) : null}
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
        aria-valuetext={`${t.toFixed(2)} seconds${phase ? `, ${PHASE_LABELS[phase.name]} phase` : ""}`}
        onChange={(e) => {
          store.getState().pause();
          store.getState().seek(Number(e.currentTarget.value));
        }}
        className="mt-2 block w-full accent-[var(--accent)]"
      />
      <ol className="mt-1 flex h-5 w-full text-[0.6875rem] text-ink-3" aria-label="Controller phases">
        {phases.map((p) => {
          const end = p.end_s ?? span;
          const width = Math.max(0, Math.min(end, span) - Math.min(p.start_s, span)) / span;
          if (width <= 0) return null;
          const active = phase?.name === p.name;
          return (
            <li
              key={p.name}
              style={{ width: `${width * 100}%` }}
              className={`truncate border-l border-rule-strong pl-1 ${active ? "text-ink" : ""}`}
              title={`${PHASE_LABELS[p.name]}: ${p.start_s.toFixed(1)} s to ${p.end_s === null ? "end" : `${p.end_s.toFixed(1)} s`}`}
            >
              {PHASE_LABELS[p.name]}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
