"use client";

import { type ReactNode, useMemo } from "react";

import { prefersReducedMotion } from "@/components/replay/clock";
import { createPlaybackStore, PlaybackContext, type PlaybackStore, usePlayback, usePlaybackStore } from "@/components/replay/playback";

/**
 * One playback clock for the whole story page: the side-by-side replays, the step timeline and
 * every step card's "watch" button drive the same time, so a click anywhere seeks both replays.
 */
export function StoryPlayback({ children }: { children: ReactNode }) {
  const store = useMemo(() => createPlaybackStore({ look: "studio", view: "iso" }), []);
  return <PlaybackContext.Provider value={store}>{children}</PlaybackContext.Provider>;
}

/** Seek every replay to `t` and play from there (paused when the visitor prefers reduced motion). */
export function seekTo(store: PlaybackStore, t: number, scrollTarget?: string) {
  store.getState().seek(t + 0.001);
  if (!prefersReducedMotion()) store.getState().play();
  if (scrollTarget) {
    const el = document.getElementById(scrollTarget);
    if (el) {
      const top = el.getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.5) el.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    }
  }
}

function clock(s: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(0).padStart(2, "0")}`;
}

/** "Watch this step": seeks both replays to the step's start in the recommended run. */
export function SeekButton({ start, label }: { start: number; label: string }) {
  const store = usePlaybackStore();
  return (
    <button
      type="button"
      onClick={() => seekTo(store, start, "replays")}
      className="inline-flex items-center gap-2 rounded-md border border-rule-strong px-3 py-1.5 text-sm text-ink hover:border-accent hover:text-accent"
      data-seek={start.toFixed(3)}
    >
      <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
        <path d="M2.5 1.5l8 4.5-8 4.5z" fill="currentColor" />
      </svg>
      {label}
      <span className="num text-xs text-ink-3">{clock(start)}</span>
    </button>
  );
}

/**
 * Whether a step card is lit: while the run is on its step, and still once the clock has ended on
 * it (a run that stops at a short last step, a fraction of a second long, keeps that step lit
 * instead of going dark the moment it ends).
 */
export function stepActive(t: number, playing: boolean, start: number | null, end: number | null, duration: number): boolean {
  if (start === null || end === null || t < start || !(playing || t > 0)) return false;
  if (t < end) return true;
  const atEnd = t >= duration - 1e-3;
  return !playing && atEnd && end >= duration - 1e-3;
}

/** A step card that lights up while the recommended run is on its step. */
export function StepFrame({ id, start, end, children }: { id: string; start: number | null; end: number | null; children: ReactNode }) {
  const t = usePlayback((s) => s.t);
  const playing = usePlayback((s) => s.playing);
  const duration = usePlayback((s) => s.duration);
  const active = stepActive(t, playing, start, end, duration);
  return (
    <article
      id={`step-${id}`}
      data-step-card={id}
      data-active={active ? "" : undefined}
      className={`panel scroll-mt-24 transition-[border-color,box-shadow] duration-300 ${active ? "border-accent/70 shadow-[0_0_0_1px_var(--accent)]" : ""}`}
    >
      {children}
    </article>
  );
}
