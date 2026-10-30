"use client";

import { useEffect } from "react";

import { onLeave } from "./leaving";
import type { PlaybackStore } from "./playback";

/** Seconds the last frame stays on screen before a looping player restarts. */
export const LOOP_PAUSE_S = 1.6;

export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Drives the shared clock with requestAnimationFrame while playing and visible, and pauses it when
 * the visitor follows a link to another page of the site (components/replay/leaving.ts).
 */
export function useClock(store: PlaybackStore, visible: React.RefObject<boolean>) {
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
    // Leaving the page also cancels a loop's pending restart.
    const stopOnLeave = onLeave(() => {
      if (restart) clearTimeout(restart);
      store.getState().pause();
    });
    return () => {
      stopOnLeave();
      unsubscribe();
      cancelAnimationFrame(raf);
      if (restart) clearTimeout(restart);
    };
  }, [store, visible]);
}

