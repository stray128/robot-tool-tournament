"use client";

import { createContext, useContext } from "react";
import { createStore, type StoreApi } from "zustand/vanilla";
import { useStore } from "zustand";

export type ViewName = "iso" | "front";

export const SPEEDS = [0.25, 0.5, 1, 2] as const;

/** One shared clock for every lane of a bout, so the replays stay frame-synchronised. */
export interface PlaybackState {
  /** Playback time in seconds. */
  t: number;
  /** Longest lane duration in seconds. */
  duration: number;
  playing: boolean;
  speed: number;
  /** Restart from 0 after a short pause at the end. */
  loop: boolean;
  view: ViewName;
  setDuration: (d: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (t: number) => void;
  setSpeed: (s: number) => void;
  setView: (v: ViewName) => void;
  /** Advance by `dt` wall seconds (scaled by speed). Returns true at the end of the recording. */
  advance: (dt: number) => boolean;
}

export type PlaybackStore = StoreApi<PlaybackState>;

export function createPlaybackStore(init: { loop?: boolean; view?: ViewName } = {}): PlaybackStore {
  return createStore<PlaybackState>()((set, get) => ({
    t: 0,
    duration: 0,
    playing: false,
    speed: 1,
    loop: init.loop ?? false,
    view: init.view ?? "front",
    setDuration: (d) => set({ duration: Math.max(0, d) }),
    play: () => {
      const { t, duration } = get();
      // Pressing play at the end starts over.
      set({ playing: true, t: t >= duration ? 0 : t });
    },
    pause: () => set({ playing: false }),
    toggle: () => (get().playing ? get().pause() : get().play()),
    seek: (t) => set({ t: Math.min(Math.max(t, 0), get().duration) }),
    setSpeed: (speed) => set({ speed }),
    setView: (view) => set({ view }),
    advance: (dt) => {
      const { t, duration, speed } = get();
      const next = Math.min(t + dt * speed, duration);
      set({ t: next });
      return next >= duration;
    },
  }));
}

export const PlaybackContext = createContext<PlaybackStore | null>(null);

export function usePlaybackStore(): PlaybackStore {
  const store = useContext(PlaybackContext);
  if (!store) throw new Error("usePlayback must be used inside a <PlaybackContext.Provider>");
  return store;
}

export function usePlayback<T>(selector: (s: PlaybackState) => T): T {
  return useStore(usePlaybackStore(), selector);
}
