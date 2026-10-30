"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import { createStore, type StoreApi } from "zustand/vanilla";

export type ViewName = "iso" | "front";

/**
 * How lanes are drawn. "drawing": the v1 engineering-drawing look (flat colours, edges,
 * orthographic). "studio": lit, shadowed and tone-mapped (always used for v2 replays,
 * which draw GLB assets; optional for v1 replays, which draw their geoms).
 */
export type LookName = "drawing" | "studio";

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
  look: LookName;
  /** Studio look: draw the part at 35% opacity so the jaws show through. */
  xray: boolean;
  /** Studio look: outline the collision geoms the simulator used (when the replay has them). */
  collision: boolean;
  setDuration: (d: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (t: number) => void;
  setSpeed: (s: number) => void;
  setView: (v: ViewName) => void;
  setLook: (l: LookName) => void;
  setXray: (on: boolean) => void;
  setCollision: (on: boolean) => void;
  /** Advance by `dt` wall seconds (scaled by speed). Returns true at the end of the recording. */
  advance: (dt: number) => boolean;
}

export type PlaybackStore = StoreApi<PlaybackState>;

export function createPlaybackStore(init: { loop?: boolean; view?: ViewName; look?: LookName } = {}): PlaybackStore {
  return createStore<PlaybackState>()((set, get) => ({
    t: 0,
    duration: 0,
    playing: false,
    speed: 1,
    loop: init.loop ?? false,
    view: init.view ?? "front",
    look: init.look ?? "drawing",
    xray: false,
    collision: false,
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
    setLook: (look) => set({ look }),
    setXray: (xray) => set({ xray }),
    setCollision: (collision) => set({ collision }),
    advance: (dt) => {
      const { t, duration, speed } = get();
      const next = Math.min(t + dt * speed, duration);
      set({ t: next });
      return next >= duration;
    },
  }));
}

/**
 * Calls `onChange` with a value derived from the clock now, then again only when that value
 * changes (by `equal`). For React state that follows the clock: setting state on every animation
 * frame keeps an urgent render queued all the time, which starves the router's navigation
 * transitions (a link or router.push can then do nothing for seconds) and re-renders everything
 * under the component, a 3D canvas included. Returns the unsubscribe function.
 */
export function watchPlayback<T>(
  store: PlaybackStore,
  select: (s: PlaybackState) => T,
  onChange: (value: T) => void,
  equal: (a: T, b: T) => boolean = Object.is,
): () => void {
  let last = select(store.getState());
  onChange(last);
  return store.subscribe((s) => {
    const next = select(s);
    if (equal(last, next)) return;
    last = next;
    onChange(next);
  });
}

export const PlaybackContext = createContext<PlaybackStore | null>(null);

export function usePlaybackStore(): PlaybackStore {
  const store = useContext(PlaybackContext);
  if (!store) throw new Error("usePlayback must be used inside a <PlaybackContext.Provider>");
  return store;
}

/**
 * A value of the shared clock in a component. It follows the store through watchPlayback, not
 * zustand's useStore: useSyncExternalStore renders every change at the highest priority, so the
 * clock's `t`, which changes on every animation frame, restarted any navigation in progress, and a
 * header link clicked while a replay played could take seconds to land. A change made by the
 * clock's frame is now a default-priority render, which React does not let interrupt a navigation;
 * a change made by a click (play, seek, view) still renders at once.
 */
export function usePlayback<T>(selector: (s: PlaybackState) => T): T {
  const store = usePlaybackStore();
  const [value, setValue] = useState(() => selector(store.getState()));
  const latest = useRef(selector);
  useEffect(() => {
    latest.current = selector;
  });
  useEffect(() => watchPlayback(store, (s) => latest.current(s), (v) => setValue(() => v)), [store]);
  return value;
}
