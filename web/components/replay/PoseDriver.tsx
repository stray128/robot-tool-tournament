"use client";

import { useThree } from "@react-three/fiber";
import { useEffect } from "react";
import type * as THREE from "three";

import type { Replay } from "@/lib/replay";

import type { PlaybackStore } from "./playback";

/** Applies the interpolated body poses for the shared clock and requests a frame. */
export function PoseDriver({
  store,
  replay,
  bodies,
}: {
  store: PlaybackStore;
  replay: Replay;
  bodies: React.RefObject<(THREE.Group | null)[]>;
}) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    const pos = [0, 0, 0];
    const quat = [0, 0, 0, 1];
    const apply = (t: number) => {
      const f = replay.frameAt(t);
      bodies.current.forEach((group, i) => {
        if (!group) return;
        replay.sample(i, f, pos, quat);
        group.position.set(pos[0]!, pos[1]!, pos[2]!);
        group.quaternion.set(quat[0]!, quat[1]!, quat[2]!, quat[3]!);
      });
      invalidate();
    };
    apply(store.getState().t);
    return store.subscribe((s, prev) => {
      if (s.t !== prev.t) apply(s.t);
    });
  }, [store, replay, bodies, invalidate]);
  return null;
}
