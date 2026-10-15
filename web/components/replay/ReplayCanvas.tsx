"use client";

import { Edges, Line } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { isParkedGeom, type Bounds, type Replay } from "@/lib/replay";
import type { ReplayGeom } from "@/lib/types";

import type { PlaybackStore, ViewName } from "./playback";
import type { SceneColors } from "./useThemeColors";

/**
 * MuJoCo is Z-up, three.js is Y-up. The replay data is left in MuJoCo coordinates and
 * this single rotation of the root group converts it: (x, y, z) -> (x, z, -y).
 * Nothing else in the viewer rotates world data.
 */
const Z_UP_TO_Y_UP = new THREE.Euler(-Math.PI / 2, 0, 0);

/** Camera directions (from the target towards the camera) in three.js world coordinates. */
const VIEW_DIRECTIONS: Record<ViewName, THREE.Vector3> = {
  // Looking along the finger width, slightly from above and the side: closing axis horizontal.
  iso: new THREE.Vector3(1, 0.55, 0.62).normalize(),
  // Straight along the finger width: the jaw closes left-right and the lift is vertical.
  front: new THREE.Vector3(1, 0, 0),
};

function toThree([x, y, z]: readonly [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(x, z, -y);
}

/** Screen space kept clear for the HTML readouts over the canvas, in CSS pixels. */
const OVERLAY_PAD = { top: 52, bottom: 60, side: 10 };

/**
 * Fits an orthographic camera to `bounds` (MuJoCo coordinates) from the chosen view,
 * leaving the bands under the lane's HTML readouts free.
 */
function CameraRig({ bounds, view }: { bounds: Bounds; view: ViewName }) {
  const camera = useThree((s) => s.camera) as THREE.OrthographicCamera;
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    const dir = VIEW_DIRECTIONS[view];
    const up = new THREE.Vector3(0, 1, 0);
    const forward = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, up).normalize();
    const camUp = new THREE.Vector3().crossVectors(right, forward).normalize();
    const corners: THREE.Vector3[] = [];
    for (const x of [bounds.min[0], bounds.max[0]])
      for (const y of [bounds.min[1], bounds.max[1]])
        for (const z of [bounds.min[2], bounds.max[2]]) corners.push(toThree([x, y, z]));
    const center = toThree([
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2,
    ]);
    let halfW = 0;
    let halfH = 0;
    for (const c of corners) {
      const d = c.clone().sub(center);
      halfW = Math.max(halfW, Math.abs(d.dot(right)));
      halfH = Math.max(halfH, Math.abs(d.dot(camUp)));
    }
    const usableW = Math.max(size.width - 2 * OVERLAY_PAD.side, 1);
    const usableH = Math.max(size.height - OVERLAY_PAD.top - OVERLAY_PAD.bottom, 1);
    const zoom = Math.min(usableW / (2 * halfW || 1), usableH / (2 * halfH || 1));
    // Centre the content in the band between the readouts: shift the camera down by
    // half the difference between the bottom and top padding (in world units).
    const shift = (OVERLAY_PAD.bottom - OVERLAY_PAD.top) / 2 / zoom;
    center.addScaledVector(camUp, -shift);
    camera.position.copy(center).addScaledVector(dir, 3);
    camera.up.copy(up);
    camera.lookAt(center);
    camera.near = 0.01;
    camera.far = 10;
    camera.zoom = zoom;
    camera.updateProjectionMatrix();
    invalidate();
  }, [bounds, view, size.width, size.height, camera, invalidate]);

  return null;
}

/** Applies the interpolated body poses for the shared clock and requests a frame. */
function PoseDriver({
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

/** Flat-shaded geometry of a convex mesh piece (vertices and faces from the replay header). */
function useMeshGeometry(geom: ReplayGeom): THREE.BufferGeometry | null {
  const geometry = useMemo(() => {
    if (geom.type !== "mesh" || !geom.vertices || !geom.faces) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(geom.vertices, 3));
    g.setIndex(geom.faces);
    const flat = g.toNonIndexed();
    g.dispose();
    flat.computeVertexNormals();
    return flat;
  }, [geom]);
  useEffect(() => () => geometry?.dispose(), [geometry]);
  return geometry;
}

function GeomMesh({ geom, color, edge }: { geom: ReplayGeom; color: string; edge: string }) {
  const meshGeometry = useMeshGeometry(geom);
  const material = <meshLambertMaterial color={color} />;
  const edges = <Edges color={edge} threshold={20} />;
  let body: React.ReactNode = null;
  switch (geom.type) {
    case "box": {
      const [hx = 0, hy = 0, hz = 0] = geom.half_size ?? [];
      body = (
        <mesh>
          <boxGeometry args={[2 * hx, 2 * hy, 2 * hz]} />
          {material}
          {edges}
        </mesh>
      );
      break;
    }
    case "cylinder":
      // three.js cylinders run along local Y; MuJoCo's along local Z.
      body = (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[geom.radius, geom.radius, 2 * (geom.half_height ?? 0), 40]} />
          {material}
          {edges}
        </mesh>
      );
      break;
    case "capsule":
      body = (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <capsuleGeometry args={[geom.radius, 2 * (geom.half_length ?? 0), 8, 32]} />
          {material}
          {edges}
        </mesh>
      );
      break;
    case "mesh":
      body = meshGeometry ? (
        <mesh geometry={meshGeometry}>
          {material}
          {edges}
        </mesh>
      ) : null;
      break;
    default:
      body = null; // planes (the floor) lie far below the view
  }
  return (
    <group position={geom.pos} quaternion={geom.quat_xyzw}>
      {body}
    </group>
  );
}

function bodyColor(name: string, colors: SceneColors): { fill: string; edge: string } {
  if (name.startsWith("finger")) return { fill: colors.accent, edge: colors.ink };
  if (name === "palm") return { fill: colors.ink3, edge: colors.ink };
  return { fill: colors.stock, edge: colors.ink };
}

/** Square dashed ring at the success height around the object's rest position. */
function Datum({ center, z, color }: { center: [number, number]; z: number; color: string }) {
  const h = 0.075;
  const [cx, cy] = center;
  const points: [number, number, number][] = [
    [cx - h, cy - h, z],
    [cx + h, cy - h, z],
    [cx + h, cy + h, z],
    [cx - h, cy + h, z],
    [cx - h, cy - h, z],
  ];
  return <Line points={points} color={color} lineWidth={1.25} dashed dashSize={0.008} gapSize={0.006} />;
}

function TableGrid({ colors }: { colors: SceneColors }) {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(0.6, 60, colors.ink3, colors.ink3);
    const mat = g.material as THREE.Material;
    mat.transparent = true;
    mat.opacity = 0.28;
    g.position.y = 0.0004;
    return g;
  }, [colors.ink3]);
  useEffect(
    () => () => {
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
    },
    [grid],
  );
  return <primitive object={grid} />;
}

export interface ReplayCanvasProps {
  replay: Replay;
  store: PlaybackStore;
  bounds: Bounds;
  view: ViewName;
  colors: SceneColors;
  /** Object rest position (x, y) and the success height (z), MuJoCo coordinates; null hides the datum. */
  datum: { center: [number, number]; z: number } | null;
}

/** One lane's 3D view: the replay's bodies driven by the shared playback clock. */
export default function ReplayCanvas({ replay, store, bounds, view, colors, datum }: ReplayCanvasProps) {
  const bodies = useRef<(THREE.Group | null)[]>([]);
  const { header } = replay;

  return (
    <Canvas
      orthographic
      frameloop="demand"
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
      camera={{ position: [1, 1, 1], zoom: 400, near: 0.01, far: 10 }}
      aria-hidden="true"
    >
      <ambientLight intensity={2.1} />
      <directionalLight position={[2, 4, 3]} intensity={1.4} />
      <CameraRig bounds={bounds} view={view} />
      <TableGrid colors={colors} />
      <group rotation={Z_UP_TO_Y_UP}>
        {header.static
          .filter((g) => g.type !== "plane")
          .map((g) => (
            <GeomMesh key={`static-${g.name}`} geom={g} color={colors.paper3} edge={colors.ink2} />
          ))}
        {header.bodies.map((body, i) => {
          const { fill, edge } = bodyColor(body.name, colors);
          return (
            <group
              key={body.name}
              ref={(el) => {
                bodies.current[i] = el;
              }}
            >
              {body.geoms
                .filter((g) => !isParkedGeom(g))
                .map((g) => (
                  <GeomMesh key={g.name} geom={g} color={fill} edge={edge} />
                ))}
            </group>
          );
        })}
        {datum ? <Datum center={datum.center} z={datum.z} color={colors.accent} /> : null}
      </group>
      <PoseDriver store={store} replay={replay} bodies={bodies} />
    </Canvas>
  );
}
