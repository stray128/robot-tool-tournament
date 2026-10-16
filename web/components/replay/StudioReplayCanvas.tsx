"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { instantiate, type LoadedAsset } from "@/lib/assets";
import { isParkedGeom, isV2, type Bounds, type Replay } from "@/lib/replay";
import type { ReplayGeom, ReplayVisual } from "@/lib/types";

import type { PlaybackStore, ViewName } from "./playback";
import { PoseDriver } from "./PoseDriver";
import { Datum, useMeshGeometry, VIEW_DIRECTIONS, Z_UP_TO_Y_UP } from "./ReplayCanvas";
import {
  fitPerspective,
  geomMaterial,
  placeFog,
  STUDIO,
  StudioFloor,
  StudioLights,
  StudioSetup,
  studioPalette,
  type StudioPalette,
} from "./studio";
import type { SceneColors } from "./useThemeColors";

/** What the studio view draws on top of the plain scene. */
export interface StudioOptions {
  /** Draw the part (object bodies and object assets) at 35% opacity. */
  xray: boolean;
  /** Outline the replay's v1-style collision geoms. */
  collision: boolean;
  /** Asset ids drawn untextured (brand screen). */
  clay: ReadonlySet<string>;
  /** Cutaway: clip everything above this height (MuJoCo z, metres); null for no cut. */
  cut: number | null;
}

export const NO_STUDIO_OPTIONS: StudioOptions = { xray: false, collision: false, clay: new Set(), cut: null };

type MeshMaterial = THREE.Material & { color?: THREE.Color; map?: THREE.Texture | null };

function meshesOf(root: THREE.Object3D): THREE.Mesh[] {
  const out: THREE.Mesh[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && !o.userData.sectionCap) out.push(o as THREE.Mesh);
  });
  return out;
}

function materialsOf(mesh: THREE.Mesh): MeshMaterial[] {
  return (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) as MeshMaterial[];
}

/**
 * Section caps for the cutaway: a back-face copy of every mesh in a flat colour. Where
 * the clipping plane opens a solid, its inner back faces show through as a cut face.
 */
function addSectionCaps(root: THREE.Object3D, color: string): THREE.Material {
  // Pulled slightly towards the camera so a cap never z-fights with a coplanar surface.
  const cap = new THREE.MeshBasicMaterial({
    color,
    side: THREE.BackSide,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  for (const mesh of meshesOf(root)) {
    const c = new THREE.Mesh(mesh.geometry, cap);
    c.userData.sectionCap = true;
    c.visible = false;
    c.renderOrder = 1;
    mesh.add(c);
  }
  return cap;
}

/** Applies x-ray, clay and cutaway state to one drawable instance. */
function useStyle(root: THREE.Object3D | null, opts: { xray: boolean; clay: boolean; cut: boolean; palette: StudioPalette }) {
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    if (!root) return;
    for (const mesh of meshesOf(root)) {
      mesh.receiveShadow = true;
      for (const m of materialsOf(mesh)) {
        if (m.userData.original === undefined) m.userData.original = { color: m.color?.clone(), map: m.map ?? null };
        const orig = m.userData.original as { color?: THREE.Color; map: THREE.Texture | null };
        if (opts.clay) {
          m.color?.set(opts.palette.clay);
          if ("map" in m) m.map = null;
        } else {
          if (orig.color) m.color?.copy(orig.color);
          if ("map" in m) m.map = orig.map;
        }
        m.transparent = opts.xray;
        m.opacity = opts.xray ? STUDIO.xrayOpacity : 1;
        m.depthWrite = !opts.xray;
        m.needsUpdate = true;
      }
      mesh.castShadow = !opts.xray && !opts.cut;
      for (const child of mesh.children) if (child.userData.sectionCap) child.visible = opts.cut && !opts.xray;
    }
    invalidate();
  }, [root, opts.xray, opts.clay, opts.cut, opts.palette, invalidate]);
}

function Visual({
  visual,
  loaded,
  part,
  xray,
  clay,
  cut,
  palette,
}: {
  visual: ReplayVisual;
  loaded: LoadedAsset;
  /** The part being grasped (its cut faces are drawn in the part's section colour). */
  part: boolean;
  xray: boolean;
  clay: boolean;
  cut: boolean;
  palette: StudioPalette;
}) {
  const capColor = part ? palette.sectionPart : palette.section;
  const { instance, cap } = useMemo(() => {
    const inst = instantiate(loaded.scene);
    return { instance: inst, cap: addSectionCaps(inst, capColor) };
  }, [loaded, capColor]);
  useEffect(
    () => () => {
      cap.dispose();
      for (const mesh of meshesOf(instance)) materialsOf(mesh).forEach((m) => m.dispose());
    },
    [instance, cap],
  );
  useStyle(instance, { xray, clay, cut, palette });
  return <primitive object={instance} position={visual.pos} quaternion={visual.quat_xyzw} scale={visual.scale} />;
}

/** A v1 geom in the studio look: physically based stand-in material, shadows, optional x-ray. */
function StudioGeom({
  geom,
  bodyName,
  palette,
  colors,
  xray,
  cut,
}: {
  geom: ReplayGeom;
  bodyName: string;
  palette: StudioPalette;
  colors: SceneColors;
  xray: boolean;
  cut: boolean;
}) {
  const meshGeometry = useMeshGeometry(geom);
  const primitive = useMemo(() => {
    switch (geom.type) {
      case "box": {
        const [hx = 0, hy = 0, hz = 0] = geom.half_size ?? [];
        return new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
      }
      case "cylinder":
        return new THREE.CylinderGeometry(geom.radius, geom.radius, 2 * (geom.half_height ?? 0), 48).rotateX(Math.PI / 2);
      case "capsule":
        return new THREE.CapsuleGeometry(geom.radius, 2 * (geom.half_length ?? 0), 8, 32).rotateX(Math.PI / 2);
      default:
        return null;
    }
  }, [geom]);
  useEffect(() => () => primitive?.dispose(), [primitive]);
  const g = geom.type === "mesh" ? meshGeometry : primitive;
  const root = useMemo(() => {
    if (!g) return null;
    const group = new THREE.Group();
    group.position.set(...geom.pos);
    group.quaternion.set(...geom.quat_xyzw);
    group.add(new THREE.Mesh(g, geomMaterial(bodyName, palette, colors)));
    addSectionCaps(group, bodyName === "object" ? palette.sectionPart : palette.section);
    return group;
  }, [g, geom, bodyName, palette, colors]);
  useEffect(
    () => () =>
      root?.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) materialsOf(m).forEach((x) => x.dispose());
      }),
    [root],
  );
  useStyle(root, { xray, clay: false, cut, palette });
  return root ? <primitive object={root} /> : null;
}

/** Outline of a collision geom, drawn over everything (the simulator's view of the body). */
function CollisionOutline({ geom, color }: { geom: ReplayGeom; color: string }) {
  const meshGeometry = useMeshGeometry(geom);
  const lines = useMemo(() => {
    let g: THREE.BufferGeometry | null = null;
    switch (geom.type) {
      case "box": {
        const [hx = 0, hy = 0, hz = 0] = geom.half_size ?? [];
        g = new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
        break;
      }
      case "cylinder":
        g = new THREE.CylinderGeometry(geom.radius, geom.radius, 2 * (geom.half_height ?? 0), 24).rotateX(Math.PI / 2);
        break;
      case "capsule":
        g = new THREE.CapsuleGeometry(geom.radius, 2 * (geom.half_length ?? 0), 4, 16).rotateX(Math.PI / 2);
        break;
      case "mesh":
        g = meshGeometry;
        break;
      default:
        g = null;
    }
    if (!g) return null;
    const edges = new THREE.EdgesGeometry(g, 20);
    if (g !== meshGeometry) g.dispose();
    return edges;
  }, [geom, meshGeometry]);
  useEffect(() => () => lines?.dispose(), [lines]);
  if (!lines) return null;
  return (
    <group position={geom.pos} quaternion={geom.quat_xyzw}>
      <lineSegments geometry={lines} renderOrder={10}>
        <lineBasicMaterial color={color} depthTest={false} transparent opacity={0.9} />
      </lineSegments>
    </group>
  );
}

/** Fixed camera fitted to the bounds, or orbit controls (design page) with an optional turntable. */
function StudioCamera({
  bounds,
  direction,
  pad,
  orbit,
}: {
  bounds: Bounds;
  direction: THREE.Vector3;
  pad: { top: number; bottom: number };
  orbit: { autoRotate: boolean } | null;
}) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const size = useThree((s) => s.size);
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);
  const controls = useRef<React.ComponentRef<typeof OrbitControls>>(null);
  const target = useRef(new THREE.Vector3());

  useEffect(() => {
    target.current.copy(fitPerspective(camera, bounds, direction, size, pad));
    placeFog(scene, camera.position.distanceTo(target.current));
    if (controls.current) {
      controls.current.target.copy(target.current);
      controls.current.update();
    }
    invalidate();
  }, [camera, bounds, direction, size, pad, scene, invalidate]);

  if (!orbit) return null;
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping={false}
      enablePan={false}
      autoRotate={orbit.autoRotate}
      autoRotateSpeed={1.2}
      target={target.current}
      minDistance={0.05}
      maxDistance={3}
    />
  );
}

/** Applies the cutaway as a renderer-wide clipping plane (MuJoCo z = three.js y). */
function Cutaway({ cut }: { cut: number | null }) {
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    gl.clippingPlanes = cut === null ? [] : [new THREE.Plane(new THREE.Vector3(0, -1, 0), cut)];
    invalidate();
    return () => {
      gl.clippingPlanes = [];
    };
  }, [gl, cut, invalidate]);
  return null;
}

export interface StudioReplayCanvasProps {
  replay: Replay;
  /** Loaded GLB assets for a version 2 replay; null draws the v1 geoms in the studio look. */
  assets: Map<string, LoadedAsset> | null;
  store: PlaybackStore;
  bounds: Bounds;
  /** A named view, or a direction (three.js coordinates, target to camera). */
  view: ViewName | THREE.Vector3;
  colors: SceneColors;
  datum?: { center: [number, number]; z: number } | null;
  options?: StudioOptions;
  /** Orbit controls (design page); omit for a fixed camera (bout lanes, hero). */
  orbit?: { autoRotate: boolean } | null;
  /** Screen space kept free above and below for HTML readouts, CSS px. */
  pad?: { top: number; bottom: number };
}

const NO_PAD = { top: 0, bottom: 0 };

/** One replay in the studio look, driven by the shared playback clock. */
export default function StudioReplayCanvas({
  replay,
  assets,
  store,
  bounds,
  view,
  colors,
  datum = null,
  options = NO_STUDIO_OPTIONS,
  orbit = null,
  pad = NO_PAD,
}: StudioReplayCanvasProps) {
  const bodies = useRef<(THREE.Group | null)[]>([]);
  const { header } = replay;
  const palette = useMemo(() => studioPalette(colors), [colors]);
  const direction = useMemo(() => (typeof view === "string" ? VIEW_DIRECTIONS[view] : view), [view]);
  const v2 = isV2(header) ? header : null;
  const cut = options.cut !== null;

  return (
    <Canvas
      frameloop="demand"
      shadows="percentage"
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      camera={{ fov: STUDIO.fov, position: [0.4, 0.3, 0.3], near: 0.005, far: 20 }}
      aria-hidden="true"
    >
      <StudioSetup palette={palette} />
      <StudioLights bounds={bounds} />
      <StudioFloor bounds={bounds} palette={palette} />
      <StudioCamera bounds={bounds} direction={direction} pad={pad} orbit={orbit} />
      <Cutaway cut={options.cut} />
      <group rotation={Z_UP_TO_Y_UP}>
        {/* The studio floor stands in for the table and the floor plane; other static geoms are drawn. */}
        {header.static
          .filter((g) => g.type !== "plane" && g.name !== "table")
          .map((g) => (
            <StudioGeom key={`static-${g.name}`} geom={g} bodyName="static" palette={palette} colors={colors} xray={false} cut={cut} />
          ))}
        {header.bodies.map((body, i) => {
          const isObject = body.name === "object";
          const visuals = v2 ? v2.bodies[i]!.visuals : [];
          return (
            <group
              key={body.name}
              ref={(el) => {
                bodies.current[i] = el;
              }}
            >
              {visuals.map((v, j) => {
                const loaded = assets?.get(v.asset);
                if (!loaded) return null;
                const objectRole = loaded.asset.role === "object" || isObject;
                return (
                  <Visual
                    key={`${v.asset}-${j}`}
                    visual={v}
                    loaded={loaded}
                    part={objectRole}
                    xray={options.xray && objectRole}
                    clay={options.clay.has(v.asset)}
                    cut={cut}
                    palette={palette}
                  />
                );
              })}
              {visuals.length === 0
                ? body.geoms
                    .filter((g) => !isParkedGeom(g))
                    .map((g) => (
                      <StudioGeom
                        key={g.name}
                        geom={g}
                        bodyName={body.name}
                        palette={palette}
                        colors={colors}
                        xray={options.xray && isObject}
                        cut={cut}
                      />
                    ))
                : null}
              {options.collision
                ? body.geoms
                    .filter((g) => !isParkedGeom(g))
                    .map((g) => <CollisionOutline key={`c-${g.name}`} geom={g} color={palette.overlay} />)
                : null}
            </group>
          );
        })}
        {v2?.static_visuals.map((v, j) => {
          const loaded = assets?.get(v.asset);
          return loaded ? (
            <Visual
              key={`static-${v.asset}-${j}`}
              visual={v}
              loaded={loaded}
              part={false}
              xray={false}
              clay={options.clay.has(v.asset)}
              cut={cut}
              palette={palette}
            />
          ) : null;
        })}
        {datum && !cut ? <Datum center={datum.center} z={datum.z} color={colors.accent} /> : null}
      </group>
      <PoseDriver store={store} replay={replay} bodies={bodies} />
    </Canvas>
  );
}
