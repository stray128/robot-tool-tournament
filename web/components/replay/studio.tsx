"use client";

/**
 * The studio look shared by v2 replays, the design page and the hero: neutral tone
 * mapping, an image-based environment generated on the GPU from three.js's
 * RoomEnvironment (no HDRI download), a shadowed key light with fill and rim, and a
 * fogged floor with a 10 mm grid mat. Settings follow the render track's prototype
 * viewer. Palette colours are written in sRGB and converted to linear by
 * THREE.Color, so lighting maths happens in linear space.
 *
 * Everything here works in three.js world coordinates (Y up). Replay data stays in
 * MuJoCo's Z-up frame under the single root rotation (see ReplayCanvas).
 */

import { useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

import type { Bounds } from "@/lib/replay";

import type { SceneColors } from "./useThemeColors";

export const STUDIO = {
  exposure: 1.0,
  envSigma: 0.04,
  envIntensity: 0.35,
  key: { color: "#fff4e8", intensity: 2.4, dir: [0.45, 1.0, 0.55], mapSize: 2048, bias: -0.0002, normalBias: 0.002, radius: 4 },
  fill: { sky: "#f2f5ff", ground: "#9a968f", intensity: 0.6 },
  rim: { color: "#ffffff", intensity: 1.4, dir: [-0.6, 0.5, -0.6] },
  fov: 34,
  /** Fog start and end, metres from the camera (the floor fades out behind the scene). */
  fog: [1.2, 3.5] as const,
  /** Grid mat size (m) and cell (m). */
  mat: { width: 0.68, depth: 0.52, cell: 0.01 },
  /** Opacity of the part in x-ray views. */
  xrayOpacity: 0.35,
} as const;

export interface StudioPalette {
  dark: boolean;
  background: string;
  floor: string;
  mat: string;
  matLine: string;
  clay: string;
  /** Cut faces of jaws and gripper in the cutaway. */
  section: string;
  /** Cut faces of the part in the cutaway. */
  sectionPart: string;
  overlay: string;
}

function luminance(hex: string): number {
  const c = new THREE.Color(hex);
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/** Studio colours for the page's current theme (a dark studio under the dark theme). */
export function studioPalette(colors: SceneColors): StudioPalette {
  const dark = luminance(colors.paper2) < 0.2;
  return dark
    ? {
        dark,
        background: "#1a1e24",
        floor: "#262a31",
        mat: "#2c3139",
        matLine: "#3d434d",
        clay: "#a9a49b",
        section: "#e0533d",
        sectionPart: "#363d48",
        overlay: colors.accent,
      }
    : {
        dark,
        background: "#e6e6e8",
        floor: "#bdbdc0",
        mat: "#d4d7da",
        matLine: "#b9bdc2",
        clay: "#c9c4bb",
        section: "#c8402c",
        sectionPart: "#4a4f57",
        overlay: colors.accent,
      };
}

/** Renderer, environment, background and fog. Mount once inside a <Canvas>. */
export function StudioSetup({ palette }: { palette: StudioPalette }) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    gl.toneMapping = THREE.NeutralToneMapping;
    gl.toneMappingExposure = STUDIO.exposure;
    gl.outputColorSpace = THREE.SRGBColorSpace;
    gl.shadowMap.enabled = true;
    gl.shadowMap.type = THREE.PCFShadowMap;
    const pmrem = new THREE.PMREMGenerator(gl);
    const room = new RoomEnvironment();
    const env = pmrem.fromScene(room, STUDIO.envSigma).texture;
    scene.environment = env;
    scene.environmentIntensity = STUDIO.envIntensity;
    invalidate();
    return () => {
      scene.environment = null;
      env.dispose();
      room.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.geometry.dispose();
          (Array.isArray(m.material) ? m.material : [m.material]).forEach((x) => x.dispose());
        }
      });
      pmrem.dispose();
    };
  }, [gl, scene, invalidate]);

  useEffect(() => {
    scene.background = new THREE.Color(palette.background);
    scene.fog = new THREE.Fog(palette.background, STUDIO.fog[0], STUDIO.fog[1]);
    invalidate();
    return () => {
      scene.background = null;
      scene.fog = null;
    };
  }, [scene, palette.background, invalidate]);

  return null;
}

/** Centre and half size of bounds in three.js coordinates ((x, y, z) -> (x, z, -y)). */
export function boundsFrame(bounds: Bounds): { center: THREE.Vector3; radius: number } {
  const c = new THREE.Vector3(
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
    -(bounds.min[1] + bounds.max[1]) / 2,
  );
  const radius =
    0.5 * Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
  return { center: c, radius: Math.max(radius, 0.01) };
}

/** Key light with a soft shadow sized to the scene, plus hemisphere fill and a rim light. */
export function StudioLights({ bounds }: { bounds: Bounds }) {
  const { center, radius } = useMemo(() => boundsFrame(bounds), [bounds]);
  const key = useMemo(() => {
    const light = new THREE.DirectionalLight(STUDIO.key.color, STUDIO.key.intensity);
    light.castShadow = true;
    light.shadow.mapSize.set(STUDIO.key.mapSize, STUDIO.key.mapSize);
    light.shadow.bias = STUDIO.key.bias;
    light.shadow.normalBias = STUDIO.key.normalBias;
    light.shadow.radius = STUDIO.key.radius;
    return light;
  }, []);

  useEffect(() => {
    const dir = new THREE.Vector3(...STUDIO.key.dir).normalize();
    const reach = radius * 1.6 + 0.05;
    key.position.copy(center).addScaledVector(dir, 1.5);
    key.target.position.copy(center);
    key.target.updateMatrixWorld();
    Object.assign(key.shadow.camera, { left: -reach, right: reach, top: reach, bottom: -reach, near: 0.05, far: 3 });
    key.shadow.camera.updateProjectionMatrix();
    key.shadow.needsUpdate = true;
  }, [key, center, radius]);

  useEffect(() => () => key.dispose(), [key]);

  const rimPos = useMemo(() => center.clone().addScaledVector(new THREE.Vector3(...STUDIO.rim.dir).normalize(), 1.5), [center]);

  return (
    <>
      <primitive object={key} />
      <primitive object={key.target} />
      <hemisphereLight args={[STUDIO.fill.sky, STUDIO.fill.ground, STUDIO.fill.intensity]} />
      <directionalLight position={rimPos} color={STUDIO.rim.color} intensity={STUDIO.rim.intensity} />
    </>
  );
}

function gridTexture(palette: StudioPalette): THREE.CanvasTexture | null {
  if (typeof document === "undefined") return null;
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  if (!g) return null;
  g.fillStyle = palette.mat;
  g.fillRect(0, 0, 256, 256);
  g.strokeStyle = palette.matLine;
  g.lineWidth = 2;
  g.strokeRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(STUDIO.mat.width / STUDIO.mat.cell, STUDIO.mat.depth / STUDIO.mat.cell);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Fogged floor and a 10 mm grid mat at table height (MuJoCo z = 0), centred under the scene. */
export function StudioFloor({ bounds, palette }: { bounds: Bounds; palette: StudioPalette }) {
  const texture = useMemo(() => gridTexture(palette), [palette]);
  useEffect(() => () => texture?.dispose(), [texture]);
  const { center } = useMemo(() => boundsFrame(bounds), [bounds]);
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center.x, -0.0008, center.z]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color={palette.floor} roughness={0.9} />
      </mesh>
      {/* just below the table top, so parts resting on it never z-fight with the mat */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center.x, -0.0003, center.z]} receiveShadow>
        <planeGeometry args={[STUDIO.mat.width, STUDIO.mat.depth]} />
        <meshStandardMaterial map={texture} color={texture ? "#ffffff" : palette.mat} roughness={0.85} />
      </mesh>
    </>
  );
}

/**
 * Places a perspective camera so `bounds` (MuJoCo coordinates) fills the view from
 * direction `dir` (three.js coordinates, target to camera), leaving `pad` (CSS px)
 * free at the top and bottom for HTML readouts.
 */
export function fitPerspective(
  camera: THREE.PerspectiveCamera,
  bounds: Bounds,
  dir: THREE.Vector3,
  size: { width: number; height: number },
  pad: { top: number; bottom: number } = { top: 0, bottom: 0 },
): THREE.Vector3 {
  const { center, radius } = boundsFrame(bounds);
  const usable = Math.max(size.height - pad.top - pad.bottom, 1);
  const vfov = THREE.MathUtils.degToRad(camera.fov) * (usable / Math.max(size.height, 1));
  const aspect = size.width / Math.max(usable, 1);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
  const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.02;
  const up = new THREE.Vector3(0, 1, 0);
  const forward = dir.clone().normalize().negate();
  const right = new THREE.Vector3().crossVectors(forward, up).normalize();
  const camUp = new THREE.Vector3().crossVectors(right, forward).normalize();
  // shift the target so the content centres in the band between the readouts
  const worldPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / Math.max(size.height, 1);
  const target = center.clone().addScaledVector(camUp, -((pad.bottom - pad.top) / 2) * worldPerPx);
  camera.position.copy(target).addScaledVector(dir.clone().normalize(), dist);
  camera.up.copy(up);
  camera.lookAt(target);
  camera.near = Math.max(dist / 100, 0.002);
  camera.far = dist * 20 + 5;
  camera.aspect = size.width / Math.max(size.height, 1);
  camera.updateProjectionMatrix();
  return target;
}

/** Keeps the fog behind the scene: it starts past the subject and hides the floor's far edge. */
export function placeFog(scene: THREE.Scene, distance: number): void {
  if (scene.fog instanceof THREE.Fog) {
    scene.fog.near = Math.max(STUDIO.fog[0], distance * 2.2);
    scene.fog.far = Math.max(STUDIO.fog[1], distance * 6.5);
  }
}

/** Physically based stand-in materials for v1 geoms drawn in the studio look. */
export function geomMaterial(bodyName: string, palette: StudioPalette, colors: SceneColors): THREE.MeshStandardMaterial {
  if (bodyName.startsWith("finger")) return new THREE.MeshStandardMaterial({ color: "#e8742a", roughness: 0.55, metalness: 0 });
  if (bodyName === "palm") return new THREE.MeshStandardMaterial({ color: "#3a3f47", roughness: 0.45, metalness: 0.3 });
  if (bodyName === "object") return new THREE.MeshStandardMaterial({ color: "#c3c7cc", roughness: 0.4, metalness: 0.25 });
  return new THREE.MeshStandardMaterial({ color: palette.dark ? colors.ink3 : colors.stock, roughness: 0.7 });
}
