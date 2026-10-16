"use client";

import { Edges } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

import { useThemeColors, type SceneColors } from "@/components/replay/useThemeColors";

type View = "angled" | "front" | "top";

/** Camera directions in the scene's Y-up frame (the STL's Z-up is rotated once, below). */
const DIRECTIONS: Record<View, THREE.Vector3> = {
  angled: new THREE.Vector3(1, 0.7, 1.1).normalize(),
  front: new THREE.Vector3(0, 0, 1),
  top: new THREE.Vector3(0, 1, 0.0001).normalize(),
};

const VIEW_LABELS: Record<View, string> = { angled: "Angled", front: "Front", top: "Top" };

interface PreviewProps {
  /** Parsed STL geometry in file units. */
  geometry: THREE.BufferGeometry;
  /** Millimetres per file unit. */
  mmPerUnit: number;
  /** Jaw opening in mm, drawn as two plates across the part's narrowest side. */
  openingMm: number;
  label: string;
}

/**
 * The part, centred, in millimetres, with the fully open jaws drawn as two thin
 * plates `openingMm` apart across its narrowest bounding-box side. STL files are Z-up
 * by CAD convention; the root group turns them to three.js's Y-up once.
 */
function Scene({ geometry, mmPerUnit, openingMm, colors, view }: Omit<PreviewProps, "label"> & { colors: SceneColors; view: View }) {
  const camera = useThree((s) => s.camera) as THREE.OrthographicCamera;
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);

  const { centered, dims, jawAxis } = useMemo(() => {
    const g = geometry.clone();
    g.scale(mmPerUnit, mmPerUnit, mmPerUnit);
    g.computeBoundingBox();
    const box = g.boundingBox!;
    const c = box.getCenter(new THREE.Vector3());
    g.translate(-c.x, -c.y, -box.min.z);
    g.computeVertexNormals();
    const d = box.getSize(new THREE.Vector3());
    // Jaws close along the narrower horizontal side (x or y in the STL's Z-up frame).
    return { centered: g, dims: d, jawAxis: d.x <= d.y ? ("x" as const) : ("y" as const) };
  }, [geometry, mmPerUnit]);

  useEffect(() => () => centered.dispose(), [centered]);

  const plate = useMemo(() => {
    const along = jawAxis === "x" ? dims.y : dims.x;
    return { span: Math.max(along, 20) + 16, height: Math.max(dims.z, 20) + 10 };
  }, [dims, jawAxis]);

  useEffect(() => {
    const half = Math.max(dims.x, dims.y, dims.z, openingMm) / 2 + 12;
    const center = new THREE.Vector3(0, dims.z / 2, 0);
    const dir = DIRECTIONS[view];
    camera.position.copy(center).addScaledVector(dir, 1000);
    camera.up.set(0, 1, 0);
    camera.lookAt(center);
    camera.near = 1;
    camera.far = 4000;
    camera.zoom = (Math.min(size.width, size.height) / (2 * half)) * 0.92;
    camera.updateProjectionMatrix();
    invalidate();
  }, [dims, openingMm, view, camera, size.width, size.height, invalidate]);

  const plateGeom = useMemo(
    () =>
      jawAxis === "x"
        ? new THREE.BoxGeometry(1.2, plate.span, plate.height)
        : new THREE.BoxGeometry(plate.span, 1.2, plate.height),
    [jawAxis, plate],
  );
  useEffect(() => () => plateGeom.dispose(), [plateGeom]);

  const offset = openingMm / 2 + 0.6;
  const z = plate.height / 2 - 5;
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh geometry={centered}>
        <meshLambertMaterial color={colors.stock} />
        <Edges color={colors.ink2} threshold={28} />
      </mesh>
      {[-1, 1].map((sign) => (
        <mesh
          key={sign}
          geometry={plateGeom}
          position={jawAxis === "x" ? [sign * offset, 0, z] : [0, sign * offset, z]}
        >
          <meshBasicMaterial color={colors.accent} transparent opacity={0.12} depthWrite={false} />
          <Edges color={colors.accent} />
        </mesh>
      ))}
      <gridHelper args={[Math.max(dims.x, dims.y, openingMm) + 60, 12, colors.paper3, colors.paper3]} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  );
}

/** Interactive-free orthographic preview of a dropped STL with the jaw opening for scale. */
export default function StlPreview({ geometry, mmPerUnit, openingMm, label }: PreviewProps) {
  const colors = useThemeColors();
  const [view, setView] = useState<View>("angled");
  return (
    <figure className="flex flex-col">
      <div className="crop-marks relative aspect-[4/3] bg-paper-2/60">
        <Canvas
          orthographic
          frameloop="demand"
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
          camera={{ position: [300, 300, 300], zoom: 2 }}
          aria-hidden="true"
        >
          <ambientLight intensity={2.0} />
          <directionalLight position={[200, 400, 300]} intensity={1.5} />
          <Scene geometry={geometry} mmPerUnit={mmPerUnit} openingMm={openingMm} colors={colors} view={view} />
        </Canvas>
        <p className="sr-only">{label}</p>
      </div>
      <figcaption className="mt-2 flex flex-wrap items-center justify-between gap-2 text-sm text-ink-3">
        <span>
          Blue plates: the jaws fully open, <span className="num">{openingMm}</span> mm apart
        </span>
        <span role="group" aria-label="Camera view" className="inline-flex border border-rule">
          {(Object.keys(DIRECTIONS) as View[]).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={v === view}
              onClick={() => setView(v)}
              className={`px-2 py-1 text-xs ${v === view ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"}`}
            >
              {VIEW_LABELS[v]}
            </button>
          ))}
        </span>
      </figcaption>
    </figure>
  );
}
