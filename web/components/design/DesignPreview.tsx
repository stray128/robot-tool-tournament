"use client";

import { Edges } from "@react-three/drei";
import { Canvas, useLoader, useThree } from "@react-three/fiber";
import { Component, Suspense, useEffect, useMemo, useState, type ReactNode } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { useThemeColors, type SceneColors } from "@/components/replay/useThemeColors";

type View = "angled" | "front" | "side";

/**
 * Camera directions in the GLB's own frame. The GLB is exported Y-up by
 * src/rtt/design/export.py, so unlike replays it is NOT rotated here.
 */
const DIRECTIONS: Record<View, THREE.Vector3> = {
  angled: new THREE.Vector3(1, 0.45, 0.8).normalize(),
  front: new THREE.Vector3(1, 0, 0),
  side: new THREE.Vector3(0, 0, 1),
};

function Model({ url, colors, view }: { url: string; colors: SceneColors; view: View }) {
  const gltf = useLoader(GLTFLoader, url);
  const camera = useThree((s) => s.camera) as THREE.OrthographicCamera;
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);

  const meshes = useMemo(() => {
    const out: { key: string; geometry: THREE.BufferGeometry; matrix: THREE.Matrix4; finger: boolean }[] = [];
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const name = `${mesh.name} ${mesh.parent?.name ?? ""}`;
        out.push({
          key: mesh.uuid,
          geometry: mesh.geometry,
          matrix: mesh.matrixWorld.clone(),
          finger: name.includes("finger"),
        });
      }
    });
    return out;
  }, [gltf]);

  useEffect(() => {
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const dir = DIRECTIONS[view];
    const up = new THREE.Vector3(0, 1, 0);
    const forward = dir.clone().negate();
    const right = new THREE.Vector3().crossVectors(forward, up).normalize();
    const camUp = new THREE.Vector3().crossVectors(right, forward).normalize();
    let hw = 0;
    let hh = 0;
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const d = new THREE.Vector3(x, y, z).sub(center);
          hw = Math.max(hw, Math.abs(d.dot(right)));
          hh = Math.max(hh, Math.abs(d.dot(camUp)));
        }
    camera.position.copy(center).addScaledVector(dir, 2);
    camera.up.copy(up);
    camera.lookAt(center);
    camera.zoom = Math.min(size.width / (2 * hw || 1), size.height / (2 * hh || 1)) * 0.82;
    camera.near = 0.01;
    camera.far = 10;
    camera.updateProjectionMatrix();
    invalidate();
  }, [gltf, view, camera, size.width, size.height, invalidate]);

  return (
    <>
      {meshes.map((m) => (
        <mesh key={m.key} geometry={m.geometry} matrix={m.matrix} matrixAutoUpdate={false}>
          <meshLambertMaterial color={m.finger ? colors.accent : colors.ink3} />
          <Edges color={colors.ink} threshold={25} />
        </mesh>
      ))}
    </>
  );
}

/** Shows a message instead of an empty canvas when the GLB fails to load or parse. */
class PreviewBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error.message : String(error) };
  }

  render() {
    if (this.state.error) {
      return (
        <p className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-fail">
          The 3D preview could not be loaded ({this.state.error}).
        </p>
      );
    }
    return this.props.children;
  }
}

/** 3D preview of a design's exported gripper GLB (palm plus both fingers, fully open). */
export default function DesignPreview({ url, label }: { url: string; label: string }) {
  const colors = useThemeColors();
  const [view, setView] = useState<View>("angled");
  return (
    <figure className="flex flex-col">
      <div className="crop-marks relative aspect-[4/3] bg-paper-2/60">
        <PreviewBoundary>
          <Canvas
            orthographic
            frameloop="demand"
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
            camera={{ position: [1, 1, 1], zoom: 600 }}
            aria-hidden="true"
          >
            <ambientLight intensity={2.1} />
            <directionalLight position={[2, 4, 3]} intensity={1.4} />
            <Suspense fallback={null}>
              <Model url={url} colors={colors} view={view} />
            </Suspense>
          </Canvas>
        </PreviewBoundary>
        <p className="sr-only">{label}</p>
      </div>
      <figcaption className="mt-2 flex flex-wrap items-center justify-between gap-2 text-sm text-ink-3">
        <span>Gripper fully open, as exported for printing</span>
        <span role="group" aria-label="Camera view" className="inline-flex border border-rule">
          {(Object.keys(DIRECTIONS) as View[]).map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={v === view}
              onClick={() => setView(v)}
              className={`px-2 py-1 text-xs ${v === view ? "bg-ink text-paper" : "text-ink-2 hover:text-ink"}`}
            >
              {v === "angled" ? "Angled" : v === "front" ? "Front" : "Side"}
            </button>
          ))}
        </span>
      </figcaption>
    </figure>
  );
}
