/**
 * Minimal GLB (glTF 2.0 binary) writer for test fixtures: flat-shaded triangle meshes
 * with PBR materials, one node per mesh, no textures, no dependencies.
 *
 * The real assets are written by the render track (rtt.render.glb) and compressed by
 * scripts/compress_glb.mjs; this writer only exists so the tests and the v2 fixture
 * can be regenerated from the repository alone.
 */

const ARRAY_BUFFER = 34962;
const ELEMENT_ARRAY_BUFFER = 34963;
const FLOAT = 5126;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;

/** sRGB component (0..1) to linear, as glTF's baseColorFactor expects. */
export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** `#rrggbb` to a linear RGBA baseColorFactor. */
export function hexToLinear(hex, alpha = 1) {
  const n = Number.parseInt(hex.replace(/^#/, ""), 16);
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => Math.round(srgbToLinear(v / 255) * 1e6) / 1e6);
  return [...c, alpha];
}

function rotate(q, v) {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  return [vx + qw * tx + (qy * tz - qz * ty), vy + qw * ty + (qz * tx - qx * tz), vz + qw * tz + (qx * ty - qy * tx)];
}

/** Flat-shaded triangles from a vertex list and index triples, posed by pos and quat (xyzw). */
export function flatMesh(vertices, faces, pos = [0, 0, 0], quat = [0, 0, 0, 1]) {
  const positions = [];
  const normals = [];
  const world = vertices.map((v) => {
    const r = rotate(quat, v);
    return [r[0] + pos[0], r[1] + pos[1], r[2] + pos[2]];
  });
  for (const [a, b, c] of faces) {
    const pa = world[a];
    const pb = world[b];
    const pc = world[c];
    const u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    const w = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const len = Math.hypot(...n) || 1;
    for (const p of [pa, pb, pc]) {
      positions.push(...p);
      normals.push(n[0] / len, n[1] / len, n[2] / len);
    }
  }
  const count = positions.length / 3;
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: Uint32Array.from({ length: count }, (_, i) => i),
  };
}

/** A box with half extents `half`, posed by pos and quat (xyzw). */
export function boxMesh(half, pos = [0, 0, 0], quat = [0, 0, 0, 1]) {
  const [hx, hy, hz] = half;
  const v = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) v.push([sx * hx, sy * hy, sz * hz]);
  // corner index = 4*ix + 2*iy + iz; faces counter-clockwise seen from outside
  const quads = [
    [0, 1, 3, 2], // -x
    [4, 6, 7, 5], // +x
    [0, 4, 5, 1], // -y
    [2, 3, 7, 6], // +y
    [0, 2, 6, 4], // -z
    [1, 5, 7, 3], // +z
  ];
  const faces = quads.flatMap(([a, b, c, d]) => [
    [a, b, c],
    [a, c, d],
  ]);
  return flatMesh(v, faces, pos, quat);
}

function pad4(n) {
  return (4 - (n % 4)) % 4;
}

/**
 * Write a GLB.
 *   meshes: [{ name, primitives: [{ positions, normals?, indices?, material }] }]
 *   materials: [{ name, baseColorFactor, metallicFactor?, roughnessFactor? }]
 * Returns an ArrayBuffer.
 */
export function writeGlb({ meshes, materials = [], extensionsUsed, extensionsRequired, generator = "rtt test glb_writer" }) {
  const views = [];
  const accessors = [];
  const chunks = [];
  let offset = 0;
  const addView = (array, target) => {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    views.push({ buffer: 0, byteOffset: offset, byteLength: bytes.byteLength, target });
    chunks.push(bytes);
    const padding = pad4(bytes.byteLength);
    if (padding) chunks.push(new Uint8Array(padding));
    offset += bytes.byteLength + padding;
    return views.length - 1;
  };
  const gltfMeshes = meshes.map((mesh) => ({
    name: mesh.name,
    primitives: mesh.primitives.map((prim) => {
      const attributes = {};
      const p = prim.positions;
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < p.length; i += 3)
        for (let k = 0; k < 3; k++) {
          min[k] = Math.min(min[k], p[i + k]);
          max[k] = Math.max(max[k], p[i + k]);
        }
      accessors.push({ bufferView: addView(p, ARRAY_BUFFER), componentType: FLOAT, count: p.length / 3, type: "VEC3", min, max });
      attributes.POSITION = accessors.length - 1;
      if (prim.normals) {
        accessors.push({ bufferView: addView(prim.normals, ARRAY_BUFFER), componentType: FLOAT, count: p.length / 3, type: "VEC3" });
        attributes.NORMAL = accessors.length - 1;
      }
      const out = { attributes, mode: 4 };
      if (prim.indices) {
        const small = p.length / 3 <= 65535;
        const idx = small ? Uint16Array.from(prim.indices) : Uint32Array.from(prim.indices);
        accessors.push({
          bufferView: addView(idx, ELEMENT_ARRAY_BUFFER),
          componentType: small ? UNSIGNED_SHORT : UNSIGNED_INT,
          count: idx.length,
          type: "SCALAR",
        });
        out.indices = accessors.length - 1;
      }
      if (prim.material !== undefined) out.material = prim.material;
      return out;
    }),
  }));
  const json = {
    asset: { version: "2.0", generator },
    scene: 0,
    scenes: [{ nodes: meshes.map((_, i) => i) }],
    nodes: meshes.map((m, i) => ({ name: m.name, mesh: i })),
    meshes: gltfMeshes,
    materials: materials.map((m) => ({
      name: m.name,
      pbrMetallicRoughness: {
        baseColorFactor: m.baseColorFactor,
        metallicFactor: m.metallicFactor ?? 0,
        roughnessFactor: m.roughnessFactor ?? 0.6,
      },
    })),
    accessors,
    bufferViews: views,
    buffers: [{ byteLength: offset }],
  };
  if (extensionsUsed) json.extensionsUsed = extensionsUsed;
  if (extensionsRequired) json.extensionsRequired = extensionsRequired;
  if (materials.length === 0) delete json.materials;
  let jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = pad4(jsonBytes.byteLength);
  if (jsonPad) jsonBytes = new Uint8Array([...jsonBytes, ...new Array(jsonPad).fill(0x20)]);
  const total = 12 + 8 + jsonBytes.byteLength + 8 + offset;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonBytes.byteLength, true);
  view.setUint32(16, 0x4e4f534a, true);
  out.set(jsonBytes, 20);
  let o = 20 + jsonBytes.byteLength;
  view.setUint32(o, offset, true);
  view.setUint32(o + 4, 0x004e4942, true);
  o += 8;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }
  return out.buffer;
}
