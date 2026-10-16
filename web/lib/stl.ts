/**
 * Pure helpers for the "Bring your part" intake: file checks, unit guessing and the
 * size readout against the gripper's opening. No three.js here, so they run (and are
 * tested) in Node; the preview component feeds them the parsed vertex positions.
 */

/** Largest STL the browser preview accepts; the server applies its own limit. */
export const MAX_STL_BYTES = 50 * 1024 * 1024;

/** Gap left on each side of the part when the jaws are fully open, mm. */
export const SIDE_CLEARANCE_MM = 2;

export type LengthUnit = "mm" | "cm" | "m" | "in";

/** Millimetres per file unit. */
export const UNIT_SCALE: Record<LengthUnit, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

export const UNIT_LABELS: Record<LengthUnit, string> = {
  mm: "millimetres",
  cm: "centimetres",
  m: "metres",
  in: "inches",
};

export type Vec3 = [number, number, number];

export interface Extents {
  min: Vec3;
  max: Vec3;
  /** max - min per axis, in file units. */
  size: Vec3;
  /** Number of vertices read (3 per triangle for an STL). */
  vertices: number;
}

/**
 * Axis-aligned bounds of flat (x, y, z) vertex triples. Non-finite coordinates are
 * skipped; returns null when nothing finite remains.
 */
export function measurePositions(positions: ArrayLike<number>): Extents | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  let count = 0;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
    count++;
  }
  if (count === 0) return null;
  return { min, max, size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]], vertices: count };
}

/**
 * The unit an STL was most likely written in. STL carries no unit; CAD and slicers
 * write millimetres, but some exporters write metres (a 60 mm part reads 0.06) or
 * inches. A part whose largest side is under 1 is almost certainly metres; between 1
 * and 12 it could be inches or a tiny millimetre part, so millimetres stays the
 * default and the user can switch.
 */
export function guessUnit(size: Vec3): LengthUnit {
  const largest = Math.max(...size);
  if (largest > 0 && largest < 1) return "m";
  return "mm";
}

export function toMillimetres(size: Vec3, unit: LengthUnit): Vec3 {
  const k = UNIT_SCALE[unit];
  return [size[0] * k, size[1] * k, size[2] * k];
}

export type FitVerdict = "fits" | "tight" | "too_wide" | "too_small" | "empty";

export interface FitReport {
  verdict: FitVerdict;
  /** Bounding-box sides in mm, smallest first. */
  sorted_mm: Vec3;
  /** The narrowest side: the best case for a grasp across the part. */
  narrowest_mm: number;
  opening_mm: number;
  /** opening - narrowest, mm (negative when the part is too wide). */
  margin_mm: number;
  /** One sentence for the readout. */
  message: string;
}

/**
 * How the part's bounding box compares with the jaw opening. The narrowest side is
 * the best case: the agent still has to find a resting pose that presents it.
 */
export function fitAgainstOpening(sizeMm: Vec3, openingMm: number): FitReport {
  const sorted = [...sizeMm].sort((a, b) => a - b) as Vec3;
  const narrowest = sorted[0];
  const margin = openingMm - narrowest;
  const base = { sorted_mm: sorted, narrowest_mm: narrowest, opening_mm: openingMm, margin_mm: margin };
  if (!(sorted[2] > 0)) {
    return { ...base, verdict: "empty", message: "The file has no measurable volume." };
  }
  if (sorted[2] < 5) {
    return {
      ...base,
      verdict: "too_small",
      message: `The largest side is ${sorted[2].toFixed(1)} mm. Check the file's units: this is smaller than a fingertip.`,
    };
  }
  if (narrowest > openingMm) {
    return {
      ...base,
      verdict: "too_wide",
      message: `Every side is wider than the ${fmt(openingMm)} mm opening (narrowest ${fmt(narrowest)} mm). The jaws cannot close around it.`,
    };
  }
  if (margin < 2 * SIDE_CLEARANCE_MM) {
    return {
      ...base,
      verdict: "tight",
      message: `The narrowest side, ${fmt(narrowest)} mm, leaves ${fmt(margin)} mm in the ${fmt(openingMm)} mm opening. It may fit only if placed precisely.`,
    };
  }
  return {
    ...base,
    verdict: "fits",
    message: `The narrowest side, ${fmt(narrowest)} mm, fits the ${fmt(openingMm)} mm opening with ${fmt(margin)} mm to spare.`,
  };
}

function fmt(mm: number): string {
  return Math.abs(mm) >= 100 ? mm.toFixed(0) : mm.toFixed(1);
}

export interface FileCheck {
  ok: boolean;
  problem: string | null;
}

/** Cheap checks before reading a dropped file: extension and size. */
export function checkStlFile(name: string, bytes: number): FileCheck {
  if (!/\.stl$/i.test(name)) return { ok: false, problem: `${name} is not an .stl file.` };
  if (bytes <= 84) return { ok: false, problem: `${name} is too small to hold a mesh.` };
  if (bytes > MAX_STL_BYTES) {
    return { ok: false, problem: `${name} is ${(bytes / 1048576).toFixed(0)} MB; the limit is ${MAX_STL_BYTES / 1048576} MB.` };
  }
  return { ok: true, problem: null };
}

/**
 * Triangle count of a binary STL from its header, or null when the buffer is ASCII
 * or the count disagrees with the file length (then the loader's own count is used).
 */
export function binaryTriangleCount(buffer: ArrayBuffer): number | null {
  if (buffer.byteLength < 84) return null;
  const view = new DataView(buffer);
  const n = view.getUint32(80, true);
  return 84 + n * 50 === buffer.byteLength ? n : null;
}

/** True when "solid" starts within the first 5 bytes: three's STLLoader then reads the file as ASCII. */
function looksAscii(bytes: Uint8Array): boolean {
  const solid = [115, 111, 108, 105, 100];
  for (let off = 0; off < 5; off++) {
    if (solid.every((c, i) => bytes[off + i] === c)) return true;
  }
  return false;
}

/**
 * Checks a dropped file's bytes before handing them to the STL loader. A binary STL
 * whose header declares more triangles than the file holds is refused here: the
 * loader would size its arrays from that count first (4 billion triangles is about
 * 100 GB) and only then fail on the short file, which can take the tab down with it.
 * Returns a problem sentence, or null when the file can be parsed.
 */
export function checkStlBuffer(buffer: ArrayBuffer): string | null {
  if (buffer.byteLength < 84) return "The file is too small to hold a mesh.";
  const declared = new DataView(buffer).getUint32(80, true);
  const expected = 84 + declared * 50;
  if (expected === buffer.byteLength) return declared > 0 ? null : "The file holds no triangles.";
  if (looksAscii(new Uint8Array(buffer, 0, Math.min(10, buffer.byteLength)))) return null;
  if (expected > buffer.byteLength) {
    const holds = Math.floor((buffer.byteLength - 84) / 50);
    return `The binary STL header declares ${declared.toLocaleString("en-US")} triangles, but the file holds at most ${holds.toLocaleString("en-US")}; it is truncated or not an STL.`;
  }
  return declared > 0 ? null : "The file holds no triangles.";
}

/**
 * A binary STL of the given vertex triples (3 vertices per triangle) scaled by `scale`,
 * with facet normals computed from the winding. Used to send a part in millimetres when
 * the user says the file is in centimetres or inches (the API accepts mm or m only).
 */
export function encodeBinaryStl(positions: ArrayLike<number>, scale = 1, header = "rtt part, millimetres"): ArrayBuffer {
  const triangles = Math.floor(positions.length / 9);
  const buf = new ArrayBuffer(84 + triangles * 50);
  const view = new DataView(buf);
  const bytes = new TextEncoder().encode(header.slice(0, 80));
  new Uint8Array(buf, 0, bytes.length).set(bytes);
  view.setUint32(80, triangles, true);
  let o = 84;
  for (let t = 0; t < triangles; t++) {
    const p = (k: number) => positions[t * 9 + k]! * scale;
    const a = [p(0), p(1), p(2)];
    const b = [p(3), p(4), p(5)];
    const c = [p(6), p(7), p(8)];
    const u = [b[0]! - a[0]!, b[1]! - a[1]!, b[2]! - a[2]!];
    const v = [c[0]! - a[0]!, c[1]! - a[1]!, c[2]! - a[2]!];
    const n = [u[1]! * v[2]! - u[2]! * v[1]!, u[2]! * v[0]! - u[0]! * v[2]!, u[0]! * v[1]! - u[1]! * v[0]!];
    const len = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
    for (const x of [n[0]! / len, n[1]! / len, n[2]! / len, ...a, ...b, ...c]) {
      view.setFloat32(o, x, true);
      o += 4;
    }
    view.setUint16(o, 0, true);
    o += 2;
  }
  return buf;
}
