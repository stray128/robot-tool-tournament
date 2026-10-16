export interface MeshData {
  positions: Float32Array;
  normals?: Float32Array;
  indices?: Uint32Array | Uint16Array | number[];
}
export interface PrimitiveSpec extends MeshData {
  material?: number;
}
export interface MaterialSpec {
  name: string;
  baseColorFactor: number[];
  metallicFactor?: number;
  roughnessFactor?: number;
}
export declare function srgbToLinear(c: number): number;
export declare function hexToLinear(hex: string, alpha?: number): number[];
export declare function flatMesh(
  vertices: number[][],
  faces: number[][],
  pos?: number[],
  quat?: number[],
): Required<MeshData>;
export declare function boxMesh(half: number[], pos?: number[], quat?: number[]): Required<MeshData>;
export declare function writeGlb(spec: {
  meshes: { name: string; primitives: PrimitiveSpec[] }[];
  materials?: MaterialSpec[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
  generator?: string;
}): ArrayBuffer;
