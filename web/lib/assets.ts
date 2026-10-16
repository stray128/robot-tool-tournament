/**
 * Content-addressed GLB assets (docs/ASSET_MANIFEST.md, replay header version 2).
 *
 * Every mesh a v2 replay draws is a GLB served at `assets/<id>.glb`, where `id` is the
 * first 16 hex characters of the file's sha256. The asset manifest
 * (`public/assets/manifest.json`) and each replay header's `assets` map carry the same
 * entry: uri, sha256, bytes, role, licence, credit line and the list of changes.
 *
 * This module validates those entries, reads the GLB container, checks that the
 * browser can decode it (meshopt yes, Draco no: no decoder is shipped and none is
 * fetched from a CDN), verifies the bytes against the entry and loads each asset once
 * per page, shared by every replay that draws it. Asset frames are the body frame,
 * Z up, metres: nothing here rotates or mirrors a mesh.
 */

import type { Group, Object3D } from "three";

import { BASE_PATH } from "./paths";
import type { AssetRole, AssetSource, ReplayAsset, ReplayHeaderV2 } from "./types";
import { Node } from "./validate";

// ---------------------------------------------------------------- entries

export const ASSET_ROLES: readonly AssetRole[] = [
  "finger_left",
  "finger_right",
  "gripper_frame",
  "carriage",
  "object",
  "prop",
];

/** Our own licence. Every other licence marks a third-party asset, which must list its changes. */
export const OWN_LICENSE = "Apache-2.0";

export const ASSET_ID_RE = /^[0-9a-f]{16}$/;
export const SHA256_RE = /^[0-9a-f]{64}$/;
const SPDX_RE = /^[A-Za-z0-9][A-Za-z0-9.+-]*$/;
const COMMIT_RE = /^[0-9a-f]{7,40}$/;
const ASSET_KEYS = [
  "uri",
  "sha256",
  "bytes",
  "role",
  "license",
  "attribution",
  "changes",
  "label",
  "media_type",
  "brand_visible",
  "source",
] as const;
const SOURCE_KEYS = ["kind", "url", "commit", "path", "sha256", "generator"] as const;

export class AssetError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssetError";
  }
}

/** Fails on any key of `n` outside `allowed` (JSON Schema `additionalProperties: false`). */
export function onlyKeys(n: Node, allowed: readonly string[]): void {
  for (const key of Object.keys(n.object())) {
    if (!allowed.includes(key)) n.fail(`unexpected key ${JSON.stringify(key)}`);
  }
}

/** A string that must match `re`; `what` names the expected form in the error. */
export function matching(n: Node, re: RegExp, what: string): string {
  const s = n.str();
  if (!re.test(s)) n.fail(`expected ${what}, got ${JSON.stringify(s.length > 40 ? `${s.slice(0, 40)}...` : s)}`);
  return s;
}

/** An asset id: the first 16 hex characters of the file's sha256. */
export function readAssetId(n: Node): string {
  return matching(n, ASSET_ID_RE, "an asset id (16 lowercase hex characters)");
}

function readSource(n: Node): AssetSource {
  onlyKeys(n, SOURCE_KEYS);
  const kind = n.get("kind").oneOf(["upstream", "generated"] as const);
  for (const key of ["path", "generator"] as const) if (n.has(key)) n.get(key).nonEmpty();
  if (n.has("commit")) matching(n.get("commit"), COMMIT_RE, "a git commit (7 to 40 hex characters)");
  if (kind === "upstream") {
    const out: AssetSource = {
      kind,
      url: matching(n.get("url"), /^https:\/\//, "an https URL"),
      sha256: matching(n.get("sha256"), SHA256_RE, "a sha256 (64 lowercase hex characters)"),
    };
    if (n.has("commit")) out.commit = n.get("commit").str();
    if (n.has("path")) out.path = n.get("path").str();
    return out;
  }
  if (n.has("url")) matching(n.get("url"), /^https:\/\//, "an https URL");
  if (n.has("sha256")) matching(n.get("sha256"), SHA256_RE, "a sha256 (64 lowercase hex characters)");
  return { kind, generator: n.get("generator").nonEmpty() };
}

/**
 * Validate one asset entry (the schema's `#/$defs/asset`) stored under key `id`,
 * including the content-addressing rules: `id` is sha256[:16] and `uri` is
 * `assets/<id>.glb`.
 */
export function readAsset(n: Node, id: string): ReplayAsset {
  onlyKeys(n, ASSET_KEYS);
  const sha256 = matching(n.get("sha256"), SHA256_RE, "a sha256 (64 lowercase hex characters)");
  const uri = matching(n.get("uri"), /^assets\/[0-9a-f]{16}\.glb$/, "assets/<16 hex>.glb");
  if (!sha256.startsWith(id)) n.get("sha256").fail(`asset ${id} is not content addressed: its sha256 starts ${sha256.slice(0, 16)}`);
  if (uri !== `assets/${id}.glb`) n.get("uri").fail(`asset ${id} must be served at assets/${id}.glb, not ${uri}`);
  const bytes = n.get("bytes").int();
  if (bytes < 1) n.get("bytes").fail(`expected at least 1 byte, got ${bytes}`);
  const license = matching(n.get("license"), SPDX_RE, "an SPDX licence identifier");
  const changes = n
    .get("changes")
    .items()
    .map((c) => c.nonEmpty());
  if (license !== OWN_LICENSE && changes.length === 0) {
    n.get("changes").fail(`third-party asset (${license}) must list what was changed from the upstream file`);
  }
  const asset: ReplayAsset = {
    uri,
    sha256,
    bytes,
    role: n.get("role").oneOf(ASSET_ROLES),
    license,
    attribution: n.get("attribution").nonEmpty(),
    changes,
  };
  if (n.has("label")) asset.label = n.get("label").nonEmpty();
  if (n.has("media_type")) asset.media_type = n.get("media_type").literal("model/gltf-binary");
  if (n.has("brand_visible")) asset.brand_visible = n.get("brand_visible").bool();
  if (n.has("source")) asset.source = readSource(n.get("source"));
  return asset;
}

/** Validate an `assets` map: keys are asset ids, values asset entries. */
export function readAssetMap(n: Node): Record<string, ReplayAsset> {
  const out: Record<string, ReplayAsset> = {};
  for (const [id, entry] of n.entries()) {
    readAssetId(new Node(n.file, `${n.path}.${id}`, id));
    out[id] = readAsset(entry, id);
  }
  return out;
}

export function isThirdParty(asset: Pick<ReplayAsset, "license">): boolean {
  return asset.license !== OWN_LICENSE;
}

// ---------------------------------------------------------------- manifest

export interface AssetManifest {
  format: "rtt-asset-manifest";
  version: 1;
  assets: Record<string, ReplayAsset>;
}

/** Validate `public/assets/manifest.json`. */
export function parseAssetManifest(raw: unknown, file = "manifest.json"): AssetManifest {
  const root = new Node(file, "$", raw);
  onlyKeys(root, ["format", "version", "assets"]);
  root.get("format").literal("rtt-asset-manifest");
  root.get("version").literal(1);
  return { format: "rtt-asset-manifest", version: 1, assets: readAssetMap(root.get("assets")) };
}

/** One credit line with every asset that carries it, for the credits page. */
export interface CreditGroup {
  attribution: string;
  license: string;
  roles: AssetRole[];
  changes: string[];
  count: number;
}

/** Group a manifest's third-party assets by credit line (our own assets are covered by the repo licence). */
export function creditGroups(assets: Record<string, ReplayAsset>): CreditGroup[] {
  const groups = new Map<string, CreditGroup>();
  for (const asset of Object.values(assets)) {
    if (!isThirdParty(asset)) continue;
    const key = `${asset.attribution}\u0000${asset.license}`;
    const g = groups.get(key) ?? { attribution: asset.attribution, license: asset.license, roles: [], changes: [], count: 0 };
    g.count += 1;
    if (!g.roles.includes(asset.role)) g.roles.push(asset.role);
    for (const c of asset.changes) if (!g.changes.includes(c)) g.changes.push(c);
    groups.set(key, g);
  }
  return [...groups.values()].sort((a, b) => a.attribution.localeCompare(b.attribution));
}

// ---------------------------------------------------------------- URLs and payload

/** URL of an asset file: `uri` is relative to the site root, so the base path is prefixed. */
export function assetUrl(asset: Pick<ReplayAsset, "uri">, base = BASE_PATH): string {
  return `${base}/${asset.uri.replace(/^\/+/, "")}`;
}

/** Ids of the assets a header draws: body visuals and static visuals, each once, in order. */
export function referencedAssets(header: Pick<ReplayHeaderV2, "bodies" | "static_visuals">): string[] {
  const seen = new Set<string>();
  for (const body of header.bodies) for (const v of body.visuals) seen.add(v.asset);
  for (const v of header.static_visuals) seen.add(v.asset);
  return [...seen];
}

/** The hero replay's download budget: header, poses, overlays and every GLB it draws. */
export const HERO_PAYLOAD_BUDGET_BYTES = 300_000;

export interface ReplayPayload {
  header: number;
  poses: number;
  overlays: number;
  assets: number;
  total: number;
}

/**
 * Bytes a page downloads to play one v2 replay with an empty asset cache. Pose and
 * overlay sizes follow from the header; `headerBytes` is the size of the JSON file.
 */
export function replayPayload(header: ReplayHeaderV2, headerBytes: number): ReplayPayload {
  const poses = header.n_frames * header.n_bodies * 7 * 4;
  const overlays = header.overlays ? header.n_frames * header.overlays.channels.length * 2 : 0;
  const assets = referencedAssets(header).reduce((sum, id) => sum + (header.assets[id]?.bytes ?? 0), 0);
  return { header: headerBytes, poses, overlays, assets, total: headerBytes + poses + overlays + assets };
}

/**
 * Asset ids that must be drawn untextured ("clay") on hero frames and in the gallery:
 * scanned objects and props whose texture shows a brand mark, or whose brand screen
 * has not run (`brand_visible` missing on a third-party object).
 */
export function clayAssets(assets: Record<string, ReplayAsset>): Set<string> {
  const out = new Set<string>();
  for (const [id, a] of Object.entries(assets)) {
    if (a.role !== "object" && a.role !== "prop") continue;
    if (a.brand_visible === true || (a.brand_visible === undefined && isThirdParty(a))) out.add(id);
  }
  return out;
}

// ---------------------------------------------------------------- GLB container

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"

/** The parts of the glTF JSON this module reads. */
export interface GltfJson {
  asset?: { version?: string; generator?: string };
  extensionsUsed?: string[];
  extensionsRequired?: string[];
  meshes?: { primitives?: { indices?: number; attributes?: Record<string, number> }[] }[];
  accessors?: { count?: number }[];
  images?: { mimeType?: string }[];
  textures?: unknown[];
  materials?: unknown[];
  nodes?: unknown[];
  [key: string]: unknown;
}

export interface GlbParts {
  json: GltfJson;
  /** The binary chunk, or null when the file has none. */
  bin: Uint8Array | null;
}

/** Split a GLB (glTF 2.0 binary) into its JSON and binary chunks, checking the container. */
export function readGlb(buffer: ArrayBuffer): GlbParts {
  if (buffer.byteLength < 20) throw new AssetError(`not a GLB: ${buffer.byteLength} bytes is shorter than the header`);
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== GLB_MAGIC) throw new AssetError("not a GLB: missing the glTF magic");
  const version = view.getUint32(4, true);
  if (version !== 2) throw new AssetError(`unsupported GLB version ${version} (expected 2)`);
  const length = view.getUint32(8, true);
  if (length !== buffer.byteLength) throw new AssetError(`GLB header says ${length} bytes, file has ${buffer.byteLength}`);
  let offset = 12;
  let json: GltfJson | null = null;
  let bin: Uint8Array | null = null;
  while (offset + 8 <= length) {
    const chunkLength = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (start + chunkLength > length) throw new AssetError("GLB chunk runs past the end of the file");
    if (type === CHUNK_JSON && json === null) {
      const text = new TextDecoder().decode(new Uint8Array(buffer, start, chunkLength));
      try {
        json = JSON.parse(text) as GltfJson;
      } catch (err) {
        throw new AssetError(`GLB JSON chunk does not parse: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else if (type === CHUNK_BIN && bin === null) {
      bin = new Uint8Array(buffer, start, chunkLength);
    }
    offset = start + chunkLength;
  }
  if (!json) throw new AssetError("GLB has no JSON chunk");
  return { json, bin };
}

/**
 * Extensions the site can load. EXT_meshopt_compression is decoded by three.js's
 * bundled meshopt decoder; KHR_draco_mesh_compression is not supported, because its
 * decoder would have to be shipped as extra WASM or fetched from a CDN.
 */
export const SUPPORTED_EXTENSIONS: ReadonlySet<string> = new Set([
  "EXT_meshopt_compression",
  "KHR_meshopt_compression",
  "KHR_mesh_quantization",
  "EXT_texture_webp",
  "KHR_texture_transform",
  "KHR_materials_emissive_strength",
  "KHR_materials_ior",
  "KHR_materials_specular",
  "KHR_materials_clearcoat",
  "KHR_materials_transmission",
  "KHR_materials_volume",
  "KHR_materials_unlit",
]);

/** Required extensions this site cannot decode (empty when the file loads). */
export function unsupportedExtensions(json: GltfJson): string[] {
  return (json.extensionsRequired ?? []).filter((e) => !SUPPORTED_EXTENSIONS.has(e));
}

export interface GlbInfo {
  extensionsUsed: string[];
  extensionsRequired: string[];
  meshes: number;
  primitives: number;
  /** Triangles of indexed primitives (index count / 3), plus non-indexed position counts / 3. */
  triangles: number;
  images: number;
  imageTypes: string[];
}

/** Counts and extensions of a GLB, read from its JSON chunk without decoding any buffer. */
export function glbInfo(json: GltfJson): GlbInfo {
  let primitives = 0;
  let triangles = 0;
  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      primitives += 1;
      const accessor = prim.indices ?? prim.attributes?.POSITION;
      const count = accessor === undefined ? 0 : (json.accessors?.[accessor]?.count ?? 0);
      triangles += Math.floor(count / 3);
    }
  }
  const imageTypes = [...new Set((json.images ?? []).map((i) => i.mimeType ?? "unknown"))];
  return {
    extensionsUsed: json.extensionsUsed ?? [],
    extensionsRequired: json.extensionsRequired ?? [],
    meshes: json.meshes?.length ?? 0,
    primitives,
    triangles,
    images: json.images?.length ?? 0,
    imageTypes,
  };
}

// ---------------------------------------------------------------- verification

/** Lowercase hex sha256 of `data`, or null where Web Crypto is unavailable (plain-HTTP pages). */
export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  const digest = await subtle.digest("SHA-256", view as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Check fetched bytes against their entry: size, sha256 (when Web Crypto is available),
 * the GLB container and its required extensions. Throws AssetError naming the asset.
 */
export async function verifyAsset(id: string, asset: ReplayAsset, buffer: ArrayBuffer): Promise<GlbParts> {
  if (buffer.byteLength !== asset.bytes) {
    throw new AssetError(`asset ${id} is ${buffer.byteLength} bytes; its entry says ${asset.bytes}`);
  }
  const sha = await sha256Hex(buffer);
  if (sha !== null && sha !== asset.sha256) {
    throw new AssetError(`asset ${id} does not match its sha256 (got ${sha.slice(0, 16)}...)`);
  }
  let parts: GlbParts;
  try {
    parts = readGlb(buffer);
  } catch (err) {
    throw new AssetError(`asset ${id}: ${err instanceof Error ? err.message : String(err)}`);
  }
  const missing = unsupportedExtensions(parts.json);
  if (missing.length > 0) throw new AssetError(`asset ${id} needs ${missing.join(", ")}, which this site does not decode`);
  return parts;
}

// ---------------------------------------------------------------- loading

/** Axis-aligned box of an asset in its own frame (the body frame), metres. */
export interface AssetBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface LoadedAsset {
  id: string;
  asset: ReplayAsset;
  /** Template scene; draw clones of it (see `instantiate`), never the template itself. */
  scene: Group;
  box: AssetBox;
  info: GlbInfo;
}

type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export interface AssetLibraryOptions {
  fetch?: FetchLike;
  /** URL prefix for `uri` (the site's base path). */
  base?: string;
  /** Verify sha256 and size before parsing (default true). */
  verify?: boolean;
}

type GltfLoaderLike = { parseAsync: (data: ArrayBuffer, path: string) => Promise<{ scene: Group }> };

let loaderPromise: Promise<GltfLoaderLike> | null = null;

/** three.js's GLTFLoader with the bundled meshopt decoder, created once (and only in code that draws). */
async function gltfLoader(): Promise<GltfLoaderLike> {
  loaderPromise ??= (async () => {
    const [{ GLTFLoader }, meshopt] = await Promise.all([
      import("three/examples/jsm/loaders/GLTFLoader.js"),
      // Typed through the `meshoptimizer` package, which is not installed: declare the shape used here.
      import("three/examples/jsm/libs/meshopt_decoder.module.js") as Promise<unknown> as Promise<{
        MeshoptDecoder: { ready: Promise<void>; supported: boolean };
      }>,
    ]);
    const { MeshoptDecoder } = meshopt;
    await MeshoptDecoder.ready;
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder as unknown as Parameters<typeof loader.setMeshoptDecoder>[0]);
    return loader as unknown as GltfLoaderLike;
  })();
  return loaderPromise;
}

/**
 * Loads GLB assets by id, once each: a replay that draws an asset another replay on
 * the page already loaded reuses it. Counts the bytes it fetched for the payload readout.
 */
export class AssetLibrary {
  private readonly cache = new Map<string, Promise<LoadedAsset>>();
  private readonly fetchImpl: FetchLike;
  private readonly base: string;
  private readonly verify: boolean;
  /** Bytes downloaded so far, over all assets. */
  bytesFetched = 0;

  constructor(options: AssetLibraryOptions = {}) {
    this.fetchImpl = options.fetch ?? ((url, init) => fetch(url, init));
    this.base = options.base ?? BASE_PATH;
    this.verify = options.verify ?? true;
  }

  has(id: string): boolean {
    return this.cache.has(id);
  }

  /** Fetch and check an asset's bytes. */
  async fetchBytes(id: string, asset: ReplayAsset, init?: RequestInit): Promise<{ buffer: ArrayBuffer; parts: GlbParts }> {
    const url = assetUrl(asset, this.base);
    const res = await this.fetchImpl(url, init);
    if (!res.ok) throw new AssetError(`asset ${id} (${url}): HTTP ${res.status}`);
    const buffer = await res.arrayBuffer();
    this.bytesFetched += buffer.byteLength;
    const parts = this.verify ? await verifyAsset(id, asset, buffer) : readGlb(buffer);
    return { buffer, parts };
  }

  /** The loaded asset, fetching and decoding it on first use. A failed load is retried next time. */
  load(id: string, asset: ReplayAsset): Promise<LoadedAsset> {
    const cached = this.cache.get(id);
    if (cached) return cached;
    const promise = (async () => {
      const { buffer, parts } = await this.fetchBytes(id, asset);
      const loader = await gltfLoader();
      const gltf = await loader.parseAsync(buffer, "");
      const { Box3 } = await import("three");
      gltf.scene.updateMatrixWorld(true);
      const b = new Box3().setFromObject(gltf.scene);
      const box: AssetBox = b.isEmpty()
        ? { min: [0, 0, 0], max: [0, 0, 0] }
        : { min: [b.min.x, b.min.y, b.min.z], max: [b.max.x, b.max.y, b.max.z] };
      return { id, asset, scene: gltf.scene, box, info: glbInfo(parts.json) };
    })();
    this.cache.set(id, promise);
    promise.catch(() => this.cache.delete(id));
    return promise;
  }

  /** Load every asset a header draws. */
  async loadAll(header: Pick<ReplayHeaderV2, "assets" | "bodies" | "static_visuals">): Promise<Map<string, LoadedAsset>> {
    const ids = referencedAssets(header);
    const loaded = await Promise.all(ids.map((id) => this.load(id, header.assets[id]!)));
    return new Map(loaded.map((a) => [a.id, a]));
  }
}

/** The page-wide library: assets are shared across replays, lanes and views. */
export const sharedAssets = new AssetLibrary();

/**
 * A drawable copy of a template scene. Geometry is shared; materials are cloned so a
 * view can make one instance transparent (x-ray) or clipped without touching others.
 */
export function instantiate<T extends Object3D>(template: T): T {
  const copy = template.clone(true) as T;
  copy.traverse((o) => {
    const mesh = o as unknown as { isMesh?: boolean; material?: unknown };
    if (!mesh.isMesh || !mesh.material) return;
    const clone = (m: { clone: () => unknown }) => m.clone();
    mesh.material = Array.isArray(mesh.material)
      ? (mesh.material as { clone: () => unknown }[]).map(clone)
      : clone(mesh.material as { clone: () => unknown });
  });
  return copy;
}

/** The 8 corners of `box` placed by a visual's pos, quat (xyzw) and scale, in the body frame. */
export function placedCorners(
  box: AssetBox,
  pos: readonly number[],
  quat: readonly number[],
  scale: readonly number[],
): number[][] {
  const out: number[][] = [];
  const [qx = 0, qy = 0, qz = 0, qw = 1] = quat;
  for (const x of [box.min[0], box.max[0]])
    for (const y of [box.min[1], box.max[1]])
      for (const z of [box.min[2], box.max[2]]) {
        const vx = x * (scale[0] ?? 1);
        const vy = y * (scale[1] ?? 1);
        const vz = z * (scale[2] ?? 1);
        const tx = 2 * (qy * vz - qz * vy);
        const ty = 2 * (qz * vx - qx * vz);
        const tz = 2 * (qx * vy - qy * vx);
        out.push([
          vx + qw * tx + (qy * tz - qz * ty) + (pos[0] ?? 0),
          vy + qw * ty + (qz * tx - qx * tz) + (pos[1] ?? 0),
          vz + qw * tz + (qx * ty - qy * tx) + (pos[2] ?? 0),
        ]);
      }
  return out;
}
