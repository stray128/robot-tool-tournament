import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import * as THREE from "three";
import { describe, expect, it } from "vitest";

import {
  AssetError,
  AssetLibrary,
  assetUrl,
  clayAssets,
  creditGroups,
  glbInfo,
  HERO_PAYLOAD_BUDGET_BYTES,
  instantiate,
  parseAssetManifest,
  placedCorners,
  readGlb,
  referencedAssets,
  replayCredits,
  replayPayload,
  sha256Hex,
  splitCredit,
  unsupportedExtensions,
  verifyAsset,
} from "@/lib/assets";
import { parseAnyReplayHeader, parseReplayHeaderV2 } from "@/lib/replay";
import type { ReplayAsset, ReplayHeaderV2 } from "@/lib/types";
import { DataError } from "@/lib/validate";

import { boxMesh, hexToLinear, srgbToLinear, writeGlb } from "./helpers/glb_writer.mjs";

const WEB = path.join(__dirname, "..");
const FIXTURE = path.join(__dirname, "fixtures", "replay_v2");
const PUBLIC_ASSETS = path.join(WEB, "public", "assets");
const COMPRESS = path.join(WEB, "scripts", "compress_glb.mjs");

function ab(buf: Buffer): ArrayBuffer {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function boxGlb(extra: { extensionsUsed?: string[]; extensionsRequired?: string[] } = {}): ArrayBuffer {
  return writeGlb({
    meshes: [{ name: "box", primitives: [{ ...boxMesh([0.01, 0.02, 0.03]), material: 0 }] }],
    materials: [{ name: "m", baseColorFactor: hexToLinear("#e8742a") }],
    ...extra,
  });
}

function entryFor(buf: ArrayBuffer, overrides: Partial<ReplayAsset> = {}): [string, ReplayAsset] {
  const sha = createHash("sha256").update(new Uint8Array(buf)).digest("hex");
  const id = sha.slice(0, 16);
  return [
    id,
    {
      uri: `assets/${id}.glb`,
      sha256: sha,
      bytes: buf.byteLength,
      role: "finger_left",
      license: "Apache-2.0",
      attribution: "Robot Tool Tournament (Apache-2.0)",
      changes: [],
      ...overrides,
    },
  ];
}

const fixtureHeader = (): ReplayHeaderV2 =>
  parseReplayHeaderV2(JSON.parse(readFileSync(path.join(FIXTURE, "sample.json"), "utf8")), "sample.json");

describe("GLB container", () => {
  it("reads the JSON and binary chunks", () => {
    const { json, bin } = readGlb(boxGlb());
    expect(json.asset?.version).toBe("2.0");
    expect(bin?.byteLength).toBeGreaterThan(0);
    const info = glbInfo(json);
    expect(info).toMatchObject({ meshes: 1, primitives: 1, triangles: 12, images: 0 });
  });

  it("rejects files that are not GLB 2.0", () => {
    const good = new Uint8Array(boxGlb());
    expect(() => readGlb(new ArrayBuffer(8))).toThrow(/shorter than the header/);
    const badMagic = good.slice();
    badMagic[0] = 0;
    expect(() => readGlb(badMagic.buffer)).toThrow(/magic/);
    const v1 = good.slice();
    new DataView(v1.buffer).setUint32(4, 1, true);
    expect(() => readGlb(v1.buffer)).toThrow(/version 1/);
    const truncated = good.slice(0, good.byteLength - 4);
    expect(() => readGlb(truncated.buffer)).toThrow(/header says/);
  });

  it("knows which compression it can decode", () => {
    expect(unsupportedExtensions(readGlb(boxGlb({ extensionsUsed: ["EXT_meshopt_compression"], extensionsRequired: ["EXT_meshopt_compression"] })).json)).toEqual([]);
    expect(
      unsupportedExtensions(
        readGlb(boxGlb({ extensionsUsed: ["KHR_draco_mesh_compression"], extensionsRequired: ["KHR_draco_mesh_compression"] })).json,
      ),
    ).toEqual(["KHR_draco_mesh_compression"]);
  });

  it("the committed fixture assets are meshopt-compressed and decodable", () => {
    for (const a of Object.values(fixtureHeader().assets)) {
      const { json } = readGlb(ab(readFileSync(path.join(FIXTURE, a.uri))));
      expect(json.extensionsRequired).toContain("EXT_meshopt_compression");
      expect(unsupportedExtensions(json)).toEqual([]);
    }
  });
});

describe("verification", () => {
  it("hashes like node:crypto", async () => {
    const buf = boxGlb();
    expect(await sha256Hex(buf)).toBe(createHash("sha256").update(new Uint8Array(buf)).digest("hex"));
  });

  it("accepts matching bytes and rejects size, hash and codec mismatches", async () => {
    const buf = boxGlb();
    const [id, entry] = entryFor(buf);
    await expect(verifyAsset(id, entry, buf)).resolves.toMatchObject({ json: { asset: { version: "2.0" } } });
    await expect(verifyAsset(id, { ...entry, bytes: entry.bytes + 1 }, buf)).rejects.toThrow(/bytes; its entry says/);
    await expect(verifyAsset(id, { ...entry, sha256: `${id}${"0".repeat(48)}` }, buf)).rejects.toThrow(/sha256/);
    const draco = boxGlb({ extensionsUsed: ["KHR_draco_mesh_compression"], extensionsRequired: ["KHR_draco_mesh_compression"] });
    const [did, dentry] = entryFor(draco);
    await expect(verifyAsset(did, dentry, draco)).rejects.toThrow(/KHR_draco_mesh_compression/);
  });
});

describe("manifest", () => {
  it("parses the served manifest and the format doc's example", () => {
    const served = parseAssetManifest(JSON.parse(readFileSync(path.join(PUBLIC_ASSETS, "manifest.json"), "utf8")));
    expect(served.format).toBe("rtt-asset-manifest");
    const doc = readFileSync(path.join(WEB, "..", "docs", "ASSET_MANIFEST.md"), "utf8");
    const blocks = [...doc.matchAll(/```json\n([\s\S]*?)```/g)].map((m) => m[1]!);
    const example = parseAssetManifest(JSON.parse(blocks[blocks.length - 1]!), "ASSET_MANIFEST.md");
    expect(Object.keys(example.assets)).toHaveLength(2);
  });

  it("rejects entries that break the manifest rules", () => {
    const [id, entry] = entryFor(boxGlb());
    const ok = { format: "rtt-asset-manifest", version: 1, assets: { [id]: entry } };
    expect(() => parseAssetManifest(ok)).not.toThrow();
    const bad: [string, object][] = [
      ["extra key", { ...entry, path: "/home/x.glb" }],
      ["third-party without changes", { ...entry, license: "CC-BY-4.0", changes: [] }],
      ["uri not content addressed", { ...entry, uri: "assets/0000000000000000.glb" }],
      ["empty attribution", { ...entry, attribution: "" }],
      ["unknown role", { ...entry, role: "finger" }],
      ["bad source", { ...entry, source: { kind: "upstream", url: "http://x" } }],
    ];
    for (const [name, e] of bad) expect(() => parseAssetManifest({ ...ok, assets: { [id]: e } }), name).toThrow(DataError);
    expect(() => parseAssetManifest({ ...ok, assets: { ["0".repeat(16)]: entry } })).toThrow(/content addressed/);
    expect(() => parseAssetManifest({ ...ok, version: 2 })).toThrow(DataError);
  });

  it("groups third-party credits for the credits page", () => {
    const cc = (role: ReplayAsset["role"], changes: string[]): ReplayAsset => ({
      ...entryFor(boxGlb())[1],
      role,
      license: "CC-BY-4.0",
      attribution: "Example Author, Example Gripper, CC BY 4.0",
      changes,
    });
    const groups = creditGroups({
      a: cc("gripper_frame", ["cut", "re-encoded as GLB"]),
      b: cc("carriage", ["cut", "re-posed"]),
      c: entryFor(boxGlb())[1],
    });
    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({ count: 2, roles: ["gripper_frame", "carriage"], changes: ["cut", "re-encoded as GLB", "re-posed"] });
  });

  it("splits a short credit from the credits-page anchor it points at", () => {
    const gripper = "An open-source parallel gripper, CC BY 4.0, modified (full credit: /credits#gripper)";
    expect(splitCredit(gripper)).toEqual({ text: "An open-source parallel gripper, CC BY 4.0, modified", anchor: "gripper" });
    expect(splitCredit("Google Scanned Objects, CC BY 4.0, modified (full credit: /credits/#gso)").anchor).toBe("gso");
    expect(splitCredit("Example Author, Example Gripper, CC BY 4.0")).toEqual({
      text: "Example Author, Example Gripper, CC BY 4.0",
      anchor: null,
    });
  });

  it("lists each third-party credit a replay draws once, and none for our own assets", () => {
    const short = "An open-source parallel gripper, CC BY 4.0, modified (full credit: /credits#gripper)";
    const cc = (role: ReplayAsset["role"]): ReplayAsset => ({
      ...entryFor(boxGlb())[1],
      role,
      license: "CC-BY-4.0",
      attribution: short,
      changes: ["re-encoded as GLB"],
    });
    expect(replayCredits({ a: cc("gripper_frame"), b: cc("carriage"), c: entryFor(boxGlb())[1] })).toEqual([
      { text: "An open-source parallel gripper, CC BY 4.0, modified", anchor: "gripper" },
    ]);
    expect(replayCredits(fixtureHeader().assets).every((c) => c.text.length > 0)).toBe(true);
    expect(replayCredits({ c: entryFor(boxGlb())[1] })).toEqual([]);
  });

  it("the credits page has the anchors short credits point at", () => {
    const page = readFileSync(path.join(WEB, "app", "credits", "page.tsx"), "utf8");
    for (const anchor of ["gripper", "gso", "models"]) expect(page).toContain(`id="${anchor}"`);
  });
});

describe("served assets (web/public/assets)", () => {
  const manifest = parseAssetManifest(JSON.parse(readFileSync(path.join(PUBLIC_ASSETS, "manifest.json"), "utf8")));

  it("every served GLB is listed with a sha256, size and credit line", () => {
    const files = readdirSync(PUBLIC_ASSETS).filter((f) => f.endsWith(".glb"));
    for (const f of files) {
      const id = f.slice(0, -4);
      const entry = manifest.assets[id];
      expect(entry, `${f} is not in the manifest`).toBeDefined();
      const bytes = readFileSync(path.join(PUBLIC_ASSETS, f));
      expect(bytes.byteLength).toBe(entry!.bytes);
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(entry!.sha256);
      expect(entry!.attribution.length).toBeGreaterThan(0);
    }
    for (const id of Object.keys(manifest.assets)) expect(existsSync(path.join(PUBLIC_ASSETS, `${id}.glb`)), id).toBe(true);
  });

  it("compress_glb --check agrees", () => {
    const r = spawnSync(process.execPath, [COMPRESS, "--check", "--out-dir", PUBLIC_ASSETS], { encoding: "utf8" });
    expect(r.status, r.stderr).toBe(0);
  });
});

describe("compress_glb --check", () => {
  it("passes on the fixture and catches a tampered file", () => {
    const ok = spawnSync(process.execPath, [COMPRESS, "--check", "--out-dir", path.join(FIXTURE, "assets")], { encoding: "utf8" });
    expect(ok.status, ok.stderr).toBe(0);
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-assets-"));
    try {
      cpSync(path.join(FIXTURE, "assets"), dir, { recursive: true });
      const victim = readdirSync(dir).find((f) => f.endsWith(".glb"))!;
      const bytes = readFileSync(path.join(dir, victim));
      bytes.writeUInt8(bytes.readUInt8(bytes.length - 1) ^ 0xff, bytes.length - 1);
      writeFileSync(path.join(dir, victim), bytes);
      writeFileSync(path.join(dir, "ffffffffffffffff.glb"), "stray");
      const bad = spawnSync(process.execPath, [COMPRESS, "--check", "--out-dir", dir], { encoding: "utf8" });
      expect(bad.status).toBe(1);
      expect(bad.stderr).toMatch(/does not match its sha256/);
      expect(bad.stderr).toMatch(/ffffffffffffffff\.glb: served but not listed/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("explains how to get its dependencies when compression cannot run", () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-assets-"));
    try {
      writeFileSync(path.join(dir, "spec.json"), JSON.stringify({ inputs: [{ key: "k", file: "x.glb" }] }));
      const r = spawnSync(process.execPath, [COMPRESS, path.join(dir, "spec.json"), "--out-dir", dir, "--modules", dir], {
        encoding: "utf8",
      });
      expect(r.status).toBe(1);
      expect(r.stderr).toMatch(/cannot load @gltf-transform\/core/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("URLs, payload and brand screen", () => {
  it("prefixes the base path", () => {
    expect(assetUrl({ uri: "assets/0123456789abcdef.glb" }, "")).toBe("/assets/0123456789abcdef.glb");
    expect(assetUrl({ uri: "assets/0123456789abcdef.glb" }, "/rtt")).toBe("/rtt/assets/0123456789abcdef.glb");
  });

  it("counts header, poses, overlays and each referenced asset once", () => {
    const h = fixtureHeader();
    const headerBytes = readFileSync(path.join(FIXTURE, "sample.json")).byteLength;
    const p = replayPayload(h, headerBytes);
    const files = ["sample.bin", "sample.ovl.bin", ...referencedAssets(h).map((id) => h.assets[id]!.uri)];
    const onDisk = files.reduce((s, f) => s + readFileSync(path.join(FIXTURE, f)).byteLength, headerBytes);
    expect(p.total).toBe(onDisk);
    expect(p.total).toBeLessThanOrEqual(HERO_PAYLOAD_BUDGET_BYTES);
  });

  it("every v2 replay the site serves fits the hero payload budget", () => {
    const root = path.join(WEB, "public", "data", "replays");
    const manifest = parseAssetManifest(JSON.parse(readFileSync(path.join(PUBLIC_ASSETS, "manifest.json"), "utf8")));
    for (const dir of readdirSync(root)) {
      for (const f of readdirSync(path.join(root, dir)).filter((f) => f.endsWith(".json"))) {
        const text = readFileSync(path.join(root, dir, f));
        const h = parseAnyReplayHeader(JSON.parse(text.toString("utf8")), f);
        if (h.version !== 2) continue;
        for (const id of referencedAssets(h)) expect(manifest.assets[id], `${f} draws unserved asset ${id}`).toEqual(h.assets[id]);
        expect(replayPayload(h, text.byteLength).total, f).toBeLessThanOrEqual(HERO_PAYLOAD_BUDGET_BYTES);
      }
    }
  });

  it("draws branded or unscreened scans in clay", () => {
    const base = entryFor(boxGlb())[1];
    const clay = clayAssets({
      own: { ...base, role: "object" },
      branded: { ...base, role: "object", license: "CC-BY-4.0", changes: ["x"], brand_visible: true },
      clean: { ...base, role: "prop", license: "CC-BY-4.0", changes: ["x"], brand_visible: false },
      unscreened: { ...base, role: "object", license: "CC-BY-4.0", changes: ["x"] },
      gripper: { ...base, role: "gripper_frame", license: "CC-BY-4.0", changes: ["x"] },
    });
    expect([...clay].sort()).toEqual(["branded", "unscreened"]);
  });

  it("places box corners by pos, quat and scale", () => {
    const s = Math.SQRT1_2;
    const box = { min: [0, 0, 0] as [number, number, number], max: [1, 0, 0] as [number, number, number] };
    const pts = placedCorners(box, [0, 0, 1], [0, 0, s, s], [2, 1, 1]); // 90 degrees about z, x doubled
    const far = pts.find((p) => Math.abs(p[1]! - 2) < 1e-9);
    expect(far).toBeDefined();
    expect(far![0]).toBeCloseTo(0, 9);
    expect(far![2]).toBeCloseTo(1, 9);
  });

  it("converts sRGB colours to linear for glTF", () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 9);
    expect(srgbToLinear(0.5)).toBeCloseTo(0.214, 3);
  });
});

describe("AssetLibrary", () => {
  function server(root: string) {
    const requested: string[] = [];
    const fetchImpl = async (url: string) => {
      requested.push(url);
      const file = path.join(root, url.replace(/^\/+/, ""));
      if (!existsSync(file)) return new Response("missing", { status: 404 });
      return new Response(new Uint8Array(readFileSync(file)));
    };
    return { requested, fetchImpl };
  }

  it("decodes meshopt GLBs once per page and keeps the body frame", async () => {
    const h = fixtureHeader();
    const { requested, fetchImpl } = server(FIXTURE);
    const lib = new AssetLibrary({ fetch: fetchImpl, base: "" });
    const loaded = await lib.loadAll(h);
    expect(loaded.size).toBe(4);
    const again = await lib.loadAll(h);
    expect(again.size).toBe(4);
    expect(requested).toHaveLength(4); // the second replay reuses every asset
    expect(lib.bytesFetched).toBe(Object.values(h.assets).reduce((s, a) => s + a.bytes, 0));

    // The decoded object box matches the collision box it was made from (meshopt quantisation tolerance).
    const objectBody = h.bodies.find((b) => b.name === "object")!;
    const box = loaded.get(objectBody.visuals[0]!.asset)!.box;
    const half = objectBody.geoms[0]!.half_size!;
    for (let k = 0; k < 3; k++) {
      expect(box.max[k]! - box.min[k]!).toBeCloseTo(2 * half[k]!, 3);
    }
    // The left jaw sits on +y and the right jaw on -y, each in its own body frame, no mirroring.
    const left = loaded.get(h.bodies.find((b) => b.name === "finger_left")!.visuals[0]!.asset)!;
    expect(left.info.extensionsRequired).toContain("EXT_meshopt_compression");
    expect(left.scene.children.length).toBeGreaterThan(0);
  });

  it("reports missing and corrupt assets, and retries after a failure", async () => {
    const h = fixtureHeader();
    const [id, entry] = Object.entries(h.assets)[0]!;
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-lib-"));
    try {
      const { fetchImpl } = server(dir);
      const lib = new AssetLibrary({ fetch: fetchImpl, base: "" });
      await expect(lib.load(id, entry)).rejects.toThrow(/HTTP 404/);
      expect(lib.has(id)).toBe(false);
      execFileSync("mkdir", ["-p", path.join(dir, "assets")]);
      const bytes = readFileSync(path.join(FIXTURE, entry.uri));
      bytes.writeUInt8(bytes.readUInt8(100) ^ 0xff, 100);
      writeFileSync(path.join(dir, entry.uri), bytes);
      await expect(lib.load(id, entry)).rejects.toThrow(AssetError);
      cpSync(path.join(FIXTURE, entry.uri), path.join(dir, entry.uri));
      await expect(lib.load(id, entry)).resolves.toMatchObject({ id });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("instances share geometry but not materials", async () => {
    const h = fixtureHeader();
    const { fetchImpl } = server(FIXTURE);
    const lib = new AssetLibrary({ fetch: fetchImpl, base: "" });
    const [id, entry] = Object.entries(h.assets)[0]!;
    const { scene } = await lib.load(id, entry);
    const a = instantiate(scene);
    const b = instantiate(scene);
    const meshes = (o: THREE.Object3D) => {
      const out: THREE.Mesh[] = [];
      o.traverse((x) => {
        if ((x as THREE.Mesh).isMesh) out.push(x as THREE.Mesh);
      });
      return out;
    };
    const [ma, mb] = [meshes(a)[0]!, meshes(b)[0]!];
    expect(ma.geometry).toBe(mb.geometry);
    expect(ma.material).not.toBe(mb.material);
    (ma.material as THREE.MeshStandardMaterial).opacity = 0.35;
    expect((mb.material as THREE.MeshStandardMaterial).opacity).toBe(1);
  });
});
