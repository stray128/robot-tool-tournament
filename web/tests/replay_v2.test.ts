import { createHash } from "node:crypto";
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { placedCorners } from "@/lib/assets";
import { checkReplayFiles } from "@/lib/data";
import {
  decodeFrames,
  decodeOverlays,
  halfToFloat,
  isV2,
  loadReplay,
  OVERLAY_CHANNELS,
  parseAnyReplayHeader,
  parseReplayHeaderV2,
  Replay,
  ReplayFormatError,
  replayCaption,
} from "@/lib/replay";
import type { BoutsIndex, ReplayHeaderV2 } from "@/lib/types";
import { DataError } from "@/lib/validate";

const WEB = path.join(__dirname, "..");
const SCHEMA = JSON.parse(readFileSync(path.join(WEB, "..", "src", "rtt", "sim", "replay_v2.schema.json"), "utf8"));
const FIXTURE = path.join(__dirname, "fixtures", "replay_v2");
const V1_DIR = path.join(WEB, "public", "data", "replays");

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);

function asset(content: string, role: string, license: string, attribution: string, changes: string[]) {
  const sha = createHash("sha256").update(content).digest("hex");
  const id = sha.slice(0, 16);
  return [id, { uri: `assets/${id}.glb`, sha256: sha, bytes: content.length, role, license, attribution, changes }] as const;
}

const visual = (id: string) => ({ asset: id, pos: [0, 0, 0], quat_xyzw: [0, 0, 0, 1], scale: [1, 1, 1] });

/** The same sample header as tests/test_contracts_v2.py, so both readers agree on one document. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function sampleHeader(): any {
  const ours = "Robot Tool Tournament (Apache-2.0)";
  const upstream = "Example Author, Example Gripper, https://example.org/gripper, CC BY 4.0";
  const [frameId, frame] = asset("glTF frame", "gripper_frame", "CC-BY-4.0", upstream, ["cut", "re-encoded as GLB"]);
  const [leftId, left] = asset("glTF left jaw", "finger_left", "Apache-2.0", ours, []);
  const [rightId, right] = asset("glTF right jaw", "finger_right", "Apache-2.0", ours, []);
  const [partId, part] = asset("glTF tee", "object", "Apache-2.0", ours, []);
  return {
    format: "rtt-replay",
    version: 2,
    fps: 30,
    dtype: "float32",
    byte_order: "little",
    layout: ["frames", "bodies", "pose"],
    pose_fields: ["px", "py", "pz", "qx", "qy", "qz", "qw"],
    up_axis: "z",
    units: "m",
    n_frames: 184,
    n_bodies: 4,
    bin: "bout-001.bin",
    assets: {
      [frameId]: {
        ...frame,
        source: {
          kind: "upstream",
          url: "https://example.org/gripper",
          commit: "0123456789abcdef0123456789abcdef01234567",
          path: "simulation/meshes/frame.stl",
          sha256: SHA_B,
        },
      },
      [leftId]: { ...left, source: { kind: "generated", generator: "rtt.render.glb" } },
      [rightId]: right,
      [partId]: part,
    },
    bodies: [
      { name: "palm", visuals: [visual(frameId)] },
      { name: "finger_left", visuals: [visual(leftId)] },
      {
        name: "finger_right",
        visuals: [visual(rightId)],
        geoms: [
          {
            name: "pad0",
            type: "box",
            half_size: [0.01, 0.0015, 0.01],
            pos: [0.0, -0.002, -0.06],
            quat_xyzw: [0.0, 0.0, 0.0, 1.0],
            rgba: [0.9, 0.5, 0.1, 1.0],
          },
        ],
      },
      { name: "object", visuals: [visual(partId)] },
    ],
    static_visuals: [visual(partId)],
    meta: { design_space: "freeform-v2", design_id: SHA_A, geometry_id: SHA_B, part_id: "pipe_tee", metrics: { success: true } },
    provenance: {
      source: "simulation",
      engine: "mujoco",
      engine_version: "3.15.0",
      run_id: "pilot-a1",
      code_version: "f60f954+dirty",
      product_mode: false,
      suite: "train_v2",
      condition: "c1",
      seed: 0,
      created_utc: "2026-10-17T10:00:00Z",
    },
    overlays: {
      bin: "bout-001.ovl.bin",
      dtype: "float16",
      byte_order: "little",
      layout: ["frames", "channels"],
      channels: ["offset_mm", "seat_short_mm", "hold_s", "n_contacts"],
    },
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Header = any;
const firstAsset = (h: Header) => Object.values(h.assets)[0] as Header;
const upstreamAsset = (h: Header) => Object.values(h.assets).find((a) => (a as Header).license !== "Apache-2.0") as Header;

/** The 27 breakages of tests/test_contracts_v2.py::_mutations, plus the cross-field rules. */
const MUTATIONS: [string, (h: Header) => void][] = [
  ["v1 header", (h) => (h.version = 1)],
  ["no provenance", (h) => delete h.provenance],
  ["no meta", (h) => delete h.meta],
  ["unknown top-level key", (h) => (h.qpos_bin = "x.qpos.bin")],
  ["bin path with a directory", (h) => (h.bin = "../x.bin")],
  ["layout changed", (h) => (h.layout = ["bodies", "frames", "pose"])],
  ["float fps", (h) => (h.fps = 29.97)],
  ["asset id not hex", (h) => (h.assets.Gripper = firstAsset(h))],
  ["asset uri elsewhere", (h) => (firstAsset(h).uri = "data/assets/x.glb")],
  ["asset sha uppercase", (h) => (firstAsset(h).sha256 = "A".repeat(64))],
  ["asset role unknown", (h) => (firstAsset(h).role = "finger")],
  ["asset extra key", (h) => (firstAsset(h).local_path = "/tmp/x.glb")],
  ["third-party asset without changes", (h) => (upstreamAsset(h).changes = [])],
  ["upstream source without sha", (h) => delete upstreamAsset(h).source.sha256],
  ["generated source without generator", (h) => delete h.assets[h.bodies[1].visuals[0].asset].source.generator],
  ["mirrored visual", (h) => (h.bodies[2].visuals[0].scale = [1, -1, 1])],
  ["visual without scale", (h) => delete h.bodies[1].visuals[0].scale],
  ["quat with 3 terms", (h) => (h.bodies[1].visuals[0].quat_xyzw = [0, 0, 1])],
  ["body with nothing to draw", (h) => (h.bodies[0].visuals = [])],
  ["meta without part_id", (h) => delete h.meta.part_id],
  ["geometry_id not a sha", (h) => (h.meta.geometry_id = "abc")],
  ["provenance from another engine", (h) => (h.provenance.engine = "isaac")],
  ["provenance product_mode missing", (h) => delete h.provenance.product_mode],
  ["provenance bad commit", (h) => (h.provenance.code_version = "main")],
  ["overlay channel unknown", (h) => h.overlays.channels.push("force_n")],
  ["overlay channel repeated", (h) => h.overlays.channels.push("hold_s")],
  ["overlay float32", (h) => (h.overlays.dtype = "float32")],
  // cross-field rules from the schema's $comment
  ["n_bodies disagrees", (h) => (h.n_bodies = 5)],
  ["visual names an unlisted asset", (h) => (h.bodies[0].visuals[0].asset = "0123456789abcdef")],
  [
    "asset key is not its sha prefix",
    (h) => {
      const [id, entry] = Object.entries(h.assets)[0] as [string, Header];
      delete h.assets[id];
      h.assets["0123456789abcdef"] = { ...entry, uri: "assets/0123456789abcdef.glb" };
    },
  ],
  ["static visual names an unlisted asset", (h) => (h.static_visuals[0].asset = "0123456789abcdef")],
  [
    "asset listed but not drawn",
    (h) => {
      const [id, entry] = asset("glTF spare prop", "prop", "Apache-2.0", "Robot Tool Tournament (Apache-2.0)", []);
      h.assets[id] = entry;
    },
  ],
  ["zero scale", (h) => (h.bodies[1].visuals[0].scale = [1, 0, 1])],
  ["geom colour out of range", (h) => (h.bodies[2].geoms[0].rgba = [2, 0, 0, 1])],
];

describe("parseReplayHeaderV2", () => {
  it("accepts the shared sample header", () => {
    const h = parseReplayHeaderV2(sampleHeader());
    expect(h.version).toBe(2);
    expect(h.bodies).toHaveLength(4);
    expect(h.bodies[0]!.geoms).toEqual([]); // geoms default to empty: no collision overlay
    expect(h.bodies[2]!.geoms).toHaveLength(1);
    expect(h.static).toEqual([]);
    expect(h.static_visuals).toHaveLength(1);
    expect(h.overlays?.channels).toEqual(["offset_mm", "seat_short_mm", "hold_s", "n_contacts"]);
    expect(h.meta.metrics).toEqual({ success: true }); // writers may add meta keys
    expect(Object.keys(h.assets)).toHaveLength(4);
  });

  it.each(MUTATIONS)("rejects: %s", (_name, mutate) => {
    const h = structuredClone(sampleHeader());
    mutate(h);
    expect(() => parseReplayHeaderV2(h)).toThrow(DataError);
  });

  it("names the JSON path that is wrong", () => {
    const h = sampleHeader();
    h.bodies[2].visuals[0].scale = [1, -1, 1];
    expect(() => parseReplayHeaderV2(h, "bout.json")).toThrow(/bout\.json: \$\.bodies\[2\]\.visuals\[0\]\.scale: .*never a mirror/);
  });

  it("allows a null geometry_id (v1 designs) and headers without optional parts", () => {
    const h = sampleHeader();
    h.meta = { design_space: "v1", design_id: "stock", geometry_id: null, part_id: "box" };
    delete h.overlays;
    delete h.static_visuals;
    const parsed = parseReplayHeaderV2(h);
    expect(parsed.meta.geometry_id).toBeNull();
    expect(parsed.overlays).toBeNull();
    expect(parsed.static_visuals).toEqual([]);
  });
});

describe("TS reader and JSON schema agree", () => {
  it("required top-level fields match the schema", () => {
    for (const key of SCHEMA.required as string[]) {
      const h = sampleHeader();
      delete h[key];
      expect(() => parseReplayHeaderV2(h), key).toThrow(DataError);
    }
  });

  it("every top-level property of the schema is accepted, and nothing else", () => {
    const allowed = Object.keys(SCHEMA.properties).sort();
    const sample = Object.keys(sampleHeader());
    // the sample uses every optional key except `static`
    expect([...sample, "static"].sort()).toEqual(allowed);
  });

  it("asset roles and overlay channels are the schema's enums", async () => {
    const { ASSET_ROLES } = await import("@/lib/assets");
    expect([...ASSET_ROLES]).toEqual(SCHEMA.$defs.asset.properties.role.enum);
    expect([...OVERLAY_CHANNELS]).toEqual(SCHEMA.$defs.overlays.properties.channels.items.enum);
  });

  it("required asset, visual, meta and provenance fields match the schema", () => {
    const cases: [string, (h: Header) => Header][] = [
      ["asset", (h) => firstAsset(h)],
      ["visual", (h) => h.bodies[0].visuals[0]],
      ["meta", (h) => h.meta],
      ["provenance", (h) => h.provenance],
    ];
    for (const [def, pick] of cases) {
      for (const key of SCHEMA.$defs[def].required as string[]) {
        const h = sampleHeader();
        delete pick(h)[key];
        expect(() => parseReplayHeaderV2(h), `${def}.${key}`).toThrow(DataError);
      }
    }
  });
});

describe("parseAnyReplayHeader", () => {
  it("still reads every committed v1 replay", () => {
    let n = 0;
    for (const dir of readdirSync(V1_DIR)) {
      for (const f of readdirSync(path.join(V1_DIR, dir)).filter((f) => f.endsWith(".json"))) {
        const h = parseAnyReplayHeader(JSON.parse(readFileSync(path.join(V1_DIR, dir, f), "utf8")), f);
        expect(h.version).toBe(1);
        expect(isV2(h)).toBe(false);
        n += 1;
      }
    }
    expect(n).toBeGreaterThan(0);
  });

  it("reads version 2 and refuses unknown versions", () => {
    expect(parseAnyReplayHeader(sampleHeader()).version).toBe(2);
    expect(() => parseAnyReplayHeader({ ...sampleHeader(), version: 3 })).toThrow(/unsupported replay version 3/);
  });
});

describe("overlays", () => {
  it("decodes IEEE half floats", () => {
    expect(halfToFloat(0x0000)).toBe(0);
    expect(halfToFloat(0x3c00)).toBe(1);
    expect(halfToFloat(0x4000)).toBe(2);
    expect(halfToFloat(0xc000)).toBe(-2);
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3);
    expect(halfToFloat(0x7bff)).toBe(65504);
    expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 9); // smallest subnormal
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(halfToFloat(0x7e00)).toBeNaN();
  });

  it("splits float16 [frames x channels] into tracks and checks the size", () => {
    const halves = new Uint16Array([0x3c00, 0x4000, 0x3800, 0x0000]); // f0: 1, 2; f1: 0.5, 0
    const header = {
      n_frames: 2,
      overlays: { bin: "x.ovl.bin", dtype: "float16", byte_order: "little", channels: ["offset_mm", "hold_s"] } as const,
    };
    const tracks = decodeOverlays(halves.buffer, { ...header, overlays: { ...header.overlays, channels: ["offset_mm", "hold_s"] } });
    expect(Array.from(tracks.offset_mm!)).toEqual([1, 0.5]);
    expect(Array.from(tracks.hold_s!)).toEqual([2, 0]);
    expect(() => decodeOverlays(new ArrayBuffer(6), { ...header, overlays: { ...header.overlays, channels: ["offset_mm"] } })).toThrow(
      ReplayFormatError,
    );
  });
});

describe("captions", () => {
  it("names engine, run and design, and flags product mode", () => {
    const h = parseReplayHeaderV2(sampleHeader());
    expect(replayCaption(h)).toBe("simulation replay · MuJoCo 3.15.0 · run pilot-a1 · design aaaaaaaaaaaa");
    h.provenance.product_mode = true;
    expect(replayCaption(h)).toMatch(/ · product mode$/);
  });

  it("falls back to the design id for v1 headers", () => {
    const v1 = parseAnyReplayHeader(JSON.parse(readFileSync(path.join(V1_DIR, "m1", "straight.json"), "utf8")));
    expect(replayCaption(v1)).toBe("simulation replay · MuJoCo · design 462150ca1475");
  });
});

describe("the committed v2 fixture", () => {
  const raw = JSON.parse(readFileSync(path.join(FIXTURE, "sample.json"), "utf8"));
  const header = parseReplayHeaderV2(raw, "sample.json");

  it("is valid and content addressed on disk", () => {
    for (const [id, a] of Object.entries(header.assets)) {
      const bytes = readFileSync(path.join(FIXTURE, a.uri));
      expect(bytes.byteLength).toBe(a.bytes);
      expect(createHash("sha256").update(bytes).digest("hex")).toBe(a.sha256);
      expect(a.sha256.startsWith(id)).toBe(true);
    }
  });

  it("keeps the v1 pose payload byte for byte", () => {
    const v1 = readFileSync(path.join(V1_DIR, "m1", "straight.bin"));
    expect(Buffer.compare(readFileSync(path.join(FIXTURE, header.bin)), v1)).toBe(0);
  });

  it("places the right jaw with its own asset and a positive scale", () => {
    const left = header.bodies.find((b) => b.name === "finger_left")!;
    const right = header.bodies.find((b) => b.name === "finger_right")!;
    expect(right.visuals[0]!.asset).not.toBe(left.visuals[0]!.asset);
    for (const b of header.bodies) for (const v of b.visuals) expect(v.scale.every((s) => s > 0)).toBe(true);
  });
});

describe("loadReplay", () => {
  afterEach(() => vi.unstubAllGlobals());

  function serve(files: Record<string, Buffer | string>) {
    const requested: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      requested.push(url);
      const body = files[url];
      if (body === undefined) return new Response("missing", { status: 404 });
      const buf = typeof body === "string" ? Buffer.from(body) : body;
      return new Response(new Uint8Array(buf));
    });
    return requested;
  }

  it("loads a v1 replay exactly as before", async () => {
    const json = readFileSync(path.join(V1_DIR, "m1", "straight.json"));
    const bin = readFileSync(path.join(V1_DIR, "m1", "straight.bin"));
    const requested = serve({ "/data/replays/m1/straight.json": json, "/data/replays/m1/straight.bin": bin });
    const r = await loadReplay("/data/replays/m1/straight");
    expect(r.version).toBe(1);
    expect(r.overlays).toBeNull();
    expect(r.bytes).toEqual({ header: json.byteLength, poses: bin.byteLength, overlays: 0 });
    expect(requested).toEqual(["/data/replays/m1/straight.json", "/data/replays/m1/straight.bin"]);
  });

  it("loads a v2 replay with its overlay sidecar", async () => {
    const files: Record<string, Buffer> = {};
    for (const f of ["sample.json", "sample.bin", "sample.ovl.bin"]) files[`/x/${f}`] = readFileSync(path.join(FIXTURE, f));
    serve(files);
    const r = await loadReplay("/x/sample");
    expect(r.version).toBe(2);
    expect(r.header.n_frames).toBe(182);
    expect(r.overlays?.offset_mm).toHaveLength(182);
    expect(r.overlayAt("offset_mm", 0)).toBeGreaterThanOrEqual(0);
    expect(r.overlayAt("seat_short_mm", 0)).toBeNull();
    // the object is lifted and held: hold time grows to at least the 2 s criterion
    expect(Math.max(...r.overlays!.hold_s!)).toBeGreaterThanOrEqual(2);
    expect(r.bytes?.overlays).toBe(182 * 2 * 2);
  });

  it("fails loudly when a declared overlay file is missing", async () => {
    serve({
      "/x/sample.json": readFileSync(path.join(FIXTURE, "sample.json")),
      "/x/sample.bin": readFileSync(path.join(FIXTURE, "sample.bin")),
    });
    await expect(loadReplay("/x/sample")).rejects.toThrow(/overlays .*HTTP 404/);
  });

  it("reports a header that is not JSON as a format error", async () => {
    serve({ "/x/bad.json": "{not json" });
    await expect(loadReplay("/x/bad")).rejects.toThrow(ReplayFormatError);
  });
});

describe("bounds with asset boxes", () => {
  it("adds the placed corners of each body's assets", () => {
    const h = parseReplayHeaderV2(sampleHeader()) as ReplayHeaderV2;
    // 2 frames, all bodies at the origin except the object, which rises 0.1 m
    const frames = new Float32Array(2 * 4 * 7);
    for (let i = 0; i < 8; i++) frames[i * 7 + 6] = 1;
    frames[(1 * 4 + 3) * 7 + 2] = 0.1;
    const replay = new Replay(h, frames);
    const unit = { min: [-0.01, -0.01, -0.01] as [number, number, number], max: [0.01, 0.01, 0.01] as [number, number, number] };
    const b = replay.bounds((i) => (i === 3 ? placedCorners(unit, [0, 0, 0.02], [0, 0, 0, 1], [1, 1, 1]) : []));
    // the object's box, raised 0.1 m in frame 1, sets the top
    expect(b.max[2]).toBeCloseTo(0.13, 6);
    // the right finger's collision box (centre z = -0.06, half height 0.01) still counts
    expect(b.min[2]).toBeCloseTo(-0.07, 6);
  });

  it("decodes the fixture's frames with the unchanged v1 decoder", () => {
    const h = parseReplayHeaderV2(JSON.parse(readFileSync(path.join(FIXTURE, "sample.json"), "utf8")));
    const buf = readFileSync(path.join(FIXTURE, h.bin));
    const frames = decodeFrames(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), h);
    expect(frames.length).toBe(h.n_frames * h.n_bodies * 7);
  });
});

describe("build-time replay check (lib/data.ts)", () => {
  /** A site tree with the v2 fixture as one lane: public/data/replays/v2/ and public/assets/. */
  function site(): { dir: string; data: string; index: BoutsIndex } {
    const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-site-"));
    const data = path.join(dir, "data");
    mkdirSync(path.join(data, "replays", "v2"), { recursive: true });
    for (const f of ["sample.json", "sample.bin", "sample.ovl.bin"]) cpSync(path.join(FIXTURE, f), path.join(data, "replays", "v2", f));
    cpSync(path.join(FIXTURE, "assets"), path.join(dir, "assets"), { recursive: true });
    const index = { bouts: [{ lanes: [{ replay: "replays/v2/sample", n_frames: 182 }] }] } as unknown as BoutsIndex;
    return { dir, data, index };
  }

  it("accepts a v2 lane whose overlays and assets are all served", () => {
    const { dir, data, index } = site();
    try {
      expect(() => checkReplayFiles(index, data)).not.toThrow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("fails the build on a missing or altered asset, or a short overlay file", () => {
    const { dir, data, index } = site();
    try {
      const glb = readdirSync(path.join(dir, "assets")).find((f) => f.endsWith(".glb"))!;
      const original = readFileSync(path.join(dir, "assets", glb));
      writeFileSync(path.join(dir, "assets", glb), Buffer.concat([original, Buffer.from([0])]));
      expect(() => checkReplayFiles(index, data)).toThrow(/does not match the size and sha256/);
      rmSync(path.join(dir, "assets", glb));
      expect(() => checkReplayFiles(index, data)).toThrow(/is missing/);
      writeFileSync(path.join(dir, "assets", glb), original);
      writeFileSync(path.join(data, "replays", "v2", "sample.ovl.bin"), Buffer.alloc(10));
      expect(() => checkReplayFiles(index, data)).toThrow(/sample\.ovl\.bin is 10 bytes/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
