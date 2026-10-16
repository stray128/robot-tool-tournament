# Asset manifest format

Every mesh that a v2 replay, the site or a print pack draws is listed once in an asset
manifest. Each entry is content-addressed and says where the file came from, its licence,
the exact credit line and what we changed. The `assets` map in a replay v2 header
(`src/rtt/sim/replay_v2.schema.json`) copies entries from the manifest verbatim, so one
entry definition serves both: `#/$defs/asset` in that schema.

Format version: 1. Owner: the render and replay code (`rtt.render`, `rtt.sim.replay`).

## Where it lives

- `web/public/assets/manifest.json` lists every served GLB.
- The GLBs sit next to it at `web/public/assets/<id>.glb` and are shared by every replay.
- Upstream source files (gripper CAD, scanned objects) are fetched at build time with
  sha256 pins. They are not committed as served assets. The `source` block of each derived
  GLB records the pin.

## File layout

```json
{
  "format": "rtt-asset-manifest",
  "version": 1,
  "assets": {
    "<id>": { "...entry..." }
  }
}
```

## Entry fields

| Field | Required | Meaning |
|---|---|---|
| `uri` | yes | `assets/<id>.glb`, relative to the site root |
| `sha256` | yes | sha256 (hex) of the file bytes |
| `bytes` | yes | file size in bytes |
| `role` | yes | `finger_left`, `finger_right`, `gripper_frame`, `carriage`, `object` or `prop` |
| `license` | yes | SPDX identifier of the asset as shipped, for example `Apache-2.0` or `CC-BY-4.0` |
| `attribution` | yes | the credit line. Our own assets: `Robot Tool Tournament (Apache-2.0)`. Third-party assets carry the short on-screen credit from `docs/ATTRIBUTION.md`: licence, "modified", and the credits-page anchor that holds the full line (for example `An open-source parallel gripper, CC BY 4.0, modified (full credit: /credits#gripper)`) |
| `changes` | yes | what we changed from the upstream file, for example `"cut at the carriage interface"`, `"simplified to 50%"`, `"re-encoded as GLB"`. Empty only for our own Apache-2.0 assets |
| `label` | no | a readable name, for example `"tee, left jaw, flat pad"` |
| `media_type` | no | `model/gltf-binary` |
| `brand_visible` | no | scanned objects only: whether a brand mark is visible. Hero frames use `false` only |
| `source` | no | provenance: `{"kind": "upstream", "url", "commit", "path", "sha256"}` for a pinned upstream file (`url` and `sha256` required), or `{"kind": "generated", "generator"}` for our own output |

## Rules

1. **Content addressing.** The id is the first 16 hex characters of `sha256`. `uri` is
   `assets/<id>.glb` and `bytes` is the file size. A changed file is a new id; ids are never
   reused.
2. **Frames.** A GLB is authored in the frame of the body it is attached to: Z up, metres.
   Jaw GLBs use the jaw frame of `rtt.design.freeform.types`. Props and objects use their
   object frame.
3. **Right jaws are their own assets.** A replay places assets with a positive `scale` only
   (the schema enforces it), so a right jaw is never drawn as a mirrored left jaw.
4. **Encoding.** `rtt.render.glb` writes deterministic GLBs with quantised attributes
   (`KHR_mesh_quantization`) and the credit in `asset.copyright`. The web export
   (`rtt export-web --replay-version 2`, `rtt.render.compress`) serves them
   meshopt-compressed by `web/scripts/compress_glb.mjs`, with WebP textures and scans
   simplified to 50%; the id is always the hash of the served file, and the published
   replay headers point at the served ids. The compressor keeps `asset.copyright` and
   `asset.extras`, and the export checks it did. Run directories keep the uncompressed
   GLBs, which the offline renderers (MuJoCo studio, Blender) read. A compressing store
   keeps the uncompressed GLB under its `raw_root` with the same id.
5. **Licences.** Every third-party asset (any licence other than our own Apache-2.0) needs
   a non-empty `changes` list and its `attribution` line (the schema enforces it). Its
   `attribution` is the short credit, which links to the full credit line on the credits
   page and in `docs/ATTRIBUTION.md` (CC BY 4.0 allows credit by link to the full notice).
   The gripper maker is named only there, in `NOTICE`, the README and
   `assets/gripper/so_arm_102/SOURCE.md`.
6. **Privacy.** `uri`, `source.path` and `label` never contain local paths, account or
   resource identifiers. `scripts/leak_check.sh` runs over the manifest like any other file.
7. **Replays.** A replay's `assets` map holds exactly the entries its `bodies[].visuals` and
   `static_visuals` reference, copied from the manifest. The scored simulation never reads
   an asset: the scored model's hash is the same with or without them.

## Example

```json
{
  "format": "rtt-asset-manifest",
  "version": 1,
  "assets": {
    "3b1f0c9a2d4e5f60": {
      "uri": "assets/3b1f0c9a2d4e5f60.glb",
      "sha256": "3b1f0c9a2d4e5f6071829a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d",
      "bytes": 11264,
      "role": "finger_left",
      "license": "Apache-2.0",
      "attribution": "Robot Tool Tournament (Apache-2.0)",
      "changes": [],
      "label": "pipe tee, left jaw",
      "media_type": "model/gltf-binary",
      "source": {"kind": "generated", "generator": "rtt.render.glb"}
    },
    "9c8d7e6f5a4b3c2d": {
      "uri": "assets/9c8d7e6f5a4b3c2d.glb",
      "sha256": "9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d",
      "bytes": 48210,
      "role": "gripper_frame",
      "license": "CC-BY-4.0",
      "attribution": "An open-source parallel gripper, CC BY 4.0, modified (full credit: /credits#gripper)",
      "changes": ["cut at the carriage interface", "re-posed", "simplified", "re-encoded as GLB"],
      "media_type": "model/gltf-binary",
      "source": {
        "kind": "upstream",
        "url": "https://example.org/upstream/repository",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "path": "simulation/meshes/frame.stl",
        "sha256": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
      }
    }
  }
}
```

The example's hashes are placeholders. Real credit lines live in `docs/ATTRIBUTION.md`.
