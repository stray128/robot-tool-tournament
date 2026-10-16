# Gripper meshes: source and licence

These files are the simulation meshes of an open-source parallel gripper, used for the
gripper's look in renders and replays and for the "stock" finger entrant.

**Credit:** Robonine, SO-ARM 102, https://github.com/roboninecom/SO-ARM-102, CC BY 4.0.
Upstream copyright notice: Copyright (c) 2026 Robonine.

- Upstream commit: `190eb162c9e59d7f6dceab850ea1b5c6a4539c14`.
- Upstream files: `simulation/urdf/meshes/link5_1.stl`, `clamp1_1.stl` and `clamp2_1.stl`,
  plus the licence text `DOCS-LICENSE.txt`, saved here as `LICENSE-CC-BY-4.0.txt`.
- Upstream licence map (`LICENSING.md` and `REUSE.toml` at that commit): the `simulation/`
  folder (URDF and meshes) is CC BY 4.0. The fabrication CAD under `models/` is
  CERN-OHL-P-2.0 and is **not** used or redistributed here.
- `lock.json` pins every file by path, size and sha256. `python scripts/fetch_assets.py`
  downloads and verifies them; `--check` only verifies.

## What is here

| File | What it is | Changes |
|---|---|---|
| `meshes/link5_1.stl` | the gripper frame (wrist servo, frame, camera plate) | none: byte-identical copy |
| `meshes/clamp1_1.stl` | the left clamp | none: byte-identical copy |
| `meshes/clamp2_1.stl` | the right clamp | none: byte-identical copy |
| `LICENSE-CC-BY-4.0.txt` | the CC BY 4.0 licence text | none |
| `stock_blade_hulls.npz` | convex collision pieces of the two stock blades | derived (see below) |
| `lock.json` | pins for the files above | ours |

## Derived material (modified)

`rtt.render.gripper` builds everything else from the three meshes at run time, and the
derived files say so in their asset entries:

- **Re-posed:** the meshes are moved from the upstream assembly frame into the simulator's
  palm frame (rotated +90 degrees about y, the frame's front face put on the jaw root plane),
  and the right clamp is moved 1.0 mm along -y so the two jaws are symmetric about the palm
  centre.
- **Cut:** each clamp is cut at the jaw root plane into a carriage (shown sliding in the
  frame) and a blade (the stock finger).
- **Simplified:** the blades' collision geometry is a CoACD convex decomposition
  (threshold 0.02, seed 0, at most 40 pieces, coacd 1.0.14), stored in
  `stock_blade_hulls.npz` with the settings and input hashes.
- **Re-encoded:** the web and render assets are written as GLB files.

The full credit line also appears in `docs/ATTRIBUTION.md`, the README, `NOTICE` and the
site's credits page. On screen the gripper is called "an open-source parallel gripper".
