# Bring Your Part and the print pack

Upload a part as an STL. The tool works out how the part rests on a table. It then
searches finger designs for that part with the same search the benchmark uses, and
stress-tests the best design in simulation. The result is a zip you can print from.
The code is in `src/rtt/part/`.

> Every result here is **simulated and untested on hardware**. The certificate says so at
> the top and the bottom, and so does the pack's README.

## Run it

```bash
source .venv/bin/activate
python -m rtt.part intake assets/parts/l_bracket.stl          # what the simulator will see
python -m rtt.part run assets/parts/l_bracket.stl --out runs/l_bracket --arm nemotron --budget 16
python -m rtt.part pack runs/l_bracket                         # rebuild the zip
```

From Python:

```python
from rtt.part import intake, conditions_for_part, run_part_search, certify_part, make_print_pack
from rtt.part.search import run_part

result = run_part("part.stl", "runs/my_part", arm="nemotron", budget=24)  # all four stages
result.pack                                                                 # .../print_pack.zip
```

Exit status 2 means the part was refused. The message says why and what to change, for
example "pass units='cm'", "at most 79 mm across its narrowest side" or "check the
material".

## The four stages

| Stage | Function | What it does |
|---|---|---|
| Intake | `intake(stl_path)` -> `PartSpec` | It loads STL, OBJ, PLY or OFF files of at most 50 MB and 300k triangles, with at most 64 separate bodies and 8,000 triangles on the convex hull (the stable-pose search grows with the square of that count). It cleans and repairs the mesh (merging vertices, filling holes, fixing the winding and normals) and detects the units: mm if the largest extent is above 0.5, otherwise m, with `units=` to override. It refuses parts under 5 mm or over 0.5 m. Mass comes from the solid volume and a density or named material, and parts over 2 kg are refused. Stable poses come from `trimesh.compute_stable_poses`, with symmetric duplicates merged. The top-k poses that fit the 84 mm opening with 5 mm clearance are kept. Each kept pose gets a CoACD convex decomposition, through `rtt.sim.mesh` and in parallel worker processes, cached as `.npz`. The decomposition has a 300 s wall-time limit; past it the workers are killed and the part is refused with a suggestion (simplify it, or `decomposition="hull"`). Each pose also gets grip candidates: the jaw axis, and where along the part to grasp. |
| Conditions | `conditions_for_part(part)` -> `Suite` | Each resting pose becomes a family (`pose0`, `pose1`, ...). Each pose is crossed with its grasp points and two placements: `centred`, and `sloppy` (6 mm and 15 deg off, the benchmark's coarse-pose case). Jitter seeds derive from the file's SHA-256. The suite is capped at 12 conditions by dropping whole unlikely poses. |
| Search | `run_part_search(part, arm, budget, evaluator, llm)` | `SimEvaluator(suite)` is handed to `rtt.search.runner.run_search` unchanged. It uses the same arms, validator, budget ledger, scoring and top-1 freeze. Manual entrants that the validator rejects for this part's opening are left out of round 0. Without a client, the LLM arms get the pre-registered Nemotron client, with a spend cap ($1 by default). |
| Certificate | `certify_part(design, part)` | `rtt.sim.certify.certify` runs on each tested pose, at the grasp point the design trained best on. That gives 18 episodes per pose: placement error 0, ±1, ±2 and ±3 mm on x and y, a shake on x, y and z, and a push on x and y. The score is weighted by how often the part lands in each pose. |

Grasp candidates use the part's convex pieces, rasterised at 1.5 mm. The centre of the
bounding box is a candidate when there is material there that a fingertip can reach.
A second, off-centre point is added where the reachable cross-section under a 16 mm
finger-wide slab is at least 1.5 times larger. "Reachable" means above the controller's
lowest fingertip height, 30% of the part height, so an L bracket standing up is gripped
by its upright and not by the 4 mm base plate.

## What is in the pack

`<run_dir>/print_pack.zip` holds one folder, `print_pack/`:

| File | Contents |
|---|---|
| `finger_left.stl`, `finger_right.stl` | Bare fingers in millimetres, in the gripper's assembly frame. The root is on the frame face (z = 0) and the finger hangs toward -Z, with the contact face on y = 0. The right finger is the mirror image of the left. |
| `jaw.stl` (print 2) | The carriage block with the finger fused on, as one watertight solid (manifold3d), in millimetres. It is pre-oriented for printing, with the contact face on the bed and the rod bores vertical. The two jaws are the same part, one turned 180 degrees about the gripper's vertical axis. |
| `certificate.html` | A self-contained page with no scripts and no external files. It covers the part preview and intake facts, the design parameters and the proposer's hypothesis and analysis, the drop-weighted score, the placement-error curve (inline SVG) and every test. It also records provenance: arm, model, spend, task and suite digests, the git SHA, the job ID if any, and the frozen-design SHA-256. |
| `certificate.json`, `design.json` | The raw certificate and the frozen design record, which is hash-verified on read. |
| `manifest.json` | It lists every file with its bytes and SHA-256, its units and its orientation, plus the part and certificate summaries. It also holds the clamp-interface dimensions with their sources, provenance, the disclaimer and the licence (generated files CC0-1.0). |
| `README.txt` | Print settings, assembly notes and what to check before printing. |

The zip is deterministic: entries are sorted and timestamps fixed. Rebuilding it gives
byte-identical STLs, design and certificate JSON.

## Jaw clamp interface

The jaw reproduces the stock clamp interface of the open-source parallel gripper the
simulator models from documented dimensions. Attribution and licence (CERN-OHL-P-2.0
hardware) are in the README. It is a new parametric model built here, and no CAD file
from that project is used or shipped. `ClampInterface.describe()` labels each value, and
the labels are copied into `manifest.json`:

| Dimension | Value | Source |
|---|---|---|
| Rod bores | 2 x Ø 6.1 mm at 32 mm centres, along the closing axis | documented |
| Block footprint | 42 mm wide x 16 mm deep (the stock jaw's root) | documented |
| Stroke | 42 mm per jaw (84 mm opening) | documented |
| Pinion | module 0.8, 18 teeth (7.2 mm pitch radius) | documented |
| Block height | 18 mm | assumed |
| Rod axis height | 9 mm above the frame face | assumed |
| Drive rack | off by default. `ClampInterface(rack=True)` adds an m0.8, 20 deg rack 58 mm long at an assumed position, with clearance slots for the opposite rack and the pinion | assumed |

Check every "assumed" value against a stock jaw before printing. Without the rack, the
drive has to be transferred from a stock jaw.

### Print pack v2 (freeform jaw pairs)

`make_print_pack_v2(pair, left_stl, right_stl, out_dir, recipe=...)` packs one compiled
v2 jaw pair (`rtt.design.freeform`) for one part: `left.stl` and `right.stl` (each jaw's
own solid, root plane on the bed), `mount_left.stl` and `mount_right.stl` (each jaw fused
with a carriage block), the GLBs, `recipe.json`, `ATTRIBUTION.txt` and `manifest.json`.

- **Block width.** A v2 jaw's root block is the clamp cut face measured on the gripper's
  simulation mesh: 46 mm across, not the 42 mm of the v1 interface. `v2_clamp()` widens
  the carriage block to 46 mm (source "measured" in the manifest); the bores, their
  spacing and the block depth are unchanged.
- **One solid per file.** Each bare jaw has two blind M3 holes (3.2 mm, 6 mm deep) in its
  root block for bolting it onto a carriage. In a mount the block covers the root plane,
  which would seal those holes as voids inside the print, so they are filled: every mount
  is exactly one closed solid (`voids_filled` in the manifest says how many were filled).
  The pack refuses any file that is open, inconsistently wound, of zero volume, several
  separate solids, or that still encloses a void.

## Measured (2026-10-15, 16-core workstation shared with other jobs)

| Run | Part | Budget | Intake | Search | Certify | Pack | Total | Model spend |
|---|---|---|---|---|---|---|---|---|
| random arm, load ~10 | dumbbell handle (2 poses, 6 conditions) | 16 | 6.3 s | 21.8 s | 5.6 s | 1.7 s | **35.3 s** | $0 |
| Nemotron arm, load ~36 | L bracket (3 poses, 6 conditions) | 16 | 25.3 s | 77.0 s | 9.8 s | 3.1 s | **115.3 s** | $0.017 (4 calls) |

CoACD takes 2 to 7 s per pose, and the poses run in parallel. A cached re-intake takes
well under a second. One episode on a mesh part takes about 0.4 s of CPU, so a design on
6 conditions x 3 grasp heights is about 7 CPU-seconds. Nemotron's calls added about
37 s of latency to the second run.

## Known limits

- **Intake run time depends on the shape.** CoACD took 2 to 7 s per pose on the examples
  but about 80 s per pose on a 60 x 40 x 40 mm box with a sealed spherical cavity (1,164
  triangles), on a loaded host. The 300 s limit bounds it. A finely tessellated round
  part (a 20,480-triangle ball) is refused at once instead of spending about 100 s on
  stable poses in intake and again in every decomposition worker.
- **Training saturates on light parts.** With centred grasps most designs hold the
  example parts every time. The `sloppy` placement and the stress tests separate them,
  but the runner's tie-break among 100 % designs prefers the lighter finger, and lighter
  fingers fail the push test more often. The certificate shows this. The selection rule
  is the benchmark's and is not changed here.
- Rotationally symmetric parts (discs, bars) produce dozens of near-identical facet
  poses. These are merged by footprint, height and centre-of-mass offset to 1 mm.
- The grasp point heuristic picks the largest reachable cross-section. It knows nothing
  about friction direction or hooks, so the search and the certificate decide.
- Only the narrow jaw axis is trained by default. Use `conditions_for_part(part,
  axes="all")` to add the 90-degree grasps that intake offers.
