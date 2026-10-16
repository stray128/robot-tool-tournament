# Real-object held-out suite (`heldout_real_v1`)

The second held-out set: household objects scanned by Google Research, scored with the
same gripper, controller and success rule as everything else. Nobody designed a finger
for these objects, and no search arm ever sees them. It is a **secondary endpoint**. The
primary endpoint is S1, speed to a good design, in `docs/TASK.md`.

- Code: `src/rtt/sim/gso.py` (fetch, screen, select, load) and
  `src/rtt/pipeline/heldout_real.py` (select, write, score).
- Rules: `configs/suites/heldout_real_v1.yaml`.
- Tests: `tests/test_gso.py` and `tests/test_gso_pipeline.py`. They run offline against a
  synthetic mirror; one live catalogue check runs with `RTT_GSO_LIVE=1`.

## Source and licences

| What | Value |
|---|---|
| Repository | `kevinzakka/mujoco_scanned_objects`, the MuJoCo conversion of Google Scanned Objects |
| Commit | `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c` (2022-07-07, the repository's last commit) |
| `models/` tree id | `f3147fe21181395ff28d8e146254e1c8c9603876`, 1,030 objects |
| Licences | MJCF (XML) files MIT; meshes and textures CC BY 4.0 |
| Paper | Downs et al., *Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items*, ICRA 2022, arXiv:2204.11918 |

**Nothing from the dataset is committed.** At run time the code fetches the object
listing once from the GitHub API and checks that it hashes to the pinned tree id. It then
fetches each examined object's `model.obj` from the commit-pinned raw URL, which is
immutable. A scan is in metres and averages 1.2 MB (143 MB for the first 120 examined). Textures and the source's own 32-piece
collision meshes are never downloaded.

Everything is cached under `$RTT_GSO_CACHE` (default `~/.cache/rtt/gso/<commit>/`), and
every fetched file's SHA-256 is recorded. `--mirror DIR` serves the same layout from a
local directory, for offline runs.

## Selection rule

The draw is a pure function of the config file, the task file, the code
(`SCREEN_VERSION = 1`), the locked library versions (MuJoCo, trimesh, CoACD, NumPy,
SciPy) and a seed. All of these except the seed go into the cache key `rules_key`.

1. **Catalogue.** The 1,030 object directories at the pinned commit, sorted by name.
2. **Category rule.** An object is dropped when any token of its name is denylisted, or
   when its full name is listed by hash.
   - Tokens are the lower-case runs of letters and digits.
   - `denylist.tokens` lists vehicles, vehicle parts and their toy models (car, truck,
     bus, engine, wheel, …). Most are wheeled, and a scan fuses the wheels to the body, so
     it cannot roll as the real object would.
   - Matching is on whole tokens, so `Cartridge` does not match `car`.
   - The rule is mechanical, so it also drops five non-vehicles whose names carry a
     vehicle word: a shoe model, two children's bags, a wooden puzzle and a game cartridge. This
     is accepted as the price of a rule fixed before the draw.
   - `denylist.name_sha256` lists six objects individually, by the SHA-256 of their full
     name, so that the names stay out of published files. The list is fixed before any
     seed is revealed. Only whole names are hashed, never words: a hash of a word is
     reversed with a word list.
   - 27 of the 1,030 objects are dropped this way: 21 by token and 6 by name. An object
     dropped by name hash is recorded by that hash only.
3. **Draw order.** A permutation of the remaining names.
   - The source is SHA-256 of a domain tag plus the seed material, fed to NumPy's PCG64.
   - The dev draw uses the public `draw.dev_seed`.
   - The scored draw uses the held-out secret. It is domain-separated from the synthetic
     held-out suite drawn from the same secret, so the one commitment already planned for
     PREREG covers both.
4. **Walk.** Objects are examined in that order. The first `n` that pass every rule below
   are accepted: `n_objects` = 64 scored, `n_dev` = 10 dev.
   - Acceptance is decided strictly in draw order, so the result does not depend on the
     number of parallel workers. A test checks this.
   - Every examined object that fails is listed in `selection.json` with its reason and
     the measured values.
5. **No object simulated during development is scored.**
   - `draw.exclude_names` lists all 16 such objects: the current dev draw (10) and the
     6 other objects of the first dev draw, made before the category rule was corrected.
     Entrants and the CoACD-vs-hull study ran on those.
   - The scored draw drops every listed name from its pool.
   - It also recomputes the current dev draw and refuses to run if any of its objects is
     missing from the list. A rule change that moves the dev draw therefore cannot let a
     smoke-tested object into the scored set.

### Per-object rules

An object's stable resting poses come from `trimesh.poses.compute_stable_poses`, most
probable first. Poses with probability below 0.05 are ignored.

- **Pose choice.** The pose used is the most probable one that passes the three fit
  rules: width, height, and length and height.
- **Other rules.** The mass, decomposition and rest rules are then checked in that pose
  only. If any of them fails, the object is rejected; no other pose is tried.

| Rule | Value | Why |
|---|---|---|
| Width across the jaw | ≤ 79 mm | The 84 mm opening (`task_v1`) minus the 5 mm margin the synthetic held-out suite uses. The width is the narrowest horizontal extent (minimum caliper width of the convex hull). The intake path turns that extent onto the closing axis. |
| Height | ≥ 6 mm | The thinnest training part, the 6 mm wire. |
| Length and height | ≤ 250 mm | About 40 % beyond the longest training part (180 mm). |
| Mass | ≤ 0.917 kg | See below. |
| Decomposition | Compiles in MuJoCo | CoACD, threshold 0.05, at most 16 pieces of at most 48 vertices, seed 0. This is the same intake path as an uploaded part (`rtt.sim.mesh`). |
| Rest test | Stays put for 2 s | See below. |

**Mass cap, derived from the grip force.**

- Two pads squeeze with F/2 each, so friction can carry at most μF.
- Jitter lowers μ by up to 10 %, so the worst case is μ_min = 0.3 × 0.9 = 0.27.
- With F = 50 N and a safety factor of 1.5 for lift accelerations and off-centre loads:
  cap = μ_min F / (g × 1.5) = 0.27 × 50 / (9.81 × 1.5) = **0.917 kg**.
- The heaviest training part is 0.91 kg.
- Mass is computed as a solid at the task's object density, 1,200 kg/m³.

  The scan has its texture seams merged, so a closed scan is watertight. Its mass and
  centre of mass then come from the solid.
- `ScreenRules.mass_cap_kg` reads μ, the jitter and F from the task, so the cap follows
  any change to them.

**Rest test** (`rest_test`).

- Setup:
  - The object is placed in its pose at the table centre, with no jitter.
  - The gripper is parked open at the approach height above it, exactly as in an
    episode's settle phase.
  - The run lasts 2 s.
- It passes when, over those 2 s, the object's centre:
  - moves at most 2 mm sideways and 2 mm vertically;
  - turns at most 2° in total;
  - ends slower than 5 mm/s.
- It must also never sink more than the task's 2 mm penetration limit into the table, and
  never touch the gripper.

### Conditions

Each accepted object becomes one condition, with a placement drawn from the same seed and
the object's own name:

- **Nominal position:** x and y uniform in ±80 mm.
- **Yaw:** uniform over a full turn.
- **Pose error unknown to the controller:** ±3 mm and ±8°.
- **Jitter seeds:** 2, plus the task's usual seed jitter on top.

These are the same ranges as the synthetic held-out suite. There is no off-centre grasp:
the palm aims at the object's bounding-box centre.

Designs are scored as follows:

- **Frozen designs:** at their frozen controller setting.
- **Manual entrants:** at their best training setting, measured with the same rule.
- **Episodes:** 64 objects × 2 seeds = 128 per design.

## Pre-registration

Hash these into PREREG before the `prereg-v1` tag:

- `sha256sum configs/suites/heldout_real_v1.yaml`
- The parsed config digest, `rtt.sim.gso.load_gso_config().digest()`, recorded in every
  selection.
- The code commit. `SCREEN_VERSION` and the screening code decide every outcome. Each
  `selection.json` also records the git SHA and lock-file hash it was screened with.
- The held-out secret's commitment, which is already planned for the synthetic suite.

The scored draw needs that secret. It runs only after every search has frozen its top-1.

- The scored `selection.json` names the held-out objects, but the search guard protects
  only the seed. So run `select --mode scored` after the freeze, and keep its output away
  from search and agent inputs until then.
- Pass `--commitment`. A report on a scored draw whose secret was not checked against the
  commitment is marked `exploratory`, as is any report run with `--limit`.
- The tool checks the seed against the value you pass, not against PREREG itself. Copy
  that value from the PREREG file.

## How to run

```bash
# Dev (exploratory) draw: public seed, 10 objects; 2-3 min on 3 workers, ~65 MB fetched
python -m rtt.pipeline.heldout_real select --mode dev --out runs/heldout_real/dev-select
python -m rtt.pipeline.heldout_real run --selection runs/heldout_real/dev-select \
    --out runs/heldout_real/dev-smoke --entrants configs/entrants.yaml --entrant-settings all

# Scored draw: after the search jobs have frozen their picks
python -m rtt.pipeline.heldout_real select --mode scored --seed-file secrets/heldout_seed.txt \
    --commitment <sha256 from PREREG> --out runs/heldout_real/scored-select
python -m rtt.pipeline.heldout_real run --selection runs/heldout_real/scored-select \
    --out runs/heldout_real/scored --frozen runs/search/* --entrants configs/entrants.yaml
```

Selection and scoring are separate steps, so the scored episodes can run as a Nebius job
without network access or CoACD.

- The `select` output directory is self-contained:
  - `selection.json` (schema `rtt.gso.selection/1`);
  - `objects/<name>.npz`, the object pack;
  - `ATTRIBUTION.md`.
- `run` verifies the selection digest and every object's geometry SHA-256 before it
  simulates. Upload the directory with the job inputs.
- `run` writes `heldout_real.json` (schema `rtt.heldout_real/1`) and `episodes.parquet`.
  The report includes, per design and per object:
  - success, failure modes and flags;
  - the penetration-flag rate;
  - the attribution list.
- The secret is never written: a test scans every output file for it.

The CoACD workers take about 3 threads and 0.5–2.4 GB of memory each, so `--workers 3`
is the default.

## Dev smoke results (exploratory, 2026-10-15, laptop)

**These are not results.** The objects are the public dev draw, which is excluded from the
scored draw. They were run once to check the pipeline end to end. Nothing was tuned on
them.

The category rule was corrected on 2026-10-15 (whole-name hashes replaced word hashes; see
the selection rule). The pool changed, so the draw changed. The numbers below are from the
corrected config. Two side studies were run on the first dev draw, before the
correction, and are labelled as such. All 16 objects of both dev draws are in
`draw.exclude_names`, so none of them can be scored.

Adding that list on 2026-10-16 changed the config digest (now `ce272cfb376a7048…`), but
not the dev draw: the list only applies to the scored draw.

### The draw

`dev_seed` 20261016 gave the following:

- **Examined:** 48 objects.
- **Accepted:** 10.
- **Excluded:** 36 too wide and 2 too long. Before the walk, the category rule removed 27
  objects.
- **Selection digest:** `f1e98c63186613b5…`, in `runs/heldout_real/dev-select-20261016-r3/`.
  It is `52bd0483452681f2…` (`dev-select-20261015-r2/`) under the config before the list;
  the objects, poses, geometry and placements are identical.
- **Wall time:** 187 s on 3 workers, under a load average of 50–58 from other jobs.
- **CoACD and rest test:** 4–86 s per accepted object.
- **Rest test:** every decomposed object passed. The largest movement was 1.0 mm and 1.0°,
  a folded washcloth standing on its edge.

A cheaper screen of the first 120 objects of the first dev draw (fit and mass only, no
CoACD) passed 28, or 23 %. It failed 86 as too wide, 4 as too long and 2 as too heavy.

- Most rejections are shoes, boxed products and kitchenware wider than 79 mm.
- The 987-object scored pool should hold about 220 eligible objects, so 64 is reachable
  after about 300 examined: roughly 360 MB of downloads and 65 CoACD runs, or about
  10–20 minutes on 3 workers.

The ten objects:

| Object | Size (mm) | Mass | Pieces |
|---|---|---|---|
| Fitness wristband, lying flat | 81 × 58 × 15 | 0.02 kg | 15 |
| Boat shoe, on its side | 246 × 78 × 90 | 0.55 kg | 16 |
| Action figure (toy) | 131 × 79 × 75 | 0.10 kg | 16 |
| Liquid-medicine bottle, on its side | 174 × 72 × 57 | 0.56 kg | 4 |
| Bird rattle (toy) | 111 × 58 × 84 | 0.10 kg | 16 |
| Skin-care cleanser pack | 148 × 59 × 81 | 0.79 kg | 2 |
| Washcloth, folded, on edge | 181 × 39 × 119 | 0.55 kg | 10 |
| Mascara carton | 129 × 51 × 44 | 0.32 kg | 1 |
| Crayon box, 24 count | 116 × 73 × 29 | 0.24 kg | 7 |
| Sponge 12-pack | 126 × 70 × 18 | 0.16 kg | 16 |

### Entrant scores

Each cell is 10 objects × 2 seeds = 20 episodes. The four entrants all score best at
k = 0.3 on training, so that column is the one a scored run would use.

| Entrant | k = 0.3 | k = 0.5 | k = 0.7 | Training score |
|---|---|---|---|---|
| straight | **80 %** | 60 % | 60 % | 69 % |
| concave | **70 %** | 65 % | 60 % | 75 % |
| hooked | **55 %** | 50 % | 40 % | 50 % |
| stock | **70 %** | 50 % | 45 % | 69 % |

- **Hooked.** 6 of its 9 failures at k = 0.3 are `missed_grasp`. It is 56.8 mm open,
  against 84 mm for the other three. It fails every episode on the 70–73 mm crayon box,
  sponge pack and medicine bottle, half of them on the 79 mm action figure, and both on the
  wristband. Yet it is the only entrant that holds the 78 mm boat shoe (2 of 2 at
  k = 0.3, no penetration flag); the other three slip, 5 of 6 times with a penetration
  flag. Why has not been examined. The shoe is 78 mm only at its widest, so the hook lip
  may catch it where it is narrower.
- **Hardest objects.** The medicine bottle on its side (4 % over its 24 episodes, all
  entrants and settings), the wristband (17 %), the boat shoe (17 %) and the action figure
  (21 %). The bottle and the shoe mostly slip during the lift; the wristband, a thin
  loop, is missed, ejected or slips.
- **Run size.** 240 episodes, plus 192 training episodes to find each entrant's setting,
  took 120 s on 12 workers, under a load average of about 55.

### Width: what this endpoint measures (decision needed before PREREG)

The width rule follows the gripper, which opens 84 mm. The design validator, however,
only guarantees an opening of 55 mm: the widest training part (50 mm) plus 5 mm. So an
object wider than 50 mm also tests whether a design opens wide enough, which training
never rewards.

The draw is dominated by such objects:

- **Dev subset:** 9 of the 10 objects are wider than 50 mm.
- **Cheap screen (first dev draw):** 22 of the 28 objects that passed are wider than
  50 mm.
- **Hooked (56.8 mm open):** at k = 0.3 it scored 100 % on the one object within the
  training width, and 50 % on the nine wider ones.

Every report therefore gives success in two bands, `success_by_width`:

- `within_training`: at most as wide as the widest training part;
- `wider`: the rest.

It also gives each design's `open_gap_mm`. Two options for PREREG:

1. **Keep 79 mm** (current file) and pre-register the band split as a planned secondary
   analysis. This tests real-object generalisation including size.
2. **Set `max_grip_width_mm: 50`**, so that every valid design opens around every object.
   - About 5 % of the pool qualifies (6 of the first 120 examined), i.e. roughly 20–100
     objects in the pool.
   - `n_objects` may have to drop to 50 or fewer.

### Decomposition: CoACD vs one convex hull (first dev draw)

The ten objects of the first dev draw, with the same poses and placements, were run with
each object as a single convex hull of its scan. That is the cheaper alternative.

| Decomposition | Mean success, all entrants and k | Episodes with a penetration flag |
|---|---|---|
| CoACD, ≤ 16 pieces (adopted) | 73.8 % | 8.8 % |
| One convex hull | 57.5 % | 17.1 % |

The single hull is not vertex-capped. Its hulls have 325–1,129 vertices on seven of these
objects, and they produced twice the contact artefacts. CoACD caps each piece at 48
vertices, which may explain the difference. CoACD stays the rule.

Penetration flags count as failures (see `docs/BENCH.md`, "Mesh objects"), so artefacts
never inflate a design. They can under-rate one: on the corrected dev draw 15.8 % of all
episodes are flagged, and the boat shoe (16 pieces) is flagged in 67 % of its episodes,
all of them with the three 84 mm-open entrants. Flag rates are reported per object and per design so
that this stays visible.

### Reproducibility

- **CoACD determinism.** Run twice on the same scan, CoACD gave the same geometry
  SHA-256.
- **Draw determinism.** The first dev draw ran three times on this laptop, twice from an
  empty cache. Each run re-downloaded the files, re-ran CoACD and re-ran the rest tests.
  All three gave the same objects, geometry digests and placements. The corrected dev draw
  was re-run from an empty cache (64 MB fetched, 258 s on 3 workers) and gave the same
  selection digest.
- **Scoring determinism.** Scoring the entrants twice gave identical success numbers.
  An independent re-run on 2026-10-16 (`dev-smoke-20261016-r3/`, 38 s on 12 workers)
  matched every design's success, per-object success and flag counts exactly.
- **Screening determinism, independent check.** The corrected dev draw was re-screened
  on 2026-10-16 from the cached scans but an empty screening cache (CoACD and rest tests
  re-run, 275 s on 3 workers under a load average of about 50). It gave the same
  selection digest, `52bd0483452681f2…`.
- **Cross-machine.** Whether CoACD is bit-identical across CPU types is not yet measured.
  The object pack and per-object geometry digests make the scored run independent of it.

## Limitations

- **Mass is solid-density.** A boxed product is modelled as solid plastic, so hollow items
  are heavier than in reality. The mass cap then excludes some bulky ones. The same density
  applies to every design, so the comparison is fair, but absolute success rates are
  conservative.
- **Friction is generic.** The task's single μ = 0.3 applies to pad and object, whatever
  the real material.
- **Contact artefacts.** Multi-piece meshes raise penetration flags in 9–16 % of
  episodes on the two dev draws (see above). They are reported per object and per design.
- **Rigid bodies only.** Soft objects (a washcloth, a shoe, a sponge pack, a rubber
  wristband) are simulated as rigid solids, so their grasps test the shape, not the
  material.
- **Object names include product brands.** For the video and site, describe objects
  generically, as the table above does. Credit stays in the attribution.
- **One pose per object.** The most probable pose that fits is used. Objects that only fit
  in a rare pose stand on edge, like the washcloth.

## Attribution

Every selection writes `ATTRIBUTION.md`, and every report carries an `attribution` list.
Per object, they record:

- the object name;
- the commit-pinned source URL;
- CC BY 4.0;
- the citation above;
- the changes made: geometry only, re-posed, and split into convex pieces for simulation.

Suggested text for `NOTICE`:

> This project uses 3D models from Google Scanned Objects (Google Research; Downs et al.,
> "Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items", ICRA
> 2022, arXiv:2204.11918), licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/), via the MuJoCo models by Kevin Zakka
> (https://github.com/kevinzakka/mujoco_scanned_objects, commit 6ff8d275, MIT). The models
> are not redistributed in this repository. They are fetched at run time, reduced to
> geometry, re-posed to a stable resting pose and split into convex pieces for simulation.
> Per-object credits are written with every selection (`ATTRIBUTION.md`).
