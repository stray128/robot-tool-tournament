# Open effector library (product mode, not pre-registered)

**Product mode. Not part of the pre-registered experiment.** The library is the Studio's
supply of real, buildable end-effectors: open-source grippers, hands, suction grippers, tools
and mounting adapters whose licences allow redistribution. Each item is pinned to an upstream
commit, hashed, licence-gated and given an explicit simulation status. Items with a native
simulation model or a valid rigid proxy load as effector entrants of the existing families.
The site's catalog page (`/library/`) renders it; makers are credited on the credits page only.

Code: `src/rtt/library/`. Files: `assets/effectors/library/`. Vendoring:
`scripts/vendor_library.py`. Tests: `tests/test_library_*.py`, `web/tests/library.test.ts`.

## Contents

1. [Decisions](#1-decisions)
2. [Layout](#2-layout)
3. [The licence gate](#3-the-licence-gate)
4. [Ingest: pinned fetch, hashes, licence drift](#4-ingest-pinned-fetch-hashes-licence-drift)
5. [Simulation status and how models are made](#5-simulation-status-and-how-models-are-made)
6. [Entrants and the compile and attach gate](#6-entrants-and-the-compile-and-attach-gate)
7. [Items](#7-items)
8. [Gate results](#8-gate-results)
9. [Listed, not mirrored; excluded](#9-listed-not-mirrored-excluded)
10. [ISO 9409-1 adapter generator](#10-iso-9409-1-adapter-generator)
11. [Commands](#11-commands)
12. [Open items](#12-open-items)

## 1. Decisions

* **Founder, 2026-10-18 (design approval):** amber-licence items are **excluded** from the
  library and from print packs. The catalog page lists them, and every red item, with the
  reason ("Not included, and why").
* **Founder, 2026-10-18 (this milestone):** the simulation models of two industrial grippers
  that were withheld earlier (a two-finger adaptive linkage gripper, BSD-2-Clause, and a cobot
  arm's parallel hand, Apache-2.0) are **included**. Their makers appear only in credits:
  each item's `SOURCE.json`, `lock.json`, README, `NOTICE`, `docs/ATTRIBUTION.md` and the
  credits page. On screen they keep generic names. The contract's frozen `BRAND_TOKENS` list
  did not name these makers (they were on the private leak list), so the library keeps an
  extended list (`rtt.library.brands.LIBRARY_BRAND_TOKENS`) and
  `tests/test_library_brands.py` scans the whole repository with it.
* **Founder, 2026-10-17 (kept):** the per-part tournament's catalog group is exactly 12
  candidates. Library entrants are therefore **not default presets**; they are handed to the
  Studio's candidate search (`rtt.library.entrants.library_entrants`) or to any caller that
  builds its own candidate list.

## 2. Layout

```
assets/effectors/library/
  <key>/SOURCE.json        head (hand-written) + records (written by the vendoring script)
  <key>/LICENSES/          upstream licence and notice texts, byte for byte
  <key>/sim/               the simulation model (MJCF, hull meshes or proxy pieces)
  <key>/print/ cad/ doc/   print files, CAD and documentation, byte for byte
  catalog.json             listed (green, not mirrored) and excluded (amber, red) items
  lock.json                every pin and hash of the library, one manifest hash
web/public/data/library/
  index.json               the catalog page's data (python -m rtt.library export-web)
  glb/<key>.glb            3D previews, derived from each item's own geometry
```

Keys and screen names are generic. A `SOURCE.json` head says what the item is (kind, class,
capability tags), where it comes from (repository, full 40-hex commit, extra pins), its
licences and licence chain, its mount, its specs (each `sourced` with a link, `derived` or
`assumed`), its simulation status and what to ingest.

## 3. The licence gate

`rtt.library.licence`:

* **Green** (shipped with attribution): CC0-1.0, CC-BY-3.0, CC-BY-4.0, MIT, MIT-0, 0BSD, ISC, Apache-2.0, BSD-2-Clause, BSD-3-Clause, BSD-3-Clause-Clear, Zlib, CERN-OHL-P-2.0.
* **Amber** (review needed; excluded in this build): share-alike, weakly or strongly
  reciprocal and copyleft licences (CC-BY-SA, CERN-OHL-W/S, GPL, LGPL, MPL).
* **Red** (never shipped): any non-commercial or no-derivatives variant, no licence,
  "personal use only", platform or vendor terms that forbid redistribution, unresolved
  conflicts between two statements.
* **Chain rule:** a file is only as clean as the worst link in its derivation chain. A link is
  the file's own licence or the licence of the design its geometry was derived from. Geometry
  copying a non-commercial design is amber at best and never printable.
* **Patents and trademarks:** copyright licences cover files, not patents or marks. Makers
  stay off screen; credits name them.
* **Credit line format** (`attribution_line`): title, author, source with commit, licence with
  its URL, changes ("Modified by Effector Lab: ...").

The contract's own vendoring list (`rtt.effectors.types.ALLOWED_LICENSES`) is a subset of the
green lane. Files under the other green licences (CERN-OHL-P-2.0 clamp meshes, for example)
are credited in `SOURCE.json` but cannot be `AssetSource` records of a build, so a build notes
them instead (see open items).

## 4. Ingest: pinned fetch, hashes, licence drift

`scripts/vendor_library.py` (with `rtt.library.ingest` and `rtt.library.fetch`):

* every upstream file is fetched at its pinned full commit from the raw host, https only, at
  most 50 MB, cached by repository and commit; a re-run is offline and byte-identical (two
  fresh runs gave the same manifest);
* licence and notice texts ship byte for byte; their sha256 at the pin is in `lock.json`. When
  a re-vendor or a bump finds a different licence text, the script stops (exit 3) and names
  the file: licence drift. `--drift` compares every pinned licence text with the upstream
  default branch today;
* print, CAD and documentation files ship byte for byte unless larger than 3 MB; larger ones
  are recorded as "linked, not mirrored" with a pinned link;
* `--check` (no network) re-hashes every shipped file against `SOURCE.json` and `lock.json`,
  finds stray files, and applies the licence gate and the screen-name rules.

Size: 25.0 MB of library files and 6.1 MB of previews and
page data (31.1 MB in all; the budget is about 60 MB). Lock manifest:
`32ec3092b3c226f86912794ae905c770107fa1f261884a7bd8d74921e88282c3`.

One source class could not be fetched: a model-sharing site behind a bot check. Bypassing bot
detection is not allowed, so those items are listed with a link and not mirrored.

## 5. Simulation status and how models are made

Every item has one status (16 native, 5 proxy, 8 visual only, 0 link only):

* **native:** an upstream MJCF, or a URDF or SDF converted by us. The effector subtree is
  written as one MJCF: upstream wrist or positioning joints and marker bodies removed, every
  body's mass properties pinned at the upstream values, visual-only geometry removed, collision
  meshes replaced by their convex hull (the shape MuJoCo collides with anyway), contact margins
  and priorities set to 0 (the shared scene refuses them), and element names that carry a
  maker's name renamed. The script then checks the written file against the upstream model:
  every body's mass and relative pose, every joint's type, axis, anchor and range, every
  named actuator's gains and ranges. Converted models list what we added (for example a
  position servo where a description declares none, sized from the published grip force).
* **proxy:** a rigid body generated from the part's own mesh (CoACD threshold 0.02, seed 0;
  mass from volume x density x infill, assumed PLA at 40 % infill for printed tools), or a
  mapping onto an existing family (a vacuum gripper onto the seal-gated suction family). A
  powered tool gets a spindle body on a hinge with a velocity servo.
* **visual_only:** files and a 3D preview, no physics, with the reason (an articulated design
  without an upstream model, or compliance or magnetism not modelled yet).
* **none:** listed with a link only.

## 6. Entrants and the compile and attach gate

`rtt.library.entrants`: grippers, hands and tools load through the catalog families
(`rtt.effectors.catalog.library_models.LIBRARY_MODELS`, read by the same loaders as the catalog
models); suction cups through the suction family. Synergies are the plainest honest choice:
the gripper actuator from its open to its closed limit, every hand flexion actuator to a fixed
fraction of its range, nothing for passive tools. None was tuned on any part; library hand
cards say so. Gains and force ranges are upstream.

The gate builds every entrant spec in the shared scene on the six training parts. It passes
when no build raises `ContractError`; a part wider than an opening is a recorded refusal. When
every training part is refused, the gate still proves the attach on a 10 mm probe block. One
nominal episode per built part is recorded as information, not as a gate. Entrants:
19 specs from 17 items, all passing.

## 7. Items

29 items: generic name, licences, simulation status, capability tags, entrant.

| Key | Screen name | Licences | Simulation | Capabilities | Entrant |
|---|---|---|---|---|---|
| `adapter-iso50-4-m6-to-iso31p5-4-m5` | Adapter from ISO 50-4-M6 to ISO 31.5-4-M5 | Apache-2.0 | Native simulation model | none | no: a mounting adapter, not an effector: it carries effectors in a build pack |
| `finger-fin-tpu` | TPU fin-style compliant finger | Apache-2.0 | Visual only (no physics) | pinch | no: fin-ray compliance is not modelled yet (a pseudo-rigid chain or flex model is planned) |
| `finger-soft-codesign` | Co-designed soft finger for a cobot hand | MIT | Visual only (no physics) | pinch | no: the upstream model is a learned soft-body simulator, not a MuJoCo model; soft fingers are not modelled here yet |
| `gripper-2f-adaptive-85` | Two-finger adaptive linkage gripper, 85 mm | BSD-2-Clause | Native simulation model | pinch, power | yes (pinch) |
| `gripper-2f-air-mini` | Air-actuated mini parallel gripper | BSD-3-Clause | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-camera-holder` | Parallel gripper with a camera holder | Apache-2.0 | Visual only (no physics) | pinch | no: articulated design without an upstream simulation model |
| `gripper-2f-cobot-hand-80` | Two-finger parallel cobot hand, 80 mm | Apache-2.0 | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-force-48` | Force-controlled parallel gripper, 48 mm stroke | Apache-2.0 | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-hobby-parallel-84` | Rack-and-pinion parallel gripper for an open hobby arm | CERN-OHL-P-2.0, Apache-2.0, CC-BY-4.0 | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-linkage-cobot` | Two-finger linkage gripper of a cobot arm | BSD-3-Clause | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-rail-jaws` | Rail-guided parallel gripper with swappable jaws | MIT | Visual only (no physics) | pinch | no: articulated design without an upstream simulation model; building one needs a joint specification checked against the CAD |
| `gripper-2f-servo-110` | Industrial servo parallel gripper, 110 mm stroke | BSD-3-Clause | Native simulation model | pinch | yes (pinch) |
| `gripper-2f-teleop` | Two-finger parallel gripper, teleop-arm fingers | BSD-3-Clause, MIT | Native simulation model | pinch | yes (pinch) |
| `gripper-angular-jaw-hobby` | Angular-jaw gripper of an open hobby arm | Apache-2.0 | Native simulation model | pinch | yes (pinch) |
| `gripper-hobby-arm-twin-jaw` | Hobby-arm servo gripper with twin jaws | MIT | Visual only (no physics) | pinch | no: articulated design without an upstream simulation model |
| `hand-3f-12dof-industrial` | Three-finger 12-joint industrial hand | Apache-2.0 | Native simulation model | pinch, power | yes (power) |
| `hand-4f-8dof-printed` | Low-cost printed four-finger hand, 8 actuators | CC-BY-4.0, Apache-2.0 | Visual only (no physics) | pinch, power | no: the upstream demo model includes servo and fastener meshes of unrecorded provenance; making it an entrant needs a per-mesh review first |
| `hand-4f-servo-13act` | Servo-driven four-finger hand, 13 actuators | MIT, CERN-OHL-P-2.0 | Native simulation model | pinch, power | yes (pinch, power) |
| `hand-5f-tendon-16dof` | Tendon-driven five-finger hand, 16 joints | MIT, CC-BY-4.0 | Native simulation model | pinch, power | yes (pinch, power) |
| `jaw-compliant-tpu` | Compliant TPU moving jaw for a hobby-arm gripper | Apache-2.0 | Visual only (no physics) | pinch | no: the jaw's compliance is not modelled yet (a pseudo-rigid finger chain is planned); a rigid copy would misstate it |
| `jaw-magnetic-static` | Magnetic static jaw for a hobby-arm gripper | MIT | Visual only (no physics) | magnetic, pinch | no: magnetic holding is not modelled yet (a ferrous-gated adhesion proxy is planned) |
| `plate-iso31p5-4-m5` | Tool plate for an ISO 9409-1-31.5-4-M5 flange | Apache-2.0 | Native simulation model | none | no: a mounting adapter, not an effector: it carries effectors in a build pack |
| `plate-iso50-4-m6` | Tool plate for an ISO 9409-1-50-4-M6 flange | Apache-2.0 | Native simulation model | none | no: a mounting adapter, not an effector: it carries effectors in a build pack |
| `suction-cup-cobot` | Vacuum suction gripper, single cup | BSD-3-Clause | Generated simulation proxy | suction | yes (seal) |
| `suction-primitive` | Generic suction primitive (simulation abstraction) | MIT | Native simulation model | suction | no: its adhesion is not gated by a seal, so it would hold any face; the seal-gated suction family replaces it, and the family's 8 to 40 mm cups do not cover its 70 mm cup |
| `tool-button-pusher` | Button-pusher tool | Apache-2.0 | Generated simulation proxy | push | yes (hold) |
| `tool-hook-drawer` | Drawer-pull hook tool | Apache-2.0 | Generated simulation proxy | hook, push | yes (hold) |
| `tool-marker-holder` | Spring-loaded marker holder tool | Apache-2.0 | Generated simulation proxy | tool, push | yes (hold) |
| `tool-screwdriver-servo` | Servo screwdriver attachment for a hobby arm | MIT | Generated simulation proxy | twist, tool | yes (hold) |

## 8. Gate results

Built on the six training parts (block, steel slug, hex standoff, L-bracket, pipe tee,
flanged bushing); "held" lists the parts held in one nominal episode (product mode,
information only).

| Key | Primitive | Family | Gate | Built | Refused | Actuators | Mass kg | Reach mm | Opening mm | Held at nominal |
|---|---|---|---|---|---|---|---|---|---|---|
| `gripper-2f-adaptive-85` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.9 | 151.9 | 86.6 | block, steel_slug |
| `gripper-2f-air-mini` | pinch | catalog_gripper | pass | 0/6 (probe block only) | 6 | 1 | 0.308 | 81.1 | 18.0 | none |
| `gripper-2f-cobot-hand-80` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.76 | 138.2 | 79.8 | flanged_bushing |
| `gripper-2f-force-48` | pinch | catalog_gripper | pass | 5/6 | 1 | 1 | 0.414 | 128.8 | 47.0 | block, hex_standoff, l_bracket, flanged_bushing |
| `gripper-2f-hobby-parallel-84` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.8638 | 134.8 | n/a | none |
| `gripper-2f-linkage-cobot` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.7514 | 169.6 | 88.9 | block, steel_slug, hex_standoff, l_bracket |
| `gripper-2f-servo-110` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 1.0889 | 147.9 | 110.1 | block, steel_slug, hex_standoff, l_bracket, pipe_tee, flanged_bushing |
| `gripper-2f-teleop` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.5942 | 150.0 | n/a | none |
| `gripper-angular-jaw-hobby` | pinch | catalog_gripper | pass | 6/6 | 0 | 1 | 0.1132 | 106.4 | n/a | steel_slug, flanged_bushing |
| `hand-3f-12dof-industrial` | power | catalog_hand | pass | 6/6 | 0 | 12 | 4.1395 | 353.2 | n/a | none |
| `hand-4f-servo-13act` | power | catalog_hand | pass | 6/6 | 0 | 13 | 0.502 | 59.0 | n/a | none |
| `hand-4f-servo-13act` | pinch | catalog_hand | pass | 6/6 | 0 | 13 | 0.502 | 59.0 | n/a | none |
| `hand-5f-tendon-16dof` | power | catalog_hand | pass | 6/6 | 0 | 16 | 1.7642 | 152.5 | n/a | none |
| `hand-5f-tendon-16dof` | pinch | catalog_hand | pass | 6/6 | 0 | 16 | 1.7642 | 152.5 | n/a | none |
| `suction-cup-cobot` | seal | suction | pass | 6/6 | 0 | 1 | 0.1072 | 26.5 | n/a | none |
| `tool-button-pusher` | hold | catalog_gripper | pass | 6/6 | 0 | 0 | 0.0364 | 50.8 | n/a | none |
| `tool-hook-drawer` | hold | catalog_gripper | pass | 6/6 | 0 | 0 | 0.0011 | 5.0 | n/a | none |
| `tool-marker-holder` | hold | catalog_gripper | pass | 6/6 | 0 | 0 | 0.0183 | 33.1 | n/a | none |
| `tool-screwdriver-servo` | hold | catalog_gripper | pass | 6/6 | 0 | 1 | 0.0063 | 77.5 | n/a | none |

## 9. Listed, not mirrored; excluded

**Listed** (17): green items not ingested in this version, with the reason.

| Item | Licence | Reason |
|---|---|---|
| Universal jamming gripper (balloon and granular fill) | CC-BY-4.0 | the host blocks scripted downloads with a bot check, so the files are not mirrored: link only; jamming physics is not modelled yet |
| Hobby six-axis arm servo gripper | CC-BY-4.0 | the host blocks scripted downloads with a bot check, so the files are not mirrored: link only |
| Two-joint servo gripper and clamp | CC-BY-4.0 | the host blocks scripted downloads with a bot check, so the files are not mirrored: link only |
| Dual-material flex gripper | CC-BY-4.0 | the host blocks scripted downloads with a bot check, so the files are not mirrored: link only |
| Three-finger TPU fin-ray adaptive gripper | CC-BY-4.0 | the host blocks scripted downloads with a bot check, so the files are not mirrored: link only |
| Fin-ray soft gripper for multi-robot caging (2026 open-hardware article) | CC-BY-4.0 | the data-repository link and its licence must be verified first (the article page needs a login for automated reading) |
| Anthropomorphic research hand (2020 open-hardware article) | MIT | the CAD is on a separate data repository whose licence must be confirmed to match the code's MIT |
| Three-joint finger modules (open dynamic robot platform) | BSD-3-Clause | next tranche: simulation model to convert |
| Five-finger prosthetic-class hand, simulation model | MIT | next tranche: simulation model of a commercial hand |
| Three-finger 7-joint humanoid hand, simulation model | BSD-3-Clause | next tranche: must be cut from a humanoid model |
| Adaptive two-finger servo gripper, simulation model | Apache-2.0 | next tranche: URDF to convert |
| Compact two-finger arm gripper, simulation model | BSD-3-Clause | next tranche: URDF to convert |
| Two-finger adaptive 140 mm and three-finger adaptive grippers, simulation models | BSD-2-Clause | next tranche: URDF to convert (archived upstream) |
| Low-cost arm gripper | Apache-2.0 | next tranche: simulation model exists |
| Arm-integrated parallel grippers, simulation models (several arms) | BSD-3-Clause | next tranche: each must be cut from its arm model (MIT, BSD-3-Clause or Apache-2.0 per folder) |
| Robot-learning gripper set: generic three-finger, wiping tool, parallel | MIT | next tranche: generic simulation templates |
| Simulation-format conversions of several grippers and hands | Apache-2.0 | a format source only: the library uses the MJCF originals |

**Excluded:** 23 amber and 26 red items, each with its licence and reason, in
`assets/effectors/library/catalog.json` and on the catalog page ("Not included, and why").

## 10. ISO 9409-1 adapter generator

`rtt.library.adapters` (our own work, Apache-2.0) makes round plates: a robot face with a
centring spigot, counterbored through holes for socket head cap screws and a blind dowel hole;
a tool face with the tool's pattern (recess, blind pilot holes, dowel) or, on a blank plate, a
grid of M3 pilot holes. It is deterministic (byte-identical STL, MJCF and JSON) and reports
every dimension with its status: `sourced`, `secondary` or `assumed`. Secondary and assumed
values are listed as unverified and must be measured before use. No ISO conformance is
claimed; nothing was checked against a purchased copy of the standard.

* Pre-built for the confirmed patterns: `plate-iso50-4-m6`, `plate-iso31p5-4-m5` and
  `adapter-iso50-4-m6-to-iso31p5-4-m5` (PLA, 10 mm).
* Pitch circles and thread sizes of both confirmed patterns are sourced; every centring
  diameter comes from one secondary source (unverified); the 31.5 mm pattern's dowel size is
  assumed; hole layouts, clearance, counterbore and tap-drill sizes are assumed.
* Bolt check: a cobot maker's rule that tool-flange screws reach at most 8 mm into the flange.
  An M6 screw cannot reach 1.5 x d (9 mm) within 8 mm, so M6 plates fall back to 1.0 x d (a
  steel screw in a steel thread) and say so in their notes.

## 11. Commands

```bash
python scripts/vendor_library.py            # vendor every item at its pin (cached), rewrite lock.json
python scripts/vendor_library.py --check    # re-hash and gate, no network
python scripts/vendor_library.py --drift    # licence texts upstream today versus the pins
python -m rtt.library check                 # the same checks from the package
python -m rtt.library gate [KEY ...]        # compile and attach every entrant (and one nominal episode each)
python -m rtt.library export-web            # web/public/data/library/index.json
```

## 12. Open items

* **Contract licence list.** `ALLOWED_LICENSES` (frozen contract) lacks CERN-OHL-P-2.0 and the
  other green licences, so a build's `sources` cannot carry those files' records. A contract
  change request would add them.
* **Contract brand list.** The frozen `BRAND_TOKENS` does not name the two newly included
  makers; the library's extended list and test cover them. A contract change could merge the
  lists.
* **Tournament entry.** Library entrants are not catalog presets (the 12-candidate rule). A
  separate "library" budget group, or sampling, needs a founder decision.
* **Hands' synergies** are untuned plain flexions: the hands build and attach but hold no
  training part at nominal. Tuning them per model (never per part) is follow-up work.
* **Assumed values to verify:** the vacuum gripper's cup diameter (the description gives none:
  modelled at the suction family's largest cup, 40 mm), the hobby servo screwdriver's spindle
  speed and torque, the printed tools' infill, every unverified adapter dimension, and the
  hobby jaw servo's torque.
* **Next tranche** (section 9): convert the remaining URDF grippers and the humanoid and
  prosthetic hands, verify the open-hardware article links, and ask the authors of unlicensed
  designs to add a licence.
