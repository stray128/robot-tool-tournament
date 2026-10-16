# Video script (first draft)

**Hard limit 3:00. Target 2:50.** The voice-over is about 340 words, about 2:15 at 150
words a minute, which leaves room for on-screen pauses.
Every shot is a real recording or a real replay. Numbers in brackets are filled in from
the recorded runs; none of them may be written from this script.

## Rules

- **Real recordings only.** Replays are rendered from recorded runs, never re-simulated
  and never staged. Stored runs are labelled "recorded run". Calibration and pilot bouts
  are labelled "pilot".
- **Burned-in caption on every replay:** "simulation replay · MuJoCo · run <id> ·
  design <id>". Offline renders add "rendered offline". Product-mode clips add
  "product mode".
- **Showcase parts** are the heroes picked by the pre-registered hero rule
  (`docs/PREREG.md` section A8). They are labelled "showcase parts chosen because the
  stock finger fails". If fewer than 3 parts qualify, the hook uses the imprint-vs-designed
  offset contrast only (variant B below).
- **On screen, the gripper is "an open-source parallel gripper".** Its name and credits
  appear in the README and the site's credits page only.
- **No third-party brands or logos.** Scanned objects appear only if the brand screen
  marks them `brand_visible: false`; props are our own industrial parts.
- **No cloud account details.** No other cloud consoles. Nebius job IDs may show (they are public
  provenance); project, subnet, bucket and registry IDs may not. Prefer the CLI log view
  and crop the console.
- **Say aloud:** "Nebius Token Factory", "Nemotron" and "Serverless Jobs".
- Royalty-free music or none.
- **Claims** are the fixed wording in `docs/PREREG.md` section A13: simulated, at
  friction 0.3 and 25 N per jaw. Round parts that do not jam are said not to jam.

## Timeline

| Time | Shot | Voice-over | Words |
|---|---|---|---|
| 0:00-0:10 | **Hook** (see below) | "Same part, placed two millimetres off. The stock finger loses it. A shape-matched imprint jams. The finger Nemotron designed pulls it to centre, and holds." | 25 |
| 0:10-0:30 | **Problem.** The CAD, print and test loop, as a graphic with sourced figures [verify each source before recording: robot installations, machined vs printed finger cost and lead time]. Then the imprint idea: the part's outline carved into a jaw. | "Every new part on a robot cell needs custom fingers. The usual shortcut carves the part's outline into the jaw, prints it and tests it. That imprint fits perfectly, until the part arrives two millimetres off. We test finger designs in physics simulation before the first print." | 47 |
| 0:30-0:45 | **Product, labelled "product mode".** Upload a part, the Grasp Engineer agent inspects it, proposes designs and launches jobs, then the print pack downloads. Close-up: the conical locator pin seating in the tee's socket, x-ray view. | "Bring your own part. The Grasp Engineer, a Nemotron agent, reads its shape, designs jaws for it, tests them under placement error and hands you printable files." | 27 |
| 0:45-1:55 | **One live round, 70 s continuous** (modules 1-6 below) | Module lines below | 111 |
| 1:55-2:20 | **Pre-registered results.** Run-level dots per arm (Nemotron with feedback vs equal-budget random), the reach curves, the p-value and the effect interval; registered and lift-only side by side; held-out parts including the unseen elbow and gear; a counter: "[N] jobs, [X] simulated grasps, [Y] CPU-hours, $[Z]". Failure clips included. | Results line below (one of three, chosen by the result) | about 40 |
| 2:20-2:38 | **How it is built.** Architecture diagram: Token Factory, Serverless Jobs on CPU, Object Storage, the replay site. "Apache-2.0", `make demo`, the STL download. | "Every number traces to a job. Jobs, not endpoints, because each batch is a fixed, pinned image that runs, writes its results and stops. The code is Apache-2.0, and `make demo` replays every bout on your laptop." | 38 |
| 2:38-2:52 | **Limits and roadmap.** Text: "simulation only · rigid fingers · friction assumed". Then a roadmap card, labelled "roadmap, not built": "fine-tune an open Nemotron on simulator rewards (GRPO) · Nebius GPUs · Serverless Endpoints · fingers → attachments → other tasks". | "This is simulation, with rigid fingers and assumed friction; printed tests come next. Then we fine-tune an open Nemotron model with reinforcement learning on simulator rewards, on Nebius GPUs, served from Serverless Endpoints, so it designs end-effectors for a task. Fingers first, then general attachments, then other tasks." | 48 |
| 2:52-2:58 | **End card** (ends by 2:58, 2 s inside the limit). Repo URL; "Built with Nebius Token Factory, Nebius Serverless Jobs, NVIDIA Nemotron, MuJoCo". | (silent) | 0 |

Total: about 340 words.

### 0:00-0:10 Hook

Replaces 0:00-0:12 of the build-plan storyboard. One part, one error, three fingers.

| Time | Shot |
|---|---|
| 0:00-0:03 | Macro, studio light. The open-source parallel gripper with orange designed jaws closes on the pipe tee, placed +2 mm off centre (ghost outline at the true centre). The tee slides to centre. Readout from the overlay channel: "[2.0] → [0.0x] mm". Lift. |
| 0:03-0:06 | Triptych at the same +2 mm: **stock** blade, knocked out (red tag); **configurator imprint**, stopped on the branch, "[5] mm short" (amber tag); **designed**, centred (green tag). All three are the same part, the same placement and the same controller. |
| 0:06-0:09 | "Same recipe, six parts": the compiled jaws for each training part on a turntable, 0.5 s each, no morphing. |
| 0:09-0:10 | Title card: "Fingers designed by NVIDIA Nemotron. Tested in simulation on Nebius." |

**Variant B** (fewer than 3 heroes): 0:00-0:06 becomes two panels, imprint vs designed,
with the offset readouts only. The voice-over drops "The stock finger loses it."

### 0:45-1:55 One live round (70 s continuous)

One take of a real bout, recorded; captions on. If it is a stored run, it is labelled
"recorded run".

| Module | Shot | Voice-over |
|---|---|---|
| 1 | Nemotron 3 Super streams its round analysis and four recipes; provenance chip with model ID, tokens and dollars. | "Nemotron on Nebius Token Factory reads the last round's results and proposes four new recipes, each with a hypothesis." |
| 2 | The validator rejects one recipe with a red reason (for example, pads closer than 1 mm at the seat); the repair call fixes it. | "Every recipe is checked before it costs any simulation. This one fails, and Nemotron repairs it." |
| 3 | One recipe compiles into a left and a right jaw for each of the six parts; x-ray view of a pad around the tee's branch. | "A recipe is a strategy, not a shape. It compiles into its own pair of jaws for every part." |
| 4 | `nebius ai job logs --follow`: the job ID, `cpu-d3`, uk-south2, episodes ticking. | "Nebius Serverless Jobs run the physics: forty-eight grasps per design, every part placed off centre, on CPU." |
| 5 | Replays arrive: close, the registration readout (offset and seat shortfall), lift, two-second hold timer. | "A grasp only counts if the part is seated and centred before the lift, and still held two seconds later." |
| 6 | Round 2's analysis cites round 1's counts ("[k] jams on close") and proposes drafted walls. | "Measured results feed the next round. Here it sees its imprints jamming and adds draft to pull parts to centre." |

Module words: about 110, spoken over 70 s with on-screen pauses.

### 1:55-2:20 Results line (choose by the result; never soften it)

- **Win:** "Pre-registered before any scored run: Nemotron with feedback reached the target in [a] designs on average, random search in [b]. One-sided p = [p]."
- **Tie:** "Pre-registered before any scored run: Nemotron and random search needed about the same number of designs, [a] and [b]. That is a tie, and we report it."
- **Loss:** "Pre-registered before any scored run: random search got there first, in [b] designs against Nemotron's [a]. We report that as it is."

Then, for all three: "On parts it never saw, including an elbow and a gear, its designs registered [h] percent of grasps."

## Shot sources

- Hook and module 5: MuJoCo studio renders from the recorded replays (wide, close, side,
  x-ray cameras). Up to 6 hook shots may be re-rendered offline (Oct 24-26), labelled
  "rendered offline"; the fallback is the MuJoCo studio render.
- Module 3: the web design page (x-ray and turntable views) from the exported GLBs.
- Module 4: a terminal recording of the job log, cropped.
- Results: figures rebuilt from the released Parquet tables (`docs/PREREG.md` section A10).
- Shot list and overlay rendering: `scripts/video/shots.yaml`, `scripts/video/overlay.py`.

## Built With tags

nebius-token-factory, nebius-ai-cloud, nebius-serverless-jobs, nebius-object-storage,
nvidia-nemotron, mujoco, next.js, react-three-fiber, three.js, fastapi. Add nvidia-warp
and mujoco-warp only if the GPU benchmark appears.

## Open items

- Source each figure in 0:10-0:30 again before recording (the build plan's sources were
  checked on 2026-10-15).
- The hook numbers come from the hero screen and the overlay channel, never from this
  script.
- If F1 runs (v1 protocol), the results section shows the v1 experiment and the v2
  fingers appear only in product-mode segments, labelled.
