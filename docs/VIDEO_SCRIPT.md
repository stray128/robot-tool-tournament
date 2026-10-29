# Effector Lab: video script (v2)

**Length 2:58** (178 s). The hackathon's hard limit is 3:00, and the cut must stay at or under 2:59
(`max_duration_s` 179 in `scripts/video/shots.json`). **The product is on screen from 0:00 to
2:51**: 171 s of the site doing its job, then a 7 s end card.

Every shot is a screen recording of the public site (or a local build of it), made by
`scripts/video/capture.mjs` from `scripts/video/shots.json`. **`shots.json` is the machine copy of
this table**: the same ids, times, routes, screens and voice-over lines. `scripts/video/shots.mjs`
holds the rules all the tools share, and `scripts/video/assemble.sh` refuses to build when a
voice-over line in `shots.json` is missing from this file, word for word.

## Rules

- **Say "simulated" plainly.** Every result is from physics simulation. No hardware results exist,
  and the last line says so.
- **Product mode stays labelled.** Every Studio shot carries the burned-in tag "Product mode: not
  part of the pre-registered experiment · simulated", and the site's own badges and fidelity tiers
  stay visible where they fit.
- **Numbers come from committed results only:** `docs/USECASE_V1_RESULTS.md`, `docs/RESULTS.md`,
  the six jobs' recorded descriptions, the atlas data and the site's published data. None is
  written for the video. A result that does not exist yet is a `{{pending: ...}}` marker.
- **Sponsor names may show; vendor brands may not.** NVIDIA, Nemotron, Nebius and Token Factory may
  appear on screen and in captions wherever they explain how the product works (founder decision,
  27 Oct 2026). Vendor brands of grippers, hands and robots stay off screen and out of the
  voice-over; they belong on the credits page only, and the site shows every third-party model
  under a generic name. `shots.mjs` scans every caption, tag, voice-over line and the end card
  against the repository's brand lists, and refuses other company and service names (such as the
  voice's provider or the code host) in anything burned in outside the end card's credit.
- **Say aloud** "Nebius Token Factory" (`s02`), "Nemotron" (`s02`, `s03`, `s11`) and "Nebius
  Serverless Jobs" (`s10`).
- **Say where each thing ran.** Nemotron 3 Super on Nebius Token Factory planned every v2 job
  live, and the recorded calls replay for free. The v2 hero searches ran locally, not on Nebius:
  never say they ran on Nebius Serverless Jobs. The first version's benchmark and the finger
  experiment did run on Nebius Serverless Jobs (job ids and costs in `docs/NEBIUS.md` and on the
  site's compute page). The hosted runner is not live, so its shot stays behind its flag.
- **Only published jobs.** The video shows the jobs the site publishes: the returns tote, the solar
  panel, the floor tile and the hotel delivery. The pipe coupling's replay failed the quality gate
  and the CNC job has no valid plan, so neither is shown or claimed. The hotel job was re-searched
  after a controller fix (its post-pull retreat had gone past the arm's travel) and re-exported, so
  its shot (`s09`) is captured from the re-exported site.
- **No pending marker in the final cut.** A marker may stand in a timing preview. The final cut
  (`assemble.sh --voiceover`) refuses any marker left in a voice-over line, a caption or the end
  card, and refuses a draft voice-over.
- **The voice is AI.** `scripts/video/voiceover.py` makes it with OpenAI text-to-speech (founder
  decision), and the end card says "Voice-over: AI text-to-speech".
- No music, so there is nothing to license. No account IDs, paths or consoles on screen; Nebius
  job IDs are public provenance and may show.

## Shots

Pace: 388 words in 178 s, about 131 words a minute of cut. The voice speaks at about 150 words a
minute, which leaves pauses between lines. Read numbers as written in the voice-over column
("thirty-two", "sixty-nine").

| Time | Route | On screen | Voice-over | Words |
|---|---|---|---|---|
| **0:00-0:13** `s01` | `/` | Home: the headline and the tote replay in the hero, then the four stages of how it works. | Robot jobs take several steps. Open a lid, put two parts in, close it. One tool rarely suits every step, so teams guess, then find out on the robot. | 29 |
| **0:13-0:26** `s02` | `/studio/new/` | Plan a job: pick the returns-tote example, press Plan, and the recorded planner calls appear one by one until the plan is valid. | Effector Lab starts with plain words. You describe the job, and Nemotron, running on Nebius Token Factory, plans it as objects, steps and pass criteria a simulator can check. | 29 |
| **0:26-0:39** `s03` | `/studio/new/` | The checks: the problems the first call sent back, then an impossible lift on step 2 turns the step red, and a sensible one makes the plan valid again. | Code checks every rule, from reach to what each step must measure. If a plan breaks one, Nemotron gets the problems back and repairs it. You can edit any step. | 30 |
| **0:39-0:53** `s04` | `/studio/h1_tote/#all-data` | Show all data on the returns tote: why candidates were ruled out before physics (among them a nut runner and a tamper pad), then the success matrix, every screened tool against every step with pass rates and ruled-out cells, scrolled slowly down to the tamper pad and the vacuum frame. | Then physics decides, in simulation. Every tool tries every step it could do, from grippers to vacuum frames, a nut runner and a tamper. Misfits are ruled out. | 28 |
| **0:53-1:08** `s05` | `/studio/h1_tote/` | The tote's answer in one sentence (a single servo gripper, 8 of 8 runs, also the best single tool) with its fidelity and 'Simulated, verify on hardware' badges, then 'Why this setup' and the comparison with the best single tool: no second tool earned its place. | The answer follows an integrator's rules. One tool is the default, and a second tool has to earn its place. For the tote, one servo gripper finished eight of eight runs, and nothing beat it. | 35 |
| **1:08-1:22** `s06` | `/studio/h5_solar/#replays` | The solar job's replay at 2x, from just before the first tool change: the 'Changing tool' card covers the change to the suction cup, the cup starts the clamp, then a second card covers the change to the nut runner. | Every replay is recorded physics that passed a quality gate. Parts are set down, and every pair of objects is checked for overlap. A tool change shows as a card. | 30 |
| **1:22-1:36** `s07` | `/studio/h5_solar/` | The solar job's answer (a tool change, vacuum area frame then suction cup then nut runner, 7 of 8 runs) and the task in plain words, then its step cards: place the panel, insert the clamp, tighten it. | An eight-kilogram solar panel goes onto its rails, then its clamp is tightened to six newton metres, a nut runner's job. A tool change, vacuum frame, suction cup, then nut runner, finished seven of eight runs. | 36 |
| **1:36-1:49** `s08` | `/studio/h6_tile/#replays` | The tile job's replay at 1x from just before the head switch: the tile set on its bed by the vacuum frame, the switch to the tamper pad on the same head, then the press that levels it. | A floor tile goes onto fresh adhesive and is tamped level, within one millimetre, at no more than sixty newtons. A two-tool head, vacuum frame plus tamper, finished eight of eight. | 31 |
| **1:49-2:02** `s09` | `/studio/h7_hotel/#replays` | The hotel job's replays at 2x: the recommended two-tool head's angular-jaw gripper pulls the door open, then the scrubber jumps to the last step, where its 25 mm suction cup places the parcel in the tray; beside it the best single tool, stopped at the button. | A hotel delivery presses an elevator call button, opens a door and places a parcel. A two-tool head, a suction cup and an angular-jaw gripper, finished eight of eight runs. | 29 |
| **2:02-2:19** `s10` | `/evidence/` | Evidence: the use-case benchmark card with its three held-out rows and the job, then the dated note on what the first version could not test. | Does it hold up? Our first version took a pre-registered test on Nebius Serverless Jobs. On thirty-two held-out conditions per job, its picks finished thirty-two, thirty-two and ten. The best single tool finished none of ninety-six. | 36 |
| **2:19-2:29** `s11` | `/results/` | The finger experiment: the S1 table and the best-so-far chart. | In a second pre-registered test, Nemotron with simulator feedback found good gripper fingers in fewer designs than random search. | 19 |
| **2:29-2:40** `s12` | `/atlas/` | The use-case atlas: 'What the research says' with the one-tool share and the bars by domain, then the browser of 95 jobs. | The use-case atlas lists ninety-five real jobs with the setup an integrator would expect, as research estimates. Sixty-nine of them need one tool. | 23 |
| **2:40-2:51** `s13` | `/docs/tutorials/run-it-yourself/` (flag off) | Run it yourself: the install commands, then the Docker commands. | It is all open source. Run the search on your own machine with one command, or in Docker. | 18 |
| **2:51-2:58** `s14` | end card | End card: the name, the links, the simulation caveat, the AI voice-over note and the credit line. | Everything here is simulated, so verify on hardware. Try it at the link below. | 14 |

The result sentences (`s05`, `s07`, `s08`) come from the site's exported data
(`data/studio/<job>.json`, from the hero searches that ran locally): the setup in words and the
runs it completed. The hotel's (`s09`) comes from its re-exported page. The CNC job's shot is gone
(no valid plan), and the hotel job took its slot. `shots.mjs check` keeps the times
honest.

## The hosted-run shot (`s13`, flag `hosted_runner`)

`s13` has two versions of the same 11 s slot, so the cut's timing never changes.

- **Flag off (the default):** the run-it-yourself guide, above.
- **Flag on** (`--flag hosted_runner` to `capture.mjs`, `voiceover.py` and `assemble.sh` alike), only
  once {{pending: hosted runner live}}:

| Time | Route | On screen | Voice-over | Words |
|---|---|---|---|---|
| **2:40-2:51** `s13` | `/studio/new/` | {{pending: hosted runner live}} Plan a job: the key field and the hosted run starting, with its daily cap shown. | It is all open source. Plan a job here with your own NVIDIA or Token Factory key, and the physics runs on our capped Nebius jobs. | 26 |

Its capture steps open `/studio/new/` and hold; add the clicks on the hosted-run control once its
markup exists.

## Captions in the final cut

With a voice-over, `assemble.sh` keeps only these short captions (`caption` in `shots.json`). Every
shot also carries its tag, top left.

| Shot | Caption |
|---|---|
| `s02` | Nemotron on Nebius Token Factory · recorded calls, replayed |
| `s04` | Every candidate × every step · simulated |
| `s05` | One tool by default · simulated, verify on hardware |
| `s06` | Recorded physics · quality-gated · simulated |
| `s07` | Tool change: vacuum frame > suction cup > nut runner · 7 of 8 runs · simulated |
| `s08` | Within 1 mm · at most 60 N · two-tool head, 8 of 8 runs · simulated |
| `s09` | Two-tool head: suction cup + angular-jaw gripper · 8 of 8 runs · simulated |
| `s10` | First version, held out: 32/32 · 32/32 · 10/32 against 0/96 for the best single tool |
| `s11` | Pre-registered: p = 0.0011 |

The silent timing preview shows the full voice-over text instead, as lower-third captions, so the
founder can read along.

## Making the cut

```bash
node scripts/video/shots.mjs check                        # the shot list's rules (markers allowed)
node scripts/video/capture.mjs --out ~/rtt-video/clips    # the live site; --serve web/out for a local build
scripts/video/assemble.sh                                 # silent timing preview with the script as captions
python scripts/video/voiceover.py --out ~/rtt-video/vo    # AI voice-over, one WAV per line plus voiceover.wav
scripts/video/assemble.sh --voiceover ~/rtt-video/vo/voiceover.wav   # the final cut
```

Add `--flag hosted_runner` to the capture, the voice-over and both assemble runs for the hosted
version. `voiceover.py --draft` speaks each pending marker as a short stand-in so the voice and the
pacing can be judged early; the final cut refuses a draft.

## Sources for every number

| Number | Source |
|---|---|
| 32/32, 32/32, 10/32; 0/96; run on Nebius Serverless Jobs | `docs/USECASE_V1_RESULTS.md` (held-out results, compute) |
| Tote: one servo gripper, eight of eight runs; nothing beat it | `data/studio/h1_tote.json`: `recommendation.label`, `completed_runs` 8 of `n_runs` 8, `best_single` the same setup, `recommendation.why` ("Ties on completion (100%) ... with the shorter cycle (78.2 s against 84.6 s), and is simpler: 1 tool against 2") |
| Solar: a tool change, vacuum frame, suction cup, then nut runner, seven of eight runs | `data/studio/h5_solar.json`: `recommendation.kind` tool_change, `recommendation.tools[].display_name`, `completed_runs` 7 of 8 |
| Tile: a two-tool head, vacuum frame plus tamper, eight of eight | `data/studio/h6_tile.json`: `recommendation.kind` multi_head, `recommendation.tools[].display_name`, `completed_runs` 8 of 8 |
| Hotel: a two-tool head, a suction cup and an angular-jaw gripper, eight of eight runs | the re-exported `data/studio/h7_hotel.json` (branch v2-final, d49eedd): `recommendation.kind` multi_head, `recommendation.tools[].display_name` (suction cup, 25 mm bellows; angular-jaw gripper of an open hobby arm), `completed_runs` 8 of `n_runs` 8, cycle 51.41 s; `best_single` 0 of 8, stops at press_button |
| The first version's wrist, the dated note | the site's Evidence page (`web/app/evidence/notes.ts`) and `docs/DEVPOST_DRAFT.md` |
| p = 0.0011, fewer designs than random search | `docs/RESULTS.md` |
| Eight kilograms, six newton metres (solar); within one millimetre, at most sixty newtons (tile) | the six jobs' recorded descriptions (Task Engineer v2 recording) and `docs/STUDIO_PRIMITIVES.md` section 13 |
| A nut runner for the clamp | integrator rule R05, `docs/STUDIO_SEARCH.md` section 2 |
| Ninety-five jobs, sixty-nine of them one tool | the atlas data (`web/public/data/atlas.json`): 69 single, 16 tool change, 10 two-tool head |
| The quality gate, set-down and all-pairs overlap | `docs/STUDIO.md` section 3.3, `docs/STUDIO_PRIMITIVES.md` sections 3 and 12, `src/rtt/studio/replay_qa.py` |
</content>
</invoke>
