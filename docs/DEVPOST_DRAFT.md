# Devpost submission draft (v2)

Text for the Nebius x NVIDIA Global AI Hackathon form, ready to paste field by field. The founder
submits. Every number comes from `docs/USECASE_V1_RESULTS.md`, `docs/RESULTS.md`,
`docs/STUDIO.md`, `docs/STUDIO_PRIMITIVES.md`, `docs/STUDIO_SEARCH.md`, `docs/STUDIO_ENGINEER.md`,
`docs/EFFECTORS.md`, `docs/LIBRARY.md`, `docs/NEBIUS.md`, `docs/FEEDBACK.md` or the site's
published data, and matches the README.

`{{pending: ...}}` marks a result of this version that does not exist yet. Fill each one from the
named run, never by estimate, and delete its brackets. The list is at the end of this file.

## Project name

Effector Lab

## Tagline

Describe a multi-step robot job in plain words. Nemotron plans it, physics simulation tests every tool, and you get a recommended setup with replays and evidence.

## Track

Physical AI

## Links

| What | URL |
|---|---|
| Live site (static, no login, no key) | https://stray128.github.io/robot-tool-tournament/ |
| The Studio: the six jobs | https://stray128.github.io/robot-tool-tournament/studio/ |
| Plan a new job | https://stray128.github.io/robot-tool-tournament/studio/new/ |
| Use-case atlas (95 researched jobs) | https://stray128.github.io/robot-tool-tournament/atlas/ |
| Evidence | https://stray128.github.io/robot-tool-tournament/evidence/ |
| Code repository (Apache-2.0) | https://github.com/stray128/robot-tool-tournament |
| Use-case benchmark v1 results | https://github.com/stray128/robot-tool-tournament/blob/main/docs/USECASE_V1_RESULTS.md |
| Finger experiment results | https://github.com/stray128/robot-tool-tournament/blob/main/docs/RESULTS.md |
| Pre-registrations (git tags) | https://github.com/stray128/robot-tool-tournament/tree/usecase-v1 and https://github.com/stray128/robot-tool-tournament/tree/prereg-v1 |
| Platform feedback | https://github.com/stray128/robot-tool-tournament/blob/main/docs/FEEDBACK.md |
| Demo video (YouTube, under 3 minutes) | {{pending: demo video URL}} |

New during the submission period: the repository's first commit is dated 2026-10-15.

## Inspiration

Robot jobs are chains of small steps. A returns station opens a tote lid, puts a carton and a flat
pack inside, and closes the lid. A solar crew sets a panel on its rails and tightens a clamp. A
machine tender opens a guard door, swaps a finished part for a blank, and shuts the door.

One gripper rarely suits every step. So teams guess at a setup. It might be one tool, two tools on
one bracket, or a tool changer. They buy the parts, print a bracket and find out on the robot.

We wanted to move that guess into simulation, before anyone buys a part. We also wanted to know
whether the answer holds up. So we wrote our tests down and locked them before we ran them.

## What it does

The tool on the end of a robot arm is called an end effector. It plays the part of a hand: a
gripper, a suction cup, a nut runner. Effector Lab is a test bench for those hands, in physics
simulation. It works in three layers.

**1. Understand the job.** You describe the job in plain words, with any sizes and limits you know.
NVIDIA Nemotron 3 Super, served by Nebius Token Factory, plans it. Think of a work order that a
machine can check. The plan lists the objects (a lid on a hinge, a panel on rails), the steps
(pull, pick, push, twist, insert) and the pass criteria of each step (the lid opens past 95
degrees, the clamp reaches 6 N m). Code then checks every rule: layout, reach, and whether the
physics can measure each criterion. When a plan breaks a rule, Nemotron gets the exact problems
back and repairs it. You can edit any step before anything runs.

**2. Test every tool.** MuJoCo physics tries every tool it can model on every step it could do.
The pool holds grippers, hands, suction cups, vacuum frames, soft arms, a nut runner, a tamper
pad, items from an open-source library and tools that Nemotron designed. From what passes, the
search builds three kinds of setup:

- one tool for the whole job;
- a two-tool head, two tools on one bracket, like a pen with a tip at each end;
- a tool change, where the wrist swaps tools the way a drill takes a new bit.

The finalists then run the whole job under varied conditions. Each object sits up to 2 mm from
where the plan expects it, and friction varies by up to 10 %. Every setup gets the same conditions
and the same random seeds. The search can send its heavy runs to Nebius Serverless Jobs, where the
pre-registered benchmarks ran; this version's job searches ran on a local machine.

**3. Recommend a setup.** The pick follows the rules a robot integrator uses. An integrator is the
engineer who fits a robot to a job. One tool is the default. A second tool has to earn its place
by finishing clearly more runs (more than 5 points per added piece, an assumed margin). Soft arms
and hands stay off lids, doors, drawers and buttons unless no rigid tool works. A fastening at
4 N m or more needs a nut runner, and a tile press needs a tamper. The answer always sits beside
the best single off-the-shelf tool, with its reason in words.

**The website shows the evidence.** Each job has its plan, every tool's result on every step,
recorded replays of the physics, and a build pack. The pack holds the bracket or changer plates as
STL, the exact tested scene, a bill of materials, a certificate and the licences.

**The six jobs** (all simulated; verify on hardware):

| Job | Steps | Recommended setup |
|---|---|---|
| Returns tote (warehouse) | open the lid, put a carton and a flat pack in, close the lid | One tool, an industrial servo parallel gripper with a 110 mm stroke: 8 of 8 runs. It is also the best single off-the-shelf tool. |
| CNC machine tending | slide the guard door open, take out the finished part, load a blank, slide the door shut | Next: its plan failed the plan gate, so it was not searched. |
| Solar panel install | lift the 8 kg panel onto its rails, start the clamp nut, tighten it to 6 N m | A tool change, a vacuum area frame, then a 10 mm suction cup, then a nut runner: 7 of 8 runs. |
| Floor tile setting | lay a 300 mm tile on fresh adhesive, tamp it level | A two-tool head, a vacuum area frame and a 200 mm tamper pad: 8 of 8 runs. |
| Pipe coupling | push a coupling onto its thread, turn it hand-tight (about 3 N m) | Not shown: replay held back. Its replay failed the quality gate. |
| Hotel delivery (bench mock-up) | press the call button, open the door, place the parcel | A two-tool head, a 25 mm suction cup for the button and the parcel and an angular-jaw gripper for the door: 8 of 8 runs. The best single tool finishes 0 of 8. |

**The use-case atlas.** 95 jobs people do today, across 13 settings in construction; factories,
warehouses and retail; and labs, food and services. Each lists its steps and the setup an
experienced integrator would expect, with sources. These are research estimates, labelled as such.
By that research, 69 of the 95 jobs need one tool, 16 a tool change and 10 a two-tool head.

**Run it yourself.** Anyone can run the same search on their own computer, with Python or Docker,
from the run-it-yourself guide. A hosted runner, where visitors bring their own NVIDIA or Token
Factory key and the physics runs on capped Nebius jobs, comes next.

**The first version was tested in advance.** Pre-registration means we wrote the test down and
locked it in a git tag before the final runs, like a sealed exam paper. Held-out conditions are
ones no search had seen, like exam questions the student never saw. In use-case benchmark v1
(tag `usecase-v1`), each of three jobs ran 32 held-out conditions. The first version's pick
finished 32 of 32 (returns tote), 32 of 32 (pipe coupling) and 10 of 32 (equipment cabinet). The
best single off-the-shelf tool finished 0 of 96. All three exact one-sided paired tests reject
after Holm adjustment (p = 2.3e-10, 2.3e-10 and 0.00098). That version used a wrist that could
only point down. Its results stand as published, with a dated note.

A second pre-registered test sits under the tool library (tag `prereg-v1`). With 40 simulated
designs per run and 32 runs per arm, Nemotron with simulator feedback reached a good gripper finger
in 23 of 32 runs, against 11 of 32 for equal-budget random search. It needed 8.97 fewer designs on
average (95 % interval 3.53 to 14.16; one-sided exact permutation p = 0.0011).

Everything is simulated. No setup has been built or tested on hardware yet.

## How we built it

- **Physics.** MuJoCo 3.15 on CPU is the executor of record. The Studio has 12 object templates
  (ready-made models such as a hinged lid, a drawer, a threaded joint, a panel on rails, a tile on
  an adhesive bed, a spring button and a sliding machine door) and 7 primitives (basic moves:
  grasp, suction, wrap, hook pull, twist, insert, push). Each carries a fidelity tier: rigid,
  approximation ("verify on hardware") or not simulated. A fairness contract gives every candidate
  the same wrist, constants, conditions and seeds, checked by hash.
- **Nemotron on Token Factory.** The stock OpenAI SDK points at the Token Factory endpoint. The
  Task Engineer sends the task contract's own schema as one forced tool call, then one repair call
  with the exact problems. A layout fixer moves objects that overlap. Every request and response
  is cached by SHA-256, so every recorded plan replays for free, and a spend ledger caps every call.
  The prompt and the rule set carry version numbers, so the first version's recorded plans still
  replay byte for byte. A planning call in this version sends about 27,000 prompt tokens and costs
  about $0.01.
- **Nebius Serverless Jobs.** Think of renting a computer for one batch: it starts, runs and
  stops. Scored batches run on uk-south2 `cpu-d3` from one image pinned by digest in Container
  Registry. Outputs go to Object Storage mounted at `/out`, and the Token Factory key reaches a job
  from MysteryBox by reference. A job takes about 2 minutes to start, so the Studio's search sends
  one job per phase of the search, and every result row carries its job ID and its share of the
  cost. The first version's benchmark ran as one job, `aijob-e05e55n905qj9va2fh` (206 s, about
  $0.06). The finger experiment's two search arms ran as `aijob-e05yxb1awgg2d68hq1` (29 min 43 s,
  $0.523) and `aijob-e05s3f8ka6r508p4mk` (85 min 56 s, $1.512, with 473 Token Factory calls from
  inside the job for $3.569). This version's job searches ran on a local machine, not on Nebius.
- **The website.** A static Next.js site with React Three Fiber replays every recorded run from
  the simulator's pose logs, on one clock, side by side.

**What changed in this version.**

- **A wrist that tilts.** The simulated wrist gained a pitch of plus or minus 90 degrees, at most
  180 degrees per second (assumed values, the same for every tool). A two-finger gripper can now
  pinch a lid's lip from the front, and a tool can push a door or a wall button from the side.
- **Parts are set down.** A part is lowered until its support takes the load, then released. A
  release more than 20 mm above the support fails as a plain failure. A two-finger gripper carries
  a lid all the way shut.
- **Honest tolerances.** Each measured quantity has a small stated tolerance: 0.1 mm on depth,
  0.5 mm on travel and lift, 0.5 degrees on angle, 0.02 N m on torque. A part 3 mm short of its
  depth still fails.
- **All-pairs overlap checks.** Every pair of bodies counts, every 2 ms: tool, objects, table and
  walls. An overlap deeper than 1 mm for more than 10 ms marks the run unreliable. A shorter one is
  flagged on the step.
- **A replay quality gate.** Replays record at 100 Hz, with a cut at every tool change. Before a
  replay reaches the site, the exporter re-runs the physics at 500 Hz. It refuses any replay that
  strays more than 2 mm from that run, jumps, or shows an overlap the physics did not have.
- **Integrator rules.** One tool by default, rigid tools before soft arms and hands, and process
  tools for process steps, as above. The rules come from the same research as the atlas, and their
  margins are assumed.
- **New tools.** Two vacuum frames (four cups rated 4.15 kg and six cups rated 8.47 kg on a smooth
  face), a nut runner with a 13 mm socket (rated 6.7 N m, clutch at 10 N m) and a 200 mm tamper pad
  (rated 40 N, capped at 60 N). In tests, each one fails at 1.6 times its rating.
- **New objects.** A solar panel on rails with a threaded clamp (1000 x 600 x 35 mm, 8 kg), a tile
  on an adhesive bed that keeps its set, a spring button for a wall or a desk, and a machine guard
  door that slides sideways.
- **Plain failures.** A step can now fail as "out of reach", "stopped on the bin rim" or "lid fell
  shut", with the numbers, where it used to miss silently.
- **Six real jobs**, each planned by Nemotron from a plain description and never hand-edited.
  Four are on the site with a tested setup; the CNC plan failed the plan gate and the pipe
  coupling's replay failed the quality gate, so both are held back.

## Challenges we ran into

- **The first version picked a tentacle for the wrong reason.** Its simulated wrist could only
  point straight down. Pointing down, a two-finger gripper could not get a grip on the tote lid's
  lip, so only a soft arm, a tentacle-like tool that curls under the lip, could open it. The ranking put finishing first, so
  the winner was a two-tool head of a soft arm and a suction cup. It finished 32 of 32 held-out
  runs, and that result stands. Still, it said as much about our wrist as about the job. With the
  tilting wrist, a single 110 mm two-finger gripper opened the same lid in 8 of 8 development runs
  and finished the first version's whole tote job in 8 of 8 (simulated, verify on hardware).
- **Parts fell instead of landing.** The old lowering step stopped once the tool felt the part
  push back by 5 N. A suction cup's lip pushes on its part from the first moment, so the cup let go
  at carry height. In our suction test a 600 g item fell about 145 mm to the tote floor. A lid
  pushed shut fell onto its stop at 340 to 400 degrees per second. On the tote it now closes at
  about 3 degrees per second.
- **A pass that sat 6 micrometres inside its limit.** The first version's coupling counted as
  inserted at a depth of "at most 4.0 mm", the exact depth where the thread starts. The published
  pass read 3.9999936 mm. After an unrelated physics change the same run read 4.0000012 mm, which
  an exact comparison calls a failure. That is why tolerances are now stated per quantity.
- **Overlaps we did not see.** The first version checked the tool against objects only. On its
  published tote replays, the all-pairs check finds the carton 3.84 mm into the tote floor for
  12 ms, and the pack 2.94 mm into the carton for 12 ms. Both came from parts dropped from above.
- **A door that stopped 40 mm short.** On the machine-tending job, a closed fist swung 43 mm on
  contact after a pick, and the push counted as done with the door still 40 mm open. A push now
  goes on until the measured door position is met. It shuts the door to 3.9 mm at up to 40 mm/s.
  Faster pushes run out of step time and fail plainly.
- **Plans that read well but cannot run.** Nemotron's plans passed the code checks, mostly after one
  repair, but a quick physics run found flaws the checks missed. The first solar plan asked for a
  contradictory overlap. A hotel door's edge sat 350 mm out, past the wrist's safe reach. A lift on
  the 8 kg panel was shorter than the vacuum frame gives under that load. Each flaw became a new
  check and a clearer prompt, and Nemotron planned again. Plan gate for this version: 5 of 6
  plans valid, none on the first call (3 after one repair call, 2 more after the layout fix in
  code); the CNC plan failed. All of this version's planning took 61 Token Factory calls for
  $0.56, two draft recordings included.
- **Platform surprises.** In one recording run Token Factory answered HTTP 529 to 8 of 12 calls,
  and our client now retries them. One repair call ran to its 5,000-token cap with no tool call. On
  Nebius, a job takes about 2 minutes to start, and the bucket mount is not a POSIX disk.

## Accomplishments that we're proud of

- From one sentence to a tested setup, with replays that are checked against the physics before
  anyone sees them.
- Six real jobs from warehouses, factories, construction and hotels, four of them on the site with
  a tested setup, plus four new tools and four new objects.
- Two pre-registered results for the first version: the pick against the best single tool (74 of
  96 held-out runs against 0 of 96, all three tests rejected after Holm), and Nemotron with
  feedback against random search on fingers (p = 0.0011).
- An open effector library of 29 items, each pinned to an upstream commit, hashed and licence
  checked, with its makers credited.
- Honest failures with reasons, shown on the site.
- Cheap, repeatable compute. The first version's benchmark was one 206 s job for about $0.06, and
  every Nemotron answer replays from its cache for free.

## What we learned

- The test bench shapes the answer. A wrist that could not tilt made a tentacle look necessary.
  Check what the simulator cannot do before trusting a ranking.
- Most jobs need one tool. The atlas research says 69 of 95, so one tool is the default and a
  second tool has to earn its place.
- Planning is the hard part. Nemotron plans well when code checks every rule and names each
  problem. A short physics run catches what the checks miss.
- Say what is approximate. Suction seals, soft arms, threads, frames and process tools are
  labelled approximations, and every such result says "verify on hardware".
- Jobs suit batch simulation better than always-on servers. A fixed image runs, writes its results
  and stops.

## What's next for Effector Lab

- **Video in.** Film a person doing the job, and a vision-language Nemotron model turns the video
  into a plan.
- **The pre-registered v2 benchmark and an unseen-jobs test.** Held-out conditions for the six
  jobs, then 20 atlas jobs drawn by a committed secret, planned and tested blind.
- **Hardware.** Build the top setups with a partner and test them on a real arm against the
  simulated certificates.
- **Learning from the simulator.** Fine-tune an open Nemotron model on the simulator's scores, on
  Nebius GPUs, so it plans jobs and designs tools from measured outcomes.
- **Scenes for Newton and NVIDIA Isaac Lab.** A first test converts a tested scene to OpenUSD with
  the open-source mujoco-usd-converter, and it loads in Newton 1.7.0, the physics engine
  Isaac Lab 3.0 can run on. Suction and the task controllers do not carry over yet; those, and a test
  in Isaac Lab itself, come next.

## Built With

nebius-token-factory, nebius-ai-cloud, nebius-serverless-jobs, nebius-object-storage,
nebius-container-registry, nebius-mysterybox, nvidia-nemotron, nemotron-3-super,
nemotron-3.5-lightning, mujoco, nvidia-warp, mujoco-warp, python, numpy, trimesh, coacd, fastapi,
next.js, react, react-three-fiber, three.js, tailwindcss, typescript, playwright, docker,
github-pages

## Testing instructions

No key, account or install is needed.

1. Open https://stray128.github.io/robot-tool-tournament/studio/ and pick a job, for example the
   returns tote. Press play on the replays, then read the steps, the comparison with the best
   single tool and the build pack.
2. Open https://stray128.github.io/robot-tool-tournament/studio/new/, pick an example and press
   Plan. The static site replays Nemotron's recorded planning calls. Edit a step to see the checks,
   then download the task file.
3. Browse /atlas/ (95 researched jobs), /evidence/ (the pre-registered results), /compute/ (every
   published Nebius job with its time and cost) and /library/.

To run it locally (Python 3.12 and uv):

```bash
git clone https://github.com/stray128/robot-tool-tournament.git && cd robot-tool-tournament
make sync && source .venv/bin/activate
mkdir -p runs/studio
python -m rtt.studio.engineer plan --case h1_tote --out runs/studio/h1_plan.json    # replayed, $0
python -c "import json; d = json.load(open('runs/studio/h1_plan.json')); json.dump(d['plan'], open('runs/studio/h1_task.json', 'w'))"
python -m rtt.studio.search runs/studio/h1_task.json --out runs/studio/h1 --conditions 3 --finalists 2 --pack
```

Live Nemotron calls need a Token Factory key in `.env` (see `.env.example`). Nebius jobs need the
one-time setup in `docs/NEBIUS.md`.

## Feedback (Token Factory, AI Cloud, NVIDIA models)

Paste the section "Feedback for the Devpost field" of `docs/FEEDBACK.md`
(https://github.com/stray128/robot-tool-tournament/blob/main/docs/FEEDBACK.md). That file is the
single source, so the two cannot drift.

## Placeholders to fill before submitting

| Placeholder | Filled from |
|---|---|
| `{{pending: demo video URL}}` | the YouTube link after upload |
</content>
</invoke>
