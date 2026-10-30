# Effector Lab: the submitted film

**Length 2:55.** A launch film first, then a demo of the website. All simulated; verify on hardware.

Every robot shot is the recorded physics of a run the Studio scored: the same tools, the same
conditions and the same outcome as the published results, re-simulated and checked against the
recorded trajectory before it was rendered. Some shots are drawn by MuJoCo itself, others are the
same recorded poses lit and rendered in Blender (only the cameras, lights and materials are new).
Nothing in a shot is animated by hand.

## Scenes and narration

| Time | Scene | Narration |
|---|---|---|
| 0:00-0:10 | Cold open: four real failures, each stamped as it happens (a three-finger hand slips off a tote lid, a small suction cup drops a carton, one cup cannot lift an 8 kg panel, a gripper cannot press a call button) | (music only) |
| 0:10-0:32 | The business problem | "Factories installed more than six hundred thousand robots last year." (source: IFR World Robotics 2026) "Every one needs a hand chosen for its job." "And the arm is often only a third of a robot cell's cost. The rest is tooling and integration, picked by experience." (source: A3, quoting an integrator, 2015) "Pick wrong, and you find out on the line." |
| 0:32-0:45 | Effector Lab: a job typed in plain words becomes a plan | "Effector Lab makes it a test. Describe the job in plain words." "NVIDIA Nemotron writes the plan. Then every tool gets its shot." |
| 0:45-1:03 | Round 1, returns tote | "Round one: open a tote, pack it, close it." "A three-finger hand slips off the lid." "A small suction cup drops the carton." "A plain two-finger gripper does the whole job." |
| 1:03-1:27 | Round 2, solar panel | "Round two: an eight-kilo solar panel, clamped and tightened." "One suction cup can't lift it." "Two tools on one head? The spare one gets in the way." "The winner changes tools. A vacuum frame lifts the panel, a small cup sets the clamp, and a nut runner tightens it." |
| 1:27-2:00 | Round 3, hotel delivery, then the winner's 8 runs laid over each other | "Round three: press the call button, open the door, deliver the parcel." "The best single tool can't press the button. A small cup drops the parcel." "So it builds a team: a hook, a gripper and a cup." "Button. Door. Parcel." "Eight runs out of eight, with the parts moved and the friction changed." |
| 2:00-2:06 | The pick | "Every job ends with one answer, why the others lost, and the parts to build it." |
| 2:06-2:50 | The website, recorded on the published build | "Here is how it works." "Type a job on the home page." "Nemotron turns it into steps, and every tool is tested on every step." "The best single tool stops at the button. The tool change finishes every run." "Watch the replay, straight from the recorded physics." "You get a build pack: parts to print, the tested scene and a bill of materials." "Ninety-five real jobs are mapped, so you can start from one like yours." |
| 2:50-2:55 | End card | "Effector Lab. Describe the job. Physics picks the hand." |

The hotel round shows the hotel result after door handles were made to be gripped from the front:
a tool change (a hook for the button, a 48 mm parallel gripper for the door, a 25 mm suction cup
for the parcel), 8 of 8 runs; the best single tool stops at the button, 0 of 8.

## How it was made

- **Physics shots:** re-simulated with the Studio's own code from the published reports; finalist
  runs reproduce their recorded trajectory hashes, and screen runs reproduce their recorded
  successes and failures. Rendered with MuJoCo's offscreen renderer, and the hero shots again in
  Blender (Cycles) from the per-frame poses.
- **The website:** recorded frame by frame in a headless browser at a fixed 30 frames per second,
  so every click, scroll and replay lands on a known frame.
- **Edit:** assembled in Remotion from one timeline file; every narration line that names
  something on screen is placed on that moment of the physics.
- **Voice:** OpenAI text-to-speech (voice "cedar").
- **Music and sound effects:** original, synthesized in code for this edit.
