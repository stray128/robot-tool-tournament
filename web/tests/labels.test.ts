/**
 * Plain words for the Studio's failure modes, caveats and refusals: every mode the simulator can
 * report has a label (the list is read from the Python source, never copied), the parsers refuse
 * any other mode, and the run notes and caveats read as short sentences, never as codes.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { failureLabel, failureText, parseStudioLane, plainReason } from "@/components/studio/schema";
import { failurePhrase } from "@/components/studio/story";
import { caveatNote, FAILURE_LABELS, runNotes, STEP_FAILURE_LABELS, STEP_FAILURE_MODES } from "@/lib/labels";
import { DataError } from "@/lib/validate";

const REPO = path.join(__dirname, "..", "..");

/** A tuple of strings from a Python source, as the build's brand lists are read. */
function pythonTuple(file: string, name: string): string[] {
  const src = readFileSync(path.join(REPO, file), "utf8");
  const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
  if (!block) throw new Error(`${name} not found in ${file}`);
  return [...block[1]!.matchAll(/^\s*"([^"]+)"/gm)].map((m) => m[1]!);
}

const step = { id: "press_call", primitive: "push", object: "button", handle: null, target: null, constraints: { max_force_n: 10, max_torque_nm: null, precision_mm: null, precision_deg: null } };
const plan = { objects: [{ name: "button", params: {} }], steps: [step] };

describe("failure modes", () => {
  it("are the simulator's, each with a plain label", () => {
    expect([...STEP_FAILURE_MODES]).toEqual(pythonTuple("src/rtt/studio/types.py", "STEP_FAILURE_MODES"));
    for (const mode of STEP_FAILURE_MODES) {
      const label = STEP_FAILURE_LABELS[mode];
      expect(label, mode).toBeTruthy();
      expect(label, mode).not.toMatch(/_/);
      expect(failureLabel(mode)).toBe(label);
      // the story's phrase for it is words too
      expect(failurePhrase(mode, step as never, plan as never), mode).not.toMatch(/_/);
    }
    expect(failureLabel("out_of_reach")).toBe("out of reach");
    expect(failureLabel("physics_unreliable")).toBe("physics unreliable: overlap over 1 mm");
    expect(failurePhrase("out_of_reach", step as never, plan as never)).toBe("was out of the arm's reach");
    expect(failureText({ out_of_reach: 3, slip: 5 })).toBe("slipped 5, out of reach 3");
    // a code the labels do not know is never shown as itself
    expect(failureLabel("brand_new_mode")).toBe("failed");
    // the recommender's reasons, as the finalists table shows them
    expect(plainReason("a run reset the simulator (physics_invalid)")).toBe("a run reset the simulator (physics reset)");
    expect(plainReason("stopped: out_of_reach 3x")).toBe("stopped: out of reach 3x");
  });

  it("v1 and v2 scorer modes include unreliable physics", () => {
    expect(FAILURE_LABELS.physics_unreliable).toBe("Physics unreliable: overlap over 1 mm");
  });

  it("a page with an unknown mode fails the build", () => {
    const entry = JSON.parse(readFileSync(path.join(__dirname, "fixtures", "studio_lanes", "entries.json"), "utf8")).recommended;
    entry.steps[0].success = false;
    entry.steps[0].failure_mode = "out_of_reach";
    expect(parseStudioLane(structuredClone(entry), "fixture").steps[0]!.failure_mode).toBe("out_of_reach");
    entry.steps[0].failure_mode = "teleported";
    expect(() => parseStudioLane(entry, "fixture")).toThrow(DataError);
  });
});

describe("caveats and refusals in short words", () => {
  // The simulator's wording (rtt.studio.primitives; pinned there by tests/test_web_labels_contract.py).
  const cases: [string, string, "run" | "model"][] = [
    [
      "The lid fell shut from 21 deg and met its stop at about 320 deg/s: the tool let go of it above its stop",
      "lid fell shut from 21° at about 320°/s",
      "run",
    ],
    [
      "The lid fell shut from 7.5 deg and met its stop at about 90 deg/s: the tool let go of it above its stop",
      "lid fell shut from 7.5° at about 90°/s",
      "run",
    ],
    [
      "The part stopped 98 mm above the bin floor (it rests on the bin's walls); it was not let go, since releasing it would drop it",
      "stopped 98 mm above the bin floor, not let go",
      "run",
    ],
    [
      "The part stopped 41 mm above where it was to be set down; it was not let go, since releasing it would drop it",
      "stopped 41 mm above where it was to be set down, not let go",
      "run",
    ],
    [
      "The step's force limit allows 10 N; this button needs at least 24 N to actuate, so it was not pressed",
      "too stiff to press: needs 24 N, 10 N allowed",
      "run",
    ],
    [
      "Without a force limit on the step, the push allows 14.4 N; this button needs at least 24 N to actuate, so it was not pressed",
      "too stiff to press: needs 24 N, 14.4 N allowed",
      "run",
    ],
    ["not pressed: the limit allows 10 N, the button actuates at 24 N", "too stiff to press: needs 24 N, 10 N allowed", "run"],
    [
      "Out of reach: the pick would take the palm to x 364 mm, y -12 mm, past its +-350 mm travel less 5 mm; the step was not attempted",
      "out of reach: the pick needs the palm at x 364 mm, y -12 mm (travel ±350 mm)",
      "run",
    ],
    [
      "Soft arm (approximation, octopus-inspired): a chain of rigid tapered capsule segments on spring joints",
      "Soft arm (approximation, octopus-inspired)",
      "model",
    ],
  ];
  it.each(cases)("%s", (text, label, kind) => {
    expect(caveatNote(text)).toEqual({ label, detail: text, kind });
  });

  it("reads a recorded step's own measurements", () => {
    expect(runNotes(null, { lid_release_deg: 21.3, lid_stop_dps: 317.4, angle_deg: 0 })).toEqual(["lid fell shut from 21° at 317°/s"]);
    expect(runNotes(null, { lid_release_deg: 2.3, lid_stop_dps: 57 })).toEqual(["lid fell shut from 2.3° at 57°/s"]);
    // a lid that met its stop slowly, or never fell, is no note
    expect(runNotes(null, { lid_release_deg: 2.3, lid_stop_dps: 12 })).toEqual([]);
    expect(runNotes(null, { lid_release_deg: 0, lid_stop_dps: 0 })).toEqual([]);
    expect(runNotes("criterion_not_met", { held_high_mm: 97.6 })).toEqual(["stopped 98 mm above where it was to be set down, not let go"]);
    expect(runNotes("criterion_not_met", { press_cap_n: 10, actuation_force_n: 24 })).toEqual(["too stiff to press: needs 24 N, 10 N allowed"]);
    expect(runNotes("out_of_reach", { reach_x_mm: 364.2, reach_y_mm: -12, palm_limit_mm: 350 })).toEqual(["out of reach: the palm would need x 364 mm, y -12 mm (travel ±350 mm)"]);
    expect(runNotes("out_of_reach", {})).toEqual(["out of reach"]);
    expect(runNotes("physics_unreliable", { penetration_mm: 1.84, penetration_ms: 24 })).toEqual(["physics unreliable: 1.8 mm overlap for 24 ms"]);
    expect(runNotes(null, { force_n: 3, penetration_mm: 0.1 })).toEqual([]);
  });
});
