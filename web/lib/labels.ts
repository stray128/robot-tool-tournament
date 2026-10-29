/** Human-readable names for scorer codes, phases and design parameters. */

import type { FailureMode, ObjectSummary, PhaseSpan } from "./types";

export const FAILURE_LABELS: Record<FailureMode, string> = {
  missed_grasp: "Missed grasp",
  slip_during_lift: "Slipped during lift",
  ejected_on_close: "Pushed out on close",
  unstable_hold: "Dropped during hold",
  finger_env_collision: "Hit the table",
  not_registered: "Held off its seat",
  physics_invalid: "Physics reset",
  physics_unreliable: "Physics unreliable: overlap over 1 mm",
  invalid_build: "Jaws not buildable",
};

export const FAILURE_EXPLANATIONS: Record<FailureMode, string> = {
  missed_grasp: "The fingers never pinched the object before the lift started.",
  slip_during_lift: "The object was pinched but slid out before reaching the target height.",
  ejected_on_close: "Closing the fingers knocked the object away.",
  unstable_hold: "The object reached the target height but did not stay there for the full hold.",
  finger_env_collision: "A finger or the palm pressed into the table.",
  not_registered: "The part was lifted and held, but not seated, centred and upright in the jaws.",
  physics_invalid: "The simulator reset its state before the outcome was decided, so the episode cannot count as a hold.",
  physics_unreliable: "The jaws and the part passed into each other by more than 1 mm while it was held, so the hold cannot be trusted.",
  invalid_build: "The recipe's jaws could not be built for this part, so it scores as a failure without a simulation.",
};

export const PHASE_LABELS: Record<PhaseSpan["name"], string> = {
  settle: "Settle",
  descend: "Descend",
  close: "Close",
  lift: "Lift",
  hold: "Hold",
};

export const PARAM_LABELS: Record<string, { label: string; unit: string }> = {
  finger_length_mm: { label: "Length", unit: "mm" },
  finger_thickness_mm: { label: "Thickness", unit: "mm" },
  finger_width_mm: { label: "Width", unit: "mm" },
  bend_angle_deg: { label: "Bend angle", unit: "deg" },
  bend_start_frac: { label: "Bend position", unit: "of length" },
  inner_face: { label: "Contact face", unit: "" },
  inner_face_depth_mm: { label: "Face recess", unit: "mm" },
  tip_profile: { label: "Tip profile", unit: "" },
  tip_size_mm: { label: "Tip feature", unit: "mm" },
  taper_ratio: { label: "Taper", unit: "tip/root" },
};

/** A lane's object in words: "Box 90 x 40 x 120 mm", "Rod 6 mm x 120 mm", "Flanged bushing 36 x 36 x 35 mm". */
export function objectLabel(object: ObjectSummary): string {
  const [a = 0, b = 0, c = 0] = object.size_mm.map((v) => Number(v.toFixed(1)));
  if (object.kind === "part") return `${humanize(object.part_id ?? "part")} ${a} x ${b} x ${c} mm`;
  if (object.kind === "box") return `Box ${a} x ${b} x ${c} mm`;
  return `${object.kind === "capsule" ? "Rod" : "Cylinder"} ${a} mm x ${b} mm`;
}

export function humanize(code: string): string {
  const s = code.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function outcomeLabel(success: boolean, mode: FailureMode | null): string {
  if (success) return "Held";
  return mode ? FAILURE_LABELS[mode] : "Failed";
}

// ---------------------------------------------------------------- Studio steps

/** The Studio's step failure modes, in the order of rtt.studio.types.STEP_FAILURE_MODES. */
export const STEP_FAILURE_MODES = [
  "criterion_not_met",
  "constraint_exceeded",
  "slip",
  "jam",
  "dropped",
  "physics_invalid",
  "physics_unreliable",
  "invalid_build",
  "not_simulated",
  "not_reached",
  "out_of_reach",
] as const;
export type StepFailureMode = (typeof STEP_FAILURE_MODES)[number];

/** Each Studio step failure mode in plain words, as a short phrase for "slipped 5, jammed 3" lists. */
export const STEP_FAILURE_LABELS: Record<StepFailureMode, string> = {
  criterion_not_met: "goal not met",
  constraint_exceeded: "over the step's limit",
  slip: "slipped",
  jam: "jammed",
  dropped: "dropped",
  physics_invalid: "physics reset",
  physics_unreliable: "physics unreliable: overlap over 1 mm",
  invalid_build: "cannot be built",
  not_simulated: "not simulated",
  not_reached: "not reached",
  out_of_reach: "out of reach",
};

/** A lid meeting its stop faster than this (deg/s) gets a caveat (rtt.studio.primitives.push.LID_STOP_DPS_MAX). */
export const LID_STOP_DPS_MAX = 30;

const num = (v: number, digits = 0) => String(Number(v.toFixed(digits)));
/** A force for a plain note: whole newtons, else one decimal (24, 14.4). */
const newtons = (f: number) => num(f, 1);
/** A lid's release angle as the simulator words it: whole degrees from 10, else one decimal. */
const degrees = (d: number) => (d >= 10 ? d.toFixed(0) : d.toFixed(1));

/** The simulator's own caveats and refusals, matched by their fixed wording (rtt.studio.primitives). */
const CAVEAT_PATTERNS: readonly [RegExp, (m: RegExpExecArray) => string][] = [
  // push.lid_fall_caveat
  [/^The lid fell shut from ([0-9.]+) deg and met its stop at about ([0-9]+) deg\/s/, (m) => `lid fell shut from ${m[1]}° at about ${m[2]}°/s`],
  // _pick verdict: a part that could not be set down is not let go
  [/^The part stopped ([0-9]+) mm above (.+?)(?: \((?:it|the tool) rests on the bin's walls\))?; it was not let go/, (m) => `stopped ${m[1]} mm above ${m[2]}, not let go`],
  // push._too_stiff_verdict
  [
    /^(?:The step's force limit allows|Without a force limit on the step, the push allows) ([0-9.]+) N; this button needs at least ([0-9.]+) N to actuate/,
    (m) => `too stiff to press: needs ${m[2]} N, ${m[1]} N allowed`,
  ],
  // push._hold_too_stiff (the plan's note)
  [/^not pressed: the limit allows ([0-9.]+) N, the button actuates at ([0-9.]+) N/, (m) => `too stiff to press: needs ${m[2]} N, ${m[1]} N allowed`],
  // _common.out_of_reach_verdict
  [
    /^Out of reach: (.+?) would take the palm to x (-?[0-9]+) mm, y (-?[0-9]+) mm, past its \+-([0-9]+) mm travel/,
    (m) => `out of reach: ${m[1]} needs the palm at x ${m[2]} mm, y ${m[3]} mm (travel ±${m[4]} mm)`,
  ],
];

/**
 * A step caveat in short plain words, with the full text beside it: the simulator's run notes
 * ("lid fell shut from 21° at about 320°/s", "stopped 98 mm above the bin floor, not let go",
 * "too stiff to press: needs 24 N, 10 N allowed", "out of reach: ...") get a short label; any
 * other caveat (a tool's modelling note) is labelled by its lead-in, up to its first colon.
 */
export function caveatNote(text: string): { label: string; detail: string; kind: "run" | "model" } {
  const clean = text.trim();
  for (const [re, label] of CAVEAT_PATTERNS) {
    const m = re.exec(clean);
    if (m) return { label: label(m), detail: clean, kind: "run" };
  }
  const colon = clean.indexOf(":");
  const lead = colon > 0 && colon <= 70 ? clean.slice(0, colon) : clean.length <= 70 ? clean : `${clean.slice(0, 67).replace(/\s+\S*$/, "")}...`;
  return { label: lead, detail: clean, kind: "model" };
}

/**
 * What a recorded step's own measurements say, in short plain words (nothing for a clean step):
 * a lid that fell shut, a part held above its support, a button too stiff to press, a target out
 * of reach, unreliable physics.
 */
export function runNotes(failureMode: string | null, q: Record<string, number | null>): string[] {
  const v = (k: string): number | null => (typeof q[k] === "number" && Number.isFinite(q[k]) ? (q[k] as number) : null);
  const out: string[] = [];
  const release = v("lid_release_deg");
  const stop = v("lid_stop_dps");
  if (release !== null && release > 0 && stop !== null && stop > LID_STOP_DPS_MAX) out.push(`lid fell shut from ${degrees(release)}° at ${Math.round(stop)}°/s`);
  const high = v("held_high_mm");
  if (high !== null && failureMode) out.push(`stopped ${Math.round(high)} mm above where it was to be set down, not let go`);
  const cap = v("press_cap_n");
  const need = v("actuation_force_n");
  if (cap !== null && need !== null && failureMode === "criterion_not_met") out.push(`too stiff to press: needs ${newtons(need)} N, ${newtons(cap)} N allowed`);
  const x = v("reach_x_mm");
  const y = v("reach_y_mm");
  const lim = v("palm_limit_mm");
  if (failureMode === "out_of_reach") {
    out.push(x !== null && y !== null && lim !== null ? `out of reach: the palm would need x ${num(x)} mm, y ${num(y)} mm (travel ±${num(lim)} mm)` : "out of reach");
  }
  if (failureMode === "physics_unreliable") {
    const pen = v("penetration_mm");
    const ms = v("penetration_ms");
    out.push(`physics unreliable: ${pen !== null ? `${pen.toFixed(1)} mm` : "over 1 mm"} overlap${ms !== null ? ` for ${num(ms)} ms` : ""}`);
  }
  return out;
}
