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
