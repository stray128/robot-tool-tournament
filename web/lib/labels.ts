/** Human-readable names for scorer codes, phases and design parameters. */

import type { FailureMode, PhaseSpan } from "./types";

export const FAILURE_LABELS: Record<FailureMode, string> = {
  missed_grasp: "Missed grasp",
  slip_during_lift: "Slipped during lift",
  ejected_on_close: "Pushed out on close",
  unstable_hold: "Dropped during hold",
  finger_env_collision: "Hit the table",
};

export const FAILURE_EXPLANATIONS: Record<FailureMode, string> = {
  missed_grasp: "The fingers never pinched the object before the lift started.",
  slip_during_lift: "The object was pinched but slid out before reaching the target height.",
  ejected_on_close: "Closing the fingers knocked the object away.",
  unstable_hold: "The object reached the target height but did not stay there for the full hold.",
  finger_env_collision: "A finger or the palm pressed into the table.",
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

export function humanize(code: string): string {
  const s = code.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function outcomeLabel(success: boolean, mode: FailureMode | null): string {
  if (success) return "Held";
  return mode ? FAILURE_LABELS[mode] : "Failed";
}
