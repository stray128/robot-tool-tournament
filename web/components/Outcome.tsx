import { FAILURE_EXPLANATIONS, FAILURE_LABELS } from "@/lib/labels";
import type { LaneOutcome } from "@/lib/types";

/** Pass/fail mark with the scorer's label; colour is never the only signal. */
export function OutcomeTag({ outcome, compact = false }: { outcome: LaneOutcome; compact?: boolean }) {
  if (outcome.success) {
    return (
      <span className="inline-flex items-center gap-1.5 text-pass">
        <svg viewBox="0 0 12 12" className="size-3 shrink-0" aria-hidden="true">
          <path d="M1.5 6.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
        <span>{compact ? "Held" : `Held ${outcome.hold_s.toFixed(1)} s`}</span>
      </span>
    );
  }
  const mode = outcome.failure_mode;
  return (
    <span className="inline-flex items-center gap-1.5 text-fail" title={mode ? FAILURE_EXPLANATIONS[mode] : undefined}>
      <svg viewBox="0 0 12 12" className="size-3 shrink-0" aria-hidden="true">
        <path d="M2 2l8 8M10 2l-8 8" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <span>{mode ? FAILURE_LABELS[mode] : "Failed"}</span>
    </span>
  );
}
