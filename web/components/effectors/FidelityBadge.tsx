import { FIDELITY_LABELS, FIDELITY_NOTES, type Fidelity } from "@/lib/effectors";

/** How much to trust a family's physics: core, rigid model or approximation (never colour alone). */
export function FidelityBadge({ fidelity }: { fidelity: Fidelity }) {
  const tone =
    fidelity === "core"
      ? "border-ink text-ink"
      : fidelity === "rigid"
        ? "border-ink-3 text-ink-2"
        : "border-pending text-pending";
  return (
    <span
      className={`num inline-flex items-center gap-1 border px-1.5 py-px text-[0.6875rem] uppercase tracking-wide ${tone}`}
      title={FIDELITY_NOTES[fidelity]}
      data-fidelity={fidelity}
    >
      {fidelity === "approximation" ? (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M6 1.5l5 9H1z" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      ) : null}
      {FIDELITY_LABELS[fidelity]}
    </span>
  );
}
