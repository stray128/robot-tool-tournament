import {
  type ExcludedRating,
  RATING_LABELS,
  SIM_STATUS_LABELS,
  SIM_STATUS_NOTES,
  type SimStatus,
  SPEC_TAG_LABELS,
  SPEC_TAG_NOTES,
  type SpecTag,
} from "@/lib/library";

const BADGE = "inline-flex items-center gap-1 rounded-full border px-2 py-px text-xs";

/** How an item is simulated: native model, generated proxy, visual only or link only (never colour alone). */
export function SimStatusBadge({ status }: { status: SimStatus }) {
  const tone =
    status === "native"
      ? "border-ink text-ink"
      : status === "proxy"
        ? "border-ink-3 text-ink-2"
        : status === "visual_only"
          ? "border-dashed border-pending text-pending"
          : "border-dashed border-rule-strong text-ink-3";
  return (
    <span className={`${BADGE} ${tone}`} title={SIM_STATUS_NOTES[status]} data-sim-status={status}>
      {SIM_STATUS_LABELS[status]}
    </span>
  );
}

/** Where a spec value comes from: a linked source, derived from the files, or our assumption. */
export function SpecTagBadge({ tag }: { tag: SpecTag }) {
  const tone = tag === "sourced" ? "border-pass text-pass" : tag === "derived" ? "border-ink-3 text-ink-2" : "border-pending text-pending";
  return (
    <span className={`${BADGE} ${tone}`} title={SPEC_TAG_NOTES[tag]} data-spec-tag={tag}>
      {SPEC_TAG_LABELS[tag]}
    </span>
  );
}

/** The licence rating of an item kept out of the library. */
export function RatingBadge({ rating }: { rating: ExcludedRating }) {
  const tone = rating === "amber" ? "border-pending text-pending" : "border-fail text-fail";
  return (
    <span className={`${BADGE} ${tone}`} data-rating={rating}>
      {rating === "red" ? (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      ) : (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M6 1.5l5 9H1z" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      )}
      {RATING_LABELS[rating]}
    </span>
  );
}

/** Whether an item takes part in the effector tournament. */
export function EntrantBadge({ entrant }: { entrant: boolean }) {
  return entrant ? (
    <span className={`${BADGE} border-accent text-accent`} data-entrant="yes">
      Entrant
    </span>
  ) : (
    <span className={`${BADGE} border-rule text-ink-3`} data-entrant="no">
      Not an entrant
    </span>
  );
}

/** A capability tag (pinch, hook, ...). */
export function CapabilityTag({ tag, note }: { tag: string; note?: string }) {
  return (
    <span className="inline-block rounded-full border border-rule bg-paper-2/60 px-2 py-px text-xs text-ink-2" title={note}>
      {tag}
    </span>
  );
}
