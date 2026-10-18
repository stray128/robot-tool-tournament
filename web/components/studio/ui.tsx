import type { ReactNode } from "react";

import { TIER_LABELS, TIER_NOTES, type Tier } from "./schema";

/** A step's or a tool's fidelity tier (docs/STUDIO.md 6): word and shape, never colour alone. */
export function TierBadge({ tier }: { tier: Tier }) {
  const tone =
    tier === "rigid" ? "border-ink-3 text-ink-2" : tier === "approximation" ? "border-pending text-pending" : "border-dashed border-fail text-fail";
  return (
    <span
      className={`num inline-flex shrink-0 items-center gap-1 border px-1.5 py-px text-[0.6875rem] uppercase tracking-wide ${tone}`}
      title={TIER_NOTES[tier]}
      data-fidelity={tier}
    >
      {tier === "approximation" ? (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M6 1.5l5 9H1z" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      ) : tier === "not_simulated" ? (
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <circle cx="6" cy="6" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 1.5" />
        </svg>
      ) : null}
      {TIER_LABELS[tier]}
    </span>
  );
}

/** "verify on hardware", next to every approximation-tier result. */
export function VerifyMark() {
  return (
    <span className="num inline-flex items-center border border-pending px-1.5 py-px text-[0.6875rem] uppercase tracking-wide text-pending" data-verify>
      verify on hardware
    </span>
  );
}

/** An [assumed] / [derived] / [sourced] tag. */
export function SpecTagBadge({ tag }: { tag: string }) {
  return (
    <span className={`num border px-1 text-[0.625rem] uppercase tracking-wide ${tag === "assumed" ? "border-pending text-pending" : "border-rule-strong text-ink-2"}`}>
      {tag}
    </span>
  );
}

/** A page section with its heading (h2) and an optional lede. */
export function Section({ id, title, lede, children }: { id: string; title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section className="mt-14 min-w-0" aria-labelledby={`${id}-title`} id={id}>
      <h2 id={`${id}-title`} className="heading text-xl">
        {title}
      </h2>
      {lede ? <div className="mt-1 max-w-[80ch] text-sm text-ink-3">{lede}</div> : null}
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** Labelled facts as a wrapping strip of cells. */
export function Facts({ items, className = "" }: { items: [string, ReactNode][]; className?: string }) {
  return (
    <dl className={`flex flex-wrap gap-px border border-rule bg-rule text-sm ${className}`}>
      {items.map(([k, v]) => (
        <div key={k} className="min-w-[8.5rem] flex-1 bg-paper px-2.5 py-1.5">
          <dt className="text-xs text-ink-3">{k}</dt>
          <dd className="num text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A success rate as a glyph: all passed, some passed, none passed. Shape plus word, not colour alone. */
export function RateGlyph({ rate }: { rate: number }) {
  const all = rate >= 1 - 1e-9;
  const none = rate <= 1e-9;
  const tone = all ? "border-pass bg-pass/15 text-pass" : none ? "border-fail text-fail" : "border-pending text-pending";
  return (
    <span className={`inline-grid size-4 shrink-0 place-items-center border ${tone}`} aria-hidden="true">
      <svg viewBox="0 0 12 12" className="size-2.5">
        <path d={all ? "M1.5 6.5l3 3 6-7" : none ? "M2 2l8 8M10 2l-8 8" : "M2 6h8"} fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </span>
  );
}
