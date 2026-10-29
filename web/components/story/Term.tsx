"use client";

import { type ReactNode, useEffect, useId, useRef, useState } from "react";

/**
 * A term with a "?" tip: hover, focus or tap shows a short definition; Escape or a tap outside
 * closes it. The definition is also the button's accessible description.
 */
export function Term({ children, tip, label }: { children?: ReactNode; tip: ReactNode; label: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const root = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onDown = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <span
      ref={root}
      className="relative inline-flex items-center gap-1 align-baseline"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children}
      <button
        type="button"
        aria-label={`What is ${label}?`}
        aria-describedby={id}
        aria-expanded={open}
        onClick={() => setOpen(true)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="inline-grid size-[1.125rem] shrink-0 place-items-center rounded-full border border-rule-strong text-[0.6875rem] leading-none text-ink-3 hover:border-ink-3 hover:text-ink"
      >
        ?
      </button>
      <span
        id={id}
        role="tooltip"
        className={`panel absolute left-1/2 top-full z-30 mt-2 w-[min(18rem,80vw)] -translate-x-1/2 p-3 text-left text-sm font-normal leading-snug text-ink-2 ${
          open ? "block" : "hidden"
        }`}
      >
        <span className="mb-1 block font-semibold text-ink">{label}</span>
        {tip}
      </span>
    </span>
  );
}
