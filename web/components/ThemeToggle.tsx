"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

/** The site is dark first; a saved choice of "light" switches it. */
function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/** Switches between the dark (graphite) and light (drafting white) themes; remembers the choice. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => setTheme(currentTheme()), []);

  const next: Theme = theme === "light" ? "dark" : "light";

  return (
    <button
      type="button"
      onClick={() => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("theme", next);
        } catch {
          // Storage can be unavailable (private mode); the choice then lasts for this page only.
        }
        setTheme(next);
      }}
      className="inline-flex size-9 shrink-0 items-center justify-center rounded-md border border-rule-strong text-ink-2 hover:bg-paper-2 hover:text-ink"
      aria-label={theme ? `Switch to the ${next} theme` : "Switch theme"}
      title={theme ? `Switch to the ${next} theme` : "Switch theme"}
    >
      <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true">
        <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor" />
      </svg>
    </button>
  );
}
