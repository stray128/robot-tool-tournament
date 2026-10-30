import { useId } from "react";

/**
 * Product mark: a parallel gripper with its fingers closed on a part, the same shape as the film's
 * logo. The palm, mount and part take the text colour, so the mark reads on both themes; the fingers
 * keep the orange gradient.
 */
export function Mark({ className }: { className?: string }) {
  const finger = `mark-finger-${useId().replace(/:/g, "")}`;
  return (
    <svg viewBox="0 0 100 100" aria-hidden="true" className={className} fill="none">
      <defs>
        <linearGradient id={finger} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffa463" />
          <stop offset="0.6" stopColor="#ff7a2e" />
          <stop offset="1" stopColor="#c24808" />
        </linearGradient>
      </defs>
      <rect x="8" y="14" width="84" height="12" rx="3" fill="currentColor" opacity="0.55" />
      <rect x="43" y="2" width="14" height="13" rx="2" fill="currentColor" opacity="0.75" />
      <rect x="37" y="56" width="26" height="26" rx="3" fill="currentColor" />
      <polygon points="23,26 37,26 37,64.5 44,64.5 44,84 23,84" fill={`url(#${finger})`} />
      <polygon points="63,26 77,26 77,84 56,84 56,64.5 63,64.5" fill={`url(#${finger})`} />
    </svg>
  );
}
