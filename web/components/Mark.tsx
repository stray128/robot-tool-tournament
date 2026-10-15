/** Product mark: two parallel fingers closing on a part, drawn as a plan view. */
export function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className} fill="none">
      <rect x="2.5" y="3" width="4" height="18" fill="currentColor" />
      <rect x="17.5" y="3" width="4" height="18" fill="var(--accent)" />
      <rect x="8.5" y="9" width="7" height="8" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8.5 6h7" stroke="currentColor" strokeWidth="1" strokeDasharray="1.5 1.5" />
    </svg>
  );
}
