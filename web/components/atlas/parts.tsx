import { type AtlasKind, KIND_LABELS, KIND_NOTES, kindCounts, type AtlasTask } from "@/lib/atlas";

/** A small drawing of a setup kind: one tool, two tools and a swap, or a two-tool head. */
export function KindGlyph({ kind, className = "size-5" }: { kind: AtlasKind; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      {kind === "single" ? (
        <>
          <path d="M12 2v6" />
          <path d="M7 8h10v3H7z" />
          <path d="M8.5 11v7.5M15.5 11v7.5" />
        </>
      ) : kind === "tool_change" ? (
        <>
          <path d="M6 3v4M3.5 7h5v2h-5zM4.5 9v5M7.5 9v5" />
          <path d="M18 3v4M15.5 7h5v2h-5z" />
          <circle cx="18" cy="12.5" r="2.5" />
          <path d="M8 19.5h8M13.5 17l2.5 2.5-2.5 2.5" />
        </>
      ) : (
        <>
          <path d="M12 2v5" />
          <path d="M12 7l-6 4M12 7l6 4" />
          <path d="M4 11h4v2H4zM5 13v6M7 13v6" />
          <path d="M16 11h4v2h-4z" />
          <circle cx="18" cy="16.5" r="2.5" />
        </>
      )}
    </svg>
  );
}

const KIND_TONE: Record<AtlasKind, string> = {
  single: "bg-accent",
  tool_change: "bg-baseline",
  two_tool_head: "bg-pending",
};

export const KIND_ORDER: readonly AtlasKind[] = ["single", "tool_change", "two_tool_head"];

/** One stacked bar of setup kinds for a set of tasks, with its counts in words for screen readers. */
export function KindBar({ tasks, label }: { tasks: readonly AtlasTask[]; label: string }) {
  const counts = kindCounts(tasks);
  const n = tasks.length || 1;
  const words = KIND_ORDER.map((k) => `${counts[k]} ${KIND_LABELS[k].toLowerCase()}`).join(", ");
  return (
    <div className="flex h-3 w-full overflow-hidden rounded-full bg-paper-2" role="img" aria-label={`${label}: ${words}`}>
      {KIND_ORDER.map((k) =>
        counts[k] ? <span key={k} className={`h-full border-r border-surface last:border-r-0 ${KIND_TONE[k]}`} style={{ width: `${(counts[k] / n) * 100}%` }} /> : null,
      )}
    </div>
  );
}

export function KindLegend() {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-ink-2">
      {KIND_ORDER.map((k) => (
        <li key={k} className="flex items-center gap-2" title={KIND_NOTES[k]}>
          <span className={`inline-block size-2.5 rounded-sm ${KIND_TONE[k]}`} aria-hidden="true" />
          <KindGlyph kind={k} className="size-4 text-ink-3" />
          {KIND_LABELS[k]}
        </li>
      ))}
    </ul>
  );
}
