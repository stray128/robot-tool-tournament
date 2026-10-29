import { getDataRevision } from "@/lib/data";

interface TitleBlockProps {
  /** Sheet title, usually the page heading. */
  title: string;
  /** Route shown as the drawing number, e.g. "/bout/m1-failures". */
  drawing: string;
  /** Position in the site, e.g. "2 of 5" or "Detail of Benchmark". */
  sheet: string;
}

/**
 * Engineering-drawing title block closing every page: what this sheet is, which data revision it
 * shows and its units. Licence and credits are in the site footer below it.
 */
export function TitleBlock({ title, drawing, sheet }: TitleBlockProps) {
  const cells: { label: string; value: string; wide?: boolean; mono?: boolean }[] = [
    { label: "Sheet", value: title, wide: true },
    { label: "Drawing", value: drawing, mono: true },
    { label: "Data revision", value: getDataRevision(), mono: true },
    { label: "Units", value: "SI, lengths in mm" },
    { label: "Section", value: sheet },
  ];
  return (
    <aside className="mt-20 overflow-hidden rounded-lg border border-rule text-sm" aria-label="Page information">
      <dl className="grid grid-cols-2 sm:grid-cols-6">
        {cells.map((cell) => (
          <div key={cell.label} className={`min-w-0 border-b border-r border-rule px-3 py-2 last:border-r-0 ${cell.wide ? "col-span-2" : ""}`}>
            <dt className="text-xs text-ink-3">{cell.label}</dt>
            <dd className={`${cell.mono ? "num" : ""} truncate text-ink-2`}>{cell.value}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}
