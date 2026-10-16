import Link from "next/link";

import { getDataRevision } from "@/lib/data";
import { APP_NAME, REPO_URL } from "@/lib/site";

interface TitleBlockProps {
  /** Sheet title, usually the page heading. */
  title: string;
  /** Route shown as the drawing number, e.g. "/bout/m1-failures". */
  drawing: string;
  /** Position in the site, e.g. "2 of 5" or "Detail of Benchmark". */
  sheet: string;
}

/**
 * Engineering-drawing title block closing every page: what this sheet is, which
 * data revision it shows, its units and licence.
 */
export function TitleBlock({ title, drawing, sheet }: TitleBlockProps) {
  const cells: { label: string; value: string; wide?: boolean; mono?: boolean }[] = [
    { label: "Title", value: title, wide: true },
    { label: "Drawing", value: drawing, mono: true },
    { label: "Data revision", value: getDataRevision(), mono: true },
    { label: "Units", value: "SI, lengths in mm" },
    { label: "Sheet", value: sheet },
  ];
  return (
    <footer className="mt-20 border border-rule-strong text-sm" aria-label="Page information">
      <dl className="grid grid-cols-2 sm:grid-cols-6">
        {cells.map((cell) => (
          <div
            key={cell.label}
            className={`border-b border-r border-rule px-3 py-2 ${cell.wide ? "col-span-2 sm:col-span-2" : ""}`}
          >
            <dt className="text-xs text-ink-3">{cell.label}</dt>
            <dd className={`${cell.mono ? "num" : ""} truncate text-ink`}>{cell.value}</dd>
          </div>
        ))}
      </dl>
      <p className="flex flex-wrap justify-between gap-2 px-3 py-2 text-xs text-ink-3">
        <span>
          {APP_NAME} is open source under the Apache-2.0 licence.{" "}
          <Link className="link" href="/credits/">
            Credits
          </Link>
          {REPO_URL ? (
            <>
              {" "}
              <a className="link" href={REPO_URL}>
                Source code
              </a>
            </>
          ) : null}
        </span>
        <span>Simulated results. Physical prints are not yet validated.</span>
      </p>
    </footer>
  );
}
