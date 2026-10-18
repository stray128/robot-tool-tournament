"use client";

import { useMemo, useState } from "react";

import { failureText, type Matrix, type MatrixCell, orderRows, pct, prunedNotes } from "./schema";
import { RateGlyph, TierBadge } from "./ui";

export interface MatrixStep {
  id: string;
  title: string;
  primitive_title: string;
}

function RunCell({ cell }: { cell: Extract<MatrixCell, { state: "run" }> }) {
  const failures = failureText(cell.failure_modes);
  return (
    <>
      <span className="flex items-center gap-1.5">
        <RateGlyph rate={cell.rate} />
        <span className="num text-ink">{pct(cell.rate)}</span>
        <span className="num text-xs text-ink-3">
          {cell.successes}/{cell.n}
        </span>
        {cell.context === "carried" ? (
          <span className="text-[0.625rem] text-ink-3" title="Screened from the state the earlier steps leave">
            carried
          </span>
        ) : null}
      </span>
      {failures !== "none" ? <span className="mt-0.5 block text-[0.6875rem] leading-tight text-fail">{failures}</span> : null}
      <span className="sr-only">
        {cell.successes} of {cell.n} screen runs passed{failures !== "none" ? `; failures: ${failures}` : ""}
        {cell.context === "carried" ? "; screened from the carried state" : ""}
      </span>
    </>
  );
}

/**
 * The screen's success matrix: candidates against steps, each cell the success rate with its run
 * count and failure modes, or "pruned" with the reason (title on hover, a numbered note, the
 * reason panel on hover or focus, and screen-reader text). Scrolls sideways in its own box.
 */
export default function SuccessMatrix({ matrix, steps }: { matrix: Matrix; steps: MatrixStep[] }) {
  const rows = useMemo(() => orderRows(matrix.rows), [matrix]);
  const notes = useMemo(() => prunedNotes(matrix), [matrix]);
  const [detail, setDetail] = useState<string | null>(null);
  const titles = new Map(steps.map((s) => [s.id, s.title]));

  return (
    <div className="grid gap-3" data-matrix>
      <p className="min-h-[2.5rem] border-l-2 border-rule-strong pl-2 text-sm text-ink-2" aria-live="polite" data-matrix-detail>
        {detail ?? "Hover or focus a pruned cell to read why the candidate was not screened on that step."}
      </p>
      <div className="relative overflow-x-auto border border-rule" role="region" aria-label="Success matrix, candidates against steps" tabIndex={0}>
        <table className="w-full border-collapse text-left text-sm" style={{ minWidth: `${16 + steps.length * 9}rem` }}>
          <caption className="sr-only">
            Screen success rate of each candidate on each step, most passing steps first. Pruned cells give their reason.
          </caption>
          <thead>
            <tr className="border-b border-rule-strong text-xs text-ink-3">
              <th scope="col" className="sticky left-0 z-[1] bg-paper px-2 py-2 font-normal">
                Candidate
              </th>
              {steps.map((s) => (
                <th key={s.id} scope="col" className="px-2 py-2 font-normal">
                  <span className="block text-ink">{s.title}</span>
                  <span className="block">{s.primitive_title}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.effector_id} className="border-b border-rule align-top" data-matrix-row>
                <th scope="row" className="sticky left-0 z-[1] max-w-[16rem] bg-paper px-2 py-1.5 font-normal">
                  <span className="block text-ink">{row.display_name}</span>
                  <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
                    {row.family_title}
                    <TierBadge tier={row.fidelity} />
                    {row.off_the_shelf ? null : <span>designed</span>}
                  </span>
                </th>
                {steps.map((s) => {
                  const cell = row.cells[s.id]!;
                  if (cell.state === "run") {
                    return (
                      <td key={s.id} className="px-2 py-1.5" data-cell="run">
                        <RunCell cell={cell} />
                      </td>
                    );
                  }
                  const reason = cell.reason ?? "no physics model for this step yet";
                  const word = cell.state === "pruned" ? "pruned" : "not simulated yet";
                  const note = cell.reason ? notes.get(cell.reason) : undefined;
                  const text = `${row.display_name}, ${titles.get(s.id) ?? s.id}: ${word}, ${reason}`;
                  return (
                    <td key={s.id} className="px-2 py-1.5" data-cell={cell.state}>
                      <span
                        tabIndex={0}
                        title={reason}
                        data-reason={reason}
                        className="cursor-help border-b border-dotted border-ink-3 text-xs text-ink-3 outline-offset-2 focus-visible:outline-2 focus-visible:outline-accent"
                        onMouseEnter={() => setDetail(text)}
                        onFocus={() => setDetail(text)}
                        onMouseLeave={() => setDetail(null)}
                        onBlur={() => setDetail(null)}
                      >
                        {word}
                        {note ? <sup className="num ml-0.5">{note}</sup> : null}
                        <span className="sr-only">: {reason}</span>
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        Rate of screen runs passed, with the count and the failure modes. &quot;Carried&quot;: screened from the state the
        earlier steps leave. Pruned: a capability or fit check ruled the candidate out before any physics.
      </p>
      {notes.size ? (
        <details className="text-xs text-ink-2">
          <summary className="cursor-pointer text-sm text-ink-2">Pruned reasons ({notes.size})</summary>
          <ol className="mt-2 grid gap-1">
            {[...notes.entries()].map(([reason, n]) => (
              <li key={n} className="grid grid-cols-[1.5rem_minmax(0,1fr)]">
                <span className="num text-ink-3">{n}</span>
                <span>{reason}</span>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </div>
  );
}
