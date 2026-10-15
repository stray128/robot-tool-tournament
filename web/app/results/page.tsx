import type { Metadata } from "next";
import type { ReactNode } from "react";

import { BestSoFarChart, medianCurve } from "@/components/charts/BestSoFarChart";
import { ForestPlot } from "@/components/charts/ForestPlot";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getResults } from "@/lib/data";
import { formatPercent } from "@/lib/format";

export const metadata: Metadata = {
  title: "Results",
  description: "Pre-registered held-out results: Nemotron-guided search against equal-budget random search.",
};

const STATUS_TEXT = {
  synthetic: "Synthetic layout data, not a result",
  pilot: "Pilot runs, excluded from the confirmatory analysis",
  final: "Final, pre-registered analysis",
} as const;

function ChartFrame({ stamp, children }: { stamp: string | null; children: ReactNode }) {
  return (
    <div className="relative border border-rule bg-paper/70 px-3 py-4 sm:px-5">
      {children}
      {stamp ? (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <span className="stamp -rotate-6 bg-paper/90 text-sm text-pending">{stamp}</span>
        </div>
      ) : null}
    </div>
  );
}

export default function ResultsPage() {
  const results = getResults();
  const stamp = results.status === "final" ? null : STATUS_TEXT[results.status];

  return (
    <>
      <PageHeader
        title="Results"
        lede={
          <p>
            Does Nemotron with measured feedback find better fingers than random search with the same number of
            simulated designs? The primary endpoint is each run&apos;s held-out success on objects the search never
            saw.
          </p>
        }
      >
        <p
          className={`inline-block border px-3 py-1.5 text-sm ${
            results.status === "final" ? "border-pass text-pass" : "border-pending text-pending"
          }`}
        >
          <span className="font-semibold">{STATUS_TEXT[results.status]}.</span> {results.note}
        </p>
      </PageHeader>

      <section className="mt-12" aria-labelledby="forest-title">
        <h2 id="forest-title" className="heading text-xl">
          {results.forest.endpoint}
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          Dots are runs, diamonds are means, lines are 95% bootstrap intervals. Dashed lines are the hand-designed
          entrants.
        </p>
        <div className="mt-4">
          <ChartFrame stamp={stamp}>
            <ForestPlot forest={results.forest} />
          </ChartFrame>
        </div>
      </section>

      <section className="mt-14" aria-labelledby="curves-title">
        <h2 id="curves-title" className="heading text-xl">
          {results.curves.endpoint}
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          How fast each search finds a good design: the best training success seen so far after each simulated design.
        </p>
        <div className="mt-4">
          <ChartFrame stamp={stamp}>
            <BestSoFarChart curves={results.curves} />
          </ChartFrame>
        </div>
      </section>

      <details className="mt-10 border border-rule px-4 py-3 text-sm">
        <summary className="cursor-pointer text-ink">Show the numbers behind both charts</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[520px] border-collapse text-left">
            <thead>
              <tr className="border-b border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-1.5 pr-4 font-normal">
                  Arm
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  Runs
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  Mean held-out
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  95% interval
                </th>
                <th scope="col" className="py-1.5 text-right font-normal">
                  Median best at budget
                </th>
              </tr>
            </thead>
            <tbody>
              {results.forest.arms.map((arm) => {
                const curve = results.curves.arms.find((a) => a.id === arm.id);
                const last = curve ? medianCurve(curve.runs).at(-1) : undefined;
                return (
                  <tr key={arm.id} className="border-b border-rule">
                    <th scope="row" className="py-1.5 pr-4 font-normal">
                      {arm.label}
                    </th>
                    <td className="num py-1.5 pr-4 text-right">{arm.runs.length}</td>
                    <td className="num py-1.5 pr-4 text-right">{formatPercent(arm.mean, 1)}</td>
                    <td className="num py-1.5 pr-4 text-right">
                      {formatPercent(arm.ci95[0], 1)} to {formatPercent(arm.ci95[1], 1)}
                    </td>
                    <td className="num py-1.5 text-right">{last !== undefined ? formatPercent(last, 1) : "n/a"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>

      <TitleBlock title="Results" drawing="/results" sheet="3 of 5" />
    </>
  );
}
