import type { Metadata } from "next";
import type { ReactNode } from "react";

import { ForestPlot } from "@/components/charts/ForestPlot";
import { S1Chart, S1DotPlot } from "@/components/charts/S1Chart";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getResults } from "@/lib/data";
import { formatDuration, formatPercent, formatUsd } from "@/lib/format";
import { medianCurve, summarizeArm } from "@/lib/results";

export const metadata: Metadata = {
  title: "Results",
  description:
    "Pre-registered comparison of Nemotron-guided finger search against equal-budget random search: designs needed to reach a good design (primary), held-out success (secondary).",
};

const STATUS_TEXT = {
  synthetic: "Synthetic layout data, not a result",
  pilot: "Pilot runs, excluded from the confirmatory analysis",
  final: "Final, pre-registered analysis",
} as const;

const STAMP_TEXT = {
  synthetic: "SYNTHETIC LAYOUT DATA",
  pilot: "PILOT, NOT THE RESULT",
  final: null,
} as const;

function ChartFrame({ stamp, children }: { stamp: string | null; children: ReactNode }) {
  return (
    <div className="relative border border-rule bg-paper/70 px-3 py-4 sm:px-5">
      {children}
      {stamp ? (
        <div className="pointer-events-none absolute inset-0 grid place-items-center" aria-hidden="true">
          <span className="stamp -rotate-6 bg-paper/90 text-sm text-pending">{stamp}</span>
        </div>
      ) : null}
    </div>
  );
}

export default function ResultsPage() {
  const results = getResults();
  const { s1, h1 } = results;
  const stamp = STAMP_TEXT[results.status];
  const threshold = s1.threshold;
  const summaries = threshold !== null ? s1.arms.map((a) => summarizeArm(a, threshold, s1.budget)) : [];
  const runs = s1.arms.reduce((n, a) => n + a.runs.length, 0);
  const medians = s1.arms.map((a) => medianCurve(a.runs).join(","));
  const coincide = medians.length > 1 && medians.every((m) => m === medians[0]);

  return (
    <>
      <PageHeader
        title="Results"
        lede={
          <p>
            Does Nemotron, reading each round&apos;s measured failures, find a good finger in fewer simulated designs
            than random search with the same budget? Each simulated design stands in for one print-and-test cycle.
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

      <section className="mt-12" aria-labelledby="s1-title">
        <p className="text-xs font-semibold tracking-wide text-accent">PRIMARY ENDPOINT</p>
        <h2 id="s1-title" className="heading mt-1 text-xl sm:text-2xl">
          S1: speed to a good design
        </h2>
        <p className="prose-measure mt-2 text-sm text-ink-2">
          {threshold !== null ? (
            <>
              Simulated designs a run needs before its best training success reaches{" "}
              <span className="num text-ink">{formatPercent(threshold)}</span>. Runs that never get there are censored at
              B = <span className="num text-ink">{s1.budget}</span>. {s1.threshold_note}
            </>
          ) : (
            <>The threshold is not set yet, so only the curves are shown. {s1.threshold_note}</>
          )}
        </p>

        <h3 className="mt-6 text-sm text-ink">Best training success so far</h3>
        <div className="mt-2">
          <ChartFrame stamp={stamp}>
            <S1Chart s1={s1} />
          </ChartFrame>
        </div>
        {s1.shared_prefix > 0 || coincide ? (
          <p className="mt-2 text-xs text-ink-3">
            {s1.shared_prefix > 0
              ? `The first ${s1.shared_prefix} designs are the same in every run (the hand-designed entrants and the seed designs), so the arms can only differ after design ${s1.shared_prefix}. `
              : ""}
            {coincide ? "The arms' curves coincide here, so the thinner dashed line lies on top of the wider one." : ""}
          </p>
        ) : null}

        {summaries.length ? (
          <>
            <h3 className="mt-8 text-sm text-ink">Designs needed to reach the threshold</h3>
            <div className="mt-2">
              <ChartFrame stamp={stamp}>
                <S1DotPlot summaries={summaries} budget={s1.budget} />
              </ChartFrame>
            </div>
          </>
        ) : null}
      </section>

      <section className="mt-14" aria-labelledby="h1-title">
        <p className="text-xs font-semibold tracking-wide text-ink-3">SECONDARY ENDPOINT</p>
        <h2 id="h1-title" className="heading mt-1 text-xl">
          H1: held-out success
        </h2>
        {h1.status === "available" ? (
          <>
            <p className="mt-1 text-sm text-ink-3">
              Each run&apos;s frozen best design on objects the search never saw. Dots are runs, diamonds are means,
              lines are 95% bootstrap intervals; dashed lines are the hand-designed entrants.
            </p>
            <div className="mt-4">
              <ChartFrame stamp={stamp}>
                <ForestPlot forest={h1} />
              </ChartFrame>
            </div>
          </>
        ) : (
          <p className="mt-2 border-l-2 border-pending pl-3 text-sm text-ink-2">
            Not run yet. After the main search runs, each run&apos;s best design is frozen and tested on held-out object
            sizes, poses and scanned household objects the search never saw; the estimates appear here.
          </p>
        )}
      </section>

      <details className="mt-10 border border-rule px-4 py-3 text-sm">
        <summary className="cursor-pointer text-ink">Show the numbers and the runs behind them</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-left">
            <caption className="sr-only">S1 summary per arm</caption>
            <thead>
              <tr className="border-b border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-1.5 pr-4 font-normal">
                  Arm
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  Runs
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  Reached threshold
                </th>
                <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                  Median designs
                </th>
                <th scope="col" className="py-1.5 text-right font-normal">
                  Final best (median)
                </th>
              </tr>
            </thead>
            <tbody>
              {s1.arms.map((arm) => {
                const s = threshold !== null ? summarizeArm(arm, threshold, s1.budget) : null;
                const finals = arm.runs.map((r) => r.best.at(-1)).filter((v): v is number => v !== undefined).sort((a, b) => a - b);
                const mid = Math.floor(finals.length / 2);
                const medFinal = finals.length ? (finals.length % 2 ? finals[mid]! : (finals[mid - 1]! + finals[mid]!) / 2) : null;
                return (
                  <tr key={arm.id} className="border-b border-rule">
                    <th scope="row" className="py-1.5 pr-4 font-normal">
                      {arm.label}
                    </th>
                    <td className="num py-1.5 pr-4 text-right">{arm.runs.length}</td>
                    <td className="num py-1.5 pr-4 text-right">{s ? `${s.reached} of ${s.runs}` : "n/a"}</td>
                    <td className="num py-1.5 pr-4 text-right">
                      {s ? (s.median_designs === null ? `> ${s1.budget}` : s.median_designs) : "n/a"}
                    </td>
                    <td className="num py-1.5 text-right">{medFinal !== null ? formatPercent(medFinal, 1) : "n/a"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {results.sources.length ? (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[700px] border-collapse text-left">
              <caption className="pb-2 text-left text-xs text-ink-3">
                Where each of the {runs} runs ran. Job cost is the Nebius job; model calls to Nemotron on Token Factory
                are billed separately.
              </caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Run
                  </th>
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Job
                  </th>
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Where
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Wall time
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Job cost
                  </th>
                  <th scope="col" className="py-1.5 text-right font-normal">
                    Model calls
                  </th>
                </tr>
              </thead>
              <tbody>
                {results.sources.map((src) => (
                  <tr key={`${src.arm}-${src.run}`} className="border-b border-rule">
                    <th scope="row" className="num py-1.5 pr-4 font-normal">
                      {src.run}
                    </th>
                    <td className="num break-all py-1.5 pr-4">{src.job_id ?? "local run"}</td>
                    <td className="py-1.5 pr-4 text-ink-2">
                      {[src.kind === "nebius" ? "Nebius" : "Local", src.platform, src.region, src.executor ? `${src.executor} executor` : null]
                        .filter(Boolean)
                        .join(" · ")}
                    </td>
                    <td className="num py-1.5 pr-4 text-right">{src.wall_s !== null ? formatDuration(src.wall_s) : "n/a"}</td>
                    <td className="num py-1.5 pr-4 text-right">{src.cost_usd !== null ? formatUsd(src.cost_usd) : "n/a"}</td>
                    <td className="num py-1.5 text-right">
                      {src.llm_usd === null || src.llm_usd === undefined
                        ? "n/a"
                        : src.llm_calls === 0
                          ? "none"
                          : `${formatUsd(src.llm_usd)}${src.llm_calls ? ` (${src.llm_calls})` : ""}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </details>

      <TitleBlock title="Results" drawing="/results" sheet="4 of 6" />
    </>
  );
}
