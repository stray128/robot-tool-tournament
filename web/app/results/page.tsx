import type { Metadata } from "next";
import type { ReactNode } from "react";

import { ForestPlot } from "@/components/charts/ForestPlot";
import { S1Chart, S1DotPlot } from "@/components/charts/S1Chart";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getResults } from "@/lib/data";
import { formatMinutes, formatP, formatPercent, formatUsd } from "@/lib/format";
import { medianCurve, secondaryDiff, secondaryValue, summarizeArm } from "@/lib/results";
import { REPO_URL } from "@/lib/site";
import type { ResultSource, ResultsIndex } from "@/lib/types";

export const metadata: Metadata = {
  title: "Results",
  description:
    "Pre-registered comparison of Nemotron-guided finger search against equal-budget random search: designs needed to reach a good finger (primary), held-out success (secondary).",
};

const STATUS_TEXT = {
  synthetic: "Synthetic layout data, not a result",
  pilot: "Pilot runs, excluded from the confirmatory analysis",
  final: "Pre-registered confirmatory result",
} as const;

const STAMP_TEXT = {
  synthetic: "SYNTHETIC LAYOUT DATA",
  pilot: "PILOT, NOT THE RESULT",
  final: null,
} as const;

const COUNT_WORDS = ["none", "one", "two", "three", "four", "five"] as const;

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

/** A link into the repository when the build knows it, else plain text. */
function RepoLink({ href, children }: { href: string; children: ReactNode }) {
  return REPO_URL ? (
    <a className="link" href={`${REPO_URL}/${href}`}>
      {children}
    </a>
  ) : (
    <span>{children}</span>
  );
}

/** "37 of 48" when the file records the trials per design, else a percentage. */
function thresholdText(s1: ResultsIndex["s1"]): string {
  if (s1.threshold === null) return "n/a";
  return s1.trials ? `${Math.round(s1.threshold * s1.trials)} of ${s1.trials}` : formatPercent(s1.threshold);
}

interface JobRow {
  key: string;
  arm: string;
  src: ResultSource;
  runs: number;
  llmCalls: number | null;
  llmUsd: number | null;
}

/** One row per job (a confirmatory job runs every seed of one arm), in arm order. */
function jobRows(results: ResultsIndex): JobRow[] {
  const rows = new Map<string, JobRow>();
  for (const src of results.sources) {
    const key = src.job_id ?? `${src.arm}/${src.run}`;
    const row = rows.get(key) ?? {
      key,
      arm: results.s1.arms.find((a) => a.id === src.arm)?.label ?? src.arm,
      src,
      runs: 0,
      llmCalls: null,
      llmUsd: null,
    };
    row.runs += 1;
    if (typeof src.llm_calls === "number") row.llmCalls = (row.llmCalls ?? 0) + src.llm_calls;
    if (typeof src.llm_usd === "number") row.llmUsd = (row.llmUsd ?? 0) + src.llm_usd;
    rows.set(key, row);
  }
  return [...rows.values()];
}

export default function ResultsPage() {
  const results = getResults();
  const { s1, h1, secondary, protocol } = results;
  const analysis = s1.analysis;
  const final = results.status === "final";
  const stamp = STAMP_TEXT[results.status];
  const threshold = s1.threshold;
  const summaries = threshold !== null ? s1.arms.map((a) => summarizeArm(a, threshold, s1.budget)) : [];
  const medians = s1.arms.map((a) => medianCurve(a.runs).join(","));
  const coincide = medians.length > 1 && medians.every((m) => m === medians[0]);
  const target = thresholdText(s1);
  const counted = s1.count_from > 1;
  const jobs = jobRows(results);
  const anyCost = results.sources.some((s) => s.cost_usd !== null);
  const digests = [...new Set(results.sources.map((s) => s.image_digest).filter((d): d is string => !!d))];
  const armA = summaries.find((s) => s.id === "a-random");
  const armC = summaries.find((s) => s.id === "c-nemotron");

  return (
    <>
      <PageHeader
        title="Results"
        lede={
          counted && threshold !== null ? (
            <p>
              Given the same simulated budget of <span className="num">{s1.budget}</span> finger designs per run, does
              NVIDIA Nemotron 3 Super, reading the measured results of earlier designs, reach a finger that seats and
              centres the part in at least <span className="num">{target}</span> training grasps in fewer designs than
              equal-budget random search? Each simulated design stands in for one print-and-test cycle.
            </p>
          ) : (
            <p>
              Does Nemotron, reading each round&apos;s measured failures, find a good finger in fewer simulated designs
              than random search with the same budget? Each simulated design stands in for one print-and-test cycle.
            </p>
          )
        }
      >
        <div className={`border-l-2 pl-3 text-sm ${final ? "border-pass" : "border-pending"}`} data-status={results.status}>
          <p>
            <span className={`font-semibold ${final ? "text-pass" : "text-pending"}`}>{STATUS_TEXT[results.status]}.</span>{" "}
            <span className="text-ink-2">{results.note}</span>
          </p>
          {protocol ? (
            <p className="mt-1 text-ink-2">
              Protocol: <RepoLink href={`tree/${protocol.tag}`}>the {protocol.tag} tag</RepoLink> (
              <RepoLink href={`blob/${protocol.tag}/${protocol.protocol_doc}`}>{protocol.protocol_doc}</RepoLink>). Full
              write-up with limits and deviations:{" "}
              <RepoLink href={`blob/main/${protocol.results_doc}`}>{protocol.results_doc}</RepoLink>.
            </p>
          ) : null}
        </div>
      </PageHeader>

      <section className="mt-12" aria-labelledby="s1-title">
        <p className="text-xs font-semibold tracking-wide text-accent">PRIMARY ENDPOINT</p>
        <h2 id="s1-title" className="heading mt-1 text-xl sm:text-2xl">
          S1: speed to a good design
        </h2>
        <p className="prose-measure mt-2 text-sm text-ink-2">
          {threshold === null ? (
            <>The threshold is not set yet, so only the curves are shown. {s1.threshold_note}</>
          ) : counted ? (
            <>
              A run&apos;s count is the number of its first design, counted from design{" "}
              <span className="num text-ink">{s1.count_from}</span>, that seats and centres the part in at least{" "}
              <span className="num text-ink">{target}</span> training grasps. Designs 1 to{" "}
              <span className="num">{s1.count_from - 1}</span> are round 0 (the hand-designed entrants and seed designs),
              the same for both arms at each seed, so they do not count. A run that never gets there within{" "}
              <span className="num text-ink">{s1.budget}</span> designs counts as{" "}
              <span className="num text-ink">{s1.budget + 1}</span>. {s1.threshold_note}
            </>
          ) : (
            <>
              Simulated designs a run needs before its best training success reaches{" "}
              <span className="num text-ink">{target}</span>. Runs that never get there are censored at B ={" "}
              <span className="num text-ink">{s1.budget}</span>. {s1.threshold_note}
            </>
          )}
        </p>

        {summaries.length ? (
          <div className="mt-6 overflow-x-auto">
            <table className="w-full border-collapse text-left text-sm" data-table="s1-summary">
              <caption className="sr-only">S1 per arm</caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    <span className="sr-only">Measure</span>
                  </th>
                  {summaries.map((s) => (
                    <th key={s.id} scope="col" className="py-1.5 pr-4 text-right font-normal">
                      {s.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-rule">
                  <th scope="row" className="py-1.5 pr-4 font-normal text-ink-2">
                    Runs that reached the target within {s1.budget} designs
                  </th>
                  {summaries.map((s) => (
                    <td key={s.id} className="num py-1.5 pr-4 text-right">
                      {s.reached} of {s.runs}
                    </td>
                  ))}
                </tr>
                <tr className="border-b border-rule">
                  <th scope="row" className="py-1.5 pr-4 font-normal text-ink-2">
                    Median designs to the target
                  </th>
                  {summaries.map((s) => (
                    <td key={s.id} className="num py-1.5 pr-4 text-right">
                      {s.median_designs === null ? `not reached (above ${s1.budget})` : s.median_designs}
                    </td>
                  ))}
                </tr>
                <tr className="border-b border-rule">
                  <th scope="row" className="py-1.5 pr-4 font-normal text-ink-2">
                    Mean designs to the target (not reached counted as {s1.budget + 1})
                  </th>
                  {summaries.map((s) => (
                    <td key={s.id} className="num py-1.5 pr-4 text-right">
                      {s.mean_designs === null ? "n/a" : s.mean_designs.toFixed(1)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        ) : null}

        {analysis && armA && armC ? (
          <div className="mt-5 border border-rule-strong bg-paper-2/40 p-4" data-block="s1-analysis">
            <p className="text-ink">
              Nemotron reached the target in{" "}
              <span className="num">
                {armC.reached} of {armC.runs}
              </span>{" "}
              runs, random search in{" "}
              <span className="num">
                {armA.reached} of {armA.runs}
              </span>
              .
            </p>
            <ul className="mt-2 grid gap-1 text-sm text-ink-2">
              <li>
                Difference in means (A minus C):{" "}
                <strong className="num text-ink">{analysis.effect.diff.toFixed(2)} designs</strong>,{" "}
                {Math.round(analysis.effect.level * 100)}% interval{" "}
                <span className="num text-ink">
                  {analysis.effect.ci_low.toFixed(2)} to {analysis.effect.ci_high.toFixed(2)}
                </span>
                .
              </li>
              <li>
                One-sided {analysis.primary.method}:{" "}
                <strong className="num text-ink">p = {formatP(analysis.primary.p_value)}</strong>.
              </li>
              <li>
                Sensitivity, {analysis.logrank.method.split(",")[0]}:{" "}
                <span className="num text-ink">p = {formatP(analysis.logrank.p_value)}</span>.
              </li>
            </ul>
          </div>
        ) : null}

        <h3 className="mt-8 text-sm text-ink">Best training success so far</h3>
        <div className="mt-2">
          <ChartFrame stamp={stamp}>
            <S1Chart s1={s1} thresholdLabel={target} />
          </ChartFrame>
        </div>
        {counted || s1.shared_prefix > 0 || coincide ? (
          <p className="mt-2 text-xs text-ink-3">
            {counted
              ? `Thin lines are runs, heavy lines each arm's median. A run whose round 0 already reached the target still needs one of its own designs (from design ${s1.count_from}) to count.`
              : s1.shared_prefix > 0
                ? `The first ${s1.shared_prefix} designs are the same in every run (the hand-designed entrants and the seed designs), so the arms can only differ after design ${s1.shared_prefix}.`
                : ""}
            {coincide ? " The arms' curves coincide here, so the thinner dashed line lies on top of the wider one." : ""}
          </p>
        ) : null}

        {summaries.length ? (
          <>
            <h3 className="mt-8 text-sm text-ink">Designs needed to reach the target</h3>
            <div className="mt-2">
              <ChartFrame stamp={stamp}>
                <S1DotPlot summaries={summaries} budget={s1.budget} />
              </ChartFrame>
            </div>
          </>
        ) : null}

        {analysis?.lift_only ? (
          <p className="prose-measure mt-6 border-l-2 border-rule-strong pl-3 text-sm text-ink-2" data-note="lift-only">
            Exploratory, lift-only scoring (the part only has to come off the table):{" "}
            {analysis.lift_only.diff === 0
              ? "no difference between arms"
              : `a difference of ${analysis.lift_only.diff.toFixed(2)} designs`}{" "}
            (p = {formatP(analysis.lift_only.p_value)}). Both arms find a finger that lifts the part almost at once. The
            advantage is in seating and centring the part, which is what the next step in a work cell depends on.
          </p>
        ) : null}
      </section>

      {secondary ? (
        <section className="mt-14" aria-labelledby="secondary-title">
          <p className="text-xs font-semibold tracking-wide text-ink-3">SECONDARY ENDPOINTS (HOLM FAMILY)</p>
          <h2 id="secondary-title" className="heading mt-1 text-xl">
            Held-out success and search efficiency
          </h2>
          <p className="prose-measure mt-2 text-sm text-ink-2">
            Each run&apos;s frozen top-1 design was scored on suites neither arm saw during search
            {secondary.job ? (
              <>
                {" "}
                (Nebius job <span className="num break-all">{secondary.job.job_id}</span>)
              </>
            ) : null}
            .
          </p>
          <ul className="mt-4 grid gap-3 sm:hidden" aria-label="Secondary endpoints, arm C against arm A">
            {secondary.tests.map((t) => (
              <li key={t.id} className="border border-rule p-3 text-sm">
                <p className="text-ink">{t.label}</p>
                <dl className="mt-2 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1">
                  {[
                    ["Random (A)", secondaryValue(t, t.mean_a)],
                    ["Nemotron (C)", secondaryValue(t, t.mean_c)],
                    ["Difference", secondaryDiff(t)],
                    ["p", formatP(t.p_value)],
                    ["Holm-adjusted p", t.p_holm === null ? "n/a" : formatP(t.p_holm)],
                  ].map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="text-ink-3">{k}</dt>
                      <dd className="num text-right">{v}</dd>
                    </div>
                  ))}
                </dl>
              </li>
            ))}
          </ul>
          <div className="mt-4 hidden overflow-x-auto sm:block">
            <table className="w-full min-w-[720px] border-collapse text-left text-sm" data-table="secondary">
              <caption className="sr-only">Secondary endpoints, arm C against arm A</caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Endpoint
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Random (A)
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Nemotron (C)
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Difference
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    p
                  </th>
                  <th scope="col" className="py-1.5 text-right font-normal">
                    Holm-adjusted p
                  </th>
                </tr>
              </thead>
              <tbody>
                {secondary.tests.map((t) => (
                  <tr key={t.id} className="border-b border-rule">
                    <th scope="row" className="py-1.5 pr-4 font-normal">
                      {t.label}
                    </th>
                    <td className="num py-1.5 pr-4 text-right">{secondaryValue(t, t.mean_a)}</td>
                    <td className="num py-1.5 pr-4 text-right">{secondaryValue(t, t.mean_c)}</td>
                    <td className="num py-1.5 pr-4 text-right">{secondaryDiff(t)}</td>
                    <td className="num py-1.5 pr-4 text-right">{formatP(t.p_value)}</td>
                    <td className="num py-1.5 text-right">{t.p_holm === null ? "n/a" : formatP(t.p_holm)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {secondary.tests.every((t) => t.rejected) ? (
            <p className="mt-3 text-sm text-ink-2">
              All {COUNT_WORDS[secondary.tests.length] ?? secondary.tests.length} reject the null at alpha{" "}
              {secondary.alpha} after Holm adjustment.
            </p>
          ) : null}
          {secondary.width_bands ? (
            <p className="prose-measure mt-3 text-sm text-ink-2" data-note="width-bands">
              H2 by object width (planned split, reported outside the Holm family):{" "}
              {secondary.width_bands.bands
                .map(
                  (b, i) =>
                    `${i === 0 ? "objects " : ""}${b.label} (${b.objects} objects) ${formatPercent(b.mean_a, 1)} vs ${formatPercent(b.mean_c, 1)} (p = ${formatP(b.p_value)})`,
                )
                .join("; ")}
              .
            </p>
          ) : null}
          <p className="prose-measure mt-3 text-sm text-ink-2">
            Real scanned objects are hard for every design in this space: the best arm seats them about a quarter of the
            time. The held-out real-object result says the Nemotron arm&apos;s picks generalise better, not that the
            fingers are ready for arbitrary objects.
          </p>
        </section>
      ) : h1 ? (
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
      ) : null}

      {final ? (
        <section className="mt-14" aria-labelledby="limits-title">
          <h2 id="limits-title" className="heading text-xl">
            Limits
          </h2>
          <ul className="prose-measure mt-2 grid list-disc gap-1 pl-5 text-sm text-ink-2">
            <li>Everything is simulated, with rigid contacts. No finger was printed or tested on hardware for this result.</li>
            <li>One gripper geometry, six training parts, one task (seat, lift, hold).</li>
            <li>
              The effector tournament (hands, grippers, soft arm, suction, Nemotron-designed effectors) is product mode
              and not part of this pre-registered result.
            </li>
          </ul>
        </section>
      ) : null}

      <details className="mt-10 border border-rule px-4 py-3 text-sm">
        <summary className="cursor-pointer text-ink">Show the jobs behind these numbers</summary>
        {jobs.length ? (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[640px] border-collapse text-left">
              <caption className="pb-2 text-left text-xs text-ink-3">
                Where the search runs ran. Model calls go to Nemotron on Token Factory and are billed separately from the
                job.
              </caption>
              <thead>
                <tr className="border-b border-rule-strong text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Arm
                  </th>
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Job
                  </th>
                  <th scope="col" className="py-1.5 pr-4 font-normal">
                    Where
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Runs
                  </th>
                  <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                    Run time
                  </th>
                  {anyCost ? (
                    <th scope="col" className="py-1.5 pr-4 text-right font-normal">
                      Job cost
                    </th>
                  ) : null}
                  <th scope="col" className="py-1.5 text-right font-normal">
                    Model calls
                  </th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(({ key, arm, src, runs, llmCalls, llmUsd }) => {
                  const time = src.run_s ?? src.wall_s;
                  return (
                    <tr key={key} className="border-b border-rule">
                      <th scope="row" className="py-1.5 pr-4 font-normal">
                        {arm}
                      </th>
                      <td className="num break-all py-1.5 pr-4">{src.job_id ?? "local run"}</td>
                      <td className="py-1.5 pr-4 text-ink-2">
                        {[src.kind === "nebius" ? "Nebius" : "Local", src.platform, src.region].filter(Boolean).join(" · ")}
                      </td>
                      <td className="num py-1.5 pr-4 text-right">{runs}</td>
                      <td className="num py-1.5 pr-4 text-right">{time !== null && time !== undefined ? formatMinutes(time) : "n/a"}</td>
                      {anyCost ? (
                        <td className="num py-1.5 pr-4 text-right">{src.cost_usd !== null ? formatUsd(src.cost_usd) : "n/a"}</td>
                      ) : null}
                      <td className="num py-1.5 text-right">
                        {llmCalls === null
                          ? "n/a"
                          : llmCalls === 0
                            ? "none"
                            : anyCost && llmUsd !== null
                              ? `${llmCalls} (${formatUsd(llmUsd)})`
                              : llmCalls}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
        {digests.length ? (
          <p className="mt-3 text-xs text-ink-3">
            Image: <span className="num break-all">{digests.join(", ")}</span>
          </p>
        ) : null}
        {analysis || secondary ? (
          <p className="mt-1 text-xs text-ink-3">
            Analysis outputs: <span className="num break-all">{[analysis?.source, secondary?.source].filter(Boolean).join(", ")}</span>{" "}
            (released with the run data).
          </p>
        ) : null}
      </details>

      <TitleBlock title="Results" drawing="/results" sheet="4 of 6" />
    </>
  );
}
