import type { Metadata } from "next";

import { ComputeGantt } from "@/components/charts/ComputeGantt";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getJobs } from "@/lib/data";
import { formatDuration, formatInt, formatUsd, formatUtc } from "@/lib/format";
import { repoBlob } from "@/lib/site";
import { DataError } from "@/lib/validate";

import { getResultJobs, ledgerText, unlistedJobIds } from "./nebius";

export const metadata: Metadata = {
  title: "Compute",
  description: "Every simulation job behind the scored numbers: where it ran, for how long and what it cost.",
};

export default function ComputePage() {
  const { jobs } = getJobs();
  const results = getResultJobs();
  const nebius = jobs.filter((j) => j.kind === "nebius");
  // The site's own Nebius runs are in the bottom table; the documents' other jobs in the top one.
  const siteIds = new Set(nebius.map((j) => j.id));
  const documentedJobs = results.jobs.filter((j) => !siteIds.has(j.id));
  const missing = unlistedJobIds(results, [...documentedJobs.map((j) => j.id), ...siteIds]);
  if (missing.length) throw new DataError("compute", "$", `the documents name jobs this page does not list: ${missing.join(", ")}`);
  const runJobs = results.runJobs;
  const ledger = results.ledger;
  const stated = [...documentedJobs, ...runJobs];
  const listedSeconds = nebius.reduce((s, j) => s + j.wall_s, 0) + stated.reduce((s, j) => s + (j.wall_s ?? 0), 0);
  const listedCost = nebius.reduce((s, j) => s + (j.cost_usd ?? 0), 0) + stated.reduce((s, j) => s + (j.cost_usd ?? 0), 0);
  const notStated = <span className="text-ink-3">not stated</span>;
  const totals: [string, string][] = [
    ["Nebius jobs listed", formatInt(nebius.length + documentedJobs.length + runJobs.length)],
    ["Their stated wall time", formatDuration(listedSeconds)],
    ["Their stated cost", formatUsd(listedCost)],
    ["Project ledger", ledger ? `${formatUsd(ledger.usd)}, ${ledger.total} jobs` : "n/a"],
    ["Token Factory, Nemotron arm", results.tokenFactoryUsd !== null ? formatUsd(results.tokenFactoryUsd) : "n/a"],
  ];

  return (
    <>
      <PageHeader
        title="Compute"
        lede={
          <p>
            Every scored simulation number on this site says where it ran. Scored simulation runs as Nebius Serverless
            Jobs from one pinned container image, writing results to Object Storage; this page lists each job with its
            platform, time and cost. Robustness certificates name their own run on their page. The Studio&apos;s worked
            examples ran locally on our machine, not as Nebius jobs; each example&apos;s page names its code, report and
            conditions.
          </p>
        }
      />

      <dl className="mt-8 grid grid-cols-2 gap-px border border-rule bg-rule text-sm sm:grid-cols-5">
        {totals.map(([k, v]) => (
          <div key={k} className="bg-paper px-3 py-2.5">
            <dt className="text-xs text-ink-3">{k}</dt>
            <dd className="num mt-0.5 text-lg text-ink">{v}</dd>
          </div>
        ))}
      </dl>

      {nebius.length === 0 ? (
        <p className="mt-6 border-l-2 border-pending pl-3 text-sm text-ink-2">
          No Nebius jobs are published yet. When the scored runs finish, each appears here with its job ID, platform,
          minutes and cost, next to the local development runs below.
        </p>
      ) : null}

      <section className="mt-12" aria-labelledby="results-jobs-title">
        <h2 id="results-jobs-title" className="heading text-xl">
          Every Nebius job the repository names
        </h2>
        <p className="mt-2 max-w-[80ch] text-sm text-ink-2">
          Every Nebius Serverless Job that the Nebius notes, the pre-registration, the bench notes, the deviations or the published results name by
          job id or by run name, with the wall time and the spend ledger&apos;s wall-based cost as those documents write them. The earlier smoke
          jobs, which have neither an id nor a run name, are not listed. The site&apos;s own early runs are in the table at the bottom.
        </p>
        {ledger ? (
          <p className="mt-2 max-w-[80ch] border-l-2 border-pending pl-3 text-sm text-ink-2" data-ledger-ids={ledger.ids.length} data-ledger-runs={ledger.runs.length}>
            {ledgerText(ledger)}
          </p>
        ) : null}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm" data-result-jobs>
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-4 font-normal">Job</th>
                <th scope="col" className="py-2 pr-4 font-normal">Result</th>
                <th scope="col" className="py-2 pr-4 font-normal">Where</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Wall time</th>
                <th scope="col" className="py-2 text-right font-normal">Cost</th>
              </tr>
            </thead>
            <tbody>
              {documentedJobs.map((job) => (
                <tr key={job.id} className="border-b border-rule align-top" data-job={job.id}>
                  <th scope="row" className="py-2.5 pr-4 font-normal">
                    <span className="block max-w-[46ch] text-ink">{job.what.charAt(0).toUpperCase() + job.what.slice(1)}</span>
                    <span className="num block text-xs text-ink-3">
                      {job.id}
                      {job.run ? `, run ${job.run}` : ""}
                    </span>
                  </th>
                  <td className="py-2.5 pr-4 text-ink-2">
                    {job.result}
                    <a className="link block text-xs" href={repoBlob(job.source)} rel="noopener">
                      {job.source.replace(/^docs\//, "")}
                    </a>
                  </td>
                  <td className="py-2.5 pr-4">
                    <span className="text-ink">
                      {/^(cpu|gpu)-/.test(job.platform) ? `Nebius ${job.platform}` : job.platform === "not stated" ? "Nebius, platform not stated" : `Nebius, ${job.platform}`}
                    </span>
                    <span className="num block text-xs text-ink-3">{job.region ?? "region not stated"}</span>
                  </td>
                  <td className="num py-2.5 pr-4 text-right">{job.wall_text ?? notStated}</td>
                  <td className="num py-2.5 text-right">{job.cost_text ?? notStated}</td>
                </tr>
              ))}
              {runJobs.map((run) => (
                <tr key={run.suffix} className="border-b border-rule align-top" data-run-job={run.suffix}>
                  <th scope="row" className="py-2.5 pr-4 font-normal">
                    <span className="block max-w-[46ch] text-ink">{run.what || "Nebius job"}</span>
                    <span className="num block text-xs text-ink-3">run {run.name}; job id not stated</span>
                  </th>
                  <td className="py-2.5 pr-4 text-ink-2">
                    set-up and gate runs, in the ledger total
                    <a className="link block text-xs" href={repoBlob("docs/NEBIUS.md")} rel="noopener">
                      NEBIUS.md
                    </a>
                  </td>
                  <td className="py-2.5 pr-4">
                    <span className="text-ink">{run.platform ? `Nebius ${run.platform}` : "Nebius, platform not stated"}</span>
                    <span className="num block text-xs text-ink-3">{run.region ?? "region not stated"}</span>
                  </td>
                  <td className="num py-2.5 pr-4 text-right">{run.wall_text ?? notStated}</td>
                  <td className="num py-2.5 text-right">{run.cost_text ?? notStated}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-ink-3">
          Sources:{" "}
          <a className="link" href={repoBlob("docs/NEBIUS.md")} rel="noopener">
            Nebius notes
          </a>
          ,{" "}
          <a className="link" href={repoBlob("docs/RESULTS.md")} rel="noopener">
            finger designer results
          </a>
          ,{" "}
          <a className="link" href={repoBlob("docs/USECASE_V1_RESULTS.md")} rel="noopener">
            use-case benchmark v1 results
          </a>
          ,{" "}
          <a className="link" href={repoBlob("docs/PREREG.md")} rel="noopener">
            pre-registration
          </a>
          ,{" "}
          <a className="link" href={repoBlob("docs/BENCH_V2.md")} rel="noopener">
            bench notes
          </a>
          ,{" "}
          <a className="link" href={repoBlob("docs/DEVIATIONS.md")} rel="noopener">
            deviations
          </a>
          . The v2 benchmark&apos;s jobs are added when it runs.
        </p>
      </section>

      <section className="mt-12" aria-labelledby="timeline-title">
        <h2 id="timeline-title" className="heading text-xl">
          Timeline
        </h2>
        <div className="mt-4 border border-rule bg-paper/70 p-4">
          {jobs.length ? <ComputeGantt jobs={jobs} /> : <p className="text-sm text-ink-3">No jobs recorded yet.</p>}
        </div>
      </section>

      <section className="mt-12" aria-labelledby="jobs-title">
        <h2 id="jobs-title" className="heading text-xl">
          Recorded runs on this site
        </h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-4 font-normal">Job</th>
                <th scope="col" className="py-2 pr-4 font-normal">Where</th>
                <th scope="col" className="py-2 pr-4 font-normal">Started</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Wall time</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Episodes</th>
                <th scope="col" className="py-2 text-right font-normal">Cost</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.id} className="border-b border-rule align-top">
                  <th scope="row" className="py-2.5 pr-4 font-normal">
                    <span className="text-ink">{job.name}</span>
                    <span className="num block text-xs text-ink-3">{job.id}</span>
                    <span className="mt-0.5 block max-w-[46ch] text-xs text-ink-2">{job.purpose}</span>
                  </th>
                  <td className="py-2.5 pr-4">
                    {job.kind === "nebius" ? (
                      <>
                        <span className="text-ink">Nebius {job.platform}</span>
                        <span className="num block text-xs text-ink-3">
                          {[job.preset, job.region, job.executor].filter(Boolean).join(", ")}
                        </span>
                      </>
                    ) : (
                      <>
                        <span className="text-ink">Local</span>
                        <span className="num block text-xs text-ink-3">{job.executor} executor</span>
                      </>
                    )}
                  </td>
                  <td className="num py-2.5 pr-4 text-ink-2">{formatUtc(job.started_at)}</td>
                  <td className="num py-2.5 pr-4 text-right">{formatDuration(job.wall_s)}</td>
                  <td className="num py-2.5 pr-4 text-right">{job.episodes !== null ? formatInt(job.episodes) : "n/a"}</td>
                  <td className="num py-2.5 text-right">{job.cost_usd !== null ? formatUsd(job.cost_usd) : "n/a"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <TitleBlock title="Compute" drawing="/compute" sheet="5 of 6" />
    </>
  );
}
