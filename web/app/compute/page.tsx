import type { Metadata } from "next";

import { ComputeGantt } from "@/components/charts/ComputeGantt";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getJobs } from "@/lib/data";
import { formatDuration, formatInt, formatUsd, formatUtc } from "@/lib/format";

export const metadata: Metadata = {
  title: "Compute",
  description: "Every simulation job behind the published numbers: where it ran, for how long and what it cost.",
};

export default function ComputePage() {
  const { jobs } = getJobs();
  const nebius = jobs.filter((j) => j.kind === "nebius");
  const nebiusSeconds = nebius.reduce((s, j) => s + j.wall_s, 0);
  const nebiusCost = nebius.reduce((s, j) => s + (j.cost_usd ?? 0), 0);
  const episodes = jobs.reduce((s, j) => s + (j.episodes ?? 0), 0);
  const totals: [string, string][] = [
    ["Nebius jobs", formatInt(nebius.length)],
    ["Nebius job time", nebius.length ? formatDuration(nebiusSeconds) : "0 s"],
    ["Nebius cost", formatUsd(nebiusCost)],
    ["Local runs", formatInt(jobs.length - nebius.length)],
    ["Simulated episodes", formatInt(episodes)],
  ];

  return (
    <>
      <PageHeader
        title="Compute"
        lede={
          <p>
            Every number on this site traces to a job. Scored simulation runs as Nebius Serverless Jobs from one pinned
            container image, writing results to Object Storage; this page lists each job with its platform, time and
            cost.
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
          Jobs
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

      <TitleBlock title="Compute" drawing="/compute" sheet="4 of 5" />
    </>
  );
}
