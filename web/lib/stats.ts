/**
 * The landing page's live numbers, computed from the published data at build time so
 * they can never drift from the pages they summarise.
 */

import type { CertificateDoc } from "./certificate";
import { formatDuration, formatInt, formatUsd } from "./format";
import type { DesignsIndex, JobsIndex, ResultsIndex } from "./types";

export interface SiteStat {
  label: string;
  value: string;
  /** Where the number comes from, one short phrase. */
  note: string;
  href: string;
}

export function siteStats(input: {
  jobs: JobsIndex;
  designs: DesignsIndex;
  results: ResultsIndex;
  certificates: CertificateDoc[];
}): SiteStat[] {
  const { jobs, designs, results, certificates } = input;
  const nebius = jobs.jobs.filter((j) => j.kind === "nebius");
  const episodes = jobs.jobs.reduce((s, j) => s + (j.episodes ?? 0), 0);
  const nebiusSeconds = nebius.reduce((s, j) => s + j.wall_s, 0);
  const nebiusCost = nebius.reduce((s, j) => s + (j.cost_usd ?? 0), 0);
  const searchDesigns = results.s1.arms.reduce((s, a) => s + a.runs.reduce((n, r) => n + r.best.length, 0), 0);
  const searchRuns = results.s1.arms.reduce((s, a) => s + a.runs.length, 0);
  const certEpisodes = certificates.reduce((s, c) => s + c.certificates.reduce((n, o) => n + o.tests.length, 0), 0);
  const label = results.status === "final" ? "" : ` (${results.status})`;
  return [
    {
      label: "Grasp trials simulated",
      value: formatInt(episodes + certEpisodes),
      note: `${formatInt(episodes)} in published jobs, ${formatInt(certEpisodes)} in certificates`,
      href: "/compute/",
    },
    {
      label: `Designs searched${label}`,
      value: formatInt(searchDesigns),
      note: `${searchRuns} search ${searchRuns === 1 ? "run" : "runs"} behind the results page`,
      href: "/results/",
    },
    {
      label: "Nebius jobs",
      value: formatInt(nebius.length),
      note: nebius.length ? `${formatDuration(nebiusSeconds)} of job time, ${formatUsd(nebiusCost)}` : "none published yet",
      href: "/compute/",
    },
    {
      label: "Finger designs",
      value: formatInt(designs.designs.length),
      note: `${certificates.length} with a robustness certificate`,
      href: "/benchmark/",
    },
  ];
}
