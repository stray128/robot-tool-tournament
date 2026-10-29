/**
 * Build-time loader of every Nebius job the repository's documents name, by id or by run (server only).
 *
 * The numbers are read from the committed documents, never typed here: the job table of
 * docs/NEBIUS.md ("Confirmatory and held-out jobs", with wall time and the ledger's cost), the gate
 * GV4 job (docs/NEBIUS.md and its wall time in docs/BENCH_V2.md), the pilot jobs of
 * docs/PREREG.md B10, the compute section of docs/USECASE_V1_RESULTS.md, and the jobs the Nebius
 * notes name only by run (runs.ts). Every job id that any document of JOB_DOCS names must be on the
 * compute page (unlistedJobIds), and the notes' ledger total must equal the jobs named by id plus
 * those named only by run (ledgerAccount), or the build fails; a fact a document does not state
 * stays null.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { getJobs } from "@/lib/data";
import { formatUsd } from "@/lib/format";
import { DataError } from "@/lib/validate";

import { jobIds } from "./ids";
import { parseRunIds, parseRunJobs, type RunJob } from "./runs";

export interface ResultJob {
  id: string;
  /** The run folder the job wrote, when the document names it. */
  run: string | null;
  what: string;
  /** Which result or gate it belongs to. */
  result: string;
  /** null when the document does not say. */
  region: string | null;
  platform: string;
  /** Wall time as the document states it; null when it does not. */
  wall_s: number | null;
  /** Wall-based estimate from the spend ledger, USD; null when the document does not state it. */
  cost_usd: number | null;
  /** The wall time and the cost exactly as the document writes them, or null. */
  wall_text: string | null;
  cost_text: string | null;
  /** The committed document the row comes from, relative to the repository root. */
  source: string;
}

export interface ResultJobs {
  jobs: ResultJob[];
  /** The notes' jobs that no document gives an id: named by run only. */
  runJobs: RunJob[];
  /** Every job id the documents of JOB_DOCS name. */
  documented: string[];
  /** The notes' ledger total against the jobs the documents name (null when the notes state none). */
  ledger: LedgerAccount | null;
  /** Token Factory spend inside the Nemotron arm, as docs/NEBIUS.md states it. */
  tokenFactoryUsd: number | null;
}

const DEFAULT_REPO = path.join(process.cwd(), "..");
const NEBIUS = "docs/NEBIUS.md";
const RESULTS = "docs/RESULTS.md";
const EFFECTORS = "docs/EFFECTORS_RESULTS.md";
const USECASE = "docs/USECASE_V1_RESULTS.md";
const USECASE_PROTOCOL = "docs/USECASE_V1.md";
const PREREG = "docs/PREREG.md";
const BENCH_V2 = "docs/BENCH_V2.md";
const DEVIATIONS = "docs/DEVIATIONS.md";
/** The documents whose job ids the compute page must list. */
export const JOB_DOCS = [NEBIUS, RESULTS, EFFECTORS, USECASE, PREREG, BENCH_V2, DEVIATIONS] as const;

function read(repo: string, rel: string): string {
  const file = path.join(repo, rel);
  if (!existsSync(file)) throw new DataError(rel, "$", "the committed result document is missing");
  return readFileSync(file, "utf8");
}

/** "29 min 43 s" -> 1783, "206 s" -> 206. */
export function wallSeconds(text: string): number {
  const m = /^(?:(\d+) h )?(?:(\d+) min)?\s*(?:(\d+(?:\.\d+)?) s)?$/.exec(text.trim());
  if (!m || (!m[1] && !m[2] && !m[3])) throw new Error(`not a wall time: ${JSON.stringify(text)}`);
  return Number(m[1] ?? 0) * 3600 + Number(m[2] ?? 0) * 60 + Number(m[3] ?? 0);
}

/** The rows of NEBIUS.md's "Confirmatory and held-out jobs" table. */
export function parseNebiusTable(md: string): ResultJob[] {
  const start = md.indexOf("## Confirmatory and held-out jobs");
  if (start < 0) throw new DataError(NEBIUS, "$", 'no "Confirmatory and held-out jobs" section');
  const section = md.slice(start, md.indexOf("\n## ", start + 4) < 0 ? undefined : md.indexOf("\n## ", start + 4));
  const where = /on ([a-z]+-[a-z]+\d) `([a-z0-9-]+)`/.exec(section);
  const rows = [...section.matchAll(/^\| `(aijob-[a-z0-9]+)` \| `([^`]+)` \| (.+?) \| ([0-9hmins ]+?) \| \$([0-9.]+) \|$/gm)];
  if (!rows.length) throw new DataError(NEBIUS, "$", "the job table has no rows");
  return rows.map((m) => ({
    id: m[1]!,
    run: m[2]!,
    what: m[3]!.replace(/`/g, ""),
    result: "finger designer (prereg-v1)",
    region: where?.[1] ?? null,
    platform: where?.[2] ?? "not stated",
    wall_s: wallSeconds(m[4]!),
    cost_usd: Number(m[5]),
    wall_text: m[4]!.trim(),
    cost_text: `$${m[5]}`,
    source: NEBIUS,
  }));
}

/**
 * Gate GV4's job: platform, region and cost from docs/NEBIUS.md ("v2 on Nebius"), and its wall time
 * from docs/BENCH_V2.md's job row when given (else null).
 */
export function parseGv4Job(md: string, bench: string | null = null): ResultJob | null {
  const m = /\| GV4 job `(aijob-[a-z0-9]+)` \(`([^`]+)`\) \| `([a-z0-9-]+)` `[^`]+`, ([a-z]+-[a-z]+\d):/.exec(md);
  if (!m) return null;
  const cost = /\| Cost \| \$([0-9.]+) wall-based/.exec(md.slice(m.index));
  const wall = bench ? new RegExp(`\`${m[1]}\`[^|]*?wall (\\d+) s`).exec(bench) : null;
  return {
    id: m[1]!,
    run: m[2]!,
    what: "Gate GV4: the v2 physics on the executor of record, 384 episodes against the laptop",
    result: "finger designer (prereg-v1), gate",
    region: m[4]!,
    platform: m[3]!,
    wall_s: wall ? Number(wall[1]) : null,
    cost_usd: cost ? Number(cost[1]) : null,
    wall_text: wall ? `${wall[1]} s` : null,
    cost_text: cost ? `$${cost[1]}` : null,
    source: wall ? BENCH_V2 : NEBIUS,
  };
}

/**
 * The pilot jobs of docs/PREREG.md B10 (before the tag), one row per job id: the run, where it ran
 * (a Nebius cpu-d3 job or an H100 host, as the row says) and its role. No wall time or cost is stated.
 */
export function parsePreregPilots(md: string): ResultJob[] {
  const start = md.indexOf("## B10. Pilot");
  if (start < 0) return [];
  const end = md.indexOf("\n## ", start + 4);
  const out: ResultJob[] = [];
  for (const row of md.slice(start, end < 0 ? undefined : end).split("\n")) {
    const cells = row.split("|").map((c) => c.trim());
    if (cells.length < 6 || !row.includes("aijob-")) continue;
    const [, run, where, seeds, role] = cells;
    for (const part of where!.split(/;\s*/)) {
      const id = /`(aijob-[a-z0-9]+)`/.exec(part)?.[1];
      if (!id) continue;
      const platform = /`(cpu-[a-z0-9]+)`/.exec(part)?.[1] ?? (/H100/.test(part) ? "H100 host" : "not stated");
      out.push({
        id,
        run: null,
        what: `Pilot before the tag: ${run!.replace(/`/g, "").toLowerCase()}, seed ${seeds}, ${role!.replace(/`/g, "")}`,
        result: "finger designer (prereg-v1), pilot",
        region: null,
        platform,
        wall_s: null,
        cost_usd: null,
        wall_text: null,
        cost_text: null,
        source: PREREG,
      });
    }
  }
  return out;
}

/**
 * The date a dated log row ("| 2026-10-19 | ... ran once (job `aijob-...`) ...") gives a job: only a
 * row that says the job ran, before naming it, dates it. A later row that only cites a job (a
 * deviation note, a reference) does not move its date. A job's own run name, when known, wins over
 * any row (runDate).
 */
export function jobDates(md: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of md.split("\n")) {
    const date = /^\| (\d{4}-\d{2}-\d{2}) \|/.exec(line)?.[1];
    if (!date) continue;
    for (const id of jobIds(line)) {
      if (/\bran\b/.test(line.slice(0, line.indexOf(id)))) out[id] ??= date;
    }
  }
  return out;
}

/** "search-20261017-062725" -> "2026-10-17"; null for a name without a date. */
export function runDate(run: string | null | undefined): string | null {
  const m = /-(20\d{2})(\d{2})(\d{2})-\d{6}/.exec(run ?? "");
  return m ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

/** The use-case benchmark's job from the compute section of its results. */
export function parseUsecaseJob(md: string): ResultJob {
  const m = /One Nebius Serverless Job on ([a-z]+-[a-z]+\d) \(`([a-z0-9-]+)`[^)]*\), `(aijob-[a-z0-9]+)`, (\d+) s wall,\s+about \$([0-9.]+)/.exec(md.replace(/\n/g, " "));
  if (!m) throw new DataError(USECASE, "$", "no job line in the compute section");
  return {
    id: m[3]!,
    run: null,
    what: "Use-case benchmark v1: three arms on 32 held-out conditions per use case, plus the parity runs",
    result: "use-case benchmark v1",
    region: m[1]!,
    platform: m[2]!,
    wall_s: Number(m[4]),
    cost_usd: Number(m[5]),
    wall_text: `${m[4]} s`,
    cost_text: `about $${m[5]}`,
    source: USECASE,
  };
}

export { jobIds };

let cached: ResultJobs | null = null;

/** A Nebius job the site itself recorded (web/public/data/jobs.json): its id and the run it exported. */
export interface SiteJob {
  id: string;
  source: string;
}

function siteNebiusJobs(): SiteJob[] {
  return getJobs()
    .jobs.filter((j) => j.kind === "nebius")
    .map((j) => ({ id: j.id, source: j.source }));
}

function suffixOf(run: string | null | undefined): string | null {
  return /-20\d{6}-(\d{6})/.exec(run ?? "")?.[1] ?? null;
}

/**
 * The job id of each run the notes describe, where the repository documents it: the notes' own
 * "Job ids by run" table (and any other table row naming one run and one job id), and the site's
 * exported runs (jobs.json `source`, "export:search-20261015-134158-search"). Nothing is inferred.
 */
export function linkRuns(nebius: string, site: SiteJob[]): Map<string, { id: string; run: string | null }> {
  const out = new Map<string, { id: string; run: string | null }>();
  for (const [suffix, v] of parseRunIds(nebius)) out.set(suffix, v);
  for (const j of site) {
    const s = suffixOf(j.source);
    if (s && !out.has(s)) out.set(s, { id: j.id, run: null });
  }
  return out;
}

/** The ledger's jobs: those with a job id, and those the notes name only by run (no id anywhere). */
export interface LedgerAccount {
  total: number;
  since: string;
  usd: number;
  /** The date of the notes' section that states the total. */
  date: string;
  ids: string[];
  runs: RunJob[];
  /** Documented jobs that ran after that date (outside the total). */
  after: string[];
}

/**
 * The ledger in one sentence, from the counts alone: "The Nebius notes' ledger counts 17 jobs since
 * 2026-10-15, $4.12 in all, every one named by job id and listed on this page." One job "is" or
 * "has"; more "are" or "have".
 */
export function ledgerText(ledger: LedgerAccount): string {
  const be = (n: number) => (n === 1 ? "is" : "are");
  const usd = formatUsd(ledger.usd);
  let named: string;
  if (ledger.runs.length) {
    const n = ledger.runs.length;
    named = `. ${ledger.ids.length} of them ${be(ledger.ids.length)} named by job id; ${n} of them ${n === 1 ? "has" : "have"} no job id in any document and ${be(n)} listed by run name.`;
  } else if (ledger.ids.length === ledger.total) {
    named = ", every one named by job id and listed on this page.";
  } else {
    named = `. ${ledger.ids.length} of them ${be(ledger.ids.length)} named by job id.`;
  }
  const n = ledger.after.length;
  const after = n ? ` ${n === 1 ? "One job listed here ran" : `${n} jobs listed here ran`} after the ${ledger.date} section that states the total, so ${n === 1 ? "it is" : "they are"} not in it.` : "";
  return `The Nebius notes' ledger counts ${ledger.total} ${ledger.total === 1 ? "job" : "jobs"} since ${ledger.since}, ${usd} in all${named}${after}`;
}

export function getResultJobs(repo: string = DEFAULT_REPO, site: SiteJob[] = siteNebiusJobs()): ResultJobs {
  if (cached && repo === DEFAULT_REPO) return cached;
  const nebius = read(repo, NEBIUS);
  const gv4 = parseGv4Job(nebius, read(repo, BENCH_V2));
  const main = [...(gv4 ? [gv4] : []), ...parseNebiusTable(nebius), parseUsecaseJob(read(repo, USECASE))];
  const known = new Set(main.map((j) => j.id));
  const pilots = parsePreregPilots(read(repo, PREREG)).filter((j) => !known.has(j.id));
  const documented = [...new Set(JOB_DOCS.flatMap((rel) => jobIds(read(repo, rel))))];
  // The runs the notes describe, and the job ids the repository documents for them.
  const allRuns = parseRunJobs(nebius);
  const linked = linkRuns(nebius, site);
  const siteIds = new Set(site.map((j) => j.id));
  const facts = (r: RunJob, link: { run: string | null }) => ({
    run: link.run && link.run.length >= r.name.length ? link.run : r.name,
    wall_s: r.wall_s,
    cost_usd: r.cost_usd,
    wall_text: r.wall_text,
    cost_text: r.cost_text,
    region: r.region,
    source: NEBIUS,
  });
  const runOfId = new Map(allRuns.flatMap((r) => (linked.has(r.suffix) ? [[linked.get(r.suffix)!.id, r] as const] : [])));
  // A pilot whose run the notes describe takes that run's stated name, wall time and cost.
  const enriched = pilots.map((j) => {
    const r = runOfId.get(j.id);
    return r ? { ...j, ...facts(r, linked.get(r.suffix)!) } : j;
  });
  // A run with a documented id that no other row lists becomes that job's row.
  const listed = new Set([...main, ...enriched].map((j) => j.id));
  const byRun: ResultJob[] = allRuns.flatMap((r) => {
    const link = linked.get(r.suffix);
    if (!link || listed.has(link.id) || siteIds.has(link.id)) return [];
    return [{ id: link.id, what: r.what || "Nebius job", result: "set-up, gate and probe runs", platform: r.platform ?? "not stated", ...facts(r, link) }];
  });
  const jobs = [...main, ...enriched, ...byRun];
  const runJobs = allRuns.filter((r) => !linked.has(r.suffix));
  const tf = /\*\*\$([0-9.]+)\*\*; p50 latency/.exec(nebius.replace(/\n/g, " "));
  const out: ResultJobs = {
    jobs,
    runJobs,
    documented,
    ledger: ledgerAccount(repo, nebius, documented, jobs, runJobs, site),
    tokenFactoryUsd: tf ? Number(tf[1]) : null,
  };
  if (repo === DEFAULT_REPO) cached = out;
  return out;
}

/**
 * The ledger's total ("N jobs since <date>" in the notes) against the jobs the documents name. A job
 * dates from its own run name, else from the dated log row that says it ran (jobDates); a job that ran
 * after the section stating the total is outside it. Every job inside must have an id or a run name:
 * when the two counts do not add up to the total, the build fails.
 */
function ledgerAccount(
  repo: string,
  nebius: string,
  documented: string[],
  jobs: ResultJob[],
  runJobs: RunJob[],
  site: SiteJob[],
): LedgerAccount | null {
  const flat = nebius.replace(/\n/g, " ");
  const total = /ledger\*\* \(([0-9]+) jobs since ([0-9-]+)[^)]*\): \*\*\$([0-9.]+)\*\*/.exec(flat);
  if (!total) return null;
  const heading = /## Confirmatory and held-out jobs([^\n]*)/.exec(nebius);
  if (!heading) throw new DataError(NEBIUS, "$", "the ledger total has no dated section");
  const date = /^ \((\d{4}-\d{2}-\d{2})\)$/.exec(heading[1]!)?.[1];
  if (!date) throw new DataError(NEBIUS, "$", `cannot read the date of the section stating the ledger total: ${JSON.stringify(heading[0])}`);
  const logged = { ...jobDates(read(repo, USECASE_PROTOCOL)), ...jobDates(read(repo, DEVIATIONS)) };
  const dateOf = (id: string): string | null =>
    runDate(jobs.find((j) => j.id === id)?.run) ?? runDate(site.find((j) => j.id === id)?.source) ?? logged[id] ?? null;
  const ids = [...new Set([...documented, ...site.map((j) => j.id)])];
  const after = ids.filter((id) => {
    const d = dateOf(id);
    return d !== null && d > date;
  });
  const inside = ids.filter((id) => !after.includes(id));
  const account: LedgerAccount = { total: Number(total[1]), since: total[2]!, usd: Number(total[3]), date, ids: inside, runs: runJobs, after };
  if (inside.length + runJobs.length !== account.total) {
    throw new DataError(
      NEBIUS,
      "$",
      `the ledger counts ${account.total} jobs, but the documents name ${inside.length} by job id and ${runJobs.length} only by run name`,
    );
  }
  return account;
}

/** Documented job ids the page does not show (`shown`: every id on the compute page). The build fails on any. */
export function unlistedJobIds(results: ResultJobs, shown: Iterable<string>): string[] {
  const on = new Set(shown);
  return results.documented.filter((id) => !on.has(id));
}
