import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { getJobs } from "@/lib/data";
import { DataError } from "@/lib/validate";

import {
  getResultJobs,
  JOB_DOCS,
  ledgerText,
  jobDates,
  jobIds,
  parseGv4Job,
  parseNebiusTable,
  parsePreregPilots,
  parseUsecaseJob,
  runDate,
  type SiteJob,
  unlistedJobIds,
  wallSeconds,
} from "../app/compute/nebius";
import { parseRunIds, parseRunJobs } from "../app/compute/runs";

const REPO = path.join(__dirname, "..", "..");
const doc = (rel: string) => readFileSync(path.join(REPO, rel), "utf8");
const SITE: SiteJob[] = getJobs()
  .jobs.filter((j) => j.kind === "nebius")
  .map((j) => ({ id: j.id, source: j.source }));
const USECASE_JOB = "aijob-e05e55n905qj9va2fh";

const tmp: string[] = [];
afterEach(() => {
  while (tmp.length) rmSync(tmp.pop()!, { recursive: true, force: true });
});

/** A copy of the repository's docs with `edit` applied to docs/NEBIUS.md. */
function editedRepo(edit: (md: string) => string): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), "rtt-compute-"));
  tmp.push(dir);
  // dereference: a preview copy whose docs/ is a symlink must not be written through to the real docs
  cpSync(path.join(REPO, "docs"), path.join(dir, "docs"), { recursive: true, dereference: true });
  const file = path.join(dir, "docs", "NEBIUS.md");
  writeFileSync(file, edit(readFileSync(file, "utf8")));
  return dir;
}

describe("the compute page's Nebius jobs", () => {
  const results = getResultJobs(REPO, SITE);
  const { jobs, ledger, tokenFactoryUsd } = results;

  it("guards the ledger sentence: 17 jobs since 2026-10-15, $4.12, every one named by job id", () => {
    expect(ledger).not.toBeNull();
    expect(ledger!.total).toBe(17);
    expect(ledger!.since).toBe("2026-10-15");
    expect(ledger!.usd).toBe(4.12);
    expect(ledger!.date).toBe("2026-10-17");
    expect(ledger!.ids).toHaveLength(17);
    expect(ledger!.runs).toHaveLength(0);
    expect(results.runJobs).toEqual([]);
    expect(ledger!.after).toEqual([USECASE_JOB]);
    expect(ledgerText(ledger!)).toBe(
      "The Nebius notes' ledger counts 17 jobs since 2026-10-15, $4.12 in all, every one named by job id and listed on this page. One job listed here ran after the 2026-10-17 section that states the total, so it is not in it.",
    );
  });

  it("words the ledger in the singular for one job", () => {
    const one = { ...ledger!, runs: [results.runJobs[0] ?? ({} as never)], ids: ledger!.ids.slice(0, 16) };
    expect(ledgerText(one)).toContain("16 of them are named by job id; 1 of them has no job id in any document and is listed by run name.");
    expect(ledgerText({ ...ledger!, runs: [], ids: ledger!.ids.slice(0, 1) })).toContain(". 1 of them is named by job id.");
    expect(ledgerText({ ...ledger!, after: [] })).not.toContain("ran after");
    expect(ledgerText({ ...ledger!, total: 1, runs: [], ids: ledger!.ids.slice(0, 1), after: [] })).toContain("counts 1 job since");
  });

  it("takes the run-named jobs' ids from the notes' table, with their stated facts as written", () => {
    const table = parseRunIds(doc("docs/NEBIUS.md"));
    const byRun = (suffix: string) => jobs.find((j) => j.id === table.get(suffix)!.id)!;
    expect(table.get("172651")).toEqual({ id: "aijob-e05te2wrh14djpf6d0", run: "search-20261015-172651" });
    const rows = ["101446", "102052", "133917", "170620", "171556", "173010"].map(byRun);
    expect(rows.map((j) => [j.run, j.wall_text, j.cost_text])).toEqual([
      ["m1-20261015-101446", null, "$0.169"],
      ["rtt-m1-20261015-102052", null, "$0.159"],
      ["rtt-m1-20261015-133917", "122 s", "$0.153"],
      ["m1-20261015-170620", null, "about $0.026"],
      ["m1-20261015-171556", "134 s", "$0.020"],
      ["probe-20261015-173010", "116 s", "$0.017"],
    ]);
    for (const j of rows) expect(j.source).toBe("docs/NEBIUS.md");
    expect(byRun("173010").what).toBe("Probe job");
    // a platform or region the notes do not state for a run stays unstated
    expect([byRun("170620").platform, byRun("170620").region]).toEqual(["not stated", null]);
    // the pre-registration's executor check, by the documented id
    const check = jobs.find((j) => j.id === "aijob-e05te2wrh14djpf6d0")!;
    expect([check.run, check.wall_text, check.cost_text, check.region]).toEqual(["search-20261015-172651", "132 s", "$0.019", "uk-south2"]);
  });

  it("the ledger's stated costs add up to its $4.12 total", () => {
    const inside = new Set(ledger!.ids);
    const byId = [...jobs.filter((j) => inside.has(j.id)), ...getJobs().jobs.filter((j) => inside.has(j.id) && !jobs.some((x) => x.id === j.id))];
    expect(byId).toHaveLength(17);
    expect(byId.reduce((s, j) => s + (j.cost_usd ?? 0), 0)).toBeCloseTo(4.12, 2);
  });

  it("without the table row a run is listed by run name; without either, the build fails", () => {
    const row = /^\| `probe-20261015-173010` \| `aijob-[a-z0-9]+` \|$/m;
    const bullet = /^- \*\*Probe job `…173010`\*\*.*$/m;
    expect(doc("docs/NEBIUS.md")).toMatch(row);
    expect(doc("docs/NEBIUS.md")).toMatch(bullet);
    const noId = getResultJobs(editedRepo((md) => md.replace(row, "")), SITE);
    expect([noId.ledger!.ids.length, noId.ledger!.runs.length]).toEqual([16, 1]);
    expect(noId.runJobs.map((r) => [r.suffix, r.wall_text, r.cost_text])).toEqual([["173010", "116 s", "$0.017"]]);
    const repo = editedRepo((md) => md.replace(row, "").replace(bullet, "- A probe job ran too."));
    expect(() => getResultJobs(repo, SITE)).toThrow(/counts 17 jobs, but the documents name 16 by job id and 0 only by run name/);
  });

  it("fails the build when the section stating the total has no readable date", () => {
    const repo = editedRepo((md) => md.replace("## Confirmatory and held-out jobs (2026-10-17)", "## Confirmatory and held-out jobs (mid October)"));
    expect(() => getResultJobs(repo, SITE)).toThrow(DataError);
    expect(() => getResultJobs(repo, SITE)).toThrow(/cannot read the date/);
  });

  it("dates a job from its own run, not from a later row that cites it", () => {
    const md = [
      "| Date | Note |",
      "|---|---|",
      "| 2026-10-16 | The pilot ran once (job `aijob-aaaaaaaa`). |",
      "| 2026-10-20 | A deviation note cites job `aijob-aaaaaaaa` and job `aijob-bbbbbbbb` again. |",
    ].join("\n");
    expect(jobDates(md)).toEqual({ "aijob-aaaaaaaa": "2026-10-16" });
    expect(runDate("search-20261017-062725")).toBe("2026-10-17");
    expect(runDate("…172651")).toBeNull();
    // the use-case job: its protocol's log says it ran on 2026-10-19
    expect(jobDates(doc("docs/USECASE_V1.md"))[USECASE_JOB]).toBe("2026-10-19");
  });

  it("lists every job id the Nebius notes and the result documents name", () => {
    expect([...JOB_DOCS]).toEqual([
      "docs/NEBIUS.md",
      "docs/RESULTS.md",
      "docs/EFFECTORS_RESULTS.md",
      "docs/USECASE_V1_RESULTS.md",
      "docs/PREREG.md",
      "docs/BENCH_V2.md",
      "docs/DEVIATIONS.md",
    ]);
    const ids = new Set(jobs.map((j) => j.id));
    for (const rel of JOB_DOCS) for (const id of jobIds(doc(rel))) expect(ids.has(id), `${rel}: ${id}`).toBe(true);
    const shown = [...jobs.filter((j) => !SITE.some((s) => s.id === j.id)).map((j) => j.id), ...SITE.map((s) => s.id)];
    expect(unlistedJobIds(results, shown)).toEqual([]);
    expect(unlistedJobIds(results, shown.filter((id) => id !== "aijob-e05te2wrh14djpf6d0"))).toEqual(["aijob-e05te2wrh14djpf6d0"]);
  });

  it("reads the GV4 job's wall time from the bench notes, and leaves it unstated without them", () => {
    const gv4 = parseGv4Job(doc("docs/NEBIUS.md"), doc("docs/BENCH_V2.md"))!;
    expect([gv4.wall_s, gv4.cost_usd]).toEqual([135, 0.04]);
    expect(parseGv4Job(doc("docs/NEBIUS.md"))!.wall_s).toBeNull();
  });

  it("reads the use-case v1 job and the finger designer's table as published", () => {
    expect(parseUsecaseJob(doc("docs/USECASE_V1_RESULTS.md"))).toMatchObject({ id: USECASE_JOB, region: "uk-south2", platform: "cpu-d3", wall_s: 206, cost_usd: 0.06 });
    const rows = parseNebiusTable(doc("docs/NEBIUS.md"));
    expect(rows).toHaveLength(7);
    for (const r of rows) expect(r.wall_s! * r.cost_usd!).toBeGreaterThan(0);
    expect(parsePreregPilots(doc("docs/PREREG.md")).map((j) => j.platform)).toContain("cpu-d3");
    expect(tokenFactoryUsd).toBe(3.569);
    // every run the notes name next to a job id is that job's, not a run-named job
    expect(parseRunJobs(doc("docs/NEBIUS.md")).map((r) => r.suffix)).not.toContain("062725");
  });

  it("parses wall times", () => {
    expect(wallSeconds("29 min 43 s")).toBe(1783);
    expect(wallSeconds("206 s")).toBe(206);
    expect(() => wallSeconds("soon")).toThrow();
  });
});
