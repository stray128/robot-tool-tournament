/**
 * Nebius jobs that the Nebius notes (docs/NEBIUS.md) name by run name, read from the notes' own
 * tables and sentences, never typed here. A run is named either in full (`rtt-m1-20261015-133917`,
 * `runs/m1-20261015-101446/`) or by its time suffix (`…101446`); both forms are the same run.
 *
 * Facts per run, each exactly as the notes state it, else null:
 *  - wall time: a "Wall" row of a table whose columns are runs, or "<n> s wall" in a sentence;
 *  - cost: an "Estimated cost, wall-based" row or column, or the first "$<x>" after the run in a
 *    sentence ("cost about $0.026");
 *  - platform and region: `cpu-d3` or H100 / `gpu-h100-sxm`, uk-south2 or eu-north1, in the run's
 *    column label, its row or sentence, or the heading of its section.
 * Runs named on a line that also names a job id are that job's run (the id is the key), and are
 * left out here.
 */

import { jobIds } from "./ids";

export interface RunJob {
  /** The six-digit time suffix that identifies the run in the notes. */
  suffix: string;
  /** The longest form of the run's name the notes give ("rtt-m1-20261015-133917" or "…101446"). */
  name: string;
  /** What the notes say the run was. */
  what: string;
  platform: string | null;
  region: string | null;
  wall_s: number | null;
  cost_usd: number | null;
  /** The wall time and the cost exactly as the notes write them ("122 s", "$0.020"), or null. */
  wall_text: string | null;
  cost_text: string | null;
  /** The label of the run's column in a comparison table ("random search, H100"), when it has one. */
  column: string | null;
  /** The other runs of that table, by label, for pairing hosts that ran the same work. */
  peers: Record<string, string>;
}

const FULL = /(?:rtt-)?[a-z0-9]+-20\d{6}-(\d{6})/g;
const SHORT = /…(\d{6})/g;

function suffixes(text: string): string[] {
  return [...new Set([...[...text.matchAll(FULL)].map((m) => m[1]!), ...[...text.matchAll(SHORT)].map((m) => m[1]!)])];
}

function clean(text: string): string {
  return text
    .replace(/\*\*/g, "")
    .replace(/`/g, "")
    .replace(/\(\s*…\d{6}\s*\)/g, "")
    .replace(/…\d{6},?\s*/g, "")
    .replace(/\(\s*,?\s*/g, "(")
    .replace(/\(\s*\)/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[,.]$/, "");
}

/** The one platform a text names; null when it names none or both (a comparison). */
function platformOf(text: string): string | null {
  const cpu = /cpu-d3/.test(text);
  const gpu = /gpu-h100-sxm|\bH100\b/.test(text);
  if (cpu === gpu) return null;
  return cpu ? "cpu-d3" : "gpu-h100-sxm";
}

/** The one region a text names; null when it names none or both. */
function regionOf(text: string): string | null {
  const found = [...new Set([...text.matchAll(/\b(uk-south2|eu-north1)\b/g)].map((m) => m[1]!))];
  return found.length === 1 ? found[0]! : null;
}

/** The region a heading pairs with a platform ("uk-south2 `cpu-d3` against eu-north1 H100"). */
function regionFor(platform: string | null, heading: string): string | null {
  if (!platform) return null;
  const token = platform === "cpu-d3" ? "`?cpu-d3`?" : "`?(?:gpu-h100-sxm|H100)`?";
  return new RegExp(`\\b(uk-south2|eu-north1) ${token}`).exec(heading)?.[1] ?? null;
}

function seconds(cell: string): number | null {
  const m = /^(\d+(?:\.\d+)?) s$/.exec(clean(cell));
  return m ? Number(m[1]) : null;
}

/** The "$<x>" a text states, as written, with an "about" in front of it kept. */
function dollarText(cell: string): string | null {
  return /(?:about )?\$[0-9]+(?:\.[0-9]+)?/.exec(cell)?.[0] ?? null;
}

function dollars(cell: string): number | null {
  const m = /\$([0-9]+(?:\.[0-9]+)?)/.exec(cell);
  return m ? Number(m[1]) : null;
}

/**
 * The runs the notes describe on lines without a job id, with their stated facts. (A line naming a
 * run next to a job id only maps the one to the other: parseRunIds. A run named only there is not
 * returned.)
 */
export function parseRunJobs(md: string): RunJob[] {
  const lines = md.split("\n");
  const runs = new Map<string, RunJob>();
  // The runs some line without a job id names. The facts come from those lines; a run named only
  // next to a job id is that job's row (parseRunIds), so it is skipped here.
  const withoutId = new Set<string>();
  for (const line of lines) if (!jobIds(line).length) for (const s of suffixes(line)) withoutId.add(s);
  const onlyBesideAnId = (s: string): boolean => !withoutId.has(s);
  const get = (s: string): RunJob => {
    let r = runs.get(s);
    if (!r) {
      r = { suffix: s, name: `…${s}`, what: "", platform: null, region: null, wall_s: null, cost_usd: null, wall_text: null, cost_text: null, column: null, peers: {} };
      runs.set(s, r);
    }
    return r;
  };
  const fill = (r: RunJob, f: Partial<RunJob>) => {
    for (const [k, v] of Object.entries(f) as [keyof RunJob, unknown][]) {
      if (v === null || v === undefined || v === "") continue;
      if (r[k] === null || r[k] === "") (r as unknown as Record<string, unknown>)[k] = v;
    }
  };
  // Every full name the notes give, for the display name (from any line).
  for (const m of md.matchAll(FULL)) {
    const s = m[1]!;
    if (onlyBesideAnId(s)) continue;
    const r = get(s);
    if (m[0].length > r.name.length) r.name = m[0];
  }
  let heading = "";
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (/^#{2,4} /.test(line)) heading = line.replace(/^#+ /, "");
    if (!line.startsWith("|")) {
      // A sentence or bullet naming a run (and no job id).
      if (jobIds(line).length) continue;
      for (const s of suffixes(line)) {
        if (onlyBesideAnId(s)) continue;
        const at = line.search(new RegExp(`…${s}|-${s}`));
        const rest = line.slice(at);
        const wall = /(\d+(?:\.\d+)?) s wall/.exec(rest);
        const bold = /^\s*-?\s*\*\*([^*]+)\*\*/.exec(line)?.[1];
        const cost = /\$/.test(rest) ? dollars(rest) : null;
        // A sentence describes the run only when it states a fact about it (a wall time or a cost).
        if (!wall && cost === null) continue;
        const platform = platformOf(line) ?? platformOf(heading);
        fill(get(s), {
          what: clean(bold ?? line.replace(/^\s*-\s*/, "").split(/ cost | ran | still /)[0]!),
          wall_s: wall ? Number(wall[1]) : null,
          wall_text: wall ? `${wall[1]} s` : null,
          cost_usd: cost,
          cost_text: cost !== null ? dollarText(rest) : null,
          platform,
          region: regionOf(line) ?? regionOf(heading) ?? regionFor(platform, heading),
        });
      }
      continue;
    }
    // A markdown table: read it whole.
    const table: string[][] = [];
    let j = i;
    while (j < lines.length && lines[j]!.startsWith("|")) {
      if (!/^\|[\s|:-]+\|$/.test(lines[j]!)) table.push(lines[j]!.split("|").slice(1, -1).map((c) => c.trim()));
      j++;
    }
    i = j - 1;
    const [head, ...body] = table;
    if (!head || jobIds(head.join(" ")).length) continue;
    const rows = body.filter((r) => !jobIds(r.join(" ")).length);
    const runCols = head.map((h) => suffixes(h).filter((s) => !onlyBesideAnId(s))[0] ?? null);
    if (runCols.some(Boolean)) {
      // Columns are runs: "Wall" and "Estimated cost, wall-based" rows hold their facts.
      const labels: Record<string, string> = {};
      runCols.forEach((s, c) => {
        if (s) labels[clean(head[c]!)] = s;
      });
      runCols.forEach((s, c) => {
        if (!s) return;
        const label = clean(head[c]!);
        const r = get(s);
        const row = (re: RegExp) => rows.find((x) => re.test(clean(x[0] ?? "")))?.[c] ?? null;
        const wall = row(/^Wall$/i);
        const cost = row(/^Estimated cost, wall-based/i);
        const platform = platformOf(head[c]!) ?? platformOf(heading);
        fill(r, {
          what: label && !/^Job \d+$/.test(label) ? `${clean(heading)}: ${label}` : clean(heading),
          wall_s: wall ? seconds(wall) : null,
          wall_text: wall && seconds(wall) !== null ? clean(wall) : null,
          cost_usd: cost ? dollars(cost) : null,
          cost_text: cost ? dollarText(cost) : null,
          platform,
          region: regionOf(head[c]!) ?? regionOf(heading) ?? regionFor(platform, heading),
          column: label || null,
        });
        r.peers = { ...r.peers, ...labels };
      });
      continue;
    }
    // Rows are runs: a cost column and a wall column, when the header has them.
    const costCol = head.findIndex((h) => /Estimated cost|Est\. cost/i.test(h));
    const wallCol = head.findIndex((h) => /^Wall$/i.test(clean(h)));
    for (const row of rows) {
      const s = suffixes(row[0] ?? "").filter((x) => !onlyBesideAnId(x))[0];
      if (!s) continue;
      const work = head.findIndex((h) => /^Work$/i.test(clean(h)));
      const before = (row[0] ?? "").replace(/`[^`]*`/g, "").trim();
      const context = (head.slice(0, 1).join("") + " " + heading).trim();
      fill(get(s), {
        what: clean(`${clean(heading)}${before ? `: ${before.replace(/^\(|\)$/g, "")}` : ""}${work >= 0 && row[work] ? `, ${row[work]}` : ""}`),
        wall_s: wallCol >= 0 && row[wallCol] ? seconds(row[wallCol]!) : null,
        wall_text: wallCol >= 0 && row[wallCol] && seconds(row[wallCol]!) !== null ? clean(row[wallCol]!) : null,
        cost_usd: costCol >= 0 && row[costCol] ? dollars(row[costCol]!) : null,
        cost_text: costCol >= 0 && row[costCol] ? dollarText(row[costCol]!) : null,
        platform: platformOf(row.join(" ")) ?? platformOf(context) ?? platformOf(md.slice(md.indexOf(heading), md.indexOf(line))),
        region: regionOf(row.join(" ")) ?? regionOf(context) ?? regionOf(md.slice(md.indexOf(heading), md.indexOf(line))),
      });
    }
  }
  return [...runs.values()].filter((r) => !onlyBesideAnId(r.suffix));
}

/**
 * Job ids the notes give runs: every table row naming one run in full and one job id ("| `m1-20261015-101446` |
 * `aijob-...` |"), keyed by the run's time suffix.
 */
export function parseRunIds(md: string): Map<string, { id: string; run: string }> {
  const out = new Map<string, { id: string; run: string }>();
  for (const line of md.split("\n")) {
    if (!line.startsWith("|")) continue;
    const ids = jobIds(line);
    const names = [...line.matchAll(FULL)];
    if (ids.length !== 1 || names.length !== 1) continue;
    out.set(names[0]![1]!, { id: ids[0]!, run: names[0]![0] });
  }
  return out;
}
