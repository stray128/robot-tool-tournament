import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";

/**
 * Credits of the vendored effector models, rendered from each model's SOURCE.json under
 * assets/effectors/ (the licence record of every vendored file: upstream URL and commit, SPDX
 * licence, copyright and credit lines, changes). Read at build time; the credits page is the
 * only page that names the models' makers.
 */

interface SourceRecord {
  path: string;
  license: string;
  copyright: string;
  attribution: string;
  upstream_url: string;
  upstream_commit: string;
  upstream_path?: string;
  changes: string[];
  status: string;
  note: string;
}

export interface EffectorModelCredit {
  model: string;
  family: string;
  records: SourceRecord[];
}

function asRecords(raw: unknown): SourceRecord[] {
  const list = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object"
      ? ((raw as Record<string, unknown>).files ?? (raw as Record<string, unknown>).sources ?? [])
      : [];
  if (!Array.isArray(list)) return [];
  return list
    .filter((r): r is Record<string, unknown> => !!r && typeof r === "object")
    .map((r) => ({
      path: String(r.path ?? ""),
      license: String(r.license ?? ""),
      copyright: String(r.copyright ?? ""),
      attribution: String(r.attribution ?? ""),
      upstream_url: String(r.upstream_url ?? ""),
      upstream_commit: String(r.upstream_commit ?? ""),
      upstream_path: r.upstream_path ? String(r.upstream_path) : undefined,
      changes: Array.isArray(r.changes) ? r.changes.map(String) : [],
      status: String(r.status ?? "green"),
      note: String(r.note ?? ""),
    }))
    .filter((r) => r.license && r.attribution);
}

/** Every SOURCE.json under assets/effectors/ (repository root), one entry per model directory. */
export function effectorCredits(root: string = path.join(process.cwd(), "..", "assets", "effectors")): EffectorModelCredit[] {
  if (!existsSync(root)) return [];
  const out: EffectorModelCredit[] = [];
  const walk = (dir: string, family: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const full = path.join(dir, entry.name);
      // The effector library (assets/effectors/library/) has its own credits section.
      if (entry.isDirectory() && !(dir === root && entry.name === "library")) walk(full, family || entry.name);
      else if (entry.name === "SOURCE.json") {
        const records = asRecords(JSON.parse(readFileSync(full, "utf8")));
        if (records.length > 0) out.push({ model: path.basename(dir), family, records });
      }
    }
  };
  walk(root, "");
  return out;
}

function unique<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

/** The credits page section: one entry per vendored model, with its licence and changes. */
export function EffectorCredits() {
  const models = effectorCredits();
  if (models.length === 0) {
    return (
      <p>
        No third-party effector models are vendored yet. The effector tournament&apos;s other families (the parallel jaw,
        procedural grippers, suction cups, the soft arm and designed morphologies) are our own geometry under Apache-2.0.
      </p>
    );
  }
  return (
    <>
      {models.map((m) => {
        const licences = unique(m.records.map((r) => r.license));
        const attributions = unique(m.records.map((r) => r.attribution));
        const copyrights = unique(m.records.map((r) => r.copyright));
        const upstream = unique(m.records.map((r) => `${r.upstream_url}@${r.upstream_commit.slice(0, 8)}`));
        const changes = unique(m.records.flatMap((r) => r.changes));
        const amber = m.records.filter((r) => r.status === "amber");
        return (
          <div key={`${m.family}/${m.model}`} id={`effector-${m.model}`} className="grid gap-1 border-b border-rule pb-3 text-sm">
            <p className="text-ink">{attributions.join("; ")}</p>
            <p className="num text-ink">{licences.join(", ")}</p>
            <p>
              Model <span className="num">{m.model}</span> ({m.records.length} {m.records.length === 1 ? "file" : "files"}). {copyrights.join("; ")}.
            </p>
            <p className="break-words">Upstream: {upstream.join(", ")}.</p>
            {changes.length > 0 ? <p>Modified: {changes.join("; ")}.</p> : <p>Unmodified.</p>}
            {amber.length > 0 ? <p className="text-pending">Caveat: {unique(amber.map((r) => r.note)).join("; ")}</p> : null}
          </div>
        );
      })}
    </>
  );
}
