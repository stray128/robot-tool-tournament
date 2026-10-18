import { ProvenanceChips } from "@/components/ProvenanceChips";
import { formatBytes, formatDuration } from "@/lib/format";

import { constantValue, humanize, type HeroPage } from "./schema";
import { SpecTagBadge } from "./ui";

type Chip = Parameters<typeof ProvenanceChips>[0]["chips"][number];

const short = (sha: string) => sha.slice(0, 12);

/** Chips for the page: each lane's trajectory hash, the evidence, the code and the report. */
export function studioChips(page: HeroPage): Chip[] {
  const p = page.provenance;
  const chips: Chip[] = page.lanes.map((lane) => ({
    key: lane.id,
    subject: "Replay",
    headline: `${lane.title}: trajectory ${short(lane.traj_sha)}`,
    facts: [lane.condition_text, `${lane.n_frames} frames at ${lane.fps} fps`, formatBytes(lane.bytes + lane.asset_bytes)],
    scored: true,
  }));
  chips.push({
    key: "evidence",
    subject: "Physics",
    headline: `${humanize(p.controllers)} controllers, evidence ${p.evidence}`,
    facts: [
      `${p.conditions.length} conditions`,
      ...(p.workers !== null ? [`${p.workers} workers`] : []),
      ...(p.timings.total_s !== undefined ? [`${formatDuration(p.timings.total_s)} in all`] : []),
    ],
    scored: p.evidence === "accepted",
  });
  chips.push({
    key: "code",
    subject: "Code",
    headline: `code ${p.code_version}`,
    facts: [`report ${short(p.report_sha256)}`, ...(p.fairness_sha ? [`fairness ${short(p.fairness_sha)}`] : []), `task ${short(p.task_id)}`],
    scored: true,
  });
  return chips;
}

/** Provenance, the conditions, determinism, assumed constants, fixed caveats and the sensitivity summary. */
export function ProvenanceView({ page }: { page: HeroPage }) {
  const p = page.provenance;
  const s = page.sensitivity;
  const labels = new Map(page.plan.objects.map((o) => [o.name, o.label.toLowerCase()]));
  return (
    <div className="grid gap-8" data-provenance>
      <ProvenanceChips chips={studioChips(page)} />

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="min-w-0">
          <h3 className="heading text-base">Conditions</h3>
          <p className="mt-1 text-sm text-ink-3">Every finalist ran every condition, in the same order with the same seeds.</p>
          <ul className="num mt-2 grid gap-1 text-xs text-ink-2">
            {p.conditions.map((c) => {
              const offsets = Object.entries(c.aim_offsets_mm);
              return (
                <li key={c.condition_id}>
                  <span className="text-ink">{c.condition_id}</span>, seed {c.seed}
                  {offsets.length ? (
                    <span className="text-ink-3">
                      {": "}
                      {offsets.map(([obj, v]) => `${labels.get(obj) ?? humanize(obj).toLowerCase()} (${v.map((x) => x.toFixed(1)).join(", ")}) mm`).join("; ")}
                    </span>
                  ) : (
                    <span className="text-ink-3">: nominal placement</span>
                  )}
                </li>
              );
            })}
          </ul>
          {p.determinism.length ? (
            <p className="mt-2 text-xs text-ink-2">
              Rerun check:{" "}
              {p.determinism.map((d) => `${d.condition} ${d.same ? "reproduced its trajectory" : "did not reproduce"}`).join("; ")}.
            </p>
          ) : null}
        </div>
        <div className="min-w-0">
          <h3 className="heading text-base">Sensitivity to the change times</h3>
          {s ? (
            <p className="mt-2 text-sm text-ink-2" data-sensitivity>
              Re-ranked with tool changes of {s.tool_change_s.join(", ")} s and head switches of {s.present_s.join(", ")} s ({s.n_pairs}{" "}
              pairs): the recommendation {s.recommendation_changes ? "changes" : "does not change"}, and the order changes in {s.order_changes} of{" "}
              {s.n_pairs}. {s.note ? `${s.note.charAt(0).toUpperCase()}${s.note.slice(1)}.` : ""}
            </p>
          ) : (
            <p className="mt-2 text-sm text-ink-3">Not computed for this use case.</p>
          )}
        </div>
      </div>

      <div className="min-w-0">
        <h3 className="heading text-base">Assumed constants</h3>
        <p className="mt-1 text-sm text-ink-3">Equal for every candidate; none is sourced from a target arm or tool changer.</p>
        <div className="relative mt-2 overflow-x-auto" role="region" aria-label="Assumed constants" tabIndex={0}>
          <table className="w-full min-w-[520px] border-collapse text-left text-sm" data-constants>
            <tbody>
              {page.constants.map((c) => (
                <tr key={c.key} className="border-b border-rule align-top">
                  <th scope="row" className="py-1.5 pr-3 font-normal text-ink-2">
                    {c.note ? c.note.charAt(0).toUpperCase() + c.note.slice(1) : humanize(c.key)}
                  </th>
                  <td className="num py-1.5 pr-3 text-right text-ink">{constantValue(c)}</td>
                  <td className="py-1.5">
                    <SpecTagBadge tag={c.tag} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h3 className="heading text-base">Caveats</h3>
        <ul className="prose-measure mt-2 grid list-disc gap-1.5 pl-5 text-sm text-ink-2" data-fixed-caveats>
          {page.fixed_caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
