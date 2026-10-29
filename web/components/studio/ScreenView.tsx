import { formatDuration, formatInt } from "@/lib/format";

import { humanize, type Plan, type Screen, stepTitle } from "./schema";
import { Facts, TierBadge } from "./ui";

const SOURCE_LABELS: Record<string, string> = {
  preset: "family preset",
  library: "library",
  designed: "designed",
  catalog: "catalog",
  manual: "hand-made",
};

/** Candidates screened and pruned, with the grouped reasons and the full list on request. */
export function ScreenView({ screen, plan }: { screen: Screen; plan: Plan }) {
  const pruned = screen.candidates.filter((c) => c.status === "pruned");
  const partly = screen.candidates.filter((c) => c.status === "screened" && c.pruned_steps.length > 0).length;
  const most = Math.max(1, ...screen.pruned_reasons.map((r) => r.count));
  return (
    <div className="grid gap-5" data-screen>
      <Facts
        items={[
          ["Candidates", formatInt(screen.n_candidates)],
          ["Screened on a step", formatInt(screen.n_screened)],
          ["Pruned from every step", formatInt(pruned.length)],
          ["Pruned from some steps", formatInt(partly)],
          ["Screen conditions", formatInt(screen.n_conditions)],
          ["Screen runs", formatInt(screen.runs)],
          ["Wall time", formatDuration(screen.wall_s)],
        ]}
      />
      {screen.pruned_reasons.length ? (
        <div>
          <h3 className="heading text-base">Why candidates were pruned</h3>
          <p className="mt-1 text-sm text-ink-3">A capability or fit check, before any physics. Counted per candidate and step, most common first.</p>
          <ul className="mt-2 grid gap-1.5 text-sm" data-pruned-reasons>
            {screen.pruned_reasons.map((r) => (
              <li key={r.reason} className="grid grid-cols-[3rem_minmax(0,1fr)] items-center gap-3 sm:grid-cols-[3rem_8rem_minmax(0,1fr)]">
                <span className="num text-right text-ink">{r.count}</span>
                <span className="hidden h-2 border border-rule-strong sm:block" aria-hidden="true">
                  <span className="block h-full bg-ink-3" style={{ width: `${(r.count / most) * 100}%` }} />
                </span>
                <span className="text-ink-2">{r.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <details className="border-t border-rule pt-3" data-candidates>
        <summary className="cursor-pointer text-sm text-ink-2">Every candidate ({screen.candidates.length})</summary>
        <div className="relative mt-3 overflow-x-auto" role="region" aria-label="Every candidate" tabIndex={0}>
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-3 font-normal">
                  Candidate
                </th>
                <th scope="col" className="py-2 pr-3 font-normal">
                  From
                </th>
                <th scope="col" className="py-2 pr-3 font-normal">
                  Fidelity
                </th>
                <th scope="col" className="py-2 font-normal">
                  Screen
                </th>
              </tr>
            </thead>
            <tbody>
              {screen.candidates.map((c) => (
                <tr key={c.effector_id} className="border-b border-rule align-top" data-candidate={c.status}>
                  <th scope="row" className="py-2 pr-3 font-normal">
                    <span className="text-ink">{c.display_name}</span>
                    <span className="mt-0.5 block text-xs text-ink-3">
                      {c.family_title}
                      {c.built_for ? `, shaped for the ${humanize(c.built_for).toLowerCase()}` : ""}
                      {c.studio_only ? ", Studio only" : ""}
                    </span>
                    {c.rating ? (
                      <span className="mt-0.5 block text-xs text-ink-2" data-rating>
                        {c.rating}
                      </span>
                    ) : null}
                    {c.studio_only && c.caveat ? (
                      <details className="mt-1 text-xs" data-tool-caveat>
                        <summary className="cursor-pointer text-pending">Caveat</summary>
                        <p className="mt-1 max-w-[60ch] text-ink-2">{c.caveat}</p>
                      </details>
                    ) : null}
                  </th>
                  <td className="py-2 pr-3 text-xs text-ink-2">
                    {SOURCE_LABELS[c.source] ?? humanize(c.source).toLowerCase()}
                    {c.off_the_shelf ? ", off the shelf" : ""}
                  </td>
                  <td className="py-2 pr-3">
                    <TierBadge tier={c.fidelity} />
                  </td>
                  <td className="py-2 text-xs">
                    {c.status === "pruned" ? <span className="text-fail">Pruned{c.pruned_reason ? `: ${c.pruned_reason}` : " from every step"}</span> : <span className="text-pass">Screened</span>}
                    {c.pruned_steps.length ? (
                      <ul className="mt-1 grid gap-0.5 text-ink-3">
                        {c.pruned_steps.map((p) => (
                          <li key={p.step_id}>
                            {stepTitle(plan, p.step_id)}: {p.reason}
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
