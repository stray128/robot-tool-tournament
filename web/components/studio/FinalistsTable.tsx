import { failureText, type FinalistRow, massText, pct, type Plan, plainReason, roleText, runsText, seconds, stepTitle } from "./schema";
import { TierBadge } from "./ui";

/** Where a finalist's runs stop: "2. Tighten coupling: 5 runs (slip 5)". */
export function stopsText(row: FinalistRow, plan: Plan): string[] {
  const list = Object.entries(row.stops_at_detail).sort((a, b) => b[1].runs - a[1].runs);
  return list.map(([step, d]) => `${stepTitle(plan, step)}: ${d.runs} ${d.runs === 1 ? "run" : "runs"} (${failureText(d.failure_modes)})`);
}

/** Every finalist on the full condition grid, the best single off-the-shelf tool marked as the baseline. */
export function FinalistsTable({ rows, plan, baselineId }: { rows: FinalistRow[]; plan: Plan; baselineId: string | null }) {
  return (
    <div className="relative overflow-x-auto" role="region" aria-label="Finalists against the best single off-the-shelf tool" tabIndex={0}>
      <table className="w-full min-w-[1040px] border-collapse text-left text-sm" data-finalists>
        <thead>
          <tr className="border-y border-rule-strong text-xs text-ink-3">
            <th scope="col" className="py-2 pr-3 font-normal">
              Setup
            </th>
            <th scope="col" className="py-2 pr-3 font-normal">
              Completion
            </th>
            <th scope="col" className="w-28 py-2 pr-3 font-normal">
              Weakest step
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">
              Cycle time
            </th>
            <th scope="col" className="w-48 py-2 pr-3 font-normal">
              Tools
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">
              Actuators
            </th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">
              Mass
            </th>
            <th scope="col" className="py-2 pr-3 font-normal">
              Fidelity
            </th>
            <th scope="col" className="py-2 font-normal">
              Where runs stop
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const baseline = r.setup_id === baselineId;
            const stops = stopsText(r, plan);
            return (
              <tr
                key={r.setup_id}
                className={`border-b border-rule align-top ${baseline ? "bg-paper-2/60" : ""}`}
                data-finalist={r.roles.join(" ")}
              >
                <th scope="row" className={`py-2 pr-3 font-normal ${baseline ? "border-l-2 border-ink-3 pl-2" : ""}`}>
                  <span className="block text-xs font-semibold uppercase tracking-wide text-accent">{roleText(r.roles)}</span>
                  <span className="block text-ink">{r.label}</span>
                  <span className="mt-0.5 block text-xs text-ink-3">{r.kind_title}</span>
                </th>
                <td className="py-2 pr-3">
                  <span className="num block whitespace-nowrap text-ink">{runsText(r.completed_runs, r.n_runs)}</span>
                  <span className="num text-xs text-ink-3">{pct(r.completion_rate)}</span>
                </td>
                <td className="py-2 pr-3 text-xs">
                  {r.weakest_step ? (
                    <>
                      <span className="block text-ink">{stepTitle(plan, r.weakest_step.step_id)}</span>
                      <span className="num text-ink-3">{pct(r.weakest_step.success_rate)}</span>
                    </>
                  ) : (
                    "n/a"
                  )}
                </td>
                <td className="num whitespace-nowrap py-2 pr-3 text-right">{seconds(r.cycle_time_s)}</td>
                <td className="py-2 pr-3 text-xs">
                  <span className="num text-ink">{r.complexity.n_tools}</span>
                  <ul className="mt-0.5 grid gap-0.5 text-ink-2">
                    {r.tool_details.map((t) => (
                      <li key={`${t.index}-${t.display_name}`}>{t.display_name}</li>
                    ))}
                  </ul>
                  {r.complexity.n_changes ? <span className="text-ink-3">{r.complexity.n_changes} {r.complexity.n_changes === 1 ? "change" : "changes"}</span> : null}
                  {r.complexity.bracket ? <span className="text-ink-3"> · printed bracket</span> : null}
                </td>
                <td className="num py-2 pr-3 text-right">{r.complexity.n_actuators}</td>
                <td className="num whitespace-nowrap py-2 pr-3 text-right">{massText(r.complexity.total_mass_kg)}</td>
                <td className="py-2 pr-3">
                  <TierBadge tier={r.fidelity} />
                </td>
                <td className="py-2 text-xs text-ink-2">
                  {stops.length ? (
                    <ul className="grid gap-0.5">
                      {stops.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-pass">completes every run</span>
                  )}
                  {!r.eligible && r.reason && !r.reason.startsWith("never completed") ? <span className="mt-1 block text-ink-3">Not eligible: {plainReason(r.reason)}</span> : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
