import { stopsText } from "./FinalistsTable";
import {
  type Blocker,
  blockerRange,
  type FinalistRow,
  failureText,
  humanize,
  massText,
  type NoWinner,
  pct,
  type Plan,
  quantityName,
  type Recommendation,
  runsText,
  seconds,
  stepTitle,
  targetText,
} from "./schema";
import { Facts, TierBadge, VerifyMark } from "./ui";

function geometryText(rec: Recommendation): string | null {
  const g = rec.geometry;
  if (!g) return null;
  if (g.kind === "changer") {
    const parts = [g.stack_mm !== null ? `${g.stack_mm} mm stack` : null, g.changeover_s !== null ? `${g.changeover_s} s per change` : null].filter(Boolean);
    return `Tool changer${parts.length ? `: ${parts.join(", ")}` : ""}`;
  }
  const mounts = g.mounts.map(
    (m, i) => `tool ${i + 1} at ${m.angle_deg.toFixed(0)} deg, ${m.standoff_mm.toFixed(0)} mm out${m.lateral_mm ? `, ${m.lateral_mm.toFixed(0)} mm aside` : ""}${m.spin_deg ? `, spun ${m.spin_deg.toFixed(0)} deg` : ""}`,
  );
  const extra = [g.clearance_mm !== null ? `${g.clearance_mm.toFixed(1)} mm swept clearance` : null, g.present_s !== null ? `${g.present_s} s head switch` : null].filter(Boolean);
  return `Bracket: ${mounts.join("; ")}${extra.length ? ` (${extra.join(", ")})` : ""}`;
}

/** The recommendation: what it is, why, every step with its tier and caveat, and the baseline. */
export function RecommendationCard({ rec, plan, bestSingle }: { rec: Recommendation; plan: Plan; bestSingle: FinalistRow | null }) {
  const stepCaveats = new Set(rec.steps.flatMap((s) => s.caveats));
  // Each caveat in full once; a later step with the same caveat points back to the first.
  const firstStep = new Map<string, string>();
  for (const s of rec.steps) for (const c of s.caveats) if (!firstStep.has(c)) firstStep.set(c, s.step_id);
  const vs = rec.versus_best_single;
  const geometry = geometryText(rec);
  return (
    <article className="border border-rule-strong bg-paper p-4 sm:p-5" aria-label="Recommendation" data-recommendation>
      <p className="text-xs font-semibold uppercase tracking-wide text-accent">Recommended</p>
      <h3 className="heading mt-1 text-2xl text-ink">{rec.label}</h3>
      <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span>{rec.kind_title}</span>
        <TierBadge tier={rec.fidelity} />
        {rec.verify_on_hardware ? <VerifyMark /> : null}
      </p>
      {rec.caveat ? (
        <p className="mt-3 border-l-2 border-pending pl-2 text-sm font-semibold text-pending" data-caveat>
          {rec.caveat}
        </p>
      ) : null}
      <p className="mt-3 text-ink-2">{rec.why}</p>
      <Facts
        className="mt-4"
        items={[
          ["Completes the task", runsText(rec.completed_runs, rec.n_runs)],
          ["Median cycle time", seconds(rec.cycle_time_s)],
          ["Tools", String(rec.tools.length)],
          ...(bestSingle ? ([["Best single off-the-shelf", runsText(bestSingle.completed_runs, bestSingle.n_runs)]] as [string, string][]) : []),
        ]}
      />

      <h4 className="heading mt-5 text-base">Tools</h4>
      <ul className="mt-2 grid gap-2 text-sm">
        {rec.tools.map((t) => (
          <li key={`${t.index}-${t.display_name}`}>
            <span className="flex flex-wrap items-center gap-2">
              <span className="num text-xs text-ink-3">Tool {t.index + 1}</span>
              <span className="text-ink">{t.display_name}</span>
              {t.family_title ? <span className="text-xs text-ink-3">{t.family_title}</span> : null}
              {t.fidelity ? <TierBadge tier={t.fidelity} /> : null}
            </span>
            {t.caveat && !stepCaveats.has(t.caveat) ? <span className="mt-1 block border-l-2 border-pending pl-2 text-xs text-pending">{t.caveat}</span> : null}
          </li>
        ))}
      </ul>
      {geometry ? <p className="num mt-2 text-xs text-ink-3">{geometry}</p> : null}

      <h4 className="heading mt-5 text-base">Per step</h4>
      <ol className="mt-2 grid gap-3 text-sm" data-rec-steps>
        {rec.steps.map((s) => (
          <li key={s.step_id} className="border-t border-rule pt-2">
            <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <span className="text-ink">{stepTitle(plan, s.step_id)}</span>
              <span className="text-xs text-ink-3">
                {humanize(s.primitive_used ?? s.primitive)}
                {s.primitive_used && s.primitive_used !== s.primitive ? ` (a ${s.primitive.replace(/_/g, " ")} step)` : ""} with tool {s.tool + 1}, {s.tool_label}
              </span>
              <span className="num text-ink">{pct(s.success_rate)}</span>
              <TierBadge tier={s.fidelity} />
              {s.verify_on_hardware ? <VerifyMark /> : null}
            </span>
            {s.caveats.length ? (
              <ul className="mt-1 grid gap-1 border-l-2 border-pending pl-2 text-xs text-pending" data-caveat>
                {s.caveats.map((c) =>
                  firstStep.get(c) === s.step_id ? (
                    <li key={c}>{c}</li>
                  ) : (
                    <li key={c}>The same caveat as {stepTitle(plan, firstStep.get(c)!)}.</li>
                  ),
                )}
              </ul>
            ) : null}
          </li>
        ))}
      </ol>

      <dl className="mt-5 grid grid-cols-1 gap-x-5 gap-y-0.5 text-sm sm:grid-cols-[auto_minmax(0,1fr)] sm:gap-y-1.5">
        {rec.runner_up_rigid ? (
          <>
            <dt className="text-ink-3 max-sm:mt-1.5">Rigid runner-up</dt>
            <dd className="text-ink">
              {rec.runner_up_rigid.label ?? "none"}
              <span className="text-ink-3"> ({rec.runner_up_rigid.basis})</span>
            </dd>
          </>
        ) : null}
        {vs && bestSingle ? (
          <>
            <dt className="text-ink-3 max-sm:mt-1.5">Against the best single</dt>
            <dd className="text-ink-2">
              {vs.same_setup ? (
                "The recommendation is itself the best single off-the-shelf tool."
              ) : (
                <>
                  {bestSingle.label}: {runsText(bestSingle.completed_runs, bestSingle.n_runs)}
                  {bestSingle.stops_at ? `, stops at ${stepTitle(plan, bestSingle.stops_at)}` : ""}. The recommendation completes{" "}
                  {vs.completion_delta >= 0 ? `${Math.round(vs.completion_delta * 100)} points more` : `${Math.round(-vs.completion_delta * 100)} points fewer`} of its runs
                  {vs.extra_tools > 0 ? ` with ${vs.extra_tools} more ${vs.extra_tools === 1 ? "tool" : "tools"}` : ""}
                  {bestSingle.cycle_time_s !== null && vs.cycle_delta_s !== null ? `, ${vs.cycle_delta_s >= 0 ? "+" : ""}${vs.cycle_delta_s.toFixed(1)} s per cycle` : ""}.
                </>
              )}
            </dd>
          </>
        ) : null}
      </dl>
      <p className="mt-4 text-[0.6875rem] text-ink-3">A simulation ranking under the stated conditions, not a hardware guarantee.</p>
    </article>
  );
}

function BlockerBars({ blocker }: { blocker: Blocker }) {
  const target = blocker.at_least ?? blocker.at_most;
  const peaks = blocker.families.map((f) => f.peak ?? 0);
  const scale = Math.max(target ?? 0, ...peaks, blocker.max_peak ?? 0) * 1.15 || 1;
  const u = blocker.unit ? ` ${blocker.unit}` : "";
  return (
    <div className="grid gap-1.5">
      <ul className="grid gap-2 text-sm" aria-hidden="true" data-blocker-bars>
        {blocker.families.map((f) => (
          <li key={f.family} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] items-center gap-3">
            <span className="truncate text-ink-2">{f.family_title}</span>
            <span className="relative h-4 border border-rule-strong">
              <span className="absolute inset-y-0 left-0 bg-ink-3" style={{ width: `${((f.peak ?? 0) / scale) * 100}%` }} />
              {target !== null ? <span className="absolute -inset-y-1 w-0.5 bg-fail" style={{ left: `${(target / scale) * 100}%` }} /> : null}
              <span className="num absolute inset-y-0 right-1 text-[0.6875rem] leading-4 text-ink">{f.peak !== null ? `${f.peak.toFixed(2)}${u}` : "n/a"}</span>
            </span>
          </li>
        ))}
      </ul>
      {target !== null ? (
        <p className="flex items-center gap-2 text-xs text-ink-3">
          <span className="inline-block h-3 w-0.5 bg-fail" aria-hidden="true" />
          The vertical mark is the target, {targetText(blocker)}; each bar is the peak a tool family reached.
        </p>
      ) : null}
    </div>
  );
}

/** The step nothing passes, and how far every tool family gets against its target. */
export function BlockerTable({ blocker, plan }: { blocker: Blocker; plan: Plan }) {
  const u = blocker.unit ? ` ${blocker.unit}` : "";
  return (
    <div className="grid gap-3" data-blocker>
      <p className="text-ink">
        {stepTitle(plan, blocker.step_id)} ({humanize(blocker.primitive).toLowerCase()}): peak {quantityName(blocker.quantity)}{" "}
        <span className="num">{blockerRange(blocker)}</span>.
      </p>
      <p className="text-sm text-ink-2">{blocker.text}</p>
      <BlockerBars blocker={blocker} />
      <div className="relative overflow-x-auto" role="region" aria-label="Peak per tool family" tabIndex={0}>
        <table className="w-full min-w-[560px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-y border-rule-strong text-xs text-ink-3">
              <th scope="col" className="py-2 pr-3 font-normal">
                Tool family
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-normal">
                Peak
              </th>
              <th scope="col" className="py-2 pr-3 text-right font-normal">
                Candidates, runs
              </th>
              <th scope="col" className="py-2 pr-3 font-normal">
                Best of the family
              </th>
              <th scope="col" className="py-2 font-normal">
                Failures
              </th>
            </tr>
          </thead>
          <tbody>
            {blocker.families.map((f) => (
              <tr key={f.family} className="border-b border-rule align-top">
                <th scope="row" className="py-2 pr-3 font-normal text-ink">
                  {f.family_title}
                </th>
                <td className="num whitespace-nowrap py-2 pr-3 text-right">
                  {f.peak !== null ? `${f.peak.toFixed(2)}${u}` : "n/a"}
                  <span className="sr-only"> against {targetText(blocker)}</span>
                </td>
                <td className="num py-2 pr-3 text-right">
                  {f.n_candidates}, {f.n_runs}
                </td>
                <td className="py-2 pr-3 text-xs text-ink-2">{f.best ?? "n/a"}</td>
                <td className="py-2 text-xs text-ink-2">{failureText(f.failure_modes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {blocker.conditions.length ? <p className="num text-xs text-ink-3">Conditions: {blocker.conditions.join(", ")}</p> : null}
    </div>
  );
}

/** The honest no-winner block: nothing completes the task, the closest setups and where they stop. */
export function NoWinnerBlock({ noWinner, plan }: { noWinner: NoWinner; plan: Plan }) {
  const reason = noWinner.reason.charAt(0).toUpperCase() + noWinner.reason.slice(1);
  return (
    <article className="border border-dashed border-rule-strong p-4 sm:p-5" aria-label="No recommendation" data-no-winner>
      <p className="text-xs font-semibold uppercase tracking-wide text-fail">No recommendation</p>
      <h3 className="heading mt-1 text-2xl text-ink">No setup completes this task</h3>
      <p className="mt-2 text-ink-2">{reason}.</p>
      {noWinner.closest.length ? (
        <>
          <h4 className="heading mt-5 text-base">Closest setups</h4>
          <ol className="mt-2 grid gap-3 text-sm">
            {noWinner.closest.map((c) => (
              <li key={c.setup_id} className="border-t border-rule pt-2" data-closest>
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-ink">{c.label}</span>
                  <span className="text-xs text-ink-3">{c.kind_title}</span>
                  <TierBadge tier={c.fidelity} />
                  <span className="num text-ink-2">{runsText(c.completed_runs, c.n_runs)}</span>
                  <span className="num text-xs text-ink-3">{massText(c.complexity.total_mass_kg)}</span>
                </span>
                <ul className="mt-1 grid gap-0.5 text-xs text-ink-2">
                  {stopsText(c, plan).map((s) => (
                    <li key={s}>Stops at {s}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ol>
        </>
      ) : null}
      {noWinner.blocker ? (
        <>
          <h4 className="heading mt-5 text-base">What blocks it</h4>
          <div className="mt-2">
            <BlockerTable blocker={noWinner.blocker} plan={plan} />
          </div>
        </>
      ) : null}
    </article>
  );
}

