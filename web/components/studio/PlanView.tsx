import Link from "next/link";

import { humanize, paramText, type Plan, type PlanStep, PLAN_RESULT_LABELS } from "./schema";
import { TierBadge, VerifyMark } from "./ui";

function limitsText(c: PlanStep["constraints"]): string[] {
  const out: string[] = [];
  if (c.max_force_n !== null) out.push(`force at most ${c.max_force_n} N`);
  if (c.max_torque_nm !== null) out.push(`torque at most ${c.max_torque_nm} N m`);
  if (c.precision_mm !== null) out.push(`within ${c.precision_mm} mm`);
  if (c.precision_deg !== null) out.push(`within ${c.precision_deg} deg`);
  return out;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-ink-3">{label}</dt>
      <dd className="min-w-0 text-ink">{children}</dd>
    </>
  );
}

/** The frozen plan the search ran on: objects and steps, read only. */
export function PlanView({ plan }: { plan: Plan }) {
  const objects = new Map(plan.objects.map((o) => [o.name, o]));
  const name = (n: string | null) => (n === null ? null : (objects.get(n)?.label ?? humanize(n)));
  const how = [
    plan.source === "recorded" ? "a recorded Task Engineer plan" : `a ${plan.source} plan`,
    plan.model ? `model ${plan.model}` : null,
    PLAN_RESULT_LABELS[plan.result] ?? humanize(plan.result).toLowerCase(),
    plan.calls !== null ? `${plan.calls} ${plan.calls === 1 ? "call" : "calls"}` : null,
  ].filter(Boolean);
  return (
    <div className="grid gap-6" data-plan>
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <div className="min-w-0 max-w-[80ch]">
          <p className="heading text-lg text-ink">{plan.title}</p>
          {plan.description ? <p className="mt-1 text-ink-2">{plan.description}</p> : null}
          <p className="num mt-1 text-xs text-ink-3">{how.join(" · ")}</p>
        </div>
        <Link href="/studio/new/" className="link text-sm" data-edit-plan>
          Edit this plan on /studio/new/
        </Link>
      </div>

      {plan.layout_moves.length || plan.assumptions.length || plan.clarifications.length || plan.unsupported.length ? (
        <div className="grid gap-3 text-sm sm:grid-cols-2">
          {plan.assumptions.length ? (
            <div>
              <p className="text-xs text-ink-3">The planner assumed</p>
              <ul className="mt-1 grid list-disc gap-1 pl-5 text-ink-2">
                {plan.assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {plan.layout_moves.length ? (
            <div>
              <p className="text-xs text-ink-3">Layout fix</p>
              <ul className="mt-1 grid list-disc gap-1 pl-5 text-ink-2">
                {plan.layout_moves.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {plan.clarifications.length ? (
            <div>
              <p className="text-xs text-ink-3">Open questions, answered by assumption</p>
              <ul className="mt-1 grid gap-1 text-ink-2">
                {plan.clarifications.map((c) => (
                  <li key={c.question}>
                    {c.question} <span className="text-ink-3">Assumed: {c.assumed}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {plan.unsupported.length ? (
            <div>
              <p className="text-xs text-ink-3">Not supported</p>
              <ul className="mt-1 grid gap-1 text-ink-2">
                {plan.unsupported.map((u) => (
                  <li key={u.what}>
                    {u.what}: {u.why}
                    {u.nearest ? <span className="text-ink-3"> Nearest primitive: {humanize(u.nearest)}.</span> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}

      <div>
        <h3 className="heading text-base">Objects</h3>
        <ul className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {plan.objects.map((o) => (
            <li key={o.name} className="border border-rule p-2.5 text-sm" data-object={o.name}>
              <p className="text-ink">
                {o.label} <span className="text-xs text-ink-3">{o.kind_title}</span>
              </p>
              <p className="num mt-1 text-xs text-ink-2">
                {Object.entries(o.params)
                  .map(([k, v]) => paramText(k, v).join(" "))
                  .join(" · ")}
              </p>
              <p className="num mt-0.5 text-xs text-ink-3">
                at ({o.pose.x_mm.toFixed(0)}, {o.pose.y_mm.toFixed(0)}) mm{o.pose.yaw_deg ? `, yaw ${o.pose.yaw_deg.toFixed(0)} deg` : ""}
                {o.ref ? `, relative to ${name(o.ref)}` : ""}
              </p>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="heading text-base">Steps</h3>
        <ol className="mt-2 grid gap-3">
          {plan.steps.map((s, i) => {
            const limits = limitsText(s.constraints);
            return (
              <li key={s.id} className="border border-rule-strong bg-paper p-3 sm:p-4" data-step={s.id}>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="num grid size-6 place-items-center bg-ink text-xs text-paper" aria-hidden="true">
                    {i + 1}
                  </span>
                  <h4 className="heading text-base text-ink">
                    <span className="sr-only">Step {i + 1}: </span>
                    {humanize(s.id)}
                  </h4>
                  <span className="text-sm text-ink-2">{s.primitive_title}</span>
                  <TierBadge tier={s.fidelity} />
                  {s.fidelity === "approximation" ? <VerifyMark /> : null}
                </div>
                {s.description ? <p className="mt-2 text-sm text-ink-2">{s.description}</p> : null}
                <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-5 gap-y-1 text-sm">
                  <Row label="Object">
                    {name(s.object)}
                    {s.handle ? <span className="text-ink-3"> by its {s.handle.replace(/_/g, " ")}</span> : null}
                  </Row>
                  {s.target ? <Row label="Target">{name(s.target)}</Row> : null}
                  {Object.keys(s.params).length ? (
                    <Row label="Parameters">
                      <span className="num">
                        {Object.entries(s.params)
                          .map(([k, v]) => paramText(k, v).join(" "))
                          .join(" · ")}
                      </span>
                    </Row>
                  ) : null}
                  <Row label="Limits">{limits.length ? limits.join(", ") : <span className="text-ink-3">none stated</span>}</Row>
                  <Row label="Success">
                    {s.success.length ? (
                      <ul className="grid gap-0.5">
                        {s.success.map((c) => (
                          <li key={`${c.quantity}-${c.text}`}>{c.text}</li>
                        ))}
                      </ul>
                    ) : (
                      <span className="text-ink-3">the primitive&apos;s own check</span>
                    )}
                  </Row>
                  {s.after.length ? <Row label="After">{s.after.map(humanize).join(", ")}</Row> : null}
                  {s.hold_after ? <Row label="Holds">keeps holding the object after this step</Row> : null}
                </dl>
                {s.caveats.length ? (
                  <ul className="mt-3 grid gap-1 border-l-2 border-pending pl-2 text-sm text-pending" data-caveat>
                    {s.caveats.map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                ) : null}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
