import {
  actuationText,
  type Card,
  type CertSummary,
  FAMILY_TITLES,
  type FamilyCard as FamilyCardData,
  FLAG_LABELS,
  massText,
  OUTCOME_LABELS,
  PRODUCT_MODE_LABEL,
  pct,
  type Requirement,
  REQUIREMENT_LABELS,
  score,
  toleranceText,
} from "@/lib/effectors";

import { FidelityBadge } from "./FidelityBadge";

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-ink-3">{label}</dt>
      <dd className="num text-ink">{children}</dd>
    </>
  );
}

/** The recommended effector (or a runner-up): what it is, its certificate, and why it ranks here. */
export function RecommendationCard({
  card,
  requirement,
  title,
  basis,
}: {
  card: Card;
  requirement: Requirement;
  title: string;
  basis?: string | null;
}) {
  const cert = card.certificate;
  const certScore = cert ? (requirement === "held" ? cert.held_score : cert.registered_score) : null;
  const otherScore = cert ? (requirement === "held" ? cert.registered_score : cert.held_score) : null;
  const other: Requirement = requirement === "held" ? "registered" : "held";
  return (
    <article className="border border-rule-strong bg-paper p-4 sm:p-5" aria-label={title} data-card={title}>
      <p className="text-xs font-semibold uppercase tracking-wide text-accent">{title}</p>
      <h3 className="heading mt-1 text-2xl text-ink">{card.effector.display_name}</h3>
      <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-ink-2">
        <span>{FAMILY_TITLES[card.effector.family]}</span>
        <FidelityBadge fidelity={card.effector.fidelity} />
        <span className="text-ink-3">primitive {card.effector.primitive}</span>
      </p>
      {card.caveat ? (
        <p className="mt-3 border-l-2 border-pending pl-2 text-sm font-semibold text-pending" data-caveat>
          {card.caveat}
        </p>
      ) : null}
      {card.effector.note ? (
        <p className="mt-3 border-l-2 border-pending pl-2 text-sm text-pending" data-note>
          {card.effector.note}
        </p>
      ) : null}
      {basis && basis !== "eligible" ? <p className="mt-2 text-sm text-ink-3">Named as {basis}.</p> : null}
      {card.why ? <p className="mt-3 text-ink-2">{card.why}</p> : null}
      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
        <Fact label={`Certificate (${REQUIREMENT_LABELS[requirement].toLowerCase()})`}>
          {score(certScore)} {cert ? <span className="text-ink-3">({requirement === "held" ? cert.held : cert.registered} tests)</span> : null}
        </Fact>
        <Fact label={`Certificate (${REQUIREMENT_LABELS[other].toLowerCase()})`}>{score(otherScore)}</Fact>
        <Fact label="Placement tolerance">{toleranceText(card.tolerance_mm)}</Fact>
        <Fact label="Screen">
          {pct(card.screen.registered)} registered, {pct(card.screen.held)} held of {card.screen.n}
        </Fact>
        <Fact label="Actuators">{card.n_actuators ?? "n/a"}</Fact>
        <Fact label="Mass">{massText(card.mass_kg)}</Fact>
        <Fact label="Declared limits">{actuationText(card.actuation).join("; ") || "n/a"}</Fact>
      </dl>
      {card.effector.hypothesis ? <p className="mt-3 text-sm text-ink-3">Design hypothesis: {card.effector.hypothesis}</p> : null}
      <p className="mt-4 text-[0.6875rem] text-ink-3">{PRODUCT_MODE_LABEL}</p>
    </article>
  );
}

/** One family's card: generic name, fidelity, whether this build has it, and how it did here. */
export function FamilyCard({ f }: { f: FamilyCardData }) {
  return (
    <article
      className={`flex flex-col border p-3 ${f.available ? "border-rule-strong" : "border-dashed border-rule"}`}
      aria-label={f.title}
      data-family={f.family}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className={`heading text-base ${f.available ? "text-ink" : "text-ink-3"}`}>{f.title}</h3>
        <FidelityBadge fidelity={f.fidelity} />
      </div>
      <p className="mt-1 text-sm text-ink-2">{f.blurb}</p>
      <p className="mt-2 text-sm">
        {f.available ? (
          f.best ? (
            <>
              Best here: <span className="text-ink">{f.best.display_name}</span>
              {f.best.screen ? (
                <span className="num text-ink-3">
                  {" "}
                  · screen {pct(f.best.screen.registered)} registered, {pct(f.best.screen.held)} held
                </span>
              ) : null}
              {f.best.note ? (
                <span className="mt-1 block text-xs text-pending" data-note>
                  {f.best.note}
                </span>
              ) : null}
            </>
          ) : (
            <span className="text-ink-3">No candidate of this family could be screened on this part.</span>
          )
        ) : (
          <span className="text-ink-3">Not built yet: no result.</span>
        )}
      </p>
      {f.available ? (
        <p className="num mt-1 text-xs text-ink-3">
          {f.candidates} candidates, {f.screened} screened{f.invalid ? `, ${f.invalid} could not be built` : ""}
        </p>
      ) : null}
      <p className="mt-2 text-xs text-ink-3">{f.caveat}</p>
      <p className="mt-auto pt-2 text-[0.6875rem] text-ink-3">{f.label}</p>
    </article>
  );
}

function Cell({ held, registered, mode, flags }: { held: boolean; registered: boolean; mode: string | null; flags: string[] }) {
  const ok = registered;
  const why = mode ? OUTCOME_LABELS[mode] ?? mode : held ? "held" : "failed";
  const text = registered ? "registered" : held ? `held, not registered${flags.length ? ` (${flags.map((x) => FLAG_LABELS[x] ?? x).join(", ")})` : ""}` : why;
  return (
    <span
      className={`inline-grid size-5 place-items-center border ${ok ? "border-pass bg-pass/15 text-pass" : held ? "border-pending text-pending" : "border-fail text-fail"}`}
      title={text}
    >
      <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
        <path d={ok ? "M1.5 6.5l3 3 6-7" : held ? "M2 6h8" : "M2 2l8 8M10 2l-8 8"} fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <span className="sr-only">{text}</span>
    </span>
  );
}

/** A certificate's 18 tests: placement errors per axis, then the shakes and pushes. */
export function CertificateMatrix({ cert, name }: { cert: CertSummary; name: string }) {
  const placement = cert.tests.filter((t) => t.category === "placement");
  const nominal = placement.find((t) => t.name === "nominal");
  const errors = [-3, -2, -1, 0, 1, 2, 3];
  const find = (axis: "x" | "y", e: number) =>
    e === 0 ? nominal : placement.find((t) => t.name === `offset_${axis}_${e > 0 ? "+" : ""}${e}mm`);
  const disturb = cert.tests.filter((t) => t.category !== "placement");
  return (
    <figure className="min-w-0" data-certificate={name}>
      <figcaption className="text-sm">
        <span className="text-ink">{name}</span>
        <span className="num ml-2 text-xs text-ink-3">
          registered {score(cert.registered_score)} ({cert.registered}) · held {score(cert.held_score)} ({cert.held}) · reference{" "}
          {cert.reference.replace("_", " ")}
        </span>
      </figcaption>
      <div className="mt-2 overflow-x-auto">
        <table className="border-collapse text-sm">
          <thead>
            <tr className="text-xs text-ink-3">
              <th scope="col" className="pr-3 text-left font-normal">
                Axis
              </th>
              {errors.map((e) => (
                <th key={e} scope="col" className="num px-1 text-center font-normal">
                  {e > 0 ? `+${e}` : e}
                </th>
              ))}
              <th scope="col" className="num pl-3 text-left font-normal">
                tol. reg / held
              </th>
            </tr>
          </thead>
          <tbody>
            {(["x", "y"] as const).map((axis) => (
              <tr key={axis}>
                <td className="num pr-3 text-xs text-ink-3">{axis === "x" ? "across" : "along"}</td>
                {errors.map((e) => {
                  const t = find(axis, e);
                  return (
                    <td key={e} className="px-1 py-0.5 text-center">
                      {t ? <Cell held={t.held} registered={t.registered} mode={t.failure_mode} flags={t.registration_flags} /> : "·"}
                    </td>
                  );
                })}
                <td className="num pl-3 text-xs">
                  {cert.registered_tolerance_mm[axis] ?? "none"} / {cert.held_tolerance_mm[axis] ?? "none"} mm
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-2">
        {disturb.map((t) => (
          <li key={t.name} className="flex items-center gap-1">
            <Cell held={t.held} registered={t.registered} mode={t.failure_mode} flags={t.registration_flags} />
            <span className="num">{t.name.replace(/_/g, " ")}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
