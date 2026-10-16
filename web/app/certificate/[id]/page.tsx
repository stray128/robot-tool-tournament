import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { PlacementCurve } from "@/components/charts/PlacementCurve";
import { PageHeader } from "@/components/PageHeader";
import { computeChip, ProvenanceChips } from "@/components/ProvenanceChips";
import { TitleBlock } from "@/components/TitleBlock";
import {
  type CertificateDoc,
  describeDisturbance,
  disturbanceRows,
  type ObjectCertificate,
  placementCurve,
  summarizeCertificate,
} from "@/lib/certificate";
import { getCertificate, getCertificates, getDesign } from "@/lib/data";
import { formatPercent, formatUtc } from "@/lib/format";
import { FAILURE_LABELS, humanize } from "@/lib/labels";
import { dataUrl } from "@/lib/paths";
import type { FailureMode } from "@/lib/types";

export const dynamicParams = false;

export function generateStaticParams() {
  return getCertificates().map((c) => ({ id: c.id }));
}

type Params = { params: Promise<{ id: string }> };

function designTitle(doc: CertificateDoc): string {
  const design = getDesign(doc.design_id);
  return `${humanize(design?.name ?? doc.design_name ?? doc.design_id)} finger`;
}

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const doc = getCertificate((await params).id);
  return doc ? { title: `Certificate: ${designTitle(doc)}`, description: "Placement error, shake and push results for one finger design." } : {};
}

function Mark({ ok, mode = null }: { ok: boolean; mode?: string | null }) {
  return ok ? (
    <span className="inline-flex items-center gap-1 text-pass">
      <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
        <path d="M1.5 6.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      Held
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-fail">
      <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
        <path d="M2 2l8 8M10 2l-8 8" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      {failureLabel(mode) || "Failed"}
    </span>
  );
}

function failureLabel(mode: string | null): string {
  if (!mode) return "";
  return FAILURE_LABELS[mode as FailureMode] ?? humanize(mode);
}

/** One object's placement results as a strip of cells, error by error, per axis. */
function PlacementMatrix({ certs }: { certs: ObjectCertificate[] }) {
  const errors = Array.from(
    new Set(certs.flatMap((c) => c.tests.filter((t) => t.category === "placement").map((t) => Number(t.params.error_mm)))),
  ).sort((a, b) => a - b);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-rule-strong text-xs text-ink-3">
            <th scope="col" className="py-1.5 pr-3 text-left font-normal">
              Object
            </th>
            <th scope="col" className="py-1.5 pr-3 text-left font-normal">
              Axis
            </th>
            {errors.map((e) => (
              <th key={e} scope="col" className="num px-1 py-1.5 text-center font-normal">
                {e > 0 ? `+${e}` : e}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {certs.flatMap((c) =>
            (["x", "y"] as const).map((axis) => (
              <tr key={`${c.object}-${axis}`} className={axis === "y" ? "border-b border-rule" : ""}>
                {axis === "x" ? (
                  <th scope="rowgroup" rowSpan={2} className="py-1 pr-3 text-left align-top font-normal text-ink">
                    {c.object}
                    <span className="num block text-[0.6875rem] text-ink-3">setting k={c.controller_setting}</span>
                  </th>
                ) : null}
                <td className="num py-1 pr-3 text-xs text-ink-3">{axis}</td>
                {errors.map((e) => {
                  const t = c.tests.find(
                    (x) => x.category === "placement" && Number(x.params.error_mm) === e && (e === 0 || x.params.axis === axis),
                  );
                  if (!t) return <td key={e} className="px-1 py-1 text-center text-ink-3">·</td>;
                  return (
                    <td key={e} className="px-1 py-1 text-center">
                      <span
                        className={`inline-grid size-5 place-items-center border ${t.success ? "border-pass bg-pass/15 text-pass" : "border-fail text-fail"}`}
                        title={t.success ? "held" : failureLabel(t.failure_mode)}
                      >
                        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
                          <path d={t.success ? "M1.5 6.5l3 3 6-7" : "M2 2l8 8M10 2l-8 8"} fill="none" stroke="currentColor" strokeWidth="2" />
                        </svg>
                        <span className="sr-only">{t.success ? "held" : `dropped: ${failureLabel(t.failure_mode)}`}</span>
                      </span>
                    </td>
                  );
                })}
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

function DisturbanceTable({ certs, category }: { certs: ObjectCertificate[]; category: "shake" | "push" }) {
  const rows = disturbanceRows(certs, category);
  const held = rows.filter((r) => r.success).length;
  const shared = describeDisturbance(rows);
  const heights = new Set(rows.map((r) => r.push_height_mm).filter((h) => h !== null));
  return (
    <>
      <p className="mt-1 text-sm text-ink-3">
        <span className="num text-ink">
          {held} of {rows.length}
        </span>{" "}
        held. {shared ? <span>Each: {shared}.</span> : <span>The settings differ by object; see the certificate data.</span>}
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[440px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-rule-strong text-xs text-ink-3">
              <th scope="col" className="py-1.5 pr-3 font-normal">
                Object
              </th>
              <th scope="col" className="py-1.5 pr-3 font-normal">
                Axis
              </th>
              <th scope="col" className="py-1.5 pr-3 font-normal">
                Outcome
              </th>
              <th scope="col" className="py-1.5 text-right font-normal">
                Held for
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.object}-${r.axis}`} className="border-b border-rule">
                <th scope="row" className="py-1.5 pr-3 font-normal text-ink">
                  {r.object}
                  {heights.size > 1 && r.push_height_mm !== null ? (
                    <span className="num block text-[0.6875rem] text-ink-3">{r.push_height_mm} mm above centre of mass</span>
                  ) : null}
                </th>
                <td className="num py-1.5 pr-3 text-ink-2">{r.axis}</td>
                <td className="py-1.5 pr-3">
                  <Mark ok={r.success} mode={r.failure_mode} />
                </td>
                <td className="num py-1.5 text-right text-ink-2">{r.hold_time_s.toFixed(2)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default async function CertificatePage({ params }: Params) {
  const doc = getCertificate((await params).id);
  if (!doc) notFound();
  const design = getDesign(doc.design_id);
  const summary = summarizeCertificate(doc);
  const curves = { x: placementCurve(doc.certificates, "x"), y: placementCurve(doc.certificates, "y") };
  const bothTol =
    summary.tolerance_mm.x !== null && summary.tolerance_mm.y !== null ? Math.min(summary.tolerance_mm.x, summary.tolerance_mm.y) : null;
  const first = doc.certificates[0]!;
  const settings = first.settings as {
    placement_errors_mm?: number[];
    post_perturb_hold_s?: number;
    jitter?: string;
    seed?: number;
  };
  const title = designTitle(doc);
  const tol = (v: number | null) => (v === null ? "not defined" : `±${v} mm`);
  const undefinedTol = summary.tolerance_mm.x === null || summary.tolerance_mm.y === null;

  return (
    <>
      <p className="pt-8 text-sm">
        {design ? (
          <Link href={`/design/${design.id}/`} className="link">
            {title}
          </Link>
        ) : (
          <span className="text-ink-3">{title}</span>
        )}
      </p>
      <PageHeader
        title="Robustness certificate"
        lede={
          <p>
            How much the {title.toLowerCase()} tolerates before its grasp fails: the object placed up to{" "}
            {Math.max(...(settings.placement_errors_mm ?? [3]))} mm off target, a shake while it is held and a sideways
            push.{" "}
            {settings.jitter === "off"
              ? "Every episode is deterministic: seed jitter is off, so each condition is exactly as stated."
              : `Seed jitter: ${settings.jitter ?? "not recorded"}, so conditions vary slightly around the values stated.`}
          </p>
        }
      >
        <ProvenanceChips chips={doc.provenance ? [computeChip(doc.provenance)] : []} />
        {!doc.provenance ? <p className="text-sm text-pending">This certificate file does not record where it ran.</p> : null}
      </PageHeader>

      <dl className="mt-8 grid grid-cols-2 gap-px border border-rule bg-rule text-sm sm:grid-cols-5">
        {(
          [
            ["Score", formatPercent(summary.score)],
            ["Checks held", `${summary.passed} of ${summary.episodes}`],
            ["Objects", String(summary.objects)],
            ["Tolerance x", tol(summary.tolerance_mm.x)],
            ["Tolerance y", tol(summary.tolerance_mm.y)],
          ] as const
        ).map(([k, v]) => (
          <div key={k} className="bg-paper px-3 py-2.5">
            <dt className="text-xs text-ink-3">{k}</dt>
            <dd className="num mt-0.5 text-lg text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-ink-3">
        Score: the mean over objects of each object&apos;s placement, shake and push pass rates. Tolerance: the largest
        error that held on every object, in both directions
        {undefinedTol ? "; not defined here because at least one object is not held even at 0 mm" : ""}.
      </p>

      <section className="mt-12" aria-labelledby="placement-title">
        <h2 id="placement-title" className="heading text-xl">
          Placement error
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          Share of the {summary.objects} objects still held when the object sits off target by the given distance.{" "}
          {summary.category.placement.passed} of {summary.category.placement.total} placement episodes held.
        </p>
        <div className="mt-4 border border-rule bg-paper/70 px-3 py-4 sm:px-5">
          <PlacementCurve curves={curves} toleranceMm={bothTol} />
        </div>
        <details className="mt-4 border border-rule px-4 py-3 text-sm">
          <summary className="cursor-pointer text-ink">Every placement episode, object by object</summary>
          <p className="mt-2 text-xs text-ink-3">
            Setting k is the controller setting used for that object. Each object gets the setting whose undisturbed
            grasp worked best, and every check on that object then uses it. A different setting can tolerate less.
          </p>
          <div className="mt-3">
            <PlacementMatrix certs={doc.certificates} />
          </div>
        </details>
      </section>

      <div className="mt-12 grid grid-cols-[minmax(0,1fr)] gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section aria-labelledby="shake-title">
          <h2 id="shake-title" className="heading text-xl">
            Shake during the hold
          </h2>
          <DisturbanceTable certs={doc.certificates} category="shake" />
        </section>
        <section aria-labelledby="push-title">
          <h2 id="push-title" className="heading text-xl">
            Sideways push
          </h2>
          <DisturbanceTable certs={doc.certificates} category="push" />
        </section>
      </div>

      <section className="mt-12" aria-labelledby="prov-title">
        <h2 id="prov-title" className="heading text-xl">
          Provenance
        </h2>
        <dl className="mt-3 grid gap-px border border-rule bg-rule text-sm sm:grid-cols-2">
          {(
            [
              ["Design id", doc.design_id],
              ["Task", `${first.task} (digest ${first.task_digest})`],
              ["Issued", doc.issued_at ? formatUtc(doc.issued_at) : "not recorded"],
              ["Seed jitter", settings.jitter ?? "not recorded"],
              ["Seed", settings.seed !== undefined ? String(settings.seed) : "not recorded"],
              ["Hold after a disturbance", settings.post_perturb_hold_s !== undefined ? `${settings.post_perturb_hold_s} s` : "not recorded"],
              ...(doc.provenance?.kind === "local" && doc.provenance.note ? ([["Run note", doc.provenance.note]] as const) : []),
            ] as const
          ).map(([k, v]) => (
            <div key={k} className="bg-paper px-3 py-2">
              <dt className="text-xs text-ink-3">{k}</dt>
              <dd className="num break-words text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        <ul className="mt-4 grid gap-1.5 text-sm">
          <li>
            <a className="link" href={dataUrl(`certificates/${doc.id}.json`)} download={`certificate-${doc.id}.json`}>
              Download the certificate data (JSON)
            </a>
          </li>
          {design?.files.certificate_html ? (
            <li>
              <a className="link" href={dataUrl(design.files.certificate_html)} download={`${design.name}-certificate.html`}>
                Download the printable certificate (HTML)
              </a>
            </li>
          ) : null}
          {design ? (
            <li>
              <Link className="link" href={`/design/${design.id}/`}>
                The design, its parameters and print files
              </Link>
            </li>
          ) : null}
        </ul>
      </section>

      <TitleBlock title={`Certificate: ${title}`} drawing={`/certificate/${doc.id}`} sheet="Detail of sheet 3" />
    </>
  );
}
