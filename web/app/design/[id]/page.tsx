import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import DesignPreviewLazy from "@/components/design/DesignPreviewLazy";
import { PageHeader } from "@/components/PageHeader";
import { designChip, ProvenanceChips } from "@/components/ProvenanceChips";
import { TitleBlock } from "@/components/TitleBlock";
import { summarizeCertificate } from "@/lib/certificate";
import { certificateIdsByDesign, getBouts, getCertificate, getDesign, getDesigns } from "@/lib/data";
import { formatPercent } from "@/lib/format";
import { humanize, PARAM_LABELS } from "@/lib/labels";
import { dataUrl } from "@/lib/paths";
import type { Design, ParamSpec } from "@/lib/types";

export const dynamicParams = false;

export function generateStaticParams() {
  return getDesigns().designs.map((d) => ({ id: d.id }));
}

type Params = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const design = getDesign((await params).id);
  return design ? { title: `${humanize(design.name)} finger`, description: design.rationale ?? undefined } : {};
}

function formatValue(value: number | string, unit: string): string {
  if (typeof value === "string") return humanize(value);
  const digits = Number.isInteger(value) ? 0 : Math.abs(value) < 1 ? 2 : 1;
  return `${value.toFixed(digits)}${unit && unit !== "of length" && unit !== "tip/root" ? ` ${unit}` : ""}`;
}

/** Where a value sits in its allowed range, drawn as a dial-like gauge. */
function RangeGauge({ value, spec }: { value: number; spec: ParamSpec }) {
  if (spec.min === undefined || spec.max === undefined || spec.max <= spec.min) return null;
  const frac = Math.min(Math.max((value - spec.min) / (spec.max - spec.min), 0), 1);
  return (
    <div className="flex items-center gap-2" aria-hidden="true">
      <span className="num w-8 text-right text-[0.6875rem] text-ink-3">{spec.min}</span>
      <div className="relative h-3 flex-1 border-x border-rule-strong">
        <div className="absolute inset-x-0 top-1/2 h-px bg-rule-strong" />
        <div className="absolute top-0 h-3 w-0.5 -translate-x-1/2 bg-accent" style={{ left: `${frac * 100}%` }} />
      </div>
      <span className="num w-8 text-[0.6875rem] text-ink-3">{spec.max}</span>
    </div>
  );
}

function ParamTable({ design, specs }: { design: Design; specs: Record<string, ParamSpec> }) {
  const keys = Object.keys(design.params);
  return (
    <dl className="border-t border-rule-strong">
      {keys.map((key) => {
        const value = design.params[key]!;
        const meta = PARAM_LABELS[key] ?? { label: humanize(key), unit: "" };
        const spec = specs[key];
        return (
          <div key={key} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-b border-rule py-2.5 sm:grid-cols-[10rem_7rem_minmax(0,1fr)]">
            <dt className="text-sm text-ink" title={spec?.description}>
              {meta.label}
              <span className="num block text-[0.6875rem] text-ink-3">{key}</span>
            </dt>
            <dd className="num text-right text-sm text-ink sm:text-left">{formatValue(value, meta.unit)}</dd>
            <dd className="col-span-2 sm:col-span-1">
              {typeof value === "number" && spec ? (
                <RangeGauge value={value} spec={spec} />
              ) : spec?.options ? (
                <span className="flex flex-wrap gap-1.5 text-xs">
                  {spec.options.map((o) => (
                    <span
                      key={o}
                      className={o === value ? "bg-ink px-1.5 py-0.5 text-paper" : "border border-rule px-1.5 py-0.5 text-ink-3"}
                    >
                      {humanize(o)}
                    </span>
                  ))}
                </span>
              ) : null}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

const FILE_ROWS: { key: keyof Design["files"]; label: string; name: (d: Design) => string; what: string }[] = [
  { key: "stl_left", label: "Left finger", name: (d) => `${d.name}-finger-left.stl`, what: "STL, millimetres" },
  { key: "stl_right", label: "Right finger", name: (d) => `${d.name}-finger-right.stl`, what: "STL, millimetres, mirrored" },
  { key: "mount_plate", label: "Mount plate", name: (d) => `${d.name}-mount-plate.stl`, what: "STL, millimetres" },
  { key: "certificate_html", label: "Robustness certificate", name: (d) => `${d.name}-certificate.html`, what: "HTML" },
  { key: "glb", label: "Gripper preview", name: (d) => `${d.name}-gripper.glb`, what: "GLB, metres, Y-up" },
];

export default async function DesignPage({ params }: Params) {
  const design = getDesign((await params).id);
  if (!design) notFound();
  const { params: specs } = getDesigns();
  const bouts = getBouts().bouts.filter((b) => design.bouts.includes(b.id));
  const cert = design.certificate;
  const certId = certificateIdsByDesign().get(design.id);
  const published = certId ? getCertificate(certId) : undefined;
  const certSummary = published ? summarizeCertificate(published) : null;

  return (
    <>
      <p className="pt-8 text-sm">
        <Link href="/benchmark/" className="link">
          Benchmark
        </Link>
      </p>
      <PageHeader
        title={`${humanize(design.name)} finger`}
        lede={design.rationale ? <p>{design.rationale}</p> : undefined}
      >
        <ProvenanceChips chips={[designChip(design.provenance)]} />
      </PageHeader>

      <div className="mt-10 grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <section aria-labelledby="preview-title">
          <h2 id="preview-title" className="sr-only">
            3D preview
          </h2>
          {design.files.glb ? (
            <DesignPreviewLazy url={dataUrl(design.files.glb)} label={`3D preview of the ${design.name} finger pair`} />
          ) : (
            <p className="border border-dashed border-rule-strong p-6 text-sm text-ink-3">
              No 3D preview was exported for this design.
            </p>
          )}
          <dl className="mt-6 grid grid-cols-2 gap-px border border-rule bg-rule text-sm">
            {[
              ["Design id", design.id],
              ["Validation", design.valid ? "Valid" : "Invalid"],
              ["Printed mass", design.mass_printed_g !== null ? `${design.mass_printed_g.toFixed(1)} g per finger` : "n/a"],
              ["Simulated mass", design.mass_sim_g !== null ? `${design.mass_sim_g.toFixed(1)} g per finger` : "n/a"],
              ["Bounding box", design.bbox_mm ? `${design.bbox_mm.map((v) => v.toFixed(1)).join(" x ")} mm` : "n/a"],
              ["Collision boxes", design.n_boxes !== null ? `${design.n_boxes} per finger` : "n/a"],
            ].map(([k, v]) => (
              <div key={k} className="bg-paper px-3 py-2">
                <dt className="text-xs text-ink-3">{k}</dt>
                <dd className="num text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          {design.warnings.length > 0 || design.errors.length > 0 ? (
            <ul className="mt-4 grid gap-2 text-sm">
              {design.errors.map((e) => (
                <li key={e} className="border-l-2 border-fail pl-3 text-ink-2">
                  <span className="text-fail">Error: </span>
                  {e}
                </li>
              ))}
              {design.warnings.map((w) => (
                <li key={w} className="border-l-2 border-pending pl-3 text-ink-2">
                  <span className="text-pending">Note: </span>
                  {w}
                </li>
              ))}
            </ul>
          ) : null}
        </section>

        <section aria-labelledby="params-title">
          <h2 id="params-title" className="heading text-xl">
            Parameters
          </h2>
          <p className="mt-1 text-sm text-ink-3">
            The ten numbers a designer, human or model, chooses. The marker shows where each sits in its allowed range.
          </p>
          <div className="mt-4">
            <ParamTable design={design} specs={specs} />
          </div>
        </section>
      </div>

      <section className="mt-14" aria-labelledby="pack-title">
        <h2 id="pack-title" className="heading text-xl">
          Print pack
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          The finger mesh is the union of exactly the boxes the simulator collides with, so what you print is what was
          tested.
        </p>
        <ul className="mt-4 border-t border-rule-strong text-sm">
          {FILE_ROWS.map((row) => {
            const file = design.files[row.key];
            return (
              <li key={row.key} className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule py-2.5">
                <span>
                  <span className="text-ink">{row.label}</span>
                  <span className="ml-2 text-xs text-ink-3">{row.what}</span>
                </span>
                {file ? (
                  <a className="link" href={dataUrl(file)} download={row.name(design)}>
                    Download {row.name(design)}
                  </a>
                ) : (
                  <span className="text-pending">Not generated yet</span>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="mt-14" aria-labelledby="cert-title">
        <h2 id="cert-title" className="heading text-xl">
          Robustness certificate
        </h2>
        {published && certSummary ? (
          <>
            <p className="mt-1 text-sm text-ink-3">
              Each check reruns the grasp with one disturbance on {certSummary.objects} training objects; the count is how
              many episodes still held.
            </p>
            <ul className="mt-4 border-t border-rule-strong text-sm">
              {(
                [
                  ["Placement error", "Object up to 3 mm off target, along x and along y", certSummary.category.placement],
                  ["Shake during hold", "Palm shakes along x, y and z while holding the object", certSummary.category.shake],
                  ["Lateral push", "Sideways push at the object's top, along x and along y", certSummary.category.push],
                ] as const
              ).map(([name, condition, c]) => (
                <li key={name} className="grid gap-x-6 gap-y-0.5 border-b border-rule py-2.5 sm:grid-cols-[14rem_minmax(0,1fr)_8rem]">
                  <span className="text-ink">{name}</span>
                  <span className="text-ink-2">{condition}</span>
                  <span className="num text-ink sm:text-right">
                    {c.passed} of {c.total}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-sm">
              <Link className="link" href={`/certificate/${published.id}/`}>
                Open the full certificate: placement-error curve, every shake and push, provenance
              </Link>
            </p>
          </>
        ) : cert ? (
          <>
            <p className="mt-1 text-sm text-ink-3">
              {cert.status === "issued"
                ? "Each check reruns the grasp with one disturbance; the rate is the share of episodes that still held."
                : "These checks have not run for this design yet. Each reruns the grasp with one disturbance."}
            </p>
            <ul className="mt-4 border-t border-rule-strong text-sm">
              {cert.checks.map((c) => (
                <li key={c.name} className="grid gap-x-6 gap-y-0.5 border-b border-rule py-2.5 sm:grid-cols-[14rem_minmax(0,1fr)_8rem]">
                  <span className="text-ink">{c.name}</span>
                  <span className="text-ink-2">{c.condition}</span>
                  <span className={`num sm:text-right ${c.success_rate === null ? "text-pending" : "text-ink"}`}>
                    {c.success_rate === null ? "Pending" : `${formatPercent(c.success_rate)} of ${c.episodes}`}
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p className="mt-2 text-sm text-ink-3">No certificate is planned for this design.</p>
        )}
      </section>

      {bouts.length > 0 ? (
        <section className="mt-14" aria-labelledby="bouts-title">
          <h2 id="bouts-title" className="heading text-xl">
            Appears in
          </h2>
          <ul className="mt-3 grid gap-1.5 text-sm">
            {bouts.map((b) => (
              <li key={b.id}>
                <Link href={`/bout/${b.id}/`} className="link">
                  {b.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <TitleBlock title={`${humanize(design.name)} finger`} drawing={`/design/${design.id}`} sheet="Detail of sheet 3" />
    </>
  );
}
