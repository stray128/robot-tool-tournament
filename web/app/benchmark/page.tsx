import type { Metadata } from "next";
import Link from "next/link";

import { OutcomeTag } from "@/components/Outcome";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getBouts, getDesigns } from "@/lib/data";
import { formatInt } from "@/lib/format";
import { humanize, objectLabel } from "@/lib/labels";
import type { Bout } from "@/lib/types";

export const metadata: Metadata = {
  title: "Benchmark",
  description: "Every recorded bout: finger designs tested against the same object, gripper and controller.",
};

const KIND_LABELS: Record<Bout["kind"], string> = {
  gate: "Simulator check",
  training: "Training round",
  heldout: "Held-out evaluation",
  live: "Live bout",
};

function objectsOf(bout: Bout): string {
  const set = new Set(
    bout.lanes.map((l) =>
      l.object.kind === "part"
        ? objectLabel(l.object)
        : `${l.object.size_mm.map((v) => Number(v.toFixed(1))).join(" x ")} mm ${l.object.kind}`,
    ),
  );
  return [...set].join(", ");
}

export default function BenchmarkPage() {
  const { bouts } = getBouts();
  const { designs } = getDesigns();
  const nebius = bouts.filter((b) => b.compute.kind === "nebius").length;

  return (
    <>
      <PageHeader
        title="Benchmark"
        lede={
          <p>
            A bout puts several finger designs on the same object, gripper and controller, so the only thing that
            changes is the fingers. Each lane replays in your browser from the simulator&apos;s own pose log.
          </p>
        }
      >
        <p className="num text-sm text-ink-3">
          {bouts.length} bouts recorded, {nebius} on Nebius, {formatInt(bouts.reduce((n, b) => n + b.lanes.length, 0))} lanes
        </p>
      </PageHeader>

      <section className="mt-10" aria-labelledby="bouts-title">
        <h2 id="bouts-title" className="heading text-xl">
          Bouts
        </h2>
        <ul className="mt-4 border-t border-rule-strong">
          {bouts.map((bout) => (
            <li key={bout.id} className="grid gap-x-6 gap-y-3 border-b border-rule py-5 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div className="min-w-0">
                <p className="text-xs text-ink-3">
                  {KIND_LABELS[bout.kind]}
                  <span className="mx-2 text-rule-strong" aria-hidden="true">
                    /
                  </span>
                  {bout.compute.kind === "nebius" ? "Nebius Serverless Job" : "Local run"}
                </p>
                <h3 className="heading mt-1 text-lg">
                  <Link href={`/bout/${bout.id}/`} className="link">
                    {bout.title}
                  </Link>
                </h3>
                <p className="mt-1 text-sm text-ink-2">{objectsOf(bout)}</p>
              </div>
              <ol className="grid content-start gap-1.5 text-sm" aria-label={`Lanes of ${bout.title}`}>
                {bout.lanes.map((lane) => (
                  <li key={lane.id} className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-dashed border-rule pb-1.5 last:border-0">
                    <span className="text-ink">{lane.label}</span>
                    <OutcomeTag outcome={lane.outcome} />
                  </li>
                ))}
              </ol>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-14" aria-labelledby="designs-title">
        <h2 id="designs-title" className="heading text-xl">
          Finger designs
        </h2>
        <p className="prose-measure mt-2 text-ink-2">
          Every design on file, with its parameters, rationale and print files.
        </p>
        <ul className="mt-4 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-3">
          {designs.map((d) => (
            <li key={d.id} className="bg-paper p-4">
              <p className="num text-xs text-ink-3">{d.id}</p>
              <h3 className="heading mt-1 text-lg">
                <Link href={`/design/${d.id}/`} className="link">
                  {humanize(d.name)}
                </Link>
              </h3>
              <p className="mt-1 text-sm text-ink-2">
                {d.origin === "manual" ? "Hand-designed baseline" : `Proposed by ${d.origin}`}
                {d.mass_printed_g !== null ? `, ${d.mass_printed_g.toFixed(1)} g printed` : ""}
              </p>
              {d.rationale ? <p className="mt-2 text-sm text-ink-2">{d.rationale}</p> : null}
            </li>
          ))}
        </ul>
      </section>

      <TitleBlock title="Benchmark" drawing="/benchmark" sheet="3 of 6" />
    </>
  );
}
