import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { OutcomeTag } from "@/components/Outcome";
import { PageHeader } from "@/components/PageHeader";
import { computeChip, ProvenanceChips, uniqueDesignChips } from "@/components/ProvenanceChips";
import BoutPlayer from "@/components/replay/BoutPlayer";
import { TitleBlock } from "@/components/TitleBlock";
import { getBout, getBouts, getDesigns } from "@/lib/data";
import { formatDuration, formatInt } from "@/lib/format";
import { FAILURE_EXPLANATIONS, humanize, PHASE_LABELS } from "@/lib/labels";

export const dynamicParams = false;

export function generateStaticParams() {
  return getBouts().bouts.map((b) => ({ id: b.id }));
}

type Params = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const bout = getBout((await params).id);
  return bout ? { title: bout.title, description: bout.summary } : {};
}

export default async function BoutPage({ params }: Params) {
  const bout = getBout((await params).id);
  if (!bout) notFound();
  const designIds = getDesigns().designs.map((d) => d.id);
  const known = new Set(designIds);
  const { task } = bout;
  const fromScorer = bout.lanes.every((l) => l.outcome.source === "scorer");

  return (
    <>
      <p className="pt-8 text-sm">
        <Link href="/benchmark/" className="link">
          Benchmark
        </Link>
      </p>
      <PageHeader title={bout.title} lede={<p>{bout.summary}</p>}>
        <ProvenanceChips
          chips={[computeChip(bout.compute), ...uniqueDesignChips(bout.lanes.map((l) => l.design_provenance))]}
        />
      </PageHeader>

      <section className="mt-8" aria-label="Synchronised replays">
        <BoutPlayer bout={bout} designIds={designIds} />
        <p className="mt-4 max-w-[80ch] text-sm text-ink-3">
          All lanes share one clock and one camera. The dashed square is the success height (object centre{" "}
          {(task.success_lift_m * 1000).toFixed(0)} mm above its resting height); the bar under each lane fills while the
          object stays above it and turns green at {task.success_hold_s.toFixed(1)} s. The verdict appears when the
          scorer decides.
        </p>
      </section>

      <section className="mt-14" aria-labelledby="numbers-title">
        <h2 id="numbers-title" className="heading text-xl">
          Scorer numbers
        </h2>
        <p className="mt-1 text-sm text-ink-3">
          {fromScorer
            ? "From the simulator's scorer, sampled every 10 ms."
            : "Derived from the 30 fps replay frames; the scorer's own values were not exported."}
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[640px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-4 font-normal">Lane</th>
                <th scope="col" className="py-2 pr-4 font-normal">Fingers</th>
                <th scope="col" className="py-2 pr-4 font-normal">Outcome</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Peak lift</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Longest hold</th>
                <th scope="col" className="py-2 pr-4 text-right font-normal">Sim steps</th>
                <th scope="col" className="py-2 text-right font-normal">Wall time</th>
              </tr>
            </thead>
            <tbody>
              {bout.lanes.map((lane) => (
                <tr key={lane.id} className="border-b border-rule align-top">
                  <th scope="row" className="py-2.5 pr-4 font-normal">
                    <span className="text-ink">{lane.label}</span>
                    <span className="mt-0.5 block max-w-[34ch] text-xs text-ink-3 first-letter:uppercase">
                      {lane.outcome.failure_mode ? FAILURE_EXPLANATIONS[lane.outcome.failure_mode] : lane.condition}
                    </span>
                  </th>
                  <td className="py-2.5 pr-4">
                    {known.has(lane.design_id) ? (
                      <Link href={`/design/${lane.design_id}/`} className="link">
                        {humanize(lane.entrant)}
                      </Link>
                    ) : (
                      humanize(lane.entrant)
                    )}
                  </td>
                  <td className="py-2.5 pr-4">
                    <OutcomeTag outcome={lane.outcome} />
                  </td>
                  <td className="num py-2.5 pr-4 text-right">{lane.outcome.max_lift_mm.toFixed(1)} mm</td>
                  <td className="num py-2.5 pr-4 text-right">{lane.outcome.hold_s.toFixed(2)} s</td>
                  <td className="num py-2.5 pr-4 text-right">
                    {lane.outcome.sim_steps !== null ? formatInt(lane.outcome.sim_steps) : "n/a"}
                  </td>
                  <td className="num py-2.5 text-right">
                    {lane.outcome.wall_s !== null ? formatDuration(lane.outcome.wall_s) : "n/a"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-14 grid gap-8 md:grid-cols-2" aria-label="Task">
        <div>
          <h2 className="heading text-xl">Task {task.name}</h2>
          <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
            <dt className="text-ink-3">Success</dt>
            <dd>
              Centre rises {(task.success_lift_m * 1000).toFixed(0)} mm and stays {task.success_hold_s.toFixed(1)} s,
              before {task.max_time_s.toFixed(0)} s
            </dd>
            <dt className="text-ink-3">Grip force</dt>
            <dd className="num">{task.grip_force_n} N</dd>
            <dt className="text-ink-3">Max opening</dt>
            <dd className="num">{task.max_opening_mm} mm</dd>
            <dt className="text-ink-3">Controller</dt>
            <dd>
              Open loop, grasp-height setting k = {bout.lanes[0]?.grasp_height_frac ?? "n/a"}
            </dd>
          </dl>
        </div>
        <div>
          <h2 className="heading text-xl">Phases</h2>
          <ol className="mt-3 grid gap-1.5 text-sm">
            {task.phases.map((p) => (
              <li key={p.name} className="grid grid-cols-[6rem_1fr] gap-4">
                <span className="text-ink">{PHASE_LABELS[p.name]}</span>
                <span className="num text-ink-2">
                  {p.start_s.toFixed(1)} s to {p.end_s === null ? "the end" : `${p.end_s.toFixed(1)} s`}
                </span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <TitleBlock title={bout.title} drawing={`/bout/${bout.id}`} sheet="Detail of sheet 2" />
    </>
  );
}
