import type { Metadata } from "next";
import Link from "next/link";

import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { PageHeader } from "@/components/PageHeader";
import { humanize, type HeroEntry, runsText, type SetupBrief } from "@/components/studio/schema";
import { TierBadge } from "@/components/studio/ui";
import { TitleBlock } from "@/components/TitleBlock";

import { getStudioIndex } from "./data";

export const metadata: Metadata = {
  title: "Studio",
  description:
    "Product mode: describe a job with real objects, and the Studio plans it, screens every tool, combines them and recommends a setup with a build pack, against the best single off-the-shelf tool.",
};

const STAGES: [string, string][] = [
  ["Use case", "You describe the job in plain words: the objects and what has to happen to them."],
  ["Plan", "The Task Engineer turns it into objects and steps the simulator can run. You can check and edit every step."],
  ["Candidates", "Every tool the Studio can simulate: family presets, the open library and designed effectors."],
  ["Screen", "Each candidate tries each step it could do, on a few placements. A tool that cannot do a step is pruned, with the reason."],
  ["Combinations", "Single tools, tool-change sequences and two-tool heads are built from what passed the screen."],
  ["Finalists", "The best few run the whole task on the full condition grid, every setup on the same conditions and seeds."],
  ["Recommendation", "Completion first, then the weakest step, cycle time and simplicity, always against the best single off-the-shelf tool."],
  ["Build pack", "Brackets or changer plates, the tested simulation scene, a bill of materials and a certificate, in one zip."],
];

function Setup({ brief, role }: { brief: SetupBrief; role: string }) {
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
      {role ? <span className="text-xs text-ink-3">{role}</span> : null}
      <span className="text-ink">{brief.label}</span>
      <TierBadge tier={brief.fidelity} />
      <span className="num text-ink-2">{runsText(brief.completed_runs, brief.n_runs)}</span>
      {brief.stops_at ? <span className="text-xs text-ink-3">stops at {humanize(brief.stops_at).toLowerCase()}</span> : null}
    </p>
  );
}

function HeroCard({ h }: { h: HeroEntry }) {
  return (
    <article className="flex min-w-0 flex-col border border-rule-strong bg-paper p-4" aria-label={h.title} data-hero={h.hero}>
      <p className="num text-xs uppercase tracking-wide text-ink-3">{h.setting}</p>
      <h3 className="heading mt-1 text-xl">
        <Link href={`/studio/${h.hero}/`} className="link">
          {h.title}
        </Link>
      </h3>
      <p className="mt-2 text-sm text-ink-2">{h.use_case}</p>
      <div className="mt-4 grid gap-1.5 border-t border-rule pt-3" data-outcome={h.outcome}>
        {h.outcome === "recommended" && h.recommended ? (
          <>
            <p className="text-xs font-semibold uppercase tracking-wide text-accent">Recommended</p>
            <Setup brief={h.recommended} role={h.recommended.label.startsWith(h.recommended.kind_title) ? "" : h.recommended.kind_title} />
          </>
        ) : (
          <>
            <p className="text-xs font-semibold uppercase tracking-wide text-fail">No setup completes this task</p>
            {h.closest ? <Setup brief={h.closest} role="Closest" /> : null}
          </>
        )}
        {h.best_single ? <Setup brief={h.best_single} role="Best single off-the-shelf" /> : null}
      </div>
      <p className="mt-3 text-sm text-ink">{h.headline}</p>
      <p className="mt-auto pt-4 text-sm">
        <Link href={`/studio/${h.hero}/`} className="link">
          The plan, the screen, the finalists and the replays
        </Link>
      </p>
    </article>
  );
}

export default function StudioIndexPage() {
  const index = getStudioIndex();
  return (
    <>
      <ProductModeBanner />
      <PageHeader
        title="Studio"
        lede={
          <p>
            Describe a job a robot has to do with a few real objects. The Studio plans it as steps, tries every tool it can
            simulate on every step, combines the ones that work into tool changes and two-tool heads, and recommends a setup with
            a build pack. Every answer stands next to the best single off-the-shelf tool on the same conditions.
          </p>
        }
      >
        <p className="text-sm">
          <Link href="/studio/new/" className="link" data-plan-own>
            Plan your own use case
          </Link>{" "}
          <span className="text-ink-3">on /studio/new/: describe it, check the steps, edit them.</span>
        </p>
      </PageHeader>

      <section className="mt-12" aria-labelledby="heroes-title">
        <h2 id="heroes-title" className="heading text-xl">
          Recorded use cases
        </h2>
        {index && index.heroes.length > 0 ? (
          <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {index.heroes.map((h) => (
              <HeroCard key={h.hero} h={h} />
            ))}
          </div>
        ) : (
          <p className="mt-3 text-ink-3" data-empty>
            No use case has been exported yet.
          </p>
        )}
      </section>

      <section className="mt-14" aria-labelledby="how-title">
        <h2 id="how-title" className="heading text-xl">
          What the Studio does
        </h2>
        <ol className="mt-4 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
          {STAGES.map(([name, text], i) => (
            <li key={name} className="bg-paper p-3">
              <p className="flex items-baseline gap-2">
                <span className="num text-xs text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                <span className="heading text-base text-ink">{name}</span>
              </p>
              <p className="mt-1 text-sm text-ink-2">{text}</p>
            </li>
          ))}
        </ol>
        <p className="mt-4 max-w-[80ch] text-sm text-ink-3">
          Every step carries its fidelity tier. Rigid: a rigid multibody model with a scripted controller. Approximation: a
          labelled approximation (suction seals, soft arms, latches, threads, valves), shown with its caveat and &quot;verify on
          hardware&quot;. Not simulated yet: never scored as a pass. A recommendation is a simulation ranking under the stated
          conditions, not a hardware guarantee.
        </p>
      </section>

      <section className="mt-12" aria-label="Plan your own use case">
        <h2 className="heading text-xl">Your own use case</h2>
        <p className="mt-2 max-w-[80ch] text-ink-2">
          <Link className="link" href="/studio/new/">
            Plan a use case
          </Link>{" "}
          turns your description into objects and steps you can check, edit and download as a task file. This static site does
          not run the search itself; it shows the recorded use cases above.
        </p>
      </section>

      <TitleBlock title="Studio" drawing="/studio" sheet="Product mode" />
    </>
  );
}
