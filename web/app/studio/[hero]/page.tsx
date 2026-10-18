import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { PageHeader } from "@/components/PageHeader";
import { BuildPackView } from "@/components/studio/BuildPackView";
import { FinalistsTable, stopsText } from "@/components/studio/FinalistsTable";
import { NoWinnerBlock, RecommendationCard } from "@/components/studio/Outcome";
import { PlanView } from "@/components/studio/PlanView";
import { ProvenanceView } from "@/components/studio/ProvenanceView";
import { ScreenView } from "@/components/studio/ScreenView";
import {
  failureText,
  finalistsWithBaseline,
  type FinalistRow,
  humanize,
  type HeroPage,
  massText,
  type Plan,
  runsText,
  stepTitle,
} from "@/components/studio/schema";
import { SessionView } from "@/components/studio/SessionView";
import StudioLanes from "@/components/studio/StudioLanes";
import SuccessMatrix from "@/components/studio/SuccessMatrix";
import { Facts, Section, TierBadge } from "@/components/studio/ui";
import { TitleBlock } from "@/components/TitleBlock";

import { getStudioHero, getStudioHeroIds, getStudioSession } from "../data";

export const dynamicParams = false;

export function generateStaticParams() {
  return getStudioHeroIds().map((hero) => ({ hero }));
}

type Params = { params: Promise<{ hero: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const page = getStudioHero((await params).hero);
  return page ? { title: `Studio: ${page.title}`, description: `Product mode: ${page.headline}` } : {};
}

const COUNT_LABELS: Record<string, string> = {
  single: "Single tools",
  tool_change: "Tool-change sequences",
  multi_head: "Two-tool heads scored",
  multi_head_geometry_searched: "Two-tool head layouts searched",
};

function BaselineCard({ row, plan }: { row: FinalistRow; plan: Plan }) {
  return (
    <article className="border border-rule-strong bg-paper-2/40 p-4" aria-label="Best single off-the-shelf tool" data-best-single>
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-2">Best single off-the-shelf tool</p>
      <h3 className="heading mt-1 text-xl text-ink">{row.label}</h3>
      <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-ink-2">
        {row.tool_details[0]?.family_title ? <span>{row.tool_details[0].family_title}</span> : null}
        <TierBadge tier={row.fidelity} />
      </p>
      <Facts
        className="mt-3"
        items={[
          ["Completes the task", runsText(row.completed_runs, row.n_runs)],
          ["Weakest step", row.weakest_step ? stepTitle(plan, row.weakest_step.step_id) : "n/a"],
          ["Mass", massText(row.complexity.total_mass_kg)],
        ]}
      />
      {stopsText(row, plan).length ? (
        <ul className="mt-3 grid gap-0.5 text-sm text-ink-2">
          {stopsText(row, plan).map((s) => (
            <li key={s}>Stops at {s}</li>
          ))}
        </ul>
      ) : null}
      {Object.keys(row.failure_modes).length ? <p className="mt-2 text-xs text-ink-3">Failures over every run: {failureText(row.failure_modes, ["not_reached"])}</p> : null}
      <p className="mt-3 text-xs text-ink-3">
        The baseline every answer is measured against: the off-the-shelf single tool that did best on the same conditions and seeds.
      </p>
    </article>
  );
}

const TOC: [string, string][] = [
  ["answer", "Answer"],
  ["replays", "Replays"],
  ["plan", "Plan"],
  ["candidates", "Candidates"],
  ["matrix", "Success matrix"],
  ["combinations", "Combinations"],
  ["finalists", "Finalists"],
  ["build-pack", "Build pack"],
  ["session", "Session"],
  ["provenance", "Provenance"],
];

export default async function StudioHeroPage({ params }: Params) {
  const page: HeroPage | undefined = getStudioHero((await params).hero);
  if (!page) notFound();
  const session = getStudioSession(page);
  const finalists = finalistsWithBaseline(page);
  const steps = page.plan.steps.map((s) => ({ id: s.id, title: stepTitle(page.plan, s.id), primitive_title: s.primitive_title }));
  const toc = TOC.filter(([id]) => id !== "session" || session);

  return (
    <>
      <p className="pt-8 text-sm">
        <Link href="/studio/" className="link">
          Studio
        </Link>
      </p>
      <ProductModeBanner />
      <PageHeader
        title={page.title}
        lede={
          <>
            <p className="num text-xs uppercase tracking-wide text-ink-3">{page.setting}</p>
            <blockquote className="mt-1 border-l-2 border-rule-strong pl-3 text-ink-2" data-use-case>
              {page.use_case}
            </blockquote>
          </>
        }
      >
        <p className="prose-measure text-lg text-ink" data-headline>
          {page.headline}
        </p>
        <nav aria-label="On this page" className="mt-4">
          <ul className="flex flex-wrap gap-x-3 gap-y-1 text-sm">
            {toc.map(([id, name]) => (
              <li key={id}>
                <a href={`#${id}`} className="link">
                  {name}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </PageHeader>

      <Section id="answer" title={page.recommendation ? "The recommendation" : "No recommendation"}>
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          {page.recommendation ? (
            <RecommendationCard rec={page.recommendation} plan={page.plan} bestSingle={page.best_single} />
          ) : page.no_winner ? (
            <NoWinnerBlock noWinner={page.no_winner} plan={page.plan} />
          ) : null}
          {page.best_single ? <BaselineCard row={page.best_single} plan={page.plan} /> : null}
        </div>
      </Section>

      <Section
        id="replays"
        title="Side by side"
        lede={
          page.lanes.length ? (
            <p>
              {page.recommendation ? "The recommended setup" : "The closest setup"} and the best single off-the-shelf tool on the same condition, on one
              clock. Each lane re-ran a scored run with recording on; its trajectory hash matches the report. A tool change or head switch is a
              fixed time, not simulated: the replay holds the last pose through it, then cuts to the next tool.
            </p>
          ) : null
        }
      >
        {page.lanes.length ? (
          <StudioLanes lanes={page.lanes} plan={{ steps: page.plan.steps }} />
        ) : (
          <p className="border border-dashed border-rule-strong p-3 text-sm text-ink-3" data-no-lanes>
            The replays of this use case have not been exported yet.
          </p>
        )}
      </Section>

      <Section id="plan" title="The plan" lede={<p>The frozen plan the search ran on, read only. Every step is checked against the simulator&apos;s primitives and object templates.</p>}>
        <PlanView plan={page.plan} />
      </Section>

      <Section id="candidates" title="Candidates" lede={<p>Every tool the Studio can simulate, screened step by step on the same conditions and seeds.</p>}>
        <ScreenView screen={page.screen} plan={page.plan} />
      </Section>

      <Section
        id="matrix"
        title="Success matrix"
        lede={<p>Each candidate screened on at least one step, against every step of the plan. The candidates pruned from every step are only in the list above.</p>}
      >
        <SuccessMatrix matrix={page.matrix} steps={steps} />
      </Section>

      <Section id="combinations" title="Combinations" lede={<p>Setups built from the screen, before the finalists ran the whole task.</p>}>
        <Facts items={Object.entries(page.combinations.counts).map(([k, v]) => [COUNT_LABELS[k] ?? humanize(k), String(v)] as [string, string])} />
        {page.combinations.rejected.length ? (
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-ink-2">Rejected two-tool heads ({page.combinations.rejected.length})</summary>
            <ul className="mt-2 grid gap-1.5 text-sm text-ink-2">
              {page.combinations.rejected.map((r, i) => (
                <li key={`${i}-${r.reason}`}>
                  <span className="text-ink">{r.label || "A two-tool head"}</span>: {r.reason}
                </li>
              ))}
            </ul>
          </details>
        ) : null}
      </Section>

      <Section
        id="finalists"
        title="Finalists against the best single"
        lede={
          <p>
            Every finalist ran the whole task on the {page.provenance.conditions.length} conditions, in ranking order: completion, then the weakest
            step, the median cycle time with changes, then simplicity.
          </p>
        }
      >
        <FinalistsTable rows={finalists} plan={page.plan} baselineId={page.best_single?.setup_id ?? null} />
      </Section>

      <Section id="build-pack" title="Build pack">
        <BuildPackView pack={page.build_pack} note={page.build_pack_note} />
      </Section>

      {page.session && session ? (
        <Section id="session" title="The recorded session" lede={<p>The Grasp Engineer&apos;s recorded run of this use case, read only.</p>}>
          <SessionView block={page.session} session={session} />
        </Section>
      ) : null}

      <Section id="provenance" title="Provenance and caveats">
        <ProvenanceView page={page} />
      </Section>

      <TitleBlock title={`Studio: ${page.title}`} drawing={`/studio/${page.hero}`} sheet="Product mode" />
    </>
  );
}
