import Link from "next/link";
import type { ReactNode } from "react";

import { BaselineCard } from "@/components/story/BaselineCard";
import { BuildPackCard } from "@/components/story/BuildPackCard";
import { HeadToHead } from "@/components/story/HeadToHead";
import { StepCard } from "@/components/story/StepCard";
import { StoryPlayback } from "@/components/story/StoryPlayback";
import StoryReplays from "@/components/story/StoryReplays";
import { Term } from "@/components/story/Term";
import { FinalistsTable } from "@/components/studio/FinalistsTable";
import { NoWinnerBlock } from "@/components/studio/Outcome";
import { PlanView } from "@/components/studio/PlanView";
import { ProvenanceView } from "@/components/studio/ProvenanceView";
import { ScreenView } from "@/components/studio/ScreenView";
import {
  finalistsWithBaseline,
  type HeroEntry,
  type HeroPage,
  humanize,
  PRODUCT_MODE_LABEL,
  runEnd,
  stepTitle,
  TIER_LABELS,
  TIER_NOTES,
} from "@/components/studio/schema";
import { SessionView } from "@/components/studio/SessionView";
import { baselineStory, buildStory, focusOf, laneNoun, laneStops } from "@/components/studio/story";
import SuccessMatrix from "@/components/studio/SuccessMatrix";
import { Facts } from "@/components/studio/ui";
import { TitleBlock } from "@/components/TitleBlock";
import type { RecordedSession } from "@/lib/session";

/** The held-out benchmark's arms for this page's setups (app/studio/heldout.ts), as the page needs them. */
export interface StoryHeldout {
  n_conditions: number;
  p: number;
  recommended: { n: number; completed: number };
  best_single: { n: number; completed: number; steps: Record<string, { n: number; successes: number; failure_modes: Record<string, number> }> };
  /** A dated note on the benchmark this result belongs to, when one was published. */
  note: { date: string; text: string } | null;
}

const COUNT_LABELS: Record<string, string> = {
  single: "Single tools",
  tool_change: "Tool-change sequences",
  multi_head: "Two-tool heads scored",
  multi_head_geometry_searched: "Two-tool head layouts searched",
};

function Block({ id, title, lede, children }: { id: string; title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section id={id} className="mt-20 scroll-mt-20" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`} className="heading text-[1.75rem] text-ink sm:text-[2rem]">
        {title}
      </h2>
      {lede ? <div className="prose-measure mt-2 text-ink-2">{lede}</div> : null}
      <div className="mt-6">{children}</div>
    </section>
  );
}

function DataPart({ title, lede, children }: { title: string; lede?: ReactNode; children: ReactNode }) {
  return (
    <section className="mt-10 min-w-0 first:mt-2">
      <h3 className="heading text-lg text-ink">{title}</h3>
      {lede ? <div className="mt-1 max-w-[80ch] text-sm text-ink-3">{lede}</div> : null}
      <div className="mt-3">{children}</div>
    </section>
  );
}

/**
 * The use case as a short visual story: answer, replays, step by step, head to head, build pack.
 * Renders a recommendation, or, when no setup completes the task, the closest setup in its place.
 * Pure: every number comes from the page data and the props (app/studio/[hero]/page.tsx loads them).
 */
export function StoryPage({
  page,
  session,
  held,
  others,
}: {
  page: HeroPage;
  session: RecordedSession | null;
  held: StoryHeldout | null;
  others: HeroEntry[];
}) {
  const story = buildStory(page);
  const rec = page.recommendation;
  const focus = focusOf(page);
  const closest = focus?.role === "closest";
  const recFinalist = focus?.row ?? null;
  const base = page.best_single;
  const baseLane = page.lanes.find((l) => l.id === "best_single") ?? null;
  const placement = page.constants.find((c) => c.key === "placement_error_mm");
  const placementMm = placement && typeof placement.value === "number" ? placement.value : null;
  const present = page.constants.find((c) => c.key === "present_s");
  const steps = story.steps.map((s) => ({ id: s.step.id, index: s.index, name: s.name }));
  const stops = laneStops(page);
  const stopLabels = Object.fromEntries(Object.entries(stops).map(([id, s]) => [id, s?.phrase ?? null]));
  const finalists = finalistsWithBaseline(page);
  const matrixSteps = page.plan.steps.map((s) => ({ id: s.id, title: stepTitle(page.plan, s.id), primitive_title: s.primitive_title }));
  const recLane = focus?.lane ?? null;
  const focusName = closest ? "closest setup" : "recommended setup";
  const baseline = baselineStory(page);

  return (
    <StoryPlayback>
      <header className="pt-8 sm:pt-12">
        <nav aria-label="Breadcrumb" className="text-sm text-ink-3">
          <Link href="/studio/" className="text-ink-2 no-underline hover:text-ink">
            Studio
          </Link>
          <span className="mx-2" aria-hidden="true">
            /
          </span>
          <span>Worked example, {page.setting}</span>
        </nav>
        <div className="mt-5 grid gap-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:items-end">
          <div>
            <h1 className="display text-[clamp(2.4rem,6vw,4.4rem)] text-ink">{page.title}</h1>
            <p className="mt-6 max-w-[34ch] text-[clamp(1.25rem,2.4vw,1.625rem)] leading-snug text-ink" data-answer>
              {story.answer}
            </p>
            <div className="mt-5 flex flex-wrap items-center gap-2">
              {closest ? (
                <span className="inline-flex items-center rounded-full border border-dashed border-fail/70 px-2.5 py-0.5 text-sm text-fail" data-outcome-badge="closest">
                  No recommendation: closest setup shown
                </span>
              ) : null}
              {focus ? (
                <Term label={`${TIER_LABELS[focus.fidelity]} fidelity`} tip={TIER_NOTES[focus.fidelity]}>
                  <span
                    className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-sm ${
                      focus.fidelity === "approximation" ? "border-pending/70 bg-pending-soft text-pending" : "border-rule-strong text-ink-2"
                    }`}
                    data-fidelity={focus.fidelity}
                  >
                    {TIER_LABELS[focus.fidelity]}
                  </span>
                </Term>
              ) : null}
              <span className="inline-flex items-center rounded-full border border-rule-strong px-2.5 py-0.5 text-sm text-ink-2" data-verify>
                Simulated, verify on hardware
              </span>
              <Term label="Product mode" tip="The Studio's answers are simulation rankings under stated conditions. The pre-registered finger experiment is a separate result, on the Evidence page.">
                <span className="inline-flex items-center rounded-full border border-dashed border-rule-strong px-2.5 py-0.5 text-sm text-ink-3" data-product-mode>
                  {PRODUCT_MODE_LABEL}
                </span>
              </Term>
            </div>
          </div>
          <figure className="panel p-4 text-sm">
            <figcaption className="text-xs text-ink-3">The task, in plain words</figcaption>
            <blockquote className="mt-2 text-ink-2" data-use-case>
              {page.use_case}
            </blockquote>
            <p className="mt-3 text-xs text-ink-3">
              Planned into {page.plan.steps.length} steps; {page.screen.n_candidates} tools tested on {page.screen.n_conditions} conditions each, {page.screen.runs} screen runs.
            </p>
          </figure>
        </div>
        {focus && (rec?.caveat || (closest && focus.steps.some((s) => s.fidelity === "approximation"))) ? (
          <p className="mt-6 max-w-[80ch] rounded-lg border border-pending/50 bg-pending-soft px-3 py-2 text-sm text-ink-2" data-caveat>
            <span className="font-semibold text-pending">Caveat: </span>
            {focus.steps.filter((s) => s.fidelity === "approximation").length} of {focus.steps.length} steps of this setup run on approximation models.{" "}
            {TIER_NOTES.approximation} Each tool&apos;s full modelling note is under &ldquo;Show all data&rdquo;.
          </p>
        ) : null}
      </header>

      {page.no_winner ? (
        <section className="mt-12">
          <NoWinnerBlock noWinner={page.no_winner} plan={page.plan} />
        </section>
      ) : null}

      <Block
        id="replays"
        title={page.lanes.length > 1 ? "Watch both runs" : "Watch the run"}
        lede={
          <p>
            Recorded physics, not animation: each replay re-ran a scored run with recording on, on the same condition ({recLane?.condition_text ?? "the same placement"}), and plays
            on one clock.
          </p>
        }
      >
        {page.lanes.length ? (
          <>
            <StoryReplays lanes={page.lanes} plan={{ steps: page.plan.steps }} steps={steps} stopLabels={stopLabels} />
            <p className="mt-3 max-w-[80ch] text-sm text-ink-3" data-text-equivalent>
              In words:{" "}
              {page.lanes
                .map((l) => {
                  const stop = stops[l.id];
                  return l.completed
                    ? `the ${laneNoun(l)} (${l.label}) does all ${l.steps.length} steps and finishes at ${(l.cycle_time_s ?? runEnd(l)).toFixed(1)} s`
                    : `the ${laneNoun(l)} (${l.label}) stops at step ${stop ? stop.step : "?"} after ${runEnd(l).toFixed(1)} s${stop && stop.phrase !== "failed" ? `: it ${stop.phrase}` : ""}`;
                })
                .join("; ")}
              .
            </p>
          </>
        ) : (
          <p className="panel p-4 text-sm text-ink-3">The replays of this use case have not been exported yet.</p>
        )}
      </Block>

      <Block
        id="steps"
        title="Step by step"
        lede={
          <p>
            Every tool the Studio can simulate tried every step it could do, on {page.screen.n_conditions}{" "}
            <Term
              label="Screen conditions"
              tip={`The first ${page.screen.n_conditions} conditions of the grid: nominal placement, then objects shifted${placementMm !== null ? ` by up to ${placementMm} mm` : ""}, each with its own seed. Tools that cannot do a step at all are ruled out first, with a reason.`}
            >
              <span>conditions</span>
            </Term>{" "}
            each. Here is who could do each step, which tool the {focusName} uses, and why.
          </p>
        }
      >
      {story.whySetup.length || story.rankingReason ? (
        <section className="mb-8 grid gap-4 lg:grid-cols-2" aria-label="Why this setup">
          {story.whySetup.length ? (
            <div className="panel p-5">
              <h2 className="heading text-lg text-ink">Why it takes {rec && rec.tools.length > 1 ? `${rec.tools.length} tools` : "this tool"}</h2>
              <ul className="mt-3 grid gap-2 text-ink-2">
                {story.whySetup.map((w) => (
                  <li key={w} className="flex gap-2">
                    <span className="mt-2.5 size-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
                    <span>{w}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {story.rankingReason ? (
            <div className="panel p-5">
              <h2 className="heading text-lg text-ink">Why this setup over the other finalists</h2>
              <p className="mt-3 text-ink-2">{story.rankingReason}</p>
              <p className="mt-3 text-sm text-ink-3">
                The rule: finish the task most often, then the strongest weakest step, then the shortest cycle, then the simplest setup.
                {rec?.kind === "multi_head" && present && typeof present.value === "number"
                  ? ` A head switch is a fixed ${present.value} s tilt of the wrist, not a simulated motion.`
                  : ""}
              </p>
            </div>
          ) : null}
        </section>
      ) : null}

        <div className="grid gap-6">
          {story.steps.map((s) => (
            <StepCard
              key={s.step.id}
              story={s}
              plan={page.plan}
              nSteps={story.steps.length}
              kind={focus?.kind ?? "single"}
              role={focus?.role ?? "recommended"}
              heldoutBaseline={held?.best_single.steps[s.step.id]}
            />
          ))}
        </div>
      </Block>

      {baseline || (recFinalist && base && recFinalist.setup_id !== base.setup_id) ? (
        <Block
          id="head-to-head"
          title="Against the best single tool"
          lede={
            <p>
              One tool is the default; a second tool or a tool change has to earn its place. {closest ? "The closest setup" : "The recommendation"} against the best
              single tool, on the same conditions and seeds.
            </p>
          }
        >
          {baseline ? (
            <div className="mb-6">
              <BaselineCard baseline={baseline} />
            </div>
          ) : null}
          {recFinalist && base && recFinalist.setup_id !== base.setup_id ? (
          <HeadToHead
            rec={recFinalist}
            recTitle={closest ? "Closest setup" : "Recommended"}
            base={base}
            baseLane={baseLane}
            heldout={held ? { n: held.recommended.n, recommended: held.recommended.completed, best_single: held.best_single.completed } : null}
            conditions={page.provenance.conditions.length}
            placementMm={placementMm}
          />
          ) : null}
          {held ? (
            <p className="mt-3 text-sm text-ink-3">
              Held-out check from the pre-declared use-case benchmark v1 ({held.n_conditions} new conditions per use case, exact paired test, p ={" "}
              {held.p < 0.001 ? held.p.toExponential(1) : held.p.toFixed(4)}). <Link className="link" href="/evidence/#v1-note">See the evidence</Link>.
            </p>
          ) : null}
          {held?.note ? (
            <p className="mt-3 max-w-[80ch] rounded-lg border border-rule-strong bg-surface px-3 py-2 text-sm text-ink-2" data-v1-note>
              <span className="font-semibold text-ink">Note, {held.note.date}: </span>
              {held.note.text}
            </p>
          ) : null}
        </Block>
      ) : null}

      <Block id="build-pack" title="Build pack" lede={<p>Everything to build and check the {focusName}, in one download.</p>}>
        <BuildPackCard pack={page.build_pack} note={page.build_pack_note} />
      </Block>

      <section className="mt-20" aria-labelledby="all-data-title">
        <details className="group panel" id="all-data">
          <summary className="flex list-none items-center justify-between gap-4 px-5 py-4 [&::-webkit-details-marker]:hidden">
            <span>
              <span id="all-data-title" className="heading block text-lg text-ink">
                Show all data
              </span>
              <span className="block text-sm text-ink-3">The plan, every candidate, the success matrix, the finalists, the conditions and the provenance.</span>
            </span>
            <svg viewBox="0 0 12 12" className="size-4 shrink-0 text-ink-2 transition-transform group-open:rotate-180" aria-hidden="true">
              <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          </summary>
          <div className="border-t border-rule px-4 pb-8 sm:px-5">
            <DataPart title="The plan" lede={<p>The frozen plan the search ran on. Every step is checked against the simulator&apos;s skills and object templates.</p>}>
              <PlanView plan={page.plan} />
            </DataPart>
            <DataPart title="Every candidate" lede={<p>Every tool the Studio can simulate, screened step by step on the same conditions and seeds.</p>}>
              <ScreenView screen={page.screen} plan={page.plan} />
            </DataPart>
            <DataPart title="Success matrix" lede={<p>Each screened tool against every step. Tools ruled out of every step are only in the list above.</p>}>
              <SuccessMatrix matrix={page.matrix} steps={matrixSteps} />
            </DataPart>
            <DataPart title="Combinations searched">
              <Facts items={Object.entries(page.combinations.counts).map(([k, v]) => [COUNT_LABELS[k] ?? humanize(k), String(v)] as [string, string])} />
            </DataPart>
            <DataPart title="Finalists" lede={<p>Every finalist ran the whole task on the {page.provenance.conditions.length} conditions, in ranking order.</p>}>
              <FinalistsTable rows={finalists} plan={page.plan} baselineId={page.best_single?.setup_id ?? null} />
            </DataPart>
            {page.session && session ? (
              <DataPart title="The recorded agent session" lede={<p>The Grasp Engineer&apos;s recorded run of this use case, read only.</p>}>
                <SessionView block={page.session} session={session} />
              </DataPart>
            ) : null}
            <DataPart title="Conditions, provenance and caveats">
              <ProvenanceView page={page} />
            </DataPart>
          </div>
        </details>
      </section>

      {others.length ? (
        <nav className="mt-16" aria-label="Other worked examples">
          <h2 className="text-sm text-ink-3">Other worked examples</h2>
          <ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {others.map((h) => (
              <li key={h.hero}>
                <Link href={`/studio/${h.hero}/`} className="panel block h-full p-4 no-underline hover:border-rule-strong">
                  <span className="block text-xs text-ink-3">{humanize(h.setting)}</span>
                  <span className="heading mt-1 block text-lg text-ink">{h.title}</span>
                  <span className="mt-1 block text-sm text-ink-2">{h.headline}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}

      <TitleBlock title={`${page.title}: worked example`} drawing={`/studio/${page.hero}`} sheet="Studio, product mode" />
    </StoryPlayback>
  );
}

