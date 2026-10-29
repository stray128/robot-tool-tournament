import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { DocSection, DocsShell, Marker, Note } from "@/components/docs/DocsShell";
import { BuildPackCard } from "@/components/story/BuildPackCard";
import { HeadToHead } from "@/components/story/HeadToHead";
import { StepCard } from "@/components/story/StepCard";
import { StoryPlayback } from "@/components/story/StoryPlayback";
import StoryReplays from "@/components/story/StoryReplays";
import { PRODUCT_MODE_LABEL, TIER_LABELS, TIER_NOTES } from "@/components/studio/schema";
import { buildStory, focusOf, laneStops } from "@/components/studio/story";

import { featuredHero, getStudioHero } from "../../../studio/data";

export const metadata: Metadata = {
  title: "Read a recommendation",
  description: "The parts of a story page, on a real worked example: the answer, the replays, each step, the head to head and the build pack.",
};

const SECTIONS = [
  { id: "answer", title: "The answer" },
  { id: "replays", title: "The replays" },
  { id: "step", title: "A step card" },
  { id: "head-to-head", title: "Head to head" },
  { id: "build-pack", title: "The build pack" },
  { id: "nothing", title: "When nothing finishes" },
];

/** A real piece of the story page beside numbered notes on how to read it. */
function Anatomy({ figure, notes }: { figure: ReactNode; notes: { n: number; title: string; text: ReactNode }[] }) {
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div className="min-w-0">{figure}</div>
      <ol className="grid content-start gap-3">
        {notes.map((n) => (
          <li key={n.n} className="flex gap-3">
            <Marker n={n.n} />
            <span className="min-w-0 text-sm">
              <span className="block font-semibold text-ink">{n.title}</span>
              <span className="mt-0.5 block text-ink-2">{n.text}</span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function ReadARecommendationPage() {
  const featured = featuredHero();
  const page = featured ? getStudioHero(featured.hero) : undefined;
  if (!page) {
    return (
      <DocsShell current="/docs/tutorials/read-a-recommendation/" kicker="Tutorial" title="Read a recommendation" lede={<p>No worked example is exported yet.</p>} sections={[]}>
        <p>
          <Link className="link" href="/studio/#examples">
            See the worked examples
          </Link>
        </p>
      </DocsShell>
    );
  }
  const story = buildStory(page);
  const focus = focusOf(page);
  const base = page.best_single;
  const baseLane = page.lanes.find((l) => l.id === "best_single") ?? null;
  const stops = laneStops(page);
  const stopLabels = Object.fromEntries(Object.entries(stops).map(([id, s]) => [id, s?.phrase ?? null]));
  const stopLabel = stops.best_single?.phrase ?? null;
  const steps = story.steps.map((s) => ({ id: s.step.id, index: s.index, name: s.name }));
  // The bottleneck: the step the fewest tools could do (the card that explains the most).
  const hardest = [...story.steps].sort((a, b) => a.passed.length - b.passed.length || a.index - b.index)[0]!;
  const placement = page.constants.find((c) => c.key === "placement_error_mm");
  const change = focus?.lane?.segments.find((s) => s.change !== "none") ?? null;
  const story_url = `/studio/${page.hero}/`;

  return (
    <DocsShell
      current="/docs/tutorials/read-a-recommendation/"
      kicker="Tutorial · 10 minutes"
      title="Read a recommendation"
      lede={
        <p>
          Every worked example is one page: the answer, the runs, the reason for each choice, and what to build. This guide reads the{" "}
          <Link className="link" href={story_url}>
            {page.title.toLowerCase()}
          </Link>{" "}
          page part by part, with its own data.
        </p>
      }
      sections={SECTIONS}
    >
      <StoryPlayback>
        <DocSection id="answer" n={1} title="The answer">
          <Anatomy
            figure={
              <div className="panel p-5">
                <p className="heading text-2xl text-ink">{page.title}</p>
                <p className="mt-3 flex gap-2 text-lg leading-snug text-ink">
                  <Marker n={1} />
                  <span>{story.answer}</span>
                </p>
                <p className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                  <Marker n={2} />
                  {focus ? (
                    <span className={`rounded-full border px-2.5 py-0.5 ${focus.fidelity === "approximation" ? "border-pending/70 bg-pending-soft text-pending" : "border-rule-strong text-ink-2"}`}>
                      {TIER_LABELS[focus.fidelity]}
                    </span>
                  ) : null}
                  <Marker n={3} />
                  <span className="rounded-full border border-rule-strong px-2.5 py-0.5 text-ink-2">Simulated, verify on hardware</span>
                  <span className="rounded-full border border-dashed border-rule-strong px-2.5 py-0.5 text-ink-3">{PRODUCT_MODE_LABEL}</span>
                </p>
              </div>
            }
            notes={[
              { n: 1, title: "One sentence first", text: "What to use, how often it finished, and how the best single off-the-shelf tool did on the same runs." },
              {
                n: 2,
                title: "Fidelity",
                text: focus ? `${TIER_LABELS[focus.fidelity]}: ${TIER_NOTES[focus.fidelity]}` : TIER_NOTES.rigid,
              },
              { n: 3, title: "Two labels on every answer", text: "These are simulation rankings under stated conditions, not hardware results, and not the pre-registered experiment." },
            ]}
          />
        </DocSection>

        <DocSection id="replays" n={2} title="The replays">
          <p>
            Both runs are recorded physics on the same condition, played on one clock. Press play, drag the scrubber, or click a step on the
            timeline to jump both replays there.
          </p>
          <StoryReplays lanes={page.lanes} plan={{ steps: page.plan.steps }} steps={steps} stopLabels={stopLabels} />
          <ol className="grid gap-3 md:grid-cols-2">
            {[
              { n: 1, title: "Two lanes", text: "The recommended setup on the left, the best single tool on the right, same placement and seed." },
              {
                n: 2,
                title: "Changing tool",
                text: change
                  ? `At ${change.change_start_s.toFixed(1)} s the ${change.change === "present" ? "head switches tools" : "arm changes tools"}: a fixed ${change.change_s} s that is not simulated. The replay holds still under a "Changing tool" card instead of animating the swap.`
                  : 'A tool change or head switch is a fixed time, not simulated; the replay holds still under a "Changing tool" card instead of animating it.',
              },
              { n: 3, title: "The step timeline", text: "Each block is a step, green when it passed and red when it failed; hatched gaps are tool changes." },
              { n: 4, title: "Where it stops", text: stopLabel ? `The best single tool's run ends at the red line: it ${stopLabel}.` : "A run that fails ends at a red line, with the reason." },
            ].map((n) => (
              <li key={n.n} className="flex gap-3 text-sm">
                <Marker n={n.n} />
                <span>
                  <span className="block font-semibold text-ink">{n.title}</span>
                  <span className="text-ink-2">{n.text}</span>
                </span>
              </li>
            ))}
          </ol>
        </DocSection>

        <DocSection id="step" n={3} title="A step card">
          <p>
            One card per step says what the step takes, how many tools could do it, which one the recommendation uses and why. This is step{" "}
            {hardest.index + 1}, the one the fewest tools could do. Its &ldquo;Watch this step&rdquo; button jumps the replays above to it.
          </p>
          <StepCard story={hardest} plan={page.plan} nSteps={story.steps.length} kind={focus?.kind ?? "single"} role={focus?.role ?? "recommended"} />
          <ol className="grid gap-3 md:grid-cols-2">
            {[
              { n: 1, title: "What it takes", text: "The skill, the object, the limit and the test that says the step is done." },
              {
                n: 2,
                title: "Who could do it",
                text: `${hardest.passed.length} of ${hardest.tried} tools tried in physics passed every screen run; the bar splits the whole field of ${hardest.n_candidates} by outcome, and tools that could not do the step at all were ruled out first, with a reason.`,
              },
              { n: 3, title: "The pick, and why", text: "Counted from the success matrix, the candidates and the finalists. Where the data cannot separate two tools, the card says so." },
              { n: 4, title: "The baseline on this step", text: "How the best single tool did here, and on the held-out conditions when the benchmark covered it." },
            ].map((n) => (
              <li key={n.n} className="flex gap-3 text-sm">
                <Marker n={n.n} />
                <span>
                  <span className="block font-semibold text-ink">{n.title}</span>
                  <span className="text-ink-2">{n.text}</span>
                </span>
              </li>
            ))}
          </ol>
        </DocSection>
      </StoryPlayback>

      <DocSection id="head-to-head" n={4} title="Head to head">
        {focus?.row && base ? (
          <Anatomy
            figure={
              <HeadToHead
                rec={focus.row}
                recTitle={focus.role === "closest" ? "Closest setup" : "Recommended"}
                base={base}
                baseLane={baseLane}
                heldout={null}
                conditions={page.provenance.conditions.length}
                placementMm={placement && typeof placement.value === "number" ? placement.value : null}
              />
            }
            notes={[
              { n: 1, title: "Same conditions", text: `Both setups ran the same ${page.provenance.conditions.length} conditions and seeds.` },
              { n: 2, title: "Cost of the answer", text: "Cycle time, number of tools and their mass: what the extra completion costs." },
              { n: 3, title: "Held-out check", text: "On the story page, a pre-declared benchmark adds new conditions neither setup was chosen on, when one covered the use case." },
            ]}
          />
        ) : (
          <p>This example has no baseline to compare with.</p>
        )}
      </DocSection>

      <DocSection id="build-pack" n={5} title="The build pack">
        <p>One download with everything to build and check the setup. Every file states its licence.</p>
        <BuildPackCard pack={page.build_pack} note={page.build_pack_note} />
      </DocSection>

      <DocSection id="nothing" n={6} title="When nothing finishes">
        <p>
          Some jobs have no setup that completes them. Then the page says so first, shows the closest setup in place of a recommendation, with
          its replay and step cards, and names the step every finalist stops at and how far each tool family got against the target.
        </p>
        <Note title="Read the caveats" tone="caution">
          Approximation steps (suction seals, soft arms, latches, threads, valves) are modelled, not measured. Under &ldquo;Show all data&rdquo; each
          story page lists every candidate, the success matrix, the conditions and each tool&apos;s modelling note.
        </Note>
        <p>
          <Link className="btn btn-primary text-sm" href={story_url}>
            Open the full page
          </Link>
        </p>
      </DocSection>
    </DocsShell>
  );
}
