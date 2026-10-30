import type { Metadata } from "next";
import Link from "next/link";

import { Term } from "@/components/story/Term";
import catalog from "@/components/studio/intake/catalog.json";
import IntakeWorkbench, { type RecordedPlans } from "@/components/studio/intake/IntakeWorkbench";
import recorded from "@/components/studio/intake/recorded.json";
import type { Catalog } from "@/components/studio/intake/taskspec";
import { humanize, PRODUCT_MODE_LABEL } from "@/components/studio/schema";
import { buildStory } from "@/components/studio/story";
import { TitleBlock } from "@/components/TitleBlock";
import { liveCopy } from "@/lib/live_copy";

import { getStudioHero, getStudioIndex } from "./data";

export const metadata: Metadata = {
  title: "Studio",
  description:
    "Plan a task for your robot, see how every tool the Studio can simulate does on every step, and get a recommendation with a build pack, next to the best single off-the-shelf tool. Product mode.",
};

function Meter({ value, n, tone, label }: { value: number; n: number; tone: "accent" | "baseline"; label: string }) {
  return (
    <div>
      <p className="flex items-baseline justify-between gap-2 text-xs">
        <span className={tone === "accent" ? "text-accent" : "text-baseline"}>{label}</span>
        <span className="num text-ink">
          {value} of {n}
        </span>
      </p>
      <span className="mt-1 block h-1.5 overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
        <span className={`block h-full rounded-full ${tone === "accent" ? "bg-accent" : "bg-baseline"}`} style={{ width: `${n ? Math.max((value / n) * 100, 1) : 0}%` }} />
      </span>
    </div>
  );
}

export default function StudioPage() {
  const plans = recorded as unknown as RecordedPlans;
  const summary = plans.gate.summary;
  const index = getStudioIndex();
  const heroes = index?.heroes ?? [];
  const results = Object.fromEntries(heroes.map((h) => [h.case_id, `/studio/${h.hero}/`]));

  return (
    <>
      <header className="pt-10 sm:pt-16">
        <h1 className="display text-[clamp(2.4rem,5.6vw,4rem)] text-ink">Studio</h1>
        <p className="prose-measure mt-5 text-lg text-ink-2">
          Plan a task for your robot, see how every tool does on every step, and get a recommendation you can build. Start with your own words, or open a
          worked example.
        </p>
        <div className="mt-6 flex flex-wrap items-center gap-2 text-sm">
          <Term
            label="Explore mode"
            tip={liveCopy("studioTip")}
          >
            <span className="inline-flex items-center gap-1.5 rounded-full border border-rule-strong px-2.5 py-0.5 text-ink-2">
              <span className="size-1.5 rounded-full bg-pass" aria-hidden="true" />
              Explore mode: recorded plans and results
            </span>
          </Term>
          <Term label="Product mode" tip="Studio results are simulation rankings under stated conditions, not hardware guarantees. The pre-registered finger experiment is a separate result, on the Evidence page.">
            <span className="inline-flex items-center rounded-full border border-dashed border-rule-strong px-2.5 py-0.5 text-ink-3" data-product-mode>
              {PRODUCT_MODE_LABEL}
            </span>
          </Term>
          <Link className="link ml-1" href="/docs/tutorials/run-it-yourself/">
            Run it locally
          </Link>
        </div>
      </header>

      <section id="plan" className="mt-14 scroll-mt-20" aria-labelledby="plan-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="plan-title" className="heading text-[1.5rem] text-ink sm:text-[1.75rem]">
              Plan your own task
            </h2>
            <p className="prose-measure mt-2 text-ink-2">
              Say what the robot has to do with a few real objects. The planner turns it into objects and steps the simulator can run; you check and edit
              every step, then download the plan.
            </p>
          </div>
          <p className="max-w-[46ch] text-xs text-ink-3">
            Planner test set, one live recorded run: {summary.plans.valid} of {summary.plans.n} use cases planned validly, {summary.refusals.refused_with_nearest}{" "}
            of {summary.refusals.n} unsupported ones refused with the nearest skill. {plans.label}.
          </p>
        </div>
        <IntakeWorkbench catalog={catalog as unknown as Catalog} recorded={plans} results={results} />
      </section>

      <section id="examples" className="mt-24 scroll-mt-20" aria-labelledby="examples-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 id="examples-title" className="heading text-[1.5rem] text-ink sm:text-[1.75rem]">
            Worked examples
          </h2>
          <p className="text-sm text-ink-3">Recorded end to end: plan, screen, finalists, replays and build pack. Simulated, verify on hardware.</p>
        </div>
        <ul className="mt-5 grid gap-4 md:grid-cols-3">
          {heroes.map((h) => {
            const page = getStudioHero(h.hero);
            const story = page ? buildStory(page) : null;
            return (
              <li key={h.hero}>
                <Link href={`/studio/${h.hero}/`} className="panel group flex h-full flex-col p-5 no-underline hover:border-accent/60" data-hero={h.hero}>
                  <span className="text-xs text-ink-3">
                    {humanize(h.setting)}, {h.n_steps} steps, {h.n_candidates} tools tested
                  </span>
                  <span className="heading mt-2 text-xl text-ink group-hover:text-accent">{h.title}</span>
                  <span className="mt-2 text-sm text-ink-2">{story?.answer ?? h.headline}</span>
                  <span className="mt-auto grid gap-2.5 pt-5">
                    {h.recommended ? <Meter value={h.recommended.completed_runs} n={h.recommended.n_runs} tone="accent" label="Recommended" /> : null}
                    {h.best_single ? <Meter value={h.best_single.completed_runs} n={h.best_single.n_runs} tone="baseline" label="Best single tool" /> : null}
                  </span>
                  <span className="mt-4 text-sm text-accent">Open the worked example</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </section>

      <TitleBlock title="Studio" drawing="/studio" sheet="Studio, product mode" />
    </>
  );
}
