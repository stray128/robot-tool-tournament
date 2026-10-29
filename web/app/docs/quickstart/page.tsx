import type { Metadata } from "next";
import Link from "next/link";

import { KindGlyph } from "@/components/atlas/parts";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { DocSection, DocsShell, Note } from "@/components/docs/DocsShell";
import { PRODUCT_MODE_LABEL } from "@/components/studio/schema";
import { buildStory, focusOf } from "@/components/studio/story";
import { kindCounts } from "@/lib/atlas";
import { BASE_PATH } from "@/lib/paths";
import { GITHUB_URL } from "@/lib/site";

import { getAtlas } from "../../atlas/data";
import { FEATURED_CASE, featuredHero, getStudioHero } from "../../studio/data";

export const metadata: Metadata = {
  title: "Quickstart",
  description: "Five minutes in the browser: open a worked example, plan a task of your own, find a job like yours in the atlas, then run it yourself.",
};

const SECTIONS = [
  { id: "example", title: "Open a worked example" },
  { id: "plan", title: "Plan your own task" },
  { id: "atlas", title: "Find a job like yours" },
  { id: "run", title: "Run it on your machine" },
];

export default function QuickstartPage() {
  const featured = featuredHero();
  const page = featured ? getStudioHero(featured.hero) : undefined;
  const story = page ? buildStory(page) : null;
  const focus = page ? focusOf(page) : null;
  const atlas = getAtlas();
  const counts = kindCounts(atlas.tasks);
  // The posters are stills of the returns tote's recommended replay; shown only on that page's card.
  const poster = featured?.case_id === FEATURED_CASE;
  const ex = featured ? `/studio/${featured.hero}/` : "/studio/";

  return (
    <DocsShell
      current="/docs/quickstart/"
      kicker="Getting started · 5 minutes"
      title="Quickstart"
      lede={
        <p>
          Everything here runs in your browser: no install, no account, no key. The site replays recorded plans and runs: open a worked example,
          edit a recorded plan or build one by hand, and find a job like yours. Live planning of new words runs on your own machine.
        </p>
      }
      sections={SECTIONS}
    >
      <DocSection id="example" n={1} title="Open a worked example">
        <p>
          A worked example is a real job run end to end: planned into steps, every tool the Studio can simulate tried on every step, and a
          recommendation tested against the best single off-the-shelf tool. Start with one.
        </p>
        {page && story ? (
          <Link href={ex} className="panel grid overflow-hidden no-underline transition-colors hover:border-accent/60 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
            {poster ? (
              <span className="relative block aspect-[16/10] bg-paper-2 md:aspect-auto">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${BASE_PATH}/posters/h1-recommended-dark.jpg`} alt="" className="poster-dark absolute inset-0 size-full object-cover" />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`${BASE_PATH}/posters/h1-recommended-light.jpg`} alt="" className="poster-light absolute inset-0 size-full object-cover" />
                <span className="absolute left-3 top-3 rounded bg-paper/85 px-1.5 py-0.5 text-xs text-ink-2">A still from the recorded physics run</span>
              </span>
            ) : null}
            <span className="block p-5 sm:p-6">
              <span className="block text-xs text-ink-3">Worked example</span>
              <span className="heading mt-1 block text-2xl text-ink">{page.title}</span>
              <span className="mt-3 block text-ink-2">{story.answer}</span>
              <span className="mt-4 flex flex-wrap gap-2 text-xs">
                {focus ? <span className="rounded-full border border-rule-strong px-2 py-px text-ink-2">{page.plan.steps.length} steps</span> : null}
                <span className="rounded-full border border-rule-strong px-2 py-px text-ink-2">{page.screen.n_candidates} tools tried</span>
                <span className="rounded-full border border-rule-strong px-2 py-px text-ink-2">Simulated, verify on hardware</span>
              </span>
              <span className="mt-5 inline-flex items-center gap-2 text-sm text-accent">
                Open the worked example <span aria-hidden="true">→</span>
              </span>
            </span>
          </Link>
        ) : (
          <p className="panel p-4 text-sm">
            <Link className="link" href="/studio/#examples">
              See the worked examples
            </Link>{" "}
            and open one.
          </p>
        )}
        <p>
          Read it top to bottom: the answer in one sentence, the two replays playing on one clock (recorded physics, not animation), one card per
          step saying which tools could do it and why the pick won, then the head to head and the build pack.{" "}
          <Link className="link" href="/docs/tutorials/read-a-recommendation/">
            Read a recommendation
          </Link>{" "}
          walks through every part.
        </p>
      </DocSection>

      <DocSection id="plan" n={2} title="Plan your own task">
        <ol className="grid gap-3 sm:grid-cols-3">
          {[
            ["Describe", "In the Studio, write what the robot has to do with a few real objects, with sizes and limits, or pick an example."],
            ["Plan", "Press Plan. The planner turns the words into objects and steps the simulator can run, and flags anything it cannot."],
            ["Check and download", "Edit any object or step, then download the plan as TaskSpec JSON to run it yourself."],
          ].map(([t, d], i) => (
            <li key={t} className="panel p-4">
              <span className="num text-sm text-accent">{i + 1}</span>
              <span className="heading mt-1 block text-base text-ink">{t}</span>
              <span className="mt-1 block text-sm">{d}</span>
            </li>
          ))}
        </ol>
        <Note title="On this public site">
          The site has no server, so Plan replays the planner&apos;s recorded run for the examples. To plan new words live, run the Studio on your
          own machine (step 4). Every result is labelled &ldquo;{PRODUCT_MODE_LABEL}&rdquo;.
        </Note>
        <p>
          <Link className="btn btn-primary text-sm" href="/studio/#plan">
            Open the planner
          </Link>{" "}
          <Link className="link ml-3" href="/docs/tutorials/plan-a-task/">
            Tutorial: plan a task
          </Link>
        </p>
      </DocSection>

      <DocSection id="atlas" n={3} title="Find a job like yours">
        <p>
          The use-case atlas lists {atlas.tasks.length} researched jobs, from solar farms to hospital labs, each with the setup an experienced
          integrator would expect and why. Search it for your parts or your process.
        </p>
        <Link href="/atlas/" className="panel flex flex-wrap items-center gap-x-8 gap-y-3 p-5 no-underline transition-colors hover:border-accent/60">
          {(["single", "tool_change", "two_tool_head"] as const).map((k) => (
            <span key={k} className="flex items-center gap-2.5">
              <KindGlyph kind={k} className="size-6 text-accent" />
              <span>
                <span className="num block text-2xl leading-none text-ink">{counts[k]}</span>
                <span className="block text-xs text-ink-3">{k === "single" ? "one tool" : k === "tool_change" ? "tool change" : "two-tool head"}</span>
              </span>
            </span>
          ))}
          <span className="ml-auto text-sm text-accent">
            Browse the atlas <span aria-hidden="true">→</span>
          </span>
        </Link>
      </DocSection>

      <DocSection id="run" n={4} title="Run it on your machine">
        <p>The same code runs the search behind every example. With Python 3.12 and uv installed, three commands run your downloaded plan:</p>
        <CodeBlock
          label="Terminal"
          code={`git clone ${GITHUB_URL}.git && cd robot-tool-tournament
make sync && source .venv/bin/activate
python -m rtt.studio.search taskspec.json --out runs/studio/my-task --pack`}
        />
        <p>
          <Link className="link" href="/docs/tutorials/run-it-yourself/">
            Run it yourself
          </Link>{" "}
          covers Docker, the options and what the run writes.
        </p>
      </DocSection>
    </DocsShell>
  );
}
