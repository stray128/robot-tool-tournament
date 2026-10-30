import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { DocSection, DocsShell, Marker, Note } from "@/components/docs/DocsShell";
import { HostedDocSection, withHosted } from "@/components/docs/HostedDocs";
import { paramText, PLAN_RESULT_LABELS, TIER_LABELS, TIER_NOTES, type Plan } from "@/components/studio/schema";
import { stepName } from "@/components/studio/story";
import { TierBadge } from "@/components/studio/ui";
import { liveCopy } from "@/lib/live_copy";

import { featuredHero, getStudioHero } from "../../../studio/data";

export const metadata: Metadata = {
  title: "Plan a task",
  description: "Describe a job in plain words, let the planner turn it into objects and steps the simulator can run, check and edit them, and download the plan.",
};

const SECTIONS = [
  { id: "plan", title: "What a plan is" },
  { id: "describe", title: "Describe the job" },
  { id: "planner", title: "Press Plan" },
  { id: "check", title: "Check and edit" },
  { id: "download", title: "Download the plan" },
];

/** Sizes, masses and limits in a description: the numbers the planner turns into parameters. */
const QUANTITY = /(\d+(?:\.\d+)?(?:\s?x\s?\d+(?:\.\d+)?)*\s?(?:mm|cm|m|g|kg|N m|N|deg|degrees|s)\b)/g;

function Highlighted({ text }: { text: string }) {
  const parts = text.split(QUANTITY);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="rounded bg-accent-soft px-1 text-ink">
            {p}
          </mark>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

function ObjectsAndSteps({ plan }: { plan: Plan }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
      <div className="panel min-w-0 p-4">
        <p className="flex items-center gap-2 text-sm font-semibold text-ink">
          <Marker n={1} /> Objects ({plan.objects.length})
        </p>
        <ul className="mt-3 grid gap-2.5 text-sm">
          {plan.objects.map((o) => {
            const params = Object.entries(o.params)
              .filter(([, v]) => typeof v === "number")
              .slice(0, 3)
              .map(([k, v]) => paramText(k, v).join(" "));
            return (
              <li key={o.name} className="border-t border-rule pt-2 first:border-t-0 first:pt-0">
                <span className="block text-ink">{o.label}</span>
                <span className="block text-xs text-ink-3">
                  {o.kind_title}
                  {params.length ? ` · ${params.join(", ")}` : ""}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="panel min-w-0 p-4">
        <p className="flex items-center gap-2 text-sm font-semibold text-ink">
          <Marker n={2} /> Steps ({plan.steps.length})
        </p>
        <ol className="mt-3 grid gap-3 text-sm">
          {plan.steps.map((s, i) => (
            <li key={s.id} className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 border-t border-rule pt-2.5 first:border-t-0 first:pt-0">
              <span className="num grid size-6 place-items-center rounded border border-rule-strong text-xs text-ink-2">{i + 1}</span>
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-2">
                  <span className="text-ink">{stepName(s.id)}</span>
                  <span className="rounded border border-accent/50 px-1.5 text-[0.6875rem] uppercase tracking-wide text-accent">{s.primitive_title}</span>
                  <TierBadge tier={s.fidelity} />
                </span>
                <span className="mt-1 block text-xs text-ink-2">
                  <span className="text-ink-3">Done when </span>
                  {s.success.map((c) => c.text).join("; ") || "the skill finishes"}
                  {s.constraints.max_force_n !== null ? <span className="text-ink-3"> · at most {s.constraints.max_force_n} N</span> : null}
                  {s.constraints.max_torque_nm !== null ? <span className="text-ink-3"> · at most {s.constraints.max_torque_nm} N m</span> : null}
                </span>
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-4 flex items-center gap-2 border-t border-rule pt-3 text-xs text-ink-3">
          <Marker n={3} /> Every step names a skill, the object it acts on, when it counts as done, and its limits.
        </p>
      </div>
    </div>
  );
}

function Tip({ title, children }: { title: string; children: ReactNode }) {
  return (
    <li className="panel p-4">
      <p className="heading text-base text-ink">{title}</p>
      <p className="mt-1 text-sm">{children}</p>
    </li>
  );
}

export default function PlanATaskPage() {
  const featured = featuredHero();
  const page = featured ? getStudioHero(featured.hero) : undefined;
  const plan = page?.plan ?? null;

  return (
    <DocsShell
      current="/docs/tutorials/plan-a-task/"
      kicker="Tutorial · 10 minutes"
      title="Plan a task"
      lede={
        <p>
          The Studio starts from a job in plain words. The planner turns it into objects and steps the simulator can run; you check and edit
          them, then download the plan to run the search on it.
        </p>
      }
      sections={withHosted(SECTIONS)}
    >
      <DocSection id="plan" n={1} title="What a plan is">
        <p>
          A plan has two lists. <strong className="text-ink">Objects</strong> are parametric templates the simulator knows: a part, a bin, a hinged
          lid or door, a drawer, a latch, a threaded joint, a valve, a port. <strong className="text-ink">Steps</strong> are skills on those objects:
          grasp, suction, wrap, hook pull, twist, insert, push. Every step carries the test that says it is done and the limits it must respect.
        </p>
        {plan && page ? (
          <>
            <p className="text-sm text-ink-3">The plan behind the {page.title.toLowerCase()} example, as the simulator reads it:</p>
            <ObjectsAndSteps plan={plan} />
          </>
        ) : null}
      </DocSection>

      <DocSection id="describe" n={2} title="Describe the job">
        <p>
          Open the{" "}
          <Link className="link" href="/studio/#plan">
            Studio
          </Link>{" "}
          and write what the robot has to do, or start from an example. The planner reads what a technician would say: each object with its size
          and mass, what happens in what order, and the limits.
        </p>
        {page ? (
          <figure className="panel p-4">
            <figcaption className="text-xs text-ink-3">The example&apos;s description; the highlighted numbers become parameters and limits</figcaption>
            <blockquote className="mt-2 text-[0.9375rem] text-ink">
              <Highlighted text={page.use_case} />
            </blockquote>
          </figure>
        ) : null}
        <ul className="grid gap-3 sm:grid-cols-3">
          <Tip title="Name the objects">Say what each thing is and how big and heavy it is: a 120 x 80 x 60 mm carton of 300 g, not a box.</Tip>
          <Tip title="Say the order">One sentence per action, in order. The planner keeps your order as step dependencies.</Tip>
          <Tip title="Give the limits">A force the lid can take, a torque to tighten to, a precision. Limits become checks on every run.</Tip>
        </ul>
      </DocSection>

      <DocSection id="planner" n={3} title="Press Plan">
        <p>
          The Task Engineer (NVIDIA Nemotron 3 Super) answers with one forced tool call: the objects as templates and the steps as skills, with
          bounded parameters. Code then checks every rule. A plan that breaks one goes back once with the list of problems.
        </p>
        {plan ? (
          <div className="panel flex flex-wrap items-center gap-x-6 gap-y-2 p-4 text-sm">
            <span>
              <span className="text-ink-3">The example: </span>
              <span className="text-ink">{PLAN_RESULT_LABELS[plan.result] ?? plan.result}</span>
            </span>
            {plan.calls !== null ? (
              <span>
                <span className="text-ink-3">Model calls: </span>
                <span className="num text-ink">{plan.calls}</span>
              </span>
            ) : null}
            {plan.layout_moves.length ? (
              <span>
                <span className="text-ink-3">Layout fixes: </span>
                <span className="num text-ink">{plan.layout_moves.length}</span>
              </span>
            ) : null}
          </div>
        ) : null}
        <Note title="Explore mode on this site">
          {liveCopy("planTaskNote")}
          <Link className="link" href="/docs/tutorials/run-it-yourself/">
            Run it yourself
          </Link>
          ), or press &ldquo;Build a plan by hand&rdquo; and fill in the objects and steps yourself.
        </Note>
      </DocSection>

      <DocSection id="check" n={4} title="Check and edit">
        <p>
          Every object and step can be edited, added, moved or removed, and the plan is checked again on each change: the line above the lists
          says whether it is valid. Look at three things before you run it.
        </p>
        <ul className="grid gap-3 md:grid-cols-3">
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">Fidelity</p>
            <p className="mt-1">Each step says how it is simulated:</p>
            <ul className="mt-2 grid gap-2">
              {(["rigid", "approximation", "not_simulated"] as const).map((t) => (
                <li key={t} className="flex items-start gap-2">
                  <TierBadge tier={t} />
                  <span className="text-xs text-ink-2">{TIER_NOTES[t]}</span>
                </li>
              ))}
            </ul>
          </li>
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">Assumptions and questions</p>
            <p className="mt-1">
              What the planner assumed, and what it would ask you. An answer is assumed until you change the plan.
              {plan?.assumptions.length ? ` The example made ${plan.assumptions.length}, for instance: “${plan.assumptions[0]}”` : ""}
            </p>
          </li>
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">What cannot be simulated</p>
            <p className="mt-1">
              Cloth, liquids, food and other soft things are out of scope. The planner lists them with the nearest skill it can run, and a{" "}
              {TIER_LABELS.not_simulated.toLowerCase()} step never counts as a pass.
            </p>
          </li>
        </ul>
      </DocSection>

      <DocSection id="download" n={5} title="Download the plan">
        <p>
          When the plan is valid, <strong className="text-ink">Download TaskSpec JSON</strong> saves it as <code className="num text-ink">taskspec-&lt;id&gt;.json</code>{" "}
          (or <strong className="text-ink">Copy JSON</strong> puts it on the clipboard). That file is the input of the search: every tool tried
          on every step, the combinations, the finalists and the build pack.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link className="btn btn-primary text-sm" href="/studio/#plan">
            Open the planner
          </Link>
          <Link className="btn btn-quiet text-sm" href="/docs/tutorials/run-it-yourself/">
            Run the search on it
          </Link>
        </div>
      </DocSection>

      <HostedDocSection n={6} />
    </DocsShell>
  );
}
