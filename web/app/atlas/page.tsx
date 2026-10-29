import type { Metadata } from "next";
import Link from "next/link";

import { AtlasExplorer } from "@/components/atlas/AtlasExplorer";
import { KindBar, KindGlyph, KindLegend, KIND_ORDER } from "@/components/atlas/parts";
import { PageHeader } from "@/components/PageHeader";
import { humanize } from "@/components/studio/schema";
import { TitleBlock } from "@/components/TitleBlock";
import { KIND_LABELS, kindCounts, ruleNumber, settingLabel, simStatus } from "@/lib/atlas";

import { getStudioIndex } from "../studio/data";
import { getAtlas, heroLinks } from "./data";

export const metadata: Metadata = {
  title: "Use-case atlas",
  description:
    "Researched robot jobs in construction, factories, warehouses, labs, kitchens and services, each with the end-effector setup an experienced integrator would expect, and which ones the Studio simulates.",
};

function pct(n: number, of: number): string {
  return `${Math.round((n / Math.max(of, 1)) * 100)}%`;
}

/** The researched jobs: what integrators pick and why, which ones are simulated, and the rules behind the picks. */
export default function AtlasPage() {
  const atlas = getAtlas();
  const heroes = heroLinks();
  const index = getStudioIndex();
  const tasks = atlas.tasks;
  const counts = kindCounts(tasks);
  const simulated = tasks.filter((t) => simStatus(t, heroes).kind === "simulated");
  const headline = (caseId: string) => index?.heroes.find((h) => h.case_id === caseId) ?? null;
  const settings = new Set(tasks.map((t) => t.setting)).size;

  return (
    <>
      <PageHeader
        title="Use-case atlas"
        lede={
          <p>
            {tasks.length} jobs people do today across {settings} settings, from solar farms to hospital labs, each with the end-effector setup an
            experienced integrator would expect and why. Built from public deployments and research; brand names are left out, and each job links to
            its sources.
          </p>
        }
      >
        <div className="flex flex-wrap gap-2 text-sm">
          <a href="#browse" className="btn btn-primary">
            Browse the {tasks.length} jobs
          </a>
          <a href="#rules" className="btn btn-quiet">
            How integrators choose
          </a>
        </div>
      </PageHeader>

      <section className="mt-14 grid gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] lg:items-stretch" aria-labelledby="split-title">
        <div className="panel flex flex-col justify-between p-5 sm:p-6">
          <div>
            <h2 id="split-title" className="text-sm text-ink-3">
              What the research says
            </h2>
            <p className="mt-3 flex flex-wrap items-baseline gap-x-3">
              <span className="num text-[3.25rem] leading-none text-accent">{pct(counts.single, tasks.length)}</span>
              <span className="text-lg text-ink">of the jobs need one tool.</span>
            </p>
            <p className="mt-3 text-ink-2">
              An integrator would use one tool for {counts.single} of the {tasks.length} jobs, a tool change for {counts.tool_change} and a two-tool head
              for {counts.two_tool_head}. A second tool earns its place when the steps need different kinds of tool, or when a part must stay held
              while another tool works on it.
            </p>
          </div>
          <p className="mt-5 text-xs text-ink-3">Research estimates from public sources, {atlas.generated}. Not simulation results.</p>
        </div>
        <div className="panel p-5 sm:p-6">
          <KindLegend />
          <dl className="mt-5 grid gap-4">
            <div>
              <dt className="flex items-baseline justify-between text-sm text-ink">
                All jobs <span className="num text-xs text-ink-3">{tasks.length}</span>
              </dt>
              <dd className="mt-1.5">
                <KindBar tasks={tasks} label="All jobs" />
              </dd>
            </div>
            {atlas.domains.map((d) => {
              const list = tasks.filter((t) => t.domain === d.id);
              const c = kindCounts(list);
              return (
                <div key={d.id}>
                  <dt className="flex items-baseline justify-between gap-3 text-sm text-ink-2">
                    {d.title}
                    <span className="num text-xs text-ink-3">
                      {KIND_ORDER.map((k) => c[k]).join(" · ")} of {list.length}
                    </span>
                  </dt>
                  <dd className="mt-1.5">
                    <KindBar tasks={list} label={d.title} />
                  </dd>
                </div>
              );
            })}
          </dl>
        </div>
      </section>

      <section className="mt-16" aria-labelledby="simulated-title">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 id="simulated-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
            Simulated in the Studio
          </h2>
          <Link href="/studio/" className="link text-sm">
            Plan your own task
          </Link>
        </div>
        <p className="prose-measure mt-2 text-ink-2">
          These jobs run in physics as worked examples: every tool the Studio can simulate tries every step, and the answer comes from the runs, not
          from the research estimate above. The rest are researched, not yet simulated.
        </p>
        {simulated.length ? (
          <ul className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {simulated.map((t) => {
              const s = simStatus(t, heroes);
              if (s.kind !== "simulated") return null;
              const h = headline(t.sim.hero ?? "");
              return (
                <li key={t.id}>
                  <Link href={`/studio/${s.hero}/`} className="panel flex h-full flex-col p-5 no-underline transition-colors hover:border-accent/60" data-simulated-hero={s.hero}>
                    <span className="flex items-center justify-between gap-2 text-xs text-ink-3">
                      {h ? humanize(h.setting) : settingLabel(t.setting)}
                      <span className="rounded-full border border-accent/70 bg-accent-soft px-2 py-px text-accent">
                        {s.match === "same" ? "Worked example" : "Related job"}
                      </span>
                    </span>
                    <span className="heading mt-2 block text-xl text-ink">{s.title}</span>
                    {h ? <span className="mt-2 block text-sm text-ink-2">{h.headline}</span> : null}
                    <span className="mt-4 flex items-center gap-2 border-t border-rule pt-3 text-xs text-ink-3">
                      <KindGlyph kind={t.setup.kind} className="size-4" />
                      Expected setup (research estimate): {KIND_LABELS[t.setup.kind].toLowerCase()}
                    </span>
                    <span className="mt-1 block text-xs text-ink-3">Researched as: {t.title}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="panel mt-6 p-5 text-ink-2">No researched job has a worked example yet.</p>
        )}
      </section>

      <section id="browse" className="mt-16 scroll-mt-20" aria-labelledby="browse-title">
        <h2 id="browse-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
          Browse the jobs
        </h2>
        <p className="prose-measure mt-2 text-ink-2">
          Each job lists its steps, the setup we expect an integrator to pick and why, and the public sources it was built from. The badge says
          whether the Studio simulates it, or how hard it would be to model.
        </p>
        <p className="prose-measure mt-2 mb-6 text-sm text-ink-3" data-physics-decides>
          The expected setups are our research estimates. When a job runs in the Studio, physics decides the final answer.
        </p>
        <AtlasExplorer tasks={tasks} domains={atlas.domains} heroes={heroes} />
      </section>

      <section id="rules" className="mt-20 scroll-mt-20" aria-labelledby="rules-title">
        <h2 id="rules-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
          How integrators choose: research summary
        </h2>
        <p className="prose-measure mt-2 text-sm text-ink-3" data-physics-decides>
          Our summary of the research, not a rule the simulator follows blindly: in the Studio, physics decides the final answer.
        </p>
        <p className="prose-measure mt-2 text-ink-2">
          {atlas.rules.length} rules distilled from the same research: one tool by default, and a tool change or a two-tool head only when the
          steps need different kinds of tool or a part must stay held while another tool works.
        </p>
        <ol className="mt-6 grid gap-3 md:grid-cols-2">
          {atlas.rules.map((r) => (
            <li key={r.id} id={`rule-${ruleNumber(r.id)}`} className="panel flex gap-3 p-4 scroll-mt-24" data-atlas-rule={ruleNumber(r.id)}>
              <span className="num grid h-7 min-w-10 shrink-0 place-items-center rounded-md border border-rule-strong px-1.5 text-xs text-ink-2">{ruleNumber(r.id)}</span>
              <div className="min-w-0">
                <h3 className="heading text-base text-ink">{r.title}</h3>
                <p className="mt-1 text-sm text-ink-2">{r.text}</p>
                {r.sources.length ? (
                  <p className="mt-2 flex flex-wrap gap-x-2 text-xs text-ink-3">
                    Sources
                    {r.sources.map((src, i) => (
                      <a key={src.url} href={src.url} rel="noopener noreferrer" target="_blank" className="link num" aria-label={`Source ${i + 1} for rule ${ruleNumber(r.id)} (opens in a new tab)`}>
                        {i + 1}
                      </a>
                    ))}
                  </p>
                ) : null}
              </div>
            </li>
          ))}
        </ol>
      </section>

      <p className="mt-10 max-w-[80ch] text-sm text-ink-3">{atlas.note} Source links lead to the public pages each job was built from; they may name companies.</p>

      <TitleBlock title="Use-case atlas" drawing="/atlas" sheet="Use cases" />
    </>
  );
}
