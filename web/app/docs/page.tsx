import type { Metadata } from "next";

import Link from "next/link";

import { DOC_PAGES } from "@/components/docs/DocsShell";
import { LinkCard } from "@/components/LinkCard";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { GITHUB_URL, repoBlob } from "@/lib/site";

export const metadata: Metadata = {
  title: "Docs",
  description: "How Effector Lab works: the Studio, the library, the benchmarks, running it yourself, and the credits.",
};

const GROUPS: { title: string; text: string; items: { href: string; title: string; text: string; external?: boolean }[] }[] = [
  {
    title: "More to read",
    text: "The project as a whole, and the cloud runs.",
    items: [
      { href: repoBlob("README.md"), title: "Read me", text: "What Effector Lab does, how to install it and how to reproduce the results.", external: true },
      { href: "/atlas/", title: "Use-case atlas", text: "95 researched jobs with the setup an integrator would expect, and which ones are simulated." },
      { href: repoBlob("docs/NEBIUS.md"), title: "Running on the cloud", text: "How the benchmarks fan out as cloud jobs.", external: true },
    ],
  },
  {
    title: "The Studio",
    text: "From a task in plain words to a recommendation and a build pack.",
    items: [
      { href: repoBlob("docs/STUDIO.md"), title: "How the Studio works", text: "Plan, screen, combine, recommend; fidelity tiers and fairness.", external: true },
      { href: repoBlob("docs/STUDIO_ENGINEER.md"), title: "The planner", text: "How a description becomes objects and steps, and how plans are checked.", external: true },
      { href: repoBlob("docs/STUDIO_PRIMITIVES.md"), title: "Skills and objects", text: "The object templates and the skills the simulator can run.", external: true },
      { href: repoBlob("docs/STUDIO_SEARCH.md"), title: "Search and ranking", text: "The screen, combinations, finalists, the ranking rule and the build pack.", external: true },
    ],
  },
  {
    title: "Tools and the library",
    text: "Where the tools come from and what each one may be used for.",
    items: [
      { href: "/library/", title: "Tool library", text: "Every open item with its files, specs, licence and simulation status." },
      { href: repoBlob("docs/LIBRARY.md"), title: "Library rules", text: "The licence policy, the audit and how items are added.", external: true },
      { href: repoBlob("docs/EFFECTORS.md"), title: "Effector families", text: "Grippers, hands, the soft arm, the suction cup and designed effectors.", external: true },
    ],
  },
  {
    title: "Benchmarks",
    text: "The pre-declared tests and their rules.",
    items: [
      { href: repoBlob("docs/USECASE_V1.md"), title: "Studio benchmark protocol", text: "Frozen before the held-out runs, with the committed secret.", external: true },
      { href: repoBlob("docs/PREREG.md"), title: "Finger designer pre-registration", text: "The frozen protocol of the finger experiment.", external: true },
      { href: "/evidence/", title: "Evidence", text: "Both results, the methods and the compute log." },
    ],
  },
];

export default function DocsPage() {
  return (
    <>
      <PageHeader
        title="Docs"
        lede={
          <p>
            How Effector Lab works and how to run it. Start with the quickstart and the three tutorials; the reference guides live with the code in
            the repository.
          </p>
        }
      />
      <section className="mt-12" aria-labelledby="docs-start">
        <h2 id="docs-start" className="heading text-[1.375rem] text-ink">
          Getting started
        </h2>
        <p className="mt-1 text-ink-2">Four short guides, in order. Each works on the site&apos;s real examples.</p>
        <ol className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {DOC_PAGES.map((p, i) => (
            <li key={p.href}>
              <Link href={p.href} className={`panel flex h-full flex-col p-4 no-underline transition-colors hover:border-accent/60 ${i === 0 ? "border-accent/50" : ""}`}>
                <span className="flex items-center justify-between text-xs text-ink-3">
                  <span className="num text-accent">{String(i + 1).padStart(2, "0")}</span>
                  <span className="num">{p.minutes} min</span>
                </span>
                <span className="heading mt-2 block text-lg text-ink">{p.title}</span>
                <span className="mt-1 block flex-1 text-sm text-ink-2">{p.blurb}</span>
                <span className="mt-3 text-sm text-accent">
                  {i === 0 ? "Start here" : "Open"} <span aria-hidden="true">→</span>
                </span>
              </Link>
            </li>
          ))}
        </ol>
      </section>
      <div className="mt-14 grid gap-14">
        {GROUPS.map((g) => (
          <section key={g.title} aria-labelledby={`docs-${g.title}`}>
            <h2 id={`docs-${g.title}`} className="heading text-[1.375rem] text-ink">
              {g.title}
            </h2>
            <p className="mt-1 text-ink-2">{g.text}</p>
            <ul className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {g.items.map((it) => (
                <li key={it.title}>
                  <LinkCard href={it.href} title={it.title} text={it.text} external={it.external} />
                </li>
              ))}
            </ul>
          </section>
        ))}
        <section aria-labelledby="docs-credits">
          <h2 id="docs-credits" className="heading text-[1.375rem] text-ink">
            Credits and licences
          </h2>
          <p className="mt-1 text-ink-2">Every third-party model, scan and file, with its licence and changes.</p>
          <ul className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <li>
              <LinkCard href="/credits/" title="Credits" text="Attribution for every model the site draws and every item in the library." />
            </li>
            <li>
              <LinkCard href={repoBlob("docs/ATTRIBUTION.md")} external title="Attribution file" text="The attribution record in the repository." />
            </li>
            <li>
              <LinkCard href={GITHUB_URL} external title="Source code" text="The whole project under the Apache-2.0 licence." />
            </li>
          </ul>
        </section>
      </div>
      <TitleBlock title="Docs" drawing="/docs" sheet="Docs" />
    </>
  );
}
