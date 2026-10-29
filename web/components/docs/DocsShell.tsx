import Link from "next/link";
import type { ReactNode } from "react";

import { TitleBlock } from "@/components/TitleBlock";
import { repoBlob } from "@/lib/site";

export interface DocPage {
  href: string;
  title: string;
  /** One line under the title in lists. */
  blurb: string;
  minutes: number;
}

/** The guides, in reading order: the quickstart, then the three core tutorials. */
export const DOC_PAGES: readonly DocPage[] = [
  { href: "/docs/quickstart/", title: "Quickstart", blurb: "Five minutes in the browser: a worked example, your own plan, the atlas.", minutes: 5 },
  { href: "/docs/tutorials/plan-a-task/", title: "Plan a task", blurb: "Describe a job, check the objects and steps, fix what the simulator cannot run, download the plan.", minutes: 10 },
  {
    href: "/docs/tutorials/read-a-recommendation/",
    title: "Read a recommendation",
    blurb: "The parts of a story page: the answer, the replays, each step, the head to head and the build pack.",
    minutes: 10,
  },
  { href: "/docs/tutorials/run-it-yourself/", title: "Run it yourself", blurb: "Install, run the search on your plan from the command line or in Docker, and read what it writes.", minutes: 20 },
];

const REFERENCE: { href: string; title: string }[] = [
  { href: repoBlob("docs/STUDIO.md"), title: "How the Studio works" },
  { href: repoBlob("docs/STUDIO_PRIMITIVES.md"), title: "Skills and objects" },
  { href: repoBlob("docs/STUDIO_SEARCH.md"), title: "Search and ranking" },
  { href: repoBlob("docs/STUDIO_ENGINEER.md"), title: "The planner" },
];

function NavList({ current }: { current: string }) {
  return (
    <nav aria-label="Docs" className="text-sm">
      <p className="text-xs text-ink-3">Getting started</p>
      <ul className="mt-2 grid gap-0.5">
        <li>
          <Link href="/docs/" className={`block rounded-md px-2.5 py-1.5 no-underline ${current === "/docs/" ? "bg-paper-2 text-ink" : "text-ink-2 hover:text-ink"}`}>
            Docs home
          </Link>
        </li>
        {DOC_PAGES.map((p) => (
          <li key={p.href}>
            <Link
              href={p.href}
              aria-current={current === p.href ? "page" : undefined}
              className={`flex items-center justify-between gap-2 rounded-md px-2.5 py-1.5 no-underline ${
                current === p.href ? "bg-paper-2 text-ink shadow-[inset_2px_0_0_var(--accent)]" : "text-ink-2 hover:text-ink"
              }`}
            >
              {p.title}
              <span className="num shrink-0 whitespace-nowrap text-[0.6875rem] text-ink-3">{p.minutes} min</span>
            </Link>
          </li>
        ))}
      </ul>
      <p className="mt-6 text-xs text-ink-3">Reference, in the repository</p>
      <ul className="mt-2 grid gap-0.5">
        {REFERENCE.map((r) => (
          <li key={r.href}>
            <a href={r.href} rel="noopener" className="block rounded-md px-2.5 py-1.5 text-ink-2 no-underline hover:text-ink">
              {r.title} <span aria-hidden="true">↗</span>
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/**
 * A docs page: the sidebar (a menu on phones), the title block, the sections listed on the right
 * on wide screens, and the next guide at the end.
 */
export function DocsShell({
  current,
  kicker,
  title,
  lede,
  sections,
  children,
}: {
  current: string;
  kicker: string;
  title: string;
  lede: ReactNode;
  sections: { id: string; title: string }[];
  children: ReactNode;
}) {
  const i = DOC_PAGES.findIndex((p) => p.href === current);
  const prev = i > 0 ? DOC_PAGES[i - 1]! : null;
  const next = i >= 0 && i < DOC_PAGES.length - 1 ? DOC_PAGES[i + 1]! : null;
  return (
    <>
      <div className="grid gap-8 pt-8 sm:pt-12 lg:grid-cols-[13rem_minmax(0,1fr)] xl:grid-cols-[13rem_minmax(0,1fr)_11rem]">
        <aside className="lg:sticky lg:top-20 lg:self-start">
          <details className="panel group p-3 lg:hidden">
            <summary className="flex list-none items-center justify-between text-sm text-ink [&::-webkit-details-marker]:hidden">
              Docs menu
              <svg viewBox="0 0 12 12" className="size-3 transition-transform group-open:rotate-180" aria-hidden="true">
                <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </summary>
            <div className="mt-3 border-t border-rule pt-3">
              <NavList current={current} />
            </div>
          </details>
          <div className="hidden lg:block">
            <NavList current={current} />
          </div>
        </aside>

        <article className="min-w-0" data-doc={current}>
          <header>
            <p className="text-sm text-ink-3">{kicker}</p>
            <h1 className="display mt-2 text-[clamp(2.1rem,4.6vw,3.25rem)] text-ink">{title}</h1>
            <div className="prose-measure mt-4 text-lg text-ink-2">{lede}</div>
          </header>
          <div className="mt-10 grid gap-14">{children}</div>
          {prev || next ? (
            <nav className="mt-16 grid gap-3 sm:grid-cols-2" aria-label="Next and previous guides">
              {prev ? (
                <Link href={prev.href} className="panel block p-4 no-underline hover:border-rule-strong">
                  <span className="block text-xs text-ink-3">Previous</span>
                  <span className="heading mt-1 block text-base text-ink">{prev.title}</span>
                </Link>
              ) : (
                <span className="hidden sm:block" />
              )}
              {next ? (
                <Link href={next.href} className="panel block p-4 text-right no-underline hover:border-accent/60">
                  <span className="block text-xs text-ink-3">Next</span>
                  <span className="heading mt-1 block text-base text-ink">{next.title}</span>
                  <span className="mt-1 block text-sm text-ink-2">{next.blurb}</span>
                </Link>
              ) : null}
            </nav>
          ) : null}
        </article>

        <aside className="hidden xl:block">
          <nav aria-label="On this page" className="sticky top-20 text-sm">
            <p className="text-xs text-ink-3">On this page</p>
            <ul className="mt-2 grid gap-1.5 border-l border-rule pl-3">
              {sections.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className="text-ink-2 no-underline hover:text-ink">
                    {s.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </aside>
      </div>
      <TitleBlock title={title} drawing={current} sheet="Docs" />
    </>
  );
}

/** A numbered section of a guide. */
export function DocSection({ id, n, title, children }: { id: string; n?: number; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24" aria-labelledby={`${id}-title`}>
      <h2 id={`${id}-title`} className="heading flex items-baseline gap-3 text-[1.5rem] text-ink sm:text-[1.75rem]">
        {n !== undefined ? <span className="num text-base text-accent">{String(n).padStart(2, "0")}</span> : null}
        {title}
      </h2>
      <div className="mt-4 grid gap-4 text-ink-2 [&_p]:max-w-[68ch]">{children}</div>
    </section>
  );
}

/** A labelled aside: a tip, a limit or what to watch for. */
export function Note({ tone = "info", title, children }: { tone?: "info" | "caution"; title: string; children: ReactNode }) {
  const cls = tone === "caution" ? "border-pending/60 bg-pending-soft" : "border-rule-strong bg-surface";
  return (
    <div className={`max-w-[72ch] rounded-lg border px-4 py-3 text-sm ${cls}`}>
      <p className={`font-semibold ${tone === "caution" ? "text-pending" : "text-ink"}`}>{title}</p>
      <div className="mt-1 text-ink-2">{children}</div>
    </div>
  );
}

/** A numbered marker that ties a callout in the text to a spot on a figure. */
export function Marker({ n }: { n: number }) {
  return (
    <span className="num inline-grid size-6 shrink-0 place-items-center rounded-full bg-accent text-xs font-bold text-accent-ink" aria-label={`Callout ${n}`}>
      {n}
    </span>
  );
}
