import Link from "next/link";

import { HeroAsk } from "@/components/home/HeroAsk";
import HeroScene from "@/components/home/HeroScene";
import { homeLede } from "@/components/home/lede";
import { FamilyIcon } from "@/components/story/FamilyIcon";
import { humanize } from "@/components/studio/schema";
import { setupPhrase, stepName } from "@/components/studio/story";
import { TitleBlock } from "@/components/TitleBlock";
import { getResults } from "@/lib/data";
import { toCard } from "@/lib/library";
import { liveCopy } from "@/lib/live_copy";
import { BASE_PATH } from "@/lib/paths";
import { runnerConfigured } from "@/lib/runner";
import { repoBlob } from "@/lib/site";

import { getLibraryIndex } from "./library/data";
import { featuredHero, getStudioHero, getStudioIndex } from "./studio/data";
import { getHeldout } from "./studio/heldout";

function kindFamily(kind: string): string {
  if (kind.includes("suction")) return "Suction cup";
  if (kind.includes("hand")) return "Multi-finger hand";
  if (kind.includes("finger") || kind.includes("jaw")) return "Parallel jaw";
  return "Industrial gripper";
}

function Bar({ value, n, tone }: { value: number; n: number; tone: "accent" | "baseline" }) {
  return (
    <span className="block h-2.5 w-full overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
      <span className={`block h-full rounded-full ${tone === "accent" ? "bg-accent" : "bg-baseline"}`} style={{ width: `${n ? Math.max((value / n) * 100, 0.6) : 0}%` }} />
    </span>
  );
}

export default function HomePage() {
  const index = getStudioIndex();
  const heroes = index?.heroes ?? [];
  const featured = featuredHero();
  const h1 = featured ? getStudioHero(featured.hero) : undefined;
  const ex = featured ? `/studio/${featured.hero}/` : "/studio/";
  const rec = h1?.recommendation ?? null;
  const lane = h1?.lanes.find((l) => l.id === "recommended") ?? null;
  const held = getHeldout();
  const s1 = getResults().s1.analysis ?? null;
  const library = getLibraryIndex();
  const usedIn = new Map<string, string[]>();
  for (const h of heroes) {
    const page = getStudioHero(h.hero);
    for (const t of page?.recommendation?.tools ?? []) usedIn.set(t.display_name, [...(usedIn.get(t.display_name) ?? []), h.title]);
  }
  const libCards = (library?.items.map(toCard) ?? []).filter((c) => usedIn.has(c.display_name)).slice(0, 4);
  const heroSteps = h1 ? h1.plan.steps.map((s) => ({ id: s.id, name: stepName(s.id), description: s.description })) : [];

  const stages = [
    {
      name: "Plan",
      href: "/studio/#plan",
      cta: "Plan a task",
      text: "Describe the job in plain words, and the planner turns it into objects and steps the simulator can run. " + liveCopy("homeCard"),
      fact: h1 ? `${h1.title}: one paragraph, ${h1.plan.steps.length} steps, ${h1.plan.objects.length} objects` : null,
    },
    {
      name: "Test",
      href: `${ex}#steps`,
      cta: "See a screen",
      text: "Every tool it can simulate tries every step it could do, in physics, on the same placements and seeds. Tools that cannot do a step are ruled out with a reason.",
      fact: h1 ? `${h1.screen.n_candidates} tools, ${h1.screen.runs} screen runs, ${Math.round(h1.screen.wall_s / 60)} minutes` : null,
    },
    {
      name: "Recommend",
      href: `${ex}#head-to-head`,
      cta: "See a recommendation",
      text: "Single tools, tool changes and two-tool heads are combined from what passed, and the finalists run the whole task against the best single off-the-shelf tool.",
      fact:
        rec && h1?.best_single
          ? h1.best_single.setup_id === rec.setup_id
            ? `${rec.completed_runs} of ${rec.n_runs} runs; the pick is the best single tool`
            : `${rec.completed_runs} of ${rec.n_runs} runs against ${h1.best_single.completed_runs} of ${h1.best_single.n_runs}`
          : null,
    },
    {
      name: "Build",
      href: `${ex}#build-pack`,
      cta: "See a build pack",
      text: "Download a build pack: the printed bracket or changer plates, a bill of materials, the tested scenes and a certificate.",
      fact: h1?.build_pack ? `${h1.build_pack.files.length} files, ${Math.round(h1.build_pack.bytes / 1000)} kB` : null,
    },
  ];

  return (
    <>
      <section className="grid gap-10 pt-10 sm:pt-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.12fr)] lg:items-center lg:gap-12" aria-labelledby="hero-title">
        <div>
          <h1 id="hero-title" className="display text-[clamp(2.3rem,4.6vw,3.75rem)] text-ink">
            Try every tool in physics before you buy one.
          </h1>
          <p className="prose-measure mt-6 text-lg text-ink-2 sm:text-xl">{homeLede(runnerConfigured())}</p>
          <HeroAsk
            examples={heroes.map((h) => ({ href: `/studio/${h.hero}/`, title: h.title, setting: humanize(h.setting) }))}
            note={liveCopy("heroAsk")}
          />
        </div>
        {lane && rec ? (
          <HeroScene
            lane={lane}
            steps={heroSteps}
            setup={setupPhrase(rec)}
            useCase={h1!.title}
            posterDark={`${BASE_PATH}/posters/h1-recommended-dark.jpg`}
            posterLight={`${BASE_PATH}/posters/h1-recommended-light.jpg`}
          />
        ) : null}
      </section>

      <section className="mt-24" aria-labelledby="how-title">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <h2 id="how-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
            How it works
          </h2>
          <Link href="/studio/#examples" className="link text-sm">
            See the recorded runs
          </Link>
        </div>
        <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stages.map((s, i) => (
            <li key={s.name} className="panel flex flex-col p-5">
              <p className="flex items-center gap-3">
                <span className="num grid size-8 place-items-center rounded-md border border-accent/60 text-sm text-accent">{i + 1}</span>
                <span className="heading text-xl text-ink">{s.name}</span>
              </p>
              <p className="mt-3 flex-1 text-[0.9375rem] text-ink-2">{s.text}</p>
              {s.fact ? <p className="num mt-4 border-t border-rule pt-3 text-xs text-ink-3">{s.fact}</p> : null}
              <Link href={s.href} className="mt-3 text-sm text-accent no-underline hover:underline">
                {s.cta}
              </Link>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-24" aria-labelledby="proof-title">
        <h2 id="proof-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
          Tested before we said it
        </h2>
        <p className="prose-measure mt-2 text-ink-2">Two pre-declared tests, frozen before their final runs, with the data in the repository.</p>
        <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
          <article className="panel p-5 sm:p-6" data-proof="usecase" data-proof-version="v1">
            <p className="text-sm text-ink-3">Studio benchmark v1 (top-down wrist, 19 Oct): held-out conditions, three use cases</p>
            <div className="mt-4 grid gap-4">
              <div>
                <p className="flex items-baseline justify-between gap-3">
                  <span className="font-semibold text-accent">The Studio&apos;s pick</span>
                  <span className="num text-2xl text-ink">
                    {held.totals.recommended} of {held.totals.n}
                  </span>
                </p>
                <Bar value={held.totals.recommended} n={held.totals.n} tone="accent" />
              </div>
              <div>
                <p className="flex items-baseline justify-between gap-3">
                  <span className="font-semibold text-baseline">Best single off-the-shelf tool</span>
                  <span className="num text-2xl text-ink">
                    {held.totals.best_single} of {held.totals.n}
                  </span>
                </p>
                <Bar value={held.totals.best_single} n={held.totals.n} tone="baseline" />
              </div>
            </div>
            <p className="mt-4 text-sm text-ink-2">
              Runs that finished the whole task, on conditions neither setup was chosen on, with the v1 setups.{" "}
              {Object.values(held.heroes)
                .map((h) => `${h.title}: ${h.recommended.completed} of ${h.recommended.n}`)
                .join("; ")}
              .
            </p>
            <p className="mt-3 text-sm text-ink-2" data-proof-v2>
              v2 adds wrist tilt and honest tolerances; its benchmark is pending.
            </p>
            <Link href="/evidence/" className="mt-3 inline-block text-sm text-accent no-underline hover:underline">
              Read the benchmark
            </Link>
          </article>
          {s1 ? (
            <article className="panel p-5 sm:p-6" data-proof="fingers">
              <p className="text-sm text-ink-3">Finger designer, pre-registered</p>
              <p className="mt-4 flex items-baseline gap-3">
                <span className="num text-[2.75rem] leading-none text-ink">{Math.round(s1.effect.diff)}</span>
                <span className="text-ink-2">fewer designs to a finger that seats the part</span>
              </p>
              <p className="mt-3 text-sm text-ink-2">
                The language-model designer with feedback against equal-budget random search: {s1.effect.diff.toFixed(2)} fewer on average (
                {Math.round(s1.effect.level * 100)}% interval {s1.effect.ci_low.toFixed(1)} to {s1.effect.ci_high.toFixed(1)}), exact permutation test p ={" "}
                {s1.primary.p_value.toFixed(4)}.
              </p>
              <Link href="/results/" className="mt-3 inline-block text-sm text-accent no-underline hover:underline">
                Read the result
              </Link>
            </article>
          ) : null}
        </div>
        <p className="mt-3 text-sm text-ink-3">Simulated results. They rank tools under stated conditions; verify on hardware before you rely on them.</p>
      </section>

      {library ? (
        <section className="mt-24" aria-labelledby="library-title">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 id="library-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
                An open tool library
              </h2>
              <p className="prose-measure mt-2 text-ink-2">
                {library.counts.items} open, licence-checked end effectors with files, specs and a simulation status, alongside the families the Studio
                designs. These ones made it into a recommendation.
              </p>
            </div>
            <Link href="/library/" className="link text-sm">
              Browse the library
            </Link>
          </div>
          <ul className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {libCards.map((c) => (
              <li key={c.key}>
                <Link href={`/library/${c.key}/`} className="panel flex h-full flex-col p-4 no-underline hover:border-rule-strong">
                  <FamilyIcon family={kindFamily(c.kind)} className="size-7 text-accent" />
                  <span className="heading mt-3 text-base text-ink">{c.display_name}</span>
                  <span className="mt-1 text-sm text-ink-3">{c.kind_title}</span>
                  <span className="mt-auto pt-4 text-xs text-ink-2">Recommended in {usedIn.get(c.display_name)!.join(", ")}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="mt-24" aria-labelledby="docs-title">
        <h2 id="docs-title" className="heading text-[1.75rem] text-ink sm:text-[2rem]">
          Open source, end to end
        </h2>
        <p className="prose-measure mt-2 text-ink-2">The same code runs this site, the planner and the physics. Read how it works, or run it yourself.</p>
        <ul className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { href: "/docs/", title: "Docs", text: "Start here: what each part does and where to read more.", external: false },
            { href: repoBlob("docs/STUDIO.md"), title: "How the Studio works", text: "Plans, the screen, combinations and the ranking rule.", external: true },
            { href: repoBlob("docs/USECASE_V1.md"), title: "The benchmark protocol", text: "Frozen before the held-out runs, with its secret.", external: true },
            { href: "/docs/tutorials/run-it-yourself/", title: "Run it yourself", text: "Run the Studio on your own plan, with Python or Docker, on your machine.", external: false },
          ].map((d) => (
            <li key={d.title}>
              {d.external ? (
                <a href={d.href} rel="noopener" className="panel block h-full p-4 no-underline hover:border-rule-strong">
                  <span className="heading block text-base text-ink">
                    {d.title} <span aria-hidden="true">↗</span>
                  </span>
                  <span className="mt-1 block text-sm text-ink-2">{d.text}</span>
                </a>
              ) : (
                <Link href={d.href} className="panel block h-full p-4 no-underline hover:border-rule-strong">
                  <span className="heading block text-base text-ink">{d.title}</span>
                  <span className="mt-1 block text-sm text-ink-2">{d.text}</span>
                </Link>
              )}
            </li>
          ))}
        </ul>
      </section>

      <TitleBlock title="Effector Lab" drawing="/" sheet="Home" />
    </>
  );
}
