import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { StoryPage, type StoryHeldout } from "@/components/story/StoryPage";
import { type HeroPage } from "@/components/studio/schema";
import { buildStory, focusOf } from "@/components/studio/story";

import { V1_NOTE } from "../../evidence/notes";
import { getStudioHero, getStudioHeroIds, getStudioIndex, getStudioSession, legacyHeroRoutes } from "../data";
import { heldoutFor } from "../heldout";

export const dynamicParams = false;

/**
 * One story page per hero the Studio's data index lists, whatever their number, plus a redirect
 * for each published v1 route the index no longer lists (a renamed or left-out hero).
 */
export function generateStaticParams() {
  return [...getStudioHeroIds(), ...legacyHeroRoutes().map((r) => r.hero)].map((hero) => ({ hero }));
}

type Params = { params: Promise<{ hero: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const hero = (await params).hero;
  const page = getStudioHero(hero);
  if (page) return { title: `${page.title}: worked example`, description: `Product mode. ${buildStory(page).answer}` };
  const moved = legacyHeroRoutes().find((r) => r.hero === hero);
  return moved ? { title: "Worked example moved", description: "This worked example moved.", robots: { index: false } } : {};
}

/** An old route: a static redirect (meta refresh and a link), so links to it keep working. */
function Moved({ to, title }: { to: string; title: string | null }) {
  return (
    <>
      <meta httpEquiv="refresh" content={`0; url=${to}`} />
      <section className="pt-16" data-moved>
        <h1 className="display text-[2.2rem] text-ink">This worked example moved</h1>
        <p className="mt-4 text-ink-2">
          {title ? `It is now "${title}". ` : "It is no longer one of the recorded examples. "}
          <a className="link" href={to}>
            {title ? "Open it" : "See the worked examples"}
          </a>
          .
        </p>
      </section>
    </>
  );
}

/** The held-out benchmark's arms when they are the setups this page shows (v1 heroes only). */
function heldoutOf(page: HeroPage): StoryHeldout | null {
  const focus = focusOf(page);
  if (!focus || focus.role !== "recommended" || !page.best_single) return null;
  const h = heldoutFor(page.case_id, focus.setup_id, page.best_single.setup_id);
  if (!h) return null;
  const note = V1_NOTE.byHero[page.case_id];
  return {
    n_conditions: h.n_conditions,
    p: h.p,
    recommended: { n: h.recommended.n, completed: h.recommended.completed },
    best_single: { n: h.best_single.n, completed: h.best_single.completed, steps: h.best_single.steps },
    note: note ? { date: V1_NOTE.date, text: note } : null,
  };
}

export default async function StudioStoryPage({ params }: Params) {
  const hero = (await params).hero;
  const page = getStudioHero(hero);
  if (!page) {
    const moved = legacyHeroRoutes().find((r) => r.hero === hero);
    if (moved) return <Moved to={moved.to} title={moved.title} />;
    notFound();
  }
  const others = getStudioIndex()?.heroes.filter((h) => h.hero !== page.hero) ?? [];
  return <StoryPage page={page} session={getStudioSession(page)} held={heldoutOf(page)} others={others} />;
}
