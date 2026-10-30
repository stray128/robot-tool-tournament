/**
 * The home hero's replay clock and the site's navigation. The hero once set React state on every
 * animation frame; that urgent render, plus the 3D frames, kept the router's navigation from
 * finishing, so "Plan it in the Studio" or a header link could do nothing once the page had
 * settled. Its readout now follows the clock through watchPlayback, which sets state only when the
 * readout changes, and the replay stops when the visitor leaves through a link or the question box.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { getStudioHero, featuredHero } from "@/app/studio/data";
import { heroReadout, type HeroReadout, sameReadout } from "@/components/home/heroReadout";
import { announceLeave, leavesPage, onLeave } from "@/components/replay/leaving";
import { createPlaybackStore, watchPlayback } from "@/components/replay/playback";
import { stepName } from "@/components/studio/story";

function homeHero() {
  const featured = featuredHero();
  const page = featured ? getStudioHero(featured.hero) : undefined;
  const lane = page?.lanes.find((l) => l.id === "recommended");
  if (!page || !lane) throw new Error("no featured hero with a recommended lane");
  const steps = page.plan.steps.map((s) => ({ id: s.id, name: stepName(s.id), description: s.description }));
  return { lane, steps };
}

describe("the home hero's readout", () => {
  it("sets state only when the readout changes, and shows what a per-frame readout would", () => {
    const { lane, steps } = homeHero();
    const store = createPlaybackStore({ loop: true, look: "studio", view: "iso" });
    store.getState().setDuration(lane.duration_s);
    const shown: HeroReadout[] = [];
    const stop = watchPlayback(store, (s) => heroReadout(lane, steps, s.t), (r) => shown.push(r), sameReadout);
    store.getState().play();
    const dt = 1 / 60;
    let frames = 0;
    let ended = false;
    while (!ended) {
      ended = store.getState().advance(dt);
      frames += 1;
      // what the page shows is exactly what computing the readout on this frame gives
      expect(shown.at(-1)).toEqual(heroReadout(lane, steps, store.getState().t));
    }
    stop();
    expect(frames).toBeGreaterThan(600);
    // one update per step, tool change, pause between steps and the end, not one per frame
    for (let i = 1; i < shown.length; i++) expect(sameReadout(shown[i - 1]!, shown[i]!)).toBe(false);
    expect(shown.length).toBeLessThanOrEqual(2 * steps.length + 4);
    expect(shown.at(-1)!.done).toBe(true);
    // after the unsubscribe the clock no longer reaches React state
    const after = shown.length;
    store.getState().seek(0);
    expect(shown.length).toBe(after);
  });

  it("is the only way HeroScene's React state follows the clock", () => {
    const src = readFileSync(path.join(__dirname, "..", "components", "home", "HeroScene.tsx"), "utf8");
    expect(src).not.toMatch(/\.subscribe\(/);
    expect(src).toMatch(/watchPlayback\(store, \(s\) => heroReadout\(/);
  });

  it("follows the play state only when it changes", () => {
    const store = createPlaybackStore({ loop: true });
    store.getState().setDuration(10);
    const playing: boolean[] = [];
    const stop = watchPlayback(store, (s) => s.playing, (p) => playing.push(p));
    store.getState().play();
    for (let i = 0; i < 120; i++) store.getState().advance(1 / 60);
    store.getState().pause();
    stop();
    expect(playing).toEqual([false, true, false]);
  });
});

type FakeAnchor = { target: string; download?: boolean; href: string };

function click(anchor: FakeAnchor | null, mods: Partial<Pick<MouseEvent, "button" | "metaKey" | "ctrlKey" | "shiftKey" | "altKey">> = {}) {
  const a = anchor && {
    target: anchor.target,
    getAttribute: (k: string) => (k === "href" ? anchor.href : null),
    hasAttribute: (k: string) => k === "download" && !!anchor.download,
  };
  return { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, ...mods, target: { closest: () => a } as unknown as EventTarget };
}

describe("leaving the home page", () => {
  const here = { href: "https://example.org/robot-tool-tournament/" };
  const link = (href: string, extra: Partial<FakeAnchor> = {}): FakeAnchor => ({ target: "", href, ...extra });

  it("counts a plain click on a link to another page of the site", () => {
    expect(leavesPage(click(link("/robot-tool-tournament/studio/")), here)).toBe(true);
    expect(leavesPage(click(link("/robot-tool-tournament/studio/#examples")), here)).toBe(true);
    expect(leavesPage(click(link("/robot-tool-tournament/library", { target: "_self" })), here)).toBe(true);
  });

  it("does not count clicks that keep this page open", () => {
    expect(leavesPage(click(null), here)).toBe(false);
    expect(leavesPage(click(link("/robot-tool-tournament/")), here)).toBe(false);
    expect(leavesPage(click(link("/robot-tool-tournament")), here)).toBe(false);
    expect(leavesPage(click(link("#how-title")), here)).toBe(false);
    expect(leavesPage(click(link("https://github.com/x/y")), here)).toBe(false);
    expect(leavesPage(click(link("/robot-tool-tournament/studio/", { target: "_blank" })), here)).toBe(false);
    expect(leavesPage(click(link("/robot-tool-tournament/studio/", { download: true })), here)).toBe(false);
    for (const mod of [{ button: 1 }, { metaKey: true }, { ctrlKey: true }, { shiftKey: true }, { altKey: true }])
      expect(leavesPage(click(link("/robot-tool-tournament/studio/"), mod), here)).toBe(false);
  });

  it("tells the replay when the question box starts a navigation", () => {
    let calls = 0;
    const doc = globalThis.document;
    // onLeave also listens for link clicks on the document; a stand-in is enough here
    globalThis.document = { addEventListener: () => {}, removeEventListener: () => {} } as unknown as Document;
    try {
      const stop = onLeave(() => (calls += 1));
      announceLeave();
      expect(calls).toBe(1);
      stop();
      announceLeave();
      expect(calls).toBe(1);
    } finally {
      globalThis.document = doc;
    }
  });
});
