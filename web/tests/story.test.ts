import { readFileSync } from "node:fs";
import path from "node:path";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { createPlaybackStore, PlaybackContext } from "@/components/replay/playback";
import { stepActive } from "@/components/story/StoryPlayback";
import { StoryPage } from "@/components/story/StoryPage";
import { changeCardAt, parseStudioHero, PRODUCT_MODE_LABEL, type Recommendation, type SetupKind } from "@/components/studio/schema";
import { answerSentence, buildStory, failurePhrase, focusOf, laneName, laneStops, noteLead, ruledOutGroup, setupPhrase, stepNotes, stepVerb } from "@/components/studio/story";
import { ChangeCardOverlay } from "@/components/studio/StudioLanes";

import { sixHeroes } from "./helpers/heroes";

/** The published v1 hero pages (h1..h3), kept as fixtures when the v2 heroes replaced them. */
const DATA = path.join(__dirname, "fixtures", "studio_v1", "data", "studio");
const load = (hero: string) => parseStudioHero(JSON.parse(readFileSync(path.join(DATA, `${hero}.json`), "utf8")), `studio/${hero}.json`);

describe("story words", () => {
  it("turns step ids into verb phrases", () => {
    expect(stepVerb("open_lid")).toBe("open the lid");
    expect(stepVerb("tighten_coupling")).toBe("tighten the coupling");
    expect(stepVerb("push")).toBe("push");
  });

  it("groups pruning reasons like the exporter", () => {
    expect(ruledOutGroup("carton: the part is 80.0 mm across the closing axis; the gripper opens 52.0 mm")).toBe("opens narrower than the part");
    expect(ruledOutGroup("a suction cup has no hook or fingers to engage a handle or lip")).toBe("a cup has nothing to hook with");
    expect(ruledOutGroup(null)).toBe("not screened on this step");
    // the exporter's plain words for the checks' codes group like the codes
    expect(ruledOutGroup("Parcel: too narrow: open, digit 1 (at 90 deg on the part) has its inner face 35.0 mm from the centre")).toBe("digits stand too close for the part");
    expect(ruledOutGroup("Test port: digits collide: closing over the probe part, digits 1 and 2 run into each other")).toBe("digits collide when they close");
  });
});

describe("the returns tote (h1) story, from the exported page", () => {
  const page = load("h1");
  const story = buildStory(page);

  it("answers first with the recommendation against the best single tool", () => {
    expect(answerSentence(page)).toBe(
      "A two-tool head (a soft arm plus a suction cup) finishes the task in 8 of 8 runs; the best single off-the-shelf tool finishes 0 of 8: it can't open the lid.",
    );
    expect(setupPhrase(page.recommendation!)).toBe("two-tool head (a soft arm plus a suction cup)");
  });

  it("says so when the pick is itself the best single off-the-shelf tool", () => {
    const same = { ...page, best_single: { ...page.best_single!, setup_id: page.recommendation!.setup_id } };
    expect(answerSentence(same)).toBe("A two-tool head (a soft arm plus a suction cup) finishes the task in 8 of 8 runs; it is also the best single off-the-shelf tool.");
  });

  it("counts the lid step from the success matrix", () => {
    const lid = story.steps[0]!;
    expect(lid.step.id).toBe("open_lid");
    expect(lid.n_candidates).toBe(52);
    expect(lid.tried).toBe(37);
    expect(lid.passed.map((p) => p.family_title)).toEqual(["Soft arm", "Soft arm", "Soft arm"]);
    expect(lid.passed.length + lid.partial.length + lid.failed.length + lid.ruledOut.length).toBe(52);
    expect(lid.runFailures).toEqual({ slip: 95, constraint_exceeded: 7 });
    expect(lid.bar.map((b) => [b.label, b.tools])).toEqual([
      ["passed", 3],
      ["slipped off the lip", 32],
      ["pushed over the 20 N limit", 2],
      ["ruled out before physics", 15],
    ]);
    expect(lid.why[0]).toBe("Only soft arms could open the lid: 3 of 37 tools tried, all soft arms.");
    expect(lid.why.join(" ")).toContain("faster (62.57 s against 62.67 s) and lighter (373 g against 376 g)");
  });

  it("explains the second tool from the steps it serves", () => {
    const carton = story.steps[1]!;
    expect(carton.pick?.tool_label).toBe("Suction cup, 40 mm bellows");
    expect(carton.why[0]).toBe(
      "The head's second tool does steps 2 to 4, so it has to pass all three: 4 of the 52 tools did (an industrial gripper and three suction cups).",
    );
    expect(carton.why[1]).toContain("weighs 1.09 kg on its own; the whole recommended setup weighs 373 g");
    expect(carton.whyGap).toMatch(/^That leaves three suction cups that pass: the Studio's ranking chose between them/);
    expect(story.steps[0]!.whyGap).toMatch(/^One other passing tool was not in a finalist/);
    expect(story.steps[2]!.why).toEqual(["Same tool as step 2, already on the wrist; it was chosen there for the steps it has to cover."]);
  });

  it("says why it takes two tools", () => {
    expect(story.whySetup[1]).toBe(
      "Step 1 is the bottleneck: only three soft arms could open the lid, and none of them passes any other step (steps 2 and 3: the part is wider than the arm reaches; step 4: a slack arm has no stiff face to push with).",
    );
  });

  it("says where the best single tool stops", () => {
    expect(story.steps[0]!.baseline?.text).toBe("Stops here in 8 of 8 runs: slipped off the lip in 5 and pushed over the 20 N limit in 3.");
    expect(story.steps[1]!.baseline?.tone).toBe("muted");
  });

  it("phrases failure modes per step", () => {
    const lid = page.plan.steps[0]!;
    expect(failurePhrase("slip", lid, page.plan)).toBe("slipped off the lip");
  });
});

describe("a setup in words names its tools, not their families", () => {
  // every tool is given the same family on purpose: the phrase must come from the tools' own names, as the lanes show them
  const setup = (kind: SetupKind, ...names: string[]) =>
    ({ kind, label: `${kind}: ${names.join(" > ")}`, tools: names.map((display_name, index) => ({ index, display_name, family_title: "Industrial gripper" })) }) as unknown as Pick<
      Recommendation,
      "kind" | "tools" | "label"
    >;

  it("lists a tool change's tools in order, each by its name without the spec after the first comma", () => {
    expect(setupPhrase(setup("tool_change", "Drawer-pull hook tool", "Force-controlled parallel gripper, 48 mm stroke", "Suction cup, 25 mm bellows"))).toBe(
      "tool change (a drawer-pull hook tool, then a force-controlled parallel gripper, then a suction cup)",
    );
  });

  it("joins a two-tool head's tools with plus", () => {
    expect(setupPhrase(setup("multi_head", "Vacuum area frame, 2 x 2 cups, 300 mm", "Tamper pad, 200 mm"))).toBe("two-tool head (a vacuum area frame plus a tamper pad)");
    expect(setupPhrase(setup("multi_head", "Soft arm, 12 segments, 140 mm", "Suction cup, 40 mm bellows"))).toBe("two-tool head (a soft arm plus a suction cup)");
  });

  it("counts instead of listing when every tool is the same kind", () => {
    expect(setupPhrase(setup("tool_change", "Suction cup, 10 mm bellows", "Suction cup, 25 mm bellows"))).toBe("tool change between two suction cups");
    expect(setupPhrase(setup("multi_head", "Suction cup, 10 mm bellows", "Suction cup, 25 mm bellows"))).toBe("two-tool head with two suction cups");
    expect(setupPhrase(setup("tool_change", "Parallel jaw, flat pad: Parcel", "Parallel jaw, stock blade: Parcel", "Parallel jaw, flat pad: Parcel"))).toBe("tool change between three parallel jaws");
  });

  it("never says the same name twice in a row: tools whose short names collide keep their full names", () => {
    expect(setupPhrase(setup("tool_change", "Two-finger adaptive gripper, pinch", "Two-finger adaptive gripper, power grasp", "Suction cup, 25 mm bellows"))).toBe(
      "tool change (a two-finger adaptive gripper, pinch, then a two-finger adaptive gripper, power grasp, then a suction cup)",
    );
  });

  it("keeps a name with no spec whole, and puts the right article before each", () => {
    expect(setupPhrase(setup("tool_change", "Air-actuated mini parallel gripper", "Suction cup, 10 mm bellows", "One-finger test hand", "Servo screwdriver attachment for a hobby arm"))).toBe(
      "tool change (an air-actuated mini parallel gripper, then a suction cup, then a one-finger test hand, then a servo screwdriver attachment for a hobby arm)",
    );
  });

  it("gives a single tool its full name, and a setup with no tools its label", () => {
    expect(setupPhrase(setup("single", "Industrial servo parallel gripper, 110 mm stroke"))).toBe("single industrial servo parallel gripper, 110 mm stroke");
    expect(setupPhrase({ kind: "tool_change", label: "Tool change: A > B", tools: [] })).toBe("tool change: A > B");
  });

  it("starts with the kind, up to the first comma or bracket, which the home hero's caption quotes", () => {
    const kindOf = (phrase: string) => phrase.split(/\s*[,(]/)[0];
    expect(kindOf(setupPhrase(setup("single", "Industrial servo parallel gripper, 110 mm stroke")))).toBe("single industrial servo parallel gripper");
    expect(kindOf(setupPhrase(setup("multi_head", "Soft arm, 12 segments, 140 mm", "Suction cup, 40 mm bellows")))).toBe("two-tool head");
    expect(kindOf(setupPhrase(setup("tool_change", "Drawer-pull hook tool", "Suction cup, 25 mm bellows")))).toBe("tool change");
  });

  // the published job pages, as the site serves them
  const PUBLISHED = path.join(__dirname, "..", "public", "data", "studio");
  const heroes = (JSON.parse(readFileSync(path.join(PUBLISHED, "index.json"), "utf8")).heroes as { hero: string }[]).map((h) => ({
    hero: h.hero,
    page: parseStudioHero(JSON.parse(readFileSync(path.join(PUBLISHED, `${h.hero}.json`), "utf8")), `studio/${h.hero}.json`),
  }));

  it("reads the hotel page's headline by its three tools, not 'industrial gripper then industrial gripper then suction cup'", () => {
    const hotel = heroes.find((h) => h.hero === "h7_hotel")!.page;
    expect(hotel.recommendation!.kind).toBe("tool_change");
    expect(setupPhrase(hotel.recommendation!)).toBe("tool change (a drawer-pull hook tool, then a force-controlled parallel gripper, then a suction cup)");
    const answer = answerSentence(hotel);
    expect(answer).toMatch(
      /^A tool change \(a drawer-pull hook tool, then a force-controlled parallel gripper, then a suction cup\) finishes the task in \d+ of \d+ runs; the best single off-the-shelf tool finishes \d+ of \d+/,
    );
    expect(answer).not.toMatch(/industrial gripper/i);
    expect(answer).not.toContain("), finishes");
  });

  it("reads every published page's headline by the names of the tools it picks", () => {
    expect(heroes.length).toBeGreaterThanOrEqual(4);
    for (const { hero, page } of heroes) {
      const rec = page.recommendation;
      if (!rec) continue;
      const answer = answerSentence(page);
      expect(answer, hero).toMatch(/^A (?:single |two-tool head|tool change)/);
      expect(answer, hero).toContain(" finishes the task in ");
      // no family title stands in for a tool, and a bracketed list is not followed by a stray comma
      expect(answer, hero).not.toMatch(/process tool|industrial gripper then|\), finishes/i);
      if (rec.kind === "single") continue;
      for (const t of rec.tools) {
        const own = t.display_name.split(",")[0]!.trim().toLowerCase();
        expect(answer.toLowerCase(), `${hero}: ${own}`).toContain(own.endsWith("s") ? own.slice(0, -1) : own);
      }
    }
  });

  it("keeps the single-tool headline as it was, with its spec set off by commas", () => {
    const tote = heroes.find((h) => h.hero === "h1_tote")!.page;
    expect(tote.recommendation!.kind).toBe("single");
    expect(answerSentence(tote)).toMatch(/^A single industrial servo parallel gripper, 110 mm stroke, finishes the task in \d+ of \d+ runs/);
  });

  it("reads the solar page's tool change and the tile page's two-tool head by their tools", () => {
    expect(setupPhrase(heroes.find((h) => h.hero === "h5_solar")!.page.recommendation!)).toBe("tool change (a vacuum area frame, then a suction cup, then a nut runner)");
    expect(setupPhrase(heroes.find((h) => h.hero === "h6_tile")!.page.recommendation!)).toBe("two-tool head (a vacuum area frame plus a tamper pad)");
  });
});

describe("step notes", () => {
  const LID = "The lid fell shut from 21 deg and met its stop at about 320 deg/s: the tool let go of it above its stop";
  const pick = (caveats: string[]) => ({ caveats, fidelity: "approximation" as const });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const laneStep = (q: Record<string, number>): any => ({ step_id: "close_lid", failure_mode: null, quantities: q });

  it("say where a run note comes from: the replayed run, or a recorded run's caveat", () => {
    const fromCaveat = stepNotes(pick([LID]), null);
    expect(fromCaveat.map((n) => [n.kind, n.source])).toEqual([["run", "recorded"], ["model", null]]);
    expect(noteLead(fromCaveat[0]!)).toBe("In a recorded run: ");
    const fromLane = stepNotes(pick([]), laneStep({ lid_release_deg: 21.3, lid_stop_dps: 317.4 }));
    expect(noteLead(fromLane[0]!)).toBe("In this run: ");
    expect(noteLead(fromLane[1]!)).toBe("Caveat: ");
  });

  it("keep the caveat's explanation when the replayed run's measurement of the same event wins", () => {
    const notes = stepNotes(pick([LID]), laneStep({ lid_release_deg: 21.3, lid_stop_dps: 317.4 }));
    const run = notes.filter((n) => n.kind === "run");
    expect(run).toHaveLength(1);
    expect(run[0]).toMatchObject({ label: "lid fell shut from 21° at 317°/s", source: "lane", detail: LID });
  });
});

describe("a step card lights while its step plays", () => {
  it("and stays lit when the run ends on a short last step", () => {
    // the hotel's closest setup stops at the door pull, 0.2 s long, at the end of the run
    expect(stepActive(12.0, true, 11.974, 12.174, 12.174)).toBe(true);
    expect(stepActive(12.174, false, 11.974, 12.174, 12.174)).toBe(true);
    expect(stepActive(12.17, false, 11.974, 12.174, 12.17)).toBe(true);
    // an earlier step goes dark once the run has moved past it
    expect(stepActive(12.174, false, 0, 10.474, 12.174)).toBe(false);
    expect(stepActive(5, true, 0, 10.474, 12.174)).toBe(true);
    expect(stepActive(0, false, 0, 10.474, 12.174)).toBe(false); // not started
  });
});

describe("every hero builds a story", () => {
  for (const hero of ["h1", "h2", "h3"]) {
    it(hero, () => {
      const page = load(hero);
      const story = buildStory(page);
      expect(story.steps.length).toBe(page.plan.steps.length);
      for (const s of story.steps) {
        expect(s.passed.length + s.partial.length + s.failed.length + s.ruledOut.length).toBe(page.screen.n_candidates);
        expect(s.bar.reduce((n, b) => n + b.tools, 0)).toBe(page.screen.n_candidates);
      }
      if (process.env.STORY_DUMP) {
        console.log(`\n===== ${hero}\n${story.answer}\nwhySetup: ${story.whySetup.join(" | ")}`);
        for (const s of story.steps) {
          console.log(`-- ${s.index + 1} ${s.name}: ${s.passed.length} of ${s.tried} tried (of ${s.n_candidates}); pick ${s.pick?.tool_label}`);
          console.log(`   bar: ${s.bar.map((b) => `${b.tools} ${b.label}`).join(" / ")}`);
          for (const w of s.why) console.log(`   why: ${w}`);
          console.log(`   baseline: ${s.baseline?.text}`);
        }
      }
    });
  }
});

// ---------------------------------------------------------------- N heroes, recommendation or closest

describe("story pages for the six v2 heroes", () => {
  // the six synthetic hero pages of the test fixtures (helpers/heroes.ts), CNC included: the site
  // publishes the heroes whose plans are valid and whose lanes pass the replay gate, and renders whatever
  // set its index lists
  const { pages, entries } = sixHeroes();

  it("builds six pages from whatever heroes the index lists", () => {
    expect(pages.map((p) => p.hero)).toEqual(["h1_tote", "h4_cnc", "h5_solar", "h6_tile", "h2_coupling", "h7_hotel"]);
    expect(pages.filter((p) => p.recommendation).length).toBe(4);
    expect(pages.filter((p) => p.no_winner).length).toBe(2);
  });

  for (const hero of ["h1_tote", "h4_cnc", "h5_solar", "h6_tile", "h2_coupling", "h7_hotel"]) {
    it(`${hero}: the story page renders a recommendation or a closest setup`, () => {
      const page = pages.find((p) => p.hero === hero)!;
      const others = entries.filter((e) => e.hero !== hero);
      const html = renderToStaticMarkup(createElement(StoryPage, { page, session: null, held: null, others }));
      const focus = focusOf(page)!;
      expect(focus).not.toBeNull();
      expect(html).toContain("data-answer");
      expect(html).toContain("Simulated, verify on hardware");
      expect(html).toContain(PRODUCT_MODE_LABEL);
      expect(html).toContain(`data-fidelity="${focus.fidelity}"`);
      // one step card per plan step, each naming the focus setup's tool
      expect(html.match(/data-step-card=/g)?.length).toBe(page.plan.steps.length);
      expect(html.match(new RegExp(`data-pick-role="${focus.role}"`, "g"))?.length).toBe(page.plan.steps.length);
      // (outside the Next runtime a link drops its trailing slash)
      for (const o of others) expect(html).toMatch(new RegExp(`href="/studio/${o.hero}/?"`));
      if (page.recommendation) {
        expect(focus.role).toBe("recommended");
        expect(html).not.toContain("data-no-winner");
        expect(html).toContain(">Recommended<");
      } else {
        expect(focus.role).toBe("closest");
        expect(html).toContain('data-outcome-badge="closest"');
        expect(html).toContain("data-no-winner");
        expect(html).toContain("No setup completes this task");
        expect(html).toContain(">Closest setup<");
        expect(answerSentence(page)).toMatch(/^No setup finishes this task\. The closest, a .+, finishes 0 of 8 runs: it can't /);
      }
    });
  }

  it("each lane's stop marker names that lane's own failure", () => {
    for (const hero of ["h6_tile", "h7_hotel"]) {
      const page = pages.find((p) => p.hero === hero)!;
      const html = renderToStaticMarkup(createElement(StoryPage, { page, session: null, held: null, others: [] }));
      const stops = laneStops(page);
      const tracks = Object.fromEntries(
        html
          .split('data-track="')
          .slice(1)
          .map((chunk) => [chunk.slice(0, chunk.indexOf('"')), chunk]),
      );
      const markers: Record<string, string> = {};
      for (const lane of page.lanes) {
        const m = /data-stop-marker[^>]*>Stops here: <!-- -->([^<]*)</.exec(tracks[lane.id] ?? "") ?? /data-stop-marker[^>]*>Stops here: ([^<]*)</.exec(tracks[lane.id] ?? "");
        if (m) markers[lane.id] = m[1]!.replace(/&#x27;/g, "'");
      }
      // The closest setup stops at the last step (slipped), its own reason, not the baseline's.
      expect(stops.closest?.phrase, hero).toBeTruthy();
      expect(markers.closest, hero).toBe(stops.closest!.phrase);
      if (markers.best_single) expect(markers.best_single).toBe(stops.best_single!.phrase);
      expect(stops.closest!.step).toBe(page.plan.steps.length);
      // the text equivalent says it in words, per lane
      expect(html).toContain(`the closest setup (`);
      expect(html).toContain(`stops at step ${stops.closest!.step}`);
    }
    // the v1 tote: the recommended setup finishes, the best single tool stops at the lid
    const tote = load("h1");
    expect(laneStops(tote)).toEqual({ recommended: null, best_single: { step: 1, phrase: "slipped off the lip" } });
  });

  it("explains the closest setup step by step and where it stops", () => {
    const page = pages.find((p) => p.hero === "h6_tile")!;
    const story = buildStory(page);
    expect(story.steps.every((s) => s.pick !== null)).toBe(true);
    const last = story.steps[story.steps.length - 1]!;
    expect(last.why[0]).toBe("This is where it stops: in 8 of 8 runs it slipped while turning.");
    expect(story.steps[0]!.why[0]).toMatch(/^It passed this step in \d of 8 full-task runs\.$/);
    expect(story.whySetup).toEqual([]);
    expect(story.rankingReason).toBeNull();
  });

  it("keeps the three published v1 heroes unchanged", () => {
    for (const hero of ["h1", "h2", "h3"]) {
      const page = load(hero);
      expect(focusOf(page)?.role).toBe("recommended");
      expect(focusOf(page)?.setup_id).toBe(page.recommendation!.setup_id);
      expect(laneName(page.lanes[0]!)).toBe("Recommended");
    }
  });
});

describe("replay cuts show the card", () => {
  const h1 = load("h1");
  const h3 = load("h3");
  const headLane = h1.lanes.find((l) => l.id === "recommended")!;
  const changeLane = h3.lanes.find((l) => l.id === "recommended")!;

  it("covers a head switch and a tool change for their whole time", () => {
    // h1: the head switch runs from 16.692 s to 18.192 s
    expect(changeCardAt(headLane, 5)).toBeNull();
    const card = changeCardAt(headLane, 17.442)!;
    expect(card.title).toBe("Changing tool");
    expect(card.detail).toBe("Head switch: the wrist turns the next tool to the work, a fixed 1.5 s, not simulated");
    expect(card.next).toBe("Next: tool 2, Suction cup, 40 mm bellows");
    expect(card.progress).toBeCloseTo(0.5, 6);
    expect(changeCardAt(headLane, 18.2)).toBeNull();
    // h3: three tool changes of 6 s
    expect(changeCardAt(changeLane, 13)?.detail).toBe("Tool change: a fixed 6 s, not simulated");
    expect(changeCardAt(changeLane, 47)?.next).toBe("Next: tool 2, Force-controlled parallel gripper, 48 mm stroke");
  });

  it("covers a keyframed replay's own jump between the frames either side of a cut", () => {
    const best = h1.lanes.find((l) => l.id === "best_single")!;
    expect(changeCardAt(best, 4, { start_s: 3, end_s: 10.5 })?.detail).toBe("Not simulated: the replay holds for 7.5 s, then cuts to the next segment");
    expect(changeCardAt(best, 2.9, { start_s: 3, end_s: 10.5 })).toBeNull();
    expect(changeCardAt(best, 10.5, { start_s: 3, end_s: 10.5 })).toBeNull();
    expect(changeCardAt(best, 4, { start_s: 3.95, end_s: 4.05 })).not.toBeNull();
    expect(changeCardAt(best, 4, { start_s: 3.95, end_s: 4.05 }, 0.25)).toBeNull();
  });

  it("renders over the replay while the clock is in the change", () => {
    const store = createPlaybackStore();
    // A server render reads the store's initial state; point it at the live one.
    store.getInitialState = store.getState;
    store.getState().setDuration(70);
    const render = () => renderToStaticMarkup(createElement(PlaybackContext.Provider, { value: store }, createElement(ChangeCardOverlay, { lane: headLane, replay: null })));
    store.getState().seek(17);
    const html = render();
    expect(html).toContain("data-change-card");
    expect(html).toContain("Changing tool");
    expect(html).toContain("Next: tool 2, Suction cup, 40 mm bellows");
    store.getState().seek(30);
    expect(render()).toBe("");
  });
});
