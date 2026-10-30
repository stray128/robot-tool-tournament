/**
 * The sentences that say where live planning happens (lib/live_copy.ts). With NEXT_PUBLIC_RUNNER_URL
 * unset every one of them reads exactly as the approved public copy; with it set they read the hosted
 * variant. The pages are rendered both ways, so a sentence that was missed, or one that checks the flag
 * on its own, shows up here.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { HeroAsk } from "@/components/home/HeroAsk";
import { homeLede, LEDE_HEAD } from "@/components/home/lede";
import catalogJson from "@/components/studio/intake/catalog.json";
import IntakeWorkbench, { type RecordedPlans } from "@/components/studio/intake/IntakeWorkbench";
import recordedJson from "@/components/studio/intake/recorded.json";
import type { Catalog } from "@/components/studio/intake/taskspec";
import { LIVE_COPY, type LiveCopyKey, liveCopy } from "@/lib/live_copy";

import Home from "../app/page";
import PlanATaskPage from "../app/docs/tutorials/plan-a-task/page";
import QuickstartPage from "../app/docs/quickstart/page";
import StudioPage from "../app/studio/page";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: () => {} }) }));

const RUNNER = "https://runner.example";
const catalog = catalogJson as unknown as Catalog;
const recorded = recordedJson as unknown as RecordedPlans;

afterEach(() => vi.unstubAllEnvs());

/** The approved public text, word for word, as it stood before the hosted runner. Do not edit these. */
const APPROVED: Record<LiveCopyKey, string> = {
  homeCard: "Here you replay recorded plans and edit their steps; live planning runs on your own machine.",
  lede: "This site replays recorded plans and runs; live planning runs on your own machine.",
  heroAsk: "This site runs no model: it replays recorded plans, and you can build yours by hand. To plan new words,",
  quickstartLedeHead: "Everything here runs in your browser: no install, no account, no key.",
  quickstartLede: "Live planning of new words runs on your own machine.",
  quickstartNote:
    "The site has no server, so Plan replays the planner's recorded run for the examples. To plan new words live, run the Studio on your own machine (step 4).",
  studioTip:
    "This public site has no backend. It replays the recorded plans and the recorded test runs, and lets you edit and download a plan. To plan and test your own text, run the Studio locally with your own model key.",
  planTaskNote:
    "The public site has no server. Plan replays the recorded run of each example, model calls included, so you see what the planner did. To plan your own words live, run the Studio locally with your own Token Factory key (",
  intakeBox: "This static site has no backend: Plan replays the recorded live plan of an example, model calls included.",
  intakeMessage:
    "This public site runs in Explore mode: it has no model to call, so it can only replay the recorded plans of the examples. Pick an example, or build your plan by hand below. To plan your own words, run the Studio locally with its API.",
};

/** The variants for a build with the runner. The first three are the wording the brief set. */
const HOSTED: Record<LiveCopyKey, string> = {
  homeCard: "Here you replay recorded plans and edit their steps, or plan a new job live with your own NVIDIA or Token Factory key, or a small free tier.",
  lede: "This site replays recorded plans and runs, and can plan and test a new job live.",
  heroAsk: "This site replays recorded plans, and you can build yours by hand. To plan new words, open the Studio and use your own key or a small free tier, or",
  quickstartLedeHead: "Browsing and editing run in your browser: no install and no account.",
  quickstartLede: "Live planning of new words runs here with your own key or a small free tier, or on your own machine.",
  quickstartNote:
    "Plan replays the planner's recorded run for the examples. To plan new words live, use your own key or a small free tier in the “Plan and run it live” panel, or run the Studio on your own machine (step 4).",
  studioTip:
    "This public site replays the recorded plans and the recorded test runs, and lets you edit and download a plan. To plan and test your own text, use your own key or a small free tier in the “Plan and run it live” panel, or run the Studio locally with your own model key.",
  planTaskNote:
    "Plan replays the recorded run of each example, model calls included, so you see what the planner did. To plan your own words live, use your own key or a small free tier in the “Plan and run it live” panel, or run the Studio locally with your own Token Factory key (",
  intakeBox: "Plan replays the recorded plan of an example, model calls included. To plan your own words live, use the panel below.",
  intakeMessage:
    "This public site runs in Explore mode, so Plan can only replay the recorded plans of the examples. Pick an example, or build your plan by hand below. To plan your own words, use your own key or a small free tier in the “Plan and run it live” panel, or run the Studio locally with its API.",
};

/** The quickstart lede's middle sentence, the same in both builds. */
const QUICKSTART_MIDDLE = "The site replays recorded plans and runs: open a worked example, edit a recorded plan or build one by hand, and find a job like yours.";

const KEYS = Object.keys(APPROVED) as LiveCopyKey[];

/** The text a page renders, with the markup's entity escapes undone so a sentence can be searched for as written. */
function textOf(html: string): string {
  return html
    .replace(/<!-- -->/g, "")
    .replaceAll("&#x27;", "'")
    .replaceAll("&quot;", '"')
    .replaceAll("&amp;", "&");
}

describe("liveCopy", () => {
  it("has a sentence for every place, and no key without a test", () => {
    expect(Object.keys(LIVE_COPY).sort()).toEqual([...KEYS].sort());
    expect(Object.keys(HOSTED).sort()).toEqual([...KEYS].sort());
  });

  it("with the flag unset gives the exact approved text", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    for (const key of KEYS) expect(liveCopy(key), key).toBe(APPROVED[key]);
  });

  it("with the flag set gives the hosted variant", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    for (const key of KEYS) expect(liveCopy(key), key).toBe(HOSTED[key]);
  });

  it("an unusable runner URL counts as unset, so a key never travels in the clear", () => {
    for (const bad of ["http://runner.example", "not a url", "   "]) {
      vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", bad);
      for (const key of KEYS) expect(liveCopy(key), `${key} with ${bad}`).toBe(APPROVED[key]);
    }
  });

  it("a test can ask for either wording without touching the environment", () => {
    for (const key of KEYS) {
      expect(liveCopy(key, false)).toBe(APPROVED[key]);
      expect(liveCopy(key, true)).toBe(HOSTED[key]);
    }
  });

  it("the hosted wording follows the writing rules: no em dash, no vendor brand but the sponsors, and it still allows your own machine or the key", () => {
    for (const key of KEYS) {
      const text = HOSTED[key];
      expect(text, key).not.toMatch(/—|–| -- /);
      for (const brand of ["OpenAI", "Anthropic", "Claude", "GPT", "Gemini", "AWS", "Amazon", "Google"]) expect(text, `${key} names ${brand}`).not.toContain(brand);
      // every hosted sentence is about live planning or the key, except the lede's first one, which says what runs in the browser
      if (key !== "quickstartLedeHead") expect(text, key).toMatch(/key|live/i);
    }
    // the sponsors may be named; the home card names two of them
    expect(HOSTED.homeCard).toContain("NVIDIA");
    expect(HOSTED.homeCard).toContain("Token Factory");
  });

  it("the hosted variant never says there is no key, beside a sentence that says to bring one", () => {
    for (const key of KEYS) {
      expect(HOSTED[key], key).not.toMatch(/\bno key\b/i);
      expect(liveCopy(key, true), key).not.toMatch(/\bno key\b/i); // and the shipped text, not only this file's copy of it
    }
    // the approved first sentence does say it, and stays as it was
    expect(APPROVED.quickstartLedeHead).toContain("no key");
  });

  it("the hosted variant never says live planning is only local", () => {
    for (const key of KEYS) {
      expect(HOSTED[key], key).not.toMatch(/has no (server|backend)|runs no model|no model to call/i);
    }
    expect(HOSTED.lede).not.toContain("on your own machine");
    expect(HOSTED.homeCard).not.toContain("on your own machine");
  });
});

describe("the home lede and card", () => {
  it("the lede keeps its head and swaps only its last sentence", () => {
    expect(homeLede(false)).toBe(LEDE_HEAD + APPROVED.lede);
    expect(homeLede(true)).toBe(LEDE_HEAD + HOSTED.lede);
    expect(homeLede(true).startsWith(LEDE_HEAD)).toBe(true);
  });

  it("the home page shows the approved sentences without the runner", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = textOf(renderToStaticMarkup(createElement(Home)));
    expect(text).toContain(LEDE_HEAD + APPROVED.lede);
    expect(text).toContain("the planner turns it into objects and steps the simulator can run. " + APPROVED.homeCard);
    expect(text).toContain(APPROVED.heroAsk);
    for (const gone of [HOSTED.lede, HOSTED.homeCard, HOSTED.heroAsk]) expect(text).not.toContain(gone);
  });

  it("the home page shows the hosted variants with the runner, and no sentence that says planning is only local", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = textOf(renderToStaticMarkup(createElement(Home)));
    expect(text).toContain(LEDE_HEAD + HOSTED.lede);
    expect(text).toContain("the planner turns it into objects and steps the simulator can run. " + HOSTED.homeCard);
    expect(text).toContain(HOSTED.heroAsk);
    for (const gone of [APPROVED.lede, APPROVED.homeCard, APPROVED.heroAsk, "live planning runs on your own machine"]) expect(text).not.toContain(gone);
  });
});

describe("the question box", () => {
  const examples = [{ href: "/studio/x/", title: "X", setting: "y" }];
  // the page hands the box its sentence, so the box is rendered with each wording
  const box = () => textOf(renderToStaticMarkup(createElement(HeroAsk, { examples, note: liveCopy("heroAsk") })));

  it("says what it always said without the runner, with the link to run it yourself after it", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = box();
    expect(text).toContain(APPROVED.heroAsk + " ");
    expect(text).toMatch(/To plan new words, <a [^>]*>run it yourself<\/a>\./);
  });

  it("with the runner points to the Studio's key and free tier, and still offers running it yourself", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = box();
    expect(text).toContain(HOSTED.heroAsk + " ");
    expect(text).toMatch(/or <a [^>]*>run it yourself<\/a>\./);
    expect(text).not.toContain("runs no model");
  });
});

describe("the docs", () => {
  const quickstart = () => textOf(renderToStaticMarkup(createElement(QuickstartPage)));
  const planATask = () => textOf(renderToStaticMarkup(createElement(PlanATaskPage)));

  it("the quickstart reads as approved without the runner, label line included", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = quickstart();
    expect(text).toContain(`${APPROVED.quickstartLedeHead} ${QUICKSTART_MIDDLE} ${APPROVED.quickstartLede}`);
    // word for word what it said before the runner existed
    expect(text).toContain(
      "Everything here runs in your browser: no install, no account, no key. The site replays recorded plans and runs: open a worked example, edit a recorded plan or build one by hand, and find a job like yours. Live planning of new words runs on your own machine.",
    );
    expect(text).not.toContain(HOSTED.quickstartLedeHead);
    expect(text).toContain(APPROVED.quickstartNote + " Every result is labelled “Product mode: not part of the pre-registered experiment”.");
    expect(text).not.toContain(HOSTED.quickstartLede);
    expect(text).not.toContain(HOSTED.quickstartNote);
  });

  it("the quickstart swaps its first and last sentences and its note with the runner, and keeps the label line", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = quickstart();
    expect(text).toContain(`${HOSTED.quickstartLedeHead} ${QUICKSTART_MIDDLE} ${HOSTED.quickstartLede}`);
    // the lede no longer says "no key" next to "with your own key"
    const lede = text.slice(text.indexOf(HOSTED.quickstartLedeHead), text.indexOf(HOSTED.quickstartLede) + HOSTED.quickstartLede.length);
    expect(lede).not.toMatch(/\bno key\b/i);
    expect(lede).toContain("with your own key");
    expect(text).not.toContain(APPROVED.quickstartLedeHead);
    expect(text).toContain(HOSTED.quickstartNote + " Every result is labelled “Product mode: not part of the pre-registered experiment”.");
    expect(text).not.toContain(APPROVED.quickstartLede);
    expect(text).not.toContain("The site has no server");
    // the step that teaches running it on your machine is still there
    expect(text).toContain("Run it on your machine");
  });

  it("the plan-a-task note reads as approved without the runner, and keeps its link and its second option", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = planATask();
    expect(text).toMatch(new RegExp(`${escape(APPROVED.planTaskNote)}<a [^>]*>Run it yourself</a>\\), or press`));
    expect(text).not.toContain(HOSTED.planTaskNote);
  });

  it("the plan-a-task note swaps its sentence with the runner and keeps its link and its second option", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = planATask();
    expect(text).toMatch(new RegExp(`${escape(HOSTED.planTaskNote)}<a [^>]*>Run it yourself</a>\\), or press`));
    expect(text).not.toContain("The public site has no server");
  });
});

describe("the Studio", () => {
  const studio = () => textOf(renderToStaticMarkup(createElement(StudioPage)));
  // The planner's custom-text message is set on a click, so it is checked through its wiring here
  // (the planner reads it from liveCopy and holds no copy of its own, see the next block) and in the
  // browser by scripts/smoke_studio_intake.mjs, which presses Plan with words that are not an example.
  const intakeSource = readFileSync(path.resolve(__dirname, "../components/studio/intake/IntakeWorkbench.tsx"), "utf8");

  it("the Explore mode tip reads as approved without the runner", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = studio();
    expect(text).toContain(APPROVED.studioTip);
    expect(text).not.toContain(HOSTED.studioTip);
    expect(liveCopy("intakeMessage")).toBe(APPROVED.intakeMessage);
    expect(intakeSource).toContain('setMessage(liveCopy("intakeMessage"))');
  });

  it("the planner's Plan box says what it always said without the runner, and holds no copy of its own", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const text = studio();
    expect(text).toContain(APPROVED.intakeBox);
    expect(text).not.toContain(HOSTED.intakeBox);
    expect(intakeSource).toContain('liveCopy("intakeBox")');
  });

  it("the planner's Plan box, with the runner, no longer says the site has no backend and points to the panel below it", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = studio();
    expect(text).toContain(HOSTED.intakeBox);
    expect(text).not.toContain(APPROVED.intakeBox);
    expect(text).not.toContain("This static site has no backend");
    // "below" is true: in the planner the panel comes after the Plan box and before the step editor
    expect(intakeSource.indexOf("<HostedRunner")).toBeGreaterThan(intakeSource.indexOf('liveCopy("intakeBox")'));
    expect(intakeSource.indexOf("<HostedRunner")).toBeLessThan(intakeSource.indexOf('aria-labelledby="edit-title"'));
  });

  it("the Explore mode tip points to the key and free tier with the runner", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = studio();
    expect(text).toContain(HOSTED.studioTip);
    expect(text).not.toContain("This public site has no backend");
    expect(liveCopy("intakeMessage")).toBe(HOSTED.intakeMessage);
  });

  it("the planner itself still renders in both builds", () => {
    for (const url of ["", RUNNER]) {
      vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", url);
      const html = renderToStaticMarkup(createElement(IntakeWorkbench, { catalog, recorded, results: {} }));
      expect(html).toContain("Describe the use case");
    }
  });
});

describe("no sentence is written twice", () => {
  const root = path.resolve(__dirname, "..");
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const full = path.join(dir, name);
      if (statSync(full).isDirectory()) walk(full);
      else if (/\.(tsx?|mjs)$/.test(name)) files.push(full);
    }
  };
  for (const d of ["app", "components", "lib"]) walk(path.join(root, d));

  it("each approved or hosted sentence lives only in lib/live_copy.ts", () => {
    const own = path.join(root, "lib", "live_copy.ts");
    const stray: string[] = [];
    for (const file of files) {
      if (file === own) continue;
      const src = readFileSync(file, "utf8");
      for (const key of KEYS) {
        for (const text of [APPROVED[key], HOSTED[key]]) {
          const lead = text.length > 60 ? text.slice(0, 60) : text;
          if (src.includes(lead)) stray.push(`${path.relative(root, file)}: ${key}`);
        }
      }
    }
    expect(stray).toEqual([]);
  });

  it("no page says live planning runs only on your own machine except through liveCopy", () => {
    const own = path.join(root, "lib", "live_copy.ts");
    const phrases = [/live planning runs on your own machine/i, /live planning of new words runs on your own machine/i, /has no model to call/i, /this site runs no model/i];
    const stray: string[] = [];
    for (const file of files) {
      if (file === own) continue;
      const src = readFileSync(file, "utf8");
      for (const re of phrases) if (re.test(src)) stray.push(`${path.relative(root, file)}: ${re}`);
    }
    expect(stray).toEqual([]);
  });
});

function escape(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
