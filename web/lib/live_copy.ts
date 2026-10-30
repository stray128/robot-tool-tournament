/**
 * The sentences that say where live planning happens. A build without the hosted runner
 * (NEXT_PUBLIC_RUNNER_URL unset) keeps the approved `local` text word for word: live planning
 * runs on your own machine. A build with it swaps in the `hosted` text, which also points to
 * the visitor's own key or the free allowance. Every place that makes this claim reads it from
 * here through `liveCopy`, so the flag is checked in one place and the two wordings sit side by side.
 */

import { runnerConfigured } from "./runner";

export interface LiveCopyPair {
  /** The approved text of the public build. Do not edit it. */
  readonly local: string;
  /** The text of a build that has the hosted runner. */
  readonly hosted: string;
}

export const LIVE_COPY = {
  /** Home page, the "Plan" step card, after its first sentence. */
  homeCard: {
    local: "Here you replay recorded plans and edit their steps; live planning runs on your own machine.",
    hosted: "Here you replay recorded plans and edit their steps, or plan a new job live with your own NVIDIA or Token Factory key, or a small free tier.",
  },
  /** Home page lede, last sentence. */
  lede: {
    local: "This site replays recorded plans and runs; live planning runs on your own machine.",
    hosted: "This site replays recorded plans and runs, and can plan and test a new job live.",
  },
  /** Home page "What should your robot do?" form, the note beside the button (the link "run it yourself" follows it). */
  heroAsk: {
    local: "This site runs no model: it replays recorded plans, and you can build yours by hand. To plan new words,",
    hosted: "This site replays recorded plans, and you can build yours by hand. To plan new words, open the Studio and use your own key or a small free tier, or",
  },
  /**
   * Quickstart lede, first sentence. A runner build must not say "no key" beside "with your own key", so it says
   * what runs in the browser instead (browsing and editing need no key; live planning is in the last sentence).
   */
  quickstartLedeHead: {
    local: "Everything here runs in your browser: no install, no account, no key.",
    hosted: "Browsing and editing run in your browser: no install and no account.",
  },
  /** Quickstart lede, last sentence. */
  quickstartLede: {
    local: "Live planning of new words runs on your own machine.",
    hosted: "Live planning of new words runs here with your own key or a small free tier, or on your own machine.",
  },
  /** Quickstart, the note "On this public site", before the line about the product-mode label. */
  quickstartNote: {
    local: "The site has no server, so Plan replays the planner's recorded run for the examples. To plan new words live, run the Studio on your own machine (step 4).",
    hosted: "Plan replays the planner's recorded run for the examples. To plan new words live, use your own key or a small free tier in the \u201cPlan and run it live\u201d panel, or run the Studio on your own machine (step 4).",
  },
  /** Studio header, the tooltip of the "Explore mode" tag. */
  studioTip: {
    local: "This public site has no backend. It replays the recorded plans and the recorded test runs, and lets you edit and download a plan. To plan and test your own text, run the Studio locally with your own model key.",
    hosted: "This public site replays the recorded plans and the recorded test runs, and lets you edit and download a plan. To plan and test your own text, use your own key or a small free tier in the \u201cPlan and run it live\u201d panel, or run the Studio locally with your own model key.",
  },
  /** Plan-a-task tutorial, the note "Explore mode on this site", up to the link "Run it yourself". */
  planTaskNote: {
    local: "The public site has no server. Plan replays the recorded run of each example, model calls included, so you see what the planner did. To plan your own words live, run the Studio locally with your own Token Factory key (",
    hosted: "Plan replays the recorded run of each example, model calls included, so you see what the planner did. To plan your own words live, use your own key or a small free tier in the \u201cPlan and run it live\u201d panel, or run the Studio locally with your own Token Factory key (",
  },
  /**
   * Studio planner, the line in the "02 Plan it" box that says what the Plan button does on this site. In a runner
   * build the panel "Plan and run it live" sits below the box, so its variant points down to it.
   */
  intakeBox: {
    local: "This static site has no backend: Plan replays the recorded live plan of an example, model calls included.",
    hosted: "Plan replays the recorded plan of an example, model calls included. To plan your own words live, use the panel below.",
  },
  /** Studio planner, the message shown when Plan gets words that are not a recorded example. */
  intakeMessage: {
    local: "This public site runs in Explore mode: it has no model to call, so it can only replay the recorded plans of the examples. Pick an example, or build your plan by hand below. To plan your own words, run the Studio locally with its API.",
    hosted: "This public site runs in Explore mode, so Plan can only replay the recorded plans of the examples. Pick an example, or build your plan by hand below. To plan your own words, use your own key or a small free tier in the \u201cPlan and run it live\u201d panel, or run the Studio locally with its API.",
  },
} as const satisfies Record<string, LiveCopyPair>;

export type LiveCopyKey = keyof typeof LIVE_COPY;

/**
 * The wording for this build. `hosted` defaults to the build flag; a test passes it to check
 * both wordings in one run.
 */
export function liveCopy(key: LiveCopyKey, hosted: boolean = runnerConfigured()): string {
  return hosted ? LIVE_COPY[key].hosted : LIVE_COPY[key].local;
}
