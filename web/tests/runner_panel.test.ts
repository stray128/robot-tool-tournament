/**
 * The hosted runner's panel as markup, in states the controller can be in: the key field (a password field,
 * so a screen reader is given bullets and not the key), the progress wording while planning and submitting
 * (the timeouts are 330 s and 210 s, so the page says how long it can take), the off notice and the auto-off
 * line before and after the date, and the busy message. The controller is real except that its snapshot is
 * the state a test sets.
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { RunnerState } from "@/components/studio/runner/controller";
import HostedRunner from "@/components/studio/runner/HostedRunner";
import { KEY_FIELD_ATTRIBUTES } from "@/components/studio/runner/keyfield";
import { RUNNER_TRIAL_LIMITS } from "@/lib/runner_limits";
import { capSentence, MESSAGES, parseHealth } from "@/lib/runner";

import { HEALTH } from "./helpers/runner_mock";

const shared = vi.hoisted(() => ({ state: {} as Partial<RunnerState> }));

vi.mock("@/components/studio/runner/controller", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/studio/runner/controller")>();
  class SetStateController extends actual.RunnerController {
    constructor(opts: ConstructorParameters<typeof actual.RunnerController>[0]) {
      super(opts);
      const base = this.getSnapshot;
      let from: Partial<RunnerState> | null = null;
      let merged: RunnerState | null = null;
      this.getSnapshot = (): RunnerState => {
        if (merged === null || from !== shared.state) {
          from = shared.state;
          merged = { ...base(), ...shared.state };
        }
        return merged;
      };
    }
  }
  return { ...actual, RunnerController: SetStateController };
});

const NOW = Date.parse("2026-10-28T10:00:00Z");

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
});
afterEach(() => {
  vi.useRealTimers();
  shared.state = {};
});

const ready = (over: Partial<RunnerState> = {}): Partial<RunnerState> => ({ gate: "ready", offReason: null, health: parseHealth(HEALTH), ...over });

function panel(state: Partial<RunnerState>): { html: string; text: string } {
  shared.state = state;
  const html = renderToStaticMarkup(
    createElement(HostedRunner, { base: "https://runner.example/api", description: "Open the lid.", runDoc: null, runLimit: null, hasPlan: false, hasStl: false, onPlanned: () => {} }),
  );
  const text = html
    .replace(/<!-- -->/g, "")
    .replace(/<[^>]+>/g, " ")
    .replaceAll("&#x27;", "'")
    .replaceAll("&amp;", "&")
    .replace(/\s+/g, " ")
    .trim();
  return { html, text };
}

/** The text of the fault notice alone (role="alert"), for what a refusal says: the panel's own note about the daily limits is not part of it. */
function alertText(html: string): string {
  const notice = /<div role="alert"[^>]*>([\s\S]*?)<\/div>/.exec(html);
  if (!notice) throw new Error("no fault notice in the markup");
  return notice[1]!.replace(/<!-- -->/g, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

/** The attributes of the key field, lower-cased, from the markup. */
function keyFieldTag(html: string): string {
  const tag = /<input\b[^>]*data-hosted-key[^>]*>/i.exec(html)?.[0];
  if (!tag) throw new Error("no key field in the markup");
  return tag.toLowerCase();
}

describe("the key field", () => {
  const SECRET = "nvapi-SECRET-KEY-0123456789";

  it("is a password field, so a screen reader is given bullets and the key is not copied out", () => {
    const tag = keyFieldTag(panel(ready()).html);
    expect(tag).toContain('type="password"');
    expect(tag).not.toContain('type="text"');
    // not drawn as dots by a style: a text field masked that way exposes the typed key to assistive technology
    expect(tag).not.toMatch(/text-security|style=/);
  });

  it("asks not to be filled, corrected, checked or read: autocomplete off (not new-password), autocorrect, capitals and spell check off", () => {
    const tag = keyFieldTag(panel(ready()).html);
    expect(tag).toContain('autocomplete="off"');
    // new-password makes browsers offer to generate a password
    expect(tag).not.toContain("new-password");
    expect(tag).toContain('autocorrect="off"');
    expect(tag).toContain('autocapitalize="off"');
    // a spell checker can send what it checks to a server
    expect(tag).toContain('spellcheck="false"');
  });

  it("carries the opt-out markers of the password managers and of the extensions that read fields", () => {
    const tag = keyFieldTag(panel(ready()).html);
    expect(tag).toMatch(/data-1p-ignore(=|\s|\/|>)/); // 1Password
    expect(tag).toContain('data-lpignore="true"'); // LastPass
    expect(tag).toMatch(/data-bwignore(=|\s|\/|>)/); // Bitwarden
    expect(tag).toContain('data-form-type="other"'); // Dashlane
    expect(tag).toContain('data-gramm="false"'); // Grammarly
    expect(tag).toContain('data-gramm_editor="false"');
    expect(tag).toContain('data-enable-grammarly="false"');
    expect(tag).toContain('data-lt-active="false"'); // LanguageTool
  });

  it("is never given a name or an id a form or a browser could remember it by", () => {
    expect(keyFieldTag(panel(ready()).html)).not.toMatch(/\s(name|id)=/);
  });

  it("takes every attribute from one place, with no choice of field and no text-security style", () => {
    expect(KEY_FIELD_ATTRIBUTES).toMatchObject({ type: "password", autoComplete: "off", autoCorrect: "off", autoCapitalize: "off", spellCheck: false, "data-lpignore": "true" });
    expect(KEY_FIELD_ATTRIBUTES).not.toHaveProperty("style");
    expect(JSON.stringify(KEY_FIELD_ATTRIBUTES)).not.toMatch(/text-security|new-password/i);
  });

  it("is a password field whatever the browser (it no longer looks at CSS.supports)", () => {
    for (const css of [undefined, { supports: () => true }, { supports: () => false }, { supports: () => { throw new Error("no"); } }]) {
      vi.stubGlobal("CSS", css);
      expect(keyFieldTag(panel(ready()).html), String(css)).toContain('type="password"');
    }
    vi.unstubAllGlobals();
  });

  it("holds the key only in its value: it is not echoed into any other attribute or into the page text", () => {
    const { html, text } = panel(ready({ key: SECRET }));
    expect(html.split(SECRET)).toHaveLength(2); // exactly once, in the field's value
    expect(keyFieldTag(html)).toContain(`value="${SECRET.toLowerCase()}"`); // the helper lower-cases the tag
    expect(text).not.toContain(SECRET);
  });

  it("is absent for the free allowance, which needs no key", () => {
    expect(panel(ready({ provider: "free" })).html).not.toContain("data-hosted-key");
  });
});

describe("the progress wording says how long it can take", () => {
  it("planning: usually 15 to 45 seconds, up to five minutes, and the seconds so far", () => {
    const { text } = panel(ready({ provider: "free", planPhase: "planning", planStartedAt: NOW - 7_000 }));
    expect(text).toContain("Nemotron is planning, usually 15–45 s. Planning can take up to five minutes. 7 s so far.");
  });

  it("planning past a minute counts in minutes", () => {
    const { text } = panel(ready({ provider: "free", planPhase: "planning", planStartedAt: NOW - 185_000 }));
    expect(text).toContain("Planning can take up to five minutes. 3 min 05 s so far.");
  });

  it("submitting: a plan the runner has not seen can take up to three minutes to be accepted", () => {
    const { text } = panel(ready({ runPhase: "submitting", runStartedAt: NOW - 42_000 }));
    expect(text).toContain("Sending the plan to the runner. A plan it has not seen before can take up to three minutes to be accepted. 42 s so far.");
  });

  it("promises nothing longer than the page waits (330 s to plan, 210 s to submit)", () => {
    const planning = panel(ready({ provider: "free", planPhase: "planning", planStartedAt: NOW })).text;
    const submitting = panel(ready({ runPhase: "submitting", runStartedAt: NOW })).text;
    expect(planning).not.toMatch(/\b(six|seven|ten) minutes\b/);
    expect(submitting).not.toMatch(/\b(four|five|six) minutes\b/);
  });
});

describe("the off notice", () => {
  const OFF = "The hosted runner is off. Run it on your own machine:";

  it("shows for each reason, with no controls", () => {
    for (const offReason of ["kill_switch", "closed", "auto_off"] as const) {
      const { html, text } = panel({ gate: "off", offReason, health: parseHealth({ ...HEALTH, open: offReason === "closed" ? false : true }) });
      expect(text, offReason).toContain(OFF);
      expect(html, offReason).toContain("data-hosted-off");
      expect(html, offReason).not.toContain("data-hosted-key");
      expect(html, offReason).not.toContain("data-plan-live");
      expect(html, offReason).not.toContain("data-run-physics");
      // next/link adds the trailing slash at build time; outside Next the path stays as given
      expect(html, offReason).toMatch(/href="\/docs\/tutorials\/run-it-yourself\/?"/);
    }
  });

  it("says the trial switched itself off once the date has passed, and that it will while the date is ahead", () => {
    const past = panel({ gate: "off", offReason: "auto_off", health: parseHealth({ ...HEALTH, open: false, auto_off: "2026-10-27" }) }).text;
    expect(past).toContain("This hosted service was a limited trial and switched itself off on 27 October 2026.");
    expect(past).not.toContain("switches itself off");
    const ahead = panel(ready()).text;
    expect(ahead).toContain("This hosted service is a limited trial and switches itself off on 15 December 2026.");
    expect(ahead).not.toContain("switched itself off");
    // closed early, with the date still ahead: the line is still about the date that is ahead
    const early = panel({ gate: "off", offReason: "closed", health: parseHealth({ ...HEALTH, open: false, auto_off: "2026-12-15" }) }).text;
    expect(early).toContain("switches itself off on 15 December 2026.");
  });

  it("the date itself is not yet past (the runner serves through it)", () => {
    const { text } = panel(ready({ health: parseHealth({ ...HEALTH, auto_off: "2026-10-28" }) }));
    expect(text).toContain("switches itself off on 28 October 2026.");
  });
});

describe("the trial's limits are stated up front", () => {
  const NOTE =
    "Free trial, with daily limits. Each visitor gets 3 free plans and 2 physics runs a day (up to 30 plans with your own key), from a small shared budget. If you reach a limit, the page says which one and when it resets. Everything here also runs on your own computer, without limits.";

  it("is always on the ready panel, whichever way the visitor chooses to pay, and before anything has been refused", () => {
    for (const provider of ["nvidia", "tokenfactory", "free"] as const) {
      const { html, text } = panel(ready({ provider }));
      expect(text, provider).toContain(NOTE);
      expect(html, provider).toContain("data-trial-limits");
    }
    // still there while planning, after a refusal, and once a run is under way
    for (const state of [
      ready({ provider: "free", planPhase: "planning", planStartedAt: NOW }),
      ready({ planPhase: "error", planFault: { kind: "cap", message: capSentence("free_plans_per_ip_per_day"), retry: null } }),
      ready({ runPhase: "running", runStartedAt: NOW }),
    ]) {
      expect(panel(state).text).toContain(NOTE);
    }
  });

  it("takes its numbers from the one place that holds them", () => {
    const { freePlansPerVisitor: free, keyedPlansPerVisitor: keyed, runsPerVisitor: runs } = RUNNER_TRIAL_LIMITS;
    expect(panel(ready()).text).toContain(`Each visitor gets ${free} free plans and ${runs} physics runs a day (up to ${keyed} plans with your own key)`);
  });

  it("comes after the introduction and before the choice of who pays, so it is read first", () => {
    const { text } = panel(ready());
    const intro = text.indexOf("Product mode, simulated, verify on hardware.");
    expect(intro).toBeGreaterThan(-1);
    expect(text.indexOf(NOTE)).toBeGreaterThan(intro);
    expect(text.indexOf(NOTE)).toBeLessThan(text.indexOf("Who pays for the planning call"));
    expect(text.indexOf(NOTE)).toBeLessThan(text.indexOf("Use the free allowance"));
  });

  it("leaves the whole-site counts and the switch-off line as they were", () => {
    const { text } = panel(ready());
    expect(text).toContain("Use the free allowance (3 free plans left today across the site)");
    expect(text).toContain("Runs on Nebius Serverless Jobs, capped at $2 a day. 9 runs left today across the site.");
    expect(text).toContain("This hosted service is a limited trial and switches itself off on 15 December 2026.");
    expect(text).not.toContain("Everything on this page also runs on your own machine.");
  });

  it("is not shown when the runner is off, where the notice already sends the visitor to their own machine", () => {
    const { html } = panel({ gate: "off", offReason: "kill_switch", health: parseHealth({ ...HEALTH, open: true, kill_switch: true }) });
    expect(html).not.toContain("data-trial-limits");
    expect(html).not.toContain("Free trial, with daily limits");
  });

  it("follows the writing rules: no em dash or en dash, no raw status code", () => {
    expect(NOTE).not.toMatch(/—|–| -- /);
    expect(panel(ready()).text).not.toMatch(/\b(400|401|403|404|413|422|429|500|502|503)\b/);
  });
});

describe("the counts beside the buttons are for the whole site", () => {
  it("the free allowance and the run cap line say so", () => {
    const { text } = panel(ready());
    expect(text).toContain("Use the free allowance (3 free plans left today across the site)");
    expect(text).toContain("Runs on Nebius Serverless Jobs, capped at $2 a day. 9 runs left today across the site.");
  });

  it("one left reads in the singular, none left says so", () => {
    const one = panel(ready({ health: parseHealth({ ...HEALTH, plans_left_today: 1, runs_left_today: 1 }) })).text;
    expect(one).toContain("Use the free allowance (1 free plan left today across the site)");
    expect(one).toContain("capped at $2 a day. 1 run left today across the site.");
    const none = panel(ready({ health: parseHealth({ ...HEALTH, plans_left_today: 0, runs_left_today: 0 }) })).text;
    expect(none).toContain("Use the free allowance (0 free plans left today across the site)");
    expect(none).toContain("capped at $2 a day. 0 runs left today across the site.");
  });

  it("a runner that gives no count leaves the labels plain", () => {
    const { text } = panel(ready({ health: parseHealth({ ...HEALTH, plans_left_today: undefined, runs_left_today: undefined }) }));
    expect(text).toContain("Use the free allowance");
    expect(text).toContain("capped at $2 a day.");
    expect(text).not.toContain("across the site");
    expect(text).not.toMatch(/left today/);
  });

  it("the per-visitor limits keep their own words when one is reached", () => {
    expect(capSentence("free_plans_per_ip_per_day")).toBe("You have used your free plans for today.");
    expect(capSentence("keyed_plans_per_ip_per_day")).toBe("You have reached the daily limit on plans with your own key.");
    expect(capSentence("runs_per_ip_per_day")).toBe("You have reached the daily limit on physics runs.");
    expect(capSentence("free_plans_per_day")).toBe("The free allowance for the whole site is used up for today.");
  });
});

describe("a 503 that is not the runner being off", () => {
  const OFF = "The hosted runner is off";

  it("the free tier being down says so and points to the visitor's own key, with no tutorial link and no retry", () => {
    const { html, text } = panel(ready({ planPhase: "error", planFault: { kind: "free_unavailable", message: MESSAGES.free_unavailable, retry: null } }));
    expect(html).toContain('data-hosted-error="free_unavailable"');
    expect(text).toContain("The free tier is not available right now. Use your own key, or try later.");
    expect(text).not.toContain(OFF);
    expect(html).not.toContain("run-it-yourself");
    expect(html).not.toContain("data-hosted-retry");
    // the panel itself is still the working one
    expect(html).toContain("data-plan-live");
  });

  it("a simulator that could not start a job says nothing was charged, and offers Try again", () => {
    const { html, text } = panel(ready({ runFault: { kind: "sim_unavailable", message: MESSAGES.sim_unavailable, retry: "run" } }));
    expect(html).toContain('data-hosted-error="sim_unavailable"');
    expect(text).toContain("The simulator could not start a job right now. Nothing was charged; try again in a few minutes.");
    expect(text).not.toContain(OFF);
    expect(html).not.toContain("run-it-yourself");
    expect(html).toContain("data-hosted-retry");
  });

  it("any other 503 reads plainly as unavailable, with Try again and no claim that the runner is off", () => {
    const { html, text } = panel(ready({ planPhase: "error", planFault: { kind: "unavailable", message: MESSAGES.unavailable, retry: "plan" } }));
    expect(html).toContain('data-hosted-error="unavailable"');
    expect(text).toContain("The hosted runner cannot take this request right now. Try again in a few minutes.");
    expect(text).not.toContain(OFF);
    expect(html).toContain("data-hosted-retry");
  });

  it("a runner that says it is off still gets the tutorial link beside the message", () => {
    const { html, text } = panel(ready({ runFault: { kind: "off", message: MESSAGES.off, retry: null } }));
    expect(text).toContain(`${OFF}. Run it on your own machine: Run it yourself`);
    expect(html).toMatch(/href="\/docs\/tutorials\/run-it-yourself\/?"/);
    expect(html).not.toContain("data-hosted-retry");
  });
});

describe("a cap that does not reset", () => {
  const TOTAL = "The hosted runner's total compute budget is used up. It does not reset.";

  it("offers Run it yourself beside the refusal, as the off notice does, with no Try again to repeat it", () => {
    const { html, text } = panel(ready({ runFault: { kind: "cap", message: TOTAL, retry: null, neverResets: true } }));
    expect(html).toContain('data-hosted-error="cap"');
    expect(text).toContain(`${TOTAL} Run it on your own machine: Run it yourself`);
    expect(html).toMatch(/href="\/docs\/tutorials\/run-it-yourself\/?"/);
    expect(html).not.toContain("data-hosted-retry");
  });

  it("the same link on a refused plan", () => {
    const { html, text } = panel(ready({ planPhase: "error", planFault: { kind: "cap", message: TOTAL, retry: null, neverResets: true } }));
    expect(text).toContain(`${TOTAL} Run it on your own machine: Run it yourself`);
    expect(html).toMatch(/href="\/docs\/tutorials\/run-it-yourself\/?"/);
  });

  it("a cap that resets keeps its words and has no tutorial link", () => {
    for (const cap of ["free_plans_per_ip_per_day", "keyed_plans_per_ip_per_day", "runs_per_ip_per_day", "spend_per_day_usd"]) {
      const message = `${capSentence(cap)} It resets at 29 October 2026, 00:00 UTC.`;
      const { html, text } = panel(ready({ runFault: { kind: "cap", message, retry: null } }));
      expect(text, cap).toContain(message);
      expect(html, cap).not.toContain("run-it-yourself");
    }
  });

  it("the off notice reads as it did, with the same words before the same link", () => {
    const { text } = panel(ready({ runFault: { kind: "off", message: MESSAGES.off, retry: null } }));
    expect(text).toContain("The hosted runner is off. Run it on your own machine: Run it yourself");
    expect(text).not.toContain("Run it on your own machine: Run it on your own machine");
  });
});

describe("a busy runner", () => {
  it("shows the plain message with a Try again, and no daily limit", () => {
    const { html, text } = panel(ready({ planPhase: "error", planFault: { kind: "busy", message: MESSAGES.busy, retry: "plan" } }));
    expect(html).toContain('data-hosted-error="busy"');
    expect(text).toContain("The runner is busy. Try again in a minute.");
    // the notice itself says nothing of a limit (the panel's note above it states the trial's limits, which is not a claim that one was reached)
    expect(alertText(html)).toBe("The runner is busy. Try again in a minute. Try again Paste your key again; it is forgotten after each request.");
    expect(alertText(html)).not.toMatch(/daily limit/i);
    expect(html).toContain("data-hosted-retry");
  });
});
