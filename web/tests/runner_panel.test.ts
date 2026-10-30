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

describe("a busy runner", () => {
  it("shows the plain message with a Try again, and no daily limit", () => {
    const { html, text } = panel(ready({ planPhase: "error", planFault: { kind: "busy", message: MESSAGES.busy, retry: "plan" } }));
    expect(html).toContain('data-hosted-error="busy"');
    expect(text).toContain("The runner is busy. Try again in a minute.");
    expect(text).not.toMatch(/daily limit/i);
    expect(html).toContain("data-hosted-retry");
  });
});
