/**
 * The trial's per-visitor limits as the page states them (lib/runner_limits.ts) against the runner's own
 * defaults (`Caps` in src/rtt/runner/config.py, read here and never copied by hand). The page must not promise
 * a visitor more than the runner gives, and must not leave out a limit the runner has: a default that changes
 * there, a per-visitor cap added there, or a limit added here without its counterpart fails this file.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { liveCopy } from "@/lib/live_copy";
import { RUNNER_TRIAL_LIMITS } from "@/lib/runner_limits";

const CONFIG = path.resolve(__dirname, "..", "..", "src", "rtt", "runner", "config.py");

/** The runner's cap that each limit on this side states, by its name in `Caps`. */
const CAP_OF = {
  freePlansPerVisitor: "free_plans_per_ip_per_day",
  keyedPlansPerVisitor: "keyed_plans_per_ip_per_day",
  runsPerVisitor: "runs_per_ip_per_day",
} as const satisfies Record<keyof typeof RUNNER_TRIAL_LIMITS, string>;

/** The integer defaults of the `Caps` dataclass, by field name. */
function capsDefaults(): Record<string, number> {
  const src = readFileSync(CONFIG, "utf8");
  const start = src.indexOf("class Caps:");
  if (start < 0) throw new Error("class Caps is not in src/rtt/runner/config.py any more; point this test at the runner's caps");
  const next = src.slice(start + 1).search(/\n(?:@dataclass|class )/);
  const block = src.slice(start, next < 0 ? undefined : start + 1 + next);
  const out: Record<string, number> = {};
  for (const m of block.matchAll(/^ {4}(\w+): int = (\d+)\s*$/gm)) out[m[1]!] = Number(m[2]);
  return out;
}

describe("the trial's per-visitor limits", () => {
  const defaults = capsDefaults();

  it("reads the runner's caps (a parse that found nothing would pass every check below for the wrong reason)", () => {
    expect(Object.keys(defaults).length).toBeGreaterThanOrEqual(6);
    expect(defaults).toMatchObject({ job_wall_limit_s: expect.any(Number), keyed_plans_per_day: expect.any(Number) });
  });

  it("each limit the page states is the runner's own default", () => {
    for (const [name, cap] of Object.entries(CAP_OF)) {
      expect(defaults[cap], `${cap} is not an integer default of Caps in config.py`).toBeTypeOf("number");
      expect(RUNNER_TRIAL_LIMITS[name as keyof typeof RUNNER_TRIAL_LIMITS], `${name} states ${cap}`).toBe(defaults[cap]);
    }
  });

  it("every limit is paired with a runner cap, and the page states every per-visitor cap the runner has", () => {
    expect(Object.keys(CAP_OF).sort()).toEqual(Object.keys(RUNNER_TRIAL_LIMITS).sort());
    const perVisitor = Object.keys(defaults).filter((name) => /_per_ip_/.test(name));
    expect(perVisitor.sort()).toEqual(Object.values(CAP_OF).sort());
  });

  it("the sentences in the panel and in the docs say the runner's numbers", () => {
    const free = defaults.free_plans_per_ip_per_day;
    const keyed = defaults.keyed_plans_per_ip_per_day;
    const runs = defaults.runs_per_ip_per_day;
    for (const key of ["trialLimitsPanel", "trialLimitsDocs"] as const) {
      const text = liveCopy(key, true);
      expect(text, key).toMatch(new RegExp(`\\b${free} free plans and ${runs} physics runs a day \\(up to ${keyed} plans with your own key\\)`));
    }
  });

  it("the public build says nothing about them: there is no panel and no hosted docs section to say it in", () => {
    for (const key of ["trialLimitsPanel", "trialLimitsDocs"] as const) expect(liveCopy(key, false), key).toBe("");
  });
});
