/**
 * Where the hosted runner meets the pages: the panel is absent unless the build has the runner
 * (and renders nothing until /health answers), a live plan loads into the editor as a replayed one
 * does, the docs and the home lede gate on the same flag.
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { HostedDocSection, withHosted } from "@/components/docs/HostedDocs";
import { homeLede, LEDE_HEAD, LEDE_TAIL, LEDE_TAIL_HOSTED } from "@/components/home/lede";
import catalogJson from "@/components/studio/intake/catalog.json";
import IntakeWorkbench, { type RecordedPlans } from "@/components/studio/intake/IntakeWorkbench";
import recordedJson from "@/components/studio/intake/recorded.json";
import { type Catalog, type PlanArgs, planArgsOf, taskId, validatePlan } from "@/components/studio/intake/taskspec";
import { exportSource, HOSTED_MAX_OBJECTS, HOSTED_MAX_STEPS, hostedLimit, planFromReply, runDocument } from "@/components/studio/runner/handoff";
import HostedRunner from "@/components/studio/runner/HostedRunner";
import { liveCopy } from "@/lib/live_copy";
import type { PlanReply } from "@/lib/runner";

import { REFUSAL_PLAN, VALID_PLAN } from "./helpers/runner_mock";

const catalog = catalogJson as unknown as Catalog;
const recorded = recordedJson as unknown as RecordedPlans;
const RUNNER = "https://runner.example";

afterEach(() => vi.unstubAllEnvs());

const workbench = () => renderToStaticMarkup(createElement(IntakeWorkbench, { catalog, recorded, results: {} }));

describe("the Studio planner", () => {
  it("has no hosted panel in a build without the runner", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const html = workbench();
    expect(html).not.toContain("Plan and run it live");
    expect(html).not.toContain("data-hosted-runner");
    expect(html).not.toContain("Run in physics");
    // the planner it always had
    expect(html).toContain("Describe the use case");
    expect(html).toContain("Check and edit the steps");
  });

  it("still shows no panel until /health answers: with the runner the planner differs only in the Plan box sentence that points to the panel and the scroll margin of the step editor", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    const without = workbench();
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const withRunner = workbench();
    expect(withRunner).not.toBe(without);
    expect(withRunner.replace(liveCopy("intakeBox", true), liveCopy("intakeBox", false)).replace('class="mt-14 scroll-mt-20 outline-none"', 'class="mt-14 outline-none"')).toBe(without);
    // the public build keeps the editor's markup as it was
    expect(without).toContain('class="mt-14 outline-none"');
    expect(withRunner).toContain('class="mt-14 scroll-mt-20 outline-none"');
  });

  it("the panel itself renders nothing before the health check, and when the URL is not usable the page never mounts it", () => {
    const html = renderToStaticMarkup(
      createElement(HostedRunner, { base: RUNNER, description: "", runDoc: null, runLimit: null, hasPlan: false, hasStl: false, onPlanned: () => {} }),
    );
    expect(html).toBe("");
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "http://runner.example");
    expect(workbench()).not.toContain("data-hosted-runner");
  });
});

describe("a live plan loads into the editor as a replayed one does", () => {
  const examples = recorded.examples.filter((e) => e.plan);

  it("gives the same editor plan as the replay path", () => {
    for (const e of examples) {
      const replayed = e.plan ? planArgsOf(e.plan) : null;
      expect(planFromReply({ plan: e.plan }), e.case_id).toEqual(replayed);
    }
  });

  it("a valid hosted plan passes the validator and gets a task id", async () => {
    const args = planFromReply({ plan: VALID_PLAN })!;
    const verdict = validatePlan(catalog, args);
    expect(verdict.status).toBe("valid");
    expect(await taskId(catalog, verdict)).toMatch(/^[0-9a-f]{64}$/);
  });

  it("a refusal loads as a refusal, with the reasons the editor shows", () => {
    const args = planFromReply({ plan: REFUSAL_PLAN })!;
    expect(args.refusal).toBeTruthy();
    expect(validatePlan(catalog, args).status).toBe("refused");
  });

  it("a reply with no plan loads nothing", () => {
    expect(planFromReply({ plan: null })).toBeNull();
  });

  it("copies the plan, so editing the editor's copy never changes the reply", () => {
    const reply: Pick<PlanReply, "plan"> = { plan: structuredClone(VALID_PLAN) };
    const args = planFromReply(reply)!;
    args.steps.pop();
    expect(reply.plan).toEqual(VALID_PLAN);
  });

  it("the run starts from the plan as edited, as the document Download writes", async () => {
    const args: PlanArgs = planFromReply({ plan: VALID_PLAN })!;
    const edited = { ...args, steps: args.steps.slice(0, 3), title: "Edited" };
    const verdict = validatePlan(catalog, edited);
    const doc = runDocument(catalog, edited, verdict.status === "valid" ? await taskId(catalog, verdict) : null, "manual");
    expect(doc).toMatchObject({ contract: catalog.contract, label: catalog.label, source: "manual", title: "Edited" });
    expect((doc.steps as unknown[]).length).toBe(3);
    expect(Object.keys(doc.fidelity as object)).toHaveLength(3);
  });

  it("names where the plan came from", () => {
    expect(exportSource({ edited: true, hosted: true, shownSource: "live" })).toBe("manual");
    expect(exportSource({ edited: false, hosted: true, shownSource: null })).toBe("nemotron");
    expect(exportSource({ edited: false, hosted: false, shownSource: "live" })).toBe("nemotron");
    expect(exportSource({ edited: false, hosted: false, shownSource: "recorded" })).toBe("recorded");
    expect(exportSource({ edited: false, hosted: false, shownSource: null })).toBe("recorded");
  });
});

describe("what the hosted runner takes", () => {
  const doc = (objects: number, steps: number) => ({ objects: Array.from({ length: objects }, (_, i) => ({ name: `o${i}` })), steps: Array.from({ length: steps }, (_, i) => ({ id: `s${i}` })) });

  it("runs every recorded plan that has no uploaded part", () => {
    for (const e of recorded.examples.filter((x) => x.plan && x.uploads.length === 0)) expect(hostedLimit(e.plan!, []), e.case_id).toBeNull();
  });

  it("turns down a plan with an uploaded part, in plain words", () => {
    const upload = recorded.examples.find((x) => x.uploads.length > 0)!;
    expect(hostedLimit(upload.plan!, upload.uploads.map((u) => u.upload_id))).toMatch(/no part upload/);
  });

  it("allows up to 6 objects and 8 steps, and says why it stops at more", () => {
    expect(hostedLimit(doc(HOSTED_MAX_OBJECTS, HOSTED_MAX_STEPS), [])).toBeNull();
    expect(hostedLimit(doc(HOSTED_MAX_OBJECTS + 1, 2), [])).toBe("The hosted runner takes plans of up to 6 objects; this one has 7. Run it on your own machine.");
    expect(hostedLimit(doc(2, HOSTED_MAX_STEPS + 1), [])).toBe("The hosted runner takes plans of up to 8 steps; this one has 9. Run it on your own machine.");
  });

  it("turns down a plan over 48 KiB", () => {
    expect(hostedLimit({ ...doc(2, 2), note: "x".repeat(49 * 1024) }, [])).toMatch(/too large/);
  });
});

describe("the docs show the hosted runner only when the build has it", () => {
  const SECTIONS = [{ id: "a", title: "A" }];
  const render = () => renderToStaticMarkup(createElement(HostedDocSection, { n: 6 }));

  it("hide the section and its entry in a build without the runner", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", "");
    expect(render()).toBe("");
    expect(withHosted(SECTIONS)).toEqual(SECTIONS);
  });

  it("show the providers, the caps and the key privacy line in a build with it", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const html = render();
    expect(html).toContain("Plan and run it on this site");
    expect(html).toContain("Your NVIDIA API key");
    expect(html).toContain("Your Nebius Token Factory key");
    expect(html).toContain("The free allowance");
    expect(html).toContain("capped at $2 a day");
    expect(html).toContain("Your key is sent once over HTTPS for this request and is never stored or logged.");
    expect(html).toContain("Replays are shown for the recorded examples; hosted runs show the summary.");
    expect(withHosted(SECTIONS).map((s) => s.id)).toEqual(["a", "hosted"]);
  });

  it("say how long planning and a first submission can take, and what a busy runner says", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const text = render().replace(/<!-- -->/g, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
    expect(text).toContain("Planning usually takes 15 to 45 seconds and can take up to five minutes.");
    expect(text).toContain("which can take up to three minutes to accept a plan it has not seen before.");
    expect(text).toContain("If the runner is busy, the page says to try again in a minute.");
  });

  it("the short form points to the tutorial and keeps the privacy line", () => {
    vi.stubEnv("NEXT_PUBLIC_RUNNER_URL", RUNNER);
    const html = renderToStaticMarkup(createElement(HostedDocSection, { n: 5, brief: true }));
    expect(html).toContain("Plan live");
    // next/link adds the trailing slash and the base path at build time; outside Next it leaves the path as given
    expect(html).toMatch(/href="\/docs\/tutorials\/plan-a-task\/?#hosted"/);
    expect(html).toContain("never stored or logged");
    expect(html).not.toContain("capped at $2 a day");
  });
});

describe("the home lede", () => {
  it("says what it always said without the runner", () => {
    expect(homeLede(false)).toBe(
      "Describe the job your robot has to do. Effector Lab plans it as steps, tests every end effector it can simulate on every step, and recommends one tool, a tool change or a two-tool head, next to the best single off-the-shelf tool. This site replays recorded plans and runs; live planning runs on your own machine.",
    );
  });

  it("changes only its last sentence with the runner", () => {
    expect(homeLede(true)).toBe(LEDE_HEAD + LEDE_TAIL_HOSTED);
    expect(homeLede(true)).toContain("This site replays recorded plans and runs, and can plan and test a new job live.");
    expect(homeLede(true).startsWith(LEDE_HEAD)).toBe(true);
    expect(homeLede(true)).not.toContain(LEDE_TAIL);
  });
});
