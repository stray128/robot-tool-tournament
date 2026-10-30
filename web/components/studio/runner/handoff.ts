/**
 * The seam between the hosted runner and the planner's editor. A plan planned live loads into
 * the step editor the way a replayed one does (`planArgsOf` on the plan document), and a run is
 * started from whatever the editor holds then, edits included, as the same TaskSpec document the
 * Download button writes.
 */

import type { PlanReply } from "@/lib/runner";

import { type Catalog, exportDocument, type PlanArgs, planArgsOf } from "../intake/taskspec";

/** The editor's plan for a hosted reply, or null when the reply carries no plan document. */
export function planFromReply(reply: Pick<PlanReply, "plan">): PlanArgs | null {
  return reply.plan ? planArgsOf(reply.plan) : null;
}

/** Where the exported plan came from: an edit, a live hosted plan or a recorded replay. */
export function exportSource(from: { edited: boolean; hosted: boolean; shownSource: "recorded" | "live" | null }): "manual" | "nemotron" | "recorded" {
  if (from.edited) return "manual";
  return from.hosted || from.shownSource === "live" ? "nemotron" : "recorded";
}

/** The TaskSpec document POST /runs takes: the plan as the editor holds it now. */
export function runDocument(catalog: Catalog, plan: PlanArgs, id: string | null, source: string): Record<string, unknown> {
  return exportDocument(catalog, plan, id, source);
}

/** What the hosted runner takes (docs/RUNNER.md): the server checks these again, this only says why a button is off. */
export const HOSTED_MAX_OBJECTS = 6;
export const HOSTED_MAX_STEPS = 8;
export const HOSTED_MAX_PLAN_BYTES = 48 * 1024;

/** Why the hosted runner would turn this plan down before running it, in plain words; null when it can run it. */
export function hostedLimit(doc: Record<string, unknown>, uploads: readonly string[]): string | null {
  if (uploads.length > 0) return "The hosted runner has no part upload, so it cannot run a plan that uses one. Run it on your own machine.";
  const objects = Array.isArray(doc.objects) ? doc.objects.length : 0;
  const steps = Array.isArray(doc.steps) ? doc.steps.length : 0;
  if (objects > HOSTED_MAX_OBJECTS) return `The hosted runner takes plans of up to ${HOSTED_MAX_OBJECTS} objects; this one has ${objects}. Run it on your own machine.`;
  if (steps > HOSTED_MAX_STEPS) return `The hosted runner takes plans of up to ${HOSTED_MAX_STEPS} steps; this one has ${steps}. Run it on your own machine.`;
  if (new TextEncoder().encode(JSON.stringify(doc)).length > HOSTED_MAX_PLAN_BYTES) return "This plan is too large for the hosted runner. Run it on your own machine.";
  return null;
}
