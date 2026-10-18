/**
 * Live mode of the intake page: the local API's Task Engineer routes (src/rtt/studio/engineer/
 * api.py, mounted next to the Grasp Engineer's in src/rtt/api/app.py). Live mode is on only when
 * NEXT_PUBLIC_API_URL is set at build time; the public static site plays recorded plans.
 *
 *   POST /parts?name=&units=mm   raw STL body -> {part: {part_id, ...}} (the existing intake)
 *   POST /studio/plan            {text, part_id?} + X-RTT-Passcode -> {source, result, spend}
 *   POST /studio/validate        {plan, part_ids?} -> {status, problems, warnings, task_id}
 *
 * The passcode goes in a request header, never in a URL.
 */

import { ApiError, errorMessage, PASSCODE_HEADER } from "@/lib/api";

export interface PlanAttemptView {
  stage: string;
  outcome: string;
  problems: string[];
  cost_usd: number;
  latency_s: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface UploadView {
  upload_id: string;
  ref: string;
  name: string;
  size_mm: number[];
  mass_g: number;
}

/** `PlanResult.to_dict()` as the page uses it (recorded plans have the same shape). */
export interface PlanResultView {
  status: "valid" | "refused" | "invalid" | "failed";
  first_try: boolean;
  repaired: boolean;
  problems: string[];
  warnings: string[];
  notes: string[];
  uploads: UploadView[];
  plan: Record<string, unknown> | null;
  attempts: PlanAttemptView[];
  cost_usd: number;
  model?: string;
  use_case?: string;
}

export interface LivePlan {
  source: "recorded" | "live";
  result: PlanResultView;
  spend: { spent_usd: number; session_cap_usd: number; day_spent_usd: number; day_cap_usd: number } | null;
}

async function ok(res: Response): Promise<Record<string, unknown>> {
  if (!res.ok) {
    const hint: Record<number, string> = {
      403: " Check the passcode.",
      409: " This server is offline.",
      413: " The request is too large.",
      429: " Wait a minute, or check the spend caps.",
    };
    throw new ApiError(res.status, `${(await errorMessage(res)).replace(/[.\s]*$/, ".")}${hint[res.status] ?? ""}`);
  }
  return (await res.json()) as Record<string, unknown>;
}

function headers(passcode: string): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (passcode) h[PASSCODE_HEADER] = passcode;
  return h;
}

/** Upload an STL through the existing part intake; returns the part id. */
export async function uploadStl(base: string, body: Blob | ArrayBuffer, name: string, passcode: string, signal?: AbortSignal): Promise<string> {
  const params = new URLSearchParams({ name: name.replace(/\.stl$/i, "").slice(0, 120), units: "mm" });
  const h: Record<string, string> = { "Content-Type": "application/octet-stream" };
  if (passcode) h[PASSCODE_HEADER] = passcode;
  const out = await ok(await fetch(`${base}/parts?${params}`, { method: "POST", headers: h, body, signal }));
  const part = (out.part ?? {}) as Record<string, unknown>;
  if (typeof part.part_id !== "string") throw new ApiError(500, "The API accepted the part but returned no part_id.");
  return part.part_id;
}

export async function planLive(base: string, text: string, partId: string | null, passcode: string, signal?: AbortSignal): Promise<LivePlan> {
  const body = JSON.stringify({ text, ...(partId ? { part_id: partId } : {}) });
  const out = await ok(await fetch(`${base}/studio/plan`, { method: "POST", headers: headers(passcode), body, signal }));
  return out as unknown as LivePlan;
}

export interface RemoteVerdict {
  status: string;
  problems: string[];
  warnings: string[];
  task_id: string | null;
}

export async function validateRemote(base: string, plan: unknown, partIds: string[], signal?: AbortSignal): Promise<RemoteVerdict> {
  const body = JSON.stringify({ plan, part_ids: partIds });
  return (await ok(await fetch(`${base}/studio/validate`, { method: "POST", headers: headers(""), body, signal }))) as unknown as RemoteVerdict;
}
