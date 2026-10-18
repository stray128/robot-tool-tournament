import { formatOffset, formatUsd } from "@/lib/format";
import { buildTimeline, type RecordedSession, type Step, toolKind, TOOL_TITLES } from "@/lib/session";

import { humanize, type SessionBlock } from "./schema";
import { Facts } from "./ui";

/** The Studio's own agent tools (src/rtt/agent/studio_tool.py), in words. */
const STUDIO_TOOL_TITLES: Record<string, string> = {
  plan_task: "Plan the task",
  run_task_tournament: "Run the task tournament",
  export_build_pack: "Export the build pack",
};

const URL_RE = /https?:\/\/\S+|www\.\S+/gi;
const URL_TEST = /https?:\/\/|www\./i;
const SKIP_KEY = /url|href|path|file|sha|(^|_)id$|token|passcode/i;

/** Text from a session with any URL cut (URLs are never shown as text). */
function safe(text: string, max = 600): string {
  const s = text.replace(URL_RE, "(link)").trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

function scalar(v: unknown): string | null {
  if (typeof v === "number" && Number.isFinite(v)) return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(3)));
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string" && v.length > 0 && v.length <= 120 && !URL_TEST.test(v)) return v;
  return null;
}

/** A few readable fields of a tool's arguments or result: scalars, list sizes; no ids, paths or URLs. */
export function briefFields(v: unknown, limit = 5): string[] {
  if (typeof v === "string") return [safe(v, 160)];
  if (!v || typeof v !== "object" || Array.isArray(v)) return [];
  const out: string[] = [];
  for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
    if (out.length >= limit) break;
    if (SKIP_KEY.test(k)) continue;
    const s = scalar(x);
    const key = humanize(k).toLowerCase();
    if (s !== null) out.push(`${key}: ${s}`);
    else if (Array.isArray(x)) out.push(`${key}: ${x.length} ${x.length === 1 ? "item" : "items"}`);
  }
  return out;
}

function title(step: Step): string {
  if (step.tool && STUDIO_TOOL_TITLES[step.tool]) return STUDIO_TOOL_TITLES[step.tool]!;
  const kind = toolKind(step.tool);
  return kind === "other" ? humanize(step.tool ?? "tool call") : TOOL_TITLES[kind];
}

/** The recorded Grasp Engineer session, read only: tool calls and results, the agent's words, spend. */
export function SessionView({ block, session }: { block: SessionBlock; session: RecordedSession }) {
  const tl = buildTimeline(session.events);
  let n = 0;
  return (
    <div className="grid gap-4" data-session>
      <Facts
        items={[
          ["Model", block.model ?? tl.model ?? "n/a"],
          ["Status", block.status ?? tl.status],
          ["Turns", String(block.turns)],
          ["Tool calls", String(block.tool_calls || tl.tool_calls)],
          ["Spent", formatUsd(block.spend_usd)],
          ["Session clock", formatOffset(tl.t_last)],
        ]}
      />
      {block.answer ? (
        <p className="border-l-2 border-accent pl-2 text-ink" data-session-answer>
          <span className="mr-1.5 text-xs text-accent">Answer</span>
          {safe(block.answer, 1200)}
        </p>
      ) : null}
      <ol className="grid gap-2 text-sm" data-session-steps>
        {tl.steps.map((step) => {
          if (step.kind === "message" || step.kind === "error") {
            if (!step.text) return null;
            return (
              <li key={step.id} className="grid grid-cols-[3rem_minmax(0,1fr)] gap-2">
                <span className="num text-xs text-ink-3">{formatOffset(step.t_start)}</span>
                <p className={step.kind === "error" ? "text-fail" : step.role === "summary" ? "text-ink" : "text-ink-2"}>
                  <span className="mr-1.5 text-xs text-ink-3">{step.kind === "error" ? "Error" : step.role === "summary" ? "Summary" : "Agent"}</span>
                  {safe(step.text)}
                </p>
              </li>
            );
          }
          const args = briefFields(step.args);
          const result = step.status === "error" ? [] : briefFields(step.result);
          return (
            <li key={step.id} className="grid grid-cols-[3rem_minmax(0,1fr)] gap-2" data-session-tool={step.tool ?? ""}>
              <span className="num text-xs text-ink-3">{formatOffset(step.t_start)}</span>
              <div className="min-w-0 border border-rule p-2">
                <p className="flex flex-wrap items-baseline gap-x-2">
                  <span className="num text-xs text-ink-3">{String(++n).padStart(2, "0")}</span>
                  <span className="heading text-ink">{title(step)}</span>
                  <span className={`text-xs ${step.status === "error" ? "text-fail" : step.status === "ok" ? "text-pass" : "text-pending"}`}>
                    {step.status === "error" ? (step.refused ? "refused" : "failed") : step.status === "ok" ? "done" : "no result recorded"}
                  </span>
                  {step.cost_usd ? <span className="num text-xs text-ink-3">{formatUsd(step.cost_usd)}</span> : null}
                </p>
                {args.length ? <p className="num mt-1 text-xs text-ink-3">Asked: {args.join(" · ")}</p> : null}
                {result.length ? <p className="num mt-0.5 text-xs text-ink-2">Got: {result.join(" · ")}</p> : null}
                {step.error ? <p className="mt-0.5 text-xs text-fail">{safe(step.error, 240)}</p> : null}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
