"use client";

import Link from "next/link";
import type { ReactNode } from "react";

import { JobChip } from "@/components/agent/JobChip";
import { formatBytes, formatInt, formatOffset, formatPercent, formatUsd } from "@/lib/format";
import { FAILURE_LABELS, humanize, PARAM_LABELS } from "@/lib/labels";
import { dataUrl } from "@/lib/paths";
import {
  activeJobs,
  inspectView,
  printpackView,
  proposeView,
  readLedger,
  resultsView,
  spend,
  type Step,
  stressView,
  type Timeline,
  TOOL_TITLES,
} from "@/lib/session";
import type { FailureMode } from "@/lib/types";

export interface TimelineLinks {
  /** Published design ids: a proposal with one of these links to its design page. */
  designIds: string[];
  /** Published certificate ids by design id. */
  certificates: Record<string, string>;
  /** Bout id by replay stem, so a result row with a published replay links to it. */
  replayBouts: Record<string, string>;
}

interface AgentTimelineProps {
  timeline: Timeline;
  illustrative: boolean;
  links: TimelineLinks;
  /** Live mode: the API origin, for the print pack download and run results. */
  apiBase?: string | null;
}

function failureLabel(mode: string): string {
  return FAILURE_LABELS[mode as FailureMode] ?? humanize(mode);
}

function StepStatus({ step }: { step: Step }) {
  if (step.status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-pending">
        <span className="relative flex size-2" aria-hidden="true">
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-60 motion-reduce:hidden" />
          <span className="relative inline-flex size-2 rounded-full bg-current" />
        </span>
        Working
      </span>
    );
  }
  if (step.status === "error") return <span className="text-xs text-fail">{step.refused ? "Refused" : "Failed"}</span>;
  return <span className="text-xs text-pass">Done</span>;
}

/** Pass/fail against the agent's physics threshold; shape and word, not colour alone. */
function Verdict({ verdict }: { verdict: "pass" | "fail" | null }) {
  if (!verdict) return null;
  return (
    <span className={`stamp text-[0.6875rem] ${verdict === "pass" ? "text-pass" : "text-fail"}`}>{verdict === "pass" ? "PASS" : "FAIL"}</span>
  );
}

function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="flex flex-wrap gap-px border border-rule bg-rule text-sm">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-[8.5rem] flex-1 basis-[30%] bg-paper px-2.5 py-1.5">
          <dt className="text-xs text-ink-3">{k}</dt>
          <dd className="num text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A 0..1 share as a thin bar with its percentage; the number carries the meaning, the bar the scale. */
function ShareBar({ value, label, mark }: { value: number | null; label: string; mark?: number | null }) {
  const pct = value === null ? 0 : Math.min(Math.max(value, 0), 1) * 100;
  return (
    <span className="flex items-center gap-2">
      <span className="relative h-2 w-20 shrink-0 border border-rule-strong sm:w-28" aria-hidden="true">
        <span className="absolute inset-y-0 left-0 bg-ink-2" style={{ width: `${pct}%` }} />
        {mark !== null && mark !== undefined ? (
          <span className="absolute -inset-y-1 w-px bg-accent" style={{ left: `${Math.min(Math.max(mark, 0), 1) * 100}%` }} />
        ) : null}
      </span>
      <span className="num text-ink">{value === null ? "n/a" : formatPercent(value)}</span>
      <span className="sr-only">{label}</span>
    </span>
  );
}

const mm = (v: number) => v.toFixed(1);

function InspectBody({ step }: { step: Step }) {
  const v = inspectView(step.result);
  const items: [string, ReactNode][] = [];
  if (v.extents_mm) items.push(["Size, mm", v.extents_mm.map(mm).join(" x ")]);
  if (v.grip_width_mm !== null) items.push(["Width across the jaws", `${mm(v.grip_width_mm)} mm`]);
  if (v.fits_opening !== null) items.push(["Fits the opening", `${v.fits_opening ? "Yes" : "No"}${v.opening_mm ? `, ${v.opening_mm} mm` : ""}`]);
  if (v.volume_cm3 !== null) items.push(["Volume", `${v.volume_cm3.toFixed(1)} cm³`]);
  if (v.mass_g !== null) items.push(["Mass", `${v.mass_g.toFixed(0)} g`]);
  if (v.convex_pieces !== null) items.push(["Convex pieces", formatInt(v.convex_pieces)]);
  if (v.watertight !== null) items.push(["Watertight", v.watertight ? "Yes" : "No"]);
  if (v.conditions !== null || v.episodes_per_design !== null) {
    const over = v.conditions !== null ? ` over ${v.conditions} ${v.conditions === 1 ? "condition" : "conditions"}` : "";
    items.push(["Trials per design", `${v.episodes_per_design ?? "?"}${over}`]);
  }
  if (v.pass_threshold !== null) items.push(["Pass mark", `${formatPercent(v.pass_threshold)} held`]);
  return (
    <div className="grid gap-3">
      {items.length ? <Facts items={items} /> : null}
      {v.stable_poses.length ? (
        <div>
          <h4 className="text-xs text-ink-3">Resting {v.stable_poses.length === 1 ? "pose" : `poses (${v.stable_poses.length})`}</h4>
          <ul className="mt-1 grid gap-1 text-sm">
            {v.stable_poses.map((p, i) => (
              <li key={i} className="flex flex-wrap items-center justify-between gap-x-4 border-b border-rule py-1">
                <span className="text-ink-2">
                  {p.label ?? `Pose ${i + 1}`}
                  {p.width_mm !== null ? <span className="num ml-2 text-xs text-ink-3">grasp width {mm(p.width_mm)} mm</span> : null}
                </span>
                {p.probability !== null ? <ShareBar value={p.probability} label="likelihood of resting this way" /> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {v.baselines.length ? (
        <p className="text-sm text-ink-2">
          <span className="text-xs text-ink-3">Hand-designed baselines: </span>
          {v.baselines.map((b, i) => (
            <span key={b.name}>
              {i > 0 ? ", " : ""}
              <span className={b.ok ? "text-ink" : "text-ink-3 line-through"} title={b.why ?? undefined}>
                {b.name}
              </span>
              {!b.ok ? <span className="sr-only"> (does not fit: {b.why})</span> : null}
            </span>
          ))}
        </p>
      ) : null}
      {v.notes.map((n) => (
        <p key={n} className="text-sm text-ink-2">
          {n}
        </p>
      ))}
    </div>
  );
}

const KEY_PARAMS = ["inner_face", "tip_profile", "finger_length_mm", "finger_width_mm", "bend_angle_deg", "tip_size_mm"];

function paramTag(key: string, value: number | string): string | null {
  if (typeof value === "string") return humanize(value);
  if ((key === "bend_angle_deg" || key === "tip_size_mm") && value === 0) return null;
  const meta = PARAM_LABELS[key];
  const unit = meta?.unit === "deg" ? "°" : meta?.unit === "mm" ? " mm" : "";
  return `${meta?.label ?? humanize(key)} ${Number.isInteger(value) ? value : value.toFixed(1)}${unit}`;
}

function ProposeBody({ step, links }: { step: Step; links: TimelineLinks }) {
  const v = proposeView(step.result, step);
  const facts = [
    v.tokens_in !== null && v.tokens_out !== null ? `${formatInt(v.tokens_in + v.tokens_out)} tokens` : null,
    v.latency_s !== null ? `${v.latency_s.toFixed(1)} s` : null,
    v.cost_usd !== null ? formatUsd(v.cost_usd) : null,
  ].filter(Boolean);
  return (
    <div className="grid gap-3">
      <p className="flex flex-wrap items-baseline gap-x-3 text-xs text-ink-3">
        <span className="num text-ink-2">{v.model ?? "model not reported"}</span>
        {facts.map((f) => (
          <span key={f} className="num">
            {f}
          </span>
        ))}
      </p>
      {v.analysis ? (
        <blockquote className="border-l-2 border-accent pl-3 text-sm text-ink-2">
          <span className="sr-only">Model analysis: </span>
          {v.analysis}
        </blockquote>
      ) : null}
      {v.designs.length ? (
        <ol className="grid gap-px border border-rule bg-rule sm:grid-cols-2">
          {v.designs.map((d) => {
            const tags = KEY_PARAMS.flatMap((k) => (d.params[k] !== undefined ? [paramTag(k, d.params[k]!)] : [])).filter(
              (t): t is string => t !== null,
            );
            const linked = d.design_id && links.designIds.includes(d.design_id);
            return (
              <li key={`${d.label}-${d.design_id}`} className="bg-paper p-3">
                <p className="num text-sm text-accent">
                  {linked ? (
                    <Link className="link" href={`/design/${d.design_id}/`}>
                      {d.label}
                    </Link>
                  ) : (
                    d.label
                  )}
                </p>
                {d.hypothesis ? (
                  <p className="mt-1 text-sm text-ink">
                    <span className="sr-only">Hypothesis: </span>
                    {d.hypothesis}
                  </p>
                ) : null}
                {tags.length ? (
                  <ul className="mt-2 flex flex-wrap gap-1 text-[0.6875rem] text-ink-2" aria-label="Key parameters">
                    {tags.map((t) => (
                      <li key={t} className="border border-rule px-1.5 py-0.5">
                        {t}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            );
          })}
        </ol>
      ) : null}
      {v.rejections.length ? (
        <details className="text-sm">
          <summary className="cursor-pointer text-ink-3">
            {v.rejections.length} {v.rejections.length === 1 ? "proposal" : "proposals"} rejected before simulation
          </summary>
          <ul className="mt-1 grid gap-0.5 text-xs text-ink-2">
            {v.rejections.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </details>
      ) : v.rejected ? (
        <p className="text-xs text-ink-3">{v.rejected} proposals rejected before simulation.</p>
      ) : null}
    </div>
  );
}

function JobBody({ step, timeline, illustrative }: { step: Step; timeline: Timeline; illustrative: boolean }) {
  const designs = step.args.designs;
  const target = typeof step.args.target === "string" ? step.args.target : null;
  const ask = Array.isArray(designs) ? `${designs.length} designs${target ? `, on ${target === "nebius" ? "Nebius" : target}` : ""}` : null;
  return (
    <div className="grid gap-2">
      {ask ? <p className="text-sm text-ink-2">{ask}</p> : null}
      {step.job_ids.length ? (
        <div className="flex flex-wrap gap-2">
          {step.job_ids.map((id) => {
            const job = timeline.jobs[id];
            return job ? <JobChip key={id} job={job} illustrative={illustrative} /> : null;
          })}
        </div>
      ) : step.status === "running" ? (
        <p className="text-sm text-ink-3">Submitting the job...</p>
      ) : null}
      {step.polls > 0 ? (
        <p className="text-xs text-ink-3">
          The agent checked on it {step.polls} {step.polls === 1 ? "time" : "times"}.
        </p>
      ) : null}
    </div>
  );
}

function topFailure(modes: Record<string, number>): string | null {
  const top = Object.entries(modes).sort((a, b) => b[1] - a[1])[0];
  if (!top || top[1] <= 0) return null;
  return `${failureLabel(top[0])} (${top[1]})`;
}

function ResultsBody({ step, links, apiBase }: { step: Step; links: TimelineLinks; apiBase: string | null }) {
  const v = resultsView(step.result);
  if (!v.rows.length) return <p className="text-sm text-ink-3">No scores in this result.</p>;
  return (
    <div className="grid gap-2">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[300px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-rule-strong text-xs text-ink-3">
              <th scope="col" className="py-1 pr-3 font-normal">
                Design
              </th>
              <th scope="col" className="py-1 pr-3 font-normal">
                Held{v.pass_threshold !== null ? ` (pass at ${formatPercent(v.pass_threshold)})` : ""}
              </th>
              <th scope="col" className="hidden py-1 font-normal sm:table-cell">
                Most common failure
              </th>
            </tr>
          </thead>
          <tbody>
            {v.rows.map((r) => {
              const bout = r.replay ? links.replayBouts[r.replay] : undefined;
              return (
                <tr key={`${r.label}-${r.design_id}`} className="border-b border-rule">
                  <th scope="row" className="num py-1.5 pr-3 align-top font-normal text-ink">
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      {r.label}
                      <Verdict verdict={r.verdict} />
                      {r === v.best ? <span className="font-sans text-xs text-accent">best</span> : null}
                      {bout ? (
                        <Link href={`/bout/${bout}/`} className="link font-sans text-xs">
                          replay
                        </Link>
                      ) : null}
                    </span>
                    <span className="mt-0.5 block font-sans text-xs text-ink-3 sm:hidden">{topFailure(r.failure_modes) ?? ""}</span>
                  </th>
                  <td className="py-1.5 pr-3 align-top">
                    <ShareBar value={r.success} label={`of ${r.episodes ?? "?"} trials held`} mark={v.pass_threshold} />
                    {r.episodes !== null ? <span className="num text-xs text-ink-3">of {r.episodes} trials</span> : null}
                  </td>
                  <td className="hidden py-1.5 align-top text-ink-2 sm:table-cell">{topFailure(r.failure_modes) ?? "None"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {apiBase && v.run_id ? (
        <p className="text-xs">
          <a className="link" href={`${apiBase}/runs/${encodeURIComponent(v.run_id)}`} target="_blank" rel="noreferrer">
            Every episode of this run (JSON)
          </a>
        </p>
      ) : null}
    </div>
  );
}

function StressBody({ step, links }: { step: Step; links: TimelineLinks }) {
  const v = stressView(step.result);
  const certId = v.certificate_id ?? (v.design_id ? links.certificates[v.design_id] : undefined);
  const items: [string, ReactNode][] = [];
  if (v.score !== null) items.push(["Certificate score", formatPercent(v.score)]);
  if (v.passed) items.push(["Checks held", v.passed]);
  if (v.tolerance_mm.x !== null || v.tolerance_mm.y !== null) {
    items.push(["Placement tolerance", `x ${v.tolerance_mm.x ?? "none"} / y ${v.tolerance_mm.y ?? "none"} mm`]);
  }
  return (
    <div className="grid gap-3">
      <p className="flex flex-wrap items-center gap-2">
        {v.design_id ? <span className="num text-sm text-accent">{v.label ?? v.design_id}</span> : null}
        <Verdict verdict={v.verdict} />
      </p>
      {items.length ? <Facts items={items} /> : null}
      <ul className="grid gap-1 text-sm">
        {(
          [
            ["Placement error up to 3 mm", v.categories.placement],
            ["Shake during the hold", v.categories.shake],
            ["Sideways push", v.categories.push],
          ] as const
        ).map(([label, value]) =>
          value === null ? null : (
            <li key={label} className="flex flex-wrap items-center justify-between gap-x-4 border-b border-rule py-1">
              <span className="text-ink-2">{label}</span>
              <ShareBar value={value} label="of these checks held" />
            </li>
          ),
        )}
      </ul>
      {v.failed.length ? (
        <p className="text-xs text-ink-3">
          Dropped in:{" "}
          {v.failed.map((f, i) => (
            <span key={f.test}>
              {i > 0 ? "; " : ""}
              <span className="num">{f.test}</span>
              {f.failure_mode ? ` (${failureLabel(f.failure_mode).toLowerCase()})` : ""}
            </span>
          ))}
        </p>
      ) : null}
      {certId ? (
        <p className="text-sm">
          <Link className="link" href={`/certificate/${certId}/`}>
            Open the full robustness certificate
          </Link>
        </p>
      ) : null}
    </div>
  );
}

function PrintPackBody({ step, illustrative, apiBase }: { step: Step; illustrative: boolean; apiBase: string | null }) {
  const v = printpackView(step.result);
  const href = apiBase && v.printpack_id ? `${apiBase}${v.url ?? `/printpack/${encodeURIComponent(v.printpack_id)}`}` : null;
  return (
    <div className="grid gap-3">
      {v.design_id || v.verdict ? (
        <p className="flex flex-wrap items-center gap-2 text-sm">
          {v.design_id ? <span className="num text-accent">{v.design_id}</span> : null}
          <Verdict verdict={v.verdict} />
          {v.certificate_included ? <span className="text-xs text-ink-3">certificate included</span> : null}
        </p>
      ) : null}
      {v.files.length ? (
        <ul className="border-t border-rule-strong text-sm">
          {v.files.map((f) => (
            <li key={f.name} className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-rule py-1.5">
              {f.path ? (
                <a className="link num" href={dataUrl(f.path)} download={f.name}>
                  {f.name}
                </a>
              ) : (
                <span className="num text-ink">{f.name}</span>
              )}
              {f.bytes !== null ? <span className="num text-xs text-ink-3">{formatBytes(f.bytes)}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
      {href ? (
        <p>
          <a href={href} className="inline-block bg-accent px-4 py-2 text-sm text-accent-ink no-underline">
            Download the print pack{v.zip_bytes !== null ? ` (zip, ${formatBytes(v.zip_bytes)})` : ""}
          </a>
        </p>
      ) : (
        <p className="text-sm text-ink-3">
          {illustrative
            ? "This illustrative session made no files. A live run serves its print pack for download here."
            : v.files.some((f) => f.path)
              ? "The pack's files are linked above; the zip itself is served only by a live run."
              : "This recorded session's print pack is not hosted on the static site."}
        </p>
      )}
    </div>
  );
}

function LedgerBody({ step, timeline }: { step: Step; timeline: Timeline }) {
  const l = readLedger(step.result);
  const s = spend(timeline);
  const items: [string, ReactNode][] = [
    ["Model calls", formatUsd(l.llm_usd ?? s.llm_usd)],
    ["Simulation jobs", `${formatUsd(l.compute_usd ?? s.compute_usd)}${s.estimated ? " est." : ""}`],
    ["Spent so far", formatUsd(l.total_usd ?? s.total_usd)],
  ];
  const cap = l.budget_usd ?? timeline.caps?.budget_usd ?? null;
  if (cap !== null) items.push(["Session cap", formatUsd(cap)]);
  if (l.tool_calls !== null) items.push(["Tool calls", `${l.tool_calls}${l.max_tool_calls ? ` of ${l.max_tool_calls}` : ""}`]);
  return <Facts items={items} />;
}

function OtherBody({ step }: { step: Step }) {
  return (
    <details className="text-sm">
      <summary className="cursor-pointer text-ink-2">Show the raw result</summary>
      <pre className="num mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-all bg-paper-2 p-2 text-xs text-ink-2">
        {JSON.stringify(step.result, null, 1)?.slice(0, 4000) ?? "null"}
      </pre>
    </details>
  );
}

function StepBody(props: { step: Step; timeline: Timeline; illustrative: boolean; links: TimelineLinks; apiBase: string | null }) {
  const { step } = props;
  if (step.status === "error" && step.error) {
    return (
      <p role="alert" className="text-sm text-fail">
        {step.refused ? "A session cap refused this call: " : ""}
        {step.error}
      </p>
    );
  }
  if (step.status === "running" && step.kind !== "job") {
    return <p className="text-sm text-ink-3">Waiting for the tool to answer...</p>;
  }
  switch (step.kind) {
    case "inspect":
      return <InspectBody step={step} />;
    case "propose":
      return <ProposeBody step={step} links={props.links} />;
    case "job":
      return <JobBody step={step} timeline={props.timeline} illustrative={props.illustrative} />;
    case "results":
      return <ResultsBody step={step} links={props.links} apiBase={props.apiBase} />;
    case "stress":
      return <StressBody step={step} links={props.links} />;
    case "printpack":
      return <PrintPackBody step={step} illustrative={props.illustrative} apiBase={props.apiBase} />;
    case "ledger":
      return <LedgerBody step={step} timeline={props.timeline} />;
    default:
      return <OtherBody step={step} />;
  }
}

/** Caps and spend for the session so far: tool calls, concurrent jobs, dollars, model calls. */
export function SessionMeter({ timeline }: { timeline: Timeline }) {
  const s = spend(timeline);
  const caps = timeline.caps;
  const cells: [string, string][] = [
    ["Tool calls", `${timeline.tool_calls}${caps?.max_tool_calls ? ` of ${caps.max_tool_calls}` : ""}`],
    ["Jobs running", `${activeJobs(timeline).length}${caps?.max_concurrent_jobs ? ` of ${caps.max_concurrent_jobs} max` : ""}`],
    ["Spent", `${formatUsd(s.total_usd)}${caps?.budget_usd ? ` of ${formatUsd(caps.budget_usd)}` : ""}${s.estimated ? " est." : ""}`],
    // An offline session's "calls" are the scripted stand-in, which bills nothing; say so
    // rather than show token counts beside the "no model calls" banner.
    timeline.offline
      ? ["Model calls", `none (${timeline.llm.calls} scripted turns)`]
      : ["Model calls", `${timeline.llm.calls}${timeline.llm.calls ? `, ${formatInt(timeline.llm.tokens_in + timeline.llm.tokens_out)} tokens` : ""}`],
    ["Session clock", formatOffset(timeline.t_last)],
  ];
  return (
    <dl className="flex flex-wrap gap-px border border-rule bg-rule text-sm">
      {cells.map(([k, v]) => (
        <div key={k} className="min-w-[9rem] flex-1 bg-paper px-3 py-2">
          <dt className="text-xs text-ink-3">{k}</dt>
          <dd className="num text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

const MESSAGE_LABEL = { agent: "Agent", system: "System", summary: "Summary" } as const;

/**
 * The Grasp Engineer's session as a vertical run of tool cards: inspect, propose
 * (with Nemotron's hypotheses), simulation jobs with live status, results, stress test,
 * print pack, ledger. The agent's own words and the step summaries sit between them.
 */
export function AgentTimeline({ timeline, illustrative, links, apiBase = null }: AgentTimelineProps) {
  let n = 0;
  return (
    <ol className="relative grid grid-cols-[minmax(0,1fr)] gap-3 before:absolute before:bottom-2 before:left-[0.9375rem] before:top-2 before:w-px before:bg-rule-strong">
      {timeline.steps.map((step) => {
        if (step.kind === "message" || step.kind === "error") {
          const label = step.kind === "error" ? "Error" : MESSAGE_LABEL[step.role ?? "agent"];
          return (
            <li key={step.id} className="relative min-w-0 pl-10">
              <span
                className={`absolute left-[0.6875rem] top-2 size-2.5 rotate-45 border ${
                  step.kind === "error" ? "border-fail bg-fail" : step.role === "summary" ? "border-accent bg-accent-soft" : "border-ink-3 bg-paper"
                }`}
                aria-hidden="true"
              />
              <p className={`text-sm ${step.kind === "error" ? "text-fail" : step.role === "summary" ? "text-ink" : "text-ink-2"}`}>
                <span className="num mr-2 text-xs text-ink-3">{formatOffset(step.t_start)}</span>
                <span className={step.role === "summary" ? "mr-1.5 text-xs text-accent" : "sr-only"}>{label}: </span>
                {step.text}
              </p>
            </li>
          );
        }
        const number = ++n;
        const title = step.kind === "other" && step.tool ? humanize(step.tool) : TOOL_TITLES[step.kind];
        // A job card stays open until its jobs finish, not just until the launch call returns.
        const jobEnd = Math.max(-Infinity, ...step.job_ids.map((id) => timeline.jobs[id]?.updated_t ?? -Infinity));
        const end = step.t_end === null ? null : Math.max(step.t_end, jobEnd);
        return (
          <li key={step.id} className="relative min-w-0 pl-10">
            <span
              className={`num absolute left-0 top-2.5 grid size-[1.875rem] place-items-center border text-xs ${
                step.status === "running"
                  ? "border-pending bg-paper text-pending"
                  : step.status === "error"
                    ? "border-fail bg-paper text-fail"
                    : "border-ink bg-ink text-paper"
              }`}
              aria-hidden="true"
            >
              {String(number).padStart(2, "0")}
            </span>
            <article className="min-w-0 border border-rule bg-paper/90 p-3 sm:p-4" aria-label={`Step ${number}: ${title}`}>
              <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <h3 className="heading text-base">
                  {title}
                  {step.tool ? <span className="num ml-2 text-xs font-normal text-ink-3">{step.tool}</span> : null}
                </h3>
                <span className="flex items-center gap-3">
                  <span className="num text-xs text-ink-3">
                    {formatOffset(step.t_start)}
                    {end !== null && end - step.t_start >= 1 ? ` to ${formatOffset(end)}` : ""}
                  </span>
                  <StepStatus step={step} />
                </span>
              </header>
              <div className="mt-3">
                <StepBody step={step} timeline={timeline} illustrative={illustrative} links={links} apiBase={apiBase} />
              </div>
            </article>
          </li>
        );
      })}
    </ol>
  );
}
