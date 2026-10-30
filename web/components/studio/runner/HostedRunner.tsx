"use client";

import Link from "next/link";
import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";

import { FamilyIcon } from "@/components/story/FamilyIcon";
import { PRODUCT_MODE_LABEL } from "@/components/studio/schema";
import { formatDuration } from "@/lib/format";
import { autoOffPassed, freeAllowanceLabel, KEY_PRIVACY_LINE, type PlanReply, type Provider, reportLink, RUN_CAP_LINE, runsLeftLine, type RunSummary, type SetupBrief } from "@/lib/runner";

import { type Fault, RunnerController, type RunnerState } from "./controller";
import { KEY_FIELD_ATTRIBUTES } from "./keyfield";

const RUN_YOURSELF = "/docs/tutorials/run-it-yourself/";

const PROVIDER_NAME: Record<Exclude<Provider, "free">, string> = {
  nvidia: "NVIDIA API key",
  tokenfactory: "Nebius Token Factory key",
};

const KIND_TEXT: Record<string, string> = {
  single: "One tool",
  tool_change: "Tool change",
  two_tool_head: "Two-tool head",
  multi_head: "Two-tool head",
};

function kindText(kind: string): string {
  return KIND_TEXT[kind] ?? kind.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

/** "42 s", "3 min 05 s". */
function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

function Meter({ k, n, tone, label }: { k: number; n: number; tone: "accent" | "baseline"; label: string }) {
  return (
    <div>
      <p className="flex items-baseline justify-between gap-2 text-xs">
        <span className={tone === "accent" ? "text-accent" : "text-baseline"}>{label}</span>
        <span className="num text-ink">
          {k} of {n} runs
        </span>
      </p>
      <span className="mt-1 block h-1.5 overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
        <span className={`block h-full rounded-full ${tone === "accent" ? "bg-accent" : "bg-baseline"}`} style={{ width: `${n ? Math.max((k / n) * 100, 1) : 0}%` }} />
      </span>
    </div>
  );
}

function FaultNotice({ fault, onRetry, retryDisabled, hint }: { fault: Fault; onRetry: () => void; retryDisabled?: boolean; hint?: string }) {
  return (
    <div role="alert" className="mt-3 border-l-2 border-fail pl-3 text-sm" data-hosted-error={fault.kind}>
      <p className="text-ink">
        {fault.message}{" "}
        {fault.kind === "off" ? (
          <Link className="link" href={RUN_YOURSELF}>
            Run it yourself
          </Link>
        ) : null}
      </p>
      {fault.problems?.length ? (
        <ul className="mt-1 grid gap-0.5 text-xs text-ink-2">
          {fault.problems.map((p, i) => (
            <li key={`${i}-${p}`} className="break-words">
              {p}
            </li>
          ))}
        </ul>
      ) : null}
      {fault.retry ? (
        <p className="mt-2 flex flex-wrap items-center gap-3">
          <button type="button" className="btn btn-quiet text-sm" onClick={onRetry} disabled={retryDisabled} data-hosted-retry>
            Try again
          </button>
          {retryDisabled && hint ? <span className="text-xs text-ink-3">{hint}</span> : null}
        </p>
      ) : null}
    </div>
  );
}

function SetupBlock({ setup, tone }: { setup: SetupBrief; tone: "accent" | "baseline" }) {
  return (
    <>
      <p className="heading mt-1 break-words text-xl text-ink">{setup.label}</p>
      <p className="mt-0.5 text-sm text-ink-2">{kindText(setup.kind)}</p>
      {setup.tools.length ? (
        <ul className="mt-3 grid gap-1.5 text-sm">
          {setup.tools.map((t, i) => (
            <li key={`${t.label}-${i}`} className="flex items-center gap-2">
              <FamilyIcon family={t.family || t.label} className={`size-5 shrink-0 ${tone === "accent" ? "text-accent" : "text-baseline"}`} />
              <span className="min-w-0 break-words text-ink">{t.label}</span>
              {t.family ? <span className="text-xs text-ink-3">{t.family}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

function ResultCard({ summary, base, runId, costUsd }: { summary: RunSummary; base: string; runId: string; costUsd: number | null }) {
  const report = reportLink(base, runId, summary.reportUrl);
  const shown = summary.pick ?? summary.closest;
  const out = summary.ruledOut;
  return (
    <div className="panel mt-5 p-5" data-run-result data-run-outcome={summary.pick ? "recommended" : "none"} aria-label="Result of the physics run">
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <span className="stamp text-pass">DONE</span>
        <span className="text-ink">Simulated, verify on hardware</span>
        <span className="text-xs text-ink-3" data-product-mode>
          {PRODUCT_MODE_LABEL}
        </span>
      </p>

      <div className="mt-4 grid gap-5 md:grid-cols-2">
        <div className="min-w-0" data-run-pick>
          <h3 className="text-xs text-ink-3">{summary.pick ? "Recommended setup" : "No setup completed the task"}</h3>
          {shown ? (
            <>
              {!summary.pick ? <p className="mt-1 text-xs text-ink-3">The closest setup was:</p> : null}
              <SetupBlock setup={shown} tone="accent" />
            </>
          ) : (
            <p className="mt-1 text-sm text-ink-2">The run found no setup to recommend.</p>
          )}
          {summary.completion ? (
            <div className="mt-4">
              <Meter k={summary.completion.k} n={summary.completion.n} tone="accent" label="Completes the task" />
            </div>
          ) : null}
        </div>

        <div className="min-w-0" data-run-baseline>
          <h3 className="text-xs text-ink-3">Best single tool</h3>
          {summary.baseline ? (
            <>
              <p className="heading mt-1 break-words text-xl text-ink">{summary.baseline.label}</p>
              <p className="mt-0.5 text-sm text-ink-2">The best single off-the-shelf tool, on the same runs</p>
              <div className="mt-4">
                <Meter k={summary.baseline.completion.k} n={summary.baseline.completion.n} tone="baseline" label="Completes the task" />
              </div>
            </>
          ) : (
            <p className="mt-1 text-sm text-ink-2">No single-tool comparison came back for this run.</p>
          )}
        </div>
      </div>

      {summary.why ? <p className="mt-5 max-w-[68ch] text-ink-2">{summary.why}</p> : null}
      {out && out.pruned > 0 ? (
        <p className="mt-2 text-sm text-ink-3">
          {out.candidates !== null ? `${out.pruned} of ${out.candidates} tools were` : `${out.pruned} ${out.pruned === 1 ? "tool was" : "tools were"}`} ruled out before the full runs.
        </p>
      ) : null}

      {summary.steps.length ? (
        <div className="mt-5 overflow-x-auto" data-run-steps>
          <table className="w-full table-fixed border-collapse text-left text-sm">
            <caption className="sr-only">Each step, the tool that does it and how often it passed</caption>
            <colgroup>
              <col className="w-[38%]" />
              <col className="w-[42%]" />
              <col className="w-[20%]" />
            </colgroup>
            <thead>
              <tr className="border-b border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-1.5 pr-3 font-normal">
                  Step
                </th>
                <th scope="col" className="py-1.5 pr-3 font-normal">
                  Tool
                </th>
                <th scope="col" className="py-1.5 text-right font-normal">
                  Passed
                </th>
              </tr>
            </thead>
            <tbody>
              {summary.steps.map((st) => {
                const pct = st.successRate === null ? null : Math.round((st.successRate > 1 ? st.successRate / 100 : st.successRate) * 100);
                return (
                  <tr key={st.step} className="border-b border-rule align-top">
                    <th scope="row" className="break-words py-1.5 pr-3 font-normal text-ink">
                      {st.label}
                    </th>
                    <td className="break-words py-1.5 pr-3 text-ink-2">{st.tool ?? "No tool"}</td>
                    <td className="num py-1.5 text-right text-ink">{pct === null ? "n/a" : `${pct}%`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      <p className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        {report ? (
          <a className="link" href={report} download rel="noopener noreferrer" data-run-report>
            Download the full report (JSON)
          </a>
        ) : null}
        {summary.codeVersion ? <span className="num text-xs text-ink-3">code {summary.codeVersion.slice(0, 12)}</span> : null}
        {costUsd !== null ? <span className="num text-xs text-ink-3">run cost ${costUsd.toFixed(2)}</span> : null}
      </p>
      <p className="mt-3 text-xs text-ink-3">Replays are shown for the recorded examples; hosted runs show the summary.</p>
    </div>
  );
}

function PlanOutcome({ reply, loaded }: { reply: PlanReply; loaded: boolean }) {
  const detail = [reply.model, reply.calls ? `${reply.calls} model ${reply.calls === 1 ? "call" : "calls"}` : null].filter(Boolean).join(", ");
  return (
    <div className="mt-3 border-l-2 border-accent pl-3 text-sm" data-hosted-plan-status={reply.status}>
      <p className="flex flex-wrap items-center gap-2">
        <span className={`stamp ${reply.status === "valid" ? "text-pass" : reply.status === "refused" ? "text-pending" : "text-fail"}`}>
          {reply.status === "valid" ? "VALID PLAN" : reply.status === "refused" ? "REFUSED" : "NO PLAN"}
        </span>
        <span className="text-ink">
          {reply.status === "valid"
            ? "A plan the simulator can run. It is loaded in the steps below: check it and edit it, then run it."
            : reply.status === "refused"
              ? "The simulator cannot do this use case."
              : "The planner could not make a valid plan from this description."}
        </span>
      </p>
      {detail ? <p className="num mt-1 text-xs text-ink-3">{detail}</p> : null}
      {reply.problems.length ? (
        <ul className="mt-2 grid gap-0.5 text-xs text-ink-2">
          {reply.problems.map((p, i) => (
            <li key={`${i}-${p}`} className="break-words">
              {p}
            </li>
          ))}
        </ul>
      ) : null}
      {reply.status !== "valid" ? (
        <p className="mt-2 text-xs text-ink-3">
          {loaded ? "What came back is loaded in the steps below, so you can fix it. " : ""}Change the description and plan again, or build the plan by hand.
        </p>
      ) : null}
    </div>
  );
}

/**
 * "Plan and run it live": plan the description with Nemotron on a hosted service, then test the
 * plan in physics on a hosted runner. It shows only when the build has NEXT_PUBLIC_RUNNER_URL and
 * the runner's /health answers, so the public site reads as before everywhere else. The key stays in
 * the controller's memory (see controller.ts).
 */
export default function HostedRunner({
  base,
  description,
  runDoc,
  runLimit,
  hasPlan,
  hasStl,
  onPlanned,
}: {
  base: string;
  description: string;
  /** The editor's current plan as the TaskSpec document, or null unless it is valid. */
  runDoc: Record<string, unknown> | null;
  /** Why the hosted runner would turn the plan down (too many steps, an uploaded part...), or null. */
  runLimit: string | null;
  hasPlan: boolean;
  hasStl: boolean;
  /** A plan came back: load it into the editor. */
  onPlanned: (reply: PlanReply) => void;
}) {
  const onPlannedRef = useRef(onPlanned);
  useEffect(() => {
    onPlannedRef.current = onPlanned;
  });
  const [c] = useState(() => new RunnerController({ base, onPlan: (r) => onPlannedRef.current(r) }));
  const s: RunnerState = useSyncExternalStore(c.subscribe, c.getSnapshot, c.getSnapshot);
  const rootRef = useRef<HTMLElement | null>(null);
  const scrolled = useRef(false);
  const id = useId();

  useEffect(() => {
    void c.start();
    return () => c.stop();
  }, [c]);

  const planning = s.planPhase === "planning";
  const inRun = s.runPhase === "submitting" || s.runPhase === "checking" || s.runPhase === "queued" || s.runPhase === "running";
  const now = useNow(planning || inRun);

  // A reload with #run-<id> lands on the page top; bring the run into view once.
  useEffect(() => {
    if (s.gate === "ready" && s.runPhase !== "idle" && !scrolled.current) {
      scrolled.current = true;
      if (s.runPhase === "checking") rootRef.current?.scrollIntoView({ block: "start" });
    }
  }, [s.gate, s.runPhase]);

  if (s.gate === "checking" || s.gate === "hidden") return null;

  const free = s.provider === "free";
  const canPlan = c.canPlan(description);
  const needKey = !free && s.key.trim().length === 0;
  const planHint = !description.trim() ? "Describe the use case above first." : needKey ? "Paste your key, or use the free allowance." : null;
  const plansLeft = s.health?.plansLeft ?? null;
  const runsLeft = s.health?.runsLeft ?? null;
  const planElapsed = s.planStartedAt !== null ? (now - s.planStartedAt) / 1000 : 0;
  const runElapsed = s.runStartedAt !== null ? (now - s.runStartedAt) / 1000 : 0;
  const done = s.runPhase === "done" && s.runRecord?.result;
  const failedRun = s.runPhase === "failed";

  const providers: { value: Provider; label: string }[] = [
    { value: "nvidia", label: "Use my NVIDIA API key" },
    { value: "tokenfactory", label: "Use my Nebius Token Factory key" },
    { value: "free", label: freeAllowanceLabel(plansLeft) },
  ];

  return (
    <section
      ref={rootRef}
      id="hosted"
      className="mt-10 scroll-mt-20 rounded-xl border border-dashed border-accent/50 bg-surface/60 p-4 sm:p-6"
      aria-labelledby="hosted-title"
      data-hosted-runner
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 id="hosted-title" className="heading text-xl">
          Plan and run it live
        </h2>
        <span className="stamp text-accent">HOSTED</span>
      </div>
      <p className="prose-measure mt-2 text-sm text-ink-2">
        Plan your description with NVIDIA Nemotron, check the plan in the steps below, then test it in MuJoCo physics on a hosted runner. Product mode, simulated, verify
        on hardware.
      </p>

      {s.gate === "off" ? (
        <p role="status" className="mt-4 border-l-2 border-pending pl-3 text-sm text-ink" data-hosted-off>
          The hosted runner is off. Run it on your own machine:{" "}
          <Link className="link" href={RUN_YOURSELF}>
            Run it yourself
          </Link>
        </p>
      ) : (
        <>
          <div className="mt-5 grid gap-6 lg:grid-cols-2">
            <fieldset className="min-w-0">
              <legend className="text-xs text-ink-3">Who pays for the planning call</legend>
              <div className="mt-2 grid gap-2">
                {providers.map((p) => (
                  <label
                    key={p.value}
                    className={`flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2 text-sm ${s.provider === p.value ? "border-accent bg-accent-soft" : "border-rule-strong hover:bg-paper-2"}`}
                  >
                    <input
                      type="radio"
                      name={`${id}-provider`}
                      value={p.value}
                      checked={s.provider === p.value}
                      onChange={() => c.setProvider(p.value)}
                      disabled={planning}
                      className="mt-1 accent-[var(--accent)]"
                    />
                    <span className="min-w-0 text-ink">{p.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="min-w-0">
              {free ? (
                <p className="text-sm text-ink-2" data-free-note>
                  No key is needed. The free allowance is shared by everyone who visits and is limited each day.
                </p>
              ) : (
                <>
                  <label className="grid gap-1 text-sm">
                    <span className="text-xs text-ink-3">{PROVIDER_NAME[s.provider as Exclude<Provider, "free">]}</span>
                    <input
                      {...KEY_FIELD_ATTRIBUTES}
                      value={s.key}
                      onChange={(e) => c.setKey(e.target.value)}
                      disabled={planning}
                      aria-describedby={`${id}-privacy`}
                      className="w-full min-w-0 rounded-md border border-rule-strong bg-surface px-2 py-1.5 text-ink"
                      data-hosted-key
                    />
                  </label>
                  <label className="mt-2 flex items-center gap-2 text-sm text-ink-2">
                    <input type="checkbox" checked={s.forget} onChange={(e) => c.setForget(e.target.checked)} className="accent-[var(--accent)]" />
                    Forget the key after each request
                  </label>
                  <p id={`${id}-privacy`} className="mt-2 text-xs text-ink-3" data-key-privacy>
                    {KEY_PRIVACY_LINE}
                  </p>
                </>
              )}
              {hasStl ? <p className="mt-2 text-xs text-ink-3">The hosted planner reads the description only. The part file stays in your browser.</p> : null}
            </div>
          </div>

          <div className="mt-6 border-t border-rule pt-5">
            <div className="flex flex-wrap items-center gap-3">
              <button type="button" className="btn btn-primary" disabled={!canPlan} onClick={() => void c.plan(description)} data-plan-live>
                Plan live
              </button>
              {planHint && !planning ? <span className="text-xs text-ink-3">{planHint}</span> : null}
            </div>
            <p className="mt-3 min-h-5 text-sm text-ink-2" role="status" data-hosted-progress>
              {planning ? `Nemotron is planning, usually 15–45 s. Planning can take up to five minutes. ${clock(planElapsed)} so far.` : ""}
            </p>
            {s.planFault ? (
              <FaultNotice
                fault={s.planFault}
                onRetry={() => void c.plan(description)}
                retryDisabled={!canPlan}
                hint={needKey ? "Paste your key again; it is forgotten after each request." : undefined}
              />
            ) : null}
            {s.planReply && s.planPhase === "done" ? <PlanOutcome reply={s.planReply} loaded={s.planReply.plan !== null} /> : null}
          </div>

          <div className="mt-6 border-t border-rule pt-5" data-run-area>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                className="btn btn-primary"
                disabled={!runDoc || inRun || runLimit !== null}
                onClick={() => runDoc && !runLimit && void c.run(runDoc)}
                data-run-physics
              >
                Run in physics
              </button>
              {!inRun && (!runDoc || runLimit) ? (
                <span className="text-xs text-ink-3">
                  {runLimit ?? (hasPlan ? "Fix every problem in the steps to run it." : "Plan live, play an example or build a plan in the steps below to run it.")}
                </span>
              ) : null}
            </div>
            <p className="mt-2 text-xs text-ink-3" data-run-cap>
              {RUN_CAP_LINE}.{runsLeftLine(runsLeft)}
            </p>

            <div role="status" aria-live="polite" className="mt-3 text-sm text-ink-2" data-run-status={s.runPhase}>
              {s.runPhase === "submitting" ? `Sending the plan to the runner. A plan it has not seen before can take up to three minutes to be accepted. ${clock(runElapsed)} so far.` : null}
              {s.runPhase === "checking" ? "Checking the run..." : null}
              {s.runPhase === "queued" ? (
                <p>
                  <span className="stamp mr-2 text-pending">QUEUED</span>
                  Waiting for a physics worker. {clock(runElapsed)} so far
                  {s.runEtaS ? `, about ${formatDuration(s.runEtaS)} expected` : ""}.
                </p>
              ) : null}
              {s.runPhase === "running" ? (
                <p>
                  <span className="stamp mr-2 text-accent">RUNNING</span>
                  Testing every tool in physics. {clock(runElapsed)} so far.
                </p>
              ) : null}
              {failedRun ? (
                <p className="border-l-2 border-fail pl-3 text-ink" data-run-failed>
                  <span className="stamp mr-2 text-fail">FAILED</span>
                  The run stopped before it finished.
                  {s.runRecord?.error ? <span className="mt-1 block break-words text-ink-2">{s.runRecord.error}</span> : null}
                </p>
              ) : null}
            </div>
            {s.runId && s.runPhase !== "idle" ? (
              <p className="mt-2 text-xs text-ink-3">
                Run <span className="num break-all">{s.runId}</span> is in the address, so reloading this page picks it back up. Anyone with the address can see it.{" "}
                {inRun ? null : (
                  <button type="button" className="link" onClick={() => c.reset()}>
                    Clear this run
                  </button>
                )}
              </p>
            ) : null}
            {s.runFault ? (
              <FaultNotice fault={s.runFault} onRetry={() => (s.runId ? c.poll() : runDoc ? void c.run(runDoc) : undefined)} retryDisabled={!s.runId && !runDoc} />
            ) : null}
            {done && s.runRecord?.result && s.runId ? <ResultCard summary={s.runRecord.result} base={base} runId={s.runId} costUsd={s.runRecord.costEstimateUsd} /> : null}
          </div>
        </>
      )}

      {s.health?.autoOff ? (
        <p className="mt-5 text-xs text-ink-3" data-hosted-auto-off>
          {autoOffPassed(s.health.autoOff, now)
            ? `This hosted service was a limited trial and switched itself off on ${formatOff(s.health.autoOff)}.`
            : `This hosted service is a limited trial and switches itself off on ${formatOff(s.health.autoOff)}.`}{" "}
          Everything on this page also runs on your own machine.
        </p>
      ) : null}
    </section>
  );
}

/** "2026-12-15" as "15 December 2026". */
function formatOff(iso: string): string {
  const ms = Date.parse(`${iso.slice(0, 10)}T00:00:00Z`);
  if (!Number.isFinite(ms)) return iso;
  return new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
}
