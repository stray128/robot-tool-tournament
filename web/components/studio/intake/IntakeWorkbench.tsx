"use client";

import Link from "next/link";
import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiBase } from "@/lib/api";
import { countOf, formatBytes, formatInt } from "@/lib/format";
import { liveCopy } from "@/lib/live_copy";
import { type PlanReply, runnerBase } from "@/lib/runner";
import { PREFILL_KEY } from "@/lib/site";
import { binaryTriangleCount, checkStlBuffer, checkStlFile, guessUnit, measurePositions, toMillimetres, type Vec3 } from "@/lib/stl";

import HostedRunner from "../runner/HostedRunner";
import { exportSource, hostedLimit, planFromReply, runDocument } from "../runner/handoff";
import { planLive, type PlanResultView, type RemoteVerdict, uploadStl, validateRemote } from "./api";
import StepEditor from "./StepEditor";
import { addObject, addStep, type Catalog, emptyPlan, type PlanArgs, taskId, validatePlan } from "./taskspec";

export interface RecordedExample extends PlanResultView {
  case_id: string;
  setting: string;
  title: string;
  text: string;
  hero: boolean;
  expect: "plan" | "refusal";
  /** When this example was recorded live (the v2 heroes and the v1 test set were recorded apart). */
  recorded_at?: string;
}

export interface RecordedPlans {
  recorded_at: string;
  model: string;
  label: string;
  gate: {
    passed: boolean;
    summary: {
      plans: { n: number; valid: number; valid_first_call: number; valid_after_repair: number; valid_after_layout_fix: number; failed: number };
      refusals: { n: number; refused_with_nearest: number };
    };
  };
  examples: RecordedExample[];
}

type Phase = "idle" | "uploading" | "planning" | "done" | "error";

interface LoadedStl {
  name: string;
  bytes: number;
  triangles: number;
  sizeMm: Vec3;
  file: File;
}

interface Shown {
  source: "recorded" | "live";
  result: PlanResultView;
  /** How many attempts are revealed (recorded plans replay their calls one by one). */
  revealed: number;
  caseId: string | null;
  spend: string | null;
}

const STATUS: Record<PlanResultView["status"], { stamp: string; tone: string; text: string }> = {
  valid: { stamp: "VALID PLAN", tone: "text-pass", text: "A plan the simulator can run." },
  refused: { stamp: "REFUSED", tone: "text-pending", text: "The simulator cannot do this use case." },
  invalid: { stamp: "NEEDS FIXES", tone: "text-fail", text: "The plan still breaks rules after one repair; fix it below." },
  failed: { stamp: "NO PLAN", tone: "text-fail", text: "The model call failed." },
};

const OUTCOME_TEXT: Record<string, string> = {
  valid: "valid plan",
  refused: "valid refusal",
  invalid: "refused by the checks",
  parse_error: "unreadable answer",
  call_failed: "call failed",
  spend_cap: "stopped by the spend cap",
};

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function readStl(file: File): Promise<LoadedStl> {
  const check = checkStlFile(file.name, file.size);
  if (!check.ok) throw new Error(check.problem ?? "unreadable file");
  const buffer = await file.arrayBuffer();
  const problem = checkStlBuffer(buffer);
  if (problem) throw new Error(problem);
  const { STLLoader } = await import("three/examples/jsm/loaders/STLLoader.js");
  const geometry = new STLLoader().parse(buffer);
  const positions = geometry.getAttribute("position");
  const extents = positions ? measurePositions(positions.array) : null;
  geometry.dispose();
  if (!extents) throw new Error("The file holds no readable triangles.");
  const unit = guessUnit(extents.size);
  return { name: file.name, bytes: file.size, triangles: binaryTriangleCount(buffer) ?? Math.floor((positions?.count ?? 0) / 3), sizeMm: toMillimetres(extents.size, unit), file };
}

/**
 * The Studio intake: describe a use case (three examples), optionally drop an STL, Plan, then
 * edit the step list. Without NEXT_PUBLIC_API_URL (the public static site) Plan replays the
 * recorded plan of an example, calls and all; with it, Plan calls the local API (passcode).
 */
export default function IntakeWorkbench({
  catalog,
  recorded,
  results = {},
}: {
  catalog: Catalog;
  recorded: RecordedPlans;
  /** Recorded use cases with a results page: case id to its route. */
  results?: Record<string, string>;
}) {
  const base = useMemo(() => apiBase(), []);
  const live = base !== null;
  // The hosted runner (Plan and run it live): null unless this build has NEXT_PUBLIC_RUNNER_URL.
  const runner = useMemo(() => runnerBase(), []);
  const heroes = recorded.examples.filter((e) => e.hero);
  const others = recorded.examples.filter((e) => !e.hero);

  const [text, setText] = useState("");
  const [stl, setStl] = useState<LoadedStl | null>(null);
  const [stlError, setStlError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [passcode, setPasscode] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [shown, setShown] = useState<Shown | null>(null);
  const [plan, setPlan] = useState<PlanArgs | null>(null);
  const [edited, setEdited] = useState(false);
  const [uploads, setUploads] = useState<string[]>([]);
  const [id, setId] = useState<string | null>(null);
  const [remote, setRemote] = useState<RemoteVerdict | string | null>(null);
  /** The plan in the editor came from the hosted planner (and has not been replaced since). */
  const [hosted, setHosted] = useState(false);
  /** How many plans have come back from the hosted planner; each one scrolls the editor into view (see the effect below). */
  const [livePlans, setLivePlans] = useState(0);
  const timers = useRef<number[]>([]);
  const abort = useRef<AbortController | null>(null);
  const editorRef = useRef<HTMLElement | null>(null);

  const clearTimers = () => {
    for (const t of timers.current) window.clearTimeout(t);
    timers.current = [];
  };
  useEffect(() => () => {
    clearTimers();
    abort.current?.abort();
  }, []);

  // Text handed over by the home page's question box (session storage), a ?q= link or an
  // ?example=<id> link: prefill it and plan it once, as if the visitor had pressed Plan.
  const [pendingPlan, setPendingPlan] = useState(false);
  useEffect(() => {
    let incoming: string | null = null;
    try {
      incoming = sessionStorage.getItem(PREFILL_KEY);
      if (incoming) sessionStorage.removeItem(PREFILL_KEY);
    } catch {
      incoming = null;
    }
    const params = new URLSearchParams(window.location.search);
    if (!incoming) incoming = params.get("q");
    const ex = params.get("example");
    if (!incoming && ex) incoming = recorded.examples.find((e) => e.case_id === ex || e.case_id.startsWith(`${ex}_`))?.text ?? null;
    if (incoming && incoming.trim()) {
      setText(incoming.trim().slice(0, 2000));
      setPendingPlan(true);
    }
  }, [recorded]);

  const verdict = useMemo(() => (plan ? validatePlan(catalog, plan, uploads) : null), [catalog, plan, uploads]);
  useEffect(() => {
    let alive = true;
    setId(null);
    if (verdict?.status === "valid") void taskId(catalog, verdict).then((v) => alive && setId(v));
    return () => {
      alive = false;
    };
  }, [catalog, verdict]);

  const load = useCallback((result: Pick<PlanResultView, "plan">, ups: string[]) => {
    setPlan(planFromReply(result));
    setUploads(ups);
    setEdited(false);
    setRemote(null);
    setHosted(false);
  }, []);

  /** A plan planned live on the hosted runner loads into the editor exactly as a replayed one does. */
  function onHostedPlan(reply: PlanReply) {
    clearTimers();
    setShown(null);
    setPhase("idle");
    setMessage(null);
    load(reply, []);
    setHosted(true);
    editorRef.current?.focus({ preventScroll: true });
    setLivePlans((n) => n + 1);
  }

  // The plan a hosted planner sent loads below the panel, off screen. Once the editor has rendered it, bring the editor
  // into view: smoothly, unless the visitor prefers reduced motion. Only plans from the hosted planner do this; replaying
  // a recorded plan behaves as it always did.
  useEffect(() => {
    if (livePlans === 0) return;
    editorRef.current?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }, [livePlans]);

  const pickExample = (caseId: string) => {
    const ex = recorded.examples.find((e) => e.case_id === caseId);
    if (!ex) return;
    setText(ex.text);
    setMessage(null);
  };

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setStlError(null);
    try {
      setStl(await readStl(file));
    } catch (err) {
      setStl(null);
      setStlError(`${file.name} could not be read as an STL: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const onDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    setDragging(false);
    void onFile(e.dataTransfer.files[0]);
  };

  function playRecorded(ex: RecordedExample) {
    clearTimers();
    setPhase("planning");
    setShown({ source: "recorded", result: ex, revealed: 0, caseId: ex.case_id, spend: null });
    setPlan(null);
    const finish = () => {
      setShown({ source: "recorded", result: ex, revealed: ex.attempts.length, caseId: ex.case_id, spend: null });
      load(ex, ex.uploads.map((u) => u.upload_id));
      setPhase("done");
      editorRef.current?.focus({ preventScroll: true });
    };
    if (prefersReducedMotion()) {
      finish();
      return;
    }
    ex.attempts.forEach((_, i) => {
      timers.current.push(
        window.setTimeout(() => setShown({ source: "recorded", result: ex, revealed: i + 1, caseId: ex.case_id, spend: null }), 700 * (i + 1)),
      );
    });
    timers.current.push(window.setTimeout(finish, 700 * ex.attempts.length + 400));
  }

  async function runLive() {
    if (!base) return;
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setMessage(null);
    setShown(null);
    try {
      let partId: string | null = null;
      if (stl) {
        setPhase("uploading");
        partId = await uploadStl(base, stl.file, stl.name, passcode, ac.signal);
      }
      setPhase("planning");
      const out = await planLive(base, text, partId, passcode, ac.signal);
      const spend = out.spend ? `$${out.spend.spent_usd.toFixed(4)} of the $${out.spend.session_cap_usd.toFixed(2)} plan cap` : null;
      setShown({ source: out.source, result: out.result, revealed: out.result.attempts.length, caseId: null, spend });
      load(out.result, out.result.uploads.map((u) => u.upload_id));
      setPhase("done");
    } catch (err) {
      if (ac.signal.aborted) return;
      setMessage(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }

  function onPlan() {
    const clean = text.trim();
    if (!clean) {
      setMessage("Describe the use case first, or pick an example.");
      return;
    }
    if (live) {
      void runLive();
      return;
    }
    const ex = recorded.examples.find((e) => e.text === clean);
    if (ex) {
      setMessage(null);
      playRecorded(ex);
      return;
    }
    setMessage(liveCopy("intakeMessage"));
  }

  useEffect(() => {
    if (!pendingPlan || !text) return;
    setPendingPlan(false);
    onPlan();
    // onPlan reads the text just set; it runs once per hand-over.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPlan, text]);

  /** A hand-built plan with one step of `prim` on a default object it acts on. */
  function startFrom(prim: string) {
    const kind = catalog.primitives.find((p) => p.kind === prim)?.objects[0];
    if (!kind) return;
    clearTimers();
    setPlan(addStep(catalog, addObject(catalog, emptyPlan(), kind), prim));
    setUploads([]);
    setEdited(true);
    setRemote(null);
  }

  async function checkRemote() {
    if (!base || !plan) return;
    try {
      setRemote(await validateRemote(base, plan, uploads.filter((u) => u.startsWith("part-"))));
    } catch (err) {
      setRemote(err instanceof Error ? err.message : String(err));
    }
  }

  const busy = phase === "uploading" || phase === "planning";
  const result = shown?.result ?? null;
  const allShown = result ? shown!.revealed >= result.attempts.length : false;
  const uploadExample = others.find((e) => e.uploads.length);
  const source = exportSource({ edited, hosted, shownSource: shown?.source ?? null });
  const runDoc = plan && verdict?.status === "valid" ? runDocument(catalog, plan, id, source) : null;
  const exportText = runDoc ? JSON.stringify(runDoc, null, 2) : null;
  const runLimit = runDoc ? hostedLimit(runDoc, uploads) : null;
  const refusal = plan?.refusal ? plan : null;

  return (
    <>
      <div className="mt-10 grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <section aria-labelledby="usecase-title">
          <h2 id="usecase-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">01</span> Describe the use case
          </h2>
          <p className="mt-3 text-sm text-ink-2">Start from an example, or write your own in plain words: the objects, what has to happen to each, and any sizes, weights or limits you know.</p>
          <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3" role="group" aria-label="Example use cases">
            {heroes.map((e) => (
              <button
                key={e.case_id}
                type="button"
                onClick={() => pickExample(e.case_id)}
                aria-pressed={text === e.text}
                className={`rounded-lg border px-3 py-2 text-left text-sm transition-colors ${text === e.text ? "border-accent bg-accent-soft" : "border-rule-strong hover:bg-paper-2"}`}
              >
                <span className="block text-xs text-ink-3">{e.setting.charAt(0).toUpperCase() + e.setting.slice(1)}</span>
                <span className="block text-ink">{e.title}</span>
              </button>
            ))}
          </div>
          {others.length ? (
            <label className="mt-3 grid gap-1 text-sm">
              <span className="text-xs text-ink-3">More recorded descriptions (the rest of the planner&apos;s test set)</span>
              <select
                className="rounded-md border border-rule-strong bg-surface px-2 py-1.5 text-ink"
                value={recorded.examples.find((e) => e.text === text && !e.hero)?.case_id ?? ""}
                onChange={(e) => e.target.value && pickExample(e.target.value)}
              >
                <option value="">Choose one...</option>
                {others.map((e) => (
                  <option key={e.case_id} value={e.case_id}>
                    {e.setting}: {e.title}
                    {e.expect === "refusal" ? " (not simulated)" : ""}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <label className="mt-4 grid gap-1">
            <span className="text-xs text-ink-3">Use case</span>
            <textarea
              aria-label="Use case"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              maxLength={2000}
              placeholder="For example: open the hinged lid of the tote, put the carton from the bench in it, close the lid."
              className="w-full rounded-lg border border-rule-strong bg-surface px-3 py-2 text-ink focus:border-accent/70"
            />
          </label>

          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`mt-4 flex cursor-pointer flex-col gap-1 rounded-lg border border-dashed px-4 py-3 text-sm focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent ${
              dragging ? "border-accent bg-accent-soft" : "border-rule-strong bg-paper-2/40 hover:bg-paper-2/70"
            }`}
          >
            <input
              type="file"
              accept=".stl,model/stl,application/sla"
              className="sr-only"
              onChange={(e) => {
                void onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
            <span className="text-ink">{stl ? "Drop another STL, or choose a file" : "Optional: drop the STL of a part in the use case"}</span>
            <span className="text-xs text-ink-3">
              {live
                ? "On Plan it goes through the part intake on your local API, and the plan must use it."
                : "It stays in your browser: this static site cannot plan with it."}
            </span>
          </label>
          {stlError ? (
            <p role="alert" className="mt-2 border-l-2 border-fail pl-3 text-sm text-fail">
              {stlError}
            </p>
          ) : null}
          {stl ? (
            <p className="num mt-2 text-xs text-ink-2">
              {stl.name} · {stl.sizeMm.map((v) => v.toFixed(1)).join(" x ")} mm · {formatInt(stl.triangles)} triangles · {formatBytes(stl.bytes)}{" "}
              <button type="button" className="link ml-1" onClick={() => setStl(null)}>
                Remove
              </button>
            </p>
          ) : null}
          {stl && !live && uploadExample ? (
            <p className="mt-2 border-l-2 border-pending pl-3 text-sm text-ink-2">
              The recorded example{" "}
              <button type="button" className="link" onClick={() => pickExample(uploadExample.case_id)}>
                {uploadExample.title.toLowerCase()}
              </button>{" "}
              shows how the planner uses an uploaded part.
            </p>
          ) : null}
        </section>

        <section aria-labelledby="plan-title">
          <h2 id="plan-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">02</span> Plan it
          </h2>
          <div className="panel mt-4 p-4">
            <p className="flex flex-wrap items-center gap-2 text-sm">
              <span className={`stamp ${live ? "text-pass" : "text-pending"}`}>{live ? "LIVE" : "RECORDED PLANS"}</span>
              <span className="text-ink-2">
                {live ? (
                  <>
                    Local API at <span className="num break-all text-ink">{base}</span>
                  </>
                ) : (
                  liveCopy("intakeBox")
                )}
              </span>
            </p>
            <p className="mt-3 text-sm text-ink-2">
              The Task Engineer, NVIDIA Nemotron 3 Super, answers with one forced tool call: the objects as parametric templates and
              the steps as primitive skills with bounded parameters and success criteria. Code checks every rule; a plan that breaks
              one gets a single repair call that lists the problems. Product mode, not part of the pre-registered result.
            </p>
            {live ? (
              <label className="mt-4 grid gap-1 text-sm">
                <span className="text-xs text-ink-3">Passcode</span>
                <input
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  value={passcode}
                  onChange={(e) => setPasscode(e.target.value)}
                  disabled={busy}
                  className="rounded-md border border-rule-strong bg-surface px-2 py-1.5 text-ink"
                />
                <span className="text-xs text-ink-3">A new description calls a paid model, so the API asks for it. It goes in a request header, never in a URL.</span>
              </label>
            ) : null}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button type="button" onClick={onPlan} disabled={busy} className="btn btn-primary" data-plan-button>
                Plan
              </button>
              <button
                type="button"
                className="link text-sm"
                onClick={() => {
                  clearTimers();
                  setShown(null);
                  setPhase("idle");
                  setPlan(emptyPlan());
                  setUploads([]);
                  setEdited(true);
                  setRemote(null);
                }}
              >
                Build a plan by hand
              </button>
            </div>
            <p className="mt-3 min-h-5 text-sm text-ink-2" role="status">
              {phase === "uploading" ? "Sending the part through the intake..." : phase === "planning" ? "Nemotron is planning..." : ""}
            </p>
            {message ? (
              <p role="alert" className="border-l-2 border-pending pl-3 text-sm text-ink-2">
                {message}
              </p>
            ) : null}
          </div>

          {result ? (
            <div className="mt-5 grid gap-3" aria-label="Planner calls">
              <ol className="grid gap-2">
                {result.attempts.slice(0, shown!.revealed).map((a, i) => (
                  <li key={i} className="border-l-2 border-rule-strong pl-3 text-sm">
                    <p className="flex flex-wrap items-baseline gap-x-2">
                      <span className="text-ink">
                        Call {i + 1} ({a.stage === "main" ? "plan" : "repair"}): {OUTCOME_TEXT[a.outcome] ?? a.outcome}
                      </span>
                      <span className="num text-xs text-ink-3">
                        {formatInt(a.prompt_tokens)} in, {formatInt(a.completion_tokens)} out · {a.latency_s.toFixed(1)} s · ${a.cost_usd.toFixed(4)}
                      </span>
                    </p>
                    {a.problems.length && a.outcome === "invalid" ? (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-xs text-ink-3">
                          {a.problems.length} problem{a.problems.length === 1 ? "" : "s"} {a.stage === "main" ? "sent back" : "left after the repair"}
                        </summary>
                        <ul className="mt-1 grid gap-0.5 text-xs text-ink-2">
                          {a.problems.map((p) => (
                            <li key={p} className="break-words">
                              {p}
                            </li>
                          ))}
                        </ul>
                      </details>
                    ) : null}
                  </li>
                ))}
                {!allShown ? <li className="pl-3 text-sm text-ink-3">Waiting for call {shown!.revealed + 1}...</li> : null}
              </ol>
              {allShown ? (
                <div className="border-l-2 border-accent pl-3">
                  <p className="flex flex-wrap items-center gap-2 text-sm">
                    <span className={`stamp ${STATUS[result.status].tone}`}>{STATUS[result.status].stamp}</span>
                    <span className="text-ink">{STATUS[result.status].text}</span>
                  </p>
                  <p className="mt-1 text-xs text-ink-3">
                    {result.first_try
                      ? "Valid on the first call."
                      : result.layout_fixed
                        ? "Valid after one repair call and the layout fix (code moved objects; see the assumptions)."
                        : result.repaired
                          ? "Valid after one repair call."
                          : ""}{" "}
                    {shown!.source === "recorded"
                      ? `Recorded live on ${(recorded.examples.find((e) => e.case_id === shown!.caseId)?.recorded_at || recorded.recorded_at).slice(0, 10)}; replayed, no model called.`
                      : "Planned live."}{" "}
                    {shown!.spend ?? ""}
                  </p>
                  {result.layout_moves?.length ? (
                    <ul className="mt-1 grid gap-0.5 text-xs text-ink-2" aria-label="Layout fix">
                      {result.layout_moves.map((m) => (
                        <li key={m} className="break-words">
                          Layout fix: {m}.
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  {result.notes.length ? <p className="mt-1 text-xs text-ink-3">{result.notes.join(" ")}</p> : null}
                  {shown?.caseId && results[shown.caseId] ? (
                    <p className="mt-3 flex flex-wrap items-center gap-3">
                      <Link href={results[shown.caseId]!} className="btn btn-quiet" data-recorded-results>
                        See how every tool did on this task
                      </Link>
                      <span className="text-xs text-ink-3">Recorded test results: the screen, the recommendation and the replays.</span>
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </section>
      </div>

      {runner ? <HostedRunner base={runner} description={text.trim()} runDoc={runDoc} runLimit={runLimit} hasPlan={plan !== null && !refusal} hasStl={stl !== null} onPlanned={onHostedPlan} /> : null}

      {/* A runner build scrolls here when a live plan loads, and clears the sticky header as the panel above does; the build without the runner keeps its markup as it was. */}
      <section ref={editorRef} tabIndex={-1} className={runner ? "mt-14 scroll-mt-20 outline-none" : "mt-14 outline-none"} aria-labelledby="edit-title">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 id="edit-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">03</span> Check and edit the steps
          </h2>
          {verdict ? (
            <p className="text-sm" role="status" aria-live="polite">
              {verdict.status === "valid" ? (
                <span className="text-pass">
                  Plan is valid: {countOf(plan!.objects.length, "object")}, {countOf(plan!.steps.length, "step")}{id ? <span className="num ml-1 text-xs text-ink-3">task {id.slice(0, 12)}</span> : null}
                </span>
              ) : verdict.status === "refused" ? (
                <span className="text-pending">A valid refusal</span>
              ) : (
                <span className="text-fail">
                  {verdict.problems.length} problem{verdict.problems.length === 1 ? "" : "s"} to fix
                </span>
              )}
            </p>
          ) : null}
        </div>

        {refusal ? (
          <div className="mt-4 rounded-lg border border-pending p-4">
            <p className="text-sm text-ink">
              <span className="stamp mr-2 text-pending">NOT SIMULATED</span>
              {refusal.refusal}
            </p>
            {refusal.unsupported.length ? (
              <ul className="mt-3 grid gap-1 text-sm text-ink-2">
                {refusal.unsupported.map((u) => (
                  <li key={u.what}>
                    <span className="text-ink">{u.what}</span>: {u.reason}{" "}
                    {u.nearest_primitive ? (
                      <span>
                        Nearest primitive: <span className="stamp text-accent">{u.nearest_primitive.replace("_", " ").toUpperCase()}</span>{" "}
                        <button type="button" className="link" onClick={() => startFrom(u.nearest_primitive as string)}>
                          Start a plan with it
                        </button>
                      </span>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {plan && !refusal ? (
          <div className="mt-4">
            {plan.assumptions.length || plan.clarifications.length || plan.unsupported.length ? (
              <div className="mb-5 grid gap-3 text-sm sm:grid-cols-2">
                {plan.assumptions.length ? (
                  <div>
                    <h3 className="text-xs text-ink-3">Assumptions</h3>
                    <ul className="mt-1 list-disc pl-5 text-ink-2">
                      {plan.assumptions.map((a) => (
                        <li key={a}>{a}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {plan.clarifications.length ? (
                  <div>
                    <h3 className="text-xs text-ink-3">Questions (answer assumed until you change the plan)</h3>
                    <ul className="mt-1 grid gap-1 text-ink-2">
                      {plan.clarifications.map((c) => (
                        <li key={c.question}>
                          {c.question} <span className="text-ink-3">Assumed: {c.assumed_answer}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {plan.unsupported.length ? (
                  <div>
                    <h3 className="text-xs text-ink-3">Not simulated</h3>
                    <ul className="mt-1 grid gap-1 text-ink-2">
                      {plan.unsupported.map((u) => (
                        <li key={u.what}>
                          {u.what}: {u.reason}
                          {u.nearest_primitive ? ` Nearest primitive: ${u.nearest_primitive}.` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : null}
            <StepEditor
              catalog={catalog}
              plan={plan}
              verdict={verdict!}
              onChange={(p) => {
                setPlan(p);
                setEdited(true);
                setRemote(null);
              }}
            />
          </div>
        ) : null}

        {!plan ? (
          <p className="mt-4 rounded-lg border border-dashed border-rule-strong p-6 text-sm text-ink-3">Plan a use case, or build one by hand, to edit its steps here. Nothing runs until you export the plan.</p>
        ) : null}

        {plan && !refusal ? (
          <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-rule pt-4">
            <button
              type="button"
              disabled={!exportText}
              onClick={() => exportText && download(`taskspec-${(id ?? "plan").slice(0, 12)}.json`, `${exportText}\n`)}
              className="btn btn-primary"
            >
              Download TaskSpec JSON
            </button>
            <button
              type="button"
              disabled={!exportText}
              onClick={() => exportText && void navigator.clipboard?.writeText(exportText)}
              className="btn btn-quiet"
            >
              Copy JSON
            </button>
            {live ? (
              <button type="button" onClick={() => void checkRemote()} className="link text-sm">
                Check with the API
              </button>
            ) : null}
            <span className="text-xs text-ink-3">
              {exportText ? `Contract ${catalog.contract}, source ${source}. ${catalog.label}.` : "Fix every problem to export the plan."}
            </span>
            {remote ? (
              <p className="w-full text-sm text-ink-2">
                {typeof remote === "string" ? remote : `API: ${remote.status}${remote.problems.length ? `; ${remote.problems.join("; ")}` : ""}`}
              </p>
            ) : null}
          </div>
        ) : null}
      </section>
    </>
  );
}
