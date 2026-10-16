"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import type * as THREE from "three";

import { AgentTimeline, SessionMeter, type TimelineLinks } from "@/components/agent/AgentTimeline";
import StlPreviewLazy from "@/components/part/StlPreviewLazy";
import { apiBase, getHealth, type Health, startSession, stopSession, streamEvents, uploadPart } from "@/lib/api";
import { formatBytes, formatInt, formatOffset } from "@/lib/format";
import { dataUrl } from "@/lib/paths";
import {
  type AgentEvent,
  applyEvent,
  buildTimeline,
  emptyTimeline,
  normalizeEvent,
  parseRecordedSession,
  playbackTimes,
  playedCount,
  rawTime,
  type RecordedSession,
  type SessionSummary,
  type Timeline,
  TOOL_TITLES,
} from "@/lib/session";
import {
  binaryTriangleCount,
  checkStlBuffer,
  checkStlFile,
  encodeBinaryStl,
  type Extents,
  fitAgainstOpening,
  guessUnit,
  type LengthUnit,
  measurePositions,
  toMillimetres,
  UNIT_LABELS,
  UNIT_SCALE,
} from "@/lib/stl";

interface LoadedPart {
  name: string;
  bytes: number;
  geometry: THREE.BufferGeometry;
  extents: Extents;
  triangles: number;
  /** The user's own file, or the example part of a recorded session. */
  source: "upload" | "session";
  file: File | null;
}

type Phase = "idle" | "uploading" | "starting" | "streaming" | "stopping" | "playing" | "paused" | "done" | "error";

const PHASE_TEXT: Record<Phase, string> = {
  idle: "",
  uploading: "Uploading the part...",
  starting: "Starting the agent...",
  streaming: "The agent is working. Cards appear as each tool answers.",
  stopping: "Stopping after the current step...",
  playing: "Playing the demo session.",
  paused: "Paused.",
  done: "Session finished.",
  error: "The run stopped.",
};

async function parseStl(buffer: ArrayBuffer): Promise<{ geometry: THREE.BufferGeometry; extents: Extents; triangles: number }> {
  const problem = checkStlBuffer(buffer);
  if (problem) throw new Error(problem);
  const { STLLoader } = await import("three/examples/jsm/loaders/STLLoader.js");
  const geometry = new STLLoader().parse(buffer);
  const positions = geometry.getAttribute("position");
  if (!positions || positions.count === 0) throw new Error("The file holds no triangles.");
  const extents = measurePositions(positions.array);
  if (!extents) throw new Error("The file's coordinates are not finite numbers.");
  return { geometry, extents, triangles: binaryTriangleCount(buffer) ?? Math.floor(positions.count / 3) };
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

const VERDICT_STYLE = {
  fits: { tone: "text-pass", mark: "M1.5 6.5l3 3 6-7" },
  tight: { tone: "text-pending", mark: "M6 1.5v6M6 9.5v1" },
  too_wide: { tone: "text-fail", mark: "M2 2l8 8M10 2l-8 8" },
  too_small: { tone: "text-pending", mark: "M6 1.5v6M6 9.5v1" },
  empty: { tone: "text-fail", mark: "M2 2l8 8M10 2l-8 8" },
} as const;

function SizeReadout({ part, unit, setUnit, openingMm }: { part: LoadedPart; unit: LengthUnit; setUnit: (u: LengthUnit) => void; openingMm: number }) {
  const sizeMm = toMillimetres(part.extents.size, unit);
  const fit = fitAgainstOpening(sizeMm, openingMm);
  const style = VERDICT_STYLE[fit.verdict];
  const frac = Math.min(fit.narrowest_mm / openingMm, 1.25) / 1.25;
  return (
    <div className="mt-4 grid gap-3">
      <dl className="grid grid-cols-3 gap-px border border-rule bg-rule text-sm">
        {(["X", "Y", "Z"] as const).map((axis, i) => (
          <div key={axis} className="bg-paper px-3 py-2">
            <dt className="text-xs text-ink-3">{axis}, file axes</dt>
            <dd className="num text-lg text-ink">
              {sizeMm[i]!.toFixed(1)}
              <span className="ml-1 text-xs text-ink-3">mm</span>
            </dd>
          </div>
        ))}
      </dl>
      <div>
        <div className="flex items-center gap-2 text-xs text-ink-3" aria-hidden="true">
          <span className="w-16">Narrowest</span>
          <div className="relative h-3 flex-1 border-x border-rule-strong">
            <div className="absolute inset-x-0 top-1/2 h-px bg-rule-strong" />
            <div className="absolute inset-y-0 left-0 bg-ink-3/40" style={{ width: `${frac * 100}%` }} />
            <div className="absolute -top-1 h-5 w-0.5 bg-accent" style={{ left: `${(1 / 1.25) * 100}%` }} title={`${openingMm} mm opening`} />
          </div>
          <span className="num w-16 text-right">{openingMm} mm</span>
        </div>
        <p className={`mt-2 flex items-start gap-2 text-sm ${style.tone}`}>
          <svg viewBox="0 0 12 12" className="mt-1 size-3 shrink-0" aria-hidden="true">
            <path d={style.mark} fill="none" stroke="currentColor" strokeWidth="2" />
          </svg>
          <span>{fit.message}</span>
        </p>
        <p className="mt-1 text-xs text-ink-3">
          Bounding box in the file&apos;s own axes. The agent finds every way the part can rest and checks the width each one
          presents to the jaws.
        </p>
      </div>
      <fieldset className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
        <legend className="sr-only">Units of the STL file</legend>
        <span className="text-xs text-ink-3">File units</span>
        {(Object.keys(UNIT_LABELS) as LengthUnit[]).map((u) => (
          <label key={u} className="inline-flex cursor-pointer items-center gap-1.5">
            <input type="radio" name="stl-unit" value={u} checked={unit === u} onChange={() => setUnit(u)} className="accent-[var(--accent)]" />
            <span className={unit === u ? "text-ink" : "text-ink-2"}>{UNIT_LABELS[u]}</span>
          </label>
        ))}
      </fieldset>
      <p className="num text-xs text-ink-3">
        {part.name} · {formatBytes(part.bytes)} · {formatInt(part.triangles)} triangles
        {part.source === "session" ? " · example part of the demo session" : " · read in your browser"}
      </p>
    </div>
  );
}

/**
 * "Bring your part": drop an STL, see it against the jaw opening, then run the Grasp
 * Engineer. With NEXT_PUBLIC_API_URL set it talks to the local API and streams the
 * agent's events; without it (the public static site) Run plays a published demo session:
 * a recorded agent run, or an illustrative one, which the page then calls illustrative
 * everywhere (never "recorded").
 */
export default function PartWorkbench({
  sessions,
  openingMm,
  links,
}: {
  sessions: SessionSummary[];
  openingMm: number;
  links: TimelineLinks;
}) {
  const base = useMemo(() => apiBase(), []);
  const live = base !== null;

  const [part, setPart] = useState<LoadedPart | null>(null);
  const [unit, setUnit] = useState<LengthUnit>("mm");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [passcode, setPasscode] = useState("");

  const [phase, setPhase] = useState<Phase>("idle");
  const [runError, setRunError] = useState<string | null>(null);
  const [liveTimeline, setLiveTimeline] = useState<Timeline | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(sessions[0]?.id ?? null);
  const [recorded, setRecorded] = useState<RecordedSession | null>(null);
  const [clock, setClock] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    if (!base) return;
    const ac = new AbortController();
    getHealth(base, ac.signal)
      .then(setHealth)
      .catch((err: unknown) => {
        if (!ac.signal.aborted) setHealthError(err instanceof Error ? err.message : String(err));
      });
    return () => ac.abort();
  }, [base]);
  const timelineRef = useRef<HTMLElement | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => () => part?.geometry.dispose(), [part]);

  const loadBuffer = useCallback(async (buffer: ArrayBuffer, meta: Omit<LoadedPart, "geometry" | "extents" | "triangles">, unitHint?: LengthUnit) => {
    setLoading(true);
    setLoadError(null);
    try {
      const parsed = await parseStl(buffer);
      setPart({ ...meta, ...parsed });
      setUnit(unitHint ?? guessUnit(parsed.extents.size));
    } catch (err) {
      setLoadError(`${meta.name} could not be read as an STL: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const acceptFile = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      const check = checkStlFile(file.name, file.size);
      if (!check.ok) {
        setLoadError(check.problem);
        return;
      }
      await loadBuffer(await file.arrayBuffer(), { name: file.name, bytes: file.size, source: "upload", file });
    },
    [loadBuffer],
  );

  const onDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    setDragging(false);
    void acceptFile(e.dataTransfer.files[0]);
  };

  // ---------------------------------------------------------------- recorded playback

  const times = useMemo(() => (recorded ? playbackTimes(recorded.events) : []), [recorded]);
  const played = recorded ? playedCount(times, clock) : 0;
  const recordedTimeline = useMemo(
    () => (recorded ? buildTimeline(recorded.events.slice(0, played)) : null),
    [recorded, played],
  );

  useEffect(() => {
    if (phase !== "playing" || !recorded) return;
    let last = performance.now();
    const id = window.setInterval(() => {
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      setClock((c) => c + dt);
    }, 100);
    return () => window.clearInterval(id);
  }, [phase, recorded]);

  useEffect(() => {
    if (phase === "playing" && recorded && played >= recorded.events.length) setPhase("done");
  }, [phase, recorded, played]);

  async function runRecorded(stepByStep: boolean) {
    const summary = sessions.find((s) => s.id === selectedId) ?? sessions[0];
    if (!summary) return;
    setRunError(null);
    try {
      let session = recorded?.id === summary.id ? recorded : null;
      if (!session) {
        const res = await fetch(dataUrl(`sessions/${summary.id}.json`));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        session = parseRecordedSession(await res.json(), `sessions/${summary.id}.json`);
        setRecorded(session);
      }
      if (!part && session.part.stl) {
        const res = await fetch(dataUrl(session.part.stl));
        if (res.ok) {
          const buf = await res.arrayBuffer();
          await loadBuffer(buf, { name: session.part.name, bytes: buf.byteLength, source: "session", file: null }, session.part.units);
        }
      }
      const instant = !stepByStep && prefersReducedMotion();
      setClock(instant ? Number.POSITIVE_INFINITY : 0);
      setPhase(instant ? "done" : "playing");
      timelineRef.current?.focus({ preventScroll: true });
    } catch (err) {
      setRunError(`The demo session could not be loaded (${err instanceof Error ? err.message : String(err)}).`);
      setPhase("error");
    }
  }

  // ---------------------------------------------------------------- live run

  async function runLive() {
    if (!base || !part?.file) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    sessionIdRef.current = null;
    setRunError(null);
    setLiveTimeline(emptyTimeline());
    try {
      setPhase("uploading");
      // The API reads millimetres or metres; other units are re-encoded in millimetres here.
      const native = unit === "mm" || unit === "m";
      const positions = part.geometry.getAttribute("position").array;
      const body = native ? part.file : encodeBinaryStl(positions, UNIT_SCALE[unit]);
      const uploaded = await uploadPart(base, body, part.name, native ? unit : "mm", ac.signal);
      setPhase("starting");
      const sessionId = await startSession(base, uploaded.part_id, passcode, ac.signal);
      sessionIdRef.current = sessionId;
      setPhase("streaming");
      const t0 = performance.now();
      let firstAbs: number | null = null;
      await streamEvents(
        base,
        sessionId,
        (payload, sseEvent) => {
          const { rel, abs } = rawTime(payload);
          let t: number;
          if (rel !== null) t = rel;
          else if (abs !== null) {
            firstAbs ??= abs;
            t = abs - firstAbs;
          } else t = (performance.now() - t0) / 1000;
          const ev: AgentEvent | null = normalizeEvent(payload, Math.max(0, t), sseEvent);
          if (ev) setLiveTimeline((tl) => applyEvent(tl ?? emptyTimeline(), ev));
        },
        ac.signal,
      );
      setPhase("done");
    } catch (err) {
      if (ac.signal.aborted) {
        setPhase("idle");
        return;
      }
      setRunError(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }

  async function stopLive() {
    const id = sessionIdRef.current;
    if (!base || !id) {
      abortRef.current?.abort();
      return;
    }
    setPhase("stopping");
    try {
      await stopSession(base, id);
    } catch {
      abortRef.current?.abort();
    }
  }

  const timeline = live ? liveTimeline : recordedTimeline;
  const illustrative = !live && recorded?.kind === "illustrative";
  const busy = phase === "uploading" || phase === "starting" || phase === "streaming" || phase === "stopping";
  const selected = sessions.find((s) => s.id === selectedId) ?? sessions[0] ?? null;
  // A hand-written session must never be called "recorded", not even before it plays.
  const demoIllustrative = selected?.kind === "illustrative";
  const fitVerdict = part ? fitAgainstOpening(toMillimetres(part.extents.size, unit), openingMm).verdict : null;
  const canRunLive = live && part?.source === "upload" && fitVerdict !== "empty" && !busy;
  const lastStep = timeline?.steps.at(-1);
  const announcement = lastStep
    ? lastStep.kind === "message" || lastStep.kind === "error"
      ? (lastStep.text ?? "")
      : `${TOOL_TITLES[lastStep.kind]}: ${lastStep.status === "running" ? "working" : lastStep.status === "ok" ? "done" : "failed"}.`
    : "";

  return (
    <>
      <div className="mt-10 grid grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <section aria-labelledby="intake-title">
          <h2 id="intake-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">01</span> Your part
          </h2>
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`mt-4 flex cursor-pointer flex-col items-center justify-center gap-2 border border-dashed px-4 text-center transition-colors focus-within:outline focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent ${
              part ? "py-4" : "aspect-[4/3] py-10"
            } ${dragging ? "border-accent bg-accent-soft" : "border-rule-strong bg-paper-2/40 hover:bg-paper-2/70"}`}
          >
            <input
              type="file"
              accept=".stl,model/stl,application/sla"
              className="sr-only"
              onChange={(e) => {
                void acceptFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
            {!part ? (
              <svg viewBox="0 0 48 48" className="size-10 text-ink-3" aria-hidden="true">
                <path d="M24 31V9m0 0l-8 8m8-8l8 8M8 30v9h32v-9" fill="none" stroke="currentColor" strokeWidth="2" />
              </svg>
            ) : null}
            <span className="text-ink">
              {loading ? "Reading the file..." : part ? "Drop another STL, or choose a file" : "Drop an STL here, or choose a file"}
            </span>
            <span className="text-xs text-ink-3">
              Millimetres, up to 50 MB.{" "}
              {live
                ? "It is sent to your local API only when you press Run (the API takes up to 20 MB unless RTT_AGENT_MAX_UPLOAD_MB says otherwise)."
                : "It stays in your browser: this static site has no server."}
            </span>
          </label>
          {loadError ? (
            <p role="alert" className="mt-3 border-l-2 border-fail pl-3 text-sm text-fail">
              {loadError}
            </p>
          ) : null}
          {part ? (
            <div className="mt-5">
              <StlPreviewLazy
                geometry={part.geometry}
                mmPerUnit={toMillimetres([1, 1, 1], unit)[0]}
                openingMm={openingMm}
                label={`3D preview of ${part.name} between the open jaws`}
              />
              <SizeReadout part={part} unit={unit} setUnit={setUnit} openingMm={openingMm} />
            </div>
          ) : null}
        </section>

        <section aria-labelledby="run-title">
          <h2 id="run-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">02</span> Run the Grasp Engineer
          </h2>
          <div className="mt-4 border border-rule bg-paper/80 p-4">
            <p className="flex flex-wrap items-center gap-2 text-sm">
              <span className={`stamp ${live ? "text-pass" : "text-pending"}`}>{live ? "LIVE" : demoIllustrative ? "ILLUSTRATIVE DEMO" : "RECORDED DEMO"}</span>
              <span className="text-ink-2">
                {live ? (
                  <>
                    Local API at <span className="num break-all text-ink">{base}</span>
                    {health ? (
                      <span className="num ml-1 text-xs text-ink-3">
                        ({health.mode ?? "mode unknown"}
                        {health.backend ? `; ${health.backend === "fake" ? "synthetic physics" : `${health.backend} physics`}` : ""}; targets{" "}
                        {health.targets.join(", ") || "none"}
                        {health.passcode_configured === false ? "; no passcode set, so nothing that spends will run" : ""})
                      </span>
                    ) : healthError ? (
                      <span className="ml-1 text-fail">is not answering ({healthError})</span>
                    ) : null}
                  </>
                ) : (
                  demoIllustrative
                    ? "This static site has no backend, so Run plays an illustrative session: written by hand in the agent's event format, not a recorded run."
                    : "This static site has no backend, so Run plays a recorded agent session."
                )}
              </span>
            </p>
            <p className="mt-3 text-sm text-ink-2">
              An NVIDIA Nemotron 3 Super agent works through its tools: it inspects the part, proposes finger designs,
              runs them as simulation jobs (on Nebius when the server enables it, otherwise locally), stress-tests the
              best one and exports a print pack. Product mode, not part of the pre-registered result.
            </p>

            {!live && sessions.length > 1 ? (
              <label className="mt-4 grid gap-1 text-sm">
                <span className="text-xs text-ink-3">Demo session</span>
                <select
                  value={selectedId ?? ""}
                  onChange={(e) => {
                    setSelectedId(e.target.value);
                    setRecorded(null);
                    setPhase("idle");
                    setClock(0);
                  }}
                  className="border border-rule-strong bg-paper px-2 py-1.5 text-ink"
                >
                  {sessions.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.title}
                      {s.kind === "illustrative" ? " (illustrative)" : ""}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <label className="mt-4 grid gap-1 text-sm">
              <span className="text-xs text-ink-3">Passcode</span>
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={passcode}
                onChange={(e) => setPasscode(e.target.value)}
                disabled={!live || busy}
                aria-describedby="passcode-hint"
                className="border border-rule-strong bg-paper px-2 py-1.5 text-ink disabled:cursor-not-allowed disabled:bg-paper-2/60 disabled:text-ink-3"
              />
              <span id="passcode-hint" className="text-xs text-ink-3">
                {live
                  ? "Live runs spend real compute, so the API asks for a passcode. It is sent in a request header, never in a URL."
                  : "Needed only when the site is connected to a live backend."}
              </span>
            </label>

            <div className="mt-5 flex flex-wrap items-center gap-3">
              {live ? (
                <>
                  <button
                    type="button"
                    onClick={() => void runLive()}
                    disabled={!canRunLive}
                    className="bg-accent px-5 py-2.5 text-accent-ink disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Run
                  </button>
                  {busy ? (
                    <button
                      type="button"
                      onClick={() => void stopLive()}
                      disabled={phase === "stopping"}
                      className="border border-rule-strong px-4 py-2.5 text-ink hover:bg-paper-2 disabled:opacity-50"
                    >
                      Stop
                    </button>
                  ) : null}
                </>
              ) : (
                <>
                  <button
                    type="button"
                    onClick={() => void runRecorded(false)}
                    disabled={!selected || phase === "playing"}
                    className="bg-accent px-5 py-2.5 text-accent-ink disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {phase === "done" || phase === "paused" ? "Run again" : "Run the demo session"}
                  </button>
                  {phase === "playing" ? (
                    <button type="button" onClick={() => setPhase("paused")} className="border border-rule-strong px-4 py-2.5 text-ink hover:bg-paper-2">
                      Pause
                    </button>
                  ) : null}
                  {phase === "paused" ? (
                    <button type="button" onClick={() => setPhase("playing")} className="border border-rule-strong px-4 py-2.5 text-ink hover:bg-paper-2">
                      Resume
                    </button>
                  ) : null}
                  {(phase === "playing" || phase === "paused") && recorded ? (
                    <button
                      type="button"
                      onClick={() => {
                        setClock(Number.POSITIVE_INFINITY);
                        setPhase("done");
                      }}
                      className="link text-sm"
                    >
                      Skip to the end
                    </button>
                  ) : null}
                  {phase === "done" && recorded ? (
                    <button type="button" onClick={() => void runRecorded(true)} className="link text-sm">
                      Replay step by step
                    </button>
                  ) : null}
                </>
              )}
            </div>
            {live && !part ? <p className="mt-3 text-sm text-ink-3">Drop an STL first; Run sends it to the agent.</p> : null}
            {live && part?.source === "session" ? <p className="mt-3 text-sm text-ink-3">Drop your own STL to run it live.</p> : null}
            {!live && part?.source === "upload" ? (
              <p className="mt-3 border-l-2 border-pending pl-3 text-sm text-ink-2">
                The demo session is for {selected ? <span className="num">{selected.part.name}</span> : "its own part"}, not
                your file. To run your part, start the local API and build the site with{" "}
                <span className="num">NEXT_PUBLIC_API_URL</span> set (see the project README).
              </p>
            ) : null}
            <p className="mt-3 min-h-5 text-sm text-ink-2" role="status">
              {PHASE_TEXT[phase]}
            </p>
            {runError ? (
              <p role="alert" className="mt-1 border-l-2 border-fail pl-3 text-sm text-fail">
                {runError}
              </p>
            ) : null}
          </div>
        </section>
      </div>

      <section ref={timelineRef} tabIndex={-1} className="mt-14 outline-none" aria-labelledby="session-title">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 id="session-title" className="heading flex items-baseline gap-3 text-xl">
            <span className="num text-sm text-accent">03</span> Agent session
          </h2>
          {!live && recorded ? (
            <span className="num text-xs text-ink-3">
              {recorded.title} · {played} of {recorded.events.length} events
              {recorded.recorded_at ? ` · recorded ${recorded.recorded_at.slice(0, 10)}` : ""}
            </span>
          ) : null}
        </div>
        <p className="sr-only" aria-live="polite">
          {announcement}
        </p>
        {timeline && timeline.steps.length > 0 ? (
          <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-5">
            {timeline.offline ? (
              <p className="flex flex-wrap items-center gap-3 border border-pending px-3 py-2 text-sm text-ink-2">
                <span className="stamp -rotate-2 text-pending">OFFLINE</span>
                <span>
                  <span className="font-semibold text-ink">Scripted agent, no model calls.</span> The agent&apos;s choices are
                  scripted for testing; the tools, caps and recording are the real ones.
                </span>
              </p>
            ) : null}
            {illustrative && recorded ? (
              <p className="flex flex-wrap items-center gap-3 border border-pending px-3 py-2 text-sm text-ink-2">
                <span className="stamp -rotate-2 text-pending">ILLUSTRATIVE</span>
                <span>
                  <span className="font-semibold text-ink">Illustrative session, not a recorded run.</span> {recorded.note}
                </span>
              </p>
            ) : null}
            <SessionMeter timeline={timeline} />
            <AgentTimeline
              timeline={timeline}
              illustrative={illustrative}
              links={links}
              apiBase={live ? base : null}
            />
            {timeline.status !== "running" && timeline.status !== "idle" ? (
              <div
                className={`border-l-2 pl-4 ${timeline.status === "completed" ? "border-pass" : "border-fail"}`}
                aria-label="Session outcome"
              >
                <p className="text-sm text-ink-3">
                  Session {timeline.status === "completed" ? "finished" : timeline.status} at{" "}
                  <span className="num">{formatOffset(timeline.t_last)}</span> after {timeline.tool_calls} tool calls
                </p>
                {timeline.summary ? <p className="mt-1 text-ink">{timeline.summary}</p> : null}
                {timeline.answer && timeline.answer !== timeline.summary ? (
                  <p className="mt-1 text-sm text-ink-2">
                    <span className="text-xs text-ink-3">The agent&apos;s answer: </span>
                    {timeline.answer}
                  </p>
                ) : null}
                {timeline.reason ? <p className="mt-1 text-sm text-fail">{timeline.reason}</p> : null}
              </div>
            ) : null}
          </div>
        ) : (
          <p className="mt-4 border border-dashed border-rule-strong p-6 text-sm text-ink-3">
            {busy ? "Waiting for the first event..." : "Press Run to watch the agent work. Each tool call appears here as a card."}
          </p>
        )}
      </section>
    </>
  );
}
