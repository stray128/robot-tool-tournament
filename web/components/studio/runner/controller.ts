/**
 * The hosted runner's state machine, free of React so it can be tested without a browser. The
 * panel (HostedRunner.tsx) binds it with useSyncExternalStore.
 *
 * What it keeps, and what it never does with the key:
 *   - the key lives in `state.key`, in memory only. It is sent in the JSON body of one POST /plan
 *     and nowhere else: no storage, no URL, no header, no log. With "forget after use" on (the
 *     default) it is cleared as soon as that request settles, whether it worked or not, and it is
 *     cleared when the panel goes away (`stop`).
 *   - a message from the server is struck of the key before it is kept (`redact`).
 *   - the run id, which is not secret, goes in the URL hash as `#run-<id>` so a reload resumes.
 */

import {
  getHealth,
  getRun,
  hashForRun,
  MESSAGES,
  type OffReason,
  offReason,
  type PlanReply,
  type Provider,
  postPlan,
  postRun,
  redact,
  type RunnerError,
  type RunnerErrorKind,
  type RunnerHealth,
  type RunRecord,
  runIdFromHash,
} from "@/lib/runner";

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

/** Whether the panel is shown: not at all until /health answers, a notice when the runner is switched off. */
export type Gate = "checking" | "ready" | "off" | "hidden";

/** What went wrong, in the words shown, and which call a Try again button repeats. */
export interface Fault {
  kind: RunnerErrorKind;
  message: string;
  retry: "plan" | "run" | "poll" | null;
  /** The runner's own checks that a refused plan failed, when it listed them. */
  problems?: string[];
}

export type PlanPhase = "idle" | "planning" | "done" | "error";
export type RunPhase = "idle" | "submitting" | "checking" | "queued" | "running" | "done" | "failed";

export interface RunnerState {
  gate: Gate;
  /** Why the gate is "off" (the kill switch, `open: false`, or an auto-off date that has passed); null otherwise. */
  offReason: OffReason | null;
  health: RunnerHealth | null;
  provider: Provider;
  /** The key as typed. In memory only; see the file header. */
  key: string;
  forget: boolean;
  planPhase: PlanPhase;
  planStartedAt: number | null;
  planReply: PlanReply | null;
  planFault: Fault | null;
  runPhase: RunPhase;
  runId: string | null;
  /** When the clock for "elapsed" starts, in ms: the submit, or the run's own created_at after a reload. */
  runStartedAt: number | null;
  runEtaS: number | null;
  runRecord: RunRecord | null;
  runFault: Fault | null;
}

/** Where the run id lives in the address bar. */
export interface HashStore {
  read(): string;
  write(id: string | null): void;
}

/** The address bar: `#run-<id>` is set with replaceState (no history entry, no scroll, the path untouched). */
export const browserHash: HashStore = {
  read: () => window.location.hash,
  write(id) {
    try {
      if (id) window.history.replaceState(null, "", hashForRun(id));
      else if (runIdFromHash(window.location.hash)) window.history.replaceState(null, "", window.location.pathname + window.location.search);
    } catch {
      /* a frame that forbids it only loses the resume on reload; the run itself is unaffected */
    }
  },
};

export interface ControllerOptions {
  base: string;
  fetchImpl?: FetchLike;
  hash?: HashStore;
  /** Called with every plan reply that carries a plan document, valid or not (the page loads it into the editor). */
  onPlan?: (reply: PlanReply) => void;
  /** The wait between status checks, before backoff. */
  pollMs?: number;
  /** The clock, in ms; today's date in UTC decides whether an older runner (no `open` field) is past its auto-off date. */
  now?: () => number;
}

export const POLL_MS = 10_000;
const POLL_MAX_MS = 60_000;
const POLL_MAX_FAILURES = 5;
/** A run that has not finished after this long is no longer watched; Check again looks once more. */
const WATCH_MAX_MS = 2 * 60 * 60 * 1000;
export const HEALTH_TIMEOUT_MS = 8_000;
/**
 * Planning is usually 15 to 45 s, but a busy provider or a retry inside the runner can make it much longer,
 * so a plan is given 5.5 minutes before the page gives up (the panel says "up to five minutes").
 */
export const PLAN_TIMEOUT_MS = 330_000;
/** A plan the runner has not seen can take 2 to 3 minutes to be checked before it answers 202; 3.5 minutes leaves room. */
export const SUBMIT_TIMEOUT_MS = 210_000;

const initial = (): RunnerState => ({
  gate: "checking",
  offReason: null,
  health: null,
  provider: "nvidia",
  key: "",
  forget: true,
  planPhase: "idle",
  planStartedAt: null,
  planReply: null,
  planFault: null,
  runPhase: "idle",
  runId: null,
  runStartedAt: null,
  runEtaS: null,
  runRecord: null,
  runFault: null,
});

function faultOf(err: unknown, retry: Fault["retry"], key = ""): Fault {
  const e = err as Partial<RunnerError> | null;
  if (e && typeof e === "object" && typeof e.kind === "string" && typeof e.message === "string") {
    const retryable = e.kind === "network" || e.kind === "server" || e.kind === "busy";
    const problems = Array.isArray(e.problems) ? e.problems.map((p) => redact(p, key)) : [];
    return { kind: e.kind, message: redact(e.message, key), retry: retryable ? retry : null, ...(problems.length ? { problems } : {}) };
  }
  return { kind: "server", message: MESSAGES.server, retry };
}

/** An abort signal that fires after `ms`, and says whether it was the timeout that fired it. */
function deadline(ms: number, parent?: AbortController): { controller: AbortController; timedOut: () => boolean; clear: () => void } {
  const controller = parent ?? new AbortController();
  let fired = false;
  const t = setTimeout(() => {
    fired = true;
    controller.abort();
  }, ms);
  return { controller, timedOut: () => fired, clear: () => clearTimeout(t) };
}

export class RunnerController {
  private state: RunnerState;
  private readonly listeners = new Set<() => void>();
  private readonly base: string;
  private readonly fetchImpl: FetchLike | undefined;
  private readonly hash: HashStore;
  private readonly onPlan: ((reply: PlanReply) => void) | undefined;
  private readonly pollMs: number;
  private readonly now: () => number;
  /** Bumped by stop() so answers that arrive afterwards are dropped. */
  private epoch = 0;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private inflight = new Set<AbortController>();
  private failures = 0;

  constructor(opts: ControllerOptions) {
    this.base = opts.base;
    this.fetchImpl = opts.fetchImpl;
    this.hash = opts.hash ?? browserHash;
    this.onPlan = opts.onPlan;
    this.pollMs = opts.pollMs ?? POLL_MS;
    this.now = opts.now ?? Date.now;
    this.state = initial();
  }

  // ------------------------------------------------------------ store

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): RunnerState => this.state;

  private set(patch: Partial<RunnerState>): void {
    this.state = { ...this.state, ...patch };
    for (const l of [...this.listeners]) l();
  }

  setProvider(provider: Provider): void {
    this.set({ provider, planFault: this.state.planFault?.kind === "key_missing" ? null : this.state.planFault });
  }

  setKey(key: string): void {
    this.set({ key, planFault: this.state.planFault?.kind === "key_missing" ? null : this.state.planFault });
  }

  setForget(forget: boolean): void {
    this.set({ forget });
  }

  /** Whether Plan live can be pressed: a description, and a key unless the free allowance is chosen. */
  canPlan(description: string): boolean {
    const s = this.state;
    return s.planPhase !== "planning" && description.trim().length > 0 && (s.provider === "free" || s.key.trim().length > 0);
  }

  // ------------------------------------------------------------ lifecycle

  /** Ask /health whether to show the panel, then pick up a run named in the URL hash. */
  async start(): Promise<void> {
    const epoch = ++this.epoch;
    this.set({ gate: "checking" });
    const ac = new AbortController();
    const dl = deadline(HEALTH_TIMEOUT_MS, ac);
    this.inflight.add(ac);
    let health: RunnerHealth | null = null;
    try {
      health = await getHealth(this.base, ac.signal, this.fetchImpl);
    } catch {
      health = null;
    } finally {
      dl.clear();
      this.inflight.delete(ac);
    }
    if (epoch !== this.epoch) return;
    if (!health || !health.ok) {
      this.set({ gate: "hidden", health, offReason: null });
      return;
    }
    // /health answers 200 with ok: true after the auto-off date too; `open` (or, from an older runner, the date) says it is off.
    const reason = offReason(health, this.now());
    this.set({ gate: reason ? "off" : "ready", health, offReason: reason });
    if (reason) return;
    const id = runIdFromHash(this.hash.read());
    if (id && this.state.runPhase === "idle") void this.track(id);
  }

  /** The panel is going away: stop every request and timer, and forget the key. */
  stop(): void {
    this.epoch++;
    this.clearPoll();
    for (const ac of this.inflight) ac.abort();
    this.inflight.clear();
    this.state = { ...this.state, key: "" };
  }

  private clearPoll(): void {
    if (this.pollTimer !== null) clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }

  private async refreshHealth(): Promise<void> {
    const epoch = this.epoch;
    try {
      const health = await getHealth(this.base, undefined, this.fetchImpl);
      if (epoch === this.epoch && health.ok) this.set({ health });
    } catch {
      /* the numbers beside the buttons just stay as they were */
    }
  }

  // ------------------------------------------------------------ plan

  /** Plan the description with the chosen provider; hands any plan document to `onPlan`. */
  async plan(description: string): Promise<void> {
    const text = description.trim();
    const { provider, key } = this.state;
    if (!text) {
      this.set({ planPhase: "error", planFault: { kind: "bad_request", message: "Describe the use case first, or pick an example.", retry: null } });
      return;
    }
    if (provider !== "free" && !key.trim()) {
      this.set({ planPhase: "error", planFault: { kind: "key_missing", message: MESSAGES.key_missing, retry: null } });
      return;
    }
    const epoch = this.epoch;
    const ac = new AbortController();
    const dl = deadline(PLAN_TIMEOUT_MS, ac);
    this.inflight.add(ac);
    this.set({ planPhase: "planning", planStartedAt: Date.now(), planReply: null, planFault: null });
    const used = provider === "free" ? "" : key.trim();
    let reply: PlanReply | null = null;
    let fault: Fault | null = null;
    try {
      reply = await postPlan(this.base, { description: text.slice(0, 4000), provider, key: provider === "free" ? null : used }, ac.signal, this.fetchImpl);
    } catch (err) {
      fault = dl.timedOut()
        ? { kind: "server", message: "The planner took too long. Try again.", retry: "plan" }
        : ac.signal.aborted
          ? null
          : faultOf(err, "plan", used);
    } finally {
      dl.clear();
      this.inflight.delete(ac);
      // The request is over: with "forget after use" on, the key goes now, whatever the outcome.
      if (this.state.forget && epoch === this.epoch) this.set({ key: "" });
    }
    if (epoch !== this.epoch) return;
    if (fault) {
      this.set({ planPhase: "error", planFault: fault });
      return;
    }
    if (!reply) {
      this.set({ planPhase: "idle" });
      return;
    }
    const clean: PlanReply = { ...reply, problems: reply.problems.map((p) => redact(p, used)) };
    this.set({ planPhase: "done", planReply: clean });
    if (clean.plan) this.onPlan?.(clean);
    void this.refreshHealth();
  }

  // ------------------------------------------------------------ run

  /** Start a hosted physics run of the plan document and watch it. */
  async run(plan: Record<string, unknown>): Promise<void> {
    const epoch = this.epoch;
    this.clearPoll();
    this.failures = 0;
    const ac = new AbortController();
    const dl = deadline(SUBMIT_TIMEOUT_MS, ac);
    this.inflight.add(ac);
    this.set({ runPhase: "submitting", runFault: null, runRecord: null, runId: null, runStartedAt: Date.now(), runEtaS: null });
    try {
      const ticket = await postRun(this.base, plan, ac.signal, this.fetchImpl);
      if (epoch !== this.epoch) return;
      this.hash.write(ticket.runId);
      this.set({ runPhase: "queued", runId: ticket.runId, runEtaS: ticket.etaS });
      this.scheduleTick(ticket.runId, this.pollMs);
    } catch (err) {
      if (epoch !== this.epoch) return;
      const fault: Fault = dl.timedOut() ? { kind: "server", message: MESSAGES.server, retry: "run" } : faultOf(err, "run");
      this.set({ runPhase: "idle", runFault: fault, runStartedAt: null });
    } finally {
      dl.clear();
      this.inflight.delete(ac);
    }
  }

  /** Watch a run by id (a reload with `#run-<id>`, or Check again after a fault). */
  async track(id: string): Promise<void> {
    this.clearPoll();
    this.failures = 0;
    this.set({
      runId: id,
      runPhase: this.state.runRecord ? this.state.runPhase : "checking",
      runFault: null,
      runStartedAt: this.state.runStartedAt ?? Date.now(),
    });
    await this.tick(id);
  }

  /** Check again after a fault. */
  poll(): void {
    if (this.state.runId) void this.track(this.state.runId);
  }

  /** Leave the run: drop its result and take its id out of the address. */
  reset(): void {
    this.epoch++;
    this.clearPoll();
    this.hash.write(null);
    this.set({ runPhase: "idle", runId: null, runStartedAt: null, runEtaS: null, runRecord: null, runFault: null });
  }

  private scheduleTick(id: string, delay: number): void {
    this.clearPoll();
    const epoch = this.epoch;
    this.pollTimer = setTimeout(() => {
      this.pollTimer = null;
      if (epoch === this.epoch) void this.tick(id);
    }, delay);
  }

  private async tick(id: string): Promise<void> {
    const epoch = this.epoch;
    const ac = new AbortController();
    this.inflight.add(ac);
    let record: RunRecord;
    try {
      record = await getRun(this.base, id, ac.signal, this.fetchImpl);
    } catch (err) {
      this.inflight.delete(ac);
      if (epoch !== this.epoch || ac.signal.aborted) return;
      const fault = faultOf(err, "poll");
      const transient = fault.kind === "network" || fault.kind === "server" || fault.kind === "busy" || fault.kind === "cap" || fault.kind === "bad_reply";
      this.failures++;
      if (transient && this.failures < POLL_MAX_FAILURES) {
        this.scheduleTick(id, Math.min(this.pollMs * 1.5 ** this.failures, POLL_MAX_MS));
        return;
      }
      this.set({ runFault: { ...fault, retry: fault.kind === "not_found" || fault.kind === "bad_request" ? null : "poll" } });
      return;
    }
    this.inflight.delete(ac);
    if (epoch !== this.epoch) return;
    this.failures = 0;
    const created = record.createdAt ? Date.parse(record.createdAt) : NaN;
    const startedAt = this.state.runStartedAt !== null && this.state.runPhase !== "checking" ? this.state.runStartedAt : Number.isFinite(created) ? Math.min(created, Date.now()) : (this.state.runStartedAt ?? Date.now());
    this.set({ runRecord: record, runPhase: record.status, runStartedAt: startedAt, runFault: null });
    if (record.status === "done" || record.status === "failed") {
      void this.refreshHealth();
      return;
    }
    if (Date.now() - startedAt > WATCH_MAX_MS) {
      this.set({ runFault: { kind: "server", message: "This run is taking longer than expected. Check again in a few minutes.", retry: "poll" } });
      return;
    }
    this.scheduleTick(id, this.pollMs);
  }
}
