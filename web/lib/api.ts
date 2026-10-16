/**
 * Client for the local Grasp Engineer API (src/rtt/api/app.py, FastAPI), used by /part
 * in live mode. Live mode is on only when NEXT_PUBLIC_API_URL is set at build time; the
 * public static site never has it, so it plays a published demo session instead.
 *
 * Endpoints used (see the module docstring of src/rtt/api/app.py):
 *   GET  /health                  -> {ok, mode, passcode_configured, targets, ...}
 *   POST /parts?name=&units=mm|m  raw STL body (application/octet-stream) -> {part: PartInfo}
 *   POST /sessions                JSON {part_id, passcode, goal?} -> {session_id}
 *   GET  /sessions/{id}/events    text/event-stream: `id` = seq, `event` = type, JSON data;
 *                                 resumes after Last-Event-ID; ends with `event: end`
 *   POST /sessions/{id}/stop      stop after the current step
 *   GET  /sessions/{id}           the session document, read once after `end` for any
 *                                 event the stream missed
 *   GET  /runs/{id}, GET /printpack/{id}   plain links (no passcode needed)
 *
 * The passcode only goes to POST /sessions, in the JSON body and the `X-RTT-Passcode`
 * header, never in a URL. The stream is read with fetch() (not EventSource) so a
 * dropped connection can resume from the last event id.
 */

import { createSseParser } from "./sse";
import { isObj, pick, str } from "./session";

export const PASSCODE_HEADER = "X-RTT-Passcode";

/** The API origin from NEXT_PUBLIC_API_URL (inlined at build time), without a trailing slash; null when unset. */
export function apiBase(raw: string | undefined = process.env.NEXT_PUBLIC_API_URL): string | null {
  const v = raw?.trim();
  if (!v) return null;
  try {
    const url = new URL(v);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url.toString().replace(/\/+$/, "");
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** A readable message from an error response (FastAPI puts it in `detail`). */
export async function errorMessage(res: Response): Promise<string> {
  const fallback = `${res.status} ${res.statusText || "error"}`;
  try {
    const text = await res.text();
    try {
      const body: unknown = JSON.parse(text);
      if (isObj(body)) {
        const detail = pick(body, "detail", "error", "message");
        if (typeof detail === "string") return detail;
        if (Array.isArray(detail)) return detail.map((d) => (isObj(d) ? (str(d.msg) ?? JSON.stringify(d)) : String(d))).join("; ");
      }
    } catch {
      /* not JSON */
    }
    return text.trim().slice(0, 300) || fallback;
  } catch {
    return fallback;
  }
}

const HINTS: Record<number, string> = {
  403: " Check the passcode.",
  413: " The file is larger than the API accepts.",
  429: " Too many attempts, or the server is busy; wait a minute.",
};

async function json(res: Response): Promise<Record<string, unknown>> {
  if (!res.ok) {
    const detail = await errorMessage(res);
    const hint = HINTS[res.status];
    throw new ApiError(res.status, hint ? `${detail.replace(/[.\s]*$/, ".")}${hint}` : detail);
  }
  const body: unknown = await res.json().catch(() => ({}));
  return isObj(body) ? body : {};
}

export interface Health {
  ok: boolean;
  /** `live` calls Nemotron and spends; `offline` is the scripted agent. */
  mode: string | null;
  passcode_configured: boolean | null;
  targets: string[];
  /** Physics backend name; "fake" is synthetic physics for UI work. */
  backend: string | null;
}

export async function getHealth(base: string, signal?: AbortSignal): Promise<Health> {
  const body = await json(await fetch(`${base}/health`, { signal, cache: "no-store" }));
  return {
    ok: body.ok === true,
    mode: str(body.mode),
    passcode_configured: typeof body.passcode_configured === "boolean" ? body.passcode_configured : null,
    targets: Array.isArray(body.targets) ? body.targets.map(String) : [],
    backend: str(body.backend),
  };
}

export interface UploadedPart {
  part_id: string;
  /** The API's PartInfo (size, fit, convex pieces...), kept for display. */
  part: Record<string, unknown>;
}

/**
 * Upload an STL as the raw request body. `units` is the file's unit as the API knows
 * it (mm or m); a file in other units should be re-encoded in mm first.
 */
export async function uploadPart(
  base: string,
  body: Blob | ArrayBuffer,
  name: string,
  units: "mm" | "m",
  signal?: AbortSignal,
): Promise<UploadedPart> {
  const params = new URLSearchParams({ name: name.replace(/\.stl$/i, "").slice(0, 120), units });
  const res = await fetch(`${base}/parts?${params}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body,
    signal,
  });
  const out = await json(res);
  const part = isObj(out.part) ? out.part : out;
  const part_id = str(pick(part, "part_id", "id"));
  if (!part_id) throw new ApiError(500, "The API accepted the part but returned no part_id.");
  return { part_id, part };
}

export async function startSession(base: string, partId: string, passcode: string, signal?: AbortSignal): Promise<string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (passcode) headers[PASSCODE_HEADER] = passcode;
  const res = await fetch(`${base}/sessions`, {
    method: "POST",
    headers,
    body: JSON.stringify({ part_id: partId, ...(passcode ? { passcode } : {}) }),
    signal,
  });
  const body = await json(res);
  const id = str(pick(body, "session_id", "id"));
  if (!id) throw new ApiError(500, "The API started a session but returned no session_id.");
  return id;
}

/** Ask the agent to stop after its current step; the stream then ends with session.finished. */
export async function stopSession(base: string, sessionId: string): Promise<void> {
  await json(await fetch(`${base}/sessions/${encodeURIComponent(sessionId)}/stop`, { method: "POST" }));
}

/**
 * After `event: end`, hand over any events the stream did not carry, from the session
 * document (`GET /sessions/{id}`). The server can mark a session done a moment before it
 * appends `session.finished`, and a stream that looks in between ends without it; the
 * finished session's document always has it. Best effort: a failed fetch is ignored.
 */
async function catchUp(
  base: string,
  sessionId: string,
  lastId: string | null,
  onEvent: (payload: unknown, sseEvent: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const after = lastId !== null && /^\d+$/.test(lastId) ? Number(lastId) : 0;
  try {
    const res = await fetch(`${base}/sessions/${encodeURIComponent(sessionId)}`, { signal, cache: "no-store" });
    if (!res.ok) return;
    const doc: unknown = await res.json();
    const events = isObj(doc) && Array.isArray(doc.events) ? doc.events : [];
    for (const e of events) {
      if (isObj(e) && typeof e.seq === "number" && e.seq > after) onEvent(e, str(e.type) ?? "message");
    }
  } catch {
    /* the stream already ended; nothing more to show */
  }
}

/**
 * Read the session's event stream until `event: end` or `signal` aborts. Each SSE
 * message's data is parsed as JSON (a non-JSON payload becomes a message event) and
 * handed to `onEvent` with the SSE event name. A dropped connection is resumed from
 * the last event id; `retries` bounds consecutive failed attempts, and a connection
 * that delivered events before it dropped resets the count, so a long session survives
 * several drops. After `end`, events the stream missed are read from the session
 * document (see `catchUp`).
 */
export async function streamEvents(
  base: string,
  sessionId: string,
  onEvent: (payload: unknown, sseEvent: string) => void,
  signal?: AbortSignal,
  retries = 3,
): Promise<void> {
  let lastId: string | null = null;
  let ended = false;
  let failures = 0;
  for (;;) {
    const idBefore = lastId;
    try {
      const headers: Record<string, string> = { Accept: "text/event-stream" };
      if (lastId) headers["Last-Event-ID"] = lastId;
      const res = await fetch(`${base}/sessions/${encodeURIComponent(sessionId)}/events`, { headers, signal, cache: "no-store" });
      if (!res.ok || !res.body) throw new ApiError(res.status, await errorMessage(res));
      const parser = createSseParser((m) => {
        if (m.event === "end") {
          ended = true;
          return;
        }
        if (m.id) lastId = m.id;
        let payload: unknown;
        try {
          payload = JSON.parse(m.data);
        } catch {
          payload = { type: "message", text: m.data };
        }
        onEvent(payload, m.event);
      });
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      try {
        while (!ended) {
          const { value, done } = await reader.read();
          if (done) break;
          parser.push(value);
        }
        if (!ended) parser.flush();
      } finally {
        await reader.cancel().catch(() => {});
      }
      if (ended) {
        await catchUp(base, sessionId, lastId, onEvent, signal);
        return;
      }
      throw new ApiError(0, "The event stream closed before the session ended.");
    } catch (err) {
      if (signal?.aborted || (err instanceof ApiError && err.status >= 400 && err.status < 500)) throw err;
      failures = lastId !== idBefore ? 1 : failures + 1;
      if (failures > retries) throw err;
      await new Promise((r) => setTimeout(r, 1000 * failures));
    }
  }
}
