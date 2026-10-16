/**
 * A small Server-Sent Events parser for fetch() streams.
 *
 * EventSource cannot send headers, and the live API wants the passcode in a header
 * (never in the URL), so the session stream is read with fetch() and parsed here.
 * Follows the WHATWG event-stream rules that matter for this API: `event:`, `data:`
 * (multi-line, joined with "\n"), `id:`, `retry:`, comment lines starting with ":",
 * CRLF / CR / LF line endings, and dispatch on a blank line.
 */

export interface SseMessage {
  /** The `event:` field, "message" when absent. */
  event: string;
  data: string;
  id: string | null;
}

export interface SseParser {
  /** Feed the next decoded text chunk; complete messages are dispatched synchronously. */
  push(chunk: string): void;
  /** Dispatch a trailing message that had no blank line after it (stream ended). */
  flush(): void;
  /** Last `retry:` value in ms, if the server sent one. */
  readonly retryMs: number | null;
}

export function createSseParser(onMessage: (message: SseMessage) => void): SseParser {
  let buffer = "";
  let event = "";
  let data: string[] = [];
  let id: string | null = null;
  let lastId: string | null = null;
  let retryMs: number | null = null;
  let pendingCr = false;

  const dispatch = () => {
    if (data.length > 0) {
      lastId = id ?? lastId;
      onMessage({ event: event || "message", data: data.join("\n"), id: lastId });
    }
    event = "";
    data = [];
    id = null;
  };

  const line = (text: string) => {
    if (text === "") return dispatch();
    if (text.startsWith(":")) return;
    const colon = text.indexOf(":");
    const field = colon < 0 ? text : text.slice(0, colon);
    let value = colon < 0 ? "" : text.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    switch (field) {
      case "event":
        event = value;
        break;
      case "data":
        data.push(value);
        break;
      case "id":
        if (!value.includes("\0")) id = value;
        break;
      case "retry":
        if (/^\d+$/.test(value)) retryMs = Number(value);
        break;
      default:
        break;
    }
  };

  return {
    push(chunk: string) {
      let text = chunk;
      // A CR at the end of the previous chunk may be the first half of a CRLF.
      if (pendingCr && text.startsWith("\n")) text = text.slice(1);
      pendingCr = text.endsWith("\r");
      buffer += text;
      const parts = buffer.split(/\r\n|\r|\n/);
      buffer = parts.pop() ?? "";
      for (const p of parts) line(p);
    },
    flush() {
      if (buffer) {
        line(buffer);
        buffer = "";
      }
      dispatch();
    },
    get retryMs() {
      return retryMs;
    },
  };
}
