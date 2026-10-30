/** Number and time formatting shared by pages and overlays (en-US, SI units). */

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function formatInt(n: number): string {
  return intFmt.format(n);
}

/** A count with its noun, the noun in the plural unless the count is exactly one: "1 step", "2 steps", "0 steps". */
export function countOf(n: number, one: string, many: string = `${one}s`): string {
  return `${formatInt(n)} ${n === 1 ? one : many}`;
}

/** Seconds as "0.47 s", "41.2 s", "7 min 41 s" or "2 h 05 min". */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "n/a";
  if (seconds < 10) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h} h ${String(m).padStart(2, "0")} min`;
  return `${m} min ${String(s).padStart(2, "0")} s`;
}

/** Whole minutes, as docs/RESULTS.md gives job times: 1671 s -> "28 min". */
export function formatMinutes(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "n/a";
  return `${Math.round(seconds / 60)} min`;
}

/**
 * A p value as docs/RESULTS.md writes it: two significant figures with trailing zeros
 * dropped, in exponent form below 1e-4 ("0.0011", "0.012", "0.001", "5.5e-05",
 * "7e-05"); 1 reads "1.0". The two figures are rounded half up from the value at three
 * figures, as RESULTS.md does, so a Holm-adjusted 0.000164999 reads "0.00017".
 */
export function formatP(p: number): string {
  if (!Number.isFinite(p) || p < 0) return "n/a";
  if (p === 0) return "0";
  let e = Math.floor(Math.log10(p));
  let m3 = Math.round(p / 10 ** (e - 2));
  if (m3 >= 1000) {
    e += 1;
    m3 = Math.round(p / 10 ** (e - 2));
  }
  let m2 = Math.round(m3 / 10);
  if (m2 >= 100) {
    e += 1;
    m2 = 10;
  }
  if (e >= 0) return "1.0";
  const figures = m2 % 10 === 0 ? String(m2 / 10) : String(m2);
  if (e < -4) return `${figures.length > 1 ? `${figures[0]}.${figures.slice(1)}` : figures}e-${String(-e).padStart(2, "0")}`;
  return `0.${"0".repeat(-e - 1)}${figures}`;
}

/** Playback clock as "s.ss" with a fixed width, e.g. "04.27". */
export function formatClock(t: number): string {
  const clamped = Math.max(0, t);
  return clamped.toFixed(2).padStart(5, "0");
}

/** US dollars with enough precision for sub-cent LLM calls: $0.0034, $0.20, $12.40. */
export function formatUsd(usd: number): string {
  if (!Number.isFinite(usd)) return "n/a";
  if (usd === 0) return "$0";
  if (usd < 0.01) return `$${usd.toPrecision(2)}`;
  if (usd < 100) return `$${usd.toFixed(2)}`;
  return `$${formatInt(usd)}`;
}

export function formatMm(metres: number, digits = 1): string {
  return `${(metres * 1000).toFixed(digits)} mm`;
}

export function formatPercent(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** "2026-10-15 09:58 UTC" from an ISO time. */
export function formatUtc(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** Signed lift in mm with one decimal ("+149.1"); noise below 0.05 mm reads "0.0". */
export function formatLift(metres: number): string {
  const mm = metres * 1000;
  if (Math.abs(mm) < 0.05) return "0.0";
  return `${mm > 0 ? "+" : ""}${mm.toFixed(1)}`;
}

/** The text with its first character upper-cased ("the box" -> "The box"), for captions from data. */
export function capitalizeFirst(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Session clock offset as "m:ss" ("0:06", "5:42", "72:10"). */
export function formatOffset(seconds: number): string {
  const total = Math.max(0, Math.round(Number.isFinite(seconds) ? seconds : 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** Byte size as "940 B", "18.0 kB", "2.4 MB". */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "n/a";
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1e6) return `${(bytes / 1000).toFixed(1)} kB`;
  return `${(bytes / 1e6).toFixed(1)} MB`;
}
