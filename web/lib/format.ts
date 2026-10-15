/** Number and time formatting shared by pages and overlays (en-US, SI units). */

const intFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function formatInt(n: number): string {
  return intFmt.format(n);
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
