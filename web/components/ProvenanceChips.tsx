import type { ComputeProvenance, DesignProvenance } from "@/lib/types";
import { formatDuration, formatInt, formatUsd } from "@/lib/format";

interface Chip {
  key: string;
  /** What the chip is about: "Physics" or "Design". */
  subject: string;
  headline: string;
  facts: string[];
  /** Local or hand-made runs are not scored results; they are drawn quieter. */
  scored: boolean;
}

export function computeChip(c: ComputeProvenance): Chip {
  if (c.kind === "nebius") {
    return {
      key: `nebius-${c.job_id}`,
      subject: "Physics",
      headline: `Nebius Serverless Job ${c.job_id}`,
      facts: [c.platform, c.preset, c.region, formatDuration(c.wall_s), formatUsd(c.cost_usd), c.engine],
      scored: true,
    };
  }
  const facts = [c.engine, `${c.executor.toUpperCase()} executor`];
  if (c.wall_s !== null) facts.push(`${formatDuration(c.wall_s)} sim wall time`);
  if (c.sim_steps !== null) facts.push(`${formatInt(c.sim_steps)} steps`);
  return {
    key: "local",
    subject: "Physics",
    headline: `Local run on a ${c.host.toLowerCase()}, not a Nebius job`,
    facts,
    scored: false,
  };
}

export function designChip(d: DesignProvenance): Chip {
  switch (d.kind) {
    case "llm":
      return {
        key: `llm-${d.model}`,
        subject: "Design",
        headline: `${d.model} via ${d.provider}`,
        facts: [
          `${formatInt(d.tokens_in + d.tokens_out)} tokens`,
          `${d.latency_s.toFixed(1)} s`,
          formatUsd(d.cost_usd),
          ...(d.round !== null ? [`round ${d.round}`] : []),
        ],
        scored: true,
      };
    case "random":
      return {
        key: "random",
        subject: "Design",
        headline: `Random sample (${d.sampler})`,
        facts: [`seed ${d.seed}`],
        scored: true,
      };
    default:
      return {
        key: "manual",
        subject: "Design",
        headline: "Hand-designed baseline, no model call",
        facts: d.note ? [d.note] : [],
        scored: false,
      };
  }
}

/** Distinct design chips for a set of lanes (one per provenance). */
export function uniqueDesignChips(list: DesignProvenance[]): Chip[] {
  const seen = new Map<string, Chip>();
  for (const d of list) {
    const chip = designChip(d);
    if (!seen.has(chip.key)) seen.set(chip.key, chip);
  }
  return [...seen.values()];
}

/**
 * Provenance chips: where the physics ran and who designed the fingers. Values
 * come only from the data; a local run says so plainly.
 */
export function ProvenanceChips({ chips, className = "" }: { chips: Chip[]; className?: string }) {
  return (
    <ul className={`flex flex-wrap gap-2 ${className}`} aria-label="Provenance">
      {chips.map((chip) => (
        <li
          key={`${chip.subject}-${chip.key}`}
          className={`inline-grid max-w-full grid-cols-[auto_1fr] border text-sm ${
            chip.scored ? "border-accent" : "border-dashed border-rule-strong"
          }`}
        >
          <span
            className={`px-2 py-1 text-xs leading-5 ${
              chip.scored ? "bg-accent text-accent-ink" : "bg-paper-2 text-ink-2"
            }`}
          >
            {chip.subject}
          </span>
          <span className="min-w-0 px-2 py-1">
            <span className="block text-ink">{chip.headline}</span>
            {chip.facts.length > 0 ? (
              <span className="num mt-0.5 flex flex-wrap gap-x-3 text-xs text-ink-2">
                {chip.facts.map((f) => (
                  <span key={f}>{f}</span>
                ))}
              </span>
            ) : null}
          </span>
        </li>
      ))}
    </ul>
  );
}
