import type { ReactNode } from "react";

import { type FinalistRow, runEnd, type StudioLane, TIER_LABELS, TIER_NOTES } from "@/components/studio/schema";
import { massShort } from "@/components/studio/story";

import { Term } from "./Term";

export interface HeldoutPair {
  n: number;
  recommended: number;
  best_single: number;
}

function Meter({ value, n, tone }: { value: number; n: number; tone: "accent" | "baseline" }) {
  return (
    <span className="mt-1.5 block h-1.5 w-full max-w-[12rem] overflow-hidden rounded-full bg-paper-2" aria-hidden="true">
      <span className={`block h-full rounded-full ${tone === "accent" ? "bg-accent" : "bg-baseline"}`} style={{ width: `${n ? (value / n) * 100 : 0}%` }} />
    </span>
  );
}

interface Row {
  key: string;
  label: ReactNode;
  rec: ReactNode;
  base: ReactNode;
}

/** The recommendation against the best single off-the-shelf tool, metric by metric. */
export function HeadToHead({
  rec,
  recTitle = "Recommended",
  base,
  baseLane,
  heldout,
  conditions,
  placementMm,
}: {
  rec: FinalistRow;
  /** The left column's name: "Recommended", or "Closest setup" when nothing completes the task. */
  recTitle?: string;
  base: FinalistRow;
  baseLane: StudioLane | null;
  heldout: HeldoutPair | null;
  conditions: number;
  placementMm: number | null;
}) {
  const rows: Row[] = [
    {
      key: "dev",
      label: (
        <Term
          label="Development conditions"
          tip={`The ${conditions} conditions the search ranked setups on: nominal placement plus ${conditions - 1} with each object shifted${
            placementMm !== null ? ` by up to ${placementMm} mm` : ""
          }, each with its own seed. Both setups ran the same ones.`}
        >
          <span>Finishes the task</span>
        </Term>
      ),
      rec: (
        <>
          <span className="num text-2xl text-ink">
            {rec.completed_runs} of {rec.n_runs}
          </span>
          <Meter value={rec.completed_runs} n={rec.n_runs} tone="accent" />
        </>
      ),
      base: (
        <>
          <span className="num text-2xl text-ink">
            {base.completed_runs} of {base.n_runs}
          </span>
          <Meter value={base.completed_runs} n={base.n_runs} tone="baseline" />
        </>
      ),
    },
  ];
  if (heldout) {
    rows.push({
      key: "heldout",
      label: (
        <Term
          label="Held-out check"
          tip="New conditions drawn from a committed secret after both setups were frozen: new seeds, frictions and a hidden placement error of up to 2 mm on each axis per object. Neither setup was chosen on them."
        >
          <span>Held-out check</span>
        </Term>
      ),
      rec: (
        <>
          <span className="num text-2xl text-ink">
            {heldout.recommended} of {heldout.n}
          </span>
          <Meter value={heldout.recommended} n={heldout.n} tone="accent" />
        </>
      ),
      base: (
        <>
          <span className="num text-2xl text-ink">
            {heldout.best_single} of {heldout.n}
          </span>
          <Meter value={heldout.best_single} n={heldout.n} tone="baseline" />
        </>
      ),
    });
  }
  rows.push(
    {
      key: "cycle",
      label: "Cycle time",
      rec: <span className="num text-xl text-ink">{rec.cycle_time_s !== null ? `${rec.cycle_time_s.toFixed(1)} s` : "never finishes"}</span>,
      base: (
        <span className="text-ink">
          {base.cycle_time_s !== null ? (
            <span className="num text-xl">{base.cycle_time_s.toFixed(1)} s</span>
          ) : (
            <>
              <span className="text-lg">Never finishes</span>
              {baseLane && !baseLane.completed ? <span className="block text-xs text-ink-3">its replay stops at {runEnd(baseLane).toFixed(1)} s</span> : null}
            </>
          )}
        </span>
      ),
    },
    {
      key: "tools",
      label: "Tools",
      rec: (
        <span className="text-ink">
          <span className="num text-xl">{rec.complexity.n_tools}</span>{" "}
          <span className="text-sm text-ink-2">
            {rec.kind_title.toLowerCase()}
            {rec.complexity.n_changes ? `, ${rec.complexity.n_changes} ${rec.kind === "multi_head" ? "head switch" : "tool change"}${rec.complexity.n_changes === 1 ? "" : "es"} per run` : ""}
          </span>
        </span>
      ),
      base: (
        <span className="text-ink">
          <span className="num text-xl">{base.complexity.n_tools}</span> <span className="text-sm text-ink-2">{base.kind_title.toLowerCase()}</span>
        </span>
      ),
    },
    {
      key: "mass",
      label: "Tool mass",
      rec: (
        <span className="text-ink">
          <span className="num text-xl">{massShort(rec.complexity.total_mass_kg)}</span>
          {rec.payload_kg !== null && Math.abs(rec.payload_kg - rec.complexity.total_mass_kg) > 0.0005 ? (
            <span className="block text-xs text-ink-3">at most {massShort(rec.payload_kg)} on the wrist at once</span>
          ) : null}
          {rec.kind === "multi_head" ? <span className="block text-xs text-ink-3">the whole head, as simulated</span> : null}
        </span>
      ),
      base: <span className="num text-xl text-ink">{massShort(base.complexity.total_mass_kg)}</span>,
    },
    {
      key: "fidelity",
      label: (
        <Term label="Fidelity" tip="How each step was simulated. Rigid: a rigid multibody model with a scripted controller. Approximation: a labelled approximation (suction seals, soft arms, latches, threads, valves); verify on hardware.">
          <span>Fidelity</span>
        </Term>
      ),
      rec: (
        <span className={rec.fidelity === "approximation" ? "text-pending" : "text-ink"} title={TIER_NOTES[rec.fidelity]}>
          {TIER_LABELS[rec.fidelity]}
        </span>
      ),
      base: (
        <span className={base.fidelity === "approximation" ? "text-pending" : "text-ink"} title={TIER_NOTES[base.fidelity]}>
          {TIER_LABELS[base.fidelity]}
        </span>
      ),
    },
  );

  return (
    <div className="panel overflow-hidden" data-head-to-head>
      <div className="grid grid-cols-2 border-b border-rule sm:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <span className="hidden sm:block" />
        <div className="border-r border-rule px-4 py-3 sm:border-l">
          <p className="text-sm font-semibold text-accent">{recTitle}</p>
          <p className="text-sm leading-snug text-ink">{rec.label}</p>
        </div>
        <div className="px-4 py-3">
          <p className="text-sm font-semibold text-baseline">Best single off-the-shelf</p>
          <p className="text-sm leading-snug text-ink">{base.label}</p>
        </div>
      </div>
      <dl>
        {rows.map((r) => (
          <div key={r.key} className="grid grid-cols-2 border-b border-rule last:border-b-0 sm:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)_minmax(0,1fr)]">
            <dt className="col-span-2 px-4 pt-3 text-sm text-ink-2 sm:col-span-1 sm:py-4">{r.label}</dt>
            <dd className="border-r border-rule px-4 pb-3 pt-1 sm:border-l sm:py-4">{r.rec}</dd>
            <dd className="px-4 pb-3 pt-1 sm:py-4">{r.base}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
