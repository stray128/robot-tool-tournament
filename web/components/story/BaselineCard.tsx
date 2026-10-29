import { runsText, TIER_LABELS } from "@/components/studio/schema";
import type { BaselineStory } from "@/components/studio/story";

/**
 * The recommender's baseline beside the pick: the best eligible single tool (rule R01, one tool by
 * default) and why the pick is it or beats it, or, when no single tool is eligible, how each stops.
 */
export function BaselineCard({ baseline }: { baseline: BaselineStory }) {
  const row = baseline.row;
  return (
    <section className="panel min-w-0 p-4 sm:p-5" aria-labelledby="baseline-title" data-baseline={row ? row.setup_id : "none"}>
      <p className="text-sm font-semibold text-baseline">The baseline: the best single tool</p>
      {row ? (
        <>
          <h3 id="baseline-title" className="heading mt-1 text-lg text-ink">
            {row.label}
          </h3>
          <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-2">
            <span className="num text-ink">{runsText(row.completed_runs, row.n_runs)}</span>
            <span>{TIER_LABELS[row.fidelity]}</span>
            {baseline.offTheShelf ? <span className="text-ink-3">also the best single off-the-shelf tool</span> : null}
          </p>
        </>
      ) : (
        <h3 id="baseline-title" className="heading mt-1 text-lg text-ink">
          No single tool is eligible
        </h3>
      )}
      <ul className="mt-3 grid gap-1.5 text-[0.9375rem] text-ink-2" data-baseline-why>
        {(row ? baseline.lines : baseline.lines.slice(1)).map((l) => (
          <li key={l} className="flex gap-2">
            <span className="mt-2 size-1.5 shrink-0 rounded-full bg-baseline" aria-hidden="true" />
            <span>{l}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
