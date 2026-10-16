import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getBouts, getDesigns } from "@/lib/data";
import { FAILURE_EXPLANATIONS, FAILURE_LABELS, humanize, PARAM_LABELS } from "@/lib/labels";
import { APP_NAME } from "@/lib/site";
import type { FailureMode } from "@/lib/types";

export const metadata: Metadata = {
  title: "Method and limitations",
  description: "How grasps are scored, what the model designs, how the search is compared, and what this does not show.",
};

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-14 grid gap-x-10 gap-y-3 md:grid-cols-[14rem_minmax(0,1fr)]" aria-labelledby={id}>
      <h2 id={id} className="heading text-xl">
        {title}
      </h2>
      <div className="prose-measure grid gap-4 text-ink-2">{children}</div>
    </section>
  );
}

export default function AboutPage() {
  const task = getBouts().bouts[0]?.task;
  const { params } = getDesigns();

  return (
    <>
      <PageHeader
        title="Method and limitations"
        lede={
          <p>
            {APP_NAME} is a design loop and an open benchmark behind it. This page says how a grasp is judged, what the
            model is allowed to change, how its search is compared with a baseline, and where the evidence stops.
          </p>
        }
      />

      <Section id="grasp" title="What counts as a grasp">
        {task ? (
          <p>
            The gripper descends, closes with a fixed force, lifts and holds. A trial succeeds when the object&apos;s
            centre rises at least <span className="num text-ink">{(task.success_lift_m * 1000).toFixed(0)} mm</span>{" "}
            above where it rested and stays there for a continuous{" "}
            <span className="num text-ink">{task.success_hold_s.toFixed(1)} s</span>, finishing before{" "}
            <span className="num text-ink">{task.max_time_s.toFixed(0)} s</span>. Every failure gets exactly one
            label:
          </p>
        ) : null}
        <dl className="grid gap-2 text-sm">
          {(Object.keys(FAILURE_LABELS) as FailureMode[]).map((mode) => (
            <div key={mode} className="grid gap-x-4 border-b border-rule pb-2 sm:grid-cols-[11rem_minmax(0,1fr)]">
              <dt className="text-ink">{FAILURE_LABELS[mode]}</dt>
              <dd>{FAILURE_EXPLANATIONS[mode]}</dd>
            </div>
          ))}
        </dl>
        <p>
          Physics checks run alongside: a finger pressing into the table, deep interpenetration, or an object
          moving impossibly fast all fail the trial, so a design cannot win by exploiting the simulator.
        </p>
      </Section>

      <Section id="gripper" title="The gripper">
        <p>
          The simulated gripper matches an open-source parallel gripper
          {task ? (
            <>
              : <span className="num text-ink">{task.max_opening_mm} mm</span> maximum opening and{" "}
              <span className="num text-ink">{task.grip_force_n} N</span> of grip force
            </>
          ) : null}
          . Every design uses the same gripper, the same open-loop controller and the same force; only the fingers
          change. Fingers are modelled in PLA and built from 14 boxes each, and the printable mesh is the union of
          exactly those boxes.
        </p>
      </Section>

      <Section id="design-space" title="What the model chooses">
        <p>
          A design is ten numbers. The model proposes them through a forced tool call, and a validator rejects
          anything unprintable or unsafe before it reaches the simulator.
        </p>
        <dl className="grid gap-1.5 text-sm">
          {Object.entries(params).map(([key, spec]) => {
            const meta = PARAM_LABELS[key] ?? { label: humanize(key), unit: "" };
            const range =
              spec.options !== undefined
                ? spec.options.map(humanize).join(", ")
                : `${spec.min} to ${spec.max}${meta.unit && !["of length", "tip/root"].includes(meta.unit) ? ` ${meta.unit}` : ""}`;
            return (
              <div key={key} className="grid grid-cols-[9rem_minmax(0,1fr)] gap-x-4 border-b border-rule pb-1.5">
                <dt className="text-ink">{meta.label}</dt>
                <dd className="num">{range}</dd>
              </div>
            );
          })}
        </dl>
      </Section>

      <Section id="comparison" title="How the search is judged">
        <p>
          Nemotron with measured feedback is compared with random search under an equal budget of simulated designs
          per run, standing in for print-and-test cycles. Tokens, dollars and wall time are reported separately.
        </p>
        <p>
          The primary endpoint, S1, is speed to a good design: how many simulated designs a run needs before its best
          training success reaches a pre-registered threshold, with runs that never reach it censored at the budget.
          Held-out success (H1) is secondary. CPU MuJoCo is the executor of record. The pre-registered runs are set
          to use Nebius CPU instances; the pilot ran the same CPU executor on a Nebius GPU machine type, as the{" "}
          <Link className="link" href="/compute/">compute page</Link> shows.
        </p>
        <p>
          Each run freezes its best design before any held-out trial. The held-out suite uses object sizes and poses
          the search never saw, plus real household objects from Google Scanned Objects (CC BY 4.0). The hypothesis,
          budget, number of runs, seeds and analysis script are written down and tagged before the main runs start, and
          the result is published whether it is a win, a tie or a loss.
        </p>
        <p>
          The <Link className="link" href="/results/">results page</Link> shows the run-level estimates; the{" "}
          <Link className="link" href="/compute/">compute page</Link> lists the jobs behind them.
        </p>
      </Section>

      <Section id="infrastructure" title="Where it runs">
        <ul className="grid gap-2">
          <li>
            <span className="text-ink">Nebius Token Factory</span> serves NVIDIA Nemotron 3 Super, which proposes the
            designs. Every request and response is cached by hash, so the search can be replayed without API calls.
          </li>
          <li>
            <span className="text-ink">Nebius Serverless Jobs</span> run the scored MuJoCo trials from one pinned
            container image, writing results to Nebius Object Storage.
          </li>
          <li>
            <span className="text-ink">This site</span> is static. Replays are the simulator&apos;s own body poses at
            30 frames per second, drawn in your browser.
          </li>
        </ul>
      </Section>

      <Section id="limitations" title="Limitations">
        <ul className="grid list-disc gap-2 pl-5 marker:text-ink-3">
          <li>Everything here is simulated. No printed finger has been tested on hardware yet.</li>
          <li>Fingers are rigid and approximated by boxes; printed PLA bends a little under load.</li>
          <li>The controller is open loop: it does not sense contact or correct a slipping grasp.</li>
          <li>
            One friction coefficient governs every object contact; real finger pads and objects vary.
          </li>
          <li>
            Bouts recorded during development ran on a laptop, not on Nebius. Each bout&apos;s provenance chips say
            where it ran; only Nebius runs feed the results.
          </li>
          <li>Replays store body poses only, so contact forces are not drawn.</li>
        </ul>
      </Section>

      <TitleBlock title="Method and limitations" drawing="/about" sheet="6 of 6" />
    </>
  );
}
