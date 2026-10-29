import type { Plan, Tier } from "@/components/studio/schema";
import { TIER_LABELS, TIER_NOTES } from "@/components/studio/schema";
import { type BarSegment, failurePhrase, noteLead, ruledOutGroups, type StepStory, type ToolOutcome } from "@/components/studio/story";

import { FamilyIcon } from "./FamilyIcon";
import { SeekButton, StepFrame } from "./StoryPlayback";
import { Term } from "./Term";

const SEGMENT_TONE: Record<BarSegment["kind"], string> = {
  passed: "bg-pass",
  partial: "bg-pending",
  failed: "bg-fail",
  ruled_out: "bg-muted-bar",
};

/** Failure segments get lighter as they get rarer, so the main cause reads first. */
function segmentStyle(seg: BarSegment, i: number, failedIndex: number): React.CSSProperties | undefined {
  if (seg.kind !== "failed") return undefined;
  return { opacity: Math.max(0.45, 1 - failedIndex * 0.2) };
}

/** The whole field of tools as one bar: passed, failed by cause, ruled out before physics. */
export function FailureBar({ bar, total }: { bar: BarSegment[]; total: number }) {
  let failed = -1;
  const styled = bar.map((seg, i) => {
    if (seg.kind === "failed") failed += 1;
    return { seg, style: segmentStyle(seg, i, failed) };
  });
  return (
    <div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-paper-2" role="img" aria-label={bar.map((b) => `${b.tools} ${b.label}`).join(", ")}>
        {styled.map(({ seg, style }) => (
          <span
            key={seg.key}
            className={`h-full border-r border-surface last:border-r-0 ${SEGMENT_TONE[seg.kind]}`}
            style={{ width: `${(seg.tools / total) * 100}%`, ...style }}
          />
        ))}
      </div>
      <ul className="mt-2.5 grid gap-x-4 gap-y-1 text-sm sm:grid-cols-2">
        {styled.map(({ seg, style }) => (
          <li key={seg.key} className="flex items-baseline gap-2">
            <span className={`inline-block size-2.5 shrink-0 translate-y-px rounded-sm ${SEGMENT_TONE[seg.kind]}`} style={style} aria-hidden="true" />
            <span className="num w-6 shrink-0 text-right text-ink">{seg.tools}</span>
            <span className={seg.kind === "passed" ? "text-ink" : "text-ink-2"}>{seg.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ToolRow({ t, detail, tone }: { t: ToolOutcome; detail: string; tone: string }) {
  return (
    <li className="flex items-start gap-2 py-1">
      <FamilyIcon family={t.family_title} className={`mt-0.5 size-4 shrink-0 ${tone}`} />
      <span className="min-w-0">
        <span className="block text-ink">{t.display_name}</span>
        <span className="block text-xs text-ink-3">
          {t.family_title}
          {detail ? `, ${detail}` : ""}
        </span>
        {t.rating ? (
          <span className="block text-xs text-ink-2" data-rating>
            {t.rating}
          </span>
        ) : null}
      </span>
    </li>
  );
}

/** "See which tools": every candidate on this step, by outcome. */
function WhichTools({ story, plan, pickId, pickWord }: { story: StepStory; plan: Plan; pickId: string | null; pickWord: string }) {
  const groups = ruledOutGroups(story.ruledOut);
  const tried = [...story.partial, ...story.failed];
  return (
    <details className="group mt-4 rounded-lg border border-rule" data-which-tools>
      <summary className="flex list-none items-center justify-between gap-3 px-3 py-2 text-sm text-ink-2 hover:text-ink [&::-webkit-details-marker]:hidden">
        <span>See which tools ({story.n_candidates})</span>
        <svg viewBox="0 0 12 12" className="size-3 transition-transform group-open:rotate-180" aria-hidden="true">
          <path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </summary>
      <div className="grid gap-5 border-t border-rule px-3 py-3 text-sm md:grid-cols-3">
        <section>
          <h4 className="text-sm font-semibold text-pass">Passed every screen run ({story.passed.length})</h4>
          <ul className="mt-1">
            {story.passed.map((t) => (
              <ToolRow key={t.effector_id} t={t} tone="text-pass" detail={t.effector_id === pickId ? pickWord : `${t.successes} of ${t.n} runs`} />
            ))}
            {story.passed.length === 0 ? <li className="text-ink-3">None.</li> : null}
          </ul>
        </section>
        <section>
          <h4 className="text-sm font-semibold text-fail">Tried and failed ({tried.length})</h4>
          <ul className="mt-1">
            {tried.map((t) => (
              <ToolRow
                key={t.effector_id}
                t={t}
                tone={t.status === "partial" ? "text-pending" : "text-fail"}
                detail={`${t.successes} of ${t.n} runs${t.dominant ? `, ${failurePhrase(t.dominant, story.step, plan)}` : ""}`}
              />
            ))}
            {tried.length === 0 ? <li className="text-ink-3">None.</li> : null}
          </ul>
        </section>
        <section>
          <h4 className="text-sm font-semibold text-ink-2">Ruled out before physics ({story.ruledOut.length})</h4>
          <div className="mt-1 grid gap-3">
            {groups.map((g) => (
              <div key={g.group}>
                <p className="text-xs text-ink-3">
                  {g.group} ({g.tools.length})
                </p>
                <ul>
                  {g.tools.map((t) => (
                    <ToolRow key={t.effector_id} t={t} tone="text-ink-3" detail="" />
                  ))}
                </ul>
              </div>
            ))}
            {groups.length === 0 ? <p className="text-ink-3">None.</p> : null}
          </div>
        </section>
      </div>
    </details>
  );
}

function TierTip({ tier }: { tier: Tier }) {
  const tone = tier === "rigid" ? "border-rule-strong text-ink-2" : tier === "approximation" ? "border-pending/70 bg-pending-soft text-pending" : "border-dashed border-fail text-fail";
  return (
    <Term label={`${TIER_LABELS[tier]} fidelity`} tip={TIER_NOTES[tier]}>
      <span className={`inline-flex items-center rounded-full border px-2 py-px text-xs ${tone}`} data-fidelity={tier}>
        {TIER_LABELS[tier]}
      </span>
    </Term>
  );
}

/** The step's caveats: what this run recorded, then the tool's modelling caveat (full text on open). */
function StepNotes({ notes }: { notes: StepStory["notes"] }) {
  if (!notes.length) return null;
  return (
    <ul className="mt-3 grid gap-1.5 border-l-2 border-pending pl-2.5 text-sm" data-step-notes aria-label="Caveats">
      {notes.map((n) =>
        n.detail ? (
          <li key={n.label} data-note-kind={n.kind}>
            <details className="group">
              <summary className="cursor-pointer list-none text-pending [&::-webkit-details-marker]:hidden">
                <span className="font-semibold">{noteLead(n)}</span>
                {n.label}
                <span className="ml-1 text-xs text-ink-3 group-open:hidden">(more)</span>
              </summary>
              <p className="mt-1 text-xs text-ink-2">{n.detail}</p>
            </details>
          </li>
        ) : (
          <li key={n.label} className="text-pending" data-note-kind={n.kind}>
            <span className="font-semibold">{noteLead(n)}</span>
            {n.label}
          </li>
        ),
      )}
    </ul>
  );
}

function needsOf(story: StepStory, plan: Plan): [string, string][] {
  const s = story.step;
  const obj = plan.objects.find((o) => o.name === s.object);
  const rows: [string, string][] = [];
  rows.push(["Skill", s.primitive_title]);
  const feature = obj && typeof obj.params.feature === "string" ? obj.params.feature.replace(/_/g, " ") : null;
  rows.push(["Object", `${obj?.label ?? s.object}${s.handle === "handle" && feature ? `, by its ${feature}` : ""}`]);
  const target = s.target ? plan.objects.find((o) => o.name === s.target) : null;
  if (target) rows.push(["Into", target.label]);
  const limits: string[] = [];
  if (s.constraints.max_force_n !== null) limits.push(`at most ${s.constraints.max_force_n} N`);
  if (s.constraints.max_torque_nm !== null) limits.push(`at most ${s.constraints.max_torque_nm} N m`);
  if (s.constraints.precision_mm !== null) limits.push(`within ${s.constraints.precision_mm} mm`);
  if (s.constraints.precision_deg !== null) limits.push(`within ${s.constraints.precision_deg} deg`);
  if (limits.length) rows.push(["Limit", limits.join(", ")]);
  if (s.success.length) rows.push(["Done when", s.success.map((c) => c.text).join("; ")]);
  return rows;
}

/** One step of the plan: what it takes, who could do it, the pick and why, and who failed and how. */
export interface HeldoutStep {
  n: number;
  successes: number;
  failure_modes: Record<string, number>;
}

/** The best single tool on this step over the held-out conditions, in words, or null. */
function heldoutText(h: HeldoutStep | undefined, story: StepStory, plan: Plan): string | null {
  if (!h || h.n === 0) return null;
  const modes = Object.entries(h.failure_modes)
    .filter(([m, k]) => k > 0 && m !== "not_reached")
    .sort((a, b) => b[1] - a[1]);
  const reached = h.n - (h.failure_modes.not_reached ?? 0);
  if (reached === 0) return null;
  const parts = modes.map(([m, k]) => `${failurePhrase(m, story.step, plan)} in ${k}`);
  const fails = parts.length ? `: ${parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0]}` : "";
  return `On the ${h.n} held-out conditions it passed ${h.successes} of ${reached}${fails}.`;
}

export function StepCard({
  story,
  plan,
  nSteps,
  kind,
  role = "recommended",
  heldoutBaseline,
}: {
  story: StepStory;
  plan: Plan;
  nSteps: number;
  kind: string;
  /** Whose tool the card shows: the recommendation's, or the closest setup's when nothing completes. */
  role?: "recommended" | "closest";
  /** The best single tool's held-out result on this step (use-case benchmark v1), when the page has it. */
  heldoutBaseline?: HeldoutStep;
}) {
  const held = heldoutText(heldoutBaseline, story, plan);
  const pick = story.pick;
  const ls = story.laneStep;
  const ratio = `${story.passed.length} of ${story.tried}`;
  const toolNo = pick ? pick.tool + 1 : null;
  const closest = role === "closest";
  const toolWord = (kind === "multi_head" ? `Head tool ${toolNo}` : kind === "tool_change" ? `Tool ${toolNo} of the change` : "The tool") + (closest ? " of the closest setup" : "");
  const stopsHere = closest && pick !== null && story.why[0]?.startsWith("This is where it stops");
  return (
    <StepFrame id={story.step.id} start={ls?.start_s ?? null} end={ls ? ls.start_s + ls.duration_s : null}>
      <header className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-x-4 gap-y-3 border-b border-rule px-4 py-4 sm:grid-cols-[auto_minmax(0,1fr)_auto] sm:px-5">
        <span className="num grid size-9 shrink-0 place-items-center rounded-md border border-rule-strong text-base text-ink" aria-hidden="true">
          {story.index + 1}
        </span>
        <div className="min-w-0">
          <p className="text-xs text-ink-3">
            Step {story.index + 1} of {nSteps}
          </p>
          <h3 className="heading text-xl text-ink sm:text-2xl">{story.name}</h3>
          <p className="mt-1 text-sm text-ink-2">{story.step.description}</p>
        </div>
        {ls ? (
          <div className="col-span-2 sm:col-span-1">
            <SeekButton start={ls.start_s} label="Watch this step" />
          </div>
        ) : null}
      </header>

      <dl className="grid grid-cols-2 gap-px bg-rule text-sm sm:grid-cols-4" aria-label="What the step takes">
        {needsOf(story, plan).map(([k, v]) => (
          <div key={k} className="min-w-0 bg-surface px-4 py-2.5 sm:px-5">
            <dt className="text-xs text-ink-3">{k}</dt>
            <dd className="text-ink">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-6 px-4 py-5 sm:px-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-8">
        <section aria-label="Who could do it">
          <p className="flex flex-wrap items-baseline gap-x-3">
            <span className="num text-[2.25rem] leading-none text-ink">{ratio}</span>
            <span className="text-ink-2">tools tried in physics could {story.name.charAt(0).toLowerCase() + story.name.slice(1)}</span>
          </p>
          <p className="mt-1 text-sm text-ink-3">
            {story.ruledOut.length} more of the {story.n_candidates} were ruled out before physics; {story.runs} screen runs in all.
          </p>
          <div className="mt-4">
            <FailureBar bar={story.bar} total={story.n_candidates} />
          </div>
          {story.baseline ? (
            <p
              className={`mt-4 rounded-md border-l-2 bg-baseline-soft px-3 py-2 text-sm ${
                story.baseline.tone === "fail" ? "border-fail" : story.baseline.tone === "pass" ? "border-pass" : "border-rule-strong"
              }`}
              data-baseline-step
            >
              <span className="font-semibold text-baseline">Best single tool: </span>
              <span className="text-ink-2">{story.baseline.text}</span>
              {held ? <span className="mt-1 block text-ink-2" data-heldout-step>{held}</span> : null}
            </p>
          ) : null}
        </section>

        {pick && story.reusedFrom !== null ? (
          <section aria-label={closest ? "The closest setup's tool" : "The recommendation's tool"} className="min-w-0 self-start rounded-lg border border-rule p-4" data-pick-role={role}>
            <p className="text-sm text-ink-3">{toolWord}, kept from step {story.reusedFrom + 1}</p>
            <p className="mt-1 flex items-start gap-2">
              <FamilyIcon family={pick.family_title ?? ""} className="mt-0.5 size-5 shrink-0 text-accent" />
              <span className="heading text-base text-ink">{pick.tool_label}</span>
            </p>
            <p className="mt-2 flex flex-wrap items-center gap-2 text-sm text-ink-2">
              <span className="num text-ink">{Math.round(pick.success_rate * 100)}%</span> of full-task runs passed this step
              <TierTip tier={pick.fidelity} />
            </p>
            <StepNotes notes={story.notes} />
            <p className="mt-2 text-sm text-ink-3">Already on the wrist, so there is no change here; it was chosen at step {story.reusedFrom + 1} for every step it covers.</p>
            {closest && story.why.length ? <p className={`mt-2 text-sm ${stopsHere ? "text-fail" : "text-ink-2"}`}>{story.why[0]}</p> : null}
          </section>
        ) : pick ? (
          <section
            aria-label={closest ? "The closest setup's tool" : "The recommendation's tool"}
            className={`min-w-0 rounded-lg border p-4 ${closest ? (stopsHere ? "border-fail/50 bg-fail-soft" : "border-rule-strong bg-surface") : "border-accent/40 bg-accent-soft"}`}
            data-pick-role={role}
          >
            <p className={`text-sm ${closest ? "text-ink-2" : "text-accent"}`}>{toolWord}</p>
            <p className="mt-1 flex items-start gap-2">
              <FamilyIcon family={pick.family_title ?? ""} className="mt-0.5 size-6 shrink-0 text-accent" />
              <span className="heading text-lg text-ink">{pick.tool_label}</span>
            </p>
            {pick.rating ? (
              <p className="mt-1 text-sm text-ink-2" data-rating>
                {pick.rating}
              </p>
            ) : null}
            <p className="mt-2 flex flex-wrap items-center gap-2 text-sm text-ink-2">
              <span className="num text-ink">{Math.round(pick.success_rate * 100)}%</span> of full-task runs passed this step
              <TierTip tier={pick.fidelity} />
            </p>
            <StepNotes notes={story.notes} />
            {story.why.length ? (
              <>
                <h4 className="mt-4 text-sm font-semibold text-ink">{closest ? "How it did" : "Why this tool"}</h4>
                <ul className="mt-1.5 grid gap-2 text-[0.9375rem] text-ink-2">
                  {story.why.map((w) => (
                    <li key={w} className="flex gap-2">
                      <span className="mt-2 size-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {story.whyGap ? (
              <p className="mt-3 border-t border-accent/20 pt-3 text-sm text-ink-3" data-why-gap>
                <span className="font-semibold text-ink-2">What the data cannot say: </span>
                {story.whyGap}
              </p>
            ) : null}
          </section>
        ) : null}
      </div>
      <div className="px-4 pb-4 sm:px-5">
        <WhichTools story={story} plan={plan} pickId={pick?.effector_id ?? null} pickWord={closest ? "the closest setup's tool" : "the recommendation's pick"} />
      </div>
    </StepFrame>
  );
}
