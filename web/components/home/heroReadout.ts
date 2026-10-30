import { changeText, laneAt, type StudioLane } from "@/components/studio/schema";

export interface HeroStep {
  id: string;
  name: string;
  description: string;
}

/** What the home hero's step readout shows at one time on the replay clock. */
export interface HeroReadout {
  /** Index of the step on screen, or -1. */
  current: number;
  /** The run has ended (every step bar turns green). */
  done: boolean;
  /** The readout's data-hero-step: the step's id, or the moment's kind. */
  moment: string;
  title: string;
  detail: string;
}

/**
 * The readout at time `t`. It changes only at step boundaries, tool changes and the end of the
 * run, so the hero sets React state a few times per loop instead of on every animation frame.
 */
export function heroReadout(lane: StudioLane, steps: readonly HeroStep[], t: number): HeroReadout {
  const m = laneAt(lane, t);
  const n = steps.length;
  const current = m.kind === "step" ? steps.findIndex((s) => s.id === m.step.step_id) : -1;
  const done = m.kind === "ended" || (t >= lane.duration_s - 0.05 && lane.completed);
  const toolOf = (toolIndex: number) => lane.segments.find((s) => s.tool === toolIndex)?.tool_label ?? "";

  let title = "";
  let detail = "";
  if (m.kind === "step" && current >= 0) {
    title = `Step ${current + 1} of ${n}: ${steps[current]!.name}`;
    detail = `${toolOf(m.step.tool)}. ${steps[current]!.description}`;
  } else if (m.kind === "change" && m.change === "present") {
    title = `Head switch to tool ${m.to.tool + 1}`;
    detail = `${m.to.tool_label}. A fixed ${m.change_s} s tilt of the wrist, not simulated.`;
  } else if (m.kind === "change") {
    title = "Changing tool";
    detail = `To ${m.to.tool_label}. ${changeText(m.change, m.change_s)}.`;
  } else if (done) {
    title = lane.completed && lane.cycle_time_s !== null ? `Done in ${lane.cycle_time_s.toFixed(1)} s` : "Stopped";
    detail = lane.completed ? `${n} of ${n} steps passed on this recorded run.` : "This recorded run did not finish the task.";
  }
  return { current, done, moment: current >= 0 ? steps[current]!.id : m.kind, title, detail };
}

export function sameReadout(a: HeroReadout, b: HeroReadout): boolean {
  return a.current === b.current && a.done === b.done && a.moment === b.moment && a.title === b.title && a.detail === b.detail;
}
