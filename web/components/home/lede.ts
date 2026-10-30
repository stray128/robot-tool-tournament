/**
 * The home page's lede. The hosted runner (NEXT_PUBLIC_RUNNER_URL) changes only its last sentence
 * (kept in lib/live_copy.ts with the other live-planning sentences); a build without the runner
 * says what it always said.
 */

import { LIVE_COPY, liveCopy } from "@/lib/live_copy";

export const LEDE_HEAD =
  "Describe the job your robot has to do. Effector Lab plans it as steps, tests every end effector it can simulate on every step, and recommends one tool, a tool change or a two-tool head, next to the best single off-the-shelf tool. ";

export const LEDE_TAIL = LIVE_COPY.lede.local;

export const LEDE_TAIL_HOSTED = LIVE_COPY.lede.hosted;

export function homeLede(hosted: boolean): string {
  return LEDE_HEAD + liveCopy("lede", hosted);
}
