/**
 * What each visitor may do on the hosted runner each day, in one place. The runner enforces these
 * (`Caps` in src/rtt/runner/config.py); the page states them before anything is refused, so a visitor who
 * reaches one does not take it for a broken app. Every sentence that quotes a number takes it from here
 * (lib/live_copy.ts), and tests/runner_limits.test.ts reads config.py and fails when a default there differs,
 * so the page never promises more than the runner gives.
 *
 * These are per visitor and per UTC day. The counts GET /health gives (free plans left, runs left) are for
 * the whole site, and a count above zero is not a promise that a visitor still has any of these left.
 */
export const RUNNER_TRIAL_LIMITS = {
  /** Plans on the free allowance, per visitor per day (`free_plans_per_ip_per_day`). */
  freePlansPerVisitor: 3,
  /** Plans on the visitor's own key, per visitor per day (`keyed_plans_per_ip_per_day`). */
  keyedPlansPerVisitor: 30,
  /** Physics runs, per visitor per day (`runs_per_ip_per_day`). */
  runsPerVisitor: 2,
} as const;
