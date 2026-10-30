/**
 * The smoke scripts and the hosted runner. A build made with NEXT_PUBLIC_RUNNER_URL=http://127.0.0.1:<port>
 * (or http://localhost:<port>) asks that address for /health when the Studio page opens, so a smoke test of
 * such a build starts the mock runner (mock_runner.mjs) on that port first. With the variable unset, which
 * is how the public site is built, nothing starts and the scripts behave exactly as before.
 */

import { startMockRunner } from "./mock_runner.mjs";

/** The two names a build may be given for the mock, which listens on the loopback address. */
export const MOCK_HOSTS = ["127.0.0.1", "localhost"];

/** Whether the environment asks for a runner build. */
export function runnerUrlFromEnv(env = process.env) {
  const v = (env.NEXT_PUBLIC_RUNNER_URL ?? "").trim();
  return v === "" ? null : v;
}

/** Start the mock on the port the build points to; null when no runner is configured. */
export async function startMockFromEnv(env = process.env) {
  const raw = runnerUrlFromEnv(env);
  if (raw === null) return null;
  const url = new URL(raw);
  if (!MOCK_HOSTS.includes(url.hostname)) {
    throw new Error(`NEXT_PUBLIC_RUNNER_URL=${raw} is not a local address, so the smoke test cannot answer for it; point it at http://127.0.0.1:<port> or unset it`);
  }
  return startMockRunner({ port: Number(url.port || 80) });
}

/**
 * Whether href is the runner's own report redirect for runId, as the result card links it (a presigned link
 * would expire after 15 minutes). The host is whatever name the build was given for the mock, 127.0.0.1 or
 * localhost, so either passes; the scheme, the port, the path and the absence of a query, a fragment and
 * credentials are exact.
 */
export function isReportRedirect(href, runId, port) {
  let link;
  try {
    link = new URL(href ?? "");
  } catch {
    return false;
  }
  return (
    link.protocol === "http:" &&
    MOCK_HOSTS.includes(link.hostname) &&
    Number(link.port || 80) === port &&
    link.username === "" &&
    link.password === "" &&
    link.pathname === `/runs/${runId}/report` &&
    link.search === "" &&
    link.hash === ""
  );
}
