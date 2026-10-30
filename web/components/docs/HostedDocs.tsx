import Link from "next/link";

import { DocSection, Note } from "@/components/docs/DocsShell";
import { PRODUCT_MODE_LABEL } from "@/components/studio/schema";
import { KEY_PRIVACY_LINE, RUN_CAP_LINE, runnerConfigured } from "@/lib/runner";

/** The docs section id and title for the hosted runner; the pages list it in their "On this page" only when it is on. */
export const HOSTED_SECTION = { id: "hosted", title: "Plan and run it on this site" } as const;

/** The sections of a docs page, with the hosted one added at the end in a build that has the runner. */
export function withHosted<T extends { id: string; title: string }>(sections: readonly T[]): (T | typeof HOSTED_SECTION)[] {
  return runnerConfigured() ? [...sections, HOSTED_SECTION] : [...sections];
}

/**
 * "Plan and run it on this site": the providers, the caps and the key privacy line. It renders
 * nothing unless this build has the hosted runner (NEXT_PUBLIC_RUNNER_URL), so the docs never
 * promise a feature the site does not have.
 */
export function HostedDocSection({ n, brief = false }: { n: number; brief?: boolean }) {
  if (!runnerConfigured()) return null;
  return (
    <DocSection id={HOSTED_SECTION.id} n={n} title={HOSTED_SECTION.title}>
      <p>
        The{" "}
        <Link className="link" href="/studio/#plan">
          Studio
        </Link>{" "}
        can plan your own words live and then test the plan in physics, with nothing to install. In the panel called &ldquo;Plan and run it live&rdquo;, choose who pays
        for the planning call:
      </p>
      <ul className="grid gap-2 text-sm sm:grid-cols-3">
        <li className="panel p-3">
          <span className="block text-ink">Your NVIDIA API key</span>
          <span className="mt-1 block text-xs">Nemotron plans on your own account.</span>
        </li>
        <li className="panel p-3">
          <span className="block text-ink">Your Nebius Token Factory key</span>
          <span className="mt-1 block text-xs">Nemotron plans on your own account there.</span>
        </li>
        <li className="panel p-3">
          <span className="block text-ink">The free allowance</span>
          <span className="mt-1 block text-xs">A daily allowance of plans that every visitor shares. The panel counts down what is left.</span>
        </li>
      </ul>
      <Note title="Your key">
        {KEY_PRIVACY_LINE} It stays in the page&apos;s memory only, and it is forgotten after each request unless you untick the box.
      </Note>
      {brief ? (
        <p>
          Press <strong className="text-ink">Plan live</strong>, check the plan in the steps, then press <strong className="text-ink">Run in physics</strong>.{" "}
          <Link className="link" href="/docs/tutorials/plan-a-task/#hosted">
            The tutorial
          </Link>{" "}
          has the details.
        </p>
      ) : (
        <>
          <p>
            Press <strong className="text-ink">Plan live</strong>. Planning usually takes 15 to 45 seconds and can take up to five minutes. The plan loads into the steps below the
            panel exactly as a recorded one does, so you can check and edit it. When the plan is valid, press <strong className="text-ink">Run in physics</strong>. The run goes
            to a hosted runner, which can take up to three minutes to accept a plan it has not seen before. The address gets a <span className="num text-ink">#run-</span> token,
            so reloading the page picks the run back up.
          </p>
          <p>
            {RUN_CAP_LINE}, and the free allowance has its own daily cap. When a cap is reached, the page says which one and when it resets. If the runner is busy, the page says to try
            again in a minute. If the hosted runner is off, the page says so and points to{" "}
            <Link className="link" href="/docs/tutorials/run-it-yourself/">
              Run it yourself
            </Link>
            .
          </p>
          <p>
            A finished run shows a summary card: the recommended setup and how often it completed the task, the best single tool beside it, a table of the steps, and a
            link to the full report. Replays are shown for the recorded examples; hosted runs show the summary. Simulated, verify on hardware. {PRODUCT_MODE_LABEL}.
          </p>
        </>
      )}
    </DocSection>
  );
}
