import type { Metadata } from "next";
import Link from "next/link";

import { LinkCard } from "@/components/LinkCard";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { getResults } from "@/lib/data";
import { repoBlob } from "@/lib/site";

import { getResultJobs } from "../compute/nebius";
import { getStudioIndex } from "../studio/data";
import { getHeldout } from "../studio/heldout";
import { V1_NOTE } from "./notes";

export const metadata: Metadata = {
  title: "Evidence",
  description: "The pre-declared tests behind Effector Lab's claims, their methods, the effector tournament and every compute job.",
};

function pText(p: number): string {
  return p < 0.001 ? p.toExponential(1) : p.toFixed(4);
}

export default function EvidencePage() {
  const held = getHeldout();
  const heroes = getStudioIndex()?.heroes ?? [];
  const s1 = getResults().s1.analysis ?? null;
  const jobs = getResultJobs();
  const v1Job = jobs.jobs.find((j) => j.result === "use-case benchmark v1") ?? null;

  return (
    <>
      <PageHeader
        title="Evidence"
        lede={<p>Every number on this site comes from a committed result file. These are the tests behind the claims, how they were run, and what they cost.</p>}
      />

      <section className="mt-12 grid gap-4 lg:grid-cols-2" aria-label="Headline results">
        <article className="panel min-w-0 p-5 sm:p-6" data-evidence="usecase">
          <p className="text-sm text-ink-3">Studio benchmark, use-case benchmark v1 (top-down wrist, 19 Oct), pre-declared</p>
          <h2 className="heading mt-1 text-xl text-ink">Does the Studio&apos;s pick finish the job more often than the best single tool?</h2>
          <p className="mt-4 flex flex-wrap items-baseline gap-x-3">
            <span className="num text-4xl text-accent">
              {held.totals.recommended} of {held.totals.n}
            </span>
            <span className="text-ink-2">held-out runs finished by the Studio&apos;s pick, against</span>
            <span className="num text-2xl text-baseline">{held.totals.best_single}</span>
            <span className="text-ink-2">by the best single tool.</span>
          </p>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[22rem] text-left text-sm">
              <thead>
                <tr className="border-b border-rule text-xs text-ink-3">
                  <th className="py-2 pr-3 font-normal">Use case</th>
                  <th className="py-2 pr-3 font-normal">Studio pick</th>
                  <th className="py-2 pr-3 font-normal">Best single</th>
                  <th className="py-2 font-normal">p (Holm)</th>
                </tr>
              </thead>
              <tbody>
                {Object.values(held.heroes).map((h) => {
                  // linked only to a story page that still shows the setups this v1 result tested
                  const hero = heroes.find(
                    (x) => x.case_id === h.case_id && x.recommended?.setup_id === h.recommended.setup_id && x.best_single?.setup_id === h.best_single.setup_id,
                  );
                  return (
                    <tr key={h.case_id} className="border-b border-rule last:border-b-0">
                      <th scope="row" className="py-2 pr-3 font-normal">
                        {hero ? (
                          <Link className="link" href={`/studio/${hero.hero}/`}>
                            {h.title}
                          </Link>
                        ) : (
                          h.title
                        )}
                      </th>
                      <td className="num py-2 pr-3 text-ink">
                        {h.recommended.completed} of {h.recommended.n}
                      </td>
                      <td className="num py-2 pr-3 text-ink">
                        {h.best_single.completed} of {h.best_single.n}
                      </td>
                      <td className="num py-2 text-ink-2">{pText(h.p_holm)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-ink-3">
            {held.test}; each condition is a new seed, friction and hidden placement error. {held.label}.
          </p>
          <p className="mt-4 text-sm text-ink-2">
            A dated note on this result follows below.{" "}
            <a className="link" href="#v1-note">
              Read the note
            </a>
            .
          </p>
          {v1Job ? (
            <p className="num mt-3 text-xs text-ink-3">
              One Nebius job, {v1Job.id}, {v1Job.region} {v1Job.platform}
              {v1Job.wall_s !== null ? `, ${v1Job.wall_s} s` : ""}
              {v1Job.cost_usd !== null ? `, about $${v1Job.cost_usd.toFixed(2)}` : ""}.
            </p>
          ) : null}
          <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm">
            <a className="link" href={repoBlob("docs/USECASE_V1_RESULTS.md")} rel="noopener">
              Full results
            </a>
            <a className="link" href={repoBlob("docs/USECASE_V1.md")} rel="noopener">
              Protocol
            </a>
            <a className="link" href={repoBlob(held.source)} rel="noopener">
              Analysis file
            </a>
          </p>
        </article>

        {s1 ? (
          <article className="panel min-w-0 p-5 sm:p-6" data-evidence="fingers">
            <p className="text-sm text-ink-3">Finger designer, pre-registered (prereg-v1)</p>
            <h2 className="heading mt-1 text-xl text-ink">Does a language-model designer with feedback find a good finger in fewer designs?</h2>
            <p className="mt-4 flex flex-wrap items-baseline gap-x-3">
              <span className="num text-4xl text-accent">{s1.effect.diff.toFixed(2)}</span>
              <span className="text-ink-2">fewer designs on average than equal-budget random search</span>
            </p>
            <p className="mt-4 text-sm text-ink-2">
              {Math.round(s1.effect.level * 100)}% interval {s1.effect.ci_low.toFixed(2)} to {s1.effect.ci_high.toFixed(2)}; {s1.primary.method}, p ={" "}
              {s1.primary.p_value.toFixed(4)}. Sensitivity ({s1.logrank.method}): p = {s1.logrank.p_value.toFixed(4)}.
            </p>
            <p className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm">
              <Link className="link" href="/results/">
                Results page
              </Link>
              <a className="link" href={repoBlob("docs/PREREG.md")} rel="noopener">
                Pre-registration
              </a>
              <a className="link" href={repoBlob("docs/DEVIATIONS.md")} rel="noopener">
                Deviations
              </a>
            </p>
          </article>
        ) : null}
      </section>

      <section className="mt-4 grid gap-4 lg:grid-cols-2" aria-label="The v1 note and the next result">
        <article id="v1-note" className="panel min-w-0 scroll-mt-24 p-5 sm:p-6" data-evidence="v1-note" data-v1-note>
          <p className="text-sm text-ink-3">Note on use-case benchmark v1, added {V1_NOTE.date}</p>
          <h2 className="heading mt-1 text-xl text-ink">What v1 could not test</h2>
          <p className="mt-3 text-ink-2" data-v1-note-lead>
            {V1_NOTE.lead}
          </p>
          <ul className="mt-4 grid gap-2.5 text-sm text-ink-2">
            {V1_NOTE.points.map((pt) => (
              <li key={pt.title} className="flex gap-2">
                <span className="mt-2 size-1.5 shrink-0 rounded-full bg-pending" aria-hidden="true" />
                <span>
                  <span className="font-semibold text-ink">{pt.title}: </span>
                  {pt.text}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-sm text-ink-3">The tagged protocol, code and numbers stay as published; a note never changes a result.</p>
        </article>
        <article className="min-w-0 rounded-[10px] border border-dashed border-rule-strong p-5 sm:p-6" data-evidence="usecase-v2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-ink-3">Studio benchmark, use-case benchmark v2</p>
            <span className="inline-flex items-center gap-2 rounded-full border border-pending/70 bg-pending-soft px-2.5 py-0.5 text-sm text-pending" data-v2-status="pending">
              <span className="size-1.5 rounded-full bg-pending" aria-hidden="true" />
              v2 pre-registration: results pending
            </span>
          </div>
          <h2 className="heading mt-2 text-xl text-ink">Next: six real jobs, a wrist that tilts, success tests the physics can meet</h2>
          <p className="prose-measure mt-3 text-ink-2">
            The next benchmark will ask the same question on six jobs from the{" "}
            <Link className="link" href="/atlas/">
              use-case atlas
            </Link>
            . It will fix both v1 limits: the simulated wrist will tilt, and every success test will have a tolerance the physics can meet. Its
            protocol, arms and secret will be frozen and tagged before any held-out run, and its results will replace this card.
          </p>
        </article>
      </section>

      <section className="mt-16" aria-labelledby="more-title">
        <h2 id="more-title" className="heading text-[1.5rem] text-ink">
          Methods, runs and compute
        </h2>
        <ul className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <li>
            <LinkCard href="/effectors/" title="Effector tournament" text="Hands, grippers, a soft arm, a suction cup and designed effectors on the same parts and trials. Product mode." />
          </li>
          <li>
            <LinkCard href="/benchmark/" title="Recorded bouts" text="Every recorded bout: finger designs tested against the same object, gripper and controller, replayed from the pose log." />
          </li>
          <li>
            <LinkCard href="/results/" title="Finger designer results" text="The pre-registered endpoints, the held-out suites and the planned splits." />
          </li>
          <li>
            <LinkCard href="/about/" title="Method and limitations" text="How grasps are scored, how the search is compared, and what this does not show." />
          </li>
          <li>
            <LinkCard
              href="/compute/"
              title="Compute log"
              text="Every Nebius job behind the published numbers: where it ran, for how long and what it cost."
              meta={jobs.ledger ? `${jobs.jobs.length} result jobs; project ledger $${jobs.ledger.usd.toFixed(2)}` : undefined}
            />
          </li>
          <li>
            <LinkCard href={repoBlob("docs/STUDIO_SEARCH.md")} external title="How the Studio ranks setups" text="The screen, the combination search, the finalists and the ranking rule." />
          </li>
          <li>
            <LinkCard href="/atlas/" title="Use-case atlas" text="The researched jobs behind the benchmark's use cases, with the setup an integrator would expect for each." />
          </li>
        </ul>
      </section>

      <TitleBlock title="Evidence" drawing="/evidence" sheet="Evidence" />
    </>
  );
}
