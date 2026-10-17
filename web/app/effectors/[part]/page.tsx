import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { CertificateMatrix, FamilyCard, RecommendationCard } from "@/components/effectors/Cards";
import EffectorLanes from "@/components/effectors/EffectorLanes";
import { FidelityBadge } from "@/components/effectors/FidelityBadge";
import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import {
  FAMILY_TITLES,
  GROUP_TITLES,
  massText,
  OUTCOME_LABELS,
  pct,
  type RequirementBlock,
  REQUIREMENT_LABELS,
  REQUIREMENT_NOTES,
  score,
  toleranceText,
} from "@/lib/effectors";

import { getEffectorPage, getEffectorPartIds } from "../data";

export const dynamicParams = false;

export function generateStaticParams() {
  return getEffectorPartIds().map((part) => ({ part }));
}

type Params = { params: Promise<{ part: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const page = getEffectorPage((await params).part);
  return page
    ? {
        title: `Which effector for the ${page.part.title.toLowerCase()}?`,
        description: "Product mode: every effector family on the same placements and seeds, one recommendation with its certificate.",
      }
    : {};
}

function NoWinner({ block }: { block: RequirementBlock }) {
  if (!block.no_winner) return null;
  return (
    <article className="border border-dashed border-rule-strong p-4" aria-label="No recommendation">
      <p className="text-xs font-semibold uppercase tracking-wide text-fail">No recommendation</p>
      <p className="mt-1 text-ink-2">{block.no_winner.reason}.</p>
      {block.no_winner.closest.length > 0 ? (
        <ul className="mt-3 grid gap-1 text-sm">
          {block.no_winner.closest.map((c) => (
            <li key={c.effector_id}>
              <span className="text-ink">{c.display_name}</span>{" "}
              <span className="text-ink-3">
                ({c.reason ?? "not eligible"}; failures:{" "}
                {Object.entries(c.failure_modes)
                  .map(([m, n]) => `${OUTCOME_LABELS[m] ?? m} ${n}`)
                  .join(", ") || "none"}
                )
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

function RankingTable({ block, requirement }: { block: RequirementBlock; requirement: "held" | "registered" }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[860px] border-collapse text-left text-sm" data-ranking={requirement}>
        <thead>
          <tr className="border-y border-rule-strong text-xs text-ink-3">
            <th scope="col" className="py-2 pr-3 font-normal">#</th>
            <th scope="col" className="py-2 pr-3 font-normal">Effector</th>
            <th scope="col" className="py-2 pr-3 font-normal">Group</th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">Certificate</th>
            <th scope="col" className="py-2 pr-3 font-normal">Tolerance</th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">Screen</th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">Actuators</th>
            <th scope="col" className="py-2 pr-3 text-right font-normal">Mass</th>
            <th scope="col" className="py-2 font-normal">Eligible</th>
          </tr>
        </thead>
        <tbody>
          {block.ranking.map((r, i) => (
            <tr key={r.effector_id} className="border-b border-rule align-top">
              <td className="num py-2 pr-3 text-ink-3">{r.eligible ? i + 1 : "–"}</td>
              <th scope="row" className="py-2 pr-3 font-normal">
                <span className="text-ink">{r.display_name}</span>
                <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
                  {FAMILY_TITLES[r.family]} <FidelityBadge fidelity={r.fidelity} />
                </span>
              </th>
              <td className="py-2 pr-3 text-ink-2">{GROUP_TITLES[r.group]}</td>
              <td className="num py-2 pr-3 text-right">{score(r.score)}</td>
              <td className="num py-2 pr-3 text-xs">{r.certified ? toleranceText(r.tolerance_mm) : "n/a"}</td>
              <td className="num py-2 pr-3 text-right">{pct(r.screen_rate)}</td>
              <td className="num py-2 pr-3 text-right">{r.n_actuators ?? "n/a"}</td>
              <td className="num py-2 pr-3 text-right">{massText(r.mass_kg)}</td>
              <td className="py-2 text-xs">{r.eligible ? <span className="text-pass">yes</span> : <span className="text-ink-3">{r.reason}</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default async function EffectorPartPage({ params }: Params) {
  const page = getEffectorPage((await params).part);
  if (!page) notFound();
  const req = page.requirement;
  const other = req === "registered" ? "held" : "registered";
  const block = page.recommendation;
  const otherBlock = page.by_requirement[other];
  const names = new Map(page.candidates.map((c) => [c.effector_id, c.display_name]));
  const certified = page.candidates.filter((c) => c.certified && page.certificates[c.effector_id]);
  const condition = page.lanes[0];

  return (
    <>
      <p className="pt-8 text-sm">
        <Link href="/effectors/" className="link">
          Which effector
        </Link>
      </p>
      <ProductModeBanner />
      <PageHeader
        title={`Which effector for the ${page.part.title.toLowerCase()}?`}
        lede={
          <p>
            Every available effector family grasped the {page.part.title.toLowerCase()} ({page.part.size_mm.map((v) => v.toFixed(0)).join(" x ")} mm,{" "}
            {massText(page.part.mass_kg)}) on the same four placements and two seeds; the best two of each group were
            certified on the same 18 tests. The headline asks for a <strong>{REQUIREMENT_LABELS[req].toLowerCase()}</strong> grasp:{" "}
            {REQUIREMENT_NOTES[req]}.
          </p>
        }
      />

      <section className="mt-10 grid gap-5 lg:grid-cols-2" aria-labelledby="rec-title">
        <h2 id="rec-title" className="sr-only">
          Recommendation
        </h2>
        {block.winner ? (
          <RecommendationCard card={block.winner} requirement={req} title={`Recommended (${REQUIREMENT_LABELS[req].toLowerCase()})`} />
        ) : (
          <NoWinner block={block} />
        )}
        {block.runner_up ? (
          block.runner_up.card ? (
            <RecommendationCard
              card={block.runner_up.card}
              requirement={req}
              title="Rigid runner-up"
              basis={block.runner_up.basis}
            />
          ) : (
            <article className="border border-dashed border-rule-strong p-4" aria-label="Rigid runner-up">
              <p className="text-xs font-semibold uppercase tracking-wide text-accent">Rigid runner-up</p>
              <p className="mt-1 text-ink-2">{block.runner_up.basis}.</p>
            </article>
          )
        ) : otherBlock.winner ? (
          <RecommendationCard
            card={otherBlock.winner}
            requirement={other}
            title={`If ${REQUIREMENT_LABELS[other].toLowerCase()} is enough`}
          />
        ) : (
          <NoWinner block={otherBlock} />
        )}
      </section>

      <section className="mt-14" aria-labelledby="lanes-title">
        <h2 id="lanes-title" className="heading text-xl">
          The finalists, side by side
        </h2>
        <p className="mt-1 max-w-[80ch] text-sm text-ink-3">
          One condition for every lane: {condition?.condition ?? "the screen's c2 placement"} ({condition?.condition_id}, seed{" "}
          <span className="num">{condition?.seed}</span>), on one clock. Each lane re-ran its scored screen episode with
          recording on and reproduced it bit for bit. Bodies are drawn from the collision shapes the simulator used.
        </p>
        <div className="mt-4">
          <EffectorLanes lanes={page.lanes} task={page.task} />
        </div>
      </section>

      <section className="mt-14" aria-labelledby="families-title">
        <h2 id="families-title" className="heading text-xl">
          Families
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {page.families.map((f) => (
            <FamilyCard key={f.family} f={f} />
          ))}
        </div>
      </section>

      <section className="mt-14" aria-labelledby="ranking-title">
        <h2 id="ranking-title" className="heading text-xl">
          Ranking ({REQUIREMENT_LABELS[req].toLowerCase()})
        </h2>
        <p className="mt-1 max-w-[80ch] text-sm text-ink-3">
          Eligible: certified, the nominal grasp passes, no simulator reset and no penetration over 1 mm in the nominal
          test. Ranked by certificate score, then the smaller placement tolerance, then screen success; within 0.05 and
          the same tolerance the simpler effector wins (fewer actuators, then lower mass, then higher fidelity).
        </p>
        <div className="mt-4">
          <RankingTable block={block} requirement={req} />
        </div>
        <details className="mt-6">
          <summary className="cursor-pointer text-sm text-ink-2">Ranking ({REQUIREMENT_LABELS[other].toLowerCase()})</summary>
          <div className="mt-3">
            <RankingTable block={otherBlock} requirement={other} />
          </div>
        </details>
      </section>

      {certified.length > 0 ? (
        <section className="mt-14" aria-labelledby="certs-title">
          <h2 id="certs-title" className="heading text-xl">
            Certificates
          </h2>
          <p className="mt-1 max-w-[80ch] text-sm text-ink-3">
            The pre-registered battery, unchanged: the nominal grasp, the part 1, 2 and 3 mm off where the wrist aims on
            both axes, a 1 g shake on three axes and a 0.5 N s push at the part&apos;s top on two. Green: registered;
            amber dash: held but not registered; red: dropped. Jitter is off.
          </p>
          <div className="mt-4 grid gap-6 md:grid-cols-2">
            {certified.map((c) => (
              <CertificateMatrix key={c.effector_id} cert={page.certificates[c.effector_id]!} name={c.display_name} />
            ))}
          </div>
        </section>
      ) : null}

      <section className="mt-14" aria-labelledby="budget-title">
        <h2 id="budget-title" className="heading text-xl">
          Budget and fairness
        </h2>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[720px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-3 font-normal">Group</th>
                <th scope="col" className="py-2 pr-3 font-normal">Status</th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">Candidates</th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">Screen episodes</th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">Reference episodes</th>
                <th scope="col" className="py-2 pr-3 text-right font-normal">Certificates</th>
                <th scope="col" className="py-2 font-normal">Finalists</th>
              </tr>
            </thead>
            <tbody>
              {page.groups.map((g) => (
                <tr key={g.group} className="border-b border-rule align-top">
                  <th scope="row" className="py-2 pr-3 font-normal text-ink">
                    {GROUP_TITLES[g.group]}
                  </th>
                  <td className="py-2 pr-3 text-xs text-ink-2">{g.status === "run" ? "run" : `${g.status}: ${g.reason ?? ""}`}</td>
                  <td className="num py-2 pr-3 text-right">{g.budget.candidates}</td>
                  <td className="num py-2 pr-3 text-right">
                    {g.budget.screen_episodes} <span className="text-ink-3">/ {g.budget.max_screen_episodes}</span>
                  </td>
                  <td className="num py-2 pr-3 text-right">{g.budget.reference_episodes}</td>
                  <td className="num py-2 pr-3 text-right">
                    {g.budget.certificates} <span className="text-ink-3">/ {g.budget.max_certificates}</span>
                  </td>
                  <td className="py-2 text-xs text-ink-2">{g.finalists.map((id) => names.get(id) ?? id.slice(0, 12)).join(", ") || "none"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <dl className="mt-6 grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
          <dt className="text-ink-3">Same for every candidate</dt>
          <dd>
            {page.fairness.identical ? (
              <span className="text-pass">yes</span>
            ) : (
              <span className="text-fail">no: {page.fairness.problems.join("; ")}</span>
            )}
            <span className="text-ink-2">
              {" "}
              (wrist servos and force limits, solver, the part&apos;s contact parameters, phase timings, conditions and
              seeds, read from each compiled model)
            </span>
          </dd>
          <dt className="text-ink-3">Conditions</dt>
          <dd className="num text-xs">
            {page.conditions.map((c) => (
              <span key={c.condition_id} className="mr-3 inline-block">
                {c.label}: {c.placement.err_x_mm >= 0 ? "+" : ""}
                {c.placement.err_x_mm} mm{c.placement.err_yaw_deg ? `, ${c.placement.err_yaw_deg} deg` : ""} (seeds {c.seeds.join(", ")})
              </span>
            ))}
          </dd>
          <dt className="text-ink-3">Physics</dt>
          <dd className="num text-xs">
            {page.physics.episodes} screen and reference episodes, {page.physics.certificate_episodes} certificate episodes;{" "}
            {page.physics.bad_qacc} simulator resets, {page.physics.physics_unreliable} over 1 mm penetration
          </dd>
          <dt className="text-ink-3">Signature</dt>
          <dd className="num truncate text-xs">{page.fairness.signature_sha256 ?? "n/a"}</dd>
        </dl>
      </section>

      <section className="mt-14" aria-labelledby="caveats-title">
        <h2 id="caveats-title" className="heading text-xl">
          Caveats
        </h2>
        <ul className="prose-measure mt-3 grid gap-2 text-sm text-ink-2">
          <li>{page.caveats.common}</li>
          {Object.entries(page.caveats)
            .filter(([k]) => k !== "common")
            .map(([k, v]) => (
              <li key={k}>
                <span className="text-ink">{FAMILY_TITLES[k as keyof typeof FAMILY_TITLES] ?? k}:</span> {v}
              </li>
            ))}
        </ul>
        <p className="num mt-4 text-xs text-ink-3">
          {page.provenance.engine} · code {page.provenance.code_version} · task {page.provenance.task_digest} · run{" "}
          {page.provenance.run_id} · {page.provenance.created_utc}
        </p>
      </section>

      <TitleBlock title={`Which effector: ${page.part.title}`} drawing={`/effectors/${page.part.part_id}`} sheet="Product mode" />
    </>
  );
}
