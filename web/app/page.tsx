import Link from "next/link";

import { computeChip, ProvenanceChips, uniqueDesignChips } from "@/components/ProvenanceChips";
import BoutPlayer from "@/components/replay/BoutPlayer";
import { TitleBlock } from "@/components/TitleBlock";
import { getCertificates, getDesigns, getFeaturedBout, getJobs, getResults } from "@/lib/data";
import { APP_NAME, APP_TAGLINE } from "@/lib/site";
import { siteStats } from "@/lib/stats";

const STEPS = [
  {
    title: "Drop your part",
    body: "Upload an STL of the part you need to pick. The agent measures it, splits it into convex pieces and finds every way it can rest on a table.",
    detail: "STL in, resting poses out",
  },
  {
    title: "Fingers shaped to your part",
    body: "NVIDIA Nemotron 3 Super, served by Nebius Token Factory, proposes fingers shaped to your part, then reads each round's measured failures before it proposes the next.",
    detail: "Part-shaped fingers",
  },
  {
    title: "Every effector, same trials",
    body: "Multi-finger hands, industrial grippers, a soft arm, a suction cup and effectors Nemotron designs all face the same placements and seeds in simulation. The best one that qualifies is recommended.",
    detail: "Hands, grippers, soft arm, suction",
  },
  {
    title: "Certify it, then print it",
    body: "The pick is stress-tested under placement error for a robustness certificate. For fingers you download a print pack: the finger STLs, printable jaws and the certificate.",
    detail: "Placement error, shake, push",
  },
] as const;

export default function HomePage() {
  const bout = getFeaturedBout();
  const designs = getDesigns();
  const designIds = designs.designs.map((d) => d.id);
  const stats = siteStats({ jobs: getJobs(), designs, results: getResults(), certificates: getCertificates() });
  const [line1, ...rest] = APP_TAGLINE.split(". ");

  return (
    <>
      <section className="pt-12 sm:pt-20" aria-labelledby="hero-title">
        <h1 id="hero-title" className="display max-w-[14ch] text-[clamp(2.6rem,8.4vw,5.75rem)] text-ink">
          <span className="block">{line1}.</span>
          <span className="block text-accent">{rest.join(". ")}</span>
        </h1>
        <p className="prose-measure mt-6 text-lg text-ink-2 sm:text-xl">
          Custom gripper fingers are usually found by printing and testing until one works. {APP_NAME} moves that loop
          into simulation: an NVIDIA Nemotron agent designs fingers for your part, and Nebius runs the grasp trials that
          keep or reject them.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href="/part/" className="bg-accent px-5 py-2.5 text-accent-ink no-underline">
            Bring your part
          </Link>
          {bout ? (
            <Link href={`/bout/${bout.id}/`} className="border border-rule-strong px-4 py-2.5 text-ink no-underline hover:bg-paper-2">
              Watch a recorded bout
            </Link>
          ) : null}
        </div>
      </section>

      <section className="mt-16" aria-labelledby="flow-title">
        <h2 id="flow-title" className="sr-only">
          How it works
        </h2>
        <ol className="grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((step, i) => (
            <li key={step.title} className="relative flex flex-col bg-paper p-5">
              <span className="num text-sm text-accent" aria-hidden="true">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h3 className="heading mt-3 text-lg">{step.title}</h3>
              <p className="mt-2 flex-1 text-[0.9375rem] text-ink-2">{step.body}</p>
              <p className="num mt-4 border-t border-rule pt-2 text-xs text-ink-3">{step.detail}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-10" aria-labelledby="stats-title">
        <h2 id="stats-title" className="text-xs font-semibold tracking-wide text-ink-3">
          FROM THE PUBLISHED DATA
        </h2>
        <ul className="mt-3 grid grid-cols-2 gap-px border border-rule bg-rule lg:grid-cols-4">
          {stats.map((s) => (
            <li key={s.label} className="bg-paper">
              <Link href={s.href} className="group flex h-full flex-col px-4 py-3 no-underline hover:bg-paper-2/60">
                <span className="num text-[clamp(1.6rem,4vw,2.25rem)] leading-tight text-ink">{s.value}</span>
                <span className="text-sm text-ink group-hover:underline">{s.label}</span>
                <span className="mt-1 text-xs text-ink-3">{s.note}</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {bout ? (
        <section className="mt-20" aria-labelledby="featured-title">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="prose-measure">
              <h2 id="featured-title" className="heading text-2xl sm:text-3xl">
                {bout.title}
              </h2>
              <p className="mt-2 text-ink-2">{bout.summary}</p>
            </div>
            <Link href={`/bout/${bout.id}/`} className="link text-sm">
              Open this bout with its numbers
            </Link>
          </div>
          <ProvenanceChips
            className="mt-4"
            chips={[computeChip(bout.compute), ...uniqueDesignChips(bout.lanes.map((l) => l.design_provenance))]}
          />
          <div className="mt-5">
            <BoutPlayer bout={bout} autoPlay loop designIds={designIds} />
          </div>
          <p className="mt-4 text-sm text-ink-3">
            Recorded run, replayed from the simulator&apos;s own pose log. The dashed square marks the success
            height: the object&apos;s centre must rise {(bout.task.success_lift_m * 1000).toFixed(0)} mm and stay
            there for {bout.task.success_hold_s.toFixed(1)} s.
          </p>
        </section>
      ) : null}

      <TitleBlock title={APP_TAGLINE} drawing="/" sheet="1 of 6" />
    </>
  );
}
