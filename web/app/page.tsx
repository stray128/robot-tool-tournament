import Link from "next/link";

import { computeChip, ProvenanceChips, uniqueDesignChips } from "@/components/ProvenanceChips";
import BoutPlayer from "@/components/replay/BoutPlayer";
import { TitleBlock } from "@/components/TitleBlock";
import { getDesigns, getFeaturedBout } from "@/lib/data";
import { APP_NAME, APP_TAGLINE } from "@/lib/site";

const STEPS = [
  {
    title: "Upload your part",
    body: "Send an STL of the object you need to pick. It is split into convex pieces and every pose it can rest in is found.",
    detail: "STL in, stable poses out",
  },
  {
    title: "Nemotron designs fingers",
    body: "NVIDIA Nemotron 3 Super on Nebius Token Factory proposes finger shapes as ten physical parameters, then reads each round's measured failures before proposing the next.",
    detail: "10 parameters, 14 boxes per finger",
  },
  {
    title: "Nebius physics proves them",
    body: "Nebius Serverless Jobs run every candidate through MuJoCo grasp trials, then a robustness certificate: placement errors up to 3 mm, a shake during the hold and a sideways push.",
    detail: "Same gripper, same controller, same force",
  },
  {
    title: "Download the print pack",
    body: "Left and right finger STLs sized for an open-source parallel gripper, a mount plate and the certificate, ready for a desktop printer.",
    detail: "STL, mount plate, certificate",
  },
] as const;

export default function HomePage() {
  const bout = getFeaturedBout();
  const designIds = getDesigns().designs.map((d) => d.id);
  const [line1, ...rest] = APP_TAGLINE.split(". ");

  return (
    <>
      <section className="pt-12 sm:pt-20" aria-labelledby="hero-title">
        <h1 id="hero-title" className="display max-w-[14ch] text-[clamp(2.6rem,8.4vw,5.75rem)] text-ink">
          <span className="block">{line1}.</span>
          <span className="block">{rest.join(". ")}</span>
        </h1>
        <p className="prose-measure mt-6 text-lg text-ink-2 sm:text-xl">
          Custom gripper fingers are usually found by printing and testing until one works. {APP_NAME} moves that
          loop into simulation: an NVIDIA Nemotron model proposes finger shapes for your part, and Nebius runs the
          grasp trials that keep or reject them.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          {bout ? (
            <Link href={`/bout/${bout.id}/`} className="bg-accent px-4 py-2.5 text-accent-ink no-underline">
              Watch a recorded bout
            </Link>
          ) : null}
          <Link href="/about/" className="border border-rule-strong px-4 py-2.5 text-ink no-underline hover:bg-paper-2">
            How the trials are scored
          </Link>
        </div>
      </section>

      <section className="mt-20" aria-labelledby="flow-title">
        <h2 id="flow-title" className="heading text-2xl sm:text-3xl">
          From part to printable fingers
        </h2>
        <ol className="mt-6 grid gap-px border border-rule bg-rule sm:grid-cols-2 lg:grid-cols-4">
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

      <TitleBlock title={APP_TAGLINE} drawing="/" sheet="1 of 5" />
    </>
  );
}
