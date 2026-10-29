import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Plan a task",
  description: "The planner moved into the Studio.",
  robots: { index: false },
};

/** Old route: the planner now lives on the Studio page (static redirect, so old links keep working). */
export default function StudioNewRedirect() {
  return (
    <>
      <meta httpEquiv="refresh" content="0; url=../#plan" />
      <section className="pt-16">
        <h1 className="display text-[2.2rem] text-ink">The planner moved</h1>
        <p className="mt-4 text-ink-2">
          Planning a task is now part of the Studio.{" "}
          <Link className="link" href="/studio/#plan">
            Open the planner
          </Link>
          .
        </p>
      </section>
    </>
  );
}
