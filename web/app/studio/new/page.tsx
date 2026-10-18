import type { Metadata } from "next";
import Link from "next/link";

import { PageHeader } from "@/components/PageHeader";
import catalog from "@/components/studio/intake/catalog.json";
import IntakeWorkbench, { type RecordedPlans } from "@/components/studio/intake/IntakeWorkbench";
import recorded from "@/components/studio/intake/recorded.json";
import type { Catalog } from "@/components/studio/intake/taskspec";
import { TitleBlock } from "@/components/TitleBlock";

export const metadata: Metadata = {
  title: "Plan a use case",
  description:
    "Describe what a robot has to do with a few real objects. NVIDIA Nemotron turns it into objects and steps a physics simulator can run, and you edit every step.",
};

export default function StudioNewPage() {
  const plans = recorded as unknown as RecordedPlans;
  const summary = plans.gate.summary;
  return (
    <>
      <PageHeader
        title="Plan a use case"
        lede={
          <p>
            Say what the robot has to do: open a lid, pick parts into a tote, start a coupling on a thread, turn a valve. The
            Task Engineer turns it into objects and steps the simulator can run. You check and edit every step before anything
            runs.
          </p>
        }
      >
        <p className="flex flex-wrap items-center gap-2 text-sm text-ink-3">
          <span className="stamp text-pending">PRODUCT MODE</span>
          <span>{plans.label}.</span>
          <span>
            Recorded test set, one live run: {summary.plans.valid} of {summary.plans.n} use cases planned validly ({summary.plans.valid_first_call} on
            the first call, {summary.plans.valid_after_repair} after one repair call, {summary.plans.valid_after_layout_fix} after the layout fix),{" "}
            {summary.plans.failed} failed; {summary.refusals.refused_with_nearest} of {summary.refusals.n} unsupported ones refused with the nearest
            primitive.
          </span>
          <Link className="link" href="/about/">
            Method and limitations
          </Link>
        </p>
      </PageHeader>

      <IntakeWorkbench catalog={catalog as unknown as Catalog} recorded={plans} />

      <TitleBlock title="Plan a use case" drawing="/studio/new" sheet="Studio intake" />
    </>
  );
}
