import type { Metadata } from "next";
import Link from "next/link";

import { EffectorsLink } from "@/components/effectors/EffectorsLink";
import PartWorkbench from "@/components/part/PartWorkbench";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { certificateIdsByDesign, getBouts, getDesigns, getSessionSummaries } from "@/lib/data";

export const metadata: Metadata = {
  title: "Bring your part",
  description: "Drop an STL, check it against the gripper's opening, and watch a Nemotron agent design and test fingers for it.",
};

/** Opening of the simulated parallel gripper when no bout states it (configs/task_v1.yaml: 2 x 42 mm). */
const DEFAULT_OPENING_MM = 84;

export default function PartPage() {
  const sessions = getSessionSummaries();
  const bouts = getBouts().bouts;
  const openingMm = bouts[0]?.task.max_opening_mm ?? DEFAULT_OPENING_MM;
  const replayBouts: Record<string, string> = {};
  for (const b of bouts) for (const lane of b.lanes) replayBouts[lane.replay] ??= b.id;
  const links = {
    designIds: getDesigns().designs.map((d) => d.id),
    certificates: Object.fromEntries(certificateIdsByDesign()),
    replayBouts,
  };

  return (
    <>
      <PageHeader
        title="Bring your part"
        lede={
          <p>
            Drop the STL of an object you need a robot to pick up. You see it against the jaws of an open-source
            parallel gripper, then an agent designs fingers for it, proves them in simulation and hands you the files
            to print.
          </p>
        }
      >
        <p className="text-sm text-ink-3">
          Simulated results; printed fingers are not yet validated on hardware. See{" "}
          <Link className="link" href="/about/">
            method and limitations
          </Link>
          .
        </p>
        <EffectorsLink />
      </PageHeader>

      <PartWorkbench sessions={sessions} openingMm={openingMm} links={links} />

      <TitleBlock title="Bring your part" drawing="/part" sheet="2 of 6" />
    </>
  );
}
