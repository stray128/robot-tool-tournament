import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { ExcludedTable, IsoPatternTable, ListedTable } from "@/components/library/ExcludedTable";
import LibraryCatalog from "@/components/library/LibraryCatalog";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { type LibraryIndex, SIM_STATUS_LABELS, SIM_STATUSES, sentence, toCard } from "@/lib/library";

import { getLibraryIndex } from "./data";

export const metadata: Metadata = {
  title: "Effector library",
  description:
    "Open, licence-clean end-effectors with their files, sourced specs, mounting interface, capability tags and simulation status. Product mode.",
};

function Counts({ index }: { index: LibraryIndex }) {
  const c = index.counts;
  const cells: { label: string; value: number; note?: string }[] = [
    { label: "Items", value: c.items },
    { label: "Entrant specs", value: c.entrants, note: `from ${c.entrant_items} items` },
    ...SIM_STATUSES.filter((s) => s !== "none" || c.by_sim_status.none > 0).map((s) => ({
      label: SIM_STATUS_LABELS[s],
      value: c.by_sim_status[s],
      note: "simulation",
    })),
    { label: "Listed, not mirrored", value: c.listed },
    { label: "Excluded, amber", value: c.excluded_amber },
    { label: "Excluded, red", value: c.excluded_red },
  ];
  return (
    <dl className="mt-8 grid grid-cols-2 border-l border-t border-rule-strong sm:grid-cols-4 lg:grid-cols-8" aria-label="Library in numbers" data-library-counts>
      {cells.map((cell) => (
        <div key={cell.label} className="border-b border-r border-rule-strong px-3 py-2">
          <dt className="text-xs text-ink-3">{cell.label}</dt>
          <dd className="num text-2xl text-ink">{cell.value}</dd>
          {cell.note ? <dd className="text-xs text-ink-3">{cell.note}</dd> : null}
        </div>
      ))}
    </dl>
  );
}

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-14" aria-labelledby={id}>
      <h2 id={id} className="heading text-xl">
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function LibraryPage() {
  const index = getLibraryIndex();
  const lede = (
    <p>
      Open, licence-clean end-effectors: grippers, hands, suction cups, tools and mounting adapters. Each item comes with
      its files, its specs and their sources, its mounting interface, capability tags and a simulation status. Names on
      screen are generic; the makers are credited on the{" "}
      <Link className="link" href="/credits/#library">
        credits page
      </Link>
      .
    </p>
  );
  if (!index) {
    return (
      <>
        <ProductModeBanner detail={false} />
        <PageHeader title="Effector library" lede={lede} />
        <p className="mt-8 text-ink-3">The library has not been exported yet.</p>
        <TitleBlock title="Effector library" drawing="/library" sheet="Product mode" />
      </>
    );
  }
  const cards = index.items.map(toCard);
  const licences = [...new Set(cards.flatMap((c) => c.licences))].sort();
  const simStatuses = SIM_STATUSES.filter((s) => index.counts.by_sim_status[s] > 0);

  return (
    <>
      <ProductModeBanner detail={false} />
      <PageHeader title="Effector library" lede={lede}>
        <p className="prose-measure text-sm text-ink-3">
          Each entrant is built in the tournament&apos;s shared scene on the training parts. That gate, and its nominal
          episodes, are product mode. The benchmark&apos;s pre-registered result is on{" "}
          <Link className="link" href="/results/">
            Results
          </Link>
          .
        </p>
      </PageHeader>

      <Counts index={index} />

      <Section id="catalog" title="Catalog">
        <LibraryCatalog cards={cards} capabilities={index.capabilities} licences={licences} simStatuses={simStatuses} />
      </Section>

      <Section id="listed" title="Listed, not mirrored">
        <p className="prose-measure mt-2 text-ink-2">
          These items are open, but their files are not in this repository. Each row says why. The link goes to the
          upstream files.
        </p>
        <ListedTable items={index.listed} />
      </Section>

      <Section id="excluded" title="Not included, and why">
        <p className="prose-measure mt-2 text-ink-2">
          Amber items have share-alike, reciprocal or unclear licences. They are excluded by the founder&apos;s decision of
          2026-10-18. Red items are non-commercial, unlicensed or under terms that forbid redistribution.
        </p>
        <p className="prose-measure mt-2 text-sm text-ink-3">
          Licence audit of <span className="num">{index.audit.date}</span>: {sentence(index.audit.note)}
        </p>
        <ExcludedTable items={index.excluded} />
      </Section>

      <Section id="vendored" title="Already in the effector catalog">
        <p className="prose-measure mt-2 text-ink-2">
          These models were vendored for the effector tournament before the library existed, so the library does not mirror
          them again. They are on the{" "}
          <Link className="link" href="/effectors/">
            effector pages
          </Link>
          .
        </p>
        <ul className="mt-3 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {index.already_vendored.map((v) => (
            <li key={v.key} className="flex flex-wrap items-baseline justify-between gap-x-3 border-b border-rule py-1.5">
              <Link className="link" href="/effectors/">
                {v.name}
              </Link>
              <span className="text-xs text-ink-3">{v.where}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section id="policy" title="Licence policy">
        <div className="prose-measure mt-2 grid gap-3 text-ink-2">
          <p>Only files under these permissive licences are mirrored (the green lane):</p>
          <ul className="flex flex-wrap gap-1.5" aria-label="Green-lane licences">
            {index.green_licences.map((l) => (
              <li key={l} className="num border border-pass/60 px-1.5 py-px text-xs text-ink">
                {l}
              </li>
            ))}
          </ul>
          <p>
            The chain rule: a file is only as clean as the worst link in its derivation chain. A converted, simplified or
            re-encoded file carries the licences of every file it came from.
          </p>
          <p>
            Copyright licences cover files, not patents or trademarks. A simulation model of a product does not license the
            product&apos;s design or its name, so names on screen are generic.
          </p>
          <p>
            Attribution: each item&apos;s credit line, copyright, pinned commit and changes are on the{" "}
            <Link className="link" href="/credits/#library">
              credits page
            </Link>
            .
          </p>
        </div>
        <h3 className="heading mt-8 text-lg">ISO 9409-1 flange patterns</h3>
        <p className="prose-measure mt-2 text-ink-2">
          Dimension values are facts from published sources; the standard&apos;s text is not reproduced. Confirmed patterns
          are the ones the library ships adapters for. No ISO conformance is claimed: measure every unverified dimension
          before you use it.
        </p>
        <IsoPatternTable patterns={index.iso_patterns} />
        <p className="mt-6 break-words text-xs text-ink-3">
          Provenance: <span className="num">{index.provenance.lock}</span>, manifest sha256{" "}
          <span className="num">{index.provenance.manifest_sha256.slice(0, 16)}</span>.
        </p>
      </Section>

      <TitleBlock title="Effector library" drawing="/library" sheet="Product mode" />
    </>
  );
}
