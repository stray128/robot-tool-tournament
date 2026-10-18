import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

import { FidelityBadge } from "@/components/effectors/FidelityBadge";
import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { CapabilityTag, EntrantBadge, SimStatusBadge, SpecTagBadge } from "@/components/library/Badges";
import LibraryViewerLazy from "@/components/library/LibraryViewerLazy";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { massText, OUTCOME_LABELS } from "@/lib/effectors";
import { formatBytes } from "@/lib/format";
import {
  type GateEntrant,
  type LibraryItem,
  licenceText,
  MOUNT_STATUS_LABELS,
  repoFileUrl,
  sentence,
  shortCommit,
} from "@/lib/library";
import { APP_NAME, REPO_URL } from "@/lib/site";

import { getLibraryIndex, getLibraryItem, getLibraryKeys, getPartTitles } from "../data";

export const dynamicParams = false;

export function generateStaticParams() {
  return getLibraryKeys().map((item) => ({ item }));
}

type Params = { params: Promise<{ item: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const item = getLibraryItem((await params).item);
  return item ? { title: item.display_name, description: item.summary } : {};
}

const TH = "py-2 pr-4 font-normal";

function Block({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-12" aria-labelledby={id}>
      <h2 id={id} className="heading text-xl">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-x-4 gap-y-0.5 border-b border-rule py-2 sm:grid-cols-[9rem_minmax(0,1fr)]">
      <dt className="text-sm text-ink-3">{label}</dt>
      <dd className="min-w-0 text-ink">{children}</dd>
    </div>
  );
}

const isUrl = (s: string) => /^https?:\/\//.test(s);

function mm(v: number | null): string {
  return v === null ? "n/a" : `${v} mm`;
}

function outcomeText(outcome: string | null, held: boolean): string {
  if (outcome) return OUTCOME_LABELS[outcome] ?? outcome.replace(/_/g, " ");
  return held ? "Held" : "Not held";
}

function GateCard({ e, titles }: { e: GateEntrant; titles: Record<string, string> }) {
  const parts = [...new Set([...e.built, ...Object.keys(e.refused), ...Object.keys(e.nominal)])];
  const title = (p: string) => titles[p] ?? p;
  return (
    <article className="mt-4 border border-rule-strong bg-paper p-4" aria-label={`Gate of ${e.display_name}`} data-gate-entrant>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="heading text-lg text-ink">{e.display_name}</h3>
        <span className={`stamp ${e.passed ? "text-pass" : "text-fail"}`}>{e.passed ? "GATE PASSED" : "GATE FAILED"}</span>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-5">
        {[
          ["Primitive", e.primitive],
          ["Actuators", e.n_actuators ?? "n/a"],
          ["Mass", massText(e.mass_kg)],
          ["Reach", mm(e.reach_mm)],
          ["Opening", mm(e.opening_mm)],
        ].map(([k, v]) => (
          <div key={String(k)}>
            <dt className="text-xs text-ink-3">{k}</dt>
            <dd className="num text-ink">{v}</dd>
          </div>
        ))}
      </dl>
      {e.error ? <p className="mt-3 border-l-2 border-fail pl-2 text-sm text-fail">Error: {sentence(e.error)}</p> : null}
      {e.probe_only ? (
        <p className="mt-3 border-l-2 border-pending pl-2 text-sm text-pending">
          Every training part is too big for it, so it was built on the 10 mm probe block only.
        </p>
      ) : null}
      {parts.length > 0 ? (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[680px] border-collapse text-left text-sm">
            <thead>
              <tr className="border-y border-rule-strong text-xs text-ink-3">
                <th scope="col" className={TH}>Training part</th>
                <th scope="col" className={TH}>Gate</th>
                <th scope="col" className={TH}>Held (nominal)</th>
                <th scope="col" className={TH}>Nominal outcome</th>
                <th scope="col" className={`${TH} text-right`}>Penetration</th>
                <th scope="col" className="py-2 font-normal">Why refused</th>
              </tr>
            </thead>
            <tbody>
              {parts.map((p) => {
                const ep = e.nominal[p];
                const built = e.built.includes(p);
                return (
                  <tr key={p} className="border-b border-rule align-top">
                    <th scope="row" className="py-2 pr-4 font-normal text-ink">
                      {title(p)}
                    </th>
                    <td className="py-2 pr-4">{built ? <span className="text-pass">built</span> : <span className="text-ink-3">refused</span>}</td>
                    <td className="py-2 pr-4">{ep ? (ep.held ? <span className="text-pass">yes</span> : <span className="text-fail">no</span>) : "n/a"}</td>
                    <td className="py-2 pr-4 text-ink-2">{ep ? outcomeText(ep.outcome, ep.held) : "not run"}</td>
                    <td className="num py-2 pr-4 text-right">{ep?.penetration_mm != null ? `${ep.penetration_mm.toFixed(2)} mm` : "n/a"}</td>
                    <td className="py-2 text-ink-2">{e.refused[p] ? sentence(e.refused[p]!) : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
      <p className="mt-3 text-sm text-ink-2">
        {e.nominal_held.length > 0
          ? `Held at nominal: ${e.nominal_held.map(title).join(", ")}.`
          : "Held on no training part in the nominal episode."}
      </p>
    </article>
  );
}

function Simulation({ item, titles }: { item: LibraryItem; titles: Record<string, string> }) {
  const sim = item.sim;
  return (
    <>
      <dl className="mt-3">
        <Fact label="Status">
          <span className="flex flex-wrap items-center gap-2">
            <SimStatusBadge status={sim.status} />
            <span>{sim.title}</span>
          </span>
        </Fact>
        {sim.source ? <Fact label="Model source">{sentence(sim.source)}</Fact> : null}
        <Fact label="Fidelity">{sim.fidelity ? <FidelityBadge fidelity={sim.fidelity} /> : <span className="text-ink-3">n/a (no physics)</span>}</Fact>
        {sim.family ? <Fact label="Family">{sim.family.replace(/_/g, " ")}</Fact> : null}
        {sim.n_actuators !== null || sim.mass_kg !== null ? (
          <Fact label="Model">
            <span className="num">
              {sim.n_actuators ?? "n/a"} {sim.n_actuators === 1 ? "actuator" : "actuators"} · {massText(sim.mass_kg)}
            </span>
          </Fact>
        ) : null}
        {sim.note ? <Fact label="Note">{sentence(sim.note)}</Fact> : null}
        {sim.reason ? <Fact label="Reason">{sentence(sim.reason)}</Fact> : null}
        <Fact label="Tournament">
          <span className="flex flex-wrap items-center gap-2">
            <EntrantBadge entrant={sim.entrant} />
            <span className="text-ink-2">
              {!sim.entrant
                ? sentence(sim.entrant_reason ?? "")
                : item.gate.entrants.length > 0
                  ? `${item.gate.entrants.length} entrant ${item.gate.entrants.length === 1 ? "spec" : "specs"} in the effector tournament.`
                  : "It takes part in the effector tournament."}
            </span>
          </span>
        </Fact>
      </dl>
      {sim.entrant ? (
        item.gate.ran ? (
          <>
            <h3 className="heading mt-8 text-lg">Compile and attach gate</h3>
            <p className="prose-measure mt-1 text-sm text-ink-2">
              Each entrant spec is built in the tournament&apos;s shared scene on every training part. One nominal episode
              per built part (no placement error, seed 0) is information, not part of the gate.
            </p>
            {item.gate.entrants.map((e) => (
              <GateCard key={`${e.display_name}-${e.primitive}`} e={e} titles={titles} />
            ))}
          </>
        ) : (
          <p className="mt-3 text-sm text-ink-3">The gate was not run for this export.</p>
        )
      ) : null}
    </>
  );
}

function Downloads({ item }: { item: LibraryItem }) {
  if (item.downloads.length === 0) return <p className="mt-3 text-ink-3">No files are mirrored for this item.</p>;
  const own = item.upstream.generated;
  return (
    <>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-left text-sm" data-library-table="downloads">
          <thead>
            <tr className="border-y border-rule-strong text-xs text-ink-3">
              <th scope="col" className={TH}>File</th>
              <th scope="col" className={TH}>Type</th>
              <th scope="col" className={`${TH} text-right`}>Size</th>
              <th scope="col" className={TH}>Licence</th>
              <th scope="col" className="py-2 font-normal">Changed</th>
            </tr>
          </thead>
          <tbody>
            {item.downloads.map((d) => {
              const href = repoFileUrl(REPO_URL, d.path);
              return (
                <tr key={d.path} className="border-b border-rule align-top">
                  <th scope="row" className="py-2 pr-4 font-normal" title={`sha256 ${d.sha256}`}>
                    {href ? (
                      <a className="link break-all" href={href} rel="noreferrer">
                        {d.name}
                      </a>
                    ) : (
                      <span className="break-all text-ink">{d.name}</span>
                    )}
                    <span className="num mt-0.5 block break-all text-xs text-ink-3">{d.path}</span>
                  </th>
                  <td className="py-2 pr-4 text-ink-2">{d.role_title}</td>
                  <td className="num py-2 pr-4 text-right">{formatBytes(d.bytes)}</td>
                  <td className="num py-2 pr-4">{d.license}</td>
                  <td className="py-2">
                    {d.modified ? <span className="text-pending">modified</span> : <span className="text-ink-3">{own ? "generated" : "as upstream"}</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-ink-3">
        {REPO_URL
          ? "Each file links to its copy in the repository. "
          : "Files are in the repository at the paths shown. "}
        Licence texts sit next to them in <span className="num">LICENSES/</span>. What changed is on the{" "}
        <Link className="link" href={`/credits/#${item.credit_anchor}`}>
          credits page
        </Link>
        .
      </p>
    </>
  );
}

export default async function LibraryItemPage({ params }: Params) {
  const item = getLibraryItem((await params).item);
  if (!item) notFound();
  const index = getLibraryIndex()!;
  const notes = Object.fromEntries(index.capabilities.map((c) => [c.tag, c.note]));
  const names = new Map(index.items.map((i) => [i.key, i.display_name]));
  const adapterFor = index.items.filter((i) => i.mount.adapter === item.key);
  const titles = getPartTitles(
    item.gate.entrants.flatMap((e) => [...e.built, ...Object.keys(e.refused), ...Object.keys(e.nominal)]),
  );
  const m = item.mount;

  return (
    <>
      <p className="pt-8 text-sm">
        <Link href="/library/" className="link">
          Effector library
        </Link>
      </p>
      <ProductModeBanner detail={false} />
      <PageHeader title={item.display_name} lede={<p>{item.summary}</p>}>
        <p className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
          <span>{item.kind_title}</span>
          <SimStatusBadge status={item.sim.status} />
          <EntrantBadge entrant={item.sim.entrant} />
          <span className="num text-ink-3">{[...new Set(item.licences.map((l) => l.spdx))].join(", ")}</span>
        </p>
      </PageHeader>

      <div className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          {item.preview ? (
            <LibraryViewerLazy url={item.preview.url} bytes={item.preview.bytes} label={`3D preview of the ${item.display_name}`} />
          ) : (
            <p className="crop-marks grid aspect-[4/3] place-items-center bg-paper-2/60 p-4 text-center text-sm text-ink-3">
              No 3D preview for this item.
            </p>
          )}
        </div>
        <section aria-labelledby="facts" className="min-w-0">
          <h2 id="facts" className="heading text-xl">
            Facts
          </h2>
          <dl className="mt-2">
            <Fact label="Kind">{item.kind_title}</Fact>
            <Fact label="Class">{item.class_title}</Fact>
            <Fact label="Capabilities">
              {item.capabilities.length > 0 ? (
                <ul className="grid gap-1">
                  {item.capabilities.map((t) => (
                    <li key={t} className="flex flex-wrap items-baseline gap-2 text-sm">
                      <CapabilityTag tag={t} />
                      <span className="text-ink-2">{notes[t] ?? ""}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="text-sm text-ink-2">None: it carries other items.</span>
              )}
            </Fact>
            <Fact label="Mount">
              <span className="block text-sm">{sentence(m.interface)}</span>
              <span className="mt-1 block text-sm text-ink-2">
                ISO pattern: {m.iso_pattern ? <span className="num text-ink">{m.iso_pattern}</span> : "none (own interface)"}
                {m.iso_pattern ? <> ({MOUNT_STATUS_LABELS[m.status]})</> : null}
                {isUrl(m.source) ? (
                  <>
                    {" "}
                    <a className="link" href={m.source} rel="noreferrer">
                      Source
                    </a>
                  </>
                ) : null}
              </span>
              {m.adapter ? (
                <span className="mt-1 block text-sm text-ink-2">
                  Suggested adapter:{" "}
                  <Link className="link" href={`/library/${m.adapter}/`}>
                    {names.get(m.adapter) ?? m.adapter}
                  </Link>
                </span>
              ) : null}
              {m.notes.length > 0 ? (
                <ul className="mt-1 grid gap-0.5 text-sm text-pending">
                  {m.notes.map((n) => (
                    <li key={n}>{sentence(n)}</li>
                  ))}
                </ul>
              ) : null}
            </Fact>
            {adapterFor.length > 0 ? (
              <Fact label="Suggested for">
                <ul className="grid gap-0.5 text-sm">
                  {adapterFor.map((i) => (
                    <li key={i.key}>
                      <Link className="link" href={`/library/${i.key}/`}>
                        {i.display_name}
                      </Link>
                    </li>
                  ))}
                </ul>
              </Fact>
            ) : null}
          </dl>
        </section>
      </div>

      <Block id="specs" title="Specifications">
        {item.specs.length > 0 ? (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[600px] border-collapse text-left text-sm" data-library-table="specs">
              <thead>
                <tr className="border-y border-rule-strong text-xs text-ink-3">
                  <th scope="col" className={TH}>Spec</th>
                  <th scope="col" className={TH}>Value</th>
                  <th scope="col" className={TH}>Basis</th>
                  <th scope="col" className="py-2 font-normal">Source</th>
                </tr>
              </thead>
              <tbody>
                {item.specs.map((s, i) => (
                  <tr key={`${s.name}-${i}`} className="border-b border-rule align-top">
                    <th scope="row" className="py-2 pr-4 font-normal text-ink">
                      {s.name}
                    </th>
                    <td className="num py-2 pr-4 text-ink">
                      {s.value}
                      {s.unit ? ` ${s.unit}` : ""}
                    </td>
                    <td className="py-2 pr-4">
                      <SpecTagBadge tag={s.tag} />
                    </td>
                    <td className="py-2 text-ink-2">
                      {isUrl(s.source) ? (
                        <a className="link" href={s.source} rel="noreferrer">
                          Source
                        </a>
                      ) : s.tag === "derived" ? (
                        "derived from the files or the model"
                      ) : s.tag === "assumed" ? (
                        "our assumption, not read from a source"
                      ) : (
                        "no source recorded"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-3 text-ink-3">No specifications are recorded for this item.</p>
        )}
      </Block>

      <Block id="simulation" title="Simulation">
        <Simulation item={item} titles={titles} />
      </Block>

      <Block id="licence" title="Licence">
        <ul className="mt-3 grid gap-2">
          {item.licences.map((l, i) => (
            <li key={`${l.spdx}-${i}`} className="grid gap-x-4 border-b border-rule pb-2 sm:grid-cols-[9rem_minmax(0,1fr)]">
              <span className="num text-ink">{licenceText(l.spdx)}</span>
              <span className="text-ink-2">
                {l.covers ? sentence(`covers ${l.covers}`) : ""}
                {l.url ? (
                  <>
                    {" "}
                    <a className="link whitespace-nowrap" href={l.url} rel="noreferrer license">
                      Licence text
                    </a>
                  </>
                ) : null}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-sm text-ink-2">
          Licence rating: <span className="num text-ink">{item.rating}</span>. Full attribution: see the{" "}
          <Link className="link" href={`/credits/#${item.credit_anchor}`}>
            credits page
          </Link>
          .
        </p>
      </Block>

      <Block id="files" title="Files">
        <Downloads item={item} />
      </Block>

      <Block id="upstream" title="Upstream">
        {item.upstream.generated ? (
          <p className="mt-3 text-ink-2">Generated by {APP_NAME}. Our own work under Apache-2.0, with no upstream.</p>
        ) : (
          <ul className="mt-3 grid gap-1 text-ink-2">
            <li>
              <a className="link" href={item.upstream.url} rel="noreferrer">
                Upstream repository, commit <span className="num">{shortCommit(item.upstream.commit)}</span>
              </a>
            </li>
            {item.upstream.extra.map((e) => (
              <li key={e.url}>
                <a className="link" href={e.url} rel="noreferrer">
                  Pinned repository, commit <span className="num">{shortCommit(e.commit)}</span>
                </a>
              </li>
            ))}
          </ul>
        )}
        {item.links.length > 0 ? (
          <>
            <h3 className="heading mt-6 text-lg">Linked, not mirrored</h3>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[560px] border-collapse text-left text-sm" data-library-table="links">
                <thead>
                  <tr className="border-y border-rule-strong text-xs text-ink-3">
                    <th scope="col" className={TH}>What, and why</th>
                    <th scope="col" className={`${TH} text-right`}>Size</th>
                    <th scope="col" className="py-2 font-normal">Link</th>
                  </tr>
                </thead>
                <tbody>
                  {item.links.map((l) => (
                    <tr key={l.url} className="border-b border-rule align-top">
                      <th scope="row" className="py-2 pr-4 font-normal text-ink-2">
                        {sentence(l.reason || l.label)}
                      </th>
                      <td className="num py-2 pr-4 text-right">{l.bytes > 0 ? formatBytes(l.bytes) : "n/a"}</td>
                      <td className="py-2">
                        <a className="link whitespace-nowrap" href={l.url} rel="noreferrer">
                          Pinned file
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </Block>

      {item.notes.length > 0 ? (
        <Block id="notes" title="Notes">
          <ul className="prose-measure mt-3 grid list-disc gap-1 pl-5 text-ink-2">
            {item.notes.map((n) => (
              <li key={n}>{sentence(n)}</li>
            ))}
          </ul>
        </Block>
      ) : null}

      <TitleBlock title={item.display_name} drawing={`/library/${item.key}`} sheet="Detail of Effector library" />
    </>
  );
}
