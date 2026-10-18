import { DIM_STATUS_LABELS, type Dimension, type ExcludedItem, type IsoPattern, licenceText, type ListedItem, sentence } from "@/lib/library";

import { CapabilityTag, RatingBadge } from "./Badges";

const TH = "py-2 pr-4 font-normal";

/** Items worth knowing about whose files are not mirrored (link only, with the reason). */
export function ListedTable({ items }: { items: ListedItem[] }) {
  if (items.length === 0) return <p className="mt-3 text-ink-3">Nothing is listed without files.</p>;
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full min-w-[720px] border-collapse text-left text-sm" data-library-table="listed">
        <thead>
          <tr className="border-y border-rule-strong text-xs text-ink-3">
            <th scope="col" className={TH}>Item</th>
            <th scope="col" className={TH}>Licence</th>
            <th scope="col" className={TH}>Why it is not mirrored</th>
            <th scope="col" className="py-2 font-normal">Upstream</th>
          </tr>
        </thead>
        <tbody>
          {items.map((l, i) => (
            <tr key={`${l.name}-${i}`} className="border-b border-rule align-top">
              <th scope="row" className="py-2.5 pr-4 font-normal">
                <span className="text-ink">{l.name}</span>
                {l.capabilities.length > 0 ? (
                  <span className="mt-1 flex flex-wrap gap-1">
                    {l.capabilities.map((t) => (
                      <CapabilityTag key={t} tag={t} />
                    ))}
                  </span>
                ) : null}
              </th>
              <td className="num py-2.5 pr-4">{licenceText(l.license)}</td>
              <td className="py-2.5 pr-4 text-ink-2">{sentence(l.reason)}</td>
              <td className="py-2.5">
                <a className="link whitespace-nowrap" href={l.url} rel="noreferrer">
                  Link
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Items kept out of the library, with their licence rating and the reason. */
export function ExcludedTable({ items }: { items: ExcludedItem[] }) {
  if (items.length === 0) return <p className="mt-3 text-ink-3">Nothing was excluded.</p>;
  const sorted = [...items].sort((a, b) => (a.rating === b.rating ? 0 : a.rating === "amber" ? -1 : 1));
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full min-w-[720px] border-collapse text-left text-sm" data-library-table="excluded">
        <thead>
          <tr className="border-y border-rule-strong text-xs text-ink-3">
            <th scope="col" className={TH}>Item</th>
            <th scope="col" className={TH}>Rating</th>
            <th scope="col" className={TH}>Licence</th>
            <th scope="col" className={TH}>Reason</th>
            <th scope="col" className="py-2 font-normal">Upstream</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((e, i) => (
            <tr key={`${e.name}-${i}`} className="border-b border-rule align-top">
              <th scope="row" className="py-2.5 pr-4 font-normal text-ink">
                {e.name}
              </th>
              <td className="py-2.5 pr-4">
                <RatingBadge rating={e.rating} />
              </td>
              <td className="num py-2.5 pr-4">{licenceText(e.license)}</td>
              <td className="py-2.5 pr-4 text-ink-2">{sentence(e.reason)}</td>
              <td className="py-2.5">
                {e.url ? (
                  <a className="link whitespace-nowrap" href={e.url} rel="noreferrer">
                    Link
                  </a>
                ) : (
                  <span className="text-ink-3">no public files</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Dim({ d }: { d: Dimension }) {
  return (
    <span title={d.source || undefined}>
      <span className="num text-ink">{d.value} mm</span>
      <span className={`block text-xs ${d.status === "sourced" ? "text-ink-3" : "text-pending"}`}>{DIM_STATUS_LABELS[d.status]}</span>
    </span>
  );
}

const DIM_NAMES: Record<string, string> = { d1: "pitch circle", d2: "centring diameter", dowel_d: "dowel" };

/** The ISO 9409-1 flange patterns the library knows, each dimension with its provenance. */
export function IsoPatternTable({ patterns }: { patterns: IsoPattern[] }) {
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse text-left text-sm" data-library-table="iso">
        <thead>
          <tr className="border-y border-rule-strong text-xs text-ink-3">
            <th scope="col" className={TH}>Pattern</th>
            <th scope="col" className={TH}>Bolts</th>
            <th scope="col" className={TH}>Pitch circle</th>
            <th scope="col" className={TH}>Centring</th>
            <th scope="col" className={TH}>Dowel</th>
            <th scope="col" className={TH}>Adapter shipped</th>
            <th scope="col" className="py-2 font-normal">Unverified</th>
          </tr>
        </thead>
        <tbody>
          {patterns.map((p) => (
            <tr key={p.designation} className="border-b border-rule align-top">
              <th scope="row" className="py-2.5 pr-4 font-normal">
                <span className="num text-ink">{p.designation}</span>
                {p.seen_on ? <span className="mt-0.5 block text-xs text-ink-3">Seen on {p.seen_on}</span> : null}
                {p.notes.map((n) => (
                  <span key={n} className="mt-0.5 block max-w-[36ch] text-xs text-ink-3">
                    {sentence(n)}
                  </span>
                ))}
              </th>
              <td className="num py-2.5 pr-4">
                {p.n_holes} x {p.thread}
              </td>
              <td className="py-2.5 pr-4">
                <Dim d={p.d1_mm} />
              </td>
              <td className="py-2.5 pr-4">
                <Dim d={p.d2_mm} />
              </td>
              <td className="py-2.5 pr-4">
                <Dim d={p.dowel_mm} />
              </td>
              <td className="py-2.5 pr-4">{p.confirmed ? <span className="text-pass">yes, confirmed</span> : <span className="text-ink-3">no</span>}</td>
              <td className="py-2.5 text-ink-2">{p.unverified.length > 0 ? p.unverified.map((u) => DIM_NAMES[u] ?? u).join(", ") : "none"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
