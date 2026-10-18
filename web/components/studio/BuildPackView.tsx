import { formatBytes } from "@/lib/format";
import { dataUrl } from "@/lib/paths";

import type { BuildPack } from "./schema";

const KIND_LABELS: Record<string, string> = {
  effector_cad: "tool reference geometry",
  adapter: "adapter",
  bracket: "printed bracket",
  changer_plate: "changer plate",
  mjcf: "simulation scene",
  bom: "bill of materials",
  certificate: "certificate",
  notes: "notes",
  licence: "licence",
};

/** The build pack: the zip with its size and hash, what is in it, upstream links and caveats. */
export function BuildPackView({ pack, note }: { pack: BuildPack | null; note: string | null }) {
  if (!pack) {
    return (
      <p className="border border-dashed border-rule-strong p-3 text-sm text-ink-2" data-build-pack="none">
        {note ?? "No build pack has been exported for this use case yet."}
      </p>
    );
  }
  return (
    <div className="grid gap-4" data-build-pack>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <a
          href={dataUrl(pack.zip)}
          download
          className="inline-flex h-9 items-center gap-2 bg-accent px-3 text-sm text-accent-ink"
          data-build-pack-link
        >
          <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
            <path d="M6 1v7M3 5l3 3 3-3M1.5 10.5h9" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          Download the build pack (zip, {formatBytes(pack.bytes)})
        </a>
        <span className="num text-xs text-ink-3">sha256 {pack.sha256.slice(0, 16)}</span>
      </div>
      <p className="text-sm text-ink-2">
        For the {pack.for === "recommended" ? "recommended" : "closest"} setup: <span className="text-ink">{pack.label}</span>.
        {note ? ` ${note}` : ""}
      </p>
      <div className="relative overflow-x-auto" role="region" aria-label="Files in the build pack" tabIndex={0}>
        <table className="w-full min-w-[620px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-y border-rule-strong text-xs text-ink-3">
              <th scope="col" className="py-2 pr-3 font-normal">
                File
              </th>
              <th scope="col" className="py-2 pr-3 font-normal">
                What
              </th>
              <th scope="col" className="py-2 pr-3 font-normal">
                Licence
              </th>
              <th scope="col" className="py-2 text-right font-normal">
                Size
              </th>
            </tr>
          </thead>
          <tbody>
            {pack.files.map((f) => (
              <tr key={f.path} className="border-b border-rule align-top">
                <th scope="row" className="num break-all py-1.5 pr-3 text-xs font-normal text-ink">
                  {f.path}
                </th>
                <td className="py-1.5 pr-3 text-xs text-ink-2">
                  {KIND_LABELS[f.kind] ?? f.kind.replace(/_/g, " ")}, {f.format.toUpperCase()}
                </td>
                <td className="num py-1.5 pr-3 text-xs">{f.license}</td>
                <td className="num py-1.5 text-right text-xs">{formatBytes(f.bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {pack.links.length ? (
        <div>
          <h3 className="heading text-base">Upstream models</h3>
          <ul className="mt-2 grid gap-1 text-sm">
            {pack.links.map((l) => (
              <li key={`${l.name}-${l.url}`}>
                <a className="link" href={l.url} rel="noopener noreferrer">
                  {l.name}
                </a>{" "}
                <span className="num text-xs text-ink-3">{l.license}</span>
                {l.reason ? <span className="text-xs text-ink-3"> · {l.reason}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {pack.caveats.length ? (
        <ul className="grid gap-1 border-l-2 border-pending pl-2 text-sm text-pending">
          {pack.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
