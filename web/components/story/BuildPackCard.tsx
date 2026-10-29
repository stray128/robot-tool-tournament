import type { BuildPack } from "@/components/studio/schema";
import { formatBytes } from "@/lib/format";
import { dataUrl } from "@/lib/paths";

/** What each kind of file in a build pack is for, in the order a builder needs them. */
const KINDS: { kind: string; title: string; text: string }[] = [
  { kind: "bracket", title: "Printed bracket", text: "Holds the tools at the tested angles; print it." },
  { kind: "changer_plate", title: "Changer plates", text: "Mount each tool on the tool changer." },
  { kind: "adapter", title: "Adapters", text: "Mount the tools on the arm's flange." },
  { kind: "effector_cad", title: "Tool reference geometry", text: "The tools as simulated, for fit checks." },
  { kind: "bom", title: "Bill of materials", text: "What to buy or print, with quantities." },
  { kind: "notes", title: "Assembly notes", text: "How it goes together, and a read-me." },
  { kind: "mjcf", title: "Simulation scenes", text: "The tested scenes, to re-run or extend." },
  { kind: "certificate", title: "Certificate", text: "The runs behind the recommendation." },
  { kind: "licence", title: "Credits", text: "Licences of everything inside." },
];

/** The build pack: what is inside, and the download. */
export function BuildPackCard({ pack, note }: { pack: BuildPack | null; note: string | null }) {
  if (!pack) {
    return (
      <p className="panel p-4 text-sm text-ink-2" data-build-pack="none">
        {note ?? "No build pack has been exported for this use case yet."}
      </p>
    );
  }
  const groups = KINDS.map((k) => ({ ...k, files: pack.files.filter((f) => f.kind === k.kind) })).filter((g) => g.files.length);
  const other = pack.files.filter((f) => !KINDS.some((k) => k.kind === f.kind));
  return (
    <div className="panel grid gap-6 p-4 sm:p-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]" data-build-pack>
      <div>
        <p className="text-sm text-ink-3">For the {pack.for === "recommended" ? "recommended" : "closest"} setup</p>
        <p className="heading mt-1 text-xl text-ink">{pack.label}</p>
        <p className="mt-3 text-sm text-ink-2">
          {pack.files.length} files in one zip, every one under an open licence. Print the parts, buy the tools on the bill of materials, and check the
          result on your hardware before you rely on it.
        </p>
        <a href={dataUrl(pack.zip)} download className="btn btn-primary mt-5" data-build-pack-link>
          <svg viewBox="0 0 12 12" className="size-3.5" aria-hidden="true">
            <path d="M6 1v7M3 5l3 3 3-3M1.5 10.5h9" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          Download the build pack
        </a>
        <p className="num mt-2 text-xs text-ink-3">
          zip, {formatBytes(pack.bytes)}, sha256 {pack.sha256.slice(0, 12)}
        </p>
      </div>
      <ul className="grid gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-2">
        {groups.map((g) => (
          <li key={g.kind} className="bg-surface px-3 py-2.5">
            <p className="text-sm font-semibold text-ink">{g.title}</p>
            <p className="text-sm text-ink-2">{g.text}</p>
            <p className="num mt-1 break-all text-xs text-ink-3">{g.files.map((f) => f.path).join(", ")}</p>
          </li>
        ))}
        {other.map((f) => (
          <li key={f.path} className="bg-surface px-3 py-2.5">
            <p className="num break-all text-xs text-ink-3">{f.path}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
