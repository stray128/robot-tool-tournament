import type { Metadata } from "next";
import Link from "next/link";

import { FidelityBadge } from "@/components/effectors/FidelityBadge";
import { ProductModeBanner } from "@/components/effectors/ProductModeBanner";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { FAMILY_TITLES, type IndexWinner, score, toleranceText } from "@/lib/effectors";

import { getEffectorIndex } from "./data";

export const metadata: Metadata = {
  title: "Which effector for this part?",
  description:
    "Product mode: every end-effector family tried on the same part, placements and seeds, with one recommendation and its robustness certificate.",
};

function Winner({ w, empty }: { w: IndexWinner | null; empty: string }) {
  if (!w) return <span className="text-ink-3">{empty}</span>;
  return (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="text-ink">{w.display_name}</span>
      <FidelityBadge fidelity={w.fidelity} />
      <span className="num text-xs text-ink-3">
        score {score(w.score)} · {toleranceText(w.tolerance_mm)}
      </span>
      {w.caveat ? <span className="text-xs text-pending">{w.caveat}</span> : null}
    </span>
  );
}

export default function EffectorsIndexPage() {
  const index = getEffectorIndex();
  return (
    <>
      <ProductModeBanner />
      <PageHeader
        title="Which effector for this part?"
        lede={
          <p>
            For each part, every end-effector family we can simulate (the parallel jaw, multi-finger hands, grippers,
            suction cups, a soft arm and designed morphologies) grasps it on the same placements, seeds and physics. The
            best of each family are certified, and one effector is recommended with its certificate.
          </p>
        }
      />

      <section className="mt-10" aria-labelledby="families-title">
        <h2 id="families-title" className="heading text-xl">
          Families in this build
        </h2>
        <ul className="mt-3 flex flex-wrap gap-2 text-sm">
          {(index?.families ?? []).map((f) => (
            <li key={f.family} className={`flex items-center gap-2 border px-2 py-1 ${f.available ? "border-rule-strong" : "border-rule text-ink-3"}`}>
              <span>{f.title}</span>
              <FidelityBadge fidelity={f.fidelity} />
              <span className="text-xs">{f.available ? "available" : "not built yet"}</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-12" aria-labelledby="parts-title">
        <h2 id="parts-title" className="heading text-xl">
          Training parts
        </h2>
        {index && index.parts.length > 0 ? (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-y border-rule-strong text-xs text-ink-3">
                  <th scope="col" className="py-2 pr-4 font-normal">Part</th>
                  <th scope="col" className="py-2 pr-4 font-normal">Recommended (registered)</th>
                  <th scope="col" className="py-2 pr-4 font-normal">Recommended (held)</th>
                  <th scope="col" className="py-2 text-right font-normal">Episodes</th>
                </tr>
              </thead>
              <tbody>
                {index.parts.map((p) => (
                  <tr key={p.part_id} className="border-b border-rule align-top">
                    <th scope="row" className="py-2.5 pr-4 font-normal">
                      <Link href={`/effectors/${p.part_id}/`} className="link">
                        {p.title}
                      </Link>
                      <span className="num mt-0.5 block text-xs text-ink-3">
                        {p.size_mm.map((v) => v.toFixed(0)).join(" x ")} mm · groups: {p.groups_run.length}
                      </span>
                    </th>
                    <td className="py-2.5 pr-4">
                      <Winner w={p.by_requirement.registered} empty="No winner" />
                    </td>
                    <td className="py-2.5 pr-4">
                      <Winner w={p.by_requirement.held} empty="No winner" />
                    </td>
                    <td className="num py-2.5 text-right">{p.episodes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-3 text-ink-3">No tournament has been exported yet.</p>
        )}
        <p className="mt-4 max-w-[80ch] text-sm text-ink-3">
          Registered is the headline: the part is lifted, held and back in the same pose whatever the placement error,
          which a downstream placement needs. Held only asks that it is lifted and held. Families not built yet are not
          in the ranking. {FAMILY_TITLES.parallel_jaw} results use the core experiment&apos;s own scene and scorer.
        </p>
      </section>

      <section className="mt-12 grid gap-6 md:grid-cols-2" aria-label="Your own part">
        <div>
          <h2 className="heading text-xl">Your own part</h2>
          <p className="mt-2 text-ink-2">
            The Grasp Engineer can run the same tournament on an uploaded part: ask it which effector suits the part on{" "}
            <Link className="link" href="/part/">
              Bring your part
            </Link>
            .
          </p>
        </div>
      </section>

      <TitleBlock title="Which effector for this part?" drawing="/effectors" sheet="Product mode" />
    </>
  );
}
