import Link from "next/link";

import { replayCredits } from "@/lib/assets";
import type { ReplayAsset } from "@/lib/types";

/**
 * The short credit of every third-party model a clip draws (from its replay header),
 * each linked to the full credit line on the credits page. Renders nothing when the
 * clip draws only our own geometry.
 */
export function ReplayCredits({ assets, className = "" }: { assets: Record<string, ReplayAsset>; className?: string }) {
  const credits = replayCredits(assets);
  if (credits.length === 0) return null;
  return (
    <span className={`block text-xs text-ink-3 ${className}`} data-credits>
      {credits.map((c, i) => (
        <span key={c.text}>
          {i > 0 ? "; " : "Models: "}
          {c.text}{" "}
          <Link href={`/credits/#${c.anchor ?? "models"}`} className="link whitespace-nowrap">
            (full credit)
          </Link>
        </span>
      ))}
    </span>
  );
}
