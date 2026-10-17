import Link from "next/link";

import { PRODUCT_MODE_LABEL } from "@/lib/effectors";

/**
 * The product-mode banner, above every effector result. The label is word for word the one
 * every record, lane and agent answer carries; effector results never appear on the benchmark,
 * results or pre-registration pages.
 */
export function ProductModeBanner({ detail = true }: { detail?: boolean }) {
  return (
    <aside
      className="mt-8 flex flex-wrap items-baseline gap-x-3 gap-y-1 border border-pending/60 bg-paper px-3 py-2 text-sm"
      aria-label="Product mode"
      data-product-mode
    >
      <strong className="num text-xs uppercase tracking-wide text-pending">{PRODUCT_MODE_LABEL}</strong>
      {detail ? (
        <span className="text-ink-2">
          A simulation ranking under stated conditions, not a hardware guarantee. The benchmark&apos;s pre-registered
          result is on{" "}
          <Link className="link" href="/results/">
            Results
          </Link>
          .
        </span>
      ) : null}
    </aside>
  );
}
