import type { ReactNode } from "react";

/** Page heading block: an optional kicker (breadcrumb), the title, a lede and an aside (chips, actions). */
export function PageHeader({
  title,
  lede,
  kicker,
  children,
}: {
  title: string;
  lede?: ReactNode;
  kicker?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="pt-10 sm:pt-16">
      {kicker ? <div className="mb-4 text-sm text-ink-3">{kicker}</div> : null}
      <h1 className="display text-[clamp(2.2rem,5.4vw,3.9rem)] text-ink">{title}</h1>
      {lede ? <div className="prose-measure mt-5 text-lg text-ink-2">{lede}</div> : null}
      {children ? <div className="mt-6">{children}</div> : null}
    </header>
  );
}
