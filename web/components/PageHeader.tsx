import type { ReactNode } from "react";

/** Page heading block: title, optional lede and an optional aside (chips, actions). */
export function PageHeader({ title, lede, children }: { title: string; lede?: ReactNode; children?: ReactNode }) {
  return (
    <header className="pt-10 sm:pt-14">
      <h1 className="display text-[clamp(2.1rem,5.6vw,3.75rem)]">{title}</h1>
      {lede ? <div className="prose-measure mt-4 text-lg text-ink-2">{lede}</div> : null}
      {children ? <div className="mt-5">{children}</div> : null}
    </header>
  );
}
