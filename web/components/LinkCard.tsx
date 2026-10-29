import Link from "next/link";

/** A card that links to one page of the site or one file of the repository. */
export function LinkCard({ href, title, text, meta, external = false }: { href: string; title: string; text: string; meta?: string; external?: boolean }) {
  const body = (
    <>
      <span className="heading block text-base text-ink">
        {title}
        {external ? <span aria-hidden="true"> ↗</span> : null}
      </span>
      <span className="mt-1 block text-sm text-ink-2">{text}</span>
      {meta ? <span className="num mt-3 block text-xs text-ink-3">{meta}</span> : null}
    </>
  );
  const cls = "panel block h-full p-4 no-underline transition-colors hover:border-rule-strong";
  return external ? (
    <a href={href} rel="noopener" className={cls}>
      {body}
    </a>
  ) : (
    <Link href={href} className={cls}>
      {body}
    </Link>
  );
}
