import Link from "next/link";

import { Mark } from "@/components/Mark";
import { APP_NAME, GITHUB_URL, repoBlob } from "@/lib/site";

type FooterLink = { href: string; label: string; external?: boolean };

function columns(example: FooterLink | null): { title: string; links: FooterLink[] }[] {
  return [
    {
      title: "Product",
      links: [
        { href: "/studio/", label: "Studio" },
        ...(example ? [example] : []),
        { href: "/atlas/", label: "Use-case atlas" },
        { href: "/library/", label: "Tool library" },
      ],
    },
    {
      title: "Evidence",
      links: [
        { href: "/evidence/", label: "All results" },
        { href: "/results/", label: "Finger designer result" },
        { href: "/compute/", label: "Compute log" },
      ],
    },
    {
      title: "Open source",
      links: [
        { href: "/docs/", label: "Docs" },
        { href: "/docs/quickstart/", label: "Quickstart" },
        { href: "/credits/", label: "Credits and licences" },
        { href: GITHUB_URL, label: "Source on GitHub", external: true },
      ],
    },
  ];
}

/** Licence, credits and the simulation note, on every page. */
export function SiteFooter({ example = null }: { example?: FooterLink | null }) {
  const COLUMNS = columns(example);
  return (
    <footer className="mt-24 border-t border-rule bg-surface/60">
      <div className="mx-auto grid w-full max-w-[1240px] gap-10 px-4 py-12 sm:px-6 lg:grid-cols-[minmax(0,1.3fr)_repeat(3,minmax(0,1fr))] lg:px-8">
        <div className="max-w-[40ch]">
          <p className="flex items-center gap-2 text-ink">
            <Mark className="size-5" />
            <span className="heading">{APP_NAME}</span>
          </p>
          <p className="mt-3 text-sm text-ink-2">
            Every result on this site is a physics simulation under stated conditions. It ranks tools; it does not
            guarantee hardware. Simulated, verify on hardware.
          </p>
          <p className="mt-3 text-sm text-ink-3">
            Code and data are open source under the{" "}
            <a className="link" href={repoBlob("LICENSE")} rel="noopener">
              Apache-2.0 licence
            </a>
            . Third-party models keep their own licences, listed on{" "}
            <Link className="link" href="/credits/">
              Credits
            </Link>
            .
          </p>
        </div>
        {COLUMNS.map((col) => (
          <nav key={col.title} aria-label={col.title}>
            <h2 className="text-sm text-ink-3">{col.title}</h2>
            <ul className="mt-3 grid gap-2 text-sm">
              {col.links.map((l) => (
                <li key={l.href}>
                  {l.external ? (
                    <a className="text-ink-2 no-underline hover:text-ink" href={l.href} rel="noopener">
                      {l.label} <span aria-hidden="true">↗</span>
                    </a>
                  ) : (
                    <Link className="text-ink-2 no-underline hover:text-ink" href={l.href}>
                      {l.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </nav>
        ))}
      </div>
    </footer>
  );
}
