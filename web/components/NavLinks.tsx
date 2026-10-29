"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { NAV, type NavItem } from "@/lib/site";

function isActive(pathname: string, item: NavItem): boolean {
  return [item.href, ...(item.also ?? [])].some((href) => {
    const base = href.replace(/\/$/, "");
    return pathname === base || pathname.startsWith(`${base}/`);
  });
}

function External() {
  return (
    <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
      <path d="M4 2h6v6M10 2L3 9" fill="none" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

/** The four sections and the source link: a row on wide screens, a menu on phones. */
export function NavLinks({ githubUrl }: { githubUrl: string }) {
  const pathname = usePathname() ?? "/";
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const links = (mobile: boolean) => (
    <>
      {NAV.map((item) => {
        const active = isActive(pathname, item);
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`block rounded-md no-underline transition-colors ${mobile ? "px-3 py-2.5 text-base" : "px-3 py-1.5 text-[0.9375rem]"} ${
                active ? "bg-paper-2 text-ink" : "text-ink-2 hover:bg-paper-2/70 hover:text-ink"
              }`}
            >
              {item.label}
              {active && !mobile ? <span className="mx-auto mt-0.5 block h-0.5 w-4 rounded-full bg-accent" aria-hidden="true" /> : null}
            </Link>
          </li>
        );
      })}
      <li>
        <a
          href={githubUrl}
          className={`flex items-center gap-1 rounded-md text-ink-2 no-underline hover:bg-paper-2/70 hover:text-ink ${mobile ? "px-3 py-2.5 text-base" : "px-3 py-1.5 text-[0.9375rem]"}`}
          rel="noopener"
        >
          GitHub <External />
        </a>
      </li>
    </>
  );

  return (
    <>
      <nav aria-label="Primary" className="ml-auto hidden md:block">
        <ul className="flex items-center gap-0.5">{links(false)}</ul>
      </nav>
      <div className="relative ml-auto md:hidden" ref={panel}>
        <button
          type="button"
          aria-expanded={open}
          aria-controls="mobile-nav"
          onClick={() => setOpen((v) => !v)}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-rule-strong px-2.5 text-sm text-ink"
          aria-label="Menu"
        >
          <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true">
            {open ? (
              <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" />
            ) : (
              <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.6" />
            )}
          </svg>
          <span className="hidden min-[420px]:inline">Menu</span>
        </button>
        {open ? (
          <nav
            id="mobile-nav"
            aria-label="Primary"
            className="panel absolute right-0 top-11 z-50 w-56 p-1.5"
          >
            <ul className="grid gap-0.5">{links(true)}</ul>
          </nav>
        ) : null}
      </div>
    </>
  );
}
