"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { NAV } from "@/lib/site";

function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  const base = href.replace(/\/$/, "");
  if (pathname === base || pathname.startsWith(`${base}/`)) return true;
  // Detail pages belong to the benchmark section.
  return (
    base === "/benchmark" &&
    (pathname.startsWith("/bout/") || pathname.startsWith("/design/") || pathname.startsWith("/certificate/"))
  );
}

export function NavLinks() {
  const pathname = usePathname() ?? "/";
  return (
    <nav aria-label="Primary" className="order-3 w-full sm:order-none sm:w-auto sm:flex-1">
      <ul className="flex flex-wrap gap-x-3 gap-y-1 text-[0.8125rem] sm:gap-x-5 sm:text-[0.9375rem]">
        {NAV.map((item) => {
          const active = isActive(pathname, item.href);
          return (
            <li key={item.href} className={item.wideOnly ? "hidden sm:list-item" : undefined}>
              <Link
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={
                  active
                    ? "text-ink underline decoration-accent decoration-2 underline-offset-[6px]"
                    : "text-ink-2 no-underline hover:text-ink"
                }
              >
                {item.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
