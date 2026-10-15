import Link from "next/link";

import { Mark } from "@/components/Mark";
import { NavLinks } from "@/components/NavLinks";
import { ThemeToggle } from "@/components/ThemeToggle";
import { APP_NAME } from "@/lib/site";

export function SiteHeader() {
  return (
    <header className="border-b border-rule bg-paper/90 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-[1200px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2 text-ink no-underline">
          <Mark className="size-6" />
          <span className="heading text-lg">{APP_NAME}</span>
        </Link>
        <NavLinks />
        <ThemeToggle />
      </div>
    </header>
  );
}
