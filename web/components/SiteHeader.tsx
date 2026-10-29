import Link from "next/link";

import { Mark } from "@/components/Mark";
import { NavLinks } from "@/components/NavLinks";
import { ThemeToggle } from "@/components/ThemeToggle";
import { APP_NAME, GITHUB_URL } from "@/lib/site";

/** The bar on every page: the product, its four sections, the source and the theme. */
export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-rule bg-paper/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 w-full max-w-[1240px] items-center gap-2.5 px-4 sm:gap-3 sm:px-6 lg:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-2 text-ink no-underline" aria-label={`${APP_NAME}, home`}>
          <Mark className="size-6" />
          <span className="heading text-[1.0625rem]">{APP_NAME}</span>
        </Link>
        <NavLinks githubUrl={GITHUB_URL} />
        <ThemeToggle />
      </div>
    </header>
  );
}
