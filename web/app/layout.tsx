import type { Metadata, Viewport } from "next";
import { Archivo, Atkinson_Hyperlegible_Next, Red_Hat_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { APP_DESCRIPTION, APP_NAME, APP_TAGLINE } from "@/lib/site";

import { featuredHero } from "./studio/data";

import "./globals.css";

/** Display: an expanded grotesk (its width axis drives the headings). */
const archivo = Archivo({
  subsets: ["latin"],
  axes: ["wdth"],
  variable: "--font-archivo",
  display: "swap",
});

/** Text: a face drawn for legibility, for running text and controls. */
const text = Atkinson_Hyperlegible_Next({
  subsets: ["latin"],
  variable: "--font-text",
  display: "swap",
});

/** Numbers and ids: a compact mono with clear decimals. */
const mono = Red_Hat_Mono({
  subsets: ["latin"],
  variable: "--font-mono-face",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: `${APP_NAME}: ${APP_TAGLINE}`, template: `%s | ${APP_NAME}` },
  description: APP_DESCRIPTION,
  applicationName: APP_NAME,
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#15181c",
};

/** Applies a saved light/dark choice before first paint (no flash of the wrong theme). Dark is the default. */
const THEME_SCRIPT = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

/** The footer's worked-example link: the featured hero of the Studio's data, when there is one. */
function footerExample(): { href: string; label: string } | null {
  const h = featuredHero();
  return h ? { href: `/studio/${h.hero}/`, label: `Worked example: ${h.title.charAt(0).toLowerCase()}${h.title.slice(1)}` } : null;
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${text.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-dvh flex-col antialiased">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:text-ink"
        >
          Skip to content
        </a>
        <SiteHeader />
        <main id="main" className="mx-auto w-full max-w-[1240px] flex-1 px-4 sm:px-6 lg:px-8">
          {children}
        </main>
        <SiteFooter example={footerExample()} />
      </body>
    </html>
  );
}
