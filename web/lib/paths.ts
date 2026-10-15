/**
 * URLs of runtime-fetched files. Next.js prefixes routes and <Link> hrefs with the
 * base path itself, but not fetch() or plain <a href> to files, so those go through
 * here. NEXT_PUBLIC_BASE_PATH is inlined at build time (see next.config.ts).
 */

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/** URL of a file under public/data, given its path relative to /data. */
export function dataUrl(relative: string): string {
  return `${BASE_PATH}/data/${relative.replace(/^\/+/, "")}`;
}
