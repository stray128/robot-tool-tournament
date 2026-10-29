/**
 * Product naming and site-wide constants. The product name lives here and nowhere
 * else; the repository and the Python package keep their own names.
 */

export const APP_NAME = "Effector Lab";

export const APP_TAGLINE = "Find the end effector your robot needs, tested in physics first.";

export const APP_DESCRIPTION =
  "Describe what your robot has to do. Effector Lab plans it as steps, tries every tool it can simulate on every step in physics, and recommends a single tool, a tool change or a two-tool head, next to the best single off-the-shelf tool. Open source, simulated, verify on hardware.";

/** Key under which the home page hands the visitor's text to the Studio planner (this tab only). */
export const PREFILL_KEY = "effector-lab:prefill";

/** Public source repository, set at build time; links are hidden while it is unset. */
export const REPO_URL: string | null = process.env.NEXT_PUBLIC_REPO_URL || null;

/** The repository the navigation's GitHub link opens (the build-time URL when set). */
export const GITHUB_URL: string = REPO_URL ?? "https://github.com/stray128/robot-tool-tournament";

/** A file or folder of the public repository, for Docs links. */
export function repoBlob(path: string): string {
  return `${GITHUB_URL}/blob/main/${path.replace(/^\/+/, "")}`;
}

export interface NavItem {
  href: string;
  label: string;
  /** Older routes that belong to this section (for the active state). */
  also?: readonly string[];
}

/** Primary navigation, in reading order. */
export const NAV: readonly NavItem[] = [
  { href: "/studio/", label: "Studio", also: ["/part/"] },
  { href: "/atlas/", label: "Use cases" },
  { href: "/library/", label: "Library" },
  {
    href: "/evidence/",
    label: "Evidence",
    also: ["/results/", "/benchmark/", "/bout/", "/design/", "/certificate/", "/compute/", "/about/", "/effectors/"],
  },
  { href: "/docs/", label: "Docs", also: ["/credits/"] },
];
