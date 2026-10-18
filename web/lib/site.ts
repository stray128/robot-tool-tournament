/**
 * Product naming and site-wide constants. The product name lives here and nowhere
 * else; the repository and the Python package keep their own names.
 */

export const APP_NAME = "Effector Lab";

export const APP_TAGLINE = "Bring your part. Get fingers that hold it.";

export const APP_DESCRIPTION =
  "Upload a part. NVIDIA Nemotron designs gripper fingers and other end-effectors for it, and physics simulation on Nebius tests them under placement error before you print anything.";

/** Public source repository, set at build time; links are hidden while it is unset. */
export const REPO_URL: string | null = process.env.NEXT_PUBLIC_REPO_URL || null;

export interface NavItem {
  href: string;
  label: string;
  /** Hidden on phones, where the product mark already links here. */
  wideOnly?: boolean;
}

/** Primary navigation, in reading order. */
export const NAV: readonly NavItem[] = [
  { href: "/", label: "Overview", wideOnly: true },
  { href: "/part/", label: "Your part" },
  { href: "/effectors/", label: "Effectors" },
  { href: "/library/", label: "Library" },
  { href: "/studio/", label: "Studio" },
  { href: "/benchmark/", label: "Benchmark" },
  { href: "/results/", label: "Results" },
  { href: "/compute/", label: "Compute" },
  { href: "/about/", label: "Method" },
];
