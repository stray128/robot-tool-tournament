/**
 * Product naming and site-wide constants. The product name is not final: change
 * APP_NAME here and nowhere else.
 */

export const APP_NAME = "GripForge";

export const APP_TAGLINE = "Bring your part. Get fingers that hold it.";

export const APP_DESCRIPTION =
  "Upload a part, let an NVIDIA Nemotron model design parallel-gripper fingers for it, and see them proven in physics simulation on Nebius before you print them.";

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
  { href: "/benchmark/", label: "Benchmark" },
  { href: "/results/", label: "Results" },
  { href: "/compute/", label: "Compute" },
  { href: "/about/", label: "Method" },
];
