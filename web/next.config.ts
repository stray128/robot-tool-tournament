import type { NextConfig } from "next";

/**
 * Static export for GitHub Pages (or any static host).
 *
 * NEXT_PUBLIC_BASE_PATH serves the site from a sub-path, e.g. "/robot-tool-tournament"
 * for a GitHub Pages project site. It is read here for routing and again at runtime by
 * lib/paths.ts when fetching replay files.
 */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: basePath || undefined,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
