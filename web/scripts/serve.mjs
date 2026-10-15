#!/usr/bin/env node
/**
 * Minimal static server for the exported site (web/out), for local checks only.
 *
 *   node scripts/serve.mjs [--dir out] [--port 4173] [--base /robot-tool-tournament]
 *
 * Mirrors GitHub Pages closely enough for testing: "/x/" serves "/x/index.html",
 * unknown paths get 404.html with status 404, and NEXT_PUBLIC_BASE_PATH builds can
 * be served under --base.
 */

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".bin": "application/octet-stream",
  ".glb": "model/gltf-binary",
  ".stl": "model/stl",
};

/** Resolve a URL path to a file inside `root`, or null (also for a malformed escape). */
export function resolveFile(root, urlPath) {
  let rel;
  try {
    rel = decodeURIComponent(urlPath.split("?")[0] ?? "/");
  } catch {
    return null;
  }
  if (rel.endsWith("/")) rel += "index.html";
  const base = path.resolve(root);
  const file = path.normalize(path.join(base, rel));
  if (file !== base && !file.startsWith(base + path.sep)) return null;
  if (existsSync(file) && statSync(file).isFile()) return file;
  if (existsSync(`${file}.html`)) return `${file}.html`;
  if (existsSync(path.join(file, "index.html"))) return path.join(file, "index.html");
  return null;
}

export function startServer({ dir, port = 4173, base = "" }) {
  const root = path.resolve(dir);
  const server = createServer((req, res) => {
    let urlPath = req.url ?? "/";
    if (base) {
      if (!urlPath.startsWith(base)) {
        res.writeHead(404).end();
        return;
      }
      urlPath = urlPath.slice(base.length) || "/";
    }
    const file = resolveFile(root, urlPath);
    const target = file ?? path.join(root, "404.html");
    res.writeHead(file ? 200 : 404, {
      "content-type": TYPES[path.extname(target)] ?? "application/octet-stream",
      "cache-control": "no-store",
    });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    createReadStream(target).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({
    options: {
      dir: { type: "string", default: path.join(webDir, "out") },
      port: { type: "string", default: "4173" },
      base: { type: "string", default: process.env.NEXT_PUBLIC_BASE_PATH ?? "" },
    },
  });
  const server = await startServer({ dir: values.dir, port: Number(values.port), base: values.base });
  const { port } = server.address();
  console.log(`serving ${values.dir} at http://127.0.0.1:${port}${values.base}/`);
}
