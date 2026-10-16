#!/usr/bin/env node
/**
 * Publish a recorded Grasp Engineer session to the static site.
 *
 *   node scripts/import-session.mjs --in runs/agent/sessions/sess-<id>.json --stl part.stl
 *        [--pack runs/agent/printpacks/pack-<id>/print_pack] [--id ID] [--title TITLE]
 *        [--note TEXT] [--keep-messages]
 *
 * Reads the agent's session file (src/rtt/agent/recorder.py, `format:
 * "rtt-agent-session"`), copies the part's STL to public/data/sessions/parts/<id>.stl
 * and writes public/data/sessions/<id>.json with `kind: "recorded"` and `part.stl` set,
 * so /part can preview the part and play the session. The chat transcript (`messages`,
 * which repeats the system prompt) is dropped unless --keep-messages. Refuses a file
 * that is not a session, has no events, or contains anything that looks like a local
 * filesystem path, a Nebius resource ID other than an AI job ID, or a container
 * registry path. Run scripts/leak_check.sh on the result before committing it.
 *
 * With --pack, the files of that exported print pack (its unzipped directory) are
 * copied to public/data/printpacks/<pack_id>/ and each file listed by the session's
 * export_print_pack result gets a data-relative `path`, so the static site links the
 * finger STLs and the certificate for download. Only files the result lists are copied.
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const SLUG = /^[a-z0-9][a-z0-9._-]*$/;
/** Absolute paths on Linux, macOS and Windows, or a home-directory shorthand. */
const LOCAL_PATH = /(^|[\s"'(=:/])(\/(home|Users|tmp|root|var|private|mnt)\/|[A-Za-z]:\\|~\/)/;

/** Strings in `value` that look like local paths, with their JSON path (at most 5). */
export function findLocalPaths(value, where = "$", out = []) {
  if (out.length >= 5) return out;
  if (typeof value === "string") {
    if (LOCAL_PATH.test(value)) out.push(`${where}: ${value.slice(0, 80)}`);
  } else if (Array.isArray(value)) {
    value.forEach((v, i) => findLocalPaths(v, `${where}[${i}]`, out));
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) findLocalPaths(v, `${where}.${k}`, out);
  }
  return out;
}

/**
 * Nebius resource IDs (`<type>-e0...`: project, tenant, subnet, registry, bucket,
 * secret...) and container-registry references, which must never be published. AI job
 * IDs (`aijob-...`) are the one exception: the site shows them as provenance. Agent IDs
 * (`sess-`/`part-`/`run-`/`pack-` plus 12 hex digits) are too short to match.
 */
const NEBIUS_ID = /\b(?!aijob-)[a-z][a-z0-9]*-e0[0-9a-z]{14,}\b/i;
const REGISTRY_REF = /\bcr\.[a-z0-9-]+\.nebius\.cloud\b/i;

/** JSON paths of strings in `value` holding a Nebius resource ID or registry path (at most 5; the value itself is not echoed). */
export function findPrivateIds(value, where = "$", out = []) {
  if (out.length >= 5) return out;
  if (typeof value === "string") {
    if (NEBIUS_ID.test(value) || REGISTRY_REF.test(value)) out.push(where);
  } else if (Array.isArray(value)) {
    value.forEach((v, i) => findPrivateIds(v, `${where}[${i}]`, out));
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) findPrivateIds(v, `${where}.${k}`, out);
  }
  return out;
}

/**
 * The publishable session: the agent's document with `id`, `kind`, `title`, `note`,
 * `recorded_at` and `part.stl` added and the chat transcript dropped (unless kept).
 *
 * @param {any} doc
 * @param {{ id: string, stlPath: string, title?: string, note?: string, keepMessages?: boolean }} opts
 */
export function publishableSession(doc, { id, stlPath, title, note, keepMessages = false, packFiles = null }) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc)) throw new Error("session: expected a JSON object");
  if ((doc.format ?? doc.schema) !== "rtt-agent-session") throw new Error("session: not an rtt-agent-session file");
  if (!Array.isArray(doc.events) || doc.events.length === 0) throw new Error("session: no events");
  if (!SLUG.test(id)) throw new Error(`session: id ${id} is not URL-safe`);
  const { messages, ...rest } = doc;
  if (packFiles) rest.events = withPackPaths(rest.events, packFiles);
  const part = { ...(rest.part ?? {}), stl: stlPath, units: "mm" };
  if (typeof part.name === "string" && !part.name.toLowerCase().endsWith(".stl")) part.name = `${part.name}.stl`;
  const out = {
    ...rest,
    id,
    kind: "recorded",
    title: title ?? (typeof doc.part?.name === "string" ? doc.part.name : id),
    note: note ?? "",
    recorded_at: doc.created_at ?? null,
    part,
    ...(keepMessages && messages ? { messages } : {}),
  };
  const leaks = findLocalPaths(out);
  if (leaks.length) throw new Error(`session: refusing to publish local paths:\n  ${leaks.join("\n  ")}`);
  const ids = findPrivateIds(out);
  if (ids.length) throw new Error(`session: refusing to publish Nebius resource IDs or registry paths at:\n  ${ids.join("\n  ")}`);
  return out;
}

/**
 * Events with a data-relative `path` added to each file of a successful
 * export_print_pack result whose pack ID has entries in `packFiles`
 * ({ [packId]: { [fileName]: relPath } }). Nothing else changes.
 */
export function withPackPaths(events, packFiles) {
  return events.map((ev) => {
    if (ev?.type !== "tool.result" || ev.name !== "export_print_pack" || !ev.ok) return ev;
    const result = ev.result ?? {};
    const byName = packFiles[result.pack_id];
    if (!byName || !Array.isArray(result.files)) return ev;
    const files = result.files.map((f) => (f && byName[f.name] ? { ...f, path: byName[f.name] } : f));
    return { ...ev, result: { ...result, files } };
  });
}

/** The pack ID and listed file names of the session's (last) successful print pack export. */
function exportedPack(doc) {
  const ev = [...doc.events].reverse().find((e) => e?.type === "tool.result" && e.name === "export_print_pack" && e.ok);
  if (!ev || !SLUG.test(String(ev.result?.pack_id ?? ""))) return null;
  return { packId: ev.result.pack_id, names: (ev.result.files ?? []).map((f) => f?.name).filter((n) => SLUG.test(String(n ?? "").toLowerCase())) };
}

function main() {
  const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const { values } = parseArgs({
    options: {
      in: { type: "string" },
      stl: { type: "string" },
      id: { type: "string" },
      title: { type: "string" },
      note: { type: "string" },
      "keep-messages": { type: "boolean", default: false },
      pack: { type: "string" },
    },
  });
  if (!values.in || !values.stl) {
    console.error("import-session: --in SESSION.json and --stl PART.stl are required");
    process.exit(2);
  }
  const doc = JSON.parse(readFileSync(values.in, "utf8"));
  const id = (values.id ?? doc.session_id ?? "").toLowerCase();
  const stlRel = `sessions/parts/${id}.stl`;
  const dataDir = path.join(webDir, "public", "data");
  let packFiles = null;
  const copies = [];
  if (values.pack) {
    const exported = Array.isArray(doc.events) ? exportedPack(doc) : null;
    if (!exported) {
      console.error("import-session: --pack given but the session exported no print pack");
      process.exit(2);
    }
    const byName = {};
    for (const name of exported.names) {
      const src = path.join(values.pack, name);
      if (!existsSync(src)) continue;
      const rel = `printpacks/${exported.packId}/${name}`;
      byName[name] = rel;
      copies.push([src, path.join(dataDir, rel)]);
    }
    packFiles = { [exported.packId]: byName };
  }
  const out = publishableSession(doc, {
    id,
    stlPath: stlRel,
    title: values.title,
    note: values.note,
    keepMessages: values["keep-messages"],
    packFiles,
  });
  for (const [src, dest] of copies) {
    mkdirSync(path.dirname(dest), { recursive: true });
    copyFileSync(src, dest);
  }
  if (copies.length) console.log(`import-session: copied ${copies.length} print-pack files to public/data/printpacks/`);
  mkdirSync(path.join(dataDir, "sessions", "parts"), { recursive: true });
  copyFileSync(values.stl, path.join(dataDir, stlRel));
  const file = path.join(dataDir, "sessions", `${id}.json`);
  writeFileSync(file, `${JSON.stringify(out, null, 1)}\n`, "utf8");
  console.log(`import-session: wrote ${path.relative(webDir, file)} and ${path.join("public/data", stlRel)}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
