#!/usr/bin/env node
/**
 * The shot list's shared rules, for capture.mjs, assemble.sh and the tests.
 *
 *   node scripts/video/shots.mjs check   [--shots FILE] [--flag NAME]... [--final]
 *   node scripts/video/shots.mjs resolve [--shots FILE] [--flag NAME]...
 *
 * Flags. shots.json lists its flags with their default (`flags`, all false). A shot with a `flag`
 * also carries a `fallback`: the fields that replace the shot's own (page, on_screen, voiceover,
 * caption, tag, setup, timeline) when its flag is off. The start and the duration never change,
 * so the cut's timing is the same either way. `--flag NAME` turns a flag on for one run;
 * capture.mjs, voiceover.py and assemble.sh must be given the same flags for one cut (assemble.sh
 * checks the clips' and the voice-over's records).
 *
 * Pending results. A line may hold a `{{pending: ...}}` marker where a result does not exist yet.
 * `check` allows markers in a timing preview and refuses them with `--final` (assemble.sh's final
 * cut), so no marker is ever spoken or burned in.
 *
 * `check` prints every problem and exits 1, or prints "ok" and the shot count, length and flags.
 * `resolve` prints the resolved shot list as JSON.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(here, "..", "..");
export const DEFAULT_SHOTS = path.join(here, "shots.json");
export const FORMAT = "rtt-video-shots/2";
export const PENDING = /\{\{\s*pending:/i;
const OVERRIDABLE = ["page", "on_screen", "voiceover", "caption", "tag", "setup", "timeline"];

/** The shot list in `file` with `on` (a list of flag names) turned on, every flagged shot resolved. */
export function loadShots(file = DEFAULT_SHOTS, on = []) {
  const spec = JSON.parse(readFileSync(file, "utf8"));
  return resolveShots(spec, on);
}

/** `spec` with the flags in `on` turned on; unknown flag names throw. */
export function resolveShots(spec, on = []) {
  const flags = { ...(spec.flags ?? {}) };
  for (const name of on) {
    if (!(name in flags)) throw new Error(`unknown flag ${name}: shots.json lists ${Object.keys(flags).join(", ") || "none"}`);
    flags[name] = true;
  }
  const shots = spec.shots.map((s) => {
    if (!s.flag) return { ...s };
    const { fallback, ...rest } = s;
    if (flags[s.flag]) return { ...rest, variant: "flag" };
    const out = { ...rest, variant: "fallback" };
    for (const k of OVERRIDABLE) if (fallback && k in fallback) out[k] = fallback[k];
    return out;
  });
  return { ...spec, flags, shots };
}

/** The voice-over word count of a line, with each pending marker counted as its stated length. */
export function words(text) {
  return String(text ?? "")
    .replace(/\{\{[^}]*\}\}/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;
}

function brandList(repo) {
  const tokens = (rel, name) => {
    const src = readFileSync(path.join(repo, rel), "utf8");
    const block = src.match(new RegExp(`${name}: Final\\[tuple\\[str, \\.\\.\\.\\]\\] = \\(([\\s\\S]*?)\\n\\)`));
    if (!block) throw new Error(`${name} not found in ${rel}`);
    return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  };
  const list = [...new Set([...tokens("src/rtt/effectors/types.py", "BRAND_TOKENS"), ...tokens("src/rtt/library/brands.py", "LIBRARY_ONLY_TOKENS")])];
  return new RegExp(`(?<![a-z0-9])(${list.sort((a, b) => b.length - a.length).map((x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?![a-z0-9])`, "i");
}

/**
 * The sponsor and platform names, which may be spoken and shown wherever they explain how the
 * product works (founder decision, 27 Oct 2026). Vendor brands of grippers, hands and robots (the
 * repository's lists) stay off screen and out of the voice-over; they belong on the credits only.
 */
export const SPONSOR_TERMS = ["NVIDIA", "Nemotron", "Nebius", "Token Factory"];

/** Other company and service names: they may be spoken, never burned in (the end card's credit excepted). */
export const NOT_ON_SCREEN = /(?<![a-z0-9])(openai|claude|anthropic|github|youtube|mujoco)(?![a-z0-9])/i;

/**
 * Every problem of a resolved shot list: timing back to back from 0:00 and under the limit, every
 * voice-over line word for word in the script document, no vendor brand token (grippers, hands,
 * robots) spoken, burned in or in the credit, no other company name burned in outside the end
 * card's credit (the sponsor terms are allowed), known tags, flagged shots with a fallback, and
 * (final) no pending marker.
 */
export function checkShots(spec, { repo = REPO, final = false, raw = null } = {}) {
  const problems = [];
  if (spec.format !== FORMAT) problems.push(`format is ${spec.format}, expected ${FORMAT}`);
  let t = 0;
  const ids = new Set();
  for (const s of spec.shots) {
    if (ids.has(s.id)) problems.push(`${s.id} appears twice`);
    ids.add(s.id);
    if (Math.abs(s.start - t) > 1e-6) problems.push(`${s.id} starts at ${s.start} s, expected ${t} s`);
    if (!(s.duration > 0)) problems.push(`${s.id} has no duration`);
    if (typeof s.voiceover !== "string" || !s.voiceover.trim()) problems.push(`${s.id} has no voice-over line`);
    if (!Array.isArray(s.setup) || !Array.isArray(s.timeline)) problems.push(`${s.id} needs setup and timeline lists`);
    if (s.tag && !spec.labels[s.tag]) problems.push(`${s.id}: unknown tag ${s.tag}`);
    t += s.duration;
  }
  if (t > spec.max_duration_s) problems.push(`the cut is ${t} s, over ${spec.max_duration_s} s`);
  for (const s of raw?.shots ?? []) {
    if (s.flag && !(s.flag in (raw.flags ?? {}))) problems.push(`${s.id}: flag ${s.flag} is not listed in flags`);
    if (s.flag && !s.fallback) problems.push(`${s.id}: a flagged shot needs a fallback`);
  }

  const scriptFile = path.join(repo, spec.script);
  const norm = (x) => String(x).replace(/\s+/g, " ").trim();
  const doc = existsSync(scriptFile) ? norm(readFileSync(scriptFile, "utf8")) : "";
  if (!doc) problems.push(`no ${spec.script}`);
  for (const s of spec.shots) if (doc && !doc.includes(norm(s.voiceover))) problems.push(`${s.id}: its voice-over is not in ${spec.script}`);

  const brand = brandList(repo);
  const burned = [
    ...spec.shots.flatMap((s) => [s.caption ?? "", s.tag ? spec.labels[s.tag] ?? "" : ""]),
    spec.thumbnail.title, spec.thumbnail.subtitle, spec.end_card.title, ...spec.end_card.lines,
  ];
  for (const x of [...spec.shots.map((s) => s.voiceover), ...burned, spec.end_card.credit]) {
    const hit = String(x).match(brand);
    if (hit) problems.push(`a brand token ("${hit[1]}") in: ${String(x).slice(0, 60)}`);
  }
  for (const x of burned) {
    const hit = String(x).match(NOT_ON_SCREEN);
    if (hit) problems.push(`a company name ("${hit[1]}") burned in outside the end card's credit: ${String(x).slice(0, 60)}`);
  }
  if (final) {
    const all = [...spec.shots.flatMap((s) => [s.voiceover, s.caption ?? ""]), ...burned, ...spec.end_card.links, spec.end_card.credit];
    for (const x of all) if (PENDING.test(String(x))) problems.push(`a pending marker in the final cut: ${String(x).slice(0, 70)}`);
  }
  return { problems, total_s: t };
}

function cli(argv) {
  const [cmd, ...rest] = argv;
  let file = DEFAULT_SHOTS;
  const on = [];
  let final = false;
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === "--shots") file = rest[++i];
    else if (rest[i] === "--flag") on.push(rest[++i]);
    else if (rest[i] === "--final") final = true;
    else throw new Error(`unknown argument ${rest[i]}`);
  }
  const raw = JSON.parse(readFileSync(file, "utf8"));
  const spec = resolveShots(raw, on);
  if (cmd === "resolve") {
    process.stdout.write(JSON.stringify(spec, null, 1) + "\n");
    return 0;
  }
  if (cmd !== "check") throw new Error("usage: shots.mjs check|resolve [--shots FILE] [--flag NAME]... [--final]");
  const { problems, total_s } = checkShots(spec, { final, raw });
  if (problems.length) {
    console.error("shots: the shot list does not pass:\n  " + problems.join("\n  "));
    return 1;
  }
  const n = spec.shots.reduce((a, s) => a + words(s.voiceover), 0);
  const flagsOn = Object.entries(spec.flags).filter(([, v]) => v).map(([k]) => k);
  console.log(`ok: ${spec.shots.length} shots, ${total_s} s, ${n} spoken words outside markers, flags on: ${flagsOn.join(", ") || "none"}${final ? ", final" : ""}`);
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    process.exit(cli(process.argv.slice(2)));
  } catch (err) {
    console.error(`shots: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(2);
  }
}
