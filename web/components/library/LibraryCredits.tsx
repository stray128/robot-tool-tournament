import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";

import Link from "next/link";
import type { ReactNode } from "react";

import { getLibraryIndex } from "@/app/library/data";
import { shortCommit } from "@/lib/library";

/**
 * Credits of the effector library, rendered from each item's SOURCE.json under
 * assets/effectors/library/<key>/ (read at build time): the credit line, licences, copyright,
 * pinned upstream and the changes made to its files. The credits page is the only page that
 * names the makers; each entry carries the anchor library-<key> that the item's page links to.
 */

interface Pin {
  repository: string;
  commit: string;
}

export interface LibraryItemCredit {
  key: string;
  display_name: string;
  own: boolean;
  attribution: string;
  licences: { spdx: string; covers: string }[];
  copyright: string;
  upstream: Pin | null;
  extra: Pin[];
  n_files: number;
  /** Distinct changes over the item's files, each with the number of files it applies to. */
  changes: { text: string; files: number }[];
  preview_changes: string[];
}

const str = (v: unknown): string => (typeof v === "string" ? v : v === undefined || v === null ? "" : String(v));
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});

function readCredit(dir: string, raw: unknown): LibraryItemCredit | null {
  const r = obj(raw);
  const key = str(r.key) || path.basename(dir);
  const attribution = str(r.attribution);
  if (!attribution) return null;
  const up = obj(r.upstream);
  const own = str(r.class) === "own" || str(up.commit) === "generated";
  const files = list(r.files).map(obj);
  const counts = new Map<string, number>();
  for (const f of files) for (const c of new Set(list(f.changes).map(str).filter(Boolean))) counts.set(c, (counts.get(c) ?? 0) + 1);
  return {
    key,
    display_name: str(r.display_name) || key,
    own,
    attribution,
    licences: list(r.licence)
      .map(obj)
      .map((l) => ({ spdx: str(l.spdx), covers: str(l.covers) }))
      .filter((l) => l.spdx),
    copyright: str(r.copyright),
    upstream: own || !str(up.repository) ? null : { repository: str(up.repository), commit: str(up.commit) },
    extra: list(up.extra)
      .map(obj)
      .map((e) => ({ repository: str(e.repository), commit: str(e.commit) }))
      .filter((e) => e.repository),
    n_files: files.length,
    changes: [...counts].map(([text, n]) => ({ text, files: n })),
    preview_changes: list(obj(r.preview).changes).map(str).filter(Boolean),
  };
}

/** Every SOURCE.json under assets/effectors/library/ (repository root), one entry per item, by key. */
export function libraryCredits(root: string = path.join(process.cwd(), "..", "assets", "effectors", "library")): LibraryItemCredit[] {
  if (!existsSync(root)) return [];
  const out: LibraryItemCredit[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue;
    const file = path.join(root, entry.name, "SOURCE.json");
    if (!existsSync(file)) continue;
    const credit = readCredit(path.join(root, entry.name), JSON.parse(readFileSync(file, "utf8")));
    if (credit) out.push(credit);
  }
  return out;
}

const URL_IN_TEXT = /(https?:\/\/[^\s,;()]+)/g;

/** A credit line with its URLs as links (a trailing full stop stays text). */
function Linked({ text }: { text: string }) {
  const parts = text.split(URL_IN_TEXT);
  return (
    <>
      {parts.map((p, i) => {
        if (i % 2 === 0) return p;
        const href = p.replace(/[.]+$/, "");
        return (
          <span key={i}>
            <a className="link break-all" href={href} rel="noreferrer">
              {href}
            </a>
            {p.slice(href.length)}
          </span>
        );
      })}
    </>
  );
}

function pinLink(p: Pin): ReactNode {
  return (
    <>
      <a className="link break-all" href={`${p.repository}/tree/${p.commit}`} rel="noreferrer">
        {p.repository}
      </a>{" "}
      at commit <span className="num">{shortCommit(p.commit)}</span>
    </>
  );
}

function CreditEntry({ c, urls }: { c: LibraryItemCredit; urls: Map<string, string | null> }) {
  return (
    <div id={`library-${c.key}`} className="grid scroll-mt-24 gap-1 border-b border-rule pb-3 text-sm" data-library-credit={c.key}>
      <p className="break-words text-ink">
        <Linked text={c.attribution} />
      </p>
      <p className="num text-ink">
        {c.licences.map((l, i) => {
          const href = urls.get(l.spdx) ?? `https://spdx.org/licenses/${l.spdx}.html`;
          return (
            <span key={`${l.spdx}-${i}`}>
              {i > 0 ? ", " : null}
              <a className="link" href={href} rel="noreferrer license">
                {l.spdx}
              </a>
            </span>
          );
        })}
      </p>
      <p>
        Library item{" "}
        <Link className="link" href={`/library/${c.key}/`}>
          {c.display_name}
        </Link>{" "}
        ({c.n_files} {c.n_files === 1 ? "file" : "files"}).{c.copyright ? ` ${c.copyright}.` : ""}
      </p>
      {c.licences.some((l) => l.covers) ? (
        <p>
          Licence covers:{" "}
          {c.licences
            .filter((l) => l.covers)
            .map((l) => `${l.spdx}: ${l.covers}`)
            .join("; ")}
          .
        </p>
      ) : null}
      {c.upstream ? (
        <p className="break-words">
          Upstream: {pinLink(c.upstream)}
          {c.extra.length > 0 ? (
            <>
              ; also pinned:{" "}
              {c.extra.map((e, i) => (
                <span key={e.repository}>
                  {i > 0 ? ", " : null}
                  {pinLink(e)}
                </span>
              ))}
            </>
          ) : null}
          .
        </p>
      ) : (
        <p>Our own work, generated by this project, under Apache-2.0. No upstream.</p>
      )}
      {c.changes.length > 0 ? (
        <div>
          <p>Modified:</p>
          <ul className="ml-4 list-disc">
            {c.changes.map((ch) => (
              <li key={ch.text}>
                {ch.text} <span className="text-ink-3">({ch.files === 1 ? "1 file" : `${ch.files} files`})</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <p>{c.own ? "Generated files, not modified after generation." : "Files mirrored unmodified."}</p>
      )}
      {c.preview_changes.length > 0 ? <p>Web preview: {c.preview_changes.join("; ")}.</p> : null}
    </div>
  );
}

/** The credits page section of the effector library. */
export function LibraryCredits() {
  const credits = libraryCredits();
  if (credits.length === 0) return <p>No effector library items are vendored yet.</p>;
  const urls = new Map<string, string | null>();
  try {
    for (const item of getLibraryIndex()?.items ?? []) for (const l of item.licences) if (l.url) urls.set(l.spdx, l.url);
  } catch {
    // The library pages fail the build on a broken index; here SPDX links are enough.
  }
  const third = credits.filter((c) => !c.own);
  const own = credits.filter((c) => c.own);
  return (
    <>
      <p>
        Each item below is mirrored at a pinned upstream commit, with its licence files, under the licence its maker chose.
        Library pages call items by generic names; this section is where makers are named.
      </p>
      {third.map((c) => (
        <CreditEntry key={c.key} c={c} urls={urls} />
      ))}
      {own.length > 0 ? (
        <>
          <h3 className="heading mt-2 text-base text-ink">Generated adapters (our own work, Apache-2.0)</h3>
          {own.map((c) => (
            <CreditEntry key={c.key} c={c} urls={urls} />
          ))}
        </>
      ) : null}
    </>
  );
}
