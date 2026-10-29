"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  filterCards,
  filtersFromQuery,
  filtersToQuery,
  isFiltered,
  type LibraryCard,
  type LibraryFilters,
  mountText,
  NO_FILTERS,
  SIM_STATUS_LABELS,
  type SimStatus,
} from "@/lib/library";

import { FamilyIcon } from "@/components/story/FamilyIcon";

import { CapabilityTag, EntrantBadge, SimStatusBadge } from "./Badges";

/** The family drawing that fits a library item's kind. */
function kindFamily(kind: string): string {
  if (kind.includes("suction")) return "Suction cup";
  if (kind.includes("hand")) return "Multi-finger hand";
  if (kind.includes("finger") || kind.includes("jaw")) return "Parallel jaw";
  return "Industrial gripper";
}

const FIELD = "rounded-md border border-rule-strong bg-surface px-2 py-1.5 text-ink";

function Card({ card, notes }: { card: LibraryCard; notes: Record<string, string> }) {
  return (
    <li className="min-w-0">
      <article
        className="panel relative flex h-full flex-col p-4 transition-colors hover:border-accent/60"
        data-library-card={card.key}
        aria-label={card.display_name}
      >
        <p className="flex items-center gap-2 text-sm text-ink-3"><FamilyIcon family={kindFamily(card.kind)} className="size-5 text-accent" />{card.kind_title}</p>
        <h3 className="heading mt-1 text-lg text-ink">
          <Link
            href={`/library/${card.key}/`}
            className="no-underline after:absolute after:inset-0 after:content-[''] hover:underline focus-visible:underline"
          >
            {card.display_name}
          </Link>
        </h3>
        {card.capabilities.length > 0 ? (
          <p className="mt-2 flex flex-wrap gap-1" aria-label="Capabilities">
            {card.capabilities.map((t) => (
              <CapabilityTag key={t} tag={t} note={notes[t]} />
            ))}
          </p>
        ) : (
          <p className="mt-2 text-xs text-ink-3">No grasp capability: it carries other items.</p>
        )}
        <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-sm">
          <dt className="text-ink-3">Licence</dt>
          <dd className="num break-words text-ink">{card.licences.join(", ")}</dd>
          <dt className="text-ink-3">Simulation</dt>
          <dd>
            <SimStatusBadge status={card.sim_status} />
          </dd>
          <dt className="text-ink-3">Mount</dt>
          <dd className={card.iso_pattern ? "num text-ink" : "text-ink-2"}>{mountText(card.iso_pattern)}</dd>
        </dl>
        <p className="mt-auto pt-3">
          <EntrantBadge entrant={card.entrant} />
        </p>
      </article>
    </li>
  );
}

/**
 * The catalog: text search, filters (capability, licence, mount, sim status, entrants) and a
 * card per item. The filter state lives in the page's URL query, so a filtered view can be shared.
 */
export default function LibraryCatalog({
  cards,
  capabilities,
  licences,
  simStatuses,
}: {
  cards: LibraryCard[];
  capabilities: { tag: string; note: string }[];
  licences: string[];
  simStatuses: SimStatus[];
}) {
  const [filters, setFilters] = useState<LibraryFilters>(NO_FILTERS);
  const [ready, setReady] = useState(false);
  const notes = useMemo(() => Object.fromEntries(capabilities.map((c) => [c.tag, c.note])), [capabilities]);
  const known = useMemo(() => ({ capabilities: capabilities.map((c) => c.tag), licences }), [capabilities, licences]);

  useEffect(() => {
    setFilters(filtersFromQuery(window.location.search, known));
    setReady(true);
  }, [known]);

  useEffect(() => {
    if (!ready) return;
    try {
      const url = `${window.location.pathname}${filtersToQuery(filters)}${window.location.hash}`;
      window.history.replaceState(window.history.state, "", url);
    } catch {
      // A sandboxed frame may refuse history changes; the filters still work.
    }
  }, [filters, ready]);

  const shown = useMemo(() => filterCards(cards, filters), [cards, filters]);
  const set = (patch: Partial<LibraryFilters>) => setFilters((f) => ({ ...f, ...patch }));
  const toggle = (tag: string) =>
    setFilters((f) => ({
      ...f,
      capabilities: f.capabilities.includes(tag) ? f.capabilities.filter((t) => t !== tag) : [...f.capabilities, tag],
    }));

  return (
    <div className="mt-4">
      <form
        role="search"
        aria-label="Search the library"
        onSubmit={(e) => e.preventDefault()}
        className="panel grid gap-4 p-4"
      >
        <label className="grid gap-1 text-sm">
          <span className="text-xs text-ink-3">Search</span>
          <input
            type="search"
            value={filters.query}
            onChange={(e) => set({ query: e.target.value })}
            placeholder="Name, summary or kind"
            autoComplete="off"
            spellCheck={false}
            className={`${FIELD} w-full min-w-0`}
            data-library-search
          />
        </label>

        <fieldset className="min-w-0">
          <legend className="text-xs text-ink-3">Capability (an item matches when it has any selected tag)</legend>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {capabilities.map((c) => {
              const on = filters.capabilities.includes(c.tag);
              return (
                <button
                  key={c.tag}
                  type="button"
                  aria-pressed={on}
                  title={c.note}
                  onClick={() => toggle(c.tag)}
                  className={`rounded-full border px-2.5 py-0.5 text-sm ${on ? "border-accent bg-accent text-accent-ink" : "border-rule-strong text-ink-2 hover:border-ink hover:text-ink"}`}
                >
                  {c.tag}
                </button>
              );
            })}
          </div>
        </fieldset>

        <div className="grid gap-3 sm:grid-cols-3">
          <label className="grid min-w-0 gap-1 text-sm">
            <span className="text-xs text-ink-3">Licence</span>
            <select value={filters.licence} onChange={(e) => set({ licence: e.target.value })} className={`${FIELD} min-w-0`}>
              <option value="">Any licence</option>
              {licences.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </label>
          <label className="grid min-w-0 gap-1 text-sm">
            <span className="text-xs text-ink-3">Mount</span>
            <select
              value={filters.mount}
              onChange={(e) => set({ mount: e.target.value as LibraryFilters["mount"] })}
              className={`${FIELD} min-w-0`}
            >
              <option value="any">Any mount</option>
              <option value="iso">ISO 9409-1 pattern</option>
              <option value="own">No ISO pattern (own interface)</option>
            </select>
          </label>
          <label className="grid min-w-0 gap-1 text-sm">
            <span className="text-xs text-ink-3">Simulation</span>
            <select
              value={filters.sim}
              onChange={(e) => set({ sim: e.target.value as LibraryFilters["sim"] })}
              className={`${FIELD} min-w-0`}
            >
              <option value="any">Any status</option>
              {simStatuses.map((s) => (
                <option key={s} value={s}>
                  {SIM_STATUS_LABELS[s]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={filters.entrantOnly}
            onChange={(e) => set({ entrantOnly: e.target.checked })}
            className="accent-[var(--accent)]"
          />
          <span className="text-ink-2">Entrants only (items that take part in the effector tournament)</span>
        </label>
      </form>

      <p className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-ink-2" aria-live="polite" data-library-count={shown.length}>
        <span>
          Showing <span className="num text-ink">{shown.length}</span> of <span className="num">{cards.length}</span> items.
        </span>
        {isFiltered(filters) ? (
          <button type="button" onClick={() => setFilters(NO_FILTERS)} className="link text-sm">
            Clear filters
          </button>
        ) : null}
      </p>

      {shown.length > 0 ? (
        <ul className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map((c) => (
            <Card key={c.key} card={c} notes={notes} />
          ))}
        </ul>
      ) : (
        <p className="mt-4 rounded-lg border border-dashed border-rule-strong p-4 text-ink-2">No item matches these filters.</p>
      )}
    </div>
  );
}
