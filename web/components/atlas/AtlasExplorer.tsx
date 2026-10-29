"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  type AtlasFilter,
  type AtlasKind,
  type AtlasSort,
  type AtlasTask,
  FEASIBILITY_LABELS,
  FEASIBILITY_NOTES,
  type Feasibility,
  filterTasks,
  type HeroLinks,
  KIND_LABELS,
  KIND_NOTES,
  NO_FILTER,
  settingLabel,
  simStatus,
  sortTasks,
} from "@/lib/atlas";

import { KIND_ORDER, KindGlyph } from "./parts";

const STATUS_ORDER: readonly ("simulated" | Feasibility)[] = ["simulated", "rigid", "approximation", "out_of_scope"];
const STATUS_LABELS: Record<"simulated" | Feasibility, string> = {
  simulated: "Worked example",
  rigid: "Rigid",
  approximation: "Approximation",
  out_of_scope: "Out of scope",
};
const SORTS: { id: AtlasSort; label: string }[] = [
  { id: "demo", label: "Clearest demos first" },
  { id: "simulated", label: "Worked examples first" },
  { id: "title", label: "A to Z" },
];

function Chip({ active, onClick, children, count }: { active: boolean; onClick: () => void; children: React.ReactNode; count?: number }) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm transition-colors ${
        active ? "border-accent bg-accent-soft text-ink" : "border-rule-strong text-ink-2 hover:border-ink-3 hover:text-ink"
      }`}
    >
      {children}
      {count !== undefined ? <span className="num text-xs text-ink-3">{count}</span> : null}
    </button>
  );
}

function FilterRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-2 sm:grid-cols-[7.5rem_minmax(0,1fr)] sm:items-center">
      <p className="text-xs text-ink-3">{label}</p>
      <div className="flex flex-wrap gap-2" role="group" aria-label={label}>
        {children}
      </div>
    </div>
  );
}

function DemoDots({ score }: { score: number }) {
  return (
    <span className="inline-flex items-center gap-1" title={`Demo value ${score} of 5: how clearly the job reads in a simulation (research estimate)`}>
      <span className="sr-only">Demo value {score} of 5</span>
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`size-1.5 rounded-full ${i <= score ? "bg-ink-2" : "bg-paper-3"}`} aria-hidden="true" />
      ))}
    </span>
  );
}

function StatusBadge({ task, heroes }: { task: AtlasTask; heroes: HeroLinks }) {
  const s = simStatus(task, heroes);
  if (s.kind === "simulated") {
    return (
      <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-accent/70 bg-accent-soft px-2 py-px text-xs text-accent" data-sim="simulated">
        <svg viewBox="0 0 12 12" className="size-2.5" aria-hidden="true">
          <path d="M3 2l7 4-7 4z" fill="currentColor" />
        </svg>
        {s.match === "same" ? "Worked example" : "Related worked example"}
      </span>
    );
  }
  const tone =
    s.feasibility === "rigid"
      ? "border-rule-strong text-ink-2"
      : s.feasibility === "approximation"
        ? "border-pending/70 bg-pending-soft text-pending"
        : "border-dashed border-rule-strong text-ink-3";
  return (
    <span className={`inline-flex shrink-0 items-center rounded-full border px-2 py-px text-xs ${tone}`} title={FEASIBILITY_NOTES[s.feasibility]} data-sim={s.feasibility}>
      {STATUS_LABELS[s.feasibility]}
    </span>
  );
}

/** One researched job: what it is, its steps, the expected setup and why, and where it was found. */
export function TaskCard({ task, heroes, domainTitle }: { task: AtlasTask; heroes: HeroLinks; domainTitle: string }) {
  const s = simStatus(task, heroes);
  const notes = task.subtasks.filter((st) => st.note);
  return (
    <article id={`task-${task.id}`} className="panel flex min-w-0 scroll-mt-24 flex-col p-4 sm:p-5" data-atlas-task={task.id}>
      <header className="flex items-start justify-between gap-3">
        <p className="min-w-0 text-xs text-ink-3">
          {settingLabel(task.setting)}
          {settingLabel(task.setting) !== domainTitle ? <span className="hidden sm:inline"> · {domainTitle}</span> : null}
        </p>
        <StatusBadge task={task} heroes={heroes} />
      </header>
      <h3 className="heading mt-2 text-lg text-ink">{task.title}</h3>

      <div className="mt-3 flex items-start gap-2.5 rounded-lg border border-rule bg-surface-2 px-3 py-2.5">
        <KindGlyph kind={task.setup.kind} className="mt-0.5 size-5 shrink-0 text-accent" />
        <div className="min-w-0">
          <p className="text-[0.6875rem] text-ink-3">Expected setup (research estimate)</p>
          <p className="text-sm font-semibold text-ink" title={KIND_NOTES[task.setup.kind]}>
            {KIND_LABELS[task.setup.kind]}
          </p>
          <ul className="mt-0.5 text-sm text-ink-2">
            {task.setup.tools.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </div>

      <ol className="mt-3 grid gap-1.5 text-sm" aria-label="Steps">
        {task.subtasks.map((st, i) => (
          <li key={`${st.verb}-${i}`} className="flex items-baseline gap-2">
            <span className="num grid size-5 shrink-0 place-items-center rounded border border-rule-strong text-[0.6875rem] text-ink-3">{i + 1}</span>
            <span className="min-w-0">
              <span className="font-semibold text-ink">{st.verb}</span> <span className="text-ink-2">{st.object}</span>
            </span>
          </li>
        ))}
      </ol>

      <p className="mt-3 text-xs text-ink-3">Why we expect this</p>
      <p className="mt-0.5 text-[0.9375rem] text-ink-2">{task.why}</p>

      {notes.length ? (
        <details className="group mt-3 text-sm">
          <summary className="flex list-none items-center gap-1.5 text-ink-3 hover:text-ink [&::-webkit-details-marker]:hidden">
            <svg viewBox="0 0 12 12" className="size-3 transition-transform group-open:rotate-90" aria-hidden="true">
              <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
            </svg>
            Step notes
          </summary>
          <ul className="mt-2 grid gap-1.5 border-l border-rule pl-3 text-ink-2">
            {task.subtasks.map((st, i) =>
              st.note ? (
                <li key={`n-${i}`}>
                  <span className="text-ink-3">{i + 1}. </span>
                  {st.note}
                </li>
              ) : null,
            )}
          </ul>
        </details>
      ) : null}

      <div className="min-h-4 flex-1" aria-hidden="true" />
      <footer className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-t border-rule pt-3 text-xs text-ink-3">
        <DemoDots score={task.demo_score} />
        {task.sources.length ? (
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span>Sources</span>
            {task.sources.map((src, i) => (
              <a key={src.url} href={src.url} rel="noopener noreferrer" target="_blank" className="link num" aria-label={`Source ${i + 1} for ${task.title} (opens in a new tab)`}>
                {i + 1}
              </a>
            ))}
          </span>
        ) : null}
      </footer>
      {s.kind === "simulated" ? (
        <Link href={`/studio/${s.hero}/`} className="btn btn-quiet mt-3 w-full text-sm" data-hero-link={s.hero}>
          {s.match === "same" ? "See the worked example" : `See the related worked example: ${s.title}`}
          <span aria-hidden="true">→</span>
        </Link>
      ) : null}
    </article>
  );
}

/** Search, filter and sort the researched jobs; every control is a real button or input. */
export function AtlasExplorer({ tasks, domains, heroes }: { tasks: AtlasTask[]; domains: { id: string; title: string }[]; heroes: HeroLinks }) {
  const [filter, setFilter] = useState<AtlasFilter>(NO_FILTER);
  const [sort, setSort] = useState<AtlasSort>("demo");

  // A link to #task-<id> opens with that task in view, whatever the filters were.
  useEffect(() => {
    const id = window.location.hash.replace(/^#task-/, "");
    if (id && tasks.some((t) => t.id === id)) document.getElementById(`task-${id}`)?.scrollIntoView({ block: "start" });
  }, [tasks]);

  const shown = useMemo(() => sortTasks(filterTasks(tasks, filter, heroes), sort, heroes), [tasks, filter, sort, heroes]);
  const domainTitle = (id: string) => domains.find((d) => d.id === id)?.title ?? id;
  const count = (f: Partial<AtlasFilter>) => filterTasks(tasks, { ...filter, ...f }, heroes).length;
  const set = (f: Partial<AtlasFilter>) => setFilter((old) => ({ ...old, ...f }));
  const filtered = filter.q !== "" || filter.domain !== null || filter.kind !== null || filter.status !== null;

  return (
    <div data-atlas-explorer>
      <div className="panel grid gap-4 p-4 sm:p-5">
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
          <label className="grid gap-1.5">
            <span className="text-xs text-ink-3">Search jobs, steps and tools</span>
            <input
              type="search"
              value={filter.q}
              onChange={(e) => set({ q: e.target.value })}
              placeholder="for example: vacuum, door, torque"
              className="w-full rounded-md border border-rule-strong bg-paper px-3 py-2 text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none"
            />
          </label>
          <label className="grid gap-1.5">
            <span className="text-xs text-ink-3">Order</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as AtlasSort)}
              className="rounded-md border border-rule-strong bg-paper px-3 py-2 text-ink focus:border-accent focus:outline-none"
            >
              {SORTS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <FilterRow label="Where">
          <Chip active={filter.domain === null} onClick={() => set({ domain: null })} count={count({ domain: null })}>
            All
          </Chip>
          {domains.map((d) => (
            <Chip key={d.id} active={filter.domain === d.id} onClick={() => set({ domain: filter.domain === d.id ? null : d.id })} count={count({ domain: d.id })}>
              {d.title}
            </Chip>
          ))}
        </FilterRow>
        <FilterRow label="Setup">
          <Chip active={filter.kind === null} onClick={() => set({ kind: null })} count={count({ kind: null })}>
            Any
          </Chip>
          {KIND_ORDER.map((k: AtlasKind) => (
            <Chip key={k} active={filter.kind === k} onClick={() => set({ kind: filter.kind === k ? null : k })} count={count({ kind: k })}>
              <KindGlyph kind={k} className="size-4" />
              {KIND_LABELS[k]}
            </Chip>
          ))}
        </FilterRow>
        <FilterRow label="Simulation">
          <Chip active={filter.status === null} onClick={() => set({ status: null })} count={count({ status: null })}>
            Any
          </Chip>
          {STATUS_ORDER.map((st) => (
            <Chip key={st} active={filter.status === st} onClick={() => set({ status: filter.status === st ? null : st })} count={count({ status: st })}>
              {st === "simulated" ? STATUS_LABELS[st] : FEASIBILITY_LABELS[st]}
            </Chip>
          ))}
        </FilterRow>
      </div>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-ink-2" aria-live="polite" data-atlas-count={shown.length}>
          Showing <span className="num text-ink">{shown.length}</span> of <span className="num">{tasks.length}</span> jobs
        </p>
        {filtered ? (
          <button type="button" onClick={() => setFilter(NO_FILTER)} className="text-sm text-accent hover:underline">
            Clear filters
          </button>
        ) : null}
      </div>

      {shown.length ? (
        <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {shown.map((t) => (
            <TaskCard key={t.id} task={t} heroes={heroes} domainTitle={domainTitle(t.domain)} />
          ))}
        </div>
      ) : (
        <p className="panel mt-4 p-6 text-center text-ink-2">
          No job matches these filters.{" "}
          <button type="button" onClick={() => setFilter(NO_FILTER)} className="text-accent hover:underline">
            Clear them
          </button>
        </p>
      )}
    </div>
  );
}
