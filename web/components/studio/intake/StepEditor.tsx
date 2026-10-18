"use client";

import { type ReactNode, useEffect, useId, useState } from "react";

import {
  addObject,
  addStep,
  type Catalog,
  CONSTRAINT_KEYS,
  type Criterion,
  deleteObject,
  deleteStep,
  handlesFor,
  measurable,
  moveStep,
  num,
  type ParamDesc,
  type ParamValue,
  type PlanArgs,
  type PlanObject,
  type PlanStep,
  primitive,
  problemTarget,
  quantity,
  stepFidelity,
  template,
  type Tier,
  type Verdict,
} from "./taskspec";

const TIER_STYLE: Record<Tier, string> = {
  rigid: "text-pass",
  approximation: "text-pending",
  not_simulated: "text-fail",
};

const TIER_TEXT: Record<Tier, string> = {
  rigid: "rigid",
  approximation: "approximation",
  not_simulated: "not simulated",
};

const CONSTRAINT_TEXT: Record<(typeof CONSTRAINT_KEYS)[number], string> = {
  max_force_n: "Max force (N)",
  max_torque_nm: "Max torque (N m)",
  precision_mm: "Precision (mm)",
  precision_deg: "Precision (deg)",
};

const INPUT =
  "w-full min-w-0 border border-rule-strong bg-paper px-2 py-1 text-sm text-ink aria-[invalid=true]:border-fail aria-[invalid=true]:text-fail";

/** "open_torque_nm" -> "open torque": the unit suffix goes, the unit is printed beside it. */
export function humanName(name: string): string {
  return name.replace(/_(mm_s|deg_s|mm|deg|nms|nm|n|g|s)$/, "").replace(/_/g, " ");
}

function stepOf(p: ParamDesc): number | "any" {
  if (p.kind === "int") return 1;
  const span = (p.high ?? 1) - (p.low ?? 0);
  return span <= 5 ? 0.01 : span <= 60 ? 0.1 : 1;
}

function outOfBounds(p: ParamDesc, v: ParamValue): boolean {
  if (p.kind !== "float" && p.kind !== "int") return false;
  return typeof v !== "number" || !Number.isFinite(v) || v < (p.low as number) || v > (p.high as number);
}

function Field({ label, children, wide }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <label className={`grid min-w-0 gap-0.5 ${wide ? "col-span-full" : ""}`}>
      <span className="truncate text-xs text-ink-3">{label}</span>
      {children}
    </label>
  );
}

/** A number box: empty or unreadable text becomes NaN (or null when `optional`), which the validator names. */
function NumberBox({
  value,
  onChange,
  optional,
  min,
  max,
  step,
  invalid,
  label,
}: {
  value: number | null;
  onChange: (v: number | null) => void;
  optional?: boolean;
  min?: number;
  max?: number;
  step?: number | "any";
  invalid?: boolean;
  label: string;
}) {
  return (
    <input
      type="number"
      inputMode="decimal"
      aria-label={label}
      aria-invalid={invalid || undefined}
      className={INPUT}
      value={value === null || !Number.isFinite(value) ? "" : value}
      min={min}
      max={max}
      step={step ?? "any"}
      placeholder={optional ? "none" : undefined}
      onChange={(e) => {
        const raw = e.target.value.trim();
        if (raw === "") onChange(optional ? null : Number.NaN);
        else onChange(e.target.valueAsNumber);
      }}
    />
  );
}

function ParamField({ spec, value, onChange, prefix }: { spec: ParamDesc; value: ParamValue; onChange: (v: ParamValue) => void; prefix: string }) {
  const unit = spec.unit && spec.unit !== "count" ? ` (${spec.unit})` : "";
  const title = `${spec.description}${spec.low !== null ? `; ${spec.low} to ${spec.high}${unit}` : ""}`;
  if (spec.kind === "bool") {
    return (
      <label className="flex min-w-0 items-center gap-2 self-end py-1 text-sm" title={spec.description}>
        <input type="checkbox" checked={value === true} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
        <span className="truncate text-ink-2">{humanName(spec.name)}</span>
      </label>
    );
  }
  if (spec.kind === "choice") {
    return (
      <Field label={humanName(spec.name)}>
        <select aria-label={`${prefix} ${spec.name}`} title={spec.description} className={INPUT} value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {spec.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </Field>
    );
  }
  return (
    <Field label={`${humanName(spec.name)}${unit}`}>
      <span title={title}>
        <NumberBox
          label={`${prefix} ${spec.name}`}
          value={typeof value === "number" ? value : Number.NaN}
          onChange={(v) => onChange(v ?? Number.NaN)}
          min={spec.low ?? undefined}
          max={spec.high ?? undefined}
          step={stepOf(spec)}
          invalid={outOfBounds(spec, value)}
        />
      </span>
    </Field>
  );
}

function Problems({ items, tone = "fail" }: { items: string[]; tone?: "fail" | "pending" }) {
  if (!items.length) return null;
  return (
    <ul className={`mt-3 grid gap-1 border-l-2 pl-3 text-sm ${tone === "fail" ? "border-fail text-fail" : "border-pending text-ink-2"}`}>
      {items.map((p) => (
        <li key={p} className="break-words">
          {p}
        </li>
      ))}
    </ul>
  );
}

function TierChip({ tier, caveats }: { tier: Tier; caveats: string[] }) {
  return (
    <span className={`stamp ${TIER_STYLE[tier]}`} title={caveats.join(" ") || undefined}>
      {TIER_TEXT[tier].toUpperCase()}
    </span>
  );
}

function ObjectCard({
  catalog,
  object,
  index,
  problems,
  warnings,
  onChange,
  onDelete,
}: {
  catalog: Catalog;
  object: PlanObject;
  index: number;
  problems: string[];
  warnings: string[];
  onChange: (o: PlanObject) => void;
  onDelete: () => void;
}) {
  const t = template(catalog, object.kind);
  const lim = catalog.limits.pose_mm;
  const prefix = `Object ${index + 1}`;
  const [open, setOpen] = useState(false);
  const flagged = problems.length > 0;
  useEffect(() => {
    if (flagged) setOpen(true);
  }, [flagged]);
  return (
    <article aria-label={`Object ${index + 1}: ${object.label}`} className={`border bg-paper/80 p-3 ${problems.length ? "border-fail" : "border-rule"}`}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="heading min-w-0 text-base">
          <span className="num mr-2 text-xs text-accent">O{index + 1}</span>
          <span className="num break-all text-ink">{object.name}</span>
          <span className="ml-2 text-sm text-ink-3">{t?.display_name ?? object.kind}</span>
        </h4>
        <span className="flex items-center gap-3">
          {t ? <TierChip tier={t.fidelity} caveats={t.caveat ? [t.caveat] : []} /> : null}
          <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="link text-sm" aria-label={`${open ? "Hide" : "Edit"} object ${index + 1}`}>
            {open ? "Hide" : "Edit"}
          </button>
          <button type="button" onClick={onDelete} className="link text-sm" aria-label={`Delete object ${object.name}`}>
            Delete
          </button>
        </span>
      </header>
      {!open ? (
        <p className="mt-1 text-sm text-ink-2">
          {object.label} · at ({num(object.pose.x_mm)}, {num(object.pose.y_mm)}) mm ·{" "}
          <span className="text-ink-3">
            {(t?.params ?? [])
              .filter((p) => p.kind !== "bool")
              .slice(0, 4)
              .map((p) => `${humanName(p.name)} ${typeof object.params[p.name] === "number" ? num(object.params[p.name] as number) : String(object.params[p.name])}${p.unit && p.unit !== "count" && typeof object.params[p.name] === "number" ? ` ${p.unit}` : ""}`)
              .join(", ")}
          </span>
        </p>
      ) : null}
      <div className={`mt-3 grid-cols-2 gap-2 sm:grid-cols-4 ${open ? "grid" : "hidden"}`}>
        <Field label="Label" wide>
          <input aria-label={`${prefix} label`} className={INPUT} value={object.label} maxLength={80} onChange={(e) => onChange({ ...object, label: e.target.value })} />
        </Field>
        {(["x_mm", "y_mm", "yaw_deg"] as const).map((k) => (
          <Field key={k} label={k === "yaw_deg" ? "yaw (deg)" : `${k[0]} (mm)`}>
            <NumberBox
              label={`${prefix} pose ${k}`}
              value={object.pose[k]}
              onChange={(v) => onChange({ ...object, pose: { ...object.pose, [k]: v ?? Number.NaN } })}
              min={k === "yaw_deg" ? -180 : -lim}
              max={k === "yaw_deg" ? 180 : lim}
              step={k === "yaw_deg" ? 1 : 5}
              invalid={!Number.isFinite(object.pose[k]) || Math.abs(object.pose[k]) > (k === "yaw_deg" ? 180 : lim)}
            />
          </Field>
        ))}
        {(t?.params ?? []).map((p) =>
          p.name === "shape" && object.ref ? null : (
            <ParamField key={p.name} spec={p} prefix={prefix} value={object.params[p.name] as ParamValue} onChange={(v) => onChange({ ...object, params: { ...object.params, [p.name]: v } })} />
          ),
        )}
      </div>
      {object.ref ? (
        <p className="mt-2 text-xs text-ink-3">
          Geometry from <span className="num text-ink-2">{object.ref}</span> (shape mesh).
        </p>
      ) : null}
      <Problems items={problems} />
      <Problems items={warnings} tone="pending" />
    </article>
  );
}

function CriteriaEditor({ catalog, step, kind, onChange, prefix }: { catalog: Catalog; step: PlanStep; kind: string; onChange: (rows: Criterion[]) => void; prefix: string }) {
  const options = measurable(catalog, step.primitive, kind);
  return (
    <div className="col-span-full">
      <span className="text-xs text-ink-3">Success criteria (when the step ends)</span>
      <ul className="mt-1 grid gap-1.5">
        {step.success.map((c, j) => {
          const q = quantity(catalog, c.quantity);
          const set = (patch: Partial<Criterion>) => onChange(step.success.map((x, k) => (k === j ? { ...x, ...patch } : x)));
          return (
            <li key={j} className="grid grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-2">
              <Field label="quantity">
                <select aria-label={`${prefix} criterion ${j + 1} quantity`} className={INPUT} value={c.quantity} onChange={(e) => set({ quantity: e.target.value })}>
                  {[...new Set([c.quantity, ...options])].map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`at least${q ? ` (${q.unit})` : ""}`}>
                <NumberBox label={`${prefix} criterion ${j + 1} at least`} optional value={c.at_least} onChange={(v) => set({ at_least: v })} />
              </Field>
              <Field label="at most">
                <NumberBox label={`${prefix} criterion ${j + 1} at most`} optional value={c.at_most} onChange={(v) => set({ at_most: v })} />
              </Field>
              <button type="button" className="link pb-1 text-sm" aria-label={`Remove criterion ${j + 1}`} onClick={() => onChange(step.success.filter((_, k) => k !== j))}>
                Remove
              </button>
            </li>
          );
        })}
      </ul>
      {step.success.length < 4 ? (
        <button
          type="button"
          className="link mt-1.5 text-sm"
          onClick={() => onChange([...step.success, { quantity: options[0] ?? "force_n", at_least: null, at_most: null }])}
        >
          Add a criterion
        </button>
      ) : null}
    </div>
  );
}

function StepCard({
  catalog,
  plan,
  step,
  index,
  problems,
  onChange,
  onMove,
  onDelete,
}: {
  catalog: Catalog;
  plan: PlanArgs;
  step: PlanStep;
  index: number;
  problems: string[];
  onChange: (s: PlanStep) => void;
  onMove: (delta: -1 | 1) => void;
  onDelete: () => void;
}) {
  const info = primitive(catalog, step.primitive);
  const obj = plan.objects.find((o) => o.name === step.object);
  const kind = obj?.kind ?? info?.objects[0] ?? "part";
  const handles = handlesFor(catalog, step.primitive, kind);
  const candidates = plan.objects.filter((o) => info?.objects.includes(o.kind));
  const bins = plan.objects.filter((o) => info?.targets.includes(o.kind));
  const earlier = plan.steps.slice(0, index);
  const fid = stepFidelity(catalog, plan, step);
  const prefix = `Step ${index + 1}`;
  const [limitsOpen, setLimitsOpen] = useState(() => CONSTRAINT_KEYS.some((k) => step.constraints[k] !== null));
  return (
    <article aria-label={`Step ${index + 1}: ${step.description}`} className={`border bg-paper/80 p-3 ${problems.length ? "border-fail" : "border-rule"}`}>
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="heading flex min-w-0 flex-wrap items-baseline gap-2 text-base">
          <span className="num text-xs text-accent">S{index + 1}</span>
          <span className="stamp text-accent">{step.primitive.replace("_", " ").toUpperCase()}</span>
          <span className="num break-all text-sm text-ink-2">{step.id}</span>
        </h4>
        <span className="flex flex-wrap items-center gap-3 text-sm">
          <TierChip tier={fid.tier} caveats={fid.caveats} />
          <button type="button" className="link disabled:opacity-40" disabled={index === 0} onClick={() => onMove(-1)} aria-label={`Move step ${index + 1} up`}>
            Up
          </button>
          <button type="button" className="link disabled:opacity-40" disabled={index === plan.steps.length - 1} onClick={() => onMove(1)} aria-label={`Move step ${index + 1} down`}>
            Down
          </button>
          <button type="button" className="link" onClick={onDelete} aria-label={`Delete step ${index + 1}`}>
            Delete
          </button>
        </span>
      </header>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Field label="Description" wide>
          <input aria-label={`${prefix} description`} className={INPUT} value={step.description} onChange={(e) => onChange({ ...step, description: e.target.value })} />
        </Field>
        <Field label="Object">
          <select
            aria-label={`${prefix} object`}
            className={INPUT}
            value={step.object}
            onChange={(e) => {
              const next = plan.objects.find((o) => o.name === e.target.value);
              const hs = next ? handlesFor(catalog, step.primitive, next.kind) : [];
              onChange({ ...step, object: e.target.value, handle: hs.includes(step.handle) ? step.handle : (hs[0] ?? step.handle) });
            }}
          >
            {[...new Set([step.object, ...candidates.map((o) => o.name)])].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Handle">
          <select aria-label={`${prefix} handle`} className={INPUT} value={step.handle} onChange={(e) => onChange({ ...step, handle: e.target.value })} title={catalog.handles[step.handle]}>
            {[...new Set([step.handle, ...handles])].map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </Field>
        {info?.targets.length ? (
          <Field label="Target">
            <select aria-label={`${prefix} target`} className={INPUT} value={step.target ?? ""} onChange={(e) => onChange({ ...step, target: e.target.value || null })}>
              <option value="">none</option>
              {[...new Set([...(step.target ? [step.target] : []), ...bins.map((o) => o.name)])].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </Field>
        ) : null}
        {info?.may_hold ? (
          <label className="flex min-w-0 items-center gap-2 self-end py-1 text-sm">
            <input type="checkbox" checked={step.hold_after} onChange={(e) => onChange({ ...step, hold_after: e.target.checked })} className="accent-[var(--accent)]" />
            <span className="text-ink-2">Ends holding it</span>
          </label>
        ) : null}
        {(info?.params ?? []).map((p) => (
          <ParamField key={p.name} spec={p} prefix={prefix} value={step.params[p.name] as ParamValue} onChange={(v) => onChange({ ...step, params: { ...step.params, [p.name]: v } })} />
        ))}
        <CriteriaEditor catalog={catalog} step={step} kind={kind} prefix={prefix} onChange={(rows) => onChange({ ...step, success: rows })} />
        <details className="col-span-full" open={limitsOpen} onToggle={(e) => setLimitsOpen(e.currentTarget.open)}>
          <summary className="cursor-pointer text-xs text-ink-3">
            Task limits:{" "}
            {CONSTRAINT_KEYS.filter((k) => step.constraints[k] !== null)
              .map((k) => `${CONSTRAINT_TEXT[k].toLowerCase()} ${step.constraints[k]}`)
              .join(", ") || "none"}
          </summary>
          <div className="mt-1 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {CONSTRAINT_KEYS.map((k) => {
              const [lo, hi] = catalog.limits.constraints[k] as [number, number];
              const v = step.constraints[k];
              return (
                <Field key={k} label={CONSTRAINT_TEXT[k]}>
                  <NumberBox
                    label={`${prefix} ${CONSTRAINT_TEXT[k]}`}
                    optional
                    value={v}
                    min={lo}
                    max={hi}
                    invalid={v !== null && (!Number.isFinite(v) || v < lo || v > hi)}
                    onChange={(x) => onChange({ ...step, constraints: { ...step.constraints, [k]: x } })}
                  />
                </Field>
              );
            })}
          </div>
        </details>
        {earlier.length ? (
          <fieldset className="col-span-full flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
            <legend className="text-xs text-ink-3">After</legend>
            {earlier.map((e) => (
              <label key={e.id} className="inline-flex items-center gap-1.5">
                <input
                  type="checkbox"
                  className="accent-[var(--accent)]"
                  checked={step.after.includes(e.id)}
                  onChange={(ev) => onChange({ ...step, after: ev.target.checked ? [...step.after, e.id] : step.after.filter((a) => a !== e.id) })}
                />
                <span className="num text-ink-2">{e.id}</span>
              </label>
            ))}
          </fieldset>
        ) : null}
      </div>
      {fid.tier !== "rigid" && fid.caveats.length ? <p className="mt-2 text-xs text-ink-3">{fid.caveats.join(" ")}</p> : null}
      <Problems items={problems} />
    </article>
  );
}

/** The editable plan: objects and ordered steps, every field inside its bounds, problems on the entry they name. */
export default function StepEditor({
  catalog,
  plan,
  verdict,
  onChange,
}: {
  catalog: Catalog;
  plan: PlanArgs;
  verdict: Verdict;
  onChange: (plan: PlanArgs) => void;
}) {
  const ids = useId();
  const byObject = new Map<number, string[]>();
  const byStep = new Map<number, string[]>();
  const general: string[] = [];
  for (const p of verdict.problems) {
    const t = problemTarget(p, plan);
    const into = t.kind === "object" ? byObject : t.kind === "step" ? byStep : null;
    if (into && t.index < (t.kind === "object" ? plan.objects.length : plan.steps.length)) into.set(t.index, [...(into.get(t.index) ?? []), p]);
    else general.push(p);
  }
  const warnObject = new Map<number, string[]>();
  for (const w of verdict.warnings) {
    const t = problemTarget(w, plan);
    if (t.kind === "object") warnObject.set(t.index, [...(warnObject.get(t.index) ?? []), w]);
  }
  const setObject = (i: number, o: PlanObject) => {
    const old = plan.objects[i]?.name;
    const objects = plan.objects.map((x, k) => (k === i ? o : x));
    const steps = old && old !== o.name ? plan.steps.map((s) => ({ ...s, object: s.object === old ? o.name : s.object, target: s.target === old ? o.name : s.target })) : plan.steps;
    onChange({ ...plan, objects, steps });
  };
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-6">
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <Field label="Title">
          <input aria-label="Plan title" className={INPUT} value={plan.title} onChange={(e) => onChange({ ...plan, title: e.target.value })} />
        </Field>
        <Field label="Description">
          <input aria-label="Plan description" className={INPUT} value={plan.description} onChange={(e) => onChange({ ...plan, description: e.target.value })} />
        </Field>
      </div>
      <Problems items={general} />

      <section aria-labelledby={`${ids}-objects`}>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 id={`${ids}-objects`} className="heading text-lg">
            Objects <span className="num text-sm text-ink-3">{plan.objects.length}</span>
          </h3>
          <Adder
            label="Add an object"
            options={catalog.templates.map((t) => ({ value: t.kind, text: t.display_name }))}
            onAdd={(kind) => onChange(addObject(catalog, plan, kind))}
          />
        </div>
        <div className="mt-3 grid grid-cols-[minmax(0,1fr)] gap-3">
          {plan.objects.map((o, i) => (
            <ObjectCard
              key={`${i}-${o.name}`}
              catalog={catalog}
              object={o}
              index={i}
              problems={byObject.get(i) ?? []}
              warnings={warnObject.get(i) ?? []}
              onChange={(next) => setObject(i, next)}
              onDelete={() => onChange(deleteObject(plan, i))}
            />
          ))}
          {!plan.objects.length ? <p className="border border-dashed border-rule-strong p-4 text-sm text-ink-3">No objects yet.</p> : null}
        </div>
      </section>

      <section aria-labelledby={`${ids}-steps`}>
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 id={`${ids}-steps`} className="heading text-lg">
            Steps <span className="num text-sm text-ink-3">{plan.steps.length}</span>
          </h3>
          <Adder
            label="Add a step"
            options={catalog.primitives.map((p) => ({ value: p.kind, text: p.kind.replace("_", " ") }))}
            onAdd={(prim) => onChange(addStep(catalog, plan, prim))}
          />
        </div>
        <ol className="mt-3 grid grid-cols-[minmax(0,1fr)] gap-3">
          {plan.steps.map((s, i) => (
            <li key={`${i}-${s.id}`}>
              <StepCard
                catalog={catalog}
                plan={plan}
                step={s}
                index={i}
                problems={byStep.get(i) ?? []}
                onChange={(next) => onChange({ ...plan, steps: plan.steps.map((x, k) => (k === i ? next : x)) })}
                onMove={(d) => onChange(moveStep(plan, i, d))}
                onDelete={() => onChange(deleteStep(plan, i))}
              />
            </li>
          ))}
          {!plan.steps.length ? <li className="border border-dashed border-rule-strong p-4 text-sm text-ink-3">No steps yet.</li> : null}
        </ol>
      </section>
    </div>
  );
}

function Adder({ label, options, onAdd }: { label: string; options: { value: string; text: string }[]; onAdd: (value: string) => void }) {
  const id = useId();
  return (
    <form
      className="flex items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        const value = new FormData(e.currentTarget).get("kind");
        if (typeof value === "string" && value) onAdd(value);
      }}
    >
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <select id={id} name="kind" className="border border-rule-strong bg-paper px-2 py-1 text-sm text-ink">
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.text}
          </option>
        ))}
      </select>
      <button type="submit" className="border border-rule-strong px-3 py-1 text-sm text-ink hover:bg-paper-2">
        {label}
      </button>
    </form>
  );
}
