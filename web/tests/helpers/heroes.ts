/**
 * Six v2 heroes for tests, built from the three exported v1 pages (h1..h3), kept as fixtures in
 * tests/fixtures/studio_v1 (its data/ and, for the GLBs their lanes draw, its site/) when the real
 * v2 data replaced them under public/data. Each page is re-labelled with a v2 case id and its own paths, two of
 * them become honest no-recommendation pages (the recommended setup becomes the closest one,
 * stopping at the last step), and each carries what the v2 export adds, so every v2 feature runs
 * through the strict parser:
 *
 * - the recommender's baseline (the best single tool) and the pick against it;
 * - Studio-only tools (vacuum area frames, the nut runner, the tamper pad) as the exporter writes
 *   them (tests/fixtures/studio_tools.json, made by tests/test_web_studio_v2_export.py);
 * - the new failure modes and caveats: out of reach, unreliable physics, a lid that fell shut,
 *   a part held above its support, a button too stiff to press.
 *
 * These are test fixtures: the physics behind each page is the v1 hero it is built from, so a
 * page's steps are not its title's job. writeHeroFixtures puts them on disk as a complete data
 * folder (pages, index, replays, build packs, sessions) for the build-time loaders and previews.
 */

import { copyFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

import { type HeroEntry, type HeroPage, parseStudioHero } from "@/components/studio/schema";

const FIXTURES = path.join(__dirname, "..", "fixtures");
/** The published v1 Studio data (pages, index, replays, packs, sessions) and the site root of its GLBs. */
export const V1_FIXTURE = { data: path.join(FIXTURES, "studio_v1", "data"), site: path.join(FIXTURES, "studio_v1", "site") };
const DATA = path.join(V1_FIXTURE.data, "studio");

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;

/** What the exporter writes for the four Studio-only presets (scripts/export_studio_web.py studio_tool). */
export const STUDIO_TOOLS: { effector_id: string; display_name: string; family: string; model: string; fidelity: string; studio_only: true; family_title: string; rating: string; caveat: string }[] =
  JSON.parse(readFileSync(path.join(FIXTURES, "studio_tools.json"), "utf8"));

/** The simulator's own wording of the caveats the fixtures carry (pinned by tests/test_web_labels_contract.py). */
export const LID_FALL_CAVEAT = "The lid fell shut from 21 deg and met its stop at about 320 deg/s: the tool let go of it above its stop";
export const SET_DOWN_CAVEAT = "The part stopped 41 mm above the bin floor (it rests on the bin's walls); it was not let go, since releasing it would drop it";

/** v2 hero id, the v1 page it is built from, its title, and how nothing completes it (null: a recommendation). */
export const V2_HEROES: readonly { hero: string; from: string; title: string; setting: string; noWinner: null | "slip" | "out_of_reach" }[] = [
  { hero: "h1_tote", from: "h1", title: "Returns tote", setting: "warehouse", noWinner: null },
  { hero: "h4_cnc", from: "h3", title: "Machine tending", setting: "manufacturing", noWinner: null },
  { hero: "h5_solar", from: "h1", title: "Solar panel installation", setting: "construction", noWinner: null },
  { hero: "h6_tile", from: "h2", title: "Floor tile setting", setting: "construction", noWinner: "slip" },
  { hero: "h2_coupling", from: "h2", title: "Pipe coupling", setting: "construction", noWinner: null },
  { hero: "h7_hotel", from: "h3", title: "Hotel delivery", setting: "hospitality", noWinner: "out_of_reach" },
];

function relabel(value: Json, from: string, to: string): Json {
  if (typeof value === "string") {
    return value
      .replace(`studio/replays/${from}/`, `studio/replays/${to}/`)
      .replace(`studio/${from}/build_pack.zip`, `studio/${to}/build_pack.zip`)
      .replace(`studio/sessions/${from}.json`, `studio/sessions/${to}.json`);
  }
  if (Array.isArray(value)) return value.map((v) => relabel(v, from, to));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, relabel(v, from, to)]));
  return value;
}

/** Turn a recommended page into a no-recommendation one: its recommendation stops at the last step. */
function toNoWinner(raw: Json, mode: "slip" | "out_of_reach"): Json {
  const rec = raw.recommendation;
  const last = raw.plan.steps[raw.plan.steps.length - 1].id;
  const row = raw.finalists.find((f: Json) => f.setup_id === rec.setup_id);
  const n = row.n_runs;
  const closest = {
    ...row,
    roles: ["closest"],
    completed_runs: 0,
    completion_rate: 0,
    cycle_time_s: null,
    stops_at: last,
    step_success: { ...row.step_success, [last]: 0 },
    weakest_step: { step_id: last, success_rate: 0 },
    failure_modes: { [mode]: n },
    stops_at_detail: { [last]: { runs: n, failure_modes: { [mode]: n } } },
  };
  raw.outcome = "no_recommendation";
  raw.headline = `No setup completes the task in ${n} runs.`;
  raw.recommendation = null;
  raw.finalists = raw.finalists.map((f: Json) => (f.setup_id === rec.setup_id ? closest : f));
  raw.no_winner = { reason: "every finalist stops at the last step", closest: [closest], blocker: null };
  for (const lane of raw.lanes) {
    if (lane.id !== "recommended") continue;
    lane.id = "closest";
    lane.title = "Closest setup";
    lane.completed = false;
    lane.cycle_time_s = null;
    for (const s of lane.steps) {
      if (s.step_id !== last) continue;
      s.success = false;
      s.failure_mode = mode;
      // where the palm would have had to go (rtt.studio.primitives._common.out_of_reach_verdict)
      if (mode === "out_of_reach") s.quantities = { ...s.quantities, reach_x_mm: 364, reach_y_mm: -12, palm_limit_mm: 350 };
    }
  }
  if (raw.build_pack) raw.build_pack.for = "closest";
  return raw;
}

/** The recommender's baseline: the best single tool beside the pick, or, with no pick, why none. */
function withBaseline(raw: Json): Json {
  const base = raw.best_single;
  if (!raw.recommendation) {
    raw.baseline = { setup_id: null, basis: "no single tool is eligible: no single tool completed the task once" };
    return raw;
  }
  const rec = raw.recommendation;
  raw.baseline = { setup_id: base.setup_id, basis: null };
  raw.finalists = raw.finalists.map((f: Json) => (f.setup_id === base.setup_id ? { ...f, roles: [...f.roles, "baseline"] } : f));
  raw.best_single = { ...base, roles: [...base.roles, "baseline"] };
  const added = rec.kind === "tool_change" ? rec.tools.length : rec.tools.length - 1;
  const gain = rec.completion_rate - base.completion_rate;
  rec.versus_baseline = {
    rule: "R01",
    same_setup: false,
    completion_delta: Number(gain.toFixed(4)),
    cycle_delta_s: base.cycle_time_s === null || rec.cycle_time_s === null ? null : Number((rec.cycle_time_s - base.cycle_time_s).toFixed(3)),
    added_pieces: added,
    needed_gain: Number((0.05 * added).toFixed(4)),
    basis: base.weakest_step?.success_rate < 0.5 ? "step_floor" : "margin",
    note: null,
  };
  return raw;
}

/** Studio-only tools among the candidates, ruled out on this job before physics. */
function withStudioTools(raw: Json, models: string[]): Json {
  const tools = STUDIO_TOOLS.filter((t) => models.includes(t.model)).map((t) => ({
    effector_id: t.effector_id,
    display_name: t.display_name,
    family: t.family,
    family_title: t.family_title,
    source: "preset",
    off_the_shelf: true,
    mechanism: t.family === "process_tool" ? "process" : "suction",
    fidelity: t.fidelity,
    built_for: null,
    status: "pruned",
    pruned_reason: "its capability tags do not include this job's steps (a test fixture)",
    pruned_steps: [],
    studio_only: true,
    rating: t.rating,
    caveat: t.caveat,
  }));
  raw.screen = { ...raw.screen, n_candidates: raw.screen.n_candidates + tools.length, candidates: [...raw.screen.candidates, ...tools] };
  return raw;
}

/** Per hero, what the v2 export adds beyond the baseline. */
function withV2(raw: Json, hero: string): Json {
  switch (hero) {
    case "h1_tote": {
      // the lid fell shut when the tool let go of it: a caveat on the closing step, and its measurements
      const close = raw.recommendation.steps.find((s: Json) => s.step_id === "close_lid");
      close.caveats = [LID_FALL_CAVEAT, ...close.caveats];
      const lane = raw.lanes.find((l: Json) => l.id === "recommended");
      const ls = lane.steps.find((s: Json) => s.step_id === "close_lid");
      ls.quantities = { ...ls.quantities, lid_release_deg: 21.3, lid_stop_dps: 317.4 };
      return raw;
    }
    case "h4_cnc": {
      // the best single tool stops at a button too stiff for the step's limit
      const stop = raw.best_single.stops_at;
      const n = raw.best_single.stops_at_detail[stop].runs;
      for (const row of [raw.best_single, raw.finalists.find((f: Json) => f.setup_id === raw.best_single.setup_id)]) {
        if (!row) continue;
        row.failure_modes = { criterion_not_met: n };
        row.stops_at_detail = { [stop]: { runs: n, failure_modes: { criterion_not_met: n } } };
      }
      const ls = raw.lanes.find((l: Json) => l.id === "best_single").steps.find((s: Json) => s.step_id === stop);
      ls.failure_mode = "criterion_not_met";
      ls.quantities = { ...ls.quantities, press_cap_n: 10, actuation_force_n: 24 };
      return raw;
    }
    case "h5_solar": {
      const pick = raw.recommendation.steps.find((s: Json) => s.step_id === "pick_pack");
      pick.caveats = [SET_DOWN_CAVEAT, ...pick.caveats];
      return withStudioTools(raw, ["vacuum-frame-2x3-600", "nut-runner-socket-13"]);
    }
    case "h6_tile":
      return withStudioTools(raw, ["tamper-pad-200", "vacuum-frame-2x2-300"]);
    case "h2_coupling": {
      // some screen runs of a failing tool were unreliable physics, not slips
      for (const row of raw.matrix.rows) {
        const cell = Object.values(row.cells).find((c: Json) => c.state === "run" && (c.failure_modes?.slip ?? 0) >= 2) as Json;
        if (!cell) continue;
        cell.failure_modes = { ...cell.failure_modes, slip: cell.failure_modes.slip - 1, physics_unreliable: 1 };
        break;
      }
      return withStudioTools(raw, ["nut-runner-socket-13"]);
    }
    default:
      return raw;
  }
}

/** The raw page JSON of one v2 hero, as the export would write it. */
export function rawHero(h: (typeof V2_HEROES)[number]): Json {
  let raw: Json = relabel(JSON.parse(readFileSync(path.join(DATA, `${h.from}.json`), "utf8")), h.from, h.hero);
  raw = { ...raw, hero: h.hero, case_id: h.hero, title: h.title, setting: h.setting };
  if (h.noWinner) raw = toNoWinner(raw, h.noWinner);
  return withV2(withBaseline(raw), h.hero);
}

/** An index entry in the exporter's shape (scripts/export_studio_web.py index_entry). */
function indexEntry(page: Json): Json {
  const strip = (row: Json) =>
    row
      ? {
          setup_id: row.setup_id,
          kind: row.kind,
          kind_title: row.kind_title,
          label: row.label,
          tools: row.tools.map((t: Json) => (typeof t === "string" ? t : t.display_name)),
          completion_rate: row.completion_rate,
          completed_runs: row.completed_runs,
          n_runs: row.n_runs,
          fidelity: row.fidelity,
          stops_at: row.stops_at,
          cycle_time_s: row.cycle_time_s,
        }
      : null;
  return {
    hero: page.hero,
    case_id: page.case_id,
    title: page.title,
    setting: page.setting,
    use_case: page.use_case,
    page: `studio/${page.hero}.json`,
    task_id: page.plan.task_id,
    n_steps: page.plan.steps.length,
    n_candidates: page.screen.n_candidates,
    outcome: page.outcome,
    headline: page.headline,
    recommended: strip(page.recommendation),
    best_single: strip(page.best_single),
    closest: page.no_winner ? strip(page.no_winner.closest[0]) : null,
  };
}

/** The six pages, parsed, and the index entries a v2 data index would list for them. */
export function sixHeroes(): { pages: HeroPage[]; entries: HeroEntry[] } {
  const pages = V2_HEROES.map((h) => parseStudioHero(rawHero(h), `studio/${h.hero}.json`));
  const entries: HeroEntry[] = pages.map((p) => ({
    hero: p.hero,
    case_id: p.case_id,
    title: p.title,
    setting: p.setting,
    use_case: p.use_case,
    page: `studio/${p.hero}.json`,
    task_id: p.plan.task_id,
    n_steps: p.plan.steps.length,
    n_candidates: p.screen.n_candidates,
    outcome: p.outcome,
    headline: p.headline,
    recommended: p.recommendation,
    best_single: p.best_single,
    closest: p.no_winner?.closest[0] ?? null,
  }));
  return { pages, entries };
}

/**
 * Write the v2 heroes as a complete Studio data folder under `dataDir` (its `studio/`): each page,
 * the index, and copies of the v1 hero's replays (headers re-labelled to the hero, so each lane's
 * served bytes are re-counted), build pack and recorded session. `omit` leaves heroes out, as an
 * export that cut them would (the cut order is the hotel, then the tile).
 */
export function writeHeroFixtures(dataDir: string, opts: { omit?: string[] } = {}): string[] {
  const out = path.join(dataDir, "studio");
  const real: Json = JSON.parse(readFileSync(path.join(DATA, "index.json"), "utf8"));
  const heroes = V2_HEROES.filter((h) => !(opts.omit ?? []).includes(h.hero));
  const entries: Json[] = [];
  for (const h of heroes) {
    const page = rawHero(h);
    for (const lane of page.lanes) {
      const stem = lane.replay.split("/").pop();
      const src = path.join(DATA, "replays", h.from);
      const dst = path.join(out, "replays", h.hero);
      mkdirSync(dst, { recursive: true });
      const header = JSON.parse(readFileSync(path.join(src, `${stem}.json`), "utf8"));
      const before = statSync(path.join(src, `${stem}.json`)).size;
      header.meta.part_id = h.hero;
      const text = JSON.stringify(header);
      writeFileSync(path.join(dst, `${stem}.json`), text);
      copyFileSync(path.join(src, header.bin), path.join(dst, header.bin));
      if (header.overlays) copyFileSync(path.join(src, header.overlays.bin), path.join(dst, header.overlays.bin));
      lane.bytes += Buffer.byteLength(text) - before;
    }
    if (page.build_pack) {
      mkdirSync(path.join(out, h.hero), { recursive: true });
      copyFileSync(path.join(DATA, h.from, "build_pack.zip"), path.join(out, h.hero, "build_pack.zip"));
    }
    if (page.session) {
      mkdirSync(path.join(out, "sessions"), { recursive: true });
      copyFileSync(path.join(DATA, "sessions", `${h.from}.json`), path.join(out, "sessions", `${h.hero}.json`));
    }
    writeFileSync(path.join(out, `${h.hero}.json`), `${JSON.stringify(page, null, 1)}\n`);
    entries.push(indexEntry(page));
  }
  const index = { schema: real.schema, version: real.version, label: real.label, contract: real.contract, code_version: real.code_version, heroes: entries };
  writeFileSync(path.join(out, "index.json"), `${JSON.stringify(index, null, 1)}\n`);
  return heroes.map((h) => h.hero);
}
