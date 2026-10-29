/**
 * Build-time loader of the Studio's held-out benchmark (use-case benchmark v1, tagged
 * `usecase-v1`): results/usecase_v1/analysis.json at the repository root, written by
 * scripts/studio_usecase_v1.py analyse. Server only (reads the file from disk).
 *
 * The site shows, per hero, the best single tool (arm a) and the Studio's recommendation (arm b)
 * on the 32 held-out conditions, and the totals over the three heroes. An arm is matched to a
 * page's setup by setup id; a mismatch is not shown. A missing or malformed file fails the build.
 * Each v1 hero's title comes from the v1 recording of its plan (the v2 heroes that replaced the v1
 * pages need not list it, and the cabinet is not one of them).
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import { DataError, Node } from "@/lib/validate";

const FILE = "results/usecase_v1/analysis.json";
/** The v1 Task Engineer recording (rtt.studio.engineer.gate.RECORDED_GATE_V1): the v1 heroes' titles. */
const V1_RECORDING = "src/rtt/studio/engineer/recorded/gate-2026-10-18-s4e/manifest.json";
const DEFAULT_REPO = path.join(process.cwd(), "..");

export interface HeldoutArm {
  setup_id: string;
  label: string;
  n: number;
  completed: number;
  ci95: [number, number];
  /** Per step: successes of n, and the failure modes. */
  steps: Record<string, { n: number; successes: number; failure_modes: Record<string, number> }>;
}

export interface HeldoutHero {
  case_id: string;
  /** The v1 hero's title, as its recorded plan names it. */
  title: string;
  n_conditions: number;
  best_single: HeldoutArm;
  recommended: HeldoutArm;
  p: number;
  p_holm: number;
  reject: boolean;
}

export interface Heldout {
  label: string;
  test: string;
  heroes: Record<string, HeldoutHero>;
  totals: { n: number; recommended: number; best_single: number };
  /** The committed file the numbers come from, relative to the repository root. */
  source: string;
}

function readArm(n: Node): HeldoutArm {
  const ci = n.get("ci95").items().map((x) => x.num());
  const steps: HeldoutArm["steps"] = {};
  for (const [id, s] of n.get("steps").entries()) {
    const modes: Record<string, number> = {};
    for (const [m, k] of s.get("failure_modes").entries()) modes[m] = k.num();
    steps[id] = { n: s.get("n").num(), successes: s.get("successes").num(), failure_modes: modes };
  }
  return {
    setup_id: n.get("setup_id").str(),
    label: n.get("label").str(),
    n: n.get("n").num(),
    completed: n.get("completed").num(),
    ci95: [ci[0] ?? 0, ci[1] ?? 0],
    steps,
  };
}

let cached: Heldout | null = null;

/** The held-out benchmark, parsed; throws a DataError when the committed result is missing or malformed. */
export function getHeldout(repo: string = DEFAULT_REPO): Heldout {
  if (cached && repo === DEFAULT_REPO) return cached;
  const file = path.join(repo, FILE);
  if (!existsSync(file)) throw new DataError(FILE, "$", "the committed held-out result is missing");
  const root = new Node(FILE, "$", JSON.parse(readFileSync(file, "utf8")));
  const recording = path.join(repo, V1_RECORDING);
  if (!existsSync(recording)) throw new DataError(V1_RECORDING, "$", "the v1 recording is missing");
  const titles = new Map<string, string>();
  for (const c of new Node(V1_RECORDING, "$", JSON.parse(readFileSync(recording, "utf8"))).get("cases").items()) {
    titles.set(c.get("case_id").str(), c.get("title").str());
  }
  const heroes: Record<string, HeldoutHero> = {};
  const holm = root.get("holm");
  for (const [caseId, h] of root.get("heroes").entries()) {
    const arms = h.get("arms");
    const test = holm.get(caseId);
    const title = titles.get(caseId);
    if (!title) throw new DataError(V1_RECORDING, "$.cases", `no recorded v1 case ${caseId}`);
    heroes[caseId] = {
      case_id: caseId,
      title,
      n_conditions: h.get("n_conditions").num(),
      best_single: readArm(arms.get("a")),
      recommended: readArm(arms.get("b")),
      p: test.get("p").num(),
      p_holm: test.get("p_holm").num(),
      reject: test.get("reject").bool(),
    };
  }
  const list = Object.values(heroes);
  const out: Heldout = {
    label: root.get("label").str(),
    test: root.get("test").str(),
    heroes,
    totals: {
      n: list.reduce((s, h) => s + h.recommended.n, 0),
      recommended: list.reduce((s, h) => s + h.recommended.completed, 0),
      best_single: list.reduce((s, h) => s + h.best_single.completed, 0),
    },
    source: FILE,
  };
  if (repo === DEFAULT_REPO) cached = out;
  return out;
}

/** One hero's held-out arms, when they are the setups its page shows (matched by setup id). */
export function heldoutFor(caseId: string, recommendedId: string | null, bestSingleId: string | null): HeldoutHero | null {
  const h = getHeldout().heroes[caseId];
  if (!h) return null;
  if (recommendedId !== h.recommended.setup_id || bestSingleId !== h.best_single.setup_id) return null;
  return h;
}
