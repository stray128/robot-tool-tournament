/**
 * The Studio intake page (track S4, product mode): its plan validator against the Python side
 * (parity.json, written by `python -m rtt.studio.engineer export-web`), task ids, fidelity, the
 * edit operations and the recorded plans the static site plays.
 */

import { describe, expect, it } from "vitest";

import catalogJson from "@/components/studio/intake/catalog.json";
import parityJson from "@/components/studio/intake/parity.json";
import recordedJson from "@/components/studio/intake/recorded.json";
import {
  addObject,
  addStep,
  type Catalog,
  canonicalJson,
  deleteObject,
  deleteStep,
  emptyPlan,
  exportDocument,
  fmtG,
  moveStep,
  num,
  type PlanArgs,
  planArgsOf,
  problemTarget,
  pyFloat,
  pyRepr,
  stepFidelity,
  taskId,
  validatePlan,
} from "@/components/studio/intake/taskspec";

const catalog = catalogJson as unknown as Catalog;

interface ParityCase {
  name: string;
  base: string;
  args: PlanArgs;
  status: string;
  problems: string[];
  warnings: string[];
  task_id: string | null;
  fidelity: Record<string, string>;
  caveats: Record<string, string[]>;
}

const cases = (parityJson as unknown as { cases: ParityCase[] }).cases;
const examples = (recordedJson as unknown as { examples: { case_id: string; hero: boolean; status: string; plan: Record<string, unknown>; uploads: { upload_id: string }[] }[] }).examples;

describe("Python-compatible text", () => {
  it("formats numbers like Python's g, repr and json", () => {
    expect(fmtG(400)).toBe("400");
    expect(fmtG(-300)).toBe("-300");
    expect(fmtG(0.02)).toBe("0.02");
    expect(fmtG(2000)).toBe("2000");
    expect(fmtG(1e-5)).toBe("1e-05");
    expect(fmtG(123456789)).toBe("1.23457e+08");
    expect(fmtG(0.5)).toBe("0.5");
    expect(pyFloat(60)).toBe("60");
    expect(pyFloat(0.2)).toBe("0.2");
    expect(pyFloat(5e-5)).toBe("5e-05");
    expect(num(110)).toBe("110");
    expect(num(0.125, 1)).toBe("0.1");
    expect(num(5.5)).toBe("5.5");
    expect(pyRepr("tote")).toBe("'tote'");
    expect(pyRepr(["a", "b"])).toBe("['a', 'b']");
    expect(pyRepr("it's")).toBe('"it\'s"');
    expect(canonicalJson({ b: 1, a: [0.5, null, "x"] })).toBe('{"a":[0.5,null,"x"],"b":1}');
  });
});

describe("validator parity with the Python judge", () => {
  it("has every case", () => {
    expect(cases.length).toBeGreaterThanOrEqual(40);
  });

  for (const c of cases) {
    it(`${c.name}: same status, problems and warnings`, () => {
      const verdict = validatePlan(catalog, c.args);
      expect(verdict.problems).toEqual(c.problems);
      expect(verdict.warnings).toEqual(c.warnings);
      expect(verdict.status).toBe(c.status);
    });
  }

  for (const c of cases.filter((x) => x.task_id)) {
    it(`${c.name}: same task id, fidelity and caveats`, async () => {
      const verdict = validatePlan(catalog, c.args);
      expect(await taskId(catalog, verdict)).toBe(c.task_id);
      for (const step of c.args.steps) {
        const f = stepFidelity(catalog, c.args, step);
        expect(f.tier).toBe(c.fidelity[step.id]);
        expect(f.caveats).toEqual(c.caveats[step.id]);
      }
    });
  }
});

describe("recorded plans", () => {
  it("validate to their recorded status, heroes first", () => {
    expect(examples.filter((e) => e.hero).map((e) => e.case_id)).toEqual(["h1_tote", "h2_coupling", "h3_cabinet"]);
    for (const e of examples) {
      const verdict = validatePlan(catalog, planArgsOf(e.plan), e.uploads.map((u) => u.upload_id));
      expect(verdict.status, e.case_id).toBe(e.status);
    }
  });

  it("an upload plan needs its upload", () => {
    const e = examples.find((x) => x.uploads.length)!;
    expect(validatePlan(catalog, planArgsOf(e.plan)).problems[0]).toMatch(/was not uploaded/);
  });
});

describe("edits", () => {
  const h3 = () => planArgsOf(examples.find((e) => e.case_id === "h3_cabinet")!.plan);

  it("reorder and delete steps, flip a direction, and point problems at their entry", () => {
    const moved = moveStep(h3(), 2, 1);
    expect(moved.steps.map((s) => s.primitive)).toEqual(["twist", "hook_pull", "insert", "twist"]);
    expect(moveStep(h3(), 0, -1)).toEqual(h3());
    const flipped = h3();
    flipped.steps[0]!.params.direction = "cw";
    const verdict = validatePlan(catalog, flipped);
    expect(verdict.status).toBe("invalid");
    expect(verdict.problems[0]).toMatch(/opens counter-clockwise; twist it ccw/);
    expect(problemTarget(verdict.problems[0]!, flipped)).toEqual({ kind: "step", index: 0 });
    const removed = deleteStep(h3(), 1);
    expect(removed.steps).toHaveLength(3);
    expect(removed.steps.every((s) => !s.after.includes(h3().steps[1]!.id))).toBe(true);
    expect(validatePlan(catalog, removed).status).toBe("valid");
  });

  it("add objects and steps with defaults that validate", () => {
    let plan = addObject(catalog, emptyPlan(), "drawer");
    plan = addObject(catalog, plan, "drawer");
    expect(plan.objects.map((o) => o.name)).toEqual(["drawer", "drawer_2"]);
    plan = addStep(catalog, plan, "hook_pull");
    expect(plan.steps[0]).toMatchObject({ object: "drawer", handle: "handle", primitive: "hook_pull" });
    expect(validatePlan(catalog, plan).problems).toEqual([]);
    expect(validatePlan(catalog, plan).warnings).toEqual(["objects[1] (drawer_2): no step acts on it or places into it"]);
    plan = deleteObject(plan, 0);
    expect(validatePlan(catalog, plan).problems[0]).toBe("step hook_pull_drawer: unknown object 'drawer'");
    let box = addObject(catalog, emptyPlan(), "part");
    box = addStep(catalog, box, "grasp");
    expect(validatePlan(catalog, box).status).toBe("valid");
    let twist = addObject(catalog, emptyPlan(), "handwheel_valve");
    twist = addStep(catalog, twist, "twist");
    expect(twist.steps[0]!.params.direction).toBe("ccw");
    expect(validatePlan(catalog, twist).status).toBe("valid");
    let thread = addObject(catalog, emptyPlan(), "threaded_joint");
    thread = addStep(catalog, thread, "insert");
    thread = addStep(catalog, thread, "twist");
    expect(thread.steps.map((s) => [s.primitive, s.params.direction ?? null])).toEqual([["insert", null], ["twist", "cw"]]);
    expect(validatePlan(catalog, thread).problems).toEqual([]);
    expect(addStep(catalog, thread, "grasp").steps[2]!.object).not.toBe(thread.objects[0]!.name); // no pick on a thread
  });

  it("exports TaskSpec.to_dict with the task id and fidelity", async () => {
    const plan = h3();
    const verdict = validatePlan(catalog, plan);
    const id = await taskId(catalog, verdict);
    const doc = exportDocument(catalog, plan, id, "recorded");
    expect(doc.contract).toBe("rtt-effectors/2");
    expect(doc.label).toBe("Product mode: not part of the pre-registered experiment");
    expect(doc.task_id).toBe(id);
    expect(Object.keys(doc.fidelity as object)).toEqual(plan.steps.map((s) => s.id));
    const recorded = examples.find((e) => e.case_id === "h3_cabinet")!.plan;
    expect(doc.task_id).toBe(recorded.task_id);
    expect(doc.fidelity).toEqual(recorded.fidelity);
  });
});
