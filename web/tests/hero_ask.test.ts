/**
 * The home question box and the navigation fix. "Plan it in the Studio" calls router.push, and a hero
 * replay that is still playing used to keep the router from finishing that navigation, so the box
 * first tells the replays that the visitor is leaving (announceLeave, which pauses their clocks).
 * The tests of leaving.ts cover that function but not its caller: if an edit dropped the call, every
 * test would stay green and the button would again do nothing once the hero had been playing for a
 * few seconds. This suite has no DOM, so the guard reads the component's source. It wants an import
 * of announceLeave from the leaving module and, for every router.push, an unconditional
 * announceLeave() earlier in the same function. The checker is run on edited copies of the real file
 * to show that each way of losing the call fails.
 */

import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const HERO_ASK = readFileSync(path.join(__dirname, "..", "components", "home", "HeroAsk.tsx"), "utf8");
const LEAVING = "@/components/replay/leaving";

/** The source without its comments, so a call that is only commented out does not count. A "//" right after a colon is a URL. */
function withoutComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
}

/** The bodies of the functions in code: the text between the braces that follow an arrow or a function header. */
function functionBodies(code: string): { start: number; end: number }[] {
  const bodies: { start: number; end: number }[] = [];
  for (const m of code.matchAll(/=>\s*\{|\bfunction\b[^{]*\)\s*\{/g)) {
    const open = (m.index ?? 0) + m[0].length - 1;
    let depth = 0;
    for (let i = open; i < code.length; i++) {
      if (code[i] === "{") depth += 1;
      else if (code[i] === "}") {
        depth -= 1;
        if (depth === 0) {
          bodies.push({ start: open + 1, end: i });
          break;
        }
      }
    }
  }
  return bodies;
}

/** How many braces are open at position at, counting from position from. */
function depthAt(code: string, from: number, at: number): number {
  let depth = 0;
  for (let i = from; i < at; i++) {
    if (code[i] === "{") depth += 1;
    else if (code[i] === "}") depth -= 1;
  }
  return depth;
}

/** What is wrong with how source leaves the page; empty when every router.push has an unconditional announceLeave() before it in the same function. */
function leavePushProblems(source: string): string[] {
  const code = withoutComments(source);
  const problems: string[] = [];
  if (!new RegExp(`import\\s*\\{[^}]*\\bannounceLeave\\b[^}]*\\}\\s*from\\s*["']${LEAVING}["']`).test(code)) {
    problems.push(`announceLeave is not imported from ${LEAVING}`);
  }
  const pushes = [...code.matchAll(/\brouter\.push\(/g)].map((m) => m.index ?? 0);
  if (pushes.length === 0) problems.push("there is no router.push: if the question box navigates another way, this guard has to follow it");
  const bodies = functionBodies(code);
  for (const at of pushes) {
    const inner = bodies.filter((b) => b.start <= at && at < b.end).sort((a, b) => b.start - a.start)[0];
    if (!inner) {
      problems.push("a router.push is not inside a function body written with braces");
      continue;
    }
    const calls = [...code.slice(inner.start, inner.end).matchAll(/\bannounceLeave\(\)/g)].map((m) => inner.start + (m.index ?? 0));
    const before = calls.filter((c) => c < at);
    if (before.length === 0) problems.push(calls.length === 0 ? "a router.push is not preceded by announceLeave()" : "announceLeave() comes after the router.push, not before it");
    else if (!before.some((c) => depthAt(code, inner.start, c) === 0)) problems.push("announceLeave() is called only inside a branch, so some visits skip it");
  }
  return problems;
}

describe("the question box leaves the page politely", () => {
  it("tells the replays, in the same function and before it navigates", () => {
    expect(HERO_ASK).toMatch(/router\.push\("\/studio\/#plan"\)/);
    expect(leavePushProblems(HERO_ASK)).toEqual([]);
  });

  describe("the guard fails when", () => {
    const cases: [string, (s: string) => string, RegExp][] = [
      ["the call is removed", (s) => s.replace("announceLeave();", ""), /not preceded by announceLeave/],
      ["the call is only commented out", (s) => s.replace("announceLeave();", "// announceLeave();"), /not preceded by announceLeave/],
      [
        "the call moves after the push",
        (s) => s.replace(/announceLeave\(\);(\s*)(router\.push\("[^"]*"\);)/, "$2$1announceLeave();"),
        /comes after the router\.push/,
      ],
      ["the call is made conditional", (s) => s.replace("announceLeave();", "if (text.length > 3) { announceLeave(); }"), /only inside a branch/],
      ["the import goes", (s) => s.replace(`import { announceLeave } from "${LEAVING}";\n`, ""), /not imported/],
      ["the import comes from somewhere else", (s) => s.replace(LEAVING, "@/lib/leaving"), /not imported/],
      [
        "another way to navigate is added without it",
        (s) => `${s}\nexport function Other() {\n  const router = useRouter();\n  return () => {\n    router.push("/atlas/");\n  };\n}\n`,
        /not preceded by announceLeave/,
      ],
      ["the push goes", (s) => s.replace(/router\.push\("[^"]*"\);/, ""), /no router\.push/],
    ];

    it.each(cases)("%s", (_name, edit, expected) => {
      const edited = edit(HERO_ASK);
      expect(edited).not.toBe(HERO_ASK); // the edit applied, so the case is not vacuous
      expect(leavePushProblems(edited).join("\n")).toMatch(expected);
    });
  });
});
