/**
 * The Evidence page: the v1 result as published, a dated note card on what v1 could not test, and
 * the v2 card saying plainly that its results are pending.
 */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import EvidencePage from "@/app/evidence/page";
import { V1_NOTE } from "@/app/evidence/notes";
import { getHeldout } from "@/app/studio/heldout";

describe("the Evidence page", () => {
  const html = renderToStaticMarkup(createElement(EvidencePage));

  it("keeps the v1 result card and points at its note", () => {
    expect(html).toContain('data-evidence="usecase"');
    expect(html).toContain('href="#v1-note"');
  });

  it("titles every v1 held-out row from the v1 recording, linked only to a page that still shows its setups", () => {
    const card = html.slice(html.indexOf('data-evidence="usecase"'), html.indexOf('data-evidence="fingers"'));
    for (const title of ["Returns tote", "Pipe coupling", "Equipment cabinet"]) expect(card).toContain(title);
    expect(card).not.toMatch(/h[0-9]_[a-z]+/); // no case id on screen
    // the v2 pages show other setups than the v1 result tested, so the v1 rows link nowhere
    expect(card).not.toContain('href="/studio/h3_cabinet/"');
    for (const h of Object.values(getHeldout().heroes)) {
      const linked = card.includes(`>${h.title}</a>`);
      if (linked) expect(card).toContain(`href="/studio/${h.case_id}/"`);
    }
  });

  it("carries the dated v1 note card", () => {
    const card = html.slice(html.indexOf('data-evidence="v1-note"'));
    expect(card).toContain(`added ${V1_NOTE.date}`);
    expect(V1_NOTE.lead).toBe("v1 used a top-down wrist; v2 adds wrist tilt and honest tolerances; v1 results stand as published.");
    expect(card).toContain(V1_NOTE.lead);
    for (const pt of V1_NOTE.points) expect(card).toContain(pt.title);
    expect(html).toContain('id="v1-note"');
  });

  it("says v2's results are pending, nothing more", () => {
    expect(html).toContain('data-evidence="usecase-v2"');
    expect(html).toContain('data-v2-status="pending"');
    expect(html).toContain("v2 pre-registration: results pending");
    expect(html).not.toContain("Pre-registration in progress");
  });
});
