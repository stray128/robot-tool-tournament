import { describe, expect, it } from "vitest";

import {
  capitalizeFirst,
  formatClock,
  formatDuration,
  formatLift,
  formatPercent,
  formatUsd,
  formatUtc,
} from "@/lib/format";

describe("formatting", () => {
  it("durations pick a unit by size", () => {
    expect(formatDuration(0.469)).toBe("0.47 s");
    expect(formatDuration(41.23)).toBe("41.2 s");
    expect(formatDuration(461)).toBe("7 min 41 s");
    expect(formatDuration(7500)).toBe("2 h 05 min");
    expect(formatDuration(-1)).toBe("n/a");
  });

  it("dollars keep sub-cent precision", () => {
    expect(formatUsd(0.0034)).toBe("$0.0034");
    expect(formatUsd(0.2)).toBe("$0.20");
    expect(formatUsd(0)).toBe("$0");
  });

  it("lift readouts are signed and quiet near zero", () => {
    expect(formatLift(0.1491)).toBe("+149.1");
    expect(formatLift(-0.00002)).toBe("0.0");
    expect(formatLift(-0.0031)).toBe("-3.1");
  });

  it("clock, percent and UTC", () => {
    expect(formatClock(4.271)).toBe("04.27");
    expect(formatPercent(0.4789, 1)).toBe("47.9%");
    expect(formatUtc("2026-10-15T09:58:50Z")).toBe("2026-10-15 09:58 UTC");
  });

  it("captions from data start with a capital", () => {
    expect(capitalizeFirst("the straight entrant lifts")).toBe("The straight entrant lifts");
    expect(capitalizeFirst("mu = 0.05")).toBe("Mu = 0.05");
    expect(capitalizeFirst("")).toBe("");
  });
});
