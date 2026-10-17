import { describe, expect, it } from "vitest";

import {
  capitalizeFirst,
  formatBytes,
  formatClock,
  formatDuration,
  formatLift,
  formatMinutes,
  formatOffset,
  formatP,
  formatPercent,
  formatUsd,
  formatUtc,
} from "@/lib/format";

describe("formatting", () => {
  it("p values as docs/RESULTS.md writes them", () => {
    // values from the confirmatory analysis outputs, strings as in RESULTS.md
    expect(formatP(0.0011198008689374398)).toBe("0.0011");
    expect(formatP(0.0010999450027498626)).toBe("0.0011");
    expect(formatP(6.999965000175e-5)).toBe("7e-05");
    expect(formatP(5.4999725001374995e-5)).toBe("5.5e-05");
    expect(formatP(0.00076999615001925)).toBe("0.00077");
    expect(formatP(0.000164999175004125)).toBe("0.00017");
    expect(formatP(0.012099939500302498)).toBe("0.012");
    expect(formatP(0.0010099949500252498)).toBe("0.001");
    expect(formatP(1)).toBe("1.0");
    expect(formatP(0.5)).toBe("0.5");
    expect(formatP(0.18175909)).toBe("0.18");
    expect(formatP(0.99996)).toBe("1.0");
    expect(formatP(Number.NaN)).toBe("n/a");
  });

  it("job times in whole minutes", () => {
    expect(formatMinutes(1671.307803)).toBe("28 min");
    expect(formatMinutes(5038.469007)).toBe("84 min");
    expect(formatMinutes(-1)).toBe("n/a");
  });

  it("session offsets and byte sizes", () => {
    expect(formatOffset(6.4)).toBe("0:06");
    expect(formatOffset(342.2)).toBe("5:42");
    expect(formatOffset(-3)).toBe("0:00");
    expect(formatBytes(940)).toBe("940 B");
    expect(formatBytes(18484)).toBe("18.5 kB");
    expect(formatBytes(2.4e6)).toBe("2.4 MB");
  });

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
