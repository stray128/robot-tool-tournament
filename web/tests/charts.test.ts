import { describe, expect, it } from "vitest";

import { medianCurve } from "@/components/charts/BestSoFarChart";
import { labelRows } from "@/components/charts/ForestPlot";

describe("chart helpers", () => {
  it("puts close reference labels on separate lines", () => {
    expect(labelRows([0.376, 0.554, 0.402])).toEqual([0, 0, 1]);
    expect(labelRows([0.1, 0.5, 0.9])).toEqual([0, 0, 0]);
    expect(labelRows([])).toEqual([]);
  });

  it("takes the per-evaluation median across runs of unequal length", () => {
    expect(medianCurve([{ best: [0.1, 0.2, 0.3] }, { best: [0.3, 0.4] }, { best: [0.2, 0.6, 0.9] }])).toEqual([0.2, 0.4, 0.6]);
  });
});
