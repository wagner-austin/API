import { describe, expect, it } from "vitest";

import { isExactScale, RenderError, ScaledContext } from "../src/scaled_context.js";
import { RecordingSurface } from "./fakes.js";

const image = document.createElement("canvas");
const source = { x: 24, y: 16, w: 24, h: 16 };

describe("ScaledContext", () => {
  it("draws exactly at a whole number of quarters", () => {
    const surface = new RecordingSurface();
    const context = new ScaledContext(surface, 1.5);
    expect(context.exact).toBe(true);
    expect(surface.imageSmoothingEnabled).toBe(false);
    context.draw(image, source, 10, 20);
    expect(surface.calls).toEqual([{ op: "draw", image, fill: "", source: [24, 16, 24, 16], dest: [15, 30, 36, 24] }]);
  });

  it("floors the corner and draws 3% oversize, rounded up, at any other scale", () => {
    const surface = new RecordingSurface();
    const context = new ScaledContext(surface, 1.1);
    expect(context.exact).toBe(false);
    context.draw(image, source, 10, 20);
    expect(surface.calls[0]?.dest).toEqual([11, 22, Math.ceil(24 * 1.1 * 1.03), Math.ceil(16 * 1.1 * 1.03)]);
  });

  it("clears and fills in game pixels", () => {
    const surface = new RecordingSurface();
    const context = new ScaledContext(surface, 2);
    context.clear(1, 2, 3, 4);
    context.fill("#123456", 5, 6, 7, 8);
    expect(surface.calls).toEqual([
      { op: "clear", image: null, fill: "", source: [], dest: [2, 4, 6, 8] },
      { op: "fill", image: null, fill: "#123456", source: [], dest: [10, 12, 14, 16] },
    ]);
  });

  it("refuses a scale that is not a positive finite number", () => {
    const surface = new RecordingSurface();
    for (const scale of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => new ScaledContext(surface, scale)).toThrow(RenderError);
    }
    expect(() => new ScaledContext(surface, 0)).toThrow("RENDER_SCALE: scale 0 is not a positive number");
  });
});

describe("isExactScale", () => {
  it("is true for quarters and false between them", () => {
    expect([1, 1.25, 1.5, 2, 3].map(isExactScale)).toEqual([true, true, true, true, true]);
    expect([1.1, 1.333, 2.2].map(isExactScale)).toEqual([false, false, false]);
  });
});
