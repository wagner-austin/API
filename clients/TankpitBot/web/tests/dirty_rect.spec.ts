import { describe, expect, it } from "vitest";

import { DirtyRect, overlaps } from "../src/dirty_rect.js";
import { ScaledContext } from "../src/scaled_context.js";
import { RecordingSurface } from "./fakes.js";

describe("DirtyRect", () => {
  it("grows to the union of everything drawn and erases exactly that", () => {
    const dirty = new DirtyRect();
    expect(dirty.current).toBeNull();
    dirty.include({ x: 10, y: 10, w: 5, h: 5 });
    dirty.include({ x: 4, y: 12, w: 3, h: 8 });
    expect(dirty.current).toEqual({ x: 4, y: 10, w: 11, h: 10 });
    const surface = new RecordingSurface();
    expect(dirty.erase(new ScaledContext(surface, 2))).toEqual({ x: 4, y: 10, w: 11, h: 10 });
    expect(surface.calls).toEqual([{ op: "clear", image: null, fill: "", source: [], dest: [8, 20, 22, 20] }]);
    expect(dirty.current).toBeNull();
  });

  it("clears nothing when nothing was drawn", () => {
    const surface = new RecordingSurface();
    expect(new DirtyRect().erase(new ScaledContext(surface, 1))).toBeNull();
    expect(surface.calls).toEqual([]);
  });
});

describe("overlaps", () => {
  const a = { x: 0, y: 0, w: 10, h: 10 };

  it("is true when the rectangles share a pixel", () => {
    expect(overlaps(a, { x: 9, y: 9, w: 5, h: 5 })).toBe(true);
  });

  it("is false for rectangles that only touch, on either axis", () => {
    expect(overlaps(a, { x: 10, y: 0, w: 5, h: 5 })).toBe(false);
    expect(overlaps(a, { x: 0, y: 10, w: 5, h: 5 })).toBe(false);
    expect(overlaps({ x: 10, y: 0, w: 5, h: 5 }, a)).toBe(false);
    expect(overlaps({ x: 0, y: 10, w: 5, h: 5 }, a)).toBe(false);
  });
});
