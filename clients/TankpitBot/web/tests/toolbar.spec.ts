import { describe, expect, it } from "vitest";

import { equipmentSlot, hitTest, mapScroll, regionAt, scopeDirection, TOOLBAR_REGIONS, ToolbarAction } from "../src/toolbar.js";

describe("TOOLBAR_REGIONS", () => {
  it("are the client's four arrays, region by region", () => {
    expect(TOOLBAR_REGIONS.map((r) => r.x)).toEqual([10, 53, 97, 151, 175, 197, 151, 166, 206, 151, 175, 197, 233, 263, 282, 304, 328, 362]);
    expect(TOOLBAR_REGIONS.map((r) => r.y)).toEqual([2, 2, 2, 2, 2, 2, 16, 16, 16, 31, 31, 31, 8, 8, 8, 8, 8, 8]);
    expect(TOOLBAR_REGIONS.map((r) => r.w)).toEqual([43, 44, 43, 24, 22, 24, 15, 40, 15, 24, 22, 24, 30, 19, 22, 24, 31, 20]);
    expect(TOOLBAR_REGIONS.map((r) => r.h)).toEqual([44, 44, 44, 14, 14, 14, 15, 15, 15, 15, 15, 15, 26, 26, 26, 26, 26, 30]);
    expect(TOOLBAR_REGIONS.map((r) => r.action)).toEqual(Array.from({ length: 18 }, (_, index) => index));
  });
});

describe("hitTest", () => {
  it("lands each region's top-left pixel, three pixels above its stored y", () => {
    for (const region of TOOLBAR_REGIONS) {
      expect(hitTest(region.x, region.y - 3)).toBe(region.action);
    }
  });

  it("matches the wiki's region map at its listed corners", () => {
    expect(hitTest(10, 2)).toBe(ToolbarAction.OpenMap);
    expect(hitTest(166, 16)).toBe(ToolbarAction.ScopeCenter);
    expect(hitTest(362, 8)).toBe(ToolbarAction.Promotion);
  });

  it("misses between regions, above the shifted top and past the right edge", () => {
    expect(hitTest(5, 10)).toBe(-1);
    expect(hitTest(10, -2)).toBe(-1);
    expect(hitTest(382, 10)).toBe(-1);
    expect(regionAt(5, 10)).toBeNull();
    expect(regionAt(53, 0)).toBe(TOOLBAR_REGIONS[1]);
  });
});

describe("equipmentSlot", () => {
  it("numbers the five equipment buttons 0 to 4 and nothing else", () => {
    expect([12, 13, 14, 15, 16].map((action) => equipmentSlot(action))).toEqual([0, 1, 2, 3, 4]);
    expect(equipmentSlot(ToolbarAction.Radar)).toBeNull();
    expect(equipmentSlot(ToolbarAction.Promotion)).toBeNull();
  });
});

describe("scopeDirection", () => {
  it("is the client's qe table and passes other values through", () => {
    expect([0, 1, 2, 3, 4, 5, 6, 7].map(scopeDirection)).toEqual([4, 5, 6, 7, 0, 1, 2, 3]);
    expect([8, -1].map(scopeDirection)).toEqual([8, -1]);
  });
});

describe("mapScroll", () => {
  it("is the client's le step, 64 map pixels clockwise from north", () => {
    expect([0, 1, 2, 3, 4, 5, 6, 7].map(mapScroll)).toEqual([
      { dx: 0, dy: -64 },
      { dx: 64, dy: -64 },
      { dx: 64, dy: 0 },
      { dx: 64, dy: 64 },
      { dx: 0, dy: 64 },
      { dx: -64, dy: 64 },
      { dx: -64, dy: 0 },
      { dx: -64, dy: -64 },
    ]);
  });

  it("resets on the centre and refuses anything else", () => {
    expect(mapScroll(8)).toBeNull();
    expect(() => mapScroll(9)).toThrow("RENDER_SCROLL: direction 9 is not 0 to 8");
  });
});
