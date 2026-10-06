import { describe, expect, it } from "vitest";

import { decodeTerrain, openCorners, TerrainKind } from "../src/terrain.js";

describe("decodeTerrain", () => {
  it("reads the base from bits 4-6 and the shared corners from bits 0-3", () => {
    expect(decodeTerrain(0x00)).toEqual({ kind: TerrainKind.Ground, sharedCorners: 0 });
    expect(decodeTerrain(0x2a)).toEqual({ kind: TerrainKind.Water, sharedCorners: 0b1010 });
    expect(decodeTerrain(0x4f)).toEqual({ kind: TerrainKind.Obstacle, sharedCorners: 0b1111 });
  });

  it("ignores the variant bit", () => {
    expect(decodeTerrain(0xa3)).toEqual(decodeTerrain(0x23));
  });

  it("refuses a base the game does not define and a value outside a byte", () => {
    expect(() => decodeTerrain(0x10)).toThrow("RENDER_TERRAIN: byte 0x10 has base 1, which the game does not define");
    expect(() => decodeTerrain(256)).toThrow("RENDER_TERRAIN: 256 is not a terrain byte");
    expect(() => decodeTerrain(-1)).toThrow("RENDER_TERRAIN: -1 is not a terrain byte");
    expect(() => decodeTerrain(1.5)).toThrow("RENDER_TERRAIN: 1.5 is not a terrain byte");
  });
});

describe("openCorners", () => {
  it("lists, in bit order, the corners whose neighbour differs", () => {
    expect(openCorners({ kind: TerrainKind.Water, sharedCorners: 0b0000 })).toEqual([0, 1, 2, 3]);
    expect(openCorners({ kind: TerrainKind.Water, sharedCorners: 0b0101 })).toEqual([1, 3]);
    expect(openCorners({ kind: TerrainKind.Water, sharedCorners: 0b1111 })).toEqual([]);
  });
});
