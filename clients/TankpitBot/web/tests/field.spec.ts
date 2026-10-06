import { describe, expect, it } from "vitest";

import { FIELD_SPAN, FieldClass, FieldTerrain } from "../src/field.js";
import { fieldOf } from "./fields.js";

describe("FieldTerrain", () => {
  it("reads one class byte per tile, row by row, and nothing off the field", () => {
    const field = fieldOf([
      [3, 0, FieldClass.Rock],
      [0, 2, FieldClass.Water],
    ]);
    expect([field.classAt(3, 0), field.classAt(0, 2), field.classAt(1, 1)]).toEqual([FieldClass.Rock, FieldClass.Water, FieldClass.Ground]);
    expect([field.classAt(-1, 0), field.classAt(0, -1), field.classAt(256, 0), field.classAt(0, 256)]).toEqual([null, null, null, null]);
  });

  it("refuses a body that is not one byte per tile, or a byte that is no class", () => {
    expect(() => FieldTerrain.decode(new Uint8Array(10))).toThrow("FIELD_SIZE: a field is 65536 bytes, not 10");
    const bytes = new Uint8Array(FIELD_SPAN * FIELD_SPAN);
    bytes[2 * FIELD_SPAN + 5] = 3;
    expect(() => FieldTerrain.decode(bytes)).toThrow("FIELD_CLASS: tile (5, 2) is class 3");
  });

  it("does not share the caller's buffer", () => {
    const bytes = new Uint8Array(FIELD_SPAN * FIELD_SPAN);
    const field = FieldTerrain.decode(bytes);
    bytes[0] = FieldClass.Water;
    expect(field.classAt(0, 0)).toBe(FieldClass.Ground);
  });
});

describe("terrainByte", () => {
  it("is the base and the diagonal neighbours sharing it, NE SE SW NW from bit 0", () => {
    const field = fieldOf([
      [10, 10, FieldClass.Water],
      [11, 9, FieldClass.Water],
      [9, 11, FieldClass.Water],
      [20, 20, FieldClass.Rock],
    ]);
    expect(field.terrainByte(10, 10)).toBe(0x20 | 0b0101);
    expect(field.terrainByte(20, 20)).toBe(0x40);
    expect(field.terrainByte(50, 50)).toBe(0x0f);
  });

  it("counts a neighbour off the field as sharing", () => {
    const field = fieldOf([[0, 0, FieldClass.Rock]]);
    expect(field.terrainByte(0, 0)).toBe(0x40 | 0b1101);
    expect(field.terrainByte(255, 255)).toBe(0x0f);
  });

  it("refuses a tile off the field", () => {
    expect(() => fieldOf([]).terrainByte(256, 0)).toThrow("FIELD_TILE: (256, 0) is not on the field");
  });
});
