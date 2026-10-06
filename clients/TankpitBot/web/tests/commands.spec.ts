import { describe, expect, it } from "vitest";

import { enterGameCommand, moveCommand } from "../src/commands.js";
import { buildTable, xorBody } from "../src/wire.js";

const TABLE = buildTable("static-key-for-tests", "magic5");

describe("commands", () => {
  it("are a '!' and the ciphered type, code and arguments", () => {
    const enter = enterGameCommand(TABLE);
    expect(enter[0]).toBe(0x21);
    expect(Array.from(xorBody(enter, TABLE, 1))).toEqual([2, 63]);
    expect(Array.from(xorBody(moveCommand(TABLE, 0, 255), TABLE, 1))).toEqual([4, 112, 0, 255]);
  });

  it("refuse a move to a tile off the field", () => {
    expect(() => moveCommand(TABLE, 256, 3)).toThrow("WIRE_TILE: (256, 3) is not a tile on the field");
    expect(() => moveCommand(TABLE, 3, -1)).toThrow("WIRE_TILE: (3, -1)");
    expect(() => moveCommand(TABLE, 1.5, 3)).toThrow("WIRE_TILE");
  });
});
