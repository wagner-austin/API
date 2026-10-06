import { describe, expect, it } from "vitest";

import { buildTable, byteAt, joinFrames, splitFrames, xorBody } from "../src/wire.js";

const bytes = (...values: number[]): Uint8Array => Uint8Array.from(values);

describe("frames", () => {
  it("join each body behind its two-byte little-endian length, and split back", () => {
    const long = new Uint8Array(300).fill(7);
    const joined = joinFrames([bytes(0x2b, 0x31), long]);
    expect(Array.from(joined.subarray(0, 6))).toEqual([2, 0, 0x2b, 0x31, 44, 1]);
    expect(splitFrames(joined).map((frame) => Array.from(frame))).toEqual([[0x2b, 0x31], Array.from(long)]);
    expect(splitFrames(new Uint8Array(0))).toEqual([]);
  });

  it("refuse a message whose length is cut off, runs past the end, or is zero", () => {
    expect(() => splitFrames(bytes(2, 0, 9, 9, 1))).toThrow("WIRE_TORN: a length at byte 4 is cut off");
    expect(() => splitFrames(bytes(3, 0, 9))).toThrow("WIRE_TORN: a frame of 3 bytes at byte 0 does not fit the 3-byte message");
    expect(() => splitFrames(bytes(0, 0))).toThrow("WIRE_TORN: a frame of 0 bytes at byte 0");
  });

  it("refuse to send an empty body or one past a length's reach", () => {
    expect(() => joinFrames([new Uint8Array(0)])).toThrow("WIRE_FRAME_SIZE: a frame of 0 bytes cannot be sent");
    expect(() => joinFrames([new Uint8Array(0x10000)])).toThrow("WIRE_FRAME_SIZE: a frame of 65536 bytes");
  });
});

describe("byteAt", () => {
  it("reads a byte that is there and refuses one past the end", () => {
    expect(byteAt(bytes(5, 6), 1)).toBe(6);
    expect(() => byteAt(bytes(5, 6), 2)).toThrow("WIRE_RANGE: byte 2 is past the end of 2");
  });
});

describe("the cipher", () => {
  it("folds the magic into the key, repeating the magic", () => {
    const table = buildTable("abc", "xy");
    expect(Array.from(table)).toEqual([0x61 ^ 0x78, 0x62 ^ 0x79, 0x63 ^ 0x78]);
  });

  it("refuses an empty key or magic and a character outside a byte", () => {
    expect(() => buildTable("", "m")).toThrow("WIRE_KEY: the static key and the magic must both be non-empty");
    expect(() => buildTable("k", "")).toThrow("WIRE_KEY");
    expect(() => buildTable("aĀ", "m")).toThrow("WIRE_KEY: character 1 of the key or magic is outside a byte");
  });

  it("XORs from an offset, wraps the table and undoes itself", () => {
    const table = bytes(1, 2);
    const body = bytes(0x2e, 10, 20, 30);
    const ciphered = xorBody(body, table, 1);
    expect(Array.from(ciphered)).toEqual([11, 22, 31]);
    expect(Array.from(xorBody(ciphered, table, 0))).toEqual([10, 20, 30]);
    expect(xorBody(bytes(1), table, 3).length).toBe(0);
  });

  it("refuses an empty table under a span", () => {
    expect(() => xorBody(bytes(1), new Uint8Array(0), 0)).toThrow("WIRE_RANGE");
  });
});
