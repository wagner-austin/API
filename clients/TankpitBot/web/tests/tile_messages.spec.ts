import { describe, expect, it } from "vitest";

import { NO_MINE } from "../src/tile_grid.js";
import { decodePickup, decodeRadarScan, decodeTerrainUpdate, decodeViewport, EQUIPMENT, isPickup, isRadarScan } from "../src/tile_messages.js";
import { GOLDEN, hex } from "./golden.js";

describe("decodeViewport", () => {
  it("walks the server's skip-coded patch: tiles, a pure skip, caches, mines, rocks", () => {
    expect(decodeViewport(GOLDEN.viewport.subarray(1))).toEqual({
      left: 238,
      top: 0,
      tiles: [
        { column: 1, row: 1, cache: 0, mine: NO_MINE, terrain: 5 },
        { column: 3, row: 16, cache: 730, mine: NO_MINE, terrain: 0 },
        { column: 17, row: 17, cache: EQUIPMENT, mine: 2, terrain: 1 },
      ],
    });
  });

  it("refuses a body without its origin or with a tile record cut off", () => {
    expect(() => decodeViewport(hex("ee"))).toThrow("WIRE_SHAPE: a viewport update needs its origin, not 1 bytes");
    expect(() => decodeViewport(hex("ee00130000"))).toThrow("WIRE_SHAPE: a viewport tile record at byte 3 is cut off");
  });
});

describe("pickups", () => {
  it("are whole four-byte records, each the volume a tile has left", () => {
    const inner = GOLDEN.pickup.subarray(1);
    expect(isPickup(inner)).toBe(true);
    expect(decodePickup(inner)).toEqual([{ x: 40, y: 41, value: 257 }]);
    expect(isPickup(hex("282901"))).toBe(false);
    expect(isPickup(hex("2829010100"))).toBe(false);
  });
});

describe("radar scans", () => {
  it("are a count of caches, then mine records, team or none", () => {
    const inner = GOLDEN.radar.subarray(1);
    expect(isRadarScan(inner)).toBe(true);
    expect(decodeRadarScan(inner)).toEqual({
      caches: [
        { x: 5, y: 6, value: 300 },
        { x: 7, y: 8, value: EQUIPMENT },
      ],
      mines: [
        { x: 9, y: 10, value: 3 },
        { x: 11, y: 12, value: NO_MINE },
      ],
    });
  });

  it("are not a body too short for its count, its caches or whole mine records", () => {
    expect(isRadarScan(hex("01"))).toBe(false);
    expect(isRadarScan(hex("0100050600"))).toBe(false);
    expect(isRadarScan(hex("0000090a"))).toBe(false);
  });
});

describe("decodeTerrainUpdate", () => {
  it("reads whole three-byte records and refuses a partial one", () => {
    expect(decodeTerrainUpdate(GOLDEN.terrain.subarray(1))).toEqual([
      { x: 10, y: 20, value: 2 },
      { x: 11, y: 20, value: 0 },
    ]);
    expect(() => decodeTerrainUpdate(hex("0a14"))).toThrow("WIRE_SHAPE: a terrain update of 2 bytes is not whole records");
  });
});
