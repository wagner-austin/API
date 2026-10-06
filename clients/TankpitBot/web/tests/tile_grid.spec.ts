import { describe, expect, it } from "vitest";

import { defaultManifest } from "../src/default_pack.js";
import { ScaledContext } from "../src/scaled_context.js";
import type { SheetImages } from "../src/sprites.js";
import { decodeTerrain } from "../src/terrain.js";
import { drawTile, GRID_SIZE, NO_MINE, RockKind, TileGrid, type TileState } from "../src/tile_grid.js";
import { RecordingSurface } from "./fakes.js";

const manifest = defaultManifest("sheet.png");
const sheet = document.createElement("canvas");
const sheets: SheetImages = { images: new Map([["default", sheet]]) };

function tile(byte: number, extra: Partial<TileState> = {}): TileState {
  return { terrain: decodeTerrain(byte), cache: 0, mine: NO_MINE, rock: RockKind.None, ...extra };
}

function sources(surface: RecordingSurface): number[][] {
  return surface.only("draw").map((call) => [...call.source.slice(0, 2), ...call.dest.slice(0, 2)]);
}

describe("drawTile", () => {
  it("draws ground alone", () => {
    const surface = new RecordingSurface();
    drawTile(new ScaledContext(surface, 1), manifest, sheets, tile(0x0f), 24, 16);
    expect(sources(surface)).toEqual([[0, 0, 24, 16]]);
  });

  it("draws water and a corner overlay toward each neighbour that is not water", () => {
    const surface = new RecordingSurface();
    drawTile(new ScaledContext(surface, 1), manifest, sheets, tile(0x20 | 0b0110), 0, 0);
    expect(sources(surface)).toEqual([
      [24, 0, 0, 0],
      [0, 16, 0, 0],
      [72, 16, 0, 0],
    ]);
  });

  it("draws an obstacle's corners from the obstacle set", () => {
    const surface = new RecordingSurface();
    drawTile(new ScaledContext(surface, 1), manifest, sheets, tile(0x40 | 0b1110), 0, 0);
    expect(sources(surface)).toEqual([
      [48, 0, 0, 0],
      [0, 32, 0, 0],
    ]);
  });

  it("stacks base, cache dot, rock and mine in the game's order", () => {
    const surface = new RecordingSurface();
    drawTile(new ScaledContext(surface, 1), manifest, sheets, tile(0x0f, { cache: -3, rock: RockKind.Ferry, mine: 2 }), 0, 0);
    expect(sources(surface)).toEqual([
      [0, 0, 0, 0],
      [manifest.equipment.x, manifest.equipment.y, 0, 0],
      [0, 64, 0, 0],
      [48, 80, 0, 0],
    ]);
  });

  it("draws the fuel dot for a positive cache", () => {
    const surface = new RecordingSurface();
    drawTile(new ScaledContext(surface, 1), manifest, sheets, tile(0x0f, { cache: 5, rock: RockKind.A }), 0, 0);
    expect(sources(surface)).toEqual([
      [0, 0, 0, 0],
      [manifest.fuel.x, manifest.fuel.y, 0, 0],
      [48, 48, 0, 0],
    ]);
  });
});

describe("TileGrid", () => {
  it("starts dirty everywhere and draws only the 16x16 viewport, at its offset", () => {
    const grid = new TileGrid();
    const surface = new RecordingSurface();
    expect(grid.dirtyCount()).toBe(GRID_SIZE * GRID_SIZE);
    expect(grid.drawDirty(new ScaledContext(surface, 1), manifest, sheets)).toBe(256);
    expect(grid.dirtyCount()).toBe(GRID_SIZE * GRID_SIZE - 256);
    const dests = surface.only("draw").map((call) => call.dest.slice(0, 2));
    expect(dests[0]).toEqual([0, 0]);
    expect(dests[255]).toEqual([15 * 24, 15 * 16]);
  });

  it("redraws only a tile that was set since the last draw", () => {
    const grid = new TileGrid();
    const context = new ScaledContext(new RecordingSurface(), 1);
    grid.drawDirty(context, manifest, sheets);
    const surface = new RecordingSurface();
    grid.set(3, 2, tile(0x2f));
    expect(grid.get(3, 2)).toEqual(tile(0x2f));
    expect(grid.drawDirty(new ScaledContext(surface, 1), manifest, sheets)).toBe(1);
    expect(surface.only("draw").map((call) => call.dest.slice(0, 2))).toEqual([[48, 16]]);
    expect(grid.drawDirty(new ScaledContext(surface, 1), manifest, sheets)).toBe(0);
  });

  it("keeps a border tile dirty until a scroll would bring it on screen", () => {
    const grid = new TileGrid();
    grid.drawDirty(new ScaledContext(new RecordingSurface(), 1), manifest, sheets);
    grid.set(0, 0, tile(0x20));
    expect(grid.drawDirty(new ScaledContext(new RecordingSurface(), 1), manifest, sheets)).toBe(0);
  });

  it("refuses a tile outside the grid and a mine that is not a team", () => {
    const grid = new TileGrid();
    expect(() => grid.set(18, 0, tile(0))).toThrow("RENDER_TILE: (18, 0) is outside the 18x18 grid");
    expect(() => grid.get(0, -1)).toThrow("RENDER_TILE: (0, -1) is outside the 18x18 grid");
    expect(() => grid.get(0.5, 1)).toThrow("RENDER_TILE: (0.5, 1) is outside the 18x18 grid");
    expect(() => grid.set(1, 1, tile(0, { mine: 4 }))).toThrow("RENDER_MINE: mine 4 is not a team (0 to 3) or none (255)");
    expect(() => grid.set(1, 1, tile(0, { mine: 1.5 }))).toThrow("RENDER_MINE");
    expect(() => grid.set(1, 1, tile(0, { mine: -1 }))).toThrow("RENDER_MINE");
  });
});
