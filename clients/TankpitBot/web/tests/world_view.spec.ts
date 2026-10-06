import { afterEach, describe, expect, it } from "vitest";

import { resetHooks } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { FieldClass } from "../src/field.js";
import { Renderer } from "../src/renderer.js";
import { TerrainKind } from "../src/terrain.js";
import { NO_MINE, RockKind } from "../src/tile_grid.js";
import { EQUIPMENT } from "../src/tile_messages.js";
import { tileKey, WorldView } from "../src/world_view.js";
import { installFakes, type Fakes } from "./fakes.js";
import { fieldOf } from "./fields.js";

afterEach(resetHooks);

interface Scene {
  readonly view: WorldView;
  readonly renderer: Renderer;
  readonly fakes: Fakes;
}

async function scene(): Promise<Scene> {
  const fakes = installFakes();
  const renderer = await Renderer.create(document.createElement("div"), 1, defaultManifest("s.png"));
  const field = fieldOf([
    [101, 50, FieldClass.Water],
    [102, 51, FieldClass.Rock],
  ]);
  return { view: new WorldView(field, renderer), renderer, fakes };
}

/** Where the tanks layer last drew, as [x, y] game pixels: a tank at window (c, r) is at 24c - 2, 16r - 2. */
function lastTankDraw(fakes: Fakes): readonly number[] {
  return fakes.surfaces[1]?.only("draw").at(-1)?.dest.slice(0, 2) ?? [];
}

describe("the window and its tiles", () => {
  it("paints all 256 window tiles when the window moves, and only touched tiles after", async () => {
    const { view, renderer } = await scene();
    view.viewport({ left: 100, top: 50, tiles: [] });
    expect(view.window).toEqual([100, 50]);
    expect(view.paint()).toEqual({ tilesDrawn: 256, tanksDrawn: 0 });
    expect(renderer.grid.get(2, 1).terrain).toEqual({ kind: TerrainKind.Water, sharedCorners: 0 });
    expect(renderer.grid.get(3, 2).terrain.kind).toBe(TerrainKind.Obstacle);
    view.viewport({ left: 100, top: 50, tiles: [{ column: 3, row: 3, cache: EQUIPMENT, mine: 2, terrain: 5 }] });
    expect(view.paint()).toEqual({ tilesDrawn: 1, tanksDrawn: 0 });
    expect(renderer.grid.get(3, 3)).toMatchObject({ cache: EQUIPMENT, mine: 2, rock: RockKind.Ferry });
    expect(view.paint()).toEqual({ tilesDrawn: 0, tanksDrawn: 0 });
  });

  it("writes caches, radar mines and rocks to field tiles, the window's and others", async () => {
    const { view } = await scene();
    view.caches([{ x: 104, y: 52, value: 257 }]);
    view.radar({ caches: [{ x: 7, y: 8, value: EQUIPMENT }], mines: [{ x: 104, y: 52, value: 3 }] });
    view.rocks([
      { x: 104, y: 53, value: 2 },
      { x: 104, y: 54, value: 3 },
      { x: 104, y: 55, value: 1 },
      { x: 104, y: 56, value: 7 },
      { x: 104, y: 57, value: 0 },
    ]);
    expect(view.tileState(104, 52)).toMatchObject({ cache: 257, mine: 3, rock: RockKind.None });
    expect(view.tileState(7, 8).cache).toBe(EQUIPMENT);
    expect([53, 54, 55, 56, 57].map((y) => view.tileState(104, y).rock)).toEqual([RockKind.B, RockKind.B, RockKind.A, RockKind.Ferry, RockKind.None]);
    expect(view.tileState(0, 0)).toMatchObject({ cache: 0, mine: NO_MINE, rock: RockKind.None });
  });

  it("refuses dynamic terrain the game does not define, and tiles off the field", async () => {
    const { view } = await scene();
    expect(() => view.rocks([{ x: 1, y: 1, value: 4 }])).toThrow("RENDER_ROCK: tile (1, 1) carries dynamic terrain 4, which the game does not define");
    expect(() => tileKey(-1, 0)).toThrow("RENDER_TILE: (-1, 0) is not on the field");
    expect(() => tileKey(0, 256)).toThrow("RENDER_TILE");
    expect(() => tileKey(0.5, 1)).toThrow("RENDER_TILE");
  });
});

describe("tanks", () => {
  it("draws a placed tank of a known team inside the window, where it stands", async () => {
    const { view, fakes } = await scene();
    view.viewport({ left: 100, top: 50, tiles: [] });
    view.identity({ tankId: 2000, team: 2, name: "austin" });
    view.place({ tankId: 2000, x: 103, y: 55, facing: 4 }, 2);
    expect(view.paint().tanksDrawn).toBe(1);
    expect(lastTankDraw(fakes)).toEqual([3 * 24 - 2, 5 * 16 - 2]);
    view.walk({ tankId: 2000, x: 104, y: 55, facing: 4 });
    expect(view.paint().tanksDrawn).toBe(1);
    expect(lastTankDraw(fakes)).toEqual([4 * 24 - 2, 5 * 16 - 2]);
  });

  it("does not draw a tank whose team it has not heard, outside the window, or taken off", async () => {
    const { view } = await scene();
    view.viewport({ left: 100, top: 50, tiles: [] });
    view.walk({ tankId: 7, x: 101, y: 51, facing: 0 });
    view.entry(8, 1);
    view.place({ tankId: 9, x: 99, y: 51, facing: 0 }, 1);
    view.place({ tankId: 10, x: 101, y: 66, facing: 0 }, 1);
    expect(view.paint().tanksDrawn).toBe(0);
    view.place({ tankId: 11, x: 102, y: 52, facing: 0 }, 3);
    view.place({ tankId: 12, x: 103, y: 52, facing: 0 }, 3);
    expect(view.paint().tanksDrawn).toBe(2);
    view.remove(11);
    view.exit(12);
    expect(view.paint().tanksDrawn).toBe(0);
  });

  it("turns a destroyed tank into a corpse once, and back into a tank on its next placing", async () => {
    const { view, renderer } = await scene();
    view.viewport({ left: 100, top: 50, tiles: [] });
    view.place({ tankId: 5, x: 101, y: 51, facing: 0 }, 0);
    view.paint();
    view.destroyed(5);
    view.destroyed(5);
    expect(view.paint().tanksDrawn).toBe(1);
    expect(renderer.manifest.corpses).toHaveLength(2);
    view.place({ tankId: 5, x: 101, y: 52, facing: 0 }, 0);
    expect(view.paint().tanksDrawn).toBe(1);
  });

  it("redraws every tank it knows when the window moves", async () => {
    const { view } = await scene();
    view.viewport({ left: 100, top: 50, tiles: [] });
    view.place({ tankId: 1, x: 101, y: 51, facing: 0 }, 0);
    view.place({ tankId: 2, x: 120, y: 51, facing: 0 }, 1);
    expect(view.paint().tanksDrawn).toBe(1);
    view.viewport({ left: 110, top: 50, tiles: [] });
    expect(view.paint()).toEqual({ tilesDrawn: 256, tanksDrawn: 1 });
  });
});

describe("unrenderedCounts", () => {
  it("counts each subtype the view was not drawn from", async () => {
    const { view } = await scene();
    view.unrendered(0x3f);
    view.unrendered(0x3f);
    view.unrendered(0x53);
    expect([...view.unrenderedCounts]).toEqual([
      [0x3f, 2],
      [0x53, 1],
    ]);
  });
});
