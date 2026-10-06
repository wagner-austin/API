import { afterEach, describe, expect, it } from "vitest";

import { resetHooks } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { LayerName } from "../src/layers.js";
import { Renderer, TOOLBAR_BUTTON, TOOLBAR_FACE, TOOLBAR_PRESSED } from "../src/renderer.js";
import { decodeTerrain } from "../src/terrain.js";
import { NO_MINE, RockKind } from "../src/tile_grid.js";
import { TOOLBAR_REGIONS, ToolbarAction } from "../src/toolbar.js";
import { installFakes, type RecordingSurface } from "./fakes.js";

afterEach(resetHooks);

const tank = { team: 0, facing: 0, alive: true, deaths: 0, column: 2, row: 2 };

async function built(): Promise<{ renderer: Renderer; surfaces: RecordingSurface[] }> {
  const fakes = installFakes();
  const renderer = await Renderer.create(document.createElement("div"), 1, defaultManifest("sheet.png"));
  return { renderer, surfaces: fakes.surfaces };
}

function layer(surfaces: RecordingSurface[], name: LayerName): RecordingSurface {
  const order = [LayerName.Background, LayerName.Tanks, LayerName.Action, LayerName.Map, LayerName.Overlay, LayerName.Menu];
  const surface = surfaces[order.indexOf(name)];
  if (surface === undefined) {
    throw new Error(`no ${name} surface`);
  }
  return surface;
}

describe("Renderer.create", () => {
  it("loads the pack's sheet and draws the toolbar face and every button raised", async () => {
    const fakes = installFakes();
    await Renderer.create(document.createElement("div"), 1, defaultManifest("sheet.png"));
    expect(fakes.loaded).toEqual(["sheet.png"]);
    const fills = layer(fakes.surfaces, LayerName.Menu).only("fill");
    expect(fills[0]).toEqual({ op: "fill", image: null, fill: TOOLBAR_FACE, source: [], dest: [0, 0, 384, 48] });
    expect(fills.slice(1).map((call) => call.fill)).toEqual(Array.from({ length: 18 }, () => TOOLBAR_BUTTON));
    expect(fills[1]?.dest).toEqual([11, 3, 41, 42]);
  });
});

describe("renderFrame", () => {
  it("draws all viewport tiles first, then only the tiles set since", async () => {
    const { renderer } = await built();
    expect(renderer.renderFrame()).toEqual({ tilesDrawn: 256, tanksDrawn: 0 });
    renderer.setTile(1, 1, { terrain: decodeTerrain(0x2f), cache: 0, mine: NO_MINE, rock: RockKind.None });
    expect(renderer.renderFrame()).toEqual({ tilesDrawn: 1, tanksDrawn: 0 });
  });

  it("draws a placed tank once and again only when it moves, erasing its old place", async () => {
    const { renderer, surfaces } = await built();
    renderer.setTank(7, tank);
    expect(renderer.renderFrame().tanksDrawn).toBe(1);
    expect(renderer.renderFrame().tanksDrawn).toBe(0);
    const tanks = layer(surfaces, LayerName.Tanks);
    renderer.setTank(7, { ...tank, column: 3 });
    expect(renderer.renderFrame().tanksDrawn).toBe(1);
    expect(tanks.calls.map((call) => [call.op, ...call.dest])).toEqual([
      ["draw", 46, 30, 28, 20],
      ["clear", 46, 30, 28, 20],
      ["draw", 70, 30, 28, 20],
    ]);
  });

  it("redraws a still tank whose pixels a moving tank's erase cleared, and only that one", async () => {
    const { renderer, surfaces } = await built();
    renderer.setTank(1, tank);
    renderer.setTank(2, { ...tank, column: 3 });
    renderer.setTank(3, { ...tank, column: 10 });
    renderer.renderFrame();
    const tanks = layer(surfaces, LayerName.Tanks);
    tanks.calls.length = 0;
    renderer.setTank(1, { ...tank, row: 8 });
    expect(renderer.renderFrame().tanksDrawn).toBe(2);
    expect(tanks.calls.map((call) => [call.op, ...call.dest])).toEqual([
      ["clear", 46, 30, 28, 20],
      ["clear", 70, 30, 28, 20],
      ["draw", 46, 126, 28, 20],
      ["draw", 70, 30, 28, 20],
    ]);
  });

  it("follows an erase through a chain of overlapping tanks", async () => {
    const { renderer, surfaces } = await built();
    renderer.setTank(1, tank);
    renderer.setTank(2, { ...tank, column: 3 });
    renderer.setTank(3, { ...tank, column: 4 });
    renderer.renderFrame();
    const tanks = layer(surfaces, LayerName.Tanks);
    tanks.calls.length = 0;
    renderer.setTank(3, { ...tank, column: 4, facing: 1 });
    expect(renderer.renderFrame().tanksDrawn).toBe(3);
    expect(tanks.only("clear")).toHaveLength(3);
  });

  it("refuses at once a tank the pack cannot draw", async () => {
    const { renderer } = await built();
    expect(() => renderer.setTank(1, { ...tank, team: 9 })).toThrow("RENDER_FRAME: the pack has no tanks for team 9");
  });
});

describe("removeTank", () => {
  it("erases the tank now and redraws what it overlapped on the next frame", async () => {
    const { renderer, surfaces } = await built();
    renderer.setTank(1, tank);
    renderer.setTank(2, { ...tank, column: 3 });
    renderer.renderFrame();
    const tanks = layer(surfaces, LayerName.Tanks);
    tanks.calls.length = 0;
    expect(renderer.removeTank(1)).toBe(true);
    expect(renderer.renderFrame().tanksDrawn).toBe(1);
    expect(tanks.calls.map((call) => [call.op, ...call.dest])).toEqual([
      ["clear", 46, 30, 28, 20],
      ["clear", 70, 30, 28, 20],
      ["draw", 70, 30, 28, 20],
    ]);
  });

  it("erases nothing for a tank placed but never drawn, and refuses an unknown id", async () => {
    const { renderer, surfaces } = await built();
    renderer.setTank(1, tank);
    expect(renderer.removeTank(1)).toBe(true);
    expect(renderer.removeTank(1)).toBe(false);
    expect(renderer.renderFrame().tanksDrawn).toBe(0);
    expect(layer(surfaces, LayerName.Tanks).calls).toEqual([]);
  });
});

describe("clickToolbar", () => {
  it("presses the clicked button, releases the last one, and releases on a miss", async () => {
    const { renderer, surfaces } = await built();
    const menu = layer(surfaces, LayerName.Menu);
    menu.calls.length = 0;
    expect(renderer.clickToolbar(53, 0)).toBe(ToolbarAction.Radar);
    expect(renderer.pressedButton).toBe(ToolbarAction.Radar);
    expect(renderer.clickToolbar(233, 10)).toBe(ToolbarAction.ArmorShield);
    expect(renderer.clickToolbar(5, 10)).toBe(-1);
    expect(renderer.pressedButton).toBe(-1);
    const radar = TOOLBAR_REGIONS[ToolbarAction.Radar];
    expect(menu.calls.map((call) => [call.fill, call.dest[0]])).toEqual([
      [TOOLBAR_PRESSED, (radar?.x ?? 0) + 1],
      [TOOLBAR_BUTTON, 54],
      [TOOLBAR_PRESSED, 234],
      [TOOLBAR_BUTTON, 234],
    ]);
  });
});
