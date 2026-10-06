import { afterEach, describe, expect, it } from "vitest";

import { resetHooks } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { encodeManifest } from "../src/manifest.js";
import { bootPreview, buildDefaultPack, layPreview, startPreview } from "../src/preview.js";
import { Renderer } from "../src/renderer.js";
import { TerrainKind } from "../src/terrain.js";
import { RockKind } from "../src/tile_grid.js";
import { ToolbarAction } from "../src/toolbar.js";
import { installFakes } from "./fakes.js";

afterEach(() => {
  resetHooks();
  document.body.innerHTML = "";
});

function noFetch(url: string): Promise<unknown> {
  return Promise.reject(new Error(`fetched ${url}`));
}

describe("layPreview", () => {
  it("lays every terrain corner mask, every item and every tank frame", async () => {
    installFakes();
    const renderer = await Renderer.create(document.createElement("div"), 1, defaultManifest("s.png"));
    expect(layPreview(renderer)).toBe(66);
    expect(renderer.grid.get(1, 1).terrain).toEqual({ kind: TerrainKind.Water, sharedCorners: 0 });
    expect(renderer.grid.get(16, 2).terrain).toEqual({ kind: TerrainKind.Obstacle, sharedCorners: 15 });
    expect([1, 2, 3, 4, 5].map((column) => [renderer.grid.get(column, 3).cache, renderer.grid.get(column, 3).rock])).toEqual([
      [1, RockKind.None],
      [-1, RockKind.None],
      [0, RockKind.A],
      [0, RockKind.B],
      [0, RockKind.Ferry],
    ]);
    expect([6, 7, 8, 9].map((column) => renderer.grid.get(column, 3).mine)).toEqual([0, 1, 2, 3]);
    expect(renderer.renderFrame()).toEqual({ tilesDrawn: 256, tanksDrawn: 66 });
  });
});

describe("buildDefaultPack", () => {
  it("paints the sheet and names it by the painted canvas's URL", () => {
    const fakes = installFakes();
    const manifest = buildDefaultPack(document);
    expect(manifest.sheets.get("default")).toBe("data:image/png;base64,ZmFrZQ==");
    expect(fakes.surfaces[0]?.only("fill").length).toBeGreaterThan(64);
  });

  it("refuses a platform whose canvas has no 2D context", () => {
    installFakes({ noContext: true });
    expect(() => buildDefaultPack(document)).toThrow("RENDER_NO_CANVAS: the default pack's sheet canvas has no 2D context");
  });
});

describe("bootPreview", () => {
  it("renders the default pack with no pack URL, and presses a toolbar button on a click below the game", async () => {
    installFakes();
    const container = document.createElement("div");
    const renderer = await bootPreview(container, 1, null, noFetch);
    expect(container.querySelectorAll("canvas")).toHaveLength(6);
    container.dispatchEvent(new MouseEvent("click", { clientX: 60, clientY: 256 + 10 }));
    expect(renderer.pressedButton).toBe(ToolbarAction.Radar);
    container.dispatchEvent(new MouseEvent("click", { clientX: 60, clientY: 100 }));
    expect(renderer.pressedButton).toBe(ToolbarAction.Radar);
  });

  it("decodes and renders a pack fetched from a URL", async () => {
    const fakes = installFakes();
    const fetched: string[] = [];
    const renderer = await bootPreview(document.createElement("div"), 1, "pack.json", (url) => {
      fetched.push(url);
      return Promise.resolve(JSON.parse(JSON.stringify(encodeManifest(defaultManifest("other.png")))));
    });
    expect(fetched).toEqual(["pack.json"]);
    expect(fakes.loaded).toEqual(["other.png"]);
    expect(renderer.manifest.sheets.get("default")).toBe("other.png");
  });

  it("fails on a pack that does not decode", async () => {
    installFakes();
    await expect(bootPreview(document.createElement("div"), 1, "pack.json", () => Promise.resolve({}))).rejects.toThrow("MANIFEST_SHAPE");
  });
});

describe("startPreview", () => {
  it("draws into #game with the page's pixel ratio and its ?pack= manifest", async () => {
    installFakes();
    document.body.innerHTML = '<div id="game"></div>';
    const body = JSON.stringify(encodeManifest(defaultManifest("p.png")));
    const page = {
      document,
      location: { search: "?pack=packs/p.json" },
      devicePixelRatio: 2,
      fetch: (url: string): Promise<Response> => Promise.resolve(new Response(url === "packs/p.json" ? body : "", { status: 200 })),
    };
    const renderer = await startPreview(page);
    expect(renderer.stack.scale).toBe(2);
    expect(renderer.manifest.sheets.get("default")).toBe("p.png");
  });

  it("fails when the pack URL does not answer 200", async () => {
    installFakes();
    document.body.innerHTML = '<div id="game"></div>';
    const page = { document, location: { search: "?pack=x.json" }, devicePixelRatio: 1, fetch: (): Promise<Response> => Promise.resolve(new Response("", { status: 404 })) };
    await expect(startPreview(page)).rejects.toThrow("RENDER_PACK: x.json answered 404");
  });

  it("uses the default pack without ?pack=, and refuses a page with no #game", async () => {
    installFakes();
    document.body.innerHTML = '<div id="game"></div>';
    const page = { document, location: { search: "" }, devicePixelRatio: 1, fetch: (): Promise<Response> => Promise.reject(new Error("no fetch")) };
    const renderer = await startPreview(page);
    expect(renderer.manifest.name).toBe("default");
    document.body.innerHTML = "";
    await expect(startPreview(page)).rejects.toThrow("RENDER_NO_GAME: the page has no #game element");
  });
});
