import { afterEach, describe, expect, it } from "vitest";

import { resetHooks } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { ScaledContext } from "../src/scaled_context.js";
import { drawFrame, loadSheets, pickFrame } from "../src/sprites.js";
import { installFakes, RecordingSurface } from "./fakes.js";

afterEach(resetHooks);

describe("loadSheets", () => {
  it("loads every sheet the manifest names, keyed by sheet id", async () => {
    const fakes = installFakes();
    const manifest = { ...defaultManifest("a.png"), sheets: new Map([["default", "a.png"], ["extra", "b.png"]]) };
    const sheets = await loadSheets(manifest);
    expect(fakes.loaded).toEqual(["a.png", "b.png"]);
    expect([...sheets.images.keys()]).toEqual(["default", "extra"]);
    expect(sheets.images.get("extra")).toBe(fakes.images.get("b.png"));
  });

  it("fails when any sheet fails to load", async () => {
    installFakes({ failUrl: "a.png" });
    await expect(loadSheets(defaultManifest("a.png"))).rejects.toThrow("RENDER_IMAGE: a.png did not load");
  });
});

describe("drawFrame", () => {
  it("copies the frame's rectangle from its sheet", () => {
    const surface = new RecordingSurface();
    const image = document.createElement("canvas");
    drawFrame(new ScaledContext(surface, 1), { images: new Map([["s", image]]) }, { sheet: "s", x: 1, y: 2, w: 3, h: 4 }, 5, 6);
    expect(surface.calls).toEqual([{ op: "draw", image, fill: "", source: [1, 2, 3, 4], dest: [5, 6, 3, 4] }]);
  });

  it("refuses a frame whose sheet was not loaded", () => {
    const context = new ScaledContext(new RecordingSurface(), 1);
    expect(() => drawFrame(context, { images: new Map() }, { sheet: "s", x: 0, y: 0, w: 1, h: 1 }, 0, 0)).toThrow('RENDER_SHEET: sheet "s" is not loaded');
  });
});

describe("pickFrame", () => {
  it("returns the frame at an index and names a missing one", () => {
    const frame = { sheet: "s", x: 0, y: 0, w: 1, h: 1 };
    expect(pickFrame([frame], 0, "rock")).toBe(frame);
    expect(() => pickFrame([frame], 1, "rock")).toThrow("RENDER_FRAME: the pack has no rock frame 1 (it has 1)");
  });
});
