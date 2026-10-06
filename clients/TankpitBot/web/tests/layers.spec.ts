import { afterEach, describe, expect, it } from "vitest";

import { resetHooks } from "../src/_test_hooks.js";
import { createLayerStack, GAME_HEIGHT, GAME_WIDTH, LayerName, MENU_HEIGHT } from "../src/layers.js";
import { installFakes } from "./fakes.js";

afterEach(resetHooks);

describe("createLayerStack", () => {
  it("stacks six canvases bottom to top, the menu strip under the game area", () => {
    const fakes = installFakes();
    const container = document.createElement("div");
    const stack = createLayerStack(container, 2);
    const canvases = [...container.querySelectorAll("canvas")];
    expect(canvases.map((canvas) => [canvas.dataset["layer"], canvas.style.zIndex, canvas.style.top, canvas.width, canvas.height])).toEqual([
      ["Background", "0", "0px", 768, 512],
      ["Tanks", "1", "0px", 768, 512],
      ["Action", "2", "0px", 768, 512],
      ["Map", "3", "0px", 768, 512],
      ["Overlay", "4", "0px", 768, 512],
      ["Menu", "5", "256px", 768, 96],
    ]);
    expect(container.style.width).toBe(`${GAME_WIDTH}px`);
    expect(container.style.height).toBe(`${GAME_HEIGHT + MENU_HEIGHT}px`);
    expect(stack.layers[LayerName.Menu].canvas).toBe(canvases[5]);
    expect(stack.layers[LayerName.Tanks].context.scale).toBe(2);
    expect(fakes.surfaces).toHaveLength(6);
  });

  it("sizes canvases up to whole device pixels at a fractional scale", () => {
    installFakes();
    const container = document.createElement("div");
    const stack = createLayerStack(container, 1.1);
    expect([stack.layers[LayerName.Background].canvas.width, stack.layers[LayerName.Menu].canvas.height]).toEqual([423, 53]);
  });

  it("refuses a platform whose canvas has no 2D context, before adding the canvas", () => {
    installFakes({ noContext: true });
    const container = document.createElement("div");
    expect(() => createLayerStack(container, 1)).toThrow("RENDER_NO_CANVAS: the Background layer's canvas has no 2D context");
    expect(container.children).toHaveLength(0);
  });
});
