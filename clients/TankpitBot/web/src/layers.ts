/**
 * The six canvas layers the game is drawn on.
 *
 * The game's client stacks six canvases, composited by CSS z-index (wiki
 * rendering-pipeline, "Canvas Layer Stack"): terrain at the bottom,
 * tanks, the action layer for shots and sweeps, the minimap, the
 * overlay, and the toolbar strip below the 384x256 game area. Each layer
 * here is a canvas sized in device pixels and a ScaledContext that draws
 * on it in game pixels.
 */

import { hooks } from "./_test_hooks.js";
import { RenderError, ScaledContext } from "./scaled_context.js";

export const GAME_WIDTH = 384;
export const GAME_HEIGHT = 256;
export const MENU_HEIGHT = 48;

/** The layers, bottom to top. */
export enum LayerName {
  Background = "Background",
  Tanks = "Tanks",
  Action = "Action",
  Map = "Map",
  Overlay = "Overlay",
  Menu = "Menu",
}

/** Where a layer sits. */
export interface LayerSpec {
  readonly name: LayerName;
  readonly z: number;
  readonly top: number;
  readonly height: number;
}

export const LAYER_SPECS: Readonly<Record<LayerName, LayerSpec>> = {
  [LayerName.Background]: { name: LayerName.Background, z: 0, top: 0, height: GAME_HEIGHT },
  [LayerName.Tanks]: { name: LayerName.Tanks, z: 1, top: 0, height: GAME_HEIGHT },
  [LayerName.Action]: { name: LayerName.Action, z: 2, top: 0, height: GAME_HEIGHT },
  [LayerName.Map]: { name: LayerName.Map, z: 3, top: 0, height: GAME_HEIGHT },
  [LayerName.Overlay]: { name: LayerName.Overlay, z: 4, top: 0, height: GAME_HEIGHT },
  [LayerName.Menu]: { name: LayerName.Menu, z: 5, top: GAME_HEIGHT, height: MENU_HEIGHT },
};

/** One layer: its canvas and the context that draws on it. */
export interface Layer {
  readonly spec: LayerSpec;
  readonly canvas: HTMLCanvasElement;
  readonly context: ScaledContext;
}

/** The whole stack. */
export interface LayerStack {
  readonly scale: number;
  readonly layers: Readonly<Record<LayerName, Layer>>;
}

function createLayer(container: HTMLElement, spec: LayerSpec, scale: number): Layer {
  const canvas = container.ownerDocument.createElement("canvas");
  canvas.width = Math.ceil(GAME_WIDTH * scale);
  canvas.height = Math.ceil(spec.height * scale);
  canvas.dataset["layer"] = spec.name;
  canvas.style.position = "absolute";
  canvas.style.left = "0px";
  canvas.style.top = `${spec.top}px`;
  canvas.style.width = `${GAME_WIDTH}px`;
  canvas.style.height = `${spec.height}px`;
  canvas.style.zIndex = String(spec.z);
  const surface = hooks().canvasContext(canvas);
  if (surface === null) {
    throw new RenderError(`RENDER_NO_CANVAS: the ${spec.name} layer's canvas has no 2D context`);
  }
  const context = new ScaledContext(surface, scale);
  container.appendChild(canvas);
  return { spec, canvas, context };
}

/**
 * Create the six canvases inside a container, positioned and stacked.
 *
 * @param container - The element the game is drawn in.
 * @param scale - Device pixels per game pixel.
 * @returns The stack.
 * @throws RenderError RENDER_NO_CANVAS when the platform gives a canvas no 2D context, RENDER_SCALE for a bad scale.
 */
export function createLayerStack(container: HTMLElement, scale: number): LayerStack {
  container.style.position = "relative";
  container.style.width = `${GAME_WIDTH}px`;
  container.style.height = `${GAME_HEIGHT + MENU_HEIGHT}px`;
  const make = (name: LayerName): Layer => createLayer(container, LAYER_SPECS[name], scale);
  return {
    scale,
    layers: {
      [LayerName.Background]: make(LayerName.Background),
      [LayerName.Tanks]: make(LayerName.Tanks),
      [LayerName.Action]: make(LayerName.Action),
      [LayerName.Map]: make(LayerName.Map),
      [LayerName.Overlay]: make(LayerName.Overlay),
      [LayerName.Menu]: make(LayerName.Menu),
    },
  };
}
