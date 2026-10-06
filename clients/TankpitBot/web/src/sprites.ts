/**
 * Sheets and frames: turning a manifest's rectangles into pixels.
 *
 * A pack's sheets are loaded once (loadSheets) into a map from sheet id
 * to decoded image; drawFrame copies one manifest frame from it onto a
 * layer. A frame naming a sheet that was not loaded is an error, never a
 * blank: a missing picture is a broken pack, and it says so.
 */

import { hooks } from "./_test_hooks.js";
import type { SpriteFrame, SpriteManifest } from "./manifest.js";
import { RenderError, type ScaledContext } from "./scaled_context.js";

/** A pack's decoded sheets, by sheet id. */
export interface SheetImages {
  readonly images: ReadonlyMap<string, CanvasImageSource>;
}

/**
 * Load every sheet a manifest names.
 *
 * @param manifest - The pack's manifest.
 * @returns The decoded sheets.
 * @throws Error RENDER_IMAGE when a sheet does not load.
 */
export async function loadSheets(manifest: SpriteManifest): Promise<SheetImages> {
  const load = async (id: string, url: string): Promise<[string, CanvasImageSource]> => [id, await hooks().loadImage(url)];
  const pairs = await Promise.all([...manifest.sheets.entries()].map(([id, url]) => load(id, url)));
  return { images: new Map(pairs) };
}

function missing(sheet: string): never {
  throw new RenderError(`RENDER_SHEET: sheet ${JSON.stringify(sheet)} is not loaded`);
}

/**
 * One frame of a list, by index.
 *
 * @param frames - The list.
 * @param index - The frame wanted.
 * @param what - What the list is, for the message.
 * @returns The frame.
 * @throws RenderError RENDER_FRAME when the pack has no frame there.
 */
export function pickFrame(frames: readonly SpriteFrame[], index: number, what: string): SpriteFrame {
  const frame = frames[index];
  if (frame === undefined) {
    throw new RenderError(`RENDER_FRAME: the pack has no ${what} frame ${index} (it has ${frames.length})`);
  }
  return frame;
}

/**
 * Copy one frame onto a layer at a point, in game pixels.
 *
 * @param context - The layer's context.
 * @param sheets - The pack's decoded sheets.
 * @param frame - The frame.
 * @param x - Destination x.
 * @param y - Destination y.
 * @throws RenderError RENDER_SHEET when the frame's sheet is not loaded.
 */
export function drawFrame(context: ScaledContext, sheets: SheetImages, frame: SpriteFrame, x: number, y: number): void {
  const image = sheets.images.get(frame.sheet) ?? missing(frame.sheet);
  context.draw(image, frame, x, y);
}
