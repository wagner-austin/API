/**
 * The renderer's seams onto the browser.
 *
 * Two things the renderer needs cannot run in a test's DOM: a 2D canvas
 * context (jsdom has none) and an image that actually decodes. Each is a
 * hook bound to its real implementation here and called unconditionally;
 * a test binds a fake through setHooks and restores the real ones with
 * resetHooks.
 *
 * @internal Private to the renderer and its tests.
 */

import type { Surface } from "./scaled_context.js";

/** Read a canvas's 2D context, or null where the platform has none. */
export interface CanvasContextHook {
  (canvas: HTMLCanvasElement): Surface | null;
}

/** Load an image from a URL, resolving once it has decoded. */
export interface LoadImageHook {
  (url: string): Promise<CanvasImageSource>;
}

/** Anything that encodes itself as a data URL, as a canvas does. */
export interface DataUrlSource {
  toDataURL(type: string): string;
}

/** Encode a painted canvas as a URL an image can load. */
export interface CanvasUrlHook {
  (canvas: DataUrlSource): string;
}

/** Every hook. */
export interface Hooks {
  readonly canvasContext: CanvasContextHook;
  readonly loadImage: LoadImageHook;
  readonly canvasUrl: CanvasUrlHook;
}

function realCanvasContext(canvas: HTMLCanvasElement): Surface | null {
  return canvas.getContext("2d");
}

function realCanvasUrl(canvas: DataUrlSource): string {
  return canvas.toDataURL("image/png");
}

/**
 * Point an image at a URL and settle once it has decoded or failed.
 *
 * @param image - A fresh image element.
 * @param url - The sheet's URL.
 * @returns The image, once loaded.
 * @throws Error RENDER_IMAGE when the image fails to load.
 */
export function loadImageInto(image: HTMLImageElement, url: string): Promise<CanvasImageSource> {
  const loaded = new Promise<CanvasImageSource>((resolve, reject) => {
    image.onload = (): void => resolve(image);
    image.onerror = (): void => reject(new Error(`RENDER_IMAGE: ${url} did not load`));
  });
  image.src = url;
  return loaded;
}

function realLoadImage(url: string): Promise<CanvasImageSource> {
  return loadImageInto(new Image(), url);
}

/** The real implementations. */
export const realHooks: Hooks = { canvasContext: realCanvasContext, loadImage: realLoadImage, canvasUrl: realCanvasUrl };

let current: Hooks = realHooks;

/** The hooks in force. */
export function hooks(): Hooks {
  return current;
}

/** Bind hooks, for a test. */
export function setHooks(next: Hooks): void {
  current = next;
}

/** Restore the real hooks. */
export function resetHooks(): void {
  current = realHooks;
}
