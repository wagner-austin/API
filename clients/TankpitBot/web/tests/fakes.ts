/**
 * Real stand-ins for the two things jsdom cannot do: draw and decode.
 *
 * RecordingSurface implements the renderer's Surface interface and keeps
 * every call in order, so a test reads exactly what was drawn where.
 * installFakes binds the hooks to it and to an image loader that hands
 * back a distinct canvas per URL; resetHooks undoes it.
 */

import { setHooks } from "../src/_test_hooks.js";
import type { Surface } from "../src/scaled_context.js";

/** One recorded drawing call. */
export interface DrawCall {
  readonly op: "draw" | "clear" | "fill";
  readonly image: CanvasImageSource | null;
  readonly fill: string;
  readonly source: readonly number[];
  readonly dest: readonly number[];
}

/** A Surface that records instead of rasterising. */
export class RecordingSurface implements Surface {
  public fillStyle: string | CanvasGradient | CanvasPattern = "#000000";
  public imageSmoothingEnabled = true;
  public readonly calls: DrawCall[] = [];

  public drawImage(image: CanvasImageSource, sx: number, sy: number, sw: number, sh: number, dx: number, dy: number, dw: number, dh: number): void {
    this.calls.push({ op: "draw", image, fill: "", source: [sx, sy, sw, sh], dest: [dx, dy, dw, dh] });
  }

  public clearRect(x: number, y: number, w: number, h: number): void {
    this.calls.push({ op: "clear", image: null, fill: "", source: [], dest: [x, y, w, h] });
  }

  public fillRect(x: number, y: number, w: number, h: number): void {
    this.calls.push({ op: "fill", image: null, fill: String(this.fillStyle), source: [], dest: [x, y, w, h] });
  }

  /** The calls of one kind. */
  public only(op: DrawCall["op"]): DrawCall[] {
    return this.calls.filter((call) => call.op === op);
  }
}

/** What installFakes bound. */
export interface Fakes {
  readonly surfaces: RecordingSurface[];
  readonly images: Map<string, HTMLCanvasElement>;
  readonly loaded: string[];
}

/**
 * Bind every hook to a recording fake.
 *
 * @param options - noContext makes every canvas report no 2D context; failUrl makes that URL fail to load.
 * @returns The surfaces handed out, in order, and the images by URL.
 */
export function installFakes(options: { readonly noContext?: boolean; readonly failUrl?: string } = {}): Fakes {
  const fakes: Fakes = { surfaces: [], images: new Map(), loaded: [] };
  setHooks({
    canvasContext: () => {
      if (options.noContext === true) {
        return null;
      }
      const surface = new RecordingSurface();
      fakes.surfaces.push(surface);
      return surface;
    },
    loadImage: (url) => {
      if (url === options.failUrl) {
        return Promise.reject(new Error(`RENDER_IMAGE: ${url} did not load`));
      }
      fakes.loaded.push(url);
      const image = document.createElement("canvas");
      fakes.images.set(url, image);
      return Promise.resolve(image);
    },
    canvasUrl: () => "data:image/png;base64,ZmFrZQ==",
  });
  return fakes;
}
