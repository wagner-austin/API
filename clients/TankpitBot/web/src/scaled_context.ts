/**
 * Drawing in game pixels on a canvas of any density.
 *
 * Every layer is laid out in the game's own pixels (384 wide), and its
 * canvas holds that many times the display scale. The game's client
 * does the same with its ScaledContext (wiki rendering-pipeline, "DPI
 * Scaling"): coordinates are multiplied by the scale, and two draw modes
 * exist. A scale that is a multiple of 25% lands every pixel exactly; any
 * other scale rounds the destination up and draws 3% oversize, so
 * neighbouring tiles overlap rather than leave hairline gaps.
 */

/** The part of a 2D canvas context the renderer draws with. */
export interface Surface {
  fillStyle: string | CanvasGradient | CanvasPattern;
  imageSmoothingEnabled: boolean;
  drawImage(
    image: CanvasImageSource,
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    dx: number,
    dy: number,
    dw: number,
    dh: number,
  ): void;
  clearRect(x: number, y: number, w: number, h: number): void;
  fillRect(x: number, y: number, w: number, h: number): void;
}

/** A rectangle of a source image, in that image's pixels. */
export interface SourceRect {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

const OVERSIZE = 1.03;

/** A scale is exact when it is a whole number of quarters. */
export function isExactScale(scale: number): boolean {
  return Number.isInteger(scale * 4);
}

/** A rendering error, with a RENDER_* code in its message. */
export class RenderError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "RenderError";
  }
}

/** A surface drawn on in game pixels. */
export class ScaledContext {
  public readonly scale: number;
  public readonly exact: boolean;
  private readonly surface: Surface;

  /**
   * Bind a surface to a display scale, with smoothing off so sprite edges stay sharp.
   *
   * @param surface - The canvas context, sized in device pixels.
   * @param scale - Device pixels per game pixel.
   * @throws RenderError RENDER_SCALE for a scale that is not positive and finite.
   */
  public constructor(surface: Surface, scale: number) {
    if (!Number.isFinite(scale) || scale <= 0) {
      throw new RenderError(`RENDER_SCALE: scale ${scale} is not a positive number`);
    }
    surface.imageSmoothingEnabled = false;
    this.surface = surface;
    this.scale = scale;
    this.exact = isExactScale(scale);
  }

  /**
   * Copy a rectangle of an image to a point, at its own size in game pixels.
   *
   * @param image - The source image.
   * @param source - The rectangle to copy.
   * @param x - Destination x, in game pixels.
   * @param y - Destination y, in game pixels.
   */
  public draw(image: CanvasImageSource, source: SourceRect, x: number, y: number): void {
    const s = this.scale;
    if (this.exact) {
      this.surface.drawImage(image, source.x, source.y, source.w, source.h, x * s, y * s, source.w * s, source.h * s);
      return;
    }
    this.surface.drawImage(
      image,
      source.x,
      source.y,
      source.w,
      source.h,
      Math.floor(x * s),
      Math.floor(y * s),
      Math.ceil(source.w * s * OVERSIZE),
      Math.ceil(source.h * s * OVERSIZE),
    );
  }

  /** Clear a rectangle, in game pixels. */
  public clear(x: number, y: number, w: number, h: number): void {
    const s = this.scale;
    this.surface.clearRect(x * s, y * s, w * s, h * s);
  }

  /** Fill a rectangle with a colour, in game pixels. */
  public fill(colour: string, x: number, y: number, w: number, h: number): void {
    const s = this.scale;
    this.surface.fillStyle = colour;
    this.surface.fillRect(x * s, y * s, w * s, h * s);
  }
}
