/**
 * The region a moving picture last covered, so only that is erased.
 *
 * The game's client gives each animated object a rectangle (its Hc
 * class, wiki rendering-pipeline, "Dirty-Rect System"): drawing grows it
 * to the union of everything drawn since the last erase, erasing clears
 * exactly that union and empties it. A layer erased this way never
 * clears pixels another object owns outside the union.
 */

import type { ScaledContext } from "./scaled_context.js";

/** A rectangle in game pixels. */
export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

/** One object's dirty region. */
export class DirtyRect {
  private region: Rect | null = null;

  /** The region to erase, or null when nothing has been drawn since the last erase. */
  public get current(): Rect | null {
    return this.region;
  }

  /**
   * Grow the region to cover a newly drawn rectangle.
   *
   * @param drawn - What was drawn, in game pixels.
   */
  public include(drawn: Rect): void {
    const was = this.region;
    if (was === null) {
      this.region = drawn;
      return;
    }
    const left = Math.min(was.x, drawn.x);
    const top = Math.min(was.y, drawn.y);
    const right = Math.max(was.x + was.w, drawn.x + drawn.w);
    const bottom = Math.max(was.y + was.h, drawn.y + drawn.h);
    this.region = { x: left, y: top, w: right - left, h: bottom - top };
  }

  /**
   * Clear the region on a layer and empty it.
   *
   * @param context - The layer the object draws on.
   * @returns What was cleared, or null when there was nothing to clear.
   */
  public erase(context: ScaledContext): Rect | null {
    const was = this.region;
    if (was !== null) {
      context.clear(was.x, was.y, was.w, was.h);
      this.region = null;
    }
    return was;
  }
}

/** Whether two rectangles share any pixel. */
export function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}
