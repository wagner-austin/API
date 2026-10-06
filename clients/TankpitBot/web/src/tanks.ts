/**
 * Tanks: one frame per team and facing, and a corpse in two variants.
 *
 * The game's client cuts its tank sheet into four team columns of 28x20
 * frames, sixteen facings each, with two corpse rows below (wiki
 * rendering-pipeline, "Tank Sprites"); a manifest names the same frames
 * in whatever sheet a pack provides. A tank's frame is centred on its
 * tile, which is narrower and shorter than the frame.
 */

import type { Rect } from "./dirty_rect.js";
import type { SpriteFrame, SpriteManifest } from "./manifest.js";
import { RenderError, type ScaledContext } from "./scaled_context.js";
import { drawFrame, pickFrame, type SheetImages } from "./sprites.js";

/** A tank as the renderer draws it. */
export interface TankView {
  readonly team: number;
  readonly facing: number;
  readonly alive: boolean;
  readonly deaths: number;
  readonly column: number;
  readonly row: number;
}

/**
 * The frame a tank draws with.
 *
 * @param manifest - The pack's manifest.
 * @param tank - The tank.
 * @returns Its team's facing frame while alive, else the corpse its death count's parity picks.
 * @throws RenderError RENDER_FRAME for a team or facing the pack has no frame for.
 */
export function tankFrame(manifest: SpriteManifest, tank: TankView): SpriteFrame {
  if (!tank.alive) {
    return pickFrame(manifest.corpses, tank.deaths % 2, "corpse");
  }
  const facings = manifest.tanks[tank.team];
  if (facings === undefined) {
    throw new RenderError(`RENDER_FRAME: the pack has no tanks for team ${tank.team}`);
  }
  return pickFrame(facings, tank.facing, `team ${tank.team} facing`);
}

/**
 * The rectangle a tank's frame covers, centred on its viewport tile.
 *
 * @param manifest - The pack's manifest.
 * @param tank - The tank, its tile in viewport columns and rows from 0.
 * @returns The rectangle, in game pixels.
 */
export function tankRect(manifest: SpriteManifest, tank: TankView): Rect {
  const frame = tankFrame(manifest, tank);
  return {
    x: tank.column * manifest.tileWidth + (manifest.tileWidth - frame.w) / 2,
    y: tank.row * manifest.tileHeight + (manifest.tileHeight - frame.h) / 2,
    w: frame.w,
    h: frame.h,
  };
}

/**
 * Draw a tank centred on its viewport tile.
 *
 * @param context - The tanks layer.
 * @param manifest - The pack's manifest.
 * @param sheets - The pack's sheets.
 * @param tank - The tank.
 * @returns The rectangle drawn, for the tank's dirty rect.
 */
export function drawTank(context: ScaledContext, manifest: SpriteManifest, sheets: SheetImages, tank: TankView): Rect {
  const rect = tankRect(manifest, tank);
  drawFrame(context, sheets, tankFrame(manifest, tank), rect.x, rect.y);
  return rect;
}
