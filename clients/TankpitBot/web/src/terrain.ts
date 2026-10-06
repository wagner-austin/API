/**
 * A viewport tile's terrain byte, read as the game's client reads it.
 *
 * Wiki rendering-pipeline, "Terrain Byte Encoding": bits 0-3 say which of
 * the four diagonal neighbours (NE, SE, SW, NW) share this tile's
 * terrain; bits 4-6 are the base terrain, 0 ground, 2 water, 4 obstacle;
 * bit 7 is a variant flag the renderer does not draw differently. Water
 * and obstacles draw a corner overlay toward each neighbour that does
 * not share them.
 */

import { RenderError } from "./scaled_context.js";

/** The base terrains a tile can be. */
export enum TerrainKind {
  Ground = "ground",
  Water = "water",
  Obstacle = "obstacle",
}

/** A tile's terrain, decoded. */
export interface Terrain {
  readonly kind: TerrainKind;
  readonly sharedCorners: number;
}

const CORNER_MASK = 0x0f;
const BASE_SHIFT = 4;
const BASE_MASK = 0x07;

/**
 * Decode a terrain byte.
 *
 * @param value - The byte, 0 to 255.
 * @returns Its kind and the mask of corners whose neighbour shares it.
 * @throws RenderError RENDER_TERRAIN for a value outside a byte or a base the game does not define.
 */
export function decodeTerrain(value: number): Terrain {
  if (!Number.isInteger(value) || value < 0 || value > 0xff) {
    throw new RenderError(`RENDER_TERRAIN: ${value} is not a terrain byte`);
  }
  const base = (value >> BASE_SHIFT) & BASE_MASK;
  const sharedCorners = value & CORNER_MASK;
  if (base === 0) {
    return { kind: TerrainKind.Ground, sharedCorners };
  }
  if (base === 2) {
    return { kind: TerrainKind.Water, sharedCorners };
  }
  if (base === 4) {
    return { kind: TerrainKind.Obstacle, sharedCorners };
  }
  throw new RenderError(`RENDER_TERRAIN: byte 0x${value.toString(16)} has base ${base}, which the game does not define`);
}

/** The corners, in bit order, whose neighbour does not share the tile's terrain. */
export function openCorners(terrain: Terrain): number[] {
  return [0, 1, 2, 3].filter((corner) => (terrain.sharedCorners & (1 << corner)) === 0);
}
