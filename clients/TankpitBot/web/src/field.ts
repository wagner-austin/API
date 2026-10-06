/**
 * A field's static terrain, as the server serves it, and each tile's terrain byte.
 *
 * The server's /terrain/<room> answers 256 x 256 class bytes, row by row
 * from the north-west corner: 0 ground, 1 rock, 2 water (the sim's
 * net_web.py, read off the TerrainMap the sim plays on). The renderer
 * draws a tile from a terrain byte (terrain.ts): its base in bits 4 to 6
 * and, in bits 0 to 3, which diagonal neighbours (NE, SE, SW, NW) share
 * it, so a corner overlay is drawn toward each that does not. A
 * neighbour off the field counts as sharing, so the field's edge draws no
 * overlay.
 */

import { byteAt, WireError } from "./wire.js";

/** Tiles along each side of a field. */
export const FIELD_SPAN = 256;

/** The terrain classes the server serves. */
export enum FieldClass {
  Ground = 0,
  Rock = 1,
  Water = 2,
}

const BASE_BITS = new Map<number, number>([
  [FieldClass.Ground, 0x00],
  [FieldClass.Rock, 0x40],
  [FieldClass.Water, 0x20],
]);

/** The diagonal neighbours, in corner-bit order: NE, SE, SW, NW. */
const CORNERS: readonly (readonly [number, number])[] = [
  [1, -1],
  [1, 1],
  [-1, 1],
  [-1, -1],
];

/** A field's terrain classes. */
export class FieldTerrain {
  private constructor(private readonly classes: Uint8Array) {}

  /**
   * Read the server's terrain response.
   *
   * @param bytes - The response body.
   * @returns The field.
   * @throws WireError FIELD_SIZE for a body that is not one byte per tile, FIELD_CLASS for a byte that is no class.
   */
  public static decode(bytes: Uint8Array): FieldTerrain {
    if (bytes.length !== FIELD_SPAN * FIELD_SPAN) {
      throw new WireError(`FIELD_SIZE: a field is ${FIELD_SPAN * FIELD_SPAN} bytes, not ${bytes.length}`);
    }
    const bad = bytes.findIndex((value) => !BASE_BITS.has(value));
    if (bad >= 0) {
      throw new WireError(`FIELD_CLASS: tile (${bad % FIELD_SPAN}, ${Math.floor(bad / FIELD_SPAN)}) is class ${byteAt(bytes, bad)}`);
    }
    return new FieldTerrain(bytes.slice());
  }

  /**
   * A tile's class, or null off the field.
   *
   * @param x - The tile's x.
   * @param y - The tile's y.
   * @returns Its class.
   */
  public classAt(x: number, y: number): number | null {
    if (x < 0 || y < 0 || x >= FIELD_SPAN || y >= FIELD_SPAN) {
      return null;
    }
    return byteAt(this.classes, y * FIELD_SPAN + x);
  }

  /**
   * A tile's terrain byte, its base and the diagonal neighbours sharing it.
   *
   * @param x - The tile's x, on the field.
   * @param y - The tile's y, on the field.
   * @returns The byte terrain.ts decodes.
   * @throws WireError FIELD_TILE for a tile off the field.
   */
  public terrainByte(x: number, y: number): number {
    const own = this.classAt(x, y);
    const base = own === null ? undefined : BASE_BITS.get(own);
    if (base === undefined) {
      throw new WireError(`FIELD_TILE: (${x}, ${y}) is not on the field`);
    }
    let shared = 0;
    CORNERS.forEach(([dx, dy], bit) => {
      const neighbour = this.classAt(x + dx, y + dy);
      if (neighbour === null || neighbour === own) {
        shared |= 1 << bit;
      }
    });
    return base | shared;
  }
}
