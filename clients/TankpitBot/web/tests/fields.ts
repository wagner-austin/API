/** Fields built tile by tile, as the server's terrain response would carry them. */

import { FIELD_SPAN, FieldTerrain, type FieldClass } from "../src/field.js";

/**
 * A field of ground with the given tiles set to other classes.
 *
 * @param tiles - Each tile's x, y and class.
 * @returns The decoded field.
 */
export function fieldOf(tiles: readonly (readonly [number, number, FieldClass])[]): FieldTerrain {
  return FieldTerrain.decode(fieldBytes(tiles));
}

/**
 * The terrain response for such a field.
 *
 * @param tiles - Each tile's x, y and class.
 * @returns One class byte per tile.
 */
export function fieldBytes(tiles: readonly (readonly [number, number, FieldClass])[]): Uint8Array {
  const bytes = new Uint8Array(FIELD_SPAN * FIELD_SPAN);
  for (const [x, y, kind] of tiles) {
    bytes[y * FIELD_SPAN + x] = kind;
  }
  return bytes;
}
