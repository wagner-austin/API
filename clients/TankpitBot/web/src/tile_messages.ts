/**
 * The envelope messages that write tiles: the viewport, pickups, radar, rocks.
 *
 * Each reads the plaintext body after its subtype byte, as the server's
 * encoders write it (protocol/encoders/world.py and radar.py, and the
 * container pickup in container/decoders/events.py):
 *
 * - 0x5A viewport: the window's left and top, then a skip-coded walk of
 *   the 18-wide patch around it. A step of 255 only moves the cursor; any
 *   other step lands on a tile and is followed by three bytes packing its
 *   cache (16 bits, 0xFFFF for equipment), its mine nibble (8 or more for
 *   none) and its dynamic terrain nibble.
 * - 0x43 pickup: four-byte records of x, y and the volume left (LE).
 * - 0x4F radar: a LE count of four-byte cache records, then three-byte
 *   mine records of x, y and a mine value (team in the low bits, 8 or
 *   more for none).
 * - 0x4A terrain: three-byte records of x, y and a dynamic terrain value.
 */

import { NO_MINE } from "./tile_grid.js";
import { byteAt, WireError } from "./wire.js";

/** The patch a viewport update walks: the 16x16 window and a one-tile border. */
export const PATCH_WIDTH = 18;

/** A cache value meaning equipment rather than fuel. */
export const EQUIPMENT = -1;

const SKIP = 255;
const EQUIPMENT_WIRE = 0xffff;
const MINE_NONE_FROM = 8;

/** One tile of a viewport update, in patch columns and rows. */
export interface PatchTile {
  readonly column: number;
  readonly row: number;
  readonly cache: number;
  readonly mine: number;
  readonly terrain: number;
}

/** A viewport update: where the window is and the tiles it reports. */
export interface ViewportUpdate {
  readonly left: number;
  readonly top: number;
  readonly tiles: readonly PatchTile[];
}

/** One write of a value to a field tile. */
export interface TileWrite {
  readonly x: number;
  readonly y: number;
  readonly value: number;
}

/** A radar scan: the caches and mines it rewrites. */
export interface RadarScan {
  readonly caches: readonly TileWrite[];
  readonly mines: readonly TileWrite[];
}

function wireCache(value: number): number {
  return value === EQUIPMENT_WIRE ? EQUIPMENT : value;
}

function wireMine(value: number): number {
  return value >= MINE_NONE_FROM ? NO_MINE : value & 3;
}

/**
 * Decode a 0x5A viewport update.
 *
 * @param inner - The body after the subtype byte.
 * @returns The window and its tiles.
 * @throws WireError WIRE_SHAPE for a body without its origin or with a tile record cut off.
 */
export function decodeViewport(inner: Uint8Array): ViewportUpdate {
  if (inner.length < 2) {
    throw new WireError(`WIRE_SHAPE: a viewport update needs its origin, not ${inner.length} bytes`);
  }
  const tiles: PatchTile[] = [];
  let cursor = 0;
  let at = 2;
  while (at < inner.length) {
    const step = byteAt(inner, at);
    cursor += step;
    at += 1;
    if (step === SKIP) {
      continue;
    }
    if (at + 3 > inner.length) {
      throw new WireError(`WIRE_SHAPE: a viewport tile record at byte ${at} is cut off`);
    }
    const packed = byteAt(inner, at) * 65536 + byteAt(inner, at + 1) * 256 + byteAt(inner, at + 2);
    at += 3;
    tiles.push({
      column: cursor % PATCH_WIDTH,
      row: Math.floor(cursor / PATCH_WIDTH),
      cache: wireCache(Math.floor(packed / 256)),
      mine: wireMine((packed >> 4) & 0x0f),
      terrain: packed & 0x0f,
    });
  }
  return { left: byteAt(inner, 0), top: byteAt(inner, 1), tiles };
}

function records(inner: Uint8Array, from: number, size: number, read: (at: number) => TileWrite): TileWrite[] {
  const out: TileWrite[] = [];
  for (let at = from; at < inner.length; at += size) {
    out.push(read(at));
  }
  return out;
}

/**
 * Whether a body is a container pickup: one or more whole four-byte records.
 *
 * @param inner - The body after the subtype byte.
 * @returns True for a pickup's shape.
 */
export function isPickup(inner: Uint8Array): boolean {
  return inner.length >= 4 && inner.length % 4 === 0;
}

/**
 * Decode a 0x43 container pickup as cache writes: each tile keeps what is left.
 *
 * @param inner - The body after the subtype byte, whole records (isPickup).
 * @returns One write per record.
 */
export function decodePickup(inner: Uint8Array): TileWrite[] {
  return records(inner, 0, 4, (at) => ({
    x: byteAt(inner, at),
    y: byteAt(inner, at + 1),
    value: wireCache(byteAt(inner, at + 2) + 256 * byteAt(inner, at + 3)),
  }));
}

/**
 * Whether a body is a radar scan: a count, that many caches, then whole mine records.
 *
 * @param inner - The body after the subtype byte.
 * @returns True for a radar scan's shape.
 */
export function isRadarScan(inner: Uint8Array): boolean {
  if (inner.length < 2) {
    return false;
  }
  const minesFrom = 2 + 4 * (byteAt(inner, 0) + 256 * byteAt(inner, 1));
  return minesFrom <= inner.length && (inner.length - minesFrom) % 3 === 0;
}

/**
 * Decode a 0x4F radar scan.
 *
 * @param inner - The body after the subtype byte, in a radar scan's shape (isRadarScan).
 * @returns The cache and mine writes.
 */
export function decodeRadarScan(inner: Uint8Array): RadarScan {
  const minesFrom = 2 + 4 * (byteAt(inner, 0) + 256 * byteAt(inner, 1));
  const caches = records(inner.subarray(0, minesFrom), 2, 4, (at) => ({
    x: byteAt(inner, at),
    y: byteAt(inner, at + 1),
    value: wireCache(byteAt(inner, at + 2) + 256 * byteAt(inner, at + 3)),
  }));
  const mines = records(inner, minesFrom, 3, (at) => ({ x: byteAt(inner, at), y: byteAt(inner, at + 1), value: wireMine(byteAt(inner, at + 2)) }));
  return { caches, mines };
}

/**
 * Decode a 0x4A terrain update: dynamic terrain written to tiles.
 *
 * @param inner - The body after the subtype byte.
 * @returns One write per whole record.
 * @throws WireError WIRE_SHAPE for a body that is not whole three-byte records.
 */
export function decodeTerrainUpdate(inner: Uint8Array): TileWrite[] {
  if (inner.length % 3 !== 0) {
    throw new WireError(`WIRE_SHAPE: a terrain update of ${inner.length} bytes is not whole records`);
  }
  return records(inner, 0, 3, (at) => ({ x: byteAt(inner, at), y: byteAt(inner, at + 1), value: byteAt(inner, at + 2) }));
}
