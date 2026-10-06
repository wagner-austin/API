/**
 * One frame of play: an 0x2E envelope, un-XOR'd and handed to what it says.
 *
 * Every frame the server sends in play is the lead byte 0x2E and a body
 * XOR'd with the connection's table; the body's first byte is the
 * message's subtype (the server's transport.encode_tick_payload). The
 * subtype alone does not name a message: the production decoder
 * (protocol/decoders/tank.py, decode_0x2e_message) also asks the body's
 * shape, and a subtype whose body has another message's shape is handed
 * on to the container decoders. This module asks the same shapes, so a
 * body it does not draw from goes to `unrendered` by the same rule.
 *
 * What the renderer draws from, by subtype:
 *
 * - 0x21 tank info and 0x3E tank status: a tank's team and name;
 * - 0x28 entry: a tank's team as it joins; 0x29 exit: it left the game;
 * - 0x3D position: a tank stands at a tile, facing a direction;
 * - 0x47 movement: a tank walked a path of n, s, e and w steps;
 * - 0x58 remove: a tank left the client's view;
 * - 0x41 deactivation: a tank was destroyed and is a corpse;
 * - 0x5A, 0x43, 0x4F, 0x4A: tiles (tile_messages.ts).
 */

import {
  decodePickup,
  decodeRadarScan,
  decodeTerrainUpdate,
  decodeViewport,
  isPickup,
  isRadarScan,
  type RadarScan,
  type TileWrite,
  type ViewportUpdate,
} from "./tile_messages.js";
import { byteAt, ENVELOPE, WireError, xorBody } from "./wire.js";

/** A tank's identity: who it is and which team. */
export interface TankIdentity {
  readonly tankId: number;
  readonly team: number;
  readonly name: string;
}

/** A tank standing on a tile. */
export interface TankPlace {
  readonly tankId: number;
  readonly x: number;
  readonly y: number;
  readonly facing: number;
}

/** What the client does with each message it draws from; one call per frame. */
export interface EnvelopeHandler {
  identity(tank: TankIdentity): void;
  entry(tankId: number, team: number): void;
  place(place: TankPlace, team: number): void;
  walk(place: TankPlace): void;
  remove(tankId: number): void;
  exit(tankId: number): void;
  destroyed(tankId: number): void;
  viewport(update: ViewportUpdate): void;
  caches(writes: readonly TileWrite[]): void;
  radar(scan: RadarScan): void;
  rocks(writes: readonly TileWrite[]): void;
  unrendered(subtype: number): void;
}

const DECODER = new TextDecoder("utf-8");
const FACINGS = 16;
interface Step {
  readonly dx: number;
  readonly dy: number;
  readonly facing: number;
}

const STEPS = new Map<string, Step>([
  ["n", { dx: 0, dy: -1, facing: 0 }],
  ["e", { dx: 1, dy: 0, facing: 4 }],
  ["s", { dx: 0, dy: 1, facing: 8 }],
  ["w", { dx: -1, dy: 0, facing: 12 }],
]);

function u16(inner: Uint8Array, at: number): number {
  return byteAt(inner, at) + 256 * byteAt(inner, at + 1);
}

/**
 * Where a walk ends and which way the tank faces there.
 *
 * The path is the route the server chose, one letter a step; the tank
 * faces along its last step. A path with no steps leaves it where it
 * started, facing the direction byte's low nibble (the sprite row the
 * game's client draws; the high nibble is the trailing tile's).
 *
 * @param tankId - The walking tank.
 * @param startX - Where it started.
 * @param startY - Where it started.
 * @param direction - The message's direction byte.
 * @param path - The steps, as the wire's letters; other bytes are not steps.
 * @returns Where it stands and how it faces.
 */
export function walkEnd(tankId: number, startX: number, startY: number, direction: number, path: string): TankPlace {
  let x = startX;
  let y = startY;
  let facing = direction % FACINGS;
  for (const letter of path) {
    const step = STEPS.get(letter);
    if (step !== undefined) {
      x += step.dx;
      y += step.dy;
      facing = step.facing;
    }
  }
  return { tankId, x, y, facing };
}

function tankMessage(subtype: number, inner: Uint8Array, handler: EnvelopeHandler): boolean {
  if (subtype === 0x21 && inner.length >= 10) {
    handler.identity({ tankId: u16(inner, 1), team: byteAt(inner, 0) & 3, name: DECODER.decode(inner.subarray(10)) });
  } else if (subtype === 0x3e && inner.length >= 13) {
    handler.identity({ tankId: u16(inner, 1), team: byteAt(inner, 0) & 3, name: DECODER.decode(inner.subarray(13)) });
  } else if (subtype === 0x28 && inner.length >= 9) {
    handler.entry(u16(inner, 1), byteAt(inner, 3) & 3);
  } else if (subtype === 0x29 && inner.length === 5) {
    handler.exit(u16(inner, 1));
  } else if (subtype === 0x3d && inner.length >= 11) {
    handler.place({ tankId: u16(inner, 1), x: byteAt(inner, 3), y: byteAt(inner, 4), facing: byteAt(inner, 5) % FACINGS }, byteAt(inner, 0) & 3);
  } else if (subtype === 0x47 && inner.length >= 12) {
    handler.walk(walkEnd(u16(inner, 0), byteAt(inner, 2), byteAt(inner, 3), byteAt(inner, 4), DECODER.decode(inner.subarray(12))));
  } else if (subtype === 0x58 && inner.length >= 2) {
    handler.remove(u16(inner, 0));
  } else if (subtype === 0x41 && inner.length >= 6) {
    handler.destroyed(u16(inner, 1));
  } else {
    return false;
  }
  return true;
}

function tileMessage(subtype: number, inner: Uint8Array, handler: EnvelopeHandler): boolean {
  if (subtype === 0x5a && inner.length >= 2) {
    handler.viewport(decodeViewport(inner));
  } else if (subtype === 0x43 && isPickup(inner)) {
    handler.caches(decodePickup(inner));
  } else if (subtype === 0x4f && isRadarScan(inner)) {
    handler.radar(decodeRadarScan(inner));
  } else if (subtype === 0x4a) {
    handler.rocks(decodeTerrainUpdate(inner));
  } else {
    return false;
  }
  return true;
}

/**
 * Read one frame of play and hand it to its handler method.
 *
 * @param frame - The frame body, its 0x2E lead byte included.
 * @param table - The connection's XOR table.
 * @param handler - What to do with each message.
 * @throws WireError WIRE_ENVELOPE for a frame that is not an envelope or has no subtype; WIRE_SHAPE from a tile decoder.
 */
export function readEnvelope(frame: Uint8Array, table: Uint8Array, handler: EnvelopeHandler): void {
  if (frame.length < 2 || byteAt(frame, 0) !== ENVELOPE) {
    throw new WireError(`WIRE_ENVELOPE: a ${frame.length}-byte frame is not an envelope with a body`);
  }
  const body = xorBody(frame, table, 1);
  const subtype = byteAt(body, 0);
  const inner = body.subarray(1);
  if (!tankMessage(subtype, inner, handler) && !tileMessage(subtype, inner, handler)) {
    handler.unrendered(subtype);
  }
}
