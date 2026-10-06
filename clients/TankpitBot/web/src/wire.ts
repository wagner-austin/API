/**
 * The sim server's wire, below any message: frames and the session cipher.
 *
 * One WebSocket message is a run of frames, each a two-byte little-endian
 * length and that many bytes (wiki sim-network-server; the server's
 * transport.py). Lobby frames travel in the clear; every frame of play is a
 * 0x2E envelope whose body after the lead byte is XOR'd, and every command
 * the client sends is a "!" and its XOR'd bytes.
 *
 * The cipher table is the static key with the session's magic folded in,
 * character by character, the magic repeating (codec.build_xor_table). A
 * body longer than the table wraps around it, as the game client's own
 * decode does (`l[ja] ^= B[ja % pa]`). The key is not part of this
 * package: the server hands it out at /cipher-key, and this module only
 * builds tables from whatever key it is given.
 */

/** A frame or key the wire cannot carry. */
export class WireError extends Error {}

/** The lead byte of every frame of play. */
export const ENVELOPE = 0x2e;

/** The lead byte of every command the client sends. */
export const COMMAND_PREFIX = 0x21;

const LENGTH_BYTES = 2;
const MAX_FRAME = 0xffff;

/**
 * Read one byte that must be there.
 *
 * @param bytes - The bytes.
 * @param index - The byte's index.
 * @returns The byte.
 * @throws WireError WIRE_RANGE when the index is past the end.
 */
export function byteAt(bytes: Uint8Array, index: number): number {
  const value = bytes[index];
  if (value === undefined) {
    throw new WireError(`WIRE_RANGE: byte ${index} is past the end of ${bytes.length}`);
  }
  return value;
}

/**
 * Split one WebSocket message into its frame bodies.
 *
 * @param bytes - The message.
 * @returns Each frame's body, lead byte included, in order.
 * @throws WireError WIRE_TORN when a length is cut off or runs past the end, or a frame is empty.
 */
export function splitFrames(bytes: Uint8Array): Uint8Array[] {
  const frames: Uint8Array[] = [];
  let at = 0;
  while (at < bytes.length) {
    if (at + LENGTH_BYTES > bytes.length) {
      throw new WireError(`WIRE_TORN: a length at byte ${at} is cut off`);
    }
    const length = byteAt(bytes, at) + 256 * byteAt(bytes, at + 1);
    const start = at + LENGTH_BYTES;
    if (length === 0 || start + length > bytes.length) {
      throw new WireError(`WIRE_TORN: a frame of ${length} bytes at byte ${at} does not fit the ${bytes.length}-byte message`);
    }
    frames.push(bytes.subarray(start, start + length));
    at = start + length;
  }
  return frames;
}

/**
 * Join frame bodies into one WebSocket message.
 *
 * @param bodies - The bodies, lead byte included.
 * @returns The length-prefixed frames, concatenated.
 * @throws WireError WIRE_FRAME_SIZE for an empty body or one too long for its length.
 */
export function joinFrames(bodies: readonly Uint8Array[]): Uint8Array {
  const total = bodies.reduce((sum, body) => sum + LENGTH_BYTES + body.length, 0);
  const out = new Uint8Array(total);
  let at = 0;
  for (const body of bodies) {
    if (body.length === 0 || body.length > MAX_FRAME) {
      throw new WireError(`WIRE_FRAME_SIZE: a frame of ${body.length} bytes cannot be sent`);
    }
    out[at] = body.length & 0xff;
    out[at + 1] = body.length >> 8;
    out.set(body, at + LENGTH_BYTES);
    at += LENGTH_BYTES + body.length;
  }
  return out;
}

/**
 * Build a session's XOR table.
 *
 * @param staticKey - The server's static key, as /cipher-key serves it.
 * @param magic - The session magic this connection's AUTH names.
 * @returns One byte per key character.
 * @throws WireError WIRE_KEY when the key or the magic is empty or holds a character outside a byte.
 */
export function buildTable(staticKey: string, magic: string): Uint8Array {
  if (staticKey.length === 0 || magic.length === 0) {
    throw new WireError("WIRE_KEY: the static key and the magic must both be non-empty");
  }
  const table = new Uint8Array(staticKey.length);
  for (let index = 0; index < staticKey.length; index++) {
    const value = staticKey.charCodeAt(index) ^ magic.charCodeAt(index % magic.length);
    if (value > 0xff) {
      throw new WireError(`WIRE_KEY: character ${index} of the key or magic is outside a byte`);
    }
    table[index] = value;
  }
  return table;
}

/**
 * XOR a body against a table, from an offset, wrapping the table.
 *
 * The cipher is its own inverse, so this both reads a batch and writes a command.
 *
 * @param body - The bytes.
 * @param table - The session's table.
 * @param offset - The first byte to cipher; the bytes before it are dropped.
 * @returns The ciphered span.
 * @throws WireError WIRE_RANGE for an empty table under a non-empty span.
 */
export function xorBody(body: Uint8Array, table: Uint8Array, offset: number): Uint8Array {
  const out = new Uint8Array(Math.max(body.length - offset, 0));
  for (let index = 0; index < out.length; index++) {
    out[index] = byteAt(body, index + offset) ^ byteAt(table, index % table.length);
  }
  return out;
}
