/**
 * The commands the client sends in play, as the frames the server reads.
 *
 * A command is "!" and then its bytes XOR'd with the connection's table:
 * a type number, the command's code and its arguments (the server's
 * transport.route_client_frames; the bot's protocol/commands.py names
 * them). The client sends two: enter the game, which asks for the join
 * burst, and move, a click on a tile, which the server routes the tank to.
 */

import { COMMAND_PREFIX, WireError, xorBody } from "./wire.js";

const TYPE_QUERY = 2;
const TYPE_MOVEMENT = 4;
const ENTER_GAME = 63;
const MOVE = 112;
const FIELD_LAST = 255;

function command(table: Uint8Array, plain: readonly number[]): Uint8Array {
  const ciphered = xorBody(Uint8Array.from(plain), table, 0);
  const frame = new Uint8Array(ciphered.length + 1);
  frame[0] = COMMAND_PREFIX;
  frame.set(ciphered, 1);
  return frame;
}

/**
 * Enter the game: the server answers with the join burst on the next tick.
 *
 * @param table - The connection's XOR table.
 * @returns The frame body.
 */
export function enterGameCommand(table: Uint8Array): Uint8Array {
  return command(table, [TYPE_QUERY, ENTER_GAME]);
}

/**
 * Move to a field tile; the server chooses the route.
 *
 * @param table - The connection's XOR table.
 * @param x - The tile's x, 0 to 255.
 * @param y - The tile's y, 0 to 255.
 * @returns The frame body.
 * @throws WireError WIRE_TILE for a tile off the field.
 */
export function moveCommand(table: Uint8Array, x: number, y: number): Uint8Array {
  for (const value of [x, y]) {
    if (!Number.isInteger(value) || value < 0 || value > FIELD_LAST) {
      throw new WireError(`WIRE_TILE: (${x}, ${y}) is not a tile on the field`);
    }
  }
  return command(table, [TYPE_MOVEMENT, MOVE, x, y]);
}
