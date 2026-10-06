/**
 * Envelope bodies exactly as the sim server writes them.
 *
 * Each is the plaintext of one 0x2E envelope (subtype byte first),
 * produced by the server's own encoder, tankpit_bot.protocol.encoders.
 * envelope.encode_envelope_body, from the message noted beside it. They
 * hold the TypeScript decoders to the server's bytes rather than to a
 * second reading of the layouts.
 */

import { buildTable, ENVELOPE, xorBody } from "../src/wire.js";

/** Bytes from a hex string. */
export function hex(text: string): Uint8Array {
  return Uint8Array.from(text.match(/../g) ?? [], (pair) => parseInt(pair, 16));
}

export const GOLDEN = {
  /** 0x21 TankInfo: tank 2000, team 2, "austin". */
  info: hex("2102d0070900000000000061757374696e"),
  /** 0x3E TankStatus: tank 513, team 1, rank 3, "red-513". */
  status: hex("3e3d0102000000000000000000007265642d353133"),
  /** 0x28 TankEntry: tank 527, team 3, at (0, 0). */
  entry: hex("28030f021b0000000000"),
  /** 0x29 TankExit: tank 500, team 0. */
  exit: hex("2900f4010000"),
  /** 0x3D MovementResponse: tank 2000, team 2, at (246, 1), direction 0x35. */
  position: hex("3d02d007f60135030300000000"),
  /** 0x47 Movement: tank 2000 from (246, 1) along "see". */
  movement: hex("47d007f6010003000000030100736565"),
  /** 0x58 TankRemove: tank 509. */
  remove: hex("58fd01"),
  /** 0x41 Deactivation: tank 518, by 2000. */
  deactivation: hex("4101060200d007"),
  /**
   * 0x5A ViewportUpdate at (238, 0): patch (1, 1) a ferry (5); patch (3, 16)
   * fuel 730; patch (17, 17) equipment, a team-2 mine and rock A (1).
   */
  viewport: hex("5aee0013000085ff1102da8020ffff21"),
  /** 0x4A TerrainUpdate: (10, 20) rock B, (11, 20) cleared. */
  terrain: hex("4a0a14020b1400"),
  /** 0x4F RadarScan: fuel 300 at (5, 6), equipment at (7, 8); a team-3 mine at (9, 10), none at (11, 12). */
  radar: hex("4f020005062c010708ffff090a030b0cff"),
  /** 0x43 ContainerPickup: (40, 41) has 257 left. */
  pickup: hex("4328290101"),
};

/** The table the tests cipher with. */
export const TABLE = buildTable("a-static-key-of-the-tests-own-making", "magic5uk3et4");

/**
 * An envelope frame as the server sends it: 0x2E, then the body ciphered.
 *
 * @param body - The plaintext body.
 * @returns The frame.
 */
export function envelope(body: Uint8Array): Uint8Array {
  const frame = new Uint8Array(body.length + 1);
  frame[0] = ENVELOPE;
  frame.set(xorBody(body, TABLE, 0), 1);
  return frame;
}
