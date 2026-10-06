/**
 * The half of the connection before play: AUTH, the room list, select, enter.
 *
 * Every lobby frame is plaintext and is not enveloped (wiki
 * sim-network-server; the server's lobby.py). The exchange, in order:
 *
 *     --> %AUTH !be <account>|<token>|<stamp> <magic>
 *     <-- +<room>|<name>|<field id>|<modes>|<troop>|<mode>|<image>|<year>   one per room
 *     --> *<room>
 *     <-- =<room>|<game start>|<name>|<rank>|<four force counts>
 *     --> +<room>|<troop>|<preview x>|<preview y>|<metadata>
 *     <-- $<room>|0
 *
 * The account and token are this server's own, issued by
 * tankpit-sim-accounts; the magic is the client's choice and builds the
 * connection's cipher table on both ends. Quit is "-", which the server
 * echoes.
 */

/** A lobby frame the client does not understand, or cannot send. */
export class LobbyError extends Error {}

/** One room the server lists. */
export interface LobbyRoom {
  readonly roomId: string;
  readonly name: string;
  readonly image: string;
  readonly practice: boolean;
}

/** The server's answer to a select. */
export interface JoinConfirm {
  readonly roomId: string;
  readonly name: string;
  readonly rank: number;
}

/** What the client does with each lobby reply; readLobbyFrame calls exactly one. */
export interface LobbyHandler {
  room(room: LobbyRoom): void;
  confirm(confirm: JoinConfirm): void;
  entered(roomId: string): void;
  quit(): void;
}

const ENCODER = new TextEncoder();
const DECODER = new TextDecoder("utf-8");
const FIELD_PATTERN = /^[^|\s]+$/;
const ROOM_PATTERN = /^\+([^|]+)\|([^|]*)\|[^|]*\|[^|]*\|[^|]*\|([pn])\|([^|]+)\|[^|]*$/;
const CONFIRM_PATTERN = /^=([^|]+)\|[^|]*\|([^|]+)\|(\d+)(?:\|\d+){4}$/;
const ENTERED_PATTERN = /^\$([^|]+)\|0$/;

/**
 * Hold a field the client puts in a frame to the lobby's shape: no pipe, no space.
 *
 * @param name - What the field is, for the error.
 * @param value - The field.
 * @returns The field.
 * @throws LobbyError LOBBY_FIELD for an empty field or one holding a pipe or whitespace.
 */
function lobbyField(name: string, value: string): string {
  if (!FIELD_PATTERN.test(value)) {
    throw new LobbyError(`LOBBY_FIELD: the ${name} ${JSON.stringify(value)} is empty or holds a pipe or a space`);
  }
  return value;
}

/**
 * One capture group of a match that the pattern guarantees.
 *
 * @param match - The match.
 * @param group - The group's number.
 * @returns Its text.
 * @throws LobbyError LOBBY_FRAME when the group did not take part.
 */
export function captured(match: RegExpExecArray, group: number): string {
  const text = match[group];
  if (text === undefined) {
    throw new LobbyError(`LOBBY_FRAME: group ${group} of ${JSON.stringify(match[0])} did not take part`);
  }
  return text;
}

/**
 * The AUTH frame, the connection's first.
 *
 * @param account - The account's id.
 * @param token - Its token.
 * @param magic - The session magic.
 * @returns The frame body.
 * @throws LobbyError LOBBY_FIELD for a field the frame cannot carry.
 */
export function authFrame(account: string, token: string, magic: string): Uint8Array {
  return ENCODER.encode(`%AUTH !be ${lobbyField("account", account)}|${lobbyField("token", token)}|0 ${lobbyField("magic", magic)}`);
}

/**
 * The select frame, asking for a room's join confirm.
 *
 * @param roomId - The room.
 * @returns The frame body.
 * @throws LobbyError LOBBY_FIELD for a room id the frame cannot carry.
 */
export function selectFrame(roomId: string): Uint8Array {
  return ENCODER.encode(`*${lobbyField("room", roomId)}`);
}

/**
 * The enter frame, taking a seat in a room on a troop.
 *
 * The preview point and metadata are what the game's page sends; the
 * server reads only the room and the troop.
 *
 * @param roomId - The room.
 * @param troop - The team, 0 to 3.
 * @returns The frame body.
 * @throws LobbyError LOBBY_FIELD for a room id the frame cannot carry, LOBBY_TROOP for a troop that is not a team.
 */
export function enterFrame(roomId: string, troop: number): Uint8Array {
  if (!Number.isInteger(troop) || troop < 0 || troop > 3) {
    throw new LobbyError(`LOBBY_TROOP: troop ${troop} is not a team (0 to 3)`);
  }
  return ENCODER.encode(`+${lobbyField("room", roomId)}|${troop}|128|128|web`);
}

/** The quit frame. */
export function quitFrame(): Uint8Array {
  return ENCODER.encode("-");
}

/**
 * Read one plaintext frame the server sent and hand it to its handler method.
 *
 * @param body - The frame body.
 * @param handler - What to do with each kind of reply.
 * @throws LobbyError LOBBY_FRAME for a frame that is none of the lobby's replies.
 */
export function readLobbyFrame(body: Uint8Array, handler: LobbyHandler): void {
  const text = DECODER.decode(body);
  const room = ROOM_PATTERN.exec(text);
  if (room !== null) {
    handler.room({ roomId: captured(room, 1), name: captured(room, 2), image: captured(room, 4), practice: captured(room, 3) === "p" });
    return;
  }
  const confirm = CONFIRM_PATTERN.exec(text);
  if (confirm !== null) {
    handler.confirm({ roomId: captured(confirm, 1), name: captured(confirm, 2), rank: Number(captured(confirm, 3)) });
    return;
  }
  const entered = ENTERED_PATTERN.exec(text);
  if (entered !== null) {
    handler.entered(captured(entered, 1));
    return;
  }
  if (text !== "-") {
    throw new LobbyError(`LOBBY_FRAME: ${JSON.stringify(text.slice(0, 40))} is not a lobby reply`);
  }
  handler.quit();
}
