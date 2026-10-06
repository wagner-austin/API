/**
 * One player's connection to the sim server, from AUTH to play.
 *
 * The client writes to a FrameSink (the open socket) and is handed every
 * message the socket receives. Its phases follow the lobby's own order
 * (lobby.ts): AUTH on open; the room list, from which it selects the room
 * it was asked to join; the join confirm, on which it enters on its
 * troop; the enter response, on which it asks for the join burst. From
 * then each message is one tick's batch of 0x2E envelopes, read into the
 * world view, which paints once per batch. A click on the field is a move
 * to that tile.
 *
 * Every frame is told apart by its lead byte: 0x2E is play, anything else
 * is the plaintext lobby (the server writes lobby replies, and the echo of
 * a quit, in the clear). An envelope before the client has entered, or a
 * lobby reply out of turn, is a protocol error and is raised.
 */

import type { FrameSink, SocketHandlers } from "./_test_hooks.js";
import { enterGameCommand, moveCommand } from "./commands.js";
import { readEnvelope } from "./envelope.js";
import { authFrame, enterFrame, LobbyError, quitFrame, readLobbyFrame, selectFrame, type JoinConfirm, type LobbyHandler, type LobbyRoom } from "./lobby.js";
import type { FrameReport } from "./renderer.js";
import { buildTable, byteAt, ENVELOPE, joinFrames, splitFrames, WireError } from "./wire.js";
import { WINDOW, type WorldView } from "./world_view.js";

/** Who joins, where, and with which cipher. */
export interface JoinOptions {
  readonly account: string;
  readonly token: string;
  readonly magic: string;
  readonly staticKey: string;
  readonly roomId: string;
  readonly troop: number;
}

/** Where the connection is. */
export enum Phase {
  Connecting = "connecting",
  Lobby = "lobby",
  Selected = "selected",
  Entering = "entering",
  Playing = "playing",
  Quit = "quit",
  Closed = "closed",
}

/** Open the socket, reporting its events to the handlers. */
export interface Connect {
  (handlers: SocketHandlers): FrameSink;
}

/** Told after every event, so a page can show where the connection is. */
export interface ClientListener {
  changed(client: WireClient): void;
}

/** A player's connection: its socket's handler, its lobby half and its play. */
export class WireClient implements LobbyHandler, SocketHandlers {
  private current = Phase.Connecting;
  private readonly table: Uint8Array;
  private readonly listed: LobbyRoom[] = [];
  private joined: JoinConfirm | null = null;
  private lastFrame: FrameReport | null = null;
  private closing = "";
  private readonly sink: FrameSink;

  /**
   * Open a connection: build its cipher table, then its socket.
   *
   * @param connect - How the socket is opened; the client is its handlers.
   * @param options - The account, room and troop, the magic and the server's static key.
   * @param view - The room's world view.
   * @param listener - Told after every event.
   * @throws WireError WIRE_KEY for an empty key or magic.
   */
  public constructor(
    connect: Connect,
    private readonly options: JoinOptions,
    private readonly view: WorldView,
    private readonly listener: ClientListener,
  ) {
    this.table = buildTable(options.staticKey, options.magic);
    this.sink = connect(this);
  }

  /** Where the connection is. */
  public get phase(): Phase {
    return this.current;
  }

  /** The last frame a batch of play drew, or null before play. */
  public get frame(): FrameReport | null {
    return this.lastFrame;
  }

  /** The close code and reason, once the socket closed. */
  public get closedWith(): string {
    return this.closing;
  }

  /** The rooms the server listed. */
  public get rooms(): readonly LobbyRoom[] {
    return this.listed;
  }

  /** The join confirm, once the server sent it. */
  public get joinConfirm(): JoinConfirm | null {
    return this.joined;
  }

  private send(...bodies: Uint8Array[]): void {
    this.sink.send(joinFrames(bodies));
  }

  private expect(phase: Phase, what: string): void {
    if (this.current !== phase) {
      throw new LobbyError(`LOBBY_ORDER: ${what} arrived in phase ${this.current}, not ${phase}`);
    }
  }

  /** The socket opened: send AUTH. */
  public open(): void {
    this.expect(Phase.Connecting, "the socket's open");
    this.current = Phase.Lobby;
    this.send(authFrame(this.options.account, this.options.token, this.options.magic));
    this.listener.changed(this);
  }

  /**
   * Take one message the socket received; a message carrying play is painted.
   *
   * @param bytes - The message.
   * @throws WireError for a torn message or an envelope before entry; LobbyError for a lobby reply out of turn.
   */
  public message(bytes: Uint8Array): void {
    let played = false;
    for (const frame of splitFrames(bytes)) {
      if (byteAt(frame, 0) !== ENVELOPE) {
        readLobbyFrame(frame, this);
        continue;
      }
      if (this.current !== Phase.Playing) {
        throw new WireError(`WIRE_PHASE: a frame of play arrived in phase ${this.current}`);
      }
      readEnvelope(frame, this.table, this.view);
      played = true;
    }
    if (played) {
      this.lastFrame = this.view.paint();
    }
    this.listener.changed(this);
  }

  /**
   * The socket closed.
   *
   * @param code - The close code.
   * @param reason - The close reason.
   */
  public close(code: number, reason: string): void {
    this.current = Phase.Closed;
    this.closing = `${code} ${reason}`.trim();
    this.listener.changed(this);
  }

  public room(room: LobbyRoom): void {
    this.expect(Phase.Lobby, `room ${room.roomId}'s listing`);
    this.listed.push(room);
    if (room.roomId === this.options.roomId) {
      this.current = Phase.Selected;
      this.send(selectFrame(room.roomId));
    }
  }

  public confirm(confirm: JoinConfirm): void {
    this.expect(Phase.Selected, `room ${confirm.roomId}'s join confirm`);
    this.joined = confirm;
    this.current = Phase.Entering;
    this.send(enterFrame(this.options.roomId, this.options.troop));
  }

  public entered(roomId: string): void {
    this.expect(Phase.Entering, `room ${roomId}'s enter response`);
    this.current = Phase.Playing;
    this.send(enterGameCommand(this.table));
  }

  public quit(): void {
    this.current = Phase.Quit;
  }

  /**
   * Move to a tile of the window, as a click on the field asks.
   *
   * @param column - The window column, 0 to 15.
   * @param row - The window row, 0 to 15.
   * @throws WireError WIRE_PHASE before play, WIRE_TILE for a tile outside the window.
   */
  public moveTo(column: number, row: number): void {
    if (this.current !== Phase.Playing) {
      throw new WireError(`WIRE_PHASE: a move in phase ${this.current}`);
    }
    if (!Number.isInteger(column) || !Number.isInteger(row) || column < 0 || row < 0 || column >= WINDOW || row >= WINDOW) {
      throw new WireError(`WIRE_TILE: (${column}, ${row}) is not a tile of the window`);
    }
    const [left, top] = this.view.window;
    this.send(moveCommand(this.table, left + column, top + row));
  }

  /** Leave the room; the server echoes the quit. */
  public leave(): void {
    this.expect(Phase.Playing, "a quit");
    this.send(quitFrame());
  }
}
