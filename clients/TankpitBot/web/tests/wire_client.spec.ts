import { afterEach, describe, expect, it } from "vitest";

import { resetHooks, type FrameSink, type SocketHandlers } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { Renderer } from "../src/renderer.js";
import { buildTable, joinFrames, splitFrames, xorBody } from "../src/wire.js";
import { Phase, WireClient, type ClientListener, type JoinOptions } from "../src/wire_client.js";
import { WorldView } from "../src/world_view.js";
import { installFakes } from "./fakes.js";
import { fieldOf } from "./fields.js";
import { envelope, GOLDEN } from "./golden.js";

afterEach(resetHooks);

const OPTIONS: JoinOptions = {
  account: "1001",
  token: "token-of-austin",
  magic: "magic5uk3et4",
  staticKey: "a-static-key-of-the-tests-own-making",
  roomId: "5",
  troop: 2,
};

/** A socket that keeps every message the client sends. */
class RecordingSink implements FrameSink {
  public readonly sent: Uint8Array[] = [];
  public send(data: Uint8Array): void {
    this.sent.push(data);
  }
  /** Every frame sent, as text for the lobby's and as ciphered bytes for commands. */
  public frames(): string[] {
    return this.sent.flatMap((message) => splitFrames(message)).map((frame) => new TextDecoder().decode(frame));
  }
}

/** A listener that keeps the phase after every event. */
class PhaseLog implements ClientListener {
  public readonly phases: Phase[] = [];
  public changed(client: WireClient): void {
    this.phases.push(client.phase);
  }
}

interface Rig {
  readonly client: WireClient;
  readonly sink: RecordingSink;
  readonly handlers: SocketHandlers;
  readonly log: PhaseLog;
  readonly view: WorldView;
}

async function rig(): Promise<Rig> {
  installFakes();
  const renderer = await Renderer.create(document.createElement("div"), 1, defaultManifest("s.png"));
  const view = new WorldView(fieldOf([]), renderer);
  const sink = new RecordingSink();
  const log = new PhaseLog();
  const opened: SocketHandlers[] = [];
  const client = new WireClient(
    (handlers) => {
      opened.push(handlers);
      return sink;
    },
    OPTIONS,
    view,
    log,
  );
  const handlers = opened[0];
  if (handlers === undefined) {
    throw new Error("the client opened no socket");
  }
  return { client, sink, handlers, log, view };
}

const lobby = (...texts: string[]): Uint8Array => joinFrames(texts.map((text) => new TextEncoder().encode(text)));

async function playing(): Promise<Rig> {
  const setup = await rig();
  setup.handlers.open();
  setup.handlers.message(lobby("+5|World (field05)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026"));
  setup.handlers.message(lobby("=5|Oct. 05, 2026|austin|3|9|9|9|9"));
  setup.handlers.message(lobby("$5|0"));
  return setup;
}

describe("the lobby half", () => {
  it("sends AUTH on open, selects its room from the list, enters on the confirm and asks for the join burst", async () => {
    const { client, sink, handlers, log } = await rig();
    expect(client.phase).toBe(Phase.Connecting);
    handlers.open();
    handlers.message(lobby("+1|Practice|1|0,0,0,0,0,0,0|2|p|field01.gif|2026"));
    expect(client.phase).toBe(Phase.Lobby);
    handlers.message(lobby("+5|World (field05)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026"));
    handlers.message(lobby("=5|Oct. 05, 2026|austin|3|9|9|9|9"));
    handlers.message(lobby("$5|0"));
    expect(client.rooms.map((room) => room.roomId)).toEqual(["1", "5"]);
    expect(client.joinConfirm).toEqual({ roomId: "5", name: "austin", rank: 3 });
    expect(sink.frames().slice(0, 3)).toEqual(["%AUTH !be 1001|token-of-austin|0 magic5uk3et4", "*5", "+5|2|128|128|web"]);
    const table = buildTable(OPTIONS.staticKey, OPTIONS.magic);
    const enterGame = splitFrames(sink.sent[3] ?? new Uint8Array(0))[0] ?? new Uint8Array(0);
    expect(Array.from(xorBody(enterGame, table, 1))).toEqual([2, 63]);
    expect(log.phases).toEqual([Phase.Lobby, Phase.Lobby, Phase.Selected, Phase.Entering, Phase.Playing]);
    expect(client.phase).toBe(Phase.Playing);
  });

  it("refuses a lobby reply out of turn", async () => {
    const { handlers } = await rig();
    expect(() => handlers.message(lobby("+5|World (field05)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026"))).toThrow(
      "LOBBY_ORDER: room 5's listing arrived in phase connecting, not lobby",
    );
    handlers.open();
    expect(() => handlers.open()).toThrow("LOBBY_ORDER: the socket's open arrived in phase lobby");
    expect(() => handlers.message(lobby("=5|Oct. 05, 2026|austin|3|9|9|9|9"))).toThrow("LOBBY_ORDER: room 5's join confirm");
    expect(() => handlers.message(lobby("$5|0"))).toThrow("LOBBY_ORDER: room 5's enter response");
  });
});

describe("play", () => {
  it("reads each batch into the view and paints it once", async () => {
    const { client, handlers, view, log } = await playing();
    expect(client.frame).toBeNull();
    handlers.message(joinFrames([envelope(GOLDEN.viewport), envelope(GOLDEN.info), envelope(GOLDEN.position)]));
    expect(view.window).toEqual([238, 0]);
    expect(client.frame).toEqual({ tilesDrawn: 256, tanksDrawn: 1 });
    expect(log.phases.at(-1)).toBe(Phase.Playing);
  });

  it("refuses a frame of play before entry", async () => {
    const { handlers } = await rig();
    handlers.open();
    expect(() => handlers.message(joinFrames([envelope(GOLDEN.info)]))).toThrow("WIRE_PHASE: a frame of play arrived in phase lobby");
  });

  it("moves to a window tile, offset by the window, and quits", async () => {
    const { client, sink, handlers } = await playing();
    handlers.message(joinFrames([envelope(GOLDEN.viewport)]));
    client.moveTo(10, 3);
    const table = buildTable(OPTIONS.staticKey, OPTIONS.magic);
    const move = splitFrames(sink.sent.at(-1) ?? new Uint8Array(0))[0] ?? new Uint8Array(0);
    expect(Array.from(xorBody(move, table, 1))).toEqual([4, 112, 248, 3]);
    client.leave();
    expect(sink.frames().at(-1)).toBe("-");
    handlers.message(lobby("-"));
    expect(client.phase).toBe(Phase.Quit);
  });

  it("refuses a move before play or off the window, and a quit before play", async () => {
    const before = await rig();
    expect(() => before.client.moveTo(1, 1)).toThrow("WIRE_PHASE: a move in phase connecting");
    expect(() => before.client.leave()).toThrow("LOBBY_ORDER: a quit arrived in phase connecting, not playing");
    const { client } = await playing();
    for (const [column, row] of [
      [16, 0],
      [0, -1],
      [0.5, 2],
      [2, 0.5],
    ] as const) {
      expect(() => client.moveTo(column, row)).toThrow(`WIRE_TILE: (${column}, ${row}) is not a tile of the window`);
    }
  });

  it("records the socket's close", async () => {
    const { client, handlers, log } = await playing();
    handlers.close(1011, "");
    expect([client.phase, client.closedWith]).toEqual([Phase.Closed, "1011"]);
    handlers.close(1000, "bye");
    expect(client.closedWith).toBe("1000 bye");
    expect(log.phases.at(-1)).toBe(Phase.Closed);
  });
});
