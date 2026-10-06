/**
 * The wire client against a session the real server played.
 *
 * sim_session.ts is a whole session recorded from the sim's NetHost (and
 * held to it by tests/sim/test_web_session.py). The client must send
 * exactly the bytes the recording's client sent, read every message the
 * server answered, and draw the player's tank where the server left it.
 */

import { afterEach, describe, expect, it } from "vitest";

import { resetHooks, type SocketHandlers } from "../src/_test_hooks.js";
import { defaultManifest } from "../src/default_pack.js";
import { Renderer, type FrameReport } from "../src/renderer.js";
import { Phase, WireClient } from "../src/wire_client.js";
import { WorldView } from "../src/world_view.js";
import { installFakes } from "./fakes.js";
import { fieldOf } from "./fields.js";
import { hex } from "./golden.js";
import { SESSION } from "./sim_session.js";

afterEach(resetHooks);

const toHex = (bytes: Uint8Array): string => Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");

describe("a session the server played", () => {
  it("is sent byte for byte, read whole, and drawn where the server left the tank", async () => {
    installFakes();
    const renderer = await Renderer.create(document.createElement("div"), 1, defaultManifest("s.png"));
    const view = new WorldView(fieldOf([]), renderer);
    const sent: string[] = [];
    const opened: SocketHandlers[] = [];
    const frames: (FrameReport | null)[] = [];
    const client = new WireClient(
      (handlers) => {
        opened.push(handlers);
        return { send: (data) => sent.push(toHex(data)) };
      },
      { account: "1001", token: SESSION.token, magic: SESSION.magic, staticKey: SESSION.key, roomId: "1", troop: 2 },
      view,
      { changed: (changed) => frames.push(changed.frame) },
    );
    const handlers = opened[0];
    expect(handlers).toBeDefined();
    handlers?.open();
    const clientSent = SESSION.exchange.filter((message) => message.from === "client").map((message) => message.hex);
    // The client sends its first four messages itself (AUTH on open, then
    // select, enter and enter-game as the lobby answers); the fifth is the
    // player's click.
    let clientMessages = 0;
    for (const message of SESSION.exchange) {
      if (message.from === "server") {
        handlers?.message(hex(message.hex));
        continue;
      }
      clientMessages += 1;
      if (clientMessages === 5) {
        const [left, top] = view.window;
        client.moveTo(SESSION.targetX - left, SESSION.targetY - top);
      }
    }
    expect(sent).toEqual(clientSent);
    expect(client.phase).toBe(Phase.Playing);
    const [left, top] = view.window;
    expect(view.drawnAs(SESSION.tankId)).toEqual({ team: 2, facing: expect.any(Number), alive: true, deaths: 0, column: SESSION.endX - left, row: SESSION.endY - top });
    expect(view.drawnAs(65535)).toBeNull();
    const burst = frames.find((frame) => frame !== null);
    expect(burst?.tilesDrawn).toBe(256);
    expect(burst?.tanksDrawn).toBeGreaterThan(0);
  });
});
