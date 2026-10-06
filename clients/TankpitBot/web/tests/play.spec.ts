import { afterEach, describe, expect, it } from "vitest";

import { hooks, resetHooks, setHooks, type FrameSink, type SocketHandlers } from "../src/_test_hooks.js";
import { FieldClass } from "../src/field.js";
import { joinGame, serverUrls, startPlay, type PlayPage } from "../src/play.js";
import { ToolbarAction } from "../src/toolbar.js";
import { buildTable, joinFrames, splitFrames, xorBody } from "../src/wire.js";
import { Phase } from "../src/wire_client.js";
import { installFakes } from "./fakes.js";
import { fieldBytes } from "./fields.js";
import { envelope, GOLDEN } from "./golden.js";

afterEach(() => {
  resetHooks();
  document.body.innerHTML = "";
});

const KEY = "a-static-key-of-the-tests-own-making";
const MAGIC = "magic5uk3et4";
const PAGE_URL = "http://hub.test/tankpit-sim/?room=5";

/** What the page's fetches and socket did. */
interface Wire {
  readonly fetched: string[];
  readonly sockets: string[];
  readonly sent: Uint8Array[];
  readonly handlers: SocketHandlers[];
}

/** Bind the renderer's fakes, and the server's answers to the page's fetches and socket. */
function serve(): Wire {
  installFakes();
  const wire: Wire = { fetched: [], sockets: [], sent: [], handlers: [] };
  const answers = new Map<string, Uint8Array>([
    ["http://hub.test/tankpit-sim/cipher-key", new TextEncoder().encode(`${KEY}\n`)],
    ["http://hub.test/tankpit-sim/terrain/5", fieldBytes([[3, 3, FieldClass.Water]])],
  ]);
  const sink: FrameSink = { send: (data) => wire.sent.push(data) };
  setHooks({
    ...hooks(),
    fetchBytes: (url) => {
      wire.fetched.push(url);
      const answer = answers.get(url);
      return answer === undefined ? Promise.reject(new Error(`FETCH_STATUS: ${url} answered 404`)) : Promise.resolve(answer);
    },
    openSocket: (url, handlers) => {
      wire.sockets.push(url);
      wire.handlers.push(handlers);
      return sink;
    },
    randomMagic: () => MAGIC,
  });
  return wire;
}

function page(): PlayPage {
  return { document, location: { href: PAGE_URL }, devicePixelRatio: 2 };
}

const lobby = (...texts: string[]): Uint8Array => joinFrames(texts.map((text) => new TextEncoder().encode(text)));

describe("serverUrls", () => {
  it("are the page's own directory, with the socket on ws: or wss:", () => {
    expect(serverUrls("http://hub.test/tankpit-sim/?x=1", "5")).toEqual({
      key: "http://hub.test/tankpit-sim/cipher-key",
      terrain: "http://hub.test/tankpit-sim/terrain/5",
      socket: "ws://hub.test/tankpit-sim/",
    });
    expect(serverUrls("https://hub.test/play.html", "a b").socket).toBe("wss://hub.test/");
    expect(serverUrls("https://hub.test/play.html", "a b").terrain).toBe("https://hub.test/terrain/a%20b");
  });
});

describe("joinGame", () => {
  it("fetches the key and terrain, opens the socket, joins, plays and moves on a click", async () => {
    const wire = serve();
    const game = document.createElement("div");
    const status = document.createElement("p");
    const session = await joinGame(page(), game, status, { account: "1001", token: "tok", roomId: "5", troop: 1 });
    expect(wire.fetched).toEqual(["http://hub.test/tankpit-sim/cipher-key", "http://hub.test/tankpit-sim/terrain/5"]);
    expect(wire.sockets).toEqual(["ws://hub.test/tankpit-sim/"]);
    const handlers = wire.handlers[0];
    expect(handlers).toBeDefined();
    handlers?.open();
    expect(status.textContent).toBe("lobby");
    handlers?.message(lobby("+5|World (field05)|5|1,1,1,0,1,0,0|2|n|field05.gif|2026"));
    handlers?.message(lobby("=5|Oct. 05, 2026|austin|3|9|9|9|9"));
    handlers?.message(lobby("$5|0"));
    handlers?.message(joinFrames([envelope(GOLDEN.viewport), envelope(GOLDEN.info), envelope(GOLDEN.position)]));
    expect(status.textContent).toBe("playing, last frame 256 tiles and 1 tanks");
    expect(session.view.tileState(3, 3).terrain.kind).toBe("water");
    game.dispatchEvent(new MouseEvent("click", { clientX: 24 * 5 + 3, clientY: 16 * 7 + 3 }));
    const move = splitFrames(wire.sent.at(-1) ?? new Uint8Array(0))[0] ?? new Uint8Array(0);
    expect(Array.from(xorBody(move, buildTable(KEY, MAGIC), 1))).toEqual([4, 112, 243, 7]);
    game.dispatchEvent(new MouseEvent("click", { clientX: 60, clientY: 256 + 10 }));
    expect(session.renderer.pressedButton).toBe(ToolbarAction.Radar);
    handlers?.close(1006, "");
    expect([session.client.phase, status.textContent]).toEqual([Phase.Closed, "closed (1006), last frame 256 tiles and 1 tanks"]);
  });

  it("fails as the fetch fails, before any socket", async () => {
    const wire = serve();
    await expect(joinGame(page(), document.createElement("div"), document.createElement("p"), { account: "1", token: "t", roomId: "9", troop: 0 })).rejects.toThrow(
      "FETCH_STATUS: http://hub.test/tankpit-sim/terrain/9 answered 404",
    );
    expect(wire.sockets).toEqual([]);
  });
});

function form(omit = ""): void {
  const ids = ["game", "status", "join", "account", "token", "room", "troop"].filter((id) => id !== omit);
  document.body.innerHTML = ids.map((id) => (id === "join" ? '<form id="join"></form>' : ["account", "token", "room", "troop"].includes(id) ? `<input id="${id}" />` : `<div id="${id}"></div>`)).join("");
}

describe("startPlay", () => {
  it("joins with the form's fields on submit, hiding the form", async () => {
    const wire = serve();
    form();
    const values: Record<string, string> = { account: " 1001 ", token: "tok", room: "5", troop: "3" };
    for (const [id, value] of Object.entries(values)) {
      const input = document.getElementById(id);
      if (input instanceof HTMLInputElement) {
        input.value = value;
      }
    }
    const started = startPlay(page());
    const join = document.getElementById("join");
    join?.dispatchEvent(new Event("submit", { cancelable: true }));
    const session = await started;
    expect(join?.hidden).toBe(true);
    wire.handlers[0]?.open();
    expect(new TextDecoder().decode(splitFrames(wire.sent[0] ?? new Uint8Array(0))[0])).toBe(`%AUTH !be 1001|tok|0 ${MAGIC}`);
    expect(session.client.phase).toBe(Phase.Lobby);
  });

  it("refuses a page missing any of its elements, by id", () => {
    for (const id of ["game", "status", "join", "account", "token", "room", "troop"]) {
      form(id);
      expect(() => startPlay(page())).toThrow(`RENDER_PAGE: the page has no #${id}`);
    }
  });
});
