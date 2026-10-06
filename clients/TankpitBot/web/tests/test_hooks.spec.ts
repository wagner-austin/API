import { afterEach, describe, expect, it } from "vitest";

import { bindSocket, hooks, loadImageInto, messageBytes, realHooks, resetHooks, responseBytes, setHooks, type SocketHandlers } from "../src/_test_hooks.js";
import { installFakes } from "./fakes.js";

afterEach(resetHooks);

describe("the hooks", () => {
  it("are the real implementations until a test binds others, and again after reset", () => {
    expect(hooks()).toBe(realHooks);
    installFakes();
    expect(hooks()).not.toBe(realHooks);
    resetHooks();
    expect(hooks()).toBe(realHooks);
    setHooks(realHooks);
    expect(hooks()).toBe(realHooks);
  });

  it("read a canvas's real 2D context, which jsdom does not have", () => {
    expect(realHooks.canvasContext(document.createElement("canvas"))).toBeNull();
  });

  it("encode a canvas as a PNG data URL", () => {
    const asked: string[] = [];
    const source = {
      toDataURL(type: string): string {
        asked.push(type);
        return "data:image/png;base64,AA==";
      },
    };
    expect(realHooks.canvasUrl(source)).toBe("data:image/png;base64,AA==");
    expect(asked).toEqual(["image/png"]);
  });

  it("start loading a real image from its URL", () => {
    const pending = realHooks.loadImage("sheet.png");
    expect(pending).toBeInstanceOf(Promise);
  });
});

/** Handlers that keep every event a socket reported. */
function recordingHandlers(): { readonly events: string[]; readonly handlers: SocketHandlers } {
  const events: string[] = [];
  return {
    events,
    handlers: {
      open: () => events.push("open"),
      message: (bytes) => events.push(`message ${Array.from(bytes).join(",")}`),
      close: (code, reason) => events.push(`close ${code} ${reason}`),
    },
  };
}

describe("the wire hooks", () => {
  it("open a real socket, which reports its close when nothing answers", async () => {
    const { events, handlers } = recordingHandlers();
    const closed = new Promise<void>((resolve) => {
      const sink = realHooks.openSocket("ws://127.0.0.1:9/", {
        ...handlers,
        close: (code, reason) => {
          handlers.close(code, reason);
          resolve();
        },
      });
      expect(sink).toBeInstanceOf(WebSocket);
    });
    await closed;
    expect(events).toEqual(["close 1006 "]);
  });

  it("bind a socket's open, binary messages and close to the handlers", () => {
    const { events, handlers } = recordingHandlers();
    const socket = new WebSocket("ws://127.0.0.1:9/");
    bindSocket(socket, handlers);
    expect(socket.binaryType).toBe("arraybuffer");
    socket.dispatchEvent(new Event("open"));
    socket.dispatchEvent(new MessageEvent("message", { data: Uint8Array.from([1, 2, 3]).buffer }));
    socket.dispatchEvent(new CloseEvent("close", { code: 1000, reason: "done" }));
    expect(events.slice(0, 3)).toEqual(["open", "message 1,2,3", "close 1000 done"]);
  });

  it("refuse a text message", () => {
    expect(messageBytes(new ArrayBuffer(2))).toEqual(new Uint8Array(2));
    expect(() => messageBytes("hello")).toThrow("SOCKET_TEXT: the server sent a message that is not binary");
  });

  it("fetch a URL's bytes, and refuse a status outside 2xx", async () => {
    await expect(realHooks.fetchBytes("data:application/octet-stream;base64,AQID")).resolves.toEqual(Uint8Array.from([1, 2, 3]));
    await expect(responseBytes(new Response("gone", { status: 404 }), "terrain/9")).rejects.toThrow("FETCH_STATUS: terrain/9 answered 404");
  });

  it("choose a fresh twenty-character magic of letters and digits", () => {
    const first = realHooks.randomMagic();
    expect(first).toMatch(/^[a-z0-9]{20}$/);
    expect(realHooks.randomMagic()).not.toBe(first);
  });
});

describe("loadImageInto", () => {
  it("resolves with the image once it loads", async () => {
    const image = document.createElement("img");
    const loaded = loadImageInto(image, "data:image/png;base64,AA==");
    expect(image.src).toBe("data:image/png;base64,AA==");
    image.dispatchEvent(new Event("load"));
    await expect(loaded).resolves.toBe(image);
  });

  it("rejects, naming the URL, when it fails", async () => {
    const image = document.createElement("img");
    const loaded = loadImageInto(image, "missing.png");
    image.dispatchEvent(new Event("error"));
    await expect(loaded).rejects.toThrow("RENDER_IMAGE: missing.png did not load");
  });
});
