import { afterEach, describe, expect, it } from "vitest";

import { hooks, loadImageInto, realHooks, resetHooks, setHooks } from "../src/_test_hooks.js";
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
