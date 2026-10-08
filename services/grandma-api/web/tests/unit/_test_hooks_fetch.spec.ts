/**
 * The real fetch hook, answering a request and rejecting a refused one.
 *
 * Every other spec installs a fake fetch through setHooks(), so none of them
 * shows the production binding answering, or rejecting when nothing listens.
 * These call createDefaultHooks() itself (effect-seam-twin, board task
 * cc7222ca). This package carries no Node type declarations, so the success
 * is a data: URL, which the same fetch resolves without a server, and the
 * refusal is loopback port 1, tcpmux, where nothing on these machines
 * listens; the same fixture TankpitBot's existence probe uses.
 */
import { describe, expect, it } from "vitest";
import { createDefaultHooks } from "../../src/_test_hooks.js";

describe("createDefaultHooks().fetch", () => {
  it("answers a request with its body", async () => {
    const response = await createDefaultHooks().fetch("data:text/plain,language%3A%20en");

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("language: en");
  });

  it("rejects when nothing listens on the port", async () => {
    await expect(createDefaultHooks().fetch("http://127.0.0.1:1/detect")).rejects.toThrow(
      "fetch failed"
    );
  });
});
