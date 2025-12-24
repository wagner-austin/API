import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks } from "../../src/testing.js";
import { saveToken, loadToken, clearToken } from "../../src/storage.js";

describe("storage", () => {
  let getStorageData: () => Map<string, string>;

  beforeEach(() => {
    const fakes = createFakeHooks();
    setHooks(fakes.hooks);
    getStorageData = fakes.getStorageData;
  });

  afterEach(() => {
    resetHooks();
  });

  describe("saveToken", () => {
    it("saves token to storage", () => {
      saveToken("test-token");

      expect(getStorageData().get("grandma_token")).toBe("test-token");
    });

    it("overwrites existing token", () => {
      saveToken("first");
      saveToken("second");

      expect(getStorageData().get("grandma_token")).toBe("second");
    });
  });

  describe("loadToken", () => {
    it("returns null when no token saved", () => {
      expect(loadToken()).toBeNull();
    });

    it("returns saved token", () => {
      saveToken("my-token");

      expect(loadToken()).toBe("my-token");
    });
  });

  describe("clearToken", () => {
    it("removes token from storage", () => {
      saveToken("to-remove");
      clearToken();

      expect(getStorageData().has("grandma_token")).toBe(false);
    });

    it("does nothing if no token exists", () => {
      clearToken();

      expect(loadToken()).toBeNull();
    });
  });
});
