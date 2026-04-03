import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks, createFakeResponse, createFakeErrorResponse } from "../../src/testing.js";
import { loadConfig, clearConfigCache } from "../../src/config.js";

describe("config", () => {
  beforeEach(() => {
    clearConfigCache();
  });

  afterEach(() => {
    resetHooks();
    clearConfigCache();
  });

  describe("loadConfig", () => {
    it("loads config from config.json", async () => {
      const { hooks, getFetchCalls } = createFakeHooks({
        fetchResponses: [createFakeResponse({ API_BASE_URL: "https://api.example.com" })],
        locationHostname: "api.example.com", // Non-local hostname to skip auto-detection
      });
      setHooks(hooks);

      const config = await loadConfig();

      expect(config.API_BASE_URL).toBe("https://api.example.com");
      expect(getFetchCalls()).toHaveLength(1);
      expect(getFetchCalls()[0]?.input).toBe("config.json");
    });

    it("caches config after first load", async () => {
      const { hooks, getFetchCalls } = createFakeHooks({
        fetchResponses: [createFakeResponse({ API_BASE_URL: "https://api.example.com" })],
        locationHostname: "api.example.com",
      });
      setHooks(hooks);

      const c1 = await loadConfig();
      const c2 = await loadConfig();

      expect(c1).toBe(c2);
      expect(getFetchCalls()).toHaveLength(1);
    });

    it("throws on fetch error", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeErrorResponse(404)],
      });
      setHooks(hooks);

      await expect(loadConfig()).rejects.toThrow("Failed to load config: 404");
    });

    it("throws on invalid config", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeResponse({ invalid: true })],
      });
      setHooks(hooks);

      await expect(loadConfig()).rejects.toThrow("Invalid config.json");
    });

    it("throws on empty API_BASE_URL", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeResponse({ API_BASE_URL: "" })],
      });
      setHooks(hooks);

      await expect(loadConfig()).rejects.toThrow("Invalid config.json");
    });

    it("throws on non-object response", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeResponse("string")],
      });
      setHooks(hooks);

      await expect(loadConfig()).rejects.toThrow("Invalid config.json");
    });
  });

  describe("clearConfigCache", () => {
    it("resets cache allowing fresh fetch", async () => {
      const { hooks: hooks1, getFetchCalls: getCalls1 } = createFakeHooks({
        fetchResponses: [createFakeResponse({ API_BASE_URL: "https://first.com" })],
        locationHostname: "first.com",
      });
      setHooks(hooks1);

      const c1 = await loadConfig();
      expect(c1.API_BASE_URL).toBe("https://first.com");
      expect(getCalls1()).toHaveLength(1);

      clearConfigCache();

      const { hooks: hooks2, getFetchCalls: getCalls2 } = createFakeHooks({
        fetchResponses: [createFakeResponse({ API_BASE_URL: "https://second.com" })],
        locationHostname: "second.com",
      });
      setHooks(hooks2);

      const c2 = await loadConfig();
      expect(c2.API_BASE_URL).toBe("https://second.com");
      expect(getCalls2()).toHaveLength(1);
    });
  });
});
