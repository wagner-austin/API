import { describe, it, expect, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks, createFakeResponse, createFakeErrorResponse } from "../../src/testing.js";
import { translateAudio } from "../../src/api.js";

describe("api", () => {
  afterEach(() => {
    resetHooks();
  });

  describe("translateAudio", () => {
    it("sends audio and returns translated text", async () => {
      const { hooks, getFetchCalls } = createFakeHooks({
        fetchResponses: [createFakeResponse({ text: "Hello grandmother" })],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      const result = await translateAudio("https://api.example.com", "token123", blob);

      expect(result).toBe("Hello grandmother");
      expect(getFetchCalls()).toHaveLength(1);

      const call = getFetchCalls()[0];
      expect(call).toBeDefined();
      expect(call?.input).toBe("https://api.example.com/translate");
      expect(call?.init?.method).toBe("POST");
      expect(call?.init?.body).toBeInstanceOf(FormData);
    });

    it("throws on 401 with detail message", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeErrorResponse(401, "Invalid token")],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "bad-token", blob)).rejects.toThrow(
        "Invalid token"
      );
    });

    it("throws on 400 with detail message", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeErrorResponse(400, "No audio file")],
      });
      setHooks(hooks);

      const blob = new Blob([], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "No audio file"
      );
    });

    it("throws HTTP status when no detail", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeErrorResponse(500)],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "HTTP 500"
      );
    });

    it("throws on invalid response structure", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [createFakeResponse({ invalid: true })],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "Invalid response from server"
      );
    });

    it("throws when fetch itself throws (network error)", async () => {
      const { hooks } = createFakeHooks({
        fetchThrow: new Error("Network error"),
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "Network error"
      );
    });
  });
});
