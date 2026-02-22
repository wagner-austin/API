import { describe, it, expect, afterEach, vi } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks, createFakeResponse, createFakeErrorResponse } from "../../src/testing.js";
import { translateAudio } from "../../src/api.js";

/**
 * Create a full translation response for testing.
 */
function createFullResponse(
  text: string,
  detectedLanguage: string = "vi",
  sourceText: string = "Xin chào",
  confidence: number = 0.95
): Record<string, unknown> {
  return {
    text,
    detected_language: detectedLanguage,
    source_text: sourceText,
    confidence,
  };
}

describe("api", () => {
  afterEach(() => {
    resetHooks();
  });

  describe("translateAudio", () => {
    it("sends audio and returns full translation response", async () => {
      const { hooks, getFetchCalls } = createFakeHooks({
        fetchResponses: [createFakeResponse(createFullResponse("Hello grandmother"))],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      const result = await translateAudio("https://api.example.com", "token123", blob);

      expect(result.text).toBe("Hello grandmother");
      expect(result.detected_language).toBe("vi");
      expect(result.source_text).toBe("Xin chào");
      expect(result.confidence).toBe(0.95);
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

    it("throws HTTP status when response body is not valid JSON", async () => {
      // Create a response with text that can't be parsed as JSON
      const { hooks } = createFakeHooks({
        fetchResponses: [
          {
            ok: false,
            status: 502,
            json: () => Promise.reject(new SyntaxError("Unexpected token")),
          } as Response,
        ],
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "HTTP 502"
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

    it("throws timeout error when request is aborted", async () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";
      const { hooks } = createFakeHooks({
        fetchThrow: abortError,
      });
      setHooks(hooks);

      const blob = new Blob(["audio"], { type: "audio/webm" });
      await expect(translateAudio("https://api.example.com", "token", blob)).rejects.toThrow(
        "Translation request timed out"
      );
    });

    it("aborts request after 30 second timeout", async () => {
      vi.useFakeTimers();

      try {
        // Create a fetch that never resolves, simulating a slow network
        let abortSignal: AbortSignal | null | undefined;
        const { hooks } = createFakeHooks();
        // Override the default fetch with one that never resolves
        const originalHooks = { ...hooks };
        originalHooks.fetch = (
          _url: string | URL,
          init?: RequestInit
        ): Promise<Response> =>
          new Promise<Response>((_resolve, reject) => {
            abortSignal = init?.signal;
            // When abort is called, reject with AbortError
            if (abortSignal) {
              abortSignal.addEventListener("abort", () => {
                const abortError = new Error("The operation was aborted");
                abortError.name = "AbortError";
                reject(abortError);
              });
            }
          });
        setHooks(originalHooks);

        const blob = new Blob(["audio"], { type: "audio/webm" });

        // Start the translation and capture any error
        let caughtError: Error | undefined;
        const translatePromise = translateAudio("https://api.example.com", "token", blob).catch(
          (err: Error) => {
            caughtError = err;
          }
        );

        // Advance past the 30 second timeout
        await vi.advanceTimersByTimeAsync(31000);

        // Wait for the promise to settle
        await translatePromise;

        // The request should have been aborted with our specific error message
        expect(caughtError?.message).toBe("Translation request timed out");
        expect(abortSignal?.aborted).toBe(true);
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
