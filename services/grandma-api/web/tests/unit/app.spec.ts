import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import {
  createFakeHooks,
  createFakeResponse,
  createFakeErrorResponse,
  createTestElements,
  FakeHooksResult,
} from "../../src/testing.js";
import { createApp, App, autoInit, clearConfigCache } from "../../src/app.js";

describe("app", () => {
  let elements: Map<string, HTMLElement>;
  let fakes: FakeHooksResult;

  beforeEach(() => {
    // Clear any leftover timers from previous tests
    vi.clearAllTimers();
    elements = createTestElements();
    clearConfigCache();
  });

  afterEach(() => {
    // Clear timers and reset state
    vi.clearAllTimers();
    resetHooks();
    clearConfigCache();
  });

  function setupHooks(config: {
    fetchResponses?: Response[];
    initialStorage?: Map<string, string>;
    getUserMediaError?: Error;
    getUserMediaThrow?: unknown;
    documentReadyState?: DocumentReadyState;
    webmMuxerConfig?: { decodeError?: Error };
  } = {}): void {
    const hooksConfig: {
      fetchResponses: Response[];
      elements: Map<string, HTMLElement>;
      initialStorage?: Map<string, string>;
      getUserMediaResult?: MediaStream | Error;
      getUserMediaThrow?: unknown;
      documentReadyState?: DocumentReadyState;
      webmMuxerConfig?: { decodeError?: Error };
    } = {
      fetchResponses: config.fetchResponses ?? [
        createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
      ],
      elements,
    };
    if (config.initialStorage !== undefined) {
      hooksConfig.initialStorage = config.initialStorage;
    }
    if (config.getUserMediaError !== undefined) {
      hooksConfig.getUserMediaResult = config.getUserMediaError;
    }
    if (config.getUserMediaThrow !== undefined) {
      hooksConfig.getUserMediaThrow = config.getUserMediaThrow;
    }
    if (config.documentReadyState !== undefined) {
      hooksConfig.documentReadyState = config.documentReadyState;
    }
    if (config.webmMuxerConfig !== undefined) {
      hooksConfig.webmMuxerConfig = config.webmMuxerConfig;
    }
    fakes = createFakeHooks(hooksConfig);
    setHooks(fakes.hooks);
  }

  describe("createApp", () => {
    it("creates new App instance", () => {
      const app = createApp();

      expect(app).toBeInstanceOf(App);
    });

    it("handleRecordClick does nothing before init", async () => {
      setupHooks();
      const app = createApp();

      // Call handleRecordClick before init - should return early without error
      await app.handleRecordClick();

      // Verify no state change
      const state = app.getState();
      expect(state.isRecording).toBe(false);
    });

    it("handleClear works before init", async () => {
      setupHooks();
      const app = createApp();

      // Call handleClear before init - should not throw
      await app.handleClear();

      const state = app.getState();
      expect(state.transcripts).toEqual([]);
    });

    it("handleLogin does nothing before init", async () => {
      setupHooks();
      const app = createApp();

      const event = new Event("submit", { cancelable: true });
      await app.handleLogin(event);

      const state = app.getState();
      expect(state.token).toBeNull();
    });

    it("handleLogout works before init", async () => {
      setupHooks();
      const app = createApp();

      await app.handleLogout();

      const state = app.getState();
      expect(state.token).toBeNull();
    });
  });

  describe("App.init", () => {
    it("initializes and shows login when no token", async () => {
      setupHooks();
      const app = createApp();

      await app.init();

      const loginSection = elements.get("login-section");
      const mainSection = elements.get("main-section");
      expect(loginSection?.classList.contains("hidden")).toBe(false);
      expect(mainSection?.classList.contains("hidden")).toBe(true);
    });

    it("shows main section when token exists", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "saved-token"]]),
      });
      const app = createApp();

      await app.init();

      const loginSection = elements.get("login-section");
      const mainSection = elements.get("main-section");
      expect(loginSection?.classList.contains("hidden")).toBe(true);
      expect(mainSection?.classList.contains("hidden")).toBe(false);
    });

    it("throws when element not found", async () => {
      // Remove a required element
      elements.delete("login-section");
      setupHooks();
      const app = createApp();

      await expect(app.init()).rejects.toThrow("Element not found: #login-section");
    });

    it("throws on config load failure", async () => {
      setupHooks({
        fetchResponses: [createFakeErrorResponse(500)],
      });
      const app = createApp();

      await expect(app.init()).rejects.toThrow("Failed to load config: 500");
    });
  });

  describe("App.handleLogin", () => {
    it("saves token and shows main section", async () => {
      setupHooks();
      const app = createApp();
      await app.init();

      const tokenInput = elements.get("token-input") as HTMLInputElement;
      tokenInput.value = "my-password";

      const event = new Event("submit", { bubbles: true, cancelable: true });
      await app.handleLogin(event);

      const mainSection = elements.get("main-section");
      expect(mainSection?.classList.contains("hidden")).toBe(false);
      expect(fakes.getStorageData().get("grandma_token")).toBe("my-password");
    });

    it("shows error on empty password", async () => {
      setupHooks();
      const app = createApp();
      await app.init();

      const tokenInput = elements.get("token-input") as HTMLInputElement;
      tokenInput.value = "   ";

      const event = new Event("submit", { bubbles: true, cancelable: true });
      await app.handleLogin(event);

      const loginError = elements.get("login-error");
      expect(loginError?.classList.contains("hidden")).toBe(false);
      expect(loginError?.textContent).toBe("Please enter a password");
    });

    it("prevents default form submission", async () => {
      setupHooks();
      const app = createApp();
      await app.init();

      const event = new Event("submit", { bubbles: true, cancelable: true });
      let defaultPrevented = false;
      event.preventDefault = () => {
        defaultPrevented = true;
      };

      const tokenInput = elements.get("token-input") as HTMLInputElement;
      tokenInput.value = "test";

      await app.handleLogin(event);

      expect(defaultPrevented).toBe(true);
    });
  });

  describe("App.handleLogout", () => {
    it("clears token and shows login", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      await app.handleLogout();

      const loginSection = elements.get("login-section");
      expect(loginSection?.classList.contains("hidden")).toBe(false);
      expect(fakes.getStorageData().has("grandma_token")).toBe(false);
    });

    it("clears transcripts", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Simulate having transcripts by checking state
      await app.handleLogout();

      const transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("Translations will appear here...");
    });
  });

  describe("App.handleRecordClick", () => {
    it("starts recording on first click", async () => {
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // for stop recording
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      await app.handleRecordClick();

      const recordBtn = elements.get("record-btn");
      const status = elements.get("status");
      expect(recordBtn?.classList.contains("recording")).toBe(true);
      expect(status?.textContent).toBe("Recording... (updates every 15s)");

      // Clean up - stop recording to prevent RAF loop from continuing
      await app.handleRecordClick();
    });

    it("stops recording and keeps chunk translations", async () => {
      // Chunk translations are kept when recording stops (no final re-translation)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello grandmother" }), // chunk translation (kept)
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      // Simulate a chunk being recorded (triggers onChunk callback with translation)
      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }
      recorder.requestData();

      // Wait for async chunk translation to complete
      await app.waitForPendingOperations();

      // Stop recording - chunk translations are kept
      await app.handleRecordClick();

      const transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("Hello grandmother");
    });

    it("throws if stopPromise is null when stopping", async () => {
      // Remove AudioContext to avoid RAF loop during this error test
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording to set isRecording to true
      await app.handleRecordClick();

      // Manually clear stopPromise to simulate invalid state
      Object.assign(app, { stopPromise: null });

      // Try to stop - should throw
      await expect(app.handleRecordClick()).rejects.toThrow(
        "Recording stopped but no stop promise available"
      );

      globalThis.AudioContext = savedAudioContext;
    });

    it("accumulates multiple transcripts", async () => {
      // Each recording session creates a separate transcript entry
      // Chunk translations are kept (no final re-translation)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "First" }), // session 1 chunk (kept)
          createFakeResponse({ text: "Second" }), // session 2 chunk (kept)
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // First recording - start, trigger chunk, stop
      await app.handleRecordClick();
      let recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      let recorder = recorders[recorders.length - 1];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }
      recorder.requestData();
      await app.waitForPendingOperations();
      await app.handleRecordClick();

      // Second recording - start, trigger chunk, stop
      await app.handleRecordClick();
      recorders = fakes.getMediaRecorderInstances();
      recorder = recorders[recorders.length - 1];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }
      recorder.requestData();
      await app.waitForPendingOperations();
      await app.handleRecordClick();

      const transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("First\n\nSecond");
    });

    it("appends text when multiple chunks arrive", async () => {
      // Each chunk is transcribed independently and APPENDED to the transcript
      // Chunk translations are kept when recording stops (no final re-translation)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "First chunk" }), // chunk 1 translation
          createFakeResponse({ text: "second chunk" }), // chunk 2 translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // First chunk creates new transcript entry
      recorder.requestData();
      await app.waitForPendingOperations();

      let transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("First chunk");

      // Second chunk APPENDS to existing entry
      recorder.requestData();
      await app.waitForPendingOperations();

      transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("First chunk second chunk");

      // Still only one transcript entry (appended in place)
      let state = app.getState();
      expect(state.transcripts.length).toBe(1);

      // Stop recording - chunk translations are kept (no final re-translation)
      await app.handleRecordClick();

      transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("First chunk second chunk");
      state = app.getState();
      expect(state.transcripts.length).toBe(1);
    });

    it("waits for pending chunk translation when stopping", async () => {
      // This test verifies that stopping recording waits for any in-progress
      // chunk translation to complete before keeping those results
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Chunk result" }), // chunk translation (kept)
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Trigger chunk - this starts async translation
      recorder.requestData();

      // Stop recording immediately WITHOUT waiting for pending operations
      // This exercises the code path that awaits pendingChunkTranslation
      await app.handleRecordClick();

      // Chunk result should be kept (no final re-translation)
      const transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("Chunk result");
    });

    it("updates timer display after 1 second", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // Capture setInterval callback directly (like the recordingStartTime test)
      let intervalCallback: (() => void) | null = null;
      const originalSetInterval = globalThis.setInterval;
      globalThis.setInterval = ((fn: () => void, _ms: number) => {
        intervalCallback = fn;
        return 999 as unknown as ReturnType<typeof setInterval>;
      }) as typeof setInterval;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording to start the timer
      await app.handleRecordClick();

      // Restore setInterval
      globalThis.setInterval = originalSetInterval;

      // Verify callback was captured
      if (intervalCallback === null) {
        throw new Error("intervalCallback was not captured");
      }

      // Assign to const with explicit type for TypeScript narrowing
      const callback: () => void = intervalCallback;

      // Manually set recordingStartTime to 1 second in the past
      const oneSecondAgo = Date.now() - 1000;
      Object.assign(app, { recordingStartTime: oneSecondAgo });

      // Call the interval callback directly
      callback();

      const timer = elements.get("timer");
      expect(timer?.textContent).toBe("0:01");

      // Stop recording
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("works without AudioContext", async () => {
      // Save and remove AudioContext
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording - should not throw even without AudioContext
      await app.handleRecordClick();

      const recordingFeedback = elements.get("recording-feedback");
      expect(recordingFeedback?.classList.contains("hidden")).toBe(false);

      // Stop recording (no chunks triggered)
      await app.handleRecordClick();

      // Restore AudioContext
      globalThis.AudioContext = savedAudioContext;
    });

    it("throws if recordingStartTime is null when timer fires", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // Capture the interval callback so we can call it directly
      let intervalCallback: (() => void) | null = null;
      const originalSetInterval = globalThis.setInterval;
      globalThis.setInterval = ((fn: () => void, _ms: number) => {
        intervalCallback = fn;
        return 999 as unknown as ReturnType<typeof setInterval>;
      }) as typeof setInterval;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // for stop recording
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording to start the timer
      await app.handleRecordClick();

      // Restore setInterval before test assertions
      globalThis.setInterval = originalSetInterval;

      // Verify callback was captured
      if (intervalCallback === null) {
        throw new Error("intervalCallback was not captured");
      }

      // Assign to const with explicit type for TypeScript narrowing in closure
      const callback: () => void = intervalCallback;

      // Manually clear recordingStartTime to simulate invalid state
      Object.assign(app, { recordingStartTime: null });

      // Call the interval callback directly - should throw
      expect(() => callback()).toThrow(
        "Timer running but recording start time is null"
      );

      // Clean up: stop recording to clear interval
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("throws if analyser is null when animation frame fires", async () => {
      // This test verifies the error branch in updateLevel when analyser becomes null
      // We capture the RAF callback and call it after setting analyser to null

      // Capture the requestAnimationFrame callback
      let rafCallback: (() => void) | null = null;
      const originalRAF = globalThis.requestAnimationFrame;
      globalThis.requestAnimationFrame = ((fn: () => void) => {
        rafCallback = fn;
        return 999 as unknown as number;
      }) as typeof requestAnimationFrame;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // for stop recording
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording to trigger the RAF loop with AudioContext
      await app.handleRecordClick();

      // Restore RAF before assertions
      globalThis.requestAnimationFrame = originalRAF;

      // Verify callback was captured
      if (rafCallback === null) {
        throw new Error("rafCallback was not captured");
      }

      // Assign to const for TypeScript narrowing
      const callback: () => void = rafCallback;

      // Manually clear analyser to simulate invalid state
      Object.assign(app, { analyser: null });

      // Call the RAF callback directly - should throw
      expect(() => callback()).toThrow(
        "Animation frame running but analyser is null"
      );

      // Clean up: stop recording (analyser is already null, which is fine)
      await app.handleRecordClick();
    });

    it("handles stopRecordingFeedback when timerInterval is already null", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      // Manually clear timerInterval to simulate edge case
      Object.assign(app, { timerInterval: null });

      // Stop recording - should handle null timerInterval gracefully
      await app.handleRecordClick();

      // Verify recording stopped successfully
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });

    it("auto-stops recording after max duration", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // Use wrapper object so TypeScript tracks mutations through closures
      const captured: { callback: (() => void) | null } = { callback: null };
      const originalSetTimeout = globalThis.setTimeout;
      vi.spyOn(globalThis, "setTimeout").mockImplementation((callback, delay) => {
        // Only capture the 60000ms auto-stop timeout, let others through
        if (delay === 60000) {
          captured.callback = callback as () => void;
          return 12345 as unknown as ReturnType<typeof setTimeout>;
        }
        return originalSetTimeout(callback, delay);
      });

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Auto-stopped translation" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(true);

      // Callback should have been captured
      expect(captured.callback).not.toBeNull();

      // Invoke the auto-stop callback (simulating 60 seconds passing)
      if (captured.callback !== null) {
        captured.callback();
        // Need to wait for the async handleRecordClick to complete
        await new Promise((resolve) => originalSetTimeout(resolve, 50));
      }

      // Recording should have stopped
      expect(app.getState().isRecording).toBe(false);

      vi.restoreAllMocks();
      globalThis.AudioContext = savedAudioContext;
    });

    it("clears auto-stop timeout when manually stopping", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Manual stop translation" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Start recording
      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(true);
      expect(appAny.autoStopTimeout).not.toBeNull();

      // Manually stop
      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(false);

      // Auto-stop timeout should have been cleared
      expect(appAny.autoStopTimeout).toBeNull();

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles stop when autoStopTimeout is already null", async () => {
      // Edge case: timeout was already cleared (e.g., by a race condition)
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Translation" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Start recording
      await app.handleRecordClick();

      // Simulate the timeout already being cleared (edge case)
      appAny.autoStopTimeout = null;

      // Stop recording - should not throw even with null timeout
      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });
  });

  describe("App.handleClear", () => {
    it("clears transcripts", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Some text" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Record something (start and stop, no chunks triggered)
      await app.handleRecordClick();
      await app.handleRecordClick();

      // Clear
      await app.handleClear();

      const transcript = elements.get("transcript");
      expect(transcript?.textContent).toBe("Translations will appear here...");

      globalThis.AudioContext = savedAudioContext;
    });

    it("waitForPendingOperations returns immediately when no pending translation", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Call waitForPendingOperations when there's NO pending translation
      // This should return immediately (covers the false branch of the if check)
      await app.waitForPendingOperations();

      // Verify app state is unchanged
      expect(app.getState().transcripts).toEqual([]);

      globalThis.AudioContext = savedAudioContext;
    });
  });

  describe("App.getState", () => {
    it("returns current state", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "my-token"]]),
      });
      const app = createApp();
      await app.init();

      const state = app.getState();

      expect(state.token).toBe("my-token");
      expect(state.transcripts).toEqual([]);
      expect(state.isRecording).toBe(false);
    });

    it("reflects recording state", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // for stop recording
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(true);

      // Clean up
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });
  });

  describe("autoInit", () => {
    it("initializes app when document is complete", () => {
      setupHooks({
        documentReadyState: "complete",
      });

      // autoInit creates and initializes app
      // Since we can't easily test the async init in autoInit,
      // we just verify it doesn't throw
      expect(() => autoInit()).not.toThrow();
    });

    it("adds event listener when document is loading", () => {
      setupHooks({
        documentReadyState: "loading",
      });

      // autoInit should add event listener for DOMContentLoaded
      expect(() => autoInit()).not.toThrow();
    });
  });

  describe("error handling", () => {
    it("handles microphone access denied", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
        getUserMediaError: new Error("Permission denied"),
      });
      const app = createApp();
      await app.init();

      // This should catch the error internally via the event listener
      // The App class has error handling in setupEventListeners
      await expect(app.handleRecordClick()).rejects.toThrow("Permission denied");
    });

    it("handles translation error gracefully during chunk", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // With chunked streaming, translation errors are caught and logged
      // so recording can continue (doesn't throw)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeErrorResponse(401, "Invalid token"), // chunk translation error
          createFakeResponse({ text: "Final transcript" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      // Trigger a chunk - this will get a 401 error but should not crash
      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }
      recorder.requestData();

      // Wait for async error handling
      await app.waitForPendingOperations();

      // App should still be recording (error was caught and logged)
      expect(app.getState().isRecording).toBe(true);

      // Stop recording - should complete without error
      await app.handleRecordClick();
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles chunk translation success when recording already stopped", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the branch where recording is stopped before
      // the chunk translation completes successfully (line 453 false branch)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Chunk text" }), // chunk translation
          createFakeResponse({ text: "Final text" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Trigger a chunk (starts async translation)
      recorder.requestData();

      // Directly set recording state to false WITHOUT going through handleRecordClick
      // This simulates the recording being stopped by the time the translation completes
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;
      appAny.recorderState = { ...appAny.recorderState, isRecording: false };

      // Now wait for the chunk translation to complete
      await app.waitForPendingOperations();

      // Recording state should still be false
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles chunk translation error when recording already stopped", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the branch where recording is stopped before
      // the chunk translation error is handled (line 460 false branch)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeErrorResponse(500, "Server error"), // chunk translation error
          createFakeResponse({ text: "Final text" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Trigger a chunk (starts async translation that will fail)
      recorder.requestData();

      // Directly set recording state to false WITHOUT going through handleRecordClick
      // This simulates the recording being stopped by the time the error is handled
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;
      appAny.recorderState = { ...appAny.recorderState, isRecording: false };

      // Now wait for the chunk translation error to be handled
      await app.waitForPendingOperations();

      // Recording state should still be false, no crash from the error
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles header extraction failure gracefully", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the catch block when extractWebmHeader fails (lines 417-419)
      // Note: decodeError affects both extractWebmHeader AND fixWebmBlob, so both will fail
      // But the point is to verify the catch block runs and tries doChunkTranslation
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Final text" }), // final translation (after muxer reset)
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
        webmMuxerConfig: {
          // Force decode to throw an error
          decodeError: new Error("Decode failed"),
        },
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Trigger a chunk - header extraction will fail
      // The catch block (lines 417-419) will be hit
      recorder.requestData();
      await app.waitForPendingOperations();

      // Status should show translation error (since fixWebmBlob also fails)
      const status = elements.get("status");
      expect(status?.textContent).toBe("Recording... (translation error, retrying)");

      // Reset hooks without error to allow final translation
      fakes = createFakeHooks({
        fetchResponses: [createFakeResponse({ text: "Final text" })],
        elements,
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      setHooks(fakes.hooks);

      // Stop recording
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("skips chunk when header not ready", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the branch where chunk 2+ arrives before header is extracted
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Final text" }), // final translation only
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Manually set webmHeader to null and simulate a chunk with index 2
      // This shouldn't happen normally but tests the defensive branch
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;
      appAny.webmHeader = null;

      // Call handleChunkReceived directly with chunk index 2 (no header available)
      const fakeChunk = new Blob(["audio data"], { type: "audio/webm" });
      appAny.handleChunkReceived(
        fakeChunk,
        2, // chunk index 2, but no header
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // No translation should happen for this chunk (skipped)
      // But final translation should work
      await app.handleRecordClick();

      expect(elements.get("transcript")?.textContent).toContain("Final text");

      globalThis.AudioContext = savedAudioContext;
    });

    it("translates chunk 2+ with prepended header (raw chunk translation)", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the raw chunk translation path for chunks 2+
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Chunk 2 text" }), // chunk 2 raw translation (no fixWebmBlob)
          createFakeResponse({ text: "Final" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Pre-set a header (normally extracted from chunk 1)
      appAny.webmHeader = new Uint8Array([0x18, 0x53, 0x80, 0x67]).buffer;
      appAny.currentSessionIndex = 0;
      appAny.transcripts = ["Chunk 1 text"];

      // Call handleChunkReceived with chunk index 2 (header available)
      // This uses the raw chunk translation path (no fixWebmBlob)
      const fakeChunk = new Blob(["audio data chunk 2"], { type: "audio/webm" });
      appAny.handleChunkReceived(
        fakeChunk,
        2, // chunk index 2, header available
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Wait for the raw chunk translation to complete
      await app.waitForPendingOperations();

      // Chunk 2 text should be appended to existing transcript
      expect(appAny.transcripts[0]).toBe("Chunk 1 text Chunk 2 text");

      // Stop recording
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("starts new transcript when raw chunk translation is first success", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the branch where currentSessionIndex is -1 during raw chunk translation
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "First successful chunk" }), // chunk 2 is first success
          createFakeResponse({ text: "Final" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Pre-set header but leave currentSessionIndex at -1 (no prior successful translations)
      appAny.webmHeader = new Uint8Array([0x18, 0x53, 0x80, 0x67]).buffer;
      // currentSessionIndex is already -1 by default, and transcripts is empty

      // Call handleChunkReceived with chunk index 2 (header available)
      const fakeChunk = new Blob(["audio data chunk 2"], { type: "audio/webm" });
      appAny.handleChunkReceived(
        fakeChunk,
        2,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Wait for the raw chunk translation to complete
      await app.waitForPendingOperations();

      // Should have created a new transcript entry
      expect(appAny.transcripts.length).toBe(1);
      expect(appAny.transcripts[0]).toBe("First successful chunk");
      expect(appAny.currentSessionIndex).toBe(0);

      // Stop recording
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("appends text when doChunkTranslation is called with existing transcript", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the else branch in doChunkTranslation (appending to existing transcript)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Appended text" }), // will be appended
          createFakeResponse({ text: "Final" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Pre-set transcript and index to simulate a prior successful translation
      appAny.transcripts = ["First part"];
      appAny.currentSessionIndex = 0;

      // Directly call doChunkTranslation (normally only called for chunk 1)
      const fakeChunk = new Blob(["audio"], { type: "audio/webm" });
      await appAny.doChunkTranslation(
        fakeChunk,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Text should be appended
      expect(appAny.transcripts[0]).toBe("First part Appended text");

      // Stop recording
      await app.handleRecordClick();

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles raw chunk translation error gracefully", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the error handling in raw chunk translation
      // When a chunk translation fails, the app falls back to translating the full blob on stop
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ status: 500, ok: false }), // chunk 2 translation fails
          createFakeResponse({ text: "Full recording translation" }), // fallback full blob translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Pre-set a header and transcript (simulating a successful first chunk)
      appAny.webmHeader = new Uint8Array([0x18, 0x53, 0x80, 0x67]).buffer;
      appAny.currentSessionIndex = 0;
      appAny.transcripts = ["First"];
      appAny.updateTranscriptDisplay();

      // Call handleChunkReceived with chunk index 2
      const fakeChunk = new Blob(["audio data"], { type: "audio/webm" });
      appAny.handleChunkReceived(
        fakeChunk,
        2,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Wait for the raw chunk translation to complete (with error)
      await app.waitForPendingOperations();

      // Transcript should be unchanged (error path)
      expect(appAny.transcripts[0]).toBe("First");
      // chunkTranslationFailed should be set
      expect(appAny.chunkTranslationFailed).toBe(true);

      // Status should indicate error
      expect(elements.get("status")?.textContent).toContain("translation error");

      // Stop recording - since chunk failed, falls back to full blob translation
      await app.handleRecordClick();

      // Full blob translation replaces the partial transcript
      expect(elements.get("transcript")?.textContent).toBe("Full recording translation");

      globalThis.AudioContext = savedAudioContext;
    });

    it("does not update status after raw chunk success when not recording", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Chunk text" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Directly call doRawChunkTranslation while NOT recording
      appAny.webmHeader = new Uint8Array([0x18, 0x53, 0x80, 0x67]).buffer;
      appAny.recorderState = { isRecording: false, mediaRecorder: null, audioChunks: [] };
      appAny.currentSessionIndex = 0;
      appAny.transcripts = ["First"];

      const fakeChunk = new Blob(["audio"], { type: "audio/webm" });
      await appAny.doRawChunkTranslation(
        fakeChunk,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Status should NOT say "Recording..." since we're not recording
      expect(elements.get("status")?.textContent).toBe("Translating...");

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles raw chunk error when not recording", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ status: 500, ok: false }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Set up state but NOT recording
      appAny.recorderState = { isRecording: false, mediaRecorder: null, audioChunks: [] };
      appAny.transcripts = [];

      const fakeChunk = new Blob(["audio"], { type: "audio/webm" });
      await appAny.doRawChunkTranslation(
        fakeChunk,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // Status should NOT say "translation error, retrying" since we're not recording
      expect(elements.get("status")?.textContent).toBe("Translating...");

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles undefined transcript entry with nullish coalescing", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "New text" }),
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const appAny = app as any;

      // Set up state where transcript[0] is undefined but array has length 1
      // This triggers the ?? "" fallback in the else branch
      appAny.transcripts = [undefined as unknown as string];
      appAny.currentSessionIndex = 0; // Points to transcripts[0] which is undefined

      // Directly call doChunkTranslation - will hit else branch since 0 < length(1)
      const fakeChunk = new Blob(["audio"], { type: "audio/webm" });
      await appAny.doChunkTranslation(
        fakeChunk,
        appAny.elements,
        appAny.config,
        appAny.token
      );

      // The ?? "" fallback means existing = "" + " " + text = " New text"
      expect(appAny.transcripts[0]).toBe(" New text");

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles undefined transcript entry gracefully", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      // This test covers the defensive check for undefined transcript (line 447)
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "First chunk" }), // first chunk
          createFakeResponse({ text: "Second chunk" }), // second chunk
          createFakeResponse({ text: "Final" }), // final translation
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Start recording
      await app.handleRecordClick();

      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }

      // Trigger first chunk to create a transcript entry
      recorder.requestData();
      await app.waitForPendingOperations();

      // Manually corrupt the state: set valid index but undefined transcript
      Object.assign(app, {
        currentSessionIndex: 0,
        transcripts: [undefined as unknown as string],
      });

      // Trigger second chunk - should handle undefined gracefully
      recorder.requestData();
      await app.waitForPendingOperations();

      // Stop recording
      await app.handleRecordClick();

      // Should not crash
      expect(app.getState().isRecording).toBe(false);

      globalThis.AudioContext = savedAudioContext;
    });
  });

  describe("DOM event handlers", () => {
    it("handles record button click with error via event listener", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
        getUserMediaError: new Error("Permission denied"),
      });
      const app = createApp();
      await app.init();

      // Get the record button and click it to trigger the event listener
      const recordBtn = elements.get("record-btn");
      expect(recordBtn).toBeDefined();

      // The event listener catches the error and updates status
      // Use a promise to wait for the async handler to complete
      const clickPromise = new Promise<void>((resolve) => {
        recordBtn?.click();
        // Allow microtask queue to flush
        queueMicrotask(resolve);
      });
      await clickPromise;
      // Wait one more tick for promise rejection handler
      await Promise.resolve();

      const status = elements.get("status");
      expect(status?.textContent).toBe("Microphone access denied");
      expect(status?.classList.contains("error")).toBe(true);
    });

    it("handles clear button click via event listener", async () => {
      // Remove AudioContext to avoid RAF loop
      const savedAudioContext = globalThis.AudioContext;
      // @ts-expect-error - intentionally removing for test
      delete globalThis.AudioContext;

      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
          createFakeResponse({ text: "Hello" }), // chunk translation
          createFakeResponse({ text: "Hello final" }), // final translation on stop
        ],
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      // Add a transcript via recording with chunk
      await app.handleRecordClick();

      // Trigger chunk to get translation
      const recorders = fakes.getMediaRecorderInstances();
      if (recorders.length === 0) {
        throw new Error("No MediaRecorder instance found");
      }
      const recorder = recorders[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance is undefined");
      }
      recorder.requestData();
      await app.waitForPendingOperations();

      await app.handleRecordClick();

      expect(app.getState().transcripts.length).toBe(1);

      // Click the clear button via DOM event
      const clearBtn = elements.get("clear-btn");
      clearBtn?.click();

      expect(app.getState().transcripts.length).toBe(0);

      globalThis.AudioContext = savedAudioContext;
    });

    it("handles login form submit via event listener", async () => {
      setupHooks({
        fetchResponses: [
          createFakeResponse({ API_BASE_URL: "https://api.test.com" }),
        ],
      });
      const app = createApp();
      await app.init();

      // Set token value
      const tokenInput = elements.get("token-input") as HTMLInputElement;
      tokenInput.value = "test-token";

      // Submit the form via DOM event - handleLogin is synchronous
      const loginForm = elements.get("login-form") as HTMLFormElement;
      const submitEvent = new Event("submit", { cancelable: true });
      loginForm.dispatchEvent(submitEvent);

      // State should be updated immediately (handleLogin is synchronous)
      expect(app.getState().token).toBe("test-token");
    });

    it("handles logout button click via event listener", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
      });
      const app = createApp();
      await app.init();

      const state = app.getState();
      expect(state.token).toBe("token");

      // Click logout via DOM event - handleLogout is synchronous
      const logoutBtn = elements.get("logout-btn");
      logoutBtn?.click();

      // State should be updated immediately (handleLogout is synchronous)
      expect(app.getState().token).toBeNull();
    });

    it("handles non-Permission denied error message", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
        getUserMediaError: new Error("Device not available"),
      });
      const app = createApp();
      await app.init();

      // Click record button to trigger error
      const recordBtn = elements.get("record-btn");
      const clickPromise = new Promise<void>((resolve) => {
        recordBtn?.click();
        queueMicrotask(resolve);
      });
      await clickPromise;
      await Promise.resolve();

      const status = elements.get("status");
      expect(status?.textContent).toBe("Device not available");
    });

    it("handles non-Error throw with fallback message", async () => {
      setupHooks({
        initialStorage: new Map([["grandma_token", "token"]]),
        getUserMediaThrow: "string error",
      });
      const app = createApp();
      await app.init();

      // Click record button to trigger error
      const recordBtn = elements.get("record-btn");
      const clickPromise = new Promise<void>((resolve) => {
        recordBtn?.click();
        queueMicrotask(resolve);
      });
      await clickPromise;
      await Promise.resolve();

      const status = elements.get("status");
      expect(status?.textContent).toBe("Recording failed");
    });
  });

  describe("autoInit error handling", () => {
    it("displays error when init fails", async () => {
      // Setup with a fetch that will fail
      setupHooks({
        fetchResponses: [], // No responses, will throw
      });

      autoInit();

      // Wait for async init to complete (fetch throws -> error handler runs)
      const loginError = elements.get("login-error");
      await vi.waitFor(
        () => {
          expect(loginError?.textContent).toContain("No fake response configured");
        },
        { timeout: 5000 }
      );
      expect(loginError?.classList.contains("hidden")).toBe(false);
    });

    it("displays fallback message when init throws non-Error", async () => {
      // Need to directly call createFakeHooks with fetchThrow
      fakes = createFakeHooks({
        fetchThrow: "string error",
        elements,
      });
      setHooks(fakes.hooks);

      autoInit();

      const loginError = elements.get("login-error");
      await vi.waitFor(
        () => {
          expect(loginError?.textContent).toBe("Failed to load config");
        },
        { timeout: 5000 }
      );
      expect(loginError?.classList.contains("hidden")).toBe(false);
    });

    it("handles missing login-error element gracefully", async () => {
      // Ensure real timers are used
      vi.useRealTimers();

      // Remove login-error element
      elements.delete("login-error");
      fakes = createFakeHooks({
        fetchThrow: "string error",
        elements,
      });
      setHooks(fakes.hooks);

      // Should not throw even if element is missing
      autoInit();

      // Wait for the async error handling to complete
      await new Promise((resolve) => setTimeout(resolve, 50));

      // Verify the element is still missing (no error thrown)
      expect(elements.has("login-error")).toBe(false);
    });
  });
});
