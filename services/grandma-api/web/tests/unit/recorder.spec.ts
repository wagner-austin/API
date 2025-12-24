import { describe, it, expect, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks } from "../../src/testing.js";
import { createRecorderState, startRecording, stopRecording } from "../../src/recorder.js";

describe("recorder", () => {
  afterEach(() => {
    resetHooks();
  });

  describe("createRecorderState", () => {
    it("creates initial state", () => {
      const state = createRecorderState();

      expect(state.mediaRecorder).toBeNull();
      expect(state.audioChunks).toEqual([]);
      expect(state.isRecording).toBe(false);
    });
  });

  describe("startRecording", () => {
    it("starts recording and returns state with stopPromise", async () => {
      const { hooks, getMediaRecorderInstances, getMediaStreamTracks } = createFakeHooks();
      setHooks(hooks);

      const state = createRecorderState();
      const result = await startRecording(state);

      expect(result.state.isRecording).toBe(true);
      expect(result.state.mediaRecorder).not.toBeNull();
      expect(result.stopPromise).toBeInstanceOf(Promise);
      expect(getMediaRecorderInstances()).toHaveLength(1);
      expect(getMediaStreamTracks()).toHaveLength(1);
    });

    it("stopPromise resolves with blob when recording stops", async () => {
      const { hooks, getMediaRecorderInstances, getMediaStreamTracks } = createFakeHooks();
      setHooks(hooks);

      const state = createRecorderState();
      const result = await startRecording(state);

      // Stop recording
      const recorder = getMediaRecorderInstances()[0];
      expect(recorder).toBeDefined();
      recorder?.stop();

      // Wait for stopPromise to resolve
      const blob = await result.stopPromise;

      expect(blob).toBeInstanceOf(Blob);
      expect(getMediaStreamTracks()[0]?.stopped).toBe(true);
    });

    it("throws when already recording", async () => {
      const { hooks } = createFakeHooks();
      setHooks(hooks);

      const state = createRecorderState();
      const result = await startRecording(state);

      await expect(startRecording(result.state)).rejects.toThrow("Already recording");
    });

    it("throws when microphone access denied", async () => {
      const { hooks } = createFakeHooks({
        getUserMediaResult: new Error("Permission denied"),
      });
      setHooks(hooks);

      const state = createRecorderState();

      await expect(startRecording(state)).rejects.toThrow("Permission denied");
    });

    it("ignores empty data chunks", async () => {
      const { hooks, getMediaRecorderInstances } = createFakeHooks();
      setHooks(hooks);

      const chunks: Blob[] = [];
      const state = createRecorderState();
      const result = await startRecording(state, {
        chunkIntervalMs: 1000,
        onChunk: (blob) => {
          chunks.push(blob);
        },
      });

      // Send empty data - should be ignored
      const recorder = getMediaRecorderInstances()[0];
      if (recorder === undefined) {
        throw new Error("MediaRecorder instance not found");
      }
      recorder.requestEmptyData();

      // No chunks should have been processed
      expect(chunks.length).toBe(0);

      // Stop recording
      stopRecording(result.state);
      recorder.stop();
    });
  });

  describe("stopRecording", () => {
    it("stops recording and updates state", async () => {
      const { hooks, getMediaRecorderInstances } = createFakeHooks();
      setHooks(hooks);

      const state = createRecorderState();
      const result = await startRecording(state);
      const stoppedState = stopRecording(result.state);

      expect(stoppedState.isRecording).toBe(false);

      const recorder = getMediaRecorderInstances()[0];
      expect(recorder?.state).toBe("inactive");
    });

    it("handles already stopped state", () => {
      const state = createRecorderState();
      const stoppedState = stopRecording(state);

      expect(stoppedState.isRecording).toBe(false);
    });
  });
});
