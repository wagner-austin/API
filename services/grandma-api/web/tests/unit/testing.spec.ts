import { describe, it, expect } from "vitest";
import {
  createFakeResponse,
  createFakeErrorResponse,
  createFakeStorage,
  FakeMediaRecorder,
  FakeMediaStream,
  FakeMediaStreamTrack,
  createFakeDocument,
  createFakeHooks,
  createTestElements,
  waitFor,
  waitForAsync,
} from "../../src/testing.js";

describe("testing utilities", () => {
  describe("createFakeResponse", () => {
    it("creates JSON response with default status 200", async () => {
      const response = createFakeResponse({ text: "hello" });

      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toBe("application/json");

      const body = await response.json();
      expect(body).toEqual({ text: "hello" });
    });

    it("creates JSON response with custom status", async () => {
      const response = createFakeResponse({ error: true }, 201);

      expect(response.status).toBe(201);
    });
  });

  describe("createFakeErrorResponse", () => {
    it("creates error response without detail", () => {
      const response = createFakeErrorResponse(500);

      expect(response.status).toBe(500);
    });

    it("creates error response with detail", async () => {
      const response = createFakeErrorResponse(401, "Invalid token");

      expect(response.status).toBe(401);
      const body = await response.json();
      expect(body).toEqual({ detail: "Invalid token" });
    });
  });

  describe("createFakeStorage", () => {
    it("implements full StorageInterface", () => {
      const { storage, getData } = createFakeStorage();

      storage.setItem("key", "value");
      expect(storage.getItem("key")).toBe("value");
      expect(getData().get("key")).toBe("value");

      storage.removeItem("key");
      expect(storage.getItem("key")).toBeNull();

      storage.setItem("a", "1");
      storage.setItem("b", "2");
      storage.clear();
      expect(getData().size).toBe(0);
    });
  });

  describe("FakeMediaRecorder", () => {
    it("records start/stop calls", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      expect(recorder.state).toBe("inactive");
      expect(recorder.calls).toHaveLength(0);

      recorder.start();
      expect(recorder.state).toBe("recording");
      expect(recorder.calls).toHaveLength(1);
      expect(recorder.calls[0]?.method).toBe("start");

      recorder.stop();
      expect(recorder.state).toBe("inactive");
      expect(recorder.calls).toHaveLength(2);
      expect(recorder.calls[1]?.method).toBe("stop");
    });

    it("emits events on start/stop", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      let startCalled = false;
      let stopCalled = false;
      let dataAvailable = false;

      recorder.onstart = () => {
        startCalled = true;
      };
      recorder.onstop = () => {
        stopCalled = true;
      };
      recorder.ondataavailable = () => {
        dataAvailable = true;
      };

      recorder.start();
      expect(startCalled).toBe(true);

      recorder.stop();
      expect(stopCalled).toBe(true);
      expect(dataAvailable).toBe(true);
    });

    it("handles pause/resume", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      recorder.start();
      recorder.pause();
      expect(recorder.state).toBe("paused");

      recorder.resume();
      expect(recorder.state).toBe("recording");
    });

    it("calls onpause/onresume handlers", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      let pauseCalled = false;
      let resumeCalled = false;

      recorder.onpause = () => {
        pauseCalled = true;
      };
      recorder.onresume = () => {
        resumeCalled = true;
      };

      recorder.start();
      recorder.pause();
      expect(pauseCalled).toBe(true);

      recorder.resume();
      expect(resumeCalled).toBe(true);
    });

    it("handles requestData", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);
      let dataCalled = false;

      recorder.ondataavailable = () => {
        dataCalled = true;
      };

      recorder.requestData();
      expect(dataCalled).toBe(true);
    });

    it("handles requestData without handler", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      // Should not throw when handler is not set
      expect(() => recorder.requestData()).not.toThrow();
    });

    it("handles requestEmptyData", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);
      let dataBlob: Blob | null = null;

      recorder.ondataavailable = (e: BlobEvent) => {
        dataBlob = e.data;
      };

      recorder.requestEmptyData();

      if (dataBlob === null) {
        throw new Error("dataBlob was not set");
      }
      const blob: Blob = dataBlob;
      expect(blob.size).toBe(0);
    });

    it("handles requestEmptyData without handler", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      // Should not throw when handler is not set
      expect(() => recorder.requestEmptyData()).not.toThrow();
    });

    it("supports isTypeSupported static method", () => {
      expect(FakeMediaRecorder.isTypeSupported("audio/webm")).toBe(true);
    });

    it("has working EventTarget methods", () => {
      const stream = new FakeMediaStream();
      const recorder = new FakeMediaRecorder(stream);

      // These are no-ops but should not throw
      recorder.addEventListener("test", () => {});
      recorder.removeEventListener("test", () => {});
      expect(recorder.dispatchEvent(new Event("test"))).toBe(true);
    });
  });

  describe("FakeMediaStreamTrack", () => {
    it("has correct initial state", () => {
      const track = new FakeMediaStreamTrack();

      expect(track.kind).toBe("audio");
      expect(track.enabled).toBe(true);
      expect(track.muted).toBe(false);
      expect(track.readyState).toBe("live");
      expect(track.stopped).toBe(false);
    });

    it("tracks stop calls", () => {
      const track = new FakeMediaStreamTrack();

      track.stop();
      expect(track.stopped).toBe(true);
    });

    it("supports clone", () => {
      const track = new FakeMediaStreamTrack();
      const clone = track.clone();

      expect(clone).toBeInstanceOf(FakeMediaStreamTrack);
      expect(clone).not.toBe(track);
    });

    it("returns empty capability/constraint objects", () => {
      const track = new FakeMediaStreamTrack();

      expect(track.getCapabilities()).toEqual({});
      expect(track.getConstraints()).toEqual({});
      expect(track.getSettings()).toEqual({});
    });

    it("applyConstraints resolves", async () => {
      const track = new FakeMediaStreamTrack();
      await expect(track.applyConstraints()).resolves.toBeUndefined();
    });

    it("has working EventTarget methods", () => {
      const track = new FakeMediaStreamTrack();

      track.addEventListener("test", () => {});
      track.removeEventListener("test", () => {});
      expect(track.dispatchEvent(new Event("test"))).toBe(true);
    });
  });

  describe("FakeMediaStream", () => {
    it("has correct initial state", () => {
      const stream = new FakeMediaStream();

      expect(stream.id).toBe("fake-stream-id");
      expect(stream.active).toBe(true);
    });

    it("returns tracks", () => {
      const track = new FakeMediaStreamTrack();
      const stream = new FakeMediaStream([track]);

      expect(stream.getTracks()).toEqual([track]);
      expect(stream.getAudioTracks()).toEqual([track]);
      expect(stream.getVideoTracks()).toEqual([]);
    });

    it("getTrackById returns first track", () => {
      const stream = new FakeMediaStream();

      expect(stream.getTrackById("any")).toBeInstanceOf(FakeMediaStreamTrack);
    });

    it("getTrackById returns null for empty stream", () => {
      const stream = new FakeMediaStream([]);

      expect(stream.getTrackById("any")).toBeNull();
    });

    it("supports clone", () => {
      const stream = new FakeMediaStream();
      const clone = stream.clone();

      expect(clone).toBeInstanceOf(FakeMediaStream);
      expect(clone).not.toBe(stream);
    });

    it("has working EventTarget methods", () => {
      const stream = new FakeMediaStream();

      stream.addEventListener("test", () => {});
      stream.removeEventListener("test", () => {});
      expect(stream.dispatchEvent(new Event("test"))).toBe(true);
    });

    it("has no-op addTrack/removeTrack methods", () => {
      const stream = new FakeMediaStream();
      const track = new FakeMediaStreamTrack();

      // These are no-ops but should not throw
      expect(() => stream.addTrack(track)).not.toThrow();
      expect(() => stream.removeTrack(track)).not.toThrow();
    });
  });

  describe("createFakeDocument", () => {
    it("returns elements from map", () => {
      const elements = new Map<string, HTMLElement>();
      const div = document.createElement("div");
      elements.set("test-id", div);

      const doc = createFakeDocument(elements);

      expect(doc.getElementById("test-id")).toBe(div);
      expect(doc.getElementById("missing")).toBeNull();
    });

    it("has configurable readyState", () => {
      const elements = new Map<string, HTMLElement>();

      const loadingDoc = createFakeDocument(elements, "loading");
      expect(loadingDoc.readyState).toBe("loading");

      const completeDoc = createFakeDocument(elements, "complete");
      expect(completeDoc.readyState).toBe("complete");
    });

    it("supports addEventListener", () => {
      const elements = new Map<string, HTMLElement>();
      const doc = createFakeDocument(elements);

      // Should not throw
      doc.addEventListener("DOMContentLoaded", () => {});
    });
  });

  describe("createFakeHooks", () => {
    it("creates complete hooks object", () => {
      const { hooks } = createFakeHooks();

      expect(hooks.fetch).toBeDefined();
      expect(hooks.getUserMedia).toBeDefined();
      expect(hooks.MediaRecorder).toBeDefined();
      expect(hooks.storage).toBeDefined();
      expect(hooks.document).toBeDefined();
    });

    it("tracks fetch calls", async () => {
      const { hooks, getFetchCalls } = createFakeHooks({
        fetchResponses: [createFakeResponse({ ok: true })],
      });

      await hooks.fetch("https://example.com/api", { method: "POST" });

      const calls = getFetchCalls();
      expect(calls).toHaveLength(1);
      expect(calls[0]?.input).toBe("https://example.com/api");
      expect(calls[0]?.init?.method).toBe("POST");
    });

    it("returns configured fetch responses in order", async () => {
      const { hooks } = createFakeHooks({
        fetchResponses: [
          createFakeResponse({ first: true }),
          createFakeResponse({ second: true }),
        ],
      });

      const resp1 = await hooks.fetch("/first");
      const resp2 = await hooks.fetch("/second");

      expect(await resp1.json()).toEqual({ first: true });
      expect(await resp2.json()).toEqual({ second: true });
    });

    it("throws when no fetch response configured", async () => {
      const { hooks } = createFakeHooks({ fetchResponses: [] });

      await expect(hooks.fetch("/test")).rejects.toThrow(
        "No fake response configured for fetch call 1"
      );
    });

    it("initializes storage with provided data", () => {
      const { getStorageData } = createFakeHooks({
        initialStorage: new Map([["key", "value"]]),
      });

      expect(getStorageData().get("key")).toBe("value");
    });

    it("tracks MediaRecorder instances", async () => {
      const { hooks, getMediaRecorderInstances } = createFakeHooks();

      const stream = await hooks.getUserMedia({ audio: true });
      new hooks.MediaRecorder(stream);

      expect(getMediaRecorderInstances()).toHaveLength(1);
    });

    it("tracks MediaStream tracks", async () => {
      const { hooks, getMediaStreamTracks } = createFakeHooks();

      await hooks.getUserMedia({ audio: true });

      expect(getMediaStreamTracks()).toHaveLength(1);
    });

    it("getUserMedia throws configured error", async () => {
      const { hooks } = createFakeHooks({
        getUserMediaResult: new Error("Permission denied"),
      });

      await expect(hooks.getUserMedia({ audio: true })).rejects.toThrow("Permission denied");
    });

    it("getUserMedia returns configured stream", async () => {
      const customStream = new FakeMediaStream();
      const { hooks } = createFakeHooks({
        getUserMediaResult: customStream,
      });

      const stream = await hooks.getUserMedia({ audio: true });
      expect(stream).toBe(customStream);
    });

    it("uses provided elements for document", () => {
      const div = document.createElement("div");
      const elements = new Map<string, HTMLElement>([["my-element", div]]);

      const { hooks } = createFakeHooks({ elements });

      expect(hooks.document.getElementById("my-element")).toBe(div);
    });

    it("uses provided document ready state", () => {
      const { hooks } = createFakeHooks({ documentReadyState: "loading" });

      expect(hooks.document.readyState).toBe("loading");
    });
  });

  describe("createTestElements", () => {
    it("creates all required elements", () => {
      const elements = createTestElements();

      expect(elements.get("login-section")).toBeDefined();
      expect(elements.get("main-section")).toBeDefined();
      expect(elements.get("login-form")).toBeDefined();
      expect(elements.get("token-input")).toBeDefined();
      expect(elements.get("login-error")).toBeDefined();
      expect(elements.get("logout-btn")).toBeDefined();
      expect(elements.get("record-btn")).toBeDefined();
      expect(elements.get("record-icon")).toBeDefined();
      expect(elements.get("status")).toBeDefined();
      expect(elements.get("transcript")).toBeDefined();
      expect(elements.get("clear-btn")).toBeDefined();
    });

    it("creates elements with correct types", () => {
      const elements = createTestElements();

      expect(elements.get("login-form")?.tagName).toBe("FORM");
      expect(elements.get("token-input")?.tagName).toBe("INPUT");
      expect(elements.get("logout-btn")?.tagName).toBe("BUTTON");
    });

    it("creates elements with correct initial classes", () => {
      const elements = createTestElements();

      expect(elements.get("login-error")?.className).toBe("hidden");
      expect(elements.get("main-section")?.className).toBe("hidden");
    });
  });

  describe("waitFor", () => {
    it("resolves immediately when condition is true", async () => {
      await expect(waitFor(() => true)).resolves.toBeUndefined();
    });

    it("waits for condition to become true", async () => {
      let ready = false;
      setTimeout(() => {
        ready = true;
      }, 50);

      await expect(waitFor(() => ready, 1000)).resolves.toBeUndefined();
    });

    it("throws on timeout", async () => {
      await expect(waitFor(() => false, 50)).rejects.toThrow("waitFor timed out after 50ms");
    });
  });

  describe("waitForAsync", () => {
    it("resolves immediately when condition is true", async () => {
      await expect(waitForAsync(async () => true)).resolves.toBeUndefined();
    });

    it("waits for async condition to become true", async () => {
      let ready = false;
      setTimeout(() => {
        ready = true;
      }, 50);

      await expect(waitForAsync(async () => ready, 1000)).resolves.toBeUndefined();
    });

    it("throws on timeout", async () => {
      await expect(waitForAsync(async () => false, 50)).rejects.toThrow(
        "waitForAsync timed out after 50ms"
      );
    });
  });
});
