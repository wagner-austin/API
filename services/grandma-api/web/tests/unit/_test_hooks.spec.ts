import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  setHooks,
  getHooks,
  resetHooks,
  getFetch,
  getGetUserMedia,
  getMediaRecorderConstructor,
  getStorage,
  getDocument,
  getWebmMuxer,
  initTsEbml,
  resetTsEbml,
  Hooks,
  DocumentInterface,
} from "../../src/_test_hooks.js";
import {
  createFakeStorage,
  FakeMediaRecorder,
  FakeMediaStream,
  createFakeWebmMuxer,
} from "../../src/testing.js";

describe("_test_hooks", () => {
  beforeEach(() => {
    resetHooks();
  });

  afterEach(() => {
    resetHooks();
  });

  describe("setHooks/getHooks", () => {
    it("returns default hooks when not set", () => {
      // In jsdom, some browser globals like MediaRecorder are not available
      // This test verifies the hooks object structure when defaults are used
      const hooks = getHooks();

      expect(hooks.fetch).toBeDefined();
      expect(hooks.getUserMedia).toBeDefined();
      // MediaRecorder may be undefined in jsdom, which is expected
      expect("MediaRecorder" in hooks).toBe(true);
      expect(hooks.storage).toBeDefined();
      expect(hooks.document).toBeDefined();
    });

    it("default getUserMedia calls navigator.mediaDevices.getUserMedia", () => {
      // Test that the default getUserMedia hook calls the browser API
      const hooks = getHooks();

      // jsdom doesn't implement mediaDevices, so accessing it throws TypeError
      // but we verify the hook attempts to call the real API
      expect(() => hooks.getUserMedia({ audio: true })).toThrow(TypeError);
    });

    it("returns custom hooks when set", () => {
      const customFetch = async (_input: string | URL): Promise<Response> => {
        return new Response("custom");
      };

      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();

      const customHooks: Hooks = {
        fetch: customFetch,
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      };

      setHooks(customHooks);
      const hooks = getHooks();

      expect(hooks.fetch).toBe(customFetch);
      expect(hooks.storage).toBe(storage);
    });
  });

  describe("resetHooks", () => {
    it("clears custom hooks and returns to defaults", () => {
      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      const customHooks: Hooks = {
        fetch: async () => new Response(),
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      };

      setHooks(customHooks);
      expect(getHooks().storage).toBe(storage);

      resetHooks();
      expect(getHooks().storage).not.toBe(storage);
    });
  });

  describe("individual accessors", () => {
    it("getFetch returns fetch hook", () => {
      const customFetch = async (_input: string | URL): Promise<Response> => {
        return new Response("test");
      };

      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      setHooks({
        fetch: customFetch,
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      });

      expect(getFetch()).toBe(customFetch);
    });

    it("getGetUserMedia returns getUserMedia hook", () => {
      const customGetUserMedia = async (): Promise<MediaStream> => {
        return new FakeMediaStream();
      };

      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      setHooks({
        fetch: async () => new Response(),
        getUserMedia: customGetUserMedia,
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      });

      expect(getGetUserMedia()).toBe(customGetUserMedia);
    });

    it("getMediaRecorderConstructor returns MediaRecorder hook", () => {
      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      setHooks({
        fetch: async () => new Response(),
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      });

      const Ctor = getMediaRecorderConstructor();
      expect(Ctor).toBeDefined();
    });

    it("getStorage returns storage hook", () => {
      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      setHooks({
        fetch: async () => new Response(),
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      });

      expect(getStorage()).toBe(storage);
    });

    it("getDocument returns document hook", () => {
      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();
      const customDocument: DocumentInterface = {
        getElementById: () => null,
        readyState: "complete",
        addEventListener: () => {},
      };

      setHooks({
        fetch: async () => new Response(),
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: customDocument,
        webmMuxer: muxer,
      });

      expect(getDocument()).toBe(customDocument);
    });

    it("getWebmMuxer returns webmMuxer hook", () => {
      const { storage } = createFakeStorage();
      const { muxer } = createFakeWebmMuxer();

      setHooks({
        fetch: async () => new Response(),
        getUserMedia: async () => new FakeMediaStream(),
        MediaRecorder: FakeMediaRecorder as unknown as new (
          stream: MediaStream,
          options?: MediaRecorderOptions
        ) => MediaRecorder,
        storage,
        document: {
          getElementById: () => null,
          readyState: "complete",
          addEventListener: () => {},
        },
        webmMuxer: muxer,
      });

      expect(getWebmMuxer()).toBe(muxer);
    });
  });

  describe("fake storage", () => {
    it("implements StorageInterface correctly", () => {
      const { storage } = createFakeStorage();

      // Test setItem/getItem
      storage.setItem("key1", "value1");
      expect(storage.getItem("key1")).toBe("value1");

      // Test getItem returns null for missing key
      expect(storage.getItem("missing")).toBeNull();

      // Test removeItem
      storage.removeItem("key1");
      expect(storage.getItem("key1")).toBeNull();

      // Test clear
      storage.setItem("a", "1");
      storage.setItem("b", "2");
      storage.clear();
      expect(storage.getItem("a")).toBeNull();
      expect(storage.getItem("b")).toBeNull();
    });

    it("allows inspection of stored data", () => {
      const { storage, getData } = createFakeStorage();

      storage.setItem("test", "data");
      const data = getData();

      expect(data.get("test")).toBe("data");
    });
  });

  describe("real WebM muxer", () => {
    it("initTsEbml loads ts-ebml module and caches it", async () => {
      await initTsEbml();
      // Second call should be cached (hit the early return)
      await initTsEbml();
    });

    it("real muxer works after initTsEbml", async () => {
      await initTsEbml();
      resetHooks(); // Force default hooks with real muxer
      const muxer = getWebmMuxer();

      // Create a minimal WebM buffer for testing
      // EBML header: 0x1A, 0x45, 0xDF, 0xA3 is the EBML element ID
      const webmData = new Uint8Array([
        0x1a, 0x45, 0xdf, 0xa3, // EBML element ID
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, // Size (31 bytes)
        0x42, 0x86, 0x81, 0x01, // EBMLVersion
        0x42, 0xf7, 0x81, 0x01, // EBMLReadVersion
        0x42, 0xf2, 0x81, 0x04, // EBMLMaxIDLength
        0x42, 0xf3, 0x81, 0x08, // EBMLMaxSizeLength
        0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6d, // DocType "webm"
        0x42, 0x87, 0x81, 0x04, // DocTypeVersion
        0x42, 0x85, 0x81, 0x02, // DocTypeReadVersion
      ]);

      // Decode should work
      const elements = muxer.decode(webmData.buffer);
      expect(Array.isArray(elements)).toBe(true);
      expect(elements.length).toBeGreaterThan(0);

      // processElements should work
      const result = muxer.processElements(elements);
      expect(result).toHaveProperty("metadatas");
      expect(result).toHaveProperty("duration");
      expect(result).toHaveProperty("cues");
      expect(result).toHaveProperty("metadataSize");

      // makeMetadataSeekable should work
      const refined = muxer.makeMetadataSeekable(result.metadatas, result.duration, result.cues);
      // Use Object.prototype.toString for cross-realm ArrayBuffer check
      const typeString = Object.prototype.toString.call(refined);
      expect(typeString).toBe("[object ArrayBuffer]");
    });

    it("real muxer decode reuses decoder on second call", async () => {
      await initTsEbml();
      resetHooks();
      const muxer = getWebmMuxer();

      const webmData = new Uint8Array([
        0x1a, 0x45, 0xdf, 0xa3, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f,
        0x42, 0x86, 0x81, 0x01, 0x42, 0xf7, 0x81, 0x01, 0x42, 0xf2, 0x81, 0x04,
        0x42, 0xf3, 0x81, 0x08, 0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6d,
        0x42, 0x87, 0x81, 0x04, 0x42, 0x85, 0x81, 0x02,
      ]);

      // First call creates decoder
      muxer.decode(webmData.buffer);
      // Second call reuses decoder (hits the decoder !== null branch)
      const elements = muxer.decode(webmData.buffer);
      expect(Array.isArray(elements)).toBe(true);
    });

    it("decode throws when ts-ebml not loaded", async () => {
      // Reset ts-ebml module to test error branch
      resetTsEbml();
      resetHooks(); // Force default hooks with real muxer
      const muxer = getWebmMuxer();

      const buffer = new ArrayBuffer(10);
      expect(() => muxer.decode(buffer)).toThrow(
        "EBML global not found. Ensure ts-ebml script is loaded."
      );

      // Restore for subsequent tests
      await initTsEbml();
    });

    it("processElements throws when ts-ebml not loaded", async () => {
      // Reset ts-ebml module to test error branch
      resetTsEbml();
      resetHooks();
      const muxer = getWebmMuxer();

      expect(() => muxer.processElements([])).toThrow(
        "EBML global not found. Ensure ts-ebml script is loaded."
      );

      // Restore for subsequent tests
      await initTsEbml();
    });

    it("makeMetadataSeekable throws when ts-ebml not loaded", async () => {
      // Reset ts-ebml module to test error branch
      resetTsEbml();
      resetHooks();
      const muxer = getWebmMuxer();

      expect(() => muxer.makeMetadataSeekable([], 0, [])).toThrow(
        "EBML global not found. Ensure ts-ebml script is loaded."
      );

      // Restore for subsequent tests
      await initTsEbml();
    });

    it("initTsEbml uses global EBML when available (browser path)", async () => {
      // Reset module state
      resetTsEbml();

      // Simulate browser environment with global EBML
      const mockEBML = {
        Decoder: class {
          decode(): unknown[] {
            return [];
          }
        },
        Reader: class {
          metadatas: unknown[] = [];
          duration = 0;
          cues: unknown[] = [];
          metadataSize = 0;
          read(): void {
            /* no-op */
          }
          stop(): void {
            /* no-op */
          }
        },
        tools: {
          makeMetadataSeekable(): ArrayBuffer {
            return new ArrayBuffer(0);
          },
        },
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (globalThis as any).EBML = mockEBML;

      try {
        // This should use the global EBML instead of dynamic import
        await initTsEbml();

        // Verify we can use the muxer (it uses the cached module)
        resetHooks();
        const muxer = getWebmMuxer();
        const result = muxer.decode(new ArrayBuffer(0));
        expect(result).toEqual([]);
      } finally {
        // Clean up global
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).EBML;
        // Restore real ts-ebml for subsequent tests
        resetTsEbml();
        await initTsEbml();
      }
    });
  });
});
