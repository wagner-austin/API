import { describe, it, expect, afterEach } from "vitest";
import { setHooks, resetHooks } from "../../src/_test_hooks.js";
import { createFakeHooks } from "../../src/testing.js";
import { fixWebmBlob, extractWebmHeader, createWebmWithHeader } from "../../src/webm.js";

describe("webm", () => {
  afterEach(() => {
    resetHooks();
  });

  describe("fixWebmBlob", () => {
    it("processes blob and returns fixed version", async () => {
      const {
        hooks,
        getWebmMuxerDecodeInputs,
        getWebmMuxerProcessInputs,
        getWebmMuxerMakeSeekableInputs,
      } = createFakeHooks();
      setHooks(hooks);

      const inputBlob = new Blob(["test audio data"], { type: "audio/webm" });
      const fixedBlob = await fixWebmBlob(inputBlob);

      // Verify muxer methods were called
      expect(getWebmMuxerDecodeInputs()).toHaveLength(1);
      expect(getWebmMuxerProcessInputs()).toHaveLength(1);
      expect(getWebmMuxerMakeSeekableInputs()).toHaveLength(1);

      // Verify output blob is created
      expect(fixedBlob).toBeInstanceOf(Blob);
      expect(fixedBlob.type).toBe("audio/webm");
    });

    it("passes correct data through muxer pipeline", async () => {
      const customElements = [
        { name: "EBML", type: "m", tagStart: 0, tagEnd: 4, dataStart: 4, dataEnd: 10, dataSize: 6 },
      ];
      const customProcessResult = {
        metadatas: customElements,
        duration: 5000,
        cues: [],
        metadataSize: 200,
      };
      const {
        hooks,
        getWebmMuxerDecodeInputs,
        getWebmMuxerProcessInputs,
        getWebmMuxerMakeSeekableInputs,
      } = createFakeHooks({
        webmMuxerConfig: {
          elements: customElements,
          processResult: customProcessResult,
        },
      });
      setHooks(hooks);

      const inputBlob = new Blob(["audio"], { type: "audio/webm" });
      await fixWebmBlob(inputBlob);

      // Check decode was called with blob contents
      const decodeInputs = getWebmMuxerDecodeInputs();
      expect(decodeInputs).toHaveLength(1);
      expect(decodeInputs[0]).toBeInstanceOf(ArrayBuffer);

      // Check processElements received decode output
      const processInputs = getWebmMuxerProcessInputs();
      expect(processInputs).toHaveLength(1);
      expect(processInputs[0]).toHaveLength(1);

      // Check makeMetadataSeekable received process output
      const makeSeekableInputs = getWebmMuxerMakeSeekableInputs();
      expect(makeSeekableInputs).toHaveLength(1);
      expect(makeSeekableInputs[0]?.duration).toBe(5000);
    });
  });

  describe("extractWebmHeader", () => {
    it("extracts header based on metadataSize when no Segment found", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 100, // Header is first 100 bytes
      };
      const { hooks, getWebmMuxerDecodeInputs, getWebmMuxerProcessInputs } = createFakeHooks({
        webmMuxerConfig: {
          processResult: customProcessResult,
        },
      });
      setHooks(hooks);

      // Create a blob with no Segment element (just 0xAB bytes)
      const inputData = new Uint8Array(500).fill(0xab);
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      // Verify muxer methods were called
      expect(getWebmMuxerDecodeInputs()).toHaveLength(1);
      expect(getWebmMuxerProcessInputs()).toHaveLength(1);

      // Verify header is correct size (unchanged since no Segment to patch)
      expect(header.byteLength).toBe(100);
    });

    it("patches Segment size to unknown for 1-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // Create WebM with: EBML header (10 bytes) + Segment ID + 1-byte VINT size + data
      // Segment ID: 0x18 0x53 0x80 0x67
      // 1-byte VINT (e.g. 0x85 = size 5): first bit is 1
      const inputData = new Uint8Array(100);
      inputData.set([0x1a, 0x45, 0xdf, 0xa3, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0); // EBML header (fake)
      inputData.set([0x18, 0x53, 0x80, 0x67], 10); // Segment ID at position 10
      inputData.set([0x85], 14); // 1-byte VINT (size=5, starts with 0x80 bit)
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      // Verify Segment size was patched to unknown (0xFF for 1-byte)
      const headerBytes = new Uint8Array(header);
      expect(headerBytes[14]).toBe(0xff);
    });

    it("patches Segment size to unknown for 2-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 2-byte VINT: first byte 0x40-0x7F
      const inputData = new Uint8Array(100);
      inputData.set([0x1a, 0x45, 0xdf, 0xa3, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0);
      inputData.set([0x18, 0x53, 0x80, 0x67], 10); // Segment ID
      inputData.set([0x40, 0x00], 14); // 2-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[14]).toBe(0x7f);
      expect(headerBytes[15]).toBe(0xff);
    });

    it("patches Segment size to unknown for 4-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 4-byte VINT: first byte 0x10-0x1F
      const inputData = new Uint8Array(100);
      inputData.set([0x1a, 0x45, 0xdf, 0xa3, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0);
      inputData.set([0x18, 0x53, 0x80, 0x67], 10); // Segment ID
      inputData.set([0x10, 0x00, 0x00, 0x00], 14); // 4-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[14]).toBe(0x1f);
      expect(headerBytes[15]).toBe(0xff);
      expect(headerBytes[16]).toBe(0xff);
      expect(headerBytes[17]).toBe(0xff);
    });

    it("patches Segment size to unknown for 8-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 8-byte VINT: first byte 0x01
      const inputData = new Uint8Array(100);
      inputData.set([0x1a, 0x45, 0xdf, 0xa3, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 0);
      inputData.set([0x18, 0x53, 0x80, 0x67], 10); // Segment ID
      inputData.set([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], 14); // 8-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[14]).toBe(0x01);
      expect(headerBytes[15]).toBe(0xff);
      expect(headerBytes[16]).toBe(0xff);
      expect(headerBytes[17]).toBe(0xff);
      expect(headerBytes[18]).toBe(0xff);
      expect(headerBytes[19]).toBe(0xff);
      expect(headerBytes[20]).toBe(0xff);
      expect(headerBytes[21]).toBe(0xff);
    });

    it("handles Segment at position 0", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 20,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // Segment ID right at the start
      const inputData = new Uint8Array(50);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID at position 0
      inputData.set([0x85], 4); // 1-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[4]).toBe(0xff); // Patched to unknown
    });

    it("handles 3-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 3-byte VINT: first byte 0x20-0x3F
      const inputData = new Uint8Array(100);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID
      inputData.set([0x20, 0x00, 0x00], 4); // 3-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[4]).toBe(0x3f);
      expect(headerBytes[5]).toBe(0xff);
      expect(headerBytes[6]).toBe(0xff);
    });

    it("handles 5-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 5-byte VINT: first byte 0x08-0x0F
      const inputData = new Uint8Array(100);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID
      inputData.set([0x08, 0x00, 0x00, 0x00, 0x00], 4); // 5-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[4]).toBe(0x0f);
      expect(headerBytes[5]).toBe(0xff);
      expect(headerBytes[6]).toBe(0xff);
      expect(headerBytes[7]).toBe(0xff);
      expect(headerBytes[8]).toBe(0xff);
    });

    it("handles 6-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 6-byte VINT: first byte 0x04-0x07
      const inputData = new Uint8Array(100);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID
      inputData.set([0x04, 0x00, 0x00, 0x00, 0x00, 0x00], 4); // 6-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[4]).toBe(0x07);
      expect(headerBytes[5]).toBe(0xff);
      expect(headerBytes[6]).toBe(0xff);
      expect(headerBytes[7]).toBe(0xff);
      expect(headerBytes[8]).toBe(0xff);
      expect(headerBytes[9]).toBe(0xff);
    });

    it("handles 7-byte VINT", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 50,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // 7-byte VINT: first byte 0x02-0x03
      const inputData = new Uint8Array(100);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID
      inputData.set([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], 4); // 7-byte VINT
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      const headerBytes = new Uint8Array(header);
      expect(headerBytes[4]).toBe(0x03);
      expect(headerBytes[5]).toBe(0xff);
      expect(headerBytes[6]).toBe(0xff);
      expect(headerBytes[7]).toBe(0xff);
      expect(headerBytes[8]).toBe(0xff);
      expect(headerBytes[9]).toBe(0xff);
      expect(headerBytes[10]).toBe(0xff);
    });

    it("handles buffer too short to search for Segment", async () => {
      const customProcessResult = {
        metadatas: [],
        duration: 1000,
        cues: [],
        metadataSize: 4,
      };
      const { hooks } = createFakeHooks({
        webmMuxerConfig: { processResult: customProcessResult },
      });
      setHooks(hooks);

      // Buffer exactly 4 bytes (Segment ID length) - search range is 0
      const inputData = new Uint8Array(4);
      inputData.set([0x18, 0x53, 0x80, 0x67], 0); // Segment ID
      const inputBlob = new Blob([inputData], { type: "audio/webm" });

      const header = await extractWebmHeader(inputBlob);

      // Should return unchanged since search loop doesn't run (maxSearch = 0)
      expect(header.byteLength).toBe(4);
    });
  });

  describe("createWebmWithHeader", () => {
    it("combines header and chunk data into blob", () => {
      const header = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]).buffer; // EBML magic bytes
      const chunkData = new Blob([new Uint8Array([0x01, 0x02, 0x03])], { type: "audio/webm" });

      const result = createWebmWithHeader(header, chunkData);

      expect(result).toBeInstanceOf(Blob);
      expect(result.type).toBe("audio/webm");
      expect(result.size).toBe(7); // 4 header + 3 chunk
    });
  });
});
