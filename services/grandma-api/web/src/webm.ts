/**
 * WebM muxing utilities for creating seekable audio blobs.
 *
 * This module provides functionality to fix WebM blobs from MediaRecorder
 * by adding proper seek metadata (Duration, SeekHead, Cues) that allows
 * the audio to be fully read by consumers like Whisper.
 *
 * Uses ts-ebml library via hooks for testability.
 */

import { getWebmMuxer, EBMLElement, CueInfo, WebmProcessResult } from "./_test_hooks.js";
import { createLogger } from "./logger.js";

const log = createLogger("webm");

// Re-export types for consumers
export type { EBMLElement, CueInfo, WebmProcessResult };

// ============================================================================
// EBML Constants
// ============================================================================

/** Segment element ID: 0x18 0x53 0x80 0x67 */
const SEGMENT_ID = new Uint8Array([0x18, 0x53, 0x80, 0x67]);

/**
 * Get the VINT "unknown size" marker for a given VINT length.
 *
 * All data bits set to 1 indicates unknown/indefinite size.
 *
 * Args:
 *   length: VINT length in bytes (1-8)
 *
 * Returns:
 *   Uint8Array with unknown size marker bytes
 */
function getUnknownSizeMarker(length: number): Uint8Array {
  switch (length) {
    case 1:
      return new Uint8Array([0xff]);
    case 2:
      return new Uint8Array([0x7f, 0xff]);
    case 3:
      return new Uint8Array([0x3f, 0xff, 0xff]);
    case 4:
      return new Uint8Array([0x1f, 0xff, 0xff, 0xff]);
    case 5:
      return new Uint8Array([0x0f, 0xff, 0xff, 0xff, 0xff]);
    case 6:
      return new Uint8Array([0x07, 0xff, 0xff, 0xff, 0xff, 0xff]);
    case 7:
      return new Uint8Array([0x03, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
    default:
      // 8 bytes (or fallback for any invalid length)
      return new Uint8Array([0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
  }
}

// ============================================================================
// EBML VINT Utilities
// ============================================================================

/**
 * Get the length of a VINT (Variable Integer) from its first byte.
 *
 * VINT encoding: the number of leading zero bits + 1 = total bytes.
 * - 0x80-0xFF: 1 byte (leading 1)
 * - 0x40-0x7F: 2 bytes (leading 01)
 * - 0x20-0x3F: 3 bytes (leading 001)
 * - etc.
 *
 * Args:
 *   firstByte: First byte of the VINT
 *
 * Returns:
 *   Number of bytes in the VINT (1-8)
 */
function getVintLength(firstByte: number): number {
  if (firstByte >= 0x80) return 1;
  if (firstByte >= 0x40) return 2;
  if (firstByte >= 0x20) return 3;
  if (firstByte >= 0x10) return 4;
  if (firstByte >= 0x08) return 5;
  if (firstByte >= 0x04) return 6;
  if (firstByte >= 0x02) return 7;
  return 8;
}

/**
 * Find the Segment element in a WebM buffer and return its position info.
 *
 * Args:
 *   data: WebM data as Uint8Array
 *
 * Returns:
 *   Object with segment position info, or null if not found
 */
function findSegmentElement(
  data: Uint8Array
): { idStart: number; sizeStart: number; sizeLength: number } | null {
  // Search for Segment ID in first 100 bytes (should be right after EBML header)
  const maxSearch = Math.min(100, data.length - SEGMENT_ID.length);

  for (let i = 0; i < maxSearch; i++) {
    // Check if we found the Segment ID
    let match = true;
    for (let j = 0; j < SEGMENT_ID.length; j++) {
      if (data[i + j] !== SEGMENT_ID[j]) {
        match = false;
        break;
      }
    }

    if (match) {
      const sizeStart = i + SEGMENT_ID.length;
      // firstByte is always defined because the search loop ensures
      // sizeStart <= data.length - 1 (since i < data.length - SEGMENT_ID.length)
      const firstByte = data[sizeStart] as number;
      const sizeLength = getVintLength(firstByte);

      return {
        idStart: i,
        sizeStart,
        sizeLength,
      };
    }
  }

  return null;
}

/**
 * Patch the Segment element's size to "unknown" in a WebM header.
 *
 * This modifies the header bytes in-place to use the EBML "unknown size"
 * encoding, which allows the Segment to contain any amount of data.
 * This is essential for prepending headers to subsequent chunks.
 *
 * Args:
 *   headerBytes: WebM header as ArrayBuffer (will be copied, not modified)
 *
 * Returns:
 *   New ArrayBuffer with patched Segment size
 */
function patchSegmentSize(headerBytes: ArrayBuffer): ArrayBuffer {
  const data = new Uint8Array(headerBytes.slice(0)); // Copy to avoid mutation
  const segmentInfo = findSegmentElement(data);

  if (segmentInfo === null) {
    log.warn("Could not find Segment element to patch size");
    return data.buffer;
  }

  const { sizeStart, sizeLength } = segmentInfo;
  const unknownSizeMarker = getUnknownSizeMarker(sizeLength);

  log.info(
    "Patching Segment size at offset",
    sizeStart,
    "length",
    sizeLength,
    "to unknown"
  );

  // Replace the size bytes with "unknown size" marker
  data.set(unknownSizeMarker, sizeStart);

  return data.buffer;
}

// ============================================================================
// WebM Processing
// ============================================================================

/**
 * Extract the WebM header from a blob and patch it for reuse.
 *
 * The header includes EBML declaration, Segment start, Info, and Tracks.
 * The Segment's size field is patched to "unknown" so this header can be
 * prepended to subsequent chunks to make them valid standalone WebM files.
 *
 * Args:
 *   blob: WebM blob (must be from MediaRecorder's first chunk)
 *
 * Returns:
 *   Promise resolving to patched header bytes as ArrayBuffer
 */
export async function extractWebmHeader(blob: Blob): Promise<ArrayBuffer> {
  log.info("extractWebmHeader() - input size:", blob.size, "bytes");

  const muxer = getWebmMuxer();
  const buffer = await blob.arrayBuffer();

  // Decode and process to get metadata size
  const elements = muxer.decode(buffer);
  const processResult = muxer.processElements(elements);

  log.info("Header size (metadataSize):", processResult.metadataSize, "bytes");

  // Extract header and patch Segment size to "unknown"
  const rawHeader = buffer.slice(0, processResult.metadataSize);
  const patchedHeader = patchSegmentSize(rawHeader);
  log.info("Patched header size:", patchedHeader.byteLength, "bytes");

  return patchedHeader;
}

/**
 * Create a valid WebM blob by prepending header to chunk data.
 *
 * Args:
 *   header: WebM header from extractWebmHeader()
 *   chunkData: Raw chunk data (Cluster elements only)
 *
 * Returns:
 *   Complete WebM blob with header + chunk data
 */
export function createWebmWithHeader(header: ArrayBuffer, chunkData: Blob): Blob {
  return new Blob([header, chunkData], { type: "audio/webm" });
}

/**
 * Fix a WebM blob from MediaRecorder to be seekable.
 *
 * MediaRecorder produces streaming WebM that lacks proper seek metadata.
 * This function adds Duration, SeekHead, and Cues elements so that
 * consumers like Whisper can read the full audio.
 *
 * Args:
 *   blob: Raw WebM blob from MediaRecorder
 *
 * Returns:
 *   Promise resolving to a fixed, seekable WebM blob
 *
 * Raises:
 *   Error if the blob cannot be processed
 */
export async function fixWebmBlob(blob: Blob): Promise<Blob> {
  log.info("fixWebmBlob() - input size:", blob.size, "bytes");

  const muxer = getWebmMuxer();
  const buffer = await blob.arrayBuffer();

  // Decode buffer into EBML elements
  const elements = muxer.decode(buffer);
  log.info("Decoded", elements.length, "EBML elements");

  // Process elements through reader to get metadata, duration, cues
  const processResult = muxer.processElements(elements);
  log.info("Processed - duration:", processResult.duration, "metadataSize:", processResult.metadataSize);

  // Create seekable metadata
  const refinedMetadata = muxer.makeMetadataSeekable(
    processResult.metadatas,
    processResult.duration,
    processResult.cues
  );
  log.info("Created refined metadata:", refinedMetadata.byteLength, "bytes");

  // Combine refined metadata with body
  const body = buffer.slice(processResult.metadataSize);
  const fixedBlob = new Blob([refinedMetadata, body], { type: "audio/webm" });
  log.info("fixWebmBlob() - output size:", fixedBlob.size, "bytes");

  return fixedBlob;
}
