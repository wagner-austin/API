/**
 * Audio recording functionality using hooks for testability.
 *
 * This module provides audio recording via MediaRecorder, using the _test_hooks
 * system for dependency injection. Production code uses real browser APIs,
 * while tests can substitute fakes.
 */

import { getGetUserMedia, getMediaRecorderConstructor } from "./_test_hooks.js";
import { createLogger } from "./logger.js";
import { RecorderState, createRecorderState } from "./types.js";

const log = createLogger("recorder");

// Re-export for backward compatibility
export { RecorderState, createRecorderState };

/**
 * Callback type for when recording stops.
 *
 * Args:
 *   audioBlob: The recorded audio as a Blob
 */
export type RecordingStopCallback = (audioBlob: Blob) => void;

/**
 * Callback type for chunk available during streaming recording.
 *
 * Args:
 *   audioBlob: The audio chunk as a Blob
 *   chunkIndex: 1-based index of the chunk (1 = first chunk with headers)
 */
export type ChunkCallback = (audioBlob: Blob, chunkIndex: number) => void;

/**
 * Result of starting a recording.
 */
export interface StartRecordingResult {
  readonly state: RecorderState;
  readonly stopPromise: Promise<Blob>;
  readonly stream: MediaStream;
}

/**
 * Options for starting a recording.
 */
export interface StartRecordingOptions {
  /** Interval in ms to emit chunks (default: no chunking, only on stop) */
  readonly chunkIntervalMs?: number;
  /** Callback for each chunk (required if chunkIntervalMs is set) */
  readonly onChunk?: ChunkCallback;
}

/**
 * Start audio recording.
 *
 * Requests microphone access and begins recording. Returns both the new recorder
 * state and a Promise that resolves with the recorded audio blob when recording
 * stops.
 *
 * Args:
 *   state: Current recorder state (must not be recording)
 *   options: Optional recording options for chunked streaming
 *
 * Returns:
 *   StartRecordingResult with new state and stop promise
 *
 * Raises:
 *   Error if microphone access is denied
 *   Error if already recording
 */
export async function startRecording(
  state: RecorderState,
  options?: StartRecordingOptions
): Promise<StartRecordingResult> {
  if (state.isRecording) {
    throw new Error("Already recording");
  }

  const getUserMedia = getGetUserMedia();
  const MediaRecorderClass = getMediaRecorderConstructor();

  const stream = await getUserMedia({ audio: true });
  const mediaRecorder = new MediaRecorderClass(stream, { mimeType: "audio/webm" });
  const audioChunks: Blob[] = [];

  const chunkIntervalMs = options?.chunkIntervalMs;
  const onChunk = options?.onChunk;

  // Create a Promise that resolves when recording stops
  const stopPromise = new Promise<Blob>((resolve) => {
    mediaRecorder.ondataavailable = (e: BlobEvent): void => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
        const chunkIndex = audioChunks.length;
        log.info("Chunk received:", e.data.size, "bytes, chunk #", chunkIndex);

        // If streaming mode and callback provided, send chunk for translation
        // Skip the final chunk (when stop() is called, state is 'inactive')
        // because handleRecordClick will translate the complete blob after stop
        const isRecording = mediaRecorder.state === "recording";
        if (chunkIntervalMs !== undefined && onChunk !== undefined && isRecording) {
          log.info("Sending chunk #", chunkIndex, "for translation:", e.data.size, "bytes");
          // Send raw chunk with index - app will handle header extraction/prepending
          onChunk(e.data, chunkIndex);
        }
      }
    };

    mediaRecorder.onstop = (): void => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      log.info("Recording stopped, final blob:", blob.size, "bytes");
      // Stop all tracks to release microphone
      stream.getTracks().forEach((track) => track.stop());
      resolve(blob);
    };
  });

  // Start with timeslice for chunked recording, or without for single-blob mode
  if (chunkIntervalMs !== undefined) {
    log.info("Starting chunked recording with interval:", chunkIntervalMs, "ms");
    mediaRecorder.start(chunkIntervalMs);
  } else {
    log.info("Starting standard recording (no chunking)");
    mediaRecorder.start();
  }

  const newState: RecorderState = {
    mediaRecorder,
    audioChunks,
    isRecording: true,
  };

  return { state: newState, stopPromise, stream };
}

/**
 * Stop audio recording.
 *
 * Stops the MediaRecorder, which will trigger the onstop callback and resolve
 * the stopPromise from startRecording.
 *
 * Args:
 *   state: Current recorder state
 *
 * Returns:
 *   New recorder state with isRecording=false
 */
export function stopRecording(state: RecorderState): RecorderState {
  if (state.mediaRecorder !== null && state.isRecording) {
    state.mediaRecorder.stop();
  }

  return {
    ...state,
    isRecording: false,
  };
}
