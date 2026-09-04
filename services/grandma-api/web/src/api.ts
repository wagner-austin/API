/**
 * Translation API client using hooks for testability.
 *
 * Provides functions to interact with the grandma-api translation endpoint.
 * Uses the _test_hooks system for fetch dependency injection.
 */

import { getFetch } from "./_test_hooks.js";
import { createLogger } from "./logger.js";
import {
  TranslateResponse,
  requireTranslateResponse,
  decodeErrorResponse,
} from "./types.js";

const log = createLogger("api");

/** Timeout for translation API requests (30 seconds). */
const TRANSLATE_TIMEOUT_MS = 30000;

/**
 * Send audio to the translation API.
 *
 * Uploads audio data to the /translate endpoint and returns the translation result
 * including detected language, source text, confidence, and English translation.
 * Uses the fetch hook from _test_hooks for testability.
 *
 * Args:
 *   baseUrl: API base URL (e.g., "https://api.example.com")
 *   token: Authentication token
 *   audioBlob: Audio data as Blob
 *
 * Returns:
 *   Promise resolving to TranslateResponse with text, detected_language,
 *   source_text, and confidence
 *
 * Raises:
 *   Error if request fails with HTTP error
 *   Error if response is invalid or missing required fields
 */
export async function translateAudio(
  baseUrl: string,
  token: string,
  audioBlob: Blob
): Promise<TranslateResponse> {
  const fetch = getFetch();

  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  formData.append("token", token);

  const url = `${baseUrl}/translate`;
  log.info("Sending translation request to:", url);
  log.info("Audio blob size:", audioBlob.size, "bytes");

  // Use AbortController for timeout on slow networks
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TRANSLATE_TIMEOUT_MS);

  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      log.error("Translation request timed out after", TRANSLATE_TIMEOUT_MS, "ms");
      // `cause` carries the AbortError forward. Without it the DOMException
      // that actually fired -- the only thing that distinguishes our own
      // abort from a network stack that gave up -- is discarded at the one
      // point a reader of the stack would want it.
      throw new Error("Translation request timed out", { cause: err });
    }
    log.error("Fetch failed:", err);
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }

  log.info("Response status:", resp.status);

  if (!resp.ok) {
    // Attempt to parse error response for detail message
    const errorBody: unknown = await resp.json().catch(() => null);
    log.error("Error response body:", errorBody);
    const errorResponse = errorBody !== null ? decodeErrorResponse(errorBody) : null;

    if (errorResponse !== null) {
      throw new Error(errorResponse.detail);
    }

    throw new Error(`HTTP ${resp.status}`);
  }

  const data: unknown = await resp.json();
  log.info("Translation response:", data);
  const response = requireTranslateResponse(data);

  return response;
}
