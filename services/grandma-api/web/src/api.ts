/**
 * Translation API client using hooks for testability.
 *
 * Provides functions to interact with the grandma-api translation endpoint.
 * Uses the _test_hooks system for fetch dependency injection.
 */

import { getFetch } from "./_test_hooks.js";
import { createLogger } from "./logger.js";
import { requireTranslateResponse, decodeErrorResponse } from "./types.js";

const log = createLogger("api");

/**
 * Send audio to the translation API.
 *
 * Uploads audio data to the /translate endpoint and returns the translated text.
 * Uses the fetch hook from _test_hooks for testability.
 *
 * Args:
 *   baseUrl: API base URL (e.g., "https://api.example.com")
 *   token: Authentication token
 *   audioBlob: Audio data as Blob
 *
 * Returns:
 *   Promise resolving to translated text string
 *
 * Raises:
 *   Error if request fails with HTTP error
 *   Error if response is invalid or missing text field
 */
export async function translateAudio(
  baseUrl: string,
  token: string,
  audioBlob: Blob
): Promise<string> {
  const fetch = getFetch();

  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  formData.append("token", token);

  const url = `${baseUrl}/translate`;
  log.info("Sending translation request to:", url);
  log.info("Audio blob size:", audioBlob.size, "bytes");

  let resp: Response;
  try {
    resp = await fetch(url, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    log.error("Fetch failed:", err);
    throw err;
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

  return response.text;
}
