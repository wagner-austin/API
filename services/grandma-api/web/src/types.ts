/**
 * Type definitions with encode/decode/require pattern.
 *
 * Every TypedDict-like interface has:
 * - encode*: Convert typed object to JSON-serializable form
 * - decode*: Parse unknown value into typed object (returns null on failure)
 * - require*: Parse unknown value into typed object (throws on failure)
 */

// ============================================================================
// AppConfig
// ============================================================================

/**
 * Application configuration loaded from config.json.
 */
export interface AppConfig {
  readonly API_BASE_URL: string;
}

/**
 * Encode AppConfig to JSON-serializable object.
 *
 * Args:
 *   config: AppConfig to encode
 *
 * Returns:
 *   JSON-serializable object
 */
export function encodeAppConfig(config: AppConfig): Record<string, unknown> {
  return {
    API_BASE_URL: config.API_BASE_URL,
  };
}

/**
 * Decode unknown value to AppConfig.
 *
 * Args:
 *   value: Unknown value to decode
 *
 * Returns:
 *   AppConfig if valid, null otherwise
 */
export function decodeAppConfig(value: unknown): AppConfig | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const obj = value as Record<string, unknown>;
  const apiBaseUrl = obj["API_BASE_URL"];
  if (typeof apiBaseUrl !== "string" || apiBaseUrl.length === 0) {
    return null;
  }
  return { API_BASE_URL: apiBaseUrl };
}

/**
 * Require unknown value to be valid AppConfig.
 *
 * Args:
 *   value: Unknown value to validate
 *
 * Returns:
 *   AppConfig
 *
 * Raises:
 *   Error if value is not valid AppConfig
 */
export function requireAppConfig(value: unknown): AppConfig {
  const config = decodeAppConfig(value);
  if (config === null) {
    throw new Error("Invalid config.json: missing or invalid API_BASE_URL");
  }
  return config;
}

// ============================================================================
// TranslateResponse
// ============================================================================

/**
 * Response from /translate endpoint.
 */
export interface TranslateResponse {
  readonly text: string;
  readonly detected_language: string;
  readonly source_text: string;
  readonly confidence: number;
}

/**
 * Encode TranslateResponse to JSON-serializable object.
 *
 * Args:
 *   response: TranslateResponse to encode
 *
 * Returns:
 *   JSON-serializable object
 */
export function encodeTranslateResponse(response: TranslateResponse): Record<string, unknown> {
  return {
    text: response.text,
    detected_language: response.detected_language,
    source_text: response.source_text,
    confidence: response.confidence,
  };
}

/**
 * Decode unknown value to TranslateResponse.
 *
 * Args:
 *   value: Unknown value to decode
 *
 * Returns:
 *   TranslateResponse if valid, null otherwise
 */
export function decodeTranslateResponse(value: unknown): TranslateResponse | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const obj = value as Record<string, unknown>;
  const text = obj["text"];
  const detectedLanguage = obj["detected_language"];
  const sourceText = obj["source_text"];
  const confidence = obj["confidence"];
  if (typeof text !== "string") {
    return null;
  }
  if (typeof detectedLanguage !== "string") {
    return null;
  }
  if (typeof sourceText !== "string") {
    return null;
  }
  if (typeof confidence !== "number") {
    return null;
  }
  return {
    text,
    detected_language: detectedLanguage,
    source_text: sourceText,
    confidence,
  };
}

/**
 * Require unknown value to be valid TranslateResponse.
 *
 * Args:
 *   value: Unknown value to validate
 *
 * Returns:
 *   TranslateResponse
 *
 * Raises:
 *   Error if value is not valid TranslateResponse
 */
export function requireTranslateResponse(value: unknown): TranslateResponse {
  const response = decodeTranslateResponse(value);
  if (response === null) {
    throw new Error("Invalid response from server");
  }
  return response;
}

// ============================================================================
// ErrorResponse
// ============================================================================

/**
 * Error response from API.
 */
export interface ErrorResponse {
  readonly detail: string;
}

/**
 * Encode ErrorResponse to JSON-serializable object.
 *
 * Args:
 *   response: ErrorResponse to encode
 *
 * Returns:
 *   JSON-serializable object
 */
export function encodeErrorResponse(response: ErrorResponse): Record<string, unknown> {
  return {
    detail: response.detail,
  };
}

/**
 * Decode unknown value to ErrorResponse.
 *
 * Args:
 *   value: Unknown value to decode
 *
 * Returns:
 *   ErrorResponse if valid, null otherwise
 */
export function decodeErrorResponse(value: unknown): ErrorResponse | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const obj = value as Record<string, unknown>;
  const detail = obj["detail"];
  if (typeof detail !== "string") {
    return null;
  }
  return { detail };
}

// ============================================================================
// RecorderState
// ============================================================================

/**
 * Recorder state for audio recording.
 */
export interface RecorderState {
  readonly mediaRecorder: MediaRecorder | null;
  readonly audioChunks: readonly Blob[];
  readonly isRecording: boolean;
}

/**
 * Create initial recorder state.
 *
 * Returns:
 *   Initial RecorderState with no active recording
 */
export function createRecorderState(): RecorderState {
  return {
    mediaRecorder: null,
    audioChunks: [],
    isRecording: false,
  };
}

// ============================================================================
// AppState
// ============================================================================

/**
 * Application UI state.
 */
export interface AppState {
  readonly config: AppConfig | null;
  readonly token: string | null;
  readonly recorderState: RecorderState;
  readonly transcripts: readonly string[];
  readonly pendingOperation: Promise<void> | null;
}

/**
 * Create initial application state.
 *
 * Returns:
 *   Initial AppState
 */
export function createAppState(): AppState {
  return {
    config: null,
    token: null,
    recorderState: createRecorderState(),
    transcripts: [],
    pendingOperation: null,
  };
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Custom event types emitted by the app.
 */
export type AppEventType =
  | "app:initialized"
  | "app:login"
  | "app:logout"
  | "app:recording-start"
  | "app:recording-stop"
  | "app:translation-complete"
  | "app:translation-error"
  | "app:clear";

/**
 * Event detail for translation complete event.
 */
export interface TranslationCompleteDetail {
  readonly text: string;
  readonly detected_language: string;
  readonly source_text: string;
  readonly confidence: number;
}

/**
 * Event detail for translation error event.
 */
export interface TranslationErrorDetail {
  readonly message: string;
}
