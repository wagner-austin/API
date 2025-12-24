/**
 * Main application module with Promise-returning handlers.
 *
 * This module provides the main application logic with:
 * - Promise-returning handlers for testable async operations
 * - Event-based signaling via custom DOM events
 * - Hooks-based dependency injection for all external dependencies
 *
 * Usage:
 *   import { createApp } from "./app.js";
 *   const app = createApp();
 *   await app.init();
 */

import { getDocument } from "./_test_hooks.js";
import { loadConfig, clearConfigCache } from "./config.js";
import { translateAudio } from "./api.js";
import { saveToken, loadToken, clearToken } from "./storage.js";
import { startRecording, stopRecording } from "./recorder.js";
import { createLogger } from "./logger.js";
import { fixWebmBlob, extractWebmHeader, createWebmWithHeader } from "./webm.js";
import {
  AppConfig,
  RecorderState,
  createRecorderState,
  AppEventType,
  TranslationErrorDetail,
} from "./types.js";

/** Chunk interval for streaming recordings (15 seconds). */
const CHUNK_INTERVAL_MS = 15000;

/** Maximum recording duration (1 minute). Auto-stops after this. */
const MAX_RECORDING_MS = 60000;

const log = createLogger("app");

// ============================================================================
// DOM Element References
// ============================================================================

/**
 * DOM element references for the application UI.
 */
interface AppElements {
  readonly loginSection: HTMLElement;
  readonly mainSection: HTMLElement;
  readonly loginForm: HTMLFormElement;
  readonly tokenInput: HTMLInputElement;
  readonly loginError: HTMLElement;
  readonly logoutBtn: HTMLButtonElement;
  readonly recordBtn: HTMLButtonElement;
  readonly recordIcon: HTMLElement;
  readonly status: HTMLElement;
  readonly transcript: HTMLElement;
  readonly clearBtn: HTMLButtonElement;
  readonly recordingFeedback: HTMLElement;
  readonly timer: HTMLElement;
  readonly audioLevel: HTMLElement;
}

/**
 * Get element by ID with type safety.
 *
 * Args:
 *   id: Element ID
 *
 * Returns:
 *   Element
 *
 * Raises:
 *   Error if element not found
 */
function getElementById<T extends HTMLElement>(id: string): T {
  const doc = getDocument();
  const el = doc.getElementById(id);
  if (el === null) {
    throw new Error(`Element not found: #${id}`);
  }
  return el as T;
}

/**
 * Initialize DOM element references.
 *
 * Returns:
 *   AppElements with all required element references
 *
 * Raises:
 *   Error if any required element is not found
 */
function initElements(): AppElements {
  return {
    loginSection: getElementById("login-section"),
    mainSection: getElementById("main-section"),
    loginForm: getElementById<HTMLFormElement>("login-form"),
    tokenInput: getElementById<HTMLInputElement>("token-input"),
    loginError: getElementById("login-error"),
    logoutBtn: getElementById<HTMLButtonElement>("logout-btn"),
    recordBtn: getElementById<HTMLButtonElement>("record-btn"),
    recordIcon: getElementById("record-icon"),
    status: getElementById("status"),
    transcript: getElementById("transcript"),
    clearBtn: getElementById<HTMLButtonElement>("clear-btn"),
    recordingFeedback: getElementById("recording-feedback"),
    timer: getElementById("timer"),
    audioLevel: getElementById("audio-level"),
  };
}

// ============================================================================
// Event Emission
// ============================================================================

/**
 * Emit a custom application event.
 *
 * Args:
 *   type: Event type
 *   detail: Optional event detail
 */
function emitEvent<T>(type: AppEventType, detail?: T): void {
  const event = new CustomEvent(type, { detail });
  document.dispatchEvent(event);
}

// ============================================================================
// Application Class
// ============================================================================

/**
 * Application controller with Promise-returning methods.
 *
 * All handler methods return Promises that resolve when the operation is complete,
 * allowing tests to await them directly without setTimeout hacks.
 */
export class App {
  private elements: AppElements | null = null;
  private config: AppConfig | null = null;
  private token: string | null = null;
  private recorderState: RecorderState = createRecorderState();
  private transcripts: string[] = [];
  private currentSessionIndex: number = -1; // Tracks current recording session's transcript
  private cumulativeChunks: Blob[] = []; // All chunks for current recording session
  private webmHeader: ArrayBuffer | null = null; // WebM header from first chunk
  private stopPromise: Promise<Blob> | null = null;
  private pendingChunkTranslation: Promise<void> | null = null; // Track pending chunk translations
  private timerInterval: number | null = null;
  private recordingStartTime: number | null = null;
  private autoStopTimeout: number | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private animationFrame: number | null = null;

  // ============================================================================
  // Initialization
  // ============================================================================

  /**
   * Initialize the application.
   *
   * Loads configuration, sets up elements, and restores saved token.
   *
   * Returns:
   *   Promise that resolves when initialization is complete
   *
   * Raises:
   *   Error if config.json cannot be loaded
   */
  async init(): Promise<void> {
    log.info("init() starting...");
    this.elements = initElements();
    log.info("Elements initialized");

    // Load config
    log.info("Loading config...");
    try {
      this.config = await loadConfig();
      log.info("Config loaded successfully:", this.config);
    } catch (err) {
      log.error("Config load failed:", err);
      throw err;
    }

    // Check for saved token
    const savedToken = loadToken();
    log.info("Saved token exists:", savedToken !== null);
    if (savedToken !== null) {
      this.token = savedToken;
      this.showMainView(this.elements);
      log.info("Showing main view (token found)");
    } else {
      this.showLoginView(this.elements);
      log.info("Showing login view (no token)");
    }

    this.setupEventListeners(this.elements);
    log.info("Event listeners set up");
    emitEvent("app:initialized");
    log.info("init() complete");
  }

  // ============================================================================
  // View Switching
  // ============================================================================

  /**
   * Show login screen, hide main app.
   *
   * Args:
   *   elements: DOM element references
   */
  private showLoginView(elements: AppElements): void {
    elements.loginSection.classList.remove("hidden");
    elements.mainSection.classList.add("hidden");
    elements.tokenInput.focus();
  }

  /**
   * Show main app, hide login screen.
   *
   * Args:
   *   elements: DOM element references
   */
  private showMainView(elements: AppElements): void {
    elements.loginSection.classList.add("hidden");
    elements.mainSection.classList.remove("hidden");
  }

  // ============================================================================
  // Handlers
  // ============================================================================

  /**
   * Handle login form submission.
   *
   * Args:
   *   e: Form submit event
   *
   * Returns:
   *   Promise that resolves when login is complete
   */
  async handleLogin(e: Event): Promise<void> {
    e.preventDefault();
    if (this.elements === null) return;

    this.elements.loginError.classList.add("hidden");

    const inputToken = this.elements.tokenInput.value.trim();
    if (inputToken.length === 0) {
      this.elements.loginError.textContent = "Please enter a password";
      this.elements.loginError.classList.remove("hidden");
      return;
    }

    this.token = inputToken;
    saveToken(inputToken);
    this.showMainView(this.elements);
    emitEvent("app:login");
  }

  /**
   * Handle logout button click.
   *
   * Returns:
   *   Promise that resolves when logout is complete
   */
  async handleLogout(): Promise<void> {
    if (this.elements === null) return;

    this.token = null;
    clearToken();
    this.transcripts = [];
    this.updateTranscriptDisplay();
    this.showLoginView(this.elements);
    emitEvent("app:logout");
  }

  /**
   * Handle record button click.
   *
   * Toggles recording state. When stopping, waits for translation to complete.
   *
   * Returns:
   *   Promise that resolves when the operation is complete
   */
  async handleRecordClick(): Promise<void> {
    log.info("handleRecordClick() called");
    log.info("State check - elements:", this.elements !== null);
    log.info("State check - config:", this.config);
    log.info("State check - token:", this.token !== null ? "(set)" : "(null)");
    log.info("State check - isRecording:", this.recorderState.isRecording);

    if (this.elements === null || this.config === null || this.token === null) {
      log.warn("handleRecordClick() aborted - missing state");
      return;
    }

    // Capture references for closures (needed because elements/config/token are narrowed here)
    const elements = this.elements;
    const config = this.config;
    const token = this.token;

    if (this.recorderState.isRecording) {
      // Stop recording
      log.info("Stopping recording...");

      // Wait for any pending chunk translation to complete first
      if (this.pendingChunkTranslation !== null) {
        log.info("Waiting for pending chunk translation...");
        await this.pendingChunkTranslation;
        log.info("Pending chunk translation complete");
      }

      // Clear auto-stop timeout
      if (this.autoStopTimeout !== null) {
        clearTimeout(this.autoStopTimeout);
        this.autoStopTimeout = null;
      }

      this.recorderState = stopRecording(this.recorderState);
      this.stopRecordingFeedback(elements);
      elements.recordBtn.classList.remove("recording");
      elements.recordIcon.textContent = "🎤";
      elements.status.textContent = "Translating...";
      emitEvent("app:recording-stop");

      // Wait for recording to finish and translate the full audio
      if (this.stopPromise === null) {
        throw new Error("Recording stopped but no stop promise available");
      }
      log.info("Waiting for final audio blob...");
      const blob = await this.stopPromise;
      log.info("Got final blob:", blob.size, "bytes");
      this.stopPromise = null;

      // Fix the WebM blob to make it seekable, then translate
      const fixedBlob = await fixWebmBlob(blob);
      log.info("Fixed blob:", fixedBlob.size, "bytes");

      const text = await translateAudio(config.API_BASE_URL, token, fixedBlob);
      log.info("Translation complete:", text.length, "chars");

      // Update the current session's transcript with final translation
      if (this.currentSessionIndex >= 0 && this.currentSessionIndex < this.transcripts.length) {
        this.transcripts[this.currentSessionIndex] = text;
      } else {
        this.transcripts.push(text);
      }
      this.updateTranscriptDisplay();
      elements.status.textContent = "Tap to record";

      // Reset session tracking
      this.cumulativeChunks = [];
      this.currentSessionIndex = -1;
    } else {
      // Start recording with chunked streaming
      log.info("Starting recording with", CHUNK_INTERVAL_MS, "ms chunk interval...");

      // Reset session state
      this.cumulativeChunks = [];
      this.currentSessionIndex = -1;
      this.webmHeader = null;

      try {
        const result = await startRecording(this.recorderState, {
          chunkIntervalMs: CHUNK_INTERVAL_MS,
          onChunk: (chunk: Blob, chunkIndex: number) => {
            this.handleChunkReceived(chunk, chunkIndex, elements, config, token);
          },
        });
        log.info("Recording started successfully");
        this.recorderState = result.state;
        this.stopPromise = result.stopPromise;

        elements.recordBtn.classList.add("recording");
        elements.recordIcon.textContent = "⏹";
        elements.status.textContent = "Recording... (updates every 15s)";
        this.startRecordingFeedback(elements, result.stream);
        emitEvent("app:recording-start");

        // Auto-stop after max duration to prevent runaway recordings
        this.autoStopTimeout = window.setTimeout(() => {
          log.info("Auto-stopping recording after", MAX_RECORDING_MS, "ms");
          void this.handleRecordClick();
        }, MAX_RECORDING_MS);
      } catch (err) {
        log.error("Recording start failed:", err);
        throw err;
      }
    }
  }

  /**
   * Handle a chunk received during recording.
   *
   * For the first chunk: extracts and stores the WebM header, sends as-is.
   * For subsequent chunks: prepends the stored header to make valid WebM.
   * Each chunk is translated independently and results are appended.
   *
   * Args:
   *   chunk: Raw audio chunk from MediaRecorder
   *   chunkIndex: 1-based index (1 = first chunk with headers)
   *   elements: DOM elements
   *   config: App config
   *   token: Auth token
   */
  private handleChunkReceived(
    chunk: Blob,
    chunkIndex: number,
    elements: AppElements,
    config: AppConfig,
    token: string
  ): void {
    log.info("handleChunkReceived() - chunk #", chunkIndex, "size:", chunk.size, "bytes");

    // Store chunk for final full transcription
    this.cumulativeChunks.push(chunk);

    if (chunkIndex === 1) {
      // First chunk has WebM header - extract it and send chunk as-is
      log.info("First chunk - extracting header");

      // Track the full operation (header extraction + translation)
      this.pendingChunkTranslation = extractWebmHeader(chunk)
        .then((header) => {
          this.webmHeader = header;
          log.info("Header extracted:", header.byteLength, "bytes");
          // Return the translation promise so it's awaited
          return this.doChunkTranslation(chunk, elements, config, token);
        })
        .catch((err: unknown) => {
          log.error("Failed to extract header:", err);
          // Still try to translate even if header extraction fails
          return this.doChunkTranslation(chunk, elements, config, token);
        })
        .finally(() => {
          this.pendingChunkTranslation = null;
        });
    } else if (this.webmHeader !== null) {
      // Subsequent chunks need header prepended to be valid WebM
      const chunkWithHeader = createWebmWithHeader(this.webmHeader, chunk);
      log.info("Chunk with header:", chunkWithHeader.size, "bytes");
      // Skip fixWebmBlob for subsequent chunks - the header+cluster structure
      // isn't perfect but OpenAI can usually process it
      this.handleRawChunkTranslation(chunkWithHeader, elements, config, token);
    } else {
      // Header not ready yet - this shouldn't happen normally
      log.warn("Header not ready for chunk #", chunkIndex, "- skipping");
    }
  }

  /**
   * Do the actual chunk translation (called from handleChunkReceived).
   * Returns a promise so it can be chained with header extraction.
   */
  private async doChunkTranslation(
    blob: Blob,
    elements: AppElements,
    config: AppConfig,
    token: string
  ): Promise<void> {
    log.info("doChunkTranslation() - blob size:", blob.size);
    elements.status.textContent = "Translating...";

    try {
      const fixedBlob = await fixWebmBlob(blob);
      log.info("Fixed chunk blob:", fixedBlob.size, "bytes");
      const text = await translateAudio(config.API_BASE_URL, token, fixedBlob);
      log.info("Chunk translation received:", text.length, "chars");

      // APPEND chunk text to current session's transcript
      if (this.currentSessionIndex === -1 || this.currentSessionIndex >= this.transcripts.length) {
        this.transcripts.push(text);
        this.currentSessionIndex = this.transcripts.length - 1;
      } else {
        const existing = this.transcripts[this.currentSessionIndex] ?? "";
        this.transcripts[this.currentSessionIndex] = existing + " " + text;
      }
      this.updateTranscriptDisplay();

      if (this.recorderState.isRecording) {
        elements.status.textContent = "Recording... (updates every 15s)";
      }
    } catch (err) {
      log.error("Chunk translation failed:", err);
      if (this.recorderState.isRecording) {
        elements.status.textContent = "Recording... (translation error, retrying)";
      }
    }
  }

  /**
   * Handle raw chunk translation (for chunks 2+).
   *
   * Skips fixWebmBlob since header+cluster concatenation isn't a perfect WebM
   * that ts-ebml can process. Sends directly to translation API.
   *
   * Args:
   *   blob: Audio chunk with header prepended
   *   elements: DOM elements
   *   config: App config
   *   token: Auth token
   */
  private handleRawChunkTranslation(
    blob: Blob,
    elements: AppElements,
    config: AppConfig,
    token: string
  ): void {
    this.pendingChunkTranslation = this.doRawChunkTranslation(blob, elements, config, token)
      .finally(() => {
        this.pendingChunkTranslation = null;
      });
  }

  /**
   * Do raw chunk translation without fixWebmBlob.
   *
   * For subsequent chunks (2+), the header+cluster blob isn't perfect WebM
   * but OpenAI/Whisper can often still process it.
   */
  private async doRawChunkTranslation(
    blob: Blob,
    elements: AppElements,
    config: AppConfig,
    token: string
  ): Promise<void> {
    log.info("doRawChunkTranslation() - blob size:", blob.size, "(no fixWebmBlob)");
    elements.status.textContent = "Translating...";

    try {
      // Skip fixWebmBlob - send directly to API
      const text = await translateAudio(config.API_BASE_URL, token, blob);
      log.info("Raw chunk translation received:", text.length, "chars");

      // APPEND chunk text to current session's transcript
      if (this.currentSessionIndex === -1 || this.currentSessionIndex >= this.transcripts.length) {
        this.transcripts.push(text);
        this.currentSessionIndex = this.transcripts.length - 1;
      } else {
        const existing = this.transcripts[this.currentSessionIndex] ?? "";
        this.transcripts[this.currentSessionIndex] = existing + " " + text;
      }
      this.updateTranscriptDisplay();

      if (this.recorderState.isRecording) {
        elements.status.textContent = "Recording... (updates every 15s)";
      }
    } catch (err) {
      log.error("Raw chunk translation failed:", err);
      if (this.recorderState.isRecording) {
        elements.status.textContent = "Recording... (translation error, retrying)";
      }
    }
  }

  /**
   * Handle clear button click.
   *
   * Returns:
   *   Promise that resolves when clear is complete
   */
  async handleClear(): Promise<void> {
    this.transcripts = [];
    this.updateTranscriptDisplay();
    emitEvent("app:clear");
  }

  // ============================================================================
  // Display Updates
  // ============================================================================

  /**
   * Update transcript display with current transcripts.
   */
  private updateTranscriptDisplay(): void {
    if (this.elements === null) return;

    if (this.transcripts.length === 0) {
      this.elements.transcript.textContent = "Translations will appear here...";
    } else {
      this.elements.transcript.textContent = this.transcripts.join("\n\n");
    }
  }

  // ============================================================================
  // Recording Feedback (Timer + Audio Level)
  // ============================================================================

  /**
   * Format seconds as M:SS string.
   *
   * Args:
   *   seconds: Number of seconds
   *
   * Returns:
   *   Formatted time string
   */
  private formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  /**
   * Start recording feedback (timer and audio level visualization).
   *
   * Args:
   *   elements: DOM element references
   *   stream: MediaStream from recording
   */
  private startRecordingFeedback(elements: AppElements, stream: MediaStream): void {
    // Show feedback container
    elements.recordingFeedback.classList.remove("hidden");

    // Start timer
    this.recordingStartTime = Date.now();
    elements.timer.textContent = "0:00";
    this.timerInterval = window.setInterval(() => {
      if (this.recordingStartTime === null) {
        throw new Error("Timer running but recording start time is null");
      }
      const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
      elements.timer.textContent = this.formatTime(elapsed);
    }, 1000);

    // Start audio level visualization (only if AudioContext available)
    if (typeof AudioContext === "undefined") {
      return;
    }

    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    source.connect(this.analyser);

    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

    const updateLevel = (): void => {
      if (this.analyser === null) {
        throw new Error("Animation frame running but analyser is null");
      }

      this.analyser.getByteFrequencyData(dataArray);

      // Calculate average level using reduce (avoids index access issues)
      const sum = dataArray.reduce((acc, val) => acc + val, 0);
      const average = sum / dataArray.length;
      const level = Math.min(100, (average / 128) * 100);

      elements.audioLevel.style.width = `${level}%`;

      this.animationFrame = requestAnimationFrame(updateLevel);
    };

    updateLevel();
  }

  /**
   * Stop recording feedback (timer and audio level visualization).
   *
   * Args:
   *   elements: DOM element references
   */
  private stopRecordingFeedback(elements: AppElements): void {
    // Hide feedback container
    elements.recordingFeedback.classList.add("hidden");

    // Stop timer
    if (this.timerInterval !== null) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    this.recordingStartTime = null;
    elements.timer.textContent = "0:00";

    // Stop audio level visualization
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.audioContext !== null) {
      void this.audioContext.close();
      this.audioContext = null;
    }
    this.analyser = null;
    elements.audioLevel.style.width = "0%";
  }

  // ============================================================================
  // Event Listeners
  // ============================================================================

  /**
   * Set up event listeners with Promise-returning wrappers.
   *
   * Args:
   *   elements: DOM element references
   */
  private setupEventListeners(elements: AppElements): void {
    elements.loginForm.addEventListener("submit", (e) => {
      void this.handleLogin(e);
    });

    elements.logoutBtn.addEventListener("click", () => {
      void this.handleLogout();
    });

    elements.recordBtn.addEventListener("click", () => {
      void this.handleRecordClick().catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Recording failed";
        elements.status.textContent =
          message === "Permission denied" ? "Microphone access denied" : message;
        elements.status.classList.add("error");
        emitEvent<TranslationErrorDetail>("app:translation-error", { message });
      });
    });

    elements.clearBtn.addEventListener("click", () => {
      void this.handleClear();
    });
  }

  // ============================================================================
  // Testing Utilities
  // ============================================================================

  /**
   * Get current application state for testing.
   *
   * Returns:
   *   Object containing current state values
   */
  getState(): {
    token: string | null;
    transcripts: readonly string[];
    isRecording: boolean;
  } {
    return {
      token: this.token,
      transcripts: this.transcripts,
      isRecording: this.recorderState.isRecording,
    };
  }

  /**
   * Wait for any pending chunk translations to complete.
   *
   * This is useful for tests to ensure all async operations finish before
   * verifying state or ending the test.
   *
   * Returns:
   *   Promise that resolves when all pending operations are complete
   */
  async waitForPendingOperations(): Promise<void> {
    if (this.pendingChunkTranslation !== null) {
      await this.pendingChunkTranslation;
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a new App instance.
 *
 * Returns:
 *   New App instance
 */
export function createApp(): App {
  return new App();
}

// ============================================================================
// Auto-initialization
// ============================================================================

/**
 * Initialize app on DOM ready.
 *
 * This function is called automatically when the module is imported in a browser
 * context. It checks document.readyState and either initializes immediately or
 * waits for DOMContentLoaded.
 */
export function autoInit(): void {
  const doc = getDocument();

  const doInit = (): void => {
    const app = createApp();
    void app.init().catch((err: unknown) => {
      const loginError = doc.getElementById("login-error");
      if (loginError !== null) {
        const message = err instanceof Error ? err.message : "Failed to load config";
        loginError.textContent = message;
        loginError.classList.remove("hidden");
      }
    });
  };

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", doInit);
  } else {
    doInit();
  }
}

// Re-export for backward compatibility
export { clearConfigCache };
