/**
 * Public test utilities for grandma-api-frontend.
 *
 * This module provides fake implementations and helper functions for testing.
 * It exports typed fakes that can be used via the _test_hooks system.
 *
 * Usage:
 *   import { createFakeHooks, createFakeResponse } from "./testing.js";
 *   import { setHooks } from "./_test_hooks.js";
 *
 *   beforeEach(() => {
 *     const fakes = createFakeHooks();
 *     setHooks(fakes.hooks);
 *   });
 */

import {
  Hooks,
  FetchFn,
  GetUserMediaFn,
  MediaRecorderConstructor,
  StorageInterface,
  DocumentInterface,
  WebmMuxerInterface,
  EBMLElement,
  CueInfo,
  WebmProcessResult,
  LocationInterface,
} from "./_test_hooks.js";

// ============================================================================
// Response Builders
// ============================================================================

/**
 * Create a fake successful JSON response.
 *
 * Args:
 *   body: JSON-serializable body
 *   status: HTTP status code (default 200)
 *
 * Returns:
 *   Response object
 */
export function createFakeResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Create a fake error response.
 *
 * Args:
 *   status: HTTP status code
 *   detail: Optional error detail message
 *
 * Returns:
 *   Response object
 */
export function createFakeErrorResponse(status: number, detail?: string): Response {
  const body = detail !== undefined ? { detail } : null;
  return new Response(body !== null ? JSON.stringify(body) : null, { status });
}

// ============================================================================
// Fake Storage
// ============================================================================

/**
 * Create a fake in-memory storage implementation.
 *
 * Returns:
 *   Object containing storage interface and inspection methods
 */
export function createFakeStorage(): {
  storage: StorageInterface;
  getData: () => Map<string, string>;
} {
  const data = new Map<string, string>();

  const storage: StorageInterface = {
    getItem(key: string): string | null {
      const value = data.get(key);
      return value !== undefined ? value : null;
    },
    setItem(key: string, value: string): void {
      data.set(key, value);
    },
    removeItem(key: string): void {
      data.delete(key);
    },
    clear(): void {
      data.clear();
    },
  };

  return { storage, getData: () => data };
}

// ============================================================================
// Fake MediaRecorder
// ============================================================================

/**
 * Recorded call to a fake MediaRecorder.
 */
export interface MediaRecorderCall {
  readonly method: "start" | "stop";
  readonly timestamp: number;
}

/**
 * Fake MediaRecorder that allows controlled testing.
 */
export class FakeMediaRecorder implements MediaRecorder {
  // Required MediaRecorder properties
  readonly stream: MediaStream;
  readonly mimeType: string;
  readonly videoBitsPerSecond: number = 0;
  readonly audioBitsPerSecond: number = 0;

  // State
  private _state: RecordingState = "inactive";

  // Event handlers
  ondataavailable: ((event: BlobEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onpause: ((event: Event) => void) | null = null;
  onresume: ((event: Event) => void) | null = null;
  onstart: ((event: Event) => void) | null = null;
  onstop: ((event: Event) => void) | null = null;

  // Test inspection
  readonly calls: MediaRecorderCall[] = [];
  private readonly audioData: Blob;

  constructor(stream: MediaStream, options?: MediaRecorderOptions) {
    this.stream = stream;
    this.mimeType = options?.mimeType ?? "audio/webm";
    this.audioData = new Blob(["fake-audio-data"], { type: this.mimeType });
  }

  get state(): RecordingState {
    return this._state;
  }

  start(_timeslice?: number): void {
    this._state = "recording";
    this.calls.push({ method: "start", timestamp: Date.now() });
    if (this.onstart !== null) {
      this.onstart(new Event("start"));
    }
  }

  stop(): void {
    this._state = "inactive";
    this.calls.push({ method: "stop", timestamp: Date.now() });

    // Emit data before stop event (matches real MediaRecorder behavior)
    if (this.ondataavailable !== null) {
      const blobEvent = new BlobEvent("dataavailable", { data: this.audioData });
      this.ondataavailable(blobEvent);
    }

    if (this.onstop !== null) {
      this.onstop(new Event("stop"));
    }
  }

  pause(): void {
    this._state = "paused";
    if (this.onpause !== null) {
      this.onpause(new Event("pause"));
    }
  }

  resume(): void {
    this._state = "recording";
    if (this.onresume !== null) {
      this.onresume(new Event("resume"));
    }
  }

  requestData(): void {
    if (this.ondataavailable !== null) {
      const blobEvent = new BlobEvent("dataavailable", { data: this.audioData });
      this.ondataavailable(blobEvent);
    }
  }

  /**
   * Request an empty data blob for testing edge cases.
   * This simulates the browser sending an empty chunk.
   */
  requestEmptyData(): void {
    if (this.ondataavailable !== null) {
      const emptyBlob = new Blob([], { type: this.mimeType });
      const blobEvent = new BlobEvent("dataavailable", { data: emptyBlob });
      this.ondataavailable(blobEvent);
    }
  }

  // EventTarget methods (required but not used in tests)
  addEventListener(_type: string, _listener: EventListener): void {
    // No-op for testing
  }

  removeEventListener(_type: string, _listener: EventListener): void {
    // No-op for testing
  }

  dispatchEvent(_event: Event): boolean {
    return true;
  }

  // Static method required by MediaRecorder interface
  static isTypeSupported(_mimeType: string): boolean {
    return true;
  }
}

// ============================================================================
// Fake MediaStream
// ============================================================================

/**
 * Fake MediaStreamTrack for testing.
 */
export class FakeMediaStreamTrack implements MediaStreamTrack {
  readonly kind: string = "audio";
  readonly id: string = "fake-track-id";
  readonly label: string = "Fake Audio Track";
  enabled: boolean = true;
  muted: boolean = false;
  readonly readyState: MediaStreamTrackState = "live";
  contentHint: string = "";

  // Track if stop was called
  private _stopped: boolean = false;

  onended: ((event: Event) => void) | null = null;
  onmute: ((event: Event) => void) | null = null;
  onunmute: ((event: Event) => void) | null = null;

  get stopped(): boolean {
    return this._stopped;
  }

  stop(): void {
    this._stopped = true;
  }

  clone(): MediaStreamTrack {
    return new FakeMediaStreamTrack();
  }

  getCapabilities(): MediaTrackCapabilities {
    return {};
  }

  getConstraints(): MediaTrackConstraints {
    return {};
  }

  getSettings(): MediaTrackSettings {
    return {};
  }

  applyConstraints(_constraints?: MediaTrackConstraints): Promise<void> {
    return Promise.resolve();
  }

  addEventListener(_type: string, _listener: EventListener): void {
    // No-op
  }

  removeEventListener(_type: string, _listener: EventListener): void {
    // No-op
  }

  dispatchEvent(_event: Event): boolean {
    return true;
  }
}

/**
 * Fake MediaStream for testing.
 */
export class FakeMediaStream implements MediaStream {
  readonly id: string = "fake-stream-id";
  readonly active: boolean = true;
  private readonly tracks: FakeMediaStreamTrack[];

  onaddtrack: ((event: MediaStreamTrackEvent) => void) | null = null;
  onremovetrack: ((event: MediaStreamTrackEvent) => void) | null = null;

  constructor(tracks: FakeMediaStreamTrack[] = [new FakeMediaStreamTrack()]) {
    this.tracks = tracks;
  }

  getTracks(): MediaStreamTrack[] {
    return this.tracks;
  }

  getAudioTracks(): MediaStreamTrack[] {
    return this.tracks.filter((t) => t.kind === "audio");
  }

  getVideoTracks(): MediaStreamTrack[] {
    return this.tracks.filter((t) => t.kind === "video");
  }

  getTrackById(_trackId: string): MediaStreamTrack | null {
    return this.tracks[0] ?? null;
  }

  addTrack(_track: MediaStreamTrack): void {
    // No-op
  }

  removeTrack(_track: MediaStreamTrack): void {
    // No-op
  }

  clone(): MediaStream {
    return new FakeMediaStream();
  }

  addEventListener(_type: string, _listener: EventListener): void {
    // No-op
  }

  removeEventListener(_type: string, _listener: EventListener): void {
    // No-op
  }

  dispatchEvent(_event: Event): boolean {
    return true;
  }
}

// ============================================================================
// Fake Document
// ============================================================================

/**
 * Create a fake document interface.
 *
 * Args:
 *   elements: Map of element IDs to elements
 *   readyState: Document ready state
 *
 * Returns:
 *   Document interface
 */
export function createFakeDocument(
  elements: Map<string, HTMLElement>,
  readyState: DocumentReadyState = "complete"
): DocumentInterface {
  const listeners: EventListener[] = [];

  return {
    getElementById(id: string): HTMLElement | null {
      const element = elements.get(id);
      return element !== undefined ? element : null;
    },
    readyState,
    addEventListener(_type: string, listener: EventListener): void {
      listeners.push(listener);
    },
  };
}

// ============================================================================
// Fake WebM Muxer
// ============================================================================

/**
 * Configuration for fake WebM muxer behavior.
 */
export interface FakeWebmMuxerConfig {
  /** Elements to return from decode() */
  readonly elements?: readonly EBMLElement[];
  /** Process result to return */
  readonly processResult?: WebmProcessResult;
  /** Refined metadata to return */
  readonly refinedMetadata?: ArrayBuffer;
  /** Error to throw from decode() */
  readonly decodeError?: Error;
}

/**
 * Create a fake WebM muxer for testing.
 *
 * Args:
 *   config: Optional configuration for fake behavior
 *
 * Returns:
 *   Object with muxer interface and inspection methods
 */
export function createFakeWebmMuxer(config: FakeWebmMuxerConfig = {}): {
  muxer: WebmMuxerInterface;
  getDecodeInputs: () => ArrayBuffer[];
  getProcessInputs: () => Array<readonly EBMLElement[]>;
  getMakeSeekableInputs: () => Array<{
    metadatas: readonly EBMLElement[];
    duration: number;
    cues: readonly CueInfo[];
  }>;
} {
  const decodeInputs: ArrayBuffer[] = [];
  const processInputs: Array<readonly EBMLElement[]> = [];
  const makeSeekableInputs: Array<{
    metadatas: readonly EBMLElement[];
    duration: number;
    cues: readonly CueInfo[];
  }> = [];

  // Default fake elements
  const defaultElements: EBMLElement[] = [
    {
      name: "EBML",
      type: "m",
      tagStart: 0,
      tagEnd: 4,
      dataStart: 4,
      dataEnd: 100,
      dataSize: 96,
    },
  ];

  // Default process result
  const defaultProcessResult: WebmProcessResult = {
    metadatas: defaultElements,
    duration: 1000,
    cues: [],
    metadataSize: 100,
  };

  // Default refined metadata (simple ArrayBuffer)
  const defaultRefinedMetadata = new ArrayBuffer(50);

  const muxer: WebmMuxerInterface = {
    decode(buffer: ArrayBuffer): EBMLElement[] {
      decodeInputs.push(buffer);
      if (config.decodeError !== undefined) {
        throw config.decodeError;
      }
      return [...(config.elements ?? defaultElements)];
    },

    processElements(elements: readonly EBMLElement[]): WebmProcessResult {
      processInputs.push(elements);
      return config.processResult ?? defaultProcessResult;
    },

    makeMetadataSeekable(
      metadatas: readonly EBMLElement[],
      duration: number,
      cues: readonly CueInfo[]
    ): ArrayBuffer {
      makeSeekableInputs.push({ metadatas, duration, cues });
      return config.refinedMetadata ?? defaultRefinedMetadata;
    },
  };

  return {
    muxer,
    getDecodeInputs: () => decodeInputs,
    getProcessInputs: () => processInputs,
    getMakeSeekableInputs: () => makeSeekableInputs,
  };
}

// ============================================================================
// Fake Hooks Factory
// ============================================================================

/**
 * Configuration for creating fake hooks.
 */
export interface FakeHooksConfig {
  readonly fetchResponses?: Response[];
  readonly fetchThrow?: unknown;
  readonly getUserMediaResult?: MediaStream | Error;
  readonly getUserMediaThrow?: unknown;
  readonly initialStorage?: ReadonlyMap<string, string>;
  readonly elements?: Map<string, HTMLElement>;
  readonly documentReadyState?: DocumentReadyState;
  readonly webmMuxerConfig?: FakeWebmMuxerConfig;
  readonly locationHostname?: string;
}

/**
 * Result of creating fake hooks with inspection methods.
 */
export interface FakeHooksResult {
  readonly hooks: Hooks;
  readonly getFetchCalls: () => Array<{ input: string | URL; init?: RequestInit }>;
  readonly getStorageData: () => Map<string, string>;
  readonly getMediaRecorderInstances: () => FakeMediaRecorder[];
  readonly getMediaStreamTracks: () => FakeMediaStreamTrack[];
  readonly getWebmMuxerDecodeInputs: () => ArrayBuffer[];
  readonly getWebmMuxerProcessInputs: () => Array<readonly EBMLElement[]>;
  readonly getWebmMuxerMakeSeekableInputs: () => Array<{
    metadatas: readonly EBMLElement[];
    duration: number;
    cues: readonly CueInfo[];
  }>;
}

/**
 * Create a complete set of fake hooks for testing.
 *
 * Args:
 *   config: Optional configuration for fakes
 *
 * Returns:
 *   Fake hooks with inspection methods
 */
export function createFakeHooks(config: FakeHooksConfig = {}): FakeHooksResult {
  // Fetch tracking
  const fetchCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const fetchResponses = config.fetchResponses ?? [];
  let fetchCallIndex = 0;

  const fakeFetch: FetchFn = async (input, init) => {
    // Support throwing arbitrary non-Error values for testing error handling
    if (config.fetchThrow !== undefined) {
      throw config.fetchThrow;
    }
    if (init !== undefined) {
      fetchCalls.push({ input, init });
    } else {
      fetchCalls.push({ input });
    }
    const response = fetchResponses[fetchCallIndex];
    fetchCallIndex++;
    if (response === undefined) {
      throw new Error(`No fake response configured for fetch call ${fetchCallIndex}`);
    }
    return response;
  };

  // Storage
  const { storage, getData: getStorageData } = createFakeStorage();
  if (config.initialStorage !== undefined) {
    for (const [key, value] of config.initialStorage) {
      storage.setItem(key, value);
    }
  }

  // MediaRecorder tracking
  const mediaRecorderInstances: FakeMediaRecorder[] = [];
  const mediaStreamTracks: FakeMediaStreamTrack[] = [];

  const FakeMediaRecorderClass = class extends FakeMediaRecorder {
    constructor(stream: MediaStream, options?: MediaRecorderOptions) {
      super(stream, options);
      mediaRecorderInstances.push(this);
    }
  } as unknown as MediaRecorderConstructor;

  // getUserMedia
  const fakeGetUserMedia: GetUserMediaFn = async (_constraints) => {
    // Support throwing arbitrary non-Error values for testing error handling
    if (config.getUserMediaThrow !== undefined) {
      throw config.getUserMediaThrow;
    }
    const result = config.getUserMediaResult;
    if (result instanceof Error) {
      throw result;
    }
    if (result !== undefined) {
      return result;
    }
    const track = new FakeMediaStreamTrack();
    mediaStreamTracks.push(track);
    return new FakeMediaStream([track]);
  };

  // Document
  const elements = config.elements ?? new Map<string, HTMLElement>();
  const fakeDocument = createFakeDocument(elements, config.documentReadyState ?? "complete");

  // WebM Muxer
  const {
    muxer: fakeWebmMuxer,
    getDecodeInputs: getWebmMuxerDecodeInputs,
    getProcessInputs: getWebmMuxerProcessInputs,
    getMakeSeekableInputs: getWebmMuxerMakeSeekableInputs,
  } = createFakeWebmMuxer(config.webmMuxerConfig);

  // Location
  const fakeLocation: LocationInterface = {
    hostname: config.locationHostname ?? "localhost",
    protocol: "https:",
    port: "8091",
  };

  const hooks: Hooks = {
    fetch: fakeFetch,
    getUserMedia: fakeGetUserMedia,
    MediaRecorder: FakeMediaRecorderClass,
    storage,
    document: fakeDocument,
    webmMuxer: fakeWebmMuxer,
    location: fakeLocation,
  };

  return {
    hooks,
    getFetchCalls: () => fetchCalls,
    getStorageData,
    getMediaRecorderInstances: () => mediaRecorderInstances,
    getMediaStreamTracks: () => mediaStreamTracks,
    getWebmMuxerDecodeInputs,
    getWebmMuxerProcessInputs,
    getWebmMuxerMakeSeekableInputs,
  };
}

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Wait for a condition to become true.
 *
 * Args:
 *   condition: Function that returns true when condition is met
 *   timeoutMs: Maximum time to wait in milliseconds
 *   intervalMs: Polling interval in milliseconds
 *
 * Returns:
 *   Promise that resolves when condition is true
 *
 * Raises:
 *   Error if timeout is reached before condition becomes true
 */
export async function waitFor(
  condition: () => boolean,
  timeoutMs: number = 1000,
  intervalMs: number = 10
): Promise<void> {
  const startTime = Date.now();

  while (!condition()) {
    if (Date.now() - startTime > timeoutMs) {
      throw new Error(`waitFor timed out after ${timeoutMs}ms`);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/**
 * Wait for an async condition to become true.
 *
 * Args:
 *   condition: Async function that returns true when condition is met
 *   timeoutMs: Maximum time to wait in milliseconds
 *   intervalMs: Polling interval in milliseconds
 *
 * Returns:
 *   Promise that resolves when condition is true
 *
 * Raises:
 *   Error if timeout is reached before condition becomes true
 */
export async function waitForAsync(
  condition: () => Promise<boolean>,
  timeoutMs: number = 1000,
  intervalMs: number = 10
): Promise<void> {
  const startTime = Date.now();

  while (!(await condition())) {
    if (Date.now() - startTime > timeoutMs) {
      throw new Error(`waitForAsync timed out after ${timeoutMs}ms`);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/**
 * Create HTML elements for testing the app.
 *
 * Returns:
 *   Map of element IDs to elements with proper event handling
 */
export function createTestElements(): Map<string, HTMLElement> {
  const elements = new Map<string, HTMLElement>();

  // Login section
  const loginSection = document.createElement("section");
  loginSection.id = "login-section";
  loginSection.className = "card";
  elements.set("login-section", loginSection);

  // Login form
  const loginForm = document.createElement("form");
  loginForm.id = "login-form";
  elements.set("login-form", loginForm);

  // Token input
  const tokenInput = document.createElement("input");
  tokenInput.id = "token-input";
  tokenInput.type = "password";
  elements.set("token-input", tokenInput);

  // Login error
  const loginError = document.createElement("div");
  loginError.id = "login-error";
  loginError.className = "hidden";
  elements.set("login-error", loginError);

  // Main section
  const mainSection = document.createElement("section");
  mainSection.id = "main-section";
  mainSection.className = "hidden";
  elements.set("main-section", mainSection);

  // Logout button
  const logoutBtn = document.createElement("button");
  logoutBtn.id = "logout-btn";
  elements.set("logout-btn", logoutBtn);

  // Record button
  const recordBtn = document.createElement("button");
  recordBtn.id = "record-btn";
  elements.set("record-btn", recordBtn);

  // Record icon
  const recordIcon = document.createElement("span");
  recordIcon.id = "record-icon";
  recordIcon.textContent = "🎤";
  elements.set("record-icon", recordIcon);

  // Status
  const status = document.createElement("div");
  status.id = "status";
  status.textContent = "Tap to record";
  elements.set("status", status);

  // Transcript
  const transcript = document.createElement("div");
  transcript.id = "transcript";
  transcript.textContent = "Translations will appear here...";
  elements.set("transcript", transcript);

  // Clear button
  const clearBtn = document.createElement("button");
  clearBtn.id = "clear-btn";
  elements.set("clear-btn", clearBtn);

  // Recording feedback container
  const recordingFeedback = document.createElement("div");
  recordingFeedback.id = "recording-feedback";
  recordingFeedback.className = "recording-feedback hidden";
  elements.set("recording-feedback", recordingFeedback);

  // Timer
  const timer = document.createElement("div");
  timer.id = "timer";
  timer.className = "timer";
  timer.textContent = "0:00";
  elements.set("timer", timer);

  // Audio level
  const audioLevel = document.createElement("div");
  audioLevel.id = "audio-level";
  audioLevel.className = "audio-level";
  elements.set("audio-level", audioLevel);

  return elements;
}
