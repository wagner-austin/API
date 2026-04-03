/**
 * Internal hooks for dependency injection.
 *
 * This module provides typed hooks for external dependencies (fetch, MediaRecorder,
 * localStorage, etc.) that allow production code to use real implementations and
 * tests to substitute fakes without mocking.
 *
 * Usage:
 * - Production: Call setHooks() at app startup with real implementations
 * - Tests: Call setHooks() in beforeEach with fake implementations
 *
 * @internal This module is private to grandma-api-frontend
 */

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Typed fetch function signature.
 *
 * Args:
 *   input: Request URL or Request object
 *   init: Optional request configuration
 *
 * Returns:
 *   Promise resolving to Response
 */
export type FetchFn = (input: string | URL, init?: RequestInit) => Promise<Response>;

/**
 * Typed getUserMedia function signature.
 *
 * Args:
 *   constraints: Media stream constraints
 *
 * Returns:
 *   Promise resolving to MediaStream
 *
 * Raises:
 *   Error if permission denied or device unavailable
 */
export type GetUserMediaFn = (constraints: MediaStreamConstraints) => Promise<MediaStream>;

/**
 * MediaRecorder constructor signature.
 *
 * Args:
 *   stream: Media stream to record
 *   options: Recording options
 *
 * Returns:
 *   MediaRecorder instance
 */
export type MediaRecorderConstructor = new (
  stream: MediaStream,
  options?: MediaRecorderOptions
) => MediaRecorder;

/**
 * Storage interface matching localStorage API.
 */
export interface StorageInterface {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
}

/**
 * Document interface for getElementById.
 */
export interface DocumentInterface {
  getElementById(id: string): HTMLElement | null;
  readonly readyState: DocumentReadyState;
  addEventListener(type: string, listener: EventListener): void;
}

/**
 * EBML element as returned by ts-ebml decoder.
 */
export interface EBMLElement {
  readonly name: string;
  readonly type: string;
  readonly tagStart: number;
  readonly tagEnd: number;
  readonly dataStart: number;
  readonly dataEnd: number;
  readonly dataSize: number;
}

/**
 * Cue point information for seeking.
 */
export interface CueInfo {
  readonly CueTrack: number;
  readonly CueClusterPosition: number;
  readonly CueTime: number;
}

/**
 * Result from processing EBML elements.
 */
export interface WebmProcessResult {
  readonly metadatas: readonly EBMLElement[];
  readonly duration: number;
  readonly cues: readonly CueInfo[];
  readonly metadataSize: number;
}

/**
 * WebM muxer interface for ts-ebml operations.
 *
 * This provides typed access to ts-ebml functionality for:
 * - Decoding WebM data into EBML elements
 * - Processing elements to extract metadata, duration, cues
 * - Creating seekable metadata
 */
export interface WebmMuxerInterface {
  /**
   * Decode an ArrayBuffer into EBML elements.
   *
   * Args:
   *   buffer: Raw WebM data
   *
   * Returns:
   *   Array of EBML elements
   */
  decode(buffer: ArrayBuffer): EBMLElement[];

  /**
   * Process EBML elements to extract metadata, duration, and cues.
   *
   * Args:
   *   elements: EBML elements from decode()
   *
   * Returns:
   *   Processing result with metadata, duration, cues, metadataSize
   */
  processElements(elements: readonly EBMLElement[]): WebmProcessResult;

  /**
   * Create seekable metadata from processed elements.
   *
   * Args:
   *   metadatas: Original metadata elements
   *   duration: Total duration
   *   cues: Cue points for seeking
   *
   * Returns:
   *   ArrayBuffer with fixed metadata
   */
  makeMetadataSeekable(
    metadatas: readonly EBMLElement[],
    duration: number,
    cues: readonly CueInfo[]
  ): ArrayBuffer;
}

/**
 * Location interface for hostname detection.
 */
export interface LocationInterface {
  readonly hostname: string;
  readonly protocol: string;
  readonly port: string;
}

/**
 * All hooks bundled together.
 */
export interface Hooks {
  readonly fetch: FetchFn;
  readonly getUserMedia: GetUserMediaFn;
  readonly MediaRecorder: MediaRecorderConstructor;
  readonly storage: StorageInterface;
  readonly document: DocumentInterface;
  readonly webmMuxer: WebmMuxerInterface;
  readonly location: LocationInterface;
}

// ============================================================================
// Default Implementations (Browser Globals)
// ============================================================================

/**
 * ts-ebml module type definition.
 */
type TsEbmlModule = {
  Decoder: new () => { decode(buffer: ArrayBuffer): EBMLElement[] };
  Reader: new () => {
    read(element: EBMLElement): void;
    stop(): void;
    metadatas: EBMLElement[];
    duration: number;
    cues: CueInfo[];
    metadataSize: number;
  };
  tools: {
    makeMetadataSeekable(
      metadatas: readonly EBMLElement[],
      duration: number,
      cues: readonly CueInfo[]
    ): ArrayBuffer;
  };
};

/**
 * Cached ts-ebml module reference.
 */
let tsEbmlModule: TsEbmlModule | null = null;

/**
 * Get the ts-ebml module synchronously.
 *
 * Returns cached module or throws if not initialized.
 *
 * Returns:
 *   ts-ebml module
 *
 * Throws:
 *   Error if module not initialized via initTsEbml()
 */
function getTsEbml(): TsEbmlModule {
  if (tsEbmlModule === null) {
    throw new Error("EBML global not found. Ensure ts-ebml script is loaded.");
  }
  return tsEbmlModule;
}

/**
 * Load ts-ebml module.
 *
 * In browser: uses global EBML from CDN script tag
 * In Node.js/tests: uses dynamic import
 *
 * Returns:
 *   Promise resolving to ts-ebml module
 */
async function loadTsEbml(): Promise<TsEbmlModule> {
  // Check for browser global first (from CDN script tag)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const globalEBML = (globalThis as any).EBML as TsEbmlModule | undefined;
  if (globalEBML !== undefined) {
    return globalEBML;
  }

  // Fall back to dynamic import for Node.js/test environment
  const module = await import("ts-ebml");
  return module as unknown as TsEbmlModule;
}

/**
 * Create the real WebM muxer using ts-ebml.
 *
 * Note: ts-ebml must be loaded via script tag before use.
 * Call initTsEbml() at app startup to validate availability.
 *
 * Returns:
 *   WebmMuxerInterface wrapping ts-ebml
 */
function createRealWebmMuxer(): WebmMuxerInterface {
  return {
    decode(buffer: ArrayBuffer): EBMLElement[] {
      const ebml = getTsEbml();
      // Create fresh decoder each time - decoders are stateful and can't be reused
      // for different buffers without corrupting internal state
      const decoder = new ebml.Decoder();
      return decoder.decode(buffer);
    },

    processElements(elements: readonly EBMLElement[]): WebmProcessResult {
      const ebml = getTsEbml();
      const reader = new ebml.Reader();
      for (const element of elements) {
        reader.read(element);
      }
      reader.stop();

      return {
        metadatas: reader.metadatas,
        duration: reader.duration,
        cues: reader.cues,
        metadataSize: reader.metadataSize,
      };
    },

    makeMetadataSeekable(
      metadatas: readonly EBMLElement[],
      duration: number,
      cues: readonly CueInfo[]
    ): ArrayBuffer {
      const ebml = getTsEbml();
      return ebml.tools.makeMetadataSeekable(metadatas, duration, cues);
    },
  };
}

/**
 * Initialize the ts-ebml module.
 *
 * Loads ts-ebml from global (browser) or dynamic import (Node.js/tests).
 * Must be called before using the webmMuxer hook.
 *
 * Returns:
 *   Promise that resolves when ts-ebml is loaded
 */
export async function initTsEbml(): Promise<void> {
  tsEbmlModule = await loadTsEbml();
}

/**
 * Create default hooks using browser globals.
 *
 * Returns:
 *   Hooks object with real browser implementations
 */
function createDefaultHooks(): Hooks {
  return {
    fetch: globalThis.fetch.bind(globalThis),
    getUserMedia: (constraints: MediaStreamConstraints) =>
      navigator.mediaDevices.getUserMedia(constraints),
    MediaRecorder: globalThis.MediaRecorder,
    storage: globalThis.localStorage,
    document: globalThis.document,
    webmMuxer: createRealWebmMuxer(),
    location: globalThis.location,
  };
}

// ============================================================================
// Hook State
// ============================================================================

let currentHooks: Hooks | null = null;

/**
 * Set all hooks at once.
 *
 * Call this at application startup with real implementations,
 * or in test setup with fake implementations.
 *
 * Args:
 *   hooks: Complete hooks object
 */
export function setHooks(hooks: Hooks): void {
  currentHooks = hooks;
}

/**
 * Get current hooks, initializing to defaults if not set.
 *
 * Returns:
 *   Current hooks object
 */
export function getHooks(): Hooks {
  if (currentHooks === null) {
    currentHooks = createDefaultHooks();
  }
  return currentHooks;
}

/**
 * Reset hooks to null, forcing re-initialization on next getHooks() call.
 *
 * This is useful for tests that need to verify default initialization behavior.
 */
export function resetHooks(): void {
  currentHooks = null;
}

// ============================================================================
// Individual Hook Accessors
// ============================================================================

/**
 * Get the fetch hook.
 *
 * Returns:
 *   Fetch function
 */
export function getFetch(): FetchFn {
  return getHooks().fetch;
}

/**
 * Get the getUserMedia hook.
 *
 * Returns:
 *   getUserMedia function
 */
export function getGetUserMedia(): GetUserMediaFn {
  return getHooks().getUserMedia;
}

/**
 * Get the MediaRecorder constructor hook.
 *
 * Returns:
 *   MediaRecorder constructor
 */
export function getMediaRecorderConstructor(): MediaRecorderConstructor {
  return getHooks().MediaRecorder;
}

/**
 * Get the storage hook.
 *
 * Returns:
 *   Storage interface
 */
export function getStorage(): StorageInterface {
  return getHooks().storage;
}

/**
 * Get the document hook.
 *
 * Returns:
 *   Document interface
 */
export function getDocument(): DocumentInterface {
  return getHooks().document;
}

/**
 * Get the WebM muxer hook.
 *
 * Returns:
 *   WebM muxer interface
 */
export function getWebmMuxer(): WebmMuxerInterface {
  return getHooks().webmMuxer;
}

/**
 * Get the location hook.
 *
 * Returns:
 *   Location interface
 */
export function getLocation(): LocationInterface {
  return getHooks().location;
}

/**
 * Reset ts-ebml module state.
 *
 * For testing only - resets the ts-ebml module to null so error branches
 * can be tested.
 */
export function resetTsEbml(): void {
  tsEbmlModule = null;
}
