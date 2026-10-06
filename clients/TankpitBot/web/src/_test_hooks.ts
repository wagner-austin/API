/**
 * The client's seams onto the browser.
 *
 * Two things the renderer needs cannot run in a test's DOM: a 2D canvas
 * context (jsdom has none) and an image that actually decodes. The wire
 * client needs three more that reach outside the page: a WebSocket to the
 * server, a fetch of what the server hands out, and a random session
 * magic. Each is a hook bound to its real implementation here and called
 * unconditionally; a test binds a fake through setHooks and restores the
 * real ones with resetHooks.
 *
 * @internal Private to the renderer and its tests.
 */

import type { Surface } from "./scaled_context.js";

/** Read a canvas's 2D context, or null where the platform has none. */
export interface CanvasContextHook {
  (canvas: HTMLCanvasElement): Surface | null;
}

/** Load an image from a URL, resolving once it has decoded. */
export interface LoadImageHook {
  (url: string): Promise<CanvasImageSource>;
}

/** Anything that encodes itself as a data URL, as a canvas does. */
export interface DataUrlSource {
  toDataURL(type: string): string;
}

/** Encode a painted canvas as a URL an image can load. */
export interface CanvasUrlHook {
  (canvas: DataUrlSource): string;
}

/** What a socket tells the client: it opened, a message came, it closed. */
export interface SocketHandlers {
  open(): void;
  message(bytes: Uint8Array): void;
  close(code: number, reason: string): void;
}

/** Where the client writes its frames: an open socket. */
export interface FrameSink {
  send(data: Uint8Array): void;
}

/** Open a binary WebSocket to a URL, reporting to the handlers. */
export interface OpenSocketHook {
  (url: string, handlers: SocketHandlers): FrameSink;
}

/** Fetch a URL's body as bytes. */
export interface FetchBytesHook {
  (url: string): Promise<Uint8Array>;
}

/** Choose a fresh session magic. */
export interface RandomMagicHook {
  (): string;
}

/** Every hook. */
export interface Hooks {
  readonly canvasContext: CanvasContextHook;
  readonly loadImage: LoadImageHook;
  readonly canvasUrl: CanvasUrlHook;
  readonly openSocket: OpenSocketHook;
  readonly fetchBytes: FetchBytesHook;
  readonly randomMagic: RandomMagicHook;
}

function realCanvasContext(canvas: HTMLCanvasElement): Surface | null {
  return canvas.getContext("2d");
}

function realCanvasUrl(canvas: DataUrlSource): string {
  return canvas.toDataURL("image/png");
}

/**
 * Point an image at a URL and settle once it has decoded or failed.
 *
 * @param image - A fresh image element.
 * @param url - The sheet's URL.
 * @returns The image, once loaded.
 * @throws Error RENDER_IMAGE when the image fails to load.
 */
export function loadImageInto(image: HTMLImageElement, url: string): Promise<CanvasImageSource> {
  const loaded = new Promise<CanvasImageSource>((resolve, reject) => {
    image.onload = (): void => resolve(image);
    image.onerror = (): void => reject(new Error(`RENDER_IMAGE: ${url} did not load`));
  });
  image.src = url;
  return loaded;
}

function realLoadImage(url: string): Promise<CanvasImageSource> {
  return loadImageInto(new Image(), url);
}

/**
 * A socket message's bytes; the wire is binary.
 *
 * @param data - A message event's data, an ArrayBuffer for a binary socket.
 * @returns The bytes.
 * @throws Error SOCKET_TEXT for anything but an ArrayBuffer.
 */
export function messageBytes(data: unknown): Uint8Array {
  if (!(data instanceof ArrayBuffer)) {
    throw new Error("SOCKET_TEXT: the server sent a message that is not binary");
  }
  return new Uint8Array(data);
}

/**
 * Bind a WebSocket's events to the client's handlers.
 *
 * @param socket - A fresh WebSocket.
 * @param handlers - What the client does with its events.
 * @returns The socket, as the sink the client writes to.
 */
export function bindSocket(socket: WebSocket, handlers: SocketHandlers): FrameSink {
  socket.binaryType = "arraybuffer";
  socket.onopen = (): void => handlers.open();
  socket.onmessage = (event: MessageEvent<unknown>): void => handlers.message(messageBytes(event.data));
  socket.onclose = (event: CloseEvent): void => handlers.close(event.code, event.reason);
  return socket;
}

function realOpenSocket(url: string, handlers: SocketHandlers): FrameSink {
  return bindSocket(new WebSocket(url), handlers);
}

/**
 * A response's body, refusing one that did not answer 2xx.
 *
 * @param response - The response.
 * @param url - What was fetched, for the error.
 * @returns The body's bytes.
 * @throws Error FETCH_STATUS for a status outside 2xx.
 */
export async function responseBytes(response: Response, url: string): Promise<Uint8Array> {
  if (!response.ok) {
    throw new Error(`FETCH_STATUS: ${url} answered ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function realFetchBytes(url: string): Promise<Uint8Array> {
  return responseBytes(await fetch(url), url);
}

const MAGIC_LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789";
const MAGIC_LENGTH = 20;

function realRandomMagic(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(MAGIC_LENGTH));
  return Array.from(bytes, (value) => MAGIC_LETTERS.charAt(value % MAGIC_LETTERS.length)).join("");
}

/** The real implementations. */
export const realHooks: Hooks = {
  canvasContext: realCanvasContext,
  loadImage: realLoadImage,
  canvasUrl: realCanvasUrl,
  openSocket: realOpenSocket,
  fetchBytes: realFetchBytes,
  randomMagic: realRandomMagic,
};

let current: Hooks = realHooks;

/** The hooks in force. */
export function hooks(): Hooks {
  return current;
}

/** Bind hooks, for a test. */
export function setHooks(next: Hooks): void {
  current = next;
}

/** Restore the real hooks. */
export function resetHooks(): void {
  current = realHooks;
}
